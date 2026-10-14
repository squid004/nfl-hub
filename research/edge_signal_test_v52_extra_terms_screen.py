"""v52: screen four candidate extra model terms -- explosive plays, penalty yards, time of
possession / tempo, special teams -- BEFORE any walk-forward fitting.

Stage 1 (persistence): does the stat repeat game-to-game / season-to-season? Benchmarked against
         the 8 stats production already uses, so "weak" has a reference point.
Stage 2 (redundancy + market): how much of each candidate's pre-game home-minus-away EWMA diff is
         already explained by the 8 production stats, and does the unexplained part correlate with
         the closing spread / the market residual (margin - spread_line)? This is an in-sample
         univariate screen, NOT a fit or an edge claim -- walk-forward fitting is the next stage and
         only for candidates that survive here.

Extraction mirrors production: REG season, 2007-2025, garbage-time (wp outside [0.05, 0.95])
excluded for the play-based stats; penalties and whole-game TOP share are whole-game by nature.
Per-season extracts are cached in research/cache/ (gitignored).

Run: python research/edge_signal_test_v52_extra_terms_screen.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402
from persistence import persistence_report, pregame_ewma  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE = os.path.join(HERE, "cache")
SEASONS = list(range(2007, 2026))
WP_LO, WP_HI = 0.05, 0.95

WANT = {
    "game_id", "season_type", "home_team", "away_team", "posteam", "defteam", "play_type", "epa", "wp",
    "yards_gained", "interception", "fumble_lost", "penalty", "penalty_team", "penalty_yards",
    "penalty_type", "play_deleted", "fixed_drive", "fixed_drive_result", "drive_start_transition",
    "drive_time_of_possession", "drive_play_count", "yardline_100", "field_goal_result",
    "kick_distance", "punt_blocked",
}
PRESNAP = {"False Start", "Delay of Game", "Defensive Offside", "Neutral Zone Infraction", "Encroachment",
           "Illegal Formation", "Illegal Shift", "Illegal Motion", "Illegal Substitution",
           "Defensive 12 On-field", "Offensive 12 On-field", "Offside on Free Kick"}


def _secs(s: pd.Series) -> pd.Series:
    parts = s.astype("string").str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def extract_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (team-game feature frame, per-field-goal-attempt frame) for one season."""
    tg_path = os.path.join(CACHE, f"v52_features_{season}.parquet")
    fg_path = os.path.join(CACHE, f"v52_fg_{season}.parquet")
    if os.path.exists(tg_path) and os.path.exists(fg_path):
        return pd.read_parquet(tg_path), pd.read_parquet(fg_path)

    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", low_memory=False,
                     usecols=lambda c: c in WANT)
    for c in WANT - set(df.columns):
        df[c] = np.nan
    df = df[(df["season_type"] == "REG") & (df["play_deleted"].fillna(0) != 1)].copy()
    in_band = lambda f: f[(f["wp"] >= WP_LO) & (f["wp"] <= WP_HI)]  # noqa: E731
    K = ["game_id", "team"]

    g = df.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first")).reset_index()
    base = pd.concat([
        g.rename(columns={"home": "team", "away": "opp"}).assign(is_home=1),
        g.rename(columns={"away": "team", "home": "opp"}).assign(is_home=0),
    ], ignore_index=True)[["game_id", "team", "opp", "is_home"]].set_index(K)
    feats: dict[str, pd.Series] = {}

    def put(name: str, s: pd.Series, fill: float | None = None) -> None:
        s = s.copy()
        s.index = s.index.set_names(K)
        feats[name] = s if fill is None else s.reindex(base.index).fillna(fill)

    # --- benchmark 8-stat components + explosive plays (run/pass plays, garbage time excluded)
    rp = in_band(df[df["play_type"].isin(["run", "pass"])].dropna(subset=["wp", "epa", "posteam"])).copy()
    rp["turnover"] = rp["interception"].fillna(0) + rp["fumble_lost"].fillna(0)
    run, pas = rp[rp["play_type"] == "run"], rp[rp["play_type"] == "pass"]
    put("rush_off_epa", run.groupby(["game_id", "posteam"])["epa"].mean())
    put("pass_off_epa", pas.groupby(["game_id", "posteam"])["epa"].mean())
    put("turnovers_off", rp.groupby(["game_id", "posteam"])["turnover"].sum(), fill=0.0)
    put("expl_rush_off", (run["yards_gained"] >= 10).groupby([run["game_id"], run["posteam"]]).mean())
    put("expl_pass_off", (pas["yards_gained"] >= 20).groupby([pas["game_id"], pas["posteam"]]).mean())

    # --- penalties: accepted penalties, whole game, any play type
    pen = df[df["penalty"] == 1].dropna(subset=["penalty_team"])
    put("pen_yds", pen.groupby(["game_id", "penalty_team"])["penalty_yards"].sum(), fill=0.0)
    ps = pen[pen["penalty_type"].isin(PRESNAP)]
    put("pen_presnap_yds", ps.groupby(["game_id", "penalty_team"])["penalty_yards"].sum(), fill=0.0)

    # --- drive-level: time of possession / tempo / drive success / starting field position
    d = df.dropna(subset=["fixed_drive", "posteam"])
    drv = d.groupby(["game_id", "fixed_drive"]).agg(
        team=("posteam", "first"), secs=("drive_time_of_possession", "first"), plays=("drive_play_count", "first"),
        res=("fixed_drive_result", "first"), trans=("drive_start_transition", "first"), wp0=("wp", "first"))
    drv["secs"] = _secs(drv["secs"])
    first_scrim = d[d["play_type"].isin(["run", "pass"])].groupby(["game_id", "fixed_drive"])["yardline_100"].first()
    drv["start_yd"] = 100 - first_scrim.reindex(drv.index)
    drv = drv.reset_index()
    put("top_secs", drv.groupby(["game_id", "team"])["secs"].sum())
    db = drv[(drv["wp0"] >= WP_LO) & (drv["wp0"] <= WP_HI) & (drv["plays"] > 0)]
    put("sec_per_play", db.groupby(["game_id", "team"])["secs"].sum() / db.groupby(["game_id", "team"])["plays"].sum())
    scored = db[db["res"] != "End of half"]
    put("drive_score_rate", scored["res"].isin(["Touchdown", "Field goal"]).groupby([scored["game_id"], scored["team"]]).mean())
    fp = db[db["trans"].isin(["KICKOFF", "PUNT"])].dropna(subset=["start_yd"])
    put("fp_own_start", fp.groupby(["game_id", "team"])["start_yd"].mean())

    # --- special teams EPA (zero-sum between the two teams by construction)
    st = in_band(df[df["play_type"].isin(["kickoff", "punt", "field_goal", "extra_point"])]
                 .dropna(subset=["wp", "epa", "posteam", "defteam"]))
    kicking = st.groupby(["game_id", "posteam"])["epa"].sum()
    kp = st[st["play_type"].isin(["kickoff", "punt"])]
    returning = -kp.groupby(["game_id", "defteam"])["epa"].sum()
    fgx = st[st["play_type"].isin(["field_goal", "extra_point"])]
    fgdef = -fgx.groupby(["game_id", "defteam"])["epa"].sum()
    put("st_kicking_epa", kicking, fill=0.0)
    put("st_returning_epa", returning, fill=0.0)
    kicking_a, returning_a, fgdef_a = (s.rename_axis(K).reindex(base.index).fillna(0.0) for s in (kicking, returning, fgdef))
    feats["st_epa"] = kicking_a + returning_a + fgdef_a
    # kickoff/punt only (both sides): strips the FG/XP credit a team earns just for reaching range,
    # which is offense-driven, not special-teams skill
    kp_kick = kp.groupby(["game_id", "posteam"])["epa"].sum().rename_axis(K).reindex(base.index).fillna(0.0)
    feats["st_kp_epa"] = kp_kick + returning_a

    # --- blocked kicks (diagnostic count)
    bp = df[df["punt_blocked"] == 1]
    bf = df[df["field_goal_result"] == "blocked"]
    b_for = bp.groupby(["game_id", "defteam"]).size().add(bf.groupby(["game_id", "defteam"]).size(), fill_value=0)
    b_against = bp.groupby(["game_id", "posteam"]).size().add(bf.groupby(["game_id", "posteam"]).size(), fill_value=0)
    put("blocked_for", b_for, fill=0.0)
    put("blocked_against", b_against, fill=0.0)

    out = base.join(pd.DataFrame(feats)).reset_index()
    out["season"] = season
    fgs = df[(df["play_type"] == "field_goal") & df["field_goal_result"].notna() & df["kick_distance"].notna()
             & df["posteam"].notna()]
    fg_out = pd.DataFrame({"game_id": fgs["game_id"], "team": fgs["posteam"], "dist": fgs["kick_distance"],
                           "made": (fgs["field_goal_result"] == "made").astype(float)})
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(tg_path)
    fg_out.to_parquet(fg_path)
    return out, fg_out


def build_team_games() -> pd.DataFrame:
    tgs, fgs = [], []
    for s in SEASONS:
        print(f"  extracting {s}...", file=sys.stderr)
        a, b = extract_season(s)
        tgs.append(a)
        fgs.append(b)
    tg, fg = pd.concat(tgs, ignore_index=True), pd.concat(fgs, ignore_index=True)

    # Field-goal over expected: league make rate by distance bucket (a league-wide constant fit
    # on all years -- not team-specific, so the look-ahead is negligible), per attempt, summed by game.
    fg["bucket"] = (fg["dist"].clip(lower=20, upper=65) // 5).astype(int)
    fg["exp"] = fg.groupby("bucket")["made"].transform("mean")
    fg["oe"] = fg["made"] - fg["exp"]
    tg = tg.merge(fg.groupby(["game_id", "team"])["oe"].sum().rename("fg_oe").reset_index(), on=["game_id", "team"], how="left")
    tg["fg_oe"] = tg["fg_oe"].fillna(0.0)

    games = pd.read_csv(GAMES_URL, usecols=["game_id", "season", "week", "game_type", "home_team", "away_team",
                                            "home_score", "away_score", "spread_line", "total_line"])
    games = games[(games["game_type"] == "REG") & games["home_score"].notna() & games["spread_line"].notna()]
    tg = tg.merge(games[["game_id", "week", "home_score", "away_score", "spread_line", "total_line"]], on="game_id", how="inner")

    def norm(code: str):
        try:
            return normalize_team(code)
        except UnknownTeamError:
            return None
    cmap = {c: norm(c) for c in set(tg["team"]) | set(tg["opp"])}
    tg["team"], tg["opp"] = tg["team"].map(cmap), tg["opp"].map(cmap)
    tg = tg.dropna(subset=["team", "opp"])
    tg = tg[tg.groupby("game_id")["team"].transform("size") == 2].reset_index(drop=True)

    # opponent's value for any column = (sum of the two rows) - own
    def opp(col: str) -> pd.Series:
        return tg.groupby("game_id")[col].transform("sum") - tg[col]
    own_pts = np.where(tg["is_home"] == 1, tg["home_score"], tg["away_score"])
    tg["points_off"] = own_pts
    tg["points_def_allowed_pos"] = -opp("points_off")

    # --- oriented (higher = better) columns. Benchmarks first: the 8 stats production uses.
    tg["rush_off_epa_b"] = tg["rush_off_epa"]
    tg["pass_off_epa_b"] = tg["pass_off_epa"]
    tg["rush_def_epa_b"] = -opp("rush_off_epa")
    tg["pass_def_epa_b"] = -opp("pass_off_epa")
    tg["points_off_b"] = tg["points_off"]
    tg["points_def_b"] = tg["points_def_allowed_pos"]
    tg["turnovers_off_b"] = -tg["turnovers_off"]
    tg["turnovers_def_b"] = opp("turnovers_off")
    # candidates
    tg["expl_rush_off_c"] = tg["expl_rush_off"]
    tg["expl_pass_off_c"] = tg["expl_pass_off"]
    tg["expl_rush_def_c"] = -opp("expl_rush_off")
    tg["expl_pass_def_c"] = -opp("expl_pass_off")
    tg["pen_yds_c"] = -tg["pen_yds"]
    tg["pen_drawn_c"] = opp("pen_yds")
    tg["pen_presnap_c"] = -tg["pen_presnap_yds"]
    tg["top_share_c"] = tg["top_secs"] / tg.groupby("game_id")["top_secs"].transform("sum")
    tg["sec_per_play_c"] = tg["sec_per_play"]
    tg["drive_score_off_c"] = tg["drive_score_rate"]
    tg["drive_score_def_c"] = -opp("drive_score_rate")
    tg["st_epa_c"] = tg["st_epa"]
    tg["st_kp_c"] = tg["st_kp_epa"]
    tg["st_kicking_c"] = tg["st_kicking_epa"]
    tg["st_returning_c"] = tg["st_returning_epa"]
    tg["fp_net_c"] = tg["fp_own_start"] - opp("fp_own_start")
    tg["fg_oe_c"] = tg["fg_oe"]
    tg["blocked_net_c"] = tg["blocked_for"] - tg["blocked_against"]
    return tg


BENCH = ["rush_off_epa_b", "pass_off_epa_b", "rush_def_epa_b", "pass_def_epa_b",
         "points_off_b", "points_def_b", "turnovers_off_b", "turnovers_def_b"]
CANDS = {
    "explosive": ["expl_rush_off_c", "expl_pass_off_c", "expl_rush_def_c", "expl_pass_def_c"],
    "penalty": ["pen_yds_c", "pen_drawn_c", "pen_presnap_c"],
    "tempo/TOP": ["top_share_c", "sec_per_play_c", "drive_score_off_c", "drive_score_def_c"],
    "special teams": ["st_epa_c", "st_kp_c", "st_kicking_c", "st_returning_c", "fp_net_c", "fg_oe_c", "blocked_net_c"],
}


def redundancy_and_market(tg: pd.DataFrame, cands: list[str]) -> pd.DataFrame:
    pre = pregame_ewma(tg, BENCH + cands)
    pre.columns = [c.removeprefix("pre_") for c in pre.columns]
    side = pd.concat([tg[["game_id", "team", "is_home"]], pre], axis=1)
    h, a = side[side.is_home == 1].set_index("game_id"), side[side.is_home == 0].set_index("game_id")
    diff = (h.drop(columns=["team", "is_home"]) - a.drop(columns=["team", "is_home"]).reindex(h.index)).dropna(subset=BENCH)
    gm = tg[tg.is_home == 1].set_index("game_id")[["home_score", "away_score", "spread_line"]].reindex(diff.index)
    margin = gm["home_score"] - gm["away_score"]
    market_resid = margin - gm["spread_line"]  # spread_line > 0 = home favored
    Xb = np.column_stack([np.ones(len(diff)), diff[BENCH].to_numpy()])
    rows = []
    for c in cands:
        m = diff[c].notna()
        y = diff.loc[m, c].to_numpy()
        coef, *_ = np.linalg.lstsq(Xb[m.to_numpy()], y, rcond=None)
        resid = y - Xb[m.to_numpy()] @ coef
        r2 = 1 - resid.var() / y.var()
        n = int(m.sum())
        rs = pd.Series(resid, index=diff.index[m])
        row = {"stat": c, "n": n, "R2_on_8stats": r2}
        for lab, tgt in (("spread", gm["spread_line"]), ("margin", margin), ("mkt_resid", market_resid)):
            r_raw = float(diff.loc[m, c].corr(tgt[m]))
            r_res = float(rs.corr(tgt[m]))
            row[f"r_{lab}"] = r_raw
            row[f"z_{lab}_unexpl"] = r_res * np.sqrt(n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("stat")


def main() -> None:
    print("Building team-game table...", file=sys.stderr)
    tg = build_team_games()
    print(f"{len(tg)} team-games, {tg.game_id.nunique()} games, {tg.season.min()}-{tg.season.max()}")
    print("sanity: mean(home st_epa + away st_epa) per game (should be ~0):",
          round(float(tg.groupby("game_id")["st_epa"].sum().abs().mean()), 6))

    allcols = BENCH + [c for v in CANDS.values() for c in v]
    pr = persistence_report(tg, allcols)
    weakest_kept = pr.loc["turnovers_off_b", "ewma_r"]  # the lowest-weight stat production still keeps
    pr["verdict"] = ""
    for c in pr.index:
        if c in BENCH:
            pr.loc[c, "verdict"] = "(benchmark)"
        elif pr.loc[c, "z"] < 3:
            pr.loc[c, "verdict"] = "DROP: not persistent"
        elif pr.loc[c, "ewma_r"] < weakest_kept:
            pr.loc[c, "verdict"] = "weak: below turnovers_off"
        else:
            pr.loc[c, "verdict"] = "pass"

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== Stage 1: persistence (benchmarks = the 8 production stats) ===")
    print(pr.to_string())

    survivors = [c for c in pr.index if c not in BENCH and not pr.loc[c, "verdict"].startswith("DROP")]
    print("\n=== Stage 2: redundancy vs the 8 production stats, and market correlation ===")
    print("(R2_on_8stats: share of the pre-game home-minus-away diff already explained by the 8 stats."
          " z_*_unexpl: z of the UNEXPLAINED part vs spread / margin / market residual;"
          " |z| < 3 = nothing there.)")
    red = redundancy_and_market(tg, survivors)
    print(red.to_string())

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    pr.to_csv(os.path.join(HERE, "output", "v52_persistence.csv"))
    red.to_csv(os.path.join(HERE, "output", "v52_redundancy.csv"))
    print("\nCSV written to research/output/v52_*.csv")


if __name__ == "__main__":
    main()
