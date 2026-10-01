"""Leverage/fade math for the straight moneyline pick'em pool.

Ported near-verbatim from github.com/squid004/pickem-edge (SPEC.md there has the full
rationale). Pure, stdlib-only functions — do not change the formulas without reading
SPEC.md's section 1: the objective is finishing first in the pool, not maximizing correct
picks, so "optimize this to pick more winners" is solving the wrong problem.

Exact-decimal test vectors (SPEC.md section 6) are reproduced in
nflhub/sources/edge_core_vectors.py for verification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional


# --- devig (SPEC.md 2.1) -----------------------------------------------------

@dataclass(frozen=True)
class DevigResult:
    raw_fav: float
    raw_dog: float
    vig: float
    p: float
    vig_flagged: bool


def american_to_implied_prob(odds: int) -> float:
    """Single leg: American odds -> implied probability WITH vig."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def devig_two_way(
    favorite_odds: int, underdog_odds: int, *, vig_warn_threshold: float = 0.08
) -> DevigResult:
    """Multiplicative devig (proportional normalization). Identical math regardless of
    sign, so a both-negative pick'em line (e.g. -110/-110) devigs correctly with no
    special-casing.
    """
    raw_fav = american_to_implied_prob(favorite_odds)
    raw_dog = american_to_implied_prob(underdog_odds)
    vig = raw_fav + raw_dog - 1.0
    p = raw_fav / (raw_fav + raw_dog)
    return DevigResult(
        raw_fav=raw_fav, raw_dog=raw_dog, vig=vig, p=p,
        vig_flagged=vig > vig_warn_threshold,
    )


# --- pool popularity + leverage (SPEC.md 2.2/2.3) ----------------------------

def pool_popularity(national_pick_pct: float, pool_adjustment: float = 0.0) -> float:
    """f = clamp(national_pick_pct + pool_adjustment, 0.02, 0.98).

    f is independent of p — an estimate of what fraction of THIS pool takes the
    favorite, from a national baseline plus a learned per-team correction.
    """
    return max(0.02, min(0.98, national_pick_pct + pool_adjustment))


def is_eligible(p_favorite: float, *, max_p: float = 0.65) -> bool:
    return p_favorite <= max_p


def leverage_score(p_favorite: float, f: float, *, max_p: float = 0.65) -> Optional[float]:
    """leverage = (1-p)*f, gated: None if p exceeds max_p (too heavy a lock to fade —
    fading a big underdog is worse risk/reward than fading a near-coin-flip favorite).
    """
    if not is_eligible(p_favorite, max_p=max_p):
        return None
    return (1 - p_favorite) * f


# --- deviation budget (SPEC.md 2.4) ------------------------------------------

class Standing(StrEnum):
    LEADING = "LEADING"
    EARLY = "EARLY"
    MIDDLE = "MIDDLE"
    BEHIND = "BEHIND"


def deviation_budget(standing: Standing, pool_size: int) -> int:
    """Leading late, mirror the field. Behind late, correct picks gained in parallel
    with the leader don't help — so budget grows with how far behind you are, and
    scales up for bigger pools where more separation is needed.
    """
    if standing is Standing.LEADING:
        return 0
    if standing is Standing.EARLY:
        return 1
    if standing is Standing.MIDDLE:
        return 2 if pool_size >= 30 else 1
    return 4 if pool_size >= 30 else 3  # Standing.BEHIND


# --- season standing (nfl-hub addition, NOT in pickem-edge's SPEC.md) --------
# The original spec left Standing as a manual weekly call with no derivation rule. This
# formula ties directly to deviation_budget's own stated rationale above ("behind late,
# picks gained in parallel with the leader don't help") -- BEHIND specifically means the
# gap can no longer be closed by matching the leader pick-for-pick; it requires the leader
# to actually miss some.
#
# TODO (deliberately deferred, 2026-09-30): a SEPARATE "win this one week" budget/recommendation
# is still open -- unlike the season-long case above, there's no formula here that follows
# cleanly from first principles yet. Leverage_score() below is already a single-game
# differentiation value and doesn't depend on season standing, so the two objectives can
# already disagree on any given game; what's missing is how MANY leverage-ranked games to
# act on when the sole goal is winning this week's mini-contest (pool_size probably matters
# -- bigger pools need more differentiation for a realistic shot -- but "budget ~ pool_size/10"
# was only ever a starting guess, not validated). When this gets picked back up, surface it
# as a second, clearly-labeled recommendation alongside the season one rather than replacing it.

REG_SEASON_WEEKS = 18


def derive_standing(
    my_correct_season: int, leader_correct_season: int, week: int,
    *, early_weeks: int = 3, total_weeks: int = REG_SEASON_WEEKS,
) -> Standing:
    if week <= early_weeks:
        return Standing.EARLY
    gap = leader_correct_season - my_correct_season
    if gap <= 0:
        return Standing.LEADING
    games_remaining = max(total_weeks - week, 0)
    if gap >= games_remaining:
        return Standing.BEHIND
    return Standing.MIDDLE


# --- recommendation (SPEC.md 2.5) --------------------------------------------

class Recommendation(StrEnum):
    FADE = "FADE"
    CHALK = "CHALK"
    NO_PLAY = "NO_PLAY"  # leverage exists but blocked by budget or below MIN_LEVERAGE


@dataclass(frozen=True)
class GameRecommendation:
    recommendation: Recommendation
    leverage: Optional[float]
    eligible: bool
    reason: str


def recommend(
    p_favorite: float, f: float, *,
    budget_remaining: int, max_p: float = 0.65, min_leverage: float = 0.25,
) -> GameRecommendation:
    """Does NOT decrement budget_remaining — the caller sorts a week's games by leverage
    descending and consumes budget greedily, so this stays a pure per-game decision.
    """
    eligible = is_eligible(p_favorite, max_p=max_p)
    if not eligible:
        return GameRecommendation(
            recommendation=Recommendation.CHALK, leverage=None, eligible=False,
            reason=f"p {p_favorite:.3f} > MAX_P {max_p:.3f}",
        )

    leverage = leverage_score(p_favorite, f, max_p=max_p)
    assert leverage is not None

    if leverage < min_leverage:
        return GameRecommendation(
            recommendation=Recommendation.NO_PLAY, leverage=leverage, eligible=True,
            reason=f"leverage {leverage:.3f} < MIN_LEVERAGE {min_leverage:.3f}",
        )
    if budget_remaining <= 0:
        return GameRecommendation(
            recommendation=Recommendation.NO_PLAY, leverage=leverage, eligible=True,
            reason="deviation budget exhausted",
        )
    return GameRecommendation(
        recommendation=Recommendation.FADE, leverage=leverage, eligible=True,
        reason=f"leverage {leverage:.3f} >= MIN_LEVERAGE {min_leverage:.3f}, budget available",
    )


# --- pool bias learning (SPEC.md 4) ------------------------------------------

@dataclass(frozen=True)
class BiasResult:
    bias_value: float
    n_observations: int


def compute_bias(
    pool_pcts: list[float], national_pcts: list[float], *, damping_floor: int = 3
) -> BiasResult:
    """raw_bias = mean(pool_pct) - mean(national_pct), damped toward zero under
    damping_floor observations (SPEC.md states this narratively, not as a formula; this
    is a linear ramp — 0% weight at n=0, full weight at n>=damping_floor).
    """
    if len(pool_pcts) != len(national_pcts):
        raise ValueError("pool_pcts and national_pcts must be the same length")
    n = len(pool_pcts)
    if n == 0:
        return BiasResult(bias_value=0.0, n_observations=0)
    raw_bias = (sum(pool_pcts) / n) - (sum(national_pcts) / n)
    damped = raw_bias * min(n, damping_floor) / damping_floor
    return BiasResult(bias_value=damped, n_observations=n)


# --- Monday-night tiebreaker (SPEC.md 2.6) -----------------------------------
# Interactive 2-input formula; the frontend implements this directly in JS (no data to
# persist), but it's kept here too since it's one line and this module is the reference.

def monday_night_tiebreaker(my_estimate: float, crowd_estimate: float) -> float:
    """Shade away from the crowd: people cluster on the posted total and the next round
    number up, so being closest wins more than being right.
    """
    return my_estimate + (-2 if my_estimate < crowd_estimate else 2)


# --- pool leaderboard / "chalk is king" reminder (nfl-hub addition, not in SPEC.md) ----------
# Ad-hoc analysis (2026-10-01) of this specific pool's weeks 1-3: both season co-leaders and
# every individual week's winner picked chalk on the large majority of games (85-98% of
# picks for the season leaders; never more than 5 of 16 picks on a dog for any week's
# winner), and beat a robot that always picks the favorite on the identical slate by only
# +1 or +2 games out of 48 -- well within noise. This surfaces that same read live, every
# refresh, instead of it being a one-off finding that goes stale.

def pool_leaderboard_summary(
    picks: list[dict[str, Any]], results: dict[tuple[int, str], dict[str, bool]],
) -> dict[str, Any]:
    """`picks`: {week, opponent, team_picked} rows, team codes already normalized to this
    app's own convention. `results`: (week, team_code) -> {"fav": is this team that week's
    market favorite, "won": did this team win} for every graded game this season -- built by
    the caller from whatever historical odds/results source it has (see refresh.py), so this
    stays pure and doesn't need to know where that data comes from.

    "Dog pick" = a pick that was NOT the market favorite. Returns each week's winner(s) (ties
    included) and the season leader(s), each with how much of their pick history was chalk --
    the reminder this exists for.
    """
    per_week: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
    season_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "dog_picks": 0})

    for p in picks:
        week, opp, team = p["week"], p["opponent"], p["team_picked"]
        r = results.get((week, team))
        if r is None:
            continue
        stat = per_week[week].setdefault(opp, {"correct": 0, "total": 0, "dog_picks": 0})
        for d in (stat, season_totals[opp]):
            d["total"] += 1
            if r["won"]:
                d["correct"] += 1
            if not r["fav"]:
                d["dog_picks"] += 1

    week_winners: list[dict[str, Any]] = []
    winner_dog_sum = winner_total_sum = 0
    for week in sorted(per_week):
        wk = per_week[week]
        if not wk:
            continue
        best = max(s["correct"] for s in wk.values())
        names = sorted(opp for opp, s in wk.items() if s["correct"] == best)
        dogs = sum(wk[n]["dog_picks"] for n in names)
        tot = sum(wk[n]["total"] for n in names)
        week_winners.append({
            "week": week, "names": names, "correct": best,
            "dog_pct": round(dogs / tot, 4) if tot else None,
        })
        winner_dog_sum += dogs
        winner_total_sum += tot

    season_leaders = None
    if season_totals:
        best = max(s["correct"] for s in season_totals.values())
        names = sorted(opp for opp, s in season_totals.items() if s["correct"] == best)
        dogs = sum(season_totals[n]["dog_picks"] for n in names)
        tot = sum(season_totals[n]["total"] for n in names)
        season_leaders = {
            "names": names, "correct": best, "total": tot // len(names),
            "dog_pct": round(dogs / tot, 4) if tot else None,
        }

    return {
        "week_winners": week_winners,
        "week_winner_dog_pct": round(winner_dog_sum / winner_total_sum, 4) if winner_total_sum else None,
        "season_leaders": season_leaders,
    }
