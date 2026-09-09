"""Compute the lineup FantasyPros' numbers would start, from a roster snapshot.

Uses fp_points when present (set by sources/fantasypros.enrich), else the league's native
projection. The slot template is inferred from whatever the snapshot is currently starting,
so it adapts to each league without needing the league settings endpoint.

Greedy fill from the most-constrained slot outward is optimal here because standard NFL
slot eligibilities are either disjoint (QB vs RB vs WR ...) or nested supersets (FLEX
contains RB/WR/TE, SUPERFLEX adds QB).
"""

from __future__ import annotations

from typing import Any, Optional

_STRICT = {"QB", "RB", "WR", "TE", "K", "DEF"}

_ELIG = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
    "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"},
    "W/R": {"RB", "WR"},
    "REC": {"WR", "TE"},
    "SUPERFLEX": {"QB", "RB", "WR", "TE"},
}


def _norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    if p in {"D/ST", "DST", "D", "DEFENSE"}:
        return "DEF"
    if p in {"PK"}:
        return "K"
    return p


def _canon_slot(label: str) -> str:
    s = (label or "").upper().strip()
    if s in _ELIG:
        return s
    parts = {_norm_pos(x) for x in s.replace("-", "/").split("/") if x}
    if parts == {"RB", "WR", "TE"}:
        return "FLEX"
    if parts == {"QB", "RB", "WR", "TE"}:
        return "SUPERFLEX"
    if parts == {"RB", "WR"}:
        return "W/R"
    if parts == {"WR", "TE"}:
        return "REC"
    if len(parts) == 1:
        only = next(iter(parts))
        if only in _STRICT:
            return only
    if "FLEX" in s or "W/R/T" in s or "R/W/T" in s:
        return "FLEX"
    return "FLEX"


def _elig_positions(player: dict) -> set[str]:
    raw = player.get("elig") or []
    out = {_norm_pos(x) for x in raw if _norm_pos(x) in _STRICT}
    if not out:
        # fall back to the display position string
        for tok in (player.get("position") or "").replace("-", "/").split("/"):
            if _norm_pos(tok) in _STRICT:
                out.add(_norm_pos(tok))
    return out


def _value(player: dict) -> float:
    v = player.get("fp_points")
    if v is None:
        v = player.get("projected")
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def optimize(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    starters = payload.get("starters", [])
    bench = payload.get("bench", [])
    if not starters:
        return None

    pool = [p for p in (starters + bench) if _norm_pos(p.get("position", "")) != "" ]
    slots = [_canon_slot(p.get("slot", "")) for p in starters]
    slots.sort(key=lambda s: len(_ELIG.get(s, {"RB", "WR", "TE"})))

    used: set[int] = set()
    lineup: list[dict[str, Any]] = []
    for s in slots:
        elig = _ELIG.get(s, {"RB", "WR", "TE"})
        cands = [p for p in pool if id(p) not in used and (_elig_positions(p) & elig)]
        if not cands:
            lineup.append({"slot": s, "name": None})
            continue
        best = max(cands, key=_value)
        used.add(id(best))
        lineup.append(
            {
                "slot": s,
                "name": best["name"],
                "pos": best.get("position", ""),
                "team": best.get("pro_team", ""),
                "val": round(_value(best), 2),
                "fp_ecr": best.get("fp_ecr"),
            }
        )

    opt_names = {row["name"] for row in lineup if row["name"]}
    cur_names = {p["name"] for p in starters}
    starts = [row for row in lineup if row["name"] and row["name"] not in cur_names]
    sits = sorted((p for p in starters if p["name"] not in opt_names), key=_value, reverse=True)

    swaps = []
    for i, row in enumerate(sorted(starts, key=lambda r: r["val"], reverse=True)):
        out = sits[i] if i < len(sits) else None
        swaps.append(
            {
                "start": row["name"],
                "start_pos": row["pos"],
                "slot": row["slot"],
                "sit": out["name"] if out else None,
                "sit_pos": out.get("position", "") if out else "",
                "gain": round(row["val"] - (_value(out) if out else 0.0), 2),
            }
        )

    for p in starters:
        p["fp_suggest"] = "sit" if p["name"] not in opt_names else "start"
    for p in bench:
        p["fp_suggest"] = "start" if p["name"] in opt_names else None

    cur_pts = round(sum(_value(p) for p in starters), 2)
    opt_pts = round(sum(row.get("val", 0) for row in lineup), 2)
    return {
        "current_points": cur_pts,
        "optimal_points": opt_pts,
        "delta": round(opt_pts - cur_pts, 2),
        "lineup": lineup,
        "swaps": swaps,
        "using_fp": any(p.get("fp_points") is not None for p in starters + bench),
    }
