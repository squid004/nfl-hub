"""Compute the lineup FantasyPros' numbers would start, from a roster snapshot.

Uses fp_points when present (set by sources/fantasypros.enrich), else the league's native
projection. The slot template is inferred from whatever the snapshot is currently starting.

Swap suggestions are position-legal: a player can only be swapped for another of the same
position, or into/out of a FLEX slot if flex-eligible (RB/WR/TE). The returned `lineup` is
in a fixed display order (QB, RB, RB, WR, WR, TE, FLEX, DEF, K) — never sorted by points.
"""

from __future__ import annotations

from typing import Any, Optional

_STRICT = {"QB", "RB", "WR", "TE", "K", "DEF"}
_FLEX_POS = {"RB", "WR", "TE"}

_ELIG = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
    "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPERFLEX": {"QB", "RB", "WR", "TE"},
}
_FLEX_SLOTS = {"FLEX", "SUPERFLEX"}

# fixed top-to-bottom display order
_SLOT_RANK = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4, "SUPERFLEX": 5, "DEF": 6, "K": 7}


def _norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    if p in {"D/ST", "DST", "D", "DEFENSE"}:
        return "DEF"
    if p == "PK":
        return "K"
    return p


def _canon_slot(label: str) -> str:
    s = (label or "").upper().strip()
    if s in {"D/ST", "DST", "D", "DEF", "DEFENSE"}:
        return "DEF"
    if s in {"K", "PK"}:
        return "K"
    if s in _ELIG:
        return s
    parts = {_norm_pos(x) for x in s.replace("-", "/").split("/") if x}
    if parts == {"QB", "RB", "WR", "TE"}:
        return "SUPERFLEX"
    if parts and parts <= {"RB", "WR", "TE"}:
        return "FLEX"
    if len(parts) == 1:
        only = next(iter(parts))
        if only in _STRICT:
            return only
    if any(t in s for t in ("FLEX", "W/R/T", "R/W/T", "W/R", "REC")):
        return "FLEX"
    return "FLEX"


def _prime_pos(player: dict) -> str:
    """The player's real position (QB/RB/WR/TE/K/DEF), ignoring flex eligibility."""
    for src in player.get("elig") or []:
        n = _norm_pos(src)
        if n in _STRICT:
            return n
    for tok in (player.get("position") or "").replace("-", "/").split("/"):
        n = _norm_pos(tok)
        if n in _STRICT:
            return n
    return ""


def _elig_positions(player: dict) -> set[str]:
    out = {p for p in (_prime_pos(player),) if p}
    for src in player.get("elig") or []:
        n = _norm_pos(src)
        if n in _STRICT:
            out.add(n)
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

    pool = starters + bench
    template = [_canon_slot(p.get("slot", "")) for p in starters]

    # fill the most-constrained slots first; nested/disjoint eligibility -> greedy is optimal
    fill_order = sorted(range(len(template)), key=lambda i: len(_ELIG.get(template[i], _FLEX_POS)))
    used: set[int] = set()
    assigned: list[tuple[str, Optional[dict]]] = [("", None)] * len(template)
    for i in fill_order:
        slot = template[i]
        elig = _ELIG.get(slot, set(_FLEX_POS))
        cands = [p for p in pool if id(p) not in used and (_elig_positions(p) & elig)]
        if not cands:
            continue
        best = max(cands, key=_value)
        used.add(id(best))
        assigned[i] = (slot, best)

    # fixed display order: QB, RB, RB, WR, WR, TE, FLEX, DEF, K (never by points)
    ordered = sorted(
        [a for a in assigned if a[1] is not None],
        key=lambda a: (_SLOT_RANK.get(a[0], 8), -_value(a[1])),
    )
    lineup = [
        {
            "slot": slot,
            "name": pl["name"],
            "pos": _prime_pos(pl),
            "team": pl.get("pro_team", ""),
            "val": round(_value(pl), 2),
            "fp_ecr": pl.get("fp_ecr"),
        }
        for slot, pl in ordered
    ]

    opt_names = {row["name"] for row in lineup}
    cur_names = {p["name"] for p in starters}

    # position-legal swaps. Pass 1 pairs each incoming player with a same-position player
    # being benched. Pass 2 routes any leftovers through the lineup's FLEX/SUPERFLEX slot
    # (a WR entering bumps a WR to FLEX, which can bump an RB out — shown as a FLEX swap,
    # never as "start WR for RB" against a hard RB slot).
    adds = sorted(
        [(slot, pl) for slot, pl in ordered if pl["name"] not in cur_names],
        key=lambda a: -_value(a[1]),
    )
    sit_pool = [p for p in starters if p["name"] not in opt_names]
    has_flex = any(_canon_slot(s) in _FLEX_SLOTS for s in template)
    has_superflex = any(_canon_slot(s) == "SUPERFLEX" for s in template)
    swaps: list[dict[str, Any]] = []
    leftovers: list[tuple[str, dict]] = []

    for slot, pl in adds:
        in_pos = _prime_pos(pl)
        same = [y for y in sit_pool if _prime_pos(y) == in_pos]
        if not same:
            leftovers.append((slot, pl))
            continue
        out = min(same, key=_value)
        sit_pool.remove(out)
        swaps.append({
            "start": pl["name"], "start_pos": in_pos,
            "sit": out["name"], "sit_pos": in_pos, "slot": slot,
            "gain": round(_value(pl) - _value(out), 2),
        })

    for slot, pl in leftovers:
        in_pos = _prime_pos(pl)
        routable = has_superflex or (has_flex and in_pos in _FLEX_POS)
        if sit_pool and routable:
            out = min(sit_pool, key=_value)
            sit_pool.remove(out)
            swaps.append({
                "start": pl["name"], "start_pos": in_pos,
                "sit": out["name"], "sit_pos": _prime_pos(out),
                "slot": "SUPERFLEX" if in_pos == "QB" else "FLEX",
                "gain": round(_value(pl) - _value(out), 2),
            })
        else:
            swaps.append({"start": pl["name"], "start_pos": in_pos, "slot": slot,
                          "sit": None, "sit_pos": "", "gain": round(_value(pl), 2)})

    swaps.sort(key=lambda s: -s["gain"])

    for p in starters:
        p["fp_suggest"] = "sit" if p["name"] not in opt_names else "start"
    for p in bench:
        p["fp_suggest"] = "start" if p["name"] in opt_names else None

    cur_pts = round(sum(_value(p) for p in starters), 2)
    opt_pts = round(sum(row["val"] for row in lineup), 2)
    return {
        "current_points": cur_pts,
        "optimal_points": opt_pts,
        "delta": round(opt_pts - cur_pts, 2),
        "lineup": lineup,
        "swaps": swaps,
        "using_fp": any(p.get("fp_points") is not None for p in pool),
    }
