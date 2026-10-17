"""Weekly win-probability optimizer for the straight pick'em pool (replaces the deviation-budget heuristic in
edge_core.py as the *recommendation* engine; edge_core's devig/bias helpers are still its inputs).

Pool rules (docs/improvement-plan-2026-10-15.md A6): each week 80% of that week's pot goes to the
weekly winner (most correct picks; ties go to the closest Monday-night total guess, exact ties split);
the other 20% of every week's pot builds a season pot paid 60/30/10 to the top three in season wins.
THIS ENGINE OPTIMIZES THE WEEK ONLY: it maximizes the chance of winning the weekly prize and ignores the season pot
(an earlier version added a rest-of-season term; removed 2026-10-17 so the two goals stay separate). The metric is
simulated win percentage, not dollars: each game's impact rating is the card's win chance minus the win chance with just that
game on the other side and every other pick re-optimized.

The model is market-based on purpose -- it never uses the Power Rankings model. It needs only:
the de-vigged market win probability per game, each opponent's pick tendencies, and the standings.

  * Outcomes: each unfinished game is an independent Bernoulli draw from the market probability.
  * Opponents: each picks the favorite with probability
        sigmoid(A + B*logit(market P(fav)) + C*logit(national pick % of the favorite) + offset_j)
    where offset_j is that entrant's persistent chalk tendency, shrunk from their own past picks
    (v65: the field is chalkier than the market, and entrants differ persistently -- the independence
    assumption in the first-cut plan failed its check). Picks already revealed (games that have kicked
    off, picks lock by game day) are used as-is; only open games are sampled.
  * Monday-night tiebreaker: opponents' guesses ~ Normal(posted total + TB_MEAN_OFFSET, TB_SD) rounded;
    the actual total ~ Normal(posted total, TOTAL_SD).
  * Objective: P(win the week) -- the weekly prize EV is 0.8 * P * pot. Correct picks matter only through that probability.

`optimize()` re-solves for only the still-open picks and the tiebreaker given the state of the week,
which is what makes the sequential, per-game-day use work: finished games and revealed picks are
conditioned on, everything else is simulated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_SIMS = 20000
WEEKLY_SHARE = 0.8

# Pick-share model (v66 all-data fit on 2026 weeks 1-5, see research/edge_signal_test_v66_*.py).
PICK_A, PICK_B, PICK_C = 0.02, 0.33, 0.76
OFFSET_TAU = 0.36          # sd of entrants' persistent chalk offset, logit scale (v65 method of moments)
TB_MEAN_OFFSET, TB_SD = -1.5, 4.5   # opponents' tiebreaker guess vs the posted total (v65)
TOTAL_SD = 13.3            # actual game total vs the posted total (nflverse history)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class PoolGame:
    game_id: str
    home: str
    away: str
    p_home: float                     # de-vigged market P(home wins)
    lock_key: str                     # picks for games with the same key lock together (the game day)
    total_line: float | None = None
    national_fav_pct: float | None = None   # national pick % of the market favorite
    home_won: bool | None = None      # set once final
    locked: bool = False              # has kicked off: picks are revealed and my pick is fixed

    @property
    def fav(self) -> str:
        return self.home if self.p_home >= 0.5 else self.away

    @property
    def p_fav(self) -> float:
        return max(self.p_home, 1 - self.p_home)


@dataclass
class Entrant:
    name: str
    offset: float = 0.0               # persistent chalk tendency (logit)
    picks: dict[str, str | None] = field(default_factory=dict)   # game_id -> team, revealed picks only
    tiebreak: float | None = None     # revealed only after Monday kickoff


def pick_home_prob(game: PoolGame, offset: float, a=PICK_A, b=PICK_B, c=PICK_C) -> float:
    nat = game.national_fav_pct if game.national_fav_pct is not None else game.p_fav
    f = float(_sig(a + b * _logit(game.p_fav) + c * _logit(nat) + offset))
    return f if game.fav == game.home else 1.0 - f


def estimate_offsets(history: list[tuple[list[PoolGame], dict[str, dict[str, str]]]], names: list[str], tau: float = OFFSET_TAU,
                     a=PICK_A, b=PICK_B, c=PICK_C) -> dict[str, float]:
    """Posterior-mean chalk offset per entrant from past weeks. `history` = [(games, {name: {game_id: team}})].
    One Newton step under a Normal(0, tau^2) prior: u = sum(y - f) / (sum f(1-f) + 1/tau^2)."""
    num = {n: 0.0 for n in names}
    den = {n: 0.0 for n in names}
    for games, picks_by_name in history:
        for g in games:
            f0 = pick_home_prob(g, 0.0, a, b, c)
            f_fav = f0 if g.fav == g.home else 1.0 - f0
            for n in names:
                t = picks_by_name.get(n, {}).get(g.game_id)
                if t is None:
                    continue
                num[n] += (1.0 if t == g.fav else 0.0) - f_fav
                den[n] += f_fav * (1 - f_fav)
    return {n: num[n] / (den[n] + 1.0 / tau ** 2) for n in names}


@dataclass
class Result:
    card: dict[str, str]              # game_id -> recommended team (locked games = my actual pick)
    guess: int | None                 # recommended tiebreaker
    p_win: float
    expected_correct: float
    ev: float                         # in weekly-pot units: 0.8 * p_win
    steps: list[dict]                 # flips applied, in order
    per_game: dict[str, dict]


class Sim:
    """Common-random-number simulation of the week from a given state."""

    def __init__(self, games: list[PoolGame], opponents: list[Entrant], me: Entrant, n: int = N_SIMS, seed: int = 0,
                 params=(PICK_A, PICK_B, PICK_C), tiebreak_game_id: str | None = None, actual_total: float | None = None):
        rng = np.random.default_rng(seed)
        self.games, self.opponents, self.me, self.n = games, opponents, me, n
        G, J = len(games), len(opponents)
        self.gidx = {g.game_id: i for i, g in enumerate(games)}
        # outcomes
        p_home = np.array([g.p_home for g in games])
        out = rng.random((n, G)) < p_home
        for i, g in enumerate(games):
            if g.home_won is not None:
                out[:, i] = g.home_won
        self.out_home = out
        # opponents' picks
        a, b, c = params
        ph = np.array([[pick_home_prob(g, e.offset, a, b, c) for g in games] for e in opponents]) if J else np.zeros((0, G))
        picks = rng.random((n, J, G)) < ph[None]
        valid = np.ones((J, G), bool)
        for j, e in enumerate(opponents):
            for i, g in enumerate(games):
                if g.game_id in e.picks:
                    t = e.picks[g.game_id]
                    if t is None:
                        valid[j, i] = False
                    else:
                        picks[:, j, i] = (t == g.home)
        self.opp_correct = ((picks == out[:, None, :]) & valid[None]).sum(2)          # (n, J)
        # tiebreaker
        if tiebreak_game_id is None:
            lines = [(i, g) for i, g in enumerate(games) if g.total_line is not None]
            tiebreak_game_id = max(lines, key=lambda t: (t[1].lock_key, t[0]))[1].game_id if lines else None
        self.tb_game = tiebreak_game_id
        line = games[self.gidx[tiebreak_game_id]].total_line if tiebreak_game_id else 45.0
        self.line = float(line if line is not None else 45.0)
        total = np.round(self.line + TOTAL_SD * rng.standard_normal(n)) if actual_total is None else np.full(n, float(actual_total))
        self.total = total
        guesses = np.round(self.line + TB_MEAN_OFFSET + TB_SD * rng.standard_normal((n, J)))
        for j, e in enumerate(opponents):
            if e.tiebreak is not None:
                guesses[:, j] = e.tiebreak
        self.max_opp = self.opp_correct.max(1) if J else np.zeros(n)
        tied = self.opp_correct == self.max_opp[:, None]
        dist = np.where(tied, np.abs(guesses - total[:, None]), np.inf)
        self.dmin = dist.min(1)
        self.kmin = (dist == self.dmin[:, None]).sum(1)
        self.rng = rng
        self.contrib_home = out.astype(np.int16)
        self.contrib_away = (~out).astype(np.int16)
        # a game that has locked with NO pick from me (key present, value None) can never score
        self.my_valid = np.array([not (g.locked and g.game_id in me.picks and me.picks[g.game_id] is None) for g in games])

    # -- card evaluation ----------------------------------------------------------------------------
    def correct(self, side_home: np.ndarray) -> np.ndarray:
        return (np.where(side_home[None, :], self.contrib_home, self.contrib_away) * self.my_valid[None, :]).sum(1)

    def p_win_grid(self, c: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """P(win week) for each tiebreak guess in `grid`, given my correct counts c (n,)."""
        win_clear = (c > self.max_opp).mean()
        idx = np.where(c == self.max_opp)[0]
        if not len(idx):
            return np.full(len(grid), win_clear)
        d = np.abs(grid[None, :] - self.total[idx][:, None])                       # (m, K)
        dmin, kmin = self.dmin[idx][:, None], self.kmin[idx][:, None]
        share = np.where(d < dmin, 1.0, np.where(d == dmin, 1.0 / (kmin + 1), 0.0))
        return win_clear + share.sum(0) / self.n

    def guess_grid(self) -> np.ndarray:
        return np.arange(self.line - 14, self.line + 15, 1.0)


def optimize(games: list[PoolGame], opponents: list[Entrant], me: Entrant, *, n: int = N_SIMS, seed: int = 0,
             params=(PICK_A, PICK_B, PICK_C), start: dict[str, str] | None = None, actual_total: float | None = None) -> Result:
    """Greedy single-flip search over the OPEN games (locked games keep my revealed pick), jointly with the
    tiebreaker guess. Start = my given card for open games if provided else the market favorite.
    Objective = WEEKLY_SHARE * P(win week)."""
    sim = Sim(games, opponents, me, n, seed, params, actual_total=actual_total)
    G = len(games)
    side = np.zeros(G, bool)
    for i, g in enumerate(games):
        pick = me.picks.get(g.game_id) if g.locked else (start or {}).get(g.game_id)
        side[i] = (pick == g.home) if pick is not None else (g.fav == g.home)  # no pick on a locked game: masked in Sim.correct
    open_idx = [i for i, g in enumerate(games) if not g.locked]
    grid = sim.guess_grid()

    def objective(s):
        c = sim.correct(s)
        pw = sim.p_win_grid(c, grid)
        k = int(np.argmax(pw))
        return WEEKLY_SHARE * pw[k], pw[k], c.mean(), grid[k]

    steps = []
    best = objective(side)
    improved = True
    while improved:
        improved = False
        cand = None
        for i in open_idx:
            side[i] = not side[i]
            val = objective(side)
            side[i] = not side[i]
            if val[0] > best[0] + 1e-5 and (cand is None or val[0] > cand[1][0]):
                cand = (i, val)
        if cand is not None:
            i, val = cand
            side[i] = not side[i]
            steps.append({"game_id": games[i].game_id, "to": games[i].home if side[i] else games[i].away, "ev": float(val[0])})
            best = val
            improved = True
    ev, pw, ec, guess = best
    per_game = {}
    for i, g in enumerate(games):
        base = objective(side)
        alt = side.copy()
        alt[i] = not alt[i]
        v = objective(alt) if not g.locked else base
        pf = pick_home_prob(g, 0.0, *params)
        per_game[g.game_id] = {"pick": g.home if side[i] else g.away, "fav": g.fav, "p_fav": g.p_fav,
                               "pool_fav_share": pf if g.fav == g.home else 1 - pf,
                               "ev_if_flipped": float(v[0]), "p_win_if_flipped": float(v[1]), "locked": g.locked,
                               "contrarian": bool((g.home if side[i] else g.away) != g.fav)}
    return Result(card={g.game_id: (g.home if side[i] else g.away) for i, g in enumerate(games)}, guess=int(guess), p_win=float(pw),
                  expected_correct=float(ec), ev=float(ev), steps=steps, per_game=per_game)


def evaluate_card(games: list[PoolGame], opponents: list[Entrant], me: Entrant, card: dict[str, str], guess: float, *,
                  n: int = N_SIMS, seed: int = 0, params=(PICK_A, PICK_B, PICK_C), actual_total: float | None = None) -> dict:
    """Model P(win week) and E[correct] for a given card and tiebreak guess (same simulation as optimize)."""
    sim = Sim(games, opponents, me, n, seed, params, actual_total=actual_total)
    side = np.array([card.get(g.game_id, g.fav) == g.home for g in games])
    c = sim.correct(side)
    return {"p_win": float(sim.p_win_grid(c, np.array([float(guess)]))[0]), "expected_correct": float(c.mean())}
