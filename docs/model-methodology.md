# NFL Hub prediction model — methodology, timeline, and null results

Written 2026-10-07. Covers `nflhub/sources/team_ratings.py`, the module that produces every
prediction surfaced on the Moneyline Pick'em, Against the Spread, Power Rankings, and Model
Trends tabs. Every number below was pulled from production constants or the live
`historical_power_rankings` backtest dataset (4,976–4,991 graded games, 2007–2025) at the
time of writing, not recalled from memory — it will drift as the model keeps evolving;
treat this as a snapshot, not a living spec.

## 1. What the current model does

### 1.1 Foundation: per-team ratings

Every prediction starts from the same 8 team-level stats, computed per team per week:

| Stat | What it measures |
|---|---|
| `rush_off_epa` / `pass_off_epa` | EPA/play on offense, rush and pass separately |
| `rush_def_epa_allowed` / `pass_def_epa_allowed` | EPA/play allowed on defense |
| `points_off` / `points_def_allowed` | Points scored / allowed per game |
| `turnovers_off` / `turnovers_def_forced` | Turnovers committed / forced per game |

- **Garbage-time excluded** from the four EPA stats and both turnover stats: any play with
  win probability outside `[0.05, 0.95]` is dropped before averaging (`WP_LO`/`WP_HI`).
  Points are **not** filtered this way — a final score is the whole-game outcome, filtering
  it would fight the stat's own meaning.
- **EWMA + season-prior fade**: each stat is an exponentially-weighted moving average
  (`EWMA_ALPHA = 0.2`, ~4-game half-life) within a season. A new season starts from last
  season's own ending rating, which fades out **linearly to exactly zero** over
  `SEASON_PRIOR_GAMES = 5` games — a decade-old season has zero path into this year's rating
  once last season is behind it, not an ever-shrinking nonzero weight.
- **Data window**: `FIRST_SEASON = 2007` (a project choice, not nflverse's actual floor —
  the underlying play-by-play goes back to 1999; see `docs/nflverse-data-catalog.md`).
- **Display scale**: the 4 EPA stats are shown 0–100, anchored to the best/worst value *ever
  recorded* across the full dataset, so a weak season's best team doesn't read as inflated.

### 1.2 The composite Power Score

```
POWER_WEIGHTS = {
  rush_off_epa: 0.0850, pass_off_epa: 0.1536,
  rush_def_epa_allowed: 0.0572, pass_def_epa_allowed: 0.0578,
  points_off: 0.1587, points_def_allowed: 0.1006,
  turnovers_off: 0.0217, turnovers_def_forced: 0.0000,
}
```

One joint L2-regularized logistic regression (`L2 = 0.3`, chosen by walk-forward AUC over a
7-point log grid) across all 8 stats at once, each first oriented "higher = always better,"
coefficients constrained `>= 0`. The non-negativity constraint fixes a real failure mode: an
unconstrained fit sign-flipped `turnovers_off` (committing more turnovers came out
*positively* weighted) because the 8 stats overlap heavily — non-negative coefficients can
only shrink a redundant stat toward zero, never flip its sign. `turnovers_def_forced`
optimizes to exactly 0.0 once the other 7 are known. **Walk-forward AUC 0.6746 vs. 0.7224
for the closing market spread alone — this does not beat the market at picking winners.**
Shown as descriptive context (Power Rankings tab, college-poll-style display), not a
betting edge.

### 1.3 `delta`: the per-matchup prediction

For one matchup, each stat's home-minus-away difference is standardized (z-scored against
the full backtest population), weighted by `POWER_WEIGHTS`, and summed to `delta_raw`. Four
corrections are then added on top — all additive, all fit **once, offline**, on the full
graded dataset, none of them touch `POWER_WEIGHTS` or the 8-stat composite itself:

| Correction | Weight | What it captures | Hit-rate mover? |
|---|---|---|---|
| QB health | `QB_QUALITY_GAP_WEIGHT = -0.5903` | Trailing EPA/dropback gap between a team's normal starter and whoever's actually playing | No (calibration only) |
| Skill-position health | `SKILL_EPA_OUT_WEIGHT = -0.0362` | Summed trailing EPA of every Out/Doubtful RB/WR/TE/FB | No (calibration only) |
| Home-field advantage | `HFA_WEIGHT = 0.9169` | Trailing-5-season home win rate, shrunk toward all-time average (`HFA_SHRINKAGE_K = 100`) | **Yes** |
| Momentum | `MOMENTUM_WEIGHT = 0.0262` | Signed current win/loss streak, reset at season boundaries and ties | **Yes** |

```
delta = delta_raw
      + QB_QUALITY_GAP_WEIGHT     * (home_qb_gap  - away_qb_gap)
      + SKILL_EPA_OUT_WEIGHT      * (home_skill_out - away_skill_out)
      + HFA_WEIGHT                * home_field_logit_by_season[season]
      + MOMENTUM_WEIGHT           * (home_streak  - away_streak)
```

`delta_raw` is kept alongside `delta` everywhere for comparison. `delta_win_prob(delta)`
converts the composite into a calibrated win probability via a curve fit against real
backtested accuracy at each confidence level (QB-injury games excluded from that specific
fit, so an injury-widened delta doesn't inflate its own calibration).

**Why QB/skill health don't move hit rate, mechanically**: the term is small relative to
everything else in `delta` (mean magnitude ≈0.10, 90th percentile ≈0.23) and only nonzero
for the ~1,200 graded games with a report-listed Out/Doubtful QB. It *can* flip a pick (62
of those 1,196 games flip, 5.2%), but the flips split almost evenly (32 improved, 30
regressed) — which is exactly why it's framed as a calibration correction (does `delta`
honestly reflect known health context?) rather than a pick-accuracy play.

### 1.4 Points prediction and weather

A **separate** regression (`POINTS_WEIGHTS`, own-offense + opponent-defense → predicted
points, non-negative L2 fit, same methodology as `POWER_WEIGHTS` but *independently fit* —
the two weight vectors are not proportional, so two games with identical `delta` can still
have different predicted point totals). A further, separately-fit residual adjustment
(`WEATHER_ADJUSTMENT_WEIGHTS`: wind, precip, extreme cold) applies only when live forecast
data is available — restricted to teams confirmed permanently outdoor, within Open-Meteo's
~16-day reliable forecast window. "No weather data" always means "no adjustment," never a
silent calm-weather assumption.

### 1.5 Score distribution

Each side's full predicted-score curve (used on the ATS tab):
1. A bias-corrected Normal curve around the weather-adjusted points prediction (home/away
   bias + favorite/underdog bias, same trailing-shrinkage estimator style as HFA, fit fresh
   for points).
2. Reshaped to match **real historical NFL scoring frequency** since the 2015 PAT-distance
   rule change (verified live: owners moved the PAT to the 15-yard line in May 2015) —
   validated as a genuine improvement (+0.16 nats/game-side log-likelihood), not cosmetic.
3. Calibrated via bisection so the distribution's own implied win probability exactly
   matches `delta_win_prob()` — the two models (points and win probability) are forced to
   agree, even though they're fit independently.

Frozen at kickoff (same pattern as market odds elsewhere in this project) — a live or
just-finished game keeps showing its last pre-kickoff distribution.

### 1.6 Mid-game QB injury (descriptive only)

Detected via two independent, both-required signals: a real passer change mid-game outside
garbage time, confirmed by nflverse's own play text explicitly saying the departing passer
"was injured during the play." **Not folded into `delta`** — by definition unknowable before
kickoff, so there's nothing to correct for pregame. Surfaced as a Model Trends filter
(`home_qb_injured_ingame`/`away_qb_injured_ingame`) for historical analysis only.

### 1.7 Freeze-at-kickoff and historical reconstruction

A recurring pattern reused five separate times in this codebase: market odds, the ATS score
distribution, the Parlays tab's rating snapshot, each week's Power Rankings snapshot, and
mid-game QB injury detection (which is permanently frozen by construction, being
backward-looking). Each past week's Power Rankings/Parlays/Moneyline/ATS data is
reconstructed **exactly** (not approximated) using the rule "every completed prior season
plus this season's weeks strictly before week W" — identical whether computed today or
live back when week W actually started, since no future game has a result yet at that
point either way.

### 1.8 Pool-edge leverage system (adjacent, not part of the prediction model)

A separate decision layer on the Moneyline tab (ported from github.com/squid004/pickem-edge):
de-vigged market probability, pool-popularity estimate (scraped national pick% + a learned
per-team bias), and a deviation budget by season standing, producing FADE/CHALK/NO_PLAY
recommendations. This operates on market odds and the user's specific pool dynamics — it
does not feed into or draw from `delta`.

---

## 2. Development timeline

Each row is a change to **what's actually live**, with the backtested effect measured at
that point. "Hit rate" is always the model's own SU pick vs. actual winner, over every
graded decided game available at the time.

| Date | Change | Backtested effect | Shipped? |
|---|---|---|---|
| 2026-09-30 | 8-stat EWMA rating system (`research/edge_signal_test_v2`–`v7`) | Walk-forward: none of it beats the closing market spread (AUC ≤0.68 vs market's ~0.72); points scored alone (0.66 AUC) nearly as predictive as the whole system | Shipped as **descriptive** display, not an edge claim |
| 2026-09-30 | Non-negative joint logistic composite (`POWER_WEIGHTS`, v8) | Fixed the turnovers-committed sign-flip from naive per-stat fitting; AUC 0.6746 vs market 0.7224 | Shipped |
| 2026-10-05 | QB-injury blind spot identified (v14) | Model hit rate craters 63.6%→48.9% (z=-4.5) when the favorite's QB is Out/Doubtful; confirmed NOT a training-contamination problem (refit on clean-only games changed nothing) | Investigation only, nothing shipped yet |
| 2026-10-06 | QB health + skill-position health folded into `delta` (v16–v26) | Hit rate 62.87%→63.07% (+10 games); every walk-forward hit-rate test came back statistically flat — shipped anyway for the **calibration** effect (confidence reduced 29.0% of misses vs. 25.6% of hits, p=0.01) | **Shipped** (calibration play) |
| 2026-10-06 | Home-field advantage added (v28) | Hit rate 63.07%→63.11% (+12 vs. baseline); calibration z jumps 2.58→4.25 — the first correction to move raw hit-rate, because it corrects a structural bias rather than adding situational variance | **Shipped** |
| 2026-10-06 | Weather tested against win probability (v29) | Clean null: bad-weather hit rate (63.1%) statistically identical to good weather (63.7%), z=-0.24; optimal confidence-dampening found by grid search is exactly zero | Not shipped |
| 2026-10-07 | Rest/travel + divisional-game tested (v30) | Null: rest-mismatch coefficient doesn't survive calibration check; divisional games show no real calibration signal either | Not shipped |
| 2026-10-07 | Score-distribution pipeline (v31–v33) | +0.16 nats/game-side log-likelihood from the historical-reweighting step specifically (validated, not cosmetic) | **Shipped** (points/O-U dimension, separate from SU hit rate) |
| 2026-10-07 | Mid-game QB injury detection (v34) | Model's own favorite's QB hurt mid-game → miss rate 2.34% vs. 1.11% for an otherwise-identical hit (z=3.36, p=0.0008) | **Shipped as a filter**, not folded into `delta` (unknowable pregame) |
| 2026-10-07 | Momentum (signed win/loss streak) added (v35) | Isolated: 63.10%→63.22% (+6 games). Cumulative (vs. the pure 8-stat baseline): 62.88%→63.22% (+17 games total, 829 picks flipped — 423 improved, 406 regressed). Calibration z=4.35, p<0.0001 — survives *on top of* an already-recency-weighted rating | **Shipped** |
| 2026-10-07 | Trap game tested (v36) | Sign consistent (hypothesis-correct) at every threshold tried, but a proper chronological select/confirm split failed to replicate independently (select z=-3.08, confirm z=-1.77) | Not shipped — parked as "plausible, underpowered" |

**Net effect across the whole timeline**: the model's backtested SU hit rate moved from
**62.88%** (pure 8-stat composite, no corrections) to **63.22%** (current, with QB/skill
health + HFA + momentum all stacked) across 4,976 graded decided games — a **+0.34
percentage point / +17 game** lift from four additions, two of which (QB and skill health)
were deliberately shipped for calibration rather than accuracy. The model still does not
beat the closing market spread at picking straight-up winners; it was never designed to.

---

## 3. Tried but not implemented

Organized by why each one didn't ship — the reason matters for whether it's worth
revisiting.

### 3.1 Confirmed null (don't re-attempt without new reasoning)

- **Weather vs. win probability** (v29, 2026-10-06) — real, modest effect on *predicted
  points* (shipped, see §1.4), but genuinely nothing to add to win probability: hit rate in
  bad vs. good weather is statistically identical, and a grid search's own optimal
  confidence-dampening is exactly zero.
- **Rest/travel differential and divisional games vs. win probability** (v30, 2026-10-07) —
  the rest-mismatch coefficient doesn't survive the calibration check; divisional games show
  no real signal either.
- **Offensive line and defensive line unit health** (v27, 2026-10-06) — tested 3 separate
  ways (raw count, snap-weighted, proper select/confirm threshold scan). Sign even flips
  between select and confirm — pure noise, confirmed null all 3 times.
- **Linebacker unit health** — downgraded from an earlier "plausible, underpowered" read to
  confirmed-null under the stricter select/confirm test (select p=0.055, confirm p=0.899).
- **QB-injury blind spot as a training-contamination problem** (v14, 2026-10-05) — tested
  whether backup-QB games were biasing the joint weight fit itself. Refitting on clean-only
  games barely moved the weights (3rd-decimal differences); scoring the held-out backup-QB
  games with old vs. new weights was flat (63.5% vs. 63.2%). The blind spot isn't fixable by
  reweighting the existing 8 stats — none of them carry *any* current-week injury signal
  regardless of combination.
- **Single-pass opponent adjustment and full SRS/Ridge simultaneous solve** (v5/v6,
  2026-09-30) — both *worse* than the plain EWMA rating, likely because NFL scheduling
  already balances out and the market already prices in schedule strength.

### 3.2 Plausible but not confirmed (parked, worth revisiting with more data or a cleaner test)

- **Secondary (DB/S) unit health** (v27, 2026-10-06) — the one real survivor of the
  OL/DL/LB/secondary sweep: confirmed in the correct direction on a proper train-early/
  confirm-late split (p=0.082/0.094, consistent sign, a genuine p97 threshold). Fitted
  delta-equivalent weight is directionally right but falls just short of the calibration
  significance bar (z=1.61, p=0.106, vs. QB's p=0.01). **On hold pending a decision on
  buying Pro Football Focus (PFF) per-player grades** — a direct quality measure instead of
  the snap-share/touches proxies used so far, which could plausibly sharpen this into a real
  signal.
- **Trap game** (v36, 2026-10-07) — see timeline above. Consistent direction everywhere,
  but doesn't independently clear significance on a held-out half. Worth another look with
  more seasons of data, team power rating instead of market spread as the "tough opponent"
  measure, or a "letdown after a big emotional win" framing instead of schedule-lookahead.

### 3.3 Deferred (not tested — blocked on data, not found wanting)

- **"Steam" / line movement as a predictive signal** (2026-10-02) — the `Steam:` chip is
  currently cosmetic (flags `|Δspread| >= 1.5`, feeds nothing). Testing whether it predicts
  outcomes needs opening-vs-closing-line data; nflverse's `games.csv` only has closing lines,
  and this project's own `spread_history` table doesn't yet cover enough completed weeks for
  a real backtest.
- **Broader roster-depth signal via `weekly_rosters`** (flagged 2026-10-07, not built) —
  nflverse's Active/IR/PUP/PS status field, back to 2002, could extend the QB-only health
  work toward a general roster-depth signal without waiting on PFF — not yet attempted.
- **`stats_team`'s pre-aggregated EPA back to 1999** (flagged 2026-10-07, not built) — could
  extend every backtest's window by 8 seasons, but isn't garbage-time filtered the way this
  project's own PBP-derived EPA is, so it needs its own "does the longer unfiltered window
  actually backtest better" test before trusting it.
- **NextGen Stats / PFR advanced stats / FTN charting** (flagged 2026-10-07, not built) —
  genuinely new signal types (CPOE, pressure rate, motion/RPO), not more EPA, but shorter
  backtest windows (8–10 seasons for NGS/PFR, ~4 for FTN) mean less out-of-sample room than
  the 19-season window the current 8-stat system backtests on.
- **"Win this single week" leverage budget** (flagged 2026-09-30, not solved) — the
  pool-edge system's season-long deviation budget has no validated single-week equivalent;
  `pool_size / 10` was only ever a starting guess, never backtested.

---

## 4. Current production constants (reference)

```
FIRST_SEASON = 2007                FIRST_INJURY_SEASON = 2009
EWMA_ALPHA = 0.2                   SEASON_PRIOR_GAMES = 5
WP_LO, WP_HI = 0.05, 0.95          HFA_SHRINKAGE_K = 100

QB_QUALITY_GAP_WEIGHT  = -0.5903   (calibration only)
SKILL_EPA_OUT_WEIGHT   = -0.0362   (calibration only)
HFA_WEIGHT             = +0.9169   (moves hit rate)
MOMENTUM_WEIGHT        = +0.0262   (moves hit rate)

MIDGAME_MIN_BACKUP_ATTEMPTS = 3    (display/filter only, not in delta)
SCORE_HIST_FIRST_SEASON = 2015     MAX_TEAM_SCORE = 60
```

`POWER_WEIGHTS`, `POINTS_WEIGHTS`, and `WEATHER_ADJUSTMENT_WEIGHTS` are each their own
8-term (or 3-term) fitted dictionaries — see `nflhub/sources/team_ratings.py` directly for
current values rather than duplicating them here, since they're more likely to be re-fit
than the single-number weights above.
