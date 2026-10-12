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
| QB health (EPA) | `QB_QUALITY_GAP_WEIGHT = -0.8818` | Trailing EPA/dropback gap between a team's **current primary** (most starts in the trailing 5 games, ties broken toward the more recent start) and whoever's actually playing, **decayed** by `QB_GAP_DECAY_RATE = 0.4` for each additional straight game the same non-primary passer has started — see below | No (calibration only) |
| QB health (fantasy) | `QB_FANTASY_GAP_WEIGHT = -0.0429` | Same gap/decay structure as above, but sourced from FantasyPros' own weekly QB projections instead of trailing EPA — a forward-looking, weekly-refreshed measure, additive on top of the EPA gap, not a replacement — see below | No (calibration only) |
| Skill-position health | `SKILL_EPA_OUT_WEIGHT = -0.0363` | Summed trailing EPA of every Out/Doubtful RB/WR/TE/FB | No (calibration only) |
| Home-field advantage | `HFA_WEIGHT = 0.9169` | Trailing-5-season home win rate, shrunk toward all-time average (`HFA_SHRINKAGE_K = 100`) | **Yes** |
| Momentum | `MOMENTUM_WEIGHT = 0.0262` | Signed current win/loss streak, reset at season boundaries and ties | **Yes** |

```
delta = delta_raw
      + QB_QUALITY_GAP_WEIGHT     * (home_qb_gap  - away_qb_gap)
      + SKILL_EPA_OUT_WEIGHT      * (home_skill_out - away_skill_out)
      + HFA_WEIGHT                * home_field_logit_by_season[season]   -- 0 if neutral site
      + MOMENTUM_WEIGHT           * (home_streak  - away_streak)
```

**Neutral-site exception**: the HFA term (and the score-distribution pipeline's home/away
points bias, same assumption) is forced to exactly 0 for neutral-site games —
`is_neutral_site()` combines nflverse's own `location` field with a small manual fallback
list, since `location` alone is inconsistently applied (JAX's 2026 wk5 London "home" game
is tagged "Home" in nflverse's own data despite clearly having no true home-field edge;
Buffalo's old Toronto Series games show the same mislabeling in some seasons but not
others). Confirmed and fixed 2026-10-07, including three already-backfilled weeks that had
been silently getting undeserved HFA credit.

`delta_raw` is kept alongside `delta` everywhere for comparison. `delta_win_prob(delta)`
converts the composite into a calibrated win probability via a curve fit against real
backtested accuracy at each confidence level (QB-injury games excluded from that specific
fit, so an injury-widened delta doesn't inflate its own calibration).

**QB Out/Doubtful detection uses two different sources depending on purpose.** Every fit,
backtest, and historical reconstruction uses nflverse's weekly injury report exclusively
(`_qb_out_doubtful_by_week()`) — the only source with a historical archive to validate
against. The LIVE prediction for the current/upcoming week instead uses ESPN's live injury
feed (`_primary_qb_out_live()`, same feed the game-card's QB chip reads), because nflverse's
`report_status` field routinely sits blank until the Friday pregame report, days after ESPN
already has a real answer — confirmed against Caleb Williams' 2026 wk5 report (ESPN: Out,
Monday; nflverse: still blank, Wednesday). This is a **timeliness** swap for that one path
only, not an accuracy claim — same reasoning as using a separate live source for weather,
since nflverse doesn't forecast either. Matches on last name only (nflverse spells passer
names "F.Lastname", e.g. "C.Williams"; ESPN spells them "Firstname Lastname").

**Why QB/skill health don't move hit rate, mechanically**: the term is small relative to
everything else in `delta` (mean magnitude ≈0.10, 90th percentile ≈0.23) and only nonzero
for the ~1,200 graded games with a report-listed Out/Doubtful QB. It *can* flip a pick (62
of those 1,196 games flip, 5.2%), but the flips split almost evenly (32 improved, 30
regressed) — which is exactly why it's framed as a calibration correction (does `delta`
honestly reflect known health context?) rather than a pick-accuracy play. (Flip counts above
are from the season-cumulative primary definition in place before 2026-10-11; the trailing-
window definition below is expected to hold the same directional pattern but hasn't had its
own flip count re-measured.)

**Who counts as a team's "primary" starter, for this correction, uses a trailing 5-game
window** (most starts in that window, ties broken toward the more recent start) — not a
season-long cumulative-attempts count. Switched 2026-10-11 after the cumulative definition
mislabeled a live matchup: Atlanta's Details card blamed a backup-QB penalty on a passer who
hadn't actually started in two games, because the real current starter had been hurt to
*begin* the season, so the backup's early cumulative-attempts lead hadn't been caught up to
yet even though the depth chart clearly had already changed. Checked empirically — 3-4 game
windows overreact to a single-game blip and tested worse than cumulative; 5-6 games ties/
slightly beats it on walk-forward AUC while reacting within 2-3 games to a real change
instead of however long a cumulative crossover takes (research/edge_signal_test_v40_qb_
recency_decay.py).

**The gap also decays the longer a backup has already started** (`QB_GAP_DECAY_RATE = 0.4`),
instead of applying the same flat-size correction on his 1st start and his 8th. A bundled
version of this — decay plus a symmetric boost crediting the correction back when the
primary returns — was tested first and tied production's AUC at best, underperforming it
everywhere else in the sweep. Decomposing the bundle (research/edge_signal_test_v44) found it
was hiding a real effect: decay ALONE (no reinsertion) beats static, isolated to the games
where an absence is actually active (AUC +0.0020, logloss -0.0021, agreeing for the first
time in this investigation); reinsertion alone made things worse every time, confirming it
was the half dragging the bundle down (see §3.1 for the rejected reinsertion idea). The 0.4
rate itself was then swept as its own free parameter rather than assumed equal to EWMA_ALPHA
(research/edge_signal_test_v45) and validated with a chronological select/confirm split
(_v46): rate 0.4, selected using only 2014-2020, replicated on the held-out 2021-2025 confirm
set it never touched during selection (AUC +0.0018) — a 30-repeat random-split version of the
same test agreed in direction but was noisier (70% of repeats favorable). Shipped as a real
but not airtight finding, not a slam dunk.

**A second, independent QB-health gap is added on top, sourced from FantasyPros' own weekly
projections instead of trailing EPA.** Prompted by a broader question: could fantasy-football
projections feed the power model at all? First tried as a TEAM-TOTAL signal (summed
QB/RB/WR/TE projected points for offense, a team's own DST projected points-allowed for
defense) — confirmed null (research/edge_signal_test_v48): isolated to games with real
fantasy data, it looked like a big win (AUC 0.6623→0.7148) until checked against the market
spread directly — correlation **0.956**, and `delta + spread_line` alone reproduces almost
the exact same AUC (0.7278). The "signal" was FantasyPros' own projections leaking a noisy
copy of the Vegas line, which this project already knows beats the model and deliberately
keeps OUT of `delta` by design — not new information.

A narrower, WITHIN-team version escaped that trap: compare a team's own healthy-QB baseline
(trailing average of the primary's own weekly fantasy-point projections) against the actual
passer's current-week projection — same gap/decay structure as the EPA version, QB_GAP_
DECAY_RATE=0.4. Checked directly: this gap's correlation with -spread_line is only 0.187
(the EPA gap's own is 0.201) — a within-team comparison isn't restating who's favored the
way a team-total inherently does. Validated with the same chronological + random
select/confirm discipline as the decay rate above (research/edge_signal_test_v49/v50):
selecting on 2017-2023 and confirming on the held-out 2024-2025 games replicated (AUC
+0.0097), and a 30-repeat random split never once selected the EPA gap alone as best
(always the fantasy-inclusive version, 76.7% of repeats favorable on the specific AUC
comparison).

That first pass tested fantasy-gap-ADDED-to-EPA-gap against EPA-gap-alone, and fantasy won.
But a direct bootstrap of EPA+fantasy (both, summed) against fantasy-gap ALONE — the
comparison that actually matters once fantasy is already in the model — came back far
weaker: mean AUC edge only +0.0059, favorable in just 69.5% of resamples (research/
edge_signal_test_v50). Summing both barely outperforms fantasy alone, and FantasyPros is a
paid third-party API that can lapse, change terms, or go down mid-season; EPA/dropback has
zero dependency on it. So the shipped design does not sum the two gaps — it SWITCHES
between them per game-week. `QB_FANTASY_GAP_WEIGHT` (-0.0429, z=-3.24, much smaller in
magnitude than the EPA weight since fantasy points have a far larger raw scale than
EPA/dropback) applies whenever `fantasypros.qb_projections_by_season()` actually returned
data for that season/week; `QB_QUALITY_GAP_WEIGHT` (-0.8818, z=-3.46, refit jointly in
`_v51` alongside the fantasy weight) is the automatic fallback the instant that data is
missing — a lapsed/revoked API key, a FantasyPros outage, or a week not yet published.
`team_ratings.qb_fantasy_gap_data()` returns a `fantasy_available` flag per (season, week)
specifically to drive this switch; `matchup_callouts()` and `compute_backtest_scatter()`
both pick EPA-vs-fantasy per week from that flag, never adding them, and the live refresh
path logs a warning and records which source was active (`qb_health_source`, surfaced in
the power-model breakdown UI) so a fallback week is visible, not silent. Requires its own
FantasyPros API access in refresh() (`nflhub.sources.fantasypros.qb_projections_by_season`)
— a second, independent consumer of the same key already used by the fantasy-pool lineup
optimizer, with its own per-season cache (committed to the repo, same convention as every
other completed-season cache here) since this CI runner's disk doesn't persist between runs.

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
| 2026-10-07 | Momentum (signed win/loss streak) added (v35) | Isolated: 63.10%→63.22% (+6 games). Cumulative (vs. the pure 8-stat baseline): 62.88%→63.22% (+17 games, BEFORE the neutral-site fix below). Calibration z=4.35, p<0.0001 — survives *on top of* an already-recency-weighted rating | **Shipped** |
| 2026-10-07 | Trap game tested (v36) | Sign consistent (hypothesis-correct) at every threshold tried, but a proper chronological select/confirm split failed to replicate independently (select z=-3.08, confirm z=-1.77) | Not shipped — parked as "plausible, underpowered" |
| 2026-10-07 | Neutral-site HFA bug found and fixed | 71 REG-season games since 2007 (international series, old Buffalo Toronto Series) were wrongly getting home-field credit because nflverse's own `location` field mislabels some of them "Home." Cumulative lift from HFA+momentum+QB+skill revised down to 62.87%→63.07% (+10 games) once corrected. 3 already-backfilled weeks (1, 3, 4) self-healed in place via a new schema-version mechanism | **Bug fix** |
| 2026-10-08 | Live QB Out/Doubtful gate switched to ESPN's feed for the current/upcoming week | No backtested-effect number — this changes WHEN the existing QB-health correction fires this week, not what it's fit against. Found via a live case: Caleb Williams (CHI, 2026 wk5) was already "Out" on ESPN Monday but nflverse's `report_status` was still blank Wednesday, so the live card's QB chip and the model's QB-health term disagreed | **Shipped** — timeliness fix for the live prediction only; every backtest/fit/historical path stays on nflverse exclusively |
| 2026-10-08 | QB-health "replacement" estimate switched from league average to backup average (v37) | League-wide EPA/play (+0.035) is dominated by starters (83% of all attempts) — backup-only appearances actually average **−0.070**, a ~10x-larger, opposite-sign gap. The old estimate made a below-average-but-real starter (Caleb Williams, career EPA ≈ −0.003) look like an UPGRADE over his own backup once hurt. Walk-forward on the existing `QB_QUALITY_GAP_WEIGHT` (no refit needed — that weight was always fit against real hindsight backup performance, never this estimate): hit rate flat on the affected subset (374→375 of 591, 23 total flips, z=+0.21, same "statistically flat" pattern as every other QB/skill check) but the calibration gap (confidence reduced more on misses than hits) widened in the right direction (z 0.91→1.36, neither significant alone). Didn't regress anything; fixed a conceptually wrong estimator with real data behind the replacement | **Shipped** — research/edge_signal_test_v37_qb_backup_baseline.py |
| 2026-10-11 | Specific identified backup's own trailing EPA, instead of pooled backup-average (v38) | Name resolution via ESPN's QB depth chart worked correctly against a real case (CHI → Bagent). On the 287 historical games where a real backup would've had ≥50 attempts (66.6% coverage), using their own EPA scored *worse* (176/281) than the pool average (178/281); 12-flip paired check confirms noise, not a real effect (z=-0.58) | Not shipped — reverted after backtesting; see §3.1 |
| 2026-10-11 | Garbage-time-filtered points_off/points_def_allowed, composite reweighted (v39) | `points_off`/`points_def_allowed` are the #1 and #3 heaviest-weighted of all 8 stats (~41% combined) and the only two NOT garbage-time-filtered like every EPA stat already is. Rebuilt filtered (per-play score-state deltas, same WP band) and refit the whole composite from scratch: AUC 0.6746→0.6709, log loss and accuracy both worse too — garbage-time points aren't noise, a team that wins big enough to rack them up is still genuinely better on average | Not shipped — confirmed slightly worse, not better; see §3.1 |
| 2026-10-11 | QB "primary" definition switched from season-cumulative attempts to a trailing 5-game window (v40/v42); refit `QB_QUALITY_GAP_WEIGHT` | Found via a live mislabeling bug (Atlanta's Details card blamed the wrong team's backup-QB situation — the cumulative definition hadn't caught up to a real in-season starter change yet). Window sweep: 3-4 games overreacts to a single-game blip and tests worse than cumulative; 5-6 games ties/slightly beats it (AUC 0.6739 vs 0.6739, accuracy 62.08% vs 61.79%) while reacting within 2-3 games instead of however long a cumulative crossover takes. Refit `QB_QUALITY_GAP_WEIGHT` -0.5903→-0.5202 (still z=-3.25) | **Shipped** — research/edge_signal_test_v40_qb_recency_decay.py, _v42_qb_window5_static_refit.py |
| 2026-10-11 | Decaying the QB-health correction by how long a backup has started, with a symmetric reinsertion boost on the primary's return, bundled together (v40/v41) | Hypothesis: the team-level rolling rating (same EWMA half-life) independently catches up to a backup's true quality after ~4-5 starts, so a flat, never-decaying correction double-counts what the rolling rating already reflects; symmetric logic on return. Bundled, AUC ties production at best (0.6739) and underperforms everywhere else (0.6707-0.6720) | Not shipped as a bundle — decomposed below |
| 2026-10-11 | Decomposing decay from reinsertion (v44): decay alone beats static, reinsertion alone is what was dragging the bundle down | Isolated to the games where an absence is actually active: decay-only AUC 0.6836 vs static 0.6816 (+0.0020), logloss 0.6439 vs 0.6460 (-0.0021) — both metrics agreeing for the first time in this investigation. Reinsertion-only underperformed static in every view tested | Decay alone validated further below; reinsertion confirmed null, see §3.1 |
| 2026-10-11 | Decay rate swept as its own free parameter (not assumed = EWMA_ALPHA) and validated with chronological + random select/confirm splits (v45/v46/v47); refit `QB_QUALITY_GAP_WEIGHT` | Sweep showed a smooth inverted-U peaking near rate 0.4 in both AUC and logloss. Chronological split (select on 2014-2020, confirm on 2021-2025, never touched during selection): rate 0.4 replicated, AUC +0.0018. A 30-repeat random 50/50 split agreed in direction but was noisier (70% of repeats favorable, vs chronological's clean replication) — real effect, not airtight. Refit `QB_QUALITY_GAP_WEIGHT` -0.5202→-1.0556 (MORE significant, z=-4.23, was z=-3.25) | **Shipped** — research/edge_signal_test_v44 through _v47 |
| 2026-10-12 | Team-total fantasy-projection signal tested (summed QB/RB/WR/TE points vs. market spread) | Looked like a huge win (AUC 0.6623→0.7148) until checked against the market directly — correlation with -spread_line 0.956, and `delta + spread_line` alone reproduces almost the identical AUC (0.7278). Not new information, just a noisy proxy for the market the model already knows beats it | Not shipped — confirmed null; see §3.1 |
| 2026-10-12 | QB-health fantasy gap (within-team, FantasyPros projections vs. trailing EPA) added alongside the EPA gap; refit both weights | Escaped the market-proxy trap the team-total version fell into (corr with -spread_line only 0.187, vs the EPA gap's own 0.201). Chronological select/confirm (select 2017-2023, confirm 2024-2025): AUC +0.0097, replicated. 30-repeat random split never once picked the EPA gap alone as best (always fantasy-inclusive, 76.7% of repeats favorable). Refit `QB_QUALITY_GAP_WEIGHT` -1.0556→-0.8818 (still z=-3.46), new `QB_FANTASY_GAP_WEIGHT` -0.0429 (z=-3.24) | **Shipped** — research/edge_signal_test_v48 through _v51 |

**Net effect across the whole timeline**: the model's backtested SU hit rate moved from
**62.87%** (pure 8-stat composite, no corrections) to **63.09%** (current, with QB/skill
health + HFA + momentum all stacked, after the neutral-site HFA fix and the backup-baseline
fix above) across 4,977 graded decided games — a **+0.22 percentage point / +11 game** lift
from five additions, two of which (QB and skill health) were deliberately shipped for
calibration rather than accuracy. The model still does not beat the closing market spread
at picking straight-up winners; it was never designed to. (This headline figure predates the
2026-10-11 primary-window change above, which was measured on its own narrower walk-forward
subset — 2014-2025, 3,151 games — rather than folded into this full-history number.)

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
- **Specific identified backup's own trailing EPA, instead of the pooled backup-average**
  (v38, 2026-10-11) — the obvious next step after v38's own predecessor (v37, backup-average
  fix): resolve WHICH backup via ESPN's QB depth chart and use their own number when they
  have a real track record (tried ≥50 attempts). Name resolution itself worked correctly
  against a real live case (CHI → Tyson Bagent, 152 attempts, correctly skipping Caleb
  Williams still listed first). The backtest killed it anyway: on the 287 historical games
  where a real backup would have had enough sample (66.6% of affected games), using their
  own EPA scored *worse* (176/281, 62.63%) than the pooled average (178/281, 63.35%), and
  the 12-flip paired check confirms that's noise either direction (z=-0.58), not a real
  effect. A specific player's own EPA over 50-150 attempts is noisier than the SAME average
  pooled over 46,000 backup attempts — more specific isn't automatically more accurate at
  this sample size. Shrinking the specific number toward the pool average by sample size,
  instead of using it outright, might recover something — not attempted.
- **Garbage-time-filtered points_off/points_def_allowed, composite reweighted** (v39,
  2026-10-11) — prompted by noticing `points_off`/`points_def_allowed` are the #1 and #3
  heaviest-weighted of all 8 stats (~41% of total weight combined) *and* the only two not
  garbage-time-filtered the way every EPA stat already is (literal final box score, blowout
  garbage-time scoring included). Rebuilt both as a garbage-time-filtered stat (same WP_LO-
  WP_HI band, applied per-play via score-state deltas so it works for scoring plays without
  excluding non-run/pass types -- field goals, kick/punt return TDs, defensive scores) and
  refit the full 8-stat composite from scratch against it, same non-negative L2-regularized
  walk-forward methodology as the original fit. Made it slightly *worse*, not better: AUC
  0.6746 → 0.6709, log loss and accuracy both worse too, all three metrics agreeing. The
  refit weights barely moved (`points_off` still heaviest at 0.1520 vs 0.1587) -- garbage-
  time points aren't noise the model was being fooled by; a team that wins big enough to
  rack them up is still, on average, genuinely better, and filtering them out throws away
  real signal along with whatever noise came with it. Full removal (not just filtering) of
  the points terms, reweighting the remaining 6, hasn't been tested.
- **A symmetric reinsertion boost crediting the QB-health gap back when the primary
  returns, decaying over the following games instead of snapping to 0** (v40/v41/v44,
  2026-10-11) — the natural complement to the decay idea that DID ship (§1.3): if the
  correction shrinks while a backup starts because the team's own rolling rating is
  independently catching up, shouldn't it also credit something back once the real primary
  is back and the rolling rating is still lagging behind him? Bundled with decay, this tied
  production's AUC at best and underperformed everywhere else (after fixing a real bug where
  the first attempt only credited ONE week of recovery instead of decaying over several).
  DECOMPOSING decay from reinsertion (v44) confirmed it's reinsertion specifically that's the
  problem, not decay: reinsertion-only underperformed static in every view tested (full
  population AND isolated to active-absence games), while decay-only beat static in both.
  Confirmed null on its own, independent of decay's fate.
- **Team-total fantasy-projection signal** (v48, 2026-10-12) — summed QB/RB/WR/TE projected
  points for offense, a team's own DST projected points-allowed for defense, tested as a
  team-vs-team differential on top of `delta`. Looked like a large win in isolation (AUC
  0.6623→0.7148 on games with real fantasy data, consistent across every one of 5 seasons
  tested) until checked against the market spread directly: correlation with -spread_line is
  **0.956**, and `delta + spread_line` alone reproduces almost the exact same AUC (0.7278) --
  once the real market line is already in the model, every fantasy-derived candidate (margin,
  offense-only, defense-only) adds nothing (AUC deltas -0.0001 to -0.0013, pure noise). Not a
  new signal -- FantasyPros' own weekly point-allowed projections are themselves built partly
  from the game's own spread/total, so this was re-discovering "the market beats the model"
  through a noisy back door, something already known and deliberately kept OUT of `delta` by
  design. The lesson generalizes: any TEAM-TOTAL signal should be checked against the market
  directly before trusting an improvement over delta-alone as real. (A narrower, WITHIN-team
  version of the same data -- the QB-health fantasy gap -- escaped this trap; see §1.3.)

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

QB_QUALITY_GAP_WEIGHT  = -0.8818   (calibration only)
QB_FANTASY_GAP_WEIGHT  = -0.0429   (calibration only, additive on top of the EPA gap)
SKILL_EPA_OUT_WEIGHT   = -0.0363   (calibration only)
PRIMARY_WINDOW         = 5 games   (trailing-window plurality, ties -> more recent start)
QB_GAP_DECAY_RATE      = 0.4       (shrinks the gap each straight game a backup keeps starting)
HFA_WEIGHT             = +0.9169   (moves hit rate)
MOMENTUM_WEIGHT        = +0.0262   (moves hit rate)

MIDGAME_MIN_BACKUP_ATTEMPTS = 3    (display/filter only, not in delta)
SCORE_HIST_FIRST_SEASON = 2015     MAX_TEAM_SCORE = 60
```

`POWER_WEIGHTS`, `POINTS_WEIGHTS`, and `WEATHER_ADJUSTMENT_WEIGHTS` are each their own
8-term (or 3-term) fitted dictionaries — see `nflhub/sources/team_ratings.py` directly for
current values rather than duplicating them here, since they're more likely to be re-fit
than the single-number weights above.
