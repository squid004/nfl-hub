# NFL Hub prediction model — methodology, timeline, and null results

Written 2026-10-07. Covers `nflhub/sources/team_ratings.py`, the module that produces every
prediction surfaced on the Moneyline Pick'em, Against the Spread, Power Rankings, and Model
Trends tabs. Every number below was pulled from production constants or the live
`historical_power_rankings` backtest dataset (4,976–4,991 graded games, 2007–2025) at the
time of writing, not recalled from memory — it will drift as the model keeps evolving;
treat this as a snapshot, not a living spec.

## 1. What the current model does

### 1.1 Foundation: per-team ratings

Every prediction starts from the same 9 rolling team-level stats plus an Elo rating (below), computed per team per week:

| Stat | What it measures |
|---|---|
| `rush_off_epa` / `pass_off_epa` | EPA/play on offense, rush and pass separately |
| `rush_def_epa_allowed` / `pass_def_epa_allowed` | EPA/play allowed on defense |
| `points_off` / `points_def_allowed` | Points scored / allowed per game |
| `turnovers_off` / `turnovers_def_forced` | Turnovers committed / forced per game |
| `pen_presnap_yds` | Yards lost per game to pre-snap penalties (false start, delay of game, offside, encroachment, ...); fewer is better. Added 2026-10-13 |
| `elo` | FiveThirtyEight-style Elo (K = 20, home field 48 inside the update only, margin-of-victory multiplier, one-third regression to 1505 each new season, neutral sites no home field), updated game by game from 2007 (`elo_ratings()`). Not an EWMA of a raw stat, so it sits outside the rolling-stat list. The composite uses the plain home-minus-away rating difference (delta adds its own home-field term). Added 2026-10-16 |

- **Garbage-time excluded** from the four EPA stats and both turnover stats: any play with
  win probability outside `[0.05, 0.95]` is dropped before averaging (`WP_LO`/`WP_HI`).
  Points and pre-snap penalty yards are **not** filtered this way — a final score is the
  whole-game outcome, and a penalty is a discipline event whose cost doesn't depend on game
  state; filtering either would fight the stat's own meaning.
- **EWMA + season-prior fade**: each stat is an exponentially-weighted moving average
  (`EWMA_ALPHA = 0.08`, ~8-game half-life, **bias-corrected**: a geometric-weighted average of
  the season's games so far, not seeded with game 1) within a season. A new season starts from
  last season's own ending rating, which fades out **linearly to exactly zero** over
  `SEASON_PRIOR_GAMES = 8` games — a decade-old season has zero path into this year's rating
  once last season is behind it, not an ever-shrinking nonzero weight.
- **Data window**: `FIRST_SEASON = 2007` (a project choice, not nflverse's actual floor —
  the underlying play-by-play goes back to 1999; see `docs/nflverse-data-catalog.md`).
- **Display scale**: the 4 EPA stats are shown 0–100, anchored to the best/worst value *ever
  recorded* across the full dataset, so a weak season's best team doesn't read as inflated.
- **EPA distribution tiers** (Power Rankings tab, display only — not folded into the
  composite): every team on each 0–100 EPA stat falls into one of five groups — Elite/Weak
  when a real gap (not just rank) cuts a group off from the rest, however small; otherwise
  Above/Below average if meaningfully off the mean, else just Average. This catches a
  cluster of similarly-bad teams that's nonetheless well below average as a real group,
  instead of it disappearing into an undifferentiated middle.

### 1.2 The composite Power Score

```
POWER_WEIGHTS = {
  rush_off_epa: 0.0765, pass_off_epa: 0.1609,
  rush_def_epa_allowed: 0.0590, pass_def_epa_allowed: 0.0000,
  points_off: 0.1913, points_def_allowed: 0.1418,
  turnovers_off: 0.0000, turnovers_def_forced: 0.0000,
  pen_presnap_yds: 0.0401, elo: 0.2931,
}
```

One joint L2-regularized logistic regression (`L2 = 0.03`, chosen by walk-forward AUC over a
grid of 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3; the original grid stopped at 0.3, which turned
out to sit above the optimum, see the 2026-10-13 timeline row) across all 10 stats (9 rolling stats + Elo) at once, each first oriented "higher = always better,"
coefficients constrained `>= 0`. The non-negativity constraint fixes a real failure mode: an
unconstrained fit sign-flipped `turnovers_off` (committing more turnovers came out
*positively* weighted) because the 8 stats overlap heavily — non-negative coefficients can
only shrink a redundant stat toward zero, never flip its sign. `turnovers_def_forced`
optimizes to exactly 0.0 once the other 8 are known (`turnovers_off` also falls to 0.0 at the
lower L2). **Walk-forward AUC 0.6983 (test seasons 2011+; 0.6941 before Elo, 0.6787 before the 2026-10-15 speed retune) vs. 0.7224
for the closing market spread alone — this does not beat the market at picking winners.**
Shown as descriptive context (Power Rankings tab, college-poll-style display), not a
betting edge.

**Historical Power Rankings tab** uses the same formula but each team's simple full-season
average, not the live tab's EWMA rolling rating — every team-season is already fully known,
so there's no "as of this week" to track. Z-scored against every team-season at once so
different eras compare directly.

### 1.3 `delta`: the per-matchup prediction

For one matchup, each stat's home-minus-away difference is standardized (z-scored against
the full backtest population), weighted by `POWER_WEIGHTS`, and summed to `delta_raw`. Four
corrections are then added on top — all additive, all fit **once, offline**, on the full
graded dataset, none of them touch `POWER_WEIGHTS` or the 9-stat composite itself:

| Correction | Weight | What it captures | Hit-rate mover? |
|---|---|---|---|
| QB health (EPA) | `QB_QUALITY_GAP_WEIGHT = -1.2268` | Trailing EPA/dropback gap between a team's **current primary** (most starts in the trailing 5 games, ties broken toward the more recent start) and whoever's actually playing, **decayed** by `QB_GAP_DECAY_RATE = 0.4` for each additional straight game the same non-primary passer has started. Active only as the **fallback** when FantasyPros has no data for that week — never summed with the fantasy gap below — see below | No (calibration only) |
| QB health (fantasy) | `QB_FANTASY_GAP_WEIGHT = -0.0619` | Same gap/decay structure as above, but sourced from FantasyPros' own weekly QB projections instead of trailing EPA. **Switches in** per game-week whenever that data is available, replacing the EPA gap entirely for that week rather than adding to it — see below | No (calibration only) |
| Skill-position health | `SKILL_EPA_OUT_WEIGHT = -0.0747` | Summed trailing EPA of every Out/Doubtful RB/WR/TE/FB | No (calibration only) |
| Home-field advantage | `HFA_WEIGHT = 1.0366` | Trailing-5-season home win rate, shrunk toward all-time average (`HFA_SHRINKAGE_K = 100`) | **Yes** |

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
between them per game-week. `QB_FANTASY_GAP_WEIGHT` (-0.0619 after the 2026-10-23 momentum-removal refit, z=-3.1, much smaller in
magnitude than the EPA weight since fantasy points have a far larger raw scale than
EPA/dropback) applies whenever `fantasypros.qb_projections_by_season()` actually returned
data for that season/week; `QB_QUALITY_GAP_WEIGHT` (-1.2268 after the 2026-10-23 momentum-removal refit, z=-3.2, first fit jointly in
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

**Prospective log (added 2026-10-16).** The freeze patterns above are rebuilt when the model is
refit, so none of them is a true record of what the model predicted at the time. `prospective_log`
(`_log_prospective()` in `team_ratings.py`, kv key `prospective_log`) is the exception: every game's
pre-kickoff `delta`, calibrated P(home win) and the market's spread/moneylines are written the day
before its gameday and never changed or rebuilt afterward. Model Trends scores it (hit rate, log
loss, Brier) against the de-vigged moneyline. Weeks 1-5 of 2026 predate it and can't be scored
this way. 2026 is treated as the untouched holdout from here on.

### 1.8 Pool optimizer (adjacent, not part of the prediction model)

A separate decision layer on the Moneyline tab that picks the card maximizing the **simulated chance of winning the week** in
this specific pool (`nflhub/sources/pool_mc.py`, inputs built by `pool_recommend.py`). It uses market odds
and the pool's own behavior only; it does not feed into or draw from `delta`. It replaced the earlier
FADE / CHALK / NO_PLAY tags (ported from github.com/squid004/pickem-edge: de-vigged probability,
national pick % + learned per-team bias, a hand-set deviation budget by standing), which are no longer
shown (their per-game log is still written for comparison).

**Pool rules.** $180 entry = $10 per entrant per week, 27 entrants → a $270 weekly pot. 80% ($216) goes to
the week's winner (most correct picks); ties go to the closest Monday-night total guess, and entrants still
tied after that split the prize. The other 20% ($54/week, $972 over 18 weeks) pays the season's top three
in total wins 60 / 30 / 10 ($583.20 / $291.60 / $97.20). The optimizer's metric is win percentage, not dollars.

**Model.** 20,000 simulated weeks. Each unfinished game is a Bernoulli draw from the de-vigged closing
moneyline. Each opponent takes the favorite with probability
`sigmoid(0.02 + 0.33·logit(market P(fav)) + 0.76·logit(national pick %) + offset_j)`, where `offset_j` is
that entrant's persistent chalk tendency shrunk from their own past picks. v65 (2026 weeks 1-5) found the
field takes the favorite 78% of the time vs. 65% market probability, and entrants differ persistently
(chalk-residual correlation +0.46 between weeks 1-2 and 3-5; observed spread 0.076 vs. 0.045 if everyone
picked independently), so the plan's "independent opponents" model was replaced. Each opponent's tiebreak
guess follows their own habit (used only once Sunday is done and the user is within one pick of the leader -- leading, tied or
one back -- otherwise the pooled guess model applies and no contenders are computed): with probability p_fixed they repeat their favorite number (their modal past guess), otherwise
Normal(posted total + their typical offset, 4.5), both shrunk toward the pooled values (v67); a guess CBS has revealed after
Monday's kickoff replaces the model. The actual total ~ Normal(posted total, 13.3). In that same situation the Moneyline tab lists the
"Contenders" (entrants with a real chance at the top score, how often each would be tied with you there, and their tiebreak
habit). **Freeze at lock:** a game's displayed pick and impact are frozen from the last run before it locked (picks lock by
game day), and once the game is under way the pick is coloured green/red like ELWAY, History and Power Model.
**Objective** = P(win the week) (the weekly prize is 80% of the pot). The season pot is deliberately NOT part of it: the first version added the marginal value of one more season win (a rest-of-season simulation, about $7 a win), and it was removed 2026-10-17 so the optimizer is purely about the week being played. The card is found by greedy single flips from all-favorites, jointly with the
tiebreaker guess. The Moneyline tab shows each game's pick and its **impact rating** (win chance minus the win chance with only that pick switched, in percentage points; every other pick stays optimized) versus the other
side with the rest of the card fixed.

**Sequential use.** Picks lock by game day. Once a day's games have kicked off, the user's saved pick and
the game's result are fixed and only the open games and the tiebreaker are re-optimized (recomputed every
45 minutes; stops when the week is final). The field's revealed picks and the user's own official picks
come from the merged live-picks store (`pool_live.py`; sources and the CBS-fetch plan in `docs/pool-live-picks.md`;
agreeing sources are deduplicated and disagreements are logged), currently fed by an optional **"Week N live" tab** in the pool sheet, written by a scraper/Cowork run after each game day's
first kickoff: same row layout as the final "Week N" tab (name, weekly, season, then pick cells), but only the picks
visible at that moment are filled, plus one row whose first cell is `Captured` and whose second cell is an ISO
timestamp. The CBS pool is the official record, so the user's locked picks are read from it (the in-app pick is only an
unverified, labelled fallback when no live tab exists, and a locked game with no pick anywhere counts as no pick). Each
game is flagged when the app and CBS disagree, when CBS has a pick the app doesn't, or when CBS shows no pick after
lock. Without live picks, the field's locked picks are predicted. ntfy reminders (`pool_notify.py`) fire when a lock window opens
(every earlier game day final, lock within 40 hours) and once more as a last call within 3 hours if picks still differ
from the recommendation or the app and CBS disagree; they list the recommended changes against the user's current picks
and the tiebreaker guess, and never submit anything. The final "Week N" tab, once it exists, supersedes it.

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
| 2026-10-13 | Pre-snap penalty yards added as the 9th composite stat; weights refit (v52-v54) | Judged against the market-free model (the market spread is never an input to `delta`, so any replicated signal is a real improvement, even if the market already prices it). Walk-forward 2011-2026: AUC 0.6749→0.6763, log loss 0.6464→0.6457; better log loss in 14 of 16 seasons, weight positive in 16 of 16 folds, same direction in both chronological halves (select +0.0016 / confirm +0.0007 AUC), bootstrap log-loss p=0.002. QB/skill/HFA/momentum/calibration constants re-checked under the new composite (<2% shift) and left unchanged | **Shipped** — research/edge_signal_test_v52 through _v54 |
| 2026-10-13 | Re-ran every v52 candidate market-free, one at a time against the 9-stat baseline (v55/v56) | v52 had screened partly on correlation with the closing spread, the wrong gate for a market-free model. Pre-registered SHIP rule (both halves positive on AUC and log loss, bootstrap p<=0.01, weight positive in >=90% of folds, better in >=60% of seasons): no candidate qualified. Explosive plays, penalty yards total/drawn, FG over expected, return and kicking EPA, defensive drive-score rate: null or negative. `drive_score_off` had the largest log-loss gain but it shrank to +0.0003 at low L2 and the confirm half turned negative: an artifact of the composite being over-regularized, not a signal. `st_kp` (kickoff/punt EPA) is consistently positive but tiny, and fades to noise once the baseline is re-judged at lower L2 (v58) | Nothing shipped; see §3.1 and §3.2 |
| 2026-10-13 | Composite L2 lowered 0.3 → 0.03; every delta-scale constant and the delta→probability calibration refit; momentum wired into upcoming/backfilled games (v57/v58) | The original L2 grid bottomed out at 0.3, so the optimum sat off its edge (v55/v56 hinted at it: the baseline improved at L2=0.01). Extended the grid on the production `compute_ratings` game_log, 9 stats, walk-forward 2011-2026. Pre-registered ADOPT rule (vs. 0.3: AUC and log loss better in BOTH halves, bootstrap p<=0.01, better in >=60% of seasons): L2=0.03 AUC 0.6763→0.6787, log loss 0.6457→0.6390, select +0.0033 / confirm +0.0021 AUC, p≈0, better in 75% of seasons; 0–0.01 indistinguishable, 0.03 chosen as the largest within 0.0005 AUC of the best. `pen_presnap_yds` still earns its place at 0.03 (+0.0012 AUC, +0.0008 log loss, p=0.018). Weights roughly double in scale (sum 0.68→0.97), so delta is ~1.45x wider (`delta_raw` std 0.44→0.66) and every delta-equivalent constant was refit with the same joint recipe at identical z-scores: QB -0.8818→-1.3268, fantasy gap -0.0429→-0.0651, skill -0.0363→-0.0608, HFA 0.9169→1.0407, momentum 0.0262→0.0256; calibration 0.0484/1.4497→-0.0841/1.0852. Backtest favorite hit rate 63.5%; reliability by |delta| bin within ~3 pts everywhere. Separately found that `team_streak_by_week` only has entries for games already played, so `MOMENTUM_WEIGHT` contributed 0 to every live and backfilled-week matchup; now computed via `upcoming_streaks()`. Stored weekly snapshots (Power Scores, parlay tiers, Moneyline/ATS matchups) and the frozen per-game score distributions were rebuilt under the new model (`RATINGS_SCHEMA_VERSION`, `MATCHUPS_SCHEMA_VERSION` 5) | **Shipped** — research/edge_signal_test_v57, v58 |
| 2026-10-15 | Rating update speed retuned: `EWMA_ALPHA` 0.2 → 0.08 with a bias-corrected EWMA, `SEASON_PRIOR_GAMES` 5 → 8; composite weights, QB/fantasy/skill/HFA/momentum constants and delta calibration refit (v61/v62) | `EWMA_ALPHA = 0.2` was "reused convention, not independently tuned". A first 24-cell sweep with the classic first-game-seeded EWMA found slower speeds monotonically WORSE (alpha 0.06 AUC 0.626): a slow EWMA seeded with game 1 stays anchored to it. Adding a bias-corrected EWMA (geometric-weighted average of the season's games so far) as a second estimator (48 cells; amendment documented in the v61 docstring, rule not loosened) reversed that. Pre-registered ADOPT rule on the production `compute_ratings` game_log, 9 stats, weights refit per walk-forward fold at L2=0.03, test seasons 2011-2026 (3,984 games), candidate chosen on <=2018 only: alpha 0.08 / prior 8 / bias-corrected vs. 0.2 / 5 / classic: AUC 0.6787→0.6941, log loss 0.6390→0.6304, select +0.0162 / confirm +0.0150 AUC, +0.0089 / +0.0083 log loss, bootstrap p<0.001, better in 13 of 16 seasons. The whole alpha 0.06-0.12, prior 8-12 bias-corrected region is a plateau; v62 found no gain from separate speeds for EPA / points / turnovers+penalties groups (dAUC +0.0002, p=0.19) or from a different L2 (0.003/0.01 within noise, 0.1/0.3 worse). The points-prediction model was re-checked on the new ratings (MAE 7.57, margin AUC 0.695) and left unchanged. Refit delta constants: QB -1.3268→-1.2518, fantasy gap -0.0651→-0.0598, skill -0.0608→-0.0726, HFA 1.0407→1.0319, momentum 0.0256→0.0164 (now z=1.6, below the usual bar; kept, same recipe); calibration -0.0841/1.0852→0.0336/0.9921 (the favorite no longer gets under 50% at delta 0). delta sd 0.66→0.78. Full-delta hit rate over all decided games 63.5%→65.3% | **Shipped** — research/edge_signal_test_v61, v62 (+ `_ship_refit`, `_points_model_refit`) |
| 2026-10-16 | Elo added as a 10th composite stat; weights, delta constants and calibration refit (v63) | The cowork review found Elo alone scores AUC 0.692 and carries most of what a slower update recovers (corr 0.88 with the composite). Tested on top of the 2026-10-15 composite under the pre-registered ADOPT rule (production `compute_ratings` game_log, weights refit per walk-forward fold at L2=0.03, test seasons 2011-2026, 3,984 games): 9 stats → 9 stats + Elo AUC 0.6941→0.6983, log loss 0.6304→0.6270, select +0.0028 / confirm +0.0058 AUC, +0.0040 / +0.0029 log loss, bootstrap p=0.001, better in 13 of 16 seasons; Elo's weight is positive in all 16 folds (mean 0.30 standardized). Elo alone ties the 9-stat composite (AUC 0.6925, not adopted as a replacement). Elo is the published 538 formulation, no constants tuned; a standalone 1999-warmed Elo gives the same result as production's 2007-start Elo. Elo's weight comes mostly out of the points and EPA weights; `pass_def_epa_allowed` falls to 0. Refit delta constants: QB -1.2518→-1.2399, fantasy gap -0.0598→-0.0618, skill -0.0726→-0.0767, HFA 1.0319→1.0079, momentum 0.0164→0.0071 (z=0.7 now, i.e. noise; Elo's margin-aware results already carry the streak signal; kept pending a drop test), calibration 0.0336/0.9921→0.0017/1.0505. Full-history hit rate 65.3%→65.6% (in-sample) | **Shipped** — research/edge_signal_test_v63 (+ `_ship_refit`) |
| 2026-10-16 | Validation hygiene: honest walk-forward backtest, prospective log, P(favorite) floor (v64) | Measured, not a candidate. The backtest scatter's `delta` uses constants fit once on ALL data (z-score stats, composite weights, QB/fantasy/skill/HFA/momentum constants, calibration), so its hit rate is in-sample. v64 rebuilt the whole pipeline walk-forward one piece at a time (test seasons 2011-2026, 3,984 games; training = earlier seasons only): V0 production in-sample hit 65.3% / log loss 0.6209 → weights walk-forward 65.1% → + z-score stats from training seasons only 65.1% → + QB/fantasy/skill/HFA/momentum refit walk-forward **63.5%** (log loss 0.6421, AUC 0.7066→0.6891) → + calibration walk-forward 63.5% (log loss 0.6299). Almost all of the drop is the corrections: with only 2-5 training seasons the early walk-forward home-field constant is unstable (4.4 in 2011 vs ~1.0 since 2017). Since 2017 (8+ training seasons) honest vs. in-sample is **64.4% vs. 65.5%**. Walk-forward calibration by |delta| bin is within ±4 points everywhere (largest miss: 1.0-1.5, predicted 73.2% vs. actual 69.4%). Shipped: `delta_wf` / `p_home_wf` / `outcome_wf` on every scatter row from 2011, a summary on Model Trends (the season hit-rate chart now scores Power Rankings with the walk-forward outcome), P(favorite) floored at 50% (moot at the current 0.0017 intercept, but it was 47.9% at delta 0 before the Elo refit; 8% of walk-forward games would otherwise give the favorite under 50%, floor changes log loss 0.6303→0.6299), and the append-only prospective log with a scorecard. Not changed: the live constants stay all-data fits (a walk-forward constant for the upcoming week would just be the all-data fit from the prior season) | **Shipped** — research/edge_signal_test_v64 |
| 2026-10-16 | Pool optimizer replaces the FADE / CHALK / NO_PLAY tags (v65/v66) | The hand-set deviation budget (`pool_size / 10`-style heuristics) was never validated, and the objective is dollars, not correct picks. Built a market-only Monte Carlo (`pool_mc.py`) of the week under the pool's payout rules ($10/entrant/week, 80% weekly winner, 20% to the season top three 60/30/10; ties → closest Monday-night guess, then split). Calibrated on the pool sheet's weeks 1-5 (v65): the field takes the favorite 78% of the time vs. 65% market (logit slope 0.33 on market + 0.76 on national pick %), and entrants differ persistently (+0.46 correlation), so each opponent gets a shrunk chalk offset. Tiebreak rule verified against all five real weekly winners. Backtest weeks 1-5 (v66, opponent model fit leave-one-week-out; the model's own P(win week), not a significance test): what was played 5.2%, the app's old card 4.0%, always-favorite 3.4%, optimizer 6.9%. Realized: every card won 0 of 5 weeks (10.0 correct a week for the optimizer, all-favorites and the old card; 9.6 for the card played). Sequential, 11 post-game-day windows: re-optimizing would have changed an open pick or the tiebreaker in 8; the only one that changed a result was week 4, where a tiebreak of 52 instead of 44 would have won the four-way tie at 12. Simulated opponent scores ran slightly above real ones (9.6 vs. 9.1), consistent with five weeks of luck | **Shipped** — research/edge_signal_test_v65, v66 |
| 2026-10-17 | Pool optimizer scoped to the week only (season-pot term removed) | Requested: optimize the week being played, separately from whole-year scoring. The objective is now P(win the week); dollars shown are the weekly-prize expectation (0.8 × P × $270) and each game's value is the change in that. Removing the term changed nothing in the upset picks the engine chose in weeks 1-5 (17 upset picks, favorite's market win chance 50.4%-65.7%, pool taking the favorite 58%-89%), but the realized-and-modelled backtest numbers moved (v66, weeks 1-5, the model's own P(win week), not a significance test): optimizer 7.3% (was 6.9%), played card 5.2%, old FADE/CHALK card 4.0%, always-favorite 3.4%; realized correct picks per week 9.4 for the optimizer (was 10.0), 10.0 for all favorites and the old card, 9.6 for the card played; every card won 0 of 5 weeks. Re-optimizing after each game day still changes an open pick or the tiebreaker in 8 of 11 windows | **Shipped** — research/edge_signal_test_v66 (first-version output kept in output/v66_run_with_season_term.txt) |
| 2026-10-17 | Pool optimizer metric changed from dollars to simulated win percentage | Requested. Each game's rating is now its impact: the recommended card's simulated weekly win chance minus the win chance with just that game on the other side (all other picks re-optimized), in percentage points. No dollar figures are shown; the picks themselves are unchanged | **Shipped** |
| 2026-10-20 | Per-entrant Monday-night tiebreak habits in the pool optimizer, a Contenders panel, and a tiebreaker box on the last game card (v67) | The first optimizer drew every opponent's tiebreak guess from one pooled distribution. The pool sheet's weeks 1-5 show real personal habits: 17% of all guesses are exactly 42 and 13% exactly 45, one entrant guessed 42 in all five weeks, another 45 in all five, and several repeat a number 4 of 5 weeks. Per-entrant model (repeat a favorite number with a shrunk probability, else a shrunk personal offset from the posted total) vs. the pooled model on held-out guesses: +0.35 nats per guess walk-forward (weeks 3-5, 77 guesses) and +0.33 leave-one-week-out (129 guesses). On the weeks-4/5 'after Sunday' states the habits change the recommended tiebreak (week 4: 52 → 41; week 5 unchanged at 62) because the always-42 entrants sit in the tie pack. Shipped: the habit model in the simulation, a Contenders table (correct so far, chance at the top, chance tied with you, tiebreak habit; names shortened), and a Tiebreaker: total points box on the Monday card showing the optimizer's guess, the posted total and your CBS guess when known | **Shipped** — research/edge_signal_test_v67 |
| 2026-10-23 | Momentum (win/loss streak) removed from delta; other delta constants and calibration refit (v71/v72) | Drop test (v71, fully walk-forward, 3,999 games 2011-2026, v57 rule): with vs. without momentum dLL -0.0001, dAUC -0.0005, p=0.83, better in 56% of seasons: it adds nothing once Elo is in the composite (weight already z=0.7). The same ablation found only the two QB gaps together earn their place under the rule (dLL +0.0027, p=0.001); nothing else removed. Refit without the streak term: QB -1.2399→-1.2268, fantasy gap -0.0618→-0.0619, skill -0.0767→-0.0747, HFA 1.0079→1.0366, calibration 0.0017/1.0505→0.0177/1.0481; composite weights unchanged. Full-history hit rate 65.6%→65.7% (in-sample). `MATCHUPS_SCHEMA_VERSION` 8 so stored weekly matchups rebuild; the Momentum row is hidden on the Moneyline card breakdown | **Shipped** — research/edge_signal_test_v71, v72 |
| 2026-10-25 | Model Trends: pool optimizer vs. the pool, point-in-time (`pool_backtest.py`) | Per completed week, the optimizer's static pre-week card is rebuilt using only information available then and its correct picks are charted against the week winner's and the user's. Entrant chalk bias comes from EARLIER weeks only and is not used at all before week 4 (three earlier weeks of picks), replacing v66's leave-one-week-out opponent fit, which used future weeks; closing moneylines and the week's scraped national pick % are inputs; the pooled pick-share constants (`PICK_A/B/C`, fit on weeks 1-5) are the one remaining leak for weeks 1-5. Weeks 1-6: optimizer 9, 9, 11, 8, 9, 9 correct vs. winners 13, 12, 11, 12, 11, 10 and the user 8, 8, 10, 12, 10, 8 (optimizer mean 9.2, user 9.3, pool 8.8). It maximizes P(win the week), not correct picks | **Shipped** (display; each week frozen in kv `pool_backtest` once its final tab exists) |

**Net effect across the whole timeline**: the model's backtested SU hit rate moved from
**62.87%** (pure 8-stat composite, no corrections) to **63.09%** (current, with QB/skill
health + HFA + momentum all stacked, after the neutral-site HFA fix and the backup-baseline
fix above) across 4,977 graded decided games — a **+0.22 percentage point / +11 game** lift
from five additions, two of which (QB and skill health) were deliberately shipped for
calibration rather than accuracy. The model still does not beat the closing market spread
at picking straight-up winners; it was never designed to. (This headline figure predates the
2026-10-11 primary-window change above, which was measured on its own narrower walk-forward
subset — 2014-2025, 3,151 games — rather than folded into this full-history number.)
After the 2026-10-15 speed retune the same full-history figure (QB/skill health + HFA + momentum
stacked) is **65.3%** (3,258 of 4,991 decided games; the pre-adjustment composite alone is 64.2%),
and **65.6%** after Elo was added on 2026-10-16. **Those are all in-sample.** The honest walk-forward figure (every fitted piece trained on earlier seasons only, v64) is **63.5%** over 2011-2026 and **64.4%** over 2017-2026 against 65.5% in-sample for the same recent games; the Model Trends tab shows both.
Both include the in-sample all-data weights; the walk-forward figures in the timeline row are
the honest out-of-sample ones.

---

## 3. Tried but not implemented

Organized by why each one didn't ship — the reason matters for whether it's worth
revisiting.

### 3.1 Confirmed null (don't re-attempt without new reasoning)

- **QB weight refit against the ATS residual** (v68, 2026-10-22; cowork B6.1) — the market
  residual (margin - closing spread) correlates with the QB gap even after delta uses it: the
  fitted shift is consistently toward a *larger* QB weight (about 0.7 delta units on the EPA gap,
  1.4-2.7 on QB-flag games only), i.e. the market prices a QB change more heavily than the
  win/loss fit does. Shifting the weights that way, walk-forward 2011-2026 vs. the fully
  walk-forward win/loss-fit baseline: dLogLoss +0.0006 to +0.0015, dAUC +0.0006 to +0.0009,
  better in both halves and in 69-75% of seasons — but paired-bootstrap P(dLL <= 0) = 0.022 to
  0.042, so **fails the pre-registered p <= 0.01 bar** (all three variants). Model-side ATS on
  QB-flag games stays 44-48%, unchanged. Real direction, too small to ship; B6.2 (QB-adjusted
  team rating prototype) not attempted.
- **Empirical-Bayes shrinkage of the actual passer's trailing EPA toward `backup_avg_epa`**
  (v69, 2026-10-22; the item v38 left untried) — K in {25, 50, 100, 200, 400} (backup only) and
  K=100 on both sides, same walk-forward pipeline. All six improve both metrics by a hair
  (dLL +0.0002 to +0.0005, dAUC +0.0003 to +0.0006) but P(dLL <= 0) is 0.10 to 0.18 and only
  K=100 reaches 62% of seasons. Not adopted. The `QB_EPA_SHRINK_K` / `QB_EPA_SHRINK_BOTH`
  constants (default off) stay in `team_ratings.py` only so the script can run.

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
  instead of using it outright, might recover something — tried as v69 (see below): not enough.
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
- **Explosive-play rates (rush/pass, offense/defense), total penalty yards, penalties drawn,
  field goals over expected, return EPA, kicking EPA, defensive drive-score rate, field-position
  net** (v52/v53/v55, 2026-10-13) — each added alone to the 9-stat market-free baseline under
  walk-forward validation; none replicated positively in both chronological halves with a
  stable weight. (Pre-snap penalty yards is the one penalty cut that did, and shipped; total
  penalty yards and penalties drawn did not.) Note that this is judged market-free: these are
  not "the market already has it" rejections.
- **Offensive drive-score rate (`drive_score_off`)** (v55/v56, 2026-10-13) — the largest
  log-loss gain of any candidate (+0.0027, bootstrap p=0.0005) but it missed the AUC-confirm
  criterion, and the gain collapses to +0.0003 (v56) / +0.0005 with a negative confirm half
  (v58) when the baseline composite is allowed lower regularization. It was partly
  compensating for over-regularization, not adding information.

### 3.2 Plausible but not confirmed (parked, worth revisiting with more data or a cleaner test)

- **Correction ablation: do the shipped delta corrections still earn their place?** (v71, 2026-10-23; cowork A8 re-check and the
  momentum drop test). Fully walk-forward pipeline (v64's V4), test seasons 2011-2026 (3,999 games), each correction removed one at a
  time and everything else refit; v57 rule applied to "with vs. without". Only the QB gaps taken together pass (dLL +0.0027, dAUC
  +0.0036, both halves, p = 0.001, better in 100% of seasons). Everything else fails the rule as an individual piece:
  momentum dLL -0.0001 / dAUC -0.0005 (adds nothing; its weight is already z = 0.7, so this is the drop test: **candidate for
  removal**), skill-out dLL +0.0007 but negative in the early half (p = 0.11), home-field dAUC -0.0045 (the walk-forward constant is
  unstable early; log loss mixed), EPA QB gap alone p = 0.11, FantasyPros gap alone dLL +0.0004 (p = 0.34, better in only 38% of
  seasons), QB decay dLL +0.0010 (p = 0.015, both halves positive: close to the bar, consistent with its "real but not airtight"
  status). All corrections together vs. none: dLL +0.0025 (p = 0.23). Nothing was removed: dropping a shipped term is its own
  decision, and the QB corrections exist for calibration and for flagging games, not just hit rate. Re-run when a season of
  FantasyPros history longer than 2017+ accrues. The structural choices from v14-v51 (5-start primary window, mid-game QB injury
  filter, weather in the points model only) were not re-run.

- **Direct margin model for the ATS tab's cover probability** (v70, 2026-10-22; cowork B7.2) —
  margin distribution = the 400 nearest training games by predicted margin (shifted by the
  rounded gap), versus production's independent per-side scores re-weighted by historical score
  frequency. Against the weak form of the baseline (no mean calibration) it wins everything,
  but that is mostly the empirical margin recalibrating the *scale* of the predicted margin.
  Against the fair baseline (production's own bisection onto B's win probability, so only the
  distribution shape differs): margin log-likelihood +0.009 overall but -0.008 in the second
  half and better in only 50% of seasons (P = 0.275) — shape alone is NOT better; cover-probability
  log loss is better by +0.015 in both halves (P < 0.001) — the one piece that looks real. Not
  shipped: the pre-registered rule needs both metrics. Worth a follow-up that tests only the
  cover-probability piece (e.g. key-number mass at 3 and 7 on top of the production pipeline).

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
- **Kickoff/punt special-teams EPA (`st_kp`)** (v55/v58, 2026-10-13) — at the previous
  production L2=0.3 it was the one candidate positive in both halves with a positive weight in every
  fold (+0.0007 AUC select/confirm, bootstrap p=0.0145, better in 87% of seasons: WATCH). With
  the baseline composite at the better-regularized L2=0.03 (v58) the effect shrinks to
  +0.0004 AUC / +0.0004 log loss, p=0.12, better in 53% of seasons: indistinguishable from
  noise. Re-test if the composite's regularization changes again or more seasons accrue.

### 3.3 Deferred (not tested — blocked on data, not found wanting)

- **ESPN vs. nflverse QB injury sources** (cowork A8, logging started 2026-10-23) — the live gate reads ESPN while every backtest and the
  QB weights come from nflverse's weekly report. `refresh` now appends each team's QB statuses from both sources whenever either
  changes (kv `injury_source_log`, `nflhub/sources/injury_log.py`; nflverse polled at most hourly). Compare after the 2026 season:
  agreement rate on "Out/Doubtful", how many hours earlier ESPN knows, and whether disagreements matter.

- **State-space (Kalman) team ratings** (cowork B7.3, 2026-10-22) — a research branch that must
  beat the B1 composite under the ADOPT rule; not started (largest item on the plan, no evidence
  yet that the EWMA composite is the bottleneck given v61/v62/v63 all landed within noise).

- **"Steam" / line movement as a predictive signal** (2026-10-02) — the `Steam:` chip is
  currently cosmetic (flags `|Δspread| >= 1.5`, feeds nothing). Testing whether it predicts
  outcomes needs opening-vs-closing-line data; nflverse's `games.csv` only has closing lines,
  and this project's own `spread_history` table doesn't yet cover enough completed weeks for
  a real backtest.
  **Update 2026-10-22 (B5):** `spread_history` only began in week 3 and its first row per game was the line at the time the page's week
  rolled, not the true opener. The refresh now also polls NEXT week's ESPN lines every run (verified: week 8 had all 14 spreads
  during week 7), so from week 8 onward the first row per game is the opener and the last is the close. Nothing is logged for a week until Tuesday 6am ET of that week (`_opener_window_open`), so placeholder lines posted while the previous week is still settling never count as the opener (week 8 starts logging Tue 2026-10-27). Test (model-side ATS vs.
  opener, and vs. steam, split by halves, v60 discipline) is deferred until a full season of openers exists (earliest: after the
  2027 season).
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
EWMA_ALPHA = 0.08  (bias-corrected)  SEASON_PRIOR_GAMES = 8
WP_LO, WP_HI = 0.05, 0.95          HFA_SHRINKAGE_K = 100

DELTA_CALIBRATION_INTERCEPT = 0.0177   DELTA_CALIBRATION_SLOPE = 1.0481   DELTA_PROB_MAX = 0.85   (P(favorite) also floored at 0.5)
                                   (P(favorite wins) = sigmoid(a + b*|delta|), clean games, 4348)
QB_QUALITY_GAP_WEIGHT  = -1.2268   (calibration only, fallback when FantasyPros has no data that week)
QB_FANTASY_GAP_WEIGHT  = -0.0619   (calibration only, switches in per week INSTEAD of the EPA gap, never summed with it)
SKILL_EPA_OUT_WEIGHT   = -0.0747   (calibration only)
PRIMARY_WINDOW         = 5 games   (trailing-window plurality, ties -> more recent start)
QB_GAP_DECAY_RATE      = 0.4       (shrinks the gap each straight game a backup keeps starting)
HFA_WEIGHT             = +1.0366   (moves hit rate)
MOMENTUM_WEIGHT        = 0         (removed 2026-10-23, v71/v72; the streak plumbing stays and contributes 0)

MIDGAME_MIN_BACKUP_ATTEMPTS = 3    (display/filter only, not in delta)
SCORE_HIST_FIRST_SEASON = 2015     MAX_TEAM_SCORE = 60
```

`POWER_WEIGHTS`, `POINTS_WEIGHTS`, and `WEATHER_ADJUSTMENT_WEIGHTS` are each their own
9-term (POWER_WEIGHTS), 8-term (POINTS_WEIGHTS) and 3-term fitted dictionaries — see `nflhub/sources/team_ratings.py` directly for
current values rather than duplicating them here, since they're more likely to be re-fit
than the single-number weights above.
