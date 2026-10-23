# NFL Hub model: methodology review and improvement plan

Written 2026-10-15 in a Claude (Cowork) review session, against commit `03783d2`. The living version with charts is the Claude Doc "NFL Hub Model: Methodology & Improvement Avenues". This file is the hand-off copy for Claude Code.

**For Claude Code: start with [Part B: Instructions for Claude Code](#part-b-instructions-for-claude-code).** Part A is the reasoning and evidence behind each instruction.

**Status, updated 2026-10-16:** B1 (speed retune, v61/v62) and B2 (Elo as 10th stat, v63) have shipped; walk-forward AUC is now 0.6983 (from 0.6787). Added A6b and expanded B4/B4a: picks lock by game day (Thursday, all of Sunday at the first Sunday kickoff, Monday), so the pool optimizer should re-run before Sunday's first kickoff and before Monday's kickoff, using the field's already-locked picks and results so far.

---

## Part A: Review

### A1. Summary

The NFL Hub model is a careful, honestly validated team-strength model. It is not a betting edge. The biggest gain found in this review is a tuning fix, not a new feature: a slower rating update.

| Measure (walk-forward) | Model | Closing spread |
| --- | --- | --- |
| AUC, production composite (methodology doc) | 0.6787 | 0.7224 |
| AUC, v60 test seasons 2011+ | 0.6891 | 0.7220 |
| Log loss, v60 | 0.6323 | 0.6085 |
| Straight-up hit rate, 4,987 games | 63.5% | 66.8% |
| Hit rate when they pick different winners (925 games) | 41.0% | 59.0% |

Priorities, in order:

1. **Slow the ratings down.** `EWMA_ALPHA = 0.2` was never tuned. On points ratings, alpha 0.08 to 0.12 beat 0.2 by about +0.013 AUC in both halves and in 14 of 15 seasons, about ten times the lift of any feature added since September. A standard Elo points the same way (A7).
2. **Pool Monte Carlo.** Replace the hand-set deviation budget with a simulation that maximizes expected dollars under the pool's payout rules (A6).
3. **Log opening lines.** A model this far behind the closer may still beat the opener.
4. **Bigger, faster QB adjustment.** Model-side ATS is 46.8% in QB-flag games, in both halves.
5. **State-space ratings.** A Kalman-style rating generalizes item 1.
6. **Validation hygiene.** Fit correction constants walk-forward; hold out 2026.

Tested in this review and dropped: margin target instead of win/loss (tie), totals model vs. the closing O/U (49.9%).

### A2. How the model works today

**Ratings.** Nine per-team stats from nflverse play-by-play since 2007: rush/pass offense EPA, rush/pass defense EPA allowed, points for/against, turnovers committed/forced, pre-snap penalty yards. Garbage time (WP outside 0.05 to 0.95) is removed from EPA and turnover stats only. Each stat is an EWMA with `EWMA_ALPHA = 0.2` (half-life about 3 games; the code comment says "reused convention, not independently tuned"). Last season's final rating fades linearly to zero over `SEASON_PRIOR_GAMES = 5`. Only prior games are used.

**Power Score.** One non-negative, L2 = 0.03 logistic regression on home win (0/1). Share of total weight: points_off 30%, pass_off_epa 23%, points_def 22%, rush_off 10%, pen_presnap 8%, rush_def 6%, pass_def 2%, both turnover stats 0.

**delta.** Z-scored home-minus-away composite plus four corrections:

```
delta = delta_raw - 1.3268*dQB_EPA (or -0.0651*dQB_fantasy, whichever is active that week)
        - 0.0608*dSkill_out + 1.0407*HFA_season + 0.0256*dStreak
```

QB and skill health are calibration-only (they don't move hit rate). HFA and momentum move picks slightly. All corrections together: 62.87% to 63.09% hit rate (about 11 games in 4,977).

**Win probability.** `P(fav) = sigmoid(-0.0841 + 1.0852*|delta|)`, capped at 0.85 (reached at |delta| about 1.68). One sd of delta (0.66) gives a 65% favorite. At delta 0 the favorite gets 47.9%, under 50% (see A8).

**Points and scores.** A separate points regression (pass offense EPA weight 6.26 dominates), a weather adjustment for outdoor games inside the forecast window (-0.17 pts per mph wind, -0.05 per mm rain, -3.0 extreme cold), and per-team score distributions reshaped to post-2015 score frequencies and forced to agree with delta's win probability. The two teams' distributions are treated as independent. Everything is frozen at kickoff.

### A3. Validation results (v60, 2026-10-14)

- Model and market pick the same winner in 81.5% of games. Blending the model into the market improves log loss by 0.0004 (p = 0.18): noise.
- Betting the model's side against the closing spread: 51.6% over 4,861 games [50.2, 53.0]. By gap size: 0-1 pts 53.2%, 1-2 pts 54.3%, 2-3 pts 50.3%, 3-4 pts 47.9%, 4-6 pts 48.9%, 6+ pts 54.4%. No bin's interval clears the 52.4% break-even; the pattern is U-shaped and looks like noise across six bins.
- One segment held in both halves: model-side ATS 46.8% in QB-flag games (282 games). The market reacts to QB news more than the model does.

### A4. Already tried (don't repeat without new reasoning)

| Idea | Verdict | Why |
| --- | --- | --- |
| Weather vs. win prob (v29) | Null | Optimal dampening is zero; kept for points only |
| Rest, travel, divisional (v30) | Null | No calibration signal |
| OL, DL, LB unit health (v27) | Null | Sign flips between halves |
| Opponent adjustment, SRS (v5, v6) | Null | Worse than plain EWMA |
| Specific backup QB's own EPA (v38) | Null | Noisier than pooled average; shrinkage untried |
| Garbage-time-filtered points (v39) | Null | Blowout points are signal |
| QB return boost (v41, v44) | Null | Worse everywhere |
| Team-total fantasy projections (v48) | Null | 0.956 correlation with the spread |
| Explosive plays, FG over expected, return/kicking EPA, drive-score rate (v52-v58) | Null | Failed pre-registered rule |
| Margin target (this review) | Null | LL 0.6351 vs. 0.6351 |
| Totals vs. closing O/U (this review) | Null | 49.9% over 3,628 games; best bin 51.8% |
| DB/S unit health (v27) | Parked | p = 0.08-0.11; waiting on PFF |
| Trap game (v36) | Parked | Failed confirm half |
| Kick/punt EPA (v55, v58) | Parked | Noise at L2 = 0.03 |
| Line movement ("steam") | Deferred | Needs opening lines |
| Roster depth via `weekly_rosters` | Deferred | Not built |
| EPA back to 1999 via `stats_team` | Deferred | Not garbage-time filtered |
| NGS/PFR/FTN | Deferred | 4-10 seasons only |
| Single-week pool budget | Deferred | `pool_size / 10` untested (solved by B4) |

### A5. Evidence for slowing the ratings

Points-for/against EWMA ratings only, walk-forward 2011-2025, 4,080 games (script: `docs/cowork-tests/ewma_alpha_sweep.py`):

| alpha / carry | AUC | Log loss | <=2018 AUC | >2018 AUC |
| --- | --- | --- | --- | --- |
| 0.20 / 0.65 (production-like) | 0.6821 | 0.6358 | 0.6846 | 0.6793 |
| 0.10 / 0.80 | 0.6955 | 0.6292 | 0.6998 | 0.6910 |
| 0.08 / 0.65 | 0.6955 | 0.6297 | 0.7002 | 0.6903 |
| 0.06 / 0.65 | 0.6940 | 0.6315 | 0.6991 | 0.6885 |
| 0.30 / 0.65 | 0.6659 | 0.6440 | | |
| 0.40 / 0.65 | 0.6505 | 0.6514 | | |

alpha 0.10 vs. 0.20: dLL +0.0065, bootstrap p = 0.0005, better in 14 of 15 seasons. The whole 0.06-0.12 range beats 0.2. Caveat: the proxy uses a 0.65 regression-to-mean carryover, not the production linear 5-game fade.

### A6. Pool payout math (for the Monte Carlo)

Pool rules: each week 80% of the pot goes to the weekly winner (1st only). 20% builds a season pot paid 60/30/10 to the top three in season wins. Over a season, weekly prizes are 80% of all money paid out.

```
EV_week = 0.8 * P * Pr(win the week) + lambda * E[correct picks]
lambda  = marginal value of one season win
        = d/dwins [ 0.2 * P * weeks * (0.6*Pr(1st) + 0.3*Pr(2nd) + 0.1*Pr(3rd)) ]
```

- Weekly prize first: only 1st pays, so take contrarian picks that set you apart.
- The season term is a brake: lambda is about 0 when far from the top three and grows late in the year when you're in contention.
- Bigger fields justify more contrarian picks.
- Weekly ties go to the Monday-night tiebreaker guess; model it (`edge_core.monday_night_tiebreaker`).
- Pool data: 27 entrants (CBS "PARK'S NFL POOL"). Weekly picks land in the "2026 Pickem" Google Sheet every Tuesday via a Cowork scheduled task; `edge_sheet.py` already reads it.

### A6b. Progressive reveal: picks lock game by game

CBS locks picks by day: Thursday's game locks at its kickoff, and every Sunday game (early, late and SNF) locks together at the first Sunday kickoff (9:30 AM ET on London weeks). Monday's game(s) lock at Monday kickoff. Locked picks become visible to everyone. So in a normal week there are two decision points after the first lock:

| Decision point | Already visible | Still changeable |
| --- | --- | --- |
| Before the first Sunday kickoff (Fri to Sun morning) | Everyone's TNF pick and the TNF result | All Sunday and Monday picks |
| Before Monday kickoff (~8:00 PM Mon) | Everyone's Sunday picks and exact weekly standings through SNF | Monday pick(s) and the tiebreaker (the MNF total locks at Monday kickoff) |

Holiday weeks (Thanksgiving, late-season Saturdays, Christmas) add a decision point per extra game day, since each day's games lock at that day's first kickoff. The tiebreaker stays open until Monday kickoff, so it can be set knowing exactly who is in contention.

Opponents' picks for games not yet kicked off stay hidden, so the optimizer still predicts those (pick-share model), but it conditions on real standings. Two of the first five weeks show the value, reconstructed from the "2026 Pickem" sheet (weeks 4 and 5, where every game result is pinned down):

- **Week 5:** Jeremy was tied for the weekly lead (10) with Jorge Sanchez entering MNF (BUF at LAR). Jorge had BUF, Jeremy had LAR; BUF won and Jorge took the week at 11. With BUF, Jeremy ties Jorge at 11 and wins the tiebreaker (his guess 58 vs. Jorge's 42; actual total 71). That is hindsight, but it shows the decision that matters: entering MNF tied for the lead, the right pick depends on what the co-leaders probably have plus the tiebreaker edge, not just on who is likelier to win the game.
- **Week 4:** Jeremy entered MNF one behind Mario Sanchez with four others close. ATL won, producing a four-way tie at 12 settled by the tiebreaker: Jaime Flores and Mario Sanchez shared 1st with guesses of 45; Jeremy guessed 44 and finished 3rd. The tiebreaker number can be chosen knowing exactly who is in contention.

Entering MNF, 6 to 9 entrants were within one point of the lead in each of weeks 4 and 5, so late, state-aware decisions are often live.

### A7. Elo check

FiveThirtyEight-style Elo (K = 20, HFA 48, margin-of-victory multiplier, one-third regression to the mean), walk-forward 2011-2025 (script: `docs/cowork-tests/elo_check.py`):

| Model | AUC | Log loss |
| --- | --- | --- |
| Points ratings, alpha 0.2 | 0.682 | 0.635 |
| Elo alone | 0.692 | 0.630 |
| Points (0.2) + Elo | 0.695 | 0.628 |
| Points, alpha 0.1 | 0.696 | 0.628 |
| Points (0.1) + Elo | 0.696 | 0.627 |
| Closing spread | 0.723 | 0.608 |
| Spread + Elo | 0.722 | 0.608 |

Elo over alpha 0.2: dLL +0.0076, p < 0.001. Elo over alpha 0.1: dLL +0.0008, p = 0.21. Correlation 0.91. Elo mostly carries the information a slower update recovers. K 16-24 and regression 0.25-0.5 all land at 0.689-0.693.

### A8. Methodology risks

| Issue | Fix |
| --- | --- |
| Correction weights and calibration fit once on all data | Refit inside the walk-forward loop |
| Stats z-scored against the full 2007-2026 population | Standardize with training seasons only |
| About 60 experiments on the same history | Hold 2026 out; re-check v14-v51 ships against the v57 rule |
| Calibration curve gives 47.9% at delta 0 | Floor at 50% or refit with intercept 0 |
| Live injury source (ESPN) differs from backtest (nflverse) | Log both; compare after a season |
| Headline 63.09% hit rate predates 10/11-10/13 changes | Recompute after each refit |
| Public repo + Supabase RLS "anon all" | Fine for personal use; add a write key if needed |

---

## Part B: Instructions for Claude Code

### B0. Ground rules

- Read `docs/model-methodology.md` sections 1-4 first. This plan assumes its conventions.
- Follow the existing research pattern: one script per experiment in `research/`, numbered from the next free version (`edge_signal_test_v61_...`, `v62_...`), docstring stating question, data, method and result. Save console output to `research/output/vNN_run.txt`.
- Judge every change with the **v57 pre-registered ADOPT rule**: walk-forward 2011-2026; vs. current production, AUC and log loss better in **both** chronological halves (<=2018 / >2018), bootstrap log-loss p <= 0.01, better in >= 60% of seasons. Write the rule into the script before running it. Don't loosen it after seeing results.
- Judge market-free (the spread is never an input to `delta`), as the v52-v58 work does.
- When something ships: update production constants in `nflhub/sources/team_ratings.py`, bump `RATINGS_SCHEMA_VERSION` / `MATCHUPS_SCHEMA_VERSION` so stored snapshots rebuild, refit delta-scale constants and the calibration curve if the composite's scale changes (as v57 did), and add a row to the timeline in `docs/model-methodology.md` (sections 2 and 4). Nulls go into section 3.
- Never read, print or commit `.env`, `oauth2.json` or secrets. Don't change GitHub Actions secrets.
- One commit per experiment or shipped change, with the test's headline numbers in the message.
- `docs/cowork-tests/` holds the proxy scripts from this review. They run from the repo root against `research/cache/games.csv` and `research/output/team_ratings.csv`. They are leads, not production tests: re-test on the production `compute_ratings` game_log.

### B1. EWMA speed and season prior (do first)

Goal: tune the update speed the composite actually uses.

1. v61: on the production game_log, sweep `EWMA_ALPHA` in {0.06, 0.08, 0.10, 0.12, 0.15, 0.20} and `SEASON_PRIOR_GAMES` in {3, 5, 8, 12}, refitting `POWER_WEIGHTS` (same non-negative L2 = 0.03 fit) for each cell. Select on <=2018, confirm on >2018.
2. If v61 passes: try per-group alphas (offense EPA, defense EPA, points, penalties/turnovers) as v62, since EPA is noisier than points and may want a slower update. Keep only if it passes the rule against the v61 winner.
3. Re-check L2 at the new alpha (v57 found the optimum moves with the inputs).
4. Ship: update constants, refit delta-scale constants + calibration (QB, fantasy gap, skill, HFA, momentum, `DELTA_CALIBRATION_*`), bump schema versions, rebuild snapshots, update methodology timeline.

Done when: production AUC and log loss are recorded before and after, and the timeline row is written.

### B2. Elo as a 10th stat (right after B1)

1. v63: build Elo inside `team_ratings.py`'s pre-game loop (copy `elo()` from `docs/cowork-tests/_common.py`: K = 20, HFA 48, MOV multiplier, one-third regression, neutral sites get no HFA via `is_neutral_site()`).
2. Compare three models under the ADOPT rule: (a) B1-shipped composite, (b) composite + Elo as a 10th non-negative stat, (c) Elo alone.
3. If (b) fails: don't add it to delta. Optionally show Elo on the Power Rankings tab for reference (display only, no effect on `delta`).

### B3. Validation hygiene (can run in parallel with B1)

1. Move the fit of QB, fantasy gap, skill, HFA, momentum and calibration constants inside the walk-forward loop in the backtest path (`compute_backtest_scatter`), so each test season uses constants fit on earlier seasons only. Report how much the backtest hit rate drops; that is the honest number.
2. Standardize z-scores with training-season statistics only.
3. Floor `delta_win_prob` at 0.5 for the model's favorite, or refit calibration with intercept 0. Check reliability by |delta| bin before and after.
4. Add a prospective 2026 scorecard: from the frozen-at-kickoff predictions already stored in Supabase, compute weekly and cumulative log loss, Brier and hit rate vs. the de-vigged closing line, and show it on the Pipeline Health or Model Trends tab. Treat 2026 as the untouched holdout from now on.
5. Recompute the headline hit rate in the methodology doc.

### B4. Pool Monte Carlo (biggest practical value)

Goal: replace the deviation-budget heuristic in `nflhub/sources/edge_core.py` with expected-dollar optimization. It needs market probabilities, not the model.

1. Inputs per game: de-vigged market win probability (`odds.py` / Action Network), national pick % plus learned per-team bias (`edge_national.py`, `edge_bias.py`), each entrant's season wins so far (from the "2026 Pickem" sheet via `edge_sheet.py`), entrant count (27).
2. Opponent model: each opponent picks each game independently with probability = estimated pool pick share for that team. Calibrate against the weekly sheet tabs (weeks 1-5 exist; check log loss of predicted vs. actual pick shares).
3. Simulate N = 20,000 weeks: draw outcomes from market probs, draw opponents' cards, score everyone, apply the Monday-night tiebreaker (model the crowd's guesses from history in the sheet; use `monday_night_tiebreaker` for yours).
4. Objective = A6's `EV_week`: 0.8 x P x Pr(win week) plus lambda x E[correct picks], with lambda from a season-standings simulation (rest-of-season sims using the same pick models). Search pick cards greedily: start from all-favorites, flip the single pick that most improves EV, repeat until no flip helps. Cap runtime for the Actions cron.
5. Backtest on 2026 weeks 1-5 (and later weeks as they come): compare realized results and EV of the simulator's card vs. the current FADE/CHALK/NO_PLAY card vs. all-favorites. Report it; don't over-read five weeks.
6. Surface it on the Moneyline tab: recommended card, Pr(win week), EV in pool dollars, and which picks are contrarian. Keep the old recommendations visible until the simulator has a few weeks of record.
7. **Make it sequential (see A6b).** The optimizer must accept a mid-week state: results of finished games, every entrant's locked picks, and the set of games still open. It then simulates only the open games (market probs) and only opponents' open picks (pick-share model, which can now also use each opponent's revealed picks this week as a signal of how contrarian they are), and re-optimizes only Jeremy's still-open picks plus the tiebreaker. Pipeline per game day (derive times from the schedule, never fixed clock times):
    1. **Scrape at lock:** right after the first kickoff of each game day (Thursday, Sunday, Monday, plus holiday game days), capture every entrant's newly locked picks. That is the moment they become visible.
    2. **Wait for results:** don't optimize yet. The refresh job already pulls scores; when every game of that day is final, mark the day complete.
    3. **Run the Monte Carlo then:** conditioning on the day's results and the field's revealed picks, re-optimize Jeremy's still-open picks for the next lock. In a normal week this produces two recommendation runs: after TNF goes final (for the Sunday card, well before Sunday's first kickoff) and after SNF goes final (for the Monday pick and the tiebreaker, a full day before Monday kickoff).
    4. Re-run on demand if inputs change before the next lock (injury news, line moves).
8. **Tiebreaker optimization.** The tiebreaker locks at Monday kickoff, so set it in the post-SNF run: choose the MNF total guess that maximizes Pr(win) given the entrants who can still tie, the projected MNF total distribution (the score-distribution model), and opponents' historical guessing behavior (their current-week guesses stay hidden until Monday kickoff; use past guesses from the weekly sheet tabs, last column).
9. **Notify, never submit.** Push the recommended changes (which open picks to flip, the tiebreaker number, Pr(win week) before and after) via the existing ntfy reminders before each lock. The project stays read + reminder only.
10. Backtest the sequential mode on weeks 1-5 using the state at each window (reconstructable from the weekly sheet tabs plus game results and kickoff times from `games.csv`): how often would a re-optimized late pick or tiebreaker have changed Jeremy's weekly result? Report it honestly; five weeks is a small sample.
11. Document it in `docs/model-methodology.md` section 1.8.

### B4a. Live pool data feed (needed for sequential B4)

The weekly "2026 Pickem" sheet tab is written once, on Tuesday, by a Cowork scheduled task ("CBS pool picks to sheet") that scrapes the CBS weekly standings page in the Claude app's browser on Jeremy's PC. Sequential B4 needs the same data mid-week. Two routes; pick one after a short spike:

- **Preferred if it works: scrape CBS from the Actions job.** The CBS pick'em page is a Next.js app backed by `https://picks.cbssports.com/graphql` (operation names visible in network traffic, e.g. `CurrentPoolCBSMessage`). Investigate whether the weekly-standings query can be called with a session cookie stored as an Actions secret (same pattern as `ESPN_S2`/`ESPN_SWID`). If so, add `nflhub/sources/cbs_pool.py` that pulls, on the first refresh after each game day's first kickoff (and again after games go final): per entrant, locked picks per game, weekly points, season points, and (once visible) tiebreakers. Write to a new Supabase table (e.g. `pool_live_picks`, keyed by season, week, entrant, game) rather than the Google Sheet. Handle cookie expiry by alerting on the Pipeline Health tab. Pool id: `kbxw63b2ge3dknryg44ts===`; the URL `?week=` param is ignored by the page, but the GraphQL query likely takes a period id (the page uses `poolPeriodId`).
- **Fallback: more Cowork scrape runs.** Jeremy can add a Cowork run right after each game day's first kickoff (Thursday ~8:25 PM, Sunday ~1:10 PM, Monday ~8:25 PM ET; holiday game days need their own run). The Monte Carlo then runs in the Actions job once that day's games are final. Those would write a separate "Week N live" tab (locked picks filled, open games blank, plus a timestamp cell) so `edge_sheet.py` never mistakes a partial week for a final one. If this route is chosen, teach `edge_sheet.py` to read the live tab and ignore it once the final "Week N" tab exists. These runs need the PC on and the Claude app open.

Either way: never write picks to CBS.

### B5. Opening lines (start logging now, test later)

1. Make sure `spread_history` stores the first line seen each week (the opener) and the close for every game. Add a column if needed and a short test.
2. After at least one full season of openers: v6x test of model-side ATS against the opener and against line movement ("steam"), split by halves, with the same discipline as v60. Until then, record it as deferred in methodology section 3.3.

### B6. Bigger, faster QB adjustment

1. v6x: refit the QB weight (EPA and fantasy variants) against the ATS residual (actual margin minus closing spread) instead of win/loss, on QB-flag games. Check whether a larger weight fixes the 46.8% model-side ATS in QB-flag games, both halves.
2. Optional, bigger: prototype a QB-adjusted team rating (team-without-QB + QB value, as ELWAY does) and compare against the current correction under the ADOPT rule.

### B7. Smaller items

- Shrink the specific backup's trailing EPA toward the pooled backup average by sample size (empirical Bayes; v38 flagged this as untried). Ship only under the ADOPT rule.
- Correlated score model for the ATS tab: model the margin directly with key-number mass at 3 and 7 (e.g. historical margin distribution conditioned on the spread), and compare cover-probability calibration against the current independent-scores method.
- State-space ratings (Kalman filter per team, time-varying variance): only after B1/B2, as a research branch. It must beat the B1 composite under the ADOPT rule.

### B8. Don't do

- Don't retry anything in A4 marked Null without a new reason written in the script docstring.
- Don't add the market spread or market-derived features (team-total fantasy projections) into `delta`.
- Don't ship on log loss alone or AUC alone; both halves, both metrics.
