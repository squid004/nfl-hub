'use strict';

// Static content -- no live data dependency, so render() builds the DOM once and does
// nothing on later calls (same guard style as every other tab's "shell built once" pattern,
// just with no dynamic part at all here).
//
// MAINTENANCE RULE (per explicit user direction, 2026-10-07): whenever team_ratings.py's
// model changes -- a new signal ships, an existing weight is refit, something gets tested
// and rejected or put on hold -- update BOTH this file and docs/model-methodology.md in the
// same change. This page is the user-facing mirror of that doc, not a one-time snapshot.
// Bump LAST_UPDATED below whenever either is touched.
const METHODOLOGY_LAST_UPDATED = '2026-10-13';

const Methodology = {
  _built: false,

  render(ctx) {
    const el = document.getElementById('methodology');
    if (!el || this._built) return;
    this._built = true;

    el.innerHTML = `
      ${this._intro()}
      ${this._whatItDoes()}
      ${this._timeline()}
      ${this._notImplemented()}
      ${this._constants()}
    `;
  },

  _intro() {
    return `
      <div class="panel">
        <h2>Model Methodology</h2>
        <p class="muted small">How nfl-hub's prediction model works, how it got here, and
          what's been tried and rejected along the way -- updated alongside every change to
          <code>nflhub/sources/team_ratings.py</code>, including signals that were tested and
          put on hold or rejected, not just the ones that shipped. Full written version in the
          repo: <a href="https://github.com/squid004/nfl-hub/blob/master/docs/model-methodology.md"
          target="_blank" rel="noopener">docs/model-methodology.md</a>.
          <span class="small">Last updated ${METHODOLOGY_LAST_UPDATED}.</span></p>
      </div>`;
  },

  _whatItDoes() {
    return `
      <div class="panel">
        <h2>What the model does</h2>

        <h3>1. Foundation: per-team ratings</h3>
        <p class="small">Every prediction starts from the same 9 team-level stats, computed
          per team per week:</p>
        <table>
          <thead><tr><th>Stat</th><th>What it measures</th></tr></thead>
          <tbody>
            <tr><td>rush_off_epa / pass_off_epa</td><td>EPA/play on offense, rush and pass separately</td></tr>
            <tr><td>rush_def_epa_allowed / pass_def_epa_allowed</td><td>EPA/play allowed on defense</td></tr>
            <tr><td>points_off / points_def_allowed</td><td>Points scored / allowed per game</td></tr>
            <tr><td>turnovers_off / turnovers_def_forced</td><td>Turnovers committed / forced per game</td></tr>
            <tr><td>pen_presnap_yds</td><td>Yards lost per game to the team&rsquo;s own pre-snap penalties (false start, delay of game, offside, encroachment, neutral zone infraction, illegal formation/shift/motion/substitution, too many men) -- lower is better</td></tr>
          </tbody>
        </table>
        <ul class="small">
          <li><strong>Garbage time excluded</strong> from the EPA and turnover stats -- any
            play with win probability outside [0.05, 0.95] is dropped before averaging.
            Points and penalty yards are NOT filtered this way; a final score and a flag are whole-game facts.</li>
          <li><strong>EWMA + season-prior fade</strong>: each stat is an exponentially-weighted
            moving average (~4-game half-life) within a season. A new season starts from last
            season's own ending rating, fading to exactly zero over 5 games -- a decade-old
            season has zero path into this year's rating once last season is behind it.</li>
          <li><strong>Data window</strong>: 2007-present (a project choice; the underlying
            play-by-play actually goes back to 1999 -- see the nflverse data catalog doc).</li>
          <li><strong>Display scale</strong>: the 4 EPA stats show 0-100, anchored to the
            best/worst value EVER recorded across the whole dataset, so a weak season's best
            team doesn't read as inflated.</li>
          <li><strong>EPA distribution tiers</strong> (Power Rankings tab): every team on
            each 0-100 EPA stat falls into one of five groups -- Elite/Weak when a real gap
            (not just rank) cuts a group off from the rest, however small; otherwise Above/
            Below average if meaningfully off the mean, else just Average. Catches a cluster
            of similarly-bad teams that's nonetheless well below average as a real group,
            instead of it disappearing into an undifferentiated middle. Display only, not
            folded into the composite itself.</li>
        </ul>

        <h3>2. The composite Power Score</h3>
        <p class="small">One joint L2-regularized logistic regression (L2 = 0.03, chosen by
          walk-forward AUC over 0 to 0.3) across all 9 stats at
          once, each oriented "higher = always better" first, coefficients constrained
          <code>&gt;= 0</code>. The non-negativity constraint fixes a real failure mode: an
          unconstrained fit sign-flipped turnovers committed (more turnovers came out
          <em>positively</em> weighted) because the stats overlap heavily -- non-negative
          coefficients can only shrink a redundant stat toward zero, never flip its sign.</p>
        <p class="lean">Walk-forward AUC 0.6787 vs. 0.7224 for the closing market spread
          alone -- <strong>this does not beat the market</strong> at picking winners. Shown as
          descriptive context (Power Rankings tab), not a betting edge.</p>
        <p class="small"><strong>Historical Power Rankings tab</strong> uses the same formula
          but each team's simple full-season average, not the live tab's EWMA rolling rating
          -- every team-season is already fully known, so there's no "as of this week" to
          track. Z-scored against every team-season at once so different eras compare directly.</p>

        <h3>3. delta: the per-matchup prediction</h3>
        <p class="small">Four corrections are added on top of the raw composite
          (<code>delta_raw</code>) -- all additive, all fit once offline on the full graded
          dataset, none touch the 8-stat composite itself:</p>
        <table>
          <thead><tr><th>Correction</th><th>Weight</th><th>Captures</th><th>Moves hit rate?</th></tr></thead>
          <tbody>
            <tr><td>QB health (EPA)</td><td class="num">-1.3268</td><td>Trailing EPA/dropback gap, current primary (most starts in trailing 5 games) vs. whoever's actually playing, decayed 0.4/game for each straight start by the same backup (live games: a backup-average estimate, not league average). Active only as the FALLBACK when FantasyPros has no data for that week -- never summed with the fantasy gap below</td><td><span class="chip bad">No</span> (calibration)</td></tr>
            <tr><td>QB health (fantasy)</td><td class="num">-0.0651</td><td>Same gap/decay structure, sourced from FantasyPros' own weekly QB projections instead of EPA. SWITCHES in per game-week whenever that data is available, replacing the EPA gap entirely for that week rather than adding to it</td><td><span class="chip bad">No</span> (calibration)</td></tr>
            <tr><td>Skill-position health</td><td class="num">-0.0608</td><td>Summed trailing EPA of every Out/Doubtful RB/WR/TE/FB</td><td><span class="chip bad">No</span> (calibration)</td></tr>
            <tr><td>Home-field advantage</td><td class="num">+1.0407</td><td>Trailing-5-season home win rate, shrunk toward all-time average</td><td><span class="chip good">Yes</span></td></tr>
            <tr><td>Momentum</td><td class="num">+0.0256</td><td>Signed current win/loss streak, reset at season boundaries/ties</td><td><span class="chip good">Yes</span></td></tr>
          </tbody>
        </table>
        <p class="small" style="font-family:monospace; background:var(--panel2); padding:8px 10px; border-radius:6px;">
          delta = delta_raw<br>
          &nbsp;&nbsp;+ QB_QUALITY_GAP_WEIGHT &times; (home_qb_gap &minus; away_qb_gap)<br>
          &nbsp;&nbsp;+ SKILL_EPA_OUT_WEIGHT &times; (home_skill_out &minus; away_skill_out)<br>
          &nbsp;&nbsp;+ HFA_WEIGHT &times; home_field_logit[season] &nbsp;(0 if neutral site)<br>
          &nbsp;&nbsp;+ MOMENTUM_WEIGHT &times; (home_streak &minus; away_streak)
        </p>
        <p class="small"><code>delta_raw</code> is kept alongside <code>delta</code>
          everywhere for comparison. <code>delta_win_prob(delta)</code> converts the
          composite into a calibrated win probability via a curve fit against real
          backtested accuracy at each confidence level (intercept -0.0841, slope 1.0852,
          capped at 85%). The Against the Spread tab's spread and total come from the same
          probability: its score distribution is calibrated so the implied chance the home
          team wins equals this one.</p>
        <p class="small"><strong>QB Out/Doubtful detection uses two different sources:</strong>
          every fit/backtest/historical reconstruction uses nflverse's weekly injury report
          exclusively (the only source with a historical archive to validate against). The
          LIVE prediction for the current/upcoming week instead uses ESPN's live injury feed
          (same one the game-card's QB chip reads), because nflverse's status field routinely
          sits blank until the Friday pregame report, days after ESPN already has a real
          answer -- confirmed against a live case (Caleb Williams, CHI 2026 wk5: ESPN said
          Out Monday, nflverse was still blank Wednesday). Timeliness fix for that one path
          only, same reasoning as using a separate live source for weather -- nflverse
          doesn't forecast that either.</p>
        <p class="small"><strong>QB-health replacement estimate (backup average, not
          league average):</strong> league-wide EPA/play (+0.035) is dominated by starters
          (83% of all pass attempts) -- backup-only appearances actually average
          <strong>-0.070</strong>, a ~10x-larger, opposite-sign gap. Using the league figure
          made a below-average-but-real starter (Caleb Williams, career EPA &asymp; -0.003)
          look like an upgrade over his own backup once hurt -- backwards. Didn't need a
          refit of QB_QUALITY_GAP_WEIGHT (it was always fit against the real hindsight
          backup performance, never this estimate) -- walk-forward checked anyway
          (research/edge_signal_test_v37): hit rate flat on the affected games (same
          "statistically flat" pattern every QB/skill check has shown), calibration gap
          trending the right direction but not alone significant. Fixed 2026-10-08.</p>
        <p class="small"><strong>Who counts as a team's "primary" starter</strong> uses a
          trailing 5-game window (most starts in that window, ties broken toward the more
          recent start) -- not a season-long cumulative-attempts count. Switched 2026-10-11
          after the cumulative definition mislabeled a live matchup: a Details card blamed a
          backup-QB penalty on a passer who hadn't actually started in two games, because the
          real current starter had been hurt to <em>begin</em> the season, so the backup's
          early cumulative-attempts lead hadn't been caught up to yet even though the depth
          chart had already changed. 3-4 game windows overreact to a single-game blip and
          tested worse than cumulative; 5-6 games ties/slightly beats it on walk-forward AUC
          while reacting within 2-3 games instead of however long a cumulative crossover
          takes (research/edge_signal_test_v40).</p>
        <p class="small"><strong>The gap also decays the longer a backup has already
          started</strong> (QB_GAP_DECAY_RATE = 0.4) instead of applying the same flat-size
          correction on his 1st start and his 8th. A bundled version with a symmetric
          reinsertion boost on the primary's return tied production's AUC at best and
          underperformed everywhere else -- but decomposing the two (research/edge_signal_
          test_v44) found the bundle was hiding a real effect: decay ALONE beats static,
          isolated to games with an active absence (AUC +0.0020, logloss -0.0021, agreeing
          for the first time in this investigation); reinsertion alone made things worse
          every time. The 0.4 rate was then swept as its own free parameter (not assumed
          equal to EWMA_ALPHA) and validated with a chronological select/confirm split
          (_v45/_v46): selected on 2014-2020, replicated on the held-out 2021-2025 confirm
          set (AUC +0.0018) -- a 30-repeat random-split version agreed in direction but was
          noisier (70% of repeats favorable). Real, but not airtight.</p>
        <p class="small"><strong>A second, independent QB-health gap</strong> is added on top,
          sourced from FantasyPros' own weekly projections instead of trailing EPA. First
          tried as a TEAM-TOTAL signal (summed QB/RB/WR/TE projected points vs. a team's own
          DST projected points-allowed) -- confirmed null (research/edge_signal_test_v48): it
          looked like a huge win (AUC 0.6623&rarr;0.7148) until checked against the market
          spread directly -- correlation 0.956, and delta+spread_line alone reproduces almost
          the same AUC. Just a noisy proxy for the market the model already knows beats it,
          not new information. A narrower WITHIN-team version escaped that trap: primary's
          own trailing fantasy-point average vs. the actual passer's current-week projection,
          same gap/decay structure as the EPA version. Correlation with -spread_line is only
          0.187 (the EPA gap's own is 0.201) -- a within-team comparison isn't restating who's
          favored. Validated the same way as the decay rate above (research/edge_signal_
          test_v49/v50): chronological select/confirm (select 2017-2023, confirm 2024-2025)
          replicated (AUC +0.0097), and a 30-repeat random split never once picked the EPA gap
          alone as best (that pass compared fantasy-added-to-EPA against EPA alone).
          The comparison that actually matters once fantasy is in the model -- EPA+fantasy
          summed vs. fantasy alone -- came back far weaker: a direct bootstrap gave only
          +0.0059 mean AUC, favorable in 69.5% of resamples. Summing barely beats fantasy
          alone, and FantasyPros is a paid third-party API that can lapse or go down;
          EPA/dropback has zero dependency on it. So the shipped mechanism does not sum the
          two gaps -- it SWITCHES between them per game-week. QB_FANTASY_GAP_WEIGHT (-0.0651 after the 2026-10-13 L2 refit,
          z=-3.1) applies whenever FantasyPros actually returned data for that season/week;
          QB_QUALITY_GAP_WEIGHT (first fit jointly in _v51, now -1.3268 after the L2 refit, z=-3.3)
          is the automatic fallback the instant that data is missing -- lapsed key, API
          outage, or an unpublished week. <code>qb_fantasy_gap_data()</code> returns a
          <code>fantasy_available</code> flag per week to drive the switch; the power-model
          breakdown shows which source (<code>qb_health_source</code>) was actually active,
          so a fallback week is visible rather than silent.</p>
        <p class="small"><strong>Neutral-site exception:</strong> the HFA term (and the
          score-distribution pipeline's home/away points bias) is forced to exactly 0 for
          neutral-site games. nflverse's own <code>location</code> field is inconsistent --
          JAX's 2026 wk5 London game is tagged "Home" despite having no real home-field
          edge, and Buffalo's old Toronto Series games show the same mislabeling in some
          seasons but not others -- so <code>is_neutral_site()</code> combines that field
          with a small manual fallback list. Confirmed and fixed 2026-10-07, including
          three already-backfilled weeks that had been silently getting undeserved HFA
          credit.</p>
        <p class="small"><strong>Why QB/skill health don't move hit rate, mechanically:</strong>
          the term is small relative to everything else in delta (mean magnitude &asymp;0.10)
          and only nonzero for the ~1,200 graded games with a report-listed Out/Doubtful QB.
          It CAN flip a pick (62 of those 1,196 games flip, 5.2%), but the flips split almost
          evenly (32 improved, 30 regressed) -- exactly why it's a calibration correction
          (does delta honestly reflect known health context?), not a pick-accuracy play.</p>

        <details>
          <summary class="muted small">Points prediction, weather, and score distribution (click to expand)</summary>
          <h3>4. Points prediction and weather</h3>
          <p class="small">A SEPARATE regression (own-offense + opponent-defense &rarr;
            predicted points, same non-negative-L2 methodology as the Power Score but
            <em>independently fit</em> -- the two weight vectors aren't proportional, so two
            games with identical delta can still have different predicted point totals). A
            further, separately-fit residual adjustment (wind, precipitation, extreme cold)
            applies only when live forecast data is available, restricted to confirmed
            permanently-outdoor stadiums within Open-Meteo's ~16-day reliable window. "No
            weather data" always means "no adjustment," never a silent calm-weather assumption.</p>
          <h3>5. Score distribution</h3>
          <p class="small">Each side's full predicted-score curve (Against the Spread tab):
            a bias-corrected Normal curve around the weather-adjusted points prediction,
            reshaped to match REAL historical NFL scoring frequency since the 2015
            PAT-distance rule change (a genuine, validated improvement, not cosmetic), then
            calibrated via bisection so the distribution's own implied win probability
            exactly matches <code>delta_win_prob()</code> -- the points model and the win-
            probability model are forced to agree even though they're fit independently.
            Frozen at kickoff, same pattern as market odds.</p>
        </details>

        <details>
          <summary class="muted small">Mid-game QB injury, freeze-at-kickoff, and the pool-edge system (click to expand)</summary>
          <h3>6. Mid-game QB injury (descriptive only)</h3>
          <p class="small">Detected via two independent, both-required signals: a real
            passer change mid-game outside garbage time, confirmed by nflverse's own play
            text explicitly saying the departing passer "was injured during the play." NOT
            folded into delta -- by definition unknowable before kickoff. Surfaced as a Model
            Trends filter for historical analysis only.</p>
          <h3>7. Freeze-at-kickoff and historical reconstruction</h3>
          <p class="small">A pattern reused across market odds, the score distribution, the
            Parlays rating snapshot, and each week's Power Rankings snapshot. Past weeks are
            reconstructed EXACTLY using "every completed prior season plus this season's
            weeks strictly before week W" -- identical whether computed today or live back
            when week W actually started.</p>
          <h3>8. Pool-edge leverage system (adjacent, not part of the prediction model)</h3>
          <p class="small">A separate decision layer on the Moneyline tab: de-vigged market
            probability, a pool-popularity estimate (scraped national pick% + a learned
            per-team bias), and a deviation budget by season standing, producing FADE/CHALK/
            NO_PLAY recommendations. Operates on market odds and this specific pool's
            dynamics -- doesn't feed into or draw from delta.</p>
        </details>
      </div>`;
  },

  _timeline() {
    const rows = [
      ['2026-09-30', '8-stat EWMA rating system', 'Walk-forward: none of it beats the closing market spread (AUC &le;0.68 vs market&rsquo;s ~0.72)', 'muted', 'Descriptive only'],
      ['2026-09-30', 'Non-negative joint logistic composite (Power Score)', 'Fixed the turnovers-committed sign-flip; AUC 0.6746 vs market 0.7224', 'good', 'Shipped'],
      ['2026-10-05', 'QB-injury blind spot identified', 'Hit rate craters 63.6%&rarr;48.9% (z=-4.5) when the favorite&rsquo;s QB is Out/Doubtful; confirmed NOT a training-contamination problem', 'muted', 'Investigation only'],
      ['2026-10-06', 'QB health + skill-position health folded into delta', 'Hit rate 62.87%&rarr;63.07% (+10 games); every hit-rate backtest flat -- shipped for calibration (misses 29.0% vs hits 25.6%, p=0.01)', 'warn', 'Shipped (calibration)'],
      ['2026-10-06', 'Home-field advantage added', 'Hit rate 63.07%&rarr;63.11% (+12 vs baseline); calibration z 2.58&rarr;4.25 -- first correction to move raw hit rate', 'good', 'Shipped'],
      ['2026-10-06', 'Weather tested against win probability', 'Clean null: bad-weather hit rate (63.1%) vs good (63.7%), z=-0.24; optimal dampening found by grid search is exactly zero', 'bad', 'Not shipped'],
      ['2026-10-07', 'Rest/travel + divisional game tested', 'Null: rest-mismatch coefficient doesn&rsquo;t survive calibration check; divisional games show no real signal', 'bad', 'Not shipped'],
      ['2026-10-07', 'Score-distribution pipeline', '+0.16 nats/game-side log-likelihood from the historical-reweighting step (validated, not cosmetic)', 'good', 'Shipped (points/O-U)'],
      ['2026-10-07', 'Mid-game QB injury detection', 'Favorite&rsquo;s QB hurt mid-game &rarr; miss rate 2.34% vs 1.11% for an otherwise-identical hit (z=3.36, p=0.0008)', 'warn', 'Shipped as filter only'],
      ['2026-10-07', 'Momentum (signed win/loss streak) added', 'Isolated +6 games (63.10%&rarr;63.22%); cumulative 62.88%&rarr;63.22% (+17 games, 829 flips) BEFORE the neutral-site fix below. Calibration z=4.35, p&lt;0.0001', 'good', 'Shipped'],
      ['2026-10-07', 'Trap game tested', 'Sign consistent everywhere tried, but select/confirm split failed to replicate independently (select z=-3.08, confirm z=-1.77)', 'warn', 'Parked, unconfirmed'],
      ['2026-10-07', 'Neutral-site HFA bug fixed', '71 neutral-site games (international series, old Toronto Series) were wrongly getting home-field credit. Cumulative lift from HFA+momentum+QB+skill revised down to +10 games (62.87%&rarr;63.07%) once corrected -- smaller, but honest. 3 already-backfilled weeks self-healed in place', 'good', 'Bug fix'],
      ['2026-10-08', 'Live QB Out/Doubtful gate switched to ESPN&rsquo;s feed', 'No backtested-effect number -- changes WHEN the existing QB-health term fires this week, not what it&rsquo;s fit against. Found live: Caleb Williams (CHI, wk5) was already Out on ESPN Monday, nflverse&rsquo;s report_status was still blank Wednesday', 'good', 'Shipped (timeliness only)'],
      ['2026-10-08', 'QB-health replacement estimate: backup average, not league average', 'League EPA/play (+0.035) is mostly starters (83% of attempts) -- backups alone average -0.070. No refit needed (weight was always fit against real hindsight backup data). Hit rate flat on affected games (374&rarr;375 of 591, z=+0.21) but calibration gap widened the right way (z 0.91&rarr;1.36); net historical lift +10&rarr;+11 games (62.87%&rarr;63.09%)', 'good', 'Shipped (research/edge_signal_test_v37)'],
      ['2026-10-11', 'Specific identified backup&rsquo;s own EPA, instead of pooled backup-average', 'ESPN depth-chart name resolution worked correctly against a real case (CHI &rarr; Bagent, 152 attempts). On the 287 games with enough sample to test (66.6% coverage), using it scored worse (176/281) than the pool average (178/281) -- 12-flip paired check confirms noise either way (z=-0.58)', 'bad', 'Not shipped (research/edge_signal_test_v38)'],
      ['2026-10-11', 'Garbage-time-filtered points_off/points_def_allowed, composite reweighted', 'points_off/points_def_allowed are the #1 and #3 heaviest-weighted of all 8 stats (~41% combined) and the only two not garbage-time-filtered like every EPA stat already is. Rebuilt filtered and refit the whole composite from scratch: AUC 0.6746&rarr;0.6709, log loss and accuracy both worse too', 'bad', 'Not shipped (research/edge_signal_test_v39)'],
      ['2026-10-11', 'QB &ldquo;primary&rdquo; definition switched from season-cumulative attempts to a trailing 5-game window; refit QB_QUALITY_GAP_WEIGHT', 'Found via a live mislabeling bug (a Details card blamed the wrong team&rsquo;s backup-QB situation -- the cumulative definition hadn&rsquo;t caught up to a real in-season starter change yet). Window sweep: 3-4 games overreacts to a single-game blip and tests worse than cumulative; 5-6 games ties/slightly beats it (AUC 0.6739 vs 0.6739, accuracy 62.08% vs 61.79%). Refit -0.5903&rarr;-0.5202 (still z=-3.25)', 'good', 'Shipped (research/edge_signal_test_v40/v42)'],
      ['2026-10-11', 'Decaying the QB-health gap by backup tenure, with a symmetric reinsertion boost on return, bundled', 'Hypothesis: the team-level rolling rating (same EWMA half-life) independently catches up to a backup&rsquo;s true quality after ~4-5 starts, so a flat correction double-counts it. Bundled, ties production&rsquo;s AUC at best (0.6739) and underperforms everywhere else (0.6707-0.6720)', 'warn', 'Not shipped as a bundle -- decomposed next'],
      ['2026-10-11', 'Decomposed decay from reinsertion: decay alone beats static, reinsertion alone was the problem', 'Isolated to games with an active absence: decay-only AUC 0.6836 vs static 0.6816 (+0.0020), logloss 0.6439 vs 0.6460 (-0.0021) -- both metrics agreeing for the first time in this investigation. Reinsertion-only underperformed static in every view tested', 'good', 'Decay validated further (research/edge_signal_test_v44)'],
      ['2026-10-11', 'Decay rate swept as its own free parameter and validated with chronological + random select/confirm splits; refit QB_QUALITY_GAP_WEIGHT', 'Smooth inverted-U peaking near rate 0.4 in both AUC and logloss. Chronological split (select 2014-2020, confirm 2021-2025, never touched during selection): rate 0.4 replicated, AUC +0.0018. A 30-repeat random 50/50 split agreed in direction but was noisier (70% of repeats favorable). Refit -0.5202&rarr;-1.0556 (MORE significant, z=-4.23, was z=-3.25)', 'good', 'Shipped (research/edge_signal_test_v45-v47)'],
      ['2026-10-12', 'Team-total fantasy-projection signal tested (summed QB/RB/WR/TE points vs. market spread)', 'Looked like a huge win (AUC 0.6623&rarr;0.7148) until checked against the market directly -- correlation with -spread_line 0.956, and delta+spread_line alone reproduces almost the identical AUC (0.7278). Just a noisy proxy for the market the model already knows beats it, not new information', 'bad', 'Not shipped (research/edge_signal_test_v48)'],
      ['2026-10-12', 'QB-health fantasy gap (within-team, FantasyPros vs. trailing EPA) added alongside the EPA gap; refit both weights', 'Escaped the team-total version&rsquo;s market-proxy trap (corr with -spread_line only 0.187 vs the EPA gap&rsquo;s own 0.201). Chronological select/confirm (select 2017-2023, confirm 2024-2025): AUC +0.0097, replicated. 30-repeat random split never once picked the EPA gap alone as best (76.7% of repeats favorable). Refit QB_QUALITY_GAP_WEIGHT -1.0556&rarr;-0.8818 (z=-3.46), new QB_FANTASY_GAP_WEIGHT -0.0429 (z=-3.24)', 'good', 'Shipped (research/edge_signal_test_v48-v51)'],
      ['2026-10-12', 'QB-health mechanism changed from additive (EPA+fantasy summed) to a per-week switch/fallback', 'Direct bootstrap of EPA+fantasy (both) vs. fantasy alone -- the comparison that actually matters once fantasy is in the model -- came back weak: +0.0059 mean AUC, only 69.5% of resamples favorable. Not worth the dependency: FantasyPros is a paid third-party API that can lapse or outage mid-season. Switched to EITHER/OR per game-week (never summed), gated on a fantasy_available flag, with the EPA gap as fallback specifically because it has zero third-party dependency', 'good', 'Shipped'],
      ['2026-10-13', 'Pre-snap penalty yards added as the 9th composite stat; weights refit', 'Judged against the market-free model (the market spread is never an input here, so any replicated signal is a real improvement). Walk-forward 2011-2026: AUC 0.6749&rarr;0.6763, log loss 0.6464&rarr;0.6457; better on log loss in 14 of 16 seasons, weight positive in 16 of 16 folds, same direction in both chronological halves (select +0.0016 / confirm +0.0007 AUC), bootstrap log-loss p=0.002. Downstream QB/skill/HFA/momentum/calibration constants re-checked under the new composite (&lt;2% shift) and left unchanged', 'good', 'Shipped (research/edge_signal_test_v52-v54)'],
      ['2026-10-13', 'Every v52 candidate re-run market-free, one at a time vs. the 9-stat baseline', 'v52 had screened partly on correlation with the closing spread, the wrong gate for a model that never uses the spread. Pre-registered SHIP rule (both halves positive on AUC and log loss, bootstrap p&le;0.01, weight positive in &ge;90% of folds, better in &ge;60% of seasons): no candidate qualified. Explosive plays, total penalty yards, penalties drawn, FG over expected, return and kicking EPA, defensive drive-score rate: null or negative. Offensive drive-score had the biggest log-loss gain, but it shrank to near zero (negative confirm half) once the baseline composite was re-judged at lower regularization &mdash; compensating for over-regularization, not new information. Kickoff/punt EPA is consistently positive but fades to noise (p=0.12) at lower L2', 'warn', 'Nothing shipped (research/edge_signal_test_v55-v58)'],
      ['2026-10-13', 'Composite L2 lowered 0.3 to 0.03; every delta-scale constant and the delta-to-probability calibration refit; momentum wired into upcoming games', 'The original L2 grid bottomed out at 0.3, so the optimum sat off its edge. Extended it on the production compute_ratings game_log (9 stats, walk-forward 2011-2026). Pre-registered ADOPT rule (vs. 0.3: AUC and log loss better in BOTH halves, bootstrap p&lt;=0.01, better in &gt;=60% of seasons): L2=0.03 AUC 0.6763&rarr;0.6787, log loss 0.6457&rarr;0.6390, select +0.0033 / confirm +0.0021 AUC, better in 75% of seasons; 0 to 0.01 are indistinguishable, 0.03 is the largest within 0.0005 AUC of the best. Pre-snap penalty yards still earns its place (+0.0012 AUC, p=0.018). Weights roughly double in scale, so delta is ~1.45x wider and every delta-equivalent constant was refit at identical z-scores (QB -0.8818&rarr;-1.3268, fantasy gap -0.0429&rarr;-0.0651, skill -0.0363&rarr;-0.0608, HFA 0.9169&rarr;1.0407, momentum 0.0262&rarr;0.0256; calibration 0.0484/1.4497&rarr;-0.0841/1.0852). Backtest favorite hit rate 63.5%; predicted vs. actual win rate within ~3 points in every |delta| bin. Also fixed: momentum was contributing 0 to every live and backfilled-week matchup (streaks only existed for games already played). Stored weekly snapshots (Power Scores, Moneyline and ATS probabilities) and frozen score distributions were rebuilt under the new model', 'Shipped (research/edge_signal_test_v57, v58)'],
    ];
    const body = rows.map(([date, change, effect, cls, status]) => `
      <tr>
        <td class="small">${date}</td>
        <td>${change}</td>
        <td class="small">${effect}</td>
        <td><span class="chip ${cls}">${status}</span></td>
      </tr>`).join('');
    return `
      <div class="panel">
        <h2>Development timeline</h2>
        <p class="muted small">Each row is a change to what's actually live, with the
          backtested effect measured at that point against every graded decided game
          available at the time.</p>
        <table class="methodology-timeline">
          <thead><tr><th>Date</th><th>Change</th><th>Backtested effect</th><th>Status</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
        <p class="lean">Net effect across the whole timeline: the model's backtested SU hit
          rate moved from <strong>62.87%</strong> (pure 8-stat composite) to
          <strong>63.09%</strong> (current, QB/skill health + HFA + momentum all stacked,
          after the neutral-site HFA fix and the backup-baseline fix) across 4,977 graded
          decided games -- a <strong>+0.22 point / +11 game</strong> lift from five
          additions, two of which were deliberately shipped for calibration rather than
          accuracy. The model still does not beat the closing market spread at picking
          straight-up winners; it was never designed to.</p>
      </div>`;
  },

  _notImplemented() {
    const section = (title, cls, items) => `
      <h3><span class="chip ${cls}" style="margin-right:6px;">&nbsp;</span>${title}</h3>
      <ul class="small">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
    return `
      <div class="panel">
        <h2>Tried but not implemented</h2>
        <p class="muted small">Organized by why each one didn't ship -- the reason matters
          for whether it's worth revisiting.</p>

        ${section('Confirmed null -- don&rsquo;t re-attempt without new reasoning', 'bad', [
          '<strong>Weather vs. win probability</strong> -- real effect on predicted points (shipped separately), but nothing to add to win probability: hit rate in bad vs. good weather is statistically identical, grid search&rsquo;s own optimal dampening is zero.',
          '<strong>Rest/travel differential and divisional games</strong> -- rest-mismatch coefficient doesn&rsquo;t survive the calibration check; divisional games show no real signal either.',
          '<strong>Offensive line and defensive line unit health</strong> -- tested 3 separate ways (raw count, snap-weighted, select/confirm threshold scan). Sign even flips between select and confirm -- pure noise, confirmed null all 3 times.',
          '<strong>Linebacker unit health</strong> -- downgraded from an earlier "plausible, underpowered" read to confirmed-null under a stricter select/confirm test (select p=0.055, confirm p=0.899).',
          '<strong>QB-injury blind spot as a training-contamination problem</strong> -- tested whether backup-QB games were biasing the joint weight fit. Refitting on clean-only games barely moved the weights; scoring held-out backup-QB games with old vs. new weights was flat. The blind spot isn&rsquo;t fixable by reweighting the existing 8 stats.',
          '<strong>Single-pass opponent adjustment and full SRS/Ridge simultaneous solve</strong> -- both worse than the plain EWMA rating, likely because NFL scheduling already balances out and the market already prices in schedule strength.',
          '<strong>Specific identified backup&rsquo;s own trailing EPA, instead of pooled backup-average</strong> -- name resolution via ESPN&rsquo;s QB depth chart worked correctly (verified live), but on the 287 historical games with enough sample to test it, using the real backup&rsquo;s own EPA scored worse (176/281) than the pooled average (178/281) -- a 12-flip paired check confirms that&rsquo;s noise, not a real effect (z=-0.58). More specific isn&rsquo;t automatically more accurate at a 50-150 attempt sample size.',
          '<strong>Garbage-time-filtered points_off/points_def_allowed, composite reweighted</strong> -- points_off/points_def_allowed are the #1 and #3 heaviest-weighted of all 8 stats (~41% combined) and the only two NOT garbage-time-filtered the way every EPA stat already is. Rebuilt filtered (per-play score-state deltas, same WP band) and refit the whole 8-stat composite from scratch: AUC 0.6746&rarr;0.6709, log loss and accuracy both worse too, refit weights barely moved. Garbage-time points aren&rsquo;t noise the model was being fooled by -- a team that wins big enough to rack them up is still genuinely better on average.',
          '<strong>A symmetric reinsertion boost crediting the QB-health gap back when the primary returns, decaying over the following games instead of snapping to 0</strong> -- the natural complement to the decay idea that DID ship (see the table above): shouldn&rsquo;t the correction credit something back once the rolling rating, still lagging the recovered primary, catches up? Bundled with decay, this tied production&rsquo;s AUC at best and underperformed everywhere else (after fixing a real bug where the first attempt only credited ONE week of recovery instead of decaying over several). Decomposing decay from reinsertion confirmed it&rsquo;s reinsertion specifically that&rsquo;s the problem: reinsertion-only underperformed static in every view tested (full population AND isolated to active-absence games), while decay-only beat static in both. Confirmed null on its own, independent of decay&rsquo;s fate.',
          '<strong>Team-total fantasy-projection signal</strong> (v48) -- summed QB/RB/WR/TE projected points for offense, a team&rsquo;s own DST projected points-allowed for defense, tested as a team-vs-team differential on top of delta. Looked like a large win in isolation (AUC 0.6623&rarr;0.7148, consistent across every one of 5 seasons tested) until checked against the market spread directly: correlation with -spread_line is 0.956, and delta+spread_line alone reproduces almost the exact same AUC (0.7278) -- once the real market line is already in the model, every fantasy-derived candidate adds nothing (AUC deltas -0.0001 to -0.0013, pure noise). Not a new signal -- FantasyPros&rsquo; own weekly projections are themselves built partly from the game&rsquo;s own spread/total, so this was re-discovering &ldquo;the market beats the model&rdquo; through a noisy back door. A narrower WITHIN-team version of the same data (the QB-health fantasy gap) escaped this trap -- see the table above.',
          '<strong>Explosive-play rates, total penalty yards, penalties drawn, field goals over expected, return EPA, kicking EPA, defensive drive-score rate, field-position net</strong> -- each added alone to the 9-stat market-free baseline under walk-forward validation; none replicated positively in both chronological halves with a stable weight. (Pre-snap penalty yards is the one penalty cut that did, and shipped.) Judged market-free, so these are not &ldquo;the market already has it&rdquo; rejections.',
          '<strong>Offensive drive-score rate</strong> -- the largest log-loss gain of any candidate (+0.0027, bootstrap p=0.0005) but it missed the AUC-confirm criterion, and the gain collapses to near zero (negative confirm half) when the baseline composite is allowed lower regularization. It was partly compensating for over-regularization, not adding information.',
        ])}

        ${section('Plausible but not confirmed -- parked, worth revisiting', 'warn', [
          '<strong>Secondary (DB/S) unit health</strong> -- the one real survivor of the OL/DL/LB/secondary sweep: confirmed in the correct direction on a proper train-early/confirm-late split, but falls just short of the calibration significance bar (z=1.61, p=0.106, vs QB&rsquo;s p=0.01). On hold pending a decision on buying Pro Football Focus (PFF) per-player grades, which could plausibly sharpen this into a real signal.',
          '<strong>Trap game</strong> -- consistent direction everywhere tried, but doesn&rsquo;t independently clear significance on a held-out half. Worth another look with more seasons of data, team power rating instead of market spread as the "tough opponent" measure, or a "letdown after a big win" framing instead of schedule-lookahead.',
          '<strong>Kickoff/punt special-teams EPA</strong> -- at the previous production L2=0.3 it was the one candidate positive in both halves with a positive weight in every fold (+0.0007 AUC each half, bootstrap p=0.0145, better in 87% of seasons). With the baseline composite at the better-regularized L2=0.03 the effect shrinks to +0.0004 AUC, p=0.12, better in 53% of seasons: indistinguishable from noise. Re-test if the composite&rsquo;s regularization changes again or more seasons accrue.',
        ])}

        ${section('Deferred -- blocked on data, not found wanting', 'muted', [
          '<strong>"Steam" / line movement</strong> as a predictive signal -- the Steam chip is currently cosmetic. Testing needs opening-vs-closing-line data nflverse doesn&rsquo;t provide, and this project&rsquo;s own spread-history table doesn&rsquo;t yet cover enough completed weeks.',
          '<strong>Broader roster-depth signal via weekly_rosters</strong> -- nflverse&rsquo;s Active/IR/PUP/PS status field could extend the QB-only health work toward general roster depth without waiting on PFF -- not yet attempted.',
          '<strong>stats_team&rsquo;s pre-aggregated EPA back to 1999</strong> -- could extend every backtest&rsquo;s window by 8 seasons, but isn&rsquo;t garbage-time filtered the way this project&rsquo;s own PBP-derived EPA is -- needs its own validation first.',
          '<strong>NextGen Stats / PFR advanced stats / FTN charting</strong> -- genuinely new signal types (CPOE, pressure rate, motion/RPO), but shorter backtest windows (4-10 seasons) mean less out-of-sample room than the 19-season window the current system uses.',
          '<strong>"Win this single week" leverage budget</strong> -- the pool-edge system&rsquo;s season-long deviation budget has no validated single-week equivalent; never backtested.',
        ])}
      </div>`;
  },

  _constants() {
    return `
      <div class="panel">
        <h2>Current production constants</h2>
        <table>
          <tbody>
            <tr><td>FIRST_SEASON</td><td class="num">2007</td><td>FIRST_INJURY_SEASON</td><td class="num">2009</td></tr>
            <tr><td>EWMA_ALPHA</td><td class="num">0.2</td><td>SEASON_PRIOR_GAMES</td><td class="num">5</td></tr>
            <tr><td>WP_LO / WP_HI</td><td class="num">0.05 / 0.95</td><td>HFA_SHRINKAGE_K</td><td class="num">100</td></tr>
            <tr><td>DELTA_CALIBRATION_INTERCEPT / SLOPE</td><td class="num">-0.0841 / 1.0852</td><td class="muted small">P(favorite wins) = sigmoid(a + b&times;|delta|), capped at 0.85 (DELTA_PROB_MAX)</td><td></td></tr>
            <tr><td>QB_QUALITY_GAP_WEIGHT</td><td class="num">-1.3268</td><td class="muted small">calibration only, fallback when FantasyPros has no data that week</td><td></td></tr>
            <tr><td>QB_FANTASY_GAP_WEIGHT</td><td class="num">-0.0651</td><td class="muted small">calibration only, switches in per week INSTEAD of the EPA gap, never summed with it</td><td></td></tr>
            <tr><td>SKILL_EPA_OUT_WEIGHT</td><td class="num">-0.0608</td><td class="muted small">calibration only</td><td></td></tr>
            <tr><td>HFA_WEIGHT</td><td class="num">+1.0407</td><td class="muted small">moves hit rate</td><td></td></tr>
            <tr><td>MOMENTUM_WEIGHT</td><td class="num">+0.0256</td><td class="muted small">moves hit rate</td><td></td></tr>
            <tr><td>PRIMARY_WINDOW</td><td class="num">5 games</td><td class="muted small">trailing-window plurality, ties -> more recent start</td><td></td></tr>
            <tr><td>QB_GAP_DECAY_RATE</td><td class="num">0.4</td><td class="muted small">shrinks the gap each straight game a backup keeps starting</td><td></td></tr>
            <tr><td>SCORE_HIST_FIRST_SEASON</td><td class="num">2015</td><td>MAX_TEAM_SCORE</td><td class="num">60</td></tr>
          </tbody>
        </table>
        <p class="tablefoot muted">POWER_WEIGHTS, POINTS_WEIGHTS, and WEATHER_ADJUSTMENT_WEIGHTS
          are each their own fitted multi-term dictionaries -- see
          <code>nflhub/sources/team_ratings.py</code> directly for current values rather than
          duplicating them here, since they're more likely to be re-fit than the single-number
          weights above.</p>
      </div>`;
  },
};
