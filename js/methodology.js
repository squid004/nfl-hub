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
const METHODOLOGY_LAST_UPDATED = '2026-10-07';

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
        <p class="small">Every prediction starts from the same 8 team-level stats, computed
          per team per week:</p>
        <table>
          <thead><tr><th>Stat</th><th>What it measures</th></tr></thead>
          <tbody>
            <tr><td>rush_off_epa / pass_off_epa</td><td>EPA/play on offense, rush and pass separately</td></tr>
            <tr><td>rush_def_epa_allowed / pass_def_epa_allowed</td><td>EPA/play allowed on defense</td></tr>
            <tr><td>points_off / points_def_allowed</td><td>Points scored / allowed per game</td></tr>
            <tr><td>turnovers_off / turnovers_def_forced</td><td>Turnovers committed / forced per game</td></tr>
          </tbody>
        </table>
        <ul class="small">
          <li><strong>Garbage time excluded</strong> from the EPA and turnover stats -- any
            play with win probability outside [0.05, 0.95] is dropped before averaging.
            Points are NOT filtered this way; a final score is the whole-game outcome.</li>
          <li><strong>EWMA + season-prior fade</strong>: each stat is an exponentially-weighted
            moving average (~4-game half-life) within a season. A new season starts from last
            season's own ending rating, fading to exactly zero over 5 games -- a decade-old
            season has zero path into this year's rating once last season is behind it.</li>
          <li><strong>Data window</strong>: 2007-present (a project choice; the underlying
            play-by-play actually goes back to 1999 -- see the nflverse data catalog doc).</li>
          <li><strong>Display scale</strong>: the 4 EPA stats show 0-100, anchored to the
            best/worst value EVER recorded across the whole dataset, so a weak season's best
            team doesn't read as inflated.</li>
        </ul>

        <h3>2. The composite Power Score</h3>
        <p class="small">One joint L2-regularized logistic regression across all 8 stats at
          once, each oriented "higher = always better" first, coefficients constrained
          <code>&gt;= 0</code>. The non-negativity constraint fixes a real failure mode: an
          unconstrained fit sign-flipped turnovers committed (more turnovers came out
          <em>positively</em> weighted) because the 8 stats overlap heavily -- non-negative
          coefficients can only shrink a redundant stat toward zero, never flip its sign.</p>
        <p class="lean">Walk-forward AUC 0.6746 vs. 0.7224 for the closing market spread
          alone -- <strong>this does not beat the market</strong> at picking winners. Shown as
          descriptive context (Power Rankings tab), not a betting edge.</p>

        <h3>3. delta: the per-matchup prediction</h3>
        <p class="small">Four corrections are added on top of the raw composite
          (<code>delta_raw</code>) -- all additive, all fit once offline on the full graded
          dataset, none touch the 8-stat composite itself:</p>
        <table>
          <thead><tr><th>Correction</th><th>Weight</th><th>Captures</th><th>Moves hit rate?</th></tr></thead>
          <tbody>
            <tr><td>QB health</td><td class="num">-0.5903</td><td>Trailing EPA/dropback gap, normal starter vs. whoever's actually playing</td><td><span class="chip bad">No</span> (calibration)</td></tr>
            <tr><td>Skill-position health</td><td class="num">-0.0362</td><td>Summed trailing EPA of every Out/Doubtful RB/WR/TE/FB</td><td><span class="chip bad">No</span> (calibration)</td></tr>
            <tr><td>Home-field advantage</td><td class="num">+0.9169</td><td>Trailing-5-season home win rate, shrunk toward all-time average</td><td><span class="chip good">Yes</span></td></tr>
            <tr><td>Momentum</td><td class="num">+0.0262</td><td>Signed current win/loss streak, reset at season boundaries/ties</td><td><span class="chip good">Yes</span></td></tr>
          </tbody>
        </table>
        <p class="small" style="font-family:monospace; background:var(--panel2); padding:8px 10px; border-radius:6px;">
          delta = delta_raw<br>
          &nbsp;&nbsp;+ QB_QUALITY_GAP_WEIGHT &times; (home_qb_gap &minus; away_qb_gap)<br>
          &nbsp;&nbsp;+ SKILL_EPA_OUT_WEIGHT &times; (home_skill_out &minus; away_skill_out)<br>
          &nbsp;&nbsp;+ HFA_WEIGHT &times; home_field_logit[season]<br>
          &nbsp;&nbsp;+ MOMENTUM_WEIGHT &times; (home_streak &minus; away_streak)
        </p>
        <p class="small"><code>delta_raw</code> is kept alongside <code>delta</code>
          everywhere for comparison. <code>delta_win_prob(delta)</code> converts the
          composite into a calibrated win probability via a curve fit against real
          backtested accuracy at each confidence level.</p>
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
      ['2026-10-07', 'Momentum (signed win/loss streak) added', 'Isolated +6 games (63.10%&rarr;63.22%); cumulative 62.88%&rarr;63.22% (+17 games, 829 flips). Calibration z=4.35, p&lt;0.0001', 'good', 'Shipped'],
      ['2026-10-07', 'Trap game tested', 'Sign consistent everywhere tried, but select/confirm split failed to replicate independently (select z=-3.08, confirm z=-1.77)', 'warn', 'Parked, unconfirmed'],
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
          rate moved from <strong>62.88%</strong> (pure 8-stat composite) to
          <strong>63.22%</strong> (current, QB/skill health + HFA + momentum all stacked)
          across 4,976 graded decided games -- a <strong>+0.34 point / +17 game</strong> lift
          from four additions, two of which were deliberately shipped for calibration rather
          than accuracy. The model still does not beat the closing market spread at picking
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
        ])}

        ${section('Plausible but not confirmed -- parked, worth revisiting', 'warn', [
          '<strong>Secondary (DB/S) unit health</strong> -- the one real survivor of the OL/DL/LB/secondary sweep: confirmed in the correct direction on a proper train-early/confirm-late split, but falls just short of the calibration significance bar (z=1.61, p=0.106, vs QB&rsquo;s p=0.01). On hold pending a decision on buying Pro Football Focus (PFF) per-player grades, which could plausibly sharpen this into a real signal.',
          '<strong>Trap game</strong> -- consistent direction everywhere tried, but doesn&rsquo;t independently clear significance on a held-out half. Worth another look with more seasons of data, team power rating instead of market spread as the "tough opponent" measure, or a "letdown after a big win" framing instead of schedule-lookahead.',
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
            <tr><td>QB_QUALITY_GAP_WEIGHT</td><td class="num">-0.5903</td><td class="muted small">calibration only</td><td></td></tr>
            <tr><td>SKILL_EPA_OUT_WEIGHT</td><td class="num">-0.0362</td><td class="muted small">calibration only</td><td></td></tr>
            <tr><td>HFA_WEIGHT</td><td class="num">+0.9169</td><td class="muted small">moves hit rate</td><td></td></tr>
            <tr><td>MOMENTUM_WEIGHT</td><td class="num">+0.0262</td><td class="muted small">moves hit rate</td><td></td></tr>
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
