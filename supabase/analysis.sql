-- End-of-season analysis: how the "take N dogs" model held up vs. your picks vs. reality.
-- Run in the Supabase SQL editor after the season. Reads budget_snapshot (frozen weekly),
-- pickem_pick / ats_pick (your picks), and game (final scores).
--
-- Per week x mode x spread bin it returns:
--   model_take    - dogs the model said to take in that bin
--   you_took_dog  - dogs you actually picked in that bin
--   dogs_hit      - dogs that actually won (ml) / covered (ats)
--   flagged_n     - games the model checked as the ones to fade
--   flagged_hit   - of those, how many the dog actually came through

with snap_games as (
  select bs.week, bs.mode, bs.bin, bs.suggested,
         (gg->>'game_id')            as game_id,
         (gg->>'fav')                as fav,
         (gg->>'dog')                as dog,
         (gg->>'dog_home')::boolean  as dog_home,
         coalesce((gg->>'flagged')::boolean, false) as flagged
  from budget_snapshot bs
       cross join lateral jsonb_array_elements(bs.games) gg
  where bs.bin <> 'TOTAL'
),
outcome as (
  select sg.*,
         g.home_score, g.away_score, o.spread,
         case
           when g.home_score is null then null
           when sg.mode = 'ml' then
             case when sg.dog_home then g.home_score > g.away_score
                  else g.away_score > g.home_score end
           else  -- ats: does the dog beat the number?
             case when sg.dog_home
                  then (g.home_score - g.away_score) + abs(coalesce(o.spread, 0)) > 0
                  else (g.away_score - g.home_score) + abs(coalesce(o.spread, 0)) > 0 end
         end as dog_hit
  from snap_games sg
       join game g on g.game_id = sg.game_id
       left join odds o on o.game_id = sg.game_id
),
picks as (
  select 'ml'::text  as mode, week, game_id, pick from pickem_pick
  union all
  select 'ats'::text as mode, week, game_id, pick from ats_pick
)
select o.week, o.mode, o.bin,
       max(o.suggested)                                       as model_take,
       count(*) filter (where p.pick is not null
                          and p.pick = o.dog)                 as you_took_dog,
       count(*) filter (where o.dog_hit)                      as dogs_hit,
       count(*) filter (where o.flagged)                      as flagged_n,
       count(*) filter (where o.flagged and o.dog_hit)        as flagged_hit,
       count(*)                                               as games_in_bin
from outcome o
     left join picks p on p.mode = o.mode and p.week = o.week and p.game_id = o.game_id
group by o.week, o.mode, o.bin
order by o.week, o.mode,
         array_position(array['≤2.5','3','3.5–6','6.5–9.5','10+'], o.bin);

-- Season roll-up: model calibration by bin (did "take N" match how many dogs hit?)
--   select mode, bin,
--          sum(model_take) as model_said,
--          sum(dogs_hit)   as actually_hit,
--          sum(you_took_dog) as you_took,
--          round(sum(flagged_hit)::numeric / nullif(sum(flagged_n),0), 3) as flagged_hit_rate
--   from ( <the query above> ) t
--   group by mode, bin
--   order by mode, array_position(array['≤2.5','3','3.5–6','6.5–9.5','10+'], bin);
