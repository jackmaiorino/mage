# 24h Experiment Summary

Window: 2026-05-13 16:33 ET to 2026-05-14 16:33 ET

## Executive Summary

The last 24 hours closed a large set of local thesis-clean quick-win surfaces. No branch is ready for HPC. The repeated pattern is now clear: small local updates can sometimes improve Rally or Affinity point estimates in isolation, but none preserve both hard pressure matchups while staying at or above the accepted aggregate.

The most useful new signal is diagnostic, not promotable. Fresh accepted-policy compact logs narrowed the immediate hard target to CP7 Grixis Affinity: accepted went `5/8` into Rally but only `2/8` into Affinity in the pressure sample, then `3/16` in a larger Affinity-only log sample. Those logs show a mixed Affinity failure: many games never cast Spy, Spy-cast games still usually lose, visible public-board pressure dominates, and at least one loss hard-cast `Dread Return` from hand in a low-graveyard state.

The best local continuation signal was the first tranche of isolated four-profile self-play, which improved Affinity to `3/8`, but the second tranche regressed to `2/8` Affinity and `1/8` Rally. That rejects the smoke for promotion while preserving the D-backed isolated-profile setup as useful infrastructure.

No Zaratan compute should be launched from this window's results.

## Reference Points

Accepted long-run reference:

```text
CP1: 133/256 = 51.95%
CP3: 132/256 = 51.56%
CP7: 108/242 = 44.63%
CP7 Grixis Affinity: 13/62 = 20.97%
```

Fresh accepted pressure samples from this window:

```text
Rally/Affinity pressure g8: 7/16 = 43.75%
  Mono Red Rally: 5/8 = 62.50%
  Grixis Affinity: 2/8 = 25.00%

Affinity-only compact log g16:
  Grixis Affinity: 3/16 = 18.75%
```

## Completed Results

| Surface | Best signal | Gate result | Decision |
| --- | --- | --- | --- |
| Semantic effect flags and soups | 75% semantic soup kept a small Affinity lift | CP7 `27/64 = 42.19%`; Jund `7/16` | Reject |
| Branch-trajectory policy family | Broad branch trajectory initially hit `29/64`; policy-miss tiny gate hit `10/16` | Pressure expansion: Rally `7/16`, Affinity `1/16`; Affinity-only control `1/16` | Reject family |
| Policy ensembles | Branch companion improved Rally | Branch ensemble pressure: Rally `7/16`, Affinity `1/16` | Reject |
| Critic-only terminal value | Policy frozen, value trained for 256 episodes | Repaired top-K MCTS Affinity `0/4` with 307 activations | Reject |
| Generic action-class features | Added mana/land/spell/activated/stack candidate flags | Full and policy-path variants both `2/16` pressure, Rally `0/8` | Reject |
| Accepted snapshot selection | Historical snapshots had some Rally variance | Best snapshot only `1/4`; no tested snapshot won Affinity | Reject |
| Corrected train-time MCTS aux | Root map healthy at `1824/1824` | Weighted pressure `12/68 = 17.65%` | Reject |
| Branch-pair return contrastive | Imported existing paired branch corpus | CP7 `6/16`, Affinity `0/4` | Reject |
| Branch-pair policy rank | Added generic selected-action pair-rank loss | CP7 `6/16`, Rally `0/4`, Affinity `1/4` | Reject |
| Pressure branch-pair policy rank | Balanced Rally/Affinity pair generator | Pressure `4/16`, only 11 paired episodes | Reject |
| Affinity branch-pair policy rank | 128 Affinity scenarios, 21 paired episodes | Affinity `3/16 = 18.75%` | Reject |
| Action pair-rank | Softer same-state best-vs-worst branch-return ranking | Affinity `1/16 = 6.25%` | Reject |
| Source-zone features | Added hand/graveyard/battlefield/exile/spell-not-from-hand candidate bits | Pressure `4/16`, Affinity `1/8` | Reject |
| Checkpoint-selected continuation | 128ep slice recovered Rally | Pressure `6/16`, Affinity `1/8` | Reject |
| Gamma 0.97 continuation | Shorter terminal-return horizon | Pressure `3/16`, Affinity `1/8` | Reject |
| GAE 0.99 continuation | Value-bootstrapped return estimator | Pressure `2/16`, `1/8` each | Reject |
| Multi-deck self-play smoke | Tranche 1 Affinity `3/8` | Tranche 2 pressure `3/16`, Affinity `2/8`, Rally `1/8` | Reject promotion |

## Key Diagnostics

The fresh pressure logs shifted priority away from Rally and toward Affinity. Rally was `5/8` in the accepted pressure sample, while Affinity stayed weak at `2/8` and `3/16` in the larger Affinity-only sample.

The Affinity-only log split is mixed:

```text
baseline_winrate: 3/16 = 18.75%
selected_cast_spy: 1/7 games won
no_spy_cast: 2/9 games won
selected_cast_dread_return: 0/1 games won
opponent_battlefield_count >= 10: 0/11 games won
own_graveyard_count >= 8: 2/10 games won
```

Value-head diagnostics were not uniformly bad in the Affinity-only log sample:

```text
decision_rows=1863
all AUC=0.660896
nontrivial_options AUC=0.717537
game mean_value_score AUC=0.846154
game last_value_score AUC=0.346154
```

Read: another small value-head patch is unlikely to be the highest-EV next step. The bottleneck looks more like robust policy execution under public-board pressure and opponent interaction.

## Infrastructure Added Or Fixed

- Probability-policy ensemble support for eval-only companion models.
- Trajectory import now honors `PolicyMissOnly` and target-margin filters.
- CP7 eval DB setup now defaults to real per-job H2 copies; hardlinking is opt-in with `CP7_EVAL_DB_HARDLINK=1`.
- Generic action-class candidate features behind `RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE`.
- Generic source-zone candidate features behind `RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE`.
- Generic policy pair-rank and action pair-rank loss hooks.
- Branch pair export can require baseline-losing-alternative training examples.
- Compact log exporter now parses `STATE:` rows into state summaries.
- D-backed isolated profile junction setup for multi-profile local self-play.
- Wall-clock stopping path for multi-profile local training where absolute per-profile episode targets are uneven.

## In Progress At Cutoff

`20260514_traj_pair_rank_affinity_fulltraj_collect24_stop8` was still running at the cutoff. The partial artifact at `16:32 ET` had completed 4 scenarios, skipped all 4, and produced no candidate examples or exported full-trajectory pairs yet. Treat it as not part of the completed evidence above.

### Post-Cutoff Follow-Up

The run later completed all 24 scenarios at `2026-05-14T16:49:39`. It remained sparse: 20 scenarios skipped, 4 trained scenarios, 5 candidate examples, and 0 exported winning trajectories under `baseline_losing_alternative_only=true`. The produced examples are useful diagnostic evidence, but they do not meet the failure-context corpus threshold and should not be used as a training or HPC-promotion signal.

## Recommendation

Keep HPC parked. The immediate target should remain CP7 Grixis Affinity, but not through another small representation-only or one-step/pair-rank import.

The next high-value local direction should change the data-generation mechanism, not just the loss:

```text
1. Generate Affinity-pressure data that reaches fresh live states where accepted fails, not only sparse branch-pair snapshots.
2. Preserve trajectory context through the downstream interaction, especially opponent public-board pressure.
3. Gate first on CP7 Affinity against the fresh accepted sample before spending any aggregate CP7 or HPC cycles.
```
