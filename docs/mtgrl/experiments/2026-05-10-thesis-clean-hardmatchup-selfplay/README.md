# Thesis-Clean Hard-Matchup Self-Play

Date: 2026-05-10

## Objective

Continue the May 2 thesis-clean four-profile terminal-only self-play line, but apply generic curriculum pressure toward matchups where the baseline loses.

This is intended to test whether the same RL mechanism can improve Spy without card-specific Spy teaching.

## Baseline Trigger

The 2026-05-09 May 2 clean baseline showed:

| Skill | Overall | Mirror | Wildfire | Rally | Affinity |
| --- | ---: | ---: | ---: | ---: | ---: |
| CP1 | 36.13% | 71.09% | 37.50% | 25.78% | 10.16% |
| CP3 | 34.57% | 77.34% | 28.91% | 19.53% | 12.50% |
| CP7 | 28.40% | 62.99% | 22.69% | 14.84% | 12.70% |

## Thesis Boundary

Allowed:

- Four-profile self-play continuation.
- Terminal win/loss returns only.
- Generic opponent-pool weighting based on measured weak matchups.

Explicitly disabled:

- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `ISMCTS_ROLLOUTS_PER_TURN=0`

No action-text regex filters, Spy-specific terminal modes, Spy-reachable hand pools, or selective MCTS keyword gates are used.

## Curriculum

Opponent pool: `decklist.active_profile_pool_thesis_pressure_20260509.txt`

Weights by repeated deck entries:

- Spy Combo: 1
- Jund Wildfire: 3
- Mono Red Rally: 4
- Grixis Affinity: 4

This uses deck-level difficulty from evaluation, not card-level or combo-step knowledge.

## Run

Planned local run:

- Run id: `20260510_thesis_clean_hardmatchup_selfplay_3h`
- Duration: 3 hours
- Profiles: Spy, Wildfire, Rally, Affinity value/anchor profiles
- Runners: 64 total
- Eval: skipped during training; post-run eval will use the clean eval registry and C-drive eval output root

## Status

Training complete.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 200,919 | 212,763 | 11,844 |
| Pauper-Wildfire-Value | 183,921 | 194,248 | 10,327 |
| Pauper-Rally-Anchor-Value | 272,056 | 294,388 | 22,332 |
| Pauper-Affinity-Anchor-Value | 184,344 | 200,449 | 16,105 |

Total delta: 60,608 episodes in 3 hours.

Post-run evaluation is in progress.

### Post-Run CP1, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_3h_spy_cp1_eval64`

Overall: `129/256 = 50.39%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 46 | 64 | 71.88% |
| Jund Wildfire | 32 | 64 | 50.00% |
| Mono Red Rally | 33 | 64 | 51.56% |
| Grixis Affinity | 18 | 64 | 28.12% |

Compared with the May 2 clean CP1 baseline, this is +14.26 pp overall on a half-size sample, with the biggest gains in Rally and Affinity.

### Post-Run CP3, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_3h_spy_cp3_eval64`

Overall: `114/256 = 44.53%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 47 | 64 | 73.44% |
| Jund Wildfire | 30 | 64 | 46.88% |
| Mono Red Rally | 17 | 64 | 26.56% |
| Grixis Affinity | 20 | 64 | 31.25% |

Compared with the May 2 clean CP3 baseline, this is +9.96 pp overall on a half-size sample. Affinity improved materially; Rally is only modestly above baseline and remains the least stable matchup.

### Post-Run CP7, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_3h_spy_cp7_eval64`

Overall: `99/256 = 38.67%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 45 | 64 | 70.31% |
| Jund Wildfire | 23 | 64 | 35.94% |
| Mono Red Rally | 21 | 64 | 32.81% |
| Grixis Affinity | 10 | 64 | 15.62% |

Compared with the May 2 clean CP7 baseline, this is +10.27 pp overall on a half-size sample. The improvement is broad except Affinity, where the winrate remains low despite a small gain.

## Interpretation

This is the first clear thesis-clean improvement since the May 1-2 run. The training did not use Spy-specific labels, Spy-specific state features, Spy-specific terminal modes, action text filters, hand pools, or search gates. The only curriculum intervention was generic deck-level hard-matchup oversampling from evaluation results.

The next experiment should scale the same run before changing mechanisms. The open question is whether the Affinity and Rally gains continue with more terminal-only self-play, or whether this 3-hour jump is mostly early adaptation that saturates.

## Continuation

Started follow-up run `20260510_thesis_clean_hardmatchup_selfplay_6h_cont` from the improved 3-hour checkpoint with the same thesis-clean registry and flags. Planned duration: 6 hours.

6-hour continuation complete.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 212,763 | 240,891 | 28,128 |
| Pauper-Wildfire-Value | 194,248 | 215,244 | 20,996 |
| Pauper-Rally-Anchor-Value | 294,388 | 336,336 | 41,948 |
| Pauper-Affinity-Anchor-Value | 200,449 | 237,009 | 36,560 |

Total continuation delta: 127,632 episodes in 6 hours.

Post-continuation evaluation is in progress.

### 9h-Total CP1, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_9h_total_spy_cp1_eval64`

Overall: `121/256 = 47.27%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 49 | 64 | 76.56% |
| Jund Wildfire | 31 | 64 | 48.44% |
| Mono Red Rally | 20 | 64 | 31.25% |
| Grixis Affinity | 21 | 64 | 32.81% |

This remains above the May 2 CP1 baseline but is below the 3-hour checkpoint overall, mainly from Rally regression.

### 9h-Total CP3, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_9h_total_spy_cp3_eval64`

Overall: `123/256 = 48.05%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 51 | 64 | 79.69% |
| Jund Wildfire | 32 | 64 | 50.00% |
| Mono Red Rally | 24 | 64 | 37.50% |
| Grixis Affinity | 16 | 64 | 25.00% |

This is above both the May 2 CP3 baseline and the 3-hour checkpoint. Rally improved from the 3-hour CP3 result; Affinity regressed from the 3-hour CP3 result but remains above the May 2 baseline.

### 9h-Total CP7, 64 Games Requested Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_hardmatchup_selfplay_9h_total_spy_cp7_eval64`

Overall: `113/254 = 44.49%`

Two long jobs returned 7 games before timeout, so totals are 254 rather than 256.

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 47 | 63 | 74.60% |
| Jund Wildfire | 24 | 64 | 37.50% |
| Mono Red Rally | 27 | 64 | 42.19% |
| Grixis Affinity | 15 | 63 | 23.81% |

This is above both the May 2 CP7 baseline and the 3-hour checkpoint. The remaining bottleneck is Affinity.

## Next Experiment

Run another thesis-clean continuation with the same mechanism but a refreshed generic opponent-pool curriculum based on the 9h CP7 result. Keep Spy 1x, Wildfire 3x, Rally 3x, and increase Affinity to 6x because it is now the clear worst matchup.

Registry: `local-training/local_pbt/thesis_clean/20260510_thesis_clean_affinity_pressure_selfplay_registry.json`

Planned run id: `20260510_thesis_clean_affinity_pressure_3h`

## Affinity-Pressure 3h Continuation

Run `20260510_thesis_clean_affinity_pressure_3h` continued from the 9h checkpoint with a thesis-clean weighted opponent pool:

- Spy Combo 1x
- Jund Wildfire 3x
- Mono Red Rally 3x
- Grixis Affinity 6x

The run used terminal rewards only and kept the Spy-specific aids disabled: no action-facts features, no Spy terminal mode, no regex labels, no MCTS/ISMCTS gates.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 240,891 | 253,716 | 12,825 |
| Pauper-Wildfire-Value | 215,244 | 224,151 | 8,907 |
| Pauper-Rally-Anchor-Value | 336,336 | 353,869 | 17,533 |
| Pauper-Affinity-Anchor-Value | 237,009 | 253,786 | 16,777 |

Total delta: 56,042 episodes in 3 hours.

### Weighted Eval Mistake

The first post-run CP1 command used the weighted training deck pool as the eval opponent pool, so it is not comparable to the earlier 64-games-per-unique-matchup sweeps.

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp1_eval64`

Weighted result: `314/744 = 42.20%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 47 | 64 | 73.44% |
| Jund Wildfire | 93 | 176 | 52.84% |
| Mono Red Rally | 74 | 192 | 38.54% |
| Grixis Affinity | 100 | 312 | 32.05% |

Corrected evaluation uses `local-training/local_pbt/thesis_clean/20260510_thesis_clean_affinity_pressure_spy_eval_registry.json`, which points to a one-copy-per-opponent deck list.

### Affinity-Pressure CP1, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp1_unique_eval64`

Overall: `133/256 = 51.95%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 49 | 64 | 76.56% |
| Jund Wildfire | 31 | 64 | 48.44% |
| Mono Red Rally | 22 | 64 | 34.38% |
| Grixis Affinity | 31 | 64 | 48.44% |

This is above the 9h CP1 result, with the intended Affinity improvement.

### Affinity-Pressure CP3, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp3_unique_eval64`

Overall: `132/256 = 51.56%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 54 | 64 | 84.38% |
| Jund Wildfire | 29 | 64 | 45.31% |
| Mono Red Rally | 24 | 64 | 37.50% |
| Grixis Affinity | 25 | 64 | 39.06% |

This is above the 9h CP3 result, again mostly from Affinity and mirror gains.

### Affinity-Pressure CP7, 64 Games Requested Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp7_unique_eval64`

Overall: `108/242 = 44.63%`

Some long CP7 jobs returned fewer than 8 games, and one mirror chunk returned 0 games.

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 43 | 56 | 76.79% |
| Jund Wildfire | 36 | 60 | 60.00% |
| Mono Red Rally | 16 | 64 | 25.00% |
| Grixis Affinity | 13 | 62 | 20.97% |

The result preserves the overall 9h CP7 level but shifts strength: Wildfire improves sharply, while Rally and Affinity remain the CP7 bottlenecks.

### HPC Note

A 15-minute A100 smoke job was queued from a mid-run checkpoint to validate the Zaratan path, but it stayed pending with zero allocation. It was canceled before start:

- Job: `19245248`
- Final state: `CANCELLED`
- Elapsed: `00:00:00`
- Allocated CPUs: `0`

No kSU were spent.

## Next Experiment: Rally+Affinity CP7 Pressure

The next thesis-clean continuation should target the two remaining CP7 hard matchups without adding card-specific knowledge. Use another generic deck-level curriculum with Spy 1x, Wildfire 2x, Rally 6x, and Affinity 6x. Keep the same terminal-only training flags and compare against the corrected Affinity-pressure checkpoint above.

## Rally+Affinity-Pressure 3h Continuation

Run `20260510_thesis_clean_rally_affinity_pressure_3h` continued from the Affinity-pressure checkpoint with a thesis-clean weighted opponent pool:

- Spy Combo 1x
- Jund Wildfire 2x
- Mono Red Rally 6x
- Grixis Affinity 6x

The run remained thesis-clean: terminal rewards only, no action-facts features, no Spy terminal mode, no regex labels, and no MCTS/ISMCTS gates.

The wrapper stopped cleanly at `MAX_WALL_SECONDS=10800`, but did not write `episodes_after.csv`; after-counts were reconstructed from each profile's `training_stats.csv`.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 253,716 | 265,728 | 12,012 |
| Pauper-Wildfire-Value | 224,151 | 233,276 | 9,125 |
| Pauper-Rally-Anchor-Value | 353,869 | 370,930 | 17,061 |
| Pauper-Affinity-Anchor-Value | 253,786 | 269,620 | 15,834 |

Total delta: 54,032 episodes in 3 hours.

### Rally+Affinity-Pressure CP1, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260510_rally_affinity_pressure_3h_spy_cp1_unique_eval64`

Overall: `119/256 = 46.48%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 52 | 64 | 81.25% |
| Jund Wildfire | 29 | 64 | 45.31% |
| Mono Red Rally | 20 | 64 | 31.25% |
| Grixis Affinity | 18 | 64 | 28.12% |

This regressed from the Affinity-pressure CP1 result (`133/256 = 51.95%`), especially against Affinity.

### Rally+Affinity-Pressure CP3, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260510_rally_affinity_pressure_3h_spy_cp3_unique_eval64`

Overall: `117/256 = 45.70%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 47 | 64 | 73.44% |
| Jund Wildfire | 28 | 64 | 43.75% |
| Mono Red Rally | 26 | 64 | 40.62% |
| Grixis Affinity | 16 | 64 | 25.00% |

This regressed from the Affinity-pressure CP3 result (`132/256 = 51.56%`).

### Rally+Affinity-Pressure CP7, 64 Games Requested Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260510_rally_affinity_pressure_3h_spy_cp7_unique_eval64`

Overall: `106/255 = 41.57%`

One Wildfire chunk returned 15 games, so total games are 255 rather than 256.

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 52 | 64 | 81.25% |
| Jund Wildfire | 19 | 63 | 30.16% |
| Mono Red Rally | 19 | 64 | 29.69% |
| Grixis Affinity | 16 | 64 | 25.00% |

Rally and Affinity CP7 improved slightly from the Affinity-pressure checkpoint, but the gain was too small and came with a large Wildfire collapse plus CP1/CP3 regression. Do not promote this checkpoint.

Operational note: the first CP1 attempt exposed a shared-GPU startup race where the metrics endpoint was live before the service answered `getDeviceInfo`; `scripts/run_cp7_eval_sweep.py` now waits briefly after metrics readiness before launching the first JVM.

## Next Experiment: Moderate Stabilization Pressure

Reject the Rally+Affinity-pressure checkpoint and restore active models from `local-training/local_pbt/model_backups/pre_20260510_thesis_clean_rally_affinity_pressure_3h`, which is the stronger Affinity-pressure checkpoint.

The next thesis-clean run should test a less aggressive hard-matchup mix to see whether Rally/Affinity CP7 can be improved without catastrophic Wildfire forgetting:

- Spy Combo 1x
- Jund Wildfire 3x
- Mono Red Rally 4x
- Grixis Affinity 5x

Run locally first. Do not spend HPC on this branch until it beats the Affinity-pressure checkpoint on corrected unique CP1/CP3/CP7.

## Moderate-Pressure 3h Continuation

Run `20260510_thesis_clean_moderate_pressure_3h` restored the active models to `local-training/local_pbt/model_backups/pre_20260510_thesis_clean_rally_affinity_pressure_3h` first, then continued from the stronger Affinity-pressure checkpoint with:

- Spy Combo 1x
- Jund Wildfire 3x
- Mono Red Rally 4x
- Grixis Affinity 5x

The run remained thesis-clean: terminal rewards only, no action-facts features, no Spy terminal mode, no regex labels, and no MCTS/ISMCTS gates.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 265,728 | 278,387 | 12,659 |
| Pauper-Wildfire-Value | 233,276 | 242,153 | 8,877 |
| Pauper-Rally-Anchor-Value | 370,930 | 388,723 | 17,793 |
| Pauper-Affinity-Anchor-Value | 269,620 | 286,109 | 16,489 |

Total delta: 55,818 episodes in 3 hours.

### Moderate-Pressure CP1, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_moderate_pressure_3h_spy_cp1_unique_eval64`

Overall: `128/256 = 50.00%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 53 | 64 | 82.81% |
| Jund Wildfire | 31 | 64 | 48.44% |
| Mono Red Rally | 24 | 64 | 37.50% |
| Grixis Affinity | 20 | 64 | 31.25% |

This partially recovers from the rejected 6x/6x branch but remains below the Affinity-pressure CP1 result (`133/256 = 51.95%`).

### Moderate-Pressure CP3, 64 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_moderate_pressure_3h_spy_cp3_unique_eval64`

Overall: `107/256 = 41.80%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 51 | 64 | 79.69% |
| Jund Wildfire | 30 | 64 | 46.88% |
| Mono Red Rally | 17 | 64 | 26.56% |
| Grixis Affinity | 9 | 64 | 14.06% |

This is a clear regression from the Affinity-pressure CP3 result (`132/256 = 51.56%`).

### Moderate-Pressure CP7, 64 Games Requested Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_moderate_pressure_3h_spy_cp7_unique_eval64`

Overall: `106/254 = 41.73%`

One Wildfire chunk returned 14 games, so total games are 254 rather than 256.

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 55 | 64 | 85.94% |
| Jund Wildfire | 21 | 62 | 33.87% |
| Mono Red Rally | 13 | 64 | 20.31% |
| Grixis Affinity | 17 | 64 | 26.56% |

This is also below the Affinity-pressure CP7 result (`108/242 = 44.63%`).

Conclusion: do not promote. Two deck-weight-only continuations after the Affinity-pressure checkpoint failed to improve the accepted result. The next experiment should leave deck weighting alone and test a generic representation or generic search change instead.
