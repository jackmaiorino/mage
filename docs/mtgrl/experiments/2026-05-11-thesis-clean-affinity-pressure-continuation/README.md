# Thesis-Clean Affinity-Pressure Continuation

Date: 2026-05-11

## Question

Does simply continuing the accepted thesis-clean Affinity-pressure line improve the CP7 gate, or was that checkpoint a local peak?

This stays within the thesis boundary:

- terminal win/loss returns only
- four-profile training
- no Spy action-facts features
- no Spy terminal mode
- no regex labels
- no Spy-reachable hand pools
- no MCTS or ISMCTS

## Setup

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_affinity_pressure_continue_1h/phase_001_train`

Backup: `local-training/local_pbt/model_backups/pre_20260511_thesis_clean_affinity_pressure_continue_1h`

Registry: `local-training/local_pbt/thesis_clean/20260510_thesis_clean_affinity_pressure_selfplay_registry.json`

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- profiles: Spy, Wildfire, Rally, Affinity
- `NUM_GAME_RUNNERS=64`
- `MAX_WALL_SECONDS=3600`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`

Opponent pool:

- Spy Combo 1x
- Jund Wildfire 3x
- Mono Red Rally 3x
- Grixis Affinity 6x

## Training Result

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 293087 | 297941 | 4854 |
| Pauper-Wildfire-Value | 251542 | 254898 | 3356 |
| Pauper-Rally-Anchor-Value | 407952 | 414897 | 6945 |
| Pauper-Affinity-Anchor-Value | 304362 | 310878 | 6516 |

## Reduced CP7 Eval

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260511_affinity_pressure_continue_1h_spy_cp7_unique_eval16`

Overall: `24/64 = 37.50%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 1 | 16 | 6.25% |

## Decision

Rejected.

The accepted checkpoint remains stronger (`108/242 = 44.63%` corrected unique CP7). A short continuation on the same deck distribution does not preserve the Affinity/Rally gains; it collapses Affinity to `1/16`.

Restore active models from `pre_20260511_thesis_clean_affinity_pressure_continue_1h`.

## Next

Do not spend HPC on plain continuation from this checkpoint.

The local evidence now rejects:

- short hard-skill continuations;
- short sparse train-time MCTS continuations;
- short plain continuation on the accepted Affinity-pressure mix;
- zone-count continuation on this line.

The next experiment should change the generic learning signal rather than only the opponent distribution. Candidate: add a generic checkpoint-selection gate during continuation so a bad short branch can roll back at smaller intervals, or improve generic search target quality before another training branch.
