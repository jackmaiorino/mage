# Thesis-Clean Hard-Skill Terminal Control

Date: 2026-05-11

## Question

Did the sparse-MCTS branches regress CP7 because of generic train-time search, or because the hard-skill opponent mix itself destabilizes the accepted checkpoint?

This is a control against the hard-skill sparse-MCTS branch. It keeps the same opponent mix and disables train-time MCTS.

## Setup

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_hardskill_terminal_control_200ep/phase_001_train`

Backup: `local-training/local_pbt/model_backups/pre_20260511_thesis_clean_hardskill_terminal_control_200ep`

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- one active Spy profile
- `NUM_GAME_RUNNERS=8`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- `RL_ZONE_COUNT_FEATURES_ENABLE=0`
- `OPPONENT_SAMPLER=hybrid`
- `SKILL_MIX=1:0.25,3:0.25,7:0.50`
- `HYBRID_SELFPLAY_P=0.50`

No action-text regex labels, Spy terminal modes, Spy-reachable hand pools, action-facts features, or selective MCTS keyword gates were used.

## Training Result

- parsed training rows: `199`
- training wins: `56/199`
- self-play: `27/105`
- CP1: `7/24`
- CP3: `7/19`
- CP7: `15/51`
- average parsed game seconds: `25.57`
- `MCTS_GATE: total=0 ... activations=0`

## Reduced CP7 Eval

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260511_terminal_hardskill_200ep_spy_cp7_unique_eval16`

Overall: `25/63 = 39.68%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 7 | 15 | 46.67% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 3 | 16 | 18.75% |

## Decision

Rejected.

The same hard-skill mix still lands below the accepted checkpoint's corrected unique CP7 result (`108/242 = 44.63%`) even with train-time MCTS disabled. Sparse MCTS made the branch slower and did not help, but the stronger conclusion is that short 200-episode hard-skill continuations from this checkpoint are not improving the CP7 gate.

Restore the accepted checkpoint from `pre_20260511_thesis_clean_hardskill_terminal_control_200ep`.

## Next

Do not spend HPC on sparse MCTS or hard-skill 200-episode continuations.

Next local experiment should return to a thesis-clean generic mechanism that is not just more opponent pressure. The highest-value options are:

- a longer terminal-only continuation at the already accepted Affinity-pressure distribution, gated by reduced CP7 before any HPC spend;
- a generic uncertainty-triggered search target instead of random sparse MCTS sampling;
- a generic representation change that does not name Spy cards or combo pieces.
