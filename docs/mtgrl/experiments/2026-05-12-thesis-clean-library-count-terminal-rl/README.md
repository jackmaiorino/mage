# Thesis-Clean Library-Count Terminal RL

Date: 2026-05-12

## Question

Can a narrow, generic library-count representation improve thesis-clean terminal-RL training without encoding Spy card names or combo rules?

This branch tests a smaller version of the broader zone-count idea. The encoder receives only own-library land and creature counts in existing generic player-stat feature slots. No card names, Spy-specific candidate facts, hidden combo labels, heuristic rewards, or selective MCTS gates are enabled.

## Code Change

Added `RL_LIBRARY_COUNT_FEATURES_ENABLE=1` in `StateSequenceBuilder`.

When enabled:

```text
playerStats[19] = own library land count
playerStats[20] = own library creature count
```

The broader `RL_ZONE_COUNT_FEATURES_ENABLE` path remains separate and fills the larger zone-count block. This branch intentionally keeps the representation narrow after the broader zone-count branch failed reduced evals.

## Clone Setup

Registry:

```text
local-training/local_pbt/thesis_clean/20260512_thesis_clean_library_count_selfplay_registry.json
```

Profiles:

```text
Pauper-Spy-Combo-Value-LibCount-20260512
Pauper-Wildfire-Value-LibCount-20260512
Pauper-Rally-Anchor-Value-LibCount-20260512
Pauper-Affinity-Anchor-Value-LibCount-20260512
```

Each clone starts from the accepted active checkpoint for its deck. The new feature columns `19:21` in `input_proj.weight` and `critic_input_proj.weight` were zeroed before training, and optimizer state was removed, so the branch does not inherit stale optimizer moments for newly exposed inputs.

## Training Settings

```text
TRAIN_PROFILES=4
OPPONENT_SAMPLER=self
SELFPLAY_OPPONENT_TRAINING=1
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_ZONE_COUNT_FEATURES_ENABLE=0
RL_LIBRARY_COUNT_FEATURES_ENABLE=1
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
PY_SERVICE_MODE=hybrid
USE_TRT_INFERENCE=1
```

This is terminal win/loss learning only.

## Status

Rejected after the first reduced gate.

The local orchestrator was interrupted by a tool timeout, leaving the Java trainer orphaned after the shared GPU host exited. Training-loss files stopped updating around `2026-05-12T04:21:07Z`; the orphaned trainer then emitted failed shared-GPU batch flush warnings. The orphan was stopped manually and archived:

```text
local-training/local_pbt/trainer_logs/20260512T0040Z_library_count_terminal_orphan_gpu_dead.log
```

Usable Spy checkpoint:

```text
profile: Pauper-Spy-Combo-Value-LibCount-20260512
episode: 335
model.pt SHA-256: 5E34C238ED5F
model_latest.pt SHA-256: 972353D24EA9
```

Final clone counters at stop:

| Profile | Episode | Recent self-play winrate |
| --- | ---: | ---: |
| Spy | 335 | 17.5% |
| Wildfire | 240 | 33.0% |
| Rally | 487 | 66.0% |
| Affinity | 478 | 46.5% |

Because the run was interrupted, this checkpoint is only a partial-run screen, not a clean 500-episode branch.

Reduced CP1 screen:

```text
20260512_library_count_terminal_rl_partial335_spy_cp1_unique_eval16
30/64 = 46.88%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 11 | 16 | 68.75% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 3 | 16 | 18.75% |

Result: failed the CP1 screen. CP7 was not run.

## Decision Rule

Do not promote or spend HPC on this branch unless it beats or ties the accepted Spy profile on reduced CP1 and CP7 gates. If it passes both, run a larger CP1/CP3/CP7 confirmation before any HPC scaling.

## Verdict

Do not promote and do not rerun this exact branch on HPC.

The partial run already failed CP1, and its weakest matchups are the same pressure decks that the accepted checkpoint is trying to improve: Rally and Affinity. The narrow library-count representation remains thesis-clean, but local evidence says representation-only branches are not the next high-EV path.

Next direction: move to generic search-policy improvement with better search targets or cheaper/batched value evaluation. Keep Zaratan reserved for a mechanism that first clears reduced local CP1 and CP7 gates.
