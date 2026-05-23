# Thesis-Clean Generic Branch-Return Q

Date: 2026-05-12

## Question

Can generic terminal branch returns train a candidate-Q reranker without naming Spy cards or directly changing the policy target?

This differs from earlier branch-Q work: no action-text regex, no Spy terminal mode, no Spy hand pool, no action-facts features, and no heuristic rewards.

## Wrapper Change

`scripts/run_action_counterfactual.ps1` now exposes existing trainer flags:

```text
-GenericBranchOrder
-NoSearchModelScoring
```

The experiment uses `-GenericBranchOrder`, which disables the old card-name tactic branch ordering and branches by model probability plus generic pass-last ordering.

Verification:

```text
PowerShell parser check for scripts/run_action_counterfactual.ps1
```

## Failed Local-Mode Collection

Run:

```text
20260512_thesis_clean_generic_branchret_collect64_cp7
```

Result:

```text
completed_scenarios=96
candidate_examples=0
```

The local Py4J scoring channel repeatedly failed with `Candidate scoring failed` / `Error while obtaining a new communication channel`, so this run produced no usable data.

## ONNX CP7 Collection

Run:

```text
20260512_thesis_clean_generic_branchret_onnx_collect64_cp7
```

Settings:

```text
ServiceMode=onnx
ONNX_FORCE_CPU=1
opponent=cp7
terminal_mode=WIN
branch_return_targets=true
generic_branch_order=true
max_decision_depth=5
top_k=2
random_extra=1
max_game_turns=12
```

Result:

```text
completed_scenarios=41
candidate_examples=66
rows_with_positive_target=46
rows_with_negative_target=48
mixed_positive_negative_rows=28
all_loss_rows=20
best_pass_rows=6
```

Matchup balance:

| Opponent | Examples |
| --- | ---: |
| Spy Combo | 30 |
| Mono Red Rally | 24 |
| Jund Wildfire | 9 |
| Grixis Affinity | 3 |

## Branch Q64 Clone

Fresh clone:

```text
Pauper-Spy-Combo-Value-GenericBranchQ64-20260512
```

Training run:

```text
20260512_thesis_clean_generic_branchq64_import_e8p2_signed
```

Settings:

```text
CANDIDATE_Q_ONLY=1
CANDIDATE_Q_LOSS_COEF=1.0
CANDIDATE_Q_MCTS_SIGNED_TARGETS=1
CANDIDATE_Q_BLEND=0.0
train_epochs=8
candidate_permutations=2
```

Result:

```text
importedTrainingExamples=66
trainPassSamples=1056
train_steps=40
```

## Reduced CP7 Screens

Blend `0.25`:

```text
20260512_generic_branchq64_blend025_spy_cp7_unique_eval16
27/64 = 42.19%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 6 | 16 | 37.50% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 3 | 16 | 18.75% |

Blend `0.10`:

```text
20260512_generic_branchq64_blend010_spy_cp7_unique_eval16
26/64 = 40.62%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 5 | 16 | 31.25% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 5 | 16 | 31.25% |

## Verdict

Q64 rejected. Do not run CP1 and do not spend HPC for this clone.

The mechanism is wired, but the first dataset is badly skewed away from Affinity and still fails the CP7 gate. The next local branch should collect balanced hard-matchup branch-return data first, then repeat the same Q-only fit.

## Balanced CP7 Collection

Balanced collection targeted the hard CP7 matchups without card-name or Spy-specific filters.

| Source | Run | Examples | Positive Rows | Negative Rows | Mixed Rows | All-Loss Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | `20260512_thesis_clean_generic_branchret_onnx_collect64_cp7` | 66 | 46 | 48 | 28 | 20 |
| Affinity | `20260512_thesis_clean_generic_branchret_onnx_affinity48_cp7` | 49 | 27 | 42 | 20 | 22 |
| Rally | `20260512_thesis_clean_generic_branchret_onnx_rally48_cp7` | 49 | 27 | 45 | 23 | 22 |
| Wildfire | `20260512_thesis_clean_generic_branchret_onnx_wildfire32_cp7` | 33 | 16 | 31 | 14 | 17 |

Combined import directory:

```text
local-training/local_pbt/action_counterfactual/20260512_thesis_clean_generic_branchret_balanced_cp7
```

## Branch Q Balanced Clone

Fresh clone:

```text
Pauper-Spy-Combo-Value-GenericBranchQBalanced-20260512
```

Training run:

```text
20260512_thesis_clean_generic_branchqbalanced_import_e6p2_signed
```

Settings:

```text
CANDIDATE_Q_ONLY=1
CANDIDATE_Q_LOSS_COEF=1.0
CANDIDATE_Q_MCTS_SIGNED_TARGETS=1
CANDIDATE_Q_BLEND=0.0
train_epochs=6
candidate_permutations=2
```

Result:

```text
importedTrainingExamples=197
trainPassSamples=2364
train_steps=78
```

## Balanced Reduced CP7 Screen

Blend `0.25`:

```text
20260512_generic_branchqbalanced_blend025_spy_cp7_unique_eval16
17/64 = 26.56%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 10 | 16 | 62.50% |
| Jund Wildfire | 5 | 16 | 31.25% |
| Mono Red Rally | 1 | 16 | 6.25% |
| Grixis Affinity | 1 | 16 | 6.25% |

## Balanced Verdict

Rejected. The balanced dataset made the reranker substantially worse, especially into Rally and Affinity. Do not run the `0.10` blend, CP1, or HPC for branch-return Q-only reranking.

The likely failure is not just matchup imbalance. Offline candidate-Q-only training is moving the eval-time action ranker without retraining the policy/value distribution that creates the state visitation. The next thesis-clean experiment should integrate generic branch returns into online terminal RL as an auxiliary loss or move to a deeper generic method such as belief/determinization, rather than collecting more branch-Q reranker datasets.
