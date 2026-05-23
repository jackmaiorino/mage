# Thesis-Clean Root Mulligan Prefix Search

Date: 2026-05-12

## Question

Can generic terminal prefix search improve the root keep/mulligan decision for hard pressure matchups without touching Spy-specific action logic?

This branch forces only root keep/mulligan candidates, lets the normal policy play out, and labels the root choice by terminal win/loss search. If no winning root branch is found, `-TrainRootMulliganOnNoWin` labels mulligan. Training is `DISTILL_HEAD_ONLY`, so only policy scorer heads are trainable and the shared encoder/critic stay fixed.

## Collection

Affinity CP7:

```text
20260512_root_mulligan_prefix_affinity_cp7_collect
examples=16
winning_trajectories=10
policy_misses_by_baseline_vs_target=1
targets: 11 mulligan, 5 keep
elapsed_sec=589.5
```

Rally CP7:

```text
20260512_root_mulligan_prefix_rally_cp7_collect
examples=16
winning_trajectories=6
policy_misses_by_baseline_vs_target=3
targets: 12 mulligan, 4 keep
elapsed_sec=117.5
```

Settings:

```text
terminal_mode=WIN
opponent=cp7
cp7_skill=7
train_root_mulligan_only=true
train_root_mulligan_on_no_win=true
action_types=MULLIGAN
generic_branch_order=true
max_prefix_depth=1
max_search_nodes=7
no action-text regex
no Spy terminal mode
no Spy hand pool
```

## Clone

Fresh clone:

```text
Pauper-Spy-Combo-Value-RootMullPrefix-20260512
```

Source:

```text
accepted Pauper-Spy-Combo-Value checkpoint
```

## Training

Run:

```text
20260512_root_mulligan_prefix_headonly_train32_e6p2
```

Settings:

```text
import_training_data_path=local-training/local_pbt/generic_prefix_data/20260512_root_mulligan_cp7_hard
action_types=MULLIGAN
DISTILL_HEAD_ONLY=1
train_epochs=6
candidate_permutations=2
MCTS_KL_LOSS_COEF=1.0
MCTS_TARGET_POLICY_MIX=1.0
LOAD_OPTIMIZER_STATE=0
RESET_TRAINING_STATE_ON_LOAD=1
```

Result:

```text
importedTrainingExamples=32
trainPassSamples=384
train_steps=12
```

Clone hashes after training:

```text
model=343A58E2B4A4
model_latest=343A58E2B4A4
```

Accepted active profile hashes stayed unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Reduced CP7 Screen

Run:

```text
20260512_root_mulligan_prefix_headonly_spy_cp7_unique_eval16
20/63 = 31.75%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 9 | 16 | 56.25% |
| Jund Wildfire | 3 | 15 | 20.00% |
| Mono Red Rally | 6 | 16 | 37.50% |
| Grixis Affinity | 2 | 16 | 12.50% |

## Verdict

Rejected. Do not run CP1 or HPC, and do not expand this root-mulligan label set.

The root labels were mostly consistent with the accepted policy and the small head-only update weakened the mirror and Wildfire while leaving Affinity broken. The hard-pressure failure is not explained by root keep/mulligan alone; the post-mulligan action policy still cannot convert enough hard-matchup states.
