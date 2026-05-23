# Thesis-clean checkpoint-selection continuation - 2026-05-14

## Question

Short thesis-clean continuations from the accepted checkpoint often regress by
the final checkpoint. Can small continuation slices plus generic eval-based
checkpoint selection recover a useful transient checkpoint without changing the
policy representation?

Thesis boundary: clean. Training used terminal win/loss only, no heuristic step
rewards, no Spy terminal mode, no Spy candidate facts, no action-text regex
labels, and no MCTS/ISMCTS/search gates. The only curriculum is generic
deck-level Rally/Affinity oversampling from prior eval results.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-CheckpointSelect-20260514
```

Start checkpoint:

```text
Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Training registry:

```text
local-training/local_pbt/thesis_clean/20260514_checkpoint_select_train_registry.json
```

Eval registry:

```text
local-training/local_pbt/thesis_clean/20260514_checkpoint_select_eval_registry.json
```

Training settings:

```text
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
deck pool=Spy 1x, Jund 2x, Rally 6x, Affinity 6x
REFERENCE_POLICY_KL_COEF=0.25
PPO_EPSILON=0.05
ACTOR_LR=5e-5
CRITIC_LR=1e-4
OTHER_LR=2e-5
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
```

## Slice 64

Training:

```text
local-training/local_pbt/20260514_checkpoint_select_slice064_train.out.log
episodes=64
MCTS activations=0
model_latest_ep064.pt sha256=53192FA9A1BB1A1E78EC5ABCF22279FE8C319B6FBB8912459848B3E235780E96
```

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_checkpoint_select_ep064_pressure_g8
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 2 | 8 | 25.00% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Combined | 3 | 16 | 18.75% |

## Slice 128

Training:

```text
local-training/local_pbt/20260514_checkpoint_select_slice128_train.out.log
episodes=128
MCTS activations=0
model_latest_ep128.pt sha256=8E09DAB311BC26EB4834AABDEE77BFB22ECDB25731EF618D1944A393A34BD281
```

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_checkpoint_select_ep128_pressure_g8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 5 | 8 | 62.50% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Combined | 6 | 16 | 37.50% |

Reference fresh accepted pressure sample:

```text
Mono Red Rally: 5/8
Grixis Affinity: 2/8
combined: 7/16
```

## Decision

Rejected. Do not submit to HPC.

Checkpoint selection found a Rally recovery at episode 128, but Affinity stayed
at `1/8`, below the accepted fresh pressure sample. This closes small
reference-anchored hard-opponent continuation as a local quick win. The next
generic training lever should change the return horizon or policy-improvement
signal rather than only slicing the same continuation more finely.
