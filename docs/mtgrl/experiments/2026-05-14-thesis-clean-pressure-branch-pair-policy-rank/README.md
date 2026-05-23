# Thesis-clean pressure branch-pair policy rank - 2026-05-14

## Question

The first policy-pair rank import failed on the existing broad branch-pair
corpus. Was that mainly because the corpus was not focused on actual hard
pressure mistakes?

This experiment tightens pair collection to Rally/Affinity pressure states and
requires each exported pair to come from a generic baseline-losing-alternative
decision point.

## Implementation

`ActionCounterfactualTrainer` now lets
`--branch-trajectory-require-training-example` apply to pair-mode exports too.
With pair mode plus `--baseline-losing-alternative-only`, a paired trajectory is
exported only when the same decision also produced a generic training example:
the accepted baseline branch loses and a sibling branch wins.

This stays thesis-clean:

- terminal `WIN` labels only;
- no card-name/action-text filters;
- no Spy terminal mode;
- no heuristic rewards;
- no selective search keywords.

## Collection

Pressure deck pool:

```text
local-training/local_pbt/thesis_clean/20260514_rally_affinity_pressure_pool.txt
```

Run:

```text
local-training/local_pbt/action_counterfactual/20260514_branch_pair_pressure_baselinealt_collect64_stop24
```

Settings:

```text
scenarios=64
opponent=cp7
opponent_pool=Mono Red Rally x2, Grixis Affinity x2
max_decision_depth=6
top_k=4
random_extra=1
baseline_losing_alternative_only=true
branch_trajectory_mode=true
branch_trajectory_first_post_target_only=true
branch_trajectory_pair_mode=true
branch_trajectory_require_training_example=true
generic_branch_order=true
```

Result:

```text
completed_scenarios=64
trained_scenarios=13
candidate_examples=19
serialized paired episodes=11
serialized paired steps=22
```

Candidate examples by opponent:

| Opponent | Examples |
| --- | ---: |
| Grixis Affinity | 10 |
| Mono Red Rally | 9 |

## Import

Profile:

```text
Pauper-Spy-Combo-Value-BranchPairPolicyRankPressure-20260514
```

Import run:

```text
local-training/local_pbt/action_counterfactual/20260514_branch_pair_policy_rank_pressure_e8
```

Settings:

```text
train_epochs=8
imported_trajectory_episodes=11
imported_trajectory_steps=22
train_pass_samples=176
POLICY_PAIR_RANK_LOSS_COEF=0.10
POLICY_PAIR_RANK_MARGIN=0.20
REFERENCE_POLICY_KL_COEF=1.0
POLICY_LOSS_COEF=0
VALUE_LOSS_COEF=0
MCTS_KL_LOSS_COEF=0
```

Resulting checkpoint:

```text
model_latest.pt sha256=CD4AD089DCF4CFF61DE0CF88EE9361E799AA3AC88DB7DFE243503C506CABD331
```

## Pressure Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_branch_pair_policy_rank_pressure_cp7_pressure_g8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 2 | 8 | 25.00% |
| Grixis Affinity | 2 | 8 | 25.00% |
| Combined | 4 | 16 | 25.00% |

## Verdict

Rejected for promotion and HPC. The stricter corpus is balanced across Rally and
Affinity, but it is too sparse: only 11 paired episodes from 64 pressure
scenarios. The imported checkpoint is neutral at best on the pressure gate and
does not justify larger evaluation.

This narrows the trajectory-preference path: baseline-losing-alternative pairs
are useful diagnostics, but local branch search does not produce enough
pressure-pair data for the current paired-loss import strategy.
