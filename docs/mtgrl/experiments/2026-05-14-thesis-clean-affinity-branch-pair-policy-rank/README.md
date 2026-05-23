# Thesis-clean Affinity branch-pair policy rank - 2026-05-14

## Question

Can an Affinity-only baseline-losing-alternative paired branch corpus repair the
accepted checkpoint's weakest CP7 matchup when imported with a generic
sequence-level policy-pair rank loss?

Thesis boundary: clean. The collection and import used terminal `WIN` labels
only, no heuristic step rewards, no Spy terminal mode, no Spy candidate facts,
no action-text regex filters, and no selective MCTS keyword gates.

## Collection

Run:

```text
local-training/local_pbt/action_counterfactual/20260514_branch_pair_affinity_baselinealt_collect128_stop32
profile=Pauper-Spy-Combo-Value
opponent=cp7
opponent_deck=Deck - Grixis Affinity.dek
scenarios=128
stop_after_examples=32
baseline_losing_alternative_only=true
branch_trajectory_mode=true
branch_trajectory_pair_mode=true
branch_trajectory_first_post_target_only=true
branch_trajectory_require_training_example=true
terminal_mode=WIN
```

Result:

```text
completed_scenarios=128
trained_scenarios=22
skipped_scenarios=106
candidate_examples=27
elapsed_sec=6023.8
export=branch_pair_trajectories.ser
```

All 27 selected examples were `ACTIVATE_ABILITY_OR_SPELL` decisions. Import
loaded 21 paired episodes / 42 steps.

## Import

Profile:

```text
Pauper-Spy-Combo-Value-BranchPairPolicyRankAffinity-20260514
```

Start checkpoint:

```text
Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Run:

```text
local-training/local_pbt/action_counterfactual/20260514_branch_pair_policy_rank_affinity_e8
train_epochs=8
trajectory_episodes=21
trajectory_steps=42
train_pass_samples=336
POLICY_PAIR_RANK_LOSS_COEF=0.10
POLICY_PAIR_RANK_MARGIN=0.20
REFERENCE_POLICY_KL_COEF=1.0
policy/value/MCTS KL losses disabled
```

Result checkpoint:

```text
sha256=8CF7D1B4AC58C04BBBFC3B42D727643D76C2CB500176869D188C746B21730F0E
```

## Eval

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_branch_pair_policy_rank_affinity_cp7_g16
skill=7
opponent=Grixis Affinity
games=16
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Grixis Affinity | 3 | 16 | 18.75% |

Reference:

```text
accepted CP7 Grixis Affinity: 13/62 = 20.97%
fresh accepted pressure sample: 2/8 = 25.00%
```

## Decision

Rejected. Do not submit this branch to HPC.

The larger Affinity-only corpus is still sparse and the policy-pair rank import
does not beat the accepted Affinity reference. This closes the current paired
branch import family unless the target generator changes materially. The filter
remains useful for diagnostics because it finds accepted-policy losing decisions
with terminal-winning siblings, but fitting those sparse pairs is not yet a
working policy-improvement operator.
