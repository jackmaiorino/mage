# Thesis-clean branch-pair policy rank - 2026-05-14

## Question

The existing branch-pair corpus exposes generic terminal win/loss alternatives.
Value ranking and simple return-contrastive policy import both failed. Does a
direct policy pair-ranking objective work better?

The objective ranks adjacent paired records by terminal outcome:

```text
logp(selected action in winning branch record)
  >
logp(selected action in losing branch record) + margin
```

It uses only the reward sign from the paired branch rollout.

## Implementation

Added default-off Python training envs:

```text
POLICY_PAIR_RANK_LOSS_COEF
POLICY_PAIR_RANK_MARGIN
```

`scripts/run_action_counterfactual.ps1` now exposes:

```text
-PolicyPairRankLossCoef
-PolicyPairRankMargin
```

The loss is generic and only applies when configured. It does not inspect card
names, action text, deck identity, or Spy-specific state.

Verification:

```text
.mtgrl_venv/Scripts/python.exe -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
PowerShell parser check for scripts/run_action_counterfactual.ps1
```

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-BranchPairPolicyRank-20260514
```

Starting checkpoint:

```text
accepted model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Dataset:

```text
local-training/local_pbt/action_counterfactual/20260513_branch_pair_collect24/branch_pair_trajectories.ser
```

Import settings:

```text
train_epochs=8
imported_trajectory_episodes=17
imported_trajectory_steps=34
train_pass_samples=272
POLICY_LOSS_COEF=0
VALUE_LOSS_COEF=0
MCTS_KL_LOSS_COEF=0
POLICY_PAIR_RANK_LOSS_COEF=0.10
POLICY_PAIR_RANK_MARGIN=0.20
REFERENCE_POLICY_KL_COEF=1.0
```

Resulting checkpoint:

```text
model_latest.pt sha256=099E3AF628375476C3D95FBD365F1681010F0E3C329891AD2EEFE48693639420
```

## CP7 Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_branch_pair_policy_rank_cp7_g4
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 3 | 4 | 75.00% |
| Jund Wildfire | 2 | 4 | 50.00% |
| Mono Red Rally | 0 | 4 | 0.00% |
| Grixis Affinity | 1 | 4 | 25.00% |
| Combined | 6 | 16 | 37.50% |

## Verdict

Rejected. Policy pair-ranking on the current small branch-pair corpus is not a
promotion candidate. It recovered one Affinity game but collapsed Rally and did
not clear the accepted CP7 anchor.

This closes the current paired-branch import family: value-rank, critic-only
value-rank, return-contrastive, and policy-pair rank all failed local gates on
the available corpus. Future work needs a better trajectory generator or a
substantially larger Affinity/Rally-balanced corpus before trying another paired
loss.
