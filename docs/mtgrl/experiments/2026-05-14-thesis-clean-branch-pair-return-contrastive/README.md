# Thesis-clean branch-pair return contrastive - 2026-05-14

## Question

The branch-pair value-ranking objective produced a real generic win/loss signal
but did not improve play. Would using the same paired terminal branch data as a
policy-side return-contrastive objective work better than value-only ranking?

This is a small operator smoke on existing paired branch data before collecting
more examples.

## Thesis Boundary

Clean:

- paired labels come from generic terminal branch win/loss outcomes;
- no heuristic rewards;
- no `SPY_LANDLESS_COMBO_WIN`;
- no card-name/action-text filters;
- no Spy action facts;
- no selective search keywords.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-BranchPairReturnContrast-20260514
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
RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.10
RETURN_CONTRASTIVE_CRITICAL_ONLY=0
RETURN_CONTRASTIVE_NEG_PROB_FLOOR=0.40
REFERENCE_POLICY_KL_COEF=1.0
```

Resulting checkpoint:

```text
model_latest.pt sha256=52E0C4448D33AE9D6191275B5E099E2FE1704D2ADB4F835F9185C7C48D4E806A
```

## CP7 Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_branch_pair_return_contrast_cp7_g4
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 4 | 4 | 100.00% |
| Jund Wildfire | 1 | 4 | 25.00% |
| Mono Red Rally | 1 | 4 | 25.00% |
| Grixis Affinity | 0 | 4 | 0.00% |
| Combined | 6 | 16 | 37.50% |

## Verdict

Rejected. The return-contrastive operator did not rescue the existing paired
branch corpus and immediately failed Affinity. Do not expand or submit this
profile to HPC.

The result does not prove trajectory preferences are useless, but it closes this
simple import recipe. A future trajectory-preference attempt needs either a
larger Affinity-robust pair corpus or a sequence-level objective that compares
whole branch rollouts, not just the first retained paired records.
