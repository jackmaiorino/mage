# Thesis-clean action pair-rank - 2026-05-14

## Question

Prior branch-return imports may have over-steered because they matched whole
target distributions. Can a weaker same-state preference objective transfer
better by only ranking the best terminal sibling above the worst terminal
sibling?

Thesis boundary: clean. The loss reads only generic signed terminal branch
returns in `[-1, 1]` and the legal candidate mask. It does not inspect action
text, card names, Spy terminal states, or heuristic rewards.

## Implementation

Added default-off Python loss knobs:

```text
ACTION_PAIR_RANK_LOSS_COEF
ACTION_PAIR_RANK_MARGIN
ACTION_PAIR_RANK_MIN_GAP
```

For each signed branch-return row with at least two observed legal candidates,
the loss compares the policy log-probability of the best-return candidate
against the worst-return candidate:

```text
softplus(margin - (log p(best) - log p(worst)))
```

Rows are weighted by terminal return gap. This is intentionally weaker than
`BRANCH_RETURN_POLICY_LOSS_COEF`, which asks the policy to match a full target
distribution.

## Import

Profile:

```text
Pauper-Spy-Combo-Value-ActionPairRankAffinity-20260514
```

Start checkpoint:

```text
Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Dataset:

```text
local-training/local_pbt/action_counterfactual/20260512_thesis_clean_generic_branchret_balanced_cp7/affinity48.ser
source run: 20260512_thesis_clean_generic_branchret_onnx_affinity48_cp7
signed branch-return rows collected from CP7 Grixis Affinity
```

Import:

```text
local-training/local_pbt/action_counterfactual/20260514_action_pair_rank_affinity_e8
training_examples=35
train_epochs=8
train_pass_samples=280
ACTION_PAIR_RANK_LOSS_COEF=0.10
ACTION_PAIR_RANK_MARGIN=0.20
ACTION_PAIR_RANK_MIN_GAP=0.25
REFERENCE_POLICY_KL_COEF=1.0
policy/value/MCTS KL/branch-return policy losses disabled
```

Result checkpoint:

```text
sha256=BF5AB69CC2FE74428579B678006AF7A92FF2A231529CC26D66A3CC7793909034
```

## Eval

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_action_pair_rank_affinity_cp7_g16
skill=7
opponent=Grixis Affinity
games=16
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Grixis Affinity | 1 | 16 | 6.25% |

## Decision

Rejected. Do not submit to HPC.

This is worse than both the accepted Affinity reference and the full
branch-return policy import family. The current branch-return datasets are not
failing only because the target distribution is too sharp; they are not
transferring through policy fitting at all.
