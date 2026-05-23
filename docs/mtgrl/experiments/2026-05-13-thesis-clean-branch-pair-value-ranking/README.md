# Thesis-Clean Branch Pair Value Ranking

Date: 2026-05-13

## Question

Can the value head learn a useful local ordering from generic counterfactual sibling branches: states reached after terminally winning branches should rank above states reached after terminally losing sibling branches?

This stays thesis-clean because pair construction uses only generic forced-branch terminal outcomes. It does not inspect card names, action text, Spy-specific terminal rules, heuristic step rewards, selective MCTS keywords, or Spy candidate facts.

## Implementation

- `ActionCounterfactualTrainer`
  - Added `--branch-trajectory-pair-mode`.
  - When paired mode is enabled, branch trajectory export emits two-record episodes: winning sibling post-branch state with reward `+1`, then losing sibling post-branch state with reward `-1`.
  - Import filtering requires paired episodes to remain exactly two records with descending reward.
- `py4j_entry_point.py`
  - Added `VALUE_PAIR_RANK_LOSS_COEF` and `VALUE_PAIR_RANK_MARGIN`.
  - Loss is a margin ranking objective over adjacent pairs: higher-return state value should exceed lower-return state value by the configured margin.
- `scripts/run_action_counterfactual.ps1`
  - Added `-BranchTrajectoryPairMode`, `-ValuePairRankLossCoef`, and `-ValuePairRankMargin`.

## Runs

### Smoke Collection And Import

- Collect: `20260513_branch_pair_smoke4`
- Import: `20260513_branch_pair_import_smoke4_e4`
- Dataset: 8 paired episodes, 16 steps
- Import: 4 epochs, 64 training samples, policy loss `0`, value MSE `0`, pair-rank loss `1.0`, margin `0.20`
- Reduced CP7 gate: `20260513_branch_pair_rank_smoke4_cp7_g4`

Result:

| Opponent | Wins | Total |
| --- | ---: | ---: |
| Spy mirror | 4 | 4 |
| Jund Wildfire | 2 | 4 |
| Mono Red Rally | 1 | 4 |
| Grixis Affinity | 1 | 4 |
| Overall | 8 | 16 |

Verdict: inconclusive but not promotable.

### 24-Scenario Collection And Import

- Collect: `20260513_branch_pair_collect24`
- Branch records: 242
- Candidate examples: 39
- Paired trajectory dataset: 28 paired episodes, 56 steps
- Import: `20260513_branch_pair_import_collect24_e4`
- Import: 4 epochs, 224 training samples, policy loss `0`, value MSE `0`, pair-rank loss `1.0`, margin `0.20`
- Reduced CP7 gate: `20260513_branch_pair_rank_collect24_cp7_g4`

Result:

| Opponent | Wins | Total |
| --- | ---: | ---: |
| Spy mirror | 1 | 4 |
| Jund Wildfire | 2 | 4 |
| Mono Red Rally | 0 | 4 |
| Grixis Affinity | 2 | 4 |
| Overall | 5 | 16 |

Verdict: rejected. The larger value-rank import degraded the policy-facing behavior, especially the mirror and Rally matchups.

## Read

The paired branch signal exists and can be exported/imported, but updating the shared model with this ranking loss is not a quick win. The result is consistent with shared-encoder disruption: a value-only ranking objective can move representations used by the policy even when policy loss is disabled.

## Critic-Only Containment

After the shared-model regression, a default-off `VALUE_PAIR_RANK_CRITIC_ONLY=1` mode was added to freeze all non-critic/value parameters during pair-rank imports.

- Import: `20260513_branch_pair_import_collect24_critic_e4`
- Dataset: same 28 paired episodes, 56 steps
- Import: 4 epochs, 224 training samples, pair-rank loss `1.0`, margin `0.20`, critic/value parameters only
- Reduced CP7 gate: `20260513_branch_pair_rank_critic_collect24_cp7_g4`

Result:

| Opponent | Wins | Total |
| --- | ---: | ---: |
| Spy mirror | 2 | 4 |
| Jund Wildfire | 3 | 4 |
| Mono Red Rally | 1 | 4 |
| Grixis Affinity | 1 | 4 |
| Overall | 7 | 16 |

The containment avoided the severe mirror collapse, but still did not improve the accepted reduced gate. It is not a direct promotion candidate.

### MCTS Probe

To test whether critic-only value calibration helped the repaired generic MCTS backend:

- Run: `20260513_branch_pair_rank_critic_mcts_topk_affinity_g2`
- Settings: CP7 Grixis Affinity only, `MULTI_PLY_MCTS=1`, `MCTS_ROOT_TOP_K=4`, `MCTS_ROOT_INCLUDE_PASS=1`, `MCTS_ITERATIONS=8`, `MCTS_DETERMINIZATIONS=1`, `MCTS_SKIP_TOP_PROB=0.80`
- Result: `0/2`, with 77 total MCTS activations

## Decision

Do not spend HPC on branch-pair value ranking as implemented. Shared-model pair ranking regresses, critic-only ranking does not improve, and generic MCTS remains negative after critic-only calibration. Keep the exporter/loss as diagnostic infrastructure, but close this branch as a scaling candidate.
