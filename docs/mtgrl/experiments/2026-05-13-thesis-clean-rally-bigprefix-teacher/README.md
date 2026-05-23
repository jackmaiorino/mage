# Thesis-Clean Rally Big-Prefix Teacher

Date: 2026-05-13

## Question

Can a larger generic terminal-prefix search produce useful Rally-specific policy-improvement targets without card-name tactics or Spy-specific labels?

This is thesis-clean: generic branch order, terminal `WIN` only, no heuristic rewards, no Spy terminal mode, no Spy candidate facts, no action-text filters, and no selective MCTS keywords.

## Trigger

Deep-mulligan line collection found `0/48` Rally line wins across six sampled Rally scenarios. Small-budget generic online prefix search also found no Rally wins. A larger online-prefix eval did find one terminal-winning prefix, but live autopilot diverged after five applied actions and the game still lost. That suggested the search backend may find terminal lines, while live long-prefix execution remains brittle.

## Collection

Run:

```text
20260513_rally_bigprefix_policymiss_collect8
```

Settings:

- Profile: accepted `Pauper-Spy-Combo-Value`
- Opponent: CP7 Mono Red Rally only
- Scenarios: 8
- `max_prefix_depth=10`
- `train_prefix_depth=10`
- `max_search_nodes=127`
- `top_k=4`
- `generic_branch_order=true`
- `tactic_autopilot=false`
- `skip_pass_training=true`
- `skip_blank_training=true`
- `skip_mulligan_training=true`
- collect-only

Result:

```text
trainedScenarios=8
skippedScenarios=0
candidateExamples=60
winningTrajectories=8
selectedExamples=60
elapsed_sec=142.6
```

Export:

```text
local-training/local_pbt/action_counterfactual_data/20260513_rally_bigprefix_policymiss_collect8.ser
```

## Accepted Score

Run:

```text
20260513_rally_bigprefix_policymiss_score_accepted
```

Result:

```text
top1=60/60 = 100.0%
targetSetTop1=60/60 = 100.0%
avgTargetProb=0.474155
avgRank=1.0000
```

## Receding-Horizon Eval

To avoid brittle long autopilot suffixes, a one-game receding-horizon variant disabled autopilot and reran larger generic search at each live decision:

```text
20260513_accepted_online_prefix_generic_rally_receding_cp7_eval1
```

Result:

```text
Mono Red Rally CP7: 0/1
calls=8 found=0 timeouts=8
```

## Decision

Rejected for training and HPC.

The larger generic prefix collector can find Rally terminal-winning lines, but the resulting serialized decisions are already top-1 under the accepted policy. This repeats the solved-prefix-corpus failure mode from the corrected trajectory teacher. The live larger-budget autopilot signal is not robust, and the receding-horizon variant found no wins.

Keep the corpus as diagnostic evidence: Rally is not failing because the policy cannot rank these searched states; it is failing because fresh-game state distribution, opponent interaction, and long-horizon execution diverge before those states become reliable wins.

## Policy-Miss Filter Fix

The first collect run used `-PolicyMissOnly`, but `ActionCounterfactualTrainer.selectExamples(...)` only applied that filter during import/training, not during collect/export selection. That made collect-only runs able to export solved corpora under a policy-miss name.

Patch:

- `selectExamples(...)` now calls `passesPolicyMissFilter(example.trainingData, args)` before deduping/exporting.

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

Fixed collect check:

```text
20260513_rally_bigprefix_policymiss_collect4_fixed
```

Result:

```text
trainedScenarios=3
skippedScenarios=1
candidateExamples=14
winningTrajectories=3
selectedExamples=0
```

Read: the fixed selector confirms the Rally big-prefix labels are solved by accepted and should not be exported for training.
