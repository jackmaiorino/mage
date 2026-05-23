# Thesis-Clean Generic Prefix Search

Date: 2026-05-11

## Question

Can offline terminal-win prefix search produce useful policy-improvement targets without explicitly teaching Spy card logic?

This is the offline version of the search-as-policy-improvement direction. It avoids live train-time `priority()` search, which previously caused deadline and forced-pass failures.

## Thesis Boundary

Allowed:

- terminal `WIN` labels only;
- search over legal candidate actions;
- generic branch ordering by model probability plus generic pass-last ordering;
- measured hard-matchup data balance.

Disabled:

- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`;
- no `SPY_LANDLESS_COMBO_WIN` or other Spy terminal mode;
- no action-text include regex;
- no Spy-reachable opening-hand pool;
- no tactic autopilot;
- no MCTS/ISMCTS during eval or training;
- no heuristic step rewards.

## Plumbing Change

`ActionCounterfactualTrainer` had card-name tactical branch ordering baked into branch expansion. That is not thesis-clean for this experiment. Added:

```text
--generic-branch-order=true
```

and PowerShell wrapper support:

```powershell
-GenericBranchOrder
```

When enabled, branch expansion does not use the `preferredTacticPriority` card-name rules. The remaining ordering is policy probability plus generic pass-last ordering.

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
PowerShell parser check for scripts/run_spy_line_search.ps1
```

## Branch 1: Mixed Unique CP1 Prefix Distillation

Clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefix-20260511
```

Source hashes matched accepted before training:

```text
accepted model.pt       14E1A1DA4F3E
accepted model_latest   72857AA2975A
clone model.pt          14E1A1DA4F3E
clone model_latest      72857AA2975A
```

Collection run:

```text
20260511_thesis_clean_generic_prefix_win_collect12
```

Settings:

- unique opponent pool: Spy, Wildfire, Rally, Affinity;
- CP1 opponent;
- `terminal_mode=WIN`;
- `generic_branch_order=true`;
- `max_prefix_depth=5`;
- `max_search_nodes=15`;
- `top_k=3`, `random_extra=1`;
- `collect_only=true`;
- no action-text regex.

Result:

```text
completed_scenarios=8
trained_scenarios=6
candidate_examples=26
winning_trajectories=6
```

Training run:

```text
20260511_thesis_clean_generic_prefix_win_train26
```

Settings:

```text
training_examples=26
train_epochs=6
candidate_permutations=2
train_pass_samples=312
MCTS_KL_LOSS_COEF=1.0
```

Clone hashes after training:

```text
model.pt        CDC1C9352006
model_latest.pt CDC1C9352006
```

Accepted active profile hashes stayed unchanged.

### Reduced CP1 Screen

Run:

```text
20260511_generic_prefix_train26_spy_cp1_unique_eval16
```

Overall:

```text
33/64 = 51.56%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 15 | 16 | 93.75% |
| Jund Wildfire | 11 | 16 | 68.75% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 2 | 16 | 12.50% |

### Reduced CP7 Screen

Run:

```text
20260511_generic_prefix_train26_spy_cp7_unique_eval16
```

Overall:

```text
31/64 = 48.44%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 9 | 16 | 56.25% |
| Mono Red Rally | 6 | 16 | 37.50% |
| Grixis Affinity | 2 | 16 | 12.50% |

Verdict: not promotable. Aggregate CP7 is interesting, but Affinity is too weak and the result is likely matchup-shifted rather than broadly stronger.

## Branch 2: Hard-Matchup-Balanced Prefix Distillation

Reason:

The first branch was carried by mirror/Wildfire. Collect extra generic terminal-win prefixes from the measured hard matchups before training a fresh accepted-weight clone.

Additional collection runs:

```text
20260511_thesis_clean_generic_prefix_affinity_collect
20260511_thesis_clean_generic_prefix_rally_collect
```

Both used:

- single opponent deck for measured hard matchup;
- CP1 opponent;
- `terminal_mode=WIN`;
- `generic_branch_order=true`;
- no action-text regex.

Result:

```text
mixed unique examples: 26
Affinity examples:    26
Rally examples:       26
combined examples:    78
```

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixHard-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_hard_train78
```

Settings:

```text
training_examples=78
train_epochs=4
candidate_permutations=2
train_pass_samples=624
MCTS_KL_LOSS_COEF=0.75
```

Clone hashes after training:

```text
model.pt        9B76DFA491B6
model_latest.pt 9B76DFA491B6
```

Accepted active profile hashes stayed unchanged.

### Reduced CP1 Screen

Run:

```text
20260511_generic_prefix_hard_train78_spy_cp1_unique_eval16
```

Overall:

```text
31/64 = 48.44%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 10 | 16 | 62.50% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 5 | 16 | 31.25% |

Verdict: rejected. CP1 missed the reduced local screen, so CP7 was skipped to avoid spending another long eval on a branch that already failed. Affinity improved versus branch 1, but Rally collapsed.

## Branch 3: Head-Only Stabilizer

Reason:

The balanced branch looked like policy over-steer. Test whether keeping the shared encoder/value path fixed and training only policy heads makes the prefix targets usable.

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixHead-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_head_lowcoef_train78
```

Settings:

```text
training_examples=78
train_epochs=3
candidate_permutations=2
train_pass_samples=468
MCTS_KL_LOSS_COEF=0.20
DISTILL_HEAD_ONLY=true
```

Clone hashes after training:

```text
model.pt        BAC710350CE6
model_latest.pt BAC710350CE6
```

Reduced CP1 screen:

```text
20260511_generic_prefix_head_lowcoef_train78_spy_cp1_unique_eval16
27/64 = 42.19%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 15 | 16 | 93.75% |
| Jund Wildfire | 6 | 16 | 37.50% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 3 | 16 | 18.75% |

Verdict: rejected. Head-only preserved mirror performance but damaged non-mirror matchups too much.

## Branch 4: Policy-Path-Only Stabilizer

Reason:

Allow policy candidate-path adaptation while freezing the state encoder and critic.

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixPath-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_path_soft_train78
```

Settings:

```text
training_examples=78
train_epochs=2
candidate_permutations=1
train_pass_samples=156
MCTS_KL_LOSS_COEF=0.10
DISTILL_POLICY_PATH_ONLY=true
```

Clone hashes after training:

```text
model.pt        3969881BCA56
model_latest.pt 3969881BCA56
```

Reduced CP1 screen:

```text
20260511_generic_prefix_path_soft_train78_spy_cp1_unique_eval16
27/64 = 42.19%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 9 | 16 | 56.25% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 1 | 16 | 6.25% |

Verdict: rejected. Soft policy-path training was still worse than accepted on the CP1 screen.

## Branch 5: Target-Policy Anchor Mix

Reason:

Instead of forcing each prefix target directly, blend the normalized prefix target with the current model policy distribution. This is generic and card-agnostic; it acts as a trust-region-like target without loading a separate reference model.

Plumbing change:

```text
MCTS_TARGET_POLICY_MIX
```

Wrapper support:

```powershell
-MctsTargetPolicyMix
```

When set below `1.0`, the MCTS/prefix distillation target becomes:

```text
target = mix * prefix_target + (1 - mix) * current_policy.detach()
```

Verification:

```text
py -3.12 -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
```

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixAnchor-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_anchor_mix25_train78
```

Settings:

```text
training_examples=78
train_epochs=4
candidate_permutations=2
train_pass_samples=624
MCTS_KL_LOSS_COEF=0.75
MCTS_TARGET_POLICY_MIX=0.25
```

Clone hashes after training:

```text
model.pt        D750281158E2
model_latest.pt D750281158E2
```

### Reduced CP1 Screen

Run:

```text
20260511_generic_prefix_anchor_mix25_train78_spy_cp1_unique_eval16
```

Overall:

```text
37/64 = 57.81%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 12 | 16 | 75.00% |
| Mono Red Rally | 9 | 16 | 56.25% |
| Grixis Affinity | 3 | 16 | 18.75% |

### Reduced CP7 Screen

Run:

```text
20260511_generic_prefix_anchor_mix25_train78_spy_cp7_unique_eval16
```

Overall:

```text
23/62 = 37.10%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 10 | 16 | 62.50% |
| Jund Wildfire | 4 | 14 | 28.57% |
| Mono Red Rally | 6 | 16 | 37.50% |
| Grixis Affinity | 3 | 16 | 18.75% |

Verdict: rejected. Anchoring fixed CP1 over-steer and is the best local stabilizer so far, but CP1-derived prefix data did not transfer to CP7 pressure.

## Branch 6: CP7 Hard-Matchup Prefix Data

Reason:

The anchored branch passed the CP1 screen but failed CP7, especially Wildfire and Affinity. Collect CP7-specific generic terminal-win prefixes from hard matchups, without changing the thesis boundary.

Collection runs:

```text
20260511_thesis_clean_generic_prefix_cp7_affinity_collect
20260511_thesis_clean_generic_prefix_cp7_wildfire_collect
20260511_thesis_clean_generic_prefix_cp7_rally_collect
```

Settings:

- CP7 skill 7 opponent;
- `terminal_mode=WIN`;
- `generic_branch_order=true`;
- no action-text regex;
- `max_prefix_depth=5`;
- `max_search_nodes=63`;
- `top_k=3`, `random_extra=1`;
- collect-only.

Result:

```text
CP1 mixed/hard examples: 78
Affinity CP7 examples:  16
Wildfire CP7 examples:  18
Rally CP7 examples:     12
combined examples:      124
```

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixCp7-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_cp1cp7_anchor_mix25_train124
```

Settings:

```text
training_examples=124
train_epochs=3
candidate_permutations=2
train_pass_samples=744
MCTS_KL_LOSS_COEF=0.75
MCTS_TARGET_POLICY_MIX=0.25
anchor=current policy
```

Clone hashes after training:

```text
model.pt        E475531BFA06
model_latest.pt E475531BFA06
```

Reduced CP7 screen:

```text
20260511_generic_prefix_cp1cp7_anchor_mix25_train124_spy_cp7_unique_eval16
25/62 = 40.32%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 8 | 15 | 53.33% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 2 | 15 | 13.33% |

Verdict: rejected. Adding CP7 hard-matchup data improved Wildfire versus the CP1-only anchor, but did not fix Affinity and did not beat the previous CP7 screen.

## Branch 7: Frozen Accepted-Policy Reference Anchor

Reason:

Current-policy anchoring still moves as the clone learns. Add a generic frozen-reference option so prefix targets can be mixed with the accepted checkpoint distribution for the same candidate set.

Plumbing change:

```text
MCTS_REFERENCE_MODEL_PATH
```

Wrapper support:

```powershell
-MctsReferenceModelPath
```

When `MCTS_TARGET_POLICY_MIX < 1.0` and a reference path is set, the target becomes:

```text
target = mix * prefix_target + (1 - mix) * frozen_reference_policy
```

Verification:

```text
py -3.12 -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
PowerShell parser check for scripts/run_spy_line_search.ps1
```

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixCp7Ref-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_cp1cp7_refanchor_mix25_train124_fix
```

Settings:

```text
training_examples=124
train_epochs=4
candidate_permutations=2
train_pass_samples=992
MCTS_KL_LOSS_COEF=0.75
MCTS_TARGET_POLICY_MIX=0.25
MCTS_REFERENCE_MODEL_PATH=accepted model.pt
```

Clone hashes after training:

```text
model.pt        3CDC4FBF1199
model_latest.pt 3CDC4FBF1199
```

Reduced CP7 screen:

```text
20260511_generic_prefix_cp1cp7_refanchor_mix25_train124_spy_cp7_unique_eval16
28/62 = 45.16%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 9 | 14 | 64.29% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 4 | 16 | 25.00% |

Reduced CP1 screen:

```text
20260511_generic_prefix_cp1cp7_refanchor_mix25_train124_spy_cp1_unique_eval16
32/64 = 50.00%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 10 | 16 | 62.50% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 7 | 16 | 43.75% |

Verdict: rejected. Frozen anchoring is better than moving anchoring on CP7 and improves Affinity, but the mix is still too strong for Rally and does not beat accepted on CP1.

## Branch 8: Lower Frozen-Reference Mix

Reason:

The `0.25` frozen-reference mix helped CP7 and Affinity but hurt CP1/Rally. Test whether a weaker prefix target preserves the accepted policy while keeping some hard-matchup gain.

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixCp7RefMix15-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_cp1cp7_refanchor_mix15_train124
```

Settings:

```text
training_examples=124
train_epochs=4
candidate_permutations=2
train_pass_samples=992
MCTS_KL_LOSS_COEF=0.75
MCTS_TARGET_POLICY_MIX=0.15
MCTS_REFERENCE_MODEL_PATH=accepted model.pt
```

### Reduced CP1 Screen

Run:

```text
20260511_generic_prefix_cp1cp7_refanchor_mix15_train124_spy_cp1_unique_eval16
```

Overall:

```text
37/64 = 57.81%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 10 | 16 | 62.50% |
| Mono Red Rally | 7 | 16 | 43.75% |
| Grixis Affinity | 6 | 16 | 37.50% |

### Invalid CP7 Attempt

Run:

```text
20260511_generic_prefix_cp1cp7_refanchor_mix15_train124_spy_cp7_unique_eval16
```

Invalid: most chunks returned `0/0` because the shared GPU host died or refused connections. Do not use this result.

### Reduced CP7 Rerun

Run:

```text
20260511_generic_prefix_cp1cp7_refanchor_mix15_train124_spy_cp7_unique_eval16_rerun
```

Overall:

```text
24/63 = 38.10%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 3 | 16 | 18.75% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 5 | 15 | 33.33% |

Verdict: rejected. Lowering the prefix mix restored CP1 performance and improved CP7 Affinity, but Wildfire collapsed at CP7. The current direct-distillation family is producing tradeoffs, not broad gains.

## Branch 9: Low-Coefficient Frozen-Reference KL

Reason:

The mix knob traded off CP1 and CP7. Keep the `0.25` prefix/reference target blend, but reduce the KL coefficient so prefix search acts as a small auxiliary instead of dominating the update.

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixCp7RefCoef20-20260511
```

Training run:

```text
20260511_thesis_clean_generic_prefix_cp1cp7_refanchor_coef20_train124
```

Settings:

```text
training_examples=124
train_epochs=4
candidate_permutations=2
train_pass_samples=992
MCTS_KL_LOSS_COEF=0.20
MCTS_TARGET_POLICY_MIX=0.25
MCTS_REFERENCE_MODEL_PATH=accepted model.pt
```

Reduced CP1 screen:

```text
20260511_generic_prefix_cp1cp7_refanchor_coef20_train124_spy_cp1_unique_eval16
33/64 = 51.56%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 10 | 16 | 62.50% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 4 | 16 | 25.00% |

Reduced CP7 screen:

```text
20260511_generic_prefix_cp1cp7_refanchor_coef20_train124_spy_cp7_unique_eval16
22/64 = 34.38%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 0 | 16 | 0.00% |
| Mono Red Rally | 6 | 16 | 37.50% |
| Grixis Affinity | 5 | 16 | 31.25% |

Verdict: rejected. A smaller auxiliary avoids large CP1 damage but fails the hard-skill gate badly. Stop coefficient/mix cycling for direct prefix KL.

## Branch 10: Narrow Library-Count Features Plus Frozen-Reference KL

Reason:

Expose only generic own-library composition (`RL_LIBRARY_COUNT_FEATURES_ENABLE=1`) while keeping the same frozen-reference prefix target. This is a narrower thesis-clean representation than the earlier broad zone-count branch.

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-GenericPrefixLibRefMix25-20260512
```

Training run:

```text
20260512_thesis_clean_generic_prefix_library_refanchor_mix25_train124
```

Settings:

```text
training_examples=124
train_epochs=4
candidate_permutations=2
train_pass_samples=992
MCTS_KL_LOSS_COEF=0.75
MCTS_TARGET_POLICY_MIX=0.25
MCTS_REFERENCE_MODEL_PATH=accepted model.pt
RL_LIBRARY_COUNT_FEATURES_ENABLE=1
```

Reduced CP1 screen:

```text
20260512_generic_prefix_library_refanchor_mix25_train124_spy_cp1_unique_eval16
35/64 = 54.69%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 8 | 16 | 50.00% |
| Mono Red Rally | 7 | 16 | 43.75% |
| Grixis Affinity | 5 | 16 | 31.25% |

Reduced CP7 screen:

```text
20260512_generic_prefix_library_refanchor_mix25_train124_spy_cp7_unique_eval16
23/64 = 35.94%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 4 | 16 | 25.00% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 2 | 16 | 12.50% |

Verdict: rejected. Library counts helped CP1/Rally relative to non-library mix `0.25`, but hard-skill performance failed badly. The issue is not only missing library-composition input; direct KL distillation is still over-steering the policy.

## Interpretation

Generic offline prefix search can cheaply find terminal-winning Spy lines without card-name filters. The failure is not collection; it is distillation stability and matchup balance.

The pure supervised update appears to over-steer the policy. Extra hard-matchup data can move one weak matchup up, but it shifts weakness elsewhere. Head-only and policy-path-only updates were not enough. Current-policy target anchoring is directionally useful on CP1. Frozen accepted-policy anchoring is directionally better for CP7 at mix `0.25`, while mix `0.15` restores CP1 but collapses CP7 Wildfire. Lower KL coefficient also fails CP7. Adding a narrow generic library-count representation does not rescue CP7. The direct target-matching family looks exhausted locally.

## Decision

Do not promote any clone from this experiment.

Do not use HPC for the current distillation shape. The next local experiment should use the prefix data less directly, for example as a small auxiliary during terminal RL or as a candidate-Q target, rather than replacing the policy target. Only escalate to HPC if a reduced local gate beats accepted on both CP1 and CP7 or if collection, not model quality, becomes the bottleneck.

Remaining stabilizer ideas:

- mixing prefix targets into terminal RL instead of replacing the policy objective;
- collecting a larger balanced CP7 dataset and using a reduced local gate before any full sweep.
