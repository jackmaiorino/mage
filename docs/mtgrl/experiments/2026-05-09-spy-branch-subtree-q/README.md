# 2026-05-09 Spy Branch-Subtree Q

Goal: test whether candidate-level Q targets become useful if a forced candidate is evaluated by a bounded terminal prefix subtree instead of by the immediate branch-return rollout.

## Implementation

- `ActionCounterfactualTrainer` now carries baseline prefix choice text and forced target text through `ActionPlayer`.
- Forced replay can remap by action text when candidate indices change, which removes one source of false forced-branch failure.
- Added `--branch-subtree-search-nodes` / `-BranchSubtreeSearchNodes`.
- Branch-subtree mode marks a forced candidate positive when bounded `TerminalPrefixSearch` from the actual baseline prefix plus the forced candidate finds a terminal win.
- No heuristic rewards were used.

## Text-Remap Branch-Return Smoke

Accepted profile, critical Spy/Dread actions only:

- Run: `20260509_textremap_branchret_smoke64`
- Scenarios: 512
- Examples: 15
- Forced-applied: 78 true, 46 false

Zone-count parent, critical Spy/Dread actions only:

- Run: `20260509_textremap_zonecounts_branchret_collect64`
- Scenarios: 512
- Examples: 23
- Forced-applied: 112 true, 83 false

Conclusion: text-consistent forcing is mechanically better, but critical-only branch-return labels are still too sparse to train from.

## Branch-Subtree Collection

Critical Spy/Dread action probe:

- Run: `20260509_branch_subtree_smoke_n128_s7`
- Scenarios: 128
- Search nodes per branch: 7
- Examples: 5

All action probe:

- Run: `20260509_branch_subtree_allactions_smoke_n64_s7`
- Scenarios completed before early stop: 14
- Examples: 35
- Accepted score: 8/35 top-1, 34/35 target-set top-1, average target probability 0.219404

Scaled all-action collection:

- Run: `20260509_branch_subtree_allactions_collect128_s7`
- Scenarios completed before early stop: 52
- Examples: 128
- Forced-applied: 401 true, 14 false
- Accepted score: 30/128 top-1, 127/128 target-set top-1, average target probability 0.222298

The dense all-action dataset was dominated by early resource sequencing labels: Forest, Land Grant, Lotus Petal, Generous Ent forestcycling, Saruli Caretaker, Forest mana, Overgrown Battlement, and Elves of Deep Shadow.

## Q Fit And Gate

Profile:

- `Pauper-Spy-Combo-BranchSubtreeQ128-20260509`
- Seeded from accepted `Pauper-Spy-Combo-FastT5Contrast-20260506`

Fit:

- Run: `20260509_branch_subtree_q128_import_e8p2_signed`
- Imported examples: 128
- Epochs: 8
- Candidate permutations: 2
- Candidate-Q only, signed targets

Reduced CP1 gate:

- Run: `20260509_branch_subtree_q128_blend01_cp1_eval16`
- `CANDIDATE_Q_BLEND=0.1`
- Total: 10/64
- Grixis Affinity: 1/16
- Jund Wildfire: 2/16
- Mono Red Rally: 1/16
- Spy mirror: 6/16

Conclusion: branch-subtree Q is negative. It provides a stronger offline branch labeler than raw branch returns, but the promotable labels are mostly early resource choices and Q blending badly damages fresh CP1 play. Do not run the 0.25/0.5 blend registries unless intentionally doing a sanity check.

## Policy-Target Control

Reason:

- The Q result could have been caused by Q-head integration rather than by the label source itself.
- Re-collected the same branch-subtree style data with `branch_return_targets=false`, so the trainer wrote soft policy targets instead of signed candidate-Q labels.

Collection:

- Run: `20260509_branch_subtree_policytargets_collect128_s7`
- Accepted profile, CP1, all action candidates
- Search nodes per branch: 7
- Completed scenarios: 80
- Examples: 130
- Forced replay again hit the known early-stop deck-import warning and a few shutdown activation failures.

Accepted score:

- Run: `20260509_score_branch_subtree_policytargets_accepted`
- Top-1: 48/130
- Target-set top-1: 125/130
- Average target probability: 0.367391

The dataset was weak before training: many rows marked every tested candidate positive, and several best labels were `Pass`.

Low-dose head-only KL fit:

- Profile: `Pauper-Spy-Combo-BranchSubtreePolicy128-20260509`
- Run: `20260509_branch_subtree_policy128_import_e2p1_head`
- 130 examples, 2 epochs, 1 permutation, `DISTILL_HEAD_ONLY=1`
- Post-fit score was unchanged: 48/130 top-1, 125/130 target-set top-1.

Direct-BC fit:

- Profile: `Pauper-Spy-Combo-BranchSubtreePolicyDirect128-20260509`
- Run: `20260509_branch_subtree_policydirect128_import_e4p1`
- 130 examples, 4 epochs, direct BC, no MCTS KL
- Post-fit score regressed to 45/130 top-1, 126/130 target-set top-1, average target probability 0.358780.

Conclusion: the negative result is not just Q-head integration. The branch-subtree label source is too soft/noisy as currently collected, so do not run CP1 gates or epoch/blend sweeps for this branch.

## Skip-Pass Label-Quality Control

Reason:

- The first policy-target dataset had 39/130 rows where the best label was `Pass`.
- Run a narrow collect with `skip_pass_best=true` to test whether pass pollution was the main problem.

Collection:

- Run: `20260509_branch_subtree_policytargets_skip_pass_collect64_s7`
- Accepted profile, CP1, all action candidates
- Search nodes per branch: 7
- Completed scenarios: 45
- Examples: 64
- One forced `Dread Return` activation failure appeared during early-stop cleanup.

Label distribution:

- Forest play: 15
- Forest mana: 9
- Land Grant: 8
- Roost Seek: 6
- Generous Ent forestcycling: 4
- Saruli Caretaker: 4
- Smaller counts for Tinder Wall, Elves of Deep Shadow, Overgrown Battlement, Gatecreeper Vine, Swamp mana, and Wall of Roots.

Accepted score:

- Run: `20260509_score_branch_subtree_policytargets_skip_pass_accepted`
- Top-1: 38/64
- Target-set top-1: 62/64
- Average target probability: 0.592657

Conclusion: skip-pass filtering removes the worst-looking labels, but the remaining data is still mostly multi-positive early resource sequencing that the accepted policy already mostly ranks inside the target set. Do not train a profile from this dataset.

## Smaller Subtree Budget Control

Reason:

- If a 7-node branch subtree is too permissive, most candidates eventually find a terminal suffix and the labeler loses contrast.
- Re-ran the skip-pass policy-target collect with 3 subtree nodes.

Collection:

- Run: `20260509_branch_subtree_policytargets_skip_pass_collect64_s3`
- Completed scenarios: 35
- Examples: 66
- `target_negative`: 0 for all 66 examples

Accepted score:

- Run: `20260509_score_branch_subtree_policytargets_skip_pass_s3_accepted`
- Top-1: 38/66
- Target-set top-1: 64/66
- Average target probability: 0.563791

Conclusion: lowering subtree budget did not create negative candidate contrast. It only changed which early resource labels were selected. Branch-subtree policy targets remain closed.

## Strict Critical Avoid-Losing Probe

Reason:

- Broad branch-subtree labels were dominated by resource sequencing.
- Try only the highest-value negative signal: baseline selected `Balustrade Spy` or `Dread Return`, that forced baseline branch failed, and another forced branch found a terminal win.

Collection:

- Run: `20260509_critical_strict_avoid_subtree_mask_collect32_s7`
- Accepted profile, CP1
- Critical action regex: `Balustrade Spy|Dread Return`
- `avoid_losing_strict_negative=true`
- `avoid_losing_mask_baseline_only=true`
- Search nodes per branch: 7
- Scenarios: 512
- Examples: 2

Examples found:

- Scenario 143: baseline `Balustrade Spy`, target `Forest: {T}: Add {G}.`
- Scenario 487: baseline `Dread Return`, target `Mesmeric Fiend`

Conclusion: the strict critical avoid-losing signal is the right shape but far too sparse under the accepted policy and local CP1 scenario distribution. Do not train from this two-example dataset. If revisited, it needs a different state sampler, not more local full-game scenarios.
