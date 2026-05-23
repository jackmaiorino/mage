# Profile Path Fix And Direct BC

date: 2026-05-05

## Question

After the Spy natural winning-line trajectory smoke failed to fit, determine whether the failure was caused by stale optimizer state, missing BC targets, bad profile loading, or a real state/action representation problem.

## Critical Finding

`PythonMLBridge` was hardcoding `MTG_MODEL_PATH` to the legacy flat checkpoint:

`Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt`

That meant local Py4J profile train/score runs could silently load the global model even when `MODEL_PROFILE` was set. Java still saved to the requested profile path, so the bug was easy to miss.

Fix:

- `PythonMLBridge` now passes the profile-aware `RLLogPaths.MODEL_FILE_PATH` to Python.
- It also sets `MODEL_LATEST_PATH` to the same profile directory.

## London Counterfactual Bug

`MulliganCounterfactualTrainer` shuffled copied decks but did not force that order back into the player libraries before starting each branch game.

Observed symptom:

- old London line samples had different first hands for different branches inside the same scenario.
- that means keep-vs-mull and bottom-combo outcomes were often not same-state counterfactuals.

Fix:

- after `game.loadCards(...)`, the trainer now calls `forceLibraryOrder(...)` for both players in normal mulligan and London line modes.
- smoke run `20260505_london_libraryorder_smoke` verified each branch in a scenario now has the same first hand.

This invalidates the old London512 repair set as a precise counterfactual dataset. Its CP1 result is still useful as a historical artifact, but the next experiment must regenerate London data with the fixed branch harness.

## Direct BC Path

Added `BC_DIRECT_LOSS=1` support inside `trainCandidatesMultiFlat`.

This path:

- trains directly on `mctsVisitTargets` or chosen-index fallback targets;
- uses logits and cross-entropy;
- bypasses GAE, PPO, value, entropy, and belief losses;
- is exposed in `scripts/run_spy_line_search.ps1` as `-DirectBcLoss`.

## Data

Smoke trajectory set:

- collection run: `20260504_spy_natural_winline_collect64_smoke`
- scenarios: 64
- successful/actionable scenarios: 26
- exported winning-line samples: 2,760

## Results After Profile Fix

Source profile score on the same smoke set:

- profile: `Pauper-Spy-Combo-ActualWin-Big-Offline-20260504`
- score run: `20260505_bridgefix_natural_winline_source_score_smoke64`
- exact-label score: 797/2760 = 28.88%
- average target probability: 0.321

Old MCTS-KL path after fix:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-BridgeFix-Smoke-20260505`
- 20 epochs: 968/2760 = 35.07%
- 100 epochs: 789/2760 = 28.59%

Direct BC path:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-Direct-Smoke-20260505`
- 20 epochs: 1090/2760 = 39.49%
- 100 epochs: 1055/2760 = 38.22%

Action-head-only direct BC:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-Direct-ActionOnly-20260505`
- action type: `ACTIVATE_ABILITY_OR_SPELL`
- 50 epochs: 747/2142 = 34.87%

No-pass direct BC:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-Direct-NoPass-Smoke-20260505`
- no-pass source score: 590/2120 = 27.83%
- roughly 20 epochs: 929/2120 = 43.82%
- roughly 100 epochs: 1120/2120 = 52.83%
- full score including target-pass rows after no-pass training: 1120/2760 = 40.58%

Fixed-subset in-process fit probes:

- 16 non-pass rows, 100 epochs, 1 candidate order: 5/16 -> 15/16 = 93.75%
- 32 non-pass rows, 100 epochs, 1 candidate order: 11/32 -> 25/32 = 78.13%
- 64 non-pass rows, 100 epochs, 1 candidate order: 19/64 -> 42/64 = 65.63%
- 64 non-pass rows, 100 epochs, 4 candidate-order permutations: 19/64 -> 57/64 = 89.06%

Full no-pass candidate-permutation gate:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-Direct-NoPass-Perm4-Smoke-20260505`
- 2,120 non-pass rows, 50 epochs, 4 candidate-order permutations
- train-pass samples: 424,000
- no-pass score: 1089/2120 = 51.37%
- high-cardinality candidate sets remained weak; 2-candidate choices scored 91.44%, but 6+ candidate choices mostly stayed near chance.

London exact-fit repair on the actual-win big profile:

- dataset: `20260504_spy_mulligan_exactfit_collect256/selected_training_data.ser`
- source score: 67/302 = 22.19%
- direct fit, 100 epochs, 4 candidate-order permutations: exact top-1 120/302 = 39.74%
- corrected multi-positive target-set top-1: 300/302 = 99.34%
- source first-hand probe: 512/512 keeps, 0 mulligans
- repaired first-hand probe: nontrivial mulligans across all effective-land buckets
- CP1 eval after repair: 5/64 = 7.81%

London repair CP1 breakdown:

- vs Spy Combo: 4/16 = 25.00%
- vs Jund Wildfire: 1/16 = 6.25%
- vs Mono Red Rally: 0/16 = 0.00%
- vs Grixis Affinity: 0/16 = 0.00%

Larger London repair:

- collection: `20260505_spy_london_collect512_bridgefix`
- scenarios: 512
- actionable scenarios: 175
- selected tensors: 433
- rollout activation failures: 18 observed in logs
- profile: `Pauper-Spy-Combo-London512-Direct-20260505`
- direct fit, 100 epochs, 4 candidate-order permutations: target-set top-1 429/433 = 99.08%
- first-hand probe: about 35-42% mulligans across effective-land buckets
- CP1 eval: 8/64 = 12.50%

Larger London CP1 breakdown:

- vs Spy Combo: 5/16 = 31.25%
- vs Jund Wildfire: 1/16 = 6.25%
- vs Mono Red Rally: 0/16 = 0.00%
- vs Grixis Affinity: 2/16 = 12.50%

Larger London full hard-binary repair:

- profile: `Pauper-Spy-Combo-London512-FullHardBinary-20260505`
- direct fit, 100 epochs, 4 candidate-order permutations, hard one-hot targets for binary keep/mull rows
- exact top-1: 188/433 = 43.42%
- corrected multi-positive target-set top-1: 431/433 = 99.54%
- first-hand probe: nontrivial mulligans across all effective-land buckets; 0-2 effective lands mulligan about 58-67%
- CP1 eval: 14/64 = 21.88%

Full hard-binary CP1 breakdown:

- vs Spy Combo: 7/16 = 43.75%
- vs Jund Wildfire: 4/16 = 25.00%
- vs Mono Red Rally: 0/16 = 0.00%
- vs Grixis Affinity: 3/16 = 18.75%

Small CP1 eval of the best direct-BC smoke profile:

- run: `20260505_directbc_spy_smoke_cp1_eval8`
- games: 32 total, 8 per opponent deck
- result: 0/32 = 0.00%

## Interpretation

The profile-path bug was real and important, but fixing it did not make winning-line BC fit. Direct BC is better than the old path at 20 epochs, but it plateaus far below the >70% imitation gate and produces 0% CP1 winrate.

Do not scale this generator to 10k trajectories yet without candidate-order augmentation and another full imitation gate. More rows from the same label source are unlikely to solve the problem until the label/representation issue is isolated.

Most likely blockers:

- Winning-line labels are one arbitrary line among several equivalent winning lines, so exact top-1 is noisy.
- The action-state representation may not expose enough information for the policy head to identify combo-critical choices in high-cardinality action sets.
- London/action labels are mixed with many forced-prefix states that do not match full random-start play.
- Candidate order is a major confounder: 64-row in-process fitting failed with one order but succeeded with four candidate-order permutations.
- Exact top-1 is the wrong metric for multi-positive London bottom labels; target-set top-1 shows the bottom ranking can fit, but fitted bottom/mulligan behavior still does not produce CP1 wins.
- The current London counterfactual labels can make the model mulligan. Hard binary keep/mull targets recover above the actual-win action-only seed in this small CP1 sweep, but the result is still far below the >70% imitation gate.

## Next Experiment

The next useful experiment should connect terminal-only start-state labels to actual game outcomes, not generate more action trajectories:

1. Build a held-out London-line eval set with terminal outcomes per opener.
2. Score keep/mull and bottom ranking against terminal-winning line choices using target-set metrics.
3. Add an outcome replay probe: take model-chosen keep/mull/bottom choices for sampled openers, force only those London choices, then let the actual-win action policy play.
4. If replayed model London choices lose while labeled winning alternatives win, train a calibrated start-state policy on the held-out London data.
5. Only after the start-state replay probe clears should we spend compute on larger trajectory collection or CP1 sweeps.

Primary artifacts:

- `local-training/local_pbt/spy_line_search/20260505_bridgefix_natural_winline_source_score_smoke64`
- `local-training/local_pbt/spy_line_search/20260505_directbc_natural_winline_score_smoke64_e20`
- `local-training/local_pbt/spy_line_search/20260505_directbc_natural_winline_score_smoke64_e100`
- `local-training/local_pbt/spy_line_search/20260505_directbc_natural_winline_actiononly_score_e50`
- `local-training/local_pbt/cp7_eval_sweeps/20260505_directbc_spy_smoke_cp1_eval8`
- `local-training/local_pbt/spy_line_search/20260505_directbc_nopass_perm4_natural_winline_score_smoke64_e50_nonpass`
- `local-training/local_pbt/spy_line_search/20260505_london_exactfit_perm4_e100_targetset_score`
- `local-training/local_pbt/mulligan_probes/20260505_spy_london_exactfit_probe512`
- `local-training/local_pbt/cp7_eval_sweeps/20260505_london_exactfit_cp1_eval16`
- `local-training/local_pbt/mulligan_counterfactual/20260505_spy_london_collect512_bridgefix`
- `local-training/local_pbt/spy_line_search/20260505_london512_fitprobe433_perm4_e100`
- `local-training/local_pbt/mulligan_probes/20260505_spy_london512_probe512`
- `local-training/local_pbt/cp7_eval_sweeps/20260505_london512_cp1_eval16`
- `local-training/local_pbt/spy_line_search/20260505_london512_full_hardbinary_fitprobe_e100`
- `local-training/local_pbt/mulligan_probes/20260505_spy_london512_fullhard_probe512`
- `local-training/local_pbt/cp7_eval_sweeps/20260505_london512_fullhard_cp1_eval16`
