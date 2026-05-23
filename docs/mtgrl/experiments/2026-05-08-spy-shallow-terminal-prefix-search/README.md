# Spy Shallow Terminal Prefix Search

Date: 2026-05-08

## Question

The prior RL/BC variants converged to the same action-health failure: premature `Balustrade Spy` and `Dread Return`, with CP1 results clustered in the high 30s per 128 games. This experiment asks whether a terminal-only prefix search with a deterministic Spy tactic continuation can find substantially better natural-start lines, and whether those lines can be distilled into the current policy.

## Baseline And Search Budget Curve

All runs used:

- Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Seed: `1778279839004`
- CP1 natural starts across the active Pauper pool
- `RL_ZONE_COUNT_FEATURES_ENABLE=1`
- `-CollectOnly`
- `-TacticAutopilot`
- `-NoSearchModelScoring`
- `-TerminalMode WIN`
- skip mulligan/pass/blank training labels

Results:

| Run | Search | Result | Elapsed |
| --- | --- | ---: | ---: |
| `20260508_spy_root_tactic_autopilot_cp1_n128` | root tactic only, 1 node | 20/128 | 74.8s |
| `20260508_spy_search_budget_nodes7_n128` | 7 nodes, depth 6 | 59/128 | 201.2s |
| `20260508_spy_search_budget_nodes15_n128` | 15 nodes, depth 6 | 77/128 | 300.0s |
| `20260508_spy_shallow_search_autopilot_cp1_n128_nodes31_ser` | 31 nodes, depth 6 | 84/128 | 382.6s |
| `20260508_spy_shallow_search_autopilot_cp1_n128_nodes63_depth8_ser` | 63 nodes, depth 8 | 86/128 | 834.6s |

The 31-node run was the first clearly strong teacher in this surface. The 63-node/depth-8 run barely improved total wins and shifted matchup coverage rather than uniformly improving it:

| Matchup | 31 nodes | 63 nodes/depth 8 |
| --- | ---: | ---: |
| Affinity | 20/27 | 14/27 |
| Wildfire | 24/35 | 21/35 |
| Rally | 15/28 | 16/28 |
| Spy mirror | 26/38 | 35/38 |

Interpretation: terminal prefix search is valuable, but local budget/depth has diminishing returns and depth 8 causes enough scenario timeouts that it is not an obvious better teacher than the 31-node/depth-6 run.

## Accepted Policy On Search Tensors

Score probe:

- Run: `20260508_score_shallow_search_ser_accepted`
- Data: `20260508_spy_shallow_search_autopilot_cp1_n128_nodes31_ser/training_data.ser`
- Result: `461/490` top-1, accuracy `0.9408`, target-set accuracy `0.9408`, mean target probability `0.915299`

The accepted checkpoint already locally ranks nearly all searched-prefix training tensors correctly. The blocker is not simple one-step fit on those tensors.

## Replay Of Searched Winning Starts

Launcher change:

- `scripts/run_spy_line_replay_probe.ps1` now accepts `-ModelDModel` and `-ModelNumLayers`; accepted Spy profile is 128x2, not the previous hardcoded 256x4.

Accepted unforced replay:

- Run: `20260508_replay_shallow_search_accepted_unforced`
- Result: `408/504` matched, `52/84` searched starts won

Accepted forced-prefix replay:

- Run: `20260508_replay_shallow_search_accepted_forcedprefix`
- Result: `414/504` matched, `82/84` groups with at least one forced partial-prefix win

The policy can often execute searched states locally, but unforced play drifts. The failure is sequential distribution/compounding, not a lack of local logit knowledge.

## Bulk Prefix Fit Control

Profile:

- `Pauper-Spy-Combo-ShallowSearchPrefixFit-20260508`

Training:

- Run: `20260508_shallow_search_prefix_fit_e20p4`
- Import: 489 records from the 31-node search data
- 20 epochs, 4 candidate permutations, direct BC + MCTS KL, distill-head-only

Results:

- Score: `462/490` top-1, mean target probability fell to `0.887991`
- Same-start unforced replay: `56/84` wins, `395/504` matched
- Reduced CP1 eval: `20260508_shallow_search_prefixfit_cp1_eval16`, `12/64`

Conclusion: bulk positive prefix distillation does not transfer and should not be scaled.

## Prefix-Sibling Contrast

Dataset:

- Run: `20260508_spy_prefix_sibling_contrast_natural_cp1`
- 91 scenarios processed before `StopAfterExamples=300`
- 63 trained scenarios, 28 skipped, 303 candidate examples, 36 winning trajectories, 269 selected examples

Accepted score:

- Run: `20260508_score_prefix_sibling_contrast_accepted`
- Result: `249/269` top-1, target-set `264/269`, mean target probability `0.898292`

Conclusion: this sibling-contrast dataset is also mostly already locally solved by the accepted checkpoint. It is not a good next training target.

## First-Deviation Repair

Implementation:

- Added optional replay export `--replay-deviation-training-data-file`.
- `run_spy_line_replay_probe.ps1` exposes it as `-ReplayDeviationTrainingDataFile`.
- For losing unforced replay groups, the trainer serializes the first decision where the live policy diverges from the searched winning prefix, but only when the searched action can be remapped into the live candidate set by normalized text.
- Added a second DAgger-style export path, `--replay-dagger-training-data-file`, exposed as `-ReplayDaggerTrainingDataFile`.
  - Winning replay groups export self-anchor targets for the policy's actual choices.
  - Losing groups export self-anchors until the first searched-prefix divergence, then repeat the remapped repair target with `--replay-deviation-repeat`.

Accepted export:

- Run: `20260508_replay_shallow_search_accepted_deviation_export`
- Replay result: `389/504` matched, `60/84` searched starts won
- Exported repair tensors: 14

Accepted score on repair tensors:

- Run: `20260508_score_first_deviation_accepted`
- Result: `0/14` top-1, mean target probability `0.015639`

This proved the exported examples were genuinely novel local errors. Training still did not transfer.

High-dose repair:

- Profile: `Pauper-Spy-Combo-FirstDeviationRepair-20260508`
- Run: `20260508_first_deviation_repair_fit_e80p8`
- 14 examples, 80 epochs, 8 permutations, direct BC + MCTS KL, distill-head-only
- Score after fit: `11/14` top-1, mean target probability `0.663748`
- Same-start replay: `61/84` wins but only `125/504` matched
- Reduced CP1 eval: `20260508_first_deviation_repair_cp1_eval16`, `8/64`

Low-dose repair:

- Profile: `Pauper-Spy-Combo-FirstDeviationLowDose-20260508`
- Run: `20260508_first_deviation_lowdose_fit_e12p1`
- 14 examples, 12 epochs, 1 permutation, lower loss coefficients
- Score after fit: `0/14` top-1, mean target probability `0.049645`
- Same-start replay: `55/84` wins, `396/504` matched

Anchored DAgger repair:

- Export: `20260508_replay_shallow_search_accepted_dagger_export_r16`
- Exported 621 tensors using 16x deviation repeats.
- Accepted score on mixed set: `461/621` top-1, mean target probability `0.719122`
- Profile: `Pauper-Spy-Combo-DaggerAnchoredR16-20260508`
- Fit: `20260508_dagger_anchored_r16_fit_e8p2`
- Score after fit: `495/621` top-1, mean target probability `0.643586`
- Same-start replay: `50/84` wins, `389/504` matched

Scaled anchored DAgger:

- Search corpus: `20260509_spy_shallow_search_autopilot_cp1_n512_nodes31_ser`
- Settings: 512 scenarios, 31 nodes, depth 6, 90s per-scenario cap
- Search result: 265/512 searched wins, 1,561 prefix tensors; many skipped scenarios were timeouts.
- Accepted score on prefix tensors: `1527/1561` top-1, mean target probability `0.940167`.
- Accepted replay/export: `20260509_replay_shallow_search_n512_accepted_dagger_export_r16`
- Accepted replay result: `168/265` searched starts won, `1301/1590` matched.
- Exported 22 pure first-deviation tensors and 1,800 anchored DAgger tensors with 16x deviation repeats.
- Accepted score:
  - pure deviation: `0/22` top-1, mean target probability `0.076324`
  - anchored DAgger: `1448/1800` top-1, mean target probability `0.781804`
- Profile: `Pauper-Spy-Combo-DaggerN512R16-20260509`
- Fit: `20260509_dagger_n512_r16_fit_e4p1`
- Score after fit: `1455/1800` top-1, mean target probability `0.698791`
- Same-start replay: `183/265` wins, `1155/1590` matched
- Reduced CP1 eval: `20260509_dagger_n512_r16_cp1_eval16`, `14/64`
  - Spy mirror: 7/16
  - Wildfire: 5/16
  - Rally: 0/16
  - Affinity: 2/16

Conclusion: first-deviation labels identify real local errors, and scaling anchored DAgger can improve same-start replay wins, but it still does not transfer to fresh CP1 starts. The supervised replay-patch surface is now exhausted unless the data source changes qualitatively.

## Wider Search Teacher

Changing the search branch set is more useful than adding another supervised learner variant.

Wide branch probe:

- Run: `20260509_spy_shallow_search_wide_top3r1_cp1_n128_nodes31`
- Settings: 128 scenarios, 31 nodes, depth 6, `top_k=3`, `random_extra=1`
- Search result: 106/128 searched wins, 625 exported prefix tensors, elapsed 2517.4s.
- First 64 shared scenarios: 51 wins versus 41 for the original `top_k=2`, `random_extra=0` 31-node run.
- Accepted score on prefix tensors: `607/625` top-1, mean target probability `0.9445`.
- Accepted replay/export: `20260509_replay_wide_top3r1_n128_accepted_dagger_export_r16`
- Replay result: `60/106` searched starts won, `512/636` matched.
- Exported 13 pure first-deviation tensors and 777 anchored DAgger tensors.
- Accepted score:
  - pure deviation: `0/13` top-1, mean target probability `0.083648`
  - anchored DAgger: `569/777` top-1, mean target probability `0.734093`

Top-3-only ablation:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n64_nodes31`
- Settings: 64 scenarios, 31 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 53/64 searched wins, 311 exported prefix tensors, elapsed 1237.5s.
- Accepted score on prefix tensors: `308/311` top-1, mean target probability `0.952326`.
- Accepted replay/export: `20260509_replay_top3r0_n64_accepted_dagger_export_r16`
- Replay result: `31/53` searched starts won, `254/318` matched.
- Exported 6 pure first-deviation tensors and 381 anchored DAgger tensors.
- Accepted score:
  - pure deviation: `0/6` top-1, mean target probability `0.018936`
  - anchored DAgger: `285/381` top-1, mean target probability `0.725489`

Conclusion: `top_k=3` appears to capture the useful teacher gain without needing random extra branches. The stronger teacher still shows the same distillation failure: accepted logits already score the searched prefix tensors almost perfectly, but unforced replay loses many starts. The next useful teacher-side experiment is a full n128 `top_k=3`, `random_extra=0` run.

Full top-3-only confirmation:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes31`
- Settings: 128 scenarios, 31 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 111/128 searched wins, 657 exported prefix tensors, elapsed 2301.8s.
- Accepted score on prefix tensors: `634/657` top-1, mean target probability `0.938368`.
- Accepted replay/export: `20260509_replay_top3r0_n128_accepted_dagger_export_r16`
- Replay result: `70/111` searched starts won, `542/666` matched.
- Exported 18 pure first-deviation tensors and 892 anchored DAgger tensors.
- Accepted score:
  - pure deviation: `0/18` top-1, mean target probability `0.085211`
  - anchored DAgger: `604/892` top-1, mean target probability `0.682469`

Top-3 DAgger distillation:

- Profile: `Pauper-Spy-Combo-Top3R0DaggerR16-20260509`
- Seeded from `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Fit: `20260509_top3r0_dagger_r16_fit_e4p1`
- Data: 892 anchored DAgger tensors, 4 epochs, 1 candidate permutation, `MCTS_KL_LOSS_COEF=0.5`, direct BC coefficient `0.1`, distill-head-only
- Score after fit: `620/892` top-1, mean target probability `0.669081`
- Same-start replay: `71/111` wins, `519/666` matched
- Reduced CP1 eval: `20260509_top3r0_dagger_r16_cp1_eval16`, `12/63` completed games
  - Spy mirror: 5/16
  - Wildfire: 3/15
  - Rally: 2/16
  - Affinity: 2/16

Conclusion: top-3 search is the best teacher so far, but the DAgger-style supervised profile still does not transfer. The fit barely improves same-start wins and hurts exact prefix matching; CP1 is far below the 48/128 historical gate.

Budget curve:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes15`
- Settings: 128 scenarios, 15 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 104/128 searched wins, 615 exported prefix tensors, elapsed 1977.0s.
- Accepted score on prefix tensors: `598/615` top-1, mean target probability `0.941917`.
- Accepted replay: `20260509_replay_top3r0_n128_nodes15_accepted`, `62/104` searched starts won, `519/624` matched.

The 15-node point retains most of the 31-node teacher quality but saves only about 14% wall-clock (`1977s` versus `2302s`). Node count is not the main runtime lever. Top-3 search finds many late terminal wins:

- `top_k=3`, nodes 31: average winning turn `12.91`; turn distribution includes 34 wins after turn 17.
- original `top_k=2`, nodes 31: average winning turn `8.10`; all 84 wins were by turn 9.

Next budget experiment: cap terminal rollouts with `max_game_turns=17`. This should test whether the useful top-3 teacher can be made cheaper by pruning very late wins and long losses.

Terminal turn cap:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes31_turn17`
- Settings: 128 scenarios, 31 nodes, depth 6, `top_k=3`, `random_extra=0`, `max_game_turns=17`
- Search result: 103/128 searched wins, 610 exported prefix tensors, elapsed 2618.5s.

Conclusion: the turn cap is negative. It loses 8 wins versus uncapped nodes31 and is slower, likely because late wins become no-win searches that continue exploring other branches. Do not use a hard turn cap as the first runtime lever.

Cheap top-3 point:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes7`
- Settings: 128 scenarios, 7 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 100/128 searched wins, 594 exported prefix tensors, elapsed 1136.9s.
- Accepted score on prefix tensors: `590/594` top-1, mean target probability `0.962315`.
- Accepted replay: `20260509_replay_top3r0_n128_nodes7_accepted`, `66/100` searched starts won, `483/600` matched.

Conclusion: nodes7 is the best cost-quality point so far. It keeps most of the top-3 teacher gain while cutting wall-clock roughly in half versus nodes31. This is the current candidate budget for an online/search-improvement prototype.

Too-cheap point:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes3`
- Settings: 128 scenarios, 3 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 75/128 searched wins, 445 exported prefix tensors, elapsed 918.4s.
- Accepted score on prefix tensors: `443/445` top-1, mean target probability `0.965543`.

Conclusion: nodes3 is below the useful knee. It is cheaper than nodes7, but gives up too much teacher quality and is not better than earlier narrow 15-node search. The local budget curve now points to nodes7 as the smallest useful teacher budget.

Intermediate cheap point:

- Run: `20260509_spy_shallow_search_top3r0_cp1_n128_nodes5`
- Settings: 128 scenarios, 5 nodes, depth 6, `top_k=3`, `random_extra=0`
- Search result: 87/128 searched wins, 518 exported prefix tensors, elapsed 1154.8s.
- Accepted score on prefix tensors: `515/518` top-1, mean target probability `0.961849`.

Conclusion: nodes5 is also below the useful knee. It is much weaker than nodes7 and slightly slower on this seed. The budget sweep should stop here; nodes7 is the smallest useful top-3 teacher setting.

## Online Prefix Search Prototype

Implementation:

- Added opt-in eval hook in `ComputerPlayerRL` behind `RL_ONLINE_PREFIX_SEARCH_ENABLE=1`.
- Added `TerminalPrefixSearch`, a clone-based terminal prefix search that tries root action prefixes and returns only an alternate root action.
- Defaults remain off. Same-action simulated wins no longer terminate search, because they do not improve live play and were producing false confidence.

Smoke/debug sequence:

- `20260509_online_prefix_smoke_g1_fix1`: fixed clone player identity by copying the simulated `ComputerPlayerRL` instead of constructing a fresh player id.
- `20260509_online_prefix_smoke_g1_inline`: fixed thread ownership by running simulation inline on the current game thread; background `sim.resume()` violates XMage game-thread checks.
- `20260509_online_prefix_smoke_g1_mctssim`: `MCTSSimPlayer` backend was clean and bounded, but too weak for Spy because it only controls priority/targets.
- `20260509_online_prefix_smoke_g1_nomodel`: no-model `ComputerPlayerRL` search subclass can find simulated terminal lines, but those were same-action wins and did not translate to live wins.

Reduced evals:

- `20260509_online_prefix_nomodel_n3_eval8`: 0/8. Logged 24 search calls, 10 simulated terminal wins, 0 overrides. All simulated wins were the same root action the accepted policy already chose.
- `20260509_online_prefix_nomodel_n3_alt_eval4`: after changing the objective to ignore same-action wins, 1/4 with 12 search calls and 0 alternate-root wins.

Conclusion: this online root override surface is negative in its current form. The faithful `ComputerPlayerRL` branch executor controls generic Spy choices but is slow and overoptimistic; the fast `MCTSSimPlayer` executor is bounded but lacks enough non-priority choice control. Do not scale this hook as a CP1 gate until branch simulation fidelity is improved or it is folded into a real search-as-training loop.

## Conclusion

Terminal prefix search is the useful discovery: 31 nodes with the original narrow branch set finds 84/128 natural-start wins, and widening to top-3 actions raises the teacher above that. Distilling those lines with the current supervised import path is not working because the accepted policy already solves most individual tensors and fails through compounding drift.

## Multi-Ply MCTS Revisit

After the supervised repair failures, a small accepted-checkpoint probe revisited the existing multi-ply ISMCTS implementation with a broader budget:

- Run: `20260508_accepted_multiply_mcts_unlimited_probe_g2`
- Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- CP1, 2 games per matchup, 1 game per JVM job
- `MULTI_PLY_MCTS=1`
- `MCTS_ITERATIONS=8`
- `MCTS_DETERMINIZATIONS=1`
- `MCTS_MAX_OUR_ACTIONS=0`
- `MCTS_TIMEOUT_MS=2500`
- `MCTS_ITER_TIMEOUT_MS=1000`
- `MCTS_SKIP_TOP_PROB=0.99`
- no selective gate

Result: 3/8 total.

Matchups:

- Spy mirror: 2/2
- Wildfire: 1/2
- Rally: 0/2
- Affinity: 0/2

Runtime was prohibitive for scaling: single-game jobs took up to 325.6s, with 90 MCTS activations in one Wildfire game and 87 in one mirror game. This does not justify an AlphaZero-scale loop with the current online MCTS/value evaluator.

Next useful work should be one of:

- make terminal prefix search available as an online policy/improvement operator,
- build a real DAgger loop with replay-state anchoring and repeated first-deviation collection,
- or move to the planned belief/ISMCTS or AlphaZero-scale implementation.

Do not continue plain positive prefix BC, sibling-contrast BC, or tiny first-deviation patches as standalone profile variants.
