# 2026-05-06 Spy Fast-Line Imitation And Mulligan CF

## Goal

Test whether terminal-only Spy Combo imitation can seed real play, then test whether paired terminal mulligan counterfactual repair improves that seed without violating the terminal-reward thesis.

## Baseline Seed

Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`

Training:
- Source checkpoint: `Pauper-Spy-Combo-TerminalWin-BalMullCF-20260506`
- Collected 501 actual terminal WIN trajectories using turn-5 max game length and prefix sibling contrast.
- Offline BC trained 8,498 examples for 4 epochs.
- Mulligan training was skipped.

CP1 eval:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 23 | 32 | 71.88% |
| Jund Wildfire | 6 | 32 | 18.75% |
| Mono Red Rally | 1 | 32 | 3.12% |
| Grixis Affinity | 3 | 32 | 9.38% |
| Total | 33 | 128 | 25.78% |

Conclusion: terminal win trajectory BC can seed Spy mirror play above 70%, but does not yet produce robust fast-matchup play.

## Rejected Mulligan CF Runs

### Full-Model CF

Profile: `Pauper-Spy-Combo-FastT5ContrastMullCF-20260506`

Training:
- 1024 paired keep-vs-mull games against CP1.
- 333 trained labels, 691 skipped.
- Label mix: 217 `TIE_SOFT`, 72 `KEEP`, 44 `MULLIGAN`.

CP1 eval:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 13 | 32 | 40.62% |
| Jund Wildfire | 4 | 32 | 12.50% |
| Mono Red Rally | 2 | 32 | 6.25% |
| Grixis Affinity | 4 | 32 | 12.50% |
| Total | 23 | 128 | 17.97% |

Diagnosis: `scripts/run_mulligan_counterfactual.ps1` claimed head-only isolation but did not set `DISTILL_HEAD_ONLY=1`, so the BC loss updated the shared policy path and damaged the fast-line seed.

### Head-Only CF

Profile: `Pauper-Spy-Combo-FastT5ContrastMullHeadCF-20260506`

Training:
- Same source seed as above.
- 1024 paired keep-vs-mull games against CP1.
- 307 trained labels, 717 skipped.
- Label mix: 195 `TIE_SOFT`, 68 `KEEP`, 44 `MULLIGAN`.
- Verified by checkpoint diff: only 5 tensors changed, all under `policy_scorer_mulligan.*`.

CP1 eval:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 11 | 32 | 34.38% |
| Jund Wildfire | 3 | 32 | 9.38% |
| Mono Red Rally | 1 | 32 | 3.12% |
| Grixis Affinity | 5 | 32 | 15.62% |
| Total | 20 | 128 | 15.62% |

Conclusion: the root paired mulligan CF target is harmful for Spy as currently defined, even when isolated to the mulligan head. Do not continue this branch without redesigning the target generation.

## Code Fix

Patched `scripts/run_mulligan_counterfactual.ps1` to expose and set:
- `DISTILL_HEAD_ONLY=1` by default
- `DISTILL_POLICY_PATH_ONLY=0` by default

This prevents future mulligan CF experiments from silently updating the shared trunk when the intent is head-only training.

## Next Experiment

Return to `Pauper-Spy-Combo-FastT5Contrast-20260506` as the best checkpoint.

Run logged fast-matchup eval against Rally and Affinity, then use those logs to drive a post-mulligan action/line repair experiment. Mulligan repair is paused. The likely target is more turn-4/turn-5 terminal winning-line contrast focused on fast matchups, with `SkipMulliganTraining=true`.

## Fast-Matchup Diagnostic

Run: `20260506_spy_fastt5contrast_logged_rally_affinity_cp1`

Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`

Result:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 0 | 8 | 0.00% |
| Grixis Affinity | 0 | 8 | 0.00% |
| Total | 0 | 16 | 0.00% |

Action-health findings:
- Land play is no longer the blocker: `pass_over_land=0`, `land_play_selected=61/65`.
- Mulligan is still crude: `16/16` keeps, `15/16` keeps with 0-1 true lands. For Spy that is not automatically wrong because `Land Grant` and landcyclers are pseudo-lands, but it is not selective.
- The core fast-matchup failure is premature Spy execution: `9/9` Spy casts happened with hidden true lands still estimated in deck, and there were `0` no-hidden-land Spy-cast opportunities.
- Dread Return is still sometimes premature: `3/3` flashbacks were not combo-ready.

Updated next step: run a focused fast-matchup line-search smoke using legal generated Spy-reachable openers versus Rally/Affinity only. Keep terminal-win-only labels and skip mulligan training.

## Focused Fast-Matchup Line Search

Run: `20260506_spy_fastmatch_reachable_t5_250`

Profile: `Pauper-Spy-Combo-FastMatchReachT5-20260506`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponents during collection: Rally/Affinity fast-match pool.
- Agent openers: generated Spy-reachable hand pool.
- Collection stopped at the 2-hour cap with 218 terminal winning trajectories.
- Offline BC imported 3,703 base examples, 14,812 train passes across 4 epochs.

CP1 eval: `20260506_spy_fastmatch_reachable_t5_250_cp1_218`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 7 | 32 | 21.88% | +1 win |
| Mono Red Rally | 2 | 32 | 6.25% | +1 win |
| Grixis Affinity | 7 | 32 | 21.88% | +4 wins |
| Total | 32 | 128 | 25.00% | -1 win |

Conclusion: focused terminal-line BC improves the fast-matchup signal, especially Affinity, but damages the Spy mirror enough that the checkpoint is not better overall. The next diagnostic is a logged CP1 sweep on this checkpoint to determine whether it learned better combo timing or merely shifted failure modes.

Logged diagnostic: `20260506_spy_fastmatch_reach_logged_cp1`

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 3 | 8 | 37.50% |
| Jund Wildfire | 0 | 8 | 0.00% |
| Mono Red Rally | 2 | 8 | 25.00% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Total | 6 | 32 | 18.75% |

Action-health findings:
- Land play remains healthy: `pass_over_land=0`, `land_play_selected=207/221`.
- The model created more true Spy windows than the baseline diagnostic: `spy_cast_no_hidden_lands_opportunities=8/176`, but selected Spy in only `2/8` of those windows.
- Premature Spy remains the dominant failure: `19/21` Spy casts still had hidden true lands.
- Dread Return timing is worse than acceptable: `9/10` Dread Return flashbacks were not combo-ready.

Updated next step: run replay-mixed offline BC from the accepted baseline checkpoint, combining the original 501 broad terminal-win trajectories with the new 218 fast-match trajectories. This tests whether fast-match gains can be added without overwriting the mirror seed.

## Replay-Mixed BC From Baseline

Run: `20260506_spy_base501_fast218_mix_replay`

Profile: `Pauper-Spy-Combo-FastT5MixReach-20260506`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Import data: 501 broad terminal-win trajectories plus 218 focused fast-match trajectories.
- Imported 12,228 base examples, 48,912 train passes across 4 epochs.
- Used direct/hardened/balanced BC settings.

CP1 eval: `20260506_spy_base501_fast218_mix_replay_cp1_719_v2`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 15 | 32 | 46.88% | -8 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 23 | 128 | 17.97% | -10 wins |

Conclusion: replaying mixed data on top of the already-trained baseline with direct hardened BC is destructive. Next test the same mixed dataset from the pre-fastline source checkpoint with softer KL/BC targets, matching the original broad baseline training path more closely.

## Replay-Mixed BC From Source, Soft Targets

Run: `20260506_spy_base501_fast218_mix_fromsource_soft`

Profile: `Pauper-Spy-Combo-FastT5MixSource-20260506`

Training:
- Source checkpoint: `Pauper-Spy-Combo-TerminalWin-BalMullCF-20260506`
- Import data: same 501 broad terminal-win trajectories plus 218 focused fast-match trajectories.
- Imported 12,228 base examples, 48,912 train passes across 4 epochs.
- Used the softer original training path: no direct BC loss, no hardened binary targets, no balanced binary targets.

CP1 eval: `20260506_spy_base501_fast218_mix_fromsource_soft_cp1_719`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 3 | 32 | 9.38% | -3 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 2 | 32 | 6.25% | -1 win |
| Total | 17 | 128 | 13.28% | -16 wins |

Conclusion: mixed all-step replay is destructive even when retrained from the pre-fastline source with softer targets. The failure is not just replay order or overly hard targets. The next branch should stop cloning every action in a terminal-winning line and instead train only the critical Spy execution decisions: when to cast `Balustrade Spy`, when to cast/flashback `Dread Return`, and the associated combo target choices.

## Critical-Action-Only Terminal Imitation

Run: `20260506_spy_critical_fast_t5_single250_w128`

Profile: `Pauper-Spy-Combo-CriticalFastT5W128-20260506`

Code change:
- Added `--include-action-text-regex` to `ActionCounterfactualTrainer`, `run_spy_line_search.ps1`, and `run_spy_imitation_offline_bc_experiment.ps1`.
- The filter is generation-time for new serialized data. Old `.ser` files cannot be filtered by action text because serialized `TrainingData` stores tensors and action IDs, not candidate text.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Collection: single JVM, 16,000 generated Spy-reachable scenarios, Rally/Affinity opponents, turn-5 terminal wins, 128 workers.
- The run did not hit the 250 trajectory target. Final collection was 103 winning trajectories, 2,009 candidate examples, 626 selected critical examples.
- Offline BC imported 626 examples, 2,504 train passes across 4 epochs.
- Selected labels: 153 `Balustrade Spy` targets, 97 `Dread Return` targets, 91 `Lotleth Giant` targets, and 285 nearby candidate-context decisions.

CP1 eval: `20260506_spy_critical_fast_t5_single250_w128_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 7 | 32 | 21.88% | -16 wins |
| Jund Wildfire | 4 | 32 | 12.50% | -2 wins |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 2 | 32 | 6.25% | -1 win |
| Total | 14 | 128 | 10.94% | -19 wins |

Logged diagnostic: `20260506_spy_critical_w128_logged_cp1`

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 2 | 8 | 25.00% |
| Jund Wildfire | 2 | 8 | 25.00% |
| Mono Red Rally | 1 | 8 | 12.50% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Total | 6 | 32 | 18.75% |

Action-health findings:
- Land play remains healthy: `pass_over_land=0`, `land_play_selected=155/167`.
- Critical BC created real no-hidden-land Spy windows: `spy_cast_no_hidden_lands_opportunities=12`, but selected Spy in only `1/12`.
- Premature Spy remains dominant: `15/16` Spy casts still had hidden true lands.
- Dread Return timing is still wrong: `13/14` flashbacks were not combo-ready, and `11/14` happened with no `Lotleth Giant` in graveyard.
- When `Lotleth Giant` is actually available as a Dread Return target, target choice is fine: `6/6`.

Conclusion: action-filtered terminal imitation still overgeneralizes combo actions and does not produce negative timing labels. The next branch should use terminal-only branch counterfactuals at critical decision states, not imitation from successful trajectories. That means branch from states where `Balustrade Spy`, `Dread Return`, or `Lotleth Giant` is legal, roll candidates to terminal, and train on terminal outcome contrast.

## Critical-Action Branch Counterfactual Smoke

Run: `20260506_spy_actioncf_critical_smoke256`

Profile: `Pauper-Spy-Combo-ActionCFCriticalSmoke-20260506`

Code change:
- `ActionCounterfactualTrainer` now applies `--include-action-text-regex` before branch rollouts in non-winning-prefix mode. This avoids spending rollout budget on generic decisions when the experiment only needs critical combo timing states.
- `run_action_counterfactual.ps1` now accepts agent/opponent opening-hand pool files and the action-text regex filter.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Agent opening hands: generated Spy-reachable pool from `20260506_spy_critical_fast_t5_single250_w128`.
- Opponents during collection: Rally/Affinity fast-match pool.
- Scenarios: 256, workers: 64, top-k candidates: 4 plus 1 random extra, max decision depth: 24.
- The regex was `Balustrade Spy|Dread Return|Lotleth Giant`.
- Final data: 38 trained scenarios, 218 skipped scenarios, 45 selected branch-counterfactual examples.
- Train fit was low but plausible for a smoke: tensor top-1 `13/45` (`28.89%`) after 2 epochs.

Quick CP1 eval: `20260506_spy_actioncf_critical_smoke256_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 3 | 8 | 37.50% | inconclusive |
| Jund Wildfire | 3 | 8 | 37.50% | inconclusive |
| Mono Red Rally | 0 | 8 | 0.00% | inconclusive |
| Grixis Affinity | 2 | 8 | 25.00% | inconclusive |
| Total | 8 | 32 | 25.00% | approximately baseline on a tiny sample |

Label-health finding:
- The selected labels include both `Balustrade Spy` as the best terminal action and non-combo actions like `Wall of Roots`, `Swamp`, and `Forest` at decision points where a combo action was legal. That means this collector can provide the missing negative timing signal that terminal imitation lacks.

Conclusion: the smoke is not evidence of improvement, but it is evidence that the branch-counterfactual formulation is viable and not immediately destructive. Scale this next from the accepted baseline with more scenarios and more train epochs, then run a full CP1 sweep.

## Scaled Critical-Action Branch Counterfactual

Run: `20260506_spy_actioncf_critical_s2048`

Profile: `Pauper-Spy-Combo-ActionCFCriticalS2048-20260506`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Agent opening hands: generated Spy-reachable pool from `20260506_spy_critical_fast_t5_single250_w128`.
- Opponents during collection: Rally/Affinity fast-match pool.
- Scenarios: 2,048, workers: 96, top-k candidates: 4 plus 1 random extra, max decision depth: 24.
- Final data: 144 selected branch-counterfactual examples, 576 train passes across 4 epochs.
- Tensor replay fit stayed weak: `41/144` top-1 (`28.47%`).

CP1 eval: `20260506_spy_actioncf_critical_s2048_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 14 | 32 | 43.75% | -9 wins |
| Jund Wildfire | 8 | 32 | 25.00% | +2 wins |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 26 | 128 | 20.31% | -7 wins |

Conclusion: scaled action-CF is not an improvement. The small Wildfire gain does not offset mirror regression, and fast-matchups did not improve.

## Direct-Fit Action-CF Probe

Run: `20260506_spy_actioncf_directfit_w48_probe`

Profile: `Pauper-Spy-Combo-ActionCFDirectFitW48-20260506`

Reason:
- The scaled KL-only run did not fit even a small 144-example branch-label set.
- This tested whether stronger direct supervised pressure can fit critical action-CF labels, and whether better fit translates to play.

Training data:
- Recollected critical action-CF labels with lower concurrency because the 96-worker collect/export path caused Py4J candidate-scoring failures.
- Workers: 48, stop after 160 examples, same generated Spy-reachable hand pool, Rally/Affinity opponents.
- Export produced 160 serialized labels from 863 scenarios.

Fit probe:
- Direct BC, coefficient 3.0, 25 epochs, 4,000 train passes.
- Before: top-1 `46/160` (`28.75%`), target-set top-1 `122/160` (`76.25%`).
- After: top-1 `78/160` (`48.75%`), target-set top-1 `139/160` (`86.88%`).

CP1 eval: `20260506_spy_actioncf_directfit_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 9 | 32 | 28.12% | -14 wins |
| Jund Wildfire | 2 | 32 | 6.25% | -4 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 2 | 32 | 6.25% | -1 win |
| Total | 13 | 128 | 10.16% | -20 wins |

Conclusion: stronger fitting made action-CF worse. The problem is not simply that the KL loss was too weak. These labels are likely contaminated by the continuation policy: a single forced action followed by the current greedy policy does not reliably measure whether the forced action belongs to a winning Spy line.

Updated next direction:
- Do not scale this single-action branch-CF path.
- The next promising branch is line-level terminal search again, but with better data mix and diagnostics: the accepted checkpoint came from turn-5 terminal winning-prefix search with tactic autopilot, no model scoring during search, and sibling contrast. Fast-match focused single-action repair did not help.

## Synthetic-Return / History Probe

Run directory: `local-training/analysis/synthetic_return_20260506`

Purpose:
- Test whether exported decision-level features can predict eventual terminal outcome well enough to motivate synthetic-return wiring or previous-state history in the main model.
- This is only an offline diagnostic. It does not train the MTGRL model.

Inputs:
- `local-training/analysis/spy_wildfire_recent_trajectories.jsonl`
- `local-training/analysis/spy_recent_trajectories_sample.jsonl`

Results for `PlayerRL1` decisions:

| History Window | Test AUC | Test Loss | Base Loss |
| ---: | ---: | ---: | ---: |
| 0 | 0.5338 | 0.7026 | 0.6388 |
| 4 | 0.5430 | 0.7014 | 0.6388 |
| 8 | 0.5456 | 0.7017 | 0.6388 |
| 16 | 0.5468 | 0.7016 | 0.6388 |

All-actor history-16 check:
- Test AUC `0.5499`
- Test loss `0.7088`
- Base loss `0.6432`

Conclusion:
- The coarse exported features do not support synthetic returns yet. History gives a small AUC bump, but loss is still worse than a base-rate predictor.
- Do not do a large architecture migration just to add previous-state history based on this signal. If we revisit history, it should use richer native state tensors, not these coarse log-derived flags.

## 1k Terminal-Line Scaling

Run: `20260506_spy_terminalwin_t5_contrast_1000_w96`

Profile: `Pauper-Spy-Combo-FastT5Contrast1k-20260506`

Reason:
- Test whether the accepted fast terminal-line method improves simply by adding more winning-prefix data.
- The run seeded the new training directory with the accepted 501-trajectory dataset, then collected 501 additional turn-5 terminal-win trajectories.

Training:
- Source checkpoint: `Pauper-Spy-Combo-TerminalWin-BalMullCF-20260506`
- Seeded import files: 172 existing `.ser` files from `20260506_spy_terminalwin_t5_contrast_500_w96`
- New collection: 501 terminal winning trajectories over 176 batches
- Total import files: 348
- Training examples: 17,071
- Train passes: 68,284 across 4 epochs
- Search settings matched the accepted baseline: terminal mode `WIN`, max game turns 5, max prefix depth 16, train prefix depth 20, tactic autopilot, no search model scoring, and sibling contrast with 80 search nodes.

CP1 eval: `20260506_spy_terminalwin_t5_contrast_1000_w96_cp1_501`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 11 | 32 | 34.38% | -12 wins |
| Jund Wildfire | 3 | 32 | 9.38% | -3 wins |
| Mono Red Rally | 5 | 32 | 15.62% | +4 wins |
| Grixis Affinity | 7 | 32 | 21.88% | +4 wins |
| Total | 26 | 128 | 20.31% | -7 wins |

Conclusion:
- Do not promote this checkpoint. More broad terminal-line data did not improve the accepted result; it traded mirror and Wildfire strength for small Rally/Affinity gains.
- The next controlled check is whether the extra data was useful but overtrained. Reuse the same 348 serialized files from the original source checkpoint with a much smaller import-training dose.

## 1k Terminal-Line Reduced Dose

Run: `20260507_spy_terminalwin_t5_contrast_1k_e1`

Profile: `Pauper-Spy-Combo-FastT5Contrast1kE1-20260507`

Reason:
- Test whether the 1k terminal-line dataset was useful but overtrained.
- Reused the same 348 serialized files from `20260506_spy_terminalwin_t5_contrast_1000_w96`, but imported for only one epoch.

Training:
- Source checkpoint: `Pauper-Spy-Combo-TerminalWin-BalMullCF-20260506`
- Import examples: 17,071
- Train passes: 17,071 across 1 epoch
- Import batching fix was active: serialized files were accumulated across the epoch and drained once at epoch end instead of draining after every file.

CP1 eval: `20260507_spy_terminalwin_t5_contrast_1k_e1_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 4 | 32 | 12.50% | +1 win |
| Total | 22 | 128 | 17.19% | -11 wins |

Conclusion:
- Reduced-dose replay did not rescue the 1k data. The additional terminal-line mix is harmful, not merely overtrained.

## Accepted-Checkpoint New-Only Replay

Run: `20260507_spy_terminalwin_t5_contrast_newonly_from_accepted_e1`

Profile: `Pauper-Spy-Combo-FastT5ContrastNewOnlyE1-20260507`

Reason:
- Test whether only the 176 new files from the 1k extension improve the accepted checkpoint when applied as a light incremental update.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Import files: 176 new `.ser` files
- Import examples: 8,573
- Train passes: 8,573 across 1 epoch

CP1 eval: `20260507_spy_terminalwin_t5_contrast_newonly_from_accepted_e1_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 8 | 32 | 25.00% | +2 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 1 | 32 | 3.12% | -2 wins |
| Total | 25 | 128 | 19.53% | -8 wins |

Conclusion:
- The new-only replay is also not promotable. It does show the same pattern as other fast-line variants: some Wildfire gain, but mirror and Rally damage.

## Accepted-Checkpoint Logged Diagnostic

Run: `20260507_spy_fastt5_accepted_cp1_logged`

Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`

Small logged CP1 sample:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Total | 5 | 32 | 15.62% |

Action-health findings:
- Land play is healthy: `pass_over_land=0/177`, `land_play_selected=167/177`.
- The core failure is unsafe Spy timing: `spy_cast_hidden_land_opportunities=37/40`, `spy_casts=16/40`, and `spy_casts_with_hidden_lands=15/16`.
- Dread Return target choice is mostly fine when the good target is available: `dread_return_target_lotleth_when_available=9/9`.

Conclusion:
- The accepted checkpoint has a clear midrange trap: it often casts `Balustrade Spy` while true lands are still hidden in library. Future repair should create negative labels for premature Spy, not just more positive winning-line imitation.

## Strict Landless Terminal Imitation

Run: `20260507_spy_landless_combo_win_t5_w96`

Profile: `Pauper-Spy-Combo-LandlessComboWinT5-20260507`

Code change:
- Added terminal mode `SPY_LANDLESS_COMBO_WIN` to `ActionCounterfactualTrainer`.
- This terminal mode requires an actual win plus landless Spy-combo evidence: no true lands in library, `Lotleth Giant` resolved or pending, and `Dread Return` in graveyard/exile/stack.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Strict positive trajectories: 152
- Import examples: 2,606
- Train passes: 10,424 across 4 epochs

CP1 eval: `20260507_spy_landless_combo_win_t5_w96_cp1_152`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 14 | 32 | 43.75% | -9 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 2 | 32 | 6.25% | +1 win |
| Grixis Affinity | 1 | 32 | 3.12% | -2 wins |
| Total | 22 | 128 | 17.19% | -11 wins |

Conclusion:
- Strict positive-only terminal imitation is still not enough. It filters for real combo wins, but it does not teach the model which tempting Spy casts are bad.

## Strict Spy-Cast Branch Counterfactual

Run: `20260507_spy_strict_spycast_cf_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCF-20260507`

Reason:
- Generate direct terminal contrast at decision states where `Balustrade Spy` is legal.
- Use `SPY_LANDLESS_COMBO_WIN` as the terminal success criterion so premature Spy casts are penalized by terminal outcome, not by a heuristic reward.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponents during collection: Rally/Affinity fast-match pool.
- Scenarios: 2,048
- Trained scenarios: 83
- Selected examples: 91
- Train passes: 728 across 8 epochs
- Fit: tensor top-1 `33/91` (`36.26%`)
- Loss mode: KL plus direct/hardened BC pressure.

CP1 eval: `20260507_spy_strict_spycast_cf_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 14 | 32 | 43.75% | -9 wins |
| Jund Wildfire | 9 | 32 | 28.12% | +3 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 7 | 32 | 21.88% | +4 wins |
| Total | 30 | 128 | 23.44% | -3 wins |

Conclusion:
- This is the best repair signal since the accepted checkpoint: it improves Wildfire and Affinity, but damages Spy mirror and Rally enough to miss promotion.
- The next control is a KL-only version of the same strict Spy-cast branch-CF. If KL preserves more of the accepted policy while still nudging Spy timing, it may be the better value-speed tradeoff.

## Strict Spy-Cast Branch Counterfactual, KL-Only Control

Run: `20260507_spy_strict_spycast_cf_kl_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCFKL-20260507`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Same collection setup as `20260507_spy_strict_spycast_cf_w48`.
- Scenarios: 2,048
- Trained scenarios: 83
- Selected examples: 90
- Train passes: 720 across 8 epochs
- Fit: tensor top-1 `25/90` (`27.78%`)
- Loss mode: KL only, no direct BC, no hardened targets.

CP1 eval: `20260507_spy_strict_spycast_cf_kl_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 1 | 32 | 3.12% | -2 wins |
| Total | 23 | 128 | 17.97% | -10 wins |

Conclusion:
- KL-only preserves more of the mirror policy than direct/hardened training, but loses the Affinity/Wildfire gains. The strict Spy-cast signal only moved the target behavior when direct supervised pressure was present.
- Next control: keep hardened direct labels but lower `BC_DIRECT_LOSS_COEF` from `0.5` to `0.2` to test whether there is a better pressure point between fast-matchup repair and mirror retention.

## Strict Spy-Cast Branch Counterfactual, Direct 0.2

Run: `20260507_spy_strict_spycast_cf_direct02_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCFDirect02-20260507`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Same collection setup as the direct 0.5 and KL-only controls.
- Scenarios: 2,048
- Trained scenarios: 72
- Selected examples: 77
- Train passes: 616 across 8 epochs
- Fit: tensor top-1 `28/77` (`36.36%`)
- Loss mode: KL plus direct hardened BC at coefficient `0.2`.

CP1 eval: `20260507_spy_strict_spycast_cf_direct02_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 7 | 32 | 21.88% | -16 wins |
| Jund Wildfire | 3 | 32 | 9.38% | -3 wins |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 7 | 32 | 21.88% | +4 wins |
| Total | 18 | 128 | 14.06% | -15 wins |

Conclusion:
- Direct 0.2 is worse than direct 0.5 and KL-only. Simple loss-coefficient tuning is not enough.
- Next control: keep the only useful pressure setting, direct hardened BC at `0.5`, but freeze training to the policy scorer heads. This tests whether the fast-matchup repair can be made surgical instead of rewriting the shared policy representation.

## Strict Spy-Cast Branch Counterfactual, Head-Only

Run: `20260507_spy_strict_spycast_cf_headonly_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCFHeadOnly-20260507`

Code change:
- `scripts/run_action_counterfactual.ps1` now exposes `-DistillHeadOnly` and `-DistillPolicyPathOnly`, forwarding them to `DISTILL_HEAD_ONLY` and `DISTILL_POLICY_PATH_ONLY`.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Same strict Spy-cast branch-CF collection setup.
- Scenarios: 2,048
- Trained scenarios: 81
- Selected examples: 94
- Train passes: 752 across 8 epochs
- Fit: tensor top-1 `25/94` (`26.60%`)
- Loss mode: direct hardened BC at coefficient `0.5`, but only policy scorer heads were trainable.

CP1 eval: `20260507_spy_strict_spycast_cf_headonly_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 7 | 32 | 21.88% | +1 win |
| Mono Red Rally | 2 | 32 | 6.25% | +1 win |
| Grixis Affinity | 4 | 32 | 12.50% | +1 win |
| Total | 29 | 128 | 22.66% | -4 wins |

Conclusion:
- Head-only is the best surgical variant so far: it preserves more of the accepted checkpoint and improves three non-mirror matchups slightly, but the fit is weak and total winrate is still below the accepted `33/128`.
- Next control: policy-path-only. This allows candidate embeddings and candidate-attention layers to adapt while still freezing the shared state encoder and critic.

## Strict Spy-Cast Branch Counterfactual, Policy-Path-Only

Run: `20260507_spy_strict_spycast_cf_policypath_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCFPolicyPath-20260507`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Same strict Spy-cast branch-CF collection setup.
- Scenarios: 2,048
- Trained scenarios: 95
- Selected examples: 99
- Train passes: 792 across 8 epochs
- Fit: tensor top-1 `32/99` (`32.32%`)
- Loss mode: direct hardened BC at coefficient `0.5`, with only candidate-policy routing and policy scorer heads trainable.

CP1 eval: `20260507_spy_strict_spycast_cf_policypath_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 3 | 32 | 9.38% | -3 wins |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 19 | 128 | 14.84% | -14 wins |

Conclusion:
- Policy-path-only is worse than head-only and full direct. Freezing the state encoder was not sufficient; allowing candidate routing layers to move caused broad regression without preserving the fast-matchup gain.
- Strict Spy-cast-only branch-CF is now exhausted as a promotion path:
  - Full direct `0.5`: best fast-matchup repair, `30/128`, but mirror/Rally damage.
  - Head-only direct `0.5`: best surgical variant, `29/128`, but not enough fast-matchup gain.
  - KL-only, direct `0.2`, and policy-path-only all underperform.
- Next branch: strict critical-combo action-CF across `Balustrade Spy|Dread Return|Lotleth Giant`, so the model receives terminal contrast for both execution timing and post-Spy combo completion rather than only the initial Spy cast.

## Strict Critical-Combo Branch Counterfactual

Run: `20260507_spy_strict_critical_cf_w48`

Profile: `Pauper-Spy-Combo-StrictCriticalCF-20260507`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Regex: `Balustrade Spy|Dread Return|Lotleth Giant`
- Terminal mode: `SPY_LANDLESS_COMBO_WIN`
- Scenarios: 2,048
- Trained scenarios: 66
- Selected examples: 75
- Train passes: 600 across 8 epochs
- Fit: tensor top-1 `25/75` (`33.33%`)
- Loss mode: full-model direct hardened BC at coefficient `0.5`.

CP1 eval: `20260507_spy_strict_critical_cf_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 14 | 32 | 43.75% | -9 wins |
| Jund Wildfire | 7 | 32 | 21.88% | +1 win |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 4 | 32 | 12.50% | +1 win |
| Total | 26 | 128 | 20.31% | -7 wins |

Conclusion:
- Adding Dread Return and Lotleth Giant decision points did not improve the repair. It produced fewer selected labels than Spy-cast-only and worse total eval.
- Code inspection found the likely cause: branch rollouts accepted a `tacticAutopilot` argument, but forced-action branches did not use tactic continuation after the tested action because `targetOrdinal` remained set. That means the label asked whether the current flawed policy could finish after a forced action, not whether the action belonged to a terminal winning line.
- Patch: when `tacticAutopilot` is enabled, allow tactic choices after the forced target ordinal. The next run repeats strict critical-combo CF with post-forced tactic continuation.

## Strict Critical-Combo Branch Counterfactual, Tactic Continuation

Run: `20260507_spy_strict_critical_cf_tactic_w48`

Profile: `Pauper-Spy-Combo-StrictCriticalCFTactic-20260507`

Code change:
- Forced-action branch rollouts now allow tactic-autopilot choices after the tested action ordinal when `-TacticAutopilot` is enabled.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Regex: `Balustrade Spy|Dread Return|Lotleth Giant`
- Terminal mode: `SPY_LANDLESS_COMBO_WIN`
- Tactic continuation: enabled
- Scenarios: 2,048
- Trained scenarios: 51
- Selected examples: 62
- Train passes: 496 across 8 epochs
- Fit: tensor top-1 `24/62` (`38.71%`)
- Loss mode: full-model direct hardened BC at coefficient `0.5`.

CP1 eval: `20260507_spy_strict_critical_cf_tactic_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 14 | 32 | 43.75% | -9 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 4 | 32 | 12.50% | +3 wins |
| Grixis Affinity | 1 | 32 | 3.12% | -2 wins |
| Total | 24 | 128 | 18.75% | -9 wins |

Conclusion:
- Tactic continuation did not rescue the broader critical-action target. It made selected labels even sparser than the non-tactic critical run and lost the Affinity/Wildfire improvements seen in the strict Spy-cast-only run.
- The next control applies the same tactic-continuation fix to the narrower `Balustrade Spy` cast target, because that target remains the best counterfactual action family so far.

## Strict Spy-Cast Branch Counterfactual, Tactic Continuation

Run: `20260507_spy_strict_spycast_cf_tactic_w48`

Profile: `Pauper-Spy-Combo-StrictSpyCastCFTactic-20260507`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Regex: `Balustrade Spy`
- Terminal mode: `SPY_LANDLESS_COMBO_WIN`
- Tactic continuation: enabled
- Scenarios: 2,048
- Trained scenarios: 58
- Selected examples: 62
- Train passes: 496 across 8 epochs
- Fit: tensor top-1 `26/62` (`41.94%`)
- Loss mode: full-model direct hardened BC at coefficient `0.5`.

CP1 eval: `20260507_spy_strict_spycast_cf_tactic_w48_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 9 | 32 | 28.12% | +3 wins |
| Mono Red Rally | 2 | 32 | 6.25% | +1 win |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 30 | 128 | 23.44% | -3 wins |

Conclusion:
- This matches the best single-action CF total (`30/128`) but does not beat the accepted checkpoint.
- Compared with non-tactic Spy-cast CF, tactic continuation preserved mirror and Rally better but lost the Affinity repair. The signal is still too local and too sparse.
- Single-action counterfactuals are now exhausted as a promotion path. The next experiment should train full winning prefixes or multi-action branch labels, because Spy Combo's important errors are coordinated sequences rather than one isolated cast decision.

## Strict Critical Prefix Head-Only, Fast Pool

Run: `20260507_spy_strict_critical_prefix_head_fast_w96`

Profile: `Pauper-Spy-Combo-StrictCriticalPrefixHeadFast-20260507`

Reason:
- Move past single-action counterfactuals by training labels from full strict winning prefixes.
- Keep the update surgical by filtering to critical combo contexts and distilling only policy scorer heads.
- Collect against Rally/Affinity fast-matchups, where the accepted checkpoint fails hardest.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponents during collection: Rally/Affinity fast pool.
- Terminal mode: `SPY_LANDLESS_COMBO_WIN`
- Search: tactic autopilot, no model scoring, prefix sibling contrast with 80 search nodes.
- Collected trajectories: 200
- Collection batches: 82
- Collection elapsed: 6,761.6 sec
- Trajectory throughput: 106.5/hour
- Imported training examples after critical filter: 1,224
- Train passes: 2,448 across 2 epochs
- Distillation: direct hardened BC at coefficient `0.5`, head-only.

CP1 eval: `20260507_spy_strict_critical_prefix_head_fast_w96_cp1_200`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 13 | 32 | 40.62% | -10 wins |
| Jund Wildfire | 8 | 32 | 25.00% | +2 wins |
| Mono Red Rally | 2 | 32 | 6.25% | +1 win |
| Grixis Affinity | 2 | 32 | 6.25% | -1 win |
| Total | 25 | 128 | 19.53% | -8 wins |

Conclusion:
- Prefix-level critical BC did not beat the accepted checkpoint. It gave the familiar small Wildfire improvement but failed to fix Rally/Affinity and damaged mirror.
- This weakens the case that "more terminal winning prefix BC" is the direct route from the accepted seed to robust Spy play.
- Next diagnostic: run selective search at eval time from the accepted checkpoint. If search itself improves decisions, then our distillation targets or update isolation are the problem. If search does not improve eval, the current search/tactic formulation is not a useful teacher.

## Accepted Checkpoint Selective Multi-Ply MCTS Diagnostic

Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`

Settings:
- MCTS enabled only at eval time.
- Multi-ply MCTS on.
- Selective keywords: `Balustrade Spy,Dread Return,Lotleth Giant`
- Iterations: 8
- Determinizations: 2
- Rollout depth: 2
- Skip-top-prob: `1.01` to prevent confidence gating from skipping critical states.

Smoke run: `20260507_spy_accepted_selective_multiply_mcts_smoke`

| Matchup | Wins | Games | MCTS Activations |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 2 | 4 | 39 |
| Grixis Affinity | 0 | 4 | 20 |

Expanded fast-match run: `20260507_spy_accepted_selective_multiply_mcts_fast32`

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 1 | 32 | 3.12% |
| Grixis Affinity | 3 | 32 | 9.38% |
| Total | 4 | 64 | 6.25% |

Diagnostics:
- MCTS was active and correctly selective: logs show `selectiveKeywords=[balustrade spy, dread return, lotleth giant]`.
- Total MCTS activations across the expanded run: 120.
- Average multi-ply search call was roughly 110-130 ms at 8 iterations in representative chunks.

Conclusion:
- Eval-time selective multi-ply MCTS did not improve fast-match play from the accepted checkpoint. The smoke Rally result was noise.
- This weakens the case that the current value/search stack can rescue Spy decisions online. Before abandoning the 200 strict-prefix dataset, run one underfit control: same data, head-only, but many more train passes and candidate-order permutations.

## Strict Critical Prefix Head-Only Hard-Fit Control

Run: `20260507_spy_strict_critical_prefix_head_fast_fit_e50p4`

Profile: `Pauper-Spy-Combo-StrictCriticalPrefixHeadFastFit-20260507`

Reason:
- Test whether the prior 2-epoch critical-prefix result was simply underfit.
- Reuse the same 200 strict fast-pool winning-prefix dataset.
- Keep the update head-only, but train much harder with candidate-order augmentation.

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Imported training examples after critical filter: 1,224
- Candidate-order permutations: 4
- Epochs: 50
- Train passes: 244,800
- Distillation: direct hardened BC at coefficient `0.5`, head-only.

CP1 eval: `20260507_spy_strict_critical_prefix_head_fast_fit_e50p4_cp1`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 3 | 32 | 9.38% | +2 wins |
| Grixis Affinity | 5 | 32 | 15.62% | +2 wins |
| Total | 25 | 128 | 19.53% | -8 wins |

Conclusion:
- Hard-fitting the strict critical-prefix dataset did not beat either the accepted checkpoint or the prior 2-epoch run.
- The result rules out simple undertraining of this dataset as the explanation.
- Stop scaling strict-prefix/head-only BC on the current labels. The next branch should return to the start-state/London layer: earlier replay probes showed Spy can often win from saved reachable hands, while full CP1 starts remain weak.

## Accepted Checkpoint CP1 London-Line Repair

Run: `20260507_spy_accepted_london_cp1_512`

Profile: `Pauper-Spy-Combo-AcceptedLondonCP1-20260507`

Reason:
- The accepted checkpoint still kept essentially every first Spy opening hand.
- Test whether terminal-only London-line outcomes can repair start-state decisions without touching the action policy.

Code/config note:
- `scripts/run_mulligan_probe.ps1` now exposes `-ModelDModel` and `-ModelNumLayers`, so probes can match profile architecture. The accepted Spy checkpoint is `d_model=128`, `num_layers=2`.

Accepted checkpoint first-hand probe:
- Probe: `20260507_spy_accepted_mulligan_probe512_d128`
- Result: `512/512` keeps.
- Even 0-effective-land diagnostic bucket kept `120/120`.

Collection:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponent skill: CP1
- Agent deck: Spy Combo
- Opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- Scenarios: 512
- Line specs per scenario: 15
- Actionable scenarios: 469
- Selected London/mulligan samples: 1,447
- Fastest winning-line action samples observed but not exported: 15,186
- Elapsed collection: 5,188.2 sec

Line outcome diagnostic:

| Forced line family | Wins | Samples | Winrate |
| --- | ---: | ---: | ---: |
| keep7 | 186 | 512 | 36.33% |
| best mull-to-6 family | 161 | 512 | 31.45% |
| best mull-to-5 family | 146 | 512 | 28.52% |

This means keep7 was globally better than the sampled mulligan branches under the accepted action policy.

Fit:
- Training data: `20260507_spy_accepted_london_cp1_collect512/selected_training_data.ser`
- Action types: `MULLIGAN,LONDON_MULLIGAN`
- Head-only direct BC, hard binary targets, 100 epochs, 4 candidate-order permutations.
- Train passes: 578,800
- Fit score before: exact top-1 `377/1447` (`26.05%`), target-set top-1 `1323/1447` (`91.43%`)
- Fit score after: exact top-1 `431/1447` (`29.79%`), target-set top-1 `1405/1447` (`97.10%`)

Post-fit first-hand probe: `20260507_spy_accepted_london_cp1_probe512`

| Effective-land bucket | Keep Rate | Mulligan Rate |
| --- | ---: | ---: |
| 0 | 27.50% | 72.50% |
| 1 | 35.71% | 64.29% |
| 2 | 28.57% | 71.43% |
| 3 | 23.68% | 76.32% |
| 4 | 0.00% | 100.00% |

CP1 eval: `20260507_spy_accepted_london_cp1_eval32`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 10 | 32 | 31.25% | -13 wins |
| Jund Wildfire | 4 | 32 | 12.50% | -2 wins |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 2 | 32 | 6.25% | -1 win |
| Total | 17 | 128 | 13.28% | -16 wins |

Conclusion:
- The mulligan head is trainable, but this target construction over-mulliganed and badly damaged CP1 performance.
- The line outcomes show the accepted all-keep policy is not obviously the current CP1 blocker for Spy: keep7 was the best sampled line family.
- Do not continue hard-binary London repair from this label set. The useful signal is the natural-start winning action data found during London-line search. Next branch: export fastest terminal-winning natural-start lines and train action decisions only, with mulligan/London training skipped.

## Natural-Start London Winning-Action BC

Run: `20260507_spy_natural_london_action_cp1_512`

Profile: `Pauper-Spy-Combo-NaturalLondonActionCP1-20260507`

Reason:
- Reuse the London-line search machinery, but train only post-mulligan action decisions from fastest terminal-winning natural-start branches.
- Avoid the prior over-mulliganing failure by skipping `MULLIGAN`, `LONDON_MULLIGAN`, `PASS`, and blank-action training.

Collection:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponent skill: CP1
- Agent deck: Spy Combo
- Opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- Scenarios: 512
- Workers: 32
- Actionable scenarios: 470
- Skipped scenarios: 42
- Candidate line samples: 1,634
- Selected London/mulligan samples written: 1,415
- Winning-line action samples observed: 15,695
- Exported action training file: `20260507_spy_natural_london_action_cp1_collect512_w32b/winning_action_training_data.ser` (`2.16 GB`)

Fit:
- Imported action examples: 12,000
- Train passes: 12,000
- Epochs: 1
- Candidate-order permutations: 1
- Distillation: head-only direct hardened BC at coefficient `0.5`.
- Action types: `ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X`

CP1 eval: `20260507_spy_natural_london_action_cp1_actionfit_e1_eval32`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 17 | 32 | 53.12% | -6 wins |
| Jund Wildfire | 7 | 32 | 21.88% | +1 win |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 4 | 32 | 12.50% | +1 win |
| Total | 29 | 128 | 22.66% | -4 wins |

Conclusion:
- Natural-start action BC did not beat the accepted checkpoint.
- It slightly improved Wildfire and Affinity, but not enough to offset mirror regression.
- The expensive collection succeeded and is reusable. Next branch: reuse the same `winning_action_training_data.ser`, but make the update more surgical by filtering to critical combo action text and lowering direct BC strength.

## Natural-Start London Critical-Action Low-BC

Run: `20260507_spy_natural_london_action_critical_lowbc_cp1_512`

Profile: `Pauper-Spy-Combo-NaturalLondonActionCriticalLowBC-20260507`

Reason:
- Reuse the completed natural-start winning-action dataset.
- Make the action update more surgical than the unfiltered branch by filtering action text to `Balustrade Spy|Dread Return|Lotleth Giant`.
- Lower direct BC coefficient from `0.5` to `0.1` to reduce disruption to the accepted policy.

Fit:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Imported action examples: 12,000
- Train passes: 12,000
- Epochs: 1
- Candidate-order permutations: 1
- Distillation: head-only direct hardened BC at coefficient `0.1`.
- Diagnostic: binary rows `183`; head counts `{action: 131, target: 45, card_select: 7}`; direct-BC gradient norm `0.02015810`.

CP1 eval: `20260507_spy_natural_london_action_critical_lowbc_e1_eval32`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 8 | 32 | 25.00% | +2 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 23 | 128 | 17.97% | -10 wins |

Conclusion:
- Filtering and lowering BC strength did not preserve the accepted checkpoint. It damaged mirror badly and lost Rally entirely.
- The result weakens the case for additional local BC variations on this dataset.
- Next diagnostic: score accepted vs trained profiles on the same imported winning-action data to determine whether the policy is failing to fit the teacher labels or fitting labels that do not transfer to real CP1 starts.

## Natural Winning-Action Label-Fit Diagnostic

Data: `20260507_spy_natural_london_action_cp1_collect512_w32b/winning_action_training_data.ser`

Score probe: first 500 imported action examples, mulligan/pass/blank skipped.

| Profile | CP1 Eval | Top-1 | Target Prob | Mean Rank |
| --- | --- | ---: | ---: | ---: |
| `Pauper-Spy-Combo-FastT5Contrast-20260506` | 33/128 accepted reference | 427/500 = 85.40% | 0.8120 | 1.168 |
| `Pauper-Spy-Combo-NaturalLondonActionCP1-20260507` | 29/128 | 442/500 = 88.40% | 0.8345 | 1.140 |
| `Pauper-Spy-Combo-NaturalLondonActionCriticalLowBC-20260507` | 23/128 | 442/500 = 88.40% | 0.8345 | 1.140 |

Conclusion:
- The accepted checkpoint already agrees with most natural winning-action labels.
- The BC variants only increase label fit by about 3 points, while reducing CP1 performance.
- The missing behavior is not in this successful-line label source. We need labels from failure states or learned credit over actual terminal outcomes.

## Accepted CP1 Logged Credit Diagnostic

Run: `20260507_spy_fastt5_accepted_cp1_logged128`

Purpose:
- Collect full decision logs from the accepted checkpoint and test whether terminal outcomes are predictable from decision/state features.
- This checks the synthetic-returns direction before wiring any learned credit into training.

Logged CP1 sample:
- Games: 128
- Wins: 24
- Losses: 104
- Exported decisions: 16,483
- Export path: `local-training/analysis/synthetic_return_20260507_logged128/accepted_logged128_trajectories.jsonl`

Patch:
- `scripts/export_game_log_trajectories.py` now reads `RESULT: WIN/LOSS` lines. Eval logs did not emit `Winner: PlayerRL1`, so the old exporter labeled every game as a loss.
- `scripts/train_synthetic_return_probe.py` now includes visible state-count and key-card presence features from `state_summary`.

Action-health summary:
- Land-play choices are not the current failure: `656/699` land-play opportunities selected a land, `0` pass-over-land cases.
- The checkpoint keeps every opener in this sample: `128/128` keeps, including `122/128` with 0-1 true lands. This is partly expected for Spy because Land Grant and cyclers are pseudo-resources, but it remains untrained behavior.
- Critical combo timing is the real failure:
  - Spy cast opportunities: 192 across 51 games.
  - Spy casts: 63, but `60/63` were with hidden true lands still estimated in library.
  - No-hidden-land Spy opportunities: 14, but Spy was cast only 3 times.
  - Dread Return flashbacks: 57, with `56/57` not combo-ready and `53/57` before Lotleth Giant was in graveyard.

Synthetic-return probe results on PlayerRL1 decisions:

| Feature Set | History Window | Test AUC | Test Loss | Base Loss |
| --- | ---: | ---: | ---: | ---: |
| action-only | 0 | 0.5642 | 0.6838 | 0.5191 |
| action-only | 4 | 0.5503 | 0.6920 | 0.5191 |
| action-only | 8 | 0.5364 | 0.6978 | 0.5191 |
| action-only | 16 | 0.5179 | 0.7015 | 0.5191 |
| state + action | 0 | 0.6258 | 0.8445 | 0.5191 |
| state + action | 4 | 0.6266 | 0.8523 | 0.5191 |
| state + action | 8 | 0.6262 | 0.8607 | 0.5191 |
| state + action | 16 | 0.6324 | 0.8606 | 0.5191 |

Lower-LR calibration check (`lr=0.01`, 150 epochs):
- Best held-out AUC stayed only `0.5761`.
- Test loss remained worse than the base-rate predictor.

Conclusion:
- Visible state features contain real terminal-outcome signal, but the current linear synthetic-return probe is not calibrated enough to train from.
- Compact history did not improve held-out performance on this sample.
- Next experiment should stay terminal-only but collect branch counterfactual labels at critical states reached from natural CP1 starts. The health logs show exactly why: successful-line cloning mostly agrees with the policy, while failures are premature Spy/Dread Return timing decisions.

## Natural-Start Critical Action-CF, Tactic Continuation

Run: `20260507_spy_natural_actioncf_critical_tactic_cp1`

Profile: `Pauper-Spy-Combo-NaturalActionCFCriticalTactic-20260507`

Reason:
- Collect terminal-only branch labels from natural CP1 starts, not saved reachable hands.
- Focus the branch points on critical combo contexts: `Balustrade Spy|Dread Return|Lotleth Giant`.
- Use tactic continuation after the forced action so the branch asks whether the forced decision can lead to a terminal Spy win, not whether the current flawed policy can finish unaided.

Collection:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponent skill: CP1
- Agent deck: Spy Combo
- Opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- Scenarios completed before stop gate: 1,183 / 2,048
- Trained scenarios: 276
- Skipped scenarios: 907
- Candidate examples: 300
- Selected examples: 300
- Elapsed collection: 2,391.3 sec

Fit:
- Imported action examples: 287
- Train passes: 2,296 across 4 epochs and 2 candidate-order permutations
- Loss mode: head-only direct BC, coefficient `0.5`, hard binary targets.
- Diagnostic: binary rows `12`; head counts `{action: 8, target: 4, card_select: 0}`; direct-BC gradient norm `3.41422200`.

CP1 eval: `20260507_spy_natural_actioncf_critical_tactic_fit_e4p2_eval32`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 12 | 32 | 37.50% | -11 wins |
| Jund Wildfire | 9 | 32 | 28.12% | +3 wins |
| Mono Red Rally | 0 | 32 | 0.00% | -1 win |
| Grixis Affinity | 3 | 32 | 9.38% | +0 wins |
| Total | 24 | 128 | 18.75% | -9 wins |

Label audit:
- Training rows: 300
- Action types: 244 `ACTIVATE_ABILITY_OR_SPELL`, 56 `SELECT_TARGETS`
- Baseline already matched best label in 94 rows; 206 rows changed the preferred action.
- Best-label counts for actual critical actions: 45 `Balustrade Spy`, 15 `Dread Return`, 10 `Lotleth Giant`.
- Top non-critical best labels included 31 `Lead the Stampede`, 22 `Winding Way`, 40 mana-tap labels, 8 `Forest: Play Forest`, and 13 `Ability: Pass`.

Conclusion:
- Natural failure-state action-CF did not beat the accepted checkpoint and regressed to `24/128`.
- The label audit shows why: the regex selected critical contexts, but the target still forced exact non-critical alternatives when a critical action lost. That can correctly say "do not cast Spy now," but it also teaches arbitrary midrange continuations and even pass labels.
- Next experiment: keep the terminal-only branch signal, but change the target construction to preserve the current policy distribution while suppressing terminal-losing critical candidates. This tests "avoid bad Spy/Dread timing" without forcing a specific Winding Way/Lead/Pass alternative.

## Natural Critical Action-CF, Negative-Only Preserve-Policy Target

Run: `20260507_spy_natural_actioncf_avoidloss_neg_w48`

Profile: `Pauper-Spy-Combo-NaturalActionCFAvoidLossNegW48-20260507`

Code change:
- `ActionCounterfactualTrainer` now supports `--avoid-losing-action-text-regex`.
- When a matching critical candidate reaches a terminal loss, the target preserves the current policy distribution over the other legal candidates and suppresses only the losing critical candidate.
- When a matching critical candidate reaches a terminal win, the target can still positively train that critical candidate.
- This remains terminal-only credit, but avoids forcing arbitrary exact alternatives such as `Lead the Stampede`, `Winding Way`, or `Ability: Pass`.

Collection:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Opponent skill: CP1
- Agent deck: Spy Combo
- Opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- Workers: 48
- Scenarios completed: 2,048 / 2,048
- Trained scenarios: 86
- Skipped scenarios: 1,962
- Candidate examples: 88
- Selected examples: 88
- Elapsed collection: about 55.3 minutes
- Observed issue: 48 workers improved scenario throughput versus 32, but Python inference latency rose from about 205 ms to 536 ms by the end of collection, so the collector is not cleanly CPU scalable.

Fit:
- Imported action examples: 87
- Train passes: 696 across 4 epochs and 2 candidate-order permutations
- Loss mode: head-only direct BC, coefficient `0.5`, soft targets, no binary hardening.
- Diagnostic: direct-BC gradient norm `1.69184184`.

Label audit:
- Training rows: 88
- Action types: 72 `ACTIVATE_ABILITY_OR_SPELL`, 16 `SELECT_TARGETS`
- Best-label counts: 47 `Balustrade Spy: Cast Balustrade Spy`, 5 `Balustrade Spy`, 5 `Dread Return: Flashback`, 3 `Lotleth Giant`, 1 `Ability: Pass`
- Target sparsity: 62 one-hot rows, with the remaining rows preserving a wider policy distribution after masking losing critical candidates.

CP1 eval: `20260507_spy_natural_actioncf_avoidloss_neg_w48_fit_e4p2_eval32`

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 16 | 32 | 50.00% | -7 wins |
| Jund Wildfire | 5 | 32 | 15.62% | -1 win |
| Mono Red Rally | 1 | 32 | 3.12% | +0 wins |
| Grixis Affinity | 9 | 32 | 28.12% | +6 wins |
| Total | 31 | 128 | 24.22% | -2 wins |

Conclusion:
- This is not a promotion candidate, but it is the best natural failure-state action-CF result so far.
- The target construction has the right sign for Affinity and does not collapse Rally, but it still damages mirror enough to miss the accepted `33/128`.
- Because collection was expensive and the data showed signal, next controls should reuse the same 87 examples with lower head-only direct-BC coefficients (`0.2`, then `0.1`) before collecting more data.

### Negative-Only Refit Controls

Same 87 terminal counterfactual examples were refit from the accepted checkpoint with lower head-only direct-BC coefficients.

| Profile | Coef | Eval | Spy Mirror | Wildfire | Rally | Affinity | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Pauper-Spy-Combo-NaturalActionCFAvoidLossNegHead02-20260507` | 0.2 | `20260507_spy_natural_actioncf_avoidloss_neg_head02_fit_e4p2_eval32` | 13/32 | 6/32 | 4/32 | 8/32 | 31/128 |
| `Pauper-Spy-Combo-NaturalActionCFAvoidLossNegHead01-20260507` | 0.1 | `20260507_spy_natural_actioncf_avoidloss_neg_head01_fit_e4p2_eval32` | 12/32 | 2/32 | 0/32 | 3/32 | 17/128 |

Conclusion:
- Lowering the coefficient did not promote the action-CF branch. Coef `0.2` kept the same total as `0.5` but merely redistributed wins; coef `0.1` collapsed.
- Stop one-off action-CF refits for now. The marginal information has dropped.

## Logged Value-Head Diagnostic on Accepted Checkpoint

Run: `local-training/analysis/value_head_20260507_accepted_logged128`

Input:
- Logged CP1 eval: `20260507_spy_fastt5_accepted_cp1_logged128`
- Exported trajectories: `local-training/analysis/synthetic_return_20260507_logged128/accepted_logged128_trajectories.jsonl`
- Actor filter: `PlayerRL1`
- Rows analyzed: 128 games, 16,355 logged value-head decision rows

Headline metrics:

| Subset | Rows | Base Winrate | AUC | Mean Win Score | Mean Loss Score | Score Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All decisions | 16,355 | 17.77% | 0.538 | 0.116 | 0.171 | -0.055 |
| Nontrivial options | 8,128 | 18.75% | 0.586 | 0.221 | 0.349 | -0.128 |
| Action-head-like | 15,512 | 17.42% | 0.530 | 0.112 | 0.167 | -0.055 |
| Critical action option (`Spy`/`Dread`) | 383 | 28.98% | 0.739 | 0.138 | 0.133 | +0.006 |
| Critical combo incl. target picks | 542 | 31.18% | 0.811 | 0.140 | 0.133 | +0.007 |
| Per-game mean value | 128 | 18.75% | 0.433 | 0.120 | 0.160 | -0.040 |

Interpretation:
- The accepted `FastT5Contrast` checkpoint has a useful policy seed, but its value head is not healthy overall. Decision-level AUC is weak, per-game mean value is inverted, and high raw value scores above `1.0` occur mostly in losing target-pick/pass-heavy games.
- Critical combo states have some ranking signal, but the score gap is tiny, so it is not a calibrated critic that can safely gate action selection or power train-time MCTS yet.
- This is expected in hindsight: the line-search/import path that created the accepted policy explicitly used `VALUE_LOSS_COEF=0`, `VALUE_LOSS_COEF_WARMUP=0`, and policy/BC distillation only. The policy improved while the critic remained stale.

Next experiment:
- Resume terminal-only RL from `Pauper-Spy-Combo-FastT5Contrast-20260506`, rather than more offline BC variants.
- Use a value-first warmup with `VALUE_LOSS_COEF_WARMUP=20`, `POLICY_LOSS_COEF_WARMUP=0`, `FREEZE_ENCODER_IN_WARMUP=1`, then normal low-LR policy/value PPO.
- Keep rewards terminal-only: `RL_HEURISTIC_STEP_REWARDS=0`.
- Start with Monte Carlo returns (`USE_GAE=0`, `GAE_AUTO_ENABLE=1`) so the stale critic does not bootstrap itself.
- Train against a hybrid mix of self-play and CP1 (`HYBRID_SELFPLAY_P=0.7`, `SKILL_MIX=1:1.0`) to balance throughput with an external race signal.

## FastT5 Value-RL Hybrid Resume, 10k

Profile: `Pauper-Spy-Combo-FastT5ValueRLHybrid-20260507`

Run roots:
- Training registry: `local-training/local_pbt/value_rl/20260507_spy_fastt5_value_rl_hybrid_registry.json`
- Eval registry: `local-training/local_pbt/value_rl/20260507_spy_fastt5_value_rl_hybrid_eval_registry.json`
- Final training run: `local-training/local_pbt/value_rl/20260507_spy_fastt5_value_rl_hybrid_r24_b4_resume`
- CP1 eval: `20260508_spy_fastt5_value_rl_hybrid_10k_cp1_eval32`

Training:
- Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Episodes: 10,000
- Reward: terminal only
- Opponent sampler: 70% self-play, 30% CP1
- Value warmup: `VALUE_LOSS_COEF_WARMUP=20`, `POLICY_LOSS_COEF_WARMUP=0`, `FREEZE_ENCODER_IN_WARMUP=1`
- Main losses: `POLICY_LOSS_COEF=0.5`, `VALUE_LOSS_COEF=5`
- Returns: MC first (`USE_GAE=0`, `GAE_AUTO_ENABLE=1`)
- Search: no MCTS

Throughput notes:
- `96` runners with batch `8` filled the train queue and reduced effective throughput; train service averaged about 17.7s.
- `96`, `64`, and `48` runners with batch `4` kept game generation high but starved the Python learner; train service stayed around 4.5s.
- `24` runners with batch `4` was the best local value-speed tradeoff for this run: about 2.5-3.0 eps/s late in the run, train service about 2.3-2.5s, and no VRAM requeues.
- Full CPU utilization was not the best metric. Once Java game runners starved PyTorch, lower visible CPU utilization produced higher trained-episode throughput.

CP1 eval:

| Matchup | Wins | Games | Winrate | Baseline Delta |
| --- | ---: | ---: | ---: | ---: |
| Spy mirror | 18 | 32 | 56.25% | -5 wins |
| Jund Wildfire | 11 | 32 | 34.38% | +5 wins |
| Mono Red Rally | 3 | 32 | 9.38% | +2 wins |
| Grixis Affinity | 5 | 32 | 15.62% | +2 wins |
| Total | 37 | 128 | 28.91% | +4 wins |

Interpretation:
- This is the first terminal-only RL continuation from the accepted Spy seed that beats the accepted total, but the margin is modest.
- The improvement came from non-mirror matchups, especially Wildfire, while mirror strength regressed.
- The training `value_accuracy.csv` remained weak at the end: reported accuracy was `0.0000`, with average win/loss values close together (`~0.021` vs `~0.018`). That metric may be poorly calibrated, but it does not show a clearly healthy critic.
- Next diagnostic should be a logged CP1 eval of this checkpoint and a repeat of the value-head/action-health analysis. Do not promote this checkpoint or start MCTS from it until we know whether the critic actually improved on logged decision states.

### Logged Diagnostic After Value-RL 10k

Logged eval: `20260508_spy_fastt5_value_rl_hybrid_10k_cp1_logged8`

Analysis root: `local-training/analysis/value_head_20260508_value_rl_logged8`

Small logged CP1 result:
- Total: 7/32
- Spy mirror: 4/8
- Jund Wildfire: 1/8
- Mono Red Rally: 1/8
- Grixis Affinity: 1/8

Value-head metrics:

| Subset | Rows | Base Winrate | AUC | Mean Win Score | Mean Loss Score | Score Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All decisions | 4,007 | 21.92% | 0.479 | 0.021 | 0.030 | -0.009 |
| Nontrivial options | 1,958 | 21.23% | 0.544 | 0.034 | 0.043 | -0.008 |
| Action-head-like | 3,814 | 21.56% | 0.469 | 0.019 | 0.029 | -0.010 |
| Critical action option (`Spy`/`Dread`) | 107 | 29.40% | 0.670 | 0.038 | 0.034 | +0.003 |
| Critical combo incl. target picks | 149 | 31.17% | 0.628 | 0.037 | 0.035 | +0.003 |
| `Balustrade Spy` cast option present | 55 | 36.36% | 0.794 | 0.035 | 0.029 | +0.006 |
| Per-game mean value | 32 | 21.88% | 0.411 | 0.024 | 0.030 | -0.005 |

Action-health metrics:
- Mulligan keep: 32/32; keeps with 0-1 real lands: 29/32.
- Land play health is now good: land selected 146/155 opportunities, with 0 pass-over-land events.
- Spy cast opportunities: 55; Spy cast selected 24/55.
- No-hidden-land Spy opportunities: 1/55; Spy cast in no-hidden-land spots: 0/1.
- Hidden-land Spy casts: 24/24 Spy casts were with hidden true lands.
- Dread Return flashbacks: 13; premature non-combo flashbacks: 13/13, with 12/13 before `Lotleth Giant` was in the graveyard.
- Target choice is not the main issue when the combo state exists: `Lotleth Giant` was selected 5/5 when available.

Conclusion:
- The 10k value-RL run slightly improved CP1 total, but it did not repair the actual Spy combo behavior. It still keeps extremely low-land hands, casts Spy while lands remain hidden, and flashes back Dread Return before the kill is assembled.
- Basic action ordering has improved enough that pass-over-land is no longer the blocker.
- The critic still does not separate wins and losses on logged games. The likely implementation cause is that the active value head used the shared policy CLS from the FastT5 policy seed, while the separate critic encoder existed in the model but was not used by `score_candidates`.

Next experiment:
- Enable the separate critic encoder behind `VALUE_USE_SEPARATE_CRITIC_ENCODER=1`.
- Resume from `Pauper-Spy-Combo-FastT5Contrast-20260506`, reset the critic head, and skip saved optimizer state with `LOAD_OPTIMIZER_STATE=0`.
- Keep terminal-only MC-return RL and the same 70% self-play / 30% CP1 training mix.
- Re-evaluate at 10k episodes with the same full CP1 sweep plus logged value/action diagnostics.
