# Spy Terminal Prefix Curriculum

Date: 2026-04-29

## Objective

Test whether terminal-only winning-prefix search can teach Pauper Spy Combo real combo execution against CP1/CP7-style opponents, without heuristic step rewards or intermittent milestone rewards.

The target behavior is not just casting Balustrade Spy. The policy must learn the prerequisites and the full terminal line: setup mana and lands, resolve Spy, self-target when appropriate, Dread Return Lotleth Giant, and target the opponent with the Lotleth trigger.

## Infrastructure Changes

- `scripts/run_cp7_eval_sweep.py`
  - Starts the shared GPU service with `.mtgrl_venv` Python when available. This fixed eval sweeps launched from a non-venv Python that could not import `torch`.
- `ActionCounterfactualTrainer.java`
  - Added opponent life and stack state to action-search snapshots.
  - Added source-aware target priority: Spy trigger prefers self, Lotleth trigger prefers opponent.
  - Dread Return priority now requires a plausible lethal Lotleth state before preferring flashback.
  - Prefix replay now stores and matches choice text, not only raw candidate indices. This fixed drift where replayed target/card choices landed on the wrong option.
  - Winning-prefix training now uses the winning run's actual chosen indices rather than stale prefix indices.
- `scripts/run_mulligan_counterfactual.ps1`
  - Added model dimension parameters and safer argument quoting.
- `MulliganCounterfactualTrainer.java`
  - Accepts a single `.dek` path as deck input.
- `scripts/run_action_counterfactual.ps1`
  - Fixed Windows quoting for paths with spaces.
  - Added `-Seed`, `-AgentOpeningHand`, and `-OppOpeningHand` pass-throughs for reproducible targeted probes.

## Runs

Baseline after text-stable replay:

- `20260429_spy_active_terminal_rally_textstable_batch8`
  - Hand: `Swamp,Land Grant,Winding Way,Sagu Wildling,Balustrade Spy,Gatecreeper Vine,Overgrown Battlement`
  - Rally CP1, terminal mode `WIN`
  - `trainedScenarios=4/4`, `candidateExamples=83`, `tensorTop1=59/83`

Follow-up Rally rehearsal:

- `20260429_spy_active_terminal_rally_textstable_rehearsal_batch11`
  - Same reliable hand vs Rally
  - `trainedScenarios=5/8`, `candidateExamples=151`, `selectedExamples=111`, `tensorTop1=92/111`

Eval after batch11:

- `20260429_spy_terminal_textstable_rally_cp1_probe5`
  - Rally CP1: `2/8 = 25%`
  - First direct evidence that terminal winning-prefix labels transferred into real eval wins.
  - Wins were true combo wins: Spy self-target, Dread Return, Lotleth.

Broader curated positive-only batches:

- `20260429_spy_active_terminal_rally_mixedmana_batch13`
  - Hand: `Forest,Swamp,Land Grant,Balustrade Spy,Tinder Wall,Saruli Caretaker,Elves of Deep Shadow`
  - `trainedScenarios=1/8`, low value.
- `20260429_spy_active_terminal_elveswall_textstable_batch14`
  - Hand: `Forest,Forest,Swamp,Balustrade Spy,Tinder Wall,Elves of Deep Shadow,Wall of Roots`
  - Active pool CP1
  - `trainedScenarios=7/12`, `selectedExamples=117`, `tensorTop1=102/117`
- `20260429_spy_active_terminal_quirion_textstable_batch15`
  - Hand: `Forest,Forest,Swamp,Balustrade Spy,Tinder Wall,Saruli Caretaker,Quirion Ranger`
  - Active pool CP1
  - `trainedScenarios=6/12`, `selectedExamples=117`, `tensorTop1=97/117`

Eval after batches13-15:

- `20260429_spy_terminal_textstable_rally_cp1_probe6`
  - Rally CP1: `0/16`
  - This was a regression from `2/8`.
  - Current regressed checkpoint was backed up under `local-training/local_pbt/model_backups/20260429_spy_after_batches13_15_regressed_rally0of16`.
  - Active model was restored to the `probe5` snapshot.

Generic branch-counterfactual probe:

- `20260429_spy_terminal_branchcf_rally_probe1`
  - Restored from `probe5`, then ran early-decision terminal branch labels vs Rally.
  - `LossTurnBonus=0.20` produced only 3 examples, mostly weak keep/survival labels.
  - This was rejected as a corrective source and backed up under `local-training/local_pbt/model_backups/20260429_spy_after_branchcf_probe1_sparse_bad`.
  - Active model was restored again to the `probe5` snapshot.

## Findings

Terminal winning-prefix search is now technically viable. It can find and replay deep Spy wins, and it can transfer into eval wins.

Positive-only curated prefix cloning is not enough. It teaches high-confidence local actions from successful contexts, but it does not teach when those actions are invalid in live distributions. The clearest regression was batches14-15: they had strong tensor replay metrics, but Rally eval fell to `0/16`.

Mulligan is still broken. The model keeps almost every 7-card hand. In probe6 it kept all 16 hands, including 14 hands with 0-1 lands. This is not fixed by more winning-prefix examples because those examples mostly reinforce keep on curated playable hands.

Branch counterfactuals are the right generic direction, but the current random probe is too sparse. If no branch wins, using loss-survival as a target creates noisy or harmful labels. We should require true terminal wins for strong action labels, or make the branch search deeper before training.

## Next Experiment

Build a denser terminal-only counterfactual source instead of adding more positive-only curated hands.

Recommended next branch:

1. Add targeted branch-counterfactual support around winning-prefix search nodes:
   - For each prefix node, evaluate sibling candidate branches to terminal.
   - Train a distribution over candidates from terminal outcomes at that same state.
   - This keeps terminal-only reward while giving negative contrast for context-blind actions.
2. Keep positive winning-prefix examples only when paired with sibling outcome contrast or when eval shows no regression.
3. Run Rally-first eval gates:
   - 8-game smoke after each corrective batch.
   - 16-game Rally gate before full CP1 sweep.
4. Do not scale batch14/batch15-style positive-only data until mulligan and context contrast are fixed.

## Throughput Follow-Up: 3050 Split

Date: 2026-04-30

Hardware changed during this experiment: an RTX 3050 6GB was added alongside the RTX 4070 Super. The useful split is:

- 4070: PyTorch learner (`TRAIN_CUDA_DEVICE=cuda:0`)
- 3050: Java ONNX inference (`INFER_CUDA_DEVICE=cuda:1`, mapped into `ONNX_CUDA_DEVICE_ID=1`)

Measured local throughput:

- CP7 skill-7 ladder, old hybrid-ish sample: about `0.40 eps/s` total across four profiles.
- CP7 skill-7 ladder, ONNX on 3050 + train on 4070: about `0.37 eps/s` total. This did not help because CP7/game simulation remained the bottleneck.
- Pure self-play, 64 runners, default learner batches: `3.20 eps/s` total. GPU1 averaged `42%`, GPU0 averaged `6%`, but learner backpressure appeared.
- Pure self-play, 64 runners, `TRAIN_GPU_MAX_CONCURRENT=2`, batch 8: `4.50 eps/s` total. GPU1 averaged `53%`, GPU0 averaged `21%`, CPU averaged `46%`.
- Pure self-play, 96 runners, `TRAIN_GPU_MAX_CONCURRENT=3`, batch 8: `4.78 eps/s` total. Slightly faster, but learner service time and pending queue grew sharply.
- Pure self-play, 96 runners, `TRAIN_GPU_MAX_CONCURRENT=2`, batch 8: invalid. Windows exposed only one CUDA GPU by then, so ONNX fell back to CPU before the new guard was added.
- Pure self-play, single visible 3050, 32 runners, train+ONNX on `cuda:0`, batch 4: `1.68 eps/s` total. This is the best measured fallback while the 4070 is unavailable, but it is only about 37% of the valid two-GPU split.
- Pure self-play, single visible 3050, 48 runners, train+ONNX on `cuda:0`, batch 4: `0.26 eps/s` total. The 6GB card saturated VRAM and produced a 190s learner service time, so increasing runners on the single GPU is counterproductive.
- Pure self-play, single visible 3050, 32 runners, batch 2/1024 steps: `1.64 eps/s` total. Smaller learner batches reduced per-update size but still hit VRAM guard/backpressure and did not beat batch 4.

Conclusion: with two GPUs, pure self-play is now roughly 8-12x faster than CP7 skill-7 training on this machine. The previous conclusion that pure self-play was not worth it should be revisited. The old bottleneck was CP7 CPU cost; with the 3050 split, a pure self-play or mostly-self-play curriculum becomes more attractive as a throughput-first phase, followed by CP7 eval gates.

Operational note: Windows Update installed `NVIDIA - Display - 32.0.15.6094` at 2026-04-30 00:29 and left the 4070 in `CM_PROB_FAILED_ADD`. From the non-elevated shell, `pnputil /restart-device` was denied. `scripts/run_local_pbt.py` now validates requested `cuda:N` devices before startup so `INFER_CUDA_DEVICE=cuda:1` fails fast if the 4070/3050 visibility changes and only one CUDA device remains.

Benchmarking note: `scripts/run_local_pbt_benchmark.ps1` runs bounded local PBT probes with CPU/GPU sampling and restores active model weights afterward. A first version had a wildcard restore bug; active model weights were restored manually from `pre_pure_selfplay_split_benchmark_20260430_001130`, ONNX snapshots were regenerated from those restored weights, and hashes were verified against the pre-throughput baseline.

## Two-GPU Autonomous Cycle

Date: 2026-04-30

After reinstall/reboot, CUDA visibility recovered:

- `cuda:0`: RTX 4070 Super
- `cuda:1`: RTX 3050

Started `20260430_twogpu_selfplay_eval_cycle_24h` via `scripts/run_mtgrl_autonomous_cycle.ps1`.

Cycle design:

- 24 hours max.
- 6-hour two-GPU pure self-play phases.
- 4070 trains PyTorch (`TRAIN_CUDA_DEVICE=cuda:0`).
- 3050 serves Java ONNX inference (`INFER_CUDA_DEVICE=cuda:1`).
- 64 game runners.
- Learner: concurrency 2, batch 8, max 4096 steps.
- After each train phase, run reduced CP1 and CP3 gates for Spy and Wildfire only.

Warmup health:

- Initial steady profile rates summed to roughly `4.5-5.0 eps/s`.
- 4070 sampled around `47%` GPU utilization / `5.1GB` used.
- 3050 sampled around `56%` GPU utilization / `5.0GB` used.
- Learner queue was elevated but below the configured cap (`PENDING_TRAIN_MAX=128`), with no immediate CPU-only fallback.

Backup before cycle: `local-training/local_pbt/model_backups/pre_20260430_twogpu_selfplay_eval_cycle_24h`.

Phase 1 outcome:

- Train phase ran from `2026-04-30T12:29:21Z` to `2026-04-30T18:30:29Z`.
- It stopped at the intended 6-hour wall clock limit.
- The wrapper initially treated the normal wall-clock exit as failure because Windows PowerShell left `Start-Process.ExitCode` unset after `Wait-Process`.
- `scripts/run_mtgrl_autonomous_cycle.ps1` now normalizes an unset exit code to 0 after process refresh so the cycle proceeds to eval.
- Phase 1 reached approximately:
  - Spy: `108545` episodes
  - Wildfire: `85055` episodes
  - Rally: `57292` episodes
  - Affinity: `50470` episodes
- Late phase throughput remained roughly `4.7-5.4 eps/s` total across profiles, with higher rates on Rally/Affinity than Spy/Wildfire.

Recovery eval:

- Recovery run: `20260430_twogpu_selfplay_eval_cycle_24h_recover`
- CP1 reduced gate after phase 1:
  - Spy: `4/16 = 25%`
  - Wildfire: `4/16 = 25%`
- This is weak but not a total collapse. It justifies continuing evidence gathering before discarding the self-play phase.
- CP3 reduced gate started immediately after CP1.

Credit-assignment tooling added during recovery:

- `scripts/export_game_log_trajectories.py`
  - Converts existing text game logs into trajectory JSONL with terminal winner, mulligans, selected actions, candidate probabilities, value scores, visible state summaries, and event flags.
  - Smoke export: `local-training/analysis/spy_wildfire_recent_trajectories.jsonl`
  - Result: 40 logged Spy/Wildfire self-play games, 18,843 decision records.
- `scripts/analyze_trajectory_event_credit.py`
  - Computes simple event/outcome lift as a sanity check before a neural synthetic-return probe.
  - Initial 40-game sample baseline: `12/40 = 30%` PlayerRL1 winrate.
  - Dread Return availability/casting had positive lift.
  - Selected Balustrade Spy had negative lift, which suggests Spy is often being cast in non-converting lines. The next credit-assignment experiment should reward complete combo-line states rather than merely "cast Spy happened."

Methodology note:

- `docs/mtgrl/2026-04-30-credit-history-methodology-notes.md` records the current plan for synthetic returns, compact previous-state context, and whether to pursue a new MTGRL-specific methodology.

Resumed phase health sample:

- Resume run: `20260430_twogpu_selfplay_eval_cycle_24h_recover_resume`
- Phase 1 resumed at `2026-04-30T19:19:06Z`.
- Latest observed throughput during early resume: roughly `4.5-5.0 eps/s` total.
- GPU split remains healthy:
  - 4070 learner: roughly `4.7-6.4GB` used depending on train/export timing.
  - 3050 ONNX inference: roughly `5.1-5.3GB` used.

Action-health sample from the current logs:

- Spy:
  - `pass_over_land`: `200/774 = 25.8%`
  - `turn1_pass_over_land`: `11/30 = 36.7%`
  - `keep_0_1_land`: `30/40 = 75%` of keeps
  - `spy_cast_opportunities`: 89
  - `spy_casts`: 10
  - `dread_return_flashback_selected`: 11
  - `premature_dread_flashback_not_combo_ready`: `10/11 = 90.9%`
  - `dread_return_target_lotleth`: 0
- Wildfire:
  - `pass_over_land`: `20/1408 = 1.4%`
  - `turn1_pass_over_land`: 0
  - `cleansing_target_opponent`: `28/53 = 52.8%`
  - `cleansing_target_self_other`: `25/53 = 47.2%`

Interpretation: Spy is still bottlenecked on line conversion and mulligan quality. Wildfire is much less broken at basic development, but still has target-selection leakage/noise around Cleansing Wildfire. This supports the synthetic-credit/line-quality direction if the resumed phase does not materially improve CP gates.
