# 2026-05-09 Spy Search Policy Improvement

Goal: test whether search can be used as a policy-improvement operator for the accepted Spy profile without heuristic rewards.

## Tiny Flat-MCTS Training Smoke

Profile:

- `Pauper-Spy-Combo-MCTSTrainSmoke-20260509`
- Seeded from `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Run: `20260509_spy_mcts_train_smoke_r8_e200`
- 200 episodes, 8 runners, terminal rewards only
- Selective flat MCTS training targets:
  - `MCTS_TRAINING_ENABLE=1`
  - `MCTS_ITERATIONS=2`
  - `MCTS_DETERMINIZATIONS=1`
  - `MCTS_ROLLOUT_DEPTH=0`
  - `MCTS_TIMEOUT_MS=500`
  - `MCTS_SKIP_TOP_PROB=0.99`
  - combo-spell selective keywords

Training finished cleanly in about 10 minutes. Rolling training winrate ended around 0.39.

Reduced CP1 eval:

- Run: `20260509_spy_mcts_train_smoke_cp1_eval16`
- Result: `0/64`
  - Spy mirror: `0/16`
  - Wildfire: `0/16`
  - Rally: `0/16`
  - Affinity: `0/16`

Sanity checks:

- Accepted checkpoint through the same eval runner: `20260509_accepted_cp1_eval4_sanity`, `3/16`.
- Accepted checkpoint plus matched tiny flat-MCTS eval override:
  - First run had selective enabled with no keywords, so MCTS activations were zero.
  - Corrected run: `20260509_accepted_flatmcts_tiny_kw_cp1_eval4`, `1/16` with 200 MCTS activations.

Conclusion: the current 1-ply value-net MCTS target is a bad teacher at this budget. The AlphaZero-style train loop plumbing works, but scaling this target is not justified.

## Online Prefix Autopilot

Implementation:

- `TerminalPrefixSearch.Result` now carries the executed winning trace, not just the BFS root prefix.
- `ComputerPlayerRL` gained opt-in eval-only suffix forcing behind `RL_ONLINE_PREFIX_AUTOPILOT_ENABLE=1`.
- The autopilot queue is not overwritten by a new prefix search while a discovered trace is still active.
- Training remains unaffected by default.

Runs:

- `20260509_online_prefix_autopilot_n3_logged2`: original same-root result carried `prefixLen=1`, so no suffix was forced. Result: `0/8`.
- `20260509_online_prefix_autopilot_trace_n3_logged2`: after executed-trace capture, suffix forcing activated. Result: `2/8`.
- `20260509_online_prefix_autopilot_trace_n3_cp1_eval4`: `3/16`.
- `20260509_online_prefix_autopilot_trace_n3_nooverwrite_cp1_eval4`: no-overwrite queue fix, `3/16`.
- `20260509_online_prefix_autopilot_trace_n7_probe_g4`: nodes7/depth6 upper-budget probe, `2/4`.
- `20260509_online_prefix_autopilot_trace_n7_cp1_eval4`: nodes7/depth6, `4/16`.

The n7 run produced 32 online prefix searches, 17 found traces, 79 forced suffix actions, and 17 misses. Matchups:

- Spy mirror: `2/4`
- Wildfire: `0/4`
- Rally: `0/4`
- Affinity: `2/4`

Conclusion: executed-trace forcing proves the compounding-drift hypothesis mechanically, but it does not clear even a small promotion gate. Runtime is still several seconds per search, and live traces often diverge from branch simulations. Do not scale online prefix autopilot as an eval overlay.

## Current Read

Search remains useful for discovering lines offline, but the two direct policy-improvement routes tested here are not enough:

- Flat value-net MCTS is actively harmful as a teacher.
- Terminal prefix autopilot can force parts of found lines, but the live winrate is only parity/slightly below the accepted checkpoint at small sample sizes.

## Train-Time Prefix Trace Smoke

Implementation attempt:

- Added opt-in training flags:
  - `RL_ONLINE_PREFIX_SEARCH_DURING_TRAINING`
  - `RL_ONLINE_PREFIX_AUTOPILOT_DURING_TRAINING`
  - `RL_ONLINE_PREFIX_AUTOPILOT_TARGETS`
- When enabled, forced prefix root/suffix choices write one-hot MCTS-style targets into `TrainingData.mctsVisitTargets`.
- Added smoke harness:
  - `Pauper-Spy-Combo-OnlinePrefixTraceTrain-20260509`
  - `20260509_spy_online_prefix_trace_train_r4_e80`

Result:

- Aborted before completion.
- Trainer entered games but completed 0 episodes after warmup.
- `trainer.log` filled with live `RL priority() caught exception; forcing pass: deadline` warnings.

Conclusion: the eval-only prefix search hook is safe enough for diagnostics, but train-time integration is not usable at the current call site. Search deadlines can leak into live priority handling under training load, causing forced passes and stalled games. Do not relaunch this training harness until prefix search is moved out of live `priority()` or converted into an offline/queued data-generation step.

Next work should not scale these exact operators. The remaining high-EV surface is a training-data or representation change that handles multi-step drift without relying on the current value-net MCTS target or live train-time prefix search.

## Root-Only Train-Time Prefix Search Smoke

Reason:

- The aborted train-time prefix trace run combined root search, suffix autopilot, and target logging.
- Test a narrower mechanism: terminal-prefix search may override only the current root action and write a one-hot MCTS-style target; suffix autopilot stays disabled.

Profile:

- `Pauper-Spy-Combo-OnlinePrefixRootTrain-20260509`
- Seeded from `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Registry: `local-training/local_pbt/value_rl/20260509_spy_online_prefix_root_targets_registry.json`

Training setup:

- 80 episodes, 4 runners
- Terminal rewards only, no heuristic step rewards
- Hybrid CP1/self-play, 25% self-play
- `RL_ONLINE_PREFIX_SEARCH_ENABLE=1`
- `RL_ONLINE_PREFIX_SEARCH_DURING_TRAINING=1`
- `RL_ONLINE_PREFIX_AUTOPILOT_ENABLE=0`
- `RL_ONLINE_PREFIX_AUTOPILOT_TARGETS=1`
- Root search: 3 nodes, depth 6, top-3, 2 activations per game, 750 ms total timeout, 200 ms branch timeout

Training result:

- Completed 80/80 episodes cleanly with trainer `rc=0`.
- Health counters stayed clean: 0 game kills, 0 activation failures, 0 GPU OOMs, 0 Python errors, 0 model NaNs.
- Throughput was slow, roughly 80 episodes in 35 minutes.
- Rolling training winrate ended at 0.32.
- Value accuracy stayed weak: final row 0.1786.

Reduced CP1 eval:

- Initial bad registry covered only the Spy mirror: `20260509_online_prefix_root_train_e80_cp1_eval16`, 9/16 mirror.
- Corrected full registry: `20260509_online_prefix_root_train_e80_cp1_eval16_full`
- Total: 12/64
- Spy mirror: 8/16
- Jund Wildfire: 1/16
- Mono Red Rally: 1/16
- Grixis Affinity: 2/16

Conclusion:

Root-only train-time prefix search is a plumbing success but a policy-learning failure at this setting. It avoids the deadline/stall failure from suffix autopilot training, but the learned policy is far below the accepted reduced CP1 baseline. Do not scale this exact root-search configuration.
