# Execution Log

## 2026-04-25 Phase A/B Setup

Implemented:

- Added `phase-b-generalist-registry.json` for `Pauper-Generalist-Value-v2`.
- Added `scripts/run_arch_validation_generalist.ps1`.
- Added `scripts/run_arch_validation_cp7_reduced.ps1`.
- Added `--split-agent-decks` to `scripts/run_cp7_eval_sweep.py` so a generalist checkpoint can be evaluated as separate Spy/Wildfire/Rally/Affinity rows.
- Extended `scripts/export_training_profile_metrics.py` with head-usage and health metrics.
- Fixed `RLTrainer` head-usage logging to write under the active profile path and include RL/opponent deck labels.
- Updated `scripts/start_profile_metrics_exporter.ps1` with `-Restart` and autodiscovery-friendly default profile handling.

Validation:

- `python -m py_compile scripts/run_local_pbt.py scripts/export_training_profile_metrics.py scripts/run_cp7_eval_sweep.py`
- PowerShell parser checks for the new launch/eval scripts.
- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`

## 2026-04-25 Phase B Launch Notes

Training profile:

- `Pauper-Generalist-Value-v2`

Training pool:

- Spy Combo
- Jund Wildfire
- Mono Red Rally
- Grixis Affinity

Initial launch exposed two infrastructure issues:

1. The shared GPU learner service was not inheriting common registry `train_env` model-shape settings. This allowed the learner to load/checkpoint a 128-wide model while Java/ONNX expected the 256-wide registry model.
2. The shared GPU host reads `GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS`, while the local orchestrator only set `GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS`. That caused very small learner batches.

Fixes:

- `scripts/run_local_pbt.py` now applies common `train_env` values to the GPU service environment before Python loads checkpoints.
- `scripts/run_local_pbt.py` now forwards `GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS`, falling back to `GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS`.
- Invalid generated `Pauper-Generalist-Value-v2` artifacts from the first 128-wide startup were removed and recreated cleanly.

Throughput tuning tried:

| Setting | Result |
| --- | --- |
| 4 episodes / 2048 steps | Best tested tradeoff |
| 8 episodes / 4096 steps | Worse; service time rose sharply |
| `TRAIN_CHUNK_SIZE=256` | Worse/no gain; reverted |

Backpressure fix:

- The learner queue fills because training long trajectories is the bottleneck.
- Java's default 30s shared-GPU control timeout caused train enqueue retries during queue backpressure.
- The Phase B launcher now sets `GPU_SERVICE_CONTROL_TIMEOUT_MS=300000` and `PENDING_TRAIN_OFFER_TIMEOUT_MS=300000` so bounded no-drop backpressure does not turn into timeout/requeue storms.

Current state after restart:

- Grafana stack is running at `http://localhost:3000`.
- Profile exporter is running on `http://localhost:9102/metrics`.
- GPU service metrics are on `http://localhost:27100/metrics`.
- Training is stable but learner-bound.
- No GPU OOMs, no VRAM guard requeues, no model NaNs observed in the post-fix run.

Early signal around 3k to 3.6k recorded episodes:

- Value accuracy: about `0.30`.
- Games killed: `0`.
- GPU OOMs: `0`.
- Model NaNs: `0`.

Interpretation:

The Phase B generalist run is viable, but useful throughput is constrained by learner updates over long trajectories. Completed-game eps/s is no longer the only metric that matters; trained steps/sec and train queue age need to be tracked as first-class convergence metrics.

## 2026-04-25 Learner-Bottleneck Adjustment

Observation:

- With full two-seat self-play trajectories, the shared GPU train queue stayed near `64/64`.
- Average learner batch size was about `1,477` decision steps for about `4` queued player-trajectories.
- Train service time was typically `5-7s`, and producer backpressure made completed-game throughput fall to roughly `0.5-4 eps/s`.

Change tested:

- Added `TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER`.
- The Phase B launcher briefly tested `TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER=256`.
- This keeps terminal-only learning: no intermittent reward is added.
- The cap keeps the ordered suffix of each player's trajectory, so the terminal reward remains on the final transition and return propagation still covers the last `N` decisions.
- Both sides of self-play are still trained.

Reason:

This tests whether a terminal-reward suffix is a better value-speed tradeoff than training every early-game priority/action decision. It should reduce learner service time while retaining the endgame/combo decisions closest to the observed win or loss.

Result:

- Not kept for the Phase B gate.
- The GPU service continued to fill train batches to roughly `1.5k-2k` steps by grouping multiple Java train batches.
- Train service time stayed around `5-7s`.
- Value accuracy deteriorated during the capped segment.
- The launcher is back to `TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER=0` so the gate uses full trajectories.

## 2026-04-25 Inference/Training GPU Contention Adjustment

Observation:

- Java ONNX inference was using the CUDA provider while the shared GPU learner was also training on CUDA.
- During full-trajectory training, Java ONNX inference latency climbed from roughly `78ms` to roughly `159ms`.
- CPU utilization remained low enough that CPU inference was a plausible tradeoff.

Change tested:

- The Phase B launcher briefly tested `ONNX_FORCE_CPU=1`.
- The learner remains on CUDA through the shared GPU service.
- This keeps training/eval semantics unchanged and only moves Java policy inference off the learner GPU.

Result:

- Rejected immediately.
- CPU ONNX inference was much slower on this machine: early action-head batches took about `0.6-3.0s`, versus tens of milliseconds with the CUDA provider.
- The launcher is back to `ONNX_FORCE_CPU=0`.

## 2026-04-26 Phase B Gate Result

Training completed:

- Profile: `Pauper-Generalist-Value-v2`
- Target: `TOTAL_EPISODES=10000`
- Trainer exit: `rc=0`
- Orchestrator stopped cleanly after all selected profiles reached target.
- Final value accuracy row: `0.3700` at episode `10000`.

Final self-play rolling-200 winrates by RL deck:

| RL deck | Rolling winrate |
| --- | ---: |
| Spy Combo | `0.415` |
| Jund Wildfire | `0.355` |
| Mono Red Rally | `0.700` |
| Grixis Affinity | `0.600` |

CP7 Skill 7 reduced sweep:

- Command: `.\scripts\run_arch_validation_cp7_reduced.ps1 -GamesPerMatchup 5 -Skill 7 -Parallel 4`
- Run directory: `local-training/local_pbt/cp7_eval_sweeps/20260425T235446Z`
- Split agent decks: enabled.
- Result: `0/79`, `0.0%`.
- All 16 matchups had zero wins.
- One matchup completed `4/5` games; the rest completed `5/5`.
- No eval activation/model-loading errors were found with quick log scans.

Interpretation:

- The pure-selfplay generalist failed the first architecture gate.
- Self-play produced nontrivial internal deck winrates, but the policy did not transfer to CP7 at all.
- This is strong evidence that pure terminal self-play needs an external anchor or stronger action-head diagnostics before longer runs.

Next experiment:

- Do not continue this run to the 6-hour gate.
- Start an anchored-curriculum diagnostic from this checkpoint:
  - `Pauper-Generalist-Anchor-v1`
  - four-deck random pool
  - terminal reward only
  - hybrid opponents: about `70%` self-play, `30%` CP7 Skill 1/3 mix
  - train both RL-controlled sides
  - gate after 2 hours or 5k additional episodes
  - run CP7 Skill 3 split sweep first, then Skill 7 only if Skill 3 is nonzero
- In parallel, capture failed CP7 game logs and build fixed-position action-head tests from the first obvious decision failures.
