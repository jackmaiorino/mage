# Branchable Checkpoint Refactor

Date: 2026-05-22

## Purpose

Validate whether in-memory engine checkpoints can replace forced-prefix log reconstruction for replay-branch probes. This is a validation spike only: no durable checkpoint corpus, no training data generation, no HPC promotion, and no training launch.

## Implementation

- Added `RandomUtil.captureState()` / `RandomUtil.restoreState()` using an owned copyable RNG stream, preserving wrapper/direct counters and replay seed metadata.
- Added `ActionCounterfactualTrainer --checkpoint-branch-probe=true`.
- Captured `EngineDecisionCheckpoint` values from the AIRL replay decision path using `Game.createSimulationForAI()`.
- Reentered checkpoint clones twice to verify deterministic candidate/action/state hashes, then attempted source and alternate continuations when the source checkpoint matched the target row.
- Added `captured_checkpoints.csv` as a small diagnostic manifest for checkpoint probe runs.
- Added `scripts/run_spy_line_replay_probe.ps1 -CheckpointBranchProbe`.
- Corrected the checkpoint probe to use replay prefix choices only for initial checkpoint capture, then branch source and alternate continuations directly from the in-memory checkpoint.
- Aligned the local Py4J bridge with `RLLogPaths.MODEL_FILE_PATH` so profile-scoped source-policy probes load the same model path Java uses for training/eval.
- Fixed isolated workspace copy rules outside the repo so real source packages under `Mage/src/main/java/mage/target` and `Mage/src/main/java/mage/cards/repository` are not accidentally excluded.

## Validation

Passed:

- `python -m py_compile "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\cli_orchestrator.py" "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\cli_worker.py"`
- `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile`
- `mvn -o -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install`
- `mvn -o -q -pl Mage.Tests -am "-Dtest=org.mage.test.utils.RandomTest#test_CaptureAndRestoreState" "-Dsurefire.failIfNoSpecifiedTests=false" test`

## D030 Probe Summary

Target:

- Replay file: `D:\codex-mage-cli-workspaces\20260522_v209_chunk013_d030_source_profile_gate_isolated\local-training\local_pbt\corpora\20260522_v207_chunk013_d030_source_profile_gate_isolated\forced_prefix_replay_game_20260521_163738_0001_through_D030_target_D030_v207_bridge.csv`
- Source anchor: `game_20260521_163738_0001_D030`
- Expected action: `SELECT_TARGETS`
- Expected candidates: `Balustrade Spy||Balustrade Spy||Dread Return||Generous Ent||Lotleth Giant||Overgrown Battlement`
- Expected source choice: `Lotleth Giant`

Runs:

| Run ID | Mode | Result |
| --- | --- | --- |
| `20260522_d030_checkpoint_branch_probe_noop_debug` | `PY_SERVICE_MODE=none` | Exposed stale local Maven artifact: runtime `RandomUtil` lacked `captureState()`. |
| `20260522_d030_checkpoint_branch_probe_noop_after_install` | `none` after reactor install | Captured 84 checkpoints and reentered one deterministically, but ordinal lookup selected the wrong checkpoint surface. |
| `20260522_d030_checkpoint_branch_probe_noop_manifest` | `none` with target-aware selector | Captured 111 checkpoint surfaces, including 11 `SELECT_TARGETS`, but not the exact D030 candidate set. |
| `20260522_d030_checkpoint_branch_probe_local_profile_after_install` | `local` | Source-policy path blocked by repeated `Failed to install PyTorch, exit code: 1`; source run timed out before checkpoints. |
| `20260522_d030_checkpoint_branch_probe_onnx` | `onnx` | Captured 72 checkpoint surfaces and 13 `SELECT_TARGETS`, but still did not reach the exact D030 candidate set. |
| `20260522_d030_checkpoint_branch_probe_local_py312` | `local` with Python 3.12 venv | Removed the PyTorch install blocker, but autonomous source replay still did not reach D030 because first-search RandomUtil count started at `0` instead of source `172`. |
| `20260522_d030_checkpoint_branch_probe_prefix_capture_py312` | `local` with Python 3.12 venv and forced-prefix checkpoint capture | Captured exact D030 checkpoint, reentered source choice twice, source continuation lost terminal, alternate `Balustrade Spy` continuation lost terminal; classified clean negative. |

Conclusion:

- In-memory checkpoint capture and deterministic reentry are implemented and mechanically validated.
- The D030 branchability gate is now met for this target: the captured checkpoint presented the exact source candidate set and source choice (`Lotleth Giant`), both reentry probes matched the candidate hash, and both source and alternate continuations reached terminal losses.
- This is a clean negative, not correction evidence. No training should start from this D030 result.
- The important implementation distinction is that prefix replay is used once to seed the checkpoint; source and alternate terminal continuations do not reconstruct the prefix.

## Artifact Handling

Raw probe directories were generated under ignored `local-training/local_pbt/spy_line_replay/20260522_d030_checkpoint_branch_probe*`. They were summarized here and are disposable local artifacts, not commit material. The successful local probe used a generated Python 3.12 venv outside the repo at `C:\Users\Jack\.codex\cache\mage-mtgrl-venv-py312` with dependency install disabled to avoid the Python 3.14 PyTorch wheel blocker.
