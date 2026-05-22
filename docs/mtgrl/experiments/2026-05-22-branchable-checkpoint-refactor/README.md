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
- Added a fail-fast guard requiring `--force-opponent-transcript` / `-ForceOpponentTranscript` to run with `--opponent=cp7` / `-Opponent cp7`, because an RL opponent does not replay the transcript.
- Corrected the checkpoint probe to use replay prefix choices only for initial checkpoint capture, then branch source and alternate continuations directly from the in-memory checkpoint.
- Corrected checkpoint source-choice reentry matching to use the replay text matcher for selected text comparisons, so source-log action text like `Cast Balustrade Spy` can match checkpoint candidate text like `Balustrade Spy: Cast Balustrade Spy` when the candidate index and candidate set also match.
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

## Follow-Up Target Probes

### Chunk009 D033 Fallback

Target:

- Replay file: `local-training/local_pbt/corpora/20260522_v217_chunk009_d033_checkpoint_branch_probe/forced_prefix_replay_game_20260521_163356_0001_through_D033_target_D033_v217_bridge.csv`
- Source log: `local-training/local_pbt/cp7_eval_sweeps/20260521_v203_affinity_richer_metadata_g16/game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity__chunk_009/game_20260521_163356_0001.txt`
- Source anchor: `game_20260521_163356_0001_D033`
- Expected action: `ACTIVATE_ABILITY_OR_SPELL`
- Expected source choice: `Cast Balustrade Spy`

Runs:

| Run ID | Mode | Result |
| --- | --- | --- |
| `20260522_chunk009_d033_checkpoint_branch_probe_cp7_py312` | local policy with `-Opponent cp7 -ForceOpponentTranscript` | Captured the exact checkpoint and reproduced candidate hashes twice, but classified `checkpoint_reentry_mismatch` because selected text comparison required exact normalized equality between `Cast Balustrade Spy` and `Balustrade Spy: Cast Balustrade Spy`. |
| `20260522_chunk009_d033_checkpoint_branch_probe_cp7_py312_reentryfix` | same run after selected-text reentry matcher fix | Correction candidate. Captured exact D033 checkpoint with candidate hash `81fea6d964305aa1b7db596aa0a4043067710f23eaf7279fcd371b5d9e0afe14`, state hash `a41dd289e93ffd72a7005bd5370a507291a17b72e1cc537f14432045ef3e588e`, and RNG state hash `23842d84168976f`. Source choice reentered twice. Source continuation (`Cast Balustrade Spy`) lost terminal at turn 9; alternate `Ability: Pass` won terminal at turn 9. |

Conclusion:

- This is the first checkpoint-derived correction candidate in this lane.
- The successful result validates the branchable checkpoint path on an independent fallback target, not only on the chunk013/D030 and chunk005/D070 diagnostics.
- The source prefix still records an early opponent transcript diagnostic (`D009`, `transcript_empty`), but terminal branch continuations are launched from the in-memory checkpoint and do not replay that prefix. This field should remain diagnostic metadata, not an admission blocker after checkpoint reentry succeeds.

Same-log follow-ons:

| Target | Run ID | Result |
| --- | --- | --- |
| `chunk009_D041_ord017_SELECT_TARGETS` | `20260522_chunk009_d041_checkpoint_branch_probe_cp7_py312` | Bridge built with 14 rows and stable target object metadata, but checkpoint capture stopped at 13 decisions and never reached the D041 object-target surface. Classification: `source_prefix_divergence` / no checkpoint. |
| `chunk009_D046_ord018_SELECT_TARGETS` | `20260522_chunk009_d046_checkpoint_branch_probe_cp7_py312` | Bridge built with 15 rows and stable target object metadata, but checkpoint capture again stopped at 13 decisions and never reached the D046 object-target surface. Classification: `source_prefix_divergence` / no checkpoint. |

Interpretation:

- D041 and D046 are not evidence and should not be retried unchanged. The source branch that produced the D033 correction candidate does not expose those later target surfaces under the current checkpoint-capture replay path.
- The next candidate-mining unit should move away from later chunk009 rows unless the checkpoint capture path is extended to branch from D033 first and then search deeper surfaces inside that branch.

### D033

Target:

- Replay file: `local-training/local_pbt/corpora/20260522_v212_chunk013_d033_checkpoint_branch_probe/forced_prefix_replay_game_20260521_163738_0001_through_D033_target_D033_v212_bridge.csv`
- Source anchor: `game_20260521_163738_0001_D033`
- Expected action: `SELECT_TARGETS`
- Expected candidates: `Balustrade Spy||Balustrade Spy||Dread Return||Generous Ent||Overgrown Battlement`
- Expected source choice: `Dread Return`

Runs:

| Run ID | Mode | Result |
| --- | --- | --- |
| `20260522_d033_checkpoint_branch_probe_py312` | local policy, default RL opponent | `no_checkpoint`; the source path reproduced D030 at ordinal 16 but reached an `ACTIVATE_ABILITY_OR_SPELL` surface at ordinal 17 instead of D033. |
| `20260522_d033_checkpoint_branch_probe_forced_opp_py312` | forced opponent transcript requested, default RL opponent | Same `no_checkpoint`; this exposed that transcript forcing was ineffective unless the opponent mode is `cp7`. |
| `20260522_d033_checkpoint_branch_probe_forced_cp7_py312` | local policy with `-Opponent cp7 -ForceOpponentTranscript` | Captured exact D033 checkpoint, source choice reentered twice with matching candidate hash `47fe1b8cfc397be3ab224652d827b5206009cefac6e3ba5c39488b664544ac84`. Source continuation won terminal and first alternate `Balustrade Spy` also won terminal, so it is not correction evidence. |

Conclusion:

- D033 proves branchable in-memory checkpoints can work for a post-D030 target when the cp7 opponent transcript is actually replayed.
- The current classification is `source_terminal_not_loss`, not `clean_negative` and not `correction_candidate`.
- The run recorded an early opponent object-id mismatch while still reaching the D033 surface; this is a diagnostic warning for object-id strictness, not a branchability failure for this target.

### D070

Target:

- Replay file: `local-training/local_pbt/corpora/20260522_v211_chunk005_d070_checkpoint_branch_probe/forced_prefix_replay_game_20260521_162902_0001_through_D070_target_D070_v211_bridge.csv`
- Source anchor: `game_20260521_162902_0001_D070`
- Expected action: `SELECT_TARGETS`
- Expected source choice: `Lotleth Giant`

Runs:

| Run ID | Mode | Result |
| --- | --- | --- |
| `20260522_d070_checkpoint_branch_probe_py312` | local policy, default RL opponent | `no_checkpoint`; ordinal 63 reached an `ACTIVATE_ABILITY_OR_SPELL` surface instead of D070. |
| `20260522_d070_checkpoint_branch_probe_forced_cp7_py312` | local policy with `-Opponent cp7 -ForceOpponentTranscript` | Still `no_checkpoint`; the bridge reaches repeated `ACTIVATE_ABILITY_OR_SPELL`/Quirion Ranger surfaces around D066-D072 and misses the D070 target surface. |
| `20260522_d070_checkpoint_branch_probe_repeated_payloads_prefix_divergence_cp7_py312` | rebuilt bridge preserving repeated payloads plus prefix-divergence reporting | `source_prefix_divergence`; first failure was D011 `Cast Quirion Ranger`, because D002 Forestcycling had no attached search RNG provenance and the live hand did not contain Quirion Ranger. |
| `20260522_d070_checkpoint_branch_probe_search_metadata_cp7_py312` | D002/D022 Forestcycling search counts attached | `source_prefix_divergence`; D011 matched, then D012 exposed same-decision target preview rows before the live activation/pass choice. |
| `20260522_d070_checkpoint_branch_probe_action_only_prefix_cp7_py312` | same-decision `SELECT_TARGETS` previews skipped when an activation/action payload exists | Captured the exact D070 checkpoint: `SELECT_TARGETS`, candidates `Balustrade Spy||Lead the Stampede||Lotleth Giant||Sagu Wildling||Tinder Wall||Troll of Khazad-dum`, source choice `Lotleth Giant`, candidate hash `d639f8f37903853dbe24290cee0904f0fa39e442e6855a81c3b95716e41af158`. Reentry matched twice. Source continuation won; alternate `Balustrade Spy` lost. Classification: `source_terminal_not_loss`. |

Conclusion:

- D070 is no longer blocked by forced-prefix startup or missing in-memory checkpoint capture. The bridge now preserves repeated replay payloads, attaches cycling/search RNG provenance to the triggering activation row, and filters same-decision target-preview rows that are not live prefix decisions.
- The branchable checkpoint mechanism works on this D070 target: the source candidate set and source choice reentered deterministically from the in-memory snapshot without reconstructing the original log.
- This is not correction evidence. The source branch won and the first alternate lost, so no training should start from D070.

## Artifact Handling

Raw probe directories were generated under ignored `local-training/local_pbt/spy_line_replay/20260522_d030_checkpoint_branch_probe*`, `20260522_d033_checkpoint_branch_probe*`, and `20260522_d070_checkpoint_branch_probe*`, with bridge CSVs under ignored `local-training/local_pbt/corpora/20260522_v211_*` through `20260522_v216_*`. They were summarized here and are disposable local artifacts, not commit material. The successful local probes used a generated Python 3.12 venv outside the repo at `C:\Users\Jack\.codex\cache\mage-mtgrl-venv-py312` with dependency install disabled to avoid the Python 3.14 PyTorch wheel blocker.
