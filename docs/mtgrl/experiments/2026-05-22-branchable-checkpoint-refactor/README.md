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

## v221 Candidate Sweep

Summary artifact:

- `local-training/local_pbt/corpora/20260522_v221_checkpoint_candidate_sweep/v221_checkpoint_candidate_sweep_summary.csv`

Scope:

- Native controller sweep over the remaining strict non-chunk001 v204 Affinity targets after D030, D070, chunk013 D033, chunk009 D033, chunk009 D041, and chunk009 D046 had already been classified.
- All 18 bridge/probe rows completed with `bridge_status=ok` and `probe_status=ok`.

Classification counts:

| Classification | Count | Interpretation |
| --- | ---: | --- |
| `source_prefix_divergence` | 12 | The bridge was strict enough to build, but the forced capture path did not reach the target checkpoint surface. The dominant visible blocker was D059 stack context drift; secondary blockers were D046 object id mismatch and early action type mismatch. |
| `clean_negative` | 3 | Checkpoint captured, reentry matched, source continuation lost, and the first alternate also lost. These are branchability evidence, not correction evidence. |
| `checkpoint_reentry_mismatch` | 2 | Checkpoint captured and source choice matched the expected source row, but the clone reentry resumed into a later `ACTIVATE_ABILITY_OR_SPELL` surface instead of the captured pending `SELECT_TARGETS`/`DECLARE_BLOCKS` prompt. These rows need richer pending-choice snapshot/reentry support before they can be admitted or rejected. |
| `source_terminal_not_loss` | 1 | Checkpoint captured and reentered, but the source branch was not a terminal loss. |
| `correction_candidate` | 0 | No new checkpoint-derived correction evidence. |

Notable rows:

| Candidate | Classification | Notes |
| --- | --- | --- |
| `chunk005_D110_ord091_ACTIVATE_ABILITY_OR_SPELL` | `clean_negative` | Source and alternate both lost terminal; candidate hash `22056cdfaee3efaa3a40b4c7233bb52bd8c20f85bcee5f3b457dac24a411ab2a`. |
| `chunk016_D060_ord021_ACTIVATE_ABILITY_OR_SPELL` | `clean_negative` | Source and alternate both lost terminal; candidate hash `12b90b2311146c80c10e89dc3072c230813512a2bc799f98f24eaea130a9619e`. |
| `chunk005_D125_ord096_ACTIVATE_ABILITY_OR_SPELL` | `clean_negative` | Source and alternate both lost terminal; candidate hash `efaa0a6841817cd834d462873d781b20d9966059e02e97ee980723850a720fb9`. |
| `chunk014_D022_ord011_ACTIVATE_ABILITY_OR_SPELL` | `source_terminal_not_loss` | Source and alternate both won terminal, so it is not correction evidence. |
| `chunk015_D111_ord092_DECLARE_BLOCKS` | `checkpoint_reentry_mismatch` | Captured `DECLARE_BLOCKS` candidates `Saruli Caretaker||Balustrade Spy||DONE`, but clone reentry resumed into a `Generous Ent`/`Saruli Caretaker` activation surface. |
| `chunk015_D099_ord080_SELECT_TARGETS` | `checkpoint_reentry_mismatch` | Captured player target candidates `Player:ACF-CP7||Player:ACF-Prefix`, but clone reentry resumed into the same later activation surface. |

Conclusion:

- v221 found zero new correction candidates. The only admitted checkpoint-derived correction candidate remains `chunk009_D033_ord015_ACTIVATE_ABILITY_OR_SPELL`, where source `Cast Balustrade Spy` lost terminal and alternate `Ability: Pass` won terminal.
- Training, HPC, and promotion remain blocked under the accepted CP7 Grixis Affinity gate: one correction candidate is far below the durable prior that calls for a much larger accepted-policy failure set with terminal-winning corrected siblings.
- The next unit should broaden accepted-policy failure collection and candidate mining rather than retry v221 rows unchanged. Separately, the two `checkpoint_reentry_mismatch` rows identify a real checkpoint-completeness gap for pending non-priority choices; that is a checkpoint-engine follow-up, not training evidence.

## Accepted-Policy Metadata Refresh

Purpose:

- After v221 found no new correction candidates, the next research unit pivoted back to accepted-policy CP7 Grixis Affinity failure collection so the checkpoint branch probe has a larger, current replay-metadata surface to mine.

Environment repair:

- `20260522_v227_affinity_richer_metadata_g32` is invalid infrastructure output. It launched under a Python 3.14 `.mtgrl_venv` with only `pip`/`py4j`, so every chunk failed with `No module named 'torch'` and the summary stayed `0/0`.
- The local generated `.mtgrl_venv` was rebuilt with Python 3.12 and the ML runtime dependencies (`torch`, `numpy`, `py4j`, `transformers`).
- `20260522_v229_affinity_richer_metadata_g2_envfix_smoke` produced real logs but is also invalid for accepted-policy mining because the shared service still defaulted to `INFER_CUDA_DEVICE=cuda:0` while the repaired torch install is CPU-only; the JVM fell back on failed shared-GPU requests.
- `20260522_v230_affinity_richer_metadata_g1_cpu_smoke` is the clean environment gate: `1/1`, replay metadata present, shared service `infer=cpu train=cpu`, and no `Shared GPU batch request failed` or CUDA assertion.

Durable run:

- `20260522_v231_affinity_richer_metadata_g32_cpu` was launched as the accepted-policy metadata refresh with `INFER_CUDA_DEVICE=cpu`, `TRAIN_CUDA_DEVICE=cpu`, `MULLIGAN_DEVICE=cpu`, `--games-per-matchup 32`, `--games-per-job 1`, `--parallel 1`, `--eval-game-logging`, `--game-log-format compact`, and `--replay-metadata`.
- First verified chunk: chunk 1 completed `0/1` in 39.1 seconds, wrote replay-compatible compact metadata (`action_counterfactual_compatible=true` plus `REPLAY_RANDOM`), and had no shared-GPU fallback or CUDA errors.

Next action when v231 completes:

- Mine the v231 compact logs into a fresh target manifest, preferring terminal losses with strict pressure, bounded alternatives, stable source/target metadata, and non-duplicated surfaces.
- Run checkpoint-branch probes on the best new target(s). Do not start training or HPC until checkpoint-derived correction evidence expands beyond the single chunk009 D033 candidate.

## v232 v231 Metadata Mining

The accepted-policy refresh `20260522_v231_affinity_richer_metadata_g32_cpu` completed `13/32` against CP7 Grixis Affinity. All 32 compact game logs included replay-compatible metadata (`action_counterfactual_compatible=true` plus `REPLAY_RANDOM`), and the run produced 19 terminal losses for checkpoint target mining.

Local v232 mining consumed the v231 logs after the detached CLI worker hit the known pre-lease Responses transport blocker. The loss corpus contains 19 losing games and 1,988 loss-decision rows. The raw target manifest contains 369 candidates: 332 fresh v231 failure candidates, 37 older current-family candidates, and 97 pressure candidates. Its top raw ranks still prefer old v77/current-family rows that are already excluded by the v115/v221 ladder, and the manifest builder only marks the older current-family rows `replay_ready`; direct log inspection verified the fresh v231 rows are replay-compatible.

Filtered next target:

- Source log: `local-training/local_pbt/cp7_eval_sweeps/20260522_v231_affinity_richer_metadata_g32_cpu/game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity__chunk_001/game_20260522_140523_0001.txt`
- Candidate: `game_20260522_140523_0001_D086`
- Replay metadata: scenario `1`, seed `763880686`, random-util seed `7640891576595197415`
- Decision: ordinal `39`, action type `ACTIVATE_ABILITY_OR_SPELL`
- Source choice: `Cast Dread Return`
- First alternate to probe: `Cast Masked Vandal`
- Candidate set: `Pass||Cast Dread Return||Cast Lead the Stampede||Cast Masked Vandal||Cast Land Grant||{T}: Add {G}.||{T}: Add {G} for each creature you control with defender.||{T}: Add {B}.||{T}: Add {B}. {this} deals 1 damage to you.`
- Pressure context: opponent permanents `10`, own life `14`, own graveyard count `5`

Next exact unit:

- Build a forced-prefix bridge for `game_20260522_140523_0001_D086` and run a checkpoint-branch probe.
- Preserve the action row over same-decision target-preview payloads if the bridge sees the D070-style repeated-payload shape.
- Admit evidence only if checkpoint capture and reentry match, the source continuation is terminal loss, and a sibling continuation reaches terminal win.
- Do not start training or HPC from this manifest alone.

## v232 Blocker Repairs

Two blockers from the v232 handoff were repaired before the D086 checkpoint probe:

- The failure-corpus collector now exports compact replay metadata (`action_counterfactual_compatible`, scenario, seed, `random_util_seed`, and replay-random scope), and the target-selection manifest builder can also parse those fields directly from compact log headers. Regenerating the v231 corpus keeps `1,988` loss-decision rows and now ranks `game_20260522_140523_0001_D086` first with `replay_ready=true`, scenario `1`, seed `763880686`, and random-util seed `7640891576595197415`.
- The detached CLI worker harness now launches Codex children with a sanitized environment (`CODEX_HOME`, `HOME`, `USERPROFILE`, `APPDATA`, and `LOCALAPPDATA` pinned to Jack's profile and app-thread inheritance removed), classifies Responses transport failures in `status.json`, and exposes a no-lease `transport-smoke` command through both `cli_worker.py` and `cli_orchestrator.py`. The post-repair smoke returned `cli transport smoke ok`.

Next exact unit remains unchanged: run the D086 forced-prefix bridge and checkpoint-branch probe directly through the repaired local path, and only admit evidence under the terminal-loss/source and terminal-win/sibling gate.

## v242-v256 D086 Replay Parity Repairs

D086 was used as the first fresh v231 loss target after the v232 mining repair. The probe did not produce admissible checkpoint evidence, but it isolated and repaired several replay-prefix blockers:

- The bridge builder can now retain targeted singleton combat pass rows in ACF ordinal mode via `--include-singleton-combat-pass-source`. For D086 this preserves source D062 as a `DONE` declare-blockers row without retaining every singleton combat pass.
- The bridge builder now ignores shuffle-only search metadata payloads marked `without_search_context`, avoiding false source provenance on secondary shuffle rows.
- Opponent transcript replay now has a late strict-priority fallback: after agent ordinal 26 and source turn 4, missing or non-legal opponent priority transcript actions pass instead of allowing autonomous CP7 actions to create a new branch. Combat selection remains on the previous fallback path because strict combat no-op was too broad.
- Forced-prefix spell payment now reserves mana sources that the source log shows as available in later same-turn prefix rows. This is gated to late prefix ordinals to avoid perturbing early mana lines.

Representative runs:

| Run ID | Result |
| --- | --- |
| `20260522_v242_d086_checkpoint_branch_probe_cp7_py312_combatpass_d062_only` | D062 retention got past the earlier combat pass surface, then failed at D068 because the opponent replay branch cast Refurbished Familiar before the source line. |
| `20260522_v243_d086_checkpoint_branch_probe_cp7_py312_strict_transcript` | Strict combat handling was too broad and failed early at D043. Combat strictness was reverted. |
| `20260522_v246_d086_checkpoint_branch_probe_cp7_py312_priority_strict_ord26` | Gated strict priority got past D068 and exposed D073 mana-payment divergence: the replay tapped Overgrown Battlement to cast Sagu Wildling, leaving Quirion Ranger uncastable. |
| `20260522_v249_d086_checkpoint_branch_probe_cp7_py312_payment_reservation_ord26` | Late payment reservation got past D073 and captured the later ordinal-42 surface, but the D086 target still did not match. The next blocker was source-prefix stack context around D077/Refurbished Familiar. |
| `20260522_v256_d086_checkpoint_branch_probe_cp7_py312_final_scoped` | Final scoped code compiled and retained the D073 candidate surface, but D086 remains `source_prefix_divergence`; no checkpoint evidence is admitted. |

Conclusion:

- These repairs improve the replay bridge but do not validate D086 as evidence. `checkpoint_captured=false` remains the correct classification for the D086 target.
- Speculative combat and player-target object-id relaxations were tried and backed out because they regressed early prefix parity.
- The next exact unit is not training. It is source/opponent transcript completeness around the remaining prefix mismatch: either make combat/player-target transcript validation advance without corrupting early mana parity, or pick a fresh v231 target whose checkpoint surface is not behind this transcript stack-context ladder.

## Live Checkpoint Capture Pivot

The D086 repair ladder shows the long-term problem clearly: forced-prefix replay can be useful as a diagnostic, but it is too fragile as the primary branch source for deep accepted-policy losses. The opponent transcript, mana payment choices, stack context, and search RNG all have enough valid degrees of freedom that a replay from the game start may legally miss the historical surface even when the target log is real.

Implementation:

- Added opt-in live checkpoint recording in `LiveCheckpointRecorder`, called from `ComputerPlayerRL.logReplayDecision` before each `REPLAY_DECISION_JSON` line.
- Each captured checkpoint stores a `game.createSimulationForAI()` snapshot, the acting player id/name, ordinal, decision number, action type, candidate texts/object ids, selected indices/texts/object ids, value/probability metadata, compact state hash, and `RandomUtil.State`.
- `RandomUtil.State` and the owned copyable RNG state are now serializable so checkpoint artifacts can be written as `*.ser.gz`.
- `ComputerPlayerRL.model` and `ReplayOpponentDecisionPlayer.gameLogger` are transient so the simulation snapshot does not try to serialize the live Python model bridge or logger.
- `scripts/run_cp7_eval_sweep.py` can enable this path with `--live-checkpoints`, writing manifests under `<run>/live_checkpoints/<matchup>/manifest.csv`.

Validation:

| Run ID | Result |
| --- | --- |
| `20260522_v258_live_checkpoint_smoke_g1` | One-game smoke reached terminal, but checkpoint serialization failed on `LazyPythonModel`. |
| `20260522_v259_live_checkpoint_smoke_g1_model_transient` | Model serialization was fixed; checkpoint serialization then failed on `GameLogger`. |
| `20260522_v260_live_checkpoint_smoke_g1_transient_loggers` | Logger serialization was fixed; stale installed `Mage` artifact still loaded the old non-serializable `RandomUtil.State`. |
| `20260522_v261_live_checkpoint_smoke_g1_after_install` | Passed after offline reactor install. The game completed `0/1`, wrote 91 `*.ser.gz` checkpoints, and the manifest had 91 `captured` rows with no errors. |

Conclusion:

- The long-term path is now live checkpoint collection during accepted-policy evaluation. That bypasses the old-log startup drift because the branchable snapshot is captured at the original policy decision, not reconstructed later from a prefix.
- Existing forced-prefix probes remain useful for diagnosis and targeted historical validation, but should not block corpus growth.
- No training or HPC promotion is admitted from live checkpoints alone. The next unit is a larger accepted-policy live-checkpoint collection run, followed by a loader/miner that branches from the serialized snapshots and only emits correction evidence when source continuation loses terminal and a sibling continuation wins terminal.

## v263 Live Snapshot Branch Miner

Purpose:

- Implement the long-term path for serialized live checkpoints: load a `LiveCheckpointRecorder.Snapshot`, attach branch control to the normal copied `ComputerPlayerRL`, and resume directly from the captured engine state without reconstructing any replay prefix.

Implementation:

- Added `EngineDecisionBranchController` as a reusable transient branch-control hook on `ComputerPlayerRL`.
- Added `LiveCheckpointBranchMiner`, a CLI that recursively loads `*.ser.gz` live snapshots, restores `RandomUtil.State`, verifies source reentry twice, then attempts source and alternate terminal continuations.
- Pinned `ComputerPlayerRL.serialVersionUID` to the v262 snapshot stream value so checkpoint artifacts remain readable after adding transient branch-control fields.
- Made `LiveCheckpointRecorder.compactState` and `sha256` reusable so reentry probes compare the same state and candidate hashes that live capture wrote.
- Fixed durable snapshot compatibility in the engine: `Exile` now uses a stable permanent exile-zone id, migrates legacy serialized permanent zones on read/copy, and `ZonesHandler` defensively recreates the permanent exile zone before placing cards there. This removed the terminal-branch crash where old snapshots deserialized with an unreachable permanent exile zone.
- Updated the branch runner to keep resuming until terminal or timeout, including the Maven `main` thread path that XMage treats as a game thread.
- Alternate selection now prefers a non-pass sibling when available instead of blindly choosing the first non-source candidate.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed after each source patch. |
| `mvn -o -q -pl Mage -am -DskipTests install` | Required so `exec:java` used the fixed engine classes instead of stale installed `mage` artifacts. |
| `mvn -o -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install` | Passed after the live miner and branch runner changes. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v263_smoke_reentry/live_checkpoint_branch_probe.csv` | Single v262 checkpoint reentered twice with matching candidate hash `97b74ddf7697b1fea210f56f71da4d467989dd339b1ef5814453ca7ed5fc038a` and state hash `24acacb2a740cf70d278a938906360cc63519ddfbb98d30fe77d7885c560c02e`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v263_smoke_reentry_3/live_checkpoint_branch_probe.csv` | Three v262 checkpoints classified `reentry_matched`; both clones reproduced source candidate/state hashes. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v263_smoke_terminal_exilefix_installed/live_checkpoint_branch_probe.csv` | Source branch from `ord001_D001_ACTIVATE_ABILITY_OR_SPELL` reached terminal loss from the serialized live checkpoint. Alternate `Pass` branch did not terminate in the bounded probe. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v263_smoke_terminal_ord004/live_checkpoint_branch_probe.csv` | Deeper source branch reentered cleanly but timed out under the unavailable local Python policy gateway; classified `source_error`, not evidence. |

Conclusion:

- The long-term live snapshot branch path is now real: serialized v262 snapshots can be deserialized after code changes, cloned, reentered twice with identical hashes, and resumed into a source terminal-loss branch without prefix replay.
- The current remaining blocker is not forced-prefix reconstruction. It is terminal mining throughput/completeness after the branch: local policy inference is falling back because the Python gateway is unavailable, and pass-like alternates often do not produce a terminal result inside the bounded probe.
- No new training evidence is admitted from v263. The next research unit should run the miner against a larger v262/v264 live-checkpoint slice with a healthy policy backend or an explicit post-branch autopilot mode, then admit only rows where source loses terminal and a non-pass sibling wins terminal.

## v264 Terminal Mining Repair

Problem:

- The v263 terminal miner rejected alternate branches during reentry because the same source-choice match check was used for source probes and sibling probes. A valid sibling choice therefore looked like `source_choice_mismatch` and could stop before the alternate branch actually played.

Implementation:

- Split branch reentry validation into source-choice probes and alternate-choice probes. Source reentry and source terminal continuation still require the captured source indices/texts to match. Alternate continuations require the same action type and candidate set, plus a valid forced sibling index, but intentionally do not require the source-selected choice.
- Added `--max-alternates` and `--alternate-timeout-sec` so a snapshot can try multiple non-source siblings and report each attempt.
- Added CSV fields `alternate_attempt_count`, `alternate_terminal_count`, `alternate_win_count`, and `alternate_outcomes` so terminal mining quality is visible instead of hidden behind one selected alternate.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed. |
| `mvn -o -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v264_alt_semantics_ord001/live_checkpoint_branch_probe.csv` | `ord001_D001` classified `clean_negative`: source `Cast Land Grant` lost terminal, alternate `Pass` lost terminal, `alternate_attempt_count=1`, `alternate_terminal_count=1`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v264_alt_semantics_ord003/live_checkpoint_branch_probe.csv` | Five-candidate `ord003_D003` classified `clean_negative`: source lost terminal, two non-pass alternates lost terminal, and the third alternate timeout was explicitly recorded in `alternate_outcomes`. |

Conclusion:

- The immediate terminal-mining quality blocker is fixed: alternate branches now execute as alternates, multi-alternate attempts are visible, and bounded terminal losses are recorded from serialized live checkpoints without prefix replay.
- These two rows are clean negatives, not correction evidence. The next evidence-mining step is to run the live snapshot miner across a larger accepted-policy live-checkpoint slice and look for rows where the source branch loses terminal and at least one sibling branch wins terminal.
- The remaining quality caveat is branch throughput under unavailable Python gateway/autopilot paths. Timeout/error outcomes are now visible per alternate, so they can be filtered or used to prioritize a backend/autopilot repair without corrupting admitted evidence.

## v265-v266 Live Checkpoint Mining

v265 sorted-prefix slice:

- Run artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v265_live_checkpoint_mining_slice40`.
- Scope: first 40 sorted `ACTIVATE_ABILITY_OR_SPELL` snapshots from the v262 live-checkpoint corpus, source timeout 90 seconds, alternate timeout 45 seconds, up to 3 alternates.
- Result: 40 rows completed with 22 `clean_negative`, 9 `alternate_error`, 7 `source_terminal_not_loss`, and 2 `source_error`.
- Correction candidates: 0. Alternate terminal wins: 0.

Interpretation:

- The live snapshot branch miner works mechanically at slice scale, but sorted path-order mining over-samples adjacent early decisions. The first 40 rows were mostly clean negatives plus timeout/errors, not correction evidence.
- The next harness improvement is candidate selection before terminal probing, not a larger blind sorted run.

v266 ranked selection:

- Added `LiveCheckpointBranchMiner --selection-mode ranked`, `--ranked-max-per-game`, and `selected_snapshots.csv`.
- The selector scores snapshots by candidate breadth, non-pass alternates, spell-like choices, later turn/decision depth, low policy confidence, negative value, low-life/graveyard/opponent-board pressure, and penalties for pass/mana/land-play-only surfaces.
- Validation artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v266_ranked_selection_reentry_smoke`.
- Reentry smoke command selected 12 ranked snapshots with `--ranked-max-per-game 6` and `--reentry-only true`.
- Result: 12/12 `reentry_matched`.
- The top selected rows are later, high-branching spell decisions across multiple chunks, for example `Cast Generous Ent`, `Cast Sagu Wildling`, `Cast Balustrade Spy`, and `Cast Masked Vandal`, rather than the first chronological decisions.

Next action:

- Run a ranked terminal slice from the same v262 corpus. Admit evidence only if a ranked source-loss row has an alternate terminal win. If the ranked slice still yields zero correction candidates, the next bottleneck becomes corpus/target scarcity or branch backend/autopilot quality, not checkpoint reentry.

v267 ranked terminal slice:

- Run artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v267_ranked_terminal_slice40`.
- Scope: ranked top 40 `ACTIVATE_ABILITY_OR_SPELL` snapshots from the v262 live-checkpoint corpus, `--ranked-max-per-game 6`, source timeout 90 seconds, alternate timeout 45 seconds, up to 3 alternates.
- Initial result: 40 rows completed with 2 `clean_positive`, 35 `clean_negative`, 2 `alternate_error`, and 1 `source_terminal_not_loss`.
- Admitted correction candidates: 0 after reproducibility checks.

Initial apparent correction rows:

| Rank | Snapshot | Source loss | Winning alternate | Other alternate outcomes |
| ---: | --- | --- | --- | --- |
| 1 | `chunk_002` `ord038_D092_ACTIVATE_ABILITY_OR_SPELL` | `Cast Generous Ent` | `Cast Overgrown Battlement` | `Cast Lotleth Giant` loss; `Cast Lead the Stampede` loss |
| 38 | `chunk_003` `ord037_D048_ACTIVATE_ABILITY_OR_SPELL` | `Cast Lotleth Giant` | `{T}: Add {G}.` | first attempted alternate won |

Interpretation:

- Ranked selection is still useful, because it moved mining away from adjacent early decisions and surfaced richer branch points than the sorted-prefix v265 slice.
- The two apparent v267 positives are not admitted evidence. Fresh-process reprobes did not reproduce the terminal-winning sibling outcomes.
- At this point, training still remained blocked until checkpoint-derived positives passed deterministic continuation and isolated confirmation gates.

v268-v280 reproducibility closeout:

| Run | Snapshot | Result |
| --- | --- | --- |
| `v268_positive_reprobe_chunk002_ord038` | `chunk_002` `ord038_D092_ACTIVATE_ABILITY_OR_SPELL` | Source reentry and source terminal loss reproduced, but alternates timed out under the 45 second sibling bound. |
| `v269_positive_reprobe_chunk002_ord038_longtimeout` | same | With 120 second sibling timeout, all three alternates terminalized as losses. The v267 `Cast Overgrown Battlement` terminal win did not reproduce. |
| `v270_positive_reprobe_chunk003_ord037_longtimeout` | `chunk_003` `ord037_D048_ACTIVATE_ABILITY_OR_SPELL` | With 120 second sibling timeout, all three alternates terminalized as losses. The v267 `{T}: Add {G}.` terminal win did not reproduce. |
| `v276_autopilot_no_belief_model_reprobe_chunk002_ord038` | `chunk_002` `ord038_D092_ACTIVATE_ABILITY_OR_SPELL` | Deterministic post-branch continuation classified `source_terminal_not_loss`; no sibling branch was admitted. No Py4J gateway warnings occurred. |
| `v277_autopilot_no_belief_model_reprobe_chunk003_ord037` | `chunk_003` `ord037_D048_ACTIVATE_ABILITY_OR_SPELL` | Deterministic post-branch continuation classified `source_terminal_not_loss`; no sibling branch was admitted. No Py4J gateway warnings occurred. |
| `v278_ranked_autopilot_smoke5` | ranked top 5 | Found one same-process `clean_positive` with repeat confirmation, but it was not admitted because direct fresh-process reprobe was still required. |
| `v279_autopilot_positive_reprobe_chunk007_ord016` | `chunk_007` `ord016_D052_ACTIVATE_ABILITY_OR_SPELL` | Fresh-process direct reprobe of the v278 apparent positive classified `source_terminal_not_loss`; the same-process positive was invalidated. |
| `v280_ranked_autopilot_stable_smoke5` | ranked top 5 after stable-order autopilot | Stable candidate-text/object-id ordering removed the batch-only positive: 3 `source_terminal_not_loss`, 2 `clean_negative`, 0 positives. No Py4J gateway warnings occurred. |
| `v281_ranked_autopilot_stable_slice40_retry` | ranked top 40 after stable-order autopilot | Completed 40 rows: 19 `source_terminal_not_loss`, 19 `clean_negative`, 2 `clean_positive_needs_isolated_reprobe`. No batch positive was admitted directly. |
| `v282_isolated_reprobe_chunk004_ord027` | `chunk_004` `ord027_D054_ACTIVATE_ABILITY_OR_SPELL` | Fresh-JVM direct reprobe classified `clean_positive` with `confirm_positive_repeats=2`. Source `Cast Saruli Caretaker` lost terminal; sibling `Cast Lead the Stampede` won terminal. |
| `v283_isolated_reprobe_chunk004_ord053` | `chunk_004` `ord053_D089_ACTIVATE_ABILITY_OR_SPELL` | Fresh-JVM direct reprobe classified `clean_positive` with `confirm_positive_repeats=2`. Source `Cast Balustrade Spy` lost terminal; sibling `Cast Lead the Stampede` won terminal. |
| `v284_ranked_autopilot_stable_slice100` | ranked wider slice after stable-order autopilot | Completed 78 selected rows under `--ranked-max-per-game 10`: 44 `source_terminal_not_loss`, 26 `clean_negative`, 4 `clean_positive_needs_isolated_reprobe`, and 4 `clean_positive_unstable`. |
| `v285_isolated_reprobe_chunk005_ord016` | `chunk_005` `ord016_D020_ACTIVATE_ABILITY_OR_SPELL` | Fresh-JVM direct reprobe classified `clean_positive_unstable`: the first alternate attempt won, but both repeat confirmations made the sibling lose. Not admitted. |
| `v286_isolated_reprobe_chunk005_ord036` | `chunk_005` `ord036_D042_ACTIVATE_ABILITY_OR_SPELL` | Fresh-JVM direct reprobe classified `clean_positive` with `confirm_positive_repeats=2`. Source `Cast Tinder Wall` lost terminal; sibling `Return a Forest you control to its owner's hand: Untap target creature. Activate only once each turn.` won terminal. |

Harness repair:

- `LiveCheckpointBranchMiner` now defaults to deterministic post-branch autopilot. The captured source/sibling decision is still forced from the checkpoint, but later AIRL choices for all branch-controlled RL players use the same deterministic branch policy instead of the normal model path.
- `ComputerPlayerRL` now lets active branch controllers force target, card, mode, X, optional-use, attack, attack-target, and block choices before model scoring. It also bypasses candidate inference and belief/card-belief prediction when a branch controller is active, preventing unavailable local Python gateway state from contaminating branch outcomes.
- Apparent terminal-winning sibling rows now require `--confirm-positive-repeats` repeat confirmation before the miner can label them `clean_positive`; otherwise they are classified as `clean_positive_unstable`.
- Batch runs from `--checkpoint-root` also require an isolated direct reprobe before admission. A batch-discovered positive is downgraded to `clean_positive_needs_isolated_reprobe` unless it is rerun via `--snapshot` in a fresh JVM and passes the same source-loss/sibling-win confirmation gate.
- The deterministic autopilot now chooses later candidates by stable candidate text plus object-id ordering rather than raw engine list position, which avoids admitting branch results that only appear after prior same-JVM snapshots perturb later candidate order.

Current admitted evidence:

| Artifact | Source | Confirmed sibling | Candidate hash | State hash | RNG hash |
| --- | --- | --- | --- | --- | --- |
| `v282_isolated_reprobe_chunk004_ord027` | `Cast Saruli Caretaker` terminal loss | `Cast Lead the Stampede` terminal win | `4afec00e0b8824ce9106834d4468e76ca332ed5790df1887711554d82b5e6846` | `a2446accbda286e63d8b85b88e3df6536fc61b26b17aac1e9d6cbd66a1629b67` | `3991df502d5ac7cb` |
| `v283_isolated_reprobe_chunk004_ord053` | `Cast Balustrade Spy` terminal loss | `Cast Lead the Stampede` terminal win | `deda1a2784be4abcac67a9b3c2e39c47f75f40e7cd7c8496f403873007960adf` | `253f322f2bf7bbc18c378efdeea82639a5cb5a922a028f55213c53ddead12114` | `27e9ff743915f11b` |
| `v286_isolated_reprobe_chunk005_ord036` | `Cast Tinder Wall` terminal loss | `Return a Forest you control to its owner's hand: Untap target creature. Activate only once each turn.` terminal win | `5739755198258cfa501cac89d8e615899257172df23fc2a5e788c79b34542404` | `75b107292cdce353ad97c8993afa7ab38c4032233ac12496c5c9c0ad6c9e0298` | `d7a77237c52bb781` |

Conclusion:

- The live-checkpoint branch path has now produced three isolated, repeat-confirmed deterministic correction rows from v262.
- Training/HPC still should not start automatically from the raw miner outputs. Correction evidence must come from the fail-closed export gate below, not from raw batch `clean_positive_needs_isolated_reprobe` rows.

## v287 Confirmed Correction Manifest

Purpose:

- Add a checkpoint-derived correction export path that records only source-loss/sibling-win rows that passed deterministic reentry and isolated positive confirmation.

Implementation:

- Added `scripts/mtgrl/export_checkpoint_corrections.py`.
- Inputs can be probe CSV files or run directories containing `live_checkpoint_branch_probe.csv`.
- The admission gate is `source_loss_alternate_win_isolated_reentry_confirmed`.
- A row is admitted only when it is `clean_positive`, has at least two candidates, source reentry A/B matched, reentry candidate/state hashes match the captured candidate/state hashes, the source branch is a terminal loss, the selected sibling branch is a terminal win, there are no branch errors, and all positive confirmation repeats pass.
- The script writes `confirmed_checkpoint_corrections.csv`, `confirmed_checkpoint_corrections.jsonl`, `rejected_checkpoint_corrections.csv`, `manifest_summary.json`, and an optional generated README under the requested output directory.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/export_checkpoint_corrections.py` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v287_confirmed_correction_manifest` | Exported v282/v283/v286 with `admitted_rows=3`, `rejected_rows=0`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v287_confirmed_correction_manifest_audit` | Audited v282/v283/v284/v285/v286 with `input_rows=82`, `admitted_rows=3`, `rejected_rows=79`. The rejected audit rows include all `clean_positive_needs_isolated_reprobe` and `clean_positive_unstable` outcomes. |

Admitted manifest rows:

| Artifact | Source | Confirmed sibling | Candidate hash | State hash | RNG hash |
| --- | --- | --- | --- | --- | --- |
| `v282_isolated_reprobe_chunk004_ord027` | `Cast Saruli Caretaker` terminal loss | `Cast Lead the Stampede` terminal win | `4afec00e0b8824ce9106834d4468e76ca332ed5790df1887711554d82b5e6846` | `a2446accbda286e63d8b85b88e3df6536fc61b26b17aac1e9d6cbd66a1629b67` | `3991df502d5ac7cb` |
| `v283_isolated_reprobe_chunk004_ord053` | `Cast Balustrade Spy` terminal loss | `Cast Lead the Stampede` terminal win | `deda1a2784be4abcac67a9b3c2e39c47f75f40e7cd7c8496f403873007960adf` | `253f322f2bf7bbc18c378efdeea82639a5cb5a922a028f55213c53ddead12114` | `27e9ff743915f11b` |
| `v286_isolated_reprobe_chunk005_ord036` | `Cast Tinder Wall` terminal loss | `Return a Forest you control to its owner's hand: Untap target creature. Activate only once each turn.` terminal win | `5739755198258cfa501cac89d8e615899257172df23fc2a5e788c79b34542404` | `75b107292cdce353ad97c8993afa7ab38c4032233ac12496c5c9c0ad6c9e0298` | `d7a77237c52bb781` |

Next unit:

- Build a gated training-data generation path that consumes `confirmed_checkpoint_corrections.csv` only, writes a small supervised correction dataset with provenance and hashes, and refuses to run when the manifest has zero admitted rows or any row fails the same admission invariants.

## v288 Gated Correction Dataset

Purpose:

- Convert the confirmed v287 manifest into a supervised first-action correction dataset while preserving the snapshot/hash proof needed to replay or audit each label.

Implementation:

- Added `scripts/mtgrl/build_checkpoint_correction_dataset.py`.
- The script consumes `confirmed_checkpoint_corrections.csv`, revalidates the same admission invariants, requires non-empty examples, checks source/target indices, requires local snapshot files by default, joins `selected_snapshots.csv` ranking context, and writes JSONL/CSV examples plus `dataset_summary.json`.
- Each example labels the terminal-winning sibling as `target_indices`/`target_texts` and records the accepted-policy terminal-loss source action under `negative`.
- This is dataset generation only. It does not import into a trainer or launch training.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/build_checkpoint_correction_dataset.py` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v288_checkpoint_correction_dataset` | Generated `checkpoint_correction_v1` with `examples=3`, `errors=0`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v288_rejected_manifest_dataset_gate_check` | Expected fail-closed check against `rejected_checkpoint_corrections.csv`: `examples=0`, `errors=79`; all rows failed the manifest/admission proof fields. |

Dataset rows:

| Example | Source loss | Target win | Selected prob | Value score | Turn | Pressure |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `e54e81574c0753ee1d615a39` | `Cast Saruli Caretaker` | `Cast Lead the Stampede` | `0.14072587` | `-0.25376269` | `9` | own life `16`, opponent permanents `3` |
| `48eb0eb2a60db9c087cfbc96` | `Cast Balustrade Spy` | `Cast Lead the Stampede` | `0.26092055` | `0.00349175` | `15` | own life `9`, opponent permanents `7` |
| `23ffdd4cf59398c4f11f35c4` | `Cast Tinder Wall` | `Return a Forest you control to its owner's hand: Untap target creature. Activate only once each turn.` | `0.20412067` | `-0.13766083` | `7` | own life `18`, opponent permanents `5` |

Next unit:

- Mine a larger deterministic live-checkpoint slice from the accepted-policy v262/v231-family checkpoint corpus to grow this dataset beyond 3 examples before considering any trainer import.

## v289-v291 Counterfactual Value Tree

Purpose:

- Move beyond binary source-vs-one-sibling correction rows toward graded action-importance estimates: "the source action loses in all sampled continuations, while action C wins in most/all sampled continuations" should be stronger evidence than a small win-rate gap.

Implementation:

- Extended `LiveCheckpointBranchMiner` with `--value-tree=true`.
- Value-tree mode reuses serialized live checkpoints, validates source reentry twice, enumerates root action choices, runs configurable continuations for each action, and writes:
  - `counterfactual_value_tree.csv`: one row per root action with rollout counts, terminal/win/loss/error counts, win/loss/terminal rates, delta vs source, and importance score.
  - `counterfactual_value_tree_summary.csv`: one row per checkpoint with source-vs-best action rates, classification, reentry hashes, and aggregate rollout counts.
- Scalable knobs:
  - `--tree-rollouts N`: sampled continuations per root action.
  - `--tree-max-actions K`: cap root action branching; `0` means all legal root candidates.
  - `--tree-include-pass true|false`: include or skip pass/done/stop-like root choices, while always retaining the source action.
  - `--tree-continuation-policy stable|sample`: deterministic stable autopilot or seeded sampled autopilot after the forced root action.
  - `--tree-timeout-sec N`: per-rollout terminal bound.
  - `--tree-seed N`: reproducible sampled-continuation seed.
- This is a bounded sampled tree, not an exhaustive MTG game tree. HPC scaling should increase checkpoints, root actions, and rollouts; it should not assume exhaustive game enumeration is tractable.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed after the value-tree patch. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v289_value_tree_smoke_v286` | Stable continuation, v286 snapshot, all 6 root actions, 1 rollout each. Classification `dominant_correction`; source `Cast Tinder Wall` win rate `0.000000`, best action win rate `1.000000`, importance `1.000000`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v290_value_tree_sample_smoke_v286` | Sampled continuation, v286 snapshot, 3 root actions, 2 rollouts each. Classification `strong_correction`; source win rate `0.000000`, best sampled action win rate `0.500000`, importance `0.500000`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v291_branch_miner_legacy_reentry_smoke` | Legacy non-value-tree mode still reentered the same v286 snapshot and classified `reentry_matched`. |

Interpretation:

- The value-tree path now captures exactly the graded evidence shape we wanted: source action loss rate, alternate action win rate, delta, terminal coverage, and an importance score.
- `stable` mode is deterministic and cheap; it is suitable for artifact gates and reproducibility checks.
- `sample` mode explores downstream continuation variation; it is the knob to scale on local background runs or Slurm/HPC after local smoke gates.
- No trainer import is attached yet. The next unit is to run a larger value-tree mining slice and export only high-confidence value-tree corrections into the dataset gate.

## v292-v293 Value-Tree Slice And Weighted Export

v292 run:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v292_value_tree_ranked_sample_slice20`.
- Scope: ranked top 20 `ACTIVATE_ABILITY_OR_SPELL` v262 live checkpoints, `--ranked-max-per-game 4`, `--tree-max-actions 6`, `--tree-rollouts 2`, sampled continuation policy, `120s` rollout timeout.
- Result: 20 checkpoint summaries, 119 action rows, and classification counts `{no_better_action=15, strong_correction=3, dominant_correction=2}`.

Top weighted rows:

| Classification | Source loss | Best sampled target | Source win rate | Target win rate | Delta | Importance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `dominant_correction` | `Cast Balustrade Spy` | `{T}: Add {G}.` | `0.000000` | `1.000000` | `1.000000` | `1.000000` |
| `dominant_correction` | `Cast Gatecreeper Vine` | `Cast Roost Seek` | `0.000000` | `1.000000` | `1.000000` | `1.000000` |
| `strong_correction` | `Cast Masked Vandal` | `Cast Lead the Stampede` | `0.000000` | `0.500000` | `0.500000` | `0.500000` |
| `strong_correction` | `Cast Generous Ent` | `{T}: Add {G}.` | `0.000000` | `0.500000` | `0.500000` | `0.500000` |
| `strong_correction` | `Cast Sagu Wildling` | `Cast Lead the Stampede` | `0.000000` | `0.500000` | `0.500000` | `0.500000` |

v293 export:

- Added `scripts/mtgrl/export_value_tree_corrections.py`.
- The export gate is `value_tree_source_loss_best_win_delta`.
- Defaults admit only `dominant_correction` and `strong_correction` rows with source loss rate `>=1.0`, source and target terminal rates `>=1.0`, target win rate `>=0.5`, delta `>=0.5`, positive importance, and matched reentry hashes.
- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v293_value_tree_correction_manifest`.
- Result: `admitted_rows=5`, `rejected_rows=15`; rejected rows were all `no_better_action` or below the value/importance thresholds.

## v294 Weighted Value-Tree Dataset

Purpose:

- Convert the admitted value-tree manifest into training-shaped examples without flattening graded branch evidence into a binary label.

Implementation:

- Added `scripts/mtgrl/build_value_tree_correction_dataset.py`.
- Dataset version: `weighted_checkpoint_correction_v1`.
- The builder rechecks manifest status, admission gate, candidate/state/RNG anchors, local snapshot presence, source/target index ranges, duplicate ids, reentry hashes, positive source-loss/target-win rates, terminal outcome coverage, and positive `label_weight`.
- JSONL examples keep structured `label`, `negative`, `decision`, `value_tree`, `proof`, and `provenance` blocks. CSV examples expose the same training-critical fields plus `label_weight`.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/build_value_tree_correction_dataset.py` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v294_value_tree_correction_dataset` | Built 5 examples from v293 with `errors=0`; classification counts `{dominant_correction=2, strong_correction=3}`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v294_rejected_value_tree_dataset_gate_check` | Expected fail-closed run against rejected v293 rows: `examples=0`, `errors=15`; no rejected row entered the dataset. |

Next unit:

- Scale the miner on a wider local/HPC-ready slice using the same value-tree knobs, then compare yield and label-weight distribution before attaching trainer import.

## v295-v296 Parallel Value-Tree Sharding

Problem:

- v295 proved the wider sampled value-tree run was productive, but it used one JVM and therefore underused the machine. It was stopped as a partial single-worker run after 45 summaries: `{no_better_action=25, moderate_correction=15, strong_correction=5}`.

Implementation:

- Added deterministic selection sharding to `LiveCheckpointBranchMiner`:
  - `--selection-shards N`
  - `--selection-shard-index I`
  - aliases: `--shards`, `--shard-index`
- Added `scripts/mtgrl/run_value_tree_shards.py`, a local process-level shard launcher. It starts one Maven/JVM miner per shard, writes per-shard logs and CSVs, then merges `selected_snapshots.csv`, `counterfactual_value_tree.csv`, and `counterfactual_value_tree_summary.csv`.
- Sharding partitions the already-ranked selected checkpoint set by modulo over selection rank. That keeps every shard on the same top-N frontier and avoids duplicated checkpoint work.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/run_value_tree_shards.py` | Passed. |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed after Java sharding patch. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v296_value_tree_sharded_smoke` | 2 shards, 4 ranked snapshots, 3 root actions, 1 stable rollout each. Both shard exit codes `0`; merged output has 4 summary rows and 12 action rows. |

Next unit:

- Relaunch the wider v295 shape through the sharded runner with enough local workers to use available CPU, then export admitted weighted corrections from the merged output.

## v297-v300 Sharded Scale Run And Focused Lists

v297 sharded run:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v297_value_tree_sharded_slice100_r4_s8`.
- Scope: 8 local JVM shards on the v262 live-checkpoint corpus, ranked top 100 request, `--ranked-max-per-game 10`, `--tree-max-actions 8`, `--tree-rollouts 4`, sampled continuation policy, `120s` rollout timeout.
- Result: all 8 shard exit codes `0`; the merged output contains 78 selected summaries and 467 action rows in 300 seconds.
- Classification counts: `{no_better_action=49, moderate_correction=23, strong_correction=6}`.

v298-v299 export:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v298_value_tree_sharded_correction_manifest`.
- Strict default gate admitted 4 weighted corrections and rejected 74 rows. The two rejected `strong_correction` rows had `source_loss_rate=0.750000`, so they were correctly held out rather than treated as confirmed source mistakes.
- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v299_value_tree_sharded_correction_dataset`.
- Dataset result: 4 `weighted_checkpoint_correction_v1` examples, `errors=0`.

Focused-list support:

- Added `LiveCheckpointBranchMiner --snapshot-list <file>` and `scripts/mtgrl/run_value_tree_shards.py --snapshot-list <file>`.
- Snapshot lists support deterministic sharding like checkpoint roots, so a broad low-rollout pass can feed a narrower high-rollout confirmation pass.
- The Java list reader strips a leading UTF-8 BOM to tolerate PowerShell-generated list files.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `local-training/local_pbt/live_checkpoint_branch_miner/v300_snapshot_list_sharded_smoke` | 2 shards over 4 focused paths, 3 root actions, 1 stable rollout. Exit codes `0`; merged output has 4 summaries and 12 action rows with no snapshot load errors. |

Next unit:

- Rerun the 29 v297 `strong_correction`/`moderate_correction` rows with higher sampled rollout count to refine weights and promote only rows that stay above the strict admission gate.

## v301-v304 Focused High-Rollout Preference Tier

v301 focused rerun:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v301_value_tree_focus29_r12_s8`.
- Input list: 29 snapshots from v297 whose first pass classified as `strong_correction` or `moderate_correction`.
- Scope: 8 local JVM shards, `--tree-max-actions 8`, `--tree-rollouts 12`, sampled continuation policy, `120s` rollout timeout.
- Result: all 8 shard exit codes `0`; merged output has 29 summaries and 197 action rows in 398 seconds.
- Classification counts after higher rollout: `{no_better_action=12, weak_correction=11, moderate_correction=6}`.

Strict gate:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v302_value_tree_focus29_strict_manifest`.
- Result: `admitted_rows=0`.
- Interpretation: none of the v297 strong/moderate rows stayed above the strict `target_win_rate >= 0.5` / `delta >= 0.5` correction threshold after 12 sampled continuations per root action.

Preference tier:

- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v303_value_tree_focus29_preference_manifest`.
- Thresholds: admit `moderate_correction` or stronger rows with source loss rate `>=1.0`, source/target terminal rates `>=1.0`, target win rate `>=0.25`, and delta `>=0.25`.
- Result: `admitted_rows=6`, `rejected_rows=23`.
- Artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v304_value_tree_focus29_preference_dataset`.
- Dataset result: 6 `weighted_checkpoint_correction_v1` examples, `errors=0`.

Preference rows:

| Source loss | Preferred action | Target win rate | Label weight | Terminal rollouts |
| --- | --- | ---: | ---: | ---: |
| `Cast Tinder Wall` | `Return a Forest you control to its owner's hand: Untap target creature. Activate only once each turn.` | `0.333333` | `0.333333` | `72` |
| `Cast Lotleth Giant` | `{T}: Add {G} for each creature you control with defender.` | `0.250000` | `0.250000` | `72` |
| `Play Forest` | `Pass` | `0.250000` | `0.250000` | `72` |
| `Cast Generous Ent` | `Cast Roost Seek` | `0.250000` | `0.250000` | `96` |
| `Cast Overgrown Battlement` | `Cast Saruli Caretaker` | `0.333333` | `0.333333` | `72` |
| `Cast Lotus Petal` | `Forestcycling {1}` | `0.416667` | `0.416667` | `96` |

Conclusion:

- Full exhaustive MTG branching is still intractable, but branchable checkpoints now support a scalable sampled tree: broad sharded discovery, focused high-rollout confirmation, strict correction export, and lower-weight preference export.
- The current high-rollout result argues against treating v297 as hard correction evidence, but it does produce calibrated preference weights that can support a separate lower-confidence training lane.

## v305-v309 True Model Continuation Probe

Problem:

- The v301/v304 preference tier included a suspicious row where source `Play Forest` was labeled worse than `Pass`.
- Investigation found that the original value-tree continuation mode was still using post-root branch autopilot for model decisions whenever a branch controller was installed. A previous `--post-branch-autopilot false` probe was therefore not a true model-continuation run.

Implementation:

- Added `EngineDecisionBranchController.shouldBypassModelInference()`.
- `ComputerPlayerRL` now skips model inference for branch controllers only when that hook returns true.
- `LiveCheckpointBranchMiner` returns `postBranchAutopilot` from the hook, so `--post-branch-autopilot false` forces only the root checkpoint decision and then lets normal model scoring handle later decisions for both players.
- `scripts/mtgrl/run_value_tree_shards.py` now records `--post-branch-autopilot`, injects safe single-backend local Python defaults for true model-continuation probes, and assigns per-shard Py4J ports. This avoids the earlier learner-plus-four-inference-worker startup storm for tiny branch probes.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/run_value_tree_shards.py` | Passed. |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v307_play_forest_true_model_continuation_probe` | Failed before evidence: default local multi-backend launched learner plus four inference gateways and hit repeated Py4J channel failures while Python workers spun. Processes were stopped. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v308_play_forest_true_model_single_backend_smoke` | True model continuation, source `Play Forest` and alternate `Pass`, 1 rollout each. Reentry matched twice; both branches reached terminal loss; classification `no_better_action`. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v309_play_forest_true_model_all_actions_r1` | True model continuation, all 6 root actions, 1 rollout each. Reentry matched twice; all six actions reached terminal loss; classification `no_better_action`. |

Conclusion:

- The `Play Forest` -> `Pass` preference should not be admitted as training evidence. Under true model continuation, `Pass` no longer wins; it ties the source as a terminal loss in the smoke probes.
- The earlier preference row is best classified as a post-root autopilot artifact, not a model-policy counterfactual.
- Batching is still the right scaling direction for true model continuations, but it needs a frontier/service design: branch workers should enqueue model-decision states to a central batched inference service and resume when policy/value results return. The current runner batches only within a single Python bridge and otherwise relies on process-level JVM sharding.

## v310 Focused True-Model Continuation Pass

Artifact:

- `local-training/local_pbt/live_checkpoint_branch_miner/v310_focus29_true_model_r1_s4`.

Scope:

- Reused the v301 focused 29-snapshot list.
- Ran 4 local shards with `--post-branch-autopilot false`, `model_continuation_backend=single`, `--tree-max-actions 4`, `--tree-rollouts 1`, sampled continuation policy, and `60s` branch timeout.

Result:

- All 4 shard exit codes were `0`.
- Runtime was 1290 seconds.
- Merged output has 29 selected summaries and 115 action rows.
- Classification counts: `{dominant_correction=1, no_better_action=19, no_terminal_evidence=9}`.

Flagged event:

| Snapshot | Source action | Preferred action | Source result | Preferred result | Delta | Notes |
| --- | --- | --- | --- | --- | ---: | --- |
| `chunk_004/game_2a141cb3-4106-4c74-95fa-770c53b4b1ee_ord026_D053_ACTIVATE_ABILITY_OR_SPELL.ser.gz` | `Cast Lotus Petal` | `Cast Winding Way` | `0/1` wins, terminal loss | `1/1` wins, terminal win | `1.000000` | Reentry matched twice; candidate hash `a09a41fc65348151dcece6de73dcd2933cef61b22ed7b2ff542384ea4de53ecd`; state hash `b2ddc771eba7e7603a6f431c60194b6b9412bd2c42e28b4cabed4521c507058e`. |

Candidate-set detail for the flagged snapshot:

| Action | Result |
| --- | --- |
| `Cast Lotus Petal` | `terminal_loss` |
| `Pass` | `error=IllegalStateException: Live checkpoint branch timeout` |
| `Cast Lead the Stampede` | `terminal_loss` |
| `Cast Winding Way` | `terminal_win` |

Interpretation:

- This is the first true-model-continuation focused run to produce a final correction flag, but it is still a low-sample signal: one rollout per evaluated root action and only 3 of 4 evaluated actions reached terminal.
- The flagged row is stronger than the invalidated `Play Forest` -> `Pass` row because both reentry checks matched and the source/preferred branches each reached terminal with opposite outcomes.
- Shard logs still show RL activation failures on mana/untap abilities and recurring no-source continuous-effect logs, so this event should be treated as a rerun target before training admission rather than immediate dataset evidence.

## v311-v313 Sequence-Aware Order Probes

Problem:

- One-step branch labels can confuse "bad first action" with "good action but bad follow-up order".
- In Magic, many action orders are equivalent, but some are not. The miner needed a way to force short prefixes such as `A then B` and `B then A`, record whether the resulting states converge, and keep order-sensitive cases out of immediate one-step training evidence.

Implementation:

- Added optional value-tree sequence mode:
  - `--sequence-tree true`
  - `--tree-sequence-depth <N>`; v1 supports depth 2 ordered pairs.
  - `--tree-sequence-beam <N>`; pairs are formed from the first N value-tree root choices.
  - `--tree-sequence-rollouts <N>`.
- Added `counterfactual_sequence_tree.csv`, one row per forced ordered prefix rollout.
- Added `counterfactual_sequence_tree_summary.csv`, one row per unordered pair with forward/reverse aggregate classification.
- Extended `SnapshotBranchController` to force a root checkpoint choice by index, then force later same-action-type source decisions by candidate text. This lets the controller test `A then B` even though the second candidate index is only known after `A` changes the game state.
- Sequence rows record forced-step completion, prefix failure reason, post-prefix state hash, terminal result, and errors/timeouts.
- Sequence summaries classify pairs as `order_converged`, `order_sensitive_forward_better`, `order_sensitive_reverse_better`, `order_diverged_same_value`, `sequence_incomplete`, or `sequence_error`.
- `scripts/mtgrl/run_value_tree_shards.py` now exposes the sequence flags and merges the two new sequence CSV artifacts.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python -m py_compile scripts/mtgrl/run_value_tree_shards.py` | Passed. |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v312_sequence_order_flagged_smoke_classifier` | Single true-model checkpoint smoke on the v310 flagged snapshot. Wrote 4 value rows, 1 value summary, 12 sequence rows, and 6 sequence summaries. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v313_sequence_sharded_runner_smoke` | 1-shard runner smoke over one snapshot-list entry. Exit code `0`; merged 3 value rows, 1 value summary, 6 sequence rows, and 3 sequence summaries. |

v312 flagged-pair result:

| Pair | Forward | Reverse | Classification | State convergence |
| --- | --- | --- | --- | --- |
| `Cast Lotus Petal` / `Cast Winding Way` | `Cast Lotus Petal` then `Cast Winding Way`: terminal loss | `Cast Winding Way` then `Cast Lotus Petal`: terminal loss | `order_diverged_same_value` | Post-prefix hashes differed, so the orders did not converge, but both sampled continuations lost. |

Interpretation:

- The earlier v310 one-step `Cast Lotus Petal` -> `Cast Winding Way` dominant flag does not survive this sequence-aware confirmation smoke as immediate training evidence.
- The pair is not order-equivalent: both orders completed and produced different post-prefix state hashes.
- In this smoke it is also not a clean order-sensitive correction: both completed orders reached terminal losses.
- The new sequence mode gives the miner the missing distinction between commutative orders, truly order-sensitive wins/losses, and incomplete or timed-out forced-prefix attempts.

## v314 Focused Sequence True-Model Pass

Artifact:

- `local-training/local_pbt/live_checkpoint_branch_miner/v314_focus29_sequence_true_model_r1_s4`

Scope:

- Reused the v301 focused 29-snapshot list.
- Ran 4 local JVM shards with `--post-branch-autopilot false`, `model_continuation_backend=single`, `--tree-max-actions 4`, `--tree-rollouts 1`, sampled continuation policy, `--tree-timeout-sec 30`, and sequence-tree depth 2 / beam 4 / 1 rollout.

Result:

- All 4 shard exit codes were `0`.
- Runtime was `1980.243` seconds.
- Merged value-tree output has 29 summaries and 115 action rows.
- Value-tree classification counts: `{no_better_action=19, no_terminal_evidence=10}`.
- Strict value candidates: `0`.
- Value action wins: `0`.
- Merged sequence-tree output has 342 ordered-prefix rows and 171 pair summaries.
- Sequence classification counts: `{sequence_incomplete=116, order_diverged_same_value=24, sequence_error=25, order_converged=6}`.
- Sequence row outcomes: `terminal_loss=114`, `prefix_step_1_unavailable=136`, `timeout=92`, terminal wins `0`.
- Sequence prefix completion rate: `206/342 = 60.23%`.
- Sequence terminal rate: `114/342 = 33.33%`.
- Sequence timeout rate: `92/342 = 26.90%`.

Artifact summarizer:

- Added `scripts/mtgrl/summarize_value_tree_run.py`.
- The script reads merged or sharded value-tree/sequence-tree CSVs and writes compact JSON/Markdown quality reports with value classification counts, strict-candidate count, terminal rates, sequence prefix-completion rates, timeout rates, and top value rows.
- Validation: `python -m py_compile scripts/mtgrl/summarize_value_tree_run.py`.
- Final v314 reports: `artifact_summary.json` and `artifact_summary.md` under the v314 artifact directory.

Interpretation:

- v314 invalidates the remaining focused-list true-model sequence candidates as immediate training evidence. The v310 one-step positive was already rejected by v312; the full 29-row focused sequence pass found no terminal-winning value or ordered-prefix branch.
- The checkpoint and sequence machinery is working, but true-model continuation quality is currently sparse: roughly one third of sequence rows terminalized, roughly one quarter timed out, and many ordered pairs were legitimately infeasible because the second action was unavailable after the first.
- The next unit should move from this exhausted focused list back to corpus density: collect or mine a larger accepted-policy Affinity live-checkpoint set, then use the summarizer as the first quality gate. Do not train from v314.

## v315-v316 Fresh Affinity Live-Checkpoint Corpus

Artifacts:

- `local-training/local_pbt/cp7_eval_sweeps/20260523_v315_affinity_live_checkpoints_g16_cpu`
- `local-training/local_pbt/live_checkpoint_branch_miner/v316_v315_loss_true_model_value_r1_s4`

Scope:

- v315 collected a fresh accepted-policy `Pauper-Spy-Combo-Value` vs CP7 `Grixis Affinity` corpus with compact replay metadata and live checkpoints.
- The run produced 16 live-checkpoint manifests and 840 serialized checkpoint snapshots.
- Counted eval result was 6 wins over 15 counted games; chunk 2 reported `0/0`.
- v316 mined 427 snapshots from counted-loss chunks 1, 5, 6, 8, 9, 13, 14, 15, and 16.
- The value-tree pass selected 60 ranked snapshots, used 4 local shards, `--tree-max-actions 4`, `--tree-rollouts 1`, sampled true-model continuation, `--tree-timeout-sec 45`, and no sequence tree.

Result:

- All 4 v316 shard exit codes were `0`.
- Runtime was 2310 seconds.
- Merged value-tree output has 60 summaries and 231 action rows.
- Classification counts before the source-terminal label repair: `{no_better_action=35, no_terminal_evidence=24, strong_correction=1}`.
- Strict value candidates: `0`.
- Value action wins: `1`.
- Value terminal rate: `0.562771`.

False-positive label:

| Snapshot | Source action | Best action | Source result | Best result | Prior label | Strict gate |
| --- | --- | --- | --- | --- | --- | --- |
| `chunk_001/game_2f3f7172-5e40-4b98-917d-50f8ac7b1123_ord003_D003_ACTIVATE_ABILITY_OR_SPELL.ser.gz` | `Cast Land Grant` | `Cast Roost Seek` | timeout / nonterminal | terminal win | `strong_correction` | rejected |

Interpretation:

- The raw v316 `strong_correction` label was not admissible evidence because the source branch never reached terminal loss.
- The strict gate correctly rejected it, but the broad label was misleading for triage and summaries.
- Do not train from v316.

## v317 Source-Terminal Classification Repair

Implementation:

- `LiveCheckpointBranchMiner.valueTreeClassification` now requires terminal source evidence before any correction label can be emitted.
- If the source action never terminalizes, the value-tree summary is classified as `source_not_terminal`.
- If the source terminalizes but does not lose, the summary is classified as `source_terminal_not_loss`.
- If the source loses but the best sibling never terminalizes, the summary remains `no_terminal_evidence`.
- Correction labels are now reserved for cases with source terminal-loss evidence plus a better terminal sibling.

Validation:

| Command / Artifact | Result |
| --- | --- |
| `python "$env:USERPROFILE\.codex\skills\mage-research-agent\scripts\airl_maven.py" compile` | Passed. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v317_v316_source_unresolved_classification_smoke` | One-snapshot smoke on the v316 false-positive checkpoint. The source and sibling branches timed out; the summary classified as `source_not_terminal`, not a correction. |
| `local-training/local_pbt/live_checkpoint_branch_miner/v317b_v316_seed_source_unresolved_smoke` | Same checkpoint with the v316 seed. This rerun terminalized all four root choices as losses and classified as `no_better_action`; strict candidates remained `0`. |

Interpretation:

- The false-positive mechanism is fixed at the classifier gate: a nonterminal source branch can no longer produce `strong_correction` from a sibling win-rate delta.
- The exact continuation outcome remains stochastic enough that a single checkpoint rerun may not reproduce the same sibling win from v316, so source-terminal gating must remain the primary evidence guard.
- The next mining pass should use the repaired labels and continue corpus-density search, with training still blocked until strict checkpoint-derived correction evidence exists.

## v318-v319 Repaired-Label Mining and Confirmation

Artifacts:

- `local-training/local_pbt/live_checkpoint_branch_miner/v318_v315_loss_true_model_value_r1_s8_more`
- `local-training/local_pbt/live_checkpoint_branch_miner/v319_v318_strict_sequence_confirm_r3_s2`

v318 scope:

- Reused the v315 counted-loss snapshot list.
- Selected up to 160 ranked snapshots with `ranked_max_per_game=16`.
- Ran 8 local shards with repaired source-terminal labels, `--tree-max-actions 4`, `--tree-rollouts 1`, sampled true-model continuation, `--tree-timeout-sec 60`, no sequence tree, and no post-branch autopilot.

v318 result:

- All 8 shard exit codes were `0`.
- Runtime was 2100 seconds.
- Merged value-tree output has 127 summaries and 411 action rows.
- Classification counts: `{no_better_action=77, source_not_terminal=41, source_terminal_not_loss=7, dominant_correction=2}`.
- Strict value candidates: `2`.
- Value action wins: `16`.
- Value terminal rate: `0.695864`.

v318 strict candidates:

| Snapshot | Source action | Best action | Source result | Best result | Notes |
| --- | --- | --- | --- | --- | --- |
| `chunk_014/game_d0e844fd-bc6c-4a5b-b4c8-449c034046de_ord013_D025_ACTIVATE_ABILITY_OR_SPELL.ser.gz` | `Play Swamp` | `Flashback sacrifice three creatures` | 1/1 terminal loss | 1/1 terminal win | One-rollout `dominant_correction`. |
| `chunk_008/game_04d8b181-859f-47e3-8a68-53e1b567bc95_ord052_D080_ACTIVATE_ABILITY_OR_SPELL.ser.gz` | `Pass` | `Forestcycling {1}` | 1/1 terminal loss | 1/1 terminal win | One-rollout `dominant_correction`. |

v319 confirmation scope:

- Rechecked the two v318 strict candidates only.
- Ran 2 local shards with `--tree-rollouts 3`, sequence tree enabled, depth 2, beam 4, `--tree-sequence-rollouts 2`, sampled true-model continuation, `--tree-timeout-sec 90`, and no post-branch autopilot.

v319 result:

- All 2 shard exit codes were `0`.
- Runtime was 240 seconds.
- Merged value-tree output has 2 summaries and 7 action rows.
- Value classification counts: `{no_better_action=2}`.
- Strict value candidates: `0`.
- Every root action terminalized and lost: 21 terminal losses over 21 value rollouts, with no value wins and no errors.
- Sequence output has 36 rows and 9 pair summaries.
- Sequence classification counts: `{order_diverged_same_value=7, sequence_incomplete=2}`.
- Sequence rows produced 30 terminal losses, 6 unavailable second steps, no wins, no errors, and no timeouts.

Interpretation:

- The two v318 positives were one-rollout noise, not training evidence.
- Sequence checks were useful but did not rescue either candidate: completed ordered prefixes generally diverged in state hash while preserving the same terminal-loss value.
- The next broad mining pass should use repeat value rollouts as the first-stage gate. One-rollout scans are useful for corpus coverage, but they are too noisy to form a correction shortlist without immediate repeat confirmation.
- Do not train from v318 or v319.

## v320 Repeat-Rollout Value Gate

Artifact:

- `local-training/local_pbt/live_checkpoint_branch_miner/v320_v315_loss_true_model_value_r3_s8_repeat_gate`

Scope:

- Reused the v315 counted-loss snapshot list after v319 invalidated the v318 one-rollout positives.
- Selected up to 96 ranked snapshots with `ranked_max_per_game=12`.
- Ran 8 local shards with `--tree-rollouts 3`, `--tree-max-actions 4`, sampled true-model continuation, `--tree-timeout-sec 75`, no sequence tree, and no post-branch autopilot.

Result:

- All 8 shard exit codes were `0`.
- Runtime was 5281 seconds.
- Merged value-tree output has 96 summaries and 339 action rows.
- Classification counts: `{no_better_action=70, source_not_terminal=22, source_terminal_not_loss=3, strong_correction=1}`.
- Strict value candidates: `1`.
- Value action wins: `27`.
- Value terminal rate: `0.738446`.

Surviving repeat-rollout candidate:

| Snapshot | Source action | Best action | Source result | Best result | Classification |
| --- | --- | --- | --- | --- | --- |
| `chunk_008/game_04d8b181-859f-47e3-8a68-53e1b567bc95_ord010_D018_ACTIVATE_ABILITY_OR_SPELL.ser.gz` | `Cast Overgrown Battlement` | `Cast Tinder Wall` | 0/3 wins, 3/3 terminal losses | 2/3 wins, 3/3 terminal | `strong_correction` |

Interpretation:

- Repeat-rollout gating sharply reduced the one-rollout noise: only 1 of 96 ranked snapshots survived the strict correction gate.
- This D018 row is the strongest local checkpoint-derived candidate in this lane so far, but it is not yet training evidence. It needs isolated repeat confirmation plus sequence/order probing to make sure the win signal is not a short-prefix or continuation-order artifact.
- Next unit: run a focused confirmation pass on this snapshot with more value rollouts and sequence tree enabled.

## v321 D018 Sequence Confirmation

Artifact:

- `local-training/local_pbt/live_checkpoint_branch_miner/v321_v320_d018_sequence_confirm_r7_s1`

Scope:

- Rechecked the surviving v320 candidate only: `chunk_008/...ord010_D018_ACTIVATE_ABILITY_OR_SPELL.ser.gz`.
- Ran one local shard with `--tree-rollouts 7`, `--tree-max-actions 4`, sequence tree enabled, depth 2, beam 4, `--tree-sequence-rollouts 3`, sampled true-model continuation, `--tree-timeout-sec 90`, and no post-branch autopilot.

Result:

- The focused confirmation produced one value summary, four value rows, 36 sequence rows, and six sequence pair summaries.
- Value classification counts: `{no_better_action=1}`.
- Source action `Cast Overgrown Battlement`: 0/7 wins, 7/7 terminal losses.
- Previously best action `Cast Tinder Wall`: 0/7 wins, 7/7 terminal losses.
- Sequence classifications were `{order_diverged_same_value=3, sequence_incomplete=3}` with no terminal wins.

Interpretation:

- The v320 D018 candidate did not survive isolated repeat confirmation.
- Do not train from v320 or v321.
- The remaining question is no longer whether this specific D018 correction is real; it is whether the branchable checkpoint search can find a true Spy Combo terminal-win line at all under terminal-only reward.

## v322 Terminal-Line Findability Proof

Implementation:

- Added `LiveCheckpointBranchMiner --terminal-line-search true`.
- The mode forces a root checkpoint action, uses the branch controller for subsequent sampled continuations, records a bounded decision trace, and writes `terminal_line_search.csv`.
- Added `--tree-continuation-policy explore`, which still uses only terminal outcomes but samples spells, mana, and pass/done choices instead of always preferring nonterminal actions. This makes phase advancement reachable during controlled search.

Artifacts:

- `local-training/local_pbt/live_checkpoint_branch_miner/v322_terminal_line_search_findable_smoke`
- `local-training/local_pbt/live_checkpoint_branch_miner/v322_terminal_line_search_findable_repeat`

Command shape:

```powershell
mvn -o -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests '-Dexec.mainClass=mage.player.ai.rl.LiveCheckpointBranchMiner' '-Dexec.args=--snapshot-list <winning_late_snapshots.txt> --out <run_dir> --terminal-line-search true --max-snapshots 6 --tree-max-actions 8 --line-max-root-actions 8 --line-attempts 24 --line-timeout-sec 20 --tree-timeout-sec 20 --tree-seed 2026052401 --tree-continuation-policy explore --tree-include-pass true --selection-mode path' exec:java
```

Result:

- The smoke searched late checkpoints from the six v315 counted winning chunks.
- It stopped after the first processed checkpoint, `chunk_011/...ord128_D230_ACTIVATE_ABILITY_OR_SPELL.ser.gz`.
- Attempt 0 forced `{T}: Add {G} for each creature you control with defender.` and reached a terminal loss.
- Attempt 1 forced `Pass` and reached `terminal_win`.
- Smoke classification counts: `{terminal_loss=1, terminal_win=1}`.
- Repeat run with the same seed, same first snapshot, and two attempts reproduced the same root outcomes: attempt 0 terminal loss, attempt 1 terminal win.

Winning-line evidence:

- Root: `Pass` at ordinal 128 / decision 230.
- Recorded trace length: 166 decisions in the smoke artifact.
- The trace includes `Cast Balustrade Spy`, opponent target selection, later `Flashback sacrifice three creatures`, target selections including `Lotleth Giant`, and terminal win.

Interpretation:

- The true combo/win line is findable from a real branchable checkpoint with terminal-only reward; no combo-specific reward was needed for this proof.
- This is not yet correction-quality evidence because it starts from a late checkpoint in a game already known to be winnable.
- The repeat reproduced win/loss outcomes and root choices, but the winning final state hash was not identical across runs. Treat v322 as a findability proof and keep exact deterministic-line stability as a separate engineering gate before using full traces as supervised training targets.

## v323 Local Terminal-Line Batch

Implementation:

- Extended `scripts/mtgrl/run_value_tree_shards.py` with `--mode terminal-line` so checkpoint terminal-line search can run across local JVM shards and merge `terminal_line_search.csv`.
- Added `scripts/mtgrl/summarize_terminal_line_search.py`, which converts raw `line_trace` fields into compact per-attempt markers: terminal outcome, root action, decision count, Spy/Dread Return/Lotleth markers, pass/mana/cast counts, and key event steps.

Artifact:

- `local-training/local_pbt/live_checkpoint_branch_miner/v323_winning_late_mid_terminal_line_r8_s4`

Scope:

- 48 late and midgame snapshots from the six v315 counted winning chunks.
- 4 local shards.
- `--mode terminal-line`
- `--line-attempts 8`
- `--line-max-root-actions 8`
- `--line-timeout-sec 20`
- `--tree-continuation-policy explore`
- `--line-stop-on-win true`
- `--line-stop-on-win-all false`

Result:

- All four shards exited `0`.
- Runtime was 70 seconds.
- Merged `terminal_line_search.csv` has 253 attempts.
- Outcome counts: `{terminal_win=18, terminal_loss=134, not_terminal=action_type_mismatch=96, error=checkpoint_no_reentry_decision=5}`.
- Compact summary counts:
  - terminal rows: 152 / 253
  - valid terminal wins: 18
  - Spy rows: 137
  - Spy wins: 13
  - Dread Return rows: 131
  - Lotleth target rows: 53
  - full combo-pattern wins: 6

Interpretation:

- The local search now repeatedly finds terminal wins and multiple full combo-pattern wins from real branchable checkpoints.
- The many `action_type_mismatch` rows are mostly a checkpoint/reentry quality signal, especially from target-selection surfaces. The next larger run should favor `ACTIVATE_ABILITY_OR_SPELL` checkpoints first and treat target-selection roots as a separate repair lane.
- This is enough for an artifact-only HPC smoke: scale terminal-line mining and compact summaries, but do not start training yet.

## v325 Zaratan Terminal-Line Smoke

Implementation:

- Built and uploaded runtime bundle `rl-runtime-d31e5d942a-20260523-231808.tar.gz`.
- Uploaded a compact 48-snapshot payload from v323 plus the terminal-line summarizer.
- Ran the AIRL miner directly from the runtime bundle classpath in Slurm, without depending on the remote source checkout.
- Fixed the summarizer for Zaratan's default Python 3.6 in commit `69ca3f2f3e`.

Artifacts:

- Remote run: `/home/jmaior/scratch.msml603/jmaior/mage/local-training/hpc/terminal_line_smoke/v325_terminal_line_hpc_smoke`
- Local payload staging: `local-training/hpc/terminal_line_v325_payload_69ca3f2f3e`

Result:

- Slurm job `19379187` completed successfully in 97 seconds on `compute-a5-8`.
- All four Java shards exited `0`.
- Snapshot count: 48.
- Selected snapshots: 30.
- Merged `terminal_line_search.csv` rows: 145.
- Outcome counts: `{terminal_loss=126, terminal_win=19}`.
- Compact summary counts:
  - terminal rows: 145 / 145
  - valid terminal wins: 19
  - Spy rows: 125
  - Spy wins: 13
  - Dread Return rows: 114
  - Lotleth target rows: 57
  - full combo-pattern wins: 5
  - max combo score: 11

Interpretation:

- The terminal-line miner is now proven to run on Zaratan from a branchable checkpoint corpus, with direct-jar execution and compact summary artifacts.
- Restricting the smoke to `ACTIVATE_ABILITY_OR_SPELL` roots removed the v323 `action_type_mismatch` noise and produced only terminal outcomes.
- This still is not training evidence by itself. The next quality gate is a teacher-label extractor that groups terminal-line rows by checkpoint/root action, requires paired terminal siblings, and emits preference labels only when win-rate or terminal-value separation clears strict thresholds.

## v326-v328 Teacher-Label Quality Gate

Implementation:

- Added `scripts/mtgrl/export_terminal_line_teacher_labels.py`.
- The exporter groups terminal-line rows by checkpoint and compares root actions using terminal win/loss only.
- Labels are admitted only when a target root is paired against a worse source root from the same checkpoint and clears configured win-rate, terminal-rate, attempt-count, and delta thresholds.
- Added `LiveCheckpointBranchMiner --line-common-continuation-seeds true` and runner support so sibling roots can be evaluated under the same continuation sample ids.
- New terminal-line CSVs include `continuation_sample` and `continuation_seed`.

Artifacts:

- One-sample local labels: `local-training/local_pbt/terminal_line_teacher_labels/v326_v323_strict_labels`
- One-sample HPC-smoke labels: `local-training/hpc/terminal_line_v325_results/teacher_labels_strict`
- Repeat confirmation without common seeds: `local-training/local_pbt/live_checkpoint_branch_miner/v327_v325_label_repeat_confirm`
- Repeat confirmation with common seeds: `local-training/local_pbt/live_checkpoint_branch_miner/v328_v325_label_repeat_common_seed`
- Paired-label export: `local-training/local_pbt/terminal_line_teacher_labels/v328_common_seed_paired_labels`

Result:

- v323 one-sample strict gate admitted 4 labels from 31 checkpoint groups.
- v325 one-sample strict gate admitted 9 labels from 30 checkpoint groups.
- v327 repeated those 9 candidate-label checkpoints with 32 attempts per snapshot and `line_stop_on_win=false`; strict repeat gate admitted 0 labels.
- v328 repeated the same slice with common continuation seeds:
  - 288 terminal rows
  - 57 wins
  - 21 full combo-pattern wins
  - strict repeat gate admitted 0 labels
  - moderate paired gate admitted 3 labels

Moderate paired labels from v328:

| Target | Source | Paired wins | Aggregate rates |
| --- | --- | --- | --- |
| `Cast Mesmeric Fiend` | `Cast Roost Seek` | 2/4 vs 0/4 | 0.50 vs 0.00 |
| `Cast Overgrown Battlement` | `{T}: Add {G}.` | 2/4 vs 0/4 | 0.50 vs 0.00 |
| `{T}, Sacrifice {this}: Add one mana of any color.` | `Cast Dread Return` | 3/5 vs 1/5 | 0.60 vs 0.166667 |

Interpretation:

- Terminal-line search is finding wins and combo-pattern wins, but one-sample hard labels are not stable enough for training.
- Common continuation seeds make the comparison fairer and reveal that many root choices share the same downstream winning continuation. This supports treating the miner as a value-estimation teacher first, not as a hard correction-label generator yet.
- The next scale unit should mine many more checkpoints with common continuation seeds, aggregate paired value estimates, and train only from labels that survive repeat gates or use low-weight pairwise/value targets rather than hard corrections.

## v329 Zaratan Common-Seed Scale Smoke

Artifact:

- Remote run: `/home/jmaior/scratch.msml603/jmaior/mage/local-training/hpc/terminal_line_common_seed/v329_terminal_line_common_seed_r32_s8`
- Slurm job: `19379200`
- Runtime bundle: `rl-runtime-86711ab584-20260523-234430.tar.gz`

Scope:

- 48-snapshot v323 payload.
- 8 Java shards.
- `--line-attempts 32`
- `--line-max-root-actions 8`
- `--line-stop-on-win false`
- `--line-common-continuation-seeds true`
- Post-run compact summary plus paired-label export.

Result:

- Slurm completed successfully in 4 minutes 29 seconds.
- All eight shards exited `0`.
- Selected snapshots: 30.
- Merged terminal-line rows: 960.
- Outcome counts: `{terminal_loss=747, terminal_win=213}`.
- Compact summary:
  - terminal rows: 960 / 960
  - win rate: 0.221875
  - Spy rows: 772
  - Spy wins: 102
  - Dread Return rows: 696
  - Lotleth target rows: 370
  - full combo-pattern wins: 56
  - max combo score: 11
- Paired-label export:
  - checkpoint groups: 30
  - admitted moderate labels: 3
  - rejected groups: 27

Admitted paired moderate labels:

| Target | Source | Notes |
| --- | --- | --- |
| `Cast Tinder Wall` | `Cast Saruli Caretaker` | moderate paired terminal delta |
| `Forestcycling {1}` | `Cast Generous Ent` | moderate paired terminal delta |
| `Cast Quirion Ranger` | `Cast Roost Seek` | moderate paired terminal delta |

Interpretation:

- The common-seed terminal-line miner scales cleanly on Zaratan and produces many terminal wins without combo-specific rewards.
- Even at 960 terminal rows, strict hard-correction labels remain sparse; the useful signal is currently moderate paired value deltas.
- Next unit: build a value-target dataset/export that preserves paired win-rate deltas and confidence weights instead of forcing every surviving comparison into a hard correction target.

## v330 Terminal-Line Value Target Bridge

Implementation:

- Added `scripts/mtgrl/export_terminal_line_value_targets.py`.
- The exporter groups terminal-line rows by checkpoint, prefers paired common-continuation win rates when available, and emits soft candidate distributions plus confidence weights.
- Pass-best rows are not treated as universally invalid. They stay visible through `quality_flags`; likely phase-artifact rows such as low-evidence `Pass` over setup actions are excluded from training manifests by default as `suspect_pass_best`, with `--include-suspect-pass-best` available for diagnostics.
- Added `TerminalLineValueTargetTrainingDataExporter`, which reloads the checkpoint snapshot, reenters the root decision, captures the normal AIRL candidate tensors, attaches the soft value target as `mctsVisitTargets`, and writes serialized `TrainingData`.
- Added a branch-controller training-data capture hook so simulation checkpoint copies can export tensors without enabling normal simulation training.

Local validation artifact:

- Value target export: `local-training/local_pbt/terminal_line_value_targets/v330_v328_value_targets_softpass`

Validation result on the v328 common-seed repeat artifact:

- Value-target CSV:
  - checkpoint groups: 9
  - trainable examples: 5
  - rejected groups: 4
  - suspect pass-best exclusions: 1
  - normal low-delta rejections: 3
- TrainingData export:
  - exported records: 5 / 5
  - all five rows reentered with matching candidate hash and state hash
  - all five rows captured `TrainingData`
  - all five soft targets normalized to sum 1.0
- Imported score probe:
  - examples scored: 5
  - top1: 1 / 5
  - target-set top1: 5 / 5
  - average target probability: 0.171492
- BC-direct fit-score smoke:
  - train passes: 40
  - before strict top1: 0 / 5
  - after strict top1: 4 / 5
  - before average target probability: 0.152297
  - after average target probability: 0.286678

Interpretation:

- The pipeline can now convert terminal-only branch search evidence into actual AIRL training tensors without replaying from the original game log.
- This is still a tiny local fit smoke, not an evaluation result. The next unit is to run the same value-target export and serialized TrainingData bridge on the v329 Zaratan scale artifact, then train a candidate model from the larger dataset before launching a policy evaluation sweep.

## v331 v329 Value Target Export

Implementation:

- Added `--snapshot-path-prefix-from` / `--snapshot-path-prefix-to` to `TerminalLineValueTargetTrainingDataExporter`.
- This lets locally downloaded Zaratan snapshot corpora be reentered without rewriting value-target CSV artifacts by hand.

Artifact:

- Value target export: `local-training/local_pbt/terminal_line_value_targets/v331_v329_value_targets_softpass`
- Diagnostic export with suspect pass-best rows visible: `local-training/local_pbt/terminal_line_value_targets/v331_v329_value_targets_softpass_diagnostic`

Validation result on the v329 Zaratan common-seed scale artifact:

- Value-target CSV:
  - checkpoint groups: 30
  - trainable examples: 2
  - rejected groups: 28
  - suspect pass-best exclusions: 8
  - low-delta rejections: 14
- TrainingData export:
  - exported records: 2 / 2
  - both rows reentered with matching candidate hash and state hash
  - both rows captured `TrainingData`
  - both soft targets normalized to sum 1.0
- Imported score probe:
  - examples scored: 2
  - strict top1: 1 / 2
  - target-set top1: 2 / 2
  - average target probability: 0.407844
- BC-direct fit-score smoke:
  - train passes: 16
  - strict top1 stayed 1 / 2
  - target-set top1 stayed 2 / 2
  - average target-set mass improved from 0.441548 to 0.507690

Interpretation:

- The local replay bridge for HPC snapshot corpora is validated.
- v329 is too sparse for a meaningful model/evaluation sweep: only two clean trainable value-target examples survived the current admission gate.
- The diagnostic suspect pass-best rows mostly remain low-value or low-delta pass-over-setup cases, so the correct next unit is more terminal-line mining breadth/depth, not training from suspect pass-best examples.

## v332-v334 Full-Corpus Value Target Candidate

Artifacts:

- Full-corpus miner: `local-training/local_pbt/live_checkpoint_branch_miner/v332_v315_full_common_seed_r64_s12`
- Value targets: `local-training/local_pbt/terminal_line_value_targets/v332_v315_full_common_seed_r64_s12_softpass`
- Candidate eval: `local-training/local_pbt/cp7_eval_sweeps/20260524_v334_terminal_line_v332_candidate_affinity_g16_gpu`

v332 full-corpus terminal-line result:

- Selected checkpoint groups: 706.
- Terminal rows: 45,184 / 45,184.
- Terminal wins: 4,313.
- Win rate: 0.095454.
- Spy rows: 33,675.
- Spy wins: 3,396.
- Dread Return rows: 28,712.
- Lotleth target rows: 10,855.
- Full combo-pattern wins: 1,328.

Value-target bridge result:

- Admitted value-target examples: 166.
- Rejected groups: 540.
- Major rejection reasons: `value_delta_below_threshold=477`, `suspect_pass_best=282`.
- TrainingData export: 166 / 166 records reentered and captured.
- Baseline score probe on exported records: strict top1 25 / 166, target-set top1 164 / 166, average target probability 0.227512.
- Fit-score probe after four local epochs: strict top1 61 / 166, target-set top1 166 / 166, average target probability 0.261963.
- Candidate profile score probe after import training: strict top1 60 / 166, target-set top1 166 / 166, average target probability 0.252503.

Eval result:

- The first attempted candidate eval, `20260524_v333_terminal_line_v332_candidate_affinity_g16`, is invalid. The shared inference service launched from `.mtgrl_venv` while that venv still had `torch 2.12.0+cpu`, then failed repeatedly with `AssertionError: Torch not compiled with CUDA enabled`.
- `.mtgrl_venv` was repaired locally to `torch 2.12.0+cu130`; the clean v334 eval logged `split-device mode: infer=cuda:0 train=cpu`.
- Clean v334 candidate result vs Grixis Affinity skill 7: 1 / 16, win rate 0.0625.
- Recent v315 comparator artifact, `local-training/local_pbt/cp7_eval_sweeps/20260523_v315_affinity_live_checkpoints_g16_cpu`, was 6 / 15, win rate 0.4.
- Because v334 did not include replay metadata or live logs, two follow-up logged GPU diagnostics were run:
  - v335 small logged diagnostic: `local-training/local_pbt/cp7_eval_sweeps/20260524_v335_terminal_line_v332_candidate_affinity_g4_logs_gpu`, 2 / 4.
  - v336 replay-logged/live-checkpoint candidate eval: `local-training/local_pbt/cp7_eval_sweeps/20260524_v336_terminal_line_v332_candidate_affinity_g16_logs_gpu`, 5 / 16, win rate 0.3125.

Interpretation:

- The terminal-line value-target bridge scales mechanically: checkpoint reentry, tensor export, and supervised import training all work on a 166-example full-corpus dataset.
- The imported terminal-line candidate is not a promotion candidate. It improves the offline value-target score probe but does not improve the current Affinity gate; the best replay-logged comparison is v336 at 5 / 16 versus v315 at 6 / 15.
- The v334 1 / 16 result should be treated as a clean but pessimistic unlogged eval, not the sole policy conclusion. Replay-logged v336 is the better comparator for this lane.
- Action-health diagnostics on v336 found no pass-over-land issue. The notable deltas versus the v315 logged baseline were more mulligans/London bottoms, much higher mana-ability churn, more Quirion Ranger untap usage, and fewer Spy/Dread Return executions per game.
- The next unit should train a safer low-weight or mixed replay candidate before any HPC promotion sweep. The likely issue is that direct BC/value import on sparse, off-policy terminal-line roots perturbs general play and mulligan/trunk behavior while only improving the offline terminal-line target set.

## v337 Low-Weight Terminal-Line Import

Artifact:

- Candidate profile: `Pauper-Spy-Combo-Value-TerminalLine-v337-LowWeight`
- Import output: `local-training/local_pbt/terminal_line_value_targets/v332_v315_full_common_seed_r64_s12_softpass/import_train_candidate_v337_lowweight`
- Score probe: `local-training/local_pbt/terminal_line_value_targets/v332_v315_full_common_seed_r64_s12_softpass/candidate_v337_lowweight_score_probe`
- Eval: `local-training/local_pbt/cp7_eval_sweeps/20260524_v337_lowweight_terminal_line_candidate_affinity_g16_logs_gpu`

Setup:

- Started from the current `Pauper-Spy-Combo-Value` baseline model.
- Imported the same 166 v332 terminal-line value-target examples.
- Used `BC_DIRECT_LOSS_COEF=0.20`, `train-epochs=2`, `candidate-permutations=1`.

Result:

- Import training passes: 332.
- Score probe: strict top1 43 / 166, target-set top1 166 / 166, average target probability 0.248136, average rank 2.7892.
- Logged GPU eval vs Grixis Affinity skill 7: 3 / 16, win rate 0.1875.
- Action-health scan again found no pass-over-land issue.

Interpretation:

- Lowering the direct-loss coefficient reduced offline overfit relative to v332 while still improving over baseline on the terminal-line target set.
- It did not preserve live play. The current terminal-line direct-import recipe should not be promoted or scaled to HPC.
- The next training direction should avoid trunk-wide direct BC import as the first use of terminal-line evidence. Prefer either a candidate-value/ranking head, a mixed replay anchor with baseline-policy retention, or a playtime search consumer that uses branch values without immediately distorting the policy.

## v338-v340 Candidate-Q Value Head Integration

Implementation:

- Extended `TerminalLineValueTargetTrainingDataExporter` with `--target-mode`:
  - `distribution`: existing soft target distribution.
  - `signed-values`: observed candidate win rates mapped to `[-1, 1]`, with unobserved candidates masked out.
  - `advantage-values`: observed candidates mapped to decision-local relative value from `-1` for the worst observed sibling to `+1` for the best observed sibling.
- Exposed per-candidate Q predictions through the Py4J and shared-GPU inference paths.
- Added score-probe diagnostics for the direct Q vector: target Q, Q-top index, Q rank, Q-top target mass, and policy/Q agreement.

Artifacts:

- v338 profile: `Pauper-Spy-Combo-Value-TerminalLine-v338-QOnly`
- v339 profile: `Pauper-Spy-Combo-Value-TerminalLine-v339-QOnlySigned`
- v340 profile: `Pauper-Spy-Combo-Value-TerminalLine-v340-QOnlyAdvantage`
- Training corpus root: `local-training/local_pbt/terminal_line_value_targets/v332_v315_full_common_seed_r64_s12_softpass`
- v338 eval: `local-training/local_pbt/cp7_eval_sweeps/20260524_v338_qonly_blend025_affinity_g8_logs_gpu`
- v340 evals:
  - `local-training/local_pbt/cp7_eval_sweeps/20260524_v340_qonly_advantage_blend10_affinity_g8_logs_gpu`
  - `local-training/local_pbt/cp7_eval_sweeps/20260524_v340_qonly_advantage_blend025_affinity_g8_logs_gpu`

v338 soft-distribution Q-only result:

- Trained only the candidate-Q head from the 166 soft terminal-line examples.
- Offline score probe barely moved with Q blending:
  - blend 0.00: strict top1 39 / 166, average target probability 0.252359.
  - blend 0.25: strict top1 43 / 166, average target probability 0.252323.
  - blend 1.00: strict top1 43 / 166, average target probability 0.252207.
- Logged GPU eval vs Grixis Affinity skill 7 at blend 0.25: 2 / 8.
- Action-health scan: 5 Spy-cast games / 8, 3 premature Dread Return flashbacks, and no Lotleth-ready Dread Return target opportunities.

v339 signed absolute-return Q-only diagnosis:

- Re-exported the same 166 terminal-line examples as signed absolute terminal values.
- Offline score probes stayed weak: strict top1 39-42 / 166 across blends 0.00-1.00.
- Direct Q diagnostic at blend 0.00:
  - Q top1: 48 / 166.
  - Average target Q: -0.8383.
  - Average top Q: -0.8150.
  - Q-top positive-target count: 6 / 166.
- Corpus diagnosis explains the pessimism: best observed sibling win rate averaged only 0.2576, and only 6 / 166 trainable rows had best sibling value above 0.5. Most decisions in this corpus are "least bad" local choices, not absolute winning actions.

v340 advantage-value Q-only result:

- Re-exported the 166 rows as decision-local sibling advantages, preserving relative action quality even when all siblings usually lose.
- Offline score probes improved monotonically with Q blending:
  - blend 0.00: strict top1 39 / 166, average target probability 0.252359, average rank 2.8193.
  - blend 0.25: strict top1 47 / 166, average target probability 0.255176, average rank 2.7892.
  - blend 0.50: strict top1 53 / 166, average target probability 0.257894, average rank 2.7711.
  - blend 1.00: strict top1 56 / 166, average target probability 0.263027, average rank 2.7470.
- Direct Q diagnostic at blend 0.00:
  - Q top1: 53 / 166.
  - Average Q rank: 2.6265.
  - Q-top positive-target count: 79 / 166.
- Logged GPU eval vs Grixis Affinity skill 7:
  - blend 1.00: 1 / 8.
  - blend 0.25: 2 / 8.
- Blend 0.25 action-health scan: 5 Spy-cast games / 8, 2 premature Dread Return flashbacks, and no Lotleth-ready Dread Return target opportunities.

Interpretation:

- The candidate-Q head now carries a measurable terminal-line sibling-ranking signal. The v340 advantage target is the first form that clearly improves offline Q ranking.
- Directly blending that Q signal into live action logits is not a promotion path yet. It remains below the replay-logged v336 direct candidate result of 5 / 16 and the recent v315 comparator of 6 / 15, and it still shows premature Dread Return behavior.
- The evidence supports keeping terminal-line search as a value/ranking teacher, not as a direct policy overwrite. The next useful direction is a gated playtime-search consumer or branch-relevant decision hook that consults the Q/value signal only where branch evidence is applicable, plus broader terminal-line data before any HPC-scale promotion sweep.

## v341 Candidate-Q Blend Gate Probe

Implementation:

- Added optional inference-time gates for `CANDIDATE_Q_BLEND`:
  - `CANDIDATE_Q_BLEND_MIN_TOP_Q`: require the top candidate-Q value to clear a threshold.
  - `CANDIDATE_Q_BLEND_MIN_MARGIN`: require the top candidate-Q value to beat the runner-up by a threshold.
- Defaults preserve the existing behavior: no gate is active unless one of those environment variables is set.

Local probes:

- top-Q gate `CANDIDATE_Q_BLEND_MIN_TOP_Q=0.0`, blend 0.25:
  - offline strict top1: 39 / 166.
- top-Q gate `CANDIDATE_Q_BLEND_MIN_TOP_Q=-0.10`, blend 0.25:
  - offline strict top1: 42 / 166.
- margin gate `CANDIDATE_Q_BLEND_MIN_MARGIN=0.02`, blend 0.25:
  - offline strict top1: 45 / 166.
  - logged GPU eval vs Grixis Affinity skill 7: 2 / 8.
  - action-health scan: 2 Spy-cast games / 8, 1 premature Dread Return flashback, and no Lotleth-ready Dread Return target opportunities.

Interpretation:

- Simple confidence gating reduces some live perturbation but also gives up offline sibling-ranking lift. It did not improve the Affinity gate.
- This confirms that the next step should not be more direct Q logit blending. The terminal-line signal needs a decision-local search/verification consumer or richer value data, not another global score-shaping variant.

## v342-v343 Playtime Search Diagnostic

Setup:

- Used the existing eval-time `ISMCTS_ENABLE` path in `ComputerPlayerRL` with `BeliefISMCTS.searchRoot`.
- Q blending was disabled (`CANDIDATE_Q_BLEND=0.0`) so the test isolated playtime root search.
- Root-search settings were intentionally small:
  - `ISMCTS_RANDOM_ROLLOUT_ROOT=1`
  - `ISMCTS_ROOT_DETERMINIZATIONS=1`
  - `ISMCTS_ROOT_ROLLOUTS_PER_ACTION=1`
  - `MCTS_ROOT_TOP_K=4`
  - `MCTS_SELECTIVE_ENABLE=1`
  - Spy-relevant selective keywords.

Results:

- First v342 attempt without `run_cp7_eval_sweep.py --mcts` did not activate MCTS:
  - logged `ismctsEnabled=false`
  - `mcts_activations=0`
  - root cause: the eval harness overwrites `ISMCTS_ENABLE` unless launched with `--mcts`.
- Corrected v342 playtime-search diagnostic:
  - artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v342_playtime_belief_rollout_mcts_affinity_g4_logs_gpu`
  - result vs Grixis Affinity skill 7: 2 / 4.
  - MCTS activations: 43 across 4 games.
  - gate stats showed selective filtering was active: many candidate decisions were skipped as `not_tactical`, while Spy-relevant roots did activate.
- v343 one-game logging smoke:
  - artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v343_mcts_log_labels_affinity_g1_logs_gpu`
  - result: 1 / 1, with 18 MCTS activations.
  - Added candidate labels to `[MCTS]` game-log lines so visits/values/picked indices can be audited directly.

Interpretation:

- Playtime branching exists and can fire from the live evaluator when the harness `--mcts` switch is used.
- The tiny random-rollout root setup is too noisy to treat as a promotion signal. Many root values are single-sample `-1/0/1`, and timeout/top-k effects can leave useful candidates unvisited.
- This path is still the right shape for a decision-local consumer, but it needs a cleaner branch budget/admission rule and candidate-labelled diagnostics before a larger sweep.

## v344-v345 Outcome-Only All-Action Value Discovery

Purpose:

- Re-center the checkpoint branch work on the core thesis: terminal wins/losses are the only training signal, with no hand-authored Spy Combo milestones used as labels or rewards.
- Broaden terminal-line mining beyond `ACTIVATE_ABILITY_OR_SPELL` roots so target and selection decisions can also receive terminal-derived value.
- Keep pass-best cases visible as diagnostics, but do not promote low-evidence pass-over-setup rows into the strict training set.

v344 fresh capture:

- Eval artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v344_outcome_only_checkpoint_capture_g4_logs_gpu`
- Policy/profile: baseline `Pauper-Spy-Combo-Value`
- Opponent/gate: CP7 `Grixis Affinity`, skill 7
- MCTS and candidate-Q blending: disabled
- Result: 1 / 4
- Live checkpoints: 262 total
  - `ACTIVATE_ABILITY_OR_SPELL`: 214
  - `SELECT_TARGETS`: 37
  - `SELECT_CARD`: 6
  - `DECLARE_BLOCKS`: 5

v344 terminal-line mining:

- Miner artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v344_outcome_only_g4_r16_s8_allactions`
- Settings: 8 shards, all captured action types, `line_attempts=16`, `line_max_root_actions=8`, common continuation seeds, `tree_continuation_policy=explore`
- Terminal-line rows: 4,192
- Terminal wins/losses: 600 wins, 2,952 losses
- Diagnostics: 624 `action_type_mismatch`, 16 `checkpoint_no_reentry_decision`
- Strict value targets: `local-training/local_pbt/terminal_line_value_targets/v344_outcome_only_g4_r16_s8_allactions_softpass`
  - Checkpoint groups: 222
  - Admitted examples: 87
  - Rejected groups: 135
  - Suspect pass-best exclusions: 82
  - Low-delta rejections: 96
  - Action mix: 82 `ACTIVATE_ABILITY_OR_SPELL`, 5 `SELECT_TARGETS`
  - TrainingData export: 87 / 87 reentered and serialized
  - Baseline score probe: strict top1 28 / 87, target-set top1 86 / 87, average target probability 0.328188, average rank 2.4483
- Diagnostic inclusive value targets: `local-training/local_pbt/terminal_line_value_targets/v344_outcome_only_g4_r16_s8_allactions_include_pass_diagnostic`
  - Admitted examples with suspect pass-best included: 126

v345 all-action scale-up on the existing v315 corpus:

- Source checkpoint corpus: `local-training/local_pbt/cp7_eval_sweeps/20260523_v315_affinity_live_checkpoints_g16_cpu/live_checkpoints`
- Miner artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v345_outcome_only_v315_g16_r16_s8_allactions`
- Settings: 12 shards, all captured action types, `line_attempts=16`, `line_max_root_actions=8`, common continuation seeds, `tree_continuation_policy=explore`
- Selected snapshots: 840
- Terminal-line rows: 13,440
- Terminal wins/losses: 1,110 wins, 10,506 losses
- Diagnostics: 1,696 `action_type_mismatch`, 128 `checkpoint_no_reentry_decision`
- Strict value targets: `local-training/local_pbt/terminal_line_value_targets/v345_outcome_only_v315_g16_r16_s8_allactions_softpass`
  - Checkpoint groups: 726
  - Admitted examples: 255
  - Rejected groups: 471
  - Suspect pass-best exclusions: 310
  - Low-delta rejections: 344
  - Action mix: 246 `ACTIVATE_ABILITY_OR_SPELL`, 8 `SELECT_TARGETS`, 1 `SELECT_CARD`
  - TrainingData export: 255 / 255 reentered and serialized
  - Baseline score probe: strict top1 88 / 255, target-set top1 254 / 255, average target probability 0.340051, average rank 2.3059
- Diagnostic inclusive value targets: `local-training/local_pbt/terminal_line_value_targets/v345_outcome_only_v315_g16_r16_s8_allactions_include_pass_diagnostic`
  - Admitted examples with suspect pass-best included: 382

Interpretation:

- The outcome-only branch pipeline now produces more strict value targets by broadening action types, without adding Spy-specific labels or rewards.
- The larger v345 strict dataset is mechanically usable: every admitted row reentered its checkpoint and serialized into normal AIRL `TrainingData`.
- The baseline model already keeps nearly every target action inside its support set, but often does not rank the terminal-derived best sibling first. That makes this dataset useful as a ranking/value-head teacher, not as evidence for another direct policy overwrite.
- The next thesis-aligned unit is to train a contained candidate-Q or ranking-head candidate from v345, evaluate it as an offline score probe first, and only then consider a small live eval. Do not mutate the baseline profile with fit probes; use a cloned candidate profile or Q-head-only import.

## v346-v350 Signed Candidate-Q and On-Policy Iteration

Purpose:

- Test whether v345 outcome-only all-action terminal targets can train a contained candidate-Q head without policy overwrite.
- Keep the training target thesis-clean: all labels come from downstream terminal wins/losses, not Spy-specific combo detectors.
- Use live checkpoints from the improved candidate as the next on-policy branch corpus.

v346 unsigned target failure:

- Candidate profile: `Pauper-Spy-Combo-Value-TerminalLine-v346-QOnlyAdvantageAllAction`
- Training data: v345 advantage-value `TrainingData`, 255 examples, 4 epochs, 1,020 train passes.
- Mistake: imported with `-CandidateQFromMctsTargets` but without signed branch-return mode.
- Result: offline top1 did not improve:
  - blend 0.00: 83 / 255
  - blend 0.25: 82 / 255
  - blend 0.50: 82 / 255
  - blend 1.00: 82 / 255
- Diagnosis: Q values saturated near 1.0 for nearly all candidates because the unsigned MCTS-target loss masks non-positive advantage entries and trains only positive entries.

v347 signed target repair:

- Candidate profile: `Pauper-Spy-Combo-Value-TerminalLine-v347-QOnlySignedAdvantageAllAction`
- Same 255 v345 examples, but trained with `-BranchReturnTargets` so the Q loss consumes signed candidate advantages and masks only out-of-range sentinel values.
- Offline score probe:
  - baseline, blend 0.00 on same `.ser`: 88 / 255, average rank 2.3059
  - baseline, blend 1.00 on same `.ser`: 91 / 255, average rank 2.3333
  - v347 blend 0.00: 83 / 255, average rank 2.3294
  - v347 blend 0.25: 92 / 255, average rank 2.2902
  - v347 blend 0.50: 102 / 255, average rank 2.2471
  - v347 blend 1.00: 106 / 255, average rank 2.2314
- Direct Q diagnostic at blend 1.00: Q top1 125 / 255, average Q rank 2.267, with non-saturated negative-valued advantage outputs.
- Live eval artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v347_qonly_signed_advantage_blend10_affinity_g8_logs_gpu`
  - Grixis Affinity skill 7, no MCTS, `CANDIDATE_Q_BLEND=1.0`
  - Result: 2 / 8
  - Captured 562 live checkpoint files.
  - Log mining showed at least one clean Spy -> Dread Return -> Lotleth Giant win, plus one loss that reached the combo shell but did not convert.

v348-v349 on-policy terminal-line mining:

- v348 smoke miner artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v348_onpolicy_v347_g8_r16_s8_allactions`
  - Default `ranked_max_per_game=10` selected only 80 checkpoints.
  - Rows: 1,280; terminal wins/losses: 48 / 992.
  - Strict value targets: 18 admitted examples. This was a useful smoke but too small for the next candidate.
- v349 expanded miner artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v349_onpolicy_v347_g8_r16_s12_allactions_rank96`
  - `ranked_max_per_game=96`, 12 shards, 530 selected checkpoints.
  - Rows: 8,480.
  - Terminal wins/losses: 416 / 7,232.
  - Diagnostics: 736 `action_type_mismatch`, 96 `checkpoint_no_reentry_decision`.
  - Strict value targets: `local-training/local_pbt/terminal_line_value_targets/v349_onpolicy_v347_g8_r16_s12_allactions_rank96_softpass`
    - Checkpoint groups: 478
    - Admitted examples: 134
    - Rejected groups: 344
    - Suspect pass-best exclusions: 201
    - Low-delta rejections: 300
    - Advantage `TrainingData` export: 134 / 134 reentered and serialized.

v350 combined signed-Q candidate:

- Candidate profile: `Pauper-Spy-Combo-Value-TerminalLine-v350-QOnlySignedAdvantageCombined`
- Training data: combined v345 + v349 signed advantage exports, 389 examples total.
- Training: Q-head-only import from a fresh baseline clone, 4 epochs, 1,556 train passes, `-BranchReturnTargets`.
- Offline combined score probe:
  - blend 0.00: 121 / 389, average rank 2.3188
  - blend 0.50: 149 / 389, average rank 2.2416
  - blend 1.00: 170 / 389, average rank 2.1877
- Live eval artifacts:
  - `local-training/local_pbt/cp7_eval_sweeps/20260524_v350_qonly_signed_advantage_combined_blend10_affinity_g8_logs_gpu`: 4 / 8 vs Grixis Affinity skill 7, no MCTS.
  - `local-training/local_pbt/cp7_eval_sweeps/20260524_v350_qonly_signed_advantage_combined_blend10_affinity_g16_seed9999_metric_gpu`: 4 / 16 vs Grixis Affinity skill 7, no MCTS.
  - `local-training/local_pbt/cp7_eval_sweeps/20260524_v350_qonly_signed_advantage_combined_blend10_affinity_g16_seed8888_logs_gpu`: partial 1 / 9 before disk-full during DB copy; treat as an infra-interrupted diagnostic, not a completed sweep.

v351-v352 broader disk-light metric sweep and seed-key correction:

- Artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v351_v350_broader_active_pool_g8_metric_gpu`
- Settings: v350 candidate, `CANDIDATE_Q_BLEND=1.0`, skill 7, no MCTS, no game logs or live checkpoints, DB hardlinks enabled.
- Results:
  - Overall: 11 / 24
  - Spy mirror: 6 / 8
  - Jund Wildfire: 2 / 8
  - Mono Red Rally: 3 / 8

- Artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v352_baseline_broader_active_pool_g8_metric_gpu`
- Settings: baseline `Pauper-Spy-Combo-Value`, same v351 opponent filter, skill, no-MCTS setting, replay seed base `4242`, one-game job chunking, no live checkpoints, and DB cleanup after each chunk.
- Results:
  - Overall: 14 / 24
  - Spy mirror: 5 / 8
  - Jund Wildfire: 6 / 8
  - Mono Red Rally: 3 / 8
- Correction: v351/v352 were not exact paired seeds. `run_cp7_eval_sweep.py` historically included the profile name in the stable replay-seed offset key, so two profiles with the same `--replay-seed-base` saw different per-chunk shuffled games.
- Resulting code repair: `scripts/run_cp7_eval_sweep.py` now supports `--seed-key-mode matchup`, which removes the profile name from the seed offset key while preserving it in output filenames. Use this mode for candidate/baseline paired comparisons.

v353-v356 Q-blend consumer probes:

- Added `CANDIDATE_Q_BLEND_HEADS` to the Python model as a generic runtime allow-list for Q blending by decision head. Default remains all heads for backward compatibility; `action,target,card_select` matches the current terminal-line evidence coverage.
- Offline v353 gate sweep on the 389 combined v350 value records:
  - Full blend baseline: 170 / 389, average rank 2.1877.
  - Margin-only gate `CANDIDATE_Q_BLEND_MIN_MARGIN=0.05`: 170 / 389, average rank 2.1774.
  - Positive top-Q gates fell back to baseline-like behavior: 121 / 389, average rank 2.3188. The Q head is mostly relative/negative on these records, so absolute positive-Q thresholding is the wrong confidence signal.
- Live Jund probes before exact seed repair:
  - `20260524_v354_v350_qgate_margin005_jund_g8_metric_gpu`: 2 / 8.
  - `20260524_v356_v350_qhead_action_target_card_jund_g8_metric_gpu`: 2 / 8.
  - These are now classified as unpaired smoke results, not candidate-vs-baseline evidence, because they used the old profile-key seed mode.
- Offline v355 head-gate sanity probe:
  - `CANDIDATE_Q_BLEND_HEADS=action,target,card_select`: 170 / 389, average rank 2.1877, matching full blend on the terminal-line evidence records.

v357-v361 exact-seed paired controls:

- Exact paired seed mode: `--seed-key-mode matchup`, replay seed base `4242`, skill 7, no MCTS, no live checkpoints, DB cleanup after each chunk.
- Head-gated v350 (`CANDIDATE_Q_BLEND_HEADS=action,target,card_select`) versus exact baseline:
  - Jund Wildfire: v357 6 / 8, baseline v358 2 / 8.
  - Spy mirror: v359 6 / 8, baseline v360 4 / 8.
  - Mono Red Rally: v359 2 / 8, baseline v360 2 / 8.
  - Combined: v350 head-gated 14 / 24, baseline 8 / 24.
- Full-blend exact control:
  - `20260524_v361_v350_fullblend_exactseed_active3_g8_metric_gpu`: 14 / 24.
  - Per matchup: Jund Wildfire 6 / 8, Spy mirror 6 / 8, Mono Red Rally 2 / 8.
- Interpretation of the control: the seed-pairing repair, not the head allow-list, explains the reversal from the v351/v352 read. The head allow-list remains useful as a generic safety knob, but it is not proven to improve this small live slice.

v362-v363 exact-seed Grixis Affinity hard-gate check:

- Exact paired seed mode: `--seed-key-mode matchup`, replay seed base `5151`, skill 7, no MCTS, no live checkpoints, DB cleanup after each chunk.
- Candidate artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v362_v350_fullblend_exactseed_grixis_g16_metric_gpu`
  - v350 full blend: 5 / 16.
- Baseline artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v363_baseline_exactseed_grixis_g16_metric_gpu`
  - Baseline `Pauper-Spy-Combo-Value`: 8 / 16.
- Paired chunk deltas:
  - Candidate lost where baseline won: chunks 3, 7, 8, 11.
  - Candidate won where baseline lost: chunk 14.
  - Net: -3 wins for v350 on this exact Grixis slice.

Cleanup / archival:

- Preserved CSV summaries, manifests, logs, checkpoints, and value-target artifacts.
- Removed generated DB and model-snapshot copies from the failed partial seed8888 run.
- Removed reconstructable generated DB and model-snapshot copies from recent v347/v350/v351/v352/v354/v356/v357/v358/v359/v360/v361/v362/v363 eval runs after summaries/logs/checkpoints were preserved.
- Removed stale generated isolated CLI workspaces from `D:\codex-mage-cli-workspaces`.
- D: free space recovered from 0 bytes to about 5 GB after cleanup.

Interpretation:

- The signed-target fix is important. Unsigned Q import is a blocker because it teaches "all observed positive entries are good" and collapses Q discrimination.
- Signed local sibling advantages now produce a measurable offline ranking signal and a stronger local live result than the earlier q-only variants.
- The v351/v352 negative read was an evaluation-control error, not a reliable policy result. Exact paired seeds show the v350 signed-Q candidate at 14 / 24 against an 8 / 24 baseline over Spy mirror, Jund Wildfire, and Mono Red Rally.
- The original Grixis Affinity hard gate remains negative under exact pairing: v350 is 5 / 16 against an 8 / 16 baseline. This blocks promotion and HPC scaling of the current v350 direct-blend consumer.
- The next unit should inspect or mine the exact Grixis disagreement chunks, especially baseline-win/candidate-loss chunks 3, 7, 8, and 11, to determine whether the Q consumer is misranking concrete decisions or whether the current outcome-derived corpus simply lacks Affinity-pressure coverage.

## v364-v380 Deterministic Eval Repair and Grixis Reclassification

Reason for reopening v362/v363:

- The exact Grixis gate result changed when the same chunks were replayed in different run shapes.
- The Java eval side was deterministic-greedy for the RL player, but the shared Python inference path still used four service channels, batching, score-worker concurrency, a dedicated CUDA stream, and normal torch/CUDA settings.
- This made close policy decisions unstable enough that the v362/v363 Grixis "hard gate" could not be trusted as promotion evidence.

Code checkpoint:

- Commit `eee0889e73` (`RL: Add deterministic eval mode`).
- `scripts/run_cp7_eval_sweep.py` now supports `--deterministic-eval`, which forces `parallel=1`, `ai_threads=1`, `GPU_SERVICE_NUM_CHANNELS=1`, score-worker count 1, batch size 1, fixed seed env, `CUBLAS_WORKSPACE_CONFIG`, and `TORCH_DETERMINISTIC_EVAL=1`.
- `gpu_service_core.py` skips the dedicated CUDA inference stream when deterministic eval is enabled.
- `py4j_entry_point.py` disables torch benchmark/TF32 paths and enables deterministic algorithms in warn-only mode when deterministic eval is enabled.
- Validation: Python compile checks, `git diff --check`, and AIRL Maven compile passed before commit.

Deterministic proof runs:

- `20260524_v375_v350_grixis_chunk007_detmode_a`: v350 chunk 7 exact repeat, 0 / 1.
- `20260524_v376_v350_grixis_chunk007_detmode_b`: same command and seed as v375, 0 / 1.
- v375/v376 show the previously unstable chunk 7 is repeatable under deterministic eval mode.

Logged disagreement slice under deterministic eval:

- Candidate artifact: `20260524_v377_v350_grixis_disagreement_chunks_detmode_logs`
  - Chunks 3, 7, 8, 11, 14: 4 / 5.
  - Wins: 3, 8, 11, 14. Loss: 7.
- Baseline artifact: `20260524_v378_baseline_grixis_disagreement_chunks_detmode_logs`
  - Same chunks and seeds: 4 / 5.
  - Wins: 3, 7, 8, 11. Loss: 14.
- First divergent decisions:
  - Chunk 7: v350 chooses `Forestcycling {1}` on turn 3 Precombat Main where baseline chooses `Pass`; v350 loses and baseline wins.
  - Chunk 14: v350 chooses the Quirion Ranger line `Return a Forest you control to its owner's hand: Untap target creature` on turn 1 Precombat Main where baseline chooses `Pass`; v350 wins and baseline loses.
- These are real setup-timing disagreements, not explicit combo labels or hard-coded pass rules.

Full deterministic Grixis gate:

- Candidate artifact: `20260524_v379_v350_grixis_g16_detmode_nolog`
  - v350 full blend: 9 / 16.
- Baseline artifact: `20260524_v380_baseline_grixis_g16_detmode_nolog`
  - Baseline `Pauper-Spy-Combo-Value`: 9 / 16.
- Paired deltas:
  - All chunks match except 7 and 14.
  - Candidate loses where baseline wins: chunk 7.
  - Candidate wins where baseline loses: chunk 14.
  - Net: 0.

Updated interpretation:

- The v362/v363 Grixis hard-gate result is reclassified as nondeterministic eval evidence, not a reliable policy blocker.
- Under the deterministic eval harness, v350 ties the exact Grixis seed set rather than losing it.
- This still is not promotion evidence: v350 needs broader deterministic paired evaluation and likely more Affinity-pressure terminal-line data before HPC scale-up.
- The next thesis-aligned unit is to run the deterministic exact paired controls across the active three-matchup pool, then decide whether to mine additional live checkpoints from the deterministic v350/baseline disagreement surfaces.

Deterministic active-three follow-up:

- Candidate artifact: `20260524_v381_v350_active3_g8_detmode_nolog`
  - Overall: 13 / 24.
  - Spy mirror: 6 / 8.
  - Jund Wildfire: 4 / 8.
  - Mono Red Rally: 3 / 8.
- Baseline artifact: `20260524_v382_baseline_active3_g8_detmode_nolog`
  - Overall: 11 / 24.
  - Spy mirror: 7 / 8.
  - Jund Wildfire: 3 / 8.
  - Mono Red Rally: 1 / 8.
- Paired deltas:
  - Candidate gains Jund chunk 1.
  - Candidate gains Mono Red Rally chunks 5 and 6.
  - Candidate loses Spy mirror chunk 4.
  - Net: +2 wins for v350.
- Updated active-three interpretation:
  - The deterministic harness preserves a positive v350 signal, but the margin shrinks from the earlier shared-service read of 14 / 24 versus 8 / 24 to 13 / 24 versus 11 / 24.
  - This supports continuing the outcome-only branch-value path, but it argues against immediate HPC scale-up from v350 alone. The next higher-value work is mining deterministic disagreement surfaces, especially the Grixis chunk 7 loss and active-three net-positive chunks, to expand terminal-line evidence without adding human combo heuristics.

## v383-v391 Live Checkpoint Model-Continuation Probe

Purpose:

- Test whether checkpoints from a deterministic winning Grixis chunk can reproduce a terminal win without replaying the original prefix.
- Keep the reward/evidence thesis-clean: terminal win/loss only, no combo-specific intermediate reward or action labels.

Artifacts:

- Live checkpoint capture: `local-training/local_pbt/cp7_eval_sweeps/20260524_v383_v350_grixis_chunks07_14_detmode_livecheckpoints`
  - Deterministic eval, v350 candidate, Grixis chunks 7 and 14, compact logs, replay metadata, and live checkpoints.
  - Results matched the deterministic gate: chunk 7 loss, chunk 14 win.
  - Captured 135 checkpoint snapshots under `live_checkpoints`.
- Autopilot miner artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v383_grixis_det_line_r16_ranked12`
  - Ranked 12 snapshots, 192 terminal-line attempts, 0 wins.
  - Classification counts: 176 terminal losses and 16 `checkpoint_no_reentry_decision` errors.
  - Diagnosis: terminal-line search was forcing root actions, then unconditionally using the miner's heuristic autopilot for continuation. This did not test the trained model's continuation policy.
- Local Py4J model-continuation artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v384_grixis_chunk14_ord073_source_model_cont`
  - Forced the known chunk 14 `Cast Balustrade Spy` root from snapshot `ord073_D147_ACTIVATE_ABILITY_OR_SPELL`.
  - Result: terminal loss, but the Maven exec log showed repeated Python gateway connection failures. This was a launcher/environment artifact, not valid model-continuation evidence.
- Shared-GPU model-continuation artifact: `local-training/local_pbt/live_checkpoint_branch_miner/v385_grixis_chunk14_ord073_source_sharedgpu`
  - Same snapshot and forced root as v384, but launched through the deterministic shared GPU service from the v383 eval manifest.
  - Result: 1 / 1 terminal win, source root `Cast Balustrade Spy`, reentry matched, final state hash `6306f436014520f8a1e4fe0ab6dd06f6e6996db47e786ced06c970854e23d1e8`.
- Chunk 14 source-continuation sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v386_grixis_chunk14_source_model_path24`
  - First 24 chunk 14 checkpoints, source root only, model continuation.
  - Result: 21 terminal wins, 1 terminal loss, 2 nonterminal action-type mismatches.
- Chunk 14 root-sibling sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v387_grixis_chunk14_root8_model_path24`
  - Same first 24 chunk 14 checkpoints, up to 8 root choices per checkpoint, model continuation.
  - Result: 157 terminal wins, 19 terminal losses, 16 nonterminal action-type mismatches.
  - Mixed-outcome setup states included `D071` with 2 wins / 6 losses and several earlier states with 50% to 75% terminal win rates across siblings.
- Chunk 7 root-sibling sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v388_grixis_chunk7_root8_model_path24`
  - First 24 chunk 7 checkpoints, up to 8 root choices per checkpoint, model continuation.
  - Result: 93 terminal wins, 43 terminal losses, 56 nonterminal action-type mismatches.
  - Even the deterministic-loss chunk contains terminal-winning siblings, including late states where one or two root actions recovered wins from otherwise loss-heavy surfaces.
- Action-only ranked sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v389_grixis_action_root8_model_rank64`
  - Ranked 64 `ACTIVATE_ABILITY_OR_SPELL` checkpoints across chunks 7 and 14, up to 8 root choices per checkpoint.
  - Result: 512 / 512 terminal rows, 265 wins and 247 losses, with zero action-type mismatches.
  - Chunk split: chunk 14 was 227 wins / 29 losses; chunk 7 was 38 wins / 218 losses.
- Denser action-only ranked sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v390_grixis_action_root16_model_rank64`
  - Same ranked 64 action-only checkpoint set, 16 attempts per checkpoint, common continuation seeds.
  - Result: 1,024 / 1,024 terminal rows, 527 wins and 497 losses, with zero action-type mismatches.
  - Chunk split: chunk 14 was 452 wins / 60 losses; chunk 7 was 75 wins / 437 losses.
- Value-target export: `local-training/local_pbt/terminal_line_value_targets/v390_grixis_action_root16_model_rank64_include_pass`
  - Gate: `min_actions=2`, `min_attempts_per_action=2`, terminal rate 1.0, common samples required, min value delta 0.10.
  - Primary export keeps pass-best rows trainable when explicitly requested, while preserving pass-best quality flags for diagnostics.
  - Exported 18 admitted value examples and 18 serialized advantage-value `TrainingData` records.
- Diagnostic soft-pass export: `local-training/local_pbt/terminal_line_value_targets/v390_grixis_action_root16_model_rank64_softpass_diagnostic`
  - Same inputs and thresholds, but default pass-best exclusion.
  - Admitted 16 value examples, showing that only two v390 trainable rows depend on allowing pass-best.
- Full action-only ranked sweep: `local-training/local_pbt/live_checkpoint_branch_miner/v391_grixis_action_root16_model_rank110`
  - All 110 eligible `ACTIVATE_ABILITY_OR_SPELL` checkpoints from the v383 live-checkpoint capture, 16 attempts per checkpoint, common continuation seeds.
  - Result: 1,760 / 1,760 terminal rows, 1,116 wins and 644 losses, with zero action-type mismatches.
  - Chunk split: chunk 14 was 925 wins / 67 losses; chunk 7 was 191 wins / 577 losses.
  - Snapshot groups: 22 mixed-outcome groups, 57 all-win groups, and 31 all-loss groups.
- Full action-only value-target export: `local-training/local_pbt/terminal_line_value_targets/v391_grixis_action_root16_model_rank110_include_pass`
  - Same gate as v390, with explicit pass-best inclusion preserved for the primary import.
  - Exported 22 admitted value examples and 22 serialized advantage-value `TrainingData` records.
- Full action-only soft-pass diagnostic export: `local-training/local_pbt/terminal_line_value_targets/v391_grixis_action_root16_model_rank110_softpass_diagnostic`
  - Same inputs and thresholds, but default pass-best exclusion.
  - Admitted 17 value examples, so five v391 rows depend on explicitly allowing pass-best.

Code checkpoint:

- `LiveCheckpointBranchMiner` terminal-line mode now honors `--post-branch-autopilot`; with `false`, the forced root is applied and later decisions use the normal model path.
- Added `scripts/run_live_checkpoint_branch_miner.py`, which loads an eval manifest, starts the same deterministic shared GPU service/model snapshot, and runs `LiveCheckpointBranchMiner` under that environment.
- Corrected `scripts/mtgrl/export_terminal_line_value_targets.py --include-suspect-pass-best` so explicitly included pass-best rows are trainable, not merely kept in the CSV as non-trainable suspect rows.

Conclusion:

- Branchable checkpoints can reproduce at least one known winning combo line from a captured in-memory game state when continuation uses the trained model through the correct shared GPU service.
- The v383 zero-win autopilot slice is reclassified as a harness diagnostic, not evidence against the checkpoint or model.
- Shared-GPU model-continuation mining now finds outcome-only branch contrast on both the deterministic winning chunk 14 and deterministic losing chunk 7.
- Action-only mining removes the current pending-choice reentry noise and yields clean terminal-only value targets without combo-specific labels; v391 scales this across all eligible v383 action checkpoints.
- The next unit is to combine v391 with the existing v345/v349 terminal-line corpus for the next small Q-head import candidate, then evaluate it under deterministic paired controls before considering any HPC scale-up.

## v392 Combined v345/v349/v391 Q-Head Candidate

Purpose:

- Test whether the new v391 Grixis action-only terminal-line evidence improves the contained signed candidate-Q path without adding combo-specific reward shaping.
- Keep the import narrow: fresh clone of baseline `Pauper-Spy-Combo-Value`, Q-head-only training, signed branch-return targets, no direct policy overwrite.

Artifacts:

- Combined import data: `local-training/local_pbt/terminal_line_value_targets/v392_combined_v345_v349_v391_signed_advantage_import_data`
  - v345: 255 examples.
  - v349: 134 examples.
  - v391: 22 examples.
  - Total: 411 serialized advantage-value `TrainingData` examples.
- Candidate profile: `Pauper-Spy-Combo-Value-TerminalLine-v392-QOnlySignedAdvantageGrixisAction`
- Import run: `local-training/local_pbt/action_counterfactual/20260524_v392_combined_qonly_signed_advantage_import`
  - 411 examples, 4 epochs, 1,644 Q-head train passes, `branch_return_targets=true`.
- Offline score probes on the same 411-example corpus:
  - Blend 0.00: 127 / 411 top-1, average rank 2.3698.
  - Blend 0.50: 163 / 411 top-1, average rank 2.2798.
  - Blend 1.00: 189 / 411 top-1, average rank 2.1946.

Deterministic paired eval:

- Active-three candidate artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v392_active3_g8_detmode_nolog`
  - Overall: 14 / 24.
  - Spy mirror: 7 / 8.
  - Jund Wildfire: 5 / 8.
  - Mono Red Rally: 2 / 8.
- Paired comparison on the same seed set:
  - v350: 13 / 24.
  - Baseline: 11 / 24.
  - v392 gains Jund chunk 7 and Spy mirror chunk 4 versus v350, but loses Mono Red Rally chunk 6.
- Grixis hard-gate candidate artifact: `local-training/local_pbt/cp7_eval_sweeps/20260524_v392_grixis_g16_detmode_nolog`
  - Result: 6 / 15 counted games, with chunk 1 invalid at 0 / 0 after game-thread timeout.
  - Even if the invalid chunk were rerun as a win, the best case would be 7 / 16.
  - v350 and baseline both scored 9 / 16 on the matching deterministic Grixis seed set.
  - Regressions versus both v350 and baseline occurred on Grixis chunks 3 and 11; v392 still preserves the chunk 14 gain over baseline.

Conclusion:

- v392 improves the offline candidate-Q ranking signal and slightly improves the active-three control, but it fails the Grixis Affinity hard gate.
- This is not an HPC or promotion candidate.
- The next thesis-aligned unit is to log and inspect the v392 Grixis regressions, especially chunks 3 and 11, then decide whether the issue is Q-consumer overreach, noisy/suspect v391 pass-best rows, or missing Affinity-pressure coverage in the outcome-derived corpus.

## v393 London-Bottom Q-Blend Gate

Purpose:

- Test whether the v392 Grixis hard-gate regression came from applying terminal-line candidate-Q evidence to London-bottom card ordering.
- Keep the change generic: no card-specific rule, no combo-state reward, and no explicit "good Spy hand" heuristic.

Diagnosis:

- Focused v392 Grixis chunk 3 with `CANDIDATE_Q_BLEND_HEADS=action,target,card_select` still lost.
- The bad path came from London-bottom card ordering: Q blending on the `card_select` head also affected `LONDON_MULLIGAN`, so terminal-line action evidence could perturb pregame bottoming.
- Splitting London into a separate model policy head fixed chunk 3 but changed v350 chunk 1 behavior, so that approach was too broad.

Code checkpoint:

- London-bottom now keeps the historical `card_select` policy scorer but sends a composite head id, `card_select|q=london_mulligan`.
- `mtg_transformer.py` parses composite head ids into a policy head and a separate candidate-Q blend gate head.
- The shared GPU PyTorch path honors the separate gate; the ONNX/TensorRT fast path is bypassed for composite head ids to avoid stale exported-head routing.

Validation:

- Compile checks passed:
  - `python -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mtg_transformer.py Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py`
  - `python %USERPROFILE%/.codex/skills/mage-research-agent/scripts/airl_maven.py compile`
- Focused fixed chunk: `local-training/local_pbt/cp7_eval_sweeps/20260524_v393_grixis_chunk003_detmode_nolog_composite_qhead`
  - Result: 1 / 1 win on Grixis chunk 3.
- Completed pre-composite diagnostic gate: `local-training/local_pbt/cp7_eval_sweeps/20260524_v393_london_headsplit_grixis_g16_detmode_nolog`
  - Result: 6 / 16, with chunk 3 fixed but chunks 1, 11, and 14 still losing.
- Composite-head disagreement slice: `local-training/local_pbt/cp7_eval_sweeps/20260524_v393_grixis_disagreement_chunks011114_detmode_nolog_composite_qhead`
  - Chunk 1: 0 / 1.
  - Chunk 11: 0 / 1.
  - Chunk 14: 1 / 1.
  - Expected Grixis hard-gate count after this fix is 7 / 16 if all other chunks remain unchanged.

Conclusion:

- The London-bottom Q-blend contamination is fixed without changing the policy scorer used for London-bottom decisions.
- v393 is still not a promotion or HPC-scale candidate because the Grixis hard gate remains below the 9 / 16 v350 and baseline controls.
- The next research unit is to inspect chunks 1 and 11 as the remaining outcome-derived blockers, likely by comparing candidate-Q deltas against baseline/v350 choices and either gating low-confidence Q application or adding more terminal evidence for Affinity-pressure states.

## v438 Deterministic CP7 Eval Stability Gate

Purpose:

- Repair the local deterministic-eval control before comparing v392/v393 against baseline on the remaining Grixis chunks.
- Keep the fix generic: deterministic ordering, stable replay/search choices, and fresh Maven exec classpaths only. No combo-specific reward, no card-specific action override, and no hard-coded "pass cannot be best" rule.

Diagnosis:

- Earlier paired full-log repeats still diverged from the same visible and hidden state at opponent decision D39 on Grixis chunk 1: one run chose `Thoughtcast`, another chose `Silverbluff Bridge`.
- Increasing `AI_DETERMINISTIC_MAX_NODES` to 20000 did not solve the split and made one repeat time out.
- The first actionable root cause was stale Maven exec classpaths: `run_cp7_eval_sweep.py --skip-compile` could run `exec:java` against an older AI.MA dependency even after a separate compile had passed. The CP7 root trace hook only appeared when the job used compile and exec in the same Maven invocation.

Code checkpoint:

- Deterministic CP7 eval now:
  - disables random equal-score root tie breaks;
  - disables carried-forward planned-action reuse;
  - uses stable action, source-copy, mana-producer, target, and combat ordering;
  - isolates CP7 simulation RNG for deterministic search;
  - exposes guarded `AI_DETERMINISTIC_ROOT_TRACE` / `--deterministic-root-trace` diagnostics;
  - automatically uses compile+exec in the same Maven invocation for `--deterministic-eval` jobs.
- Related engine option ordering fixes use insertion-stable collections and deterministic cost-target/source selection in the replay-relevant paths.

Validation:

- Compile/script checks passed:
  - `python -m py_compile scripts\run_cp7_eval_sweep.py`
  - `git diff --check`
  - `python %USERPROFILE%\.codex\skills\mage-research-agent\scripts\airl_maven.py compile`
- Root-trace classpath proof:
  - `20260524_v436_baseline_chunk001_fulllog_roottrace_compileexec_a`
  - `20260524_v437_baseline_chunk001_fulllog_roottrace_compileexec_b`
  - Both completed `0 / 1`, emitted CP7 root score rows, and produced 198 normalized decision rows with no semantic diff.
- Normal no-trace deterministic proof:
  - `20260524_v438_baseline_chunk001_fulllog_autocompileexec_a`
  - `20260524_v439_baseline_chunk001_fulllog_autocompileexec_b`
  - Both completed `0 / 1`, produced 198 normalized decision rows, and had no first diff, including RNG/probability fields.
- Root-trace leakage hotfix:
  - The initial diagnostic flag accidentally inherited the generic `TORCH_DETERMINISTIC_EVAL` fallback and emitted `CP7_ROOT_SCORE_JSON` rows for normal deterministic eval jobs.
  - `AI_DETERMINISTIC_ROOT_TRACE` now requires its own explicit env/property.
  - `20260524_v444_baseline_chunk003_no_roottrace_leak_sanity` completed `1 / 1`; the job log contained no `CP7_ROOT_SCORE_JSON` rows and stayed at normal diagnostic size.

Conclusion:

- The chunk 1 deterministic-control blocker is resolved for the normal deterministic eval command shape.
- The stale exec-classpath failure mode is now guarded by script behavior, so future deterministic paired gates should exercise the just-compiled AI.MA code.
- Next unit: run paired deterministic no-log checks for v392/v393 versus baseline on the remaining Grixis blocker chunks, starting with chunks 1 and 11, before considering any HPC scale-up.

## v440-v446 Refreshed Grixis Hard Gate

Purpose:

- Re-run the v392 Q-head candidate against baseline after the deterministic CP7 eval repairs, using the same Grixis Affinity chunk seeds and no game-log overhead.

Runs:

| Run ID | Profile | Scope | Result |
| --- | --- | --- | --- |
| `20260524_v440_v392_grixis_chunks0111_deterministic_eval_blend0_autocompileexec` | v392 candidate | chunks 1, 11 | `1 / 2`: chunk 1 lost, chunk 11 won. |
| `20260524_v441_baseline_grixis_chunks0111_deterministic_eval_autocompileexec` | baseline | chunks 1, 11 | `1 / 2`: chunk 1 lost, chunk 11 won. |
| `20260524_v442_v392_grixis_g16_deterministic_eval_blend0_autocompileexec` | v392 candidate | all 16 chunks | `7 / 15` counted; chunk 8 invalid `0 / 0`. |
| `20260524_v443_baseline_grixis_g16_deterministic_eval_autocompileexec` | baseline | all 16 chunks | `7 / 15` counted; chunk 8 invalid `0 / 0`. |
| `20260524_v445_v392_grixis_chunk008_no_roottrace_rerun` | v392 candidate | chunk 8 rerun after root-trace hotfix | Repeated invalid `0 / 0`. |
| `20260524_v446_baseline_grixis_chunk008_no_roottrace_rerun` | baseline | chunk 8 rerun after root-trace hotfix | Repeated invalid `0 / 0`. |

Chunk-level comparison:

- Candidate and baseline have identical outcomes on every completed chunk: losses on 1, 2, 4, 5, 6, 12, 15, 16; wins on 3, 7, 9, 10, 11, 13, 14.
- Chunk 8 is a shared deterministic game-thread timeout for both profiles, including after disabling leaked root-trace output. The clean rerun logs are small and show the same `Game thread did not finish after timeout and forced end` failure.

Conclusion:

- v392/v393 no longer shows a Grixis regression under the repaired deterministic eval, but it also does not beat baseline. It is a neutral candidate, not a promotion or HPC-scale candidate.
- The immediate thesis path should move back to outcome-derived evidence quality: mine or generate new terminal-line evidence that actually changes policy/Q choices in baseline-losing chunks, rather than scaling v392 as-is.
- Chunk 8 is a separate deterministic evaluation-health issue. Because it is shared by candidate and baseline, it should not drive model selection, but it should be tracked if Grixis hard-gate accounting requires all 16 chunks.

## v447-v450 Live Checkpoint Capture Repair

Purpose:

- Collect branchable live checkpoints from baseline-losing Grixis chunks without paying compact game-log overhead.
- Keep the path thesis-clean: no combo labels, no intermediate combo reward, only terminal-loss states and later branch outcomes.

Diagnosis:

- `20260524_v447_baseline_grixis_loss_chunks_livecheckpoints_autocompileexec` was launched over chunks 1, 2, 4, 5, 6, 12, 15, and 16 with `--live-checkpoints` and `--replay-metadata`, but without `--eval-game-logging`.
- Chunks 1, 2, 4, and 5 all reached terminal losses, yet no `live_checkpoints` directory was produced.
- Root cause: `LiveCheckpointRecorder.maybeCapture(...)` was called inside `ComputerPlayerRL.logReplayDecision(...)` only after resolving an enabled `GameLogger`, so no-log capture runs silently returned before snapshot serialization.

Code checkpoint:

- `ComputerPlayerRL.logReplayDecision(...)` now computes replay metadata and calls `LiveCheckpointRecorder` even when text game logging is disabled; `REPLAY_DECISION_JSON` still only writes when `GameLogger` is enabled.
- `scripts/run_cp7_eval_sweep.py --live-checkpoints` now implies replay metadata, preventing silent zero-checkpoint runs where the replay decision hook is inactive.
- `scripts/run_live_checkpoint_branch_miner.py` now injects `--out <run_dir>` unless explicitly overridden, so wrapper run IDs own the Java miner outputs instead of leaving probes in a timestamped default directory.

Validation:

| Run ID | Scope | Result |
| --- | --- | --- |
| `20260524_v448_live_checkpoint_no_gamelog_smoke` | Baseline Grixis chunk 4, deterministic no-game-log live checkpoint smoke. | Terminal loss `0 / 1`; manifest has `eval_game_logging=false`, `replay_metadata=true`, `live_checkpoints=true`; wrote 24 captured snapshots and 24 manifest rows with no errors. |
| `v450_v448_no_gamelog_reentry_smoke_outfix` | Branch-miner reentry over the first two v448 snapshots. | Wrote `live_checkpoint_branch_probe.csv` inside the requested run dir; both rows classified `reentry_matched` with matching candidate and state hashes. |

Conclusion:

- No-log live checkpoint capture works again, and snapshots from the repaired path are branchable.
- The next unit is to relaunch the baseline-losing Grixis capture across chunks 1, 2, 4, 5, 6, 12, 15, and 16, then mine terminal/value evidence from those snapshots instead of extending the neutral v392 candidate.

## v451-v461 Baseline-Loss Corpus and Sharded Miner Repair

Purpose:

- Build a fresh outcome-only branch corpus from baseline-losing deterministic Grixis chunks, then mine it with model continuation.
- Keep this as evidence generation only: no explicit combo heuristics, no training admission until terminal rows are summarized and gated.

Artifacts and results:

| Run ID | Scope | Result |
| --- | --- | --- |
| `20260524_v451_baseline_grixis_loss_chunks_livecheckpoints_nolog_autocompileexec` | Baseline Grixis loss chunks 1, 2, 4, 5, 6, 12, 15, and 16, deterministic no-log live checkpoints. | All 8 chunks lost (`0 / 8`); captured 552 snapshots with zero serialization errors. Action-root snapshots by chunk: 70, 73, 69, 71, 75, 77, 0, 77. |
| `v452_v451_grixis_loss_action_root16_model_rank64` / `v454_v451_grixis_loss_action_root16_model_rank64_compile` | Initial sharded ranked terminal-line attempts. | Both exited 0 but selected 0 rows; not evidence. |
| `v453_v451_path_reentry_debug`, `v455_v451_ranked_reentry_debug`, `v456_v451_ranked_shard0_reentry_debug` | Direct wrapper debug probes. | Confirmed v451 snapshots load, ranked selection works, and sharding works when launched through the manifest-aware compile+exec wrapper. |
| `v460_v451_sharded_path_loaderror_debug` | Sharded path-mode load-error probe. | Exposed `InvalidClassException: mage.player.ai.ComputerPlayer6`; ranked mode had hidden the load errors by dropping failed snapshots. |
| `v461_v451_sharded_compileexec_smoke` | Patched sharded runner smoke, 2 shards, 4 ranked snapshots, 1 root attempt each. | Selected 4 snapshots and wrote 4 terminal-line rows: 2 terminal wins, 2 terminal losses. |

Code checkpoint:

- `scripts/mtgrl/run_value_tree_shards.py` now:
  - auto-detects the eval manifest from `<eval-run>/live_checkpoints`;
  - injects `MODEL_PROFILE`, `RL_ARTIFACTS_ROOT`, and deterministic eval env into shard JVMs;
  - starts a shared GPU inference service for manifest-backed snapshots;
  - includes `compile exec:java` in each shard Maven invocation by default, avoiding stale reactor classpaths during snapshot deserialization;
  - retains opt-outs for compile and shared-GPU behavior for explicit diagnostic use.

Conclusion:

- v451 is the first repaired baseline-losing Grixis checkpoint corpus after the deterministic control cleanup.
- The sharded miner failure was harness infrastructure, not lack of branchable snapshots.
- The next unit is a full v451 ranked action-root terminal-line pass: 64 ranked snapshots, 16 root attempts/checkpoint, common continuation seeds, model continuation, 4 shards.

## v462 Baseline-Loss Terminal-Line Mining

Purpose:

- Run the first full terminal-line mining pass over the repaired v451 baseline-losing Grixis checkpoint corpus.
- Keep the pass outcome-only: no Spy-specific labels, no combo-ready detector, no intermediate reward shaping.

Run:

| Run ID | Scope | Result |
| --- | --- | --- |
| `v462_v451_grixis_loss_action_root16_model_rank64` | v451 ranked `ACTIVATE_ABILITY_OR_SPELL` roots, 64 snapshots, 4 shards, 16 attempts/checkpoint, common continuation seeds, model continuation, `post_branch_autopilot=false`, shared GPU service. | Selected 64 snapshots and wrote 547 terminal-line rows: 33 terminal wins, 514 terminal losses, terminal rate `1.0`. |

Outcome summary:

- Checkpoint groups: 64.
- Mixed win/loss groups: 11.
- All-win groups: 22.
- All-loss groups: 31.
- `summarize_terminal_line_search.py` reported `spy_rows=0`, `full_combo_wins=0`, `dread_return_rows=1`, and `max_combo_score=3`; this pass found terminal-winning siblings, but not the true Spy combo line.
- Representative mixed rows included:
  - chunk 2, D063: `Cast Tinder Wall` won while `Cast Elves of Deep Shadow`, `Forestcycling`, and `Pass` lost.
  - chunk 6, D072: `Cast Winding Way` won while `Flashback sacrifice three creatures` and `Pass` lost.
  - chunk 16, D007/D006: `Cast Winding Way` won in both nearby checkpoint groups.
  - chunk 5, D065: `Forestcycling` won while available siblings lost.

Strict export:

| Export | Gate | Result |
| --- | --- | --- |
| `v462_v451_grixis_loss_action_root16_model_rank64_softpass_diagnostic` | Common-seed value target gate, suspect pass-best rows excluded. | `0` admitted examples; rejection counts were `actions_lt_min=22`, `eligible_actions_lt_min=33`, `suspect_pass_best=16`, `value_delta_below_threshold=31`. |
| `v462_v451_grixis_loss_action_root16_model_rank64_include_pass` | Same gate, suspect pass-best rows included. | `0` admitted examples; rejection counts were `actions_lt_min=22`, `eligible_actions_lt_min=33`, `value_delta_below_threshold=31`. |

Interpretation:

- v462 proves that fresh baseline-losing Grixis checkpoints contain terminal-winning sibling continuations under model-driven outcome-only search.
- The zero-example strict export is a sampling-density issue, not a snapshot/reentry failure: `line_stop_on_win=true` and sparse one-row action samples did not provide enough paired common-seed evidence per root action.
- Pass-best mixed groups remain diagnostic until they clear pass-specific common-sample, value, and delta gates.

Next unit:

- Rerun only the 11 v462 mixed checkpoint groups with `line_stop_on_win=false`, common continuation seeds, and denser attempts per root action.
- Admit training/value evidence only if the repeated pass clears the same common-seed value-target exporter without relaxing the outcome-only constraints.

## v463 Dense Mixed Terminal-Line Rerun

Purpose:

- Test whether the v462 zero-export result was caused by sparse one-row action samples and early stop-on-win behavior.
- Rerun only the 11 v462 mixed checkpoint groups with denser outcome-only sampling.

Run:

| Run ID | Scope | Result |
| --- | --- | --- |
| `v463_v462_mixed_commonseed_dense` | 11 mixed v462 checkpoints, 4 shards, `line_attempts=32`, common continuation seeds, `line_stop_on_win=false`, model continuation, shared GPU service via the v451 manifest. | Selected 11 snapshots and wrote 352 terminal-line rows: 126 terminal wins, 226 terminal losses, terminal rate `1.0`. |

Terminal summary:

- Win rate across sampled rows: `126 / 352 = 35.80%`.
- `summarize_terminal_line_search.py` reported `spy_rows=0`, `full_combo_wins=0`, `dread_return_rows=5`, and `max_combo_score=3`.
- The repeated rows confirm the same broad pattern as v462: the brancher can find terminal-winning siblings in baseline-losing checkpoints, but this slice still is not discovering the full Spy combo finish.

Strict export:

| Export | Gate | Result |
| --- | --- | --- |
| `v463_v462_mixed_commonseed_dense_softpass_diagnostic` | Common-seed value target gate, suspect pass-best rows excluded. | `7` admitted examples, `4` rejected suspect pass-best groups; classifications were 1 strong delta and 6 moderate deltas. |
| `v463_v462_mixed_commonseed_dense_include_pass` | Same gate, suspect pass-best rows included for diagnosis. | `11` admitted examples; classifications were 5 strong deltas and 6 moderate deltas. |

Clean soft-pass top targets:

- chunk 2 D063: `Cast Tinder Wall` over `{T}: Add {G}.`, value delta `1.0`, weight `0.866`.
- chunk 5 D065: `Forestcycling` and `Cast Gatecreeper Vine` both reached value `1.0` in the soft target distribution, over `{T}: Add {G}.`.
- chunk 6 D039: `Cast Roost Seek` over `{T}: Add {G} for each creature you control with defender.`, value delta `1.0`.
- chunk 6 D072 and chunk 16 D006/D007: `Cast Winding Way` over mana-only actions, value delta `1.0`.
- chunk 12 D046: `Cast Overgrown Battlement` over `Cast Lead the Stampede`, value delta `0.75`.

Interpretation:

- v463 converts v462's mixed one-row diagnostics into strict, trainable, terminal-derived value targets without adding combo labels or intermediate rewards.
- The pass-best rows remain useful diagnostics but are not admitted to the clean soft-pass set because all 4 were flagged `suspect_pass_best`.
- Seven examples is too small to train or promote. The next unit is to scale the same dense no-stop sampling from the 11 mixed proof back to the full 64 ranked v451 snapshots, then export the common-seed target set before considering any model update.

## v464 Full Dense Ranked Terminal-Line Rerun

Purpose:

- Scale the v463 dense/no-stop proof from the 11 known mixed groups back to the full 64 ranked v451 `ACTIVATE_ABILITY_OR_SPELL` snapshot set.
- Measure whether the clean common-seed value-target count grows enough to justify a model update.

Run:

| Run ID | Scope | Result |
| --- | --- | --- |
| `v464_v451_grixis_loss_action_root16_model_rank64_dense_nostop` | v451 ranked `ACTIVATE_ABILITY_OR_SPELL` roots, 64 snapshots, 4 shards, `line_attempts=32`, common continuation seeds, `line_stop_on_win=false`, model continuation, shared GPU service. | Selected 64 snapshots and wrote 2,048 terminal-line rows: 764 terminal wins, 1,284 terminal losses, terminal rate `1.0`. |

Terminal summary:

- Win rate across sampled rows: `764 / 2048 = 37.30%`.
- `summarize_terminal_line_search.py` reported `spy_rows=0`, `full_combo_wins=0`, `dread_return_rows=5`, and `max_combo_score=3`.
- Root actions with many samples included `Pass`, mana abilities, `Forestcycling`, `Cast Roost Seek`, `Cast Tinder Wall`, `Cast Overgrown Battlement`, and `Cast Winding Way`.

Strict export:

| Export | Gate | Result |
| --- | --- | --- |
| `v464_v451_grixis_loss_action_root16_model_rank64_dense_nostop_softpass_diagnostic` | Common-seed value target gate, suspect pass-best rows excluded. | `13` admitted examples, `51` rejected groups; classifications were 4 strong deltas and 9 moderate deltas. |
| `v464_v451_grixis_loss_action_root16_model_rank64_dense_nostop_include_pass` | Same gate, suspect pass-best rows included for diagnosis. | `18` admitted examples, `46` rejected groups; classifications were 9 strong deltas and 9 moderate deltas. |

Clean soft-pass target examples:

- chunk 2 D063: `Cast Tinder Wall` over `{T}: Add {G}.`, value delta `1.0`.
- chunk 4 D030: `Cast Sagu Wildling` over `Cast Roost Seek`, value delta `1.0`.
- chunk 4 D045: `Cast Winding Way` over `{T}: Add {B}. {this} deals 1 damage to you.`, value delta `1.0`.
- chunk 4 D046: `Cast Troll of Khazad-dum` over `{T}, Tap an untapped creature you control: Add one mana of any color.`, value delta `1.0`.
- chunk 5 D065 and chunk 6/16 Winding Way/Roost Seek decisions repeated the v463 signal.

Interpretation:

- The dense no-stop method scales mechanically and produces more strict outcome-derived value targets than the sparse v462 pass.
- The yield is still too low for a fresh model update: `13 / 64` clean groups, with 21 pass-best groups held out as suspect and 46 groups rejected for low value delta.
- The next unit is a larger dense ACTIVATE pass over lower-ranked v451 snapshots by raising `ranked_max_per_game`; target is roughly 50 clean soft-pass examples before training or candidate-Q import.

## v465 Rank-256 Dense ACTIVATE Terminal-Line Expansion

Purpose:

- Increase terminal-line evidence breadth from the v464 top-64 ranked ACTIVATE slice toward a training-scale clean value-target set.
- Keep the exact same outcome-only settings: terminal win/loss value, common continuation seeds, no stop-on-win, no combo labels, and no intermediate reward shaping.

Run:

| Run ID | Scope | Result |
| --- | --- | --- |
| `v465_v451_grixis_loss_action_root16_model_rank256_dense_nostop` | v451 ranked `ACTIVATE_ABILITY_OR_SPELL` roots, `ranked_max_per_game=40`, 256 snapshots, 4 shards, `line_attempts=32`, common continuation seeds, `line_stop_on_win=false`, model continuation, shared GPU service. | Selected 256 snapshots and wrote 8,192 terminal-line rows: 3,719 terminal wins, 4,473 terminal losses, terminal rate `1.0`. |

Terminal summary:

- Win rate across sampled rows: `3719 / 8192 = 45.40%`.
- `summarize_terminal_line_search.py` reported `spy_rows=0`, `full_combo_wins=0`, `dread_return_rows=5`, and `max_combo_score=3`.
- High-volume root actions included `Pass`, mana abilities, `Forestcycling`, `Return a Forest`, `Cast Roost Seek`, `Cast Land Grant`, `Cast Tinder Wall`, and `Cast Winding Way`.

Strict export:

| Export | Gate | Result |
| --- | --- | --- |
| `v465_v451_grixis_loss_action_root16_model_rank256_dense_nostop_softpass_diagnostic` | Common-seed value target gate, suspect pass-best rows excluded. | `39` admitted examples, `217` rejected groups; classifications were 24 strong deltas, 14 moderate deltas, and 1 weak delta. |
| `v465_v451_grixis_loss_action_root16_model_rank256_dense_nostop_include_pass` | Same gate, suspect pass-best rows included for diagnosis. | `52` admitted examples, `204` rejected groups; classifications were 33 strong deltas, 18 moderate deltas, and 1 weak delta. |

Interpretation:

- The clean strict target count scaled from `13 / 64` in v464 to `39 / 256` in v465.
- This is close to the rough 50-example clean threshold, but the clean set still falls short; the pass-including diagnostic set reaches 52 but is not acceptable as the main training source because 156 groups are still flagged `suspect_pass_best`.
- The next unit is one more broader dense ACTIVATE run over a larger ranked slice, then export the clean soft-pass set before deciding whether to train a small candidate-Q/value consumer.

## v468 Online Terminal-Mining Harness Smoke

Purpose:

- Validate the online-distribution loop: let the current model play a real deterministic eval game, capture live branchable checkpoints from the states it actually reaches, then terminal-line mine those checkpoints.
- Keep the teacher signal terminal-only: no combo milestone labels, no card-name rewards, and no hand-authored combo-ready detector.

Code checkpoints:

| Commit | Change |
| --- | --- |
| `8d7ae098de` | Added `scripts/mtgrl/run_online_terminal_mining_loop.py`, which chains live checkpoint eval, terminal-line mining, summary, and value-target export into one manifest-backed run. |
| `9c4cac9b42` | Added a fail-fast guard against skipping both miner compile stages after online eval. The v467 fast smoke showed this can produce stale-classpath `InvalidClassException` load errors and zero ranked selections. |

Run:

| Run ID | Scope | Result |
| --- | --- | --- |
| `v468_online_terminal_mining_smoke_compiled_grixis_g1_rank2` | One deterministic online Grixis eval game, live checkpoint cap 16, ranked ACTIVATE mining over 2 snapshots, 1 shard, 2 root attempts/checkpoint, common continuation seeds, no stop-on-win. | Eval completed and wrote live checkpoints. Miner selected 2 ranked snapshots and wrote 4 terminal-line rows, all terminal losses. |

Target export:

| Export | Result |
| --- | --- |
| Soft-pass strict export | `0` admitted examples; `2` checkpoint groups rejected with `eligible_actions_lt_min`. |
| Include-pass diagnostic export | `0` admitted examples; same rejection shape. |

Interpretation:

- The online-distribution harness works end to end when eval and miner classpaths are compiled consistently.
- The tiny smoke is intentionally too small to produce value targets; it only proves that live online checkpoints can flow into terminal-outcome mining and export without replay reconstruction.
- This is directly useful for local and HPC scaling: increase games, snapshots, shards, and attempts, and the same artifact structure will produce online-distribution terminal-derived targets.

## v470 Closed Online Mining Training Loop Harness

Purpose:

- Wire the promising online-distribution evidence path into a full iteration loop instead of stopping at target export.
- Keep the learning signal thesis-clean: terminal win/loss outcomes only, no combo-state labels, no card-specific reward, and no intermediate combo milestones.

Implementation:

- Added `scripts/mtgrl/run_online_mining_training_loop.py`.
- The driver runs: online eval with live checkpoint capture -> terminal-line mining -> clean soft-pass value-target export -> `TerminalLineValueTargetTrainingDataExporter` -> cloned-profile candidate-Q-only training -> deterministic post-train eval.
- Training uses `advantage-values` by default, `-CandidateQOnly`, and `-BranchReturnTargets`, matching the signed local sibling-ranking path that previously gave the cleanest offline Q signal.
- Candidate profiles are cloned from the current play profile before import training; the loop never overwrites the source profile.
- Added `--existing-online-run-dir` and `--wait-existing-online-run-sec` so an already-running online-mining artifact, such as v469, can be consumed as soon as its value-target summary exists.
- Added `--registry` pass-through to `run_online_terminal_mining_loop.py` so generated one-profile registries can run candidate profiles that are not in the default PBT registry.

Validation:

- `python -m py_compile scripts\mtgrl\run_online_terminal_mining_loop.py scripts\mtgrl\run_online_mining_training_loop.py`
- `python %USERPROFILE%\.codex\skills\mage-research-agent\scripts\airl_maven.py compile`
- Dry-run command construction was checked with a one-game, one-snapshot plan.

Current execution state:

- v469 is still active in the mining stage while this harness is committed.
- Once v469 writes `terminal_line_value_target_summary.json`, the next command is to run the closed-loop driver with `--existing-online-run-dir local-training/local_pbt/online_terminal_mining/v469_online_terminal_mining_grixis_g4_rank32_a16`.

## v469-v470 First Closed-Loop Completion

Purpose:

- Complete the first full online-distribution loop: model play -> live checkpoints -> terminal-line mining -> value targets -> TrainingData -> Q-head candidate -> exact-seed eval.

Artifacts:

| Artifact | Path |
| --- | --- |
| Online mining run | `local-training/local_pbt/online_terminal_mining/v469_online_terminal_mining_grixis_g4_rank32_a16` |
| Closed-loop run | `local-training/local_pbt/online_mining_training_loop/v470_closed_loop_from_v469` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v470` |

v469 result:

- Eval source: `Pauper-Spy-Combo-Value` vs Grixis Affinity, skill 7, deterministic exact-seed mode, 4 games.
- Online eval result before mining: `2 / 4`.
- Live-checkpoint mining selected 32 ranked `ACTIVATE_ABILITY_OR_SPELL` snapshots.
- Terminal-line rows: `512`.
- Terminal outcomes: `308` wins, `204` losses.
- Strict soft-pass value targets: `16 / 32` checkpoint groups admitted.
- Include-pass diagnostic value targets: `17 / 32` admitted.
- The only pass-best training exclusion was flagged as `suspect_pass_best`; the clean set excluded it.

v470 bridge/training result:

- Advantage-value `TrainingData` export: `16 / 16` rows reentered with matching candidate hash and state hash.
- All 16 rows captured TrainingData from checkpoint reentry.
- Candidate training: cloned from `Pauper-Spy-Combo-Value`, Q-head only, `-BranchReturnTargets`, 4 epochs, 64 train-pass samples.
- Training run: `local-training/local_pbt/action_counterfactual/v470_closed_loop_from_v469_cycle01_train_Pauper-Spy-Combo-Value-OnlineLoop-v470`.

Exact-seed Grixis eval:

| Profile | Result | Notes |
| --- | ---: | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v470` | `1 / 4` | Won chunk 4. |
| `Pauper-Spy-Combo-Value` source comparator | `0 / 4` | Same matchup seed base and chunk split. |

Target-row interpretation:

- The admitted actions were generic terminal-outcome discoveries, not hand-authored combo milestones: examples include `Cast Land Grant`, `Cast Saruli Caretaker`, `Cast Overgrown Battlement`, `Cast Balustrade Spy`, `Cast Lead the Stampede`, `Cast Wall of Roots`, `Cast Winding Way`, and mana/Forestcycling actions.
- `14 / 16` admitted rows had value delta `1.0`; `2 / 16` had value delta `0.5`.
- The target rows are still small and mostly moderate-delta by the current classifier, so this is mechanical proof and a weak positive eval signal, not promotion evidence.

Interpretation:

- The full loop works mechanically end to end on online-reached states.
- A 16-example Q-only branch-return update produced a candidate that beat the source comparator by one exact-seed Grixis chunk, but the sample is far too small to conclude policy improvement.
- This supports the path, not the candidate. The next useful unit is a broader online loop, likely 8-16 Grixis games and/or more snapshots/attempts, then a paired exact-seed eval with enough games to see whether the one-chunk gain persists.

## v471-v472 Online-Distribution Iteration From v470

Purpose:

- Continue the thesis-clean closed loop from the generated v470 model rather than returning to the base profile.
- Test whether online-reached checkpoint mining can produce another terminal-only branch-return update without combo-state labels, card-specific rewards, or hand-authored setup detectors.

Implementation delta:

- Commit `7657c204ce` added `--play-profile` to `scripts/mtgrl/run_online_mining_training_loop.py`.
- This lets a generated model such as `Pauper-Spy-Combo-Value-OnlineLoop-v470` play cycle 1 while reusing the base registry deck/env entry from `Pauper-Spy-Combo-Value`.

Artifacts:

| Artifact | Path |
| --- | --- |
| Online loop run | `local-training/local_pbt/online_mining_training_loop/v471_online_loop_from_v470_grixis_g8_r96_a16` |
| Resume/train/eval run | `local-training/local_pbt/online_mining_training_loop/v472_train_v471_from_v471_18strict_grixis` |
| Source play profile | `Pauper-Spy-Combo-Value-OnlineLoop-v470` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v471` |

v471 online mining:

- Online eval source: v470 vs Grixis Affinity, skill 7, deterministic exact-seed mode, 8 chunk attempts.
- Valid online eval result: `3 / 7`; one chunk produced a `0 / 0` game-thread timeout.
- Live-checkpoint mining selected 96 ranked `ACTIVATE_ABILITY_OR_SPELL` snapshots.
- Terminal-line rows: `1536`.
- Terminal outcomes: `816` wins, `720` losses.
- Strict soft-pass value targets: `18 / 96` checkpoint groups admitted.
- Include-pass diagnostic value targets: `28 / 96` admitted, but 16 groups were flagged `suspect_pass_best`; these were excluded from training.
- v471 initially stopped at `insufficient_targets` because this run set `--min-trainable-targets 30`.

v472 resume/training:

- Reused the completed v471 online-mining run via `--existing-online-run-dir`.
- Lowered the train gate to `--min-trainable-targets 16`, matching the already validated v470 scale, and trained only from the 18 strict soft-pass targets.
- Advantage-value `TrainingData` export: `18 / 18` rows reentered with matching candidate hash and state hash.
- All 18 rows captured TrainingData from checkpoint reentry.
- Candidate training: cloned from v470 into `Pauper-Spy-Combo-Value-OnlineLoop-v471`, Q-head only, `-BranchReturnTargets`, 4 epochs, 72 train-pass samples.

Exact-seed Grixis eval:

| Profile | Result | Notes |
| --- | ---: | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v471` | `3 / 14` | Two chunks produced `0 / 0` timeouts. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v470` source comparator | `1 / 14` | Same seed base and chunk split; two chunks produced `0 / 0` timeouts. |

Interpretation:

- This is another weak-positive mechanical result: the next generated model beat the previous generated model on the paired exact-seed Grixis comparator, but the absolute win rate is still low and the eval contains timeout noise.
- The important thesis signal is not the candidate strength yet; it is that online-reached states can be mined into strict terminal-only branch-return targets, exported through checkpoint reentry, and consumed by the Q-head without replay reconstruction or explicit combo rewards.
- The current bottleneck is target density and label quality. v471 mined 96 online checkpoints but only 18 strict trainable targets; 68 groups were below the value-delta threshold and 16 more were excluded as suspect pass-best rows.
- Next useful unit: scale online capture/mining enough to produce at least roughly 50 strict non-pass-suspect targets from online-reached states, then train a candidate and rerun the same paired exact-seed comparator. Good knobs are more online games, more ranked snapshots per game, and more root attempts/common samples before changing the reward formulation.

## v473-v474 Decisive Terminal-Target Gate

Purpose:

- Test whether the scaled online loop failed because branch labels were too diffuse, not because checkpoint mining or combo discovery stopped working.
- Stay thesis-clean: the new filter uses only terminal win-rate spread across candidate actions. It does not encode card names, combo milestones, setup predicates, or "pass is bad" rules.

Implementation:

- Commit `84c30f5c63` added positive-action quality gates to `scripts/mtgrl/export_terminal_line_value_targets.py` and threaded them through both online loop drivers:
  - `--max-positive-actions`
  - `--max-positive-fraction`
  - `--positive-value-threshold`
- The gate rejects checkpoint groups where too many legal candidates have positive terminal value. This preserves decisive "few actions win, alternatives lose" labels and filters broad states where many different actions all reach terminal wins.

Artifacts:

| Artifact | Path |
| --- | --- |
| Scaled broad-label loop | `local-training/local_pbt/online_mining_training_loop/v473_online_loop_from_v471_grixis_g16_r256_a24_s6` |
| Decisive re-export/train run | `local-training/local_pbt/online_mining_training_loop/v474_decisive_train_from_v473_pos3_frac50` |
| Decisive candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` |

v473 scaled broad-label result:

- Online play profile: `Pauper-Spy-Combo-Value-OnlineLoop-v471`.
- Live-checkpoint mining selected `256` ranked `ACTIVATE_ABILITY_OR_SPELL` snapshots from 16 Grixis games.
- Terminal-line rows: `6144`.
- Terminal outcomes: `2919` wins, `3225` losses.
- The search still found combo-relevant roots, including `Cast Balustrade Spy` and `Flashback sacrifice three creatures`.
- Strict soft-pass value targets: `72 / 256` checkpoint groups admitted.
- TrainingData export: `72 / 72` rows reentered with matching candidate hash and state hash.
- Broad candidate `Pauper-Spy-Combo-Value-OnlineLoop-v472` eval: `2 / 14`.
- Source comparator `Pauper-Spy-Combo-Value-OnlineLoop-v471` eval: `4 / 16`.

Diagnosis:

- v473 was mechanically healthy but label quality was weak.
- Many admitted groups assigned positive value to most legal candidates, so the Q update was broad and noisy. Examples had 5, 6, or 7 positive candidates out of the candidate list, meaning the target was often "many actions are fine" rather than "this choice is important."

v474 decisive re-export:

- Reused the completed v473 terminal-line mining artifact.
- Gate: `--max-positive-actions 3 --max-positive-fraction 0.5 --positive-value-threshold 0.0`.
- Decisive targets admitted: `16 / 256`.
- Classification counts: `6` strong delta, `10` moderate delta.
- Rejections newly attributable to decisiveness: `positive_actions_gt_max=116`, `positive_fraction_gt_max=120`.
- TrainingData export: `16 / 16` rows reentered and captured.
- Candidate training: cloned v471 into `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive`, Q-only branch-return update, 4 epochs, 64 train-pass samples.
- Eval: `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` scored `5 / 16` vs Grixis Affinity on deterministic seed base `12161`.

Interpretation:

- This is the first evidence that label-quality filtering matters in the online-mining loop: the same v473 search artifact produced a worse broad candidate (`2 / 14`) and a better decisive candidate (`5 / 16`) after filtering diffuse states.
- The result is promising but not promotion-grade. The margin over the v471 comparator is one game, and the dataset is only 16 decisive rows.
- The next thesis-relevant unit is another online loop from `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` with the decisive gate enabled during target export, then a paired exact-seed comparator against the v473-Decisive source.

Storage note:

- Completed cold v469-v472 local artifacts were moved to `E:\mage-archive\IdeaProjects\mage` and NTFS junctions were left at the original `local-training` paths.
- Active v473 and v474 artifacts remain on `C:` for immediate analysis.

## v475-v476 Repaired Anchor Mining Follow-Up

Purpose:

- Continue the thesis-clean online loop from `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive`.
- Validate that repaired live-checkpoint loading and anchor-based checkpoint reentry can mine, export, train, and evaluate from the v475 online run without forced-prefix replay.
- Keep the reward signal terminal-only: no combo-state labels, card-specific intermediate rewards, or hard-coded "pass is bad" rules.

Artifacts:

| Artifact | Path |
| --- | --- |
| v475 online run and repaired mining artifact | `local-training/local_pbt/online_mining_training_loop/v475_decisive_loop_from_v473_decisive_grixis_g24_r384_a24_pos3` |
| v476 train/eval run | `local-training/local_pbt/online_mining_training_loop/v476_train_from_v475_anchor_retry_grixis_g24_r384_pos3` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v475-Decisive` |
| Source chunk-6 timeout rerun | `local-training/local_pbt/online_mining_training_loop/v476_train_from_v475_anchor_retry_grixis_g24_r384_pos3/post_train_eval_reruns/v476_source_chunk006_timeoutfix_rerun` |

v475 repaired mining:

- Online play profile: `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive`.
- Online eval scored `11 / 24` vs Grixis Affinity before mining.
- Live-checkpoint mining selected `384` ranked `ACTIVATE_ABILITY_OR_SPELL` snapshots from the v475 online eval.
- Terminal-line rows: `9216`.
- Terminal outcomes: `4472` wins, `3298` losses, `1436` forced-text mismatches, and `10` anchor text/size mismatches.
- Terminal rate after filtering: `7770 / 9216` (`0.843099`).
- Combo-relevant terminal mining was present but sparse: `83` Spy rows, `63` Spy wins, `60` Dread Return rows, `0` full-combo wins, max combo score `3`.
- Decisive soft-pass value targets admitted `32 / 383` checkpoint groups.
- Classification counts: `26` moderate terminal-value deltas, `6` strong terminal-value deltas.
- TrainingData export reentered and captured `32 / 32` rows with matching candidate hashes and state hashes.

v476 train/eval:

- Candidate training cloned v473-Decisive into `Pauper-Spy-Combo-Value-OnlineLoop-v475-Decisive`.
- Training used Q-only branch-return targets, 4 epochs, and `128` train-pass samples.
- Exact-seed Grixis eval used seed base `12161`, 16 chunks, one game per chunk.

| Profile | Result | Win Chunks | Notes |
| --- | ---: | --- | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v475-Decisive` candidate | `3 / 16` | `9,10,12` | Completed without zero-total chunks. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` source | `4 / 16` | `8,10,13,15` | Main source eval had chunk 6 as `0 / 0`; rerun after timeout fix produced a real loss. |

Harness fix:

- The eval path started `GameHealthMonitor` with `GAME_TIMEOUT_SEC=900` but still joined the game thread with a hard-coded `300` second timeout.
- Chunk 6 of the source comparator hit that mismatch and produced `0 / 0`.
- The eval join now honors `EVAL_GAME_THREAD_TIMEOUT_SEC`, falling back to `GAME_TIMEOUT_SEC`, plus a 30 second shutdown margin.

Interpretation:

- Mechanically, the checkpoint mining/training loop is now healthy: repaired v475 checkpoints can be mined, exported, trained, and evaluated without prefix replay startup.
- The v476 policy update was not a promotion candidate: it lost the paired comparator by one game (`3 / 16` vs repaired source `4 / 16`).
- The admitted v475 labels were decisive but still midrange-heavy. The most common best labels were `Cast Lead the Stampede` (`5`), `Cast Winding Way` (`4`), `Cast Masked Vandal` (`4`), `Cast Balustrade Spy` (`3`), `Cast Gatecreeper Vine` (`3`), and `Cast Quirion Ranger` (`3`).
- This result supports the current diagnosis: terminal-only branch mining is viable, but online target selection still does not put enough high-quality pressure on combo-relevant decision states.

Next useful unit:

- Keep the same terminal-only reward thesis, but change target selection rather than adding heuristic combo rewards.
- Prioritize online/mined checkpoints whose terminal-line search exposes rare high-leverage wins or source/candidate disagreements, then train a small candidate and rerun the same exact-seed comparator.
- The immediate local implementation target is a selector/export gate that increases density of rare, high-delta terminal wins while staying card-name agnostic in the reward itself.

## v477 Source-Regret Gate

Purpose:

- Test whether the v475 labels were weak because they included decisions where the searched best action was not clearly better than the source policy's selected action.
- Stay thesis-clean: the gate uses only terminal branch outcomes and the actual source-selected candidate metadata. It does not encode card names, combo predicates, or intermediate setup rewards.

Implementation:

- Commit `3c0fe65b28` added source-selected action metadata to terminal value-target export and introduced `--min-source-regret`.
- The export rejects a checkpoint group when the searched best action improves over the source-selected action by less than the requested terminal-value margin.
- v477 reused the repaired v475 mining artifact and trained only on the `--min-source-regret 0.10` target subset.

Artifacts:

| Artifact | Path |
| --- | --- |
| Source-regret target export | `local-training/local_pbt/debug/v475_source_regret010_value_targets` |
| v477 train/eval run | `local-training/local_pbt/online_mining_training_loop/v477_train_v475_source_regret010_grixis` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v477-Regret010` |

Target export:

- Source-regret gate admitted `14` trainable rows from the v475 repaired mining artifact.
- TrainingData export reentered and captured `14 / 14` rows with matching candidate hashes and state hashes.
- Candidate training used Q-only branch-return targets, 4 epochs, and `56` train-pass samples.
- The admitted labels still skewed toward broad setup and midrange corrections. Most common best labels were `Cast Masked Vandal` (`3`), forestcycling (`2`), `Cast Quirion Ranger` (`2`), `Cast Lead the Stampede` (`2`), and singletons including `Cast Balustrade Spy`.

Exact-seed Grixis eval:

| Profile | Result | Win Chunks | Notes |
| --- | ---: | --- | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v477-Regret010` candidate | `2 / 16` | `10,11` | Completed without zero-total chunks. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` source | `3 / 16` | `4,10,12` | Completed without zero-total chunks under the eval timeout fix. |

Interpretation:

- The source-regret gate worked mechanically, but it was not a useful promotion step: v477 lost the paired exact-seed comparator (`2 / 16` vs `3 / 16`) and was also weaker than v476's broader 32-row update (`3 / 16`).
- The result rejects source-regret as a standalone target-quality filter. It overfiltered the dataset to 14 rows while still leaving mostly broad setup/midrange labels.
- The next unit should not add card-name rewards. It should make the terminal-only selector more selective for rare high-leverage terminal wins, for example by measuring how exceptional the best action is relative to the checkpoint's overall terminal win rate.

## v478 Rare-Edge Terminal Selector

Purpose:

- Test whether target quality improves when the exporter keeps checkpoints where the best terminal action is both high-value and exceptional relative to the checkpoint's overall terminal win rate.
- Stay thesis-clean: the gate uses only terminal branch outcomes. It does not name combo cards, define combo-ready state, or reward setup milestones.

Implementation:

- Commit `da85249e77` added terminal rarity metadata and gates to `scripts/mtgrl/export_terminal_line_value_targets.py`:
  - `group_terminal_wins`, `group_terminal_attempts`, and `group_win_rate`
  - `best_over_group_edge`
  - `--min-best-value`
  - `--max-group-win-rate`
  - `--min-best-over-group-edge`
- The same knobs are threaded through the online mining and online mining training loop drivers.

Artifacts:

| Artifact | Path |
| --- | --- |
| Rare-edge target export | `local-training/local_pbt/debug/v475_rare_edge050_gwr045_value_targets` |
| v478 train/eval run | `local-training/local_pbt/online_mining_training_loop/v478_train_v475_rare_edge050_gwr045_grixis` |
| Exact-seed candidate rerun | `local-training/local_pbt/online_mining_training_loop/v478_train_v475_rare_edge050_gwr045_grixis/post_train_eval_reruns/v478_rare_edge_candidate_seed12161` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` |

Target export:

- Gate: `--min-best-value 0.9 --max-group-win-rate 0.45 --min-best-over-group-edge 0.5`, layered on the existing decisive target gates.
- Rare-edge targets admitted `17 / 383` checkpoint groups.
- Classification counts: `14` moderate terminal-value deltas, `3` strong terminal-value deltas.
- Mean admitted `group_win_rate` was about `0.313`; mean `best_over_group_edge` was about `0.687`.
- TrainingData export reentered and captured `17 / 17` rows.
- Candidate training used Q-only branch-return targets, 4 epochs, and `68` train-pass samples.

Evaluation:

| Profile | Seed Base | Result | Win Chunks | Notes |
| --- | ---: | ---: | --- | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` candidate | `7161` | `4 / 16` | `6,9,10,13` | Initial run used default post-eval seed base. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v473-Decisive` source | `7161` | `3 / 16` | `3,12,13` | Same accidental paired seed as candidate. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` candidate | `12161` | `4 / 16` | `3,4,11,16` | Direct exact-seed rerun for comparison to v476/v477 family. |

Interpretation:

- v478 is a weak-positive selector result, not a promotion result.
- It is the best candidate in the v475-derived training family so far: v476 scored `3 / 16`, v477 scored `2 / 16`, and v478 scored `4 / 16` on the exact `12161` seed base.
- The accidental paired smoke comparator also favored v478 by one game (`4 / 16` vs source `3 / 16`), but the known source measurements around seed `12161` are noisy (`3 / 16` to `4 / 16`), so this is not enough to claim model improvement.
- The useful signal is that card-agnostic rare-edge target selection improved over source-regret and broad decisive filtering. The next unit should scale this selector to a larger online-mined corpus before HPC promotion, using `--post-eval-seed-base 12161` explicitly for paired local evals.

## v479 Scaled Rare-Edge Online Loop

Purpose:

- Scale the v478 rare-edge terminal selector from the repaired v475 artifact into a fresh online-mined corpus.
- Keep the experiment thesis-clean: target admission still uses terminal branch outcomes only, without naming combo cards or defining combo-ready state.

Artifacts:

| Artifact | Path |
| --- | --- |
| v479 online/training/eval run | `local-training/local_pbt/online_mining_training_loop/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24` |
| Online mining run | `local-training/local_pbt/online_mining_training_loop/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24/online_mining/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` |
| Rare-edge value targets | `local-training/local_pbt/online_mining_training_loop/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24/online_mining/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge/value_targets/v479_scaled_rare_edge_from_v478_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge_cycle01_mine_softpass` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` |

Setup:

- Started from `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge`.
- Online play used `32` Grixis games, `--max-snapshots 512`, `--ranked-max-per-game 24`, `--line-attempts 24`, and `8` mining shards.
- Rare-edge gate reused v478's thresholds: `--min-best-value 0.9 --max-group-win-rate 0.45 --min-best-over-group-edge 0.5`.
- Paired post-train eval explicitly used `--post-eval-seed-base 12161`.

Mining:

- Online source play scored `11 / 32` against Grixis.
- Terminal-line mining wrote `12,288` terminal rows from `512` selected checkpoints.
- Terminal outcomes: `7,027` wins and `5,261` losses.
- Combo-relevant mined rows were present without combo-specific rewards: `119` `Cast Balustrade Spy` rows with `89` wins, and `99` `Flashback sacrifice three creatures` rows. `full_combo_wins` remained `0`, but `max_combo_score` was `3`.

Target export and training:

- Rare-edge target export admitted `31 / 512` checkpoint groups.
- Classification counts: `26` moderate terminal-value deltas and `5` strong terminal-value deltas.
- TrainingData export reentered and captured `31 / 31` rows with matching candidate/state hashes.
- Candidate training used Q-only branch-return targets, 4 epochs, and `124` train-pass samples.
- Most common admitted best labels were `Cast Winding Way` (`6`), `Cast Roost Seek` (`5`), Forestcycling (`5`), Swampcycling (`4`), `Sacrifice {this}: Add {R}{R}.` (`2`), and `Cast Saruli Caretaker` (`2`). One admitted label was `Cast Balustrade Spy`.

Evaluation:

| Profile | Seed Base | Result | Win Chunks | Notes |
| --- | ---: | ---: | --- | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` candidate | `12161` | `4 / 16` | `2,12,13,16` | Completed without zero-total chunks. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v478-RareEdge` source | `12161` | `3 / 16` | `2,3,6` | Same exact paired seed set. |

Interpretation:

- v479 is a weak-positive result: it beat the paired source by one game (`4 / 16` vs `3 / 16`), but this is not enough for promotion or an HPC-scale claim.
- The scaled selector kept the useful thesis signal from v478: branch mining found real terminal Spy/Dread Return wins, and terminal-only target filtering produced trainable labels without human combo heuristics.
- The admitted target set is still mostly broad setup and card-flow decisions rather than dense combo-finish decisions, so the next thesis-relevant unit should improve the way terminal wins are converted into online policy pressure.
- Harness issue: post-train eval still ran with `parallel=1`, despite the earlier need for better CPU utilization. Fixing the post-eval parallelism plumbing should happen before the next longer local/HPC run.
- Harness caveat: the cycle-1 source play/profile comparison used the old `--initial-q-blend 0.0` default, while the trained candidate eval used `CANDIDATE_Q_BLEND=1.0`. Treat the paired source result as a smoke baseline, not a clean v478-Q comparison. The next run should start supplied generated `--play-profile` values with Q blending enabled.

## v480 Parallel Rare-Edge Follow-Up

Purpose:

- Rerun the scaled rare-edge loop from v479 with both throughput fixes active:
  - deterministic evals may opt into `parallel=4` via `--allow-deterministic-parallel`;
  - generated `--play-profile` starts now default to `CANDIDATE_Q_BLEND=1.0`.
- Keep the reward thesis-clean: terminal outcomes only, no combo-state labels, card-specific rewards, or hand-authored combo milestones.

Artifacts:

| Artifact | Path |
| --- | --- |
| v480 online/training/eval run | `local-training/local_pbt/online_mining_training_loop/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24` |
| Online mining run | `local-training/local_pbt/online_mining_training_loop/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24/online_mining/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeS` |
| Rare-edge value targets | `local-training/local_pbt/online_mining_training_loop/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24/online_mining/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeS/value_targets/v480_parallel_rare_edge_from_v479_grixis_g32_r512_a24_cycle01_online_Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeS_cycle01_mine_softpass` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v480-RareEdgeParallel` |

Mining and targets:

- Online source play scored `10 / 32` against Grixis with Q blending enabled.
- Terminal-line mining wrote `12,288` terminal rows from `512` selected checkpoints using `8` mining shards.
- Terminal outcomes: `6,208` wins and `6,080` losses.
- Combo-relevant mined rows remained present without combo rewards: `116` `Cast Balustrade Spy` rows with `73` wins, `71` Dread Return rows, `6` Lotleth Giant win rows, `0` full-combo wins by the current detector, and max combo score `3`.
- Soft-pass rare-edge target export admitted `27 / 512` checkpoint groups.
- Classification counts: `21` moderate terminal-value deltas and `6` strong terminal-value deltas.
- TrainingData export reentered and captured `27 / 27` rows with matching candidate/state hashes.
- Candidate training used Q-only branch-return targets, 4 epochs, and `108` train-pass samples.
- The admitted best labels were still mostly setup/mana/card-flow decisions: Forestcycling (`5`), `{T}, Tap an untapped creature you control: Add one mana of any color.` (`4`), `Cast Mesmeric Fiend` (`3`), `Cast Lead the Stampede` (`3`), `Cast Winding Way` (`2`), and one `Cast Balustrade Spy`.

Parallel paired eval:

| Profile | Seed Base | Result | Notes |
| --- | ---: | ---: | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v480-RareEdgeParallel` candidate | `12161` | `4 / 16` | Completed with `parallel=4`. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` source | `12161` | `5 / 16` | Same seed base and profile model hash as later checks. |

Interpretation:

- v480 was mechanically healthy and faster: online eval used 4 JVMs after warmup, mining used 8 shard JVMs, and the root README now records actual eval summaries rather than ambiguous return-code zeroes.
- It was not a promotion result. The candidate lost the paired smoke by one game (`4 / 16` vs `5 / 16`).
- The target set is small and clustered: `27` admitted rows came from only `14` source games, with repeated adjacent decisions from the same games. This can overrepresent one local state without adding new terminal evidence.

## v481 Diversified Rare-Edge Re-Export

Purpose:

- Test a generic target-diversity gate on the completed v480 mining artifact before spending another full mining run.
- The gate is thesis-clean: it does not inspect card names or reward combo milestones. It only limits correlated admitted labels from the same source game.

Implementation:

- Added `--max-targets-per-game` and `--min-ordinal-gap-per-game` to `export_terminal_line_value_targets.py`.
- Threaded both options through the online mining and online mining training loop drivers.
- Added clearer top-level loop README columns: `Train RC`, `Candidate Eval`, and `Source Eval`.

Artifacts:

| Artifact | Path |
| --- | --- |
| Diversity target export | `local-training/local_pbt/debug/v480_rare_edge_diversity_m2_gap8_value_targets` |
| v481 train/eval run | `local-training/local_pbt/online_mining_training_loop/v481_diverse_rare_edge_from_v479_v480artifact_grixis` |
| Candidate profile | `Pauper-Spy-Combo-Value-OnlineLoop-v481-DiverseRareEdge` |
| Source profile | `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` |

Diversity target export:

- Gate layered on v480 rare-edge targets: `--max-targets-per-game 2 --min-ordinal-gap-per-game 8`.
- Admitted examples dropped from `27` to `17`.
- Diversity rejection counts: `diversity_ordinal_gap_per_game=10`, `diversity_max_targets_per_game=4`.
- Classification counts: `11` moderate terminal-value deltas and `6` strong terminal-value deltas.
- TrainingData export reentered and captured `17 / 17` rows.
- Candidate training used Q-only branch-return targets, 4 epochs, and `68` train-pass samples.

Parallel paired eval:

| Profile | Seed Base | Result | Notes |
| --- | ---: | ---: | --- |
| `Pauper-Spy-Combo-Value-OnlineLoop-v481-DiverseRareEdge` candidate | `12161` | `4 / 16` | Same win count as v480. |
| `Pauper-Spy-Combo-Value-OnlineLoop-v479-RareEdgeScaled` source | `12161` | `3 / 16` | Same model hash and manifest settings as v480 source, but different chunk outcomes. |

Parallel-eval reproducibility finding:

- The v479 source model hash matched across the v480 source snapshot, the v481 source snapshot, and the live profile model: `7E1EEDC6338787EA898B4A27B598C47E48F77119318B9C54EBAA31605E298768`.
- Source eval manifests also matched on profile, Q blend, deterministic env, seed base `12161`, seed key mode `matchup`, `parallel=4`, and `allow_deterministic_parallel=true`.
- Despite that, chunk outcomes differed between v480 and v481 source evals. Example: source chunk 1 was a win in v480 but a loss in v481.
- A serial deterministic chunk-1 probe repeated twice under `local-training/local_pbt/debug/v479_source_serial_repro_12161` produced stable losses both times, matching v481 and contradicting the v480 parallel result.

Interpretation:

- The diversity gate is mechanically valid and useful as data hygiene, but the v481 result is not promotion evidence. The candidate still scored only `4 / 16`.
- The bigger blocker is evaluation quality: `--allow-deterministic-parallel` is useful for throughput, but not for exact paired claims. Promotion or comparison claims should use serial deterministic eval, or larger repeated sweeps, not a single 16-game parallel smoke.
- A full serial paired v481/source eval was launched under `local-training/local_pbt/debug/v481_diverse_serial_pair_seed12161` to get a trustworthy local comparison.
