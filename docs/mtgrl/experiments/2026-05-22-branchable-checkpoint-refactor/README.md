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
