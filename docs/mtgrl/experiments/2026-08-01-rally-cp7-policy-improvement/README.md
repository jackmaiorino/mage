# Rapid Rally CP7 Policy Improvement

## Status

The behavior-cloning route and the first terminal-outcome update sweep are rejected. On 32 fresh matched base-990001 pairs the full behavior-cloning derivative scored `23-41` (35.9375%) against CP7, while generation 384 scored `30-34` (46.875%) on the same pairs. Three smaller behavior-cloning updates then scored no better than `8-8` in the eight-pair development screen. The large optimization-split imitation gain did not transfer to live play strength, and reducing update distance did not produce a positive signal.

Outcome v1 then trained five one-update derivatives from 32 actual generation-384-versus-CP7 pairs at base seed 1010001. Generation 384 scored `27-37` while producing that corpus. Every derivative scored exactly `8-8` on the base-960001 eight-pair development screen. Larger steps changed policy decisions and trajectory lengths, but none changed the win count, so no outcome derivative advanced to a fresh 32-pair block.

Centered outcome v2 changed the estimator rather than only its scale. It froze the source value baseline, centered and standardized advantages, and gave each contributing episode equal total objective mass. The aggressive arm passed an offline policy-change gate, repeated `8-8` on the development screen, and then scored `30-34` against CP7 versus matched generation 384 at `27-37` on fresh base-1020001 pairs. On a second fresh block at base 1030001, both policies scored `29-35`. Pooled across the 128 fresh games, the aggressive derivative scored `59-69` versus generation 384 at `56-72`, a three-win and 2.34375-percentage-point lift. This was the first promising matched signal in the campaign, but it remained insufficient for promotion.

Iteration 2 warm-started from that aggressive checkpoint, collected 32 new on-policy pairs at base 1040001, inherited the complete parent optimizer state at Adam step 1, and trained seven one-update derivatives to Adam step 2. The predeclared replay gate required at least 60 fixed-CDF sampled-action flips in 2,995 decisions. Every arm failed, with 27 to 51 flips. The later policy-scale sweep also exposed that old flip count as non-monotone: mean action total variation rose from `0.01129` to `0.01220` while fixed-CDF flips fell from 50 to 47 across the policy-scale 1.0 to 2.0 endpoints.

The policy-scale 2.0 arm was nevertheless run as an explicitly experimental rapid live candidate after the replay metric proved unsuitable. It was not promoted and was never described as having passed the old gate. It scored `2-6` in a four-pair integration smoke, then `50-78` against CP7 across fresh bases 1050001 and 1060001 versus the matched parent at `48-80`. The direct paired comparison was only `G=3, L=1, T=124`, with one-sided exact `p=0.3125`. Promotion failed and the parent checkpoint was retained.

Iteration 3 then continued experimentally from the rejected policy-scale 2.0 child using 64 new on-policy pairs at base 1070001 and one further warm-start update to Adam step 3. On final-harness fresh bases 1100001 and 1110001, the child scored `55-73` versus the retained centered-v2 control at `58-70`. The paired result was `G=1, L=4, T=123`, with one-sided exact `p=0.96875`. Iteration 3 failed every promotion condition and was retired.

The archived-generation screen also failed to improve generation 384. Generation 384 remains the original anchor baseline, while centered-v2 aggressive manifest `706b3a...` remains the retained outcome-training parent after the iteration-2 and iteration-3 rejections. Both are below 50% in the relevant fresh CP7 blocks, and neither has demonstrated CP7 superiority. Nothing here supports a pro-level play claim.

## Code provenance

XMage repository commits:

- `a6e9d2d223f9bb87aa985e6ce56c98236b9a6a5e`: exact Rally-versus-CP7 anchor
- `988482462bb473df34ec24dafc5dc6f9fee104f0`: recorded exact anchor result
- `08105f238aa9e71d764073c9beb85bffe30b9c9d`: CP7 teacher-export pass-through
- `ecf62adc5aa0116ccfe8ce64fd548c954bcb2fb1`: checkpoint-generation selector
- `e1d4ce80ccebca26095b5694a89c9468cd1f41ab`: selected-generation identity validation
- `5d89881adcdb62c55ba4c260c4d34e1f76ad8e35`: derivative-root evaluation, candidate-controlled outcome export, dynamic fail-closed derivative identity validation, and this campaign report
- `95bc86944a0b625c79aa4b505e87f0049f673901`: restricted selected-action projection, stable source validation, projection instrumentation, and checked alignment accounting

Rust repository commits:

- `54e7fdc8f11dac2f2c44d1dd380b1d6d2c37b494`: exact checkpoint shadow scorer
- `208738a2605907602ae5550accfb74ad155e2d79`: partial Chain Lightning copy-payment match
- `af05f8ba4aa6f6e11da17ddb9d5d816cbac7345a`: mapped CP7 teacher export
- `0fba419f79e5ddc2eebf906d068ebea815abae04`: validated checkpoint-generation selection
- `1a938364707ec1e93eb93995518329cae2924f6f`: strict CP7 behavior-clone training and scoring
- `41fdd71a7195841545b81fe6f8c7e7d4e6c61669`: candidate-controlled XMage outcome export, strict terminal REINFORCE/value training, derivative verification, and checkpoint-scorer authority
- `705b87284f7bc519870e43117969f27533c9829a`: iterative parent-bound CP7 outcome export and training, full parent Adam-state inheritance, legacy compatibility, and fail-closed export poisoning
- `2c1fbfab37603114a3e25e5cc50c418294735105`: gated base-1090001 native phase-cursor diagnostic

The iteration-2 corpus, artifacts, and live results were produced while the Rust changes were still uncommitted on top of `41fdd71a7195841545b81fe6f8c7e7d4e6c61669`. Those changes, plus the final fail-closed exporter-poison repair, are now committed as `705b87284f7bc519870e43117969f27533c9829a`. The implementation is confined to `native_policy_train_step_v1.rs`, `native_xmage_cp7_outcome_reinforce_v1.rs`, and `native_checkpoint_shadow_stdio_v1.rs`. The historical executable hash still pins the completed games, while the post-fix rebuild below is the current executable for future work.

The behavior-cloning trainer and derivative-scorer integration were uncommitted when these games ran. The Windows scorer executable used for both behavior-cloning blocks was:

- Path: `C:\Users\Jack\IdeaProjects\mtg-kernel-entropy-smoke-v1\target-windows-cp7-bc\release\checkpoint_shadow_stdio_v1.exe`
- SHA-256: `b3ab1d2293dd7bf3bbcff5d7333187431c6bc414cd4e6c6bfa7992ccd41edbd2`

The executable hash pins the live behavior. Artifact-verifier hardening is reviewed separately below.

The post-commit scorer used for the strict small-update screens was 4,318,720 bytes with SHA-256 `481fdcc379db32409aa4824d4d2145e84b3d93bc9b915f82e535d3c9d186dfa1`.

The Windows scorer used for the outcome-derivative screens was 4,577,792 bytes with SHA-256 `9b1397be8acde45ad1a97eefed90ce2a72a576811f004326cf33e475382ab728`.

The Windows scorer used for centered outcome v2 was 4,634,624 bytes with SHA-256 `559c2704ded38a6f884f4d804aab95c830e7ee4cb11e02495c0c6299005b408c`.

The pre-fix Windows scorer used for iteration-2 corpus export and live evaluation was:

- Path: `C:\Users\Jack\IdeaProjects\mtg-kernel-entropy-smoke-v1\target-windows-outcome-iterative\release\checkpoint_shadow_stdio_v1.exe`
- Bytes: 4,675,584
- SHA-256: `17d0125af2667bdb389c10df1b06e3b38446f9e760107a772f7b2dc74593250a`

The corresponding Linux release scorer used for strict corpus, parent-state, and artifact verification was 3,783,968 bytes with SHA-256 `4ae88966d2b93d7cd8869b3da869ce455a734ead68b3718e947dde8385fa4e12`.

After commit `705b87284f7bc519870e43117969f27533c9829a`, the Windows scorer was rebuilt at the same path with fail-closed export poisoning:

- Bytes: 4,671,488
- SHA-256: `16fa033a1646db18272e5e8c8b46c2cd93572314484cbaade75256feaa949f95`

This post-fix executable is the current scorer for future exports and live work. The completed iteration-2 games remain attributed to the pre-fix executable above; the exporter repair changes failure handling, not the recorded successful game outcomes. The iteration-3 base-1070001 export and final live base-1100001/base-1110001 runs used this post-fix executable.

The isolated Windows diagnostic scorer for base 1090001 was 4,736,000 bytes with SHA-256 `88b5c42d6b96e9dfa60e1a9440f35f5f54aae0cc8cdc551d93f10a72e3f4d732`. It was built from commit `2c1fbfab37603114a3e25e5cc50c418294735105` at `target-windows-base109-step64-diagnostic\release\checkpoint_shadow_stdio_v1.exe` and used only for the gated episode-45, entry-step-64 snapshots described below.

## Fixed source checkpoint

- Promoted run SHA-256: `2c9b7423004428c0e2bb138afafc15ec65957f6bd98c4587bea704fbf9549aae`
- Generation: `384`
- Checkpoint SHA-256: `4bd38cf3a9af3fb03fb04428fbc4286d4635007e848c7b9f0740122e430cbba8`
- Original store: `D:\mtg-kernel-ladder-pilot-20260725\pool3\primary`

The exact anchor and its 112-game result are recorded in [the preceding anchor report](../2026-08-01-xmage-rally-cp7-anchor/README.md).

## Teacher corpora

| Corpus | Base seed | Pairs | Decision rows | Terminal rows | Physical groups | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot | 960001 | 8 | 832 | 16 | 698 | 14,338,542 | `8666062cbb31e6aaa98861205ad5ce068aea3c622fe85c8daa41a125bbdd2b14` |
| Train and optimization | 970001 | 32 | 4,952 | 64 | 4,025 | 94,839,535 | `24211ca83cc56d40fd2b574bbb120345aa602d9bec66e0e1b938ff9cb91bf6b0` |

Local files:

- `C:\Users\Jack\AppData\Local\Temp\mtg-cp7-teacher-v1\pilot-base960001-pairs8.jsonl`
- `C:\Users\Jack\AppData\Local\Temp\mtg-cp7-teacher-v1\train-base970001-pairs32.jsonl`

Both files contain one additional header row. The pilot passed ordinal, action, tensor, pair, terminal, and tensor-commitment audits. The 32-pair corpus passed streaming ordinal, shape, action, and pair audits. Its 4,952 exported decisions also passed the declared Windows-MSVC to Linux-GNU forward-recompute envelope of `3e-5` absolute plus `3e-5` relative tolerance.

## Archived-generation screen

The development screen used the same eight environment-seed pairs at base seed 960001. Candidate seat was swapped within every pair.

| Generation | Candidate result | On play | On draw | Candidate sweeps | CP7 sweeps | Split pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 10-6 (62.50%) | 4-4 | 6-2 | 2 | 0 | 6 |
| 320 | 7-9 (43.75%) | 4-4 | 3-5 | 0 | 1 | 7 |
| 384 | 8-8 (50.00%) | 4-4 | 4-4 | 1 | 1 | 6 |
| 448 | 8-8 (50.00%) | 3-5 | 5-3 | 0 | 0 | 8 |
| 512 | 8-8 (50.00%) | 3-5 | 5-3 | 0 | 0 | 8 |

Generation 384 exactly repeated the accepted pilot counts and trajectory. Only generation 256 earned a larger confirmation run.

On the 32 base-970001 pairs, generation 256 scored `27-37` (42.19%). Generation 384 scored `28-36` (43.75%) on those same pairs while generating the full teacher corpus. The apparent generation-256 lift did not reproduce: it finished one win behind generation 384. Archived-generation selection is therefore closed as a policy-improvement route for this checkpoint.

## Behavior-cloning fit

The 32 pairs were split deterministically by pair index: 24 training pairs and 8 optimization pairs. Training contained 2,967 physical groups and 3,700 autoregressive substeps; optimization contained 1,058 physical groups and 1,252 substeps. The selected grid point used learning rate `3e-5`, three epochs, Adam reset, no value gradient, and finished at Adam step 141.

| Optimization metric | Generation 384 | Selected derivative | Change |
| --- | ---: | ---: | ---: |
| Mean NLL per physical group | 2.858831 | 1.539000 | -46.2% |
| Substep top-1 accuracy | 69.169% | 70.208% | +1.038 pp |
| Physical exact accuracy | 70.983% | 71.645% | +0.662 pp |
| Surface top-1 accuracy | 77.962% | 79.147% | +1.185 pp |
| Blocker top-1 accuracy | 39.623% | 41.038% | +1.415 pp |
| Attacker top-1 accuracy | 63.265% | 63.265% | 0.000 pp |

Selected live artifact directory: `D:\mtg-kernel-cp7-bc-train-base970001-grid-v1`

- Original manifest `checkpoint.json` SHA-256: `a1737ead5c36e5abeb0e0948c0fe095554f33601a55e37ec7f91de232fa9088a`
- Payload `checkpoint.state.f32le` SHA-256: `de1132f6b8b55975154133b91a2f2ea90bc1159676a041057fd827e728eca4e1`
- Model-parameter SHA-256: `3f4da9d761771cf0d7cfe2da19b52dd93dd0bc59466d92318cc11fc850d8c4dc`
- Native-state SHA-256: `64df1692fae7f78d0d4d4a4d6489325d253125276ca578c94912c9bd12374b56`

The optimization pairs selected the grid winner, so these metrics are evidence that the update learned the CP7 action distribution, not an unbiased estimate of generalization or play strength.

## Live XMage evaluation

Every block used deterministic CP7 skill 7 on the Rally mirror, with candidate seat swapped inside each environment-seed pair.

| Candidate | Base seed | Pairs | Candidate result | On play | On draw | Candidate sweeps | CP7 sweeps | Split pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full BC, integration screen | 980001 | 8 | 7-9 (43.75%) | 3-5 | 4-4 | 1 | 2 | 5 |
| Full BC, fresh comparison | 990001 | 32 | 23-41 (35.9375%) | 13-19 | 10-22 | 1 | 10 | 21 |
| Generation 384, matched control | 990001 | 32 | 30-34 (46.875%) | 16-16 | 14-18 | 2 | 4 | 26 |
| BC `3e-6`, one epoch | 960001 | 8 | 8-8 (50.00%) | 4-4 | 4-4 | 1 | 1 | 6 |
| BC `1e-5`, one epoch | 960001 | 8 | 7-9 (43.75%) | 4-4 | 3-5 | 0 | 1 | 7 |
| BC `1e-5`, three epochs | 960001 | 8 | 7-9 (43.75%) | 4-4 | 3-5 | 0 | 1 | 7 |

The eight-pair screens established executable end-to-end behavior but gave no positive play-strength signal. The larger matched block resolves the full-update decision: behavior cloning at `3e-5` for three epochs is seven wins and 10.9375 percentage points behind generation 384 on identical fresh pairs. The smaller candidates also fail the development gate, so none advances to another 64-game block.

| Update | Relative parameter L2 from g384 | Optimization NLL | Live result |
| --- | ---: | ---: | ---: |
| `3e-6`, one epoch | 0.0697146% | 2.79637 | 8-8 |
| `1e-5`, one epoch | 0.232523% | 2.65042 | 7-9 |
| `1e-5`, three epochs | 0.672775% | 2.25906 | 7-9 |
| `3e-5`, three epochs | 1.782009% | 1.53900 | 23-41 on the matched block |

Matched g384 scored `30-34` in the large-update comparison. This closes the simple explanation that only the full update was too large: a roughly 25-times smaller update tied the baseline development count but did not improve it, while both intermediate updates lost. Lower imitation NLL tracked larger source displacement, not better live play.

## Post-run integrity review

Review found three artifact-verification gaps in the implementation used to create the original manifest:

- Teacher ingestion checked action indices and tensors but did not fully bind action semantics, selected semantic, exact model-input commitment framing, or a caller-pinned teacher JSONL hash.
- The selected payload digest was not bound tightly enough to the selected grid trial and its Adam step.
- The zero-value-gradient assertion matched `value.*`, while the actual four parameters are named `value_head.*`, so that assertion was vacuous.

These gaps weaken artifact-promotion guarantees, but they do not erase the observed gameplay. The scored payload and executable are pinned above, the games completed behaviorally, and the live conclusion is rejection rather than promotion.

The shared Rust source now contains fixes for all three findings. A strict artifact was regenerated at `D:\mtg-kernel-cp7-bc-train-base970001-grid-strict-v1` with manifest SHA-256 `6ba733fead0d36c26cd24630245fa6f2a1216ae60c73f46d45e83b4cc714676c`. Its payload, model-parameter, and native-state hashes are bit-identical to the live artifact.

The strict retrain accepted all 4,952 committed teacher rows and reproduced the payload bit-for-bit. Strict derivative load-and-score identity, all four zero-value-head assertions, 10 shadow tests, and two CLI parser tests passed. The old unbound manifest and an incorrect teacher hash both rejected. The implementation is committed as `1a938364707ec1e93eb93995518329cae2924f6f`. The original artifact remains withheld from promotion; regardless, the full derivative is not a promotion candidate because its live play result failed.

## Outcome v1 corpus and trainer

The outcome corpus came from actual XMage games between generation 384 and deterministic CP7 skill 7 in the Rally mirror. Candidate seat was swapped inside every pair.

| Base seed | Pairs | Candidate result | Decision rows | Terminal rows | Physical groups | Bytes | SHA-256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1010001 | 32 | 27-37 (42.1875%) | 2,629 | 64 | 2,198 | 46,838,288 | `ee42241ae8a508260746840b80eca2aa7e8abc8c89ae1caafd08bad755ff96b3` |

Local corpus: `C:\Users\Jack\AppData\Local\Temp\mtg-cp7-outcome-v1\train-base1010001-pairs32.jsonl`.

The strict loader bound the whole-file hash, exact promoted generation-384 authority, model-input commitments, candidate-selected actions, complete autoregressive physical groups, seat-swapped pair metadata, and natural terminal rewards. Linux recomputation of the Windows-exported generation-384 forwards stayed inside the declared `3e-5` absolute plus `3e-5` relative envelope. Across 2,629 rows, the maximum value delta was `5.9604645e-7` and the maximum logit delta was `8.940697e-6`.

Every arm reset Adam and made one full-corpus update over all 2,198 physical groups. The canonical `terminal_reinforce_value/v3` objective already used the model value prediction as the REINFORCE baseline and also trained the value head toward terminal return. These were therefore value-baseline REINFORCE experiments, not plain unbaselined terminal-return updates. The `vc=0.05` arm reduced the value-loss weight; it did not remove the value baseline.

Initial value MSE was `0.8038571` and initial selected-action NLL was `0.4757370` for every arm. Initial total objective was `0.1649335` at `vc=0.5` and `-0.1968022` at `vc=0.05`.

| Arm | Learning rate | Value coefficient | Final total objective | Final value MSE | Final selected NLL | Manifest SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lr3e-5-vc0p5` | `3e-5` | 0.5 | 0.1639528 | 0.7858296 | 0.4758884 | `8ddcf37c0e2b93373e70426b0cfae605a040cf39868c36c8e99165b58d2ba616` |
| `lr1e-4-vc0p5` | `1e-4` | 0.5 | 0.1628939 | 0.7461711 | 0.4763051 | `b02c16c403fa09bd435b46b3763a819635644dc61a02330ccd12861ebb5244e0` |
| `lr3e-4-vc0p5` | `3e-4` | 0.5 | 0.1695740 | 0.6521820 | 0.4779608 | `2825b68b1a9084bf8ac9b00b0ae3d4f5e68cc207efb6d01d7359d2f7a72dbece` |
| `lr1e-3-vc0p05` | `1e-3` | 0.05 | 0.0360624 | 0.5553263 | 0.4996909 | `28be4032ca48943d56348ba3cba44807d7442dd6950f9b784a1f7431e638d0ff` |
| `lr1e-3-vc0p5` | `1e-3` | 0.5 | 0.3037257 | 0.5541869 | 0.4886847 | `f81fba6f5c1f2694a3fdc9545f7926bb5bc9967cdcb3f812b63d409218d6ae90` |

The artifact roots are `D:\mtg-kernel-xmage-cp7-outcome-base1010001-<arm>-v1`. Each manifest binds its 14,771,928-byte native train-state payload, model-parameter digest, source checkpoint, corpus hash, optimizer reset, and Adam step 1. Strict derivative verification and real checkpoint-scorer load-and-score passed for the center `lr1e-4-vc0p5` arm, whose payload SHA-256 is `a61084a0e505a4aecdf84123dff6dfc8d1ba2296eb54c71f4b3fedb5f25c9b7b` and model-parameter SHA-256 is `aeeb6f6e51131e983743814f59494b799e43898c5e06da339f4d6649e72f5b74`.

The offline metrics do not provide play-strength evidence. In particular, stronger value fit and larger update scale did not predict more wins in the live screen.

## Outcome v1 live screen

All five derivatives were evaluated on the same eight matched base-960001 pairs used as the development screen. This was a rapid gate, not a fresh or independent estimate.

| Candidate | Result | On play | On draw | Candidate sweeps | CP7 sweeps | Split pairs | Turns | Rust steps | Physical decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr3e-5-vc0p5` | 8-8 | 4-4 | 4-4 | 1 | 1 | 6 | 198 | 1,446 | 1,207 |
| `lr1e-4-vc0p5` | 8-8 | 4-4 | 4-4 | 1 | 1 | 6 | 198 | 1,434 | 1,195 |
| `lr3e-4-vc0p5` | 8-8 | 4-4 | 4-4 | 1 | 1 | 6 | 196 | 1,407 | 1,173 |
| `lr1e-3-vc0p05` | 8-8 | 4-4 | 4-4 | 1 | 1 | 6 | 194 | 1,391 | 1,157 |
| `lr1e-3-vc0p5` | 8-8 | 4-4 | 4-4 | 1 | 1 | 6 | 196 | 1,399 | 1,166 |

The identical win allocation does not mean the policies were behaviorally identical. Rust-step and physical-decision totals diverged as update scale increased, and total game turns also changed for the larger arms. The updates altered choices and downstream trajectories, but those alterations produced no win-count lift in this screen. No candidate passed the development gate, so none received a fresh matched 32-pair evaluation.

## Centered outcome v2 implementation and review

The v1 objective averaged uniformly over physical groups. Episodes with more candidate decisions therefore supplied more policy and value mass. Centered v2 retained the terminal-return and value-baseline structure but made three targeted changes:

- Freeze each physical group's baseline to its exported generation-384 first-substep value.
- Center and population-standardize the resulting source advantages over the corpus.
- Weight every group by `G / (E * n_e)`, where `G` is corpus group count, `E` is the number of contributing episodes, and `n_e` is the group's episode length in physical decisions. The trainer's outer `1 / G` mean then gives every contributing episode exactly `1 / E` objective mass before floating-point rounding.

The declared objective is `terminal_reinforce_frozen_source_value_standardized_episode_balanced/v1`. Both policy and value objectives are episode-balanced, all 64 corpus episodes contribute, and policy scale is 1.0. This remains value-baseline REINFORCE; centering and episode balancing change the estimator but do not remove the learned baseline or terminal-return target.

A read-only review found the transform math, frozen weighted train step, legacy compatibility, and scorer identity sound. The optional transform field is absent when loading raw v1 manifests, preserving their canonical bytes and load compatibility. The scorer continues to report exact source generation-384 pins plus derivative manifest, payload, native-state, and model hashes. Focused transform tests passed `4/4` with one external-corpus test ignored, the exact frozen-objective core test passed, `git diff --check` passed, and the old raw `lr3e-5-vc0p5` artifact reverified unchanged. The first v2 artifacts subsequently passed strict manifest/payload verification and loaded through the scorer used for live XMage. The centered-v2 Rust changes were uncommitted when these games ran and are now included in commit `705b87284f7bc519870e43117969f27533c9829a`; the historical executable and artifact hashes continue to pin the evaluated behavior.

## Centered outcome v2 offline gate

Both candidates reset Adam and made one full-corpus update from generation 384 with value coefficient 0.5. Their common initial transformed objective was `0.3595237`.

| Arm | Learning rate | Final objective | Selected-action flips versus g384 | Episode coverage | Mean action TV | P90 action TV | Manifest SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Moderate | `3e-4` | 0.2931055 | 24 / 2,629 (0.913%) | Not used as a gate metric | Not used as a gate metric | Not used as a gate metric | `1e473c1b0f9ba14003f6899f976552dbfd5bb57636502edbd3b641ce499a84a0` |
| Aggressive | `1e-3` | 0.2490596 | 68 / 2,629 (2.586%) | 35 / 64 episodes | 0.01872 | 0.03988 | `706b3aa80ec7a3c067d458fef06bb2237320543f202fb2349c5cb885975fdbbb` |

Artifact identities:

| Arm | Payload SHA-256 | Native-state SHA-256 | Model-parameter SHA-256 |
| --- | --- | --- | --- |
| Moderate | `6c823fea4df3d213f585d6f87913aa4884f429e7ba35b646d3500312b94e78a3` | `973639dcbb96146337a5639fbe4d436322c64e9522e751e28487c7096358b695` | `7f5f300d8f8c17d4661640336a0c39775cf3242c76ad83debf2266349fa05082` |
| Aggressive | `eb83be33bcb7418b6f85ec9687da4b7ca5620a1df64721a1942d2793588bbd3c` | `2c55a13abb3157f3f4ba012af663ffa56599c5d6cb90743c1ba6e024ca47a9c8` | `883e4882d01d9cb55ecd7a4ae00e3c95793b6147baf3df08650ef1fa7f8e9546` |

The artifact roots are:

- Moderate: `D:\mtg-kernel-xmage-cp7-outcome-base1010001-std-epbal-lr3e-4-vc0p5-ps1-v1`
- Aggressive: `D:\mtg-kernel-xmage-cp7-outcome-base1010001-std-epbal-lr1e-3-vc0p5-ps1-v1`

The offline replay gate was intentionally behavioral. The moderate arm changed only 24 selected actions in the fixed 2,629-row source replay. The aggressive arm changed 68, spread across 35 episodes, with nontrivial action-distribution movement. Only the aggressive arm advanced to live XMage; the moderate arm was retained as a verified artifact but was not screened live.

## Centered outcome v2 live evidence

The aggressive arm first repeated the base-960001 development screen at `8-8`. That run used 1,385 Rust policy steps and 1,153 physical decisions, differing from both generation 384 and the v1 derivatives while preserving the same aggregate win count.

It then received two fresh 32-pair blocks. Each block included a generation-384 control on exactly the same environment seeds.

| Base seed | Candidate | Result | On play | On draw |
| ---: | --- | ---: | ---: | ---: |
| 1020001 | Centered v2 aggressive | 30-34 (46.875%) | 17-15 | 13-19 |
| 1020001 | Generation 384 control | 27-37 (42.1875%) | 15-17 | 12-20 |
| 1030001 | Centered v2 aggressive | 29-35 (45.3125%) | 19-13 | 10-22 |
| 1030001 | Generation 384 control | 29-35 (45.3125%) | 18-14 | 11-21 |
| Pooled fresh | Centered v2 aggressive | 59-69 (46.09375%) | 36-28 | 23-41 |
| Pooled fresh | Generation 384 control | 56-72 (43.75%) | 33-31 | 23-41 |

The pooled lift is three wins, all on the play; both candidates scored the same `23-41` on the draw. The first block favored centered v2 by three wins and the second was tied, so the result is directionally positive but not yet stable or large enough to establish promotion. It supports another controlled iteration from the improved checkpoint with newly collected on-policy outcomes.

## Iteration 2 warm-start implementation

Iteration 2 made the centered-v2 aggressive checkpoint the exact parent rather than restarting from generation 384. The implementation added these narrow contracts:

- Outcome JSONL schema v2 repeats the complete derivative-parent identity on the header and every decision and terminal row. The loader rejects a missing or mismatched row binding.
- Original-store and outcome-parent training roots are mutually exclusive. The old generation-384 corpus and manifest formats remain loadable through optional fields.
- Parent training loads parameters, first moments, second moments, and Adam step from the verified native state. It does not reset the optimizer. These artifacts start at parent Adam step 1, make one full-corpus update, and end at Adam step 2.
- The standardized, frozen-source-value, equal-episode-mass objective remains unchanged. `policy_scale` is now explicit and separately bound in the transform manifest, while the value objective retains its episode-balanced weight.
- Published manifests state the parent, starting and ending Adam steps, optimizer-reset status, and whether the artifact is an exact one-update on-policy child. The scorer reports parent and child identities, and XMage requires the declared Adam step plus manifest, payload, native-state, and model-parameter hashes together.

The exact parent root was `D:\mtg-kernel-xmage-cp7-outcome-base1010001-std-epbal-lr1e-3-vc0p5-ps1-v1`.

| Parent field | Exact value |
| --- | --- |
| Manifest SHA-256 | `706b3aa80ec7a3c067d458fef06bb2237320543f202fb2349c5cb885975fdbbb` |
| Payload SHA-256 | `eb83be33bcb7418b6f85ec9687da4b7ca5620a1df64721a1942d2793588bbd3c` |
| Native-state SHA-256 | `2c55a13abb3157f3f4ba012af663ffa56599c5d6cb90743c1ba6e024ca47a9c8` |
| Model-parameter SHA-256 | `883e4882d01d9cb55ecd7a4ae00e3c95793b6147baf3df08650ef1fa7f8e9546` |
| Parent-corpus SHA-256 | `ee42241ae8a508260746840b80eca2aa7e8abc8c89ae1caafd08bad755ff96b3` |
| Adam step | 1 |

The parent manifest in turn preserves the original source pins: run `2c9b7423004428c0e2bb138afafc15ec65957f6bd98c4587bea704fbf9549aae`, generation 384, checkpoint `4bd38cf3a9af3fb03fb04428fbc4286d4635007e848c7b9f0740122e430cbba8`, sidecar `7511c0377edd4e8d918fa5843f89a0270a8264e5466c329f6b4ef18bbf9e76bb`, payload `a6c87366b2da9fc33923abab3c0e22d70c884cd9420477df3a475117be6beb99`, native state `fc471f85d28293d72b42dc61de628859173bd67426e251a51bfbbe86c7d586d8`, and model parameters `db58dbe3f1f76b5bdf3bae4de657711dc818393b2bf1eeae88c02d8866b4d01d`.

## Iteration 2 on-policy corpus

The parent played deterministic CP7 skill 7 on 32 seat-swapped pairs at base seed 1040001. The export contains 3,060 LF-terminated records: one header, 2,995 decisions, and 64 terminals.

| Policy | Base seed | Pairs | Result | Decision rows | Terminal rows | Physical groups | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Centered-v2 parent `706b3a...` | 1040001 | 32 | 27-37 (42.1875%) | 2,995 | 64 | 2,541 | 58,840,916 | `b75677397c8461a702bdb5d0f7dfc47fe651e2cd1d4f048cc218001055a828cd` |

Local corpus: `C:\Users\Jack\AppData\Local\Temp\mtg-cp7-outcome-iterative-v1\parent706b-base1040001-pairs32.jsonl`.

All 64 episodes contributed. The exact schema-v2 parent binding was repeated on every row, the whole-file hash and natural terminal returns passed the strict loader, and the Windows-exported forwards passed the existing `3e-5` absolute plus `3e-5` relative Linux transport envelope. The corpus is on-policy for the centered-v2 parent, not for generation 384 and not for any iteration-2 child.

## Iteration 2 one-update sweep and replay gate

Seven artifacts inherited parent Adam state and made one update over all 2,541 physical groups. The common standardized source-advantage mean was `0.08255235223710083`, its population standard deviation was `0.8143143348762791`, and all artifacts ended at Adam step 2.

The old predeclared behavior gate replayed the same fixed action-sampling CDF point for each exported decision and counted whether the child sampled a different action than the parent. Parent replay was exact on `2,995 / 2,995` decisions for every comparison. The threshold was 2%, or a minimum integer count of 60 flips.

| Arm | Learning rate | Value coefficient | Policy scale | Final objective | Fixed-CDF flips | Episodes with flips | Mean action TV | P90 action TV | Manifest SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lr5e-4-vc0p05-ps1` | `5e-4` | 0.05 | 1.0 | 0.02140054 | 27 / 2,995 | 21 / 64 | 0.00540 | 0.009743 | `b4ff37ec6724204281c562b1b5b281cdb92e0824095b17bac4100a433dc217df` |
| `lr1e-3-vc0p05-ps1` | `1e-3` | 0.05 | 1.0 | 0.02272690 | 50 / 2,995 | 30 / 64 | 0.01129 | 0.02046 | `54dd3b9660fb6a7cbd3f40152f0790fd505ac6500d06afc5649802ee5dc05a54` |
| `lr1e-3-vc0p1-ps1` | `1e-3` | 0.1 | 1.0 | 0.06234568 | 51 / 2,995 | 30 / 64 | not retained | 0.02026 | `e3dd6eab7ef84322ac8cd7cd9cd27b1a9ecd71a2c5d64274df03eef0aee32d7a` |
| `lr1e-3-vc0p5-ps1` | `1e-3` | 0.5 | 1.0 | 0.35443637 | 46 / 2,995 | 28 / 64 | 0.01072 | 0.01967 | `d61e0bec563696ff255c915fd49f325a097243bf2bfa282bebb4e3d494e1142a` |
| `lr1e-3-vc0p02-ps1p25` | `1e-3` | 0.02 | 1.25 | -0.00682081 | 49 / 2,995 | 30 / 64 | 0.01155 | 0.01954 | `6579f17a394945d384d7f578ba2f1a090b7b5922280dc22309d3cb58b36009d0` |
| `lr1e-3-vc0p02-ps1p5` | `1e-3` | 0.02 | 1.5 | -0.01221064 | 48 / 2,995 | 29 / 64 | 0.01178 | 0.01790 | `acdc552d405498f10681d9e5f2fe9872bc36a18ccd5f700249155aaec67aa93c` |
| `lr1e-3-vc0p05-ps2` | `1e-3` | 0.05 | 2.0 | 0.00107962 | 47 / 2,995 | 31 / 64 | 0.01220 | 0.01963 | `34cd78edf4c10f3398cc8ed798b08ae8e1d3caecacbaa1520a697b15c620f2ad` |

No arm reached 60 flips, so the predeclared gate failed without a winner. The added policy-scale arms also showed why this fixed-CDF count should not be reused as an update-distance or promotion metric. From the policy-scale 1.0 endpoint to 2.0, mean action TV increased monotonically in the observed sequence while fixed-CDF flips decreased from 50 through 49 and 48 to 47. A single fixed CDF point can cross categorical boundaries in either direction and is not monotone in distributional distance.

Every artifact root begins with `D:\mtg-kernel-xmage-cp7-outcome-iter2-base1040001-std-epbal-`; the table gives the exact suffix and identity pins.

| Root suffix | Payload SHA-256 | Native-state SHA-256 | Model-parameter SHA-256 |
| --- | --- | --- | --- |
| `lr5e-4-vc0p05-ps1-v1` | `224afedd4dc42f3e1f95833a491299b48977a48e3e9bd18f615f414180df7824` | `925733829b95f379f5ad38498369e9e743254217e1435ada7f04fb600b2c9889` | `8b90b14c3f7e08aa29d9e8e6d8bf8ee5f62c68ace3a4df860e54fcbdb8b7fb9f` |
| `lr1e-3-vc0p05-ps1-v1` | `5f97605c1bf26270f32a6a25055d45bfd05fae05f50bda95f71807da66e258c9` | `1e3513ec15ee53a8b7200fcfaf1c928477885b1ec94b8f57b08d38374ca527b0` | `86f190a59b162f251cf5c014396409be29809d1ace7219181e06831fadf87565` |
| `lr1e-3-vc0p1-ps1-v1` | `060342886a2d8bec5b352c2c4f4daa08461cee1022b62ce9d3fda5af68947bb5` | `04721f93838984a77ee193a9f0cc2a6790d4c7da7389c54032ae18ddb2ef9077` | `7296f1d87740c9342c1eee5081d8aed428ab3ac4558e61bb234c1f57509f996a` |
| `lr1e-3-vc0p5-ps1-v1` | `9055f0154ca0ee8451fc966f6d40a8205bf902779a6f8f882dfe95d270424b39` | `3e8e36ddf0c8aa928ed8de003236eb4ff2336cc57b9ea8b636ff83ba93fa5b2e` | `18c2de9b9d8ae8de507655613cc8392aecc8c462d05ac5725b27fab9ef70566f` |
| `lr1e-3-vc0p02-ps1p25-v1` | `35c0e2ba379985159ab3a5d1b320d78cd009fb044eea1e8da068101812808dba` | `b173d90b5793826fa47abb48b9b3dcc215c6239feeed83aa11e6b333f9611374` | `c471b845a29357b039f1cfbdde797799aa9c7f2f87b75b85cceadfc7665ab864` |
| `lr1e-3-vc0p02-ps1p5-v1` | `c967de5896193525a6f16e0658d244ed707ed73ad3459d69909dfeca07bfa974` | `339ce7dd158d4e488c34ad2220dfaddcb40545efbc554ae144ac5126ced7c849` | `162651b07e1fb4fa79dee8905709c453485dd3465270369deebff28a3397ddeb` |
| `lr1e-3-vc0p05-ps2-v1` | `fa6d7c8bfb89c7e4217cc883a1ad1a815e8f230c0df6045f64199ffb5e590d62` | `a0465a2181b0e6adb45aba94a9ca12f09e45c695f82bc282f1ee853a32bdd563` | `c9ece235a1580ddfa01cb7643a543ad255c6b9b036093793a8752b47c64a05be` |

All payloads are 14,771,928 bytes. The manifest SHA-256 values in the sweep table and the payload, native-state, and model-parameter hashes in the identity table were rechecked directly against the local artifacts.

## Iteration 2 ps2 smoke and fresh matched evidence

After the fixed-CDF gate was found unsuitable, policy-scale 2.0 was allowed to proceed as an experimental rapid-validation candidate. This was a deliberate live probe, not a gate pass or promotion. Its four-pair base-1040001 run was an integration smoke on reused seeds.

| Base seed | Policy | Pairs | Result | On play | On draw | Candidate sweeps | CP7 sweeps | Split pairs | Turns | Rust steps | Physical decisions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1040001 | ps2 integration smoke | 4 | 2-6 (25.00%) | 0-4 | 2-2 | 0 | 2 | 2 | 105 | 822 | 713 |
| 1050001 | ps2 child | 32 | 22-42 (34.375%) | 9-23 | 13-19 | 2 | 12 | 18 | 846 | 6,974 | 5,643 |
| 1050001 | Parent `706b3a...` | 32 | 22-42 (34.375%) | 9-23 | 13-19 | 2 | 12 | 18 | 844 | 6,884 | 5,555 |
| 1060001 | ps2 child | 32 | 28-36 (43.75%) | 13-19 | 15-17 | 2 | 6 | 24 | 840 | 6,621 | 5,539 |
| 1060001 | Parent `706b3a...` | 32 | 26-38 (40.625%) | 12-20 | 14-18 | 1 | 7 | 24 | 828 | 6,533 | 5,475 |
| Pooled fresh | ps2 child | 64 | 50-78 (39.0625%) | 22-42 | 28-36 | 4 | 18 | 42 | 1,686 | 13,595 | 11,182 |
| Pooled fresh | Parent `706b3a...` | 64 | 48-80 (37.50%) | 21-43 | 27-37 | 3 | 19 | 42 | 1,672 | 13,417 | 11,030 |

The policies tied on base 1050001 and ps2 gained two wins on base 1060001. In the direct paired outcome comparison, base 1050001 had `G=1, L=1, T=62`, base 1060001 had `G=2, L=0, T=62`, and the pooled result was `G=3, L=1, T=124`. The one-sided exact sign-test value is `p=0.3125`. Although 46 of 128 matched trajectories diverged, only four paired game outcomes differed. The two-win, 1.5625-percentage-point pooled lift is therefore weak, unstable evidence, and both policies remained far below 50% against CP7.

The ps2 candidate failed promotion. It remains a verified experimental artifact only, and centered-v2 parent manifest `706b3a...` remains the retained training parent.

## Iteration 2 validation evidence

The focused Rust validation recorded the following exact results:

- `native_checkpoint_shadow_stdio_v1::tests::`: 16 passed, 0 failed, 4 ignored.
- `native_xmage_cp7_outcome_reinforce_v1::tests::`: 6 passed, 0 failed, 2 ignored.
- `native_policy_train_step_v1::tests::`: 22 passed, 0 failed, 0 ignored.
- `iterative_outcome_export_repeats_exact_parent_identity_on_all_rows_v2`: 1 exact focused pass within the shadow-scorer filter.
- `outcome_export_write_failure_poisoning_prevents_retry_v1`: 1 exact focused pass with an injected export flush failure. The failed request poisoned the service, wrote no retry row, and every subsequent direct request failed with `export_poisoned`.
- `external_outcome_corpus_passes_strict_loader_v1`: 1 pass when explicitly enabled against corpus SHA-256 `b7567739...`.
- `external_outcome_parent_load_preserves_full_adam_state_v1`: 1 pass when explicitly enabled against parent manifest `706b3a...`.

The three module filters plus the two explicitly enabled external fixtures account for 46 passing tests. Four unrelated external-authority fixtures remained ignored. `cargo test --manifest-path mtg-kernel/Cargo.toml native_xmage_cp7_outcome_reinforce_v1::tests:: --no-run` also passed. Both old raw and standardized outcome derivatives continued to pass `verify`, demonstrating legacy manifest compatibility. The poison-fix re-review confirmed that outcome-reset, teacher step/terminal, and outcome step/terminal persistence failures all poison the service without rollback; `run_jsonl_v1` emits and flushes the failure response once, then returns an I/O error instead of consuming another request. Finally, `cargo fmt --check` and `git diff --check` passed on the Rust worktree, and `git diff --check` passed on this XMage worktree after the documentation update.

## Iteration 3 experimental continuation

Iteration 3 deliberately continued from the already rejected iteration-2 policy-scale 2.0 artifact to test whether another on-policy warm-start update could reverse its weak live result. This was an experimental branch, not a change to the retained parent.

The Adam-step-2 parent, manifest `34cd78edf4c10f3398cc8ed798b08ae8e1d3caecacbaa1520a697b15c620f2ad`, played 64 seat-swapped pairs at base seed 1070001 and scored `56-72`. The schema-v2 export contains 5,495 decision rows, 128 terminal rows, and 4,616 physical groups.

- Corpus: `C:\Users\Jack\AppData\Local\Temp\mtg-cp7-outcome-iterative-v1\parent34cd-base1070001-pairs64.jsonl`
- Bytes: 104,202,166
- Corpus SHA-256: `5e6b86824fb4ec2fbfbebd931af4c9b3ea357b63e78ca71a759be072953ed01f`

The child inherited the complete Adam-step-2 parent state and made one full-corpus update with learning rate `1e-3`, value coefficient `0.05`, and policy scale `2.0`, ending at Adam step 3.

| Artifact field | Exact value |
| --- | --- |
| Root | `D:\mtg-kernel-xmage-cp7-outcome-iter3-base1070001-std-epbal-lr1e-3-vc0p05-ps2-v1` |
| Manifest SHA-256 | `feab2ea3a44d59c7c82d6dec683a49b8d5fcb0d3262b613cb151d66427086c73` |
| Payload SHA-256 | `4b94a58306a13eb12228b4fc91107a0564631a53600bf7350e0efec2cc23b654` |
| Native-state SHA-256 | `ad9aa56b2767d76e3df449bc66a9b968e3d05d5e67b6aa6269b6961ceb179415` |
| Model-parameter SHA-256 | `efd1e2b2aeb86de7544dc423f7bc29de7e5c84cebe96d9653f9b3b9138d172fe` |
| Adam step | 3 |

The corrected offline guard replayed the parent exactly on `5,495 / 5,495` decisions. It reported action-TV p90 `0.0179`, an affected rate of `46.09%`, and 30 argmax changes across the 5,495 decisions. Mean total objective decreased from `0.1034777305` to `0.1014655242`, but selected-action NLL increased from `0.5093699159` to `0.5142263608` and value MSE worsened from `0.7976085078` to `0.9809876624`. These are optimization and policy-change measurements, not play-strength evidence.

## Selected-action projection repair and excluded runs

The first iteration-3 live attempts exposed a harness alignment defect. At an active-player, empty-stack postcombat priority point, XMage could expose an additional legal action, such as a Great Furnace mana ability, that was absent from the Rust menu even though the Rust-selected action itself was present exactly. The old fallback passed in XMage without consuming the Rust-selected row, leaving the Rust cursor behind for a later decision.

The repaired harness retains strict full-menu matching as the normal path. A narrow selected-action projection is permitted only for the model-controlled candidate at the active-player, empty-stack postcombat rendezvous. It maps the Rust-selected semantic to exactly one XMage ability, validates the original card identity, card database identity, zone, translated zone-change count, owner, and controller, advances the exact Rust-selected index once, and executes that same XMage action. It never substitutes a different action. The admitted forms are pass, an ordinary spell cast from hand, an ordinary land play from hand, and the proven red Mountain mana action. Generated-token sources, absent or ambiguous mappings, and other activation forms fail closed. Every leg records its projection count and labels its alignment.

The earlier attempts are excluded from evaluation:

- Base 1080001 failed at episode 0 under the old cursor behavior.
- Base 1090001 remains invalid because phase-cursor drift leaves Rust in the opponent's Declare Attackers/Main2 transition when XMage reaches the Mountain owner's next Precombat Main. The Mountain remains validly in the Rust hand. This is a phase-alignment defect, not a land-legality or selected-action-projection case.
- The original base-1100001 block completed, but its results were not reused after the harness behavior changed. The final-harness rerun reproduced its aggregate result exactly.
- The original base-1110001 attempt failed at episode 2 under the old cursor behavior. A separate concurrent restart was also discarded after an H2 database setup lock prevented a matched completion.

The gated base-1090001 diagnostic captured Rust immediately before and after policy step 64. The exact selected semantic was a P0 pass. Before applying it, Rust was on turn 4 with P1 active in Declare Attackers; after advancing, Rust was still on turn 4 with P1 active in Main2. Arena 4 was a P0 Mountain in hand in both snapshots, P0 had played zero lands that turn, and the next Rust menu correctly contained mana abilities plus pass for priority during P1's turn. XMage had already reached P0's next Precombat Main and attempted to play that Mountain. This proves phase-cursor drift and rules out native Mountain identity or land-count corruption at the failure point.

Focused replays cleared the original base-1080001 and base-1110001 menu-mismatch episodes under the restricted projection. A uniform-opponent base-1120001 pair completed with zero projections, confirming that the projection remains candidate-only.

## Iteration 3 final matched evidence

The rapid promotion rule required, across two fresh 32-pair blocks, paired `G-L >= 4`, a one-sided exact sign-test value no greater than `0.10`, and a paired net of at least `-2` in each physical-seat stratum. The retained Adam-step-1 centered-v2 checkpoint, manifest `706b3aa80ec7a3c067d458fef06bb2237320543f202fb2349c5cb885975fdbbb`, was the rollback control.

| Base seed | Adam-step-3 child | Retained control | Paired G/L/T | P0 G/L/T | P1 G/L/T | Child/control projections | Evidence label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1100001 | 28-36 (43.75%) | 30-34 (46.875%) | 0/2/62 | 0/1/31 | 0/1/31 | 0/0 | No selected-action projection |
| 1110001 | 27-37 (42.1875%) | 28-36 (43.75%) | 1/2/61 | 1/0/31 | 0/2/30 | 0/2 | Control used selected-action projection |
| Pooled | 55-73 (42.96875%) | 58-70 (45.3125%) | 1/4/123 | 1/1/62 | 0/3/61 | 0/2 | Projected diagnostic overall |

The child lost three wins and `2.34375` percentage points relative to the retained control. Its pooled paired result was `G=1, L=4, T=123`, giving one-sided exact `p=0.96875`. The P0 paired net was zero and the P1 net was `-3`. It therefore failed all three promotion conditions: pooled paired net, exact significance, and the P1 seat floor.

Base 1100001 is exact no-projection evidence and already favors the retained control. Base 1110001 is valid as a selected-action-projection diagnostic because the control required two exact selected-action projections; it is not full-menu promotion evidence. The combined direction is adverse, so iteration 3 is retired rather than expanded. The Adam-step-3 child is not promoted, the rejected Adam-step-2 branch is not revived, and centered-v2 Adam-step-1 manifest `706b3a...` remains the retained outcome-training parent.

## Decision and next measurement

- Drop generation 256 and retain generation 384 as the live baseline.
- Reject the full derivative and all three smaller behavior-clone candidates.
- Close CP7 imitation as the current improvement route. Offline action agreement is not a sufficient objective for winning games.
- Record outcome v1 as completed: candidate-controlled generation-384 decisions, physical grouping, and natural terminal rewards were exported from actual XMage-versus-CP7 games and trained through the canonical value-baseline objective.
- Reject all five outcome v1 derivatives. Learning-rate and value-coefficient scaling changed optimization metrics and live trajectories but left every development result at `8-8`.
- Record centered outcome v2 as the first promising policy-improvement signal: `59-69` pooled versus matched generation 384 at `56-72` across two fresh blocks.
- Do not promote centered v2 from this evidence. The lift is three wins, appears only on the play, and reproduced as a tie rather than another lift in the second block.
- Record iteration 2 as a completed warm-start implementation with exact parent-state inheritance, on-policy schema-v2 corpus binding, seven verified Adam-step-2 artifacts, and two fresh matched blocks.
- Retire the fixed-CDF flip threshold as a promotion gate. It failed to select an arm and was empirically non-monotone with action-distribution movement. Any replacement replay gate must use a declared distance statistic whose interpretation does not depend on one fixed categorical CDF point.
- Record policy-scale 2.0 as an experimental live candidate, not a promoted checkpoint and not an offline-gate winner.
- Reject policy-scale 2.0 for promotion. Its fresh result was `50-78` versus the matched parent at `48-80`, with paired `G=3, L=1, T=124` and one-sided exact `p=0.3125`.
- Record iteration 3 as a completed experimental continuation from the rejected Adam-step-2 policy-scale 2.0 branch, with a 128-episode on-policy corpus and one verified warm-start update to Adam step 3.
- Reject the Adam-step-3 child. It scored `55-73` versus the retained control at `58-70`, with paired `G=1, L=4, T=123`, one-sided exact `p=0.96875`, and a P1 paired net of `-3`.
- Treat blocks containing selected-action projection as diagnostic evidence rather than exact full-menu promotion evidence. Keep the projection restricted, fail closed, and explicitly counted.
- Keep base 1090001 excluded until phase identity is carried and validated across the shadow bridge.
- Retain centered-v2 aggressive manifest `706b3aa80ec7a3c067d458fef06bb2237320543f202fb2349c5cb885975fdbbb` as the current outcome-training parent. Do not roll the live parent forward to either experimental child.

This remains one deck mirror against XMage CP7. Pro-level status would require substantially stronger evidence across decks, matchups, hidden-information decisions, sideboarding or match structure, and stronger external opponents. No such claim is warranted here.
