# Rapid Rally CP7 Policy Improvement

## Status

The behavior-cloning route and the first terminal-outcome update sweep are rejected. On 32 fresh matched base-990001 pairs the full behavior-cloning derivative scored `23-41` (35.9375%) against CP7, while generation 384 scored `30-34` (46.875%) on the same pairs. Three smaller behavior-cloning updates then scored no better than `8-8` in the eight-pair development screen. The large optimization-split imitation gain did not transfer to live play strength, and reducing update distance did not produce a positive signal.

Outcome v1 then trained five one-update derivatives from 32 actual generation-384-versus-CP7 pairs at base seed 1010001. Generation 384 scored `27-37` while producing that corpus. Every derivative scored exactly `8-8` on the base-960001 eight-pair development screen. Larger steps changed policy decisions and trajectory lengths, but none changed the win count, so no outcome derivative advanced to a fresh 32-pair block.

Centered outcome v2 changed the estimator rather than only its scale. It froze the source value baseline, centered and standardized advantages, and gave each contributing episode equal total objective mass. The aggressive arm passed an offline policy-change gate, repeated `8-8` on the development screen, and then scored `30-34` against CP7 versus matched generation 384 at `27-37` on fresh base-1020001 pairs. On a second fresh block at base 1030001, both policies scored `29-35`. Pooled across the 128 fresh games, the aggressive derivative scored `59-69` versus generation 384 at `56-72`, a three-win and 2.34375-percentage-point lift.

This is the first promising matched signal in the campaign, but it is insufficient for promotion. Both policies remained below 50% against CP7, one fresh block improved while the other tied, and the evidence still covers only one deterministic Rally mirror. The result motivates an iterative warm-start outcome update, not a CP7-superiority or pro-level claim.

The archived-generation screen also failed to improve generation 384. Generation 384 remains the live baseline: it is approximately CP7-competitive in this one Rally mirror and has not demonstrated CP7 superiority. Nothing here supports a pro-level play claim.

## Code provenance

XMage repository commits:

- `a6e9d2d223f9bb87aa985e6ce56c98236b9a6a5e`: exact Rally-versus-CP7 anchor
- `988482462bb473df34ec24dafc5dc6f9fee104f0`: recorded exact anchor result
- `08105f238aa9e71d764073c9beb85bffe30b9c9d`: CP7 teacher-export pass-through
- `ecf62adc5aa0116ccfe8ce64fd548c954bcb2fb1`: checkpoint-generation selector
- `e1d4ce80ccebca26095b5694a89c9468cd1f41ab`: selected-generation identity validation

Rust repository commits:

- `54e7fdc8f11dac2f2c44d1dd380b1d6d2c37b494`: exact checkpoint shadow scorer
- `208738a2605907602ae5550accfb74ad155e2d79`: partial Chain Lightning copy-payment match
- `af05f8ba4aa6f6e11da17ddb9d5d816cbac7345a`: mapped CP7 teacher export
- `0fba419f79e5ddc2eebf906d068ebea815abae04`: validated checkpoint-generation selection
- `1a938364707ec1e93eb93995518329cae2924f6f`: strict CP7 behavior-clone training and scoring
- `41fdd71a7195841545b81fe6f8c7e7d4e6c61669`: candidate-controlled XMage outcome export, strict terminal REINFORCE/value training, derivative verification, and checkpoint-scorer authority

The behavior-cloning trainer and derivative-scorer integration were uncommitted when these games ran. The Windows scorer executable used for both behavior-cloning blocks was:

- Path: `C:\Users\Jack\IdeaProjects\mtg-kernel-entropy-smoke-v1\target-windows-cp7-bc\release\checkpoint_shadow_stdio_v1.exe`
- SHA-256: `b3ab1d2293dd7bf3bbcff5d7333187431c6bc414cd4e6c6bfa7992ccd41edbd2`

The executable hash pins the live behavior. Artifact-verifier hardening is reviewed separately below.

The post-commit scorer used for the strict small-update screens was 4,318,720 bytes with SHA-256 `481fdcc379db32409aa4824d4d2145e84b3d93bc9b915f82e535d3c9d186dfa1`.

The Windows scorer used for the outcome-derivative screens was 4,577,792 bytes with SHA-256 `9b1397be8acde45ad1a97eefed90ce2a72a576811f004326cf33e475382ab728`.

The Windows scorer used for centered outcome v2 was 4,634,624 bytes with SHA-256 `559c2704ded38a6f884f4d804aab95c830e7ee4cb11e02495c0c6299005b408c`.

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

A read-only review found the transform math, frozen weighted train step, legacy compatibility, and scorer identity sound. The optional transform field is absent when loading raw v1 manifests, preserving their canonical bytes and load compatibility. The scorer continues to report exact source generation-384 pins plus derivative manifest, payload, native-state, and model hashes. Focused transform tests passed `4/4` with one external-corpus test ignored, the exact frozen-objective core test passed, `git diff --check` passed, and the old raw `lr3e-5-vc0p5` artifact reverified unchanged. The first v2 artifacts subsequently passed strict manifest/payload verification and loaded through the scorer used for live XMage. The centered-v2 Rust changes were uncommitted when these games ran, so the executable and artifact hashes pin the evaluated behavior.

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

## Decision and next measurement

- Drop generation 256 and retain generation 384 as the live baseline.
- Reject the full derivative and all three smaller behavior-clone candidates.
- Close CP7 imitation as the current improvement route. Offline action agreement is not a sufficient objective for winning games.
- Record outcome v1 as completed: candidate-controlled generation-384 decisions, physical grouping, and natural terminal rewards were exported from actual XMage-versus-CP7 games and trained through the canonical value-baseline objective.
- Reject all five outcome v1 derivatives. Learning-rate and value-coefficient scaling changed optimization metrics and live trajectories but left every development result at `8-8`.
- Record centered outcome v2 as the first promising policy-improvement signal: `59-69` pooled versus matched generation 384 at `56-72` across two fresh blocks.
- Do not promote centered v2 from this evidence. The lift is three wins, appears only on the play, and reproduced as a tie rather than another lift in the second block.
- Use the aggressive centered-v2 checkpoint as the warm start for the next controlled outcome iteration. Collect outcomes under the updated policy, retain matched generation-384 or prior-checkpoint controls, and require another positive fresh signal before promotion.

This remains one deck mirror against XMage CP7. Pro-level status would require substantially stronger evidence across decks, matchups, hidden-information decisions, sideboarding or match structure, and stronger external opponents. No such claim is warranted here.
