# Thesis-Clean Semantic Effect Flags

Date: 2026-05-13

## Objective

Test whether improving generic public-card semantic features helps the accepted Spy policy handle Affinity, without adding Spy-specific action rules, reward shaping, card-name filters, or search gates.

## Trigger

Compact CP7 logs against Grixis Affinity:

`local-training/local_pbt/cp7_eval_sweeps/20260513_accepted_affinity_cp7_logged4_compact`

Result: `0/4`.

Failure taxonomy:

| Chunk | Spy line | Main observed failure |
| --- | --- | --- |
| 001 | Cast `Balustrade Spy`, flashbacked `Dread Return` | Combo attempt happened with visible Affinity interaction and later collapsed into exile/board-control pressure. |
| 002 | Cast `Balustrade Spy`; later hardcast `Dread Return` | Post-mill conversion failed; `Dread Return` targeted `Generous Ent` instead of a lethal creature line. |
| 003 | No Spy cast | Slow setup into visible `Nihil Spellbomb` pressure. |
| 004 | Cast `Balustrade Spy`, flashbacked `Dread Return` targeting `Lotleth Giant` | Opponent activated `Nihil Spellbomb`; graveyard was exiled before `Dread Return` resolved. |

This suggested the Affinity bottleneck is not only hidden real lands in library. A visible public-board interaction problem is also present: the policy fires graveyard-dependent lines into open graveyard hate and artifact-sacrifice sweepers.

## Representation Finding

`StateSequenceBuilder` already includes:

- opponent battlefield permanents;
- card token IDs;
- text embeddings;
- coarse semantic effect flags.

However, the generic semantic flag extractor missed important existing XMage effect classes:

- `ExileGraveyardAllTargetPlayerEffect` did not set the existing `fExiles` flag, so `Nihil Spellbomb` was not cleanly marked as an exile effect through this path.
- `DamageAllEffect` did not set the existing `fDealsDamage` flag, so `Krark-Clan Shaman` was not cleanly marked as a sweep-damage effect through this path.

Patch:

- default-off gate: `RL_EXTENDED_EFFECT_FLAGS_ENABLE=1`;
- `StateSequenceBuilder.isExileEffect(...)` now covers explicit exile effects plus class names that encode graveyard exile or `AndExile` search effects.
- `StateSequenceBuilder.isDamageEffect(...)` now covers target/player/all damage effects plus other `Damage*` effect classes.

This is thesis-clean because it improves generic effect-category extraction for visible public cards. It does not name Spy, Dread Return, Lotleth Giant, Balustrade Spy, Nihil Spellbomb, or Krark-Clan Shaman in training logic.

## Eval-Only Probe

Run:

`20260513_accepted_semanticflags_affinity_cp7_eval16`

Setup:

- accepted checkpoint;
- current code with improved semantic flags;
- no retraining;
- Grixis Affinity CP7 only;
- 16 one-game chunks.

Result:

`1/16 = 6.25%`

Interpretation:

This is not a promotable eval-only improvement. The accepted model was trained with the old incomplete feature distribution, so flipping these generic state bits at inference time appears to create harmful distribution shift. The valid test is a retrain/adaptation clone.

## Retrain Clone

Profile:

`Pauper-Spy-Combo-Value-SemFlags-20260513`

Source:

accepted `Pauper-Spy-Combo-Value` checkpoint, `model_latest.pt` hash prefix `72857AA2975A`.

Training registry:

`local-training/local_pbt/thesis_clean/20260513_thesis_clean_semantic_flags_spyclone_registry.json`

Settings:

- terminal win/loss returns only;
- `RL_HEURISTIC_STEP_REWARDS=0`;
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`;
- `RL_ZONE_COUNT_FEATURES_ENABLE=0`;
- `RL_LIBRARY_COUNT_FEATURES_ENABLE=0`;
- `MCTS_TRAINING_ENABLE=0`;
- `ISMCTS_ENABLE=0`;
- `OPPONENT_SAMPLER=self`;
- `SELFPLAY_OPPONENT_TRAINING=1`;
- weighted Affinity-pressure opponent deck pool from 2026-05-10.

Run:

`20260513_thesis_clean_semantic_flags_spyclone_500ep`

Training result:

500/500 episodes completed cleanly. MCTS gate stayed inactive:

```text
MCTS_GATE: total=0 ... activations=0
```

Recent training winrate was unstable and ended weak:

```text
episode 326: rolling winrate ~= 0.23
episode 500: rolling winrate ~= 0.12
```

## Reduced Screens

### Narrow CP7 Affinity Screen

Run:

`20260513_semflags_spyclone500_affinity_cp7_eval16`

Result:

`5/15 = 33.33%`, with one 0/0 timed-out chunk.

This is a meaningful narrow Affinity improvement over the eval-only probe (`1/16`) and above the accepted Affinity CP7 rate (`13/62 = 20.97%`). It justified one reduced aggregate screen.

### Reduced CP1 Unique Screen

Run:

`20260513_semflags_spyclone500_unique_cp1_eval16`

Overall:

`30/64 = 46.88%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 15 | 16 | 93.75% |
| Jund Wildfire | 9 | 16 | 56.25% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 3 | 16 | 18.75% |

## Verdict

Rejected for promotion and HPC.

The generic semantic flag patch is worth keeping as a default-off representation option, but the 500-episode adaptation branch failed the reduced CP1 gate (`30/64`, below the `32/64` threshold and below accepted CP1). The narrow CP7 Affinity gain did not transfer to broad low-skill stability, and Rally/Affinity CP1 collapsed.

Do not run full CP3/CP7 aggregate or submit this branch to Zaratan. Future tests of `RL_EXTENDED_EFFECT_FLAGS_ENABLE=1` need a less destructive adaptation method, likely policy-improvement/distillation with an explicit stability gate, not plain terminal continuation.

## Reference-Anchored Adaptation

Question:

Can the same generic semantic flags be adapted with less drift by anchoring the policy to the accepted checkpoint?

Profile:

`Pauper-Spy-Combo-Value-SemFlagsAnchored-20260513`

Source:

accepted `Pauper-Spy-Combo-Value` checkpoint, `model_latest.pt` hash prefix `72857AA2975A`.

Training registry:

`local-training/local_pbt/thesis_clean/20260513_thesis_clean_semantic_flags_anchored_registry.json`

Settings:

- terminal win/loss returns only;
- `RL_HEURISTIC_STEP_REWARDS=0`;
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`;
- `RL_EXTENDED_EFFECT_FLAGS_ENABLE=1`;
- `REFERENCE_POLICY_KL_COEF=1.00`;
- `MCTS_REFERENCE_MODEL_PATH=.../Pauper-Spy-Combo-Value/models/model_latest.pt`;
- conservative PPO/update settings matching the previous reference-anchor probes;
- no MCTS/ISMCTS.

Run:

`20260513_thesis_clean_semantic_flags_anchored_128ep` via the local PBT orchestrator.

Training result:

128/128 episodes completed cleanly. MCTS stayed inactive:

```text
MCTS_GATE: total=0 ... activations=0
```

The 128-episode checkpoint hash prefix is `A8B38313532B`.

### Reduced CP1 Unique Screen

Run:

`20260513_semflagsanchored128_unique_cp1_eval16`

Overall:

`34/64 = 53.12%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 9 | 16 | 56.25% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 6 | 16 | 37.50% |

Read:

This clears the reduced CP1 gate that the unanchored 500-episode semantic-flags clone failed (`30/64`). Continue locally with a targeted CP7 Affinity screen before considering broader CP7/CP3 evaluation. It is still not promotable or HPC-worthy from CP1 alone.

### Targeted CP7 Affinity Screen

Run:

`20260513_semflagsanchored128_affinity_cp7_eval16`

Result:

`3/16 = 18.75%`

Read:

Rejected for promotion and HPC. The anchored adaptation preserved enough low-skill aggregate strength to pass the reduced CP1 gate, but it did not improve the hard interaction-heavy matchup it was meant to address. The accepted CP7 Affinity reference remains `13/62 = 20.97%`, and the unanchored semantic-flags clone reached `5/15 = 33.33%` narrowly while failing CP1. Do not continue this branch without a different adaptation objective.

## Checkpoint Interpolation Follow-Up

Question:

Can the narrow Affinity gain from the unanchored semantic-flags checkpoint be recovered while reducing CP1 drift by generic checkpoint interpolation with the accepted model?

This is thesis-clean and eval-only: it averages matching floating-point model tensors and does not add action rules, reward shaping, card-name filters, or search gates.

Profiles:

```text
Pauper-Spy-Combo-Value-SemFlagsSoup025-20260513
Pauper-Spy-Combo-Value-SemFlagsSoup050-20260513
Pauper-Spy-Combo-Value-SemFlagsSoup075-20260513
```

Sources:

```text
base: Pauper-Spy-Combo-Value/model_latest.pt
semantic source: Pauper-Spy-Combo-Value-SemFlags-20260513/model_latest.pt
```

Infrastructure note:

`scripts/run_cp7_eval_sweep.py` now hardlinks `cards.h2.mv.db` into each job DB directory before falling back to copy. This avoids repeated local disk pressure from identical H2 DB copies during chunked evals.

### Affinity CP7 Screen

Initial 8-game narrow screen:

| Soup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| 25% semantic | 1 | 8 | 12.50% |
| 50% semantic | 2 | 8 | 25.00% |
| 75% semantic | 2 | 7 | 28.57% |

Expanded screens:

```text
20260513_semflags_soup050_affinity_cp7_eval16: 3/16 = 18.75%
20260513_semflags_soup075_affinity_cp7_eval16: 4/16 = 25.00%
```

The 50% soup failed the accepted Affinity anchor (`13/62 = 20.97%`). The 75% soup cleared it narrowly, so it received reduced CP1 and non-Affinity CP7 screens.

### 75% Soup Reduced CP1

Run:

```text
20260513_semflags_soup075_unique_cp1_eval16e
```

Result:

```text
Overall: 33/64 = 51.56%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 7 | 16 | 43.75% |
| Grixis Affinity | 5 | 16 | 31.25% |

### 75% Soup Reduced CP7

Runs:

```text
20260513_semflags_soup075_affinity_cp7_eval16
20260513_semflags_soup075_nonaffinity_cp7_eval16
```

Combined result:

```text
Overall: 27/64 = 42.19%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 4 | 16 | 25.00% |

Accepted CP7 reference:

```text
Overall: 108/242 = 44.63%
Spy mirror: 43/56 = 76.79%
Jund Wildfire: 36/60 = 60.00%
Mono Red Rally: 16/64 = 25.00%
Grixis Affinity: 13/62 = 20.97%
```

Verdict:

Rejected for promotion and HPC. The 75% soup preserved the small Affinity gain and passed reduced CP1, but the Jund CP7 regression is too large and the aggregate remains below accepted. This closes the semantic-flags adaptation/interpolation line for now: generic public interaction flags are useful diagnostics, but the current adaptation methods trade one pressure matchup for another instead of improving the board broadly.
