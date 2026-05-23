# 2026-05-09 Spy Belief Representation

Goal: test whether the planned archetype-belief representation surface helps Spy learn without heuristic rewards.

## Nine-Archetype Belief Wiring

Implementation:

- `StateSequenceBuilder.TrainingData.NUM_ARCHETYPES` widened from 4 to 9 to match `DeterminizationSampler.pauperDefaults()`.
- Archetype-label signatures now cover:
  - Wildfire, Rally, Affinity, Elves
  - SpyCombo, Burn, Terror, CawGates, Faeries
- `mtg_transformer.py` now defaults `NUM_ARCHETYPES=9`.
- Checkpoint loading now skips tensors with incompatible shapes, so old four-class checkpoints can seed a nine-class belief run. The skipped optimizer state is cleared in `py4j_entry_point.py`.
- Belief logging/export comments were updated to the same nine-class order.

Validation:

- RL module compile passed.
- Accepted four-class Spy checkpoint loaded into the nine-class model by skipping only:
  - `belief_head.3.weight`
  - `belief_head.3.bias`
- `DeterminizationSamplerTest` passed with all nine archetypes loaded.

## Belief9 RL Smoke

Profile:

- `Pauper-Spy-Combo-Belief9RL-20260509`
- Seeded from `Pauper-Spy-Combo-FastT5Contrast-20260506`
- Run: `20260509_spy_belief9_rl_r16_e1000`
- 1000 episodes, 16 runners, terminal rewards only
- `NUM_ARCHETYPES=9`
- `BELIEF_LOSS_COEF=0.5`
- No MCTS/ISMCTS, no heuristic rewards

Training result:

- Completed cleanly with trainer `rc=0`.
- Final status: 1000 episodes at about 1.1 eps/s.
- Health counters stayed clean: 0 game kills, 0 activation failures, 0 GPU OOMs, 0 Python errors, 0 model NaNs.
- Value probe collapsed late: `value_accuracy.csv` ended at `0.0000` for episodes 960, 970, 990, and 1000.

Reduced CP1 eval:

- `20260509_spy_belief9_rl_cp1_eval4`: `3/16`
- `20260509_spy_belief9_rl_cp1_eval4_logged`: `2/16`

Logged action health from `20260509_spy_belief9_rl_cp1_eval4_logged`:

- `spy_cast_opportunities`: 17
- `spy_cast_hidden_land_opportunities`: 16/17
- `spy_casts`: 5/17
- `spy_casts_with_hidden_lands`: 4/5
- `dread_return_flashback_selected`: 4
- `premature_dread_flashback_no_lotleth_graveyard`: 4/4
- `premature_dread_flashback_not_combo_ready`: 4/4

Conclusion:

The nine-archetype belief auxiliary branch is correctly wired but not promotable in this form. It does not improve CP1 winrate over the accepted sanity baseline, does not fix Spy timing, and appears to destabilize value learning under the 1k terminal-RL recipe.

Do not scale this as a plain auxiliary loss. If belief is revisited, it should be part of a real card-level belief/determinization search loop rather than another small terminal-RL variant.

## 2026-05-12 Thesis-Clean Label Fix

The original `StateSequenceBuilder.computeArchetypeLabel` scanned opponent library and hand when assigning the belief auxiliary target. That is hidden information and should not be used in thesis-clean training.

Patch:

```text
computeArchetypeLabel now scans only opponent public zones:
battlefield, graveyard, exile, and stack.
```

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java -Dexec.mainClass=mage.player.ai.rl.DeterminizationSamplerTest
```

`DeterminizationSamplerTest` passed all 7 checks.
