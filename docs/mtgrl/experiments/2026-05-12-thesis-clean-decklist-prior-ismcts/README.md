# Thesis-Clean Deck-List Prior ISMCTS

Date: 2026-05-12

## Question

Was generic MCTS/ISMCTS underperforming partly because its determinization prior sampled from the hard-coded 9-deck Pauper default while the local thesis eval pool contains only 4 decks?

This is thesis-clean: the sampler uses the generic opponent deck pool, not Spy action names, Spy reward shaping, or card-specific combo rules.

## Implementation

Added an optional determinization sampler source:

```text
ISMCTS_ARCHETYPE_DECK_LIST=<deck-list-file>
```

`DeterminizationSampler.loadFromDeckListFile(...)` reads the same deck-list format used by the training/eval harness, resolves relative `.dek` entries against the list's parent directory, and derives generic archetype names from deck filenames.

`ComputerPlayerRL` now prefers that sampler when `ISMCTS_ENABLE=1` or `MCTS_TRAINING_ENABLE=1`; otherwise it falls back to `DeterminizationSampler.pauperDefaults()`.

Validation:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java "-Dexec.mainClass=mage.player.ai.rl.DeterminizationSamplerTest"
```

The smoke test loaded the 4-deck thesis eval pool as:

```text
[SpyCombo, JundWildfire, MonoRedRally, GrixisAffinity]
```

## Flat MCTS Gate

Registry:

```text
local-training/local_pbt/autonomous_runs/20260512_thesis_clean_decklist_prior_ismcts_eval/eval_registry.json
```

Run:

```text
20260512_accepted_decklist_prior_flat_mcts_spy_cp7_eval4
```

Settings:

```text
profile=Pauper-Spy-Combo-Value
skill=7
games-per-matchup=4 requested
MCTS enabled
MULTI_PLY_MCTS=0
MCTS_ITERATIONS=8
MCTS_DETERMINIZATIONS=1
MCTS_ROLLOUT_DEPTH=0
MCTS_SKIP_TOP_PROB=0.80
ISMCTS_ARCHETYPE_DECK_LIST=decklist.active_profile_pool_thesis_eval_unique_20260510.txt
```

Stopped early after the first two completed chunks:

```text
Overall: 0/4 = 0.00%
Spy mirror:      0/2, 130 MCTS activations
Mono Red Rally:  0/2,  84 MCTS activations
```

The log confirmed the intended sampler:

```text
samplerSource=deck-list:...decklist.active_profile_pool_thesis_eval_unique_20260510.txt
sampler=[SpyCombo, JundWildfire, MonoRedRally, GrixisAffinity]
```

## Verdict

Rejected as a quick win.

The deck-list prior is cleaner and should remain available for future search diagnostics, but it did not repair the current flat value-MCTS target. Early stopping was appropriate because the first completed 4 games were all losses despite heavy MCTS activation.

Do not spend HPC on scaling this flat MCTS configuration. If search is revisited, it needs a different target or rollout policy, not just a better archetype prior.

## Terminal Rollout-Root Gate

Registry:

```text
local-training/local_pbt/autonomous_runs/20260512_thesis_clean_decklist_prior_rollout_ismcts_eval/eval_registry.json
```

Run:

```text
20260512_accepted_decklist_prior_rollout_ismcts_spy_cp7_eval1
```

Settings:

```text
profile=Pauper-Spy-Combo-Value
skill=7
games-per-matchup=1 requested
ISMCTS_RANDOM_ROLLOUT_ROOT=1
ISMCTS_ROOT_DETERMINIZATIONS=1
ISMCTS_ROOT_ROLLOUTS_PER_ACTION=1
ISMCTS_ROOT_SEARCH_TIMEOUT_MS=1000
ISMCTS_ARCHETYPE_DECK_LIST=decklist.active_profile_pool_thesis_eval_unique_20260510.txt
```

Stopped after the first three completed games:

```text
Overall: 0/3 = 0.00%
Spy mirror:       0/1, 110 MCTS activations, 121.3 sec
Mono Red Rally:   0/1, 140 MCTS activations, 106.6 sec
Grixis Affinity:  0/1,  28 MCTS activations,  66.5 sec
```

The log confirmed:

```text
randomRolloutRoot=true
sampler=[SpyCombo, JundWildfire, MonoRedRally, GrixisAffinity]
```

Verdict: rejected. The value-free rollout backend remains too slow and too weak even with the correct 4-deck prior. The failure is not just the hard-coded 9-deck prior.
