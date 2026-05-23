# Thesis-clean generic action-class candidate features - 2026-05-14

## Question

Recent hard-matchup counterfactual corpora are dominated by low-level mana and
land sequencing labels. The policy candidate tensor marks pass actions, spell
status, source type, and mana cost, but it did not expose a clean generic flag
for ability candidates that are mana abilities or land plays.

Can default-off generic action-class candidate features help the model adapt to
those sequencing decisions without Spy-specific teaching?

This branch stays inside the thesis boundary:

- terminal win/loss returns only;
- no heuristic step rewards;
- no Spy terminal mode;
- no Spy/card-name action facts;
- no action-text/card-name regex labels;
- no MCTS or selective search gates.

## Data Diagnosis

The selected target labels in recent thesis-clean counterfactual corpora were
low-level surprisingly often:

```text
branchtraj_broad: rows=16, low_level_best=3/16
branchtraj_affinity: rows=16, low_level_best=8/16
baselinealt_affinity: rows=22, low_level_best=8/22
baselinealt_unique: rows=66, low_level_best=36/66
```

`low_level_best` counts selected best actions classified as pass, mana ability,
or land/object play by generic text shape for diagnosis only.

## Patch

Added default-off:

```text
RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE=1
```

For `ACTIVATE_ABILITY_OR_SPELL` ability candidates, the feature writes generic
class flags:

```text
mana ability
land play
spell
other activated ability
uses stack
```

No card names or deck-specific tactics are encoded.

Compile:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Full-Update Adaptation

Profile:

```text
Pauper-Spy-Combo-Value-ActionClass-20260514
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260514_action_class_features_registry.json
```

Training:

```text
TOTAL_EPISODES_DELTA=128
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
REFERENCE_POLICY_KL_COEF=0.50
RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE=1
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

Result:

```text
episodes=128
model_latest.pt sha256=C1563A8CF3E40AABD3EB3CD4A3C81922A8BA5D07DF01CC475749B2113BA52F1D
```

Pressure gate:

```text
cdrive-eval-sweeps/20260514_action_class_features_pressure_g8
CP7 Mono Red Rally: 0/8
CP7 Grixis Affinity: 2/8
combined: 2/16 = 12.50%
```

## Verdict

Rejected. The feature may be useful infrastructure, but short full-policy
adaptation with the new candidate bits destroys Rally and does not improve the
combined pressure surface.

Do not submit this full-update branch to HPC.

## Containment Follow-Up

Containment run from accepted:

```text
Pauper-Spy-Combo-Value-ActionClassPolicyPath-20260514
```

Only the candidate-policy path is trainable:

```text
DISTILL_POLICY_PATH_ONLY=1
VALUE_LOSS_COEF=0
REFERENCE_POLICY_KL_COEF=0.50
RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE=1
```

Decision rule: gate on Rally/Affinity pressure before any broader eval or HPC
consideration.

Result:

```text
episodes=128
model_latest.pt sha256=A00C72430BF8C60B2CDA8BD0765DA1F394EF498CE253D81A740FB52DFAD2CFC2
```

Pressure gate:

```text
cdrive-eval-sweeps/20260514_action_class_policy_path_pressure_g8
CP7 Mono Red Rally: 0/8
CP7 Grixis Affinity: 2/8
combined: 2/16 = 12.50%
```

## Final Read

Generic action-class candidate features do not clear a local gate under either
full-policy adaptation or policy-path-only containment. Both variants preserve
some Affinity wins but completely lose Rally in this pressure sample.

Keep the feature default-off as a diagnostic/representation hook. Do not scale
this surface locally or on HPC without a different training objective.
