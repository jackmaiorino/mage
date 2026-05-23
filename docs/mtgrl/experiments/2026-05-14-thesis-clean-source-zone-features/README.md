# Thesis-clean source-zone candidate features - 2026-05-14

## Question

The accepted Affinity logs showed a generic action-source mistake: one loss
hard-cast `Dread Return` from visible hand in a low-graveyard state, with the
value head still positive. Can a narrower generic representation of candidate
source zone help without naming Spy cards or adding strategy rules?

Thesis boundary: clean. The feature only marks generic source zones for action
candidates: hand, graveyard, battlefield, exile, and spell-not-from-hand. It
does not inspect action text, card names, Spy terminal states, heuristic rewards,
or matchup-specific rules.

## Patch

Added default-off:

```text
RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE=1
```

For `ACTIVATE_ABILITY_OR_SPELL` candidates, the policy candidate feature vector
now optionally includes:

```text
source currently in hand
source currently in graveyard
source currently on battlefield
source currently exiled
spell not from hand
declared ability zone differs from current source zone
```

Validation:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Training

Profile:

```text
Pauper-Spy-Combo-Value-SourceZonePolicyPath-20260514
```

Start checkpoint:

```text
Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260514_source_zone_policy_path_registry.json
```

Settings:

```text
TOTAL_EPISODES_DELTA=128
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE=1
RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
DISTILL_POLICY_PATH_ONLY=1
VALUE_LOSS_COEF=0
REFERENCE_POLICY_KL_COEF=0.50
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
RL_HEURISTIC_STEP_REWARDS=0
```

Training completed cleanly:

```text
episodes=128
MCTS activations=0
model_latest.pt sha256=4E8514CDE110C6D4FFBDACFEBEC4F9967A0C9DB1565F8055CE5B018DE583E0F7
```

## Pressure gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_source_zone_policy_path_pressure_g8
skill=7
games_per_matchup=8
opponents=Mono Red Rally,Grixis Affinity
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 3 | 8 | 37.50% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Combined | 4 | 16 | 25.00% |

## Decision

Rejected. Do not submit to HPC.

The source-zone signal is thesis-clean and worth keeping default-off as
infrastructure, but short policy-path adaptation does not improve the hard
pressure surface. The failure also weakens the hypothesis that the logged
hard-cast `Dread Return` mistake is the main Affinity bottleneck.
