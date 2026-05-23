# Thesis-Clean Archetype Belief Features

Date: 2026-05-12

## Question

Can a generic public-information archetype posterior help the Spy policy improve without explicit Spy knowledge?

This is thesis-clean if the posterior is derived from public cards and deck priors only. It does not name Balustrade Spy, Dread Return, Lotleth Giant, or any Spy-specific tactic. The same feature mechanism applies to every trained deck.

## Implementation

`StateSequenceBuilder` now supports:

```text
RL_ARCHETYPE_BELIEF_FEATURES_ENABLE=1
```

When enabled, it uses `ArchetypeBeliefSampler` to classify the opponent archetype from public information and writes the posterior into existing archetype slots `40..48`. The feature is off by default.

## Clone

Created a fresh Spy clone from the accepted 2026-05-10 Affinity-pressure checkpoint:

```text
Pauper-Spy-Combo-Value-ArchetypeBelief-20260512
```

The clone reset counters and optimizer state. New feature columns `40:49` were zeroed in `input_proj.weight` and `critic_input_proj.weight` before training.

Accepted baseline hashes were rechecked after the experiment and remained unchanged:

```text
Pauper-Spy-Combo-Value model.pt        14E1A1DA4F3E
Pauper-Spy-Combo-Value model_latest.pt 72857AA2975A
Pauper-Wildfire-Value model.pt         217707FE2EF0
Pauper-Wildfire-Value model_latest.pt  6F72507293CC
Pauper-Rally-Anchor-Value model.pt     C4ADFD672072
Pauper-Rally-Anchor-Value model_latest.pt 5A04AFF24179
Pauper-Affinity-Anchor-Value model.pt  A692C64954F3
Pauper-Affinity-Anchor-Value model_latest.pt B7FD1930779E
```

## Training

Run id:

```text
20260512_thesis_clean_archetype_belief_spyclone_500ep
```

Settings:

```text
TOTAL_EPISODES=500
TRAIN_PROFILES=1
NUM_GAME_RUNNERS=8
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.34,3:0.33,7:0.33
RL_ARCHETYPE_BELIEF_FEATURES_ENABLE=1
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_ZONE_COUNT_FEATURES_ENABLE=0
RL_LIBRARY_COUNT_FEATURES_ENABLE=0
RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

Training completed cleanly at `500/500` episodes. No MCTS activations occurred.

## Reduced CP7 Eval

Run id:

```text
20260512_archetype_belief_spyclone500_spy_cp7_unique_eval16
```

Result:

```text
Overall: 22/63 = 34.92%
```

Per matchup:

```text
Spy mirror:       12/16 = 75.00%
Jund Wildfire:     5/16 = 31.25%
Mono Red Rally:    3/16 = 18.75%
Grixis Affinity:   2/15 = 13.33%
```

Accepted CP7 reference:

```text
108/242 = 44.63%
```

## Verdict

Rejected.

The branch preserved mirror competence but regressed sharply into hard-pressure matchups. This repeats the common failed pattern from representation-only and direct distillation branches: the model can still play some Spy mirrors, but does not improve the cross-board CP7 target that matters.

No CP1 expansion or HPC submission is warranted. Public archetype posterior features remain a valid generic representation idea, but this single-feature local branch does not clear the reduced gate.
