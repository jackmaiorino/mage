# Thesis-Clean True Belief Labels

Date: 2026-05-12

## Question

Can the existing belief head become useful for generic hidden-information reasoning if it is trained on the true opponent archetype for every state, instead of only receiving labels after a public signature card appears?

This remains thesis-clean under the current framing: the label is a generic hidden-state target for the opponent deck archetype. It does not add Spy action features, Spy terminal shaping, card-name regex labels, selective MCTS gates, or heuristic rewards.

## Implementation

Shared-GPU belief inference was missing, so logged evals with `SharedGpuPythonModel` silently skipped `[BELIEF]` lines. Added:

```text
SharedGpuProtocol.OP_PREDICT_ARCHETYPE = 17
SharedGpuPythonModel.predictArchetype(...)
gpu_service_core.ProfileContext.predict_archetype(...)
gpu_service_host opcode 17 handler
```

Added opt-in true hidden-state belief targets:

```text
RL_BELIEF_TRUE_ARCHETYPE_LABELS_ENABLE=1
```

`RLTrainer` now sets a thread-local player-id -> archetype-label map from the known deck paths. `StateSequenceBuilder.computeArchetypeLabel(...)` uses that map only when the env flag is enabled; otherwise it keeps the public-signature behavior.

Validation:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
python -m py_compile gpu_service_core.py gpu_service_host.py
```

## Accepted Belief Calibration

Run:

```text
20260512_accepted_spy_belief_calibration_cp7_logged4_sharedgpu_belief
```

Result:

```text
CP7 logged4: 8/15 = 53.33% (one Wildfire game missing from summary)
Belief lines: 278
```

Turn-level belief accuracy:

```text
Grixis Affinity: 38/79 = 48.10%
Jund Wildfire:   31/78 = 39.70%
Mono Red Rally:  28/59 = 47.50%
Spy Combo:       62/62 = 100.00%
```

The head strongly overpredicted `SpyCombo`, especially early. Neural-belief ISMCTS should not consume this head directly yet.

## Training

Clone:

```text
Pauper-Spy-Combo-Value-TrueBeliefLabels-20260512
```

Initial hashes matched accepted Spy:

```text
model.pt        14E1A1DA4F3E
model_latest.pt 72857AA2975A
```

Run:

```text
20260512_thesis_clean_true_belief_labels_spyclone_300ep
```

Settings:

```text
TOTAL_EPISODES=300
TRAIN_PROFILES=1
NUM_GAME_RUNNERS=8
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SELFPLAY_OPPONENT_TRAINING=1
SKILL_MIX=1:0.34,3:0.33,7:0.33
RL_BELIEF_TRUE_ARCHETYPE_LABELS_ENABLE=1
BELIEF_LOSS_COEF=1.0
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_ARCHETYPE_BELIEF_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

Completed cleanly:

```text
300/300 episodes
model.pt        9750489F8C4A
model_latest.pt AED26E5910A9
MCTS activations: 0
```

Accepted hashes were rechecked afterward and unchanged.

## Clone Calibration

Run:

```text
20260512_true_belief_labels_spyclone300_cp7_logged4_belief
```

Result:

```text
CP7 logged4: 7/16 = 43.75%
Belief lines: 263
```

Turn-level belief accuracy:

```text
Grixis Affinity: 28/61 = 45.90%
Jund Wildfire:   46/89 = 51.70%
Mono Red Rally:  22/48 = 45.80%
Spy Combo:       59/65 = 90.80%
```

True labels reduced the perfect Spy-collapse in mirror but did not make the non-Spy posterior reliable. Early states still overpredicted `SpyCombo` in Affinity and Rally.

## Reduced CP7 Gate

Run:

```text
20260512_true_belief_labels_spyclone300_spy_cp7_unique_eval16
```

Result:

```text
Overall: 29/64 = 45.31%
```

Per matchup:

```text
Spy mirror:       12/16 = 75.00%
Jund Wildfire:     8/16 = 50.00%
Mono Red Rally:    7/16 = 43.75%
Grixis Affinity:   2/16 = 12.50%
```

Accepted CP7 reference:

```text
Overall: 108/242 = 44.63%
Spy mirror: 43/56 = 76.79%
Jund Wildfire: 36/60 = 60.00%
Mono Red Rally: 16/64 = 25.00%
Grixis Affinity: 13/62 = 20.97%
```

## Verdict

Do not promote and do not submit to HPC.

The branch is directionally interesting because Rally improved in the reduced gate, but the overall delta is only +0.68pp against the accepted CP7 reference and well inside sampling noise. It also worsened Affinity and Wildfire while failing the belief-calibration goal. Treat true hidden-state labels as useful infrastructure, not a solved belief model.

Next experiment should either reduce the belief coefficient or isolate the belief update from policy/value updates before trying neural-belief ISMCTS.

## Low-Coefficient Follow-Up

Clone:

```text
Pauper-Spy-Combo-Value-TrueBeliefLowCoef-20260512
```

Run:

```text
20260512_thesis_clean_true_belief_lowcoef_spyclone_300ep
```

Only change from the first branch:

```text
BELIEF_LOSS_COEF=0.3
```

Training completed cleanly at `300/300`. `model_latest.pt` changed to:

```text
3E1561BA4C11
```

Calibration run:

```text
20260512_true_belief_lowcoef_spyclone300_cp7_logged4_belief
```

Result:

```text
CP7 logged4: 6/16 = 37.50%
Belief lines: 304
```

Turn-level belief accuracy:

```text
Grixis Affinity: 31/62 = 50.00%
Jund Wildfire:   54/106 = 50.90%
Mono Red Rally:  27/55 = 49.10%
Spy Combo:       78/81 = 96.30%
```

Reduced CP7 gate:

```text
20260512_true_belief_lowcoef_spyclone300_spy_cp7_unique_eval16
Overall: 23/64 = 35.94%
```

Per matchup:

```text
Spy mirror:       13/16 = 81.25%
Jund Wildfire:     6/16 = 37.50%
Mono Red Rally:    3/16 = 18.75%
Grixis Affinity:   1/16 =  6.25%
```

Verdict: rejected. Lowering the coefficient slightly improved non-Spy belief calibration but made gameplay much worse, especially into Affinity and Rally. Do not continue simple true-label coefficient sweeps.

## Belief-Head-Only Follow-Up

Clone:

```text
Pauper-Spy-Combo-Value-TrueBeliefHeadOnly-20260512
```

Run:

```text
20260512_thesis_clean_true_belief_headonly_spyclone_300ep
```

Settings:

```text
BELIEF_HEAD_ONLY=1
RL_BELIEF_TRUE_ARCHETYPE_LABELS_ENABLE=1
BELIEF_LOSS_COEF=1.0
```

The trainer still generated terminal-return episodes, but `py4j_entry_point.py` set `requires_grad=false` for every parameter outside `belief_head.*`.

Training completed cleanly at `300/300`:

```text
model.pt        8C1EFD465138
model_latest.pt C2E31AD50FDC
MCTS activations: 0
```

Calibration/eval run:

```text
20260512_true_belief_headonly_spyclone300_cp7_logged4_belief
```

One Jund chunk stalled with an empty log, so the completed result is 14 games:

```text
CP7 logged4 partial: 8/14 = 57.14%
Belief lines: 300
```

Turn-level belief accuracy:

```text
Grixis Affinity: 25/49  = 51.00%
Jund Wildfire:   63/132 = 47.70%
Mono Red Rally:  28/61  = 45.90%
Spy Combo:       55/58  = 94.80%
```

First belief prediction per game:

```text
Grixis Affinity: 2 Affinity, 2 SpyCombo
Jund Wildfire:   1 Wildfire, 3 SpyCombo
Mono Red Rally:  1 Rally,    3 SpyCombo
Spy Combo:       4 SpyCombo
```

Prediction distribution:

```text
SpyCombo: 177/300 = 59.00%
Wildfire:  64/300 = 21.30%
Affinity: 31/300 = 10.30%
Rally:    28/300 =  9.30%
```

Verdict: rejected as a belief source. Freezing policy/value avoided the gameplay collapse seen in the low-coefficient branch, but the belief head still overpredicts `SpyCombo` and is not calibrated enough for neural-belief ISMCTS. Do not wire this posterior into determinization yet.
