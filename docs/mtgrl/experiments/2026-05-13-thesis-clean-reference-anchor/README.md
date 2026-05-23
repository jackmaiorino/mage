# Thesis-Clean Reference-Policy Anchor

Date: 2026-05-13

## Mechanism

Added a default-off generic frozen reference-policy anchor:

```text
REFERENCE_POLICY_KL_COEF=<coef>
MCTS_REFERENCE_MODEL_PATH=<frozen model_latest.pt>
```

When enabled, the learner adds cross-entropy/KL from the frozen reference policy distribution to the current policy distribution over the same legal candidate set. This is thesis-clean because it uses no action text, card names, Spy terminal labels, Spy hand pools, heuristic step rewards, or selective search gates.

Validation:

```powershell
.mtgrl_venv\Scripts\python.exe -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
```

## Hard CP7 Anchor Probe

Question: can a frozen reference policy let the accepted Spy checkpoint adapt to CP7 pressure without forgetting?

Profile:

```text
Pauper-Spy-Combo-Value-HardCP7Anchored-20260513
```

Registries:

```text
local-training/local_pbt/thesis_clean/20260513_thesis_clean_hardcp7_anchored_registry.json
local-training/local_pbt/thesis_clean/20260513_thesis_clean_hardcp7_anchored_eval_registry.json
```

Training setup:

```text
Start checkpoint: accepted Pauper-Spy-Combo-Value model_latest.pt 72857AA2975A
Episodes: 128
Opponent pool: Affinity-pressure weighted pool
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.20
SKILL_MIX=7:1.0
REFERENCE_POLICY_KL_COEF=0.50
MCTS_REFERENCE_MODEL_PATH=.../Pauper-Spy-Combo-Value/models/model_latest.pt
MCTS/ISMCTS: off
heuristic rewards: off
Spy action facts: off
card-belief labels: off
```

Training result:

```text
overall: 20/128 = 15.62%
CP7:     17/105 = 16.19%
self:     3/23  = 13.04%
```

Model hashes:

```text
model.pt              14E1A1DA4F3E
model_latest.pt       4431A81C9F2A
model_latest_ep128.pt 4431A81C9F2A
```

Corrected CP7 gate:

```text
Run: 20260513_hardcp7anchored128_cp7_eval4_unique
Skill: CP7
Stopped after hard matchups failed.
Recorded result files: 3/12 = 25.00%
Spy mirror:     2/4 = 50.00%
Jund Wildfire:  0/4 =  0.00%
Mono Red Rally: 1/4 = 25.00%
Grixis Affinity: no completed games before stop
```

An earlier eval was aborted immediately because it used the weighted training pool and expanded to 52 jobs. The corrected eval registry uses the unique four-deck thesis-clean eval pool.

## Verdict

Rejected. Do not spend HPC on this branch.

The setting was too hard and/or the anchor too weak to preserve the accepted policy. The branch failed both during training and in the reduced CP7 gate. The useful information is that a frozen reference anchor alone does not make aggressive CP7-only continuation safe.

## Next Experiment

Run a lower-risk curriculum anchor probe from the accepted checkpoint:

```text
unique four-deck pool
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.25,3:0.25,7:0.50
REFERENCE_POLICY_KL_COEF=1.00
128 local episodes
```

This tests the actual anchor mechanism under a non-destructive curriculum. If it still collapses, stop short reference-anchor cycling and return to deeper policy-improvement teacher work.

## Curriculum Anchor Probe

Profile:

```text
Pauper-Spy-Combo-Value-CurriculumAnchored-20260513
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260513_thesis_clean_curriculum_anchored_registry.json
```

Training setup:

```text
Start checkpoint: accepted Pauper-Spy-Combo-Value model_latest.pt 72857AA2975A
Episodes: 128 target, 127 rows recorded in training_stats.csv
Opponent pool: unique four-deck thesis-clean eval pool
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.25,3:0.25,7:0.50
REFERENCE_POLICY_KL_COEF=1.00
MCTS_REFERENCE_MODEL_PATH=.../Pauper-Spy-Combo-Value/models/model_latest.pt
MCTS/ISMCTS: off
heuristic rewards: off
Spy action facts: off
card-belief labels: off
```

Training result:

```text
overall recorded: 44/127 = 34.65%
CP1:               6/9   = 66.67%
CP3:               8/18  = 44.44%
CP7:              10/33  = 30.30%
self-play:        20/67  = 29.85%
```

Model hashes:

```text
model.pt              14E1A1DA4F3E
model_latest.pt       DBF514B03455
model_latest_ep128.pt DBF514B03455
```

Reduced CP7 gate:

```text
Run: 20260513_curriculumanchored128_cp7_eval4
Skill: CP7
Stopped before Affinity because hard matchups already failed.
Recorded result files: 4/12 = 33.33%
Spy mirror:     4/4 = 100.00%
Jund Wildfire:  0/4 =   0.00%
Mono Red Rally: 0/4 =   0.00%
Grixis Affinity: no completed games before stop
```

## Reference-Anchor Surface Verdict

Rejected as a short local quick-win surface.

The anchor preserved or even sharpened mirror behavior, but both tested schedules collapsed the hard pressure matchups. Increasing anchor strength and softening the curriculum did not fix Jund/Rally. Do not spend HPC on reference-anchored terminal continuation in its current form.

Next work should move away from continuation schedules and toward a stronger generic policy-improvement teacher, with reference anchoring kept only as a stability regularizer if that teacher first clears a local gate.
