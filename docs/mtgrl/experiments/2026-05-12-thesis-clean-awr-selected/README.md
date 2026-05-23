# Thesis-Clean AWR Selected-Action Targets

Date: 2026-05-12

## Question

Can a generic AWR/AWAC-style policy-improvement target improve Spy from terminal outcomes without heuristic rewards, Spy-specific action facts, card-name filters, or search targets?

The target is thesis-clean: each selected action gets a target weight from terminal discounted return minus the model's old value estimate. No action text, deck-specific combo rule, or hidden Spy condition is used.

## Implementation

Added optional online selected-action targets:

```text
RL_AWR_SELECTED_ACTION_TARGETS_ENABLE=1
RL_AWR_GAMMA=0.99
RL_AWR_TEMPERATURE=0.50
RL_AWR_MIN_WEIGHT=0.05
RL_AWR_MAX_WEIGHT=5.0
RL_AWR_POSITIVE_ADVANTAGE_ONLY=0
MCTS_TARGET_ROW_SUM_WEIGHT_ENABLE=1
MCTS_TARGET_ROW_SUM_WEIGHT_MAX=5.0
```

`RLTrainer` attaches weighted selected-action targets after terminal rewards are known. `py4j_entry_point.py` treats the pre-normalization target row sum as the row's cross-entropy weight when enabled. Normal MCTS rows still behave as before when the flag is disabled.

Validation:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
.mtgrl_venv\Scripts\python.exe -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
.mtgrl_venv\Scripts\python.exe -m py_compile scripts/run_cp7_eval_sweep.py
```

## Run

Profile:

```text
Pauper-Spy-Combo-Value-AWRSelected-20260512
```

Training:

```text
run: 20260512_thesis_clean_awr_selected_hardmix_train128
episodes: 128/128
profiles: Spy clone only
opponent sampler: hybrid
skill mix: 1:0.25,3:0.25,7:0.50
heuristic rewards: off
Spy action facts: off
MCTS/ISMCTS: off
```

Model hashes after training:

```text
model.pt        14E1A1DA4F3E
model_latest.pt BC3077B0C4E4
```

## CP7 Gate

Run:

```text
20260512_awr_selected_hardmix_train128_spy_cp7_unique_eval16_db0_lazy
```

The eval was stopped early after 54 completed games because it had already failed the local gate and the remaining Affinity chunks could not repair the Rally regression.

Partial CP7 result:

```text
overall:        22/54 = 40.74%
Spy mirror:     11/16 = 68.75%
Jund Wildfire:   8/15 = 53.33%
Mono Red Rally:  1/16 =  6.25%
Grixis Affinity: 2/7  = 28.57%
```

Accepted CP7 reference:

```text
overall:        108/242 = 44.63%
Spy mirror:      43/56 = 76.79%
Jund Wildfire:   36/60 = 60.00%
Mono Red Rally:  16/64 = 25.00%
Grixis Affinity: 13/62 = 20.97%
```

## Verdict

Rejected.

AWR selected-action targets did not improve the pressure surface. They slightly lifted the partial Affinity sample versus accepted, but the Rally collapse is too large and the mirror/Jund regressions remove any acceptance case. Do not scale this exact AWR formulation on HPC.

The generic mechanism may still be reusable if paired with a stronger off-policy correction or per-decision uncertainty filter, but plain terminal-advantage weighting is not the next best deep path.

## Hygiene Notes

`run_cp7_eval_sweep.py` now supports:

```text
CP7_EVAL_DB_SOURCE=<file-or-directory>
CP7_EVAL_CLEAN_DB_AFTER_JOB=1
```

This avoids failures when the default `db/cards.h2.mv.db` is locked and prevents CP7 sweeps from pre-allocating every per-job DB copy at once.

During this run an orphaned four-profile trainer was found mutating accepted profile checkpoints directly. It was stopped after losing its GPU service. The overwritten accepted files were preserved under:

```text
local-training/local_pbt/model_backups/orphan_trainer_overwrite_20260512_1929
```

The accepted checkpoint files were restored from exact hash-matching backups and verified:

```text
Pauper-Spy-Combo-Value model.pt                      14E1A1DA4F3E
Pauper-Spy-Combo-Value model_latest.pt               72857AA2975A
Pauper-Wildfire-Value model.pt                       217707FE2EF0
Pauper-Wildfire-Value model_latest.pt                6F72507293CC
Pauper-Rally-Anchor-Value model.pt                   C4ADFD672072
Pauper-Rally-Anchor-Value model_latest.pt            5A04AFF24179
Pauper-Affinity-Anchor-Value model.pt                A692C64954F3
Pauper-Affinity-Anchor-Value model_latest.pt         B7FD1930779E
```
