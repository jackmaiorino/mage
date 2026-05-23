# Accepted Checkpoint History Scale Sweep

Date: 2026-05-11

## Question

The full-strength generic-history H4 feature slots perturb the accepted checkpoint. Can scaling the history vector reduce that perturbation while preserving useful CP7 signal?

## Code Change

Added:

```text
RL_GENERIC_ACTION_HISTORY_SCALE
```

Range is clamped to `[0.0, 1.0]`; default is `1.0`.

## Eval-Only Setup

No training. Active models were restored accepted Affinity-pressure checkpoint.

```text
RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1
RL_GENERIC_ACTION_HISTORY_WINDOW=4
RL_GENERIC_ACTION_HISTORY_SCALE=0.25
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
```

## Results

CP1 reduced:

```text
Run: 20260511_accepted_history_h4_scale025_evalonly_spy_cp1_unique_eval16
Total: 32/64 = 50.00%
Spy mirror: 11/16
Jund Wildfire: 10/16
Mono Red Rally: 6/16
Grixis Affinity: 5/16
```

CP7 reduced:

```text
Run: 20260511_accepted_history_h4_scale025_evalonly_spy_cp7_unique_eval16
Total: 31/64 = 48.44%
Spy mirror: 13/16
Jund Wildfire: 8/16
Mono Red Rally: 6/16
Grixis Affinity: 4/16
```

CP3 reduced:

```text
Run: 20260512_accepted_history_h4_scale025_evalonly_spy_cp3_unique_eval16
Total: 26/64 = 40.62%
Spy mirror: 10/16
Jund Wildfire: 5/16
Mono Red Rally: 6/16
Grixis Affinity: 5/16
```

Scale `0.10` CP3 reduced:

```text
Run: 20260512_accepted_history_h4_scale010_evalonly_spy_cp3_unique_eval16
Total: 23/64 = 35.94%
Spy mirror: 11/16
Jund Wildfire: 6/16
Mono Red Rally: 4/16
Grixis Affinity: 2/16
```

## Interpretation

Scale `0.25` removes most of the CP1 regression seen at scale `1.0` (`28/64`) while keeping CP7 above the accepted corrected aggregate (`108/242 = 44.63%`). This is still eval-only perturbation, not learned improvement, but it identifies a safer representation magnitude for training.

The missing CP3 check changed the decision. Both scale `0.25` and a smaller `0.10` fail the reduced CP3 gate, so the eval-only history perturbation is not promotable and should not receive a full CP1/CP3/CP7 sweep.

## Next

Run a small thesis-clean training branch with:

- `RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1`
- `RL_GENERIC_ACTION_HISTORY_WINDOW=4`
- `RL_GENERIC_ACTION_HISTORY_SCALE=0.25`

Do not use HPC unless a local branch improves CP1/CP3/CP7 jointly.

Updated decision after the CP3 checks: do not continue history-scale cycling. The next experiment should target the learning signal, not another raw feature perturbation.
