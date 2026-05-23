# Accepted Checkpoint With History Features Eval-Only

Date: 2026-05-11

## Question

Did the generic-history H4 training branches improve because of learning, or did the new history feature slots perturb the accepted model by themselves?

## Setup

No training. Active models were the restored accepted Affinity-pressure checkpoint.

Eval-only environment:

```text
RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1
RL_GENERIC_ACTION_HISTORY_WINDOW=4
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
```

## Results

CP1 reduced:

```text
Run: 20260511_accepted_history_h4_evalonly_spy_cp1_unique_eval16
Total: 28/64 = 43.75%
Spy mirror: 8/16
Jund Wildfire: 7/16
Mono Red Rally: 8/16
Grixis Affinity: 5/16
```

CP7 reduced:

```text
Run: 20260511_accepted_history_h4_evalonly_spy_cp7_unique_eval16
Total: 29/64 = 45.31%
Spy mirror: 14/16
Jund Wildfire: 4/16
Mono Red Rally: 9/16
Grixis Affinity: 2/16
```

## Interpretation

The feature gate alone changes behavior materially. CP7 remains near the accepted aggregate, but the matchup distribution shifts sharply. CP1 falls to the same aggregate as the pure-history 200-episode branch (`28/64`), so the lower-skill regression is mostly representation perturbation rather than learned policy improvement.

The pure-history branch's CP7 `32/64` is only a small gain over eval-only `29/64`, not enough to justify continuing this exact encoding.

## Decision

Do not continue generic history H4 in the current player-stats slot encoding. Any future history work should use a fresh model or a backward-compatible zero-initialized projection path, not untrained values inserted into an existing checkpoint's dense token slots.
