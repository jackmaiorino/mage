# RL Activation Failure Tracking

## Overview

We've implemented comprehensive tracking and alerting for RL player activation failures to ensure they don't pollute the training signal.

## Why This Matters

When the RL agent selects an action that fails to activate:
1. **Pollutes training signal**: The model learns incorrect state-action associations
2. **Wastes compute**: The agent passes instead of taking a real action
3. **Indicates bugs**: Either in filtering logic or game state handling

**Goal: Zero RL activation failures during training.**

## What Was Added

### 1. Impossible-to-Miss Error Logging

When the RL player fails to activate an ability, you'll see:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! RL PLAYER ACTIVATION FAILED - THIS HURTS TRAINING !!!
!!! TOTAL RL ACTIVATION FAILURES THIS RUN: 1 !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
RL Player: PlayerRL1 | Thread: Thread-59
Failed ability: Cast Land Grant
RL ACTIVATION DIAGNOSTICS: player=PlayerRL1 abilityType=Spell usesStack=true canActivate=true canChooseTarget=true canPlayLand=true approvingObjects=0 source=Land Grant zone=HAND
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**Key identifiers:**
- `RL PLAYER ACTIVATION FAILED` in the message
- `PlayerRL1` as the player name
- Thread ID (RL player always runs on same thread per game)
- Running failure count

### 2. Automatic Failure Counting

A static counter tracks total RL activation failures across the entire training/eval run:

```java
private static final AtomicInteger RL_ACTIVATION_FAILURES = new AtomicInteger(0);
```

### 3. Periodic Reporting

The failure count is included in:

**Training progress logs:**
```
Training progress: episode=40/50000 (run=40, 1.846 eps/s), ETA 27058s, RL_activation_failures=0
```

**Evaluation logs:**
```
Evaluation win rate: 1.0000, RL_activation_failures=0
```

### 4. Programmatic Access

```java
// Get current count
int failures = ComputerPlayerRL.getRLActivationFailureCount();

// Reset for new training run
ComputerPlayerRL.resetRLActivationFailureCount();
```

## How to Use

### During Training

1. **Monitor progress logs** - Check the `RL_activation_failures` count every N episodes
2. **Watch for error banners** - If you see the `!!!` banners, investigate immediately
3. **Expected value: 0** - Any non-zero value indicates a problem

### During Evaluation

1. Check the final evaluation log for `RL_activation_failures`
2. If > 0, the evaluation results may be unreliable

### When You See Failures

1. **Check the thread ID** - Opponent failures will be on different threads (Thread-60, Thread-61, etc.)
2. **Look at the diagnostics** - The error includes full ability details
3. **Check simulation filtering** - Look for "SIM-FILTER" logs showing what was filtered
4. **Verify game state** - Check the battlefield printout before the failure

## Known Issues

### Opponent (ComputerPlayer7) Failures

The opponent (EvalBot, ComputerPlayer7) may show activation failures for cards like Land Grant and Fireblast. These are:
- **NOT RL player failures** - Different thread IDs
- **Expected behavior** - The opponent doesn't have our simulation-based filtering
- **Not harmful to training** - They don't pollute the RL agent's learning

**How to tell them apart:**

**RL Player failure (BAD):**
```
!!! RL PLAYER ACTIVATION FAILED - THIS HURTS TRAINING !!!
RL Player: PlayerRL1 | Thread: Thread-59
```

**Opponent failure (OK):**
```
Activation failed diagnostics: ... source=Land Grant zone=HAND
Thread-64 (different from RL player thread)
```

## Current Status

As of the latest fix:
- ✅ Simulation-based filtering implemented for RL player
- ✅ Land Grant bug fixed for RL player
- ✅ Mulligan bug fixed for RL player
- ✅ Comprehensive failure tracking in place
- ✅ Expected RL_activation_failures: **0**

## Future Improvements

If you see persistent RL activation failures:

1. **Investigate the specific abilities** - What cards are failing?
2. **Check alternative costs** - Are conditional costs being handled correctly?
3. **Verify simulation fidelity** - Does the simulation match real game state?
4. **Consider additional filters** - May need special handling for specific ability types

## Related Documentation

- `CRITICAL_BUG_FOUND.md` - Details on the land-play and mulligan bugs
- `ACTION_FILTERING_ANALYSIS.md` - Analysis of simulation-based filtering performance
- `curriculum_usage.md` - Curriculum learning implementation details
