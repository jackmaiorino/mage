# Thesis-Clean Corrected Trajectory Teacher

Date: 2026-05-13

## Harness Fix

Several counterfactual wrappers had stale `MODEL_D_MODEL=256` /
`MODEL_NUM_LAYERS=4` defaults while the canonical thesis-clean Spy checkpoint
uses `128/2`. Defaults were corrected in:

```text
scripts/run_action_counterfactual.ps1
scripts/run_spy_line_search.ps1
scripts/run_spy_line_replay_probe.ps1
scripts/run_mulligan_counterfactual.ps1
```

This reopened the May 12 trajectory path: the prior zero-trajectory result was
not reliable because line search could have been running through the wrong model
shape.

## Corrected Collection

CP1 unique pool:

```text
20260513_thesis_clean_trajectory128_cp1_collect16
8 winning trajectories / 8 completed scenarios
39 selected decisions
```

CP7 unique pool:

```text
20260513_thesis_clean_trajectory128_cp7_collect64
32 winning trajectories / 48 completed scenarios
138 selected decisions
```

Affinity-only CP7 pressure pool:

```text
20260513_thesis_clean_prefix128_affinity_cp7_collect64_targets
28 winning trajectories / 64 scenarios
125 selected decisions
```

All collections used terminal `WIN`, generic branch order, no heuristic rewards,
no Spy action facts, no Spy terminal labels, no action-text/card-name filters,
and no selective MCTS card gates.

## Training Checks

CP1 trajectory RL, 8 trajectories, 4 epochs:

```text
profile: Pauper-Spy-Combo-Value-TrajRL128-20260513
CP7 eval4:  9/16
CP7 eval16: 27/64 = 42.19%
```

CP7 trajectory RL, 32 trajectories, 4 epochs:

```text
profile: Pauper-Spy-Combo-Value-TrajRL128CP7-20260513
CP7 eval4: 4/15 = 26.67%
```

CP7 trajectory RL, 32 trajectories, 1 epoch:

```text
profile: Pauper-Spy-Combo-Value-TrajRL128CP7E1-20260513
CP7 eval4:  11/16
CP7 eval16: 21/64 = 32.81%
```

CP7 prefix-target KL, 32 trajectories, 1 epoch, low-dose target/reference mix:

```text
profile: Pauper-Spy-Combo-Value-PrefixKL128CP7E1-20260513
CP7 eval4:  9/16
CP7 eval16: 21/64 = 32.81%
```

Affinity-only prefix-target KL, 28 trajectories, 1 epoch:

```text
profile: Pauper-Spy-Combo-Value-PrefixKL128AffinityE1-20260513
Affinity CP7 eval16: 1/16 = 6.25%
```

## Replay Diagnostics

Scoring the corrected CP7 prefix tensors:

```text
accepted checkpoint: 119/127 top-1 = 93.70%
PrefixKL branch:      119/127 top-1 = 93.70%
TrajectoryRL branch:  119/127 top-1 = 93.70%
```

Replay against the same CP7 trajectory file:

```text
accepted unforced: 19/32 wins, 79/160 matches
accepted forced:   31/32 wins, 85/160 matches
PrefixKL unforced: 19/32 wins, 75/160 matches
TrajRL unforced:   22/32 wins, 75/160 matches
```

Affinity depth-10 collection:

```text
20260513_thesis_clean_prefix128_affinity_cp7_depth10_collect32
16 winning trajectories / 24 completed scenarios
124 selected decisions
accepted score: 121/124 top-1 = 97.58%
accepted unforced replay: 12/16 wins
accepted forced replay:   16/16 wins
```

The depth-10 corpus finally includes a small number of `Balustrade Spy`
decisions, but the accepted policy already ranks almost every serialized target
top-1. The remaining gap is not "policy cannot score searched states"; it is
that the searched states are a favorable subset and the same labels do not
change broader CP7 play.

## Read

Corrected terminal-prefix search can now find thesis-clean winning lines. That
is the useful result.

Naive distillation from those lines is still not a promotable policy-improvement
operator. Small 4-game screens looked good twice, but both expanded CP7 screens
collapsed back below the accepted checkpoint, especially into Affinity.

Next work should stop training from shallow/depth-10 prefix tensors unless the
collector is changed to produce genuinely unsolved states. The useful next
teacher would need to find states where the accepted policy's top action is not
already inside the winning target set, or it needs to become an online search
operator rather than an offline distillation corpus.
