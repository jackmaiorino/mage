# Thesis-clean accepted snapshot selection - 2026-05-14

## Question

The accepted profile has many historical snapshots. Before spending new
training compute, check whether the current accepted `model_latest.pt` is simply
a bad checkpoint on an already-trained trajectory.

This is thesis-clean and eval-only:

- no new training;
- no heuristic rewards;
- no Spy terminal mode;
- no card/action regex labels;
- no Spy action facts;
- no MCTS/search gate.

## Setup

Created eval-only hardlink clones for spaced accepted snapshots:

```text
Pauper-Spy-Combo-Value-LatestClone-20260514
Pauper-Spy-Combo-Value-Snap60000-20260514
Pauper-Spy-Combo-Value-Snap90000-20260514
Pauper-Spy-Combo-Value-Snap97000-20260514
Pauper-Spy-Combo-Value-Snap117000-20260514
Pauper-Spy-Combo-Value-Snap122000-20260514
Pauper-Spy-Combo-Value-Snap132000-20260514
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260514_accepted_snapshot_pressure_registry.json
```

Pressure screen:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_accepted_snapshot_pressure_g2
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=2
```

## Results

| Profile | Rally | Affinity | Combined |
| --- | ---: | ---: | ---: |
| LatestClone | 0/2 | 0/2 | 0/4 |
| Snap60000 | 0/2 | 0/2 | 0/4 |
| Snap90000 | 1/2 | 0/2 | 1/4 |
| Snap97000 | 0/2 | 0/2 | 0/4 |
| Snap117000 | 1/2 | 0/2 | 1/4 |
| Snap122000 | 1/2 | 0/2 | 1/4 |
| Snap132000 | 0/2 | 0/2 | 0/4 |

## Verdict

Rejected as a promotion path. The screen found small Rally variance in older
snapshots, but no tested snapshot won an Affinity game. The best combined result
was only `1/4`, too weak to justify expanding all snapshots or promoting a
historical checkpoint.

Do not spend HPC on checkpoint selection from the existing accepted snapshots.
