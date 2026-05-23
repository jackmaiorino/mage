# Thesis-Clean Deep Mulligan Line Distillation

Date: 2026-05-13

## Question

Can generic terminal-outcome line distillation fix a concrete failure mode where the accepted Spy policy mulligans too deeply and can reach a zero-card start?

This stays thesis-clean because labels come from full-game terminal outcomes under alternative London mulligan depths. The trainer does not inspect card names, action text, Spy-specific win rules, heuristic step rewards, selective search keywords, or Spy candidate facts.

## Triggering Diagnostic

Compact Rally/Jund logs for accepted showed one Rally loss where the policy repeatedly chose to mulligan from 7 down to 1, then bottomed to an empty hand and started with zero cards. The aggregate first-hand mulligan probe looked reasonable, so the failure appears concentrated in deeper mulligan states rather than ordinary opening-hand keep/mull behavior.

## Implementation

- `MulliganCounterfactualTrainer`
  - Increased the line-mode `--line-max-mulls` clamp from `3` to `7`.
- Training mode:
  - `--line-mode true`
  - `--line-max-mulls 7`
  - `--line-bottom-combos -1`
  - `--line-train-bottoms false`
  - Binary keep-vs-deeper-mull labels only.

The bottom-combo branch was deliberately disabled for this smoke to avoid combinatorial rollout growth and to keep the test focused on the catastrophic deep-mull decision.

## Run

- Clone: `Pauper-Spy-Combo-Value-MullLineDeep-20260513`
- Source checkpoint: accepted `Pauper-Spy-Combo-Value`, SHA prefix `72857AA2975A`
- Train run: `20260513_mull_line_deep_binary_smoke4`
- Scenarios: 4
- Line specs per scenario: 8
- Opponent mode: CP7
- Opponent pool: `decklist.active_profile_pool_thesis_rally_affinity_pressure_20260510.txt`
- Actionable scenarios: 2
- Candidate binary samples: 6
- Train pass samples: 12
- Checkpoint SHA prefix after training: `A3985CF1FB48`

Two attempted background launches failed before training because `Start-Process` mangled typed boolean arguments for the PowerShell wrapper. They produced no model update and are not counted as experiment runs.

## Probe

Run: `20260513_mull_line_deep_binary_smoke4_probe100`

First-hand behavior remained close to accepted:

| Effective lands | Samples | Keep rate | Mean P(keep) |
| ---: | ---: | ---: | ---: |
| 0 | 26 | 0.0% | 0.038 |
| 1 | 40 | 25.0% | 0.278 |
| 2 | 25 | 60.0% | 0.596 |
| 3 | 9 | 88.9% | 0.909 |

This did not show obvious first-hand damage, but it also did not prove the deep-mull state was repaired.

## Reduced CP7 Gate

Run: `20260513_mull_line_deep_smoke4_cp7_g4`

| Opponent | Wins | Total |
| --- | ---: | ---: |
| Spy mirror | 3 | 4 |
| Jund Wildfire | 0 | 3 |
| Mono Red Rally | 2 | 4 |
| Grixis Affinity | 2 | 4 |
| Overall | 7 | 15 |

One Jund chunk returned `0/0` after a long timeout, so the scored total is 15 rather than 16.

## Decision

Rejected. The tiny deep-mull binary line pass did not clear the reduced CP7 gate and badly regressed Jund. Do not scale this exact setup or spend HPC on it.

The useful result is diagnostic: deep mulligan failures are real, but isolated head-only correction from a tiny line set is too brittle. If revisited, collect a larger line dataset in collect-only mode first and inspect label balance by mulligan depth before training.

## Larger Collect-Only Check

Run: `20260513_mull_line_deep_binary_collect16`

Settings matched the smoke, but used accepted read-only and `--collect-only true`.

Result:

- Scenarios: 16
- Actionable scenarios: 10
- Candidate samples: 32
- Selected samples after balancing: 22
- Winning-line samples exported for inspection: 744
- Accepted checkpoint hash after collection: unchanged at `72857AA2975A`

Selected binary labels by depth:

| Prompt | Samples | Keep labels | Mull labels | Mean target keep |
| --- | ---: | ---: | ---: | ---: |
| First hand | 4 | 1 | 3 | 0.522 |
| After 1 mull | 5 | 2 | 3 | 0.549 |
| After 2 mulls | 6 | 3 | 3 | 0.620 |
| After 3 mulls | 4 | 2 | 2 | 0.604 |
| After 4 mulls | 3 | 3 | 0 | 0.881 |

Full line outcomes by opponent:

| Opponent | Scenarios | Line wins | Line losses | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| Jund Wildfire | 4 | 8 | 24 | 5 |
| Spy mirror | 1 | 0 | 7 | 0 |
| Mono Red Rally | 6 | 0 | 48 | 0 |
| Grixis Affinity | 5 | 6 | 34 | 5 |

Read: the corpus explains why the tiny training pass was brittle. Deep labels are depth-dependent, and the random pressure-pool sample produced no actionable Rally labels at all: every Rally line lost at every tested mulligan depth. This is evidence against scaling mulligan-line training as the next Rally fix. The Rally bottleneck is downstream action execution or search, not simply keep/mull depth.
