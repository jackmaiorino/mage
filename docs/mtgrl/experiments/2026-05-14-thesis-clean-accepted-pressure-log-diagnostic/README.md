# Thesis-clean accepted pressure log diagnostic - 2026-05-14

## Question

After several local branches failed Rally/Affinity gates, collect a fresh compact
accepted-policy pressure sample and inspect whether the bottleneck is still
Rally plus Affinity or mainly Affinity.

Diagnostic only. No training used this log data.

## Run

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_accepted_pressure_logged_cp7_g8
profile=Pauper-Spy-Combo-Value
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=8
game_log_format=compact
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 5 | 8 | 62.50% |
| Grixis Affinity | 2 | 8 | 25.00% |
| Combined | 7 | 16 | 43.75% |

## Diagnostics

Exported compact logs:

```text
diagnostics/accepted_pressure_games.jsonl
games=16
decisions=1876
```

Value-head analysis:

```text
decision_rows=1806
all AUC=0.530104
nontrivial_options AUC=0.516613
game mean_value_score AUC=0.460317
game first_value_score AUC=0.317460
```

Event credit:

```text
baseline_winrate=7/16 = 43.75%
selected_cast_spy: 6/10 games won
has_cast_spy_option: 6/10 games won
selected_cast_dread_return: 3/3 games won
selected_land_grant: 6/10 games won
selected_pass: 7/16 games won
```

The compact action-health parser is incomplete for some hidden-land metrics on
this format, so do not use its hidden-land rows from this run as authoritative.

## Read

This sample says the immediate pressure target should be Affinity, not Rally.
Accepted was `5/8` into Rally but only `2/8` into Affinity. The value head is
also weak as a branch evaluator on the same logs: decision-level AUC is barely
above chance and game-level first-value AUC is inverted.

The next data-generation experiment is therefore Affinity-only and
baseline-losing-alternative filtered. It should answer whether the branch-pair
trajectory path is failing because the paired corpus is too small or because the
paired-loss import strategy is wrong even with more hard-matchup data.
