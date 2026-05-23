# Thesis-clean accepted Affinity log diagnostic - 2026-05-14

## Question

After the pressure-pair diagnostic narrowed the hard target to CP7 Grixis
Affinity, inspect a larger accepted-policy Affinity-only compact log sample.
The goal is to identify the next thesis-clean mechanism, not to create training
labels.

Diagnostic only. No training consumed these logs.

## Run

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_accepted_affinity_logged_cp7_g16
profile=Pauper-Spy-Combo-Value
skill=7
opponent=Grixis Affinity
games=16
game_log_format=compact
```

Result:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Grixis Affinity | 3 | 16 | 18.75% |

Reference:

```text
accepted CP7 Grixis Affinity: 13/62 = 20.97%
```

## Exporter fix

`scripts/export_game_log_trajectories.py` now parses compact `STATE:` lines
into `state_summary` fields. This is diagnostics-only: it exposes life totals,
stack count, zone counts, and visible compact card snippets from the logs. It
does not affect training or policy inference.

Validation:

```text
python -m py_compile scripts/export_game_log_trajectories.py
exported games=16 decisions=1937
```

## Diagnostics

Value-head analysis:

```text
games=16
decision_rows=1863
all AUC=0.660896 gap=0.067533
nontrivial_options AUC=0.717537 gap=0.115441
game mean_value_score AUC=0.846154 gap=0.067526
game first_value_score AUC=0.487179 gap=-0.014424
game last_value_score AUC=0.346154 gap=-0.069011
```

Event credit:

```text
baseline_winrate=3/16 = 18.75%
selected_cast_spy: 1/7 games won
has_cast_spy_option: 1/7 games won
selected_cast_dread_return: 0/1 games won
selected_land_grant: 3/9 games won
```

State-aware split:

```text
spy_cast: 1/7 wins
no_spy_cast: 2/9 wins
dread_cast: 0/1 wins
dread_from_visible_hand: 0/1 wins
opponent_battlefield_count >= 10: 0/11 wins
own_graveyard_count >= 8: 2/10 wins
```

Observed concrete failure:

```text
game 1 decision 84
turn=8 life=12
selected=Cast Dread Return
Dread Return visible in hand=true, visible in graveyard=false
own graveyard_count=3
value_score=+0.045488
result=loss
```

## Read

The Affinity problem is mixed, not just one Spy-timing bug:

- 9/16 games never cast Spy; 7 of those were losses.
- Spy was not sufficient when it happened; only 1/7 Spy-cast games won.
- One logged loss explicitly hard-cast `Dread Return` from hand in a low-graveyard
  state, which points to a generic action-source/zone representation issue rather
  than a card-name issue.
- Affinity public-board pressure dominates this sample: every game where the
  opponent reached 10 or more battlefield permanents was lost.
- The value head is useful in this narrow sample at midgame decision separation,
  so another small value calibration branch is not the highest-EV next move.

## Next experiment

Run a narrower generic representation probe: source-zone candidate features.
This should mark whether an action source is currently in hand, graveyard,
battlefield, or exile, plus whether a spell is not from hand. That remains
thesis-clean because it applies to all decks and all cards with non-hand casting
or non-battlefield abilities; it does not name Spy, Dread Return, Lotleth Giant,
or Affinity hate cards.
