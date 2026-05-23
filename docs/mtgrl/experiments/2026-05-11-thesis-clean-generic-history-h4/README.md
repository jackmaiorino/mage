# Thesis-Clean Generic Action History H4

Date: 2026-05-11

## Question

Can a compact, card-agnostic recent-action history feature improve Spy execution without teaching specific Spy cards or using heuristic rewards?

## Implementation

Added opt-in history features:

- `RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1`
- `RL_GENERIC_ACTION_HISTORY_WINDOW=4`

The feature path writes only generic recent-decision summaries into unused slots of the current player stats token:

- recent pass/play-land/mana-action rates
- recent non-pass-option rate
- recent selected-probability and max-probability summaries
- last generic action type and option statistics

It does not add tokens, shift card/permanent token indexes, name cards, add rewards, or constrain actions.

## Training

Run:

```text
20260511_thesis_clean_generic_history_h4_200ep
```

Setup:

- Registry: `local-training/local_pbt/thesis_clean/20260510_thesis_clean_affinity_pressure_selfplay_registry.json`
- 4 profiles
- 8 runners
- `TOTAL_EPISODES=200`
- `OPPONENT_SAMPLER=self`
- `SELFPLAY_OPPONENT_TRAINING=1`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- MCTS/ISMCTS disabled

Backup:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_200ep
```

## Evaluation

All evals used the history gate enabled.

CP1 reduced:

```text
Run: 20260511_generic_history_h4_200ep_spy_cp1_unique_eval16
Total: 28/64 = 43.75%
Spy mirror: 12/16
Jund Wildfire: 7/16
Mono Red Rally: 4/16
Grixis Affinity: 5/16
```

CP3 reduced:

```text
Run: 20260511_generic_history_h4_200ep_spy_cp3_unique_eval16
Total: 29/64 = 45.31%
Spy mirror: 10/16
Jund Wildfire: 9/16
Mono Red Rally: 6/16
Grixis Affinity: 4/16
```

CP7 reduced:

```text
Run: 20260511_generic_history_h4_200ep_spy_cp7_unique_eval16
Total: 32/64 = 50.00%
Spy mirror: 14/16
Jund Wildfire: 10/16
Mono Red Rally: 5/16
Grixis Affinity: 3/16
```

## Verdict

Reject as a promoted checkpoint and restore the accepted Affinity-pressure checkpoint.

The result is still informative: the history representation appears to help harder CP7 mirror/Wildfire games, but it regresses CP1/CP3 and remains poor into Affinity. This suggests the representation is not harmful in principle, but the tiny continuation overfit the hardest-skill surface instead of improving robust play.

Active models restored from:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_200ep
```

Verified restored hashes:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Next

Test the same generic history representation with a more balanced training/eval surface:

- keep history enabled
- include CP1/CP3/CP7 pressure during training rather than pure self-play
- keep all thesis constraints
- use a small local branch first; do not submit to HPC unless the branch improves CP1/CP3 while preserving the CP7 gain
