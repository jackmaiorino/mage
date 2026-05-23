# Thesis-Clean Generic History Scale-0.25 Training

Date: 2026-05-11

## Question

Scale `0.25` made generic history features much less disruptive in eval-only mode. Does a small thesis-clean continuation with the same scale improve over eval-only?

## Training

Run:

```text
20260511_thesis_clean_generic_history_h4_scale025_200ep
```

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- 4 profiles
- `NUM_GAME_RUNNERS=8`
- `TOTAL_EPISODES=200`
- `OPPONENT_SAMPLER=self`
- `SELFPLAY_OPPONENT_TRAINING=1`
- `RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1`
- `RL_GENERIC_ACTION_HISTORY_WINDOW=4`
- `RL_GENERIC_ACTION_HISTORY_SCALE=0.25`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- MCTS/ISMCTS disabled

Backup:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_scale025_200ep
```

## Evaluation

History gate enabled with scale `0.25`.

CP1 reduced:

```text
Run: 20260511_generic_history_h4_scale025_200ep_spy_cp1_unique_eval16
Total: 32/64 = 50.00%
Spy mirror: 15/16
Jund Wildfire: 8/16
Mono Red Rally: 5/16
Grixis Affinity: 4/16
```

CP7 reduced:

```text
Run: 20260511_generic_history_h4_scale025_200ep_spy_cp7_unique_eval16
Total: 28/64 = 43.75%
Spy mirror: 14/16
Jund Wildfire: 6/16
Mono Red Rally: 5/16
Grixis Affinity: 3/16
```

## Verdict

Rejected and restored.

Training did not improve over the eval-only scale-0.25 diagnostic:

- CP1 stayed `32/64`, but shifted more wins into the mirror and lost non-mirror strength.
- CP7 fell from eval-only `31/64` to `28/64`.

This reinforces that the current online continuation objective is not turning the generic history signal into robust policy improvement.

Active models restored from:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_scale025_200ep
```

Verified restored hashes:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Next

Do not continue online terminal-only history branches at 200 episodes. The representation can be useful as eval perturbation, but the training loop is not exploiting it. Next work should target the learning signal itself:

- generic synthetic-return/value calibration from terminal trajectories, or
- a zero-initialized auxiliary history projection trained jointly from a fresh branch, with larger local gates before HPC.
