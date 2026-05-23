# Thesis-Clean Generic History Balanced-Skill Branch

Date: 2026-05-11

## Question

The pure generic-history H4 branch improved reduced CP7 but regressed CP1/CP3. Can the same representation trained against a balanced CP1/CP3/CP7 hybrid opponent mix keep the CP7 gain without lower-skill regression?

## Training

Run:

```text
20260511_thesis_clean_generic_history_h4_balanced_skill_200ep
```

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- active profile: `Pauper-Spy-Combo-Value`
- `NUM_GAME_RUNNERS=8`
- `TOTAL_EPISODES=200`
- `OPPONENT_SAMPLER=hybrid`
- `HYBRID_SELFPLAY_P=0.50`
- `SKILL_MIX=1:0.34,3:0.33,7:0.33`
- `RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=1`
- `RL_GENERIC_ACTION_HISTORY_WINDOW=4`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- MCTS/ISMCTS disabled

Backup:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_balanced_skill_200ep
```

## Evaluation

History feature gate enabled during eval.

CP1 reduced:

```text
Run: 20260511_generic_history_h4_balanced_skill_200ep_spy_cp1_unique_eval16
Total: 31/64 = 48.44%
Spy mirror: 11/16
Jund Wildfire: 9/16
Mono Red Rally: 6/16
Grixis Affinity: 5/16
```

CP7 reduced:

```text
Run: 20260511_generic_history_h4_balanced_skill_200ep_spy_cp7_unique_eval16
Total: 25/63 = 39.68%
Spy mirror: 11/16
Jund Wildfire: 7/16
Mono Red Rally: 2/16
Grixis Affinity: 5/15
```

## Verdict

Rejected and restored.

Balanced skill pressure improved CP1 relative to the pure-history branch (`31/64` vs `28/64`), but it erased the CP7 gain (`25/63` vs `32/64`) and Rally collapsed badly at CP7. This is not a candidate for HPC.

Active models restored from:

```text
local-training/local_pbt/model_backups/pre_20260511_thesis_clean_generic_history_h4_balanced_skill_200ep
```

Verified restored hashes:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Next

Do not spend HPC on generic history H4 in its current form. The next useful local test should separate representation from training instability:

- eval the accepted checkpoint with history features enabled but no history training, to measure representation perturbation alone; or
- run a longer pure-history continuation only if CP1/CP3 are checked early and do not drift down.
