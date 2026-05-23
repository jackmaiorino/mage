# Thesis-Clean Terminal-Only Spy Clone Control

Date: 2026-05-11

## Question

After discovering that several bounded active-profile runs were no-ops due to absolute episode counters, run a real terminal-only control from the accepted Spy checkpoint with reset counters. This isolates whether short single-profile PPO continuation helps without any auxiliary credit mechanism.

## Setup

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-Terminal-20260511
```

Source:

```text
accepted Pauper-Spy-Combo-Value checkpoint
```

Counters reset:

```text
episodes.txt=0
mulligan_episodes.txt=0
```

Training run:

```text
local-training/local_pbt/autonomous_runs/20260511_thesis_clean_terminal_spyclone_500ep
```

Training settings:

- `TOTAL_EPISODES=500`
- `TRAIN_PROFILES=1`
- `NUM_GAME_RUNNERS=8`
- `OPPONENT_SAMPLER=hybrid`
- `HYBRID_SELFPLAY_P=0.50`
- `SKILL_MIX=1:0.34,3:0.33,7:0.33`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- `RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=0`
- `CANDIDATE_Q_LOSS_COEF=0.0`
- `CANDIDATE_Q_BLEND=0.0`
- `RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.0`
- MCTS/ISMCTS disabled

The run reached 500 real episodes and exited cleanly. MCTS diagnostics showed zero activations.

Clone hashes after training:

```text
model=DE59BCDCE400 latest=938B3ECCEC15
```

Accepted active profile hashes were unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Evaluation

Reduced CP1, 16 games per matchup:

```text
Run: 20260511_terminal_spyclone500_spy_cp1_unique_eval16
Total: 32/64 = 50.00%
Spy mirror: 13/16
Jund Wildfire: 10/16
Mono Red Rally: 6/16
Grixis Affinity: 3/16
```

Reduced CP7, 16 games per matchup:

```text
Run: 20260511_terminal_spyclone500_spy_cp7_unique_eval16
Total: 25/64 = 39.06%
Spy mirror: 14/16
Jund Wildfire: 7/16
Mono Red Rally: 2/16
Grixis Affinity: 2/16
```

## Verdict

Rejected. Do not promote and do not use HPC for this single-profile 500-episode continuation shape.

The control matches the candidate-Q branch on CP1 aggregate but is worse on CP7 than the accepted checkpoint, with severe losses into Rally and Affinity. The branch seems to reinforce mirror competence while sacrificing pressure matchups.

## Implication

The local reset-counter fix matters operationally, but short single-profile Spy continuation is not enough. The next clean experiment should avoid the single-profile pressure collapse:

- either train a fresh four-profile clone set with reset counters, so opponent/pilot co-adaptation stays balanced;
- or run a targeted eval/logging diagnostic to identify whether the 500-episode single-profile branches are losing through mulligans, land sequencing, or combo timing.

## Logged Diagnostic

Small CP7 full-log samples were collected after the reduced evals:

```text
Terminal clone: 20260511_terminal_spyclone500_spy_cp7_logged4_full
Accepted:       20260511_accepted_spy_cp7_logged4_full
```

Action-health summaries:

```text
Terminal clone sample: 7/16
Accepted sample:       6/16
```

Key action-health comparison:

| Metric | Terminal Clone | Accepted |
| --- | ---: | ---: |
| pass over land | 0/80 | 4/92 |
| keep 0-1 land | 13/16 | 14/16 |
| Spy casts with hidden lands | 14/14 | 19/19 |
| Spy no-hidden-land opportunities | 0/16 | 0/20 |
| premature Dread flashbacks | 4/4 | 8/8 |
| combo-ready graveyard games | 4/16 | 3/16 |

The clone did not regress because of obvious land sequencing in this sample. It also reduced premature Dread Return frequency relative to accepted. The persistent core failure is unchanged: every Spy cast still happened with hidden real lands estimated, and neither sample produced no-hidden-land Spy windows.

This points away from more short single-profile continuation and toward either balanced multi-profile continuation or a generic representation/search change that lets the agent learn "library composition after hidden information changes" without naming Spy cards in training.
