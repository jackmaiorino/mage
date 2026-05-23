# Thesis-Clean Candidate-Q Auxiliary Clone

Date: 2026-05-11

## Question

Can a generic selected-action terminal candidate-Q auxiliary improve action selection if it is trained on a fresh accepted-checkpoint clone, with no direct policy-logit shaping during training?

This was the follow-up to the rejected soft return-contrastive clone. It keeps terminal credit action-conditioned, but avoids directly pushing down the policy from lost trajectories.

## Setup

Fresh clone profile:

```text
Pauper-Spy-Combo-Value-QAux-20260511
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
local-training/local_pbt/autonomous_runs/20260511_thesis_clean_candidate_qaux_spyclone_500ep
```

Training settings:

- `TOTAL_EPISODES=500`
- `TRAIN_PROFILES=1`
- `NUM_GAME_RUNNERS=8`
- `OPPONENT_SAMPLER=hybrid`
- `HYBRID_SELFPLAY_P=0.50`
- `SKILL_MIX=1:0.34,3:0.33,7:0.33`
- `CANDIDATE_Q_ONLY=1`
- `CANDIDATE_Q_LOSS_COEF=1.0`
- `CANDIDATE_Q_CRITICAL_ONLY=1`
- `CANDIDATE_Q_HUBER_BETA=0.25`
- `CANDIDATE_Q_FROM_MCTS_TARGETS=0`
- `CANDIDATE_Q_BLEND=0.0`
- `RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.0`

The run reached 500 real episodes and exited cleanly. MCTS diagnostics showed zero activations.

Clone hashes after training:

```text
model=A7B1B0911BD7 latest=EA5497A51550
```

Accepted active profile hashes were unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Evaluation

Eval-time candidate-Q blend:

```text
CANDIDATE_Q_BLEND=0.25
```

Reduced CP1, 16 games per matchup:

```text
Run: 20260511_candidate_qaux_spyclone500_blend025_spy_cp1_unique_eval16
Total: 32/64 = 50.00%
Spy mirror: 11/16
Jund Wildfire: 8/16
Mono Red Rally: 9/16
Grixis Affinity: 4/16
```

Reduced CP7, 16 games per matchup:

```text
Run: 20260511_candidate_qaux_spyclone500_blend025_spy_cp7_unique_eval16
Total: 22/63 = 34.92%
Spy mirror: 10/15
Jund Wildfire: 9/16
Mono Red Rally: 2/16
Grixis Affinity: 1/16
```

## Verdict

Rejected. Do not run larger blends and do not spend HPC on this branch.

The CP1 aggregate was tolerable but not better than the accepted local gate, and the CP7 result collapsed into Rally and Affinity. The candidate-Q scorer learned a terminal-return signal, but selected-action-only terminal labels are still too sparse and biased to trust as an eval-time reranker.

## Next

Stop cycling selected-action-only terminal-credit variants. The next thesis-clean local experiment should target either:

- better data, using generic full-game trajectory mining and offline supervised value calibration without card-name filters; or
- a training-loop fix for the accepted multi-profile line, now that the no-op bounded-run issue has been identified.
