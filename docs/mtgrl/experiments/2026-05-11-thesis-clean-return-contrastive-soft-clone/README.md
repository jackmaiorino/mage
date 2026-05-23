# Thesis-Clean Soft Return-Contrastive Clone

Date: 2026-05-11

## Question

The original return-contrastive policy loss used coefficient `0.50` and collapsed basic action development. Test whether the same generic terminal-only mechanism is usable at a much smaller coefficient from the accepted thesis-clean checkpoint.

This remains thesis-clean:

- terminal win/loss returns only
- no Spy-specific candidate facts
- no action-text or card-name filters
- no `SPY_LANDLESS_COMBO_WIN`
- no MCTS/selective keyword gate
- no generic action-history feature perturbation

## Setup

The first active-profile launch with `TOTAL_EPISODES=200` was a no-op because the active accepted profiles already had episode counters far above `200`. Accepted hashes stayed unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

To make the bounded run real, a fresh clone profile was created:

```text
Pauper-Spy-Combo-Value-RC005-20260511
```

Clone source:

```text
Pauper-Spy-Combo-Value accepted checkpoint
```

Clone counters:

```text
episodes.txt=0
mulligan_episodes.txt=0
```

Training run:

```text
local-training/local_pbt/autonomous_runs/20260511_thesis_clean_return_contrastive005_spyclone_500ep
```

Training settings:

- `TOTAL_EPISODES=500`
- `TRAIN_PROFILES=1`
- `NUM_GAME_RUNNERS=8`
- `OPPONENT_SAMPLER=hybrid`
- `HYBRID_SELFPLAY_P=0.50`
- `SKILL_MIX=1:0.34,3:0.33,7:0.33`
- `RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.05`
- `RETURN_CONTRASTIVE_CRITICAL_ONLY=1`
- `RETURN_CONTRASTIVE_RETURN_EPS=0.10`
- `RETURN_CONTRASTIVE_NEG_PROB_FLOOR=0.60`

The run reached 500 real episodes and exited cleanly. MCTS diagnostics showed zero activations.

Clone hashes after training:

```text
model=F41E9E31C7E6 latest=B44FA17B260D
```

## Evaluation

Reduced CP1, 16 games per matchup:

```text
Run: 20260511_return_contrastive005_spyclone500_spy_cp1_unique_eval16
Total: 29/64 = 45.31%
Spy mirror: 13/16
Jund Wildfire: 8/16
Mono Red Rally: 4/16
Grixis Affinity: 4/16
```

Reduced CP7, 16 games per matchup:

```text
Run: 20260511_return_contrastive005_spyclone500_spy_cp7_unique_eval16
Total: 28/64 = 43.75%
Spy mirror: 11/16
Jund Wildfire: 6/16
Mono Red Rally: 5/16
Grixis Affinity: 6/16
```

## Verdict

Rejected. Do not spend HPC on this branch.

The lower coefficient avoided the catastrophic 0/128 collapse from the earlier `0.50` experiment, but it still failed to beat the accepted line and remained weak into Rally/Affinity. The final training window also showed low rolling training winrate (`~0.105`), so this is not a promising scale-up candidate.

The active accepted profiles were not modified.

## Next

Do not continue direct return-contrastive policy shaping. The safer generic direction is to make terminal credit action-conditioned without directly pushing the policy:

- fresh clone with selected-action terminal candidate-Q auxiliary only, no eval blend during training;
- evaluate candidate-Q blend as an eval-only knob after the branch trains;
- keep the clone/counter-reset pattern for all future bounded local branches.
