# Thesis-Clean Candidate-Q Shared Auxiliary Clone

Date: 2026-05-12

## Question

Can selected-action terminal candidate-Q help as a shared auxiliary representation loss when normal terminal PPO/value training remains active?

This differs from the rejected `QAux-20260511` run: that branch froze everything except the Q scorer and then needed eval-time Q blending. This branch keeps the full model trainable, uses a smaller Q coefficient, and evaluates the policy directly with `CANDIDATE_Q_BLEND=0.0`.

## Setup

Fresh clone:

```text
Pauper-Spy-Combo-Value-QAuxShared-20260512
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
local-training/local_pbt/autonomous_runs/20260512_thesis_clean_candidate_qaux_shared_spyclone_500ep
```

Training settings:

```text
TOTAL_EPISODES=500
TRAIN_PROFILES=1
NUM_GAME_RUNNERS=8
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.34,3:0.33,7:0.33
LOAD_OPTIMIZER_STATE=0
RESET_TRAINING_STATE_ON_LOAD=1
CANDIDATE_Q_ONLY=0
CANDIDATE_Q_LOSS_COEF=0.25
CANDIDATE_Q_CRITICAL_ONLY=1
CANDIDATE_Q_FROM_MCTS_TARGETS=0
CANDIDATE_Q_BLEND=0.0
RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.0
```

Thesis cleanliness:

```text
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
no card-name filters
no Spy terminal mode
no Spy hand pool
```

## Gate Plan

Run the 500-episode local branch first. Then use reduced CP7 as the first gate because recent failed branches look acceptable at CP1 while collapsing under hard pressure.

Promotion criteria before any HPC:

```text
CP7 reduced >= 31/64 and no Affinity/Rally collapse
then CP1 reduced >= accepted local screen
then CP3 reduced before any larger confirmation
```

## Training Result

The branch reached 500 episodes and exited cleanly. The orchestrator command returned nonzero only because an ONNX exporter `FutureWarning` was emitted on stderr after shutdown.

Clone hashes after training:

```text
model=EFA84DE8DB50
model_latest=36FBF4799262
```

Accepted active profile hashes stayed unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Reduced CP7 Screen

Eval-time Q blend:

```text
CANDIDATE_Q_BLEND=0.0
```

Run:

```text
20260512_qauxshared_spyclone500_noblend_spy_cp7_unique_eval16
24/64 = 37.50%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 4 | 16 | 25.00% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 3 | 16 | 18.75% |

## Verdict

Rejected. Do not run CP1, CP3, eval-time Q blend, or HPC for this branch.

Selected-action terminal candidate-Q has now failed in all three local forms tested: Q-only scorer with eval blend, offline branch-return Q-only reranking, and shared auxiliary training. The repeated pattern is mirror competence with persistent collapse into Rally/Affinity pressure. Stop candidate-Q cycling until the data source or training objective changes substantially.
