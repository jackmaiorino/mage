# Thesis-clean critic-only terminal value calibration - 2026-05-14

## Question

The current search blocker appears to be value/target quality. Can real-game terminal-return training repair the separate critic enough that generic repaired MCTS stops hurting CP7 Affinity, while leaving the policy frozen?

This is thesis-clean:

- terminal win/loss returns only;
- no heuristic step rewards;
- no Spy-specific terminal mode;
- no card/action regex labels;
- no Spy combo candidate facts;
- no train-time MCTS.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-CriticOnlyTerminal-20260514
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260514_critic_only_terminal_registry.json
```

Training settings:

```text
TOTAL_EPISODES_DELTA=256
TRAIN_PROFILES=1
NUM_GAME_RUNNERS=8
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
VALUE_PAIR_RANK_CRITIC_ONLY=1
POLICY_LOSS_COEF=0
VALUE_LOSS_COEF=5
ENTROPY_LOSS_MULT=0
USE_GAE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

Result:

```text
episodes: 256
model_latest.pt sha256=491E2A650BCBE4429159B747B6EE0908593AEDA680423B1A02187A92768E146A
```

## MCTS Gate

Run:

```text
20260514_critic_only_terminal_mcts_topk4_affinity_g4
```

Settings:

```text
CP7 Grixis Affinity only
MULTI_PLY_MCTS=1
MCTS_ROOT_TOP_K=4
MCTS_ROOT_INCLUDE_PASS=1
MCTS_ITERATIONS=8
MCTS_DETERMINIZATIONS=1
MCTS_ROLLOUT_DEPTH=0
MCTS_SKIP_TOP_PROB=0.80
```

Result:

```text
Grixis Affinity: 0/4
MCTS activations: 307
```

## Verdict

Rejected. Real-game critic-only terminal calibration did not make the repaired generic MCTS backend useful on the weakest matchup. This strengthens the prior read: the current MCTS failure is not fixed by a small amount of additional critic training, even when policy parameters are frozen.

Do not spend HPC scaling this critic-only terminal-value path.
