# Thesis-clean gamma 0.97 continuation - 2026-05-14

## Question

Does a shorter terminal-return horizon improve hard-matchup credit assignment by
giving late decisions stronger terminal signal and mildly preferring faster wins
or longer losses, without adding heuristic step rewards?

Thesis boundary: clean. Training used terminal win/loss only, no heuristic step
rewards, no Spy terminal mode, no Spy candidate facts, no action-text regex
labels, no generic source-zone/action-class candidate features, and no
MCTS/ISMCTS/search gates. The only curriculum was generic deck-level
Rally/Affinity pressure.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-Gamma097-20260514
```

Start checkpoint:

```text
Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Training registry:

```text
local-training/local_pbt/thesis_clean/20260514_gamma097_train_registry.json
```

Eval registry:

```text
local-training/local_pbt/thesis_clean/20260514_gamma097_eval_registry.json
```

Training settings:

```text
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
deck pool=Spy 1x, Jund 2x, Rally 6x, Affinity 6x
REFERENCE_POLICY_KL_COEF=0.25
PPO_EPSILON=0.05
PPO_GAMMA=0.97
USE_GAE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE=0
```

Training:

```text
local-training/local_pbt/20260514_gamma097_train128.out.log
episodes=128
MCTS activations=0
model_latest.pt sha256=CBD6938C7069710FBF589C1353AB3CA34DEED5A0E5726547099984B8113291CA
```

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_gamma097_pressure_g8
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 2 | 8 | 25.00% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Combined | 3 | 16 | 18.75% |

Reference fresh accepted pressure sample:

```text
Mono Red Rally: 5/8
Grixis Affinity: 2/8
combined: 7/16
```

## Decision

Rejected. Do not submit to HPC.

Lowering `PPO_GAMMA` to `0.97` degraded both pressure matchups and did not
recover the Affinity weakness. The issue does not look like insufficient
near-terminal weighting from Monte Carlo returns. Future thesis-clean work
should either keep the accepted long horizon or test a different generic
policy-improvement/variance-reduction mechanism, such as value-bootstrapped GAE,
rather than shortening the return horizon further.
