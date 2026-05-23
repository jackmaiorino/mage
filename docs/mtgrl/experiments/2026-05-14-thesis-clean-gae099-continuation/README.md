# Thesis-clean GAE 0.99 continuation - 2026-05-14

## Question

Does enabling generic value-bootstrapped GAE from the accepted checkpoint improve
hard-matchup credit assignment compared with pure Monte Carlo terminal returns?

This tests a thesis-clean variance-reduction mechanism. It uses terminal
win/loss rewards only and does not add Spy-specific labels, card-name filters,
heuristic step rewards, candidate facts, MCTS, or ISMCTS.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-GAE099-20260514
```

Start checkpoint:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_accepted_affinity_logged_cp7_g16/snapshot/rl/profiles/Pauper-Spy-Combo-Value/models/model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

The clone and frozen-reference policy anchor both used this immutable accepted
snapshot path. The canonical `Pauper-Spy-Combo-Value` profile was not used as
the reference because a previous orphaned local trainer had already mutated that
profile name in place.

Training registry:

```text
local-training/local_pbt/thesis_clean/20260514_gae099_train_registry.json
```

Eval registry:

```text
local-training/local_pbt/thesis_clean/20260514_gae099_eval_registry.json
```

Training settings:

```text
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.25
SKILL_MIX=1:0.20,3:0.30,7:0.50
deck pool=Spy 1x, Jund 2x, Rally 6x, Affinity 6x
REFERENCE_POLICY_KL_COEF=0.25
PPO_EPSILON=0.05
PPO_GAMMA=0.995
USE_GAE=1
GAE_LAMBDA_HIGH=0.99
GAE_LAMBDA_LOW=0.99
GAE_AUTO_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE=0
RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE=0
```

Training:

```text
local-training/local_pbt/20260514_gae099_train128.out.log
episodes=128
MCTS activations=0
model_latest.pt sha256=8E4894D91624B6A913258CBFE306E4A28CD65BE3909ABB0325EBAB31AAB55242
```

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_gae099_pressure_g8
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=8
```

Eval manifest confirmed:

```text
USE_GAE=1
PPO_GAMMA=0.995
GAE_LAMBDA_HIGH=0.99
GAE_LAMBDA_LOW=0.99
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
RL_HEURISTIC_STEP_REWARDS=0
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 1 | 8 | 12.50% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Combined | 2 | 16 | 12.50% |

Reference fresh accepted pressure sample:

```text
Mono Red Rally: 5/8
Grixis Affinity: 2/8
combined: 7/16
```

## Decision

Rejected. Do not submit to HPC.

Forcing GAE from the accepted checkpoint degraded both hard pressure matchups.
Together with the `PPO_GAMMA=0.97` failure, this closes the current
return-estimator/horizon quick-win surface. The next thesis-clean work should
not be another small PPO-return variant; it should move to a deeper generic
mechanism or a larger controlled training run only after a local gate shows
positive signal.
