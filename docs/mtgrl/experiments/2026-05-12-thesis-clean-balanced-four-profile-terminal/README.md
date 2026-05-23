# Thesis-Clean Balanced Four-Profile Terminal Continuation

Date: 2026-05-12

## Question

Does a balanced four-profile terminal-only continuation avoid the pressure-matchup collapse seen in short single-profile Spy continuation?

This is thesis-clean: no heuristic rewards, no Spy combo candidate facts, no Spy terminal labels, no action-text filters, no selective MCTS.

## Setup

Fresh clone profiles:

```text
Pauper-Spy-Combo-Value-BalancedCont-20260512
Pauper-Wildfire-Value-BalancedCont-20260512
Pauper-Rally-Anchor-Value-BalancedCont-20260512
Pauper-Affinity-Anchor-Value-BalancedCont-20260512
```

Training run:

```text
local-training/local_pbt/autonomous_runs/20260512_thesis_clean_balanced_four_profile_terminal_500ep
```

Training settings:

```text
TRAIN_PROFILES=4
NUM_GAME_RUNNERS=8
TOTAL_EPISODES=500
MAX_WALL_SECONDS=7200
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SELFPLAY_OPPONENT_TRAINING=1
SKILL_MIX=1:0.34,3:0.33,7:0.33
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
RL_ZONE_COUNT_FEATURES_ENABLE=0
RL_LIBRARY_COUNT_FEATURES_ENABLE=0
RL_GENERIC_ACTION_HISTORY_FEATURES_ENABLE=0
RL_ARCHETYPE_BELIEF_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
LOAD_OPTIMIZER_STATE=0
```

Note: in `trainAll`, `TOTAL_EPISODES=500` is an absolute per-profile target. The run was stopped after a stalled tail once counters were flat for several minutes:

```text
Spy:      339 episodes
Wildfire: 252 episodes
Rally:    499 episodes
Affinity: 408 episodes
```

The latest exported clone hashes after training:

```text
Pauper-Spy-Combo-Value-BalancedCont-20260512      model=D5C3BA35D7A6 latest=BAFA838148F0
Pauper-Wildfire-Value-BalancedCont-20260512       model=F956F492781D latest=CA6B73141E05
Pauper-Rally-Anchor-Value-BalancedCont-20260512   model=A190B774E650 latest=70661E9D74BB
Pauper-Affinity-Anchor-Value-BalancedCont-20260512 model=177B6899188F latest=D75AD16734CB
```

Accepted active profile hashes were unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Evaluation

Reduced CP7, 16 games per matchup:

```text
Run: 20260512_balanced_four_profile_terminal_partial_spy_cp7_unique_eval16
Total: 23/64 = 35.94%
Spy mirror: 10/16
Jund Wildfire: 5/16
Mono Red Rally: 5/16
Grixis Affinity: 3/16
```

Accepted CP7 reference:

```text
108/242 = 44.63%
```

## Verdict

Rejected. Do not promote and do not run this branch on HPC.

Balanced terminal continuation preserved mirror strength better than the single-profile terminal control, but it still regressed hard-pressure matchups and finished well below the accepted CP7 baseline. The result argues against spending local or Zaratan budget on more plain terminal-only continuation without a stronger generic policy-improvement signal.

## Follow-Up

The next experiment should not be another continuation schedule tweak. The more useful next target is a generic teacher that changes the training signal, for example:

```text
generic uncertainty/disagreement-triggered search data
card-level belief / determinization trained from public history
trajectory-level policy improvement from successful generic rollouts
```

The operational lesson is also clear: future multi-profile local gates should use `TOTAL_EPISODES_DELTA` or explicit lower absolute targets, because `TOTAL_EPISODES` is interpreted per profile in `trainAll`.
