# Thesis-Clean Public Board Features

Date: 2026-05-13

## Question

The compact Affinity log review showed losses where the value head underestimated public-board pressure: wide Affinity boards, attackers, and lethal/nonlethal pressure were not reflected reliably in logged value scores. Can a generic public-board aggregate representation help without explicit Spy knowledge?

This is thesis-clean:

- no Spy terminal mode;
- no card-name regex labels;
- no Spy candidate facts;
- no heuristic step rewards;
- no MCTS/ISMCTS gates;
- terminal win/loss training only.

## Code Change

Added default-off player-stat features:

```text
RL_PUBLIC_BOARD_FEATURES_ENABLE=1
```

When enabled, `StateSequenceBuilder.embedPlayerStats` writes generic aggregate battlefield features into the player and opponent stat tokens:

```text
permanents
nonland permanents
untapped creatures
tapped creatures
attacking creatures
blocking creatures
summoning-sick creatures
token permanents
total creature power
untapped creature power
attacking creature power
max creature power
```

These are public state features derivable from the visible battlefield and apply to every deck.

Validation:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Eval-Only Perturbation Check

Before training, the accepted checkpoint was evaluated with `RL_PUBLIC_BOARD_FEATURES_ENABLE=1` to ensure the new inputs did not immediately break inference.

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_public_board_evalonly_cp7_g4
```

Result:

```text
Overall: 8/16 = 50.00%
Spy mirror: 4/4
Jund Wildfire: 1/4
Mono Red Rally: 2/4
Grixis Affinity: 1/4
```

This was not a promotion result, but it was stable enough to justify a small training smoke.

## Clone

Created:

```text
Pauper-Spy-Combo-Value-PublicBoard-20260513
```

Source:

```text
accepted Pauper-Spy-Combo-Value model_latest.pt
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260513_thesis_clean_public_board_spyclone_registry.json
local-training/local_pbt/thesis_clean/20260513_thesis_clean_public_board_spyclone_eval_registry.json
```

Training settings:

```text
TOTAL_EPISODES=128
TRAIN_PROFILES=1
NUM_GAME_RUNNERS=8
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.25,3:0.25,7:0.50
RL_PUBLIC_BOARD_FEATURES_ENABLE=1
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

Training completed cleanly:

```text
127 recorded episodes
final recent winrate: 0.160
MCTS activations: 0
model_latest.pt SHA-256: 4F4B54840B20
```

## Reduced CP7 Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_public_board_spyclone128_cp7_g4
```

Result:

```text
Overall: 7/16 = 43.75%
Spy mirror: 4/4
Jund Wildfire: 1/4
Mono Red Rally: 1/4
Grixis Affinity: 1/4
```

The accepted CP7 anchor is `108/242 = 44.63%`, with Affinity at `13/62 = 20.97%`. This branch is neutral overall on a tiny sample and does not improve the weakest Affinity slice.

## Verdict

Rejected as a promotion or HPC candidate.

The feature is kept default-off because it is thesis-clean and may be useful in a later broader model, but plain terminal training on this representation did not improve the pressure-matchup failure. Do not spend Zaratan budget on public-board representation-only training.

Next work should return to stronger generic policy-improvement targets rather than another representation-only continuation.
