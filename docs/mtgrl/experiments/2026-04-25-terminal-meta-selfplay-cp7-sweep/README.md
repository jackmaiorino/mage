# 2026-04-25 Terminal Meta Selfplay CP7 Sweep

## Question

Can terminal-reward-only meta selfplay over a small Pauper pool produce playable policies for complex decks, especially Spy Combo and Jund Wildfire, without intermittent rewards or online MCTS?

## Setup

Training used the four active profile/deck pairs from `pauper_spy_pbt_registry.json`:

| Profile | RL deck |
| --- | --- |
| `Pauper-Spy-Combo-Value` | `Deck - Spy Combo.dek` |
| `Pauper-Wildfire-Value` | `Deck - Jund Wildfire.dek` |
| `Pauper-Rally-Anchor-Value` | `Deck - Mono Red Rally.dek` |
| `Pauper-Affinity-Anchor-Value` | `Deck - Grixis Affinity.dek` |

Core training assumptions:

- `OPPONENT_SAMPLER=meta`, using random active profile-vs-profile games from the four-deck pool.
- `SELFPLAY_OPPONENT_TRAINING=1`, so both RL sides of selfplay contribute training data.
- `RL_HEURISTIC_STEP_REWARDS=0`.
- `MCTS_TRAINING_ENABLE=0`, `ISMCTS_ENABLE=0`.
- `USE_GAE=0`.
- Actor/learner async pipeline enabled with bounded train queues and VRAM guard.

This was intentionally a terminal-reward-only experiment. The goal was to see whether the generic policy/value learner could reach credible play without deck-specific shaping.

## Checkpoint

The CP7 sweep froze local `model_latest.pt` checkpoints before evaluation.

Raw artifact directory:

`local-training/local_pbt/cp7_eval_sweeps/20260425T182721Z_cp7_skill7_20g`

Sweep timing:

- Started: `2026-04-25T18:27:29Z`
- Ended: `2026-04-25T19:42:28Z`

Frozen checkpoint episodes from `manifest.json`:

| Profile | Episode |
| --- | ---: |
| `Pauper-Spy-Combo-Value` | 17,464 |
| `Pauper-Wildfire-Value` | 17,466 |
| `Pauper-Rally-Anchor-Value` | 17,464 |
| `Pauper-Affinity-Anchor-Value` | 17,342 |

## Evaluation

Evaluation command path:

`scripts/run_cp7_eval_sweep.py`

Evaluation matrix:

- 4 RL profiles.
- 4 CP7 opponent decks: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity.
- CP7 skill: 7.
- Requested games: 20 per profile/opponent matchup.
- Parallel matchup JVMs: 4.
- Total requested games: 320.
- Completed counted games: 318.

Two games timed out and were not counted:

- Spy Combo profile vs CP7 Jund Wildfire: 19/20 completed.
- Wildfire profile vs CP7 Jund Wildfire: 19/20 completed.

Counting both missing games as losses does not materially change the result.

## Results

Profile aggregate:

| Profile | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| `Pauper-Spy-Combo-Value` | 5 | 79 | 6.3% |
| `Pauper-Wildfire-Value` | 15 | 79 | 19.0% |
| `Pauper-Rally-Anchor-Value` | 9 | 80 | 11.3% |
| `Pauper-Affinity-Anchor-Value` | 20 | 80 | 25.0% |
| Overall | 49 | 318 | 15.4% |

Per-matchup results:

| RL profile | CP7 deck | Wins | Games | Winrate |
| --- | --- | ---: | ---: | ---: |
| `Pauper-Spy-Combo-Value` | Spy Combo | 5 | 20 | 25.0% |
| `Pauper-Spy-Combo-Value` | Jund Wildfire | 0 | 19 | 0.0% |
| `Pauper-Spy-Combo-Value` | Mono Red Rally | 0 | 20 | 0.0% |
| `Pauper-Spy-Combo-Value` | Grixis Affinity | 0 | 20 | 0.0% |
| `Pauper-Wildfire-Value` | Spy Combo | 10 | 20 | 50.0% |
| `Pauper-Wildfire-Value` | Jund Wildfire | 4 | 19 | 21.1% |
| `Pauper-Wildfire-Value` | Mono Red Rally | 0 | 20 | 0.0% |
| `Pauper-Wildfire-Value` | Grixis Affinity | 1 | 20 | 5.0% |
| `Pauper-Rally-Anchor-Value` | Spy Combo | 7 | 20 | 35.0% |
| `Pauper-Rally-Anchor-Value` | Jund Wildfire | 1 | 20 | 5.0% |
| `Pauper-Rally-Anchor-Value` | Mono Red Rally | 1 | 20 | 5.0% |
| `Pauper-Rally-Anchor-Value` | Grixis Affinity | 0 | 20 | 0.0% |
| `Pauper-Affinity-Anchor-Value` | Spy Combo | 11 | 20 | 55.0% |
| `Pauper-Affinity-Anchor-Value` | Jund Wildfire | 5 | 20 | 25.0% |
| `Pauper-Affinity-Anchor-Value` | Mono Red Rally | 1 | 20 | 5.0% |
| `Pauper-Affinity-Anchor-Value` | Grixis Affinity | 3 | 20 | 15.0% |

## Conclusion

This experiment failed the CP7 gate.

The weak value accuracy was a warning, but the CP7 sweep is the decisive evidence. Terminal-reward-only meta selfplay did not produce credible CP7 Skill 7 performance for any profile by roughly 17k episodes. Spy and Wildfire remained far below the target, and the anchor decks did not stabilize the system.

Do not continue this exact setup blindly. The next experiment should add a stronger policy-improvement signal while preserving terminal win/loss as the reward objective.

## Notes

- `scripts/run_cp7_eval_sweep.py` now makes this sweep repeatable.
- The first full run parsed results from JVM stdout. Per-matchup `EVAL_RESULTS_FILE` writes failed because the result directory did not exist; the runner was patched afterward.
- Parallel eval JVMs also tried to bind the same metrics port; the runner was patched afterward with `METRICS_PORT=0`.
- Training was left stopped after this sweep.
