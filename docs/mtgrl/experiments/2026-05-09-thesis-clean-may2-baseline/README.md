# Thesis-Clean May 2 Spy Baseline

Date: 2026-05-09

## Objective

Re-establish the canonical thesis-clean Spy baseline from the May 1-2 four-profile terminal-only self-play run before starting new experiments.

The checkpoint evaluated here is `Pauper-Spy-Combo-Value` at episode 200,919, with `model_latest.pt` mtime `2026-05-02T11:40:45Z`.

## Thesis Boundary

Allowed:

- Terminal win/loss returns only.
- Generic league evaluation against the four active Pauper decks.
- No train-time MCTS or ISMCTS for this baseline.

Explicitly disabled:

- `RL_HEURISTIC_STEP_REWARDS=0`
- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `ISMCTS_ROLLOUTS_PER_TURN=0`

No action-text regex filters, Spy-specific terminal modes, Spy-reachable hand pools, or selective MCTS keyword gates were used.

## Registry

`local-training/local_pbt/thesis_clean/20260509_may2_clean_spy_value_eval_registry.json`

## Results So Far

### CP1, 128 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260509_may2_clean_spy_cp1_eval128`

Overall: `185/512 = 36.13%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 91 | 128 | 71.09% |
| Jund Wildfire | 48 | 128 | 37.50% |
| Mono Red Rally | 33 | 128 | 25.78% |
| Grixis Affinity | 13 | 128 | 10.16% |

### CP3, 128 Games Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260509_may2_clean_spy_cp3_eval128`

Overall: `177/512 = 34.57%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 99 | 128 | 77.34% |
| Jund Wildfire | 37 | 128 | 28.91% |
| Mono Red Rally | 25 | 128 | 19.53% |
| Grixis Affinity | 16 | 128 | 12.50% |

### CP7, 128 Games Requested Per Matchup

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260509_may2_clean_spy_cp7_eval128_r1`

Overall: `142/500 = 28.40%`

Some long Wildfire/Affinity jobs returned fewer than 8 games before the per-job timeout, so totals are slightly below 128 for three matchups.

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 80 | 127 | 62.99% |
| Jund Wildfire | 27 | 119 | 22.69% |
| Mono Red Rally | 19 | 128 | 14.84% |
| Grixis Affinity | 16 | 126 | 12.70% |

## Interpretation

The larger CP1/CP3/CP7 samples confirm the May 2 clean checkpoint is meaningfully better than the recent single-deck off-thesis imitation/CF branch on mirror and learns some Wildfire play, but its aggregate is pulled down hard by Rally and especially Affinity. This matches the original May 2 read: the policy learned part of the Spy plan without explicit Spy teaching, but not robust matchup adaptation under aggro/artifact pressure.

## Next

Start a thesis-clean continuation run from this checkpoint. Keep the same four-profile terminal-only self-play setup, but add generic sampling pressure toward bad matchups rather than any Spy-specific labels or features.
