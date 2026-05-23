# Thesis-Clean Zone Counts On Canonical Line

Date: 2026-05-11

## Question

Do generic hand/library/graveyard land and creature counts help the accepted thesis-clean multi-deck Spy checkpoint convert terminal reward into better action timing?

This is thesis-clean: no card-name rules, no Spy terminal mode, no action-facts features, no regex labels, no heuristic step rewards, and no search gate. The only change is enabling existing generic state features with `RL_ZONE_COUNT_FEATURES_ENABLE=1`.

## Setup

Starting checkpoint: restored Affinity-pressure checkpoint, backed by `local-training/local_pbt/model_backups/pre_20260510_thesis_clean_moderate_pressure_3h`.

Training registry: `local-training/local_pbt/thesis_clean/20260511_thesis_clean_zone_counts_affinity_pressure_selfplay_registry.json`

Eval registry: `local-training/local_pbt/thesis_clean/20260511_thesis_clean_zone_counts_spy_eval_registry.json`

Opponent pool remains the accepted Affinity-pressure pool:

- Spy Combo 1x
- Jund Wildfire 3x
- Mono Red Rally 3x
- Grixis Affinity 6x

Promotion baseline is the corrected Affinity-pressure checkpoint:

- CP1: `133/256 = 51.95%`
- CP3: `132/256 = 51.56%`
- CP7: `108/242 = 44.63%`

## 3h Training Run

Run: `20260511_thesis_clean_zone_counts_affinity_pressure_3h`

The run completed cleanly with trainer exit `0`.

Episode deltas:

| Profile | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 278,387 | 292,239 | 13,852 |
| Pauper-Wildfire-Value | 242,153 | 251,542 | 9,389 |
| Pauper-Rally-Anchor-Value | 388,723 | 407,952 | 19,229 |
| Pauper-Affinity-Anchor-Value | 286,109 | 304,362 | 18,253 |

Total delta: 60,723 episodes in 3 hours.

## CP1 Result

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_zone_counts_affinity_pressure_3h_spy_cp1_unique_eval64`

Overall: `126/256 = 49.22%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 51 | 64 | 79.69% |
| Jund Wildfire | 34 | 64 | 53.12% |
| Mono Red Rally | 17 | 64 | 26.56% |
| Grixis Affinity | 24 | 64 | 37.50% |

CP1 improved Wildfire compared with the accepted checkpoint but regressed overall, mainly from Rally.

## CP3 Result

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_zone_counts_affinity_pressure_3h_spy_cp3_unique_eval64`

Overall: `107/256 = 41.80%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 48 | 64 | 75.00% |
| Jund Wildfire | 29 | 64 | 45.31% |
| Mono Red Rally | 16 | 64 | 25.00% |
| Grixis Affinity | 14 | 64 | 21.88% |

This is a clear regression from the accepted CP3 baseline.

## CP7 Result

Run: `local-training/local_pbt/cp7_eval_sweeps/20260511_zone_counts_affinity_pressure_3h_spy_cp7_unique_eval64`

Overall: `91/256 = 35.55%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 50 | 64 | 78.12% |
| Jund Wildfire | 19 | 64 | 29.69% |
| Mono Red Rally | 9 | 64 | 14.06% |
| Grixis Affinity | 13 | 64 | 20.31% |

## Conclusion

Do not promote. Zone counts on the canonical multi-deck line reproduced the earlier single-deck finding: the feature is thesis-clean and expressive, but plain terminal-only continuation does not convert it into better Spy play. It appears to destabilize Rally/Affinity timing.

Active models were restored afterward from `local-training/local_pbt/model_backups/pre_20260511_thesis_clean_zone_counts_affinity_pressure_3h`, returning to the accepted Affinity-pressure checkpoint.

Next direction: stop spending full 3h cycles on deck weighting or plain representation toggles. Use a small generic search-training smoke to measure whether current engine optimizations make non-selective AlphaZero-style targets tractable before considering HPC.
