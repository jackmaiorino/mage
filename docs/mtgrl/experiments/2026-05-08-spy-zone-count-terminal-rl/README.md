# Spy Zone-Count Terminal RL Experiment

Date: 2026-05-08

## Question

Is Spy Combo failing because the model can technically see the library, but has to learn count-like facts from many card tokens?

The state sequence already includes own library cards and recent logs showed zero sequence truncation. However, aggregate zone-count features were disabled by default. This experiment enabled generic hand/library/graveyard land and creature counts with `RL_ZONE_COUNT_FEATURES_ENABLE=1`.

This keeps the experiment generic and terminal-only: no deck-specific reward, no rule constraint, and no Spy-specific action override.

## Setup

- Source checkpoint: `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`
- New profile: `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Training run: `local-training/local_pbt/value_rl/20260508_spy_fastt5_zone_counts_r24_b4`
- Episodes: 5,000
- Training mode:
  - terminal reward only
  - hybrid self-play / CP1, `HYBRID_SELFPLAY_P=0.70`
  - separate critic encoder enabled
  - zone counts enabled
  - encoder warmup not frozen so the new feature slots could be learned
- Eval opponent: CP1

## Throughput

The 5k run completed from about 2:28 AM to 3:01 AM ET. Throughput warmed up near 3.0 to 3.6 eps/s, then later ran mostly 2.0 to 2.8 eps/s after exports. The run completed cleanly without VRAM overcommit.

The system was not CPU-starved during the run. The learner queue backed up while full-model policy/value updates ran, so adding more Java runners would mostly have filled the pending-train queue faster.

## CP1 Result

Full 32-game-per-matchup sweep:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Affinity | 3 | 32 | 9.4% |
| Wildfire | 14 | 32 | 43.8% |
| Rally | 3 | 32 | 9.4% |
| Spy mirror | 18 | 32 | 56.2% |
| Total | 38 | 128 | 29.7% |

This ties the separate-critic total score, but the distribution changed. Wildfire improved materially; Affinity and Rally stayed bad.

## Logged Action Health

Logged CP1 sample: 11 wins / 32 games.

- Land play stayed healthy: 170 selected from 181 opportunities, 0 pass-over-land cases.
- Mulligans remained broken: 32/32 hands kept, 31/32 kept hands had 0-1 true lands.
- Spy became more selective, but still wrong when selected: 24 casts from 66 opportunities, and 22/24 casts still had hidden true lands.
- Dread Return timing remained broken: 22 flashbacks, 22/22 premature and not combo-ready.
- Lotleth target selection was still correct when available: 6/6.

Value-head diagnostics:

- All decisions: AUC 0.626
- Nontrivial decisions: AUC 0.642
- Critical combo decisions: AUC 0.815
- Per-game mean value: AUC 0.866

The value head improved on critical combo states, but the policy still did not use that information well enough to gate Spy or Dread Return.

## Selective MCTS Probe

Small CP1 selective MCTS probe:

- Profile: `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Run: `20260508_spy_fastt5_zone_counts_5k_cp1_selective_mcts4_logged`
- Games: 4 per matchup, 16 total
- MCTS: flat policy-value MCTS, 16 iterations, 1 determinization, depth 3
- Selective keywords: `Balustrade Spy,Dread Return`
- Result: 6/16

Logged action health with MCTS:

- Mulligans remained broken: 16/16 hands kept, 15/16 kept hands had 0-1 true lands.
- Spy casts were still bad: 15/15 casts had hidden true lands.
- Dread Return was still bad: 6/6 flashbacks were premature and not combo-ready.

MCTS activated on the intended decisions, but flat MCTS still rated some bad Spy/Dread lines highly. This does not justify scaling train-time flat MCTS as the next expensive local run.

## Conclusion

Do not promote this checkpoint.

Zone counts helped representation enough to improve the value signal and the Wildfire matchup, but they did not solve the core action-selection problem. The main remaining failure is no longer "the state cannot express the fact"; it is "the policy does not reliably convert the value signal into action timing."

## Next Direction

The next experiment should focus on policy improvement from terminal-value counterfactuals, not another plain RL extension:

- generate counterfactual labels at Spy/Dread tactical decision points from terminal outcomes or value-guided search
- train the policy to avoid proven-losing tactical actions while preserving terminal-only reward semantics
- keep zone counts enabled as the new default for Spy experiments

Flat MCTS can still be useful as a label generator or diagnostic, but the current eval-time behavior says it should not be scaled blindly as the next full training approach.
