# Spy Candidate-Q Terminal Action Value Experiment

Date: 2026-05-08

## Question

Can a per-candidate terminal-return scorer improve Spy Combo action timing without adding heuristic or intermittent rewards?

The hypothesis was that the shared value head may know useful state value information but the policy head is not using it to rank legal actions. A small candidate-Q scorer was added on top of the existing candidate representation, trained from terminal Monte Carlo returns for the selected action only.

## Setup

- Source checkpoint: `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`
- New profile: `Pauper-Spy-Combo-FastT5CandidateQ-20260508`
- Training run: `local-training/local_pbt/value_rl/20260508_spy_fastt5_candidate_q_r24_b4`
- Episodes: 5,000
- Runners: 24
- Training mode:
  - terminal reward only
  - candidate-Q scorer only trainable
  - base encoder, policy heads, and value heads frozen
  - Q blend disabled during training
- Eval opponent: CP1
- Eval deck: Spy Combo against the four small-meta CP1 decks

## Throughput

The 5k run completed cleanly from 1:11 AM to 1:40 AM ET, about 29 minutes wall clock. Observed throughput was usually 2.5 to 4.1 episodes per second.

## Results

Baseline to compare against:

- `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`: 38/127 CP1, with one CP7 crash.

Candidate-Q blend 1.0:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Affinity | 8 | 32 | 25.0% |
| Wildfire | 6 | 32 | 18.8% |
| Rally | 5 | 32 | 15.6% |
| Spy mirror | 19 | 32 | 59.4% |
| Total | 38 | 128 | 29.7% |

Candidate-Q blend 2.0:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Affinity | 3 | 32 | 9.4% |
| Wildfire | 11 | 32 | 34.4% |
| Rally | 3 | 32 | 9.4% |
| Spy mirror | 21 | 32 | 65.6% |
| Total | 38 | 128 | 29.7% |

Reduced blend sweep:

| Blend | Wins | Games | Winrate |
| ---: | ---: | ---: | ---: |
| 0.5 | 16 | 64 | 25.0% |
| 2.0 | 22 | 64 | 34.4% |

## Logged Action Health, Blend 2.0

Logged CP1 sample: 8 wins / 32 games.

- Land play was healthy: 157 selected from 163 opportunities, with 0 pass-over-land cases.
- Mulligans remained broken: 32/32 hands kept, and 32/32 kept hands had 0-1 true lands.
- Spy timing remained poor: 16 Spy casts from 25 opportunities, with 15/16 casts still leaving hidden true lands.
- Dread Return timing remained poor: 12 flashbacks, 12/12 premature and not combo-ready.
- Lotleth target selection was correct when actually available: 3/3.

Value-head diagnostics on the logged sample:

- All decisions: AUC 0.617
- Nontrivial decisions: AUC 0.640
- Critical combo decisions: AUC 0.601
- Per-game mean value: AUC 0.870

## Conclusion

Do not promote this checkpoint.

Candidate-Q is not a dead idea, but the selected-action-only terminal target did not provide enough counterfactual information to fix Spy Combo timing. The blend can shift matchups, especially Wildfire in the reduced sweep, but it does not improve the total CP1 score over the separate-critic checkpoint and it leaves the two most important Spy failures intact: keeping unusable openers and flashing back Dread Return before combo-ready graveyards.

## Next Experiment

Before spending more time on action-value or MCTS variants, test whether the model is missing an easy generic state signal:

- enable `RL_ZONE_COUNT_FEATURES_ENABLE=1`
- fine-tune from `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`
- keep terminal-only rewards
- evaluate whether explicit hand/library/graveyard land and creature counts improve:
  - mulligan quality
  - Spy timing with hidden true lands
  - Dread Return timing

This is still generic and terminal-only. It does not encode a Spy heuristic; it exposes zone composition counts that any Magic deck can use.
