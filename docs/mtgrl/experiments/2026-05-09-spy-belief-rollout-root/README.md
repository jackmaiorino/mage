# 2026-05-09 Spy Belief Rollout Root

Goal: test the untried belief/ISMCTS path as an action selector, not just as logging or an auxiliary loss.

## Implementation

Added an eval-only MCTS backend behind `ISMCTS_RANDOM_ROLLOUT_ROOT=1`.

Behavior:

- Use `DeterminizationSampler` to sample an opponent archetype determinization.
- For each legal root candidate, clone the game, apply the determinization, force that candidate, then let `SimulatedPlayerMCTS` play to terminal outcome.
- Score candidates as terminal win `+1`, loss `-1`, other/error `0`.
- Aggregate visits and mean values in the same shape as the existing MCTS hook, so `ComputerPlayerRL` can reuse the current `ISMCTS_ENABLE` eval override path.

This deliberately avoids the current value-head leaf estimate because flat value-MCTS was already a bad teacher.

Validation:

- RL module compile passed.
- Smoke run confirmed the backend fired and logged `backend=belief-rollout`.

## Smoke

Run: `20260509_accepted_belief_rollout_root_smoke_g1`

Settings:

- Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- CP1, 1 game per matchup, 4 games total
- `ISMCTS_ENABLE=1`
- `ISMCTS_RANDOM_ROLLOUT_ROOT=1`
- `ISMCTS_ROOT_DETERMINIZATIONS=1`
- `ISMCTS_ROOT_ROLLOUTS_PER_ACTION=1`
- `ISMCTS_ROOT_SEARCH_TIMEOUT_MS=1000`
- `MCTS_SELECTIVE_ENABLE=1`
- `MCTS_SELECTIVE_KEYWORDS=balustrade spy,dread return`
- No heuristic rewards, no training

Result:

- Total: `1/4`
- MCTS activations: `21`
- Matchups:
  - Spy mirror: `0/1`, 13 MCTS activations
  - Jund Wildfire: `0/1`, 0 activations
  - Mono Red Rally: `1/1`, 2 activations
  - Grixis Affinity: `0/1`, 6 activations

Action health:

- `spy_cast_opportunities`: 20
- `spy_casts`: 6
- `spy_cast_hidden_land_opportunities`: 19/20
- `spy_casts_with_hidden_lands`: 6/6
- `dread_return_flashback_selected`: 1
- `premature_dread_flashback_no_lotleth_graveyard`: 1/1
- `premature_dread_flashback_not_combo_ready`: 1/1

MCTS log pattern:

- Most candidate rows were all-loss, for example values like `[-1.000, -1.000, ...]`.
- When all candidates are losses, the backend can only pick an arbitrary tiebreaker.
- Runtime is already high at one rollout per action: 4 games took about 192 seconds, with the mirror game taking 74.6 seconds.

Conclusion:

Do not scale terminal random-rollout ISMCTS. It is wired correctly enough for a smoke, but `SimulatedPlayerMCTS` terminal rollouts do not find enough Spy combo wins to rank tactical candidates. The root action-health failure is unchanged.

This narrows the belief/search path: useful belief-driven search will need either a stronger rollout policy or a search teacher closer to `TerminalPrefixSearch`, not random terminal playouts.
