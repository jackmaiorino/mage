# Spy Terminal Line Search Diagnostics

Date: 2026-04-29

## Context

This experiment continued from the restored 80,669-episode `Pauper-Spy-Combo-Value`
checkpoint. The active checkpoint was not trained during this diagnostic. Line-search
training was run against a copied probe profile:

`Pauper-Spy-Combo-LineSearch-Probe`

## Code Changes

- Fixed `scripts/run_spy_line_search.ps1` argument quoting so opening hands with
  spaces, such as `Balustrade Spy`, reach Java intact.
- Added `--initial-prefix` support to `ActionCounterfactualTrainer` and
  `run_spy_line_search.ps1`.
- Added low-noise prefix diagnostics:
  - first mulligan hand
  - first priority hand
  - next candidate texts
- Added ordered-library forcing inside `ActionCounterfactualTrainer` so fixed
  opening-hand probes actually draw the requested hand. The normal `Deck.getCards()`
  order was correct, but the runtime maindeck path used for `Player.useDeck` did not
  preserve the stacked order in this harness.
- Changed main-phase `PassAbility` handling in `ComputerPlayerRL.priorityPlay` so
  choosing pass returns `false` from priority instead of behaving like a normal action.

## Key Runs

### Safe Hand Quoting / Library Probe

Run:

`20260429_spy_safehand_clean_smoke_s1_n15`

Opening sequence:

`Forest,Forest,Forest,Swamp,Balustrade Spy,Tinder Wall,Saruli Caretaker`

Result:

- Fixed hand now appears correctly at mulligan and first priority.
- Prefix search can branch from `KEEP` into real gameplay actions.

### Spy Milestone Search

Run:

`20260429_spy_safehand_keepanchored_spy_reached_s1_n512`

Result:

- Found a `Balustrade Spy` line.
- Winning milestone prefix:

`0/1/1/3/1/1`

Decoded line:

1. Keep
2. Play Forest
3. Cast Saruli Caretaker
4. Cast Tinder Wall
5. Play Forest
6. Cast Balustrade Spy

This produced 6 milestone examples on the copied probe profile only.

### Self-Target Milestone Probe

Run:

`20260429_spy_safehand_selftarget0_milestone_probe`

Result:

- Correct post-Spy target index is `0` for self.
- Prefix `0/1/1/3/1/1/0` reaches the Spy combo milestone.

### Terminal Win Probe

Run:

`20260429_spy_safehand_selftarget0_win_s1_n1024`

Result:

- No terminal win found from the self-targeted Spy prefix.
- Reason: the fast Spy line appears to require sacrificing `Tinder Wall` for mana,
  leaving only two battlefield creatures after Spy resolves. The milestone condition
  can be true from graveyard state, but Dread Return still needs three battlefield
  creatures.

## Evaluation Check

After the pass handling change, a small CP1 eval was run with the correct venv:

`20260429_spy_80669_passfix_cp1_eval_g10_p16_venv`

Result:

- 11/40 overall, 27.5%.
- This is not an obvious improvement over the previous small 15/40 baseline.
- The first failed eval attempt used system Python without `torch`; use
  `.mtgrl_venv\Scripts\python.exe` for eval scripts.

## Conclusions

- The earlier safe-hand line-search failures were tooling issues, not evidence that
  terminal Spy lines cannot be found.
- Milestone-only labels are dangerous for the terminal-only goal: they reward casting
  Spy as soon as possible, including lines that cannot flash back Dread Return.
- The terminal generator needs to reason over turn advancement and creature
  preservation. Raw priority actions produce many `Pass` and mana-ability surfaces,
  which makes search slow and can miss the terminal line.
- The next useful line-search improvement is a higher-level search abstraction:
  preserve terminal reward, but add search-side macros or filters such as "advance to
  next own main phase" and "do not sacrifice the third creature before Dread Return."

## Next Experiment

Implement a terminal-line generator for Spy that uses the same game engine but searches
over a reduced action surface:

1. Keep terminal reward only.
2. Use fixed or sampled opening hands with all true lands out of library.
3. Add a search-side phase macro to advance through priority passes to the next own
   main phase.
4. Add diagnostics for battlefield creature count, graveyard Dread Return, graveyard
   Lotleth Giant, and hidden true lands at each Spy cast.
5. Only emit training examples for actual wins, not Spy milestones.
