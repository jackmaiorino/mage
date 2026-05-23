# 2026-04-25 Architecture Validation Plan

## Question

Are we on the right path with an object-centric Transformer policy/value model, structured multi-action heads, terminal win/loss rewards, and selective MCTS or expert iteration for hard Magic decks?

This plan is meant to answer that directly. It separates model capacity, training curriculum, action-head quality, and search usefulness so we do not spend weeks on a setup that is failing for the wrong reason.

## Current Evidence

What we know:

- Terminal-only four-profile meta selfplay reached roughly 17k episodes per profile but failed the CP7 Skill 7 sweep.
- The best full-sweep profile was Affinity at 25.0%; Spy Combo was 6.3%, Wildfire was 19.0%, and overall was 15.4%.
- Flat one-ply MCTS was wired and measurable, but did not improve frozen Spy/Wildfire checkpoints.
- Multi-ply MCTS is now operational after the timeout fix, but is too slow for broad always-on use at the tested budget.
- The current state representation is already partly deck-conditioned because the model sees hand, graveyard, and library card tokens; a single generalist is feasible to test without first adding a profile embedding.

Working hypothesis:

The architecture is still plausible, but the next proof point should be a fast generalist foundation run plus targeted search diagnostics, not broad MCTS training from every decision.

## Principles

- Keep reward terminal-only. Search targets are policy-improvement labels, not shaped reward.
- Use CP7 primarily as an evaluation anchor, not the main training opponent.
- Measure per-deck progress. Aggregate winrate can hide Spy/Wildfire failure.
- Use wall-clock gates. Do not wait for 25k to 100k games before deciding whether the run has signal.
- Prefer fixed-position diagnostics for MCTS before spending whole-game training throughput on search.
- Preserve failed checkpoints and run artifacts so comparisons remain honest.

## Hypotheses

### H1: A Generalist Learns Faster In Wall-Clock

A single model trained across Spy Combo, Jund Wildfire, Mono Red Rally, and Grixis Affinity will learn common Magic mechanics faster than four isolated specialists.

Expected signal:

- Better overall CP7 reduced-sweep performance than the previous specialist terminal-only run at the same wall-clock budget.
- Spy and Wildfire improve without collapsing Rally and Affinity.
- Value accuracy and calibration improve across multiple decks, not only one.

Falsifier:

- The generalist improves aggregate play but still misses Spy/Wildfire lines, or shows severe deck interference.

### H2: The Structured Action Heads Are Sufficient

The multi-action-head design can learn legal, useful Magic decisions if the representation and curriculum are good enough.

Expected signal:

- Lower illegal or failed action rate over training.
- Falling action-type loss for spell casting, activated abilities, targets, modes, and attacks/blocks.
- Sensible top-k action agreement on fixed tactical positions.
- Policy entropy decreases on forced or obvious actions while remaining nonzero on genuinely flexible choices.

Falsifier:

- Winrate stays flat and diagnostics show persistent target/mode/action-type mistakes even in simple board states.

### H3: MCTS Is Useful Only Selectively

Broad MCTS on every decision is too slow and too noisy right now, but multi-ply search may still be valuable as an offline or selective teacher for Spy Combo and Wildfire decisions.

Expected signal:

- On a fixed tactical suite, multi-ply MCTS disagrees with the raw policy in useful spots and finds better lines in qualitative game logs.
- Search latency is bounded enough for offline labeling or selective in-game activation.
- Distilling targeted search labels improves hard-deck reduced CP7 results without destroying throughput.

Falsifier:

- MCTS disagreement is mostly noise, does not produce better lines, or requires latency that makes even offline teacher generation impractical.

## Phase A: Instrumentation Gate

Goal:

Make the next runs diagnosable before starting long training.

Required metrics:

| Metric | Why |
| --- | --- |
| Episodes/sec total | Detect throughput regressions |
| Episodes/sec by RL deck/profile | Detect hidden starvation |
| Rolling winrate by RL deck and opponent deck | Avoid misleading aggregate results |
| Value accuracy/calibration by deck | Tell whether terminal outcome learning is working |
| Policy/action loss by action type | Identify action-head failures |
| Illegal or failed action rate | Catch malformed policy outputs |
| Action entropy by action type | Detect collapse or indecision |
| GPU batch size, wait time, and samples/sec | Diagnose inference/training contention |
| MCTS activations, p50/p95 wall time, timeout count | Bound search cost |
| MCTS depth reached and policy/MCTS disagreement | Tell whether search is actually changing decisions |

Acceptance:

- A run can produce a per-deck report and dashboard data without manual log spelunking.
- Metrics are split by deck/profile, not only global totals.

## Phase B: Generalist Foundation Bakeoff

Goal:

Test whether the project should move from four isolated deck specialists to a shared Pauper generalist as the foundation model.

Training profile:

- `Pauper-Generalist-Value-v2`

Deck pool:

| Deck | Role |
| --- | --- |
| Spy Combo | Hard combo deck |
| Jund Wildfire | Hard resource deck |
| Mono Red Rally | Aggro anchor |
| Grixis Affinity | Synergy/value anchor |

Training setup:

- Random RL deck vs random RL deck from the four-deck pool.
- Train both sides of selfplay.
- Terminal reward only: `RL_HEURISTIC_STEP_REWARDS=0`.
- No MCTS during this phase.
- CP7 anchor should be small, at most 10% to 20% of games, and weighted toward cheaper skill levels if used at all.

Comparison:

- Compare against the existing terminal-only specialist run, using wall-clock time and CP7 reduced sweeps rather than raw episode count alone.
- If possible, keep one specialist control running with the same total machine budget for a short bakeoff.

First gates:

| Gate | Decision |
| --- | --- |
| 2 hours or 10k total games | Run reduced CP7 sweep |
| 6 hours if metrics are improving | Run second reduced CP7 sweep |
| Stop early if throughput or value metrics collapse | Diagnose before continuing |

Reduced CP7 sweep:

- Four RL-piloted decks.
- Four CP7 opponent decks.
- 5 games per matchup.
- CP7 Skill 7 for the main gate; optionally Skill 3 as a cheap trend check.

Pass criteria:

- Overall reduced CP7 winrate beats the previous 15.4% full-sweep baseline, or
- Spy beats 6.3% / Wildfire beats 19.0% by a meaningful margin, and
- Rally/Affinity do not collapse below the already weak previous baselines.

Fail criteria:

- No per-deck improvement after 2 to 6 hours and value/action diagnostics are flat.
- Clear deck interference where the generalist averages away hard-deck behavior.

### Phase B Result: Failed First Gate

Run:

- Profile: `Pauper-Generalist-Value-v2`
- Training: 10k total target, random four-deck self-play, both sides trained, terminal reward only, MCTS off.
- Final value accuracy: `0.37` at episode `10000`.
- Final self-play rolling-200 winrates by RL deck:
  - Spy Combo: `0.415`
  - Jund Wildfire: `0.355`
  - Mono Red Rally: `0.700`
  - Grixis Affinity: `0.600`

CP7 Skill 7 reduced sweep:

- Run directory: `local-training/local_pbt/cp7_eval_sweeps/20260425T235446Z`
- Split-deck 4-by-4 sweep, 5 games per matchup.
- Result: `0/79`, `0.0%`.
- MCTS activations: `0`.
- No eval activation/model-loading errors were found in the sweep logs.

Decision:

- Do not continue this pure-selfplay generalist to the 6-hour gate.
- Pure terminal self-play produced self-play structure but did not transfer to CP7 at all.
- The next run must add an external anchor or action-head diagnostic before more long self-play.

## Phase C: Generalist-To-Specialist Fork Test

Goal:

Determine whether the best path is one model for all decks or a generalist pretrain followed by hard-deck specialists.

Trigger:

- Run this only if Phase B shows the generalist learns broad play but Spy or Wildfire remains weak.
- Do not run this directly from the first Phase B checkpoint if CP7 anchor winrate is zero across all decks; in that case, inspect action/state failures and add an anchor curriculum first.

Profiles:

- Fork `Pauper-Generalist-Value-v2` into a Spy Combo specialist.
- Fork `Pauper-Generalist-Value-v2` into a Wildfire specialist.

Training setup:

- Keep terminal reward only.
- Train each specialist on its own deck against the four-deck meta pool.
- Continue training both RL sides when both sides are model-controlled.
- Keep CP7 as an eval anchor, not the main training source.

Pass criteria:

- Forked specialists improve Spy/Wildfire faster than the prior from-scratch specialists at the same wall-clock budget.
- The fork starts from better value/action diagnostics than a cold specialist.

Decision:

- If the fork works, use generalist pretrain plus specialist fine-tune as the default path for hard decks.
- If the fork does not work, focus on representation/action-head diagnostics before adding more search.

## Phase D: Fixed-Position Search Diagnostic

Goal:

Test whether multi-ply MCTS is a useful teacher before using it in full training.

Position suite:

| Suite | Target size | Examples |
| --- | ---: | --- |
| Spy tactical states | 100 | Combo setup, mana bottlenecks, graveyard threshold, protection choices |
| Wildfire tactical states | 100 | Land sequencing, sacrifice targets, removal timing, value recursion |
| Anchor sanity states | 50 | Straightforward Rally/Affinity actions where search should not thrash |

Search settings to test:

- `MULTI_PLY_MCTS=1`
- `MCTS_DETERMINIZATIONS=1`
- `MCTS_ITERATIONS=2` to `4`
- `MCTS_MAX_OUR_ACTIONS=4` to `6`
- `MCTS_TIMEOUT_MS=150` to `300` for selective viability
- Optional higher-quality oracle run at `MCTS_TIMEOUT_MS=750`

Metrics:

- Search wall time p50/p95.
- Timeout rate.
- Depth reached.
- Leaves evaluated.
- Policy action vs MCTS action disagreement rate.
- MCTS visit entropy.
- Qualitative line review for the highest-impact Spy/Wildfire disagreements.

Pass criteria:

- p95 search time is less than 300 ms for selective online use, or less than 1 second for offline teacher labeling.
- Timeout rate stays below 2%.
- MCTS produces useful disagreement on 10% to 40% of hard-deck tactical states.
- At least several Spy positions show search advancing a plausible combo line that the raw policy misses.

Fail criteria:

- Search mostly agrees with bad policy choices.
- Search disagreement is not qualitatively better.
- Search cost remains too high even for offline labeling.

## Phase E: Targeted Search Distillation Pilot

Goal:

Test the combined architecture: neural generalist or forked specialist plus targeted multi-ply search labels.

Trigger:

- Run this only if Phase D passes.

Design:

- Keep games generated mostly without MCTS for throughput.
- Activate MCTS only on selected tactical states or label them offline from replay/position captures.
- Train policy on MCTS visit/action targets for those states.
- Train value only on terminal game outcomes.

Arms:

| Arm | Description |
| --- | --- |
| Baseline | Best Phase B/C model, no search labels |
| Search-distilled | Same starting checkpoint plus targeted MCTS labels |

Budget:

- 1k to 2k search-labeled hard-deck decisions, or
- 2 hours wall-clock, whichever comes first.

Evaluation:

- Fixed-position action agreement before and after distillation.
- Reduced CP7 sweep for Spy and Wildfire against all four CP7 decks.
- Full 20-game CP7 sweep only if reduced sweep moves.

Pass criteria:

- Spy or Wildfire improves over the matching baseline in reduced CP7.
- Fixed tactical suite improves without increasing illegal/failed action rate.
- Search labeling overhead does not reduce total useful training throughput by more than roughly 25% to 40%.

Fail criteria:

- No hard-deck improvement after the targeted labels.
- Distillation improves position-suite agreement but does not improve games.
- Search overhead dominates actor throughput.

## Phase F: Architecture Decision

After Phases B through E, choose one of these paths:

| Evidence | Decision |
| --- | --- |
| Generalist improves broad CP7 and hard decks | Continue generalist as primary model |
| Generalist learns basics but hard decks lag; forks improve | Use generalist pretrain plus deck specialists |
| MCTS position suite passes and distillation improves games | Add selective/offline expert iteration |
| MCTS position suite fails | Do not spend training budget on MCTS; debug representation/action heads/curriculum |
| Generalist and specialists both fail with bad diagnostics | Revisit state encoding, action heads, and supervised/expert data before more selfplay |

## Success Definition

This architecture path is validated if, within short local runs, we see all of the following:

- Better wall-clock progress than the failed terminal-only specialist baseline.
- Spy Combo and Wildfire trend upward in CP7 reduced sweeps.
- Action-head diagnostics improve in the specific decision types those decks need.
- Search either proves useful as a targeted teacher or is cleanly ruled out without blocking the neural training path.

The next major milestone after validation is a full CP7 Skill 7 4-by-4 sweep where at least one hard deck clears a meaningful threshold:

- Spy Combo at least doubles the 6.3% baseline and trends toward 20%+.
- Wildfire improves by at least 10 percentage points over the 19.0% baseline.
- No anchor deck regresses catastrophically.

The 50% CP7 target remains the long-term bar, but the immediate experiment is designed to answer whether the architecture has a credible slope toward that target.

## Immediate Next Run

Recommended next command sequence:

1. Finish any missing metrics from Phase A.
2. Launch the `Pauper-Generalist-Value-v2` Phase B run with the four-deck random pool:

   ```powershell
   .\scripts\run_arch_validation_generalist.ps1 -TotalEpisodes 10000 -NumGameRunners 96 -StartMonitoring -StartDashboard
   ```

3. Run a reduced CP7 sweep at 2 hours or 10k total games. The reduced sweep must split the generalist's four RL-piloted decks into separate eval rows:

   ```powershell
   .\scripts\run_arch_validation_cp7_reduced.ps1 -GamesPerMatchup 5 -Skill 7
   ```

4. In parallel or immediately after, build the fixed-position Spy/Wildfire search suite for Phase D.

Decision after the first reduced sweep:

- The first reduced sweep failed at `0/79`; the generalist did not have CP7 signal.
- Next experiment should be an anchored-curriculum diagnostic, not a longer pure-selfplay continuation.
- Fixed-position MCTS remains useful as a diagnostic, but MCTS training should wait until the raw policy/value model can beat at least weak anchors.

Recommended next experiment:

1. Create `Pauper-Generalist-Anchor-v1` from the `Pauper-Generalist-Value-v2` checkpoint.
2. Train terminal-only with `OPPONENT_SAMPLER=hybrid`, approximately `70%` self-play and `30%` CP7 Skill 1/3 mix.
3. Keep the four-deck random pool and continue training both RL-controlled sides.
4. Gate after 2 hours or 5k additional episodes, whichever comes first.
5. Evaluate with a cheaper CP7 Skill 3 split sweep first; run Skill 7 only if Skill 3 is nonzero.
6. In parallel, capture several failed CP7 game logs per deck and build a fixed-position action-head suite from the first obvious decision failures.
