# Next Experiment: MCTS-Guided Policy Distillation

## Rationale

The failed CP7 sweep suggests the learner is not discovering long tactical lines from terminal reward alone. Spy Combo is the clearest case: many winning lines require several precise actions before the terminal reward appears.

The next experiment should add search as a policy-improvement mechanism, not as reward shaping. Rewards remain terminal win/loss. MCTS supplies better action targets and action selection during data generation.

## Hypothesis

MCTS-guided selfplay will improve Spy Combo and Jund Wildfire faster than pure policy sampling because search can expose winning or stabilizing lines that are too sparse for the current policy to discover.

## Scope

Start with a two-profile pilot:

- `Pauper-Spy-Combo-MCTS-Distill`
- `Pauper-Wildfire-MCTS-Distill`

Use the current four-deck active meta pool as opponents, but keep the first run focused on whether MCTS improves the two hard profiles.

Do not start with all four profiles unless the MCTS throughput is acceptable. The previous CP7 sweep showed Rally and Affinity are not strong enough to justify spending the first MCTS budget on them.

## Training Design

Core settings:

- Terminal reward only: keep `RL_HEURISTIC_STEP_REWARDS=0`.
- Enable MCTS during training: `MCTS_TRAINING_ENABLE=1`.
- Use one sampled opponent hidden-hand determinization per search unless code inspection shows the current MCTS path already does this differently.
- Record MCTS visit distributions into the existing `mcts_visits` training path.
- Train policy on MCTS-improved targets and value on terminal outcomes.
- Train both sides when both players are RL profiles.

Initial search budget:

- Start with 16 to 32 rollouts per decision.
- Cap search depth at the smallest value that can still see Spy Combo setup lines, likely 6 to 8 actions.
- Add a per-decision wall-clock cap so throughput cannot collapse silently.

Throughput guardrails:

- Target at least 0.3 to 0.5 completed episodes/sec in the pilot.
- If MCTS drops below that, reduce runners before reducing rollout quality.
- Track MCTS activations, rollouts/sec, average decision latency, and timeout count.
- For MCTS-heavy runs, prefer game-level concurrency over more rollout threads per game. A thread dump during the Phase 0 eval showed rollout workers mostly parked inside `SharedGpuPythonModel.scoreCandidates`, so low whole-machine CPU can mean inference-latency stalls rather than enough compute.
- Use a short inference batch timeout for MCTS. The tuned Phase 0 restart used `PY_BATCH_TIMEOUT_MS=3`, `GPU_SERVICE_LOCAL_BATCH_TIMEOUT_MS=1`, `SCORE_WORKER_THREADS=4`, and higher matchup concurrency; GPU-service throughput improved materially versus the under-parallelized first run.

## Phase 0: MCTS Eval Sanity Check

Before training, run inference-time MCTS on the frozen failed checkpoints.

Goal:

- Determine whether MCTS can improve the current policy at all.

Small eval:

- Spy Combo vs CP7 Skill 7 on Spy, Rally, and Affinity.
- Wildfire vs CP7 Skill 7 on Spy, Rally, and Affinity.
- 10 games per matchup.
- Compare no-MCTS vs MCTS with the same checkpoints.

Pass signal:

- Any hard-profile aggregate gain of at least 10 percentage points, or clear game logs where MCTS finds materially better combo/setup lines.

Fail signal:

- No winrate improvement and no qualitative evidence of better lines. If this happens, MCTS is probably not wired deeply enough or the search policy/value is too weak to help.

Result on 2026-04-25:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260425T202154Z_phase0_mcts_harddecks_10g_fastbatch`
- Configuration: flat `PolicyValueMCTS`, `MCTS_ROLLOUT_DEPTH=0`, `MCTS_ITERATIONS=8`, `MCTS_DETERMINIZATIONS=4`, 10 games per matchup.
- Spy Combo: 0/30 across Spy, Rally, and Affinity.
- Wildfire: 4/30 across Spy, Rally, and Affinity.
- Decision: do not start training on flat 1-ply MCTS targets. It is wired and measurable, but it does not improve the frozen checkpoints and can be worse than no MCTS.

Phase 0b:

- Run a tiny `MULTI_PLY_MCTS=1` sanity check before any MCTS training.
- Use a deep but small budget: one determinization, 4 to 8 iterations, `MCTS_MAX_OUR_ACTIONS=6`, and a per-search wall-clock cap.
- Start with 1 to 2 games per hard matchup. This is a correctness/trajectory check, not a winrate estimate.
- If multi-ply cannot produce better lines or a nonzero hard matchup, stop the MCTS-training path and focus on curriculum/generalist training instead.

Phase 0b result on 2026-04-25:

- First attempt exposed a timeout bug: inline multi-ply MCTS returned `false` on deadline and let the simulated engine keep passing priority for about 304 seconds. `MCTSSimPlayer` now throws `WalkTerminated` on inline deadline/halt so the search unwinds.
- Fixed run: `local-training/local_pbt/cp7_eval_sweeps/20260425T205131Z_phase0b_multiply_mcts_1g_timeoutfix`
- Configuration: `MULTI_PLY_MCTS=1`, `MCTS_ITERATIONS=4`, `MCTS_DETERMINIZATIONS=1`, `MCTS_MAX_OUR_ACTIONS=6`, `MCTS_TIMEOUT_MS=750`, `MCTS_ITER_TIMEOUT_MS=300`, one game per matchup.
- Results: Spy vs Spy 1/1, Spy vs Affinity 0/1, Wildfire vs Spy 0/1, Wildfire vs Affinity 0/1.
- Decision: multi-ply is now operational, but too slow for broad always-on training at this budget. Use it only as targeted expert/search data unless a reduced selective budget proves much faster.

## Phase 1: Short MCTS-Distillation Pilot

Run 1k to 2k MCTS-generated episodes per hard profile.

Profiles:

- Spy Combo MCTS-distill profile.
- Wildfire MCTS-distill profile.

Opponent mix:

- 70% active meta selfplay/cross-profile.
- 30% CP7 Skill 1/3/5/7 mix, weighted toward cheap skills for throughput.

Why not pure CP7:

- CP7 Skill 7 is too CPU expensive for training throughput.
- CP7 remains useful as an anchor, but the previous work showed full CP7 training can make convergence too slow.

Evaluation cadence:

- Run a reduced CP7 sweep every 1k MCTS-generated episodes.
- 10 games per hard-profile matchup.
- Full 20-game sweep only after a reduced sweep shows movement.

## Phase 2: Scale Or Stop

Scale if both are true:

- Spy or Wildfire improves by at least 10 percentage points over the failed checkpoint aggregate.
- Throughput remains high enough to reach 5k MCTS-generated episodes in less than roughly 24 hours locally.

Stop or redesign if either is true:

- MCTS training remains below 0.2 episodes/sec after tuning.
- CP7 reduced sweeps show no improvement after 2k MCTS-generated episodes per hard profile.

## Success Gate

The immediate success gate is not 50% aggregate yet. The next experiment succeeds if it proves search-guided data generation changes the trajectory:

- Spy Combo reaches at least 20% aggregate vs CP7 Skill 7 across the four-deck pool, or doubles from the 6.3% baseline.
- Wildfire reaches at least 30% aggregate vs CP7 Skill 7, or improves by at least 10 percentage points from the 19.0% baseline.
- At least one matchup that was 0% becomes nonzero with a meaningful sample.

If that happens, extend to 5k to 10k MCTS-generated episodes and re-run the full CP7 sweep.

## Implementation Checklist

1. Confirm current MCTS training path writes `mcts_visits` and that the Python learner consumes it.
2. Add metrics for MCTS rollout count, decision latency, rollout timeout, and MCTS-policy KL if missing.
3. Create new profile entries for Spy/Wildfire MCTS-distill so the failed value-only checkpoints remain preserved.
4. Run Phase 0 no-MCTS vs MCTS eval comparison.
5. If Phase 0 passes, start Phase 1 MCTS-distillation training.
6. Run reduced CP7 sweeps at 1k and 2k MCTS-generated episodes.
7. Decide whether to scale to a full four-profile MCTS run.

## Side Investigation: Generalist Vs Per-Deck Specialists

Question:

Would a single model trained across all four decks converge faster in wall-clock time than four per-deck profiles?

Short answer:

It is worth testing, but the expected gain is not literally 4x deck-specific throughput.

Why:

- With four specialists, each model only receives its own deck's episodes.
- With one generalist, one model receives all episodes, so optimizer updates and shared representation learning happen from the full stream.
- But if deck sampling is uniform, each deck still appears in roughly 25% of total games. The deck-specific data rate is similar to the specialist case when total game runners are held constant.
- The potential win is transfer: mana sequencing, combat, removal timing, mulligan structure, and generic XMage action handling can be learned once instead of four times.
- The potential loss is interference: Spy Combo and Wildfire may need very different priors than Rally/Affinity, and a shared model may average away important deck-specific lines.

Important implementation caveat:

The current implementation is already partly deck-conditioned: `StateSequenceBuilder` includes the RL player's hand, graveyard, and library card tokens in the gameplay state, and the mulligan model receives hand IDs plus deck IDs. That makes a generalist feasible without adding a profile-id input first, though an explicit deck/profile embedding may still be useful later if the generalist shows interference.

Recommended bakeoff:

- Keep the MCTS Phase 0 hard-deck sanity eval as the immediate next step.
- If MCTS shows useful lift, run a generalist-vs-specialist bakeoff with the same total game-runner budget.

Bakeoff arms:

| Arm | Profiles | Training pool | Notes |
| --- | --- | --- | --- |
| Specialist | Spy + Wildfire MCTS profiles | Each profile plays its own deck | Lower interference, lower per-model update volume |
| Generalist | One `Pauper-Generalist-MCTS-Distill` profile | All four active decks | Higher shared update volume, needs deck conditioning |

Minimum fair comparison:

- Same wall-clock budget.
- Same total number of game runners.
- Same MCTS settings.
- Same CP7 reduced eval cadence.
- Eval reports per RL-piloted deck, not just aggregate generalist winrate.
- Track total episodes/sec and per-deck episodes/sec separately. A generalist gets more updates into one model and may batch better, but uniform deck sampling still gives each deck only a fraction of the games.

Decision rule:

- Prefer the generalist if it improves Spy/Wildfire at least as fast as specialists and does not collapse Rally/Affinity.
- Prefer specialists if the generalist improves aggregate winrate but still misses Spy/Wildfire lines.
- Consider a hybrid if the generalist learns common play faster: pretrain a generalist, then fork deck specialists for MCTS distillation.
