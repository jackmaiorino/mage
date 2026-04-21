# Multi-Ply MCTS with Tree Reuse — Implementation Plan

**Date**: 2026-04-22
**Goal**: Rewrite MCTS to (1) branch correctly on every compound sub-decision, (2) reuse accumulated search across decisions, and (3) approach the correctness ceiling of Cowling-Ward ISMCTS for Magic.

## Motivation

Current MCTS in `PolicyValueMCTS.java` treats each priority call as a flat, 1-ply search over "bare abilities" and lets `SimulatedPlayerMCTS` pick targets randomly during clone execution. That loses the signal that matters most in MTG — target/mode choice is often THE strategic decision (e.g. self-target Cleansing Wildfire vs opp-target). Flat MCTS also rebuilds the tree from scratch every priority call, wasting compute and preventing deep lookahead.

This plan staged the fix into four phases. Each phase is independently testable and produces a measurable quality bump.

## Current state (baseline)

- `PolicyValueMCTS.search()` builds a 1-ply tree of bare `ActivatedAbility` objects
- Uses `DeterminizationSampler` to sample opp hand/library
- For each iteration: clone game → activate ability → either run truncated engine rollout (now disabled) or evaluate V(state) directly
- Pooled rollouts via `ROLLOUT_POOL` ExecutorService
- `CoroutineRollout` + `MCTSSimPlayer` provide engine-pause-at-priority infrastructure (currently used for rollouts; repurposed for phase 1)
- All trees are discarded after `search()` returns

## Architecture target

A persistent, multi-level search tree where:
- **Each node** represents an engine-paused state at a priority/choice prompt
- **Each edge** is a specific decision (spell to cast, target to pick, attack declaration, etc.)
- **V is computed only at truly-leaf states** (engine has finished the current compound action and is back to top-level priority)
- **PUCT at each level** uses the appropriate policy head's prior (action / target / card_select / attack / block)
- **Tree persists across priority calls**: walking to the subtree that matches the actually-taken decision and continuing search from there

## Phase 1 — Multi-ply factored MCTS

**Scope**: new recursive search driver that walks the engine prompt-by-prompt, building a tree where every `genericChoose` is a tree level.

### Design

New files:

- **`MCTSNode.java`** — a tree node holding:
  - `actionType` (ACTIVATE_ABILITY_OR_SPELL, SELECT_TARGETS, SELECT_CHOICE, etc.)
  - `priors: float[]` (one per child candidate, from the head matching actionType)
  - `children: MCTSNode[]` (created lazily on expansion)
  - `Q: float[]`, `N: int[]` (per-child mean value and visit count)
  - `selfN: int` (visits through this node)
  - `engineState` (reference to the paused sim — only for "live" internal nodes; null for pruned siblings)
  - `isTerminal: boolean` (game ended at this node)
  - `leafValue: float` (cached V for terminal/leaf states)

- **`MultiPlyMCTS.java`** — the driver:
  - `SearchResult search(liveGame, selfId, policyPriors, sampler, iterations)`
  - One iteration = selection (walk PUCT down from root to unexpanded node) + expansion (attach one new child using engine resume + next prompt) + evaluation (V(leaf) if this is the last prompt for this compound action) + backup (propagate value up to root)

Modified files:

- **`PolicyValueMCTS.java`** — becomes a thin shim that calls `MultiPlyMCTS` or is replaced entirely. Keep `PolicyValueMCTS.SearchResult` as a stable output type so `ComputerPlayerRL`'s call site doesn't change.

- **`ComputerPlayerRL.java`** — minimal change. Pass `actionType` and a policy-head-dispatch callback into the search so PUCT can ask "what's the prior for child `c` of this node?" and get the answer from the right head.

- **`MCTSSimPlayer.java`** + **`CoroutineRollout.java`** — repurpose the coroutine infrastructure. Each tree node holds a paused sim + a request-pending queue. Expanding a child = sending a response to unblock the sim for one step, capturing the next prompt (or terminal), creating the child node.

### Key design choices

1. **Leaf definition**: a node is a "leaf" (where V is evaluated) when the engine's next prompt is back to the outer-level priority call — i.e. the compound action we were evaluating has fully resolved. Between root and leaf, internal nodes represent "waiting for target", "waiting for mode", etc.

2. **Policy head dispatch**: the PUCT prior at node N depends on N's actionType. The root uses action priors; a target-level node uses target-head priors; etc. Implementation: when expanding, call `ComputerPlayerRL.scoreCandidatesForHeadType(headType)` to get priors for the N's candidates. Cache on the node.

3. **Determinization per tree**: sample a small set of determinizations at the ROOT (not per iteration). Each iteration picks one randomly (weighted by posterior). All tree nodes below a given det carry that det's sampled hand forward. This matches the ISMCTS paper.

4. **Tree traversal loop** (pseudo):
   ```
   fn search(root, iterations):
       for i in 1..iterations:
           node, path = select(root)          # PUCT walk until unexpanded
           child = expand(node)                # one engine step to next prompt
           value = evaluate(child)             # V(s) if leaf else 0 and continue next iter
           backup(path + child, value)         # propagate Q and N up
       return best_action(root)                # by visit count
   ```

### Phase 1 acceptance criteria

- `MultiPlyMCTS` returns same-shape `SearchResult` as `PolicyValueMCTS.search`
- On a simple test case (2 candidates, no targets), behaves identically to current 1-ply MCTS
- On a Cleansing Wildfire scenario (3 target lands), produces distinguishable Q values per target
- `MCTS_TIMING` breakdown: per-leaf V eval still ~28ms on wide model; per-expansion ~30-50ms (engine step + policy head inference)
- No Java `Wrong code usage: game related code must run in GAME thread` errors
- At `MCTS_ITERATIONS=16`, search completes in <3s for typical priority decisions

### Phase 1 tests

1. **Unit**: build a minimal `MCTSNode` tree manually, run `select` + `backup` — verify Q and N math
2. **Integration**: run 16-game eval with multi-ply vs current flat MCTS on same wide checkpoint. Multi-ply should be ≥ flat within noise (not 2pp worse).
3. **Scenarios with targeting**: tabulate cases where Cleansing-on-self vs Cleansing-on-opp gets different Q values. If Q(self) - Q(opp) ≈ 0 always, we have a bug in target encoding.

### Phase 1 effort

3-5 days. Most uncertainty is in the coroutine-pause bookkeeping (each node has its own paused sim; memory footprint scales with tree size).

## Phase 2 — Fast-path tree reuse

**Scope**: persist the MCTSNode tree across priority calls. After the agent takes action a, the new root = old_root.children[a]. Discard siblings.

### Design

New files:
- **`MCTSSession.java`** — per-runner object:
  - `root: MCTSNode`
  - `lastAppliedAction: Integer` (which root child corresponds to the real move)
  - `iterationBudget` remaining (accumulated across priority calls; each call spends some)

Modified files:
- `ComputerPlayerRL` holds an `MCTSSession` field, per-runner. Zero out session on new game.

### Reuse mechanism

Before each `priority()` call:
1. If `session.root != null` and `session.lastAppliedAction != null`:
   - `new_root = session.root.children[session.lastAppliedAction]` (if exists and consistent)
   - Discard siblings, free their sims and child subtrees
   - `session.root = new_root`
2. Continue search from this root

When expanding during search, any sim state previously captured in the subtree is reused directly.

### Correctness caveats (deferred to phase 3)

This phase does NOT verify that the real game state actually matches the subtree's assumed state. Specifically:
- If opp played something the tree didn't expect (a response we didn't sample), the subtree's downstream nodes reflect a hypothetical game that diverged from reality
- If we draw a card we hadn't sampled in our determinization, the subtree's opp-hand assumption is now wrong
- We accept both of these as noise for phase 2 — the tree is mostly right most of the time

In practice: fast-path reuse gives significant speedup (tree grows over the game, accumulated search iterations add up to hundreds/thousands of visits at leaf positions) with modest correctness loss in pathological cases.

### Phase 2 acceptance criteria

- 32-game MCTS eval on same wide checkpoint: win rate ≥ flat MCTS within noise
- Tree sizes grow over the game: log `root.selfN` at each priority call. By mid-game should be 100-500+ visits.
- Effective iterations: `(total_tree_visits_over_game) / (priority_calls)` should be substantially higher than `MCTS_ITERATIONS` alone

### Phase 2 tests

1. **State-equality invariant**: when reusing, log a fingerprint of the new root's expected state vs actual state. Mismatches = candidates for phase 3.
2. **Memory bound**: tree must not grow unboundedly. Add a max-subtree-size cap; evict oldest siblings if exceeded.
3. **Correctness under mulligan / shuffle**: reset session when a shuffle happens or the game restarts.

### Phase 2 effort

3-5 days. Main risk: subtle bugs where subtree-assumed state differs from reality in ways we don't notice (phase 3 fixes this).

## Phase 3 — Determinization-consistent reuse

**Scope**: tag each node with the determinization it was built under. When reusing after observing opp actions, filter subtrees to keep only those consistent with observations. When no determinization is consistent, re-sample with the new constraints.

### Design

Changes to `MCTSNode`:
- `sampledDet: Determinization` — which opp-hand/library sample this subtree assumed

Changes to `DeterminizationSampler`:
- Extend with constraint: "opp has been observed to play cards [X, Y]" → any determinization must include those cards in their original hand
- Method `isConsistent(Determinization, ObservedOpponentActions)` — checks if the sampled hand could have produced the observed plays

Changes to `MCTSSession`:
- Maintains `observedOppActions: List<OppAction>` (spells cast, lands played, triggers activated) since last determinization sampling
- Before reuse: walk tree, mark subtrees whose `sampledDet` is no longer consistent as "stale"
- Stale subtrees are either:
  - Pruned (safer) — next iteration resamples a fresh determinization
  - Downweighted (lossy but faster) — their accumulated Q and N are kept but visit priority is lowered

### Observation tracking

After each opp move in the real game:
1. Record what public-information action they took (spell cast, attack declaration, target chosen)
2. Update observed set
3. On next priority call, apply constraint filter

### Phase 3 acceptance criteria

- When opp plays a card we didn't sample in any current det, tree correctly resamples and doesn't reuse garbage
- `session.staleness_rate` (fraction of subtrees pruned as inconsistent) < 30% on typical games — otherwise our determinization samples are too narrow
- Win rate improves modestly vs phase 2 (full correctness is worth 1-3pp)

### Phase 3 tests

1. **Adversarial opponent**: simulated opp that plays unusual cards (outside common decklist samples). Verify our tree doesn't get confused.
2. **Determinization diversity**: check distribution of sampled opp archetypes over a game. Should broaden as uncertainty grows; narrow as opp commits to play pattern.

### Phase 3 effort

5-7 days. Complex because observation tracking interleaves with engine's event stream.

## Phase 4 — Transpositions

**Scope**: hashmap keyed on canonical game-state hash. Different paths leading to the same state share node statistics.

### Design

New files:
- **`StateFingerprint.java`** — canonical state hash. Combines: zones (sorted card IDs per zone), stack (ordered), life totals, mana pool, counters, phase, turn, priority holder, active continuous effects.

Changes to tree:
- `Map<Long, MCTSNode> transpositionTable` — shared across the whole session
- When expanding a child, first check if the new state is already in the map; if yes, link to existing node (DAG instead of tree)

### Tradeoffs

- Pure correctness win: two paths that lead to the same state should share Q and N (more samples → better estimates)
- Memory bounded by number of distinct states visited (could be smaller than tree total)
- Must guard against cycles (rare in MTG but possible with repeat-resolve effects)

### Phase 4 acceptance criteria

- Fewer total node creations (log ratio `expanded / unique_states`)
- No correctness regression on eval

### Phase 4 effort

3-4 days. Main risk: state-fingerprint equality edge cases (e.g. triggers with different arrival order may be semantically equivalent but hash differently).

## Rollout strategy

1. **Ship phase 1 behind feature flag** `MULTI_PLY_MCTS=1`. Default off. Gate at `ComputerPlayerRL.calculateRLAction` — falls back to current flat MCTS.
2. **Measure**: 32-game A/B eval between flat and multi-ply. If multi-ply is at worst ±2pp, keep it.
3. **Enable by default** after validation. Delete the flag.
4. **Phase 2, 3, 4 added incrementally** — each behind its own sub-flag.

## Timeline

| Phase | Effort | Calendar time |
|---|---|---|
| 1. Multi-ply factored MCTS | 3-5 days | ~1 week |
| 2. Fast-path tree reuse | 3-5 days | ~1 week |
| 3. Determinization-consistent reuse | 5-7 days | ~1.5 weeks |
| 4. Transpositions | 3-4 days | ~1 week |
| **Total** | ~2 weeks of engineering | ~5 weeks calendar (with testing, A/B, integration) |

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Coroutine per-node sim is too expensive (memory) | Tree too small to help | Node eviction; cap tree depth at e.g. 3 plys initially |
| Policy head dispatch is more complex than anticipated | Phase 1 slips | Start with action-head-only PUCT; extend head-by-head |
| Engine state fingerprint has subtle bugs | Phase 4 gives wrong results | Extensive unit tests on fingerprint; phase 4 is optional |
| Value head plateau persists | MCTS doesn't pay off regardless | Parallel work on value loss / reward shaping (already in flight) |
| Tree reuse amplifies bugs | Subtle correctness drift | Phase-by-phase acceptance tests; feature flags allow rollback |

## What this plan does NOT include

- Neural architecture changes (wider/deeper nets) — separate track
- Reward shaping — separate track
- MCTS-in-training (AlphaZero distillation) — viable only after phase 1-3 make MCTS cheap enough to run during self-play
- Speculative/batched MCTS (evaluating multiple leaves in one ONNX call) — belongs to phase 5 after phase 1-4 are stable
