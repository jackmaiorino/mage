# 2026-05-13 - Thesis-Clean MCTS Root Index Audit and Top-K Gate

## Question

Earlier MCTS probes were bad enough that scaling them on Zaratan was not justified. Before abandoning search-as-policy-improvement, audit whether the generic multi-ply MCTS implementation had a mechanical action-mapping bug.

## Thesis Boundary

Clean. This branch does not add card-name rules, Spy-specific rewards, Spy-specific terminal labels, or selective card gates. It only fixes generic search bookkeeping and adds a default-off generic root top-K gate.

## Code Changes

- `MultiPlyMCTS`: map live root candidates to cloned simulation request options by ability identity instead of assuming raw list indices match.
- `MCTSSimPlayer`: put `Pass` at slot 0 to match `ComputerPlayerRL`'s live priority-action convention.
- `ComputerPlayerRL`: add default-off `MCTS_ROOT_TOP_K` and `MCTS_ROOT_INCLUDE_PASS` so MCTS can rerank a small generic policy-prior subset while mapping visits back to the original candidate slots.

Why this mattered: the live action list is validated/deduped and inserts `Pass` at index 0; the simulation player re-enumerates from the clone. Multi-ply MCTS was backing values up to live root child indices while executing potentially different simulated options.

## Validation

Compile:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

### Root-Mapping Smoke

Run:

```text
20260513_multiply_rootmap_fix_affinity_cp7_smoke2
```

Settings:

```text
accepted checkpoint
CP7 Grixis Affinity only
MULTI_PLY_MCTS=1
MCTS_ITERATIONS=8
MCTS_DETERMINIZATIONS=1
MCTS_MAX_OUR_ACTIONS=1
MCTS_SKIP_TOP_PROB=1.01
MCTS_FINAL_SELECTION=value
```

Result:

```text
0/2 vs Affinity CP7
root_map=960/960 and 496/496 in the two chunks
mcts_activations=120 and 62
```

The mapper works. Forced MCTS-everywhere still fails the local gate.

### Uncertainty-Gated Smoke

Run:

```text
20260513_multiply_rootmap_fix_affinity_cp7_gated4
```

Settings:

```text
MCTS_SKIP_TOP_PROB=0.80
default final selection
no root top-K
```

Result:

```text
0/4 vs Affinity CP7
root_map all hits in MCTS-active chunks
mcts_activations: 59, 53, 0, 86
```

### Root Top-K Smoke

Run:

```text
20260513_multiply_rootmap_topk4_affinity_cp7_smoke4
```

Settings:

```text
MCTS_ROOT_TOP_K=4
MCTS_ROOT_INCLUDE_PASS=1
MCTS_SKIP_TOP_PROB=0.80
MCTS_ITERATIONS=8
```

Result:

```text
0/4 vs Affinity CP7
root_map all hits
mcts_activations: 89, 95, 85, 81
per-iter eval cost roughly 46-49ms
```

## Decision

Keep the root-index fix and top-K gate as search infrastructure, but do not scale this MCTS line on HPC yet.

The mechanical bug is real and fixed, but the local gate is still negative after the fix. The remaining failure is likely search target quality and leaf-value calibration under cloned post-action states, not only action-index bookkeeping or root branching.

Next search work should be diagnostic first: measure whether the value head ranks terminally winning sibling branches above terminally losing baseline branches in the baseline-losing-alternative corpus. If value ranking is poor there, MCTS scaling is compute waste until the leaf evaluator is repaired or replaced.

## Flat MCTS Top-K Follow-Up

Because the top-K gate is backend-independent, it was also tested on the flat value-net MCTS backend.

Runs:

```text
20260513_flatmcts_topk4_affinity_cp7_smoke4: 1/4
20260513_flatmcts_topk4_affinity_cp7_eval8:  0/8
combined: 1/12 vs Affinity CP7
```

Settings:

```text
MCTS_ROOT_TOP_K=4
MCTS_ROOT_INCLUDE_PASS=1
MCTS_ITERATIONS=8
MCTS_DETERMINIZATIONS=1
MCTS_ROLLOUT_DEPTH=0
MCTS_SKIP_TOP_PROB=0.80
MULTI_PLY_MCTS=0
```

Decision: flat top-K MCTS is also rejected for promotion. It is no better than the accepted Affinity baseline and likely worse on this local slice.

## Rally Top-K Follow-Up

Natural compact Rally logs showed slow setup and missing/late Spy conversion rather than the visible graveyard-hate failure seen in Affinity, so the repaired generic multi-ply backend was checked on Rally as well.

Run:

```text
20260513_mcts_topk4_rally_cp7_g4
```

Settings:

```text
accepted checkpoint
CP7 Mono Red Rally only
MULTI_PLY_MCTS=1
MCTS_ROOT_TOP_K=4
MCTS_ROOT_INCLUDE_PASS=1
MCTS_ITERATIONS=8
MCTS_DETERMINIZATIONS=1
MCTS_ROLLOUT_DEPTH=0
MCTS_SKIP_TOP_PROB=0.80
```

Result:

```text
1/4 vs Rally CP7
mcts_activations=187
duration=200.2s for 4 games
avg search wall time ~= 382-428ms/call
root_map all hits
```

The winrate matched the no-search compact Rally sample (`1/4`) while costing far more wall time. The winning game still looked like a long, late conversion line rather than reliable early combo execution. Reject Rally top-K MCTS as a local scaling candidate.

Two Zaratan login-node checks were attempted after this result to inspect queue/account state before considering any compute use. Both reached Duo and timed out before approval. No Slurm job was submitted and no allocation was used.
