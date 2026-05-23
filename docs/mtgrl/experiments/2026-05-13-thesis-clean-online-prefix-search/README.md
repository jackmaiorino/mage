# Thesis-Clean Online Prefix Search Gate

Date: 2026-05-13

## Question

Can online terminal-prefix search act as a generic search-as-policy-improvement operator against the hard Affinity pressure slice?

This is eval-only. It does not promote a checkpoint and does not spend HPC.

## Control-Flow Fix

The first online-prefix Affinity probe exposed a search plumbing bug:

```text
20260513_accepted_online_prefix_autopilot_affinity_cp7_eval4
Grixis Affinity CP7: 0/4
runtime: 320.7 sec
logs: repeated "RL priority() caught exception; forcing pass: deadline"
```

Cause: `TerminalPrefixSearch.SearchPlayer` threw its internal `SearchTerminated`
control-flow error from `genericChoose`, but inherited `ComputerPlayerRL.priority()`
swallowed all `Throwable` and forced pass. This made branch deadlines turn into
slow simulated pass loops.

Patch:

- `TerminalPrefixSearch.SearchPlayer` now overrides `priority()` and calls
  `priorityPlay(...)` directly after the branch deadline check.
- This is isolated to the simulated search player.
- Normal training/eval policy priority handling is unchanged.

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Historical Diagnostic Smoke

After the control-flow fix, the old tactic-ordered online prefix search became active:

```text
20260513_accepted_online_prefix_autopilot_affinity_cp7_eval2_fixsmoke
Grixis Affinity CP7: 1/2
runtime: 139.2 sec
```

The logs contained real `[ONLINE_PREFIX]` and `[ONLINE_PREFIX_AUTOPILOT]` events.
Example final stats in the winning chunk:

```text
calls=7 found=6 overrides=2 timeouts=3 autopilot_started=5 autopilot_applied=19 autopilot_misses=5
```

Read: the online hook now works mechanically. This is not a thesis-clean policy
result because the default `TerminalPrefixSearch` branch ordering still contains
explicit Spy-tactic/card-name preferences. Keep this mode diagnostic-only.

## Generic Branch-Order Patch

Added default-off generic branch ordering:

```text
RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER=1
```

When enabled:

- root and child expansion skip `preferredTacticPriority(...)`;
- simulated fallback choice skips `tacticChoice(...)`;
- remaining ordering is policy probability plus generic pass-last ordering;
- no card-name tactical ordering is used.

Registry:

```text
local-training/local_pbt/thesis_clean/20260513_thesis_clean_online_prefix_generic_affinity_eval_registry.json
```

Key settings:

```text
RL_ONLINE_PREFIX_SEARCH_ENABLE=1
RL_ONLINE_PREFIX_AUTOPILOT_ENABLE=1
RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER=1
RL_ONLINE_PREFIX_MODEL_GUIDED_FALLBACK=1
RL_ONLINE_PREFIX_SEARCH_MAX_NODES=31
RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH=8
RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS=3000
RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS=400
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
MCTS_TRAINING_ENABLE=0
ISMCTS_ENABLE=0
```

## Generic Probe Result

Run:

```text
20260513_accepted_online_prefix_generic_affinity_cp7_eval2
```

Result:

```text
Grixis Affinity CP7: 0/2
runtime: 134.7 sec
```

Activation logs:

```text
chunk 1: calls=6 found=0 timeouts=6 autopilot_started=0
chunk 2: calls=6 found=0 timeouts=6 autopilot_started=0
```

Every generic online-prefix call timed out at about 3 seconds without finding a
terminal win.

## Policy-Prior Child Expansion

Patch:

- Added `ComputerPlayerRL.getLastActionProbsSnapshot()` for subclasses.
- In `TerminalPrefixSearch.SearchPlayer`, model-guided fallback branch points now
  carry the model's latest action probabilities instead of uniform scores.

This is generic plumbing: no card text or Spy-specific state is inspected.

Run:

```text
20260513_accepted_online_prefix_generic_policyprior_affinity_cp7_eval2
```

Result:

```text
Grixis Affinity CP7: 1/2
runtime: 116.5 sec
```

However, the apparent win did not come from search:

```text
chunk 1: calls=6 found=0 timeouts=6 autopilot_started=0
chunk 2: calls=6 found=0 timeouts=6 autopilot_started=0
```

Read: capturing policy priors at deeper generic branch points is correct
plumbing, but it still does not make generic online prefix search find terminal
wins in this pressure slice at the tested budget.

## Verdict

Do not scale this on HPC.

The fixed online prefix hook is useful as a diagnostic, and the generic branch
mode is now available for future search probes. But the thesis-clean generic
configuration did not find wins in the hard Affinity slice, even after child
branch expansion used policy priors, and it is too slow for training as-is.

The next policy-improvement teacher needs either:

- a stronger generic rollout/search backend that can find terminal wins without
  card-name tactics, or
- a different training signal than prefix/autopilot distillation.

## Rally Probe

After the deep-mulligan line corpus showed `0/48` Rally line wins across six
Rally scenarios, the same thesis-clean generic online-prefix search was tested
against Mono Red Rally:

```text
20260513_accepted_online_prefix_generic_rally_cp7_eval2
```

Settings matched the generic Affinity probe: no Spy-specific candidate facts,
no heuristic rewards, no Spy terminal mode, generic branch order, model-guided
fallback, max 31 nodes, depth 8, 3 second search timeout, and compact game
logging enabled.

Result:

```text
Mono Red Rally CP7: 0/2
```

Search activity from compact game logs:

```text
chunk 1: calls=2 found=0 timeouts=2
chunk 2: calls=6 found=0 timeouts=6
```

Read: Rally matches the Affinity result. The generic online-prefix backend
fires, but it does not find terminal wins at this budget, so it is not a
near-term teacher for the weak hard matchups.

## Rally Larger-Budget Probe

To check whether the small Rally result was only a search-budget issue, one
local eval game was rerun with a larger generic budget:

```text
20260513_accepted_online_prefix_generic_rally_bigbudget_cp7_eval1
```

Settings:

- `RL_ONLINE_PREFIX_SEARCH_MAX_NODES=127`
- `RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH=10`
- `RL_ONLINE_PREFIX_SEARCH_TOP_K=4`
- `RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS=4`
- `RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS=15000`
- `RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS=1500`
- generic branch order and model-guided fallback remained enabled

Result:

```text
Mono Red Rally CP7: 0/1
```

Search activity:

```text
calls=4 found=1 overrides=1 timeouts=4 autopilot_started=1 autopilot_applied=5 autopilot_misses=1
```

Read: this is more nuanced than the small-budget failure. Generic search can
find at least one terminal-winning prefix with a larger budget, but eval-time
autopilot is brittle: the live game diverged after five applied prefix actions,
then missed on the expected `Land Grant` step and still lost. This is not a
promotion result, but it reopens search as an offline teacher if the collector
can keep only robust, genuinely missed root decisions instead of relying on
long live autopilot execution.

## Rally Receding-Horizon Probe

The same larger budget was tested with autopilot disabled, so search could only
choose the current root action and would need to re-search after the live state
changed:

```text
20260513_accepted_online_prefix_generic_rally_receding_cp7_eval1
```

Result:

```text
Mono Red Rally CP7: 0/1
calls=8 found=0 timeouts=8
```

Read: disabling autopilot removes the brittle long-suffix execution problem,
but the root-only receding search did not find any terminal wins in the sampled
game. The positive signal remains confined to offline larger-prefix collection,
not live eval.
