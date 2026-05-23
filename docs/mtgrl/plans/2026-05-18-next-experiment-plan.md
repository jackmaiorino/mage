# MTGRL Next Experiment Plan - 2026-05-18

Window reviewed: 2026-04-18 through 2026-05-18.

Primary evidence:

- `docs/mtgrl/experiments/2026-04-25-terminal-meta-selfplay-cp7-sweep/README.md`
- `docs/mtgrl/experiments/2026-04-26-spy-terminal-winning-line-search/README.md`
- `docs/mtgrl/experiments/2026-04-29-spy-terminal-prefix-curriculum/README.md`
- `docs/mtgrl/experiments/2026-05-06-spy-fastline-and-mulligan-cf/README.md`
- `docs/mtgrl/SESSION_HANDOFF_2026-05-08.md`
- `docs/mtgrl/experiments/2026-05-10-thesis-clean-hardmatchup-selfplay/README.md`
- `docs/mtgrl/experiments/2026-05-12-thesis-clean-negative-surface-review/README.md`
- `docs/mtgrl/experiments/2026-05-13-24h-experiment-summary/README.md`
- `docs/mtgrl/experiments/2026-05-14-24h-experiment-summary/README.md`
- `docs/mtgrl/experiments/2026-05-14-thesis-clean-accepted-affinity-log-diagnostic/README.md`

## Current Baseline

The best accepted checkpoint is the 2026-05-10 thesis-clean Affinity-pressure
continuation of the four-profile self-play line.

| Skill | Overall | Spy | Wildfire | Rally | Affinity |
| --- | ---: | ---: | ---: | ---: | ---: |
| CP1 | 133/256 = 51.95% | 49/64 | 31/64 | 22/64 | 31/64 |
| CP3 | 132/256 = 51.56% | 54/64 | 29/64 | 24/64 | 25/64 |
| CP7 | 108/242 = 44.63% | 43/56 | 36/60 | 16/64 | 13/62 |

Immediate hard target: CP7 Grixis Affinity. The fresh accepted compact log
sample on 2026-05-14 was 3/16 = 18.75%, close to the accepted long-run Affinity
reference of 13/62 = 20.97%.

HPC status: keep Zaratan parked. The last two reviewed 24-hour windows submitted
no Slurm training job and spent no kSU; one A100 smoke job was canceled before
allocation.

## What We Learned

### 1. Pure terminal self-play became useful, but only after curriculum pressure.

The early 2026-04-25 terminal-only four-profile run failed CP7 badly:
49/318 overall and Spy only 5/79. The first generalist attempt was worse at the
CP7 gate: 0/79.

The two-GPU pure self-play path then made self-play cheap enough to revisit. By
2026-05-01 to 2026-05-02, Spy was no longer zero but remained fragile into Rally
and Affinity. The real breakthrough came on 2026-05-10: generic deck-level
hard-matchup pressure, still terminal-only and thesis-clean, raised the May 2
CP7 Spy baseline from 28.40% to the accepted 44.63%.

Conclusion: terminal-only self-play is viable when the opponent curriculum is
externally steered by measured matchup weakness. Blind continuation and pure
deck-balanced continuation are not enough.

### 2. Search can find winning lines, but our transfer operators are weak.

Terminal winning-prefix search eventually found real Spy lines and even produced
the first direct Rally wins in a small probe. Later shallow terminal prefix
search became a strong teacher: top-k search reached up to 111/128 searched
starts, and nodes7 became a good cost-quality point.

But distillation failed repeatedly. Accepted already scored many searched prefix
corpora at very high one-step accuracy, often above 93% and sometimes 100%.
Bulk prefix KL, DAgger, first-deviation repair, trajectory RL, and policy-miss
imports improved saved-start replay or tensor fit without improving fresh CP7
games.

Conclusion: the gap is sequential compounding and distribution drift, not local
one-step ranking on solved prefix tensors.

### 3. The value head improved, but value does not currently drive action timing.

Separate-critic value RL improved logged value ranking: all-decision AUC moved
to 0.586, critical combo AUC to 0.761, and per-game mean AUC to 0.731 in one
logged sample. Later accepted Affinity diagnostics also showed useful midgame
value separation.

However, current-state value gates, candidate-Q blending, branch-return Q,
critic-only calibration, branch-pair value rank, and repaired MCTS all failed
local gates.

Conclusion: value signal exists but is not yet a reliable policy-improvement
mechanism. More value-only patches are low value unless tied to a better teacher.

### 4. Small feature and one-step target surfaces are exhausted.

The local negative surface is now broad:

- Candidate-Q and signed branch-return Q.
- Prefix KL, trajectory KL, and one-step policy-miss imports.
- Root mulligan repair and deep-mull quick fixes.
- AWR/AWAC selected-action terminal-advantage targets.
- Representation-only probes: zone/library counts, public board features,
  semantic effect flags, action-class flags, source-zone flags, and belief labels.
- Current MCTS/search targets, including repaired root mapping, top-k gating,
  flat value MCTS, sparse train-time MCTS, random rollout ISMCTS, and online
  prefix root/trace hooks.
- Checkpoint soups, policy ensembles, checkpoint selection, gamma/GAE tweaks,
  and short hard-opponent continuations.

Conclusion: stop spending local cycles on another small coefficient/blend/feature
variant unless it changes data generation or preserves trajectory context.

### 5. The live failure is Affinity pressure in fresh states.

The 2026-05-14 accepted Affinity compact logs show a mixed failure:

- 9/16 games never cast Spy.
- Spy-cast games won only 1/7.
- One loss hard-cast Dread Return from hand in a low-graveyard state.
- Opponent battlefield count >= 10 was 0/11.
- The value head was not uniformly blind in this sample.

Conclusion: the next mechanism must reach fresh accepted-policy failure states
under Affinity pressure and preserve downstream context. It should not be another
offline label set made mostly from states the accepted policy already handles.

## Stop List

Do not start another run whose only change is:

- More one-step BC/KL on searched prefix tensors.
- More candidate-Q or branch-return Q blend settings.
- More root mulligan or hard timing masks.
- More representation-only feature toggles.
- More repaired MCTS rollouts with the current value target.
- More deck-weight-only continuation from the accepted checkpoint.
- More checkpoint soups or policy ensembles using already rejected branches.

These are useful as infrastructure or diagnostics, not as next scaling branches.

## Next Experiment Ladder

### Experiment 0: Freeze The Gate

Goal: prevent another false positive from noisy small samples.

Use the accepted 2026-05-10 Affinity-pressure checkpoint as immutable reference.
Before promoting any branch, require:

1. CP7 Grixis Affinity local screen: at least 5/16, then at least 10/32 before
   full four-opponent CP7.
2. Pressure pair screen: Rally plus Affinity must not fall below the fresh
   accepted pressure sample, roughly 7/16 combined.
3. Expanded four-opponent CP7 must be at or above accepted aggregate and must
   not regress Affinity below 13/62.
4. Final promotion uses `scripts/compare_thesis_clean_eval.py` against accepted
   CP1/CP3/CP7 sweeps.

HPC remains blocked until these local gates pass.

### Experiment 1: Affinity Failure-Context Corpus

Hypothesis: accepted-policy losses contain repeatable, unsolved correction points
under Affinity pressure, but the current collectors either miss them or strip too
much downstream context.

Run a larger accepted-policy CP7 Affinity compact-log sample, preferably 64
games if local time allows. Export:

- compact state summaries,
- selected actions and candidate probabilities,
- value scores,
- source zones,
- public-board pressure aggregates,
- baseline-losing-alternative points,
- whether a forced sibling can reach terminal win under a generic offline search.

No training in this experiment.

Pass criteria before training:

- At least 50 accepted-policy failure states where baseline loses and a sibling
  or short corrected prefix reaches a terminal win.
- Accepted top-1 on those corrective first actions below 50%; ideally much lower.
- Examples cover multiple Affinity pressure clusters, not only a single
  hand-crafted Spy timing pattern.

### Experiment 2: Counterfactual Recovery Trajectory Teacher

Hypothesis: one-step correction labels fail because they do not teach how the
game should continue after the correction. A teacher must export the corrective
decision plus the downstream recovery trajectory.

Data generation:

1. Replay accepted games into a baseline-losing-alternative state.
2. Force the winning sibling or corrected short prefix.
3. Continue with generic terminal prefix search, not Spy-name tactics.
4. Export the full corrected suffix only when the forced correction reaches a
   terminal win.
5. Mark which decisions are true policy misses versus downstream context labels.

Training:

- Start from the accepted checkpoint clone.
- Use a frozen-reference KL anchor.
- Train primarily on the first corrective action and only downstream decisions
  that remain inside the successful forced trajectory.
- Keep terminal returns only; no step rewards or card-specific heuristic rewards.
- Avoid training arbitrary "best" alternatives when all branches lose.

Local gates:

- Fit should improve corrective-action top-1 on the unsolved corpus without
  materially moving already-solved prefix corpora.
- Same-start corrected replay should improve by at least 15 percentage points.
- First CP7 Affinity gate must hit at least 5/16 before any broader eval.

### Experiment 3: Replay-Integrated Terminal RL

Hypothesis: direct offline import is too brittle, but interleaving corrected
failure trajectories with terminal-only pressure self-play can reduce drift and
forgetting.

Run only if Experiment 2 passes its small Affinity gate.

Design:

- Accepted clone, reset counters.
- Opponent pool: Affinity-heavy but not single-opponent; keep Jund/Rally present
  to control forgetting.
- Alternate normal terminal-only episodes with a small replay buffer of recovery
  trajectories from Experiment 2.
- Keep reference KL on, low learning rate, and immutable accepted eval snapshot.

Promotion gate:

- CP7 Affinity >= 10/32.
- CP7 Rally pressure not worse than accepted.
- Four-opponent CP7 aggregate >= accepted.
- `compare_thesis_clean_eval.py` passes CP1/CP3/CP7.

### Experiment 4: Belief/Determinization Calibration Branch

Run in parallel only as a diagnostic branch, not as a training-scale job.

Hypothesis: Affinity pressure may require hidden-zone and opponent-plan belief,
but the current belief-head smoke was not calibrated enough to drive search.

First build calibration, not gameplay:

- Predict hidden-zone/card-presence summaries from public history and deck priors.
- Evaluate on accepted compact logs across Affinity, Rally, Wildfire, and mirror.
- Require calibration for non-Spy opponents before using the belief head in
  ISMCTS or prefix search.

Pass criteria:

- Non-Spy archetype/posterior calibration no longer overpredicts Spy in early
  non-Spy games.
- Hidden-zone summaries relevant to land/library/graveyard state beat simple
  deck-prior baselines.
- Only then wire belief into the offline teacher, not online random rollouts.

### Experiment 5: HPC Scale

Only after a local branch passes Experiment 3 or a belief-calibrated teacher
passes a comparable local gate:

- Submit one Zaratan smoke using the canonical remote checkout path.
- Use GPU Slurm only for scaled training after the local mechanism is validated.
- Do not spend kSU on speculative feature/blend/coefficient sweeps.

## Near-Term Recommendation

Start with Experiment 1. The next high-value artifact is not another model; it is
an Affinity failure-context corpus that proves we can repeatedly find unsolved
accepted-policy failure states and terminal-winning corrections in the same live
distribution where accepted fails.

If that corpus is dense, build the counterfactual recovery trajectory teacher.
If it is sparse, pivot to belief/determinization calibration before training.
