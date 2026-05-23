# Thesis-Clean Negative Surface Review

Date: 2026-05-12

## Scope

This note consolidates the local experiments after the accepted 2026-05-10 Affinity-pressure checkpoint. The goal is to avoid repeating surfaces that now have multiple negative local gates.

Accepted reference:

```text
CP1 133/256 = 51.95%
CP3 132/256 = 51.56%
CP7 108/242 = 44.63%
```

## Rejected Surfaces

### Candidate-Q

Rejected variants:

```text
selected-action Q-only + eval blend: 22/63 CP7
balanced branch-return Q-only + eval blend: 17/64 CP7
shared candidate-Q auxiliary, no eval blend: 24/64 CP7
signed branch-return policy-head preference loss: 0/64 CP7
```

Decision: stop candidate-Q and one-step branch-return preference cycling. The
repeated pattern was mirror competence with Rally/Affinity collapse; direct
signed branch-return policy-head training was worse, collapsing all matchups.

### Prefix Distillation

Rejected variants:

```text
direct generic prefix KL
hard-matchup-balanced prefix KL
head-only prefix KL
policy-path-only prefix KL
current-policy anchored prefix KL
frozen-reference anchored prefix KL
low-mix / low-coef frozen-reference KL
library-count + frozen-reference KL
policy-miss-only prefix KL: 22/64 CP7
prefix tensors as one-step terminal-return PPO episodes: 5/64 CP7
```

Decision: stop positive-only offline prefix-data training. Prefix search can find lines in some settings, but neither direct target matching nor terminal-return training transfers to hard-pressure games.

### Root Mulligan Prefix

Rejected variant:

```text
root mulligan prefix search, head-only, CP7 Rally/Affinity labels: 20/63 CP7
```

Decision: root mulligan alone is not the hard-pressure bottleneck. The post-mulligan action policy still fails to convert enough states.

### Terminal Continuation

Rejected variants:

```text
single-profile Spy terminal-only continuation, reset counters: 25/64 CP7
balanced four-profile terminal-only continuation: 23/64 CP7
```

Decision: stop short continuation scheduling as a quick-win surface. Multi-profile balance did not fix the pressure-matchup regression; it only made the negative result more expensive to collect.

### Terminal Advantage Targets

Rejected variant:

```text
online AWR/AWAC-style selected-action targets, 128ep hardmix: 22/54 partial CP7
```

Partial matchup result:

```text
Spy mirror      11/16 = 68.75%
Jund Wildfire    8/15 = 53.33%
Mono Red Rally   1/16 =  6.25%
Grixis Affinity  2/7  = 28.57%
```

Decision: reject this exact AWR selected-action formulation. It is thesis-clean, but terminal-advantage weighting collapsed Rally and regressed mirror/Jund. Do not scale this run on HPC.

### Representation-Only

Rejected variants:

```text
generic history H4 / scaled history
generic library-count partial terminal RL
library-count + prefix KL
public archetype posterior features: 22/63 CP7
true opponent-archetype belief labels, 300ep: 29/64 CP7
true opponent-archetype belief labels, lower coef, 300ep: 23/64 CP7
true opponent-archetype belief labels, belief-head-only, 300ep: non-Spy calibration still ~46-51%
```

Decision: representation-only changes have not cleared the local gate. Generic library composition and public archetype posterior features remain thesis-clean, but they are not sufficient without a stronger learning target.

True hidden-state belief labels are useful infrastructure, but the first 300ep branch only matched the accepted CP7 baseline within noise and did not produce a calibrated non-Spy posterior. Lowering the coefficient improved calibration slightly but collapsed gameplay. Freezing policy/value and training only `belief_head.*` avoided policy collapse, but still overpredicted `SpyCombo` in early non-Spy games. Do not use the current belief head for neural-belief ISMCTS without a stronger calibration gate, and do not continue simple true-label coefficient sweeps.

### Current MCTS/Search Targets

Rejected variants:

```text
flat value-net MCTS
sparse train-time MCTS, forced
sparse train-time MCTS, no-force
sparse MCTS hard-skill mix
eval-time multi-ply MCTS value/visit selectors
eval-time flat policy-rollout MCTS
terminal random-rollout ISMCTS
online prefix root/trace training
flat value-MCTS with 4-deck deck-list determinization prior: 0/4 CP7, stopped early
terminal rollout-root ISMCTS with 4-deck deck-list determinization prior: 0/3 CP7, stopped early
```

Decision: do not spend HPC scaling the current MCTS target family. The issue is target quality, not just rollout count or archetype-prior mismatch.

## Current Conclusion

The short local variant surface is exhausted. More small tweaks to Q heads, prefix KL, root mulligans, or search coefficients are unlikely to produce useful information.

The remaining thesis-clean path needs a deeper mechanism:

```text
1. A better generic policy-improvement teacher that produces trajectory-level targets without live priority() search.
2. Card-level belief / determinization that predicts hidden zones from public history and deck priors.
3. Checkpoint-selected longer continuation only after a local mechanism clears CP1/CP7 gates.
```

HPC stays parked under the 50 kSU budget until one of those mechanisms clears reduced local gates.

2026-05-13 addendum: the first one-step signed branch-return policy import used
a mismatched `256/4` action-counterfactual wrapper default against the canonical
`128/2` checkpoint, so its 0/64 CP7 result is invalid as policy-quality evidence.
The wrapper default was fixed to `128/2`, then a low-dose anchored rerun reached
only 6/16 CP7, with Jund at 0/4. A corrected original-strength rerun reached
4/16 CP7, with Jund and Affinity both at 0/4. This rejects the current isolated
branch-preference dataset and narrows item 1: the teacher likely needs coherent
multi-step/trajectory targets, not more isolated branch preferences from the
current dataset.

2026-05-13 trajectory addendum: fixing the same `128/2` wrapper default in
`run_spy_line_search.ps1` reopened terminal-prefix trajectory collection. The
corrected collector produced 32 CP7 winning trajectories locally, so the old
"zero trajectories" result is invalid as a search-feasibility claim. However,
trajectory-RL and low-dose KL distillation from the corrected corpora still
failed expanded CP7 screens (`27/64`, `21/64`, `21/64`) and an Affinity-only
KL pressure branch reached only `1/16` into Affinity. The search teacher can
find lines; the current distillation/update mechanism does not transfer them
robustly. Tensor/replay diagnostics explain why: accepted already scores the
corrected CP7 prefix tensors at `119/127` top-1 and the depth-10 Affinity
prefix tensors at `121/124` top-1. The teacher is finding lines mostly through
states the policy already ranks correctly, so more KL on the same tensors is
low value.

2026-05-13 online-prefix addendum: the eval-only online prefix hook had a
control-flow bug where simulated `SearchTerminated` deadlines were swallowed by
the normal RL `priority()` safety catch and turned into slow forced-pass loops.
`TerminalPrefixSearch.SearchPlayer` now overrides `priority()` so branch
deadlines return to the search caller. After the fix, the old tactic-ordered
diagnostic hook became active (`1/2` CP7 Affinity, with real `[ONLINE_PREFIX]`
and autopilot logs), but that mode contains explicit Spy/card-name tactic
ordering and is diagnostic-only. A new thesis-clean generic branch-order mode
(`RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER=1`) removed those preferences; it reached
`0/2` CP7 Affinity and every search call timed out without finding a terminal
win. Capturing policy probabilities for deeper model-guided generic branch
points was then added and retested (`1/2` CP7 Affinity), but all online-prefix
calls still logged `found=false`; the win was normal policy variance rather than
search improvement. Do not scale online prefix search on HPC unless the generic
backend changes substantially.

2026-05-13 teacher-quality addendum: generic prefix sibling-contrast was tested
against CP7 Affinity with no tactic autopilot or card filters. It collected 47
selected examples, but the accepted checkpoint scored them at `47/47` top-1, so
it is another locally solved searched-prefix corpus. A new default-off generic
baseline-losing-alternative filter was then added to `ActionCounterfactualTrainer`.
That filter keeps only decisions where the accepted baseline action loses and a
sibling branch wins. It found 22 CP7 Affinity labels scored `0/22` top-1 by the
accepted checkpoint, which is the first useful unsolved offline signal in this
round. However, three direct imports from that 22-example corpus failed the
Affinity CP7 gate (`2/16`, `1/16`, `2/16`). Keep the filter as infrastructure,
but do not scale this sparse one-step Affinity corpus on HPC.

The broader four-opponent CP7 collection produced 66 labels, also scored `0/66`
top-1 by accepted. A policy-path import fit the corpus (`51/66` top-1,
`59/66` target-set top-1) but failed the reduced CP7 gate at `5/16`, with Rally
at `0/4`. This rejects isolated one-step imports from the current filter. The
filter remains useful for identifying genuine policy mistakes; the training
operator needs coherent trajectory replay or an online loop that forces
corrected decisions in context and then trains on the downstream trajectory.
A replay/DAgger export from those corrections produced 115 examples and fitted
cleanly (`98/115` top-1), but its reduced CP7 gate was worse at `3/16`. This
closes the current offline counterfactual teacher family as a scaling candidate.

2026-05-13 MCTS audit addendum: multi-ply MCTS had a real root action-index bug.
The live policy list is validated/deduped and puts `Pass` at slot 0, while the
simulation player re-enumerated clone actions with `Pass` appended. Values could
therefore be backed up to one live root action while the clone executed another.
The fix maps root children by generic ability identity and aligns simulated
`Pass` ordering. The mapper was verified in local CP7 Affinity smokes
(`root_map=960/960`, `496/496`, and all hits in later chunks). However, MCTS
still failed the local gate after the repair: forced value-selected multi-ply was
`0/2`, uncertainty-gated multi-ply was `0/4`, and the new default-off generic
`MCTS_ROOT_TOP_K=4` gate was also `0/4`. Keep the mapping fix and top-K gate as
infrastructure, but do not spend HPC on this MCTS line until value-head branch
ranking or rollout target quality improves.

Flat value-net MCTS with the same root top-K gate was also expanded locally:
`1/4` followed by `0/8` into CP7 Affinity (`1/12` combined). That is baseline
or worse for the accepted checkpoint's weakest matchup, so root top-K does not
rescue the flat backend either.

2026-05-13 branch-value addendum: a new thesis-clean branch-value probe scored
baseline-losing-alternative branches with the accepted value head. In the
16-scenario CP7 smoke, there were 4 comparable win/loss decision pairs and the
value head preferred the winning branch only once. Winning rows averaged
`0.187849`; losing rows averaged `0.222264`. This supports the MCTS conclusion:
the current blocker is value/target quality, not just search branching or root
indexing. A first branch-trajectory value-calibration pass was then implemented
and run on a cloned profile with policy loss off and MC value loss on. The
16-game CP7 smoke was neutral (`7/16`, close to the accepted CP7 anchor), so the
plumbing is viable but not yet a promotion candidate. Scale branch-trajectory
value calibration before revisiting MCTS scaling.

Scaling that same recipe to 24 collection scenarios and 2056 value-only
training steps regressed the CP7 smoke to `6/16`, with Affinity at `0/4`.
Reject blunt value-only fine-tuning on all forced branch trajectories. If this
line continues, make it paired/decision-local rather than training every noisy
downstream state from forced branches.

The decision-local first-post-target variant was also tried: 185 one-step
post-branch states, 740 value-only train passes, then a 16-game CP7 smoke.
It reached `7/16` overall but again `0/4` into Affinity. This rejects the
current value-only branch-state calibration family. Keep the probes as
diagnostics, but pivot away from value-only local imports.

2026-05-13 public-board addendum: compact Affinity log review showed noisy value
scores under visible board pressure, so a default-off thesis-clean representation
feature was added for public battlefield aggregates: permanent count, untapped
and tapped creatures, attacking/blocking creatures, token count, and aggregate
creature power. Eval-only perturbation on accepted was stable (`8/16` CP7), but
a 128-episode terminal-only Spy clone with `RL_PUBLIC_BOARD_FEATURES_ENABLE=1`
reached only `7/16` CP7, with Affinity still `1/4`. Keep the feature default-off
as infrastructure, but reject public-board representation-only training as an
HPC candidate.

2026-05-13 branch-pair ranking addendum: a thesis-clean paired branch trajectory
export and value-ranking loss were added. The exporter emits two-record episodes:
a post-branch state from a terminally winning sibling followed by a post-branch
state from a terminally losing sibling. An 8-pair smoke import was neutral
(`8/16` CP7), but a larger 24-scenario collection produced 28 pairs and the
same isolated value-rank import regressed to `5/16` CP7, with mirror `1/4` and
Rally `0/4`. The paired signal is real, but direct shared-model ranking updates
are not a scaling candidate. Do not spend HPC on branch-pair value ranking as
implemented; if revisited, isolate critic/value parameters or use the pair
corpus as a diagnostic for value-gated search rather than direct policy-facing
fine-tuning.

The containment follow-up froze all non-critic/value parameters during the same
28-pair import. That avoided the severe mirror collapse but still reached only
`7/16` CP7, with Affinity `1/4`. Pair-rank critic calibration also failed to
rescue repaired top-K multi-ply MCTS (`0/2` into CP7 Affinity with 77 MCTS
activations). Close branch-pair ranking as a scaling candidate.

2026-05-13 reference-anchored continuation addendum: a clean clone of accepted
was continued for 128 terminal-only episodes on the hard-opponent deck pool with
a frozen-reference policy KL anchor (`REFERENCE_POLICY_KL_COEF=0.50`) and
conservative PPO/lower learning rates. The run completed cleanly, but the
reduced CP7 gate was `5/16`, including Affinity `0/4` and Jund `1/4`. This
rejects hard-opponent reference-anchored continuation as an HPC candidate.

2026-05-13 deep-mulligan line addendum: accepted logs exposed a real Rally
failure where the policy mulliganed down to a zero-card start. The existing
London line trainer was extended from max 3 to max 7 mulligans and run in a
thesis-clean binary keep-vs-deeper-mull mode with terminal labels only. A tiny
4-scenario smoke produced 6 binary labels and did not damage first-hand probe
behavior, but the reduced CP7 gate was only `7/15`, with Jund `0/3` and one
Jund chunk timing out at `0/0`. Reject isolated deep-mull head distillation as
a scaling candidate. The failure is real, but future work should first collect
and inspect a larger depth-balanced line corpus before training from it.

The larger collect-only line corpus (`16` scenarios, `22` selected labels)
confirmed that the mulligan signal is sparse and depth-dependent: all selected
labels after four mulligans were "keep", while shallower prompts were mixed.
More importantly, Rally produced `0/48` line wins across six sampled Rally
scenarios, so mulligan depth alone is not a plausible Rally repair path.
Generic online-prefix search was then tested against Rally and also failed at
the small budget: `0/2`, with 8 search calls, 0 found wins, and 8 timeouts.
A larger one-game Rally budget did find one terminal-winning prefix, but the
live autopilot diverged after five applied actions and the game still lost.
This closes the deep-mull quick fix and eval-time generic autopilot as local
scaling candidates, while leaving larger-budget generic prefix collection as a
possible offline-teacher diagnostic.

That offline diagnostic was run next: 8 Rally scenarios, generic branch order,
depth 10, 127 search nodes, no tactics, and collect-only. It found 8 winning
trajectories and 60 selected examples, but accepted scored the serialized corpus
at `60/60` top-1. A larger-budget receding-horizon online probe with autopilot
disabled also lost (`0/1`, 8 calls, 0 found, 8 timeouts). This rejects the
Rally big-prefix corpus as another solved-prefix teacher.

While closing that branch, a collection-filter bug was fixed:
`-PolicyMissOnly` now filters collect/export selection, not just import/training.
The fixed Rally rerun found 3 winning prefixes but exported `0` selected
examples, confirming the corpus is solved by accepted and preventing accidental
training from it.

2026-05-13 Rally MCTS addendum: the repaired root-mapped generic multi-ply MCTS
was also checked against CP7 Mono Red Rally, since compact no-search Rally logs
showed slow development and late/missing Spy conversion rather than the Affinity
graveyard-hate pattern. The top-K run (`MCTS_ROOT_TOP_K=4`, 8 iterations, 1
determinization, rollout depth 0) finished `1/4` with 187 MCTS activations and
about 200 seconds of wall time. This matches the no-search Rally log sample
(`1/4`) at much higher cost. Combined with the Affinity MCTS failures, this
closes repaired value-net MCTS as a local scaling candidate until the evaluator
or teacher target changes.

Zaratan status: two login-node-only checks were attempted to inspect queue and
account state before considering any compute. Both reached Duo and timed out
before approval. No job was submitted and no kSU allocation was used.

2026-05-13 semantic-flags addendum: compact Affinity logs identified a real
generic representation gap around visible graveyard-exile and damage-all effects.
`RL_EXTENDED_EFFECT_FLAGS_ENABLE=1` fixed those effect categories, but eval-only
activation was destructive (`1/16` into CP7 Affinity). A 500-episode adaptation
clone reached `5/15` into CP7 Affinity but failed reduced CP1 (`30/64`), while
a reference-anchored 128-episode clone passed reduced CP1 (`34/64`) but failed
CP7 Affinity (`3/16`). Checkpoint interpolation was then tested between the
accepted model and the unanchored semantic-flags checkpoint. The best soup
(`75%` semantic) passed reduced CP1 (`33/64`) and narrowly improved CP7 Affinity
(`4/16`), but the combined reduced CP7 screen was still below accepted
(`27/64`) because Jund regressed to `7/16`. Keep the semantic flag extractor and
the DB-hardlink eval-runner fix as infrastructure, but do not scale this branch
or submit it to HPC.

2026-05-13 branch-trajectory soup addendum: checkpoint interpolation was also
tested between accepted and the thesis-clean branch-trajectory policy import,
which had previously improved Rally/mirror but regressed Affinity. The 25% soup
screened at `3/8` CP7 Rally and `1/8` CP7 Affinity; the 50% soup screened at
`1/8` for both Rally and Affinity. This rejects checkpoint interpolation as a
rescue for the branch-trajectory policy line.

2026-05-13 policy-ensemble addendum: eval-only probability ensembling was added
to test whether rejected thesis-clean branches contained complementary action
signal that tensor soups destroyed. Accepted plus the semantic-effect-flags
companion at `0.75/0.25` scored Rally `2/8` and Affinity `1/8`. Accepted plus
the branch-trajectory companion at `0.75/0.25` improved Rally to `7/16`, but
collapsed Affinity to `1/16`. This rejects simple checkpoint combination as a
high-EV local surface unless a future branch is already broadly strong before
combination.

2026-05-13 policy-miss trajectory addendum: trajectory import now honors the
generic `-PolicyMissOnly` and target-margin filters, matching flat imports.
Reimporting the baseline-losing branch-trajectory corpus with that filter gave
a promising tiny gate (`10/16` CP7), but the pressure expansion repeated the
same split: Rally `7/16`, Affinity `1/16`. This closes the current
branch-trajectory family until a new target generator produces Affinity-robust
examples.

The Affinity-only policy-miss trajectory control also failed: using the
Affinity-only corpus kept 25 episodes / 515 steps and reached only `1/16`
against CP7 Grixis Affinity. The runner bug found during this control was fixed:
parallel eval DB setup now uses real per-job copies by default instead of H2
hardlinks, with hardlinking available only through `CP7_EVAL_DB_HARDLINK=1`.

2026-05-14 critic-only terminal addendum: a fresh accepted clone trained for
256 real terminal episodes with policy parameters frozen
(`VALUE_PAIR_RANK_CRITIC_ONLY=1`, policy/entropy loss off, value loss on). The
resulting checkpoint did not rescue repaired generic MCTS: CP7 Grixis Affinity
top-K multi-ply MCTS went `0/4` with 307 activations. This rejects small
real-game critic-only calibration as an MCTS scaling prerequisite.

2026-05-14 generic action-class feature addendum: the counterfactual label
diagnosis showed that recent hard-matchup target corpora often select generic
low-level mana/land actions (`8/16` best labels in the Affinity branch-trajectory
set; `36/66` in the broad baseline-losing-alt set). A default-off thesis-clean
feature flag, `RL_GENERIC_ACTION_CLASS_FEATURES_ENABLE`, now marks ability
candidates as mana ability, land play, spell, other activated ability, and
uses-stack. A 128-episode reference-anchored full-policy adaptation from
accepted reached only `2/16` on the Rally/Affinity CP7 pressure gate
(`0/8` Rally, `2/8` Affinity). Reject the full-update action-class branch and
do not spend HPC on it. The policy-path-only containment follow-up also reached
only `2/16` on the same pressure gate (`0/8` Rally, `2/8` Affinity), so close
generic action-class feature adaptation as a local scaling candidate.

2026-05-14 accepted snapshot-selection addendum: spaced historical snapshots
from the accepted profile were cloned and screened against the hard CP7
pressure pair. The latest clone and snapshots at steps 60000, 97000, and 132000
all reached `0/4`; snapshots at 90000, 117000, and 122000 reached only `1/4`.
No tested checkpoint won an Affinity game. This rejects existing accepted
snapshot selection as a quick promotion path and avoids spending HPC on a broad
historical checkpoint sweep.

2026-05-14 corrected train-time MCTS addendum: because the earlier sparse
train-time MCTS auxiliary runs predated the root action-mapping fix, a fresh
post-fix branch was trained from accepted with no forced MCTS behavior,
generic root top-k (`MCTS_ROOT_TOP_K=4`), sparse confidence/sample gating, and
hard-skill hybrid opponents. The run completed cleanly and all root mappings
hit (`1824/1824`), but the weighted Rally/Affinity CP7 pressure screen was only
`12/68` (`4/24` Rally, `8/44` Affinity). This closes the corrected no-force
top-k MCTS auxiliary as an HPC candidate. The search plumbing is now valid; the
current value-guided MCTS target is still not a useful teacher.

2026-05-14 branch-pair return-contrastive addendum: the existing generic
branch-pair corpus was reimported with policy/value/MCTS losses disabled, a
low-dose terminal return-contrastive policy loss, and a frozen-reference policy
anchor. The import used 17 paired episodes / 34 steps over 8 epochs. The CP7
smoke reached only `6/16`, including `0/4` into Grixis Affinity. This rejects
the simple policy-side contrastive import recipe on the current paired corpus.
Do not scale it without a larger Affinity-robust pair corpus or a different
sequence-level preference objective.

2026-05-14 branch-pair policy-rank addendum: a default-off generic
`POLICY_PAIR_RANK_LOSS_COEF` was added to rank adjacent paired branch records by
terminal outcome using selected-action log probability. Reimporting the same
17 paired episodes / 34 steps with a frozen-reference policy anchor reached
only `6/16` CP7 (`0/4` Rally, `1/4` Affinity). This closes the current
paired-branch import family: value-rank, critic-only value-rank,
return-contrastive, and policy-pair rank all failed local gates on this corpus.

2026-05-14 pressure pair-rank addendum: pair-mode export now honors
`--branch-trajectory-require-training-example`, enabling pairs to be restricted
to generic baseline-losing-alternative decisions. A Rally/Affinity pressure
collection found 19 such labels but only 11 paired first-post-target episodes
from 64 scenarios. Importing those 11 pairs with the policy-pair rank loss
reached `4/16` on the CP7 pressure gate (`2/8` Rally, `2/8` Affinity). This
rejects scaling the current pressure pair-rank branch; the local generator is
too sparse for this import strategy.

2026-05-14 Affinity pair-rank addendum: the planned Affinity-only follow-up
expanded the same generator to 128 CP7 Grixis Affinity scenarios. It found 27
candidate baseline-losing-alternative examples and exported 21 paired
first-post-target episodes / 42 steps. Importing those pairs with the generic
policy-pair rank loss and a frozen-reference KL anchor reached only `3/16`
against CP7 Grixis Affinity, below both the accepted long-run Affinity reference
(`13/62 = 20.97%`) and the fresh accepted pressure sample (`2/8 = 25.0%`). This
closes the current paired-branch import family. The filter still diagnoses real
accepted-policy mistakes, but sparse pair fitting is not transferring to fresh
hard-matchup games.

2026-05-14 action pair-rank addendum: a weaker same-state branch-return
preference loss was added behind `ACTION_PAIR_RANK_LOSS_COEF`. Instead of
matching the full signed branch-return distribution, it only ranks the best
observed terminal sibling above the worst observed sibling. Importing the
existing signed Affinity branch-return corpus (35 rows, 8 epochs, frozen
reference KL anchor) reached only `1/16` against CP7 Grixis Affinity. This
rejects "softer branch-return fitting" as the missing transfer operator.

2026-05-14 accepted Affinity log diagnostic addendum: a fresh accepted-policy
CP7 Grixis Affinity compact-log sample scored `3/16`, matching the established
weak matchup. The compact-state exporter now parses `STATE:` rows, allowing
state-aware diagnostics without full verbose logs. The split was mixed:
`9/16` games never cast Spy, Spy-cast games won only `1/7`, and one loss
hard-cast `Dread Return` from visible hand with only three own graveyard cards
and positive value score. Public-board pressure was also severe: games where
the opponent reached at least 10 battlefield permanents went `0/11`. This
points away from more branch-return/value-head variants and toward a narrower
generic source-zone representation probe.

2026-05-14 source-zone feature addendum: the generic source-zone candidate
feature probe was implemented behind `RL_GENERIC_SOURCE_ZONE_FEATURES_ENABLE`.
It marks whether an action source is currently in hand, graveyard, battlefield,
or exile, plus spell-not-from-hand. A 128-episode policy-path-only anchored
adaptation from accepted reached only `4/16` on the CP7 Rally/Affinity pressure
gate (`3/8` Rally, `1/8` Affinity). This rejects source-zone representation
adaptation as a quick fix and weakens the hypothesis that the logged hand-cast
`Dread Return` mistake is the main Affinity bottleneck.

2026-05-14 checkpoint-selection continuation addendum: a clean continuation
clone used Rally+Affinity deck-level pressure, terminal-only rewards, no search,
and a lower frozen-reference policy anchor (`REFERENCE_POLICY_KL_COEF=0.25`).
The 64-episode snapshot reached only `3/16` on the CP7 Rally/Affinity pressure
gate (`2/8` Rally, `1/8` Affinity). The 128-episode snapshot recovered Rally
to `5/8` but Affinity remained `1/8`, for `6/16` combined, still below the fresh
accepted pressure sample (`7/16`, including `2/8` Affinity). This rejects finer
checkpoint slicing of the same hard-opponent continuation as a local quick win.

2026-05-14 gamma 0.97 continuation addendum: a clean continuation clone kept the
same Rally+Affinity pressure setup but shortened the terminal-return horizon to
`PPO_GAMMA=0.97` with Monte Carlo returns (`USE_GAE=0`). The 128-episode run
reached only `3/16` on the CP7 pressure gate (`2/8` Rally, `1/8` Affinity),
below the fresh accepted sample. This rejects shorter-horizon terminal returns
as the missing local quick win and points next toward a different generic
variance-reduction or policy-improvement signal rather than more gamma lowering.

2026-05-14 GAE 0.99 continuation addendum: a clean continuation clone was seeded
from the immutable accepted eval snapshot and forced `USE_GAE=1` with
`PPO_GAMMA=0.995`, `GAE_LAMBDA_HIGH=0.99`, and `GAE_LAMBDA_LOW=0.99`. The
128-episode run reached only `2/16` on the CP7 Rally/Affinity pressure gate
(`1/8` each). This rejects generic value-bootstrapped GAE as a quick local
repair from the accepted checkpoint and closes the current
return-estimator/horizon variant surface.

2026-05-14 multi-deck self-play smoke addendum: isolated D-backed profile clones
retested the older thesis-clean four-profile pure self-play direction without
mutating canonical profile names. Tranche 1 reached only 92 Spy episodes but
showed a small Affinity bump (`3/8` Affinity, `2/8` Rally). A 30-minute
wall-clock continuation brought Spy to 532 episodes, but the pressure gate
regressed to `3/16` (`2/8` Affinity, `1/8` Rally). This rejects the small local
multi-deck self-play smoke as an HPC candidate, while preserving the D-backed
junction setup as useful infrastructure for future controlled multi-profile
runs.
