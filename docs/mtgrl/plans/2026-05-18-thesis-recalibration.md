# Thesis Recalibration - 2026-05-18

## North Star

The thesis is not "can we hand-engineer a better Spy policy." The thesis is:

An RL model can learn to pilot a complex Magic combo deck at a strong level from
win-driven feedback, without being explicitly guided toward human Magic
intuition. The extension is to use the same methodology to discover decks,
strategies, and emergent behavior where the only objective is winning.

This means experiment choices should be judged by whether they strengthen that
claim:

1. Does the policy learn winning behavior from terminal outcomes or generic
   game information?
2. Does the experiment avoid card-specific tactical supervision and human combo
   scripts?
3. Does the evaluation prove fresh-game winning strength, not just tensor fit or
   replay success?
4. Does the mechanism plausibly scale to other decks or strategy discovery?

## What The Past Month Shows

### The positive thesis evidence is real

The early terminal-only four-profile meta self-play run on 2026-04-25 failed
badly at CP7. Spy went 5/79 overall and 0/20 against Rally and Affinity.

The 2026-05-01 to 2026-05-02 two-GPU terminal-only self-play continuation was a
turning point operationally. It showed the local system can generate hundreds of
thousands of terminal-reward games per day, and Spy was no longer a zero-result
policy. It still failed hard matchups, but the learning signal existed.

The main thesis-positive result remains the 2026-05-10 thesis-clean
hard-matchup curriculum. It used terminal win/loss returns only plus generic
deck-level opponent weighting from measured weakness. It did not use
Spy-specific labels, card-name features, regex targets, hand pools, heuristic
step rewards, MCTS gates, or human combo scripts. The accepted checkpoint reached:

```text
CP1: 133/256 = 51.95%
CP3: 132/256 = 51.56%
CP7: 108/242 = 44.63%
CP7 Grixis Affinity: 13/62 = 20.97%
```

This is the result that best supports the thesis: the agent learned much more
credible combo-deck play from winning pressure and generic curriculum design.

### The negative result is also clear

Most work after the accepted checkpoint tested local repair surfaces. These
surfaces produced useful diagnostics but did not improve fresh games:

- one-step prefix BC/KL, DAgger, first-deviation repair, trajectory KL;
- candidate-Q, branch-return Q, branch-pair value or policy ranking;
- root mulligan repair and deep-mulligan distillation;
- representation-only features such as zone counts, public-board features,
  source-zone features, semantic flags, action-class flags, and belief labels;
- repaired MCTS, flat value MCTS, sparse train-time MCTS, random-rollout ISMCTS,
  and online prefix root/search hooks;
- checkpoint soups, policy ensembles, gamma/GAE tweaks, and more deck-weight
  continuation after the accepted checkpoint.

The repeated pattern is important: many teachers found labels or winning lines,
and the accepted policy often scored those labels well locally, but fresh CP
games did not improve. The bottleneck is not just "the model does not know the
one best action." The bottleneck is distribution shift and sequential
compounding in live games.

### Search and replay are tools, not the thesis

Terminal prefix search and branch/counterfactual tools have proven useful for
diagnosis. They can find winning lines, identify accepted-policy mistakes, and
explain drift. However, direct distillation from these tools has repeatedly
failed fresh-game gates.

The 2026-05-18 replay work is justified only as substrate work for honest
experiments. The v83 result is a useful milestone:

```text
current-family D013 source-profile CP7 replay: matched=1/1
Land Grant: 338->392
Roost: 499->552
Omen cleanup: 552->606
D013: Cast Overgrown Battlement matched
```

That proves we can replay at least one current-family Affinity decision segment
with enough provenance to make counterfactual/search conclusions meaningful. It
does not, by itself, advance the thesis unless it leads back to winning-policy
experiments.

### Affinity is the right diagnostic pressure, but not the whole thesis

Affinity remains the sharpest CP7 weakness. Accepted-policy samples are
consistent:

```text
long-run accepted CP7 Grixis Affinity: 13/62 = 20.97%
2026-05-14 accepted Affinity log sample: 3/16 = 18.75%
2026-05-18 combined Affinity corpus: 7/32 = 21.88%
```

The failure is mixed:

- many games never cast Spy;
- Spy-cast games still often lose;
- opponent battlefield pressure around >=10 permanents is severe;
- some losses include source-zone mistakes such as Dread Return visible in hand,
  not graveyard.

This makes Affinity a good stress test for whether the learned combo policy is
robust. But the goal is still general win-driven learning. Affinity repair must
remain generic enough to support the thesis.

## Recalibrated Trajectory

### 1. Keep the accepted checkpoint as the thesis anchor

The 2026-05-10 Affinity-pressure checkpoint is the accepted reference until a
new branch beats it on fresh games. Do not treat small replay or tensor metrics
as promotion evidence.

Promotion gates remain:

```text
CP7 Grixis Affinity screen: >= 5/16, then >= 10/32
Pressure pair screen: Rally plus Affinity >= accepted fresh sample, about 7/16
Four-opponent CP7 aggregate: >= accepted 108/242, with Affinity not below 13/62
Final gate: compare_thesis_clean_eval.py across CP1/CP3/CP7
```

### 2. Cap replay-parity work to policy-relevant gates

Do not walk every decision ordinal for its own sake. Replay work should stop as
soon as it can answer a policy-relevant question.

Allowed replay targets:

- a first reproducible setup decision needed to prove state reconstruction;
- the first nontrivial pressure-region decision where accepted likely loses;
- a branch point where a sibling or short prefix can plausibly recover a win.

Low-value targets, such as routine mana/pass-adjacent decisions, should be used
only if they are required bridges to reach a pressure target.

After v83, the controller should choose the next target before launching work.
The likely next target is not "D025 because it is next." It should be the first
current-family decision that tests winning recovery under Affinity pressure,
probably around the first non-pass pressure region such as D089, unless a bridge
gate is strictly required to reach it.

### 3. Make the next experiment a corpus density test, not training

Before any new training branch, prove the data source is dense enough:

1. Use replay-metadata/current-family Affinity logs.
2. Select accepted-policy loss states under public-board pressure.
3. For each candidate, test whether a sibling action or short generic prefix can
   reach a terminal win or materially improve the terminal outcome.
4. Export only cases where the baseline loses and the correction wins.

Training should remain blocked unless the corpus reaches roughly:

```text
>= 50 accepted-policy failure states
accepted top-1 on corrective first action < 50%
multiple pressure clusters represented
terminal-winning or terminal-improving corrected suffixes preserved
```

This aligns with the thesis because it tests whether the learned policy has
generic, correctable failure modes in live game states, without injecting human
combo scripts.

### 4. If training resumes, use trajectory context and terminal outcomes

One-step imports have failed too often to be the next primary mechanism.

The next training attempt should use a recovery trajectory teacher only after
the corpus density test passes:

- force the corrected action or short prefix;
- continue with generic terminal search or normal policy execution;
- keep the corrected suffix only if it reaches a terminal win;
- train mainly on the corrective action plus downstream decisions inside the
  successful terminal trajectory;
- keep frozen-reference KL and low learning rate;
- evaluate first on CP7 Affinity, then pressure pair, then full CP gates.

This is still thesis-clean if the branch generator is generic and the learning
target remains winning.

### 5. Keep terminal-only curriculum as the main thesis line

The best positive evidence came from terminal-only self-play with generic
hard-matchup pressure. If the recovery-corpus path is sparse or too fragile,
return to the main thesis line instead of inventing another small supervised
repair:

- longer controlled terminal-only continuation from the accepted checkpoint;
- generic opponent weighting from measured weakness;
- strict gates against accepted CP1/CP3/CP7;
- no card-specific features, no heuristic rewards, no human combo scripts.

The purpose is to demonstrate win-driven learning, not to maximize local
engineering cleverness.

### 6. Keep belief/determinization separate until calibrated

Belief remains a plausible long-term route for imperfect-information strategy,
especially for future deck/strategy discovery. But recent belief experiments
were not calibrated enough to drive gameplay.

Next belief work should be calibration-only:

- predict hidden-zone and archetype summaries from public history and deck
  priors;
- beat simple deck-prior baselines across non-Spy opponents;
- only then wire belief into search or training.

## Immediate Controller Policy

For autonomous research workers:

1. Start every cycle by restating the north-star thesis.
2. Treat replay parity as a dependency for honest counterfactuals, not a
   deliverable.
3. Do not launch training from v83. First choose a policy-relevant pressure
   target and ask whether it can produce terminal-winning corrections.
4. Prefer foreground CLI workers for short gates because v82/v83 showed that
   foreground launch works while earlier background attempts failed.
5. Log every run to the automation ledger, durable memory, and a per-run
   diagnostic path.

## Next Concrete Unit

Controller-selected next unit:

Build a target-selection manifest for the v77/v83 current-family Affinity loss
and the existing 32-game Affinity corpus. The manifest should rank candidate
targets by thesis value:

```text
1. pressure-region decisions with opponent permanents >= 10
2. accepted low-probability corrective candidates
3. source-zone or cast-timing mistakes that are generic, not card-scripted
4. whether a bounded sibling/short-prefix search can reach terminal win
```

Only after that manifest selects a target should a worker build the next replay
CSV and run one bounded source-profile CP7 gate.
