# Spy Natural Winning-Line BC

date: 2026-05-04

## Question

Can we avoid the noisy London keep-vs-mull counterfactual labels by cloning whole natural-start lines that actually won?

This keeps the reward source terminal-only: a trajectory is used only if the forced London line won the game.

## Setup

- base profile: `Pauper-Spy-Combo-ActualWin-Big-Offline-20260504`
- agent deck: `Deck - Spy Combo.dek`
- opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- opponent skill: CP1
- line enumerator: keep 7, mulligan to 6 with 7 bottom combos, mulligan to 5 with 7 bottom combos
- export: one winning line per scenario, converted to one-hot BC targets

Smoke collection:

- run: `20260504_spy_natural_winline_collect64_smoke`
- scenarios: 64
- trained/actionable scenarios: 26
- skipped scenarios: 38
- selected counterfactual prompt samples: 59
- exported winning-line BC samples: 2,760
- elapsed: 1,060.7 sec

## Results

Before training on the exported winning-line samples:

- score run: `20260504_natural_winline_big_before_score_smoke64_alltypes_q`
- exact-label score: 999/2760 = 36.20%
- average target probability: 0.334

After 20 epochs:

- profile: `Pauper-Spy-Combo-NaturalWinLineBC-Smoke-20260504`
- train run: `20260504_natural_winline_bc_smoke64_train_e20`
- train passes: 55,200
- score run: `20260504_natural_winline_bc_smoke64_score_e20`
- exact-label score: 1040/2760 = 37.68%
- average target probability: 0.331

After 100 total epochs:

- additional train run: `20260504_natural_winline_bc_smoke64_train_add80`
- additional train passes: 220,800
- score run: `20260504_natural_winline_bc_smoke64_score_e100`
- exact-label score: 929/2760 = 33.66%
- average target probability: 0.330

The head-only variant was stopped after more than one hour with no completed output.

## Interpretation

This path is not ready to scale. The winning-line data can be generated at a useful rate, but the current imported distillation path did not fit the exported trajectory tensors. More data would be wasted until the supervised objective or tensor conversion is fixed.

This is different from the actual-win safe-hand dataset, where the big model scored 81.13% top-1 on 10k held imported examples. The failure is specific to these trajectory-clone tensors or their noisier labels, not proof that imported training is globally broken.

## Next Experiment

Fix the distillation path before generating more data:

1. Add an explicit supervised cross-entropy BC loss for imported one-hot `mctsVisitTargets`, separate from PPO/value terms.
2. Log per-head CE/top-1 before and after a train call in the same JVM.
3. Re-run the 64-scenario winning-line smoke and require exact-label score to move materially before CP1 eval.
4. Only if exact-label score improves, scale collection to 512+ scenarios.

Primary artifacts:

- `local-training/local_pbt/mulligan_counterfactual/20260504_spy_natural_winline_collect64_smoke`
- `local-training/local_pbt/spy_line_search/20260504_natural_winline_bc_smoke64_train_e20`
- `local-training/local_pbt/spy_line_search/20260504_natural_winline_bc_smoke64_train_add80`
- `local-training/local_pbt/spy_line_search/20260504_natural_winline_bc_smoke64_score_e100`
