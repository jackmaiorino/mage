# Thesis-Clean Next Steps

Date: 2026-05-11

## Current Read

The accepted thesis-clean checkpoint is the 2026-05-10 Affinity-pressure 3h line:

| Skill | Overall | Spy | Wildfire | Rally | Affinity |
| --- | ---: | ---: | ---: | ---: | ---: |
| CP1 | 133/256 = 51.95% | 49/64 | 31/64 | 22/64 | 31/64 |
| CP3 | 132/256 = 51.56% | 54/64 | 29/64 | 24/64 | 25/64 |
| CP7 | 108/242 = 44.63% | 43/56 | 36/60 | 16/64 | 13/62 |

Two follow-up deck-weight continuations after that checkpoint did not improve the accepted result. The main quick-iteration surface is therefore no longer "try another single-deck RL tweak"; the remaining quick wins are experiment hygiene:

- make bounded runs mean "train more from here" instead of an absolute counter that can silently no-op;
- require corrected unique-matchup evals before promotion;
- compare against the accepted checkpoint across CP1/CP3/CP7 and per-matchup regressions;
- keep HPC parked until a local branch beats the accepted gate.

## Implemented Quick Wins

### Episode-Delta Local Training

`scripts/run_local_pbt.py` now supports:

```text
TOTAL_EPISODES_DELTA=<n>
```

When set, the orchestrator reads the selected profiles' current training counters and sets Java's absolute `TOTAL_EPISODES` to:

```text
max(selected profile current episodes) + TOTAL_EPISODES_DELTA
```

This prevents the known failure mode where `TOTAL_EPISODES=200` exits immediately because active profiles already have hundreds of thousands of episodes. For a single-profile clone, the delta is exact. For multi-profile active runs with uneven counters, every selected profile trains at least the requested delta; lower-counter profiles may train more to reach the shared Java target.

`scripts/run_mtgrl_autonomous_cycle.ps1` also accepts:

```powershell
-EpisodeDelta <n>
```

and records `TOTAL_EPISODES` plus `TOTAL_EPISODES_DELTA` in each phase config.

### Eval Promotion Gate

New script:

```text
scripts/compare_thesis_clean_eval.py
```

It compares candidate CP1/CP3/CP7 sweep directories against baseline CP1/CP3/CP7 sweep directories and fails if:

- CP7 aggregate is below the baseline by default;
- CP1 or CP3 aggregate regresses by more than 2 percentage points;
- any individual matchup regresses by more than 10 percentage points;
- candidate game count is below 90% of the baseline game count.

Example against the accepted checkpoint:

```powershell
py -3.12 scripts/compare_thesis_clean_eval.py `
  --candidate-cp1 local-training/local_pbt/cp7_eval_sweeps_cdrive/<candidate_cp1> `
  --candidate-cp3 local-training/local_pbt/cp7_eval_sweeps_cdrive/<candidate_cp3> `
  --candidate-cp7 local-training/local_pbt/cp7_eval_sweeps_cdrive/<candidate_cp7> `
  --baseline-cp1 local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp1_unique_eval64 `
  --baseline-cp3 local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp3_unique_eval64 `
  --baseline-cp7 local-training/local_pbt/cp7_eval_sweeps_cdrive/20260510_affinity_pressure_3h_spy_cp7_unique_eval64 `
  --markdown-out local-training/local_pbt/<run_id>/promotion_gate.md `
  --json-out local-training/local_pbt/<run_id>/promotion_gate.json
```

## Next Experiment

### Name

Thesis-clean generic library-composition representation, low-risk local branch.

### Hypothesis

The remaining Spy failure is not "cannot identify Balustrade Spy"; logged games show it casts Spy, but almost always while true lands remain hidden in library. A thesis-clean fix should expose generic library-composition information, not card-specific Spy rules.

The existing zone-count feature path is generic but the canonical 3h run with `RL_ZONE_COUNT_FEATURES_ENABLE=1` destabilized Rally/Affinity. The next experiment should test a lower-blast-radius version before spending HPC:

- clone all four accepted profiles;
- reset counters on the clones;
- enable generic zone/library count features on the clones only;
- use the accepted Affinity-pressure opponent pool;
- train a short local branch with `TOTAL_EPISODES=500` or `TOTAL_EPISODES_DELTA=500`;
- evaluate Spy clone on corrected unique CP1 and CP7 first;
- only run full CP1/CP3/CP7 gate if the reduced CP1/CP7 result is not obviously worse.

### Thesis Boundary

Allowed:

- terminal win/loss returns only;
- generic zone/library count features;
- four-profile training;
- generic deck-level opponent curriculum from measured matchup weakness.

Disabled:

- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0`;
- no `SPY_LANDLESS_COMBO_WIN`;
- no action-text regex labels;
- no Spy-reachable opening-hand pools;
- no selective MCTS keywords;
- no heuristic step rewards.

### Local Gate

Do not promote or submit HPC unless the branch beats or ties the accepted checkpoint through `scripts/compare_thesis_clean_eval.py`.

The first reduced screen should be:

```text
CP1 >= 32/64 and CP7 >= 31/64 on Spy-vs-unique-opponents reduced 16-game matchups
```

Those thresholds roughly match the accepted small-sample level and prevent spending hours on a branch that repeats the single-profile regression pattern.

## Deep Work After The Next Experiment

1. Generic search-as-policy-improvement.

   The current value-net MCTS and sparse train-time MCTS variants were not strong enough. The next deep attempt should not just increase MCTS rollouts; it should create better policy-improvement targets from generic prefix search or a stronger terminal search teacher, then distill them without card-name filters.

2. Belief and determinization.

   A card-level belief head plus ISMCTS-style determinization remains standard for imperfect-information games. Keep it thesis-clean by predicting hidden zones from public game history and deck priors, not by naming Spy combo components.

3. Checkpoint-selected multi-day HPC run.

   Zaratan should be used only after a local branch passes the promotion gate. With the current 50 kSU / 30 day constraint, HPC is for scaling a validated thesis-clean mechanism, not for speculative variant cycling.
