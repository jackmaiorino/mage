# Thesis-Clean Generic Credit Probe

Date: 2026-05-11

## Question

Can visible, generic decision/state features predict terminal outcome well enough to justify a thesis-clean credit-assignment or history experiment?

This is local-only and does not spend HPC budget.

## Data

- Source model: accepted Affinity-pressure 3h checkpoint.
- Eval: `20260511_accepted_cp7_eval8_full_logs_credit_probe`
- Setup: CP7, 8 games per matchup, 4 matchups, full game logs enabled.
- Result: 14/32 = 43.75%.
  - Spy mirror: 4/8 = 50.00%
  - Jund Wildfire: 3/8 = 37.50%
  - Mono Red Rally: 5/8 = 62.50%
  - Grixis Affinity: 2/8 = 25.00%

Export:

```text
py -3.12 scripts/export_game_log_trajectories.py --root local-training/local_pbt/cp7_eval_sweeps_cdrive/20260511_accepted_cp7_eval8_full_logs_credit_probe/game_logs --output local-training/local_pbt/trajectory_diagnostics/20260511_accepted_cp7_eval8_full_logs_credit_probe/trajectories.jsonl --include-state
```

Exported 32 games and 4,282 decisions.

## Probe Change

`scripts/train_synthetic_return_probe.py` now supports:

```text
--feature-mode generic
```

The generic mode excludes card-name indicators and uses only generic quantities: turn, phase buckets, value score, option/probability summaries, action category flags, life totals, and visible zone counts. The older `card_keywords` mode remains diagnostic-only.

## Results

Generic, no history:

```text
games=32 train_decisions=3242 test_decisions=1040 features=40 actor=PlayerRL1 history=0 feature_mode=generic
train_loss=0.4933 test_loss=0.6122 test_base_loss=0.6983 train_auc=0.8355 test_auc=0.7460
```

Generic, history window 4:

```text
games=32 train_decisions=3242 test_decisions=1040 features=47 actor=PlayerRL1 history=4 feature_mode=generic
train_loss=0.4893 test_loss=0.6004 test_base_loss=0.6983 train_auc=0.8372 test_auc=0.7566
```

Card-keyword diagnostic, history window 4:

```text
games=32 train_decisions=3242 test_decisions=1040 features=160 actor=PlayerRL1 history=4 feature_mode=card_keywords
train_loss=0.2232 test_loss=0.6845 test_base_loss=0.6983 train_auc=0.9730 test_auc=0.8267
```

The card-keyword probe is stronger, but it is off-thesis. The useful result is that generic visible features already carry measurable terminal-outcome signal, and a compact generic history window helps slightly.

## Value-Head Diagnostic

```text
games=32 decision_rows=4150
all: n=4150 base=0.462651 auc=0.638869 gap=0.077749
nontrivial_options: n=2500 base=0.408400 auc=0.678641 gap=0.136368
game mean_value_score: n=32 auc=0.916667 gap=0.091031
game first_value_score: n=32 auc=0.559524 gap=0.009442
game last_value_score: n=32 auc=0.460317 gap=-0.017335
```

Interpretation: the value head has useful aggregate outcome signal, especially over complete games, but single-decision values remain noisy. That matches the observed failure mode: the policy has enough local signal to evaluate broad trajectories but still drifts under multi-step execution.

## Verdict

Proceed to a generic history/credit experiment before spending HPC budget. The evidence is not strong enough to wire synthetic returns into PPO yet, but it is strong enough to test whether a compact, card-agnostic action-history representation improves the online learner.

## Next Experiment

Implement a thesis-clean compact action-history feature path:

- No card names.
- No Spy-specific action flags.
- Use only recent generic action categories, recent selected/top probability summaries, and phase/turn context.
- Run a small local terminal-only continuation from the accepted checkpoint.
- Promote to a larger local run or HPC only if reduced CP7 improves over the accepted checkpoint without harming non-Spy profiles.
