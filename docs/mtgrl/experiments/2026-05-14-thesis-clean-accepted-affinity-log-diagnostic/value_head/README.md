# Logged Value-Head Diagnostic

This analyzes the model's logged `VALUE SCORE` against the terminal game result. No probe is trained here.

## Headline

- Decision rows: n=1863, win base rate=0.194847, AUC=0.660896, mean win score=0.039738, mean loss score=-0.027795.
- Nontrivial decision rows: n=1124, AUC=0.717537, score gap=0.115441.
- Critical combo rows: n=12, AUC=0.650000, score gap=0.189996.
- Per-game mean value score: n=16, AUC=0.846154, score gap=0.067526.

## Files

- `summary.csv`: decision-level subsets.
- `by_matchup.csv`: decision-level matchup splits.
- `game_aggregate_metrics.csv`: one-row-per-game value aggregations.
- `game_values.csv`: raw per-game aggregate values.
- `decision_samples.csv`: high-score losses and low-score wins for log inspection.
