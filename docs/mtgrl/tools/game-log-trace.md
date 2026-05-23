# Game Log Trace Viewer

Date: 2026-05-13

## Purpose

Manual game review should not require scrolling through full `GAME STATE` blocks for every decision. `GameLogger` now supports compact logs, and `scripts/view_game_log_trace.py` can summarize either compact or full logs into an action-by-action trace.

## Generate Compact Eval Logs

Example:

```powershell
.\.mtgrl_venv\Scripts\python.exe scripts/run_cp7_eval_sweep.py `
  --profiles Pauper-Spy-Combo-Value `
  --run-id 20260513_compact_log_smoke_affinity1 `
  --games-per-matchup 1 `
  --games-per-job 1 `
  --skill 7 `
  --opponents affinity `
  --eval-game-logging `
  --game-log-format compact `
  --skip-compile
```

Compact logging can also be enabled directly with:

```powershell
$env:GAME_LOG_FORMAT = "compact"
```

Useful knobs:

```text
GAME_LOG_COMPACT_ZONE_CHARS=96
GAME_LOG_COMPACT_ACTION_CHARS=180
GAME_LOG_COMPACT_TOP_OPTIONS=5
```

## View A Trace

```powershell
.\.mtgrl_venv\Scripts\python.exe scripts/view_game_log_trace.py `
  local-training/local_pbt/cp7_eval_sweeps/20260513_compact_log_smoke_affinity1/game_logs `
  --latest 1 `
  --top-options 3 `
  --zone-chars 60 `
  --action-chars 100
```

Output shape:

```text
D023 T3 PlayerRL1-turn actor=PlayerRL1 phase=Precombat Main
  selected[3] p=0.2476 value=-0.011760: Cast Overgrown Battlement
  state: stack=0 items || PlayerRL1 L20 H5[...] B4[...] G1[...] X0 || EvalBot-Skill7 L19 H3 B4[...] G0 X0
  top: n=7 | [1] 0.1384 Cast Lead the Stampede | [2] 0.1879 Cast Wall of Roots | *[3] 0.2476 Cast Overgrown Battlement
```

## Smoke Result

Verified on:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_compact_log_smoke_affinity1
```

The compact log and viewer both worked. The disposable eval DB/snapshot artifacts were removed after verification; the result CSVs and game log remain.

## Export For Analysis

Compact logs can be exported to JSONL for diagnostic scripts:

```powershell
.\.mtgrl_venv\Scripts\python.exe scripts/export_game_log_trajectories.py `
  --root local-training/local_pbt/cp7_eval_sweeps/<run_id>/game_logs `
  --output local-training/local_pbt/cp7_eval_sweeps/<run_id>/diagnostics/games.jsonl
```

The exporter parses compact `TOP:` summaries as the selected action plus the
highest-probability alternatives. This is not a full legal-action list, but it
is enough for manual reviews and value-head diagnostics over compact logs.

Example value-head check:

```powershell
.\.mtgrl_venv\Scripts\python.exe scripts/analyze_logged_value_head.py `
  --input local-training/local_pbt/cp7_eval_sweeps/<run_id>/diagnostics/games.jsonl `
  --out local-training/local_pbt/cp7_eval_sweeps/<run_id>/diagnostics/value_head_analysis
```
