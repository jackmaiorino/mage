# Terminal-Only Mulligan Repair

Date: 2026-04-26

## Goal

Test whether the current terminal-reward generalist can learn useful keep/mulligan behavior without adding heuristic keep rules or non-terminal reward shaping.

The live mulligan decision already goes through the main model as `ActionType.MULLIGAN` and records `stepReward=0.0`; the terminal win/loss return is what trains it. The suspected failure mode is weak data coverage and weak optimization pressure for the first decision of the game, not the absence of a separate hand heuristic.

## Baseline

Checkpoint/profile: `Pauper-Generalist-Value-v2` after the Phase B 10k run.

Valid reduced CP7 sweep:

`local-training/local_pbt/cp7_eval_sweeps/20260426_env_fix_full_reduced_p12`

Summary:

- Overall: 27/80 = 33.75%
- Grixis Affinity as agent: 6/20 = 30%
- Jund Wildfire as agent: 5/20 = 25%
- Mono Red Rally as agent: 11/20 = 55%
- Spy Combo as agent: 5/20 = 25%

Observed issue: game logs included bad keeps, including zero-land hands.

## Changes Under Test

- Add mulligan-specific epsilon mixing for `ActionType.MULLIGAN`.
- Add optional sample-loss weighting for `MULLIGAN` and `LONDON_MULLIGAN` samples.
- Keep `RL_HEURISTIC_STEP_REWARDS=0`.
- Keep `MULLIGAN_HARD_OVERRIDES_ENABLE=0`.
- Keep CP7 and MCTS out of training.
- Use `MULLIGAN_TERMINAL_ONLY=1` for the legacy standalone mulligan trainer path, even though live play uses the main model.

This is one bundled experiment because all changes target the same bottleneck: making terminal-only mulligan learning visible to the optimizer. MCTS, architecture, deck/profile structure, and opponent curriculum are intentionally held fixed.

## Run

Continuation target: 15k total episodes, then run a reduced CP7 sweep.

Training:

```powershell
.\scripts\run_arch_validation_generalist.ps1 `
  -Registry .\docs\mtgrl\experiments\2026-04-26-terminal-mulligan-repair\phase-b1-terminal-mulligan-registry.json `
  -TotalEpisodes 15000 `
  -NumGameRunners 96 `
  -GpuServicePort 26100
```

Evaluation:

```powershell
.\scripts\run_arch_validation_cp7_reduced.ps1 `
  -GamesPerMatchup 5 `
  -Parallel 12 `
  -AiThreads 2 `
  -BatchTimeoutMs 10 `
  -SkipCompile `
  -RunId 20260426_terminal_mulligan_15k_p12
```

## Decision Criteria

At 15k total episodes:

- CP7 overall and by-agent winrates should not regress from baseline.
- Zero/one-land keeps should be materially lower in mulligan logs.
- Spy Combo and Jund Wildfire should show improved opening-hand quality, even if total winrate is noisy.

If mulligans improve, extend to 25k total before changing anything else. If mulligans do not improve, inspect action logits/features and the discount/return path before trying MCTS or architecture changes.
