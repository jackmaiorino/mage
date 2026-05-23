# Mulligan Counterfactual Repair

Date: 2026-04-26

## Reason

The 15k terminal mulligan repair run did not improve mulligans. The greedy probe
kept every opening hand across the active four-deck pool, including all zero-land
and one-land hands.

## Change

Train opening keep/mull with terminal-only paired branch labels instead of hand
heuristics or intermittent rewards.

- Add a dedicated `mulligan` policy head.
- Keep London bottom decisions on the existing `card_select` head.
- For each sampled opening hand, run two full terminal branches from the same
  initial deck order:
  - first decision forced to KEEP
  - first decision forced to MULLIGAN
- Convert terminal branch outcomes into a soft candidate target:
  - KEEP win / MULL loss => `[1.0, 0.0]`
  - KEEP loss / MULL win => `[0.0, 1.0]`
  - same result on both branches => `[0.5, 0.5]`
- Train via the existing `mctsVisitTargets` policy-distillation loss with PPO,
  value, entropy, and belief losses disabled for this repair pass.

## Implementation

- Java entrypoint:
  `mage.player.ai.rl.MulliganCounterfactualTrainer`
- Runner:
  `scripts/run_mulligan_counterfactual.ps1`
- Local Py4J training now forwards optional archetype labels and
  `mctsVisitTargets`, matching the shared GPU path.
- ONNX export and inference head lists now include `model_mulligan.onnx`.

## Smoke Result

Command:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 `
  -Profile Mulligan-CF-Smoke `
  -Pairs 1 `
  -BatchSize 1 `
  -TimeoutSec 20 `
  -Opponent rl `
  -ServiceMode local `
  -RunId smoke_20260426_000412 `
  -SkipCompile
```

Result:

- `trained=1`
- `skipped=0`
- learner stats: `train_samples=1`, `train_steps=1`
- sample label: KEEP target from terminal branch outcome

The temporary smoke checkpoint profile was deleted after verification.

## Next Run

Recommended first real run:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 `
  -Profile Pauper-Generalist-Value-v2 `
  -Pairs 256 `
  -BatchSize 32 `
  -TimeoutSec 45 `
  -Opponent rl `
  -ServiceMode local
```

Success gate:

- Run `scripts/run_mulligan_probe.ps1` after the pass.
- Continue only if greedy opening-hand behavior stops being an all-keep policy,
  especially on zero-land and one-land first hands.
