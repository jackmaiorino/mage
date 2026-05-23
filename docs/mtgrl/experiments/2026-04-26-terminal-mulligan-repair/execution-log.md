# Execution Log

## 2026-04-26 02:12 UTC

Started Phase B1 terminal-only mulligan repair continuation.

Command:

```powershell
.\scripts\run_arch_validation_generalist.ps1 `
  -Registry .\docs\mtgrl\experiments\2026-04-26-terminal-mulligan-repair\phase-b1-terminal-mulligan-registry.json `
  -TotalEpisodes 15000 `
  -NumGameRunners 96 `
  -GpuServicePort 26100
```

Processes:

- Launcher PowerShell PID: 34032
- `run_local_pbt.py` PID: 96948
- GPU service PID: 131352
- Maven trainer command PID: 125260
- Java trainer PID: 32620

Logs:

- `local-training/local_pbt/terminal_mulligan_repair_20260425_221202.out.log`
- `local-training/local_pbt/terminal_mulligan_repair_20260425_221202.err.log`
- `local-training/local_pbt/trainer.log`
- `local-training/local_pbt/gpu_service.log`

Startup check:

- GPU service ready at 02:12:20 UTC.
- Trainer started at 02:12:20 UTC.
- Episodes advanced from 10,000 to 10,395 by 02:14:20 UTC.
- Early throughput after startup: roughly 4-5 eps/s.

## 2026-04-26 03:11 UTC

Run completed.

- Reached `TOTAL_EPISODES=15000`.
- Trainer exited with `rc=0`.
- Orchestrator stopped the trainer and GPU service.
- Later-run throughput was mostly around 1-2 eps/s, with learner backpressure warnings.
- Shutdown warning: actor learner dropped 297 queued tasks after drain timeout.

## Mulligan Log Check

Parsed `logs/python/mulligan_training.log` after the 15k run.

Summary by episode range:

| Range | First prompts | First mull rate | 0-land keep rate | 1-land keep rate | 0/1-land keep rate | Mean `P_mull` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9500-10000 | 1002 | 43.9% | 54.6% | 58.1% | 56.6% | 0.438 |
| 10001-10500 | 1000 | 46.3% | 51.8% | 54.4% | 53.2% | 0.456 |
| 14500-15000 | 1002 | 32.8% | 71.0% | 68.5% | 69.7% | 0.312 |
| 10001-15000 | 10000 | 39.3% | 61.6% | 61.2% | 61.4% | 0.402 |

Conclusion: this run did not repair mulligans. By the final 500 episodes, the policy was more keep-biased and kept most zero/one-land opening hands. Because logged probabilities are behavior probabilities under mulligan epsilon mixing, the underlying greedy policy is likely even more keep-biased than the final `P_keep ~= 0.68` logs suggest.

## Greedy Mulligan Probe

Ran a greedy opening-hand probe after the 15k checkpoint:

```powershell
.\scripts\run_mulligan_probe.ps1 `
  -SamplesPerDeck 500 `
  -RunId 20260426_terminal_mulligan_15k_greedy_probe `
  -SkipCompile
```

Output directory:

`local-training/local_pbt/mulligan_probes/20260426_terminal_mulligan_15k_greedy_probe`

Deck-level result:

| Deck | First prompts | Keep rate | Mulligan rate | Mean `P_keep` | Mean `P_mull` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spy Combo | 500 | 100.0% | 0.0% | 0.674 | 0.326 |
| Jund Wildfire | 500 | 100.0% | 0.0% | 0.658 | 0.342 |
| Mono Red Rally | 500 | 100.0% | 0.0% | 0.731 | 0.269 |
| Grixis Affinity | 500 | 100.0% | 0.0% | 0.699 | 0.301 |

Bad-hand buckets were also 100% keeps:

- Spy Combo: 301/301 zero-land hands kept, 163/163 one-land hands kept.
- Jund Wildfire: 18/18 zero-land hands kept, 114/114 one-land hands kept.
- Mono Red Rally: 44/44 zero-land hands kept, 132/132 one-land hands kept.
- Grixis Affinity: 39/39 zero-land hands kept, 96/96 one-land hands kept.

Conclusion: do not spend CP7 eval time on this checkpoint as a mulligan-fix candidate. Greedy policy has collapsed to always keep.
