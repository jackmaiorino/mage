# Project Guidelines for AI Assistants

## Critical Rules

1. **DO NOT create or modify documentation files unless explicitly requested**
2. **Be minimally verbose** - Short, direct responses. Code changes speak for themselves.
3. **Focus on implementation over discussion** - Make changes, don't just suggest them.
4. **NO EMOJIS** - Never use emojis in any response or file, ever.

## What User Values

- **Efficiency** - Quick, accurate changes
- **Brevity** - Get to the point
- **Action** - Implement, don't just discuss
- **Honesty** - Disagree when wrong
- **No fluff** - No unnecessary documentation

## Project Overview

**XMage RL Player** - Reinforcement learning AI for Magic: The Gathering (XMage fork).

**Current focus:** Training the Pauper Spy Combo deck via Population-Based Training (PBT) at scale on Zaratan HPC (UMD cluster). Goal: massive parallel runs that teach the agent to execute the Spy combo reliably.

### Architecture

```
RLTrainer (16+ workers)
  └─ GameRunner → XMage game engine
       └─ ComputerPlayerRL (RL agent)
            ├─ StateSequenceBuilder → encode game state → tensor
            ├─ SharedGpuPythonModel → TCP socket → gpu_service_host.py (port 26100)
            │    ├─ score_worker: batch inference (200ms flush, max 64)
            │    └─ train_worker: PPO/policy gradient updates
            └─ action selection (ε-greedy from logits)
```

**Alternative bridge:** `PythonMLBridge` (Py4J, port 25334) — used when not running shared GPU service.

### Key Classes

| File | Role |
|------|------|
| `ComputerPlayerRL.java` | Main RL agent: mulligan, casting, combat decisions |
| `RLTrainer.java` | Training orchestrator; spawns workers, manages episodes |
| `StateSequenceBuilder.java` | Game state → neural net input tensors |
| `SharedGpuPythonModel.java` | Multiplexed GPU inference client (TCP, opcodes 1-16) |
| `SharedGpuTensorSerde.java` | Binary tensor serialization over socket |
| `PythonMLBridge.java` | Py4J-based bridge (alternative to shared GPU service) |
| `MetricsCollector.java` | Prometheus HTTP metrics server |
| `RLLogPaths.java` | Centralized artifact paths (profile-aware layout) |
| `EnvConfig.java` | Environment variable parser |

### Key Python Files (MLPythonCode/)

| File | Role |
|------|------|
| `gpu_service_host.py` | Shared GPU server; multiplexes score/train requests |
| `py4j_entry_point.py` | Py4J entry point (single-process mode) |
| `mtg_transformer.py` | PyTorch transformer policy/value network |
| `model_persistence.py` | Checkpoint save/load with atomic writes |
| `mulligan_model.py` | Separate lightweight mulligan network |
| `snapshot_manager.py` | Checkpoint versioning for curriculum/league |
| `profile_paths.py` | Python-side path resolution (mirrors RLLogPaths) |
| `logging_utils.py` | Categorized logging with per-category on/off flags |
| `cuda_manager.py` | CUDA init, warmup, memory diagnostics |

### HPC Scripts (scripts/hpc/)

| File | Role |
|------|------|
| `build_rl_runtime_bundle.ps1` | Bundle Java+Python → `rl-runtime-*.tar.gz`, SCP to HPC |
| `spy_saturation.py` | PBT sweep orchestrator: submit/summarize/collect |
| `run_spy_pbt.sh` / `run_spy_pbt_native.py` | Submit PBT jobs to Zaratan |
| `test_pbt_hpc.py` | Smoke test PBT setup on HPC |
| `zaratan_spy_saturation.sh` | Quick saturation test (20 min) |
| `probe_job_metrics.sh` | Extract metrics from finished/running job |
| `collect_job_metrics_local.sh` | Collect metrics from local run |
| `observe_job.sh` / `watch_job.sh` | Live job monitoring |
| `stop_job.sh` / `stop_all_jobs.sh` | Cancel jobs via scancel |
| `slurm_availability.py` | JSON partition availability |
| `transfer_hpc_file.py` | File transfer utility |
| `commands.txt` | 88 documented HPC workflow commands |

### Deck Lists (decks/PauperSubset/)

- `decklist.txt` — default training pool
- `decklist.spy_combo.txt` — **Spy Combo** (primary target deck)
- `decklist.spy_combo_plus_pool.txt` — extended variant
- `decklist.elves.txt`, `decklist.monored-rally.txt` — other archetypes

### Artifact Layout (profile-aware)

```
rl/profiles/<MODEL_PROFILE>/
  models/   → model.pt, model_latest.pt, mulligan_model.pt, episode_counter
  logs/     → training_stats.csv, game logs, mulligan logs, health stats
```

Without `MODEL_PROFILE`: legacy `rl/models/` and `rl/logs/`.

## Build Commands

**Compile:**
```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

**Train locally (spy combo):**
```powershell
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.spy_combo.txt"
$env:MODEL_PROFILE="spy"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
```

**HPC submission:**
```bash
# Build bundle first (Windows)
.\scripts\hpc\build_rl_runtime_bundle.ps1

# Quick smoke test
bash scripts/hpc/zaratan_spy_saturation.sh

# Full PBT sweep
python3 scripts/hpc/spy_saturation.py submit --bundle <tar.gz> --tag <id> --partition gpu-a100 --gres gpu:a100:2 --cpus-per-task 32

# Monitor
bash scripts/hpc/probe_job_metrics.sh <jobid>
bash scripts/hpc/observe_job.sh <jobid>
```

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_GAME_RUNNERS` | 16 | Parallel game workers |
| `TOTAL_EPISODES` | 10000 | Training episodes |
| `MODEL_PROFILE` | — | Artifact subdirectory name |
| `DECK_LIST_FILE` | — | Deck(s) for training |
| `RL_AGENT_DECK_LIST` | — | Override agent-side deck |
| `ADAPTIVE_CURRICULUM` | 1 | Enable adaptive opponents |
| `WINRATE_WINDOW` | 100 | Rolling winrate window |
| `GPU_SERVICE_PORT` | 26100 | Shared GPU service TCP port |
| `GPU_SERVICE_METRICS_PORT` | 27100 | GPU service Prometheus port |
| `PY_BATCH_TIMEOUT_MS` | 200 | Inference batch flush timeout |
| `PY_BATCH_MAX_SIZE` | 64 | Max inference batch size |
| `CUDA_MEM_FRACTION` | 0.90 | GPU memory cap |
| `RL_HEURISTIC_STEP_REWARDS` | 0 | Enable shaped step rewards |
| `RUNNER_OVERSUBSCRIPTION_FACTOR` | — | Game parallelism multiplier (HPC) |

## Tech Stack

- **Java 8+** / Maven — XMage game engine + RL agent
- **Python 3.8+** / PyTorch — transformer model
- **Py4J** — Java↔Python bridge (single-process mode)
- **TCP socket protocol** — shared GPU service (multi-profile mode)
- **Slurm / Zaratan** — HPC cluster (UMD)

## Code Style

- Java 8+ features OK
- Thread-safe for parallel execution
- Minimal comments
- No over-engineering

## Remember

**Less is more. User will ask if they want more.**
