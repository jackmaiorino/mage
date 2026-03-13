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

**Target combo line:** Balustrade Spy targeting self (with no lands in deck) to mill entire library, then Flashback Dread Return targeting Lotleth Giant for lethal Undergrowth damage.

### Architecture

```
RLTrainer (16+ workers per profile)
  └─ GameRunner → XMage game engine
       └─ ComputerPlayerRL (RL agent)
            ├─ StateSequenceBuilder → encode game state → tensor
            ├─ SharedGpuPythonModel → TCP socket → gpu_service_host.py (port 26100)
            │    ├─ score_worker: batched inference (50ms flush, max 64)
            │    └─ train_worker: PPO/policy gradient updates
            └─ action selection (epsilon-greedy from logits)
```

**Two models per profile:**
- **Action model** (`model.pt` / `model_latest.pt`) — transformer policy/value network for in-game decisions
- **Mulligan model** (`mulligan_model.pt`) — separate lightweight network for keep/mulligan + London bottom decisions

**Alternative bridge:** `PythonMLBridge` (Py4J, port 25334) — used when not running shared GPU service.

### PBT (Population-Based Training)

Multiple profiles (A-T in the registry) train the same Spy Combo deck with different hyperparameters. Periodically, the orchestrator (`run_spy_pbt_native.py`) compares winrates:
- **Winner** = highest rolling winrate in the population group
- **Losers** = bottom half of the population
- Exploitation: copy winner's model weights (action + mulligan) to loser, perturb winner's hyperparameters (+/-20%), restart loser trainer with new seed

Registry: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json`

Mutable PBT hyperparameters: `ENTROPY_START`, `ENTROPY_END`, `RL_ACTION_EPS_START`, `RL_FULL_TURN_RANDOM_START`, `TEMPERATURE_FLOOR`, `ACTOR_LR` — all have explicit bounds in `PBT_BOUNDS`.

### Inference Batching

Score requests are grouped by `BatchKey = (policyKey, headId, bucketedSeqLen, dModel, maxCandidates, candFeatDim)`. Candidates are padded to `MAX_CANDIDATES` and sequences bucketed to power-of-2 lengths to maximize batch collisions. The Python GPU service uses a 50ms batch timeout.

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
| `run_spy_pbt.sh` | Slurm sbatch payload — sets up env, modules, launches orchestrator |
| `run_spy_pbt_native.py` | Native Python PBT orchestrator: manages profiles, exploitation, restarts |
| `submit_spy_pbt.slurm` | Slurm batch script (wraps run_spy_pbt.sh) |
| `test_pbt_hpc.py` | Unit tests for PBT orchestrator and GPU service |
| `zaratan_spy_saturation.sh` | Quick saturation test (20 min) |
| `probe_job_metrics.sh` | Extract metrics from finished/running job |
| `collect_job_metrics_local.sh` | Collect metrics from local run |
| `observe_job.sh` / `watch_job.sh` | Live job monitoring |
| `stop_job.sh` / `stop_all_jobs.sh` | Cancel jobs via scancel |
| `slurm_availability.py` | JSON partition availability |
| `transfer_hpc_file.py` | Paramiko SSH transport helper (credential-based) |
| `upload_via_ssh_stdin.py` | Upload files by piping base64 through SSH stdin |
| `download_via_ssh.py` | Download files by piping base64 through SSH exec |
| `commands.txt` | Documented HPC workflow commands |

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
# Build bundle first (Windows), optionally upload with -CredentialFile
.\scripts\hpc\build_rl_runtime_bundle.ps1

# Quick smoke test (on login node)
bash scripts/hpc/zaratan_spy_saturation.sh

# Full PBT sweep
python3 scripts/hpc/spy_saturation.py submit --bundle <tar.gz> --tag <id> --partition gpu-a100 --gres gpu:a100:2 --cpus-per-task 32

# Direct sbatch for overnight runs (on login node)
BUNDLE=$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz | head -1)
sbatch \
  --job-name=spy-overnight \
  --partition=gpu-h100 --gres=gpu:h100:2 --cpus-per-task=32 --mem=128G \
  --time=12:00:00 --account=msml603-class \
  --output=local-training/hpc/bundles/spy_%j.out \
  --error=local-training/hpc/bundles/spy_%j.err \
  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$BUNDLE,RUNNER_OVERSUBSCRIPTION_FACTOR=20,TRAIN_PROFILES=3,TOTAL_EPISODES=10000000,PY_SERVICE_MODE=shared_gpu,GAME_LOG_FREQUENCY=500 \
  scripts/hpc/run_spy_pbt.sh

# Monitor
bash scripts/hpc/probe_job_metrics.sh <jobid>
bash scripts/hpc/observe_job.sh <jobid>
squeue -u $USER
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
| `PY_BATCH_TIMEOUT_MS` | 50 | Inference batch flush timeout (ms) |
| `PY_BATCH_MAX_SIZE` | 64 | Max inference batch size |
| `CUDA_MEM_FRACTION` | 0.90 | GPU memory cap |
| `RL_HEURISTIC_STEP_REWARDS` | 0 | Enable shaped step rewards |
| `RUNNER_OVERSUBSCRIPTION_FACTOR` | 1 | Game parallelism multiplier (use 20+ on HPC) |
| `PY_SERVICE_MODE` | local | `shared_gpu` for multi-profile GPU service |
| `TRAIN_PROFILES` | 3 | Number of profiles to train concurrently |
| `PBT_EXPLOIT_INTERVAL_MINUTES` | 240 | Max minutes between PBT exploitation attempts |
| `PBT_MUTATION_PCT` | 0.20 | Hyperparam perturbation range (+/- fraction) |
| `PBT_MIN_WINNER_GAP` | 0.02 | Min winrate gap for exploitation |
| `PBT_MIN_WINNER_WINRATE` | 0.03 | Min absolute winrate for winner |

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
