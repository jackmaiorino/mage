# Multi-Node RL-vs-RL League Training System

## Summary

Replace the current single-node CP7-bot training setup with a two-node Slurm job running pure RL-vs-RL self-play across 5 pauper deck archetypes. GPU node handles inference/training, CPU node runs game simulation. PBT exploitation decisions use a periodic eval benchmark against CP7 heuristic bots.

## Motivation

- CP7 heuristic bots are the slowest component in the training pipeline and provide weak training signal
- Current 80% winrate is misleading -- the agent beats braindead bots via creature beatdown, not by executing the combo
- Co-evolving agents across deck archetypes produces realistic metagame pressure
- Separating GPU and CPU onto dedicated nodes removes the CPU bottleneck (512 cores vs current 96)

## Architecture

```
Slurm 2-node heterogeneous job (sbatch het-group syntax)

  Het-group 0: gpu-h100 node (4x H100, 96 cores)
    +-- gpu_service_host.py x4 (one per GPU, ports 26100-26103, bind 0.0.0.0)
    +-- run_spy_pbt_native.py (orchestrator)
    +-- Metrics collection

  Het-group 1: compute node (128 cores)
    +-- 10x Java RLTrainer processes (via srun --het-group=1)
         +-- ~250 game runners each (128 cores * oversubscription_factor=20 / 10 profiles)
            +-- SharedGpuPythonModel -> <gpu-node>:26100-26103 over InfiniBand
```

### Filesystem Assumption

Zaratan uses Lustre shared filesystem for both home and scratch directories. All nodes in the job see the same file paths. Model files (`model_latest.pt`, `mulligan_model.pt`) written by the GPU service on the GPU node are immediately visible to Java trainers on the CPU node. No cross-node file transfer is needed.

### Node Discovery

Orchestrator parses `SLURM_JOB_NODELIST` on startup to identify:
- GPU node: first node in the list (where orchestrator is running)
- CPU node: second node in the list

Exports `GPU_SERVICE_ENDPOINT=<gpu-hostname>:26100` for all Java trainer launches.

### Process Placement

- `gpu_service_host.py` starts locally on GPU node with `GPU_SERVICE_BIND_HOST=0.0.0.0`
- Profiles are distributed across 4 GPU instances (profiles 0-2 on GPU 0, 3-4 on GPU 1, 5-7 on GPU 2, 8-9 on GPU 3) using the existing `gpu_slot` mechanism and port offsets 26100-26103
- Java trainers launch on CPU node via `srun --het-group=1 --overlap`
- Trainer kill/restart uses the srun process handle (SIGTERM)

## League System

### Profiles

10 active profiles across 5 deck archetypes, 2 profiles each:

| Population Group | Profiles | Deck |
|---|---|---|
| spy-combo | Pauper-Spy-Combo-A, Pauper-Spy-Combo-B | Deck - Spy Combo.dek |
| elves | Pauper-Elves-A, Pauper-Elves-B | Deck - Elves.dek |
| rally | Pauper-Rally-A, Pauper-Rally-B | Deck - Mono Red Rally.dek |
| affinity | Pauper-Affinity-A, Pauper-Affinity-B | Deck - Grixis Affinity.dek |
| wildfire | Pauper-Wildfire-A, Pauper-Wildfire-B | Deck - Jund Wildfire.dek |

Existing Spy Combo profiles A and B are retained. All other existing Spy Combo profiles (C through T) and frozen opponent-pool profiles are removed or marked inactive.

### Opponent Selection

Pure RL-vs-RL. No CP7 bots during training.

For each game, the trainer:
1. Picks a random profile from the registry (excluding self)
2. Creates a `ComputerPlayerRL` using that profile's `model_latest.pt` as the policy
3. Sets `THREAD_LOCAL_OPPONENT_DECK_OVERRIDE` to that profile's deck path

Controlled by new env var `LEAGUE_MODE=rl_only`. When set, `createLeagueOpponent()` skips the qualified/promoted check and the CP7 fallback. All meta candidates become RL opponents.

**Cold start:** At job start, no profiles have a `model_latest.pt` yet. The `rl_only` mode handles this: if a picked opponent profile has no model file, the opponent `ComputerPlayerRL` uses randomly initialized weights (the default behavior when `SharedGpuPythonModel` has no checkpoint loaded for a policy key). This is fine -- random-vs-random produces meaningful gradient signal because the agent that plays more spells wins.

### PBT Exploitation

Within same-deck population groups only (e.g., Spy-A vs Spy-B).

Uses **eval winrate** (not training rolling winrate) to compare profiles. Training winrate trends toward 50% in self-play and is not useful for PBT decisions. Training winrate is still logged for monitoring.

**Eval-to-PBT data flow:**
1. `run_eval_benchmark()` returns a dict `{profile_name: eval_winrate}` (float 0.0-1.0)
2. Orchestrator stores this in `self.eval_results[profile_name] = eval_winrate`
3. `invoke_pbt_exploit()` reads `self.eval_results[profile]` instead of `snapshot["rolling_current"]` when ranking candidates within a population group
4. If a profile has no eval result yet (first exploitation check before first eval completes), it is excluded from PBT comparison for that round

All existing PBT mechanics remain: model copy (action + mulligan), hyperparameter perturbation with bounds, winner's env as perturbation base.

## Eval Benchmark

### Trigger

Runs before each PBT exploitation check (every `PBT_EXPLOIT_INTERVAL_MINUTES`, default 240 min).

### Process

1. Orchestrator stops all 10 Java trainers on the CPU node (SIGTERM, wait for clean shutdown)
2. For each of the 10 profiles, run 100 eval games:
   - 50 vs CP7-Skill1 playing Mono Red Rally deck
   - 50 vs CP7-Skill1 playing Mono-Blue Terror deck
3. Eval games run on CPU node via temporary RLTrainer in eval mode (`args=eval`)
4. Profile's current `model_latest.pt` loaded as frozen policy (no training updates, no `gpu_service_host.py` train worker calls)
5. Results: win count / 100, logged per profile to `eval_results.csv` with timestamp and stored in `self.eval_results` dict
6. Orchestrator uses eval winrates for PBT exploitation decisions
7. Orchestrator restarts all 10 trainers (with any PBT-exploited changes applied)

Stopping trainers before eval ensures model weights are stable during evaluation. Restarting after eval naturally applies any PBT exploitation changes.

### Performance Cost

1000 eval games total (100 x 10 profiles). These can run in parallel across all 128 cores. With oversubscription, completes in a few minutes. The stop/eval/exploit/restart cycle adds ~5 minutes of downtime per PBT interval.

## Code Changes

### `run_spy_pbt_native.py` (orchestrator)

1. **Node discovery** -- parse `SLURM_JOB_NODELIST` into GPU_NODE and CPU_NODE on startup
2. **Remote process launch** -- use `srun --het-group=1 --overlap` for trainer start/stop on CPU node
3. **League-mode flag** -- set `LEAGUE_MODE=rl_only` in trainer env
4. **Eval harness** -- new `run_eval_benchmark()` method called before `attempt_pbt_exploitation()`. Stops trainers, launches eval trainer processes on CPU node, collects results into `self.eval_results` dict, returns `{profile: winrate}`.
5. **PBT uses eval winrate** -- `invoke_pbt_exploit()` reads `self.eval_results[profile]` instead of `snapshot["rolling_current"]` when ranking candidates. Profiles without eval results are skipped.
6. **GPU service distribution** -- launch 4 `gpu_service_host.py` instances (one per H100), assign profiles to GPU slots via round-robin or balanced allocation

### `RLTrainer.java`

1. **`LEAGUE_MODE` env var** -- when `rl_only`, `createLeagueOpponent()` always creates `ComputerPlayerRL` for picked meta candidate, never falls back to CP7. If opponent has no model file, `ComputerPlayerRL` uses random weights.
2. **Eval mode** -- when `args=eval`, run N games against a specified CP7 opponent (deck + skill level), report results to stdout or file, exit. Reuses game runner infrastructure with training disabled.

### `run_spy_pbt.sh`

1. Rewrite as heterogeneous job using Slurm het-group syntax (see Slurm Submission section)
2. Parse `SLURM_JOB_NODELIST` into GPU_NODE and CPU_NODE
3. Start `gpu_service_host.py` x4 locally with `GPU_SERVICE_BIND_HOST=0.0.0.0` and `CUDA_VISIBLE_DEVICES=0,1,2,3`
4. Pass GPU_NODE hostname to orchestrator via env vars

### `pauper_spy_pbt_registry.json`

- Remove all Spy Combo profiles except A and B (drop C through T)
- Remove or mark inactive all frozen opponent-pool profiles (Elves, Caw-Gates, etc. -- these become active trainable profiles now)
- Add 8 new active profile entries (2 each for Elves, Rally, Affinity, Wildfire)
- Each with `deck_path`, `population_group`, and `pbt_mutable_env`

### No Changes Needed

- `SharedGpuPythonModel.java` -- `GPU_SERVICE_ENDPOINT` already supports `host:port`
- `gpu_service_host.py` -- bind host already configurable via `GPU_SERVICE_BIND_HOST` env var

## Slurm Submission

Zaratan heterogeneous jobs use the `#SBATCH` pack-group separator. The sbatch script (`run_spy_pbt.sh`) header becomes:

```bash
#!/bin/bash
#SBATCH --job-name=league-multinode
#SBATCH --account=msml603-class
#SBATCH --time=12:00:00
#SBATCH --output=local-training/hpc/bundles/league_%j.out
#SBATCH --error=local-training/hpc/bundles/league_%j.err

# Het-group 0: GPU node
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

#SBATCH hetjob

# Het-group 1: CPU node
#SBATCH --partition=compute
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
```

The exact partition names and resource limits need validation against Zaratan's current configuration (`scontrol show partition`). The `compute` partition may use a different name or have per-node core limits that differ from 128.

Environment variables passed via `--export` on the sbatch command line:
```
LEAGUE_MODE=rl_only
TRAIN_PROFILES=10
TOTAL_EPISODES=10000000
PY_SERVICE_MODE=shared_gpu
GAME_LOG_FREQUENCY=500
RUNNER_OVERSUBSCRIPTION_FACTOR=20
```

## Success Criteria

- All 10 profiles train concurrently across 2 nodes
- No CP7 bots in training games -- all opponents are RL
- Eval benchmark produces comparable absolute winrates across runs
- PBT exploitation uses eval winrate, not training winrate
- Episode throughput significantly higher than single-node (target: 5x+)
