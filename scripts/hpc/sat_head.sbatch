#!/bin/bash
#SBATCH --job-name=spy-sathead
#SBATCH --account=msml603-class
#SBATCH --time=01:00:00
#SBATCH --output=local-training/hpc/bundles/sathead_%j.out
#SBATCH --error=local-training/hpc/bundles/sathead_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Satellite-scale smoke HEAD: GPU inference + minimal local runners; CPU satellites
# (add_satellites.sh) provide the bulk of game generation. Tests whether throughput
# multiplies with satellites (the real HPC win). Fixes baked in: auto score workers,
# longer game timeout (survive the cold-start warmup), satellite env written.
set -euo pipefail

TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then echo "FATAL: no tarball" >&2; exit 1; fi
export MAGE_RL_RUNTIME_TARBALL="$TARBALL"

export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"
export TOTAL_EPISODES=500000
export TRAIN_PROFILES=1
export GAME_LOG_FREQUENCY=0
export OPPONENT_SAMPLER=self
export GPU_SERVICE_SPLIT_ROLES=0

# satellites provide game generation; head runs FEW local runners
export WRITE_SATELLITE_ENV=1
export AUTO_SATELLITES=0          # add satellites MANUALLY for the smoke (control + verify)
export PRIMARY_RUNNER_CAP=8
export RUNNER_OVERSUBSCRIPTION_FACTOR=1

# FIXES from the single-node smoke
export SCORE_WORKER_THREADS=0     # auto-scale inference workers (was bottlenecked at 1)
export GAME_TIMEOUT_SEC=900       # survive the ~7s cold-start warmup (was dying at 300s)
export PY_BATCH_MAX_SIZE=1024
export PY_BATCH_TIMEOUT_MS=25

# size S for the throughput proof
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2

# CLEAN raw PPO
export SEARCH_OP_ENABLE=0
export MCTS_TRAINING_ENABLE=0

echo "sat-head starting on $(hostname) job=$SLURM_JOB_ID at $(date -u)"
exec bash scripts/hpc/run_spy_pbt.sh
