#!/bin/bash
#SBATCH --job-name=spy-smoke
#SBATCH --account=msml603-class
#SBATCH --time=00:30:00
#SBATCH --output=local-training/hpc/bundles/smoke_%j.out
#SBATCH --error=local-training/hpc/bundles/smoke_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# SINGLE-NODE smoke: clean self-play PPO on one GPU node (no satellites) to prove
# the run is healthy + measure single-node throughput (eps/s). The engine sim is
# CPU-bound, so request many CPUs and run game runners on the head node itself.
set -euo pipefail

TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then echo "FATAL: no tarball" >&2; exit 1; fi
export MAGE_RL_RUNTIME_TARBALL="$TARBALL"

export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"
export TOTAL_EPISODES=200000
export TRAIN_PROFILES=1
export GAME_LOG_FREQUENCY=0
export OPPONENT_SAMPLER=self
export GPU_SERVICE_SPLIT_ROLES=0

# single-node: run all game runners here, NO satellites
export AUTO_SATELLITES=0
export WRITE_SATELLITE_ENV=0
export PRIMARY_RUNNER_CAP=48
export RUNNER_OVERSUBSCRIPTION_FACTOR=1

export PY_BATCH_MAX_SIZE=1024
export PY_BATCH_TIMEOUT_MS=25

# size S (matches baseline arch) for the throughput proof
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2

# CLEAN raw PPO: no SEARCH_OP / candidate_q / de-myopia / MCTS
export SEARCH_OP_ENABLE=0
export MCTS_TRAINING_ENABLE=0

echo "smoke starting on $(hostname) job=$SLURM_JOB_ID at $(date -u) cpus=$SLURM_CPUS_PER_TASK"
echo "TARBALL=$TARBALL"
exec bash scripts/hpc/run_spy_pbt.sh
