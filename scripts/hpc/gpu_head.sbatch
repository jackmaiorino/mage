#!/bin/bash
#SBATCH --job-name=gpu-head
#SBATCH --account=msml603-class
#SBATCH --time=03:00:00
#SBATCH --output=local-training/hpc/bundles/gpu-head_%j.out
#SBATCH --error=local-training/hpc/bundles/gpu-head_%j.err
#
# GPU head node: runs the orchestrator + shared GPU inference service.
# No het-groups — submit CPU satellites separately via add_satellites.sh.
#
# Default: 1x H100, minimal CPUs (satellites do game simulation).
# Override partition/gres at submission time:
#   sbatch --partition=gpu-a100 --gres=gpu:a100:1 gpu_head.sbatch <tarball>
#
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

TARBALL="${1:-${MAGE_RL_RUNTIME_TARBALL:-}}"
if [[ -z "$TARBALL" ]]; then
    TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
fi
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then
    echo "FATAL: No runtime tarball found. Pass as arg or set MAGE_RL_RUNTIME_TARBALL." >&2
    exit 1
fi

export MAGE_RL_RUNTIME_TARBALL="$TARBALL"
export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"
export TOTAL_EPISODES="${TOTAL_EPISODES:-10000000}"
export TRAIN_PROFILES="${TRAIN_PROFILES:-3}"
export GAME_LOG_FREQUENCY="${GAME_LOG_FREQUENCY:-0}"
export OPPONENT_SAMPLER="${OPPONENT_SAMPLER:-self}"
export WINRATE_WINDOW="${WINRATE_WINDOW:-200}"

# With 1 GPU, no split roles needed. Enable if using 2+ GPUs.
export GPU_SERVICE_SPLIT_ROLES="${GPU_SERVICE_SPLIT_ROLES:-0}"

# GPU head runs minimal local runners — satellites provide the bulk.
export PRIMARY_RUNNER_CAP="${PRIMARY_RUNNER_CAP:-50}"
export RUNNER_OVERSUBSCRIPTION_FACTOR="${RUNNER_OVERSUBSCRIPTION_FACTOR:-1}"

# Higher batch sizes for saturation throughput.
export PY_BATCH_MAX_SIZE="${PY_BATCH_MAX_SIZE:-1024}"
export PY_BATCH_TIMEOUT_MS="${PY_BATCH_TIMEOUT_MS:-25}"

# Write satellite env file so add_satellites.sh can discover this job.
export WRITE_SATELLITE_ENV=1

echo "GPU head starting on $(hostname) job=$SLURM_JOB_ID at $(date -u)"
echo "TARBALL=$TARBALL"
exec bash scripts/hpc/run_spy_pbt.sh
