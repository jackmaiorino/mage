#!/usr/bin/env bash
#SBATCH --job-name=spy-pbt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
## Optional site-specific settings (uncomment and set):
## #SBATCH --account=<account>
## #SBATCH --qos=<qos>

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting spy PBT job"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "REPO_ROOT=$REPO_ROOT"

export MTG_VENV_PATH="${MTG_VENV_PATH:-$REPO_ROOT/.mtgrl_venv_hpc}"
export PY_BRIDGE_CREATE_VENV=0
export PY_BRIDGE_INSTALL_DEPS=0
export MAGE_DB_AUTO_SERVER=false

# Safe defaults; override via sbatch --export or environment.
export REGISTRY_PATH="${REGISTRY_PATH:-Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
export TOTAL_EPISODES="${TOTAL_EPISODES:-1000000}"
export PBT_EXPLOIT_INTERVAL_MINUTES="${PBT_EXPLOIT_INTERVAL_MINUTES:-240}"
export PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT="${PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT:-12000}"
export PBT_MIN_EPISODE_DELTA_PER_PROFILE="${PBT_MIN_EPISODE_DELTA_PER_PROFILE:-10000}"
export PBT_MUTATION_PCT="${PBT_MUTATION_PCT:-0.20}"
export PBT_MIN_POPULATION_SIZE="${PBT_MIN_POPULATION_SIZE:-3}"
export EVAL_EVERY_MINUTES="${EVAL_EVERY_MINUTES:-180}"
export STALL_RESTART_MINUTES="${STALL_RESTART_MINUTES:-25}"
export GAME_LOG_FREQUENCY="${GAME_LOG_FREQUENCY:-500}"
export TRAIN_PROFILES="${TRAIN_PROFILES:-3}"
export CPU_HEADROOM="${CPU_HEADROOM:-4}"
export INFER_WORKERS="${INFER_WORKERS:-1}"

bash "$REPO_ROOT/scripts/hpc/run_spy_pbt.sh"
