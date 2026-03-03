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

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  local init
  for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    if [[ -r "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init" >/dev/null 2>&1 || true
      if command -v module >/dev/null 2>&1; then
        return 0
      fi
    fi
  done
  return 1
}

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting spy PBT job"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "REPO_ROOT=$REPO_ROOT"

native_orch_raw="${HPC_NATIVE_ORCH:-1}"
native_orch_norm="$(echo "$native_orch_raw" | tr '[:upper:]' '[:lower:]')"
use_native_orch=0
if [[ "$native_orch_norm" == "1" || "$native_orch_norm" == "true" || "$native_orch_norm" == "yes" ]]; then
  use_native_orch=1
fi
export HPC_NATIVE_ORCH="${HPC_NATIVE_ORCH:-1}"

if [[ "$use_native_orch" -eq 1 ]]; then
  echo "Orchestrator mode: native (python)"
else
  if ! command -v pwsh >/dev/null 2>&1 && ! command -v powershell >/dev/null 2>&1; then
    if ensure_module_cmd; then
      module load powershell >/dev/null 2>&1 || module load pwsh >/dev/null 2>&1 || true
    fi
  fi

  if command -v pwsh >/dev/null 2>&1; then
    echo "PowerShell executable: $(command -v pwsh)"
  elif command -v powershell >/dev/null 2>&1; then
    echo "PowerShell executable: $(command -v powershell)"
  else
    echo "PowerShell unavailable on compute node; expected 'pwsh' or 'powershell' on PATH." >&2
    echo "Tip: set HPC_NATIVE_ORCH=1 (default) or verify module name on login node with: module avail powershell" >&2
    exit 1
  fi
fi

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
