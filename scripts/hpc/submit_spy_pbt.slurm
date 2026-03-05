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

load_first_module_match() {
  local query="$1"
  if ! ensure_module_cmd; then
    return 1
  fi
  local mod_name=""
  mod_name="$(module -t avail "$query" 2>&1 | awk 'NF && $1 !~ /^-+$/ { print $1; exit }')"
  if [[ -z "$mod_name" ]]; then
    return 1
  fi
  module load "$mod_name" >/dev/null 2>&1
}

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting spy PBT job"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "REPO_ROOT=$REPO_ROOT"

if [[ -n "${MAGE_RL_RUNTIME_TARBALL:-}" ]]; then
  if [[ ! -f "${MAGE_RL_RUNTIME_TARBALL}" ]]; then
    echo "Artifact tarball not found: ${MAGE_RL_RUNTIME_TARBALL}" >&2
    exit 1
  fi
  runtime_stage_root="${SLURM_TMPDIR:-/tmp/${USER}/mtgrl-runtime-${SLURM_JOB_ID}}"
  runtime_extract_dir="${runtime_stage_root}/runtime"
  rm -rf "$runtime_extract_dir"
  mkdir -p "$runtime_extract_dir"
  top_entry="$(tar -tzf "$MAGE_RL_RUNTIME_TARBALL" | head -n 1 || true)"
  top_dir="${top_entry%%/*}"
  if [[ -z "$top_dir" ]]; then
    echo "Could not detect top-level directory in tarball: ${MAGE_RL_RUNTIME_TARBALL}" >&2
    exit 1
  fi
  tar -xzf "$MAGE_RL_RUNTIME_TARBALL" -C "$runtime_extract_dir"
  export MAGE_RL_RUNTIME_DIR="${runtime_extract_dir}/${top_dir}"
fi

if [[ -n "${MAGE_RL_RUNTIME_DIR:-}" ]]; then
  echo "Artifact runtime dir: ${MAGE_RL_RUNTIME_DIR}"
elif ! command -v mvn >/dev/null 2>&1; then
  if ensure_module_cmd; then
    module load maven >/dev/null 2>&1 || \
    module load apache-maven >/dev/null 2>&1 || \
    load_first_module_match maven || true
  fi
fi

if [[ -n "${MAGE_RL_RUNTIME_DIR:-}" ]]; then
  :
elif command -v mvn >/dev/null 2>&1; then
  echo "Maven executable: $(command -v mvn)"
else
  echo "Maven unavailable on compute node; expected 'mvn' on PATH." >&2
  echo "Tip: verify module name on login node with: module avail maven, or provide MAGE_RL_RUNTIME_TARBALL." >&2
  exit 1
fi

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
export TRAINER_START_STAGGER_SECONDS="${TRAINER_START_STAGGER_SECONDS:-20}"
export PY_BRIDGE_CONNECT_RETRIES="${PY_BRIDGE_CONNECT_RETRIES:-45}"
export PY_BRIDGE_CONNECT_RETRY_DELAY_MS="${PY_BRIDGE_CONNECT_RETRY_DELAY_MS:-2000}"
export PY_BACKEND_MODE="${PY_BACKEND_MODE:-single}"
export PY_BATCH_TIMEOUT_MS="${PY_BATCH_TIMEOUT_MS:-10}"
if [[ -z "${INFER_WORKERS:-}" ]]; then
  tp="${TRAIN_PROFILES:-3}"
  if [[ "$tp" =~ ^[0-9]+$ ]] && (( tp > 1 )); then
    export INFER_WORKERS="0"
  else
    export INFER_WORKERS="1"
  fi
else
  export INFER_WORKERS
fi

bash "$REPO_ROOT/scripts/hpc/run_spy_pbt.sh"
