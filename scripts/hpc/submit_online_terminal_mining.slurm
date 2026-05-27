#!/usr/bin/env bash
#SBATCH --job-name=online-mine
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=04:00:00
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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting online terminal mining"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "REPO_ROOT=$REPO_ROOT"

if ! command -v mvn >/dev/null 2>&1; then
  if ensure_module_cmd; then
    module load maven >/dev/null 2>&1 || \
    module load apache-maven >/dev/null 2>&1 || \
    load_first_module_match maven || true
  fi
fi

if command -v mvn >/dev/null 2>&1; then
  echo "Maven executable: $(command -v mvn)"
else
  echo "Maven unavailable on compute node; expected 'mvn' on PATH." >&2
  exit 1
fi

export MTG_VENV_PATH="${MTG_VENV_PATH:-$REPO_ROOT/.mtgrl_venv_hpc}"
export PY_BRIDGE_CREATE_VENV=0
export PY_BRIDGE_INSTALL_DEPS=0
export MAGE_DB_AUTO_SERVER=false

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" && -x "$MTG_VENV_PATH/bin/python" ]]; then
  python_bin="$MTG_VENV_PATH/bin/python"
fi
if [[ -z "$python_bin" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "$python_bin" ]]; then
  echo "No Python executable found; set PYTHON_BIN or create $MTG_VENV_PATH." >&2
  exit 1
fi
echo "Python executable: $python_bin"

run_id="${ONLINE_RUN_ID:-hpc_${SLURM_JOB_ID:-manual}_online_terminal_mining}"
output_root="${ONLINE_OUTPUT_ROOT:-local-training/local_pbt/online_terminal_mining_hpc}"

cmd=(
  "$python_bin" scripts/mtgrl/run_online_terminal_mining_loop.py
  --run-id "$run_id"
  --output-root "$output_root"
  --cycles "${ONLINE_CYCLES:-1}"
  --profiles "${ONLINE_PROFILES:-Pauper-Spy-Combo-Value}"
  --opponents "${ONLINE_OPPONENTS:-grixis}"
  --games-per-matchup "${ONLINE_GAMES_PER_MATCHUP:-8}"
  --games-per-job "${ONLINE_GAMES_PER_JOB:-1}"
  --skill "${ONLINE_SKILL:-7}"
  --eval-parallel "${ONLINE_EVAL_PARALLEL:-1}"
  --serial-warmup-jobs "${ONLINE_SERIAL_WARMUP_JOBS:-1}"
  --ai-threads "${ONLINE_AI_THREADS:-1}"
  --eval-timeout-sec "${ONLINE_EVAL_TIMEOUT_SEC:-1800}"
  --live-checkpoint-max-per-game "${ONLINE_LIVE_CHECKPOINT_MAX_PER_GAME:-96}"
  --max-snapshots "${ONLINE_MAX_SNAPSHOTS:-128}"
  --selection-mode "${ONLINE_SELECTION_MODE:-ranked}"
  --ranked-max-per-game "${ONLINE_RANKED_MAX_PER_GAME:-32}"
  --action-types "${ONLINE_ACTION_TYPES:-ACTIVATE_ABILITY_OR_SPELL}"
  --mine-shards "${ONLINE_MINE_SHARDS:-8}"
  --line-training-max-records-per-branch "${ONLINE_LINE_TRAINING_MAX_RECORDS_PER_BRANCH:-64}"
  --line-attempts "${ONLINE_LINE_ATTEMPTS:-32}"
  --line-max-root-actions "${ONLINE_LINE_MAX_ROOT_ACTIONS:-16}"
  --line-timeout-sec "${ONLINE_LINE_TIMEOUT_SEC:-120}"
  --tree-timeout-sec "${ONLINE_TREE_TIMEOUT_SEC:-120}"
  --tree-continuation-policy "${ONLINE_TREE_CONTINUATION_POLICY:-explore}"
  --eval-gpu-port "${ONLINE_EVAL_GPU_PORT:-26100}"
  --eval-gpu-metrics-port "${ONLINE_EVAL_GPU_METRICS_PORT:-27100}"
  --mine-gpu-port "${ONLINE_MINE_GPU_PORT:-26384}"
  --mine-gpu-metrics-port "${ONLINE_MINE_GPU_METRICS_PORT:-27384}"
)

if [[ -n "${ONLINE_CHUNK_INDICES:-}" ]]; then
  cmd+=(--chunk-indices "$ONLINE_CHUNK_INDICES")
fi
if [[ "${ONLINE_LINE_CAPTURE_TRAINING_DATA:-0}" == "1" || "${ONLINE_LINE_CAPTURE_TRAINING_DATA:-}" == "true" ]]; then
  cmd+=(--line-capture-training-data)
fi
if [[ -n "${ONLINE_MAX_CONCURRENT_MINE_SHARDS:-}" ]]; then
  cmd+=(--max-concurrent-mine-shards "$ONLINE_MAX_CONCURRENT_MINE_SHARDS")
fi
if [[ "${ONLINE_FORCE:-1}" == "1" || "${ONLINE_FORCE:-}" == "true" ]]; then
  cmd+=(--force)
fi
if [[ "${ONLINE_MAVEN_ONLINE:-0}" == "1" || "${ONLINE_MAVEN_ONLINE:-}" == "true" ]]; then
  cmd+=(--online-maven)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
