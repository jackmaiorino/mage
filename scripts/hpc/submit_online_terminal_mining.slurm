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

prepend_maven_path_if_present() {
  local mvn_path="$1"
  if [[ -n "$mvn_path" && -x "$mvn_path" ]]; then
    export PATH="$(dirname "$mvn_path"):$PATH"
    return 0
  fi
  return 1
}

ensure_maven_cmd() {
  if command -v mvn >/dev/null 2>&1; then
    return 0
  fi
  if prepend_maven_path_if_present "${MAVEN_BIN:-}"; then
    return 0
  fi
  if [[ -n "${MAVEN_HOME:-}" ]] && prepend_maven_path_if_present "$MAVEN_HOME/bin/mvn"; then
    return 0
  fi
  local candidate
  for candidate in \
    "$REPO_ROOT/local-training/hpc/tools/apache-maven/bin/mvn" \
    "$REPO_ROOT/local-training/hpc/tools/apache-maven-3.9.8/bin/mvn" \
    "$HOME/.local/apache-maven/bin/mvn" \
    "$HOME/.local/apache-maven-3.9.8/bin/mvn"; do
    if prepend_maven_path_if_present "$candidate"; then
      return 0
    fi
  done
  if ensure_module_cmd; then
    module load maven >/dev/null 2>&1 || \
    module load apache-maven >/dev/null 2>&1 || \
    load_first_module_match maven || true
  fi
  command -v mvn >/dev/null 2>&1
}

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting online terminal mining"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "REPO_ROOT=$REPO_ROOT"
if ensure_maven_cmd; then
  echo "Maven executable: $(command -v mvn)"
else
  echo "Maven unavailable on compute node; set MAVEN_BIN, MAVEN_HOME, or install Maven under local-training/hpc/tools/." >&2
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

preflight_cmd=(
  "$python_bin" scripts/mtgrl/preflight_online_terminal_mining.py
  --registry "${ONLINE_REGISTRY:-Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
  --profiles "${ONLINE_PROFILES:-Pauper-Spy-Combo-Value}"
  --opponents "${ONLINE_OPPONENTS:-grixis}"
  --require-maven
  --check-gpu
  --output-json "$output_root/$run_id/preflight/compute_preflight.json"
)
if [[ "${ONLINE_PREFLIGHT_MAVEN_COMPILE:-0}" == "1" || "${ONLINE_PREFLIGHT_MAVEN_COMPILE:-}" == "true" ]]; then
  preflight_cmd+=(--check-maven-compile)
fi
if [[ -n "${ONLINE_PREFLIGHT_MAVEN_TIMEOUT_SEC:-}" ]]; then
  preflight_cmd+=(--maven-compile-timeout-sec "$ONLINE_PREFLIGHT_MAVEN_TIMEOUT_SEC")
fi
if [[ "${ONLINE_PREFLIGHT_MAVEN_ONLINE:-0}" == "1" || "${ONLINE_PREFLIGHT_MAVEN_ONLINE:-}" == "true" ||
      "${ONLINE_MAVEN_ONLINE:-0}" == "1" || "${ONLINE_MAVEN_ONLINE:-}" == "true" ]]; then
  preflight_cmd+=(--maven-online)
fi
if [[ "${ONLINE_PREFLIGHT_REACTOR_MODULES:-1}" == "1" || "${ONLINE_PREFLIGHT_REACTOR_MODULES:-}" == "true" ]]; then
  preflight_cmd+=(--check-reactor-modules)
fi

if [[ "${ONLINE_PREFLIGHT_ONLY:-0}" == "1" || "${ONLINE_PREFLIGHT_ONLY:-}" == "true" ]]; then
  printf 'Preflight command:'
  printf ' %q' "${preflight_cmd[@]}"
  printf '\n'
  "${preflight_cmd[@]}"
  exit $?
fi

cmd=(
  "$python_bin" scripts/mtgrl/run_online_terminal_mining_loop.py
  --run-id "$run_id"
  --output-root "$output_root"
  --cycles "${ONLINE_CYCLES:-1}"
  --registry "${ONLINE_REGISTRY:-Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
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
if [[ "${ONLINE_NO_COMPILE_EXEC:-0}" == "1" || "${ONLINE_NO_COMPILE_EXEC:-}" == "true" ]]; then
  cmd+=(--no-compile-exec)
fi
if [[ "${ONLINE_SKIP_EVAL_COMPILE:-0}" == "1" || "${ONLINE_SKIP_EVAL_COMPILE:-}" == "true" ]]; then
  cmd+=(--skip-eval-compile)
fi
if [[ "${ONLINE_SKIP_MINE_COMPILE:-0}" == "1" || "${ONLINE_SKIP_MINE_COMPILE:-}" == "true" ]]; then
  cmd+=(--skip-mine-compile)
fi
if [[ "${ONLINE_SKIP_SHARD_COMPILE:-0}" == "1" || "${ONLINE_SKIP_SHARD_COMPILE:-}" == "true" ]]; then
  cmd+=(--skip-shard-compile)
fi
if [[ "${ONLINE_ALLOW_STALE_MINER_CLASSPATH:-0}" == "1" || "${ONLINE_ALLOW_STALE_MINER_CLASSPATH:-}" == "true" ]]; then
  cmd+=(--allow-stale-miner-classpath)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
