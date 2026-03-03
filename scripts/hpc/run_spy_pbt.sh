#!/usr/bin/env bash
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

load_powershell_module() {
  if ! ensure_module_cmd; then
    return 1
  fi
  module load powershell >/dev/null 2>&1 && return 0
  module load pwsh >/dev/null 2>&1 && return 0

  local mod_name=""
  mod_name="$(module -t avail powershell 2>&1 | awk 'NF {print $1; exit}')"
  if [[ -n "$mod_name" ]]; then
    module load "$mod_name" >/dev/null 2>&1 && return 0
  fi
  mod_name="$(module -t avail pwsh 2>&1 | awk 'NF {print $1; exit}')"
  if [[ -n "$mod_name" ]]; then
    module load "$mod_name" >/dev/null 2>&1 && return 0
  fi
  return 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
registry_path="${REGISTRY_PATH:-Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
total_episodes="${TOTAL_EPISODES:-1000000}"
pbt_interval="${PBT_EXPLOIT_INTERVAL_MINUTES:-240}"
pbt_first_exploit_min_ep="${PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT:-12000}"
pbt_episode_delta="${PBT_MIN_EPISODE_DELTA_PER_PROFILE:-10000}"
pbt_mutation_pct="${PBT_MUTATION_PCT:-0.20}"
pbt_min_population="${PBT_MIN_POPULATION_SIZE:-3}"
eval_every_minutes="${EVAL_EVERY_MINUTES:-180}"
stall_restart_minutes="${STALL_RESTART_MINUTES:-25}"
game_log_frequency="${GAME_LOG_FREQUENCY:-500}"
cpu_headroom="${CPU_HEADROOM:-4}"
train_profiles="${TRAIN_PROFILES:-3}"
job_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
reports_dir="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs/$job_id"
mkdir -p "$reports_dir"
orchestrator_log="$reports_dir/orchestrator.log"
telemetry_log="$reports_dir/telemetry.log"

export PY_BRIDGE_CREATE_VENV="${PY_BRIDGE_CREATE_VENV:-0}"
export PY_BRIDGE_INSTALL_DEPS="${PY_BRIDGE_INSTALL_DEPS:-0}"
export MAGE_DB_AUTO_SERVER="false"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export MAGE_DB_DIR="$SLURM_TMPDIR/rl-db"
else
  export MAGE_DB_DIR="/tmp/${USER}/mtgrl-${job_id}/rl-db"
fi
mkdir -p "$MAGE_DB_DIR"

if [[ -z "${MTG_VENV_PATH:-}" ]]; then
  export MTG_VENV_PATH="$repo_root/.mtgrl_venv_hpc"
fi

cpu_total="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
usable_cpus=$(( cpu_total - cpu_headroom ))
if (( usable_cpus < train_profiles * 2 )); then
  usable_cpus=$(( train_profiles * 2 ))
fi
runners_per_profile=$(( usable_cpus / train_profiles ))
if (( runners_per_profile < 2 )); then
  runners_per_profile=2
fi

shell_exe=""
if command -v pwsh >/dev/null 2>&1; then
  shell_exe="pwsh"
elif command -v powershell >/dev/null 2>&1; then
  shell_exe="powershell"
else
  load_powershell_module || true
  if command -v pwsh >/dev/null 2>&1; then
    shell_exe="pwsh"
  elif command -v powershell >/dev/null 2>&1; then
    shell_exe="powershell"
  fi
fi

native_orch_raw="${HPC_NATIVE_ORCH:-1}"
native_orch_norm="$(echo "$native_orch_raw" | tr '[:upper:]' '[:lower:]')"
use_native_orch=0
if [[ "$native_orch_norm" == "1" || "$native_orch_norm" == "true" || "$native_orch_norm" == "yes" ]]; then
  use_native_orch=1
fi

if [[ "$use_native_orch" -eq 0 && -z "$shell_exe" ]]; then
  echo "PowerShell is required to run rl-league-run.ps1; expected 'pwsh' or 'powershell' on PATH." >&2
  exit 1
fi

telemetry_pid=""
cleanup() {
  local exit_code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cleanup start (exit_code=${exit_code})" | tee -a "$orchestrator_log"
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ -f "$repo_root/scripts/rl-stop.sh" ]]; then
    bash "$repo_root/scripts/rl-stop.sh" -q || true
  else
    "$shell_exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$repo_root/scripts/rl-stop.ps1" -Quiet || true
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cleanup done" | tee -a "$orchestrator_log"
}
trap cleanup EXIT INT TERM

(
  while true; do
    {
      echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      echo "host=$(hostname -f)"
      echo "cpu_total=$cpu_total runners_per_profile=$runners_per_profile"
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits || true
      else
        echo "nvidia-smi unavailable"
      fi
      if [[ -r /proc/meminfo ]]; then
        grep -E 'MemTotal|MemAvailable' /proc/meminfo || true
      fi
      echo
    } >>"$telemetry_log" 2>&1
    sleep "${TELEMETRY_INTERVAL_SEC:-30}"
  done
) &
telemetry_pid=$!

echo "Repo root: $repo_root" | tee -a "$orchestrator_log"
echo "Job ID: $job_id" | tee -a "$orchestrator_log"
echo "Registry: $registry_path" | tee -a "$orchestrator_log"
echo "NumGameRunners (per profile): $runners_per_profile" | tee -a "$orchestrator_log"
echo "PBT gating: firstMinEp=$pbt_first_exploit_min_ep deltaPerProfile=$pbt_episode_delta maxIntervalMin=$pbt_interval" | tee -a "$orchestrator_log"
echo "MAGE_DB_DIR: $MAGE_DB_DIR" | tee -a "$orchestrator_log"
echo "MTG_VENV_PATH: $MTG_VENV_PATH" | tee -a "$orchestrator_log"
if [[ -n "${MAGE_RL_RUNTIME_DIR:-}" ]]; then
  echo "MAGE_RL_RUNTIME_DIR: $MAGE_RL_RUNTIME_DIR" | tee -a "$orchestrator_log"
elif [[ -n "${MAGE_RL_RUNTIME_TARBALL:-}" ]]; then
  echo "MAGE_RL_RUNTIME_TARBALL: $MAGE_RL_RUNTIME_TARBALL" | tee -a "$orchestrator_log"
fi
echo "Reports dir: $reports_dir" | tee -a "$orchestrator_log"
if [[ "$use_native_orch" -eq 1 ]]; then
  echo "Orchestrator mode: native (python)" | tee -a "$orchestrator_log"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for native orchestrator mode" >&2
    exit 1
  fi
  python3 "$repo_root/scripts/hpc/run_spy_pbt_native.py" 2>&1 | tee -a "$orchestrator_log"
else
  echo "Orchestrator mode: powershell" | tee -a "$orchestrator_log"
  ps_command="& '$repo_root/scripts/rl-league-run.ps1' \
    -RegistryPath '$registry_path' \
    -SequentialTraining \$false \
    -EnablePbt \$true \
    -PbtExploitIntervalMinutes $pbt_interval \
    -PbtMinEpisodesBeforeFirstExploit $pbt_first_exploit_min_ep \
    -PbtMinEpisodeDeltaPerProfile $pbt_episode_delta \
    -PbtMutationPct $pbt_mutation_pct \
    -PbtMinPopulationSize $pbt_min_population \
    -GameLogFrequency $game_log_frequency \
    -StallRestartMinutes $stall_restart_minutes \
    -EvalEveryMinutes $eval_every_minutes \
    -NoEval \
    -NumGameRunners $runners_per_profile \
    -TotalEpisodes $total_episodes"

  "$shell_exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ps_command" 2>&1 | tee -a "$orchestrator_log"
fi
