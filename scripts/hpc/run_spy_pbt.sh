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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
resolve_path() {
  local candidate="${1:-}"
  local fallback="${2:-}"
  if [[ -z "$candidate" ]]; then
    candidate="$fallback"
  fi
  if [[ "$candidate" == /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "$repo_root/$candidate"
  fi
}
registry_path="${REGISTRY_PATH:-Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
total_episodes="${TOTAL_EPISODES:-1000000}"
pbt_interval="${PBT_EXPLOIT_INTERVAL_MINUTES:-240}"
pbt_first_exploit_min_ep="${PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT:-6000}"
pbt_episode_delta="${PBT_MIN_EPISODE_DELTA_PER_PROFILE:-3000}"
pbt_time_fallback_episode_delta="${PBT_TIME_FALLBACK_MIN_EPISODE_DELTA:-1000}"
pbt_mutation_pct="${PBT_MUTATION_PCT:-0.20}"
pbt_min_population="${PBT_MIN_POPULATION_SIZE:-3}"
pbt_min_winner_gap="${PBT_MIN_WINNER_GAP:-0.02}"
pbt_min_winner_wr="${PBT_MIN_WINNER_WINRATE:-0.03}"
eval_every_minutes="${EVAL_EVERY_MINUTES:-180}"
stall_restart_minutes="${STALL_RESTART_MINUTES:-25}"
game_log_frequency="${GAME_LOG_FREQUENCY:-500}"
cpu_headroom="${CPU_HEADROOM:-4}"
train_profiles="${TRAIN_PROFILES:-3}"
runner_oversubscription_factor="${RUNNER_OVERSUBSCRIPTION_FACTOR:-1}"
job_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
job_reports_root="$(resolve_path "${HPC_JOB_REPORTS_ROOT:-}" "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs")"
orch_compat_reports_root="$(resolve_path "${ORCHESTRATOR_COMPAT_REPORTS_ROOT:-}" "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator")"
export HPC_JOB_REPORTS_ROOT="$job_reports_root"
export ORCHESTRATOR_COMPAT_REPORTS_ROOT="$orch_compat_reports_root"
reports_dir="$job_reports_root/$job_id"
orch_reports_dir="$orch_compat_reports_root/runs/$job_id"
rl_artifacts_root="$orch_reports_dir/rl_artifacts"
mkdir -p "$reports_dir"
mkdir -p "$rl_artifacts_root"
orchestrator_log="$reports_dir/orchestrator.log"
telemetry_log="$reports_dir/telemetry.log"
final_probe_metrics_log="$reports_dir/final_probe_metrics.txt"
metrics_collector_script="$repo_root/scripts/hpc/collect_job_metrics_local.sh"
export ORCH_RUN_ID="${ORCH_RUN_ID:-$job_id}"
export RL_ARTIFACTS_ROOT="${RL_ARTIFACTS_ROOT:-$rl_artifacts_root}"
mkdir -p "$RL_ARTIFACTS_ROOT"

export PY_BRIDGE_CREATE_VENV="${PY_BRIDGE_CREATE_VENV:-0}"
export PY_BRIDGE_INSTALL_DEPS="${PY_BRIDGE_INSTALL_DEPS:-0}"
export MAGE_DB_AUTO_SERVER="false"
export REGISTRY_PATH="$registry_path"
export TOTAL_EPISODES="$total_episodes"
export PBT_EXPLOIT_INTERVAL_MINUTES="$pbt_interval"
export PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT="$pbt_first_exploit_min_ep"
export PBT_MIN_EPISODE_DELTA_PER_PROFILE="$pbt_episode_delta"
export PBT_TIME_FALLBACK_MIN_EPISODE_DELTA="$pbt_time_fallback_episode_delta"
export PBT_MUTATION_PCT="$pbt_mutation_pct"
export PBT_MIN_POPULATION_SIZE="$pbt_min_population"
export PBT_MIN_WINNER_GAP="$pbt_min_winner_gap"
export PBT_MIN_WINNER_WINRATE="$pbt_min_winner_wr"
export EVAL_EVERY_MINUTES="$eval_every_minutes"
export STALL_RESTART_MINUTES="$stall_restart_minutes"
export GAME_LOG_FREQUENCY="$game_log_frequency"
export TRAIN_PROFILES="$train_profiles"
export CPU_HEADROOM="$cpu_headroom"
# Avoid cross-job localhost Py4J port collisions when multiple jobs land on the same node.
# Caller can still override by setting PY4J_BASE_PORT explicitly.
job_num="${SLURM_JOB_ID:-0}"
if [[ -z "${PY4J_BASE_PORT:-}" ]]; then
  if [[ "$job_num" =~ ^[0-9]+$ ]]; then
    export PY4J_BASE_PORT="$((25000 + (job_num % 30000)))"
  else
    export PY4J_BASE_PORT="25334"
  fi
fi
if [[ "$job_num" =~ ^[0-9]+$ ]]; then
  port_slot="$((job_num % 20))"
else
  port_slot=0
fi
if [[ -z "${METRICS_PORT_BASE:-}" ]]; then
  export METRICS_PORT_BASE="$((19100 + (port_slot * 40)))"
fi
if [[ -z "${GPU_SERVICE_PORT_BASE:-}" ]]; then
  export GPU_SERVICE_PORT_BASE="$((26100 + (port_slot * 40)))"
fi
if [[ -z "${GPU_SERVICE_METRICS_PORT_BASE:-}" ]]; then
  export GPU_SERVICE_METRICS_PORT_BASE="$((27100 + (port_slot * 40)))"
fi
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
target_total_runners="$(awk -v cpu="$cpu_total" -v head="$cpu_headroom" -v factor="$runner_oversubscription_factor" '
BEGIN {
  usable = cpu - head
  if (usable < 0) usable = 0
  total = int(usable * factor)
  if (total < 0) total = 0
  print total
}')"
if (( target_total_runners < train_profiles * 2 )); then
  target_total_runners=$(( train_profiles * 2 ))
fi
runners_per_profile=$(( target_total_runners / train_profiles ))
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
write_final_metrics_snapshot() {
  if [[ ! -f "$metrics_collector_script" ]]; then
    return 0
  fi
  {
    echo "source=final_snapshot"
    bash "$metrics_collector_script" \
      --status-path "$orch_reports_dir/orchestrator_status.json" \
      --orchestrator-log "$orchestrator_log"
  } >"$final_probe_metrics_log" 2>&1 || true
  if [[ -s "$final_probe_metrics_log" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wrote final metrics snapshot to $final_probe_metrics_log" | tee -a "$orchestrator_log"
  else
    rm -f "$final_probe_metrics_log" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cleanup start (exit_code=${exit_code})" | tee -a "$orchestrator_log"
  write_final_metrics_snapshot
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
  prev_cpu_total=0
  prev_cpu_idle=0
  while true; do
    {
      echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      echo "host=$(hostname -f)"
      echo "cpu_total=$cpu_total cpu_headroom=$cpu_headroom runner_oversubscription_factor=$runner_oversubscription_factor target_total_runners=$target_total_runners runners_per_profile=$runners_per_profile"
      if [[ -r /proc/stat ]]; then
        cpu_line="$(grep '^cpu ' /proc/stat || true)"
        if [[ -n "$cpu_line" ]]; then
          read -r _ cpu_user cpu_nice cpu_system cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal _ <<<"$cpu_line"
          cpu_idle_all=$((cpu_idle + cpu_iowait))
          cpu_non_idle=$((cpu_user + cpu_nice + cpu_system + cpu_irq + cpu_softirq + cpu_steal))
          cpu_total_now=$((cpu_idle_all + cpu_non_idle))
          if (( prev_cpu_total > 0 && cpu_total_now > prev_cpu_total )); then
            total_delta=$((cpu_total_now - prev_cpu_total))
            idle_delta=$((cpu_idle_all - prev_cpu_idle))
            busy_delta=$((total_delta - idle_delta))
            cpu_usage_pct="$(awk -v busy="$busy_delta" -v total="$total_delta" 'BEGIN { if (total <= 0) printf "0.0"; else printf "%.1f", (100.0 * busy) / total }')"
            echo "cpu_usage_pct=$cpu_usage_pct"
          else
            echo "cpu_usage_pct=warmup"
          fi
          prev_cpu_total=$cpu_total_now
          prev_cpu_idle=$cpu_idle_all
        fi
      fi
      if [[ -r /proc/loadavg ]]; then
        read -r load1 load5 load15 task_counts _ < /proc/loadavg || true
        if [[ -n "${load1:-}" && -n "${task_counts:-}" ]]; then
          tasks_running="${task_counts%%/*}"
          tasks_total="${task_counts##*/}"
          echo "load1=$load1 load5=$load5 load15=$load15 tasks_running=$tasks_running tasks_total=$tasks_total"
        fi
      fi
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
echo "Runner sizing: cpu_total=$cpu_total cpu_headroom=$cpu_headroom runner_oversubscription_factor=$runner_oversubscription_factor target_total_runners=$target_total_runners" | tee -a "$orchestrator_log"
echo "NumGameRunners (per profile): $runners_per_profile" | tee -a "$orchestrator_log"
echo "Bridge retries: PY_BRIDGE_CONNECT_RETRIES=${PY_BRIDGE_CONNECT_RETRIES:-unset} PY_BRIDGE_CONNECT_RETRY_DELAY_MS=${PY_BRIDGE_CONNECT_RETRY_DELAY_MS:-unset}" | tee -a "$orchestrator_log"
echo "Run ID: $ORCH_RUN_ID" | tee -a "$orchestrator_log"
if [[ -n "${SATURATION_EXPERIMENT_LABEL:-}" ]]; then
  echo "Saturation label: ${SATURATION_EXPERIMENT_LABEL}" | tee -a "$orchestrator_log"
fi
echo "PBT reports dir: $orch_reports_dir" | tee -a "$orchestrator_log"
echo "RL artifacts root: $RL_ARTIFACTS_ROOT" | tee -a "$orchestrator_log"
echo "PBT gating: firstMinEp=$pbt_first_exploit_min_ep deltaPerProfile=$pbt_episode_delta timeFallbackDelta=$pbt_time_fallback_episode_delta maxIntervalMin=$pbt_interval minGap=$pbt_min_winner_gap minWinnerWr=$pbt_min_winner_wr" | tee -a "$orchestrator_log"
echo "MAGE_DB_DIR: $MAGE_DB_DIR" | tee -a "$orchestrator_log"
echo "Port bases: metrics=$METRICS_PORT_BASE gpuService=$GPU_SERVICE_PORT_BASE gpuServiceMetrics=$GPU_SERVICE_METRICS_PORT_BASE py4j=$PY4J_BASE_PORT" | tee -a "$orchestrator_log"
echo "MTG_VENV_PATH: $MTG_VENV_PATH" | tee -a "$orchestrator_log"
if [[ -z "${MAGE_RL_RUNTIME_DIR:-}" && -n "${MAGE_RL_RUNTIME_TARBALL:-}" ]]; then
  runtime_extract_base="/tmp/${USER}/mtgrl-runtime-${job_id}/runtime"
  mkdir -p "$runtime_extract_base"
  echo "Extracting runtime tarball: $MAGE_RL_RUNTIME_TARBALL" | tee -a "$orchestrator_log"
  tar -xzf "$MAGE_RL_RUNTIME_TARBALL" -C "$runtime_extract_base"
  extracted_dir="$(find "$runtime_extract_base" -mindepth 1 -maxdepth 1 -type d | head -1)"
  if [[ -z "$extracted_dir" ]]; then
    echo "FATAL: tarball extraction produced no directory in $runtime_extract_base" >&2
    exit 1
  fi
  export MAGE_RL_RUNTIME_DIR="$extracted_dir"
  echo "MAGE_RL_RUNTIME_DIR: $MAGE_RL_RUNTIME_DIR (extracted from tarball)" | tee -a "$orchestrator_log"
elif [[ -n "${MAGE_RL_RUNTIME_DIR:-}" ]]; then
  echo "MAGE_RL_RUNTIME_DIR: $MAGE_RL_RUNTIME_DIR" | tee -a "$orchestrator_log"
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
  echo "Orchestrator mode: powershell (legacy HPC fallback)" | tee -a "$orchestrator_log"
  ps_command="& '$repo_root/scripts/rl-league-run.ps1' \
    -RegistryPath '$registry_path' \
    -SequentialTraining \$false \
    -EnablePbt \$true \
    -PbtExploitIntervalMinutes $pbt_interval \
    -PbtMinEpisodesBeforeFirstExploit $pbt_first_exploit_min_ep \
    -PbtMinEpisodeDeltaPerProfile $pbt_episode_delta \
    -PbtTimeFallbackMinEpisodeDelta $pbt_time_fallback_episode_delta \
    -PbtMutationPct $pbt_mutation_pct \
    -PbtMinPopulationSize $pbt_min_population \
    -PbtMinWinnerGap $pbt_min_winner_gap \
    -PbtMinWinnerWinrate $pbt_min_winner_wr \
    -GameLogFrequency $game_log_frequency \
    -StallRestartMinutes $stall_restart_minutes \
    -EvalEveryMinutes $eval_every_minutes \
    -NoEval \
    -NumGameRunners $runners_per_profile \
    -TotalEpisodes $total_episodes"

  "$shell_exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ps_command" 2>&1 | tee -a "$orchestrator_log"
fi
