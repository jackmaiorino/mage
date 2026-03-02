#!/usr/bin/env bash
set -euo pipefail

quiet=0
if [[ "${1:-}" == "-q" || "${1:-}" == "--quiet" ]]; then
  quiet=1
  shift || true
fi

log() {
  if [[ "$quiet" -eq 0 ]]; then
    echo "$*"
  fi
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_name="${USER:-$(id -un)}"
slurm_job_id="${SLURM_JOB_ID:-}"

declare -A candidate_pids=()
declare -A scope_pids=()

if [[ -n "$slurm_job_id" ]] && command -v scontrol >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    scope_pids["$pid"]=1
  done < <(scontrol listpids "$slurm_job_id" 2>/dev/null | awk 'NR > 1 && $1 ~ /^[0-9]+$/ { print $1 }')
fi

pid_in_scope() {
  local pid="$1"
  if [[ "${#scope_pids[@]}" -eq 0 ]]; then
    return 0
  fi
  [[ -n "${scope_pids[$pid]:-}" ]]
}

collect_candidates() {
  local pattern="$1"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    pid_in_scope "$pid" || continue
    local cmd
    cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    [[ -n "$cmd" ]] || continue
    if [[ "$cmd" == *"$repo_root"* || "$cmd" == *"Mage.Server.Plugins/Mage.Player.AIRL"* || "$cmd" == *"py4j_entry_point.py"* || "$cmd" == *"RLTrainer"* ]]; then
      candidate_pids["$pid"]=1
    fi
  done < <(pgrep -u "$user_name" -f "$pattern" 2>/dev/null || true)
}

collect_candidates "py4j_entry_point.py"
collect_candidates "draft_py4j_entry_point.py"
collect_candidates "mage.player.ai.rl.RLTrainer"
collect_candidates "-Dexec.mainClass=mage.player.ai.rl.RLTrainer"
collect_candidates "mvn.*exec:java.*RLTrainer"

if command -v lsof >/dev/null 2>&1; then
  for port in {25334..25345} {26334..26345}; do
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      pid_in_scope "$pid" || continue
      local_cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
      if [[ "$local_cmd" == *"$repo_root"* || "$local_cmd" == *"Mage.Server.Plugins/Mage.Player.AIRL"* || "$local_cmd" == *"py4j_entry_point.py"* || "$local_cmd" == *"RLTrainer"* ]]; then
        candidate_pids["$pid"]=1
      fi
    done < <(lsof -t -i "TCP:${port}" -s TCP:LISTEN 2>/dev/null || true)
  done
fi

if [[ "${#candidate_pids[@]}" -eq 0 ]]; then
  log "rl-stop.sh: no matching RL processes found"
  exit 0
fi

for pid in "${!candidate_pids[@]}"; do
  cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  log "Stopping PID=${pid} cmd=${cmd}"
  kill -TERM "$pid" 2>/dev/null || true
done

deadline=$((SECONDS + 8))
while (( SECONDS < deadline )); do
  alive=0
  for pid in "${!candidate_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 0.2
done

for pid in "${!candidate_pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    log "Force killing PID=${pid} cmd=${cmd}"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

log "rl-stop.sh: done"
