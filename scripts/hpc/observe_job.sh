#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
job_id="${1:-}"
watch_mode=1

if [[ "${2:-}" == "--no-watch" ]]; then
  watch_mode=0
fi

if [[ -z "$job_id" ]]; then
  echo "Usage: bash scripts/hpc/observe_job.sh <job_id> [--no-watch]" >&2
  exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is required on PATH." >&2
  exit 1
fi

node="$(squeue -j "$job_id" -h -o %N 2>/dev/null | head -n1 | tr -d '[:space:]')"
if [[ -z "$node" || "$node" == "(null)" || "$node" == "None" ]]; then
  echo "Could not resolve node for job $job_id (is it still running?)." >&2
  exit 1
fi

status_path="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/orchestrator_status.json"
targets_path="$repo_root/monitoring/file_sd/mage_hpc_targets.json"
sync_log="$repo_root/monitoring/file_sd/target_sync.log"
sync_pid_file="$repo_root/monitoring/file_sd/target_sync.pid"

mkdir -p "$(dirname "$targets_path")"

# Seed targets once before starting the loop.
METRICS_HOST="$node" bash "$repo_root/scripts/hpc/refresh_prometheus_targets.sh" "$status_path" "$targets_path" >/dev/null

# Start or reuse background target sync loop.
if [[ -f "$sync_pid_file" ]]; then
  old_pid="$(cat "$sync_pid_file" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    :
  else
    rm -f "$sync_pid_file"
  fi
fi

if [[ ! -f "$sync_pid_file" ]]; then
  nohup env METRICS_HOST="$node" SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-5}" \
    bash "$repo_root/scripts/hpc/sync_prometheus_targets_loop.sh" "$status_path" "$targets_path" \
    >"$sync_log" 2>&1 &
  echo "$!" >"$sync_pid_file"
fi

# Start monitoring stack if docker compose is available.
if command -v docker >/dev/null 2>&1; then
  (cd "$repo_root" && docker compose -f docker-compose-observe-hpc.yml up -d >/dev/null)
else
  echo "docker not found; target sync started, but Prometheus/Grafana stack not started." >&2
fi

echo "job=$job_id node=$node"
echo "target_sync_pid=$(cat "$sync_pid_file")"
echo "prometheus=http://localhost:9091"
echo "grafana=http://localhost:3000"

if [[ "$watch_mode" -eq 0 ]]; then
  exit 0
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
  echo "curl/python3 required for --watch mode." >&2
  exit 0
fi

prom_url="http://127.0.0.1:9091"
query() {
  local q="$1"
  curl -fsS --get "$prom_url/api/v1/query" --data-urlencode "query=$q" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("data",{}).get("result",[]); print(r[0]["value"][1] if r else "0")'
}

echo
echo "Live metrics (Ctrl+C to stop):"
while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  eps="$(query 'sum(rate(mage_episodes_completed_total[1m]))')"
  ups="$(query 'sum(rate(mage_training_updates_total[1m]))')"
  active="$(query 'sum(mage_active_episodes)')"
  echo "[$ts] episodes/s=$eps updates/s=$ups active_episodes=$active"
  sleep 10
done
