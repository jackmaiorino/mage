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
docker_available=0

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
  docker_available=1
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

get_ports() {
  python3 - "$status_path" "$targets_path" <<'PY'
import json
import pathlib
import re
import sys

status_path = pathlib.Path(sys.argv[1])
targets_path = pathlib.Path(sys.argv[2])

ports = []
if status_path.exists():
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
        for row in payload.get("trainers", []):
            if not isinstance(row, dict):
                continue
            if not bool(row.get("running", False)):
                continue
            try:
                p = int(row.get("metrics_port", 0))
            except Exception:
                p = 0
            if p > 0:
                ports.append(p)
    except Exception:
        pass

if not ports and targets_path.exists():
    try:
        payload = json.loads(targets_path.read_text(encoding="utf-8-sig"))
        for row in payload:
            if not isinstance(row, dict):
                continue
            for target in row.get("targets", []):
                m = re.search(r":(\d+)$", str(target))
                if m:
                    ports.append(int(m.group(1)))
    except Exception:
        pass

ports = sorted(set(ports))
print(" ".join(str(p) for p in ports))
PY
}

sample_direct() {
  local ports="$1"
  if [[ -z "$ports" ]]; then
    return 0
  fi
  srun --jobid="$job_id" -N1 -n1 -w "$node" env PORTS="$ports" bash -lc '
for p in $PORTS; do
  m=$(curl -fsS "http://127.0.0.1:${p}/metrics" 2>/dev/null || true)
  c=$(printf "%s\n" "$m" | awk "/^mage_episodes_completed_total /{print \$2; exit}")
  u=$(printf "%s\n" "$m" | awk "/^mage_training_updates_total /{print \$2; exit}")
  a=$(printf "%s\n" "$m" | awk "/^mage_active_episodes /{print \$2; exit}")
  if [[ -z "$c" ]]; then c=0; fi
  if [[ -z "$u" ]]; then u=0; fi
  if [[ -z "$a" ]]; then a=0; fi
  echo "$p $c $u $a"
done
'
}

echo
echo "Live metrics (Ctrl+C to stop):"
if [[ "$docker_available" -eq 1 ]]; then
  prom_url="http://127.0.0.1:9091"
  query() {
    local q="$1"
    curl -fsS --get "$prom_url/api/v1/query" --data-urlencode "query=$q" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("data",{}).get("result",[]); print(r[0]["value"][1] if r else "0")' \
      || echo "0"
  }
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    eps="$(query 'sum(rate(mage_episodes_completed_total[1m]))')"
    ups="$(query 'sum(rate(mage_training_updates_total[1m]))')"
    active="$(query 'sum(mage_active_episodes)')"
    echo "[$ts] episodes/s=$eps updates/s=$ups active_episodes=$active"
    sleep 10
  done
else
  declare -A prev_c=()
  declare -A prev_u=()
  prev_ts=0
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    now_sec="$(date +%s)"
    ports="$(get_ports)"
    if [[ -z "$ports" ]]; then
      echo "[$ts] no running metrics ports discovered yet"
      sleep 10
      continue
    fi
    snap="$(sample_direct "$ports")"
    total_active=0
    delta_c=0
    delta_u=0
    resets=0
    while IFS=' ' read -r p c u a; do
      [[ -z "${p:-}" ]] && continue
      if [[ -n "${prev_c[$p]:-}" ]]; then
        dc=$(( c - prev_c[$p] ))
        du=$(( u - prev_u[$p] ))
        if (( dc < 0 || du < 0 )); then
          resets=$((resets + 1))
        else
          delta_c=$((delta_c + dc))
          delta_u=$((delta_u + du))
        fi
      fi
      prev_c[$p]="$c"
      prev_u[$p]="$u"
      total_active=$((total_active + a))
    done <<< "$snap"

    if (( prev_ts > 0 )); then
      dt=$(( now_sec - prev_ts ))
      if (( dt < 1 )); then dt=1; fi
      eps="$(awk -v d="$delta_c" -v t="$dt" 'BEGIN{printf "%.3f", d/t}')"
      ups="$(awk -v d="$delta_u" -v t="$dt" 'BEGIN{printf "%.3f", d/t}')"
      echo "[$ts] node=$node episodes/s=$eps updates/s=$ups active_episodes=$total_active ports=$(wc -w <<<"$ports") resets=$resets"
    else
      echo "[$ts] node=$node priming sample active_episodes=$total_active ports=$(wc -w <<<"$ports")"
    fi
    prev_ts="$now_sec"
    sleep 10
  done
fi
