#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_ID="${1:-${JOB:-}}"
if [[ -z "${JOB_ID}" ]]; then
  echo "usage: bash scripts/hpc/probe_job_metrics.sh <job_id>" >&2
  exit 2
fi

collector_script="$repo_root/scripts/hpc/collect_job_metrics_local.sh"
final_metrics_path="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs/$JOB_ID/final_probe_metrics.txt"

TRAINER_PORT_START="${TRAINER_PORT_START:-19100}"
TRAINER_PORT_END="${TRAINER_PORT_END:-19131}"
GPU_HOST_METRICS_PORT_START="${GPU_HOST_METRICS_PORT_START:-27100}"
GPU_HOST_METRICS_PORT_END="${GPU_HOST_METRICS_PORT_END:-27115}"

status_path="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs/$JOB_ID/orchestrator_status.json"
orchestrator_log="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs/$JOB_ID/orchestrator.log"

job_state="$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
if [[ "$job_state" != "RUNNING" ]]; then
  if [[ -f "$final_metrics_path" ]]; then
    echo "source=final_snapshot"
    cat "$final_metrics_path"
    exit 0
  fi
  echo "job ${JOB_ID} is not running and no final metrics snapshot exists at ${final_metrics_path}" >&2
  exit 1
fi

if output="$(srun --jobid="${JOB_ID}" --overlap -N1 -n1 -c1 bash "$collector_script" \
  --status-path "$status_path" \
  --orchestrator-log "$orchestrator_log" 2>&1)"; then
  echo "source=live_srun"
  printf '%s\n' "$output"
  exit 0
fi

if [[ -f "$final_metrics_path" ]]; then
  echo "source=final_snapshot"
  cat "$final_metrics_path"
  exit 0
fi

printf '%s\n' "$output" >&2
exit 1
