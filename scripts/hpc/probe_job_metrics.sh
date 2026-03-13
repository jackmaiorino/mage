#!/usr/bin/env bash
set -euo pipefail

local_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$local_repo_root"
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
JOB_ID="${1:-${JOB:-}}"
if [[ -z "${JOB_ID}" ]]; then
  echo "usage: bash scripts/hpc/probe_job_metrics.sh <job_id>" >&2
  exit 2
fi

job_info="$(scontrol show job "${JOB_ID}" 2>/dev/null || true)"
job_workdir="$(printf '%s\n' "$job_info" | grep -o 'WorkDir=[^ ]*' | head -1 | cut -d= -f2- || true)"
if [[ -n "$job_workdir" && -x "$job_workdir/scripts/hpc/collect_job_metrics_local.sh" ]]; then
  repo_root="$job_workdir"
fi

collector_script="$repo_root/scripts/hpc/collect_job_metrics_local.sh"
job_reports_root="$(resolve_path "${HPC_JOB_REPORTS_ROOT:-}" "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs")"
orch_compat_reports_root="$(resolve_path "${ORCHESTRATOR_COMPAT_REPORTS_ROOT:-}" "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator")"
final_metrics_path="$job_reports_root/$JOB_ID/final_probe_metrics.txt"

TRAINER_PORT_START="${TRAINER_PORT_START:-19100}"
TRAINER_PORT_END="${TRAINER_PORT_END:-19131}"
GPU_HOST_METRICS_PORT_START="${GPU_HOST_METRICS_PORT_START:-27100}"
GPU_HOST_METRICS_PORT_END="${GPU_HOST_METRICS_PORT_END:-27115}"

status_path="$orch_compat_reports_root/runs/$JOB_ID/orchestrator_status.json"
orchestrator_log="$job_reports_root/$JOB_ID/orchestrator.log"

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
