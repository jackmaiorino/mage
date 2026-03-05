#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
status_path="${1:-$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/orchestrator_status.json}"
output_path="${2:-$repo_root/monitoring/file_sd/mage_hpc_targets.json}"
metrics_host="${METRICS_HOST:-localhost}"
include_stopped="${INCLUDE_STOPPED_TARGETS:-0}"

args=(
  --repo-root "$repo_root"
  --status "$status_path"
  --output "$output_path"
  --host "$metrics_host"
)

if [[ "$include_stopped" == "1" || "$include_stopped" == "true" || "$include_stopped" == "yes" ]]; then
  args+=(--include-stopped)
fi

python3 "$repo_root/scripts/hpc/generate_prometheus_targets.py" "${args[@]}"
