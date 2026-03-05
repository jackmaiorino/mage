#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
interval_sec="${SYNC_INTERVAL_SEC:-5}"
status_path="${1:-$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/orchestrator_status.json}"
output_path="${2:-$repo_root/monitoring/file_sd/mage_hpc_targets.json}"

while true; do
  if ! bash "$repo_root/scripts/hpc/refresh_prometheus_targets.sh" "$status_path" "$output_path"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] target refresh failed" >&2
  fi
  sleep "$interval_sec"
done
