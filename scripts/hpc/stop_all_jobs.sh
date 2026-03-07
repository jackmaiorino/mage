#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
states="RUNNING"
if [[ "${1:-}" == "--all" ]]; then
  states="RUNNING,PENDING"
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found on PATH." >&2
  exit 1
fi

if ! command -v scancel >/dev/null 2>&1; then
  echo "scancel not found on PATH." >&2
  exit 1
fi

mapfile -t job_ids < <(squeue -u "$USER" -t "$states" -h -o "%i")

if [[ "${#job_ids[@]}" -eq 0 ]]; then
  echo "No ${states} jobs found for user ${USER}."
  exit 0
fi

echo "Cancelling ${#job_ids[@]} ${states} job(s) for user ${USER}:"
printf '  %s\n' "${job_ids[@]}"
scancel "${job_ids[@]}"

echo
echo "Remaining queue entries for ${USER}:"
squeue -u "$USER" -o "%.18i %.10T %.9M %.12l %.20R" || true

echo
echo "Running RL cleanup..."
if [[ -f "$repo_root/scripts/rl-stop.sh" ]]; then
  bash "$repo_root/scripts/rl-stop.sh" -q || true
elif command -v pwsh >/dev/null 2>&1; then
  pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$repo_root/scripts/rl-stop.ps1" -Quiet || true
fi

echo "Done."
