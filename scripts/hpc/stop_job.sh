#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
job_id="${1:-}"

if [[ -z "$job_id" ]]; then
  echo "Usage: $0 <slurm_job_id>" >&2
  exit 1
fi

if command -v scancel >/dev/null 2>&1; then
  echo "Cancelling Slurm job ${job_id}..."
  scancel "$job_id" || true
else
  echo "scancel not found; skipping Slurm cancel" >&2
fi

echo "Running RL cleanup..."
if [[ -f "$repo_root/scripts/rl-stop.sh" ]]; then
  bash "$repo_root/scripts/rl-stop.sh" -q || true
else
  pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$repo_root/scripts/rl-stop.ps1" -Quiet || true
fi

if command -v squeue >/dev/null 2>&1; then
  echo
  echo "Current queue status for ${job_id}:"
  squeue -j "$job_id" -o "%.18i %.10T %.9M %.12l %.20R" || true
fi

echo "Done."

