#!/usr/bin/env bash
set -euo pipefail

# Run this on a Zaratan login node from any repo checkout path.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

partition="${ONLINE_PARTITION:-gpu-a100}"
gres="${ONLINE_GRES-gpu:a100:1}"
cpus_per_task="${ONLINE_CPUS_PER_TASK:-24}"
mem="${ONLINE_MEM:-64G}"
time_limit="${ONLINE_TIME:-04:00:00}"
job_name="${ONLINE_JOB_NAME:-online-mine}"

echo "Submitting online terminal mining"
echo "partition=${partition} gres=${gres} cpus=${cpus_per_task} mem=${mem} time=${time_limit}"

sbatch_args=(
    --parsable
    --job-name="${job_name}"
    --partition="${partition}"
    --cpus-per-task="${cpus_per_task}"
    --mem="${mem}"
    --time="${time_limit}"
    --export=ALL
)
if [[ -n "${gres}" ]]; then
  sbatch_args+=(--gres="${gres}")
fi
sbatch_args+=(scripts/hpc/submit_online_terminal_mining.slurm "$@")

job_id="$(
  sbatch "${sbatch_args[@]}"
)"

echo "Submitted JOB_ID=${job_id}"
squeue -j "${job_id}" -o "%.18i %.10T %.9M %.12l %.8C %.8m %.20R" || true

if [[ "${ONLINE_WATCH:-1}" == "1" || "${ONLINE_WATCH:-}" == "true" ]]; then
  echo "Watching job logs..."
  bash scripts/hpc/watch_job.sh "${job_id}"
fi
