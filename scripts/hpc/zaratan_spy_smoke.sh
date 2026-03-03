#!/usr/bin/env bash
set -euo pipefail

# Run this on a Zaratan login node after syncing the repo.
cd /home/jmaior/mage

BUNDLE="$(ls -1t /home/jmaior/mage/local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -n 1 || true)"
if [[ -z "${BUNDLE}" ]]; then
  echo "No runtime bundle found under /home/jmaior/mage/local-training/hpc/bundles/" >&2
  echo "Build/upload one first with scripts/hpc/build_rl_runtime_bundle.ps1" >&2
  exit 1
fi

echo "Using bundle: ${BUNDLE}"

JOB_ID="$(
  sbatch --parsable \
    --job-name=spy-smoke \
    --time=00:25:00 \
    --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL="${BUNDLE}",TOTAL_EPISODES=40,TRAIN_PROFILES=1,INFER_WORKERS=1,EVAL_EVERY_MINUTES=10,GAME_LOG_FREQUENCY=20 \
    scripts/hpc/submit_spy_pbt.slurm
)"

echo "Submitted JOB_ID=${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.10T %.9M %.12l %.8C %.8m %.20R" || true

echo "Watching job logs..."
bash scripts/hpc/watch_job.sh "${JOB_ID}"
