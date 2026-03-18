#!/bin/bash
set -euo pipefail
cd /home/jmaior/scratch.msml603/jmaior/mage

# Cancel all current jobs
echo "Cancelling all jobs..."
scancel 18598748 18598760 18598761 18598762 18598763 2>/dev/null || true
sleep 3

echo "=== Verifying clean ==="
squeue -u $USER

BUNDLE=$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz | head -1)
echo "Bundle: $BUNDLE"

# Submit GPU head with multi-threaded score workers (auto-scale)
GPU_JOB=$(sbatch \
  --parsable \
  --job-name=gpu-mt \
  --partition=gpu-h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=64G \
  --time=04:00:00 --account=msml603-class \
  --output=local-training/hpc/bundles/gpu-mt_%j.out \
  --error=local-training/hpc/bundles/gpu-mt_%j.err \
  --export=ALL,MAGE_RL_RUNTIME_TARBALL=$BUNDLE,HPC_NATIVE_ORCH=1,MULTI_PROFILE_JVM=1,PY_SERVICE_MODE=shared_gpu,GPU_SERVICE_BIND_HOST=0.0.0.0,TOTAL_EPISODES=10000000,TRAIN_PROFILES=3,GAME_LOG_FREQUENCY=0,OPPONENT_SAMPLER=self,WINRATE_WINDOW=200,PRIMARY_RUNNER_CAP=50,RUNNER_OVERSUBSCRIPTION_FACTOR=1,PY_BATCH_MAX_SIZE=256,PY_BATCH_TIMEOUT_MS=25,WRITE_SATELLITE_ENV=1,SCORE_WORKER_THREADS=0 \
  scripts/hpc/run_spy_pbt.sh)
echo "GPU head submitted: $GPU_JOB"

# Submit 4 CPU satellites
for i in 1 2 3 4; do
  SAT_JOB=$(sbatch \
    --parsable \
    --job-name=sat-${GPU_JOB}-${i} \
    --partition=standard --cpus-per-task=128 --mem=128G \
    --time=04:00:00 --account=msml603-class \
    --output=local-training/hpc/bundles/sat-${GPU_JOB}-${i}_%j.out \
    --error=local-training/hpc/bundles/sat-${GPU_JOB}-${i}_%j.err \
    --dependency=after:${GPU_JOB}+1 \
    --export=ALL,GPU_SERVICE_ENDPOINT=PLACEHOLDER,MAGE_RL_RUNTIME_TARBALL=$BUNDLE,RL_ARTIFACTS_ROOT=/home/jmaior/scratch.msml603/jmaior/mage/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs/${GPU_JOB}/rl_artifacts,JVMS_PER_NODE=4,RUNNERS_PER_JVM=150,TRAIN_PROFILES_LIST=Pauper-Spy-A,Pauper-Spy-B,Pauper-Spy-C,SOURCE_REPO_ROOT=/home/jmaior/scratch.msml603/jmaior/mage,GAME_LOG_FREQUENCY=0,OPPONENT_SAMPLER=self \
    scripts/hpc/cpu_worker.sh)
  echo "Satellite $i: $SAT_JOB"
done

echo ""
echo "=== FINAL QUEUE ==="
squeue -u $USER
