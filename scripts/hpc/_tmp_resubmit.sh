#!/bin/bash
set -euo pipefail
cd /home/jmaior/scratch.msml603/jmaior/mage

scancel 18598749 18598750 18598751 18598752
echo 'Cancelled old satellites'
sleep 2

echo '=== QUEUE ==='
squeue -u $USER

GPU_JOB=18598748
BUNDLE=$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz | head -1)

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
  echo "Satellite $i resubmitted: $SAT_JOB"
done

echo ''
echo '=== FINAL QUEUE ==='
squeue -u $USER
