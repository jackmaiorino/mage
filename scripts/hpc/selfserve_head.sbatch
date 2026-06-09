#!/bin/bash
#SBATCH --job-name=ss-head
#SBATCH --account=msml603-class
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=local-training/hpc/bundles/sshead_%j.out
#SBATCH --error=local-training/hpc/bundles/sshead_%j.err
#
# Self-serve smoke HEAD: GPU PPO training service (accepts remote training from
# self-serve CPU satellites) + onnx_publisher sidecar (exports ONNX for satellites
# to a fixed shared RL_ARTIFACTS_ROOT). Continues from the Value 52% model so games
# are realistic. Add the satellite manually after the head publishes ONNX.
set -euo pipefail
TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then echo "FATAL: no tarball" >&2; exit 1; fi
export MAGE_RL_RUNTIME_TARBALL="$TARBALL"

export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"     # accept remote satellites
export TOTAL_EPISODES=500000
export TRAIN_PROFILES=1                     # -> Pauper-Spy-Combo-Value
export GAME_LOG_FREQUENCY=0
export OPPONENT_SAMPLER=self
export GPU_SERVICE_SPLIT_ROLES=0
export WINRATE_WINDOW=200

# fixed shared artifacts root (pre-populated with the Value model); satellites read it
export RL_ARTIFACTS_ROOT="/scratch/zt1/project/msml603/user/jmaior/jmaior/mage/local-training/selfserve_run/rl_artifacts"
# weight-sync publisher for self-serve satellites
export ONNX_PUBLISH_ENABLE=1
export ONNX_PUBLISH_INTERVAL_S=300   # 5 min: satellite weights can lag; avoids reload thrash + inference stalls
export WRITE_SATELLITE_ENV=1
export AUTO_SATELLITES=0                    # add the satellite manually after ONNX is published

export PRIMARY_RUNNER_CAP=8                 # a few local runners; satellite does the bulk
export SCORE_WORKER_THREADS=0
export GAME_TIMEOUT_SEC=900
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2
export SEARCH_OP_ENABLE=0
export MCTS_TRAINING_ENABLE=0

echo "ss-head on $(hostname) job=$SLURM_JOB_ID at $(date -u)"
echo "RL_ARTIFACTS_ROOT=$RL_ARTIFACTS_ROOT  TARBALL=$TARBALL"
exec bash scripts/hpc/run_spy_pbt.sh
