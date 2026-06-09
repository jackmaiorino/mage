#!/bin/bash
#SBATCH --job-name=fm-head
#SBATCH --account=msml603-class
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=local-training/hpc/bundles/fmhead_%j.out
#SBATCH --error=local-training/hpc/bundles/fmhead_%j.err
#
# FRESH-on-META head (the unconfounded diet test). FRESH Spy trains vs aggressive
# NN opponents (Rally/Wildfire/Affinity) + eval-aligned Grixis via OPPONENT_SAMPLER=meta.
# Cold-start is bridged by a TerminalPrefixSearch bootstrap on the SATELLITES (gated to
# Balustrade-Spy cast decisions). Head = PPO training (4 profiles) + onnx_publisher (all 4).
# Uses an ISOLATED registry (does not disturb the shared one). REQUIRES preflight first.
set -euo pipefail
TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then echo "FATAL: no tarball" >&2; exit 1; fi
export MAGE_RL_RUNTIME_TARBALL="$TARBALL"

export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"
export TOTAL_EPISODES=200000
export TRAIN_PROFILES=4
export OPPONENT_SAMPLER=meta
export GPU_SERVICE_SPLIT_ROLES=0
export WINRATE_WINDOW=200
export GAME_LOG_FREQUENCY=200

# isolated registry (OPPONENT_SAMPLER=meta x4 + Spy high-exploration); does NOT touch the shared registry
export REGISTRY_PATH="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_fresh_meta_registry.json"
export LEAGUE_REGISTRY_PATH="$REGISTRY_PATH"

# fixed shared artifacts root: preflight pre-seeds FRESH Spy + the 3 trained opponents + all 4 ONNX
export RL_ARTIFACTS_ROOT="/scratch/zt1/project/msml603/user/jmaior/jmaior/mage/local-training/fresh_meta_run/rl_artifacts"
export ONNX_PUBLISH_ENABLE=1
export ONNX_PUBLISH_INTERVAL_S=120
export WRITE_SATELLITE_ENV=1
export AUTO_SATELLITES=0

export PRIMARY_RUNNER_CAP=4
export SCORE_WORKER_THREADS=0
export GAME_TIMEOUT_SEC=900
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2
export SEARCH_OP_ENABLE=0          # search bootstrap runs on SATELLITES, not the head
export MCTS_TRAINING_ENABLE=0

echo "fm-head on $(hostname) job=$SLURM_JOB_ID at $(date -u)"
echo "REGISTRY=$REGISTRY_PATH  RL_ARTIFACTS_ROOT=$RL_ARTIFACTS_ROOT"
exec bash scripts/hpc/run_spy_pbt.sh
