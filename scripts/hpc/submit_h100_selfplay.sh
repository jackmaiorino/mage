#!/usr/bin/env bash
set -euo pipefail
cd /home/jmaior/scratch.msml603/jmaior/mage

BUNDLE=/home/jmaior/scratch.msml603/jmaior/mage/local-training/hpc/bundles/rl-runtime-19cc8d4465-20260312-202443.tar.gz

sbatch \
  --job-name=spy-h100-selfplay-25x \
  --partition=gpu-h100 \
  --gres=gpu:h100:1 \
  --cpus-per-task=24 \
  --mem=64G \
  --time=04:00:00 \
  --account=msml603-gpu \
  --output=local-training/hpc/bundles/spy-h100-selfplay-25x_%j.out \
  --error=local-training/hpc/bundles/spy-h100-selfplay-25x_%j.err \
  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$BUNDLE,RUNNER_OVERSUBSCRIPTION_FACTOR=25,TRAIN_PROFILES=1,TOTAL_EPISODES=100000,OPPONENT_SAMPLER=fixed,FIXED_WEAK_UNTIL=0,FIXED_MEDIUM_UNTIL=0,FIXED_STRONG_UNTIL=0,BOT_FLOOR=0,RL_SKIP_SIM_VALIDATION=1,DECK_LIST_FILE=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.spy_combo.txt,PY_SERVICE_MODE=shared,GAME_LOG_FREQUENCY=500 \
  scripts/hpc/run_spy_pbt.sh

echo "Submitted. Checking queue..."
squeue -u $USER --format="%.10i %.15P %.30j %.8T %.10M %.6D %R" | head -20
