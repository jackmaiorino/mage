#!/bin/bash
#SBATCH --job-name=selfserve-smoke
#SBATCH --account=msml603-class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=local-training/hpc/bundles/selfserve_%j.out
#SBATCH --error=local-training/hpc/bundles/selfserve_%j.err
#
# Self-serve smoke: prove a standard EPYC node, doing its OWN ORT-CPU inference
# (PY_SERVICE_MODE=onnx + ONNX_FORCE_CPU=1, NO GPU, NO shared service), reaches
# engine-bound throughput. This is the per-node throughput of the proposed
# self-serve-satellite architecture. THREADS=32 (1x/2core) and 64 (1x/core).
set -uo pipefail
REPO="/scratch/zt1/project/msml603/user/jmaior/jmaior/mage"
cd "$REPO"
echo "=== CPU ==="; lscpu | grep -E 'Model name|^CPU\(s\)' | head -2; echo "nproc=$(nproc)"

TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
RTDIR="/tmp/$USER/selfserve-${SLURM_JOB_ID}"; mkdir -p "$RTDIR"
tar xzf "$TARBALL" -C "$RTDIR"
RDIR=$(ls -1d "$RTDIR"/rl-runtime-* | head -1)
CP=$(echo "$RDIR"/app/*.jar "$RDIR"/lib/*.jar | tr ' ' ':')
# Prepend patched classes (OnnxInferenceModel intra-op thread cap) so they shadow
# the bundle jar without a full bundle rebuild.
PATCH_DIR="$REPO/local-training/patch"
if [[ -d "$PATCH_DIR" ]]; then CP="$PATCH_DIR:$CP"; echo "patched classpath: $PATCH_DIR prepended"; fi
module load java 2>/dev/null || true
java -version 2>&1 | head -1

export MAGE_DB_DIR="${SLURM_TMPDIR:-/tmp/$USER}/selfserve-${SLURM_JOB_ID}/db"; mkdir -p "$MAGE_DB_DIR"
export MAGE_DB_AUTO_SERVER=false
export CUDA_VISIBLE_DEVICES=""
export MODEL_PROFILE=Pauper-Spy-Combo-Value
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2
export RL_ARTIFACTS_ROOT="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl"
DECK="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt"

run_phase() {
  local THREADS=$1; local SECS=$2; local LABEL=$3
  echo "===== PHASE $LABEL : THREADS=$THREADS for ${SECS}s (onnx CPU self-serve) ====="
  local OUT="$REPO/local-training/hpc/bundles/selfserve_${SLURM_JOB_ID}_${LABEL}.log"
  PY_SERVICE_MODE=onnx ONNX_FORCE_CPU=1 ONNX_INTRA_OP_THREADS=1 \
  ONNX_BATCH_MAX_SIZE=1 ONNX_BATCH_TIMEOUT_MS=1 ONNX_BATCH_TIMEOUT_MAX_MS=2 ONNX_INFER_THREADS=$THREADS \
  MODE=benchmark BENCHMARK_THREADS=$THREADS GAMES_PER_MATCHUP=1000000 BENCHMARK_HEARTBEAT_SEC=15 \
  BENCHMARK_LIVE_REPORT_PATH="$REPO/local-training/hpc/bundles/selfserve_${SLURM_JOB_ID}_${LABEL}_live.txt" \
  DECK_LIST_FILE="$DECK" RL_SILENCE_THREAD_LOGGER=1 GAME_HEALTH_MONITOR=1 \
  java -Xmx48g -cp "$CP" mage.player.ai.rl.RLTrainer benchmark > "$OUT" 2>&1 &
  local JPID=$!
  sleep "$SECS"
  kill "$JPID" 2>/dev/null; sleep 3; kill -9 "$JPID" 2>/dev/null; wait "$JPID" 2>/dev/null
  echo "--- $LABEL live report ---"; cat "$REPO/local-training/hpc/bundles/selfserve_${SLURM_JOB_ID}_${LABEL}_live.txt" 2>/dev/null | tail -8
  echo "--- $LABEL: Benchmark/throughput/onnx lines ---"
  grep -aiE 'Benchmark started|ONNX|Forced CPU|completed|games/s|throughput|ETA|elapsed|plannedGames' "$OUT" 2>/dev/null | grep -aviE 'Slow inference' | tail -12
  echo "--- $LABEL: errors? ---"; grep -aiE 'error|exception|requires|Invalid|not ready|fatal' "$OUT" 2>/dev/null | head -4
}

run_phase 32 360 "t32"
run_phase 64 300 "t64"
echo "===== SELF-SERVE SMOKE DONE ====="
