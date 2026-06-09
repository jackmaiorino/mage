#!/bin/bash
#SBATCH --job-name=engine-bench
#SBATCH --account=msml603-class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=local-training/hpc/bundles/enginebench_%j.out
#SBATCH --error=local-training/hpc/bundles/enginebench_%j.err
#
# Decisive test: pure XMage engine speed on a standard EPYC node, at 1x and 2x
# oversubscription, to separate "EPYC core intrinsically slow at XMage" from
# "oversubscription/round-trip". Self-contained: a CPU-torch inference service +
# RLTrainer MODE=benchmark (self-contained game player). Measures games/s per thread.
set -uo pipefail

REPO="/scratch/zt1/project/msml603/user/jmaior/jmaior/mage"
cd "$REPO"
echo "=== CPU MODEL ==="; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz' || true
echo "=== nproc ==="; nproc
echo "=== SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?} ==="

TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
echo "TARBALL=$TARBALL"
RTDIR="/tmp/$USER/enginebench-${SLURM_JOB_ID}"
mkdir -p "$RTDIR"
tar xzf "$TARBALL" -C "$RTDIR"
RDIR=$(ls -1d "$RTDIR"/rl-runtime-* | head -1)
CP=$(echo "$RDIR"/app/*.jar "$RDIR"/lib/*.jar | tr ' ' ':')
echo "CP entries: $(echo "$CP" | tr ':' '\n' | grep -c jar)"

module load java 2>/dev/null || true
java -version 2>&1 | head -1
PYBIN="$REPO/.mtgrl_venv_hpc/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"
echo "PYBIN=$PYBIN"

export MAGE_DB_DIR="${SLURM_TMPDIR:-/tmp/$USER}/enginebench-${SLURM_JOB_ID}/db"
mkdir -p "$MAGE_DB_DIR"
export MAGE_DB_AUTO_SERVER=false

# CPU-only torch inference service
export CUDA_VISIBLE_DEVICES=""
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_PORT=26199
export GPU_SERVICE_METRICS_PORT=27199
export GPU_SERVICE_BIND_HOST=127.0.0.1
export MODEL_D_MODEL=128
export MODEL_NUM_LAYERS=2
export MODEL_PROFILE=Pauper-Spy-Combo-Value
export INFER_CUDA_DEVICE=cpu
export TRAIN_CUDA_DEVICE=cpu
export SCORE_WORKER_THREADS=4
# CPU checkpoint load is slow -> raise JVM-side control/score timeouts so registration completes
export GPU_SERVICE_CONTROL_TIMEOUT_MS=180000
export GPU_SERVICE_CONNECT_TIMEOUT_MS=60000
export PY_SCORE_TIMEOUT_MS=180000
export PYTHON_LOGS_DIR="$REPO/local-training/hpc/bundles/enginebench_pylogs_${SLURM_JOB_ID}"
mkdir -p "$PYTHON_LOGS_DIR"
export MTG_AI_LOG_FILE="$PYTHON_LOGS_DIR/mtg_ai.log"
HOST_SCRIPT="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py"

echo "Starting CPU-torch inference service..."
"$PYBIN" "$HOST_SCRIPT" > "$PYTHON_LOGS_DIR/service.log" 2>&1 &
SVC_PID=$!

ready=0
for i in $(seq 1 90); do
  if (exec 3<>/dev/tcp/127.0.0.1/26199) 2>/dev/null; then exec 3>&- 3<&-; echo "service ready after ${i}s"; ready=1; break; fi
  if ! kill -0 "$SVC_PID" 2>/dev/null; then echo "SERVICE DIED; tail:"; tail -30 "$PYTHON_LOGS_DIR/service.log"; exit 1; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "service never ready; tail:"; tail -30 "$PYTHON_LOGS_DIR/service.log"; exit 1; }

DECK="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt"

run_phase() {
  local THREADS=$1; local SECS=$2; local LABEL=$3
  echo "===== PHASE $LABEL : BENCHMARK_THREADS=$THREADS for ${SECS}s ====="
  local OUT="$REPO/local-training/hpc/bundles/enginebench_${SLURM_JOB_ID}_${LABEL}.log"
  PY_SERVICE_MODE=shared_gpu GPU_SERVICE_ENDPOINT=127.0.0.1:26199 \
  MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODE=benchmark \
  BENCHMARK_THREADS=$THREADS GAMES_PER_MATCHUP=1000000 BENCHMARK_HEARTBEAT_SEC=15 \
  BENCHMARK_GAME_TIMEOUT_SEC=900 GAME_HEALTH_MONITOR=1 \
  DECK_LIST_FILE="$DECK" RL_SILENCE_THREAD_LOGGER=1 \
  java -Xmx16g -cp "$CP" mage.player.ai.rl.RLTrainer benchmark > "$OUT" 2>&1 &
  local JPID=$!
  sleep "$SECS"
  kill "$JPID" 2>/dev/null; sleep 3; kill -9 "$JPID" 2>/dev/null; wait "$JPID" 2>/dev/null
  echo "--- $LABEL: heartbeat/throughput lines ---"
  grep -aiE 'completed|done=|games/s|throughput|elapsed|ETA|Benchmark started|device info' "$OUT" | tail -12
  echo "--- $LABEL: tail ---"; tail -6 "$OUT"
}

run_phase 32 420 "t32_1x"
run_phase 64 360 "t64_2x"

kill "$SVC_PID" 2>/dev/null
echo "===== ENGINE BENCH DONE ====="
