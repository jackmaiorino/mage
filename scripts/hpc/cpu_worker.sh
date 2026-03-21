#!/bin/bash
#SBATCH --job-name=cpu-satellite
#SBATCH --account=msml603-class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=local-training/hpc/bundles/cpu-satellite_%j.out
#SBATCH --error=local-training/hpc/bundles/cpu-satellite_%j.err
#
# cpu_worker.sh - Launch multiple JVMs on a CPU node, connecting to a remote GPU service.
# Submit directly or via add_satellites.sh:
#   sbatch --export=ALL,GPU_SERVICE_ENDPOINT=gpu-node:port,MAGE_RL_RUNTIME_TARBALL=path,
#          RL_ARTIFACTS_ROOT=path scripts/hpc/cpu_worker.sh
set -euo pipefail

# Auto-discover GPU endpoint from satellite.env if not explicitly set
REPO_ROOT="${SOURCE_REPO_ROOT:-/home/jmaior/scratch.msml603/jmaior/mage}"
if [[ -z "${GPU_SERVICE_ENDPOINT:-}" || "${GPU_SERVICE_ENDPOINT:-}" == "PLACEHOLDER" ]]; then
    SAT_ENV="${REPO_ROOT}/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/satellite.env"
    if [[ -f "$SAT_ENV" ]]; then
        echo "Auto-discovering endpoint from $SAT_ENV"
        # shellcheck disable=SC1090
        source "$SAT_ENV"
    fi
fi

: "${GPU_SERVICE_ENDPOINT:?GPU_SERVICE_ENDPOINT must be set (e.g. gpu-a6-6:26180)}"
: "${MAGE_RL_RUNTIME_TARBALL:?MAGE_RL_RUNTIME_TARBALL must be set}"
: "${RL_ARTIFACTS_ROOT:?RL_ARTIFACTS_ROOT must be set (shared with GPU job)}"

JVMS_PER_NODE="${JVMS_PER_NODE:-4}"
RUNNERS_PER_JVM="${RUNNERS_PER_JVM:-150}"
TOTAL_EPISODES="${TOTAL_EPISODES:-10000000}"
PROFILES="${TRAIN_PROFILES_LIST:-Pauper-Spy-Combo-A,Pauper-Spy-Combo-B}"
HEAP_XMX="${AUX_JVM_XMX:-8g}"

# sbatch copies scripts to /var/spool, so dirname won't work. Use SOURCE_REPO_ROOT or detect.
REPO_ROOT="${SOURCE_REPO_ROOT:-/home/jmaior/scratch.msml603/jmaior/mage}"

echo "CPU worker starting on $(hostname) at $(date -u)"
echo "GPU_SERVICE_ENDPOINT=$GPU_SERVICE_ENDPOINT"
echo "JVMS_PER_NODE=$JVMS_PER_NODE RUNNERS_PER_JVM=$RUNNERS_PER_JVM"
echo "PROFILES=$PROFILES"
echo "RL_ARTIFACTS_ROOT=$RL_ARTIFACTS_ROOT"
echo "REPO_ROOT=$REPO_ROOT"

# Load Java
module load java 2>/dev/null || true

# Extract runtime tarball
RTDIR="/tmp/$USER/mtgrl-cpu-worker-${SLURM_JOB_ID:-$$}"
mkdir -p "$RTDIR"
tar xzf "$MAGE_RL_RUNTIME_TARBALL" -C "$RTDIR"
RDIR=$(ls -1d "$RTDIR"/rl-runtime-* | head -1)
CP=$(echo "$RDIR"/app/*.jar "$RDIR"/lib/*.jar | tr ' ' ':')
echo "Runtime extracted: $RDIR"

# Common env vars for JVMs
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_ENDPOINT
export GAME_STATS_WRITER=0
export GPU_SERVICE_NUM_CHANNELS="${GPU_SERVICE_NUM_CHANNELS:-4}"
export GPU_SERVICE_NUM_GPUS="${GPU_SERVICE_NUM_GPUS:-1}"
export TOTAL_EPISODES
export MODE=trainAll
export TRAIN_PROFILES_LIST="$PROFILES"
export MULLIGAN_DEVICE=cpu
export GAME_LOG_FREQUENCY="${GAME_LOG_FREQUENCY:-0}"
export OPPONENT_SAMPLER="${OPPONENT_SAMPLER:-self}"
export RL_ARTIFACTS_ROOT
export LEAGUE_REGISTRY_PATH="${LEAGUE_REGISTRY_PATH:-$REPO_ROOT/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json}"
export ORCHESTRATED_RUN=1

# Build deck list from registry profiles
DECK_LIST_FILE="${DECK_LIST_FILE:-$REPO_ROOT/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt}"
export DECK_LIST_FILE
echo "DECK_LIST_FILE=$DECK_LIST_FILE"
echo "LEAGUE_REGISTRY_PATH=$LEAGUE_REGISTRY_PATH"

# Persistent log directory (survives job cleanup)
PERSISTENT_LOGDIR="${REPO_ROOT}/local-training/hpc/bundles/cpu-worker-${SLURM_JOB_ID:-$$}"
mkdir -p "$PERSISTENT_LOGDIR"
echo "Logs: $PERSISTENT_LOGDIR"

PIDS=()
for i in $(seq 0 $((JVMS_PER_NODE - 1))); do
    METRICS_PORT=$((19900 + ${SLURM_JOB_ID:-0} % 100 * 10 + i))
    LOGDIR="$PERSISTENT_LOGDIR/jvm_$i"
    mkdir -p "$LOGDIR"

    NUM_GAME_RUNNERS=$RUNNERS_PER_JVM \
    METRICS_PORT=$METRICS_PORT \
    MAGE_DB_DIR="$LOGDIR/db" \
    MAGE_DB_AUTO_SERVER=false \
    java -Xms512m -Xmx${HEAP_XMX} -cp "$CP" \
        mage.player.ai.rl.RLTrainer trainAll ${PROFILES//,/ } \
        > "$LOGDIR/stdout.log" 2> "$LOGDIR/stderr.log" &
    PIDS+=($!)
    echo "Started JVM $i pid=$! runners=$RUNNERS_PER_JVM metricsPort=$METRICS_PORT"
done

echo "All $JVMS_PER_NODE JVMs launched. Waiting..."

# Wait for all, track failures
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

# Print last lines of each JVM's stderr for diagnostics
for i in $(seq 0 $((JVMS_PER_NODE - 1))); do
    LOGDIR="$PERSISTENT_LOGDIR/jvm_$i"
    echo "=== JVM $i stderr (last 10) ==="
    tail -10 "$LOGDIR/stderr.log" 2>/dev/null || true
done

echo "CPU worker done. failed=$FAILED"
exit $FAILED
