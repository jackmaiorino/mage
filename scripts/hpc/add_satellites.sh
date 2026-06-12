#!/bin/bash
# add_satellites.sh Ã¢â‚¬â€ Submit CPU satellite workers to feed a running GPU head job.
#
# Usage:
#   bash scripts/hpc/add_satellites.sh <GPU_JOB_ID> [NUM_SATELLITES] [OPTIONS]
#
# Options (env vars or flags):
#   PARTITION       Slurm partition for CPU jobs (default: standard)
#   CPUS            CPUs per satellite node (default: 128)
#   MEM             Memory per satellite (default: 128G)
#   JVMS_PER_NODE   JVMs per satellite node (default: 4)
#   RUNNERS_PER_JVM Game runners per JVM (default: 150)
#   TIME            Wall time (default: matches GPU job remaining)
#   BUNDLE          Runtime tarball path (auto-detected from satellite.env)
#
# The script reads satellite.env written by the GPU head orchestrator to
# discover GPU_SERVICE_ENDPOINT, RL_ARTIFACTS_ROOT, and TARBALL path.
# If satellite.env is not found, it falls back to squeue discovery.

set -euo pipefail

GPU_JOB_ID="${1:?Usage: add_satellites.sh <GPU_JOB_ID> [NUM_SATELLITES]}"
NUM_SATELLITES="${2:-1}"
PARTITION="${PARTITION:-standard}"
CPUS="${CPUS:-128}"
MEM="${MEM:-128G}"
TIME="${TIME:-}"
JVMS_PER_NODE="${JVMS_PER_NODE:-4}"
RUNNERS_PER_JVM="${RUNNERS_PER_JVM:-150}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Try to read satellite.env from the orchestrator's reports directory
ORCH_REPORTS="$REPO_ROOT/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
SAT_ENV=""
for candidate in \
    "$ORCH_REPORTS/runs/$GPU_JOB_ID/satellite.env" \
    "$ORCH_REPORTS/satellite.env"; do
    if [[ -f "$candidate" ]]; then
        SAT_ENV="$candidate"
        break
    fi
done

GPU_NODE=""
GPU_SERVICE_ENDPOINT=""
BUNDLE="${BUNDLE:-}"
RL_ARTIFACTS_ROOT="${RL_ARTIFACTS_ROOT:-}"
TRAIN_PROFILES_LIST="${TRAIN_PROFILES_LIST:-}"

if [[ -n "$SAT_ENV" ]]; then
    echo "Reading satellite config from $SAT_ENV"
    # shellcheck disable=SC1090
    source "$SAT_ENV"
else
    echo "No satellite.env found; falling back to squeue discovery"
fi

# Discover GPU node from squeue if not set
if [[ -z "$GPU_NODE" ]]; then
    GPU_NODE=$(squeue -j "$GPU_JOB_ID" -h -o "%N" 2>/dev/null | head -1)
fi
if [[ -z "$GPU_NODE" ]]; then
    echo "ERROR: Could not determine GPU node for job $GPU_JOB_ID. Is it running?" >&2
    exit 1
fi

# Compute endpoint if not set (same port formula as run_spy_pbt.sh)
if [[ -z "$GPU_SERVICE_ENDPOINT" ]]; then
    PORT_SLOT=$((GPU_JOB_ID % 20))
    GPU_SERVICE_PORT=$((26100 + PORT_SLOT * 40))
    GPU_SERVICE_ENDPOINT="${GPU_NODE}:${GPU_SERVICE_PORT}"
fi

# Find bundle if not set
if [[ -z "$BUNDLE" ]]; then
    BUNDLE="$(ls -1t "$REPO_ROOT/local-training/hpc/bundles/rl-runtime-"*.tar.gz 2>/dev/null | head -1)"
fi
if [[ -z "$BUNDLE" || ! -f "$BUNDLE" ]]; then
    echo "ERROR: No runtime tarball found. Set BUNDLE= or place in local-training/hpc/bundles/" >&2
    exit 1
fi

# Find RL_ARTIFACTS_ROOT from orchestrator status if not set
if [[ -z "$RL_ARTIFACTS_ROOT" ]]; then
    STATUS_JSON="$ORCH_REPORTS/orchestrator_status.json"
    if [[ -f "$STATUS_JSON" ]] && command -v python3 >/dev/null 2>&1; then
        RL_ARTIFACTS_ROOT=$(python3 -c "
import json, sys
try:
    d = json.load(open('$STATUS_JSON'))
    print(d.get('paths', {}).get('artifacts_root', ''))
except Exception:
    pass
" 2>/dev/null || true)
    fi
fi
if [[ -z "$RL_ARTIFACTS_ROOT" ]]; then
    # Fallback: use the standard run path
    RL_ARTIFACTS_ROOT="$ORCH_REPORTS/runs/$GPU_JOB_ID/rl_artifacts"
fi

# Auto-detect remaining time from GPU job
if [[ -z "$TIME" ]]; then
    REMAINING=$(squeue -j "$GPU_JOB_ID" -h -o "%L" 2>/dev/null | head -1)
    if [[ -n "$REMAINING" && "$REMAINING" != "INVALID" ]]; then
        TIME="$REMAINING"
    else
        TIME="12:00:00"
    fi
fi

# Discover profiles if not set
if [[ -z "$TRAIN_PROFILES_LIST" ]]; then
    STATUS_JSON="$ORCH_REPORTS/orchestrator_status.json"
    if [[ -f "$STATUS_JSON" ]] && command -v python3 >/dev/null 2>&1; then
        TRAIN_PROFILES_LIST=$(python3 -c "
import json, sys
try:
    d = json.load(open('$STATUS_JSON'))
    profiles = d.get('selected_profiles', [])
    print(','.join(profiles))
except Exception:
    pass
" 2>/dev/null || true)
    fi
fi

echo "=== Satellite submission ==="
echo "GPU_JOB_ID=$GPU_JOB_ID"
echo "GPU_NODE=$GPU_NODE"
echo "GPU_SERVICE_ENDPOINT=$GPU_SERVICE_ENDPOINT"
echo "BUNDLE=$BUNDLE"
echo "RL_ARTIFACTS_ROOT=$RL_ARTIFACTS_ROOT"
echo "TRAIN_PROFILES_LIST=$TRAIN_PROFILES_LIST"
echo "NUM_SATELLITES=$NUM_SATELLITES"
echo "PARTITION=$PARTITION CPUS=$CPUS MEM=$MEM TIME=$TIME"
echo "JVMS_PER_NODE=$JVMS_PER_NODE RUNNERS_PER_JVM=$RUNNERS_PER_JVM"
echo ""

SUBMITTED=0
for i in $(seq 1 "$NUM_SATELLITES"); do
    SAT_JOB_NAME="sat-${GPU_JOB_ID}-${i}"
    SAT_OUT="$REPO_ROOT/local-training/hpc/bundles/${SAT_JOB_NAME}_%j.out"
    SAT_ERR="$REPO_ROOT/local-training/hpc/bundles/${SAT_JOB_NAME}_%j.err"

    JOB_ID=$(sbatch \
        --job-name="$SAT_JOB_NAME" \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --account=msml603-class \
        --output="$SAT_OUT" \
        --error="$SAT_ERR" \
        --dependency="after:${GPU_JOB_ID}" \
        --export="ALL,GPU_SERVICE_ENDPOINT=${GPU_SERVICE_ENDPOINT},MAGE_RL_RUNTIME_TARBALL=${BUNDLE},RL_ARTIFACTS_ROOT=${RL_ARTIFACTS_ROOT},JVMS_PER_NODE=${JVMS_PER_NODE},RUNNERS_PER_JVM=${RUNNERS_PER_JVM},TRAIN_PROFILES_LIST=${TRAIN_PROFILES_LIST},SOURCE_REPO_ROOT=${REPO_ROOT}" \
        "$REPO_ROOT/scripts/hpc/cpu_worker.sh" 2>&1)

    echo "Satellite $i: $JOB_ID"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted $SUBMITTED satellite jobs targeting GPU job $GPU_JOB_ID"
echo "Monitor: squeue -u \$USER"
echo "GPU metrics: curl http://${GPU_NODE}:$((${GPU_SERVICE_ENDPOINT##*:} + 1000))/metrics 2>/dev/null | grep -E 'duty|throughput|pending'"
