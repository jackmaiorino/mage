#!/usr/bin/env bash
# rally_skill7_chunk.sh - per-array-task body for the deterministic skill-7 Rally
# gauntlet eval. Invoked by scripts/hpc/rally_skill7_array.sbatch with
# SLURM_ARRAY_TASK_ID (0-7) selecting exactly one opponent deck from
# decklist.all_opponents.txt. Can also be run standalone (e.g. on a login-node
# allocation) by exporting SLURM_ARRAY_TASK_ID manually before calling this script.
#
# DETERMINISM CONSTRAINT (read before changing anything here):
# scripts/run_cp7_eval_sweep.py --deterministic-eval forces args.parallel=1 and
# single-channel/batch-size-1 inference (DETERMINISTIC_EVAL_ENV) so that replay
# seeds reproduce bit-identical games. That guarantee holds PER PROCESS: an
# in-process thread pool of parallel matchups (multiple --parallel workers, or
# multiple opponents sharing one gpu_service_host.py) reintroduces nondeterministic
# request-ordering into the shared GPU service and breaks reproducibility
# (confirmed empirically before this eval: intra-process parallel decisions
# flip on identical seeds). Running SEPARATE OS PROCESSES (one JVM + one
# gpu_service_host.py per task, on its own ports, own DB dir, own registry
# file) is fine -- there is no shared mutable inference-ordering state across
# processes. That is exactly what this Slurm ARRAY gives us for free: 8
# opponents x 1 task each x 1 JVM x 1 GPU-service-instance = 8 fully isolated,
# fully deterministic serial runs, running in parallel with each other because
# they are different processes/nodes, not because of in-process parallelism.
# Do NOT add --parallel > 1 or --allow-deterministic-parallel to the sweep
# invocation below, and do NOT point two tasks at the same --gpu-port /
# --registry / --run-id / MAGE_DB_DIR.
set -euo pipefail

REPO_ROOT="${SOURCE_REPO_ROOT:-/home/jmaior/scratch.msml603/jmaior/mage}"
cd "$REPO_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_JOB_ID:-manual_$$}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$JOB_ID}"

# --- opponent-per-task map: 8 tasks == 8 decks in decklist.all_opponents.txt ---
# Tokens are substring-matched (case-insensitive) against each deck's filename
# by run_cp7_eval_sweep.py's --opponents filter; each is unique in the pool.
OPPONENTS=(affinity burn rally faeries terror wildfire caw-gates elves)
if (( TASK_ID < 0 || TASK_ID >= ${#OPPONENTS[@]} )); then
    echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID out of range (expected 0-$(( ${#OPPONENTS[@]} - 1 )))" >&2
    exit 1
fi
OPPONENT="${OPPONENTS[$TASK_ID]}"

echo "=== rally_skill7_chunk starting on $(hostname) at $(date -u) ==="
echo "TASK_ID=$TASK_ID OPPONENT=$OPPONENT ARRAY_JOB_ID=$ARRAY_JOB_ID JOB_ID=$JOB_ID"
echo "REPO_ROOT=$REPO_ROOT"

module load java 2>/dev/null || true
java -version 2>&1 | head -1

# --- python: prefer the HPC venv (has torch, needed because run_cp7_eval_sweep.py
# spins up gpu_service_host.py as a subprocess of whatever interpreter runs it) ---
MTG_VENV_PATH="${MTG_VENV_PATH:-$REPO_ROOT/.mtgrl_venv_hpc}"
PYBIN="$MTG_VENV_PATH/bin/python"
if [[ ! -x "$PYBIN" ]]; then
    PYBIN="$(command -v python3 || true)"
fi
if [[ -z "$PYBIN" ]]; then
    echo "FATAL: no usable python (checked $MTG_VENV_PATH/bin/python and python3 on PATH)" >&2
    exit 1
fi
echo "PYBIN=$PYBIN"

# --- runtime bundle (compiled Java classpath) is REQUIRED for this eval.
# run_cp7_eval_sweep.py falls back to `mvn compile` on the checkout when
# MAGE_RL_RUNTIME_DIR is unset -- avoid that here: Java changed today
# (ComputerPlayerRL.java, GameHealthMonitor.java, MulliganLogger.java,
# PythonEntryPoint.java, PythonMLBatchManager.java, RLTrainer.java,
# StateSequenceBuilder.java, TerminalPrefixSearch.java) and 8 array tasks each
# running `mvn compile` cold on a standard-partition node is slow and racy.
# Rebuild+upload the bundle first: scripts/hpc/build_rl_runtime_bundle.ps1 ---
: "${MAGE_RL_RUNTIME_TARBALL:=}"
if [[ -z "${MAGE_RL_RUNTIME_DIR:-}" ]]; then
    if [[ -z "$MAGE_RL_RUNTIME_TARBALL" ]]; then
        MAGE_RL_RUNTIME_TARBALL="$(ls -1t "$REPO_ROOT"/local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1 || true)"
    fi
    if [[ -z "$MAGE_RL_RUNTIME_TARBALL" || ! -f "$MAGE_RL_RUNTIME_TARBALL" ]]; then
        echo "FATAL: no runtime bundle found. Set MAGE_RL_RUNTIME_TARBALL or MAGE_RL_RUNTIME_DIR," >&2
        echo "       or build+upload one: scripts/hpc/build_rl_runtime_bundle.ps1 (must include" >&2
        echo "       today's Java changes -- see header comment for the file list)." >&2
        exit 1
    fi
    RTDIR="/tmp/${USER}/rally-skill7-${JOB_ID}"
    mkdir -p "$RTDIR"
    tar xzf "$MAGE_RL_RUNTIME_TARBALL" -C "$RTDIR"
    MAGE_RL_RUNTIME_DIR="$(ls -1d "$RTDIR"/rl-runtime-* 2>/dev/null | head -1)"
    if [[ -z "$MAGE_RL_RUNTIME_DIR" ]]; then
        echo "FATAL: tarball extraction produced no rl-runtime-* dir in $RTDIR" >&2
        exit 1
    fi
fi
export MAGE_RL_RUNTIME_DIR
echo "MAGE_RL_RUNTIME_DIR=$MAGE_RL_RUNTIME_DIR"

# --- checkpoint sanity check: fail fast instead of silently evaluating a
# missing/stale model. The Rally checkpoint must be uploaded to this path in
# the remote checkout BEFORE submitting (it is not produced by this script).
RALLY_MODELS_DIR="$REPO_ROOT/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Rally-Anchor-Value/models"
if [[ ! -f "$RALLY_MODELS_DIR/model.pt" ]]; then
    echo "FATAL: missing $RALLY_MODELS_DIR/model.pt -- upload the Rally checkpoint" >&2
    echo "       you intend to gate (e.g. the armF-era model backed up locally at" >&2
    echo "       E:\\mage-training\\backups\\rally_latest_armF_20260630\\model.pt) before submitting." >&2
    exit 1
fi

# --- base registry: pauper_spy_pbt_registry.json is git-tracked so it exists in
# the checkout; local-training/_brew_win_registry.json (what
# run_rally_baseline_det.ps1 used as its base) is gitignored/local-only and will
# NOT be present remotely -- ASSUMPTION: the checked-in registry's Rally entry is
# equivalent for our purposes (same train_env; only deck_path differs and we
# overwrite that below anyway). Verify this matches the intended arm if the
# local registry has diverged. ---
BASE_REGISTRY="$REPO_ROOT/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
RESULTS_ROOT="$REPO_ROOT/local-training/hpc/rally_skill7_array"
mkdir -p "$RESULTS_ROOT"
TASK_REGISTRY="$RESULTS_ROOT/_registry_${ARRAY_JOB_ID}_${TASK_ID}.json"
OPPONENT_POOL_DECK="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
if [[ ! -f "$REPO_ROOT/$OPPONENT_POOL_DECK" ]]; then
    echo "FATAL: missing $OPPONENT_POOL_DECK in the checkout -- it is untracked/new" >&2
    echo "       locally and must be uploaded (see header comment on the sbatch file)." >&2
    exit 1
fi

"$PYBIN" -c "
import json
reg = json.load(open(r'$BASE_REGISTRY'))
rows = reg if isinstance(reg, list) else reg.get('profiles', reg)
for e in (rows if isinstance(rows, list) else []):
    if e.get('profile') == 'Pauper-Rally-Anchor-Value':
        e['deck_path'] = '$OPPONENT_POOL_DECK'
json.dump(reg, open(r'$TASK_REGISTRY', 'w'), indent=2)
print('wrote $TASK_REGISTRY deck_path=$OPPONENT_POOL_DECK')
"

# --- per-task isolation: own DB dir, own GPU-service ports, own results dir ---
export MAGE_DB_DIR="${SLURM_TMPDIR:-/tmp/$USER}/rally-skill7-${JOB_ID}/db"
mkdir -p "$MAGE_DB_DIR"
export MAGE_DB_AUTO_SERVER=false
export CUDA_VISIBLE_DEVICES=""   # standard partition: force CPU-only ONNX/torch inference
GPU_PORT=$((26200 + TASK_ID))
GPU_METRICS_PORT=$((27200 + TASK_ID))
RUN_ID="rally_det_skill7_${OPPONENT}_${ARRAY_JOB_ID}"
GAMES_PER_JOB="${GAMES_PER_JOB:-16}"   # matches the chunking run_rally_baseline_det.ps1 uses

echo "OPPONENT=$OPPONENT GPU_PORT=$GPU_PORT GPU_METRICS_PORT=$GPU_METRICS_PORT RUN_ID=$RUN_ID"
echo "TASK_REGISTRY=$TASK_REGISTRY"
echo "RESULTS_ROOT=$RESULTS_ROOT"

"$PYBIN" "$REPO_ROOT/scripts/run_cp7_eval_sweep.py" \
    --registry "$TASK_REGISTRY" \
    --profiles Pauper-Rally-Anchor-Value \
    --opponents "$OPPONENT" \
    --skill 7 \
    --games-per-matchup 256 \
    --games-per-job "$GAMES_PER_JOB" \
    --deterministic-eval \
    --skip-compile \
    --replay-seed-base 5151 \
    --run-id "$RUN_ID" \
    --output-root "$RESULTS_ROOT/results" \
    --gpu-port "$GPU_PORT" \
    --gpu-metrics-port "$GPU_METRICS_PORT" \
    2>&1 | tee "$RESULTS_ROOT/${RUN_ID}.log"

echo "=== rally_skill7_chunk done for OPPONENT=$OPPONENT at $(date -u) ==="
