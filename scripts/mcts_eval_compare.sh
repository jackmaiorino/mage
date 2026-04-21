#!/usr/bin/env bash
# Runs two eval batches — baseline vs MCTS — and prints winrate comparison.
# Assumes a model checkpoint exists at profiles/Pauper-Standard/models/model_latest.pt.
# Safe to run standalone (doesn't touch training_stats.csv in training run).
#
# Run this AFTER pausing or alongside training. Uses hybrid ONNX so no GPU
# service is required. CUDA_VISIBLE_DEVICES=0 shares the GPU with training.
#
# Usage:   bash scripts/mcts_eval_compare.sh
# Env:     EVAL_GAMES_PER_RUNNER (default 10), EVAL_RUNNERS (default 4)

set -euo pipefail

cd "$(dirname "$0")/.."

RUNNERS="${EVAL_RUNNERS:-4}"
GAMES_PER_RUNNER="${EVAL_GAMES_PER_RUNNER:-10}"
PROFILE="${MODEL_PROFILE:-Pauper-Standard}"
DECK_POOL="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.standard_pool.txt"
OUT_DIR="local-training/local_pbt/mcts_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

COMMON_ENV=(
    MODE=eval
    MODEL_PROFILE="$PROFILE"
    DECK_LIST_FILE="$DECK_POOL"
    RL_AGENT_DECK_LIST="$DECK_POOL"
    NUM_GAME_RUNNERS="$RUNNERS"
    EVAL_EPISODES="$GAMES_PER_RUNNER"
    PY_SERVICE_MODE=hybrid
    USE_TRT_INFERENCE=0
    EVAL_CP7_SKILL=7
    METRICS_PORT=9391
)

run_eval() {
    local label="$1"
    shift
    local logfile="$OUT_DIR/$label.log"
    echo "[$label] starting — logging to $logfile"
    env "${COMMON_ENV[@]}" "$@" \
        mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests \
            exec:java -Dexec.mainClass=mage.player.ai.rl.RLTrainer \
            -Dexec.args=eval > "$logfile" 2>&1 || echo "[$label] completed with non-zero exit"
    local winrate
    winrate=$(grep -oE "Evaluation win rate: [0-9.]+" "$logfile" | tail -1 | sed 's/Evaluation win rate: //')
    winrate=${winrate:-"?"}
    echo "[$label] winrate=$winrate"
    grep -aE "Evaluation win rate|Eval progress|per-deck|archetype" "$logfile" | tail -10
}

# 1. Baseline: no MCTS
run_eval baseline ISMCTS_ENABLE=0 MCTS_TRAINING_ENABLE=0

# 2. MCTS 1-ply (fast): rollout_depth=0 disables engine rollouts, uses only value net leaf eval
run_eval mcts_1ply \
    ISMCTS_ENABLE=1 MCTS_TRAINING_ENABLE=0 \
    MCTS_ITERATIONS=8 MCTS_DETERMINIZATIONS=4 \
    MCTS_ROLLOUT_DEPTH=0 \
    MCTS_PARALLEL_ROLLOUTS=4 \
    MCTS_SKIP_TOP_PROB=0.80

# 3. (Optional, slow) MCTS with truncated engine rollouts
# Uncomment when GPU is free.
# run_eval mcts_rollout \
#     ISMCTS_ENABLE=1 MCTS_TRAINING_ENABLE=0 \
#     MCTS_ITERATIONS=6 MCTS_DETERMINIZATIONS=3 \
#     MCTS_ROLLOUT_DEPTH=2 MCTS_ROLLOUT_TIMEOUT_MS=500 \
#     MCTS_PARALLEL_ROLLOUTS=4 \
#     MCTS_SKIP_TOP_PROB=0.80

echo
echo "=== Done. Logs in $OUT_DIR ==="
