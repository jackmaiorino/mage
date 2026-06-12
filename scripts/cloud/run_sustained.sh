#!/usr/bin/env bash
# Chunked sustained training on a rented Linux box -- bash port of
# scripts/run_overnight_dominance.ps1 (same gates, checkpoints, best-tracking).
#
# Env knobs (defaults in parens):
#   NUM_CHUNKS (12)  CHUNK_MIN (45)  RUN_ENTROPY_END (0.03)
#   START_MODEL (local-training/backups/spy_value_baseline_20260531/model_latest.pt)
#   CKPT_DIR (local-training/backups/sustained_cloud)
#   RUNNERS (64)  EVAL_GAMES (96)
#
# Run inside tmux: tmux new -s train 'bash scripts/cloud/run_sustained.sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYBIN="${PYBIN:-.mtgrl_venv/bin/python}"
OUT="local-training/sustained_RESULT.log"
REG="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
MD="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Spy-Combo-Value/models"
GL="local-training/local_pbt/cp7_eval_sweeps"
NUM_CHUNKS="${NUM_CHUNKS:-12}"
CHUNK_MIN="${CHUNK_MIN:-45}"
RUNNERS="${RUNNERS:-64}"
EVAL_GAMES="${EVAL_GAMES:-96}"
START_MODEL="${START_MODEL:-local-training/backups/spy_value_baseline_20260531/model_latest.pt}"
CKPT_DIR="${CKPT_DIR:-local-training/backups/sustained_cloud}"
mkdir -p "$CKPT_DIR" local-training

kill_train() {
  pkill -f 'gpu_service_host' 2>/dev/null
  pkill -f 'run_local_pbt' 2>/dev/null
  pkill -f 'RLTrainer' 2>/dev/null
  pkill -f 'exec:java' 2>/dev/null
  sleep 5
}

eval_now() { # $1=run_id $2=seed
  SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 PY_SERVICE_MODE= \
  "$PYBIN" scripts/run_cp7_eval_sweep.py --registry "$REG" --profiles Pauper-Spy-Combo-Value \
    --opponents grixis --skill 1 --games-per-matchup "$EVAL_GAMES" --parallel 8 \
    --eval-game-logging --replay-metadata --skip-compile \
    --replay-seed-base "$2" --run-id "$1" 2>&1 | grep "wr=" >> "$OUT"
  "$PYBIN" scripts/mtgrl/orchestration_metric.py "$GL/$1" --label "$1" 2>&1 | sed 's/^/>>> /' >> "$OUT"
}

get_wr() { # $1=run_id -> prints wr or -1
  "$PYBIN" - "$GL/$1/matchups.csv" <<'EOF'
import csv, sys
try:
    with open(sys.argv[1]) as f:
        r = next(csv.DictReader(f))
    print(int(r['wins']) / int(r['total']))
except Exception:
    print(-1)
EOF
}

export_onnx() {
  local stamp="v$(date -u +%Y%m%dT%H%M%S)_000000"
  local dest="$MD/onnx/$stamp"
  mkdir -p "$dest"
  if ONNX_EXPORT_FP16=0 "$PYBIN" \
      Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export.py \
      --model-path "$MD/model_latest.pt" --output-dir "$dest" >> "$OUT" 2>&1 \
      && [[ -f "$dest/model_action.onnx" ]]; then
    printf '3' > "$dest/.export_version"
    printf '%s' "$stamp" > "$MD/onnx/.active_dir"
    echo "onnx exported -> $stamp" >> "$OUT"
  else
    echo "WARNING: onnx export failed; keeping previous .active_dir" >> "$OUT"
  fi
}

echo "=== SUSTAINED CLOUD RUN started $(date) ; ${NUM_CHUNKS}x${CHUNK_MIN}min ; diet=CP7-skill-1 ===" > "$OUT"
kill_train
cp -f "$START_MODEL" "$MD/model_latest.pt"
cp -f "$START_MODEL" "$MD/model.pt"
echo "starting model: $START_MODEL" >> "$OUT"

train_env() {
  export SEARCH_OP_ENABLE=0 MCTS_TRAINING_ENABLE=0 MULTI_PLY_MCTS=0 CANDIDATE_Q_ONLY=0
  unset CANDIDATE_Q_LOSS_COEF CANDIDATE_Q_FROM_MCTS_TARGETS CANDIDATE_Q_MCTS_SIGNED_TARGETS \
        CANDIDATE_Q_BLEND CANDIDATE_Q_DUMP_DIR CANDIDATE_Q_DETACH_ENCODER \
        SEARCH_OP_APPLY_OVERRIDE SEARCH_OP_ARBITER_CAST_FILTER WORLD_MODEL_LOSS_COEF \
        REFERENCE_POLICY_KL_COEF PY_SERVICE_MODE 2>/dev/null || true
  export OPPONENT_SAMPLER=ladder LADDER_SKILLS=1 LADDER_MIX_LOWER_P=0.0 LEAGUE_MODE=
  export ENTROPY_START=0.25 ENTROPY_END="${RUN_ENTROPY_END:-0.03}" ENTROPY_DECAY_STEPS=100000
  export ONNX_BATCH_TIMEOUT_MS=25 ONNX_BATCH_TIMEOUT_MAX_MS=50
  export ONNX_EXPORT_ENABLE=1
  export TRAIN_PROFILES=1 NUM_GAME_RUNNERS="$RUNNERS" TOTAL_EPISODES=99999999
  # single-GPU rental boxes: train + infer share device 0
  export TRAIN_CUDA_DEVICE="${TRAIN_CUDA_DEVICE:-cuda:0}" INFER_CUDA_DEVICE="${INFER_CUDA_DEVICE:-cuda:0}"
}

best_wr=0
low_streak=0
for (( c=1; c<=NUM_CHUNKS; c++ )); do
  train_env
  export_onnx
  echo "=== chunk $c train start $(date) ===" >> "$OUT"
  "$PYBIN" scripts/run_local_pbt.py > local-training/sustained_train.log 2>&1 &
  sleep $(( CHUNK_MIN * 60 ))
  kill_train
  if (( c == 1 || c == 2 || c % 2 == 0 )); then
    eval_now "sc_c$c" 5151
    wr=$(get_wr "sc_c$c")
    cp -f "$MD/model_latest.pt" "$CKPT_DIR/model_after_c$c.pt"
    better=$("$PYBIN" -c "print(1 if $wr > $best_wr else 0)")
    if [[ "$better" == "1" ]]; then
      best_wr="$wr"
      cp -f "$MD/model_latest.pt" "$CKPT_DIR/model_best.pt"
      echo "new best wr=$wr at chunk $c (model_best.pt updated)" >> "$OUT"
    fi
    is_low=$("$PYBIN" -c "print(1 if 0 <= $wr < 0.42 else 0)")
    if [[ "$is_low" == "1" ]]; then low_streak=$(( low_streak + 1 )); else low_streak=0; fi
    if (( low_streak >= 2 )); then
      echo "=== ABORT at chunk $c: 2 consecutive evals < 0.42. Autopsy saved; best restored. ===" >> "$OUT"
      cp -f "$MD/model_latest.pt" "$CKPT_DIR/model_aborted_c$c.pt"
      if [[ -f "$CKPT_DIR/model_best.pt" ]]; then
        cp -f "$CKPT_DIR/model_best.pt" "$MD/model_latest.pt"
        cp -f "$CKPT_DIR/model_best.pt" "$MD/model.pt"
      fi
      break
    fi
  fi
done
echo "=== SUSTAINED CLOUD RUN DONE $(date) ; best_wr=$best_wr ; checkpoints in $CKPT_DIR ===" >> "$OUT"
