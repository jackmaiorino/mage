#!/usr/bin/env bash
set -euo pipefail

# Run this on a Zaratan login node from any repo checkout path.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

BUNDLE_DIR="${BUNDLE_DIR:-$repo_root/local-training/hpc/bundles}"
BUNDLE="${BUNDLE:-$(ls -1t "$BUNDLE_DIR"/rl-runtime-*.tar.gz 2>/dev/null | head -n 1 || true)}"
if [[ -z "${BUNDLE}" ]]; then
  echo "No runtime bundle found under $BUNDLE_DIR" >&2
  echo "Build/upload one first with scripts/hpc/build_rl_runtime_bundle.ps1" >&2
  exit 1
fi

echo "Using bundle: ${BUNDLE}"

python3 scripts/hpc/spy_saturation.py submit \
  --bundle "${BUNDLE}" \
  --tag "${SAT_TAG:-zaratan-single-node}" \
  --partition "${SAT_PARTITION:-gpu-a100}" \
  --gres "${SAT_GRES:-gpu:a100:2}" \
  --mem "${SAT_MEM:-128G}" \
  --time "${SAT_TIME:-02:00:00}" \
  --train-profiles "${SAT_TRAIN_PROFILES:-2,4}" \
  --cpus-per-task "${SAT_CPUS_PER_TASK:-64,128,192}" \
  --runner-oversubscription-factor "${SAT_RUNNER_OVERSUBSCRIPTION_FACTOR:-8,12}" \
  --infer-workers "${SAT_INFER_WORKERS:-1}" \
  --cpu-headroom "${SAT_CPU_HEADROOM:-0}" \
  --trainer-start-stagger-seconds "${SAT_TRAINER_START_STAGGER_SECONDS:-45}" \
  "$@"
