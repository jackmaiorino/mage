#!/usr/bin/env bash
# One-shot provisioning for a rented Linux GPU box (RunPod / Vast.ai / Lambda).
# Assumes: Ubuntu 22.04/24.04 image with NVIDIA driver + CUDA runtime (standard
# on GPU rental images), run from the repo root after unpacking the source
# tarball (or git clone).
#
#   bash scripts/cloud/provision_rental_box.sh [--models-tarball mage-models.tar.gz]
#
# After it finishes:
#   bash scripts/cloud/run_sustained.sh        # chunked training (see header)
set -euo pipefail

MODELS_TARBALL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-tarball) MODELS_TARBALL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "=== [1/5] apt dependencies ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# JDK: 21 preferred, 17 fallback (both build this repo)
apt-get install -y -qq --no-install-recommends openjdk-21-jdk maven git tmux curl \
    python3 python3-venv python3-pip unzip rsync \
  || apt-get install -y -qq --no-install-recommends openjdk-17-jdk maven git tmux curl \
    python3 python3-venv python3-pip unzip rsync

echo "=== [2/5] GPU sanity ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "WARNING: nvidia-smi failed -- is this a GPU instance?"; }

echo "=== [3/5] python venv (torch cu121 wheels: works on 4090/A100) ==="
bash scripts/hpc/build_venv.sh --venv-path .mtgrl_venv --python python3

echo "=== [4/5] models ==="
if [[ -n "$MODELS_TARBALL" && -f "$MODELS_TARBALL" ]]; then
  tar xzf "$MODELS_TARBALL"
  echo "models unpacked from $MODELS_TARBALL"
else
  echo "no models tarball given -- profiles will fresh-init unless models are already in place"
fi

echo "=== [5/5] java build ==="
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile

echo "=== smoke test: 8-game eval (verifies engine + GPU + eval harness) ==="
PYBIN=".mtgrl_venv/bin/python"
"$PYBIN" scripts/run_cp7_eval_sweep.py \
  --registry Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json \
  --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 \
  --games-per-matchup 8 --parallel 4 --skip-compile --replay-seed-base 5151 \
  --run-id provision_smoke 2>&1 | grep -E "wr=" || {
    echo "SMOKE TEST FAILED -- inspect before launching long runs"; exit 1; }
echo "=== PROVISIONED OK ==="
