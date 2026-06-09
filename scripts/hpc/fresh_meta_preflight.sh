#!/bin/bash
# Preflight for the fresh-on-meta fleet run. Run ON HPC before submitting fm-head.
# Stages a FRESH (random) Spy + the 3 trained opponent models into a fixed shared
# RL_ARTIFACTS_ROOT, exports ONNX for all 4 (so the multi-profile self-serve router is
# ready at satellite startup -- avoids the ONNX-gap deadlock). Verifies .active_dir x4.
set -uo pipefail
REPO="/scratch/zt1/project/msml603/user/jmaior/jmaior/mage"
cd "$REPO"
AROOT="$REPO/local-training/fresh_meta_run/rl_artifacts"
VENV="$REPO/.mtgrl_venv_hpc/bin/python"
MLP="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode"
CKPROF="$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
SPY=Pauper-Spy-Combo-Value
OPPS="Pauper-Wildfire-Value Pauper-Rally-Anchor-Value Pauper-Affinity-Anchor-Value"

echo "=== preflight: artifacts root $AROOT ==="
mkdir -p "$AROOT/profiles/$SPY/models"

# 1. opponent models (continue from trained) -- copy from checkout if present
miss=0
for P in $OPPS; do
  mkdir -p "$AROOT/profiles/$P/models"
  if [ -f "$CKPROF/$P/models/model_latest.pt" ]; then
    cp -f "$CKPROF/$P/models/model_latest.pt" "$AROOT/profiles/$P/models/model_latest.pt"
    cp -f "$CKPROF/$P/models/model.pt" "$AROOT/profiles/$P/models/model.pt" 2>/dev/null || true
    echo "opp $P: model copied"
  else
    echo "opp $P: MODEL MISSING on HPC checkout -> upload it first"; miss=1
  fi
done
[ "$miss" = 1 ] && { echo "ABORT: opponent models missing"; exit 2; }

# 2. FRESH Spy random-weight checkpoint (NO continue)
rm -f "$AROOT/profiles/$SPY/models/model_latest.pt" "$AROOT/profiles/$SPY/models/model.pt" 2>/dev/null || true
MODEL_PROFILE=$SPY MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 "$VENV" -c "
import sys
sys.path.insert(0,'$MLP')
from py4j_entry_point import PythonEntryPoint
ep=PythonEntryPoint(); ep.initializeModel()
ep.saveLatestModelAtomic('$AROOT/profiles/$SPY/models/model_latest.pt')
print('fresh Spy random checkpoint written')
" || { echo "ABORT: fresh Spy init failed"; exit 3; }

# 3. export ONNX for all 4 (one-shot)
CSV="$SPY,${OPPS// /,}"
ONNX_PUBLISH_ONCE=1 ONNX_PUBLISH_PROFILES="$CSV" RL_ARTIFACTS_ROOT="$AROOT" \
  MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 "$VENV" "$REPO/scripts/hpc/onnx_publisher.py" || true

# 4. verify .active_dir for all 4
echo "=== verify .active_dir ==="
ok=1
for P in $SPY $OPPS; do
  ad="$AROOT/profiles/$P/models/onnx/.active_dir"
  if [ -f "$ad" ]; then echo "$P: $(cat $ad)"; else echo "$P: .active_dir MISSING"; ok=0; fi
done
[ "$ok" = 1 ] && echo "=== PREFLIGHT OK ===" || { echo "=== PREFLIGHT FAILED ==="; exit 4; }
