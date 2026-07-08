#!/usr/bin/env bash
# On-pod: provision + run ONE small eval job with FULL logging to diagnose the
# jul6 0-games failure (whose logs were discarded by `| grep wr=`).
# Run: bash pod_eval_diag.sh   (expects /root/mage-src.tar.gz, /root/mage-target-fix.tar.gz,
#                               /root/ckpts/affinity.pt already uploaded)
set -uo pipefail
LOGD=/root/pod_logs; mkdir -p "$LOGD"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOGD/_diag.log"; }

log "=== [1/5] extract ==="
mkdir -p /root/mage && cd /root/mage
if [ ! -d Mage.Server.Plugins ]; then
  tar -xzf /root/mage-src.tar.gz
  # target-fix: restores the mage/target SOURCE package the original tar clobbered
  tar -xzf /root/mage-target-fix.tar.gz -C Mage/src/main/java/mage/ 2>/dev/null \
    || tar -xzf /root/mage-target-fix.tar.gz
fi
log "extracted: $(ls | head -5 | tr '\n' ' ')"

log "=== [2/5] apt + pip deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends openjdk-21-jdk maven python3 python3-pip rsync >/dev/null 2>&1 \
  || apt-get install -y -qq --no-install-recommends openjdk-17-jdk maven python3 python3-pip rsync >/dev/null 2>&1
log "java: $(java -version 2>&1 | head -1)  cores: $(nproc)"
pip3 install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
# transformers is REQUIRED: mtg_transformer.py imports HuggingFace AutoModel at module
# level; omitting it was the jul6 0-games root cause (profile registration failed).
pip3 install -q --no-input numpy onnx onnxruntime py4j transformers >/dev/null 2>&1
log "torch: $(python3 -c 'import torch;print(torch.__version__)' 2>&1)"

log "=== [3/5] mvn install (AIRL + deps) ==="
mvn -q -U -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install >"$LOGD/_mvn.log" 2>&1
RC=$?
log "mvn rc=$RC"
if [ $RC -ne 0 ]; then tail -30 "$LOGD/_mvn.log" | tee -a "$LOGD/_diag.log"; exit 1; fi

log "=== [4/5] profile + registry ==="
PROF=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models
mkdir -p "$PROF" local-training
cp /root/ckpts/affinity.pt "$PROF/model.pt"
cp /root/ckpts/affinity.pt "$PROF/model_latest.pt"
python3 - <<'PYEOF'
import json
gaunt='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
dk='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper'
ent=[{'profile':'Pauper-Affinity-Anchor-Value','priority':1,'active':True,'train_enabled':False,
      'deck_path':gaunt,
      'train_env':{'RL_AGENT_DECK_LIST': dk+'/Deck - Grixis Affinity.dek',
                   'MODEL_D_MODEL':'128','MODEL_NUM_LAYERS':'2'}}]
json.dump(ent,open('local-training/_diag_reg.json','w'),indent=2)
print('registry written')
PYEOF

log "=== [5/5] ONE eval job, 4 games, FULL logging ==="
export SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODEL_NHEAD=4 MODEL_DIM_FEEDFORWARD=512
python3 scripts/run_cp7_eval_sweep.py --registry local-training/_diag_reg.json \
  --profiles Pauper-Affinity-Anchor-Value --opponents burn --skill 7 \
  --games-per-matchup 4 --games-per-job 4 --deterministic-eval --skip-compile \
  --replay-seed-base 5151 --run-id diag_burn >"$LOGD/diag_sweep.log" 2>&1
RC=$?
log "sweep rc=$RC"
log "--- sweep stdout tail ---"
tail -30 "$LOGD/diag_sweep.log" | tee -a "$LOGD/_diag.log"
RUND=local-training/local_pbt/cp7_eval_sweeps/diag_burn
log "--- job logs ---"
for f in "$RUND"/logs/*.log; do
  log "### $f (tail 60):"
  tail -60 "$f" | tee -a "$LOGD/_diag.log"
done
log "--- results ---"
cat "$RUND"/matchups.csv 2>/dev/null | tee -a "$LOGD/_diag.log"
log "DIAG COMPLETE"
