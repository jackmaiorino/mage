#!/usr/bin/env bash
# On-pod: RALLY DECK-CONFOUND CONTROL (Codex #78 / user jul8): CP7-s7 piloting
# Mono Red Rally vs the 8-deck CP7-s7 gauntlet, n=128/matchup fast non-det.
# Isolates pilot_edge for the Rally flagship: agent-Rally 75.8% minus THIS number.
# No self-terminate: controller pulls results and deletes the pod.
set -uo pipefail
LOGD=/root/pod_logs; mkdir -p "$LOGD"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOGD/_rally.log"; }

log "=== [1/5] extract ==="
mkdir -p /root/mage && cd /root/mage
if [ ! -d Mage.Server.Plugins ]; then tar -xzf /root/mage-src-jul7.tar.gz; fi

log "=== [2/5] deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends openjdk-21-jdk maven python3 python3-pip rsync >/dev/null 2>&1 \
  || apt-get install -y -qq --no-install-recommends openjdk-17-jdk maven python3 python3-pip rsync >/dev/null 2>&1
pip3 install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
pip3 install -q --no-input numpy onnx onnxruntime py4j transformers >/dev/null 2>&1
log "java: $(java -version 2>&1 | head -1) cores: $(nproc)"

log "=== [3/5] mvn install ==="
mvn -q -U -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install >"$LOGD/_mvn.log" 2>&1
RC=$?; log "mvn rc=$RC"
[ $RC -ne 0 ] && { tail -30 "$LOGD/_mvn.log" | tee -a "$LOGD/_rally.log"; exit 1; }

log "=== [4/5] profile + registry (Rally agent deck) ==="
PROF=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Rally-Anchor-Value/models
mkdir -p "$PROF" local-training
# pilot mode never queries the model, but the sweep harness requires the file
cp /root/ckpts/affinity.pt "$PROF/model.pt"; cp /root/ckpts/affinity.pt "$PROF/model_latest.pt"
python3 - <<'PYEOF'
import json
gaunt='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
dk='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper'
ent=[{'profile':'Pauper-Rally-Anchor-Value','priority':1,'active':True,'train_enabled':False,
      'deck_path':gaunt,
      'train_env':{'RL_AGENT_DECK_LIST': dk+'/Deck - Mono Red Rally.dek',
                   'MODEL_D_MODEL':'128','MODEL_NUM_LAYERS':'2',
                   'GAME_TIMEOUT_SEC':'2400'}}]
json.dump(ent,open('local-training/_rally_ctl_reg.json','w'),indent=2)
print('registry written')
PYEOF

log "=== [5/5] CP7-pilot Rally vs gauntlet, n=128, parallel 8 ==="
export SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODEL_NHEAD=4 MODEL_DIM_FEEDFORWARD=512
export INFER_CUDA_DEVICE=cpu TRAIN_CUDA_DEVICE=cpu MULLIGAN_DEVICE=cpu
export RL_CP7_PILOT=1

# SANITY: 8 quick games vs burn; require wins>0 (CP7-Rally should beat CP7-Burn sometimes)
python3 scripts/run_cp7_eval_sweep.py --registry local-training/_rally_ctl_reg.json \
  --profiles Pauper-Rally-Anchor-Value --opponents burn --skill 7 \
  --games-per-matchup 8 --games-per-job 8 --parallel 1 \
  --gpu-port 26090 --gpu-metrics-port 26091 --skip-compile \
  --run-id sanity_rally --timeout-sec 3600 --allow-incomplete-results >"$LOGD/sanity_rally.log" 2>&1
SAN=$(python3 -c "
import csv
try:
    rows=list(csv.DictReader(open('local-training/local_pbt/cp7_eval_sweeps/sanity_rally/matchup_summary.csv')))
    print(sum(int(r['wins']) for r in rows))
except Exception: print(-1)")
log "SANITY: wins=$SAN (need >0)"
[ "$SAN" -le 0 ] && { log "SANITY FAILED"; tail -30 local-training/local_pbt/cp7_eval_sweeps/sanity_rally/logs/*.log | tee -a "$LOGD/_rally.log"; exit 2; }

python3 scripts/run_cp7_eval_sweep.py --registry local-training/_rally_ctl_reg.json \
  --profiles Pauper-Rally-Anchor-Value --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" \
  --skill 7 --games-per-matchup 128 --games-per-job 8 --parallel 8 \
  --skip-compile --run-id rally_cp7_control --timeout-sec 14400 \
  --allow-incomplete-results >"$LOGD/rally_control.log" 2>&1
log "sweep rc=$?"
log "=== RALLY CONTROL DONE; summary ==="
cat local-training/local_pbt/cp7_eval_sweeps/rally_cp7_control/matchup_summary.csv 2>/dev/null | tee -a "$LOGD/_rally.log"
tar -czf /root/rally_control_results.tar.gz local-training/local_pbt/cp7_eval_sweeps/rally_cp7_control "$LOGD" 2>/dev/null
log "ALL DONE -- waiting for controller"
