#!/usr/bin/env bash
# On-pod: deterministic n=256/matchup Affinity baseline vs the 8-deck CP7-s7 gauntlet.
# 8 concurrent per-matchup sweeps, each internally serial (deterministic). Seed block
# 5151 with --seed-key-mode matchup so later candidate arms pair per-chunk.
# Full logging everywhere. Self-terminates after a grace window.
# Run: bash pod_det_baseline.sh <POD_ID>
set -uo pipefail
POD_ID="${1:?POD_ID required}"
LOGD=/root/pod_logs; mkdir -p "$LOGD"
KEYFILE=/root/.runpod_api_key
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOGD/_det.log"; }

log "=== [1/5] extract ==="
mkdir -p /root/mage && cd /root/mage
if [ ! -d Mage.Server.Plugins ]; then tar -xzf /root/mage-src-jul7.tar.gz; fi
ls db/cards.h2.mv.db >/dev/null 2>&1 || { cp db_eval/cards.h2.mv.db db/ 2>/dev/null || log "WARN: no card db in bundle"; }

log "=== [2/5] deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends openjdk-21-jdk maven python3 python3-pip rsync >/dev/null 2>&1 \
  || apt-get install -y -qq --no-install-recommends openjdk-17-jdk maven python3 python3-pip rsync >/dev/null 2>&1
pip3 install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
# transformers REQUIRED (module-level HF import in mtg_transformer.py; jul6 0-games root cause)
pip3 install -q --no-input numpy onnx onnxruntime py4j transformers >/dev/null 2>&1
log "java: $(java -version 2>&1 | head -1) cores: $(nproc) torch: $(python3 -c 'import torch;print(torch.__version__)' 2>&1)"

log "=== [3/5] mvn install ==="
mvn -q -U -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install >"$LOGD/_mvn.log" 2>&1
RC=$?; log "mvn rc=$RC"
[ $RC -ne 0 ] && { tail -30 "$LOGD/_mvn.log" | tee -a "$LOGD/_det.log"; exit 1; }

log "=== [4/5] profile + registry ==="
PROF=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models
mkdir -p "$PROF" local-training
cp /root/ckpts/affinity.pt "$PROF/model.pt"; cp /root/ckpts/affinity.pt "$PROF/model_latest.pt"
python3 - <<'PYEOF'
import json
gaunt='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
dk='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper'
ent=[{'profile':'Pauper-Affinity-Anchor-Value','priority':1,'active':True,'train_enabled':False,
      'deck_path':gaunt,
      'train_env':{'RL_AGENT_DECK_LIST': dk+'/Deck - Grixis Affinity.dek',
                   'MODEL_D_MODEL':'128','MODEL_NUM_LAYERS':'2',
                   'GAME_TIMEOUT_SEC':'2400'}}]
json.dump(ent,open('local-training/_det_reg.json','w'),indent=2)
print('registry written')
PYEOF

log "=== [5/5] 8 concurrent deterministic matchup sweeps, n=256 ==="
export SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODEL_NHEAD=4 MODEL_DIM_FEEDFORWARD=512
# CPU-only pod: the sweep's gpu-service launcher defaults INFER_CUDA_DEVICE=cuda:0,
# which asserts on CPU torch -> every score fails -> agent force-passes -> 0% winrate
# (jul8 incident: 2000 garbage games). Force cpu everywhere.
export INFER_CUDA_DEVICE=cpu TRAIN_CUDA_DEVICE=cpu MULLIGAN_DEVICE=cpu

# SANITY GATE: one quick 8-game non-det chunk vs caw; require >0 wins and no
# force-pass storm before burning hours on the full deterministic grid.
python3 scripts/run_cp7_eval_sweep.py --registry local-training/_det_reg.json \
  --profiles Pauper-Affinity-Anchor-Value --opponents caw --skill 7 \
  --games-per-matchup 8 --games-per-job 8 --parallel 1 \
  --gpu-port 26090 --gpu-metrics-port 26091 --skip-compile \
  --run-id sanity_caw --timeout-sec 3600 --allow-incomplete-results >"$LOGD/sanity_caw.log" 2>&1
SAN_WINS=$(python3 -c "
import csv
try:
    rows=list(csv.DictReader(open('local-training/local_pbt/cp7_eval_sweeps/sanity_caw/matchup_summary.csv')))
    print(sum(int(r['wins']) for r in rows))
except Exception:
    print(-1)")
FP=$(grep -c 'forcing pass' local-training/local_pbt/cp7_eval_sweeps/sanity_caw/logs/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
log "SANITY: wins=$SAN_WINS force_pass_lines=$FP (need wins>0 and force_pass==0)"
if [ "$SAN_WINS" -le 0 ] || [ "$FP" -gt 0 ]; then
  log "SANITY FAILED -- aborting before the det grid. Inspect $LOGD/sanity_caw.log"
  tail -20 local-training/local_pbt/cp7_eval_sweeps/sanity_caw/logs/*.log | tee -a "$LOGD/_det.log"
  exit 2
fi
OPPS=(grixis burn faeries terror wildfire caw elves rally)
i=0
for opp in "${OPPS[@]}"; do
  port=$((26100+i*4))
  ( python3 scripts/run_cp7_eval_sweep.py --registry local-training/_det_reg.json \
      --profiles Pauper-Affinity-Anchor-Value --opponents "$opp" --skill 7 \
      --games-per-matchup 256 --games-per-job 16 \
      --gpu-port "$port" --gpu-metrics-port $((port+1)) \
      --deterministic-eval --skip-compile \
      --replay-seed-base 5151 --seed-key-mode matchup \
      --run-id "det_affinity_${opp}" --timeout-sec 14400 --allow-incomplete-results \
      >"$LOGD/det_${opp}.log" 2>&1
    echo "rc=$? opp=$opp" >>"$LOGD/_det.log" ) &
  i=$((i+1)); sleep 45
done
log "all 8 sweeps launched; waiting..."
wait
log "=== DONE; summaries ==="
for opp in "${OPPS[@]}"; do
  S=local-training/local_pbt/cp7_eval_sweeps/det_affinity_${opp}/matchup_summary.csv
  [ -f "$S" ] && { echo "== $opp"; cat "$S"; } | tee -a "$LOGD/_det.log"
done
tar -czf /root/det_results.tar.gz local-training/local_pbt/cp7_eval_sweeps/det_affinity_* "$LOGD" 2>/dev/null
log "results tarred -> /root/det_results.tar.gz"
# NO self-terminate: a zombie grace-timer from a superseded run deleted the pod under a
# healthy relaunch (jul8). The controller pulls results and deletes the pod via API,
# with its own failsafe timer.
log "ALL DONE -- waiting for controller to pull results and terminate the pod"
