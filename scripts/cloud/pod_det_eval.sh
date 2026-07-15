#!/usr/bin/env bash
# On-pod: parameterized deterministic gauntlet eval. Reads /root/eval_params.env:
#   EVAL_PROFILE   registry profile name (models loaded from /root/ckpts/model.pt)
#   EVAL_AGENT_DECK  path fragment under decks/Pauper (e.g. "Deck - Mono Red Rally.dek")
#   EVAL_OPPONENTS   comma list for --opponents
#   EVAL_PILOT       1 = CP7-s7 pilots the agent deck (RL model unused), 0 = RL agent plays
#   EVAL_N           games per matchup (default 256)
#   EVAL_PREFIX      run-id prefix
# No self-terminate: controller pulls /root/det_results.tar.gz and deletes the pod.
set -uo pipefail
LOGD=/root/pod_logs; mkdir -p "$LOGD"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOGD/_det.log"; }
# set -a: auto-export everything the params file defines (the python heredoc below
# reads them via os.environ; a bare `source` leaves them shell-local -> KeyError)
set -a
source /root/eval_params.env
EVAL_N="${EVAL_N:-256}"
set +a
export EVAL_PROFILE EVAL_AGENT_DECK EVAL_OPPONENTS EVAL_PILOT EVAL_N EVAL_PREFIX

log "=== [1/5] extract (profile=$EVAL_PROFILE pilot=$EVAL_PILOT deck=$EVAL_AGENT_DECK) ==="
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
RC=$?; log "mvn rc=$RC"; [ $RC -ne 0 ] && { tail -30 "$LOGD/_mvn.log" | tee -a "$LOGD/_det.log"; exit 1; }

log "=== [4/5] profile + registry ==="
PROF="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$EVAL_PROFILE/models"
mkdir -p "$PROF" local-training
cp /root/ckpts/model.pt "$PROF/model.pt"; cp /root/ckpts/model.pt "$PROF/model_latest.pt"
python3 - <<PYEOF
import json, os
gaunt='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
dk='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper'
ent=[{'profile': os.environ['EVAL_PROFILE'],'priority':1,'active':True,'train_enabled':False,
      'deck_path':gaunt,
      'train_env':{'RL_AGENT_DECK_LIST': dk+'/'+os.environ['EVAL_AGENT_DECK'],
                   'MODEL_D_MODEL':'128','MODEL_NUM_LAYERS':'2',
                   'GAME_TIMEOUT_SEC':'2400'}}]
json.dump(ent,open('local-training/_det_eval_reg.json','w'),indent=2)
print('registry written for', os.environ['EVAL_PROFILE'])
PYEOF

log "=== [5/5] det sweeps: $EVAL_OPPONENTS n=$EVAL_N pilot=$EVAL_PILOT ==="
export SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODEL_NHEAD=4 MODEL_DIM_FEEDFORWARD=512
export INFER_CUDA_DEVICE=cpu TRAIN_CUDA_DEVICE=cpu MULLIGAN_DEVICE=cpu
[ "$EVAL_PILOT" = "1" ] && export RL_CP7_PILOT=1

# SANITY: 8 quick non-det games vs first opponent; wins>0 and no force-pass storm
FIRST=$(echo "$EVAL_OPPONENTS" | cut -d, -f1)
python3 scripts/run_cp7_eval_sweep.py --registry local-training/_det_eval_reg.json \
  --profiles "$EVAL_PROFILE" --opponents "$FIRST" --skill 7 \
  --games-per-matchup 8 --games-per-job 8 --parallel 1 \
  --gpu-port 26090 --gpu-metrics-port 26091 --skip-compile \
  --run-id sanity --timeout-sec 3600 --allow-incomplete-results >"$LOGD/sanity.log" 2>&1
SAN=$(python3 -c "
import csv
try:
    rows=list(csv.DictReader(open('local-training/local_pbt/cp7_eval_sweeps/sanity/matchup_summary.csv')))
    print(sum(int(r['wins']) for r in rows))
except Exception: print(-1)")
FP=$(grep -c 'forcing pass' local-training/local_pbt/cp7_eval_sweeps/sanity/logs/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
log "SANITY: wins=$SAN force_pass=$FP"
{ [ "$SAN" -le 0 ] || [ "$FP" -gt 0 ]; } && { log "SANITY FAILED"; tail -30 local-training/local_pbt/cp7_eval_sweeps/sanity/logs/*.log | tee -a "$LOGD/_det.log"; exit 2; }

i=0
IFS=',' read -ra OPPS <<< "$EVAL_OPPONENTS"
for opp in "${OPPS[@]}"; do
  port=$((26100+i*4))
  ( python3 scripts/run_cp7_eval_sweep.py --registry local-training/_det_eval_reg.json \
      --profiles "$EVAL_PROFILE" --opponents "$opp" --skill 7 \
      --games-per-matchup "$EVAL_N" --games-per-job 16 \
      --gpu-port "$port" --gpu-metrics-port $((port+1)) \
      --deterministic-eval --skip-compile \
      --replay-seed-base 5151 --seed-key-mode matchup \
      --run-id "${EVAL_PREFIX}_${opp}" --timeout-sec 14400 --allow-incomplete-results \
      >"$LOGD/${EVAL_PREFIX}_${opp}.log" 2>&1
    echo "rc=$? opp=$opp" >>"$LOGD/_det.log" ) &
  i=$((i+1)); sleep 45
done
log "all sweeps launched; waiting..."
wait
log "=== DONE; summaries ==="
for opp in "${OPPS[@]}"; do
  S="local-training/local_pbt/cp7_eval_sweeps/${EVAL_PREFIX}_${opp}/matchup_summary.csv"
  [ -f "$S" ] && { echo "== $opp"; cat "$S"; } | tee -a "$LOGD/_det.log"
done
tar -czf /root/det_results.tar.gz local-training/local_pbt/cp7_eval_sweeps/${EVAL_PREFIX}_* "$LOGD" 2>/dev/null
log "ALL DONE -- waiting for controller"
