#!/usr/bin/env bash
# On-pod: provision (deps + CPU torch + mvn install), set up 2 profiles, run the
# scoreboard triage (Affinity+Spy x 8-deck gauntlet, s7, n=128, play-side, deterministic)
# as a 16-job parallel array, then SELF-TERMINATE the pod. Run: bash pod_scoreboard.sh <POD_ID>
set -uo pipefail
POD_ID="${1:?POD_ID required for self-terminate}"
cd /root/mage
LOGD=/root/pod_logs; mkdir -p "$LOGD"
KEYFILE=/root/.runpod_api_key

log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOGD/_provision.log"; }

log "=== [1/6] apt deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends openjdk-21-jdk maven python3 python3-pip python3-venv rsync >/dev/null 2>&1 \
  || apt-get install -y -qq --no-install-recommends openjdk-17-jdk maven python3 python3-pip python3-venv rsync >/dev/null 2>&1
log "java: $(java -version 2>&1 | head -1)  cores: $(nproc)"

log "=== [2/6] python CPU deps ==="
pip3 install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
# transformers is REQUIRED: mtg_transformer.py imports HuggingFace AutoModel at module
# level; omitting it was the jul6 0-games root cause (profile registration failed).
pip3 install -q --no-input numpy onnx onnxruntime py4j transformers >/dev/null 2>&1
log "torch: $(python3 -c 'import torch;print(torch.__version__)' 2>&1)"

log "=== [3/6] mvn install (AIRL + deps, ~15-30min) ==="
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install >"$LOGD/_mvn.log" 2>&1
if [ $? -ne 0 ]; then log "MVN FAILED (tail):"; tail -20 "$LOGD/_mvn.log" | tee -a "$LOGD/_provision.log"; fi
log "mvn done"

log "=== [4/6] profiles + registries ==="
PROF=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles
GAUNT=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt
mkdir -p "$PROF/Pauper-Affinity-Anchor-Value/models" "$PROF/Pauper-Spy-Combo-Value/models"
cp /root/ckpts/affinity_7k.pt "$PROF/Pauper-Affinity-Anchor-Value/models/model.pt"
cp /root/ckpts/affinity_7k.pt "$PROF/Pauper-Affinity-Anchor-Value/models/model_latest.pt"
cp /root/ckpts/spy_22k.pt "$PROF/Pauper-Spy-Combo-Value/models/model.pt"
cp /root/ckpts/spy_22k.pt "$PROF/Pauper-Spy-Combo-Value/models/model_latest.pt"
# per-deck gauntlet registries (agent plays its own deck vs the 8-deck gauntlet)
python3 - <<PYEOF
import json
base='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_registry.json'
r=json.load(open(base))
gaunt='$GAUNT'
def mk(profile, agentdeck, out):
    ent=[e for e in r if e.get('profile')==profile]
    if not ent:  # synthesize a minimal entry
        ent=[{'profile':profile,'priority':1,'active':True,'train_enabled':False,'train_env':{}}]
    for e in ent:
        e['deck_path']=gaunt
        e.setdefault('train_env',{})['RL_AGENT_DECK_LIST']=agentdeck
    json.dump(ent,open(out,'w'),indent=2); print('wrote',out)
DK='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper'
mk('Pauper-Affinity-Anchor-Value', DK+'/Deck - Grixis Affinity.dek', 'local-training/_sb_affinity_reg.json')
mk('Pauper-Spy-Combo-Value', DK+'/Deck - Spy Winning.dek', 'local-training/_sb_spy_reg.json')
PYEOF
mkdir -p local-training

log "=== [5/6] scoreboard batch: 16 jobs (2 decks x 8 opp), s7 n=128 play-side ==="
OPPS=(grixis burn faeries terror wildfire caw elves rally)
export SEARCH_OP_ENABLE=0 USE_TRT_INFERENCE=0 MODEL_D_MODEL=128 MODEL_NUM_LAYERS=2 MODEL_NHEAD=4 MODEL_DIM_FEEDFORWARD=512
run_job(){
  local deck="$1" prof="$2" reg="$3" init="$4" opp="$5" port="$6"
  RL_INITIATIVE_FEATURES_ENABLE=$init python3 scripts/run_cp7_eval_sweep.py --registry "$reg" \
    --profiles "$prof" --opponents "$opp" --skill 7 --games-per-matchup 128 --games-per-job 16 \
    --gpu-port "$port" --gpu-metrics-port $((port+1)) \
    --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "sb_${deck}_${opp}" 2>&1 \
    | grep -aE 'wr=' > "$LOGD/sb_${deck}_${opp}.log"
}
export -f run_job; export LOGD
i=0
for opp in "${OPPS[@]}"; do
  run_job affinity Pauper-Affinity-Anchor-Value local-training/_sb_affinity_reg.json 0 "$opp" $((26100+i*4)) &
  i=$((i+1)); sleep 90
done
for opp in "${OPPS[@]}"; do
  run_job spy Pauper-Spy-Combo-Value local-training/_sb_spy_reg.json 1 "$opp" $((26100+i*4)) &
  i=$((i+1)); sleep 90
done
log "all 16 jobs launched; waiting..."
wait
log "=== [6/6] ALL JOBS DONE ==="
grep -c 'wr=' "$LOGD"/sb_*.log 2>/dev/null | tee -a "$LOGD/_provision.log"

# grace period so main can pull results, then self-terminate
log "DONE marker written; 1800s grace before self-terminate"
sleep 1800
if [ -f "$KEYFILE" ]; then
  log "self-terminating pod $POD_ID"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $(cat $KEYFILE)" >>"$LOGD/_provision.log" 2>&1
fi
