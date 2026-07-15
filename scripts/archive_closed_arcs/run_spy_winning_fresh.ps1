# FRESH-TRAIN d_model=128/2L on the tournament-winning Spy list ("Deck - Spy
# Winning.dek") over the FAITHFUL LEAGUE DIET -- the exact recipe that
# bootstrapped the original fresh-128 control (0.05->0.52 in ~12.5k eps).
# Every diet env var below is copied verbatim from scripts/run_fresh128_control.ps1
# (the per-chunk loop body, lines ~80-90: META-H clock, OPPONENT_SAMPLER=league,
# entropy schedule, ONNX batch timing, split-GPU device assignment). Only three
# things differ from that control: (1) agent deck = Spy Winning list instead of
# the old pool file, applied via a temp registry copy exactly like
# scripts/run_pretest_decks.ps1 / local-training/_pretest_reg_win.json do
# (RL_AGENT_DECK_LIST override in train_env, Spy profile only -- Wildfire/Rally/
# Affinity entries untouched so the META-H clock is unaffected); (2)
# RL_INITIATIVE_FEATURES_ENABLE=1 is turned on (new opt-in designation-state
# feature -- initiative/Undercity/monarch slots v[23-26] in
# StateSequenceBuilder.java -- safe to enable from a fresh model, no legacy
# shape mismatch); (3) this is a single non-chunked run (no eval/backup loop)
# sized by $env:SPY_TARGET, using the single-shot launch/log pattern from
# scripts/run_ar_league_continue.ps1 (Tee-Object, TOTAL_EPISODES=target,
# default 8000 -- same default that script uses for LEAGUE_TARGET).
# Terminal-only reward: RL_HEURISTIC_STEP_REWARDS=0 set explicitly (also the
# documented default; no shaped step rewards).
# Back up nothing -- this IS a from-scratch run, not a continuation.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/spy_winning_fresh.log"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$tmpReg = "local-training/_spy_winning_fresh_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$agentDeck = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Winning.dek"
$target = if ($env:SPY_TARGET) { $env:SPY_TARGET } else { "8000" }

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 4

# --- fresh model: clear Pauper-Spy-Combo-Value models dir (no restore) ---
# Guarded: only wipes when SPY_FRESH=1. Default (unset) = CONTINUATION from the
# existing model.pt (the orchestrator counts episodes from training_stats.csv,
# which the caller clears for a fresh episode budget).
if ($env:SPY_FRESH -eq "1") {
  if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
  New-Item -ItemType Directory -Force $md | Out-Null
} else {
  if (-not (Test-Path "$md\model.pt")) { "FATAL: continuation requested but $md\model.pt missing (set SPY_FRESH=1 for a fresh run)" | Out-File $log -Append; exit 1 }
  # wipe stale onnx so it re-exports from the continued weights
  if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
}

# --- temp registry: copy of the main registry with RL_AGENT_DECK_LIST overridden
# for Pauper-Spy-Combo-Value only, exactly like run_pretest_decks.ps1 does ---
py -3.12 -c "
import json
reg = json.load(open(r'$reg'))
rows = reg if isinstance(reg, list) else reg.get('profiles', reg)
for e in (rows if isinstance(rows, list) else []):
    if e.get('profile') == 'Pauper-Spy-Combo-Value':
        e.setdefault('train_env', {})['RL_AGENT_DECK_LIST'] = r'$agentDeck'
json.dump(reg, open(r'$tmpReg', 'w'), indent=2)
print('wrote $tmpReg agent_deck=$agentDeck')
" 2>&1 | Out-File $log

# --- restore 128-dim meta pins + promote ALL 4 profiles so meta opponents are
# QUALIFIED -> META-RL clock (identical to run_fresh128_control.ps1) ---
foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  $pin = "local-training\backups\meta_pins_pristine\$p.model_latest.pt"
  if (Test-Path $pin) { Copy-Item $pin "$prof\$p\models\model_latest.pt" -Force; Copy-Item $pin "$prof\$p\models\model.pt" -Force }
}
py -3.12 -c "
import json,os
base='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles'
for p in ['Pauper-Spy-Combo-Value','Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value']:
    f=f'{base}/{p}/logs/league/agent_status.json'
    os.makedirs(os.path.dirname(f),exist_ok=True)
    d=json.load(open(f)) if os.path.exists(f) else {}
    d['promoted']=True; d['baseline_wr']=max(0.45,d.get('baseline_wr',0))
    json.dump(d,open(f,'w'),indent=2)
print('all 4 promoted; meta=QUALIFIED -> META-RL clock; Spy gets 20/40/40 mirror mix')
" 2>&1 | Out-File $log -Append

"=== SPY WINNING FRESH-128 started $(Get-Date) ; deck=$agentDeck ; target=$target ===" | Out-File $log -Append

# --- env: verbatim faithful league diet from run_fresh128_control.ps1 ---
$env:REGISTRY_PATH = $tmpReg
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_MCTS_SIGNED_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','CANDIDATE_Q_DETACH_ENCODER','SEARCH_OP_APPLY_OVERRIDE','SEARCH_OP_ARBITER_CAST_FILTER','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:CANDIDATE_Q_ONLY="0"
# FAITHFUL LEAGUE DIET (META-H clock): aggressive pool decks via CP7 provide the clock that forces combo discovery
$env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
$env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
$env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="$target"

# --- new: opt-in initiative/monarch designation-state features (StateSequenceBuilder.java) ---
$env:RL_INITIATIVE_FEATURES_ENABLE="1"

# --- terminal-only reward (explicit; also the documented default) ---
$env:RL_HEURISTIC_STEP_REWARDS="0"

"=== SPY WINNING FRESH-128 run (TOTAL_EPISODES=$target) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== SPY WINNING FRESH-128 DONE $(Get-Date) ===" | Out-File $log -Append
