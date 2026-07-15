# DECK-BREWING ablation arm 1: CONSISTENCY-ONLY fresh-train (Codex's clean combo-reach
# test). Same recipe/diet as run_fresh128_control.ps1 (the OLD-list baseline) -- fresh-128,
# faithful league diet, terminal-only, zone-count OFF -- but the AGENT plays the
# consistency-only list (winning list's mana tuning, NO Avenging Hunter initiative).
# Compare legal-reach / cast / winrate trajectory vs fresh128_control (f128_cN).
# READ: reach UP + wr UP => list was the bottleneck; reach UP but cast/wr flat => PLAY is.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/brew_win_RESULT.log"
$tlog = "local-training/brew_win.log"; $telog = "local-training/brew_win.err"
$baseReg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$reg  = "local-training/_brew_win_registry.json"
$deck = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Winning.dek"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$ck   = "E:\mage-training\backups\brew_winning_20260619"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$statsCsv = "$prof\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 60 }
New-Item -ItemType Directory -Force $ck | Out-Null

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Clean-Sweep($rid) {
  # keep game_logs (needed for legal-reach oracle); drop heavy dirs
  foreach ($sub in 'snapshot','db','results') { if (Test-Path "$GL/$rid/$sub") { Remove-Item "$GL/$rid/$sub" -Recurse -Force -ErrorAction SilentlyContinue } }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'OPPONENT_SAMPLER','SEARCH_OP_ENABLE','LEAGUE_DEBUG'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"; $env:OPPONENT_SAMPLER="grixis"; Remove-Item Env:\LEAGUE_DEBUG -ErrorAction SilentlyContinue
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) { ">>> $((py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim())" | Out-File $out -Append }
function ReachMetric($rid) {
  $r = (py -3.12 scripts/mtgrl/candidate_offer_oracle.py "$GL/$rid" 2>&1 | Select-String "never-cast=|cast Spy:|WR\|cast|NEVER offered|OFFERED but" | ForEach-Object { $_.Line.Trim() }) -join " | "
  "REACH $rid : $r" | Out-File $out -Append
}
function TrainWinfrac {
  if (-not (Test-Path $statsCsv)) { return "n/a" }
  $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds
  $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count)
  return "winfrac=$([Math]::Round($w,3))"
}

"=== BREW WINNING (fresh-128, faithful diet, consistency deck) started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
# registry copy: point Spy agent deck at the consistency list
py -3.12 -c "
import json
reg=json.load(open(r'$baseReg'))
for e in reg:
    if e.get('profile')=='Pauper-Spy-Combo-Value':
        e.setdefault('train_env',{})['RL_AGENT_DECK_LIST']=r'$deck'
json.dump(reg,open(r'$reg','w'),indent=2)
print('brew-cons registry: Spy agent deck -> $deck')
" 2>&1 | Out-File $out -Append
$env:REGISTRY_PATH = (Resolve-Path $reg).Path

if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null
foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  $pin = "local-training\backups\meta_pins_pristine\$p.model_latest.pt"
  if (Test-Path $pin) { Copy-Item $pin "$prof\$p\models\model_latest.pt" -Force; Copy-Item $pin "$prof\$p\models\model.pt" -Force }
}
py -3.12 -c "
import json,os
base='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles'
for p in ['Pauper-Spy-Combo-Value','Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value']:
    f=f'{base}/{p}/logs/league/agent_status.json'; os.makedirs(os.path.dirname(f),exist_ok=True)
    d=json.load(open(f)) if os.path.exists(f) else {}
    d['promoted']=True; d['baseline_wr']=max(0.45,d.get('baseline_wr',0)); json.dump(d,open(f,'w'),indent=2)
print('all 4 promoted -> META-RL clock + mirror + CP7 (faithful diet)')
" 2>&1 | Out-File $out -Append

$evalChunks = @(2,3,4,6,8,10,12,15,18,22,26,30,36,42,50,60)
$bestWr = 0.0
for ($c = 1; $c -le $nChunks; $c++) {
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','VALUE_USE_SEPARATE_CRITIC_ENCODER','RL_WORLD_MODEL_LABELS_ENABLE','WORLD_MODEL_DIM','RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
  $env:REGISTRY_PATH = (Resolve-Path $reg).Path

  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  $freeGB = [Math]::Round((Get-PSDrive C).Free/1GB,1)
  "chunk $c done; in-train $(TrainWinfrac) ; freeGB=$freeGB" | Out-File $out -Append
  if ($freeGB -lt 5) { "=== ABORT: disk <5GB ===" | Out-File $out -Append; break }
  if ($evalChunks -contains $c) {
    Eval-Now "bw_c$c" 5151; Metric "bw_c$c"; ReachMetric "bw_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/bw_c$c/matchups.csv"
    if (Test-Path $csv) { $row = Import-Csv $csv | Select-Object -First 1; $wr = [double]$row.wins/[double]$row.total
      if ($wr -gt $bestWr) { $bestWr=$wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append } }
    Clean-Sweep "bw_c$c"
  }
}
"=== BREW WINNING DONE $(Get-Date) ; best_wr=$bestWr ; ckpts $ck ===" | Out-File $out -Append
