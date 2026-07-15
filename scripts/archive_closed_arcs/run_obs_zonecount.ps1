# OBSERVATION-ENCODING experiment (Codex-endorsed, thesis-clean): give the agent
# the generic zone-count scalars it is currently BLIND to. The Spy combo's go/no-go
# discriminator is "lands remaining in library" (Balustrade Spy mills UNTIL it hits a
# land; the deck strips lands so 0-left => full mill => win, >0 => partial => fizzle).
# That scalar (v[19]) is GATED OFF by default and the individual library tokens
# truncate at MAX_LEN=256 -> the agent cannot see the #1 combo signal. Board-creature
# count (the other requirement) IS already visible. THESIS-CLEAN: objective per-zone
# counts a pilot derives from a known decklist; no card names, no comboReady label.
# CONTROL = run_fresh128_control.ps1 (identical recipe, these slots = 0). Fresh-start
# (Codex: a plateaued teacher confounds a schema change). Metric = fizzle rate
# (executed/cast_spy) + dominated-cast (NO_BOARD) + winrate vs the fresh-128 control.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/obs_zonecount_RESULT.log"
$tlog = "local-training/obs_zonecount.log"; $telog = "local-training/obs_zonecount.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$ck   = "E:\mage-training\backups\obs_zonecount_20260618"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$trainerLog = "local-training/local_pbt/trainer.log"
$statsCsv = "$prof\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 120 }
New-Item -ItemType Directory -Force $ck | Out-Null

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Clean-Sweep($rid) {
  $d = "$GL/$rid"
  foreach ($sub in 'snapshot','db','results') { if (Test-Path "$d/$sub") { Remove-Item "$d/$sub" -Recurse -Force -ErrorAction SilentlyContinue } }
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
function TrainWinfrac {
  if (-not (Test-Path $statsCsv)) { return "n/a" }
  $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds
  $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count)
  return "winfrac=$([Math]::Round($w,3))"
}

"=== OBS ZONE-COUNT (fresh-128, faithful diet, lib/zone counts ON) started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null

# Faithful diet: restore 128-dim meta pins + promote all 4 (META-RL clock + mirror + CP7)
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

$evalChunks = @(2,3,4,6,8,10,12,15,18,22,26,30,36,42,50,60,72,84,96,108,120)
$bestWr = 0.0; $engaged = $false
for ($c = 1; $c -le $nChunks; $c++) {
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','VALUE_USE_SEPARATE_CRITIC_ENCODER','RL_WORLD_MODEL_LABELS_ENABLE','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  # THE LEVER: thesis-clean generic zone-count scalars (lib lands/creatures, gy creatures) + engagement diag
  $env:RL_ZONE_COUNT_FEATURES_ENABLE="1"; $env:RL_LIBRARY_COUNT_FEATURES_ENABLE="1"; $env:RL_ZONE_COUNT_DIAG="1"
  # faithful league diet (META-RL clock + mirror + CP7)
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"

  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8

  $freeGB = [Math]::Round((Get-PSDrive C).Free/1GB,1)
  "chunk $c done; in-train $(TrainWinfrac) ; freeGB=$freeGB" | Out-File $out -Append
  # ENGAGEMENT CHECK (once): confirm the zone-count scalars are actually NONZERO at runtime
  if (-not $engaged -and (Test-Path $trainerLog)) {
    $zc = (Select-String -Path $trainerLog -Pattern '\[ZONE_COUNT_DIAG\]' -ErrorAction SilentlyContinue | Select-Object -Last 4 | ForEach-Object { $_.Line.Trim() }) -join " || "
    if ($zc) { "ZONE_COUNT_DIAG: $zc" | Out-File $out -Append; $engaged = $true }
    else { "ZONE_COUNT_DIAG: (none yet)" | Out-File $out -Append }
  }
  if ($freeGB -lt 5) { "=== ABORT: disk <5GB ===" | Out-File $out -Append; break }
  if ($evalChunks -contains $c) {
    Eval-Now "oz_c$c" 5151; Metric "oz_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/oz_c$c/matchups.csv"
    if (Test-Path $csv) { $row = Import-Csv $csv | Select-Object -First 1; $wr = [double]$row.wins/[double]$row.total
      if ($wr -gt $bestWr) { $bestWr=$wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append } }
    Clean-Sweep "oz_c$c"
  }
}
"=== OBS ZONE-COUNT DONE $(Get-Date) ; best_wr=$bestWr ; ckpts in $ck ===" | Out-File $out -Append
