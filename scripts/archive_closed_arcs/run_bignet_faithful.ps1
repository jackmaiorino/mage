# FRESH-TRAIN the wide net (d_model=256/4L) on the FAITHFUL LEAGUE DIET --
# the recipe that bootstrapped the original 128 net 0.05->0.52 in ~12.5k eps.
# THE proper capacity test (prior fresh runs used pure CP7 = no clock = no
# combo discovery). Thesis-clean: terminal-only reward, fixed-reference opponents.
# META-H clock (CP7 skill-3 on the aggressive pool decks) -- meta profiles kept
# UNQUALIFIED so no 128-dim NN snapshot is loaded (would shape-mismatch vs 256).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/bignet_faithful_RESULT.log"
$tlog = "local-training/bignet_faithful.log"; $telog = "local-training/bignet_faithful.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$ck   = "local-training\backups\bignet_faithful_20260616"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
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
  foreach ($sub in 'snapshot','game_logs','logs','db','results') { if (Test-Path "$d/$sub") { Remove-Item "$d/$sub" -Recurse -Force -ErrorAction SilentlyContinue } }
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
function RepSep($rid) {
  $r = (py -3.12 scripts/mtgrl/candq_dump_analyze.py "local-training/candq_dumps_v6" "$md\model_latest.pt" 2>&1 | Select-String 'detected|policy\(best\)|Q separation' | ForEach-Object { $_.Line.Trim() }) -join " | "
  "REPSEP $rid : $r" | Out-File $out -Append
}
function TrainWinfrac {
  if (-not (Test-Path $statsCsv)) { return "n/a" }
  $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds
  $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count)
  $opp = ($rows | Group-Object opponent_type | Sort-Object Count -Descending | Select-Object -First 2 | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ","
  return "winfrac=$([Math]::Round($w,3)) opp[$opp]"
}

"=== BIGNET FAITHFUL (faithful diet, META-H clock) started $(Get-Date) ; d_model=256/4L ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null

# Restore 128-dim meta pins + promote ALL 4 profiles so meta opponents are
# QUALIFIED -> META-RL (the strong NN clock; loadable vs the 256 net via the
# dim-aware snapshot fix). Spy promoted -> gets the restored 20/40/40
# CP7/local-mirror/cross mix (the faithful original diet).
$pinTs = (Get-Content "local-training\backups\meta_pins_LATEST.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
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
" 2>&1 | Out-File $out -Append

$evalChunks = @(2,3,4,6,8,10,12,15,18,22,26,30,36,42,50,60,72,84,96,108,120)
$bestWr = 0.0; $dimChecked = $false
for ($c = 1; $c -le $nChunks; $c++) {
  $env:MODEL_D_MODEL="256"; $env:MODEL_NUM_LAYERS="4"; $env:MODEL_NHEAD="8"; $env:MODEL_DIM_FEEDFORWARD="1024"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_MCTS_SIGNED_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','CANDIDATE_Q_DETACH_ENCODER','SEARCH_OP_APPLY_OVERRIDE','SEARCH_OP_ARBITER_CAST_FILTER','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  # FAITHFUL LEAGUE DIET (META-H clock): aggressive pool decks via CP7 provide the clock that forces combo discovery
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  if ($c -le 2) { $env:LEAGUE_DEBUG="1" } else { Remove-Item Env:\LEAGUE_DEBUG -ErrorAction SilentlyContinue }
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
  if ($c -le 2) {
    $dbg = (Select-String -Path $tlog -Pattern '\[LEAGUE_DBG\] meta-candidates|cand .* qualified' -ErrorAction SilentlyContinue | Select-Object -First 5 | ForEach-Object { $_.Line.Trim() }) -join " || "
    "LEAGUE_DBG: $dbg" | Out-File $out -Append
  }
  if ($freeGB -lt 5) { "=== ABORT: disk <5GB ===" | Out-File $out -Append; break }
  if (-not $dimChecked -and (Test-Path "$md\model_latest.pt")) {
    $dim = (py -3.12 -c "import torch; print(torch.load(r'$md/model_latest.pt',map_location='cpu',weights_only=False)['state_dict']['cls_token'].shape[-1])" 2>&1 | Select-Object -Last 1)
    "DIM CHECK: d_model=$dim (expect 256)" | Out-File $out -Append; $dimChecked = $true
  }
  if ($evalChunks -contains $c) {
    Eval-Now "bf_c$c" 5151; Metric "bf_c$c"; RepSep "bf_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/bf_c$c/matchups.csv"
    if (Test-Path $csv) { $row = Import-Csv $csv | Select-Object -First 1; $wr = [double]$row.wins/[double]$row.total
      if ($wr -gt $bestWr) { $bestWr=$wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append } }
    Clean-Sweep "bf_c$c"
  }
}
"=== BIGNET FAITHFUL DONE $(Get-Date) ; best_wr=$bestWr ; ckpts in $ck ===" | Out-File $out -Append

