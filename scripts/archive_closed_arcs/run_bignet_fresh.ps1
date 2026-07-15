# FRESH-TRAIN the wide net (d_model=256, 4 layers) from scratch toward the
# representational-capacity question: can a bigger encoder learn the
# board-conditional distinction candidate_q proved the 128/2-layer net cannot?
# NO abort gate (a fresh net legitimately starts near 0 and must climb).
# NO baseline restore (the fresh net IS the product).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/bignet_RESULT.log"
$tlog = "local-training/bignet_train.log"; $telog = "local-training/bignet_train.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$ck   = "local-training\backups\bignet_20260614"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 60 }
New-Item -ItemType Directory -Force $ck | Out-Null
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) {
  $m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim()
  ">>> $m" | Out-File $out -Append
}
function RepSep($rid) {
  # rep-separation on the fixed v6 search-labeled probe states (auto-detects dims)
  $r = (py -3.12 scripts/mtgrl/candq_dump_analyze.py "local-training/candq_dumps_v6" "$md\model_latest.pt" 2>&1 | Select-String 'detected|policy\(best\)|Q separation' | ForEach-Object { $_.Line.Trim() }) -join " | "
  "REPSEP $rid : $r" | Out-File $out -Append
}

"=== BIGNET FRESH TRAIN started $(Get-Date) ; ${nChunks}x${chunkMin}min ; d_model=256/4L ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
# fresh-init: clear the 128 model so the 256 net initializes from scratch
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null

$trainEnv = {
  $env:MODEL_D_MODEL="256"; $env:MODEL_NUM_LAYERS="4"; $env:MODEL_NHEAD="8"; $env:MODEL_DIM_FEEDFORWARD="1024"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_MCTS_SIGNED_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','CANDIDATE_Q_DETACH_ENCODER','SEARCH_OP_APPLY_OVERRIDE','SEARCH_OP_ARBITER_CAST_FILTER','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="ladder"; $env:LADDER_SKILLS="1"; $env:LADDER_MIX_LOWER_P="0.0"; $env:LEAGUE_MODE=""
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"   # match small-net known-good schedule (200k was too slow)
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
}

$evalChunks = @(2,4,6,9,12,15,18,21,24,27,30,36,42,48,54,60)
$bestWr = 0.0
$dimChecked = $false
for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  # one-time dimension verification after the first checkpoint exists
  if (-not $dimChecked -and (Test-Path "$md\model_latest.pt")) {
    $dim = (py -3.12 -c "import torch; d=torch.load(r'$md/model_latest.pt',map_location='cpu',weights_only=False)['state_dict']; print(d['cls_token'].shape[-1])" 2>&1 | Select-Object -Last 1)
    "DIM CHECK: model cls_token d_model=$dim (expect 256)" | Out-File $out -Append
    $dimChecked = $true
  }
  if ($evalChunks -contains $c) {
    Eval-Now "bn_c$c" 5151; Metric "bn_c$c"; RepSep "bn_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/bn_c$c/matchups.csv"
    if (Test-Path $csv) {
      $row = Import-Csv $csv | Select-Object -First 1
      $wr = [double]$row.wins / [double]$row.total
      if ($wr -gt $bestWr) { $bestWr = $wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append }
    }
  }
}
"=== BIGNET FRESH TRAIN DONE $(Get-Date) ; best_wr=$bestWr ; ckpts in $ck ===" | Out-File $out -Append
