# OVERNIGHT SUSTAINED TRAINING toward 60% vs CP7-skill-1 on the FIXED loop
# (fp32 ONNX exports, parity-gated; fp16 calibration bug root-caused 2026-06-10).
# Diet: pure CP7-skill-1 (eval-aligned ladder). One variable: training volume.
# Per-chunk fp32 re-export keeps the behavior policy tracking the learner.
# Gates: validation (c1+c2 both <0.42 -> abort), rolling (2 consecutive <0.42).
# Aborts SAVE the collapsed model for autopsy before restoring baseline.
# NO baseline restore on healthy completion -- the trained model is the product.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/overnight_RESULT.log"
$tlog = "local-training/overnight_train.log"; $telog = "local-training/overnight_train.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$ckdir= $env:CKPT_DIR; if (-not $ckdir) { $ckdir = "local-training\backups\overnight_20260610" }
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 12 }
$mode = $env:OVERNIGHT_MODE; if (-not $mode) { $mode = "hybrid" }   # hybrid | torch
New-Item -ItemType Directory -Force $ckdir | Out-Null
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER','PY_SERVICE_MODE'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  Remove-Item Env:\PY_SERVICE_MODE -ErrorAction SilentlyContinue
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) {
  $m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim()
  ">>> $m" | Out-File $out -Append
}
function Export-Onnx {
  # fp32 re-export of the CURRENT model so fresh JVMs play the latest policy.
  $stamp = "v" + (Get-Date -Format "yyyyMMddTHHmmss") + "_000000"
  $dest = "$md\onnx\$stamp"
  New-Item -ItemType Directory -Force $dest | Out-Null
  $env:ONNX_EXPORT_FP16="0"
  py -3.12 "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode\onnx_export.py" `
    --model-path "$md\model_latest.pt" --output-dir $dest 2>&1 |
    Select-String "PARITY FAILURE" | ForEach-Object { "EXPORT $($_.Line)" } | Out-File $out -Append
  if (Test-Path "$dest\model_action.onnx") {
    "3" | Out-File "$dest\.export_version" -Encoding ascii -NoNewline
    $stamp | Out-File "$md\onnx\.active_dir" -Encoding ascii -NoNewline
    "onnx exported -> $stamp" | Out-File $out -Append
  } else {
    "WARNING: onnx export failed; keeping previous .active_dir" | Out-File $out -Append
  }
}
function Get-Wr($rid) {
  $csv = "$GL/$rid/matchups.csv"
  if (Test-Path $csv) {
    $row = Import-Csv $csv | Select-Object -First 1
    return [double]$row.wins / [double]$row.total
  }
  return -1.0
}

"=== OVERNIGHT DOMINANCE RUN started $(Get-Date) ; ${nChunks}x${chunkMin}min ; mode=$mode ; diet=CP7-skill-1 ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 5
$startModel = $env:START_MODEL
if ($startModel -and (Test-Path $startModel)) {
  Copy-Item $startModel "$md\model.pt" -Force; Copy-Item $startModel "$md\model_latest.pt" -Force
  "starting model: $startModel" | Out-File $out -Append
} else {
  Copy-Item "$bak\model.pt" "$md\model.pt" -Force; Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
}
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }; Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
$pinTs = (Get-Content "local-training\backups\meta_pins_LATEST.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
if ($pinTs) {
  foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
    $pin = "local-training\backups\meta_pins_$pinTs\$p.model_latest.pt"
    if (Test-Path $pin) {
      Copy-Item $pin "$prof\$p\models\model_latest.pt" -Force
      Copy-Item $pin "$prof\$p\models\model.pt" -Force
    }
  }
  "meta-opponents restored from pin set $pinTs" | Out-File $out -Append
}

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_MCTS_SIGNED_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','CANDIDATE_Q_DETACH_ENCODER','SEARCH_OP_APPLY_OVERRIDE','SEARCH_OP_ARBITER_CAST_FILTER','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="ladder"; $env:LADDER_SKILLS="1"; $env:LADDER_MIX_LOWER_P="0.0"; $env:LEAGUE_MODE=""
  $entEnd = $env:RUN_ENTROPY_END; if (-not $entEnd) { $entEnd = "0.03" }
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END=$entEnd; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:ONNX_EXPORT_ENABLE="1"
  $env:SEARCH_OP_ENABLE="1"; $env:SEARCH_OP_ARBITER_CAST_FILTER="Balustrade Spy"
  $env:SEARCH_OP_MAX_ACTIVATIONS="1"; $env:SEARCH_OP_PLAYOUTS="2"
  $env:SEARCH_OP_PLAYOUT_TIMEOUT_MS="10000"; $env:SEARCH_OP_TOTAL_TIMEOUT_MS="30000"
  $env:SEARCH_OP_APPLY_OVERRIDE="0"; $env:CANDIDATE_Q_DETACH_ENCODER="0"   # COUPLED: encoder gradient flows from Q-head
  $env:CANDIDATE_Q_FROM_MCTS_TARGETS="1"; $env:CANDIDATE_Q_LOSS_COEF="0.05"
  $env:CANDIDATE_Q_MCTS_SIGNED_TARGETS="1"; $env:CANDIDATE_Q_BLEND="0.0"
  # KL anchor: hold the policy near the frozen 0.573 reference so the encoder can
  # reshape to encode the board-conditional distinction WITHOUT execution collapsing
  $env:REFERENCE_POLICY_KL_COEF="1.0"
  $env:MCTS_REFERENCE_MODEL_PATH="C:/Users/Jack/IdeaProjects/mage/local-training/backups/ref_frozen_0573.pt"
  $env:CANDIDATE_Q_DUMP_DIR="C:/Users/Jack/IdeaProjects/mage/local-training/candq_dumps_v7"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="64"; $env:TOTAL_EPISODES="99999999"
  if ($mode -eq "torch") { $env:PY_SERVICE_MODE="shared_gpu" } else { Remove-Item Env:\PY_SERVICE_MODE -ErrorAction SilentlyContinue }
}

$evalChunks = @(1,2,4,6,8,10,12,14,16)
$bestWr = 0.0
$lowStreak = 0
for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  if ($mode -ne "torch") { Export-Onnx }
  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  if ($evalChunks -contains $c) {
    Eval-Now "on_c$c" 5151; Metric "on_c$c"
    $wr = Get-Wr "on_c$c"
    Copy-Item "$md\model_latest.pt" "$ckdir\model_after_c$c.pt" -Force
    if ($wr -gt $bestWr) {
      $bestWr = $wr
      Copy-Item "$md\model_latest.pt" "$ckdir\model_best.pt" -Force
      "new best wr=$wr at chunk $c (model_best.pt updated)" | Out-File $out -Append
    }
    if ($wr -ge 0 -and $wr -lt 0.42) { $lowStreak++ } else { $lowStreak = 0 }
    if ($c -eq 2 -and $lowStreak -ge 2) {
      "=== ABORT: validation gate failed (c1+c2 both < 0.42). Saving autopsy + restoring baseline. ===" | Out-File $out -Append
      Copy-Item "$md\model_latest.pt" "$ckdir\model_aborted_c$c.pt" -Force
      Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force; Copy-Item "$bak\model.pt" "$md\model.pt" -Force
      break
    }
    if ($c -gt 2 -and $lowStreak -ge 2) {
      "=== ABORT at chunk ${c}: 2 consecutive evals < 0.42. Saving autopsy; best model preserved in $ckdir. ===" | Out-File $out -Append
      Copy-Item "$md\model_latest.pt" "$ckdir\model_aborted_c$c.pt" -Force
      if (Test-Path "$ckdir\model_best.pt") {
        Copy-Item "$ckdir\model_best.pt" "$md\model_latest.pt" -Force
        Copy-Item "$ckdir\model_best.pt" "$md\model.pt" -Force
        "restored best-so-far model (wr=$bestWr)" | Out-File $out -Append
      }
      break
    }
  }
}
"=== OVERNIGHT RUN DONE $(Get-Date) ; best_wr=$bestWr ; checkpoints in $ckdir ===" | Out-File $out -Append


