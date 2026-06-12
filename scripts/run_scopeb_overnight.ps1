# scope_B overnight: clean frozen-encoder Q-head training, then auto n=128 blend eval.
# Tests "undertrained" vs "frozen-myopic-encoder ceiling" for Option B.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/scopeb_overnight_RESULT.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak = "local-training\backups\spy_value_baseline_20260531"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== scope_B overnight clean-frozen-Q started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# restore pristine baseline so the Q-head trains on the FROZEN baseline encoder
Copy-Item "$bak\model.pt" "$md\model.pt" -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored" | Out-File $out -Append

# --- training env: train ONLY the Q-head, encoder TRULY frozen ---
$env:SEARCH_OP_ENABLE="1"; $env:SEARCH_OP_TOP_K="3"; $env:SEARCH_OP_PLAYOUTS="4"
$env:SEARCH_OP_PLAYOUT_TIMEOUT_MS="2500"; $env:SEARCH_OP_TOTAL_TIMEOUT_MS="15000"
$env:SEARCH_OP_MAX_ACTIVATIONS="6"; $env:SEARCH_OP_MAX_GAME_TURNS="40"
$env:SEARCH_OP_GENERIC_BRANCH_ORDER="true"; $env:SEARCH_OP_SKIP_TOP_PROB="0.85"; $env:SEARCH_OP_LOG="1"
$env:CANDIDATE_Q_ONLY="1"; $env:CANDIDATE_Q_LOSS_COEF="1.0"
$env:CANDIDATE_Q_FROM_MCTS_TARGETS="1"; $env:CANDIDATE_Q_MCTS_SIGNED_TARGETS="1"; $env:CANDIDATE_Q_HUBER_BETA="0.25"
$env:FREEZE_ENCODER_IN_WARMUP="0"   # <-- fixes the warmup-unfreeze bug; encoder stays frozen
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="10"; $env:TOTAL_EPISODES="99999999"
$env:MULTI_PLY_MCTS="0"; $env:MCTS_TRAINING_ENABLE="0"

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput "local-training/scopeb_overnight_train.log" `
  -RedirectStandardError  "local-training/scopeb_overnight_train.err" -WindowStyle Hidden
"training launched $(Get-Date); ~8h window" | Out-File $out -Append

# train ~8h, log approx target accumulation every 15 min
for ($i = 0; $i -lt 32; $i++) {
  Start-Sleep -Seconds 900
  $tl = "local-training/local_pbt/trainer.log"
  $obs = 0
  if (Test-Path $tl) {
    $obs = (Select-String -Path $tl -Pattern "observed=([2-9]|1[0-9])" -AllMatches -ErrorAction SilentlyContinue | Measure-Object).Count
  }
  "[$([int](($i+1)*15)) min] approx target rows: $obs   $(Get-Date)" | Out-File $out -Append
}

"training window done $(Get-Date); stopping trainer" | Out-File $out -Append
Kill-Train; Start-Sleep -Seconds 6

# did the encoder stay frozen this time? (clean-freeze check)
"=== weight diff vs baseline (encoder ~0 = clean freeze; candidate_q >0 = learned) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/qhead_weight_diff.py 2>&1 | Out-File $out -Append

# --- auto n=128 blend eval: blend=0 (control) vs blend=1.0 ---
$env:SEARCH_OP_ENABLE="0"; $env:CANDIDATE_Q_ONLY="0"; $env:USE_TRT_INFERENCE="0"; $env:CANDIDATE_Q_BLEND_HEADS="*"
foreach ($b in @("0.0","1.0")) {
  $env:CANDIDATE_Q_BLEND = $b
  $rid = "blendovernight_" + ($b -replace '\.','p')
  "=== eval blend=$b ($rid) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json `
    --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 5151 --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}
"=== OVERNIGHT DONE $(Get-Date) ===" | Out-File $out -Append
