# SWEEP Stage-1 Arm-S: size S (d128/L2), CONTINUE from baseline, SELF-PLAY,
# entropy decay MATCHED to the run budget (~300k train_step), GPU split, batch 25ms.
# Tests the UNDERTRAINING hypothesis on the current model. Eval vs Grixis after
# each chunk -> learning-curve slope. Climbing/crossing 60% => undertrained.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/sweep_armS_RESULT.log"
$tlog = "local-training/sweep_armS_train.log"
$telog= "local-training/sweep_armS_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 360 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 4 }

# GPU split: ONNX inference -> idle GPU1, PyTorch training -> GPU0
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','CANDIDATE_Q_LOSS_COEF','REFERENCE_POLICY_KL_COEF','CANDIDATE_Q_ONLY','OPPONENT_SAMPLER'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  "=== eval $rid (seed $seed) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}

"=== SWEEP Arm-S (size S, continue, self-play) started $(Get-Date) ; ${nChunks}x${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# continue from baseline (head start at train_step 131818)
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored (continue from train_step ~131818)" | Out-File $out -Append

# t=0 eval (the start point)
Eval-Now "armS_c0" 5151

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\CANDIDATE_Q_REPLAY_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\CANDIDATE_Q_DUMP_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="self"          # SELF-PLAY (fast; CP7 engine-search too slow to train against)
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"   # size S (matches baseline)
  # budget-matched entropy: ~0.45 eps/s x 30h ~= +48k episodes (131818 -> ~180k);
  # decay over 180k so entropy anneals ~0.09 -> 0.03 IN-WINDOW (policy actually sharpens).
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="180000"
  $env:ACTOR_LR="2e-4"
  $env:ONNX_BATCH_TIMEOUT_MS="5"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="10"  # cap death-spiral timeout
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
}

for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
    -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  $ts = (Select-String -Path "local-training/local_pbt/trainer.log" -Pattern "episodes:|train_step" -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
  "chunk $c done $(Get-Date) | $ts" | Out-File $out -Append
  $snap = "local-training\backups\spy_armS_c$c"
  if (Test-Path $snap) { Remove-Item $snap -Recurse -Force }
  New-Item -ItemType Directory -Path $snap -Force | Out-Null
  Copy-Item "$md\model_latest.pt" "$snap\model_latest.pt" -Force
  Eval-Now "armS_c$c" 5151
}
# final 2nd-seed eval + funnel
$gl = "local-training/local_pbt/cp7_eval_sweeps"
Eval-Now "armS_final_s9999" 9999
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label ARMS_FINAL "$gl/armS_c${nChunks}/game_logs" 2>&1 | Out-File $out -Append
"=== SWEEP Arm-S DONE $(Get-Date) ===" | Out-File $out -Append
