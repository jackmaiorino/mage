# TREND-CHECK: is the Spy model saturated, or just undertrained? Entropy schedule
# says ~5% trained (train_step 131818 / 2.5M decay). Resume from baseline, clean
# raw PPO (no SEARCH_OP / candidate_q / de-myopia), train vs the eval target
# (CP7 skill-1) in chunks, eval vs Grixis after each chunk -> read the SLOPE.
# Climbing => undertrained (commit to long run). Flat => investigate.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/spy_trendcheck_RESULT.log"
$tlog = "local-training/spy_trendcheck_train.log"
$telog= "local-training/spy_trendcheck_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 150 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 4 }

# Route ONNX inference to the IDLE GPU1 so it doesn't contend with PyTorch
# training on GPU0 (contention was pinning inference at ~100ms -> 0.17 eps/s).
$env:INFER_CUDA_DEVICE="cuda:1"
$env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Eval-Now($rid) {
  # eval the CURRENT profile model (read-only) vs Grixis skill-1
  $e = @{}; foreach($k in 'SEARCH_OP_ENABLE','CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_REPLAY_DIR','REFERENCE_POLICY_KL_COEF','CANDIDATE_Q_ONLY','SPY_FINISH_GATE'){ if(Test-Path "Env:\$k"){$e[$k]=(Get-Item "Env:\$k").Value; Remove-Item "Env:\$k" -ErrorAction SilentlyContinue} }
  $env:USE_TRT_INFERENCE="0"
  "=== eval $rid $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 5151 --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $e.Keys){ Set-Item "Env:\$k" $e[$k] }
}

"=== SPY TREND-CHECK started $(Get-Date) ; ${nChunks} chunks x ${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# clean start point = pristine baseline (train_step 131818, entropy ~0.238)
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored (t=0 point)" | Out-File $out -Append

# t=0 baseline eval
Eval-Now "trend_c0_baseline"

# CLEAN raw-PPO training env (no algorithmic interventions), vs CP7 skill-1
$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\CANDIDATE_Q_REPLAY_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\CANDIDATE_Q_DUMP_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="ladder"; $env:LADDER_SKILLS="1"   # train vs the eval target (CP7 skill-1)
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="2500000"
  $env:ACTOR_LR="2e-4"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
}

for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) (resume from prior; train_step continues) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
    -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  # log current train_step + entropy from trainer.log
  $ts = (Select-String -Path "local-training/local_pbt/trainer.log" -Pattern "train_step|entropy_coef|episodes:" -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
  "chunk $c done $(Get-Date) | $ts" | Out-File $out -Append
  # snapshot + eval this chunk
  $snap = "local-training\backups\spy_trend_c$c"
  if (Test-Path $snap) { Remove-Item $snap -Recurse -Force }
  New-Item -ItemType Directory -Path $snap -Force | Out-Null
  Copy-Item "$md\model_latest.pt" "$snap\model_latest.pt" -Force
  Eval-Now "trend_c$c"
}

# final: 2nd-seed eval for variance + funnels (the loop already did seed-5151 eval of the last chunk)
$gl = "local-training/local_pbt/cp7_eval_sweeps"
"=== final 2nd-seed eval + funnels ===" | Out-File $out -Append
$env:USE_TRT_INFERENCE="0"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 128 --parallel 4 --eval-game-logging `
  --replay-metadata --skip-compile --replay-seed-base 9999 --run-id "trend_final_s9999" 2>&1 |
  Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label TREND_FINAL "$gl/trend_c${nChunks}/game_logs" 2>&1 | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label BASELINE0 "$gl/trend_c0_baseline/game_logs" 2>&1 | Out-File $out -Append

"=== SPY TREND-CHECK DONE $(Get-Date) ===" | Out-File $out -Append
