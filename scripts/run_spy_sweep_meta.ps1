# SWEEP meta-diet: Spy CONTINUE from baseline, but OPPONENT_SAMPLER=meta (multi-profile)
# so Spy spars vs aggressive NN opponents (Rally/Wildfire) + eval-aligned Grixis-Affinity
# + self, instead of the passive Spy mirror (which was DEGENERATE: castable 76->48%,
# winrate 50->30%). Hypothesis: aggressive diet -> clock pressure -> reachability holds.
# Eval Spy vs Grixis after each chunk -> compare castable-Spy trajectory to self-play.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/sweep_meta_RESULT.log"
$tlog = "local-training/sweep_meta_train.log"
$telog= "local-training/sweep_meta_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 5 }

$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER','TRAIN_PROFILES'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  "=== eval $rid (seed $seed) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}

"=== SWEEP meta-diet (Spy continue, OPPONENT_SAMPLER=meta, 4 profiles) started $(Get-Date) ; ${nChunks}x${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# restore ONLY the Spy baseline (clean 50% start); Rally/Wildfire/Affinity keep their
# existing trained models as the aggressive sparring opponents.
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"Spy baseline restored; meta opponents = Rally/Wildfire/Affinity (May-14 models)" | Out-File $out -Append

Eval-Now "meta_c0" 5151

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="meta"           # AGGRESSIVE/DIVERSE NN diet (vs degenerate self-play mirror)
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="180000"
  $env:ACTOR_LR="2e-4"
  $env:ONNX_BATCH_TIMEOUT_MS="5"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="10"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="4"; $env:NUM_GAME_RUNNERS="64"; $env:TOTAL_EPISODES="99999999"
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
  Eval-Now "meta_c$c" 5151
}
"=== SWEEP meta-diet DONE $(Get-Date) ===" | Out-File $out -Append
