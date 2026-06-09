# FROM-SCRATCH REPLICATION: can the CURRENT (registry-parser-fixed) code + the faithful
# high-equilibrium diet build a FRESH random Spy up toward ~0.52, reproducing the good era?
# Fresh Spy already written + all 4 profiles promoted (so league = META-RL trained-NN +
# MIRROR frozen self-snapshots + CP7, NOT CP7-only). Multi-profile so meta NN models are served.
# Annealing entropy: explore early (discover the combo), sharpen later. Eval Spy vs Grixis each
# chunk -> learning CURVE. PASS = climbs from ~chance toward 0.4-0.5 (pipeline healthy + replicable).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/league_fromscratch_RESULT.log"
$tlog = "local-training/league_fromscratch_train.log"
$telog= "local-training/league_fromscratch_train.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$mdSpy= "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 8 }

$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER','ENTROPY_START','ENTROPY_END'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  "=== eval $rid (seed $seed) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}

"=== FROM-SCRATCH league replication started $(Get-Date) ; ${nChunks}x${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# fresh Spy + promotions already done in bash. c0 = fresh-model floor (~chance).
Eval-Now "fs_c0" 5151

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  # faithful high-equilibrium diet (registry-parser fix -> candidates resolve)
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  # annealing entropy: explore early to discover the combo, sharpen later
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="4"; $env:NUM_GAME_RUNNERS="64"; $env:TOTAL_EPISODES="99999999"
}

for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) (league multi-profile, annealing entropy) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
    -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  $ts = (Select-String -Path "local-training/local_pbt/trainer.log" -Pattern "episodes:|train_step" -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
  "chunk $c done $(Get-Date) | $ts" | Out-File $out -Append
  Eval-Now "fs_c$c" 5151
}
"=== FROM-SCRATCH league DONE $(Get-Date) ===" | Out-File $out -Append
