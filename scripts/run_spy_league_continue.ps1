# DIET-FIX CONFIRMATION: continue the 50% baseline under the LEAGUE diet (the diet that
# BUILT the baseline: fixed META snapshots + CP7-skill1 + self-snapshots) and check it
# HOLDS/CLIMBS -- vs the known SELF-PLAY degradation (armS: 50->34->34->32->29).
# If league holds ~50% while self dropped to ~30%, the regression is the diet, not the pipeline.
# Everything else identical to the armS self-play diagnostic (size S, GPU split, clean eval).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/league_continue_RESULT.log"
$tlog = "local-training/league_continue_train.log"
$telog= "local-training/league_continue_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 30 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 3 }

$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER','CANDIDATE_Q_LOSS_COEF'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  "=== eval $rid (seed $seed) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}

"=== LEAGUE-CONTINUE diet-fix test started $(Get-Date) ; ${nChunks}x${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# restore pristine baseline (continue point ~50%)
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored (continue point)" | Out-File $out -Append

Eval-Now "league_c0" 5151   # baseline anchor

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  # THE FIX (registry-parser repaired -> candidates resolve): league diet = META-H (CP7 on
  # the aggressive pool decks Wildfire/Rally/Affinity = a fixed CLOCK), NOT self-play. skill 3
  # so the CP7 pilot is a real clock; promote threshold 0.40 (< model's 0.50).
  $env:OPPONENT_SAMPLER="league"
  $env:LEAGUE_PROMOTE_WR="0.40"
  $env:LEAGUE_POST_HEURISTIC_SKILL="3"
  $env:LEAGUE_MODE=""
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"   # size S (matches baseline)
  # entropy: do NOT override -> use registry schedule (baseline regime). Keep all else baseline-default.
  $env:ONNX_BATCH_TIMEOUT_MS="5"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="10"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
}

for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) (OPPONENT_SAMPLER=league) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
    -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  $ts = (Select-String -Path "local-training/local_pbt/trainer.log" -Pattern "episodes:|train_step|opponent" -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
  "chunk $c done $(Get-Date) | $ts" | Out-File $out -Append
  Eval-Now "league_c$c" 5151
}
"=== LEAGUE-CONTINUE DONE $(Get-Date) ===" | Out-File $out -Append
