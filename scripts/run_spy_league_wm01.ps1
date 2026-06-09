# WM01: world-model aux (value de-myopia) on the FAITHFUL 0.52 platform. The cheapest clean
# test of "can a value-de-myopia lever climb above 0.52" now that the diet no longer decays.
# Multi-profile faithful league (holds 0.52) + WORLD_MODEL_LOSS_COEF=0.1, no mulligan freeze.
# Eval Grixis-skill1 + castable-Spy each chunk; AUTO-ABORT if castable-Spy collapses (<0.55) --
# the prior WM01 failure mode was castable-Spy 74->56 from shared-encoder drift.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/wm01_RESULT.log"
$tlog = "local-training/wm01_train.log"; $telog = "local-training/wm01_train.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 35 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 4 }
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function CastSpy($rid) {
  $r = (py -3.12 scripts/mtgrl/eval_report.py --run "$GL/$rid" 2>$null | Select-String "funnel")
  if ($r -match "'spy':\s*([0-9.]+)") { return [double]$Matches[1] } else { return -1.0 }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER','WORLD_MODEL_LOSS_COEF','WORLD_MODEL_LABELS_ENABLE'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  $env:WORLD_MODEL_LOSS_COEF="0.0"; $env:WORLD_MODEL_LABELS_ENABLE="0"   # clean eval, no aux
  "=== eval $rid $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}

"=== WM01 (world-model aux coef=0.1) on faithful diet started $(Get-Date) ; ${nChunks}x${chunkMin}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
Copy-Item "$bak\model.pt" "$md\model.pt" -Force; Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }; Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored (continue point ~0.52)" | Out-File $out -Append
Eval-Now "wm01_c0" 5151
$c0spy = CastSpy "wm01_c0"; "c0 castable-Spy=$c0spy" | Out-File $out -Append

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  # faithful high-equilibrium diet (the one that holds 0.52)
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  # THE LEVER: world-model aux, small coef, encoder-on, NO mulligan freeze
  $env:WORLD_MODEL_LABELS_ENABLE="1"; $env:WORLD_MODEL_DIM="18"; $env:WORLD_MODEL_LOSS_COEF="0.1"
  $env:WORLD_MODEL_DIAG="1"   # print [WORLD_MODEL_DIAG] valid=.../... so we can VERIFY the aux fires
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="4"; $env:NUM_GAME_RUNNERS="64"; $env:TOTAL_EPISODES="99999999"
}

for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) (faithful league + WM aux 0.1) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  Eval-Now "wm01_c$c" 5151
  $spy = CastSpy "wm01_c$c"
  "chunk $c done $(Get-Date) | castable-Spy=$spy" | Out-File $out -Append
  if ($spy -ge 0 -and $spy -lt 0.55) { "=== AUTO-ABORT: castable-Spy collapsed to $spy (< 0.55) -- WM shared-encoder drift, the known failure mode ===" | Out-File $out -Append; break }
}
"=== WM01 DONE $(Get-Date) ===" | Out-File $out -Append
