# DECOUPLE phase A (GENERATION): run the gate config (SEARCH_OP on) with the
# dump hook enabled to accumulate a candidate_q target dataset (npz shards) for
# offline replay. Starts from baseline. Model drift here is irrelevant -- only
# the dumped npz are kept; the mixed-mode run restores baseline.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/optionA_decouple_gen.log"
$tlog = "local-training/optionA_decouple_gen_train.log"
$telog= "local-training/optionA_decouple_gen_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$dump = "C:\Users\Jack\IdeaProjects\mage\local-training\cq_dump"
$refabs = (Resolve-Path "$bak\model_latest.pt").Path
$mins = [int]($env:GEN_MINUTES); if ($mins -le 0) { $mins = 120 }
$iters = [int]($mins / 5)

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== DECOUPLE generation started $(Get-Date) ; window ${mins}min ; dump=$dump ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path $dump) { Remove-Item $dump -Recurse -Force }
New-Item -ItemType Directory -Path $dump -Force | Out-Null

Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored" | Out-File $out -Append

# gate (Option-A) config + DUMP
$env:SEARCH_OP_ENABLE="1"; $env:SEARCH_OP_TOP_K="3"; $env:SEARCH_OP_PLAYOUTS="4"
$env:SEARCH_OP_PLAYOUT_TIMEOUT_MS="2500"; $env:SEARCH_OP_TOTAL_TIMEOUT_MS="15000"
$env:SEARCH_OP_MAX_ACTIVATIONS="6"; $env:SEARCH_OP_MAX_GAME_TURNS="40"
$env:SEARCH_OP_GENERIC_BRANCH_ORDER="true"; $env:SEARCH_OP_SKIP_TOP_PROB="0.85"; $env:SEARCH_OP_LOG="1"
$env:CANDIDATE_Q_FROM_MCTS_TARGETS="1"; $env:CANDIDATE_Q_MCTS_SIGNED_TARGETS="1"
$env:CANDIDATE_Q_LOSS_COEF="4.0"; $env:CANDIDATE_Q_HUBER_BETA="0.25"; $env:CANDIDATE_Q_ONLY="0"
$env:VALUE_USE_SEPARATE_CRITIC_ENCODER="0"; $env:VALUE_LOSS_COEF="1.0"; $env:VALUE_LOSS_COEF_WARMUP="5.0"
$env:REFERENCE_POLICY_KL_COEF="0.3"; $env:MCTS_REFERENCE_MODEL_PATH="$refabs"
$env:CANDIDATE_Q_DUMP_DIR="$dump"
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="10"; $env:TOTAL_EPISODES="99999999"
$env:MULTI_PLY_MCTS="0"; $env:MCTS_TRAINING_ENABLE="0"
Remove-Item Env:\CANDIDATE_Q_REPLAY_DIR -ErrorAction SilentlyContinue

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
"generation launched $(Get-Date)" | Out-File $out -Append

for ($i = 0; $i -lt $iters; $i++) {
  Start-Sleep -Seconds 300
  $n = 0
  if (Test-Path $dump) { $n = (Get-ChildItem "$dump\cq_*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count }
  "[$([int](($i+1)*5)) min] npz shards: $n   $(Get-Date)" | Out-File $out -Append
}

"generation window done $(Get-Date); stopping" | Out-File $out -Append
Kill-Train; Start-Sleep -Seconds 6
$n = (Get-ChildItem "$dump\cq_*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
"=== DECOUPLE generation DONE $(Get-Date) ; total npz shards: $n ===" | Out-File $out -Append
