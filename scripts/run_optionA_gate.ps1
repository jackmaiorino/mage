# Option-A GATE: does terminal-rollout candidate_q de-myopia the SHARED encoder
# WITHOUT degrading execution (the failure mode that killed WM-aux)?
# Encoder UNFROZEN (candidate_q reshapes it), candidate_q 4x value, KL-anchored to
# the frozen baseline policy to protect execution. Then eval de-myopia'd vs baseline
# and run the leading (value-AUC) + execution-guard (castable-Spy/durdle) probes.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out   = "local-training/optionA_gate_RESULT.log"
$tlog  = "local-training/optionA_gate_train.log"
$telog = "local-training/optionA_gate_train.err"
$md    = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak   = "local-training\backups\spy_value_baseline_20260531"
$demy  = "local-training\backups\spy_value_optionA_gate_20260602"
$refabs = (Resolve-Path "$bak\model_latest.pt").Path
$mins  = [int]($env:GATE_TRAIN_MINUTES); if ($mins -le 0) { $mins = 180 }
$iters = [int]($mins / 15)

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== Option-A GATE started $(Get-Date) ; train window ${mins}min ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# clean start from pristine baseline
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored; reference anchor = $refabs" | Out-File $out -Append

# --- TRAINING ENV: Option A (encoder UNFROZEN, candidate_q reshapes shared encoder) ---
$env:SEARCH_OP_ENABLE="1"; $env:SEARCH_OP_TOP_K="3"; $env:SEARCH_OP_PLAYOUTS="4"
$env:SEARCH_OP_PLAYOUT_TIMEOUT_MS="2500"; $env:SEARCH_OP_TOTAL_TIMEOUT_MS="15000"
$env:SEARCH_OP_MAX_ACTIVATIONS="6"; $env:SEARCH_OP_MAX_GAME_TURNS="40"
$env:SEARCH_OP_GENERIC_BRANCH_ORDER="true"; $env:SEARCH_OP_SKIP_TOP_PROB="0.85"; $env:SEARCH_OP_LOG="1"
$env:CANDIDATE_Q_FROM_MCTS_TARGETS="1"; $env:CANDIDATE_Q_MCTS_SIGNED_TARGETS="1"
$env:CANDIDATE_Q_LOSS_COEF="4.0"; $env:CANDIDATE_Q_HUBER_BETA="0.25"
$env:CANDIDATE_Q_ONLY="0"                       # encoder UNFROZEN <-- the whole point vs Option B
$env:VALUE_USE_SEPARATE_CRITIC_ENCODER="0"      # de-myopia reaches shared encoder + PPO advantages
$env:VALUE_LOSS_COEF="1.0"; $env:VALUE_LOSS_COEF_WARMUP="5.0"  # lowered so candidate_q can win the encoder
$env:REFERENCE_POLICY_KL_COEF="0.3"             # protect execution (the WM-aux killer)
$env:MCTS_REFERENCE_MODEL_PATH="$refabs"
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="10"; $env:TOTAL_EPISODES="99999999"
$env:MULTI_PLY_MCTS="0"; $env:MCTS_TRAINING_ENABLE="0"

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
"training launched $(Get-Date)" | Out-File $out -Append

# accumulate targets; log every 15 min
for ($i = 0; $i -lt $iters; $i++) {
  Start-Sleep -Seconds 900
  $trl = "local-training/local_pbt/trainer.log"
  $obs = 0
  if (Test-Path $trl) {
    $obs = (Select-String -Path $trl -Pattern "observed=([2-9]|[1-9][0-9])" -AllMatches -ErrorAction SilentlyContinue | Measure-Object).Count
  }
  "[$([int](($i+1)*15)) min] approx target rows (observed>=2): $obs   $(Get-Date)" | Out-File $out -Append
}

"training window done $(Get-Date); stopping trainer" | Out-File $out -Append
Kill-Train; Start-Sleep -Seconds 8

# preserve the de-myopia'd model (profile currently holds it)
if (Test-Path $demy) { Remove-Item $demy -Recurse -Force }
New-Item -ItemType Directory -Path $demy -Force | Out-Null
Copy-Item "$md\model.pt"        "$demy\model.pt"        -Force
Copy-Item "$md\model_latest.pt" "$demy\model_latest.pt" -Force
"de-myopia'd model backed up to $demy" | Out-File $out -Append

# did the encoder actually move (Option A) + candidate_q learn?
"=== weight diff vs baseline (encoder >0 = REshaped; candidate_q >0 = learned) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/qhead_weight_diff.py 2>&1 | Out-File $out -Append

# --- EVAL (search OFF, native model behavior) ---
$env:SEARCH_OP_ENABLE="0"; $env:CANDIDATE_Q_ONLY="0"; $env:CANDIDATE_Q_BLEND="0"
$env:USE_TRT_INFERENCE="0"; Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"

function Run-Eval($rid) {
  "=== eval $rid $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 5151 --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}

# eval de-myopia'd model (profile holds it now)
Run-Eval "optionA_demyopia"

# restore baseline, eval it at the same seed for a paired comparison
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
Run-Eval "optionA_baseline"

# --- PROBES: de-myopia (leading) + execution guard (the gate's verdict) ---
$gl = "local-training/local_pbt/cp7_eval_sweeps"
$dGL = "$gl/optionA_demyopia/game_logs"
$bGL = "$gl/optionA_baseline/game_logs"
"=== DE-MYOPIA: value-AUC (demyopia=wm vs baseline) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/value_auc_bootstrap.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
"=== EXECUTION GUARD: castable-Spy (WM-aux killed this 74->58%) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/castable_spy_diagnostic.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
"=== DURDLE: never-cast-Spy tempo (demyopia) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/never_cast_tempo.py $dGL 2>&1 | Out-File $out -Append
"=== DURDLE: never-cast-Spy tempo (baseline) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/never_cast_tempo.py $bGL 2>&1 | Out-File $out -Append
"=== FUNNEL demyopia ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label DEMYOPIA $dGL 2>&1 | Out-File $out -Append
"=== FUNNEL baseline ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label BASELINE $bGL 2>&1 | Out-File $out -Append

"=== Option-A GATE DONE $(Get-Date) ===" | Out-File $out -Append
