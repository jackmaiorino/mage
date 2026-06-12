# Option-A 2nd-seed CONFIRM: re-eval the saved de-myopia'd model vs baseline at
# seed 9999 (gate used 5151) + re-probe value-AUC + castable-Spy. NO retrain.
# Replicates the de-myopia AUC before spending HPC SU on the decouple.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/optionA_confirm_s9999_RESULT.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak = "local-training\backups\spy_value_baseline_20260531"
$dem = "local-training\backups\spy_value_optionA_gate_20260602"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== Option-A CONFIRM seed=9999 started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

$env:SEARCH_OP_ENABLE="0"; $env:CANDIDATE_Q_ONLY="0"; $env:CANDIDATE_Q_BLEND="0"; $env:USE_TRT_INFERENCE="0"

function Run-Eval($rid) {
  "=== eval $rid $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 9999 --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}

# de-myopia'd model
Copy-Item "$dem\model.pt"        "$md\model.pt"        -Force
Copy-Item "$dem\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
"de-myopia'd model loaded (onnx will re-export)" | Out-File $out -Append
Run-Eval "optionA_demyopia_s9999"

# baseline
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored" | Out-File $out -Append
Run-Eval "optionA_baseline_s9999"

# probes
$gl = "local-training/local_pbt/cp7_eval_sweeps"
$dGL = "$gl/optionA_demyopia_s9999/game_logs"
$bGL = "$gl/optionA_baseline_s9999/game_logs"
"=== DE-MYOPIA: value-AUC seed 9999 (demyopia=wm vs baseline) ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/value_auc_bootstrap.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
"=== EXECUTION GUARD: castable-Spy seed 9999 ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/castable_spy_diagnostic.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
"=== FUNNEL demyopia s9999 ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label DEMYOPIA9999 $dGL 2>&1 | Out-File $out -Append
"=== FUNNEL baseline s9999 ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label BASELINE9999 $bGL 2>&1 | Out-File $out -Append

"=== Option-A CONFIRM s9999 DONE $(Get-Date) ===" | Out-File $out -Append
