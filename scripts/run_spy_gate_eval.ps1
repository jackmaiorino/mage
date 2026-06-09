# DIAGNOSTIC counterfactual: does gating Balustrade Spy to finish-ready states
# (>=3 creatures in play + landless library) raise winrate? If yes, "premature
# all-in Spy -> self-deck" is the confirmed leak. Same baseline model both arms;
# only env SPY_FINISH_GATE differs. Paired seeds 5151 + 9999, n=128.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/spy_gate_eval_RESULT.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak = "local-training\backups\spy_value_baseline_20260531"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== SPY FINISH-GATE counterfactual started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3

# restore pristine baseline (both arms use the SAME model)
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored" | Out-File $out -Append

$env:SEARCH_OP_ENABLE="0"; $env:CANDIDATE_Q_ONLY="0"; $env:CANDIDATE_Q_BLEND="0"; $env:USE_TRT_INFERENCE="0"

function Run-Eval($rid, $seed) {
  "=== eval $rid (seed $seed, SPY_FINISH_GATE=$($env:SPY_FINISH_GATE)) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}

# --- ARM A: gate OFF (control) ---
Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
Run-Eval "spygate_off_s5151" 5151
Run-Eval "spygate_off_s9999" 9999

# --- ARM B: gate ON ---
$env:SPY_FINISH_GATE="1"
Run-Eval "spygate_on_s5151" 5151
Run-Eval "spygate_on_s9999" 9999

# --- funnels: did combo conversion improve under the gate? ---
$gl = "local-training/local_pbt/cp7_eval_sweeps"
foreach ($s in @("5151","9999")) {
  "=== FUNNEL gate-OFF s$s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label OFF$s "$gl/spygate_off_s$s/game_logs" 2>&1 | Out-File $out -Append
  "=== FUNNEL gate-ON s$s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label ON$s "$gl/spygate_on_s$s/game_logs" 2>&1 | Out-File $out -Append
}
"=== SPY FINISH-GATE DONE $(Get-Date) ===" | Out-File $out -Append
