# 1) Eval the checkpoint-average (n=192, seed 5151) vs model_best's known 0.582@s5151/n256.
# 2) Launch the overnight sustained hybrid run from whichever is stronger.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/avg_eval_RESULT.log"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$ck  = "local-training\backups\sustained_20260611"
$GL  = "local-training/local_pbt/cp7_eval_sweeps"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== AVG-CHECKPOINT EVAL started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 5
Copy-Item "$ck\model_avg.pt" "$md\model_latest.pt" -Force
Copy-Item "$ck\model_avg.pt" "$md\model.pt" -Force
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 192 --parallel 8 --eval-game-logging --replay-metadata `
  --skip-compile --replay-seed-base 5151 --run-id avg_s5151 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
$m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/avg_s5151" --label avg_s5151 2>&1 | Out-String).Trim()
">>> $m" | Out-File $out -Append

# Decide start model: avg needs to beat model_best's s5151 showing (0.582 at n=256)
$startModel = "$ck\model_best.pt"
$csv = "$GL/avg_s5151/matchups.csv"
if (Test-Path $csv) {
  $row = Import-Csv $csv | Select-Object -First 1
  $wr = [double]$row.wins / [double]$row.total
  if ($wr -ge 0.582) { $startModel = "$ck\model_avg.pt" }
  "avg s5151 wr=$wr -> overnight starts from $startModel" | Out-File $out -Append
} else {
  "avg eval missing; overnight starts from model_best" | Out-File $out -Append
}
"=== AVG EVAL DONE $(Get-Date) ===" | Out-File $out -Append

$env:OVERNIGHT_MODE="hybrid"; $env:NUM_CHUNKS="12"; $env:CHUNK_MIN="45"
$env:START_MODEL=$startModel; $env:CKPT_DIR="local-training\backups\sustained_20260612"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_overnight_dominance.ps1
