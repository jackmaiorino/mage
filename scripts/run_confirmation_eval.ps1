# DOMINANCE CONFIRMATION EVAL: model_best (0.594 @ n=96) at n=256 x 3 seeds.
# Pooled n=768 -> CI ~ +/-3.5pp. Gate (roadmap): wr >= 0.60 with CI-low >= 0.55.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/confirmation_RESULT.log"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$ck  = "local-training\backups\sustained_20260611"
$GL  = "local-training/local_pbt/cp7_eval_sweeps"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== CONFIRMATION EVAL started $(Get-Date) ; model_best (0.594@n96) ; n=256 x seeds 5151/9999/7777 ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 5
Copy-Item "$md\model_latest.pt" "$ck\model_final_c10.pt" -Force
Copy-Item "$ck\model_best.pt" "$md\model_latest.pt" -Force
Copy-Item "$ck\model_best.pt" "$md\model.pt" -Force
"final c10 model preserved; model_best promoted for eval" | Out-File $out -Append

$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
foreach ($seed in 5151, 9999, 7777) {
  $rid = "confirm_s$seed"
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 256 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  $m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim()
  ">>> $m" | Out-File $out -Append
}
# Pooled verdict
py -3.12 -c "
import csv, math
tot = wins = 0
for s in (5151, 9999, 7777):
    try:
        with open(f'local-training/local_pbt/cp7_eval_sweeps/confirm_s{s}/matchups.csv') as f:
            r = next(csv.DictReader(f))
            wins += int(r['wins']); tot += int(r['total'])
    except Exception as e:
        print(f'seed {s}: missing ({e})')
if tot:
    p = wins / tot
    se = math.sqrt(p * (1 - p) / tot)
    print(f'POOLED: {wins}/{tot} wr={p:.4f}  95%CI=[{p-1.96*se:.4f}, {p+1.96*se:.4f}]')
    print(f'GATE wr>=0.60 & CI-low>=0.55: {\"PASS\" if p>=0.60 and p-1.96*se>=0.55 else \"not yet\"}')
" 2>&1 | Out-File $out -Append
"=== CONFIRMATION EVAL DONE $(Get-Date) ===" | Out-File $out -Append
