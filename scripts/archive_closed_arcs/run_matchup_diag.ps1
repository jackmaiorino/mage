# MATCHUP DIAGNOSTIC: eval the best full-winning model (Spy tier-1 deck) vs a SPREAD of
# opponent archetypes to answer proficiency-ceiling vs matchup-ceiling. If the Spy deck
# DOMINATES slow decks (control/midrange -> time to combo) but loses to AGGRO (raced),
# the ~0.50 is a matchup/race property. If it caps ~0.50 vs EVERYTHING, it's general
# under-piloting (proficiency). Fixed model, skill-1, seed 5151, 96g each.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/matchup_diag_RESULT.log"
$reg = "local-training/_brew_win_registry.json"   # Spy Winning agent deck
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\brew_winning_20260619\model_best.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== MATCHUP DIAG (best full-winning model) $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $best)) { "FATAL: best model missing $best" | Out-File $out -Append; exit 1 }
# diag registry: Spy Winning agent deck + deck_path -> all-archetype opponent pool
$diagReg = "local-training/_matchup_diag_registry.json"
py -3.12 -c "
import json
r=json.load(open(r'$reg'))
pool='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
for e in r:
    if e.get('profile')=='Pauper-Spy-Combo-Value':
        e['deck_path']=pool
json.dump(r,open(r'$diagReg','w'),indent=2)
print('diag registry: opponent pool ->', pool)
" 2>&1 | Out-File $out -Append
$reg = $diagReg
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"

# opponent name filters (substring-match deck names) spanning archetypes
$opps = @('grixis','burn','rally','faeries','terror','wildfire','caw','elves')
foreach ($o in $opps) {
  "--- vs $o $(Get-Date) ---" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents $o --skill 1 --games-per-matchup 96 --parallel 8 `
    --skip-compile --replay-seed-base 5151 --run-id "mx_$o" 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}
"=== MATCHUP DIAG DONE $(Get-Date) ===" | Out-File $out -Append
"--- SUMMARY ---" | Out-File $out -Append
Get-Content $out | Where-Object { $_ -match 'wr=' } | Out-File $out -Append
