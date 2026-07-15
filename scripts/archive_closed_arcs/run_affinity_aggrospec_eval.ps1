# DECK #2: Grixis Affinity DOMINANCE baseline vs the 8-deck gauntlet (clean greedy eval, skill 1).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/affinity_aggrospec_eval_RESULT.log"
$baseReg = "local-training/_brew_win_registry.json"
$reg  = "local-training/_affinity_gauntlet_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Affinity-Anchor-Value\models"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== AFFINITY GAUNTLET BASELINE $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path "$md\model.pt")) { "FATAL: affinity model missing" | Out-File $out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$md\model.pt" "$md\model_latest.pt" -Force
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Affinity-Anchor-Value': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2)
print('affinity-gauntlet registry written')
" 2>&1 | Out-File $out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Affinity-Anchor-Value `
  --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill 1 --games-per-matchup 64 --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "affinity_aggrospec_eval" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== AFFINITY GAUNTLET BASELINE DONE $(Get-Date) ===" | Out-File $out -Append
