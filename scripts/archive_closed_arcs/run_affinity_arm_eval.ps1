# Parametrized matched-seed gauntlet eval for the win-replay paired experiment.
# Restores a given backup model into the Affinity profile, then evals vs the 8-deck
# gauntlet at the SAME replay-seed-base (5151) as Arm A/B so the comparison is paired.
# Usage: run_affinity_arm_eval.ps1 -Backup <path> -Out <resultlog> -RunId <id>
param(
  [Parameter(Mandatory=$true)][string]$Backup,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][string]$RunId
)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$baseReg = "local-training/_brew_win_registry.json"
$reg  = "local-training/_affinity_gauntlet_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Affinity-Anchor-Value\models"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== ARM EVAL $RunId $(Get-Date) ===" | Out-File $Out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $Backup)) { "FATAL: backup missing: $Backup" | Out-File $Out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $Backup "$md\model.pt" -Force
Copy-Item $Backup "$md\model_latest.pt" -Force
$h = (Get-FileHash "$md\model.pt" -Algorithm MD5).Hash
"restored backup=$Backup md5=$h" | Out-File $Out -Append
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Affinity-Anchor-Value': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2)
print('affinity-gauntlet registry written')
" 2>&1 | Out-File $Out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Affinity-Anchor-Value `
  --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill 1 --games-per-matchup 64 --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "$RunId" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $Out -Append
"=== ARM EVAL $RunId DONE $(Get-Date) ===" | Out-File $Out -Append
