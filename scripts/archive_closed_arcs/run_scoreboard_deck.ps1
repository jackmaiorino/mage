# SCOREBOARD entry: one agent deck vs the CP7 skill-7 8-deck gauntlet, deterministic.
# Triage: play-side only, n=128/matchup (find gaps); confirm gaps balanced n=256 later.
# Usage: run_scoreboard_deck.ps1 -Profile <prof> -Backup <model.pt> -Out <log> -RunId <id> [-Skill 7] [-N 128]
param(
  [Parameter(Mandatory=$true)][string]$Profile,
  [Parameter(Mandatory=$true)][string]$Backup,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][string]$RunId,
  [int]$Skill = 7,
  [int]$N = 128
)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$baseReg = "local-training/_brew_win_registry.json"
$reg  = "local-training/_scoreboard_$($Profile)_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\$Profile\models"
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"=== SCOREBOARD $RunId profile=$Profile skill=$Skill N=$N $(Get-Date) ===" | Out-File $Out
if (-not (Test-Path $Backup)) { "FATAL: backup missing $Backup" | Out-File $Out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $Backup "$md\model.pt" -Force; Copy-Item $Backup "$md\model_latest.pt" -Force
"restored md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash)" | Out-File $Out -Append
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')==r'$Profile': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2)
print('scoreboard registry written for $Profile')
" 2>&1 | Out-File $Out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_FEATURES_ENABLE','EVAL_OPPONENT_ON_PLAY'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles $Profile `
  --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill $Skill --games-per-matchup $N --games-per-job 16 `
  --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "$RunId" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $Out -Append
"=== SCOREBOARD $RunId DONE $(Get-Date) ===" | Out-File $Out -Append
