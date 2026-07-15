# R0 observable-router eval (Codex #43/#44): the agent pilots with the 7k GENERALIST by default
# and COMMITS to the 15k SPECIALIST (action model only) once its belief head detects Rally
# (>=0.80). Mulligan + pre-detect turns stay on the generalist (mulligan is a separate net).
# 512-game fixed CP-skill1 gauntlet, greedy, paired seed 5151. Compares vs 7k baseline (50.4%)
# and the oracle upper bound (53.1%, computed). Params: $env:ROUTER_THRESHOLD (default 0.80).
param(
  [string]$Out = "local-training/affinity_router_eval_RESULT.log",
  [string]$RunId = "affinity_router_eval"
)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$baseReg  = "local-training/_brew_win_registry.json"
$reg      = "local-training/_affinity_gauntlet_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md       = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Affinity-Anchor-Value\models"
$gen7k    = "E:/mage-training/backups/affinity_ar_league_7k_snap/model.pt"   # DEFAULT generalist (362c17ff)
$spec15k  = "E:/mage-training/backups/affinity_ar_league_15k/model.pt"       # SPECIALIST (923AF687)
$thr      = if ($env:ROUTER_THRESHOLD) { $env:ROUTER_THRESHOLD } else { "0.80" }

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'java' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt|run_cp7') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== ROUTER EVAL $RunId thr=$thr $(Get-Date) ===" | Out-File $Out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $gen7k))  { "FATAL: generalist missing" | Out-File $Out -Append; exit 1 }
if (-not (Test-Path $spec15k)){ "FATAL: specialist missing" | Out-File $Out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $gen7k "$md\model.pt" -Force            # primary = generalist
Copy-Item $gen7k "$md\model_latest.pt" -Force
"generalist md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash.Substring(0,8)) specialist=$((Get-FileHash $spec15k -Algorithm MD5).Hash.Substring(0,8)) thr=$thr" | Out-File $Out -Append
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Affinity-Anchor-Value': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2); print('gauntlet registry written')
" 2>&1 | Out-File $Out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','REFERENCE_ANCHOR_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
# --- R0 ROUTER config (reaches both the GPU service [loads specialist] and the game JVMs [ROUTER_ENABLE]) ---
$env:ROUTER_ENABLE="1"
$env:ROUTER_SPECIALIST_MODEL_PATH="$spec15k"
$env:ROUTER_TARGET_ARCHETYPE="1"        # Rally
$env:ROUTER_BELIEF_THRESHOLD="$thr"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Affinity-Anchor-Value `
  --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill 1 --games-per-matchup 64 --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "$RunId" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $Out -Append
"=== ROUTER EVAL $RunId DONE $(Get-Date) ===" | Out-File $Out -Append
