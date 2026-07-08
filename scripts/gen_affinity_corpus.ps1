# Temporal-history falsifier (Step 1) corpus generation.
# Runs the competent Affinity agent (Pauper-Affinity-Anchor-Value, ~46.6% s7) vs the
# 8-deck gauntlet, dumping per-decision rows (snapshot facts + card-id tokens + last-K
# public events, terminal outcome stamped) to E:\mage-training\corpus\affinity.
# One invocation per opponent so the matchup label is correct; each JVM writes its own
# shard (corpus_<opp>.<jvmtoken>.jsonl).
# Usage:  scripts/gen_affinity_corpus.ps1 [-Smoke] [-N 48] [-Parallel 8]
param(
  [switch]$Smoke,
  [int]$N = 48,
  [int]$Parallel = 8,
  [string]$Opponents = "grixis,burn,faeries,terror,wildfire,caw,elves,rally"
)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$profile  = "Pauper-Affinity-Anchor-Value"
$baseReg  = "local-training/_brew_win_registry.json"
$reg      = "local-training/_corpus_affinity_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md       = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\$profile\models"
$corpusDir = "E:\mage-training\corpus\affinity"
$log      = "local-training/gen_affinity_corpus.log"

if ($Smoke) { $N = 4; $Parallel = 4; $Opponents = "grixis,rally" }

Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt|run_cp7_eval_sweep') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
New-Item -ItemType Directory -Force $corpusDir | Out-Null
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }

"=== GEN AFFINITY CORPUS $(Get-Date) smoke=$Smoke N=$N par=$Parallel opps=$Opponents ===" | Out-File $log
"model md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash)" | Out-File $log -Append

# scoreboard-style registry: agent plays its own deck vs the gauntlet pool
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')==r'$profile': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2)
print('corpus registry written')
" 2>&1 | Out-File $log -Append

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_FEATURES_ENABLE','EVAL_OPPONENT_ON_PLAY'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:RL_CORPUS_DUMP="1"; $env:RL_CORPUS_EVENT_WINDOW="96"

$opps = $Opponents.Split(",")
foreach($opp in $opps){
  $env:RL_CORPUS_MATCHUP = $opp
  $env:RL_CORPUS_FILE = "$corpusDir\corpus_$opp"
  "=== opponent=$opp N=$N $(Get-Date) ===" | Out-File $log -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles $profile `
    --opponents $opp --skill 7 --games-per-matchup $N --games-per-job 8 --parallel $Parallel `
    --skip-compile --run-id "corpus_aff_$opp" 2>&1 |
    Select-String "wr=|EVAL_RESULT" | ForEach-Object { $_.Line } | Out-File $log -Append
}
"=== GEN DONE $(Get-Date) ===" | Out-File $log -Append
Get-ChildItem "$corpusDir\*.jsonl" | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName | Measure-Object -Line).Lines) rows" } | Out-File $log -Append
Get-Content $log -Tail 20
