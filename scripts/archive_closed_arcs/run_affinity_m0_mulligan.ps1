# M0: Forced Mulligan Counterfactual Audit (Codex #35), Affinity vs Rally (aggro slice).
# Greedy eval (exploration off, MODE=eval), matched seeds. Run the matchup once forcing
# KEEP at the first mulligan and once forcing MULL; join by opening hand -> compare terminal
# outcomes. Provenance printed (Codex hard rule). Param: $env:M0_GAMES (default 128).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$baseReg = "local-training/_brew_win_registry.json"
$reg  = "local-training/_m0_registry.json"
$gauntlet = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Affinity-Anchor-Value\models"
$ref  = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"
$gamesDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/logs/games"
$games = if ($env:M0_GAMES) { $env:M0_GAMES } else { "128" }

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

# restore ref C + provenance
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { Write-Output "FATAL: ref missing"; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $ref "$md\model.pt" -Force; Copy-Item $ref "$md\model_latest.pt" -Force
$h = (Get-FileHash "$md\model.pt" -Algorithm MD5).Hash
Write-Output "M0 PROVENANCE: ref=$ref md5=$h greedy=eval(argmax) exploration=off opponent=Rally(CP7-skill1) seed-base=5151 games=$games"
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Affinity-Anchor-Value': e['deck_path']=r'$gauntlet'
json.dump(r,open(r'$reg','w'),indent=2)
print('m0 registry written')
"
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','SEARCH_OP_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"

function Run-M0([int]$force, [string]$tag) {
  if (Test-Path $gamesDir) { Get-ChildItem $gamesDir -Recurse -Filter *.txt | Remove-Item -Force -ErrorAction SilentlyContinue }
  $env:MULLIGAN_M0_FORCE = "$force"
  Write-Output "=== M0 run tag=$tag MULLIGAN_M0_FORCE=$force $(Get-Date) ==="
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Affinity-Anchor-Value `
    --opponents "rally" --skill 1 --games-per-matchup $games --parallel 8 --games-per-job 16 `
    --eval-game-logging --replay-metadata --skip-compile --replay-seed-base 5151 --run-id "affinity_m0_$tag" 2>&1 |
    Select-String "wr=" | ForEach-Object { $_.Line }
  $dest = "local-training/m0_$tag"
  if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
  New-Item -ItemType Directory -Force $dest | Out-Null
  Get-ChildItem $gamesDir -Recurse -Filter *.txt | Copy-Item -Destination $dest -Force -ErrorAction SilentlyContinue
  $n = (Get-ChildItem $dest -Filter *.txt | Measure-Object).Count
  Write-Output "captured $n game logs -> $dest"
}

Run-M0 1 "keep"
Run-M0 2 "mull"
Remove-Item "Env:\MULLIGAN_M0_FORCE" -ErrorAction SilentlyContinue
Write-Output "=== M0 DONE $(Get-Date) ==="
