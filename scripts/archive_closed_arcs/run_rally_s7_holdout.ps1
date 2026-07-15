# Phase-1 Rally re-baseline under the NEW deterministic measurement standard (jul1).
# Two checkpoints (post-c1 = the historical 67%; latest = armF-era Rally), full 8-deck
# gauntlet, skill 1, n=256/matchup, --deterministic-eval (serial; parallel breaks
# determinism via shared-GPU-service interleaving). Seed base 5151 = dev block.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Rally-Anchor-Value\models"
$out = "local-training/rally_s7_holdout_baseline.log"

Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4

New-Item -ItemType Directory -Force "E:\mage-training\backups\rally_latest_armF_20260630" | Out-Null
if (-not (Test-Path "E:\mage-training\backups\rally_latest_armF_20260630\model.pt")) {
  Copy-Item "$md\model.pt" "E:\mage-training\backups\rally_latest_armF_20260630\model.pt" -Force
}

py -3.12 -c "
import json
r=json.load(open('local-training/_brew_win_registry.json'))
for e in r:
    if e.get('profile')=='Pauper-Rally-Anchor-Value': e['deck_path']='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt'
json.dump(r,open('local-training/_rally_gauntlet_registry.json','w'),indent=2)
print('registry ok')
" | Out-File $out

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_FEATURES_ENABLE','RL_INITIATIVE_DIAG'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"

$arms = @(
  @{ label="latest"; src="E:\mage-training\backups\rally_latest_armF_20260630\model.pt" }
)
foreach($a in $arms){
  if (-not (Test-Path $a.src)) { "SKIP $($a.label): missing $($a.src)" | Out-File $out -Append; continue }
  if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
  Copy-Item $a.src "$md\model.pt" -Force; Copy-Item $a.src "$md\model_latest.pt" -Force
  "=== RALLY BASELINE $($a.label) md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_rally_gauntlet_registry.json --profiles Pauper-Rally-Anchor-Value `
    --skill 7 --games-per-matchup 256 --games-per-job 16 `
    --deterministic-eval --skip-compile --replay-seed-base 7777 --run-id "rally_s7_holdout_$($a.label)" 2>&1 |
    Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  "=== $($a.label) DONE $(Get-Date) ===" | Out-File $out -Append
}
"=== ALL RALLY BASELINES DONE $(Get-Date) ===" | Out-File $out -Append
