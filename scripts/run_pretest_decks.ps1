# CHEAP PRE-TEST (Codex gate #4): eval the SAME fixed policy (fresh-128 control,
# trained on the OLD list, no zone-count) on 3 agent decks -- current / consistency-only
# / full winning -- at matched seeds vs Grixis CP7 skill-1. Directional only (policy is
# untrained on the new cards), but validates deck files + shows whether the better mana
# base raises LEGAL SPY REACH (Spy castable) without retraining. Metric split (Codex):
# legal-reach (candidate_offer_oracle 'offered') vs actual cast vs winrate.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/pretest_decks_RESULT.log"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$fixed = "E:\mage-training\backups\fresh128_control\model_best.pt"
$deckBase = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== DECK PRE-TEST started $(Get-Date) (fixed policy = fresh-128 control, zone-count OFF) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $fixed)) { "FATAL: fixed policy missing $fixed" | Out-File $out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $fixed "$md\model_latest.pt" -Force; Copy-Item $fixed "$md\model.pt" -Force

# make sure zone-count features are OFF (fresh-128 control was trained without them)
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','RL_ZONE_COUNT_DIAG','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_WORLD_MODEL_LABELS_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"; $env:OPPONENT_SAMPLER="grixis"

$arms = @(
  @{ label="cur";  deck="$deckBase/Deck - Spy Combo.dek" },
  @{ label="cons"; deck="$deckBase/Deck - Spy Consistency.dek" },
  @{ label="win";  deck="$deckBase/Deck - Spy Winning.dek" }
)
foreach ($a in $arms) {
  $tmpReg = "local-training/_pretest_reg_$($a.label).json"
  py -3.12 -c "
import json,sys
reg=json.load(open(r'$reg'))
rows = reg if isinstance(reg,list) else reg.get('profiles',reg)
for e in (rows if isinstance(rows,list) else []):
    if e.get('profile')=='Pauper-Spy-Combo-Value':
        e.setdefault('train_env',{})['RL_AGENT_DECK_LIST']=r'$($a.deck)'
json.dump(reg,open(r'$tmpReg','w'),indent=2)
print('wrote $tmpReg agent_deck=$($a.deck)')
" 2>&1 | Out-File $out -Append
  "=== arm $($a.label) eval start $(Get-Date) deck=$($a.deck) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $tmpReg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 5151 --run-id "pre_$($a.label)" 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}

"=== ANALYSIS ===" | Out-File $out -Append
foreach ($a in $arms) {
  "--- arm $($a.label) ---" | Out-File $out -Append
  (py -3.12 scripts/mtgrl/candidate_offer_oracle.py "$GL/pre_$($a.label)" 2>&1 | Out-String) | Out-File $out -Append
  (py -3.12 scripts/mtgrl/loss_audit.py "$GL/pre_$($a.label)" 2>&1 | Out-String) | Out-File $out -Append
}
"=== DECK PRE-TEST DONE $(Get-Date) ===" | Out-File $out -Append
