# MIRROR + SKILL-SWEEP: isolate our agent's PILOT proficiency from opponent-deck-skill confound.
# MIRROR (our best full-winning agent vs CP, BOTH on Spy Winning) = pure pilot skill, deck-controlled:
#   if our agent >> CP at the SAME deck -> we out-pilot CP (weak-opponent-ceiling story holds).
# SKILL-SWEEP (vs Grixis at CP skill 1/3/5/7) = does a stronger opponent punish our lazy/beatdown play
#   (winrate drops as skill rises -> opponent quality was capping us).
# Waits for any running eval to free the GPU first.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/mirror_skillsweep_RESULT.log"
$baseReg = "local-training/_brew_win_registry.json"   # RL_AGENT_DECK_LIST = Spy Winning
$reg = "local-training/_mirror_sweep_registry.json"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\brew_winning_20260619\model_best.pt"
$pool = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.mirror_sweep.txt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== MIRROR + SKILL-SWEEP $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $best)) { "FATAL: best model missing $best" | Out-File $out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force
# registry: agent deck = Spy Winning (from base), opponent pool = mirror_sweep
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Spy-Combo-Value': e['deck_path']=r'$pool'
json.dump(r,open(r'$reg','w'),indent=2)
print('mirror-sweep registry written')
" 2>&1 | Out-File $out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"

# opp: 'winning' = MIRROR (CP plays Spy Winning); 'grixis' = real matchup
foreach ($opp in @('winning','grixis')) {
  foreach ($skill in @(1,3,5,7)) {
    "--- opp=$opp skill=$skill $(Get-Date) ---" | Out-File $out -Append
    py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
      --opponents $opp --skill $skill --games-per-matchup 160 --parallel 8 `
      --skip-compile --replay-seed-base 5151 --run-id "ms_${opp}_s${skill}" 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  }
}
"=== MIRROR + SKILL-SWEEP DONE $(Get-Date) ===" | Out-File $out -Append
