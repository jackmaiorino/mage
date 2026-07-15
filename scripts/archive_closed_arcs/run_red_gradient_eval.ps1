# Map the winnability gradient: greedy-eval ref C (47.5%) vs real-Rally / slow-Rally /
# drastic-Rally. Tells us if ANY winnable red opponent exists to anchor a curriculum.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/red_gradient_eval_RESULT.log"
$baseReg = "local-training/_brew_win_registry.json"
$reg  = "local-training/_red_gradient_registry.json"
$pool = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.red_gradient.txt"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Affinity-Anchor-Value\models"
$ref  = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== RED GRADIENT EVAL (ref C vs real/slow/drastic Rally) $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $ref "$md\model.pt" -Force
Copy-Item $ref "$md\model_latest.pt" -Force
py -3.12 -c "
import json
r=json.load(open(r'$baseReg'))
for e in r:
    if e.get('profile')=='Pauper-Affinity-Anchor-Value': e['deck_path']=r'$pool'
json.dump(r,open(r'$reg','w'),indent=2)
print('red-gradient registry written')
" 2>&1 | Out-File $out -Append
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Affinity-Anchor-Value `
  --opponents "rally" --skill 1 --games-per-matchup 64 --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "red_gradient_eval" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== RED GRADIENT EVAL DONE $(Get-Date) ===" | Out-File $out -Append
