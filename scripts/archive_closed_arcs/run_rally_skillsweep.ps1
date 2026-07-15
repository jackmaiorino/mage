# Rally DOMINANCE skill-sweep (Codex #26): confirm the 67% (skill1) holds vs stiffer CP at skill 3 & 5.
# Uses the trained/dominant model already in the profile. Gauntlet, greedy.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/rally_skillsweep_RESULT.log"
$reg = "local-training/_rally_gauntlet_registry.json"   # already points agent=Rally, pool=gauntlet
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Rally-Anchor-Value\models"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== RALLY SKILL-SWEEP $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$md\model.pt" "$md\model_latest.pt" -Force
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
foreach ($sk in @(3,5)) {
  "--- skill=$sk $(Get-Date) ---" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Rally-Anchor-Value `
    --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill $sk --games-per-matchup 48 --parallel 8 --games-per-job 12 `
    --skip-compile --replay-seed-base 5151 --run-id "rally_skill$sk" 2>&1 |
    Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}
"=== RALLY SKILL-SWEEP DONE $(Get-Date) ===" | Out-File $out -Append
