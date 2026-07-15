# STEP-2 DIAG: why did the combo-ready online-prefix search never fire (calls=0)?
# Small search-only eval with RL_ONLINE_PREFIX_DIAG=1 -> prints:
#   [OPDIAG-ONCE] enable=... gate=... (confirms env propagation + flag values, once/JVM)
#   [OPDIAG] combo-ready ACTIVATE: cand=.. maxT=.. minT=.. selSize=..  (shows which trigger cond blocks search)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/p2_diag_RESULT.log"
$reg  = "local-training/_brew_win_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\league_step1_20260620\model_best.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== P2 DIAG $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:RL_ONLINE_PREFIX_DIAG="1"
$env:RL_ONLINE_PREFIX_SEARCH_ENABLE="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_GATE="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_MIN_CREATURES="2"
$env:RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_ENABLE="1"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH="12"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_NODES="400"
$env:RL_ONLINE_PREFIX_SEARCH_TOP_K="4"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS="20"
$env:RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS="8000"
$env:RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS="3000"
$env:RL_ONLINE_PREFIX_SEARCH_LOG="1"

Remove-Item -Recurse -Force "local-training/local_pbt/cp7_eval_sweeps/p2_diag" -ErrorAction SilentlyContinue
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 24 --parallel 4 --games-per-job 6 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "p2_diag" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

$lp = "local-training/local_pbt/cp7_eval_sweeps/p2_diag/logs/*.log"
"--- OPDIAG-ONCE (env + flags) ---" | Out-File $out -Append
Select-String -Path $lp -Pattern "OPDIAG-ONCE" | ForEach-Object { $_.Line } | Select-Object -Unique | Out-File $out -Append
"--- OPDIAG-FUNNEL (last per job = cumulative split) ---" | Out-File $out -Append
Select-String -Path $lp -Pattern "OPDIAG-FUNNEL" | ForEach-Object { $_.Line } | Out-File $out -Append
"--- combo-ready hits (libLands==0 & creatures>=2) ---" | Out-File $out -Append
Select-String -Path $lp -Pattern "\[OPDIAG\] combo-ready" | ForEach-Object { $_.Line } | Select-Object -First 40 | Out-File $out -Append
"--- counts ---" | Out-File $out -Append
$cr   = (Select-String -Path $lp -Pattern "\[OPDIAG\] combo-ready").Count
$op   = (Select-String -Path $lp -Pattern "\[ONLINE_PREFIX\]").Count
"combo-ready hits=$cr  [ONLINE_PREFIX] search-fired=$op" | Out-File $out -Append
"=== P2 DIAG DONE $(Get-Date) ===" | Out-File $out -Append
