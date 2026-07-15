# STEP-2 PART-2 SEARCH SMOKE (Codex #16 gate).
# Q: from GENERIC combo-ready states (0 library lands + >=2 creatures), does TerminalPrefixSearch
#    (depth 12, policy-guided generic ordering, NO card-name heuristics) FIND the finish often enough
#    to justify building the finish-teacher? Gate: conditional search-win (found/calls) >= 40%.
# Also: does autopilot-applying the found finish LIFT winrate/combo-rate vs the no-search baseline?
# Two passes vs Grixis skill 1, same Step-1-best model:
#   p2_baseline = search OFF (control)
#   p2_search   = online-prefix combo-ready-gated + autopilot + generic order
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/p2_search_smoke_RESULT.log"
$reg  = "local-training/_brew_win_registry.json"   # agent = Spy Winning, opp pool has Grixis
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\league_step1_20260620\model_best.pt"
$N    = 128

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== P2 SEARCH SMOKE $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $best)) { "FATAL: best model missing $best" | Out-File $out -Append; exit 1 }
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force

# clean env that isn't part of this 128/2-layer model
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
# clear any leftover online-prefix env from a previous shell
foreach($k in 'RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','RL_ONLINE_PREFIX_COMBO_READY_MIN_CREATURES','RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH','RL_ONLINE_PREFIX_SEARCH_MAX_NODES','RL_ONLINE_PREFIX_SEARCH_TOP_K','RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS','RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS','RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS','RL_ONLINE_PREFIX_SEARCH_LOG'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"

# ---- PASS 1: BASELINE (search OFF) ----
"--- p2_baseline (search OFF) $(Get-Date) ---" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup $N --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "p2_baseline" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

# ---- PASS 2: SEARCH (combo-ready gated online-prefix + autopilot + generic order) ----
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
"--- p2_search (online-prefix combo-ready gated, depth12, generic, autopilot) $(Get-Date) ---" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup $N --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "p2_search" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

# ---- AGGREGATE ----
"--- AGGREGATE $(Get-Date) ---" | Out-File $out -Append
py -3.12 scripts/mtgrl/p2_smoke_aggregate.py local-training/local_pbt/cp7_eval_sweeps 2>&1 | Out-File $out -Append
"=== P2 SEARCH SMOKE DONE $(Get-Date) ===" | Out-File $out -Append
