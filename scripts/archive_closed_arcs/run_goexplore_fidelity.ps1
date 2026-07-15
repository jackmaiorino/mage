# GO-EXPLORE PHASE 0.0: replay-fidelity test (Codex gate >=98%).
# Does deterministic action-replay (same seed + autopilot-forced prefix) reproduce an
# exploration-generated trajectory? Capture trace under seed S, replay its controllable prefix
# under the SAME seed, compare cell-equality per controllable decision. Standalone RLTrainer
# (py4j local inference auto-starts).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/goexplore_fidelity_RESULT.log"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\league_step1_20260620\model_best.pt"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'py4j_entry_point|gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== GOEXPLORE FIDELITY $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Spy-Combo-Value"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
# decks
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Spy Winning.dek"
$env:EVAL_OPPONENT_DECK="$deckDir/Deck - Grixis Affinity.dek"
$env:EVAL_OPPONENT_SKILL="1"
# determinism + moderate exploration (so capture trace is stochastic -> meaningfully tests replay)
$env:RL_BASE_SEED="12345"
$env:RL_ACTION_EPS_START="0.30"; $env:RL_ACTION_EPS_END="0.30"
$env:RL_FULL_TURN_RANDOM_START="0.10"; $env:RL_FULL_TURN_RANDOM_END="0.10"
# go-explore hooks
$env:RL_GOEXPLORE_TRACE="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_ENABLE="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_DURING_TRAINING="1"
$env:GOEXPLORE_PHASE="fidelity"
$env:GOEXPLORE_TRIALS="12"
$env:GOEXPLORE_PREFIX_DEPTH="10"
$env:GOEXPLORE_EPISODE_TAG="0"
$env:GOEXPLORE_SEED_BASE="7777"

"--- launching RLTrainer goexplore $(Get-Date) ---" | Out-File $out -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=goexplore" 2>&1 |
  Select-String "GOEXPLORE" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== GOEXPLORE FIDELITY DONE $(Get-Date) ===" | Out-File $out -Append
Kill-Train
