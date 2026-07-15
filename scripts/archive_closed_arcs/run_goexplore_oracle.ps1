# GO-EXPLORE PAIRED REPLAYED-ROOT (Codex #19 green-light test).
# Harvest STRICT roots (libLands==0 & creatures>=2 & Balustrade Spy castable) from exploration
# (harvest run = no-search BASELINE). Replay prefix-to-root under same seed+episodeTag (fidelity-proven)
# and fire finish-search ONCE at the root. Compare bigMill/comboWin: baseline vs search, per root.
# GATE: P(bigMill|root)>=30-40%, bigMill 8%->>=20-25%, comboWin materially up.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/goexplore_oracle_RESULT.log"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\league_step1_20260620\model_best.pt"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'py4j_entry_point|gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== GOEXPLORE ORACLE $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Spy-Combo-Value"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Spy Winning.dek"
$env:EVAL_OPPONENT_DECK="$deckDir/Deck - Grixis Affinity.dek"
$env:EVAL_OPPONENT_SKILL="1"
$env:RL_BASE_SEED="12345"
$env:RL_ACTION_EPS_START="0.30"; $env:RL_ACTION_EPS_END="0.30"
$env:RL_FULL_TURN_RANDOM_START="0.10"; $env:RL_FULL_TURN_RANDOM_END="0.10"
$env:RL_GOEXPLORE_TRACE="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_ENABLE="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_DURING_TRAINING="1"
# finish-search config (enable is per-arm via override; these are the search params)
$env:RL_ONLINE_PREFIX_SEARCH_ENABLE="1"
$env:RL_ONLINE_PREFIX_SEARCH_DURING_TRAINING="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_GATE="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_MIN_CREATURES="2"
$env:RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER="1"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH="12"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_NODES="300"
$env:RL_ONLINE_PREFIX_SEARCH_TOP_K="3"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS="3"
$env:RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS="6000"
$env:RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS="2500"
# paired phase params
$env:GOEXPLORE_PHASE="oracle"
$env:GOEXPLORE_TRIALS="128"
$env:GOEXPLORE_SEED_BASE="7777"
$env:GOEXPLORE_GAME_TIMEOUT_SEC="60"

"--- launching RLTrainer goexplore convert_paired $(Get-Date) ---" | Out-File $out -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=goexplore" 2>&1 |
  Select-String "GOEXPLORE" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== GOEXPLORE ORACLE DONE $(Get-Date) ===" | Out-File $out -Append
Kill-Train
