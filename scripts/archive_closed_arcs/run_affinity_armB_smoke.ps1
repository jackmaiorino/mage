# SMOKE GATE for Arm B: 300 episodes with the win-replay SIL config, to confirm
# [WIN_REPLAY] actually fires (buffer fills + replays) before committing the full 6k.
# Same config as run_affinity_armB_winreplay_sil.ps1 but 300 eps, separate log, no backup.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/affinity_armB_smoke.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
Copy-Item $ref "$md/model.pt" -Force
Copy-Item $ref "$md/model_latest.pt" -Force

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','SIL_GRAVEYARD_GATED'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.affinity_balanced_diet.txt"
$env:OPPONENT_SAMPLER="skillmix"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="300"; $env:GAME_LOG_FREQUENCY="100000"
$env:RL_HEURISTIC_STEP_REWARDS="0"
$env:WIN_REPLAY_ENABLE="1"; $env:WIN_REPLAY_BUFFER_MAX="200"; $env:WIN_REPLAY_PER_EPISODE="1"
$env:WIN_REPLAY_MAX_ROWS="150"; $env:WIN_REPLAY_DEDUP="1"
$env:SIL_LOSS_COEF="0.5"; $env:SIL_WINDOW_GATED="1"; $env:SIL_ADVANTAGE_CLIP="2.0"; $env:SIL_WEIGHT_FLOOR="0.2"
$env:REFERENCE_POLICY_KL_COEF="0.03"; $env:MCTS_REFERENCE_MODEL_PATH="$ref"
# Python stdout isn't captured in local py4j mode; have the SIL loss write a diag file instead.
$env:SIL_DIAG_FILE="C:/Users/Jack/IdeaProjects/mage/local-training/_sil_smoke_diag.txt"
Remove-Item $env:SIL_DIAG_FILE -ErrorAction SilentlyContinue

"=== ARM B SMOKE (300 eps, win-replay) $(Get-Date) ===" | Out-File $log
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== SMOKE DONE $(Get-Date) ===" | Out-File $log -Append
