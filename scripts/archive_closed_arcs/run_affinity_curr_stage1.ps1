# CURRICULUM Stage 1 (bootstrap): Affinity vs SLOW-Rally-heavy diet from ref C 47.5%.
# Goal: generate real Affinity-vs-aggro WINS (slow opponent is beatable) so the agent
# learns racing/blocking/removal sequencing -- the skill self-imitation couldn't bootstrap.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/affinity_curr_stage1.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"   # ref C 47.5%

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
Copy-Item $ref "$md/model.pt" -Force
Copy-Item $ref "$md/model_latest.pt" -Force
$h = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== Stage1 restore ref=$ref md5=$h $(Get-Date) ===" | Out-File $log

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','SIL_GRAVEYARD_GATED','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.affinity_curriculum_stage1.txt"
$env:OPPONENT_SAMPLER="skillmix"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="6000"; $env:GAME_LOG_FREQUENCY="2000"
$env:RL_HEURISTIC_STEP_REWARDS="0"

"=== AFFINITY CURRICULUM STAGE1 (slow-Rally bootstrap) $(Get-Date) ===" | Out-File $log -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== Stage1 DONE $(Get-Date) ===" | Out-File $log -Append
Copy-Item "$md/model.pt" "E:/mage-training/backups/affinity_curr_stage1/model.pt" -Force -ErrorAction SilentlyContinue
