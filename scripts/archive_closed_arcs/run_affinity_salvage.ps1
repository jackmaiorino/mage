# AFFINITY DOMINANCE TRAINING (Codex #26): warm-start fine-tune on the weighted soft-matchup diet,
# CP7 skill-1 gauntlet opponents (matches the benchmark). Chunk 1 = 30k episodes; re-baseline after.
# Goal: 58.2% -> >60% by improving Elves/Terror/Faeries/mirror/Grixis WITHOUT decaying Burn/Caw/Wildfire.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/affinity_salvage.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"          # warm-start from existing model.pt
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.affinity_balanced_diet.txt"   # weighted opponent pool
$env:OPPONENT_SAMPLER="skillmix"                        # CP7, SKILL_MIX default 1:1.0 = skill 1
$env:SKILL_MIX="1:1.0"
$env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"                                    # low fine-tune LR (Codex)
$env:ENTROPY_START="0.05"; $env:ENTROPY_END="0.01"
$env:NUM_GAME_RUNNERS="48"
$env:TOTAL_EPISODES="6000"                             # chunk 1
$env:GAME_LOG_FREQUENCY="2000"
$env:RL_HEURISTIC_STEP_REWARDS="0"                      # terminal-only reward (thesis)

"=== AFFINITY SALVAGE (warm-start, weighted diet, CP7 skill1) $(Get-Date) ===" | Out-File $log
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== AFFINITY SALVAGE DONE $(Get-Date) ===" | Out-File $log -Append
