# H1: hard-matchup-weighted terminal-only continuation from ref C (Codex #36, lever #2).
# Same FIXED Affinity list + real gauntlet; oversample hard matchups (Rally 3x / sub-45 2x /
# easy 1x). Terminal-only, constant entropy 0.10, LR 5e-5. Eval is on the UNIFORM mixture.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/affinity_h1.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
Copy-Item $ref "$md/model.pt" -Force
Copy-Item $ref "$md/model_latest.pt" -Force
$h = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== H1 PROVENANCE restore=ref-C md5=$h diet=h1_hardweighted(Rally3x/sub45 2x/easy1x) LR=5e-5 entropy=const0.10 terminal-only $(Get-Date) ===" | Out-File $log

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','SIL_GRAVEYARD_GATED','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','SEARCH_OP_ENABLE','MULLIGAN_M0_FORCE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.affinity_h1_hardweighted.txt"
$env:OPPONENT_SAMPLER="skillmix"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="6000"; $env:GAME_LOG_FREQUENCY="3000"
$env:RL_HEURISTIC_STEP_REWARDS="0"

"=== H1 TRAIN (hard-weighted, CP7 skill1) $(Get-Date) ===" | Out-File $log -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== H1 DONE $(Get-Date) ===" | Out-File $log -Append
Copy-Item "$md/model.pt" "E:/mage-training/backups/affinity_h1/model.pt" -Force -ErrorAction SilentlyContinue
