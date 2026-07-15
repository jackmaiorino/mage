# FANOUT PROBE: measure whether terminal-rollout fanout finds DISCRIMINATING decisions
# on Affinity-vs-Rally (the credit-assignment signal that was ~0.08% on Spy).
# Pure measurement: warm-start ref C 47.5%, FREEZE policy (LR~0, no SEARCH_OP loss),
# fire SEARCH_OP at each ACTIVATE_ABILITY_OR_SPELL decision with 12 playouts x top-3,
# log per-candidate win rates. Then analyze with search_op_analyze.py.
# Param: $env:PROBE_EPISODES (default 8 = smoke; set 120 for the real probe).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/fanout_probe.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"
$episodes = if ($env:PROBE_EPISODES) { $env:PROBE_EPISODES } else { "8" }

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3
if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
Copy-Item $ref "$md/model.pt" -Force
Copy-Item $ref "$md/model_latest.pt" -Force
# wipe old game logs so the analyzer only sees this probe
$gameLogDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/logs/games/training"
if (Test-Path $gameLogDir) { Remove-Item "$gameLogDir/*.txt" -Force -ErrorAction SilentlyContinue }
$h = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== FANOUT PROBE ref md5=$h episodes=$episodes $(Get-Date) ===" | Out-File $log

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','SIL_GRAVEYARD_GATED','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/decklist.grixis_affinity_only_20260513.txt"
$env:DECK_LIST_FILE="$deckDir/decklist.mono_red_rally_only_20260513.txt"
$env:OPPONENT_SAMPLER="fixed"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
# FREEZE policy: tiny LR + no SEARCH_OP loss = measure ref C's decision landscape
$env:ACTOR_LR="1e-8"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:BRANCH_RETURN_POLICY_LOSS_COEF="0"
$env:NUM_GAME_RUNNERS="6"; $env:TOTAL_EPISODES="$episodes"
$env:RL_HEURISTIC_STEP_REWARDS="0"
# don't let the health monitor kill the (now slow) real games before they produce data
$env:GAME_TIMEOUT_SEC="3600"; $env:HEALTH_REPEAT_THRESHOLD="1000000"
# SEARCH_OP fanout measurement -- NO-TIMEOUT mode: let long (winning) rollouts COMPLETE
$env:SEARCH_OP_ENABLE="true"; $env:SEARCH_OP_LOG="true"; $env:SEARCH_OP_APPLY_OVERRIDE="false"
$env:SEARCH_OP_PLAYOUTS="6"; $env:SEARCH_OP_TOP_K="3"; $env:SEARCH_OP_MAX_ACTIVATIONS="4"
$env:SEARCH_OP_SKIP_TOP_PROB="0.85"; $env:SEARCH_OP_SAMPLE_PROB="1.0"; $env:SEARCH_OP_ARBITER_CAST_FILTER=""
$env:SEARCH_OP_PLAYOUT_TIMEOUT_MS="90000"; $env:SEARCH_OP_TOTAL_TIMEOUT_MS="600000"; $env:SEARCH_OP_MAX_GAME_TURNS="0"
$env:SEARCH_OP_MODEL_GUIDED="true"   # realistic rollouts (greedy durdles -> never terminates)
$env:SEARCH_OP_MIN_TURN="3"          # fire turn 3+ : test whether long (winning) rollouts complete + show mixed wr
$env:GAME_LOG_FREQUENCY="1"

"=== FANOUT PROBE RUN (Affinity vs Rally, SEARCH_OP fanout) $(Get-Date) ===" | Out-File $log -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== FANOUT PROBE DONE $(Get-Date) ===" | Out-File $log -Append
