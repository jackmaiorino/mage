# PAIRED EXPERIMENT Arm B = CONST + win-replay SIL (the method). Codex #32 design.
# IDENTICAL to Arm A (same ref C restore, diet, LR, constant-0.10 entropy, 6k eps)
# EXCEPT the win-replay self-imitation: persistent buffer of own winning trajectories,
# terminal-win gated (silEligible -> Python SIL loss), re-enqueued across batches,
# KL-anchored to the base. Tests whether reinforcing rare own-wins breaks the
# fast-aggro floor. Thesis-clean (imitates the agent's OWN real-rules wins).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log  = "local-training/affinity_armB_winreplay_sil.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"   # fixed ref C (47.5%) - SAME as Arm A

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3

if (-not (Test-Path $ref)) { "FATAL: ref checkpoint missing: $ref" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
Copy-Item $ref "$md/model.pt" -Force
Copy-Item $ref "$md/model_latest.pt" -Force
$refHash = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== Arm B restore: ref=$ref md5=$refHash $(Get-Date) ===" | Out-File $log

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','RL_ONLINE_PREFIX_AUTOPILOT_ENABLE','RL_ONLINE_PREFIX_COMBO_READY_GATE','SIL_GRAVEYARD_GATED'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.affinity_balanced_diet.txt"
$env:OPPONENT_SAMPLER="skillmix"
$env:SKILL_MIX="1:1.0"
$env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"
$env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"   # CONSTANT (now default)
$env:NUM_GAME_RUNNERS="48"
$env:TOTAL_EPISODES="6000"
$env:GAME_LOG_FREQUENCY="2000"
$env:RL_HEURISTIC_STEP_REWARDS="0"
# --- WIN-REPLAY SIL (the only difference vs Arm A) ---
$env:WIN_REPLAY_ENABLE="1"
$env:WIN_REPLAY_BUFFER_MAX="200"
$env:WIN_REPLAY_PER_EPISODE="1"
$env:WIN_REPLAY_MAX_ROWS="150"     # store ~full won trajectories (keep early tempo decisions)
$env:WIN_REPLAY_DEDUP="0"          # dedup collapsed stereotyped Affinity wins to ~5 lines; FIFO 200-buffer gives more diverse reinforcement
$env:SIL_LOSS_COEF="0.5"
$env:SIL_WINDOW_GATED="1"          # Python masks SIL to silEligible (win) rows
$env:SIL_ADVANTAGE_CLIP="2.0"
$env:SIL_WEIGHT_FLOOR="0.2"        # min SIL weight on won rows so early racing/mulligan plays are imitated (not just the kill)
# KL anchor to the base (prevents over-fit/drift, the v1 SIL failure mode)
$env:REFERENCE_POLICY_KL_COEF="0.03"
$env:MCTS_REFERENCE_MODEL_PATH="$ref"
$env:SIL_DIAG_FILE="C:/Users/Jack/IdeaProjects/mage/local-training/_sil_armB_diag.txt"
Remove-Item $env:SIL_DIAG_FILE -ErrorAction SilentlyContinue

"=== AFFINITY Arm B CONST+WIN-REPLAY-SIL $(Get-Date) ===" | Out-File $log -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== Arm B DONE $(Get-Date) ===" | Out-File $log -Append
Copy-Item "$md/model.pt" "E:/mage-training/backups/affinity_armB_winreplay_sil/model.pt" -Force -ErrorAction SilentlyContinue
