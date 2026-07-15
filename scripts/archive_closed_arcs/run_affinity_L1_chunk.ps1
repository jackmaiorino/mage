# L1 long-clean-run chunk (Codex #38). 10k eps of UNIFORM real-gauntlet terminal-only
# training. Chunk 1 restores ref C; later chunks continue from the profile (warm-start).
# Param: $env:L1_CHUNK (1..N). Backs up to affinity_L1_ck<N>. Provenance logged.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$ref = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"
$ck  = [int]$env:L1_CHUNK
$log = "local-training/affinity_L1_ck$ck.log"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
if ($ck -le 1) {
  if (-not (Test-Path $ref)) { "FATAL: ref missing" | Out-File $log; exit 1 }
  Copy-Item $ref "$md/model.pt" -Force; Copy-Item $ref "$md/model_latest.pt" -Force
  $src = "ref-C(restored)"
} else { $src = "continue-from-profile(ck$($ck-1))" }
$h = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== L1 ck$ck PROVENANCE src=$src md5=$h diet=UNIFORM-8deck LR=5e-5 entropy=const0.10 terminal-only 10k-eps $(Get-Date) ===" | Out-File $log

foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','RL_GOEXPLORE_TRACE','RL_ONLINE_PREFIX_SEARCH_ENABLE','SIL_GRAVEYARD_GATED','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','SIL_WINDOW_GATED','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','SEARCH_OP_ENABLE','MULLIGAN_M0_FORCE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.all_opponents.txt"
$env:OPPONENT_SAMPLER="skillmix"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="10000"; $env:GAME_LOG_FREQUENCY="5000"
$env:RL_HEURISTIC_STEP_REWARDS="0"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== L1 ck$ck DONE $(Get-Date) ===" | Out-File $log -Append
$dest = "E:/mage-training/backups/affinity_L1_ck$ck"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item "$md/model.pt" "$dest/model.pt" -Force -ErrorAction SilentlyContinue
