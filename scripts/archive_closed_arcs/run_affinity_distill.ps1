# Opponent-conditional distillation (Codex #42): capture the 15k Rally-specialist's Rally
# skill WITHOUT its breadth loss. Init from the 7k generalist; during RL training on the
# 8-deck gauntlet, distill toward the 15k SPECIALIST only on Rally rows (archetype 1) and
# anchor to the 7k GENERALIST on all other rows (preserve breadth). Pure-Python mechanism
# (py4j_entry_point conditional reference-KL; archetype int already plumbed for belief head).
# Target: Rally>=35% AND uniform>=50% simultaneously. Params: $env:DISTILL_TOTAL,
# $env:SPEC_COEF, $env:ANCHOR_COEF.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$log     = "local-training/affinity_distill.log"
$deckDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
$md      = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Affinity-Anchor-Value/models"
$gen7k   = "E:/mage-training/backups/affinity_ar_league_7k_snap/model.pt"   # 362c17ff generalist (init + anchor)
$spec15k = "E:/mage-training/backups/affinity_ar_league_15k/model.pt"       # 923AF687 Rally-specialist (teacher)
$total   = if ($env:DISTILL_TOTAL) { $env:DISTILL_TOTAL } else { "6000" }
$speccoef= if ($env:SPEC_COEF)   { $env:SPEC_COEF }   else { "2.0" }
$anccoef = if ($env:ANCHOR_COEF) { $env:ANCHOR_COEF } else { "0.5" }

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 3; Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3

if (-not (Test-Path $gen7k))   { "FATAL: generalist missing: $gen7k" | Out-File $log; exit 1 }
if (-not (Test-Path $spec15k)) { "FATAL: specialist missing: $spec15k" | Out-File $log; exit 1 }
if (Test-Path "$md/onnx") { Remove-Item "$md/onnx" -Recurse -Force }
# reset stale episode CSV (left at ~8k by the league continuation) so TOTAL_EPISODES is a clean count
Remove-Item "$md/../logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
Copy-Item $gen7k "$md/model.pt" -Force            # INIT from 7k generalist
Copy-Item $gen7k "$md/model_latest.pt" -Force
$ih = (Get-FileHash "$md/model.pt" -Algorithm MD5).Hash
"=== DISTILL init=7k($ih) teacher=15k anchor=7k spec_coef=$speccoef anchor_coef=$anccoef total=$total $(Get-Date) ===" | Out-File $log

# clear stale levers from prior arms
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM','SIL_LOSS_COEF','SIL_WINDOW_GATED','WIN_REPLAY_ENABLE','RL_ONLINE_PREFIX_SEARCH_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MODEL_PROFILE="Pauper-Affinity-Anchor-Value"
$env:RL_AGENT_DECK_LIST="$deckDir/Deck - Grixis Affinity.dek"
$env:DECK_LIST_FILE="$deckDir/decklist.all_opponents.txt"   # 8-deck gauntlet diet (Rally + 7 others)
$env:OPPONENT_SAMPLER="skillmix"; $env:SKILL_MIX="1:1.0"; $env:ADAPTIVE_CURRICULUM="0"
$env:PY_SERVICE_MODE="local"; $env:USE_TRT_INFERENCE="0"; $env:SEARCH_OP_ENABLE="0"
$env:PY_BRIDGE_VERBOSE="1"   # forward Python stdout ([REF_POLICY] model-load confirmation) to the log
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:ACTOR_LR="5e-5"; $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.10"
$env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="$total"; $env:GAME_LOG_FREQUENCY="3000"
$env:RL_HEURISTIC_STEP_REWARDS="0"
$env:MULLIGAN_DECISION_LOG="0"   # disable per-mulligan file logging (hung the prior run via FileWriter lock contention)
$env:GAME_LOG_FREQUENCY="999999" # no per-game logs either (avoid file-IO stalls)
# --- conditional distillation config ---
$env:MCTS_REFERENCE_MODEL_PATH="$spec15k"          # SPECIALIST teacher (Rally rows)
$env:REFERENCE_POLICY_KL_COEF="$speccoef"
$env:REFERENCE_ANCHOR_MODEL_PATH="$gen7k"          # GENERALIST anchor (all other rows)
$env:REFERENCE_ANCHOR_KL_COEF="$anccoef"
$env:REFERENCE_AGGRO_ARCHETYPES="1"                # Rally only (specialist hurt Burn -> keep Burn on anchor)

"=== AFFINITY CONDITIONAL DISTILL run $(Get-Date) ===" | Out-File $log -Append
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train" 2>&1 | Tee-Object -FilePath $log -Append
"=== DISTILL DONE $(Get-Date) ===" | Out-File $log -Append
New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_distill" | Out-Null
Copy-Item "$md/model.pt" "E:/mage-training/backups/affinity_distill/model.pt" -Force -ErrorAction SilentlyContinue
"distill model backed up -> affinity_distill $(Get-Date)" | Out-File $log -Append
