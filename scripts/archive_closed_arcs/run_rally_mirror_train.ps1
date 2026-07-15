# RALLY MIRROR-FIX ARM (Codex #62/#63): bounded branch from the 75.8% flagship.
# Diet = CP7 LADDER at skill-4 (first deep-search tier, the style the agent loses to)
# over a Rally-OVERWEIGHTED gauntlet pool (~36% mirror, 64% breadth). Single Rally
# profile, terminal-only. Goal: lift mirror 35.5% -> >=45% balanced while uniform stays
# >=70%. Flagship is immutable-backed-up (rally_flagship_75_8); this writes a branch.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$rmd  = "$prof\Pauper-Rally-Anchor-Value\models"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_rally_mirror_registry.json"
$pool = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.rally_mirror_weighted.txt"
$rallyDeck = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek"
$target = if ($env:MIRROR_TARGET) { $env:MIRROR_TARGET } else { "8000" }
$log = "local-training/rally_mirror_train.log"

Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4

# restore the flagship into the Rally profile (branch start)
if (Test-Path "$rmd\onnx") { Remove-Item "$rmd\onnx" -Recurse -Force }
Copy-Item "E:\mage-training\backups\rally_flagship_75_8\model.pt" "$rmd\model.pt" -Force
Copy-Item "E:\mage-training\backups\rally_flagship_75_8\model.pt" "$rmd\model_latest.pt" -Force
Remove-Item "$prof\Pauper-Rally-Anchor-Value\logs\stats\training_stats.csv" -Force -ErrorAction SilentlyContinue
"=== RALLY MIRROR TRAIN start $(Get-Date) flagship=$((Get-FileHash "$rmd\model.pt" -Algorithm MD5).Hash) target=$target ===" | Out-File $log

# clear any stray flags
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_FEATURES_ENABLE','EVAL_OPPONENT_ON_PLAY'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }

$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "1"; $env:NUM_GAME_RUNNERS = "48"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"; $env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"
# --- MIRROR DIET: CP7 ladder @ skill-4 over the Rally-overweighted pool ---
$env:OPPONENT_SAMPLER = "ladder"; $env:LADDER_SKILLS = "4"; $env:LEAGUE_MODE = ""
$env:DECK_LIST_FILE = $pool
$env:RL_AGENT_DECK_LIST = $rallyDeck
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"   # terminal-only
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:TOTAL_EPISODES = "$target"

"=== run (ladder s4, mirror pool, TOTAL_EPISODES=$target) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== RALLY MIRROR TRAIN DONE $(Get-Date) ===" | Out-File $log -Append
# branch checkpoint backup for the gate eval
New-Item -ItemType Directory -Force "E:\mage-training\backups\rally_mirror_branch" | Out-Null
Copy-Item "$rmd\model.pt" "E:\mage-training\backups\rally_mirror_branch\model.pt" -Force
"branch backed up: $((Get-FileHash 'E:\mage-training\backups\rally_mirror_branch\model.pt' -Algorithm MD5).Hash)" | Out-File $log -Append
