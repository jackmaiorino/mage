# L2 TRANSPLANT league (Codex #48): warm-started 256d/2L (Net2Net-widened from the 7k, function-
# preserving) league-trained = the capacity test (fresh-256 couldn't bootstrap). Both profiles
# init from their 256 transplants. Same 2-profile Affinity+Rally league, terminal-only, ACTOR_LR=5e-5,
# audit gates ON. Param: $env:LEAGUE_TARGET (NEW episodes/profile; default 8000 = first gate ~ +8k).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$asrc = "E:/mage-training/backups/affinity_256_transplant/model.pt"   # Affinity 256/2 (function-preserving 7k)
$rsrc = "E:/mage-training/backups/rally_256_transplant/model.pt"      # Rally 256/2
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_L2_registry.json"
$target = if ($env:LEAGUE_TARGET) { $env:LEAGUE_TARGET } else { "8000" }
$log = "local-training/ar_league_L2_transplant.log"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 4
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3

# restore 256 transplants into both profiles; clear 128 snapshots (avoid dim mismatch under fail gate)
$amd = "$prof/Pauper-Affinity-Anchor-Value/models"; $rmd = "$prof/Pauper-Rally-Anchor-Value/models"
Copy-Item $asrc "$amd/model.pt" -Force; Copy-Item $asrc "$amd/model_latest.pt" -Force
Copy-Item $rsrc "$rmd/model.pt" -Force; Copy-Item $rsrc "$rmd/model_latest.pt" -Force
foreach($p in @("Pauper-Affinity-Anchor-Value","Pauper-Rally-Anchor-Value")){
  $m = "$prof/$p/models"
  if (Test-Path "$m/onnx") { Remove-Item "$m/onnx" -Recurse -Force }
  if (Test-Path "$m/snapshots") { Remove-Item "$m/snapshots/*" -Recurse -Force -ErrorAction SilentlyContinue }
  Remove-Item "$prof/$p/logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
  Remove-Item "$prof/$p/logs/league/agent_status.json" -Force -ErrorAction SilentlyContinue
}
"=== L2 TRANSPLANT 256/2 league target=$target $(Get-Date) ===" | Out-File $log

$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"; $env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
$env:MODEL_D_MODEL = "256"; $env:MODEL_NHEAD = "8"; $env:MODEL_NUM_LAYERS = "2"; $env:MODEL_DIM_FEEDFORWARD = "512"
$env:ACTOR_LR = "5e-5"; $env:BELIEF_CONDITION_POLICY = "0"
$env:RL_FAIL_ON_SKIPPED_INCOMPATIBLE = "1"; $env:RL_MODEL_LOAD_DETERMINISM_GATE = "1"
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"; $env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"; $env:MULLIGAN_DECISION_LOG = "0"
$env:TOTAL_EPISODES = "$target"

"=== L2 TRANSPLANT run (256/2, ACTOR_LR=5e-5, TOTAL_EPISODES=$target) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== L2 TRANSPLANT DONE $(Get-Date) ===" | Out-File $log -Append
$latest = Get-ChildItem "$amd/snapshots/snapshot_step_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {
  New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_L2_transplant_$target" | Out-Null
  Copy-Item $latest.FullName "E:/mage-training/backups/affinity_L2_transplant_$target/model.pt" -Force
  "L2 gate snapshot backed up: $($latest.Name) -> affinity_L2_transplant_$target $(Get-Date)" | Out-File $log -Append
}
