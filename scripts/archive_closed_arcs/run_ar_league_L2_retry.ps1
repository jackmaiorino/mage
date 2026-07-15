# L2 TRANSPLANT RETRY (Codex #49): the first 256 transplant run DEGRADED (scale collapse:
# .self_attn.scale -> 0, vs-CP 35.6%->13%). ONE targeted stability repair: FREEZE the attn
# scale params (+floor) so they can't collapse, and drop all LRs hard. From the validated
# function-preserving transplant. ABORT EARLY if vs-CP drops >8pp from the ~35.6% thermometer.
# Param: $env:LEAGUE_TARGET (episodes/profile; default 6000).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$asrc = "E:/mage-training/backups/affinity_256_transplant/model.pt"
$rsrc = "E:/mage-training/backups/rally_256_transplant/model.pt"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_L2_registry.json"
$target = if ($env:LEAGUE_TARGET) { $env:LEAGUE_TARGET } else { "6000" }
$log = "local-training/ar_league_L2_retry.log"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 4
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3

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
"=== L2 RETRY 256/2 (FREEZE_ATTN_SCALE, low LR) target=$target $(Get-Date) ===" | Out-File $log

$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"; $env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
$env:MODEL_D_MODEL = "256"; $env:MODEL_NHEAD = "8"; $env:MODEL_NUM_LAYERS = "2"; $env:MODEL_DIM_FEEDFORWARD = "512"
$env:ACTOR_LR = "1e-5"; $env:CRITIC_LR = "5e-5"; $env:OTHER_LR = "1e-5"     # Codex #49: hard LR drop
$env:FREEZE_ATTN_SCALE = "1"; $env:SCALED_MHA_MIN_SCALE = "1.0"              # anti scale-collapse
$env:BELIEF_CONDITION_POLICY = "0"
$env:RL_FAIL_ON_SKIPPED_INCOMPATIBLE = "1"; $env:RL_MODEL_LOAD_DETERMINISM_GATE = "1"
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"; $env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"; $env:MULLIGAN_DECISION_LOG = "0"
$env:TOTAL_EPISODES = "$target"

"=== L2 RETRY run (256/2, FREEZE_ATTN_SCALE=1, ACTOR/OTHER 1e-5, CRITIC 5e-5) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== L2 RETRY DONE $(Get-Date) ===" | Out-File $log -Append
$latest = Get-ChildItem "$amd/snapshots/snapshot_step_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {
  New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_L2_retry_$target" | Out-Null
  Copy-Item $latest.FullName "E:/mage-training/backups/affinity_L2_retry_$target/model.pt" -Force
  "L2 retry snapshot backed up: $($latest.Name) $(Get-Date)" | Out-File $log -Append
}
