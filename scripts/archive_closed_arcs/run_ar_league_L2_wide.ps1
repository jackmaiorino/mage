# L2 WIDER NET (256d/4-layer) league capacity test (Codex #47). Final structural lever:
# can MORE capacity hold Rally-tempo AND breadth where the 128d/2L net can't (3 single-net
# merge attempts all degraded)? FRESH 256/4 both profiles, same 2-profile Affinity+Rally
# league diet, terminal-only, lower LR (avoid the hot-failed 256 instability), audit gates ON.
# Param: $env:LEAGUE_TARGET (episodes/profile; default 2000 = BOOTSTRAP CHECK -- judge by
# in-training winfrac, fresh nets read ~0 greedy until entropy decays). Extend to 15k/30k if it bootstraps.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_registry.json"
$target = if ($env:LEAGUE_TARGET) { $env:LEAGUE_TARGET } else { "2000" }
$log = "local-training/ar_league_L2_wide.log"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 4
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3

# FRESH 256/4: delete 128 weights + onnx + snapshots (avoid 128->256 load mismatch under the fail gate)
foreach($p in @("Pauper-Affinity-Anchor-Value","Pauper-Rally-Anchor-Value")){
  $m = "$prof/$p/models"
  Remove-Item "$m/model.pt","$m/model_latest.pt" -Force -ErrorAction SilentlyContinue
  if (Test-Path "$m/onnx") { Remove-Item "$m/onnx" -Recurse -Force }
  if (Test-Path "$m/snapshots") { Remove-Item "$m/snapshots/*" -Recurse -Force -ErrorAction SilentlyContinue }
  Remove-Item "$prof/$p/logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
  Remove-Item "$prof/$p/logs/league/agent_status.json" -Force -ErrorAction SilentlyContinue
}
"=== L2 WIDE 256/4 FRESH league target=$target $(Get-Date) ===" | Out-File $log

# --- env ---
$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"; $env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
# WIDE ARCH 256/4
$env:MODEL_D_MODEL = "256"; $env:MODEL_NHEAD = "8"; $env:MODEL_NUM_LAYERS = "4"; $env:MODEL_DIM_FEEDFORWARD = "1024"
$env:ACTOR_LR = "1e-4"                 # Codex: lower than the hot-failed 256 (3e-4); watch clip_frac/approx_kl
$env:BELIEF_CONDITION_POLICY = "0"     # plain wider net (L1 conditioning was a separate dead lever)
$env:RL_FAIL_ON_SKIPPED_INCOMPATIBLE = "1"; $env:RL_MODEL_LOAD_DETERMINISM_GATE = "1"  # audit: hard-fail silent dim mismatch
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"; $env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"; $env:MULLIGAN_DECISION_LOG = "0"
$env:TOTAL_EPISODES = "$target"

"=== L2 WIDE run (256/4, ACTOR_LR=1e-4, TOTAL_EPISODES=$target) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== L2 WIDE DONE $(Get-Date) ===" | Out-File $log -Append
$amd = "$prof/Pauper-Affinity-Anchor-Value/models"
$latest = Get-ChildItem "$amd/snapshots/snapshot_step_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {
  New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_L2_wide_$target" | Out-Null
  Copy-Item $latest.FullName "E:/mage-training/backups/affinity_L2_wide_$target/model.pt" -Force
  "L2 snapshot backed up: $($latest.Name) -> affinity_L2_wide_$target $(Get-Date)" | Out-File $log -Append
}
