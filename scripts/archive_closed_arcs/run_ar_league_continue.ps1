# A/R league CONTINUATION (Codex #41: CONTINUE from the 7k snapshot, don't restart).
# Restores the valid 7k co-evolving checkpoints (Affinity=snapshot_257700 md5 362C17FF,
# Rally=model_latest md5 97E76861) -- both full state incl optimizer + train_step_counter so
# schedules/Adam continue cleanly. Keeps CSV episode counters + snapshot opponent pool.
# Infra hardening per Codex: ONNX_EXPORT_ENABLE=1 (aux/eval JVMs get ONNX -> no GPU-service
# poisoning), graceful kill (java first, let python finish saves), atomic model.pt save (code).
# Param: $env:LEAGUE_TARGET (NEW episodes/profile this segment; default 8000 = +8k on the 7k
# base = ~15k total trained = Codex first gate). CSVs are reset so this is an unambiguous delta;
# the model's train_step_counter (entropy schedule) continues via the restored snapshot extra_state.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$asrc = "E:/mage-training/backups/affinity_ar_league_7k_snap/model.pt"        # Affinity 7k (362C17FF)
$rsrc = "E:/mage-training/backups/affinity_ar_league_7k_snap/rally_model.pt"  # Rally 7k    (97E76861)
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_registry.json"
$target = if ($env:LEAGUE_TARGET) { $env:LEAGUE_TARGET } else { "8000" }
$log = "local-training/ar_league_continue.log"

# Graceful kill: stop JVMs (runners/trainer) FIRST, wait, then python (lets any in-flight save finish).
function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 4
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3

# --- restore valid 7k checkpoints into BOTH profiles (overwrite corrupt model.pt) ---
$amd = "$prof/Pauper-Affinity-Anchor-Value/models"; $rmd = "$prof/Pauper-Rally-Anchor-Value/models"
Copy-Item $asrc "$amd/model.pt" -Force; Copy-Item $asrc "$amd/model_latest.pt" -Force
Copy-Item $rsrc "$rmd/model.pt" -Force; Copy-Item $rsrc "$rmd/model_latest.pt" -Force
# wipe stale onnx so it re-exports fresh from the restored weights (ONNX export is ON this run)
foreach($m in @($amd,$rmd)){ if (Test-Path "$m/onnx") { Remove-Item "$m/onnx" -Recurse -Force } }
# RESET training_stats.csv (clean delta counting from 0; snapshot has no episode counter anyway).
# KEEP: snapshots/ (opponent pool), agent_status.json (league qualification state).
foreach($p in @("Pauper-Affinity-Anchor-Value","Pauper-Rally-Anchor-Value")){
  Remove-Item "$prof/$p/logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
}
$ah = (Get-FileHash "$amd/model.pt" -Algorithm MD5).Hash
$rh = (Get-FileHash "$rmd/model.pt" -Algorithm MD5).Hash
"=== A/R LEAGUE CONTINUE Affinity=$ah Rally=$rh target=$target $(Get-Date) ===" | Out-File $log

# --- env (same proven league config + infra hardening) ---
$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"          # HARDENING: aux/eval JVMs get ONNX -> no GPU-service poisoning
$env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"   # PBT weight-copy OFF (diff decks)
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"; $env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:TOTAL_EPISODES = "$target"

"=== A/R LEAGUE CONTINUE run (TOTAL_EPISODES=$target) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== A/R LEAGUE CONTINUE DONE $(Get-Date) ===" | Out-File $log -Append
# back up the latest valid Affinity snapshot for gate eval (immutable, md5 logged by eval script)
$latest = Get-ChildItem "$amd/snapshots/snapshot_step_*.pt" | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {
  New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_ar_league_$target" | Out-Null
  Copy-Item $latest.FullName "E:/mage-training/backups/affinity_ar_league_$target/model.pt" -Force
  "gate snapshot backed up: $($latest.Name) -> affinity_ar_league_$target $(Get-Date)" | Out-File $log -Append
}
