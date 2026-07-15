# L1 BELIEF-CONDITIONED POLICY league (Codex #46 top pick). Same 2-profile Affinity+Rally
# co-evolving league as the continuation, but with BELIEF_CONDITION_POLICY=1: the policy is
# conditioned on the model's OWN belief-head prediction (zero-init belief_proj -> warm-start =
# exact 7k identity, learns the archetype conditioning from terminal reward). ONE coherent
# archetype-adaptive net (no mid-game switch, no shared-encoder distill). Init from 7k.
# Param: $env:LEAGUE_TARGET (NEW episodes/profile; default 10000).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$asrc = "E:/mage-training/backups/affinity_ar_league_7k_snap/model.pt"        # Affinity 7k (362C17FF)
$rsrc = "E:/mage-training/backups/affinity_ar_league_7k_snap/rally_model.pt"  # Rally 7k    (97E76861)
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_registry.json"
$target = if ($env:LEAGUE_TARGET) { $env:LEAGUE_TARGET } else { "10000" }
$log = "local-training/ar_league_L1.log"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 4
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3

# restore 7k into both profiles (strict=False load -> new belief_proj zero-inits -> identity warm-start)
$amd = "$prof/Pauper-Affinity-Anchor-Value/models"; $rmd = "$prof/Pauper-Rally-Anchor-Value/models"
Copy-Item $asrc "$amd/model.pt" -Force; Copy-Item $asrc "$amd/model_latest.pt" -Force
Copy-Item $rsrc "$rmd/model.pt" -Force; Copy-Item $rsrc "$rmd/model_latest.pt" -Force
foreach($m in @($amd,$rmd)){ if (Test-Path "$m/onnx") { Remove-Item "$m/onnx" -Recurse -Force } }
foreach($p in @("Pauper-Affinity-Anchor-Value","Pauper-Rally-Anchor-Value")){
  Remove-Item "$prof/$p/logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
}
"=== L1 BELIEF-COND league Affinity=$((Get-FileHash "$amd/model.pt" -Algorithm MD5).Hash.Substring(0,8)) target=$target $(Get-Date) ===" | Out-File $log

# --- env (proven league config + L1) ---
$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"
$env:ONNX_EXPORT_ENABLE = "1"; $env:USE_TRT_INFERENCE = "0"; $env:ONNX_EXPORT_FP16 = "0"
$env:PBT_EXPLOIT_INTERVAL = "999999"; $env:PBT_MIN_EPISODES = "99999999"
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"; $env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:SEARCH_OP_ENABLE = "0"; $env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:MULLIGAN_DECISION_LOG = "0"   # avoid the FileWriter-lock deadlock found earlier
$env:BELIEF_CONDITION_POLICY = "1" # <<< L1: belief-conditioned policy
$env:TOTAL_EPISODES = "$target"

"=== L1 league run (TOTAL_EPISODES=$target, BELIEF_CONDITION_POLICY=1) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== L1 league DONE $(Get-Date) ===" | Out-File $log -Append
$latest = Get-ChildItem "$amd/snapshots/snapshot_step_*.pt" | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {
  New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_L1_belief" | Out-Null
  Copy-Item $latest.FullName "E:/mage-training/backups/affinity_L1_belief/model.pt" -Force
  "L1 gate snapshot backed up: $($latest.Name) -> affinity_L1_belief $(Get-Date)" | Out-File $log -Append
}
