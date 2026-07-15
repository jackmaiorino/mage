# A/R co-evolving league (Codex #40, FINAL bounded attempt). Affinity warm-start ref C +
# FRESH co-evolving Rally, OPPONENT_SAMPLER=league 20/40/40 (20% CP-skill1 anchor / 40% mirror
# / 40% cross=Rally), CP anchor includes CP-Rally (eval-relevant hard target). PBT weight-copy
# DISABLED (different decks). Eval on the uniform fixed CP gauntlet. Param: $env:LEAGUE_SMOKE
# (episodes; 0 = full run 60000). Provenance + opponent-composition logged (LEAGUE_DEBUG).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$ref  = "E:/mage-training/backups/affinity_const_entropy_20260626/model.pt"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_AR_league_registry.json"
$smoke = if ($env:LEAGUE_SMOKE) { $env:LEAGUE_SMOKE } else { "0" }
$log = "local-training/ar_league.log"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|py4j_entry_point|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 3

# --- setup: Affinity warm-start ref C (keep counter=0), Rally FRESH (model deleted -> random) ---
$amd = "$prof/Pauper-Affinity-Anchor-Value/models"; $rmd = "$prof/Pauper-Rally-Anchor-Value/models"
Copy-Item $ref "$amd/model.pt" -Force; Copy-Item $ref "$amd/model_latest.pt" -Force
Remove-Item "$rmd/model.pt","$rmd/model_latest.pt" -Force -ErrorAction SilentlyContinue
foreach($m in @($amd,$rmd)){ if (Test-Path "$m/onnx") { Remove-Item "$m/onnx" -Recurse -Force }; if (Test-Path "$m/snapshots") { Remove-Item "$m/snapshots/*" -Recurse -Force -ErrorAction SilentlyContinue } }
# reset BOTH episode counters (so the orchestrator targets both equally) + league status
foreach($p in @("Pauper-Affinity-Anchor-Value","Pauper-Rally-Anchor-Value")){
  Remove-Item "$prof/$p/logs/stats/training_stats.csv" -Force -ErrorAction SilentlyContinue
  Remove-Item "$prof/$p/logs/league/agent_status.json" -Force -ErrorAction SilentlyContinue
}
$h = (Get-FileHash "$amd/model.pt" -Algorithm MD5).Hash
"=== A/R LEAGUE PROVENANCE Affinity=refC(md5 $h, counter-reset) Rally=FRESH league=20/40/40 CP-anchor=skill1 PBT-exploit=DISABLED smoke=$smoke $(Get-Date) ===" | Out-File $log

# --- env ---
$env:REGISTRY_PATH = $reg
$env:TRAIN_PROFILES = "2"; $env:NUM_GAME_RUNNERS = "32"
# shared_gpu service mode: route ALL inference through the GPU service score workers (profiles
# register for scoring; no ONNX needed). hybrid mode deadlocks on a FRESH cold-start (deleted
# ONNX -> local inference empty -> shared-GPU score path has 0 registered profiles -> every
# request times out). infer=cpu (score worker) / train=cuda:0 keeps them off the same device.
$env:PY_SERVICE_MODE = "shared_gpu"
$env:GPU_SERVICE_SPLIT_ROLES = "1"; $env:INFER_CUDA_DEVICE = "cpu"; $env:TRAIN_CUDA_DEVICE = "cuda:0"
$env:SCORE_WORKER_THREADS = "0"; $env:ONNX_EXPORT_ENABLE = "0"
$env:PBT_EXPLOIT_INTERVAL = "999999"            # DISABLE PBT weight-copying (Affinity != Rally deck)
$env:PBT_MIN_EPISODES = "99999999"
$env:OPPONENT_SAMPLER = "league"; $env:LEAGUE_MODE = ""
$env:LEAGUE_POST_HEURISTIC_P = "0.20"; $env:LEAGUE_POST_LOCAL_P = "0.40"; $env:LEAGUE_POST_CROSS_P = "0.40"
$env:LEAGUE_POST_HEURISTIC_SKILL = "1"; $env:LEAGUE_BASELINE_BOT_SKILL = "1"
# co-evolution tweaks: weak Rally qualifies as a cross-opponent ASAP (foothold), snapshot+tick often
$env:LEAGUE_PROMOTE_WR = "0.0"; $env:LEAGUE_POOL_FLOOR_WR = "0.0"; $env:LEAGUE_CHAMPION_PROMOTE_WR = "0.0"
$env:SNAPSHOT_SAVE_EVERY_STEPS = "300"
$env:LEAGUE_TICK_EPISODES = "2000"; $env:LEAGUE_DEBUG = "true"
$env:ONNX_EXPORT_FP16 = "0"; $env:USE_TRT_INFERENCE = "0"; $env:SEARCH_OP_ENABLE = "0"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
if ($smoke -ne "0") { $env:TOTAL_EPISODES = "$smoke" } else { $env:TOTAL_EPISODES = "60000" }

"=== A/R LEAGUE run (TOTAL_EPISODES=$($env:TOTAL_EPISODES)) $(Get-Date) ===" | Out-File $log -Append
py -3.12 scripts/run_local_pbt.py 2>&1 | Tee-Object -FilePath $log -Append
"=== A/R LEAGUE DONE $(Get-Date) ===" | Out-File $log -Append
# back up the trained Affinity model for gauntlet eval
New-Item -ItemType Directory -Force "E:/mage-training/backups/affinity_ar_league" | Out-Null
Copy-Item "$amd/model.pt" "E:/mage-training/backups/affinity_ar_league/model.pt" -Force -ErrorAction SilentlyContinue
"Affinity model backed up -> affinity_ar_league $(Get-Date)" | Out-File $log -Append
