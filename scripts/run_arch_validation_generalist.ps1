param(
    [int]$TotalEpisodes = 10000,
    [int]$NumGameRunners = 96,
    [int]$WinrateWindow = 200,
    [int]$GpuServicePort = 26100,
    [string]$Registry = "",
    [switch]$StartMonitoring,
    [switch]$StartDashboard
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultRegistry = Join-Path $repo "docs\mtgrl\experiments\2026-04-25-architecture-validation-plan\phase-b-generalist-registry.json"
if ([string]::IsNullOrWhiteSpace($Registry)) {
    $registry = $defaultRegistry
} elseif ([System.IO.Path]::IsPathRooted($Registry)) {
    $registry = (Resolve-Path $Registry).Path
} else {
    $registry = (Resolve-Path (Join-Path $repo $Registry)).Path
}
$profile = "Pauper-Generalist-Value-v2"

if (-not (Test-Path $registry)) {
    throw "Missing experiment registry: $registry"
}

if ($StartMonitoring -or $StartDashboard) {
    & (Join-Path $repo "scripts\start_profile_metrics_exporter.ps1") -Profiles $profile -Window $WinrateWindow -Restart
}

if ($StartDashboard) {
    docker compose -f (Join-Path $repo "docker-compose-observe-local.yml") up -d
}

$env:REGISTRY_PATH = $registry
$env:TRAIN_PROFILES = "1"
$env:NUM_GAME_RUNNERS = "$NumGameRunners"
$env:TOTAL_EPISODES = "$TotalEpisodes"
$env:WINRATE_WINDOW = "$WinrateWindow"
$env:GPU_SERVICE_PORT = "$GpuServicePort"
$env:PROFILE_METRICS_PROFILES = $profile

# Phase B is terminal-reward, random-deck selfplay. Keep CP7 and MCTS out of
# the training loop; CP7 is run by scripts/run_cp7_eval_sweep.py at the gate.
$env:OPPONENT_SAMPLER = "self"
$env:SELFPLAY_OPPONENT_TRAINING = "1"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:MCTS_TRAINING_ENABLE = "0"
$env:ISMCTS_ENABLE = "0"
$env:ISMCTS_ROLLOUTS_PER_TURN = "0"
$env:EVAL_EVERY = "0"
$env:EVAL_AT_START = "0"

# Throughput defaults from the current local pipeline.
$env:PY_SERVICE_MODE = "hybrid"
$env:USE_TRT_INFERENCE = "1"
$env:ONNX_FORCE_CPU = "0"
$env:PY_BATCH_TIMEOUT_MS = "3"
$env:GPU_SERVICE_LOCAL_BATCH_TIMEOUT_MS = "1"
$env:GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS = "250"
$env:GPU_SERVICE_CONTROL_TIMEOUT_MS = "300000"
$env:LEARNER_BATCH_MAX_EPISODES = "4"
$env:LEARNER_BATCH_MAX_STEPS = "2048"
$env:TRAIN_MULTI_MAX_STEPS = "2048"
$env:TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER = "0"
$env:PENDING_TRAIN_MAX = "64"
$env:PENDING_TRAIN_BACKPRESSURE = "block"
$env:PENDING_TRAIN_OFFER_TIMEOUT_MS = "300000"
$env:SCORE_WORKER_THREADS = "4"
$env:GAME_LOG_FREQUENCY = "500"
$env:TRAIN_LOG_EVERY = "50"

$python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
if (Test-Path $python) {
    & $python (Join-Path $repo "scripts\run_local_pbt.py")
} else {
    & py -3.12 (Join-Path $repo "scripts\run_local_pbt.py")
}
