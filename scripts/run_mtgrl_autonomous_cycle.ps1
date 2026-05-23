param(
    [string]$RunId = "",
    [int]$Hours = 24,
    [int]$TrainPhaseSeconds = 21600,
    [int]$NumGameRunners = 64,
    [int]$EvalGamesPerMatchup = 4,
    [int]$EvalGamesPerJob = 2,
    [int]$EvalParallel = 4,
    [int]$EvalAiThreads = 8,
    [string]$EvalProfiles = "Pauper-Spy-Combo-Value,Pauper-Wildfire-Value",
    [string]$RegistryPath = "",
    [int]$EpisodeDelta = 0,
    [int]$GpuServicePortBase = 26500,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$runDir = Join-Path $repo "local-training\local_pbt\autonomous_runs\$RunId"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
$pythonArgsPrefix = @()
if (-not (Test-Path $python)) {
    $python = "py"
    $pythonArgsPrefix = @("-3.12")
}

$profiles = @(
    "Pauper-Spy-Combo-Value",
    "Pauper-Wildfire-Value",
    "Pauper-Rally-Anchor-Value",
    "Pauper-Affinity-Anchor-Value"
)
$profileRoot = Join-Path $repo "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$backupDir = Join-Path $repo "local-training\local_pbt\model_backups\pre_$RunId"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

function Write-CycleLog([string]$Message) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$ts] $Message"
    $line | Tee-Object -FilePath (Join-Path $runDir "cycle.log") -Append | Out-Null
}

function Backup-Models {
    foreach ($profile in $profiles) {
        $src = Join-Path $profileRoot "$profile\models"
        $dst = Join-Path $backupDir $profile
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        foreach ($name in @("model_latest.pt", "model.pt", "mulligan_model.pt", "mulligan_model_latest.pt", "episodes.txt", "mulligan_episodes.txt")) {
            $file = Join-Path $src $name
            if (Test-Path $file) {
                Copy-Item -LiteralPath $file -Destination (Join-Path $dst $name) -Force
            }
        }
    }
}

function Get-MaxEpisode([string]$Profile) {
    $stats = Join-Path $profileRoot "$Profile\logs\stats\training_stats.csv"
    if (-not (Test-Path $stats)) {
        return 0
    }
    $max = 0
    Import-Csv $stats | ForEach-Object {
        $episode = 0
        if ([int]::TryParse($_.episode, [ref]$episode) -and $episode -gt $max) {
            $max = $episode
        }
    }
    return $max
}

function Save-Episodes([string]$Path) {
    $rows = foreach ($profile in $profiles) {
        [pscustomobject]@{
            profile = $profile
            episode = (Get-MaxEpisode $profile)
        }
    }
    $rows | Export-Csv -NoTypeInformation -Path $Path
}

function Set-TrainingEnv([int]$PhaseSeconds, [int]$Port) {
    $env:GPU_SERVICE_PORT = [string]$Port
    if (-not [string]::IsNullOrWhiteSpace($RegistryPath)) {
        $env:REGISTRY_PATH = $RegistryPath
    } else {
        [Environment]::SetEnvironmentVariable("REGISTRY_PATH", $null, "Process")
    }
    $env:TRAIN_PROFILES = "4"
    $env:NUM_GAME_RUNNERS = [string]$NumGameRunners
    $env:MAX_WALL_SECONDS = [string]$PhaseSeconds
    if ($EpisodeDelta -gt 0) {
        $env:TOTAL_EPISODES_DELTA = [string]$EpisodeDelta
        $env:TOTAL_EPISODES = "999999999"
    } else {
        [Environment]::SetEnvironmentVariable("TOTAL_EPISODES_DELTA", $null, "Process")
        $env:TOTAL_EPISODES = "999999999"
    }
    $env:WINRATE_WINDOW = "200"
    $env:OPPONENT_SAMPLER = "self"
    $env:SELFPLAY_OPPONENT_TRAINING = "1"
    $env:RL_HEURISTIC_STEP_REWARDS = "0"
    $env:RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE = "0"
    $env:MCTS_TRAINING_ENABLE = "0"
    $env:ISMCTS_ENABLE = "0"
    $env:ISMCTS_ROLLOUTS_PER_TURN = "0"
    $env:EVAL_AT_START = "0"
    $env:PY_SERVICE_MODE = "hybrid"
    $env:USE_TRT_INFERENCE = "1"
    $env:ONNX_FORCE_CPU = "0"
    $env:TRAIN_CUDA_DEVICE = "cuda:0"
    $env:INFER_CUDA_DEVICE = "cuda:1"
    $env:ONNX_GPU_MEM_LIMIT_MB = "4096"
    $env:ONNX_EXPORT_INTERVAL_TICKS = "40"
    $env:CUDA_MEM_FRACTION = "0.70"
    $env:TRAIN_GPU_MAX_CONCURRENT = "2"
    $env:TRAIN_WORKER_THREADS = "4"
    $env:PENDING_TRAIN_MAX = "128"
    $env:PENDING_TRAIN_BACKPRESSURE = "block"
    $env:PENDING_TRAIN_OFFER_TIMEOUT_MS = "300000"
    $env:LEARNER_BATCH_MAX_EPISODES = "8"
    $env:GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES = "8"
    $env:LEARNER_BATCH_MAX_STEPS = "4096"
    $env:TRAIN_MULTI_MAX_STEPS = "4096"
    $env:TRAIN_CHUNK_SIZE = "256"
    $env:PY_BATCH_TIMEOUT_MS = "3"
    $env:GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS = "250"
    $env:GAME_LOG_FREQUENCY = "500"
    if ([string]::IsNullOrWhiteSpace($env:GAME_LOG_FORMAT)) {
        $env:GAME_LOG_FORMAT = "compact"
    }
    $env:TRAIN_LOG_EVERY = "50"
    $env:PBT_EXPLOIT_INTERVAL = "9999"
    $env:PBT_MIN_EPISODES = "999999999"
}

function Save-TrainingConfig([string]$Path) {
    $keys = @(
        "GPU_SERVICE_PORT",
        "REGISTRY_PATH",
        "TRAIN_PROFILES",
        "NUM_GAME_RUNNERS",
        "MAX_WALL_SECONDS",
        "TOTAL_EPISODES",
        "TOTAL_EPISODES_DELTA",
        "OPPONENT_SAMPLER",
        "SELFPLAY_OPPONENT_TRAINING",
        "RL_HEURISTIC_STEP_REWARDS",
        "RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE",
        "MCTS_TRAINING_ENABLE",
        "ISMCTS_ENABLE",
        "TRAIN_CUDA_DEVICE",
        "INFER_CUDA_DEVICE",
        "ONNX_GPU_MEM_LIMIT_MB",
        "ONNX_EXPORT_INTERVAL_TICKS",
        "CUDA_MEM_FRACTION",
        "TRAIN_GPU_MAX_CONCURRENT",
        "TRAIN_WORKER_THREADS",
        "PENDING_TRAIN_MAX",
        "LEARNER_BATCH_MAX_EPISODES",
        "LEARNER_BATCH_MAX_STEPS",
        "TRAIN_MULTI_MAX_STEPS",
        "TRAIN_CHUNK_SIZE",
        "GAME_LOG_FREQUENCY",
        "GAME_LOG_FORMAT",
        "TRAIN_LOG_EVERY"
    )
    ($keys | ForEach-Object { "$_=$([Environment]::GetEnvironmentVariable($_))" }) |
        Set-Content -Path $Path
}

function Run-TrainingPhase([int]$PhaseIndex, [int]$PhaseSeconds) {
    $phaseDir = Join-Path $runDir ("phase_{0:000}_train" -f $PhaseIndex)
    New-Item -ItemType Directory -Force -Path $phaseDir | Out-Null
    Save-Episodes (Join-Path $phaseDir "episodes_before.csv")
    $port = $GpuServicePortBase + ($PhaseIndex * 10)
    Set-TrainingEnv -PhaseSeconds $PhaseSeconds -Port $port
    Save-TrainingConfig (Join-Path $phaseDir "config.env")
    Write-CycleLog "phase $PhaseIndex train start: ${PhaseSeconds}s runners=$NumGameRunners port=$port"
    $stdout = Join-Path $phaseDir "stdout.log"
    $stderr = Join-Path $phaseDir "stderr.log"
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList ($pythonArgsPrefix + @("scripts\run_local_pbt.py")) `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -PassThru
    $proc.Id | Set-Content -Path (Join-Path $phaseDir "orchestrator.pid")
    Wait-Process -Id $proc.Id
    $proc.Refresh()
    Save-Episodes (Join-Path $phaseDir "episodes_after.csv")
    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) {
        # Start-Process can leave ExitCode unset after a normal child exit on
        # some Windows PowerShell paths. The orchestrator's own stdout carries
        # explicit failure messages; treat an unset code as a normal exit so a
        # wall-clock MAX_WALL_SECONDS stop proceeds to eval.
        $exitCode = 0
    }
    Write-CycleLog "phase $PhaseIndex train end: exit=$exitCode"
    return $exitCode
}

function Run-EvalPhase([int]$PhaseIndex, [int]$Skill) {
    if ($SkipEval) {
        return
    }
    $evalRunId = "{0}_phase{1:000}_cp{2}" -f $RunId, $PhaseIndex, $Skill
    $evalDir = Join-Path $runDir ("phase_{0:000}_eval_cp{1}" -f $PhaseIndex, $Skill)
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $log = Join-Path $evalDir "stdout.log"
    $port = $GpuServicePortBase + 500 + ($PhaseIndex * 10) + $Skill
    $metricsPort = $port + 1000
    Write-CycleLog "phase $PhaseIndex eval CP$Skill start: games=$EvalGamesPerMatchup profiles=$EvalProfiles"
    $env:INFER_CUDA_DEVICE = "cuda:1"
    $env:TRAIN_CUDA_DEVICE = "cpu"
    $env:CUDA_MEM_FRACTION = "0.80"
    $env:USE_TRT_INFERENCE = "0"
    & $python (Join-Path $repo "scripts\run_cp7_eval_sweep.py") `
        --run-id $evalRunId `
        --profiles $EvalProfiles `
        --skill $Skill `
        --games-per-matchup $EvalGamesPerMatchup `
        --games-per-job $EvalGamesPerJob `
        --parallel $EvalParallel `
        --ai-threads $EvalAiThreads `
        --gpu-port $port `
        --gpu-metrics-port $metricsPort `
        --timeout-sec 3600 `
        --skip-compile *> $log
    $code = $LASTEXITCODE
    Write-CycleLog "phase $PhaseIndex eval CP$Skill end: exit=$code run_id=$evalRunId"
    $summary = Join-Path $repo "local-training\local_pbt\cp7_eval_sweeps\$evalRunId\profile_summary.csv"
    if (Test-Path $summary) {
        Copy-Item -LiteralPath $summary -Destination (Join-Path $evalDir "profile_summary.csv") -Force
        Import-Csv $summary | ForEach-Object {
            Write-CycleLog ("phase {0} CP{1} {2}: {3}/{4} wr={5}" -f $PhaseIndex, $Skill, $_.profile, $_.wins, $_.total, $_.winrate)
        }
    }
}

$deadline = (Get-Date).AddHours($Hours)
Backup-Models
"backup=$backupDir" | Set-Content -Path (Join-Path $runDir "backup.txt")
Write-CycleLog "autonomous cycle start: run=$RunId hours=$Hours backup=$backupDir"

$phase = 1
while ((Get-Date) -lt $deadline) {
    $remaining = [int]($deadline - (Get-Date)).TotalSeconds
    if ($remaining -lt 300) {
        break
    }
    $phaseSeconds = [Math]::Min($TrainPhaseSeconds, $remaining)
    $exit = Run-TrainingPhase -PhaseIndex $phase -PhaseSeconds $phaseSeconds
    if ($exit -ne 0) {
        Write-CycleLog "stopping cycle after train failure in phase $phase"
        break
    }
    if ((Get-Date) -lt $deadline) {
        Run-EvalPhase -PhaseIndex $phase -Skill 1
    }
    if ((Get-Date) -lt $deadline) {
        Run-EvalPhase -PhaseIndex $phase -Skill 3
    }
    $phase += 1
}

Write-CycleLog "autonomous cycle complete"
