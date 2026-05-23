param(
    [string]$Name = "",
    [int]$DurationSeconds = 240,
    [int]$NumGameRunners = 64,
    [string]$OpponentSampler = "self",
    [string]$TrainCudaDevice = "cuda:0",
    [string]$InferCudaDevice = "cuda:0",
    [int]$TrainGpuMaxConcurrent = 1,
    [int]$TrainWorkerThreads = 2,
    [int]$PendingTrainMax = 64,
    [int]$LearnerBatchMaxEpisodes = 4,
    [int]$LearnerBatchMaxSteps = 2048,
    [int]$TrainChunkSize = 128,
    [int]$OnnxGpuMemLimitMb = 1024,
    [double]$CudaMemFraction = 0.55,
    [int]$GpuServicePort = 26300,
    [switch]$KeepModelUpdates
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$profiles = @(
    "Pauper-Spy-Combo-Value",
    "Pauper-Wildfire-Value",
    "Pauper-Rally-Anchor-Value",
    "Pauper-Affinity-Anchor-Value"
)
$profileRoot = Join-Path $repo "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"

if ([string]::IsNullOrWhiteSpace($Name)) {
    $Name = "$(Get-Date -Format 'yyyyMMdd_HHmmss')_pbt_benchmark"
}

$outDir = Join-Path $repo "local-training\local_pbt\split_benchmarks\$Name"
$backupDir = Join-Path $repo "local-training\local_pbt\model_backups\pre_$Name"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

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
            Profile = $profile
            Episode = (Get-MaxEpisode $profile)
        }
    }
    $rows | Export-Csv -NoTypeInformation -Path $Path
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

function Restore-Models {
    if ($KeepModelUpdates) {
        return
    }
    foreach ($profile in $profiles) {
        $src = Join-Path $backupDir $profile
        $dst = Join-Path $profileRoot "$profile\models"
        if (Test-Path $src) {
            Copy-Item -Path (Join-Path $src "*") -Destination $dst -Force
        }
    }
}

Backup-Models
Save-Episodes (Join-Path $outDir "episodes_before.csv")

$env:GPU_SERVICE_PORT = [string]$GpuServicePort
$env:TRAIN_PROFILES = "4"
$env:NUM_GAME_RUNNERS = [string]$NumGameRunners
$env:MAX_WALL_SECONDS = [string]$DurationSeconds
$env:TOTAL_EPISODES = "999999999"
$env:OPPONENT_SAMPLER = $OpponentSampler
$env:TRAIN_CUDA_DEVICE = $TrainCudaDevice
$env:INFER_CUDA_DEVICE = $InferCudaDevice
$env:ONNX_GPU_MEM_LIMIT_MB = [string]$OnnxGpuMemLimitMb
$env:CUDA_MEM_FRACTION = [string]$CudaMemFraction
$env:PY_SERVICE_MODE = "hybrid"
$env:TRAIN_GPU_MAX_CONCURRENT = [string]$TrainGpuMaxConcurrent
$env:TRAIN_WORKER_THREADS = [string]$TrainWorkerThreads
$env:PENDING_TRAIN_MAX = [string]$PendingTrainMax
$env:LEARNER_BATCH_MAX_EPISODES = [string]$LearnerBatchMaxEpisodes
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES = [string]$LearnerBatchMaxEpisodes
$env:LEARNER_BATCH_MAX_STEPS = [string]$LearnerBatchMaxSteps
$env:TRAIN_MULTI_MAX_STEPS = [string]$LearnerBatchMaxSteps
$env:TRAIN_CHUNK_SIZE = [string]$TrainChunkSize
$env:PBT_EXPLOIT_INTERVAL = "9999"
$env:PBT_MIN_EPISODES = "999999999"
$env:GAME_LOG_FREQUENCY = "0"
$env:EVAL_AT_START = "0"
$env:USE_TRT_INFERENCE = "1"

$configKeys = @(
    "GPU_SERVICE_PORT",
    "TRAIN_PROFILES",
    "NUM_GAME_RUNNERS",
    "MAX_WALL_SECONDS",
    "OPPONENT_SAMPLER",
    "TRAIN_CUDA_DEVICE",
    "INFER_CUDA_DEVICE",
    "ONNX_GPU_MEM_LIMIT_MB",
    "CUDA_MEM_FRACTION",
    "PY_SERVICE_MODE",
    "TRAIN_GPU_MAX_CONCURRENT",
    "TRAIN_WORKER_THREADS",
    "PENDING_TRAIN_MAX",
    "LEARNER_BATCH_MAX_EPISODES",
    "LEARNER_BATCH_MAX_STEPS",
    "TRAIN_MULTI_MAX_STEPS",
    "TRAIN_CHUNK_SIZE"
)
($configKeys | ForEach-Object { "$_=$([Environment]::GetEnvironmentVariable($_))" }) |
    Set-Content -Path (Join-Path $outDir "config.env")

$stdout = Join-Path $outDir "stdout.log"
$stderr = Join-Path $outDir "stderr.log"
$gpuCsv = Join-Path $outDir "gpu_samples.csv"
$cpuCsv = Join-Path $outDir "cpu_samples.csv"
"utc,index,name,mem_used_mb,mem_total_mb,util_gpu_pct,util_mem_pct,temp_c,power_w" | Set-Content -Path $gpuCsv
"utc,cpu_pct" | Set-Content -Path $cpuCsv

$start = Get-Date
$process = $null
try {
    $python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @("scripts\run_local_pbt.py") `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -PassThru

    while (-not $process.HasExited) {
        $ts = (Get-Date).ToUniversalTime().ToString("o")
        try {
            $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
            "$ts,$cpu" | Add-Content -Path $cpuCsv
        } catch {
        }
        try {
            $rows = & nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
            foreach ($row in $rows) {
                "$ts,$row" | Add-Content -Path $gpuCsv
            }
        } catch {
        }
        Start-Sleep -Seconds 6
        $process.Refresh()
    }
    "exit_code=$($process.ExitCode)" | Set-Content -Path (Join-Path $outDir "exit.txt")
} finally {
    if ($process -and -not $process.HasExited) {
        taskkill /F /T /PID $process.Id | Out-Null
    }
    Restore-Models
}

$end = Get-Date
Save-Episodes (Join-Path $outDir "episodes_after.csv")

$before = Import-Csv (Join-Path $outDir "episodes_before.csv") |
    Group-Object -AsHashTable -AsString -Property Profile
$after = Import-Csv (Join-Path $outDir "episodes_after.csv")
$elapsed = [Math]::Max(1.0, ($end - $start).TotalSeconds)
$throughput = foreach ($row in $after) {
    $b = [int]$before[$row.Profile].Episode
    $a = [int]$row.Episode
    $delta = $a - $b
    [pscustomobject]@{
        Profile = $row.Profile
        Before = $b
        After = $a
        Delta = $delta
        EpsPerSec = [Math]::Round($delta / $elapsed, 3)
    }
}
$throughput | Export-Csv -NoTypeInformation -Path (Join-Path $outDir "throughput.csv")

$summary = [pscustomobject]@{
    bench = $Name
    elapsed_sec = [Math]::Round($elapsed, 1)
    total_delta = ($throughput | Measure-Object -Property Delta -Sum).Sum
    total_eps_per_sec = [Math]::Round((($throughput | Measure-Object -Property Delta -Sum).Sum) / $elapsed, 3)
    output_dir = $outDir
    restored_model_weights = (-not $KeepModelUpdates.IsPresent)
}
$summary | ConvertTo-Json | Set-Content -Path (Join-Path $outDir "summary.json")
$summary | Format-List
$throughput | Format-Table -AutoSize
