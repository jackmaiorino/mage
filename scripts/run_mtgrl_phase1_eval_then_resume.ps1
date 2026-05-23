param(
    [string]$RunId = "20260430_twogpu_selfplay_eval_cycle_24h_recover",
    [string]$SourceRunId = "20260430_twogpu_selfplay_eval_cycle_24h",
    [int]$ResumeHours = 24,
    [int]$GpuServicePortBase = 26600
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repo "local-training\local_pbt\autonomous_runs\$RunId"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Write-RecoverLog([string]$Message) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "[$ts] $Message" | Tee-Object -FilePath (Join-Path $runDir "recover.log") -Append
}

$python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "py"
}

function Run-Gate([int]$Skill) {
    $evalRunId = "{0}_phase001_cp{1}_recover" -f $SourceRunId, $Skill
    $evalDir = Join-Path $runDir "phase001_eval_cp$Skill"
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $log = Join-Path $evalDir "stdout.log"
    $port = $GpuServicePortBase + 500 + $Skill
    $metricsPort = $port + 1000
    Write-RecoverLog "eval CP$Skill start run_id=$evalRunId"
    $env:INFER_CUDA_DEVICE = "cuda:1"
    $env:TRAIN_CUDA_DEVICE = "cpu"
    $env:CUDA_MEM_FRACTION = "0.80"
    $env:USE_TRT_INFERENCE = "0"
    & $python (Join-Path $repo "scripts\run_cp7_eval_sweep.py") `
        --run-id $evalRunId `
        --profiles "Pauper-Spy-Combo-Value,Pauper-Wildfire-Value" `
        --skill $Skill `
        --games-per-matchup 4 `
        --games-per-job 2 `
        --parallel 4 `
        --ai-threads 8 `
        --gpu-port $port `
        --gpu-metrics-port $metricsPort `
        --timeout-sec 3600 `
        --skip-compile *> $log
    $code = $LASTEXITCODE
    Write-RecoverLog "eval CP$Skill end exit=$code"
    $summary = Join-Path $repo "local-training\local_pbt\cp7_eval_sweeps\$evalRunId\profile_summary.csv"
    if (Test-Path $summary) {
        Copy-Item -LiteralPath $summary -Destination (Join-Path $evalDir "profile_summary.csv") -Force
        Import-Csv $summary | ForEach-Object {
            Write-RecoverLog ("CP{0} {1}: {2}/{3} wr={4}" -f $Skill, $_.profile, $_.wins, $_.total, $_.winrate)
        }
    }
    return $code
}

Write-RecoverLog "recover start source=$SourceRunId resume_hours=$ResumeHours"

$cp1 = Run-Gate -Skill 1
if ($cp1 -ne 0) {
    Write-RecoverLog "CP1 gate failed; continuing to CP3 for evidence"
}
$cp3 = Run-Gate -Skill 3
if ($cp3 -ne 0) {
    Write-RecoverLog "CP3 gate failed; continuing to resumed train cycle"
}

Write-RecoverLog "starting resumed train/eval cycle"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\run_mtgrl_autonomous_cycle.ps1") `
    -RunId "${RunId}_resume" `
    -Hours $ResumeHours `
    -TrainPhaseSeconds 21600 `
    -NumGameRunners 64 `
    -EvalGamesPerMatchup 4 `
    -EvalGamesPerJob 2 `
    -EvalParallel 4 `
    -EvalAiThreads 8 `
    -GpuServicePortBase ($GpuServicePortBase + 1000) *> (Join-Path $runDir "resume_cycle.log")
$resumeCode = $LASTEXITCODE
Write-RecoverLog "resumed cycle end exit=$resumeCode"
