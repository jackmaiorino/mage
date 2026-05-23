param(
    [int]$GamesPerMatchup = 5,
    [int]$GamesPerJob = 0,
    [int]$Skill = 7,
    [int]$Parallel = 4,
    [int]$AiThreads = 4,
    [int]$BatchTimeoutMs = 0,
    [switch]$SkipCompile,
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$registry = Join-Path $repo "docs\mtgrl\experiments\2026-04-25-architecture-validation-plan\phase-b-generalist-registry.json"
$profile = "Pauper-Generalist-Value-v2"

if (-not (Test-Path $registry)) {
    throw "Missing experiment registry: $registry"
}

$python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
$pythonArgs = @(
    (Join-Path $repo "scripts\run_cp7_eval_sweep.py"),
    "--registry", $registry,
    "--profiles", $profile,
    "--games-per-matchup", "$GamesPerMatchup",
    "--games-per-job", "$GamesPerJob",
    "--skill", "$Skill",
    "--parallel", "$Parallel",
    "--ai-threads", "$AiThreads",
    "--split-agent-decks"
)
if ($SkipCompile) {
    $pythonArgs += "--skip-compile"
}
if (-not [string]::IsNullOrWhiteSpace($RunId)) {
    $pythonArgs += @("--run-id", $RunId)
}

$oldBatchTimeout = $env:PY_BATCH_TIMEOUT_MS
if ($BatchTimeoutMs -gt 0) {
    $env:PY_BATCH_TIMEOUT_MS = "$BatchTimeoutMs"
}

if (Test-Path $python) {
    try {
        & $python $pythonArgs
    } finally {
        if ($null -eq $oldBatchTimeout) {
            Remove-Item Env:\PY_BATCH_TIMEOUT_MS -ErrorAction SilentlyContinue
        } else {
            $env:PY_BATCH_TIMEOUT_MS = $oldBatchTimeout
        }
    }
} else {
    try {
        & py -3.12 $pythonArgs
    } finally {
        if ($null -eq $oldBatchTimeout) {
            Remove-Item Env:\PY_BATCH_TIMEOUT_MS -ErrorAction SilentlyContinue
        } else {
            $env:PY_BATCH_TIMEOUT_MS = $oldBatchTimeout
        }
    }
}
