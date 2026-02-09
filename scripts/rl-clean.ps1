#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean training artifacts for a fresh restart

.DESCRIPTION
    Selectively deletes models, logs, stats, and snapshots.
    Choose which components to clean:
      MainModel     - Main policy model, training stats, snapshots, game logs
      MulliganModel - Mulligan model, mulligan stats/logs
      BothModels    - Everything (full clean slate)

.PARAMETER Force
    Skip confirmation prompts

.EXAMPLE
    .\scripts\rl-clean.ps1
    .\scripts\rl-clean.ps1 -Force
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to repo root
$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\models"
$gamelogsDir = Join-Path $repoRoot "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\gamelogs"
$pythonCodeDir = Join-Path $repoRoot "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode"
$logsDir = Join-Path $repoRoot "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\logs"
$snapshotsDir = Join-Path $modelsDir "snapshots"

Write-Host "`n=== RL Training Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Choose what to clean:" -ForegroundColor Yellow
Write-Host "  MainModel      - Main policy model, training stats, snapshots, game logs, episode counter" -ForegroundColor White
Write-Host "  MulliganModel  - Mulligan model, mulligan stats/logs" -ForegroundColor White
Write-Host "  BothModels     - Everything (full clean slate)" -ForegroundColor White
Write-Host ""

if (-not $Force) {
    Write-Host "Type 'MainModel', 'MulliganModel', or 'BothModels' to confirm (or anything else to cancel):" -ForegroundColor Red
    $response = Read-Host "Selection"
} else {
    $response = "BothModels"
}

$cleanMain = $false
$cleanMulligan = $false

switch ($response) {
    "MainModel" {
        $cleanMain = $true
        Write-Host "`nCleaning: Main Model" -ForegroundColor Cyan
    }
    "MulliganModel" {
        $cleanMulligan = $true
        Write-Host "`nCleaning: Mulligan Model" -ForegroundColor Cyan
    }
    "BothModels" {
        $cleanMain = $true
        $cleanMulligan = $true
        Write-Host "`nCleaning: Both Models" -ForegroundColor Cyan
    }
    default {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Will DELETE the following:" -ForegroundColor Yellow
Write-Host ""

$itemsToDelete = @()
$dirsToClean = @()

# ============================================================
# MAIN MODEL artifacts
# ============================================================
if ($cleanMain) {
    # Main model file
    $mainModelFile = Join-Path $modelsDir "model.pt"
    if (Test-Path $mainModelFile) {
        Write-Host "  Main model:" -ForegroundColor Yellow
        Write-Host "    - model.pt" -ForegroundColor Gray
        $itemsToDelete += $mainModelFile
    }

    # Episode counter
    $episodesFile = Join-Path $modelsDir "episodes.txt"
    if (Test-Path $episodesFile) {
        Write-Host "  Episode counter:" -ForegroundColor Yellow
        Write-Host "    - episodes.txt" -ForegroundColor Gray
        $itemsToDelete += $episodesFile
    }

    # Training stats (main model)
    $mainStatsPatterns = @("training_stats*.csv", "bench_summary*.txt", "head_usage*.csv")
    $mainStatsFiles = @()
    # Check models dir
    if (Test-Path $modelsDir) {
        foreach ($pattern in $mainStatsPatterns) {
            $matches = Get-ChildItem -Path $modelsDir -Filter $pattern -ErrorAction SilentlyContinue
            if ($matches) { $mainStatsFiles += $matches }
        }
    }
    # Check logs/stats dir
    $statsDir = Join-Path $logsDir "stats"
    if (Test-Path $statsDir) {
        foreach ($pattern in @("training_stats*.csv", "head_usage*.csv")) {
            $matches = Get-ChildItem -Path $statsDir -Filter $pattern -ErrorAction SilentlyContinue
            if ($matches) { $mainStatsFiles += $matches }
        }
    }
    # Check logs/health dir
    $healthDir = Join-Path $logsDir "health"
    if (Test-Path $healthDir) {
        $matches = Get-ChildItem -Path $healthDir -Filter "training_health*.csv" -ErrorAction SilentlyContinue
        if ($matches) { $mainStatsFiles += $matches }
    }
    if ($mainStatsFiles) {
        Write-Host "  Main training stats:" -ForegroundColor Yellow
        foreach ($f in $mainStatsFiles) {
            Write-Host "    - $($f.Name)" -ForegroundColor Gray
            $itemsToDelete += $f.FullName
        }
    }

    # Game logs
    $gameLogDirs = @(
        $gamelogsDir,
        (Join-Path $logsDir "games\training"),
        (Join-Path $logsDir "games\evaluation")
    )
    foreach ($gDir in $gameLogDirs) {
        if (Test-Path $gDir) {
            $gamelogs = Get-ChildItem -Path $gDir -File -ErrorAction SilentlyContinue
            if ($gamelogs) {
                Write-Host "  Game logs: $($gamelogs.Count) files in $(Split-Path -Leaf $gDir)/" -ForegroundColor Yellow
                $itemsToDelete += $gamelogs.FullName
            }
        }
    }

    # Snapshots
    if (Test-Path $snapshotsDir) {
        $snapshots = Get-ChildItem -Path $snapshotsDir -Filter "*.pt" -Recurse -ErrorAction SilentlyContinue
        if ($snapshots) {
            Write-Host "  Snapshots: $($snapshots.Count) files" -ForegroundColor Yellow
            $itemsToDelete += $snapshots.FullName
        }
        $dirsToClean += $snapshotsDir
    }

    # Python logs (main model)
    $pythonLogs = @(
        (Join-Path $pythonCodeDir "mtg_ai.log"),
        (Join-Path $pythonCodeDir "mtg_ai.log.1"),
        (Join-Path $pythonCodeDir "mtg_ai.log.2")
    )
    $existingLogs = $pythonLogs | Where-Object { Test-Path $_ }
    if ($existingLogs) {
        Write-Host "  Python logs (main):" -ForegroundColor Yellow
        foreach ($f in $existingLogs) {
            Write-Host "    - $(Split-Path -Leaf $f)" -ForegroundColor Gray
            $itemsToDelete += $f
        }
    }

    # TensorBoard logs
    $tbLogDir = Join-Path $pythonCodeDir "runs"
    if (Test-Path $tbLogDir) {
        Write-Host "  TensorBoard logs: runs/" -ForegroundColor Yellow
        $dirsToClean += $tbLogDir
    }

    # League state
    $leagueDir = Join-Path $logsDir "league"
    if (Test-Path $leagueDir) {
        $leagueFiles = Get-ChildItem -Path $leagueDir -File -ErrorAction SilentlyContinue
        if ($leagueFiles) {
            Write-Host "  League state:" -ForegroundColor Yellow
            foreach ($f in $leagueFiles) {
                Write-Host "    - $($f.Name)" -ForegroundColor Gray
                $itemsToDelete += $f.FullName
            }
        }
    }
}

# ============================================================
# MULLIGAN MODEL artifacts
# ============================================================
if ($cleanMulligan) {
    # Mulligan model file
    $mulliganModelFile = Join-Path $modelsDir "mulligan_model.pt"
    if (Test-Path $mulliganModelFile) {
        Write-Host "  Mulligan model:" -ForegroundColor Yellow
        Write-Host "    - mulligan_model.pt" -ForegroundColor Gray
        $itemsToDelete += $mulliganModelFile
    }

    # Mulligan stats
    $statsDir = Join-Path $logsDir "stats"
    if (Test-Path $statsDir) {
        $mullStats = Get-ChildItem -Path $statsDir -Filter "mulligan_stats*.csv" -ErrorAction SilentlyContinue
        if ($mullStats) {
            Write-Host "  Mulligan stats:" -ForegroundColor Yellow
            foreach ($f in $mullStats) {
                Write-Host "    - $($f.Name)" -ForegroundColor Gray
                $itemsToDelete += $f.FullName
            }
        }
    }
    # Also check models dir for legacy mulligan stats
    if (Test-Path $modelsDir) {
        $mullStatsLegacy = Get-ChildItem -Path $modelsDir -Filter "mulligan_stats*.csv" -ErrorAction SilentlyContinue
        if ($mullStatsLegacy) {
            foreach ($f in $mullStatsLegacy) {
                Write-Host "    - $($f.Name) (legacy)" -ForegroundColor Gray
                $itemsToDelete += $f.FullName
            }
        }
    }

    # Mulligan training log
    $mulliganLogs = @(
        (Join-Path $pythonCodeDir "mulligan_training.log"),
        (Join-Path $pythonCodeDir "mulligan_trace.jsonl")
    )
    $existingMullLogs = $mulliganLogs | Where-Object { Test-Path $_ }
    if ($existingMullLogs) {
        Write-Host "  Mulligan logs:" -ForegroundColor Yellow
        foreach ($f in $existingMullLogs) {
            Write-Host "    - $(Split-Path -Leaf $f)" -ForegroundColor Gray
            $itemsToDelete += $f
        }
    }
}

Write-Host ""
Write-Host "Total items: $($itemsToDelete.Count)" -ForegroundColor Cyan

if ($itemsToDelete.Count -eq 0 -and $dirsToClean.Count -eq 0) {
    Write-Host "`nNothing to clean - already fresh!" -ForegroundColor Green
    exit 0
}

Write-Host "`nDeleting files..." -ForegroundColor Cyan

# Delete files
$deleted = 0
$failed = 0
foreach ($item in $itemsToDelete) {
    try {
        Remove-Item -Path $item -Force -ErrorAction Stop
        $deleted++
    } catch {
        Write-Host "  Failed: $item - $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}

# Clean directories (remove contents but keep directory)
foreach ($dir in $dirsToClean) {
    if (Test-Path $dir) {
        try {
            Get-ChildItem -Path $dir -Recurse | Remove-Item -Force -Recurse -ErrorAction Stop
            Write-Host "  Cleaned directory: $(Split-Path -Leaf $dir)/" -ForegroundColor Gray
        } catch {
            # Ignore errors - might be empty
        }
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host "  Deleted: $deleted items" -ForegroundColor Green
if ($failed -gt 0) {
    Write-Host "  Failed: $failed items" -ForegroundColor Red
}

Write-Host ""
if ($cleanMain -and $cleanMulligan) {
    Write-Host "Both models cleaned. Start fresh training with:" -ForegroundColor Cyan
    Write-Host "  .\scripts\rl-train.ps1" -ForegroundColor White
} elseif ($cleanMain) {
    Write-Host "Main model cleaned. Mulligan model preserved." -ForegroundColor Cyan
    Write-Host "  .\scripts\rl-train.ps1" -ForegroundColor White
} else {
    Write-Host "Mulligan model cleaned. Main model preserved." -ForegroundColor Cyan
    Write-Host "  Mulligan model will retrain from scratch on next run." -ForegroundColor White
}
Write-Host ""
