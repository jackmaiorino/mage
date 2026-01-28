#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean all training artifacts for a fresh restart

.DESCRIPTION
    Deletes models, logs, stats, and snapshots to start training from scratch.
    Use this when you want a completely clean slate.

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
$snapshotsDir = Join-Path $modelsDir "snapshots"

Write-Host "`n=== RL Training Cleanup Script ===" -ForegroundColor Cyan
Write-Host "This will DELETE the following:" -ForegroundColor Yellow
Write-Host ""

# List what will be deleted
$itemsToDelete = @()

# Model files
if (Test-Path $modelsDir) {
    $modelFiles = Get-ChildItem -Path $modelsDir -Filter "*.pt" -ErrorAction SilentlyContinue
    if ($modelFiles) {
        Write-Host "  Models (*.pt):" -ForegroundColor Yellow
        foreach ($f in $modelFiles) {
            Write-Host "    - $($f.Name)" -ForegroundColor Gray
            $itemsToDelete += $f.FullName
        }
    }
}

# Training stats (including any backup copies)
$statsPatterns = @(
    "training_stats*.csv",
    "episodes*.txt",
    "bench_summary*.txt",
    "mulligan_stats*.csv"
)
$statsFiles = @()
if (Test-Path $modelsDir) {
    foreach ($pattern in $statsPatterns) {
        $matches = Get-ChildItem -Path $modelsDir -Filter $pattern -ErrorAction SilentlyContinue
        if ($matches) {
            $statsFiles += $matches
        }
    }
}
if ($statsFiles) {
    Write-Host "  Training stats:" -ForegroundColor Yellow
    foreach ($f in $statsFiles) {
        Write-Host "    - $($f.Name)" -ForegroundColor Gray
        $itemsToDelete += $f.FullName
    }
}

# Game logs (all files in gamelogs directory)
if (Test-Path $gamelogsDir) {
    $gamelogs = Get-ChildItem -Path $gamelogsDir -File -ErrorAction SilentlyContinue
    if ($gamelogs) {
        Write-Host "  Game logs: $($gamelogs.Count) files in gamelogs/" -ForegroundColor Yellow
        $itemsToDelete += $gamelogs.FullName
    }
}

# Snapshots
if (Test-Path $snapshotsDir) {
    $snapshots = Get-ChildItem -Path $snapshotsDir -Filter "*.pt" -Recurse -ErrorAction SilentlyContinue
    if ($snapshots) {
        Write-Host "  Snapshots: $($snapshots.Count) files" -ForegroundColor Yellow
        $itemsToDelete += $snapshots.FullName
    }
}

# Python logs
$pythonLogs = @(
    (Join-Path $pythonCodeDir "mtg_ai.log"),
    (Join-Path $pythonCodeDir "mtg_ai.log.1"),
    (Join-Path $pythonCodeDir "mtg_ai.log.2")
)
$existingLogs = $pythonLogs | Where-Object { Test-Path $_ }
if ($existingLogs) {
    Write-Host "  Python logs:" -ForegroundColor Yellow
    foreach ($f in $existingLogs) {
        Write-Host "    - $(Split-Path -Leaf $f)" -ForegroundColor Gray
        $itemsToDelete += $f
    }
}

# TensorBoard logs (if any)
$tbLogDir = Join-Path $pythonCodeDir "runs"
if (Test-Path $tbLogDir) {
    Write-Host "  TensorBoard logs: runs/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Total items: $($itemsToDelete.Count)" -ForegroundColor Cyan

if ($itemsToDelete.Count -eq 0) {
    Write-Host "`nNothing to clean - already fresh!" -ForegroundColor Green
    exit 0
}

# Confirmation
if (-not $Force) {
    Write-Host ""
    Write-Host "Type 'delete' to confirm deletion (or anything else to cancel):" -ForegroundColor Red
    $response = Read-Host "Confirmation"
    if ($response -ne "delete") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
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

# Delete TensorBoard logs
if (Test-Path $tbLogDir) {
    try {
        Remove-Item -Path $tbLogDir -Recurse -Force -ErrorAction Stop
        Write-Host "  Deleted TensorBoard logs" -ForegroundColor Gray
    } catch {
        Write-Host "  Failed to delete TensorBoard logs: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Clean snapshots directory (but keep the directory itself)
if (Test-Path $snapshotsDir) {
    try {
        Get-ChildItem -Path $snapshotsDir -Recurse | Remove-Item -Force -Recurse -ErrorAction Stop
        Write-Host "  Cleaned snapshots directory" -ForegroundColor Gray
    } catch {
        # Ignore errors - might be empty
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host "  Deleted: $deleted items" -ForegroundColor Green
if ($failed -gt 0) {
    Write-Host "  Failed: $failed items" -ForegroundColor Red
}

Write-Host "`nYou can now start fresh training with:" -ForegroundColor Cyan
Write-Host "  .\scripts\rl-train.ps1" -ForegroundColor White
Write-Host ""
