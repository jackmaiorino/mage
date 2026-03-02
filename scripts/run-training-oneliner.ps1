param(
  [bool]$FollowLogs = $true,
  [int]$LogPollSeconds = 2,
  [bool]$StopExistingOrchestrators = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trainCommandsPath = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/traincommands.txt"
$leagueScript = Join-Path $repoRoot "scripts/rl-league-run.ps1"
$stopScriptPs1 = Join-Path $repoRoot "scripts/rl-stop.ps1"
$stopScriptSh = Join-Path $repoRoot "scripts/rl-stop.sh"
$isWindowsHost = $false
try {
  $isWindowsHost = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
} catch {
  $isWindowsHost = ($env:OS -eq "Windows_NT")
}
$stopScript = if ($isWindowsHost) { $stopScriptPs1 } else { $stopScriptSh }
if (-not (Test-Path -LiteralPath $stopScript)) {
  $fallbackStopScript = if ($isWindowsHost) { $stopScriptSh } else { $stopScriptPs1 }
  if (Test-Path -LiteralPath $fallbackStopScript) {
    $stopScript = $fallbackStopScript
  }
}
$shellExecutable = if ($isWindowsHost) { "powershell" } else { "pwsh" }
if (-not (Get-Command $shellExecutable -ErrorAction SilentlyContinue)) {
  throw ("Required shell executable not found: {0}" -f $shellExecutable)
}

function Invoke-OrchestratorCleanup {
  if (-not (Test-Path -LiteralPath $stopScript)) {
    Write-Warning ("Cleanup script missing: {0}" -f $stopScript)
    return
  }
  try {
    if ([string]$stopScript -match "\.sh$") {
      & bash $stopScript -q
    } else {
      & $stopScript -Quiet -KillOrchestrators
    }
  } catch {
    Write-Warning ("Cleanup failed: {0}" -f $_.Exception.Message)
  }
}

function Stop-ExistingOrchestrators {
  if (-not $isWindowsHost) {
    return
  }
  try {
    $running = Get-CimInstance Win32_Process | Where-Object {
      ($_.Name -in @("powershell.exe","pwsh.exe")) -and
      ($_.CommandLine -like "*scripts\\rl-league-run.ps1*") -and
      ($_.CommandLine -like "*pauper_spy_pbt_registry.json*")
    }
    foreach ($proc in $running) {
      try {
        Write-Warning ("Stopping stale orchestrator pid={0}" -f $proc.ProcessId)
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
      } catch {
        Write-Warning ("Failed to stop stale orchestrator pid={0}: {1}" -f $proc.ProcessId, $_.Exception.Message)
      }
    }
  } catch {
    Write-Warning ("Unable to enumerate stale orchestrators: {0}" -f $_.Exception.Message)
  }
}

function Quote-Single([string]$s) {
  if ($null -eq $s) { return "''" }
  return "'" + ($s -replace "'", "''") + "'"
}

function Write-NewLogLines([string]$path, [ref]$offset, [string]$prefix) {
  if (-not (Test-Path -LiteralPath $path)) {
    return
  }
  $stream = $null
  $reader = $null
  try {
    $stream = [System.IO.File]::Open(
      $path,
      [System.IO.FileMode]::Open,
      [System.IO.FileAccess]::Read,
      [System.IO.FileShare]::ReadWrite
    )
    if ($stream.Length -lt $offset.Value) {
      $offset.Value = 0L
    }
    if ($stream.Length -le $offset.Value) {
      return
    }
    [void]$stream.Seek([long]$offset.Value, [System.IO.SeekOrigin]::Begin)
    $reader = New-Object System.IO.StreamReader($stream)
    $content = $reader.ReadToEnd()
    $offset.Value = [long]$stream.Position
    if ([string]::IsNullOrWhiteSpace($content)) {
      return
    }
    $lines = $content -split "`r?`n"
    foreach ($line in $lines) {
      if (-not [string]::IsNullOrWhiteSpace($line)) {
        Write-Host ("[{0}] {1}" -f $prefix, $line)
      }
    }
  } catch {
  } finally {
    if ($null -ne $reader) { $reader.Dispose() }
    if ($null -ne $stream) { $stream.Dispose() }
  }
}

$script:cancelRequested = $false
$cancelSubscribed = $false
$cancelHandler = [System.ConsoleCancelEventHandler]{
  param([object]$sender, [System.ConsoleCancelEventArgs]$eventArgs)
  $script:cancelRequested = $true
  $eventArgs.Cancel = $true
}
try {
  [System.Console]::add_CancelKeyPress($cancelHandler)
  $cancelSubscribed = $true
} catch {
  Write-Warning ("CancelKeyPress hook unavailable in this host; Ctrl+C cleanup may be best-effort: {0}" -f $_.Exception.Message)
}

$orchestrator = $null
$orchestratorStdoutPath = ""
$orchestratorStderrPath = ""
try {
  if ($StopExistingOrchestrators) {
    Stop-ExistingOrchestrators
    Invoke-OrchestratorCleanup
  }

  Get-Content $trainCommandsPath |
    Where-Object { $_ -match '^\s*\$env:' } |
    ForEach-Object { Invoke-Expression $_ }

  # Reduce per-game wall time and effective concurrent load for 3x PBT runs.
  $env:GAME_TIMEOUT_SEC = "420"

  $command = "& " + (Quote-Single $leagueScript) +
    " -RegistryPath " + (Quote-Single "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json") +
    " -SequentialTraining `$false" +
    " -EnablePbt `$true" +
    " -PbtExploitIntervalMinutes 240" +
    " -PbtMinEpisodesBeforeFirstExploit 12000" +
    " -PbtMinEpisodeDeltaPerProfile 10000" +
    " -PbtMutationPct 0.20" +
    " -PbtMinPopulationSize 3" +
    " -GameLogFrequency 500" +
    " -StallRestartMinutes 25" +
    " -EvalEveryMinutes 180" +
    " -NoEval" +
    " -NumGameRunners 8" +
    " -TotalEpisodes 1000000"

  $argList = @(
    "-NoLogo",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $command
  )

  $orchestratorLogsDir = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
  New-Item -ItemType Directory -Force -Path $orchestratorLogsDir | Out-Null
  $ts = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
  $orchestratorStdoutPath = Join-Path $orchestratorLogsDir ("oneliner_orchestrator_{0}.stdout.log" -f $ts)
  $orchestratorStderrPath = Join-Path $orchestratorLogsDir ("oneliner_orchestrator_{0}.stderr.log" -f $ts)

  $startArgs = @{
    FilePath = $shellExecutable
    ArgumentList = $argList
    WorkingDirectory = $repoRoot
    PassThru = $true
    RedirectStandardOutput = $orchestratorStdoutPath
    RedirectStandardError = $orchestratorStderrPath
  }
  if ($isWindowsHost) {
    $startArgs.WindowStyle = "Hidden"
  }
  $orchestrator = Start-Process @startArgs
  Write-Host ("Started league orchestrator pid={0}" -f $orchestrator.Id)
  Write-Host ("Orchestrator stdout: {0}" -f $orchestratorStdoutPath)
  Write-Host ("Orchestrator stderr: {0}" -f $orchestratorStderrPath)
  if ($FollowLogs) {
    Write-Host ("Following orchestrator logs every {0}s (use -FollowLogs `$false to disable)" -f [Math]::Max(1, $LogPollSeconds))
  }

  $stdoutOffset = 0L
  $stderrOffset = 0L

  while ($true) {
    $orchestrator.Refresh()
    if ($orchestrator.HasExited) {
      break
    }
    if ($script:cancelRequested) {
      Write-Warning ("Ctrl+C received; stopping orchestrator pid={0} and trainer processes..." -f $orchestrator.Id)
      try {
        Stop-Process -Id $orchestrator.Id -Force -ErrorAction SilentlyContinue
      } catch {
      }
      break
    }
    if ($FollowLogs) {
      Write-NewLogLines $orchestratorStdoutPath ([ref]$stdoutOffset) "ORCH"
      Write-NewLogLines $orchestratorStderrPath ([ref]$stderrOffset) "ORCH-ERR"
    }
    Start-Sleep -Seconds ([Math]::Max(1, $LogPollSeconds))
  }

  if ($FollowLogs) {
    Write-NewLogLines $orchestratorStdoutPath ([ref]$stdoutOffset) "ORCH"
    Write-NewLogLines $orchestratorStderrPath ([ref]$stderrOffset) "ORCH-ERR"
  }

  if ($null -ne $orchestrator) {
    $orchestrator.Refresh()
    if ($orchestrator.HasExited) {
      if ($orchestrator.ExitCode -ne 0) {
        $exitCodeText = if ($null -eq $orchestrator.ExitCode) { "<unknown>" } else { [string]$orchestrator.ExitCode }
        Write-Warning ("League orchestrator exited early with code {0}. Run scripts/rl-league-run.ps1 directly to see startup errors." -f $exitCodeText)
      }
      exit $orchestrator.ExitCode
    }
  }
  exit 0
} finally {
  if ($cancelSubscribed) {
    try {
      [System.Console]::remove_CancelKeyPress($cancelHandler)
    } catch {
    }
  }
  Invoke-OrchestratorCleanup
}
