param(
  # If true, suppress output except errors
  [switch]$Quiet = $false,
  # If true, also kill league orchestrator shell processes
  [switch]$KillOrchestrators = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Log($msg) {
  if (-not $Quiet) { Write-Host $msg }
}

function Get-AncestorPids([int]$processId) {
  $result = New-Object System.Collections.Generic.HashSet[int]
  $current = $processId
  while ($current -gt 0) {
    if (-not $result.Add($current)) { break }
    try {
      $proc = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $current) -ErrorAction SilentlyContinue
      if ($null -eq $proc) { break }
      $parent = [int]$proc.ParentProcessId
      if ($parent -le 0 -or $parent -eq $current) { break }
      $current = $parent
    } catch {
      break
    }
  }
  return $result
}

$excludedPids = Get-AncestorPids -processId $PID

function KillByPredicate($nameList, $predicate, $label) {
  $procs = Get-CimInstance Win32_Process | Where-Object {
    ($nameList -contains $_.Name) -and (& $predicate $_)
  }
  foreach ($p in $procs) {
    if ($excludedPids.Contains([int]$p.ProcessId)) {
      continue
    }
    try {
      Log ("Killing {0} PID={1}" -f $label, $p.ProcessId)
      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
      Write-Warning ("Failed to kill PID={0}: {1}" -f $p.ProcessId, $_.Exception.Message)
    }
  }
}

# Kill Python workers running our py4j entrypoints.
# Keep repo scoping, but also match common relative-path command lines used by wrapper scripts.
KillByPredicate @("python.exe","pythonw.exe","python3.12.exe","python3.11.exe","python3.10.exe") `
  { param($p)
    $cmd = [string]$p.CommandLine
    if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
    $isPy4j = ($cmd -like "*py4j_entry_point.py*") -or ($cmd -like "*draft_py4j_entry_point.py*")
    if (-not $isPy4j) { return $false }
    return ($cmd -like "*$repoRoot*") -or ($cmd -like "*Mage.Server.Plugins*Mage.Player.AIRL*MLPythonCode*")
  } "py4j_python"

# Extra safety: free known Py4J port ranges used by train/eval workers.
# Query listeners once (much faster than per-port calls).
$py4jPorts = @()
$py4jPorts += 25334..25345
$py4jPorts += 26334..26345
try {
  $listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {
    $py4jPorts -contains $_.LocalPort
  }
  foreach ($listener in $listeners) {
    try {
      $pid = $listener.OwningProcess
      $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pid" -ErrorAction SilentlyContinue
      if ($null -eq $proc) { continue }
      $name = [string]$proc.Name
      $cmd = [string]$proc.CommandLine
      $isKnown = ($name -in @("python.exe","pythonw.exe","python3.12.exe","python3.11.exe","python3.10.exe","java.exe","javaw.exe"))
      $isRepoProcess = ($cmd -like "*$repoRoot*") -or ($cmd -like "*Mage.Server.Plugins*Mage.Player.AIRL*") -or ($cmd -like "*py4j_entry_point.py*") -or ($cmd -like "*RLTrainer*")
      if ($isKnown -and $isRepoProcess) {
        Log ("Killing port owner PID={0} port={1}" -f $pid, $listener.LocalPort)
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
      }
    } catch {
      Write-Warning ("Failed killing listener PID={0} port={1}: {2}" -f $listener.OwningProcess, $listener.LocalPort, $_.Exception.Message)
    }
  }
} catch {
  Write-Warning ("Failed scanning listener ports for cleanup: {0}" -f $_.Exception.Message)
}

# Kill RLTrainer Java processes for this repo (mvn exec or direct java)
KillByPredicate @("java.exe","javaw.exe") `
  { param($p)
    ($p.CommandLine -like "*mage.player.ai.rl.RLTrainer*") -or ($p.CommandLine -like "*-Dexec.mainClass=mage.player.ai.rl.RLTrainer*")
  } "rltrainer_java"

# Kill Maven invocations that are running RLTrainer (best-effort)
KillByPredicate @("cmd.exe","powershell.exe","pwsh.exe") `
  { param($p)
    ($p.CommandLine -like "*mvn*exec:java*") -and ($p.CommandLine -like "*RLTrainer*")
  } "mvn_exec_java"

if ($KillOrchestrators) {
  # Kill league orchestrator shells so they cannot respawn trainers after cleanup.
  KillByPredicate @("powershell.exe","pwsh.exe") `
    { param($p)
      ($p.CommandLine -like "*scripts\rl-league-run.ps1*") -or ($p.CommandLine -like "*scripts/rl-league-run.ps1*")
    } "rl_orchestrator_shell"
}

Log "rl-stop: done"

