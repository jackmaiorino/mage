param(
  # If true, suppress output except errors
  [switch]$Quiet = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Log($msg) {
  if (-not $Quiet) { Write-Host $msg }
}

function KillByPredicate($nameList, $predicate, $label) {
  $procs = Get-CimInstance Win32_Process | Where-Object {
    ($nameList -contains $_.Name) -and (& $predicate $_)
  }
  foreach ($p in $procs) {
    try {
      Log ("Killing {0} PID={1}" -f $label, $p.ProcessId)
      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
      Write-Warning ("Failed to kill PID={0}: {1}" -f $p.ProcessId, $_.Exception.Message)
    }
  }
}

# Kill Python workers running our py4j entrypoint from this repo
KillByPredicate @("python.exe","pythonw.exe","python3.12.exe","python3.11.exe","python3.10.exe") `
  { param($p)
    ($p.CommandLine -like "*py4j_entry_point.py*") -and ($p.CommandLine -like "*$repoRoot*")
  } "py4j_python"

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

Log "rl-stop: done"

