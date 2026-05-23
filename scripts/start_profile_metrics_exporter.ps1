param(
    [int]$Port = 9102,
    [int]$Window = 200,
    [string]$Profiles = "",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repo "local-training\local_pbt"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$python = Join-Path $repo ".mtgrl_venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$stdout = Join-Path $logDir "profile_metrics_exporter.stdout.log"
$stderr = Join-Path $logDir "profile_metrics_exporter.stderr.log"
$script = Join-Path $repo "scripts\export_training_profile_metrics.py"

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match "export_training_profile_metrics.py" -and
    $_.ProcessId -ne $PID
}

if ($existing) {
    if ($Restart) {
        $existing | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 1
    } else {
        $existing | Select-Object ProcessId,CommandLine
        Write-Host "Profile metrics exporter already appears to be running. Use -Restart to replace it."
        exit 0
    }
}

$argsList = @($script, "--port", "$Port", "--window", "$Window")
if (-not [string]::IsNullOrWhiteSpace($Profiles)) {
    $argsList += @("--profiles", $Profiles)
}

$process = Start-Process -FilePath $python `
    -ArgumentList $argsList `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

Write-Host "Started profile metrics exporter pid=$($process.Id) port=$Port"
