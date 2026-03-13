param(
  [string]$OutputDir = "local-training/hpc/bundles",
  [switch]$SkipBuild,
  [string]$ScpDestination = $env:MAGE_HPC_BUNDLE_DEST,
  [bool]$ScpCreateRemoteDir = $false,
  [string]$CredentialFile = $env:MAGE_HPC_CREDENTIAL_FILE,
  [string]$SshSessionDir,
  [string]$SshControlPath,
  [string]$SshTarget
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$moduleRelPath = "Mage.Server.Plugins/Mage.Player.AIRL"
$modulePath = Join-Path $repoRoot $moduleRelPath
$moduleTarget = Join-Path $modulePath "target"
$stageRoot = Join-Path $moduleTarget "hpc-runtime-stage"
$stageAppDir = Join-Path $stageRoot "app"
$stageLibDir = Join-Path $stageRoot "lib"
$outputRoot = Join-Path $repoRoot $OutputDir
$credentialTransferScript = Join-Path $repoRoot "scripts/hpc/transfer_hpc_file.py"

if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
  throw "mvn was not found on PATH. Install Maven (or run from a shell where it is available)."
}
if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
  throw "tar was not found on PATH. This script uses tar to produce a .tar.gz bundle."
}

function Resolve-ScpUploadTarget([string]$destination, [string]$localFileName) {
  if ($destination -notmatch "^(?<host>[^:]+):(?<path>.+)$") {
    throw "ScpDestination must be in 'user@host:/absolute/path/' form. Got: $destination"
  }
  $remoteHost = $Matches["host"]
  $remotePath = $Matches["path"]
  $isDirTarget = $remotePath.EndsWith("/")
  $uploadPath = if ($isDirTarget) { $remotePath + $localFileName } else { $remotePath }
  return [PSCustomObject]@{
    Host = $remoteHost
    Path = $remotePath
    UploadPath = $uploadPath
    IsDirTarget = $isDirTarget
  }
}

function Load-CredentialMetadata([string]$credentialFile) {
  if ([string]::IsNullOrWhiteSpace($credentialFile)) {
    return $null
  }
  if (-not (Test-Path -LiteralPath $credentialFile)) {
    throw "CredentialFile does not exist: $credentialFile"
  }
  return Get-Content -LiteralPath $credentialFile -Raw | ConvertFrom-Json
}

function Assert-UploadHostMatchesCredential([string]$uploadHost, $credential) {
  if ($null -eq $credential) {
    return
  }

  $expectedTargets = @(
    [string]$credential.host
    ("{0}@{1}" -f [string]$credential.username, [string]$credential.host)
  )
  if ($expectedTargets -notcontains $uploadHost) {
    throw "ScpDestination host '$uploadHost' does not match credential host '$($credential.host)'."
  }
}

function Resolve-SshMasterSession([string]$sessionDir, [string]$controlPath, [string]$sshTarget) {
  if ([string]::IsNullOrWhiteSpace($sessionDir) -and
      [string]::IsNullOrWhiteSpace($controlPath) -and
      [string]::IsNullOrWhiteSpace($sshTarget)) {
    return $null
  }

  if (-not [string]::IsNullOrWhiteSpace($sessionDir)) {
    $metadataPath = Join-Path $sessionDir "session.json"
    if (-not (Test-Path -LiteralPath $metadataPath)) {
      throw "SSH session metadata not found: $metadataPath"
    }

    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($controlPath)) {
      $controlPath = [string]$metadata.control_path
    }
    if ([string]::IsNullOrWhiteSpace($sshTarget)) {
      $sshTarget = [string]$metadata.ssh_target
    }
  }

  if ([string]::IsNullOrWhiteSpace($controlPath)) {
    throw "SshControlPath is required when SshSessionDir is not provided."
  }
  if ([string]::IsNullOrWhiteSpace($sshTarget)) {
    throw "SshTarget is required when SshSessionDir is not provided."
  }

  return [PSCustomObject]@{
    ControlPath = $controlPath
    SshTarget = $sshTarget
  }
}

function Assert-SshMasterAvailable($session) {
  if ($null -eq $session) {
    return
  }

  if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "ssh was not found on PATH, but an SSH master session was requested."
  }

  $checkArgs = @(
    "-S"
    $session.ControlPath
    "-O"
    "check"
    $session.SshTarget
  )

  $checkExitCode = Invoke-NativeCommand -commandName "ssh" -arguments $checkArgs -Quiet
  if ($checkExitCode -ne 0) {
    throw "SSH master session is not available. Re-authenticate with start_ssh_master.ps1."
  }
}

function Invoke-NativeCommand([string]$commandName, [string[]]$arguments, [switch]$Quiet) {
  $command = Get-Command $commandName -ErrorAction Stop
  $startArgs = @{
    FilePath = $command.Source
    ArgumentList = $arguments
    Wait = $true
    PassThru = $true
  }

  $stdoutPath = $null
  $stderrPath = $null
  if ($Quiet) {
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $startArgs.RedirectStandardOutput = $stdoutPath
    $startArgs.RedirectStandardError = $stderrPath
  } else {
    $startArgs.NoNewWindow = $true
  }

  try {
    $process = Start-Process @startArgs
    return $process.ExitCode
  }
  finally {
    if ($Quiet) {
      if ($null -ne $stdoutPath -and (Test-Path -LiteralPath $stdoutPath)) {
        Remove-Item -LiteralPath $stdoutPath -Force
      }
      if ($null -ne $stderrPath -and (Test-Path -LiteralPath $stderrPath)) {
        Remove-Item -LiteralPath $stderrPath -Force
      }
    }
  }
}

function Invoke-CredentialedUpload([string]$credentialFile, [string]$localPath, [string]$remotePath, [bool]$createRemoteDir) {
  if (-not (Test-Path -LiteralPath $credentialTransferScript)) {
    throw "Credential transfer helper not found: $credentialTransferScript"
  }
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found on PATH, but CredentialFile was provided."
  }

  $arguments = @(
    $credentialTransferScript
    "--credential-file"
    $credentialFile
    "--mode"
    "upload"
    "--local-path"
    $localPath
    "--remote-path"
    $remotePath
  )
  if ($createRemoteDir) {
    $arguments += "--mkdirs"
  }

  return Invoke-NativeCommand -commandName "python" -arguments $arguments
}

function New-SshMasterConfigFile($session) {
  if ($null -eq $session) {
    return $null
  }

  $configPath = [System.IO.Path]::GetTempFileName()
  $controlPath = $session.ControlPath -replace "\\", "/"
  $configLines = @(
    "Host *"
    "  BatchMode yes"
    "  ControlMaster no"
    "  ControlPath $controlPath"
  )
  Set-Content -LiteralPath $configPath -Value $configLines -Encoding ASCII
  return $configPath
}

function Get-SshConfigArgs([string]$sshConfigPath) {
  if ([string]::IsNullOrWhiteSpace($sshConfigPath)) {
    return @()
  }

  return @(
    "-F"
    $sshConfigPath
  )
}

function Invoke-SshCommandWithSession([string]$sshConfigPath, [string]$sshTarget, [string]$remoteCommand) {
  $sshArgs = Get-SshConfigArgs -sshConfigPath $sshConfigPath
  return Invoke-NativeCommand -commandName "ssh" -arguments ($sshArgs + @($sshTarget, $remoteCommand))
}

function Invoke-ScpUploadWithSession([string]$sshConfigPath, [string]$localPath, [string]$remoteTarget) {
  $scpArgs = Get-SshConfigArgs -sshConfigPath $sshConfigPath
  return Invoke-NativeCommand -commandName "scp" -arguments ($scpArgs + @($localPath, $remoteTarget))
}

function Assert-UploadHostMatchesSession([string]$uploadHost, $session) {
  if ($null -eq $session) {
    return
  }

  if ($uploadHost -ine $session.SshTarget) {
    throw "ScpDestination host '$uploadHost' does not match SSH master target '$($session.SshTarget)'."
  }
}

if (Test-Path -LiteralPath $stageRoot) {
  Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageAppDir | Out-Null
New-Item -ItemType Directory -Force -Path $stageLibDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

Push-Location $repoRoot
try {
  if (-not $SkipBuild) {
    & mvn -q -pl $moduleRelPath -am -DskipTests install
    if ($LASTEXITCODE -ne 0) {
      throw "Maven build failed with exit code $LASTEXITCODE"
    }
  }

  & mvn -q -pl $moduleRelPath -DskipTests dependency:copy-dependencies "-DincludeScope=runtime" "-DoutputDirectory=$stageLibDir"
  if ($LASTEXITCODE -ne 0) {
    throw "Maven dependency copy failed with exit code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}

$moduleJar = Get-ChildItem -LiteralPath $moduleTarget -Filter "*.jar" -File |
  Where-Object {
    $_.Name -notmatch "sources" -and
    $_.Name -notmatch "javadoc" -and
    $_.Name -notmatch "original" -and
    $_.Name -notmatch "tests"
  } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($null -eq $moduleJar) {
  throw "Could not find built AIRL module jar in $moduleTarget"
}

Copy-Item -LiteralPath $moduleJar.FullName -Destination (Join-Path $stageAppDir $moduleJar.Name) -Force

$gitSha = ""
try {
  $gitSha = (& git -C $repoRoot rev-parse --short HEAD).Trim()
} catch {
  $gitSha = ""
}
if ([string]::IsNullOrWhiteSpace($gitSha)) {
  $gitSha = "nogit"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundleName = "rl-runtime-$gitSha-$stamp"
$bundleDir = Join-Path $outputRoot $bundleName
$bundleTar = Join-Path $outputRoot "$bundleName.tar.gz"

if (Test-Path -LiteralPath $bundleDir) {
  Remove-Item -LiteralPath $bundleDir -Recurse -Force
}
if (Test-Path -LiteralPath $bundleTar) {
  Remove-Item -LiteralPath $bundleTar -Force
}

New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null
Copy-Item -LiteralPath $stageAppDir -Destination (Join-Path $bundleDir "app") -Recurse -Force
Copy-Item -LiteralPath $stageLibDir -Destination (Join-Path $bundleDir "lib") -Recurse -Force

$manifest = [ordered]@{
  created_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  git_sha = $gitSha
  module = $moduleRelPath
  module_jar = $moduleJar.Name
  main_class = "mage.player.ai.rl.RLTrainer"
}
$manifestPath = Join-Path $bundleDir "manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Push-Location $outputRoot
try {
  # Use Windows system tar to avoid Git Bash tar path issues
  $sysTar = Join-Path $env:SystemRoot "system32\tar.exe"
  if (-not (Test-Path $sysTar)) { $sysTar = "tar" }
  & $sysTar -czf $bundleTar $bundleName
  if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}

Write-Host "Built runtime bundle:"
Write-Host "  $bundleTar"

if (-not [string]::IsNullOrWhiteSpace($ScpDestination)) {
  if (-not [string]::IsNullOrWhiteSpace($CredentialFile) -and
      (-not [string]::IsNullOrWhiteSpace($SshSessionDir) -or
       -not [string]::IsNullOrWhiteSpace($SshControlPath) -or
       -not [string]::IsNullOrWhiteSpace($SshTarget))) {
    throw "CredentialFile and SSH master session options are mutually exclusive. Choose one upload path."
  }

  $credential = Load-CredentialMetadata -credentialFile $CredentialFile
  $sshMasterSession = Resolve-SshMasterSession -sessionDir $SshSessionDir -controlPath $SshControlPath -sshTarget $SshTarget
  $sshConfigPath = $null
  if ($null -eq $credential -and -not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp was not found on PATH, but ScpDestination was provided."
  }
  if ($null -eq $credential -and $ScpCreateRemoteDir -and -not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "ssh was not found on PATH, but ScpCreateRemoteDir=true requires ssh."
  }

  $bundleFileName = [System.IO.Path]::GetFileName($bundleTar)
  $target = Resolve-ScpUploadTarget -destination $ScpDestination -localFileName $bundleFileName
  Assert-UploadHostMatchesCredential -uploadHost $target.Host -credential $credential
  Assert-UploadHostMatchesSession -uploadHost $target.Host -session $sshMasterSession
  Assert-SshMasterAvailable -session $sshMasterSession

  if ($null -ne $credential) {
    $uploadTarget = "$($target.Host):$($target.UploadPath)"
    Write-Host "Uploading bundle via credentialed SFTP:"
    Write-Host "  $uploadTarget"
    $uploadExitCode = Invoke-CredentialedUpload -credentialFile $CredentialFile -localPath $bundleTar -remotePath $target.UploadPath -createRemoteDir $ScpCreateRemoteDir
    if ($uploadExitCode -ne 0) {
      throw "credentialed upload failed with exit code $uploadExitCode"
    }
    Write-Host "Uploaded bundle to:"
    Write-Host "  $uploadTarget"
  } else {
    $sshConfigPath = New-SshMasterConfigFile -session $sshMasterSession

    try {
      if ($ScpCreateRemoteDir) {
        $remoteDir = ""
        if ($target.IsDirTarget) {
          $remoteDir = $target.Path.TrimEnd("/")
        } else {
          $lastSlash = $target.Path.LastIndexOf("/")
          if ($lastSlash -le 0) {
            throw "ScpDestination path must be absolute. Got: $($target.Path)"
          }
          $remoteDir = $target.Path.Substring(0, $lastSlash)
        }

        if (-not [string]::IsNullOrWhiteSpace($remoteDir)) {
          $remoteDirEscaped = $remoteDir.Replace('"', '\"')
          $mkdirCmd = "mkdir -p `"$remoteDirEscaped`""
          $mkdirExitCode = Invoke-SshCommandWithSession -sshConfigPath $sshConfigPath -sshTarget $target.Host -remoteCommand $mkdirCmd
          if ($mkdirExitCode -ne 0) {
            throw "Failed to create remote directory via ssh (exit code $mkdirExitCode)"
          }
        }
      }

      $uploadTarget = "$($target.Host):$($target.UploadPath)"
      Write-Host "Uploading bundle via scp:"
      Write-Host "  $uploadTarget"
      if ($null -ne $sshMasterSession) {
        Write-Host "Using SSH master session:"
        Write-Host "  $($sshMasterSession.ControlPath)"
      }
      $scpExitCode = Invoke-ScpUploadWithSession -sshConfigPath $sshConfigPath -localPath $bundleTar -remoteTarget $uploadTarget
      if ($scpExitCode -ne 0) {
        throw "scp upload failed with exit code $scpExitCode"
      }
      Write-Host "Uploaded bundle to:"
      Write-Host "  $uploadTarget"
    }
    finally {
      if (-not [string]::IsNullOrWhiteSpace($sshConfigPath) -and (Test-Path -LiteralPath $sshConfigPath)) {
        Remove-Item -LiteralPath $sshConfigPath -Force
      }
    }
  }
}

Write-Host ""
Write-Host "Set this in Slurm submissions:"
Write-Host "  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$bundleTar"
if (-not [string]::IsNullOrWhiteSpace($ScpDestination)) {
  $remoteExample = Resolve-ScpUploadTarget -destination $ScpDestination -localFileName ([System.IO.Path]::GetFileName($bundleTar))
  Write-Host "Or use remote bundle path:"
  Write-Host "  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$($remoteExample.UploadPath)"
}
