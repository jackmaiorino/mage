param(
  [string]$OutputDir = "local-training/hpc/bundles",
  [switch]$SkipBuild,
  [string]$ScpDestination = $env:MAGE_HPC_BUNDLE_DEST,
  [bool]$ScpCreateRemoteDir = $false
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
  & tar -czf $bundleTar $bundleName
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
  if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp was not found on PATH, but ScpDestination was provided."
  }
  if ($ScpCreateRemoteDir -and -not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "ssh was not found on PATH, but ScpCreateRemoteDir=true requires ssh."
  }

  $bundleFileName = [System.IO.Path]::GetFileName($bundleTar)
  $target = Resolve-ScpUploadTarget -destination $ScpDestination -localFileName $bundleFileName

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
      & ssh $target.Host $mkdirCmd
      if ($LASTEXITCODE -ne 0) {
        throw "Failed to create remote directory via ssh (exit code $LASTEXITCODE)"
      }
    }
  }

  $uploadTarget = "$($target.Host):$($target.UploadPath)"
  Write-Host "Uploading bundle via scp:"
  Write-Host "  $uploadTarget"
  & scp $bundleTar $uploadTarget
  if ($LASTEXITCODE -ne 0) {
    throw "scp upload failed with exit code $LASTEXITCODE"
  }
  Write-Host "Uploaded bundle to:"
  Write-Host "  $uploadTarget"
}

Write-Host ""
Write-Host "Set this in Slurm submissions:"
Write-Host "  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$bundleTar"
if (-not [string]::IsNullOrWhiteSpace($ScpDestination)) {
  $remoteExample = Resolve-ScpUploadTarget -destination $ScpDestination -localFileName ([System.IO.Path]::GetFileName($bundleTar))
  Write-Host "Or use remote bundle path:"
  Write-Host "  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL=$($remoteExample.UploadPath)"
}
