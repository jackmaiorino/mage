param(
  [string]$RegistryPath = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_league_registry.json",
  [ValidateSet("default", "perf", "debug")]
  [string]$TrainProfile = "perf",
  [int]$TotalEpisodes = 1000000,
  [int]$NumGameRunners = 24,
  [string]$DefaultDeckListFile = "",
  [string]$MainClass = "mage.player.ai.rl.RLTrainer",
  [ValidateSet("", "WARNING", "INFO", "DEBUG")]
  [string]$TrainLogLevel = "",
  [int]$MetricsPortBase = 9100,
  [int]$Py4jBasePort = 0,
  [int]$Py4jPortStride = 50,
  [double]$LeaguePromoteWr = 0.55,
  [int]$EvalEveryMinutes = 180,
  [int]$InitialEvalDelayMinutes = 5,
  [int]$GamesPerDirection = 6,
  [double]$KFactor = 20.0,
  [int]$AnchorGames = 20,
  [int]$CadenceEpisodes = 5000,
  [switch]$EvalForce,
  [int]$PollSeconds = 30,
  [switch]$RunOnce,
  [switch]$DryRun,
  [switch]$NoEval,
  [switch]$LeaveRunning,
  [bool]$PreflightEmbeddings = $true,
  [string]$PythonExe = "python",
  [switch]$PreflightStrict,
  [bool]$PreStartCleanup = $true,
  [int]$CrashWindowSeconds = 90,
  [int]$RestartBackoffSeconds = 5,
  [int]$RestartBackoffMaxSeconds = 60,
  [int]$TrainerStopGraceSeconds = 10,
  [bool]$HideTrainerWindows = $true,
  [bool]$SequentialTraining = $true,
  [double]$TrainTargetWinrate = 0.60,
  [int]$TrainMinEpisodes = 2000,
  [int]$WinrateWindow = 200,
  [int]$GameLogging = -1,
  [int]$GameLogFrequency = -1,
  [int]$StatusLogEverySeconds = 60,
  [int]$StallRestartMinutes = 15,
  [double]$EpsRateHalfLifeMinutes = 5.0,
  [string]$StartProfile = "",
  [switch]$CycleProfiles,
  [bool]$EnablePbt = $false,
  [int]$PbtExploitIntervalMinutes = 120,
  [int]$PbtMinEpisodesBeforeFirstExploit = 12000,
  [int]$PbtMinEpisodeDeltaPerProfile = 10000,
  [int]$PbtTimeFallbackMinEpisodeDelta = 3000,
  [int]$PbtMinPopulationSize = 3,
  [double]$PbtMutationPct = 0.20,
  [double]$PbtMinWinnerGap = 0.03,
  [double]$PbtMinWinnerWinrate = 0.06
)

$ErrorActionPreference = "Stop"

if ($Py4jBasePort -le 0) {
  $envPy4j = [string]$env:PY4J_BASE_PORT
  $parsedPy4j = 0
  if (-not [int]::TryParse($envPy4j, [ref]$parsedPy4j) -or $parsedPy4j -le 0) {
    $parsedPy4j = 25334
  }
  $Py4jBasePort = $parsedPy4j
}
$Py4jPortStride = [Math]::Max(1, $Py4jPortStride)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trainScript = Join-Path $PSScriptRoot "rl-train.ps1"
$evalScript = Join-Path $PSScriptRoot "rl-league-eval.ps1"
$stopScriptPs1 = Join-Path $PSScriptRoot "rl-stop.ps1"
$stopScriptSh = Join-Path $PSScriptRoot "rl-stop.sh"
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
$reportsDir = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
$statusPath = Join-Path $reportsDir "orchestrator_status.json"
$historyPath = Join-Path $reportsDir "curriculum_history.csv"
$pbtStatePath = Join-Path $reportsDir "pbt_state.json"

function Resolve-RepoPath([string]$pathValue) {
  if ([string]::IsNullOrWhiteSpace($pathValue)) {
    return ""
  }
  if ([System.IO.Path]::IsPathRooted($pathValue)) {
    return (Resolve-Path -LiteralPath $pathValue).Path
  }
  $candidate = Join-Path $repoRoot $pathValue
  if (Test-Path -LiteralPath $candidate) {
    return (Resolve-Path -LiteralPath $candidate).Path
  }
  return $candidate
}

function Quote-Single([string]$s) {
  if ($null -eq $s) { return "''" }
  return "'" + ($s -replace "'", "''") + "'"
}

function ConvertTo-EnvHashtable($obj) {
  $ht = @{}
  if ($null -eq $obj) {
    return $ht
  }
  if ($obj -is [System.Collections.IDictionary]) {
    foreach ($k in $obj.Keys) {
      $name = [string]$k
      if ([string]::IsNullOrWhiteSpace($name)) { continue }
      $ht[$name] = [string]$obj[$k]
    }
    return $ht
  }
  foreach ($p in $obj.PSObject.Properties) {
    if ($null -eq $p) { continue }
    $name = [string]$p.Name
    if ([string]::IsNullOrWhiteSpace($name)) { continue }
    $ht[$name] = [string]$p.Value
  }
  return $ht
}

function ConvertTo-StringArray($obj) {
  $out = @()
  if ($null -eq $obj) { return @($out) }
  if ($obj -is [string]) {
    $v = $obj.Trim()
    if ($v -ne "") { $out += $v }
    return @($out)
  }
  foreach ($item in $obj) {
    if ($null -eq $item) { continue }
    $v = ([string]$item).Trim()
    if ($v -ne "") { $out += $v }
  }
  return @($out)
}

function Try-ParseInt64($value) {
  if ($null -eq $value) { return $null }
  try {
    return [long]$value
  } catch {
    return $null
  }
}

function Get-ProfileModelsDir([string]$profile) {
  return Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/models" -f $profile)
}

function Get-ProfileModelLatestPath([string]$profile) {
  return Join-Path (Get-ProfileModelsDir $profile) "model_latest.pt"
}

function Get-ProfileModelPath([string]$profile) {
  return Join-Path (Get-ProfileModelsDir $profile) "model.pt"
}

function Get-TrainerDbDir([string]$profile) {
  return Join-Path $repoRoot ("local-training/rl-db/{0}" -f $profile)
}

function Load-ActiveLeagueProfiles([string]$resolvedRegistryPath) {
  if (-not (Test-Path -LiteralPath $resolvedRegistryPath)) {
    throw "Registry not found: $resolvedRegistryPath"
  }
  $raw = Get-Content -LiteralPath $resolvedRegistryPath -Raw
  $entries = $raw | ConvertFrom-Json
  $active = @()
  foreach ($e in $entries) {
    $profile = [string]$e.profile
    if ([string]::IsNullOrWhiteSpace($profile)) {
      continue
    }
    $isActive = $true
    if ($null -ne $e.active) {
      $isActive = [bool]$e.active
    }
    if (-not $isActive) {
      continue
    }
    $trainDeck = ""
    if ($e.PSObject.Properties.Name -contains "train_decklist") {
      $trainDeck = [string]$e.train_decklist
    }
    $trainEnabled = $true
    if ($e.PSObject.Properties.Name -contains "train_enabled") {
      $trainEnabled = [bool]$e.train_enabled
    }
    if ($e.PSObject.Properties.Name -contains "mode") {
      $mode = ([string]$e.mode).Trim().ToLowerInvariant()
      if ($mode -eq "frozen" -or $mode -eq "eval_only") {
        $trainEnabled = $false
      }
    }
    $targetWinrate = $TrainTargetWinrate
    if ($e.PSObject.Properties.Name -contains "target_winrate") {
      try { $targetWinrate = [double]$e.target_winrate } catch { $targetWinrate = $TrainTargetWinrate }
    }
    $priority = 1000
    if ($e.PSObject.Properties.Name -contains "priority") {
      try { $priority = [int]$e.priority } catch { $priority = 1000 }
    }
    $populationGroup = ""
    if ($e.PSObject.Properties.Name -contains "population_group") {
      $populationGroup = ([string]$e.population_group).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($populationGroup)) {
      $populationGroup = $profile.Trim()
    }
    $seed = $null
    if ($e.PSObject.Properties.Name -contains "seed") {
      $seed = Try-ParseInt64 $e.seed
    }
    $trainEnv = @{}
    if ($e.PSObject.Properties.Name -contains "train_env") {
      $trainEnv = ConvertTo-EnvHashtable $e.train_env
    }
    $pbtMutableEnv = @()
    if ($e.PSObject.Properties.Name -contains "pbt_mutable_env") {
      $pbtMutableEnv = ConvertTo-StringArray $e.pbt_mutable_env
    }
    $active += [PSCustomObject]@{
      profile = $profile.Trim()
      deck_path = [string]$e.deck_path
      train_decklist = $trainDeck
      train_enabled = $trainEnabled
      target_winrate = $targetWinrate
      priority = $priority
      population_group = $populationGroup
      seed = $seed
      train_env = $trainEnv
      pbt_mutable_env = $pbtMutableEnv
    }
  }
  $ordered = @($active | Sort-Object -Property @{Expression="priority"; Ascending=$true}, @{Expression="profile"; Ascending=$true})
  return @($ordered)
}

function Resolve-TrainDeckList($entry, [string]$resolvedDefaultDeckList) {
  $candidate = ""
  if (-not [string]::IsNullOrWhiteSpace($entry.train_decklist)) {
    $candidate = Resolve-RepoPath $entry.train_decklist
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
    Write-Warning ("Profile {0}: train_decklist not found: {1}" -f $entry.profile, $candidate)
  }

  if (-not [string]::IsNullOrWhiteSpace($resolvedDefaultDeckList)) {
    if (Test-Path -LiteralPath $resolvedDefaultDeckList) {
      return $resolvedDefaultDeckList
    }
    Write-Warning ("DefaultDeckListFile not found: {0}" -f $resolvedDefaultDeckList)
  }

  $autoCandidate = Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/train_decklist.txt" -f $entry.profile)
  if (Test-Path -LiteralPath $autoCandidate) {
    return (Resolve-Path -LiteralPath $autoCandidate).Path
  }

  return ""
}

function Resolve-ProfileDeckPath($entry, [string]$resolvedRegistryPath) {
  $raw = [string]$entry.deck_path
  if ([string]::IsNullOrWhiteSpace($raw)) {
    return ""
  }

  if ([System.IO.Path]::IsPathRooted($raw)) {
    if (Test-Path -LiteralPath $raw) {
      return (Resolve-Path -LiteralPath $raw).Path
    }
    return $raw
  }

  $registryDir = Split-Path -Parent $resolvedRegistryPath
  $candidate = Join-Path $registryDir $raw
  if (Test-Path -LiteralPath $candidate) {
    return (Resolve-Path -LiteralPath $candidate).Path
  }

  $repoCandidate = Join-Path $repoRoot $raw
  if (Test-Path -LiteralPath $repoCandidate) {
    return (Resolve-Path -LiteralPath $repoCandidate).Path
  }

  return $candidate
}

function Ensure-GeneratedDeckList($entry, [string]$resolvedRegistryPath) {
  $deckPath = Resolve-ProfileDeckPath $entry $resolvedRegistryPath
  if ([string]::IsNullOrWhiteSpace($deckPath)) {
    return ""
  }
  if (-not (Test-Path -LiteralPath $deckPath)) {
    return ""
  }

  $ext = [System.IO.Path]::GetExtension($deckPath).ToLowerInvariant()
  if ($ext -eq ".txt") {
    # Already a decklist-style source
    return (Resolve-Path -LiteralPath $deckPath).Path
  }

  # For .dek/.dck, create a synthetic decklist file containing the absolute path.
  $genDir = Join-Path $reportsDir "generated_decklists"
  New-Item -ItemType Directory -Force -Path $genDir | Out-Null
  $outPath = Join-Path $genDir ("{0}.decklist.txt" -f $entry.profile)
  # Use ASCII to avoid BOM prefix (Windows UTF-8 BOM can break Java path parsing).
  Set-Content -LiteralPath $outPath -Value ((Resolve-Path -LiteralPath $deckPath).Path) -Encoding Ascii
  return $outPath
}

function Ensure-ProfileEmbeddings($entry, [string]$resolvedDeckList, [string]$resolvedRegistryPath) {
  if (-not $PreflightEmbeddings) {
    return
  }

  $profile = $entry.profile
  $modelsDir = Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/models" -f $profile)
  $embeddingPath = Join-Path $modelsDir "card_embeddings.json"
  $globalLookupPath = $env:RL_GLOBAL_EMBEDDINGS_LOOKUP_PATH
  if ([string]::IsNullOrWhiteSpace($globalLookupPath)) {
    $globalLookupPath = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/card_embeddings.global.json"
  }
  if (-not [System.IO.Path]::IsPathRooted($globalLookupPath)) {
    $globalLookupPath = Join-Path $repoRoot $globalLookupPath
  }

  if (Test-Path -LiteralPath $embeddingPath) {
    Write-Host ("Embeddings present for profile={0}: {1}" -f $profile, $embeddingPath)
    return
  }
  if (Test-Path -LiteralPath $globalLookupPath) {
    Write-Host ("Profile embedding missing for profile={0}; using global lookup: {1}" -f $profile, $globalLookupPath)
    return
  }

  $apiKey = $env:OPENAI_API_KEY
  if ([string]::IsNullOrWhiteSpace($apiKey)) {
    $msg = "Embeddings missing for profile=$profile and OPENAI_API_KEY is not set. Skipping generation."
    if ($PreflightStrict) { throw $msg } else { Write-Warning $msg; return }
  }

  $generator = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/generate_card_embeddings.py"
  if (-not (Test-Path -LiteralPath $generator)) {
    $msg = "Embedding generator not found: $generator"
    if ($PreflightStrict) { throw $msg } else { Write-Warning $msg; return }
  }

  $sourceFlag = ""
  $sourcePath = ""
  if (-not [string]::IsNullOrWhiteSpace($resolvedDeckList) -and (Test-Path -LiteralPath $resolvedDeckList)) {
    $sourceFlag = "--decklist"
    $sourcePath = $resolvedDeckList
  } else {
    $deckPath = Resolve-ProfileDeckPath $entry $resolvedRegistryPath
    if (-not [string]::IsNullOrWhiteSpace($deckPath) -and (Test-Path -LiteralPath $deckPath)) {
      $ext = [System.IO.Path]::GetExtension($deckPath).ToLowerInvariant()
      if ($ext -eq ".dck") {
        $sourceFlag = "--cube"
      } elseif ($ext -eq ".txt") {
        $sourceFlag = "--decklist"
      } else {
        $sourceFlag = "--dek"
      }
      $sourcePath = $deckPath
    }
  }

  if ([string]::IsNullOrWhiteSpace($sourceFlag) -or [string]::IsNullOrWhiteSpace($sourcePath)) {
    $msg = "No valid embedding input source found for profile=$profile (need train decklist, decklist txt, .dek or .dck)."
    if ($PreflightStrict) { throw $msg } else { Write-Warning $msg; return }
  }

  New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
  $args = @(
    $generator,
    "--profile", $profile,
    $sourceFlag, $sourcePath,
    "--output", $embeddingPath
  )

  if ($DryRun) {
    Write-Host ("[dry-run] generate embeddings profile={0}: {1} {2}" -f $profile, $PythonExe, ($args -join " "))
    return
  }

  Write-Host ("Generating embeddings for profile={0} using {1}={2}" -f $profile, $sourceFlag, $sourcePath)
  try {
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
      throw "Embedding generator exited with code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $embeddingPath)) {
      throw "Expected output not found: $embeddingPath"
    }
    Write-Host ("Embeddings generated for profile={0}: {1}" -f $profile, $embeddingPath)
  } catch {
    $msg = ("Embedding preflight failed for profile={0}: {1}" -f $profile, $_.Exception.Message)
    if ($PreflightStrict) { throw $msg } else { Write-Warning $msg }
  }
}

function Build-MetaOpponentDeckList($entries, [string]$resolvedRegistryPath) {
  if ($null -eq $entries -or $entries.Count -eq 0) {
    return ""
  }

  $paths = New-Object System.Collections.Generic.List[string]
  $seen = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($entry in $entries) {
    $deckPath = Resolve-ProfileDeckPath $entry $resolvedRegistryPath
    if ([string]::IsNullOrWhiteSpace($deckPath)) {
      continue
    }
    if (-not (Test-Path -LiteralPath $deckPath)) {
      Write-Warning ("Profile {0}: deck_path not found for meta pool: {1}" -f $entry.profile, $deckPath)
      continue
    }
    $resolved = (Resolve-Path -LiteralPath $deckPath).Path
    if ($seen.Add($resolved)) {
      [void]$paths.Add($resolved)
    }
  }

  if ($paths.Count -eq 0) {
    return ""
  }

  $genDir = Join-Path $reportsDir "generated_decklists"
  New-Item -ItemType Directory -Force -Path $genDir | Out-Null
  $outPath = Join-Path $genDir "_league_meta_opponents.decklist.txt"
  Set-Content -LiteralPath $outPath -Value $paths -Encoding Ascii
  return $outPath
}

function Start-LeagueTrainer(
  $entry,
  [string]$resolvedRegistryPath,
  [int]$metricsPort,
  [string]$resolvedAgentDeckList,
  [string]$resolvedOpponentDeckList,
  $effectiveTrainEnv = $null,
  $effectiveSeed = $null
) {
  $profile = $entry.profile
  $outDir = Join-Path $reportsDir "trainers"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $stdoutPath = Join-Path $outDir ("{0}.stdout.log" -f $profile)
  $stderrPath = Join-Path $outDir ("{0}.stderr.log" -f $profile)
  $trainerDbDir = Get-TrainerDbDir $profile
  New-Item -ItemType Directory -Force -Path $trainerDbDir | Out-Null

  if ($null -eq $effectiveTrainEnv) {
    $effectiveTrainEnv = ConvertTo-EnvHashtable $entry.train_env
  }
  if ($null -eq $effectiveSeed) {
    $effectiveSeed = Try-ParseInt64 $entry.seed
  }

  $slotIndex = [Math]::Max(0, $metricsPort - $MetricsPortBase)
  $stride = [Math]::Max(1, $Py4jPortStride)
  $py4jBaseForTrainer = [int]($Py4jBasePort + ($slotIndex * $stride))

  $envSetters = @(
    ('$env:OPPONENT_SAMPLER=' + (Quote-Single "league")),
    ('$env:ORCHESTRATED_RUN=' + (Quote-Single "1")),
    ('$env:LEAGUE_REGISTRY_PATH=' + (Quote-Single $resolvedRegistryPath)),
    ('$env:LEAGUE_PROMOTE_WR=' + (Quote-Single ([string]$LeaguePromoteWr))),
    ('$env:PY4J_BASE_PORT=' + (Quote-Single ([string]$py4jBaseForTrainer))),
    # Per-trainer bridge startup cleanup is unsafe in concurrent orchestrator mode
    # (it can kill sibling trainers' Py4J workers). Orchestrator-level cleanup
    # handles stale-process teardown before launch/restart.
    ('$env:PY_BRIDGE_CLEANUP=' + (Quote-Single "0")),
    ('$env:MAGE_DB_DIR=' + (Quote-Single $trainerDbDir)),
    ('$env:MAGE_DB_AUTO_SERVER=' + (Quote-Single "false"))
  )
  if (-not [string]::IsNullOrWhiteSpace($resolvedAgentDeckList)) {
    $envSetters += ('$env:RL_AGENT_DECK_LIST=' + (Quote-Single $resolvedAgentDeckList))
  }
  if (-not [string]::IsNullOrWhiteSpace($TrainLogLevel)) {
    $envSetters += ('$env:MTG_AI_LOG_LEVEL=' + (Quote-Single $TrainLogLevel))
  }
  if ($effectiveSeed -ne $null) {
    $seedStr = [string]$effectiveSeed
    $envSetters += ('$env:RL_BASE_SEED=' + (Quote-Single $seedStr))
    $envSetters += ('$env:PY_GLOBAL_SEED=' + (Quote-Single $seedStr))
    $envSetters += ('$env:MULLIGAN_REPLAY_SEED=' + (Quote-Single $seedStr))
  }
  if ($null -ne $effectiveTrainEnv) {
    foreach ($k in @($effectiveTrainEnv.Keys)) {
      $name = [string]$k
      if ([string]::IsNullOrWhiteSpace($name)) { continue }
      $envSetters += ('$env:' + $name + '=' + (Quote-Single ([string]$effectiveTrainEnv[$k])))
    }
  }

  $invoke = "& " + (Quote-Single $trainScript) +
    " -Profile " + (Quote-Single $TrainProfile) +
    " -TotalEpisodes " + $TotalEpisodes +
    " -NumGameRunners " + $NumGameRunners +
    " -ModelProfile " + (Quote-Single $profile) +
    " -MetricsPort " + $metricsPort +
    " -MainClass " + (Quote-Single $MainClass)

  if (-not [string]::IsNullOrWhiteSpace($resolvedOpponentDeckList)) {
    $invoke += " -DeckListFile " + (Quote-Single $resolvedOpponentDeckList)
  }
  if (-not [string]::IsNullOrWhiteSpace($TrainLogLevel)) {
    $invoke += " -LogLevel " + (Quote-Single $TrainLogLevel)
  }
  if ($GameLogging -ge 0) {
    $invoke += " -GameLogging " + $GameLogging
  }
  if ($GameLogFrequency -gt 0) {
    $invoke += " -GameLogFrequency " + $GameLogFrequency
  }

  $command = "& { " + ($envSetters -join "; ") + "; " + $invoke + " }"
  $argList = @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command)

  if ($DryRun) {
    Write-Host ("[dry-run] trainer {0} command: {1}" -f $profile, $command)
    return $null
  }

  $startArgs = @{
    FilePath = $shellExecutable
    ArgumentList = $argList
    WorkingDirectory = $repoRoot
    PassThru = $true
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
  }
  if ($HideTrainerWindows -and $isWindowsHost) {
    $startArgs.WindowStyle = "Hidden"
  }
  $proc = Start-Process @startArgs
  Write-Host ("Started trainer profile={0} pid={1} metricsPort={2} py4jBasePort={3}" -f $profile, $proc.Id, $metricsPort, $py4jBaseForTrainer)
  return $proc
}

function Sync-ProfileModelLatest([string]$profile) {
  if ([string]::IsNullOrWhiteSpace($profile)) {
    return
  }
  try {
    $latest = Get-ProfileModelLatestPath $profile
    $model = Get-ProfileModelPath $profile
    if (-not (Test-Path -LiteralPath $latest)) {
      return
    }
    $targetDir = Split-Path -Parent $model
    if (-not [string]::IsNullOrWhiteSpace($targetDir)) {
      New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }
    Copy-Item -LiteralPath $latest -Destination $model -Force
    Write-Host ("Synced latest model for profile={0}" -f $profile)
  } catch {
    Write-Warning ("Failed syncing model_latest->model for profile={0}: {1}" -f $profile, $_.Exception.Message)
  }
}

function Stop-TrainerProcess([string]$profile, $proc) {
  if ($null -eq $proc) { return }
  try {
    if ($proc.HasExited) {
      return
    }
  } catch {
    return
  }

  $procId = 0
  try { $procId = $proc.Id } catch { $procId = 0 }
  Write-Host ("Stopping trainer profile={0} pid={1}" -f $profile, $procId)

  $graceSec = [Math]::Max(0, $TrainerStopGraceSeconds)
  $requested = $false

  try {
    # First try non-force termination and allow a brief grace window.
    Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
    $requested = $true
  } catch {
    $requested = $false
  }

  if ($requested -and $graceSec -gt 0) {
    $deadline = (Get-Date).AddSeconds($graceSec)
    while ((Get-Date) -lt $deadline) {
      $exited = $false
      try {
        $exited = $proc.HasExited
      } catch {
        $exited = $true
      }
      if ($exited) {
        return
      }
      Start-Sleep -Milliseconds 200
    }
  }

  try {
    if (-not $proc.HasExited) {
      Write-Warning ("Trainer still alive after grace window; force killing profile={0} pid={1}" -f $profile, $procId)
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
  } catch {
    Write-Warning ("Failed stopping profile={0}: {1}" -f $profile, $_.Exception.Message)
  }
}

function Stop-LeagueTrainers($stateMap) {
  foreach ($k in @($stateMap.Keys)) {
    $item = $stateMap[$k]
    if ($null -eq $item) { continue }
    $proc = $item.process
    if ($null -eq $proc) { continue }
    Sync-ProfileModelLatest $k
    Stop-TrainerProcess $k $proc
    Sync-ProfileModelLatest $k
  }
}

function Stop-OneTrainer($stateMap, [string]$profile) {
  if ($null -eq $stateMap -or [string]::IsNullOrWhiteSpace($profile)) {
    return
  }
  if (-not $stateMap.ContainsKey($profile)) {
    return
  }
  $item = $stateMap[$profile]
  if ($null -eq $item) { return }
  $proc = $item.process
  if ($null -eq $proc) { return }
  Sync-ProfileModelLatest $profile
  Stop-TrainerProcess $profile $proc
  Sync-ProfileModelLatest $profile
}

function Write-OrchestratorStatus($stateMap, $profiles, $resolvedRegistryPath, $lastEvalAt, $nextEvalAt, [string]$note, [string]$currentProfile, $snapshotMap, $completedMap) {
  New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
  $items = @()
  foreach ($p in $profiles) {
    $name = $p.profile
    $st = $stateMap[$name]
    $proc = $null
    if ($null -ne $st) { $proc = $st.process }
    $running = $false
    $procId = 0
    $exitCode = $null
    if ($null -ne $proc) {
      try {
        $running = -not $proc.HasExited
        $procId = $proc.Id
        if (-not $running) { $exitCode = $proc.ExitCode }
      } catch {
      }
    }
    $items += [PSCustomObject]@{
      profile = $name
      running = $running
      pid = $procId
      restart_count = if ($null -eq $st) { 0 } else { $st.restart_count }
      consecutive_failures = if ($null -eq $st) { 0 } else { $st.consecutive_failures }
      last_restart_reason = if ($null -eq $st) { "" } else { [string]$st.last_restart_reason }
      launched_at_utc = if ($null -eq $st) { "" } else { $st.launched_at_utc }
      metrics_port = if ($null -eq $st) { 0 } else { $st.metrics_port }
      train_decklist = if ($null -eq $st) { "" } else { $st.train_decklist }
      opponent_decklist = if ($null -eq $st) { "" } else { $st.opponent_decklist }
      stdout_log = if ($null -eq $st) { "" } else { $st.stdout_log }
      stderr_log = if ($null -eq $st) { "" } else { $st.stderr_log }
      exit_code = $exitCode
    }
  }

  $snapshots = @()
  if ($null -ne $snapshotMap) {
    foreach ($k in @($snapshotMap.Keys)) {
      $s = $snapshotMap[$k]
      if ($null -eq $s) { continue }
      $snapshots += [PSCustomObject]@{
        profile = $s.profile
        episode = $s.episode
        rolling_winrate = $s.rolling_current
        rolling_avg = $s.rolling_avg
        baseline_wr = $s.baseline_wr
        target_winrate = $s.target_winrate
        promoted = $s.promoted
        train_enabled = $s.train_enabled
        completed = if ($null -eq $completedMap) { $false } else { [bool]$completedMap[$s.profile] }
      }
    }
  }

  $obj = [ordered]@{
    updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    registry_path = $resolvedRegistryPath
    eval_every_minutes = $EvalEveryMinutes
    last_eval_utc = if ($null -eq $lastEvalAt) { "" } else { $lastEvalAt.ToUniversalTime().ToString("o") }
    next_eval_utc = if ($null -eq $nextEvalAt) { "" } else { $nextEvalAt.ToUniversalTime().ToString("o") }
    dry_run = [bool]$DryRun
    sequential_training = [bool]$SequentialTraining
    current_training_profile = if ([string]::IsNullOrWhiteSpace($currentProfile)) { "" } else { $currentProfile }
    note = $note
    trainers = $items
    profile_snapshots = $snapshots
  }
  $obj | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Run-LeagueEval([string]$resolvedRegistryPath) {
  if ($NoEval) {
    return
  }
  if ($EvalEveryMinutes -le 0 -and -not $RunOnce) {
    return
  }
  if ($DryRun) {
    Write-Host ("[dry-run] eval every {0}m via {1}" -f $EvalEveryMinutes, $evalScript)
    return
  }
  Write-Host ("Running league eval at {0}" -f (Get-Date))
  $params = @{
    RegistryPath = $resolvedRegistryPath
    GamesPerDirection = $GamesPerDirection
    KFactor = $KFactor
    AnchorGames = $AnchorGames
    CadenceEpisodes = $CadenceEpisodes
  }
  if ($EvalForce) {
    $params["Force"] = $true
  }
  & $evalScript @params
}

function Get-ProfileTrainingSnapshot($entry) {
  $profile = $entry.profile
  $statsPath = Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/logs/stats/training_stats.csv" -f $profile)
  $statusPath = Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/logs/league/agent_status.json" -f $profile)

  $episode = 0
  $baselineWr = 0.0
  $promoted = $false
  if (Test-Path -LiteralPath $statusPath) {
    try {
      $statusObj = (Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json)
      if ($null -ne $statusObj.episode) { $episode = [int]$statusObj.episode }
      if ($null -ne $statusObj.baseline_wr) { $baselineWr = [double]$statusObj.baseline_wr }
      if ($null -ne $statusObj.promoted) { $promoted = [bool]$statusObj.promoted }
    } catch {
    }
  }

  $rollingCurrent = $null
  $rollingAvg = $null
  $samples = 0
  if (Test-Path -LiteralPath $statsPath) {
    try {
      $tail = Get-Content -LiteralPath $statsPath -Tail ([Math]::Max(50, $WinrateWindow))
      $vals = New-Object System.Collections.Generic.List[Double]
      foreach ($line in $tail) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.StartsWith("episode,")) { continue }
        $parts = $line.Split(',')
        if ($parts.Length -lt 5) { continue }
        $episodeText = $parts[0].Trim()
        $episodeCandidate = 0
        if ([int]::TryParse($episodeText, [ref]$episodeCandidate)) {
          if ($episodeCandidate -gt $episode) {
            $episode = $episodeCandidate
          }
        }
        $wrText = $parts[4].Trim()
        $wr = 0.0
        if ([double]::TryParse($wrText, [ref]$wr)) {
          $vals.Add($wr)
        }
      }
      if ($vals.Count -gt 0) {
        $samples = $vals.Count
        $rollingCurrent = $vals[$vals.Count - 1]
        $sum = 0.0
        foreach ($v in $vals) { $sum += $v }
        $rollingAvg = $sum / [double]$vals.Count
      }
    } catch {
    }
  }

  return [PSCustomObject]@{
    profile = $profile
    episode = $episode
    baseline_wr = $baselineWr
    promoted = $promoted
    rolling_current = $rollingCurrent
    rolling_avg = $rollingAvg
    sample_count = $samples
    target_winrate = [double]$entry.target_winrate
    train_enabled = [bool]$entry.train_enabled
  }
}

function Profile-MeetsGoal($snapshot) {
  if ($null -eq $snapshot) { return $false }
  if ($snapshot.episode -lt [Math]::Max(1, $TrainMinEpisodes)) { return $false }
  if ($null -eq $snapshot.rolling_current) { return $false }
  return ([double]$snapshot.rolling_current -ge [double]$snapshot.target_winrate)
}

function Select-NextTrainingProfile($entries, $completedMap) {
  $eligible = @()
  foreach ($e in $entries) {
    if (-not [bool]$e.train_enabled) { continue }
    if ($completedMap.ContainsKey($e.profile) -and [bool]$completedMap[$e.profile]) { continue }
    $eligible += $e
  }
  if ($eligible.Count -eq 0) { return $null }
  return $eligible[0]
}

function Append-CurriculumEvent([string]$eventName, $snapshot) {
  try {
    if ($null -eq $snapshot) { return }
    if ($historyPath) {
      $parent = Split-Path -Parent $historyPath
      if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
      }
      if (-not (Test-Path -LiteralPath $historyPath)) {
        "timestamp_utc,event,profile,episode,rolling_winrate,rolling_avg,baseline_wr,target_winrate,samples,promoted" | Set-Content -LiteralPath $historyPath -Encoding Ascii
      }
      $rollingCurrentText = ""
      if ($null -ne $snapshot.rolling_current) {
        $rollingCurrentText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F4}", [double]$snapshot.rolling_current)
      }
      $rollingAvgText = ""
      if ($null -ne $snapshot.rolling_avg) {
        $rollingAvgText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F4}", [double]$snapshot.rolling_avg)
      }
      $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
        (Get-Date).ToUniversalTime().ToString("o"), `
        $eventName, `
        $snapshot.profile, `
        $snapshot.episode, `
        $rollingCurrentText, `
        $rollingAvgText, `
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F4}", [double]$snapshot.baseline_wr), `
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F4}", [double]$snapshot.target_winrate), `
        $snapshot.sample_count, `
        $snapshot.promoted
      Add-Content -LiteralPath $historyPath -Value $row -Encoding Ascii
    }
  } catch {
    Write-Warning ("Failed writing curriculum history: {0}" -f $_.Exception.Message)
  }
}

function Invoke-OrchestratorCleanup([string]$reason) {
  if (-not $PreStartCleanup) {
    return
  }
  if ($DryRun) {
    Write-Host ("[dry-run] cleanup via {0} reason={1}" -f $stopScript, $reason)
    return
  }
  if (-not (Test-Path -LiteralPath $stopScript)) {
    Write-Warning ("Cleanup requested but stop script missing: {0}" -f $stopScript)
    return
  }
  try {
    Write-Host ("Running cleanup via {0} (reason={1})" -f $stopScript, $reason)
    if ([string]$stopScript -match "\.sh$") {
      & bash $stopScript -q
    } else {
      & $stopScript -Quiet
    }
  } catch {
    Write-Warning ("Cleanup failed: {0}" -f $_.Exception.Message)
  }
}

function Mutate-EnvValue([string]$key, [string]$value) {
  $numeric = 0.0
  $ok = [double]::TryParse(
    [string]$value,
    [System.Globalization.NumberStyles]::Float,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [ref]$numeric
  )
  if (-not $ok) {
    return [string]$value
  }
  $pct = [Math]::Max(0.0, [double]$PbtMutationPct)
  $u = (Get-Random -Minimum 0.0 -Maximum 1.0)
  $factor = 1.0 + ((2.0 * $u - 1.0) * $pct)
  if ($factor -lt 0.01) { $factor = 0.01 }
  $mutated = $numeric * $factor
  $k = if ($null -eq $key) { "" } else { $key.ToUpperInvariant() }
  if ($k -match "EPS|PROB|RATE|FRAC|_P$|^P_") {
    if ($mutated -lt 0.0) { $mutated = 0.0 }
    if ($mutated -gt 1.0) { $mutated = 1.0 }
  }
  return [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:G}", $mutated)
}

function Copy-ProfileLatestModel([string]$sourceProfile, [string]$targetProfile) {
  $sourceLatest = Get-ProfileModelLatestPath $sourceProfile
  $sourceModel = Get-ProfileModelPath $sourceProfile
  if (-not (Test-Path -LiteralPath $sourceLatest)) {
    if (Test-Path -LiteralPath $sourceModel) {
      $sourceLatest = $sourceModel
    } else {
      return $false
    }
  }
  $targetLatest = Get-ProfileModelLatestPath $targetProfile
  $targetModel = Get-ProfileModelPath $targetProfile
  $targetDir = Split-Path -Parent $targetLatest
  if (-not [string]::IsNullOrWhiteSpace($targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  }
  Copy-Item -LiteralPath $sourceLatest -Destination $targetLatest -Force
  Copy-Item -LiteralPath $targetLatest -Destination $targetModel -Force
  return $true
}

function Write-PbtState($stateMap, $profiles, [datetime]$lastPbtAt, $events, $groupStateMap) {
  try {
    New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
    $profileRows = @()
    foreach ($entry in $profiles) {
      $profile = [string]$entry.profile
      $state = $stateMap[$profile]
      $seed = if ($null -eq $state) { Try-ParseInt64 $entry.seed } else { Try-ParseInt64 $state.effective_seed }
      $envObj = if ($null -eq $state) { ConvertTo-EnvHashtable $entry.train_env } else { ConvertTo-EnvHashtable $state.effective_train_env }
      $profileRows += [PSCustomObject]@{
        profile = $profile
        population_group = [string]$entry.population_group
        seed = $seed
        pbt_mutable_env = @(ConvertTo-StringArray $entry.pbt_mutable_env)
        train_env = $envObj
      }
    }
    $eventRows = @()
    if ($null -ne $events) {
      $all = @($events)
      $start = [Math]::Max(0, $all.Count - 200)
      for ($i = $start; $i -lt $all.Count; $i++) {
        $eventRows += $all[$i]
      }
    }
    $groupRows = @()
    if ($null -ne $groupStateMap) {
      foreach ($groupKey in @($groupStateMap.Keys)) {
        $st = $groupStateMap[$groupKey]
        if ($null -eq $st) { continue }
        $groupRows += [PSCustomObject]@{
          population_group = [string]$groupKey
          last_exploit_utc = [string]$st.last_exploit_utc
          last_exploit_min_episode = if ($null -eq $st.last_exploit_min_episode) { 0 } else { [int]$st.last_exploit_min_episode }
          exploit_count = if ($null -eq $st.exploit_count) { 0 } else { [int]$st.exploit_count }
        }
      }
    }
    $obj = [ordered]@{
      updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
      enable_pbt = [bool]$EnablePbt
      pbt_exploit_max_interval_minutes = [int]$PbtExploitIntervalMinutes
      pbt_min_episodes_before_first_exploit = [int]$PbtMinEpisodesBeforeFirstExploit
      pbt_min_episode_delta_per_profile = [int]$PbtMinEpisodeDeltaPerProfile
      pbt_time_fallback_min_episode_delta = [int]$PbtTimeFallbackMinEpisodeDelta
      pbt_mutation_pct = [double]$PbtMutationPct
      pbt_min_winner_gap = [double]$PbtMinWinnerGap
      pbt_min_winner_winrate = [double]$PbtMinWinnerWinrate
      last_exploit_utc = if ($lastPbtAt -eq [datetime]::MinValue) { "" } else { $lastPbtAt.ToUniversalTime().ToString("o") }
      profiles = $profileRows
      group_state = $groupRows
      events = $eventRows
    }
    $obj | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $pbtStatePath -Encoding UTF8
  } catch {
    Write-Warning ("Failed writing PBT state: {0}" -f $_.Exception.Message)
  }
}

function Invoke-PbtExploit(
  $stateMap,
  $profiles,
  $snapshotMap,
  [string]$resolvedRegistryPath,
  [string]$fallbackOpponentDeckList,
  $groupStateMap,
  [datetime]$nowUtc
) {
  $events = @()
  if (-not $EnablePbt) {
    return @($events)
  }
  if ($null -eq $groupStateMap) {
    $groupStateMap = @{}
  }

  $groupMap = @{}
  foreach ($entry in $profiles) {
    if (-not [bool]$entry.train_enabled) { continue }
    $group = ([string]$entry.population_group).Trim()
    if ([string]::IsNullOrWhiteSpace($group)) {
      $group = [string]$entry.profile
    }
    if (-not $groupMap.ContainsKey($group)) {
      $groupMap[$group] = @()
    }
    $groupMap[$group] += $entry
  }

  foreach ($group in @($groupMap.Keys)) {
    $entries = @($groupMap[$group])
    if ($entries.Count -lt [Math]::Max(2, $PbtMinPopulationSize)) {
      continue
    }

    $groupEpisodes = @()
    $episodesReady = $true
    foreach ($entry in $entries) {
      $snap = $snapshotMap[[string]$entry.profile]
      if ($null -eq $snap -or $null -eq $snap.episode) {
        $episodesReady = $false
        break
      }
      $groupEpisodes += [int]$snap.episode
    }
    if (-not $episodesReady -or $groupEpisodes.Count -lt $entries.Count) {
      continue
    }
    $groupMinEpisode = [int](($groupEpisodes | Measure-Object -Minimum).Minimum)

    if (-not $groupStateMap.ContainsKey($group)) {
      $groupStateMap[$group] = [PSCustomObject]@{
        last_exploit_utc = ""
        last_exploit_min_episode = 0
        exploit_count = 0
      }
    }
    $groupState = $groupStateMap[$group]
    $exploitCount = if ($null -eq $groupState.exploit_count) { 0 } else { [int]$groupState.exploit_count }
    $lastExploitMinEpisode = if ($null -eq $groupState.last_exploit_min_episode) { 0 } else { [int]$groupState.last_exploit_min_episode }
    $lastExploitAtUtc = [datetime]::MinValue
    try {
      if (-not [string]::IsNullOrWhiteSpace([string]$groupState.last_exploit_utc)) {
        $lastExploitAtUtc = [datetime]::Parse([string]$groupState.last_exploit_utc).ToUniversalTime()
      }
    } catch {
      $lastExploitAtUtc = [datetime]::MinValue
    }

    $firstExploitGate = [Math]::Max(0, [int]$PbtMinEpisodesBeforeFirstExploit)
    $deltaGate = [Math]::Max(1, [int]$PbtMinEpisodeDeltaPerProfile)
    $timeDeltaGate = [Math]::Max(1, [int]$PbtTimeFallbackMinEpisodeDelta)
    $episodeDeltaSinceLast = ($groupMinEpisode - $lastExploitMinEpisode)
    $dueByEpisode = $false
    if ($exploitCount -le 0) {
      $dueByEpisode = ($groupMinEpisode -ge $firstExploitGate)
    } else {
      $dueByEpisode = ($episodeDeltaSinceLast -ge $deltaGate)
    }

    $dueByTime = $false
    $timeDeltaGatePassed = $false
    $elapsedMinutes = 0.0
    if ($PbtExploitIntervalMinutes -gt 0) {
      if ($lastExploitAtUtc -eq [datetime]::MinValue) {
        $elapsedMinutes = [double]::PositiveInfinity
      } else {
        $elapsedMinutes = ($nowUtc - $lastExploitAtUtc).TotalMinutes
      }
      $timeDeltaGatePassed = ($episodeDeltaSinceLast -ge $timeDeltaGate)
      $dueByTime = ($groupMinEpisode -ge $firstExploitGate) `
        -and ($elapsedMinutes -ge [Math]::Max(1, $PbtExploitIntervalMinutes)) `
        -and $timeDeltaGatePassed
    }

    if (-not ($dueByEpisode -or $dueByTime)) {
      continue
    }
    $trigger = if ($dueByEpisode) { "episode" } else { "time" }

    $candidates = @()
    foreach ($entry in $entries) {
      $profile = [string]$entry.profile
      if (-not $stateMap.ContainsKey($profile)) { continue }
      $snap = $snapshotMap[$profile]
      if ($null -eq $snap -or $null -eq $snap.rolling_current) { continue }
      $candidates += [PSCustomObject]@{
        entry = $entry
        snapshot = $snap
      }
    }
    if ($candidates.Count -lt [Math]::Max(2, $PbtMinPopulationSize)) {
      continue
    }
    $ordered = @($candidates | Sort-Object -Property @{Expression={ [double]$_.snapshot.rolling_current }; Descending=$true}, @{Expression={ $_.entry.profile }; Ascending=$true})
    if ($ordered.Count -lt 2) { continue }
    $winner = $ordered[0]
    $loserCount = [Math]::Max(1, [int][Math]::Floor($ordered.Count / 2))
    $losers = @($ordered | Select-Object -Last $loserCount)
    $didExploitGroup = $false
    foreach ($loser in $losers) {
      $winnerProfile = [string]$winner.entry.profile
      $loserProfile = [string]$loser.entry.profile
      if ($winnerProfile -eq $loserProfile) { continue }
      $state = $stateMap[$loserProfile]
      if ($null -eq $state) { continue }
      $winnerWr = [double]$winner.snapshot.rolling_current
      $loserWr = [double]$loser.snapshot.rolling_current
      $winnerGap = ($winnerWr - $loserWr)
      $gapGatePassed = ($winnerGap -ge [double]$PbtMinWinnerGap)
      $winnerWrGatePassed = ($winnerWr -ge [double]$PbtMinWinnerWinrate)
      if (-not ($gapGatePassed -and $winnerWrGatePassed)) {
        $skipReason = if (-not $gapGatePassed -and -not $winnerWrGatePassed) {
          "gap_and_winner_wr_below_threshold"
        } elseif (-not $gapGatePassed) {
          "gap_below_threshold"
        } else {
          "winner_wr_below_threshold"
        }
        Write-Host ("PBT exploit skipped group={0} winner={1} loser={2} reason={3} winner_wr={4:N3} loser_wr={5:N3} gap={6:N3} min_gap={7:N3} min_winner_wr={8:N3} trigger={9} groupMinEp={10} deltaEp={11} timeDeltaGate={12} timeDeltaPassed={13}" -f `
          $group, $winnerProfile, $loserProfile, $skipReason, $winnerWr, $loserWr, $winnerGap, [double]$PbtMinWinnerGap, [double]$PbtMinWinnerWinrate, $trigger, [int]$groupMinEpisode, [int]$episodeDeltaSinceLast, [int]$timeDeltaGate, $timeDeltaGatePassed)
        $events += [PSCustomObject]@{
          timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
          population_group = $group
          trigger = $trigger
          group_min_episode = $groupMinEpisode
          episode_delta = $episodeDeltaSinceLast
          elapsed_minutes = if ([double]::IsInfinity($elapsedMinutes)) { -1.0 } else { [double]$elapsedMinutes }
          winner = $winnerProfile
          loser = $loserProfile
          winner_wr = $winnerWr
          loser_wr = $loserWr
          winner_gap = $winnerGap
          winner_gap_min = [double]$PbtMinWinnerGap
          winner_wr_min_gate = [double]$PbtMinWinnerWinrate
          gap_gate_passed = $gapGatePassed
          winner_wr_gate_passed = $winnerWrGatePassed
          time_delta_gate = [int]$timeDeltaGate
          time_delta_gate_passed = $timeDeltaGatePassed
          skip_reason = $skipReason
          copied = $false
          new_seed = ""
          mutated_keys = ""
        }
        continue
      }

      $copyOk = $false
      if ($DryRun) {
        Write-Host ("[dry-run] PBT copy winner={0} -> loser={1}" -f $winnerProfile, $loserProfile)
        $copyOk = $true
      } else {
        try {
          $copyOk = Copy-ProfileLatestModel $winnerProfile $loserProfile
        } catch {
          $copyOk = $false
          Write-Warning ("PBT copy failed winner={0} loser={1}: {2}" -f $winnerProfile, $loserProfile, $_.Exception.Message)
        }
      }

      $effectiveEnv = ConvertTo-EnvHashtable $state.effective_train_env
      if ($effectiveEnv.Count -eq 0) {
        $effectiveEnv = ConvertTo-EnvHashtable $loser.entry.train_env
      }
      $mutableKeys = @(ConvertTo-StringArray $loser.entry.pbt_mutable_env)
      foreach ($envKey in $mutableKeys) {
        if ($effectiveEnv.ContainsKey($envKey)) {
          $effectiveEnv[$envKey] = Mutate-EnvValue $envKey ([string]$effectiveEnv[$envKey])
        }
      }
      $seedNow = Try-ParseInt64 $state.effective_seed
      if ($seedNow -eq $null) { $seedNow = Try-ParseInt64 $loser.entry.seed }
      if ($seedNow -eq $null) { $seedNow = 1L }
      $seedNow = [long]$seedNow + [long](Get-Random -Minimum 1 -Maximum 100000)
      $state.effective_seed = $seedNow
      $state.effective_train_env = $effectiveEnv

      $oppDeckList = [string]$state.opponent_decklist
      if ([string]::IsNullOrWhiteSpace($oppDeckList)) {
        $oppDeckList = $fallbackOpponentDeckList
      }

      if ($DryRun) {
        Write-Host ("[dry-run] PBT restart loser={0} seed={1} mutable=[{2}]" -f $loserProfile, $seedNow, ($mutableKeys -join ","))
      } else {
        Stop-OneTrainer $stateMap $loserProfile
        Start-Sleep -Milliseconds 250
        $newProc = Start-LeagueTrainer $loser.entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList $state.effective_train_env $state.effective_seed
        $state.process = $newProc
        $state.launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        $state.last_restart_reason = "pbt_replace"
      }

      $events += [PSCustomObject]@{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        population_group = $group
        trigger = $trigger
        group_min_episode = $groupMinEpisode
        episode_delta = $episodeDeltaSinceLast
        elapsed_minutes = if ([double]::IsInfinity($elapsedMinutes)) { -1.0 } else { [double]$elapsedMinutes }
        winner = $winnerProfile
        loser = $loserProfile
        winner_wr = $winnerWr
        loser_wr = $loserWr
        winner_gap = $winnerGap
        winner_gap_min = [double]$PbtMinWinnerGap
        winner_wr_min_gate = [double]$PbtMinWinnerWinrate
        gap_gate_passed = $gapGatePassed
        winner_wr_gate_passed = $winnerWrGatePassed
        time_delta_gate = [int]$timeDeltaGate
        time_delta_gate_passed = $timeDeltaGatePassed
        skip_reason = ""
        copied = [bool]$copyOk
        new_seed = [string]$seedNow
        mutated_keys = ($mutableKeys -join ";")
      }
      $didExploitGroup = $true
    }
    if ($didExploitGroup) {
      $groupState.last_exploit_utc = $nowUtc.ToString("o")
      $groupState.last_exploit_min_episode = $groupMinEpisode
      $groupState.exploit_count = $exploitCount + 1
      $groupStateMap[$group] = $groupState
    }
  }

  return @($events)
}

$resolvedRegistryPath = Resolve-RepoPath $RegistryPath
$resolvedDefaultDeckList = Resolve-RepoPath $DefaultDeckListFile

if (-not (Test-Path -LiteralPath $trainScript)) {
  throw "Missing train script: $trainScript"
}
if (-not (Test-Path -LiteralPath $evalScript)) {
  throw "Missing eval script: $evalScript"
}

$activeProfiles = @(Load-ActiveLeagueProfiles $resolvedRegistryPath)
if ($activeProfiles.Count -eq 0) {
  throw "No active profiles in registry: $resolvedRegistryPath"
}
if (-not [string]::IsNullOrWhiteSpace($StartProfile)) {
  $start = $StartProfile.Trim()
  $first = $activeProfiles | Where-Object { $_.profile -eq $start } | Select-Object -First 1
  if ($null -eq $first) {
    Write-Warning ("StartProfile '{0}' not found among active profiles; using registry priority order." -f $start)
  } else {
    $rest = @($activeProfiles | Where-Object { $_.profile -ne $first.profile })
    $activeProfiles = @($first) + $rest
    Write-Host ("StartProfile override: {0}" -f $first.profile)
  }
}

$resolvedDeckLists = @{}
foreach ($entry in $activeProfiles) {
  $deckList = Resolve-TrainDeckList $entry $resolvedDefaultDeckList
  if ([string]::IsNullOrWhiteSpace($deckList)) {
    $deckList = Ensure-GeneratedDeckList $entry $resolvedRegistryPath
  }
  if ([string]::IsNullOrWhiteSpace($deckList)) {
    $repoDefault = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
    if (Test-Path -LiteralPath $repoDefault) {
      $deckList = (Resolve-Path -LiteralPath $repoDefault).Path
    }
  }
  if ([string]::IsNullOrWhiteSpace($deckList)) {
    throw ("No usable decklist found for profile={0}. Set train_decklist in registry or pass -DefaultDeckListFile." -f $entry.profile)
  }
  $resolvedDeckLists[$entry.profile] = $deckList
  Ensure-ProfileEmbeddings $entry $deckList $resolvedRegistryPath
}
$resolvedMetaOpponentDeckList = Build-MetaOpponentDeckList $activeProfiles $resolvedRegistryPath
if ([string]::IsNullOrWhiteSpace($resolvedMetaOpponentDeckList)) {
  throw "No usable meta opponent decklist entries found in registry."
}
Write-Host ("Meta opponent decklist: {0}" -f $resolvedMetaOpponentDeckList)

Write-Host ("League orchestrator starting. profiles={0} evalEveryMinutes={1} runOnce={2} dryRun={3}" -f `
  $activeProfiles.Count, $EvalEveryMinutes, $RunOnce.IsPresent, $DryRun.IsPresent)
Write-Host ("Registry: {0}" -f $resolvedRegistryPath)
Write-Host ("PBT gating: firstExploitMinEp={0} deltaEpPerProfile={1} timeFallbackMinDeltaEp={2} maxTimeFallbackMin={3} minWinnerGap={4:N3} minWinnerWr={5:N3}" -f `
  [int]$PbtMinEpisodesBeforeFirstExploit, [int]$PbtMinEpisodeDeltaPerProfile, [int]$PbtTimeFallbackMinEpisodeDelta, [int]$PbtExploitIntervalMinutes, [double]$PbtMinWinnerGap, [double]$PbtMinWinnerWinrate)

$trainerState = @{}
$completedProfiles = @{}
$snapshotMap = @{}
$episodeRateState = @{}
$episodeWatermark = @{}
$currentTrainingProfile = ""
$lastStatusLogAt = [datetime]::MinValue
$lastEvalAt = $null
$nextEvalAt = if ($RunOnce) { Get-Date } else { (Get-Date).AddMinutes([Math]::Max(0, $InitialEvalDelayMinutes)) }
$lastPbtAt = [datetime]::MinValue
$pbtEvents = New-Object System.Collections.ArrayList
$pbtGroupState = @{}

try {
  Invoke-OrchestratorCleanup "orchestrator_start"

  while ($true) {
    foreach ($entry in $activeProfiles) {
      $snapshotMap[$entry.profile] = Get-ProfileTrainingSnapshot $entry
    }

    if ($SequentialTraining) {
      # Mark profiles complete when they hit target (one-way unless CycleProfiles).
      foreach ($entry in $activeProfiles) {
        if (-not [bool]$entry.train_enabled) {
          continue
        }
        if ($completedProfiles.ContainsKey($entry.profile) -and [bool]$completedProfiles[$entry.profile]) {
          continue
        }
        $snap = $snapshotMap[$entry.profile]
        if (Profile-MeetsGoal $snap) {
          $completedProfiles[$entry.profile] = $true
          Append-CurriculumEvent "goal_reached" $snap
          if ($entry.profile -ne $currentTrainingProfile) {
            Write-Host ("Profile {0} already meets goal (rolling={1:N3}, target={2:N3}, episode={3})" -f `
              $entry.profile, [double]$snap.rolling_current, [double]$snap.target_winrate, [int]$snap.episode)
          }
        }
      }

      if (-not [string]::IsNullOrWhiteSpace($currentTrainingProfile)) {
        $currentSnap = $snapshotMap[$currentTrainingProfile]
        if (Profile-MeetsGoal $currentSnap) {
          Write-Host ("Completed profile={0}: rolling={1:N3} target={2:N3} episode={3}" -f `
            $currentTrainingProfile, [double]$currentSnap.rolling_current, [double]$currentSnap.target_winrate, [int]$currentSnap.episode)
          Append-CurriculumEvent "training_complete" $currentSnap
          $completedProfiles[$currentTrainingProfile] = $true
          Stop-OneTrainer $trainerState $currentTrainingProfile
          $currentTrainingProfile = ""
        }
      }

      if ([string]::IsNullOrWhiteSpace($currentTrainingProfile)) {
        $nextEntry = Select-NextTrainingProfile $activeProfiles $completedProfiles
        if ($null -eq $nextEntry -and $CycleProfiles) {
          Write-Host "All train-enabled profiles reached goal; cycling and retraining from start."
          $completedProfiles = @{}
          $nextEntry = Select-NextTrainingProfile $activeProfiles $completedProfiles
        }

        if ($null -ne $nextEntry) {
          $currentTrainingProfile = [string]$nextEntry.profile
          $deckList = [string]$resolvedDeckLists[$currentTrainingProfile]
          $metricsPort = $MetricsPortBase
          $effectiveTrainEnv = ConvertTo-EnvHashtable $nextEntry.train_env
          $effectiveSeed = Try-ParseInt64 $nextEntry.seed
          $proc = Start-LeagueTrainer $nextEntry $resolvedRegistryPath $metricsPort $deckList $resolvedMetaOpponentDeckList $effectiveTrainEnv $effectiveSeed
          $trainerState[$currentTrainingProfile] = [PSCustomObject]@{
            process = $proc
            restart_count = 0
            consecutive_failures = 0
            last_restart_reason = ""
            launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            metrics_port = $metricsPort
            train_decklist = $deckList
            opponent_decklist = $resolvedMetaOpponentDeckList
            effective_train_env = $effectiveTrainEnv
            effective_seed = $effectiveSeed
            stdout_log = (Join-Path $reportsDir ("trainers/{0}.stdout.log" -f $currentTrainingProfile))
            stderr_log = (Join-Path $reportsDir ("trainers/{0}.stderr.log" -f $currentTrainingProfile))
          }
          Append-CurriculumEvent "training_start" $snapshotMap[$currentTrainingProfile]
        }
      } else {
        $state = $trainerState[$currentTrainingProfile]
        $proc = if ($null -ne $state) { $state.process } else { $null }
        $dead = $false
        if ($null -eq $proc) {
          $dead = $true
        } else {
          try { $dead = $proc.HasExited } catch { $dead = $true }
        }
        if ($dead) {
          $state.restart_count = [int]$state.restart_count + 1
          $state.last_restart_reason = "exit"
          $launchedAt = $null
          try {
            if (-not [string]::IsNullOrWhiteSpace([string]$state.launched_at_utc)) {
              $launchedAt = [datetime]::Parse([string]$state.launched_at_utc)
            }
          } catch {
            $launchedAt = $null
          }
          $uptimeSec = 0
          if ($null -ne $launchedAt) {
            $uptimeSec = [int][Math]::Max(0, ((Get-Date).ToUniversalTime() - $launchedAt.ToUniversalTime()).TotalSeconds)
          }
          if ($uptimeSec -lt [Math]::Max(1, $CrashWindowSeconds)) {
            $state.consecutive_failures = [int]$state.consecutive_failures + 1
          } else {
            $state.consecutive_failures = 1
          }
          $exp = [Math]::Max(0, $state.consecutive_failures - 1)
          $baseBackoff = [Math]::Max(0, $RestartBackoffSeconds)
          $maxBackoff = [Math]::Max($baseBackoff, $RestartBackoffMaxSeconds)
          $backoff = [int][Math]::Min($maxBackoff, $baseBackoff * [Math]::Pow(2, $exp))
          Write-Warning ("Trainer exited profile={0} pid={1}; restarting reason=exit (count={2}, consecutive={3}, uptime={4}s, backoff={5}s)" -f `
            $currentTrainingProfile, $(if ($null -eq $proc) { 0 } else { $proc.Id }), $state.restart_count, $state.consecutive_failures, $uptimeSec, $backoff)
          Invoke-OrchestratorCleanup ("restart_{0}" -f $currentTrainingProfile)
          if ($backoff -gt 0) { Start-Sleep -Seconds $backoff }
          $entry = ($activeProfiles | Where-Object { $_.profile -eq $currentTrainingProfile } | Select-Object -First 1)
          if ($null -ne $entry) {
            $oppDeckList = $state.opponent_decklist
            if ([string]::IsNullOrWhiteSpace($oppDeckList)) { $oppDeckList = $resolvedMetaOpponentDeckList }
            $newProc = Start-LeagueTrainer $entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList $state.effective_train_env $state.effective_seed
            $state.process = $newProc
            $state.launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
          }
        }
      }
    } else {
      # Legacy behavior: start all active train-enabled profiles concurrently.
      $idx = 0
      foreach ($entry in $activeProfiles) {
        if (-not [bool]$entry.train_enabled) { continue }
        $profile = $entry.profile
        $deckList = [string]$resolvedDeckLists[$profile]
        $metricsPort = $MetricsPortBase + $idx
        $idx++

        if (-not $trainerState.ContainsKey($profile)) {
          $effectiveTrainEnv = ConvertTo-EnvHashtable $entry.train_env
          $effectiveSeed = Try-ParseInt64 $entry.seed
          $proc = Start-LeagueTrainer $entry $resolvedRegistryPath $metricsPort $deckList $resolvedMetaOpponentDeckList $effectiveTrainEnv $effectiveSeed
          $snap = $snapshotMap[$profile]
          $initialEpisode = 0
          if ($null -ne $snap -and $null -ne $snap.episode) {
            $initialEpisode = [int]$snap.episode
          }
          $trainerState[$profile] = [PSCustomObject]@{
            process = $proc
            restart_count = 0
            consecutive_failures = 0
            last_restart_reason = ""
            launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            metrics_port = $metricsPort
            train_decklist = $deckList
            opponent_decklist = $resolvedMetaOpponentDeckList
            effective_train_env = $effectiveTrainEnv
            effective_seed = $effectiveSeed
            stdout_log = (Join-Path $reportsDir ("trainers/{0}.stdout.log" -f $profile))
            stderr_log = (Join-Path $reportsDir ("trainers/{0}.stderr.log" -f $profile))
            last_progress_episode = $initialEpisode
            last_progress_at_utc = (Get-Date).ToUniversalTime().ToString("o")
          }
          continue
        }

        $state = $trainerState[$profile]
        $proc = $state.process
        $dead = $false
        if ($null -eq $proc) { $dead = $true } else { try { $dead = $proc.HasExited } catch { $dead = $true } }
        if ($dead) {
          $state.restart_count = [int]$state.restart_count + 1
          $state.last_restart_reason = "exit"
          Write-Warning ("Trainer exited profile={0} pid={1}; restarting reason=exit (count={2})" -f $profile, $(if ($null -eq $proc) { 0 } else { $proc.Id }), $state.restart_count)
          # In concurrent mode, do NOT run global cleanup here; it kills sibling trainers.
          Stop-OneTrainer $trainerState $profile
          if ($RestartBackoffSeconds -gt 0) {
            Start-Sleep -Seconds ([Math]::Max(1, $RestartBackoffSeconds))
          }
          $oppDeckList = $state.opponent_decklist
          if ([string]::IsNullOrWhiteSpace($oppDeckList)) { $oppDeckList = $resolvedMetaOpponentDeckList }
          $newProc = Start-LeagueTrainer $entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList $state.effective_train_env $state.effective_seed
          $state.process = $newProc
          $state.launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
          $snap = $snapshotMap[$profile]
          $restartEpisode = 0
          if ($null -ne $snap -and $null -ne $snap.episode) {
            $restartEpisode = [int]$snap.episode
          }
          $state.last_progress_episode = $restartEpisode
          $state.last_progress_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        } else {
          $snap = $snapshotMap[$profile]
          $currEpisode = 0
          if ($null -ne $snap -and $null -ne $snap.episode) {
            $currEpisode = [int]$snap.episode
          }
          $lastProgressEpisode = 0
          if ($null -ne $state.last_progress_episode) {
            $lastProgressEpisode = [int]$state.last_progress_episode
          }
          $nowUtc = (Get-Date).ToUniversalTime()
          if ($currEpisode -gt $lastProgressEpisode) {
            $state.last_progress_episode = $currEpisode
            $state.last_progress_at_utc = $nowUtc.ToString("o")
          } elseif ($StallRestartMinutes -gt 0) {
            $lastProgressAt = $null
            try {
              if (-not [string]::IsNullOrWhiteSpace([string]$state.last_progress_at_utc)) {
                $lastProgressAt = [datetime]::Parse([string]$state.last_progress_at_utc)
              }
            } catch {
              $lastProgressAt = $null
            }
            if ($null -eq $lastProgressAt) {
              $state.last_progress_at_utc = $nowUtc.ToString("o")
            } else {
              $stalledMinutes = ($nowUtc - $lastProgressAt.ToUniversalTime()).TotalMinutes
              if ($stalledMinutes -ge [double]$StallRestartMinutes) {
                $state.restart_count = [int]$state.restart_count + 1
                $state.last_restart_reason = "stall"
                Write-Warning ("Trainer stalled profile={0} pid={1}; restarting reason=stall (count={2}, stalled={3:N1}m, episode={4})" -f `
                  $profile, $proc.Id, $state.restart_count, $stalledMinutes, $currEpisode)
                Stop-OneTrainer $trainerState $profile
                if ($RestartBackoffSeconds -gt 0) {
                  Start-Sleep -Seconds ([Math]::Max(1, $RestartBackoffSeconds))
                }
                $oppDeckList = $state.opponent_decklist
                if ([string]::IsNullOrWhiteSpace($oppDeckList)) { $oppDeckList = $resolvedMetaOpponentDeckList }
                $newProc = Start-LeagueTrainer $entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList $state.effective_train_env $state.effective_seed
                $state.process = $newProc
                $state.launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
                $state.last_progress_episode = $currEpisode
                $state.last_progress_at_utc = (Get-Date).ToUniversalTime().ToString("o")
              }
            }
          }
        }
      }
    }

    $now = Get-Date
    if ($EnablePbt) {
      $pbtBatch = Invoke-PbtExploit $trainerState $activeProfiles $snapshotMap $resolvedRegistryPath $resolvedMetaOpponentDeckList $pbtGroupState $now.ToUniversalTime()
      foreach ($ev in @($pbtBatch)) {
        [void]$pbtEvents.Add($ev)
        if (-not [string]::IsNullOrWhiteSpace([string]$ev.skip_reason)) {
          Write-Host ("PBT exploit skipped group={0} winner={1} loser={2} reason={3} winner_wr={4:N3} loser_wr={5:N3} gap={6:N3} minGap={7:N3} minWinnerWr={8:N3} trigger={9} groupMinEp={10} deltaEp={11} timeDeltaPassed={12}" -f `
            $ev.population_group, $ev.winner, $ev.loser, $ev.skip_reason, [double]$ev.winner_wr, [double]$ev.loser_wr, [double]$ev.winner_gap, [double]$ev.winner_gap_min, [double]$ev.winner_wr_min_gate, $ev.trigger, [int]$ev.group_min_episode, [int]$ev.episode_delta, [bool]$ev.time_delta_gate_passed)
        } else {
          Write-Host ("PBT exploit group={0} winner={1} loser={2} winner_wr={3:N3} loser_wr={4:N3} gap={5:N3} seed={6} trigger={7} groupMinEp={8} deltaEp={9}" -f `
            $ev.population_group, $ev.winner, $ev.loser, [double]$ev.winner_wr, [double]$ev.loser_wr, [double]$ev.winner_gap, $ev.new_seed, $ev.trigger, [int]$ev.group_min_episode, [int]$ev.episode_delta)
        }
      }
      if (@($pbtBatch).Count -gt 0) {
        $lastPbtAt = $now
      }
    }
    if (-not $NoEval -and $now -ge $nextEvalAt) {
      Run-LeagueEval $resolvedRegistryPath
      $lastEvalAt = Get-Date
      $nextEvalAt = if ($RunOnce) { [datetime]::MaxValue } else { $lastEvalAt.AddMinutes([Math]::Max(1, $EvalEveryMinutes)) }
    }

    if ($StatusLogEverySeconds -gt 0 -and ((Get-Date) - $lastStatusLogAt).TotalSeconds -ge $StatusLogEverySeconds) {
      if ($SequentialTraining) {
        if (-not [string]::IsNullOrWhiteSpace($currentTrainingProfile)) {
          $snap = $snapshotMap[$currentTrainingProfile]
          if ($null -eq $snap) {
            Write-Host ("Training {0} currently, rolling winrate=n/a, episode=n/a" -f $currentTrainingProfile)
          } else {
            $wrText = if ($null -eq $snap.rolling_current) { "n/a" } else { [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F3}", [double]$snap.rolling_current) }
            $avgText = if ($null -eq $snap.rolling_avg) { "n/a" } else { [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F3}", [double]$snap.rolling_avg) }
            Write-Host ("Training {0} currently, rolling winrate={1} (avg={2}, target={3:F3}), episode={4}, promoted={5}, baseline_wr={6:F3}" -f `
              $currentTrainingProfile, $wrText, $avgText, [double]$snap.target_winrate, [int]$snap.episode, [bool]$snap.promoted, [double]$snap.baseline_wr)
          }
        } else {
          Write-Host "No active training profile currently (all train-enabled profiles reached target or are disabled)."
        }
      } else {
        $runningProfiles = @()
        $statusNowUtc = (Get-Date).ToUniversalTime()
        foreach ($entry in $activeProfiles) {
          if (-not [bool]$entry.train_enabled) { continue }
          $state = $trainerState[$entry.profile]
          if ($null -eq $state -or $null -eq $state.process) { continue }
          $isRunning = $false
          try {
            $isRunning = -not $state.process.HasExited
          } catch {
            $isRunning = $false
          }
          if ($isRunning) {
            $runningProfiles += [string]$entry.profile
          }
        }
        if ($runningProfiles.Count -eq 0) {
          Write-Host "Concurrent training active, but no trainer process is currently running."
        } else {
          Write-Host ("Concurrent training profiles running ({0}): {1}" -f $runningProfiles.Count, ($runningProfiles -join ", "))
        }

        $wrRows = @()
        foreach ($entry in $activeProfiles) {
          $snap = $snapshotMap[$entry.profile]
          if ($null -eq $snap) { continue }
          if ($null -eq $snap.rolling_current) { continue }
          $profileKey = [string]$entry.profile
          $snapEpisode = [int]$snap.episode
          $watermark = $snapEpisode
          if ($episodeWatermark.ContainsKey($profileKey)) {
            $prevWatermark = [int]$episodeWatermark[$profileKey]
            if ($prevWatermark -gt $watermark) {
              $watermark = $prevWatermark
            }
          }
          $episodeWatermark[$profileKey] = $watermark
          $wrRows += [PSCustomObject]@{
            profile = [string]$entry.profile
            train_enabled = [bool]$entry.train_enabled
            wr = [double]$snap.rolling_current
            avg = if ($null -eq $snap.rolling_avg) { $null } else { [double]$snap.rolling_avg }
            episode = $watermark
            eps_per_sec = $null
            eps_per_min_ema = $null
          }
        }
        foreach ($row in $wrRows) {
          if (-not [bool]$row.train_enabled) { continue }
          $profileKey = [string]$row.profile
          $prev = $episodeRateState[$profileKey]
          $rawRate = $null
          $emaRate = $null
          $currentEpisode = [int]$row.episode
          if ($null -ne $prev) {
            $deltaSeconds = ($statusNowUtc - [datetime]$prev.timestamp_utc).TotalSeconds
            $deltaEpisodes = [double]($currentEpisode - [int]$prev.episode)
            if ($deltaSeconds -gt 0 -and $deltaEpisodes -ge 0) {
              $rawRate = $deltaEpisodes / $deltaSeconds
              $tauSeconds = [Math]::Max(1.0, [double]$EpsRateHalfLifeMinutes * 60.0)
              $alpha = 1.0 - [Math]::Exp(-($deltaSeconds / $tauSeconds))
              $prevEma = $rawRate
              if ($null -ne $prev.ema_eps_per_sec) {
                $prevEma = [double]$prev.ema_eps_per_sec
              }
              $emaRate = ($alpha * $rawRate) + ((1.0 - $alpha) * $prevEma)
            }
          }
          $row.eps_per_sec = $rawRate
          if ($null -ne $emaRate) {
            $row.eps_per_min_ema = $emaRate * 60.0
          }
          $episodeRateState[$profileKey] = [PSCustomObject]@{
            episode = $currentEpisode
            timestamp_utc = $statusNowUtc
            ema_eps_per_sec = $emaRate
          }
        }
        if ($wrRows.Count -gt 0) {
          $orderedWrRows = @($wrRows | Sort-Object -Property @{Expression="wr"; Descending=$true}, @{Expression="profile"; Ascending=$true})
          $segments = @()
          foreach ($row in $orderedWrRows) {
            $wrText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F3}", [double]$row.wr)
            $avgText = if ($null -eq $row.avg) {
              "n/a"
            } else {
              [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F3}", [double]$row.avg)
            }
            $epsText = if ($null -eq $row.eps_per_sec) {
              "n/a"
            } else {
              [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F2}", [double]$row.eps_per_sec)
            }
            $epsMinEmaText = if ($null -eq $row.eps_per_min_ema) {
              "n/a"
            } else {
              [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F2}", [double]$row.eps_per_min_ema)
            }
            $marker = if ([bool]$row.train_enabled) { "*" } else { "" }
            $segments += ("{0}{1}=wr:{2},avg:{3},ep:{4},eps/s:{5},eps/m_ema:{6}" -f `
              $marker, $row.profile, $wrText, $avgText, [int]$row.episode, $epsText, $epsMinEmaText)
          }
          Write-Host ("Rolling winrates (*train_enabled): {0}" -f ($segments -join " | "))
        } else {
          Write-Host "Rolling winrates: n/a (no snapshot data yet)."
        }
      }
      $lastStatusLogAt = Get-Date
    }

    $note = "idle"
    if ($SequentialTraining) {
      $note = if ([string]::IsNullOrWhiteSpace($currentTrainingProfile)) { "idle" } else { "training_" + $currentTrainingProfile }
    } else {
      $runningCount = 0
      foreach ($state in @($trainerState.Values)) {
        if ($null -eq $state -or $null -eq $state.process) { continue }
        $isRunning = $false
        try {
          $isRunning = -not $state.process.HasExited
        } catch {
          $isRunning = $false
        }
        if ($isRunning) { $runningCount++ }
      }
      if ($runningCount -gt 0) {
        $note = "training_concurrent_" + $runningCount
      }
    }
    Write-PbtState $trainerState $activeProfiles $lastPbtAt $pbtEvents $pbtGroupState
    Write-OrchestratorStatus $trainerState $activeProfiles $resolvedRegistryPath $lastEvalAt $nextEvalAt $note $currentTrainingProfile $snapshotMap $completedProfiles

    if ($RunOnce) {
      break
    }
    Start-Sleep -Seconds ([Math]::Max(5, $PollSeconds))
  }
}
finally {
  Write-PbtState $trainerState $activeProfiles $lastPbtAt $pbtEvents $pbtGroupState
  Write-OrchestratorStatus $trainerState $activeProfiles $resolvedRegistryPath $lastEvalAt $nextEvalAt "stopping" $currentTrainingProfile $snapshotMap $completedProfiles
  if (-not $LeaveRunning) {
    Stop-LeagueTrainers $trainerState
  }
  Write-Host "League orchestrator stopped."
}
