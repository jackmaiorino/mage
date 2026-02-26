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
  [bool]$HideTrainerWindows = $true,
  [bool]$SequentialTraining = $true,
  [double]$TrainTargetWinrate = 0.60,
  [int]$TrainMinEpisodes = 2000,
  [int]$WinrateWindow = 200,
  [int]$StatusLogEverySeconds = 60,
  [switch]$CycleProfiles
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trainScript = Join-Path $PSScriptRoot "rl-train.ps1"
$evalScript = Join-Path $PSScriptRoot "rl-league-eval.ps1"
$stopScript = Join-Path $PSScriptRoot "rl-stop.ps1"
$reportsDir = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
$statusPath = Join-Path $reportsDir "orchestrator_status.json"
$historyPath = Join-Path $reportsDir "curriculum_history.csv"

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
    $active += [PSCustomObject]@{
      profile = $profile.Trim()
      deck_path = [string]$e.deck_path
      train_decklist = $trainDeck
      train_enabled = $trainEnabled
      target_winrate = $targetWinrate
      priority = $priority
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

  if (Test-Path -LiteralPath $embeddingPath) {
    Write-Host ("Embeddings present for profile={0}: {1}" -f $profile, $embeddingPath)
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

function Start-LeagueTrainer($entry, [string]$resolvedRegistryPath, [int]$metricsPort, [string]$resolvedAgentDeckList, [string]$resolvedOpponentDeckList) {
  $profile = $entry.profile
  $outDir = Join-Path $reportsDir "trainers"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $stdoutPath = Join-Path $outDir ("{0}.stdout.log" -f $profile)
  $stderrPath = Join-Path $outDir ("{0}.stderr.log" -f $profile)

  $envSetters = @(
    ('$env:OPPONENT_SAMPLER=' + (Quote-Single "league")),
    ('$env:LEAGUE_REGISTRY_PATH=' + (Quote-Single $resolvedRegistryPath)),
    ('$env:LEAGUE_PROMOTE_WR=' + (Quote-Single ([string]$LeaguePromoteWr)))
  )
  if (-not [string]::IsNullOrWhiteSpace($resolvedAgentDeckList)) {
    $envSetters += ('$env:RL_AGENT_DECK_LIST=' + (Quote-Single $resolvedAgentDeckList))
  }
  if (-not [string]::IsNullOrWhiteSpace($TrainLogLevel)) {
    $envSetters += ('$env:MTG_AI_LOG_LEVEL=' + (Quote-Single $TrainLogLevel))
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

  $command = "& { " + ($envSetters -join "; ") + "; " + $invoke + " }"
  $argList = @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command)

  if ($DryRun) {
    Write-Host ("[dry-run] trainer {0} command: {1}" -f $profile, $command)
    return $null
  }

  if ($HideTrainerWindows) {
    $proc = Start-Process -FilePath "powershell" -ArgumentList $argList -WorkingDirectory $repoRoot -PassThru `
      -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
  } else {
    $proc = Start-Process -FilePath "powershell" -ArgumentList $argList -WorkingDirectory $repoRoot -PassThru `
      -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
  }
  Write-Host ("Started trainer profile={0} pid={1} metricsPort={2}" -f $profile, $proc.Id, $metricsPort)
  return $proc
}

function Stop-LeagueTrainers($stateMap) {
  foreach ($k in @($stateMap.Keys)) {
    $item = $stateMap[$k]
    if ($null -eq $item) { continue }
    $proc = $item.process
    if ($null -eq $proc) { continue }
    try {
      if (-not $proc.HasExited) {
        Write-Host ("Stopping trainer profile={0} pid={1}" -f $k, $proc.Id)
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
      }
    } catch {
      Write-Warning ("Failed stopping profile={0}: {1}" -f $k, $_.Exception.Message)
    }
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
  try {
    if (-not $proc.HasExited) {
      Write-Host ("Stopping trainer profile={0} pid={1}" -f $profile, $proc.Id)
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
  } catch {
    Write-Warning ("Failed stopping profile={0}: {1}" -f $profile, $_.Exception.Message)
  }
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
      $tail = Get-Content -LiteralPath $statsPath -Tail ([Math]::Max(10, $WinrateWindow))
      $vals = New-Object System.Collections.Generic.List[Double]
      foreach ($line in $tail) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.StartsWith("episode,")) { continue }
        $parts = $line.Split(',')
        if ($parts.Length -lt 5) { continue }
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
    Write-Host ("[dry-run] cleanup via rl-stop.ps1 reason={0}" -f $reason)
    return
  }
  if (-not (Test-Path -LiteralPath $stopScript)) {
    Write-Warning ("Cleanup requested but stop script missing: {0}" -f $stopScript)
    return
  }
  try {
    Write-Host ("Running cleanup via rl-stop.ps1 (reason={0})" -f $reason)
    & $stopScript -Quiet
  } catch {
    Write-Warning ("Cleanup failed: {0}" -f $_.Exception.Message)
  }
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

$trainerState = @{}
$completedProfiles = @{}
$snapshotMap = @{}
$currentTrainingProfile = ""
$lastStatusLogAt = [datetime]::MinValue
$lastEvalAt = $null
$nextEvalAt = if ($RunOnce) { Get-Date } else { (Get-Date).AddMinutes([Math]::Max(0, $InitialEvalDelayMinutes)) }

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
          $proc = Start-LeagueTrainer $nextEntry $resolvedRegistryPath $metricsPort $deckList $resolvedMetaOpponentDeckList
          $trainerState[$currentTrainingProfile] = [PSCustomObject]@{
            process = $proc
            restart_count = 0
            consecutive_failures = 0
            launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            metrics_port = $metricsPort
            train_decklist = $deckList
            opponent_decklist = $resolvedMetaOpponentDeckList
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
          Write-Warning ("Trainer exited profile={0} pid={1}; restarting (count={2}, consecutive={3}, uptime={4}s, backoff={5}s)" -f `
            $currentTrainingProfile, $(if ($null -eq $proc) { 0 } else { $proc.Id }), $state.restart_count, $state.consecutive_failures, $uptimeSec, $backoff)
          Invoke-OrchestratorCleanup ("restart_{0}" -f $currentTrainingProfile)
          if ($backoff -gt 0) { Start-Sleep -Seconds $backoff }
          $entry = ($activeProfiles | Where-Object { $_.profile -eq $currentTrainingProfile } | Select-Object -First 1)
          if ($null -ne $entry) {
            $oppDeckList = $state.opponent_decklist
            if ([string]::IsNullOrWhiteSpace($oppDeckList)) { $oppDeckList = $resolvedMetaOpponentDeckList }
            $newProc = Start-LeagueTrainer $entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList
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
          $proc = Start-LeagueTrainer $entry $resolvedRegistryPath $metricsPort $deckList $resolvedMetaOpponentDeckList
          $trainerState[$profile] = [PSCustomObject]@{
            process = $proc
            restart_count = 0
            consecutive_failures = 0
            launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            metrics_port = $metricsPort
            train_decklist = $deckList
            opponent_decklist = $resolvedMetaOpponentDeckList
            stdout_log = (Join-Path $reportsDir ("trainers/{0}.stdout.log" -f $profile))
            stderr_log = (Join-Path $reportsDir ("trainers/{0}.stderr.log" -f $profile))
          }
          continue
        }

        $state = $trainerState[$profile]
        $proc = $state.process
        $dead = $false
        if ($null -eq $proc) { $dead = $true } else { try { $dead = $proc.HasExited } catch { $dead = $true } }
        if ($dead) {
          $state.restart_count = [int]$state.restart_count + 1
          Write-Warning ("Trainer exited profile={0} pid={1}; restarting (count={2})" -f $profile, $(if ($null -eq $proc) { 0 } else { $proc.Id }), $state.restart_count)
          Invoke-OrchestratorCleanup ("restart_{0}" -f $profile)
          $oppDeckList = $state.opponent_decklist
          if ([string]::IsNullOrWhiteSpace($oppDeckList)) { $oppDeckList = $resolvedMetaOpponentDeckList }
          $newProc = Start-LeagueTrainer $entry $resolvedRegistryPath $state.metrics_port $state.train_decklist $oppDeckList
          $state.process = $newProc
          $state.launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        }
      }
    }

    $now = Get-Date
    if (-not $NoEval -and $now -ge $nextEvalAt) {
      Run-LeagueEval $resolvedRegistryPath
      $lastEvalAt = Get-Date
      $nextEvalAt = if ($RunOnce) { [datetime]::MaxValue } else { $lastEvalAt.AddMinutes([Math]::Max(1, $EvalEveryMinutes)) }
    }

    if ($StatusLogEverySeconds -gt 0 -and ((Get-Date) - $lastStatusLogAt).TotalSeconds -ge $StatusLogEverySeconds) {
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
      $lastStatusLogAt = Get-Date
    }

    $note = if ([string]::IsNullOrWhiteSpace($currentTrainingProfile)) { "idle" } else { "training_" + $currentTrainingProfile }
    Write-OrchestratorStatus $trainerState $activeProfiles $resolvedRegistryPath $lastEvalAt $nextEvalAt $note $currentTrainingProfile $snapshotMap $completedProfiles

    if ($RunOnce) {
      break
    }
    Start-Sleep -Seconds ([Math]::Max(5, $PollSeconds))
  }
}
finally {
  Write-OrchestratorStatus $trainerState $activeProfiles $resolvedRegistryPath $lastEvalAt $nextEvalAt "stopping" $currentTrainingProfile $snapshotMap $completedProfiles
  if (-not $LeaveRunning) {
    Stop-LeagueTrainers $trainerState
  }
  Write-Host "League orchestrator stopped."
}
