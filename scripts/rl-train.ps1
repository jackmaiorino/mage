param(
  [ValidateSet("default", "perf", "debug")]
  [string]$Profile = "default",
  [int]$TotalEpisodes = 50,
  [int]$NumGameRunners = 4,
  # Default assumes we run Maven from the AIRL module directory
  [string]$DeckListFile = "",
  [string]$ModelPath = "",
  [string]$StatsPath = "",
  [int]$MetricsPort = 9090,

  # Named model profile - all artifacts (models + logs) go under rl/profiles/<ModelProfile>/
  [string]$ModelProfile = "",

  # Optional overrides for frequently tweaked knobs
  [ValidateSet("WARNING", "INFO", "DEBUG")]
  [string]$LogLevel = "",
  [int]$PyBatchMaxSize = 0,
  [int]$PyBatchTimeoutMs = 0,
  [int]$ResetEpisodeCounter = -1,
  [int]$ActivationDiag = -1,
  [int]$PauseOnActivationFailure = -1,
  [int]$GameLogging = -1,
  [int]$GameLogFrequency = 0,
  [int]$UseGae = -1,
  [int]$GaeAutoEnable = -1,
  [int]$AdaptiveCurriculum = -1,

  # Override the main class (default: RLTrainer; use DraftTrainer for cube draft training)
  [string]$MainClass = "mage.player.ai.rl.RLTrainer",

  # Embedding preflight: auto-generate profile embeddings if missing.
  [bool]$PreflightEmbeddings = $true,
  # Refresh profile embeddings on startup (recommended; cheap with global cache).
  [bool]$PreflightRefreshEmbeddings = $true,
  [string]$PythonExe = "python",
  [bool]$PreflightStrict = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modulePath = "Mage.Server.Plugins/Mage.Player.AIRL"

function Resolve-RepoPath([string]$pathValue) {
  if ([string]::IsNullOrWhiteSpace($pathValue)) {
    return ""
  }
  if ([System.IO.Path]::IsPathRooted($pathValue)) {
    try { return (Resolve-Path -LiteralPath $pathValue).Path } catch { return $pathValue }
  }
  $candidate = Join-Path $repoRoot $pathValue
  if (Test-Path -LiteralPath $candidate) {
    return (Resolve-Path -LiteralPath $candidate).Path
  }
  return $candidate
}

function Ensure-ProfileEmbeddings([string]$profile, [string]$resolvedDeckList) {
  if (-not $PreflightEmbeddings) {
    return
  }
  if ([string]::IsNullOrWhiteSpace($profile)) {
    return
  }

  $modelsDir = Join-Path $repoRoot ("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{0}/models" -f $profile)
  $embeddingPath = Join-Path $modelsDir "card_embeddings.json"
  if ((Test-Path -LiteralPath $embeddingPath) -and (-not $PreflightRefreshEmbeddings)) {
    Write-Host ("Embeddings present for profile={0}: {1}" -f $profile, $embeddingPath)
    return
  }
  if ((Test-Path -LiteralPath $embeddingPath) -and $PreflightRefreshEmbeddings) {
    Write-Host ("Refreshing embeddings for profile={0}: {1}" -f $profile, $embeddingPath)
  }

  $generator = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/generate_card_embeddings.py"
  if (-not (Test-Path -LiteralPath $generator)) {
    throw "Embedding generator not found: $generator"
  }

  if ([string]::IsNullOrWhiteSpace($resolvedDeckList) -or -not (Test-Path -LiteralPath $resolvedDeckList)) {
    throw ("Embeddings missing for profile={0}, and deck list source is unavailable: {1}" -f $profile, $resolvedDeckList)
  }

  $sourceFlag = "--decklist"
  $deckExt = [System.IO.Path]::GetExtension($resolvedDeckList).ToLowerInvariant()
  if ($deckExt -eq ".dck") {
    $sourceFlag = "--cube"
  } elseif ($deckExt -eq ".dek") {
    $sourceFlag = "--dek"
  }

  $pythonCmd = $PythonExe
  if ([string]::IsNullOrWhiteSpace($pythonCmd) -or $pythonCmd -eq "python") {
    $venvPy = Join-Path $repoRoot "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/venv/Scripts/python.exe"
    if (Test-Path -LiteralPath $venvPy) {
      $pythonCmd = $venvPy
    }
  }

  New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

  $args = @(
    $generator,
    "--profile", $profile,
    $sourceFlag, $resolvedDeckList,
    "--output", $embeddingPath
  )

  Write-Host ("Generating embeddings for profile={0} using {1}={2}" -f $profile, $sourceFlag, $resolvedDeckList)
  try {
    & $pythonCmd @args
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

# Prefer explicit parameter, otherwise respect existing env var, otherwise default.
if ($DeckListFile -eq "") {
  if ($env:DECK_LIST_FILE -and $env:DECK_LIST_FILE.Trim() -ne "") {
    $DeckListFile = $env:DECK_LIST_FILE
  } else {
    $DeckListFile = "src/mage/player/ai/decks/PauperSubset/decklist.txt"
  }
}

# If caller passed a repo-relative path (e.g. Mage.Server.Plugins/...), convert it to an absolute path
if (-not [System.IO.Path]::IsPathRooted($DeckListFile)) {
  $maybeRepoPath = Join-Path $repoRoot $DeckListFile
  if (Test-Path $maybeRepoPath) {
    $DeckListFile = (Resolve-Path $maybeRepoPath).Path
  }
}
$resolvedDeckList = Resolve-RepoPath $DeckListFile

$env:MODE = "train"
$env:TOTAL_EPISODES = "$TotalEpisodes"
$env:NUM_GAME_RUNNERS = "$NumGameRunners"
$env:DECK_LIST_FILE = "$DeckListFile"
$env:METRICS_PORT = "$MetricsPort"

if ($ModelPath -ne "") { $env:MTG_MODEL_PATH = "$ModelPath" }
if ($StatsPath -ne "") { $env:STATS_PATH = "$StatsPath" }

# Model profile - sets MODEL_PROFILE env var; Java+Python derive all paths from it
if ($ModelProfile -ne "") {
  $env:MODEL_PROFILE = "$ModelProfile"
} elseif ($env:MODEL_PROFILE -and $env:MODEL_PROFILE.Trim() -ne "") {
  $ModelProfile = $env:MODEL_PROFILE
}

if (-not [string]::IsNullOrWhiteSpace($ModelProfile)) {
  Ensure-ProfileEmbeddings $ModelProfile $resolvedDeckList
}

# -----------------------
# Profile defaults
# -----------------------
switch ($Profile) {
  "default" {
    # Minimal overrides; rely on code defaults.
  }
  "perf" {
    # Only apply perf defaults if caller did NOT already set env vars.
    if (-not $LogLevel -and (-not $env:MTG_AI_LOG_LEVEL -or $env:MTG_AI_LOG_LEVEL.Trim() -eq "")) { $LogLevel = "WARNING" }

    if ($PyBatchMaxSize -le 0) {
      if ($env:PY_BATCH_MAX_SIZE -and $env:PY_BATCH_MAX_SIZE.Trim() -ne "") {
        try { $PyBatchMaxSize = [int]$env:PY_BATCH_MAX_SIZE } catch { $PyBatchMaxSize = 768 }
      } else {
        $PyBatchMaxSize = 768
      }
    }

    if ($PyBatchTimeoutMs -le 0) {
      if ($env:PY_BATCH_TIMEOUT_MS -and $env:PY_BATCH_TIMEOUT_MS.Trim() -ne "") {
        try { $PyBatchTimeoutMs = [int]$env:PY_BATCH_TIMEOUT_MS } catch { $PyBatchTimeoutMs = 50 }
      } else {
        $PyBatchTimeoutMs = 50
      }
    }
  }
  "debug" {
    if (-not $LogLevel) { $LogLevel = "INFO" }
    if ($ActivationDiag -lt 0) { $ActivationDiag = 1 }
    if ($PauseOnActivationFailure -lt 0) { $PauseOnActivationFailure = 0 }
    if ($GameLogging -lt 0) { $GameLogging = 1 }
    if ($GameLogFrequency -le 0) { $GameLogFrequency = 200 }
  }
}

# -----------------------
# Apply overrides
# -----------------------
if ($LogLevel) { $env:MTG_AI_LOG_LEVEL = "$LogLevel" }
if ($PyBatchMaxSize -gt 0) { $env:PY_BATCH_MAX_SIZE = "$PyBatchMaxSize" }
if ($PyBatchTimeoutMs -gt 0) { $env:PY_BATCH_TIMEOUT_MS = "$PyBatchTimeoutMs" }

if ($ResetEpisodeCounter -ge 0) { $env:RESET_EPISODE_COUNTER = "$ResetEpisodeCounter" }
if ($ActivationDiag -ge 0) { $env:RL_ACTIVATION_DIAG = "$ActivationDiag" }
if ($PauseOnActivationFailure -ge 0) { $env:PAUSE_ON_ACTIVATION_FAILURE = "$PauseOnActivationFailure" }

if ($GameLogging -ge 0) { $env:GAME_LOGGING = "$GameLogging" }
if ($GameLogFrequency -gt 0) { $env:GAME_LOG_FREQUENCY = "$GameLogFrequency" }

if ($UseGae -ge 0) { $env:USE_GAE = "$UseGae" }
if ($GaeAutoEnable -ge 0) { $env:GAE_AUTO_ENABLE = "$GaeAutoEnable" }

if ($AdaptiveCurriculum -ge 0) { $env:ADAPTIVE_CURRICULUM = "$AdaptiveCurriculum" }

$profileMsg = if ($ModelProfile -ne "") { " profile=$ModelProfile" } else { "" }
Write-Host "Train($Profile):$profileMsg totalEpisodes=$TotalEpisodes runners=$NumGameRunners deckList=$DeckListFile metricsPort=$MetricsPort"

Push-Location $repoRoot
try {
  mvn -q -pl $modulePath -am -DskipTests compile exec:java `
    "-Dexec.mainClass=$MainClass" `
    "-Dexec.args=train"
}
finally {
  $orchestratedRaw = [string]$env:ORCHESTRATED_RUN
  $isOrchestrated = $false
  if (-not [string]::IsNullOrWhiteSpace($orchestratedRaw)) {
    $orchestratedNorm = $orchestratedRaw.Trim().ToLowerInvariant()
    $isOrchestrated = ($orchestratedNorm -eq "1" -or $orchestratedNorm -eq "true" -or $orchestratedNorm -eq "yes")
  }
  if (-not $isOrchestrated) {
    # Standalone runs: Ctrl+C can leave java/python subprocesses alive; attempt cleanup.
    try {
      & (Join-Path $PSScriptRoot "rl-stop.ps1") -Quiet
    } catch {
      # Don't block exit on cleanup failures
    }
  }
  Pop-Location
}

