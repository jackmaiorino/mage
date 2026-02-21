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
  [int]$AdaptiveCurriculum = -1
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modulePath = "Mage.Server.Plugins/Mage.Player.AIRL"

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
    "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
    "-Dexec.args=train"
}
finally {
  # Ctrl+C can leave java/python subprocesses alive; always attempt cleanup.
  try {
    & (Join-Path $PSScriptRoot "rl-stop.ps1") -Quiet
  } catch {
    # Don't block exit on cleanup failures
  }
  Pop-Location
}

