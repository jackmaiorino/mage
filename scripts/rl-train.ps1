param(
  [int]$TotalEpisodes = 50,
  [int]$NumGameRunners = 4,
  # Default assumes we run Maven from the AIRL module directory
  [string]$DeckListFile = "",
  [string]$ModelPath = "",
  [string]$StatsPath = "",
  [int]$MetricsPort = 9090
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

if ($ModelPath -ne "") { $env:MODEL_PATH = "$ModelPath" }
if ($StatsPath -ne "") { $env:STATS_PATH = "$StatsPath" }

Write-Host "Train: totalEpisodes=$TotalEpisodes runners=$NumGameRunners deckList=$DeckListFile"

Push-Location $repoRoot
try {
  mvn -q -pl $modulePath -am -DskipTests compile exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
    "-Dexec.args=train"
}
finally {
  Pop-Location
}

