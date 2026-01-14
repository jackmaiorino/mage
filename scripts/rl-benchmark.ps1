param(
  [int]$GamesPerMatchup = 5,
  # Default assumes we run Maven from the AIRL module directory
  [string]$DeckListFile = "src/mage/player/ai/decks/PauperSubset/decklist.txt",
  [int]$MetricsPort = 9090,
  [int]$BenchmarkThreads = 0,
  [int]$LogEvery = 10
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modulePath = "Mage.Server.Plugins/Mage.Player.AIRL"

# If caller passed a repo-relative path (e.g. Mage.Server.Plugins/...), convert it to an absolute path
if (-not [System.IO.Path]::IsPathRooted($DeckListFile)) {
  $maybeRepoPath = Join-Path $repoRoot $DeckListFile
  if (Test-Path $maybeRepoPath) {
    $DeckListFile = (Resolve-Path $maybeRepoPath).Path
  }
}

$env:MODE = "benchmark"
$env:GAMES_PER_MATCHUP = "$GamesPerMatchup"
$env:DECK_LIST_FILE = "$DeckListFile"
$env:METRICS_PORT = "$MetricsPort"

if ($BenchmarkThreads -le 0) {
  $BenchmarkThreads = [Math]::Max(1, [Environment]::ProcessorCount - 1)
}
$env:BENCHMARK_THREADS = "$BenchmarkThreads"
$env:BENCHMARK_LOG_EVERY = "$LogEvery"

Write-Host "Benchmark: gamesPerMatchup=$GamesPerMatchup deckList=$DeckListFile"
Write-Host "Benchmark threads: $BenchmarkThreads (logEvery=$LogEvery games)"

Push-Location $repoRoot
try {
  mvn -q -pl $modulePath -am -DskipTests compile exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
    "-Dexec.args=benchmark"
}
finally {
  Pop-Location
}
