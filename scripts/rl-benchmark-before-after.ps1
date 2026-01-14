param(
  [int]$GamesPerMatchup = 5,
  [int]$TotalEpisodes = 50,
  [int]$NumGameRunners = 4,
  [string]$DeckListFile = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
)

$ErrorActionPreference = "Stop"

Write-Host "=== BENCHMARK (before) ==="
& "$PSScriptRoot/rl-benchmark.ps1" -GamesPerMatchup $GamesPerMatchup -DeckListFile $DeckListFile

Write-Host "=== TRAIN ==="
& "$PSScriptRoot/rl-train.ps1" -TotalEpisodes $TotalEpisodes -NumGameRunners $NumGameRunners -DeckListFile $DeckListFile

Write-Host "=== BENCHMARK (after) ==="
& "$PSScriptRoot/rl-benchmark.ps1" -GamesPerMatchup $GamesPerMatchup -DeckListFile $DeckListFile

