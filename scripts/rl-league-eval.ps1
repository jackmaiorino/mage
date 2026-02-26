param(
  [string]$RegistryPath = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_league_registry.json",
  [int]$GamesPerDirection = 6,
  [double]$KFactor = 20.0,
  [int]$AnchorGames = 20,
  [int]$CadenceEpisodes = 5000,
  [int]$Py4jBasePort = 26334,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modulePath = "Mage.Server.Plugins/Mage.Player.AIRL"

if (-not [System.IO.Path]::IsPathRooted($RegistryPath)) {
  $RegistryPath = (Join-Path $repoRoot $RegistryPath)
}

$env:MODE = "league_eval"
$env:LEAGUE_REGISTRY_PATH = "$RegistryPath"
$env:LEAGUE_ELO_GAMES_PER_DIRECTION = "$GamesPerDirection"
$env:LEAGUE_ELO_K_FACTOR = "$KFactor"
$env:LEAGUE_ANCHOR_ENABLE = "1"
$env:LEAGUE_ANCHOR_GAMES = "$AnchorGames"
$env:LEAGUE_EVAL_CADENCE_EPISODES = "$CadenceEpisodes"
$env:LEAGUE_EVAL_FORCE = $(if ($Force) { "1" } else { "0" })
$env:PY4J_BASE_PORT = "$Py4jBasePort"

Write-Host "League Eval: registry=$RegistryPath gpd=$GamesPerDirection k=$KFactor anchorGames=$AnchorGames cadence=$CadenceEpisodes py4jBasePort=$Py4jBasePort force=$($Force.IsPresent)"

Push-Location $repoRoot
try {
  mvn -q -pl $modulePath -am -DskipTests compile exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
    "-Dexec.args=league_eval"
}
finally {
  Pop-Location
}
