param(
    [string]$Profile = "Pauper-Generalist-Value-v2",
    [string]$DeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt",
    [int]$SamplesPerDeck = 500,
    [bool]$FirstOnly = $true,
    [string]$RunId = "",
    [int]$ModelDModel = 128,
    [int]$ModelNumLayers = 2,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$outDir = Join-Path $repo ("local-training\local_pbt\mulligan_probes\" + $RunId)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not $SkipCompile) {
    & mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
    if ($LASTEXITCODE -ne 0) {
        throw "compile failed with exit code $LASTEXITCODE"
    }
}

$env:MODEL_PROFILE = $Profile
$env:PY_SERVICE_MODE = "local"
$env:PY_BACKEND_MODE = "single"
$env:MODEL_D_MODEL = [string]$ModelDModel
$env:MODEL_NUM_LAYERS = [string]$ModelNumLayers
$env:RL_RANDOM_DECISIONS = "0"
$env:RL_MULLIGAN_TRACE_JSONL = "0"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:MULLIGAN_HARD_OVERRIDES_ENABLE = "0"
$env:GAME_LOG_FREQUENCY = "0"
$env:TRAIN_LOG_EVERY = "0"
$env:CUDA_MEM_FRACTION = "0.70"
$env:PY_BATCH_TIMEOUT_MS = "5"
$env:PY_SCORE_TIMEOUT_MS = "5000"

& mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.MulliganProbe" `
    "-Dexec.args=--profile $Profile --deck-list $DeckList --samples $SamplesPerDeck --first-only $FirstOnly --out $outDir"
if ($LASTEXITCODE -ne 0) {
    throw "mulligan probe failed with exit code $LASTEXITCODE"
}

Write-Host "Mulligan probe output: $outDir"
