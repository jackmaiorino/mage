param(
    [string]$Profile = "Pauper-Generalist-Value-v2",
    [string]$DeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt",
    [string]$AgentDeckList = "",
    [string]$OppDeckList = "",
    [int]$Pairs = 128,
    [int]$BatchSize = 32,
    [int]$TimeoutSec = 45,
    [int]$PostTrainWaitMs = 3000,
    [string]$Opponent = "rl",
    [int]$Cp7Skill = 7,
    [string]$ServiceMode = "local",
    [string]$RunId = "",
    [int]$ModelDModel = 128,
    [int]$ModelNumLayers = 2,
    [int]$ReportEvery = 25,
    [int]$Workers = 8,
    [int]$AiThreads = 24,
    [switch]$LineMode,
    [int]$LineMaxMulls = 2,
    [int]$LineBottomCombos = 0,
    [double]$LineMarginMin = 0.05,
    [double]$LineTargetTemperature = 0.50,
    [bool]$LineBalanceMulliganPrompts = $true,
    [bool]$LineBucketSoftTargets = $false,
    [string]$LineBalanceKey = "deck-resource",
    [int]$LineBalanceSingletonLimit = 1,
    [bool]$LineTrainBottoms = $true,
    [int]$LineTrainEpochs = 1,
    [string]$ExportTrainingDataFile = "",
    [string]$ExportWinningLineDataFile = "",
    [double]$MctsKlLossCoef = 1.0,
    [bool]$DistillHeadOnly = $true,
    [bool]$DistillPolicyPathOnly = $false,
    [switch]$RandomDecisions,
    [switch]$CollectOnly,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$outDir = Join-Path $repo ("local-training\local_pbt\mulligan_counterfactual\" + $RunId)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not $SkipCompile) {
    & mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
    if ($LASTEXITCODE -ne 0) {
        throw "compile failed with exit code $LASTEXITCODE"
    }
}

$env:MODEL_PROFILE = $Profile
$env:PY_SERVICE_MODE = $ServiceMode
$env:PY_BACKEND_MODE = "single"
$env:MODEL_D_MODEL = [string]$ModelDModel
$env:MODEL_NUM_LAYERS = [string]$ModelNumLayers
$env:RL_RANDOM_DECISIONS = if ($RandomDecisions) { "1" } else { "0" }
$env:RL_MULLIGAN_TRACE_JSONL = "0"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:RL_SKIP_SIM_VALIDATION = "0"
$env:MULLIGAN_HARD_OVERRIDES_ENABLE = "0"
$env:GAME_LOG_FREQUENCY = "0"
$env:TRAIN_LOG_EVERY = "25"
$env:CUDA_MEM_FRACTION = "0.70"
$env:AI_MAX_THREADS_FOR_SIMULATIONS = [string]$AiThreads

# Isolate this pass to supervised policy distillation on the mulligan head.
$env:LOSS_SCHEDULE_ENABLE = "0"
$env:FREEZE_ENCODER_IN_WARMUP = "0"
$env:DISTILL_HEAD_ONLY = if ($DistillHeadOnly) { "1" } else { "0" }
$env:DISTILL_POLICY_PATH_ONLY = if ($DistillPolicyPathOnly) { "1" } else { "0" }
$env:POLICY_LOSS_COEF = "0"
$env:POLICY_LOSS_COEF_WARMUP = "0"
$env:VALUE_LOSS_COEF = "0"
$env:VALUE_LOSS_COEF_WARMUP = "0"
$env:ENTROPY_LOSS_MULT = "0"
$env:ENTROPY_LOSS_MULT_WARMUP = "0"
$env:BELIEF_LOSS_COEF = "0"
$env:MCTS_KL_LOSS_COEF = [string]$MctsKlLossCoef
$env:MULLIGAN_SAMPLE_WEIGHT = "1.0"
$env:LONDON_MULLIGAN_SAMPLE_WEIGHT = "1.0"

$env:PY_BATCH_MAX_SIZE = [string]$BatchSize
$env:PY_BATCH_TIMEOUT_MS = "5"
$env:PY_SCORE_TIMEOUT_MS = "5000"
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES = [string]$BatchSize
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS = "100"
$env:LEARNER_BATCH_MAX_EPISODES = "1"
$env:MAIN_PER_ENABLE = "0"
$env:TRAIN_CHUNK_SIZE = [string][Math]::Min(128, [Math]::Max(32, $BatchSize))

function Quote-ExecArgValue([string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        return $Value
    }
    if ($Value -notmatch '\s') {
        return $Value
    }
    "'" + ($Value -replace "'", "'\''") + "'"
}

$execArgs = "--deck-list $(Quote-ExecArgValue $DeckList) --pairs $Pairs --batch-size $BatchSize --timeout-sec $TimeoutSec --post-train-wait-ms $PostTrainWaitMs --opponent $Opponent --cp7-skill $Cp7Skill --report-every $ReportEvery --workers $Workers --out $(Quote-ExecArgValue $outDir)"
if (-not [string]::IsNullOrWhiteSpace($AgentDeckList)) {
    $execArgs += " --agent-deck-list $(Quote-ExecArgValue $AgentDeckList)"
}
if (-not [string]::IsNullOrWhiteSpace($OppDeckList)) {
    $execArgs += " --opp-deck-list $(Quote-ExecArgValue $OppDeckList)"
}
if ($LineMode) {
    $execArgs += " --line-mode true --line-max-mulls $LineMaxMulls --line-bottom-combos $LineBottomCombos --line-margin-min $LineMarginMin --line-target-temperature $LineTargetTemperature --line-balance-mulligan-prompts $LineBalanceMulliganPrompts --line-bucket-soft-targets $LineBucketSoftTargets --line-balance-key $LineBalanceKey --line-balance-singleton-limit $LineBalanceSingletonLimit --line-train-bottoms $LineTrainBottoms --line-train-epochs $LineTrainEpochs"
}
if (-not [string]::IsNullOrWhiteSpace($ExportTrainingDataFile)) {
    $execArgs += " --export-training-data-file $(Quote-ExecArgValue $ExportTrainingDataFile)"
}
if (-not [string]::IsNullOrWhiteSpace($ExportWinningLineDataFile)) {
    $execArgs += " --export-winning-line-data-file $(Quote-ExecArgValue $ExportWinningLineDataFile)"
}
if ($CollectOnly) {
    $execArgs += " --collect-only true"
}

& mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.MulliganCounterfactualTrainer" `
    "-Dexec.args=$execArgs"
if ($LASTEXITCODE -ne 0) {
    throw "mulligan counterfactual trainer failed with exit code $LASTEXITCODE"
}

if ($LineMode) {
    Write-Host "London line counterfactual wrapper output: $outDir"
} else {
    Write-Host "Mulligan counterfactual output: $outDir"
}
