param(
    [string]$Profile,
    [string]$RunId,
    [string]$DataDir,
    [string]$RegistryPath,
    [int]$Trajectories,
    [int]$TrainEpochs = 4,
    [int]$TrainBatchSize = 512,
    [string]$ActionTypes = "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    [double]$MctsKlLossCoef = 3.0,
    [switch]$DirectBcLoss,
    [double]$DirectBcLossCoef = 1.0,
    [double]$MulliganSampleWeight = 1.0,
    [double]$LondonMulliganSampleWeight = 1.0,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Profile)) {
    throw "-Profile is required"
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    throw "-RunId is required"
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    throw "-DataDir is required"
}
if ([string]::IsNullOrWhiteSpace($RegistryPath)) {
    throw "-RegistryPath is required"
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataPath = (Resolve-Path $DataDir).Path
$registry = (Resolve-Path $RegistryPath).Path

if (-not $SkipCompile) {
    & mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
    if ($LASTEXITCODE -ne 0) {
        throw "compile failed with exit code $LASTEXITCODE"
    }
}

$trainRunId = "$RunId`_offline_train_$Trajectories"
$trainArgs = @{
    Profile = $Profile
    RunId = $trainRunId
    ImportTrainingDataPath = $dataPath
    Scenarios = 1
    BatchSize = $TrainBatchSize
    ModelDModel = 128
    ModelNumLayers = 2
    TrainEpochs = $TrainEpochs
    CandidatePermutations = 1
    ActionTypes = $ActionTypes
    MctsKlLossCoef = $MctsKlLossCoef
    DirectBcLossCoef = $DirectBcLossCoef
    MulliganSampleWeight = $MulliganSampleWeight
    LondonMulliganSampleWeight = $LondonMulliganSampleWeight
    PostTrainWaitMs = 3600000
    SkipCompile = $true
}
if ($DirectBcLoss) {
    $trainArgs.DirectBcLoss = $true
}
& ./scripts/run_spy_line_search.ps1 @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "offline BC import training failed with exit code $LASTEXITCODE"
}

$evalRunId = "$RunId`_cp1_$Trajectories"
& py -3 scripts/run_cp7_eval_sweep.py `
    --registry $registry `
    --profiles $Profile `
    --skill 1 `
    --games-per-matchup 32 `
    --games-per-job 8 `
    --parallel 4 `
    --ai-threads 8 `
    --run-id $evalRunId `
    --skip-compile
if ($LASTEXITCODE -ne 0) {
    throw "offline BC eval failed with exit code $LASTEXITCODE"
}

Write-Host "spy_imitation_offline_train_eval_done run=$RunId profile=$Profile trajectories=$Trajectories data=$dataPath"
