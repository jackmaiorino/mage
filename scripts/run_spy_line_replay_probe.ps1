param(
    [string]$Profile = "Pauper-Spy-Combo-LineSearch-v1",
    [string]$ReplayFile,
    [string]$AgentDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.spy_combo.txt",
    [string]$OppDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt",
    [int]$TimeoutSec = 75,
    [string]$ServiceMode = "local",
    [string]$RunId = "",
    [int]$ReportEvery = 2,
    [int]$Workers = 12,
    [int]$AiThreads = 24,
    [int]$ModelDModel = 128,
    [int]$ModelNumLayers = 2,
    [int]$MaxDecisionDepth = 8,
    [int]$ReplayMaxScenarios = 0,
    [string]$Opponent = "rl",
    [int]$Cp7Skill = 7,
    [string]$ReplayDeviationTrainingDataFile = "",
    [string]$ReplayDaggerTrainingDataFile = "",
    [int]$ReplayDeviationRepeat = 1,
    [string]$OpponentTranscriptFile = "",
    [switch]$ForceOpponentTranscript,
    [int]$OpponentDecisionMaxSourceTurn = 0,
    [string]$ActionTypes = "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    [switch]$ForcedPrefixReplay,
    [switch]$CheckpointBranchProbe,
    [switch]$FilterPriorityManaActions,
    [switch]$TokenMetadataRngIsolation,
    [switch]$TokenMetadataRngTrace,
    [switch]$PolicyInputDump,
    [switch]$PolicyInferenceProbe,
    [switch]$PythonInferenceDuplicateProbe,
    [switch]$ModelLoadDeterminismGate,
    [switch]$FailOnSkippedIncompatible,
    [switch]$RandomUtilWrapperTrace,
    [switch]$RandomUtilDirectTrace,
    [switch]$StackPriorityTrace,
    [switch]$Cp7TokenMetadataParity,
    [string]$PyGlobalSeed = "",
    [switch]$MavenOffline,
    [switch]$CompileInExecInvocation,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ReplayFile)) {
    throw "-ReplayFile is required"
}

if ($ForceOpponentTranscript -and $Opponent.Trim().ToLowerInvariant() -ne "cp7") {
    throw "-ForceOpponentTranscript requires -Opponent cp7; current -Opponent '$Opponent' would not replay the transcript."
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReplayFile = (Resolve-Path $ReplayFile).Path

$allowedActionTypes = @{}
foreach ($t in ($ActionTypes -split ",")) {
    $key = $t.Trim().ToUpperInvariant()
    if (-not [string]::IsNullOrWhiteSpace($key)) {
        $allowedActionTypes[$key] = $true
    }
}
$csvActionTypes = @()
foreach ($row in (Import-Csv -Path $ReplayFile)) {
    if ($row.PSObject.Properties.Name -contains "action_type") {
        $key = ([string]$row.action_type).Trim().ToUpperInvariant()
        if (-not [string]::IsNullOrWhiteSpace($key)) {
            $csvActionTypes += $key
        }
    }
}
$missingActionTypes = @($csvActionTypes | Sort-Object -Unique | Where-Object { -not $allowedActionTypes.ContainsKey($_) })
if ($missingActionTypes.Count -gt 0) {
    throw ("ReplayFile action_type values are not covered by -ActionTypes: {0}. Pass -ActionTypes that includes every replay CSV action_type." -f ($missingActionTypes -join ","))
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$outDir = Join-Path $repo ("local-training\local_pbt\spy_line_replay\" + $RunId)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$mavenCommonArgs = @()
if ($MavenOffline) {
    $mavenCommonArgs += "-o"
}

if (-not $SkipCompile -and -not $CompileInExecInvocation) {
    & mvn @mavenCommonArgs -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
    if ($LASTEXITCODE -ne 0) {
        throw "compile failed with exit code $LASTEXITCODE"
    }
}

$env:MODEL_PROFILE = $Profile
$env:PY_SERVICE_MODE = $ServiceMode
$env:PY_BACKEND_MODE = "single"
$env:MODEL_D_MODEL = [string]$ModelDModel
$env:MODEL_NUM_LAYERS = [string]$ModelNumLayers
$env:RL_RANDOM_DECISIONS = "0"
$env:RL_MULLIGAN_TRACE_JSONL = "0"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:RL_SKIP_SIM_VALIDATION = "0"
$env:MULLIGAN_HARD_OVERRIDES_ENABLE = "0"
$env:RL_FILTER_PRIORITY_MANA_ACTIONS = if ($FilterPriorityManaActions) { "1" } else { "0" }
if ($TokenMetadataRngIsolation) {
    $env:EVAL_REPLAY_TOKEN_METADATA_RNG_ISOLATION = "1"
}
if ($TokenMetadataRngTrace) {
    $env:EVAL_REPLAY_TOKEN_METADATA_RNG_TRACE = "1"
}
if ($PolicyInputDump) {
    $env:EVAL_REPLAY_POLICY_INPUT_DUMP = "1"
}
if ($PolicyInferenceProbe) {
    $env:EVAL_REPLAY_POLICY_INFERENCE_PROBE = "1"
}
if ($PythonInferenceDuplicateProbe) {
    $env:EVAL_REPLAY_PYTHON_INFERENCE_DUPLICATE_PROBE = "1"
    $env:EVAL_REPLAY_PYTHON_INFERENCE_DUPLICATE_PROBE_FILE = Join-Path $outDir "python_inference_duplicate_probe.csv"
}
if ($ModelLoadDeterminismGate) {
    $env:EVAL_REPLAY_MODEL_LOAD_DETERMINISM_GATE = "1"
    $env:EVAL_REPLAY_MODEL_LOAD_DETERMINISM_GATE_FILE = Join-Path $outDir "python_model_load_determinism_gate.csv"
}
if ($FailOnSkippedIncompatible) {
    $env:EVAL_REPLAY_FAIL_ON_SKIPPED_INCOMPATIBLE = "1"
}
if ($RandomUtilWrapperTrace) {
    $env:EVAL_RANDOM_UTIL_WRAPPER_TRACE_JSON = "1"
    $env:EVAL_RANDOM_UTIL_WRAPPER_TRACE_FILE = Join-Path $outDir "random_util_wrapper_trace.log"
}
if ($RandomUtilDirectTrace) {
    $env:EVAL_RANDOM_UTIL_DIRECT_TRACE_JSON = "1"
    $env:EVAL_RANDOM_UTIL_DIRECT_TRACE_FILE = Join-Path $outDir "random_util_direct_trace.log"
}
if ($StackPriorityTrace) {
    $env:EVAL_REPLAY_STACK_PRIORITY_TRACE_JSON = "1"
    $env:EVAL_REPLAY_STACK_PRIORITY_TRACE_FILE = Join-Path $outDir "stack_priority_trace.log"
}
if ($Cp7TokenMetadataParity) {
    $env:EVAL_REPLAY_CP7_TOKEN_METADATA_PARITY = "1"
}
if (-not [string]::IsNullOrWhiteSpace($PyGlobalSeed)) {
    $env:PY_GLOBAL_SEED = $PyGlobalSeed
}
$env:GAME_LOG_FREQUENCY = "0"
$env:TRAIN_LOG_EVERY = "0"
$env:CUDA_MEM_FRACTION = "0.70"
$env:AI_MAX_THREADS_FOR_SIMULATIONS = [string]$AiThreads
$env:PY_BATCH_MAX_SIZE = "64"
$env:PY_BATCH_TIMEOUT_MS = "5"
$env:PY_SCORE_TIMEOUT_MS = "5000"
if ([string]::IsNullOrWhiteSpace($env:EVAL_OPPONENT_DECISION_FILE)) {
    $env:EVAL_OPPONENT_DECISION_FILE = Join-Path $outDir "opponent_decisions.log"
}
if ($OpponentDecisionMaxSourceTurn -gt 0) {
    $env:EVAL_OPPONENT_DECISION_MAX_SOURCE_TURN = [string]$OpponentDecisionMaxSourceTurn
}
if ($ForceOpponentTranscript) {
    $env:EVAL_REPLAY_FORCE_OPPONENT_TRANSCRIPT = "1"
    $env:EVAL_REPLAY_OPPONENT_TRANSCRIPT_MISMATCH_FILE = Join-Path $outDir "opponent_transcript_mismatch.csv"
}
if (-not [string]::IsNullOrWhiteSpace($OpponentTranscriptFile)) {
    $OpponentTranscriptFile = (Resolve-Path $OpponentTranscriptFile).Path
    $env:EVAL_REPLAY_OPPONENT_TRANSCRIPT_FILE = $OpponentTranscriptFile
}
if ([string]::IsNullOrWhiteSpace($env:EVAL_AGENT_SEARCH_TRACE_FILE)) {
    $env:EVAL_AGENT_SEARCH_TRACE_FILE = Join-Path $outDir "agent_search_trace.log"
}

$execArgsList = @(
    "--agent-deck-list=$AgentDeckList",
    "--opp-deck-list=$OppDeckList",
    "--timeout-sec=$TimeoutSec",
    "--report-every=$ReportEvery",
    "--workers=$Workers",
    "--max-decision-depth=$MaxDecisionDepth",
    "--replay-file=$ReplayFile",
    "--replay-max-scenarios=$ReplayMaxScenarios",
    "--opponent=$Opponent",
    "--cp7-skill=$Cp7Skill",
    "--forced-prefix-replay=$([bool]$ForcedPrefixReplay)",
    "--checkpoint-branch-probe=$([bool]$CheckpointBranchProbe)",
    "--force-opponent-transcript=$([bool]$ForceOpponentTranscript)",
    "--policy-input-dump=$([bool]$PolicyInputDump)",
    "--policy-inference-probe=$([bool]$PolicyInferenceProbe)",
    "--action-types=$ActionTypes",
    "--out=$outDir"
)
if (-not [string]::IsNullOrWhiteSpace($ReplayDeviationTrainingDataFile)) {
    $execArgsList += "--replay-deviation-training-data-file=$ReplayDeviationTrainingDataFile"
}
if (-not [string]::IsNullOrWhiteSpace($ReplayDaggerTrainingDataFile)) {
    $execArgsList += "--replay-dagger-training-data-file=$ReplayDaggerTrainingDataFile"
    $execArgsList += "--replay-deviation-repeat=$ReplayDeviationRepeat"
}
if (-not [string]::IsNullOrWhiteSpace($OpponentTranscriptFile)) {
    $execArgsList += "--opponent-transcript-file=$OpponentTranscriptFile"
}
$execArgs = $execArgsList -join " "
$execGoals = @()
if (-not $SkipCompile -and $CompileInExecInvocation) {
    $execGoals += "compile"
}
$execGoals += "exec:java"

& mvn @mavenCommonArgs -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests @execGoals `
    "-Dexec.mainClass=mage.player.ai.rl.ActionCounterfactualTrainer" `
    "-Dexec.args=$execArgs"
if ($LASTEXITCODE -ne 0) {
    throw "Spy line replay probe failed with exit code $LASTEXITCODE"
}

Write-Host "Spy line replay output: $outDir"
