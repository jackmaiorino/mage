param(
    [string]$Profile = "Pauper-Spy-Combo-LineSearch-v1",
    [string]$AgentDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.spy_combo.txt",
    [string]$OppDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt",
    [string]$AgentOpeningHand = "",
    [string]$AgentOpeningHandPool = "",
    [string]$AgentOpeningHandPoolFile = "",
    [string]$OppOpeningHand = "",
    [string]$OppOpeningHandPool = "",
    [string]$OppOpeningHandPoolFile = "",
    [int]$Scenarios = 24,
    [int]$BatchSize = 32,
    [int]$TimeoutSec = 75,
    [int]$ScenarioTimeoutSec = 0,
    [int]$MaxGameTurns = 0,
    [int]$PostTrainWaitMs = 300000,
    [string]$Opponent = "cp7",
    [int]$Cp7Skill = 7,
    [string]$ServiceMode = "local",
    [string]$RunId = "",
    [int]$ModelDModel = 128,
    [int]$ModelNumLayers = 2,
    [long]$Seed = 0,
    [int]$ReportEvery = 2,
    [int]$Workers = 12,
    [int]$AiThreads = 24,
    [int]$MaxDecisionDepth = 0,
    [int]$MaxPrefixDepth = 6,
    [int]$TrainPrefixDepth = 6,
    [int]$MaxSearchNodes = 31,
    [int]$MaxWinningPrefixesPerScenario = 1,
    [string]$InitialPrefix = "",
    [string]$PassMacroDepths = "",
    [int]$TopK = 2,
    [int]$RandomExtra = 0,
    [int]$TrainEpochs = 4,
    [int]$CandidatePermutations = 1,
    [int]$MaxTrainExamples = 0,
    [double]$MinTargetMargin = 0.0,
    [switch]$PolicyMissOnly,
    [int]$StopAfterExamples = 0,
    [int]$StopAfterWinningTrajectories = 0,
    [string]$ExportTrainingDataFile = "",
    [string]$ExportTrajectoryDataFile = "",
    [string]$ImportTrainingDataPath = "",
    [string]$ImportTrajectoryDataPath = "",
    [switch]$ImportFlatAsTerminalEpisodes,
    [string]$ScoreTrainingDataPath = "",
    [int]$ScoreMaxExamples = 0,
    [switch]$FitScoreProbe,
    [switch]$TrajectoryRlLoss,
    [double]$TrajectoryFinalReward = 1.0,
    [string]$TerminalMode = "WIN",
    [double]$MctsKlLossCoef = 3.0,
    [double]$MctsTargetPolicyMix = 1.0,
    [string]$MctsReferenceModelPath = "",
    [switch]$DirectBcLoss,
    [double]$DirectBcLossCoef = 1.0,
    [switch]$HardenBinaryTargets,
    [switch]$BalanceBinaryTargets,
    [double]$MulliganSampleWeight = 1.0,
    [double]$LondonMulliganSampleWeight = 1.0,
    [string]$ActionTypes = "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    [switch]$PrefixSiblingContrast,
    [int]$PrefixSiblingContrastSearchNodes = 0,
    [switch]$TrainRootMulliganOnNoWin,
    [switch]$TrainRootMulliganOnly,
    [switch]$GenericBranchOrder,
    [switch]$TacticAutopilot,
    [switch]$NoSearchModelScoring,
    [switch]$CollectOnly,
    [switch]$RandomDecisions,
    [switch]$DistillHeadOnly,
    [switch]$DistillPolicyPathOnly,
    [switch]$FilterPriorityManaActions,
    [switch]$SkipPassTraining,
    [switch]$SkipBlankTraining,
    [switch]$SkipMulliganTraining,
    [switch]$BaselineLosingAlternativeOnly,
    [string]$IncludeActionTextRegex = "",
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$outDir = Join-Path $repo ("local-training\local_pbt\spy_line_search\" + $RunId)
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
$env:RL_FILTER_PRIORITY_MANA_ACTIONS = if ($FilterPriorityManaActions) { "1" } else { "0" }
$env:GAME_LOG_FREQUENCY = "0"
$env:TRAIN_LOG_EVERY = "25"
$env:CUDA_MEM_FRACTION = "0.70"
$env:AI_MAX_THREADS_FOR_SIMULATIONS = [string]$AiThreads

# Terminal-only prefix distillation: search supplies mctsVisitTargets for winning prefixes only.
$env:LOSS_SCHEDULE_ENABLE = "0"
$env:FREEZE_ENCODER_IN_WARMUP = "0"
if ($TrajectoryRlLoss) {
    $env:POLICY_LOSS_COEF = "1.0"
    $env:POLICY_LOSS_COEF_WARMUP = "1.0"
    $env:VALUE_LOSS_COEF = "0.5"
    $env:VALUE_LOSS_COEF_WARMUP = "0.5"
    $env:ENTROPY_LOSS_MULT = "0.005"
    $env:ENTROPY_LOSS_MULT_WARMUP = "0.005"
    $env:USE_PPO = "1"
} else {
    $env:POLICY_LOSS_COEF = "0"
    $env:POLICY_LOSS_COEF_WARMUP = "0"
    $env:VALUE_LOSS_COEF = "0"
    $env:VALUE_LOSS_COEF_WARMUP = "0"
    $env:ENTROPY_LOSS_MULT = "0"
    $env:ENTROPY_LOSS_MULT_WARMUP = "0"
}
$env:BELIEF_LOSS_COEF = "0"
$env:MCTS_KL_LOSS_COEF = if ($TrajectoryRlLoss) { "0" } else { [string]$MctsKlLossCoef }
$env:MCTS_TARGET_POLICY_MIX = [string]$MctsTargetPolicyMix
if ([string]::IsNullOrWhiteSpace($MctsReferenceModelPath)) {
    Remove-Item Env:\MCTS_REFERENCE_MODEL_PATH -ErrorAction SilentlyContinue
} else {
    $env:MCTS_REFERENCE_MODEL_PATH = (Resolve-Path $MctsReferenceModelPath).Path
}
$env:BC_DIRECT_LOSS = if ($DirectBcLoss) { "1" } else { "0" }
$env:BC_DIRECT_LOSS_COEF = [string]$DirectBcLossCoef
$env:BC_HARDEN_BINARY_TARGETS = if ($HardenBinaryTargets) { "1" } else { "0" }
$env:BC_BALANCE_BINARY_TARGETS = if ($BalanceBinaryTargets) { "1" } else { "0" }
$env:DISTILL_HEAD_ONLY = if ($DistillHeadOnly) { "1" } else { "0" }
$env:DISTILL_POLICY_PATH_ONLY = if ($DistillPolicyPathOnly) { "1" } else { "0" }
$env:MULLIGAN_SAMPLE_WEIGHT = [string]$MulliganSampleWeight
$env:LONDON_MULLIGAN_SAMPLE_WEIGHT = [string]$LondonMulliganSampleWeight
$env:BC_HARDEN_BINARY_DIAG_FILE = (Join-Path $outDir "bc_harden_binary_diag.log")

$env:PY_BATCH_MAX_SIZE = [string]$BatchSize
$env:PY_BATCH_TIMEOUT_MS = "5"
$env:PY_SCORE_TIMEOUT_MS = "5000"
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES = [string]$BatchSize
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS = "100"
$env:LEARNER_BATCH_MAX_EPISODES = "1"
$env:MAIN_PER_ENABLE = "0"
$env:TRAIN_CHUNK_SIZE = [string][Math]::Min(128, [Math]::Max(32, $BatchSize))
$estimatedTrainItems = [Math]::Ceiling(($Scenarios * [Math]::Max(1, $TrainPrefixDepth) * [Math]::Max(1, $TrainEpochs)) / [double][Math]::Max(1, $BatchSize))
$env:TRAIN_QUEUE_MAX_EPISODES = [string][Math]::Max(4096, [int]($estimatedTrainItems * 2))
$env:TRAIN_QUEUE_DROP_ON_FULL = "0"
$env:TRAIN_QUEUE_OFFER_TIMEOUT_MS = "0"

$effectiveDecisionDepth = if ($MaxDecisionDepth -gt 0) {
    $MaxDecisionDepth
} else {
    [Math]::Max($MaxPrefixDepth, $TrainPrefixDepth)
}

function Quote-ExecArgValue([string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        return $Value
    }
    if ($Value -notmatch '\s') {
        return $Value
    }
    "'" + ($Value -replace "'", "'\''") + "'"
}

$execArgsList = @(
    "--agent-deck-list=$(Quote-ExecArgValue $AgentDeckList)",
    "--opp-deck-list=$(Quote-ExecArgValue $OppDeckList)",
    $(if (-not [string]::IsNullOrWhiteSpace($AgentOpeningHand)) { "--agent-opening-hand=$(Quote-ExecArgValue $AgentOpeningHand)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($AgentOpeningHandPool)) { "--agent-opening-hand-pool=$(Quote-ExecArgValue $AgentOpeningHandPool)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($AgentOpeningHandPoolFile)) { "--agent-opening-hand-pool-file=$(Quote-ExecArgValue $AgentOpeningHandPoolFile)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($OppOpeningHand)) { "--opp-opening-hand=$(Quote-ExecArgValue $OppOpeningHand)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($OppOpeningHandPool)) { "--opp-opening-hand-pool=$(Quote-ExecArgValue $OppOpeningHandPool)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($OppOpeningHandPoolFile)) { "--opp-opening-hand-pool-file=$(Quote-ExecArgValue $OppOpeningHandPoolFile)" } else { $null }),
    "--scenarios=$Scenarios",
    "--batch-size=$BatchSize",
    "--timeout-sec=$TimeoutSec",
    "--scenario-timeout-sec=$ScenarioTimeoutSec",
    "--max-game-turns=$MaxGameTurns",
    "--post-train-wait-ms=$PostTrainWaitMs",
    "--opponent=$Opponent",
    "--cp7-skill=$Cp7Skill",
    $(if ($Seed -gt 0) { "--seed=$Seed" } else { $null }),
    "--report-every=$ReportEvery",
    "--workers=$Workers",
    "--max-decision-depth=$effectiveDecisionDepth",
    "--winning-prefix-mode=true",
    "--max-prefix-depth=$MaxPrefixDepth",
    "--train-prefix-depth=$TrainPrefixDepth",
    "--max-search-nodes=$MaxSearchNodes",
    "--max-winning-prefixes-per-scenario=$MaxWinningPrefixesPerScenario",
    "--prefix-sibling-contrast=$([bool]$PrefixSiblingContrast)",
    "--prefix-sibling-contrast-search-nodes=$PrefixSiblingContrastSearchNodes",
    "--train-root-mulligan-on-no-win=$([bool]$TrainRootMulliganOnNoWin)",
    "--train-root-mulligan-only=$([bool]$TrainRootMulliganOnly)",
    "--generic-branch-order=$([bool]$GenericBranchOrder)",
    "--tactic-autopilot=$([bool]$TacticAutopilot)",
    "--no-search-model-scoring=$([bool]$NoSearchModelScoring)",
    $(if (-not [string]::IsNullOrWhiteSpace($InitialPrefix)) { "--initial-prefix=$InitialPrefix" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($PassMacroDepths)) { "--pass-macro-depths=$PassMacroDepths" } else { $null }),
    "--top-k=$TopK",
    "--random-extra=$RandomExtra",
    "--train-epochs=$TrainEpochs",
    "--candidate-permutations=$CandidatePermutations",
    "--max-train-examples=$MaxTrainExamples",
    "--min-target-margin=$MinTargetMargin",
    "--policy-miss-only=$([bool]$PolicyMissOnly)",
    "--stop-after-examples=$StopAfterExamples",
    "--stop-after-winning-trajectories=$StopAfterWinningTrajectories",
    "--trajectory-final-reward=$TrajectoryFinalReward",
    $(if (-not [string]::IsNullOrWhiteSpace($ExportTrainingDataFile)) { "--export-training-data-file=$(Quote-ExecArgValue $ExportTrainingDataFile)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ExportTrajectoryDataFile)) { "--export-trajectory-data-file=$(Quote-ExecArgValue $ExportTrajectoryDataFile)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ImportTrainingDataPath)) { "--import-training-data-path=$(Quote-ExecArgValue $ImportTrainingDataPath)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ImportTrajectoryDataPath)) { "--import-trajectory-data-path=$(Quote-ExecArgValue $ImportTrajectoryDataPath)" } else { $null }),
    "--import-flat-as-terminal-episodes=$([bool]$ImportFlatAsTerminalEpisodes)",
    $(if (-not [string]::IsNullOrWhiteSpace($ScoreTrainingDataPath)) { "--score-training-data-path=$(Quote-ExecArgValue $ScoreTrainingDataPath)" } else { $null }),
    $(if ($ScoreMaxExamples -gt 0) { "--score-max-examples=$ScoreMaxExamples" } else { $null }),
    "--fit-score-probe=$([bool]$FitScoreProbe)",
    "--collect-only=$([bool]$CollectOnly)",
    "--terminal-mode=$TerminalMode",
    "--skip-pass-training=$([bool]$SkipPassTraining)",
    "--skip-blank-training=$([bool]$SkipBlankTraining)",
    "--skip-mulligan-training=$([bool]$SkipMulliganTraining)",
    "--baseline-losing-alternative-only=$([bool]$BaselineLosingAlternativeOnly)",
    $(if (-not [string]::IsNullOrWhiteSpace($IncludeActionTextRegex)) { "--include-action-text-regex=$(Quote-ExecArgValue $IncludeActionTextRegex)" } else { $null }),
    "--action-types=$ActionTypes",
    "--out=$(Quote-ExecArgValue $outDir)"
)
$execArgs = ($execArgsList | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }) -join " "

& mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.ActionCounterfactualTrainer" `
    "-Dexec.args=$execArgs"
if ($LASTEXITCODE -ne 0) {
    throw "Spy line search trainer failed with exit code $LASTEXITCODE"
}

Write-Host "Spy line search output: $outDir"
