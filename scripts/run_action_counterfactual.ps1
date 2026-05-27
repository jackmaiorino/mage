param(
    [string]$Profile = "Pauper-Generalist-Value-v2",
    [string]$AgentDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.spy_combo.txt",
    [string]$OppDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt",
    [string]$AgentOpeningHand = "",
    [string]$AgentOpeningHandPool = "",
    [string]$AgentOpeningHandPoolFile = "",
    [string]$OppOpeningHand = "",
    [string]$OppOpeningHandPool = "",
    [string]$OppOpeningHandPoolFile = "",
    [int]$Scenarios = 32,
    [int]$BatchSize = 32,
    [int]$TimeoutSec = 60,
    [int]$ScenarioTimeoutSec = 0,
    [int]$MaxGameTurns = 0,
    [int]$PostTrainWaitMs = 300000,
    [string]$Opponent = "rl",
    [int]$Cp7Skill = 7,
    [string]$ServiceMode = "local",
    [string]$RunId = "",
    [int]$ModelDModel = 128,
    [int]$ModelNumLayers = 2,
    [long]$Seed = 0,
    [int]$AiThreads = 24,
    [int]$ReportEvery = 4,
    [int]$Workers = 8,
    [int]$MaxDecisionDepth = 6,
    [int]$TopK = 4,
    [int]$RandomExtra = 1,
    [int]$TrainEpochs = 4,
    [int]$TrainChunkSize = -1,
    [int]$CandidatePermutations = 1,
    [int]$MaxTrainExamples = 0,
    [switch]$PolicyMissOnly,
    [int]$StopAfterExamples = 0,
    [string]$ExportTrainingDataFile = "",
    [string]$ExportTrajectoryDataFile = "",
    [string]$ImportTrainingDataPath = "",
    [string]$ImportTrajectoryDataPath = "",
    [switch]$ImportFlatAsTerminalEpisodes,
    [string]$ScoreTrainingDataPath = "",
    [int]$ScoreMaxExamples = 0,
    [switch]$FitScoreProbe,
    [switch]$CollectOnly,
    [switch]$WinningPrefixMode,
    [int]$MaxPrefixDepth = 6,
    [int]$TrainPrefixDepth = 6,
    [int]$MaxSearchNodes = 64,
    [int]$BranchSubtreeSearchNodes = 0,
    [int]$MaxWinningPrefixesPerScenario = 1,
    [switch]$TacticAutopilot,
    [switch]$PrefixSiblingContrast,
    [int]$PrefixSiblingContrastSearchNodes = 0,
    [switch]$TrainRootMulliganOnNoWin,
    [switch]$TrainRootMulliganOnly,
    [double]$TargetTemperature = 0.25,
    [double]$WinTurnBonus = 1.00,
    [double]$LossTurnBonus = 0.00,
    [double]$TrajectoryFinalReward = 1.00,
    [double]$PolicyLossCoef = 0.0,
    [double]$ValueLossCoef = 0.0,
    [double]$ValuePairRankLossCoef = 0.0,
    [double]$ValuePairRankMargin = 0.20,
    [double]$PolicyPairRankLossCoef = 0.0,
    [double]$PolicyPairRankMargin = 0.20,
    [double]$TrajectoryPairRankLossCoef = 0.0,
    [double]$TrajectoryPairRankMargin = 0.20,
    [double]$ReferencePolicyKlCoef = 0.0,
    [string]$MctsReferenceModelPath = "",
    [switch]$ResetTrainingStateOnLoad,
    [switch]$SkipOptimizerStateLoad,
    [double]$ActionPairRankLossCoef = 0.0,
    [double]$ActionPairRankMargin = 0.20,
    [double]$ActionPairRankMinGap = 0.25,
    [switch]$ValuePairRankCriticOnly,
    [switch]$UseMcReturns,
    [double]$MctsKlLossCoef = 3.0,
    [double]$CandidateQLossCoef = 0.0,
    [switch]$CandidateQOnly,
    [switch]$CandidateQCriticalOnly,
    [double]$CandidateQHuberBeta = 0.25,
    [switch]$CandidateQFromMctsTargets,
    [double]$BranchReturnPolicyLossCoef = 0.0,
    [double]$BranchReturnPolicyTemperature = 0.50,
    [double]$BranchReturnPolicyMinGap = 0.25,
    [double]$BranchReturnPolicyTargetMix = 1.0,
    [Alias("CandidateQBranchReturns")]
    [switch]$BranchReturnTargets,
    [switch]$BranchReturnBalance,
    [int]$BranchReturnMaxNegativesPerPositive = 1,
    [switch]$DirectBcLoss,
    [double]$DirectBcLossCoef = 1.0,
    [switch]$DistillHeadOnly,
    [switch]$DistillPolicyPathOnly,
    [switch]$HardenBinaryTargets,
    [switch]$BalanceBinaryTargets,
    [string]$TerminalMode = "WIN",
    [string]$ActionTypes = "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    [switch]$SkipPassTraining,
    [switch]$SkipBlankTraining,
    [switch]$SkipMulliganTraining,
    [switch]$SkipPassBest,
    [switch]$GenericBranchOrder,
    [switch]$NoSearchModelScoring,
    [string]$IncludeActionTextRegex = "",
    [string]$AvoidLosingActionTextRegex = "",
    [switch]$AvoidLosingStrictNegative,
    [switch]$AvoidLosingMaskBaselineOnly,
    [switch]$BaselineLosingAlternativeOnly,
    [switch]$BranchValueProbe,
    [switch]$BranchTrajectoryMode,
    [switch]$BranchTrajectoryFirstPostTargetOnly,
    [switch]$BranchTrajectoryPairMode,
    [switch]$BranchTrajectoryRequireTrainingExample,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
$outDir = Join-Path $repo ("local-training\local_pbt\action_counterfactual\" + $RunId)
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
$env:AI_MAX_THREADS_FOR_SIMULATIONS = [string]$AiThreads
$env:RL_RANDOM_DECISIONS = "0"
$env:RL_MULLIGAN_TRACE_JSONL = "0"
$env:RL_HEURISTIC_STEP_REWARDS = "0"
$env:RL_SKIP_SIM_VALIDATION = "0"
$env:MULLIGAN_HARD_OVERRIDES_ENABLE = "0"
$env:GAME_LOG_FREQUENCY = "0"
$env:TRAIN_LOG_EVERY = "25"
$env:CUDA_MEM_FRACTION = "0.70"

# This pass is terminal-only branch distillation on action policy targets.
$env:LOSS_SCHEDULE_ENABLE = "0"
$env:FREEZE_ENCODER_IN_WARMUP = "0"
$env:POLICY_LOSS_COEF = [string]$PolicyLossCoef
$env:POLICY_LOSS_COEF_WARMUP = [string]$PolicyLossCoef
$env:VALUE_LOSS_COEF = [string]$ValueLossCoef
$env:VALUE_LOSS_COEF_WARMUP = [string]$ValueLossCoef
$env:VALUE_PAIR_RANK_LOSS_COEF = [string]$ValuePairRankLossCoef
$env:VALUE_PAIR_RANK_MARGIN = [string]$ValuePairRankMargin
$env:POLICY_PAIR_RANK_LOSS_COEF = [string]$PolicyPairRankLossCoef
$env:POLICY_PAIR_RANK_MARGIN = [string]$PolicyPairRankMargin
$env:TRAJECTORY_PAIR_RANK_LOSS_COEF = [string]$TrajectoryPairRankLossCoef
$env:TRAJECTORY_PAIR_RANK_MARGIN = [string]$TrajectoryPairRankMargin
$env:REFERENCE_POLICY_KL_COEF = [string]$ReferencePolicyKlCoef
$env:MCTS_REFERENCE_MODEL_PATH = $MctsReferenceModelPath
$env:RESET_TRAINING_STATE_ON_LOAD = if ($ResetTrainingStateOnLoad) { "1" } else { "0" }
$env:LOAD_OPTIMIZER_STATE = if ($SkipOptimizerStateLoad) { "0" } else { "1" }
$env:ACTION_PAIR_RANK_LOSS_COEF = [string]$ActionPairRankLossCoef
$env:ACTION_PAIR_RANK_MARGIN = [string]$ActionPairRankMargin
$env:ACTION_PAIR_RANK_MIN_GAP = [string]$ActionPairRankMinGap
$env:VALUE_PAIR_RANK_CRITIC_ONLY = if ($ValuePairRankCriticOnly) { "1" } else { "0" }
$env:ENTROPY_LOSS_MULT = "0"
$env:ENTROPY_LOSS_MULT_WARMUP = "0"
$env:USE_GAE = if ($UseMcReturns) { "0" } else { "1" }
$env:BELIEF_LOSS_COEF = "0"
$effectiveMctsKlLossCoef = if ($BranchReturnTargets) { 0.0 } else { $MctsKlLossCoef }
$env:MCTS_KL_LOSS_COEF = [string]$effectiveMctsKlLossCoef
$env:CANDIDATE_Q_LOSS_COEF = [string]$CandidateQLossCoef
$env:CANDIDATE_Q_ONLY = if ($CandidateQOnly) { "1" } else { "0" }
$env:CANDIDATE_Q_CRITICAL_ONLY = if ($CandidateQCriticalOnly) { "1" } else { "0" }
$env:CANDIDATE_Q_HUBER_BETA = [string]$CandidateQHuberBeta
$env:CANDIDATE_Q_FROM_MCTS_TARGETS = if ($CandidateQFromMctsTargets -or $BranchReturnTargets) { "1" } else { "0" }
$env:CANDIDATE_Q_MCTS_SIGNED_TARGETS = if ($BranchReturnTargets) { "1" } else { "0" }
$env:BRANCH_RETURN_POLICY_LOSS_COEF = [string]$BranchReturnPolicyLossCoef
$env:BRANCH_RETURN_POLICY_TEMPERATURE = [string]$BranchReturnPolicyTemperature
$env:BRANCH_RETURN_POLICY_MIN_GAP = [string]$BranchReturnPolicyMinGap
$env:BRANCH_RETURN_POLICY_TARGET_MIX = [string]$BranchReturnPolicyTargetMix
$env:BC_DIRECT_LOSS = if ($DirectBcLoss) { "1" } else { "0" }
$env:BC_DIRECT_LOSS_COEF = [string]$DirectBcLossCoef
$env:DISTILL_HEAD_ONLY = if ($DistillHeadOnly) { "1" } else { "0" }
$env:DISTILL_POLICY_PATH_ONLY = if ($DistillPolicyPathOnly) { "1" } else { "0" }
$env:BC_HARDEN_BINARY_TARGETS = if ($HardenBinaryTargets) { "1" } else { "0" }
$env:BC_BALANCE_BINARY_TARGETS = if ($BalanceBinaryTargets) { "1" } else { "0" }
$env:BC_HARDEN_BINARY_DIAG_FILE = (Join-Path $outDir "bc_harden_binary_diag.log")
$env:MULLIGAN_SAMPLE_WEIGHT = "0.0"
$env:LONDON_MULLIGAN_SAMPLE_WEIGHT = "0.0"

$env:PY_BATCH_MAX_SIZE = [string]$BatchSize
$env:PY_BATCH_TIMEOUT_MS = "5"
$env:PY_SCORE_TIMEOUT_MS = "5000"
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES = [string]$BatchSize
$env:GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS = "100"
$env:LEARNER_BATCH_MAX_EPISODES = "1"
$env:MAIN_PER_ENABLE = "0"
if ($TrainChunkSize -ge 0) {
    $env:TRAIN_CHUNK_SIZE = [string]$TrainChunkSize
} else {
    $env:TRAIN_CHUNK_SIZE = [string][Math]::Min(128, [Math]::Max(32, $BatchSize))
}
$estimatedTrainItems = [Math]::Ceiling(($Scenarios * [Math]::Max(1, $MaxDecisionDepth) * [Math]::Max(1, $TrainEpochs)) / [double][Math]::Max(1, $BatchSize))
$env:TRAIN_QUEUE_MAX_EPISODES = [string][Math]::Max(4096, [int]($estimatedTrainItems * 2))
$env:TRAIN_QUEUE_DROP_ON_FULL = "0"
$env:TRAIN_QUEUE_OFFER_TIMEOUT_MS = "0"

function Quote-ExecArgValue([string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        return $Value
    }
    if ($Value -notmatch '\s') {
        return $Value
    }
    "'" + ($Value -replace "'", "'\''") + "'"
}

$execArgs = @(
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
    "--max-decision-depth=$MaxDecisionDepth",
    "--top-k=$TopK",
    "--random-extra=$RandomExtra",
    "--train-epochs=$TrainEpochs",
    "--candidate-permutations=$CandidatePermutations",
    "--max-train-examples=$MaxTrainExamples",
    "--policy-miss-only=$([bool]$PolicyMissOnly)",
    "--stop-after-examples=$StopAfterExamples",
    $(if (-not [string]::IsNullOrWhiteSpace($ExportTrainingDataFile)) { "--export-training-data-file=$(Quote-ExecArgValue $ExportTrainingDataFile)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ExportTrajectoryDataFile)) { "--export-trajectory-data-file=$(Quote-ExecArgValue $ExportTrajectoryDataFile)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ImportTrainingDataPath)) { "--import-training-data-path=$(Quote-ExecArgValue $ImportTrainingDataPath)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($ImportTrajectoryDataPath)) { "--import-trajectory-data-path=$(Quote-ExecArgValue $ImportTrajectoryDataPath)" } else { $null }),
    "--import-flat-as-terminal-episodes=$([bool]$ImportFlatAsTerminalEpisodes)",
    $(if (-not [string]::IsNullOrWhiteSpace($ScoreTrainingDataPath)) { "--score-training-data-path=$(Quote-ExecArgValue $ScoreTrainingDataPath)" } else { $null }),
    $(if ($ScoreMaxExamples -gt 0) { "--score-max-examples=$ScoreMaxExamples" } else { $null }),
    "--fit-score-probe=$([bool]$FitScoreProbe)",
    "--collect-only=$([bool]$CollectOnly)",
    "--winning-prefix-mode=$([bool]$WinningPrefixMode)",
    "--max-prefix-depth=$MaxPrefixDepth",
    "--train-prefix-depth=$TrainPrefixDepth",
    "--max-search-nodes=$MaxSearchNodes",
    "--branch-subtree-search-nodes=$BranchSubtreeSearchNodes",
    "--max-winning-prefixes-per-scenario=$MaxWinningPrefixesPerScenario",
    "--tactic-autopilot=$([bool]$TacticAutopilot)",
    "--prefix-sibling-contrast=$([bool]$PrefixSiblingContrast)",
    "--prefix-sibling-contrast-search-nodes=$PrefixSiblingContrastSearchNodes",
    "--train-root-mulligan-on-no-win=$([bool]$TrainRootMulliganOnNoWin)",
    "--train-root-mulligan-only=$([bool]$TrainRootMulliganOnly)",
    "--target-temperature=$TargetTemperature",
    "--win-turn-bonus=$WinTurnBonus",
    "--loss-turn-bonus=$LossTurnBonus",
    "--trajectory-final-reward=$TrajectoryFinalReward",
    "--branch-return-targets=$([bool]$BranchReturnTargets)",
    "--branch-return-balance=$([bool]$BranchReturnBalance)",
    "--branch-return-max-negatives-per-positive=$BranchReturnMaxNegativesPerPositive",
    "--terminal-mode=$TerminalMode",
    "--skip-pass-training=$([bool]$SkipPassTraining)",
    "--skip-blank-training=$([bool]$SkipBlankTraining)",
    "--skip-mulligan-training=$([bool]$SkipMulliganTraining)",
    "--skip-pass-best=$([bool]$SkipPassBest)",
    "--generic-branch-order=$([bool]$GenericBranchOrder)",
    "--no-search-model-scoring=$([bool]$NoSearchModelScoring)",
    $(if (-not [string]::IsNullOrWhiteSpace($IncludeActionTextRegex)) { "--include-action-text-regex=$(Quote-ExecArgValue $IncludeActionTextRegex)" } else { $null }),
    $(if (-not [string]::IsNullOrWhiteSpace($AvoidLosingActionTextRegex)) { "--avoid-losing-action-text-regex=$(Quote-ExecArgValue $AvoidLosingActionTextRegex)" } else { $null }),
    "--avoid-losing-strict-negative=$([bool]$AvoidLosingStrictNegative)",
    "--avoid-losing-mask-baseline-only=$([bool]$AvoidLosingMaskBaselineOnly)",
    "--baseline-losing-alternative-only=$([bool]$BaselineLosingAlternativeOnly)",
    "--branch-value-probe=$([bool]$BranchValueProbe)",
    "--branch-trajectory-mode=$([bool]$BranchTrajectoryMode)",
    "--branch-trajectory-first-post-target-only=$([bool]$BranchTrajectoryFirstPostTargetOnly)",
    "--branch-trajectory-pair-mode=$([bool]$BranchTrajectoryPairMode)",
    "--branch-trajectory-require-training-example=$([bool]$BranchTrajectoryRequireTrainingExample)",
    "--action-types=$ActionTypes",
    "--out=$(Quote-ExecArgValue $outDir)"
) -join " "

& mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java `
    "-Dexec.mainClass=mage.player.ai.rl.ActionCounterfactualTrainer" `
    "-Dexec.args=$execArgs"
if ($LASTEXITCODE -ne 0) {
    throw "action counterfactual trainer failed with exit code $LASTEXITCODE"
}

Write-Host "Action counterfactual output: $outDir"
