param(
    [string]$Profile = "Pauper-Spy-Combo-Imitation-Offline-20260503",
    [string]$RunId = "",
    [string]$AgentDeckList = "",
    [string]$OppDeckList = "",
    [int]$TargetTrajectories = 10000,
    [double]$MaxCollectHours = 14.0,
    [int]$BatchScenarios = 48,
    [int]$Workers = 48,
    [int]$AiThreads = 24,
    [int]$HandPoolCount = 512,
    [int]$Seed = 7101,
    [int]$TrainEpochs = 4,
    [int]$TrainBatchSize = 512,
    [int]$EvalGamesPerMatchup = 32,
    [int]$EvalGamesPerJob = 8,
    [int]$EvalParallel = 4,
    [int]$EvalAiThreads = 8,
    [int]$EvalSkill = 1,
    [string]$TerminalMode = "SPY_COMBO_MILESTONE_ONLY",
    [int]$MaxGameTurns = 10,
    [int]$MaxPrefixDepth = 16,
    [int]$TrainPrefixDepth = 24,
    [int]$MaxSearchNodes = 350,
    [int]$TimeoutSec = 180,
    [int]$ScenarioTimeoutSec = 120,
    [string]$ActionTypes = "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    [double]$MctsKlLossCoef = 3.0,
    [switch]$DirectBcLoss,
    [double]$DirectBcLossCoef = 1.0,
    [switch]$HardenBinaryTargets,
    [switch]$BalanceBinaryTargets,
    [double]$MulliganSampleWeight = 1.0,
    [double]$LondonMulliganSampleWeight = 1.0,
    [switch]$TrainRootMulliganOnNoWin,
    [switch]$TrainRootMulliganOnly,
    [switch]$SkipMulliganTraining,
    [switch]$PrefixSiblingContrast,
    [int]$PrefixSiblingContrastSearchNodes = 0,
    [switch]$FilterPriorityManaActions,
    [switch]$SkipPassTraining,
    [switch]$SkipBlankTraining,
    [switch]$DistillHeadOnly,
    [switch]$DistillPolicyPathOnly,
    [string]$IncludeActionTextRegex = "",
    [string]$OutputBaseDir = "",
    [string]$CollectServiceMode = "none",
    [string]$TrainServiceMode = "local",
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "spy_imitation_offline_bc_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
if ([string]::IsNullOrWhiteSpace($OutputBaseDir)) {
    $OutputBaseDir = Join-Path $repo "local-training\local_pbt\spy_imitation_offline_bc"
}
$root = Join-Path $OutputBaseDir $RunId
$handsDir = Join-Path $root "hand_pools"
$dataDir = Join-Path $root "training_data"
$evalDir = Join-Path $root "evals"
New-Item -ItemType Directory -Force -Path $root, $handsDir, $dataDir, $evalDir | Out-Null

$summaryCsv = Join-Path $root "collect_summary.csv"
$registryPath = Join-Path $root "eval_registry.json"
$activePool = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt"
$spyDeck = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Combo.dek"
$defaultSpyDeckList = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.spy_combo.txt"
$collectAgentDeckList = if ([string]::IsNullOrWhiteSpace($AgentDeckList)) { $defaultSpyDeckList } else { $AgentDeckList }
$collectOppDeckList = if ([string]::IsNullOrWhiteSpace($OppDeckList)) { $activePool } else { $OppDeckList }

if (-not (Test-Path $summaryCsv)) {
    "timestamp,batch,run_id,batch_seconds,trajectories,cumulative_trajectories,candidate_examples,data_file,data_bytes" |
        Set-Content -Path $summaryCsv -Encoding UTF8
}

$registry = @(
    [ordered]@{
        profile = $Profile
        deck_path = $activePool
        active = $true
        train_enabled = $false
        target_winrate = 0.70
        priority = 0
        population_group = "spy-imitation-offline-bc"
        seed = $Seed
        train_env = [ordered]@{
            RL_AGENT_DECK_LIST = $spyDeck
            MODEL_D_MODEL = "128"
            MODEL_NUM_LAYERS = "2"
            AI_MAX_THREADS_FOR_SIMULATIONS = [string]$AiThreads
            EVAL_CP7_SKILL = "1"
            RL_RANDOM_DECISIONS = "0"
            MCTS_TRAINING_ENABLE = "0"
            ISMCTS_ENABLE = "0"
            ISMCTS_ROLLOUTS_PER_TURN = "0"
        }
        notes = "Offline aggregate behavior cloning from serialized Spy trajectories."
    }
)
$registryJson = ConvertTo-Json -InputObject $registry -Depth 8
[System.IO.File]::WriteAllText($registryPath, $registryJson, [System.Text.UTF8Encoding]::new($false))

if (-not $SkipCompile) {
    & mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
    if ($LASTEXITCODE -ne 0) {
        throw "compile failed with exit code $LASTEXITCODE"
    }
}

function Read-KeyValue([string]$Path) {
    $map = @{}
    if (-not (Test-Path $Path)) {
        return $map
    }
    foreach ($line in Get-Content $Path) {
        if ($line -match '^([^:]+):\s*(.*)$') {
            $map[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    return $map
}

$started = Get-Date
$deadline = $started.AddHours($MaxCollectHours)
$cumulative = 0
$batch = 0

while ($cumulative -lt $TargetTrajectories -and (Get-Date) -lt $deadline) {
    $batch++
    $batchStart = Get-Date
    $batchRunId = "$RunId`_collect_batch" + $batch.ToString("0000")
    $handPath = Join-Path $handsDir ("hands_batch" + $batch.ToString("0000") + ".txt")
    $dataFile = Join-Path $dataDir ("training_batch" + $batch.ToString("0000") + ".ser")
    $batchSeed = $Seed + (1009 * $batch)

    & py -3 scripts/generate_spy_reachable_hand_pool.py --out $handPath --count $HandPoolCount --seed $batchSeed
    if ($LASTEXITCODE -ne 0) {
        throw "hand pool generation failed for batch $batch"
    }

    $failed = $false
    try {
        & ./scripts/run_spy_line_search.ps1 `
            -Profile $Profile `
            -RunId $batchRunId `
            -AgentDeckList $collectAgentDeckList `
            -OppDeckList $collectOppDeckList `
            -AgentOpeningHandPoolFile $handPath `
            -Scenarios $BatchScenarios `
            -BatchSize 128 `
            -Opponent cp7 `
            -Cp7Skill 1 `
            -ServiceMode $CollectServiceMode `
            -ModelDModel 128 `
            -ModelNumLayers 2 `
            -ReportEvery 4 `
            -Workers $Workers `
            -AiThreads $AiThreads `
            -MaxPrefixDepth $MaxPrefixDepth `
            -TrainPrefixDepth $TrainPrefixDepth `
            -MaxSearchNodes $MaxSearchNodes `
            -MaxWinningPrefixesPerScenario 1 `
            -TopK 2 `
            -RandomExtra 0 `
            -TrainEpochs 1 `
            -CandidatePermutations 1 `
            -TerminalMode $TerminalMode `
            -ActionTypes $ActionTypes `
            -MaxGameTurns $MaxGameTurns `
            -TimeoutSec $TimeoutSec `
            -ScenarioTimeoutSec $ScenarioTimeoutSec `
            -PrefixSiblingContrast:$PrefixSiblingContrast `
            -PrefixSiblingContrastSearchNodes $PrefixSiblingContrastSearchNodes `
            -TrainRootMulliganOnNoWin:$TrainRootMulliganOnNoWin `
            -TrainRootMulliganOnly:$TrainRootMulliganOnly `
            -TacticAutopilot `
            -NoSearchModelScoring `
            -RandomDecisions `
            -CollectOnly `
            -ExportTrainingDataFile $dataFile `
            -FilterPriorityManaActions:$FilterPriorityManaActions `
            -SkipPassTraining:$SkipPassTraining `
            -SkipBlankTraining:$SkipBlankTraining `
            -IncludeActionTextRegex $IncludeActionTextRegex `
            -SkipMulliganTraining:$SkipMulliganTraining `
            -SkipCompile
        if ($LASTEXITCODE -ne 0) {
            $failed = $true
        }
    } catch {
        $failed = $true
        Write-Host "spy_imitation_offline_collect_failed batch=$batch run=$batchRunId error=$($_.Exception.Message)"
    }

    if ($failed) {
        $seconds = [Math]::Round(((Get-Date) - $batchStart).TotalSeconds, 1)
        $timestamp = (Get-Date).ToString("s")
        "$timestamp,$batch,$batchRunId,$seconds,0,$cumulative,0,$dataFile,0" |
            Add-Content -Path $summaryCsv -Encoding UTF8
        Start-Sleep -Seconds 15
        continue
    }

    $batchOut = Join-Path $repo ("local-training\local_pbt\spy_line_search\" + $batchRunId)
    $readme = Read-KeyValue (Join-Path $batchOut "README.md")
    $traj = if ($readme.ContainsKey("winning_trajectories")) { [int]$readme["winning_trajectories"] } else { 0 }
    $examples = if ($readme.ContainsKey("candidate_examples")) { [int]$readme["candidate_examples"] } else { 0 }
    $bytes = if (Test-Path $dataFile) { (Get-Item $dataFile).Length } else { 0 }
    $cumulative += $traj
    $seconds = [Math]::Round(((Get-Date) - $batchStart).TotalSeconds, 1)
    $timestamp = (Get-Date).ToString("s")
    "$timestamp,$batch,$batchRunId,$seconds,$traj,$cumulative,$examples,$dataFile,$bytes" |
        Add-Content -Path $summaryCsv -Encoding UTF8
    Write-Host "spy_imitation_offline_collect_progress batch=$batch trajectories=$traj cumulative=$cumulative seconds=$seconds examples=$examples bytes=$bytes"
}

if ($cumulative -le 0) {
    throw "no trajectories collected for offline BC"
}

$trainRunId = "$RunId`_offline_train_$cumulative"
$trainArgs = @{
    Profile = $Profile
    RunId = $trainRunId
    ImportTrainingDataPath = $dataDir
    Scenarios = 1
    BatchSize = $TrainBatchSize
    ServiceMode = $TrainServiceMode
    ModelDModel = 128
    ModelNumLayers = 2
    TrainEpochs = $TrainEpochs
    CandidatePermutations = 1
    ActionTypes = $ActionTypes
    MctsKlLossCoef = $MctsKlLossCoef
    DirectBcLossCoef = $DirectBcLossCoef
    HardenBinaryTargets = $HardenBinaryTargets
    BalanceBinaryTargets = $BalanceBinaryTargets
    MulliganSampleWeight = $MulliganSampleWeight
    LondonMulliganSampleWeight = $LondonMulliganSampleWeight
    FilterPriorityManaActions = $FilterPriorityManaActions
    SkipPassTraining = $SkipPassTraining
    SkipBlankTraining = $SkipBlankTraining
    DistillHeadOnly = $DistillHeadOnly
    DistillPolicyPathOnly = $DistillPolicyPathOnly
    IncludeActionTextRegex = $IncludeActionTextRegex
    SkipMulliganTraining = $SkipMulliganTraining
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

$evalRunId = "$RunId`_cp1_$cumulative"
& py -3 scripts/run_cp7_eval_sweep.py `
    --registry $registryPath `
    --profiles $Profile `
    --skill $EvalSkill `
    --games-per-matchup $EvalGamesPerMatchup `
    --games-per-job $EvalGamesPerJob `
    --parallel $EvalParallel `
    --ai-threads $EvalAiThreads `
    --run-id $evalRunId `
    --skip-compile
if ($LASTEXITCODE -ne 0) {
    throw "offline BC eval failed with exit code $LASTEXITCODE"
}

Write-Host "spy_imitation_offline_done run=$RunId profile=$Profile trajectories=$cumulative summary=$summaryCsv data=$dataDir"
