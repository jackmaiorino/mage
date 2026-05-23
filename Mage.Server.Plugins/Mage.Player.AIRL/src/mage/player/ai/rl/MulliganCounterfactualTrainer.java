package mage.player.ai.rl;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameException;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;
import mage.util.ThreadUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Terminal-only mulligan repair pass.
 *
 * For each sampled opening hand, run two full games from the same initial deck
 * order: one with the first keep/mull decision forced to KEEP and one forced to
 * MULLIGAN. The resulting branch outcome becomes a soft policy target for the
 * dedicated mulligan head via TrainingData.mctsVisitTargets.
 */
public final class MulliganCounterfactualTrainer {

    private static final String DEFAULT_DECK_LIST =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt";
    private static final String DEFAULT_OUT_ROOT =
            "local-training/local_pbt/mulligan_counterfactual";

    private MulliganCounterfactualTrainer() {
    }

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);
        Path outDir = parsed.outDir;
        if (outDir == null) {
            String stamp = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").format(LocalDateTime.now());
            outDir = Paths.get(DEFAULT_OUT_ROOT, stamp).toAbsolutePath().normalize();
        }
        Files.createDirectories(outDir);

        List<Path> agentDeckPaths = loadDeckList(parsed.agentDeckList == null ? parsed.deckList : parsed.agentDeckList);
        List<Path> oppDeckPaths = loadDeckList(parsed.oppDeckList == null ? parsed.deckList : parsed.oppDeckList);
        if (agentDeckPaths.isEmpty()) {
            throw new IllegalArgumentException("No agent decks found in "
                    + (parsed.agentDeckList == null ? parsed.deckList : parsed.agentDeckList));
        }
        if (oppDeckPaths.isEmpty()) {
            throw new IllegalArgumentException("No opponent decks found in "
                    + (parsed.oppDeckList == null ? parsed.deckList : parsed.oppDeckList));
        }

        if (parsed.lineMode) {
            runLineMode(parsed, outDir, agentDeckPaths, oppDeckPaths);
            return;
        }

        Random rand = new Random(parsed.seed);
        List<PairJob> jobs = new ArrayList<>();
        for (int pair = 1; pair <= parsed.pairs; pair++) {
            Path agentDeck = agentDeckPaths.get(rand.nextInt(agentDeckPaths.size()));
            Path oppDeck = oppDeckPaths.get(rand.nextInt(oppDeckPaths.size()));
            long pairSeed = parsed.seed + 9973L * pair;
            jobs.add(new PairJob(pair, agentDeck, oppDeck, pairSeed));
        }

        List<SampleRecord> records = new ArrayList<>();
        List<StateSequenceBuilder.TrainingData> batch = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();

        long started = System.currentTimeMillis();
        int trained = 0;
        int skipped = 0;
        int completed = 0;
        ExecutorService executor = Executors.newFixedThreadPool(parsed.workers, new MulliganWorkerThreadFactory());
        CompletionService<PairOutcome> completion = new ExecutorCompletionService<>(executor);
        ThreadLocal<RLTrainer> workerTrainer = ThreadLocal.withInitial(RLTrainer::new);
        for (PairJob job : jobs) {
            completion.submit(() -> processPair(workerTrainer.get(), job, parsed));
        }

        try {
            while (completed < jobs.size()) {
                PairOutcome outcome;
                try {
                    outcome = completion.take().get();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while waiting for mulligan counterfactual workers", ie);
                } catch (ExecutionException ee) {
                    throw new IllegalStateException("Mulligan counterfactual worker failed", ee);
                }
                completed++;
                records.add(outcome.record);
                if (outcome.trainingData != null) {
                    batch.add(outcome.trainingData);
                    rewards.add(0.0);
                    trained++;
                } else {
                    skipped++;
                }

                if (batch.size() >= parsed.batchSize) {
                    RLTrainer.sharedModel.enqueueTraining(batch, rewards);
                    batch = new ArrayList<>();
                    rewards = new ArrayList<>();
                }

                reportProgress(outDir, parsed, combinedDecks(agentDeckPaths, oppDeckPaths),
                        records, completed, trained, skipped, started,
                        outcome.label, outcome.keep, outcome.mull);
            }
        } finally {
            executor.shutdownNow();
        }

        if (!batch.isEmpty()) {
            RLTrainer.sharedModel.enqueueTraining(batch, rewards);
        }

        if (!RLTrainer.sharedModel.awaitTrainingDrained(Math.max(0, parsed.postTrainWaitMs))) {
            throw new IllegalStateException("Timed out waiting for mulligan counterfactual training queue to drain");
        }
        Path modelPath = Paths.get(RLLogPaths.MODEL_FILE_PATH).toAbsolutePath().normalize();
        if (modelPath.getParent() != null) {
            Files.createDirectories(modelPath.getParent());
        }
        RLTrainer.sharedModel.saveModel(modelPath.toString());
        Map<String, Integer> stats = RLTrainer.sharedModel.getMainModelTrainingStats();

        writeSamples(outDir.resolve("counterfactual_samples.csv"), records);
        writeReadme(outDir.resolve("README.md"), parsed, combinedDecks(agentDeckPaths, oppDeckPaths),
                trained, skipped, System.currentTimeMillis() - started);
        RLTrainer.sharedModel.shutdown();

        System.out.println("Mulligan counterfactual output: " + outDir);
        System.out.println("trained=" + trained + " skipped=" + skipped + " stats=" + stats);
    }

    private static void runLineMode(Args parsed, Path outDir, List<Path> agentDeckPaths, List<Path> oppDeckPaths) throws Exception {
        List<LineSpec> lineSpecs = buildLineSpecs(parsed);
        if (lineSpecs.isEmpty()) {
            throw new IllegalArgumentException("No London line specs generated");
        }

        Random rand = new Random(parsed.seed);
        List<ScenarioJob> jobs = new ArrayList<>();
        for (int scenario = 1; scenario <= parsed.pairs; scenario++) {
            Path agentDeck = agentDeckPaths.get(rand.nextInt(agentDeckPaths.size()));
            Path oppDeck = oppDeckPaths.get(rand.nextInt(oppDeckPaths.size()));
            long scenarioSeed = parsed.seed + 9973L * scenario;
            jobs.add(new ScenarioJob(scenario, agentDeck, oppDeck, scenarioSeed));
        }

        List<LineRecord> records = new ArrayList<>();
        List<LineTrainingExample> trainingExamples = new ArrayList<>();
        List<StateSequenceBuilder.TrainingData> winningLineTrainingData = new ArrayList<>();
        long started = System.currentTimeMillis();
        int completed = 0;
        int trainedScenarios = 0;
        int skippedScenarios = 0;

        ExecutorService executor = Executors.newFixedThreadPool(parsed.workers, new MulliganWorkerThreadFactory());
        CompletionService<LineOutcome> completion = new ExecutorCompletionService<>(executor);
        ThreadLocal<RLTrainer> workerTrainer = ThreadLocal.withInitial(RLTrainer::new);
        for (ScenarioJob job : jobs) {
            completion.submit(() -> processLineScenario(workerTrainer.get(), job, parsed, lineSpecs));
        }

        try {
            while (completed < jobs.size()) {
                LineOutcome outcome;
                try {
                    outcome = completion.take().get();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while waiting for London line workers", ie);
                } catch (ExecutionException ee) {
                    throw new IllegalStateException("London line worker failed", ee);
                }
                completed++;
                records.addAll(outcome.records);
                if (outcome.trainingExamples.isEmpty()) {
                    skippedScenarios++;
                } else {
                    trainedScenarios++;
                    trainingExamples.addAll(outcome.trainingExamples);
                }
                winningLineTrainingData.addAll(outcome.winningTrainingData);
                if (completed % Math.max(1, parsed.reportEvery) == 0) {
                    writeLineSamples(outDir.resolve("london_line_samples.csv"), records);
                    writeLineReadme(outDir.resolve("README.md"), parsed, combinedDecks(agentDeckPaths, oppDeckPaths), lineSpecs,
                            trainedScenarios, skippedScenarios, trainingExamples.size(), System.currentTimeMillis() - started);
                    System.out.println(String.format(Locale.US,
                            "scenarios=%d/%d trainedScenarios=%d skippedScenarios=%d candidateSamples=%d last=%s",
                            completed, parsed.pairs, trainedScenarios, skippedScenarios, trainingExamples.size(), outcome.label));
                }
            }
        } finally {
            executor.shutdownNow();
        }

        List<LineTrainingExample> selectedExamples = selectLineTrainingExamples(trainingExamples, parsed);
        if (parsed.lineBucketSoftTargets) {
            selectedExamples = applyBucketSoftTargets(selectedExamples);
        }
        if (parsed.exportTrainingDataFile != null) {
            writeSerializedLineTrainingData(parsed.exportTrainingDataFile, selectedExamples);
        }
        if (parsed.exportWinningLineDataFile != null) {
            writeSerializedTrainingData(parsed.exportWinningLineDataFile, winningLineTrainingData);
        }
        if (parsed.collectOnly) {
            writeLineSamples(outDir.resolve("london_line_samples.csv"), records);
            writeLineTrainingSamples(outDir.resolve("london_training_samples.csv"), selectedExamples);
            writeLineReadme(outDir.resolve("README.md"), parsed, combinedDecks(agentDeckPaths, oppDeckPaths), lineSpecs,
                    trainedScenarios, skippedScenarios, 0, System.currentTimeMillis() - started);
            RLTrainer.sharedModel.shutdown();

            System.out.println("London line counterfactual collect-only output: " + outDir);
            System.out.println("trainedScenarios=" + trainedScenarios
                    + " skippedScenarios=" + skippedScenarios
                    + " candidateSamples=" + trainingExamples.size()
                    + " selectedSamples=" + selectedExamples.size()
                    + " winningLineSamples=" + winningLineTrainingData.size()
                    + " trainPassSamples=0");
            return;
        }
        List<StateSequenceBuilder.TrainingData> batch = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();
        Random trainRand = new Random(parsed.seed ^ 0x51A7E5EED5EEDL);
        int trainPassSamples = 0;
        for (int epoch = 0; epoch < parsed.lineTrainEpochs; epoch++) {
            List<LineTrainingExample> epochExamples = new ArrayList<>(selectedExamples);
            Collections.shuffle(epochExamples, trainRand);
            for (LineTrainingExample example : epochExamples) {
                batch.add(example.trainingData);
                rewards.add(0.0);
                trainPassSamples++;
                if (batch.size() >= parsed.batchSize) {
                    RLTrainer.sharedModel.enqueueTraining(batch, rewards);
                    batch = new ArrayList<>();
                    rewards = new ArrayList<>();
                }
            }
        }
        if (!batch.isEmpty()) {
            RLTrainer.sharedModel.enqueueTraining(batch, rewards);
        }

        if (!RLTrainer.sharedModel.awaitTrainingDrained(Math.max(0, parsed.postTrainWaitMs))) {
            throw new IllegalStateException("Timed out waiting for London line training queue to drain");
        }
        Path modelPath = Paths.get(RLLogPaths.MODEL_FILE_PATH).toAbsolutePath().normalize();
        if (modelPath.getParent() != null) {
            Files.createDirectories(modelPath.getParent());
        }
        RLTrainer.sharedModel.saveModel(modelPath.toString());
        Map<String, Integer> stats = RLTrainer.sharedModel.getMainModelTrainingStats();

        writeLineSamples(outDir.resolve("london_line_samples.csv"), records);
        writeLineTrainingSamples(outDir.resolve("london_training_samples.csv"), selectedExamples);
        writeLineReadme(outDir.resolve("README.md"), parsed, combinedDecks(agentDeckPaths, oppDeckPaths), lineSpecs,
                trainedScenarios, skippedScenarios, trainPassSamples, System.currentTimeMillis() - started);
        RLTrainer.sharedModel.shutdown();

        System.out.println("London line counterfactual output: " + outDir);
        System.out.println("trainedScenarios=" + trainedScenarios
                + " skippedScenarios=" + skippedScenarios
                + " candidateSamples=" + trainingExamples.size()
                + " selectedSamples=" + selectedExamples.size()
                + " winningLineSamples=" + winningLineTrainingData.size()
                + " trainPassSamples=" + trainPassSamples
                + " stats=" + stats);
    }

    private static LineOutcome processLineScenario(
            RLTrainer trainer,
            ScenarioJob job,
            Args parsed,
            List<LineSpec> lineSpecs
    ) {
        List<LineResult> results = new ArrayList<>();
        List<LineRecord> records = new ArrayList<>();
        for (LineSpec spec : lineSpecs) {
            LineResult result = runLineBranch(trainer, job, spec, parsed);
            results.add(result);
            records.add(LineRecord.from(job, result));
        }

        boolean anyWin = results.stream().anyMatch(r -> r.won && !r.timedOut);
        boolean anyLoss = results.stream().anyMatch(r -> !r.won && !r.timedOut);
        if (!anyWin || !anyLoss) {
            return new LineOutcome(records, Collections.emptyList(), winningLineTrainingData(results),
                    anyWin ? "SKIP_ALL_WIN" : "SKIP_ALL_LOSS");
        }

        List<LineTrainingExample> train = new ArrayList<>();

        for (int promptIndex = 0; promptIndex < parsed.lineMaxMulls; promptIndex++) {
            final int prompt = promptIndex;
            LineStats keepStats = lineStats(results, r -> r.spec.keepAfterMulls == prompt);
            LineStats deeperMullStats = lineStats(results, r -> r.spec.keepAfterMulls > prompt);
            LineResult promptResult = firstResultWithMulliganPromptData(results, prompt);
            StateSequenceBuilder.TrainingData promptData = promptResult == null
                    ? null
                    : promptResult.nthMulliganData(prompt);
            if (promptData != null && keepStats.hasSamples() && deeperMullStats.hasSamples()) {
                addBinaryMarginTarget(
                        train,
                        promptData,
                        job,
                        promptResult,
                        mulliganPromptKind(prompt),
                        keepStats.winRate(),
                        deeperMullStats.winRate(),
                        parsed.lineMarginMin,
                        parsed.lineTargetTemperature,
                        parsed.lineBalanceKey,
                        StateSequenceBuilder.ActionType.MULLIGAN
                );
            }
        }

        for (int mulls = 1; mulls <= parsed.lineMaxMulls; mulls++) {
            final int mullsForBucket = mulls;
            List<LineResult> bucket = results.stream()
                    .filter(r -> r.spec.keepAfterMulls == mullsForBucket && !r.timedOut)
                    .collect(Collectors.toList());
            boolean bucketWin = bucket.stream().anyMatch(r -> r.won);
            boolean bucketLoss = bucket.stream().anyMatch(r -> !r.won);
            if (!bucketWin || !bucketLoss) {
                continue;
            }
            LineResult winner = bucket.stream().filter(r -> r.won && r.bottomData() != null).findFirst().orElse(null);
            if (winner != null) {
                train.add(LineTrainingExample.bottom(
                        job,
                        winner,
                        copyWithBottomKeepTarget(winner.bottomData(), mullsForBucket),
                        parsed.lineBalanceKey
                ));
            }
        }

        List<StateSequenceBuilder.TrainingData> winningData = winningLineTrainingData(results);
        return new LineOutcome(records, train, winningData, train.isEmpty() ? "NO_ACTIONABLE_LABEL" : "TRAIN");
    }

    private static List<StateSequenceBuilder.TrainingData> winningLineTrainingData(List<LineResult> results) {
        LineResult winner = results.stream()
                .filter(r -> r != null && r.won && !r.timedOut && r.error.isEmpty() && !r.data.isEmpty())
                .min(Comparator.comparingInt(r -> r.turns < 0 ? Integer.MAX_VALUE : r.turns))
                .orElse(null);
        if (winner == null) {
            return Collections.emptyList();
        }
        List<StateSequenceBuilder.TrainingData> out = new ArrayList<>();
        for (StateSequenceBuilder.TrainingData td : winner.data) {
            StateSequenceBuilder.TrainingData cloned;
            if (td.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN && winner.spec.keepAfterMulls > 0) {
                cloned = copyWithBottomKeepTarget(td, winner.spec.keepAfterMulls);
            } else {
                cloned = copyWithChosenTarget(td);
            }
            if (cloned != null) {
                out.add(cloned);
            }
        }
        return out;
    }

    private static PairOutcome processPair(RLTrainer trainer, PairJob job, Args parsed) {
        try {
            BranchResult keep = runBranch(trainer, job.agentDeck, job.oppDeck, job.seed, 0, parsed);
            BranchResult mull = runBranch(trainer, job.agentDeck, job.oppDeck, job.seed, 1, parsed);

            if (keep.trainingData == null || mull.trainingData == null || keep.timedOut || mull.timedOut) {
                return PairOutcome.skipped(job, "SKIP_BRANCH", keep, mull);
            }

            float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
            int labelIdx;
            String label;
            if (keep.won && !mull.won) {
                target[0] = 1.0f;
                labelIdx = 0;
                label = "KEEP";
            } else if (!keep.won && mull.won) {
                target[1] = 1.0f;
                labelIdx = 1;
                label = "MULLIGAN";
            } else if (parsed.trainTies) {
                target[0] = 0.5f;
                target[1] = 0.5f;
                labelIdx = keep.pKeep >= keep.pMull ? 0 : 1;
                label = "TIE_SOFT";
            } else {
                return PairOutcome.skipped(job, "SKIP_TIE", keep, mull);
            }

            StateSequenceBuilder.TrainingData td = copyWithLabel(keep.trainingData, labelIdx, keep, target);
            SampleRecord record = SampleRecord.trained(job.pair, job.agentDeck, job.oppDeck, job.seed,
                    keep, mull, label, target[0], target[1]);
            return new PairOutcome(record, td, label, keep, mull);
        } catch (Throwable t) {
            BranchResult failedKeep = BranchResult.failed("KEEP", t.getClass().getSimpleName() + ": " + t.getMessage());
            BranchResult failedMull = BranchResult.failed("MULL", t.getClass().getSimpleName() + ": " + t.getMessage());
            return PairOutcome.skipped(job, "SKIP_EXCEPTION", failedKeep, failedMull);
        }
    }

    private static void reportProgress(
            Path outDir,
            Args parsed,
            List<Path> deckPaths,
            List<SampleRecord> records,
            int pair,
            int trained,
            int skipped,
            long started,
            String label,
            BranchResult keep,
            BranchResult mull
    ) throws IOException {
        if (pair % Math.max(1, parsed.reportEvery) != 0) {
            return;
        }
        writeSamples(outDir.resolve("counterfactual_samples.csv"), records);
        writeReadme(outDir.resolve("README.md"), parsed, deckPaths, trained, skipped, System.currentTimeMillis() - started);
        System.out.println(String.format(Locale.US,
                "pairs=%d/%d trained=%d skipped=%d last=%s keepWin=%s mullWin=%s",
                pair, parsed.pairs, trained, skipped, label,
                keep != null && keep.won,
                mull != null && mull.won));
    }

    private static BranchResult runBranch(
            RLTrainer trainer,
            Path agentDeckPath,
            Path oppDeckPath,
            long seed,
            int forcedChoice,
            Args args
    ) {
        String branch = forcedChoice == 0 ? "KEEP" : "MULL";
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("MulliganCF", "MulliganCF", false));
        Game game = null;
        BranchPlayer rlPlayer = null;
        Player opponent = null;
        try {
            Deck agentBase = trainer.loadDeckFresh(agentDeckPath.toString());
            Deck oppBase = trainer.loadDeckFresh(oppDeckPath.toString());
            if (agentBase == null || oppBase == null) {
                return BranchResult.failed(branch, "deck_load_failed");
            }
            Deck agentDeck = shuffledCopy(agentBase, seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, seed ^ 0xC0FFEE1234L);

            match.startGame();
            game = match.getGames().get(0);

            rlPlayer = new BranchPlayer("CF-" + branch, forcedChoice);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = new ComputerPlayer7("CF-CP7", RangeOfInfluence.ALL, args.cp7Skill);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("CF-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, false, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);

            boolean won = false;
            try {
                won = game.getWinner().contains(rlPlayer.getName());
            } catch (Exception ignored) {
                won = false;
            }
            StateSequenceBuilder.TrainingData firstMulligan = null;
            for (StateSequenceBuilder.TrainingData td : rlPlayer.getTrainingBuffer()) {
                if (td.actionType == StateSequenceBuilder.ActionType.MULLIGAN) {
                    firstMulligan = td;
                    break;
                }
            }
            return new BranchResult(branch, won, false, null, firstMulligan,
                    rlPlayer.hand, rlPlayer.lands, rlPlayer.pKeep, rlPlayer.pMull, safeTurn(game));
        } catch (Exception e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            return new BranchResult(branch, false, timedOut, e.getClass().getSimpleName() + ": " + e.getMessage(),
                    null, rlPlayer == null ? "" : rlPlayer.hand, rlPlayer == null ? -1 : rlPlayer.lands,
                    rlPlayer == null ? Float.NaN : rlPlayer.pKeep,
                    rlPlayer == null ? Float.NaN : rlPlayer.pMull,
                    game == null ? -1 : safeTurn(game));
        } finally {
            try {
                if (game != null) {
                    game.end();
                    game.cleanUp();
                }
            } catch (Exception ignored) {
            }
            try {
                match.getGames().clear();
            } catch (Exception ignored) {
            }
        }
    }

    private static LineResult runLineBranch(RLTrainer trainer, ScenarioJob job, LineSpec spec, Args args) {
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("LondonLineCF", "LondonLineCF", false));
        Game game = null;
        LinePlayer rlPlayer = null;
        Player opponent = null;
        try {
            Deck agentBase = trainer.loadDeckFresh(job.agentDeck.toString());
            Deck oppBase = trainer.loadDeckFresh(job.oppDeck.toString());
            if (agentBase == null || oppBase == null) {
                return LineResult.failed(spec, "deck_load_failed");
            }
            Deck agentDeck = shuffledCopy(agentBase, job.seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, job.seed ^ 0xC0FFEE1234L);

            match.startGame();
            game = match.getGames().get(0);

            rlPlayer = new LinePlayer("LINE-" + spec.name, spec);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = new ComputerPlayer7("LINE-CP7", RangeOfInfluence.ALL, args.cp7Skill);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("LINE-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, false, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);

            boolean won;
            try {
                won = game.getWinner().contains(rlPlayer.getName());
            } catch (Exception ignored) {
                won = false;
            }
            List<StateSequenceBuilder.TrainingData> data = new ArrayList<>(rlPlayer.getTrainingBuffer());
            return new LineResult(spec, won, false, "", data,
                    rlPlayer.hand, rlPlayer.lands, safeTurn(game));
        } catch (Exception e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            return new LineResult(spec, false, timedOut,
                    e.getClass().getSimpleName() + ": " + e.getMessage(),
                    Collections.emptyList(),
                    rlPlayer == null ? "" : rlPlayer.hand,
                    rlPlayer == null ? -1 : rlPlayer.lands,
                    game == null ? -1 : safeTurn(game));
        } finally {
            try {
                if (game != null) {
                    game.end();
                    game.cleanUp();
                }
            } catch (Exception ignored) {
            }
            try {
                match.getGames().clear();
            } catch (Exception ignored) {
            }
        }
    }

    private static List<LineSpec> buildLineSpecs(Args args) {
        List<LineSpec> specs = new ArrayList<>();
        specs.add(new LineSpec(0, -1));
        for (int mulls = 1; mulls <= args.lineMaxMulls; mulls++) {
            if (args.lineBottomCombos < 0) {
                specs.add(new LineSpec(mulls, -1));
                continue;
            }
            int combos = combinationCount(7, mulls);
            int limit = args.lineBottomCombos <= 0 ? combos : Math.min(combos, args.lineBottomCombos);
            for (int combo = 0; combo < limit; combo++) {
                specs.add(new LineSpec(mulls, combo));
            }
        }
        return specs;
    }

    private static int combinationCount(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }
        if (k == 0 || k == n) {
            return 1;
        }
        k = Math.min(k, n - k);
        int result = 1;
        for (int i = 1; i <= k; i++) {
            result = (result * (n - k + i)) / i;
        }
        return result;
    }

    private static List<Integer> nthCombination(int n, int k, int ordinal) {
        List<Integer> current = new ArrayList<>();
        if (k <= 0) {
            return current;
        }
        int total = Math.max(1, combinationCount(n, k));
        int target = Math.floorMod(ordinal, total);
        int[] found = new int[]{-1};
        nthCombinationDfs(n, k, 0, current, target, found);
        return new ArrayList<>(current);
    }

    private static boolean nthCombinationDfs(int n, int k, int start, List<Integer> current, int target, int[] seen) {
        if (current.size() == k) {
            seen[0]++;
            return seen[0] == target;
        }
        for (int i = start; i <= n - (k - current.size()); i++) {
            current.add(i);
            if (nthCombinationDfs(n, k, i + 1, current, target, seen)) {
                return true;
            }
            current.remove(current.size() - 1);
        }
        return false;
    }

    private static LineStats lineStats(List<LineResult> results, Predicate<LineResult> filter) {
        int samples = 0;
        int wins = 0;
        for (LineResult result : results) {
            if (result == null || result.timedOut || !result.error.isEmpty() || !filter.test(result)) {
                continue;
            }
            samples++;
            if (result.won) {
                wins++;
            }
        }
        return new LineStats(samples, wins);
    }

    private static LineResult firstResultWithMulliganPromptData(List<LineResult> results, int promptIndex) {
        for (LineResult result : results) {
            if (result == null || result.timedOut || !result.error.isEmpty()) {
                continue;
            }
            StateSequenceBuilder.TrainingData td = result.nthMulliganData(promptIndex);
            if (td != null) {
                return result;
            }
        }
        return null;
    }

    private static String mulliganPromptKind(int promptIndex) {
        return promptIndex == 0 ? "FIRST" : "AFTER_" + promptIndex + "_MULLS";
    }

    private static boolean addBinaryMarginTarget(
            List<LineTrainingExample> train,
            StateSequenceBuilder.TrainingData source,
            ScenarioJob job,
            LineResult result,
            String kind,
            double keepRate,
            double mullRate,
            double minMargin,
            double targetTemperature,
            String balanceKeyMode,
            StateSequenceBuilder.ActionType actionType
    ) {
        if (source == null || Double.isNaN(keepRate) || Double.isNaN(mullRate)) {
            return false;
        }
        double margin = keepRate - mullRate;
        if (Math.abs(margin) < Math.max(0.0, minMargin)) {
            return false;
        }
        StateSequenceBuilder.TrainingData td = copyWithSoftBinaryTarget(source, keepRate, mullRate, targetTemperature, actionType);
        train.add(LineTrainingExample.binary(job, result, kind, td, keepRate, mullRate, balanceKeyMode));
        return true;
    }

    private static List<LineTrainingExample> selectLineTrainingExamples(List<LineTrainingExample> examples, Args args) {
        List<LineTrainingExample> filtered = examples.stream()
                .filter(e -> args.lineTrainBottoms || e.isBinaryMulligan())
                .collect(Collectors.toCollection(ArrayList::new));
        if (!args.lineBalanceMulliganPrompts) {
            return filtered;
        }
        List<LineTrainingExample> selected = new ArrayList<>();
        Map<String, List<LineTrainingExample>> binaryBuckets = new LinkedHashMap<>();
        for (LineTrainingExample example : filtered) {
            if (!example.isBinaryMulligan()) {
                if (args.lineTrainBottoms) {
                    selected.add(example);
                }
                continue;
            }
            String key = example.kind + "|" + example.balanceKey;
            binaryBuckets.computeIfAbsent(key, k -> new ArrayList<>()).add(example);
        }
        Random rand = new Random(args.seed ^ 0xBADC0FFEE0DDF00DL);
        for (List<LineTrainingExample> bucket : binaryBuckets.values()) {
            List<LineTrainingExample> keep = bucket.stream()
                    .filter(e -> e.label == 0)
                    .collect(Collectors.toCollection(ArrayList::new));
            List<LineTrainingExample> mull = bucket.stream()
                    .filter(e -> e.label == 1)
                    .collect(Collectors.toCollection(ArrayList::new));
            Collections.shuffle(keep, rand);
            Collections.shuffle(mull, rand);
            if (!keep.isEmpty() && !mull.isEmpty()) {
                int n = Math.min(keep.size(), mull.size());
                selected.addAll(keep.subList(0, n));
                selected.addAll(mull.subList(0, n));
            } else {
                List<LineTrainingExample> singleton = keep.isEmpty() ? mull : keep;
                int n = Math.min(Math.max(0, args.lineBalanceSingletonLimit), singleton.size());
                selected.addAll(singleton.subList(0, n));
            }
        }
        selected.sort(Comparator.comparingInt((LineTrainingExample e) -> e.scenario)
                .thenComparing(e -> e.kind)
                .thenComparing(e -> e.balanceKey));
        return selected;
    }

    private static List<LineTrainingExample> applyBucketSoftTargets(List<LineTrainingExample> examples) {
        Map<String, List<LineTrainingExample>> buckets = new LinkedHashMap<>();
        for (LineTrainingExample example : examples) {
            if (!example.isBinaryMulligan()) {
                continue;
            }
            String key = example.kind + "|" + example.balanceKey;
            buckets.computeIfAbsent(key, k -> new ArrayList<>()).add(example);
        }
        List<LineTrainingExample> out = new ArrayList<>(examples.size());
        for (LineTrainingExample example : examples) {
            if (!example.isBinaryMulligan()) {
                out.add(example);
                continue;
            }
            String key = example.kind + "|" + example.balanceKey;
            List<LineTrainingExample> bucket = buckets.get(key);
            if (bucket == null || bucket.isEmpty()) {
                out.add(example);
                continue;
            }
            double keep = 0.0;
            double mull = 0.0;
            for (LineTrainingExample b : bucket) {
                keep += b.targetKeep;
                mull += b.targetMull;
            }
            double sum = Math.max(1e-9, keep + mull);
            out.add(example.withBinaryTarget((float) (keep / sum), (float) (mull / sum)));
        }
        return out;
    }

    private static String balanceKey(Path agentDeck, LineResult result, String mode) {
        String deck = agentDeck == null ? "unknown" : agentDeck.getFileName().toString();
        int lands = result == null ? -1 : result.lands;
        String hand = result == null ? "" : result.hand;
        int effective = effectiveLandCount(agentDeck, lands, hand);
        String normalized = mode == null ? "deck-resource" : mode.trim().toLowerCase(Locale.ROOT);
        if ("deck".equals(normalized)) {
            return deck;
        }
        if ("resource".equals(normalized)) {
            return "effective=" + effective;
        }
        return deck + "|effective=" + effective;
    }

    private static int effectiveLandCount(Path agentDeck, int lands, String hand) {
        if (lands < 0) {
            return lands;
        }
        return lands + pseudoLandCount(agentDeck, hand);
    }

    private static int pseudoLandCount(Path agentDeck, String hand) {
        String deckName = agentDeck == null || agentDeck.getFileName() == null
                ? ""
                : agentDeck.getFileName().toString().toLowerCase(Locale.ROOT);
        if (!deckName.contains("spy") || hand == null || hand.isEmpty()) {
            return 0;
        }
        int count = 0;
        for (String raw : hand.split(";")) {
            String cardName = raw.trim();
            if ("Land Grant".equals(cardName) || "Lotus Petal".equals(cardName)) {
                count++;
            }
        }
        return count;
    }

    private static StateSequenceBuilder.TrainingData copyWithOneHotTarget(
            StateSequenceBuilder.TrainingData source,
            int labelIdx,
            int candidateCount,
            StateSequenceBuilder.ActionType actionType
    ) {
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        int idx = Math.max(0, Math.min(labelIdx, candidateCount - 1));
        target[idx] = 1.0f;
        return copyWithTargets(source, idx, 1, target, actionType);
    }

    private static StateSequenceBuilder.TrainingData copyWithSoftBinaryTarget(
            StateSequenceBuilder.TrainingData source,
            double keepRate,
            double mullRate,
            double targetTemperature,
            StateSequenceBuilder.ActionType actionType
    ) {
        double temperature = Math.max(1e-6, targetTemperature);
        double max = Math.max(keepRate, mullRate);
        double keepScore = Math.exp((keepRate - max) / temperature);
        double mullScore = Math.exp((mullRate - max) / temperature);
        double sum = Math.max(1e-12, keepScore + mullScore);
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        target[0] = (float) (keepScore / sum);
        target[1] = (float) (mullScore / sum);
        return copyWithTargets(source, target[0] >= target[1] ? 0 : 1, 1, target, actionType);
    }

    private static StateSequenceBuilder.TrainingData copyWithBottomKeepTarget(
            StateSequenceBuilder.TrainingData source,
            int bottomCount
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        float[] target = new float[max];
        int keepCount = Math.max(0, source.candidateCount - Math.max(0, bottomCount));
        for (int i = 0; i < Math.min(keepCount, source.chosenCount); i++) {
            int idx = source.chosenIndices[i];
            if (idx >= 0 && idx < max) {
                target[idx] = 1.0f / Math.max(1, keepCount);
            }
        }
        int first = source.chosenCount > 0 ? source.chosenIndices[0] : 0;
        return copyWithTargets(source, first, source.chosenCount, target, StateSequenceBuilder.ActionType.LONDON_MULLIGAN);
    }

    private static StateSequenceBuilder.TrainingData copyWithChosenTarget(StateSequenceBuilder.TrainingData source) {
        if (source == null || source.chosenCount <= 0 || source.chosenIndices == null) {
            return null;
        }
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        float[] target = new float[max];
        int validChosen = 0;
        for (int i = 0; i < Math.min(source.chosenCount, source.chosenIndices.length); i++) {
            int idx = source.chosenIndices[i];
            if (idx >= 0 && idx < max && source.candidateMask != null && source.candidateMask[idx] != 0) {
                validChosen++;
            }
        }
        if (validChosen <= 0) {
            return null;
        }
        float mass = 1.0f / validChosen;
        int first = -1;
        for (int i = 0; i < Math.min(source.chosenCount, source.chosenIndices.length); i++) {
            int idx = source.chosenIndices[i];
            if (idx >= 0 && idx < max && source.candidateMask != null && source.candidateMask[idx] != 0) {
                if (first < 0) {
                    first = idx;
                }
                target[idx] = mass;
            }
        }
        return copyWithTargets(source, first, Math.max(1, validChosen), target, source.actionType);
    }

    private static StateSequenceBuilder.TrainingData copyWithTargets(
            StateSequenceBuilder.TrainingData source,
            int firstChosen,
            int chosenCount,
            float[] target,
            StateSequenceBuilder.ActionType actionType
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int[] chosen = java.util.Arrays.copyOf(source.chosenIndices, source.chosenIndices.length);
        if (chosen.length > 0 && firstChosen >= 0) {
            chosen[0] = firstChosen;
        }
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                java.util.Arrays.copyOf(source.candidateActionIds, source.candidateActionIds.length),
                copy2d(source.candidateFeatures),
                java.util.Arrays.copyOf(source.candidateMask, source.candidateMask.length),
                Math.max(1, Math.min(chosenCount, max)),
                chosen,
                0.0f,
                0.0f,
                actionType,
                0.0
        );
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        td.setMctsVisitTargets(target);
        return td;
    }

    private static StateSequenceBuilder.TrainingData copyWithLabel(
            StateSequenceBuilder.TrainingData source,
            int labelIdx,
            BranchResult keep,
            float[] target
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int[] chosen = new int[max];
        java.util.Arrays.fill(chosen, -1);
        chosen[0] = Math.max(0, Math.min(labelIdx, 1));
        float prob = labelIdx == 1 ? keep.pMull : keep.pKeep;
        if (!(prob > 0.0f) || Float.isNaN(prob) || Float.isInfinite(prob)) {
            prob = 0.5f;
        }
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                java.util.Arrays.copyOf(source.candidateActionIds, source.candidateActionIds.length),
                copy2d(source.candidateFeatures),
                java.util.Arrays.copyOf(source.candidateMask, source.candidateMask.length),
                1,
                chosen,
                (float) Math.log(Math.max(1e-8f, prob)),
                0.0f,
                StateSequenceBuilder.ActionType.MULLIGAN,
                0.0
        );
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        td.setMctsVisitTargets(target);
        return td;
    }

    private static float[][] copy2d(float[][] src) {
        float[][] out = new float[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = java.util.Arrays.copyOf(src[i], src[i].length);
        }
        return out;
    }

    private static Deck shuffledCopy(Deck source, long seed) {
        Deck deck = source.copy();
        List<Card> cards = new ArrayList<>(deck.getCards());
        Collections.shuffle(cards, new Random(seed));
        deck.getCards().clear();
        for (Card card : cards) {
            deck.getCards().add(card);
        }
        return deck;
    }

    private static void forceLibraryOrder(Player player, Deck deck, Game game) {
        if (player == null || deck == null || game == null) {
            return;
        }
        LinkedHashSet<Card> ordered = new LinkedHashSet<>();
        for (Card card : deck.getCards()) {
            if (card != null && !card.isExtraDeckCard()) {
                ordered.add(card);
            }
        }
        player.getLibrary().clear();
        player.getLibrary().addAll(ordered, game);
    }

    private static void startGameInGameThread(Game game, UUID startingPlayerId, int joinTimeoutSec) {
        if (ThreadUtils.isRunGameThread()) {
            game.start(startingPlayerId);
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                game.start(startingPlayerId);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-MULLIGAN-CF");
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            try {
                game.end();
            } catch (Exception ignored) {
            }
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ignored) {
                // Preserve the interrupted status below.
            }
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for game thread", ie);
        }
        if (gameThread.isAlive()) {
            try {
                game.end();
            } catch (Exception ignored) {
            }
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            if (gameThread.isAlive()) {
                throw new IllegalStateException("Game thread timeout; worker did not stop after cleanup");
            }
            throw new IllegalStateException("Game thread timeout");
        }

        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        }
        throw new IllegalStateException("Error while running game", t);
    }

    private static int safeTurn(Game game) {
        try {
            return game.getTurnNum();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static List<Path> loadDeckList(String deckListPath) throws IOException {
        Path list = Paths.get(deckListPath).toAbsolutePath().normalize();
        if (Files.isRegularFile(list) && list.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".dek")) {
            return Collections.singletonList(list);
        }
        Path base = list.getParent();
        List<Path> decks = new ArrayList<>();
        for (String raw : Files.readAllLines(list, StandardCharsets.UTF_8)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Path p = Paths.get(line);
            if (!p.isAbsolute()) {
                p = base.resolve(p);
            }
            p = p.toAbsolutePath().normalize();
            if (Files.isRegularFile(p)) {
                decks.add(p);
            } else {
                Path cwdRelative = Paths.get(line).toAbsolutePath().normalize();
                if (Files.isRegularFile(cwdRelative)) {
                    decks.add(cwdRelative);
                } else {
                    throw new IllegalArgumentException("Deck path not found: " + line + " from " + deckListPath);
                }
            }
        }
        return decks;
    }

    private static List<Path> combinedDecks(List<Path> agentDecks, List<Path> oppDecks) {
        List<Path> combined = new ArrayList<>();
        if (agentDecks != null) {
            combined.addAll(agentDecks);
        }
        if (oppDecks != null) {
            combined.addAll(oppDecks);
        }
        return combined;
    }

    private static void writeSamples(Path path, List<SampleRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<SampleRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt(r -> r.pair));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("pair,agent_deck,opp_deck,seed,status,label,keep_win,mull_win,target_keep,target_mull,keep_lands,mull_lands,keep_p_keep,keep_p_mull,mull_p_keep,mull_p_mull,keep_turns,mull_turns,keep_error,mull_error,hand\n");
            for (SampleRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeReadme(Path path, Args args, List<Path> decks, int trained, int skipped, long elapsedMs) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# Mulligan Counterfactual Trainer\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("pairs: ").append(args.pairs).append('\n');
        sb.append("trained_samples: ").append(trained).append('\n');
        sb.append("skipped_samples: ").append(skipped).append('\n');
        sb.append("opponent_mode: ").append(args.opponentMode).append('\n');
        sb.append("batch_size: ").append(args.batchSize).append('\n');
        sb.append("workers: ").append(args.workers).append('\n');
        sb.append("timeout_sec: ").append(args.timeoutSec).append('\n');
        sb.append("seed: ").append(args.seed).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        sb.append("deck_list: ").append(args.deckList).append('\n');
        sb.append("decks:\n");
        for (Path deck : decks) {
            sb.append("- ").append(deck).append('\n');
        }
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeLineSamples(Path path, List<LineRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<LineRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((LineRecord r) -> r.scenario).thenComparing(r -> r.line));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,line,keep_after_mulls,bottom_combo,won,timed_out,turns,lands,hand,error\n");
            for (LineRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeLineTrainingSamples(Path path, List<LineTrainingExample> examples) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<LineTrainingExample> ordered = new ArrayList<>(examples);
        ordered.sort(Comparator.comparingInt((LineTrainingExample e) -> e.scenario)
                .thenComparing(e -> e.kind)
                .thenComparing(e -> e.balanceKey));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,kind,balance_key,agent_deck,opp_deck,label,target_keep,target_mull,lands,effective_lands,hand\n");
            for (LineTrainingExample e : ordered) {
                out.write(e.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeSerializedLineTrainingData(Path path, List<LineTrainingExample> examples) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<StateSequenceBuilder.TrainingData> data = new ArrayList<>(examples.size());
        for (LineTrainingExample example : examples) {
            if (example.trainingData != null) {
                data.add(example.trainingData);
            }
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(data);
        }
    }

    private static void writeSerializedTrainingData(Path path, List<StateSequenceBuilder.TrainingData> data) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<StateSequenceBuilder.TrainingData> filtered = data == null
                ? Collections.emptyList()
                : data.stream().filter(td -> td != null).collect(Collectors.toCollection(ArrayList::new));
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(filtered);
        }
    }

    private static void writeLineReadme(Path path, Args args, List<Path> decks, List<LineSpec> specs,
                                        int trainedScenarios, int skippedScenarios, int trainedSamples,
                                        long elapsedMs) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# London Line Counterfactual Trainer\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("scenarios: ").append(args.pairs).append('\n');
        sb.append("line_specs_per_scenario: ").append(specs.size()).append('\n');
        sb.append("line_max_mulls: ").append(args.lineMaxMulls).append('\n');
        sb.append("line_bottom_combos: ").append(args.lineBottomCombos).append('\n');
        sb.append("line_margin_min: ").append(String.format(Locale.US, "%.4f", args.lineMarginMin)).append('\n');
        sb.append("line_target_temperature: ").append(String.format(Locale.US, "%.4f", args.lineTargetTemperature)).append('\n');
        sb.append("line_balance_mulligan_prompts: ").append(args.lineBalanceMulliganPrompts).append('\n');
        sb.append("line_bucket_soft_targets: ").append(args.lineBucketSoftTargets).append('\n');
        sb.append("line_balance_key: ").append(args.lineBalanceKey).append('\n');
        sb.append("line_balance_singleton_limit: ").append(args.lineBalanceSingletonLimit).append('\n');
        sb.append("line_train_bottoms: ").append(args.lineTrainBottoms).append('\n');
        sb.append("line_train_epochs: ").append(args.lineTrainEpochs).append('\n');
        sb.append("collect_only: ").append(args.collectOnly).append('\n');
        sb.append("export_training_data_file: ").append(args.exportTrainingDataFile == null ? "" : args.exportTrainingDataFile).append('\n');
        sb.append("export_winning_line_data_file: ").append(args.exportWinningLineDataFile == null ? "" : args.exportWinningLineDataFile).append('\n');
        sb.append("trained_scenarios: ").append(trainedScenarios).append('\n');
        sb.append("skipped_scenarios: ").append(skippedScenarios).append('\n');
        sb.append("trained_pass_samples: ").append(trainedSamples).append('\n');
        sb.append("opponent_mode: ").append(args.opponentMode).append('\n');
        sb.append("batch_size: ").append(args.batchSize).append('\n');
        sb.append("workers: ").append(args.workers).append('\n');
        sb.append("timeout_sec: ").append(args.timeoutSec).append('\n');
        sb.append("seed: ").append(args.seed).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        sb.append("deck_list: ").append(args.deckList).append('\n');
        sb.append("decks:\n");
        for (Path deck : decks) {
            sb.append("- ").append(deck).append('\n');
        }
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static String csv(String s) {
        if (s == null) {
            return "";
        }
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private static final class MulliganWorkerThreadFactory implements ThreadFactory {

        private final AtomicInteger nextId = new AtomicInteger(1);

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "MULL-CF-WORKER-" + nextId.getAndIncrement());
            t.setDaemon(true);
            return t;
        }
    }

    private static final class PairJob {

        final int pair;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;

        private PairJob(int pair, Path agentDeck, Path oppDeck, long seed) {
            this.pair = pair;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
        }
    }

    private static final class PairOutcome {

        final SampleRecord record;
        final StateSequenceBuilder.TrainingData trainingData;
        final String label;
        final BranchResult keep;
        final BranchResult mull;

        private PairOutcome(SampleRecord record, StateSequenceBuilder.TrainingData trainingData,
                            String label, BranchResult keep, BranchResult mull) {
            this.record = record;
            this.trainingData = trainingData;
            this.label = label;
            this.keep = keep;
            this.mull = mull;
        }

        static PairOutcome skipped(PairJob job, String label, BranchResult keep, BranchResult mull) {
            SampleRecord record = SampleRecord.skipped(job.pair, job.agentDeck, job.oppDeck, job.seed, keep, mull);
            return new PairOutcome(record, null, label, keep, mull);
        }
    }

    private static final class ScenarioJob {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;

        private ScenarioJob(int scenario, Path agentDeck, Path oppDeck, long seed) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
        }
    }

    private static final class LineSpec {

        final int keepAfterMulls;
        final int bottomCombo;
        final String name;

        private LineSpec(int keepAfterMulls, int bottomCombo) {
            this.keepAfterMulls = Math.max(0, keepAfterMulls);
            this.bottomCombo = bottomCombo;
            this.name = this.keepAfterMulls == 0
                    ? "keep7"
                    : "mull" + this.keepAfterMulls + "_bottomCombo" + bottomCombo;
        }

        int forcedChoiceAtPrompt(int promptIndex) {
            return promptIndex < keepAfterMulls ? 1 : 0;
        }
    }

    private static final class LineStats {

        final int samples;
        final int wins;

        private LineStats(int samples, int wins) {
            this.samples = samples;
            this.wins = wins;
        }

        boolean hasSamples() {
            return samples > 0;
        }

        double winRate() {
            return samples == 0 ? Double.NaN : (double) wins / (double) samples;
        }
    }

    private static final class LineTrainingExample {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final String kind;
        final String balanceKey;
        final int label;
        final float targetKeep;
        final float targetMull;
        final int lands;
        final int effectiveLands;
        final String hand;
        final StateSequenceBuilder.TrainingData trainingData;

        private LineTrainingExample(ScenarioJob job, LineResult result, String kind,
                                    String balanceKey, int label, float targetKeep, float targetMull,
                                    StateSequenceBuilder.TrainingData trainingData) {
            this.scenario = job.scenario;
            this.agentDeck = job.agentDeck;
            this.oppDeck = job.oppDeck;
            this.kind = kind == null ? "" : kind;
            this.balanceKey = balanceKey == null ? "" : balanceKey;
            this.label = label;
            this.targetKeep = targetKeep;
            this.targetMull = targetMull;
            this.lands = result == null ? -1 : result.lands;
            this.hand = result == null ? "" : result.hand;
            this.effectiveLands = effectiveLandCount(job.agentDeck, this.lands, this.hand);
            this.trainingData = trainingData;
        }

        private LineTrainingExample(int scenario, Path agentDeck, Path oppDeck, String kind,
                                    String balanceKey, int label, float targetKeep, float targetMull,
                                    int lands, int effectiveLands, String hand,
                                    StateSequenceBuilder.TrainingData trainingData) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.kind = kind == null ? "" : kind;
            this.balanceKey = balanceKey == null ? "" : balanceKey;
            this.label = label;
            this.targetKeep = targetKeep;
            this.targetMull = targetMull;
            this.lands = lands;
            this.effectiveLands = effectiveLands;
            this.hand = hand == null ? "" : hand;
            this.trainingData = trainingData;
        }

        static LineTrainingExample binary(ScenarioJob job, LineResult result, String kind,
                                          StateSequenceBuilder.TrainingData trainingData,
                                          double keepRate, double mullRate,
                                          String balanceKeyMode) {
            float targetKeep = trainingData != null && trainingData.mctsVisitTargets != null
                    ? trainingData.mctsVisitTargets[0]
                    : (float) keepRate;
            float targetMull = trainingData != null && trainingData.mctsVisitTargets != null
                    ? trainingData.mctsVisitTargets[1]
                    : (float) mullRate;
            int label = targetKeep >= targetMull ? 0 : 1;
            return new LineTrainingExample(job, result, kind,
                    balanceKey(job.agentDeck, result, balanceKeyMode),
                    label, targetKeep, targetMull, trainingData);
        }

        static LineTrainingExample bottom(ScenarioJob job, LineResult result,
                                          StateSequenceBuilder.TrainingData trainingData,
                                          String balanceKeyMode) {
            return new LineTrainingExample(job, result, "BOTTOM",
                    balanceKey(job.agentDeck, result, balanceKeyMode),
                    -1, 0.0f, 0.0f, trainingData);
        }

        LineTrainingExample withBinaryTarget(float keepTarget, float mullTarget) {
            float keep = Math.max(0.0f, keepTarget);
            float mull = Math.max(0.0f, mullTarget);
            float sum = Math.max(1e-9f, keep + mull);
            keep /= sum;
            mull /= sum;
            float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
            target[0] = keep;
            target[1] = mull;
            int newLabel = keep >= mull ? 0 : 1;
            StateSequenceBuilder.TrainingData td = copyWithTargets(
                    trainingData,
                    newLabel,
                    1,
                    target,
                    StateSequenceBuilder.ActionType.MULLIGAN
            );
            return new LineTrainingExample(
                    scenario,
                    agentDeck,
                    oppDeck,
                    kind,
                    balanceKey,
                    newLabel,
                    keep,
                    mull,
                    lands,
                    effectiveLands,
                    hand,
                    td
            );
        }

        boolean isBinaryMulligan() {
            return trainingData != null
                    && trainingData.actionType == StateSequenceBuilder.ActionType.MULLIGAN
                    && label >= 0;
        }

        String toCsv() {
            return scenario
                    + "," + csv(kind)
                    + "," + csv(balanceKey)
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + label
                    + "," + String.format(Locale.US, "%.6f", targetKeep)
                    + "," + String.format(Locale.US, "%.6f", targetMull)
                    + "," + lands
                    + "," + effectiveLands
                    + "," + csv(hand);
        }
    }

    private static final class LineOutcome {

        final List<LineRecord> records;
        final List<LineTrainingExample> trainingExamples;
        final List<StateSequenceBuilder.TrainingData> winningTrainingData;
        final String label;

        private LineOutcome(List<LineRecord> records, List<LineTrainingExample> trainingExamples,
                            List<StateSequenceBuilder.TrainingData> winningTrainingData, String label) {
            this.records = records;
            this.trainingExamples = trainingExamples;
            this.winningTrainingData = winningTrainingData == null ? Collections.emptyList() : winningTrainingData;
            this.label = label;
        }
    }

    private static final class LinePlayer extends ComputerPlayerRL {

        private final LineSpec spec;
        private int promptIndex = 0;
        private boolean promptCaptured = false;
        private String hand = "";
        private int lands = -1;

        private LinePlayer(String name, LineSpec spec) {
            super(name, RangeOfInfluence.ALL, RLTrainer.sharedModel, false, true, "train");
            this.spec = spec;
        }

        @Override
        protected Integer forcedMulliganChoiceIndex(Game game, int handSize, int landCount) {
            return spec.forcedChoiceAtPrompt(promptIndex);
        }

        @Override
        protected <T> List<Integer> forcedChoiceIndices(
                StateSequenceBuilder.ActionType actionType,
                List<T> candidates,
                int maxTargets,
                int minTargets,
                Game game,
                mage.abilities.Ability source
        ) {
            if (actionType != StateSequenceBuilder.ActionType.LONDON_MULLIGAN || spec.keepAfterMulls <= 0) {
                return null;
            }
            if (spec.bottomCombo < 0) {
                return null;
            }
            int candidateCount = candidates == null ? 0 : candidates.size();
            int bottomCount = Math.min(spec.keepAfterMulls, candidateCount);
            if (candidateCount <= 0 || bottomCount <= 0) {
                return null;
            }
            List<Integer> bottom = nthCombination(candidateCount, bottomCount, spec.bottomCombo);
            Set<Integer> bottomSet = new java.util.HashSet<>(bottom);
            List<Integer> ranking = new ArrayList<>();
            for (int i = 0; i < candidateCount; i++) {
                if (!bottomSet.contains(i)) {
                    ranking.add(i);
                }
            }
            for (Integer idx : bottom) {
                if (idx != null && idx >= 0 && idx < candidateCount) {
                    ranking.add(idx);
                }
            }
            return ranking;
        }

        @Override
        public boolean chooseMulligan(Game game) {
            boolean firstPrompt = !promptCaptured;
            String beforeHand = firstPrompt ? handCards(game) : "";
            int beforeLands = firstPrompt ? countLands(game) : -1;
            boolean decision = super.chooseMulligan(game);
            if (firstPrompt) {
                promptCaptured = true;
                hand = beforeHand;
                lands = beforeLands;
            }
            promptIndex++;
            return decision;
        }

        private int countLands(Game game) {
            int count = 0;
            try {
                for (Card card : getHand().getCards(game)) {
                    if (card != null && card.isLand(game)) {
                        count++;
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }

        private String handCards(Game game) {
            try {
                return getHand().getCards(game).stream()
                        .map(Card::getName)
                        .collect(Collectors.joining("; "));
            } catch (Exception ignored) {
                return "";
            }
        }
    }

    private static final class LineResult {

        final LineSpec spec;
        final boolean won;
        final boolean timedOut;
        final String error;
        final List<StateSequenceBuilder.TrainingData> data;
        final String hand;
        final int lands;
        final int turns;

        private LineResult(LineSpec spec, boolean won, boolean timedOut, String error,
                           List<StateSequenceBuilder.TrainingData> data, String hand, int lands, int turns) {
            this.spec = spec;
            this.won = won;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.data = data == null ? Collections.emptyList() : data;
            this.hand = hand == null ? "" : hand;
            this.lands = lands;
            this.turns = turns;
        }

        static LineResult failed(LineSpec spec, String error) {
            return new LineResult(spec, false, false, error, Collections.emptyList(), "", -1, -1);
        }

        StateSequenceBuilder.TrainingData firstMulliganData() {
            return nthMulliganData(0);
        }

        StateSequenceBuilder.TrainingData nthMulliganData(int index) {
            int seen = 0;
            for (StateSequenceBuilder.TrainingData td : data) {
                if (td.actionType == StateSequenceBuilder.ActionType.MULLIGAN) {
                    if (seen == index) {
                        return td;
                    }
                    seen++;
                }
            }
            return null;
        }

        StateSequenceBuilder.TrainingData bottomData() {
            for (StateSequenceBuilder.TrainingData td : data) {
                if (td.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN) {
                    return td;
                }
            }
            return null;
        }
    }

    private static final class LineRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final String line;
        final int keepAfterMulls;
        final int bottomCombo;
        final boolean won;
        final boolean timedOut;
        final int turns;
        final int lands;
        final String hand;
        final String error;

        private LineRecord(int scenario, Path agentDeck, Path oppDeck, long seed, String line,
                           int keepAfterMulls, int bottomCombo, boolean won, boolean timedOut,
                           int turns, int lands, String hand, String error) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.line = line;
            this.keepAfterMulls = keepAfterMulls;
            this.bottomCombo = bottomCombo;
            this.won = won;
            this.timedOut = timedOut;
            this.turns = turns;
            this.lands = lands;
            this.hand = hand == null ? "" : hand;
            this.error = error == null ? "" : error;
        }

        static LineRecord from(ScenarioJob job, LineResult result) {
            return new LineRecord(job.scenario, job.agentDeck, job.oppDeck, job.seed,
                    result.spec.name, result.spec.keepAfterMulls, result.spec.bottomCombo,
                    result.won, result.timedOut, result.turns, result.lands, result.hand, result.error);
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + csv(line)
                    + "," + keepAfterMulls
                    + "," + bottomCombo
                    + "," + won
                    + "," + timedOut
                    + "," + turns
                    + "," + lands
                    + "," + csv(hand)
                    + "," + csv(error);
        }
    }

    private static final class BranchPlayer extends ComputerPlayerRL {

        private final int forcedChoice;
        private boolean forceUsed = false;
        private boolean promptCaptured = false;
        private String hand = "";
        private int lands = -1;
        private float pKeep = Float.NaN;
        private float pMull = Float.NaN;

        private BranchPlayer(String name, int forcedChoice) {
            super(name, RangeOfInfluence.ALL, RLTrainer.sharedModel, false, true, "train");
            this.forcedChoice = forcedChoice;
        }

        @Override
        protected Integer forcedMulliganChoiceIndex(Game game, int handSize, int landCount) {
            if (forceUsed) {
                return null;
            }
            forceUsed = true;
            return forcedChoice;
        }

        @Override
        public boolean chooseMulligan(Game game) {
            boolean firstPrompt = !promptCaptured;
            String beforeHand = firstPrompt ? handCards(game) : "";
            int beforeLands = firstPrompt ? countLands(game) : -1;
            boolean decision = super.chooseMulligan(game);
            if (firstPrompt) {
                promptCaptured = true;
                hand = beforeHand;
                lands = beforeLands;
                pKeep = getLastMulliganPKeep();
                pMull = getLastMulliganPMull();
            }
            return decision;
        }

        private int countLands(Game game) {
            int count = 0;
            try {
                for (Card card : getHand().getCards(game)) {
                    if (card != null && card.isLand(game)) {
                        count++;
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }

        private String handCards(Game game) {
            try {
                return getHand().getCards(game).stream()
                        .map(Card::getName)
                        .collect(Collectors.joining("; "));
            } catch (Exception ignored) {
                return "";
            }
        }
    }

    private static final class BranchResult {
        final String branch;
        final boolean won;
        final boolean timedOut;
        final String error;
        final StateSequenceBuilder.TrainingData trainingData;
        final String hand;
        final int lands;
        final float pKeep;
        final float pMull;
        final int turns;

        private BranchResult(String branch, boolean won, boolean timedOut, String error,
                             StateSequenceBuilder.TrainingData trainingData, String hand, int lands,
                             float pKeep, float pMull, int turns) {
            this.branch = branch;
            this.won = won;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.trainingData = trainingData;
            this.hand = hand == null ? "" : hand;
            this.lands = lands;
            this.pKeep = pKeep;
            this.pMull = pMull;
            this.turns = turns;
        }

        static BranchResult failed(String branch, String error) {
            return new BranchResult(branch, false, false, error, null, "", -1, Float.NaN, Float.NaN, -1);
        }
    }

    private static final class SampleRecord {
        final int pair;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final String status;
        final String label;
        final BranchResult keep;
        final BranchResult mull;
        final float targetKeep;
        final float targetMull;

        private SampleRecord(int pair, Path agentDeck, Path oppDeck, long seed, String status, String label,
                             BranchResult keep, BranchResult mull, float targetKeep, float targetMull) {
            this.pair = pair;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.status = status;
            this.label = label;
            this.keep = keep;
            this.mull = mull;
            this.targetKeep = targetKeep;
            this.targetMull = targetMull;
        }

        static SampleRecord trained(int pair, Path agentDeck, Path oppDeck, long seed,
                                    BranchResult keep, BranchResult mull, String label,
                                    float targetKeep, float targetMull) {
            return new SampleRecord(pair, agentDeck, oppDeck, seed, "trained", label, keep, mull, targetKeep, targetMull);
        }

        static SampleRecord skipped(int pair, Path agentDeck, Path oppDeck, long seed,
                                    BranchResult keep, BranchResult mull) {
            return new SampleRecord(pair, agentDeck, oppDeck, seed, "skipped", "", keep, mull, 0.0f, 0.0f);
        }

        String toCsv() {
            return pair
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + csv(status)
                    + "," + csv(label)
                    + "," + keep.won
                    + "," + mull.won
                    + "," + targetKeep
                    + "," + targetMull
                    + "," + keep.lands
                    + "," + mull.lands
                    + "," + keep.pKeep
                    + "," + keep.pMull
                    + "," + mull.pKeep
                    + "," + mull.pMull
                    + "," + keep.turns
                    + "," + mull.turns
                    + "," + csv(keep.error)
                    + "," + csv(mull.error)
                    + "," + csv(keep.hand);
        }
    }

    private static final class Args {
        String deckList = DEFAULT_DECK_LIST;
        String agentDeckList = null;
        String oppDeckList = null;
        Path outDir = null;
        int pairs = 128;
        int batchSize = 32;
        int timeoutSec = 45;
        int reportEvery = 25;
        int workers = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        int postTrainWaitMs = 3000;
        long seed = System.currentTimeMillis();
        boolean trainTies = true;
        String opponentMode = "rl";
        int cp7Skill = 7;
        boolean lineMode = false;
        int lineMaxMulls = 2;
        int lineBottomCombos = 0;
        double lineMarginMin = 0.05;
        double lineTargetTemperature = 0.50;
        boolean lineBalanceMulliganPrompts = true;
        boolean lineBucketSoftTargets = false;
        String lineBalanceKey = "deck-resource";
        int lineBalanceSingletonLimit = 1;
        boolean lineTrainBottoms = true;
        int lineTrainEpochs = 1;
        boolean collectOnly = false;
        Path exportTrainingDataFile = null;
        Path exportWinningLineDataFile = null;

        static Args parse(String[] args) {
            Args out = new Args();
            Map<String, String> kv = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if (!a.startsWith("--")) {
                    continue;
                }
                String key = a.substring(2);
                String value = "true";
                int eq = key.indexOf('=');
                if (eq >= 0) {
                    value = key.substring(eq + 1);
                    key = key.substring(0, eq);
                } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    value = args[++i];
                }
                kv.put(key, value);
            }
            if (kv.containsKey("deck-list")) out.deckList = kv.get("deck-list");
            if (kv.containsKey("agent-deck-list")) out.agentDeckList = kv.get("agent-deck-list");
            if (kv.containsKey("opp-deck-list")) out.oppDeckList = kv.get("opp-deck-list");
            if (kv.containsKey("out")) out.outDir = Paths.get(kv.get("out")).toAbsolutePath().normalize();
            if (kv.containsKey("pairs")) out.pairs = Integer.parseInt(kv.get("pairs"));
            if (kv.containsKey("batch-size")) out.batchSize = Integer.parseInt(kv.get("batch-size"));
            if (kv.containsKey("timeout-sec")) out.timeoutSec = Integer.parseInt(kv.get("timeout-sec"));
            if (kv.containsKey("report-every")) out.reportEvery = Integer.parseInt(kv.get("report-every"));
            if (kv.containsKey("workers")) out.workers = Integer.parseInt(kv.get("workers"));
            if (kv.containsKey("post-train-wait-ms")) out.postTrainWaitMs = Integer.parseInt(kv.get("post-train-wait-ms"));
            if (kv.containsKey("seed")) out.seed = Long.parseLong(kv.get("seed"));
            if (kv.containsKey("train-ties")) out.trainTies = Boolean.parseBoolean(kv.get("train-ties"));
            if (kv.containsKey("opponent")) out.opponentMode = kv.get("opponent").trim().toLowerCase(Locale.ROOT);
            if (kv.containsKey("cp7-skill")) out.cp7Skill = Integer.parseInt(kv.get("cp7-skill"));
            if (kv.containsKey("line-mode")) out.lineMode = Boolean.parseBoolean(kv.get("line-mode"));
            if (kv.containsKey("line-max-mulls")) out.lineMaxMulls = Integer.parseInt(kv.get("line-max-mulls"));
            if (kv.containsKey("line-bottom-combos")) out.lineBottomCombos = Integer.parseInt(kv.get("line-bottom-combos"));
            if (kv.containsKey("line-margin-min")) out.lineMarginMin = Double.parseDouble(kv.get("line-margin-min"));
            if (kv.containsKey("line-target-temperature")) out.lineTargetTemperature = Double.parseDouble(kv.get("line-target-temperature"));
            if (kv.containsKey("line-balance-mulligan-prompts")) out.lineBalanceMulliganPrompts = Boolean.parseBoolean(kv.get("line-balance-mulligan-prompts"));
            if (kv.containsKey("line-bucket-soft-targets")) out.lineBucketSoftTargets = Boolean.parseBoolean(kv.get("line-bucket-soft-targets"));
            if (kv.containsKey("line-balance-key")) out.lineBalanceKey = kv.get("line-balance-key").trim().toLowerCase(Locale.ROOT);
            if (kv.containsKey("line-balance-singleton-limit")) out.lineBalanceSingletonLimit = Integer.parseInt(kv.get("line-balance-singleton-limit"));
            if (kv.containsKey("line-train-bottoms")) out.lineTrainBottoms = Boolean.parseBoolean(kv.get("line-train-bottoms"));
            if (kv.containsKey("line-train-epochs")) out.lineTrainEpochs = Integer.parseInt(kv.get("line-train-epochs"));
            if (kv.containsKey("collect-only")) out.collectOnly = Boolean.parseBoolean(kv.get("collect-only"));
            if (kv.containsKey("export-training-data-file")) out.exportTrainingDataFile = Paths.get(kv.get("export-training-data-file")).toAbsolutePath().normalize();
            if (kv.containsKey("export-winning-line-data-file")) out.exportWinningLineDataFile = Paths.get(kv.get("export-winning-line-data-file")).toAbsolutePath().normalize();
            out.pairs = Math.max(1, out.pairs);
            out.batchSize = Math.max(1, out.batchSize);
            out.timeoutSec = Math.max(1, out.timeoutSec);
            out.reportEvery = Math.max(1, out.reportEvery);
            out.workers = Math.max(1, Math.min(out.workers, out.pairs));
            out.lineMaxMulls = Math.max(0, Math.min(7, out.lineMaxMulls));
            out.lineBottomCombos = Math.max(-1, out.lineBottomCombos);
            out.lineMarginMin = Math.max(0.0, Math.min(1.0, out.lineMarginMin));
            out.lineTargetTemperature = Math.max(0.01, Math.min(10.0, out.lineTargetTemperature));
            out.lineBalanceSingletonLimit = Math.max(0, out.lineBalanceSingletonLimit);
            out.lineTrainEpochs = Math.max(1, Math.min(100, out.lineTrainEpochs));
            if (!"deck-resource".equals(out.lineBalanceKey)
                    && !"deck".equals(out.lineBalanceKey)
                    && !"resource".equals(out.lineBalanceKey)) {
                throw new IllegalArgumentException("--line-balance-key must be deck-resource, deck, or resource");
            }
            if (!"rl".equals(out.opponentMode) && !"cp7".equals(out.opponentMode)) {
                throw new IllegalArgumentException("--opponent must be rl or cp7");
            }
            return out;
        }
    }
}
