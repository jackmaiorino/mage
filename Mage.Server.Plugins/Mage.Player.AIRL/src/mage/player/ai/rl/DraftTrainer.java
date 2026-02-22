package mage.player.ai.rl;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.ComputerPlayerRL.DraftStepData;
import mage.util.ThreadUtils;

/**
 * Co-training loop for the Vintage Cube Draft Model.
 *
 * Each episode:
 *   1. Run an 8-player draft (RL agent + 7 heuristic bots)
 *   2. RL agent constructs a deck (construction head + heuristic land base)
 *   3. Self-play games: RL game model plays both sides
 *      - RL-drafted deck vs heuristic-drafted deck
 *   4. Draft reward = RL win rate across evaluation games
 *   5. Train draft model via PPO (per-pick + construction advantages)
 *   6. Game model also trains from self-play game data (handled by RLTrainer/shared model)
 *
 * Run as: mvn exec:java -Dexec.mainClass=mage.player.ai.rl.DraftTrainer -Dexec.args=train
 */
public class DraftTrainer {

    private static final Logger logger = Logger.getLogger(DraftTrainer.class);

    // Configuration
    private static final int NUM_DRAFT_EPISODES   = EnvConfig.i32("DRAFT_TOTAL_EPISODES", 5000);
    private static final int GAMES_PER_EPISODE    = EnvConfig.i32("DRAFT_GAMES_PER_EPISODE", 3);
    private static final int SAVE_EVERY           = EnvConfig.i32("DRAFT_SAVE_EVERY", 50);
    private static final int DRAFT_LOG_FREQUENCY  = EnvConfig.i32("DRAFT_LOG_FREQUENCY", 10);
    private static final int STATS_EVERY          = EnvConfig.i32("DRAFT_STATS_EVERY", 10);
    private static final int WINRATE_WINDOW       = EnvConfig.i32("DRAFT_WINRATE_WINDOW", 100);
    private static final boolean DRAFT_EVAL_GAME_LOGGING =
            EnvConfig.bool("DRAFT_EVAL_GAME_LOGGING", false)
                    || EnvConfig.bool("GAME_LOGGING", false);
    private static final int DRAFT_GAME_LOG_FREQUENCY =
            EnvConfig.i32("DRAFT_GAME_LOG_FREQUENCY",
                    EnvConfig.i32("GAME_LOG_FREQUENCY", 0));
    private static final boolean DRAFT_BENCHMARK_ENABLE =
            EnvConfig.bool("DRAFT_BENCHMARK_ENABLE", true);
    private static final int DRAFT_BENCHMARK_EVERY =
            EnvConfig.i32("DRAFT_BENCHMARK_EVERY", 5000);
    private static final int DRAFT_BENCHMARK_DRAFTS =
            EnvConfig.i32("DRAFT_BENCHMARK_DRAFTS", 12);
    private static final int DRAFT_BENCHMARK_GAMES_PER_DRAFT =
            EnvConfig.i32("DRAFT_BENCHMARK_GAMES_PER_DRAFT", 3);
    private static final double DRAFT_BENCHMARK_PROMOTE_WR =
            EnvConfig.f64("DRAFT_BENCHMARK_PROMOTE_WR", 0.55);
    private static final boolean DRAFT_BENCHMARK_GAME_LOGGING =
            EnvConfig.bool("DRAFT_BENCHMARK_GAME_LOGGING", false);
    private static final int DRAFT_BENCHMARK_GAME_LOG_FREQUENCY =
            EnvConfig.i32("DRAFT_BENCHMARK_GAME_LOG_FREQUENCY", 0);
    private static final boolean DRAFT_COTRAIN_GAME_MODEL =
            EnvConfig.bool("DRAFT_COTRAIN_GAME_MODEL", true);
    private static final boolean DRAFT_COTRAIN_MULLIGAN_MODEL =
            EnvConfig.bool("DRAFT_COTRAIN_MULLIGAN_MODEL", true);
    private static final boolean DRAFT_COTRAIN_BOTH_SIDES =
            EnvConfig.bool("DRAFT_COTRAIN_BOTH_SIDES", true);
    private static final boolean DRAFT_COTRAIN_ON_BENCHMARK =
            EnvConfig.bool("DRAFT_COTRAIN_ON_BENCHMARK", false);
    private static final int DRAFT_MULLIGAN_SAVE_INTERVAL =
            EnvConfig.i32("DRAFT_MULLIGAN_SAVE_INTERVAL",
                    EnvConfig.i32("MULLIGAN_SAVE_INTERVAL", 100));
    private static final boolean DRAFT_SHUTDOWN_SAVE_HOOK =
            EnvConfig.bool("DRAFT_SHUTDOWN_SAVE_HOOK", true);
    private static final boolean DRAFT_SHUTDOWN_SAVE_GAME_MODELS =
            EnvConfig.bool("DRAFT_SHUTDOWN_SAVE_GAME_MODELS", true);
    private static final boolean DRAFT_SAVE_GAME_MODELS =
            EnvConfig.bool("DRAFT_SAVE_GAME_MODELS", true);
    private static final int DRAFT_SAVE_GAME_MODELS_EVERY =
            Math.max(1, EnvConfig.i32("DRAFT_SAVE_GAME_MODELS_EVERY", SAVE_EVERY));

    // Paths
    private static final String DRAFT_MODEL_PATH  = RLLogPaths.DRAFT_MODEL_FILE_PATH;
    private static final String EPISODE_COUNT_PATH = RLLogPaths.DRAFT_EPISODE_COUNT_PATH;
    private static final String STATS_PATH        = RLLogPaths.DRAFT_STATS_PATH;
    private static final String BENCHMARK_STATS_PATH = RLLogPaths.DRAFT_BENCHMARK_STATS_PATH;
    private static final String BENCHMARK_STATE_PATH = RLLogPaths.DRAFT_BENCHMARK_STATE_PATH;

    private final DraftPythonMLBridge draftBridge;
    private final PythonModel gameModel;
    private final DraftRunner draftRunner;

    // Rolling win tracking
    private final java.util.Deque<Boolean> recentWins = new java.util.ArrayDeque<>();
    private final AtomicInteger totalWins = new AtomicInteger(0);
    private final AtomicInteger totalGames = new AtomicInteger(0);
    private final AtomicInteger evalGameCounter = new AtomicInteger(0);
    private final AtomicInteger benchmarkGameCounter = new AtomicInteger(0);
    private final AtomicInteger mulliganTrainCount = new AtomicInteger(0);
    private final AtomicInteger currentEpisodeForPersistence = new AtomicInteger(0);
    private final AtomicInteger shutdownSaveOnce = new AtomicInteger(0);
    private final Object persistenceLock = new Object();

    private int startEpisode = 0;
    private int benchmarkTier = 0;
    private int lastBenchmarkEpisode = -1;
    private volatile boolean shutdownHookRegistered = false;

    public DraftTrainer() {
        this.draftBridge = DraftPythonMLBridge.getInstance();
        this.gameModel = PythonMLService.getInstance();
        this.draftRunner = new DraftRunner();
        loadBenchmarkState();
    }

    // -----------------------------------------------------------------------
    // Main training loop
    // -----------------------------------------------------------------------

    public void train() {
        startEpisode = loadEpisodeCount();
        currentEpisodeForPersistence.set(startEpisode);
        logger.info(String.format("Starting draft training at episode %d (target: %d)",
                startEpisode, NUM_DRAFT_EPISODES));
        registerShutdownHookIfEnabled();

        // Ensure output directories exist
        try {
            Files.createDirectories(Paths.get(RLLogPaths.DRAFT_LOGS_BASE_DIR));
            Files.createDirectories(Paths.get(RLLogPaths.DRAFT_MODELS_BASE_DIR));
        } catch (IOException e) {
            logger.warn("Could not create draft directories: " + e.getMessage());
        }

        Random rand = new Random();
        int episode = startEpisode;

        while (episode < NUM_DRAFT_EPISODES) {
            long t0 = System.currentTimeMillis();

            try {
                boolean logThis = (DRAFT_LOG_FREQUENCY > 0)
                        && (episode == 0 || episode % DRAFT_LOG_FREQUENCY == 0);
                DraftLogger draftLogger = DraftLogger.create(logThis);

                // 1. Create RL player and reset draft state
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL(
                        "DraftRL", RangeOfInfluence.ALL, gameModel);
                rlPlayer.resetDraftState(draftBridge, draftLogger, episode, logThis);

                // 2. Run draft (RL agent uses pick_head via pickCard override)
                DraftRunner.DraftResult draftResult = draftRunner.runDraft(rlPlayer);
                if (draftResult == null) {
                    logger.warn("Draft failed for episode " + episode + ", skipping");
                    episode++;
                    continue;
                }

                // 3. Construct deck with construction head
                Deck rlDeck = rlPlayer.constructDraftDeck();
                if (rlDeck == null || rlDeck.getCards().isEmpty()) {
                    logger.warn("Empty RL deck at episode " + episode + ", skipping");
                    episode++;
                    continue;
                }

                // Ensure minimum 40 cards - pad with basic lands if needed
                rlDeck = padDeckToMinimum(rlDeck, 40);

                // Pick a heuristic-drafted deck for the opponent side
                Deck heuristicDeck = draftResult.getRandomHeuristicDeck(rand);
                if (heuristicDeck == null
                        || (heuristicDeck.getCards().isEmpty() && heuristicDeck.getSideboard().isEmpty())) {
                    logger.warn("No heuristic deck at episode " + episode);
                    episode++;
                    continue;
                }
                // Build a playable heuristic deck from sideboard
                Deck playableHeuristicDeck = buildHeuristicDeckFromSideboard(heuristicDeck);
                if (playableHeuristicDeck == null || playableHeuristicDeck.getCards().isEmpty()) {
                    logger.warn("Heuristic playable deck construction failed at episode " + episode);
                    episode++;
                    continue;
                }

                // 4. Self-play evaluation games
                int wins = 0;
                int losses = 0;
                for (int g = 0; g < GAMES_PER_EPISODE; g++) {
                    boolean rlWon = playEvaluationGame(rlDeck, playableHeuristicDeck);
                    if (rlWon) { wins++; } else { losses++; }
                }

                float winRate = wins / (float) GAMES_PER_EPISODE;
                float reward  = winRate * 2.0f - 1.0f; // [-1, +1]

                // 5. Update rolling win stats
                recentWins.add(winRate > 0.5f);
                if (recentWins.size() > WINRATE_WINDOW) recentWins.poll();
                totalGames.addAndGet(GAMES_PER_EPISODE);
                totalWins.addAndGet(wins);

                // 6. Train draft model
                List<DraftStepData> steps = rlPlayer.getDraftSteps();
                if (!steps.isEmpty()) {
                    trainDraftModel(steps, reward);
                }

                // 7. Log
                if (logThis) {
                    draftLogger.logEpisodeResults(wins, losses, reward, null);
                    draftLogger.close();
                }

                long elapsed = System.currentTimeMillis() - t0;
                double rollingWr = getRollingWinRate();

                if (episode % STATS_EVERY == 0) {
                    logger.info(String.format(
                            "[Draft ep %d] W=%d L=%d reward=%.3f rolling_wr=%.3f (%dms)",
                            episode, wins, losses, reward, rollingWr, elapsed));
                    appendEpisodeStats(episode, wins, losses, reward, rollingWr);
                }

                // 8. Periodic benchmark tick (separate from per-episode reward games)
                maybeRunBenchmarkTick(episode);

                // 9. Periodic save
                if (episode > 0 && episode % SAVE_EVERY == 0) {
                    draftBridge.saveDraftModel(DRAFT_MODEL_PATH);
                    saveEpisodeCount(episode);
                    maybeSaveGameModelsPeriodic(episode);
                    logger.info("Saved draft model at episode " + episode);
                }

                episode++;
                currentEpisodeForPersistence.set(episode);

            } catch (Exception e) {
                logger.error("Error at draft episode " + episode + ": " + e.getMessage(), e);
                episode++;
                currentEpisodeForPersistence.set(episode);
            }
        }

        // Final save
        saveCheckpoint("final");
        logger.info("Draft training complete. Total episodes: " + episode);
    }

    // -----------------------------------------------------------------------
    // Game self-play
    // -----------------------------------------------------------------------

    /**
     * Play one evaluation game: RL game model plays both sides.
     * RL-drafted deck on one side, heuristic-drafted deck on the other.
     *
     * @return true if RL-drafted deck side won
     */
    private boolean playEvaluationGame(Deck rlDeck, Deck heuristicDeck) {
        return playMatchGame(rlDeck, heuristicDeck, false);
    }

    private boolean playBenchmarkGame(Deck rlDeck, Deck heuristicDeck) {
        return playMatchGame(rlDeck, heuristicDeck, true);
    }

    private boolean playMatchGame(Deck rlDeck, Deck heuristicDeck, boolean benchmarkMode) {
        String oldThreadName = Thread.currentThread().getName();
        boolean renamedToGameThread = false;
        if (!oldThreadName.startsWith(ThreadUtils.THREAD_PREFIX_GAME)) {
            Thread.currentThread().setName(ThreadUtils.THREAD_PREFIX_GAME);
            renamedToGameThread = true;
        }
        AtomicInteger gameCounter = benchmarkMode ? benchmarkGameCounter : evalGameCounter;
        int gameNum = gameCounter.incrementAndGet();
        boolean alwaysLog = benchmarkMode ? DRAFT_BENCHMARK_GAME_LOGGING : DRAFT_EVAL_GAME_LOGGING;
        int every = benchmarkMode ? DRAFT_BENCHMARK_GAME_LOG_FREQUENCY : DRAFT_GAME_LOG_FREQUENCY;
        boolean enableGameLogging = alwaysLog || (every > 0 && (gameNum == 1 || gameNum % every == 0));
        GameLogger prevGameLogger = RLTrainer.threadLocalGameLogger.get();
        GameLogger gameLogger = benchmarkMode
                ? GameLogger.createForDraftBenchmark(enableGameLogging)
                : GameLogger.createForDraftEval(enableGameLogging);
        RLTrainer.threadLocalGameLogger.set(gameLogger);
        try {
            TwoPlayerMatch match = new TwoPlayerMatch(
                    new MatchOptions(benchmarkMode ? "DraftBenchmark" : "DraftEval",
                            benchmarkMode ? "DraftBenchmark" : "DraftEval", false));
            match.startGame();
            Game game = match.getGames().get(0);

            // RL player uses game model (no draft model needed here)
            ComputerPlayerRL rlSide = benchmarkMode
                    ? new ComputerPlayerRL("RLSide", RangeOfInfluence.ALL, gameModel, true, false, "train")
                    : new ComputerPlayerRL("RLSide", RangeOfInfluence.ALL, gameModel);
            // Opponent also uses game model (self-play)
            ComputerPlayerRL heuristicSide = benchmarkMode
                    ? new ComputerPlayerRL("HeuristicSide", RangeOfInfluence.ALL, gameModel, true, false, "train")
                    : new ComputerPlayerRL("HeuristicSide", RangeOfInfluence.ALL, gameModel);

            game.addPlayer(rlSide, rlDeck);
            match.addPlayer(rlSide, rlDeck);
            game.addPlayer(heuristicSide, heuristicDeck);
            match.addPlayer(heuristicSide, heuristicDeck);

            game.loadCards(rlDeck.getCards(), rlSide.getId());
            game.loadCards(heuristicDeck.getCards(), heuristicSide.getId());

            GameOptions opts = new GameOptions();
            game.setGameOptions(opts);

            GameHealthMonitor monitor = GameHealthMonitor.createAndStart(game);
            game.start(rlSide.getId());
            monitor.stop();

            boolean killed = monitor.wasKilled();
            boolean rlWon;
            if (killed) {
                rlWon = false; // killed games count as losses
            } else {
                // Determine winner
                String winner = game.getWinner();
                rlWon = winner != null && winner.contains(rlSide.getName());
            }

            maybeCoTrainFromGame(game, rlSide, heuristicSide, rlWon, benchmarkMode);

            if (gameLogger.isEnabled()) {
                String winnerName = rlWon ? rlSide.getName() : heuristicSide.getName();
                String loserName = rlWon ? heuristicSide.getName() : rlSide.getName();
                gameLogger.logOutcome(winnerName, loserName, game.getTurnNum(),
                        killed
                                ? (benchmarkMode ? "Draft benchmark game killed by health monitor"
                                        : "Draft eval game killed by health monitor")
                                : benchmarkMode
                                ? String.format("Draft benchmark game %d", gameNum)
                                : String.format("Draft eval game %d", gameNum));
            }
            return rlWon;

        } catch (Exception e) {
            logger.warn("Error in " + (benchmarkMode ? "benchmark" : "evaluation") + " game: " + e.getMessage());
            if (gameLogger.isEnabled()) {
                gameLogger.log("Error in " + (benchmarkMode ? "benchmark" : "evaluation") + " game: " + e.getMessage());
            }
            return false;
        } finally {
            if (gameLogger.isEnabled()) {
                gameLogger.close();
            }
            RLTrainer.threadLocalGameLogger.set(prevGameLogger);
            if (renamedToGameThread) {
                Thread.currentThread().setName(oldThreadName);
            }
        }
    }

    private void maybeCoTrainFromGame(
            Game game,
            ComputerPlayerRL rlSide,
            ComputerPlayerRL heuristicSide,
            boolean rlWon,
            boolean benchmarkMode
    ) {
        if (!(DRAFT_COTRAIN_GAME_MODEL || DRAFT_COTRAIN_MULLIGAN_MODEL)) {
            return;
        }
        if (benchmarkMode && !DRAFT_COTRAIN_ON_BENCHMARK) {
            return;
        }
        try {
            if (DRAFT_COTRAIN_GAME_MODEL) {
                int rlSteps = enqueueGameTraining(rlSide, rlWon ? 1.0 : -1.0);
                int oppSteps = 0;
                if (DRAFT_COTRAIN_BOTH_SIDES) {
                    oppSteps = enqueueGameTraining(heuristicSide, rlWon ? -1.0 : 1.0);
                }
                gameModel.recordGameResult(rlSide.getLastValueScore(), rlWon);
                if (rlSteps > 0 || oppSteps > 0) {
                    logger.info(String.format(
                            "Draft co-train game model: rlSteps=%d oppSteps=%d mode=%s",
                            rlSteps, oppSteps, benchmarkMode ? "benchmark" : "reward"));
                }
            }

            if (DRAFT_COTRAIN_MULLIGAN_MODEL) {
                int turns = game != null ? game.getTurnNum() : 0;
                int rlMul = trainMulliganForPlayer(rlSide, rlWon, turns);
                int oppMul = 0;
                if (DRAFT_COTRAIN_BOTH_SIDES) {
                    oppMul = trainMulliganForPlayer(heuristicSide, !rlWon, turns);
                }
                if (rlMul > 0 || oppMul > 0) {
                    logger.info(String.format(
                            "Draft co-train mulligan: rlDecisions=%d oppDecisions=%d mode=%s",
                            rlMul, oppMul, benchmarkMode ? "benchmark" : "reward"));
                }
            }
        } catch (Exception e) {
            logger.warn("Draft co-train update failed: " + e.getMessage());
        }
    }

    private int enqueueGameTraining(ComputerPlayerRL player, double terminalReward) {
        if (player == null) {
            return 0;
        }
        List<StateSequenceBuilder.TrainingData> trajectory = player.getTrainingBuffer();
        if (trajectory.isEmpty()) {
            return 0;
        }
        List<Double> rewards = RLTrainer.calculateImmediateRewards(trajectory, terminalReward);
        gameModel.enqueueTraining(trajectory, rewards);
        return trajectory.size();
    }

    private int trainMulliganForPlayer(ComputerPlayerRL player, boolean won, int gameTurns) {
        if (player == null) {
            return 0;
        }
        try {
            List<float[]> features = player.getMulliganFeatures();
            List<Float> decisions = player.getMulliganDecisions();
            List<Boolean> overrides = player.getMulliganOverrides();

            if (features.isEmpty()) {
                return 0;
            }

            int batchSize = features.size();
            float outcome = won ? 1.0f : 0.0f;
            int featureSize = features.get(0).length;

            ByteBuffer featuresBuf = ByteBuffer.allocate(batchSize * featureSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float[] feat : features) {
                for (float f : feat) {
                    featuresBuf.putFloat(f);
                }
            }

            ByteBuffer decisionsBuf = ByteBuffer.allocate(batchSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (Float decision : decisions) {
                decisionsBuf.putFloat(decision);
            }

            ByteBuffer outcomesBuf = ByteBuffer.allocate(batchSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < batchSize; i++) {
                outcomesBuf.putFloat(outcome);
            }

            ByteBuffer gameLengthsBuf = ByteBuffer.allocate(batchSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < batchSize; i++) {
                gameLengthsBuf.putInt(Math.max(1, gameTurns));
            }

            float earlyLandScore = player.getEarlyLandScore();
            ByteBuffer earlyLandScoresBuf = ByteBuffer.allocate(batchSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < batchSize; i++) {
                earlyLandScoresBuf.putFloat(earlyLandScore);
            }

            ByteBuffer overridesBuf = ByteBuffer.allocate(batchSize * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (Boolean override : overrides) {
                overridesBuf.putFloat(override != null && override ? 1.0f : 0.0f);
            }

            gameModel.trainMulligan(
                    featuresBuf.array(),
                    decisionsBuf.array(),
                    outcomesBuf.array(),
                    gameLengthsBuf.array(),
                    earlyLandScoresBuf.array(),
                    overridesBuf.array(),
                    batchSize
            );
            player.clearMulliganData();
            maybeSaveDraftMulliganModel();
            return batchSize;
        } catch (Exception e) {
            logger.warn("Error training mulligan model from draft game: " + e.getMessage());
            return 0;
        }
    }

    private void maybeSaveDraftMulliganModel() {
        if (DRAFT_MULLIGAN_SAVE_INTERVAL <= 0) {
            return;
        }
        if (mulliganTrainCount.incrementAndGet() % DRAFT_MULLIGAN_SAVE_INTERVAL == 0) {
            try {
                gameModel.saveMulliganModel();
                logger.info("Draft co-train: mulligan model saved (updates=" + mulliganTrainCount.get() + ")");
            } catch (Exception e) {
                logger.warn("Draft co-train: failed to save mulligan model: " + e.getMessage());
            }
        }
    }

    private void maybeRunBenchmarkTick(int episode) {
        if (!DRAFT_BENCHMARK_ENABLE || DRAFT_BENCHMARK_EVERY <= 0) {
            return;
        }
        if (episode <= 0 || (episode % DRAFT_BENCHMARK_EVERY) != 0) {
            return;
        }
        if (lastBenchmarkEpisode == episode) {
            return;
        }

        lastBenchmarkEpisode = episode;
        Random benchRand = new Random(episode * 1_000_003L + 17L);
        logger.info(String.format(
                "Draft benchmark tick start: ep=%d every=%d drafts=%d gamesPerDraft=%d threshold=%.3f tier=%d",
                episode, DRAFT_BENCHMARK_EVERY, DRAFT_BENCHMARK_DRAFTS,
                DRAFT_BENCHMARK_GAMES_PER_DRAFT, DRAFT_BENCHMARK_PROMOTE_WR, benchmarkTier));

        long t0 = System.currentTimeMillis();
        BenchmarkResult result = runDraftBenchmark(episode, benchRand);
        long elapsed = System.currentTimeMillis() - t0;

        boolean promotedNow = false;
        if (result.winRate >= DRAFT_BENCHMARK_PROMOTE_WR) {
            benchmarkTier++;
            promotedNow = true;
            logger.info(String.format(
                    "Draft benchmark promotion: ep=%d tier=%d wr=%.3f threshold=%.3f",
                    episode, benchmarkTier, result.winRate, DRAFT_BENCHMARK_PROMOTE_WR));
        }

        logger.info(String.format(
                "Draft benchmark tick done: ep=%d wr=%.3f wins=%d losses=%d validDrafts=%d failedDrafts=%d promoted=%s tier=%d (%dms)",
                episode, result.winRate, result.wins, result.losses, result.validDrafts, result.failedDrafts,
                promotedNow, benchmarkTier, elapsed));

        appendBenchmarkStats(episode, result, promotedNow);
        saveBenchmarkState();
    }

    private BenchmarkResult runDraftBenchmark(int episode, Random rand) {
        int wins = 0;
        int losses = 0;
        int validDrafts = 0;
        int failedDrafts = 0;

        for (int d = 0; d < DRAFT_BENCHMARK_DRAFTS; d++) {
            ComputerPlayerRL rlPlayer = new ComputerPlayerRL(
                    "DraftRL-Benchmark", RangeOfInfluence.ALL, gameModel, true, false, "train");
            rlPlayer.resetDraftState(draftBridge, null, episode, false);

            DraftRunner.DraftResult draftResult = draftRunner.runDraft(rlPlayer);
            if (draftResult == null) {
                failedDrafts++;
                continue;
            }

            Deck rlDeck = rlPlayer.constructDraftDeck();
            if (rlDeck == null || rlDeck.getCards().isEmpty()) {
                failedDrafts++;
                continue;
            }
            rlDeck = padDeckToMinimum(rlDeck, 40);

            Deck heuristicDeck = draftResult.getRandomHeuristicDeck(rand);
            if (heuristicDeck == null
                    || (heuristicDeck.getCards().isEmpty() && heuristicDeck.getSideboard().isEmpty())) {
                failedDrafts++;
                continue;
            }

            Deck playableHeuristicDeck = buildHeuristicDeckFromSideboard(heuristicDeck);
            if (playableHeuristicDeck == null || playableHeuristicDeck.getCards().isEmpty()) {
                failedDrafts++;
                continue;
            }

            validDrafts++;
            for (int g = 0; g < DRAFT_BENCHMARK_GAMES_PER_DRAFT; g++) {
                boolean rlWon = playBenchmarkGame(rlDeck, playableHeuristicDeck);
                if (rlWon) {
                    wins++;
                } else {
                    losses++;
                }
            }
        }

        int totalGames = Math.max(1, wins + losses);
        double wr = wins / (double) totalGames;
        return new BenchmarkResult(wins, losses, validDrafts, failedDrafts, wr);
    }

    private void appendBenchmarkStats(int episode, BenchmarkResult r, boolean promotedNow) {
        try {
            Path p = Paths.get(BENCHMARK_STATS_PATH);
            Files.createDirectories(p.getParent());

            boolean exists = Files.exists(p);
            String header = "episode,wins,losses,win_rate,valid_drafts,failed_drafts,promoted_now,benchmark_tier,timestamp\n";
            String row = String.format("%d,%d,%d,%.4f,%d,%d,%s,%d,%s%n",
                    episode, r.wins, r.losses, r.winRate, r.validDrafts, r.failedDrafts,
                    promotedNow, benchmarkTier, LocalDateTime.now());

            if (!exists) {
                Files.write(p, header.getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE);
            }
            Files.write(p, row.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write draft benchmark stats: " + e.getMessage());
        }
    }

    private void loadBenchmarkState() {
        Path p = Paths.get(BENCHMARK_STATE_PATH);
        if (!Files.exists(p)) {
            return;
        }
        try {
            List<String> lines = Files.readAllLines(p, StandardCharsets.UTF_8);
            for (String line : lines) {
                String trimmed = line == null ? "" : line.trim();
                if (trimmed.startsWith("benchmark_tier=")) {
                    benchmarkTier = Integer.parseInt(trimmed.substring("benchmark_tier=".length()).trim());
                } else if (trimmed.startsWith("last_benchmark_episode=")) {
                    lastBenchmarkEpisode = Integer.parseInt(trimmed.substring("last_benchmark_episode=".length()).trim());
                }
            }
        } catch (Exception e) {
            logger.warn("Could not read draft benchmark state: " + e.getMessage());
        }
    }

    private void saveBenchmarkState() {
        try {
            Path p = Paths.get(BENCHMARK_STATE_PATH);
            Files.createDirectories(p.getParent());
            String payload = "benchmark_tier=" + benchmarkTier + System.lineSeparator()
                    + "last_benchmark_episode=" + lastBenchmarkEpisode + System.lineSeparator()
                    + "updated_at=" + LocalDateTime.now() + System.lineSeparator();
            Files.write(p, payload.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Could not save draft benchmark state: " + e.getMessage());
        }
    }

    private static final class BenchmarkResult {
        final int wins;
        final int losses;
        final int validDrafts;
        final int failedDrafts;
        final double winRate;

        BenchmarkResult(int wins, int losses, int validDrafts, int failedDrafts, double winRate) {
            this.wins = wins;
            this.losses = losses;
            this.validDrafts = validDrafts;
            this.failedDrafts = failedDrafts;
            this.winRate = winRate;
        }
    }

    // -----------------------------------------------------------------------
    // Draft model training
    // -----------------------------------------------------------------------

    private void trainDraftModel(List<DraftStepData> steps, float terminalReward) {
        int n = steps.size();
        if (n == 0) return;

        int seqLen = DraftStateBuilder.MAX_LEN;
        int dim = DraftStateBuilder.DIM_PER_TOKEN;
        int maxCands = computeMaxCands(steps);

        // Allocate flat byte arrays
        float[] stateArr  = new float[n * seqLen * dim];
        int[]   maskArr   = new int[n * seqLen];
        int[]   tokArr    = new int[n * seqLen];
        float[] cfArr     = new float[n * maxCands * dim];
        int[]   cidArr    = new int[n * maxCands];
        int[]   cmaskArr  = new int[n * maxCands]; // 1=real, 0=padding (matches Java convention)
        int[]   chosenArr = new int[n];
        float[] logpArr   = new float[n];
        float[] valArr    = new float[n];
        float[] rewArr    = new float[n];
        int[]   doneArr   = new int[n];
        int[]   headArr   = new int[n];

        for (int i = 0; i < n; i++) {
            DraftStepData s = steps.get(i);

            System.arraycopy(s.stateFlat, 0, stateArr, i * seqLen * dim, seqLen * dim);
            System.arraycopy(s.maskFlat,  0, maskArr,  i * seqLen, seqLen);
            System.arraycopy(s.tokenIds,  0, tokArr,   i * seqLen, seqLen);

            int nc = s.candIds.length;
            // copy candidate features (pad with zeros to maxCands)
            System.arraycopy(s.candFeats, 0, cfArr, i * maxCands * dim, nc * dim);
            System.arraycopy(s.candIds,   0, cidArr, i * maxCands, nc);
            // candidate mask: 1=real, 0=padding
            for (int j = 0; j < nc; j++) cmaskArr[i * maxCands + j] = 1;

            chosenArr[i] = s.chosenIdx;
            logpArr[i]   = s.logProb;
            valArr[i]    = s.value;

            // Reward: 0 for all intermediate steps, terminal reward for last step
            rewArr[i] = (i == n - 1) ? terminalReward : 0.0f;
            doneArr[i] = (i == n - 1) ? 1 : 0;
            headArr[i] = s.headIdx;
        }

        byte[] stateBytes  = toBytes(stateArr);
        byte[] maskBytes   = toBytes(maskArr);
        byte[] tokBytes    = toBytes(tokArr);
        byte[] cfBytes     = toBytes(cfArr);
        byte[] cidBytes    = toBytes(cidArr);
        byte[] cmaskBytes  = toBytes(cmaskArr);
        byte[] chosenBytes = toBytes(chosenArr);
        byte[] logpBytes   = toBytes(logpArr);
        byte[] valBytes    = toBytes(valArr);
        byte[] rewBytes    = toBytes(rewArr);
        byte[] doneBytes   = toBytes(doneArr);
        byte[] headBytes   = toBytes(headArr);

        draftBridge.trainDraftBatch(
                stateBytes, maskBytes, tokBytes,
                cfBytes, cidBytes, cmaskBytes,
                chosenBytes, logpBytes, valBytes,
                rewBytes, doneBytes, headBytes,
                n, seqLen, dim, maxCands);
    }

    private int computeMaxCands(List<DraftStepData> steps) {
        int max = 1;
        for (DraftStepData s : steps) {
            if (s.candIds.length > max) max = s.candIds.length;
        }
        return max;
    }

    // -----------------------------------------------------------------------
    // Deck construction helpers
    // -----------------------------------------------------------------------

    /**
     * Build a playable 40-card deck from a heuristic drafter's sideboard
     * (all drafted cards), using the heuristic AI's own logic.
     */
    private Deck buildHeuristicDeckFromSideboard(Deck sideboardDeck) {
        List<Card> pool = new ArrayList<>(sideboardDeck.getSideboard());
        if (pool.isEmpty()) {
            pool.addAll(sideboardDeck.getCards());
        }

        // Sort by CMC and take 23 non-lands
        List<Card> nonLands = new ArrayList<>();
        List<Card> lands = new ArrayList<>();
        for (Card c : pool) {
            if (c.isLand()) { lands.add(c); }
            else { nonLands.add(c); }
        }

        nonLands.sort((a, b) -> Double.compare(a.getManaValue(), b.getManaValue()));
        List<Card> mainNonLands = nonLands.subList(0, Math.min(23, nonLands.size()));

        // Heuristic land base from drafted lands + basics
        List<Card> landBase = new ArrayList<>(lands.subList(0, Math.min(17, lands.size())));
        // If not enough lands, just use what we have
        int needed = 40 - mainNonLands.size() - landBase.size();
        // Add extra non-lands if short on lands
        if (needed > 0 && nonLands.size() > mainNonLands.size()) {
            List<Card> extras = nonLands.subList(mainNonLands.size(),
                    Math.min(nonLands.size(), mainNonLands.size() + needed));
            landBase.addAll(extras);
        }

        Deck deck = new Deck();
        for (Card c : mainNonLands) deck.getCards().add(c);
        for (Card c : landBase) deck.getCards().add(c);
        return deck;
    }

    /**
     * Pad deck to minimum size with land cards from sideboard, or just leave as-is.
     * In cube, players should always have 40 cards, but handle edge cases.
     */
    private Deck padDeckToMinimum(Deck deck, int minimum) {
        int current = deck.getCards().size();
        if (current >= minimum) return deck;

        // Try to find extra cards from sideboard or just duplicate existing lands
        List<Card> sideCards = new ArrayList<>(deck.getSideboard());
        for (Card c : sideCards) {
            if (deck.getCards().size() >= minimum) break;
            deck.getCards().add(c);
        }
        return deck;
    }

    // -----------------------------------------------------------------------
    // Episode counter persistence
    // -----------------------------------------------------------------------

    private int loadEpisodeCount() {
        try {
            Path p = Paths.get(EPISODE_COUNT_PATH);
            if (Files.exists(p)) {
                return Integer.parseInt(new String(Files.readAllBytes(p)).trim());
            }
        } catch (Exception e) {
            logger.warn("Could not read draft episode count: " + e.getMessage());
        }
        return 0;
    }

    private void saveEpisodeCount(int count) {
        try {
            Path p = Paths.get(EPISODE_COUNT_PATH);
            Files.createDirectories(p.getParent());
            Files.write(p, String.valueOf(count).getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Could not save draft episode count: " + e.getMessage());
        }
    }

    private void registerShutdownHookIfEnabled() {
        if (!DRAFT_SHUTDOWN_SAVE_HOOK || shutdownHookRegistered) {
            return;
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (shutdownSaveOnce.getAndIncrement() > 0) {
                return;
            }
            int episodeToSave = Math.max(startEpisode, currentEpisodeForPersistence.get());
            logger.info("DraftTrainer shutdown hook: persisting checkpoint at episode " + episodeToSave);
            saveCheckpoint("shutdown_hook");
        }, "DraftTrainer-ShutdownHook"));
        shutdownHookRegistered = true;
        logger.info("Registered DraftTrainer shutdown hook for model + counter persistence");
    }

    private void saveCheckpoint(String reason) {
        synchronized (persistenceLock) {
            int episodeToSave = Math.max(startEpisode, currentEpisodeForPersistence.get());
            try {
                draftBridge.saveDraftModel(DRAFT_MODEL_PATH);
            } catch (Exception e) {
                logger.warn("Checkpoint (" + reason + "): failed to save draft model: " + e.getMessage());
            }
            try {
                saveEpisodeCount(episodeToSave);
            } catch (Exception e) {
                logger.warn("Checkpoint (" + reason + "): failed to save draft episode count: " + e.getMessage());
            }
            try {
                saveBenchmarkState();
            } catch (Exception e) {
                logger.warn("Checkpoint (" + reason + "): failed to save benchmark state: " + e.getMessage());
            }
            if (DRAFT_SHUTDOWN_SAVE_GAME_MODELS) {
                saveGameModels(reason, true);
            }
        }
    }

    private void maybeSaveGameModelsPeriodic(int episode) {
        if (!DRAFT_SAVE_GAME_MODELS || DRAFT_SAVE_GAME_MODELS_EVERY <= 0) {
            return;
        }
        if (episode <= 0 || (episode % DRAFT_SAVE_GAME_MODELS_EVERY) != 0) {
            return;
        }
        saveGameModels("periodic", false);
    }

    private void saveGameModels(String reason, boolean duringShutdown) {
        try {
            gameModel.saveModel(RLLogPaths.MODEL_FILE_PATH);
        } catch (Exception e) {
            logModelSaveFailure("game model", reason, duringShutdown, e);
        }
        try {
            gameModel.saveMulliganModel();
        } catch (Exception e) {
            logModelSaveFailure("mulligan model", reason, duringShutdown, e);
        }
    }

    private void logModelSaveFailure(String modelName, String reason, boolean duringShutdown, Exception e) {
        String msg = e != null ? e.getMessage() : "unknown";
        boolean bridgeGone = msg != null && msg.toLowerCase().contains("not initialized");
        if (duringShutdown && bridgeGone) {
            logger.info("Checkpoint (" + reason + "): skipped " + modelName
                    + " save because Python bridge already shut down");
        } else {
            logger.warn("Checkpoint (" + reason + "): failed to save " + modelName + ": " + msg);
        }
    }

    private double getRollingWinRate() {
        if (recentWins.isEmpty()) return 0.0;
        long wins = recentWins.stream().filter(b -> b).count();
        return wins / (double) recentWins.size();
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    private void appendEpisodeStats(int episode, int wins, int losses,
            float reward, double rollingWr) {
        try {
            Path p = Paths.get(STATS_PATH);
            Files.createDirectories(p.getParent());

            boolean exists = Files.exists(p);
            String header = "episode,wins,losses,reward,rolling_winrate\n";
            String row = String.format("%d,%d,%d,%.4f,%.4f%n",
                    episode, wins, losses, reward, rollingWr);

            if (!exists) {
                Files.write(p, header.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE);
            }
            Files.write(p, row.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write draft stats: " + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Byte conversion helpers
    // -----------------------------------------------------------------------

    private static byte[] toBytes(float[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : arr) buf.putFloat(v);
        return buf.array();
    }

    private static byte[] toBytes(int[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : arr) buf.putInt(v);
        return buf.array();
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        BasicConfigurator.configure();
        Level logLevel = parseLogLevel(System.getenv("MTG_AI_LOG_LEVEL"), Level.WARN);
        LogManager.getRootLogger().setLevel(logLevel);
        Logger.getLogger("mage.game").setLevel(Level.ERROR);
        Logger.getLogger("mage.server").setLevel(Level.ERROR);
        Logger.getLogger("mage.player.ai.ComputerPlayer6").setLevel(Level.ERROR);

        logger.info("Initializing DraftTrainer...");
        DraftTrainer trainer = new DraftTrainer();
        trainer.train();
    }

    private static Level parseLogLevel(String s, Level def) {
        if (s == null) return def;
        switch (s.trim().toUpperCase()) {
            case "DEBUG": return Level.DEBUG;
            case "INFO":  return Level.INFO;
            case "WARN":  return Level.WARN;
            case "ERROR": return Level.ERROR;
            case "OFF":   return Level.OFF;
            default:      return def;
        }
    }
}
