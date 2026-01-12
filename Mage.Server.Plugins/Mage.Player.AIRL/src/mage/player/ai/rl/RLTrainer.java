package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameException;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;

public class RLTrainer {

    private static final Logger logger = Logger.getLogger(RLTrainer.class);

    /* ================================================================
     *  Configurable parameters (env-vars override defaults)
     * ============================================================ */
    private static String env(String key, String def) {
        return System.getenv().getOrDefault(key, def);
    }

    private static int envInt(String key, int def) {
        try {
            return Integer.parseInt(env(key, String.valueOf(def)));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    // Local training command:
    // $env:TOTAL_EPISODES='1'; $env:DECKS_DIR='src/mage/player/ai/decks/Pauper'; mvn -q compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
    private static final int NUM_EPISODES = envInt("TOTAL_EPISODES", 10000);
    private static final int NUM_EVAL_EPISODES = envInt("EVAL_EPISODES", 5);

    public static final String DECKS_DIRECTORY = env("DECKS_DIR", "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper");

    public static final String MODEL_FILE_PATH = env("MODEL_PATH", "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt");
    // Episode-level statistics will be appended here (CSV)
    public static final String STATS_FILE_PATH = env("STATS_PATH", "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/training_stats.csv");
    // Path that stores the cumulative number of episodes trained so far (persisted across runs)
    public static final String EPISODE_COUNT_PATH = env("EPISODE_COUNTER_PATH", "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/episodes.txt");
    // Auto-detect optimal number of threads based on CPU cores
    private static final int DEFAULT_GAME_RUNNERS = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    public static final int NUM_THREADS = envInt("NUM_THREADS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_GAME_RUNNERS = envInt("NUM_GAME_RUNNERS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_EPISODES_PER_GAME_RUNNER = envInt("EPISODES_PER_WORKER", 500);
    public static final int EVAL_EVERY = envInt("EVAL_EVERY", 100);

    public static final PythonMLBridge sharedModel = PythonMLBridge.getInstance();
    public static final MetricsCollector metrics = MetricsCollector.getInstance();

    // Global episode counter to track total episodes across all threads
    private static final AtomicInteger EPISODE_COUNTER = new AtomicInteger(0);

    static {
        // Set default logging level for all loggers to WARN
        List<Logger> loggers = Collections.<Logger>list(LogManager.getCurrentLoggers());
        loggers.add(LogManager.getRootLogger());
        for (Logger logger : loggers) {
            logger.setLevel(Level.INFO);
        }
    }

    // ThreadLocal logger
    public static final ThreadLocal<Logger> threadLocalLogger = ThreadLocal.withInitial(() -> {
        Logger threadLogger = Logger.getLogger("Thread-" + Thread.currentThread().getId());
        threadLogger.setLevel(Level.INFO); // Default level
        return threadLogger;
    });

    public RLTrainer() {
        // No need to initialize here as PythonMLBridge handles initialization in its constructor
    }

    /* ================================================================
     *  Simple CLI entry point: java RLTrainer train|eval
     * ============================================================ */
    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : env("MODE", "train");

        // Start metrics collection server
        int metricsPort = envInt("METRICS_PORT", 9090);
        metrics.startMetricsServer(metricsPort);
        logger.info("Metrics server started on port " + metricsPort);
        if ("learner".equalsIgnoreCase(mode)) {
            try {
                Path dir = Paths.get(System.getenv().getOrDefault("TRAJ_DIR", "./trajectories"));
                int maxSamples = envInt("MAX_SAMPLES_PER_BATCH", 50000); // ~500MB for RTX 4070
                int poll = envInt("POLL_SECONDS", 60);
                int ckpt = envInt("CHECKPOINT_EVERY", 100);
                new Learner(dir, maxSamples, poll, ckpt).run();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else if ("worker".equalsIgnoreCase(mode)) {
            try {
                int eps = envInt("EPISODES_PER_WORKER", 100);
                Path dir = Paths.get(System.getenv().getOrDefault("TRAJ_DIR", "./trajectories"));
                new GameWorker(eps, dir).run();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else if ("eval".equalsIgnoreCase(mode)) {
            runEvaluation(NUM_EVAL_EPISODES);
        } else {
            new RLTrainer().train();
        }
    }

    public void train() {
        System.out.println("DEBUG: Starting train() method");
        System.out.println("DEBUG: DECKS_DIRECTORY = " + DECKS_DIRECTORY);
        System.out.println("DEBUG: Working directory = " + System.getProperty("user.dir"));

        // Initialize the card database - this is crucial for deck loading to work
        System.out.println("DEBUG: Initializing card database...");
        mage.cards.repository.CardScanner.scan();
        System.out.println("DEBUG: Card database initialized");

        try {
            List<Path> deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
            System.out.println("DEBUG: Found " + deckFiles.size() + " deck files in " + DECKS_DIRECTORY);

            Random random = new Random();

            // ------------------ 1.  Load persisted episode count ------------------
            int initialEpisodeCount = 0;
            try {
                Path epPath = Paths.get(EPISODE_COUNT_PATH);
                if (Files.exists(epPath)) {
                    String content = new String(Files.readAllBytes(epPath), StandardCharsets.UTF_8).trim();
                    int persisted = Integer.parseInt(content);
                    EPISODE_COUNTER.set(persisted);
                    initialEpisodeCount = persisted;
                    logger.info("Loaded episode counter from file: " + persisted);
                }
            } catch (Exception e) {
                logger.warn("Failed to read episode counter, starting from 0", e);
            }

            // Log system resource utilization
            int cpuCores = Runtime.getRuntime().availableProcessors();
            long maxMemory = Runtime.getRuntime().maxMemory() / 1024 / 1024; // MB
            RLTrainer.threadLocalLogger.get().info("=== SYSTEM RESOURCES ===");
            RLTrainer.threadLocalLogger.get().info("CPU Cores Available: " + cpuCores);
            RLTrainer.threadLocalLogger.get().info("Max JVM Memory: " + maxMemory + " MB");
            RLTrainer.threadLocalLogger.get().info("Game Runners: " + NUM_GAME_RUNNERS + " (using " + (NUM_GAME_RUNNERS * 100.0 / cpuCores) + "% of CPU cores)");
            RLTrainer.threadLocalLogger.get().info("Episodes per runner: " + NUM_EPISODES_PER_GAME_RUNNER);
            RLTrainer.threadLocalLogger.get().info("Total episodes target: " + NUM_EPISODES);
            RLTrainer.threadLocalLogger.get().info("========================");

            // Record start time
            long startTime = System.nanoTime();

            ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS, runnable -> {
                Thread thread = new Thread(runnable);
                thread.setPriority(Thread.MAX_PRIORITY - 1);
                return thread;
            });

            // Load a random deck for the RL player
            Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
            Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
            Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
            Deck opponentDeck = loadDeck(opponentDeckPath.toString());
            logger.info("Decks loaded. RL player deck size: " + rlPlayerDeck.getCards().size() + ", Opponent deck size: " + opponentDeck.getCards().size());
            System.out.println("DEBUG: Decks loaded. RL player deck size: " + rlPlayerDeck.getCards().size() + ", Opponent deck size: " + opponentDeck.getCards().size());

            List<Future<Void>> futures = new ArrayList<>();
            final Object lock = new Object(); // Lock object for synchronization
            final boolean[] isFirstThread = {true}; // Flag to track the first thread

            for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
                Future<Void> future = executor.submit(() -> {
                    boolean isFirst;
                    synchronized (lock) {
                        isFirst = isFirstThread[0];
                        isFirstThread[0] = false;
                    }

                    Logger currentLogger = threadLocalLogger.get();
                    // All threads will now log at INFO level by default
                    currentLogger.info("Starting Game Runner");

                    Thread.currentThread().setName("GAME");
                    Deck rlPlayerDeckThread = rlPlayerDeck.copy();
                    Deck opponentDeckThread = opponentDeck.copy();

                    while (EPISODE_COUNTER.get() < NUM_EPISODES) {
                        int epNumber = EPISODE_COUNTER.incrementAndGet();
                        if (epNumber > NUM_EPISODES) {
                            break; // Another thread reached the target
                        }
                        long episodeStartNanos = System.nanoTime();
                        // ------------------ Build a full Match so players have MatchPlayer objects ------------------
                        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false, 2));

                        // Start an empty game so we can attach players
                        match.startGame();
                        Game game = match.getGames().get(0);

                        Random threadRand = new Random();

                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel);
                        game.addPlayer(rlPlayer, rlPlayerDeckThread);
                        match.addPlayer(rlPlayer, rlPlayerDeckThread);

                        // Decide opponent type based on global episode schedule
                        boolean vsHeuristic = shouldUseHeuristicOpponent(EPISODE_COUNTER.get(), threadRand);
                        Player opponentPlayer;
                        if (vsHeuristic) {
                            opponentPlayer = new ComputerPlayer7("HeuristicBot", RangeOfInfluence.ALL, 3);
                        } else {
                            opponentPlayer = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, sharedModel);
                        }
                        game.addPlayer(opponentPlayer, opponentDeckThread);
                        match.addPlayer(opponentPlayer, opponentDeckThread);

                        logger.info("Players added to game. RL player library size: " + rlPlayer.getLibrary().size() + ", Opponent library size: " + opponentPlayer.getLibrary().size());
                        System.out.println("DEBUG: Players added to game. RL player library size: " + rlPlayer.getLibrary().size() + ", Opponent library size: " + opponentPlayer.getLibrary().size());

                        game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeckThread.getCards(), opponentPlayer.getId());

                        GameOptions options = new GameOptions();
                        game.setGameOptions(options);

                        // Restart game now that players are added
                        game.start(rlPlayer.getId());

                        logGameResult(game, rlPlayer);
                        double finalReward = updateModelBasedOnOutcome(game, rlPlayer, opponentPlayer);

                        // Defer statistics writing until after we compute episodeSeconds below
                        // -------- Episode duration & counter logging --------
                        long episodeDurationNanos = System.nanoTime() - episodeStartNanos;
                        double episodeSeconds = episodeDurationNanos / 1_000_000_000.0;
                        RLTrainer.threadLocalLogger.get().info(String.format("Episode %d completed in %.2f seconds", epNumber, episodeSeconds));

                        // ------------------ Statistics ------------------
                        List<StateSequenceBuilder.TrainingData> td = rlPlayer.getTrainingBuffer();
                        long passCnt = td.stream().filter(t -> t.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL && !t.actionCombo.isEmpty() && t.actionCombo.get(0) == 0).count();
                        double passRate = td.isEmpty() ? 0.0 : (double) passCnt / td.size();
                        int turns = game.getTurnNum();
                        boolean usedHeuristic = opponentPlayer instanceof ComputerPlayer7;

                        try {
                            Path statsPath = Paths.get(STATS_FILE_PATH);
                            boolean writeHeader = !Files.exists(statsPath);
                            StringBuilder sb = new StringBuilder();
                            if (writeHeader) {
                                sb.append("episode,turns,pass_rate,final_reward,vs_heuristic,episode_seconds\n");
                            }
                            sb.append(epNumber).append(',').append(turns).append(',')
                                    .append(String.format("%.4f", passRate)).append(',')
                                    .append(String.format("%.3f", finalReward)).append(',')
                                    .append(usedHeuristic).append(',')
                                    .append(String.format("%.2f", episodeSeconds)).append('\n');
                            Files.write(statsPath, sb.toString().getBytes(StandardCharsets.UTF_8), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                        } catch (IOException e) {
                            logger.warn("Failed to write stats CSV", e);
                        }
                    }
                    return null;
                });
                futures.add(future);
            }

            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

            for (Future<Void> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    logger.error("Error in thread execution", e.getCause());
                    throw new RuntimeException(e.getCause());
                }
            }

            // Record end time and log statistics
            long endTime = System.nanoTime();
            long totalTime = endTime - startTime;
            double totalTimeInMinutes = totalTime / 1_000_000_000.0 / 60.0;
            int episodesRun = EPISODE_COUNTER.get() - initialEpisodeCount;
            double gamesRunPerMinute = totalTimeInMinutes > 0 ? episodesRun / totalTimeInMinutes : 0;

            logger.info("Training completed:");
            logger.info("Total Games Run: " + episodesRun);
            logger.info("Games Run Per Minute: " + gamesRunPerMinute);
            logger.info("Total Training Time: " + (totalTime / 1_000_000_000.0) + " seconds");

            // Save the trained model
            sharedModel.saveModel(MODEL_FILE_PATH);
            sharedModel.shutdown();
            // ------------------ 2.  Persist updated episode counter ------------------
            try {
                Files.write(Paths.get(EPISODE_COUNT_PATH), String.valueOf(EPISODE_COUNTER.get()).getBytes(StandardCharsets.UTF_8));
            } catch (IOException e) {
                logger.error("Failed to persist episode counter", e);
            }

        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
    }

    public void eval(int numEpisodesPerThread) {
        runEvaluation(numEpisodesPerThread);
    }

    public static double runEvaluation(int numEpisodesPerThread) {
        List<Path> deckFiles = null;
        try {
            deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
        } catch (IOException e) {
            logger.error("Error during evaluation", e);
            return 0.0;
        }

        ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS);
        final Object lock = new Object();
        final boolean[] isFirstThread = {true};

        List<Future<Integer>> futures = new ArrayList<>();

        Random random = new Random();
        Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
        Deck rlPlayerDeck = new RLTrainer().loadDeck(rlPlayerDeckPath.toString());
        Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
        Deck opponentDeck = new RLTrainer().loadDeck(opponentDeckPath.toString());

        for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
            Future<Integer> future = executor.submit(() -> {
                boolean isFirst;
                synchronized (lock) {
                    isFirst = isFirstThread[0];
                    isFirstThread[0] = false;
                }

                Logger currentLogger = threadLocalLogger.get();
                if (isFirst) {
                    currentLogger.setLevel(Level.INFO);
                }

                int localWinsAgainstComputerPlayer7 = 0;

                Thread.currentThread().setName("GAME");
                Deck rlPlayerDeckThread = rlPlayerDeck.copy();
                Deck opponentDeckThread = opponentDeck.copy();

                for (int evalEpisode = 0; evalEpisode < numEpisodesPerThread; evalEpisode++) {
                    TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false, 2));
                    try {
                        match.startGame();
                    } catch (GameException e) {
                        logger.error("Error starting game", e);
                        continue;
                    }
                    Game game = match.getGames().get(0);

                    // Greedy evaluation: use deterministic arg-max player
                    ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel, true);
                    game.addPlayer(rlPlayer, rlPlayerDeckThread);
                    match.addPlayer(rlPlayer, rlPlayerDeckThread);

                    ComputerPlayer7 opponent = new ComputerPlayer7("Player7", RangeOfInfluence.ALL, 3);
                    game.addPlayer(opponent, opponentDeckThread);
                    match.addPlayer(opponent, opponentDeckThread);

                    game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                    game.loadCards(opponentDeckThread.getCards(), opponent.getId());

                    GameOptions options = new GameOptions();
                    game.setGameOptions(options);

                    game.start(rlPlayer.getId());

                    if (isFirst) {
                        logStaticGameResult(game, rlPlayer);
                    }

                    if (game.getWinner().contains(rlPlayer.getName())) {
                        localWinsAgainstComputerPlayer7++;
                    }
                }
                return localWinsAgainstComputerPlayer7;
            });
            futures.add(future);
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            logger.error("Evaluation interrupted", e);
            Thread.currentThread().interrupt();
        }

        int totalWins = 0;
        for (Future<Integer> future : futures) {
            try {
                totalWins += future.get();
            } catch (InterruptedException | ExecutionException e) {
                logger.error("Error getting evaluation results", e);
            }
        }
        double winRate = (double) totalWins / (numEpisodesPerThread * NUM_GAME_RUNNERS);
        logger.info("Evaluation win rate: " + winRate);
        return winRate;
    }

    private static void logEvaluationResult(int updateStep, double winRate) {
        try {
            Path statsPath = Paths.get(STATS_FILE_PATH.replace("training_stats.csv", "evaluation_stats.csv"));
            boolean writeHeader = !Files.exists(statsPath);
            StringBuilder sb = new StringBuilder();
            if (writeHeader) {
                sb.append("update_step,win_rate\n");
            }
            sb.append(updateStep).append(',').append(String.format("%.4f", winRate)).append('\n');
            Files.write(statsPath, sb.toString().getBytes(StandardCharsets.UTF_8), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write evaluation stats CSV", e);
        }
    }

    private void logGameResult(Game game, ComputerPlayerRL rlPlayer) {
        logStaticGameResult(game, rlPlayer);
    }

    private static void logStaticGameResult(Game game, ComputerPlayerRL rlPlayer) {
        if (game.getWinner().contains(rlPlayer.getName())) {
            logger.info("Game finished. Winner: " + rlPlayer.getName());
        } else {
            logger.info("Game finished. Loser: " + rlPlayer.getName());
        }
    }

    public Deck loadDeck(String filePath) {
        try {
            DeckCardLists deckCardLists = DeckImporter.importDeckFromFile(filePath, false);
            return Deck.load(deckCardLists, false, false, null);
        } catch (GameException e) {
            logger.error("Error loading deck: " + filePath, e);
            return null;
        }
    }

    private double updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, Player opponentPlayer) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());

        // ------------------------------------------------------------------
        // 1.  Terminal win / loss reward (ground-truth)
        // ------------------------------------------------------------------
        double finalReward = rlPlayerWon ? 1.0 : -1.0;

        // ------------------------------------------------------------------
        // 2.  Potential-based shaping term  ψ = α (Φ(s′) – Φ(s))
        // ------------------------------------------------------------------
        final double ALPHA = 0.05;
        int rlLife = Math.max(0, rlPlayer.getLife());
        UUID oppId = game.getOpponents(rlPlayer.getId()).iterator().next();
        Player opp = game.getPlayer(oppId);
        int oppLife = Math.max(0, (opp != null ? opp.getLife() : 20));
        double phiFinal = Math.signum(Math.log(rlLife + 1.0) - Math.log(oppLife + 1.0));
        finalReward += ALPHA * phiFinal;

        if (Double.isNaN(finalReward) || Double.isInfinite(finalReward)) {
            finalReward = rlPlayerWon ? 1.0 : -1.0;
            logger.warn("Reward was NaN/Inf – reverted to ±1 fallback");
        }

        // Get training data for RL player
        List<StateSequenceBuilder.TrainingData> rlPlayerTrainingData = rlPlayer.getTrainingBuffer();

        List<StateSequenceBuilder.TrainingData> opponentTrainingData = new ArrayList<>();
        if (opponentPlayer instanceof ComputerPlayerRL) {
            opponentTrainingData = ((ComputerPlayerRL) opponentPlayer).getTrainingBuffer();
        }

        // --- Calculate discounted returns for each player ---
        final double GAMMA = 0.99; // Discount factor for future rewards

        List<Double> rlPlayerReturns = calculateDiscountedReturns(rlPlayerTrainingData, finalReward, GAMMA);
        List<Double> opponentReturns = calculateDiscountedReturns(opponentTrainingData, -finalReward, GAMMA); // Opposite reward for opponent

        // Update the model with all states and rewards
        if (!rlPlayerTrainingData.isEmpty()) {
            sharedModel.train(rlPlayerTrainingData, rlPlayerReturns);
        }
        if (!opponentTrainingData.isEmpty()) {
            sharedModel.train(opponentTrainingData, opponentReturns); // Opposite reward for opponent
        }
        return finalReward;
    }

    public static List<Double> calculateDiscountedReturns(List<StateSequenceBuilder.TrainingData> trajectory, double finalReward, double gamma) {
        if (trajectory.isEmpty()) {
            return new ArrayList<>();
        }

        List<Double> discountedReturns = new ArrayList<>(Collections.nCopies(trajectory.size(), 0.0));
        double cumulativeReturn = 0.0;

        // The reward is only applied at the terminal state.
        // We iterate backwards from the end of the game.
        for (int i = trajectory.size() - 1; i >= 0; i--) {
            // The immediate reward for the last step is the final game outcome.
            // For all other steps, the immediate reward is 0.
            double immediateReward = (i == trajectory.size() - 1) ? finalReward : 0.0;

            // By adding a small penalty for each step, we incentivize shorter games.
            final double STEP_PENALTY = -0.01;
            cumulativeReturn = (immediateReward + STEP_PENALTY) + gamma * cumulativeReturn;
            discountedReturns.set(i, cumulativeReturn);
        }

        // Normalize returns for stability (optional but good practice)
        double mean = discountedReturns.stream().mapToDouble(d -> d).average().orElse(0.0);
        double std = Math.sqrt(discountedReturns.stream().mapToDouble(d -> Math.pow(d - mean, 2)).average().orElse(0.0));
        if (std > 1e-6) { // Avoid division by zero
            for (int i = 0; i < discountedReturns.size(); i++) {
                discountedReturns.set(i, (discountedReturns.get(i) - mean) / std);
            }
        }

        return discountedReturns;
    }

    public static PythonMLBridge getSharedModel() {
        return sharedModel;
    }

    /**
     * Determines whether the current episode should pit the learning agent
     * against the heuristic ComputerPlayer7. The schedule is: – Episodes < 5000
     * → always heuristic – 5000 ≤ ep < 15000 → 50% chance heuristic – ≥ 15000 →
     * pure self-play
     */
    private boolean shouldUseHeuristicOpponent(int episodeIdx, Random rand) {
        if (episodeIdx < 5000) {
            return true;
        } else if (episodeIdx < 15000) {
            return rand.nextDouble() < 0.5;
        } else {
            return false;
        }
    }
}
