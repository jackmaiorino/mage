package mage.player.ai.rl;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.log4j.Logger;

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

/**
 * Multi-threaded self-play generator. Runs on CPU nodes, does NOT update model.
 * After each episode serialises trajectory + returns to gzip file in TRAJ_DIR.
 * Now supports parallel game execution for better throughput.
 */
public class GameWorker {

    private static final Logger logger = Logger.getLogger(GameWorker.class);

    // Multi-threading configuration (borrowed from RLTrainer)
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

    private static final int DEFAULT_GAME_RUNNERS = Math.max(1, Runtime.getRuntime().availableProcessors());
    public static final int NUM_GAME_RUNNERS = envInt("NUM_GAME_RUNNERS", DEFAULT_GAME_RUNNERS);

    // Global episode counter for multi-threaded coordination
    private final AtomicInteger episodeCounter = new AtomicInteger(0);

    // ThreadLocal logger for multi-threaded logging
    public static final ThreadLocal<Logger> threadLocalLogger = ThreadLocal.withInitial(() -> {
        Logger threadLogger = Logger.getLogger("GameWorker-Thread-" + Thread.currentThread().getId());
        threadLogger.setLevel(org.apache.log4j.Level.INFO);
        return threadLogger;
    });

    private final int episodesPerRun;
    private final Path trajDir;

    public GameWorker(int episodesPerRun, Path trajDir) {
        this.episodesPerRun = episodesPerRun;
        this.trajDir = trajDir;
    }

    public void run() throws IOException, GameException {
        // This is a critical step. Without it, the card repository is empty and deck loading will fail.
        mage.cards.repository.CardScanner.scan();

        Files.createDirectories(trajDir);

        // Log configuration info
        int cpuCores = Runtime.getRuntime().availableProcessors();
        long maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);

        logger.info("========================");
        logger.info("Multi-threaded GameWorker Configuration:");
        logger.info("CPU Cores Available: " + cpuCores);
        logger.info("Max JVM Memory: " + maxMemory + " MB");
        logger.info("Game Runners: " + NUM_GAME_RUNNERS + " (using " + (NUM_GAME_RUNNERS * 100.0 / cpuCores) + "% of CPU cores)");
        logger.info("Episodes per run: " + episodesPerRun);
        logger.info("========================");

        // Create thread pool for parallel game execution
        ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS, runnable -> {
            Thread thread = new Thread(runnable);
            thread.setPriority(Thread.MAX_PRIORITY - 1);
            return thread;
        });

        List<Future<Void>> futures = new ArrayList<>();
        final boolean[] isFirstThread = {true}; // Flag to track the first thread
        final Object lock = new Object();

        // Start game runner threads
        for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
            Future<Void> future = executor.submit(() -> {
                boolean isFirst;
                synchronized (lock) {
                    isFirst = isFirstThread[0];
                    isFirstThread[0] = false;
                }

                Logger currentLogger = threadLocalLogger.get();
                currentLogger.info("Starting Multi-threaded Game Runner");

                Thread.currentThread().setName("GAME-WORKER-" + Thread.currentThread().getId());

                while (episodeCounter.get() < episodesPerRun) {
                    int epNumber = episodeCounter.incrementAndGet();
                    if (epNumber > episodesPerRun) {
                        break; // Another thread reached the target
                    }

                    try {
                        runSingleEpisode(epNumber - 1, currentLogger); // Convert to 0-based index
                    } catch (Exception e) {
                        currentLogger.error("Error in episode " + (epNumber - 1), e);
                    }
                }

                currentLogger.info("Game runner thread completed");
                return null;
            });
            futures.add(future);
        }

        // Wait for all threads to complete
        try {
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (Exception e) {
            logger.error("Error waiting for game runners", e);
        } finally {
            executor.shutdown();
        }

        logger.info("All " + episodesPerRun + " episodes completed across " + NUM_GAME_RUNNERS + " threads");
    }

    /**
     * Run a single episode (extracted for multi-threading)
     */
    private void runSingleEpisode(int episodeIndex, Logger currentLogger) throws IOException, GameException {
        // Build match & game skeleton
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));
        match.startGame();
        Game game = match.getGames().get(0);

        // Simple random decks from trainer util
        Deck deck1 = loadRandomDeck();
        Deck deck2 = loadRandomDeck();

        Random rnd = new Random();
        ComputerPlayerRL p1 = new ComputerPlayerRL("RL1", RangeOfInfluence.ALL, RLTrainer.getSharedModel());
        Player opp;
        if (rnd.nextBoolean()) {
            opp = new ComputerPlayerRL("RL2", RangeOfInfluence.ALL, RLTrainer.getSharedModel());
        } else {
            opp = new ComputerPlayer7("Heuristic", RangeOfInfluence.ALL, 3);
        }

        game.addPlayer(p1, deck1);
        match.addPlayer(p1, deck1);
        game.addPlayer(opp, deck2);
        match.addPlayer(opp, deck2);

        game.loadCards(deck1.getCards(), p1.getId());
        game.loadCards(deck2.getCards(), opp.getId());
        game.setGameOptions(new GameOptions());
        game.start(p1.getId()); // blocking until game end

        // Collect training data from p1 (and maybe opp if RL)
        double finalReward = game.getWinner().contains(p1.getName()) ? 1.0 : -1.0;
        List<StateSequenceBuilder.TrainingData> traj = p1.getTrainingBuffer();
        List<Double> returns = RLTrainer.calculateDiscountedReturns(traj, finalReward, 0.99);
        EpisodeData epData = new EpisodeData(traj, returns, finalReward);

        // Serialize gzip with thread-safe filename
        Path out = trajDir.resolve("ep-" + UUID.randomUUID() + ".ser.gz");
        synchronized (this) { // Ensure file writes don't conflict
            try (ObjectOutputStream oos = new ObjectOutputStream(new java.util.zip.GZIPOutputStream(new FileOutputStream(out.toFile())))) {
                oos.writeObject(epData);
            }
        }
        currentLogger.info("Saved episode " + episodeIndex + " to " + out);

        // Record episode completion metrics
        RLTrainer.metrics.recordEpisodeCompleted();
    }

    private Deck loadRandomDeck() throws IOException {
        List<Path> files = RLTrainer.loadDeckPool();
        if (files.isEmpty()) {
            throw new IOException("No deck files in deck pool");
        }
        Path pick = files.get(new Random().nextInt(files.size()));
        return new RLTrainer().loadDeck(pick.toString());
    }
}
