package mage.player.ai.rl;

import java.io.IOException;
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
import java.util.stream.Collectors;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameException;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.game.mulligan.LondonMulligan;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;

public class RLTrainer {

    private static final Logger logger = Logger.getLogger(RLTrainer.class);
    private static final int NUM_EPISODES = 10000;
    private static final int NUM_EVAL_EPISODES = 5;
    private static final String DECKS_DIRECTORY = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper";
    public static final String MODEL_FILE_PATH = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt";
    public static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    public static final int NUM_GAME_RUNNERS = 2;
    public static final int NUM_EPISODES_PER_GAME_RUNNER = 1;

    public static final PythonMLBridge sharedModel = new PythonMLBridge();

    static {
        // Set default logging level for all loggers to WARN
        List<Logger> loggers = Collections.<Logger>list(LogManager.getCurrentLoggers());
        loggers.add(LogManager.getRootLogger());
        for (Logger logger : loggers) {
            logger.setLevel(Level.WARN);
        }
    }

    // ThreadLocal logger
    public static final ThreadLocal<Logger> threadLocalLogger = ThreadLocal.withInitial(() -> {
        Logger threadLogger = Logger.getLogger("Thread-" + Thread.currentThread().getId());
        threadLogger.setLevel(Level.WARN); // Default level
        return threadLogger;
    });

    public RLTrainer() {
        // Initialize the Python ML model
        sharedModel.initialize();
    }

    public void train() {
        try {
            List<Path> deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());

            Random random = new Random();

            RLTrainer.threadLocalLogger.get().info("Number of threads: " + NUM_THREADS);
            RLTrainer.threadLocalLogger.get().info("Episodes per game runner: " + NUM_EPISODES_PER_GAME_RUNNER);

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
                    if (isFirst) {
                        currentLogger.setLevel(Level.INFO);
                    }
                    currentLogger.info("Starting Game Runner");

                    Thread.currentThread().setName("GAME");
                    Deck rlPlayerDeckThread = rlPlayerDeck.copy();
                    Deck opponentDeckThread = opponentDeck.copy();

                    for (int episode = 0; episode < NUM_EPISODES_PER_GAME_RUNNER; episode++) {
                        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(0), 60, 20, 7);

                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel);
                        game.addPlayer(rlPlayer, rlPlayerDeckThread);

                        ComputerPlayerRL opponent = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, sharedModel);
                        game.addPlayer(opponent, opponentDeckThread);

                        game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeckThread.getCards(), opponent.getId());

                        GameOptions options = new GameOptions();
                        game.setGameOptions(options);

                        game.start(rlPlayer.getId());

                        logGameResult(game, rlPlayer);
                        updateModelBasedOnOutcome(game, rlPlayer, opponent);
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
            double gamesRunPerMinute = NUM_GAME_RUNNERS * NUM_EPISODES_PER_GAME_RUNNER / totalTimeInMinutes;

            logger.info("Training completed:");
            logger.info("Total Games Run: " + NUM_GAME_RUNNERS * NUM_EPISODES_PER_GAME_RUNNER);
            logger.info("Games Run Per Minute: " + gamesRunPerMinute);
            logger.info("Total Training Time: " + (totalTime / 1_000_000_000.0) + " seconds");

            // Save the trained model
            sharedModel.saveModel(MODEL_FILE_PATH);

        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
    }

    public void eval(int numEpisodesPerThread) {
        List<Path> deckFiles = null;
        try {
            deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
        } catch (IOException e) {
            logger.error("Error during evaluation", e);
            return;
        }

        ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS);
        final Object lock = new Object();
        final boolean[] isFirstThread = {true};

        List<Future<Integer>> futures = new ArrayList<>();

        Random random = new Random();
        Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
        Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
        Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
        Deck opponentDeck = loadDeck(opponentDeckPath.toString());

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

                    ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel);
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
                        logGameResult(game, rlPlayer);
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

        int totalWinsAgainstComputerPlayer7 = futures.stream().mapToInt(future -> {
            try {
                return future.get();
            } catch (InterruptedException | ExecutionException e) {
                logger.error("Error in thread execution", e.getCause());
                return 0;
            }
        }).sum();

        double winRate = (double) totalWinsAgainstComputerPlayer7 / (numEpisodesPerThread * NUM_THREADS);
        logger.setLevel(Level.INFO);
        logger.info("Win rate against ComputerPlayer7: " + (winRate * 100) + "%");
    }

    private void logGameResult(Game game, ComputerPlayerRL rlPlayer) {
        logger.info("Game finished - Winner: " + game.getWinner());
        logger.info("Final life totals:");
        logger.info("[" + rlPlayer.getName() + "]: " + rlPlayer.getLife());

        UUID opponentId = game.getOpponents(rlPlayer.getId()).iterator().next();
        Player opponent = game.getPlayer(opponentId);
        logger.info("[" + opponent.getName() + "]: " + opponent.getLife());
    }

    private Deck loadDeck(String filePath) {
        try {
            DeckCardLists deckCardLists = DeckImporter.importDeckFromFile(filePath, false);
            return Deck.load(deckCardLists, false, false, null);
        } catch (GameException e) {
            logger.error("Error loading deck: " + filePath, e);
            return null;
        }
    }

    private void updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, ComputerPlayerRL opponent) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
        double reward = rlPlayerWon ? 0.15 : -0.15;

        // Get training data for both players
        List<StateSequenceBuilder.TrainingData> rlPlayerTrainingData = rlPlayer.getTrainingBuffer();
        List<StateSequenceBuilder.TrainingData> opponentTrainingData = opponent.getTrainingBuffer();

        // Update the model with all states and rewards
        // Technically these can be empty (if they keep a no lander. Might be worth logging though)
        if (!rlPlayerTrainingData.isEmpty()) {
            sharedModel.train(rlPlayerTrainingData, reward);
        }
        if (!opponentTrainingData.isEmpty()) {
            sharedModel.train(opponentTrainingData, -reward); // Opposite reward for opponent
        }
    }

    public static PythonMLBridge getSharedModel() {
        return sharedModel;
    }
}
