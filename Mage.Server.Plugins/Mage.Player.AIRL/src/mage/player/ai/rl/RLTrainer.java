package mage.player.ai.rl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

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
    private static final String DECKS_DIRECTORY = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks";
    public static final String MODEL_FILE_PATH = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/network.ser";
    public static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    public static final int NUM_GAME_RUNNERS = NUM_THREADS * 100;
    // This is a CPU/Bound value. If we can speed up CPU processing, we can increase this value
    // It is also technically a GPU bound value but cpu processing is the bottleneck
    public static final int BATCH_SIZE = (int) (NUM_GAME_RUNNERS/2);

    public static final int NUM_EPISODES_PER_GAME_RUNNER = NUM_EPISODES / NUM_GAME_RUNNERS;

    public static final RLModel sharedModel = new RLModel();

    public RLTrainer() {
        // No need to create a model here
    }

    public void train() {
        try {
            List<Path> deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                                        .filter(Files::isRegularFile)
                                        .collect(Collectors.toList());

            Random random = new Random();
            
            // Create singleton instance
            BatchPredictionRequest batchPredictionRequest = BatchPredictionRequest.getInstance(0, 10000, TimeUnit.MILLISECONDS);

            logger.info("Number of threads: " + NUM_THREADS);
            logger.info("Episodes per game runner: " + NUM_EPISODES_PER_GAME_RUNNER);

            // Record start time
            long startTime = System.nanoTime();

            ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS, runnable -> {
                Thread thread = new Thread(runnable);
                // One less prio than BatchPredictManager
                thread.setPriority(Thread.MAX_PRIORITY - 1);
                return thread;
            });

            // Load a random deck for the RL player
            Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
            Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
            Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
            Deck opponentDeck = loadDeck(opponentDeckPath.toString());

            List<Future<Void>> futures = new ArrayList<>();
            for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
                Future<Void> future = executor.submit(() -> {

                    Thread.currentThread().setName("GAME");
                    Deck rlPlayerDeckThread = rlPlayerDeck.copy();
                    Deck opponentDeckThread = opponentDeck.copy();

                    for (int episode = 0; episode < NUM_EPISODES_PER_GAME_RUNNER; episode++) {
                        batchPredictionRequest.incrementBatchSize();
                        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(7), 60, 20, 7);

                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 10, sharedModel);
                        game.addPlayer(rlPlayer, rlPlayerDeckThread);

                        ComputerPlayerRL opponent = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, 10, sharedModel);
                        game.addPlayer(opponent, opponentDeckThread);

                        game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeckThread.getCards(), opponent.getId());

                        GameOptions options = new GameOptions();
                        game.setGameOptions(options);

                        game.start(rlPlayer.getId());

                        logGameResult(game, rlPlayer);
                        batchPredictionRequest.decrementBatchSize();
                        updateModelBasedOnOutcome(game, rlPlayer, opponent, sharedModel);
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

            // Record end time
            long endTime = System.nanoTime();
            long totalTime = endTime - startTime;
            double averageTimePerEpisode = (double) totalTime / NUM_EPISODES / 1_000_000_000.0; // Convert to seconds
            double averageTimePerEpisodePerThread = (double) totalTime / NUM_EPISODES_PER_GAME_RUNNER / 1_000_000_000.0;
            logger.info("Average time per episode: " + averageTimePerEpisode + " seconds");
            logger.info("Average time per episode per thread: " + averageTimePerEpisodePerThread + " seconds");
            sharedModel.saveModel(MODEL_FILE_PATH);
        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
    }

    public void eval(int numEpisodes) {
        List<Path> deckFiles = null;
        try {  
            deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                                        .filter(Files::isRegularFile)
                                        .collect(Collectors.toList());
        } catch (IOException e) {
            logger.error("Error during evaluation", e);
        }
        if (deckFiles == null) {
            logger.error("No deck files found");
            return;
        }

        RLModel model = new RLModel();

        int winsAgainstComputerPlayer7 = 0;
        Random random = new Random();
        Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
        Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());

        for (int evalEpisode = 0; evalEpisode < numEpisodes; evalEpisode++) {
            TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false, 2));
            try {
                match.startGame();
            } catch (GameException e) {
                e.printStackTrace();
            }
            Game game = match.getGames().get(0);

            ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 10, model);
            game.addPlayer(rlPlayer, rlPlayerDeck);
            match.addPlayer(rlPlayer, rlPlayerDeck);

            Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
            Deck opponentDeck = loadDeck(opponentDeckPath.toString());
            ComputerPlayer7 opponent = new ComputerPlayer7("Player7", RangeOfInfluence.ALL, 3);
            game.addPlayer(opponent, opponentDeck);
            match.addPlayer(opponent, opponentDeck);

            game.loadCards(rlPlayerDeck.getCards(), rlPlayer.getId());
            game.loadCards(opponentDeck.getCards(), opponent.getId());

            GameOptions options = new GameOptions();
            game.setGameOptions(options);

            game.start(rlPlayer.getId());

            logGameResult(game, rlPlayer);

            if (game.getWinner().contains(rlPlayer.getName())) {
                winsAgainstComputerPlayer7++;
            }
        }

        double winRate = (double) winsAgainstComputerPlayer7 / NUM_EVAL_EPISODES;
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
            e.printStackTrace();
            return null;
        }
    }

    private synchronized void updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, ComputerPlayerRL opponent, RLModel model) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
        double reward = rlPlayerWon ? 1.0 : -1.0;

        // Get states for both players
        List<RLState> rlPlayerStates = rlPlayer.getStateBuffer();
        List<RLState> opponentStates = opponent.getStateBuffer();
        
        // Batch update for GPU
        List<RLState> allStates = new ArrayList<>(rlPlayerStates);
        allStates.addAll(opponentStates);
        List<Double> rewards = new ArrayList<>();
        // Add rewards for RL player states
        for (RLState ignored : rlPlayerStates) {
            rewards.add(reward);
        }
        // Add rewards for opponent states
        for (RLState state : opponentStates) {
            rewards.add(-reward);
        }
        // Adjust the sublist to exclude the last element
        model.updateBatch(allStates.subList(0, allStates.size() - 1), rewards, allStates.subList(1, allStates.size()));

    }

    public static RLModel getSharedModel() {
        return sharedModel;
    }

} 