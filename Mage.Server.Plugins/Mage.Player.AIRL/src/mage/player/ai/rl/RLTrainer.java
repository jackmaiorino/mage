package mage.player.ai.rl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.Callable;
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
    private static final int NUM_EPISODES = 30;
    private static final int NUM_EVAL_EPISODES = 5;
    private static final String DECKS_DIRECTORY = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks";
    public static final String MODEL_FILE_PATH = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/network.ser";

    public RLTrainer() {
        // No need to create a model here
    }

    public void train() {
        try {
            List<Path> deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                                        .filter(Files::isRegularFile)
                                        .collect(Collectors.toList());

            Random random = new Random();

            // Load model for each player
            logger.info("Current working directory: " + System.getProperty("user.dir"));
            RLModel model = new RLModel();

            // Determine the number of available threads
            int numThreads = Runtime.getRuntime().availableProcessors();
            int episodesPerThread = NUM_EPISODES / numThreads;
            logger.info("Number of threads: " + numThreads);
            logger.info("Episodes per thread: " + episodesPerThread);

            // Create a fixed thread pool
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            List<Future<Void>> futures = new ArrayList<>();

            // Self-play training
            for (int i = 0; i < numThreads; i++) {
                Future<Void> future = executor.submit(() -> {
                    Thread.currentThread().setName("GAME");
                    for (int episode = 0; episode < episodesPerThread; episode++) {
                        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(7), 60, 20, 7);

                        // Select a random deck for RL player
                        Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                        Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 10, model);
                        game.addPlayer(rlPlayer, rlPlayerDeck);

                        // Select a random deck for opponent
                        Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                        Deck opponentDeck = loadDeck(opponentDeckPath.toString());
                        ComputerPlayerRL opponent = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, 10, model);
                        game.addPlayer(opponent, opponentDeck);

                        // Load cards into the game
                        game.loadCards(rlPlayerDeck.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeck.getCards(), opponent.getId());

                        GameOptions options = new GameOptions();
                        game.setGameOptions(options);

                        // Start the game
                        game.start(rlPlayer.getId());

                        // Log final game state
                        logGameResult(game, rlPlayer);

                        // Update model based on game outcome
                        updateModelBasedOnOutcome(game, rlPlayer, opponent, model);
                    }
                    return null;
                });
                futures.add(future);
            }

            // Shutdown the executor
            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

            // Check for exceptions
            for (Future<Void> future : futures) {
                try {
                    future.get(); // This will throw an exception if the task failed
                } catch (ExecutionException e) {
                    logger.error("Error in thread execution", e.getCause());
                    throw new RuntimeException(e.getCause());
                }
            }

            // Evaluation against ComputerPlayer7
            int winsAgainstComputerPlayer7 = 0;
            for (int evalEpisode = 0; evalEpisode < NUM_EVAL_EPISODES; evalEpisode++) {
                // This match wrapper is confusing
                // TODO: Why do we need a match wrapper?
                TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false, 2));
                try {
                    match.startGame();
                } catch (GameException e) {
                    e.printStackTrace();
                }
                Game game = match.getGames().get(0);

                // Select a random deck for RL player
                Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 10, model);
                game.addPlayer(rlPlayer, rlPlayerDeck);
                match.addPlayer(rlPlayer, rlPlayerDeck);

                // Select a random deck for ComputerPlayer7
                Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck opponentDeck = loadDeck(opponentDeckPath.toString());
                ComputerPlayer7 opponent = new ComputerPlayer7("Player7", RangeOfInfluence.ALL, 3);
                game.addPlayer(opponent, opponentDeck);
                match.addPlayer(opponent, opponentDeck);

                // Load cards into the game
                game.loadCards(rlPlayerDeck.getCards(), rlPlayer.getId());
                game.loadCards(opponentDeck.getCards(), opponent.getId());

                GameOptions options = new GameOptions();
                game.setGameOptions(options);

                // Start the game
                game.start(rlPlayer.getId());

                // Log final game state
                logGameResult(game, rlPlayer);

                // Check if RL player won
                if (game.getWinner().contains(rlPlayer.getName())) {
                    winsAgainstComputerPlayer7++;
                }
            }

            // Log win rate against ComputerPlayer7
            double winRate = (double) winsAgainstComputerPlayer7 / NUM_EVAL_EPISODES;
            logger.info("Win rate against ComputerPlayer7: " + (winRate * 100) + "%");

            // Save the model after training
            model.saveModel(MODEL_FILE_PATH);
        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
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
        // Update model for RL player
        double reward = rlPlayerWon ? 1.0 : -1.0;
        List<RLState> rlPlayerStates = rlPlayer.getStateBuffer();
        for (int i = 0; i < rlPlayerStates.size() - 1; i++) {
            RLState state = rlPlayerStates.get(i);
            RLState nextState = rlPlayerStates.get(i + 1);
            model.update(state, reward, nextState);
        }

        // Update model for opponent
        reward = rlPlayerWon ? -1.0 : 1.0;
        List<RLState> opponentStates = opponent.getStateBuffer();
        for (int i = 0; i < opponentStates.size() - 1; i++) {
            RLState state = opponentStates.get(i);
            RLState nextState = opponentStates.get(i + 1);
            model.update(state, reward, nextState);
        }
    }
} 