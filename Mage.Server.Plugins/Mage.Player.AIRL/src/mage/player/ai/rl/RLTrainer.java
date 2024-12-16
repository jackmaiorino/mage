package mage.player.ai.rl;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.LondonMulligan;
import mage.player.ai.ComputerPlayerRL;
import mage.cards.decks.Deck;
import mage.game.GameOptions;
import mage.players.Player;
import org.apache.log4j.Logger;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.decks.DeckCardLists;
import mage.game.GameException;

import java.util.UUID;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.io.IOException;

public class RLTrainer {
    private static final Logger logger = Logger.getLogger(RLTrainer.class);
    private static final int NUM_EPISODES = 1;
    private static final String DECKS_DIRECTORY = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks";
    private static final String MODEL_FILE_PATH = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/model.ser";

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
            RLModel model = RLModel.loadModel(MODEL_FILE_PATH);
            if (model == null) {
                model = new RLModel();
            }

            for (int episode = 0; episode < NUM_EPISODES; episode++) {
                Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(7), 60, 60, 7);

                // Select a random deck for RL player
                Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 5, model);
                game.addPlayer(rlPlayer, rlPlayerDeck);

                // Select a random deck for opponent
                Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck opponentDeck = loadDeck(opponentDeckPath.toString());
                ComputerPlayerRL opponent = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, 5, model);
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

            // Save the model after training
            model.saveModel(MODEL_FILE_PATH);
        } catch (IOException e) {
            logger.error("Error reading decks directory", e);
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

    private void updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, ComputerPlayerRL opponent, RLModel model) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
        double reward = rlPlayerWon ? 1.0 : -1.0;

        // Update model for RL player
        for (Experience exp : rlPlayer.getExperienceBuffer()) {
            model.update(exp.state, reward, exp.nextState, exp.action);
        }
        rlPlayer.clearExperienceBuffer();

        // Update model for opponent
        reward = rlPlayerWon ? -1.0 : 1.0;
        for (Experience exp : opponent.getExperienceBuffer()) {
            model.update(exp.state, reward, exp.nextState, exp.action);
        }
        opponent.clearExperienceBuffer();
    }
} 