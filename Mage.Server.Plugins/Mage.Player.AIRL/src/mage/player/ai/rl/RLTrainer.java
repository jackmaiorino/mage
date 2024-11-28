package mage.player.ai.rl;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameState;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.LondonMulligan;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.MCTSPlayer;
import mage.cards.repository.CardRepository;
import mage.cards.decks.Deck;
import mage.cards.Card;
import mage.cards.CardSetInfo;
import mage.cards.basiclands.Forest;
import mage.cards.basiclands.Mountain;
import mage.cards.g.GrizzlyBears;
import mage.cards.r.RagingGoblin;
import mage.constants.CardType;
import mage.constants.Rarity;
import mage.game.GameOptions;
import mage.players.Player;

import java.util.HashSet;
import java.util.UUID;
import java.util.Set;
import java.io.IOException;
import org.apache.log4j.Logger;
import java.util.stream.Stream;

import static mage.constants.CardType.CREATURE;
import static mage.constants.CardType.LAND;
import static mage.constants.Rarity.COMMON;

public class RLTrainer {
    private static final Logger logger = Logger.getLogger(RLTrainer.class);
    private ComputerPlayerRL rlPlayer;
    private MCTSPlayer mctsPlayer;
    private static final int NUM_EPISODES = 1000;
    private static final String MODEL_PATH = "models/rl_model.zip";

    public RLTrainer() {
        // No initialization needed here since players are created per episode in runEpisode()
    }

    public void train() {
        for (int episode = 0; episode < NUM_EPISODES; episode++) {
            runEpisode();
        }
        
        try {
            rlPlayer.model.getNetwork().saveModel(MODEL_PATH);
        } catch (IOException e) {
            logger.error("Failed to save model", e);
        }
    }

    public void loadTrainedModel() {
        try {
            rlPlayer.model.getNetwork().loadModel(MODEL_PATH);
        } catch (IOException e) {
            logger.error("Failed to load model", e);
        }
    }

    private void runEpisode() {
        // Create players for this episode
        UUID rlPlayerId = UUID.randomUUID();
        UUID mctsPlayerId = UUID.randomUUID();
        ComputerPlayerRL rlPlayer = new ComputerPlayerRL(rlPlayerId);
        MCTSPlayer mctsPlayer = new MCTSPlayer(mctsPlayerId);
        mctsPlayer.setTestMode(true);  // Enable test mode for AI simulation

        // Create game and set options
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, 
                                    RangeOfInfluence.ONE, 
                                    new LondonMulligan(0), 
                                    20,
                                    7,
                                    60);
        GameOptions options = new GameOptions();
        options.skipInitShuffling = true;
        game.setGameOptions(options);
        
        // Generate decks
        Deck rlDeck = generateDeck(rlPlayer.getId(), 60);
        Deck mctsDeck = generateDeck(mctsPlayer.getId(), 60);
        
        // Load cards
        game.loadCards(rlDeck.getCards(), rlPlayer.getId());
        game.loadCards(mctsDeck.getCards(), mctsPlayer.getId());
        
        // Add MCTS player first
        if (mctsPlayer == null) {
            logger.error("mctsPlayer is null before addPlayer");
            return;
        }
        if (mctsDeck == null) {
            logger.error("mctsDeck is null before addPlayer");
            return;
        }
        if (mctsDeck.getCards() == null) {
            logger.error("mctsDeck cards is null before addPlayer");
            return;
        }
        if (game == null) {
            logger.error("game is null before addPlayer");
            return;
        }

        System.out.println("DEBUG: mctsPlayer ID: " + mctsPlayer.getId());
        System.out.println("DEBUG: mctsDeck cards count: " + mctsDeck.getCards().size());

        game.addPlayer(mctsPlayer, mctsDeck);

        // Then add RL player
        if (rlPlayer == null) {
            System.out.println("DEBUG: rlPlayer is null before addPlayer");
            return;
        }
        if (rlDeck == null) {
            System.out.println("DEBUG: rlDeck is null before addPlayer");
            return;
        }
        if (rlDeck.getCards() == null) {
            System.out.println("DEBUG: rlDeck cards is null before addPlayer");
            return;
        }

        System.out.println("DEBUG: rlPlayer ID: " + rlPlayer.getId());
        System.out.println("DEBUG: rlDeck cards count: " + rlDeck.getCards().size());

        game.addPlayer(rlPlayer, rlDeck);
        
        // Create simulation game for AI
        Game aiGame = game.createSimulationForAI();
        
        // Copy and restore player states
        for (Player copyPlayer : aiGame.getState().getPlayers().values()) {
            Player origPlayer = game.getState().getPlayers().get(copyPlayer.getId());
            if (copyPlayer.getId().equals(rlPlayer.getId())) {
                rlPlayer.restore(origPlayer);
                rlPlayer.init(aiGame);
            } else {
                mctsPlayer.restore(origPlayer);
                mctsPlayer.init(aiGame);
            }
        }
        
        aiGame.resume();
        aiGame.start(rlPlayer.getId());

        while (!gameIsOver(aiGame)) {
            RLState currentState = new RLState(aiGame);
            RLAction action = rlPlayer.model.getAction(currentState);
            
            // Execute action and get reward
            boolean actionSuccess = action.execute(aiGame, rlPlayer.getId());
            double reward = calculateReward(aiGame, rlPlayer);
            
            RLState nextState = new RLState(aiGame);
            
            // Update the model
            rlPlayer.model.update(currentState, action, reward, nextState);
        }
    }

    private boolean gameIsOver(Game game) {
        return game.checkIfGameIsOver() || 
               game.getState().getPlayerList(game.getActivePlayerId()).isEmpty();
    }

    private double calculateReward(Game game, ComputerPlayerRL player) {
        if (gameIsOver(game)) {
            return game.getState().getPlayers().get(player.getId()).hasWon() ? 1.0 : -1.0;
        }
        return 0.0;
    }

    private Deck generateDeck(UUID playerId, int count) {
        Deck deck = new Deck();
        
        // Add cards to deck
        Stream.generate(() -> new Forest(playerId, new CardSetInfo("Forest", "TEST", "1", COMMON)))
                .limit(count / 2 + (count & 1))
                .forEach(card -> {
                    card.setOwnerId(playerId);
                    deck.getCards().add(card);
                });
                
        Stream.generate(() -> new GrizzlyBears(playerId, new CardSetInfo("Grizzly Bears", "TEST", "2", COMMON)))
                .limit(count / 2)
                .forEach(card -> {
                    card.setOwnerId(playerId);
                    deck.getCards().add(card);
                });
                
        return deck;
    }
} 