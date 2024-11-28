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
        UUID rlPlayerId = UUID.randomUUID();
        UUID mctsPlayerId = UUID.randomUUID();
        this.rlPlayer = new ComputerPlayerRL(rlPlayerId);
        this.mctsPlayer = new MCTSPlayer(mctsPlayerId);
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
        // First create game and set options
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
        Deck rlDeck = generateDeck(this.rlPlayer.getId(), 60);
        Deck mctsDeck = generateDeck(this.mctsPlayer.getId(), 60);
        
        // Load cards
        game.loadCards(rlDeck.getCards(), this.rlPlayer.getId());
        game.loadCards(mctsDeck.getCards(), this.mctsPlayer.getId());
        
        // Add players with their decks
        game.addPlayer(this.rlPlayer, rlDeck);
        game.addPlayer(this.mctsPlayer, mctsDeck);
        
        // Create simulation game for AI
        Game aiGame = game.createSimulationForAI();
        
        // Copy and restore player states
        for (Player copyPlayer : aiGame.getState().getPlayers().values()) {
            Player origPlayer = game.getState().getPlayers().get(copyPlayer.getId());
            if (copyPlayer.getId().equals(this.rlPlayer.getId())) {
                this.rlPlayer.restore(origPlayer);
                this.rlPlayer.init(aiGame);
            } else {
                this.mctsPlayer.restore(origPlayer);
                this.mctsPlayer.init(aiGame);
            }
        }
        
        aiGame.resume();
        aiGame.start(this.rlPlayer.getId());

        while (!gameIsOver(aiGame)) {
            RLState currentState = new RLState(aiGame);
            RLAction action = this.rlPlayer.model.getAction(currentState);
            
            // Execute action and get reward
            boolean actionSuccess = action.execute(aiGame, this.rlPlayer.getId());
            double reward = calculateReward(aiGame, this.rlPlayer);
            
            RLState nextState = new RLState(aiGame);
            
            // Update the model
            this.rlPlayer.model.update(currentState, action, reward, nextState);
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