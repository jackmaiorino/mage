package mage.player.ai.rl;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.LondonMulligan;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.MCTSPlayer;
import mage.cards.decks.Deck;
import mage.game.GameOptions;
import mage.players.Player;
import mage.util.ThreadUtils;
import org.apache.log4j.Logger;
import mage.cards.CardSetInfo;
import mage.cards.basiclands.Forest;
import mage.constants.Rarity;

import java.util.UUID;
import java.util.stream.Stream;

public class RLTrainer {
    private static final Logger logger = Logger.getLogger(RLTrainer.class);
    private static final int NUM_EPISODES = 1;
    private static final String MODEL_PATH = "models/rl_model.zip";
    
    private Deck generateDeck(UUID playerId, int count) {
        Deck deck = new Deck();
        Stream.generate(() -> new Forest(playerId, new CardSetInfo("Forest", "TEST", "1", Rarity.LAND)))
                .limit(count)
                .forEach(deck.getCards()::add);
        return deck;
    }
    
    public void train() {
        logger.debug("Starting RL training for " + NUM_EPISODES + " episodes");
        
        try {
            for (int episode = 0; episode < NUM_EPISODES; episode++) {
                logger.debug("Episode " + (episode + 1) + "/" + NUM_EPISODES);
                
                // Create game instance
                Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, 
                    RangeOfInfluence.ONE, 
                    new LondonMulligan(0), 
                    20, 7, 60);
                
                // Create players
                UUID rlPlayerId = UUID.randomUUID();
                UUID mctsPlayerId = UUID.randomUUID();
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL(rlPlayerId);
                MCTSPlayer mctsPlayer = new MCTSPlayer(mctsPlayerId);
                
                // Generate decks
                Deck rlDeck = generateDeck(rlPlayerId, 60);
                Deck mctsDeck = generateDeck(mctsPlayerId, 60);
                
                // Add players and decks
                game.addPlayer(mctsPlayer, mctsDeck);
                game.addPlayer(rlPlayer, rlDeck);
                
                // Set game options
                GameOptions options = new GameOptions();
                options.testMode = true;
                game.setGameOptions(options);
                
                // Start game in separate thread
                game.start(rlPlayer.getId());
                // Thread gameThread = new Thread(() -> {
                //     ThreadUtils.ensureRunInGameThread();
                //     game.start(rlPlayer.getId());
                // }, "GAME-" + episode);
                // gameThread.start();
                
                // Wait for game initialization with timeout
                int maxWaitTime = 30; // 30 seconds timeout
                int waitTime = 0;
                boolean playersInitialized = false;
                
                while (!playersInitialized && waitTime < maxWaitTime) {
                    Player checkPlayer = game.getPlayer(rlPlayerId);
                    if (checkPlayer != null && game.getState().getPlayers().containsKey(rlPlayerId)) {
                        playersInitialized = true;
                    } else {
                        try {
                            Thread.sleep(1000);
                            waitTime++;
                        } catch (InterruptedException e) {
                            logger.error("Wait interrupted", e);
                            break;
                        }
                    }
                }
                
                if (!playersInitialized) {
                    logger.error("Game initialization failed - timeout after " + maxWaitTime + " seconds");
                    continue;
                }
                
                // Monitor and train
                monitorGameAndTrain(game, rlPlayer);
            }
        } catch (Exception e) {
            logger.error("Training error: " + e.getMessage(), e);
        }
    }
    
    private void monitorGameAndTrain(Game game, ComputerPlayerRL rlPlayer) {
        try {
            while (!game.hasEnded()) {  // Changed from isGameOver() to hasEnded()
                if (game.getActivePlayerId() != null) {
                    Player activePlayer = game.getPlayer(game.getActivePlayerId());
                    if (activePlayer instanceof ComputerPlayerRL) {
                        synchronized(game) {
                            RLState currentState = new RLState(game);
                            RLAction action = rlPlayer.model.getAction(currentState);
                            
                            if (action != null && action.execute(game, rlPlayer.getId())) {
                                RLState nextState = new RLState(game);
                                double reward = evaluateGameState(game, rlPlayer);
                                rlPlayer.model.update(currentState, action, reward, nextState);
                                
                                logState(game, action, reward);
                            }
                        }
                    }
                }
                Thread.sleep(100);
            }
        } catch (InterruptedException e) {
            logger.error("Game monitoring interrupted", e);
        }
    }
    
    private double evaluateGameState(Game game, ComputerPlayerRL player) {
        // Simple reward based on life totals
        UUID opponentId = game.getOpponents(player.getId()).iterator().next();
        return game.getPlayer(player.getId()).getLife() - game.getPlayer(opponentId).getLife();
    }
    
    private void logState(Game game, RLAction action, double reward) {
        logger.info("=================== Turn " + game.getTurnNum() + " ===================");
        logger.info("Active Player: " + game.getPlayer(game.getActivePlayerId()).getName());
        logger.info("Action: " + action);
        
        // Log RL player state
        Player rlPlayer = game.getPlayer(game.getActivePlayerId());
        logger.info("[" + rlPlayer.getName() + "]" 
                + " Life: " + rlPlayer.getLife()
                + " Hand: " + rlPlayer.getHand().size()
                + " Reward: " + String.format("%.2f", reward));
        
        // Log opponent state
        UUID opponentId = game.getOpponents(rlPlayer.getId()).iterator().next();
        Player opponent = game.getPlayer(opponentId);
        logger.info("[" + opponent.getName() + "]"
                + " Life: " + opponent.getLife()
                + " Hand: " + opponent.getHand().size());
        
        logger.info(""); // Empty line for readability
    }
} 