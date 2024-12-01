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
import mage.util.ThreadUtils;

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
    private static final int NUM_EPISODES = 1;
    private static final String MODEL_PATH = "models/rl_model.zip";
    private static final int MAX_TURNS = 1000;

    public RLTrainer() {
        // Configure logger for multi-threaded environment
        org.apache.log4j.PatternLayout layout = new org.apache.log4j.PatternLayout();
        layout.setConversionPattern("[%t] %d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n");
        
        org.apache.log4j.ConsoleAppender appender = new org.apache.log4j.ConsoleAppender();
        appender.setLayout(layout);
        appender.setTarget("System.err");
        appender.activateOptions();
        
        logger.addAppender(appender);
        logger.setLevel(org.apache.log4j.Level.DEBUG);
    }

    public void train() {
        logger.debug("Starting RL training for " + NUM_EPISODES + " episodes");
        
        Thread gameThread = new Thread(() -> {
            try {
                logger.debug("Game thread started");
                for (int episode = 0; episode < NUM_EPISODES; episode++) {
                    logger.debug("Episode " + (episode + 1) + "/" + NUM_EPISODES);
                    
                    // Create new game instance for each episode
                    Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, 
                        RangeOfInfluence.ONE, 
                        new LondonMulligan(0), 
                        20, 7, 60);
                    
                    GameOptions options = new GameOptions();
                    options.testMode = true; // Enable test mode
                    options.skipInitShuffling = true;
                    game.setGameOptions(options);
                    
                    runEpisode(episode, game);
                    
                    // Ensure game is properly ended
                    if (!gameIsOver(game)) {
                        game.end();
                    }
                }
            } catch (Exception e) {
                logger.error("Training error: " + e.getMessage(), e);
            }
        }, "GAME");
        
        gameThread.start();
        try {
            gameThread.join();
        } catch (InterruptedException e) {
            logger.error("Training interrupted", e);
        }
        
        logger.debug("Training completed");
    }

    public void loadTrainedModel() {
        try {
            rlPlayer.model.getNetwork().loadModel(MODEL_PATH);
        } catch (IOException e) {
            logger.error("Failed to load model", e);
        }
    }

    private void runEpisode(int episode, Game game) {
        // Create players for this episode
        UUID rlPlayerId = UUID.randomUUID();
        UUID mctsPlayerId = UUID.randomUUID();
        ComputerPlayerRL rlPlayer = new ComputerPlayerRL(rlPlayerId);
        MCTSPlayer mctsPlayer = new MCTSPlayer(mctsPlayerId);
        mctsPlayer.setTestMode(true);
        
        // Set game options
        GameOptions options = new GameOptions();
        options.testMode = true;
        options.skipInitShuffling = true;
        game.setGameOptions(options);
        
        // Generate and load decks
        Deck rlDeck = generateDeck(rlPlayer.getId(), 60);
        Deck mctsDeck = generateDeck(mctsPlayer.getId(), 60);
        
        // Add players and their decks
        game.addPlayer(mctsPlayer, mctsDeck);
        game.addPlayer(rlPlayer, rlDeck);
        
        // Start the game and wait for initialization
        Thread gameThread = new Thread(() -> {
            Thread.currentThread().setName(ThreadUtils.THREAD_PREFIX_GAME + game.getId());
            game.start(rlPlayer.getId());
        }, "GAME-" + episode);
        
        gameThread.start();
        
        // Wait for game to properly initialize
        try {
            int attempts = 0;
            while (game.getActivePlayerId() == null && attempts < 10) {
                logger.debug("Waiting for active player to be loaded...");
                Thread.sleep(1000);
                attempts++;
            }
            if (game.getActivePlayerId() == null) {
                logger.error("Game failed to initialize after " + attempts + " seconds");
                return;
            }
        } catch (InterruptedException e) {
            logger.error("Game initialization interrupted", e);
            return;
        }
        
        // Monitor game state
        while (!gameIsOver(game)) {
            try {
                if (game.getActivePlayerId() != null) {
                    Player activePlayer = game.getPlayer(game.getActivePlayerId());
                    logger.info(String.format("Turn %d Phase %s - %s (Life: %d)", 
                        game.getTurnNum(),
                        game.getPhase().getType(),
                        activePlayer.getName(),
                        activePlayer.getLife()));

                    if (activePlayer.getId().equals(rlPlayer.getId())) {
                        RLState currentState = new RLState(game);
                        RLAction action = rlPlayer.model.getAction(currentState);
                        logger.info(String.format("RL Player action: %s [Cards in hand: %d, Lands on battlefield: %d]",
                            action.getType(),
                            activePlayer.getHand().size(),
                            game.getBattlefield().getAllActivePermanents(activePlayer.getId())
                                .stream()
                                .filter(permanent -> permanent.isLand())
                                .count()));
                        
                        boolean actionSuccess = action.execute(game, rlPlayer.getId());
                        if (actionSuccess) {
                            // Ensure game processes the action and advances state
                            game.processAction();
                            
                            RLState nextState = new RLState(game);
                            double reward = calculateReward(game, rlPlayer);
                            rlPlayer.model.update(currentState, action, reward, nextState);
                            logger.info(String.format("Action completed - Reward: %.2f", reward));
                        } else {
                            logger.info("Action failed to execute");
                        }
                    } else {
                        logger.info(String.format("MCTS Player thinking... [Cards: %d, Life: %d]",
                            mctsPlayer.getHand().size(),
                            mctsPlayer.getLife()));
                    }
                }
                Thread.sleep(100);
            } catch (InterruptedException e) {
                logger.error("Game monitoring interrupted", e);
                break;
            }
        }
        
        // Cleanup
        try {
            gameThread.join(5000);
            if (gameThread.isAlive()) {
                game.end();
            }
        } catch (InterruptedException e) {
            logger.error("Error waiting for game thread to finish", e);
        }
    }

    private boolean gameIsOver(Game game) {
        return game.hasEnded() || 
               game.checkIfGameIsOver() || 
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