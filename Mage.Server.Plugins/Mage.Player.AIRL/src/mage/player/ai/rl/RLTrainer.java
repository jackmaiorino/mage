package mage.player.ai.rl;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.LondonMulligan;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.ComputerPlayer6;
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
        for (int episode = 0; episode < NUM_EPISODES; episode++) {
            Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(7), 60, 60, 7);
            
            // Create RL player
            ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL", RangeOfInfluence.ALL , 5);
            game.addPlayer(rlPlayer, generateDeck(rlPlayer.getId(), 60));
            
            // Create ComputerPlayer6 opponent (since ComputerPlayer7 extends it)
            ComputerPlayer6 opponent = new ComputerPlayer6("Computer6", RangeOfInfluence.ALL, 5);
            game.addPlayer(opponent, generateDeck(opponent.getId(), 60));
            
            GameOptions options = new GameOptions();
            game.setGameOptions(options);

            // Start the game
            game.start(rlPlayer.getId());
            
            // Log final game state
            logGameResult(game, rlPlayer);
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
} 