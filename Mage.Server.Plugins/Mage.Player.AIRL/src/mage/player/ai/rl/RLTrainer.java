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
import mage.cards.CardSetInfo;
import mage.cards.basiclands.Forest;
import mage.cards.basiclands.Mountain;
import mage.cards.g.GrizzlyBears;
import mage.cards.s.SwabGoblin;
import mage.constants.Rarity;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.decks.DeckCardLists;
import mage.game.GameException;

import java.util.UUID;
import java.util.stream.Stream;
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

    public void train() {
        try {
            List<Path> deckFiles = Files.list(Paths.get(DECKS_DIRECTORY))
                                        .filter(Files::isRegularFile)
                                        .collect(Collectors.toList());

            Random random = new Random();

            for (int episode = 0; episode < NUM_EPISODES; episode++) {
                Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL, new LondonMulligan(7), 60, 60, 7);

                // Select a random deck for RL player
                Path rlPlayerDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck rlPlayerDeck = loadDeck(rlPlayerDeckPath.toString());
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, 5);
                game.addPlayer(rlPlayer, rlPlayerDeck);

                // Select a random deck for opponent
                Path opponentDeckPath = deckFiles.get(random.nextInt(deckFiles.size()));
                Deck opponentDeck = loadDeck(opponentDeckPath.toString());
                ComputerPlayerRL opponent = new ComputerPlayerRL("PlayerRL2", RangeOfInfluence.ALL, 5);
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
            }
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

    private Deck generateForestGrizzlyDeck(UUID playerId) {
        Deck deck = new Deck();
        // Add 20 Forests
        Stream.generate(() -> new Forest(playerId, new CardSetInfo("Forest", "TEST", "1", Rarity.LAND)))
                .limit(20)
                .forEach(deck.getCards()::add);

        // Add 40 Grizzly Bears
        Stream.generate(() -> new GrizzlyBears(playerId, new CardSetInfo("Grizzly Bears", "TEST", "2", Rarity.COMMON)))
                .limit(40)
                .forEach(deck.getCards()::add);

        return deck;
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

    private Deck generateMountainGoblinDeck(UUID playerId) {
        // Fallback to manually creating a deck if loading fails
        Deck deck = new Deck();
        Stream.generate(() -> new Mountain(playerId, new CardSetInfo("Mountain", "TEST", "1", Rarity.LAND)))
                .limit(20)
                .forEach(deck.getCards()::add);

        Stream.generate(() -> new SwabGoblin(playerId, new CardSetInfo("Swab Goblin", "TEST", "2", Rarity.COMMON)))
                .limit(40)
                .forEach(deck.getCards()::add);

        return deck;
    }
} 