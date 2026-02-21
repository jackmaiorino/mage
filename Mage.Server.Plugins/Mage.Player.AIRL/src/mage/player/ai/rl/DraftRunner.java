package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Logger;

import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.constants.RangeOfInfluence;
import mage.game.draft.BoosterDraft;
import mage.game.draft.DraftOptions;
import mage.game.draft.DraftPlayer;
import mage.player.ai.ComputerPlayer;
import mage.player.ai.ComputerPlayerRL;
import mage.tournament.cubes.CubeFromDeck;

/**
 * Headless driver for an 8-player Vintage Cube booster draft.
 *
 * Creates a BoosterDraft with a CubeFromDeck loaded from a .dck file,
 * adds 1 RL drafter + 7 heuristic ComputerPlayer bots, runs draft.start()
 * synchronously, and returns each player's drafted cards.
 */
public class DraftRunner {

    private static final Logger logger = Logger.getLogger(DraftRunner.class.getName());

    public static final int NUM_PLAYERS = 8;
    public static final int NUM_BOOSTERS = 3;

    public static final String CUBE_DECK_PATH = EnvConfig.str("CUBE_DECK_PATH",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Vintage/Cube/MTGOVintageCube.dck");

    /**
     * Result of a single draft episode.
     */
    public static class DraftResult {
        public final UUID rlPlayerId;
        /** playerId -> sideboard (all drafted cards, before construction) */
        public final Map<UUID, Deck> decks;

        public DraftResult(UUID rlPlayerId, Map<UUID, Deck> decks) {
            this.rlPlayerId = rlPlayerId;
            this.decks = decks;
        }

        public Deck getRlDeck() {
            return decks.get(rlPlayerId);
        }

        /** Returns a random heuristic-drafted deck (opponent side of self-play). */
        public Deck getRandomHeuristicDeck(Random rand) {
            List<UUID> others = new ArrayList<>();
            for (UUID id : decks.keySet()) {
                if (!id.equals(rlPlayerId)) {
                    others.add(id);
                }
            }
            if (others.isEmpty()) {
                return null;
            }
            return decks.get(others.get(rand.nextInt(others.size())));
        }
    }

    private final String cubeDeckPath;
    private Deck cubeDeck; // cached after first load

    public DraftRunner() {
        this(CUBE_DECK_PATH);
    }

    public DraftRunner(String cubeDeckPath) {
        this.cubeDeckPath = cubeDeckPath;
    }

    /**
     * Run a full 8-player draft synchronously.
     *
     * @param rlPlayer the RL agent (picks via draft model's pickCard override)
     * @return DraftResult containing all drafted decks, or null on error
     */
    public DraftResult runDraft(ComputerPlayerRL rlPlayer) {
        try {
            Deck cube = loadCubeDeck();
            if (cube == null) {
                logger.severe("Failed to load cube deck from: " + cubeDeckPath);
                return null;
            }

            CubeFromDeck cubeDraft = new CubeFromDeck(cube);

            DraftOptions options = new DraftOptions();
            options.setDraftCube(cubeDraft);
            options.setNumberBoosters(NUM_BOOSTERS);
            options.setTiming(DraftOptions.TimingOption.NONE);

            BoosterDraft draft = new BoosterDraft(options, Collections.emptyList());

            draft.addPlayer(rlPlayer);

            List<ComputerPlayer> bots = new ArrayList<>();
            for (int i = 0; i < NUM_PLAYERS - 1; i++) {
                ComputerPlayer bot = new ComputerPlayer("DraftBot-" + (i + 1), RangeOfInfluence.ALL);
                bots.add(bot);
                draft.addPlayer(bot);
            }

            // Mark all players as joined (required for the draft to proceed)
            for (DraftPlayer dp : draft.getPlayers()) {
                dp.setJoined();
            }

            draft.start();

            Map<UUID, Deck> results = new LinkedHashMap<>();
            for (DraftPlayer dp : draft.getPlayers()) {
                results.put(dp.getPlayer().getId(), dp.getDeck());
            }

            return new DraftResult(rlPlayer.getId(), results);

        } catch (Exception e) {
            logger.severe("Error running draft: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private synchronized Deck loadCubeDeck() {
        if (cubeDeck != null) {
            return cubeDeck;
        }
        try {
            DeckCardLists lists = DeckImporter.importDeckFromFile(cubeDeckPath, false);
            if (lists == null) {
                logger.severe("DeckImporter returned null for: " + cubeDeckPath);
                return null;
            }
            cubeDeck = Deck.load(lists, false, false);
            logger.info("Loaded cube deck from " + cubeDeckPath
                    + " (" + cubeDeck.getCards().size() + " cards)");
            return cubeDeck;
        } catch (Exception e) {
            logger.severe("Failed to load cube deck from " + cubeDeckPath + ": " + e.getMessage());
            return null;
        }
    }
}
