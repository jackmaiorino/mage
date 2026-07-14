package org.mage.test.zones;

import mage.cards.CardSetInfo;
import mage.cards.basiclands.Forest;
import mage.cards.decks.Deck;
import mage.constants.MultiplayerAttackOption;
import mage.constants.Rarity;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.MulliganType;
import mage.players.StubPlayer;
import org.junit.Test;

import java.util.UUID;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * ReferenceRules v2 (Sol #106): regression test for the exact library
 * zone-duplication bug, reproduced at the call-order level that actually caused it.
 * <p>
 * RLTrainer.java's own game setup calls {@code Game.addPlayer(player, deck)} (which
 * correctly zones every maindeck card LIBRARY, via {@code PlayerImpl.useDeck ->
 * Library.addAll}) BEFORE {@code Game.loadCards(deck.getCards(), ...)} (which
 * unconditionally re-zones every one of those cards OUTSIDE via
 * {@code GameState.addCard}, clobbering the LIBRARY zone that was just set).
 * {@code MatchImpl.initGame} (the production/server path) and
 * {@code CardTestPlayerAPIImpl.createPlayer} (the one every other Mage.Tests card
 * scenario uses) both call these in the OTHER order (loadCards first, then
 * addPlayer/useDeck last) - which is exactly why this bug only ever showed up in the
 * RL harness, and why a standard {@code CardTestPlayerBase} scenario (see the
 * companion {@link ImpulseDrawAndMillZoneTest}) can't reproduce the ordering itself,
 * only its downstream symptoms.
 * <p>
 * This test builds a minimal game directly (same pattern as
 * {@code org.mage.test.mulligan.MulliganTestBase}: a real {@link TwoPlayerDuel} with
 * {@code play()} stubbed out, since only the post-init() zone state matters here, not
 * actual turns) using the RL harness's exact reversed call order, then asserts
 * {@code GameImpl.init()}'s zone-reconciliation step (see its own comment) still
 * leaves every physically-present library card recorded as {@code Zone.LIBRARY}
 * regardless. Without that reconciliation, every card in both libraries would still
 * be stuck at {@code Zone.OUTSIDE} here.
 */
public class ZoneReconciliationOrderingTest {

    @Test
    public void testAddPlayerBeforeLoadCardsStillZonesLibraryCorrectly() {
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
                MulliganType.GAME_DEFAULT.getMulligan(0), 60, 20, 7) {
            @Override
            protected void play(UUID nextPlayerId) {
                // post-init() zone state is all this test checks; no turns needed.
            }
        };
        GameOptions options = new GameOptions();
        options.skipInitShuffling = true;
        game.setGameOptions(options);

        StubPlayer player1 = new StubPlayer("p1", RangeOfInfluence.ONE);
        Deck deck1 = generateDeck(player1.getId(), 40);
        // The RL harness's exact (buggy) order: addPlayer/useDeck (zones LIBRARY)
        // THEN loadCards (re-zones OUTSIDE) - reversed from MatchImpl.initGame.
        game.addPlayer(player1, deck1);
        game.loadCards(deck1.getCards(), player1.getId());

        StubPlayer player2 = new StubPlayer("p2", RangeOfInfluence.ONE);
        Deck deck2 = generateDeck(player2.getId(), 40);
        game.addPlayer(player2, deck2);
        game.loadCards(deck2.getCards(), player2.getId());

        game.start(player1.getId());

        assertEquals("player1 should still have library cards left after opening draw",
                33, player1.getLibrary().size());
        for (UUID cardId : player1.getLibrary().getCardList()) {
            assertEquals("card " + cardId + " must be recorded as Zone.LIBRARY regardless of "
                            + "addPlayer/loadCards call order",
                    Zone.LIBRARY, game.getState().getZone(cardId));
        }
        for (UUID cardId : player2.getLibrary().getCardList()) {
            assertEquals("card " + cardId + " must be recorded as Zone.LIBRARY regardless of "
                            + "addPlayer/loadCards call order",
                    Zone.LIBRARY, game.getState().getZone(cardId));
        }
    }

    private static Deck generateDeck(UUID playerId, int count) {
        Deck deck = new Deck();
        Stream.generate(() -> new Forest(playerId, new CardSetInfo("Forest", "TEST", "1", Rarity.LAND)))
                .limit(count)
                .forEach(deck.getCards()::add);
        return deck;
    }
}
