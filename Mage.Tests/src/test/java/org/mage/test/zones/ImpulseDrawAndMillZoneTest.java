package org.mage.test.zones;

import mage.cards.Card;
import mage.constants.PhaseStep;
import mage.constants.Zone;
import mage.game.ZoneInvariants;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

import java.util.UUID;

/**
 * ReferenceRules v2 (Sol #106 zone-duplication fix) regression suite.
 * <p>
 * Root bug: {@code GameState.addCard} zones every loaded card OUTSIDE (deliberate for
 * sideboard cards), and nothing historically re-zoned an un-drawn library card to
 * LIBRARY before some caller order left it stuck OUTSIDE. Any generic zone-change
 * effect targeting such a card (impulse-draw exile, mill, search) then hit
 * {@code CardImpl.removeFromZone}'s OUTSIDE branch, which reported success without
 * touching the physical {@code Library} container - duplicating the card (still
 * physically in the library, also placed in the destination zone).
 * <p>
 * {@code GameImpl.init()} now reconciles every library card's recorded zone to
 * LIBRARY before first untap (see its own comment). This class exercises the card
 * families the bug report specifically named: impulse-draw exile (Reckless Impulse /
 * Experimental Synthesizer / Clockwork Percussionist's shared
 * {@code ExileTopXMayPlayUntilEffect}), repeated/cascading impulse effects in one
 * turn, casting a card out of an impulse-draw exile window, and self-mill
 * (Balustrade Spy - the Spy combo's own mill, explicitly called out as needing its
 * own check).
 * <p>
 * Placement note: these are engine/zone-tracking regression tests, not single-card
 * rules tests, so they live in their own {@code org.mage.test.zones} package (same
 * precedent as {@code org.mage.test.mulligan}) rather than under
 * {@code org.mage.test.cards.single.<set>}. A full {@code Mage.Tests} run is not
 * required to validate this fix - each test class here runs standalone in the usual
 * ~30-45s JVM-startup-dominated window (measured via {@code PartyThrasherTest}, the
 * closest existing analog: 3 methods, ~35s total), so running this file (plus the
 * companion {@link ZoneReconciliationOrderingTest}) is the right-sized verification,
 * not a targeted-outside-Mage.Tests harness.
 */
public class ImpulseDrawAndMillZoneTest extends CardTestPlayerBase {

    /**
     * Direct pin of the GameImpl.init() fix itself: by the time the game is ready to
     * play, every card physically sitting in a player's library must be recorded as
     * Zone.LIBRARY. This is the exact invariant the whole regression suite depends on.
     */
    @Test
    public void testEveryLibraryCardIsZonedLibraryAtGameStart() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1);

        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.UPKEEP);
        execute();

        for (UUID cardId : playerA.getLibrary().getCardList()) {
            Assert.assertEquals("library card " + cardId + " must be recorded as Zone.LIBRARY",
                    Zone.LIBRARY, currentGame.getState().getZone(cardId));
        }
        for (UUID cardId : playerB.getLibrary().getCardList()) {
            Assert.assertEquals("library card " + cardId + " must be recorded as Zone.LIBRARY",
                    Zone.LIBRARY, currentGame.getState().getZone(cardId));
        }
    }

    /**
     * "Impulse exile removes exactly one library card": Reckless Impulse exiles the
     * top 2 - each of those 2 specific cards must leave the physical library (count
     * drops to 0 for each) and appear in exile (count 1), never both places at once.
     */
    @Test
    public void testImpulseExileRemovesExactlyOneLibraryCardEach() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1); // 2nd from top
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);        // top
        addCard(Zone.HAND, playerA, "Reckless Impulse", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);

        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Reckless Impulse");

        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();

        assertLibraryCount(playerA, "Memnite", 0);
        assertLibraryCount(playerA, "Elvish Mystic", 0);
        assertExileCount(playerA, "Memnite", 1);
        assertExileCount(playerA, "Elvish Mystic", 1);
    }

    /**
     * "The exiled card cannot later be drawn": once impulse-exiled, a card must never
     * resurface via a later normal draw, however many turns pass. Pre-fix, the
     * duplication bug left the card physically in the library, so a later draw would
     * eventually pull it into hand a second time (once via the exile-permission "may
     * play", once via a genuine top-of-library draw).
     */
    @Test
    public void testImpulseExiledCardsNeverReappearAfterMultipleDraws() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1);
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);
        addCard(Zone.HAND, playerA, "Reckless Impulse", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);

        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Reckless Impulse");
        // decline every "play the exiled card" window offered - just let turns pass
        setStrictChooseMode(true);
        setStopAt(6, PhaseStep.END_TURN);
        execute();

        assertLibraryCount(playerA, "Memnite", 0);
        assertLibraryCount(playerA, "Elvish Mystic", 0);
        assertHandCount(playerA, "Memnite", 0);
        assertHandCount(playerA, "Elvish Mystic", 0);
        assertExileCount(playerA, "Memnite", 1);
        assertExileCount(playerA, "Elvish Mystic", 1);
    }

    /**
     * "Exile-cast and repeated-impulse cascades": Reckless Impulse (exiles top 2)
     * followed, same turn, by Experimental Synthesizer's ETB (exiles the new top 1).
     * The second effect must grab the card that is now physically on top of the
     * library, not re-grab (or duplicate) a card the first effect already exiled -
     * exactly the failure mode the bug report traced through the coverage ledger
     * (impulse-draw "grabs the wrong physical card").
     */
    @Test
    public void testRepeatedImpulseCascadeGrabsDistinctCardsSameTurn() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Grizzly Bears", 1); // becomes new top after first exile
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1);
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);        // top
        addCard(Zone.HAND, playerA, "Reckless Impulse", 1);
        addCard(Zone.HAND, playerA, "Experimental Synthesizer", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);

        // Experimental Synthesizer is sorcery-speed (an artifact) - it can't be cast
        // while Reckless Impulse is still on the stack, so wait for it to resolve.
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Reckless Impulse", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Experimental Synthesizer");

        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();

        assertLibraryCount(playerA, "Memnite", 0);
        assertLibraryCount(playerA, "Elvish Mystic", 0);
        assertLibraryCount(playerA, "Grizzly Bears", 0);
        assertExileCount(playerA, "Memnite", 1);
        assertExileCount(playerA, "Elvish Mystic", 1);
        assertExileCount(playerA, "Grizzly Bears", 1);
    }

    /**
     * Casting a card out of an impulse-draw exile window must fully retire the
     * object from both the library (already gone) and exile (moves to its real
     * destination) - no leftover duplicate in any zone.
     */
    @Test
    public void testExiledCardCastCleansUpAllContainers() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1);
        addCard(Zone.LIBRARY, playerA, "Memnite", 1); // top: exiled then cast, {0} cost
        addCard(Zone.HAND, playerA, "Reckless Impulse", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);

        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Reckless Impulse");
        castSpell(1, PhaseStep.POSTCOMBAT_MAIN, playerA, "Memnite");

        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();

        assertLibraryCount(playerA, "Memnite", 0);
        assertExileCount(playerA, "Memnite", 0);
        assertPermanentCount(playerA, "Memnite", 1);
    }

    /**
     * "Spy combo's self-mill is in the pool's future - CHECK mill's path
     * specifically": Balustrade Spy reveals-and-mills top library cards until a land,
     * via a bespoke inline effect (not MillCardsControllerEffect, but the same
     * ZonesHandler.moveCard dispatch). Every milled card must leave the library
     * exactly once and land in the graveyard exactly once.
     */
    @Test
    public void testSelfMillRemovesEachMilledCardFromLibraryExactlyOnce() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Forest", 1);        // the land that stops the mill
        addCard(Zone.LIBRARY, playerA, "Elvish Mystic", 1);
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);        // top
        addCard(Zone.HAND, playerA, "Balustrade Spy", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Swamp", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);

        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Balustrade Spy");
        addTarget(playerA, playerA); // Balustrade Spy's ETB targets a player; mill self

        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();

        assertLibraryCount(playerA, "Memnite", 0);
        assertLibraryCount(playerA, "Elvish Mystic", 0);
        assertLibraryCount(playerA, "Forest", 0);
        assertGraveyardCount(playerA, "Memnite", 1);
        assertGraveyardCount(playerA, "Elvish Mystic", 1);
        assertGraveyardCount(playerA, "Forest", 1);
    }

    /**
     * Direct unit check of the fail-fast {@link ZoneInvariants} utility itself:
     * manufacture the bug's exact signature (a card recorded EXILED while still
     * physically sitting in its owner's library) and confirm it trips loudly instead
     * of silently passing. This is the belt-and-suspenders check that guards against
     * a *future* regression reintroducing the same class of bug somewhere
     * ZonesHandler doesn't already cover.
     */
    @Test
    public void testZoneInvariantsDetectsManufacturedDuplication() {
        skipInitShuffling();
        addCard(Zone.LIBRARY, playerA, "Memnite", 1);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.UPKEEP);
        execute();

        Card memnite = null;
        for (Card card : playerA.getLibrary().getCards(currentGame)) {
            if ("Memnite".equals(card.getName())) {
                memnite = card;
                break;
            }
        }
        Assert.assertNotNull("test setup must have Memnite in library", memnite);
        Assert.assertTrue("ZoneInvariants must be enabled for this test to be meaningful", ZoneInvariants.isEnabled());

        // Memnite is still physically in the library, but forcibly mis-record its
        // zone as EXILED - exactly the bug's signature (duplication: two containers
        // simultaneously "true" for the same card).
        currentGame.getState().setZone(memnite.getId(), Zone.EXILED);
        try {
            ZoneInvariants.checkCard(currentGame, memnite.getId(), "test-manufactured-mismatch");
            Assert.fail("ZoneInvariants.checkCard must throw on a manufactured library/exile zone mismatch");
        } catch (IllegalStateException expected) {
            // expected: fail-fast
        } finally {
            // restore, so assertAllCommandsUsed()/game teardown doesn't trip on it
            currentGame.getState().setZone(memnite.getId(), Zone.LIBRARY);
        }
    }
}
