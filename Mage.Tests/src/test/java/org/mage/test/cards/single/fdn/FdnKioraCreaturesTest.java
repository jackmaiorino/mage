package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.SubType;
import mage.constants.SuperType;
import mage.constants.Zone;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Reference cases for Kiora's ordered loot and intervening threshold. */
public class FdnKioraCreaturesTest extends CardTestPlayerBase {
    private static final String KIORA = "Kiora, the Rising Tide";
    private static final String SCION = "Scion of the Deep";

    @Test
    public void entryDrawsTwoThenDiscardsTwoForThreeMana() {
        skipInitShuffling();
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.HAND, playerA, KIORA);
        addCard(Zone.HAND, playerA, "Forest");
        addCard(Zone.LIBRARY, playerA, "Mountain", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, KIORA, true);
        setChoice(playerA, "Forest^Mountain");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPowerToughness(playerA, KIORA, 3, 2);
        assertTappedCount("Island", true, 3);
        assertHandCount(playerA, "Mountain", 1);
        assertHandCount(playerA, 1);
        assertGraveyardCount(playerA, "Forest", 1);
        assertGraveyardCount(playerA, "Mountain", 1);
    }

    @Test
    public void emptyHandDiscardsBothNewlyDrawnCards() {
        skipInitShuffling();
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.HAND, playerA, KIORA);
        addCard(Zone.LIBRARY, playerA, "Mountain", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, KIORA, true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertHandCount(playerA, 0);
        assertGraveyardCount(playerA, "Mountain", 2);
    }

    @Test
    public void sixOwnCardsAndSevenOpposingCardsDoNotMeetThreshold() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 6);
        addCard(Zone.GRAVEYARD, playerB, "Island", 7);
        attack(1, playerA, KIORA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertLife(playerB, 17);
        assertPermanentCount(playerA, SCION, 0);
    }

    @Test
    public void sevenOwnCardsCreateOneExactPrintedScion() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 7);
        attack(1, playerA, KIORA);
        setChoice(playerA, true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertLife(playerB, 17);
        assertPermanentCount(playerA, SCION, 1);
        assertPowerToughness(playerA, SCION, 8, 8);
        for (Permanent permanent : currentGame.getBattlefield().getAllActivePermanents()) {
            if (KIORA.equals(permanent.getName())) {
                Assert.assertEquals(3, permanent.getManaValue());
                Assert.assertTrue(permanent.hasSubtype(SubType.MERFOLK, currentGame));
                Assert.assertTrue(permanent.hasSubtype(SubType.NOBLE, currentGame));
                Assert.assertTrue(permanent.getSuperType(currentGame).contains(SuperType.LEGENDARY));
            }
            if (SCION.equals(permanent.getName())) {
                Assert.assertTrue(permanent instanceof PermanentToken);
                Assert.assertEquals(0, permanent.getManaValue());
                Assert.assertTrue(permanent.getColor(currentGame).isBlue());
                Assert.assertTrue(permanent.hasSubtype(SubType.OCTOPUS, currentGame));
                Assert.assertTrue(permanent.getSuperType(currentGame).contains(SuperType.LEGENDARY));
            }
        }
    }

    @Test
    public void sevenOwnCardsStillAllowRefusingTheScion() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 7);
        attack(1, playerA, KIORA);
        setChoice(playerA, false);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertPermanentCount(playerA, SCION, 0);
    }

    @Test
    public void losingThresholdInResponseSuppressesTheOptionalChoice() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 7);
        addCard(Zone.BATTLEFIELD, playerB, "Tormod's Crypt");
        attack(1, playerA, KIORA);
        activateAbility(1, PhaseStep.DECLARE_ATTACKERS, playerB,
                "{T}, Sacrifice Tormod's Crypt: Exile all cards from target player's graveyard.", playerA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertGraveyardCount(playerA, 0);
        assertExileCount(playerA, "Forest", 7);
        assertPermanentCount(playerA, SCION, 0);
    }

    @Test
    public void departedKioraStillCreatesItsQueuedScion() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 7);
        addCard(Zone.BATTLEFIELD, playerB, "Island");
        addCard(Zone.HAND, playerB, "Unsummon");
        attack(1, playerA, KIORA);
        castSpell(1, PhaseStep.DECLARE_ATTACKERS, playerB, "Unsummon", KIORA);
        setChoice(playerA, true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertHandCount(playerA, KIORA, 1);
        assertPermanentCount(playerA, KIORA, 0);
        assertPermanentCount(playerA, SCION, 1);
        assertLife(playerB, 20);
    }

    @Test
    public void secondScionUsesTheLegendRule() {
        addCard(Zone.BATTLEFIELD, playerA, KIORA);
        addCard(Zone.GRAVEYARD, playerA, "Forest", 7);
        attack(1, playerA, KIORA);
        attack(3, playerA, KIORA);
        setChoice(playerA, true);
        setChoice(playerA, true);
        setChoice(playerA, SCION);
        setStrictChooseMode(true);
        setStopAt(3, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertPermanentCount(playerA, SCION, 1);
        assertLife(playerB, 14);
    }

    @Test
    public void lootingQueuesMysticAndHordeSecondDrawTriggersAfterDiscard() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.BATTLEFIELD, playerA, "Mischievous Mystic");
        addCard(Zone.BATTLEFIELD, playerA, "Homunculus Horde");
        addCard(Zone.HAND, playerA, KIORA);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, KIORA, true);
        setChoice(playerA, "Whenever you draw your second card each turn, create a token that's a copy of {this}.");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertHandCount(playerA, 0);
        assertPermanentCount(playerA, "Homunculus Horde", 2);
        assertPermanentCount(playerA, "Faerie Token", 1);
    }
}
