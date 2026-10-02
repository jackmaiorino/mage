package org.mage.test.cards.single.fdn;

import mage.abilities.keyword.FlyingAbility;
import mage.constants.PhaseStep;
import mage.constants.Zone;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Scenarios shared with mtg-kernel's Foundations draw creature tests. */
public class FdnDrawCreaturesTest extends CardTestPlayerBase {

    @Test
    public void fourDrawsCreateOnlyOneFaerie() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 5);
        addCard(Zone.BATTLEFIELD, playerA, "Mischievous Mystic");
        addCard(Zone.HAND, playerA, "Tidings");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Tidings", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertHandCount(playerA, 4);
        assertPermanentCount(playerA, "Faerie Token", 1);
        assertPowerToughness(playerA, "Faerie Token", 1, 1);
        assertAbility(playerA, "Faerie Token", FlyingAbility.getInstance(), true);
    }

    @Test
    public void twoMysticsEachCreateTheirOwnFaerie() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.BATTLEFIELD, playerA, "Mischievous Mystic", 2);
        addCard(Zone.HAND, playerA, "Divination");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Divination", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertHandCount(playerA, 2);
        assertPermanentCount(playerA, "Faerie Token", 2);
    }

    @Test
    public void drawingTwiceOnOpponentsTurnCreatesAnotherFaerie() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 7);
        addCard(Zone.BATTLEFIELD, playerA, "Mischievous Mystic");
        addCard(Zone.HAND, playerA, "Divination");
        addCard(Zone.HAND, playerA, "Think Twice", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Divination", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, "Think Twice", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, "Think Twice", true);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, "Faerie Token", 2);
        assertHandCount(playerA, 4);
        assertGraveyardCount(playerA, "Think Twice", 2);
    }

    @Test
    public void vigilantLookoutCanLootAfterAttacking() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Strix Lookout");
        addCard(Zone.HAND, playerA, "Forest");
        attack(1, playerA, "Strix Lookout");
        activateAbility(1, PhaseStep.POSTCOMBAT_MAIN, playerA,
                "{1}{U}, {T}: Draw a card, then discard a card.");
        setChoice(playerA, "Forest");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();
        assertLife(playerB, 19);
        assertTapped("Strix Lookout", true);
        assertHandCount(playerA, 1);
        assertGraveyardCount(playerA, "Forest", 1);
    }

    @Test
    public void lookoutSecondDrawCreatesFaerieAfterDiscard() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 4);
        addCard(Zone.BATTLEFIELD, playerA, "Strix Lookout");
        addCard(Zone.BATTLEFIELD, playerA, "Mischievous Mystic");
        addCard(Zone.HAND, playerA, "Forest");
        addCard(Zone.HAND, playerA, "Think Twice");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Think Twice", true);
        activateAbility(1, PhaseStep.PRECOMBAT_MAIN, playerA,
                "{1}{U}, {T}: Draw a card, then discard a card.");
        setChoice(playerA, "Forest");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, "Faerie Token", 1);
        assertHandCount(playerA, 2);
        assertTapped("Strix Lookout", true);
        assertGraveyardCount(playerA, "Forest", 1);
    }
}
