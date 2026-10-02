package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.game.permanent.Permanent;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

public class FdnProwlerCreaturesTest extends CardTestPlayerBase {
    private static final String PROWLER = "Cackling Prowler";

    private void finish(int turn) {
        setStrictChooseMode(true);
        setStopAt(turn, PhaseStep.END_TURN);
        execute();
    }

    @Test
    public void exactFourManaCostAndPrintedCharacteristics() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 4);
        addCard(Zone.HAND, playerA, PROWLER);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, PROWLER, true);
        finish(1);
        assertPowerToughness(playerA, PROWLER, 4, 3);
        for (Permanent permanent : currentGame.getBattlefield().getAllActivePermanents()) {
            if (PROWLER.equals(permanent.getName())) {
                Assert.assertEquals(4, permanent.getManaValue());
                Assert.assertTrue(permanent.getColor(currentGame).isGreen());
                Assert.assertTrue(permanent.hasSubtype(SubType.HYENA, currentGame));
                Assert.assertTrue(permanent.hasSubtype(SubType.ROGUE, currentGame));
            }
        }
    }

    @Test
    public void noCreatureDeathMeansNoCounter() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        finish(1);
        assertPowerToughness(playerA, PROWLER, 4, 3);
    }

    @Test
    public void ownCreatureDeathAddsOneCounter() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerA, "Elvish Mystic");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        finish(1);
        assertGraveyardCount(playerA, "Elvish Mystic", 1);
        assertPowerToughness(playerA, PROWLER, 5, 4);
    }

    @Test
    public void opposingCreatureDeathBeforeProwlerEntersStillCounts() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 4);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        addCard(Zone.HAND, playerA, PROWLER);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, PROWLER, true);
        finish(1);
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertPowerToughness(playerA, PROWLER, 5, 4);
    }

    @Test
    public void multipleCreatureDeathsStillAddOnlyOneCounter() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerA, "Elvish Mystic");
        addCard(Zone.BATTLEFIELD, playerB, "Llanowar Elves");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 2);
        addCard(Zone.HAND, playerA, "Lightning Bolt", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Llanowar Elves", true);
        finish(1);
        assertPowerToughness(playerA, PROWLER, 5, 4);
    }

    @Test
    public void creatureTokenDeathCountsAfterTheTokenDisappears() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);
        addCard(Zone.HAND, playerA, "Dragon Fodder");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Dragon Fodder", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Goblin Token", true);
        finish(1);
        assertPermanentCount(playerA, "Goblin Token", 1);
        assertPowerToughness(playerA, PROWLER, 5, 4);
    }

    @Test
    public void noncreatureDeathDoesNotCount() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Bonesplitter");
        addCard(Zone.HAND, playerA, "Naturalize");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Naturalize", "Bonesplitter", true);
        finish(1);
        assertGraveyardCount(playerB, "Bonesplitter", 1);
        assertPowerToughness(playerA, PROWLER, 4, 3);
    }

    @Test
    public void opposingEndStepAndNextTurnDoNotReuseAnEarlierDeath() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        finish(2);
        assertPowerToughness(playerA, PROWLER, 5, 4);
        assertPowerToughness(playerB, PROWLER, 4, 3);
    }

    @Test
    public void deathAfterEndStepBeginsDoesNotCreateAMissedTrigger() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.END_TURN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        finish(1);
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertPowerToughness(playerA, PROWLER, 4, 3);
    }

    @Test
    public void blinkInResponseDoesNotGiveTheNewIncarnationAnOldCounter() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        addCard(Zone.HAND, playerA, "Momentary Blink");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", "Elvish Mystic", true);
        castSpell(1, PhaseStep.END_TURN, playerA, "Momentary Blink", PROWLER, true);
        finish(1);
        assertPowerToughness(playerA, PROWLER, 4, 3);
    }

    @Test
    public void decliningWardCountersTheOpposingSpell() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 3);
        addCard(Zone.HAND, playerB, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Unsummon", PROWLER);
        setChoice(playerB, false);
        finish(1);
        assertPermanentCount(playerA, PROWLER, 1);
        assertGraveyardCount(playerB, "Unsummon", 1);
    }

    @Test
    public void payingTwoForWardAllowsTheOpposingSpell() {
        addCard(Zone.BATTLEFIELD, playerA, PROWLER);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 3);
        addCard(Zone.HAND, playerB, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Unsummon", PROWLER);
        setChoice(playerB, true);
        finish(1);
        assertHandCount(playerA, PROWLER, 1);
        assertPermanentCount(playerA, PROWLER, 0);
    }
}
