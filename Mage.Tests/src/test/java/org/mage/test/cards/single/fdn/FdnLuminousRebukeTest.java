package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.Zone;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

public class FdnLuminousRebukeTest extends CardTestPlayerBase {
    private static final String REBUKE = "Luminous Rebuke";

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.END_TURN);
        execute();
    }

    @Test
    public void tappedCreatureCostsTwoMana() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic", true);
        finish();
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertGraveyardCount(playerA, REBUKE, 1);
        assertTappedCount("Plains", true, 2);
    }

    @Test
    public void untappedCreatureCostsFiveMana() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 5);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic");
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic", true);
        finish();
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertTappedCount("Plains", true, 5);
    }

    @Test
    public void tappedTargetStillUsesOnlyTwoWithFiveAvailable() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 5);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic", true);
        finish();
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertTappedCount("Plains", true, 2);
        assertTappedCount("Plains", false, 3);
    }

    @Test
    public void whitePipRemainsRequired() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 5);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        checkPlayableAbility("white pip", 1, PhaseStep.PRECOMBAT_MAIN, playerA,
                "Cast " + REBUKE, false);
        finish();
        assertHandCount(playerA, REBUKE, 1);
        assertPermanentCount(playerB, "Elvish Mystic", 1);
    }

    @Test
    public void ownTappedCreatureIsLegal() {
        addCard(Zone.BATTLEFIELD, playerB, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerB, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, REBUKE, "Elvish Mystic", true);
        finish();
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertGraveyardCount(playerB, REBUKE, 1);
        assertTappedCount("Plains", true, 2);
    }

    @Test
    public void legendaryCreatureIsLegal() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Dwynen, Gilt-Leaf Daen", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Dwynen, Gilt-Leaf Daen", true);
        finish();
        assertGraveyardCount(playerB, "Dwynen, Gilt-Leaf Daen", 1);
    }

    @Test
    public void rangerUntapResponseDoesNotRepriceOrInvalidateTarget() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.BATTLEFIELD, playerB, "Quirion Ranger");
        addCard(Zone.BATTLEFIELD, playerB, "Forest");
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic");
        activateAbility(1, PhaseStep.PRECOMBAT_MAIN, playerB,
                "Return a Forest you control", "Elvish Mystic", REBUKE);
        setChoice(playerB, "Forest");
        finish();
        assertHandCount(playerB, "Forest", 1);
        assertPermanentCount(playerB, "Quirion Ranger", 1);
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertGraveyardCount(playerA, REBUKE, 1);
        assertTappedCount("Plains", true, 2);
    }

    @Test
    public void blinkResponsePreservesNewIncarnation() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        addCard(Zone.HAND, playerB, "Momentary Blink");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Momentary Blink", "Elvish Mystic");
        finish();
        assertPermanentCount(playerB, "Elvish Mystic", 1);
        assertGraveyardCount(playerA, REBUKE, 1);
        assertGraveyardCount(playerB, "Momentary Blink", 1);
    }

    @Test
    public void wardTwoIsPaidAfterDiscountedSpellCost() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 4);
        addCard(Zone.BATTLEFIELD, playerB, "Cackling Prowler", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Cackling Prowler");
        setChoice(playerA, true);
        finish();
        assertGraveyardCount(playerB, "Cackling Prowler", 1);
        assertTappedCount("Plains", true, 4);
    }

    @Test
    public void decliningPayableWardCountersSpell() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 4);
        addCard(Zone.BATTLEFIELD, playerB, "Cackling Prowler", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Cackling Prowler");
        setChoice(playerA, false);
        finish();
        assertPermanentCount(playerB, "Cackling Prowler", 1);
        assertGraveyardCount(playerA, REBUKE, 1);
        assertTappedCount("Plains", true, 2);
    }

    @Test
    public void killingCreatureEnablesOwnProwlerEndStep() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Cackling Prowler");
        addCard(Zone.BATTLEFIELD, playerB, "Elvish Mystic", 1, true);
        addCard(Zone.HAND, playerA, REBUKE);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, REBUKE, "Elvish Mystic", true);
        finish();
        assertGraveyardCount(playerB, "Elvish Mystic", 1);
        assertPowerToughness(playerA, "Cackling Prowler", 5, 4);
    }
}
