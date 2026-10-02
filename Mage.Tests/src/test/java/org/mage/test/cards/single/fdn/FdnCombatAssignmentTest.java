package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.Zone;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Valid damage assignments shared with the kernel's Foundations combat tests. */
public class FdnCombatAssignmentTest extends CardTestPlayerBase {
    private static final String FIRST = "Rumbling Baloth"; // 4/4
    private static final String SECOND = "Myr Enforcer"; // 4/4

    private void gangBlock(String attacker) {
        addCard(Zone.BATTLEFIELD, playerA, attacker);
        addCard(Zone.BATTLEFIELD, playerB, FIRST);
        addCard(Zone.BATTLEFIELD, playerB, SECOND);
        attack(1, playerA, attacker);
        block(1, playerB, FIRST, attacker);
        block(1, playerB, SECOND, attacker);
    }

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
    }

    @Test
    public void nontramplerCanAssignLessThanLethalToFirstBlocker() {
        gangBlock("Vorstclaw"); // 6/6 without trample
        setChoiceAmount(playerA, 2, 4);
        finish();

        assertDamageReceived(playerB, FIRST, 2);
        assertPermanentCount(playerB, FIRST, 1);
        assertGraveyardCount(playerB, SECOND, 1);
        assertGraveyardCount(playerA, "Vorstclaw", 1);
        assertLife(playerB, 20);
    }

    @Test
    public void tramplerCanOverassignSingleBlockerAndDealNoPlayerDamage() {
        addCard(Zone.BATTLEFIELD, playerA, "Colossal Dreadmaw"); // 6/6 trample
        addCard(Zone.BATTLEFIELD, playerB, SECOND);
        attack(1, playerA, "Colossal Dreadmaw");
        block(1, playerB, SECOND, "Colossal Dreadmaw");
        setChoiceAmount(playerA, 6);
        finish();

        assertGraveyardCount(playerB, SECOND, 1);
        assertPermanentCount(playerA, "Colossal Dreadmaw", 1);
        assertDamageReceived(playerA, "Colossal Dreadmaw", 4);
        assertLife(playerB, 20);
    }

    @Test
    public void tramplerCanSkipFirstBlockerWhenAllDamageStaysOnBlockers() {
        gangBlock("Colossal Dreadmaw");
        setChoiceAmount(playerA, 0, 6);
        finish();

        assertDamageReceived(playerB, FIRST, 0);
        assertPermanentCount(playerB, FIRST, 1);
        assertGraveyardCount(playerB, SECOND, 1);
        assertGraveyardCount(playerA, "Colossal Dreadmaw", 1);
        assertLife(playerB, 20);
    }

    @Test
    public void trampleDeathtouchAssignsOneToEachBlockerBeforeFourToPlayer() {
        gangBlock("Colossal Dreadmaw");
        addCard(Zone.BATTLEFIELD, playerA, "Bow of Nylea");
        setChoiceAmount(playerA, 1, 1);
        finish();

        assertGraveyardCount(playerB, FIRST, 1);
        assertGraveyardCount(playerB, SECOND, 1);
        assertGraveyardCount(playerA, "Colossal Dreadmaw", 1);
        assertLife(playerB, 16);
    }
}
