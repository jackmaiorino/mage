package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.counters.CounterType;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Reference cases for mtg-kernel's next original-UG fixture creature. */
public class FdnKomaCreaturesTest extends CardTestPlayerBase {

    private static final String KOMA = "Koma, World-Eater";
    private static final String COIL = "Koma's Coil";

    @Test
    public void combatDamageCreatesFourPrintedCoils() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        attack(1, playerA, KOMA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertLife(playerB, 12);
        assertPowerToughness(playerA, KOMA, 8, 12);
        assertPermanentCount(playerA, COIL, 4);
        for (Permanent permanent : currentGame.getBattlefield().getAllActivePermanents()) {
            if (!COIL.equals(permanent.getName())) {
                continue;
            }
            Assert.assertTrue(permanent instanceof PermanentToken);
            Assert.assertEquals(3, permanent.getPower().getValue());
            Assert.assertEquals(3, permanent.getToughness().getValue());
            Assert.assertEquals(0, permanent.getManaValue());
            Assert.assertTrue(permanent.getColor(currentGame).isBlue());
            Assert.assertTrue(permanent.hasSubtype(SubType.SERPENT, currentGame));
        }
    }

    @Test
    public void fullyBlockedCombatDoesNotCreateCoils() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Tolarian Terror");
        addCounters(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Tolarian Terror", CounterType.P1P1, 4); // 9/9
        attack(1, playerA, KOMA);
        block(1, playerB, "Tolarian Terror", KOMA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertLife(playerB, 20);
        assertPermanentCount(playerA, KOMA, 1);
        assertPermanentCount(playerA, COIL, 0);
    }

    @Test
    public void trampleDamageToPlayerCreatesFourCoils() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Treetop Snarespinner"); // 1/4
        attack(1, playerA, KOMA);
        block(1, playerB, "Treetop Snarespinner", KOMA);
        setChoice(playerA, "X=4"); // Four to the blocker, four tramples through.
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.POSTCOMBAT_MAIN);
        execute();
        assertLife(playerB, 16);
        assertGraveyardCount(playerB, "Treetop Snarespinner", 1);
        assertGraveyardCount(playerA, KOMA, 1); // Simultaneous deathtouch damage.
        assertPermanentCount(playerA, COIL, 4);
    }

    @Test
    public void noncombatDamageDoesNotCreateCoils() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.BATTLEFIELD, playerB, "Treetop Snarespinner");
        addCard(Zone.HAND, playerA, "Felling Blow");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Felling Blow", KOMA + "^Treetop Snarespinner", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPowerToughness(playerA, KOMA, 9, 13);
        assertGraveyardCount(playerB, "Treetop Snarespinner", 1);
        assertPermanentCount(playerA, COIL, 0);
    }

    @Test
    public void counterspellCanTargetKomaButCannotCounterIt() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 5);
        addCard(Zone.BATTLEFIELD, playerA, "Island", 2);
        addCard(Zone.HAND, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 2);
        addCard(Zone.HAND, playerB, "Counterspell");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, KOMA);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Counterspell", KOMA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, KOMA, 1);
        assertGraveyardCount(playerA, KOMA, 0);
        assertGraveyardCount(playerB, "Counterspell", 1);
        assertPermanentCount(playerA, COIL, 0);
    }

    @Test
    public void decliningForceSpikePaymentStillCannotCounterKoma() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 6);
        addCard(Zone.BATTLEFIELD, playerA, "Island", 2);
        addCard(Zone.HAND, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Island");
        addCard(Zone.HAND, playerB, "Force Spike");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, KOMA);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Force Spike", KOMA);
        setChoice(playerA, false); // The payable choice remains available.
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, KOMA, 1);
        assertGraveyardCount(playerA, KOMA, 0);
        assertGraveyardCount(playerB, "Force Spike", 1);
    }

    @Test
    public void decliningWardCountersTheOpposingSpell() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 5);
        addCard(Zone.HAND, playerB, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Unsummon", KOMA);
        setChoice(playerB, false);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, KOMA, 1);
        assertGraveyardCount(playerB, "Unsummon", 1);
        assertHandCount(playerA, KOMA, 0);
    }

    @Test
    public void payingWardAllowsTheOpposingSpell() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 5);
        addCard(Zone.HAND, playerB, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Unsummon", KOMA);
        setChoice(playerB, true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, KOMA, 0);
        assertHandCount(playerA, KOMA, 1);
        assertGraveyardCount(playerB, "Unsummon", 1);
    }

    @Test
    public void ownControllerDoesNotPayWard() {
        addCard(Zone.BATTLEFIELD, playerA, KOMA);
        addCard(Zone.BATTLEFIELD, playerA, "Island");
        addCard(Zone.HAND, playerA, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Unsummon", KOMA);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, KOMA, 0);
        assertHandCount(playerA, KOMA, 1);
        assertGraveyardCount(playerA, "Unsummon", 1);
    }
}
