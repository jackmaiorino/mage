package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Scenarios shared with mtg-kernel's Homunculus Horde rules checks. */
public class FdnHomunculusCreaturesTest extends CardTestPlayerBase {

    private static final String HORDE = "Homunculus Horde";
    private static final String ORDER_TRIGGER = "Whenever you draw your second card each turn, create a token that's a copy of {this}.";

    @Test
    public void fourDrawsCreateOnlyOneCopy() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 5);
        addCard(Zone.BATTLEFIELD, playerA, HORDE);
        addCard(Zone.HAND, playerA, "Tidings");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Tidings", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, HORDE, 2);
        assertHandCount(playerA, 4);
    }

    @Test
    public void twoHordesEachCreateACopy() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.BATTLEFIELD, playerA, HORDE, 2);
        addCard(Zone.HAND, playerA, "Divination");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Divination", true);
        setChoice(playerA, ORDER_TRIGGER);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, HORDE, 4);
        assertHandCount(playerA, 2);
    }

    @Test
    public void theCopyCanTriggerAgainOnOpponentsTurn() {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 7);
        addCard(Zone.BATTLEFIELD, playerA, HORDE);
        addCard(Zone.HAND, playerA, "Divination");
        addCard(Zone.HAND, playerA, "Think Twice", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Divination", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, "Think Twice", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, "Think Twice", true);
        setChoice(playerA, ORDER_TRIGGER);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, HORDE, 4);
        assertHandCount(playerA, 4);
    }

    @Test
    public void countersAreNotCopiedButNameAndManaValueAre() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.BATTLEFIELD, playerA, HORDE);
        addCard(Zone.BATTLEFIELD, playerB, "Treetop Snarespinner");
        addCard(Zone.HAND, playerA, "Felling Blow");
        addCard(Zone.HAND, playerA, "Divination");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Felling Blow", HORDE + "^Treetop Snarespinner", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Divination", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, HORDE, 2);
        int copies = 0;
        for (Permanent permanent : currentGame.getBattlefield().getAllActivePermanents()) {
            if (!HORDE.equals(permanent.getName())) {
                continue;
            }
            int expected = permanent instanceof PermanentToken ? 2 : 5;
            Assert.assertEquals(expected, permanent.getPower().getValue());
            Assert.assertEquals(expected, permanent.getToughness().getValue());
            Assert.assertEquals(4, permanent.getManaValue());
            Assert.assertTrue(permanent.getColor(currentGame).isBlue());
            Assert.assertTrue(permanent.hasSubtype(SubType.HOMUNCULUS, currentGame));
            if (permanent instanceof PermanentToken) {
                copies++;
            }
        }
        Assert.assertEquals(1, copies);
    }

    @Test
    public void opposingDrawsDoNotTriggerTheControllersHorde() {
        addCard(Zone.BATTLEFIELD, playerA, HORDE);
        addCard(Zone.BATTLEFIELD, playerB, "Island", 4);
        addCard(Zone.HAND, playerB, "Think Twice", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Think Twice", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Think Twice", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, HORDE, 1);
        assertHandCount(playerB, 2);
    }
}
