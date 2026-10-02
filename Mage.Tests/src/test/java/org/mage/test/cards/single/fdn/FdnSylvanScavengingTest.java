package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.Zone;
import mage.counters.CounterType;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Reference cases for mode selection before targets and ferocious on resolution. */
public class FdnSylvanScavengingTest extends CardTestPlayerBase {
    private static final String SCAVENGING = "Sylvan Scavenging";
    private static final String ELF = "Elvish Mystic";
    private static final String RACCOON = "Raccoon Token";

    private void setup() {
        addCard(Zone.BATTLEFIELD, playerA, SCAVENGING);
    }

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.UPKEEP);
        execute();
    }

    @Test
    public void counterModePutsOneCounterOnTheControlledCreature() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, ELF);
        setModeChoice(playerA, "1");
        addTarget(playerA, ELF);
        finish();
        assertCounterCount(ELF, CounterType.P1P1, 1);
        assertPermanentCount(playerA, RACCOON, 0);
    }

    @Test
    public void tokenModeCreatesAGreenThreeThreeAtExactlyFourPower() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Cackling Prowler");
        setModeChoice(playerA, "2");
        finish();
        assertPermanentCount(playerA, RACCOON, 1);
        assertPowerToughness(playerA, RACCOON, 3, 3);
    }

    @Test
    public void threePowerDoesNotCreateAToken() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Beast-Kin Ranger");
        setModeChoice(playerA, "2");
        finish();
        assertPermanentCount(playerA, RACCOON, 0);
    }

    @Test
    public void emptyBattlefieldCanSelectTokenModeAsANoOp() {
        setup();
        setModeChoice(playerA, "2");
        finish();
        assertPermanentCount(playerA, SCAVENGING, 1);
        assertPermanentCount(playerA, RACCOON, 0);
    }

    @Test
    public void anOpponentsLargeCreatureDoesNotSatisfyFerocious() {
        setup();
        addCard(Zone.BATTLEFIELD, playerB, "Koma, World-Eater");
        setModeChoice(playerA, "2");
        finish();
        assertPermanentCount(playerA, RACCOON, 0);
    }

    @Test
    public void enchantmentDoesNotTriggerAtTheOpponentsEndStep() {
        addCard(Zone.BATTLEFIELD, playerB, SCAVENGING);
        addCard(Zone.BATTLEFIELD, playerB, "Cackling Prowler");
        finish();
        assertPermanentCount(playerB, RACCOON, 0);
    }

    @Test
    public void powerGainedInResponseEnablesTokenCreation() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Beast-Kin Ranger");
        addCard(Zone.BATTLEFIELD, playerA, "Forest");
        addCard(Zone.HAND, playerA, "Giant Growth");
        setModeChoice(playerA, "2");
        castSpell(1, PhaseStep.END_TURN, playerA, "Giant Growth", "Beast-Kin Ranger");
        finish();
        assertPermanentCount(playerA, RACCOON, 1);
    }

    @Test
    public void powerLostInResponsePreventsTokenCreation() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Cackling Prowler");
        addCard(Zone.BATTLEFIELD, playerA, "Island");
        addCard(Zone.HAND, playerA, "Fleeting Distraction");
        setModeChoice(playerA, "2");
        castSpell(1, PhaseStep.END_TURN, playerA, "Fleeting Distraction", "Cackling Prowler");
        finish();
        assertPermanentCount(playerA, RACCOON, 0);
        assertGraveyardCount(playerA, "Fleeting Distraction", 1);
    }

    @Test
    public void counterTriggerSurvivesRemovalOfTheEnchantment() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, ELF);
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.HAND, playerA, "Disenchant");
        setModeChoice(playerA, "1");
        addTarget(playerA, ELF);
        castSpell(1, PhaseStep.END_TURN, playerA, "Disenchant", SCAVENGING);
        finish();
        assertGraveyardCount(playerA, SCAVENGING, 1);
        assertCounterCount(ELF, CounterType.P1P1, 1);
    }

    @Test
    public void counterTriggerFizzlesAfterItsTargetLeaves() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, ELF);
        addCard(Zone.BATTLEFIELD, playerA, "Island", 2);
        addCard(Zone.HAND, playerA, "Unsummon");
        setModeChoice(playerA, "1");
        addTarget(playerA, ELF);
        castSpell(1, PhaseStep.END_TURN, playerA, "Unsummon", ELF);
        finish();
        assertHandCount(playerA, ELF, 1);
        assertPermanentCount(playerA, ELF, 0);
    }

    @Test
    public void createdTokenTriggersUnicornAndAngel() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Cackling Prowler");
        addCard(Zone.BATTLEFIELD, playerA, "Good-Fortune Unicorn");
        addCard(Zone.BATTLEFIELD, playerA, "Dazzling Angel");
        setModeChoice(playerA, "2");
        setChoice(playerA, "put a +1/+1 counter on that creature");
        finish();
        assertPermanentCount(playerA, RACCOON, 1);
        assertCounterCount(RACCOON, CounterType.P1P1, 1);
        assertLife(playerA, 21);
    }

    @Test
    public void castingCostsExactlyThreeManaIncludingTwoGreen() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.HAND, playerA, SCAVENGING);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, SCAVENGING);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, SCAVENGING, 1);
        assertTappedCount("Forest", true, 3);
    }
}
