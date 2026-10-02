package org.mage.test.cards.single.fdn;

import mage.abilities.keyword.TrampleAbility;
import mage.constants.PhaseStep;
import mage.constants.Zone;
import mage.counters.CounterType;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Rules scenarios shared with mtg-kernel's Foundations counter-creature batch. */
public class FdnCounterCreaturesTest extends CardTestPlayerBase {

    @Test
    public void colonyWithoutKicker() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 5);
        addCard(Zone.HAND, playerA, "Gnarlid Colony");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Gnarlid Colony");
        setChoice(playerA, false);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertCounterCount(playerA, "Gnarlid Colony", CounterType.P1P1, 0);
        assertPowerToughness(playerA, "Gnarlid Colony", 2, 2);
        assertAbility(playerA, "Gnarlid Colony", TrampleAbility.getInstance(), false);
    }

    @Test
    public void colonyWithKicker() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 5);
        addCard(Zone.HAND, playerA, "Gnarlid Colony");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Gnarlid Colony");
        setChoice(playerA, true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertCounterCount(playerA, "Gnarlid Colony", CounterType.P1P1, 2);
        assertPowerToughness(playerA, "Gnarlid Colony", 4, 4);
        assertAbility(playerA, "Gnarlid Colony", TrampleAbility.getInstance(), true);
    }

    @Test
    public void colonyGrantRespectsControllerAndCounters() {
        addCard(Zone.BATTLEFIELD, playerA, "Gnarlid Colony");
        addCard(Zone.BATTLEFIELD, playerA, "Llanowar Elves");
        addCard(Zone.BATTLEFIELD, playerA, "Elite Vanguard");
        addCard(Zone.BATTLEFIELD, playerB, "Llanowar Elves");
        addCounters(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Llanowar Elves", CounterType.P1P1, 2);
        addCounters(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Llanowar Elves", CounterType.P1P1, 2);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertAbility(playerA, "Llanowar Elves", TrampleAbility.getInstance(), true);
        assertAbility(playerA, "Elite Vanguard", TrampleAbility.getInstance(), false);
        assertAbility(playerB, "Llanowar Elves", TrampleAbility.getInstance(), false);
    }

    @Test
    public void hydraEntersBeforeZeroToughnessCheck() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.HAND, playerA, "Mossborn Hydra");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Mossborn Hydra");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, "Mossborn Hydra", 1);
        assertCounterCount(playerA, "Mossborn Hydra", CounterType.P1P1, 1);
        assertPowerToughness(playerA, "Mossborn Hydra", 1, 1);
    }

    @Test
    public void hydraDoublesForEachControlledLand() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.HAND, playerA, "Mossborn Hydra");
        addCard(Zone.HAND, playerA, "Forest", 2);
        addCard(Zone.HAND, playerB, "Forest");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Mossborn Hydra");
        playLand(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Forest");
        playLand(2, PhaseStep.PRECOMBAT_MAIN, playerB, "Forest");
        playLand(3, PhaseStep.PRECOMBAT_MAIN, playerA, "Forest");
        setStrictChooseMode(true);
        setStopAt(3, PhaseStep.BEGIN_COMBAT);
        execute();
        assertCounterCount(playerA, "Mossborn Hydra", CounterType.P1P1, 4);
        assertPowerToughness(playerA, "Mossborn Hydra", 4, 4);
    }
}
