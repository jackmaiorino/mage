package org.mage.test.cards.single.fdn;

import mage.abilities.keyword.LifelinkAbility;
import mage.constants.PhaseStep;
import mage.constants.Zone;
import mage.counters.CounterType;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Scenarios shared with mtg-kernel's Foundations life-gain creature tests. */
public class FdnLifegainCreaturesTest extends CardTestPlayerBase {

    @Test
    public void separateGainsAddCountersButDrawOnce() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Exemplar of Light");
        addCard(Zone.BATTLEFIELD, playerA, "Dazzling Angel");
        addCard(Zone.HAND, playerA, "Llanowar Elves", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Llanowar Elves", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Llanowar Elves", true);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertLife(playerA, 22);
        assertCounterCount(playerA, "Exemplar of Light", CounterType.P1P1, 2);
        assertHandCount(playerA, 1);
    }

    @Test
    public void fellingBlowCounterTriggersOneDraw() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 3);
        addCard(Zone.BATTLEFIELD, playerA, "Exemplar of Light");
        addCard(Zone.BATTLEFIELD, playerB, "Treetop Snarespinner");
        addCard(Zone.HAND, playerA, "Felling Blow");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Felling Blow", "Exemplar of Light^Treetop Snarespinner");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertCounterCount(playerA, "Exemplar of Light", CounterType.P1P1, 1);
        assertPowerToughness(playerA, "Exemplar of Light", 4, 4);
        assertGraveyardCount(playerB, "Treetop Snarespinner", 1);
        assertHandCount(playerA, 1);
    }

    @Test
    public void drawLimitResetsOnOpponentsTurn() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Exemplar of Light");
        addCard(Zone.HAND, playerA, "Life Goes On", 2);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Life Goes On", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, "Life Goes On", true);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.BEGIN_COMBAT);
        execute();
        assertLife(playerA, 28);
        assertCounterCount(playerA, "Exemplar of Light", CounterType.P1P1, 2);
        assertHandCount(playerA, 2);
    }

    @Test
    public void unkickedHealerLeavesItsGraveyardAlone() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 4);
        addCard(Zone.HAND, playerA, "Sun-Blessed Healer");
        addCard(Zone.GRAVEYARD, playerA, "Llanowar Elves");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Sun-Blessed Healer");
        setChoice(playerA, false);
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertGraveyardCount(playerA, "Llanowar Elves", 1);
        assertPowerToughness(playerA, "Sun-Blessed Healer", 3, 1);
        assertAbility(playerA, "Sun-Blessed Healer", LifelinkAbility.getInstance(), true);
    }

    @Test
    public void kickedHealerReturnsArtifactAndItsEntryTrigger() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 4);
        addCard(Zone.HAND, playerA, "Sun-Blessed Healer");
        addCard(Zone.GRAVEYARD, playerA, "Ichor Wellspring");
        addCard(Zone.GRAVEYARD, playerA, "Exemplar of Light");
        addCard(Zone.GRAVEYARD, playerA, "Forest");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Sun-Blessed Healer");
        setChoice(playerA, true);
        addTarget(playerA, "Ichor Wellspring");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, "Ichor Wellspring", 1);
        assertGraveyardCount(playerA, "Exemplar of Light", 1);
        assertGraveyardCount(playerA, "Forest", 1);
        assertHandCount(playerA, 1);
    }

    @Test
    public void returningAuraCanChooseHexproofHost() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 4);
        addCard(Zone.BATTLEFIELD, playerB, "Slippery Bogle");
        addCard(Zone.BATTLEFIELD, playerB, "Guardian of the Guildpact");
        addCard(Zone.HAND, playerA, "Sun-Blessed Healer");
        addCard(Zone.GRAVEYARD, playerA, "Bind the Monster");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Sun-Blessed Healer");
        setChoice(playerA, true);
        addTarget(playerA, "Bind the Monster");
        addTarget(playerA, "Slippery Bogle");
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
        assertPermanentCount(playerA, "Bind the Monster", 1);
        assertAttachedTo(playerA, "Bind the Monster", "Slippery Bogle", true);
        assertTapped("Slippery Bogle", true);
        assertLife(playerA, 19);
    }
}
