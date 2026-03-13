package org.mage.test.AI.rl;

import mage.abilities.mana.ManaOptions;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.player.TestComputerPlayerRL;
import org.mage.test.player.TestPlayer;
import org.mage.test.serverside.base.CardTestPlayerBase;

import static org.mage.test.utils.ManaOptionsTestUtils.assertManaOptions;

/**
 * Local regressions for RL-specific mana availability filtering.
 */
public class RLManaOptionsTest extends CardTestPlayerBase {

    @Override
    protected TestPlayer createPlayer(String name, RangeOfInfluence rangeOfInfluence) {
        return new TestPlayer(new TestComputerPlayerRL(name, rangeOfInfluence, 6));
    }

    @Test
    public void testSaruliCaretakerDoesNotCountWithoutAnotherUntappedCreature() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Saruli Caretaker", 1);

        setStopAt(1, PhaseStep.PRECOMBAT_MAIN);
        execute();

        ManaOptions manaOptions = playerA.getAvailableManaTest(currentGame);
        Assert.assertEquals("Saruli should not count itself as the extra untapped creature", 1, manaOptions.size());
        assertManaOptions("{G}", manaOptions);
    }

    @Test
    public void testSaruliCaretakerCountsWhenAnotherUntappedCreatureExists() {
        addCard(Zone.BATTLEFIELD, playerA, "Forest", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Saruli Caretaker", 1);
        addCard(Zone.BATTLEFIELD, playerA, "Silvercoat Lion", 1);

        setStopAt(1, PhaseStep.PRECOMBAT_MAIN);
        execute();

        ManaOptions manaOptions = playerA.getAvailableManaTest(currentGame);
        Assert.assertEquals("Saruli should contribute mana when another untapped creature exists", 1, manaOptions.size());
        assertManaOptions("{G}{Any}", manaOptions);
    }
}
