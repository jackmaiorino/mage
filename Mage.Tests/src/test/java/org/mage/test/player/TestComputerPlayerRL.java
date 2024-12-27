package org.mage.test.player;

import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

import mage.constants.PhaseStep;
import mage.constants.Zone;
import mage.player.ai.rl.RLTrainer;

/**
 * Test class for the Reinforcement Learning AI player
 * 
 * @author [Your Name]
 */
public class TestComputerPlayerRL extends CardTestPlayerBase {

    @Test
    public void test_RLPlayer_BasicTraining() {
        System.err.println("Starting RL Player Basic Training Test");
        System.err.flush();

        // Change the current thread's name to "GAME" since timeout overrides it
        Thread.currentThread().setName("GAME");

        RLTrainer trainer = new RLTrainer();
        trainer.train();
        //trainer.eval(5);
        System.err.println("Test Complete");
        System.err.flush();
    }

    @Test
    public void test_RLPlayer_SimpleGameplay() {
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 2);
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        addCard(Zone.BATTLEFIELD, playerB, "Grizzly Bears");

        setStopAt(1, PhaseStep.END_TURN);
        execute();

        // Add assertions to verify AI behavior
        // assertLife(playerB, 17); // Example assertion
    }
}