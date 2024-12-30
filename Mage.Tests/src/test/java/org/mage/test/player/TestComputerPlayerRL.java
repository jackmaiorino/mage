package org.mage.test.player;

import org.junit.Test;

import mage.player.ai.rl.RLTrainer;

/**
 * Test class for the Reinforcement Learning AI player
 * 
 * @author [Your Name]
 */
public final class TestComputerPlayerRL {

    @Test
    public void test_RLPlayer_BasicTraining() {

        RLTrainer trainer = new RLTrainer();
        trainer.train();

        //trainer.eval(5);
    }
}