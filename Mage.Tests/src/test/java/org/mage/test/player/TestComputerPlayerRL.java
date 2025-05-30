package org.mage.test.player;

import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

import mage.player.ai.rl.RLTrainer;

/**
 * Test class for the Reinforcement Learning AI player
 * 
 * @author [Your Name]
 */
//TODO: This extension should maybe be something else, something that doesn't load all 70k cards
public class TestComputerPlayerRL extends CardTestPlayerBase
 {
    @Test
    public void test_RLPlayer_BasicTraining() {
        RLTrainer trainer = new RLTrainer();
        int trainingsRun = 0;
        try {
            trainer.train();
            trainingsRun++;
        }finally {
            System.out.println("Trainings run: " + trainingsRun);
        }
    }

    @Test
    public void test_RLPlayer_Eval() {
        RLTrainer trainer = new RLTrainer();
        trainer.eval(3);
    }
}