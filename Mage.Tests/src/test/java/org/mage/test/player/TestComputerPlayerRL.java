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
        // logger.info("Disabling Logging for RLTrainer");
        // List<Logger> loggers = Collections.<Logger>list(LogManager.getCurrentLoggers());
        // loggers.add(LogManager.getRootLogger());
        // for ( Logger logger : loggers ) {
        //     logger.setLevel(Level.WARN);
        // }

        // logger.warn("Warning Test");
        RLTrainer trainer = new RLTrainer();
        trainer.train();

        //trainer.eval(5);
    }
}