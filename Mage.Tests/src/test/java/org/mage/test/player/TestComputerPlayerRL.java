package org.mage.test.player;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.junit.Test;

import mage.player.ai.rl.RLTrainer;
import org.mage.test.serverside.base.CardTestPlayerBase;

import java.util.Collections;
import java.util.List;

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
        logger.info("Disabling Logging for RLTrainer");
        List<Logger> loggers = Collections.<Logger>list(LogManager.getCurrentLoggers());
        loggers.add(LogManager.getRootLogger());
        for ( Logger logger : loggers ) {
            logger.setLevel(Level.WARN);
        }

        logger.warn("Warning Test");
        RLTrainer trainer = new RLTrainer();
        trainer.train();

        //trainer.eval(5);
    }
}