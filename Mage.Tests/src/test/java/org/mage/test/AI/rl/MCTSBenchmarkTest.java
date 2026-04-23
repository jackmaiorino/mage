package org.mage.test.AI.rl;

import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import org.junit.Test;
import org.mage.test.player.TestComputerPlayerRL;
import org.mage.test.player.TestPlayer;
import org.mage.test.serverside.base.CardTestPlayerBase;

/**
 * MCTS speed profiling harness. Plays a simple game with two RL players,
 * driving {@code MultiPlyMCTS} via the production code path but with a random
 * (no-model) scoring stub — so we can profile the XMage engine + MCTS
 * coordination cost without needing a GPU or Python service.
 *
 * <p>Run with env vars (MCTS flags are read via static {@code System.getenv()}
 * blocks in {@code ComputerPlayerRL} and must be present at JVM start):
 * <pre>
 *   MCTS_TRAINING_ENABLE=1 MULTI_PLY_MCTS=1 \
 *     MCTS_ITERATIONS=8 MCTS_MAX_OUR_ACTIONS=1 \
 *     MCTS_ITER_TIMEOUT_MS=500 \
 *     mvn test -pl Mage.Tests -Dtest=MCTSBenchmarkTest
 * </pre>
 *
 * <p>Output: look for {@code [MCTS_STATS]} lines in stdout — these emit
 * every 100 MCTS searches and give per-iteration phase breakdown
 * (clone/setup/walk/eval ms).
 */
public class MCTSBenchmarkTest extends CardTestPlayerBase {

    @Override
    protected TestPlayer createPlayer(String name, RangeOfInfluence rangeOfInfluence) {
        return new TestPlayer(new TestComputerPlayerRL(name, rangeOfInfluence, 6));
    }

    /**
     * Play a short deterministic-ish game with two RL agents. Enough decisions
     * happen that MCTS_STATS will emit (~100 searches) if the env vars are on.
     */
    @Test
    public void benchmarkMCTSPhaseBreakdown() {
        // Simple creature-heavy battlefield — forces combat decisions and
        // activations, which are the expensive MCTS code paths.
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 5);
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Grizzly Bears", 2);
        addCard(Zone.BATTLEFIELD, playerA, "Silvercoat Lion", 2);
        addCard(Zone.HAND, playerA, "Lightning Bolt", 2);
        addCard(Zone.HAND, playerA, "Shock", 2);
        addCard(Zone.LIBRARY, playerA, "Mountain", 10);

        addCard(Zone.BATTLEFIELD, playerB, "Mountain", 5);
        addCard(Zone.BATTLEFIELD, playerB, "Plains", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Grizzly Bears", 2);
        addCard(Zone.BATTLEFIELD, playerB, "Silvercoat Lion", 2);
        addCard(Zone.HAND, playerB, "Lightning Bolt", 2);
        addCard(Zone.HAND, playerB, "Shock", 2);
        addCard(Zone.LIBRARY, playerB, "Mountain", 10);

        // Let the game run to turn 15 so ~hundreds of priority calls happen.
        // Each priority call with >=2 candidates is an MCTS trigger point.
        setStopAt(15, PhaseStep.END_TURN);
        execute();

        // Test "passes" unconditionally — we're profiling, not asserting.
        // Read stdout for [MCTS_STATS] output.
    }
}
