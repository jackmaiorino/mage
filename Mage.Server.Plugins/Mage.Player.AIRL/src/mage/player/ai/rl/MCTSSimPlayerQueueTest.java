package mage.player.ai.rl;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Smoke test for the queue-based coroutine protocol used by
 * {@link MCTSSimPlayer}. Does NOT require a live Game instance -- it
 * simulates the request/response flow directly with a fake engine thread.
 * <p>
 * Run with:
 *   mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java \
 *     -Dexec.mainClass=mage.player.ai.rl.MCTSSimPlayerQueueTest
 */
public final class MCTSSimPlayerQueueTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        testBasicRequestResponse();
        testControllerCanHaltPlayer();
        testDeadlineExpiryReleasesPlayer();

        System.out.println();
        System.out.println("Total: " + (passed + failed) + "  Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) System.exit(1);
    }

    /**
     * Engine thread calls priority() N times; controller replies with
     * action index. Verify all requests received, all responses applied,
     * no deadlock.
     */
    private static void testBasicRequestResponse() throws Exception {
        MCTSSimPlayer.Channel ch = new MCTSSimPlayer.Channel();
        AtomicInteger receivedDecisions = new AtomicInteger(0);

        // Fake engine thread: sends 3 requests, waits for 3 responses.
        Thread engine = new Thread(() -> {
            try {
                for (int i = 0; i < 3; i++) {
                    MCTSSimPlayer.DecisionRequest req = new MCTSSimPlayer.DecisionRequest(
                            java.util.UUID.randomUUID(),
                            java.util.Collections.emptyList(), i + 1);
                    ch.requestQueue.put(req);
                    MCTSSimPlayer.DecisionResponse resp = ch.responseQueue.poll(2, TimeUnit.SECONDS);
                    if (resp == null) return;  // timeout
                    receivedDecisions.incrementAndGet();
                }
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }, "fake-engine");
        engine.start();

        // Controller: read 3 requests, reply with increasing indices.
        for (int i = 0; i < 3; i++) {
            MCTSSimPlayer.DecisionRequest req = ch.requestQueue.poll(2, TimeUnit.SECONDS);
            if (req == null) { fail("basic", "controller got null request iter=" + i); engine.interrupt(); return; }
            if (req.turnNum != i + 1) { fail("basic", "turn mismatch iter=" + i + " got=" + req.turnNum); engine.interrupt(); return; }
            MCTSSimPlayer.sendResponse(ch.responseQueue, i);
        }
        engine.join(2000);
        if (receivedDecisions.get() != 3) {
            fail("basic", "expected 3 decisions applied, got " + receivedDecisions.get());
            return;
        }
        pass("basic-request-response-roundtrip");
    }

    /**
     * Controller sends HALT on request 1. Engine receives halt signal
     * and exits cleanly.
     */
    private static void testControllerCanHaltPlayer() throws Exception {
        MCTSSimPlayer.Channel ch = new MCTSSimPlayer.Channel();
        AtomicBoolean halted = new AtomicBoolean(false);

        Thread engine = new Thread(() -> {
            try {
                MCTSSimPlayer.DecisionRequest req = new MCTSSimPlayer.DecisionRequest(
                        java.util.UUID.randomUUID(),
                        java.util.Collections.emptyList(), 1);
                ch.requestQueue.put(req);
                MCTSSimPlayer.DecisionResponse resp = ch.responseQueue.poll(2, TimeUnit.SECONDS);
                if (resp != null && resp.actionIndex < 0) halted.set(true);
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }, "fake-engine-halt");
        engine.start();

        MCTSSimPlayer.DecisionRequest req = ch.requestQueue.poll(2, TimeUnit.SECONDS);
        if (req == null) { fail("halt", "no request"); return; }
        // Send halt signal.
        ch.responseQueue.put(MCTSSimPlayer.DecisionResponse.HALT);
        engine.join(2000);
        if (!halted.get()) {
            fail("halt", "engine did not receive halt signal");
            return;
        }
        pass("controller-halt-signal");
    }

    /**
     * If the controller never responds, the deadline cap on the player
     * side should eventually release the engine (no indefinite hang).
     * Simulated here with a manual deadline check.
     */
    private static void testDeadlineExpiryReleasesPlayer() throws Exception {
        MCTSSimPlayer.Channel ch = new MCTSSimPlayer.Channel();
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(200);
        AtomicBoolean gotNull = new AtomicBoolean(false);

        Thread fakePlayer = new Thread(() -> {
            try {
                long remaining = deadlineNanos - System.nanoTime();
                if (remaining <= 0) { gotNull.set(true); return; }
                MCTSSimPlayer.DecisionResponse resp = ch.responseQueue.poll(remaining, TimeUnit.NANOSECONDS);
                if (resp == null) gotNull.set(true);
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }, "fake-player-deadline");
        fakePlayer.start();
        fakePlayer.join(1000);
        if (!gotNull.get()) {
            fail("deadline", "player did not time out when controller was silent");
            return;
        }
        pass("deadline-expiry-releases-player");
    }

    private static void pass(String name) {
        System.out.println("  PASS  " + name);
        passed++;
    }

    private static void fail(String name, String why) {
        System.err.println("  FAIL  " + name + " : " + why);
        failed++;
    }
}
