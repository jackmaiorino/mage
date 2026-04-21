package mage.player.ai.rl;

import mage.abilities.ActivatedAbility;
import mage.game.Game;
import mage.players.Player;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Day 2 foundation: runs a cloned simulation to completion with all
 * players driven by {@link MCTSSimPlayer} coroutines. Proves the
 * engine + coroutine plumbing works end-to-end before we layer tree
 * search / policy priors / value-net leaf eval on top.
 * <p>
 * This initial version picks actions uniformly at random at each
 * DecisionRequest. Functionally equivalent to {@code SimulatedPlayerMCTS},
 * but the action source is EXTERNAL (controller-driven). Future days
 * replace random with:
 * <ol>
 *   <li>PUCT tree selection (Day 3)</li>
 *   <li>Policy-net action priors (Day 3)</li>
 *   <li>Value-net leaf evaluation at depth cap (Day 3)</li>
 * </ol>
 */
public final class CoroutineRollout {

    public static class Result {
        public final boolean selfWon;
        public final boolean selfLost;
        public final int totalDecisions;
        public final long wallMs;
        public final String terminationReason;

        Result(boolean won, boolean lost, int decisions, long ms, String reason) {
            this.selfWon = won;
            this.selfLost = lost;
            this.totalDecisions = decisions;
            this.wallMs = ms;
            this.terminationReason = reason;
        }

        public float valueFor(UUID selfId) {
            if (selfWon) return 1.0f;
            if (selfLost) return -1.0f;
            return 0f;
        }
    }

    private CoroutineRollout() {}

    /**
     * Run a simulation game to completion (or hard deadline) by stepping
     * each player's priority() via the coroutine queue.
     *
     * @param liveGame  the real game to clone from
     * @param selfId    our player's UUID
     * @param timeoutMs hard wall-time cap for the entire rollout
     * @return outcome + diagnostics
     */
    public static Result runRandomRollout(Game liveGame, UUID selfId, long timeoutMs) {
        long started = System.currentTimeMillis();
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        Game sim = liveGame.createSimulationForAI();
        MCTSSimPlayer.ReplacementResult rep = MCTSSimPlayer.replaceAllPlayers(sim, sim, deadlineNanos);

        // Run sim.resume() on a worker thread. The engine will block on
        // every priority() call waiting for a controller response.
        AtomicReference<Throwable> simError = new AtomicReference<>();
        Thread simThread = new Thread(() -> {
            try {
                sim.resume();
            } catch (Throwable t) {
                simError.set(t);
            }
        }, "AI-SIM-MCTS-sim-engine");
        simThread.setDaemon(true);
        simThread.start();

        Random rng = new Random();
        int decisions = 0;
        String reason = "ended";

        // Controller loop: answer each player's decision request with a
        // random action index until the game ends or deadline fires.
        while (simThread.isAlive()) {
            long remaining = deadlineNanos - System.nanoTime();
            if (remaining <= 0) {
                reason = "deadline";
                break;
            }
            MCTSSimPlayer.DecisionRequest request = pollAnyChannel(rep, remaining);
            if (request == null) {
                // No request before deadline -- maybe game ended already.
                if (!simThread.isAlive()) { reason = "ended"; break; }
                reason = "channel-timeout";
                break;
            }
            decisions++;
            int idx = request.options == null || request.options.isEmpty()
                    ? 0 : rng.nextInt(request.options.size());
            MCTSSimPlayer.Channel ch = rep.channels.get(request.playerId);
            if (ch != null) {
                MCTSSimPlayer.sendResponse(ch.responseQueue, idx);
            }
        }

        // Signal all players to halt in case any is blocked.
        for (MCTSSimPlayer.Channel ch : rep.channels.values()) {
            try { ch.responseQueue.put(MCTSSimPlayer.DecisionResponse.HALT); } catch (InterruptedException ignored) {}
        }
        // Give the engine thread a brief grace period to exit cleanly.
        try {
            simThread.join(500);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        if (simThread.isAlive()) {
            simThread.interrupt();
        }

        Player me = sim.getPlayer(selfId);
        boolean won = me != null && me.hasWon();
        boolean lost = me != null && me.hasLost();
        if (simError.get() != null && !won && !lost) reason = "engine-error";
        return new Result(won, lost, decisions, System.currentTimeMillis() - started, reason);
    }

    /**
     * Run a truncated rollout: play up to {@code maxOurDecisions} of our
     * decisions (interleaved with however many opp decisions the engine
     * runs between), then halt the sim and evaluate the resulting state
     * with the value net. Falls back to win/loss if the game ends naturally
     * before the cap.
     * <p>
     * This is the leaf-evaluation path for Day 4's multi-ply MCTS: we apply
     * a candidate action on the clone, let the engine run through a few
     * real decision cycles (with random action sampling in between), then
     * query the value head on a stable priority state. Gives a more
     * principled signal than pure 1-ply snapshots.
     */
    private static final java.util.concurrent.atomic.AtomicInteger DEBUG_CALLS = new java.util.concurrent.atomic.AtomicInteger();
    private static final boolean DEBUG = "1".equals(System.getenv().getOrDefault("COROUTINE_DEBUG", "0"));

    public static Result runTruncatedRollout(Game liveGame, UUID selfId,
                                             int maxOurDecisions,
                                             PythonModel model,
                                             long timeoutMs) {
        int dbgId = DEBUG ? DEBUG_CALLS.incrementAndGet() : 0;
        long started = System.currentTimeMillis();
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        Game sim = liveGame.createSimulationForAI();
        MCTSSimPlayer.ReplacementResult rep = MCTSSimPlayer.replaceAllPlayers(sim, sim, deadlineNanos);
        if (DEBUG && dbgId <= 3) {
            System.out.println("[COROUTINE-DBG #" + dbgId + "] starting sim with "
                    + rep.players.size() + " MCTSSimPlayers, deadlineMs=" + timeoutMs);
        }

        AtomicReference<Throwable> simError = new AtomicReference<>();
        Thread simThread = new Thread(() -> {
            try {
                if (DEBUG && dbgId <= 3) System.out.println("[COROUTINE-DBG #" + dbgId + "] sim.resume() called");
                sim.resume();
                if (DEBUG && dbgId <= 3) System.out.println("[COROUTINE-DBG #" + dbgId + "] sim.resume() returned");
            } catch (Throwable t) {
                simError.set(t);
                if (DEBUG && dbgId <= 3) {
                    System.out.println("[COROUTINE-DBG #" + dbgId + "] sim.resume() threw: " + t.getClass().getSimpleName() + ": " + t.getMessage());
                }
            }
        }, "AI-SIM-MCTS-sim-truncated");
        simThread.setDaemon(true);
        simThread.start();

        Random rng = new Random();
        int ourDecisions = 0;
        int totalDecisions = 0;
        String reason = "ended";

        while (simThread.isAlive()) {
            long remaining = deadlineNanos - System.nanoTime();
            if (remaining <= 0) { reason = "deadline"; break; }
            MCTSSimPlayer.DecisionRequest request = pollAnyChannel(rep, remaining);
            if (request == null) {
                if (!simThread.isAlive()) { reason = "ended"; break; }
                reason = "channel-timeout"; break;
            }
            totalDecisions++;
            if (selfId.equals(request.playerId)) {
                ourDecisions++;
                if (ourDecisions > maxOurDecisions) {
                    reason = "depth-cap";
                    break;
                }
            }
            int idx = request.options == null || request.options.isEmpty()
                    ? 0 : rng.nextInt(request.options.size());
            MCTSSimPlayer.Channel ch = rep.channels.get(request.playerId);
            if (ch != null) {
                MCTSSimPlayer.sendResponse(ch.responseQueue, idx);
            }
        }

        for (MCTSSimPlayer.Channel ch : rep.channels.values()) {
            try { ch.responseQueue.put(MCTSSimPlayer.DecisionResponse.HALT); } catch (InterruptedException ignored) {}
        }
        try { simThread.join(500); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        if (simThread.isAlive()) simThread.interrupt();

        Player me = sim.getPlayer(selfId);
        boolean won = me != null && me.hasWon();
        boolean lost = me != null && me.hasLost();

        // If the game hit the depth cap (not terminal), fall back to value-net leaf eval.
        if (!won && !lost && model != null && "depth-cap".equals(reason)) {
            try {
                StateSequenceBuilder.SequenceOutput state = StateSequenceBuilder.buildBaseState(
                        sim,
                        sim.getPhase() != null ? sim.getPhase().getType()
                                : mage.constants.TurnPhase.BEGINNING,
                        StateSequenceBuilder.MAX_LEN);
                int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
                int[] dummyIds = new int[maxCand];
                float[][] dummyFeats = new float[maxCand][candDim];
                int[] dummyMask = new int[maxCand];
                dummyMask[0] = 1;
                PythonMLBatchManager.PredictionResult pred = model.scoreCandidates(
                        state, dummyIds, dummyFeats, dummyMask,
                        "mcts_leaf", "action", 0, 0, 0);
                float v = pred != null ? pred.valueScores : 0f;
                if (Float.isNaN(v) || Float.isInfinite(v)) v = 0f;
                v = Math.max(-1f, Math.min(1f, v));
                // Re-purpose the Result: selfWon/selfLost remain false, but we
                // return value via a synthetic result where neither flag is set.
                // Callers should prefer valueFor(selfId) which returns 0 and then
                // add back the value-net estimate.
                return new ResultWithValue(false, false, totalDecisions,
                        System.currentTimeMillis() - started, reason, v);
            } catch (Throwable t) {
                // fall through to Result with no value override
            }
        }
        return new Result(won, lost, totalDecisions,
                System.currentTimeMillis() - started, reason);
    }

    /** Extended result carrying a value-net leaf estimate (for depth-cap truncation). */
    public static final class ResultWithValue extends Result {
        public final float leafValue;
        ResultWithValue(boolean won, boolean lost, int decisions, long ms, String reason, float value) {
            super(won, lost, decisions, ms, reason);
            this.leafValue = value;
        }
        @Override
        public float valueFor(UUID selfId) {
            if (selfWon) return 1.0f;
            if (selfLost) return -1.0f;
            return leafValue;
        }
    }

    /**
     * Run a truncated rollout with POLICY-DRIVEN action selection.
     * <p>
     * Same shape as {@link #runTruncatedRollout} but the controller
     * samples each response from the policy net instead of uniformly
     * at random. This gives realistic rollout trajectories -- both sides
     * play like the actual trained agent -- so the terminal outcome
     * (or depth-cap leaf value) reflects real play quality, not coin-flip
     * noise.
     * <p>
     * Cost: adds one {@code model.scoreCandidates} call per decision in
     * the rollout. With {@code maxOurDecisions=3}, expect ~6-10 total
     * decisions per rollout (counting opponent turns), ~10ms per call.
     */
    public static Result runPolicyTruncatedRollout(Game liveGame, UUID selfId,
                                                   int maxOurDecisions,
                                                   PythonModel model,
                                                   long timeoutMs) {
        if (model == null) {
            // No model -> fall back to random rollout.
            return runTruncatedRollout(liveGame, selfId, maxOurDecisions, model, timeoutMs);
        }
        long started = System.currentTimeMillis();
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        Game sim = liveGame.createSimulationForAI();
        MCTSSimPlayer.ReplacementResult rep = MCTSSimPlayer.replaceAllPlayers(sim, sim, deadlineNanos);

        AtomicReference<Throwable> simError = new AtomicReference<>();
        Thread simThread = new Thread(() -> {
            try { sim.resume(); } catch (Throwable t) { simError.set(t); }
        }, "AI-SIM-MCTS-policy-rollout");
        simThread.setDaemon(true);
        simThread.start();

        Random rng = new Random();
        int ourDecisions = 0;
        int totalDecisions = 0;
        String reason = "ended";

        while (simThread.isAlive()) {
            long remaining = deadlineNanos - System.nanoTime();
            if (remaining <= 0) { reason = "deadline"; break; }
            MCTSSimPlayer.DecisionRequest request = pollAnyChannel(rep, remaining);
            if (request == null) {
                if (!simThread.isAlive()) { reason = "ended"; break; }
                reason = "channel-timeout"; break;
            }
            totalDecisions++;
            if (selfId.equals(request.playerId)) {
                ourDecisions++;
                if (ourDecisions > maxOurDecisions) { reason = "depth-cap"; break; }
            }
            int idx = policySampleAction(sim, model, request, rng);
            MCTSSimPlayer.Channel ch = rep.channels.get(request.playerId);
            if (ch != null) {
                MCTSSimPlayer.sendResponse(ch.responseQueue, idx);
            }
        }

        for (MCTSSimPlayer.Channel ch : rep.channels.values()) {
            try { ch.responseQueue.put(MCTSSimPlayer.DecisionResponse.HALT); } catch (InterruptedException ignored) {}
        }
        try { simThread.join(500); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        if (simThread.isAlive()) simThread.interrupt();

        Player me = sim.getPlayer(selfId);
        boolean won = me != null && me.hasWon();
        boolean lost = me != null && me.hasLost();

        if (!won && !lost && "depth-cap".equals(reason)) {
            // Value-net leaf eval at depth cap.
            try {
                StateSequenceBuilder.SequenceOutput state = StateSequenceBuilder.buildBaseState(
                        sim,
                        sim.getPhase() != null ? sim.getPhase().getType()
                                : mage.constants.TurnPhase.BEGINNING,
                        StateSequenceBuilder.MAX_LEN);
                int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
                int[] dummyIds = new int[maxCand];
                float[][] dummyFeats = new float[maxCand][candDim];
                int[] dummyMask = new int[maxCand];
                dummyMask[0] = 1;
                PythonMLBatchManager.PredictionResult pred = model.scoreCandidates(
                        state, dummyIds, dummyFeats, dummyMask,
                        "mcts_leaf_policy", "action", 0, 0, 0);
                float v = pred != null ? pred.valueScores : 0f;
                if (Float.isNaN(v) || Float.isInfinite(v)) v = 0f;
                v = Math.max(-1f, Math.min(1f, v));
                return new ResultWithValue(false, false, totalDecisions,
                        System.currentTimeMillis() - started, reason, v);
            } catch (Throwable t) {
                // Fall through.
            }
        }
        return new Result(won, lost, totalDecisions,
                System.currentTimeMillis() - started, reason);
    }

    /**
     * Sample an action index from the policy's distribution over the
     * request's options. Falls back to uniform random if anything in
     * the inference pipeline fails -- the rollout shouldn't abort.
     */
    private static int policySampleAction(Game sim, PythonModel model,
                                          MCTSSimPlayer.DecisionRequest request,
                                          Random rng) {
        int n = request.options == null ? 0 : request.options.size();
        if (n == 0) return 0;
        if (n == 1) return 0;
        try {
            StateSequenceBuilder.SequenceOutput state = StateSequenceBuilder.buildBaseState(
                    sim,
                    sim.getPhase() != null ? sim.getPhase().getType()
                            : mage.constants.TurnPhase.BEGINNING,
                    StateSequenceBuilder.MAX_LEN);
            int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            int slots = Math.min(n, maxCand);
            int[] ids = new int[maxCand];
            float[][] feats = new float[maxCand][candDim];
            int[] mask = new int[maxCand];
            for (int i = 0; i < slots; i++) {
                ActivatedAbility ab = request.options.get(i);
                ids[i] = abilityActionId(ab);
                mask[i] = 1;
            }
            PythonMLBatchManager.PredictionResult pred = model.scoreCandidates(
                    state, ids, feats, mask,
                    "mcts_rollout", "action", 0, 0, 0);
            if (pred == null || pred.policyScores == null) {
                return rng.nextInt(slots);
            }
            return sampleFromProbs(pred.policyScores, slots, rng);
        } catch (Throwable t) {
            return rng.nextInt(n);
        }
    }

    /**
     * Stable pseudo-ID for an ability, suitable for the action_id embedding.
     * Uses name-based hashing -- reproduces the encoding scheme used elsewhere
     * in the codebase when feature vectors aren't available.
     */
    private static int abilityActionId(mage.abilities.Ability ab) {
        if (ab == null) return 0;
        String key = ab.getClass().getSimpleName() + ":"
                + (ab.getSourceObjectIfItStillExists(null) != null
                        ? ab.getSourceObjectIfItStillExists(null).getName() : "src")
                + ":" + ab.toString();
        int h = key.hashCode();
        // Match the [1, TOKEN_ID_VOCAB-1] hashing pattern used for card IDs.
        final int VOCAB = 65536;
        return 1 + Math.floorMod(h, VOCAB - 1);
    }

    private static int sampleFromProbs(float[] probs, int numValid, Random rng) {
        float sum = 0;
        for (int i = 0; i < numValid; i++) {
            float p = probs[i];
            if (p > 0 && !Float.isNaN(p)) sum += p;
        }
        if (sum <= 0) return rng.nextInt(numValid);
        float r = rng.nextFloat() * sum;
        float c = 0;
        for (int i = 0; i < numValid; i++) {
            float p = probs[i];
            if (p > 0 && !Float.isNaN(p)) {
                c += p;
                if (r <= c) return i;
            }
        }
        return numValid - 1;
    }

    /**
     * Poll across all channels for the next pending decision request.
     * Round-robin is fine -- only one player holds priority at a time,
     * so at most one channel will have a request ready.
     */
    private static MCTSSimPlayer.DecisionRequest pollAnyChannel(
            MCTSSimPlayer.ReplacementResult rep, long remainingNanos) {
        long perPollNanos = Math.max(1_000_000L, remainingNanos / 50);  // 20ms-ish slices
        long deadline = System.nanoTime() + remainingNanos;
        while (System.nanoTime() < deadline) {
            for (Map.Entry<UUID, MCTSSimPlayer.Channel> e : rep.channels.entrySet()) {
                MCTSSimPlayer.DecisionRequest r;
                try {
                    r = e.getValue().requestQueue.poll(perPollNanos, TimeUnit.NANOSECONDS);
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    return null;
                }
                if (r != null) return r;
            }
        }
        return null;
    }
}
