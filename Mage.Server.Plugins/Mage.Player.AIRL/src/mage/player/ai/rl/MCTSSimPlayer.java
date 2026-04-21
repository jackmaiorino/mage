package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.game.Game;
import mage.players.Player;
import mage.player.ai.SimulatedPlayerMCTS;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An MCTSPlayer whose priority decisions are driven by an external MCTS
 * controller via blocking queues, not by internal random choice.
 * <p>
 * When the game engine calls {@link #priority(Game)}, the player:
 * <ol>
 *   <li>Enumerates legal actions and posts a {@link DecisionRequest} on
 *       {@link #requestQueue}.</li>
 *   <li>Blocks on {@link #responseQueue} waiting for the controller to reply.</li>
 *   <li>Applies the chosen action (or signals halt).</li>
 * </ol>
 * <p>
 * The game engine runs on its own thread (started by the controller via
 * {@code sim.resume()}). The controller runs on the calling thread and
 * communicates via the queues. This gives the controller tree-search-style
 * control over decisions without subclassing or hooking the engine directly.
 * <p>
 * Non-priority decisions (targets, attackers, blockers, choices) fall through
 * to {@link SimulatedPlayerMCTS}'s random defaults for now. A follow-up can
 * lift those into the queue protocol too.
 */
public final class MCTSSimPlayer extends SimulatedPlayerMCTS {

    /** Decision request sent from player → controller when priority() fires. */
    public static final class DecisionRequest {
        public final UUID playerId;
        /** Legal actions, as the player currently sees them. */
        public final List<ActivatedAbility> options;
        /** Turn/phase/step context for controller logging + PUCT state matching. */
        public final int turnNum;

        DecisionRequest(UUID playerId, List<ActivatedAbility> options, int turnNum) {
            this.playerId = playerId;
            this.options = options;
            this.turnNum = turnNum;
        }
    }

    /** Controller's reply to a decision request. */
    public static final class DecisionResponse {
        /** Index into the request's options list. Negative = halt. */
        public final int actionIndex;
        public static final DecisionResponse HALT = new DecisionResponse(-1);

        public DecisionResponse(int actionIndex) {
            this.actionIndex = actionIndex;
        }
    }

    // Queues owned by the controller; player references them for send/receive.
    private final BlockingQueue<DecisionRequest> requestQueue;
    private final BlockingQueue<DecisionResponse> responseQueue;
    private final AtomicBoolean halted = new AtomicBoolean(false);
    // Wall-time cap for the whole sim run so a deadlocked engine can't hang.
    private final long deadlineNanos;

    public MCTSSimPlayer(Player originalPlayer,
                         BlockingQueue<DecisionRequest> requestQueue,
                         BlockingQueue<DecisionResponse> responseQueue,
                         long deadlineNanos) {
        super(originalPlayer, true);
        this.requestQueue = requestQueue;
        this.responseQueue = responseQueue;
        this.deadlineNanos = deadlineNanos;
        // Engine defaults to fast-fail in test mode, which turns any priority
        // hiccup into an IllegalStateException that aborts the whole sim.
        // Disable it so random-ish MCTS play can stumble through edge cases.
        setFastFailInTestMode(false);
        setTestMode(false);
    }

    public boolean isHalted() {
        return halted.get();
    }

    public void halt() {
        halted.set(true);
    }

    @Override
    public boolean priority(Game game) {
        if (halted.get()) {
            return false;  // stop participating; engine should advance to turn end
        }
        if (System.nanoTime() >= deadlineNanos) {
            halted.set(true);
            return false;
        }
        List<ActivatedAbility> playables = getPlayableAbilities(game);
        if (playables == null || playables.isEmpty()) {
            return false;
        }
        int turnNum = 0;
        try { turnNum = game.getTurnNum(); } catch (Throwable ignored) {}

        DecisionRequest req = new DecisionRequest(this.getId(), playables, turnNum);
        try {
            requestQueue.put(req);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            halted.set(true);
            return false;
        }

        DecisionResponse resp;
        long remaining = deadlineNanos - System.nanoTime();
        if (remaining <= 0) {
            halted.set(true);
            return false;
        }
        try {
            resp = responseQueue.poll(remaining, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            halted.set(true);
            return false;
        }
        if (resp == null || resp.actionIndex < 0) {
            // Timeout or explicit halt.
            halted.set(true);
            return false;
        }
        int idx = Math.max(0, Math.min(resp.actionIndex, playables.size() - 1));
        ActivatedAbility chosen = playables.get(idx);
        activateAbility(chosen, game);
        return !(chosen instanceof PassAbility);
    }

    // -----------------------------------------------------------------------
    // Controller side: helpers for the MCTS search loop.
    // -----------------------------------------------------------------------

    /**
     * Simple controller skeleton: block until the next decision request,
     * return it, the caller decides what action to send.
     */
    public static DecisionRequest takeRequest(BlockingQueue<DecisionRequest> q,
                                              long remainingNanos) throws InterruptedException {
        if (remainingNanos <= 0) return null;
        return q.poll(remainingNanos, TimeUnit.NANOSECONDS);
    }

    public static void sendResponse(BlockingQueue<DecisionResponse> q, int actionIndex) {
        try {
            q.put(new DecisionResponse(actionIndex));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Default factory for a matched pair of queues that the controller
     * owns and the player references.
     */
    public static class Channel {
        public final BlockingQueue<DecisionRequest> requestQueue = new LinkedBlockingQueue<>();
        public final BlockingQueue<DecisionResponse> responseQueue = new LinkedBlockingQueue<>();
    }

    // -----------------------------------------------------------------------

    /**
     * Helper replaces ALL players in a simulation clone with MCTSSimPlayer
     * instances so the controller can drive both sides' decisions.
     * Returns the map of playerId -> (channel, player) so the controller
     * can identify which queue belongs to which player.
     */
    public static class ReplacementResult {
        public final java.util.Map<UUID, Channel> channels = new java.util.HashMap<>();
        public final java.util.Map<UUID, MCTSSimPlayer> players = new java.util.HashMap<>();
    }

    public static ReplacementResult replaceAllPlayers(Game liveGame, Game sim, long deadlineNanos) {
        ReplacementResult result = new ReplacementResult();
        for (Player oldPlayer : new ArrayList<>(sim.getState().getPlayers().values())) {
            UUID id = oldPlayer.getId();
            Channel ch = new Channel();
            Player origPlayer = liveGame.getState().getPlayers().get(id).copy();
            MCTSSimPlayer newPlayer = new MCTSSimPlayer(oldPlayer,
                    ch.requestQueue, ch.responseQueue, deadlineNanos);
            newPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(id, newPlayer);
            result.channels.put(id, ch);
            result.players.put(id, newPlayer);
        }
        return result;
    }
}
