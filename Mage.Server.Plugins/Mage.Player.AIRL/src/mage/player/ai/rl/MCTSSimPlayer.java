package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.constants.Outcome;
import mage.constants.Zone;
import mage.game.Game;
import mage.players.Player;
import mage.player.ai.SimulatedPlayerMCTS;
import mage.target.Target;

import java.util.ArrayList;
import java.util.Collections;
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

    /** What kind of choice the engine is asking for. The controller dispatches
     *  policy-head priors / targets based on this. */
    public enum ChoiceType {
        /** priority() fired — choose which ActivatedAbility to cast/activate, or pass. */
        ACTIVATE_ABILITY,
        /** chooseTarget() fired — choose a target (UUID) from a list. */
        SELECT_TARGET,
        // Future: SELECT_MODE, SELECT_CHOICE, SELECT_TARGET_AMOUNT, etc.
    }

    /** Decision request sent from player → controller at every intercepted choice.
     *
     *  The shape of `options` depends on `choiceType`:
     *  - ACTIVATE_ABILITY → List<ActivatedAbility>
     *  - SELECT_TARGET    → List<UUID> (target IDs; index is the choice)
     */
    public static final class DecisionRequest {
        public final UUID playerId;
        public final ChoiceType choiceType;
        /** Candidate options. Generic so multiple choice types share the queue. */
        public final List<?> options;
        /** Turn/phase/step context for controller logging + PUCT state matching. */
        public final int turnNum;
        /** For SELECT_TARGET: the source ability that's asking for a target.
         *  For ACTIVATE_ABILITY: null.
         *  Used by controller to dispatch policy head and encode features. */
        public final Ability source;

        DecisionRequest(UUID playerId, ChoiceType choiceType, List<?> options,
                        int turnNum, Ability source) {
            this.playerId = playerId;
            this.choiceType = choiceType;
            this.options = options;
            this.turnNum = turnNum;
            this.source = source;
        }

        /** Convenience accessor for ACTIVATE_ABILITY requests. */
        @SuppressWarnings("unchecked")
        public List<ActivatedAbility> activateOptions() {
            if (choiceType != ChoiceType.ACTIVATE_ABILITY) {
                throw new IllegalStateException("not an ACTIVATE_ABILITY request: " + choiceType);
            }
            return (List<ActivatedAbility>) options;
        }

        /** Convenience accessor for SELECT_TARGET requests. */
        @SuppressWarnings("unchecked")
        public List<UUID> targetOptions() {
            if (choiceType != ChoiceType.SELECT_TARGET) {
                throw new IllegalStateException("not a SELECT_TARGET request: " + choiceType);
            }
            return (List<UUID>) options;
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
    // Phase 3b: optional inline controller. When set, priority() and
    // chooseTarget() bypass the queue protocol and call the controller
    // directly on the engine thread. Eliminates queue overhead and thread
    // coordination — the whole walk runs on the main thread.
    private volatile MCTSInlineController inlineController;

    /**
     * Synchronous controller callback for inline MCTS walks. Called from
     * priority()/chooseTarget() on whatever thread the engine is running on
     * (usually the main thread since Phase 3b removes simThread).
     */
    public interface MCTSInlineController {
        /** Return the chosen option index, or throw {@link WalkTerminated}
         *  to abort the engine loop immediately (walker reached leaf / depth). */
        int handleRequest(DecisionRequest req);
    }

    /**
     * Thrown from an {@link MCTSInlineController} to break out of the engine
     * loop. Extends {@link Error} (not Exception) so XMage's {@code catch
     * (Exception e)} blocks in {@code GameImpl.playPriority} do NOT catch it —
     * this is the ONLY way to unwind the engine loop cleanly when running
     * inline on the main thread. Caught by the runner at {@code sim.resume()}.
     * Fast unwind — no message, no stack.
     */
    public static final class WalkTerminated extends Error {
        public static final WalkTerminated INSTANCE = new WalkTerminated();
        private WalkTerminated() {
            super(null, null, false, false); // no stack trace — cheap throw
        }
    }

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

    /** Attach an inline controller. When set, priority()/chooseTarget()
     *  bypass the queue protocol. Pass null to revert to queue mode. */
    public void setInlineController(MCTSInlineController c) {
        this.inlineController = c;
    }

    /**
     * Override to use the fast in-place playable calculation (no nested game
     * clone) when we're inside an MCTS simulation. The default
     * MCTSPlayer.getPlayableAbilities → getPlayable() does a full game clone
     * per priority call to guarantee no side effects on the real game — but
     * we're already inside a sim clone, so state mutations inside
     * getPlayableInternal don't matter. This eliminates a large chunk of the
     * walk-phase cost in MCTS iterations.
     */
    @Override
    protected List<ActivatedAbility> getPlayableAbilities(Game game) {
        List<ActivatedAbility> playables = getPlayableFast(game, true, Zone.ALL, true);
        playables.add(new PassAbility());
        return playables;
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
        // Fast-path: if the only legal option is to pass, skip the queue
        // round-trip entirely. This eliminates thousands of ping-pongs per
        // iteration (most priority windows in a game have nothing to do).
        // The MCTS walker's depth counter only cares about ACTIVATE_ABILITY
        // prompts that have real branching — a mandatory pass is not a real
        // decision node in the tree.
        if (playables.size() == 1 && playables.get(0) instanceof PassAbility) {
            activateAbility(playables.get(0), game);
            return false;
        }
        int turnNum = 0;
        try { turnNum = game.getTurnNum(); } catch (Throwable ignored) {}

        DecisionRequest req = new DecisionRequest(
                this.getId(), ChoiceType.ACTIVATE_ABILITY, playables, turnNum, null);

        // Phase 3b inline path: if controller is attached, call it directly
        // on this (engine) thread. Controller may throw WalkTerminated to
        // abort the engine loop — that propagates up to the runner.
        MCTSInlineController inline = this.inlineController;
        int idx;
        if (inline != null) {
            idx = inline.handleRequest(req);
            if (idx < 0) {
                halted.set(true);
                return false;
            }
        } else {
            Integer idxBoxed = askController(req);
            if (idxBoxed == null) return false;
            idx = idxBoxed;
        }
        idx = Math.max(0, Math.min(idx, playables.size() - 1));
        ActivatedAbility chosen = playables.get(idx);
        activateAbility(chosen, game);
        return !(chosen instanceof PassAbility);
    }

    /** Target-selection intercept. Routes through the same queue as priority()
     *  so the controller (multi-ply MCTS) can branch on targets as tree edges.
     *
     *  Falls back to the parent's random picker if the controller halts, times
     *  out, or returns an invalid index.
     */
    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        if (halted.get() || System.nanoTime() >= deadlineNanos) {
            return super.chooseTarget(outcome, target, source, game);
        }
        java.util.Set<UUID> possible;
        try {
            possible = target.possibleTargets(this.getId(), source, game);
        } catch (Throwable t) {
            return super.chooseTarget(outcome, target, source, game);
        }
        if (possible == null || possible.isEmpty()) {
            return super.chooseTarget(outcome, target, source, game);
        }
        List<UUID> opts = new ArrayList<>(possible);
        int turnNum = 0;
        try { turnNum = game.getTurnNum(); } catch (Throwable ignored) {}

        DecisionRequest req = new DecisionRequest(
                this.getId(), ChoiceType.SELECT_TARGET,
                Collections.unmodifiableList(opts), turnNum, source);

        MCTSInlineController inline = this.inlineController;
        Integer idxBoxed;
        if (inline != null) {
            int i = inline.handleRequest(req);
            idxBoxed = i < 0 ? null : i;
        } else {
            idxBoxed = askController(req);
        }
        if (idxBoxed == null) {
            // Controller halted or timed out -- fall back to random.
            return super.chooseTarget(outcome, target, source, game);
        }
        int idx = Math.max(0, Math.min(idxBoxed, opts.size() - 1));
        try {
            target.add(opts.get(idx), game);
            return true;
        } catch (Throwable t) {
            return super.chooseTarget(outcome, target, source, game);
        }
    }

    /** Common send/wait pattern for any controller-routed decision.
     *  Returns the chosen option index, or null if halted/timed out. */
    private Integer askController(DecisionRequest req) {
        try {
            requestQueue.put(req);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            halted.set(true);
            return null;
        }
        long remaining = deadlineNanos - System.nanoTime();
        if (remaining <= 0) {
            halted.set(true);
            return null;
        }
        DecisionResponse resp;
        try {
            resp = responseQueue.poll(remaining, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            halted.set(true);
            return null;
        }
        if (resp == null || resp.actionIndex < 0) {
            halted.set(true);
            return null;
        }
        return resp.actionIndex;
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
     *
     * <p>Legacy shape — kept for compatibility. The new flow uses a single
     * shared request queue (across all players) so the walker can do a proper
     * blocking poll instead of round-robin polling with Thread.sleep(1).
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
     *
     * <p>One shared {@link #sharedRequestQueue} is used across all players
     * so the walker can do a single blocking poll on it rather than
     * spin-polling per-player queues.
     */
    public static class ReplacementResult {
        public final java.util.Map<UUID, Channel> channels = new java.util.HashMap<>();
        public final java.util.Map<UUID, MCTSSimPlayer> players = new java.util.HashMap<>();
        public final BlockingQueue<DecisionRequest> sharedRequestQueue =
                new LinkedBlockingQueue<>();
    }

    public static ReplacementResult replaceAllPlayers(Game liveGame, Game sim, long deadlineNanos) {
        ReplacementResult result = new ReplacementResult();
        for (Player oldPlayer : new ArrayList<>(sim.getState().getPlayers().values())) {
            UUID id = oldPlayer.getId();
            Channel ch = new Channel();
            // Point each player's request queue at the shared queue so all
            // player requests multiplex onto one stream the walker can poll.
            // The Channel.requestQueue field is a reference holder; we just
            // reassign it to the shared queue after construction. But
            // Channel.requestQueue is final, so we use a different approach:
            // bypass Channel.requestQueue and pass the shared queue directly
            // to the player.
            Player origPlayer = liveGame.getState().getPlayers().get(id).copy();
            MCTSSimPlayer newPlayer = new MCTSSimPlayer(oldPlayer,
                    result.sharedRequestQueue, ch.responseQueue, deadlineNanos);
            newPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(id, newPlayer);
            result.channels.put(id, ch);
            result.players.put(id, newPlayer);
        }
        return result;
    }
}
