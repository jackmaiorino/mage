package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Per-runner persistent MCTS tree state. Carries the root of the search tree
 * across priority calls so accumulated search is not thrown away between
 * decisions.
 *
 * Phase 2 (tree reuse): {@link #recordChoice} stores the index and action ref
 * the agent actually played at the current root. On the next priority call,
 * {@link #advance()} promotes the recorded child as the new root and discards
 * siblings. Callers MUST invoke recordChoice with the final action (after any
 * epsilon-greedy or full-turn-random override), not MCTS's preferred pick, or
 * the tree being advanced won't correspond to the real game state.
 *
 * Phase 3 (determinization consistency): when the opponent takes an observable
 * action, callers invoke {@link #recordOppObservation}. Before reusing a
 * subtree, {@link #hasPendingObservations} reports whether the tree's sampled
 * determinizations may have been invalidated. The current implementation is
 * conservative: any unobserved opp action triggers a reset so we don't
 * propagate Q/N accumulated under now-wrong assumptions.
 *
 * Phase 4 (transpositions): a state-fingerprint → node map lets MCTS merge
 * paths that reach the same game state, making the search graph a DAG with
 * shared Q/N accumulation at convergent states.
 */
public final class MCTSSession {

    /** Current search tree root, or null if no persistent tree is active. */
    private MCTSNode root;

    /** Action ref of our last chosen play at this root. Consulted by advance. */
    private Object lastChosenActionRef;

    /** Which root child index we took. Used as a fast-path hint for advance. */
    private Integer lastChosenChildIndex;

    /** Opp actions observed since last reset. When non-empty, any tree reuse
     *  is suspect because we sampled determinizations that might now contradict
     *  what we saw. */
    private final List<Object> observedOppActions = new ArrayList<>();

    /** Phase 4 transposition table: state fingerprint → existing tree node.
     *  When a walk reaches a node with a matching fingerprint, we splice the
     *  existing node in so its accumulated Q/N is shared. */
    private final Map<Long, MCTSNode> transpositions = new HashMap<>();

    /** Bumped when reuse is invalidated (state drifted or game changed). */
    private int invalidations = 0;

    public MCTSSession() {}

    public MCTSNode getRoot() { return root; }
    public void setRoot(MCTSNode r) { this.root = r; }

    /** Record the action ref the agent actually played at this root. Called by
     *  ComputerPlayerRL AFTER any post-MCTS override (epsilon-greedy, full-turn
     *  random, etc.) so the stored ref matches what went into the real game. */
    public void recordChoice(int rootChildIndex, Object actionRef) {
        this.lastChosenChildIndex = rootChildIndex;
        this.lastChosenActionRef = actionRef;
    }

    /** Advance the tree so the new root is the subtree of the child we most
     *  recently recorded via {@link #recordChoice}. Returns the new root
     *  (may be null if that child wasn't expanded). Resets the session when:
     *    - there's no current root,
     *    - opp observations have made the tree inconsistent,
     *    - no recorded choice is available to prune toward.
     *
     *  Callers are expected to invoke {@link #recordChoice} with the action
     *  actually played (after any post-MCTS override), not MCTS's preference. */
    public MCTSNode advance() {
        if (root == null) {
            return null;
        }
        if (hasPendingObservations()) {
            reset("opp observation invalidates tree reuse");
            return null;
        }
        if (lastChosenChildIndex == null) {
            reset("advance: no recorded choice");
            return null;
        }
        MCTSNode child = root.getChild(lastChosenChildIndex);
        promoteChild(child);
        return child;
    }

    private void promoteChild(MCTSNode child) {
        this.root = child;
        this.lastChosenChildIndex = null;
        this.lastChosenActionRef = null;
        // Transpositions remain valid — nodes in the retained subtree may
        // still be reached by fingerprint from the new search space.
    }

    /** Invalidate the persistent tree (e.g. on new game, shuffle, or observed
     *  mismatch). Next priority call will start from a fresh root. */
    public void reset(String reason) {
        this.root = null;
        this.lastChosenChildIndex = null;
        this.lastChosenActionRef = null;
        this.observedOppActions.clear();
        this.transpositions.clear();
        this.invalidations++;
    }

    public int invalidations() { return invalidations; }

    /** Record an observable opp action (a card they played, an attack, etc.).
     *  Stored until the next reset so tree reuse can be invalidated. */
    public void recordOppObservation(Object action) {
        observedOppActions.add(action);
    }

    /** True when opp has taken an action since the last reset, implying the
     *  sampled determinizations backing our accumulated tree may be wrong. */
    public boolean hasPendingObservations() {
        return !observedOppActions.isEmpty();
    }

    /** Total visit count at the current root (measure of how much accumulated
     *  search is carried forward). */
    public int rootTotalVisits() {
        return root == null ? 0 : root.totalVisits();
    }

    // ------------------------------------------------------------------
    // Phase 4: transposition table
    // ------------------------------------------------------------------

    /** Look up an existing node by state fingerprint. Returns null if no node
     *  has been registered at this fingerprint yet. */
    public MCTSNode getTransposition(long fingerprint) {
        if (fingerprint == 0L) return null;
        return transpositions.get(fingerprint);
    }

    /** Register a newly-created node under its fingerprint so future paths
     *  that reach the same state can splice it in and share Q/N. */
    public void registerTransposition(long fingerprint, MCTSNode node) {
        if (fingerprint == 0L || node == null) return;
        transpositions.put(fingerprint, node);
    }

    public int transpositionCount() { return transpositions.size(); }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

}
