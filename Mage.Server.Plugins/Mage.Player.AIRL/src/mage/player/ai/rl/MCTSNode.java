package mage.player.ai.rl;

import mage.game.Game;

import java.util.Arrays;

/**
 * One node in the multi-ply MCTS tree.
 *
 * Represents a specific decision point in the game (priority call,
 * target choice, etc.) — or a terminal/leaf state where no further
 * decision is made and V(s) is evaluated.
 *
 * Three kinds:
 *  - DECISION: engine is asking for a choice; has children (one per candidate)
 *  - LEAF:     compound action resolved; V(s) is cached in leafValue
 *  - TERMINAL: game ended at this point; terminalValue is ±1 or 0
 *
 * Does NOT own engine state. The tree is purely logical — each search
 * iteration re-clones the game and walks the tree by sending responses
 * through the coroutine queue. See {@link MultiPlyMCTS}.
 */
public final class MCTSNode {

    public enum Kind { DECISION, LEAF, TERMINAL }

    /** Set once when the node is first expanded. */
    private Kind kind = null;
    private MCTSSimPlayer.ChoiceType choiceType; // null for LEAF / TERMINAL
    private int numChildren = 0;
    private float[] priors;        // P(a) for each child, length numChildren
    private int[] visits;          // N(a)
    private float[] values;        // cumulative return through each child (Q = values / visits)
    private MCTSNode[] children;   // lazy; null entries mean "not yet visited"
    /** References to the candidates at this node. Interpretation depends on
     *  choiceType: ActivatedAbility for ACTIVATE_ABILITY, UUID for SELECT_TARGET. */
    private Object[] candidateRefs;
    private int totalVisits = 0;

    // LEAF
    private float leafValue = 0f;

    // TERMINAL
    private float terminalValue = 0f;

    // Phase 3: which determinization sample was used when this subtree was built.
    // Null means the node was built under no determinization assumption (e.g.,
    // our own pure action choices). Subtrees under opp-hand-dependent branches
    // should carry the det forward so observation-based pruning can check them.
    private DeterminizationSampler.Determinization sampledDet;

    // Phase 4: state fingerprint (optional). When non-null, the transposition
    // table can merge equivalent states via this key.
    private Long stateFingerprint;

    // Phase 3d: engine-state snapshot. Captured when walker first reaches this
    // node; subsequent iterations that walk through this node clone from the
    // snapshot instead of re-playing all actions from root. Drops to null when
    // the session inherits across decision calls or when the node is spliced
    // via transposition (stale state).
    private Game engineSnapshot;

    /** Create an empty node. Gets initialized later via one of the initializeAs* methods. */
    public MCTSNode() {}

    public Kind kind() { return kind; }
    public MCTSSimPlayer.ChoiceType choiceType() { return choiceType; }
    public int numChildren() { return numChildren; }
    public float[] priors() { return priors; }
    public int[] visits() { return visits; }
    public float[] values() { return values; }
    public Object[] candidateRefs() { return candidateRefs; }
    public int totalVisits() { return totalVisits; }
    public float leafValue() { return leafValue; }
    public float terminalValue() { return terminalValue; }

    public boolean isInitialized() { return kind != null; }

    public DeterminizationSampler.Determinization sampledDet() { return sampledDet; }
    public void setSampledDet(DeterminizationSampler.Determinization d) { this.sampledDet = d; }

    public Long stateFingerprint() { return stateFingerprint; }
    public void setStateFingerprint(Long fp) { this.stateFingerprint = fp; }

    public Game engineSnapshot() { return engineSnapshot; }
    public void setEngineSnapshot(Game snapshot) { this.engineSnapshot = snapshot; }
    public void clearEngineSnapshot() { this.engineSnapshot = null; }

    public MCTSNode getChild(int idx) {
        if (children == null) return null;
        if (idx < 0 || idx >= children.length) return null;
        return children[idx];
    }

    public void setChild(int idx, MCTSNode child) {
        if (children == null) {
            throw new IllegalStateException("cannot setChild on non-DECISION node");
        }
        children[idx] = child;
    }

    /** Initialize this node as a DECISION node with the given candidates + priors. */
    public void initializeAsDecision(MCTSSimPlayer.ChoiceType choiceType,
                                     Object[] candidateRefs,
                                     float[] priors) {
        if (this.kind != null) {
            throw new IllegalStateException("node already initialized as " + this.kind);
        }
        int n = candidateRefs.length;
        if (priors.length != n) {
            throw new IllegalArgumentException(
                    "priors length " + priors.length + " != candidates length " + n);
        }
        this.kind = Kind.DECISION;
        this.choiceType = choiceType;
        this.numChildren = n;
        this.candidateRefs = candidateRefs;
        this.priors = priors;
        this.visits = new int[n];
        this.values = new float[n];
        this.children = new MCTSNode[n];
    }

    /** Initialize this node as a LEAF with the given value-net estimate. */
    public void initializeAsLeaf(float v) {
        if (this.kind != null) {
            throw new IllegalStateException("node already initialized as " + this.kind);
        }
        this.kind = Kind.LEAF;
        this.leafValue = v;
    }

    /** Initialize this node as TERMINAL (game ended here). */
    public void initializeAsTerminal(float outcome) {
        if (this.kind != null) {
            throw new IllegalStateException("node already initialized as " + this.kind);
        }
        this.kind = Kind.TERMINAL;
        this.terminalValue = outcome;
    }

    /**
     * PUCT selection: returns index of the child with highest
     *   Q(a) + cPuct * P(a) * sqrt(N_parent + 1) / (1 + N(a))
     */
    public int selectChildPUCT(float cPuct) {
        if (kind != Kind.DECISION || numChildren == 0) {
            throw new IllegalStateException("PUCT on non-DECISION node or zero children");
        }
        int bestIdx = 0;
        float bestScore = Float.NEGATIVE_INFINITY;
        float sqrtTotal = (float) Math.sqrt(Math.max(1, totalVisits));
        for (int i = 0; i < numChildren; i++) {
            float q = visits[i] > 0 ? values[i] / visits[i] : 0f;
            float u = cPuct * priors[i] * sqrtTotal / (1 + visits[i]);
            float score = q + u;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /** Back up a value into the given child. Also increments this node's totalVisits.
     *  Synchronized because parallel rollout workers may back up to the same root
     *  concurrently (see {@link MultiPlyMCTS#runParallelSearch}). */
    public synchronized void backup(int childIdx, float value) {
        if (kind != Kind.DECISION) {
            throw new IllegalStateException("cannot backup on non-DECISION node");
        }
        visits[childIdx]++;
        values[childIdx] += value;
        totalVisits++;
    }

    /** Return index of most-visited child — the standard MCTS final action choice. */
    public int bestActionByVisits() {
        if (kind != Kind.DECISION) return 0;
        int best = 0;
        for (int i = 1; i < numChildren; i++) {
            if (visits[i] > visits[best]) best = i;
        }
        return best;
    }

    /** Return Q values (mean value per child). */
    public float[] meanValues() {
        if (kind != Kind.DECISION) return new float[0];
        float[] q = new float[numChildren];
        for (int i = 0; i < numChildren; i++) {
            q[i] = visits[i] > 0 ? values[i] / visits[i] : 0f;
        }
        return q;
    }

    @Override
    public String toString() {
        if (kind == null) return "MCTSNode(uninitialized)";
        if (kind == Kind.LEAF) return "MCTSNode(LEAF v=" + leafValue + ")";
        if (kind == Kind.TERMINAL) return "MCTSNode(TERMINAL v=" + terminalValue + ")";
        return "MCTSNode(DECISION " + choiceType + " n=" + numChildren
                + " totalVisits=" + totalVisits
                + " visits=" + Arrays.toString(visits) + ")";
    }
}
