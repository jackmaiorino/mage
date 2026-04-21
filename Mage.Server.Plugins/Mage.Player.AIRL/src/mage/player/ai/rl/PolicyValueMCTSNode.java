package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.List;

/**
 * A single node in an AlphaZero-style MCTS tree.
 * <p>
 * Each node represents a game state reached after taking {@link #action}
 * from the parent. The root node has action=null.
 * <p>
 * Statistics tracked per PUCT:
 * <ul>
 *   <li>{@link #prior} P(a) — policy net's prior probability for this action</li>
 *   <li>{@link #visitCount} N — number of times this node has been visited</li>
 *   <li>{@link #totalValue} W — sum of backpropagated values</li>
 *   <li>{@link #meanValue()} Q = W/N — average value</li>
 * </ul>
 */
public final class PolicyValueMCTSNode {

    final PolicyValueMCTSNode parent;
    final int actionIndex;    // index into parent's candidate list (-1 for root)
    final float prior;        // P(a) from policy net

    int visitCount;           // N
    float totalValue;         // W (sum of backpropagated V)
    float valueEstimate;      // V(s) from value net at expansion time

    List<PolicyValueMCTSNode> children;  // null until expanded
    boolean expanded;
    boolean terminal;         // game ended at this node

    public PolicyValueMCTSNode(PolicyValueMCTSNode parent, int actionIndex, float prior) {
        this.parent = parent;
        this.actionIndex = actionIndex;
        this.prior = prior;
        this.visitCount = 0;
        this.totalValue = 0f;
        this.expanded = false;
        this.terminal = false;
    }

    public float meanValue() {
        return visitCount > 0 ? totalValue / visitCount : 0f;
    }

    public boolean isLeaf() {
        return !expanded;
    }

    /**
     * PUCT score for this node from the parent's perspective.
     * Higher = more attractive to explore.
     */
    public float puctScore(float cPuct) {
        int parentVisits = parent != null ? parent.visitCount : 1;
        float exploitation = meanValue();
        float exploration = cPuct * prior * (float) Math.sqrt(parentVisits) / (1 + visitCount);
        return exploitation + exploration;
    }

    /**
     * Select the child with the highest PUCT score.
     * Returns null if no children.
     */
    public PolicyValueMCTSNode selectChild(float cPuct) {
        if (children == null || children.isEmpty()) return null;
        PolicyValueMCTSNode best = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (PolicyValueMCTSNode child : children) {
            float score = child.puctScore(cPuct);
            if (score > bestScore) {
                bestScore = score;
                best = child;
            }
        }
        return best;
    }

    /**
     * Expand this node: create children with the given policy priors.
     * Each child corresponds to one candidate action.
     */
    public void expand(float[] priors) {
        children = new ArrayList<>(priors.length);
        for (int i = 0; i < priors.length; i++) {
            children.add(new PolicyValueMCTSNode(this, i, priors[i]));
        }
        expanded = true;
    }

    /**
     * Backpropagate a value estimate from this node up to the root.
     * Each ancestor accumulates the value and increments visit count.
     * <p>
     * Value is always from the ROOT PLAYER's perspective. If this node
     * represents an opponent decision, the value should be negated before
     * calling backpropagate (caller's responsibility).
     */
    public void backpropagate(float value) {
        PolicyValueMCTSNode node = this;
        while (node != null) {
            node.visitCount++;
            node.totalValue += value;
            node = node.parent;
        }
    }

    /**
     * After search completes, return the index of the child with the
     * most visits (robust selection). Returns -1 if no children.
     */
    public int bestActionByVisits() {
        if (children == null || children.isEmpty()) return -1;
        int bestIdx = -1;
        int bestVisits = -1;
        for (PolicyValueMCTSNode child : children) {
            if (child.visitCount > bestVisits) {
                bestVisits = child.visitCount;
                bestIdx = child.actionIndex;
            }
        }
        return bestIdx;
    }

    /**
     * Return the visit-count distribution over children (for logging / temperature sampling).
     */
    public int[] visitDistribution() {
        if (children == null) return new int[0];
        int[] visits = new int[children.size()];
        for (int i = 0; i < children.size(); i++) {
            visits[i] = children.get(i).visitCount;
        }
        return visits;
    }
}
