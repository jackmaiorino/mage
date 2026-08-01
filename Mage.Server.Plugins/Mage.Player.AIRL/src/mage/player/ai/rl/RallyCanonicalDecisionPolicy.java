package mage.player.ai.rl;

import java.io.Serializable;
import java.util.Map;

/**
 * Decision-policy boundary for the Rally canonical surface exposed by
 * {@code ComputerPlayerUniformMirror}.
 *
 * <p>Implementations receive canonical ranks rather than XMage objects. A
 * copy must preserve the exact decision state while remaining independently
 * mutable, because XMage copies players for game snapshots and simulations.</p>
 */
public interface RallyCanonicalDecisionPolicy extends Serializable {

    RallyCanonicalDecisionPolicy copy();

    int chooseNoncombat(String category, int canonicalLegalCount);

    int[] chooseNoncombatWithoutReplacement(
            String category, int canonicalLegalCount, int picks);

    boolean[] chooseAttackers(int canonicalEligibleCount);

    boolean[] chooseBlockers(int canonicalLegalBlockerCount);

    /**
     * @return {@code -1} for no block, otherwise the canonical blocker rank
     */
    int chooseBlocker(int canonicalLegalBlockerCount);

    long getPhysicalDecisionCount();

    long getPolicyActionSelections();

    long getPolicyLeafEvaluations();

    Map<String, Long> getPhysicalDecisionCategories();

    Map<String, Long> getOutcomeHistogram();
}
