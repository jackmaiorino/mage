package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.game.Game;
import mage.player.ai.ComputerPlayerRL;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Optional control hook for resuming a copied engine state from a captured
 * decision and forcing exactly one branch choice.
 */
public interface EngineDecisionBranchController {

    default boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
        return false;
    }

    default boolean shouldBypassModelInference() {
        return true;
    }

    <T> Choice onDecision(DecisionContext<T> context);

    final class DecisionContext<T> {
        public final ComputerPlayerRL player;
        public final Game game;
        public final Ability source;
        public final StateSequenceBuilder.ActionType actionType;
        public final List<T> candidates;
        public final List<String> candidateTexts;
        public final List<String> candidateObjectIds;
        public final String compactState;
        public final String candidateHash;
        public final String stateHash;
        public final int candidateCount;
        public final int maxTargets;
        public final int minTargets;

        public DecisionContext(
                ComputerPlayerRL player,
                Game game,
                Ability source,
                StateSequenceBuilder.ActionType actionType,
                List<T> candidates,
                List<String> candidateTexts,
                List<String> candidateObjectIds,
                String compactState,
                int candidateCount,
                int maxTargets,
                int minTargets
        ) {
            this.player = player;
            this.game = game;
            this.source = source;
            this.actionType = actionType;
            this.candidates = candidates == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(candidates));
            this.candidateTexts = candidateTexts == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(candidateTexts));
            this.candidateObjectIds = candidateObjectIds == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(candidateObjectIds));
            this.compactState = compactState == null ? "" : compactState;
            this.candidateHash = LiveCheckpointRecorder.sha256(String.join("\n", this.candidateTexts));
            this.stateHash = LiveCheckpointRecorder.sha256(this.compactState);
            this.candidateCount = candidateCount;
            this.maxTargets = maxTargets;
            this.minTargets = minTargets;
        }
    }

    final class Choice {
        private static final Choice NONE = new Choice(Collections.emptyList(), false, "");

        private final List<Integer> indices;
        private final boolean terminateAfterDecision;
        private final String reason;

        private Choice(List<Integer> indices, boolean terminateAfterDecision, String reason) {
            this.indices = indices == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(indices));
            this.terminateAfterDecision = terminateAfterDecision;
            this.reason = reason == null ? "" : reason;
        }

        public static Choice none() {
            return NONE;
        }

        public static Choice choose(List<Integer> indices) {
            return new Choice(indices, false, "");
        }

        public static Choice chooseAndTerminate(List<Integer> indices, String reason) {
            return new Choice(indices, true, reason);
        }

        public boolean isNone() {
            return !terminateAfterDecision && indices.isEmpty();
        }

        public List<Integer> getIndices() {
            return indices;
        }

        public boolean isTerminateAfterDecision() {
            return terminateAfterDecision;
        }

        public String getReason() {
            return reason;
        }
    }

    final class BranchTerminated extends Error {
        private static final long serialVersionUID = 1L;

        private final String reason;

        public BranchTerminated(String reason) {
            super(reason == null ? "branch_terminated" : reason, null, false, false);
            this.reason = reason == null ? "branch_terminated" : reason;
        }

        public String getReason() {
            return reason;
        }
    }
}
