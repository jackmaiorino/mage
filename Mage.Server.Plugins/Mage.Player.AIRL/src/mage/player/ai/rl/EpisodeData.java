package mage.player.ai.rl;

import java.io.Serializable;
import java.util.List;

/**
 * Lightweight serialisable container for a single self-play episode. Stored to
 * disk by GameWorker and consumed by Learner.
 */
public class EpisodeData implements Serializable {

    private static final long serialVersionUID = 1L;

    public final List<StateSequenceBuilder.TrainingData> trajectory;
    public final List<Double> discountedReturns;
    public final double finalReward;

    public EpisodeData(List<StateSequenceBuilder.TrainingData> trajectory,
            List<Double> discountedReturns,
            double finalReward) {
        this.trajectory = trajectory;
        this.discountedReturns = discountedReturns;
        this.finalReward = finalReward;
    }
}
