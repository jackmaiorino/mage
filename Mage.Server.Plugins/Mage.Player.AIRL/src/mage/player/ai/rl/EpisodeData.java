package mage.player.ai.rl;

import java.io.Serializable;
import java.util.List;

/**
 * Lightweight serialisable container for a single self-play episode. Stored to
 * disk by GameWorker and consumed by Learner.
 * 
 * Note: Now stores immediate rewards instead of pre-computed discounted returns.
 * GAE (Generalized Advantage Estimation) will be computed in Python during training.
 */
public class EpisodeData implements Serializable {

    private static final long serialVersionUID = 2L;  // Incremented due to field change

    public final List<StateSequenceBuilder.TrainingData> trajectory;
    public final List<Double> rewards;  // Changed from discountedReturns to immediate rewards
    public final double finalReward;

    public EpisodeData(List<StateSequenceBuilder.TrainingData> trajectory,
            List<Double> rewards,
            double finalReward) {
        this.trajectory = trajectory;
        this.rewards = rewards;
        this.finalReward = finalReward;
    }
}
