package mage.player.ai.rl;

import java.util.List;
import java.util.Map;

public interface PythonModel extends AutoCloseable {

    /**
     * Candidate scoring with explicit policy head selection.
     *
     * headId: "action" | "target" | "card_select"
     */
    PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey,
            String headId,
            int pickIndex,
            int minTargets,
            int maxTargets
    );

    /**
     * Backward-compatible default: action head, no pick metadata.
     */
    default PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey
    ) {
        return scoreCandidates(state, candidateActionIds, candidateFeatures, candidateMask,
                policyKey, "action", 0, 0, 0);
    }

    void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards);

    void saveModel(String path);

    String getDeviceInfo();

    Map<String, Integer> getMainModelTrainingStats();

    Map<String, Integer> getHealthStats();

    void resetHealthStats();

    void recordGameResult(float lastValuePrediction, boolean won);

    Map<String, Object> getValueHeadMetrics();

    /**
     * Phase 2: predict opponent's deck archetype from public state.
     * Returns softmax probabilities over {Wildfire, Rally, Affinity, Elves},
     * or null if belief head is not available for this backend.
     */
    default float[] predictArchetype(StateSequenceBuilder.SequenceOutput state) {
        return null;
    }

    /**
     * Legacy mulligan scoring. The merged-mulligan refactor moved this into the
     * main model's target head, so bridges that don't expose a standalone
     * mulligan scorer return null. Kept here for backward compat with
     * MulliganModel until that class is fully retired.
     */
    default float[] predictMulliganScores(float[] features) {
        return null;
    }

    void shutdown();

    @Override
    default void close() {
        shutdown();
    }
}
