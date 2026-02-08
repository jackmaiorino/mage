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

    float predictMulligan(float[] features);

    /**
     * Return raw two-headed mulligan scores: [Q_keep, Q_mull].
     */
    float[] predictMulliganScores(float[] features);

    void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes, byte[] landCounts, int batchSize);

    void saveMulliganModel();

    void saveModel(String path);

    String getDeviceInfo();

    Map<String, Integer> getMainModelTrainingStats();

    Map<String, Integer> getMulliganModelTrainingStats();

    Map<String, Integer> getHealthStats();

    void resetHealthStats();

    void recordGameResult(float lastValuePrediction, boolean won);

    Map<String, Object> getValueHeadMetrics();

    void shutdown();

    @Override
    default void close() {
        shutdown();
    }
}
