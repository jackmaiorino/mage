package mage.player.ai.rl;

import java.util.List;
import java.util.Map;

public interface PythonModel extends AutoCloseable {

    PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey
    );

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

    void recordGameResult(float lastValuePrediction, boolean won);

    Map<String, Object> getValueHeadMetrics();

    void shutdown();

    @Override
    default void close() {
        shutdown();
    }
}

