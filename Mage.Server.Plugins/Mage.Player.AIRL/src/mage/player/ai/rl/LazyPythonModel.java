package mage.player.ai.rl;

import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Lightweight delegating model wrapper that delays Python bridge construction
 * until the first real model operation.
 */
final class LazyPythonModel implements PythonModel {

    private final Supplier<PythonModel> factory;
    private final Object lock = new Object();
    private volatile PythonModel delegate;

    LazyPythonModel(Supplier<PythonModel> factory) {
        if (factory == null) {
            throw new IllegalArgumentException("factory must not be null");
        }
        this.factory = factory;
    }

    private PythonModel model() {
        PythonModel current = delegate;
        if (current != null) {
            return current;
        }
        synchronized (lock) {
            current = delegate;
            if (current == null) {
                current = factory.get();
                delegate = current;
            }
        }
        return current;
    }

    PythonModel peekDelegate() {
        return delegate;
    }

    @Override
    public PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey,
            String headId,
            int pickIndex,
            int minTargets,
            int maxTargets
    ) {
        return model().scoreCandidates(
                state,
                candidateActionIds,
                candidateFeatures,
                candidateMask,
                policyKey,
                headId,
                pickIndex,
                minTargets,
                maxTargets
        );
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        model().enqueueTraining(trainingData, rewards);
    }

    @Override
    public float predictMulligan(float[] features) {
        return model().predictMulligan(features);
    }

    @Override
    public float[] predictMulliganScores(float[] features) {
        return model().predictMulliganScores(features);
    }

    @Override
    public void trainMulligan(
            byte[] features,
            byte[] decisions,
            byte[] outcomes,
            byte[] gameLengths,
            byte[] earlyLandScores,
            byte[] overrides,
            int batchSize
    ) {
        model().trainMulligan(features, decisions, outcomes, gameLengths, earlyLandScores, overrides, batchSize);
    }

    @Override
    public void saveMulliganModel() {
        model().saveMulliganModel();
    }

    @Override
    public void saveModel(String path) {
        model().saveModel(path);
    }

    @Override
    public String getDeviceInfo() {
        return model().getDeviceInfo();
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        return model().getMainModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getMulliganModelTrainingStats() {
        return model().getMulliganModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        return model().getHealthStats();
    }

    @Override
    public void resetHealthStats() {
        model().resetHealthStats();
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        model().recordGameResult(lastValuePrediction, won);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        return model().getValueHeadMetrics();
    }

    @Override
    public void shutdown() {
        PythonModel current = delegate;
        if (current != null) {
            current.shutdown();
            delegate = null;
        }
    }
}
