package mage.player.ai.rl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Stub model that returns random scores. Used with PY_SERVICE_MODE=none
 * to benchmark game engine throughput without any Python dependency.
 */
final class NoOpPythonModel implements PythonModel {

    private static final NoOpPythonModel INSTANCE = new NoOpPythonModel();

    static NoOpPythonModel getInstance() {
        return INSTANCE;
    }

    private NoOpPythonModel() {
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
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        float[] policy = new float[candidateActionIds.length];
        for (int i = 0; i < policy.length; i++) {
            policy[i] = rng.nextFloat();
        }
        return new PythonMLBatchManager.PredictionResult(policy, 0.0f);
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
    }

    @Override
    public float predictMulligan(float[] features) {
        return 0.5f;
    }

    @Override
    public float[] predictMulliganScores(float[] features) {
        return new float[]{0.5f, 0.5f};
    }

    @Override
    public void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes, byte[] gameLengths, byte[] earlyLandScores, byte[] overrides, int batchSize) {
    }

    @Override
    public void saveMulliganModel() {
    }

    @Override
    public void saveModel(String path) {
    }

    @Override
    public String getDeviceInfo() {
        return "none (no-op stub)";
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("train_steps", 0);
        m.put("train_samples", 0);
        return m;
    }

    @Override
    public Map<String, Integer> getMulliganModelTrainingStats() {
        return getMainModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        return Collections.singletonMap("gpu_oom_count", 0);
    }

    @Override
    public void resetHealthStats() {
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("accuracy", 0.0);
        m.put("avg_win", 0.0);
        m.put("avg_loss", 0.0);
        m.put("samples", 0);
        m.put("use_gae", false);
        return m;
    }

    @Override
    public void shutdown() {
    }
}
