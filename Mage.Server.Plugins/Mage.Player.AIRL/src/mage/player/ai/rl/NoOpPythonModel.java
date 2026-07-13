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

    // Deterministic trace generation (RL_SEED_RANDOM_UTIL=1): one seeded
    // stream instead of ThreadLocalRandom. Only valid with a single game
    // runner; parallel interleaving would break cross-run identity anyway.
    private static final boolean SEEDED = EnvConfig.bool("RL_SEED_RANDOM_UTIL", false);
    private final java.util.Random seededRng =
            new java.util.Random(EnvConfig.i32("RL_BASE_SEED", 5151) * 0x9E3779B9L);

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
        float[] policy = new float[candidateActionIds.length];
        if (SEEDED) {
            synchronized (seededRng) {
                for (int i = 0; i < policy.length; i++) {
                    policy[i] = seededRng.nextFloat();
                }
            }
        } else {
            ThreadLocalRandom rng = ThreadLocalRandom.current();
            for (int i = 0; i < policy.length; i++) {
                policy[i] = rng.nextFloat();
            }
        }
        return new PythonMLBatchManager.PredictionResult(
                policy,
                0.0f,
                "noop_random",
                "",
                "",
                -1,
                -1,
                Thread.currentThread().getName(),
                "random_scores",
                true,
                "NoOpPythonModel"
        );
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
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
