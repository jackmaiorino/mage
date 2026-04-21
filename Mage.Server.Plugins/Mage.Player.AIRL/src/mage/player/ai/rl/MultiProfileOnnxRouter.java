package mage.player.ai.rl;

import org.apache.log4j.Logger;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Routes inference requests to per-profile ONNX models based on policyKey.
 * Training methods are delegated only to the training profile's model.
 */
public final class MultiProfileOnnxRouter implements PythonModel {

    private static final Logger logger = Logger.getLogger(MultiProfileOnnxRouter.class);

    private final Map<String, OnnxInferenceModel> profileModels;
    private final String trainingProfile;
    private final OnnxInferenceModel trainingModel;

    /**
     * @param profileModels   map from profile name to its ONNX model
     * @param trainingProfile the profile name that is actively training (receives training calls)
     */
    public MultiProfileOnnxRouter(Map<String, OnnxInferenceModel> profileModels, String trainingProfile) {
        this.profileModels = new LinkedHashMap<>(profileModels);
        this.trainingProfile = trainingProfile;
        this.trainingModel = profileModels.get(trainingProfile);
        if (trainingModel == null) {
            throw new IllegalArgumentException("Training profile '" + trainingProfile
                    + "' not found in profileModels: " + profileModels.keySet());
        }
        logger.info("MultiProfileOnnxRouter: " + profileModels.size() + " profiles, training=" + trainingProfile);
    }

    /**
     * Resolve policyKey to a profile name, then look up the ONNX model.
     * Falls back to trainingModel for unrecognized keys.
     */
    private OnnxInferenceModel resolveModel(String policyKey) {
        if (policyKey == null || policyKey.isEmpty() || "train".equals(policyKey)) {
            return trainingModel;
        }
        // "profile:Pauper-Rally" format
        if (policyKey.startsWith("profile:")) {
            String name = policyKey.substring("profile:".length()).trim();
            OnnxInferenceModel m = profileModels.get(name);
            return m != null ? m : trainingModel;
        }
        // Bare profile name lookup (used by meta opponent sampler)
        OnnxInferenceModel m = profileModels.get(policyKey);
        if (m != null) {
            return m;
        }
        // snap: keys and anything else fall through to training model
        return trainingModel;
    }

    // -----------------------------------------------------------------------
    // Inference: route to resolved profile model
    // -----------------------------------------------------------------------

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
            int maxTargets) {
        return resolveModel(policyKey).scoreCandidates(
                state, candidateActionIds, candidateFeatures, candidateMask,
                policyKey, headId, pickIndex, minTargets, maxTargets);
    }

    // -----------------------------------------------------------------------
    // Training: delegate to training model only
    // -----------------------------------------------------------------------

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        trainingModel.enqueueTraining(trainingData, rewards);
    }

    @Override
    public void saveModel(String path) {
        trainingModel.saveModel(path);
    }

    @Override
    public String getDeviceInfo() {
        return "router[profiles=" + profileModels.size() + ",training=" + trainingProfile
                + "] " + trainingModel.getDeviceInfo();
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        return trainingModel.getMainModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        return trainingModel.getHealthStats();
    }

    @Override
    public void resetHealthStats() {
        trainingModel.resetHealthStats();
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        trainingModel.recordGameResult(lastValuePrediction, won);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        return trainingModel.getValueHeadMetrics();
    }

    @Override
    public float[] predictArchetype(StateSequenceBuilder.SequenceOutput state) {
        // Route to the currently-active profile's ONNX inference model.
        // Pick the first profile (typical case: per-profile router has one active profile).
        for (Map.Entry<String, OnnxInferenceModel> entry : profileModels.entrySet()) {
            float[] probs = entry.getValue().predictArchetype(state);
            if (probs != null) return probs;
        }
        return null;
    }

    @Override
    public void shutdown() {
        for (Map.Entry<String, OnnxInferenceModel> entry : profileModels.entrySet()) {
            try {
                entry.getValue().shutdown();
            } catch (Exception e) {
                logger.error("Error shutting down ONNX model for profile " + entry.getKey(), e);
            }
        }
    }
}
