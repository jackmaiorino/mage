package mage.player.ai.rl;

import org.apache.log4j.Logger;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Routes inference and training requests to per-profile ONNX models.
 * Actor threads carry the active profile in {@link ProfileContext}; meta-opponent
 * inference can also route by explicit policyKey.
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

    private OnnxInferenceModel currentProfileModel() {
        ProfileContext ctx = ProfileContext.current();
        if (ctx != null) {
            OnnxInferenceModel model = profileModels.get(ctx.profileName);
            if (model != null) {
                return model;
            }
        }
        return trainingModel;
    }

    /**
     * Resolve policyKey to a profile name, then look up the ONNX model.
     * Falls back to the current thread's profile for train-policy requests.
     */
    private OnnxInferenceModel resolveModel(String policyKey) {
        if (policyKey == null || policyKey.isEmpty() || "train".equals(policyKey)) {
            return currentProfileModel();
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
    // Training/control: delegate to the current profile's model
    // -----------------------------------------------------------------------

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        currentProfileModel().enqueueTraining(trainingData, rewards);
    }

    @Override
    public boolean awaitTrainingDrained(long timeoutMs) {
        return currentProfileModel().awaitTrainingDrained(timeoutMs);
    }

    @Override
    public void saveModel(String path) {
        currentProfileModel().saveModel(path);
    }

    @Override
    public String getDeviceInfo() {
        return "router[profiles=" + profileModels.size() + ",training=" + trainingProfile
                + "] " + trainingModel.getDeviceInfo();
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        return currentProfileModel().getMainModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        return currentProfileModel().getHealthStats();
    }

    @Override
    public void resetHealthStats() {
        currentProfileModel().resetHealthStats();
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        currentProfileModel().recordGameResult(lastValuePrediction, won);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        return currentProfileModel().getValueHeadMetrics();
    }

    @Override
    public float[] predictArchetype(StateSequenceBuilder.SequenceOutput state) {
        return currentProfileModel().predictArchetype(state);
    }

    @Override
    public float[] predictCardBelief(StateSequenceBuilder.SequenceOutput state) {
        return currentProfileModel().predictCardBelief(state);
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
