package mage.player.ai.rl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Selects the Python model backend at runtime.
 *
 * The default path remains the local Py4J service. Shared GPU mode is opt-in
 * and only activates when PY_SERVICE_MODE=shared_gpu.
 */
public final class PythonModelFactory {

    private static volatile PythonModel instance;
    private static final Object LOCK = new Object();

    private PythonModelFactory() {
    }

    private static final String SERVICE_MODE = EnvConfig.str("PY_SERVICE_MODE", "local").trim().toLowerCase();

    public static boolean isSharedGpuMode() {
        return "shared_gpu".equals(SERVICE_MODE);
    }

    public static boolean isNoneMode() {
        return "none".equals(SERVICE_MODE);
    }

    public static boolean isOnnxMode() {
        return "onnx".equals(SERVICE_MODE) || "onnx_gpu".equals(SERVICE_MODE);
    }

    public static boolean isHybridMode() {
        return "hybrid".equals(SERVICE_MODE);
    }

    public static PythonModel getInstance() {
        PythonModel current = instance;
        if (current != null) {
            return current;
        }
        synchronized (LOCK) {
            current = instance;
            if (current == null) {
                if (isNoneMode()) {
                    current = NoOpPythonModel.getInstance();
                } else if (isHybridMode()) {
                    String profilesList = EnvConfig.str("TRAIN_PROFILES_LIST", "");
                    String[] profiles = profilesList.isEmpty() ? new String[0] : profilesList.split(",");
                    String trainingProfile = EnvConfig.str("MODEL_PROFILE", "").trim();

                    if (profiles.length > 1 && !trainingProfile.isEmpty()) {
                        // Multi-profile: per-profile ONNX sharing one CUDA arena
                        SharedGpuPythonModel gpu = SharedGpuPythonModel.getInstance();
                        String artifactsRoot = EnvConfig.str("RL_ARTIFACTS_ROOT",
                                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl");
                        // Create the training profile first to get shared env + opts
                        String trainModelsDir = artifactsRoot + "/profiles/" + trainingProfile + "/models";
                        OnnxInferenceModel trainOnnx = new OnnxInferenceModel(trainModelsDir);
                        trainOnnx.setTrainingDelegate(gpu);
                        Map<String, OnnxInferenceModel> profileModels = new LinkedHashMap<>();
                        profileModels.put(trainingProfile, trainOnnx);
                        // Other profiles share the training model's env + opts (one CUDA arena)
                        for (String profile : profiles) {
                            String p = profile.trim();
                            if (p.isEmpty() || p.equals(trainingProfile)) continue;
                            String pModelsDir = artifactsRoot + "/profiles/" + p + "/models";
                            OnnxInferenceModel onnx = new OnnxInferenceModel(
                                    pModelsDir, trainOnnx.getEnv(), trainOnnx.getSessionOpts());
                            onnx.setTrainingDelegate(gpu);
                            profileModels.put(p, onnx);
                        }
                        if (profileModels.containsKey(trainingProfile)
                                && profileModels.get(trainingProfile).isReady()) {
                            current = new MultiProfileOnnxRouter(profileModels, trainingProfile);
                        } else {
                            // ONNX not ready for training profile, close all and fall back
                            for (OnnxInferenceModel m : profileModels.values()) m.shutdown();
                            current = gpu;
                        }
                    } else {
                        // Single profile: existing behavior
                        String modelsDir = resolveModelsDir();
                        OnnxInferenceModel onnx = new OnnxInferenceModel(modelsDir);
                        if (onnx.isReady()) {
                            SharedGpuPythonModel gpu = SharedGpuPythonModel.getInstance();
                            onnx.setTrainingDelegate(gpu);
                            current = onnx;
                        } else {
                            current = SharedGpuPythonModel.getInstance();
                        }
                    }
                } else if (isOnnxMode()) {
                    // ONNX only (no training, inference-only eval)
                    String modelsDir = resolveModelsDir();
                    current = new OnnxInferenceModel(modelsDir);
                } else if (isSharedGpuMode()) {
                    current = SharedGpuPythonModel.getInstance();
                } else {
                    current = PythonMLService.getInstance();
                }
                instance = current;
            }
        }
        return current;
    }

    private static String resolveModelsDir() {
        String profile = EnvConfig.str("MODEL_PROFILE", "").trim();
        String artifactsRoot = EnvConfig.str("RL_ARTIFACTS_ROOT",
                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl");
        if (!profile.isEmpty()) {
            return artifactsRoot + "/profiles/" + profile + "/models";
        }
        return artifactsRoot + "/models";
    }

    static void resetForTests() {
        synchronized (LOCK) {
            if (instance != null) {
                instance.shutdown();
                instance = null;
            }
        }
    }
}
