package mage.player.ai.rl;

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

    static void resetForTests() {
        synchronized (LOCK) {
            if (instance != null) {
                instance.shutdown();
                instance = null;
            }
        }
    }
}
