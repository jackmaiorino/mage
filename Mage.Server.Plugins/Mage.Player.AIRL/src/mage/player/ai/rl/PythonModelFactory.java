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

    public static boolean isSharedGpuMode() {
        return "shared_gpu".equalsIgnoreCase(EnvConfig.str("PY_SERVICE_MODE", "local"));
    }

    public static PythonModel getInstance() {
        PythonModel current = instance;
        if (current != null) {
            return current;
        }
        synchronized (LOCK) {
            current = instance;
            if (current == null) {
                current = isSharedGpuMode()
                        ? SharedGpuPythonModel.getInstance()
                        : PythonMLService.getInstance();
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
