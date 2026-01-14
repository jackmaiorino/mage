package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.net.InetAddress;
import javax.net.ServerSocketFactory;
import javax.net.SocketFactory;

import py4j.ClientServer;
import py4j.Py4JException;

/**
 * Singleton bridge class to interface with Python ML implementation using Py4J.
 * Handles conversion between Java and Python tensors and manages the Py4J
 * gateway.
 */
public class PythonMLBridge implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(PythonMLBridge.class.getName());
    private static final boolean VERBOSE_SUBPROCESS_LOGS
            = "1".equals(System.getenv().getOrDefault("PY_BRIDGE_VERBOSE", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("PY_BRIDGE_VERBOSE", "0"));
    private static final int DEFAULT_PORT = 25334;
    private static final int PYTHON_STARTUP_WAIT_MS = 30000;
    private static final int PROCESS_KILL_WAIT_MS = 2000;
    private static final int MAX_INIT_RETRIES = 3;
    private static final int INIT_RETRY_DELAY_MS = 1000;
    private static final int MAX_CONNECTION_RETRIES = 15;
    private static final int CONNECTION_RETRY_DELAY_MS = 2000;

    // Singleton instance with volatile for thread safety
    private static volatile PythonMLBridge instance;
    private static final Object lock = new Object();

    private final String projectRoot = resolveRepoRoot(new File(System.getProperty("user.dir")).getAbsolutePath());
    // NOTE: On Windows, long paths can break pip installs. Default to a short venv path at repo root.
    private final String venvPath = resolveVenvPath(projectRoot);
    private final String pythonScriptPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py").getAbsolutePath();

    private ClientServer clientServer;
    private PythonEntryPoint entryPoint;
    private Process pythonProcess;
    private boolean isInitialized = false;

    /**
     * Private constructor to prevent direct instantiation
     */
    private PythonMLBridge() {
        try {
            configureLogging();
            cleanupExistingPythonProcesses();
            setupPaths();
            setupVirtualEnvironment();
            installDependencies();
            startPythonProcess();
            connectToPythonGateway();
            initializeModel();
            isInitialized = true;
        } catch (Exception e) {
            logger.severe("Failed to initialize Python ML Bridge: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Failed to initialize Python ML Bridge", e);
        }
    }

    /**
     * Keep bridge logs quiet by default. Override with PY_BRIDGE_LOG_LEVEL=INFO
     * (or FINE) when debugging.
     */
    private void configureLogging() {
        try {
            String lvl = System.getenv().getOrDefault("PY_BRIDGE_LOG_LEVEL", "WARNING").toUpperCase();
            Level level;
            try {
                level = Level.parse(lvl);
            } catch (Exception ignored) {
                level = Level.WARNING;
            }
            logger.setLevel(level);
        } catch (Exception ignored) {
            // ignore
        }
    }

    /**
     * Get the singleton instance of PythonMLBridge
     *
     * @return The singleton instance
     */
    public static PythonMLBridge getInstance() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new PythonMLBridge();
                }
            }
        }
        return instance;
    }

    /**
     * Reset the singleton instance (useful for testing)
     *
     * @return The new singleton instance
     */
    public static PythonMLBridge resetInstance() {
        synchronized (lock) {
            if (instance != null) {
                instance.shutdown();
            }
            instance = new PythonMLBridge();
            return instance;
        }
    }

    /**
     * Sets up the paths for the project root, virtual environment, and Python
     * script.
     */
    private void setupPaths() {
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Project root: " + projectRoot);
            logger.info("Virtual environment path: " + venvPath);
            logger.info("Python script path: " + pythonScriptPath);
        }
    }

    /**
     * Get the path to Python executables in the venv. Uses venv/Scripts on
     * Windows and venv/bin on Unix.
     */
    private String getPythonExecutablePath(String executable) {
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        String dir = isWindows ? "Scripts" : "bin";
        String exe = executable;
        if (isWindows && !executable.endsWith(".exe")) {
            exe = executable + ".exe";
        }
        return new File(venvPath, dir + File.separator + exe).getAbsolutePath();
    }

    private static String resolveRepoRoot(String startDir) {
        try {
            Path cur = Paths.get(startDir).toAbsolutePath();
            Path bestPom = null;
            for (int i = 0; i < 8 && cur != null; i++) {
                // prefer ".git" if present
                if (Files.exists(cur.resolve(".git")) || Files.exists(cur.resolve(".git").resolve("HEAD"))) {
                    return cur.toString();
                }
                // record the HIGHEST pom.xml seen, but keep walking upward to find .git
                if (Files.exists(cur.resolve("pom.xml")) && Files.isRegularFile(cur.resolve("pom.xml"))) {
                    bestPom = cur;
                }
                cur = cur.getParent();
            }
            if (bestPom != null) {
                return bestPom.toString();
            }
        } catch (Exception ignored) {
        }
        return startDir;
    }

    private static String resolveVenvPath(String repoRoot) {
        String override = System.getenv("MTG_VENV_PATH");
        if (override != null && !override.trim().isEmpty()) {
            return Paths.get(override).toAbsolutePath().toString();
        }
        return new File(repoRoot, ".mtgrl_venv").getAbsolutePath();
    }

    /**
     * Sets up the Python virtual environment if it doesn't exist.
     */
    private void setupVirtualEnvironment() throws Exception {
        File venvDir = new File(venvPath);
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Checking virtual environment at: " + venvPath);
        }

        if (!venvDir.exists()) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Virtual environment does not exist, creating it...");
            }
            ProcessBuilder venvBuilder = new ProcessBuilder("python", "-m", "venv", venvPath);
            venvBuilder.redirectErrorStream(true);
            Process venvProcess = venvBuilder.start();

            readProcessOutput(venvProcess, "[VENV]");

            int venvExitCode = venvProcess.waitFor();
            if (venvExitCode != 0) {
                throw new RuntimeException("Failed to create virtual environment, exit code: " + venvExitCode);
            }
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Virtual environment created successfully");
            }
        } else {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Virtual environment already exists at: " + venvPath);
            }
        }
    }

    /**
     * Installs required Python dependencies.
     */
    private void installDependencies() throws Exception {
        // Install py4j
        installPackage("py4j", "--upgrade");

        // Install PyTorch with CUDA support
        installPyTorch();

        // Install other packages
        String[] packages = {
            "numpy>=1.21.0",
            "transformers>=4.30.0",
            "tensorboard>=2.12.0"
        };

        for (String pkg : packages) {
            installPackage(pkg, "--upgrade");
        }

        if (logger.isLoggable(Level.INFO)) {
            logger.info("All requirements installed/updated successfully");
        }
    }

    /**
     * Checks if a Python package is installed.
     */
    private boolean isPackageInstalled(String packageName) throws Exception {
        try {
            // First try pip list
            String pipPath = getPythonExecutablePath("pip");
            ProcessBuilder checkBuilder = new ProcessBuilder(pipPath, "list", "--format=json");
            checkBuilder.redirectErrorStream(true);
            Process checkProcess = checkBuilder.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(checkProcess.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line);
                }
            }

            int checkExitCode = checkProcess.waitFor();
            if (checkExitCode == 0) {
                // Parse JSON output to check for package
                String jsonOutput = output.toString();
                return jsonOutput.contains("\"name\":\"" + packageName + "\"");
            }

            // If pip list fails, try a direct import check
            if (logger.isLoggable(Level.INFO)) {
                logger.info("pip list failed, trying direct import check for " + packageName);
            }
            String pythonPath = getPythonExecutablePath("python");
            ProcessBuilder importBuilder = new ProcessBuilder(pythonPath, "-c",
                    "try:\n"
                    + "    import " + packageName + "\n"
                    + "    print('INSTALLED')\n"
                    + "except ImportError:\n"
                    + "    print('NOT_INSTALLED')");

            importBuilder.redirectErrorStream(true);
            Process importProcess = importBuilder.start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(importProcess.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.equals("INSTALLED")) {
                        return true;
                    }
                }
            }

            return false;
        } catch (Exception e) {
            logger.warning("Error checking package installation: " + e.getMessage());
            // If we can't check, assume not installed to be safe
            return false;
        }
    }

    /**
     * Installs PyTorch with CUDA support.
     */
    private void installPyTorch() throws Exception {
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Checking PyTorch installation");
        }

        if (!isPackageInstalled("torch")) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("PyTorch not found, installing with CUDA support");
            }
            String pipPath = getPythonExecutablePath("pip");
            ProcessBuilder torchBuilder = new ProcessBuilder(pipPath, "install",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "torch>=2.2.0");
            torchBuilder.redirectErrorStream(true);
            Process torchProcess = torchBuilder.start();

            readProcessOutput(torchProcess, "[PIP]");

            int torchExitCode = torchProcess.waitFor();
            if (torchExitCode != 0) {
                throw new RuntimeException("Failed to install PyTorch, exit code: " + torchExitCode);
            }
            if (logger.isLoggable(Level.INFO)) {
                logger.info("PyTorch installed successfully");
            }
        } else {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("PyTorch already installed, skipping installation");
            }
        }
    }

    /**
     * Installs a single Python package.
     */
    private void installPackage(String pkgName, String... args) throws Exception {
        String basePackageName = pkgName.split(">=")[0];
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Checking package: " + basePackageName);
        }

        if (!isPackageInstalled(basePackageName)) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Installing package: " + pkgName);
            }
            String pipPath = getPythonExecutablePath("pip");
            List<String> command = new ArrayList<>();
            command.add(pipPath);
            command.add("install");
            for (String arg : args) {
                command.add(arg);
            }
            command.add(pkgName);

            ProcessBuilder packageBuilder = new ProcessBuilder(command);
            packageBuilder.redirectErrorStream(true);
            Process packageProcess = packageBuilder.start();

            readProcessOutput(packageProcess, "[PIP]");

            int packageExitCode = packageProcess.waitFor();
            if (packageExitCode != 0) {
                throw new RuntimeException("Failed to install package " + pkgName + ", exit code: " + packageExitCode);
            }
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Package " + pkgName + " installed successfully");
            }
        } else {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Package " + pkgName + " already installed, skipping installation");
            }
        }
    }

    /**
     * Starts the Python process running the Py4J entry point.
     */
    private void startPythonProcess() throws Exception {
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Starting Python process with script: " + pythonScriptPath);
        }

        // Verify the file exists
        File scriptFile = new File(pythonScriptPath);
        if (!scriptFile.exists()) {
            throw new RuntimeException("Python script not found at: " + pythonScriptPath);
        }

        // Use Python from virtual environment
        String pythonPath = getPythonExecutablePath("python");
        ProcessBuilder pb = new ProcessBuilder(pythonPath, pythonScriptPath);

        // Set model path environment variable to a specific model file
        String modelPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt").getAbsolutePath();
        File modelDir = new File(modelPath).getParentFile();
        if (!modelDir.exists()) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Creating models directory at: " + modelDir.getAbsolutePath());
            }
            if (!modelDir.mkdirs()) {
                throw new RuntimeException("Failed to create models directory at: " + modelDir.getAbsolutePath());
            }
        }

        pb.environment().put("MTG_MODEL_PATH", modelPath);
        // Keep Python console output quiet by default. Override via MTG_AI_LOG_LEVEL if desired.
        pb.environment().put("MTG_AI_LOG_LEVEL", System.getenv().getOrDefault("MTG_AI_LOG_LEVEL", "WARNING"));
        if (logger.isLoggable(Level.INFO)) {
            logger.info("Set MTG_MODEL_PATH to: " + modelPath);
        }

        pb.redirectErrorStream(true);
        pythonProcess = pb.start();

        // Read and log Python output in a separate thread
        Thread outputThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (VERBOSE_SUBPROCESS_LOGS && logger.isLoggable(Level.INFO)) {
                        logger.info("[PYTHON] " + line);
                    }
                }
            } catch (Exception e) {
                logger.severe("Error reading Python output: " + e.getMessage());
            }
        });
        outputThread.setDaemon(true);
        outputThread.start();

        // Wait for Python process to start up and verify it's running
        long startTime = System.currentTimeMillis();
        while (System.currentTimeMillis() - startTime < PYTHON_STARTUP_WAIT_MS) {
            try {
                // Check if process is still running
                int exitValue = pythonProcess.exitValue();
                throw new RuntimeException("Python process exited unexpectedly with code: " + exitValue);
            } catch (IllegalThreadStateException e) {
                // Process is still running, which is good
                Thread.sleep(100);
            }
        }

        // Verify process is still running after wait
        try {
            int exitValue = pythonProcess.exitValue();
            throw new RuntimeException("Python process exited unexpectedly with code: " + exitValue);
        } catch (IllegalThreadStateException e) {
            // Process is still running, which is good
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Python process started successfully");
            }
        }
    }

    /**
     * Connects to the Python Py4J gateway.
     */
    private void connectToPythonGateway() throws Exception {
        int retries = 0;
        Exception lastException = null;

        while (retries < MAX_CONNECTION_RETRIES) {
            try {
                // IMPORTANT (Windows): avoid binding a Java callback server on a fixed port (25333),
                // which frequently gets stuck "in use" after aborted runs.
                //
                // We don't rely on Python -> Java callbacks, so we disable auto-start of the Java server.
                clientServer = new ClientServer(
                        0,
                        InetAddress.getByName("127.0.0.1"),
                        DEFAULT_PORT,
                        InetAddress.getByName("127.0.0.1"),
                        0,
                        0,
                        ServerSocketFactory.getDefault(),
                        SocketFactory.getDefault(),
                        null,
                        false,
                        true
                );
                clientServer.startServer();
                entryPoint = (PythonEntryPoint) clientServer.getPythonServerEntryPoint(new Class[]{PythonEntryPoint.class});

                // Test the connection by calling a simple method
                entryPoint.initializeModel();

                logger.info("Connected to Python ML Bridge on port " + DEFAULT_PORT);
                return;
            } catch (Exception e) {
                lastException = e;
                logger.warning("Failed to connect to Python gateway (attempt " + (retries + 1) + "): " + e.getMessage());
                retries++;

                // Clean up failed connection
                if (clientServer != null) {
                    try {
                        clientServer.shutdown();
                    } catch (Exception ex) {
                        logger.warning("Error shutting down failed connection: " + ex.getMessage());
                    }
                    clientServer = null;
                }

                if (retries < MAX_CONNECTION_RETRIES) {
                    Thread.sleep(CONNECTION_RETRY_DELAY_MS);
                }
            }
        }

        throw new RuntimeException("Failed to connect to Python gateway after " + MAX_CONNECTION_RETRIES + " attempts", lastException);
    }

    /**
     * Reads and logs the output of a process.
     */
    private void readProcessOutput(Process process, String prefix) throws Exception {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Always drain stdout to avoid blocking, but only print when explicitly requested.
                if (VERBOSE_SUBPROCESS_LOGS && logger.isLoggable(Level.INFO)) {
                    logger.info(prefix + " " + line);
                }
            }
        }
    }

    /**
     * Cleans up any existing Python processes in Linux container.
     */
    private void cleanupExistingPythonProcesses() {
        try {
            // On Windows we don't have pkill; skip cleanup.
            if (System.getProperty("os.name").toLowerCase().contains("win")) {
                // Kill only processes running our script (best-effort).
                // This avoids "callback server port already in use" failures.
                String cmd =
                        // 1) Kill lingering Python entrypoints
                        "Get-CimInstance Win32_Process | "
                                + "Where-Object { $_.CommandLine -like '*py4j_entry_point.py*' } | "
                                + "ForEach-Object { Stop-Process -Id $_.ProcessId -Force } ; "
                                // 2) Free Py4J callback/gateway ports if a previous Java run is stuck
                                + "$ports = @(25333,25334); "
                                + "foreach ($p in $ports) { "
                                + "  $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; "
                                + "  foreach ($c in $conns) { "
                                + "    $pid = $c.OwningProcess; "
                                + "    $proc = Get-CimInstance Win32_Process -Filter \"ProcessId=$pid\" -ErrorAction SilentlyContinue; "
                                + "    if ($proc -and $proc.Name -eq 'java.exe') { "
                                + "      Stop-Process -Id $pid -Force "
                                + "    } "
                                + "  } "
                                + "} ";
                ProcessBuilder pb = new ProcessBuilder("powershell", "-NoProfile", "-Command", cmd);
                pb.redirectErrorStream(true);
                Process process = pb.start();
                readProcessOutput(process, "[WIN-KILL]");
                process.waitFor();
                Thread.sleep(PROCESS_KILL_WAIT_MS);
                return;
            }
            // Use pkill to kill Python processes running our script
            String scriptName = "py4j_entry_point.py";
            ProcessBuilder pb = new ProcessBuilder("pkill", "-f", scriptName);
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                logger.info("Successfully killed Python processes with script: " + scriptName);
            } else {
                logger.info("No Python processes found running script: " + scriptName);
            }
            Thread.sleep(PROCESS_KILL_WAIT_MS);
        } catch (Exception e) {
            logger.info("Error with pkill (normal if no processes found): " + e.getMessage());
        }
    }

    /**
     * Initialize the Python ML model
     */
    public void initialize() {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }
        initializeModel();
    }

    /**
     * Internal method to initialize the model
     */
    private void initializeModel() {
        int retries = 0;
        Exception lastException = null;

        while (retries < MAX_INIT_RETRIES) {
            try {
                entryPoint.initializeModel();
                logger.info("Python ML model initialized successfully");
                return;
            } catch (Exception e) {
                lastException = e;
                logger.warning("Failed to initialize Python ML model (attempt " + (retries + 1) + "): " + e.getMessage());
                retries++;
                if (retries < MAX_INIT_RETRIES) {
                    try {
                        Thread.sleep(INIT_RETRY_DELAY_MS);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while waiting to retry initialization", ie);
                    }
                }
            }
        }

        logger.severe("Failed to initialize Python ML model after " + MAX_INIT_RETRIES + " attempts");
        throw new RuntimeException("Failed to initialize Python ML model", lastException);
    }

    /**
     * Candidate-based policy/value prediction.
     */
    public PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask) {
        CompletableFuture<PythonMLBatchManager.PredictionResult> future
                = PythonMLBatchManager.getInstance(entryPoint).scoreCandidates(state, candidateActionIds, candidateFeatures, candidateMask);
        try {
            return future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Candidate scoring failed", e);
        }
    }

    /**
     * Legacy fixed-action prediction is no longer supported. Use
     * {@link #scoreCandidates(StateSequenceBuilder.SequenceOutput, int[], float[][], int[])}.
     */
    @Deprecated
    public float[] predict(StateSequenceBuilder.SequenceOutput state, int validActions) {
        throw new UnsupportedOperationException("Use candidate-based scoring instead of fixed-action prediction");
    }

    @Deprecated
    public PythonMLBatchManager.PredictionResult predictComplete(StateSequenceBuilder.SequenceOutput state, int validActions) {
        throw new UnsupportedOperationException("Use candidate-based scoring instead of fixed-action prediction");
    }

    /**
     * Train the model with a batch of states and discounted returns.
     */
    public void train(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> discountedReturns) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            PythonMLBatchManager.getInstance(entryPoint).train(trainingData, discountedReturns).get();
        } catch (Exception e) {
            logger.severe("Error during training: " + e.getMessage());
            throw new RuntimeException("Failed to train Python model", e);
        }
    }

    /**
     * Save the current model state
     */
    public void saveModel(String path) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            entryPoint.saveModel(path);
        } catch (Py4JException e) {
            logger.severe("Py4J error saving model: " + e.getMessage());
            throw new RuntimeException("Failed to save Python model", e);
        } catch (Exception e) {
            logger.severe("Error saving model: " + e.getMessage());
            throw new RuntimeException("Failed to save Python model", e);
        }
    }

    /**
     * Load a saved model state
     */
    public void loadModel(String path) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            entryPoint.loadModel(path);
        } catch (Py4JException e) {
            logger.severe("Py4J error loading model: " + e.getMessage());
            throw new RuntimeException("Failed to load Python model", e);
        } catch (Exception e) {
            logger.severe("Error loading model: " + e.getMessage());
            throw new RuntimeException("Failed to load Python model", e);
        }
    }

    /**
     * Clean up resources
     */
    public void shutdown() {
        if (clientServer != null) {
            try {
                clientServer.shutdown();
                isInitialized = false;
                logger.info("Python ML Bridge shutdown complete");
            } catch (Exception e) {
                logger.severe("Error during shutdown: " + e.getMessage());
            }
        }
        if (pythonProcess != null) {
            try {
                // First try graceful shutdown
                pythonProcess.destroy();
                // Wait a bit for the process to terminate
                if (!pythonProcess.waitFor(5, TimeUnit.SECONDS)) {
                    // If it's still running, force kill it
                    pythonProcess.destroyForcibly();
                    logger.info("Python process force terminated");
                } else {
                    logger.info("Python process terminated gracefully");
                }
            } catch (InterruptedException e) {
                logger.severe("Error waiting for Python process to terminate: " + e.getMessage());
                // Force kill if interrupted
                pythonProcess.destroyForcibly();
            }
        }
    }

    /**
     * Get optimal batch size based on available GPU memory
     *
     * @return optimal number of samples per batch for current GPU
     */
    public int getOptimalBatchSize() {
        if (!isInitialized || entryPoint == null) {
            logger.warning("Python bridge not initialized, using fallback batch size");
            return 10000; // Safe fallback
        }

        try {
            int optimalSize = entryPoint.getOptimalBatchSize();
            logger.info("GPU-optimized batch size calculated: " + optimalSize + " samples");
            return optimalSize;
        } catch (Exception e) {
            logger.warning("Failed to get optimal batch size from Python: " + e.getMessage());
            return 10000; // Safe fallback
        }
    }

    /**
     * One-line diagnostic for CUDA/device placement from Python.
     */
    public String getDeviceInfo() {
        if (!isInitialized || entryPoint == null) {
            return "device=unknown cuda_available=false gpu= optimal_batch=null";
        }
        try {
            return entryPoint.getDeviceInfo();
        } catch (Exception e) {
            return "device_info_error=" + e.getMessage();
        }
    }

    @Override
    protected void finalize() throws Throwable {
        shutdown();
        super.finalize();
    }

    @Override
    public void close() {
        shutdown();
    }
}
