package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.concurrent.TimeUnit;
import java.io.DataInputStream;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import py4j.ClientServer;
import py4j.GatewayServer;
import py4j.Py4JException;

/**
 * Singleton bridge class to interface with Python ML implementation using Py4J.
 * Handles conversion between Java and Python tensors and manages the Py4J
 * gateway.
 */
public class PythonMLBridge {

    private static final Logger logger = Logger.getLogger(PythonMLBridge.class.getName());
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

    private final String projectRoot = new File(new File(System.getProperty("user.dir")).getParentFile().getParentFile(), "mage").getAbsolutePath();
    private final String venvPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/venv").getAbsolutePath();
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
        logger.info("Project root: " + projectRoot);
        logger.info("Virtual environment path: " + venvPath);
        logger.info("Python script path: " + pythonScriptPath);
    }

    /**
     * Sets up the Python virtual environment if it doesn't exist.
     */
    private void setupVirtualEnvironment() throws Exception {
        File venvDir = new File(venvPath);
        logger.info("Checking virtual environment at: " + venvPath);

        if (!venvDir.exists()) {
            logger.info("Virtual environment does not exist, creating it...");
            ProcessBuilder venvBuilder = new ProcessBuilder("python", "-m", "venv", venvPath);
            venvBuilder.redirectErrorStream(true);
            Process venvProcess = venvBuilder.start();

            readProcessOutput(venvProcess, "[VENV]");

            int venvExitCode = venvProcess.waitFor();
            if (venvExitCode != 0) {
                throw new RuntimeException("Failed to create virtual environment, exit code: " + venvExitCode);
            }
            logger.info("Virtual environment created successfully");
        } else {
            logger.info("Virtual environment already exists at: " + venvPath);
        }
    }

    /**
     * Installs required Python dependencies.
     */
    private void installDependencies() throws Exception {
        String pipPath = new File(venvPath, "Scripts/pip").getAbsolutePath();

        // Install py4j
        installPackage(pipPath, "py4j", "--upgrade");

        // Install PyTorch with CUDA support
        installPyTorch(pipPath);

        // Install other packages
        String[] packages = {
            "numpy>=1.21.0",
            "transformers>=4.30.0",
            "tensorboard>=2.12.0"
        };

        for (String pkg : packages) {
            installPackage(pipPath, pkg, "--upgrade");
        }

        logger.info("All requirements installed/updated successfully");
    }

    /**
     * Checks if a Python package is installed.
     */
    private boolean isPackageInstalled(String pipPath, String packageName) throws Exception {
        try {
            // First try pip list
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
            logger.info("pip list failed, trying direct import check for " + packageName);
            String pythonPath = new File(venvPath, "Scripts/python").getAbsolutePath();
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
    private void installPyTorch(String pipPath) throws Exception {
        logger.info("Checking PyTorch installation");

        if (!isPackageInstalled(pipPath, "torch")) {
            logger.info("PyTorch not found, installing with CUDA support");
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
            logger.info("PyTorch installed successfully");
        } else {
            logger.info("PyTorch already installed, skipping installation");
        }
    }

    /**
     * Installs a single Python package.
     */
    private void installPackage(String pipPath, String pkgName, String... args) throws Exception {
        String basePackageName = pkgName.split(">=")[0];
        logger.info("Checking package: " + basePackageName);

        if (!isPackageInstalled(pipPath, basePackageName)) {
            logger.info("Installing package: " + pkgName);
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
            logger.info("Package " + pkgName + " installed successfully");
        } else {
            logger.info("Package " + pkgName + " already installed, skipping installation");
        }
    }

    /**
     * Starts the Python process running the Py4J entry point.
     */
    private void startPythonProcess() throws Exception {
        logger.info("Starting Python process with script: " + pythonScriptPath);

        // Verify the file exists
        File scriptFile = new File(pythonScriptPath);
        if (!scriptFile.exists()) {
            throw new RuntimeException("Python script not found at: " + pythonScriptPath);
        }

        // Use Python from virtual environment
        String pythonPath = new File(venvPath, "Scripts/python").getAbsolutePath();
        ProcessBuilder pb = new ProcessBuilder(pythonPath, pythonScriptPath);

        // Set model path environment variable to a specific model file
        String modelPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt").getAbsolutePath();
        File modelDir = new File(modelPath).getParentFile();
        if (!modelDir.exists()) {
            logger.info("Creating models directory at: " + modelDir.getAbsolutePath());
            if (!modelDir.mkdirs()) {
                throw new RuntimeException("Failed to create models directory at: " + modelDir.getAbsolutePath());
            }
        }

        pb.environment().put("MTG_MODEL_PATH", modelPath);
        // Ensure Python logs INFO-level metrics
        pb.environment().put("MTG_AI_LOG_LEVEL", "INFO");
        logger.info("Set MTG_MODEL_PATH to: " + modelPath);

        pb.redirectErrorStream(true);
        pythonProcess = pb.start();

        // Read and log Python output in a separate thread
        Thread outputThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logger.info("[PYTHON] " + line);
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
            logger.info("Python process started successfully");
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
                clientServer = new ClientServer(null);
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
                logger.info(prefix + " " + line);
            }
        }
    }

    /**
     * Cleans up any existing Python processes.
     */
    private void cleanupExistingPythonProcesses() {
        try {
            // First try to kill by script name
            String scriptName = "py4j_entry_point.py";
            ProcessBuilder pb = new ProcessBuilder("wmic", "process", "where",
                    "commandline like '%" + scriptName + "%'", "get", "processid");
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            boolean foundProcess = false;

            // Skip header line
            reader.readLine();

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    try {
                        int pid = Integer.parseInt(line);
                        logger.info("Found Python process with PID: " + pid + ", attempting to kill it");
                        Runtime.getRuntime().exec("taskkill /F /PID " + pid);
                        foundProcess = true;
                    } catch (NumberFormatException e) {
                        // Skip non-numeric lines
                    }
                }
            }

            if (!foundProcess) {
                // If no specific process found, try killing all Python processes
                logger.info("No specific Python process found, attempting to kill all Python processes");
                Runtime.getRuntime().exec("taskkill /F /IM python.exe");
            }

            // Wait and verify processes are killed
            Thread.sleep(PROCESS_KILL_WAIT_MS);

            // Verify no Python processes are running
            pb = new ProcessBuilder("tasklist", "/FI", "IMAGENAME eq python.exe");
            process = pb.start();
            reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            while ((line = reader.readLine()) != null) {
                if (line.contains("python.exe")) {
                    logger.warning("Python process still running after kill attempt: " + line);
                    // One final attempt with a more forceful kill
                    Runtime.getRuntime().exec("taskkill /F /IM python.exe /T");
                    Thread.sleep(1000);
                }
            }

        } catch (Exception e) {
            logger.warning("Error while cleaning up Python processes: " + e.getMessage());
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
     * Make predictions using the Python ML model
     */
    public INDArray[] predict(StateSequenceBuilder.SequenceOutput state) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            PythonMLBatchManager.PredictionResult result = PythonMLBatchManager.getInstance(entryPoint).predict(state).get();
            return new INDArray[]{result.policyScores, result.valueScores};
        } catch (Exception e) {
            logger.severe("Error during prediction: " + e.getMessage());
            throw new RuntimeException("Failed to get predictions from Python model", e);
        }
    }

    public INDArray[] predictBatch(List<StateSequenceBuilder.SequenceOutput> states) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            // Create a list of futures to track all predictions
            List<CompletableFuture<PythonMLBatchManager.PredictionResult>> futures = new ArrayList<>();

            // Submit each state to the batch manager
            for (StateSequenceBuilder.SequenceOutput state : states) {
                futures.add(PythonMLBatchManager.getInstance(entryPoint).predict(state));
            }

            // Wait for all predictions to complete
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

            // Collect results
            INDArray[] policyScores = new INDArray[states.size()];
            INDArray[] valueScores = new INDArray[states.size()];

            for (int i = 0; i < futures.size(); i++) {
                PythonMLBatchManager.PredictionResult result = futures.get(i).get();
                policyScores[i] = result.policyScores;
                valueScores[i] = result.valueScores;
            }

            // Combine results into single arrays
            INDArray combinedPolicyScores = Nd4j.vstack(policyScores);
            INDArray combinedValueScores = Nd4j.vstack(valueScores);

            return new INDArray[]{combinedPolicyScores, combinedValueScores};
        } catch (Exception e) {
            logger.severe("Error during batch prediction: " + e.getMessage());
            e.printStackTrace(); // Add stack trace for better debugging
            throw new RuntimeException("Failed to get predictions from Python model", e);
        }
    }

    /**
     * Train the model with a batch of data
     */
    public void train(List<StateSequenceBuilder.TrainingData> trainingData, double reward) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            PythonMLBatchManager.getInstance(entryPoint).train(trainingData, reward).get();
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

    @Override
    protected void finalize() throws Throwable {
        shutdown();
        super.finalize();
    }
}
