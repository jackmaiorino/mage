package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.concurrent.TimeUnit;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import py4j.ClientServer;
import py4j.GatewayServer;
import py4j.Py4JException;

/**
 * Bridge class to interface with Python ML implementation using Py4J. Handles
 * conversion between Java and Python tensors and manages the Py4J gateway.
 */
public class PythonMLBridge {

    private static final Logger logger = Logger.getLogger(PythonMLBridge.class.getName());
    private static final int DEFAULT_PORT = 25334;
    private static final int PYTHON_STARTUP_WAIT_MS = 3000;
    private static final int PROCESS_KILL_WAIT_MS = 2000;

    private final String projectRoot = new File(new File(System.getProperty("user.dir")).getParentFile().getParentFile(), "mage").getAbsolutePath();
    private final String venvPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/venv").getAbsolutePath();
    private final String pythonScriptPath = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py").getAbsolutePath();

    private ClientServer clientServer;
    private PythonEntryPoint entryPoint;
    private Process pythonProcess;
    private boolean isInitialized = false;

    /**
     * Constructs a new PythonMLBridge instance. Initializes the Python
     * environment and starts the Py4J gateway.
     */
    public PythonMLBridge() {
        try {
            cleanupExistingPythonProcesses();
            setupPaths();
            setupVirtualEnvironment();
            installDependencies();
            startPythonProcess();
            connectToPythonGateway();
            isInitialized = true;
        } catch (Exception e) {
            logger.severe("Failed to initialize Python ML Bridge: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Failed to initialize Python ML Bridge", e);
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
     * Installs PyTorch with CUDA support.
     */
    private void installPyTorch(String pipPath) throws Exception {
        logger.info("Installing PyTorch with CUDA support");
        ProcessBuilder torchBuilder = new ProcessBuilder(pipPath, "install", "--upgrade",
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
    }

    /**
     * Installs a single Python package.
     */
    private void installPackage(String pipPath, String pkgName, String... args) throws Exception {
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
        pb.redirectErrorStream(true);
        pythonProcess = pb.start();

        // Read and log Python output in a separate thread
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logger.info("[PYTHON] " + line);
                }
            } catch (Exception e) {
                logger.severe("Error reading Python output: " + e.getMessage());
            }
        }).start();

        // Wait for Python process to start up
        Thread.sleep(PYTHON_STARTUP_WAIT_MS);
    }

    /**
     * Connects to the Python Py4J gateway.
     */
    private void connectToPythonGateway() throws Exception {
        clientServer = new ClientServer(null);
        entryPoint = (PythonEntryPoint) clientServer.getPythonServerEntryPoint(new Class[]{PythonEntryPoint.class});
        logger.info("Connected to Python ML Bridge on port " + DEFAULT_PORT);
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
        try {
            entryPoint.initializeModel();
            logger.info("Python ML model initialized successfully");
        } catch (Exception e) {
            logger.severe("Failed to initialize Python ML model: " + e.getMessage());
            throw new RuntimeException("Failed to initialize Python ML model", e);
        }
    }

    /**
     * Make predictions using the Python ML model
     */
    public INDArray[] predict(StateSequenceBuilder.SequenceOutput state) {
        List<StateSequenceBuilder.SequenceOutput> states = new ArrayList<>();
        states.add(state);
        return predictBatch(states);
    }

    public INDArray[] predictBatch(List<StateSequenceBuilder.SequenceOutput> states) {
        if (!isInitialized) {
            throw new IllegalStateException("Python ML Bridge not initialized");
        }

        try {
            // Convert states to format expected by Python
            float[][][] sequences = new float[states.size()][][];
            float[][] masks = new float[states.size()][];

            for (int i = 0; i < states.size(); i++) {
                StateSequenceBuilder.SequenceOutput state = states.get(i);
                List<float[]> tokens = state.getSequence();
                List<Integer> mask = state.getMask();

                // Convert tokens to 2D array
                sequences[i] = new float[tokens.size()][];
                for (int j = 0; j < tokens.size(); j++) {
                    sequences[i][j] = tokens.get(j);
                }

                // Convert mask to float array
                masks[i] = new float[mask.size()];
                for (int j = 0; j < mask.size(); j++) {
                    masks[i][j] = mask.get(j);
                }
            }

            // Get predictions from Python through Py4J
            float[] predictions = entryPoint.predictBatch(sequences, masks);

            // Split the concatenated array back into policy and value scores
            int batchSize = states.size();
            float[] policyScores = new float[batchSize];
            float[] valueScores = new float[batchSize];

            // Extract policy and value scores from the concatenated array
            System.arraycopy(predictions, 0, policyScores, 0, batchSize);
            System.arraycopy(predictions, batchSize, valueScores, 0, batchSize);

            // Convert to INDArrays
            INDArray policyScoresArray = Nd4j.create(policyScores);
            INDArray valueScoresArray = Nd4j.create(valueScores);

            return new INDArray[]{policyScoresArray, valueScoresArray};
        } catch (Py4JException e) {
            logger.severe("Py4J error during batch prediction: " + e.getMessage());
            throw new RuntimeException("Failed to get predictions from Python model", e);
        } catch (Exception e) {
            logger.severe("Error during batch prediction: " + e.getMessage());
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
            // Convert training data to format expected by Python
            float[][][] sequences = new float[trainingData.size()][][];
            float[][] masks = new float[trainingData.size()][];
            float[] policyScores = new float[trainingData.size()];
            float[] valueScores = new float[trainingData.size()];
            int[] actionTypes = new int[trainingData.size()];
            int[][] actionCombos = new int[trainingData.size()][];

            for (int i = 0; i < trainingData.size(); i++) {
                StateSequenceBuilder.TrainingData data = trainingData.get(i);
                List<float[]> tokens = data.stateActionPair.getSequence();
                List<Integer> mask = data.stateActionPair.getMask();

                // Convert tokens to 2D array
                sequences[i] = new float[tokens.size()][];
                for (int j = 0; j < tokens.size(); j++) {
                    sequences[i][j] = tokens.get(j);
                }

                // Convert mask to float array
                masks[i] = new float[mask.size()];
                for (int j = 0; j < mask.size(); j++) {
                    masks[i][j] = mask.get(j);
                }

                policyScores[i] = (float) data.policyScore;
                valueScores[i] = (float) data.valueScore;
                actionTypes[i] = data.actionType.ordinal();
                actionCombos[i] = data.actionCombo.stream().mapToInt(Integer::intValue).toArray();
            }

            // Send training data to Python through Py4J
            entryPoint.train(sequences, masks, policyScores, valueScores, actionTypes, actionCombos, (float) reward);
        } catch (Py4JException e) {
            logger.severe("Py4J error during training: " + e.getMessage());
            throw new RuntimeException("Failed to train Python model", e);
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
