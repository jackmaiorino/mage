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

        long startTime = System.nanoTime();
        try {
            // Convert states to format expected by Python
            long convertStart = System.nanoTime();
            int batchSize = states.size();
            int seqLen = states.get(0).getSequence().size();
            int dModel = states.get(0).getSequence().get(0).length;

            // Create flat arrays for faster transfer
            float[] sequencesFlat = new float[batchSize * seqLen * dModel];
            float[] masksFlat = new float[batchSize * seqLen];

            // Fill arrays in a single pass
            int seqIdx = 0;
            int maskIdx = 0;
            for (StateSequenceBuilder.SequenceOutput state : states) {
                List<float[]> tokens = state.getSequence();
                List<Integer> mask = state.getMask();

                for (int j = 0; j < seqLen; j++) {
                    float[] token = tokens.get(j);
                    System.arraycopy(token, 0, sequencesFlat, seqIdx, dModel);
                    seqIdx += dModel;
                    masksFlat[maskIdx++] = mask.get(j);
                }
            }

            // Convert float arrays to byte arrays
            byte[] sequencesBytes = new byte[sequencesFlat.length * 4];  // 4 bytes per float
            byte[] masksBytes = new byte[masksFlat.length * 4];

            // Use ByteBuffer for efficient conversion with LITTLE_ENDIAN to match Python
            java.nio.ByteBuffer.wrap(sequencesBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(sequencesFlat);
            java.nio.ByteBuffer.wrap(masksBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(masksFlat);

            long convertTime = System.nanoTime() - convertStart;
            logger.info(String.format("Java array conversion took %.3f seconds", convertTime / 1e9));

            // Get predictions from Python through Py4J
            long predictStart = System.nanoTime();
            byte[] predictionsBytes = entryPoint.predictBatchFlat(sequencesBytes, masksBytes, batchSize, seqLen, dModel);
            long predictTime = System.nanoTime() - predictStart;
            logger.info(String.format("Py4J prediction call took %.3f seconds", predictTime / 1e9));

            // Convert predictions to INDArrays
            long splitStart = System.nanoTime();
            float[] predictions = new float[predictionsBytes.length / 4];
            java.nio.ByteBuffer.wrap(predictionsBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .get(predictions);

            float[] policyScores = new float[batchSize];
            float[] valueScores = new float[batchSize];

            // Extract policy and value scores from the predictions array
            System.arraycopy(predictions, 0, policyScores, 0, batchSize);
            System.arraycopy(predictions, batchSize, valueScores, 0, batchSize);

            // Convert to INDArrays
            INDArray policyScoresArray = Nd4j.create(policyScores);
            INDArray valueScoresArray = Nd4j.create(valueScores);
            long splitTime = System.nanoTime() - splitStart;
            logger.info(String.format("Score splitting and INDArray conversion took %.3f seconds", splitTime / 1e9));

            long totalTime = System.nanoTime() - startTime;
            logger.info(String.format("Total predictBatch operation took %.3f seconds", totalTime / 1e9));

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
            long startTime = System.nanoTime();
            // Convert training data to format expected by Python
            int batchSize = trainingData.size();
            int seqLen = trainingData.get(0).stateActionPair.getSequence().size();
            int dModel = trainingData.get(0).stateActionPair.getSequence().get(0).length;
            int maxActions = trainingData.get(0).actionCombo.size();

            // Create flat arrays for faster transfer
            float[] sequencesFlat = new float[batchSize * seqLen * dModel];
            float[] masksFlat = new float[batchSize * seqLen];
            float[] policyScoresFlat = new float[batchSize];
            float[] valueScoresFlat = new float[batchSize];
            int[] actionTypesFlat = new int[batchSize];
            int[] actionCombosFlat = new int[batchSize * maxActions];

            // Fill arrays in a single pass
            int seqIdx = 0;
            int maskIdx = 0;
            int comboIdx = 0;
            for (int i = 0; i < batchSize; i++) {
                StateSequenceBuilder.TrainingData data = trainingData.get(i);
                List<float[]> tokens = data.stateActionPair.getSequence();
                List<Integer> mask = data.stateActionPair.getMask();

                // Fill sequences and masks
                for (int j = 0; j < seqLen; j++) {
                    float[] token = tokens.get(j);
                    System.arraycopy(token, 0, sequencesFlat, seqIdx, dModel);
                    seqIdx += dModel;
                    masksFlat[maskIdx++] = mask.get(j);
                }

                // Fill other arrays
                policyScoresFlat[i] = (float) data.policyScore;
                valueScoresFlat[i] = (float) data.valueScore;
                actionTypesFlat[i] = data.actionType.ordinal();
                for (int j = 0; j < maxActions; j++) {
                    actionCombosFlat[comboIdx++] = data.actionCombo.get(j);
                }
            }

            // Convert float arrays to byte arrays
            byte[] sequencesBytes = new byte[sequencesFlat.length * 4];
            byte[] masksBytes = new byte[masksFlat.length * 4];
            byte[] policyScoresBytes = new byte[policyScoresFlat.length * 4];
            byte[] valueScoresBytes = new byte[valueScoresFlat.length * 4];
            byte[] actionTypesBytes = new byte[actionTypesFlat.length * 4];
            byte[] actionCombosBytes = new byte[actionCombosFlat.length * 4];

            // Use ByteBuffer for efficient conversion with LITTLE_ENDIAN to match Python
            java.nio.ByteBuffer.wrap(sequencesBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(sequencesFlat);
            java.nio.ByteBuffer.wrap(masksBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(masksFlat);
            java.nio.ByteBuffer.wrap(policyScoresBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(policyScoresFlat);
            java.nio.ByteBuffer.wrap(valueScoresBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(valueScoresFlat);
            java.nio.ByteBuffer.wrap(actionTypesBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer()
                    .put(actionTypesFlat);
            java.nio.ByteBuffer.wrap(actionCombosBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer()
                    .put(actionCombosFlat);

            long convertTime = System.nanoTime() - startTime;
            logger.info(String.format("Java array conversion took %.3f seconds", convertTime / 1e9));

            // Send training data to Python through Py4J
            long trainStart = System.nanoTime();
            entryPoint.trainFlat(sequencesBytes, masksBytes, policyScoresBytes, valueScoresBytes,
                    actionTypesBytes, actionCombosBytes, batchSize, seqLen, dModel, maxActions, (float) reward);
            long trainTime = System.nanoTime() - trainStart;
            logger.info(String.format("Py4J training call took %.3f seconds", trainTime / 1e9));

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
