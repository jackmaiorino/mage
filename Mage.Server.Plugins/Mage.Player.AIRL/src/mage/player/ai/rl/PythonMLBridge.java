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
import py4j.Py4JException;

/**
 * Bridge class to interface with Python ML implementation using Py4J.
 * Handles conversion between Java and Python tensors and manages the Py4J gateway.
 */
public class PythonMLBridge {
    private static final Logger logger = Logger.getLogger(PythonMLBridge.class.getName());
    private static final int DEFAULT_PORT = 25334;
    private ClientServer clientServer;
    private PythonEntryPoint entryPoint;
    private Process pythonProcess;
    private boolean isInitialized = false;

    public PythonMLBridge() {
        try {
            // Get the workspace root directory
            String workspaceRoot = System.getProperty("user.dir");
            logger.info("Workspace root: " + workspaceRoot);
            
            // Start the Python process with the correct path
            // Go up two directories from Mage.Tests to reach the root, then add 'mage'
            String projectRoot = new File(workspaceRoot).getParentFile().getParentFile().getAbsolutePath();
            projectRoot = new File(projectRoot, "mage").getAbsolutePath();
            logger.info("Project root: " + projectRoot);
            
            // Construct the path to the Python script
            String pythonScript = new File(projectRoot, "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py").getAbsolutePath();
            logger.info("Starting Python process with script: " + pythonScript);
            
            // Verify the file exists
            File scriptFile = new File(pythonScript);
            if (!scriptFile.exists()) {
                throw new RuntimeException("Python script not found at: " + pythonScript);
            }
            
            ProcessBuilder pb = new ProcessBuilder("python", pythonScript);
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
            Thread.sleep(3000);
            
            // Connect to the Python process
            clientServer = new ClientServer(null);
            entryPoint = (PythonEntryPoint) clientServer.getPythonServerEntryPoint(new Class[] { PythonEntryPoint.class });
            
            logger.info("Connected to Python ML Bridge on port " + DEFAULT_PORT);
            isInitialized = true;
            
        } catch (Exception e) {
            logger.severe("Failed to initialize Python ML Bridge: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Failed to initialize Python ML Bridge", e);
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
            float[][] predictions = entryPoint.predictBatch(sequences, masks);
            
            // Convert predictions to INDArrays
            INDArray policyScores = Nd4j.create(predictions[0]);
            INDArray valueScores = Nd4j.create(predictions[1]);
            
            return new INDArray[]{policyScores, valueScores};
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