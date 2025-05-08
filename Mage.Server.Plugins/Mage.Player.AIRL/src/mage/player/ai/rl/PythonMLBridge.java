package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import py4j.GatewayServer;
import py4j.Py4JException;

/**
 * Bridge class to interface with Python ML implementation using Py4J.
 * Handles conversion between Java and Python tensors and manages the Py4J gateway.
 */
public class PythonMLBridge {
    private static final Logger logger = Logger.getLogger(PythonMLBridge.class.getName());
    private static final int DEFAULT_PORT = 25333;
    private GatewayServer gateway;
    private PythonEntryPoint entryPoint;
    private boolean isInitialized = false;
    private Process pythonProcess;

    public PythonMLBridge() {
        try {
            // Create the entry point that Python will call
            entryPoint = new PythonEntryPoint();
            
            // Start the Py4J gateway server
            gateway = new GatewayServer(entryPoint, DEFAULT_PORT);
            gateway.start();
            
            // Start the Python process
            String pythonScript = new File("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py").getAbsolutePath();
            logger.info("Attempting to start Python process with script: " + pythonScript);
            
            // Check if Python is available
            ProcessBuilder checkPython = new ProcessBuilder("python", "--version");
            Process checkProcess = checkPython.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(checkProcess.getInputStream()));
            String pythonVersion = reader.readLine();
            logger.info("Python version: " + pythonVersion);
            
            // Start the actual Python process
            ProcessBuilder pb = new ProcessBuilder("python", pythonScript);
            
            // Set up environment variables
            Map<String, String> env = pb.environment();
            env.put("PYTHONPATH", new File("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode").getAbsolutePath());
            
            // Redirect both standard output and error
            pb.redirectErrorStream(true);
            pythonProcess = pb.start();
            
            // Read and log the Python process output
            reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                logger.info("Python output: " + line);
            }
            
            // Check if process is still alive
            if (!pythonProcess.isAlive()) {
                int exitCode = pythonProcess.exitValue();
                logger.severe("Python process exited with code: " + exitCode);
                logger.severe("Python process output: " + output.toString());
                throw new RuntimeException("Python process failed to start. Exit code: " + exitCode + "\nOutput: " + output.toString());
            }
            
            // Wait a bit for Python to start up
            Thread.sleep(2000);
            
            logger.info("Python ML Bridge started on port " + DEFAULT_PORT);
            logger.info("Gateway server address: " + gateway.getAddress());
            logger.info("Gateway server port: " + gateway.getPort());
            
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
        if (pythonProcess != null) {
            pythonProcess.destroy();
        }
        if (gateway != null) {
            try {
                gateway.shutdown();
                isInitialized = false;
                logger.info("Python ML Bridge shutdown complete");
            } catch (Exception e) {
                logger.severe("Error during shutdown: " + e.getMessage());
            }
        }
    }
} 