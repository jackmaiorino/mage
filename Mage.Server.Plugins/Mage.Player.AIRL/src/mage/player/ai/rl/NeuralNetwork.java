package mage.player.ai.rl;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class NeuralNetwork {
    private static final Logger logger = Logger.getLogger(NeuralNetwork.class);
    public MultiLayerNetwork network;
    private final double explorationRate;
    private final int outputSize;
    private final BatchPredictionRequest batchPredictionRequest;
    
    // The neural net output is an outputSize x outputSize grid of probabilities
    public NeuralNetwork(int inputSize, int outputSize, double explorationRate) {
        this.explorationRate = explorationRate;
        this.outputSize = outputSize;
        //TODO: Change batch size to be thread related
        this.batchPredictionRequest = BatchPredictionRequest.getInstance(0, 10000, TimeUnit.MILLISECONDS);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam(0.0001)) // Adam optimizer with default parameters
        .weightDecay(1e-5) // L2 regularization to prevent overfitting
        .list()
        // Gradual dimensionality reduction from 15,000 -> 4096 -> 1024 -> 256 -> outputSize
        .layer(0, new DenseLayer.Builder()
            .nIn(inputSize)
            .nOut(4096)
            .activation(Activation.RELU) // ReLU activation for non-linearity
            .build())
        .layer(1, new BatchNormalization.Builder().build()) // Batch normalization to stabilize training
        .layer(2, new DenseLayer.Builder()
            .nIn(4096)
            .nOut(1024)
            .activation(Activation.RELU)
            .build())
        .layer(3, new DropoutLayer(0.3)) // Dropout to reduce overfitting (30%)
        .layer(4, new DenseLayer.Builder()
            .nIn(1024)
            .nOut(256)


            
            .activation(Activation.RELU)
            .build())
        .layer(5, new DropoutLayer(0.3))
        .layer(6, new OutputLayer.Builder()
            .nIn(256)
            .nOut(outputSize)
            .activation(Activation.SOFTMAX) // Softmax for multi-class classification
            .lossFunction(LossFunctions.LossFunction.MCXENT) // Multi-class cross-entropy loss
            .build())
        .build();
        
        network = new MultiLayerNetwork(conf);
        network.init();
    }
    
    public INDArray predict(RLState state, boolean isExploration) {
        float[] stateVector = state.getStateVector();
        if (state == null) {
            RLTrainer.threadLocalLogger.get().error("State array is null");
            throw new IllegalArgumentException("State array cannot be null");
        }
        // Epsilon-greedy exploration
        if (Math.random() <= explorationRate && isExploration) {
            RLTrainer.threadLocalLogger.get().info("Exploration!");
            // Create a 2D array for the output
            float[][] randomDist = new float[RLModel.MAX_ACTIONS][RLModel.MAX_OPTIONS];

            // Generate random values
            float totalSum = 0;
            for (int i = 0; i < state.exploreDimensions.size(); i++) {
                for(int j = 0; j < state.exploreDimensions.get(i); j++) {
                    randomDist[i][j] = (float) Math.random();
                    totalSum += randomDist[i][j];
                }
            }

            // Normalize all values to sum to 1
            for (int i = 0; i < state.exploreDimensions.size(); i++) {
                for(int j = 0; j < state.exploreDimensions.get(i); j++) {
                    randomDist[i][j] = (randomDist[i][j]/totalSum);
                }
            }

            state.output = Nd4j.create(randomDist);
            return state.output;
        }
        // Use BatchPredictionRequest for batch processing
        try {
            INDArray input = Nd4j.create(stateVector);
            INDArray output = batchPredictionRequest.predict(input);
            // Convert output to 2D array for masking
            float[][] outputArray = new float[RLModel.MAX_ACTIONS][RLModel.MAX_OPTIONS];
            for (int i = 0; i < state.exploreDimensions.size(); i++) {
                for (int j = 0; j < state.exploreDimensions.get(i); j++) {
                    outputArray[i][j] = output.getFloat(i * RLModel.MAX_OPTIONS + j);
                }
            }

            // Mask values outside explore dimensions
            float totalSum = 0;
            for (int i = 0; i < state.exploreDimensions.size(); i++) {
                for (int j = 0; j < state.exploreDimensions.get(i); j++) {
                    totalSum += outputArray[i][j];
                }
            }

            // Normalize remaining values to sum to 1
            for (int i = 0; i < state.exploreDimensions.size(); i++) {
                for (int j = 0; j < state.exploreDimensions.get(i); j++) {
                    outputArray[i][j] = outputArray[i][j] / totalSum;
                }
            }

            state.output = Nd4j.create(outputArray);
            return state.output;
        } catch (InterruptedException e) {
            RLTrainer.threadLocalLogger.get().error("Prediction interrupted", e);
            Thread.currentThread().interrupt();
            return null;
        }
    }

    public void updateWeightsBatch(List<float[]> states, INDArray[] targetQValuesList) {
        int batchSize = states.size();
        int stateLength = states.get(0).length;

        // Create a 2D INDArray for the batch of states
        INDArray inputBatch = Nd4j.create(new int[]{batchSize, stateLength});
        for (int i = 0; i < batchSize; i++) {
            inputBatch.putRow(i, Nd4j.create(states.get(i)));
        }

        // Create a 2D INDArray for the batch of target Q-values
        INDArray targetBatch = Nd4j.create(new long[]{batchSize, targetQValuesList[0].length()});
        for (int i = 0; i < batchSize; i++) {
            targetBatch.putRow(i, targetQValuesList[i]);
        }

        // Perform a single training step with the batch
        synchronized (network) {
            network.fit(inputBatch, targetBatch);
        }
    }

    public void saveNetwork(String filePath) throws IOException {
        ModelSerializer.writeModel(network, filePath, true);
    }

    public void loadNetwork(String filePath) throws IOException {
        try {
            network = ModelSerializer.restoreMultiLayerNetwork(filePath);
        } catch (IOException e) {
            logger.info("Could not load network from file, creating new network");
        }
    }

    // Helper method to convert Properties to Map<String, Object>
    private Map<String, Object> propertiesToMap(Properties properties) {
        Map<String, Object> map = new HashMap<>();
        for (String name : properties.stringPropertyNames()) {
            map.put(name, properties.getProperty(name));
        }
        return map;
    }
} 