package mage.player.ai.rl;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
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
    private MultiLayerNetwork network;
    private final double explorationRate;
    private final int outputSize;
    
    // The neural net output is an outputSize x outputSize grid of probabilities
    public NeuralNetwork(int inputSize, int outputSize, double explorationRate) {
        this.explorationRate = explorationRate;
        this.outputSize = outputSize;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new Adam())
            .list()
            .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(64).activation(Activation.RELU).build())
            .layer(1, new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
            .layer(2, new OutputLayer.Builder().nIn(32).nOut(outputSize).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();
        // Output is +1 for pass priority or no block or no attack
        
        network = new MultiLayerNetwork(conf);
        network.init();

        // Log if using GPU
        Properties envInfoProps = Nd4j.getExecutioner().getEnvironmentInformation();
        Map<String, Object> envInfo = propertiesToMap(envInfoProps);
        if ("CUDA".equals(envInfo.get("backend"))) {
            logger.info("Neural Network is using a GPU.");
        } else {
            logger.info("Neural Network is using a CPU.");
        }
    }
    
    public INDArray predict(float[] state, boolean isExploration) {
        // Epsilon-greedy exploration
        if (Math.random() <= explorationRate && isExploration) {
            logger.info("Exploration!");
            // Create a 2D array for the output
            float[][] randomDist = new float[RLModel.MAX_ACTIONS][RLModel.MAX_ACTIONS + 1];
            
            // Generate random values for the first 11 indices of the first row
            float sum = 0;
            for (int j = 0; j < RLModel.MAX_ACTIONS + 1; j++) {
                randomDist[0][j] = (float) Math.random();
                sum += randomDist[0][j];
            }
            // Normalize the first 11 values to sum to 1
            for (int j = 0; j < RLModel.MAX_ACTIONS + 1; j++) {
                randomDist[0][j] /= sum;
            }
            
            // Set the rest of the array to zeros
            for (int i = 1; i < RLModel.MAX_ACTIONS; i++) {
                for (int j = 0; j < RLModel.MAX_ACTIONS + 1; j++) {
                    randomDist[i][j] = 0;
                }
            }
            return Nd4j.create(randomDist);
        }
        return network.output(Nd4j.create(state)).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_ACTIONS + 1);
    }

    public void updateWeights(float[] state, INDArray targetQValues) {
        INDArray input = Nd4j.create(state);
        INDArray target = Nd4j.create(targetQValues.data().asFloat());
        // Perform a single training step
        network.fit(input, target);
    }

    public void saveNetwork(String filePath) throws IOException {
        ModelSerializer.writeModel(network, filePath, true);
    }

    public void loadNetwork(String filePath) throws IOException {
        try {
            network = ModelSerializer.restoreMultiLayerNetwork(filePath);
        } catch (IOException e) {
            logger.error("Can't find file: " + filePath, e);
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