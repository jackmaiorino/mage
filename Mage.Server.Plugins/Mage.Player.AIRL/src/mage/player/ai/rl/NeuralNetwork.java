package mage.player.ai.rl;

import java.io.IOException;

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
    }
    
    public INDArray predict(float[] state, boolean isExploration) {
        // Epsilon-greedy exploration
        if (Math.random() <= explorationRate && isExploration) {
            // Generate random softmax distribution
            float[] randomDist = new float[outputSize * outputSize];
            float sum = 0;
            for (int i = 0; i < randomDist.length; i++) {
                randomDist[i] = (float) Math.random();
                sum += randomDist[i];
            }
            // Normalize to sum to 1
            for (int i = 0; i < randomDist.length; i++) {
                randomDist[i] /= sum;
            }
            return Nd4j.create(randomDist).reshape(outputSize, outputSize);
        }
        return network.output(Nd4j.create(state)).reshape(outputSize, outputSize);
    }

    // New idea
    // public INDArray predict(float[] state, float[] action, boolean isExploration, INDArray actionMask) {
    //     // Combine state and action into a single input vector
    //     float[] combined = new float[state.length + action.length];
    //     System.arraycopy(state, 0, combined, 0, state.length);
    //     System.arraycopy(action, 0, combined, state.length, action.length);
    
    //     // Create input INDArray
    //     INDArray input = Nd4j.create(combined);
    
    //     // Get output from the network
    //     INDArray output = network.output(input);
    
    //     // Reshape to 2D array
    //     output = output.reshape(outputSize, outputSize);
    
    //     if (isExploration && Math.random() <= explorationRate) {
    //         // Exploration: generate random actions (but apply mask during inference)
    //         output = Nd4j.rand(outputSize, outputSize);
    //     }
    
    //     // During training: Penalize invalid actions
    //     if (!isExploration) {
    //         // Calculate penalties for invalid actions
    //         INDArray invalidProbabilities = output.mul(actionMask.rsub(1)); // 1 - mask
    //         float penalty = invalidProbabilities.sumNumber().floatValue();
    
    //         // Apply penalty to loss function (handled outside predict)
    //         // Example: totalLoss += penalty * penaltyWeight;
    //     }
    
    //     // During inference: Apply mask
    //     output.muli(actionMask); // Element-wise multiply with mask
    
    //     // Normalize the masked output using softmax
    //     INDArray expOutput = Transforms.exp(output); // Element-wise exponential
    //     INDArray sumExp = expOutput.sum(); // Sum of all exponentials
    //     INDArray softmaxOutput = expOutput.div(sumExp); // Normalize to get probabilities
    
    //     return softmaxOutput;
    // }

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
} 