package mage.player.ai.rl;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class NeuralNetwork {
    private MultiLayerNetwork network;
    
    public NeuralNetwork(int inputSize, int outputSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(new Adam())
            .list()
            .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(64).activation(Activation.RELU).build())
            .layer(1, new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
            .layer(2, new OutputLayer.Builder().nIn(32).nOut(outputSize).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();
        
        network = new MultiLayerNetwork(conf);
        network.init();
    }
    
    public INDArray predict(float[] state, float[] action) {
        float[] combined = new float[state.length + action.length];
        System.arraycopy(state, 0, combined, 0, state.length);
        System.arraycopy(action, 0, combined, state.length, action.length);
        
        INDArray input = Nd4j.create(combined);
        INDArray output = network.output(input);
        return output;
    }

    public void updateWeights(float[] state, float[] action, float targetQValue, float currentQValue) {
        float[] combined = new float[state.length + action.length];
        System.arraycopy(state, 0, combined, 0, state.length);
        System.arraycopy(action, 0, combined, state.length, action.length);

        INDArray input = Nd4j.create(combined);
        INDArray target = Nd4j.create(new float[]{targetQValue});

        // Perform a single training step
        network.fit(input, target);
    }
} 