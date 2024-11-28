package mage.player.ai.rl;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private MultiLayerNetwork network;
    private final int inputSize;
    private final int outputSize;

    public NeuralNetwork(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(64)
                        .nOut(outputSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(100));
    }

    public List<Float> predict(List<Float> input) {
        INDArray inputArray = Nd4j.create(convertToArray(input));
        INDArray output = network.output(inputArray);
        return convertToList(output);
    }

    public void train(float[][] inputs, float[][] targets, double learningRate) {
        INDArray inputArray = Nd4j.create(inputs);
        INDArray targetArray = Nd4j.create(targets);
        network.fit(inputArray, targetArray);
    }

    private float[] convertToArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    private List<Float> convertToList(INDArray array) {
        float[] floatArray = array.data().asFloat();
        List<Float> list = new ArrayList<>();
        for (float value : floatArray) {
            list.add(value);
        }
        return list;
    }

    public void saveModel(String filepath) throws IOException {
        ModelSerializer.writeModel(network, filepath, true);
    }

    public void loadModel(String filepath) throws IOException {
        network = ModelSerializer.restoreMultiLayerNetwork(filepath);
    }
} 