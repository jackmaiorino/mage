package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final NeuralNetwork network;
    private static final double EXPLORATION_RATE = 0.5;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    // TODO: Eliminate the need for a MAX_ACTIONS by finding ways to indicate multiple copies of same card efficiently
    public static final int MAX_ACTIONS = 25;
    public static final int MAX_OPTIONS = 15;
    public static final int OUTPUT_SIZE = (MAX_ACTIONS) * (MAX_OPTIONS);
    public static boolean IS_TRAINING = true;

    public RLModel() {
        // TODO: This is a little silly, creating a network and then loading it. Make it better
        network = new NeuralNetwork(RLState.STATE_VECTOR_SIZE, OUTPUT_SIZE, EXPLORATION_RATE);
        try {
            network.loadNetwork(RLTrainer.MODEL_FILE_PATH);
        } catch (IOException e) {
            logger.error("Failed to load network, initializing a new one.", e);
        }
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public void saveModel(String filePath) {
        try {
            network.saveNetwork(filePath);
        } catch (IOException e) {
            logger.error("Failed to save network.", e);
        }
    }

    public INDArray predictDistribution(RLState state, boolean isExploration) {
        if (IS_TRAINING) {
            return network.predict(state, isExploration);
        } else {
            return network.predict(state, false);
        }
    }

    public void updateBatch(List<RLState> states, List<Double> rewards, List<RLState> nextStates) {
        int batchSize = 100;
        int totalSize = states.size();

        INDArray[] targetQValuesArray = new INDArray[totalSize];
        for (int i = 0; i < totalSize; i++) {
            targetQValuesArray[i] = Nd4j.zeros(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
        }

        for (int i = 0; i < totalSize; i++) {
            double reward = rewards.get(i);
            if (nextStates.get(i).targetQValues.isEmpty()) {
                throw new IllegalArgumentException("Next state Q-values are empty");
            }
            for (QValueEntry qValueEntry : nextStates.get(i).targetQValues) {
                targetQValuesArray[i].putScalar(qValueEntry.getXIndex(), qValueEntry.getYIndex(), reward + DISCOUNT_FACTOR * qValueEntry.getQValue());
            }
        }

        List<float[]> statesArray = states.stream()
            .map(RLState::getStateVector)
            .collect(Collectors.toList());

        // Process
        network.updateWeightsBatch(statesArray, targetQValuesArray);

        // Maybe use this idea for a batch manager
        // for (int start = 0; start < totalSize; start += batchSize) {
        //     int end = Math.min(start + batchSize, totalSize);

        //     List<float[]> statesArray = states.subList(start, end).stream()
        //         .map(RLState::getStateVector)
        //         .collect(Collectors.toList());

        //     // The set/unset design here is to avoid the heavy load of creating new INDArrays
        //     // Set Q-vals
        //     for (int i = start; i < end; i++) {
        //         double reward = rewards.get(i);
        //         for (QValueEntry qValueEntry : nextStates.get(i).targetQValues) {
        //             targetQValuesArray[i].putScalar(qValueEntry.getXIndex(),qValueEntry.getYIndex(), reward + DISCOUNT_FACTOR * qValueEntry.getQValue());
        //         }
        //     }

        //     // Process
        //     network.updateWeightsBatch(statesArray, targetQValuesArray);

        //     //Unset Q-vals
        //     for (int i = start; i < end; i++) {
        //         for (QValueEntry qValueEntry : nextStates.get(i).targetQValues) {
        //             targetQValuesArray[i].putScalar(qValueEntry.getXIndex(),qValueEntry.getYIndex(), 0);
        //         }
        //     }

        // }
    }
} 