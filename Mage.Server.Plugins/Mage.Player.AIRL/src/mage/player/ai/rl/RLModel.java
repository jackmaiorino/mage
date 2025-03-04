package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final NeuralNetwork network;
    public static final double EXPLORATION_RATE = 0.5;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    // TODO: Eliminate the need for a MAX_ACTIONS by finding ways to indicate multiple copies of same card efficiently
    public static final int MAX_ACTIONS = 25;
    public static final int MAX_OPTIONS = 15;
    public static final int OUTPUT_SIZE = (MAX_ACTIONS) * (MAX_OPTIONS);
    private static boolean IS_TRAINING = false;

    public RLModel(boolean training) {
        IS_TRAINING = training;
        // TODO: This is a little silly, creating a network and then loading it. Make it better
        network = new NeuralNetwork(RLState.STATE_VECTOR_SIZE, OUTPUT_SIZE, EXPLORATION_RATE);
        try {
            network.loadNetwork(RLTrainer.MODEL_FILE_PATH);
        } catch (IOException e) {
            logger.error("Failed to load network, initializing a new one.", e);
        }
    }

    public RLModel(NeuralNetwork network) {
        this.network = network;
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

        // Initialize target Q-values array
        INDArray[] targetQValuesArray = new INDArray[totalSize];
        for (int i = 0; i < totalSize; i++) {
            targetQValuesArray[i] = nextStates.get(i).output;
        }

        // Compute target Q-values for the batch
        for (int i = 0; i < totalSize; i++) {
            double reward = rewards.get(i);
            RLState nextState = nextStates.get(i);

            if (nextState.targetQValues.isEmpty()) {
                throw new IllegalArgumentException("Next state Q-values are empty");
            }

            for (QValueEntry qValueEntry : nextState.targetQValues) {
                double updatedQValue = reward + DISCOUNT_FACTOR * qValueEntry.getQValue();
                // Normalize Q-values to the range [0, 1]
                updatedQValue = Math.max(0.0, Math.min(1.0, updatedQValue));
                targetQValuesArray[i].putScalar(qValueEntry.getIndex(), updatedQValue);
            }
        }

        // Map states to input vectors
        List<float[]> statesArray = states.stream()
                .map(RLState::getStateVector)
                .collect(Collectors.toList());

        // Update weights using the processed batch
        network.updateWeightsBatch(statesArray, targetQValuesArray);
    }
} 