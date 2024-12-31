package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
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
    public static final int MAX_ACTIONS = 10;
    public static final int OUTPUT_SIZE = (MAX_ACTIONS + 1) * (MAX_ACTIONS); // +1 for no attack/no block per attacker/blocker

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
        return network.predict(state.getStateVector(), isExploration);
    }

    // TODO: Research the algorithm used here. I don't really understand it.
    // NOTE: action here is WRONG. It is the output from state, not the input to state
    public INDArray getTargetQValue(INDArray nextQValues, double reward) {
        INDArray targetQValues = Nd4j.zeros(RLModel.OUTPUT_SIZE);
        for (int i = 0; i < nextQValues.data().length(); i++) {
            if (nextQValues.getDouble(i) != 0) {
                targetQValues.putScalar(i, reward + DISCOUNT_FACTOR * nextQValues.getDouble(i));
            }
        }
        return targetQValues;
    }

    public void update(RLState state, double reward, RLState nextState) {
        INDArray nextQValues = nextState.targetQValues;
        INDArray targetQValues;

        switch (state.actionType) {
            case DECLARE_ATTACKS:
            case DECLARE_BLOCKS:
            case ACTIVATE_ABILITY_OR_SPELL:
                targetQValues = getTargetQValue(nextQValues, reward);
                break;
            default:
                logger.error("Unknown action type: " + state.actionType);
                throw new IllegalArgumentException("Unknown action type: " + state.actionType);
        }

        network.updateWeightsCPU(state.getStateVector(), targetQValues);
    }

    public void updateBatch(List<RLState> states, List<Double> rewards, List<RLState> nextStates) {
        List<float[]> statesArray = states.stream().map(RLState::getStateVector).collect(Collectors.toList());
        List<INDArray> targetQValuesList = new ArrayList<>();

        for (int i = 0; i < states.size(); i++) {
            INDArray nextQValues = nextStates.get(i).targetQValues;
            double reward = rewards.get(i);
            INDArray targetQValues = getTargetQValue(nextQValues, reward);
            targetQValuesList.add(targetQValues);
        }

        network.updateWeightsGPU(statesArray, targetQValuesList);
    }
} 