package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final NeuralNetwork network;
    private static final double LEARNING_RATE = 0.001;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    public static final int MAX_ACTIONS = 10;
    public static final int OUTPUT_SIZE = (MAX_ACTIONS + 1) * (MAX_ACTIONS); // +1 for no attack/no block per attacker/blocker


    public RLModel() {
        // TODO: This is a little silly, creating a network and then loading it. Make it better
        network = new NeuralNetwork(RLState.STATE_VECTOR_SIZE, OUTPUT_SIZE, 0.1);
        try {
            network.loadNetwork("network.ser");
        } catch (IOException e) {
            logger.error("Failed to load network, initializing a new one.", e);
        }
    }

    public void saveModel(String filePath) {
        try {
            network.saveNetwork(filePath);
        } catch (IOException e) {
            logger.error("Failed to save network.", e);
        }
    }


    // This has to be this way to ensure its possible to attack with all creatures
    public double getAttackOrBlockThreshold() {
        return (double) 1 / MAX_ACTIONS; // Threshold can be tuned based on training
    }

    public INDArray predictDistribution(RLState state, boolean isExploration) {
        return network.predict(state.getStateVector(), isExploration);
    }

    // TODO: Research the algorithm used here. I don't really understand it.
    // NOTE: action here is WRONG. It is the output from state, not the input to state
    public void update(RLState state, double reward, RLState nextState) {
        INDArray nextQValues = predictDistribution(nextState, false);
        INDArray targetQValues = Nd4j.zeros(OUTPUT_SIZE, OUTPUT_SIZE);

        //TODO: Don't set qval if skip action was selected
        // Need to save gamestate?
        switch (state.actionType) {
            case DECLARE_ATTACKS:
                // Set target Q-values for all attackers above the threshold
                for (int i = 0; i < MAX_ACTIONS; i++) {
                    if (nextQValues.getDouble(i) > getAttackOrBlockThreshold()) {
                        targetQValues.putScalar(i, reward + DISCOUNT_FACTOR * nextQValues.getDouble(i));
                    }
                }
                break;
            case DECLARE_BLOCKS:
                // Set target Q-values for all blockers above the threshold
                for (int i = 0; i < nextQValues.length(); i++) {
                    if (nextQValues.getDouble(i) > getAttackOrBlockThreshold()) {
                        targetQValues.putScalar(i, reward + DISCOUNT_FACTOR * nextQValues.getDouble(i));
                    }
                }
                break;
            case ACTIVATE_ABILITY_OR_SPELL:
                // Find the index of the maximum Q-value for the next state
                int maxIndex = 0;
                double maxNextQValue = nextQValues.getDouble(0);
                for (int i = 1; i < nextQValues.length(); i++) {
                    if (nextQValues.getDouble(i) > maxNextQValue) {
                        maxNextQValue = nextQValues.getDouble(i);
                        maxIndex = i;
                    }
                }
                targetQValues.putScalar(maxIndex, reward + DISCOUNT_FACTOR * maxNextQValue);
                break;
            default:
                // Error since we don't know what to do with this action
                logger.error("Unknown action type: " + state.actionType);
                throw new IllegalArgumentException("Unknown action type: " + state.actionType);
        }

        // Update the network weights
        network.updateWeights(state.getStateVector(), targetQValues);
    }
} 