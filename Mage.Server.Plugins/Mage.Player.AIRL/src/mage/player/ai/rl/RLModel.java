package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;

import org.apache.log4j.Logger;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final NeuralNetwork network;
    private static final double LEARNING_RATE = 0.001;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    private static final int OUTPUT_SIZE = RLAction.MAX_ACTIONS + 1;

    public RLModel() {
        // TODO: This is a little silly, creating a network and then loading it. Make it better
        // Output is +1 for pass priority
        network = new NeuralNetwork(RLState.STATE_VECTOR_SIZE + RLAction.FEATURE_VECTOR_SIZE, OUTPUT_SIZE, 0.1);
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
    public float getActionThreshold() {
        return (float) 1 / RLAction.MAX_ACTIONS; // Threshold can be tuned based on training
    }

    public float[] predictDistribution(RLState state, RLAction action) {
        return network.predict(state.getStateVector(), action.getFeatureVector()).data().asFloat();
    }

    // TODO: Research the algorithm used here. I don't really understand it.
    public void update(RLState state, double reward, RLState nextState, RLAction action) {
        float[] nextQValues = predictDistribution(nextState, action);
        float[] targetQValues = new float[OUTPUT_SIZE];

        switch (action.getType()) {
            case DECLARE_ATTACKS:
                // Set target Q-values for all attackers above the threshold
                for (int i = 0; i < nextQValues.length; i++) {
                    if (nextQValues[i] > getActionThreshold()) {
                        targetQValues[i] = (float) (reward + DISCOUNT_FACTOR * nextQValues[i]);
                    }
                }
                break;
            case ACTIVATE_ABILITY_OR_SPELL:
                // Find the index of the maximum Q-value for the next state
                int maxIndex = 0;
                float maxNextQValue = nextQValues[0];
                for (int i = 1; i < nextQValues.length; i++) {
                    if (nextQValues[i] > maxNextQValue) {
                        maxNextQValue = nextQValues[i];
                        maxIndex = i;
                    }
                }
                targetQValues[maxIndex] = (float) (reward + DISCOUNT_FACTOR * maxNextQValue);
                break;
            default:
                // Error since we don't know what to do with this action
                logger.error("Unknown action type: " + action.getType());
                throw new IllegalArgumentException("Unknown action type: " + action.getType());
        }

        // Update the network weights
        network.updateWeights(state.getStateVector(), action.getFeatureVector(), targetQValues);
    }
} 