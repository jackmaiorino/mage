package mage.player.ai.rl;

import mage.game.Game;
import mage.abilities.ActivatedAbility;
import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class RLModel implements Serializable {
    private NeuralNetwork network;
    private double explorationRate;
    private static final double LEARNING_RATE = 0.001;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;

    public RLModel() {
        this.network = new NeuralNetwork(RLState.STATE_VECTOR_SIZE + RLAction.FEATURE_VECTOR_SIZE, RLAction.MAX_ACTIONS);
        this.explorationRate = 0.1;
    }

    public float getActionThreshold() {
        return 0.2f; // Threshold can be tuned based on training
    }

    public float[] predictDistribution(RLState state) {
        // Convert INDArray to float[]
        return network.predict(state.getStateVector(), state.getCurrentAction().getFeatureVector()).data().asFloat();
    }

    public void saveModel(String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static RLModel loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (RLModel) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public void update(RLState state, double reward, RLState nextState) {
        float[] currentQValues = predictDistribution(state);
        float maxNextQValue = 0;
        float[] nextQValues = predictDistribution(nextState);
        for (float qValue : nextQValues) {
            if (qValue > maxNextQValue) {
                maxNextQValue = qValue;
            }
        }
        float targetQValue = (float) (reward + DISCOUNT_FACTOR * maxNextQValue);
        network.updateWeights(state.getStateVector(), state.getCurrentAction().getFeatureVector(), targetQValue, currentQValues[state.getCurrentAction().getType().ordinal()]);
    }
} 