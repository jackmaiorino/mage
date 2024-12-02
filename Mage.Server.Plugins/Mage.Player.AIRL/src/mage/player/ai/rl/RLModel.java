package mage.player.ai.rl;

import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.players.Player;
import mage.util.RandomUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;
import java.util.ArrayList;
import java.util.UUID;
import mage.player.ai.ComputerPlayerRL;

public class RLModel {
    private final UUID playerId;
    private NeuralNetwork network;
    private double explorationRate;
    private static final double LEARNING_RATE = 0.001;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final int STATE_SIZE = 45;  // 5 player state + 40 battlefield state
    private static final int ACTION_SIZE = 10; // Adjust based on number of possible actions

    public RLModel(UUID playerId) {
        this.playerId = playerId;
        this.network = new NeuralNetwork(STATE_SIZE, ACTION_SIZE);
        this.explorationRate = 0.1;
    }

    public RLAction getAction(RLState state) {
        if (state == null || state.getGame() == null) {
            return new RLAction(RLAction.ActionType.PASS);
        }

        Game game = state.getGame();
        Player player = game.getPlayer(playerId);
        if (player == null) {
            System.err.println("Player not found for this game");
            throw new IllegalStateException("Player not found for ID: " + playerId);
        }

        List<RLAction> actions = ComputerPlayerRL.getPlayableActions(game, (ComputerPlayerRL)player);
        
        if (actions.isEmpty()) {
            return new RLAction(RLAction.ActionType.PASS);
        }

        // Epsilon-greedy exploration
        if (RandomUtil.nextDouble() < explorationRate) {
            return actions.get(RandomUtil.nextInt(actions.size()));
        }

        // Get Q-values for all actions
        List<Float> qValues = predictQValues(state, actions);
        
        // Find action with highest Q-value
        int bestActionIndex = 0;
        float maxQValue = qValues.get(0);
        for (int i = 1; i < qValues.size(); i++) {
            if (qValues.get(i) > maxQValue) {
                maxQValue = qValues.get(i);
                bestActionIndex = i;
            }
        }

        return actions.get(bestActionIndex);
    }

    private List<Float> predictQValues(RLState state, List<RLAction> actions) {
        return network.predict(state.getStateVector());
    }

    public void update(RLState state, RLAction action, double reward, RLState nextState) {
        float[][] currentState = new float[][] { convertToArray(state.getStateVector()) };
        float[][] nextStateArray = new float[][] { convertToArray(nextState.getStateVector()) };
        
        List<Float> nextQValues = network.predict(nextState.getStateVector());
        double maxNextQ = nextQValues.stream()
                .max(Float::compareTo)
                .orElse(0.0f);

        double targetQ = reward + DISCOUNT_FACTOR * maxNextQ;
        
        float[][] targets = new float[1][ACTION_SIZE];
        // Set target Q-value for the taken action
        targets[0][getActionIndex(action)] = (float) targetQ;
        
        network.train(currentState, targets, LEARNING_RATE);
    }

    private float[] convertToArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    private int getActionIndex(RLAction action) {
        // Implement mapping from action to index
        return 0; // Placeholder
    }

    public NeuralNetwork getNetwork() {
        return network;
    }
} 