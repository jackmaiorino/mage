package mage.player.ai.rl;

import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.players.Player;
import mage.util.RandomUtil;
import mage.abilities.ActivatedAbility;
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
    public static final int STATE_SIZE = 45;  // 5 player state + 40 battlefield state
    public static final int ACTION_SIZE = 10; // Adjust based on number of possible actions

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
        List<Float> qValues = new ArrayList<>();
        float[] stateVector = convertToArray(state.getStateVector());
        
        for (RLAction action : actions) {
            float[] actionVector = action.toFeatureVector();
            qValues.add(network.predict(stateVector, actionVector));
        }
        return qValues;
    }

    public void update(RLState state, RLAction action, double reward, RLState nextState) {
        float[] currentStateVector = convertToArray(state.getStateVector());
        float[] actionVector = action.toFeatureVector();
        float currentQ = network.predict(currentStateVector, actionVector);
        
        // Get max Q value for next state
        List<RLAction> nextActions = getPossibleActions(nextState);
        float maxNextQ = 0.0f;
        float[] nextStateVector = convertToArray(nextState.getStateVector());
        
        for (RLAction nextAction : nextActions) {
            float q = network.predict(nextStateVector, nextAction.toFeatureVector());
            maxNextQ = Math.max(maxNextQ, q);
        }
        
        float targetQ = (float)(reward + DISCOUNT_FACTOR * maxNextQ);
        
        // Train network with updated Q value
        float[][] inputs = new float[][] { currentStateVector };
        float[][] targets = new float[][] { new float[] { targetQ } };
        network.train(inputs, targets, LEARNING_RATE);
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

    public List<Float> evaluateAbilities(RLState state, List<ActivatedAbility> abilities) {
        List<Float> scores = new ArrayList<>();
        float[] stateVector = state.toFeatureVector();
        
        for (ActivatedAbility ability : abilities) {
            RLAction action = new RLAction(RLAction.ActionType.ACTIVATE_ABILITY, ability);
            float[] actionVector = action.toFeatureVector();
            scores.add(network.predict(stateVector, actionVector));
        }
        return scores;
    }

    public float getActionThreshold() {
        return 0.2f; // Threshold can be tuned based on training
    }

    private float predictQValue(RLState state, RLAction action) {
        // Use neural network to predict Q-value for this state-action pair
        return network.predict(state.toFeatureVector(), action.toFeatureVector());
    }

    private List<RLAction> getPossibleActions(RLState state) {
        if (state == null || state.getGame() == null) {
            return new ArrayList<>();
        }
        return ComputerPlayerRL.getPlayableActions(state.getGame(), 
               (ComputerPlayerRL)state.getGame().getPlayer(playerId));
    }
} 