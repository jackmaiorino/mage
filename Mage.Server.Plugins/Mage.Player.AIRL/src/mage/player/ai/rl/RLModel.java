package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.factory.Nd4j;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final TransformerNeuralNetwork network;
    private final BatchPredictionRequest batchPredictor;
    public static final double EXPLORATION_RATE = 0.5;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    public static final int CAST_OPTIONS = 50;
    public static final int ATK_OPTIONS = 10;
    public static final int BLK_OPTIONS = 10;
    private boolean IS_TRAINING = false;

    // Experience replay buffer
    private static final int REPLAY_BUFFER_SIZE = 10000;
    private static final int BATCH_SIZE = 32;
    private final List<Experience> replayBuffer;
    private int bufferPosition = 0;

    // Experience class to store transitions
    private static class Experience {
        final StateSequenceBuilder.SequenceOutput state;
        final int action;
        final double reward;
        final StateSequenceBuilder.SequenceOutput nextState;
        final boolean isTerminal;
        final INDArray currentQValues;  // Store Q-values from prediction

        Experience(StateSequenceBuilder.SequenceOutput state, int action, double reward, 
                  StateSequenceBuilder.SequenceOutput nextState, boolean isTerminal,
                  INDArray currentQValues) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.isTerminal = isTerminal;
            this.currentQValues = currentQValues;
        }
    }

    public RLModel(boolean training) {
        double EPSILON = 0.05;
        int[] headSizes = new int[]{CAST_OPTIONS, ATK_OPTIONS, BLK_OPTIONS};
        network = new TransformerNeuralNetwork(
                StateSequenceBuilder.DIM_PER_TOKEN,
                StateSequenceBuilder.MAX_LEN,
                headSizes,
                EPSILON);

        try {
            network.load(RLTrainer.MODEL_FILE_PATH);
        } catch (Exception ex) {
            System.out.println("Could not load existing model, starting fresh.");
        }

        this.IS_TRAINING = training;
        this.batchPredictor = BatchPredictionRequest.getInstance(10000, TimeUnit.MILLISECONDS);
        this.replayBuffer = new ArrayList<>(REPLAY_BUFFER_SIZE);
    }

    public TransformerNeuralNetwork getNetwork() {
        return network;
    }

    public void saveModel(String filePath) {
        try {
            network.save(filePath);
        } catch (IOException e) {
            RLTrainer.threadLocalLogger.get().error("Failed to save network.", e);
        }
    }

    public INDArray predictDistribution(StateSequenceBuilder.SequenceOutput state, boolean isExploration) {
        // Update epsilon if we're in exploration mode
        if (isExploration) {
            network.updateEpsilon();
            RLTrainer.threadLocalLogger.get().info("Epsilon updated: " + network.getEpsilon());
        }

        INDArray predictions;
        // First check if we should explore
        if (isExploration && network.getRandom().nextDouble() < network.getEpsilon()) {
            // If exploring, generate random output
            RLTrainer.threadLocalLogger.get().info("Random exploration with epsilon: " + network.getEpsilon());
            predictions = network.randomOutput(state.askType);
        } else if (IS_TRAINING) {
            // During training, use batch prediction for better performance
            try {
                predictions = batchPredictor.predict(state.sequence, state.mask);
            } catch (InterruptedException e) {
                throw new RuntimeException("Batch prediction interrupted", e);
            }
        } else {
            // For single-game inference, use direct prediction
            predictions = network.predict(state, TransformerNeuralNetwork.AskType.CAST, false);
        }
        
        // Create a mask for valid options
        INDArray mask = Nd4j.zeros(CAST_OPTIONS);
        int numOptions = state.getNumOptions();
        for (int i = 0; i < Math.min(numOptions, CAST_OPTIONS); i++) {
            mask.putScalar(i, 1.0);
        }
        
        // Apply mask to predictions
        predictions = predictions.mul(mask);
        
        // Log the Q-values
        StringBuilder sb = new StringBuilder();
        sb.append(IS_TRAINING ? "Batch " : "Direct ").append("Q-values (numOptions=").append(numOptions).append("): [");
        for (int i = 0; i < Math.min(numOptions, CAST_OPTIONS); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", predictions.getDouble(i)));
        }
        sb.append("]");
        RLTrainer.threadLocalLogger.get().info(sb.toString());
        
        return predictions;
    }

    // Add experience to replay buffer
    private void addExperience(StateSequenceBuilder.SequenceOutput state, int action, 
                             double reward, StateSequenceBuilder.SequenceOutput nextState, 
                             boolean isTerminal, INDArray currentQValues) {
        Experience exp = new Experience(state, action, reward, nextState, isTerminal, currentQValues);
        if (replayBuffer.size() < REPLAY_BUFFER_SIZE) {
            replayBuffer.add(exp);
        } else {
            replayBuffer.set(bufferPosition, exp);
            bufferPosition = (bufferPosition + 1) % REPLAY_BUFFER_SIZE;
        }
    }

    // Sample random batch from replay buffer
    private List<Experience> sampleBatch() {
        if (replayBuffer.size() < BATCH_SIZE) {
            return new ArrayList<>(replayBuffer);
        }
        List<Experience> batch = new ArrayList<>(BATCH_SIZE);
        for (int i = 0; i < BATCH_SIZE; i++) {
            int idx = (int) (Math.random() * replayBuffer.size());
            batch.add(replayBuffer.get(idx));
        }
        return batch;
    }

    public void updateBatch(List<StateSequenceBuilder.SequenceOutput> states, List<Double> rewards) {
        if (states == null || rewards == null || states.isEmpty()) {
            return;
        }

        // Collect valid experiences
        List<INDArray> batchSequences = new ArrayList<>();
        List<INDArray> batchMasks = new ArrayList<>();
        List<INDArray> batchCastTargets = new ArrayList<>();
        List<INDArray> batchAtkTargets = new ArrayList<>();
        List<INDArray> batchBlkTargets = new ArrayList<>();

        // Process each state
        for (int i = 0; i < states.size() && i < rewards.size(); i++) {
            StateSequenceBuilder.SequenceOutput state = states.get(i);
            if (state == null || state.targetIndices.isEmpty()) continue;

            double reward = rewards.get(i);
            StateSequenceBuilder.SequenceOutput nextState = i + 1 < states.size() ? states.get(i + 1) : null;
            boolean isTerminal = nextState == null;
            
            // Get current Q-values from state
            INDArray currentQValues = state.currentQValues;
            if (currentQValues == null) {
                throw new RuntimeException("No Q-values stored for next state, skipping experience");
            }

            // Create target array with same shape as current Q-values
            INDArray targets;
            switch (state.askType) {
                case CAST:
                    targets = Nd4j.zeros(CAST_OPTIONS);
                    break;
                case ATTACK:
                    targets = Nd4j.zeros(ATK_OPTIONS);
                    break;
                case BLOCK:
                    targets = Nd4j.zeros(BLK_OPTIONS);
                    break;
                default:
                    throw new IllegalStateException("Unknown ask type: " + state.askType);
            }

            // Get next state Q-values if not terminal
            INDArray nextStateQ = null;
            if (!isTerminal) {
                nextStateQ = nextState.currentQValues;
                if (nextStateQ == null) {
                    throw new RuntimeException("No Q-values stored for next state, skipping experience");
                }
            }

            // First, copy all current Q-values to targets
            for (int j = 0; j < targets.length(); j++) {
                targets.putScalar(j, currentQValues.getDouble(j));
            }

            // Then update only the targets we acted on with new Q-values
            for (int targetIndex : state.targetIndices) {
                double targetQValue;
                if (isTerminal) {
                    targetQValue = reward;
                } else {
                    // For each target, we want to consider its contribution to the next state
                    // We'll use the average of the next state's Q-values weighted by their relative importance
                    double nextStateValue = 0;
                    if (!nextState.targetIndices.isEmpty()) {
                        double totalNextQ = 0;
                        for (int nextTargetIndex : nextState.targetIndices) {
                            totalNextQ += nextStateQ.getDouble(nextTargetIndex);
                        }
                        nextStateValue = totalNextQ / nextState.targetIndices.size();
                    } else {
                        // If no targets in next state, use max Q-value
                        nextStateValue = nextStateQ.maxNumber().doubleValue();
                    }
                    targetQValue = reward + DISCOUNT_FACTOR * nextStateValue;
                }
                // Update target array with individual target Q-value
                targets.putScalar(targetIndex, targetQValue);
            }

            // Add to batch
            batchSequences.add(state.sequence);
            batchMasks.add(state.mask);

            // Add targets to appropriate list based on askType
            switch (state.askType) {
                case CAST:
                    batchCastTargets.add(targets);
                    batchAtkTargets.add(Nd4j.zeros(ATK_OPTIONS));
                    batchBlkTargets.add(Nd4j.zeros(BLK_OPTIONS));
                    break;
                case ATTACK:
                    batchCastTargets.add(Nd4j.zeros(CAST_OPTIONS));
                    batchAtkTargets.add(targets);
                    batchBlkTargets.add(Nd4j.zeros(BLK_OPTIONS));
                    break;
                case BLOCK:
                    batchCastTargets.add(Nd4j.zeros(CAST_OPTIONS));
                    batchAtkTargets.add(Nd4j.zeros(ATK_OPTIONS));
                    batchBlkTargets.add(targets);
                    break;
            }
        }

        // Skip if no valid experiences
        if (batchSequences.isEmpty()) {
            return;
        }

        // Convert lists to NDArrays
        INDArray batchSequence = Nd4j.concat(0, batchSequences.toArray(new INDArray[0]));
        INDArray batchMask = Nd4j.concat(0, batchMasks.toArray(new INDArray[0]));
        
        // Reshape targets to [batch_size, num_actions]
        INDArray batchCastTarget = Nd4j.concat(0, batchCastTargets.toArray(new INDArray[0])).reshape(batchCastTargets.size(), -1);
        INDArray batchAtkTarget = Nd4j.concat(0, batchAtkTargets.toArray(new INDArray[0])).reshape(batchAtkTargets.size(), -1);
        INDArray batchBlkTarget = Nd4j.concat(0, batchBlkTargets.toArray(new INDArray[0])).reshape(batchBlkTargets.size(), -1);

        // Log final batch shapes
        logger.info("Final batch sequence shape: " + Arrays.toString(batchSequence.shape()));
        logger.info("Final batch mask shape: " + Arrays.toString(batchMask.shape()));
        logger.info("Final batch cast target shape: " + Arrays.toString(batchCastTarget.shape()));
        logger.info("Final batch atk target shape: " + Arrays.toString(batchAtkTarget.shape()));
        logger.info("Final batch blk target shape: " + Arrays.toString(batchBlkTarget.shape()));

        // Update network with entire batch
        try {
            network.fit(batchSequence, batchMask, batchCastTarget, batchAtkTarget, batchBlkTarget);
        } catch (Exception e) {
            logger.error("Error during network.fit in updateBatch", e);
            throw e; // Re-throw to see full stack trace
        }
    }
} 