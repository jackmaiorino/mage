package mage.player.ai.rl;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class RLModel implements Serializable {
    private static final Logger logger = Logger.getLogger(RLModel.class);
    private final TransformerNeuralNetwork network;
    private final BatchPredictionRequest batchPredictor;
    public static final double EXPLORATION_RATE = 0.5;
    private static final double DISCOUNT_FACTOR = 0.95;
    private static final long serialVersionUID = 1L;
    // TODO: Eliminate the need for a MAX_ACTIONS by finding ways to indicate multiple copies of same card efficiently
    public static final int MAX_ACTIONS = 25;
    public static final int MAX_OPTIONS = 15;
    public static final int OUTPUT_SIZE = (MAX_ACTIONS) * (MAX_OPTIONS);
    private boolean IS_TRAINING = false;

    public RLModel(boolean training) {
        double EPSILON = 0.05;
        int[] headSizes = new int[]{OUTPUT_SIZE, MAX_ACTIONS, MAX_OPTIONS};
        network = new TransformerNeuralNetwork(
                StateSequenceBuilder.DIM_PER_TOKEN,
                StateSequenceBuilder.MAX_LEN,
                headSizes,
                EPSILON);

        try {
            network.load(RLTrainer.MODEL_FILE_PATH);
        } catch (Exception ex) {
            logger.warn("Could not load existing model, starting fresh.");
        }

        this.IS_TRAINING = training;
        // Initialize batch predictor with reasonable defaults
        this.batchPredictor = BatchPredictionRequest.getInstance(1, 10000, TimeUnit.MILLISECONDS);
    }

    public TransformerNeuralNetwork getNetwork() {
        return network;
    }

    public void saveModel(String filePath) {
        try {
            network.save(filePath);
        } catch (IOException e) {
            logger.error("Failed to save network.", e);
        }
    }

    public INDArray predictDistribution(StateSequenceBuilder.SequenceOutput state, boolean isExploration) {
        if (IS_TRAINING) {
            // During training, use batch prediction for better performance
            try {
                return batchPredictor.predict(state.sequence, state.mask);
            } catch (InterruptedException e) {
                logger.error("Batch prediction interrupted", e);
                // Fall back to direct prediction if batch fails
                return network.predict(state, TransformerNeuralNetwork.AskType.CAST, isExploration);
            }
        } else {
            // For single-game inference, use direct prediction
            return network.predict(state, TransformerNeuralNetwork.AskType.CAST, false);
        }
    }

    // ---------------------------------------------------------------------
    //  SIMPLE SUPERVISED-LIKE UPDATE (prototype)
    // ---------------------------------------------------------------------
    /**
     * Extremely simplified learner: for each state that contains a valid
     * {@code actionIndex}, create a one-hot target over the CAST head where the
     * chosen index gets probability&nbsp;1.  If {@code reward} is negative the
     * sample is ignored; if it is positive we fit one step.
     * <p>
     * This is <em>not</em> proper Q-learning – it is merely behavioural
     * cloning weighted by outcome, but it is enough to exercise the wiring so
     * that the end-to-end pipeline runs without crashing.
     */
    public void updateBatch(List<StateSequenceBuilder.SequenceOutput> states,
                            List<Double> rewards,
                            List<StateSequenceBuilder.SequenceOutput> nextStates) {

        if (states == null || rewards == null || states.isEmpty()) {
            return; // nothing to learn from
        }

        int batchSize = 0;
        for (int i = 0; i < states.size() && i < rewards.size(); i++) {
            if (states.get(i) != null && states.get(i).actionIndex >= 0 && rewards.get(i) > 0) {
                batchSize++;
            }
        }

        if (batchSize == 0) {
            return; // no positive-reward samples to train on
        }

        int D = StateSequenceBuilder.DIM_PER_TOKEN;
        int L = StateSequenceBuilder.MAX_LEN;

        org.nd4j.linalg.api.ndarray.INDArray seqBatch  = org.nd4j.linalg.factory.Nd4j.create(new int[]{batchSize, D, L}, 'c');
        org.nd4j.linalg.api.ndarray.INDArray maskBatch = org.nd4j.linalg.factory.Nd4j.create(new int[]{batchSize, L},   'c');

        org.nd4j.linalg.api.ndarray.INDArray castTargets = org.nd4j.linalg.factory.Nd4j.zeros(batchSize, OUTPUT_SIZE);

        // dummy heads (unused in this prototype)
        org.nd4j.linalg.api.ndarray.INDArray atkTargets  = org.nd4j.linalg.factory.Nd4j.zeros(batchSize, MAX_ACTIONS);
        org.nd4j.linalg.api.ndarray.INDArray blkTargets  = org.nd4j.linalg.factory.Nd4j.zeros(batchSize, MAX_OPTIONS);

        int row = 0;
        for (int i = 0; i < states.size() && i < rewards.size(); i++) {
            StateSequenceBuilder.SequenceOutput s = states.get(i);
            if (s == null || s.actionIndex < 0) { continue; }

            double r = rewards.get(i);
            if (r <= 0) { continue; } // ignore negative reward in this simple impl

            // Ensure the sequence has the correct shape
            if (s.sequence.shape()[1] != D || s.sequence.shape()[2] != L) {
                logger.error("Invalid sequence shape: " + java.util.Arrays.toString(s.sequence.shape()) + 
                           ", expected: [1, " + D + ", " + L + "]");
                continue;
            }

            // Copy the sequence data - ensure we're copying from the first batch dimension
            seqBatch.putSlice(row, s.sequence.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()));

            // Copy the mask data
            for (int l = 0; l < L; l++) {
                maskBatch.putScalar(row, l, s.mask.getDouble(0, l));
            }

            // one-hot
            castTargets.putScalar(row, s.actionIndex, 1.0);
            row++;
        }

        // if due to skipping negatives we ended up with 0 rows – just exit
        if (row == 0) return;

        // trim batch arrays if some entries were skipped
        if (row < batchSize) {
            seqBatch  = seqBatch.getRows(java.util.stream.IntStream.range(0, row).toArray());
            maskBatch = maskBatch.getRows(java.util.stream.IntStream.range(0, row).toArray());
            castTargets = castTargets.getRows(java.util.stream.IntStream.range(0, row).toArray());
            atkTargets  = atkTargets.getRows(java.util.stream.IntStream.range(0, row).toArray());
            blkTargets  = blkTargets.getRows(java.util.stream.IntStream.range(0, row).toArray());
        }

        try {
            network.fit(seqBatch, maskBatch, castTargets, atkTargets, blkTargets);
        } catch (Exception e) {
            logger.error("Error during network.fit in updateBatch", e);
        }
    }
} 