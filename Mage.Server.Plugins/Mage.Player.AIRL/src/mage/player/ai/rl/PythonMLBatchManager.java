package mage.player.ai.rl;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Singleton class that manages batching of requests to the Python ML server to
 * ensure single-threaded communication. This helps prevent race conditions and
 * improves performance by batching requests.
 */
public class PythonMLBatchManager {

    private static final Logger logger = Logger.getLogger(PythonMLBatchManager.class.getName());
    private static final int MAX_TRAJECTORY_LENGTH = 100;
    private static final int MAX_BATCH_SIZE = 8;

    private final PythonEntryPoint entryPoint;
    private final Map<UUID, CompletableFuture<PredictionResult>> pendingPredictions;
    private final Map<UUID, CompletableFuture<Boolean>> pendingTraining;
    private final Map<UUID, List<TrainingStep>> trajectoryBuffer;
    private final Object lock = new Object();

    private PythonMLBatchManager(PythonEntryPoint entryPoint) {
        this.entryPoint = entryPoint;
        this.pendingPredictions = new ConcurrentHashMap<>();
        this.pendingTraining = new ConcurrentHashMap<>();
        this.trajectoryBuffer = new ConcurrentHashMap<>();
    }

    public static PythonMLBatchManager getInstance(PythonEntryPoint entryPoint) {
        return new PythonMLBatchManager(entryPoint);
    }

    public CompletableFuture<PredictionResult> predict(StateSequenceBuilder.SequenceOutput state) {
        UUID id = UUID.randomUUID();
        CompletableFuture<PredictionResult> future = new CompletableFuture<>();
        pendingPredictions.put(id, future);

        try {
            // Convert state to byte arrays
            byte[] sequences = convertFloatArraysToBytes(state.getSequence());
            byte[] masks = convertIntegersToBytes(state.getMask());

            // Get dimensions
            int batchSize = 1;
            int seqLen = state.getSequence().size();
            int dModel = state.getSequence().get(0).length;

            // Call Python and get results as byte array
            byte[] resultsBytes = entryPoint.predictBatchFlat(sequences, masks, batchSize, seqLen, dModel);

            // Parse the byte array into policy and value scores
            // Each float is 4 bytes, and we have 15 policy scores + 1 value score per batch item
            float[] policyScoresArray = new float[15];
            float[] valueScoresArray = new float[1];

            ByteBuffer resBuf = ByteBuffer.wrap(resultsBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < 15; i++) {
                policyScoresArray[i] = resBuf.getFloat();
            }
            valueScoresArray[0] = resBuf.getFloat();

            // Convert to INDArray
            INDArray policyScores = Nd4j.create(policyScoresArray);
            INDArray valueScores = Nd4j.create(valueScoresArray);

            // Complete future
            future.complete(new PredictionResult(policyScores, valueScores));
            pendingPredictions.remove(id);

        } catch (Exception e) {
            logger.severe("Error during prediction: " + e.getMessage());
            future.completeExceptionally(e);
            pendingPredictions.remove(id);
        }

        return future;
    }

    public CompletableFuture<Boolean> train(List<StateSequenceBuilder.TrainingData> trainingData, double reward) {
        UUID id = UUID.randomUUID();
        CompletableFuture<Boolean> future = new CompletableFuture<>();
        pendingTraining.put(id, future);

        try {
            // Convert training data to byte arrays
            byte[] sequences = convertFloatArraysToBytes(trainingData.stream()
                    .map(d -> d.stateActionPair.getSequence())
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] masks = convertIntegersToBytes(trainingData.stream()
                    .map(d -> d.stateActionPair.getMask())
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] policyScores = convertDoublesToBytes(trainingData.stream()
                    .map(d -> d.policyScore)
                    .collect(Collectors.toList()));
            byte[] valueScores = convertDoublesToBytes(trainingData.stream()
                    .map(d -> d.valueScore)
                    .collect(Collectors.toList()));
            byte[] actionTypes = convertIntegersToBytes(trainingData.stream()
                    .map(d -> d.actionType.ordinal())
                    .collect(Collectors.toList()));
            byte[] actionCombos = convertIntegersToBytes(trainingData.stream()
                    .map(d -> d.actionCombo)
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            // Get dimensions
            int batchSize = trainingData.size();
            int seqLen = trainingData.get(0).stateActionPair.getSequence().size();
            int dModel = trainingData.get(0).stateActionPair.getSequence().get(0).length;
            int maxActions = trainingData.stream()
                    .mapToInt(d -> d.actionCombo.size())
                    .max()
                    .orElse(0);

            // Call Python
            entryPoint.trainFlat(sequences, masks, policyScores, valueScores,
                    actionTypes, actionCombos, batchSize, seqLen, dModel, maxActions, (float) reward);

            // Complete future with success
            future.complete(true);
            pendingTraining.remove(id);

        } catch (Exception e) {
            logger.severe("Error during training: " + e.getMessage());
            future.completeExceptionally(e);
            pendingTraining.remove(id);
        }

        return future;
    }

    private byte[] convertFloatArraysToBytes(List<float[]> data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.size() * data.get(0).length * 4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        for (float[] array : data) {
            for (float value : array) {
                buffer.putFloat(value);
            }
        }
        return buffer.array();
    }

    private byte[] convertIntegersToBytes(List<Integer> data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.size() * 4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        for (Integer value : data) {
            buffer.putFloat(value.floatValue());
        }
        return buffer.array();
    }

    private byte[] convertDoublesToBytes(List<Double> data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.size() * 4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        for (Double value : data) {
            buffer.putFloat(value.floatValue());
        }
        return buffer.array();
    }

    public static class PredictionResult {

        public final INDArray policyScores;
        public final INDArray valueScores;

        public PredictionResult(INDArray policyScores, INDArray valueScores) {
            this.policyScores = policyScores;
            this.valueScores = valueScores;
        }
    }

    private static class TrainingStep {

        public final StateSequenceBuilder.SequenceOutput state;
        public final double policyScore;
        public final double valueScore;
        public final StateSequenceBuilder.ActionType actionType;
        public final List<Integer> actionCombo;

        public TrainingStep(StateSequenceBuilder.TrainingData data) {
            this.state = data.stateActionPair;
            this.policyScore = data.policyScore;
            this.valueScore = data.valueScore;
            this.actionType = data.actionType;
            this.actionCombo = data.actionCombo;
        }
    }
}
