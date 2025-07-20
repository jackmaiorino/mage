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

public class PythonMLBatchManager {

    private static final Logger logger = Logger.getLogger(PythonMLBatchManager.class.getName());
    private static final int MAX_TRAJECTORY_LENGTH = 100;
    // Increase batch size to better utilise GPU; shorter timeout keeps latency low
    private static final int MAX_BATCH_SIZE = 32;         // hard cap per flush
    private static final int BATCH_TIMEOUT_MS = 4;        // flush window (ms)

    private final PythonEntryPoint entryPoint;
    private final Map<UUID, CompletableFuture<PredictionResult>> pendingPredictions;
    private final Map<UUID, CompletableFuture<Boolean>> pendingTraining;
    private final Map<UUID, List<TrainingStep>> trajectoryBuffer;
    private final Object lock;
    private final List<PredictRequest> predictionQueue;
    private final java.util.concurrent.ScheduledExecutorService scheduler;

    private static volatile PythonMLBatchManager instance;

    public static PythonMLBatchManager getInstance(PythonEntryPoint entryPoint) {
        if (instance == null) {
            synchronized (PythonMLBatchManager.class) {
                if (instance == null) {
                    instance = new PythonMLBatchManager(entryPoint);
                }
            }
        }
        return instance;
    }

    private PythonMLBatchManager(PythonEntryPoint entryPoint) {
        this.entryPoint = entryPoint;
        this.pendingPredictions = new ConcurrentHashMap<>();
        this.pendingTraining = new ConcurrentHashMap<>();
        this.trajectoryBuffer = new ConcurrentHashMap<>();
        this.lock = new Object();
        this.predictionQueue = new java.util.ArrayList<>();
        this.scheduler = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "PyBatchFlush");
            t.setDaemon(true);
            return t;
        });

        // Ensure resources freed on JVM shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(() -> scheduler.shutdown()));
    }

    // Thin holder for pending prediction
    private static class PredictRequest {

        final java.util.UUID id;
        final StateSequenceBuilder.SequenceOutput state;
        final int validActions;
        final java.util.concurrent.CompletableFuture<PredictionResult> future;

        PredictRequest(java.util.UUID id, StateSequenceBuilder.SequenceOutput state, int validActions,
                java.util.concurrent.CompletableFuture<PredictionResult> future) {
            this.id = id;
            this.state = state;
            this.validActions = validActions;
            this.future = future;
        }
    }

    public java.util.concurrent.CompletableFuture<PredictionResult> predict(StateSequenceBuilder.SequenceOutput state, int validActions) {
        java.util.UUID id = java.util.UUID.randomUUID();
        java.util.concurrent.CompletableFuture<PredictionResult> future = new java.util.concurrent.CompletableFuture<>();
        pendingPredictions.put(id, future);

        synchronized (lock) {
            predictionQueue.add(new PredictRequest(id, state, validActions, future));

            if (predictionQueue.size() >= MAX_BATCH_SIZE) {
                flushQueue();
            } else if (predictionQueue.size() == 1) {
                // first item – schedule timed flush
                scheduler.schedule(this::safeFlush, BATCH_TIMEOUT_MS, java.util.concurrent.TimeUnit.MILLISECONDS);
            }
        }

        return future;
    }

    // Called by scheduler – must handle race with manual flush
    private void safeFlush() {
        synchronized (lock) {
            if (!predictionQueue.isEmpty()) {
                flushQueue();
            }
        }
    }

    private void flushQueue() {
        // Copy & clear queue under lock
        List<PredictRequest> batch = new java.util.ArrayList<>(predictionQueue);
        predictionQueue.clear();

        try {
            // Build flat tensors
            int batchSize = batch.size();
            int seqLen = batch.get(0).state.getSequence().size();
            int dModel = batch.get(0).state.getSequence().get(0).length;
            int maxActions = 15; // fixed

            List<float[]> allSeq = new java.util.ArrayList<>(batchSize * seqLen);
            List<Integer> allMask = new java.util.ArrayList<>(batchSize * seqLen);
            List<Integer> allActionMask = new java.util.ArrayList<>(batchSize * maxActions);

            for (PredictRequest req : batch) {
                allSeq.addAll(req.state.getSequence());
                allMask.addAll(req.state.getMask());

                // Build action mask row
                for (int i = 0; i < maxActions; i++) {
                    allActionMask.add(i < req.validActions ? 1 : 0);
                }
            }

            byte[] seqBytes = convertFloatArraysToBytes(allSeq);
            byte[] maskBytes = convertIntegersToBytes(allMask);
            byte[] actionMaskBytes = convertIntegersToBytes(allActionMask);

            byte[] resultsBytes;
            // Call Python once for the batch
            synchronized (lock) { // ensure exclusive channel
                resultsBytes = entryPoint.predictBatchFlat(seqBytes, maskBytes, actionMaskBytes, batchSize, seqLen, dModel, maxActions);
            }

            java.nio.ByteBuffer resBuf = java.nio.ByteBuffer.wrap(resultsBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            int floatsPerItem = 16; // 15 policy + 1 value

            for (int bi = 0; bi < batchSize; bi++) {
                float[] policy = new float[15];
                for (int i = 0; i < 15; i++) {
                    policy[i] = resBuf.getFloat();
                }
                float value = resBuf.getFloat();

                PredictRequest req = batch.get(bi);
                req.future.complete(new PredictionResult(policy, value));
                pendingPredictions.remove(req.id);
            }

        } catch (Exception e) {
            logger.severe("Error during batch prediction: " + e.getMessage());
            for (PredictRequest req : batch) {
                req.future.completeExceptionally(e);
                pendingPredictions.remove(req.id);
            }
        }
    }

    public CompletableFuture<Boolean> train(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> discountedReturns) {
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
            byte[] policyIndices = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Math.floorMod(d.actionCombo.get(0), 15))
                    .collect(Collectors.toList()));

            // This is the main change: we now pass the calculated discounted returns
            // to be used as the target for the value function.
            byte[] discountedReturnsBytes = convertDoublesToBytes(discountedReturns);

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
            int maxActions = 15; // fixed action space size

            // Call Python - note the signature change (no final reward param)
            synchronized (lock) {
                logger.info("Invoking trainFlat: batchSize=" + batchSize + ", maxActions=" + maxActions);
                entryPoint.trainFlat(sequences, masks, policyIndices, discountedReturnsBytes,
                        actionTypes, actionCombos, batchSize, seqLen, dModel, maxActions);
            }

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
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float[] array : data) {
            for (float value : array) {
                buffer.putFloat(value);
            }
        }
        return buffer.array();
    }

    private byte[] convertIntegersToBytes(List<Integer> data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.size() * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (Integer value : data) {
            buffer.putInt(value);
        }
        return buffer.array();
    }

    private byte[] convertDoublesToBytes(List<Double> data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.size() * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (Double value : data) {
            buffer.putFloat(value.floatValue());
        }
        return buffer.array();
    }

    public static class PredictionResult {

        public final float[] policyScores;
        public final float valueScores;

        public PredictionResult(float[] policyScores, float valueScores) {
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
