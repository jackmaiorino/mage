package mage.player.ai.rl;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class PythonMLBatchManager {

    private static final Logger logger = Logger.getLogger(PythonMLBatchManager.class.getName());

    static {
        // This class can log on every single decision; keep it quiet by default.
        // Enable for debugging with PY_BATCH_LOG_LEVEL=INFO (or FINE).
        String lvl = System.getenv().getOrDefault("PY_BATCH_LOG_LEVEL", "WARNING").toUpperCase();
        Level level;
        try {
            level = Level.parse(lvl);
        } catch (Exception ignored) {
            level = Level.WARNING;
        }
        logger.setLevel(level);
    }
    // Increase batch size to better utilise GPU; tune via env without code changes.
    // With multiple concurrent games, larger batches dramatically reduce Py4J overhead.
    private static final int MAX_BATCH_SIZE = EnvConfig.i32("PY_BATCH_MAX_SIZE", 256);      // hard cap per flush
    private static final int BATCH_TIMEOUT_MS = EnvConfig.i32("PY_BATCH_TIMEOUT_MS", 200); // flush window (ms)

    public static int getConfiguredMaxBatchSize() {
        return MAX_BATCH_SIZE;
    }

    public static int getConfiguredBatchTimeoutMs() {
        return BATCH_TIMEOUT_MS;
    }

    private final PythonEntryPoint entryPoint;
    private final Map<UUID, CompletableFuture<PredictionResult>> pendingPredictions;
    private final Map<UUID, CompletableFuture<Boolean>> pendingTraining;
    private final Object lock;
    private final Object py4jLock;
    private final List<PredictRequest> predictionQueue;
    private final java.util.concurrent.ScheduledExecutorService scheduler;

    PythonMLBatchManager(PythonEntryPoint entryPoint, Object py4jLock) {
        this.entryPoint = entryPoint;
        this.pendingPredictions = new ConcurrentHashMap<>();
        this.pendingTraining = new ConcurrentHashMap<>();
        this.lock = new Object();
        this.py4jLock = py4jLock;
        this.predictionQueue = new java.util.ArrayList<>();
        this.scheduler = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "PyBatchFlush");
            t.setDaemon(true);
            return t;
        });

        // Ensure resources freed on JVM shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(() -> scheduler.shutdown()));
    }

    // Thin holder for pending candidate-scoring request
    private static class PredictRequest {

        final java.util.UUID id;
        final StateSequenceBuilder.SequenceOutput state;
        final int[] candidateActionIds;      // [MAX_CANDIDATES]
        final float[][] candidateFeatures;   // [MAX_CANDIDATES][CAND_FEAT_DIM]
        final int[] candidateMask;           // [MAX_CANDIDATES] 1=valid,0=pad
        final String policyKey;             // "train" or "snap:<id>"
        final java.util.concurrent.CompletableFuture<PredictionResult> future;

        PredictRequest(java.util.UUID id,
                StateSequenceBuilder.SequenceOutput state,
                int[] candidateActionIds,
                float[][] candidateFeatures,
                int[] candidateMask,
                String policyKey,
                java.util.concurrent.CompletableFuture<PredictionResult> future) {
            this.id = id;
            this.state = state;
            this.candidateActionIds = candidateActionIds;
            this.candidateFeatures = candidateFeatures;
            this.candidateMask = candidateMask;
            this.policyKey = policyKey;
            this.future = future;
        }
    }

    public java.util.concurrent.CompletableFuture<PredictionResult> scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey) {
        java.util.UUID id = java.util.UUID.randomUUID();
        java.util.concurrent.CompletableFuture<PredictionResult> future = new java.util.concurrent.CompletableFuture<>();
        pendingPredictions.put(id, future);

        logger.info("PythonMLBatchManager.scoreCandidates() called with id: " + id);

        if (policyKey == null || policyKey.trim().isEmpty()) {
            policyKey = "train";
        }

        boolean flushNow = false;
        synchronized (lock) {
            predictionQueue.add(new PredictRequest(id, state, candidateActionIds, candidateFeatures, candidateMask, policyKey, future));
            logger.info("Added request to queue. Queue size: " + predictionQueue.size() + ", MAX_BATCH_SIZE: " + MAX_BATCH_SIZE);

            if (predictionQueue.size() >= MAX_BATCH_SIZE) {
                logger.info("Queue full, will flush immediately");
                flushNow = true;
            } else if (predictionQueue.size() == 1) {
                // first item – schedule timed flush
                logger.info("First item in queue, scheduling timed flush in " + BATCH_TIMEOUT_MS + "ms");
                scheduler.schedule(this::safeFlush, BATCH_TIMEOUT_MS, java.util.concurrent.TimeUnit.MILLISECONDS);
            }
        }

        if (flushNow) {
            flushQueue(true);
        }

        logger.info("Returning future for id: " + id);
        return future;
    }

    public java.util.concurrent.CompletableFuture<PredictionResult> scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask) {
        return scoreCandidates(state, candidateActionIds, candidateFeatures, candidateMask, "train");
    }

    // Called by scheduler – must handle race with manual flush
    private void safeFlush() {
        flushQueue(false);
    }

    private void flushQueue(boolean dueToFull) {
        // Copy & clear queue under lock, but do heavy work outside.
        List<PredictRequest> batch;
        synchronized (lock) {
            if (predictionQueue.isEmpty()) {
                return;
            }
            batch = new java.util.ArrayList<>(predictionQueue);
            predictionQueue.clear();
        }

        try {
            // Record flush stats before splitting by policy.
            // NOTE: dueToFull is the trigger cause, but because flush happens after releasing the enqueue lock,
            // the actual flushed batch can exceed MAX_BATCH_SIZE even if a timed flush fired.
            // Count as "full" if the flushed batch size hit the cap.
            boolean effectiveDueToFull = dueToFull || batch.size() >= MAX_BATCH_SIZE;
            MetricsCollector.getInstance().recordInferBatchFlush(batch.size(), effectiveDueToFull);
        } catch (Exception ignored) {
        }

        try {
            java.util.Map<String, java.util.List<PredictRequest>> byPolicy = new java.util.HashMap<>();
            for (PredictRequest req : batch) {
                String key = (req.policyKey == null || req.policyKey.trim().isEmpty()) ? "train" : req.policyKey;
                byPolicy.computeIfAbsent(key, k -> new java.util.ArrayList<>()).add(req);
            }

            for (java.util.Map.Entry<String, java.util.List<PredictRequest>> entry : byPolicy.entrySet()) {
                flushQueueForPolicy(entry.getKey(), entry.getValue());
            }

        } catch (Exception e) {
            logger.severe("Error during batch prediction: " + e.getMessage());
            for (PredictRequest req : batch) {
                req.future.completeExceptionally(e);
                pendingPredictions.remove(req.id);
            }
        }
    }

    private void flushQueueForPolicy(String policyKey, java.util.List<PredictRequest> batch) {
        // Build flat tensors
        int batchSize = batch.size();
        int seqLen = batch.get(0).state.getSequence().length;
        int dModel = batch.get(0).state.getSequence()[0].length;
        int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

        List<float[]> allSeq = new java.util.ArrayList<>(batchSize * seqLen);
        List<Integer> allMask = new java.util.ArrayList<>(batchSize * seqLen);
        List<Integer> allTokenIds = new java.util.ArrayList<>(batchSize * seqLen);
        List<float[]> allCandFeat = new java.util.ArrayList<>(batchSize * maxCandidates);
        List<Integer> allCandIds = new java.util.ArrayList<>(batchSize * maxCandidates);
        List<Integer> allCandMask = new java.util.ArrayList<>(batchSize * maxCandidates);

        for (PredictRequest req : batch) {
            allSeq.addAll(Arrays.asList(req.state.getSequence()));
            allMask.addAll(Arrays.stream(req.state.getMask()).boxed().collect(Collectors.toList()));
            allTokenIds.addAll(Arrays.stream(req.state.getTokenIds()).boxed().collect(Collectors.toList()));

            for (int i = 0; i < maxCandidates; i++) {
                allCandFeat.add(req.candidateFeatures[i]);
                allCandIds.add(req.candidateActionIds[i]);
                allCandMask.add(req.candidateMask[i]);
            }
        }

        byte[] seqBytes = convertFloatArraysToBytes(allSeq);
        byte[] maskBytes = convertIntegersToBytes(allMask);
        byte[] tokenIdsBytes = convertIntegersToBytes(allTokenIds);
        byte[] candFeatBytes = convertFloatArraysToBytes(allCandFeat);
        byte[] candIdsBytes = convertIntegersToBytes(allCandIds);
        byte[] candMaskBytes = convertIntegersToBytes(allCandMask);

        byte[] resultsBytes;
        synchronized (py4jLock) { // ensure exclusive Py4J channel (but don't block enqueueing)
            // Require GPULock for inference (mirrors learner burst locking).
            entryPoint.acquireGPULock();
            try {
                resultsBytes = entryPoint.scoreCandidatesPolicyFlat(
                        seqBytes,
                        maskBytes,
                        tokenIdsBytes,
                        candFeatBytes,
                        candIdsBytes,
                        candMaskBytes,
                        policyKey,
                        batchSize,
                        seqLen,
                        dModel,
                        maxCandidates,
                        candFeatDim);
            } finally {
                entryPoint.releaseGPULock();
            }
        }

        java.nio.ByteBuffer resBuf = java.nio.ByteBuffer.wrap(resultsBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int bi = 0; bi < batchSize; bi++) {
            float[] policy = new float[maxCandidates];
            for (int i = 0; i < maxCandidates; i++) {
                policy[i] = resBuf.getFloat();
            }
            float value = resBuf.getFloat();

            PredictRequest req = batch.get(bi);
            req.future.complete(new PredictionResult(policy, value));
            pendingPredictions.remove(req.id);
        }
    }

    public CompletableFuture<Boolean> train(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        UUID id = UUID.randomUUID();
        CompletableFuture<Boolean> future = new CompletableFuture<>();
        pendingTraining.put(id, future);

        try {
            int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

            // Convert training data to byte arrays (each TrainingData is one decision step)
            byte[] sequences = convertFloatArraysToBytes(trainingData.stream()
                    .map(d -> Arrays.asList(d.state.getSequence()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] masks = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.state.getMask()).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] tokenIds = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.state.getTokenIds()).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candFeat = convertFloatArraysToBytes(trainingData.stream()
                    .map(d -> Arrays.asList(d.candidateFeatures))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candIds = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.candidateActionIds).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candMask = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.candidateMask).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] chosenIndexBytes = convertIntegersToBytes(trainingData.stream()
                    .map(d -> d.chosenIndex)
                    .collect(Collectors.toList()));

            // Now passing immediate rewards instead of pre-computed returns
            // GAE (Generalized Advantage Estimation) will be computed in Python
            byte[] rewardsBytes = convertDoublesToBytes(rewards);

            // Get dimensions
            int batchSize = trainingData.size();
            int seqLen = trainingData.get(0).state.getSequence().length;
            int dModel = trainingData.get(0).state.getSequence()[0].length;

            // Call Python (candidate-based training with GAE)
            synchronized (py4jLock) {
                logger.info("Invoking trainCandidatesFlat: batchSize=" + batchSize + ", maxCandidates=" + maxCandidates);
                entryPoint.trainCandidatesFlat(
                        sequences,
                        masks,
                        tokenIds,
                        candFeat,
                        candIds,
                        candMask,
                        chosenIndexBytes,
                        rewardsBytes,  // Now passing immediate rewards
                        batchSize,
                        seqLen,
                        dModel,
                        maxCandidates,
                        candFeatDim);
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

    /**
     * Train on multiple episodes concatenated in one batch. The dones array marks episode ends (1=end).
     * This allows the Python learner to compute GAE/returns per-episode while still doing one larger
     * forward/backward pass for better GPU utilization.
     */
    public CompletableFuture<Boolean> trainMulti(
            List<StateSequenceBuilder.TrainingData> trainingData,
            List<Double> rewards,
            List<Integer> dones) {
        UUID id = UUID.randomUUID();
        CompletableFuture<Boolean> future = new CompletableFuture<>();
        pendingTraining.put(id, future);

        try {
            int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

            if (trainingData == null || trainingData.isEmpty()) {
                future.complete(true);
                pendingTraining.remove(id);
                return future;
            }
            if (rewards == null || rewards.size() != trainingData.size()) {
                throw new IllegalArgumentException("rewards size must match trainingData size");
            }
            if (dones == null || dones.size() != trainingData.size()) {
                throw new IllegalArgumentException("dones size must match trainingData size");
            }

            byte[] sequences = convertFloatArraysToBytes(trainingData.stream()
                    .map(d -> Arrays.asList(d.state.getSequence()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] masks = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.state.getMask()).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
            byte[] tokenIds = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.state.getTokenIds()).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candFeat = convertFloatArraysToBytes(trainingData.stream()
                    .map(d -> Arrays.asList(d.candidateFeatures))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candIds = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.candidateActionIds).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] candMask = convertIntegersToBytes(trainingData.stream()
                    .map(d -> Arrays.stream(d.candidateMask).boxed().collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            byte[] chosenIndexBytes = convertIntegersToBytes(trainingData.stream()
                    .map(d -> d.chosenIndex)
                    .collect(Collectors.toList()));

            byte[] rewardsBytes = convertDoublesToBytes(rewards);
            byte[] donesBytes = convertIntegersToBytes(dones);

            int batchSize = trainingData.size();
            int seqLen = trainingData.get(0).state.getSequence().length;
            int dModel = trainingData.get(0).state.getSequence()[0].length;

            synchronized (py4jLock) {
                logger.info("Invoking trainCandidatesMultiFlat: batchSize=" + batchSize + ", maxCandidates=" + maxCandidates);
                entryPoint.trainCandidatesMultiFlat(
                        sequences,
                        masks,
                        tokenIds,
                        candFeat,
                        candIds,
                        candMask,
                        chosenIndexBytes,
                        rewardsBytes,
                        donesBytes,
                        batchSize,
                        seqLen,
                        dModel,
                        maxCandidates,
                        candFeatDim);
            }

            future.complete(true);
            pendingTraining.remove(id);
        } catch (Exception e) {
            logger.severe("Error during multi-episode training: " + e.getMessage());
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
}
