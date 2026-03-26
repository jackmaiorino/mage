package mage.player.ai.rl;

import ai.onnxruntime.*;
import org.apache.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-process ONNX Runtime inference with request batching.
 * Game runner threads enqueue requests; a dedicated flush thread groups them
 * by (headId, seqLen) and runs batched session.run() calls.
 */
public final class OnnxInferenceModel implements PythonModel {

    private static final Logger logger = Logger.getLogger(OnnxInferenceModel.class);
    private static final String[] HEAD_IDS = {"action", "target", "card_select", "attack", "block"};

    private static final int BATCH_TIMEOUT_MS = EnvConfig.i32("ONNX_BATCH_TIMEOUT_MS", 20);
    private static final int BATCH_MAX_SIZE = EnvConfig.i32("ONNX_BATCH_MAX_SIZE", 64);
    private static final int SCORE_TIMEOUT_MS = EnvConfig.i32("ONNX_SCORE_TIMEOUT_MS", 30000);
    private static final int DIM = StateSequenceBuilder.DIM_PER_TOKEN;
    private static final int MAX_CAND = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
    private static final int CAND_DIM = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

    private final OrtEnvironment env;
    private final OrtSession.SessionOptions sessionOpts;
    private final Map<String, OrtSession> sessions = new ConcurrentHashMap<>();
    private final Path onnxDir;
    private PythonModel trainingDelegate;

    // --- Batching infrastructure ---
    private final List<ScoreRequest> queue = new ArrayList<>();
    private final Object queueLock = new Object();
    private ScheduledExecutorService flushScheduler;
    private ExecutorService inferPool; // parallel head-group inference
    private ScheduledFuture<?> pendingFlush;
    private final AtomicLong totalBatched = new AtomicLong();
    private final AtomicLong totalFlushes = new AtomicLong();

    private static class BatchKey {
        final String headId;
        final int seqLen; // bucketed

        BatchKey(String headId, int seqLen) {
            this.headId = headId;
            this.seqLen = seqLen;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof BatchKey)) return false;
            BatchKey k = (BatchKey) o;
            return seqLen == k.seqLen && headId.equals(k.headId);
        }

        @Override
        public int hashCode() {
            return headId.hashCode() * 31 + seqLen;
        }
    }

    private static class ScoreRequest {
        final float[][] tokens;
        final int[] mask;
        final int[] tokenIds;
        final int[] candidateActionIds;
        final float[][] candidateFeatures;
        final int[] candidateMask;
        final int originalCandCount;
        final BatchKey key;
        final CompletableFuture<PythonMLBatchManager.PredictionResult> future;

        ScoreRequest(StateSequenceBuilder.SequenceOutput state,
                     int[] candidateActionIds, float[][] candidateFeatures,
                     int[] candidateMask, String headId) {
            this.tokens = state.getSequence();
            this.mask = state.getMask();
            this.tokenIds = state.getTokenIds();
            this.originalCandCount = candidateActionIds.length;
            // Pad candidates to MAX_CAND
            this.candidateActionIds = padInts(candidateActionIds, MAX_CAND);
            this.candidateFeatures = padFeatures(candidateFeatures, MAX_CAND, CAND_DIM);
            this.candidateMask = padInts(candidateMask, MAX_CAND);
            this.key = new BatchKey(headId, bucketSeqLen(tokens.length));
            this.future = new CompletableFuture<>();
        }
    }

    public OnnxInferenceModel(String modelsDir) {
        this.onnxDir = Paths.get(modelsDir, "onnx");
        OrtEnvironment tmpEnv = null;
        OrtSession.SessionOptions tmpOpts = null;
        try {
            System.out.println("[ONNX] Initializing OrtEnvironment...");
            tmpEnv = OrtEnvironment.getEnvironment();
            System.out.println("[ONNX] OrtEnvironment OK");
            tmpOpts = new OrtSession.SessionOptions();
            tmpOpts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            try {
                if (EnvConfig.bool("ONNX_CUDA_GRAPH", false)) {
                    ai.onnxruntime.providers.OrtCUDAProviderOptions cudaOpts =
                            new ai.onnxruntime.providers.OrtCUDAProviderOptions(0);
                    cudaOpts.add("enable_cuda_graph", "1");
                    tmpOpts.addCUDA(cudaOpts);
                    System.out.println("[ONNX] CUDA GPU provider added with CUDA graphs");
                } else {
                    tmpOpts.addCUDA(0);
                    System.out.println("[ONNX] CUDA GPU provider added (no CUDA graph)");
                }
            } catch (Exception e) {
                System.out.println("[ONNX] CUDA not available, using CPU: " + e.getMessage());
            }
        } catch (Throwable e) {
            System.out.println("[ONNX] Failed to initialize: " + e.getMessage());
        }
        this.env = tmpEnv;
        this.sessionOpts = tmpOpts;
        if (env != null && sessionOpts != null) {
            loadSessions();
        }
        if (!sessions.isEmpty()) {
            flushScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "OnnxBatchFlush");
                t.setDaemon(true);
                return t;
            });
            // Pool for parallel head-group inference within a flush
            int inferThreads = EnvConfig.i32("ONNX_INFER_THREADS", 3);
            inferPool = Executors.newFixedThreadPool(inferThreads, r -> {
                Thread t = new Thread(r, "OnnxInfer");
                t.setDaemon(true);
                return t;
            });
            System.out.println("[ONNX] Batch manager ready: timeout=" + BATCH_TIMEOUT_MS
                    + "ms maxBatch=" + BATCH_MAX_SIZE + " inferThreads=" + inferThreads);
        }
    }

    private void loadSessions() {
        for (String headId : HEAD_IDS) {
            Path onnxPath = onnxDir.resolve("model_" + headId + ".onnx");
            if (Files.exists(onnxPath)) {
                try {
                    OrtSession session = env.createSession(onnxPath.toString(), sessionOpts);
                    sessions.put(headId, session);
                } catch (OrtException e) {
                    logger.error("Failed to load ONNX model for head " + headId + ": " + e.getMessage());
                }
            }
        }
        if (sessions.isEmpty()) {
            logger.warn("No ONNX models loaded from " + onnxDir);
        } else {
            logger.info("Loaded " + sessions.size() + " ONNX models from " + onnxDir);
        }
    }

    public void setTrainingDelegate(PythonModel delegate) {
        this.trainingDelegate = delegate;
    }

    public boolean isReady() {
        return !sessions.isEmpty();
    }

    // -----------------------------------------------------------------------
    // scoreCandidates: enqueue + wait
    // -----------------------------------------------------------------------

    @Override
    public PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey,
            String headId,
            int pickIndex,
            int minTargets,
            int maxTargets) {

        // Resolve head -- fall back to "action" if specific head not loaded
        String resolvedHead = sessions.containsKey(headId) ? headId : "action";
        if (!sessions.containsKey(resolvedHead)) {
            if (trainingDelegate != null) {
                return trainingDelegate.scoreCandidates(state, candidateActionIds,
                        candidateFeatures, candidateMask, policyKey, headId,
                        pickIndex, minTargets, maxTargets);
            }
            return uniformResult(candidateMask);
        }

        ScoreRequest req = new ScoreRequest(state, candidateActionIds,
                candidateFeatures, candidateMask, resolvedHead);

        synchronized (queueLock) {
            queue.add(req);
            if (queue.size() >= BATCH_MAX_SIZE) {
                // Immediate flush
                if (pendingFlush != null) {
                    pendingFlush.cancel(false);
                    pendingFlush = null;
                }
                flushScheduler.execute(this::flushQueue);
            } else if (pendingFlush == null) {
                // Schedule timeout flush
                pendingFlush = flushScheduler.schedule(
                        this::flushQueue, BATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            }
        }

        try {
            return req.future.get(SCORE_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            logger.error("ONNX batch wait failed: " + e.getMessage());
            return uniformResult(candidateMask);
        }
    }

    // -----------------------------------------------------------------------
    // Flush: drain queue, group by BatchKey, run batched inference
    // -----------------------------------------------------------------------

    private void flushQueue() {
        List<ScoreRequest> batch;
        synchronized (queueLock) {
            if (queue.isEmpty()) return;
            batch = new ArrayList<>(queue);
            queue.clear();
            pendingFlush = null;
        }

        // Group by BatchKey
        Map<BatchKey, List<ScoreRequest>> groups = new LinkedHashMap<>();
        for (ScoreRequest r : batch) {
            groups.computeIfAbsent(r.key, k -> new ArrayList<>()).add(r);
        }

        long batchNum = totalFlushes.incrementAndGet();
        totalBatched.addAndGet(batch.size());

        if (batchNum <= 5 || batchNum % 500 == 0) {
            StringBuilder sb = new StringBuilder("[ONNX] flush #" + batchNum + " total=" + batch.size());
            for (Map.Entry<BatchKey, List<ScoreRequest>> e : groups.entrySet()) {
                sb.append(" ").append(e.getKey().headId).append("/s").append(e.getKey().seqLen)
                        .append("=").append(e.getValue().size());
            }
            double avg = (double) totalBatched.get() / batchNum;
            sb.append(" avg=").append(String.format("%.1f", avg));
            System.out.println(sb);
        }

        if (groups.size() == 1) {
            // Single group -- run inline, no thread overhead
            Map.Entry<BatchKey, List<ScoreRequest>> only = groups.entrySet().iterator().next();
            runBatchedInference(only.getKey(), only.getValue());
        } else {
            // Multiple groups -- run in parallel on inferPool
            List<Future<?>> futures = new ArrayList<>();
            for (Map.Entry<BatchKey, List<ScoreRequest>> entry : groups.entrySet()) {
                futures.add(inferPool.submit(() ->
                        runBatchedInference(entry.getKey(), entry.getValue())));
            }
            for (Future<?> f : futures) {
                try { f.get(SCORE_TIMEOUT_MS, TimeUnit.MILLISECONDS); }
                catch (Exception e) { logger.error("Parallel infer failed: " + e.getMessage()); }
            }
        }
    }

    // Pre-allocated direct buffers per infer thread to avoid GC pressure
    private static final int MAX_SEQ = 256; // most common bucket
    private static final ThreadLocal<TensorPool> tensorPool = ThreadLocal.withInitial(TensorPool::new);

    private static class TensorPool {
        // Direct buffers for large tensors (sequences + candidate features)
        final FloatBuffer seqBuf = allocFloat(BATCH_MAX_SIZE * MAX_SEQ * DIM);
        final LongBuffer tokBuf = allocLong(BATCH_MAX_SIZE * MAX_SEQ);
        final FloatBuffer candFeatBuf = allocFloat(BATCH_MAX_SIZE * MAX_CAND * CAND_DIM);
        final LongBuffer candIdBuf = allocLong(BATCH_MAX_SIZE * MAX_CAND);

        private static FloatBuffer allocFloat(int capacity) {
            return ByteBuffer.allocateDirect(capacity * 4)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();
        }
        private static LongBuffer allocLong(int capacity) {
            return ByteBuffer.allocateDirect(capacity * 8)
                    .order(ByteOrder.nativeOrder()).asLongBuffer();
        }
    }

    private void runBatchedInference(BatchKey key, List<ScoreRequest> requests) {
        OrtSession session = sessions.get(key.headId);
        if (session == null) {
            for (ScoreRequest r : requests) {
                r.future.complete(uniformResult(r.candidateMask));
            }
            return;
        }

        int batchSize = requests.size();
        int seqLen = key.seqLen;
        int paddedBatch = EnvConfig.bool("ONNX_CUDA_GRAPH", false) ? BATCH_MAX_SIZE : batchSize;

        try {
            long tensorStart = System.nanoTime();
            TensorPool pool = tensorPool.get();

            // Fill direct buffers (flat layout, row-major)
            pool.seqBuf.clear();
            pool.tokBuf.clear();
            boolean[][] maskArr = new boolean[paddedBatch][seqLen];
            boolean[][] candMaskArr = new boolean[paddedBatch][MAX_CAND];
            pool.candFeatBuf.clear();
            pool.candIdBuf.clear();

            for (int b = 0; b < batchSize; b++) {
                ScoreRequest r = requests.get(b);
                int srcLen = Math.min(r.tokens.length, seqLen);
                for (int i = 0; i < srcLen; i++) {
                    pool.seqBuf.put(r.tokens[i], 0, Math.min(r.tokens[i].length, DIM));
                    for (int p = r.tokens[i].length; p < DIM; p++) pool.seqBuf.put(0f);
                    maskArr[b][i] = r.mask[i] != 0;
                    pool.tokBuf.put((long) r.tokenIds[i]);
                }
                for (int i = srcLen; i < seqLen; i++) {
                    for (int d = 0; d < DIM; d++) pool.seqBuf.put(0f);
                    pool.tokBuf.put(0L);
                }
                for (int c = 0; c < MAX_CAND; c++) {
                    pool.candFeatBuf.put(r.candidateFeatures[c], 0, CAND_DIM);
                    pool.candIdBuf.put((long) r.candidateActionIds[c]);
                    candMaskArr[b][c] = r.candidateMask[c] != 0;
                }
            }
            // Zero-pad remaining batch slots for fixed shape
            for (int b = batchSize; b < paddedBatch; b++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < DIM; d++) pool.seqBuf.put(0f);
                    pool.tokBuf.put(0L);
                }
                for (int c = 0; c < MAX_CAND; c++) {
                    for (int d = 0; d < CAND_DIM; d++) pool.candFeatBuf.put(0f);
                    pool.candIdBuf.put(0L);
                }
            }
            pool.seqBuf.flip();
            pool.tokBuf.flip();
            pool.candFeatBuf.flip();
            pool.candIdBuf.flip();

            long[] seqShape = {paddedBatch, seqLen, DIM};
            long[] seqShape2 = {paddedBatch, seqLen};
            long[] candShape3 = {paddedBatch, MAX_CAND, CAND_DIM};
            long[] candShape2 = {paddedBatch, MAX_CAND};

            try (OnnxTensor tSeq = OnnxTensor.createTensor(env, pool.seqBuf, seqShape);
                 OnnxTensor tMask = OnnxTensor.createTensor(env, maskArr);
                 OnnxTensor tTok = OnnxTensor.createTensor(env, pool.tokBuf, seqShape2);
                 OnnxTensor tCandFeat = OnnxTensor.createTensor(env, pool.candFeatBuf, candShape3);
                 OnnxTensor tCandId = OnnxTensor.createTensor(env, pool.candIdBuf, candShape2);
                 OnnxTensor tCandMask = OnnxTensor.createTensor(env, candMaskArr)) {

                Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
                inputs.put("sequences", tSeq);
                inputs.put("masks", tMask);
                inputs.put("token_ids", tTok);
                inputs.put("cand_features", tCandFeat);
                inputs.put("cand_ids", tCandId);
                inputs.put("cand_mask", tCandMask);

                long runStart = System.nanoTime();
                try (OrtSession.Result result = session.run(inputs)) {
                    long runEnd = System.nanoTime();
                    long tensorUs = (runStart - tensorStart) / 1000;
                    long runUs = (runEnd - runStart) / 1000;
                    long inferCount = totalFlushes.get();
                    if (inferCount <= 5 || inferCount % 500 == 0) {
                        System.out.println("[ONNX-TIME] head=" + key.headId + " batch=" + batchSize
                                + " seqLen=" + seqLen + " tensor=" + tensorUs + "us run=" + runUs + "us");
                    }
                    float[][] probs = (float[][]) result.get(0).getValue();
                    float[][] values = (float[][]) result.get(1).getValue();
                    for (int b = 0; b < batchSize; b++) {
                        ScoreRequest r = requests.get(b);
                        float[] trimmed = probs[b];
                        if (r.originalCandCount < MAX_CAND) {
                            trimmed = new float[r.originalCandCount];
                            System.arraycopy(probs[b], 0, trimmed, 0, r.originalCandCount);
                        }
                        r.future.complete(new PythonMLBatchManager.PredictionResult(
                                trimmed, values[b][0]));
                    }
                }
            }
        } catch (OrtException e) {
            logger.error("ONNX batched inference failed for head=" + key.headId
                    + " batch=" + batchSize + " seqLen=" + seqLen + ": " + e.getMessage());
            for (ScoreRequest r : requests) {
                r.future.complete(uniformResult(r.candidateMask));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private static PythonMLBatchManager.PredictionResult uniformResult(int[] candidateMask) {
        float[] uniform = new float[candidateMask.length];
        int validCount = 0;
        for (int m : candidateMask) if (m != 0) validCount++;
        float p = validCount > 0 ? 1.0f / validCount : 0.0f;
        for (int i = 0; i < candidateMask.length; i++) {
            uniform[i] = candidateMask[i] != 0 ? p : 0.0f;
        }
        return new PythonMLBatchManager.PredictionResult(uniform, 0.0f);
    }

    static int bucketSeqLen(int seqLen) {
        if (seqLen <= 0) return 0;
        int bucket = 32;
        while (bucket < seqLen) bucket <<= 1;
        return bucket;
    }

    private static int[] padInts(int[] src, int targetLen) {
        if (src.length >= targetLen) return src;
        int[] padded = new int[targetLen];
        System.arraycopy(src, 0, padded, 0, src.length);
        return padded;
    }

    private static float[][] padFeatures(float[][] src, int targetRows, int targetCols) {
        float[][] padded = new float[targetRows][targetCols];
        for (int i = 0; i < Math.min(src.length, targetRows); i++) {
            System.arraycopy(src[i], 0, padded[i], 0, Math.min(src[i].length, targetCols));
        }
        return padded;
    }

    // -----------------------------------------------------------------------
    // Training methods: delegate to Python GPU service
    // -----------------------------------------------------------------------

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        if (trainingDelegate != null) trainingDelegate.enqueueTraining(trainingData, rewards);
    }

    @Override
    public float predictMulligan(float[] features) {
        if (trainingDelegate != null) return trainingDelegate.predictMulligan(features);
        return 0.0f;
    }

    @Override
    public float[] predictMulliganScores(float[] features) {
        if (trainingDelegate != null) return trainingDelegate.predictMulliganScores(features);
        return new float[]{0.5f, -0.5f};
    }

    @Override
    public void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes,
                               byte[] gameLengths, byte[] earlyLandScores, byte[] overrides, int batchSize) {
        if (trainingDelegate != null) trainingDelegate.trainMulligan(features, decisions, outcomes,
                gameLengths, earlyLandScores, overrides, batchSize);
    }

    @Override
    public void saveMulliganModel() {
        if (trainingDelegate != null) trainingDelegate.saveMulliganModel();
    }

    @Override
    public void saveModel(String path) {
        if (trainingDelegate != null) trainingDelegate.saveModel(path);
    }

    @Override
    public String getDeviceInfo() {
        long batched = totalBatched.get();
        long flushes = totalFlushes.get();
        double avgBatch = flushes > 0 ? (double) batched / flushes : 0;
        return "onnx_batched gpu=" + sessions.containsKey("action")
                + " heads=" + sessions.size()
                + " avg_batch=" + String.format("%.1f", avgBatch);
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        if (trainingDelegate != null) return trainingDelegate.getMainModelTrainingStats();
        return Collections.emptyMap();
    }

    @Override
    public Map<String, Integer> getMulliganModelTrainingStats() {
        if (trainingDelegate != null) return trainingDelegate.getMulliganModelTrainingStats();
        return Collections.emptyMap();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        if (trainingDelegate != null) return trainingDelegate.getHealthStats();
        return Collections.emptyMap();
    }

    @Override
    public void resetHealthStats() {
        if (trainingDelegate != null) trainingDelegate.resetHealthStats();
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        if (trainingDelegate != null) trainingDelegate.recordGameResult(lastValuePrediction, won);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        if (trainingDelegate != null) return trainingDelegate.getValueHeadMetrics();
        return Collections.emptyMap();
    }

    public void shutdown() {
        if (flushScheduler != null) {
            flushScheduler.shutdownNow();
        }
        if (inferPool != null) {
            inferPool.shutdownNow();
        }
        for (OrtSession session : sessions.values()) {
            try {
                session.close();
            } catch (OrtException e) {
                logger.error("Error closing ONNX session", e);
            }
        }
        sessions.clear();
        if (trainingDelegate != null) trainingDelegate.shutdown();
    }
}
