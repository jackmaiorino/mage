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
    private static final String[] HEAD_IDS = {"action", "target", "card_select", "attack", "block", "mulligan"};

    private static final int BATCH_TIMEOUT_MS = EnvConfig.i32("ONNX_BATCH_TIMEOUT_MS", 20);
    private static final int BATCH_TIMEOUT_MAX_MS = EnvConfig.i32("ONNX_BATCH_TIMEOUT_MAX_MS", 100);
    private static final int BATCH_MAX_SIZE = EnvConfig.i32("ONNX_BATCH_MAX_SIZE", 64);
    private static final int SCORE_TIMEOUT_MS = EnvConfig.i32("ONNX_SCORE_TIMEOUT_MS", 30000);
    private static final int DIM = StateSequenceBuilder.DIM_PER_TOKEN;
    private static final int MAX_CAND = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
    private static final int CAND_DIM = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

    private final Map<String, OrtSession> sessions = new ConcurrentHashMap<>();
    // Guards session lifecycle vs use: inference takes the read lock (concurrent),
    // reload/arena-reset take the write lock so they never close() a session while
    // another thread is mid-session.run() (that race -> SIGSEGV in libonnxruntime).
    private final java.util.concurrent.locks.ReentrantReadWriteLock onnxLock =
            new java.util.concurrent.locks.ReentrantReadWriteLock();
    private final Path onnxDir;
    private volatile Path activeOnnxDir;
    private final String modelsDir;
    private final boolean forceCpu;
    private PythonModel trainingDelegate;
    private OrtEnvironment env;
    private OrtSession.SessionOptions sessionOpts;

    // --- Batching infrastructure ---
    private final List<ScoreRequest> queue = new ArrayList<>();
    private final Object queueLock = new Object();
    private ScheduledExecutorService flushScheduler;
    private ExecutorService inferPool; // parallel head-group inference
    private ScheduledFuture<?> pendingFlush;
    private final AtomicLong totalBatched = new AtomicLong();
    private final AtomicLong totalFlushes = new AtomicLong();
    private volatile long lastOnnxReloadMs = System.currentTimeMillis();
    private volatile long lastOnnxMtime = 0; // mtime of model_action.onnx at last load
    private volatile String lastOnnxIdentity = "";

    // Adaptive batch timeout: scale up when batches are small (death spiral prevention)
    private volatile int adaptiveBatchTimeoutMs = BATCH_TIMEOUT_MS;
    private volatile long lastBatchSizeEma = BATCH_MAX_SIZE; // EMA of recent batch sizes (x100 for int math)

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
        int batchIndex = -1;
        int batchSize = -1;

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
        this(modelsDir, false);
    }

    public OnnxInferenceModel(String modelsDir, boolean forceCpu) {
        this.modelsDir = modelsDir;
        this.onnxDir = Paths.get(modelsDir, "onnx");
        this.activeOnnxDir = resolveActiveOnnxDir();
        this.forceCpu = forceCpu;
        initOrtEnvironment();
        if (env != null && sessionOpts != null) {
            loadSessions();
        }
        startBatchInfra();
    }

    private boolean sharedSessionOpts = false;

    /**
     * Construct with shared OrtEnvironment and SessionOptions.
     * All instances share one CUDA arena instead of each allocating their own.
     * The shared sessionOpts will NOT be closed during periodic reload.
     */
    public OnnxInferenceModel(String modelsDir, OrtEnvironment sharedEnv, OrtSession.SessionOptions sharedOpts) {
        this.modelsDir = modelsDir;
        this.onnxDir = Paths.get(modelsDir, "onnx");
        this.activeOnnxDir = resolveActiveOnnxDir();
        this.forceCpu = false;
        this.sharedSessionOpts = true;
        this.env = sharedEnv;
        this.sessionOpts = sharedOpts;
        if (env != null && sessionOpts != null) {
            loadSessions();
        }
        startBatchInfra();
    }

    private void initOrtEnvironment() {
        try {
            System.out.println("[ONNX] Initializing OrtEnvironment...");
            env = OrtEnvironment.getEnvironment();
            System.out.println("[ONNX] OrtEnvironment OK");
            sessionOpts = createSessionOptions();
        } catch (Throwable e) {
            System.out.println("[ONNX] Failed to initialize: " + e.getMessage());
            env = null;
            sessionOpts = null;
        }
    }

    private OrtSession.SessionOptions createSessionOptions() throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        boolean cpuMode = forceCpu || EnvConfig.bool("ONNX_FORCE_CPU", false);
        // Cap the ORT intra-op thread pool. In CPU self-serve mode each game runner
        // owns its own session, so a per-session pool sized to all node cores causes
        // a thread explosion + pthread_setaffinity_np failures on a Slurm cpuset.
        // intra=1 per session is correct: parallelism comes from the many runners.
        // 0 = leave ORT default (GPU mode unaffected). Env override: ONNX_INTRA_OP_THREADS.
        int intraThreads = EnvConfig.i32("ONNX_INTRA_OP_THREADS", cpuMode ? 1 : 0);
        if (intraThreads > 0) {
            opts.setIntraOpNumThreads(intraThreads);
            opts.setInterOpNumThreads(1);
        }
        if (!forceCpu && !EnvConfig.bool("ONNX_FORCE_CPU", false)) {
            try {
                int cudaDeviceId = EnvConfig.i32("ONNX_CUDA_DEVICE_ID", inferCudaDeviceIdDefault());
                ai.onnxruntime.providers.OrtCUDAProviderOptions cudaOpts =
                        new ai.onnxruntime.providers.OrtCUDAProviderOptions(cudaDeviceId);
                // Reduce fragmentation: use kSameAsRequested so the arena doesn't
                // over-allocate and fragment VRAM with geometrically growing blocks.
                cudaOpts.add("arena_extend_strategy", "kSameAsRequested");
                // Cap ONNX GPU memory. When PyTorch trains on CPU (hybrid mode),
                // ONNX can use more VRAM. Default 8GB when training is on CPU.
                String trainDevice = System.getenv("TRAIN_CUDA_DEVICE");
                int defaultMb = ("cpu".equals(trainDevice)) ? 8192 : 5120;
                long onnxMemLimitMb = EnvConfig.i32("ONNX_GPU_MEM_LIMIT_MB", defaultMb);
                cudaOpts.add("gpu_mem_limit", String.valueOf(onnxMemLimitMb * 1024 * 1024));
                if (EnvConfig.bool("ONNX_CUDA_GRAPH", false)) {
                    cudaOpts.add("enable_cuda_graph", "1");
                    System.out.println("[ONNX] CUDA GPU provider added device=" + cudaDeviceId
                            + " with CUDA graphs + arena_extend=SameAsRequested");
                } else {
                    System.out.println("[ONNX] CUDA GPU provider added device=" + cudaDeviceId
                            + " (arena_extend=SameAsRequested)");
                }
                opts.addCUDA(cudaOpts);
            } catch (Exception e) {
                System.out.println("[ONNX] CUDA not available, using CPU: " + e.getMessage());
            }
        } else {
            System.out.println("[ONNX] Forced CPU mode" + (forceCpu ? " (forceCpu=true)" : " (ONNX_FORCE_CPU=1)"));
        }
        return opts;
    }

    private int inferCudaDeviceIdDefault() {
        String inferDevice = System.getenv("INFER_CUDA_DEVICE");
        if (inferDevice == null) {
            return 0;
        }
        String trimmed = inferDevice.trim().toLowerCase();
        if (!trimmed.startsWith("cuda:")) {
            return 0;
        }
        try {
            return Math.max(0, Integer.parseInt(trimmed.substring("cuda:".length())));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** Expose for shared-arena multi-profile construction. */
    public OrtEnvironment getEnv() { return env; }
    /** Expose for shared-arena multi-profile construction. */
    public OrtSession.SessionOptions getSessionOpts() { return sessionOpts; }

    private void startBatchInfra() {
        if (!sessions.isEmpty() && flushScheduler == null) {
            flushScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "OnnxBatchFlush");
                t.setDaemon(true);
                return t;
            });
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
        Path loadDir = activeOnnxDir != null ? activeOnnxDir : resolveActiveOnnxDir();
        for (String headId : HEAD_IDS) {
            Path onnxPath = loadDir.resolve("model_" + headId + ".onnx");
            if (Files.exists(onnxPath)) {
                try {
                    OrtSession session = env.createSession(onnxPath.toString(), sessionOpts);
                    sessions.put(headId, session);
                    if ("action".equals(headId)) {
                        try { lastOnnxMtime = Files.getLastModifiedTime(onnxPath).toMillis(); }
                        catch (Exception ignored) {}
                    }
                } catch (OrtException e) {
                    logger.error("Failed to load ONNX model for head " + headId + ": " + e.getMessage());
                }
            }
        }
        // Phase 2 belief head: archetype classifier from shared encoder CLS.
        Path beliefPath = loadDir.resolve("model_belief.onnx");
        if (Files.exists(beliefPath)) {
            try {
                OrtSession beliefSession = env.createSession(beliefPath.toString(), sessionOpts);
                sessions.put("belief", beliefSession);
                logger.info("Loaded belief ONNX model from " + beliefPath);
            } catch (OrtException e) {
                logger.error("Failed to load belief ONNX model: " + e.getMessage());
            }
        }
        if (sessions.isEmpty()) {
            logger.warn("No ONNX models loaded from " + loadDir);
        } else {
            logger.info("Loaded " + sessions.size() + " ONNX models from " + loadDir);
        }
        lastOnnxReloadMs = System.currentTimeMillis();
        lastOnnxIdentity = onnxIdentity(loadDir);
    }

    private Path resolveActiveOnnxDir() {
        Path pointer = onnxDir.resolve(".active_dir");
        if (Files.exists(pointer)) {
            try {
                String raw = new String(Files.readAllBytes(pointer), java.nio.charset.StandardCharsets.UTF_8).trim();
                if (!raw.isEmpty()) {
                    Path resolved = Paths.get(raw);
                    if (!resolved.isAbsolute()) {
                        resolved = onnxDir.resolve(resolved);
                    }
                    if (Files.isDirectory(resolved)) {
                        return resolved;
                    }
                }
            } catch (Exception ignored) {
            }
        }
        return onnxDir;
    }

    private String onnxIdentity(Path dir) {
        if (dir == null) {
            return "";
        }
        long mtime = 0L;
        try {
            Path action = dir.resolve("model_action.onnx");
            if (Files.exists(action)) {
                mtime = Files.getLastModifiedTime(action).toMillis();
            }
        } catch (Exception ignored) {
        }
        try {
            return dir.toAbsolutePath().normalize().toString() + "#" + mtime;
        } catch (Exception e) {
            return dir.toString() + "#" + mtime;
        }
    }

    /**
     * Phase 2: predict opponent's deck archetype from public state via the
     * belief head. Returns softmax probabilities [num_archetypes] in the
     * order {Wildfire, Rally, Affinity, Elves, SpyCombo, Burn, Terror,
     * CawGates, Faeries}, or null if belief model
     * unavailable. Runs single-sample inference (no batching); callers
     * should invoke sparingly (e.g., once per turn).
     */
    public float[] predictArchetype(StateSequenceBuilder.SequenceOutput state) {
        onnxLock.readLock().lock();
        try {
        OrtSession session = sessions.get("belief");
        if (session == null || state == null) return null;
        float[][] tokens = state.getSequence();
        int[] mask = state.getMask();
        int[] tokenIds = state.getTokenIds();
        if (tokens == null || tokens.length == 0) return null;
        int seqLen = tokens.length;
        int dModel = tokens[0].length;

        FloatBuffer seqBuf = ByteBuffer.allocateDirect(seqLen * dModel * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (float[] row : tokens) seqBuf.put(row);
        seqBuf.rewind();
        ByteBuffer maskBuf = ByteBuffer.allocateDirect(seqLen).order(ByteOrder.nativeOrder());
        for (int v : mask) maskBuf.put((byte) (v != 0 ? 1 : 0));
        maskBuf.rewind();
        LongBuffer tokBuf = ByteBuffer.allocateDirect(seqLen * 8)
                .order(ByteOrder.nativeOrder()).asLongBuffer();
        for (int id : tokenIds) tokBuf.put(id);
        tokBuf.rewind();

        try (OnnxTensor seqT = OnnxTensor.createTensor(env, seqBuf, new long[]{1, seqLen, dModel});
             OnnxTensor maskT = OnnxTensor.createTensor(env, maskBuf, new long[]{1, seqLen}, OnnxJavaType.BOOL);
             OnnxTensor tokT = OnnxTensor.createTensor(env, tokBuf, new long[]{1, seqLen})) {
            Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
            inputs.put("sequences", seqT);
            inputs.put("masks", maskT);
            inputs.put("token_ids", tokT);
            try (OrtSession.Result result = session.run(inputs)) {
                float[][] logits = (float[][]) result.get(0).getValue();  // [1, num_archetypes]
                float[] row = logits[0];
                float max = -Float.MAX_VALUE;
                for (float v : row) if (v > max) max = v;
                float sum = 0;
                float[] probs = new float[row.length];
                for (int i = 0; i < row.length; i++) {
                    probs[i] = (float) Math.exp(row[i] - max);
                    sum += probs[i];
                }
                if (sum > 0) for (int i = 0; i < probs.length; i++) probs[i] /= sum;
                return probs;
            }
        } catch (Exception e) {
            logger.error("Belief inference failed: " + e.getMessage());
            return null;
        }
        } finally {
            onnxLock.readLock().unlock();
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

        // Snapshot policies are NOT this ONNX export's weights. Route them to
        // the GPU service, whose snapshot_manager loads the frozen checkpoint.
        // Without this, frozen league opponents silently play the live policy.
        if (policyKey != null && policyKey.startsWith("snap:") && trainingDelegate != null) {
            return trainingDelegate.scoreCandidates(state, candidateActionIds,
                    candidateFeatures, candidateMask, policyKey, headId,
                    pickIndex, minTargets, maxTargets);
        }

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
                // Schedule timeout flush using adaptive timeout
                int timeout = adaptiveBatchTimeoutMs;
                pendingFlush = flushScheduler.schedule(
                        this::flushQueue, timeout, TimeUnit.MILLISECONDS);
            }
        }

        try {
            return req.future.get(SCORE_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            logger.error("ONNX batch wait failed: " + e.getClass().getSimpleName()
                    + " msg=" + e.getMessage()
                    + (e.getCause() != null ? " cause=" + e.getCause().getClass().getSimpleName()
                            + "/" + e.getCause().getMessage() : ""));
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

        // Adaptive batch timeout: EMA of batch size, scale timeout inversely.
        // When batches shrink (throughput dropping), wait longer to accumulate
        // more requests -- prevents the death spiral where small batches cause
        // low GPU utilization which causes fewer requests which causes smaller batches.
        long ema = lastBatchSizeEma;
        ema = (ema * 7 + batch.size() * 100L * 3) / 10; // EMA alpha=0.3, stored as x100
        lastBatchSizeEma = ema;
        int avgBatchX100 = (int) ema;
        if (avgBatchX100 < 500) { // avg < 5 requests per flush
            adaptiveBatchTimeoutMs = BATCH_TIMEOUT_MAX_MS;
        } else if (avgBatchX100 < 1500) { // avg < 15
            adaptiveBatchTimeoutMs = (BATCH_TIMEOUT_MS + BATCH_TIMEOUT_MAX_MS) / 2;
        } else {
            adaptiveBatchTimeoutMs = BATCH_TIMEOUT_MS;
        }

        // Reload ONNX sessions when files on disk are newer (re-exported by orchestrator).
        // Check every ONNX_RELOAD_CHECK_MS (default 30s). Only actually reload if
        // model_action.onnx mtime changed, avoiding expensive session teardown/create.
        long reloadCheckMs = Long.parseLong(System.getenv().getOrDefault("ONNX_RELOAD_CHECK_MS", "30000"));
        long nowMs = System.currentTimeMillis();
        if (reloadCheckMs > 0 && (nowMs - lastOnnxReloadMs) >= reloadCheckMs) {
            lastOnnxReloadMs = nowMs;
            Path resolvedDir = resolveActiveOnnxDir();
            String currentIdentity = onnxIdentity(resolvedDir);
            if (!currentIdentity.isEmpty() && !currentIdentity.equals(lastOnnxIdentity)) {
                System.out.println("[ONNX] Detected updated ONNX export, reloading at flush #" + batchNum
                        + " dir=" + resolvedDir.getFileName());
                onnxLock.writeLock().lock();
                try {
                    for (OrtSession s : sessions.values()) {
                        try { s.close(); } catch (OrtException ignored) {}
                    }
                    sessions.clear();
                    activeOnnxDir = resolvedDir;
                    loadSessions();
                } finally {
                    onnxLock.writeLock().unlock();
                }
                // Reset adaptive timeout after reload
                lastBatchSizeEma = BATCH_MAX_SIZE * 100L;
                adaptiveBatchTimeoutMs = BATCH_TIMEOUT_MS;
            }
        }
        // Periodic full CUDA arena reset (recreate session options) to prevent fragmentation.
        long arenaResetInterval = Long.parseLong(System.getenv().getOrDefault("ONNX_ARENA_RESET_INTERVAL", "100000"));
        if (arenaResetInterval > 0 && batchNum > 0 && batchNum % arenaResetInterval == 0) {
            System.out.println("[ONNX] Full CUDA arena reset at flush #" + batchNum);
            onnxLock.writeLock().lock();
            try {
                for (OrtSession s : sessions.values()) {
                    try { s.close(); } catch (OrtException ignored) {}
                }
                sessions.clear();
                if (!sharedSessionOpts && sessionOpts != null) {
                    try { sessionOpts.close(); } catch (Exception ignored) {}
                    try {
                        sessionOpts = createSessionOptions();
                    } catch (OrtException e) {
                        System.out.println("[ONNX] Failed to recreate session options: " + e.getMessage());
                    }
                }
                loadSessions();
            } finally {
                onnxLock.writeLock().unlock();
            }
            lastBatchSizeEma = BATCH_MAX_SIZE * 100L;
            adaptiveBatchTimeoutMs = BATCH_TIMEOUT_MS;
        }

        if (batchNum <= 5 || batchNum % 500 == 0) {
            StringBuilder sb = new StringBuilder("[ONNX] flush #" + batchNum + " total=" + batch.size());
            for (Map.Entry<BatchKey, List<ScoreRequest>> e : groups.entrySet()) {
                sb.append(" ").append(e.getKey().headId).append("/s").append(e.getKey().seqLen)
                        .append("=").append(e.getValue().size());
            }
            double avg = (double) totalBatched.get() / batchNum;
            sb.append(" avg=").append(String.format("%.1f", avg)).append(" timeout=").append(adaptiveBatchTimeoutMs).append("ms");
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
        onnxLock.readLock().lock();
        try {
        OrtSession session = sessions.get(key.headId);
        long runBatchId = totalFlushes.get();
        if (session == null) {
            for (ScoreRequest r : requests) {
                r.future.complete(uniformResult(r.candidateMask, "onnx_missing_session", key.headId));
            }
            return;
        }

        // Split into sub-batches if exceeding buffer capacity
        if (requests.size() > BATCH_MAX_SIZE) {
            for (int start = 0; start < requests.size(); start += BATCH_MAX_SIZE) {
                int end = Math.min(start + BATCH_MAX_SIZE, requests.size());
                runBatchedInference(key, requests.subList(start, end));
            }
            return;
        }

        int batchSize = requests.size();
        int seqLen = key.seqLen;
        int paddedBatch = EnvConfig.bool("ONNX_CUDA_GRAPH", false) ? BATCH_MAX_SIZE : batchSize;
        for (int i = 0; i < requests.size(); i++) {
            requests.get(i).batchIndex = i;
            requests.get(i).batchSize = batchSize;
        }

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
                                trimmed,
                                values[b][0],
                                "onnx_local",
                                "",
                                "onnx-" + runBatchId,
                                r.batchIndex,
                                r.batchSize,
                                Thread.currentThread().getName(),
                                "policy_scores",
                                false,
                                "headId=" + key.headId + ";seqLen=" + key.seqLen + ";modelsDir=" + modelsDir
                        ));
                    }
                }
            }
        } catch (OrtException e) {
            logger.error("ONNX batched inference failed for head=" + key.headId
                    + " batch=" + batchSize + " seqLen=" + seqLen + ": " + e.getMessage());
            for (ScoreRequest r : requests) {
                r.future.complete(uniformResult(r.candidateMask, "onnx_exception", e.getClass().getSimpleName()));
            }
        }
        } finally {
            onnxLock.readLock().unlock();
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private static PythonMLBatchManager.PredictionResult uniformResult(int[] candidateMask) {
        return uniformResult(candidateMask, "onnx_uniform", "");
    }

    private static PythonMLBatchManager.PredictionResult uniformResult(
            int[] candidateMask,
            String backendPath,
            String detail
    ) {
        float[] uniform = new float[candidateMask.length];
        int validCount = 0;
        for (int m : candidateMask) if (m != 0) validCount++;
        float p = validCount > 0 ? 1.0f / validCount : 0.0f;
        for (int i = 0; i < candidateMask.length; i++) {
            uniform[i] = candidateMask[i] != 0 ? p : 0.0f;
        }
        return new PythonMLBatchManager.PredictionResult(
                uniform,
                0.0f,
                backendPath,
                "",
                "",
                -1,
                -1,
                Thread.currentThread().getName(),
                "uniform_fallback",
                true,
                detail
        );
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
    public boolean awaitTrainingDrained(long timeoutMs) {
        return trainingDelegate == null || trainingDelegate.awaitTrainingDrained(timeoutMs);
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
