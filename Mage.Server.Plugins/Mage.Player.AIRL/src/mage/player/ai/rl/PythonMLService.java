package mage.player.ai.rl;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

public class PythonMLService implements PythonModel {

    private static final Logger logger = Logger.getLogger(PythonMLService.class.getName());

    private static volatile PythonMLService instance;
    private static final Object lock = new Object();

    public static PythonMLService getInstance() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new PythonMLService();
                }
            }
        }
        return instance;
    }

    private static String defaultModelPath() {
        return EnvConfig.str("MTG_MODEL_PATH",
                EnvConfig.str("MODEL_PATH",
                        "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt"));
    }

    private static String defaultLatestPath(String modelPath) {
        try {
            File f = new File(modelPath);
            File dir = f.getParentFile();
            if (dir == null) {
                return "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model_latest.pt";
            }
            return new File(dir, "model_latest.pt").getPath();
        } catch (Exception ignored) {
            return "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model_latest.pt";
        }
    }

    private final PythonMLBridge learner;
    private final List<InferSlot> inference;
    private final AtomicInteger rr;
    private final boolean singleBackend;

    private final ArrayBlockingQueue<TrainItem> trainQueue;
    private final Thread learnerThread;
    private volatile boolean running;
    private final AtomicBoolean shutdownOnce = new AtomicBoolean(false);
    private final AtomicLong droppedTrainEpisodes = new AtomicLong(0);
    private final int trainQueueOfferTimeoutMs;
    private final boolean trainQueueDropOnFull;

    private final String latestWeightsPath;
    private final int syncEveryTrainSteps;
    private final long syncEveryMs;
    private final long reloadEveryMs;
    private final int learnerBatchMaxEpisodes;
    private final int learnerBatchMaxSteps;

    // Auto-batching metrics polling (per Python worker)
    private final ConcurrentHashMap<String, long[]> autoBatchLastCounts = new ConcurrentHashMap<>();
    private volatile long lastLearnerAutoBatchPollMs = 0L;
    private volatile long lastLearnerLossMetricsPollMs = 0L;

    private static final class InferSlot {

        final PythonMLBridge bridge;
        volatile long lastReloadMs;
        volatile long lastSeenLatestMtime;
        volatile long lastAutoBatchPollMs;
        final AtomicInteger autoBatchPollCounter;

        InferSlot(PythonMLBridge bridge) {
            this.bridge = bridge;
            this.lastReloadMs = 0L;
            this.lastSeenLatestMtime = 0L;
            this.lastAutoBatchPollMs = 0L;
            this.autoBatchPollCounter = new AtomicInteger(0);
        }
    }

    private static final class TrainItem {

        final List<StateSequenceBuilder.TrainingData> data;
        final List<Double> rewards;

        TrainItem(List<StateSequenceBuilder.TrainingData> data, List<Double> rewards) {
            this.data = data;
            this.rewards = rewards;
        }
    }

    private PythonMLService() {
        String backendMode = EnvConfig.str("PY_BACKEND_MODE", "multi").trim().toLowerCase();
        this.singleBackend = "single".equals(backendMode);
        int inferWorkers = Math.max(0, EnvConfig.i32("INFER_WORKERS", 4));
        int basePort = EnvConfig.i32("PY4J_BASE_PORT", 25334);
        int trainQueueMax = Math.max(8, EnvConfig.i32("TRAIN_QUEUE_MAX_EPISODES", 256));
        this.trainQueueOfferTimeoutMs = Math.max(0, EnvConfig.i32("TRAIN_QUEUE_OFFER_TIMEOUT_MS", 0));
        this.trainQueueDropOnFull = EnvConfig.bool("TRAIN_QUEUE_DROP_ON_FULL", true);

        String modelPath = defaultModelPath();
        this.latestWeightsPath = EnvConfig.str("MODEL_LATEST_PATH", defaultLatestPath(modelPath));

        this.syncEveryTrainSteps = Math.max(1, EnvConfig.i32("MODEL_SYNC_EVERY_TRAIN_STEPS", 25));
        this.syncEveryMs = Math.max(200, EnvConfig.i32("MODEL_SYNC_EVERY_MS", 2000));
        // Learner micro-batching: drain multiple completed episodes per train() call to improve GPU utilization.
        this.learnerBatchMaxEpisodes = Math.max(1, EnvConfig.i32("LEARNER_BATCH_MAX_EPISODES", 8));
        this.learnerBatchMaxSteps = Math.max(128, EnvConfig.i32("LEARNER_BATCH_MAX_STEPS", 4096));
        // Allow disabling reloads entirely with MODEL_RELOAD_EVERY_MS=0.
        // In single-backend mode we don't spin inference workers, so periodic reload is unnecessary.
        long reloadMs = EnvConfig.i32("MODEL_RELOAD_EVERY_MS", 2000);
        if (singleBackend) {
            this.reloadEveryMs = 0L;
            inferWorkers = 0;
        } else {
            this.reloadEveryMs = reloadMs <= 0 ? 0 : Math.max(200, reloadMs);
        }

        this.learner = PythonMLBridge.createAdditionalBridge(basePort, "learner", false);

        this.inference = new ArrayList<>();
        if (inferWorkers > 0) {
            int threads = Math.min(inferWorkers, Math.max(1, EnvConfig.i32("INFER_STARTUP_THREADS", 4)));
            ExecutorService exec = Executors.newFixedThreadPool(threads, r -> {
                Thread t = new Thread(r, "PyInferStartup");
                t.setDaemon(true);
                return t;
            });
            List<Future<InferSlot>> futures = new ArrayList<>(inferWorkers);
            for (int i = 0; i < inferWorkers; i++) {
                final int port = basePort + 1 + i;
                futures.add(exec.submit((Callable<InferSlot>) () -> new InferSlot(
                        PythonMLBridge.createAdditionalBridge(port, "inference", true)
                )));
            }
            exec.shutdown();
            for (Future<InferSlot> f : futures) {
                try {
                    this.inference.add(f.get());
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException ee) {
                    logger.warning("Inference worker startup failed: " + ee.getMessage());
                }
            }
        }
        this.rr = new AtomicInteger(0);

        // Bounded training queue with backpressure
        this.trainQueue = new ArrayBlockingQueue<>(trainQueueMax);
        this.running = true;
        this.learnerThread = new Thread(this::learnerLoop, "PyLearner");
        this.learnerThread.setDaemon(true);
        this.learnerThread.start();

        // Always tear down Python subprocesses on JVM exit (Ctrl+C, Maven abort, etc).
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                shutdown();
            } catch (Exception ignored) {
            }
        }, "PyService-ShutdownHook"));
    }

    private void learnerLoop() {
        long lastSyncMsLocal = System.currentTimeMillis();
        int stepsSinceSync = 0;

        while (running) {
            try {
                TrainItem first = trainQueue.poll(250, TimeUnit.MILLISECONDS);
                if (first == null) {
                    continue;
                }

                // Multi-backend: acquire GPU lock for entire training burst (hold until queue is empty).
                // Single-backend: Python-side mutex handles "training pauses inference".
                if (!singleBackend) {
                    learner.acquireGPULock();
                }
                try {
                    // Process all available episodes in queue while holding GPU lock
                    boolean hasMoreWork = true;
                    TrainItem currentFirst = first;

                    while (hasMoreWork && running) {
                        // Drain a small bundle of episodes to amortize Python overhead and better use the GPU.
                        List<StateSequenceBuilder.TrainingData> mergedData = new ArrayList<>();
                        List<Double> mergedRewards = new ArrayList<>();
                        List<Integer> dones = new ArrayList<>();

                        int episodes = 0;
                        int totalSteps = 0;

                        java.util.function.Consumer<TrainItem> appendEpisode = (it) -> {
                            if (it == null || it.data == null || it.data.isEmpty()) {
                                return;
                            }
                            int n = it.data.size();
                            if (it.rewards == null || it.rewards.size() != n) {
                                logger.warning("Learner batch: rewards size mismatch, skipping episode (data=" + n + ")");
                                return;
                            }
                            mergedData.addAll(it.data);
                            mergedRewards.addAll(it.rewards);
                            for (int i = 0; i < n; i++) {
                                dones.add((i == n - 1) ? 1 : 0);
                            }
                        };

                        appendEpisode.accept(currentFirst);
                        episodes = 1;
                        totalSteps = mergedData.size();

                        while (episodes < learnerBatchMaxEpisodes && totalSteps < learnerBatchMaxSteps) {
                            TrainItem next = trainQueue.poll(0, TimeUnit.MILLISECONDS);
                            if (next == null) {
                                break;
                            }
                            int before = mergedData.size();
                            appendEpisode.accept(next);
                            int after = mergedData.size();
                            if (after > before) {
                                episodes++;
                                totalSteps = after;
                            }
                        }

                        if (!mergedData.isEmpty()) {
                            boolean cappedByEpisodes = episodes >= learnerBatchMaxEpisodes;
                            boolean cappedBySteps = totalSteps >= learnerBatchMaxSteps;
                            try {
                                MetricsCollector.getInstance().recordTrainBatchFlush(episodes, totalSteps, cappedByEpisodes, cappedBySteps);
                            } catch (Exception ignored) {
                            }

                            long t0 = System.nanoTime();
                            learner.trainMulti(mergedData, mergedRewards, dones);
                            long trainMs = (System.nanoTime() - t0) / 1_000_000L;
                            try {
                                MetricsCollector.getInstance().recordTrainLatencyMs(trainMs);
                            } catch (Exception ignored) {
                            }
                            stepsSinceSync++;
                        }

                        // Check if there's more work in queue (non-blocking)
                        currentFirst = trainQueue.poll(0, TimeUnit.MILLISECONDS);
                        hasMoreWork = (currentFirst != null);
                    }
                } finally {
                    if (!singleBackend) {
                        // Release GPU lock after training burst completes
                        learner.releaseGPULock();
                    }
                }

                // Poll auto-batching telemetry from learner periodically (best-effort).
                long nowPoll = System.currentTimeMillis();
                if (nowPoll - lastLearnerAutoBatchPollMs > 5000L) {
                    lastLearnerAutoBatchPollMs = nowPoll;
                    tryPollAutoBatchMetrics(learner);
                }

                // Poll training loss metrics from learner periodically (best-effort).
                long nowLoss = System.currentTimeMillis();
                if (nowLoss - lastLearnerLossMetricsPollMs > 5000L) {
                    lastLearnerLossMetricsPollMs = nowLoss;
                    tryPollTrainingLossMetrics(learner);
                }

                long now = System.currentTimeMillis();
                if (stepsSinceSync >= syncEveryTrainSteps || now - lastSyncMsLocal >= syncEveryMs) {
                    if (learner.saveLatestModelAtomic(latestWeightsPath)) {
                        lastSyncMsLocal = now;
                        stepsSinceSync = 0;
                    }
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                // Don't kill the learner loop for transient Python errors.
                try {
                    logger.warning("Learner loop error: " + t.getMessage());
                } catch (Exception ignored) {
                }
            }
        }
    }

    private InferSlot pickInferenceBridge() {
        if (inference.isEmpty()) {
            return null;
        }
        int idx = Math.floorMod(rr.getAndIncrement(), inference.size());
        return inference.get(idx);
    }

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
        InferSlot slot = pickInferenceBridge();
        PythonMLBridge b = (slot == null) ? learner : slot.bridge;

        // Best-effort periodic reload to keep inference workers reasonably fresh.
        long now = System.currentTimeMillis();
        if (!singleBackend && slot != null && reloadEveryMs > 0) {
            if (now - slot.lastReloadMs > reloadEveryMs) {
                slot.lastReloadMs = now;
                try {
                    File latestFile = new File(latestWeightsPath);
                    long mtime = latestFile.exists() ? latestFile.lastModified() : 0L;
                    if (mtime > slot.lastSeenLatestMtime) {
                        if (b.reloadLatestModelIfNewer(latestWeightsPath)) {
                            slot.lastSeenLatestMtime = mtime;
                        }
                    }
                } catch (Exception ignored) {
                }
            }
        }
        PythonMLBatchManager.PredictionResult r = b.scoreCandidates(
                state, candidateActionIds, candidateFeatures, candidateMask,
                policyKey, headId, pickIndex, minTargets, maxTargets
        );

        // Poll auto-batching telemetry from inference workers occasionally (best-effort).
        if (!singleBackend && slot != null) {
            int c = slot.autoBatchPollCounter.incrementAndGet();
            long now2 = System.currentTimeMillis();
            if (c >= 500 || (now2 - slot.lastAutoBatchPollMs) > 5000L) {
                slot.autoBatchPollCounter.set(0);
                slot.lastAutoBatchPollMs = now2;
                tryPollAutoBatchMetrics(slot.bridge);
            }
        }
        return r;
    }

    private void tryPollAutoBatchMetrics(PythonMLBridge bridge) {
        try {
            Map<String, Object> m = bridge.getAutoBatchMetrics();
            if (m == null || m.isEmpty()) {
                return;
            }
            String worker = String.valueOf(m.getOrDefault("worker", ""));
            if (worker == null || worker.trim().isEmpty()) {
                worker = "unknown";
            }

            long inferCap = toLong(m.get("infer_splits_cap"));
            long inferPaging = toLong(m.get("infer_splits_paging"));
            long inferOom = toLong(m.get("infer_splits_oom"));
            long trainCap = toLong(m.get("train_splits_cap"));
            long trainPaging = toLong(m.get("train_splits_paging"));
            long trainOom = toLong(m.get("train_splits_oom"));

            long[] last = autoBatchLastCounts.computeIfAbsent(worker, k -> new long[6]);
            long dInferCap = delta(inferCap, last[0]);
            last[0] = inferCap;
            long dInferPaging = delta(inferPaging, last[1]);
            last[1] = inferPaging;
            long dInferOom = delta(inferOom, last[2]);
            last[2] = inferOom;
            long dTrainCap = delta(trainCap, last[3]);
            last[3] = trainCap;
            long dTrainPaging = delta(trainPaging, last[4]);
            last[4] = trainPaging;
            long dTrainOom = delta(trainOom, last[5]);
            last[5] = trainOom;

            MetricsCollector.getInstance().recordAutoBatchDeltas(
                    dInferCap, dInferPaging, dInferOom,
                    dTrainCap, dTrainPaging, dTrainOom
            );

            long inferSafeMax = toLong(m.get("infer_safe_max"));
            long trainSafeMaxEpisodes = toLong(m.get("train_safe_max_episodes"));
            double inferMbPerSample = toDouble(m.get("infer_mb_per_sample"));
            double trainMbPerStep = toDouble(m.get("train_mb_per_step"));
            double freeMb = toDouble(m.get("free_mb"));
            double desiredFreeMb = toDouble(m.get("desired_free_mb"));
            double trainTimeMs = toDouble(m.get("train_time_ms"));
            double inferTimeMs = toDouble(m.get("infer_time_ms"));
            double mulliganTimeMs = toDouble(m.get("mulligan_time_ms"));
            MetricsCollector.getInstance().recordAutoBatchSnapshot(
                    inferSafeMax,
                    trainSafeMaxEpisodes,
                    inferMbPerSample,
                    trainMbPerStep,
                    freeMb,
                    desiredFreeMb,
                    trainTimeMs,
                    inferTimeMs,
                    mulliganTimeMs
            );
        } catch (Exception ignored) {
        }
    }

    private void tryPollTrainingLossMetrics(PythonMLBridge bridge) {
        try {
            Map<String, Object> m = bridge.getTrainingLossMetrics();
            if (m == null || m.isEmpty()) {
                return;
            }
            MetricsCollector.getInstance().recordTrainingLossComponents(m);
        } catch (Exception ignored) {
        }
    }

    private static long delta(long now, long prev) {
        if (now < prev) {
            return now; // reset
        }
        return now - prev;
    }

    private static long toLong(Object o) {
        if (o == null) {
            return 0L;
        }
        if (o instanceof Number) {
            return ((Number) o).longValue();
        }
        try {
            return Long.parseLong(String.valueOf(o));
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private static double toDouble(Object o) {
        if (o == null) {
            return 0.0;
        }
        if (o instanceof Number) {
            return ((Number) o).doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(o));
        } catch (Exception ignored) {
            return 0.0;
        }
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        if (trainingData == null || trainingData.isEmpty()) {
            return;
        }
        try {
            TrainItem item = new TrainItem(trainingData, rewards);
            boolean ok;
            if (trainQueueOfferTimeoutMs <= 0) {
                ok = trainQueue.offer(item);
            } else {
                ok = trainQueue.offer(item, trainQueueOfferTimeoutMs, TimeUnit.MILLISECONDS);
            }
            if (!ok) {
                if (trainQueueDropOnFull) {
                    long n = droppedTrainEpisodes.incrementAndGet();
                    if (n == 1 || (n % 100) == 0) {
                        logger.warning("Training queue full: dropped episodes=" + n + " (queueMax=" + trainQueue.remainingCapacity() + ")");
                    }
                    return;
                }
                // Legacy behavior: block
                trainQueue.put(item);
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public float predictMulligan(float[] features) {
        return learner.predictMulligan(features);
    }

    @Override
    public float[] predictMulliganScores(float[] features) {
        return learner.predictMulliganScores(features);
    }

    @Override
    public void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes, byte[] gameLengths, byte[] earlyLandScores, byte[] overrides, int batchSize) {
        learner.trainMulligan(features, decisions, outcomes, gameLengths, earlyLandScores, overrides, batchSize);
    }

    @Override
    public void saveMulliganModel() {
        learner.saveMulliganModel();
    }

    @Override
    public void saveModel(String path) {
        learner.saveModel(path);
    }

    @Override
    public String getDeviceInfo() {
        return learner.getDeviceInfo();
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        return learner.getMainModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getMulliganModelTrainingStats() {
        return learner.getMulliganModelTrainingStats();
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        return learner.getHealthStats();
    }

    @Override
    public void resetHealthStats() {
        learner.resetHealthStats();
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        learner.recordGameResult(lastValuePrediction, won);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        return learner.getValueHeadMetrics();
    }

    public int getTrainQueueDepth() {
        return trainQueue.size();
    }

    public long getDroppedTrainEpisodes() {
        return droppedTrainEpisodes.get();
    }

    @Override
    public void shutdown() {
        if (!shutdownOnce.compareAndSet(false, true)) {
            return;
        }
        running = false;
        try {
            learnerThread.interrupt();
        } catch (Exception ignored) {
        }
        try {
            learner.shutdown();
        } catch (Exception ignored) {
        }
        for (InferSlot s : inference) {
            try {
                s.bridge.shutdown();
            } catch (Exception ignored) {
            }
        }
    }
}
