package mage.player.ai.rl;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
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
        return RLLogPaths.MODEL_FILE_PATH;
    }

    private static String defaultLatestPath(String modelPath) {
        try {
            File f = new File(modelPath);
            File dir = f.getParentFile();
            if (dir == null) {
                return RLLogPaths.MODELS_BASE_DIR + "/model_latest.pt";
            }
            return new File(dir, "model_latest.pt").getPath();
        } catch (Exception ignored) {
            return RLLogPaths.MODELS_BASE_DIR + "/model_latest.pt";
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
    private final boolean mainPerEnable;
    private final int mainPerCapacityEpisodes;
    private final double mainPerAlpha;
    private final double mainPerBetaStart;
    private final double mainPerBetaEnd;
    private final int mainPerBetaDecayEpisodes;
    private final double mainPerEps;
    private final double mainPerGamma;
    private final int mainPerMinFreshEpisodes;
    private final List<ReplayEpisode> replayEpisodes;
    private final Object replayLock = new Object();
    private final AtomicLong enqueueEpisodeCounter = new AtomicLong(0);
    private final AtomicLong replayInsertCounter = new AtomicLong(0);
    private final java.util.Random replayRng = new java.util.Random();

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
        final List<Integer> dones;
        final double priority;
        final long episodeNumber;

        TrainItem(
                List<StateSequenceBuilder.TrainingData> data,
                List<Double> rewards,
                List<Integer> dones,
                double priority,
                long episodeNumber
        ) {
            this.data = data;
            this.rewards = rewards;
            this.dones = dones;
            this.priority = priority;
            this.episodeNumber = episodeNumber;
        }
    }

    private static final class ReplayEpisode {

        final List<StateSequenceBuilder.TrainingData> data;
        final List<Double> rewards;
        final List<Integer> dones;
        final double priority;
        final long insertedAt;

        ReplayEpisode(
                List<StateSequenceBuilder.TrainingData> data,
                List<Double> rewards,
                List<Integer> dones,
                double priority,
                long insertedAt
        ) {
            this.data = data;
            this.rewards = rewards;
            this.dones = dones;
            this.priority = priority;
            this.insertedAt = insertedAt;
        }
    }

    private static final class ReplaySample {

        final ReplayEpisode episode;
        final double probability;

        ReplaySample(ReplayEpisode episode, double probability) {
            this.episode = episode;
            this.probability = probability;
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
        this.mainPerEnable = EnvConfig.bool("MAIN_PER_ENABLE", true);
        this.mainPerCapacityEpisodes = Math.max(8, EnvConfig.i32("MAIN_PER_CAPACITY_EPISODES", 64));
        this.mainPerAlpha = Math.max(0.0, EnvConfig.f64("MAIN_PER_ALPHA", 0.7));
        this.mainPerBetaStart = Math.max(0.0, Math.min(1.0, EnvConfig.f64("MAIN_PER_BETA_START", 0.4)));
        this.mainPerBetaEnd = Math.max(0.0, Math.min(1.0, EnvConfig.f64("MAIN_PER_BETA_END", 1.0)));
        this.mainPerBetaDecayEpisodes = Math.max(1, EnvConfig.i32("MAIN_PER_BETA_DECAY_EPISODES", 400000));
        this.mainPerEps = Math.max(1e-9, EnvConfig.f64("MAIN_PER_EPS", 1e-3));
        this.mainPerGamma = Math.max(0.0, Math.min(1.0, EnvConfig.f64("MAIN_PER_GAMMA", 0.99)));
        this.mainPerMinFreshEpisodes = Math.max(1, EnvConfig.i32("MAIN_PER_MIN_FRESH_EPISODES", 1));
        this.replayEpisodes = new ArrayList<>(this.mainPerCapacityEpisodes);
        long baseSeed = EnvConfig.i64("RL_BASE_SEED", -1L);
        if (baseSeed >= 0L) {
            this.replayRng.setSeed(baseSeed ^ 0x5DEECE66DL);
        }
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

    private static List<Integer> buildEpisodeDones(int stepCount) {
        if (stepCount <= 0) {
            return Collections.emptyList();
        }
        List<Integer> dones = new ArrayList<>(stepCount);
        for (int i = 0; i < stepCount; i++) {
            dones.add((i == stepCount - 1) ? 1 : 0);
        }
        return dones;
    }

    private double computeEpisodePriority(
            List<StateSequenceBuilder.TrainingData> data,
            List<Double> rewards,
            List<Integer> dones
    ) {
        if (data == null || rewards == null || dones == null) {
            return mainPerEps;
        }
        int n = data.size();
        if (n == 0 || rewards.size() != n || dones.size() != n) {
            return mainPerEps;
        }
        double sumAbsTd = 0.0;
        for (int i = 0; i < n; i++) {
            double reward = rewards.get(i) == null ? 0.0 : rewards.get(i);
            double done = dones.get(i) == null ? 0.0 : dones.get(i);
            double vNow = data.get(i).oldValue;
            double vNext = 0.0;
            if (i + 1 < n && done == 0.0) {
                vNext = data.get(i + 1).oldValue;
            }
            double td = reward + (mainPerGamma * (1.0 - done) * vNext) - vNow;
            sumAbsTd += Math.abs(td);
        }
        double priority = (sumAbsTd / n) + mainPerEps;
        if (!Double.isFinite(priority) || priority <= 0.0) {
            return mainPerEps;
        }
        return priority;
    }

    private double currentPerBeta() {
        long episodes = Math.max(0L, enqueueEpisodeCounter.get());
        if (episodes >= mainPerBetaDecayEpisodes) {
            return mainPerBetaEnd;
        }
        double t = episodes / (double) mainPerBetaDecayEpisodes;
        return mainPerBetaStart + (mainPerBetaEnd - mainPerBetaStart) * t;
    }

    private void addReplayEpisode(ReplayEpisode episode) {
        if (!mainPerEnable || episode == null || episode.data == null || episode.data.isEmpty()) {
            return;
        }
        synchronized (replayLock) {
            if (replayEpisodes.size() >= mainPerCapacityEpisodes) {
                int evictIdx = 0;
                ReplayEpisode evict = replayEpisodes.get(0);
                for (int i = 1; i < replayEpisodes.size(); i++) {
                    ReplayEpisode candidate = replayEpisodes.get(i);
                    if (candidate.priority < evict.priority
                            || (candidate.priority == evict.priority && candidate.insertedAt < evict.insertedAt)) {
                        evict = candidate;
                        evictIdx = i;
                    }
                }
                replayEpisodes.remove(evictIdx);
            }
            replayEpisodes.add(episode);
        }
    }

    private List<ReplaySample> sampleReplayEpisodes(int maxEpisodes, int maxSteps) {
        if (!mainPerEnable || maxEpisodes <= 0 || maxSteps <= 0) {
            return Collections.emptyList();
        }

        List<ReplayEpisode> snapshot;
        synchronized (replayLock) {
            if (replayEpisodes.isEmpty()) {
                return Collections.emptyList();
            }
            snapshot = new ArrayList<>(replayEpisodes);
        }

        int n = snapshot.size();
        double[] powered = new double[n];
        double totalPowered = 0.0;
        for (int i = 0; i < n; i++) {
            double base = Math.max(mainPerEps, snapshot.get(i).priority);
            double p = Math.pow(base, mainPerAlpha);
            if (!Double.isFinite(p) || p <= 0.0) {
                p = mainPerEps;
            }
            powered[i] = p;
            totalPowered += p;
        }
        if (!(totalPowered > 0.0)) {
            for (int i = 0; i < n; i++) {
                powered[i] = 1.0;
            }
            totalPowered = n;
        }

        List<Integer> remaining = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            remaining.add(i);
        }

        List<ReplaySample> sampled = new ArrayList<>();
        int stepsUsed = 0;
        while (sampled.size() < maxEpisodes && !remaining.isEmpty()) {
            double totalRemaining = 0.0;
            for (int idx : remaining) {
                totalRemaining += powered[idx];
            }
            if (!(totalRemaining > 0.0)) {
                break;
            }
            double r = replayRng.nextDouble() * totalRemaining;
            double cdf = 0.0;
            int chosenPos = -1;
            for (int pos = 0; pos < remaining.size(); pos++) {
                cdf += powered[remaining.get(pos)];
                if (r <= cdf) {
                    chosenPos = pos;
                    break;
                }
            }
            if (chosenPos < 0) {
                chosenPos = remaining.size() - 1;
            }
            int idx = remaining.remove(chosenPos);
            ReplayEpisode ep = snapshot.get(idx);
            int epSteps = ep.data == null ? 0 : ep.data.size();
            if (epSteps <= 0) {
                continue;
            }
            if (stepsUsed > 0 && (stepsUsed + epSteps) > maxSteps) {
                break;
            }
            stepsUsed += epSteps;
            double prob = powered[idx] / totalPowered;
            sampled.add(new ReplaySample(ep, prob));
        }
        return sampled;
    }

    private boolean appendEpisodeToBatch(
            List<StateSequenceBuilder.TrainingData> data,
            List<Double> rewards,
            List<Integer> dones,
            double sampleWeight,
            List<StateSequenceBuilder.TrainingData> mergedData,
            List<Double> mergedRewards,
            List<Integer> mergedDones,
            List<Double> mergedSampleWeights,
            int maxSteps
    ) {
        if (data == null || rewards == null || dones == null || data.isEmpty()) {
            return false;
        }
        int n = data.size();
        if (rewards.size() != n || dones.size() != n) {
            return false;
        }
        if (!mergedData.isEmpty() && (mergedData.size() + n) > maxSteps) {
            return false;
        }
        double w = Double.isFinite(sampleWeight) && sampleWeight > 0.0 ? sampleWeight : 1.0;
        mergedData.addAll(data);
        mergedRewards.addAll(rewards);
        mergedDones.addAll(dones);
        for (int i = 0; i < n; i++) {
            mergedSampleWeights.add(w);
        }
        return true;
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
                        List<Integer> mergedDones = new ArrayList<>();
                        List<Double> sampleWeights = new ArrayList<>();

                        int episodes = 0;
                        int totalSteps = 0;
                        int freshEpisodes = 0;
                        int replayEpisodesUsed = 0;
                        double replayPrioritySum = 0.0;
                        double replayWeightSum = 0.0;
                        int replayWeightCount = 0;

                        if (appendEpisodeToBatch(
                                currentFirst.data,
                                currentFirst.rewards,
                                currentFirst.dones,
                                1.0,
                                mergedData,
                                mergedRewards,
                                mergedDones,
                                sampleWeights,
                                learnerBatchMaxSteps
                        )) {
                            episodes++;
                            freshEpisodes++;
                            totalSteps = mergedData.size();
                        }

                        int freshTarget = mainPerEnable
                                ? Math.min(learnerBatchMaxEpisodes, Math.max(1, mainPerMinFreshEpisodes))
                                : learnerBatchMaxEpisodes;
                        while (episodes < freshTarget && totalSteps < learnerBatchMaxSteps) {
                            TrainItem next = trainQueue.poll(0, TimeUnit.MILLISECONDS);
                            if (next == null) {
                                break;
                            }
                            if (appendEpisodeToBatch(
                                    next.data,
                                    next.rewards,
                                    next.dones,
                                    1.0,
                                    mergedData,
                                    mergedRewards,
                                    mergedDones,
                                    sampleWeights,
                                    learnerBatchMaxSteps
                            )) {
                                episodes++;
                                freshEpisodes++;
                                totalSteps = mergedData.size();
                            }
                        }

                        if (!mainPerEnable) {
                            while (episodes < learnerBatchMaxEpisodes && totalSteps < learnerBatchMaxSteps) {
                                TrainItem next = trainQueue.poll(0, TimeUnit.MILLISECONDS);
                                if (next == null) {
                                    break;
                                }
                                if (appendEpisodeToBatch(
                                        next.data,
                                        next.rewards,
                                        next.dones,
                                        1.0,
                                        mergedData,
                                        mergedRewards,
                                        mergedDones,
                                        sampleWeights,
                                        learnerBatchMaxSteps
                                )) {
                                    episodes++;
                                    totalSteps = mergedData.size();
                                }
                            }
                        } else if (episodes < learnerBatchMaxEpisodes && totalSteps < learnerBatchMaxSteps) {
                            int replaySlots = learnerBatchMaxEpisodes - episodes;
                            int replayStepBudget = learnerBatchMaxSteps - totalSteps;
                            List<ReplaySample> replaySamples = sampleReplayEpisodes(replaySlots, replayStepBudget);
                            int replaySizeSnapshot;
                            synchronized (replayLock) {
                                replaySizeSnapshot = replayEpisodes.size();
                            }
                            double beta = currentPerBeta();
                            double maxRawWeight = 1.0;
                            List<Double> replayRawWeights = new ArrayList<>(replaySamples.size());
                            for (ReplaySample sample : replaySamples) {
                                double p = Math.max(mainPerEps, sample.probability);
                                double base = Math.max(mainPerEps, replaySizeSnapshot * p);
                                double raw = Math.pow(base, -beta);
                                if (!Double.isFinite(raw) || raw <= 0.0) {
                                    raw = 1.0;
                                }
                                replayRawWeights.add(raw);
                                if (raw > maxRawWeight) {
                                    maxRawWeight = raw;
                                }
                            }
                            for (int i = 0; i < replaySamples.size(); i++) {
                                ReplaySample sample = replaySamples.get(i);
                                double normalizedWeight = replayRawWeights.get(i) / maxRawWeight;
                                if (appendEpisodeToBatch(
                                        sample.episode.data,
                                        sample.episode.rewards,
                                        sample.episode.dones,
                                        normalizedWeight,
                                        mergedData,
                                        mergedRewards,
                                        mergedDones,
                                        sampleWeights,
                                        learnerBatchMaxSteps
                                )) {
                                    episodes++;
                                    replayEpisodesUsed++;
                                    totalSteps = mergedData.size();
                                    replayPrioritySum += sample.episode.priority;
                                    replayWeightSum += normalizedWeight;
                                    replayWeightCount++;
                                }
                                if (episodes >= learnerBatchMaxEpisodes || totalSteps >= learnerBatchMaxSteps) {
                                    break;
                                }
                            }
                        }

                        if (!mergedData.isEmpty()) {
                            boolean cappedByEpisodes = episodes >= learnerBatchMaxEpisodes;
                            boolean cappedBySteps = totalSteps >= learnerBatchMaxSteps;
                            try {
                                MetricsCollector.getInstance().recordTrainBatchFlush(episodes, totalSteps, cappedByEpisodes, cappedBySteps);
                                if (mainPerEnable) {
                                    int replayBufferSizeSnapshot;
                                    synchronized (replayLock) {
                                        replayBufferSizeSnapshot = replayEpisodes.size();
                                    }
                                    double meanPriority = replayEpisodesUsed > 0
                                            ? replayPrioritySum / replayEpisodesUsed
                                            : 0.0;
                                    double meanIsWeight = replayWeightCount > 0
                                            ? replayWeightSum / replayWeightCount
                                            : 0.0;
                                    MetricsCollector.getInstance().recordMainPerBatch(
                                            replayBufferSizeSnapshot,
                                            freshEpisodes,
                                            replayEpisodesUsed,
                                            meanPriority,
                                            meanIsWeight
                                    );
                                }
                            } catch (Exception ignored) {
                            }

                            long t0 = System.nanoTime();
                            learner.trainMulti(mergedData, mergedRewards, mergedDones, sampleWeights);
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
        if (rewards == null || rewards.size() != trainingData.size()) {
            logger.warning("enqueueTraining: rewards size mismatch, skipping episode (data=" + trainingData.size() + ")");
            return;
        }
        try {
            List<StateSequenceBuilder.TrainingData> dataCopy = new ArrayList<>(trainingData);
            List<Double> rewardsCopy = new ArrayList<>(rewards);
            List<Integer> dones = buildEpisodeDones(dataCopy.size());
            double priority = computeEpisodePriority(dataCopy, rewardsCopy, dones);
            long episodeNumber = enqueueEpisodeCounter.incrementAndGet();
            TrainItem item = new TrainItem(dataCopy, rewardsCopy, dones, priority, episodeNumber);
            addReplayEpisode(new ReplayEpisode(
                    dataCopy,
                    rewardsCopy,
                    dones,
                    priority,
                    replayInsertCounter.incrementAndGet()
            ));
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
