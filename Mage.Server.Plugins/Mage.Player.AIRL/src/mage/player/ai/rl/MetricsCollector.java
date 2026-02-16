package mage.player.ai.rl;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

import org.apache.log4j.Logger;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

/**
 * Prometheus-compatible metrics exporter for Mage AI training system. Exposes
 * metrics on /metrics endpoint for monitoring training progress.
 */
public class MetricsCollector {

    private static final Logger logger = Logger.getLogger(MetricsCollector.class);
    private static final MetricsCollector INSTANCE = new MetricsCollector();

    // Metrics storage
    private final AtomicLong episodesStarted = new AtomicLong(0);
    private final AtomicLong episodesCompleted = new AtomicLong(0);
    private final AtomicLong samplesProcessed = new AtomicLong(0);
    private final AtomicInteger activeEpisodes = new AtomicInteger(0);
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private final AtomicInteger optimalBatchSize = new AtomicInteger(0);
    private final DoubleAdder trainingLoss = new DoubleAdder();
    private final DoubleAdder winRate = new DoubleAdder();
    private final AtomicLong errorsTotal = new AtomicLong(0);
    private final AtomicLong trainingUpdates = new AtomicLong(0);
    private final AtomicLong gpuMemoryUsed = new AtomicLong(0);
    private final AtomicLong gpuMemoryTotal = new AtomicLong(0);

    // Inference health metrics (candidate scoring)
    private final AtomicLong inferRequestsTotal = new AtomicLong(0);
    private final AtomicLong inferTimeoutsTotal = new AtomicLong(0);
    private final AtomicLong inferLatencyCount = new AtomicLong(0);
    private final AtomicLong inferLatencySumMs = new AtomicLong(0);
    private final AtomicLong inferLatencyMaxMs = new AtomicLong(0);

    // Inference batching (Java-side) metrics
    private final AtomicLong inferFlushesTotal = new AtomicLong(0);
    private final AtomicLong inferFlushesFullTotal = new AtomicLong(0);
    private final AtomicLong inferFlushesTimeoutTotal = new AtomicLong(0);
    private final AtomicLong inferFlushBatchCount = new AtomicLong(0);
    private final AtomicLong inferFlushBatchSum = new AtomicLong(0);
    private final AtomicLong inferFlushBatchMax = new AtomicLong(0);

    // Batch size histogram (circular buffer for last 1000 batches)
    private final int[] batchSizeHistory = new int[1000];
    private int batchSizeHistoryIndex = 0;
    private int batchSizeHistoryCount = 0;
    private final Object batchSizeHistoryLock = new Object();

    // Training batch steps histogram (circular buffer for last 1000 training batches)
    private final int[] trainBatchStepsHistory = new int[1000];
    private int trainBatchStepsHistoryIndex = 0;
    private int trainBatchStepsHistoryCount = 0;
    private final Object trainBatchStepsHistoryLock = new Object();

    // Rolling window max for recent 100 training batches
    private final int[] trainBatchStepsRecentWindow = new int[100];
    private int trainBatchStepsRecentWindowIndex = 0;
    private int trainBatchStepsRecentWindowCount = 0;

    // Learner micro-batching metrics (train queue drain -> one learner update)
    private final AtomicLong trainFlushesTotal = new AtomicLong(0);
    private final AtomicLong trainFlushesCappedByEpisodesTotal = new AtomicLong(0);
    private final AtomicLong trainFlushesCappedByStepsTotal = new AtomicLong(0);
    private final AtomicLong trainBatchEpisodesCount = new AtomicLong(0);
    private final AtomicLong trainBatchEpisodesSum = new AtomicLong(0);
    private final AtomicLong trainBatchEpisodesMax = new AtomicLong(0);
    private final AtomicLong trainBatchStepsCount = new AtomicLong(0);
    private final AtomicLong trainBatchStepsSum = new AtomicLong(0);
    private final AtomicLong trainBatchStepsMax = new AtomicLong(0);
    private final AtomicLong trainLatencyCount = new AtomicLong(0);
    private final AtomicLong trainLatencySumMs = new AtomicLong(0);
    private final AtomicLong trainLatencyMaxMs = new AtomicLong(0);

    // Auto-batching telemetry (Python-side decisions)
    private final AtomicLong autoInferSplitsCapTotal = new AtomicLong(0);
    private final AtomicLong autoInferSplitsPagingTotal = new AtomicLong(0);
    private final AtomicLong autoInferSplitsOomTotal = new AtomicLong(0);
    private final AtomicLong autoTrainSplitsCapTotal = new AtomicLong(0);
    private final AtomicLong autoTrainSplitsPagingTotal = new AtomicLong(0);
    private final AtomicLong autoTrainSplitsOomTotal = new AtomicLong(0);

    // Latest observed caps/estimates/headroom (best-effort; last polled)
    private final AtomicLong autoInferSafeMax = new AtomicLong(0);
    private final AtomicLong autoTrainSafeMaxEpisodes = new AtomicLong(0);
    private final AtomicLong autoInferFreeMbBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong autoInferDesiredFreeMbBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong autoInferMbPerSampleBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong autoTrainMbPerStepBits = new AtomicLong(Double.doubleToRawLongBits(0.0));

    // Operation timing metrics (EMA from Python)
    private final AtomicLong trainTimeMsBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong inferTimeMsBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong mulliganTimeMsBits = new AtomicLong(Double.doubleToRawLongBits(0.0));

    // Training loss component metrics (latest values from Python)
    private final AtomicLong trainingPolicyLossBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingValueLossBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingEntropyBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingEntropyCoefBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingClipFracBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingApproxKlBits = new AtomicLong(Double.doubleToRawLongBits(0.0));
    private final AtomicLong trainingAdvantageMeanBits = new AtomicLong(Double.doubleToRawLongBits(0.0));

    // Value head quality metrics (rolling window)
    private static final int VALUE_WINDOW_SIZE = 100;
    private final double[] valueWins = new double[VALUE_WINDOW_SIZE];  // Value predictions for wins
    private final double[] valueLosses = new double[VALUE_WINDOW_SIZE];  // Value predictions for losses
    private final AtomicInteger valueWinIndex = new AtomicInteger(0);
    private final AtomicInteger valueLossIndex = new AtomicInteger(0);
    private final AtomicInteger valueWinCount = new AtomicInteger(0);
    private final AtomicInteger valueLossCount = new AtomicInteger(0);

    // Component-specific metrics
    private final Map<String, AtomicLong> componentMetrics = new ConcurrentHashMap<>();

    // HTTP server for metrics endpoint
    private HttpServer server;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "METRICS-SCHED");
        t.setDaemon(true);
        return t;
    });
    private java.util.concurrent.ExecutorService httpExecutor;

    private MetricsCollector() {
        // Initialize component counters
        componentMetrics.put("worker_episodes", new AtomicLong(0));
        componentMetrics.put("learner_batches", new AtomicLong(0));
        componentMetrics.put("python_bridge_calls", new AtomicLong(0));
        componentMetrics.put("checkpoints_saved", new AtomicLong(0));
    }

    public static MetricsCollector getInstance() {
        return INSTANCE;
    }

    /**
     * Start the metrics HTTP server on the specified port
     */
    public void startMetricsServer(int port) {
        try {
            server = HttpServer.create(new InetSocketAddress(port), 0);
            server.createContext("/metrics", new MetricsHandler());
            server.createContext("/health", new HealthHandler());
            // Use daemon threads so CLI runs can exit cleanly even if caller forgets to stop().
            httpExecutor = Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "METRICS-HTTP");
                t.setDaemon(true);
                return t;
            });
            server.setExecutor(httpExecutor);
            server.start();

            logger.info("Metrics server started on port " + port);

            // Schedule periodic GPU memory collection
            scheduler.scheduleAtFixedRate(this::collectGpuMetrics, 0, 10, TimeUnit.SECONDS);

        } catch (IOException e) {
            logger.error("Failed to start metrics server", e);
        }
    }

    /**
     * Stop the metrics server
     */
    public void stop() {
        if (server != null) {
            server.stop(0);
        }
        scheduler.shutdown();
        if (httpExecutor != null) {
            httpExecutor.shutdownNow();
        }
    }

    // Metric recording methods
    public void recordEpisodeStarted() {
        episodesStarted.incrementAndGet();
    }

    public void recordEpisodeCompleted() {
        episodesCompleted.incrementAndGet();
        componentMetrics.get("worker_episodes").incrementAndGet();
    }

    public void setActiveEpisodes(int active) {
        activeEpisodes.set(Math.max(0, active));
    }

    public void recordInferenceRequest() {
        inferRequestsTotal.incrementAndGet();
    }

    public void recordInferenceTimeout() {
        inferTimeoutsTotal.incrementAndGet();
    }

    public void recordInferenceLatencyMs(long latencyMs) {
        if (latencyMs < 0) {
            return;
        }
        inferLatencySumMs.addAndGet(latencyMs);
        inferLatencyCount.incrementAndGet();
        // best-effort max
        long prev;
        do {
            prev = inferLatencyMaxMs.get();
            if (latencyMs <= prev) {
                break;
            }
        } while (!inferLatencyMaxMs.compareAndSet(prev, latencyMs));
    }

    public void recordInferBatchFlush(int batchSize, boolean dueToFull) {
        if (batchSize <= 0) {
            return;
        }
        inferFlushesTotal.incrementAndGet();
        if (dueToFull) {
            inferFlushesFullTotal.incrementAndGet();
        } else {
            inferFlushesTimeoutTotal.incrementAndGet();
        }
        inferFlushBatchCount.incrementAndGet();
        inferFlushBatchSum.addAndGet(batchSize);
        long prev;
        do {
            prev = inferFlushBatchMax.get();
            if (batchSize <= prev) {
                break;
            }
        } while (!inferFlushBatchMax.compareAndSet(prev, batchSize));

        // Track batch size history for percentiles
        synchronized (batchSizeHistoryLock) {
            batchSizeHistory[batchSizeHistoryIndex] = batchSize;
            batchSizeHistoryIndex = (batchSizeHistoryIndex + 1) % batchSizeHistory.length;
            if (batchSizeHistoryCount < batchSizeHistory.length) {
                batchSizeHistoryCount++;
            }
        }
    }

    public void recordTrainBatchFlush(int episodesInBatch, int stepsInBatch, boolean cappedByEpisodes, boolean cappedBySteps) {
        if (episodesInBatch <= 0 || stepsInBatch <= 0) {
            return;
        }
        trainFlushesTotal.incrementAndGet();
        if (cappedByEpisodes) {
            trainFlushesCappedByEpisodesTotal.incrementAndGet();
        }
        if (cappedBySteps) {
            trainFlushesCappedByStepsTotal.incrementAndGet();
        }

        trainBatchEpisodesCount.incrementAndGet();
        trainBatchEpisodesSum.addAndGet(episodesInBatch);
        long prev;
        do {
            prev = trainBatchEpisodesMax.get();
            if (episodesInBatch <= prev) {
                break;
            }
        } while (!trainBatchEpisodesMax.compareAndSet(prev, episodesInBatch));

        trainBatchStepsCount.incrementAndGet();
        trainBatchStepsSum.addAndGet(stepsInBatch);
        do {
            prev = trainBatchStepsMax.get();
            if (stepsInBatch <= prev) {
                break;
            }
        } while (!trainBatchStepsMax.compareAndSet(prev, stepsInBatch));

        // Track training batch steps history for percentiles
        synchronized (trainBatchStepsHistoryLock) {
            trainBatchStepsHistory[trainBatchStepsHistoryIndex] = stepsInBatch;
            trainBatchStepsHistoryIndex = (trainBatchStepsHistoryIndex + 1) % trainBatchStepsHistory.length;
            if (trainBatchStepsHistoryCount < trainBatchStepsHistory.length) {
                trainBatchStepsHistoryCount++;
            }

            // Also track in recent window for rolling max (last 100 batches)
            trainBatchStepsRecentWindow[trainBatchStepsRecentWindowIndex] = stepsInBatch;
            trainBatchStepsRecentWindowIndex = (trainBatchStepsRecentWindowIndex + 1) % trainBatchStepsRecentWindow.length;
            if (trainBatchStepsRecentWindowCount < trainBatchStepsRecentWindow.length) {
                trainBatchStepsRecentWindowCount++;
            }
        }
    }

    /**
     * Calculate percentile from batch size history
     *
     * @param percentile Value between 0 and 100
     * @return Batch size at the given percentile, or 0 if no data
     */
    private int calculateBatchSizePercentile(double percentile) {
        synchronized (batchSizeHistoryLock) {
            if (batchSizeHistoryCount == 0) {
                return 0;
            }

            // Copy current data
            int[] sorted = new int[batchSizeHistoryCount];
            System.arraycopy(batchSizeHistory, 0, sorted, 0, batchSizeHistoryCount);

            // Sort
            java.util.Arrays.sort(sorted);

            // Calculate index for percentile
            int index = (int) Math.ceil((percentile / 100.0) * batchSizeHistoryCount) - 1;
            if (index < 0) {
                index = 0;
            }
            if (index >= batchSizeHistoryCount) {
                index = batchSizeHistoryCount - 1;
            }

            return sorted[index];
        }
    }

    /**
     * Calculate percentile from training batch steps history
     *
     * @param percentile Value between 0 and 100
     * @return Training batch steps at the given percentile, or 0 if no data
     */
    private int calculateTrainBatchStepsPercentile(double percentile) {
        synchronized (trainBatchStepsHistoryLock) {
            if (trainBatchStepsHistoryCount == 0) {
                return 0;
            }

            // Copy current data
            int[] sorted = new int[trainBatchStepsHistoryCount];
            System.arraycopy(trainBatchStepsHistory, 0, sorted, 0, trainBatchStepsHistoryCount);

            // Sort
            java.util.Arrays.sort(sorted);

            // Calculate index for percentile
            int index = (int) Math.ceil((percentile / 100.0) * trainBatchStepsHistoryCount) - 1;
            if (index < 0) {
                index = 0;
            }
            if (index >= trainBatchStepsHistoryCount) {
                index = trainBatchStepsHistoryCount - 1;
            }

            return sorted[index];
        }
    }

    /**
     * Calculate max from recent training batch steps (last 100 batches)
     *
     * @return Max training batch steps in recent window, or 0 if no data
     */
    private int calculateTrainBatchStepsRecentMax() {
        synchronized (trainBatchStepsHistoryLock) {
            if (trainBatchStepsRecentWindowCount == 0) {
                return 0;
            }

            int max = 0;
            for (int i = 0; i < trainBatchStepsRecentWindowCount; i++) {
                if (trainBatchStepsRecentWindow[i] > max) {
                    max = trainBatchStepsRecentWindow[i];
                }
            }
            return max;
        }
    }

    public void recordTrainLatencyMs(long latencyMs) {
        if (latencyMs < 0) {
            return;
        }
        trainLatencySumMs.addAndGet(latencyMs);
        trainLatencyCount.incrementAndGet();
        long prev;
        do {
            prev = trainLatencyMaxMs.get();
            if (latencyMs <= prev) {
                break;
            }
        } while (!trainLatencyMaxMs.compareAndSet(prev, latencyMs));
    }

    private static long dblBits(double v) {
        return Double.doubleToRawLongBits(v);
    }

    private static double bitsToDbl(long bits) {
        return Double.longBitsToDouble(bits);
    }

    private static double toDouble(Object obj) {
        if (obj == null) {
            return 0.0;
        }
        if (obj instanceof Number) {
            return ((Number) obj).doubleValue();
        }
        try {
            return Double.parseDouble(obj.toString());
        } catch (Exception e) {
            return 0.0;
        }
    }

    public void recordAutoBatchDeltas(
            long inferCapDelta,
            long inferPagingDelta,
            long inferOomDelta,
            long trainCapDelta,
            long trainPagingDelta,
            long trainOomDelta) {
        if (inferCapDelta > 0) {
            autoInferSplitsCapTotal.addAndGet(inferCapDelta);
        }
        if (inferPagingDelta > 0) {
            autoInferSplitsPagingTotal.addAndGet(inferPagingDelta);
        }
        if (inferOomDelta > 0) {
            autoInferSplitsOomTotal.addAndGet(inferOomDelta);
        }
        if (trainCapDelta > 0) {
            autoTrainSplitsCapTotal.addAndGet(trainCapDelta);
        }
        if (trainPagingDelta > 0) {
            autoTrainSplitsPagingTotal.addAndGet(trainPagingDelta);
        }
        if (trainOomDelta > 0) {
            autoTrainSplitsOomTotal.addAndGet(trainOomDelta);
        }
    }

    public void recordAutoBatchSnapshot(
            long inferSafeMaxNow,
            long trainSafeMaxEpisodesNow,
            double inferMbPerSample,
            double trainMbPerStep,
            double freeMb,
            double desiredFreeMb,
            double trainTimeMs,
            double inferTimeMs,
            double mulliganTimeMs) {
        if (inferSafeMaxNow >= 0) {
            autoInferSafeMax.set(inferSafeMaxNow);
        }
        if (trainSafeMaxEpisodesNow >= 0) {
            autoTrainSafeMaxEpisodes.set(trainSafeMaxEpisodesNow);
        }
        autoInferMbPerSampleBits.set(dblBits(inferMbPerSample));
        autoTrainMbPerStepBits.set(dblBits(trainMbPerStep));
        trainTimeMsBits.set(dblBits(trainTimeMs));
        inferTimeMsBits.set(dblBits(inferTimeMs));
        mulliganTimeMsBits.set(dblBits(mulliganTimeMs));
        autoInferFreeMbBits.set(dblBits(freeMb));
        autoInferDesiredFreeMbBits.set(dblBits(desiredFreeMb));
    }

    public void recordTrainingLossComponents(Map<String, Object> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return;
        }
        try {
            trainingPolicyLossBits.set(dblBits(toDouble(metrics.get("policy_loss"))));
            trainingValueLossBits.set(dblBits(toDouble(metrics.get("value_loss"))));
            trainingEntropyBits.set(dblBits(toDouble(metrics.get("entropy"))));
            trainingEntropyCoefBits.set(dblBits(toDouble(metrics.get("entropy_coef"))));
            trainingClipFracBits.set(dblBits(toDouble(metrics.get("clip_frac"))));
            trainingApproxKlBits.set(dblBits(toDouble(metrics.get("approx_kl"))));
            trainingAdvantageMeanBits.set(dblBits(toDouble(metrics.get("advantage_mean"))));
        } catch (Exception ignored) {
        }
    }

    public void recordSamplesProcessed(int samples) {
        samplesProcessed.addAndGet(samples);
    }

    public void recordTrainingBatch(int batchSize, double loss) {
        currentBatchSize.set(batchSize);
        trainingLoss.reset();
        trainingLoss.add(loss);
        trainingUpdates.incrementAndGet();
        componentMetrics.get("learner_batches").incrementAndGet();
    }

    public void recordOptimalBatchSize(int size) {
        optimalBatchSize.set(size);
    }

    public void recordWinRate(double rate) {
        winRate.reset();
        winRate.add(rate);
    }

    public void recordError(String component) {
        errorsTotal.incrementAndGet();
    }

    public void recordPythonBridgeCall() {
        componentMetrics.get("python_bridge_calls").incrementAndGet();
    }

    public void recordCheckpointSaved() {
        componentMetrics.get("checkpoints_saved").incrementAndGet();
    }

    public void recordGpuMemory(long used, long total) {
        gpuMemoryUsed.set(used);
        gpuMemoryTotal.set(total);
    }

    /**
     * Record value head prediction at end of game. This tracks whether the
     * value head correctly predicts wins (positive) vs losses (negative).
     *
     * @param lastValuePrediction The final value prediction from the model
     * @param won True if the RL player won the game
     */
    public void recordValuePrediction(double lastValuePrediction, boolean won) {
        if (won) {
            int idx = valueWinIndex.getAndIncrement() % VALUE_WINDOW_SIZE;
            valueWins[idx] = lastValuePrediction;
            valueWinCount.incrementAndGet();
        } else {
            int idx = valueLossIndex.getAndIncrement() % VALUE_WINDOW_SIZE;
            valueLosses[idx] = lastValuePrediction;
            valueLossCount.incrementAndGet();
        }
    }

    /**
     * Compute value head accuracy: % of games where value sign matches outcome.
     * Win should have positive value, loss should have negative value.
     *
     * Returns the MINIMUM of win_accuracy and loss_accuracy to avoid
     * confounding by winrate. This ensures the metric only shows high accuracy
     * when value head predicts BOTH correctly.
     */
    public double getValueAccuracy() {
        int winSamples = Math.min(valueWinCount.get(), VALUE_WINDOW_SIZE);
        int lossSamples = Math.min(valueLossCount.get(), VALUE_WINDOW_SIZE);
        if (winSamples == 0 || lossSamples == 0) {
            return 0.0;
        }

        int winCorrect = 0;
        for (int i = 0; i < winSamples; i++) {
            if (valueWins[i] > 0) {
                winCorrect++;  // Win should be positive

                    }}

        int lossCorrect = 0;
        for (int i = 0; i < lossSamples; i++) {
            if (valueLosses[i] < 0) {
                lossCorrect++;  // Loss should be negative

                    }}

        double winAccuracy = (double) winCorrect / winSamples;
        double lossAccuracy = (double) lossCorrect / lossSamples;

        // Return minimum to avoid confounding by winrate
        // If value head always predicts negative, win accuracy=0% but loss accuracy=100%
        // Taking the min ensures we only report high accuracy when BOTH are correct
        return Math.min(winAccuracy, lossAccuracy);
    }

    /**
     * Get average value prediction for wins (should be close to +1 ideally).
     */
    public double getAverageValueForWins() {
        int samples = Math.min(valueWinCount.get(), VALUE_WINDOW_SIZE);
        if (samples == 0) {
            return 0.0;
        }
        double sum = 0;
        for (int i = 0; i < samples; i++) {
            sum += valueWins[i];
        }
        return sum / samples;
    }

    /**
     * Get average value prediction for losses (should be close to -1 ideally).
     */
    public double getAverageValueForLosses() {
        int samples = Math.min(valueLossCount.get(), VALUE_WINDOW_SIZE);
        if (samples == 0) {
            return 0.0;
        }
        double sum = 0;
        for (int i = 0; i < samples; i++) {
            sum += valueLosses[i];
        }
        return sum / samples;
    }

    /**
     * Collect GPU metrics from Python bridge
     */
    private void collectGpuMetrics() {
        try {
            // Try to get GPU memory from Python bridge if available
            if (RLTrainer.sharedModel != null) {
                // This would require adding a method to PythonMLBridge
                // For now, we'll simulate or leave as 0
                logger.debug("Collecting GPU metrics...");
            }
        } catch (Exception e) {
            logger.debug("GPU metrics collection failed: " + e.getMessage());
        }
    }

    /**
     * HTTP handler for /metrics endpoint
     */
    private class MetricsHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("GET".equals(exchange.getRequestMethod())) {
                String response = generatePrometheusMetrics();
                exchange.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
                exchange.sendResponseHeaders(200, response.getBytes().length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(response.getBytes());
                }
            } else {
                exchange.sendResponseHeaders(405, -1);
            }
        }
    }

    /**
     * HTTP handler for /health endpoint
     */
    private class HealthHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String response = "OK";
            exchange.sendResponseHeaders(200, response.getBytes().length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response.getBytes());
            }
        }
    }

    /**
     * Generate Prometheus-formatted metrics
     */
    private String generatePrometheusMetrics() {
        StringBuilder sb = new StringBuilder();

        // Episodes metrics
        sb.append("# HELP mage_episodes_started_total Total number of episodes started\n");
        sb.append("# TYPE mage_episodes_started_total counter\n");
        sb.append("mage_episodes_started_total ").append(episodesStarted.get()).append("\n");

        sb.append("# HELP mage_episodes_completed_total Total number of episodes completed\n");
        sb.append("# TYPE mage_episodes_completed_total counter\n");
        sb.append("mage_episodes_completed_total ").append(episodesCompleted.get()).append("\n");

        sb.append("# HELP mage_active_episodes Number of episodes currently running\n");
        sb.append("# TYPE mage_active_episodes gauge\n");
        sb.append("mage_active_episodes ").append(activeEpisodes.get()).append("\n");

        // Inference metrics
        sb.append("# HELP mage_infer_requests_total Total candidate scoring requests\n");
        sb.append("# TYPE mage_infer_requests_total counter\n");
        sb.append("mage_infer_requests_total ").append(inferRequestsTotal.get()).append("\n");

        sb.append("# HELP mage_infer_timeouts_total Candidate scoring timeouts\n");
        sb.append("# TYPE mage_infer_timeouts_total counter\n");
        sb.append("mage_infer_timeouts_total ").append(inferTimeoutsTotal.get()).append("\n");

        double inferAvgMs = 0.0;
        long inferCnt = inferLatencyCount.get();
        if (inferCnt > 0) {
            inferAvgMs = inferLatencySumMs.get() / (double) inferCnt;
        }
        sb.append("# HELP mage_infer_latency_avg_ms Average candidate scoring latency (ms)\n");
        sb.append("# TYPE mage_infer_latency_avg_ms gauge\n");
        sb.append("mage_infer_latency_avg_ms ").append(String.format("%.3f", inferAvgMs)).append("\n");

        sb.append("# HELP mage_infer_latency_max_ms Max candidate scoring latency observed (ms)\n");
        sb.append("# TYPE mage_infer_latency_max_ms gauge\n");
        sb.append("mage_infer_latency_max_ms ").append(inferLatencyMaxMs.get()).append("\n");

        // Effective runtime config (helps debug "did env var apply?")
        sb.append("# HELP mage_config_py_batch_max_size Effective PY_BATCH_MAX_SIZE loaded by JVM\n");
        sb.append("# TYPE mage_config_py_batch_max_size gauge\n");
        sb.append("mage_config_py_batch_max_size ").append(PythonMLBatchManager.getConfiguredMaxBatchSize()).append("\n");

        sb.append("# HELP mage_config_py_batch_timeout_ms Effective PY_BATCH_TIMEOUT_MS loaded by JVM\n");
        sb.append("# TYPE mage_config_py_batch_timeout_ms gauge\n");
        sb.append("mage_config_py_batch_timeout_ms ").append(PythonMLBatchManager.getConfiguredBatchTimeoutMs()).append("\n");

        // Inference batching metrics
        sb.append("# HELP mage_infer_flushes_total Number of inference batch flushes\n");
        sb.append("# TYPE mage_infer_flushes_total counter\n");
        sb.append("mage_infer_flushes_total ").append(inferFlushesTotal.get()).append("\n");

        sb.append("# HELP mage_infer_flushes_full_total Inference batch flushes due to reaching PY_BATCH_MAX_SIZE\n");
        sb.append("# TYPE mage_infer_flushes_full_total counter\n");
        sb.append("mage_infer_flushes_full_total ").append(inferFlushesFullTotal.get()).append("\n");

        sb.append("# HELP mage_infer_flushes_timeout_total Inference batch flushes due to PY_BATCH_TIMEOUT_MS\n");
        sb.append("# TYPE mage_infer_flushes_timeout_total counter\n");
        sb.append("mage_infer_flushes_timeout_total ").append(inferFlushesTimeoutTotal.get()).append("\n");

        double inferBatchAvg = 0.0;
        long bc = inferFlushBatchCount.get();
        if (bc > 0) {
            inferBatchAvg = inferFlushBatchSum.get() / (double) bc;
        }
        sb.append("# HELP mage_infer_batch_avg_size Average batch size at inference flush\n");
        sb.append("# TYPE mage_infer_batch_avg_size gauge\n");
        sb.append("mage_infer_batch_avg_size ").append(String.format("%.3f", inferBatchAvg)).append("\n");

        sb.append("# HELP mage_infer_batch_max_size Max batch size seen at inference flush\n");
        sb.append("# TYPE mage_infer_batch_max_size gauge\n");
        sb.append("mage_infer_batch_max_size ").append(inferFlushBatchMax.get()).append("\n");

        // Batch size percentiles (from last 1000 batches)
        sb.append("# HELP mage_infer_batch_size_p50 50th percentile (median) of inference batch sizes\n");
        sb.append("# TYPE mage_infer_batch_size_p50 gauge\n");
        sb.append("mage_infer_batch_size_p50 ").append(calculateBatchSizePercentile(50)).append("\n");

        sb.append("# HELP mage_infer_batch_size_p90 90th percentile of inference batch sizes\n");
        sb.append("# TYPE mage_infer_batch_size_p90 gauge\n");
        sb.append("mage_infer_batch_size_p90 ").append(calculateBatchSizePercentile(90)).append("\n");

        sb.append("# HELP mage_infer_batch_size_p95 95th percentile of inference batch sizes\n");
        sb.append("# TYPE mage_infer_batch_size_p95 gauge\n");
        sb.append("mage_infer_batch_size_p95 ").append(calculateBatchSizePercentile(95)).append("\n");

        sb.append("# HELP mage_infer_batch_size_p99 99th percentile of inference batch sizes\n");
        sb.append("# TYPE mage_infer_batch_size_p99 gauge\n");
        sb.append("mage_infer_batch_size_p99 ").append(calculateBatchSizePercentile(99)).append("\n");

        // Learner micro-batching metrics
        sb.append("# HELP mage_train_flushes_total Number of learner micro-batch flushes (one learner update)\n");
        sb.append("# TYPE mage_train_flushes_total counter\n");
        sb.append("mage_train_flushes_total ").append(trainFlushesTotal.get()).append("\n");

        sb.append("# HELP mage_train_flushes_capped_by_episodes_total Learner batches capped by LEARNER_BATCH_MAX_EPISODES\n");
        sb.append("# TYPE mage_train_flushes_capped_by_episodes_total counter\n");
        sb.append("mage_train_flushes_capped_by_episodes_total ").append(trainFlushesCappedByEpisodesTotal.get()).append("\n");

        sb.append("# HELP mage_train_flushes_capped_by_steps_total Learner batches capped by LEARNER_BATCH_MAX_STEPS\n");
        sb.append("# TYPE mage_train_flushes_capped_by_steps_total counter\n");
        sb.append("mage_train_flushes_capped_by_steps_total ").append(trainFlushesCappedByStepsTotal.get()).append("\n");

        double trainEpAvg = 0.0;
        long trainEpCnt = trainBatchEpisodesCount.get();
        if (trainEpCnt > 0) {
            trainEpAvg = trainBatchEpisodesSum.get() / (double) trainEpCnt;
        }
        sb.append("# HELP mage_train_batch_avg_episodes Average episodes per learner update\n");
        sb.append("# TYPE mage_train_batch_avg_episodes gauge\n");
        sb.append("mage_train_batch_avg_episodes ").append(String.format("%.3f", trainEpAvg)).append("\n");

        sb.append("# HELP mage_train_batch_max_episodes Max episodes per learner update\n");
        sb.append("# TYPE mage_train_batch_max_episodes gauge\n");
        sb.append("mage_train_batch_max_episodes ").append(trainBatchEpisodesMax.get()).append("\n");

        double trainStepsAvg = 0.0;
        long trainStepsCnt = trainBatchStepsCount.get();
        if (trainStepsCnt > 0) {
            trainStepsAvg = trainBatchStepsSum.get() / (double) trainStepsCnt;
        }
        sb.append("# HELP mage_train_batch_avg_steps Average steps per learner update\n");
        sb.append("# TYPE mage_train_batch_avg_steps gauge\n");
        sb.append("mage_train_batch_avg_steps ").append(String.format("%.3f", trainStepsAvg)).append("\n");

        sb.append("# HELP mage_train_batch_max_steps Max steps per learner update\n");
        sb.append("# TYPE mage_train_batch_max_steps gauge\n");
        sb.append("mage_train_batch_max_steps ").append(trainBatchStepsMax.get()).append("\n");

        // Training batch steps percentiles (from last 1000 training batches)
        sb.append("# HELP mage_train_batch_steps_p50 50th percentile (median) of training batch step sizes\n");
        sb.append("# TYPE mage_train_batch_steps_p50 gauge\n");
        sb.append("mage_train_batch_steps_p50 ").append(calculateTrainBatchStepsPercentile(50)).append("\n");

        sb.append("# HELP mage_train_batch_steps_p90 90th percentile of training batch step sizes\n");
        sb.append("# TYPE mage_train_batch_steps_p90 gauge\n");
        sb.append("mage_train_batch_steps_p90 ").append(calculateTrainBatchStepsPercentile(90)).append("\n");

        sb.append("# HELP mage_train_batch_steps_p95 95th percentile of training batch step sizes\n");
        sb.append("# TYPE mage_train_batch_steps_p95 gauge\n");
        sb.append("mage_train_batch_steps_p95 ").append(calculateTrainBatchStepsPercentile(95)).append("\n");

        sb.append("# HELP mage_train_batch_steps_p99 99th percentile of training batch step sizes\n");
        sb.append("# TYPE mage_train_batch_steps_p99 gauge\n");
        sb.append("mage_train_batch_steps_p99 ").append(calculateTrainBatchStepsPercentile(99)).append("\n");

        sb.append("# HELP mage_train_batch_steps_recent_max Max training batch steps in last 100 batches (rolling window)\n");
        sb.append("# TYPE mage_train_batch_steps_recent_max gauge\n");
        sb.append("mage_train_batch_steps_recent_max ").append(calculateTrainBatchStepsRecentMax()).append("\n");

        double trainLatAvg = 0.0;
        long trainLatCnt = trainLatencyCount.get();
        if (trainLatCnt > 0) {
            trainLatAvg = trainLatencySumMs.get() / (double) trainLatCnt;
        }
        sb.append("# HELP mage_train_latency_avg_ms Average learner train() call latency (ms)\n");
        sb.append("# TYPE mage_train_latency_avg_ms gauge\n");
        sb.append("mage_train_latency_avg_ms ").append(String.format("%.3f", trainLatAvg)).append("\n");

        sb.append("# HELP mage_train_latency_max_ms Max learner train() call latency observed (ms)\n");
        sb.append("# TYPE mage_train_latency_max_ms gauge\n");
        sb.append("mage_train_latency_max_ms ").append(trainLatencyMaxMs.get()).append("\n");

        // Auto-batching telemetry (Python-side decisions)
        sb.append("# HELP mage_autobatch_infer_splits_cap_total Inference auto-batch splits due to cap\n");
        sb.append("# TYPE mage_autobatch_infer_splits_cap_total counter\n");
        sb.append("mage_autobatch_infer_splits_cap_total ").append(autoInferSplitsCapTotal.get()).append("\n");
        sb.append("# HELP mage_autobatch_infer_splits_paging_total Inference auto-batch splits to maintain VRAM headroom\n");
        sb.append("# TYPE mage_autobatch_infer_splits_paging_total counter\n");
        sb.append("mage_autobatch_infer_splits_paging_total ").append(autoInferSplitsPagingTotal.get()).append("\n");
        sb.append("# HELP mage_autobatch_infer_splits_oom_total Inference auto-batch splits after CUDA OOM\n");
        sb.append("# TYPE mage_autobatch_infer_splits_oom_total counter\n");
        sb.append("mage_autobatch_infer_splits_oom_total ").append(autoInferSplitsOomTotal.get()).append("\n");

        sb.append("# HELP mage_autobatch_train_splits_cap_total Train auto-batch splits due to cap\n");
        sb.append("# TYPE mage_autobatch_train_splits_cap_total counter\n");
        sb.append("mage_autobatch_train_splits_cap_total ").append(autoTrainSplitsCapTotal.get()).append("\n");
        sb.append("# HELP mage_autobatch_train_splits_paging_total Train auto-batch splits to maintain VRAM headroom\n");
        sb.append("# TYPE mage_autobatch_train_splits_paging_total counter\n");
        sb.append("mage_autobatch_train_splits_paging_total ").append(autoTrainSplitsPagingTotal.get()).append("\n");
        sb.append("# HELP mage_autobatch_train_splits_oom_total Train auto-batch splits after CUDA OOM\n");
        sb.append("# TYPE mage_autobatch_train_splits_oom_total counter\n");
        sb.append("mage_autobatch_train_splits_oom_total ").append(autoTrainSplitsOomTotal.get()).append("\n");

        sb.append("# HELP mage_autobatch_infer_safe_max Current learned/proactive inference cap (0=unset)\n");
        sb.append("# TYPE mage_autobatch_infer_safe_max gauge\n");
        sb.append("mage_autobatch_infer_safe_max ").append(autoInferSafeMax.get()).append("\n");
        sb.append("# HELP mage_autobatch_train_safe_max_episodes Current learned/proactive train episode cap (0=unset)\n");
        sb.append("# TYPE mage_autobatch_train_safe_max_episodes gauge\n");
        sb.append("mage_autobatch_train_safe_max_episodes ").append(autoTrainSafeMaxEpisodes.get()).append("\n");

        sb.append("# HELP mage_autobatch_infer_mb_per_sample Estimated extra MB per inference sample\n");
        sb.append("# TYPE mage_autobatch_infer_mb_per_sample gauge\n");
        sb.append("mage_autobatch_infer_mb_per_sample ").append(String.format("%.4f", bitsToDbl(autoInferMbPerSampleBits.get()))).append("\n");
        sb.append("# HELP mage_autobatch_train_mb_per_step Estimated extra MB per train step\n");
        sb.append("# TYPE mage_autobatch_train_mb_per_step gauge\n");
        sb.append("mage_autobatch_train_mb_per_step ").append(String.format("%.4f", bitsToDbl(autoTrainMbPerStepBits.get()))).append("\n");

        sb.append("# HELP mage_autobatch_free_mb Last observed free VRAM (MB) from Python\n");
        sb.append("# TYPE mage_autobatch_free_mb gauge\n");
        sb.append("mage_autobatch_free_mb ").append(String.format("%.1f", bitsToDbl(autoInferFreeMbBits.get()))).append("\n");
        sb.append("# HELP mage_autobatch_desired_free_mb Target free VRAM headroom (MB)\n");
        sb.append("# TYPE mage_autobatch_desired_free_mb gauge\n");
        sb.append("mage_autobatch_desired_free_mb ").append(String.format("%.1f", bitsToDbl(autoInferDesiredFreeMbBits.get()))).append("\n");

        // Operation timing metrics (EMA from Python)
        sb.append("# HELP mage_train_time_ms Average training operation time (ms, EMA)\n");
        sb.append("# TYPE mage_train_time_ms gauge\n");
        sb.append("mage_train_time_ms ").append(String.format("%.2f", bitsToDbl(trainTimeMsBits.get()))).append("\n");
        sb.append("# HELP mage_infer_time_ms Average inference operation time (ms, EMA)\n");
        sb.append("# TYPE mage_infer_time_ms gauge\n");
        sb.append("mage_infer_time_ms ").append(String.format("%.2f", bitsToDbl(inferTimeMsBits.get()))).append("\n");
        sb.append("# HELP mage_mulligan_time_ms Average mulligan operation time (ms, EMA)\n");
        sb.append("# TYPE mage_mulligan_time_ms gauge\n");
        sb.append("mage_mulligan_time_ms ").append(String.format("%.2f", bitsToDbl(mulliganTimeMsBits.get()))).append("\n");

        // Samples metrics
        sb.append("# HELP mage_samples_processed_total Total number of samples processed\n");
        sb.append("# TYPE mage_samples_processed_total counter\n");
        sb.append("mage_samples_processed_total ").append(samplesProcessed.get()).append("\n");

        // Training metrics
        sb.append("# HELP mage_training_loss Current training loss\n");
        sb.append("# TYPE mage_training_loss gauge\n");
        sb.append("mage_training_loss ").append(trainingLoss.doubleValue()).append("\n");

        sb.append("# HELP mage_training_policy_loss Policy component of training loss\n");
        sb.append("# TYPE mage_training_policy_loss gauge\n");
        sb.append("mage_training_policy_loss ").append(String.format("%.6f", bitsToDbl(trainingPolicyLossBits.get()))).append("\n");

        sb.append("# HELP mage_training_value_loss Value component of training loss\n");
        sb.append("# TYPE mage_training_value_loss gauge\n");
        sb.append("mage_training_value_loss ").append(String.format("%.6f", bitsToDbl(trainingValueLossBits.get()))).append("\n");

        sb.append("# HELP mage_training_entropy Policy entropy\n");
        sb.append("# TYPE mage_training_entropy gauge\n");
        sb.append("mage_training_entropy ").append(String.format("%.6f", bitsToDbl(trainingEntropyBits.get()))).append("\n");

        sb.append("# HELP mage_training_entropy_coef Entropy coefficient (loss weight)\n");
        sb.append("# TYPE mage_training_entropy_coef gauge\n");
        sb.append("mage_training_entropy_coef ").append(String.format("%.6f", bitsToDbl(trainingEntropyCoefBits.get()))).append("\n");

        sb.append("# HELP mage_training_clip_frac PPO clip fraction\n");
        sb.append("# TYPE mage_training_clip_frac gauge\n");
        sb.append("mage_training_clip_frac ").append(String.format("%.4f", bitsToDbl(trainingClipFracBits.get()))).append("\n");

        sb.append("# HELP mage_training_approx_kl PPO approximate KL divergence\n");
        sb.append("# TYPE mage_training_approx_kl gauge\n");
        sb.append("mage_training_approx_kl ").append(String.format("%.6f", bitsToDbl(trainingApproxKlBits.get()))).append("\n");

        sb.append("# HELP mage_training_advantage_mean Mean advantage value\n");
        sb.append("# TYPE mage_training_advantage_mean gauge\n");
        sb.append("mage_training_advantage_mean ").append(String.format("%.6f", bitsToDbl(trainingAdvantageMeanBits.get()))).append("\n");

        sb.append("# HELP mage_training_updates_total Total number of training updates\n");
        sb.append("# TYPE mage_training_updates_total counter\n");
        sb.append("mage_training_updates_total ").append(trainingUpdates.get()).append("\n");

        // Actor-learner queue depth (if enabled)
        int qDepth = 0;
        long qDropped = 0;
        try {
            if (RLTrainer.sharedModel instanceof PythonMLService) {
                qDepth = ((PythonMLService) RLTrainer.sharedModel).getTrainQueueDepth();
                qDropped = ((PythonMLService) RLTrainer.sharedModel).getDroppedTrainEpisodes();
            }
        } catch (Exception ignored) {
        }
        sb.append("# HELP mage_train_queue_depth Number of episodes waiting to train\n");
        sb.append("# TYPE mage_train_queue_depth gauge\n");
        sb.append("mage_train_queue_depth ").append(qDepth).append("\n");
        sb.append("# HELP mage_train_queue_dropped_total Episodes dropped because train queue was full\n");
        sb.append("# TYPE mage_train_queue_dropped_total counter\n");
        sb.append("mage_train_queue_dropped_total ").append(qDropped).append("\n");

        // Batch size metrics
        sb.append("# HELP mage_current_batch_size Current batch size used for training\n");
        sb.append("# TYPE mage_current_batch_size gauge\n");
        sb.append("mage_current_batch_size ").append(currentBatchSize.get()).append("\n");

        sb.append("# HELP mage_optimal_batch_size Optimal batch size calculated by GPU memory\n");
        sb.append("# TYPE mage_optimal_batch_size gauge\n");
        sb.append("mage_optimal_batch_size ").append(optimalBatchSize.get()).append("\n");

        // Performance metrics
        sb.append("# HELP mage_win_rate Current win rate in evaluation\n");
        sb.append("# TYPE mage_win_rate gauge\n");
        sb.append("mage_win_rate ").append(winRate.doubleValue()).append("\n");

        // Value head quality metrics
        sb.append("# HELP mage_value_accuracy Accuracy of value predictions (win=positive, loss=negative)\n");
        sb.append("# TYPE mage_value_accuracy gauge\n");
        sb.append("mage_value_accuracy ").append(String.format("%.4f", getValueAccuracy())).append("\n");

        sb.append("# HELP mage_value_avg_wins Average value prediction for games won (should be near +1)\n");
        sb.append("# TYPE mage_value_avg_wins gauge\n");
        sb.append("mage_value_avg_wins ").append(String.format("%.4f", getAverageValueForWins())).append("\n");

        sb.append("# HELP mage_value_avg_losses Average value prediction for games lost (should be near -1)\n");
        sb.append("# TYPE mage_value_avg_losses gauge\n");
        sb.append("mage_value_avg_losses ").append(String.format("%.4f", getAverageValueForLosses())).append("\n");

        // Error metrics
        sb.append("# HELP mage_errors_total Total number of errors encountered\n");
        sb.append("# TYPE mage_errors_total counter\n");
        sb.append("mage_errors_total ").append(errorsTotal.get()).append("\n");

        // GPU metrics
        if (gpuMemoryTotal.get() > 0) {
            sb.append("# HELP mage_gpu_memory_used_bytes GPU memory currently used\n");
            sb.append("# TYPE mage_gpu_memory_used_bytes gauge\n");
            sb.append("mage_gpu_memory_used_bytes ").append(gpuMemoryUsed.get()).append("\n");

            sb.append("# HELP mage_gpu_memory_total_bytes Total GPU memory available\n");
            sb.append("# TYPE mage_gpu_memory_total_bytes gauge\n");
            sb.append("mage_gpu_memory_total_bytes ").append(gpuMemoryTotal.get()).append("\n");
        }

        // Component-specific metrics
        for (Map.Entry<String, AtomicLong> entry : componentMetrics.entrySet()) {
            String metricName = "mage_" + entry.getKey() + "_total";
            sb.append("# HELP ").append(metricName).append(" Total count for ").append(entry.getKey()).append("\n");
            sb.append("# TYPE ").append(metricName).append(" counter\n");
            sb.append(metricName).append(" ").append(entry.getValue().get()).append("\n");
        }

        // System info
        Runtime runtime = Runtime.getRuntime();
        sb.append("# HELP mage_jvm_memory_used_bytes JVM memory used\n");
        sb.append("# TYPE mage_jvm_memory_used_bytes gauge\n");
        sb.append("mage_jvm_memory_used_bytes ").append(runtime.totalMemory() - runtime.freeMemory()).append("\n");

        sb.append("# HELP mage_jvm_memory_max_bytes JVM maximum memory\n");
        sb.append("# TYPE mage_jvm_memory_max_bytes gauge\n");
        sb.append("mage_jvm_memory_max_bytes ").append(runtime.maxMemory()).append("\n");

        return sb.toString();
    }
}
