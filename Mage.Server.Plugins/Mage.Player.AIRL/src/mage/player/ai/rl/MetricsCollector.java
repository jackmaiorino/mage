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
    private final AtomicLong episodesCompleted = new AtomicLong(0);
    private final AtomicLong samplesProcessed = new AtomicLong(0);
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private final AtomicInteger optimalBatchSize = new AtomicInteger(0);
    private final DoubleAdder trainingLoss = new DoubleAdder();
    private final DoubleAdder winRate = new DoubleAdder();
    private final AtomicLong errorsTotal = new AtomicLong(0);
    private final AtomicLong trainingUpdates = new AtomicLong(0);
    private final AtomicLong gpuMemoryUsed = new AtomicLong(0);
    private final AtomicLong gpuMemoryTotal = new AtomicLong(0);

    // Component-specific metrics
    private final Map<String, AtomicLong> componentMetrics = new ConcurrentHashMap<>();

    // HTTP server for metrics endpoint
    private HttpServer server;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

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
            server.setExecutor(null);
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
    }

    // Metric recording methods
    public void recordEpisodeCompleted() {
        episodesCompleted.incrementAndGet();
        componentMetrics.get("worker_episodes").incrementAndGet();
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
        sb.append("# HELP mage_episodes_completed_total Total number of episodes completed\n");
        sb.append("# TYPE mage_episodes_completed_total counter\n");
        sb.append("mage_episodes_completed_total ").append(episodesCompleted.get()).append("\n");

        // Samples metrics
        sb.append("# HELP mage_samples_processed_total Total number of samples processed\n");
        sb.append("# TYPE mage_samples_processed_total counter\n");
        sb.append("mage_samples_processed_total ").append(samplesProcessed.get()).append("\n");

        // Training metrics
        sb.append("# HELP mage_training_loss Current training loss\n");
        sb.append("# TYPE mage_training_loss gauge\n");
        sb.append("mage_training_loss ").append(trainingLoss.doubleValue()).append("\n");

        sb.append("# HELP mage_training_updates_total Total number of training updates\n");
        sb.append("# TYPE mage_training_updates_total counter\n");
        sb.append("mage_training_updates_total ").append(trainingUpdates.get()).append("\n");

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
