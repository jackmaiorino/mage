package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;

import mage.player.ai.ComputerPlayerRL;

/**
 * Centralized tracking of training health metrics with periodic logging to
 * file.
 *
 * Tracks: - Games killed by health monitor (stuck/infinite loops) - RL
 * activation failures (invalid action selections) - GPU OOM errors - Other
 * anomalies
 */
public class TrainingHealthStats {

    private static final Logger logger = Logger.getLogger(TrainingHealthStats.class);

    private static final String HEALTH_LOG_PATH = RLLogPaths.HEALTH_LOG_PATH;
    private static final int LOG_INTERVAL_SEC = EnvConfig.i32("HEALTH_LOG_INTERVAL_SEC", 60);

    // Singleton instance
    private static TrainingHealthStats instance;

    // Counters
    private final AtomicInteger gpuOomCount = new AtomicInteger(0);
    private final AtomicInteger pythonErrors = new AtomicInteger(0);
    private final AtomicInteger modelNanCount = new AtomicInteger(0);
    private final AtomicLong lastWriteMs = new AtomicLong(0);

    // Scheduler for periodic writes
    private ScheduledExecutorService scheduler;
    private volatile boolean running = false;

    // Start time for uptime tracking
    private final long startTimeMs = System.currentTimeMillis();

    private TrainingHealthStats() {
        // Initialize log file with CSV header
        try {
            Path path = Paths.get(HEALTH_LOG_PATH);
            Files.createDirectories(path.getParent());
            if (!Files.exists(path)) {
                String header = "timestamp,uptime_min,games_killed,rl_activation_failures,gpu_ooms,python_errors,model_nans\n";
                Files.write(path, header.getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException e) {
            logger.warn("Failed to initialize health log file: " + e.getMessage());
        }
    }

    public static synchronized TrainingHealthStats getInstance() {
        if (instance == null) {
            instance = new TrainingHealthStats();
        }
        return instance;
    }

    /**
     * Start periodic health logging.
     */
    public void start() {
        if (running) {
            return;
        }
        running = true;
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "HealthStatsLogger");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::writeStats, LOG_INTERVAL_SEC, LOG_INTERVAL_SEC, TimeUnit.SECONDS);
        logger.info("Training health stats logging started (interval=" + LOG_INTERVAL_SEC + "s, path=" + HEALTH_LOG_PATH + ")");
    }

    /**
     * Stop periodic health logging.
     */
    public void stop() {
        running = false;
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }
        // Final write
        writeStats();
    }

    /**
     * Record a GPU OOM error.
     */
    public void recordGpuOom() {
        gpuOomCount.incrementAndGet();
    }

    /**
     * Record a Python bridge error.
     */
    public void recordPythonError() {
        pythonErrors.incrementAndGet();
    }

    /**
     * Record model producing NaN outputs.
     */
    public void recordModelNan() {
        modelNanCount.incrementAndGet();
    }

    public int getGpuOomCount() {
        return getGpuOomCountFromPython();
    }

    /**
     * Get GPU OOM count from Python via the bridge.
     */
    private int getGpuOomCountFromPython() {
        try {
            PythonModel model = PythonMLService.getInstance();
            if (model != null) {
                java.util.Map<String, Integer> stats = model.getHealthStats();
                if (stats != null && stats.containsKey("gpu_oom_count")) {
                    return stats.get("gpu_oom_count");
                }
            }
        } catch (Exception e) {
            // Ignore errors - Python might not be initialized
        }
        return gpuOomCount.get(); // Fallback to local counter
    }

    public int getPythonErrorCount() {
        return pythonErrors.get();
    }

    public int getModelNanCount() {
        return modelNanCount.get();
    }

    /**
     * Write current stats to log file.
     */
    private void writeStats() {
        try {
            long now = System.currentTimeMillis();
            long uptimeMin = (now - startTimeMs) / 60000;

            // Gather all stats
            int gamesKilled = GameHealthMonitor.getGamesKilled();
            int rlFailures = ComputerPlayerRL.getRLActivationFailureCount();
            int gpuOoms = getGpuOomCountFromPython();
            int pyErrors = pythonErrors.get();
            int modelNans = modelNanCount.get();

            // Format: timestamp,uptime_min,games_killed,rl_activation_failures,gpu_ooms,python_errors,model_nans
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            String line = String.format("%s,%d,%d,%d,%d,%d,%d\n",
                    timestamp, uptimeMin, gamesKilled, rlFailures, gpuOoms, pyErrors, modelNans);

            Path path = Paths.get(HEALTH_LOG_PATH);
            Files.write(path, line.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);

            lastWriteMs.set(now);

            // Log to console if any issues detected
            if (gamesKilled > 0 || rlFailures > 0 || gpuOoms > 0 || pyErrors > 0 || modelNans > 0) {
                logger.info(String.format("Health stats: games_killed=%d, rl_failures=%d, gpu_ooms=%d, py_errors=%d, model_nans=%d",
                        gamesKilled, rlFailures, gpuOoms, pyErrors, modelNans));
            }
        } catch (IOException e) {
            logger.warn("Failed to write health stats: " + e.getMessage());
        }
    }

    /**
     * Get a summary string for logging.
     */
    public String getSummary() {
        int gamesKilled = GameHealthMonitor.getGamesKilled();
        int rlFailures = ComputerPlayerRL.getRLActivationFailureCount();
        int gpuOoms = getGpuOomCountFromPython();
        int pyErrors = pythonErrors.get();
        int modelNans = modelNanCount.get();

        return String.format("games_killed=%d, rl_failures=%d, gpu_ooms=%d, py_errors=%d, model_nans=%d",
                gamesKilled, rlFailures, gpuOoms, pyErrors, modelNans);
    }

    /**
     * Reset all counters (e.g., at start of new training run).
     */
    public void reset() {
        gpuOomCount.set(0);
        pythonErrors.set(0);
        modelNanCount.set(0);
        GameHealthMonitor.resetGamesKilled();
        ComputerPlayerRL.resetRLActivationFailureCount();
    }
}
