package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;

import mage.game.Game;

/**
 * Monitors game health to detect infinite loops and stuck games.
 *
 * Detects: 1. Games that run longer than a timeout 2. Repeated log messages
 * (same warning/error spam = likely infinite loop) 3. Games with no state
 * changes for extended periods
 */
public class GameHealthMonitor {

    private static final Logger logger = Logger.getLogger(GameHealthMonitor.class);

    // Config from environment
    private static final int GAME_TIMEOUT_SEC = EnvConfig.i32("GAME_TIMEOUT_SEC", 300); // 5 min default
    private static final int REPEAT_THRESHOLD = EnvConfig.i32("HEALTH_REPEAT_THRESHOLD", 50); // Same message 50x = stuck
    private static final int REPEAT_WINDOW_MS = EnvConfig.i32("HEALTH_REPEAT_WINDOW_MS", 5000); // Within 5 seconds
    private static final boolean ENABLED = EnvConfig.bool("GAME_HEALTH_MONITOR", true);

    // Stats
    private static final AtomicInteger GAMES_KILLED = new AtomicInteger(0);
    private static final AtomicInteger GAMES_MONITORED = new AtomicInteger(0);

    private final Game game;
    private final long startTimeMs;
    private final int gameTimeoutSec;
    private final Thread watchdogThread;
    private volatile boolean stopped = false;
    private volatile String killReason = null;

    // Track repeated messages
    private final Map<String, MessageTracker> messageTrackers = new ConcurrentHashMap<>();

    public GameHealthMonitor(Game game) {
        this(game, GAME_TIMEOUT_SEC);
    }

    public GameHealthMonitor(Game game, int timeoutSec) {
        this.game = game;
        this.startTimeMs = System.currentTimeMillis();
        this.gameTimeoutSec = timeoutSec > 0 ? timeoutSec : GAME_TIMEOUT_SEC;

        if (ENABLED) {
            GAMES_MONITORED.incrementAndGet();
            this.watchdogThread = new Thread(this::watchdogLoop, "GameHealthWatchdog-" + game.getId());
            this.watchdogThread.setDaemon(true);
        } else {
            this.watchdogThread = null;
        }
    }

    /**
     * Start monitoring the game.
     */
    public void start() {
        if (watchdogThread != null && !stopped) {
            watchdogThread.start();
        }
    }

    /**
     * Stop monitoring (call when game ends normally).
     */
    public void stop() {
        stopped = true;
        if (watchdogThread != null) {
            watchdogThread.interrupt();
        }
    }

    /**
     * Record a log message for repeat detection. Call this from a log appender
     * or interceptor.
     */
    public void recordMessage(String message) {
        if (stopped || message == null) {
            return;
        }

        // Normalize message (remove timestamps, IDs, etc.)
        String normalized = normalizeMessage(message);

        MessageTracker tracker = messageTrackers.computeIfAbsent(normalized, k -> new MessageTracker());
        tracker.record();

        // Check for repeat spam
        if (tracker.getCountInWindow(REPEAT_WINDOW_MS) >= REPEAT_THRESHOLD) {
            killReason = "Repeated message detected (" + REPEAT_THRESHOLD + "x in "
                    + REPEAT_WINDOW_MS + "ms): " + truncate(normalized, 100);
            killGame();
        }
    }

    /**
     * Check if game was killed by the monitor.
     */
    public boolean wasKilled() {
        return killReason != null;
    }

    /**
     * Get reason if game was killed.
     */
    public String getKillReason() {
        return killReason;
    }

    public static int getGamesKilled() {
        return GAMES_KILLED.get();
    }

    public static int getGamesMonitored() {
        return GAMES_MONITORED.get();
    }

    /**
     * Reset the games killed counter (e.g., at start of new training run).
     */
    public static void resetGamesKilled() {
        GAMES_KILLED.set(0);
        GAMES_MONITORED.set(0);
    }

    private void watchdogLoop() {
        try {
            // Check every second
            while (!stopped && !Thread.currentThread().isInterrupted()) {
                Thread.sleep(1000);

                // Timeout check
                long elapsedSec = (System.currentTimeMillis() - startTimeMs) / 1000;
                if (elapsedSec > gameTimeoutSec) {
                    if (game.getState() != null && !game.getState().isGameOver()) {
                        killReason = "Game timeout after " + elapsedSec + " seconds (limit: " + gameTimeoutSec + "s)";
                        killGame();
                        break;
                    }
                }

                // Game already over - stop monitoring
                if (game.getState() == null || game.getState().isGameOver()) {
                    break;
                }
            }
        } catch (InterruptedException e) {
            // Normal shutdown
            Thread.currentThread().interrupt();
        }
    }

    private void killGame() {
        if (stopped) {
            return;
        }
        stopped = true;

        long durationSec = (System.currentTimeMillis() - startTimeMs) / 1000;
        int turns = 0;
        try {
            turns = game.getTurnNum();
        } catch (Exception ignored) {
        }

        String logMsg = String.format("Game killed after %ds (%d turns): %s", durationSec, turns, killReason);
        logger.warn("GameHealthMonitor: " + logMsg);
        GAMES_KILLED.incrementAndGet();

        // Write to dedicated kills log file
        writeKillLog(durationSec, turns, killReason);

        try {
            game.end();
        } catch (Exception e) {
            logger.warn("Failed to end game gracefully, attempting force", e);
            try {
                // Force game over by setting all players to lost
                game.getState().getPlayers().values().forEach(p -> {
                    try {
                        if (!p.hasLost()) {
                            p.lost(game);
                        }
                    } catch (Exception ignored) {
                    }
                });
            } catch (Exception e2) {
                logger.error("Failed to force-kill game", e2);
            }
        }
    }

    private void writeKillLog(long durationSec, int turns, String reason) {
        try {
            Path logPath = Paths.get(RLLogPaths.GAME_KILLS_LOG_PATH);
            Files.createDirectories(logPath.getParent());

            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            String logLine = String.format("%s | duration=%ds | turns=%d | reason=%s\n",
                    timestamp, durationSec, turns, reason);

            Files.write(logPath, logLine.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write kill log: " + e.getMessage());
        }
    }

    private String normalizeMessage(String msg) {
        // Remove common variable parts: timestamps, UUIDs, object IDs, etc.
        return msg
                .replaceAll("\\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\b", "[UUID]")
                .replaceAll("\\b\\d{10,}\\b", "[ID]")
                .replaceAll("\\[\\w+\\d+\\]", "[REF]") // e.g., [dd1], [abc123]
                .replaceAll("\\d+\\.\\d+", "[NUM]")
                .trim();
    }

    private String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }

    /**
     * Tracks occurrences of a message over time.
     */
    private static class MessageTracker {

        private final AtomicLong lastResetMs = new AtomicLong(System.currentTimeMillis());
        private final AtomicInteger countInWindow = new AtomicInteger(0);

        void record() {
            long now = System.currentTimeMillis();
            long lastReset = lastResetMs.get();

            // If window has passed, reset counter
            if (now - lastReset > REPEAT_WINDOW_MS) {
                if (lastResetMs.compareAndSet(lastReset, now)) {
                    countInWindow.set(1);
                } else {
                    countInWindow.incrementAndGet();
                }
            } else {
                countInWindow.incrementAndGet();
            }
        }

        int getCountInWindow(int windowMs) {
            long now = System.currentTimeMillis();
            if (now - lastResetMs.get() > windowMs) {
                return 0;
            }
            return countInWindow.get();
        }
    }

    /**
     * Create a monitor and start it.
     */
    public static GameHealthMonitor createAndStart(Game game) {
        GameHealthMonitor monitor = new GameHealthMonitor(game);
        monitor.start();
        return monitor;
    }

    /**
     * Create a monitor with custom timeout and start it.
     */
    public static GameHealthMonitor createAndStart(Game game, int timeoutSec) {
        GameHealthMonitor monitor = new GameHealthMonitor(game, timeoutSec);
        monitor.start();
        return monitor;
    }
}
