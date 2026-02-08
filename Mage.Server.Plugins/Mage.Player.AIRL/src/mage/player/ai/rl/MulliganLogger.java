package mage.player.ai.rl;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Thread-safe logger for mulligan decisions. Logs to CSV for later analysis and
 * training a better mulligan model.
 */
public class MulliganLogger {

    private static final Logger logger = LoggerFactory.getLogger(MulliganLogger.class);

    private static final String DEFAULT_LOG_PATH = RLLogPaths.MULLIGAN_STATS_PATH;
    private static final ReentrantLock fileLock = new ReentrantLock();

    private final String logPath;
    private boolean headerWritten = false;

    public MulliganLogger() {
        this(DEFAULT_LOG_PATH);
    }

    public MulliganLogger(String logPath) {
        this.logPath = logPath;
        ensureDirectoryExists();
        ensureHeaderExists();
    }

    private void ensureDirectoryExists() {
        try {
            Path path = Paths.get(logPath);
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
        } catch (IOException e) {
            logger.error("Failed to create mulligan log directory", e);
        }
    }

    private void ensureHeaderExists() {
        fileLock.lock();
        try {
            Path path = Paths.get(logPath);
            if (!Files.exists(path)) {
                // Write header
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(logPath, true))) {
                    writer.write("timestamp,episode,player,mulligan_num,hand_size,decision,q_keep,q_mull,q_gap,cards,cards_kept,cards_bottomed\n");
                    headerWritten = true;
                } catch (IOException e) {
                    logger.error("Failed to write mulligan log header", e);
                }
            } else {
                // If an old-format file exists, rotate it once and start a fresh CSV with the new header.
                try (java.io.BufferedReader reader = Files.newBufferedReader(path)) {
                    String first = reader.readLine();
                    if (first != null && !first.contains("q_keep")) {
                        String legacyPath = logPath.replaceAll("\\.csv$", "")
                                + ".legacy." + System.currentTimeMillis() + ".csv";
                        Files.move(path, Paths.get(legacyPath));
                        try (BufferedWriter writer = new BufferedWriter(new FileWriter(logPath, true))) {
                            writer.write("timestamp,episode,player,mulligan_num,hand_size,decision,q_keep,q_mull,q_gap,cards,cards_kept,cards_bottomed\n");
                        }
                    }
                } catch (IOException e) {
                    logger.error("Failed to validate/rotate mulligan log header", e);
                }
                headerWritten = true;
            }
        } finally {
            fileLock.unlock();
        }
    }

    /**
     * Log a mulligan decision to CSV.
     *
     * @param episodeNum Current episode number
     * @param playerName Player making the decision
     * @param mulliganNum Which mulligan (0 = opening hand, 1 = first mulligan,
     * etc.)
     * @param handSize Current hand size
     * @param decision "KEEP" or "MULLIGAN"
     * @param qKeep Model score for KEEP (Q_keep)
     * @param qMull Model score for MULL (Q_mull)
     * @param cards List of card names in hand (semicolon-separated)
     * @param cardsKept Cards kept after London mulligan (semicolon-separated,
     * empty if full mulligan)
     * @param cardsBottomed Cards put on bottom after London mulligan
     * (semicolon-separated, empty if full mulligan)
     */
    public void logDecision(
            int episodeNum,
            String playerName,
            int mulliganNum,
            int handSize,
            String decision,
            float qKeep,
            float qMull,
            String cards,
            String cardsKept,
            String cardsBottomed) {

        fileLock.lock();
        try {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(logPath, true))) {
                long timestamp = System.currentTimeMillis();
                // Escape card lists for CSV (wrap in quotes and escape internal quotes)
                String escapedCards = "\"" + cards.replace("\"", "\"\"") + "\"";
                String escapedKept = "\"" + cardsKept.replace("\"", "\"\"") + "\"";
                String escapedBottomed = "\"" + cardsBottomed.replace("\"", "\"\"") + "\"";

                float qGap = qKeep - qMull;
                String line = String.format("%d,%d,%s,%d,%d,%s,%.6f,%.6f,%.6f,%s,%s,%s\n",
                        timestamp,
                        episodeNum,
                        playerName,
                        mulliganNum,
                        handSize,
                        decision,
                        qKeep,
                        qMull,
                        qGap,
                        escapedCards,
                        escapedKept,
                        escapedBottomed
                );
                writer.write(line);
            }
        } catch (IOException e) {
            logger.error("Failed to write mulligan log entry", e);
        } finally {
            fileLock.unlock();
        }
    }

    /**
     * Get singleton instance (lazy-initialized).
     */
    private static volatile MulliganLogger instance;
    private static final Object instanceLock = new Object();

    public static MulliganLogger getInstance() {
        if (instance == null) {
            synchronized (instanceLock) {
                if (instance == null) {
                    instance = new MulliganLogger();
                }
            }
        }
        return instance;
    }
}
