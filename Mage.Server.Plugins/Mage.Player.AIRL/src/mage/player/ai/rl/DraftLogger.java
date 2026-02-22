package mage.player.ai.rl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import mage.cards.Card;

/**
 * Logger for draft decisions. Mirrors GameLogger for draft-specific output.
 *
 * Writes per-draft files to DRAFT_LOGS_DIR.
 * Appends aggregate metrics to draft_stats.csv.
 */
public class DraftLogger {

    private static final AtomicInteger draftCounter = new AtomicInteger(0);
    private static final int LINE_WIDTH = 80;
    private static final int MAX_DRAFT_LOG_FILES = 50;

    private final BufferedWriter writer;
    private final String draftId;
    private final boolean enabled;
    private int pickNumber = 0;

    private DraftLogger(String draftId, BufferedWriter writer, boolean enabled) {
        this.draftId = draftId;
        this.writer = writer;
        this.enabled = enabled;
    }

    /** Factory: create a logger if enabled. */
    public static DraftLogger create(boolean enableLogging) {
        if (!enableLogging) {
            return noOp();
        }
        try {
            String dir = RLLogPaths.DRAFT_GAME_LOGS_DIR;
            Path logDir = Paths.get(dir);
            Files.createDirectories(logDir);
            cleanupOldLogs(logDir);

            int num = draftCounter.incrementAndGet();
            String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String draftId = String.format("draft_%s_%04d", ts, num);
            Path logFile = logDir.resolve(draftId + ".txt");

            BufferedWriter w = Files.newBufferedWriter(logFile,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            DraftLogger dl = new DraftLogger(draftId, w, true);
            dl.writeHeader();
            return dl;
        } catch (IOException e) {
            System.err.println("DraftLogger: failed to create: " + e.getMessage());
            return noOp();
        }
    }

    private static DraftLogger noOp() {
        return new DraftLogger("noop", null, false);
    }

    private void writeHeader() throws IOException {
        writer.write(repeat('=', LINE_WIDTH));
        writer.newLine();
        writer.write("DRAFT LOG: " + draftId);
        writer.newLine();
        writer.write("Started: " + LocalDateTime.now());
        writer.newLine();
        writer.write(repeat('=', LINE_WIDTH));
        writer.newLine();
        writer.newLine();
        writer.flush();
    }

    /**
     * Log a single pick decision.
     *
     * @param packNum    1-3
     * @param pickNum    1-15
     * @param packCards  all cards in the pack
     * @param scores     model scores (softmax probabilities), null if random/heuristic
     * @param chosenIdx  index into packCards of chosen card
     * @param valueEst   value head prediction (expected winrate)
     * @param poolSize   cards in pool before this pick
     */
    public void logPick(int packNum, int pickNum, List<Card> packCards,
            float[] scores, int chosenIdx, float valueEst, int poolSize) {
        if (!enabled) return;
        try {
            pickNumber++;
            writer.write(repeat('-', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format("PICK #%d - Pack %d, Pick %d  [Pool: %d cards]  [Value: %.4f]",
                    pickNumber, packNum, pickNum, poolSize, valueEst));
            writer.newLine();
            writer.write(repeat('-', LINE_WIDTH));
            writer.newLine();

            for (int i = 0; i < packCards.size(); i++) {
                String marker = (i == chosenIdx) ? " >>> " : "     ";
                String score = (scores != null && i < scores.length)
                        ? String.format("%.4f", scores[i]) : "  N/A";
                writer.write(String.format("%s[%2d] %s  %s",
                        marker, i, score, packCards.get(i).getName()));
                writer.newLine();
            }
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("DraftLogger.logPick error: " + e.getMessage());
        }
    }

    public void logConstruction(List<Card> poolCards,
            List<Card> selectedCards, List<Card> landBase,
            int rawLandCount, int floorLandCount, int appliedLandCount,
            int targetNonLandCount, boolean forcedByFloor,
            int finalNonLands, int finalLands, int finalDeckSize,
            int draftedNonBasicsUsed, int draftedBasicsUsed, int syntheticBasicsAdded,
            List<String> spellPickSteps, List<String> landPickSteps,
            List<String> repairActions, String landSourceSummary) {
        if (!enabled) return;
        try {
            writer.write(repeat('=', LINE_WIDTH));
            writer.newLine();
            writer.write("DECK CONSTRUCTION");
            writer.newLine();
            writer.write(repeat('=', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format(
                    "land_count raw=%d floor=%d applied=%d forced_by_floor=%s target_nonlands=%d",
                    rawLandCount, floorLandCount, appliedLandCount, forcedByFloor, targetNonLandCount));
            writer.newLine();
            writer.write(String.format(
                    "final_composition nonlands=%d lands=%d deck_size=%d",
                    finalNonLands, finalLands, finalDeckSize));
            writer.newLine();
            writer.write(String.format(
                    "land_sources drafted_nonbasics=%d drafted_basics=%d synthetic_basics=%d",
                    draftedNonBasicsUsed, draftedBasicsUsed, syntheticBasicsAdded));
            writer.newLine();
            writer.write("land_color_sources " + (landSourceSummary == null ? "" : landSourceSummary));
            writer.newLine();
            writer.newLine();

            writer.write("Pool (" + poolCards.size() + " cards):");
            writer.newLine();
            for (int i = 0; i < poolCards.size(); i++) {
                Card c = poolCards.get(i);
                boolean selected = selectedCards.contains(c);
                String sel = selected ? " [IN]" : "     ";
                writer.write(String.format("  %s  %s", sel, c.getName()));
                writer.newLine();
            }
            writer.newLine();

            writer.write("Maindeck non-lands (" + selectedCards.size() + "):");
            writer.newLine();
            for (Card c : selectedCards) {
                writer.write("  " + c.getName());
                writer.newLine();
            }
            writer.newLine();

            writer.write("Spell pick steps (" + (spellPickSteps == null ? 0 : spellPickSteps.size()) + "):");
            writer.newLine();
            if (spellPickSteps != null) {
                for (String line : spellPickSteps) {
                    writer.write("  " + line);
                    writer.newLine();
                }
            }
            writer.newLine();

            writer.write("Land base (" + landBase.size() + "):");
            writer.newLine();
            // Aggregate land names
            java.util.Map<String, Long> landCounts = landBase.stream()
                    .collect(Collectors.groupingBy(Card::getName, Collectors.counting()));
            landCounts.forEach((name, count) -> {
                try {
                    writer.write("  " + count + "x " + name);
                    writer.newLine();
                } catch (IOException ignored) {}
            });
            writer.newLine();

            writer.write("Land pick steps (" + (landPickSteps == null ? 0 : landPickSteps.size()) + "):");
            writer.newLine();
            if (landPickSteps != null) {
                for (String line : landPickSteps) {
                    writer.write("  " + line);
                    writer.newLine();
                }
            }
            writer.newLine();

            writer.write("Repairs (" + (repairActions == null ? 0 : repairActions.size()) + "):");
            writer.newLine();
            if (repairActions != null && !repairActions.isEmpty()) {
                for (String line : repairActions) {
                    writer.write("  " + line);
                    writer.newLine();
                }
            } else {
                writer.write("  none");
                writer.newLine();
            }
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("DraftLogger.logConstruction error: " + e.getMessage());
        }
    }

    /**
     * Log game results (reward) after evaluation games.
     */
    public void logEpisodeResults(int wins, int losses, float reward,
            List<String> pickAdvantages) {
        if (!enabled) return;
        try {
            writer.write(repeat('=', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format("EPISODE RESULTS: %d W / %d L  Reward=%.4f",
                    wins, losses, reward));
            writer.newLine();
            writer.write(repeat('=', LINE_WIDTH));
            writer.newLine();
            if (pickAdvantages != null && !pickAdvantages.isEmpty()) {
                writer.write("Top pick advantages:");
                writer.newLine();
                for (String line : pickAdvantages) {
                    writer.write("  " + line);
                    writer.newLine();
                }
            }
            writer.write("Ended: " + LocalDateTime.now());
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("DraftLogger.logEpisodeResults error: " + e.getMessage());
        }
    }

    public void log(String msg) {
        if (!enabled) return;
        try {
            writer.write(msg);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("DraftLogger.log error: " + e.getMessage());
        }
    }

    public void close() {
        if (!enabled || writer == null) return;
        try {
            writer.close();
        } catch (IOException ignored) {}
    }

    public boolean isEnabled() { return enabled; }
    public String getDraftId() { return draftId; }

    // -----------------------------------------------------------------------
    // Aggregate CSV stats
    // -----------------------------------------------------------------------

    /**
     * Append a row to draft_stats.csv with episode-level aggregate metrics.
     */
    public static void appendStats(int episode, double winRate, double avgPickEntropy,
            float policyLoss, float valueLoss, float entropyLoss,
            int wCount, int uCount, int bCount, int rCount, int gCount) {
        try {
            Path statsPath = Paths.get(RLLogPaths.DRAFT_STATS_PATH);
            Files.createDirectories(statsPath.getParent());

            boolean exists = Files.exists(statsPath);
            try (BufferedWriter w = Files.newBufferedWriter(statsPath,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                if (!exists) {
                    w.write("episode,timestamp,winrate,avg_pick_entropy,"
                            + "policy_loss,value_loss,entropy_loss,"
                            + "w_drafts,u_drafts,b_drafts,r_drafts,g_drafts");
                    w.newLine();
                }
                String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                w.write(String.format("%d,%s,%.4f,%.4f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d",
                        episode, ts, winRate, avgPickEntropy,
                        policyLoss, valueLoss, entropyLoss,
                        wCount, uCount, bCount, rCount, gCount));
                w.newLine();
            }
        } catch (IOException e) {
            System.err.println("DraftLogger.appendStats error: " + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static String repeat(char c, int n) {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; i++) sb.append(c);
        return sb.toString();
    }

    private static void cleanupOldLogs(Path logDir) {
        try {
            List<Path> files = Files.list(logDir)
                    .filter(p -> p.getFileName().toString().startsWith("draft_")
                            && p.getFileName().toString().endsWith(".txt"))
                    .sorted((a, b) -> {
                        try {
                            return Long.compare(
                                    Files.getLastModifiedTime(a).toMillis(),
                                    Files.getLastModifiedTime(b).toMillis());
                        } catch (IOException e) { return 0; }
                    })
                    .collect(Collectors.toList());

            int toDelete = files.size() - MAX_DRAFT_LOG_FILES + 1;
            for (int i = 0; i < toDelete && i < files.size(); i++) {
                try { Files.delete(files.get(i)); } catch (IOException ignored) {}
            }
        } catch (IOException e) {
            System.err.println("DraftLogger cleanup error: " + e.getMessage());
        }
    }
}
