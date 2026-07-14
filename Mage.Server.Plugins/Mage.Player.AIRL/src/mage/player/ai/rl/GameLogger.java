package mage.player.ai.rl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Logger for detailed game analysis. Writes each game to a separate file.
 * Captures game state, candidate actions, model scores, and decisions.
 */
public class GameLogger {

    private static final AtomicInteger gameCounter = new AtomicInteger(0);
    private static final String TRAIN_LOG_DIR = RLLogPaths.TRAINING_GAME_LOGS_DIR;
    private static final String EVAL_LOG_DIR = RLLogPaths.EVAL_GAME_LOGS_DIR;
    private static final String DRAFT_EVAL_LOG_DIR = RLLogPaths.DRAFT_EVAL_GAME_LOGS_DIR;
    private static final String DRAFT_BENCHMARK_LOG_DIR = RLLogPaths.DRAFT_BENCHMARK_GAME_LOGS_DIR;
    private static final int LINE_WIDTH = 80;
    private static final int COMPACT_STATE_ZONE_CHARS = envInt("GAME_LOG_COMPACT_ZONE_CHARS", 96);
    private static final int COMPACT_ACTION_CHARS = envInt("GAME_LOG_COMPACT_ACTION_CHARS", 180);
    private static final int COMPACT_TOP_OPTIONS = envInt("GAME_LOG_COMPACT_TOP_OPTIONS", 5);

    private final BufferedWriter writer;
    private final String gameId;
    private final Path logFile;
    private int decisionNumber = 0;
    private int recordIdCounter = 0;
    private final boolean enabled;
    private final LogFormat logFormat;

    private enum LogFormat {
        FULL,
        COMPACT,
        BOTH
    }

    private GameLogger(String gameId, Path logFile, BufferedWriter writer, boolean enabled, LogFormat logFormat) {
        this.gameId = gameId;
        this.logFile = logFile;
        this.writer = writer;
        this.enabled = enabled;
        this.logFormat = logFormat == null ? LogFormat.FULL : logFormat;
    }

    /**
     * Create a new game logger. Returns null if logging is disabled.
     */
    public static GameLogger create(boolean enableLogging) {
        if (!enableLogging) {
            return createNoOp();
        }
        return createInDir(resolveLogDir(), isTrainingMode() ? 50 : -1);
    }

    /**
     * Create a game logger that always writes to the evaluation log directory.
     * Used for league/ladder benchmark games; no cleanup cap (eval logs accumulate).
     */
    public static GameLogger createForEval(boolean enableLogging) {
        if (!enableLogging) {
            return createNoOp();
        }
        return createInDir(EVAL_LOG_DIR, -1);
    }

    /**
     * Create a game logger for DraftTrainer per-episode reward games.
     */
    public static GameLogger createForDraftEval(boolean enableLogging) {
        if (!enableLogging) {
            return createNoOp();
        }
        return createInDir(DRAFT_EVAL_LOG_DIR, -1);
    }

    /**
     * Create a game logger for DraftTrainer periodic benchmark games.
     */
    public static GameLogger createForDraftBenchmark(boolean enableLogging) {
        if (!enableLogging) {
            return createNoOp();
        }
        return createInDir(DRAFT_BENCHMARK_LOG_DIR, -1);
    }

    /**
     * Create a game logger writing to a specific profile directory with a max file cap.
     */
    public static GameLogger createInProfileDir(String dir, int maxFiles) {
        return createInDir(dir, maxFiles);
    }

    private static GameLogger createInDir(String dir, int maxFiles) {
        try {
            Path logDir = Paths.get(dir);
            Files.createDirectories(logDir);

            if (maxFiles > 0) {
                cleanupOldGameLogs(logDir, maxFiles);
            }

            int gameNum = gameCounter.incrementAndGet();
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String gameId = String.format("game_%s_%04d", timestamp, gameNum);

            Path logFile = logDir.resolve(gameId + ".txt");
            BufferedWriter writer = Files.newBufferedWriter(
                    logFile,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING
            );

            GameLogger logger = new GameLogger(gameId, logFile, writer, true, resolveLogFormat());
            logger.writeHeader();
            return logger;

        } catch (IOException e) {
            System.err.println("Failed to create game logger: " + e.getMessage());
            return createNoOp();
        }
    }

    private static String resolveLogDir() {
        // Explicit override wins.
        String env = System.getenv("GAME_LOG_DIR");
        if (env != null && !env.trim().isEmpty()) {
            return env.trim();
        }

        // Otherwise, default based on MODE (train vs eval/benchmark).
        String mode = System.getenv("MODE");
        if (mode != null) {
            String m = mode.trim().toLowerCase();
            if ("eval".equals(m) || "benchmark".equals(m)) {
                return EVAL_LOG_DIR;
            }
        }
        return TRAIN_LOG_DIR;
    }

    private static boolean isTrainingMode() {
        String mode = System.getenv("MODE");
        if (mode == null) {
            return true; // Default to training mode
        }
        String m = mode.trim().toLowerCase();
        return !("eval".equals(m) || "benchmark".equals(m));
    }

    private static LogFormat resolveLogFormat() {
        String raw = System.getenv("GAME_LOG_FORMAT");
        if (raw == null || raw.trim().isEmpty()) {
            raw = System.getenv("GAME_LOG_STYLE");
        }
        if ((raw == null || raw.trim().isEmpty()) && envBool("GAME_LOG_COMPACT", false)) {
            raw = "compact";
        }
        if (raw == null || raw.trim().isEmpty()) {
            return LogFormat.FULL;
        }
        String value = raw.trim().toLowerCase(Locale.ROOT);
        if ("compact".equals(value) || "short".equals(value) || "trace".equals(value)) {
            return LogFormat.COMPACT;
        }
        if ("both".equals(value) || "dual".equals(value)) {
            return LogFormat.BOTH;
        }
        return LogFormat.FULL;
    }

    private static boolean envBool(String key, boolean defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.trim().isEmpty()) {
            return defaultValue;
        }
        String value = raw.trim().toLowerCase(Locale.ROOT);
        return "1".equals(value) || "true".equals(value) || "yes".equals(value) || "on".equals(value);
    }

    private static int envInt(String key, int defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.trim().isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    /**
     * Clean up old game logs, keeping only the most recent N files.
     * Files are sorted by last modified time (oldest first).
     */
    private static void cleanupOldGameLogs(Path logDir, int maxFiles) {
        try {
            List<Path> logFiles = Files.list(logDir)
                    .filter(p -> p.toString().endsWith(".txt"))
                    .filter(p -> p.getFileName().toString().startsWith("game_"))
                    .sorted((a, b) -> {
                        try {
                            long timeA = Files.getLastModifiedTime(a).toMillis();
                            long timeB = Files.getLastModifiedTime(b).toMillis();
                            return Long.compare(timeA, timeB); // oldest first
                        } catch (IOException e) {
                            return 0;
                        }
                    })
                    .collect(java.util.stream.Collectors.toList());

            // Delete oldest files if we exceed the limit
            int toDelete = logFiles.size() - maxFiles + 1; // +1 for the file we're about to create
            if (toDelete > 0) {
                for (int i = 0; i < toDelete && i < logFiles.size(); i++) {
                    try {
                        Files.delete(logFiles.get(i));
                    } catch (IOException e) {
                        System.err.println("Failed to delete old game log: " + logFiles.get(i) + " - " + e.getMessage());
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Failed to cleanup old game logs: " + e.getMessage());
        }
    }

    private static String repeatChar(char c, int n) {
        if (n <= 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; i++) {
            sb.append(c);
        }
        return sb.toString();
    }

    /**
     * XMage turn numbers typically increment each player turn (P1=1, P2=2,
     * P1=3,...). For logs we want a per-player "round" turn (P1=1, P2=1, P1=2,
     * P2=2,...).
     */
    private static int toPerPlayerTurn(int globalTurn) {
        if (globalTurn <= 0) {
            return globalTurn;
        }
        return (globalTurn + 1) / 2;
    }

    /**
     * Create a no-op logger that does nothing.
     */
    private static GameLogger createNoOp() {
        return new GameLogger("noop", null, null, false, LogFormat.FULL);
    }

    private void writeHeader() {
        if (!enabled) {
            return;
        }
        try {
            writer.write(repeatChar('=', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format("GAME LOG: %s", gameId));
            writer.newLine();
            writer.write(String.format("Started: %s", LocalDateTime.now()));
            writer.newLine();
            writer.write(String.format("Format: %s", logFormat.toString().toLowerCase(Locale.ROOT)));
            writer.newLine();
            if (logFormat == LogFormat.COMPACT || logFormat == LogFormat.BOTH) {
                writer.write("Compact legend: D### T# actor @ phase | selected action, value, top policy alternatives, compact zones");
                writer.newLine();
            }
            writer.write(repeatChar('=', LINE_WIDTH));
            writer.newLine();
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error writing header: " + e.getMessage());
        }
    }

    /**
     * Log a decision point with game state, options, and scores.
     */
    public void logDecision(
            String playerName,
            String activePlayerName,
            String phase,
            int turn,
            String gameState,
            List<String> options,
            float[] actionProbs,
            float valueScore,
            String selectedAction
    ) {
        int idx = -1;
        if (selectedAction != null && options != null) {
            for (int i = 0; i < options.size(); i++) {
                if (selectedAction.equals(options.get(i))) {
                    idx = i;
                    break;
                }
            }
        }
        logDecision(playerName, activePlayerName, phase, turn, gameState, options, actionProbs, valueScore, idx, selectedAction);
    }

    /**
     * Log a decision point with an explicit selected index (preferred).
     */
    public void logDecision(
            String playerName,
            String activePlayerName,
            String phase,
            int turn,
            String gameState,
            List<String> options,
            float[] actionProbs,
            float valueScore,
            int selectedIndex,
            String selectedAction
    ) {
        if (!enabled) {
            return;
        }

        try {
            int currentDecision = ++decisionNumber;
            String turnOwner = (activePlayerName == null || activePlayerName.trim().isEmpty())
                    ? "Unknown"
                    : activePlayerName.trim();
            String selectedLine = selectedAction;
            if (selectedIndex >= 0 && options != null && selectedIndex < options.size()) {
                selectedLine = options.get(selectedIndex);
            }

            if (logFormat == LogFormat.COMPACT || logFormat == LogFormat.BOTH) {
                writeCompactDecision(currentDecision, playerName, turnOwner, phase, turn, gameState,
                        options, actionProbs, valueScore, selectedIndex, selectedLine);
            }
            if (logFormat == LogFormat.FULL || logFormat == LogFormat.BOTH) {
                if (logFormat == LogFormat.BOTH) {
                    writer.newLine();
                }
                writeFullDecision(currentDecision, playerName, turnOwner, phase, turn, gameState,
                        options, actionProbs, valueScore, selectedIndex, selectedLine);
            }

            writer.flush();

        } catch (IOException e) {
            System.err.println("Error logging decision: " + e.getMessage());
        }
    }

    private void writeFullDecision(
            int currentDecision,
            String playerName,
            String turnOwner,
            String phase,
            int turn,
            String gameState,
            List<String> options,
            float[] actionProbs,
            float valueScore,
            int selectedIndex,
            String selectedLine
    ) throws IOException {
        writer.write(repeatChar('-', LINE_WIDTH));
        writer.newLine();
        writer.write(String.format("DECISION #%d - Turn %d (%s turn), %s - %s",
                currentDecision, toPerPlayerTurn(turn), turnOwner, phase, playerName));
        writer.newLine();
        writer.write(repeatChar('-', LINE_WIDTH));
        writer.newLine();
        writer.newLine();

        writer.write("GAME STATE:");
        writer.newLine();
        writer.write(gameState == null ? "" : gameState);
        writer.newLine();
        writer.newLine();

        writer.write("OPTIONS & SCORES:");
        writer.newLine();
        int optionCount = options == null ? 0 : options.size();
        int probCount = actionProbs == null ? 0 : actionProbs.length;
        for (int i = 0; i < optionCount && i < probCount; i++) {
            String marker = (i == selectedIndex) ? " >>> " : "     ";
            writer.write(String.format(Locale.US, "%s[%d] %.6f - %s",
                    marker, i, actionProbs[i], options.get(i)));
            writer.newLine();
        }
        writer.newLine();

        writer.write(String.format(Locale.US, "VALUE SCORE: %.6f", valueScore));
        writer.newLine();
        writer.write(String.format("SELECTED: %s", selectedLine));
        writer.newLine();
        writer.newLine();
    }

    private void writeCompactDecision(
            int currentDecision,
            String playerName,
            String turnOwner,
            String phase,
            int turn,
            String gameState,
            List<String> options,
            float[] actionProbs,
            float valueScore,
            int selectedIndex,
            String selectedLine
    ) throws IOException {
        writer.write(String.format("DECISION #%d - Turn %d (%s turn), %s - %s",
                currentDecision, toPerPlayerTurn(turn), turnOwner, phase, playerName));
        writer.newLine();
        writer.write(String.format(Locale.US, "  SELECTED[%d] p=%s value=%.6f: %s",
                selectedIndex,
                formatProbability(actionProbs, selectedIndex),
                valueScore,
                trunc(oneLine(selectedLine), COMPACT_ACTION_CHARS)));
        writer.newLine();
        String state = compactGameState(gameState);
        if (!state.isEmpty()) {
            writer.write("  STATE: " + state);
            writer.newLine();
        }
        String topOptions = compactTopOptions(options, actionProbs, selectedIndex);
        if (!topOptions.isEmpty()) {
            writer.write("  TOP: " + topOptions);
            writer.newLine();
        }
        String exploration = compactExploration(gameState);
        if (!exploration.isEmpty()) {
            writer.write("  " + exploration);
            writer.newLine();
        }
        // Keep these legacy markers so value/selection tools can still read compact logs.
        writer.write(String.format(Locale.US, "VALUE SCORE: %.6f", valueScore));
        writer.newLine();
        writer.write(String.format("SELECTED: %s", selectedLine));
        writer.newLine();
        writer.newLine();
    }

    /**
     * Log game outcome.
     */
    public void logOutcome(String winner, String loser, int turns, String reason) {
        if (!enabled) {
            return;
        }

        try {
            writer.write(repeatChar('=', LINE_WIDTH));
            writer.newLine();
            writer.write("GAME OUTCOME");
            writer.newLine();
            writer.write(repeatChar('=', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format("Winner: %s", winner));
            writer.newLine();
            writer.write(String.format("Loser: %s", loser));
            writer.newLine();
            writer.write(String.format("Turns: %d", toPerPlayerTurn(turns)));
            writer.newLine();
            writer.write(String.format("Reason: %s", reason));
            writer.newLine();
            writer.write(String.format("Total Decisions Logged: %d", decisionNumber));
            writer.newLine();
            writer.write(String.format("Ended: %s", LocalDateTime.now()));
            writer.newLine();
            writer.write(repeatChar('=', LINE_WIDTH));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging outcome: " + e.getMessage());
        }
    }

    /**
     * Log an action taken by any player (e.g., opponent actions). Simpler than
     * logDecision - just records what happened.
     */
    public void logAction(String playerName, String phase, int turn, String action) {
        if (!enabled) {
            return;
        }

        try {
            if (logFormat == LogFormat.COMPACT || logFormat == LogFormat.BOTH) {
                writer.write(String.format("ACTION T%d %s %s: %s",
                        toPerPlayerTurn(turn), phase, playerName, trunc(oneLine(action), COMPACT_ACTION_CHARS)));
            } else {
                writer.write(String.format("[Turn %d, %s] %s: %s",
                        toPerPlayerTurn(turn), phase, playerName, action));
            }
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging action: " + e.getMessage());
        }
    }

    /**
     * Log turn start with full game state summary (shows what happened since
     * last turn).
     */
    public void logTurnStart(int turn, String activePlayer, String fullGameState) {
        if (!enabled) {
            return;
        }

        try {
            writer.newLine();
            if (logFormat == LogFormat.COMPACT || logFormat == LogFormat.BOTH) {
                writer.write(String.format("TURN %d START active=%s | %s",
                        toPerPlayerTurn(turn), activePlayer, compactGameState(fullGameState)));
                writer.newLine();
                if (logFormat == LogFormat.COMPACT) {
                    writer.flush();
                    return;
                }
            }
            writer.write(repeatChar('*', LINE_WIDTH));
            writer.newLine();
            writer.write(String.format("*** TURN %d START - Active Player: %s ***", toPerPlayerTurn(turn), activePlayer));
            writer.newLine();
            writer.write(repeatChar('*', LINE_WIDTH));
            writer.newLine();
            writer.write(fullGameState);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging turn start: " + e.getMessage());
        }
    }

    /**
     * Log arbitrary text.
     */
    public void log(String message) {
        if (!enabled) {
            return;
        }

        try {
            writer.write(message);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging message: " + e.getMessage());
        }
    }

    /**
     * Close the logger.
     */
    public void close() {
        if (!enabled || writer == null) {
            return;
        }

        try {
            writer.close();
        } catch (IOException e) {
            System.err.println("Error closing game logger: " + e.getMessage());
        }
    }

    public boolean isEnabled() {
        return enabled;
    }

    public int getNextDecisionNumber() {
        return decisionNumber + 1;
    }

    /**
     * v5 schema (local-training/kernel_oracle/v5_capture_schema_addendum.md,
     * Sol #95/#96/#98): a deterministic per-game record id, unique across ALL
     * decision types, assigned once per ComputerPlayerRL#logReplayDecision call
     * for BOTH players sharing this GameLogger instance. Unlike
     * getNextDecisionNumber() (which only peeks at decisionNumber + 1 without
     * consuming it, so two decisions landing between text-log increments can
     * read the identical value -- confirmed colliding DECLARE_ATTACKS /
     * ACTIVATE_ABILITY_OR_SPELL pairs in a 40-game corpus), this genuinely
     * increments on every call and is safe to use as a join/ordering key
     * across REPLAY_DECISION_JSON records and LiveCheckpointRecorder's
     * manifest rows. decision_number remains available only as diagnostic
     * metadata.
     */
    public int nextRecordId() {
        return ++recordIdCounter;
    }

    /**
     * Read-only counterpart to {@link #nextRecordId()} (Sol #101, BranchOracle
     * from-start combat protocol): returns the record_id the NEXT
     * logReplayDecision call will assign, WITHOUT consuming it. Needed because
     * EngineDecisionBranchController#onDecision fires before the corresponding
     * logReplayDecision call for the same decision (forcedChoiceIndices() is
     * invoked, then the caller logs the outcome) -- a branch controller that
     * needs to recognize "this decision is the one I'm targeting" by record_id
     * cannot call the mutating nextRecordId() itself without double-consuming
     * the id and desynchronizing every later record_id in the trace.
     */
    public int peekNextRecordId() {
        return recordIdCounter + 1;
    }

    public String getGameId() {
        return gameId;
    }

    public Path getLogFile() {
        return logFile;
    }

    private static String formatProbability(float[] probabilities, int index) {
        if (probabilities == null || index < 0 || index >= probabilities.length) {
            return "n/a";
        }
        return String.format(Locale.US, "%.4f", probabilities[index]);
    }

    private static String compactTopOptions(List<String> options, float[] actionProbs, int selectedIndex) {
        if (options == null || options.isEmpty() || actionProbs == null || actionProbs.length == 0) {
            return "";
        }
        int count = Math.min(options.size(), actionProbs.length);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            indices.add(i);
        }
        Collections.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return Float.compare(actionProbs[b], actionProbs[a]);
            }
        });
        List<Integer> chosen = new ArrayList<>();
        if (selectedIndex >= 0 && selectedIndex < count) {
            chosen.add(selectedIndex);
        }
        int limit = Math.max(1, COMPACT_TOP_OPTIONS);
        for (Integer idx : indices) {
            if (chosen.size() >= limit) {
                break;
            }
            if (!chosen.contains(idx)) {
                chosen.add(idx);
            }
        }
        Collections.sort(chosen);
        StringBuilder sb = new StringBuilder();
        sb.append("n=").append(count);
        for (Integer idx : chosen) {
            sb.append(" | ");
            if (idx == selectedIndex) {
                sb.append("*");
            }
            sb.append("[").append(idx).append("] ");
            sb.append(String.format(Locale.US, "%.4f ", actionProbs[idx]));
            sb.append(trunc(oneLine(options.get(idx)), COMPACT_ACTION_CHARS));
        }
        return sb.toString();
    }

    private static String compactExploration(String gameState) {
        if (gameState == null || gameState.isEmpty()) {
            return "";
        }
        String[] lines = gameState.split("\\r?\\n");
        for (String raw : lines) {
            String line = raw.trim();
            if (line.startsWith("[Exploration]")) {
                return trunc(oneLine(line), 220);
            }
        }
        return "";
    }

    private static String compactGameState(String gameState) {
        if (gameState == null || gameState.trim().isEmpty()) {
            return "";
        }
        String[] lines = gameState.split("\\r?\\n");
        List<String> parts = new ArrayList<>();
        PlayerSummary current = null;
        String stackText = "";

        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("[Exploration]")) {
                continue;
            }
            if (line.startsWith("[Stack]:")) {
                stackText = "stack=" + line.substring("[Stack]:".length()).trim();
                continue;
            }
            if (line.startsWith("- [") && stackText.length() > 0 && !stackText.contains(" top=")) {
                stackText = stackText + " top=" + trunc(oneLine(line.replaceFirst("^-\\s*", "")), 90);
                continue;
            }
            PlayerSummary parsed = tryParsePlayer(line);
            if (parsed != null) {
                if (current != null) {
                    parts.add(current.format());
                }
                current = parsed;
                continue;
            }
            if (current != null && line.startsWith("-> ")) {
                applyZoneLine(current, line);
            }
        }
        if (current != null) {
            parts.add(current.format());
        }
        if (!stackText.isEmpty()) {
            parts.add(0, stackText);
        }
        return join(parts, " || ");
    }

    private static PlayerSummary tryParsePlayer(String line) {
        if (!line.startsWith("[") || !line.contains("], life = ")) {
            return null;
        }
        int end = line.indexOf(']');
        if (end <= 1) {
            return null;
        }
        String name = line.substring(1, end);
        String life = line.substring(line.indexOf("], life = ") + "], life = ".length()).trim();
        return new PlayerSummary(name, life);
    }

    private static void applyZoneLine(PlayerSummary summary, String line) {
        int colon = line.indexOf(':');
        if (colon < 0) {
            return;
        }
        String zone = line.substring(3, colon).trim();
        String raw = line.substring(colon + 1).trim();
        if ("Hand".equals(zone)) {
            summary.hand = compactZone("H", raw, true);
        } else if ("Permanents".equals(zone)) {
            summary.permanents = compactZone("B", raw, false);
        } else if ("Graveyard".equals(zone)) {
            summary.graveyard = compactZone("G", raw, false);
        } else if ("Exile".equals(zone)) {
            summary.exile = compactZone("X", raw, false);
        }
    }

    private static String compactZone(String label, String raw, boolean hand) {
        String value = stripOuterBrackets(raw);
        int hidden = parseHiddenCount(value);
        if (hidden >= 0) {
            return label + hidden;
        }
        if (value.trim().isEmpty()) {
            return label + "0";
        }
        int count = splitCards(value).size();
        String shown = trunc(value, COMPACT_STATE_ZONE_CHARS);
        if (hand) {
            return label + count + "[" + shown + "]";
        }
        return label + count + "[" + shown + "]";
    }

    private static int parseHiddenCount(String value) {
        String trimmed = value == null ? "" : value.trim();
        if (!trimmed.endsWith(" cards")) {
            return -1;
        }
        String number = trimmed.substring(0, trimmed.length() - " cards".length()).trim();
        try {
            return Integer.parseInt(number);
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private static String stripOuterBrackets(String raw) {
        String trimmed = raw == null ? "" : raw.trim();
        if (trimmed.startsWith("[") && trimmed.endsWith("]") && trimmed.length() >= 2) {
            return trimmed.substring(1, trimmed.length() - 1).trim();
        }
        return trimmed;
    }

    private static List<String> splitCards(String value) {
        List<String> out = new ArrayList<>();
        if (value == null || value.trim().isEmpty()) {
            return out;
        }
        String[] parts = value.split(";");
        for (String part : parts) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                out.add(trimmed);
            }
        }
        return out;
    }

    private static String oneLine(String text) {
        if (text == null) {
            return "";
        }
        return text.replace('\r', ' ').replace('\n', ' ').replaceAll("\\s+", " ").trim();
    }

    private static String trunc(String text, int maxChars) {
        if (text == null) {
            return "";
        }
        if (maxChars <= 0 || text.length() <= maxChars) {
            return text;
        }
        if (maxChars <= 3) {
            return text.substring(0, maxChars);
        }
        return text.substring(0, maxChars - 3) + "...";
    }

    private static String join(List<String> parts, String delimiter) {
        StringBuilder sb = new StringBuilder();
        for (String part : parts) {
            if (part == null || part.isEmpty()) {
                continue;
            }
            if (sb.length() > 0) {
                sb.append(delimiter);
            }
            sb.append(part);
        }
        return sb.toString();
    }

    private static final class PlayerSummary {
        private final String name;
        private final String life;
        private String hand = "H?";
        private String permanents = "B?";
        private String graveyard = "G?";
        private String exile = "X?";

        private PlayerSummary(String name, String life) {
            this.name = name;
            this.life = life;
        }

        private String format() {
            return name + " L" + life + " " + hand + " " + permanents + " " + graveyard + " " + exile;
        }
    }
}
