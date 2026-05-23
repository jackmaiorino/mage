package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import mage.player.ai.ComputerPlayerRL;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * Validates and branches durable live-eval checkpoints without reconstructing
 * the game from the original replay prefix.
 */
public final class LiveCheckpointBranchMiner {

    private static final String CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,selected_indices,selected_texts,"
                    + "candidate_hash,state_hash,rng_state_hash,source_reentry_a_matched,source_reentry_b_matched,"
                    + "classification,source_terminal,source_won,source_lost,source_error,alternate_indices,"
                    + "alternate_texts,alternate_terminal,alternate_won,alternate_lost,alternate_error,"
                    + "reentry_a_candidate_hash,reentry_b_candidate_hash,reentry_a_state_hash,reentry_b_state_hash,"
                    + "reentry_a_reason,reentry_b_reason\n";

    private LiveCheckpointBranchMiner() {
    }

    public static void main(String[] args) throws Exception {
        Config cfg = Config.parse(args);
        Files.createDirectories(cfg.outDir);
        Path csvPath = cfg.outDir.resolve("live_checkpoint_branch_probe.csv");
        Files.write(csvPath, CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        List<Path> snapshots = discoverSnapshots(cfg);
        int processed = 0;
        Counts counts = new Counts();
        for (Path snapshotPath : snapshots) {
            if (cfg.maxSnapshots > 0 && processed >= cfg.maxSnapshots) {
                break;
            }
            LiveCheckpointRecorder.Snapshot snapshot;
            try {
                snapshot = loadSnapshot(snapshotPath);
            } catch (Throwable t) {
                BranchRow row = BranchRow.loadError(snapshotPath, errorSummary(t));
                appendRow(csvPath, row);
                counts.add(row.classification);
                processed++;
                continue;
            }
            if (!cfg.actionTypes.isEmpty() && !cfg.actionTypes.contains(snapshot.actionType)) {
                continue;
            }
            BranchRow row = probeSnapshot(snapshotPath, snapshot, cfg);
            appendRow(csvPath, row);
            counts.add(row.classification);
            processed++;
        }
        writeReadme(cfg, csvPath, processed, snapshots.size(), counts);
        System.out.println("live checkpoint branch miner wrote " + processed + " row(s) to " + csvPath);
        System.out.println("classification counts: " + counts.values);
    }

    private static BranchRow probeSnapshot(Path snapshotPath, LiveCheckpointRecorder.Snapshot snapshot, Config cfg) {
        BranchRow row = BranchRow.fromSnapshot(snapshotPath, snapshot);
        if (snapshot == null || snapshot.gameSnapshot == null) {
            row.classification = "snapshot_missing_game";
            return row;
        }
        if (snapshot.candidateTexts == null || snapshot.candidateTexts.size() < 2) {
            row.classification = "snapshot_too_few_candidates";
            return row;
        }
        List<Integer> sourceIndices = sanitizeIndices(snapshot.selectedIndices, snapshot.candidateTexts.size());
        if (sourceIndices.isEmpty()) {
            row.classification = "snapshot_missing_source_choice";
            return row;
        }

        BranchOutcome reentryA = runProbe(snapshot, sourceIndices, true, "source_reentry_a", cfg.timeoutSec);
        BranchOutcome reentryB = runProbe(snapshot, sourceIndices, true, "source_reentry_b", cfg.timeoutSec);
        row.applyReentry(reentryA, reentryB);
        if (!reentryA.reentryMatched || !reentryB.reentryMatched) {
            row.classification = "checkpoint_reentry_mismatch";
            return row;
        }
        if (cfg.reentryOnly) {
            row.classification = "reentry_matched";
            return row;
        }

        BranchOutcome source = runProbe(snapshot, sourceIndices, false, "source_terminal", cfg.timeoutSec);
        row.applySource(source);
        if (!source.error.isEmpty()) {
            row.classification = "source_error";
            return row;
        }
        if (!source.terminal) {
            row.classification = "source_not_terminal";
            return row;
        }
        if (source.won) {
            row.classification = "source_terminal_not_loss";
            return row;
        }

        List<Integer> alternateIndices = firstAlternateIndices(snapshot, sourceIndices);
        row.alternateIndices = joinInts(alternateIndices);
        row.alternateTexts = joinStrings(selectedTexts(snapshot.candidateTexts, alternateIndices));
        if (alternateIndices.isEmpty()) {
            row.classification = "alternate_unavailable";
            return row;
        }
        BranchOutcome alternate = runProbe(snapshot, alternateIndices, false, "alternate_terminal", cfg.timeoutSec);
        row.applyAlternate(alternate);
        if (!alternate.error.isEmpty()) {
            row.classification = "alternate_error";
        } else if (!alternate.terminal) {
            row.classification = "alternate_not_terminal";
        } else if (alternate.won) {
            row.classification = "clean_positive";
        } else if (alternate.lost) {
            row.classification = "clean_negative";
        } else {
            row.classification = "alternate_terminal_draw";
        }
        return row;
    }

    private static BranchOutcome runProbe(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> forcedIndices,
            boolean stopAtReentry,
            String label,
            int timeoutSec
    ) {
        RandomUtil.State previousRandom = RandomUtil.captureState();
        Game game = null;
        SnapshotBranchController controller =
                new SnapshotBranchController(snapshot, forcedIndices, stopAtReentry);
        BranchOutcome outcome = new BranchOutcome(label);
        try {
            RandomUtil.restoreState(snapshot.randomState);
            game = snapshot.gameSnapshot.createSimulationForAI();
            Player player = game.getPlayer(snapshot.playerId);
            if (!(player instanceof ComputerPlayerRL)) {
                outcome.error = "checkpoint_player_copy_type_mismatch:"
                        + (player == null ? "null" : player.getClass().getName());
                return outcome;
            }
            ((ComputerPlayerRL) player).setEngineDecisionBranchController(controller);
            try {
                resumeGameInGameThread(game, timeoutSec, label);
            } catch (EngineDecisionBranchController.BranchTerminated terminated) {
                outcome.terminationReason = terminated.getReason();
            }
            outcome.captureController(controller);
            outcome.captureTerminal(game, snapshot.playerName);
            return outcome;
        } catch (Throwable t) {
            outcome.captureController(controller);
            outcome.error = errorSummary(t);
            return outcome;
        } finally {
            if (game != null) {
                try {
                    game.end();
                } catch (Throwable ignored) {
                    // ignore cleanup failures
                }
                try {
                    game.cleanUp();
                } catch (Throwable ignored) {
                    // ignore cleanup failures
                }
            }
            RandomUtil.restoreState(previousRandom);
        }
    }

    private static void resumeGameInGameThread(Game game, int joinTimeoutSec, String label) {
        long deadlineNanos = System.nanoTime() + Math.max(1L, joinTimeoutSec) * 1_000_000_000L;
        if (ThreadUtils.isRunGameThread()) {
            resumeUntilTerminal(game, deadlineNanos);
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                resumeUntilTerminal(game, deadlineNanos);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-LIVE-CHECKPOINT-" + label);
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            try {
                game.end();
            } catch (Throwable ignored) {
                // ignore cleanup failures
            }
            throw new IllegalStateException("Interrupted while waiting for live checkpoint branch", ie);
        }
        if (gameThread.isAlive()) {
            try {
                game.end();
            } catch (Throwable ignored) {
                // ignore cleanup failures
            }
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            throw new IllegalStateException("Live checkpoint branch timeout");
        }
        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        }
        if (t instanceof Error) {
            throw (Error) t;
        }
        throw new IllegalStateException("Error while resuming live checkpoint branch", t);
    }

    private static void resumeUntilTerminal(Game game, long deadlineNanos) {
        while (!game.hasEnded()) {
            game.resume();
            if (!game.hasEnded() && System.nanoTime() >= deadlineNanos) {
                throw new IllegalStateException("Live checkpoint branch returned before terminal");
            }
        }
    }

    private static LiveCheckpointRecorder.Snapshot loadSnapshot(Path path) throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            Object obj = in.readObject();
            if (!(obj instanceof LiveCheckpointRecorder.Snapshot)) {
                throw new IllegalArgumentException("Unexpected snapshot object: "
                        + (obj == null ? "null" : obj.getClass().getName()));
            }
            return (LiveCheckpointRecorder.Snapshot) obj;
        }
    }

    private static List<Path> discoverSnapshots(Config cfg) throws Exception {
        List<Path> out = new ArrayList<>();
        if (cfg.snapshotPath != null) {
            out.add(cfg.snapshotPath);
            return out;
        }
        if (cfg.checkpointRoot == null) {
            throw new IllegalArgumentException("--checkpoint-root or --snapshot is required");
        }
        try (Stream<Path> stream = Files.walk(cfg.checkpointRoot)) {
            out.addAll(stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName() != null && p.getFileName().toString().endsWith(".ser.gz"))
                    .sorted()
                    .collect(Collectors.toList()));
        }
        return out;
    }

    private static List<Integer> firstAlternateIndices(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> sourceIndices
    ) {
        int candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
        Set<Integer> source = new HashSet<>(sourceIndices == null ? Collections.emptyList() : sourceIndices);
        for (int i = 0; i < candidateCount; i++) {
            if (!source.contains(i) && !isPassLike(snapshot.candidateTexts.get(i))) {
                return alternateReplacingFirstSource(sourceIndices, i, candidateCount);
            }
        }
        for (int i = 0; i < candidateCount; i++) {
            if (!source.contains(i)) {
                return alternateReplacingFirstSource(sourceIndices, i, candidateCount);
            }
        }
        return Collections.emptyList();
    }

    private static List<Integer> alternateReplacingFirstSource(List<Integer> sourceIndices, int alternate, int candidateCount) {
        List<Integer> out = new ArrayList<>(sourceIndices == null ? Collections.emptyList() : sourceIndices);
        if (out.isEmpty()) {
            out.add(alternate);
        } else {
            out.set(0, alternate);
        }
        return sanitizeIndices(out, candidateCount);
    }

    private static boolean isPassLike(String text) {
        String value = text == null ? "" : text.trim();
        return "Pass".equalsIgnoreCase(value) || "PASS".equals(value);
    }

    private static List<Integer> sanitizeIndices(List<Integer> values, int candidateCount) {
        if (values == null || values.isEmpty() || candidateCount <= 0) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();
        for (Integer value : values) {
            if (value == null || value < 0 || value >= candidateCount || seen.contains(value)) {
                continue;
            }
            out.add(value);
            seen.add(value);
        }
        return out;
    }

    private static List<String> selectedTexts(List<String> candidates, List<Integer> indices) {
        if (candidates == null || indices == null || indices.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Integer idx : indices) {
            if (idx != null && idx >= 0 && idx < candidates.size()) {
                out.add(candidates.get(idx));
            }
        }
        return out;
    }

    private static void appendRow(Path csvPath, BranchRow row) throws Exception {
        Files.write(csvPath, row.toCsvLine().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void writeReadme(
            Config cfg,
            Path csvPath,
            int processed,
            int discovered,
            Counts counts
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Live Checkpoint Branch Miner\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- reentry_only: ").append(cfg.reentryOnly).append("\n");
        sb.append("- timeout_sec: ").append(cfg.timeoutSec).append("\n");
        sb.append("- csv: ").append(csvPath).append("\n");
        sb.append("- classification_counts: ").append(counts.values).append("\n");
        Files.write(cfg.outDir.resolve("README.md"), sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (Integer value : values) {
            out.add(String.valueOf(value == null ? -1 : value));
        }
        return String.join("|", out);
    }

    private static String joinStrings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        return String.join("|", values);
    }

    private static String csv(String value) {
        String s = value == null ? "" : value;
        return "\"" + s.replace("\"", "\"\"").replace("\r", " ").replace("\n", " ") + "\"";
    }

    private static String errorSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String message = t.getMessage();
        return t.getClass().getSimpleName() + (message == null || message.isEmpty() ? "" : ": " + message);
    }

    private static final class SnapshotBranchController implements EngineDecisionBranchController {
        private final LiveCheckpointRecorder.Snapshot snapshot;
        private final List<Integer> forcedIndices;
        private final boolean stopAtReentry;

        private boolean seen;
        private boolean reentryMatched;
        private String reason = "";
        private String actualActionType = "";
        private String actualCandidateHash = "";
        private String actualStateHash = "";
        private List<String> actualCandidateTexts = Collections.emptyList();
        private List<String> selectedTexts = Collections.emptyList();

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> forcedIndices,
                boolean stopAtReentry
        ) {
            this.snapshot = snapshot;
            this.forcedIndices = forcedIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(forcedIndices);
            this.stopAtReentry = stopAtReentry;
        }

        @Override
        public boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
            return true;
        }

        @Override
        public <T> Choice onDecision(DecisionContext<T> context) {
            if (seen) {
                return Choice.none();
            }
            seen = true;
            actualActionType = context.actionType == null ? "" : context.actionType.name();
            actualCandidateHash = context.candidateHash;
            actualStateHash = context.stateHash;
            actualCandidateTexts = new ArrayList<>(context.candidateTexts);

            List<Integer> sanitized = sanitizeIndices(forcedIndices, context.candidateCount);
            selectedTexts = LiveCheckpointBranchMiner.selectedTexts(context.candidateTexts, sanitized);
            boolean actionMatched = actualActionType.equals(snapshot.actionType);
            boolean candidatesMatched = context.candidateTexts.equals(snapshot.candidateTexts);
            boolean selectedIndicesMatched = sanitized.equals(sanitizeIndices(snapshot.selectedIndices, context.candidateCount));
            boolean selectedTextsMatched = selectedTexts.equals(snapshot.selectedTexts);
            reentryMatched = actionMatched && candidatesMatched && selectedIndicesMatched && selectedTextsMatched;
            if (!actionMatched) {
                reason = "action_type_mismatch";
            } else if (!candidatesMatched) {
                reason = "candidate_text_mismatch";
            } else if (sanitized.isEmpty()) {
                reason = "forced_indices_invalid";
            } else if (!selectedIndicesMatched || !selectedTextsMatched) {
                reason = "source_choice_mismatch";
            } else {
                reason = "reentry_matched";
            }

            if (stopAtReentry || !reentryMatched) {
                return Choice.chooseAndTerminate(sanitized, reason);
            }
            return Choice.choose(sanitized);
        }
    }

    private static final class BranchOutcome {
        private final String label;
        private boolean firstDecisionSeen;
        private boolean reentryMatched;
        private boolean terminal;
        private boolean won;
        private boolean lost;
        private String error = "";
        private String terminationReason = "";
        private String actionType = "";
        private String candidateHash = "";
        private String stateHash = "";
        private String reason = "";

        private BranchOutcome(String label) {
            this.label = label == null ? "" : label;
        }

        private void captureController(SnapshotBranchController controller) {
            if (controller == null) {
                return;
            }
            firstDecisionSeen = controller.seen;
            reentryMatched = controller.reentryMatched;
            actionType = controller.actualActionType;
            candidateHash = controller.actualCandidateHash;
            stateHash = controller.actualStateHash;
            reason = controller.reason;
            if (!firstDecisionSeen && error.isEmpty()) {
                error = "checkpoint_no_reentry_decision";
            }
        }

        private void captureTerminal(Game game, String perspectiveName) {
            try {
                terminal = game != null && game.hasEnded();
                String winner = game == null ? "" : game.getWinner();
                String name = perspectiveName == null ? "" : perspectiveName;
                won = terminal && winner != null && !winner.isEmpty() && !name.isEmpty() && winner.contains(name);
                lost = terminal && winner != null && !winner.isEmpty() && !won;
            } catch (Throwable t) {
                error = error.isEmpty() ? errorSummary(t) : error;
            }
        }
    }

    private static final class BranchRow {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String selectedIndices = "";
        private String selectedTexts = "";
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private boolean sourceReentryAMatched = false;
        private boolean sourceReentryBMatched = false;
        private String classification = "";
        private boolean sourceTerminal = false;
        private boolean sourceWon = false;
        private boolean sourceLost = false;
        private String sourceError = "";
        private String alternateIndices = "";
        private String alternateTexts = "";
        private boolean alternateTerminal = false;
        private boolean alternateWon = false;
        private boolean alternateLost = false;
        private String alternateError = "";
        private String reentryACandidateHash = "";
        private String reentryBCandidateHash = "";
        private String reentryAStateHash = "";
        private String reentryBStateHash = "";
        private String reentryAReason = "";
        private String reentryBReason = "";

        private static BranchRow fromSnapshot(Path path, LiveCheckpointRecorder.Snapshot snapshot) {
            BranchRow row = new BranchRow();
            row.snapshotPath = path == null ? "" : path.toString();
            if (snapshot != null) {
                row.ordinal = snapshot.ordinal;
                row.decisionNumber = snapshot.decisionNumber;
                row.actionType = snapshot.actionType;
                row.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                row.selectedIndices = joinInts(snapshot.selectedIndices);
                row.selectedTexts = joinStrings(snapshot.selectedTexts);
                row.candidateHash = snapshot.candidateHash;
                row.stateHash = snapshot.stateHash;
                row.randomStateHash = snapshot.randomStateHash;
            }
            return row;
        }

        private static BranchRow loadError(Path path, String error) {
            BranchRow row = new BranchRow();
            row.snapshotPath = path == null ? "" : path.toString();
            row.classification = "snapshot_load_error";
            row.sourceError = error;
            return row;
        }

        private void applyReentry(BranchOutcome a, BranchOutcome b) {
            sourceReentryAMatched = a != null && a.reentryMatched;
            sourceReentryBMatched = b != null && b.reentryMatched;
            reentryACandidateHash = a == null ? "" : a.candidateHash;
            reentryBCandidateHash = b == null ? "" : b.candidateHash;
            reentryAStateHash = a == null ? "" : a.stateHash;
            reentryBStateHash = b == null ? "" : b.stateHash;
            reentryAReason = a == null ? "" : (a.reason.isEmpty() ? a.error : a.reason);
            reentryBReason = b == null ? "" : (b.reason.isEmpty() ? b.error : b.reason);
        }

        private void applySource(BranchOutcome source) {
            sourceTerminal = source != null && source.terminal;
            sourceWon = source != null && source.won;
            sourceLost = source != null && source.lost;
            sourceError = source == null ? "" : source.error;
        }

        private void applyAlternate(BranchOutcome alternate) {
            alternateTerminal = alternate != null && alternate.terminal;
            alternateWon = alternate != null && alternate.won;
            alternateLost = alternate != null && alternate.lost;
            alternateError = alternate == null ? "" : alternate.error;
        }

        private String toCsvLine() {
            List<String> cells = new ArrayList<>();
            cells.add(csv(snapshotPath));
            cells.add(String.valueOf(ordinal));
            cells.add(String.valueOf(decisionNumber));
            cells.add(csv(actionType));
            cells.add(String.valueOf(candidateCount));
            cells.add(csv(selectedIndices));
            cells.add(csv(selectedTexts));
            cells.add(csv(candidateHash));
            cells.add(csv(stateHash));
            cells.add(csv(randomStateHash));
            cells.add(String.valueOf(sourceReentryAMatched));
            cells.add(String.valueOf(sourceReentryBMatched));
            cells.add(csv(classification));
            cells.add(String.valueOf(sourceTerminal));
            cells.add(String.valueOf(sourceWon));
            cells.add(String.valueOf(sourceLost));
            cells.add(csv(sourceError));
            cells.add(csv(alternateIndices));
            cells.add(csv(alternateTexts));
            cells.add(String.valueOf(alternateTerminal));
            cells.add(String.valueOf(alternateWon));
            cells.add(String.valueOf(alternateLost));
            cells.add(csv(alternateError));
            cells.add(csv(reentryACandidateHash));
            cells.add(csv(reentryBCandidateHash));
            cells.add(csv(reentryAStateHash));
            cells.add(csv(reentryBStateHash));
            cells.add(csv(reentryAReason));
            cells.add(csv(reentryBReason));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class Counts {
        private final Map<String, Integer> values = new LinkedHashMap<>();

        private void add(String classification) {
            String key = classification == null || classification.isEmpty() ? "unknown" : classification;
            values.put(key, values.getOrDefault(key, 0) + 1);
        }
    }

    private static final class Config {
        private Path checkpointRoot;
        private Path snapshotPath;
        private Path outDir = defaultOutDir();
        private int maxSnapshots = 0;
        private int timeoutSec = 30;
        private boolean reentryOnly = false;
        private Set<String> actionTypes = Collections.emptySet();

        private static Config parse(String[] args) {
            Config cfg = new Config();
            Map<String, String> values = parseArgs(args);
            if (values.containsKey("checkpoint-root")) {
                cfg.checkpointRoot = Paths.get(values.get("checkpoint-root"));
            }
            if (values.containsKey("snapshot")) {
                cfg.snapshotPath = Paths.get(values.get("snapshot"));
            }
            if (values.containsKey("out")) {
                cfg.outDir = Paths.get(values.get("out"));
            }
            if (values.containsKey("max-snapshots")) {
                cfg.maxSnapshots = Integer.parseInt(values.get("max-snapshots"));
            }
            if (values.containsKey("timeout-sec")) {
                cfg.timeoutSec = Integer.parseInt(values.get("timeout-sec"));
            }
            if (values.containsKey("reentry-only")) {
                cfg.reentryOnly = Boolean.parseBoolean(values.get("reentry-only"));
            }
            if (values.containsKey("action-types")) {
                cfg.actionTypes = parseSet(values.get("action-types"));
            }
            return cfg;
        }

        private static Path defaultOutDir() {
            String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss", Locale.US));
            return Paths.get("local-training", "local_pbt", "live_checkpoint_branch_miner", ts);
        }

        private static Map<String, String> parseArgs(String[] args) {
            Map<String, String> out = new LinkedHashMap<>();
            if (args == null) {
                return out;
            }
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg == null || !arg.startsWith("--")) {
                    continue;
                }
                String key;
                String value;
                int eq = arg.indexOf('=');
                if (eq >= 0) {
                    key = arg.substring(2, eq);
                    value = arg.substring(eq + 1);
                } else {
                    key = arg.substring(2);
                    if (i + 1 < args.length && args[i + 1] != null && !args[i + 1].startsWith("--")) {
                        value = args[++i];
                    } else {
                        value = "true";
                    }
                }
                out.put(key, value);
            }
            return out;
        }

        private static Set<String> parseSet(String raw) {
            if (raw == null || raw.trim().isEmpty() || "*".equals(raw.trim())) {
                return Collections.emptySet();
            }
            Set<String> out = new HashSet<>();
            for (String item : raw.split(",")) {
                String value = item.trim();
                if (!value.isEmpty()) {
                    out.add(value);
                }
            }
            return out;
        }
    }
}
