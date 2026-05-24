package mage.player.ai.rl;

import mage.game.Game;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPInputStream;

/**
 * Converts terminal-line value-target CSV rows into serialized TrainingData.
 */
public final class TerminalLineValueTargetTrainingDataExporter {

    private static final String SUMMARY_HEADER =
            "row_number,example_id,status,snapshot_path,ordinal,decision_number,action_type,candidate_count,"
                    + "expected_candidate_hash,actual_candidate_hash,expected_state_hash,actual_state_hash,"
                    + "target_sum,target_positive_count,best_indices,reentry_matched,captured,quality_flags,error\n";

    private TerminalLineValueTargetTrainingDataExporter() {
    }

    public static void main(String[] args) throws Exception {
        Config cfg = Config.parse(args);
        if (cfg.valueTargets == null) {
            throw new IllegalArgumentException("--value-targets is required");
        }
        if (cfg.outFile == null) {
            throw new IllegalArgumentException("--out is required");
        }
        List<CsvRow> rows = readCsv(cfg.valueTargets);
        List<StateSequenceBuilder.TrainingData> records = new ArrayList<>();
        List<ExportSummary> summaries = new ArrayList<>();
        int eligibleRows = 0;
        for (CsvRow row : rows) {
            if (cfg.maxRecords > 0 && records.size() >= cfg.maxRecords) {
                break;
            }
            if (!isTrainable(row)) {
                summaries.add(ExportSummary.skipped(row, "not_trainable_manifest_status"));
                continue;
            }
            eligibleRows++;
            ExportSummary summary;
            try {
                ExportResult result = exportRow(row, cfg);
                summary = result.summary;
                if (result.trainingData != null) {
                    records.add(result.trainingData);
                }
            } catch (Throwable t) {
                summary = ExportSummary.error(row, errorSummary(t));
            }
            summaries.add(summary);
        }

        writeTrainingData(cfg.outFile, records);
        Path summaryFile = cfg.summaryFile == null
                ? cfg.outFile.resolveSibling(stripExtension(cfg.outFile.getFileName().toString()) + "_summary.csv")
                : cfg.summaryFile;
        writeSummary(summaryFile, summaries);
        System.out.println(String.format(Locale.US,
                "terminalLineValueTargets eligibleRows=%d exportedRecords=%d summary=%s out=%s",
                eligibleRows, records.size(), summaryFile, cfg.outFile));
        if (cfg.expectRecords >= 0 && records.size() != cfg.expectRecords) {
            throw new IllegalStateException("expected " + cfg.expectRecords
                    + " exported records, found " + records.size());
        }
    }

    private static boolean isTrainable(CsvRow row) {
        String status = row.get("manifest_status");
        String trainingStatus = row.get("training_status");
        return "admitted".equals(status) && (trainingStatus.isEmpty() || "trainable".equals(trainingStatus));
    }

    private static ExportResult exportRow(CsvRow row, Config cfg) throws Exception {
        Path snapshotPath = cfg.resolveSnapshotPath(row.get("snapshot_path"));
        LiveCheckpointRecorder.Snapshot snapshot = loadSnapshot(snapshotPath);
        float[] target = parseTarget(row, cfg.targetMode);
        List<Integer> bestIndices = parseIndices(row.get("best_indices"));
        if (!hasTargetSignal(target, cfg.targetMode)) {
            return new ExportResult(null, ExportSummary.error(row, "empty_" + cfg.targetMode.optionName + "_target"));
        }
        if (bestIndices.isEmpty()) {
            bestIndices = bestIndicesFromTarget(target, cfg.targetMode);
        }
        if (bestIndices.isEmpty()) {
            return new ExportResult(null, ExportSummary.error(row, "missing_best_indices"));
        }

        RandomUtil.State previousRandom = RandomUtil.captureState();
        CaptureController controller = new CaptureController(row, snapshot, bestIndices, target);
        try {
            if (snapshot.randomState != null) {
                RandomUtil.restoreState(snapshot.randomState);
            }
            Game game = snapshot.gameSnapshot == null ? null : snapshot.gameSnapshot.createSimulationForAI();
            if (game == null) {
                return new ExportResult(null, ExportSummary.error(row, "snapshot_missing_game"));
            }
            ComputerPlayerRL player = findPlayer(game, snapshot.playerId, snapshot.playerName);
            if (player == null) {
                return new ExportResult(null, ExportSummary.error(row, "player_not_found"));
            }
            player.setEngineDecisionBranchController(controller);
            try {
                resumeGameInGameThread(game, cfg.timeoutSec);
            } catch (EngineDecisionBranchController.BranchTerminated terminated) {
                controller.terminationReason = terminated.getReason();
            }
            StateSequenceBuilder.TrainingData captured = controller.capturedTrainingData;
            ExportSummary summary = ExportSummary.fromController(row, controller, captured != null ? "" : "training_data_not_captured");
            return new ExportResult(captured, summary);
        } finally {
            RandomUtil.restoreState(previousRandom);
        }
    }

    private static ComputerPlayerRL findPlayer(Game game, UUID playerId, String playerName) {
        Player player = playerId == null ? null : game.getPlayer(playerId);
        if (player instanceof ComputerPlayerRL) {
            return (ComputerPlayerRL) player;
        }
        if (game.getPlayers() != null) {
            for (Player candidate : game.getPlayers().values()) {
                if (candidate instanceof ComputerPlayerRL
                        && playerName != null
                        && playerName.equals(candidate.getName())) {
                    return (ComputerPlayerRL) candidate;
                }
            }
        }
        return null;
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

    private static void resumeGameInGameThread(Game game, int joinTimeoutSec) {
        long deadlineNanos = System.nanoTime() + Math.max(1L, joinTimeoutSec) * 1_000_000_000L;
        if (ThreadUtils.isRunGameThread()) {
            resumeUntilDone(game, deadlineNanos);
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                resumeUntilDone(game, deadlineNanos);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-TERMINAL-LINE-VALUE-TARGET-EXPORT");
        gameThread.setDaemon(true);
        gameThread.start();
        try {
            gameThread.join(Math.max(1L, joinTimeoutSec) * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while exporting terminal-line value target", e);
        }
        if (gameThread.isAlive()) {
            gameThread.interrupt();
            throw new IllegalStateException("Timed out waiting for terminal-line value target reentry");
        }
        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof EngineDecisionBranchController.BranchTerminated) {
            throw (EngineDecisionBranchController.BranchTerminated) t;
        }
        throw new IllegalStateException("Error while exporting terminal-line value target", t);
    }

    private static void resumeUntilDone(Game game, long deadlineNanos) {
        while (!game.hasEnded()) {
            game.resume();
            if (System.nanoTime() >= deadlineNanos) {
                throw new IllegalStateException("Timed out before value-target training data capture");
            }
        }
    }

    private static void writeTrainingData(Path path, List<StateSequenceBuilder.TrainingData> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(new ArrayList<>(records));
        }
    }

    private static void writeSummary(Path path, List<ExportSummary> summaries) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            out.write(SUMMARY_HEADER);
            for (ExportSummary summary : summaries) {
                out.write(summary.toCsv());
                out.write('\n');
            }
        }
    }

    private static List<CsvRow> readCsv(Path path) throws IOException {
        List<CsvRow> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                return rows;
            }
            List<String> fields = parseCsvLine(stripBom(header));
            String line;
            int rowNumber = 1;
            while ((line = reader.readLine()) != null) {
                rowNumber++;
                if (line.trim().isEmpty()) {
                    continue;
                }
                List<String> values = parseCsvLine(line);
                Map<String, String> map = new LinkedHashMap<>();
                for (int i = 0; i < fields.size(); i++) {
                    map.put(fields.get(i), i < values.size() ? values.get(i) : "");
                }
                rows.add(new CsvRow(rowNumber, map));
            }
        }
        return rows;
    }

    private static List<String> parseCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        out.add(cur.toString());
        return out;
    }

    private static String stripBom(String value) {
        if (value != null && !value.isEmpty() && value.charAt(0) == '\uFEFF') {
            return value.substring(1);
        }
        return value;
    }

    private static float[] parseTarget(CsvRow row, TargetMode mode) {
        if (mode == TargetMode.SIGNED_VALUES) {
            return parseSignedValueTargets(row.get("candidate_value_estimates"), row.get("candidate_attempts"));
        }
        if (mode == TargetMode.ADVANTAGE_VALUES) {
            return parseAdvantageValueTargets(row.get("candidate_value_estimates"), row.get("candidate_attempts"));
        }
        return parseTargetDistribution(row.get("target_distribution"));
    }

    private static float[] parseTargetDistribution(String raw) {
        float[] out = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        if (raw == null || raw.trim().isEmpty()) {
            return out;
        }
        for (Map.Entry<Integer, Float> entry : parseFloatPairs(raw).entrySet()) {
            int idx = entry.getKey();
            float value = entry.getValue();
            if (idx >= 0 && idx < out.length && value > 0.0f) {
                out[idx] = value;
            }
        }
        float sum = targetSum(out);
        if (sum > 0.0f) {
            for (int i = 0; i < out.length; i++) {
                out[i] /= sum;
            }
        }
        return out;
    }

    private static float[] parseSignedValueTargets(String rawValues, String rawAttempts) {
        float[] out = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        Arrays.fill(out, -2.0f);
        Map<Integer, Float> values = parseFloatPairs(rawValues);
        Map<Integer, Integer> attempts = parseIntPairs(rawAttempts);
        for (Map.Entry<Integer, Integer> entry : attempts.entrySet()) {
            int idx = entry.getKey();
            int attemptCount = entry.getValue();
            if (idx < 0 || idx >= out.length || attemptCount <= 0) {
                continue;
            }
            float value = values.containsKey(idx) ? values.get(idx) : 0.0f;
            out[idx] = Math.max(-1.0f, Math.min(1.0f, 2.0f * value - 1.0f));
        }
        return out;
    }

    private static float[] parseAdvantageValueTargets(String rawValues, String rawAttempts) {
        float[] out = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        Arrays.fill(out, -2.0f);
        Map<Integer, Float> values = parseFloatPairs(rawValues);
        Map<Integer, Integer> attempts = parseIntPairs(rawAttempts);
        List<Integer> observed = new ArrayList<>();
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (Map.Entry<Integer, Integer> entry : attempts.entrySet()) {
            int idx = entry.getKey();
            int attemptCount = entry.getValue();
            if (idx < 0 || idx >= out.length || attemptCount <= 0) {
                continue;
            }
            float value = values.containsKey(idx) ? values.get(idx) : 0.0f;
            observed.add(idx);
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        if (observed.isEmpty()) {
            return out;
        }
        float range = max - min;
        for (Integer idx : observed) {
            float value = values.containsKey(idx) ? values.get(idx) : 0.0f;
            out[idx] = range <= 1e-6f
                    ? 0.0f
                    : Math.max(-1.0f, Math.min(1.0f, (2.0f * ((value - min) / range)) - 1.0f));
        }
        return out;
    }

    private static Map<Integer, Float> parseFloatPairs(String raw) {
        Map<Integer, Float> out = new LinkedHashMap<>();
        if (raw == null || raw.trim().isEmpty()) {
            return out;
        }
        for (String part : raw.split("\\|")) {
            String[] bits = part.split(":", 2);
            if (bits.length != 2) {
                continue;
            }
            try {
                int idx = Integer.parseInt(bits[0].trim());
                float value = Float.parseFloat(bits[1].trim());
                if (!Float.isNaN(value) && !Float.isInfinite(value)) {
                    out.put(idx, value);
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return out;
    }

    private static Map<Integer, Integer> parseIntPairs(String raw) {
        Map<Integer, Integer> out = new LinkedHashMap<>();
        if (raw == null || raw.trim().isEmpty()) {
            return out;
        }
        for (String part : raw.split("\\|")) {
            String[] bits = part.split(":", 2);
            if (bits.length != 2) {
                continue;
            }
            try {
                int idx = Integer.parseInt(bits[0].trim());
                int value = Integer.parseInt(bits[1].trim());
                out.put(idx, value);
            } catch (NumberFormatException ignored) {
            }
        }
        return out;
    }

    private static List<Integer> parseIndices(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        for (String part : raw.replace(",", "|").split("\\|")) {
            try {
                int idx = Integer.parseInt(part.trim());
                if (idx >= 0 && idx < StateSequenceBuilder.TrainingData.MAX_CANDIDATES) {
                    out.add(idx);
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return out;
    }

    private static List<Integer> bestIndicesFromTarget(float[] target, TargetMode mode) {
        int best = -1;
        float bestValue = mode.usesObservedTargets() ? -2.1f : 0.0f;
        for (int i = 0; i < target.length; i++) {
            if (mode.usesObservedTargets() && !isObservedSignedTarget(target[i])) {
                continue;
            }
            if (target[i] > bestValue) {
                bestValue = target[i];
                best = i;
            }
        }
        return best < 0 ? Collections.emptyList() : Collections.singletonList(best);
    }

    private static boolean hasTargetSignal(float[] target, TargetMode mode) {
        if (target == null) {
            return false;
        }
        if (mode.usesObservedTargets()) {
            for (float value : target) {
                if (isObservedSignedTarget(value)) {
                    return true;
                }
            }
            return false;
        }
        return targetSum(target) > 0.0f;
    }

    private static boolean isObservedSignedTarget(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value) && value >= -1.0f && value <= 1.0f;
    }

    private static float targetSum(float[] target) {
        float sum = 0.0f;
        if (target != null) {
            for (float value : target) {
                sum += value;
            }
        }
        return sum;
    }

    private static int positiveTargetCount(float[] target) {
        int count = 0;
        if (target != null) {
            for (float value : target) {
                if (value > 0.0f) {
                    count++;
                }
            }
        }
        return count;
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Integer value : values) {
            if (value == null) {
                continue;
            }
            if (sb.length() > 0) {
                sb.append('|');
            }
            sb.append(value);
        }
        return sb.toString();
    }

    private static String csv(String value) {
        String text = value == null ? "" : value;
        if (text.indexOf(',') >= 0 || text.indexOf('"') >= 0 || text.indexOf('\n') >= 0 || text.indexOf('\r') >= 0) {
            return "\"" + text.replace("\"", "\"\"") + "\"";
        }
        return text;
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot <= 0 ? name : name.substring(0, dot);
    }

    private static String errorSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String message = t.getMessage();
        return t.getClass().getSimpleName() + (message == null || message.isEmpty() ? "" : ":" + message);
    }

    private enum TargetMode {
        DISTRIBUTION("distribution"),
        SIGNED_VALUES("signed-values"),
        ADVANTAGE_VALUES("advantage-values");

        private final String optionName;

        TargetMode(String optionName) {
            this.optionName = optionName;
        }

        private static TargetMode parse(String raw) {
            String normalized = raw == null ? "" : raw.trim().toLowerCase(Locale.ROOT);
            if (normalized.isEmpty() || "distribution".equals(normalized) || "soft".equals(normalized)) {
                return DISTRIBUTION;
            }
            if ("signed-values".equals(normalized) || "signed_value".equals(normalized)
                    || "signed-values-from-candidate-values".equals(normalized)) {
                return SIGNED_VALUES;
            }
            if ("advantage-values".equals(normalized) || "relative-values".equals(normalized)
                    || "rank-values".equals(normalized) || "sibling-advantage".equals(normalized)) {
                return ADVANTAGE_VALUES;
            }
            throw new IllegalArgumentException("Unsupported --target-mode: " + raw);
        }

        private boolean usesObservedTargets() {
            return this != DISTRIBUTION;
        }
    }

    private static final class CaptureController implements EngineDecisionBranchController {
        private final CsvRow row;
        private final LiveCheckpointRecorder.Snapshot snapshot;
        private final List<Integer> bestIndices;
        private final float[] target;

        private boolean seen;
        private boolean reentryMatched;
        private String reason = "";
        private String actualCandidateHash = "";
        private String actualStateHash = "";
        private String actualActionType = "";
        private StateSequenceBuilder.TrainingData capturedTrainingData;
        private String terminationReason = "";

        private CaptureController(CsvRow row, LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> bestIndices, float[] target) {
            this.row = row;
            this.snapshot = snapshot;
            this.bestIndices = bestIndices == null ? Collections.emptyList() : new ArrayList<>(bestIndices);
            this.target = Arrays.copyOf(target, target.length);
        }

        @Override
        public boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
            return false;
        }

        @Override
        public boolean shouldBypassModelInference() {
            return true;
        }

        @Override
        public boolean shouldCaptureTrainingData() {
            return true;
        }

        @Override
        public <T> Choice onDecision(DecisionContext<T> context) {
            if (seen) {
                return Choice.none();
            }
            seen = true;
            if (context == null || snapshot == null) {
                reason = "missing_context";
                return Choice.chooseAndTerminate(Collections.emptyList(), reason);
            }
            actualActionType = context.actionType == null ? "" : context.actionType.name();
            actualCandidateHash = context.candidateHash;
            actualStateHash = context.stateHash;
            boolean actionMatched = actualActionType.equals(row.get("action_type"));
            boolean candidateMatched = actualCandidateHash.equals(row.get("candidate_hash"))
                    && context.candidateTexts.equals(snapshot.candidateTexts);
            boolean stateMatched = actualStateHash.equals(row.get("state_hash"));
            boolean indicesValid = !bestIndices.isEmpty();
            for (Integer idx : bestIndices) {
                if (idx == null || idx < 0 || idx >= context.candidateCount) {
                    indicesValid = false;
                    break;
                }
            }
            reentryMatched = actionMatched && candidateMatched && stateMatched && indicesValid;
            if (!actionMatched) {
                reason = "action_type_mismatch";
            } else if (!candidateMatched) {
                reason = "candidate_mismatch";
            } else if (!stateMatched) {
                reason = "state_hash_mismatch";
            } else if (!indicesValid) {
                reason = "best_indices_invalid";
            } else {
                reason = "reentry_matched";
            }
            if (!reentryMatched) {
                return Choice.chooseAndTerminate(Collections.emptyList(), reason);
            }
            return Choice.choose(bestIndices);
        }

        @Override
        public void onTrainingData(StateSequenceBuilder.TrainingData trainingData) {
            if (!reentryMatched || trainingData == null) {
                return;
            }
            trainingData.setMctsVisitTargets(Arrays.copyOf(target, target.length));
            capturedTrainingData = trainingData;
            throw new BranchTerminated("training_data_captured");
        }
    }

    private static final class CsvRow {
        private final int rowNumber;
        private final Map<String, String> values;

        private CsvRow(int rowNumber, Map<String, String> values) {
            this.rowNumber = rowNumber;
            this.values = values == null ? Collections.emptyMap() : values;
        }

        private String get(String key) {
            String value = values.get(key);
            return value == null ? "" : value.trim();
        }
    }

    private static final class ExportResult {
        private final StateSequenceBuilder.TrainingData trainingData;
        private final ExportSummary summary;

        private ExportResult(StateSequenceBuilder.TrainingData trainingData, ExportSummary summary) {
            this.trainingData = trainingData;
            this.summary = summary;
        }
    }

    private static final class ExportSummary {
        private final CsvRow row;
        private String status;
        private String actualCandidateHash = "";
        private String actualStateHash = "";
        private String targetSum = "";
        private String positiveTargetCount = "";
        private String reentryMatched = "false";
        private String captured = "false";
        private String error = "";

        private ExportSummary(CsvRow row) {
            this.row = row;
        }

        private static ExportSummary skipped(CsvRow row, String reason) {
            ExportSummary summary = new ExportSummary(row);
            summary.status = "skipped";
            summary.error = reason;
            return summary;
        }

        private static ExportSummary error(CsvRow row, String reason) {
            ExportSummary summary = new ExportSummary(row);
            summary.status = "error";
            summary.error = reason == null ? "" : reason;
            return summary;
        }

        private static ExportSummary fromController(CsvRow row, CaptureController controller, String error) {
            ExportSummary summary = new ExportSummary(row);
            summary.status = error == null || error.isEmpty() ? "exported" : "error";
            summary.actualCandidateHash = controller.actualCandidateHash;
            summary.actualStateHash = controller.actualStateHash;
            summary.targetSum = String.format(Locale.US, "%.6f", targetSum(controller.target));
            summary.positiveTargetCount = Integer.toString(positiveTargetCount(controller.target));
            summary.reentryMatched = Boolean.toString(controller.reentryMatched);
            summary.captured = Boolean.toString(controller.capturedTrainingData != null);
            summary.error = error == null || error.isEmpty() ? controller.terminationReason : error;
            return summary;
        }

        private String toCsv() {
            return row.rowNumber
                    + "," + csv(row.get("example_id"))
                    + "," + csv(status)
                    + "," + csv(row.get("snapshot_path"))
                    + "," + csv(row.get("ordinal"))
                    + "," + csv(row.get("decision_number"))
                    + "," + csv(row.get("action_type"))
                    + "," + csv(row.get("candidate_count"))
                    + "," + csv(row.get("candidate_hash"))
                    + "," + csv(actualCandidateHash)
                    + "," + csv(row.get("state_hash"))
                    + "," + csv(actualStateHash)
                    + "," + csv(targetSum)
                    + "," + csv(positiveTargetCount)
                    + "," + csv(row.get("best_indices"))
                    + "," + csv(reentryMatched)
                    + "," + csv(captured)
                    + "," + csv(row.get("quality_flags"))
                    + "," + csv(error);
        }
    }

    private static final class Config {
        private Path valueTargets;
        private Path outFile;
        private Path summaryFile;
        private String snapshotPathPrefixFrom = "";
        private String snapshotPathPrefixTo = "";
        private TargetMode targetMode = TargetMode.DISTRIBUTION;
        private int timeoutSec = 30;
        private int maxRecords = 0;
        private int expectRecords = -1;

        private static Config parse(String[] args) {
            Config cfg = new Config();
            Map<String, String> values = parseArgs(args);
            if (values.containsKey("value-targets")) {
                cfg.valueTargets = Paths.get(values.get("value-targets"));
            }
            if (values.containsKey("out")) {
                cfg.outFile = Paths.get(values.get("out"));
            }
            if (values.containsKey("summary")) {
                cfg.summaryFile = Paths.get(values.get("summary"));
            }
            if (values.containsKey("snapshot-path-prefix-from")) {
                cfg.snapshotPathPrefixFrom = values.get("snapshot-path-prefix-from");
            }
            if (values.containsKey("snapshot-path-prefix-to")) {
                cfg.snapshotPathPrefixTo = values.get("snapshot-path-prefix-to");
            }
            if (values.containsKey("target-mode")) {
                cfg.targetMode = TargetMode.parse(values.get("target-mode"));
            }
            if (cfg.snapshotPathPrefixFrom.isEmpty() != cfg.snapshotPathPrefixTo.isEmpty()) {
                throw new IllegalArgumentException("--snapshot-path-prefix-from and --snapshot-path-prefix-to must be provided together");
            }
            if (values.containsKey("timeout-sec")) {
                cfg.timeoutSec = Math.max(1, Integer.parseInt(values.get("timeout-sec")));
            }
            if (values.containsKey("max-records")) {
                cfg.maxRecords = Math.max(0, Integer.parseInt(values.get("max-records")));
            }
            if (values.containsKey("expect-records")) {
                cfg.expectRecords = Integer.parseInt(values.get("expect-records"));
            }
            return cfg;
        }

        private Path resolveSnapshotPath(String rawPath) {
            String pathText = rawPath == null ? "" : rawPath.trim();
            if (!snapshotPathPrefixFrom.isEmpty()) {
                String normalizedPath = normalizeSnapshotPrefix(pathText);
                String normalizedFrom = normalizeSnapshotPrefix(snapshotPathPrefixFrom);
                if (normalizedPath.equals(normalizedFrom) || normalizedPath.startsWith(normalizedFrom + "/")) {
                    String suffix = normalizedPath.length() == normalizedFrom.length()
                            ? ""
                            : normalizedPath.substring(normalizedFrom.length() + 1);
                    Path path = Paths.get(snapshotPathPrefixTo);
                    if (!suffix.isEmpty()) {
                        for (String part : suffix.split("/")) {
                            if (!part.isEmpty()) {
                                path = path.resolve(part);
                            }
                        }
                    }
                    return path;
                }
            }
            return Paths.get(pathText);
        }

        private static String normalizeSnapshotPrefix(String text) {
            String normalized = text == null ? "" : text.trim().replace('\\', '/');
            while (normalized.endsWith("/") && normalized.length() > 1) {
                normalized = normalized.substring(0, normalized.length() - 1);
            }
            return normalized;
        }

        private static Map<String, String> parseArgs(String[] args) {
            Map<String, String> values = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg == null || !arg.startsWith("--")) {
                    continue;
                }
                String key = arg.substring(2);
                String value = "true";
                int eq = key.indexOf('=');
                if (eq >= 0) {
                    value = key.substring(eq + 1);
                    key = key.substring(0, eq);
                } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    value = args[++i];
                }
                values.put(key, value);
            }
            return values;
        }
    }
}
