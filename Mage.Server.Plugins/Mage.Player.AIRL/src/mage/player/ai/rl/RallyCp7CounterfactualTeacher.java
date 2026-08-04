package mage.player.ai.rl;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.constants.PhaseStep;
import mage.game.Game;
import mage.player.ai.ShadowCp7;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

/**
 * Writes non-mutating CP7 labels at candidate-controlled Rally priority states.
 *
 * <p>The live candidate still supplies the applied action. CP7 plans on its own
 * XMage simulation copy, and this writer records only the mapped Rust row. The
 * resulting JSONL is joined to the existing Rust outcome export by episode,
 * step, physical-decision id, and model-input SHA-256.</p>
 */
public final class RallyCp7CounterfactualTeacher implements AutoCloseable {

    public static final String SCHEMA =
            "xmage-rally-cp7-counterfactual-teacher-jsonl/v1";
    public static final String SELECTION_SOURCE =
            "xmage_rally_shadow_cp7_candidate_priority";

    private final BufferedWriter writer;
    private final int skill;
    private final int maxThinkSeconds;
    private long recordOrdinal;
    private long queried;
    private long matched;
    private long passes;
    private long timeouts;
    private long ambiguous;
    private long unmatched;
    private long totalMillis;
    private boolean closed;

    private RallyCp7CounterfactualTeacher(
            BufferedWriter writer, int skill, int maxThinkSeconds) throws IOException {
        this.writer = writer;
        this.skill = skill;
        this.maxThinkSeconds = maxThinkSeconds;
        JsonObject header = new JsonObject();
        header.addProperty("schema", SCHEMA);
        header.addProperty("record_type", "header");
        header.addProperty("record_ordinal", recordOrdinal++);
        header.addProperty("selection_source", SELECTION_SOURCE);
        header.addProperty("cp7_skill", skill);
        header.addProperty("max_think_seconds", maxThinkSeconds);
        header.addProperty("label_scope", "candidate_priority_surface_only/v1");
        header.addProperty("live_action_source", "candidate_checkpoint_policy");
        write(header);
    }

    public static RallyCp7CounterfactualTeacher create(
            Path output, int skill, int maxThinkSeconds) throws IOException {
        if (output == null || output.getParent() == null) {
            throw new IllegalArgumentException("counterfactual teacher output must name a file");
        }
        if (skill < 1 || skill > 10 || maxThinkSeconds < 1 || maxThinkSeconds > 120) {
            throw new IllegalArgumentException("invalid counterfactual CP7 search limits");
        }
        Path parent = output.toAbsolutePath().normalize().getParent().toRealPath();
        Path resolved = parent.resolve(output.getFileName());
        BufferedWriter writer = Files.newBufferedWriter(
                resolved,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE);
        try {
            return new RallyCp7CounterfactualTeacher(writer, skill, maxThinkSeconds);
        } catch (IOException | RuntimeException | Error error) {
            writer.close();
            throw error;
        }
    }

    public synchronized void capturePriority(
            XMageRallyBridgeProtocol.DecisionBody decision,
            List<? extends ActivatedAbility> abilitiesByRustRow,
            Game game,
            UUID physicalPlayerId) {
        requireOpen();
        if (decision == null || abilitiesByRustRow == null || game == null
                || physicalPlayerId == null || game.isSimulation()) {
            throw new IllegalArgumentException("counterfactual capture lacks a live decision");
        }
        if (!decision.isCandidateControlsCurrentActor()
                || decision.getActingPlayer() != decision.getCandidateSeat()
                || !"surface".equals(decision.getDecisionKind())
                || decision.getSubstepIndex() != 0
                || decision.getSubstepCount() != 1
                || decision.getLegalActionCount() != abilitiesByRustRow.size()) {
            throw new IllegalArgumentException("counterfactual capture decision shape mismatch");
        }

        int selectedIndex = -1;
        String status;
        String selectedText = "";
        long started = System.nanoTime();
        try {
            int passIndex = uniquePassIndex(abilitiesByRustRow);
            PhaseStep step = game.getTurnStepType();
            boolean planningStep = step == PhaseStep.PRECOMBAT_MAIN
                    || step == PhaseStep.POSTCOMBAT_MAIN
                    || step == PhaseStep.DECLARE_ATTACKERS
                    || step == PhaseStep.DECLARE_BLOCKERS;
            if (!planningStep) {
                selectedIndex = passIndex;
                selectedText = "PASS";
                status = "step_pass";
            } else {
                ShadowCp7 shadow = new ShadowCp7(physicalPlayerId, skill);
                shadow.setMaxThinkTimeSecs(maxThinkSeconds);
                List<Ability> plan = shadow.planOnce(game);
                long elapsedMillis = elapsedMillis(started);
                if (plan.isEmpty()) {
                    if (elapsedMillis >= maxThinkSeconds * 1000L - 750L) {
                        status = "timeout_pass";
                    } else {
                        selectedIndex = passIndex;
                        selectedText = "PASS";
                        status = "plan_pass";
                    }
                } else {
                    Ability first = plan.get(0);
                    if (first instanceof PassAbility) {
                        selectedIndex = passIndex;
                        selectedText = "PASS";
                        status = "plan_pass";
                    } else {
                        selectedText = TerminalPrefixSearch.describeOne(first, game);
                        Match match = match(first, selectedText, abilitiesByRustRow, game);
                        selectedIndex = match.index;
                        status = match.status;
                    }
                }
            }
        } catch (RuntimeException error) {
            status = "error:" + error.getClass().getSimpleName();
        }
        long elapsedMillis = elapsedMillis(started);
        observe(status, elapsedMillis);

        JsonObject row = new JsonObject();
        row.addProperty("schema", SCHEMA);
        row.addProperty("record_type", "decision");
        row.addProperty("record_ordinal", recordOrdinal++);
        row.addProperty("selection_source", SELECTION_SOURCE);
        row.addProperty("base_seed_u64_hex", decision.getBaseSeedU64Hex());
        row.addProperty("pair_index", decision.getPairIndex());
        row.addProperty("pair_environment_seed_u64_hex",
                decision.getPairEnvironmentSeedU64Hex());
        row.addProperty("episode_id", decision.getEpisodeId());
        row.addProperty("step", decision.getStep());
        row.addProperty("environment_revision", decision.getEnvironmentRevision());
        row.addProperty("physical_decision_id", decision.getPhysicalDecisionId());
        row.addProperty("substep_index", decision.getSubstepIndex());
        row.addProperty("substep_count", decision.getSubstepCount());
        row.addProperty("acting_player", decision.getActingPlayer().wire());
        row.addProperty("candidate_seat", decision.getCandidateSeat().wire());
        row.addProperty("decision_kind", decision.getDecisionKind());
        row.addProperty("legal_action_count", decision.getLegalActionCount());
        row.addProperty("candidate_selected_index", decision.getSelectedActionIndex());
        row.addProperty("teacher_selected_index", selectedIndex);
        row.addProperty("teacher_status", status);
        row.addProperty("teacher_text", selectedText == null ? "" : selectedText);
        row.addProperty("teacher_elapsed_ms", elapsedMillis);
        row.addProperty("candidate_order_commitment_128_hex",
                decision.getCandidateOrderCommitment128Hex());
        row.addProperty("model_input_sha256", decision.getModelInputSha256());
        JsonArray semantics = new JsonArray();
        JsonParser parser = new JsonParser();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : decision.getActionSemantics()) {
            semantics.add(parser.parse(semantic.getCanonicalJson()));
        }
        row.add("action_semantics", semantics);
        try {
            write(row);
        } catch (IOException error) {
            throw new IllegalStateException("counterfactual teacher export failed", error);
        }
    }

    private static Match match(
            Ability planned,
            String plannedText,
            List<? extends ActivatedAbility> abilitiesByRustRow,
            Game game) {
        UUID sourceId = planned.getSourceId();
        int sourceHits = 0;
        int sourceIndex = -1;
        for (int index = 0; index < abilitiesByRustRow.size(); index++) {
            ActivatedAbility candidate = abilitiesByRustRow.get(index);
            if (!(candidate instanceof PassAbility) && sourceId != null
                    && sourceId.equals(candidate.getSourceId())) {
                sourceHits++;
                sourceIndex = index;
            }
        }
        if (sourceHits == 1) {
            return new Match(sourceIndex, "source_id");
        }
        String wanted = normalize(plannedText);
        for (int index = 0; index < abilitiesByRustRow.size(); index++) {
            if (wanted.equals(normalize(TerminalPrefixSearch.describeOne(
                    abilitiesByRustRow.get(index), game)))) {
                return new Match(index, sourceHits > 1 ? "text_disambig" : "text");
            }
        }
        return sourceHits > 0
                ? new Match(sourceIndex, "source_ambig")
                : new Match(-1, "unmatched");
    }

    private static int uniquePassIndex(List<? extends ActivatedAbility> abilities) {
        int pass = -1;
        for (int index = 0; index < abilities.size(); index++) {
            if (abilities.get(index) instanceof PassAbility) {
                if (pass >= 0) {
                    throw new IllegalArgumentException("priority menu contains multiple pass rows");
                }
                pass = index;
            }
        }
        if (pass < 0) {
            throw new IllegalArgumentException("priority menu lacks a pass row");
        }
        return pass;
    }

    private static String normalize(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT)
                .replace('\r', ' ').replace('\n', ' ').trim();
    }

    private static long elapsedMillis(long started) {
        return (System.nanoTime() - started) / 1_000_000L;
    }

    private void observe(String status, long elapsedMillis) {
        queried++;
        totalMillis = Math.addExact(totalMillis, Math.max(0L, elapsedMillis));
        if ("source_id".equals(status) || "text".equals(status)
                || "text_disambig".equals(status)) {
            matched++;
        } else if ("plan_pass".equals(status) || "step_pass".equals(status)) {
            passes++;
        } else if ("timeout_pass".equals(status)) {
            timeouts++;
        } else if ("source_ambig".equals(status)) {
            ambiguous++;
        } else {
            unmatched++;
        }
    }

    private void write(JsonObject record) throws IOException {
        writer.write(record.toString());
        writer.newLine();
        writer.flush();
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("counterfactual teacher is closed");
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        JsonObject summary = new JsonObject();
        summary.addProperty("schema", SCHEMA);
        summary.addProperty("record_type", "summary");
        summary.addProperty("record_ordinal", recordOrdinal++);
        summary.addProperty("selection_source", SELECTION_SOURCE);
        summary.addProperty("queries", queried);
        summary.addProperty("matched", matched);
        summary.addProperty("passes", passes);
        summary.addProperty("timeouts", timeouts);
        summary.addProperty("ambiguous", ambiguous);
        summary.addProperty("unmatched_or_error", unmatched);
        summary.addProperty("total_elapsed_ms", totalMillis);
        write(summary);
        closed = true;
        writer.close();
    }

    private static final class Match {
        private final int index;
        private final String status;

        private Match(int index, String status) {
            this.index = index;
            this.status = status;
        }
    }
}
