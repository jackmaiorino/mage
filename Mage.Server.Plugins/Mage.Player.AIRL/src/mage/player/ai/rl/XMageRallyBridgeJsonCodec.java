package mage.player.ai.rl;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/** Strict duplicate-free JSON codec matching the Rust shadow scorer V2. */
public final class XMageRallyBridgeJsonCodec {

    private static final Pattern CANONICAL_INTEGER = Pattern.compile("0|-?[1-9][0-9]*");
    private static final Pattern LOWER_HEX_16 = Pattern.compile("[0-9a-f]{16}");
    private static final Pattern LOWER_HEX_32 = Pattern.compile("[0-9a-f]{32}");
    private static final Pattern LOWER_HEX_64 = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern ERROR_CODE = Pattern.compile("[a-z][a-z0-9_]*");
    private static final int MAX_JSON_DEPTH = 32;
    private static final int MAX_CONTAINER_ELEMENTS = 65_536;

    private static final BigInteger ZERO = BigInteger.ZERO;
    private static final BigInteger U8_MAX = BigInteger.valueOf(255L);
    private static final BigInteger U16_MAX = BigInteger.valueOf(65_535L);
    private static final BigInteger U32_MAX = new BigInteger("4294967295");
    private static final BigInteger U64_MAX = new BigInteger("18446744073709551615");
    private static final BigInteger I32_MIN = BigInteger.valueOf(Integer.MIN_VALUE);
    private static final BigInteger I32_MAX = BigInteger.valueOf(Integer.MAX_VALUE);
    private static final BigInteger JAVA_LONG_MAX = BigInteger.valueOf(Long.MAX_VALUE);

    private static final Set<String> CHECKPOINT_FIELDS = set(
            "authority_kind", "source_run_sha256", "source_generation",
            "source_checkpoint_sha256", "source_sidecar_sha256", "source_payload_sha256",
            "source_train_state_sha256", "loaded_run_sha256", "loaded_generation",
            "loaded_checkpoint_sha256", "loaded_payload_sha256",
            "loaded_train_state_sha256", "model_parameter_sha256",
            "environment_trajectory_contract", "sampler_identity",
            "sampler_contract_sha256");

    private static final Set<String> DECISION_FIELDS = set(
            "deck_ids", "randomization_identity", "base_seed_u64_hex", "pair_index",
            "pair_environment_seed_u64_hex", "episode_id", "step",
            "environment_revision", "physical_decision_id", "substep_index",
            "substep_count", "acting_player", "decision_kind", "legal_action_count",
            "candidate_seat", "candidate_controls_current_actor",
            "actor_physical_decision_ordinal", "candidate_action_seed_u64_hex",
            "selected_action_index", "candidate_order_commitment_128_hex",
            "model_input_commitment", "model_input_sha256",
            "diagnostic_state_hash_u64_hex", "core_environment_hash_u64_hex",
            "logits_f32_bits", "value_f32_bits", "action_semantics", "kernel_clock");
    private static final Set<String> DECISION_ALLOWED_FIELDS = plus(
            DECISION_FIELDS, "initial_library_card_definition_ids");

    private static final Set<String> APPLIED_FIELDS = set(
            "episode_id", "step", "candidate_order_commitment_128_hex",
            "model_input_sha256", "selected_index", "selected_logit_f32_bits", "semantic");

    private static final Set<String> TERMINAL_BODY_FIELDS = set(
            "deck_ids", "randomization_identity", "base_seed_u64_hex", "pair_index",
            "pair_environment_seed_u64_hex", "terminal", "candidate_seat",
            "diagnostic_state_hash_u64_hex", "core_environment_hash_u64_hex");
    private static final Set<String> TERMINAL_BODY_ALLOWED_FIELDS = plus(
            TERMINAL_BODY_FIELDS, "initial_library_card_definition_ids");

    private static final Set<String> TERMINAL_FIELDS = set(
            "schema_version", "deck_ids", "deck_hashes", "episode_id",
            "terminal_outcome", "terminal_classification", "terminal_code", "winner",
            "terminal_reward", "terminal_reason", "policy_step_count",
            "physical_decision_count");

    private static final Set<String> KERNEL_CLOCK_FIELDS = set(
            "turn", "phase_step", "active_player", "priority_player", "stack_depth");
    private static final Set<String> EXPECTED_CLOCK_FIELDS = set(
            "turn", "phase_step", "active_player");

    public String encodeRequest(XMageRallyBridgeProtocol.Request request) {
        if (request == null) {
            throw new IllegalArgumentException("request must not be null");
        }
        StringWriter target = new StringWriter(160);
        try (JsonWriter writer = strictWriter(target)) {
            writer.beginObject();
            writer.name("request_type").value(request.getRequestType());
            writer.name("request_id").value(request.getRequestId());
            writer.name("episode_id").value(request.getEpisodeId());
            if (request instanceof XMageRallyBridgeProtocol.ResetRequest) {
                writer.name("base_seed").value(
                        ((XMageRallyBridgeProtocol.ResetRequest) request).getBaseSeed());
            } else if (request instanceof XMageRallyBridgeProtocol.ScoreCurrentRequest) {
                writer.name("expected_step").value(
                        ((XMageRallyBridgeProtocol.ScoreCurrentRequest) request).getExpectedStep());
            } else if (request instanceof XMageRallyBridgeProtocol.StepRequest) {
                XMageRallyBridgeProtocol.StepRequest step =
                        (XMageRallyBridgeProtocol.StepRequest) request;
                writer.name("expected_step").value(step.getExpectedStep());
                writer.name("selected_index").value(step.getSelectedIndex());
                XMageRallyBridgeProtocol.ExpectedClock expectedClock =
                        step.getExpectedClock();
                if (expectedClock != null) {
                    writer.name("expected_clock").beginObject();
                    writer.name("turn").value(expectedClock.getTurn());
                    writer.name("phase_step").value(expectedClock.getPhaseStep().wire());
                    writer.name("active_player").value(
                            expectedClock.getActivePlayer().wire());
                    writer.endObject();
                }
            } else {
                throw new IllegalArgumentException("unsupported request implementation");
            }
            writer.endObject();
        } catch (IOException e) {
            throw new IllegalStateException("unexpected in-memory JSON encoding failure", e);
        }
        return target.toString();
    }

    public XMageRallyBridgeProtocol.Request decodeRequest(String json) throws ProtocolException {
        JsonObject object = parseStrictObject(json, "request");
        String type = string(object, "request_type", "request");
        String requestId = string(object, "request_id", "request");
        long episodeId = signedLong(object, "episode_id", "request");
        try {
            switch (type) {
                case "reset":
                    exactFields(object, set("request_type", "request_id", "episode_id",
                            "base_seed"), "reset request");
                    return new XMageRallyBridgeProtocol.ResetRequest(
                            requestId, episodeId, signedLong(object, "base_seed", "request"));
                case "score_current":
                    exactFields(object, set("request_type", "request_id", "episode_id",
                            "expected_step"), "score_current request");
                    return new XMageRallyBridgeProtocol.ScoreCurrentRequest(
                            requestId, episodeId,
                            signedLong(object, "expected_step", "request"));
                case "step":
                    Set<String> requiredStepFields = set(
                            "request_type", "request_id", "episode_id",
                            "expected_step", "selected_index");
                    requiredAllowedFields(object, requiredStepFields,
                            plus(requiredStepFields, "expected_clock"), "step request");
                    return new XMageRallyBridgeProtocol.StepRequest(
                            requestId, episodeId,
                            signedLong(object, "expected_step", "request"),
                            javaInt(object, "selected_index", ZERO,
                                    BigInteger.valueOf(Integer.MAX_VALUE), "request"),
                            object.has("expected_clock")
                                    ? expectedClock(object(
                                    object, "expected_clock", "step request")) : null);
                default:
                    throw new ProtocolException("unsupported request_type: " + type);
            }
        } catch (IllegalArgumentException e) {
            throw new ProtocolException("invalid request: " + safeMessage(e), e);
        }
    }

    public XMageRallyBridgeProtocol.Response decodeResponse(String json)
            throws ProtocolException {
        JsonObject root = parseStrictObject(json, "response");
        String responseType = string(root, "response_type", "response");
        Set<String> exactRoot;
        switch (responseType) {
            case "decision":
                exactRoot = set("protocol", "schema_version", "request_id", "checkpoint",
                        "response_type", "decision", "applied_action");
                break;
            case "terminal":
                exactRoot = set("protocol", "schema_version", "request_id", "checkpoint",
                        "response_type", "terminal", "applied_action");
                break;
            case "error":
                exactRoot = set("protocol", "schema_version", "request_id", "checkpoint",
                        "response_type", "error_code", "message");
                break;
            default:
                throw new ProtocolException("unsupported response_type: " + responseType);
        }
        exactFields(root, exactRoot, "response");

        String protocol = string(root, "protocol", "response");
        int schemaVersion = javaInt(root, "schema_version", ZERO, I32_MAX, "response");
        if (!XMageRallyBridgeProtocol.PROTOCOL.equals(protocol)
                || schemaVersion != XMageRallyBridgeProtocol.SCHEMA_VERSION) {
            throw new ProtocolException("protocol or schema_version mismatch");
        }
        String requestId = nullableString(root, "request_id", "response");
        if (requestId != null) {
            try {
                XMageRallyBridgeProtocol.validateRequestId(requestId);
            } catch (IllegalArgumentException e) {
                throw new ProtocolException("invalid response request_id", e);
            }
        }
        XMageRallyBridgeProtocol.CheckpointIdentity checkpoint = checkpoint(
                object(root, "checkpoint", "response"));
        XMageRallyBridgeProtocol.ResponseBody body;
        switch (responseType) {
            case "decision":
                body = new XMageRallyBridgeProtocol.DecisionResponseBody(
                        decision(object(root, "decision", "response")),
                        nullableApplied(root.get("applied_action")));
                break;
            case "terminal":
                body = new XMageRallyBridgeProtocol.TerminalResponseBody(
                        terminalBody(object(root, "terminal", "response")),
                        nullableApplied(root.get("applied_action")));
                break;
            case "error":
                String code = string(root, "error_code", "response");
                if (!ERROR_CODE.matcher(code).matches()) {
                    throw new ProtocolException("invalid error_code");
                }
                body = new XMageRallyBridgeProtocol.ErrorResponseBody(
                        code, string(root, "message", "response"));
                break;
            default:
                throw new ProtocolException("unreachable response type");
        }
        return new XMageRallyBridgeProtocol.Response(
                protocol, schemaVersion, requestId, checkpoint, body);
    }

    private static XMageRallyBridgeProtocol.CheckpointIdentity checkpoint(JsonObject object)
            throws ProtocolException {
        exactFields(object, CHECKPOINT_FIELDS, "checkpoint");
        String sourceRun = hex(object, "source_run_sha256", 64, "checkpoint");
        String sourceCheckpoint = hex(object, "source_checkpoint_sha256", 64, "checkpoint");
        String sourceSidecar = hex(object, "source_sidecar_sha256", 64, "checkpoint");
        String sourcePayload = hex(object, "source_payload_sha256", 64, "checkpoint");
        String sourceTrain = hex(object, "source_train_state_sha256", 64, "checkpoint");
        String loadedRun = hex(object, "loaded_run_sha256", 64, "checkpoint");
        String loadedCheckpoint = hex(object, "loaded_checkpoint_sha256", 64, "checkpoint");
        String loadedPayload = hex(object, "loaded_payload_sha256", 64, "checkpoint");
        String loadedTrain = hex(object, "loaded_train_state_sha256", 64, "checkpoint");
        String model = hex(object, "model_parameter_sha256", 64, "checkpoint");
        String samplerContract = hex(object, "sampler_contract_sha256", 64, "checkpoint");
        return new XMageRallyBridgeProtocol.CheckpointIdentity(
                string(object, "authority_kind", "checkpoint"),
                sourceRun,
                signedLong(object, "source_generation", "checkpoint"),
                sourceCheckpoint,
                sourceSidecar,
                sourcePayload,
                sourceTrain,
                loadedRun,
                signedLong(object, "loaded_generation", "checkpoint"),
                loadedCheckpoint,
                loadedPayload,
                loadedTrain,
                model,
                string(object, "environment_trajectory_contract", "checkpoint"),
                string(object, "sampler_identity", "checkpoint"),
                samplerContract);
    }

    private static XMageRallyBridgeProtocol.DecisionBody decision(JsonObject object)
            throws ProtocolException {
        requiredAllowedFields(object, DECISION_FIELDS, DECISION_ALLOWED_FIELDS, "decision");
        List<String> deckIds = twoStrings(object, "deck_ids", "decision");
        String randomization = string(object, "randomization_identity", "decision");
        String baseSeed = hex(object, "base_seed_u64_hex", 16, "decision");
        long pairIndex = signedLong(object, "pair_index", "decision");
        String pairEnvironment = hex(
                object, "pair_environment_seed_u64_hex", 16, "decision");
        List<List<Integer>> initialLibraries = object.has(
                "initial_library_card_definition_ids")
                ? initialLibraries(object.get("initial_library_card_definition_ids")) : null;
        long episodeId = signedLong(object, "episode_id", "decision");
        long step = signedLong(object, "step", "decision");
        long environmentRevision = signedLong(object, "environment_revision", "decision");
        long physicalDecision = signedLong(object, "physical_decision_id", "decision");
        int substepIndex = javaInt(object, "substep_index", ZERO,
                BigInteger.valueOf(Integer.MAX_VALUE), "decision");
        int substepCount = javaInt(object, "substep_count", BigInteger.ONE,
                BigInteger.valueOf(Integer.MAX_VALUE), "decision");
        if (substepIndex >= substepCount) {
            throw new ProtocolException("decision substep_index must be below substep_count");
        }
        XMageRallyBridgeProtocol.Seat acting = seat(object, "acting_player", "decision");
        String decisionKind = oneOf(string(object, "decision_kind", "decision"),
                "decision_kind", "surface", "attacker_inclusion", "blocker_inclusion");
        int legalCount = javaInt(object, "legal_action_count", BigInteger.ONE,
                BigInteger.valueOf(XMageRallyBridgeProtocol.MAX_ACTIONS), "decision");
        XMageRallyBridgeProtocol.Seat candidate = seat(object, "candidate_seat", "decision");
        boolean controls = bool(object, "candidate_controls_current_actor", "decision");
        if (controls != (acting == candidate)) {
            throw new ProtocolException("candidate_controls_current_actor is inconsistent");
        }
        long actorOrdinal = signedLong(
                object, "actor_physical_decision_ordinal", "decision");
        String actionSeed = nullableHex(object, "candidate_action_seed_u64_hex", 16,
                "decision");
        Integer selected = nullableJavaInt(object, "selected_action_index", ZERO,
                BigInteger.valueOf(Integer.MAX_VALUE), "decision");
        String candidateOrder = hex(
                object, "candidate_order_commitment_128_hex", 32, "decision");
        String modelCommitment = string(object, "model_input_commitment", "decision");
        String modelSha = hex(object, "model_input_sha256", 64, "decision");
        String diagnosticHash = hex(
                object, "diagnostic_state_hash_u64_hex", 16, "decision");
        String environmentHash = hex(
                object, "core_environment_hash_u64_hex", 16, "decision");
        List<Long> logits = u32List(object, "logits_f32_bits", "decision");
        long valueBits = u32(object, "value_f32_bits", "decision");
        List<XMageRallyBridgeProtocol.ActionSemantic> semantics = semantics(
                array(object, "action_semantics", "decision"), acting);
        XMageRallyBridgeProtocol.KernelClock kernelClock = kernelClock(
                object(object, "kernel_clock", "decision"));
        if (legalCount != logits.size() || legalCount != semantics.size()) {
            throw new ProtocolException(
                    "legal_action_count does not match logits/action_semantics width");
        }
        if (controls) {
            if (actionSeed == null || selected == null || selected >= legalCount) {
                throw new ProtocolException("candidate-controlled decision lacks valid selection");
            }
        } else if (actionSeed != null || selected != null) {
            throw new ProtocolException("opponent decision contains candidate selection fields");
        }
        if (!XMageRallyBridgeProtocol.RANDOMIZATION_IDENTITY.equals(randomization)) {
            throw new ProtocolException("randomization_identity mismatch");
        }
        if (!XMageRallyBridgeProtocol.MODEL_INPUT_COMMITMENT.equals(modelCommitment)) {
            throw new ProtocolException("model_input_commitment mismatch");
        }
        return new XMageRallyBridgeProtocol.DecisionBody(
                deckIds, randomization, baseSeed, pairIndex, pairEnvironment, initialLibraries,
                episodeId, step, environmentRevision, physicalDecision,
                substepIndex, substepCount, acting, decisionKind, legalCount, candidate,
                controls, actorOrdinal, actionSeed, selected, candidateOrder, modelCommitment,
                modelSha, diagnosticHash, environmentHash, logits, valueBits, semantics,
                kernelClock);
    }

    private static XMageRallyBridgeProtocol.KernelClock kernelClock(JsonObject object)
            throws ProtocolException {
        exactFields(object, KERNEL_CLOCK_FIELDS, "kernel_clock");
        return new XMageRallyBridgeProtocol.KernelClock(
                u32(object, "turn", "kernel_clock"),
                kernelPhaseStep(object, "phase_step", "kernel_clock"),
                seat(object, "active_player", "kernel_clock"),
                seat(object, "priority_player", "kernel_clock"),
                u32(object, "stack_depth", "kernel_clock"));
    }

    private static XMageRallyBridgeProtocol.ExpectedClock expectedClock(JsonObject object)
            throws ProtocolException {
        exactFields(object, EXPECTED_CLOCK_FIELDS, "expected_clock");
        return new XMageRallyBridgeProtocol.ExpectedClock(
                u32(object, "turn", "expected_clock"),
                kernelPhaseStep(object, "phase_step", "expected_clock"),
                seat(object, "active_player", "expected_clock"));
    }

    private static XMageRallyBridgeProtocol.KernelPhaseStep kernelPhaseStep(
            JsonObject object, String field, String context) throws ProtocolException {
        String value = string(object, field, context);
        try {
            return XMageRallyBridgeProtocol.KernelPhaseStep.fromWire(value);
        } catch (IllegalArgumentException error) {
            throw new ProtocolException(context + " has invalid " + field + ": " + value,
                    error);
        }
    }

    private static XMageRallyBridgeProtocol.AppliedAction nullableApplied(JsonElement value)
            throws ProtocolException {
        if (value == null) {
            throw new ProtocolException("missing applied_action");
        }
        if (value.isJsonNull()) {
            return null;
        }
        if (!value.isJsonObject()) {
            throw new ProtocolException("applied_action must be object or null");
        }
        JsonObject object = value.getAsJsonObject();
        exactFields(object, APPLIED_FIELDS, "applied_action");
        return new XMageRallyBridgeProtocol.AppliedAction(
                signedLong(object, "episode_id", "applied_action"),
                signedLong(object, "step", "applied_action"),
                hex(object, "candidate_order_commitment_128_hex", 32, "applied_action"),
                hex(object, "model_input_sha256", 64, "applied_action"),
                javaInt(object, "selected_index", ZERO,
                        BigInteger.valueOf(Integer.MAX_VALUE), "applied_action"),
                u32(object, "selected_logit_f32_bits", "applied_action"),
                actionSemantic(object(object, "semantic", "applied_action"), null));
    }

    private static XMageRallyBridgeProtocol.TerminalBody terminalBody(JsonObject object)
            throws ProtocolException {
        requiredAllowedFields(object, TERMINAL_BODY_FIELDS,
                TERMINAL_BODY_ALLOWED_FIELDS, "terminal body");
        List<String> deckIds = twoStrings(object, "deck_ids", "terminal body");
        String randomization = string(object, "randomization_identity", "terminal body");
        if (!XMageRallyBridgeProtocol.RANDOMIZATION_IDENTITY.equals(randomization)) {
            throw new ProtocolException("terminal randomization_identity mismatch");
        }
        List<List<Integer>> initialLibraries = object.has(
                "initial_library_card_definition_ids")
                ? initialLibraries(object.get("initial_library_card_definition_ids")) : null;
        XMageRallyBridgeProtocol.TerminalRecord terminal = terminalRecord(
                object(object, "terminal", "terminal body"));
        if (!deckIds.equals(terminal.getDeckIds())) {
            throw new ProtocolException("terminal deck_ids disagree with terminal record");
        }
        return new XMageRallyBridgeProtocol.TerminalBody(
                deckIds,
                randomization,
                hex(object, "base_seed_u64_hex", 16, "terminal body"),
                signedLong(object, "pair_index", "terminal body"),
                hex(object, "pair_environment_seed_u64_hex", 16, "terminal body"),
                initialLibraries,
                terminal,
                seat(object, "candidate_seat", "terminal body"),
                hex(object, "diagnostic_state_hash_u64_hex", 16, "terminal body"),
                hex(object, "core_environment_hash_u64_hex", 16, "terminal body"));
    }

    private static XMageRallyBridgeProtocol.TerminalRecord terminalRecord(JsonObject object)
            throws ProtocolException {
        exactFields(object, TERMINAL_FIELDS, "terminal record");
        int schema = javaInt(object, "schema_version", ZERO, I32_MAX, "terminal record");
        if (schema != 5) {
            throw new ProtocolException("terminal schema_version must be 5");
        }
        List<String> deckIds = twoStrings(object, "deck_ids", "terminal record");
        List<String> deckHashes = u64DecimalList(
                object, "deck_hashes", 2, "terminal record");
        long episodeId = signedLong(object, "episode_id", "terminal record");
        String outcome = oneOf(string(object, "terminal_outcome", "terminal record"),
                "terminal_outcome", "p0_win", "p1_win", "draw", "truncated", "halted");
        String classification = oneOf(
                string(object, "terminal_classification", "terminal record"),
                "terminal_classification", "natural", "truncated", "halted");
        String code = oneOf(string(object, "terminal_code", "terminal record"),
                "terminal_code", "natural_game_over", "decision_cap", "fail_closed");
        XMageRallyBridgeProtocol.Seat winner = nullableSeat(
                object, "winner", "terminal record");
        List<Integer> reward = i32List(object, "terminal_reward", 2, "terminal record");
        String reason = string(object, "terminal_reason", "terminal record");
        validateTerminalClassification(outcome, classification, code, winner);
        return new XMageRallyBridgeProtocol.TerminalRecord(
                schema, deckIds, deckHashes, episodeId, outcome, classification, code,
                winner, reward, reason,
                signedLong(object, "policy_step_count", "terminal record"),
                signedLong(object, "physical_decision_count", "terminal record"));
    }

    private static void validateTerminalClassification(String outcome,
                                                       String classification,
                                                       String code,
                                                       XMageRallyBridgeProtocol.Seat winner)
            throws ProtocolException {
        if (("p0_win".equals(outcome) && winner != XMageRallyBridgeProtocol.Seat.P0)
                || ("p1_win".equals(outcome) && winner != XMageRallyBridgeProtocol.Seat.P1)
                || (("draw".equals(outcome) || "truncated".equals(outcome)
                || "halted".equals(outcome)) && winner != null)) {
            throw new ProtocolException("terminal winner is inconsistent with outcome");
        }
        boolean valid = ("natural".equals(classification)
                && "natural_game_over".equals(code)
                && ("p0_win".equals(outcome) || "p1_win".equals(outcome)
                || "draw".equals(outcome)))
                || ("truncated".equals(classification) && "decision_cap".equals(code)
                && "truncated".equals(outcome))
                || ("halted".equals(classification) && "fail_closed".equals(code)
                && "halted".equals(outcome));
        if (!valid) {
            throw new ProtocolException("terminal outcome/classification/code mismatch");
        }
    }

    private static List<XMageRallyBridgeProtocol.ActionSemantic> semantics(
            JsonArray array, XMageRallyBridgeProtocol.Seat expectedActor)
            throws ProtocolException {
        if (array.size() > XMageRallyBridgeProtocol.MAX_ACTIONS) {
            throw new ProtocolException("action_semantics exceeds limit");
        }
        ArrayList<XMageRallyBridgeProtocol.ActionSemantic> result =
                new ArrayList<>(array.size());
        for (JsonElement element : array) {
            if (!element.isJsonObject()) {
                throw new ProtocolException("action_semantics entries must be objects");
            }
            result.add(actionSemantic(element.getAsJsonObject(), expectedActor));
        }
        return result;
    }

    private static XMageRallyBridgeProtocol.ActionSemantic actionSemantic(
            JsonObject object, XMageRallyBridgeProtocol.Seat expectedActor)
            throws ProtocolException {
        String kind = string(object, "action_kind", "action semantic");
        switch (kind) {
            case "pass":
                actionFields(object, kind, "actor");
                actor(object, expectedActor);
                break;
            case "play_land":
            case "cast_spell":
            case "plot_spell":
                actionFields(object, kind, "actor", "source");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                break;
            case "activate_mana_ability":
                actionFields(object, kind, "actor", "source", "mana_choice");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                nullableOneOf(object, "mana_choice", kind, "W", "U", "B", "R", "G", "C");
                break;
            case "activate_ability":
                actionFields(object, kind, "actor", "source", "ability_index");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "ability_index", ZERO, U8_MAX, kind);
                break;
            case "choose_target":
                actionFields(object, kind, "actor", "source", "remaining", "target");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "remaining", ZERO, U8_MAX, kind);
                targetRef(object(object, "target", kind), kind + ".target");
                break;
            case "choose_cost_target":
                actionFields(object, kind, "actor", "source", "cost_kind", "remaining",
                        "candidate");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                oneOf(string(object, "cost_kind", kind), "cost_kind",
                        "SacrificeLands", "SacrificePermanents", "SacrificeCreatures",
                        "SacrificeArtifacts", "DiscardCards", "ExileFromGraveyard",
                        "TapPermanents", "ReturnPermanentsToHand", "PayLife",
                        "RemoveCounters", "PutCounters");
                integer(object, "remaining", ZERO, U8_MAX, kind);
                stableRef(object(object, "candidate", kind), kind + ".candidate");
                break;
            case "choose_cast_mode":
                actionFields(object, kind, "actor", "source", "mode");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                oneOf(string(object, "mode", kind), "mode", "Normal", "Alternative");
                break;
            case "choose_kicker":
                sourceBooleanAction(object, kind, expectedActor, "pay");
                break;
            case "choose_spell_mode":
                actionFields(object, kind, "actor", "source", "mode_index", "mode_count");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "mode_index", ZERO, U8_MAX, kind);
                integer(object, "mode_count", ZERO, U8_MAX, kind);
                break;
            case "choose_effect_option":
                actionFields(object, kind, "actor", "source", "option_index", "option_count");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "option_index", ZERO, U16_MAX, kind);
                integer(object, "option_count", ZERO, U16_MAX, kind);
                break;
            case "choose_effect_target":
                actionFields(object, kind, "actor", "source", "target", "selected_count",
                        "min_targets", "max_targets");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                targetRef(object(object, "target", kind), kind + ".target");
                integer(object, "selected_count", ZERO, U16_MAX, kind);
                integer(object, "min_targets", ZERO, U16_MAX, kind);
                integer(object, "max_targets", ZERO, U16_MAX, kind);
                break;
            case "finish_effect_selection":
            case "finish_target_selection":
                actionFields(object, kind, "actor", "source", "selected_count");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "selected_count", ZERO, U16_MAX, kind);
                break;
            case "choose_effect_color":
                actionFields(object, kind, "actor", "source", "color");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                oneOf(string(object, "color", kind), "color", "W", "U", "B", "R", "G", "C");
                break;
            case "choose_effect_number":
                actionFields(object, kind, "actor", "source", "number", "minimum", "maximum");
                actor(object, expectedActor);
                stableRef(object(object, "source", kind), kind + ".source");
                integer(object, "number", I32_MIN, I32_MAX, kind);
                integer(object, "minimum", I32_MIN, I32_MAX, kind);
                integer(object, "maximum", I32_MIN, I32_MAX, kind);
                break;
            case "choose_effect_boolean":
                sourceBooleanAction(object, kind, expectedActor, "value");
                break;
            case "choose_optional_cost_use":
                actionFields(object, kind, "actor", "use_cost");
                actor(object, expectedActor);
                bool(object, "use_cost", kind);
                break;
            case "choose_optional_cost_which":
                actionFields(object, kind, "actor", "choice");
                actor(object, expectedActor);
                oneOf(string(object, "choice", kind), "choice",
                        "Decline", "Discard", "SacrificeLand");
                break;
            case "choose_spell_copy_payment":
                sourceBooleanAction(object, kind, expectedActor, "pay");
                break;
            case "choose_spell_copy_retarget":
                sourceBooleanAction(object, kind, expectedActor, "change_target");
                break;
            case "choose_madness_cast":
                actionFields(object, kind, "actor", "card", "cast_it");
                actor(object, expectedActor);
                stableRef(object(object, "card", kind), kind + ".card");
                bool(object, "cast_it", kind);
                break;
            case "discard":
                actionFields(object, kind, "actor", "cards");
                actor(object, expectedActor);
                stableRefList(array(object, "cards", kind), kind + ".cards");
                break;
            case "declare_attackers":
                actionFields(object, kind, "actor", "attackers");
                actor(object, expectedActor);
                stableRefList(array(object, "attackers", kind), kind + ".attackers");
                break;
            case "declare_blockers_for_attacker":
                actionFields(object, kind, "actor", "attacker", "blockers");
                actor(object, expectedActor);
                stableRef(object(object, "attacker", kind), kind + ".attacker");
                stableRefList(array(object, "blockers", kind), kind + ".blockers");
                break;
            case "choose_attacker_inclusion":
                actionFields(object, kind, "actor", "attacker", "include");
                actor(object, expectedActor);
                stableRef(object(object, "attacker", kind), kind + ".attacker");
                bool(object, "include", kind);
                break;
            case "choose_blocker_inclusion":
                actionFields(object, kind, "actor", "attacker", "blocker", "include");
                actor(object, expectedActor);
                stableRef(object(object, "attacker", kind), kind + ".attacker");
                stableRef(object(object, "blocker", kind), kind + ".blocker");
                bool(object, "include", kind);
                break;
            case "order_triggers":
                actionFields(object, kind, "actor", "pending_sources", "order");
                actor(object, expectedActor);
                stableRefList(array(object, "pending_sources", kind), kind + ".pending_sources");
                unsignedIntegerList(array(object, "order", kind), kind + ".order");
                break;
            case "ambiguous":
                throw new ProtocolException("ambiguous action semantic is fail-closed");
            default:
                throw new ProtocolException("unsupported action_kind: " + kind);
        }
        return new XMageRallyBridgeProtocol.ActionSemantic(kind, object.toString());
    }

    private static void sourceBooleanAction(JsonObject object,
                                            String kind,
                                            XMageRallyBridgeProtocol.Seat expectedActor,
                                            String booleanField) throws ProtocolException {
        actionFields(object, kind, "actor", "source", booleanField);
        actor(object, expectedActor);
        stableRef(object(object, "source", kind), kind + ".source");
        bool(object, booleanField, kind);
    }

    private static void actor(JsonObject object, XMageRallyBridgeProtocol.Seat expected)
            throws ProtocolException {
        XMageRallyBridgeProtocol.Seat actual = seat(object, "actor", "action semantic");
        if (expected != null && expected != actual) {
            throw new ProtocolException("action semantic actor does not match decision actor");
        }
    }

    private static void actionFields(JsonObject object, String kind, String... fields)
            throws ProtocolException {
        LinkedHashSet<String> expected = new LinkedHashSet<>();
        expected.add("action_kind");
        expected.addAll(Arrays.asList(fields));
        exactFields(object, expected, kind + " action semantic");
    }

    private static void stableRef(JsonObject object, String context) throws ProtocolException {
        exactFields(object, set("arena_id", "card_db_id", "owner", "controller", "zone",
                "zone_change_count"), context);
        integer(object, "arena_id", ZERO, U32_MAX, context);
        integer(object, "card_db_id", ZERO, U16_MAX, context);
        seat(object, "owner", context);
        seat(object, "controller", context);
        oneOf(string(object, "zone", context), "zone", "Library", "Hand", "Battlefield",
                "Graveyard", "Stack", "Exile", "Command");
        integer(object, "zone_change_count", ZERO, U32_MAX, context);
    }

    private static void targetRef(JsonObject object, String context) throws ProtocolException {
        String kind = string(object, "target_kind", context);
        if ("player".equals(kind)) {
            exactFields(object, set("target_kind", "player"), context);
            seat(object, "player", context);
        } else if ("object".equals(kind)) {
            exactFields(object, set("target_kind", "object"), context);
            stableRef(object(object, "object", context), context + ".object");
        } else {
            throw new ProtocolException("unsupported target_kind: " + kind);
        }
    }

    private static void stableRefList(JsonArray array, String context) throws ProtocolException {
        for (JsonElement element : array) {
            if (!element.isJsonObject()) {
                throw new ProtocolException(context + " entries must be objects");
            }
            stableRef(element.getAsJsonObject(), context + "[]");
        }
    }

    private static void unsignedIntegerList(JsonArray array, String context)
            throws ProtocolException {
        for (JsonElement element : array) {
            integer(element, ZERO, JAVA_LONG_MAX, context + "[]");
        }
    }

    private static List<List<Integer>> initialLibraries(JsonElement value)
            throws ProtocolException {
        if (!value.isJsonArray() || value.getAsJsonArray().size() != 2) {
            throw new ProtocolException("initial library orders must contain exactly two rows");
        }
        ArrayList<List<Integer>> result = new ArrayList<>(2);
        for (JsonElement rowElement : value.getAsJsonArray()) {
            if (!rowElement.isJsonArray() || rowElement.getAsJsonArray().size() != 60) {
                throw new ProtocolException("each initial Rally library must contain 60 cards");
            }
            ArrayList<Integer> row = new ArrayList<>(60);
            for (JsonElement card : rowElement.getAsJsonArray()) {
                row.add(integer(card, ZERO, U16_MAX, "initial library card" ).intValue());
            }
            result.add(row);
        }
        return result;
    }

    private static List<String> twoStrings(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonArray array = array(object, field, context);
        if (array.size() != 2) {
            throw new ProtocolException(field + " must contain exactly two entries");
        }
        ArrayList<String> result = new ArrayList<>(2);
        for (JsonElement element : array) {
            if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isString()) {
                throw new ProtocolException(field + " entries must be strings");
            }
            result.add(element.getAsString());
        }
        return result;
    }

    private static List<Long> u32List(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonArray array = array(object, field, context);
        if (array.size() > XMageRallyBridgeProtocol.MAX_ACTIONS) {
            throw new ProtocolException(field + " exceeds action limit");
        }
        ArrayList<Long> result = new ArrayList<>(array.size());
        for (JsonElement element : array) {
            result.add(integer(element, ZERO, U32_MAX, field + "[]").longValue());
        }
        return result;
    }

    private static List<String> u64DecimalList(JsonObject object,
                                               String field,
                                               int exactSize,
                                               String context) throws ProtocolException {
        JsonArray array = array(object, field, context);
        if (array.size() != exactSize) {
            throw new ProtocolException(field + " has wrong length");
        }
        ArrayList<String> result = new ArrayList<>(array.size());
        for (JsonElement element : array) {
            result.add(integer(element, ZERO, U64_MAX, field + "[]").toString());
        }
        return result;
    }

    private static List<Integer> i32List(JsonObject object,
                                         String field,
                                         int exactSize,
                                         String context) throws ProtocolException {
        JsonArray array = array(object, field, context);
        if (array.size() != exactSize) {
            throw new ProtocolException(field + " has wrong length");
        }
        ArrayList<Integer> result = new ArrayList<>(array.size());
        for (JsonElement element : array) {
            result.add(integer(element, I32_MIN, I32_MAX, field + "[]").intValue());
        }
        return result;
    }

    private static long u32(JsonObject object, String field, String context)
            throws ProtocolException {
        return integer(object, field, ZERO, U32_MAX, context).longValue();
    }

    private static long signedLong(JsonObject object, String field, String context)
            throws ProtocolException {
        return integer(object, field, ZERO, JAVA_LONG_MAX, context).longValue();
    }

    private static int javaInt(JsonObject object,
                               String field,
                               BigInteger minimum,
                               BigInteger maximum,
                               String context) throws ProtocolException {
        return integer(object, field, minimum, maximum, context).intValue();
    }

    private static Integer nullableJavaInt(JsonObject object,
                                           String field,
                                           BigInteger minimum,
                                           BigInteger maximum,
                                           String context) throws ProtocolException {
        JsonElement value = required(object, field, context);
        return value.isJsonNull() ? null : integer(value, minimum, maximum, context + "." + field)
                .intValue();
    }

    private static BigInteger integer(JsonObject object,
                                      String field,
                                      BigInteger minimum,
                                      BigInteger maximum,
                                      String context) throws ProtocolException {
        return integer(required(object, field, context), minimum, maximum,
                context + "." + field);
    }

    private static BigInteger integer(JsonElement element,
                                      BigInteger minimum,
                                      BigInteger maximum,
                                      String context) throws ProtocolException {
        if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber()) {
            throw new ProtocolException(context + " must be an integer JSON number");
        }
        String lexical = element.getAsString();
        if (!CANONICAL_INTEGER.matcher(lexical).matches()) {
            throw new ProtocolException(context + " uses noncanonical integer syntax");
        }
        BigInteger value;
        try {
            value = new BigInteger(lexical);
        } catch (NumberFormatException e) {
            throw new ProtocolException(context + " is not an integer", e);
        }
        if (value.compareTo(minimum) < 0 || value.compareTo(maximum) > 0) {
            throw new ProtocolException(context + " is outside its integer domain");
        }
        return value;
    }

    private static String hex(JsonObject object, String field, int length, String context)
            throws ProtocolException {
        String value = string(object, field, context);
        Pattern pattern = length == 16 ? LOWER_HEX_16 : length == 32
                ? LOWER_HEX_32 : LOWER_HEX_64;
        if (!pattern.matcher(value).matches()) {
            throw new ProtocolException(context + "." + field + " must be lower hex length "
                    + length);
        }
        return value;
    }

    private static String nullableHex(JsonObject object,
                                      String field,
                                      int length,
                                      String context) throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (value.isJsonNull()) {
            return null;
        }
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new ProtocolException(context + "." + field + " must be string or null");
        }
        String lexical = value.getAsString();
        Pattern pattern = length == 16 ? LOWER_HEX_16 : length == 32
                ? LOWER_HEX_32 : LOWER_HEX_64;
        if (!pattern.matcher(lexical).matches()) {
            throw new ProtocolException(context + "." + field + " has invalid hex value");
        }
        return lexical;
    }

    private static XMageRallyBridgeProtocol.Seat seat(JsonObject object,
                                                       String field,
                                                       String context)
            throws ProtocolException {
        try {
            return XMageRallyBridgeProtocol.Seat.fromWire(string(object, field, context));
        } catch (IllegalArgumentException e) {
            throw new ProtocolException(context + "." + field + " is not a seat", e);
        }
    }

    private static XMageRallyBridgeProtocol.Seat nullableSeat(JsonObject object,
                                                               String field,
                                                               String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (value.isJsonNull()) {
            return null;
        }
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new ProtocolException(context + "." + field + " must be seat or null");
        }
        try {
            return XMageRallyBridgeProtocol.Seat.fromWire(value.getAsString());
        } catch (IllegalArgumentException e) {
            throw new ProtocolException(context + "." + field + " is not a seat", e);
        }
    }

    private static boolean bool(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) {
            throw new ProtocolException(context + "." + field + " must be boolean");
        }
        return value.getAsBoolean();
    }

    private static String string(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new ProtocolException(context + "." + field + " must be string");
        }
        return value.getAsString();
    }

    private static String nullableString(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (value.isJsonNull()) {
            return null;
        }
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new ProtocolException(context + "." + field + " must be string or null");
        }
        return value.getAsString();
    }

    private static JsonObject object(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (!value.isJsonObject()) {
            throw new ProtocolException(context + "." + field + " must be object");
        }
        return value.getAsJsonObject();
    }

    private static JsonArray array(JsonObject object, String field, String context)
            throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (!value.isJsonArray()) {
            throw new ProtocolException(context + "." + field + " must be array");
        }
        return value.getAsJsonArray();
    }

    private static JsonElement required(JsonObject object, String field, String context)
            throws ProtocolException {
        if (!object.has(field)) {
            throw new ProtocolException(context + " missing field " + field);
        }
        return object.get(field);
    }

    private static String oneOf(String value, String field, String... allowed)
            throws ProtocolException {
        for (String candidate : allowed) {
            if (candidate.equals(value)) {
                return value;
            }
        }
        throw new ProtocolException("unsupported " + field + ": " + value);
    }

    private static void nullableOneOf(JsonObject object,
                                      String field,
                                      String context,
                                      String... allowed) throws ProtocolException {
        JsonElement value = required(object, field, context);
        if (!value.isJsonNull()) {
            if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
                throw new ProtocolException(context + "." + field + " must be string or null");
            }
            oneOf(value.getAsString(), field, allowed);
        }
    }

    private static void exactFields(JsonObject object, Set<String> exact, String context)
            throws ProtocolException {
        requiredAllowedFields(object, exact, exact, context);
    }

    private static void requiredAllowedFields(JsonObject object,
                                              Set<String> required,
                                              Set<String> allowed,
                                              String context) throws ProtocolException {
        HashSet<String> actual = new HashSet<>();
        for (Map.Entry<String, JsonElement> entry : object.entrySet()) {
            actual.add(entry.getKey());
            if (!allowed.contains(entry.getKey())) {
                throw new ProtocolException(context + " has unknown field " + entry.getKey());
            }
        }
        LinkedHashSet<String> missing = new LinkedHashSet<>(required);
        missing.removeAll(actual);
        if (!missing.isEmpty()) {
            throw new ProtocolException(context + " missing fields " + missing);
        }
    }

    private static JsonObject parseStrictObject(String json, String context)
            throws ProtocolException {
        if (json == null || json.isEmpty()) {
            throw new ProtocolException(context + " JSON must not be empty");
        }
        if (json.charAt(0) == '\ufeff') {
            throw new ProtocolException("UTF-8 BOM is forbidden");
        }
        if (json.indexOf('\n') >= 0 || json.indexOf('\r') >= 0) {
            throw new ProtocolException("embedded line terminators are forbidden");
        }
        try (JsonReader reader = new JsonReader(new StringReader(json))) {
            reader.setStrictness(Strictness.STRICT);
            JsonElement value = readStrictValue(reader, 0);
            if (reader.peek() != JsonToken.END_DOCUMENT) {
                throw new ProtocolException("trailing JSON content is forbidden");
            }
            if (!value.isJsonObject()) {
                throw new ProtocolException(context + " must be a JSON object");
            }
            return value.getAsJsonObject();
        } catch (ProtocolException e) {
            throw e;
        } catch (IOException | IllegalStateException | NumberFormatException e) {
            throw new ProtocolException("invalid " + context + " JSON: " + safeMessage(e), e);
        }
    }

    private static JsonElement readStrictValue(JsonReader reader, int depth)
            throws IOException {
        if (depth > MAX_JSON_DEPTH) {
            throw new ProtocolException("JSON nesting exceeds fixed limit");
        }
        switch (reader.peek()) {
            case BEGIN_OBJECT:
                JsonObject object = new JsonObject();
                HashSet<String> names = new HashSet<>();
                reader.beginObject();
                int fields = 0;
                while (reader.hasNext()) {
                    if (++fields > MAX_CONTAINER_ELEMENTS) {
                        throw new ProtocolException("JSON object exceeds element limit");
                    }
                    String name = reader.nextName();
                    if (!names.add(name)) {
                        throw new ProtocolException("duplicate JSON field: " + name);
                    }
                    object.add(name, readStrictValue(reader, depth + 1));
                }
                reader.endObject();
                return object;
            case BEGIN_ARRAY:
                JsonArray array = new JsonArray();
                reader.beginArray();
                while (reader.hasNext()) {
                    if (array.size() == MAX_CONTAINER_ELEMENTS) {
                        throw new ProtocolException("JSON array exceeds element limit");
                    }
                    array.add(readStrictValue(reader, depth + 1));
                }
                reader.endArray();
                return array;
            case STRING:
                return new JsonPrimitive(reader.nextString());
            case NUMBER:
                String lexical = reader.nextString();
                if (!CANONICAL_INTEGER.matcher(lexical).matches()) {
                    throw new ProtocolException("all protocol numbers must be canonical integers");
                }
                return new JsonPrimitive(new BigInteger(lexical));
            case BOOLEAN:
                return new JsonPrimitive(reader.nextBoolean());
            case NULL:
                reader.nextNull();
                return JsonNull.INSTANCE;
            default:
                throw new ProtocolException("unexpected JSON token " + reader.peek());
        }
    }

    private static JsonWriter strictWriter(StringWriter target) {
        JsonWriter writer = new JsonWriter(target);
        writer.setStrictness(Strictness.STRICT);
        return writer;
    }

    private static Set<String> set(String... values) {
        return java.util.Collections.unmodifiableSet(
                new LinkedHashSet<>(Arrays.asList(values)));
    }

    private static Set<String> plus(Set<String> values, String extra) {
        LinkedHashSet<String> result = new LinkedHashSet<>(values);
        result.add(extra);
        return java.util.Collections.unmodifiableSet(result);
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName() : message;
    }

    public static final class ProtocolException extends IOException {
        public ProtocolException(String message) {
            super(message);
        }

        public ProtocolException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
