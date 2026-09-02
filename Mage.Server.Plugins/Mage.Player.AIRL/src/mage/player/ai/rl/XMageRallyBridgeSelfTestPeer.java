package mage.player.ai.rl;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/** Pure-JDK strict child process used by the bridge standalone self-test. */
public final class XMageRallyBridgeSelfTestPeer {

    private static final XMageRallyBridgeJsonCodec CODEC = new XMageRallyBridgeJsonCodec();
    private static final long EPISODE = 2L;
    private static final long BASE_SEED = 71_501L;
    private static final String BASE_SEED_HEX = "000000000001174d";
    private static final String PAIR_ENV_HEX = "6683b059f7fd4a71";
    private static final String COMMIT_0 = "11111111111111111111111111111111";
    private static final String MODEL_0 =
            "2222222222222222222222222222222222222222222222222222222222222222";
    private static final String COMMIT_1 = "66666666666666666666666666666666";
    private static final String MODEL_1 =
            "7777777777777777777777777777777777777777777777777777777777777777";
    private static final String LAND_P0 =
            "{\"action_kind\":\"play_land\",\"actor\":\"p0\",\"source\":{"
                    + "\"arena_id\":10,\"card_db_id\":20,\"owner\":\"p0\","
                    + "\"controller\":\"p0\",\"zone\":\"Hand\","
                    + "\"zone_change_count\":0}}";

    private XMageRallyBridgeSelfTestPeer() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("expected one peer mode");
            System.exit(2);
        }
        switch (args[0]) {
            case "flow":
                serveFlow();
                return;
            case "multi_episode":
                serveMultiEpisode();
                return;
            case "wrong_checkpoint":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, true));
                return;
            case "echo_mismatch":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("wrong", true, false, false));
                return;
            case "transition_mismatch":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false));
                requireStep(readRequest(), 0L, 1);
                String response = decisionResponse("step-0", false, true, false)
                        .replace(COMMIT_0, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                writeLf(System.out, response);
                return;
            case "malformed_unknown":
                requireReset(readRequest());
                String malformed = decisionResponse("reset", true, false, false);
                malformed = malformed.substring(0, malformed.length() - 1) + ",\"unknown\":0}";
                writeLf(System.out, malformed);
                return;
            case "malformed_duplicate_nested":
                requireReset(readRequest());
                String duplicate = decisionResponse("reset", true, false, false)
                        .replace("\"authority_kind\":",
                                "\"authority_kind\":\"original-promoted2-generation384-store\","
                                        + "\"authority_kind\":");
                writeLf(System.out, duplicate);
                return;
            case "missing_clock":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false)
                        .replace(",\"kernel_clock\":" + kernelClockJson("p0"), ""));
                return;
            case "unknown_clock_field":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false)
                        .replace("\"stack_depth\":0}",
                                "\"stack_depth\":0,\"unknown\":0}"));
                return;
            case "invalid_clock_phase":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false)
                        .replace("\"phase_step\":\"Main1\"",
                                "\"phase_step\":\"MainThree\""));
                return;
            case "v1_response":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false)
                        .replace(XMageRallyBridgeProtocol.PROTOCOL,
                                "mtg-kernel-checkpoint-shadow-stdio/v1")
                        .replace("\"schema_version\":2", "\"schema_version\":1"));
                return;
            case "score_clock_drift":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false));
                requireScore(readRequest(), 0L);
                writeLf(System.out, decisionResponse("score", false, false, false)
                        .replace("\"phase_step\":\"Main1\"",
                                "\"phase_step\":\"Main2\""));
                return;
            case "clock_mismatch_error":
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false));
                requireStep(readRequest(), 0L, 1);
                writeLf(System.out, clockMismatchResponse("step-0"));
                return;
            case "error":
                requireReset(readRequest());
                writeLf(System.out, errorResponse("reset"));
                return;
            case "timeout":
                requireReset(readRequest());
                Thread.sleep(1_000L);
                return;
            case "eof":
                requireReset(readRequest());
                return;
            case "crlf":
                requireReset(readRequest());
                writeCrlf(System.out, decisionResponse("reset", true, false, false));
                return;
            case "invalid_utf8":
                requireReset(readRequest());
                System.out.write(0xc3);
                System.out.write(0x28);
                System.out.write('\n');
                System.out.flush();
                return;
            case "overlong":
                requireReset(readRequest());
                StringBuilder longLine = new StringBuilder(2048);
                for (int i = 0; i < 2048; i++) {
                    longLine.append('x');
                }
                writeLf(System.out, longLine.toString());
                return;
            case "stderr":
                System.err.println("peer diagnostic sentinel");
                System.err.flush();
                requireReset(readRequest());
                writeLf(System.out, decisionResponse("reset", true, false, false));
                return;
            default:
                System.err.println("unknown peer mode: " + args[0]);
                System.exit(2);
        }
    }

    private static void serveFlow() throws Exception {
        requireReset(readRequest());
        writeLf(System.out, decisionResponse("reset", true, false, false));

        requireScore(readRequest(), 0L);
        writeLf(System.out, decisionResponse("score", false, false, false));

        requireStep(readRequest(), 0L, 1);
        writeLf(System.out, decisionResponse("step-0", false, true, false));

        requireStep(readRequest(), 1L, 0);
        writeLf(System.out, terminalResponse("step-1"));
    }

    private static void serveMultiEpisode() throws Exception {
        requireReset(readRequest());
        writeLf(System.out, decisionResponse("reset", true, false, false));
        requireStep(readRequest(), 0L, 1);
        writeLf(System.out, decisionResponse("step-0", false, true, false));
        requireStep(readRequest(), 1L, 0);
        writeLf(System.out, terminalResponse("step-1"));

        XMageRallyBridgeProtocol.Request next = readRequest();
        if (!(next instanceof XMageRallyBridgeProtocol.ResetRequest)) {
            throw new IOException("expected next reset request");
        }
        XMageRallyBridgeProtocol.ResetRequest reset =
                (XMageRallyBridgeProtocol.ResetRequest) next;
        if (!"reset-next".equals(reset.getRequestId())
                || reset.getEpisodeId() != 4L || reset.getBaseSeed() != BASE_SEED) {
            throw new IOException("next reset request mismatch");
        }
        String nextResponse = decisionResponse("reset-next", true, false, false)
                .replace("\"pair_index\":1", "\"pair_index\":2")
                .replace(PAIR_ENV_HEX, "163d0eb1c4158f47")
                .replace("\"episode_id\":2", "\"episode_id\":4");
        writeLf(System.out, nextResponse);
    }

    private static XMageRallyBridgeProtocol.Request readRequest() throws Exception {
        return CODEC.decodeRequest(readLfLine(System.in));
    }

    private static void requireReset(XMageRallyBridgeProtocol.Request request) throws IOException {
        if (!(request instanceof XMageRallyBridgeProtocol.ResetRequest)) {
            throw new IOException("expected reset request");
        }
        XMageRallyBridgeProtocol.ResetRequest reset =
                (XMageRallyBridgeProtocol.ResetRequest) request;
        if (!"reset".equals(reset.getRequestId())
                || reset.getEpisodeId() != EPISODE || reset.getBaseSeed() != BASE_SEED) {
            throw new IOException("reset request mismatch");
        }
    }

    private static void requireScore(XMageRallyBridgeProtocol.Request request, long step)
            throws IOException {
        if (!(request instanceof XMageRallyBridgeProtocol.ScoreCurrentRequest)) {
            throw new IOException("expected score_current request");
        }
        XMageRallyBridgeProtocol.ScoreCurrentRequest score =
                (XMageRallyBridgeProtocol.ScoreCurrentRequest) request;
        if (!"score".equals(score.getRequestId())
                || score.getEpisodeId() != EPISODE || score.getExpectedStep() != step) {
            throw new IOException("score_current request mismatch");
        }
    }

    private static void requireStep(XMageRallyBridgeProtocol.Request request,
                                    long step,
                                    int selected) throws IOException {
        if (!(request instanceof XMageRallyBridgeProtocol.StepRequest)) {
            throw new IOException("expected step request");
        }
        XMageRallyBridgeProtocol.StepRequest actual =
                (XMageRallyBridgeProtocol.StepRequest) request;
        XMageRallyBridgeProtocol.ExpectedClock expectedClock = stepClock(step);
        if (!("step-" + step).equals(actual.getRequestId())
                || actual.getEpisodeId() != EPISODE
                || actual.getExpectedStep() != step
                || actual.getSelectedIndex() != selected
                || !expectedClock.equals(actual.getExpectedClock())) {
            throw new IOException("step request mismatch");
        }
    }

    private static XMageRallyBridgeProtocol.ExpectedClock stepClock(long step) {
        return new XMageRallyBridgeProtocol.ExpectedClock(
                1L,
                XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                step == 0L ? XMageRallyBridgeProtocol.Seat.P0
                        : XMageRallyBridgeProtocol.Seat.P1);
    }

    private static String decisionResponse(String requestId,
                                           boolean includeLibraries,
                                           boolean afterFirstStep,
                                           boolean wrongCheckpoint) {
        String actor = afterFirstStep ? "p1" : "p0";
        long step = afterFirstStep ? 1L : 0L;
        long revision = afterFirstStep ? 1L : 0L;
        long physical = afterFirstStep ? 1L : 0L;
        String commit = afterFirstStep ? COMMIT_1 : COMMIT_0;
        String model = afterFirstStep ? MODEL_1 : MODEL_0;
        String semantics = afterFirstStep
                ? "[{\"action_kind\":\"pass\",\"actor\":\"p1\"},"
                + "{\"action_kind\":\"choose_optional_cost_use\","
                + "\"actor\":\"p1\",\"use_cost\":false}]"
                : "[{\"action_kind\":\"pass\",\"actor\":\"p0\"}," + LAND_P0 + "]";
        String selectionSeed = afterFirstStep ? "null" : "\"5555555555555555\"";
        String selected = afterFirstStep ? "null" : "1";
        String applied = afterFirstStep ? appliedLand() : "null";
        String kernelClock = kernelClockJson(actor);
        return "{\"protocol\":\"" + XMageRallyBridgeProtocol.PROTOCOL + "\","
                + "\"schema_version\":" + XMageRallyBridgeProtocol.SCHEMA_VERSION
                + ",\"request_id\":\"" + requestId + "\","
                + checkpointJson(wrongCheckpoint)
                + ",\"response_type\":\"decision\",\"decision\":{"
                + "\"deck_ids\":[\"Rally\",\"Rally\"],"
                + "\"randomization_identity\":\"legacy_v1\","
                + "\"base_seed_u64_hex\":\"" + BASE_SEED_HEX + "\","
                + "\"pair_index\":1,\"pair_environment_seed_u64_hex\":\""
                + PAIR_ENV_HEX + "\","
                + (includeLibraries
                ? "\"initial_library_card_definition_ids\":" + librariesJson() + "," : "")
                + "\"episode_id\":2,\"step\":" + step
                + ",\"environment_revision\":" + revision
                + ",\"physical_decision_id\":" + physical
                + ",\"substep_index\":0,\"substep_count\":1,"
                + "\"acting_player\":\"" + actor + "\",\"decision_kind\":\"surface\","
                + "\"legal_action_count\":2,\"candidate_seat\":\"p0\","
                + "\"candidate_controls_current_actor\":" + (!afterFirstStep) + ","
                + "\"actor_physical_decision_ordinal\":0,"
                + "\"candidate_action_seed_u64_hex\":" + selectionSeed + ","
                + "\"selected_action_index\":" + selected + ","
                + "\"candidate_order_commitment_128_hex\":\"" + commit + "\","
                + "\"model_input_commitment\":\""
                + XMageRallyBridgeProtocol.MODEL_INPUT_COMMITMENT + "\","
                + "\"model_input_sha256\":\"" + model + "\","
                + "\"diagnostic_state_hash_u64_hex\":\"3333333333333333\","
                + "\"core_environment_hash_u64_hex\":\"4444444444444444\","
                + "\"logits_f32_bits\":[1056964608,1065353216],"
                + "\"value_f32_bits\":1048576000,\"action_semantics\":" + semantics
                + ",\"kernel_clock\":" + kernelClock
                + "},\"applied_action\":" + applied + "}";
    }

    private static String terminalResponse(String requestId) {
        return "{\"protocol\":\"" + XMageRallyBridgeProtocol.PROTOCOL + "\","
                + "\"schema_version\":" + XMageRallyBridgeProtocol.SCHEMA_VERSION
                + ",\"request_id\":\"" + requestId + "\","
                + checkpointJson(false)
                + ",\"response_type\":\"terminal\",\"terminal\":{"
                + "\"deck_ids\":[\"Rally\",\"Rally\"],"
                + "\"randomization_identity\":\"legacy_v1\","
                + "\"base_seed_u64_hex\":\"" + BASE_SEED_HEX + "\","
                + "\"pair_index\":1,\"pair_environment_seed_u64_hex\":\""
                + PAIR_ENV_HEX + "\",\"terminal\":{"
                + "\"schema_version\":5,\"deck_ids\":[\"Rally\",\"Rally\"],"
                + "\"deck_hashes\":[1,2],\"episode_id\":2,"
                + "\"terminal_outcome\":\"p0_win\","
                + "\"terminal_classification\":\"natural\","
                + "\"terminal_code\":\"natural_game_over\",\"winner\":\"p0\","
                + "\"terminal_reward\":[1,-1],\"terminal_reason\":\"game_over\","
                + "\"policy_step_count\":2,\"physical_decision_count\":2},"
                + "\"candidate_seat\":\"p0\","
                + "\"diagnostic_state_hash_u64_hex\":\"8888888888888888\","
                + "\"core_environment_hash_u64_hex\":\"9999999999999999\"},"
                + "\"applied_action\":" + appliedPassP1() + "}";
    }

    private static String errorResponse(String requestId) {
        return "{\"protocol\":\"" + XMageRallyBridgeProtocol.PROTOCOL + "\","
                + "\"schema_version\":" + XMageRallyBridgeProtocol.SCHEMA_VERSION
                + ",\"request_id\":\"" + requestId + "\","
                + checkpointJson(false)
                + ",\"response_type\":\"error\",\"error_code\":\"test_error\","
                + "\"message\":\"intentional test error\"}";
    }

    private static String clockMismatchResponse(String requestId) {
        return errorResponse(requestId)
                .replace("\"error_code\":\"test_error\"",
                        "\"error_code\":\"clock_mismatch\"")
                .replace("intentional test error",
                        "expected_clock does not match the kernel clock of the current decision");
    }

    private static String kernelClockJson(String actor) {
        return "{\"turn\":1,\"phase_step\":\"Main1\","
                + "\"active_player\":\"" + actor + "\","
                + "\"priority_player\":\"" + actor + "\",\"stack_depth\":0}";
    }

    private static String checkpointJson(boolean wrong) {
        String checkpoint = wrong
                ? "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                : XMageRallyBridgeProtocol.SOURCE_CHECKPOINT_SHA256;
        return "\"checkpoint\":{"
                + "\"authority_kind\":\"" + XMageRallyBridgeProtocol.ORIGINAL_AUTHORITY_KIND + "\","
                + "\"source_run_sha256\":\"" + XMageRallyBridgeProtocol.SOURCE_RUN_SHA256 + "\","
                + "\"source_generation\":384,\"source_checkpoint_sha256\":\""
                + checkpoint + "\",\"source_sidecar_sha256\":\""
                + XMageRallyBridgeProtocol.SOURCE_SIDECAR_SHA256 + "\","
                + "\"source_payload_sha256\":\"" + XMageRallyBridgeProtocol.SOURCE_PAYLOAD_SHA256
                + "\",\"source_train_state_sha256\":\""
                + XMageRallyBridgeProtocol.SOURCE_TRAIN_STATE_SHA256 + "\","
                + "\"loaded_run_sha256\":\"" + XMageRallyBridgeProtocol.SOURCE_RUN_SHA256 + "\","
                + "\"loaded_generation\":384,\"loaded_checkpoint_sha256\":\""
                + XMageRallyBridgeProtocol.SOURCE_CHECKPOINT_SHA256 + "\","
                + "\"loaded_payload_sha256\":\"" + XMageRallyBridgeProtocol.SOURCE_PAYLOAD_SHA256
                + "\",\"loaded_train_state_sha256\":\""
                + XMageRallyBridgeProtocol.SOURCE_TRAIN_STATE_SHA256 + "\","
                + "\"model_parameter_sha256\":\"" + XMageRallyBridgeProtocol.MODEL_PARAMETER_SHA256
                + "\",\"environment_trajectory_contract\":\"legacy-v1\","
                + "\"sampler_identity\":\"" + XMageRallyBridgeProtocol.SAMPLER_IDENTITY + "\","
                + "\"sampler_contract_sha256\":\""
                + XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256 + "\"}";
    }

    private static String appliedLand() {
        return "{\"episode_id\":2,\"step\":0,"
                + "\"candidate_order_commitment_128_hex\":\"" + COMMIT_0 + "\","
                + "\"model_input_sha256\":\"" + MODEL_0 + "\",\"selected_index\":1,"
                + "\"selected_logit_f32_bits\":1065353216,\"semantic\":" + LAND_P0 + "}";
    }

    private static String appliedPassP1() {
        return "{\"episode_id\":2,\"step\":1,"
                + "\"candidate_order_commitment_128_hex\":\"" + COMMIT_1 + "\","
                + "\"model_input_sha256\":\"" + MODEL_1 + "\",\"selected_index\":0,"
                + "\"selected_logit_f32_bits\":1056964608,"
                + "\"semantic\":{\"action_kind\":\"pass\",\"actor\":\"p1\"}}";
    }

    private static String librariesJson() {
        StringBuilder result = new StringBuilder(420).append('[');
        for (int row = 0; row < 2; row++) {
            if (row != 0) {
                result.append(',');
            }
            result.append('[');
            for (int i = 0; i < 60; i++) {
                if (i != 0) {
                    result.append(',');
                }
                result.append(row * 100 + i + 1);
            }
            result.append(']');
        }
        return result.append(']').toString();
    }

    private static String readLfLine(InputStream input) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        while (true) {
            int value = input.read();
            if (value < 0) {
                throw new IOException("EOF before request LF");
            }
            if (value == '\n') {
                break;
            }
            if (value == '\r') {
                throw new IOException("request used CR instead of LF");
            }
            bytes.write(value);
        }
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes.toByteArray())).toString();
        } catch (CharacterCodingException e) {
            throw new IOException("request is not UTF-8", e);
        }
    }

    private static void writeLf(OutputStream output, String line) throws IOException {
        output.write(line.getBytes(StandardCharsets.UTF_8));
        output.write('\n');
        output.flush();
    }

    private static void writeCrlf(OutputStream output, String line) throws IOException {
        output.write(line.getBytes(StandardCharsets.UTF_8));
        output.write('\r');
        output.write('\n');
        output.flush();
    }
}
