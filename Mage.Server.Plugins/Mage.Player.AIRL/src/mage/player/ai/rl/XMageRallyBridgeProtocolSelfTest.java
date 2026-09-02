package mage.player.ai.rl;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/** Standalone adversarial test for the rich Rally checkpoint shadow protocol. */
public final class XMageRallyBridgeProtocolSelfTest {

    private static final XMageRallyBridgeJsonCodec CODEC = new XMageRallyBridgeJsonCodec();
    private static final long EPISODE = 2L;
    private static final long BASE_SEED = 71_501L;
    private static int passed;
    private static int failed;

    private XMageRallyBridgeProtocolSelfTest() {
    }

    public static void main(String[] args) throws Exception {
        run("protocol-v2-identity", XMageRallyBridgeProtocolSelfTest::testProtocolIdentity);
        run("clock-normalization-and-phase-map",
                XMageRallyBridgeProtocolSelfTest::testClockComparator);
        run("request-round-trip", XMageRallyBridgeProtocolSelfTest::testRequestRoundTrip);
        run("pair-environment-seed-parity",
                XMageRallyBridgeProtocolSelfTest::testPairEnvironmentSeedParity);
        run("request-unknown-field-rejected",
                XMageRallyBridgeProtocolSelfTest::testRequestUnknownField);
        run("request-duplicate-field-rejected",
                XMageRallyBridgeProtocolSelfTest::testRequestDuplicateField);
        run("request-noncanonical-integer-rejected",
                XMageRallyBridgeProtocolSelfTest::testRequestNoncanonicalInteger);
        run("direct-executable-required",
                XMageRallyBridgeProtocolSelfTest::testDirectExecutableRequired);
        run("configured-derivative-authority-kind",
                XMageRallyBridgeProtocolSelfTest::testConfiguredDerivativeAuthorityKind);
        run("population-store-identity-is-exact",
                XMageRallyBridgeProtocolSelfTest::testPopulationStoreIdentity);
        run("population-store-properties-reject-stale-hash",
                XMageRallyBridgeProtocolSelfTest::testPopulationStoreProperties);
        run("reset-score-step-terminal-flow",
                XMageRallyBridgeProtocolSelfTest::testFullFlow);
        run("terminal-allows-next-episode-reset",
                XMageRallyBridgeProtocolSelfTest::testSequentialEpisodeReset);
        run("wrong-checkpoint-fails-closed",
                () -> expectResetFailure("wrong_checkpoint", 2_000L, 1_048_576));
        run("request-id-echo-fails-closed",
                () -> expectResetFailure("echo_mismatch", 2_000L, 1_048_576));
        run("applied-commitment-mismatch-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testTransitionMismatch);
        run("unknown-response-field-fails-closed",
                () -> expectResetFailure("malformed_unknown", 2_000L, 1_048_576));
        run("duplicate-nested-field-fails-closed",
                () -> expectResetFailure("malformed_duplicate_nested", 2_000L, 1_048_576));
        run("missing-decision-clock-fails-closed",
                () -> expectResetFailure("missing_clock", 2_000L, 1_048_576));
        run("reset-round-1-p1-active-accepted",
                XMageRallyBridgeProtocolSelfTest::testResetRoundOneP1Active);
        run("reset-round-2-fails-closed",
                () -> expectResetFailure("reset_round_2", 2_000L, 1_048_576));
        run("reset-binding-round-and-starter",
                XMageRallyBridgeProtocolSelfTest::testResetBindingRule);
        run("unknown-decision-clock-field-fails-closed",
                () -> expectResetFailure("unknown_clock_field", 2_000L, 1_048_576));
        run("invalid-decision-clock-phase-fails-closed",
                () -> expectResetFailure("invalid_clock_phase", 2_000L, 1_048_576));
        run("v1-response-identity-fails-closed",
                () -> expectResetFailure("v1_response", 2_000L, 1_048_576));
        run("score-current-clock-splice-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testScoreClockSplice);
        run("typed-clock-mismatch-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testClockMismatchError);
        run("typed-error-fails-closed",
                () -> expectResetFailure("error", 2_000L, 1_048_576));
        run("episode-splice-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testEpisodeSplice);
        run("step-splice-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testStepSplice);
        run("request-id-reuse-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testRequestIdReuse);
        run("second-reset-fails-closed",
                XMageRallyBridgeProtocolSelfTest::testSecondReset);
        run("timeout-fails-closed",
                () -> expectResetFailure("timeout", 100L, 1_048_576));
        run("eof-fails-closed",
                () -> expectResetFailure("eof", 2_000L, 1_048_576));
        run("crlf-fails-closed",
                () -> expectResetFailure("crlf", 2_000L, 1_048_576));
        run("invalid-utf8-fails-closed",
                () -> expectResetFailure("invalid_utf8", 2_000L, 1_048_576));
        run("overlong-line-fails-closed",
                () -> expectResetFailure("overlong", 2_000L, 1_024));
        run("stderr-is-diagnostic-only",
                XMageRallyBridgeProtocolSelfTest::testStderrIsolation);

        System.out.println("Total: " + (passed + failed)
                + "  Passed: " + passed + "  Failed: " + failed);
        if (failed != 0) {
            throw new AssertionError(failed + " Rally bridge self-test(s) failed");
        }
    }

    private static void testRequestRoundTrip() throws Exception {
        XMageRallyBridgeProtocol.ResetRequest reset =
                new XMageRallyBridgeProtocol.ResetRequest("reset", EPISODE, BASE_SEED);
        String resetJson = CODEC.encodeRequest(reset);
        require(resetJson.equals("{\"request_type\":\"reset\",\"request_id\":\"reset\","
                        + "\"episode_id\":2,\"base_seed\":71501}"),
                "reset JSON bytes changed: " + resetJson);
        XMageRallyBridgeProtocol.Request decoded = CODEC.decodeRequest(resetJson);
        require(decoded instanceof XMageRallyBridgeProtocol.ResetRequest,
                "reset decoded as wrong type");
        require(((XMageRallyBridgeProtocol.ResetRequest) decoded).getBaseSeed() == BASE_SEED,
                "base_seed changed");

        XMageRallyBridgeProtocol.StepRequest step =
                new XMageRallyBridgeProtocol.StepRequest(
                        "step-0", EPISODE, 0L, 1, clockFor(0L));
        String stepJson = CODEC.encodeRequest(step);
        require(stepJson.equals("{\"request_type\":\"step\",\"request_id\":\"step-0\","
                        + "\"episode_id\":2,\"expected_step\":0,\"selected_index\":1,"
                        + "\"expected_clock\":{\"turn\":1,\"phase_step\":\"Main1\","
                        + "\"active_player\":\"p0\"}}"),
                "v2 step JSON bytes changed: " + stepJson);
        XMageRallyBridgeProtocol.Request decodedStep = CODEC.decodeRequest(stepJson);
        require(decodedStep instanceof XMageRallyBridgeProtocol.StepRequest
                        && ((XMageRallyBridgeProtocol.StepRequest) decodedStep)
                        .getSelectedIndex() == 1
                        && clockFor(0L).equals(
                        ((XMageRallyBridgeProtocol.StepRequest) decodedStep)
                                .getExpectedClock()),
                "step round-trip failed");
        XMageRallyBridgeProtocol.StepRequest optionalClock =
                (XMageRallyBridgeProtocol.StepRequest) CODEC.decodeRequest(
                        stepJson.replace(",\"expected_clock\":{\"turn\":1,"
                                + "\"phase_step\":\"Main1\",\"active_player\":\"p0\"}", ""));
        require(optionalClock.getExpectedClock() == null,
                "optional expected_clock could not be omitted at codec boundary");
        expectProtocolFailure(stepJson.replace(
                "{\"turn\":1,\"phase_step\":\"Main1\",\"active_player\":\"p0\"}",
                "null"));
        expectProtocolFailure(stepJson.replace(
                "\"active_player\":\"p0\"",
                "\"active_player\":\"p0\",\"unknown\":0"));
        expectProtocolFailure(stepJson.replace(
                "\"phase_step\":\"Main1\"",
                "\"phase_step\":\"MainThree\""));
        expectProtocolFailure(stepJson.replace(
                "\"turn\":1", "\"turn\":4294967296"));
        expectProtocolFailure(stepJson.replace(
                "\"turn\":1", "\"turn\":1,\"turn\":1"));
        expectProtocolFailure(stepJson.replace(
                ",\"phase_step\":\"Main1\"", ""));
    }

    private static void testProtocolIdentity() {
        require("mtg-kernel-checkpoint-shadow-stdio/v2".equals(
                        XMageRallyBridgeProtocol.PROTOCOL),
                "protocol identity is not v2");
        require(XMageRallyBridgeProtocol.SCHEMA_VERSION == 2,
                "schema version is not 2");
    }

    private static void testClockComparator() {
        mage.constants.PhaseStep[] xmagePhases = {
                mage.constants.PhaseStep.UNTAP,
                mage.constants.PhaseStep.UPKEEP,
                mage.constants.PhaseStep.DRAW,
                mage.constants.PhaseStep.PRECOMBAT_MAIN,
                mage.constants.PhaseStep.BEGIN_COMBAT,
                mage.constants.PhaseStep.DECLARE_ATTACKERS,
                mage.constants.PhaseStep.DECLARE_BLOCKERS,
                mage.constants.PhaseStep.FIRST_COMBAT_DAMAGE,
                mage.constants.PhaseStep.COMBAT_DAMAGE,
                mage.constants.PhaseStep.END_COMBAT,
                mage.constants.PhaseStep.POSTCOMBAT_MAIN,
                mage.constants.PhaseStep.END_TURN,
                mage.constants.PhaseStep.CLEANUP
        };
        XMageRallyBridgeProtocol.KernelPhaseStep[] kernelPhases = {
                XMageRallyBridgeProtocol.KernelPhaseStep.UNTAP,
                XMageRallyBridgeProtocol.KernelPhaseStep.UPKEEP,
                XMageRallyBridgeProtocol.KernelPhaseStep.DRAW,
                XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                XMageRallyBridgeProtocol.KernelPhaseStep.BEGIN_COMBAT,
                XMageRallyBridgeProtocol.KernelPhaseStep.DECLARE_ATTACKERS,
                XMageRallyBridgeProtocol.KernelPhaseStep.DECLARE_BLOCKERS,
                XMageRallyBridgeProtocol.KernelPhaseStep.COMBAT_DAMAGE,
                XMageRallyBridgeProtocol.KernelPhaseStep.COMBAT_DAMAGE,
                XMageRallyBridgeProtocol.KernelPhaseStep.END_COMBAT,
                XMageRallyBridgeProtocol.KernelPhaseStep.MAIN2,
                XMageRallyBridgeProtocol.KernelPhaseStep.END,
                XMageRallyBridgeProtocol.KernelPhaseStep.CLEANUP
        };
        for (int i = 0; i < xmagePhases.length; i++) {
            XMageRallyBridgeProtocol.KernelClock kernel =
                    new XMageRallyBridgeProtocol.KernelClock(
                            6L, kernelPhases[i], XMageRallyBridgeProtocol.Seat.P0,
                            XMageRallyBridgeProtocol.Seat.P1, 4_294_967_295L);
            XMageRallyBridgeProtocol.ExpectedClock matched =
                    XMageRallyClockComparator.requireMatch(
                            kernel, 11L, xmagePhases[i],
                            XMageRallyBridgeProtocol.Seat.P0, "self_test", "phase");
            require(matched.getTurn() == 6L
                            && matched.getPhaseStep() == kernelPhases[i]
                            && matched.getActivePlayer() == XMageRallyBridgeProtocol.Seat.P0,
                    "clock phase mapping failed at " + xmagePhases[i]);
        }
        XMageRallyBridgeProtocol.KernelClock p1 =
                new XMageRallyBridgeProtocol.KernelClock(
                        6L, XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                        XMageRallyBridgeProtocol.Seat.P1,
                        XMageRallyBridgeProtocol.Seat.P0, 0L);
        require(XMageRallyClockComparator.requireMatch(
                        p1, 12L, mage.constants.PhaseStep.PRECOMBAT_MAIN,
                        XMageRallyBridgeProtocol.Seat.P1, "self_test", "p1")
                        .getTurn() == 6L,
                "kernel turn 6 p1 did not normalize to XMage T12");
        expectClockFailure(p1, 11L, mage.constants.PhaseStep.PRECOMBAT_MAIN,
                XMageRallyBridgeProtocol.Seat.P1, "turn mismatch");
        expectClockFailure(p1, 12L, mage.constants.PhaseStep.DECLARE_ATTACKERS,
                XMageRallyBridgeProtocol.Seat.P1, "phase mismatch");
        expectClockFailure(p1, 12L, mage.constants.PhaseStep.PRECOMBAT_MAIN,
                XMageRallyBridgeProtocol.Seat.P0, "active mismatch");
    }

    /**
     * The kernel clock's turn is a round counter spanning both seats, so the
     * first surfaced decision of an episode may legitimately be p1's Main1 in
     * round 1 (XMage turn 2) when p0's opening turn surfaces no decision.
     */
    private static void testResetRoundOneP1Active() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start("reset_p1_active", 2_000L,
                     1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            require(client.isUsable(), "round-1 p1-active reset poisoned the client");
            XMageRallyBridgeProtocol.DecisionBody first = client.getCurrentDecision();
            require(first != null && first.getStep() == 0L
                            && first.getKernelClock() != null
                            && first.getKernelClock().getTurn() == 1L
                            && first.getKernelClock().getPhaseStep()
                            == XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1
                            && first.getKernelClock().getActivePlayer()
                            == XMageRallyBridgeProtocol.Seat.P1
                            && first.getKernelClock().getPriorityPlayer()
                            == XMageRallyBridgeProtocol.Seat.P1
                            && first.getKernelClock().getStackDepth() == 0L,
                    "round-1 p1-active reset was not bound");
            require(XMageRallyClockComparator.requireResetBinding(
                            first.getKernelClock(), 2L,
                            mage.constants.PhaseStep.PRECOMBAT_MAIN,
                            XMageRallyBridgeProtocol.Seat.P1,
                            XMageRallyBridgeProtocol.Seat.P0,
                            "self_test", "reset_round_1_p1").getTurn() == 1L,
                    "round-1 p1 reset did not normalize to XMage turn 2");
            expectResetBindingFailure(first.getKernelClock(), 1L,
                    XMageRallyBridgeProtocol.Seat.P1,
                    XMageRallyBridgeProtocol.Seat.P0,
                    "p1 reset accepted XMage turn 1");
            expectResetBindingFailure(first.getKernelClock(), 2L,
                    XMageRallyBridgeProtocol.Seat.P1,
                    XMageRallyBridgeProtocol.Seat.P1,
                    "p1 reset accepted a non-p0 XMage starting player");
        }
        require(diagnosticBytes.size() == 0,
                "accepted p1-active reset emitted diagnostics: " + utf8(diagnosticBytes));
    }

    private static void testResetBindingRule() {
        XMageRallyBridgeProtocol.KernelClock roundOneP0 =
                new XMageRallyBridgeProtocol.KernelClock(
                        1L, XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                        XMageRallyBridgeProtocol.Seat.P0,
                        XMageRallyBridgeProtocol.Seat.P0, 0L);
        require(XMageRallyClockComparator.requireResetBinding(
                        roundOneP0, 1L, mage.constants.PhaseStep.PRECOMBAT_MAIN,
                        XMageRallyBridgeProtocol.Seat.P0,
                        XMageRallyBridgeProtocol.Seat.P0,
                        "self_test", "reset_round_1_p0").getTurn() == 1L,
                "round-1 p0 reset no longer binds XMage turn 1");
        expectResetBindingFailure(roundOneP0, 1L,
                XMageRallyBridgeProtocol.Seat.P0,
                XMageRallyBridgeProtocol.Seat.P1,
                "p0 reset accepted a non-p0 XMage starting player");
        XMageRallyBridgeProtocol.KernelClock roundTwoP0 =
                new XMageRallyBridgeProtocol.KernelClock(
                        2L, XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                        XMageRallyBridgeProtocol.Seat.P0,
                        XMageRallyBridgeProtocol.Seat.P0, 0L);
        expectResetBindingFailure(roundTwoP0, 3L,
                XMageRallyBridgeProtocol.Seat.P0,
                XMageRallyBridgeProtocol.Seat.P0,
                "reset accepted kernel round 2");
        XMageRallyBridgeProtocol.KernelClock roundTwoP1 =
                new XMageRallyBridgeProtocol.KernelClock(
                        2L, XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                        XMageRallyBridgeProtocol.Seat.P1,
                        XMageRallyBridgeProtocol.Seat.P1, 0L);
        expectResetBindingFailure(roundTwoP1, 4L,
                XMageRallyBridgeProtocol.Seat.P1,
                XMageRallyBridgeProtocol.Seat.P0,
                "reset accepted kernel round 2 on p1");
    }

    private static void expectResetBindingFailure(
            XMageRallyBridgeProtocol.KernelClock resetClock,
            long xmageGlobalTurn,
            XMageRallyBridgeProtocol.Seat xmageActivePlayer,
            XMageRallyBridgeProtocol.Seat xmageStartingPlayer,
            String label) {
        boolean threw = false;
        try {
            XMageRallyClockComparator.requireResetBinding(
                    resetClock, xmageGlobalTurn,
                    mage.constants.PhaseStep.PRECOMBAT_MAIN,
                    xmageActivePlayer, xmageStartingPlayer, "self_test", label);
        } catch (XMageRallyClockComparator.ClockMismatch expected) {
            threw = expected.getMessage().contains(XMageRallyClockComparator.MARKER);
        }
        require(threw, label);
    }

    private static void expectClockFailure(
            XMageRallyBridgeProtocol.KernelClock kernel,
            long turn,
            mage.constants.PhaseStep phase,
            XMageRallyBridgeProtocol.Seat active,
            String label) {
        boolean threw = false;
        try {
            XMageRallyClockComparator.requireMatch(
                    kernel, turn, phase, active, "self_test", label);
        } catch (XMageRallyClockComparator.ClockMismatch expected) {
            threw = expected.getMessage().contains(XMageRallyClockComparator.MARKER);
        }
        require(threw, label + " did not retain the clock mismatch marker");
    }

    private static void testRequestUnknownField() throws Exception {
        String json = CODEC.encodeRequest(
                new XMageRallyBridgeProtocol.ResetRequest("reset", EPISODE, BASE_SEED));
        expectProtocolFailure(json.substring(0, json.length() - 1) + ",\"extra\":0}");
    }

    private static void testPairEnvironmentSeedParity() {
        require("6683b059f7fd4a71".equals(
                        XMageRallyBridgeProtocol.derivePairEnvironmentSeedU64Hex(
                                BASE_SEED, EPISODE)),
                "pair-1 environment seed drifted");
        require("163d0eb1c4158f47".equals(
                        XMageRallyBridgeProtocol.derivePairEnvironmentSeedU64Hex(
                                BASE_SEED, 4L)),
                "pair-2 environment seed drifted");
    }

    private static void testRequestDuplicateField() throws Exception {
        String json = CODEC.encodeRequest(
                new XMageRallyBridgeProtocol.ResetRequest("reset", EPISODE, BASE_SEED));
        expectProtocolFailure(json.replace("\"episode_id\":2,",
                "\"episode_id\":2,\"episode_id\":2,"));
    }

    private static void testRequestNoncanonicalInteger() throws Exception {
        String json = CODEC.encodeRequest(
                new XMageRallyBridgeProtocol.ResetRequest("reset", EPISODE, BASE_SEED));
        expectProtocolFailure(json.replace("\"episode_id\":2", "\"episode_id\":2.0"));
        expectProtocolFailure(json.replace("\"base_seed\":71501", "\"base_seed\":7.1501e4"));
    }

    private static void testDirectExecutableRequired() throws Exception {
        boolean threw = false;
        try {
            XMageRallyBridgeProcessClient.start(
                    Arrays.asList("cargo", "run", "--bin", "checkpoint_shadow_stdio_v1"));
        } catch (IllegalArgumentException expected) {
            threw = true;
        }
        require(threw, "cargo wrapper was accepted");
    }

    private static void testConfiguredDerivativeAuthorityKind() {
        String authority = "qualified-policy-bounded-value-search-v1";
        String manifest =
                "204beb91c1a4b039e0c497f2b420e823b5cc9e2ceb8560f897d0b6251e916b72";
        String payload =
                "ca3c45cd69d8d60f1f921bc78c27b098064ef6b16fe7566b84e5045681781b28";
        String trainState =
                "7d854edb46119a611d4283e6cf4630d0207ceb24c12b4089a7d27a43c97fe0b3";
        String model =
                "47b10c1114efc01f9445c71c0c8c4d8cd4a4b89a2154ac68275f3b0c6ebb9ce3";
        XMageRallyBridgeProtocol.CheckpointIdentity checkpoint =
                new XMageRallyBridgeProtocol.CheckpointIdentity(
                        authority,
                        XMageRallyBridgeProtocol.SOURCE_RUN_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_GENERATION,
                        XMageRallyBridgeProtocol.SOURCE_CHECKPOINT_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_SIDECAR_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_PAYLOAD_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_TRAIN_STATE_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_RUN_SHA256,
                        1L,
                        manifest,
                        payload,
                        trainState,
                        model,
                        XMageRallyBridgeProtocol.ENVIRONMENT_TRAJECTORY_CONTRACT,
                        XMageRallyBridgeProtocol.SAMPLER_IDENTITY,
                        XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
        checkpoint.requireXMageCp7OutcomeAuthority(
                authority, 1L, manifest, payload, trainState, model,
                XMageRallyBridgeProtocol.ENVIRONMENT_TRAJECTORY_CONTRACT);

        String environmentV2 = "environment-randomization-v2";
        XMageRallyBridgeProtocol.CheckpointIdentity environmentV2Checkpoint =
                new XMageRallyBridgeProtocol.CheckpointIdentity(
                        authority,
                        XMageRallyBridgeProtocol.SOURCE_RUN_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_GENERATION,
                        XMageRallyBridgeProtocol.SOURCE_CHECKPOINT_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_SIDECAR_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_PAYLOAD_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_TRAIN_STATE_SHA256,
                        XMageRallyBridgeProtocol.SOURCE_RUN_SHA256,
                        1L,
                        manifest,
                        payload,
                        trainState,
                        model,
                        environmentV2,
                        XMageRallyBridgeProtocol.SAMPLER_IDENTITY,
                        XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
        environmentV2Checkpoint.requireXMageCp7OutcomeAuthority(
                authority, 1L, manifest, payload, trainState, model, environmentV2);
        boolean rejectedMismatch = false;
        try {
            environmentV2Checkpoint.requireXMageCp7OutcomeAuthority(
                    authority, 1L, manifest, payload, trainState, model,
                    XMageRallyBridgeProtocol.ENVIRONMENT_TRAJECTORY_CONTRACT);
        } catch (IllegalArgumentException expected) {
            rejectedMismatch = true;
        }
        require(rejectedMismatch, "environment trajectory contract mismatch was accepted");
    }

    private static void testPopulationStoreIdentity() {
        String shaA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        String shaB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        XMageRallyBridgeProtocol.CheckpointIdentity checkpoint =
                new XMageRallyBridgeProtocol.CheckpointIdentity(
                        "population-store-validated-generation", shaA, 1024L,
                        shaB, shaA, shaB, shaA, shaA, 1024L, shaB, shaA, shaB, shaA,
                        "environment-randomization-v2",
                        XMageRallyBridgeProtocol.SAMPLER_IDENTITY,
                        XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
        checkpoint.requirePopulationStoreGenerationAuthority(
                "population-store-validated-generation", shaA, 1024L, shaB, shaA,
                shaB, shaA, shaA, 1024L, shaB, shaA, shaB, shaA,
                "environment-randomization-v2", XMageRallyBridgeProtocol.SAMPLER_IDENTITY,
                XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
        boolean rejected = false;
        try {
            checkpoint.requirePopulationStoreGenerationAuthority(
                    "population-store-validated-generation", shaA, 1024L, shaB, shaA,
                    shaB, shaA, shaB, 1024L, shaB, shaA, shaB, shaA,
                    "environment-randomization-v2", XMageRallyBridgeProtocol.SAMPLER_IDENTITY,
                    XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        require(rejected, "stale loaded run identity was accepted");
    }

    private static void testPopulationStoreProperties() {
        String[] names = {
                XMageRallyBridgeProcessClient.POPULATION_STORE_AUTHORITY_KIND_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_RUN_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_GENERATION_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_CHECKPOINT_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_SIDECAR_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_PAYLOAD_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SOURCE_TRAIN_STATE_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_LOADED_RUN_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_LOADED_GENERATION_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_LOADED_CHECKPOINT_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_LOADED_PAYLOAD_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_LOADED_TRAIN_STATE_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_MODEL_PARAMETER_SHA256_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_ENVIRONMENT_TRAJECTORY_CONTRACT_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SAMPLER_IDENTITY_PROPERTY,
                XMageRallyBridgeProcessClient.POPULATION_STORE_SAMPLER_CONTRACT_SHA256_PROPERTY,
        };
        try {
            System.setProperty(names[0], "population-store-validated-generation");
            for (int index = 1; index <= 12; index++) {
                System.setProperty(names[index], index == 2 || index == 8 ? "1024"
                        : "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            }
            System.setProperty(names[13], "environment-randomization-v2");
            System.setProperty(names[14], XMageRallyBridgeProtocol.SAMPLER_IDENTITY);
            System.setProperty(names[15], XMageRallyBridgeProtocol.SAMPLER_CONTRACT_SHA256);
            XMageRallyBridgeProcessClient.validatePopulationStorePropertiesForTest(1024L);
            System.setProperty(names[7], "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
            boolean rejected = false;
            try {
                XMageRallyBridgeProcessClient.validatePopulationStorePropertiesForTest(1024L);
            } catch (IllegalArgumentException expected) {
                rejected = true;
            }
            require(rejected, "uppercase stale population hash was accepted");
        } finally {
            for (String name : names) {
                System.clearProperty(name);
            }
        }
    }

    private static void testFullFlow() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start("flow", 2_000L,
                     1_048_576, diagnostics)) {
            XMageRallyBridgeProtocol.Response reset =
                    client.reset("reset", EPISODE, BASE_SEED);
            require(reset.getBody() instanceof XMageRallyBridgeProtocol.DecisionResponseBody,
                    "reset did not return decision");
            XMageRallyBridgeProtocol.DecisionBody first = client.getCurrentDecision();
            require(first != null && first.getStep() == 0L, "reset decision missing");
            require(first.getKernelClock() != null
                            && first.getKernelClock().getTurn() == 1L
                            && first.getKernelClock().getPhaseStep()
                            == XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1
                            && first.getKernelClock().getActivePlayer()
                            == XMageRallyBridgeProtocol.Seat.P0
                            && first.getKernelClock().getPriorityPlayer()
                            == XMageRallyBridgeProtocol.Seat.P0
                            && first.getKernelClock().getStackDepth() == 0L,
                    "reset decision kernel_clock changed");
            require(first.getSelectedActionIndex() == 1, "Rust selection was not exposed");
            require(first.getActionSemantics().size() == 2, "semantic width mismatch");
            require("pass".equals(first.getActionSemantics().get(0).getActionKind())
                            && "play_land".equals(
                            first.getActionSemantics().get(1).getActionKind()),
                    "action semantic order changed");
            require(first.getActionSemantics().get(1).getCanonicalJson()
                            .contains("\"arena_id\":10"),
                    "object binding was not preserved");
            require(client.getInitialLibraryCardDefinitionIds() != null
                            && client.getInitialLibraryCardDefinitionIds().size() == 2
                            && client.getInitialLibraryCardDefinitionIds().get(0).size() == 60
                            && client.getInitialLibraryCardDefinitionIds().get(1).get(59) == 160,
                    "initial Rally library order was not retained");
            require("6683b059f7fd4a71".equals(client.getPairEnvironmentSeedU64Hex()),
                    "pair environment seed was not exposed");

            client.scoreCurrent("score", EPISODE, 0L);
            require(first.sameCurrentDecision(client.getCurrentDecision()),
                    "score_current was not bit/semantic stable");

            XMageRallyBridgeProtocol.Response stepped =
                    client.step("step-0", EPISODE, 0L, 1, clockFor(0L));
            XMageRallyBridgeProtocol.DecisionResponseBody steppedBody =
                    (XMageRallyBridgeProtocol.DecisionResponseBody) stepped.getBody();
            require(steppedBody.getAppliedAction() != null
                            && steppedBody.getAppliedAction().getSelectedIndex() == 1
                            && steppedBody.getAppliedAction().getSemantic().equals(
                            first.getActionSemantics().get(1)),
                    "applied action lost semantic/commitment binding");
            require(client.getCurrentDecision().getStep() == 1L
                            && client.getCurrentDecision().getActingPlayer()
                            == XMageRallyBridgeProtocol.Seat.P1
                            && client.getCurrentDecision().getSelectedActionIndex() == null,
                    "first transition state is wrong");
            require(client.getCurrentDecision().getKernelClock().getActivePlayer()
                            == XMageRallyBridgeProtocol.Seat.P1,
                    "post-step kernel_clock active seat changed");

            XMageRallyBridgeProtocol.Response terminal =
                    client.step("step-1", EPISODE, 1L, 0, clockFor(1L));
            require(terminal.getBody() instanceof XMageRallyBridgeProtocol.TerminalResponseBody,
                    "second step did not return terminal");
            require(client.getCurrentDecision() == null && client.getTerminal() != null,
                    "terminal/current state was not switched");
            require(client.getTerminal().getTerminal().getWinner()
                            == XMageRallyBridgeProtocol.Seat.P0
                            && client.getTerminal().getTerminal().getPolicyStepCount() == 2L,
                    "typed terminal body changed");
        }
        require(diagnosticBytes.size() == 0,
                "healthy flow emitted diagnostics: " + utf8(diagnosticBytes));
    }

    private static void testTransitionMismatch() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start("transition_mismatch", 2_000L,
                     1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            boolean threw = false;
            try {
                client.step("step-0", EPISODE, 0L, 1, clockFor(0L));
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                threw = true;
            }
            require(threw && !client.isUsable(),
                    "applied commitment mismatch did not fail closed");
        }
    }

    private static void testSequentialEpisodeReset() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start("multi_episode", 2_000L,
                     1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            client.step("step-0", EPISODE, 0L, 1, clockFor(0L));
            client.step("step-1", EPISODE, 1L, 0, clockFor(1L));
            require(client.getTerminal() != null, "first episode did not terminate");
            client.reset("reset-next", 4L, BASE_SEED);
            require(Long.valueOf(4L).equals(client.getActiveEpisodeId())
                            && client.getTerminal() == null
                            && client.getCurrentDecision() != null
                            && client.getCurrentDecision().getEpisodeId() == 4L
                            && "163d0eb1c4158f47".equals(
                            client.getPairEnvironmentSeedU64Hex()),
                    "next episode did not replace terminal binding");
        }
        require(diagnosticBytes.size() == 0,
                "sequential reset emitted diagnostics: " + utf8(diagnosticBytes));
    }

    private static void testEpisodeSplice() throws Exception {
        expectLocalFailure(client -> client.scoreCurrent("score", EPISODE + 1L, 0L));
    }

    private static void testStepSplice() throws Exception {
        expectLocalFailure(client -> client.step(
                "step-1", EPISODE, 1L, 1, clockFor(1L)));
    }

    private static void testRequestIdReuse() throws Exception {
        expectLocalFailure(client -> client.scoreCurrent("reset", EPISODE, 0L));
    }

    private static void testSecondReset() throws Exception {
        expectLocalFailure(client -> client.reset("reset-2", EPISODE, BASE_SEED));
    }

    private static void expectLocalFailure(ClientAction action) throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start("flow", 2_000L,
                     1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            boolean threw = false;
            try {
                action.run(client);
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                threw = true;
            }
            require(threw && !client.isUsable(), "local splice/reuse did not fail closed");
        }
        require(utf8(diagnosticBytes).contains("[xmage-rally-bridge failure]"),
                "local failure was not diagnosed");
    }

    private static void expectResetFailure(String mode,
                                           long timeoutMillis,
                                           int maxLineBytes) throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start(
                     mode, timeoutMillis, maxLineBytes, diagnostics)) {
            boolean threw = false;
            try {
                client.reset("reset", EPISODE, BASE_SEED);
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                threw = true;
            }
            require(threw, "mode did not fail: " + mode);
            require(!client.isUsable(), "failed client remained usable: " + mode);
            boolean secondThrew = false;
            try {
                client.reset("again", EPISODE, BASE_SEED);
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                secondThrew = true;
            }
            require(secondThrew, "failed client accepted another request: " + mode);
        }
        require(utf8(diagnosticBytes).contains("[xmage-rally-bridge failure]"),
                "failure was not diagnosed: " + mode);
    }

    private static void testStderrIsolation() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start(
                     "stderr", 2_000L, 1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            require(client.getCurrentDecision() != null, "stderr corrupted stdout protocol");
        }
        require(utf8(diagnosticBytes).contains("peer diagnostic sentinel"),
                "child stderr was not forwarded to diagnostics");
    }

    private static void expectProtocolFailure(String json) throws Exception {
        boolean threw = false;
        try {
            CODEC.decodeRequest(json);
        } catch (XMageRallyBridgeJsonCodec.ProtocolException expected) {
            threw = true;
        }
        require(threw, "invalid request JSON was accepted: " + json);
    }

    private static void testScoreClockSplice() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start(
                     "score_clock_drift", 2_000L, 1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            boolean threw = false;
            try {
                client.scoreCurrent("score", EPISODE, 0L);
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                threw = true;
            }
            require(threw && !client.isUsable(),
                    "score_current accepted a clock-only decision splice");
        }
    }

    private static void testClockMismatchError() throws Exception {
        ByteArrayOutputStream diagnosticBytes = new ByteArrayOutputStream();
        try (PrintStream diagnostics = diagnostics(diagnosticBytes);
             XMageRallyBridgeProcessClient client = start(
                     "clock_mismatch_error", 2_000L, 1_048_576, diagnostics)) {
            client.reset("reset", EPISODE, BASE_SEED);
            boolean threw = false;
            String failureMessage = null;
            try {
                client.step("step-0", EPISODE, 0L, 1, clockFor(0L));
            } catch (XMageRallyBridgeProcessClient.BridgeFailure expected) {
                threw = true;
                failureMessage = expected.getMessage();
            }
            require(threw && !client.isUsable(),
                    "typed clock_mismatch did not fail closed");
            require(failureMessage != null
                            && failureMessage.contains(
                            "XMAGE_RALLY_SCORER_ERROR error_code=clock_mismatch")
                            && !failureMessage.contains("invalid bridge response"),
                    "typed clock_mismatch did not retain its scorer marker");
        }
        require(utf8(diagnosticBytes).contains("clock_mismatch"),
                "typed clock_mismatch marker was lost from diagnostics");
    }

    private static XMageRallyBridgeProtocol.ExpectedClock clockFor(long step) {
        return new XMageRallyBridgeProtocol.ExpectedClock(
                1L,
                XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1,
                step == 0L ? XMageRallyBridgeProtocol.Seat.P0
                        : XMageRallyBridgeProtocol.Seat.P1);
    }

    private static XMageRallyBridgeProcessClient start(String mode,
                                                        long timeoutMillis,
                                                        int maxLineBytes,
                                                        PrintStream diagnostics)
            throws Exception {
        return XMageRallyBridgeProcessClient.start(
                peerCommand(mode), timeoutMillis, maxLineBytes, diagnostics);
    }

    private static List<String> peerCommand(String mode) throws Exception {
        boolean windows = System.getProperty("os.name", "")
                .toLowerCase(java.util.Locale.ROOT).contains("win");
        Path java = Paths.get(System.getProperty("java.home"), "bin",
                windows ? "java.exe" : "java");
        URI codeSource = XMageRallyBridgeSelfTestPeer.class
                .getProtectionDomain().getCodeSource().getLocation().toURI();
        URI gsonSource = com.google.gson.JsonElement.class
                .getProtectionDomain().getCodeSource().getLocation().toURI();
        String classPath = new File(codeSource).getAbsolutePath()
                + File.pathSeparator + new File(gsonSource).getAbsolutePath();
        return Arrays.asList(
                java.toString(), "-cp", classPath,
                XMageRallyBridgeSelfTestPeer.class.getName(), mode);
    }

    private static PrintStream diagnostics(ByteArrayOutputStream bytes) throws Exception {
        return new PrintStream(bytes, true, StandardCharsets.UTF_8.name());
    }

    private static String utf8(ByteArrayOutputStream bytes) {
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    private static void run(String name, CheckedRunnable test) {
        try {
            test.run();
            passed++;
            System.out.println("  PASS  " + name);
        } catch (Throwable error) {
            failed++;
            System.err.println("  FAIL  " + name + " : "
                    + error.getClass().getSimpleName() + ": " + error.getMessage());
            error.printStackTrace(System.err);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    @FunctionalInterface
    private interface CheckedRunnable {
        void run() throws Exception;
    }

    @FunctionalInterface
    private interface ClientAction {
        void run(XMageRallyBridgeProcessClient client) throws Exception;
    }
}
