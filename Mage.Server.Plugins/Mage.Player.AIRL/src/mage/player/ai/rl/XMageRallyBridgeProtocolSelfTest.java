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
                new XMageRallyBridgeProtocol.StepRequest("step-0", EPISODE, 0L, 1);
        XMageRallyBridgeProtocol.Request decodedStep = CODEC.decodeRequest(
                CODEC.encodeRequest(step));
        require(decodedStep instanceof XMageRallyBridgeProtocol.StepRequest
                        && ((XMageRallyBridgeProtocol.StepRequest) decodedStep)
                        .getSelectedIndex() == 1,
                "step round-trip failed");
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
                    client.step("step-0", EPISODE, 0L, 1);
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

            XMageRallyBridgeProtocol.Response terminal =
                    client.step("step-1", EPISODE, 1L, 0);
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
                client.step("step-0", EPISODE, 0L, 1);
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
            client.step("step-0", EPISODE, 0L, 1);
            client.step("step-1", EPISODE, 1L, 0);
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
        expectLocalFailure(client -> client.step("step-1", EPISODE, 1L, 1));
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
