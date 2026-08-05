package mage.player.ai.rl;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Stateful, fail-closed process client for one exact Rally shadow episode.
 *
 * <p>The first command element must be the final long-lived executable. Shell,
 * Cargo, WSL, and scripting wrappers are rejected because Java 8 can reliably
 * terminate only its direct child. Build the Rust scorer first and launch that
 * executable directly.
 */
public final class XMageRallyBridgeProcessClient implements Closeable {

    public static final int DEFAULT_MAX_LINE_BYTES = 1_048_576;
    public static final long DEFAULT_EXCHANGE_TIMEOUT_MILLIS = 30_000L;
    public static final String CP7_BEHAVIOR_CLONE_ADAM_STEP_PROPERTY =
            "xmage.rally.cp7BehaviorClone.adamStep";
    public static final String CP7_BEHAVIOR_CLONE_MANIFEST_SHA256_PROPERTY =
            "xmage.rally.cp7BehaviorClone.manifestSha256";
    public static final String CP7_BEHAVIOR_CLONE_PAYLOAD_SHA256_PROPERTY =
            "xmage.rally.cp7BehaviorClone.payloadSha256";
    public static final String CP7_BEHAVIOR_CLONE_TRAIN_STATE_SHA256_PROPERTY =
            "xmage.rally.cp7BehaviorClone.trainStateSha256";
    public static final String CP7_BEHAVIOR_CLONE_MODEL_PARAMETER_SHA256_PROPERTY =
            "xmage.rally.cp7BehaviorClone.modelParameterSha256";
    public static final String XMAGE_CP7_OUTCOME_ADAM_STEP_PROPERTY =
            "xmage.rally.cp7Outcome.adamStep";
    public static final String XMAGE_CP7_OUTCOME_AUTHORITY_KIND_PROPERTY =
            "xmage.rally.cp7Outcome.authorityKind";
    public static final String XMAGE_CP7_OUTCOME_MANIFEST_SHA256_PROPERTY =
            "xmage.rally.cp7Outcome.manifestSha256";
    public static final String XMAGE_CP7_OUTCOME_PAYLOAD_SHA256_PROPERTY =
            "xmage.rally.cp7Outcome.payloadSha256";
    public static final String XMAGE_CP7_OUTCOME_TRAIN_STATE_SHA256_PROPERTY =
            "xmage.rally.cp7Outcome.trainStateSha256";
    public static final String XMAGE_CP7_OUTCOME_MODEL_PARAMETER_SHA256_PROPERTY =
            "xmage.rally.cp7Outcome.modelParameterSha256";
    public static final String XMAGE_CP7_OUTCOME_ENVIRONMENT_TRAJECTORY_CONTRACT_PROPERTY =
            "xmage.rally.cp7Outcome.environmentTrajectoryContract";

    private static final int MIN_MAX_LINE_BYTES = 128;
    private static final int MAX_MAX_LINE_BYTES = 16 * 1_048_576;
    private static final int RUST_MAX_REQUEST_BYTES = 1_048_576;
    private static final long MAX_EXCHANGE_TIMEOUT_MILLIS = 10 * 60_000L;
    private static final AtomicLong CLIENT_IDS = new AtomicLong();
    private static final Set<String> FORBIDDEN_WRAPPERS = new HashSet<>(Arrays.asList(
            "cargo", "cargo.exe", "rustup", "rustup.exe",
            "cmd", "cmd.exe", "command.com", "powershell", "powershell.exe",
            "pwsh", "pwsh.exe", "sh", "sh.exe", "bash", "bash.exe",
            "wsl", "wsl.exe", "python", "python.exe", "python3", "python3.exe"));

    private final Process process;
    private final OutputStream requestOutput;
    private final BoundedUtf8LineReader responseReader;
    private final XMageRallyBridgeJsonCodec codec;
    private final long exchangeTimeoutMillis;
    private final int maxLineBytes;
    private final PrintStream diagnostics;
    private final ExecutorService ioExecutor;
    private final Thread stderrThread;
    private final Long expectedCheckpointGeneration;
    private final Cp7BehaviorCloneExpectation expectedCp7BehaviorClone;
    private final XMageCp7OutcomeExpectation expectedXMageCp7Outcome;
    private final Set<String> usedRequestIds = new HashSet<>();
    private final AtomicBoolean resourcesClosed = new AtomicBoolean();
    private final Object failureLock = new Object();

    private volatile boolean failed;
    private volatile boolean closed;
    private volatile String firstFailure;
    private volatile Long activeEpisodeId;
    private volatile Long activeBaseSeed;
    private volatile String pairEnvironmentSeedU64Hex;
    private volatile XMageRallyBridgeProtocol.Seat candidateSeat;
    private volatile XMageRallyBridgeProtocol.DecisionBody currentDecision;
    private volatile XMageRallyBridgeProtocol.TerminalBody terminal;
    private volatile List<List<Integer>> initialLibraryCardDefinitionIds;

    private XMageRallyBridgeProcessClient(Process process,
                                          long exchangeTimeoutMillis,
                                          int maxLineBytes,
                                          PrintStream diagnostics,
                                          long clientId,
                                          Long expectedCheckpointGeneration,
                                          Cp7BehaviorCloneExpectation expectedCp7BehaviorClone,
                                          XMageCp7OutcomeExpectation expectedXMageCp7Outcome) {
        this.process = process;
        this.requestOutput = process.getOutputStream();
        this.responseReader = new BoundedUtf8LineReader(process.getInputStream(), maxLineBytes);
        this.codec = new XMageRallyBridgeJsonCodec();
        this.exchangeTimeoutMillis = exchangeTimeoutMillis;
        this.maxLineBytes = maxLineBytes;
        this.diagnostics = diagnostics;
        this.expectedCheckpointGeneration = expectedCheckpointGeneration;
        this.expectedCp7BehaviorClone = expectedCp7BehaviorClone;
        this.expectedXMageCp7Outcome = expectedXMageCp7Outcome;
        this.ioExecutor = Executors.newSingleThreadExecutor(
                daemonThreadFactory("XMAGE-RALLY-BRIDGE-IO-" + clientId));
        this.stderrThread = startStderrDrainer(
                process.getErrorStream(), diagnostics, clientId);
    }

    public static XMageRallyBridgeProcessClient start(List<String> command) throws IOException {
        return start(command, DEFAULT_EXCHANGE_TIMEOUT_MILLIS,
                DEFAULT_MAX_LINE_BYTES, System.err);
    }

    public static XMageRallyBridgeProcessClient start(List<String> command,
                                                       long exchangeTimeoutMillis,
                                                       int maxLineBytes,
                                                       PrintStream diagnostics)
            throws IOException {
        List<String> checkedCommand = validateDirectCommand(command);
        Long expectedCheckpointGeneration = selectedGeneration(checkedCommand);
        boolean selectsCp7BehaviorClone = selectsCp7BehaviorClone(checkedCommand);
        boolean selectsXMageCp7Outcome = selectsXMageCp7Outcome(checkedCommand);
        if ((selectsCp7BehaviorClone && selectsXMageCp7Outcome)
                || (expectedCheckpointGeneration != null
                && (selectsCp7BehaviorClone || selectsXMageCp7Outcome))) {
            throw new IllegalArgumentException(
                    "bridge generation and derivative selections are mutually exclusive");
        }
        Cp7BehaviorCloneExpectation expectedCp7BehaviorClone = selectsCp7BehaviorClone
                ? Cp7BehaviorCloneExpectation.fromSystemProperties() : null;
        XMageCp7OutcomeExpectation expectedXMageCp7Outcome = selectsXMageCp7Outcome
                ? XMageCp7OutcomeExpectation.fromSystemProperties() : null;
        if (exchangeTimeoutMillis <= 0L
                || exchangeTimeoutMillis > MAX_EXCHANGE_TIMEOUT_MILLIS) {
            throw new IllegalArgumentException(
                    "exchangeTimeoutMillis must be in [1, "
                            + MAX_EXCHANGE_TIMEOUT_MILLIS + "]");
        }
        if (maxLineBytes < MIN_MAX_LINE_BYTES || maxLineBytes > MAX_MAX_LINE_BYTES) {
            throw new IllegalArgumentException(
                    "maxLineBytes must be in [" + MIN_MAX_LINE_BYTES
                            + ", " + MAX_MAX_LINE_BYTES + "]");
        }
        if (diagnostics == null) {
            throw new IllegalArgumentException("diagnostics must not be null");
        }
        ProcessBuilder builder = new ProcessBuilder(checkedCommand);
        builder.redirectErrorStream(false);
        Process process = builder.start();
        return new XMageRallyBridgeProcessClient(
                process, exchangeTimeoutMillis, maxLineBytes,
                diagnostics, CLIENT_IDS.incrementAndGet(), expectedCheckpointGeneration,
                expectedCp7BehaviorClone, expectedXMageCp7Outcome);
    }

    public synchronized XMageRallyBridgeProtocol.Response reset(String requestId,
                                                                 long episodeId,
                                                                 long baseSeed)
            throws BridgeFailure {
        requireUsable();
        if (activeEpisodeId != null && terminal == null) {
            throw fail("reset attempted while an episode is still active", null);
        }
        XMageRallyBridgeProtocol.ResetRequest request;
        try {
            request = new XMageRallyBridgeProtocol.ResetRequest(
                    requestId, episodeId, baseSeed);
        } catch (RuntimeException e) {
            throw fail("invalid reset request", e);
        }
        XMageRallyBridgeProtocol.Response response = exchange(request);
        XMageRallyBridgeProtocol.ResponseBody body = response.getBody();
        try {
            if (body instanceof XMageRallyBridgeProtocol.DecisionResponseBody) {
                XMageRallyBridgeProtocol.DecisionResponseBody decisionBody =
                        (XMageRallyBridgeProtocol.DecisionResponseBody) body;
                if (decisionBody.getAppliedAction() != null) {
                    throw new IllegalArgumentException("reset response contains applied_action");
                }
                XMageRallyBridgeProtocol.DecisionBody decision = decisionBody.getDecision();
                validateResetState(decision, episodeId, baseSeed);
                bindReset(decision, episodeId, baseSeed);
            } else if (body instanceof XMageRallyBridgeProtocol.TerminalResponseBody) {
                XMageRallyBridgeProtocol.TerminalResponseBody terminalBody =
                        (XMageRallyBridgeProtocol.TerminalResponseBody) body;
                if (terminalBody.getAppliedAction() != null) {
                    throw new IllegalArgumentException("reset terminal contains applied_action");
                }
                XMageRallyBridgeProtocol.TerminalBody resetTerminal = terminalBody.getTerminal();
                validateResetState(resetTerminal, episodeId, baseSeed);
                bindReset(resetTerminal, episodeId, baseSeed);
            } else {
                throw new IllegalArgumentException("reset returned non-state body");
            }
        } catch (RuntimeException e) {
            throw fail("invalid reset transition", e);
        }
        return response;
    }

    public synchronized XMageRallyBridgeProtocol.Response scoreCurrent(String requestId,
                                                                        long episodeId,
                                                                        long expectedStep)
            throws BridgeFailure {
        requireActiveDecision(episodeId, expectedStep, "score_current");
        XMageRallyBridgeProtocol.ScoreCurrentRequest request;
        try {
            request = new XMageRallyBridgeProtocol.ScoreCurrentRequest(
                    requestId, episodeId, expectedStep);
        } catch (RuntimeException e) {
            throw fail("invalid score_current request", e);
        }
        XMageRallyBridgeProtocol.DecisionBody before = currentDecision;
        XMageRallyBridgeProtocol.Response response = exchange(request);
        try {
            if (!(response.getBody() instanceof XMageRallyBridgeProtocol.DecisionResponseBody)) {
                throw new IllegalArgumentException("score_current did not return decision");
            }
            XMageRallyBridgeProtocol.DecisionResponseBody body =
                    (XMageRallyBridgeProtocol.DecisionResponseBody) response.getBody();
            if (body.getAppliedAction() != null) {
                throw new IllegalArgumentException("score_current contains applied_action");
            }
            XMageRallyBridgeProtocol.DecisionBody scored = body.getDecision();
            validateBoundState(scored, false);
            if (!before.sameCurrentDecision(scored)) {
                throw new IllegalArgumentException("score_current changed the active decision");
            }
            currentDecision = scored;
        } catch (RuntimeException e) {
            throw fail("invalid score_current response", e);
        }
        return response;
    }

    public synchronized XMageRallyBridgeProtocol.Response step(String requestId,
                                                                long episodeId,
                                                                long expectedStep,
                                                                int selectedIndex)
            throws BridgeFailure {
        requireActiveDecision(episodeId, expectedStep, "step");
        XMageRallyBridgeProtocol.DecisionBody before = currentDecision;
        if (selectedIndex < 0 || selectedIndex >= before.getLegalActionCount()) {
            throw fail("selected_index is outside the active decision", null);
        }
        if (before.isCandidateControlsCurrentActor()
                && !Integer.valueOf(selectedIndex).equals(before.getSelectedActionIndex())) {
            throw fail("candidate step does not equal Rust-side sampled action", null);
        }
        XMageRallyBridgeProtocol.StepRequest request;
        try {
            request = new XMageRallyBridgeProtocol.StepRequest(
                    requestId, episodeId, expectedStep, selectedIndex);
        } catch (RuntimeException e) {
            throw fail("invalid step request", e);
        }
        XMageRallyBridgeProtocol.Response response = exchange(request);
        try {
            XMageRallyBridgeProtocol.AppliedAction applied;
            if (response.getBody() instanceof XMageRallyBridgeProtocol.DecisionResponseBody) {
                XMageRallyBridgeProtocol.DecisionResponseBody body =
                        (XMageRallyBridgeProtocol.DecisionResponseBody) response.getBody();
                applied = body.getAppliedAction();
                validateApplied(before, applied, selectedIndex);
                XMageRallyBridgeProtocol.DecisionBody after = body.getDecision();
                validateBoundState(after, false);
                validateDecisionAdvance(before, after);
                currentDecision = after;
                terminal = null;
            } else if (response.getBody()
                    instanceof XMageRallyBridgeProtocol.TerminalResponseBody) {
                XMageRallyBridgeProtocol.TerminalResponseBody body =
                        (XMageRallyBridgeProtocol.TerminalResponseBody) response.getBody();
                applied = body.getAppliedAction();
                validateApplied(before, applied, selectedIndex);
                XMageRallyBridgeProtocol.TerminalBody after = body.getTerminal();
                validateBoundState(after, false);
                validateTerminalAdvance(before, after);
                currentDecision = null;
                terminal = after;
            } else {
                throw new IllegalArgumentException("step returned non-state body");
            }
        } catch (RuntimeException e) {
            throw fail("invalid step transition", e);
        }
        return response;
    }

    public boolean isUsable() {
        return !closed && !failed && process.isAlive();
    }

    public Long getActiveEpisodeId() {
        return activeEpisodeId;
    }

    public XMageRallyBridgeProtocol.DecisionBody getCurrentDecision() {
        return currentDecision;
    }

    public XMageRallyBridgeProtocol.TerminalBody getTerminal() {
        return terminal;
    }

    public String getPairEnvironmentSeedU64Hex() {
        return pairEnvironmentSeedU64Hex;
    }

    public XMageRallyBridgeProtocol.Seat getCandidateSeat() {
        return candidateSeat;
    }

    public List<List<Integer>> getInitialLibraryCardDefinitionIds() {
        return initialLibraryCardDefinitionIds;
    }

    @Override
    public void close() {
        closed = true;
        closeProcessResources();
    }

    private XMageRallyBridgeProtocol.Response exchange(
            XMageRallyBridgeProtocol.Request request) throws BridgeFailure {
        requireUsable();
        if (!usedRequestIds.add(request.getRequestId())) {
            throw fail("request_id was reused: " + request.getRequestId(), null);
        }
        if (!process.isAlive()) {
            throw fail("bridge process exited before request (exit="
                    + exitDescription() + ")", null);
        }
        String encoded;
        try {
            encoded = codec.encodeRequest(request);
        } catch (RuntimeException e) {
            throw fail("failed to encode bridge request", e);
        }
        byte[] requestBytes = encoded.getBytes(StandardCharsets.UTF_8);
        if (requestBytes.length > maxLineBytes || requestBytes.length > RUST_MAX_REQUEST_BYTES) {
            throw fail("encoded request exceeds the fixed byte limit", null);
        }

        Future<String> io;
        try {
            io = ioExecutor.submit(() -> {
                requestOutput.write(requestBytes);
                requestOutput.write('\n');
                requestOutput.flush();
                return responseReader.readLine();
            });
        } catch (RuntimeException e) {
            throw fail("could not schedule bridge exchange", e);
        }
        String line;
        try {
            line = io.get(exchangeTimeoutMillis, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            io.cancel(true);
            throw fail("bridge exchange timed out after "
                    + exchangeTimeoutMillis + " ms", e);
        } catch (InterruptedException e) {
            io.cancel(true);
            Thread.currentThread().interrupt();
            throw fail("interrupted while waiting for bridge exchange", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw fail("bridge exchange I/O failed", cause);
        }

        XMageRallyBridgeProtocol.Response response;
        try {
            response = codec.decodeResponse(line);
            if (!request.getRequestId().equals(response.getRequestId())) {
                throw new IllegalArgumentException("response request_id does not echo request");
            }
            if (expectedXMageCp7Outcome != null) {
                expectedXMageCp7Outcome.require(response.getCheckpoint());
            } else if (expectedCp7BehaviorClone != null) {
                expectedCp7BehaviorClone.require(response.getCheckpoint());
            } else if (expectedCheckpointGeneration == null) {
                response.getCheckpoint().requireExactOriginalAuthority();
            } else {
                response.getCheckpoint().requireSelectedOriginalGeneration(
                        expectedCheckpointGeneration);
            }
            if (response.getBody() instanceof XMageRallyBridgeProtocol.ErrorResponseBody) {
                XMageRallyBridgeProtocol.ErrorResponseBody error =
                        (XMageRallyBridgeProtocol.ErrorResponseBody) response.getBody();
                throw new IllegalArgumentException(
                        "Rust error " + error.getErrorCode() + ": " + error.getMessage());
            }
        } catch (IOException | RuntimeException e) {
            throw fail("invalid bridge response", e);
        }
        return response;
    }

    private void bindReset(XMageRallyBridgeProtocol.DecisionBody decision,
                           long episodeId,
                           long baseSeed) {
        activeEpisodeId = episodeId;
        activeBaseSeed = baseSeed;
        pairEnvironmentSeedU64Hex = decision.getPairEnvironmentSeedU64Hex();
        candidateSeat = decision.getCandidateSeat();
        initialLibraryCardDefinitionIds = decision.getInitialLibraryCardDefinitionIds();
        currentDecision = decision;
        terminal = null;
    }

    private void bindReset(XMageRallyBridgeProtocol.TerminalBody resetTerminal,
                           long episodeId,
                           long baseSeed) {
        activeEpisodeId = episodeId;
        activeBaseSeed = baseSeed;
        pairEnvironmentSeedU64Hex = resetTerminal.getPairEnvironmentSeedU64Hex();
        candidateSeat = resetTerminal.getCandidateSeat();
        initialLibraryCardDefinitionIds =
                resetTerminal.getInitialLibraryCardDefinitionIds();
        currentDecision = null;
        terminal = resetTerminal;
    }

    private static void validateResetState(XMageRallyBridgeProtocol.DecisionBody decision,
                                           long episodeId,
                                           long baseSeed) {
        validateRallyDecks(decision.getDeckIds());
        if (decision.getEpisodeId() != episodeId || decision.getStep() != 0L
                || !XMageRallyBridgeProtocol.u64Hex(baseSeed)
                .equals(decision.getBaseSeedU64Hex())
                || decision.getPairIndex() != episodeId / 2L
                || !XMageRallyBridgeProtocol.derivePairEnvironmentSeedU64Hex(
                baseSeed, episodeId).equals(decision.getPairEnvironmentSeedU64Hex())
                || decision.getCandidateSeat() != expectedCandidateSeat(episodeId)
                || decision.getInitialLibraryCardDefinitionIds() == null) {
            throw new IllegalArgumentException("reset decision metadata mismatch");
        }
    }

    private static void validateResetState(XMageRallyBridgeProtocol.TerminalBody terminal,
                                           long episodeId,
                                           long baseSeed) {
        validateRallyDecks(terminal.getDeckIds());
        if (terminal.getTerminal().getEpisodeId() != episodeId
                || !XMageRallyBridgeProtocol.u64Hex(baseSeed)
                .equals(terminal.getBaseSeedU64Hex())
                || terminal.getPairIndex() != episodeId / 2L
                || !XMageRallyBridgeProtocol.derivePairEnvironmentSeedU64Hex(
                baseSeed, episodeId).equals(terminal.getPairEnvironmentSeedU64Hex())
                || terminal.getCandidateSeat() != expectedCandidateSeat(episodeId)
                || terminal.getInitialLibraryCardDefinitionIds() == null) {
            throw new IllegalArgumentException("reset terminal metadata mismatch");
        }
    }

    private void validateBoundState(XMageRallyBridgeProtocol.DecisionBody decision,
                                    boolean allowInitialLibraries) {
        validateRallyDecks(decision.getDeckIds());
        if (activeEpisodeId == null || decision.getEpisodeId() != activeEpisodeId
                || !XMageRallyBridgeProtocol.u64Hex(activeBaseSeed)
                .equals(decision.getBaseSeedU64Hex())
                || decision.getPairIndex() != activeEpisodeId / 2L
                || !pairEnvironmentSeedU64Hex.equals(
                decision.getPairEnvironmentSeedU64Hex())
                || decision.getCandidateSeat() != candidateSeat
                || (!allowInitialLibraries
                && decision.getInitialLibraryCardDefinitionIds() != null)) {
            throw new IllegalArgumentException("decision spliced or changed episode metadata");
        }
    }

    private void validateBoundState(XMageRallyBridgeProtocol.TerminalBody terminal,
                                    boolean allowInitialLibraries) {
        validateRallyDecks(terminal.getDeckIds());
        if (activeEpisodeId == null
                || terminal.getTerminal().getEpisodeId() != activeEpisodeId
                || !XMageRallyBridgeProtocol.u64Hex(activeBaseSeed)
                .equals(terminal.getBaseSeedU64Hex())
                || terminal.getPairIndex() != activeEpisodeId / 2L
                || !pairEnvironmentSeedU64Hex.equals(
                terminal.getPairEnvironmentSeedU64Hex())
                || terminal.getCandidateSeat() != candidateSeat
                || (!allowInitialLibraries
                && terminal.getInitialLibraryCardDefinitionIds() != null)) {
            throw new IllegalArgumentException("terminal spliced or changed episode metadata");
        }
    }

    private static void validateApplied(XMageRallyBridgeProtocol.DecisionBody before,
                                        XMageRallyBridgeProtocol.AppliedAction applied,
                                        int selectedIndex) {
        if (applied == null
                || applied.getEpisodeId() != before.getEpisodeId()
                || applied.getStep() != before.getStep()
                || applied.getSelectedIndex() != selectedIndex
                || !applied.getCandidateOrderCommitment128Hex().equals(
                before.getCandidateOrderCommitment128Hex())
                || !applied.getModelInputSha256().equals(before.getModelInputSha256())
                || applied.getSelectedLogitF32Bits()
                != before.getLogitsF32Bits().get(selectedIndex)
                || !applied.getSemantic().equals(
                before.getActionSemantics().get(selectedIndex))) {
            throw new IllegalArgumentException("applied_action does not bind the prior decision");
        }
    }

    private static void validateDecisionAdvance(XMageRallyBridgeProtocol.DecisionBody before,
                                                XMageRallyBridgeProtocol.DecisionBody after) {
        if (before.getStep() == Long.MAX_VALUE
                || before.getEnvironmentRevision() == Long.MAX_VALUE
                || after.getStep() != before.getStep() + 1L
                || after.getEnvironmentRevision() != before.getEnvironmentRevision() + 1L) {
            throw new IllegalArgumentException("decision step/revision did not advance exactly");
        }
        boolean completesPhysical = before.getSubstepIndex() + 1 == before.getSubstepCount();
        if (completesPhysical) {
            if (before.getPhysicalDecisionId() == Long.MAX_VALUE
                    || after.getPhysicalDecisionId() != before.getPhysicalDecisionId() + 1L
                    || after.getSubstepIndex() != 0) {
                throw new IllegalArgumentException("physical decision transition mismatch");
            }
        } else if (after.getPhysicalDecisionId() != before.getPhysicalDecisionId()
                || after.getSubstepIndex() != before.getSubstepIndex() + 1
                || after.getSubstepCount() != before.getSubstepCount()) {
            throw new IllegalArgumentException("decision substep transition mismatch");
        }
    }

    private static void validateTerminalAdvance(XMageRallyBridgeProtocol.DecisionBody before,
                                                XMageRallyBridgeProtocol.TerminalBody after) {
        XMageRallyBridgeProtocol.TerminalRecord record = after.getTerminal();
        boolean completesPhysical = before.getSubstepIndex() + 1 == before.getSubstepCount();
        long expectedPhysical = before.getPhysicalDecisionId() + (completesPhysical ? 1L : 0L);
        if (before.getStep() == Long.MAX_VALUE
                || record.getPolicyStepCount() != before.getStep() + 1L
                || record.getPhysicalDecisionCount() != expectedPhysical) {
            throw new IllegalArgumentException("terminal counts do not bind the applied action");
        }
    }

    private void requireActiveDecision(long episodeId,
                                       long expectedStep,
                                       String operation) throws BridgeFailure {
        requireUsable();
        XMageRallyBridgeProtocol.DecisionBody decision = currentDecision;
        if (activeEpisodeId == null || terminal != null || decision == null) {
            throw fail(operation + " requires one active nonterminal episode", null);
        }
        if (episodeId != activeEpisodeId || expectedStep != decision.getStep()) {
            throw fail(operation + " attempted an episode or step splice", null);
        }
    }

    private static void validateRallyDecks(List<String> deckIds) {
        if (deckIds == null || deckIds.size() != 2
                || !XMageRallyBridgeProtocol.RALLY_DECK_ID.equals(deckIds.get(0))
                || !XMageRallyBridgeProtocol.RALLY_DECK_ID.equals(deckIds.get(1))) {
            throw new IllegalArgumentException("shadow response is not Rally versus Rally");
        }
    }

    private static XMageRallyBridgeProtocol.Seat expectedCandidateSeat(long episodeId) {
        return episodeId % 2L == 0L
                ? XMageRallyBridgeProtocol.Seat.P0
                : XMageRallyBridgeProtocol.Seat.P1;
    }

    private void requireUsable() throws BridgeFailure {
        if (closed) {
            throw new BridgeFailure("bridge client is closed");
        }
        if (failed) {
            throw new BridgeFailure("bridge client is failed closed after: " + firstFailure);
        }
    }

    private BridgeFailure fail(String message, Throwable cause) {
        boolean diagnose = false;
        synchronized (failureLock) {
            if (!failed) {
                failed = true;
                firstFailure = message;
                diagnose = true;
            }
        }
        if (diagnose) {
            diagnostics.println("[xmage-rally-bridge failure] " + message
                    + causeSuffix(cause));
            diagnostics.flush();
            closeProcessResources();
        }
        return cause == null
                ? new BridgeFailure(message)
                : new BridgeFailure(message, cause);
    }

    private void closeProcessResources() {
        if (!resourcesClosed.compareAndSet(false, true)) {
            return;
        }
        ioExecutor.shutdownNow();
        if (process.isAlive()) {
            process.destroy();
            try {
                if (!process.waitFor(250L, TimeUnit.MILLISECONDS)) {
                    process.destroyForcibly();
                    process.waitFor(250L, TimeUnit.MILLISECONDS);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
        }
        // Destroy before closing a possibly blocked pipe. Some Java 8 process
        // output streams flush from close(), which can otherwise wait forever
        // when a wedged child stopped reading a large request.
        closeQuietly(requestOutput);
        closeQuietly(responseReader);
        closeQuietly(process.getErrorStream());
        try {
            stderrThread.join(250L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private String exitDescription() {
        try {
            return Integer.toString(process.exitValue());
        } catch (IllegalThreadStateException e) {
            return "still-running";
        }
    }

    private static List<String> validateDirectCommand(List<String> command) {
        if (command == null || command.isEmpty()) {
            throw new IllegalArgumentException("bridge command must not be empty");
        }
        ArrayList<String> copy = new ArrayList<>(command.size());
        for (String part : command) {
            if (part == null || part.isEmpty()) {
                throw new IllegalArgumentException(
                        "bridge command must not contain null or empty arguments");
            }
            copy.add(part);
        }
        String executable = copy.get(0).replace('\\', '/');
        int slash = executable.lastIndexOf('/');
        String basename = (slash >= 0 ? executable.substring(slash + 1) : executable)
                .toLowerCase(Locale.ROOT);
        if (FORBIDDEN_WRAPPERS.contains(basename)) {
            throw new IllegalArgumentException(
                    "bridge must launch the compiled scorer executable directly, not "
                            + basename);
        }
        return copy;
    }

    private static Long selectedGeneration(List<String> command) {
        Long selected = null;
        for (int index = 1; index < command.size(); index++) {
            if (!"--generation".equals(command.get(index))) {
                continue;
            }
            if (selected != null || index + 1 >= command.size()) {
                throw new IllegalArgumentException("invalid bridge generation selection");
            }
            try {
                selected = Long.parseLong(command.get(++index));
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException("invalid bridge generation selection", error);
            }
            if (selected < 0L) {
                throw new IllegalArgumentException("bridge generation must be nonnegative");
            }
        }
        return selected;
    }

    private static boolean selectsCp7BehaviorClone(List<String> command) {
        boolean selected = false;
        for (int index = 1; index < command.size(); index++) {
            if (!"--cp7-behavior-clone-root".equals(command.get(index))) {
                continue;
            }
            if (selected || index + 1 >= command.size()) {
                throw new IllegalArgumentException(
                        "invalid bridge CP7 behavior-clone selection");
            }
            selected = true;
            index++;
        }
        return selected;
    }

    private static boolean selectsXMageCp7Outcome(List<String> command) {
        boolean selected = false;
        for (int index = 1; index < command.size(); index++) {
            if (!"--xmage-cp7-outcome-root".equals(command.get(index))) {
                continue;
            }
            if (selected || index + 1 >= command.size()) {
                throw new IllegalArgumentException(
                        "invalid bridge XMage CP7 outcome selection");
            }
            selected = true;
            index++;
        }
        return selected;
    }

    private static final class Cp7BehaviorCloneExpectation {
        private final long adamStep;
        private final String manifestSha256;
        private final String payloadSha256;
        private final String trainStateSha256;
        private final String modelParameterSha256;
        private final String environmentTrajectoryContract;

        private Cp7BehaviorCloneExpectation(long adamStep,
                                            String manifestSha256,
                                            String payloadSha256,
                                            String trainStateSha256,
                                            String modelParameterSha256) {
            this.adamStep = adamStep;
            this.manifestSha256 = manifestSha256;
            this.payloadSha256 = payloadSha256;
            this.trainStateSha256 = trainStateSha256;
            this.modelParameterSha256 = modelParameterSha256;
        }

        private static Cp7BehaviorCloneExpectation fromSystemProperties() {
            String adamStep = System.getProperty(CP7_BEHAVIOR_CLONE_ADAM_STEP_PROPERTY);
            String manifest = System.getProperty(CP7_BEHAVIOR_CLONE_MANIFEST_SHA256_PROPERTY);
            String payload = System.getProperty(CP7_BEHAVIOR_CLONE_PAYLOAD_SHA256_PROPERTY);
            String trainState = System.getProperty(
                    CP7_BEHAVIOR_CLONE_TRAIN_STATE_SHA256_PROPERTY);
            String model = System.getProperty(
                    CP7_BEHAVIOR_CLONE_MODEL_PARAMETER_SHA256_PROPERTY);
            int supplied = 0;
            for (String value : Arrays.asList(
                    adamStep, manifest, payload, trainState, model)) {
                if (value != null) {
                    supplied++;
                }
            }
            if (supplied == 0) {
                return new Cp7BehaviorCloneExpectation(
                        XMageRallyBridgeProtocol.CP7_BEHAVIOR_CLONE_ADAM_STEP,
                        XMageRallyBridgeProtocol.CP7_BEHAVIOR_CLONE_MANIFEST_SHA256,
                        XMageRallyBridgeProtocol.CP7_BEHAVIOR_CLONE_PAYLOAD_SHA256,
                        XMageRallyBridgeProtocol.CP7_BEHAVIOR_CLONE_TRAIN_STATE_SHA256,
                        XMageRallyBridgeProtocol.CP7_BEHAVIOR_CLONE_MODEL_PARAMETER_SHA256);
            }
            if (supplied != 5) {
                throw new IllegalArgumentException(
                        "all CP7 behavior-clone identity properties must be supplied together");
            }
            long parsedAdamStep;
            try {
                parsedAdamStep = Long.parseLong(adamStep);
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(
                        "invalid CP7 behavior-clone Adam step property", error);
            }
            if (parsedAdamStep < 0L
                    || !isLowerHexSha256(manifest)
                    || !isLowerHexSha256(payload)
                    || !isLowerHexSha256(trainState)
                    || !isLowerHexSha256(model)) {
                throw new IllegalArgumentException(
                        "invalid CP7 behavior-clone identity properties");
            }
            return new Cp7BehaviorCloneExpectation(
                    parsedAdamStep, manifest, payload, trainState, model);
        }

        private void require(XMageRallyBridgeProtocol.CheckpointIdentity checkpoint) {
            checkpoint.requireCp7BehaviorCloneAuthority(
                    adamStep, manifestSha256, payloadSha256,
                    trainStateSha256, modelParameterSha256);
        }

        private static boolean isLowerHexSha256(String value) {
            if (value == null || value.length() != 64) {
                return false;
            }
            for (int index = 0; index < value.length(); index++) {
                char character = value.charAt(index);
                if (!((character >= '0' && character <= '9')
                        || (character >= 'a' && character <= 'f'))) {
                    return false;
                }
            }
            return true;
        }
    }

    private static final class XMageCp7OutcomeExpectation {
        private final String authorityKind;
        private final long adamStep;
        private final String manifestSha256;
        private final String payloadSha256;
        private final String trainStateSha256;
        private final String modelParameterSha256;

        private XMageCp7OutcomeExpectation(String authorityKind,
                                           long adamStep,
                                           String manifestSha256,
                                           String payloadSha256,
                                           String trainStateSha256,
                                           String modelParameterSha256,
                                           String environmentTrajectoryContract) {
            this.authorityKind = authorityKind;
            this.adamStep = adamStep;
            this.manifestSha256 = manifestSha256;
            this.payloadSha256 = payloadSha256;
            this.trainStateSha256 = trainStateSha256;
            this.modelParameterSha256 = modelParameterSha256;
            this.environmentTrajectoryContract = environmentTrajectoryContract;
        }

        private static XMageCp7OutcomeExpectation fromSystemProperties() {
            String authorityKind = System.getProperty(
                    XMAGE_CP7_OUTCOME_AUTHORITY_KIND_PROPERTY,
                    XMageRallyBridgeProtocol.XMAGE_CP7_OUTCOME_AUTHORITY_KIND);
            String adamStep = System.getProperty(XMAGE_CP7_OUTCOME_ADAM_STEP_PROPERTY);
            String manifest = System.getProperty(XMAGE_CP7_OUTCOME_MANIFEST_SHA256_PROPERTY);
            String payload = System.getProperty(XMAGE_CP7_OUTCOME_PAYLOAD_SHA256_PROPERTY);
            String trainState = System.getProperty(
                    XMAGE_CP7_OUTCOME_TRAIN_STATE_SHA256_PROPERTY);
            String model = System.getProperty(
                    XMAGE_CP7_OUTCOME_MODEL_PARAMETER_SHA256_PROPERTY);
            String environmentTrajectoryContract = System.getProperty(
                    XMAGE_CP7_OUTCOME_ENVIRONMENT_TRAJECTORY_CONTRACT_PROPERTY,
                    XMageRallyBridgeProtocol.ENVIRONMENT_TRAJECTORY_CONTRACT);
            for (String value : Arrays.asList(
                    adamStep, manifest, payload, trainState, model)) {
                if (value == null) {
                    throw new IllegalArgumentException(
                            "all XMage CP7 outcome identity properties are required");
                }
            }
            long parsedAdamStep;
            try {
                parsedAdamStep = Long.parseLong(adamStep);
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(
                        "invalid XMage CP7 outcome Adam step property", error);
            }
            if (parsedAdamStep < 0L
                    || authorityKind.isEmpty()
                    || !Cp7BehaviorCloneExpectation.isLowerHexSha256(manifest)
                    || !Cp7BehaviorCloneExpectation.isLowerHexSha256(payload)
                    || !Cp7BehaviorCloneExpectation.isLowerHexSha256(trainState)
                    || !Cp7BehaviorCloneExpectation.isLowerHexSha256(model)
                    || !(XMageRallyBridgeProtocol.ENVIRONMENT_TRAJECTORY_CONTRACT.equals(
                            environmentTrajectoryContract)
                            || "environment-randomization-v2".equals(
                                    environmentTrajectoryContract))) {
                throw new IllegalArgumentException(
                        "invalid XMage CP7 outcome identity properties");
            }
            return new XMageCp7OutcomeExpectation(
                    authorityKind, parsedAdamStep, manifest, payload, trainState, model,
                    environmentTrajectoryContract);
        }

        private void require(XMageRallyBridgeProtocol.CheckpointIdentity checkpoint) {
            checkpoint.requireXMageCp7OutcomeAuthority(
                    authorityKind, adamStep, manifestSha256, payloadSha256,
                    trainStateSha256, modelParameterSha256,
                    environmentTrajectoryContract);
        }
    }

    private static Thread startStderrDrainer(InputStream stderr,
                                             PrintStream diagnostics,
                                             long clientId) {
        Thread thread = new Thread(() -> {
            byte[] buffer = new byte[4096];
            boolean announced = false;
            try {
                int count;
                while ((count = stderr.read(buffer)) >= 0) {
                    if (count == 0) {
                        continue;
                    }
                    synchronized (diagnostics) {
                        if (!announced) {
                            diagnostics.println("[xmage-rally-bridge child stderr]");
                            announced = true;
                        }
                        diagnostics.write(buffer, 0, count);
                        diagnostics.flush();
                    }
                }
            } catch (IOException e) {
                // Direct-child teardown closes this diagnostic stream normally.
            }
        }, "XMAGE-RALLY-BRIDGE-STDERR-" + clientId);
        thread.setDaemon(true);
        thread.start();
        return thread;
    }

    private static ThreadFactory daemonThreadFactory(String name) {
        return runnable -> {
            Thread thread = new Thread(runnable, name);
            thread.setDaemon(true);
            return thread;
        };
    }

    private static String causeSuffix(Throwable cause) {
        if (cause == null) {
            return "";
        }
        String message = cause.getMessage();
        if (message == null || message.trim().isEmpty()) {
            return " (" + cause.getClass().getSimpleName() + ")";
        }
        return " (" + cause.getClass().getSimpleName() + ": " + message + ")";
    }

    private static void closeQuietly(Closeable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (IOException ignored) {
        }
    }

    public static final class BridgeFailure extends IOException {
        public BridgeFailure(String message) {
            super(message);
        }

        public BridgeFailure(String message, Throwable cause) {
            super(message, cause);
        }
    }

    private static final class BoundedUtf8LineReader implements Closeable {
        private final InputStream input;
        private final int maxLineBytes;

        private BoundedUtf8LineReader(InputStream input, int maxLineBytes) {
            this.input = new BufferedInputStream(input);
            this.maxLineBytes = maxLineBytes;
        }

        private String readLine() throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream(
                    Math.min(maxLineBytes, 8192));
            while (true) {
                int value = input.read();
                if (value < 0) {
                    throw new IOException(bytes.size() == 0
                            ? "EOF before bridge response" : "EOF before LF terminator");
                }
                if (value == '\n') {
                    break;
                }
                if (value == '\r') {
                    throw new IOException("CR is forbidden; bridge responses must use LF");
                }
                if (bytes.size() == maxLineBytes) {
                    throw new IOException("bridge response exceeds "
                            + maxLineBytes + " bytes");
                }
                bytes.write(value);
            }
            if (bytes.size() == 0) {
                throw new IOException("empty bridge response line");
            }
            String decoded = decodeUtf8(bytes.toByteArray());
            if (decoded.charAt(0) == '\ufeff') {
                throw new IOException("UTF-8 BOM is forbidden");
            }
            return decoded;
        }

        private static String decodeUtf8(byte[] bytes) throws IOException {
            try {
                return StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT)
                        .decode(ByteBuffer.wrap(bytes))
                        .toString();
            } catch (CharacterCodingException e) {
                throw new IOException("bridge response is not valid UTF-8", e);
            }
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}
