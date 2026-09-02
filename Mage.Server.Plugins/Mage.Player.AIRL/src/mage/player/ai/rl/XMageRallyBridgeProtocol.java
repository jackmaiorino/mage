package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Typed values for the exact promoted(2) checkpoint and its pinned derivatives.
 *
 * <p>The wire service owns the MTG state and sampling. Java supplies only a
 * reset seed or the index that XMage actually selected. Every response carries
 * enough identity and decision commitments for the process client to reject a
 * wrong checkpoint, stale decision, reordered candidate row, or state splice.
 */
public final class XMageRallyBridgeProtocol {

    public static final String PROTOCOL = "mtg-kernel-checkpoint-shadow-stdio/v2";
    public static final int SCHEMA_VERSION = 2;
    public static final String MODEL_INPUT_COMMITMENT =
            "mtg-kernel-checkpoint-shadow-model-input-framed-sha256/v1";
    public static final String RANDOMIZATION_IDENTITY = "legacy_v1";
    public static final String RALLY_DECK_ID = "Rally";
    public static final String PYTHON_REFERENCE_SEED_VERSION =
            "kernel-python-rl-trainer-sha256-v2";

    public static final String ORIGINAL_AUTHORITY_KIND =
            "original-promoted2-generation384-store";
    public static final String SELECTED_GENERATION_AUTHORITY_KIND =
            "original-promoted2-validated-store-generation";
    public static final String CP7_BEHAVIOR_CLONE_AUTHORITY_KIND =
            "cp7-behavior-clone-derivative-v1";
    public static final String XMAGE_CP7_OUTCOME_AUTHORITY_KIND =
            "xmage-cp7-outcome-reinforce-derivative-v1";
    public static final String SOURCE_RUN_SHA256 =
            "2c9b7423004428c0e2bb138afafc15ec65957f6bd98c4587bea704fbf9549aae";
    public static final long SOURCE_GENERATION = 384L;
    public static final String SOURCE_CHECKPOINT_SHA256 =
            "4bd38cf3a9af3fb03fb04428fbc4286d4635007e848c7b9f0740122e430cbba8";
    public static final String SOURCE_SIDECAR_SHA256 =
            "7511c0377edd4e8d918fa5843f89a0270a8264e5466c329f6b4ef18bbf9e76bb";
    public static final String SOURCE_PAYLOAD_SHA256 =
            "a6c87366b2da9fc33923abab3c0e22d70c884cd9420477df3a475117be6beb99";
    public static final String SOURCE_TRAIN_STATE_SHA256 =
            "fc471f85d28293d72b42dc61de628859173bd67426e251a51bfbbe86c7d586d8";
    public static final String MODEL_PARAMETER_SHA256 =
            "db58dbe3f1f76b5bdf3bae4de657711dc818393b2bf1eeae88c02d8866b4d01d";
    public static final long CP7_BEHAVIOR_CLONE_ADAM_STEP = 141L;
    public static final String CP7_BEHAVIOR_CLONE_MANIFEST_SHA256 =
            "6ba733fead0d36c26cd24630245fa6f2a1216ae60c73f46d45e83b4cc714676c";
    public static final String CP7_BEHAVIOR_CLONE_PAYLOAD_SHA256 =
            "de1132f6b8b55975154133b91a2f2ea90bc1159676a041057fd827e728eca4e1";
    public static final String CP7_BEHAVIOR_CLONE_TRAIN_STATE_SHA256 =
            "64df1692fae7f78d0d4d4a4d6489325d253125276ca578c94912c9bd12374b56";
    public static final String CP7_BEHAVIOR_CLONE_MODEL_PARAMETER_SHA256 =
            "3f4da9d761771cf0d7cfe2da19b52dd93dd0bc59466d92318cc11fc850d8c4dc";
    public static final String ENVIRONMENT_TRAJECTORY_CONTRACT = "legacy-v1";
    public static final String SAMPLER_IDENTITY =
            "f32-q8-expq63-hamilton-splitmix64-v1";
    public static final String SAMPLER_CONTRACT_SHA256 =
            "276407494966b195b7c011caf984d2354484f7532161107b19ecc83388de92b6";

    public static final int MAX_REQUEST_ID_LENGTH = 128;
    public static final int MAX_ACTIONS = 65_536;

    private XMageRallyBridgeProtocol() {
    }

    public enum Seat {
        P0("p0"),
        P1("p1");

        private final String wire;

        Seat(String wire) {
            this.wire = wire;
        }

        public String wire() {
            return wire;
        }

        static Seat fromWire(String value) {
            for (Seat seat : values()) {
                if (seat.wire.equals(value)) {
                    return seat;
                }
            }
            throw new IllegalArgumentException("unsupported seat: " + value);
        }
    }

    public enum KernelPhaseStep {
        UNTAP("Untap"),
        UPKEEP("Upkeep"),
        DRAW("Draw"),
        MAIN1("Main1"),
        BEGIN_COMBAT("BeginCombat"),
        DECLARE_ATTACKERS("DeclareAttackers"),
        DECLARE_BLOCKERS("DeclareBlockers"),
        COMBAT_DAMAGE("CombatDamage"),
        END_COMBAT("EndCombat"),
        MAIN2("Main2"),
        END("End"),
        CLEANUP("Cleanup");

        private final String wire;

        KernelPhaseStep(String wire) {
            this.wire = wire;
        }

        public String wire() {
            return wire;
        }

        static KernelPhaseStep fromWire(String value) {
            for (KernelPhaseStep step : values()) {
                if (step.wire.equals(value)) {
                    return step;
                }
            }
            throw new IllegalArgumentException("unsupported kernel phase step: " + value);
        }
    }

    public static final class ExpectedClock {
        private final long turn;
        private final KernelPhaseStep phaseStep;
        private final Seat activePlayer;

        public ExpectedClock(long turn, KernelPhaseStep phaseStep, Seat activePlayer) {
            requireU32(turn, "expected_clock.turn");
            if (phaseStep == null || activePlayer == null) {
                throw new IllegalArgumentException(
                        "expected_clock phase_step and active_player must not be null");
            }
            this.turn = turn;
            this.phaseStep = phaseStep;
            this.activePlayer = activePlayer;
        }

        public long getTurn() { return turn; }
        public KernelPhaseStep getPhaseStep() { return phaseStep; }
        public Seat getActivePlayer() { return activePlayer; }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof ExpectedClock)) {
                return false;
            }
            ExpectedClock that = (ExpectedClock) other;
            return turn == that.turn
                    && phaseStep == that.phaseStep
                    && activePlayer == that.activePlayer;
        }

        @Override
        public int hashCode() {
            return Objects.hash(turn, phaseStep, activePlayer);
        }
    }

    public static final class KernelClock {
        private final long turn;
        private final KernelPhaseStep phaseStep;
        private final Seat activePlayer;
        private final Seat priorityPlayer;
        private final long stackDepth;

        KernelClock(long turn,
                    KernelPhaseStep phaseStep,
                    Seat activePlayer,
                    Seat priorityPlayer,
                    long stackDepth) {
            requireU32(turn, "kernel_clock.turn");
            requireU32(stackDepth, "kernel_clock.stack_depth");
            if (phaseStep == null || activePlayer == null || priorityPlayer == null) {
                throw new IllegalArgumentException(
                        "kernel_clock phase_step and players must not be null");
            }
            this.turn = turn;
            this.phaseStep = phaseStep;
            this.activePlayer = activePlayer;
            this.priorityPlayer = priorityPlayer;
            this.stackDepth = stackDepth;
        }

        public long getTurn() { return turn; }
        public KernelPhaseStep getPhaseStep() { return phaseStep; }
        public Seat getActivePlayer() { return activePlayer; }
        public Seat getPriorityPlayer() { return priorityPlayer; }
        public long getStackDepth() { return stackDepth; }

        public ExpectedClock toExpectedClock() {
            return new ExpectedClock(turn, phaseStep, activePlayer);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof KernelClock)) {
                return false;
            }
            KernelClock that = (KernelClock) other;
            return turn == that.turn
                    && phaseStep == that.phaseStep
                    && activePlayer == that.activePlayer
                    && priorityPlayer == that.priorityPlayer
                    && stackDepth == that.stackDepth;
        }

        @Override
        public int hashCode() {
            return Objects.hash(turn, phaseStep, activePlayer, priorityPlayer, stackDepth);
        }
    }

    public abstract static class Request {
        private final String requestType;
        private final String requestId;
        private final long episodeId;

        Request(String requestType, String requestId, long episodeId) {
            validateRequestId(requestId);
            requireU63(episodeId, "episode_id");
            this.requestType = requestType;
            this.requestId = requestId;
            this.episodeId = episodeId;
        }

        public String getRequestType() {
            return requestType;
        }

        public String getRequestId() {
            return requestId;
        }

        public long getEpisodeId() {
            return episodeId;
        }
    }

    public static final class ResetRequest extends Request {
        private final long baseSeed;

        public ResetRequest(String requestId, long episodeId, long baseSeed) {
            super("reset", requestId, episodeId);
            requireU63(baseSeed, "base_seed");
            this.baseSeed = baseSeed;
        }

        public long getBaseSeed() {
            return baseSeed;
        }
    }

    public static final class ScoreCurrentRequest extends Request {
        private final long expectedStep;

        public ScoreCurrentRequest(String requestId, long episodeId, long expectedStep) {
            super("score_current", requestId, episodeId);
            requireNonnegative(expectedStep, "expected_step");
            this.expectedStep = expectedStep;
        }

        public long getExpectedStep() {
            return expectedStep;
        }
    }

    public static final class StepRequest extends Request {
        private final long expectedStep;
        private final int selectedIndex;
        private final ExpectedClock expectedClock;

        public StepRequest(String requestId,
                           long episodeId,
                           long expectedStep,
                           int selectedIndex,
                           ExpectedClock expectedClock) {
            super("step", requestId, episodeId);
            requireNonnegative(expectedStep, "expected_step");
            if (selectedIndex < 0) {
                throw new IllegalArgumentException("selected_index must be nonnegative");
            }
            this.expectedStep = expectedStep;
            this.selectedIndex = selectedIndex;
            this.expectedClock = expectedClock;
        }

        public long getExpectedStep() {
            return expectedStep;
        }

        public int getSelectedIndex() {
            return selectedIndex;
        }

        public ExpectedClock getExpectedClock() {
            return expectedClock;
        }
    }

    public static final class CheckpointIdentity {
        private final String authorityKind;
        private final String sourceRunSha256;
        private final long sourceGeneration;
        private final String sourceCheckpointSha256;
        private final String sourceSidecarSha256;
        private final String sourcePayloadSha256;
        private final String sourceTrainStateSha256;
        private final String loadedRunSha256;
        private final long loadedGeneration;
        private final String loadedCheckpointSha256;
        private final String loadedPayloadSha256;
        private final String loadedTrainStateSha256;
        private final String modelParameterSha256;
        private final String environmentTrajectoryContract;
        private final String samplerIdentity;
        private final String samplerContractSha256;

        CheckpointIdentity(String authorityKind,
                           String sourceRunSha256,
                           long sourceGeneration,
                           String sourceCheckpointSha256,
                           String sourceSidecarSha256,
                           String sourcePayloadSha256,
                           String sourceTrainStateSha256,
                           String loadedRunSha256,
                           long loadedGeneration,
                           String loadedCheckpointSha256,
                           String loadedPayloadSha256,
                           String loadedTrainStateSha256,
                           String modelParameterSha256,
                           String environmentTrajectoryContract,
                           String samplerIdentity,
                           String samplerContractSha256) {
            this.authorityKind = authorityKind;
            this.sourceRunSha256 = sourceRunSha256;
            this.sourceGeneration = sourceGeneration;
            this.sourceCheckpointSha256 = sourceCheckpointSha256;
            this.sourceSidecarSha256 = sourceSidecarSha256;
            this.sourcePayloadSha256 = sourcePayloadSha256;
            this.sourceTrainStateSha256 = sourceTrainStateSha256;
            this.loadedRunSha256 = loadedRunSha256;
            this.loadedGeneration = loadedGeneration;
            this.loadedCheckpointSha256 = loadedCheckpointSha256;
            this.loadedPayloadSha256 = loadedPayloadSha256;
            this.loadedTrainStateSha256 = loadedTrainStateSha256;
            this.modelParameterSha256 = modelParameterSha256;
            this.environmentTrajectoryContract = environmentTrajectoryContract;
            this.samplerIdentity = samplerIdentity;
            this.samplerContractSha256 = samplerContractSha256;
        }

        public String getAuthorityKind() { return authorityKind; }
        public String getSourceRunSha256() { return sourceRunSha256; }
        public long getSourceGeneration() { return sourceGeneration; }
        public String getSourceCheckpointSha256() { return sourceCheckpointSha256; }
        public String getSourceSidecarSha256() { return sourceSidecarSha256; }
        public String getSourcePayloadSha256() { return sourcePayloadSha256; }
        public String getSourceTrainStateSha256() { return sourceTrainStateSha256; }
        public String getLoadedRunSha256() { return loadedRunSha256; }
        public long getLoadedGeneration() { return loadedGeneration; }
        public String getLoadedCheckpointSha256() { return loadedCheckpointSha256; }
        public String getLoadedPayloadSha256() { return loadedPayloadSha256; }
        public String getLoadedTrainStateSha256() { return loadedTrainStateSha256; }
        public String getModelParameterSha256() { return modelParameterSha256; }
        public String getEnvironmentTrajectoryContract() { return environmentTrajectoryContract; }
        public String getSamplerIdentity() { return samplerIdentity; }
        public String getSamplerContractSha256() { return samplerContractSha256; }

        void requireExactOriginalAuthority() {
            requireEqual("authority_kind", ORIGINAL_AUTHORITY_KIND, authorityKind);
            requireEqual("source_run_sha256", SOURCE_RUN_SHA256, sourceRunSha256);
            requireEqual("source_checkpoint_sha256", SOURCE_CHECKPOINT_SHA256,
                    sourceCheckpointSha256);
            requireEqual("source_sidecar_sha256", SOURCE_SIDECAR_SHA256, sourceSidecarSha256);
            requireEqual("source_payload_sha256", SOURCE_PAYLOAD_SHA256, sourcePayloadSha256);
            requireEqual("source_train_state_sha256", SOURCE_TRAIN_STATE_SHA256,
                    sourceTrainStateSha256);
            requireEqual("loaded_run_sha256", SOURCE_RUN_SHA256, loadedRunSha256);
            requireEqual("loaded_checkpoint_sha256", SOURCE_CHECKPOINT_SHA256,
                    loadedCheckpointSha256);
            requireEqual("loaded_payload_sha256", SOURCE_PAYLOAD_SHA256, loadedPayloadSha256);
            requireEqual("loaded_train_state_sha256", SOURCE_TRAIN_STATE_SHA256,
                    loadedTrainStateSha256);
            requireEqual("model_parameter_sha256", MODEL_PARAMETER_SHA256, modelParameterSha256);
            requireEqual("environment_trajectory_contract", ENVIRONMENT_TRAJECTORY_CONTRACT,
                    environmentTrajectoryContract);
            requireEqual("sampler_identity", SAMPLER_IDENTITY, samplerIdentity);
            requireEqual("sampler_contract_sha256", SAMPLER_CONTRACT_SHA256,
                    samplerContractSha256);
            if (sourceGeneration != SOURCE_GENERATION
                    || loadedGeneration != SOURCE_GENERATION) {
                throw new IllegalArgumentException("checkpoint generation identity mismatch");
            }
        }

        void requireSelectedOriginalGeneration(long expectedGeneration) {
            requireNonnegative(expectedGeneration, "expected checkpoint generation");
            requireEqual("authority_kind", SELECTED_GENERATION_AUTHORITY_KIND, authorityKind);
            requireEqual("source_run_sha256", SOURCE_RUN_SHA256, sourceRunSha256);
            requireEqual("loaded_run_sha256", SOURCE_RUN_SHA256, loadedRunSha256);
            requireEqual("loaded_checkpoint_sha256", sourceCheckpointSha256,
                    loadedCheckpointSha256);
            requireEqual("loaded_payload_sha256", sourcePayloadSha256, loadedPayloadSha256);
            requireEqual("loaded_train_state_sha256", sourceTrainStateSha256,
                    loadedTrainStateSha256);
            requireEqual("environment_trajectory_contract", ENVIRONMENT_TRAJECTORY_CONTRACT,
                    environmentTrajectoryContract);
            requireEqual("sampler_identity", SAMPLER_IDENTITY, samplerIdentity);
            requireEqual("sampler_contract_sha256", SAMPLER_CONTRACT_SHA256,
                    samplerContractSha256);
            if (sourceGeneration != expectedGeneration
                    || loadedGeneration != expectedGeneration) {
                throw new IllegalArgumentException("checkpoint generation identity mismatch");
            }
        }

        void requirePopulationStoreGenerationAuthority(
                String expectedAuthorityKind,
                String expectedSourceRunSha256,
                long expectedSourceGeneration,
                String expectedSourceCheckpointSha256,
                String expectedSourceSidecarSha256,
                String expectedSourcePayloadSha256,
                String expectedSourceTrainStateSha256,
                String expectedLoadedRunSha256,
                long expectedLoadedGeneration,
                String expectedLoadedCheckpointSha256,
                String expectedLoadedPayloadSha256,
                String expectedLoadedTrainStateSha256,
                String expectedModelParameterSha256,
                String expectedEnvironmentTrajectoryContract,
                String expectedSamplerIdentity,
                String expectedSamplerContractSha256) {
            requireNonnegative(expectedSourceGeneration, "expected population source generation");
            requireNonnegative(expectedLoadedGeneration, "expected population loaded generation");
            requireEqual("authority_kind", expectedAuthorityKind, authorityKind);
            requireEqual("source_run_sha256", expectedSourceRunSha256, sourceRunSha256);
            requireEqual("source_checkpoint_sha256", expectedSourceCheckpointSha256,
                    sourceCheckpointSha256);
            requireEqual("source_sidecar_sha256", expectedSourceSidecarSha256,
                    sourceSidecarSha256);
            requireEqual("source_payload_sha256", expectedSourcePayloadSha256,
                    sourcePayloadSha256);
            requireEqual("source_train_state_sha256", expectedSourceTrainStateSha256,
                    sourceTrainStateSha256);
            requireEqual("loaded_run_sha256", expectedLoadedRunSha256, loadedRunSha256);
            requireEqual("loaded_checkpoint_sha256", expectedLoadedCheckpointSha256,
                    loadedCheckpointSha256);
            requireEqual("loaded_payload_sha256", expectedLoadedPayloadSha256,
                    loadedPayloadSha256);
            requireEqual("loaded_train_state_sha256", expectedLoadedTrainStateSha256,
                    loadedTrainStateSha256);
            requireEqual("model_parameter_sha256", expectedModelParameterSha256,
                    modelParameterSha256);
            requireEqual("environment_trajectory_contract", expectedEnvironmentTrajectoryContract,
                    environmentTrajectoryContract);
            requireEqual("sampler_identity", expectedSamplerIdentity, samplerIdentity);
            requireEqual("sampler_contract_sha256", expectedSamplerContractSha256,
                    samplerContractSha256);
            if (sourceGeneration != expectedSourceGeneration
                    || loadedGeneration != expectedLoadedGeneration) {
                throw new IllegalArgumentException("population Store generation identity mismatch");
            }
        }

        void requireExactCp7BehaviorCloneAuthority() {
            requireCp7BehaviorCloneAuthority(
                    CP7_BEHAVIOR_CLONE_ADAM_STEP,
                    CP7_BEHAVIOR_CLONE_MANIFEST_SHA256,
                    CP7_BEHAVIOR_CLONE_PAYLOAD_SHA256,
                    CP7_BEHAVIOR_CLONE_TRAIN_STATE_SHA256,
                    CP7_BEHAVIOR_CLONE_MODEL_PARAMETER_SHA256);
        }

        void requireCp7BehaviorCloneAuthority(
                long expectedAdamStep,
                String expectedManifestSha256,
                String expectedPayloadSha256,
                String expectedTrainStateSha256,
                String expectedModelParameterSha256) {
            requireNonnegative(expectedAdamStep, "expected CP7 behavior-clone Adam step");
            requireEqual("authority_kind", CP7_BEHAVIOR_CLONE_AUTHORITY_KIND, authorityKind);
            requireEqual("source_run_sha256", SOURCE_RUN_SHA256, sourceRunSha256);
            requireEqual("source_checkpoint_sha256", SOURCE_CHECKPOINT_SHA256,
                    sourceCheckpointSha256);
            requireEqual("source_sidecar_sha256", SOURCE_SIDECAR_SHA256, sourceSidecarSha256);
            requireEqual("source_payload_sha256", SOURCE_PAYLOAD_SHA256, sourcePayloadSha256);
            requireEqual("source_train_state_sha256", SOURCE_TRAIN_STATE_SHA256,
                    sourceTrainStateSha256);
            requireEqual("loaded_run_sha256", SOURCE_RUN_SHA256, loadedRunSha256);
            requireEqual("loaded_checkpoint_sha256", expectedManifestSha256,
                    loadedCheckpointSha256);
            requireEqual("loaded_payload_sha256", expectedPayloadSha256,
                    loadedPayloadSha256);
            requireEqual("loaded_train_state_sha256", expectedTrainStateSha256,
                    loadedTrainStateSha256);
            requireEqual("model_parameter_sha256", expectedModelParameterSha256,
                    modelParameterSha256);
            requireEqual("environment_trajectory_contract", ENVIRONMENT_TRAJECTORY_CONTRACT,
                    environmentTrajectoryContract);
            requireEqual("sampler_identity", SAMPLER_IDENTITY, samplerIdentity);
            requireEqual("sampler_contract_sha256", SAMPLER_CONTRACT_SHA256,
                    samplerContractSha256);
            if (sourceGeneration != SOURCE_GENERATION
                    || loadedGeneration != expectedAdamStep) {
                throw new IllegalArgumentException("checkpoint generation identity mismatch");
            }
        }

        void requireXMageCp7OutcomeAuthority(
                String expectedAuthorityKind,
                long expectedAdamStep,
                String expectedManifestSha256,
                String expectedPayloadSha256,
                String expectedTrainStateSha256,
                String expectedModelParameterSha256,
                String expectedEnvironmentTrajectoryContract) {
            requireNonnegative(expectedAdamStep, "expected XMage CP7 outcome Adam step");
            requireEqual("authority_kind", expectedAuthorityKind, authorityKind);
            requireEqual("source_run_sha256", SOURCE_RUN_SHA256, sourceRunSha256);
            requireEqual("source_checkpoint_sha256", SOURCE_CHECKPOINT_SHA256,
                    sourceCheckpointSha256);
            requireEqual("source_sidecar_sha256", SOURCE_SIDECAR_SHA256, sourceSidecarSha256);
            requireEqual("source_payload_sha256", SOURCE_PAYLOAD_SHA256, sourcePayloadSha256);
            requireEqual("source_train_state_sha256", SOURCE_TRAIN_STATE_SHA256,
                    sourceTrainStateSha256);
            requireEqual("loaded_run_sha256", SOURCE_RUN_SHA256, loadedRunSha256);
            requireEqual("loaded_checkpoint_sha256", expectedManifestSha256,
                    loadedCheckpointSha256);
            requireEqual("loaded_payload_sha256", expectedPayloadSha256, loadedPayloadSha256);
            requireEqual("loaded_train_state_sha256", expectedTrainStateSha256,
                    loadedTrainStateSha256);
            requireEqual("model_parameter_sha256", expectedModelParameterSha256,
                    modelParameterSha256);
            requireEqual("environment_trajectory_contract", expectedEnvironmentTrajectoryContract,
                    environmentTrajectoryContract);
            requireEqual("sampler_identity", SAMPLER_IDENTITY, samplerIdentity);
            requireEqual("sampler_contract_sha256", SAMPLER_CONTRACT_SHA256,
                    samplerContractSha256);
            if (sourceGeneration != SOURCE_GENERATION
                    || loadedGeneration != expectedAdamStep) {
                throw new IllegalArgumentException("checkpoint generation identity mismatch");
            }
        }
    }

    /** Exact, duplicate-free JSON action object plus its typed discriminator. */
    public static final class ActionSemantic {
        private final String actionKind;
        private final String canonicalJson;

        ActionSemantic(String actionKind, String canonicalJson) {
            this.actionKind = Objects.requireNonNull(actionKind, "actionKind");
            this.canonicalJson = Objects.requireNonNull(canonicalJson, "canonicalJson");
        }

        public String getActionKind() {
            return actionKind;
        }

        public String getCanonicalJson() {
            return canonicalJson;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof ActionSemantic)) {
                return false;
            }
            ActionSemantic that = (ActionSemantic) other;
            return actionKind.equals(that.actionKind) && canonicalJson.equals(that.canonicalJson);
        }

        @Override
        public int hashCode() {
            return 31 * actionKind.hashCode() + canonicalJson.hashCode();
        }
    }

    public static final class AppliedAction {
        private final long episodeId;
        private final long step;
        private final String candidateOrderCommitment128Hex;
        private final String modelInputSha256;
        private final int selectedIndex;
        private final long selectedLogitF32Bits;
        private final ActionSemantic semantic;

        AppliedAction(long episodeId,
                      long step,
                      String candidateOrderCommitment128Hex,
                      String modelInputSha256,
                      int selectedIndex,
                      long selectedLogitF32Bits,
                      ActionSemantic semantic) {
            this.episodeId = episodeId;
            this.step = step;
            this.candidateOrderCommitment128Hex = candidateOrderCommitment128Hex;
            this.modelInputSha256 = modelInputSha256;
            this.selectedIndex = selectedIndex;
            this.selectedLogitF32Bits = selectedLogitF32Bits;
            this.semantic = semantic;
        }

        public long getEpisodeId() { return episodeId; }
        public long getStep() { return step; }
        public String getCandidateOrderCommitment128Hex() {
            return candidateOrderCommitment128Hex;
        }
        public String getModelInputSha256() { return modelInputSha256; }
        public int getSelectedIndex() { return selectedIndex; }
        public long getSelectedLogitF32Bits() { return selectedLogitF32Bits; }
        public ActionSemantic getSemantic() { return semantic; }
    }

    public static final class DecisionBody {
        private final List<String> deckIds;
        private final String randomizationIdentity;
        private final String baseSeedU64Hex;
        private final long pairIndex;
        private final String pairEnvironmentSeedU64Hex;
        private final List<List<Integer>> initialLibraryCardDefinitionIds;
        private final long episodeId;
        private final long step;
        private final long environmentRevision;
        private final long physicalDecisionId;
        private final int substepIndex;
        private final int substepCount;
        private final Seat actingPlayer;
        private final String decisionKind;
        private final int legalActionCount;
        private final Seat candidateSeat;
        private final boolean candidateControlsCurrentActor;
        private final long actorPhysicalDecisionOrdinal;
        private final String candidateActionSeedU64Hex;
        private final Integer selectedActionIndex;
        private final String candidateOrderCommitment128Hex;
        private final String modelInputCommitment;
        private final String modelInputSha256;
        private final String diagnosticStateHashU64Hex;
        private final String coreEnvironmentHashU64Hex;
        private final List<Long> logitsF32Bits;
        private final long valueF32Bits;
        private final List<ActionSemantic> actionSemantics;
        private final KernelClock kernelClock;

        DecisionBody(List<String> deckIds,
                     String randomizationIdentity,
                     String baseSeedU64Hex,
                     long pairIndex,
                     String pairEnvironmentSeedU64Hex,
                     List<List<Integer>> initialLibraryCardDefinitionIds,
                     long episodeId,
                     long step,
                     long environmentRevision,
                     long physicalDecisionId,
                     int substepIndex,
                     int substepCount,
                     Seat actingPlayer,
                     String decisionKind,
                     int legalActionCount,
                     Seat candidateSeat,
                     boolean candidateControlsCurrentActor,
                     long actorPhysicalDecisionOrdinal,
                     String candidateActionSeedU64Hex,
                     Integer selectedActionIndex,
                     String candidateOrderCommitment128Hex,
                     String modelInputCommitment,
                     String modelInputSha256,
                     String diagnosticStateHashU64Hex,
                     String coreEnvironmentHashU64Hex,
                     List<Long> logitsF32Bits,
                     long valueF32Bits,
                     List<ActionSemantic> actionSemantics,
                     KernelClock kernelClock) {
            this.deckIds = immutableCopy(deckIds);
            this.randomizationIdentity = randomizationIdentity;
            this.baseSeedU64Hex = baseSeedU64Hex;
            this.pairIndex = pairIndex;
            this.pairEnvironmentSeedU64Hex = pairEnvironmentSeedU64Hex;
            this.initialLibraryCardDefinitionIds = immutableNestedCopy(
                    initialLibraryCardDefinitionIds);
            this.episodeId = episodeId;
            this.step = step;
            this.environmentRevision = environmentRevision;
            this.physicalDecisionId = physicalDecisionId;
            this.substepIndex = substepIndex;
            this.substepCount = substepCount;
            this.actingPlayer = actingPlayer;
            this.decisionKind = decisionKind;
            this.legalActionCount = legalActionCount;
            this.candidateSeat = candidateSeat;
            this.candidateControlsCurrentActor = candidateControlsCurrentActor;
            this.actorPhysicalDecisionOrdinal = actorPhysicalDecisionOrdinal;
            this.candidateActionSeedU64Hex = candidateActionSeedU64Hex;
            this.selectedActionIndex = selectedActionIndex;
            this.candidateOrderCommitment128Hex = candidateOrderCommitment128Hex;
            this.modelInputCommitment = modelInputCommitment;
            this.modelInputSha256 = modelInputSha256;
            this.diagnosticStateHashU64Hex = diagnosticStateHashU64Hex;
            this.coreEnvironmentHashU64Hex = coreEnvironmentHashU64Hex;
            this.logitsF32Bits = immutableCopy(logitsF32Bits);
            this.valueF32Bits = valueF32Bits;
            this.actionSemantics = immutableCopy(actionSemantics);
            this.kernelClock = kernelClock;
        }

        public List<String> getDeckIds() { return deckIds; }
        public String getRandomizationIdentity() { return randomizationIdentity; }
        public String getBaseSeedU64Hex() { return baseSeedU64Hex; }
        public long getPairIndex() { return pairIndex; }
        public String getPairEnvironmentSeedU64Hex() { return pairEnvironmentSeedU64Hex; }
        public List<List<Integer>> getInitialLibraryCardDefinitionIds() {
            return initialLibraryCardDefinitionIds;
        }
        public long getEpisodeId() { return episodeId; }
        public long getStep() { return step; }
        public long getEnvironmentRevision() { return environmentRevision; }
        public long getPhysicalDecisionId() { return physicalDecisionId; }
        public int getSubstepIndex() { return substepIndex; }
        public int getSubstepCount() { return substepCount; }
        public Seat getActingPlayer() { return actingPlayer; }
        public String getDecisionKind() { return decisionKind; }
        public int getLegalActionCount() { return legalActionCount; }
        public Seat getCandidateSeat() { return candidateSeat; }
        public boolean isCandidateControlsCurrentActor() {
            return candidateControlsCurrentActor;
        }
        public long getActorPhysicalDecisionOrdinal() { return actorPhysicalDecisionOrdinal; }
        public String getCandidateActionSeedU64Hex() { return candidateActionSeedU64Hex; }
        public Integer getSelectedActionIndex() { return selectedActionIndex; }
        public String getCandidateOrderCommitment128Hex() {
            return candidateOrderCommitment128Hex;
        }
        public String getModelInputCommitment() { return modelInputCommitment; }
        public String getModelInputSha256() { return modelInputSha256; }
        public String getDiagnosticStateHashU64Hex() { return diagnosticStateHashU64Hex; }
        public String getCoreEnvironmentHashU64Hex() { return coreEnvironmentHashU64Hex; }
        public List<Long> getLogitsF32Bits() { return logitsF32Bits; }
        public long getValueF32Bits() { return valueF32Bits; }
        public List<ActionSemantic> getActionSemantics() { return actionSemantics; }
        public KernelClock getKernelClock() { return kernelClock; }

        boolean sameCurrentDecision(DecisionBody that) {
            return that != null
                    && deckIds.equals(that.deckIds)
                    && Objects.equals(randomizationIdentity, that.randomizationIdentity)
                    && Objects.equals(baseSeedU64Hex, that.baseSeedU64Hex)
                    && pairIndex == that.pairIndex
                    && Objects.equals(pairEnvironmentSeedU64Hex,
                    that.pairEnvironmentSeedU64Hex)
                    && episodeId == that.episodeId
                    && step == that.step
                    && environmentRevision == that.environmentRevision
                    && physicalDecisionId == that.physicalDecisionId
                    && substepIndex == that.substepIndex
                    && substepCount == that.substepCount
                    && actingPlayer == that.actingPlayer
                    && Objects.equals(decisionKind, that.decisionKind)
                    && legalActionCount == that.legalActionCount
                    && candidateSeat == that.candidateSeat
                    && candidateControlsCurrentActor == that.candidateControlsCurrentActor
                    && actorPhysicalDecisionOrdinal == that.actorPhysicalDecisionOrdinal
                    && Objects.equals(candidateActionSeedU64Hex,
                    that.candidateActionSeedU64Hex)
                    && Objects.equals(selectedActionIndex, that.selectedActionIndex)
                    && Objects.equals(candidateOrderCommitment128Hex,
                    that.candidateOrderCommitment128Hex)
                    && Objects.equals(modelInputCommitment, that.modelInputCommitment)
                    && Objects.equals(modelInputSha256, that.modelInputSha256)
                    && Objects.equals(diagnosticStateHashU64Hex,
                    that.diagnosticStateHashU64Hex)
                    && Objects.equals(coreEnvironmentHashU64Hex,
                    that.coreEnvironmentHashU64Hex)
                    && logitsF32Bits.equals(that.logitsF32Bits)
                    && valueF32Bits == that.valueF32Bits
                    && actionSemantics.equals(that.actionSemantics)
                    && Objects.equals(kernelClock, that.kernelClock);
        }
    }

    public static final class TerminalRecord {
        private final int schemaVersion;
        private final List<String> deckIds;
        private final List<String> deckHashesUnsignedDecimal;
        private final long episodeId;
        private final String terminalOutcome;
        private final String terminalClassification;
        private final String terminalCode;
        private final Seat winner;
        private final List<Integer> terminalReward;
        private final String terminalReason;
        private final long policyStepCount;
        private final long physicalDecisionCount;

        TerminalRecord(int schemaVersion,
                       List<String> deckIds,
                       List<String> deckHashesUnsignedDecimal,
                       long episodeId,
                       String terminalOutcome,
                       String terminalClassification,
                       String terminalCode,
                       Seat winner,
                       List<Integer> terminalReward,
                       String terminalReason,
                       long policyStepCount,
                       long physicalDecisionCount) {
            this.schemaVersion = schemaVersion;
            this.deckIds = immutableCopy(deckIds);
            this.deckHashesUnsignedDecimal = immutableCopy(deckHashesUnsignedDecimal);
            this.episodeId = episodeId;
            this.terminalOutcome = terminalOutcome;
            this.terminalClassification = terminalClassification;
            this.terminalCode = terminalCode;
            this.winner = winner;
            this.terminalReward = immutableCopy(terminalReward);
            this.terminalReason = terminalReason;
            this.policyStepCount = policyStepCount;
            this.physicalDecisionCount = physicalDecisionCount;
        }

        public int getSchemaVersion() { return schemaVersion; }
        public List<String> getDeckIds() { return deckIds; }
        public List<String> getDeckHashesUnsignedDecimal() {
            return deckHashesUnsignedDecimal;
        }
        public long getEpisodeId() { return episodeId; }
        public String getTerminalOutcome() { return terminalOutcome; }
        public String getTerminalClassification() { return terminalClassification; }
        public String getTerminalCode() { return terminalCode; }
        public Seat getWinner() { return winner; }
        public List<Integer> getTerminalReward() { return terminalReward; }
        public String getTerminalReason() { return terminalReason; }
        public long getPolicyStepCount() { return policyStepCount; }
        public long getPhysicalDecisionCount() { return physicalDecisionCount; }
    }

    public static final class TerminalBody {
        private final List<String> deckIds;
        private final String randomizationIdentity;
        private final String baseSeedU64Hex;
        private final long pairIndex;
        private final String pairEnvironmentSeedU64Hex;
        private final List<List<Integer>> initialLibraryCardDefinitionIds;
        private final TerminalRecord terminal;
        private final Seat candidateSeat;
        private final String diagnosticStateHashU64Hex;
        private final String coreEnvironmentHashU64Hex;

        TerminalBody(List<String> deckIds,
                     String randomizationIdentity,
                     String baseSeedU64Hex,
                     long pairIndex,
                     String pairEnvironmentSeedU64Hex,
                     List<List<Integer>> initialLibraryCardDefinitionIds,
                     TerminalRecord terminal,
                     Seat candidateSeat,
                     String diagnosticStateHashU64Hex,
                     String coreEnvironmentHashU64Hex) {
            this.deckIds = immutableCopy(deckIds);
            this.randomizationIdentity = randomizationIdentity;
            this.baseSeedU64Hex = baseSeedU64Hex;
            this.pairIndex = pairIndex;
            this.pairEnvironmentSeedU64Hex = pairEnvironmentSeedU64Hex;
            this.initialLibraryCardDefinitionIds = immutableNestedCopy(
                    initialLibraryCardDefinitionIds);
            this.terminal = terminal;
            this.candidateSeat = candidateSeat;
            this.diagnosticStateHashU64Hex = diagnosticStateHashU64Hex;
            this.coreEnvironmentHashU64Hex = coreEnvironmentHashU64Hex;
        }

        public List<String> getDeckIds() { return deckIds; }
        public String getRandomizationIdentity() { return randomizationIdentity; }
        public String getBaseSeedU64Hex() { return baseSeedU64Hex; }
        public long getPairIndex() { return pairIndex; }
        public String getPairEnvironmentSeedU64Hex() { return pairEnvironmentSeedU64Hex; }
        public List<List<Integer>> getInitialLibraryCardDefinitionIds() {
            return initialLibraryCardDefinitionIds;
        }
        public TerminalRecord getTerminal() { return terminal; }
        public Seat getCandidateSeat() { return candidateSeat; }
        public String getDiagnosticStateHashU64Hex() { return diagnosticStateHashU64Hex; }
        public String getCoreEnvironmentHashU64Hex() { return coreEnvironmentHashU64Hex; }
    }

    public abstract static class ResponseBody {
        private final String responseType;

        ResponseBody(String responseType) {
            this.responseType = responseType;
        }

        public String getResponseType() {
            return responseType;
        }
    }

    public static final class DecisionResponseBody extends ResponseBody {
        private final DecisionBody decision;
        private final AppliedAction appliedAction;

        DecisionResponseBody(DecisionBody decision, AppliedAction appliedAction) {
            super("decision");
            this.decision = decision;
            this.appliedAction = appliedAction;
        }

        public DecisionBody getDecision() { return decision; }
        public AppliedAction getAppliedAction() { return appliedAction; }
    }

    public static final class TerminalResponseBody extends ResponseBody {
        private final TerminalBody terminal;
        private final AppliedAction appliedAction;

        TerminalResponseBody(TerminalBody terminal, AppliedAction appliedAction) {
            super("terminal");
            this.terminal = terminal;
            this.appliedAction = appliedAction;
        }

        public TerminalBody getTerminal() { return terminal; }
        public AppliedAction getAppliedAction() { return appliedAction; }
    }

    public static final class ErrorResponseBody extends ResponseBody {
        private final String errorCode;
        private final String message;

        ErrorResponseBody(String errorCode, String message) {
            super("error");
            this.errorCode = errorCode;
            this.message = message;
        }

        public String getErrorCode() { return errorCode; }
        public String getMessage() { return message; }
    }

    public static final class Response {
        private final String protocol;
        private final int schemaVersion;
        private final String requestId;
        private final CheckpointIdentity checkpoint;
        private final ResponseBody body;

        Response(String protocol,
                 int schemaVersion,
                 String requestId,
                 CheckpointIdentity checkpoint,
                 ResponseBody body) {
            this.protocol = protocol;
            this.schemaVersion = schemaVersion;
            this.requestId = requestId;
            this.checkpoint = checkpoint;
            this.body = body;
        }

        public String getProtocol() { return protocol; }
        public int getSchemaVersion() { return schemaVersion; }
        public String getRequestId() { return requestId; }
        public CheckpointIdentity getCheckpoint() { return checkpoint; }
        public ResponseBody getBody() { return body; }
    }

    static void validateRequestId(String requestId) {
        if (requestId == null || requestId.isEmpty()
                || requestId.length() > MAX_REQUEST_ID_LENGTH) {
            throw new IllegalArgumentException("invalid request_id length");
        }
        for (int i = 0; i < requestId.length(); i++) {
            char c = requestId.charAt(i);
            if (c < 0x21 || c > 0x7e || c == '"' || c == '\\') {
                throw new IllegalArgumentException(
                        "request_id must use graphic ASCII excluding quote and backslash");
            }
        }
    }

    static String u64Hex(long value) {
        requireNonnegative(value, "u64 value");
        return String.format(java.util.Locale.ROOT, "%016x", value);
    }

    /** Independent Java reproduction of native_trainer_episode_schedule_v1. */
    public static String derivePairEnvironmentSeedU64Hex(long baseSeed, long episodeId) {
        requireU63(baseSeed, "base_seed");
        requireU63(episodeId, "episode_id");
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            appendSeedAtom(digest, "version",
                    PYTHON_REFERENCE_SEED_VERSION.getBytes(StandardCharsets.UTF_8));
            appendSeedAtom(digest, "namespace", "train-env".getBytes(StandardCharsets.UTF_8));
            appendSeedAtom(digest, "field-name", "base_seed".getBytes(StandardCharsets.UTF_8));
            appendSeedAtom(digest, "u63", ByteBuffer.allocate(8).putLong(baseSeed).array());
            appendSeedAtom(digest, "field-name", "pair_index".getBytes(StandardCharsets.UTF_8));
            appendSeedAtom(digest, "u63",
                    ByteBuffer.allocate(8).putLong(episodeId / 2L).array());
            byte[] hash = digest.digest();
            long seed = ByteBuffer.wrap(hash, 0, 8).getLong() & Long.MAX_VALUE;
            return u64Hex(seed);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("Java runtime lacks SHA-256", e);
        }
    }

    private static void appendSeedAtom(MessageDigest digest, String tag, byte[] payload) {
        byte[] tagBytes = tag.getBytes(StandardCharsets.UTF_8);
        digest.update(ByteBuffer.allocate(4).putInt(tagBytes.length).array());
        digest.update(tagBytes);
        digest.update(ByteBuffer.allocate(8).putLong(payload.length).array());
        digest.update(payload);
    }

    private static void requireNonnegative(long value, String field) {
        if (value < 0L) {
            throw new IllegalArgumentException(field + " must be nonnegative");
        }
    }

    private static void requireU63(long value, String field) {
        requireNonnegative(value, field);
    }

    private static void requireU32(long value, String field) {
        if (value < 0L || value > 0xffff_ffffL) {
            throw new IllegalArgumentException(field + " must be a u32");
        }
    }

    private static void requireEqual(String field, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(field + " identity mismatch");
        }
    }

    private static <T> List<T> immutableCopy(List<T> values) {
        if (values == null) {
            return null;
        }
        return Collections.unmodifiableList(new ArrayList<>(values));
    }

    private static List<List<Integer>> immutableNestedCopy(List<List<Integer>> values) {
        if (values == null) {
            return null;
        }
        ArrayList<List<Integer>> rows = new ArrayList<>(values.size());
        for (List<Integer> row : values) {
            rows.add(Collections.unmodifiableList(new ArrayList<>(row)));
        }
        return Collections.unmodifiableList(rows);
    }
}
