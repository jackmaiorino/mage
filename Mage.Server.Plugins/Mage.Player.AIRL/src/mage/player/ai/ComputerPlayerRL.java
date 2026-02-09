package mage.player.ai;

import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.mana.ManaAbility;
import mage.abilities.PlayLandAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.PassAbility;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.filter.common.FilterLandCard;
import mage.constants.RangeOfInfluence;
import mage.constants.TurnPhase;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.rl.GameLogger;
import mage.player.ai.rl.MulliganLogger;
import mage.player.ai.rl.MulliganModel;
import mage.player.ai.rl.PythonModel;
import mage.player.ai.rl.RLTrainer;
import mage.player.ai.rl.StateSequenceBuilder;
import mage.player.ai.util.CombatUtil;
import mage.players.Player;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.target.TargetCard;

public class ComputerPlayerRL extends ComputerPlayer7 {

    // Denser reward shaping to help credit assignment with sparse terminal rewards
    // These provide gradient signal for actions that are generally good in MTG
    private static final double LAND_PLAY_REWARD = 0.05;   // Playing lands is almost always good
    private static final double SPELL_CAST_REWARD = 0.02;  // Casting spells advances the game (was 0)
    private static final double ATTACK_REWARD = 0.01;      // Attacking pressures opponent
    private static final double BLOCK_REWARD = 0.005;      // Blocking prevents damage
    private static final boolean ACTIVATION_DIAG = "1".equals(System.getenv().getOrDefault("RL_ACTIVATION_DIAG", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ACTIVATION_DIAG", "0"));
    private static final boolean USE_ENGINE_CHOICES = "1".equals(System.getenv().getOrDefault("RL_ENGINE_CHOICES", "1"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ENGINE_CHOICES", "1"));
    private static final boolean MULLIGAN_TRACE = "1".equals(System.getenv().getOrDefault("RL_MULLIGAN_TRACE", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_MULLIGAN_TRACE", "0"));
    private static final boolean MULLIGAN_TRACE_JSONL = "1".equals(System.getenv().getOrDefault("RL_MULLIGAN_TRACE_JSONL", "1"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_MULLIGAN_TRACE_JSONL", "1"));

    // Track RL player activation failures (these pollute training signal)
    private static final java.util.concurrent.atomic.AtomicInteger RL_ACTIVATION_FAILURES = new java.util.concurrent.atomic.AtomicInteger(0);
    private static final java.util.concurrent.atomic.AtomicInteger SIMULATION_TRAINING_SKIPPED = new java.util.concurrent.atomic.AtomicInteger(0);

    // Track valid alternative cost choices discovered during simulation
    // Key: source card UUID, Value: set of valid choice indices (e.g., "1" for option 1, "2" for option 2)
    private final Map<UUID, Set<String>> validAlternativeCosts = new HashMap<>();

    // ThreadLocal to force a specific alternative cost choice during simulation testing
    private static final ThreadLocal<String> forcedAlternativeChoice = new ThreadLocal<>();
    // ThreadLocal to track which choice was made and what options were available
    private static final ThreadLocal<ChoiceTrackingData> choiceTrackingData = new ThreadLocal<>();

    private PythonModel model;
    private MulliganModel mulliganModel;
    protected StateSequenceBuilder.SequenceOutput currentState;
    private final List<StateSequenceBuilder.TrainingData> trainingBuffer;
    private final java.util.Map<StateSequenceBuilder.ActionType, Integer> decisionCountsByHead;
    private Ability currentAbility;
    private int mulligansTaken = 0;
    private int currentEpisode = -1; // Main model episode for logging
    private int mulliganEpisode = -1; // Mulligan model episode for epsilon-greedy
    private int lastLoggedTurn = -1; // Track turn changes for game logging

    // Track early-game land drops for mulligan reward shaping
    private boolean rlPlayerHasHadATurn = false;
    private int lastTrackedGameTurn = 0;
    private int rlPlayerTurnsTracked = 0;
    private int earlyLandHits = 0; // How many of first 3 turns were "on curve"

    // Track mulligan decisions for training
    private final List<float[]> mulliganFeatures = new ArrayList<>(); // Full feature vectors
    private final List<Float> mulliganDecisions = new ArrayList<>(); // 1.0=keep, 0.0=mulligan
    private final List<Boolean> mulliganOverrides = new ArrayList<>(); // true=decision was overridden
    private final List<Integer> mulliganLandCounts = new ArrayList<>(); // For logging only

    // Duplicate-call protection for chooseMulligan (some engine flows call it multiple times).
    // Dedupe based on current hand fingerprint (IDs) + size.
    private int lastMulliganHandFingerprint = Integer.MIN_VALUE;
    private int lastMulliganHandSize = -1;
    private Boolean lastMulliganDecisionShouldMulligan = null;

    private static String trunc(String s, int maxLen) {
        if (s == null) {
            return "";
        }
        if (s.length() <= maxLen) {
            return s;
        }
        return s.substring(0, maxLen) + "...";
    }

    private static final Object MULL_TRAIN_LOG_LOCK = new Object();
    private static final String MULL_TRAIN_LOG_FILE = System.getenv().getOrDefault(
            "MULLIGAN_TRAINING_LOG_FILE",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mulligan_training.log"
    );
    private static final DateTimeFormatter MULL_TRAIN_TS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss,SSS");

    private static final Object MULL_TRACE_JSONL_LOCK = new Object();
    private static final String MULL_TRACE_JSONL_FILE = System.getenv().getOrDefault(
            "MULLIGAN_TRACE_JSONL_FILE",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mulligan_trace.jsonl"
    );

    private long mulliganEventSeq = 0L;

    private static void mulliganTrainingLog(String line) {
        if (line == null || line.isEmpty()) {
            return;
        }
        try {
            Path p = Paths.get(MULL_TRAIN_LOG_FILE);
            Path parent = p.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String stamped = LocalDateTime.now().format(MULL_TRAIN_TS) + " - mulligan_training - INFO - " + line + System.lineSeparator();
            synchronized (MULL_TRAIN_LOG_LOCK) {
                Files.write(p, stamped.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            }
        } catch (Exception ignored) {
            // Intentionally ignore logging failures (never crash engine for diagnostics).
        }
    }

    private static String jsonEscape(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder out = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\':
                    out.append("\\\\");
                    break;
                case '"':
                    out.append("\\\"");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    out.append(c);
                    break;
            }
        }
        return out.toString();
    }

    private void mulliganTraceJsonl(String eventType, String jsonPayloadFields) {
        if (!MULLIGAN_TRACE_JSONL) {
            return;
        }
        try {
            Path p = Paths.get(MULL_TRACE_JSONL_FILE);
            Path parent = p.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            long seq = ++mulliganEventSeq;
            StringBuilder sb = new StringBuilder(512);
            sb.append("{");
            sb.append("\"event\":\"").append(jsonEscape(eventType)).append("\",");
            sb.append("\"eventSeq\":").append(seq).append(",");
            sb.append("\"episode\":").append(currentEpisode).append(",");
            sb.append("\"player\":\"").append(jsonEscape(getName())).append("\",");
            sb.append("\"mulligansTaken\":").append(mulligansTaken);
            if (jsonPayloadFields != null && !jsonPayloadFields.trim().isEmpty()) {
                sb.append(",").append(jsonPayloadFields);
            }
            sb.append("}").append(System.lineSeparator());

            byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            synchronized (MULL_TRACE_JSONL_LOCK) {
                Files.write(p, bytes, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            }
        } catch (Exception ignored) {
            // never crash engine for diagnostics
        }
    }

    private int computeHandFingerprint(Game game) {
        try {
            List<Card> cards = new ArrayList<>(getHand().getCards(game));
            if (cards.isEmpty()) {
                return 0;
            }
            List<UUID> ids = new ArrayList<>(cards.size());
            for (Card c : cards) {
                if (c != null && c.getId() != null) {
                    ids.add(c.getId());
                }
            }
            Collections.sort(ids, (a, b) -> {
                int c = Long.compare(a.getMostSignificantBits(), b.getMostSignificantBits());
                if (c != 0) {
                    return c;
                }
                return Long.compare(a.getLeastSignificantBits(), b.getLeastSignificantBits());
            });
            int h = 1;
            for (UUID id : ids) {
                h = 31 * h + id.hashCode();
            }
            return h;
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * If true, the agent will act greedily (arg-max) instead of sampling.
     * Useful for evaluation where we want a deterministic policy.
     */
    private final boolean greedyMode;

    // Which policy to use on the Python side ("train" or "snap:<id>")
    private final String policyKey;

    // Whether to record training data from decisions made by this player
    private final boolean trainingEnabled;

    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonModel model) {
        this(name, range, model, false, true, "train");
    }

    // Convenience ctor with greedy flag
    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonModel model, boolean greedy) {
        this(name, range, model, greedy, true, "train");
    }

    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonModel model, boolean greedy, boolean trainingEnabled, String policyKey) {
        super(name, range, 10);
        this.model = model;
        this.mulliganModel = new MulliganModel(model);
        this.trainingBuffer = new ArrayList<>();
        this.decisionCountsByHead = new java.util.HashMap<>();
        this.greedyMode = greedy;
        this.trainingEnabled = trainingEnabled;
        this.policyKey = (policyKey == null || policyKey.trim().isEmpty()) ? "train" : policyKey;
        // Auto-targeting disabled via getStrictChooseMode override
        // Avoid terminal spam during eval/benchmark. Enable by raising log level if needed.
        RLTrainer.threadLocalLogger.get().debug("ComputerPlayerRL initialized for " + name
                + (greedy ? " [GREEDY]" : "")
                + (this.trainingEnabled ? "" : " [NO-TRAIN]")
                + ("train".equals(this.policyKey) ? "" : " [" + this.policyKey + "]"));
    }

    // The default constructor for ComputerPlayerRL used by server to create
    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill) {
        this(name, range, RLTrainer.getSharedModel(), false, true, "train");
    }

    public ComputerPlayerRL(final ComputerPlayerRL player) {
        super(player);
        this.model = player.model;
        this.mulliganModel = player.mulliganModel;
        this.trainingBuffer = new ArrayList<>();
        this.decisionCountsByHead = new java.util.HashMap<>();
        this.currentAbility = player.currentAbility;
        this.abilitySourceToExcludeFromMana = null; // Don't copy - each activation sets its own
        this.tapTargetCostReservations = new HashSet<>(); // Don't copy - each activation sets its own
        this.mulligansTaken = player.mulligansTaken;
        this.mulliganEpisode = player.mulliganEpisode;
        this.lastMulliganHandFingerprint = player.lastMulliganHandFingerprint;
        this.lastMulliganHandSize = player.lastMulliganHandSize;
        this.lastMulliganDecisionShouldMulligan = player.lastMulliganDecisionShouldMulligan;
        this.greedyMode = player.greedyMode;
        this.policyKey = player.policyKey;
        this.trainingEnabled = player.trainingEnabled;
        // strict choose mode enforced via method override
    }

    @Override
    public ComputerPlayerRL copy() {
        return new ComputerPlayerRL(this);
    }

    @Override
    public boolean priority(Game game) {
        game.resumeTimer(getTurnControlledBy());
        boolean result;
        try {
            result = priorityPlay(game);
        } catch (Throwable t) {
            // Never let RL decision logic crash the game engine.
            // A single bad activation/choice can otherwise trigger "too many errors" and end the whole game.
            try {
                RLTrainer.threadLocalLogger.get().warn("RL priority() caught exception; forcing pass: " + t.getMessage());
            } catch (Exception ignored) {
                // ignore
            }
            pass(game);
            result = false;
        }
        game.pauseTimer(getTurnControlledBy());
        return result;
    }

    public <T> List<Integer> genericChoose(List<T> candidates, int maxTargets, int minTargets, StateSequenceBuilder.ActionType actionType, Game game, Ability source) {
        trackEarlyLands(game);
        // Candidate-based policy: score up to MAX_CANDIDATES candidates per decision.
        final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

        int candidateCount = Math.min(candidates.size(), maxCandidates);
        if (candidates.size() > maxCandidates) {
            RLTrainer.threadLocalLogger.get().warn(
                    "genericChoose: received " + candidates.size() + " options, truncating to " + maxCandidates);
        }

        // Prevent infinite loops if the engine requests more picks than we kept after truncation
        maxTargets = Math.min(maxTargets, candidateCount);
        minTargets = Math.min(minTargets, candidateCount);

        if (candidateCount == 1) {
            return Arrays.asList(0);
        } else if (candidateCount == 0) {
            return Arrays.asList();
        }

        // Get current phase
        TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;

        // Build base state and cache as current
        StateSequenceBuilder.SequenceOutput baseState = StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);
        this.currentState = baseState;

        // Build padded candidate tensors
        int[] candidateActionIds = new int[maxCandidates];
        float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
        int[] candidateMask = new int[maxCandidates]; // 1 valid, 0 padding

        for (int i = 0; i < candidateCount; i++) {
            candidateMask[i] = 1;
            T cand = candidates.get(i);
            candidateActionIds[i] = computeCandidateActionId(actionType, game, source, cand);
            candidateFeatures[i] = computeCandidateFeatures(actionType, game, source, cand, candFeatDim);
        }

        // Get model predictions - single call for both policy and value
        String headId = headForActionType(actionType);
        int pickIndex = 0; // Phase-1: single-shot scoring (sequential picking comes later)
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info("About to call model.scoreCandidates() with candidateCount: " + candidateCount);
        }
        mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction
                = model.scoreCandidates(
                        baseState,
                        candidateActionIds,
                        candidateFeatures,
                        candidateMask,
                        policyKey,
                        headId,
                        pickIndex,
                        minTargets,
                        maxTargets
                );
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info("Successfully received candidate scoring result");
        }
        float[] actionProbs = prediction.policyScores; // length = maxCandidates
        float valueScore = prediction.valueScores;

        // PPO-critical: mask in logits space then softmax to get a valid categorical distribution.
        float[] logits = new float[candidateCount];
        float maxLogit = -Float.MAX_VALUE;
        for (int i = 0; i < candidateCount; i++) {
            float p = actionProbs[i];
            if (Float.isNaN(p) || Float.isInfinite(p) || p <= 0.0f) {
                p = 1e-20f;
            }
            float logit = (float) Math.log(p);
            if (candidateMask[i] != 1) {
                logit = -1e9f;
            }
            logits[i] = logit;
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }
        float[] maskedProbs = new float[candidateCount];
        float sum = 0.0f;
        for (int i = 0; i < candidateCount; i++) {
            float e = (candidateMask[i] == 1) ? (float) Math.exp(logits[i] - maxLogit) : 0.0f;
            maskedProbs[i] = e;
            sum += e;
        }
        if (sum <= 0.0f || Float.isNaN(sum) || Float.isInfinite(sum)) {
            // Fallback: uniform over valid candidates
            float uniformProb = 1.0f / candidateCount;
            for (int i = 0; i < candidateCount; i++) {
                maskedProbs[i] = uniformProb;
            }
        } else {
            for (int i = 0; i < candidateCount; i++) {
                maskedProbs[i] /= sum;
            }
        }

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info("Action probabilities: " + Arrays.toString(maskedProbs));
            RLTrainer.threadLocalLogger.get().info("Value score: " + valueScore);
        }

        // Store for game logging
        this.lastActionProbs = maskedProbs.clone();
        this.lastValueScore = valueScore;

        // Choose indices (sequential without replacement) + joint old log-prob (PPO-critical)
        List<Integer> selectedIndices = new ArrayList<>();
        boolean[] selected = new boolean[candidateCount];
        float oldLogpTotal = 0.0f;
        Random random = new Random();
        int picks = maxTargets; // historical behavior: pick maxTargets (then ensure >= minTargets via truncation above)
        for (int t = 0; t < picks; t++) {
            float denom = 0.0f;
            for (int i = 0; i < candidateCount; i++) {
                if (!selected[i]) {
                    denom += maskedProbs[i];
                }
            }
            if (!(denom > 0.0f)) {
                // Degenerate: pick first remaining
                int fallback = -1;
                for (int i = 0; i < candidateCount; i++) {
                    if (!selected[i]) {
                        fallback = i;
                        break;
                    }
                }
                if (fallback < 0) {
                    break;
                }
                selected[fallback] = true;
                selectedIndices.add(fallback);
                oldLogpTotal += (float) Math.log(1e-8f);
                continue;
            }

            int pickIdx = -1;
            if (greedyMode) {
                float best = -1.0f;
                for (int i = 0; i < candidateCount; i++) {
                    if (!selected[i] && maskedProbs[i] > best) {
                        best = maskedProbs[i];
                        pickIdx = i;
                    }
                }
            } else {
                float r = random.nextFloat() * denom;
                float c = 0.0f;
                for (int i = 0; i < candidateCount; i++) {
                    if (selected[i]) {
                        continue;
                    }
                    c += maskedProbs[i];
                    if (r <= c) {
                        pickIdx = i;
                        break;
                    }
                }
                if (pickIdx < 0) {
                    // numerical edge: take last remaining
                    for (int i = candidateCount - 1; i >= 0; i--) {
                        if (!selected[i]) {
                            pickIdx = i;
                            break;
                        }
                    }
                }
            }
            if (pickIdx < 0) {
                break;
            }
            float pCond = maskedProbs[pickIdx] / denom;
            oldLogpTotal += (float) Math.log(Math.max(1e-8f, pCond));
            selected[pickIdx] = true;
            selectedIndices.add(pickIdx);
        }

        // Record training data for decisions (store full action + joint log-prob)
        if (trainingEnabled && !selectedIndices.isEmpty()) {
            if (game != null && game.isSimulation()) {
                // Never record training from simulation copies (alt-cost testing, playable checks, etc).
                SIMULATION_TRAINING_SKIPPED.incrementAndGet();
                return selectedIndices;
            }
            int[] chosenIndices = new int[maxCandidates];
            Arrays.fill(chosenIndices, -1);
            int chosenCount = Math.min(selectedIndices.size(), maxCandidates);
            for (int i = 0; i < chosenCount; i++) {
                chosenIndices[i] = selectedIndices.get(i);
            }
            int firstChosen = selectedIndices.get(0);
            double stepReward = computeStepReward(actionType, candidates.get(firstChosen));
            StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                    baseState,
                    candidateCount,
                    candidateActionIds,
                    candidateFeatures,
                    candidateMask,
                    chosenCount,
                    chosenIndices,
                    oldLogpTotal,
                    valueScore,
                    actionType,
                    stepReward
            );
            trainingBuffer.add(td);
            
            // Track decision count by head
            decisionCountsByHead.put(actionType, decisionCountsByHead.getOrDefault(actionType, 0) + 1);
            
            if (actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN) {
                mulliganTraceJsonl(
                        "bottom_td_recorded",
                        "\"method\":\"genericChoose\","
                        + "\"actionType\":\"" + actionType.name() + "\","
                        + "\"candidateCount\":" + candidateCount + ","
                        + "\"chosenCount\":" + chosenCount
                );
            }
        }

        return selectedIndices;
    }

    private static String headForActionType(StateSequenceBuilder.ActionType t) {
        if (t == null) {
            return "action";
        }
        switch (t) {
            case SELECT_TARGETS:
                return "target";
            case LONDON_MULLIGAN:
            case SELECT_CARD:
                return "card_select";
            default:
                return "action";
        }
    }

    private static int toVocabId(String key) {
        if (key == null || key.isEmpty()) {
            return 0;
        }
        int h = key.hashCode();
        int mod = Math.floorMod(h, StateSequenceBuilder.TOKEN_ID_VOCAB - 1);
        return 1 + mod;
    }

    private int computeCandidateActionId(StateSequenceBuilder.ActionType actionType, Game game, Ability source, Object candidate) {
        try {
            String base = actionType.name();
            if (candidate instanceof PassAbility) {
                return toVocabId("PASS");
            }
            if (candidate instanceof mage.abilities.Ability) {
                mage.abilities.Ability ab = (mage.abilities.Ability) candidate;
                MageObject srcObj = game.getObject(ab.getSourceId());
                String srcName = srcObj != null ? srcObj.getName() : "unknown";
                base += ":ABILITY:" + srcName + ":" + ab.getClass().getSimpleName();
            } else if (candidate instanceof java.util.UUID) {
                java.util.UUID tid = (java.util.UUID) candidate;
                MageObject obj = game.getObject(tid);
                if (obj != null) {
                    base += ":TARGET:" + obj.getName();
                } else if (game.getPlayer(tid) != null) {
                    base += ":TARGET:PLAYER";
                } else {
                    base += ":TARGET:UNKNOWN";
                }
            } else if (candidate != null) {
                base += ":" + candidate.getClass().getSimpleName();
            }
            return toVocabId(base);
        } catch (Exception e) {
            return 0;
        }
    }

    private float[] computeCandidateFeatures(StateSequenceBuilder.ActionType actionType, Game game, Ability source, Object candidate, int dim) {
        float[] f = new float[dim];
        try {
            // Generic context
            f[0] = actionType.ordinal() / 16.0f;

            if (candidate instanceof PassAbility) {
                f[1] = 1.0f; // is_pass
                return f;
            }

            // Ability-based candidate features
            if (candidate instanceof mage.abilities.Ability) {
                mage.abilities.Ability ab = (mage.abilities.Ability) candidate;
                MageObject srcObj = game.getObject(ab.getSourceId());
                if (srcObj instanceof Permanent) {
                    Permanent p = (Permanent) srcObj;
                    f[2] = p.isCreature() ? 1.0f : 0.0f;
                    f[3] = p.isLand() ? 1.0f : 0.0f;
                    f[4] = p.isTapped() ? 1.0f : 0.0f;
                    f[5] = p.getPower().getValue() / 10.0f;
                    f[6] = p.getToughness().getValue() / 10.0f;
                }
                // rough target count
                f[7] = ab.getTargets() != null ? ab.getTargets().size() / 5.0f : 0.0f;
                f[8] = ab.isUsesStack() ? 1.0f : 0.0f;
            }

            // Target candidate features
            if (candidate instanceof java.util.UUID) {
                java.util.UUID tid = (java.util.UUID) candidate;
                Player pl = game.getPlayer(tid);
                if (pl != null) {
                    f[10] = 1.0f; // is_player
                    f[11] = pl.getId().equals(this.getId()) ? 1.0f : 0.0f; // is_you
                    f[12] = pl.getLife() / 20.0f;
                } else {
                    Permanent perm = game.getPermanent(tid);
                    if (perm != null) {
                        f[13] = 1.0f; // is_permanent
                        f[14] = perm.isCreature() ? 1.0f : 0.0f;
                        f[15] = perm.isTapped() ? 1.0f : 0.0f;
                        f[16] = perm.getPower().getValue() / 10.0f;
                        f[17] = perm.getToughness().getValue() / 10.0f;
                    }
                }
            }
        } catch (Exception e) {
            // leave zeros
        }
        return f;
    }

    private double computeStepReward(StateSequenceBuilder.ActionType actionType, Object candidate) {
        // Denser reward shaping for better credit assignment
        // Terminal reward (+1/-1) is too sparse for long games

        if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS) {
            // Attacking is generally good - pressures opponent
            return ATTACK_REWARD;
        }

        if (actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS) {
            // Blocking prevents damage (small reward since sometimes not blocking is better)
            return BLOCK_REWARD;
        }

        if (actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL || candidate == null) {
            return 0.0;
        }

        if (candidate instanceof PlayLandAbility) {
            return LAND_PLAY_REWARD;
        }

        if (candidate instanceof SpellAbility) {
            // NOTE: Can't check if creature without Game object
            // Treat all spells equally for now
            return SPELL_CAST_REWARD;
        }

        return 0.0;
    }

    // Helper method to generate all valid combinations of targets
    private void generateCombinations(List<List<Integer>> result, List<Integer> current, int start, int n, int minSize, int maxSize) {
        // If current combination is valid size, add it
        if (current.size() >= minSize && current.size() <= maxSize) {
            result.add(new ArrayList<>(current));
        }

        // If we've reached max size, stop
        if (current.size() >= maxSize) {
            return;
        }

        // Try adding each remaining number
        for (int i = start; i < n; i++) {
            current.add(i);
            generateCombinations(result, current, i + 1, n, minSize, maxSize);
            current.remove(current.size() - 1);
        }
    }

    // Stuff like Opp agent? Investigate further how to handle. Just choosing how to handle multiple replacement effects?
    // TODO
//    @Override
//    public int chooseReplacementEffect(Map<String, String> effectsMap, Map<String, MageObject> objectsMap, Game game) {
//        log.debug("chooseReplacementEffect");
    // Stuff like sheoldred's edict, kozilek's command
    // @Override
    // public Mode chooseMode(Modes modes, Ability source, Game game) {
    //     //TODO: Testing if we can make this not a copy.
    //     ArrayList<UUID> modeIds = new ArrayList<>(modes.values().stream().map(Mode::getId).collect(Collectors.toList()));
    //     for (UUID modeId : modeIds) {
    //         Mode mode = modes.get(modeId);
    //         // Need to do this so target validation is correct
    //         modes.addSelectedMode(mode.getId());
    //         source.getModes().setActiveMode(modeId);
    //         if (!source.getAbilityType().isTriggeredAbility()) {
    //             source.adjustTargets(game);
    //         }
    //         if ((!mode.getTargets().isEmpty() && !mode.getTargets().canChoose(source.getControllerId(), source, game)) || (mode.getCost() != null && !mode.getCost().canPay(source, source, playerId, game))) {
    //             modes.remove(modeId);
    //         }
    //         modes.removeSelectedMode(modeId);
    //     }
    //     int maxTargets = Math.min(modes.getMaxModes(game, source), modes.size());
    //     int minTargets = modes.getMinModes();
    //     List<Integer> targetsToSet = genericChoose(modes.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //     if (targetsToSet.size() == 1){
    //         return (Mode) modes.values().toArray()[0];
    //     } else if(targetsToSet.isEmpty()){
    //         return null;
    //     }else {
    //         // Add all selected modes except the last one
    //         for(int i = 0; i < targetsToSet.size() - 1; i++){
    //             Mode mode = (Mode) modes.values().toArray()[targetsToSet.get(i)];
    //             modes.addSelectedMode(mode.getId());
    //         }
    //         // Return the last selected mode to let outer loop handle it
    //         return (Mode) modes.values().toArray()[targetsToSet.get(targetsToSet.size() - 1)];
    //     }
    // }
    // @Override
    // public int announceXMana(int min, int max, String message, Game game, Ability ability) {
    //     VariableManaCost variableManaCost = null;
    //     for (ManaCost cost : ability.getManaCostsToPay()) {
    //         if (cost instanceof VariableManaCost) {
    //             if (variableManaCost == null) {
    //                 variableManaCost = (VariableManaCost) cost;
    //             } else {
    //                 throw new RuntimeException("More than one VariableManaCost in spell");
    //             }
    //         }
    //     }
    //     if (variableManaCost == null) {
    //         throw new RuntimeException("No VariableManaCost in spell");
    //     }
    //     // Get all possible mana combinations
    //     ManaOptions manaOptions = getManaAvailable(game);
    //     if (manaOptions.isEmpty() && min == 0) {
    //         return 0;
    //     }
    //     // Use a Set to ensure unique X values
    //     Set<Integer> possibleXValuesSet = new HashSet<>();
    //     possibleXValuesSet.add(0); // Always allow X=0
    //     for (Mana mana : manaOptions) {
    //         if (mana instanceof ConditionalMana && !((ConditionalMana) mana).apply(ability, game, getId(), ability.getManaCosts())) {
    //             continue;
    //         }
    //         int availableMana = mana.count() - ability.getManaCostsToPay().manaValue();
    //         for (int x = min; x <= max; x++) {
    //             if (variableManaCost.getXInstancesCount() * x <= availableMana) {
    //                 possibleXValuesSet.add(x);
    //             } else {
    //                 break;
    //             }
    //         }
    //     }
    //     // Convert the Set to a List
    //     List<Integer> possibleXValues = new ArrayList<>(possibleXValuesSet);
    //     // Select the best X value using Q-values
    //     if (!possibleXValues.isEmpty() && possibleXValues.size() > 1) {
    //         List<Integer> targetsToSet = genericChoose(possibleXValues.size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, ability);
    //         return possibleXValues.get(targetsToSet.get(0));
    //     } else if (possibleXValues.size() == 1) {
    //         // No need to query model for only 1 option
    //         return possibleXValues.get(0);
    //     }
    //     return 0; // Default to 0 if no valid options are found
    // }
    // //TODO: Implement
    // //TODO: I don't know when this is used?
    // @Override
    // public int announceXCost(int min, int max, String message, Game game, Ability ability, VariableCost variableCost) {
    //     return super.announceXCost(min, max, message, game, ability, variableCost);
    // }
    // // Deciding to use FOW alt cast, choosing creaturetype for cavern of souls
    // // TODO: Implement
    // @Override
    // public boolean choose(Outcome outcome, Choice choice, Game game) {
    //     // TODO: Allow RLModel to handle this logic
    //     // choose the correct color to pay a spell (use last unpaid ability for color hint)
    //     ManaCost unpaid = null;
    //     if (!getLastUnpaidMana().isEmpty()) {
    //         unpaid = new ArrayList<>(getLastUnpaidMana().values()).get(getLastUnpaidMana().size() - 1);
    //     }
    //     if (outcome == Outcome.PutManaInPool && unpaid != null && choice.isManaColorChoice()) {
    //         if (unpaid.containsColor(ColoredManaSymbol.W) && choice.getChoices().contains("White")) {
    //             choice.setChoice("White");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.R) && choice.getChoices().contains("Red")) {
    //             choice.setChoice("Red");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.G) && choice.getChoices().contains("Green")) {
    //             choice.setChoice("Green");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.U) && choice.getChoices().contains("Blue")) {
    //             choice.setChoice("Blue");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.B) && choice.getChoices().contains("Black")) {
    //             choice.setChoice("Black");
    //             return true;
    //         }
    //         if (unpaid.getMana().getColorless() > 0 && choice.getChoices().contains("Colorless")) {
    //             choice.setChoice("Colorless");
    //             return true;
    //         }
    //     }
    //     // choose by RLModel
    //     Ability source;
    //     if (game.getStack().isEmpty()) {
    //         source = currentAbility;
    //     }else{
    //         source = game.getStack().getFirst().getStackAbility();
    //     }
    //     if (!choice.isChosen()) {
    //         if (choice.getKeyChoices() != null && !choice.getKeyChoices().isEmpty()) {
    //             for (Map.Entry<String, String> entry : choice.getKeyChoices().entrySet()) {
    //                 if (choice.getChoice() == null) {
    //                     choice.setChoice(entry.getKey());
    //                 }
    //             }
    //             //Keychoice
    //             if(choice.getKeyChoices().size() > 1){
    //                 List<Integer> targetsToSet = genericChoose(choice.getKeyChoices().size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //                 choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[targetsToSet.get(0)].toString());
    //                 return true;
    //             } else {
    //                 // Only one choice
    //                 choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[0].toString());
    //                 return true;
    //             }
    //         } else if(choice.getChoices() != null && !choice.getChoices().isEmpty()) {
    //             // Normal Choice
    //             if (choice.getChoices().size() > 1) {
    //                 List<Integer> targetsToSet = genericChoose(choice.getChoices().size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //                 choice.setChoice(choice.getChoices().toArray()[targetsToSet.get(0)].toString());
    //                 return true;
    //             } else {
    //                 choice.setChoice(choice.getChoices().toArray()[0].toString());
    //                 return true;
    //             }
    //         }
    //     }
    //     throw new RuntimeException("No choice made");
    // }
    // Deciding ponder cards, exile card from opponent's hand
    //Choose2
    // @Override
    // public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
    //     if (cards == null || cards.isEmpty()) {
    //         return true;
    //     }
    //     // sometimes a target selection can be made from a player that does not control the ability
    //     UUID abilityControllerId = playerId;
    //     if (target.getTargetController() != null
    //             && target.getAbilityController() != null) {
    //         abilityControllerId = target.getAbilityController();
    //     }
    //     List<Card> cardChoices = new ArrayList<>(cards.getCards(target.getFilter(), abilityControllerId, source, game));
    //     int maxTargets = Math.min(target.getMaxNumberOfTargets(), cardChoices.size());
    //     int minTargets = target.getMinNumberOfTargets();
    //     List<Integer> targetsToSet = genericChoose(cardChoices.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_CARD, game, source);
    //     for (int i = 0; i < targetsToSet.size(); i++) {
    //         target.add(cardChoices.get(targetsToSet.get(i)).getId(), game);
    //     }
    //     return true;
    // }
    // TODO
//    @Override
//    public boolean chooseMulligan(Game game) {
//        log.debug("chooseMulligan");
//        if (hand.size() < 6
//                || isTestsMode() // ignore mulligan in tests
//                || game.getClass().getName().contains("Momir") // ignore mulligan in Momir games
//        ) {
//            return false;
//        }
//        Set<Card> lands = hand.getCards(new FilterLandCard(), game);
//        return lands.size() < 2
//                || lands.size() > hand.size() - 2;
//    }
    // Choosing which stack ability from the stack you want to resolve
    // @Override
    // public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
    //     if (!abilities.isEmpty()) {
    //         if (abilities.size() == 1) {
    //             return abilities.get(0);
    //         }
    //         List<Integer> targetsToSet = genericChoose(abilities.size(),1,1, StateSequenceBuilder.ActionType.SELECT_TRIGGERED_ABILITY, game, null);
    //         return abilities.get(targetsToSet.get(0));
    //     }
    //     return null;
    // }
    // Examples:
    // Damage assignment from fury
    // ((TargetCreatureOrPlaneswalkerAmount) target).getAmountRemaining()
    // @Override
    // public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
    //     // TODO: Investigate what calls this
    //     return super.chooseTargetAmount(outcome, target, source, game);
    //     //return choose(outcome, target, source, game, null);
    // }
    // TODO: This breaks on mulligans? Because there is no active player?
    // Examples: Return card from graveyard to hand,
    // @Override
    // public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
    //     return choose(outcome, target, source, game, null);
    // }
    // Examples: Choosing when searching library. Fetch lands
//     @Override
//     public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
//         if (cards == null || cards.isEmpty()) {
//             return target.isRequired(source);
//         }
//         // sometimes a target selection can be made from a player that does not control the ability
//         UUID abilityControllerId = playerId;
//         if (target.getTargetController() != null
//                 && target.getAbilityController() != null) {
//             abilityControllerId = target.getAbilityController();
//         }
//         // we still use playerId when getting cards even if they don't control the search
//         List<Card> cardChoices = new ArrayList<>(cards.getCards(target.getFilter(), playerId, source, game));
//         // TODO: Fetchlands incorrectly state mintargets = 1 but you can "fail to find"
//         int maxTargets = target.getMaxNumberOfTargets();
//         int minTargets = target.getMinNumberOfTargets();
//         List<Integer> targetsToSet = genericChoose(cardChoices.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source);
//         for (int i = 0; i < targetsToSet.size(); i++) {
//             // TODO: For some reason this always fails because the card zone is OUTSIDE
//             // Pretty important to fix this for computerPlayer because I think they always fail to find
//             // so they will be rly bad, could just be with how I'm setting the game up?
// //            if (target.canTarget(abilityControllerId, card.getId(), source, game)) {
//             target.add(cardChoices.get(targetsToSet.get(i)).getId(), game);
//         }
//         return true;
//     }
    // Examples:
    // Discarding to hand size, Choosing to keep which legend for legend rule
    // @Override
    // public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
    //     return choose(outcome, target, source, game, null);
    // }
    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        trackEarlyLands(game);
        // Special case: London mulligan card selection
        // TODO: I don't think this check is sufficient to say its a mulligan
        if (source == null && outcome == Outcome.Discard && target instanceof mage.target.common.TargetCardInHand) {
            return chooseLondonMulliganCards(target, game);
        }
        // RL-only target selection. No engine fallback.
        UUID abilityControllerId = playerId;
        if (target.getTargetController() != null && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }

        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());

        java.util.HashSet<UUID> chosen = new java.util.HashSet<>();
        int chosenCount = 0;
        while (chosenCount < maxTargets) {
            java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID ctrlId = abilityControllerId;
            possible.removeIf(id -> id == null || chosen.contains(id) || !target.canTarget(ctrlId, id, source, game));

            boolean allowStop = chosenCount >= minTargets;
            if (allowStop) {
                possible.add(0, null); // STOP sentinel
            }

            if (possible.isEmpty()) {
                break;
            }

            UUID picked = null;
            if (possible.size() == 1) {
                picked = possible.get(0);
            } else {
                try {
                    TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;
                    StateSequenceBuilder.SequenceOutput baseState = StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);
                    final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                    final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

                    int candidateCount = Math.min(possible.size(), maxCandidates);
                    int[] candidateActionIds = new int[maxCandidates];
                    float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
                    int[] candidateMask = new int[maxCandidates];
                    for (int i = 0; i < candidateCount; i++) {
                        candidateMask[i] = 1;
                        UUID cand = possible.get(i);
                        candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source, cand);
                        candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source, cand, candFeatDim);
                    }

                    mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                            baseState,
                            candidateActionIds,
                            candidateFeatures,
                            candidateMask,
                            policyKey,
                            "target",
                            chosenCount,
                            minTargets,
                            maxTargets
                    );
                    float[] actionProbs = prediction.policyScores;
                    float valueScore = prediction.valueScores;

                    int chosenIdx = candidateCount - 1;
                    if (greedyMode) {
                        float best = -1.0f;
                        for (int i = 0; i < candidateCount; i++) {
                            float p = actionProbs[i];
                            if (Float.isNaN(p) || Float.isInfinite(p) || p < 0.0f) {
                                p = 0.0f;
                            }
                            if (p > best) {
                                best = p;
                                chosenIdx = i;
                            }
                        }
                    } else {
                        // Sample from returned probabilities (already masked/renormalized by Python)
                        float r = new java.util.Random().nextFloat();
                        float cdf = 0.0f;
                        for (int i = 0; i < candidateCount; i++) {
                            float p = actionProbs[i];
                            if (Float.isNaN(p) || Float.isInfinite(p) || p < 0.0f) {
                                p = 0.0f;
                            }
                            cdf += p;
                            if (r <= cdf) {
                                chosenIdx = i;
                                break;
                            }
                        }
                    }
                    picked = possible.get(chosenIdx);

                    // Track decision count by head
                    decisionCountsByHead.put(StateSequenceBuilder.ActionType.SELECT_TARGETS, 
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.SELECT_TARGETS, 0) + 1);

                    // Gamelog: target-pick decision
                    try {
                        GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                        if (gameLogger != null && gameLogger.isEnabled()) {
                            int turn = game.getTurnNum();
                            if (turn != lastLoggedTurn) {
                                String activePlayerName = game.getActivePlayerId() != null
                                        ? game.getPlayer(game.getActivePlayerId()).getName()
                                        : "Unknown";
                                gameLogger.logTurnStart(turn, activePlayerName, formatGameState(game));
                                lastLoggedTurn = turn;
                            }
                            String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                            String activePlayerName = game.getActivePlayerId() != null
                                    ? game.getPlayer(game.getActivePlayerId()).getName()
                                    : "Unknown";
                            List<String> optionNames = new ArrayList<>();
                            for (int i = 0; i < candidateCount; i++) {
                                UUID tid = possible.get(i);
                                optionNames.add(tid == null ? "STOP" : describeTargetWithOwner(tid, game));
                            }
                            String selectedName = picked == null ? "STOP" : describeTargetWithOwner(picked, game);
                            gameLogger.logDecision(
                                    this.getName(),
                                    activePlayerName,
                                    phase + " (TARGET_PICK " + chosenCount + " min=" + minTargets + " max=" + maxTargets + ")",
                                    turn,
                                    formatGameState(game),
                                    optionNames,
                                    Arrays.copyOf(actionProbs, candidateCount),
                                    valueScore,
                                    chosenIdx,
                                    selectedName
                            );
                        }
                    } catch (Exception ignored) {
                    }
                } catch (Exception e) {
                    // Deterministic internal fallback (no engine)
                    picked = possible.get(0);
                }
            }

            if (picked == null) { // STOP
                break;
            }
            target.addTarget(picked, source, game);
            chosen.add(picked);
            chosenCount++;
        }

        // Ensure minimum targets if required (deterministic fill, no engine)
        while (chosenCount < minTargets) {
            java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID ctrlId = abilityControllerId;
            possible.removeIf(id -> id == null || chosen.contains(id) || !target.canTarget(ctrlId, id, source, game));
            if (possible.isEmpty()) {
                break;
            }
            UUID picked = possible.get(0);
            target.addTarget(picked, source, game);
            chosen.add(picked);
            chosenCount++;
        }

        return chosenCount >= minTargets;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        boolean result = super.choose(outcome, target, source, game);
        // Avoid terminal spam during eval/benchmark: targets are logged via GameLogger decisions.
        // Keep the old terminal diagnostics only when explicitly enabled.
        if (ACTIVATION_DIAG && result && target != null && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().info(
                        "Player " + getName() + " chose target: " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        // RL-only selection from a provided visible card set. No engine fallback.
        if (cards == null || target == null) {
            return false;
        }
        
        // CRITICAL: Filter cards by target's filter (e.g., Islandcycling only shows Islands)
        List<Card> filteredCards;
        if (target.getFilter() != null) {
            // Use the target's filter to get only valid cards
            filteredCards = new ArrayList<>(cards.getCards(target.getFilter(), this.getId(), source, game));
        } else {
            filteredCards = new ArrayList<>(cards.getCards(game));
        }
        
        if (filteredCards.isEmpty()) {
            return false;
        }

        // Deduplicate by card name for zones where cards have no state
        // Battlefield cards have state (tapped/untapped, counters, etc.) that matters
        boolean shouldDedupe = false;
        if (!filteredCards.isEmpty()) {
            Card firstCard = filteredCards.get(0);
            if (firstCard != null) {
                // Dedupe library and hand - cards are stateless (no tapped/untapped, counters, etc.)
                // Don't dedupe battlefield (state matters) or graveyard (timestamp/order matters)
                mage.constants.Zone zone = game.getState().getZone(firstCard.getId());
                shouldDedupe = (zone == mage.constants.Zone.LIBRARY || zone == mage.constants.Zone.HAND);
            }
        }

        List<Card> choices;
        if (shouldDedupe) {
            // Deduplicate by card name - model doesn't need to distinguish between Island #1 and Island #14
            java.util.LinkedHashMap<String, Card> uniqueCards = new java.util.LinkedHashMap<>();
            for (Card card : filteredCards) {
                if (card != null) {
                    String cardName = card.getName();
                    if (!uniqueCards.containsKey(cardName)) {
                        uniqueCards.put(cardName, card);
                    }
                }
            }
            choices = new ArrayList<>(uniqueCards.values());
        } else {
            // Keep all cards - state matters (e.g., tapped vs untapped creatures)
            choices = filteredCards;
        }
        
        if (choices.isEmpty()) {
            return false;
        }

        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());
        maxTargets = Math.min(maxTargets, choices.size());
        minTargets = Math.min(minTargets, choices.size());

        java.util.HashSet<UUID> chosen = new java.util.HashSet<>();
        int chosenCount = 0;
        while (chosenCount < maxTargets) {
            List<Card> remaining = new ArrayList<>();
            for (Card c : choices) {
                if (c != null && !chosen.contains(c.getId())) {
                    remaining.add(c);
                }
            }
            if (remaining.isEmpty()) {
                break;
            }

            boolean allowStop = chosenCount >= minTargets;
            if (allowStop) {
                remaining.add(0, null); // STOP sentinel
            }

            Card picked = null;
            if (remaining.size() == 1) {
                picked = remaining.get(0);
            } else {
                try {
                    TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;
                    StateSequenceBuilder.SequenceOutput baseState = StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);
                    final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                    final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

                    int candidateCount = Math.min(remaining.size(), maxCandidates);
                    int[] candidateActionIds = new int[maxCandidates];
                    float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
                    int[] candidateMask = new int[maxCandidates];
                    for (int i = 0; i < candidateCount; i++) {
                        candidateMask[i] = 1;
                        Card cand = remaining.get(i);
                        UUID cid = cand == null ? null : cand.getId();
                        candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.SELECT_CARD, game, source, cid);
                        candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.SELECT_CARD, game, source, cid, candFeatDim);
                    }

                    mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                            baseState,
                            candidateActionIds,
                            candidateFeatures,
                            candidateMask,
                            policyKey,
                            "card_select",
                            chosenCount,
                            minTargets,
                            maxTargets
                    );
                    float[] actionProbs = prediction.policyScores;
                    float valueScore = prediction.valueScores;
                    int chosenIdx = candidateCount - 1;
                    if (greedyMode) {
                        float best = -1.0f;
                        for (int i = 0; i < candidateCount; i++) {
                            float p = actionProbs[i];
                            if (Float.isNaN(p) || Float.isInfinite(p) || p < 0.0f) {
                                p = 0.0f;
                            }
                            if (p > best) {
                                best = p;
                                chosenIdx = i;
                            }
                        }
                    } else {
                        float r = new java.util.Random().nextFloat();
                        float cdf = 0.0f;
                        for (int i = 0; i < candidateCount; i++) {
                            float p = actionProbs[i];
                            if (Float.isNaN(p) || Float.isInfinite(p) || p < 0.0f) {
                                p = 0.0f;
                            }
                            cdf += p;
                            if (r <= cdf) {
                                chosenIdx = i;
                                break;
                            }
                        }
                    }
                    picked = remaining.get(chosenIdx);

                    // Track decision count by head
                    decisionCountsByHead.put(StateSequenceBuilder.ActionType.SELECT_CARD, 
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.SELECT_CARD, 0) + 1);

                    // Gamelog: card-pick decision
                    try {
                        GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                        if (gameLogger != null && gameLogger.isEnabled()) {
                            int turn = game.getTurnNum();
                            if (turn != lastLoggedTurn) {
                                String activePlayerName = game.getActivePlayerId() != null
                                        ? game.getPlayer(game.getActivePlayerId()).getName()
                                        : "Unknown";
                                gameLogger.logTurnStart(turn, activePlayerName, formatGameState(game));
                                lastLoggedTurn = turn;
                            }
                            String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                            String activePlayerName = game.getActivePlayerId() != null
                                    ? game.getPlayer(game.getActivePlayerId()).getName()
                                    : "Unknown";
                            List<String> optionNames = new ArrayList<>();
                            for (int i = 0; i < candidateCount; i++) {
                                Card c = remaining.get(i);
                                optionNames.add(c == null ? "STOP" : c.getName());
                            }
                            String selectedName = picked == null ? "STOP" : picked.getName();
                            gameLogger.logDecision(
                                    this.getName(),
                                    activePlayerName,
                                    phase + " (CARD_PICK " + chosenCount + " min=" + minTargets + " max=" + maxTargets + ")",
                                    turn,
                                    formatGameState(game),
                                    optionNames,
                                    Arrays.copyOf(actionProbs, candidateCount),
                                    valueScore,
                                    chosenIdx,
                                    selectedName
                            );
                        }
                    } catch (Exception ignored) {
                    }
                } catch (Exception e) {
                    picked = remaining.get(0);
                }
            }

            if (picked == null) { // STOP
                break;
            }

            target.addTarget(picked.getId(), source, game);
            chosen.add(picked.getId());
            chosenCount++;
        }

        while (chosenCount < minTargets) {
            Card fallback = null;
            for (Card c : choices) {
                if (c != null && !chosen.contains(c.getId())) {
                    fallback = c;
                    break;
                }
            }
            if (fallback == null) {
                break;
            }
            target.addTarget(fallback.getId(), source, game);
            chosen.add(fallback.getId());
            chosenCount++;
        }

        return chosenCount >= minTargets;
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        // ALWAYS log when choose() is called, regardless of conditions
        if (ACTIVATION_DIAG) {
            String choiceInfo;
            if (choice != null) {
                boolean isKeyChoice = choice.isKeyChoice();
                int choiceCount = isKeyChoice ? choice.getKeyChoices().size() : choice.getChoices().size();
                choiceInfo = " isKeyChoice=" + isKeyChoice + " count=" + choiceCount;
                if (isKeyChoice && choiceCount > 0) {
                    choiceInfo += " keys=" + choice.getKeyChoices().keySet();
                } else if (!isKeyChoice && choiceCount > 0) {
                    choiceInfo += " choices=" + choice.getChoices();
                }
            } else {
                choiceInfo = " (null choice)";
            }
            RLTrainer.threadLocalLogger.get().info(
                    "CHOOSE: ENTRY - outcome=" + outcome + choiceInfo
            );
        }

        if (ACTIVATION_DIAG && choice != null && choice.getChoices() != null && choice.getChoices().size() > 1) {
            RLTrainer.threadLocalLogger.get().info(
                    "CHOOSE: Called with " + choice.getChoices().size() + " options: " + choice.getChoices()
            );
        }

        // Detect alternative cost choices (they use KEY-based choices!)
        if (choice != null && choice.isKeyChoice() && choice.getKeyChoices() != null && choice.getKeyChoices().size() > 1) {
            // Check if any of the key choice VALUES contain "alternative cost"
            boolean hasAlternativeCost = false;
            for (String choiceValue : choice.getKeyChoices().values()) {
                if (choiceValue != null && choiceValue.contains("alternative cost")) {
                    hasAlternativeCost = true;
                    break;
                }
            }

            if (hasAlternativeCost) {
                if (ACTIVATION_DIAG) {
                    RLTrainer.threadLocalLogger.get().info(
                            "CHOOSE: Alternative cost detected, currentAbility=" + currentAbility
                            + ", forcedChoice=" + forcedAlternativeChoice.get()
                            + ", trackingData=" + choiceTrackingData.get()
                    );
                }

                // Track available options for simulation testing (using KEYS)
                if (choiceTrackingData.get() == null) {
                    ChoiceTrackingData tracking = new ChoiceTrackingData();
                    tracking.availableOptions = new HashSet<>(choice.getKeyChoices().keySet());
                    choiceTrackingData.set(tracking);
                    if (ACTIVATION_DIAG) {
                        RLTrainer.threadLocalLogger.get().info(
                                "CHOOSE: Created new tracking data with key options: " + tracking.availableOptions
                        );
                    }
                }

                // Check if we're forcing a specific choice (during testing)
                String forced = forcedAlternativeChoice.get();
                if (forced != null) {
                    if (choice.getKeyChoices().containsKey(forced)) {
                        choice.setChoiceByKey(forced);
                        ChoiceTrackingData tracking = choiceTrackingData.get();
                        if (tracking != null) {
                            tracking.choiceMade = forced;
                        }
                        if (ACTIVATION_DIAG) {
                            RLTrainer.threadLocalLogger.get().info(
                                    "CHOOSE: Forced choice key " + forced + " -> " + choice.getChoice()
                            );
                        }
                        return true;
                    }
                }

                // During real activation, constrain to only valid choices
                if (currentAbility != null) {
                    Set<String> validChoices = validAlternativeCosts.get(currentAbility.getSourceId());
                    if (validChoices != null && !validChoices.isEmpty()) {
                        // Pick the first valid choice key
                        for (String key : choice.getKeyChoices().keySet()) {
                            if (validChoices.contains(key)) {
                                choice.setChoiceByKey(key);
                                if (ACTIVATION_DIAG) {
                                    RLTrainer.threadLocalLogger.get().info(
                                            "Player " + getName() + " chose (validated alternative) key=" + key
                                            + " -> " + choice.getChoice() + " (from " + validChoices.size()
                                            + " valid options, sourceId=" + currentAbility.getSourceId() + ")"
                                    );
                                }
                                return true;
                            }
                        }
                    }
                }

                // Default: use parent class logic (random)
                boolean result = super.choose(outcome, choice, game);
                if (result && choice.isChosen()) {
                    // Track which KEY was chosen
                    String chosenKey = choice.getChoiceKey();
                    ChoiceTrackingData tracking = choiceTrackingData.get();
                    if (tracking != null) {
                        tracking.choiceMade = chosenKey;
                        if (ACTIVATION_DIAG) {
                            RLTrainer.threadLocalLogger.get().info(
                                    "CHOOSE: Tracked choice key=" + chosenKey + " -> " + choice.getChoice()
                            );
                        }
                    } else {
                        if (ACTIVATION_DIAG) {
                            RLTrainer.threadLocalLogger.get().info(
                                    "CHOOSE: WARNING - tracking is null after parent.choose()!"
                            );
                        }
                    }
                }
                return result;
            }
        }

        boolean result = super.choose(outcome, choice, game);
        if (result && choice.isChosen()) {
            RLTrainer.threadLocalLogger.get().debug(
                    "Player " + getName() + " chose: " + choice.getChoiceKey() + " -> " + choice.getChoice()
            );
        }
        return result;
    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        boolean result = super.chooseTargetAmount(outcome, target, source, game);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                int amount = target.getTargetAmount(targetId);
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().debug(
                        "Player " + getName() + " chose target amount: " + targetName + " (" + targetId + "), amount: " + amount
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
        boolean result = super.choose(outcome, target, source, game, options);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().debug(
                        "Player " + getName() + " chose target with options: " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    // @Override
    // public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
    //     UUID abilityControllerId = playerId;
    //     if (target.getTargetController() != null && target.getAbilityController() != null) {
    //         abilityControllerId = target.getAbilityController();
    //     }
    //     // TODO: I guess we can make this an ai decision?
    //     if (Objects.equals(target.getTargetName(), "starting player")) {
    //         return super.choose(outcome, target, source, game, null);
    //     }
    //     List<UUID> possibleTargetsList = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
    //     // Remove targets that can't be targeted
    //     for (UUID possibleTarget : possibleTargetsList) {
    //         if (!target.canTarget(abilityControllerId, possibleTarget, source, game)) {
    //             possibleTargetsList.remove(possibleTarget);
    //         }
    //     }
    //     int maxTargets = Math.min(target.getMaxNumberOfTargets(), possibleTargetsList.size());
    //     int minTargets = target.getMinNumberOfTargets();
    //     List<Integer> qValues = genericChoose(possibleTargetsList.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source);
    //     for (int i = 0; i < qValues.size(); i++) {
    //         target.add(possibleTargetsList.get(qValues.get(i)), game);
    //     }
    //     return true;
    // }
//    @Override
//    public void selectAttackers(Game game, UUID attackingPlayerId) {
//        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
//        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) {
//            // Generate list of possible attackers
//            List<Permanent> allAttackers = game.getBattlefield().getAllActivePermanents(
//                StaticFilters.FILTER_PERMANENT_CREATURE,
//                attackingPlayerId,
//                game
//            );
//            List<Permanent> possibleAttackers = new ArrayList<>();
//
//            for (Permanent creature : allAttackers) {
//                if (creature.canAttack(null, game)) {
//                    possibleAttackers.add(creature);
//                }
//            }
//
//            if (possibleAttackers.isEmpty()) {
//                return;
//            }
//
//            currentState = StateSequenceBuilder.build(game,
//                                                      StateSequenceBuilder.ActionType.DECLARE_ATTACKS,
//                                                      game.getPhase().getType(),
//                                                      StateSequenceBuilder.MAX_LEN);
//            stateBuffer.add(currentState);
//            // Generate list of attack targets (Player, planeswalkers, battles)
//            List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
//            if (possibleAttackers.size() > RLModel.MAX_ACTIONS) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
//            }
//            if (possibleAttackTargets.size() > RLModel.MAX_OPTIONS - 1) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attack targets than max options, Model truncating");
//            }
//            int numAttackers = Math.min(RLModel.MAX_ACTIONS, possibleAttackers.size());
//            // -1 to reserve the option to not attack
//            int numAttackTargets = Math.min(RLModel.MAX_OPTIONS-1, possibleAttackTargets.size());
//
//            // predict logits once for the whole batch
//            INDArray qValues = model.predictDistribution(currentState, true)
//                                    .reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
//
//            // for each attacker we'll record its chosen index separately
//            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
//                Permanent attacker = possibleAttackers.get(attackerIndex);
//
//                // Create a list of defender indices with their Q-values for this attacker
//                List<AttackOption> attackOptions = new ArrayList<>();
//                for (int attackTargetIndex = 0; attackTargetIndex < RLModel.MAX_OPTIONS; attackTargetIndex++) {
//                    float qValue = qValues.getFloat(attackerIndex, attackTargetIndex);
//                    attackOptions.add(new AttackOption(attackTargetIndex, attackerIndex, qValue));
//                }
//
//                // Sort attack options by Q-value in descending order
//                attackOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
//
//                // Declare attacks based on sorted Q-values
//                for (AttackOption option : attackOptions) {
//                    if (option.attackTargetIndex >= numAttackTargets) {
//                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this attacker if the first choice is to not attack
//                    }
//                    UUID attackTargetId = possibleAttackTargets.get(option.attackTargetIndex);
//                    if (attacker.canAttack(attackTargetId, game)) {
//                        RLTrainer.threadLocalLogger.get().info("Declaring attacker: " + attacker.getName() + " for attack target: " + attackTargetId.toString());
//                        this.declareAttacker(attacker.getId(), attackTargetId, game, false);
//                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Once an attack is declared, move to the next attacker
//                    }
//                }
//            }
//        }
//    }
//    @Override
//    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
//        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
//        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) {
//            List<Permanent> attackers = getAttackers(game);
//            if (attackers == null) {
//                return;
//            }
//
//            List<Permanent> possibleBlockers = super.getAvailableBlockers(game);
//            possibleBlockers = filterOutNonblocking(game, attackers, possibleBlockers);
//            if (possibleBlockers.isEmpty()) {
//                return;
//            }
//
//            RLTrainer.threadLocalLogger.get().info("possibleBlockers: " + possibleBlockers);
//
//            attackers = filterOutUnblockable(game, attackers, possibleBlockers);
//            if (attackers.isEmpty()) {
//                return;
//            }
//
//            currentState = StateSequenceBuilder.build(game,
//                                                      StateSequenceBuilder.ActionType.DECLARE_BLOCKS,
//                                                      game.getPhase().getType(),
//                                                      StateSequenceBuilder.MAX_LEN);
//            stateBuffer.add(currentState);
//            // -1 to reserve the option to not block nothing no a creature. Essentially an attacker that is "nothing"
//            int numAttackers = Math.min(RLModel.MAX_ACTIONS - 1, attackers.size());
//            int numBlockers = Math.min(RLModel.MAX_OPTIONS, possibleBlockers.size());
//            if (attackers.size() > RLModel.MAX_ACTIONS - 1) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
//            }
//            if (possibleBlockers.size() > RLModel.MAX_OPTIONS) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More blockers than max actions, Model truncating");
//            }
//
//            // Build exploration dimensions
//            // +1 to explore the option to not block
//            for(int i = 0; i < numAttackers + 1; i++){
//                // exploration metadata skipped
//            }
//            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
//
//            boolean blockerDeclared = false;
//
//            // Iterate over blockers first
//            // Attacker = X, Blockers = Y
//            for (int blockerIndex = 0; blockerIndex < numBlockers; blockerIndex++) {
//                Permanent blocker = possibleBlockers.get(blockerIndex);
//
//                // Create a list of blocker indices with their Q-values for this attacker
//                List<BlockOption> blockOptions = new ArrayList<>();
//                // We use the full MAX_OPTIONS because we need to reserve the option to not block
//                for (int attackerIndex = 0; attackerIndex < RLModel.MAX_ACTIONS; attackerIndex++) {
//                    float qValue = qValues.getFloat(attackerIndex, blockerIndex);
//                    blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
//                }
//
//                // Sort block options by Q-value in descending order
//                blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
//
//                // Declare blocks based on sorted Q-values
//                for (BlockOption option : blockOptions) {
//                    if (option.attackerIndex >= numAttackers) {
//                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this blocker if the first choice is to not block
//                    }
//
//                    Permanent attacker = attackers.get(option.attackerIndex);
//                    if (blocker.canBlock(attacker.getId(), game)) {
//                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
//                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
//                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this blocker if the first choice is to not block
//                    }
//                }
//            }
//            if (blockerDeclared) {
//                game.getPlayers().resetPassed();
//            }
//            // skip training metadata cleanup
//        }
//    }
    private List<Permanent> filterOutNonblocking(Game game, List<Permanent> attackers, List<Permanent> blockers) {
        List<Permanent> blockersLeft = new ArrayList<>();
        for (Permanent blocker : blockers) {
            for (Permanent attacker : attackers) {
                if (blocker.canBlock(attacker.getId(), game)) {
                    blockersLeft.add(blocker);
                    break;
                }
            }
        }
        return blockersLeft;
    }

    private List<Permanent> filterOutUnblockable(Game game, List<Permanent> attackers, List<Permanent> blockers) {
        List<Permanent> attackersLeft = new ArrayList<>();
        for (Permanent attacker : attackers) {
            if (CombatUtil.canBeBlocked(game, attacker, blockers)) {
                attackersLeft.add(attacker);
            }
        }
        return attackersLeft;
    }

    private List<Permanent> getAttackers(Game game) {
        Set<UUID> attackersUUID = game.getCombat().getAttackers();
        if (attackersUUID.isEmpty()) {
            return null;
        }

        List<Permanent> attackers = new ArrayList<>();
        for (UUID attackerId : attackersUUID) {
            Permanent permanent = game.getPermanent(attackerId);
            attackers.add(permanent);
        }
        return attackers;
    }

    @Override
    public boolean chooseMulligan(Game game) {
        try {
            int handSizeNow = getHand() != null ? getHand().size() : -1;
            int handFp = computeHandFingerprint(game);
            if (MULLIGAN_TRACE) {
                mulliganTrainingLog(String.format(
                        "MULLIGAN_TRACE chooseMulligan ep=%d player=%s mulligansTaken=%d handSize=%d",
                        currentEpisode,
                        getName(),
                        mulligansTaken,
                        handSizeNow
                ));
            }
            mulliganTraceJsonl(
                    "keep_mull_prompt",
                    "\"method\":\"chooseMulligan\","
                    + "\"handSize\":" + handSizeNow + ","
                    + "\"handFingerprint\":" + handFp
            );

            // Duplicate-call protection: if engine asks again for same prompt/hand, reuse and don't record training twice.
            if (lastMulliganDecisionShouldMulligan != null
                    && lastMulliganHandSize == handSizeNow
                    && lastMulliganHandFingerprint == handFp) {
                mulliganTraceJsonl(
                        "duplicate_reuse",
                        "\"method\":\"chooseMulligan\","
                        + "\"shouldMulligan\":" + lastMulliganDecisionShouldMulligan
                );
                return lastMulliganDecisionShouldMulligan;
            }

            MulliganModel.MulliganDecision decision = mulliganModel.shouldMulliganWithFeatures(this, game, mulligansTaken, mulliganEpisode);

            int landCount = countLandsInHand(game);
            boolean trainingRecorded = false;
            float actionTaken = decision.shouldMulligan ? 0.0f : 1.0f; // 0=mull, 1=keep (action taken by engine)
            // For training, record the ORIGINAL model decision (before override) so bad decisions get punished
            float trainingLabel = decision.originalModelDecision ? 0.0f : 1.0f; // What the model wanted to do

            // Gamelog: mulligan decision details
            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    String cards = "";
                    try {
                        List<Card> handCards = new ArrayList<>(getHand().getCards(game));
                        cards = handCards.stream().map(Card::getName).collect(Collectors.joining("; "));
                    } catch (Exception ignored) {
                        cards = "";
                    }
                    gameLogger.log(String.format(
                            "MULLIGAN_DECISION: player=%s mulligansTaken=%d handSize=%d lands=%d decision=%s Q_keep=%.3f Q_mull=%.3f hand=[%s]",
                            getName(),
                            mulligansTaken,
                            handSizeNow,
                            landCount,
                            decision.shouldMulligan ? "MULLIGAN" : "KEEP",
                            decision.qKeep,
                            decision.qMull,
                            trunc(cards, 400)
                    ));
                }
            } catch (Exception ignored) {
                // don't fail mulligan for logging
            }

            mulliganTrainingLog(String.format(
                    "MULLIGAN_DECISION ep=%d player=%s mulligansTaken=%d handSize=%d lands=%d decision=%s Q_keep=%.3f Q_mull=%.3f action=%.1f%s",
                    currentEpisode,
                    getName(),
                    mulligansTaken,
                    handSizeNow,
                    landCount,
                    decision.shouldMulligan ? "MULLIGAN" : "KEEP",
                    decision.qKeep,
                    decision.qMull,
                    actionTaken,
                    decision.wasOverridden ? " [OVERRIDDEN]" : ""
            ));

            // Record mulligan-model training label immediately (only if hand size is sane at this prompt).
            // Use the ORIGINAL model decision (before override) so the model learns from its mistakes.
            if (trainingEnabled && handSizeNow > 0 && handSizeNow <= 7) {
                mulliganFeatures.add(decision.features);
                mulliganDecisions.add(trainingLabel); // Original model decision, not overridden action
                mulliganOverrides.add(decision.wasOverridden); // Track if this decision was overridden
                mulliganLandCounts.add(landCount);
                trainingRecorded = true;
            } else if (trainingEnabled) {
                mulliganTrainingLog(String.format(
                        "MULLIGAN_WARN ep=%d player=%s mulligansTaken=%d handSize=%d -> skip_mulligan_training_record",
                        currentEpisode,
                        getName(),
                        mulligansTaken,
                        handSizeNow
                ));
            }

            mulliganTraceJsonl(
                    "decision",
                    "\"method\":\"chooseMulligan\","
                    + "\"handSize\":" + handSizeNow + ","
                    + "\"landCount\":" + landCount + ","
                    + "\"handIdsHash\":" + Arrays.hashCode(decision.handCardIds) + ","
                    + "\"deckIdsHash\":" + Arrays.hashCode(decision.deckCardIds) + ","
                    + "\"shouldMulligan\":" + decision.shouldMulligan + ","
                    + "\"qKeep\":" + decision.qKeep + ","
                    + "\"qMull\":" + decision.qMull + ","
                    + "\"actionTaken\":" + actionTaken + ","
                    + "\"trainingRecorded\":" + trainingRecorded
            );

            // Log to CSV once per keep/mull decision (self-contained: current hand only)
            String allCardsCsv = "";
            try {
                List<Card> handCards = new ArrayList<>(getHand().getCards(game));
                allCardsCsv = handCards.stream().map(Card::getName).collect(Collectors.joining("; "));
            } catch (Exception ignored) {
                allCardsCsv = "";
            }
            MulliganLogger.getInstance().logDecision(
                    currentEpisode,
                    getName(),
                    decision.mulliganNum,
                    handSizeNow,
                    decision.shouldMulligan ? "MULLIGAN" : "KEEP",
                    decision.qKeep,
                    decision.qMull,
                    allCardsCsv,
                    "", // keptCards (not tracked here)
                    "" // bottomedCards (not tracked here)
            );

            // Cache decision for duplicate prompts.
            lastMulliganHandFingerprint = handFp;
            lastMulliganHandSize = handSizeNow;
            lastMulliganDecisionShouldMulligan = decision.shouldMulligan;

            // Advance mulligansTaken only when we actually take a mulligan.
            if (decision.shouldMulligan) {
                mulligansTaken++;
            }
            return decision.shouldMulligan;
        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn("Error in mulligan model, using fallback: " + e.getMessage());
            mulliganTraceJsonl(
                    "exception",
                    "\"method\":\"chooseMulligan\","
                    + "\"exception\":\"" + jsonEscape(e.getClass().getSimpleName() + ": " + e.getMessage()) + "\""
            );
            // Fallback: mulligan only if 0-1 lands or 6+ lands
            int landCount = countLandsInHand(game);
            boolean shouldMulligan = landCount <= 1 || landCount >= 6;
            int handSizeNow = getHand() != null ? getHand().size() : -1;
            int handFp = computeHandFingerprint(game);
            lastMulliganHandFingerprint = handFp;
            lastMulliganHandSize = handSizeNow;
            lastMulliganDecisionShouldMulligan = shouldMulligan;
            if (shouldMulligan) {
                mulligansTaken++;
            }
            return shouldMulligan;
        }
    }

    private boolean chooseLondonMulliganCards(Target target, Game game) {
        try {
            int hs = getHand() != null ? getHand().size() : -1;
            int numToPutBack = target != null ? target.getMinNumberOfTargets() : -1;

            if (MULLIGAN_TRACE) {
                mulliganTrainingLog(String.format(
                        "MULLIGAN_TRACE chooseLondonBottom ep=%d player=%s mulligansTaken=%d handSize=%d bottomN=%d",
                        currentEpisode,
                        getName(),
                        mulligansTaken,
                        hs,
                        numToPutBack
                ));
            }
            mulliganTraceJsonl(
                    "bottom_prompt",
                    "\"method\":\"chooseLondonMulliganCards\","
                    + "\"handSize\":" + hs + ","
                    + "\"numToPutBack\":" + numToPutBack
            );

            List<Card> hand = new ArrayList<>(getHand().getCards(game));
            if (hand.isEmpty()) {
                return false;
            }

            if (numToPutBack <= 0) {
                mulliganTraceJsonl(
                        "bottom_result",
                        "\"method\":\"chooseLondonMulliganCards\","
                        + "\"numToPutBack\":0,"
                        + "\"keptCards\":\"\","
                        + "\"bottomedCards\":\"\","
                        + "\"rankedSize\":" + hand.size()
                );
                return true;
            }

            if (hand.size() < numToPutBack) {
                mulliganTraceJsonl(
                        "fallback",
                        "\"method\":\"chooseLondonMulliganCards\","
                        + "\"reason\":\"hand_too_small\","
                        + "\"handSize\":" + hand.size() + ","
                        + "\"numToPutBack\":" + numToPutBack
                );
                // RL-only fallback (no engine): bottom as many as possible from current hand.
                int k = Math.min(hand.size(), Math.max(0, numToPutBack));
                for (int i = hand.size() - k; i < hand.size(); i++) {
                    target.addTarget(hand.get(i).getId(), null, game);
                }
                return true;
            }

            // Rank all cards; genericChoose will record a LONDON_MULLIGAN TrainingData immediately (if trainingEnabled).
            List<Integer> rankedIndices = genericChoose(
                    hand,
                    hand.size(),
                    hand.size(),
                    StateSequenceBuilder.ActionType.LONDON_MULLIGAN,
                    game,
                    null
            );

            // Bottom the worst N cards (take the last N from ranked list)
            for (int i = hand.size() - numToPutBack; i < hand.size(); i++) {
                int cardIndex = rankedIndices.get(i);
                target.addTarget(hand.get(cardIndex).getId(), null, game);
            }

            // Build detailed card list with probabilities
            StringBuilder keptCards = new StringBuilder();
            StringBuilder bottomedCards = new StringBuilder();
            StringBuilder cardProbs = new StringBuilder();
            cardProbs.append("[");
            
            for (int i = 0; i < hand.size(); i++) {
                Card card = hand.get(rankedIndices.get(i));
                float prob = (lastActionProbs != null && i < lastActionProbs.length) ? lastActionProbs[rankedIndices.get(i)] : 0.0f;
                
                // Add to cardProbs list (all cards with probabilities)
                if (i > 0) {
                    cardProbs.append("; ");
                }
                cardProbs.append(String.format("%s (%.3f)", card.getName(), prob));
                
                // Add to kept or bottomed list
                if (i < hand.size() - numToPutBack) {
                    if (keptCards.length() > 0) {
                        keptCards.append("; ");
                    }
                    keptCards.append(card.getName());
                } else {
                    if (bottomedCards.length() > 0) {
                        bottomedCards.append("; ");
                    }
                    bottomedCards.append(card.getName());
                }
            }
            cardProbs.append("]");

            if (MULLIGAN_TRACE) {
                mulliganTrainingLog(String.format(
                        "MULLIGAN_BOTTOM ep=%d player=%s mulligansTaken=%d handSize=%d bottomN=%d rankedSize=%d kept=[%s] bottomed=[%s]",
                        currentEpisode,
                        getName(),
                        mulligansTaken,
                        hand.size(),
                        numToPutBack,
                        rankedIndices != null ? rankedIndices.size() : -1,
                        trunc(keptCards.toString(), 240),
                        trunc(bottomedCards.toString(), 240)
                ));
            }

            mulliganTraceJsonl(
                    "bottom_result",
                    "\"method\":\"chooseLondonMulliganCards\","
                    + "\"numToPutBack\":" + numToPutBack + ","
                    + "\"keptCards\":\"" + jsonEscape(trunc(keptCards.toString(), 240)) + "\","
                    + "\"bottomedCards\":\"" + jsonEscape(trunc(bottomedCards.toString(), 240)) + "\","
                    + "\"rankedSize\":" + (rankedIndices != null ? rankedIndices.size() : -1)
            );

            // Gamelog: london bottoming details
            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    gameLogger.log(String.format(
                            "LONDON_BOTTOM: player=%s mulligansTaken=%d handSize=%d bottomN=%d kept=[%s] bottomed=[%s]",
                            getName(),
                            mulligansTaken,
                            hand.size(),
                            numToPutBack,
                            trunc(keptCards.toString(), 400),
                            trunc(bottomedCards.toString(), 400)
                    ));
                    gameLogger.log(String.format(
                            "LONDON_BOTTOM_PROBS: player=%s cards=%s",
                            getName(),
                            trunc(cardProbs.toString(), 500)
                    ));
                }
            } catch (Exception ignored) {
                // don't fail bottoming for logging
            }

            return true;

        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn("Error in London mulligan card selection, using fallback: " + e.getMessage());
            mulliganTraceJsonl(
                    "exception",
                    "\"method\":\"chooseLondonMulliganCards\","
                    + "\"exception\":\"" + jsonEscape(e.getClass().getSimpleName() + ": " + e.getMessage()) + "\""
            );
            // RL-only fallback (no engine): deterministically bottom from end of hand.
            try {
                List<Card> hand = new ArrayList<>(getHand().getCards(game));
                int numToPutBack = target != null ? target.getMinNumberOfTargets() : 0;
                int k = Math.min(hand.size(), Math.max(0, numToPutBack));
                for (int i = hand.size() - k; i < hand.size(); i++) {
                    target.addTarget(hand.get(i).getId(), null, game);
                }
                return true;
            } catch (Exception ignored) {
                return false;
            }
        }
    }

    /**
     * Get mulligan training data and clear buffer.
     */
    public List<float[]> getMulliganFeatures() {
        return new ArrayList<>(mulliganFeatures);
    }

    public List<Float> getMulliganDecisions() {
        return new ArrayList<>(mulliganDecisions);
    }

    public List<Boolean> getMulliganOverrides() {
        return new ArrayList<>(mulliganOverrides);
    }

    public List<Integer> getMulliganLandCounts() {
        return new ArrayList<>(mulliganLandCounts);
    }

    public void clearMulliganData() {
        mulliganFeatures.clear();
        mulliganDecisions.clear();
        mulliganOverrides.clear();
        mulliganLandCounts.clear();
        // Reset early-land tracking for next game
        rlPlayerHasHadATurn = false;
        lastTrackedGameTurn = 0;
        rlPlayerTurnsTracked = 0;
        earlyLandHits = 0;
    }

    /**
     * Track lands in play at the end of the RL player's first 3 turns.
     * Call from any frequently-invoked decision method during gameplay.
     * Records whether the player was "on curve" (had at least N lands after turn N).
     */
    private void trackEarlyLands(Game game) {
        if (rlPlayerTurnsTracked >= 3 || game == null) {
            return;
        }
        int turn = game.getTurnNum();
        if (turn <= 0 || turn == lastTrackedGameTurn) {
            return;
        }
        lastTrackedGameTurn = turn;

        UUID activePlayer = game.getActivePlayerId();
        if (activePlayer == null) {
            return;
        }

        if (activePlayer.equals(playerId)) {
            // RL player's turn — mark that we've had at least one turn
            rlPlayerHasHadATurn = true;
        } else if (rlPlayerHasHadATurn) {
            // Opponent's turn and RL has had at least one turn → RL's previous turn just ended
            rlPlayerTurnsTracked++;
            int landsInPlay = countLandsInPlay(game);
            // "On curve" = at least N lands after your Nth turn
            if (landsInPlay >= rlPlayerTurnsTracked) {
                earlyLandHits++;
            }
        }
    }

    /**
     * Get early-game land score: fraction of first 3 turns that were "on curve."
     * Returns 0-1 where 1.0 = hit all land drops on time.
     */
    public float getEarlyLandScore() {
        if (rlPlayerTurnsTracked <= 0) {
            return -1.0f; // No data (game ended before tracking)
        }
        return (float) earlyLandHits / (float) Math.min(rlPlayerTurnsTracked, 3);
    }

    /**
     * Count lands this player controls on the battlefield.
     */
    private int countLandsInPlay(Game game) {
        int count = 0;
        for (Permanent p : game.getBattlefield().getAllActivePermanents(playerId)) {
            if (p.isLand(game)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Count lands in current hand for mulligan heuristic.
     */
    private int countLandsInHand(Game game) {
        int landCount = 0;
        for (Card card : getHand().getCards(game)) {
            if (card.isLand(game)) {
                landCount++;
            }
        }
        return landCount;
    }

    /**
     * Set the current episode number for logging purposes.
     */
    public void setCurrentEpisode(int episodeNum) {
        this.currentEpisode = episodeNum;
        this.lastLoggedTurn = -1; // Reset turn tracking for new game
    }

    /**
     * Set the mulligan model episode number for epsilon-greedy exploration.
     */
    public void setMulliganEpisode(int episodeNum) {
        this.mulliganEpisode = episodeNum;
    }

    protected boolean priorityPlay(Game game) {
        game.getState().setPriorityPlayerId(playerId);
        game.firePriorityEvent(playerId);
        Ability ability;
        switch (game.getTurnStepType()) {
            case UPKEEP:
            case DRAW:
                pass(game);
                return false;
            case PRECOMBAT_MAIN:
                if (ACTIVATION_DIAG) {
                    printBattleField(game, "Sim PRIORITY on MAIN 1");
                }
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                if (ACTIVATION_DIAG) {
                    printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                }
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                pass(game);
                return true;
            case DECLARE_BLOCKERS:
                if (ACTIVATION_DIAG) {
                    printBattleField(game, "Sim PRIORITY on DECLARE BLOCKERS");
                }
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                pass(game);
                return true;
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
            case END_COMBAT:
                pass(game);
                return false;
            case POSTCOMBAT_MAIN:
                if (ACTIVATION_DIAG) {
                    printBattleField(game, "Sim PRIORITY on MAIN 2");
                }
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                return true;
            case END_TURN:
            case CLEANUP:
                pass(game);
                return false;
        }
        return false;
    }

    protected void printBattleField(Game game, String info) {
        if (ACTIVATION_DIAG) {
            // Clear the console line
            System.out.print("\033[2K"); // ANSI escape code to clear the current line
            // Move the cursor up one line
            System.out.print("\033[1A");

            // Print the battlefield information
            System.out.println("=================== " + info + ", turn " + game.getTurnNum() + ", " + game.getPlayer(game.getPriorityPlayerId()).getName() + " ===================");
            System.out.println("[Stack]: " + game.getStack());
            printBattleField(game, playerId);
            for (UUID opponentId : game.getOpponents(playerId)) {
                printBattleField(game, opponentId);
            }
        }
    }

    protected void printBattleField(Game game, UUID playerId) {
        Player player = game.getPlayer(playerId);
        System.out.println(new StringBuilder("[").append(game.getPlayer(playerId).getName()).append("]")
                .append(", life = ").append(player.getLife())
                .toString());
        String cardsInfo = player.getHand().getCards(game).stream()
                .map(card -> card.getName()) // Removed card score
                .collect(Collectors.joining("; "));
        StringBuilder sb = new StringBuilder("-> Hand: [")
                .append(cardsInfo)
                .append("]");
        System.out.println(sb.toString());

        // battlefield
        sb.setLength(0);
        String ownPermanentsInfo = game.getBattlefield().getAllPermanents().stream()
                .filter(p -> p.isOwnedBy(player.getId()))
                .map(p -> p.getName()
                + (p.isTapped() ? ",tapped" : "")
                + (p.isAttacking() ? ",attacking" : "")
                + (p.getBlocking() > 0 ? ",blocking" : ""))
                .collect(Collectors.joining("; "));
        sb.append("-> Permanents: [").append(ownPermanentsInfo).append("]");
        System.out.println(sb.toString());
    }

    /**
     * Format game state as a string for logging.
     */
    protected String formatGameState(Game game) {
        StringBuilder sb = new StringBuilder();

        // Stack
        int stackSize = game.getStack().size();
        sb.append("[Stack]: ").append(stackSize).append(" items\n");
        if (stackSize > 0) {
            int idx = 0;
            for (StackObject so : game.getStack()) {
                String controller = "";
                try {
                    if (so != null && so.getControllerId() != null) {
                        Player cp = game.getPlayer(so.getControllerId());
                        controller = (cp != null && cp.getName() != null) ? cp.getName() : "";
                    }
                } catch (Exception ignored) {
                    controller = "";
                }
                String name = "";
                try {
                    name = (so != null && so.getName() != null) ? so.getName() : "";
                } catch (Exception ignored) {
                    name = "";
                }
                sb.append("  - [").append(idx).append("] ").append(name);
                if (!controller.isEmpty()) {
                    sb.append(" (controller=").append(controller).append(")");
                }
                sb.append("\n");
                idx++;
            }
        }

        // Players
        for (UUID pid : game.getState().getPlayersInRange(this.playerId, game)) {
            Player player = game.getPlayer(pid);
            if (player == null) {
                continue;
            }

            sb.append("[").append(player.getName()).append("], life = ").append(player.getLife()).append("\n");

            // Hand (only show for RL player)
            if (player.getId().equals(this.playerId)) {
                String cardsInfo = player.getHand().getCards(game).stream()
                        .map(card -> card.getName())
                        .collect(Collectors.joining("; "));
                sb.append("-> Hand: [").append(cardsInfo).append("]\n");
            } else {
                sb.append("-> Hand: ").append(player.getHand().size()).append(" cards\n");
            }

            // Battlefield
            String permanentsInfo = game.getBattlefield().getAllPermanents().stream()
                    .filter(p -> p.isOwnedBy(player.getId()))
                    .map(p -> p.getName()
                    + (p.isTapped() ? ",tapped" : "")
                    + (p.isAttacking() ? ",attacking" : "")
                    + (p.getBlocking() > 0 ? ",blocking" : ""))
                    .collect(Collectors.joining("; "));
            sb.append("-> Permanents: [").append(permanentsInfo).append("]\n");

            // Graveyard
            try {
                List<Card> gy = new ArrayList<>(player.getGraveyard().getCards(game));
                int total = gy.size();
                String gyInfo = gy.stream()
                        .limit(60)
                        .map(Card::getName)
                        .collect(Collectors.joining("; "));
                if (total > 60) {
                    gyInfo = gyInfo + " ... (+" + (total - 60) + " more)";
                }
                sb.append("-> Graveyard: [").append(gyInfo).append("]\n");
            } catch (Exception ignored) {
                sb.append("-> Graveyard: []\n");
            }

            // Exile
            try {
                List<Card> ex = new ArrayList<>(game.getExile().getCardsOwned(game, player.getId()));
                int total = ex.size();
                String exInfo = ex.stream()
                        .limit(60)
                        .map(Card::getName)
                        .collect(Collectors.joining("; "));
                if (total > 60) {
                    exInfo = exInfo + " ... (+" + (total - 60) + " more)";
                }
                sb.append("-> Exile: [").append(exInfo).append("]\n");
            } catch (Exception ignored) {
                sb.append("-> Exile: []\n");
            }
        }

        return sb.toString();
    }

    // I'm changing the design here to not use an actions queue.
    // Instead, I'm passing the ability to the act method.
    // We don't calculate lists of actions, but instead just one action at a time.
    // NOTE: I think the way computerplayer6 does this is because it implements the idea
    // of holding priority
    protected void act(Game game, ActivatedAbility ability) {
        if (ability == null) {
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info("Model opted to pass priority");
            }
            pass(game);
        } else {
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(String.format("===> SELECTED ACTION for %s: %s", getName(), ability));
            }

            // Double-check canActivate status right before activation
            if (ACTIVATION_DIAG) {
                ActivatedAbility.ActivationStatus preActivateStatus = ability.canActivate(this.getId(), game);
                MageObject sourceObj = game.getObject(ability.getSourceId());
                String sourceName = sourceObj != null ? sourceObj.getName() : "unknown";
                RLTrainer.threadLocalLogger.get().info(String.format(
                        "PRE-ACTIVATE: canActivate=%s, approvingObjects=%d, card=%s",
                        preActivateStatus != null && preActivateStatus.canActivate(),
                        preActivateStatus != null ? preActivateStatus.getApprovingObjects().size() : 0,
                        sourceName
                ));
            }

            // CRITICAL FIX: getPlayable() returns abilities from a COPIED game (createSimulationForPlayableCalc)
            // We need to get a fresh ability from the REAL game's object to avoid stale state issues
            ActivatedAbility freshAbility = ability;
            mage.game.permanent.Permanent sourcePerm = game.getPermanent(ability.getSourceId());
            if (sourcePerm != null) {
                // Find the matching ability on the real permanent
                for (Ability permAbility : sourcePerm.getAbilities(game)) {
                    if (permAbility instanceof ActivatedAbility
                            && permAbility.getRule().equals(ability.getRule())) {
                        freshAbility = (ActivatedAbility) permAbility;
                        if (ACTIVATION_DIAG) {
                            RLTrainer.threadLocalLogger.get().info(
                                    "FRESH-ABILITY: Got fresh ability from real game's permanent");
                        }
                        break;
                    }
                }
            } else {
                // Check if it's a card (spell from hand/graveyard/etc)
                Card sourceCard = game.getCard(ability.getSourceId());
                if (sourceCard != null) {
                    for (Ability cardAbility : sourceCard.getAbilities(game)) {
                        if (cardAbility instanceof ActivatedAbility
                                && cardAbility.getRule().equals(ability.getRule())) {
                            freshAbility = (ActivatedAbility) cardAbility;
                            if (ACTIVATION_DIAG) {
                                RLTrainer.threadLocalLogger.get().info(
                                        "FRESH-ABILITY: Got fresh ability from real game's card");
                            }
                            break;
                        }
                    }
                }
            }

            // Track current ability for choice filtering  
            currentAbility = freshAbility;

            // Log BEFORE attempting activation to capture pre-activation state
            if (ACTIVATION_DIAG) {
                MageObject sourceObj = game.getObject(freshAbility.getSourceId());
                String sourceName = sourceObj != null ? sourceObj.getName() : "unknown";
                boolean sourceIsTapped = sourceObj instanceof mage.game.permanent.Permanent
                        && ((mage.game.permanent.Permanent) sourceObj).isTapped();

                RLTrainer.threadLocalLogger.get().info(String.format(
                        "ATTEMPTING ACTIVATION: ability=%s, source=%s, sourceTapped=%s, sourceId=%s, permanentFound=%s, usingFresh=%s",
                        freshAbility.getRule(),
                        sourceName,
                        sourceIsTapped,
                        freshAbility.getSourceId(),
                        (sourcePerm != null),
                        (freshAbility != ability)
                ));
            }

            // CRITICAL FIX: If ability has tap costs, exclude those permanents from mana producers
            // This prevents the AI from tapping them for mana when it needs to tap them for the ability
            tapTargetCostReservations.clear();
            for (mage.abilities.costs.Cost cost : freshAbility.getCosts()) {
                if (cost instanceof mage.abilities.costs.common.TapSourceCost) {
                    abilitySourceToExcludeFromMana = freshAbility.getSourceId();
                    if (ACTIVATION_DIAG) {
                        RLTrainer.threadLocalLogger.get().info(
                                "TAP-SOURCE-COST: Will exclude source " + freshAbility.getSourceId() + " from mana producers");
                    }
                } else if (cost instanceof mage.abilities.costs.common.TapTargetCost) {
                    // Reserve permanents for TapTargetCost (e.g., "Tap an untapped Gate you control")
                    mage.abilities.costs.common.TapTargetCost tapTargetCost = (mage.abilities.costs.common.TapTargetCost) cost;
                    mage.target.common.TargetControlledPermanent target = tapTargetCost.getTarget();
                    int numNeeded = target.getMinNumberOfTargets();

                    // Find permanents that can satisfy this TapTargetCost
                    List<mage.game.permanent.Permanent> candidates = new ArrayList<>();
                    for (mage.game.permanent.Permanent perm : game.getBattlefield().getAllActivePermanents(this.getId())) {
                        if (!perm.isTapped() && target.canTarget(this.getId(), perm.getId(), freshAbility, game)) {
                            candidates.add(perm);
                        }
                    }

                    // Reserve the first N candidates (prioritize non-mana-producers if possible, but for simplicity just take first N)
                    int reserved = 0;
                    for (mage.game.permanent.Permanent perm : candidates) {
                        if (reserved >= numNeeded) {
                            break;
                        }
                        // Don't reserve the source (already handled by TapSourceCost)
                        if (!perm.getId().equals(freshAbility.getSourceId())) {
                            tapTargetCostReservations.add(perm.getId());
                            reserved++;
                            if (ACTIVATION_DIAG) {
                                RLTrainer.threadLocalLogger.get().info(
                                        "TAP-TARGET-COST: Reserved " + perm.getName() + " (" + perm.getId() + ") for TapTargetCost");
                            }
                        }
                    }
                }
            }

            // Activate using the fresh ability from the real game
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(
                        "CALLING: super.activateAbility() for " + freshAbility.getClass().getSimpleName());
            }

            boolean activationResult;
            Exception activationException = null;
            try {
                activationResult = super.activateAbility(freshAbility, game);
            } catch (Exception e) {
                activationResult = false;
                activationException = e;
                RLTrainer.threadLocalLogger.get().error(
                        "ACTIVATION THREW EXCEPTION: " + e.getClass().getName() + ": " + e.getMessage());
                e.printStackTrace();
            }

            if (ACTIVATION_DIAG && !activationResult) {
                // Try to understand WHY it failed - check source state AFTER failure
                mage.game.permanent.Permanent sourcePermAfter = game.getPermanent(freshAbility.getSourceId());
                RLTrainer.threadLocalLogger.get().error(
                        "ACTIVATION FAILED INTERNALLY - checking state after failure:");
                RLTrainer.threadLocalLogger.get().error(
                        "  Source permanent found: " + (sourcePermAfter != null));
                if (sourcePermAfter != null) {
                    RLTrainer.threadLocalLogger.get().error(
                            "  Source permanent tapped (AFTER): " + sourcePermAfter.isTapped());
                }
                RLTrainer.threadLocalLogger.get().error(
                        "  FreshAbility targets: " + (!freshAbility.getTargets().isEmpty() ? freshAbility.getTargets().get(0).getTargets().size() : 0));
                RLTrainer.threadLocalLogger.get().error(
                        "  FreshAbility costs paid: " + freshAbility.getManaCostsToPay());

                // Re-check canActivate after failure
                mage.abilities.ActivatedAbility.ActivationStatus postFailStatus
                        = freshAbility.canActivate(this.getId(), game);
                RLTrainer.threadLocalLogger.get().error(
                        "  POST-FAIL canActivate: " + (postFailStatus != null && postFailStatus.canActivate()));
                RLTrainer.threadLocalLogger.get().error(
                        "  Used fresh ability from permanent: " + (freshAbility != ability));

                if (activationException != null) {
                    RLTrainer.threadLocalLogger.get().error(
                            "  Exception was thrown: " + activationException);
                }
            }

            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(String.format(
                        "ACTIVATION RESULT: %s", activationResult
                ));
            }

            if (!activationResult) {
                // If we are here it is because the RL player chose an action that isn't actually executable
                // (often due to hidden costs/choices not captured in our candidate features).
                // Treat this as a forced pass rather than crashing the engine.
                int failureCount = RL_ACTIVATION_FAILURES.incrementAndGet();

                // CRITICAL ERROR - Use both logger AND System.err to ensure visibility
                String errorMsg = "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        + "!!! RL PLAYER ACTIVATION FAILED - THIS HURTS TRAINING !!!\n"
                        + "!!! TOTAL RL ACTIVATION FAILURES THIS RUN: " + failureCount + " !!!\n"
                        + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        + "RL Player: " + getName() + " | Thread: " + Thread.currentThread().getName() + "\n"
                        + "Failed ability: " + ability + "\n";

                System.err.println(errorMsg);
                RLTrainer.threadLocalLogger.get().error(errorMsg);
                logActivationFailure(ability, game);
                System.err.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                RLTrainer.threadLocalLogger.get().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

                // Pause game on activation failure for debugging (controlled by env var)
                pauseOnActivationFailure();

                pass(game);
                currentAbility = null; // Clear after failure
                abilitySourceToExcludeFromMana = null; // Clear mana exclusions
                tapTargetCostReservations.clear();
                return;
            }

            currentAbility = null; // Clear after successful activation
            abilitySourceToExcludeFromMana = null; // Clear mana exclusions
            tapTargetCostReservations.clear();

            // Log all resolved targets for the ability (covers auto-chosen single targets)
            logAbilityTargets(ability, game);

            //TODO: Implement holding priority for abilities that don't use the stack
            if (ability.isUsesStack()) {
                pass(game);
            }
        }
    }

    /**
     * Logs every target that has been recorded on the provided ability. This is
     * called *after* the ability has been activated so it catches cases where
     * the engine auto-selected a single possible target (which bypasses the
     * usual choose/chooseTarget callbacks).
     */
    private void logAbilityTargets(Ability ability, Game game) {
        if (ability == null) {
            return;
        }

        boolean logged = false;

        for (Target tgt : ability.getTargets()) {
            if (!tgt.getTargets().isEmpty()) {
                for (UUID targetId : tgt.getTargets()) {
                    String targetName = describeTargetWithOwner(targetId, game);
                    RLTrainer.threadLocalLogger.get().info(
                            "Player " + getName() + " resolved target: " + targetName + " (" + targetId + ")"
                            + " for ability: " + ability.toString()
                    );
                    logged = true;
                }
            }
        }

        // Fallback: look at the newest stack object (in case targets were set on the stack copy)
        if (!logged && ability.isUsesStack() && !game.getStack().isEmpty()) {
            mage.game.stack.StackObject top = game.getStack().getFirst();
            if (top != null && top.getSourceId().equals(ability.getSourceId())) {
                for (Target tgt : top.getStackAbility().getTargets()) {
                    if (!tgt.getTargets().isEmpty()) {
                        for (UUID targetId : tgt.getTargets()) {
                            String targetName = describeTargetWithOwner(targetId, game);
                            RLTrainer.threadLocalLogger.get().info(
                                    "Player " + getName() + " resolved target (stack): " + targetName + " (" + targetId + ")"
                                    + " for ability: " + ability.toString()
                            );
                            logged = true;
                        }
                    }
                }
            }
        }
    }

    private void logActivationFailure(ActivatedAbility ability, Game game) {
        if (!ACTIVATION_DIAG || ability == null || game == null) {
            return;
        }
        try {
            ActivatedAbility.ActivationStatus status = ability.canActivate(this.getId(), game);
            boolean canChooseTarget = ability.canChooseTarget(game, this.getId());
            boolean canPlayLandNow = !(ability instanceof PlayLandAbility) || this.canPlayLand();
            MageObject sourceObj = game.getObject(ability.getSourceId());
            String sourceName = sourceObj != null ? sourceObj.getName() : "unknown";
            String sourceZone = sourceObj != null ? String.valueOf(game.getState().getZone(sourceObj.getId())) : "unknown";
            int approvingCount = status != null ? status.getApprovingObjects().size() : 0;

            RLTrainer.threadLocalLogger.get().error(String.format(
                    "RL ACTIVATION DIAGNOSTICS: player=%s abilityType=%s usesStack=%s canActivate=%s canChooseTarget=%s canPlayLand=%s approvingObjects=%d source=%s zone=%s",
                    getName(),
                    ability.getAbilityType(),
                    ability.isUsesStack(),
                    status != null && status.canActivate(),
                    canChooseTarget,
                    canPlayLandNow,
                    approvingCount,
                    sourceName,
                    sourceZone
            ));

            // Log detailed mana and cost information
            RLTrainer.threadLocalLogger.get().error("=== MANA & COST DIAGNOSTICS ===");
            RLTrainer.threadLocalLogger.get().error(String.format("Available mana: %s", getManaAvailable(game)));
            RLTrainer.threadLocalLogger.get().error(String.format("Ability costs: %s", ability.getManaCostsToPay()));

            // Log all untapped lands
            StringBuilder untappedLands = new StringBuilder("Untapped lands: ");
            game.getBattlefield().getAllActivePermanents(this.getId()).stream()
                    .filter(p -> p.isLand(game) && !p.isTapped())
                    .forEach(p -> untappedLands.append(p.getName()).append(", "));
            RLTrainer.threadLocalLogger.get().error(untappedLands.toString());

            // Check if source permanent is tapped (for abilities with tap cost)
            if (sourceObj != null && sourceObj instanceof mage.game.permanent.Permanent) {
                mage.game.permanent.Permanent sourcePerm = (mage.game.permanent.Permanent) sourceObj;
                RLTrainer.threadLocalLogger.get().error(String.format("Source permanent '%s' tapped status: %s",
                        sourceName, sourcePerm.isTapped()));

                // Check if ability has tap cost
                boolean hasTapCost = ability.getCosts().stream()
                        .anyMatch(cost -> cost instanceof mage.abilities.costs.common.TapSourceCost);
                RLTrainer.threadLocalLogger.get().error(String.format("Ability has tap cost: %s", hasTapCost));
            }

            RLTrainer.threadLocalLogger.get().error("=== END MANA & COST DIAGNOSTICS ===");
        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().error("RL ACTIVATION DIAGNOSTICS: error while gathering details", e);
        }
    }

    /**
     * Test all alternative cost options for an ability in simulation. Returns a
     * set of choice keys (e.g., "1", "2") that successfully activated.
     *
     * For abilities without alternative costs, tests once and returns either
     * empty set (failed) or set with "0" (succeeded).
     */
    private Set<String> testAllAlternativeCosts(ActivatedAbility ability, Game game) {
        Set<String> validChoices = new HashSet<>();

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "ALTCOST-TEST: Testing ability " + ability + " (sourceId=" + ability.getSourceId() + ")"
            );
        }

        // Test with default choice first (no forcing)
        forcedAlternativeChoice.remove();
        choiceTrackingData.remove();

        // IMPORTANT: simulation activation may trigger target/choice callbacks.
        // Do not pollute per-game gamelogs with simulation-internal prompts.
        GameLogger prevGameLogger = RLTrainer.threadLocalGameLogger.get();
        RLTrainer.threadLocalGameLogger.set(GameLogger.create(false));
        Game sim = null;
        boolean defaultWorks = false;
        try {
            sim = game.createSimulationForAI();
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(
                        "ALTCOST-TEST: Created simulation, about to activate..."
                );
            }
            defaultWorks = sim.getPlayer(this.getId()).activateAbility((ActivatedAbility) ability.copy(), sim);
        } finally {
            RLTrainer.threadLocalGameLogger.set(prevGameLogger);
        }

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "ALTCOST-TEST: activateAbility() returned " + defaultWorks + ", now checking tracking..."
            );
        }

        ChoiceTrackingData tracking = choiceTrackingData.get();

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "ALTCOST-TEST: Default test " + (defaultWorks ? "PASSED" : "FAILED")
                    + ", tracking=" + (tracking != null ? "present" : "null")
                    + (tracking != null ? ", choiceMade=" + tracking.choiceMade + ", availableOptions=" + tracking.availableOptions : "")
            );
        }

        if (defaultWorks) {
            if (tracking != null && tracking.choiceMade != null) {
                // An alternative cost choice was made
                validChoices.add(tracking.choiceMade);
            } else {
                // No alternative cost choice (normal ability)
                validChoices.add("0");
                choiceTrackingData.remove();
                return validChoices; // No alternatives to test
            }
        }

        // If we detected alternatives, test all other options
        if (tracking != null && tracking.availableOptions != null) {
            for (String option : tracking.availableOptions) {
                if (validChoices.contains(option)) {
                    continue; // Already tested this one
                }

                if (ACTIVATION_DIAG) {
                    RLTrainer.threadLocalLogger.get().info(
                            "ALTCOST-TEST: Testing option " + option + "..."
                    );
                }

                // Force this specific choice and test
                forcedAlternativeChoice.set(option);
                choiceTrackingData.remove();

                // Disable gamelog during simulation (see above).
                GameLogger prev2 = RLTrainer.threadLocalGameLogger.get();
                RLTrainer.threadLocalGameLogger.set(GameLogger.create(false));
                boolean works = false;
                try {
                    Game sim2 = game.createSimulationForAI();
                    works = sim2.getPlayer(this.getId()).activateAbility((ActivatedAbility) ability.copy(), sim2);
                } finally {
                    RLTrainer.threadLocalGameLogger.set(prev2);
                }
                if (works) {
                    validChoices.add(option);
                    if (ACTIVATION_DIAG) {
                        RLTrainer.threadLocalLogger.get().info(
                                "ALTCOST-TEST: Option " + option + " PASSED"
                        );
                    }
                } else {
                    if (ACTIVATION_DIAG) {
                        RLTrainer.threadLocalLogger.get().info(
                                "ALTCOST-TEST: Option " + option + " FAILED"
                        );
                    }
                }
            }
        }

        // Clean up ThreadLocals
        forcedAlternativeChoice.remove();
        choiceTrackingData.remove();

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "ALTCOST-TEST: Final result for " + ability + " -> validChoices=" + validChoices
            );
        }

        return validChoices;
    }

    protected Ability calculateRLAction(Game game) {
        List<ActivatedAbility> flattenedOptions = getPlayable(game, true);
        // (PassAbility will be added later after duplicate removal)

        // Filter out mana abilities
        flattenedOptions = flattenedOptions.stream()
                .filter(ability -> !(ability instanceof ManaAbility))
                .collect(java.util.stream.Collectors.toList());

        // Filter by testing activation in a simulation (like ComputerPlayer6 does)
        // For abilities with alternative costs, test ALL alternatives and track which ones work
        List<ActivatedAbility> validOptions = new ArrayList<>();
        int filteredCount = 0;
        validAlternativeCosts.clear(); // Clear previous tracking

        for (ActivatedAbility ability : flattenedOptions) {
            Set<String> validChoices = testAllAlternativeCosts(ability, game);

            if (!validChoices.isEmpty()) {
                // At least one alternative worked
                validOptions.add(ability);
                // Always track if there were alternatives tested (even if only one is valid)
                // We detect alternatives by checking if the set contains anything other than just "0"
                boolean hasAlternatives = !(validChoices.size() == 1 && validChoices.contains("0"));
                if (hasAlternatives) {
                    validAlternativeCosts.put(ability.getSourceId(), validChoices);
                    if (ACTIVATION_DIAG) {
                        RLTrainer.threadLocalLogger.get().info(
                                "SIM-FILTER: Ability has alternatives, storing validated choices: "
                                + ability + " (sourceId=" + ability.getSourceId() + ") -> " + validChoices
                        );
                    }
                }
            } else {
                filteredCount++;
                if (ACTIVATION_DIAG) {
                    RLTrainer.threadLocalLogger.get().info("SIM-FILTER: Rejected ability in simulation: " + ability);
                }
            }
        }
        if (ACTIVATION_DIAG && filteredCount > 0) {
            RLTrainer.threadLocalLogger.get().info("SIM-FILTER: Filtered out " + filteredCount + " abilities that failed simulation test");
        }
        flattenedOptions = validOptions;

        // Remove duplicate spell abilities with the same name
        List<ActivatedAbility> uniqueOptions = new ArrayList<>();
        Set<String> seenNames = new HashSet<>();

        // Remove duplicate spell abilities with the same name
        // TODO: Investigate if this is what we want. I did this because despite "setting targets" during selection. we still get prompted for choices later anyway
        for (ActivatedAbility ability : flattenedOptions) {
            String name = ability.toString();
            if (!seenNames.contains(name)) {
                seenNames.add(name);
                uniqueOptions.add(ability);
            }
        }
        flattenedOptions = uniqueOptions;

        // Finally, ensure PassAbility maps to index 0
        flattenedOptions.add(0, new PassAbility());

        // If Pass is the only option, genericChoose() will short-circuit and we won't record lastActionProbs,
        // which makes the gamelog look like turns are skipped. Log pass-only priority steps explicitly.
        if (flattenedOptions.size() == 1) {
            GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
            if (gameLogger.isEnabled()) {
                String phase = (game.getStep() != null && game.getStep().getType() != null)
                        ? game.getStep().getType().toString()
                        : "Unknown";
                int turn = game.getTurnNum();
                if (turn != lastLoggedTurn) {
                    String activePlayerName = game.getActivePlayerId() != null
                            ? game.getPlayer(game.getActivePlayerId()).getName()
                            : "Unknown";
                    gameLogger.logTurnStart(turn, activePlayerName, formatGameState(game));
                    lastLoggedTurn = turn;
                }
                List<String> optionNames = java.util.Collections.singletonList("Pass");
                float[] probs = new float[]{1.0f};
                this.lastActionProbs = probs;
                this.lastValueScore = 0.0f;
                String activePlayerName = game.getActivePlayerId() != null
                        ? game.getPlayer(game.getActivePlayerId()).getName()
                        : "Unknown";
                gameLogger.logDecision(
                        this.getName(),
                        activePlayerName,
                        phase,
                        turn,
                        formatGameState(game),
                        optionNames,
                        probs,
                        0.0f,
                        0,
                        "Pass"
                );
            }
            return flattenedOptions.get(0);
        }

        // Get model's choice of actions
        List<Integer> targetsToSet = genericChoose(flattenedOptions, 1, 1, StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL, game, null);

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info("Playable options: " + flattenedOptions);
        }

        // Log to detailed game logger if enabled
        GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
        if (gameLogger.isEnabled() && lastActionProbs != null) {
            // Get phase info and turn
            String phase = game.getStep().getType().toString();
            int turn = game.getTurnNum();

            // Log turn start if turn changed (shows full game state including opponent actions)
            if (turn != lastLoggedTurn) {
                String activePlayerName = game.getActivePlayerId() != null
                        ? game.getPlayer(game.getActivePlayerId()).getName()
                        : "Unknown";
                gameLogger.logTurnStart(turn, activePlayerName, formatGameState(game));
                lastLoggedTurn = turn;
            }

            // Convert options to string list
            List<String> optionNames = flattenedOptions.stream()
                    .map(ability -> ability.toString())
                    .collect(Collectors.toList());

            // Get selected action
            int selectedIndex = targetsToSet.get(0);
            String selectedAction = flattenedOptions.get(selectedIndex).toString();

            // Get game state
            String gameState = formatGameState(game);

            // Log decision
            gameLogger.logDecision(
                    this.getName(),
                    game.getActivePlayerId() != null ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown",
                    phase,
                    turn,
                    gameState,
                    optionNames,
                    lastActionProbs,
                    lastValueScore,
                    selectedIndex,
                    selectedAction
            );
        }

        // Return the ability corresponding to the best index
        return flattenedOptions.get(targetsToSet.get(0));
    }

    public List<StateSequenceBuilder.TrainingData> getTrainingBuffer() {
        if (!trainingEnabled) {
            return new ArrayList<>();
        }
        return new ArrayList<>(trainingBuffer);
    }

    public java.util.Map<StateSequenceBuilder.ActionType, Integer> getDecisionCountsByHead() {
        return new java.util.HashMap<>(decisionCountsByHead);
    }

    public PythonModel getModel() {
        return model;
    }

    public String getPolicyKey() {
        return policyKey;
    }

    public boolean isTrainingEnabled() {
        return trainingEnabled;
    }

    // Track permanents to exclude from mana producers when ability has tap costs
    private UUID abilitySourceToExcludeFromMana = null;
    private Set<UUID> tapTargetCostReservations = new HashSet<>(); // Permanents reserved for TapTargetCost

    // Store last decision info for game logging
    private float[] lastActionProbs = null;
    private float lastValueScore = 0.0f;

    /**
     * Get the most recent value head prediction. Used for tracking value head
     * quality metrics.
     */
    public float getLastValueScore() {
        return lastValueScore;
    }

    // Trace mana payment for debugging activation failures
    @Override
    public boolean playMana(Ability ability, mage.abilities.costs.mana.ManaCost unpaid, String promptText, Game game) {
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "PLAYMANA: Called for unpaid=" + unpaid.getText()
                    + ", ability=" + (ability != null ? ability.getRule() : "null")
                    + ", excludingSource=" + abilitySourceToExcludeFromMana);
        }
        boolean result = super.playMana(ability, unpaid, promptText, game);
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "PLAYMANA: Result=" + result + ", unpaid remaining=" + unpaid.getText());
        }
        return result;
    }

    // CRITICAL FIX: Exclude permanents from mana producers when they're needed for tap costs
    // This prevents the AI from tapping Basilisk Gate for mana when it needs to tap it for the ability,
    // and prevents tapping Gates needed for TapTargetCost (like Heap Gate's "tap another Gate")
    @Override
    public List<MageObject> getAvailableManaProducers(Game game) {
        List<MageObject> producers = super.getAvailableManaProducers(game);

        // Remove the source permanent if we're paying for an ability that needs to tap it
        if (abilitySourceToExcludeFromMana != null) {
            producers.removeIf(obj -> obj.getId().equals(abilitySourceToExcludeFromMana));
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(
                        "MANA-EXCLUDE: Excluded source " + abilitySourceToExcludeFromMana + " from mana producers");
            }
        }

        // Remove permanents reserved for TapTargetCost
        if (!tapTargetCostReservations.isEmpty()) {
            producers.removeIf(obj -> tapTargetCostReservations.contains(obj.getId()));
            if (ACTIVATION_DIAG) {
                RLTrainer.threadLocalLogger.get().info(
                        "MANA-EXCLUDE: Excluded " + tapTargetCostReservations.size() + " permanents reserved for TapTargetCost");
            }
        }

        return producers;
    }

    // Disables engine auto-targeting heuristics by forcing strict choose mode
    @Override
    public boolean getStrictChooseMode() {
        return !USE_ENGINE_CHOICES;
    }

    /**
     * Get total number of RL activation failures since process start. These
     * failures pollute the training signal and should be zero ideally.
     */
    public static int getRLActivationFailureCount() {
        return RL_ACTIVATION_FAILURES.get();
    }

    public static int getSimulationTrainingSkippedCount() {
        return SIMULATION_TRAINING_SKIPPED.get();
    }

    /**
     * Reset the RL activation failure counter (useful for tracking
     * per-training-run).
     */
    public static void resetRLActivationFailureCount() {
        RL_ACTIVATION_FAILURES.set(0);
    }

    /**
     * Pause game execution when activation failure occurs (for debugging).
     * Controlled by PAUSE_ON_ACTIVATION_FAILURE env var (seconds to pause,
     * 0=disabled).
     */
    private void pauseOnActivationFailure() {
        String pauseEnv = System.getenv("PAUSE_ON_ACTIVATION_FAILURE");
        if (pauseEnv == null || pauseEnv.isEmpty()) {
            return; // No pause by default
        }

        try {
            int pauseSeconds = Integer.parseInt(pauseEnv);
            if (pauseSeconds > 0) {
                System.err.println("\n=== PAUSING FOR " + pauseSeconds + " SECONDS FOR INSPECTION ===");
                RLTrainer.threadLocalLogger.get().error("PAUSING FOR " + pauseSeconds + " SECONDS FOR INSPECTION");
                Thread.sleep(pauseSeconds * 1000L);
                System.err.println("=== RESUMING AFTER PAUSE ===\n");
                RLTrainer.threadLocalLogger.get().error("RESUMING AFTER PAUSE");
            }
        } catch (NumberFormatException e) {
            RLTrainer.threadLocalLogger.get().warn("Invalid PAUSE_ON_ACTIVATION_FAILURE value: " + pauseEnv);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            RLTrainer.threadLocalLogger.get().warn("Pause interrupted");
        }
    }

    private String describeTargetWithOwner(UUID targetId, Game game) {
        // Players: return their name (self/opponent distinction already obvious)
        Player player = game.getPlayer(targetId);
        if (player != null) {
            return player.getName();
        }

        MageObject obj = game.getObject(targetId);
        if (obj == null) {
            return "unknown target";
        }

        // Default base name is the card/permanent name
        StringBuilder sb = new StringBuilder(obj.getName());

        // If it's a permanent on the battlefield we can check controller
        if (obj instanceof Permanent) {
            UUID ctrl = ((Permanent) obj).getControllerId();
            if (ctrl != null) {
                if (ctrl.equals(this.getId())) {
                    sb.append(" (you)");
                } else {
                    Player ownerP = game.getPlayer(ctrl);
                    if (ownerP != null) {
                        sb.append(" (").append(ownerP.getName()).append(")");
                    }
                }
            }
        }
        return sb.toString();
    }
}

// Helper class to store block options
class BlockOption {

    int attackerIndex;
    int blockerIndex;
    float qValue;

    BlockOption(int attackerIndex, int blockerIndex, float qValue) {
        this.attackerIndex = attackerIndex;
        this.blockerIndex = blockerIndex;
        this.qValue = qValue;
    }
}

/**
 * Helper class to track choice information during simulation testing
 */
class ChoiceTrackingData {

    String choiceMade = null;
    Set<String> availableOptions = null;
}

// Helper class to store attack options
class AttackOption {

    int attackTargetIndex;
    int attackerIndex;
    float qValue;

    AttackOption(int attackTargetIndex, int attackerIndex, float qValue) {
        this.attackTargetIndex = attackTargetIndex;
        this.attackerIndex = attackerIndex;
        this.qValue = qValue;
    }
}

// Helper class to store Q-value with its index
class QValueWithIndex {

    float qValue;
    int index;

    QValueWithIndex(float qValue, int index) {
        this.qValue = qValue;
        this.index = index;
    }
}
