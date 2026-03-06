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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import mage.Mana;
import mage.ConditionalMana;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.costs.mana.VariableManaCost;
import mage.abilities.mana.ManaOptions;
import mage.abilities.ActivatedAbility;
import mage.abilities.mana.ManaAbility;
import mage.abilities.PlayLandAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.keyword.DeathtouchAbility;
import mage.abilities.keyword.DoubleStrikeAbility;
import mage.abilities.keyword.FirstStrikeAbility;
import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HasteAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.IndestructibleAbility;
import mage.abilities.keyword.LifelinkAbility;
import mage.abilities.keyword.ReachAbility;
import mage.abilities.keyword.TrampleAbility;
import mage.abilities.keyword.VigilanceAbility;
import mage.counters.CounterType;
import mage.game.permanent.PermanentToken;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.filter.common.FilterLandCard;
import mage.constants.RangeOfInfluence;
import mage.constants.TurnPhase;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.GameState;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.cards.decks.Deck;
import mage.game.draft.Draft;
import mage.player.ai.rl.DraftLogger;
import mage.player.ai.rl.DraftPickRecord;
import mage.player.ai.rl.DraftPythonMLBridge;
import mage.player.ai.rl.DraftStateBuilder;
import mage.player.ai.rl.EnvConfig;
import mage.player.ai.rl.GameLogger;
import mage.player.ai.rl.MetricsCollector;
import mage.player.ai.rl.MulliganLogger;
import mage.player.ai.rl.MulliganModel;
import mage.player.ai.rl.PythonModel;
import mage.player.ai.rl.RLLogPaths;
import mage.player.ai.rl.RLTrainer;
import mage.player.ai.rl.StateSequenceBuilder;
import mage.player.ai.util.CombatUtil;
import mage.players.Player;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.target.TargetCard;
import mage.cards.repository.CardInfo;
import mage.cards.repository.CardRepository;

public class ComputerPlayerRL extends ComputerPlayer7 {

    // Heuristic step shaping is opt-in; default off so learning is driven by returns.
    private static final boolean USE_HEURISTIC_STEP_REWARDS = "1".equals(System.getenv().getOrDefault("RL_HEURISTIC_STEP_REWARDS", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_HEURISTIC_STEP_REWARDS", "0"));
    // Legacy shaping coefficients (used only when RL_HEURISTIC_STEP_REWARDS is enabled).
    private static final double LAND_PLAY_REWARD = 0.05;
    private static final double SPELL_CAST_REWARD = 0.02;
    private static final double ATTACK_REWARD = 0.0;
    private static final double BLOCK_REWARD = 0.0;
    private static final double TARGET_OPP_REWARD = 0.0;
    private static final double TARGET_SELF_PENALTY = 0.0;
    private static final boolean ACTIVATION_DIAG = "1".equals(System.getenv().getOrDefault("RL_ACTIVATION_DIAG", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ACTIVATION_DIAG", "0"));
    private static final boolean USE_ENGINE_CHOICES = "1".equals(System.getenv().getOrDefault("RL_ENGINE_CHOICES", "1"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ENGINE_CHOICES", "1"));
    private static final boolean MULLIGAN_TRACE = "1".equals(System.getenv().getOrDefault("RL_MULLIGAN_TRACE", "0"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_MULLIGAN_TRACE", "0"));
    private static final boolean MULLIGAN_TRACE_JSONL = "1".equals(System.getenv().getOrDefault("RL_MULLIGAN_TRACE_JSONL", "1"))
            || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_MULLIGAN_TRACE_JSONL", "1"));
    private static final boolean BASE_STATE_CACHE_ENABLED = EnvConfig.bool("RL_BASE_STATE_CACHE_ENABLED", true);
    private static final boolean ALTERNATIVE_COST_SIM_CACHE_ENABLED =
            EnvConfig.bool("RL_ALTCOST_SIM_CACHE_ENABLED", true);
    private static final boolean PLAYABLE_CACHE_ENABLED =
            EnvConfig.bool("RL_PLAYABLE_CACHE_ENABLED", true);
    // Experimental: bypass the playable-calculation game clone and evaluate directly
    // on the live game while forcing playable-calc flags. This can be faster, but it
    // is not enabled by default because XMage still mutates some state while checking
    // playability.
    private static final boolean PLAYABLE_NOCLONE_ENABLED =
            EnvConfig.bool("RL_PLAYABLE_NOCLONE_ENABLED", false);

    // Main policy exploration (actor) - epsilon mixture + correlated full-random turn.
    private static final double ACTION_EPS_START = EnvConfig.f64("RL_ACTION_EPS_START", 0.05);
    private static final double ACTION_EPS_END = EnvConfig.f64("RL_ACTION_EPS_END", 0.01);
    private static final int ACTION_EPS_DECAY_EPISODES = EnvConfig.i32("RL_ACTION_EPS_DECAY_EPISODES", 200000);
    private static final double TURN_RANDOM_EPS_START = EnvConfig.f64("RL_FULL_TURN_RANDOM_START", 0.03);
    private static final double TURN_RANDOM_EPS_END = EnvConfig.f64("RL_FULL_TURN_RANDOM_END", 0.03);
    private static final int TURN_RANDOM_EPS_DECAY_EPISODES = EnvConfig.i32("RL_FULL_TURN_RANDOM_DECAY_EPISODES", 400000);
    private static final String TURN_UNIFORM_OLD_LOGP_SOURCE_RAW =
            EnvConfig.str("RL_TURN_UNIFORM_OLD_LOGP_SOURCE", "policy").trim().toLowerCase(Locale.ROOT);
    private static final String TURN_UNIFORM_OLD_LOGP_SOURCE =
            "behavior".equals(TURN_UNIFORM_OLD_LOGP_SOURCE_RAW) ? "behavior" : "policy";
    private static final long RL_BASE_SEED = EnvConfig.i64("RL_BASE_SEED", -1L);

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

    // ThreadLocal for activation failure tracing - captures every callback during activateAbility()
    private static final ThreadLocal<List<String>> activationTrace = ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<Boolean> traceEnabled = ThreadLocal.withInitial(() -> false);
    private static final ThreadLocal<String> choiceExplosionContext = new ThreadLocal<>();
    // Marks callbacks triggered during mana payment; used to delegate low-level
    // payment choices to engine logic for reliability.
    private static final ThreadLocal<Integer> playManaDepth = ThreadLocal.withInitial(() -> 0);
    private static final ThreadLocal<String> currentUnpaidManaText = new ThreadLocal<>();

    private PythonModel model;
    private MulliganModel mulliganModel;
    protected StateSequenceBuilder.SequenceOutput currentState;
    private transient StateSequenceBuilder.SequenceOutput cachedBaseState;
    private transient UUID cachedBaseStateGameId;
    private transient TurnPhase cachedBaseStatePhase;
    private transient UUID cachedBaseStateActivePlayerId;
    private transient UUID cachedBaseStatePriorityPlayerId;
    private transient UUID cachedBaseStateChoosingPlayerId;
    private transient int cachedBaseStateTurnNum = Integer.MIN_VALUE;
    private transient int cachedBaseStateStepNum = Integer.MIN_VALUE;
    private transient int cachedBaseStateApplyEffectsCounter = Integer.MIN_VALUE;
    private transient int cachedBaseStateStackSize = Integer.MIN_VALUE;
    private transient Map<String, Set<String>> cachedAlternativeCostValidation = new HashMap<>();
    private transient UUID cachedAltCostValidationGameId;
    private transient TurnPhase cachedAltCostValidationPhase;
    private transient UUID cachedAltCostValidationActivePlayerId;
    private transient UUID cachedAltCostValidationPriorityPlayerId;
    private transient UUID cachedAltCostValidationChoosingPlayerId;
    private transient int cachedAltCostValidationTurnNum = Integer.MIN_VALUE;
    private transient int cachedAltCostValidationStepNum = Integer.MIN_VALUE;
    private transient int cachedAltCostValidationApplyEffectsCounter = Integer.MIN_VALUE;
    private transient int cachedAltCostValidationStackSize = Integer.MIN_VALUE;
    private transient String cachedPlayableStateKey;
    private transient List<ActivatedAbility> cachedPlayableOptions = new ArrayList<>();
    private final List<StateSequenceBuilder.TrainingData> trainingBuffer;
    private final java.util.Map<StateSequenceBuilder.ActionType, Integer> decisionCountsByHead;
    private Ability currentAbility;
    private int mulligansTaken = 0;
    private int currentEpisode = -1; // Main model episode for logging
    private int mulliganEpisode = -1; // Mulligan model episode for epsilon-greedy
    private int lastLoggedTurn = -1; // Track turn changes for game logging
    // Fallback logger attachment for callbacks that may run off the trainer thread.
    private transient GameLogger attachedGameLogger = null;
    private int lastExplorationTurn = Integer.MIN_VALUE;
    private boolean turnForceUniform = false;
    private double turnActionEps = 0.0;
    private double turnRandomEps = 0.0;
    private String turnExplorationMode = "policy";
    private final Random stochasticRng = new Random();

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

    private static final class BehaviorPolicyView {

        final float[] policyProbs;
        final float[] behaviorProbs;
        final String mode;
        final double actionEps;
        final double turnRandomEps;

        private BehaviorPolicyView(
                float[] policyProbs,
                float[] behaviorProbs,
                String mode,
                double actionEps,
                double turnRandomEps
        ) {
            this.policyProbs = policyProbs;
            this.behaviorProbs = behaviorProbs;
            this.mode = mode;
            this.actionEps = actionEps;
            this.turnRandomEps = turnRandomEps;
        }
    }

    private static final class SinglePickResult {

        final int chosenIdx;
        final float oldLogp;
        final float selectedBehaviorProb;
        final float selectedOldProb;
        final String oldLogpSource;
        final BehaviorPolicyView behavior;

        private SinglePickResult(
                int chosenIdx,
                float oldLogp,
                float selectedBehaviorProb,
                float selectedOldProb,
                String oldLogpSource,
                BehaviorPolicyView behavior
        ) {
            this.chosenIdx = chosenIdx;
            this.oldLogp = oldLogp;
            this.selectedBehaviorProb = selectedBehaviorProb;
            this.selectedOldProb = selectedOldProb;
            this.oldLogpSource = oldLogpSource;
            this.behavior = behavior;
        }
    }

    private static final class SequentialPickResult {

        final List<Integer> selectedIndices;
        final float oldLogpTotal;
        final String oldLogpSource;
        final BehaviorPolicyView behavior;

        private SequentialPickResult(
                List<Integer> selectedIndices,
                float oldLogpTotal,
                String oldLogpSource,
                BehaviorPolicyView behavior
        ) {
            this.selectedIndices = selectedIndices;
            this.oldLogpTotal = oldLogpTotal;
            this.oldLogpSource = oldLogpSource;
            this.behavior = behavior;
        }
    }

    private boolean isMainExplorationEnabled() {
        return trainingEnabled && !greedyMode && "train".equals(policyKey) && currentEpisode >= 0;
    }

    private static double linearDecay(int episodeNum, double start, double end, int decayEpisodes) {
        if (decayEpisodes <= 0) {
            return end;
        }
        if (episodeNum <= 0) {
            return start;
        }
        if (episodeNum >= decayEpisodes) {
            return end;
        }
        double t = (double) episodeNum / (double) decayEpisodes;
        return start + (end - start) * t;
    }

    private void refreshTurnExploration(Game game) {
        if (!isMainExplorationEnabled() || game == null) {
            turnForceUniform = false;
            turnActionEps = 0.0;
            turnRandomEps = 0.0;
            turnExplorationMode = "policy";
            return;
        }

        int turnNum = game.getTurnNum();
        if (turnNum == lastExplorationTurn) {
            return;
        }

        lastExplorationTurn = turnNum;
        turnActionEps = Math.max(0.0, Math.min(1.0,
                linearDecay(currentEpisode, ACTION_EPS_START, ACTION_EPS_END, ACTION_EPS_DECAY_EPISODES)));
        turnRandomEps = Math.max(0.0, Math.min(1.0,
                linearDecay(currentEpisode, TURN_RANDOM_EPS_START, TURN_RANDOM_EPS_END, TURN_RANDOM_EPS_DECAY_EPISODES)));

        UUID activeId = game.getActivePlayerId();
        boolean ownTurn = activeId != null && activeId.equals(getId());
        if (ownTurn && turnRandomEps > 0.0) {
            turnForceUniform = stochasticRng.nextDouble() < turnRandomEps;
            if (turnForceUniform) {
                MetricsCollector.getInstance().recordMainTurnUniform();
            }
        } else {
            turnForceUniform = false;
        }
        turnExplorationMode = turnForceUniform
                ? "turn_uniform"
                : (turnActionEps > 0.0 ? "epsilon_mix" : "policy");
    }

    private void invalidateBaseStateCache() {
        cachedBaseState = null;
        cachedBaseStateGameId = null;
        cachedBaseStatePhase = null;
        cachedBaseStateActivePlayerId = null;
        cachedBaseStatePriorityPlayerId = null;
        cachedBaseStateChoosingPlayerId = null;
        cachedBaseStateTurnNum = Integer.MIN_VALUE;
        cachedBaseStateStepNum = Integer.MIN_VALUE;
        cachedBaseStateApplyEffectsCounter = Integer.MIN_VALUE;
        cachedBaseStateStackSize = Integer.MIN_VALUE;
        cachedPlayableStateKey = null;
        cachedPlayableOptions.clear();
    }

    private void invalidateAlternativeCostValidationCache() {
        cachedAlternativeCostValidation.clear();
        cachedAltCostValidationGameId = null;
        cachedAltCostValidationPhase = null;
        cachedAltCostValidationActivePlayerId = null;
        cachedAltCostValidationPriorityPlayerId = null;
        cachedAltCostValidationChoosingPlayerId = null;
        cachedAltCostValidationTurnNum = Integer.MIN_VALUE;
        cachedAltCostValidationStepNum = Integer.MIN_VALUE;
        cachedAltCostValidationApplyEffectsCounter = Integer.MIN_VALUE;
        cachedAltCostValidationStackSize = Integer.MIN_VALUE;
    }

    private void refreshAlternativeCostValidationCache(Game game) {
        if (!ALTERNATIVE_COST_SIM_CACHE_ENABLED) {
            invalidateAlternativeCostValidationCache();
            return;
        }

        TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;
        GameState state = game.getState();
        UUID gameId = game.getId();
        UUID activePlayerId = game.getActivePlayerId();
        UUID priorityPlayerId = state != null ? state.getPriorityPlayerId() : null;
        UUID choosingPlayerId = state != null ? state.getChoosingPlayerId() : null;
        int turnNum = game.getTurnNum();
        int stepNum = state != null ? state.getStepNum() : Integer.MIN_VALUE;
        int applyEffectsCounter = state != null ? state.getApplyEffectsCounter() : Integer.MIN_VALUE;
        int stackSize = game.getStack() != null ? game.getStack().size() : 0;

        boolean sameState = java.util.Objects.equals(cachedAltCostValidationGameId, gameId)
                && cachedAltCostValidationPhase == turnPhase
                && java.util.Objects.equals(cachedAltCostValidationActivePlayerId, activePlayerId)
                && java.util.Objects.equals(cachedAltCostValidationPriorityPlayerId, priorityPlayerId)
                && java.util.Objects.equals(cachedAltCostValidationChoosingPlayerId, choosingPlayerId)
                && cachedAltCostValidationTurnNum == turnNum
                && cachedAltCostValidationStepNum == stepNum
                && cachedAltCostValidationApplyEffectsCounter == applyEffectsCounter
                && cachedAltCostValidationStackSize == stackSize;

        if (sameState) {
            return;
        }

        cachedAlternativeCostValidation.clear();
        cachedAltCostValidationGameId = gameId;
        cachedAltCostValidationPhase = turnPhase;
        cachedAltCostValidationActivePlayerId = activePlayerId;
        cachedAltCostValidationPriorityPlayerId = priorityPlayerId;
        cachedAltCostValidationChoosingPlayerId = choosingPlayerId;
        cachedAltCostValidationTurnNum = turnNum;
        cachedAltCostValidationStepNum = stepNum;
        cachedAltCostValidationApplyEffectsCounter = applyEffectsCounter;
        cachedAltCostValidationStackSize = stackSize;
    }

    private String buildAlternativeCostValidationKey(ActivatedAbility ability, Game game) {
        StringBuilder key = new StringBuilder(256);
        key.append(ability.getSourceId()).append('|');
        key.append(ability.getClass().getName()).append('|');
        key.append(ability.getRule()).append('|');
        key.append(ability.getManaCostsToPay() != null ? ability.getManaCostsToPay().getText() : "").append('|');
        key.append(ability.getCosts() != null ? ability.getCosts().toString() : "").append('|');

        Zone sourceZone = game.getState() != null ? game.getState().getZone(ability.getSourceId()) : null;
        key.append(sourceZone != null ? sourceZone.name() : "UNKNOWN").append('|');

        for (Target target : ability.getTargets()) {
            key.append(target.getClass().getName()).append(':')
                    .append(target.getMinNumberOfTargets()).append(':')
                    .append(target.getMaxNumberOfTargets()).append(':')
                    .append(target.getTargetName()).append('|');
        }

        return key.toString();
    }

    private StateSequenceBuilder.SequenceOutput getOrBuildBaseState(Game game) {
        TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;
        GameState state = game.getState();
        UUID gameId = game.getId();
        UUID activePlayerId = game.getActivePlayerId();
        UUID priorityPlayerId = state != null ? state.getPriorityPlayerId() : null;
        UUID choosingPlayerId = state != null ? state.getChoosingPlayerId() : null;
        int turnNum = game.getTurnNum();
        int stepNum = state != null ? state.getStepNum() : Integer.MIN_VALUE;
        int applyEffectsCounter = state != null ? state.getApplyEffectsCounter() : Integer.MIN_VALUE;
        int stackSize = game.getStack() != null ? game.getStack().size() : 0;

        if (BASE_STATE_CACHE_ENABLED
                && cachedBaseState != null
                && java.util.Objects.equals(cachedBaseStateGameId, gameId)
                && cachedBaseStatePhase == turnPhase
                && java.util.Objects.equals(cachedBaseStateActivePlayerId, activePlayerId)
                && java.util.Objects.equals(cachedBaseStatePriorityPlayerId, priorityPlayerId)
                && java.util.Objects.equals(cachedBaseStateChoosingPlayerId, choosingPlayerId)
                && cachedBaseStateTurnNum == turnNum
                && cachedBaseStateStepNum == stepNum
                && cachedBaseStateApplyEffectsCounter == applyEffectsCounter
                && cachedBaseStateStackSize == stackSize) {
            currentState = cachedBaseState;
            return cachedBaseState;
        }

        StateSequenceBuilder.SequenceOutput baseState =
                StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);
        currentState = baseState;
        if (BASE_STATE_CACHE_ENABLED) {
            cachedBaseState = baseState;
            cachedBaseStateGameId = gameId;
            cachedBaseStatePhase = turnPhase;
            cachedBaseStateActivePlayerId = activePlayerId;
            cachedBaseStatePriorityPlayerId = priorityPlayerId;
            cachedBaseStateChoosingPlayerId = choosingPlayerId;
            cachedBaseStateTurnNum = turnNum;
            cachedBaseStateStepNum = stepNum;
            cachedBaseStateApplyEffectsCounter = applyEffectsCounter;
            cachedBaseStateStackSize = stackSize;
        } else {
            invalidateBaseStateCache();
        }
        return baseState;
    }

    private String buildPlayableStateKey(Game game) {
        StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);
        int hash = 1;
        hash = 31 * hash + baseState.mask.hashCode();
        hash = 31 * hash + baseState.tokenIds.hashCode();
        for (float[] token : baseState.tokens) {
            hash = 31 * hash + Arrays.hashCode(token);
        }

        Player self = game.getPlayer(getId());
        int landsPlayed = self != null ? self.getLandsPlayed() : Integer.MIN_VALUE;
        int landsPerTurn = self != null ? self.getLandsPerTurn() : Integer.MIN_VALUE;

        StringBuilder key = new StringBuilder(160);
        key.append(game.getId()).append('|');
        key.append(game.getTurnNum()).append('|');
        key.append(game.getPhase() != null ? game.getPhase().getType() : "NONE").append('|');
        key.append(game.getActivePlayerId()).append('|');
        key.append(game.getState() != null ? game.getState().getPriorityPlayerId() : null).append('|');
        key.append(game.getState() != null ? game.getState().getChoosingPlayerId() : null).append('|');
        key.append(game.getState() != null ? game.getState().getApplyEffectsCounter() : Integer.MIN_VALUE).append('|');
        key.append(game.getState() != null ? game.getState().getStepNum() : Integer.MIN_VALUE).append('|');
        key.append(game.getStack() != null ? game.getStack().size() : 0).append('|');
        key.append(landsPlayed).append('|');
        key.append(landsPerTurn).append('|');
        key.append(hash);
        return key.toString();
    }

    private List<ActivatedAbility> getPlayableForCurrentState(Game game) {
        List<ActivatedAbility> playable;
        if (!PLAYABLE_CACHE_ENABLED) {
            cachedPlayableStateKey = null;
            cachedPlayableOptions.clear();
            return PLAYABLE_NOCLONE_ENABLED
                    ? getPlayableFast(game, true, Zone.ALL, true)
                    : getPlayable(game, true);
        }

        String stateKey = buildPlayableStateKey(game);
        if (stateKey.equals(cachedPlayableStateKey) && !cachedPlayableOptions.isEmpty()) {
            return new ArrayList<>(cachedPlayableOptions);
        }

        playable = PLAYABLE_NOCLONE_ENABLED
                ? getPlayableFast(game, true, Zone.ALL, true)
                : getPlayable(game, true);
        cachedPlayableStateKey = stateKey;
        cachedPlayableOptions = new ArrayList<>(playable);
        return playable;
    }

    private static float[] normalizePolicyScores(float[] actionScores, int[] candidateMask, int candidateCount) {
        float[] logits = new float[candidateCount];
        float maxLogit = -Float.MAX_VALUE;
        for (int i = 0; i < candidateCount; i++) {
            float p = (actionScores != null && i < actionScores.length) ? actionScores[i] : 0.0f;
            if (Float.isNaN(p) || Float.isInfinite(p) || p <= 0.0f) {
                p = 1e-20f;
            }
            float logit = (float) Math.log(p);
            if (candidateMask != null && (i >= candidateMask.length || candidateMask[i] != 1)) {
                logit = -1e9f;
            }
            logits[i] = logit;
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float[] probs = new float[candidateCount];
        float sum = 0.0f;
        for (int i = 0; i < candidateCount; i++) {
            boolean valid = candidateMask == null || (i < candidateMask.length && candidateMask[i] == 1);
            float e = valid ? (float) Math.exp(logits[i] - maxLogit) : 0.0f;
            probs[i] = e;
            sum += e;
        }
        if (sum <= 0.0f || Float.isNaN(sum) || Float.isInfinite(sum)) {
            float uniformProb = 1.0f / Math.max(1, candidateCount);
            Arrays.fill(probs, uniformProb);
        } else {
            for (int i = 0; i < candidateCount; i++) {
                probs[i] /= sum;
            }
        }
        return probs;
    }

    private static float[] buildUniformProbs(int candidateCount) {
        float[] uniform = new float[candidateCount];
        float p = 1.0f / Math.max(1, candidateCount);
        Arrays.fill(uniform, p);
        return uniform;
    }

    private BehaviorPolicyView buildBehaviorPolicy(float[] actionScores, int[] candidateMask, int candidateCount, Game game) {
        refreshTurnExploration(game);
        float[] policy = normalizePolicyScores(actionScores, candidateMask, candidateCount);
        float[] behavior = Arrays.copyOf(policy, policy.length);
        String mode = "policy";

        if (isMainExplorationEnabled()) {
            if (turnForceUniform) {
                behavior = buildUniformProbs(candidateCount);
                mode = "turn_uniform";
            } else if (turnActionEps > 0.0) {
                float[] uniform = buildUniformProbs(candidateCount);
                double eps = Math.max(0.0, Math.min(1.0, turnActionEps));
                for (int i = 0; i < candidateCount; i++) {
                    behavior[i] = (float) ((1.0 - eps) * policy[i] + eps * uniform[i]);
                }
                MetricsCollector.getInstance().recordMainActionEpsilonDecision();
                mode = "epsilon_mix";
            }
        }
        return new BehaviorPolicyView(policy, behavior, mode, turnActionEps, turnRandomEps);
    }

    private int sampleFromDistribution(float[] probs, int candidateCount) {
        float r = stochasticRng.nextFloat();
        float cdf = 0.0f;
        for (int i = 0; i < candidateCount; i++) {
            cdf += probs[i];
            if (r <= cdf) {
                return i;
            }
        }
        return Math.max(0, candidateCount - 1);
    }

    private int argmax(float[] probs, int candidateCount, boolean[] selectedMask) {
        int bestIdx = -1;
        float bestVal = -1.0f;
        for (int i = 0; i < candidateCount; i++) {
            if (selectedMask != null && selectedMask[i]) {
                continue;
            }
            if (probs[i] > bestVal) {
                bestVal = probs[i];
                bestIdx = i;
            }
        }
        return bestIdx >= 0 ? bestIdx : 0;
    }

    private boolean usePolicyOldLogpForTurnUniform(BehaviorPolicyView behavior) {
        return behavior != null
                && "turn_uniform".equals(behavior.mode)
                && "policy".equals(TURN_UNIFORM_OLD_LOGP_SOURCE);
    }

    private SinglePickResult sampleSinglePick(
            float[] actionScores,
            int[] candidateMask,
            int candidateCount,
            Game game
    ) {
        BehaviorPolicyView behavior = buildBehaviorPolicy(actionScores, candidateMask, candidateCount, game);
        int chosenIdx = greedyMode
                ? argmax(behavior.policyProbs, candidateCount, null)
                : sampleFromDistribution(behavior.behaviorProbs, candidateCount);
        int clampedChosenIdx = Math.max(0, Math.min(chosenIdx, candidateCount - 1));
        float selectedBehaviorProb = behavior.behaviorProbs[clampedChosenIdx];
        boolean policyOldLogp = usePolicyOldLogpForTurnUniform(behavior);
        float selectedOldProb = policyOldLogp
                ? behavior.policyProbs[clampedChosenIdx]
                : behavior.behaviorProbs[clampedChosenIdx];
        float oldLogp = (float) Math.log(Math.max(1e-8f, selectedOldProb));
        String oldLogpSource = policyOldLogp ? "policy" : "behavior";
        return new SinglePickResult(chosenIdx, oldLogp, selectedBehaviorProb, selectedOldProb, oldLogpSource, behavior);
    }

    private SequentialPickResult sampleSequentialWithoutReplacement(
            float[] actionScores,
            int[] candidateMask,
            int candidateCount,
            int picks,
            Game game
    ) {
        BehaviorPolicyView behavior = buildBehaviorPolicy(actionScores, candidateMask, candidateCount, game);
        List<Integer> selectedIndices = new ArrayList<>();
        boolean[] selected = new boolean[candidateCount];
        float oldLogpTotal = 0.0f;
        boolean policyOldLogp = usePolicyOldLogpForTurnUniform(behavior);
        float[] oldLogpProbs = policyOldLogp ? behavior.policyProbs : behavior.behaviorProbs;

        int safePicks = Math.max(0, Math.min(picks, candidateCount));
        for (int t = 0; t < safePicks; t++) {
            float behaviorDenom = 0.0f;
            float oldLogpDenom = 0.0f;
            for (int i = 0; i < candidateCount; i++) {
                if (!selected[i]) {
                    behaviorDenom += behavior.behaviorProbs[i];
                    oldLogpDenom += oldLogpProbs[i];
                }
            }
            if (!(behaviorDenom > 0.0f)) {
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

            int chosenIdx;
            if (greedyMode) {
                chosenIdx = argmax(behavior.policyProbs, candidateCount, selected);
            } else {
                float r = stochasticRng.nextFloat() * behaviorDenom;
                float cdf = 0.0f;
                chosenIdx = -1;
                for (int i = 0; i < candidateCount; i++) {
                    if (selected[i]) {
                        continue;
                    }
                    cdf += behavior.behaviorProbs[i];
                    if (r <= cdf) {
                        chosenIdx = i;
                        break;
                    }
                }
                if (chosenIdx < 0) {
                    for (int i = candidateCount - 1; i >= 0; i--) {
                        if (!selected[i]) {
                            chosenIdx = i;
                            break;
                        }
                    }
                }
            }
            if (chosenIdx < 0) {
                break;
            }

            if (!(oldLogpDenom > 0.0f)) {
                oldLogpTotal += (float) Math.log(1e-8f);
            } else {
                float pCond = oldLogpProbs[chosenIdx] / oldLogpDenom;
                oldLogpTotal += (float) Math.log(Math.max(1e-8f, pCond));
            }
            selected[chosenIdx] = true;
            selectedIndices.add(chosenIdx);
        }

        return new SequentialPickResult(selectedIndices, oldLogpTotal, policyOldLogp ? "policy" : "behavior", behavior);
    }

    private String explorationAnnotation(BehaviorPolicyView behavior, int chosenIdx, String oldLogpSource) {
        if (behavior == null || chosenIdx < 0 || chosenIdx >= behavior.behaviorProbs.length) {
            return "";
        }
        String source = oldLogpSource == null ? "behavior" : oldLogpSource;
        float[] oldLogpProbs = "policy".equals(source) ? behavior.policyProbs : behavior.behaviorProbs;
        float selectedOldProb = chosenIdx < oldLogpProbs.length ? oldLogpProbs[chosenIdx] : 0.0f;
        return String.format(
                "exploration{mode=%s action_eps=%.4f turn_random_eps=%.4f behavior_prob=%.6f old_logp_source=%s old_prob_selected=%.6f}",
                behavior.mode,
                behavior.actionEps,
                behavior.turnRandomEps,
                behavior.behaviorProbs[chosenIdx],
                source,
                selectedOldProb
        );
    }

    private String appendExplorationToState(String gameState, BehaviorPolicyView behavior, int chosenIdx) {
        String defaultOldLogpSource = usePolicyOldLogpForTurnUniform(behavior) ? "policy" : "behavior";
        return appendExplorationToState(gameState, behavior, chosenIdx, defaultOldLogpSource);
    }

    private String appendExplorationToState(String gameState, BehaviorPolicyView behavior, int chosenIdx, String oldLogpSource) {
        String annotation = explorationAnnotation(behavior, chosenIdx, oldLogpSource);
        if (annotation.isEmpty()) {
            return gameState;
        }
        if (gameState == null || gameState.isEmpty()) {
            return "[Exploration] " + annotation;
        }
        return gameState + "\n[Exploration] " + annotation;
    }

    private static boolean isFatalEmbeddingError(Throwable t) {
        Throwable cur = t;
        while (cur != null) {
            String msg = cur.getMessage();
            if (msg != null) {
                if (msg.contains("CardTextEmbeddings")
                        || msg.contains("card_embeddings.json")
                        || msg.contains("missing embedding for card")) {
                    return true;
                }
            }
            cur = cur.getCause();
        }
        return false;
    }

    private static boolean inPlayManaContext() {
        Integer d = playManaDepth.get();
        return d != null && d > 0;
    }

    private static String preferredManaColorForUnpaid(String unpaidText, java.util.Set<String> choices) {
        if (unpaidText == null || unpaidText.isEmpty() || choices == null || choices.isEmpty()) {
            return null;
        }
        // Prefer required colored symbols first.
        if (unpaidText.contains("{W}") && choices.contains("White")) return "White";
        if (unpaidText.contains("{U}") && choices.contains("Blue")) return "Blue";
        if (unpaidText.contains("{B}") && choices.contains("Black")) return "Black";
        if (unpaidText.contains("{R}") && choices.contains("Red")) return "Red";
        if (unpaidText.contains("{G}") && choices.contains("Green")) return "Green";
        return null;
    }

    private static int manaProducerPenalty(UUID targetId, Game game) {
        if (targetId == null || game == null) {
            return Integer.MAX_VALUE;
        }
        Permanent p = game.getPermanent(targetId);
        if (p == null) {
            return Integer.MAX_VALUE;
        }
        int penalty = 0;
        int output = estimatePermanentManaOutput(p, game);
        if (output > 0) {
            // Opportunity cost of tapping this target as a cost:
            // keep higher-output producers available for actual payment.
            penalty += 100 + (75 * output);
        }
        // Prefer tapping low-impact bodies over bigger creatures.
        try {
            penalty += Math.max(0, p.getPower().getValue());
        } catch (Exception ignored) {
        }
        return penalty;
    }

    private static int estimateAbilityManaOutput(Ability ability, Game game) {
        if (!(ability instanceof ManaAbility)) {
            return 0;
        }
        int best = 0;
        try {
            for (Mana mana : ((ManaAbility) ability).getNetMana(game)) {
                if (mana != null) {
                    best = Math.max(best, Math.max(1, mana.count()));
                }
            }
        } catch (Exception ignored) {
            best = Math.max(best, 1);
        }
        // Some mana abilities (notably "any color" selectors) may not expose
        // concrete net mana options before a choice is made. Treat those as at
        // least 1 to avoid undervaluing/incorrectly preferring dominated paths.
        return Math.max(1, best);
    }

    private static int estimatePermanentManaOutput(Permanent p, Game game) {
        if (p == null || game == null) {
            return 0;
        }
        int best = 0;
        try {
            for (Ability a : p.getAbilities(game)) {
                best = Math.max(best, estimateAbilityManaOutput(a, game));
            }
        } catch (Exception ignored) {
            return best;
        }
        return best;
    }

    private static boolean hasTapTargetCost(Ability ability) {
        if (ability == null) {
            return false;
        }
        try {
            for (mage.abilities.costs.Cost cost : ability.getCosts()) {
                if (cost instanceof mage.abilities.costs.common.TapTargetCost) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            return false;
        }
        return false;
    }

    private static boolean hasTapSourceCost(Ability ability) {
        if (ability == null) {
            return false;
        }
        try {
            for (mage.abilities.costs.Cost cost : ability.getCosts()) {
                if (cost instanceof mage.abilities.costs.common.TapSourceCost) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            return false;
        }
        return false;
    }

    private static int manaProducerUsePenalty(MageObject obj, Game game) {
        if (!(obj instanceof Permanent) || game == null) {
            return Integer.MAX_VALUE / 4;
        }
        Permanent p = (Permanent) obj;
        int bestAbilityPenalty = Integer.MAX_VALUE / 4;
        int bestOutput = 0;
        boolean foundManaAbility = false;
        try {
            for (Ability a : p.getAbilities(game)) {
                if (!(a instanceof ManaAbility)) {
                    continue;
                }
                foundManaAbility = true;
                int abilityPenalty = 0;
                if (hasTapTargetCost(a)) {
                    // Abilities like Saruli Caretaker consume another permanent.
                    abilityPenalty += 300;
                }
                int output = estimateAbilityManaOutput(a, game);
                bestOutput = Math.max(bestOutput, output);
                bestAbilityPenalty = Math.min(bestAbilityPenalty, abilityPenalty);
            }
        } catch (Exception ignored) {
            return Integer.MAX_VALUE / 8;
        }
        if (!foundManaAbility) {
            return Integer.MAX_VALUE / 8;
        }

        int penalty = bestAbilityPenalty;
        // Prefer producers with larger immediate output, but keep complexity dominant.
        penalty -= (20 * Math.max(1, bestOutput));
        if (p.isLand(game)) {
            penalty -= 40;
        }
        return penalty;
    }

    private static String requiredManaColorFromUnpaid(String unpaidText) {
        if (unpaidText == null || unpaidText.isEmpty()) {
            return null;
        }
        if (unpaidText.contains("{W}")) return "White";
        if (unpaidText.contains("{U}")) return "Blue";
        if (unpaidText.contains("{B}")) return "Black";
        if (unpaidText.contains("{R}")) return "Red";
        if (unpaidText.contains("{G}")) return "Green";
        return null;
    }

    private static boolean manaCanProduceColor(Mana mana, String color) {
        if (mana == null || color == null) {
            return false;
        }
        switch (color) {
            case "White":
                return mana.getWhite() > 0 || mana.getAny() > 0;
            case "Blue":
                return mana.getBlue() > 0 || mana.getAny() > 0;
            case "Black":
                return mana.getBlack() > 0 || mana.getAny() > 0;
            case "Red":
                return mana.getRed() > 0 || mana.getAny() > 0;
            case "Green":
                return mana.getGreen() > 0 || mana.getAny() > 0;
            default:
                return false;
        }
    }

    private static boolean permanentCanProduceColor(Permanent permanent, String color, Game game) {
        if (permanent == null || color == null || game == null) {
            return false;
        }
        try {
            for (Ability a : permanent.getAbilities(game)) {
                if (!(a instanceof ManaAbility)) {
                    continue;
                }
                for (Mana mana : ((ManaAbility) a).getNetMana(game)) {
                    if (manaCanProduceColor(mana, color)) {
                        return true;
                    }
                }
            }
        } catch (Exception ignored) {
            return false;
        }
        return false;
    }

    private static boolean abilityCanProduceColor(Ability ability, String color, Game game) {
        if (!(ability instanceof ManaAbility) || color == null || game == null) {
            return false;
        }
        try {
            for (Mana mana : ((ManaAbility) ability).getNetMana(game)) {
                if (manaCanProduceColor(mana, color)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            return false;
        }
        return false;
    }

    public void setAttachedGameLogger(GameLogger gameLogger) {
        this.attachedGameLogger = gameLogger;
    }

    private GameLogger resolveGameLogger() {
        try {
            GameLogger threadLogger = RLTrainer.threadLocalGameLogger.get();
            if (threadLogger != null && threadLogger.isEnabled()) {
                return threadLogger;
            }
        } catch (Exception ignored) {
        }
        if (attachedGameLogger != null && attachedGameLogger.isEnabled()) {
            return attachedGameLogger;
        }
        return null;
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
    private static final Object CANDIDATE_EXPLOSION_LOG_LOCK = new Object();
    private static final String CANDIDATE_EXPLOSIONS_LOG_FILE = System.getenv().getOrDefault(
            "CANDIDATE_EXPLOSIONS_LOG_FILE",
            RLLogPaths.LOGS_BASE_DIR + "/health/candidate_explosions.log"
    );

    private long mulliganEventSeq = 0L;

    // -----------------------------------------------------------------------
    // Draft model state (reset per draft episode via resetDraftState())
    // -----------------------------------------------------------------------
    private DraftPythonMLBridge draftBridge = null;
    private DraftLogger draftLogger = null;

    // Per-draft tracking
    private final List<Card> draftPool = new ArrayList<>();
    private final List<DraftPickRecord> draftSeenCards = new ArrayList<>();
    // Training data for current draft
    private final List<DraftStepData> draftSteps = new ArrayList<>();

    // Draft context set externally by DraftTrainer before each draft
    private int currentDraftEpisode = -1;
    private boolean draftLoggingEnabled = false;
    private static final boolean DRAFT_MIN_LANDS_ENABLE =
            "1".equals(System.getenv().getOrDefault("DRAFT_MIN_LANDS_ENABLE", "1"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("DRAFT_MIN_LANDS_ENABLE", "1"));
    private static final int DRAFT_MIN_LANDS_START =
            Integer.parseInt(System.getenv().getOrDefault("DRAFT_MIN_LANDS_START", "12"));
    private static final int DRAFT_MIN_LANDS_END =
            Integer.parseInt(System.getenv().getOrDefault("DRAFT_MIN_LANDS_END", "0"));
    private static final int DRAFT_MIN_LANDS_DECAY_EPISODES =
            Math.max(1, Integer.parseInt(System.getenv().getOrDefault("DRAFT_MIN_LANDS_DECAY_EPISODES", "5000")));
    private static final int DRAFT_TARGET_DECK_SIZE = 40;
    private static final int DRAFT_LAND_COUNT_MIN = 0;
    private static final int DRAFT_LAND_COUNT_MAX = 20;

    /** Lightweight container for one draft decision step's training data. */
    public static class DraftStepData {
        public final float[] stateFlat;
        public final int[]   maskFlat;
        public final int[]   tokenIds;
        public final float[] candFeats;
        public final int[]   candIds;
        public final int[]   candMask;
        public final int     chosenIdx;
        public final float   logProb;
        public final float   value;
        public final int     headIdx; // 0=pick, 1=construction
        public final boolean forcedByCurriculum;
        public final String  constructionStepKind;

        DraftStepData(float[] stateFlat, int[] maskFlat, int[] tokenIds,
                      float[] candFeats, int[] candIds, int[] candMask,
                      int chosenIdx, float logProb, float value, int headIdx) {
            this(stateFlat, maskFlat, tokenIds, candFeats, candIds, candMask, chosenIdx, logProb, value, headIdx, false, null);
        }

        DraftStepData(float[] stateFlat, int[] maskFlat, int[] tokenIds,
                      float[] candFeats, int[] candIds, int[] candMask,
                      int chosenIdx, float logProb, float value, int headIdx,
                      boolean forcedByCurriculum, String constructionStepKind) {
            this.stateFlat = stateFlat;
            this.maskFlat = maskFlat;
            this.tokenIds = tokenIds;
            this.candFeats = candFeats;
            this.candIds = candIds;
            this.candMask = candMask;
            this.chosenIdx = chosenIdx;
            this.logProb = logProb;
            this.value = value;
            this.headIdx = headIdx;
            this.forcedByCurriculum = forcedByCurriculum;
            this.constructionStepKind = constructionStepKind;
        }
    }

    /**
     * Reset draft state before starting a new draft episode.
     * Called by DraftTrainer before runDraft().
     */
    public void resetDraftState(DraftPythonMLBridge bridge, DraftLogger logger,
                                int episodeNum, boolean loggingEnabled) {
        this.draftBridge = bridge;
        this.draftLogger = logger;
        this.currentDraftEpisode = episodeNum;
        this.draftLoggingEnabled = loggingEnabled;
        this.draftPool.clear();
        this.draftSeenCards.clear();
        this.draftSteps.clear();
    }

    public List<DraftStepData> getDraftSteps() {
        return draftSteps;
    }

    public List<Card> getDraftPool() {
        return draftPool;
    }

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

    private static void candidateExplosionLog(String line) {
        if (line == null || line.isEmpty()) {
            return;
        }
        try {
            Path p = Paths.get(CANDIDATE_EXPLOSIONS_LOG_FILE);
            Path parent = p.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String stamped = LocalDateTime.now().format(MULL_TRAIN_TS)
                    + " - candidate_explosion - WARN - "
                    + line + System.lineSeparator();
            synchronized (CANDIDATE_EXPLOSION_LOG_LOCK) {
                Files.write(p, stamped.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            }
        } catch (Exception ignored) {
            // Never fail gameplay because of diagnostics logging.
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
        this.attachedGameLogger = player.attachedGameLogger;
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
            if (isFatalEmbeddingError(t)) {
                try {
                    RLTrainer.threadLocalLogger.get().error(
                            "RL priority() fatal embedding error; aborting run: " + t.getMessage());
                } catch (Exception ignored) {
                    // ignore
                }
                if (t instanceof RuntimeException) {
                    throw (RuntimeException) t;
                }
                if (t instanceof Error) {
                    throw (Error) t;
                }
                throw new IllegalStateException("Fatal embedding configuration error", t);
            }
            // Never let RL decision logic crash the game engine.
            // A single bad activation/choice can otherwise trigger "too many errors" and end the whole game.
            try {
                RLTrainer.threadLocalLogger.get().warn("RL priority() caught exception; forcing pass: " + t.getMessage());
            } catch (Exception ignored) {
                // ignore
            }
            try {
                GameLogger gameLogger = resolveGameLogger();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    String phase = "unknown";
                    try {
                        if (game != null && game.getPhase() != null && game.getPhase().getType() != null) {
                            phase = game.getPhase().getType().toString();
                        }
                    } catch (Exception ignored) {
                    }
                    int turnNum = -1;
                    try {
                        if (game != null) {
                            turnNum = game.getTurnNum();
                        }
                    } catch (Exception ignored) {
                    }
                    gameLogger.log(String.format(
                            "PRIORITY_FALLBACK_PASS: player=%s turn=%d phase=%s reason=%s",
                            getName(),
                            turnNum,
                            phase,
                            trunc(String.valueOf(t.getMessage()), 300)
                    ));
                }
            } catch (Exception ignored) {
                // don't fail on fallback logging
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
            logCandidateExplosionDebug(candidates, actionType, game, source);
        }

        // Prevent infinite loops if the engine requests more picks than we kept after truncation
        maxTargets = Math.min(maxTargets, candidateCount);
        minTargets = Math.min(minTargets, candidateCount);

        if (candidateCount == 1) {
            return Arrays.asList(0);
        } else if (candidateCount == 0) {
            return Arrays.asList();
        }

        StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

        // Build padded candidate tensors
        int[] candidateActionIds = new int[maxCandidates];
        float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
        int[] candidateMask = new int[maxCandidates]; // 1 valid, 0 padding

        for (int i = 0; i < candidateCount; i++) {
            candidateMask[i] = 1;
            T cand = candidates.get(i);
            candidateActionIds[i] = computeCandidateActionId(actionType, game, source, cand);
            candidateFeatures[i] = computeCandidateFeatures(actionType, game, source, cand, candFeatDim, baseState);
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

        int picks = maxTargets; // historical behavior: pick maxTargets (then ensure >= minTargets via truncation above)
        SequentialPickResult pickResult = sampleSequentialWithoutReplacement(
                actionProbs,
                candidateMask,
                candidateCount,
                picks,
                game
        );
        float[] maskedProbs = pickResult.behavior.behaviorProbs;
        float[] policyMaskedProbs = pickResult.behavior.policyProbs;

        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info("Policy probabilities: " + Arrays.toString(policyMaskedProbs));
            RLTrainer.threadLocalLogger.get().info("Behavior probabilities: " + Arrays.toString(maskedProbs)
                    + " mode=" + pickResult.behavior.mode
                    + " actionEps=" + pickResult.behavior.actionEps
                    + " turnRandomEps=" + pickResult.behavior.turnRandomEps);
            RLTrainer.threadLocalLogger.get().info("Value score: " + valueScore);
        }

        // Store for game logging
        this.lastActionProbs = maskedProbs.clone();
        this.lastValueScore = valueScore;

        List<Integer> selectedIndices = pickResult.selectedIndices;
        float oldLogpTotal = pickResult.oldLogpTotal;

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
            double stepReward = computeStepReward(actionType, candidates.get(firstChosen), game);
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

    private <T> void logCandidateExplosionDebug(
            List<T> candidates,
            StateSequenceBuilder.ActionType actionType,
            Game game,
            Ability source
    ) {
        try {
            String phase = "unknown";
            if (game != null && game.getStep() != null && game.getStep().getType() != null) {
                phase = game.getStep().getType().toString();
            }
            String sourceText = "none";
            if (source != null) {
                sourceText = source.toString();
                try {
                    MageObject srcObj = game != null ? game.getObject(source.getSourceId()) : null;
                    if (srcObj != null) {
                        sourceText = srcObj.getName() + " :: " + sourceText;
                    }
                } catch (Exception ignored) {
                    // best-effort only
                }
            }

            Map<String, Integer> typeCounts = new HashMap<>();
            Map<String, Integer> signatureCounts = new HashMap<>();
            for (T candidate : candidates) {
                if (candidate == null) {
                    typeCounts.merge("null", 1, Integer::sum);
                    signatureCounts.merge("null", 1, Integer::sum);
                    continue;
                }
                String type = candidate.getClass().getSimpleName();
                typeCounts.merge(type, 1, Integer::sum);

                String signature;
                if (candidate instanceof ActivatedAbility) {
                    ActivatedAbility a = (ActivatedAbility) candidate;
                    MageObject srcObj = game != null ? game.getObject(a.getSourceId()) : null;
                    String srcName = srcObj != null ? srcObj.getName() : "unknown-source";
                    signature = srcName + " :: " + a.toString();
                } else if (candidate instanceof MageObject) {
                    signature = ((MageObject) candidate).getName();
                } else {
                    signature = String.valueOf(candidate);
                }
                if (signature.length() > 140) {
                    signature = signature.substring(0, 140) + "...";
                }
                signatureCounts.merge(signature, 1, Integer::sum);
            }

            List<Map.Entry<String, Integer>> topTypes = typeCounts.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                    .limit(8)
                    .collect(Collectors.toList());
            List<Map.Entry<String, Integer>> topSignatures = signatureCounts.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                    .limit(12)
                    .collect(Collectors.toList());

            RLTrainer.threadLocalLogger.get().warn(
                    "genericChoose explosion: actionType=" + actionType
                            + ", phase=" + phase
                            + ", source=" + sourceText
                            + ", totalCandidates=" + candidates.size()
                            + ", uniqueSignatures=" + signatureCounts.size()
            );
            RLTrainer.threadLocalLogger.get().warn("genericChoose explosion types: " + topTypes);
            RLTrainer.threadLocalLogger.get().warn("genericChoose explosion top signatures: " + topSignatures);
            String ctx = choiceExplosionContext.get();
            if (ctx != null && !ctx.isEmpty()) {
                RLTrainer.threadLocalLogger.get().warn("genericChoose explosion choice_context: " + ctx);
            }
            String stack = buildCompactCallerStack(12);
            RLTrainer.threadLocalLogger.get().warn("genericChoose explosion stack: " + stack);
            candidateExplosionLog(
                    "actionType=" + actionType
                            + ", phase=" + phase
                            + ", source=" + sourceText
                            + ", gameId=" + (game != null ? game.getId() : "none")
                            + ", totalCandidates=" + candidates.size()
                            + ", uniqueSignatures=" + signatureCounts.size()
            );
            candidateExplosionLog("types=" + topTypes);
            candidateExplosionLog("top_signatures=" + topSignatures);
            if (ctx != null && !ctx.isEmpty()) {
                candidateExplosionLog("choice_context=" + ctx);
            }
            candidateExplosionLog("stack=" + stack);
            candidateExplosionLog("first_candidates=" + sampleCandidates(candidates, 0, Math.min(12, candidates.size()), game));
            if (candidates.size() > 12) {
                int start = Math.max(0, candidates.size() - 12);
                candidateExplosionLog("last_candidates=" + sampleCandidates(candidates, start, candidates.size(), game));
            }
            candidateExplosionLog("---");
        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn(
                    "genericChoose explosion debug failed: " + e.getMessage());
            candidateExplosionLog("debug_failed=" + e.getMessage());
        }
    }

    private String buildCompactCallerStack(int maxFrames) {
        StackTraceElement[] frames = Thread.currentThread().getStackTrace();
        List<String> filtered = new ArrayList<>();
        for (StackTraceElement f : frames) {
            String cls = f.getClassName();
            if (cls.contains("ComputerPlayerRL")
                    || cls.contains("mage.game")
                    || cls.contains("mage.abilities")) {
                filtered.add(f.getClassName() + "#" + f.getMethodName() + ":" + f.getLineNumber());
            }
            if (filtered.size() >= maxFrames) {
                break;
            }
        }
        return filtered.toString();
    }

    private <T> List<String> sampleCandidates(List<T> candidates, int from, int to, Game game) {
        List<String> out = new ArrayList<>();
        int start = Math.max(0, from);
        int end = Math.min(candidates.size(), to);
        for (int i = start; i < end; i++) {
            T candidate = candidates.get(i);
            String signature;
            if (candidate instanceof ActivatedAbility) {
                ActivatedAbility a = (ActivatedAbility) candidate;
                MageObject srcObj = game != null ? game.getObject(a.getSourceId()) : null;
                String srcName = srcObj != null ? srcObj.getName() : "unknown-source";
                signature = srcName + " :: " + a.toString();
            } else if (candidate instanceof MageObject) {
                signature = ((MageObject) candidate).getName();
            } else {
                signature = String.valueOf(candidate);
            }
            if (signature.length() > 180) {
                signature = signature.substring(0, 180) + "...";
            }
            out.add("[" + i + "] " + signature);
        }
        return out;
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
            case DECLARE_ATTACKS:
            case DECLARE_ATTACK_TARGET:
                return "attack";
            case DECLARE_BLOCKS:
                return "block";
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
            if (candidate instanceof CombatCandidate) {
                CombatCandidate cc = (CombatCandidate) candidate;
                if (cc.isDone()) {
                    return toVocabId(base + ":DONE");
                }
                String creatureName = cc.creature.getName().replace(' ', '_');
                if (cc.context instanceof UUID) {
                    // Phase 2: attack target
                    MageObject target = game.getObject((UUID) cc.context);
                    String targetName = target != null ? target.getName().replace(' ', '_') : "PLAYER";
                    base += ":" + creatureName + ":ATTACKS:" + targetName;
                } else if (cc.context instanceof Permanent) {
                    // Block candidate: blocker vs attacker
                    String attackerName = ((Permanent) cc.context).getName().replace(' ', '_');
                    base += ":" + creatureName + ":BLOCKS:" + attackerName;
                } else {
                    // Phase 1: attacker candidate
                    base += ":" + creatureName;
                }
                return toVocabId(base);
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
            } else if (candidate instanceof mage.abilities.Mode) {
                mage.abilities.Mode mode = (mage.abilities.Mode) candidate;
                String effectText = mode.getEffects().getText(mode);
                // Use first 40 chars of effect text as a stable key (avoids UUID churn across games)
                String key = effectText.length() > 40 ? effectText.substring(0, 40) : effectText;
                base += ":MODE:" + key.replace(' ', '_');
            } else if (candidate instanceof Integer && actionType == StateSequenceBuilder.ActionType.ANNOUNCE_X) {
                base += ":" + candidate;
            } else if (candidate != null) {
                base += ":" + candidate.getClass().getSimpleName();
            }
            return toVocabId(base);
        } catch (Exception e) {
            return 0;
        }
    }

    private float[] computeCandidateFeatures(StateSequenceBuilder.ActionType actionType, Game game, Ability source, Object candidate, int dim, StateSequenceBuilder.SequenceOutput baseState) {
        float[] f = new float[dim];
        try {
            // Generic context
            f[0] = actionType.ordinal() / 16.0f;

            // Game context features (ALL candidates, including Pass)
            f[18] = getOpponentLife(game) / 20.0f;
            f[19] = this.getLife() / 20.0f;
            f[20] = countOpponentCreatures(game) / 10.0f;
            f[21] = countOwnCreatures(game) / 10.0f;
            f[22] = (getHand() != null ? getHand().size() : 0) / 7.0f;
            f[23] = countUntappedLands(game) / 10.0f;

            if (candidate instanceof PassAbility) {
                f[1] = 1.0f; // is_pass
                return f; // Now has game context features
            }

            // chooseUse candidates: Boolean.TRUE = yes, Boolean.FALSE = no
            if (candidate instanceof Boolean) {
                boolean isYes = (Boolean) candidate;
                if (!isYes) {
                    f[1] = 1.0f; // is_pass (declining = effectively passing on the option)
                }
                f[26] = isYes ? 1.0f : 0.0f; // is_choose_use_yes
                // Encode source ability context if available
                if (source != null) {
                    MageObject srcObj = game.getObject(source.getSourceId());
                    if (srcObj instanceof Permanent) {
                        Permanent p = (Permanent) srcObj;
                        f[2] = p.isCreature() ? 1.0f : 0.0f;
                        f[3] = p.isLand() ? 1.0f : 0.0f;
                        f[4] = p.isTapped() ? 1.0f : 0.0f;
                        f[5] = p.getPower().getValue() / 10.0f;
                        f[6] = p.getToughness().getValue() / 10.0f;
                        f[32] = p.isArtifact() ? 1.0f : 0.0f;
                    }
                    f[9] = source.getManaCostsToPay().manaValue() / 10.0f;
                }
                return f;
            }

            // Combat candidates (attack/block decisions)
            if (candidate instanceof CombatCandidate) {
                CombatCandidate cc = (CombatCandidate) candidate;
                if (cc.isDone()) {
                    f[1] = 1.0f; // DONE sentinel = is_pass
                    return f;
                }
                Permanent creature = cc.creature;
                // Creature making the decision
                f[2] = creature.isCreature() ? 1.0f : 0.0f;
                f[3] = creature.isLand() ? 1.0f : 0.0f;
                f[4] = creature.isTapped() ? 1.0f : 0.0f;
                f[5] = creature.getPower().getValue() / 10.0f;
                f[6] = creature.getToughness().getValue() / 10.0f;
                if (baseState != null) {
                    Integer tokenIdx = baseState.uuidToTokenIndex.get(creature.getId());
                    f[27] = (tokenIdx != null) ? 1.0f : 0.0f;
                    f[28] = (tokenIdx != null) ? tokenIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                }
                f[29] = Math.min(creature.getDamage(), 10) / 10.0f;
                f[30] = creature.hasSummoningSickness() ? 1.0f : 0.0f;
                f[31] = creature.isAttacking() ? 1.0f : 0.0f;
                f[32] = creature.isArtifact() ? 1.0f : 0.0f;
                f[33] = creature.getAbilities(game).containsKey(FlyingAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[34] = creature.getAbilities(game).containsKey(HasteAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[35] = creature.getAbilities(game).containsKey(DeathtouchAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[36] = creature.getAbilities(game).containsKey(LifelinkAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[37] = creature.getAbilities(game).containsKey(FirstStrikeAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[38] = creature.getAbilities(game).containsKey(TrampleAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[39] = creature.getAbilities(game).containsKey(HexproofAbility.getInstance().getId()) ? 1.0f : 0.0f;
                f[42] = Math.min(creature.getCounters(game).getCount(CounterType.P1P1), 10) / 10.0f;

                if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS) {
                    // Phase 1: deciding whether this creature attacks; no target context yet
                    f[43] = countOpponentCreatures(game) / 10.0f; // blocking risk
                    f[44] = creature.getAbilities(game).containsKey(VigilanceAbility.getInstance().getId()) ? 1.0f : 0.0f;
                } else if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET) {
                    // Phase 2: choosing which defender this attacker attacks
                    Object ctx = cc.context;
                    if (ctx instanceof UUID) {
                        UUID defId = (UUID) ctx;
                        Player defPlayer = game.getPlayer(defId);
                        if (defPlayer != null) {
                            f[10] = 1.0f; // is_player
                            f[12] = defPlayer.getLife() / 20.0f;
                        } else {
                            Permanent defPerm = game.getPermanent(defId);
                            if (defPerm != null) {
                                f[13] = 1.0f; // is_permanent
                                f[14] = defPerm.isCreature() ? 1.0f : 0.0f;
                                f[16] = defPerm.getPower().getValue() / 10.0f;
                                f[17] = defPerm.getToughness().getValue() / 10.0f;
                                f[46] = defPerm.isPlaneswalker() ? 1.0f : 0.0f;
                                if (baseState != null) {
                                    Integer tIdx = baseState.uuidToTokenIndex.get(defId);
                                    // store target token index in separate slots to not clobber creature's
                                    f[47] = (tIdx != null) ? tIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                                }
                            }
                        }
                    }
                } else if (actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS) {
                    // Block candidate: this creature is a potential blocker; context is the attacker
                    if (cc.context instanceof Permanent) {
                        Permanent attacker = (Permanent) cc.context;
                        int attackerPow = attacker.getPower().getValue();
                        int attackerTou = attacker.getToughness().getValue();
                        int blockerPow = creature.getPower().getValue();
                        int blockerTou = creature.getToughness().getValue();
                        // Attacker stats in target slots
                        f[13] = 1.0f; // is_permanent (attacker)
                        f[14] = 1.0f; // attacker is always a creature
                        f[16] = attackerPow / 10.0f;
                        f[17] = attackerTou / 10.0f;
                        f[25] = 1.0f; // attacker is opponent-controlled
                        if (baseState != null) {
                            Integer tIdx = baseState.uuidToTokenIndex.get(attacker.getId());
                            f[47] = (tIdx != null) ? tIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                        }
                        // Combat outcome signals
                        f[44] = (attackerPow >= blockerTou) ? 1.0f : 0.0f; // will blocker die
                        f[45] = (blockerPow >= attackerTou) ? 1.0f : 0.0f; // will attacker die
                        // Attacker keywords (affects blocking value)
                        f[43] = attacker.getAbilities(game).containsKey(TrampleAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        // Blocker has reach (can block flyers)
                        f[46] = creature.getAbilities(game).containsKey(ReachAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    }
                }
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
                    // Token index linkage: link this ability to its source permanent's state token
                    if (baseState != null) {
                        Integer tokenIdx = baseState.uuidToTokenIndex.get(p.getId());
                        f[27] = (tokenIdx != null) ? 1.0f : 0.0f;
                        f[28] = (tokenIdx != null) ? tokenIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                    }
                    // Source keyword flags [32-38]
                    f[32] = p.isArtifact() ? 1.0f : 0.0f;
                    f[33] = p.getAbilities(game).containsKey(FlyingAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    f[34] = p.getAbilities(game).containsKey(HasteAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    f[35] = p.getAbilities(game).containsKey(DeathtouchAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    f[36] = p.getAbilities(game).containsKey(LifelinkAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    f[37] = p.getAbilities(game).containsKey(FirstStrikeAbility.getInstance().getId()) ? 1.0f : 0.0f;
                    f[38] = p.getAbilities(game).containsKey(TrampleAbility.getInstance().getId()) ? 1.0f : 0.0f;
                }
                // rough target count
                f[7] = ab.getTargets() != null ? ab.getTargets().size() / 5.0f : 0.0f;
                f[8] = ab.isUsesStack() ? 1.0f : 0.0f;
                // Mana cost
                f[9] = ab.getManaCostsToPay().manaValue() / 10.0f;
                // Is spell from hand (distinguishes from activated abilities on battlefield)
                f[24] = (ab instanceof SpellAbility) ? 1.0f : 0.0f;
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
                        // Is opponent controlled (for targeting decisions)
                        f[25] = perm.isControlledBy(this.getId()) ? 0.0f : 1.0f;
                        // Token index linkage: link this target to its state sequence position.
                        // The cross-attention can then read the full state token (damage, sickness, etc.)
                        // and discover relationships (e.g., a buff spell on the stack targeting it).
                        if (baseState != null) {
                            Integer tokenIdx = baseState.uuidToTokenIndex.get(tid);
                            f[27] = (tokenIdx != null) ? 1.0f : 0.0f;
                            f[28] = (tokenIdx != null) ? tokenIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                        }
                        // Battlefield state features for this specific permanent instance
                        f[29] = Math.min(perm.getDamage(), 10) / 10.0f;
                        f[30] = perm.hasSummoningSickness() ? 1.0f : 0.0f;
                        f[31] = perm.isAttacking() ? 1.0f : 0.0f;
                        // Target keyword/extra flags [32-42]
                        f[32] = perm.isArtifact() ? 1.0f : 0.0f;
                        f[33] = perm.getAbilities(game).containsKey(FlyingAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[34] = perm.getAbilities(game).containsKey(HasteAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[35] = perm.getAbilities(game).containsKey(DeathtouchAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[36] = perm.getAbilities(game).containsKey(LifelinkAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[37] = perm.getAbilities(game).containsKey(FirstStrikeAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[38] = perm.getAbilities(game).containsKey(TrampleAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[39] = perm.getAbilities(game).containsKey(HexproofAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[40] = perm.getAbilities(game).containsKey(IndestructibleAbility.getInstance().getId()) ? 1.0f : 0.0f;
                        f[41] = (perm instanceof PermanentToken) ? 1.0f : 0.0f;
                        f[42] = Math.min(perm.getCounters(game).getCount(CounterType.P1P1), 10) / 10.0f;
                    }
                }
                // Include source-ability context for target selection so the model can
                // distinguish "target yourself" combo enablers from hostile effects.
                if (actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS && source != null) {
                    f[43] = source.isUsesStack() ? 1.0f : 0.0f;
                    f[44] = source.getManaCostsToPay().manaValue() / 10.0f;
                    f[45] = source.getTargets() != null ? Math.min(source.getTargets().size(), 5) / 5.0f : 0.0f;
                    f[46] = (source instanceof SpellAbility) ? 1.0f : 0.0f;
                    if (baseState != null && source.getSourceId() != null) {
                        Integer sourceTokenIdx = baseState.uuidToTokenIndex.get(source.getSourceId());
                        f[47] = sourceTokenIdx != null ? sourceTokenIdx / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                    }
                }
            }

            // Mode candidate features (CHOOSE_MODE)
            if (candidate instanceof mage.abilities.Mode) {
                mage.abilities.Mode mode = (mage.abilities.Mode) candidate;
                // f[0] is overwritten externally with mode ordinal/count by chooseMode(); no duplicate here
                f[11] = mode.getTargets() != null && !mode.getTargets().isEmpty() ? 1.0f : 0.0f;
                f[12] = mode.getTargets() != null ? Math.min(mode.getTargets().size(), 5) / 5.0f : 0.0f;
                f[13] = mode.getCost() != null ? 1.0f : 0.0f;
                // Source context
                if (source != null) {
                    f[9] = source.getManaCostsToPay().manaValue() / 10.0f;
                    MageObject srcObj = game.getObject(source.getSourceId());
                    if (srcObj instanceof Permanent) {
                        Permanent p = (Permanent) srcObj;
                        f[27] = baseState != null && baseState.uuidToTokenIndex.get(p.getId()) != null ? 1.0f : 0.0f;
                        f[28] = baseState != null && baseState.uuidToTokenIndex.get(p.getId()) != null
                                ? baseState.uuidToTokenIndex.get(p.getId()) / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                    }
                }
            }

            // Integer candidate features (ANNOUNCE_X)
            if (candidate instanceof Integer && actionType == StateSequenceBuilder.ActionType.ANNOUNCE_X) {
                int xVal = (Integer) candidate;
                // Absolute magnitude (capped)
                f[10] = Math.min(xVal, 20) / 20.0f;
                // Available mana context
                try {
                    mage.abilities.mana.ManaOptions available = getManaAvailable(game);
                    int totalAvailable = available.isEmpty() ? 0 : available.stream().mapToInt(m -> m.count()).max().orElse(0);
                    f[13] = Math.min(totalAvailable, 20) / 20.0f;
                    int baseCost = source != null ? source.getManaCostsToPay().manaValue() : 0;
                    int budget = Math.max(0, totalAvailable - baseCost);
                    // How much of the budget this X value uses
                    f[14] = budget > 0 ? Math.min(xVal, budget) / (float) budget : 0.0f;
                } catch (Exception ignored) {}
                // Source context
                if (source != null) {
                    f[9] = source.getManaCostsToPay().manaValue() / 10.0f;
                    MageObject srcObj = game.getObject(source.getSourceId());
                    if (srcObj instanceof Permanent) {
                        Permanent p = (Permanent) srcObj;
                        f[27] = baseState != null && baseState.uuidToTokenIndex.get(p.getId()) != null ? 1.0f : 0.0f;
                        f[28] = baseState != null && baseState.uuidToTokenIndex.get(p.getId()) != null
                                ? baseState.uuidToTokenIndex.get(p.getId()) / (float) StateSequenceBuilder.MAX_LEN : 0.0f;
                    }
                }
            }
        } catch (Exception e) {
            // leave zeros
        }
        return f;
    }

    private double computeStepReward(StateSequenceBuilder.ActionType actionType, Object candidate, Game game) {
        // Default: no heuristic shaping; rely on returns/GAE from game outcomes.
        if (!USE_HEURISTIC_STEP_REWARDS) {
            return 0.0;
        }

        if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS) {
            return ATTACK_REWARD;
        }

        if (actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS) {
            return BLOCK_REWARD;
        }

        if (actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS && candidate instanceof java.util.UUID && game != null) {
            java.util.UUID tid = (java.util.UUID) candidate;
            Player targetPlayer = game.getPlayer(tid);
            if (targetPlayer != null) {
                if (targetPlayer.getId().equals(playerId)) {
                    return TARGET_SELF_PENALTY;
                } else {
                    return TARGET_OPP_REWARD;
                }
            }
            Permanent targetPerm = game.getPermanent(tid);
            if (targetPerm != null) {
                if (targetPerm.isControlledBy(playerId)) {
                    return TARGET_SELF_PENALTY;
                } else {
                    return TARGET_OPP_REWARD;
                }
            }
        }

        if (actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL || candidate == null) {
            return 0.0;
        }

        if (candidate instanceof PlayLandAbility) {
            return LAND_PLAY_REWARD;
        }

        if (candidate instanceof SpellAbility) {
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
        
        // Trace entry
        if (target != null) {
            trace(String.format("chooseTarget ENTRY: outcome=%s, targetName=%s, targetClass=%s, min=%d, max=%d",
                outcome, target.getTargetName(), target.getClass().getSimpleName(),
                target.getMinNumberOfTargets(), target.getMaxNumberOfTargets()));
        } else {
            trace("chooseTarget ENTRY: target=null");
        }
        
        // Special case: Choose starting player - always choose yourself (no need for model)
        if (target != null && "starting player".equalsIgnoreCase(target.getTargetName())) {
            target.addTarget(this.getId(), source, game);
            trace("chooseTarget EXIT: starting player, result=true");
            return true;
        }
        
        // Special case: London mulligan card selection
        // TODO: I don't think this check is sufficient to say its a mulligan
        if (source == null && outcome == Outcome.Discard && target instanceof mage.target.common.TargetCardInHand) {
            trace("chooseTarget: london mulligan delegation");
            boolean result = chooseLondonMulliganCards(target, game);
            trace("chooseTarget EXIT: london mulligan, result=" + result);
            return result;
        }
        // During mana payment, avoid tapping key mana producers when a target
        // choice is required (e.g., Saruli Caretaker) by using a deterministic
        // target heuristic instead of policy sampling.
        if (USE_ENGINE_CHOICES && inPlayManaContext() && outcome == Outcome.Tap
                && source instanceof ManaAbility
                && target != null
                && target.getMinNumberOfTargets() == 1
                && target.getMaxNumberOfTargets() == 1) {
            UUID abilityControllerId = playerId;
            if (target.getTargetController() != null && target.getAbilityController() != null) {
                abilityControllerId = target.getAbilityController();
            }
            java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID ctrlId = abilityControllerId;
            possible.removeIf(id -> id == null || !target.canTarget(ctrlId, id, source, game));
            // For mana abilities that already tap their source as a cost, avoid
            // selecting the same permanent as a tap-target (can become illegal
            // after costs are paid and strands mana payment).
            if (source != null && hasTapSourceCost(source)) {
                UUID sourceId = source.getSourceId();
                if (sourceId != null) {
                    possible.removeIf(id -> sourceId.equals(id));
                }
            }
            if (!possible.isEmpty()) {
                UUID best = null;
                int bestPenalty = Integer.MAX_VALUE;
                for (UUID id : possible) {
                    int pen = manaProducerPenalty(id, game);
                    if (best == null || pen < bestPenalty) {
                        best = id;
                        bestPenalty = pen;
                    }
                }
                boolean allCandidatesAreManaProducers = true;
                for (UUID id : possible) {
                    Permanent cand = game.getPermanent(id);
                    if (estimatePermanentManaOutput(cand, game) <= 0) {
                        allCandidatesAreManaProducers = false;
                        break;
                    }
                }
                if (allCandidatesAreManaProducers) {
                    // Reject dominated targeted-mana lines (e.g., Saruli tapping Battlement to make
                    // one mana while spending two producers) when the tapped producers can already
                    // satisfy the currently required color.
                    boolean sourceHasTapTargetCost = hasTapTargetCost(source);
                    int sourceOutput = estimateAbilityManaOutput(source, game);
                    boolean dominatedByTargets = sourceOutput > 0;
                    for (UUID id : possible) {
                        Permanent cand = game.getPermanent(id);
                        int targetOutput = estimatePermanentManaOutput(cand, game);
                        if (targetOutput < sourceOutput) {
                            dominatedByTargets = false;
                            break;
                        }
                    }
                    String requiredColor = requiredManaColorFromUnpaid(currentUnpaidManaText.get());
                    boolean targetsCanCoverRequiredColor = true;
                    if (requiredColor != null) {
                        targetsCanCoverRequiredColor = false;
                        for (UUID id : possible) {
                            Permanent cand = game.getPermanent(id);
                            if (permanentCanProduceColor(cand, requiredColor, game)) {
                                targetsCanCoverRequiredColor = true;
                                break;
                            }
                        }
                    }

                    // Allow lines where this source is being used for a required color
                    // that at least one legal target cannot produce. Example: Saruli
                    // makes {B} while tapping a green-only producer as target.
                    boolean sourceProvidesRequiredColor = requiredColor != null
                            && abilityCanProduceColor(source, requiredColor, game);
                    boolean hasOffColorTarget = false;
                    if (requiredColor != null) {
                        for (UUID id : possible) {
                            Permanent cand = game.getPermanent(id);
                            if (!permanentCanProduceColor(cand, requiredColor, game)) {
                                hasOffColorTarget = true;
                                break;
                            }
                        }
                    }

                    if (sourceHasTapTargetCost
                            && dominatedByTargets
                            && targetsCanCoverRequiredColor
                            && !(sourceProvidesRequiredColor && hasOffColorTarget)) {
                        trace("chooseTarget: playMana rejected dominated tap-target mana line "
                                + "(sourceOutput=" + sourceOutput
                                + ", requiredColor=" + requiredColor + ")");
                        trace("chooseTarget EXIT: playMana heuristic, result=false");
                        return false;
                    }
                    // Keep legal lines open otherwise and pick the least costly target.
                    trace("chooseTarget: playMana heuristic all targets are mana producers; picking least costly");
                }
                if (best != null) {
                    target.addTarget(best, source, game);
                    MageObject obj = game.getObject(best);
                    String pickName = obj != null ? obj.getName() : best.toString();
                    trace("chooseTarget: playMana heuristic picked=" + pickName + " penalty=" + bestPenalty);
                    trace("chooseTarget EXIT: playMana heuristic, result=true");
                    return true;
                }
            }
            trace("chooseTarget: playMana heuristic fallback to engine");
            boolean result = super.chooseTarget(outcome, target, source, game);
            trace("chooseTarget EXIT: playMana heuristic fallback, result=" + result);
            return result;
        }
        // For mana payment plumbing, engine target selection is more reliable than
        // policy-driven picks (prevents cast failures from stranding mana sources).
        if (USE_ENGINE_CHOICES && source instanceof ManaAbility) {
            trace("chooseTarget: mana ability target delegation to engine");
            boolean result = super.chooseTarget(outcome, target, source, game);
            trace("chooseTarget EXIT: mana ability delegation, result=" + result);
            return result;
        }
        // RL-only target selection. No engine fallback.
        UUID abilityControllerId = playerId;
        if (target.getTargetController() != null && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }

        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());
        int requiredTapPower = parseCrewOrSaddleRequiredPower(outcome, source, target);

        java.util.HashSet<UUID> chosen = new java.util.HashSet<>();
        int chosenCount = 0;
        while (chosenCount < maxTargets) {
            java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID ctrlId = abilityControllerId;
            possible.removeIf(id -> id == null || chosen.contains(id) || !target.canTarget(ctrlId, id, source, game));

            int chosenPower = requiredTapPower > 0 ? computeTotalPowerForTargets(chosen, game) : 0;
            boolean allowStop = chosenCount >= minTargets
                    && (requiredTapPower <= 0 || chosenPower >= requiredTapPower);
            if (allowStop) {
                possible.add(0, null); // STOP sentinel
            } else if (requiredTapPower > 0) {
                trace(String.format(
                        "chooseTarget: STOP disabled (crew/saddle), chosenPower=%d requiredPower=%d",
                        chosenPower, requiredTapPower));
            }

            // Trace possible targets (cap at 10 for readability)
            StringBuilder possibleStr = new StringBuilder();
            for (int i = 0; i < Math.min(10, possible.size()); i++) {
                UUID id = possible.get(i);
                if (id == null) {
                    possibleStr.append("STOP");
                } else {
                    MageObject obj = game.getObject(id);
                    possibleStr.append(obj != null ? obj.getName() : "unknown");
                    possibleStr.append("(").append(id.toString().substring(0, 8)).append(")");
                }
                if (i < Math.min(10, possible.size()) - 1) possibleStr.append(", ");
            }
            if (possible.size() > 10) possibleStr.append("...(+").append(possible.size() - 10).append(" more)");
            trace(String.format("chooseTarget iteration %d: possible count=%d [%s]", chosenCount, possible.size(), possibleStr));

            if (possible.isEmpty()) {
                trace("chooseTarget: no possible targets, breaking loop");
                break;
            }

            UUID picked = null;
            if (possible.size() == 1) {
                picked = possible.get(0);
                String pickName = picked == null ? "STOP" : (game.getObject(picked) != null ? game.getObject(picked).getName() : "unknown");
                trace("chooseTarget: single option, picked=" + pickName);
            } else {
                // Check if all non-null candidates have the same name (trivial decision)
                boolean allSameName = true;
                String firstName = null;
                for (UUID id : possible) {
                    if (id == null) continue; // skip STOP sentinel
                    MageObject obj = game.getObject(id);
                    String name = obj != null ? obj.getName() : null;
                    if (firstName == null) {
                        firstName = name;
                    } else if (!java.util.Objects.equals(firstName, name)) {
                        allSameName = false;
                        break;
                    }
                }

                if (allSameName && firstName != null) {
                    // All candidates are the same card - pick first non-null
                    trace("chooseTarget: all same name (" + firstName + "), picking first");
                    for (UUID id : possible) {
                        if (id != null) {
                            picked = id;
                            break;
                        }
                    }
                } else {
                    // Non-trivial decision - use model inference
                    trace("chooseTarget: calling RL model for target selection");
                    try {
                    StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);
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
                        candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source, cand, candFeatDim, baseState);
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
                    SinglePickResult pickResult = sampleSinglePick(actionProbs, candidateMask, candidateCount, game);
                    int chosenIdx = pickResult.chosenIdx;
                    picked = possible.get(chosenIdx);
                    String pickName = picked == null ? "STOP" : (game.getObject(picked) != null ? game.getObject(picked).getName() : "unknown");
                    trace(String.format("chooseTarget: RL model picked idx=%d, target=%s, behavior_prob=%.3f, mode=%s",
                            chosenIdx, pickName, pickResult.selectedBehaviorProb, pickResult.behavior.mode));

                    // Record training data for this target selection
                    if (trainingEnabled && !game.isSimulation()) {
                        int[] chosenIndices = new int[maxCandidates];
                        Arrays.fill(chosenIndices, -1);
                        chosenIndices[0] = chosenIdx;
                        float oldLogp = pickResult.oldLogp;
                        double stepReward = computeStepReward(StateSequenceBuilder.ActionType.SELECT_TARGETS, picked, game);
                        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                                baseState,
                                candidateCount,
                                candidateActionIds,
                                candidateFeatures,
                                candidateMask,
                                1, // chosenCount = 1 (single pick per iteration)
                                chosenIndices,
                                oldLogp,
                                valueScore,
                                StateSequenceBuilder.ActionType.SELECT_TARGETS,
                                stepReward
                        );
                        trainingBuffer.add(td);
                    }

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
                            String gameStateWithExploration = appendExplorationToState(
                                    formatGameState(game),
                                    pickResult.behavior,
                                    chosenIdx,
                                    pickResult.oldLogpSource
                            );
                            gameLogger.logDecision(
                                    this.getName(),
                                    activePlayerName,
                                    phase + " (TARGET_PICK " + chosenCount + " min=" + minTargets + " max=" + maxTargets + ")",
                                    turn,
                                    gameStateWithExploration,
                                    optionNames,
                                    Arrays.copyOf(pickResult.behavior.behaviorProbs, candidateCount),
                                    valueScore,
                                    chosenIdx,
                                    selectedName
                            );
                        }
                    } catch (Exception ignored) {
                    }
                    } catch (Exception e) {
                        // Deterministic internal fallback (no engine)
                        trace("chooseTarget: RL model exception, using fallback first target: " + e.getMessage());
                        picked = possible.get(0);
                        String pickName = picked == null ? "STOP" : (game.getObject(picked) != null ? game.getObject(picked).getName() : "unknown");
                        trace("chooseTarget: exception fallback picked=" + pickName);
                    }
                }
            }

            if (picked == null) { // STOP
                trace("chooseTarget: STOP selected, breaking loop");
                break;
            }
            String addedName = game.getObject(picked) != null ? game.getObject(picked).getName() : "unknown";
            trace("chooseTarget: adding target=" + addedName + ", chosenCount will be " + (chosenCount + 1));
            target.addTarget(picked, source, game);
            chosen.add(picked);
            chosenCount++;
        }

        // Ensure minimum targets if required (deterministic fill, no engine)
        if (chosenCount < minTargets) {
            trace("chooseTarget: filling to minTargets, current=" + chosenCount + ", min=" + minTargets);
        }
        while (chosenCount < minTargets) {
            java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID ctrlId = abilityControllerId;
            possible.removeIf(id -> id == null || chosen.contains(id) || !target.canTarget(ctrlId, id, source, game));
            if (possible.isEmpty()) {
                trace("chooseTarget: no targets available to fill minTargets, breaking");
                break;
            }
            UUID picked = possible.get(0);
            String pickedName = game.getObject(picked) != null ? game.getObject(picked).getName() : "unknown";
            trace("chooseTarget: deterministic fill, adding=" + pickedName);
            target.addTarget(picked, source, game);
            chosen.add(picked);
            chosenCount++;
        }

        // Crew/Saddle safety: if model choices didn't reach required total power,
        // deterministically add highest-power legal creatures until threshold is met.
        if (requiredTapPower > 0) {
            int chosenPower = computeTotalPowerForTargets(chosen, game);
            while (chosenPower < requiredTapPower && chosenCount < maxTargets) {
                java.util.List<UUID> possible = new java.util.ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
                final UUID ctrlId = abilityControllerId;
                possible.removeIf(id -> id == null || chosen.contains(id) || !target.canTarget(ctrlId, id, source, game));
                if (possible.isEmpty()) {
                    break;
                }
                UUID best = null;
                int bestPower = Integer.MIN_VALUE;
                for (UUID id : possible) {
                    int p = powerForTarget(id, game);
                    if (p > bestPower) {
                        bestPower = p;
                        best = id;
                    }
                }
                if (best == null) {
                    break;
                }
                String bestName = game.getObject(best) != null ? game.getObject(best).getName() : "unknown";
                trace(String.format(
                        "chooseTarget: crew/saddle deterministic fill add=%s power=%d chosenPower=%d requiredPower=%d",
                        bestName, Math.max(0, bestPower), chosenPower, requiredTapPower));
                target.addTarget(best, source, game);
                chosen.add(best);
                chosenCount++;
                chosenPower = computeTotalPowerForTargets(chosen, game);
            }
        }

        boolean result = chosenCount >= minTargets;
        trace(String.format("chooseTarget EXIT: chosenCount=%d, minTargets=%d, result=%s", chosenCount, minTargets, result));
        return result;
    }

    private int parseCrewOrSaddleRequiredPower(Outcome outcome, Ability source, Target target) {
        if (outcome != Outcome.Tap || source == null || target == null) {
            return 0;
        }
        String text = source.toString();
        if (text == null || (!text.contains("Crew") && !text.contains("Saddle"))) {
            return 0;
        }
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\\b(?:Crew|Saddle)\\s+(\\d+)\\b")
                .matcher(text);
        if (m.find()) {
            try {
                return Math.max(0, Integer.parseInt(m.group(1)));
            } catch (Exception ignored) {
                return 0;
            }
        }
        return 0;
    }

    private int computeTotalPowerForTargets(Set<UUID> ids, Game game) {
        int total = 0;
        for (UUID id : ids) {
            total += powerForTarget(id, game);
        }
        return total;
    }

    private int powerForTarget(UUID id, Game game) {
        if (id == null || game == null) {
            return 0;
        }
        try {
            Permanent p = game.getPermanent(id);
            if (p != null) {
                return Math.max(0, p.getPower().getValue());
            }
            MageObject o = game.getObject(id);
            if (o instanceof Card) {
                return Math.max(0, ((Card) o).getPower().getValue());
            }
        } catch (Exception ignored) {
        }
        return 0;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        // Route to chooseTarget which uses the RL model
        // This handles discard effects (Faithless Looting), sacrifice effects, etc.
        trace("choose(Target) ENTRY: delegating to chooseTarget");
        boolean result = chooseTarget(outcome, target, source, game);
        trace("choose(Target) EXIT: result=" + result);
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        // Trace-only override for card selection from a visible set (e.g., cost payments)
        String targetName = target != null ? target.getTargetName() : "null";
        int cardsCount = cards != null ? cards.size() : 0;
        trace(String.format("choose(Cards,TargetCard) ENTRY: outcome=%s, cardsCount=%d, targetName=%s",
            outcome, cardsCount, targetName));
        
        boolean result = super.choose(outcome, cards, target, source, game);
        
        trace("choose(Cards,TargetCard) EXIT: result=" + result);
        return result;
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        // Trace entry
        String targetName = target != null ? target.getTargetName() : "null";
        String filterName = (target != null && target.getFilter() != null) ? target.getFilter().getMessage() : "none";
        int cardsCount = cards != null ? cards.size() : 0;
        trace(String.format("chooseTarget(Cards) ENTRY: outcome=%s, cardsCount=%d, targetName=%s, filter=%s",
            outcome, cardsCount, targetName, filterName));
        
        // RL-only selection from a provided visible card set. No engine fallback.
        if (cards == null || target == null) {
            trace("chooseTarget(Cards) EXIT: null input, result=false");
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
        // Library/hand cards are stateless - don't dedupe battlefield (tapped/untapped, counters)
        // or graveyard (timestamp/order matters for recursion effects)
        boolean shouldDedupe = target instanceof mage.target.common.TargetCardInLibrary
                            || target instanceof mage.target.common.TargetCardInHand;

        // Build name-to-cards mapping for deduplication
        java.util.LinkedHashMap<String, List<Card>> cardsByName = new java.util.LinkedHashMap<>();
        for (Card card : filteredCards) {
            if (card != null) {
                String cardName = card.getName();
                cardsByName.computeIfAbsent(cardName, k -> new ArrayList<>()).add(card);
            }
        }

        // For deduped zones: model sees unique names. For others: model sees all cards.
        List<String> choiceNames;
        if (shouldDedupe) {
            choiceNames = new ArrayList<>(cardsByName.keySet());
        } else {
            // No deduplication - each card gets its own "name" (use card ID as unique key)
            choiceNames = new ArrayList<>();
            cardsByName.clear();
            for (Card c : filteredCards) {
                String uniqueKey = c.getId().toString();
                choiceNames.add(uniqueKey);
                cardsByName.put(uniqueKey, Arrays.asList(c));
            }
        }
        
        if (choiceNames.isEmpty()) {
            return false;
        }

        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());
        int maxPossibleSelections = 0;
        for (List<Card> copies : cardsByName.values()) {
            maxPossibleSelections += copies.size();
        }
        maxTargets = Math.min(maxTargets, maxPossibleSelections);
        minTargets = Math.min(minTargets, maxPossibleSelections);

        java.util.HashSet<UUID> chosen = new java.util.HashSet<>();
        int chosenCount = 0;
        while (chosenCount < maxTargets) {
            // Build remaining choices: names that still have available copies
            List<String> remainingNames = new ArrayList<>();
            for (String name : choiceNames) {
                List<Card> copies = cardsByName.get(name);
                boolean hasAvailable = false;
                for (Card c : copies) {
                    if (!chosen.contains(c.getId())) {
                        hasAvailable = true;
                        break;
                    }
                }
                if (hasAvailable) {
                    remainingNames.add(name);
                }
            }
            if (remainingNames.isEmpty()) {
                break;
            }

            boolean allowStop = chosenCount >= minTargets;
            if (allowStop) {
                remainingNames.add(0, null); // STOP sentinel
            }

            String pickedName = null;
            Card picked = null;
            if (remainingNames.size() == 1) {
                pickedName = remainingNames.get(0);
            } else {
                try {
                    StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);
                    final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                    final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

                    int candidateCount = Math.min(remainingNames.size(), maxCandidates);
                    int[] candidateActionIds = new int[maxCandidates];
                    float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
                    int[] candidateMask = new int[maxCandidates];
                    for (int i = 0; i < candidateCount; i++) {
                        candidateMask[i] = 1;
                        String name = remainingNames.get(i);
                        // Use first available card with this name for feature computation
                        Card representative = null;
                        if (name != null) {
                            List<Card> copies = cardsByName.get(name);
                            for (Card c : copies) {
                                if (!chosen.contains(c.getId())) {
                                    representative = c;
                                    break;
                                }
                            }
                        }
                        UUID cid = representative == null ? null : representative.getId();
                        candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.SELECT_CARD, game, source, cid);
                        candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.SELECT_CARD, game, source, cid, candFeatDim, baseState);
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
                    SinglePickResult pickResult = sampleSinglePick(actionProbs, candidateMask, candidateCount, game);
                    int chosenIdx = pickResult.chosenIdx;
                    pickedName = remainingNames.get(chosenIdx);

                    // Record training data for this card selection
                    if (trainingEnabled && !game.isSimulation()) {
                        int[] chosenIndices = new int[maxCandidates];
                        Arrays.fill(chosenIndices, -1);
                        chosenIndices[0] = chosenIdx;
                        float oldLogp = pickResult.oldLogp;
                        // Get representative card UUID for step reward computation
                        UUID pickedCardUUID = (pickedName != null && cardsByName.containsKey(pickedName) && !cardsByName.get(pickedName).isEmpty())
                                ? cardsByName.get(pickedName).get(0).getId()
                                : null;
                        double stepReward = computeStepReward(StateSequenceBuilder.ActionType.SELECT_CARD, pickedCardUUID, game);
                        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                                baseState,
                                candidateCount,
                                candidateActionIds,
                                candidateFeatures,
                                candidateMask,
                                1, // chosenCount = 1 (single pick per iteration)
                                chosenIndices,
                                oldLogp,
                                valueScore,
                                StateSequenceBuilder.ActionType.SELECT_CARD,
                                stepReward
                        );
                        trainingBuffer.add(td);
                    }

                    // Track decision count by head
                    decisionCountsByHead.put(StateSequenceBuilder.ActionType.SELECT_CARD, 
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.SELECT_CARD, 0) + 1);

                    // Gamelog: card-pick decision (shows deduplicated names)
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
                                String name = remainingNames.get(i);
                                optionNames.add(name == null ? "STOP" : name);
                            }
                            String gameStateWithExploration = appendExplorationToState(
                                    formatGameState(game),
                                    pickResult.behavior,
                                    chosenIdx,
                                    pickResult.oldLogpSource
                            );
                            gameLogger.logDecision(
                                    this.getName(),
                                    activePlayerName,
                                    phase + " (CARD_PICK " + chosenCount + " min=" + minTargets + " max=" + maxTargets + ")",
                                    turn,
                                    gameStateWithExploration,
                                    optionNames,
                                    Arrays.copyOf(pickResult.behavior.behaviorProbs, candidateCount),
                                    valueScore,
                                    chosenIdx,
                                    pickedName == null ? "STOP" : pickedName
                            );
                        }
                    } catch (Exception ignored) {
                    }
                } catch (Exception e) {
                    pickedName = remainingNames.get(0);
                }
            }

            // Convert picked name to actual card object
            if (pickedName != null) {
                List<Card> availableCopies = new ArrayList<>();
                for (Card c : cardsByName.get(pickedName)) {
                    if (!chosen.contains(c.getId())) {
                        availableCopies.add(c);
                    }
                }
                if (!availableCopies.isEmpty()) {
                    // Randomly select one copy from available copies
                    picked = availableCopies.get(new java.util.Random().nextInt(availableCopies.size()));
                }
            }

            if (picked == null) { // STOP or error
                break;
            }

            target.addTarget(picked.getId(), source, game);
            chosen.add(picked.getId());
            chosenCount++;
        }

        // Fallback: if we haven't met minimum targets, randomly pick remaining cards
        while (chosenCount < minTargets) {
            Card fallback = null;
            for (String name : choiceNames) {
                for (Card c : cardsByName.get(name)) {
                    if (!chosen.contains(c.getId())) {
                        fallback = c;
                        break;
                    }
                }
                if (fallback != null) {
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

        boolean result = chosenCount >= minTargets;
        trace(String.format("chooseTarget(Cards) EXIT: chosenCount=%d, minTargets=%d, result=%s",
            chosenCount, minTargets, result));
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        // Trace entry
        if (choice != null) {
            boolean isKeyChoice = choice.isKeyChoice();
            int choiceCount = isKeyChoice ? choice.getKeyChoices().size() : choice.getChoices().size();
            boolean isManaColor = choice.isManaColorChoice();
            String options = "";
            if (isKeyChoice && choiceCount > 0 && choiceCount <= 10) {
                options = ", keys=" + choice.getKeyChoices().keySet();
            } else if (!isKeyChoice && choiceCount > 0 && choiceCount <= 10) {
                options = ", choices=" + choice.getChoices();
            }
            trace(String.format("choose ENTRY: outcome=%s, isKey=%s, isManaColor=%s, count=%d%s", 
                outcome, isKeyChoice, isManaColor, choiceCount, options));
        } else {
            trace("choose ENTRY: outcome=" + outcome + ", choice=null");
        }

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

        // Mana color payment: delegate to base AI logic (not a strategic decision)
        if (outcome == Outcome.PutManaInPool && choice != null && choice.isManaColorChoice()) {
            if (USE_ENGINE_CHOICES && inPlayManaContext() && choice.getChoices() != null) {
                String preferred = preferredManaColorForUnpaid(currentUnpaidManaText.get(), choice.getChoices());
                if (preferred != null) {
                    choice.setChoice(preferred);
                    trace("choose: mana color forced by unpaid cost, chosen=" + preferred
                            + ", unpaid=" + currentUnpaidManaText.get());
                    return true;
                }
            }
            trace("choose: mana color delegation, calling super.choose()");
            boolean result = super.choose(outcome, choice, game);
            trace("choose EXIT: mana color delegation, result=" + result + ", chosen=" + (choice.isChosen() ? choice.getChoice() : "none"));
            return result;
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
                        trace("choose EXIT: alternative cost, forced choice key=" + forced + ", result=true");
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
                                trace("choose EXIT: alternative cost, validated choice key=" + key + ", result=true");
                                return true;
                            }
                        }
                    }
                }

                // Default: use parent class logic (random)
                trace("choose: alternative cost, calling super.choose() (random)");
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
                trace("choose EXIT: alternative cost, super result=" + result + ", chosenKey=" + (choice.isChosen() ? choice.getChoiceKey() : "none"));
                return result;
            }
        }

        // Handle regular choices (modal spells, etc.) with RL model
        if (choice == null || !choice.isRequired()) {
            trace("choose: not required, calling super.choose()");
            boolean result = super.choose(outcome, choice, game);
            trace("choose EXIT: not required, result=" + result);
            return result;
        }

        // Get available choices
        List<String> availableChoices;
        if (choice.isKeyChoice()) {
            availableChoices = new ArrayList<>(choice.getKeyChoices().keySet());
        } else {
            availableChoices = new ArrayList<>(choice.getChoices());
        }

        if (availableChoices.isEmpty()) {
            trace("choose EXIT: empty choices, result=false");
            return false;
        }

        if (availableChoices.size() == 1) {
            // Only one option - pick it
            if (choice.isKeyChoice()) {
                choice.setChoiceByKey(availableChoices.get(0));
            } else {
                choice.setChoice(availableChoices.get(0));
            }
            trace("choose EXIT: single option=" + availableChoices.get(0) + ", result=true");
            return true;
        }

        // Card-name effects can present the full card database (30k+ options).
        // RL truncation to MAX_CANDIDATES is both noisy and strategically unsound here,
        // so defer to base AI handling for these giant naming menus.
        String choiceMsg = choice.getMessage() != null ? choice.getMessage().toLowerCase() : "";
        boolean isCardNamePrompt = choiceMsg.contains("card name");
        if (isCardNamePrompt && availableChoices.size() > StateSequenceBuilder.TrainingData.MAX_CANDIDATES) {
            trace("choose: large card-name prompt, delegating to super.choose() to avoid truncation");
            return super.choose(outcome, choice, game);
        }

        // Use RL model to score the choices
        trace("choose: calling RL model, count=" + availableChoices.size());
        try {
            String choiceContext = "choiceClass=" + choice.getClass().getSimpleName()
                    + ", message=" + trunc(choice.getMessage(), 220)
                    + ", isKeyChoice=" + choice.isKeyChoice()
                    + ", isRequired=" + choice.isRequired()
                    + ", availableCount=" + availableChoices.size()
                    + ", preview=" + trunc(availableChoices.subList(0, Math.min(8, availableChoices.size())).toString(), 320);
            choiceExplosionContext.set(choiceContext);
            List<Integer> rankedIndices;
            try {
                rankedIndices = genericChoose(
                        availableChoices,
                        1, // maxTargets = 1 (pick one choice)
                        1, // minTargets = 1 (must pick)
                        StateSequenceBuilder.ActionType.SELECT_CHOICE,
                        game,
                        currentAbility
                );
            } finally {
                choiceExplosionContext.remove();
            }

            if (rankedIndices != null && !rankedIndices.isEmpty()) {
                int chosenIdx = rankedIndices.get(0);
                String chosenValue = availableChoices.get(chosenIdx);
                
                if (choice.isKeyChoice()) {
                    choice.setChoiceByKey(chosenValue);
                } else {
                    choice.setChoice(chosenValue);
                }

                if (ACTIVATION_DIAG) {
                    RLTrainer.threadLocalLogger.get().info(
                            "CHOOSE: RL model selected: " + chosenValue + " from " + availableChoices.size() + " options"
                    );
                }
                trace("choose EXIT: RL model, chosen=" + chosenValue + ", idx=" + chosenIdx + ", result=true");
                return true;
            }
        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn("Error in RL choice selection, using fallback: " + e.getMessage());
            trace("choose: RL model exception, falling back to super: " + e.getMessage());
        }

        // Fallback to parent if model fails
        trace("choose: calling super.choose() (fallback)");
        boolean result = super.choose(outcome, choice, game);
        if (result && choice.isChosen()) {
            RLTrainer.threadLocalLogger.get().debug(
                    "Player " + getName() + " chose (fallback): " + choice.getChoiceKey() + " -> " + choice.getChoice()
            );
        }
        trace("choose EXIT: fallback, result=" + result + ", chosen=" + (choice.isChosen() ? choice.getChoice() : "none"));
        return result;
    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        // Trace entry
        String targetName = target != null ? target.getTargetName() : "null";
        int min = target != null ? target.getMinNumberOfTargets() : 0;
        int max = target != null ? target.getMaxNumberOfTargets() : 0;
        trace(String.format("chooseTargetAmount ENTRY: outcome=%s, targetName=%s, min=%d, max=%d",
            outcome, targetName, min, max));
        
        boolean result = super.chooseTargetAmount(outcome, target, source, game);
        
        // Trace exit with chosen targets and amounts
        if (result && target != null && !target.getTargets().isEmpty()) {
            StringBuilder chosen = new StringBuilder();
            for (UUID targetId : target.getTargets()) {
                int amount = target.getTargetAmount(targetId);
                String tgtName = describeTargetWithOwner(targetId, game);
                chosen.append(tgtName).append("=").append(amount).append("; ");
                RLTrainer.threadLocalLogger.get().debug(
                        "Player " + getName() + " chose target amount: " + tgtName + " (" + targetId + "), amount: " + amount
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
            trace("chooseTargetAmount EXIT: result=true, targets=[" + chosen.toString() + "]");
        } else {
            trace("chooseTargetAmount EXIT: result=" + result);
        }
        
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
        trace("choose(Target,Map) ENTRY: delegating to super.choose() with options");
        boolean result = super.choose(outcome, target, source, game, options);
        if (result && !target.getTargets().isEmpty()) {
            StringBuilder targetsStr = new StringBuilder();
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                targetsStr.append(targetName).append(", ");
                RLTrainer.threadLocalLogger.get().debug(
                        "Player " + getName() + " chose target with options: " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
            trace("choose(Target,Map) EXIT: result=" + result + ", targets=[" + targetsStr + "]");
        } else {
            trace("choose(Target,Map) EXIT: result=" + result + ", no targets");
        }
        return result;
    }

    @Override
    public mage.abilities.Mode chooseMode(mage.abilities.Modes modes, Ability source, Game game) {
        trace("chooseMode ENTRY: available modes=" + (modes != null ? modes.size() : 0));

        if (modes == null || game == null) {
            mage.abilities.Mode fallback = super.chooseMode(modes, source, game);
            trace("chooseMode EXIT (null input): " + (fallback != null ? fallback.getId() : "null"));
            return fallback;
        }

        try {
            List<mage.abilities.Mode> availableModes = modes.getAvailableModes(source, game);
            if (availableModes.isEmpty()) {
                trace("chooseMode EXIT: no available modes");
                return null;
            }
            if (availableModes.size() == 1) {
                trace("chooseMode EXIT: only one mode, auto-select");
                return availableModes.get(0);
            }

            final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            int candidateCount = Math.min(availableModes.size(), maxCandidates);

            StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

            int[] candidateActionIds = new int[maxCandidates];
            float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
            int[] candidateMask = new int[maxCandidates];
            int legalModes = 0;

            for (int i = 0; i < candidateCount; i++) {
                boolean legal = isModeChoiceCurrentlyLegal(availableModes.get(i), source, game);
                candidateMask[i] = legal ? 1 : 0;
                if (legal) {
                    legalModes++;
                }
                candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.CHOOSE_MODE, game, source, availableModes.get(i));
                candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.CHOOSE_MODE, game, source, availableModes.get(i), candFeatDim, baseState);
                // inject mode index into the feature vector so the model knows ordinal position
                candidateFeatures[i][0] = i / (float) candidateCount;
            }
            if (legalModes <= 0) {
                trace("chooseMode: no legal mode candidates after masking, falling back to super");
                mage.abilities.Mode fallback = super.chooseMode(modes, source, game);
                trace("chooseMode EXIT (masked fallback): " + (fallback != null ? fallback.getId() : "null"));
                return fallback;
            }

            String headId = headForActionType(StateSequenceBuilder.ActionType.CHOOSE_MODE);
            mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                    baseState, candidateActionIds, candidateFeatures, candidateMask, policyKey, headId, 0, 1, 1);

            float[] actionProbs = prediction.policyScores;
            float valueScore = prediction.valueScores;
            SinglePickResult pickResult = sampleSinglePick(actionProbs, candidateMask, candidateCount, game);
            int appliedIdx = pickResult.chosenIdx;

            if (trainingEnabled && !game.isSimulation()) {
                int[] chosenIndices = new int[maxCandidates];
                Arrays.fill(chosenIndices, -1);
                chosenIndices[0] = appliedIdx;
                float oldLogp = pickResult.oldLogp;
                StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                        baseState, candidateCount, candidateActionIds, candidateFeatures, candidateMask,
                        1, chosenIndices, oldLogp, valueScore,
                        StateSequenceBuilder.ActionType.CHOOSE_MODE, 0.0);
                trainingBuffer.add(td);
                decisionCountsByHead.put(StateSequenceBuilder.ActionType.CHOOSE_MODE,
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.CHOOSE_MODE, 0) + 1);
            }

            mage.abilities.Mode chosen = availableModes.get(appliedIdx);
            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    int turn = game.getTurnNum();
                    String activePlayerName = game.getActivePlayerId() != null
                            ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                    String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                    List<String> modeNames = new ArrayList<>();
                    for (int i = 0; i < candidateCount; i++) {
                        modeNames.add("Mode" + i + ":" + availableModes.get(i).getEffects().getText(availableModes.get(i)).substring(0, Math.min(30, availableModes.get(i).getEffects().getText(availableModes.get(i)).length())));
                    }
                    String modeInfo = appendExplorationToState(
                            String.format("CHOOSE_MODE: chose mode %d of %d", appliedIdx, candidateCount),
                            pickResult.behavior,
                            appliedIdx,
                            pickResult.oldLogpSource
                    );
                    gameLogger.logDecision(this.getName(), activePlayerName, phase + " (CHOOSE_MODE)", turn,
                            modeInfo,
                            modeNames, Arrays.copyOf(pickResult.behavior.behaviorProbs, candidateCount), valueScore, appliedIdx,
                            modeNames.get(appliedIdx));
                }
            } catch (Exception ignored) {}

            trace("chooseMode EXIT: selected mode index=" + appliedIdx);
            return chosen;

        } catch (Exception e) {
            trace("chooseMode: model exception, falling back to super: " + e.getMessage());
            mage.abilities.Mode fallback = super.chooseMode(modes, source, game);
            trace("chooseMode EXIT (fallback): " + (fallback != null ? fallback.getId() : "null"));
            return fallback;
        }
    }

    private boolean isModeChoiceCurrentlyLegal(mage.abilities.Mode mode, Ability source, Game game) {
        if (mode == null || source == null || game == null) {
            return true;
        }
        try {
            boolean targetsOk = mode.getTargets().isEmpty()
                    || mode.getTargets().canChoose(source.getControllerId(), source, game);
            boolean costOk = mode.getCost() == null
                    || mode.getCost().canPay(source, source, playerId, game);
            return targetsOk && costOk;
        } catch (Exception e) {
            trace("isModeChoiceCurrentlyLegal: exception during legality check: " + e.getMessage());
            return false;
        }
    }

    @Override
    public int announceX(int min, int max, String message, Game game, Ability source, boolean isManaPay) {
        trace(String.format("announceX ENTRY: min=%d, max=%d, isManaPay=%s, msg=%s", min, max, isManaPay, message));

        if (game == null) {
            int fallback = super.announceX(min, max, message, game, source, isManaPay);
            trace("announceX EXIT (no game): " + fallback);
            return fallback;
        }

        try {
            int realMin = min;
            int realMax = max == Integer.MAX_VALUE ? Math.max(realMin, 20) : Math.min(max, 20);

            // Tighten the upper bound to what we can actually pay (avoids offering unpayable X values)
            if (isManaPay && source != null) {
                try {
                    mage.abilities.mana.ManaOptions available = getManaAvailable(game);
                    int totalAvailable = 0;
                    if (!available.isEmpty()) {
                        for (Mana mana : available) {
                            // Ignore conditional mana options that cannot be used for this spell/ability.
                            // Example: Mishra's Workshop mana for non-artifact spells.
                            if (mana instanceof ConditionalMana
                                    && !((ConditionalMana) mana).apply(source, game, getId(), source.getManaCosts())) {
                                continue;
                            }
                            totalAvailable = Math.max(totalAvailable, mana.count());
                        }
                    }
                    int baseCost = source.getManaCostsToPay().manaValue();
                    int xInstances = source.getManaCostsToPay().stream()
                            .filter(c -> c instanceof VariableManaCost)
                            .mapToInt(c -> ((VariableManaCost) c).getXInstancesCount())
                            .sum();
                    if (xInstances <= 0) {
                        xInstances = 1;
                    }

                    int affordableX = (totalAvailable - baseCost) / xInstances;
                    if (affordableX < 0) {
                        affordableX = 0;
                    }
                    realMax = Math.min(realMax, affordableX);
                } catch (Exception ignored) {}
            }

            if (realMax < realMin) realMax = realMin;

            if (realMin == realMax) {
                trace("announceX EXIT: only one option, returning " + realMin);
                return realMin;
            }

            final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

            // Build one candidate per integer value [realMin .. realMax], capped at MAX_CANDIDATES
            int range = realMax - realMin + 1;
            int candidateCount = Math.min(range, maxCandidates);
            // If too many values, sample evenly spaced integers
            List<Integer> xValues = new ArrayList<>(candidateCount);
            if (range <= maxCandidates) {
                for (int x = realMin; x <= realMax; x++) xValues.add(x);
            } else {
                for (int i = 0; i < maxCandidates; i++) {
                    xValues.add(realMin + (int) Math.round(i * (range - 1.0) / (maxCandidates - 1)));
                }
            }
            candidateCount = xValues.size();

            StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

            int[] candidateActionIds = new int[maxCandidates];
            float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
            int[] candidateMask = new int[maxCandidates];

            for (int i = 0; i < candidateCount; i++) {
                candidateMask[i] = 1;
                candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.ANNOUNCE_X, game, source, xValues.get(i));
                candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.ANNOUNCE_X, game, source, xValues.get(i), candFeatDim, baseState);
            }

            String headId = headForActionType(StateSequenceBuilder.ActionType.ANNOUNCE_X);
            mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                    baseState, candidateActionIds, candidateFeatures, candidateMask, policyKey, headId, 0, 1, 1);

            float[] actionProbs = prediction.policyScores;
            float valueScore = prediction.valueScores;
            SinglePickResult pickResult = sampleSinglePick(actionProbs, candidateMask, candidateCount, game);
            int chosenIdx = pickResult.chosenIdx;

            int chosenX = xValues.get(chosenIdx);

            if (trainingEnabled && !game.isSimulation()) {
                int[] chosenIndices = new int[maxCandidates];
                Arrays.fill(chosenIndices, -1);
                chosenIndices[0] = chosenIdx;
                float oldLogp = pickResult.oldLogp;
                StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                        baseState, candidateCount, candidateActionIds, candidateFeatures, candidateMask,
                        1, chosenIndices, oldLogp, valueScore,
                        StateSequenceBuilder.ActionType.ANNOUNCE_X, 0.0);
                trainingBuffer.add(td);
                decisionCountsByHead.put(StateSequenceBuilder.ActionType.ANNOUNCE_X,
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.ANNOUNCE_X, 0) + 1);
            }

            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    int turn = game.getTurnNum();
                    String activePlayerName = game.getActivePlayerId() != null
                            ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                    String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                    List<String> xNames = new ArrayList<>();
                    for (Integer x : xValues) xNames.add("X=" + x);
                    String xDecisionState = appendExplorationToState(
                            String.format("ANNOUNCE_X: msg=\"%s\" isManaPay=%s chose X=%d (range [%d..%d])",
                                    message, isManaPay, chosenX, realMin, realMax),
                            pickResult.behavior,
                            chosenIdx,
                            pickResult.oldLogpSource
                    );
                    gameLogger.logDecision(this.getName(), activePlayerName, phase + " (ANNOUNCE_X)", turn,
                            xDecisionState,
                            xNames, Arrays.copyOf(pickResult.behavior.behaviorProbs, candidateCount), valueScore, chosenIdx,
                            "X=" + chosenX);
                }
            } catch (Exception ignored) {}

            trace(String.format("announceX EXIT: chose X=%d (behavior_prob=%.3f, mode=%s)",
                    chosenX, pickResult.selectedBehaviorProb, pickResult.behavior.mode));
            return chosenX;

        } catch (Exception e) {
            trace("announceX: model exception, falling back to super: " + e.getMessage());
            int fallback = super.announceX(min, max, message, game, source, isManaPay);
            trace("announceX EXIT (fallback): " + fallback);
            return fallback;
        }
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, Ability source, Game game) {
        // Delegate to 7-param version (base class does the same, but we want RL logic here too)
        return chooseUse(outcome, message, null, "Yes", "No", source, game);
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage, String trueText, String falseText, Ability source, Game game) {
        trace(String.format("chooseUse ENTRY: outcome=%s, msg=%s, trueText=%s, falseText=%s",
                outcome, message, trueText, falseText));

        // Safety fallback: need active game for model inference
        if (game == null) {
            boolean fallback = super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
            trace("chooseUse EXIT (no game): " + fallback);
            return fallback;
        }

        // NOTE: We intentionally do NOT skip AIDontUseIt outcomes.
        // The engine tags kicker/buyback/replicate/squad as AIDontUseIt to protect
        // the heuristic AI from paying costs it can't reason about.
        // Our RL model CAN learn these decisions from the reward signal.

        // Mana feasibility gate: for optional additional costs (kicker, buyback, etc.),
        // the engine's canPay() only checks the additional cost in isolation, not the
        // combined total. Block the model from choosing an unaffordable option.
        if (outcome == Outcome.AIDontUseIt && source instanceof SpellAbility) {
            try {
                Mana additionalCost = parseManaCostFromMessage(message);
                if (additionalCost.count() > 0) {
                    Mana baseCost = source.getManaCostsToPay().getMana();
                    Mana totalCost = baseCost.copy();
                    totalCost.add(additionalCost);
                    ManaOptions available = getManaAvailable(game);
                    if (!available.enough(totalCost)) {
                        trace(String.format("chooseUse EXIT (mana infeasible): base=%s additional=%s total=%s",
                                baseCost, additionalCost, totalCost));
                        return false;
                    }
                }
            } catch (Exception e) {
                // Parsing failed - let model decide rather than silently blocking
                trace("chooseUse mana feasibility check failed: " + e.getMessage());
            }
        }

        // Phyrexian payment safety gate: if this prompt is the phyrexian "pay 2 life
        // instead of {X}" choice and we cannot produce the colored mana option,
        // force YES when legal to avoid repeated activation failures.
        if (outcome == Outcome.LoseLife && message != null
                && message.toLowerCase().contains("phyrexian cost")) {
            try {
                Mana manaOption = parseManaCostFromMessage(message); // usually parses "{U}" from the prompt
                ManaOptions available = getManaAvailable(game);
                boolean canPayWithMana = manaOption.count() == 0 || available.enough(manaOption);
                if (!canPayWithMana) {
                    int life = this.getLife();
                    if (life >= 2) {
                        trace("chooseUse EXIT (forced phyrexian YES): cannot pay mana option, life>=2");
                        return true;
                    } else {
                        trace("chooseUse EXIT (forced phyrexian NO): cannot pay mana option and life<2");
                        return false;
                    }
                }
            } catch (Exception e) {
                trace("chooseUse phyrexian feasibility check failed: " + e.getMessage());
            }
        }

        // Generic optional mana-payment gate for chooseUse prompts like:
        // "Pay {R}{R} to copy the spell?"
        // If the prompt requires an explicit mana payment and we can't pay it now,
        // force NO to avoid invalid activations.
        if (message != null && message.startsWith("Pay ")
                && message.contains("{") && !message.toLowerCase().contains("instead of")) {
            try {
                Mana optionalCost = parseManaCostFromMessage(message);
                if (optionalCost.count() > 0) {
                    ManaOptions available = getManaAvailable(game);
                    if (!available.enough(optionalCost)) {
                        trace(String.format(
                                "chooseUse EXIT (forced NO: optional mana infeasible): msg=%s required=%s available=%s",
                                message, optionalCost, available));
                        return false;
                    }
                }
            } catch (Exception e) {
                trace("chooseUse optional-cost feasibility check failed: " + e.getMessage());
            }
        }

        try {
            final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

            // Build 2-candidate decision: index 0 = Yes, index 1 = No
            List<Boolean> candidates = Arrays.asList(Boolean.TRUE, Boolean.FALSE);
            int candidateCount = 2;

            StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

            int[] candidateActionIds = new int[maxCandidates];
            float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
            int[] candidateMask = new int[maxCandidates];

            for (int i = 0; i < candidateCount; i++) {
                candidateMask[i] = 1;
                candidateActionIds[i] = toVocabId(StateSequenceBuilder.ActionType.CHOOSE_USE.name() + "_" + (i == 0 ? "YES" : "NO"));
                candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.CHOOSE_USE, game, source, candidates.get(i), candFeatDim, baseState);
            }

            String headId = headForActionType(StateSequenceBuilder.ActionType.CHOOSE_USE);
            mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                    baseState,
                    candidateActionIds,
                    candidateFeatures,
                    candidateMask,
                    policyKey,
                    headId,
                    0, 1, 1
            );

            float[] actionProbs = prediction.policyScores;
            float valueScore = prediction.valueScores;
            SinglePickResult pickResult = sampleSinglePick(actionProbs, candidateMask, candidateCount, game);
            int chosenIdx = pickResult.chosenIdx;
            boolean useIt = (chosenIdx == 0);

            // Record training data
            if (trainingEnabled && !game.isSimulation()) {
                int[] chosenIndices = new int[maxCandidates];
                Arrays.fill(chosenIndices, -1);
                chosenIndices[0] = chosenIdx;
                float oldLogp = pickResult.oldLogp;
                StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                        baseState,
                        candidateCount,
                        candidateActionIds,
                        candidateFeatures,
                        candidateMask,
                        1,
                        chosenIndices,
                        oldLogp,
                        valueScore,
                        StateSequenceBuilder.ActionType.CHOOSE_USE,
                        0.0
                );
                trainingBuffer.add(td);
                decisionCountsByHead.put(StateSequenceBuilder.ActionType.CHOOSE_USE,
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.CHOOSE_USE, 0) + 1);
            }

            // Game log
            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    int turn = game.getTurnNum();
                    if (turn != lastLoggedTurn) {
                        String activePlayerName = game.getActivePlayerId() != null
                                ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                        gameLogger.logTurnStart(turn, activePlayerName, formatGameState(game));
                        lastLoggedTurn = turn;
                    }
                    String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                    String activePlayerName = game.getActivePlayerId() != null
                            ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                    String chooseUseState = appendExplorationToState(
                            String.format("CHOOSE_USE: msg=\"%s\" outcome=%s decision=%s scores=[%.2f, %.2f]",
                                    message, outcome, useIt ? "YES" : "NO",
                                    pickResult.behavior.behaviorProbs[0], pickResult.behavior.behaviorProbs[1]),
                            pickResult.behavior,
                            chosenIdx,
                            pickResult.oldLogpSource
                    );
                    gameLogger.logDecision(
                            this.getName(),
                            activePlayerName,
                            phase + " (CHOOSE_USE)",
                            turn,
                            chooseUseState,
                            Arrays.asList(trueText != null ? trueText : "Yes", falseText != null ? falseText : "No"),
                            Arrays.copyOf(pickResult.behavior.behaviorProbs, candidateCount),
                            valueScore,
                            chosenIdx,
                            useIt ? (trueText != null ? trueText : "Yes") : (falseText != null ? falseText : "No")
                    );
                }
            } catch (Exception ignored) {
            }

            trace(String.format("chooseUse EXIT: decision=%s behavior_scores=[%.3f, %.3f] mode=%s",
                    useIt ? "YES" : "NO",
                    pickResult.behavior.behaviorProbs[0],
                    pickResult.behavior.behaviorProbs[1],
                    pickResult.behavior.mode));
            return useIt;

        } catch (Exception e) {
            // Fallback to heuristic if model fails
            trace("chooseUse: model exception, falling back to super: " + e.getMessage());
            return super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
        }
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
    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        if (game.isSimulation()) {
            return;
        }
        try {
            // Build possible attackers
            List<Permanent> possibleAttackers = new ArrayList<>();
            for (Permanent perm : game.getBattlefield().getAllActivePermanents(attackingPlayerId)) {
                if (perm.isCreature() && perm.canAttack(null, game)) {
                    possibleAttackers.add(perm);
                }
            }
            if (possibleAttackers.isEmpty()) {
                return;
            }

            final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

            // Phase 1: Multi-select which creatures attack.
            // Candidates = [attacker0, attacker1, ..., DONE]
            int numAttackers = Math.min(possibleAttackers.size(), maxCandidates - 1);
            List<CombatCandidate> phase1Candidates = new ArrayList<>();
            for (int i = 0; i < numAttackers; i++) {
                phase1Candidates.add(new CombatCandidate(possibleAttackers.get(i), null));
            }
            phase1Candidates.add(new CombatCandidate(null, null)); // DONE sentinel
            int doneIdx = phase1Candidates.size() - 1;

            int candidateCount = phase1Candidates.size();
            int[] candidateActionIds = new int[maxCandidates];
            float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
            int[] candidateMask = new int[maxCandidates];
            for (int i = 0; i < candidateCount; i++) {
                candidateMask[i] = 1;
                candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.DECLARE_ATTACKS, game, null, phase1Candidates.get(i));
                candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.DECLARE_ATTACKS, game, null, phase1Candidates.get(i), candFeatDim, baseState);
            }

            String headId = headForActionType(StateSequenceBuilder.ActionType.DECLARE_ATTACKS);
            mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                    baseState, candidateActionIds, candidateFeatures, candidateMask, policyKey, headId, 0, 1, candidateCount);

            float[] actionProbs = prediction.policyScores;
            float valueScore = prediction.valueScores;
            SequentialPickResult attackPickResult = sampleSequentialWithoutReplacement(
                    actionProbs,
                    candidateMask,
                    candidateCount,
                    candidateCount,
                    game
            );
            List<Integer> selectedIndices = attackPickResult.selectedIndices;
            float oldLogpTotal = attackPickResult.oldLogpTotal;
            List<Permanent> selectedAttackers = new ArrayList<>();
            for (int pickIdx : selectedIndices) {
                if (pickIdx == doneIdx) {
                    break; // DONE picked, stop
                }
                selectedAttackers.add(phase1Candidates.get(pickIdx).creature);
            }

            // Record Phase 1 TrainingData
            if (trainingEnabled && !game.isSimulation()) {
                int[] chosenIndices = new int[maxCandidates];
                Arrays.fill(chosenIndices, -1);
                int chosenCount = Math.min(selectedIndices.size(), maxCandidates);
                for (int i = 0; i < chosenCount; i++) chosenIndices[i] = selectedIndices.get(i);
                StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                        baseState, candidateCount, candidateActionIds, candidateFeatures, candidateMask,
                        chosenCount, chosenIndices, oldLogpTotal, valueScore,
                        StateSequenceBuilder.ActionType.DECLARE_ATTACKS, 0.0);
                trainingBuffer.add(td);
                decisionCountsByHead.put(StateSequenceBuilder.ActionType.DECLARE_ATTACKS,
                        decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.DECLARE_ATTACKS, 0) + 1);
            }

            if (selectedAttackers.isEmpty()) {
                return;
            }

            // Determine attack targets
            List<UUID> defenders = new ArrayList<>(game.getCombat().getDefenders());
            Map<UUID, UUID> attackerToDefender = new HashMap<>();

            if (defenders.size() == 1) {
                // Only one defender: auto-assign all
                UUID singleDefender = defenders.get(0);
                for (Permanent attacker : selectedAttackers) {
                    attackerToDefender.put(attacker.getId(), singleDefender);
                }
            } else if (defenders.size() > 1) {
                // Phase 2: for each selected attacker, choose which defender to attack
                for (Permanent attacker : selectedAttackers) {
                    // Build Phase 2 candidates: one CombatCandidate per defender
                    List<CombatCandidate> phase2Candidates = new ArrayList<>();
                    for (UUID defId : defenders) {
                        if (attacker.canAttack(defId, game)) {
                            phase2Candidates.add(new CombatCandidate(attacker, defId));
                        }
                    }
                    if (phase2Candidates.isEmpty()) continue;
                    if (phase2Candidates.size() == 1) {
                        CombatCandidate cc = phase2Candidates.get(0);
                        attackerToDefender.put(attacker.getId(), (UUID) cc.context);
                        continue;
                    }

                    int p2Count = Math.min(phase2Candidates.size(), maxCandidates);
                    int[] p2Ids = new int[maxCandidates];
                    float[][] p2Feats = new float[maxCandidates][candFeatDim];
                    int[] p2Mask = new int[maxCandidates];
                    for (int i = 0; i < p2Count; i++) {
                        p2Mask[i] = 1;
                        p2Ids[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET, game, null, phase2Candidates.get(i));
                        p2Feats[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET, game, null, phase2Candidates.get(i), candFeatDim, baseState);
                    }

                    String p2HeadId = headForActionType(StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET);
                    mage.player.ai.rl.PythonMLBatchManager.PredictionResult p2Pred = model.scoreCandidates(
                            baseState, p2Ids, p2Feats, p2Mask, policyKey, p2HeadId, 0, 1, 1);

                    float[] p2Probs = p2Pred.policyScores;
                    float p2Value = p2Pred.valueScores;
                    SinglePickResult p2Pick = sampleSinglePick(p2Probs, p2Mask, p2Count, game);
                    int p2PickIdx = p2Pick.chosenIdx;
                    float p2LogP = p2Pick.oldLogp;
                    UUID chosenDefId = (UUID) phase2Candidates.get(p2PickIdx).context;
                    attackerToDefender.put(attacker.getId(), chosenDefId);

                    // Record Phase 2 TrainingData
                    if (trainingEnabled && !game.isSimulation()) {
                        int[] p2Chosen = new int[maxCandidates];
                        Arrays.fill(p2Chosen, -1);
                        p2Chosen[0] = p2PickIdx;
                        StateSequenceBuilder.TrainingData td2 = new StateSequenceBuilder.TrainingData(
                                baseState, p2Count, p2Ids, p2Feats, p2Mask,
                                1, p2Chosen, p2LogP, p2Value,
                                StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET, 0.0);
                        trainingBuffer.add(td2);
                        decisionCountsByHead.put(StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET,
                                decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.DECLARE_ATTACK_TARGET, 0) + 1);
                    }
                }
            }

            // Declare attackers
            for (Permanent attacker : selectedAttackers) {
                UUID defenderId = attackerToDefender.get(attacker.getId());
                if (defenderId != null && attacker.canAttack(defenderId, game)) {
                    this.declareAttacker(attacker.getId(), defenderId, game, false);
                }
            }

            // Game log
            try {
                GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    int turn = game.getTurnNum();
                    if (turn != lastLoggedTurn) {
                        String activeName = game.getActivePlayerId() != null
                                ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                        gameLogger.logTurnStart(turn, activeName, formatGameState(game));
                        lastLoggedTurn = turn;
                    }
                    String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                    String activeName = game.getActivePlayerId() != null
                            ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                    List<String> attackerNames = selectedAttackers.stream().map(Permanent::getName).collect(Collectors.toList());
                    gameLogger.logDecision(
                            this.getName(), activeName, phase + " (DECLARE_ATTACKS)", turn,
                            appendExplorationToState(
                                    String.format("DECLARE_ATTACKS: selected=%s from %d possible",
                                            attackerNames, possibleAttackers.size()),
                                    attackPickResult.behavior,
                                    selectedIndices.isEmpty() ? doneIdx : selectedIndices.get(0),
                                    attackPickResult.oldLogpSource
                            ),
                            phase1Candidates.stream().map(cc -> cc.isDone() ? "DONE" : cc.creature.getName()).collect(Collectors.toList()),
                            Arrays.copyOf(attackPickResult.behavior.behaviorProbs, candidateCount),
                            valueScore, selectedIndices.isEmpty() ? doneIdx : selectedIndices.get(0),
                            attackerNames.toString()
                    );
                }
            } catch (Exception ignored) {}

        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn("selectAttackers: model exception, no attacks declared: " + e.getMessage());
        }
    }

    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        if (game.isSimulation()) {
            return;
        }
        try {
            List<Permanent> attackers = getAttackers(game);
            if (attackers == null || attackers.isEmpty()) {
                return;
            }
            List<Permanent> availableBlockers = new ArrayList<>(super.getAvailableBlockers(game));
            availableBlockers = filterOutNonblocking(game, attackers, availableBlockers);
            if (availableBlockers.isEmpty()) {
                return;
            }
            attackers = filterOutUnblockable(game, attackers, availableBlockers);
            if (attackers.isEmpty()) {
                return;
            }

            // Sort attackers by power descending so biggest threats are handled first
            attackers.sort((a, b) -> Integer.compare(b.getPower().getValue(), a.getPower().getValue()));

            final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            StateSequenceBuilder.SequenceOutput baseState = getOrBuildBaseState(game);

            boolean anyBlockerDeclared = false;

            for (Permanent attacker : attackers) {
                // Build blocker candidates for this attacker: only blockers that can block it
                List<Permanent> eligibleBlockers = new ArrayList<>();
                for (Permanent blocker : availableBlockers) {
                    if (blocker.canBlock(attacker.getId(), game)) {
                        eligibleBlockers.add(blocker);
                    }
                }
                if (eligibleBlockers.isEmpty()) continue;

                // Candidates: [blocker0, blocker1, ..., DONE]
                int numBlockers = Math.min(eligibleBlockers.size(), maxCandidates - 1);
                List<CombatCandidate> blockCandidates = new ArrayList<>();
                for (int i = 0; i < numBlockers; i++) {
                    blockCandidates.add(new CombatCandidate(eligibleBlockers.get(i), attacker));
                }
                blockCandidates.add(new CombatCandidate(null, attacker)); // DONE
                int doneIdx = blockCandidates.size() - 1;

                int candidateCount = blockCandidates.size();
                int[] candidateActionIds = new int[maxCandidates];
                float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
                int[] candidateMask = new int[maxCandidates];
                for (int i = 0; i < candidateCount; i++) {
                    candidateMask[i] = 1;
                    candidateActionIds[i] = computeCandidateActionId(StateSequenceBuilder.ActionType.DECLARE_BLOCKS, game, null, blockCandidates.get(i));
                    candidateFeatures[i] = computeCandidateFeatures(StateSequenceBuilder.ActionType.DECLARE_BLOCKS, game, null, blockCandidates.get(i), candFeatDim, baseState);
                }

                String headId = headForActionType(StateSequenceBuilder.ActionType.DECLARE_BLOCKS);
                mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction = model.scoreCandidates(
                        baseState, candidateActionIds, candidateFeatures, candidateMask, policyKey, headId, 0, 1, candidateCount);

                float[] actionProbs = prediction.policyScores;
                float valueScore = prediction.valueScores;
                SequentialPickResult blockPickResult = sampleSequentialWithoutReplacement(
                        actionProbs,
                        candidateMask,
                        candidateCount,
                        candidateCount,
                        game
                );
                List<Integer> selectedIndices = blockPickResult.selectedIndices;
                float oldLogpTotal = blockPickResult.oldLogpTotal;
                List<Permanent> selectedBlockers = new ArrayList<>();
                for (int pickIdx : selectedIndices) {
                    if (pickIdx == doneIdx) {
                        break;
                    }
                    selectedBlockers.add(blockCandidates.get(pickIdx).creature);
                }

                // Record TrainingData for this attacker's block decision
                if (trainingEnabled && !game.isSimulation()) {
                    int[] chosenIndices = new int[maxCandidates];
                    Arrays.fill(chosenIndices, -1);
                    int chosenCount = Math.min(selectedIndices.size(), maxCandidates);
                    for (int i = 0; i < chosenCount; i++) chosenIndices[i] = selectedIndices.get(i);
                    StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                            baseState, candidateCount, candidateActionIds, candidateFeatures, candidateMask,
                            chosenCount, chosenIndices, oldLogpTotal, valueScore,
                            StateSequenceBuilder.ActionType.DECLARE_BLOCKS, 0.0);
                    trainingBuffer.add(td);
                    decisionCountsByHead.put(StateSequenceBuilder.ActionType.DECLARE_BLOCKS,
                            decisionCountsByHead.getOrDefault(StateSequenceBuilder.ActionType.DECLARE_BLOCKS, 0) + 1);
                }

                // Declare blockers
                for (Permanent blocker : selectedBlockers) {
                    this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                    availableBlockers.remove(blocker); // remove from pool so it can't block again
                    anyBlockerDeclared = true;
                }

                // Game log
                try {
                    GameLogger gameLogger = RLTrainer.threadLocalGameLogger.get();
                    if (gameLogger != null && gameLogger.isEnabled()) {
                        int turn = game.getTurnNum();
                        String activeName = game.getActivePlayerId() != null
                                ? game.getPlayer(game.getActivePlayerId()).getName() : "Unknown";
                        String phase = game.getStep() != null ? game.getStep().getType().toString() : "Unknown";
                        List<String> blockerNames = selectedBlockers.stream().map(Permanent::getName).collect(Collectors.toList());
                        gameLogger.logDecision(
                                this.getName(), activeName, phase + " (DECLARE_BLOCKS)", turn,
                                appendExplorationToState(
                                        String.format("DECLARE_BLOCKS: %s blocks %s (%d blockers selected)",
                                                blockerNames, attacker.getName(), selectedBlockers.size()),
                                        blockPickResult.behavior,
                                        selectedIndices.isEmpty() ? doneIdx : selectedIndices.get(0),
                                        blockPickResult.oldLogpSource
                                ),
                                blockCandidates.stream().map(cc -> cc.isDone() ? "DONE" : cc.creature.getName()).collect(Collectors.toList()),
                                Arrays.copyOf(blockPickResult.behavior.behaviorProbs, candidateCount),
                                valueScore, selectedIndices.isEmpty() ? doneIdx : selectedIndices.get(0),
                                blockerNames.toString()
                        );
                    }
                } catch (Exception ignored) {}
            }

            if (anyBlockerDeclared) {
                game.getPlayers().resetPassed();
            }

        } catch (Exception e) {
            RLTrainer.threadLocalLogger.get().warn("selectBlockers: model exception, no blockers declared: " + e.getMessage());
        }
    }
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
                GameLogger gameLogger = resolveGameLogger();
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
            try {
                GameLogger gameLogger = resolveGameLogger();
                if (gameLogger != null && gameLogger.isEnabled()) {
                    String cards = "";
                    try {
                        List<Card> handCards = new ArrayList<>(getHand().getCards(game));
                        cards = handCards.stream().map(Card::getName).collect(Collectors.joining("; "));
                    } catch (Exception ignored) {
                        cards = "";
                    }
                    gameLogger.log(String.format(
                            "MULLIGAN_DECISION_FALLBACK: player=%s mulligansTaken=%d handSize=%d lands=%d decision=%s reason=%s hand=[%s]",
                            getName(),
                            mulligansTaken,
                            handSizeNow,
                            landCount,
                            shouldMulligan ? "MULLIGAN" : "KEEP",
                            trunc(e.getClass().getSimpleName() + ": " + e.getMessage(), 220),
                            trunc(cards, 400)
                    ));
                }
            } catch (Exception ignored) {
                // don't fail mulligan fallback for logging
            }
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
                GameLogger gameLogger = resolveGameLogger();
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
     * Get opponent's life total for candidate features.
     */
    private int getOpponentLife(Game game) {
        for (Player p : game.getPlayers().values()) {
            if (p != null && !p.getId().equals(playerId)) {
                return p.getLife();
            }
        }
        return 20; // Default if opponent not found
    }

    /**
     * Count creatures opponent controls.
     */
    private int countOpponentCreatures(Game game) {
        int count = 0;
        for (Player p : game.getPlayers().values()) {
            if (p != null && !p.getId().equals(playerId)) {
                for (Permanent perm : game.getBattlefield().getAllActivePermanents(p.getId())) {
                    if (perm.isCreature(game)) {
                        count++;
                    }
                }
                break;
            }
        }
        return count;
    }

    /**
     * Count creatures this player controls.
     */
    private int countOwnCreatures(Game game) {
        int count = 0;
        for (Permanent p : game.getBattlefield().getAllActivePermanents(playerId)) {
            if (p.isCreature(game)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Count untapped lands this player controls.
     */
    private int countUntappedLands(Game game) {
        int count = 0;
        for (Permanent p : game.getBattlefield().getAllActivePermanents(playerId)) {
            if (p.isLand(game) && !p.isTapped()) {
                count++;
            }
        }
        return count;
    }

    /**
     * Set the current episode number for logging purposes.
     */
    public void setCurrentEpisode(int episodeNum) {
        this.currentEpisode = episodeNum;
        this.lastLoggedTurn = -1; // Reset turn tracking for new game
        this.lastExplorationTurn = Integer.MIN_VALUE;
        this.turnForceUniform = false;
        this.turnActionEps = 0.0;
        this.turnRandomEps = 0.0;
        this.turnExplorationMode = "policy";
        if (RL_BASE_SEED >= 0L) {
            long seed = RL_BASE_SEED;
            seed = seed * 31L + episodeNum;
            String n = getName();
            seed = seed * 31L + (n == null ? 0L : n.hashCode());
            stochasticRng.setSeed(seed);
        }
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

            // Enable activation tracing to capture all callbacks during activation
            activationTrace.get().clear();
            traceEnabled.set(true);
            
            // Pre-activation diagnostic trace
            ActivatedAbility.ActivationStatus preStatus = freshAbility.canActivate(this.getId(), game);
            MageObject sourceObj = game.getObject(freshAbility.getSourceId());
            String sourceName = sourceObj != null ? sourceObj.getName() : "unknown";
            String sourceZone = sourceObj != null ? String.valueOf(game.getState().getZone(sourceObj.getId())) : "unknown";
            trace(String.format("ACT: Attempting activation of '%s' (type=%s, source=%s, zone=%s, canActivate=%s)",
                freshAbility.getRule(),
                freshAbility.getAbilityType(),
                sourceName,
                sourceZone,
                (preStatus != null && preStatus.canActivate())));

            boolean activationResult;
            Exception activationException = null;
            try {
                activationResult = super.activateAbility(freshAbility, game);
                trace("ACT: activateAbility returned " + activationResult);
            } catch (Exception e) {
                trace("ACT: activateAbility threw " + e.getClass().getName() + ": " + e.getMessage());
                activationResult = false;
                activationException = e;
                RLTrainer.threadLocalLogger.get().error(
                        "ACTIVATION THREW EXCEPTION: " + e.getClass().getName() + ": " + e.getMessage());
                e.printStackTrace();
            } finally {
                // Disable tracing after activation completes (success or failure)
                traceEnabled.set(false);
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
                writeActivationFailureToFile(freshAbility, game, activationException, new ArrayList<>(activationTrace.get()));
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
     * Adds a trace entry to the activation trace buffer if tracing is enabled.
     * Used to capture the sequence of player callbacks during activateAbility().
     */
    private void trace(String msg) {
        if (Boolean.TRUE.equals(traceEnabled.get())) {
            activationTrace.get().add(msg);
        }
    }

    /**
     * Parses mana symbols from a chooseUse prompt message (e.g. "Pay Kicker {R} ?")
     * and returns their total as a Mana object. Returns empty Mana if no symbols found
     * or if the cost is non-mana (e.g. "sacrifice a creature").
     */
    private Mana parseManaCostFromMessage(String message) {
        if (message == null) return new Mana();
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("\\{([^}]+)\\}").matcher(message);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            sb.append('{').append(m.group(1)).append('}');
        }
        if (sb.length() == 0) return new Mana();
        try {
            return new ManaCostsImpl<>(sb.toString()).getMana();
        } catch (Exception e) {
            return new Mana();
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
     * Writes detailed activation failure information to a dedicated log file for offline investigation.
     * This always runs (no flag needed) and captures full game state plus diagnostic details.
     * 
     * @param trace The activation trace buffer capturing all callbacks during the failed activation
     */
    private void writeActivationFailureToFile(ActivatedAbility ability, Game game, Exception activationException, List<String> trace) {
        if (ability == null || game == null) {
            return;
        }
        
        try {
            Path logPath = Paths.get(RLLogPaths.ACTIVATION_FAILURES_LOG_PATH);
            Files.createDirectories(logPath.getParent());
            
            StringBuilder log = new StringBuilder();
            String timestamp = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            
            // Header
            log.append("\n================================================================================\n");
            log.append("ACTIVATION FAILURE: ").append(timestamp).append("\n");
            log.append("Episode: ").append(currentEpisode).append("\n");
            log.append("================================================================================\n\n");
            
            // Failed ability details
            MageObject sourceObj = game.getObject(ability.getSourceId());
            String sourceName = sourceObj != null ? sourceObj.getName() : "unknown";
            String sourceZone = sourceObj != null ? String.valueOf(game.getState().getZone(sourceObj.getId())) : "unknown";
            
            log.append("FAILED ABILITY:\n");
            log.append("  Name: ").append(ability.toString()).append("\n");
            log.append("  Type: ").append(ability.getAbilityType()).append("\n");
            log.append("  Source: ").append(sourceName).append(" (").append(sourceZone).append(")\n");
            log.append("  Costs: ").append(ability.getManaCostsToPay()).append("\n");
            log.append("  Uses stack: ").append(ability.isUsesStack()).append("\n\n");
            
            // Game state
            log.append("GAME STATE:\n");
            log.append("  Turn: ").append(game.getTurnNum()).append("\n");
            log.append("  Phase: ").append(game.getPhase() != null ? game.getPhase().getType() : "unknown").append("\n");
            log.append("  Step: ").append(game.getStep() != null ? game.getStep().getType() : "unknown").append("\n");
            log.append("  Active player: ").append(game.getActivePlayerId() != null ? game.getPlayer(game.getActivePlayerId()).getName() : "unknown").append("\n");
            log.append("  Priority player: ").append(game.getPriorityPlayerId() != null ? game.getPlayer(game.getPriorityPlayerId()).getName() : "unknown").append("\n\n");
            
            // Stack
            log.append("STACK: ").append(game.getStack().size()).append(" objects\n");
            if (!game.getStack().isEmpty()) {
                game.getStack().forEach(stackObject -> 
                    log.append("  - ").append(stackObject.getName()).append("\n"));
            }
            log.append("\n");
            
            // Both players' board states
            log.append("BOARD STATE:\n\n");
            
            // RL Player
            Player rlPlayer = game.getPlayer(this.getId());
            if (rlPlayer != null) {
                log.append("[").append(rlPlayer.getName()).append("] (RL Player)\n");
                log.append("  Life: ").append(rlPlayer.getLife()).append("\n");
                
                String handCards = rlPlayer.getHand().getCards(game).stream()
                        .map(card -> card.getName())
                        .collect(Collectors.joining("; "));
                log.append("  Hand: [").append(handCards).append("]\n");
                
                String permanents = game.getBattlefield().getAllPermanents().stream()
                        .filter(p -> p.isOwnedBy(rlPlayer.getId()))
                        .map(p -> p.getName()
                                + (p.isTapped() ? ",tapped" : "")
                                + (p.isAttacking() ? ",attacking" : "")
                                + (p.getBlocking() > 0 ? ",blocking" : ""))
                        .collect(Collectors.joining("; "));
                log.append("  Permanents: [").append(permanents).append("]\n\n");
            }
            
            // Opponent
            for (UUID opponentId : game.getOpponents(this.getId())) {
                Player opponent = game.getPlayer(opponentId);
                if (opponent != null) {
                    log.append("[").append(opponent.getName()).append("] (Opponent)\n");
                    log.append("  Life: ").append(opponent.getLife()).append("\n");
                    
                    String oppHandCards = opponent.getHand().getCards(game).stream()
                            .map(card -> card.getName())
                            .collect(Collectors.joining("; "));
                    log.append("  Hand: [").append(oppHandCards).append("]\n");
                    
                    String oppPermanents = game.getBattlefield().getAllPermanents().stream()
                            .filter(p -> p.isOwnedBy(opponent.getId()))
                            .map(p -> p.getName()
                                    + (p.isTapped() ? ",tapped" : "")
                                    + (p.isAttacking() ? ",attacking" : "")
                                    + (p.getBlocking() > 0 ? ",blocking" : ""))
                            .collect(Collectors.joining("; "));
                    log.append("  Permanents: [").append(oppPermanents).append("]\n\n");
                }
            }
            
            // Diagnostics
            log.append("DIAGNOSTICS:\n");
            
            ActivatedAbility.ActivationStatus status = ability.canActivate(this.getId(), game);
            log.append("  canActivate(): ").append(status != null && status.canActivate()).append("\n");
            log.append("  canChooseTarget(): ").append(ability.canChooseTarget(game, this.getId())).append("\n");
            log.append("  canPlayLand(): ").append(!(ability instanceof PlayLandAbility) || this.canPlayLand()).append("\n");
            log.append("  approvingObjects: ").append(status != null ? status.getApprovingObjects().size() : 0).append("\n\n");
            
            log.append("  Available mana: ").append(getManaAvailable(game)).append("\n");
            log.append("  Ability costs: ").append(ability.getManaCostsToPay()).append("\n\n");
            
            // Untapped lands
            String untappedLands = game.getBattlefield().getAllActivePermanents(this.getId()).stream()
                    .filter(p -> p.isLand(game) && !p.isTapped())
                    .map(p -> p.getName())
                    .collect(Collectors.joining(", "));
            log.append("  Untapped lands: ").append(untappedLands).append("\n\n");
            
            // Source permanent details
            if (sourceObj instanceof mage.game.permanent.Permanent) {
                mage.game.permanent.Permanent sourcePerm = (mage.game.permanent.Permanent) sourceObj;
                log.append("  Source permanent tapped: ").append(sourcePerm.isTapped()).append("\n");
                
                boolean hasTapCost = ability.getCosts().stream()
                        .anyMatch(cost -> cost instanceof mage.abilities.costs.common.TapSourceCost);
                log.append("  Ability has tap cost: ").append(hasTapCost).append("\n\n");
            }
            
            // Re-check post-failure
            ActivatedAbility.ActivationStatus postFailStatus = ability.canActivate(this.getId(), game);
            log.append("  POST-FAIL canActivate(): ").append(postFailStatus != null && postFailStatus.canActivate()).append("\n");
            
            // State leak detection
            log.append("  STATE LEAK DETECTED: ").append(lastActivationHadStateLeak).append("\n");
            if (lastActivationHadStateLeak) {
                log.append("    (A safety bookmark was restored to clean up objects left on stack)\n");
            }
            
            // Exception details
            if (activationException != null) {
                log.append("\nEXCEPTION:\n");
                log.append("  Type: ").append(activationException.getClass().getName()).append("\n");
                log.append("  Message: ").append(activationException.getMessage()).append("\n");
                
                // Stack trace (first 5 lines)
                StackTraceElement[] stackTrace = activationException.getStackTrace();
                if (stackTrace.length > 0) {
                    log.append("  Stack trace (top 5):\n");
                    for (int i = 0; i < Math.min(5, stackTrace.length); i++) {
                        log.append("    ").append(stackTrace[i].toString()).append("\n");
                    }
                }
            }
            
            // Activation trace - the exact sequence of callbacks during activateAbility()
            if (trace != null && !trace.isEmpty()) {
                log.append("\nACTIVATION TRACE (").append(trace.size()).append(" callbacks):\n");
                for (int i = 0; i < trace.size(); i++) {
                    log.append("  [").append(i + 1).append("] ").append(trace.get(i)).append("\n");
                }
            } else {
                log.append("\nACTIVATION TRACE: (empty or not captured)\n");
            }
            
            log.append("\n");
            
            // Write to file
            Files.write(logPath, log.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                    
        } catch (Exception e) {
            // Don't let logging errors crash the game
            RLTrainer.threadLocalLogger.get().warn("Failed to write activation failure to file: " + e.getMessage());
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
        List<ActivatedAbility> flattenedOptions = getPlayableForCurrentState(game);
        // (PassAbility will be added later after duplicate removal)

        // Filter by testing activation in a simulation (like ComputerPlayer6 does)
        // For abilities with alternative costs, test ALL alternatives and track which ones work
        List<ActivatedAbility> validOptions = new ArrayList<>();
        int filteredCount = 0;
        validAlternativeCosts.clear(); // Clear previous tracking
        refreshAlternativeCostValidationCache(game);

        for (ActivatedAbility ability : flattenedOptions) {
            String validationKey = buildAlternativeCostValidationKey(ability, game);
            Set<String> validChoices = cachedAlternativeCostValidation.get(validationKey);
            if (validChoices == null) {
                validChoices = testAllAlternativeCosts(ability, game);
                if (ALTERNATIVE_COST_SIM_CACHE_ENABLED) {
                    cachedAlternativeCostValidation.put(validationKey, new HashSet<>(validChoices));
                }
            }

            if (!validChoices.isEmpty()) {
                // At least one alternative worked
                validOptions.add(ability);
                // Always track if there were alternatives tested (even if only one is valid)
                // We detect alternatives by checking if the set contains anything other than just "0"
                boolean hasAlternatives = !(validChoices.size() == 1 && validChoices.contains("0"));
                if (hasAlternatives) {
                    validAlternativeCosts.put(ability.getSourceId(), new HashSet<>(validChoices));
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

        // Remove duplicate spell abilities with the same name.
        // Mana abilities are keyed by sourceId+name so each land/creature is a distinct tap option
        // (e.g. two Forests are separate choices — which one you tap before a bounce matters).
        List<ActivatedAbility> uniqueOptions = new ArrayList<>();
        Set<String> seenNames = new HashSet<>();

        for (ActivatedAbility ability : flattenedOptions) {
            String name = (ability instanceof ManaAbility)
                    ? ability.getSourceId() + ":" + ability.toString()
                    : ability.toString();
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
            BehaviorPolicyView latestBehavior = new BehaviorPolicyView(
                    lastActionProbs,
                    lastActionProbs,
                    turnExplorationMode,
                    turnActionEps,
                    turnRandomEps
            );
            String gameState = appendExplorationToState(formatGameState(game), latestBehavior, selectedIndex);

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

    // Track state leaks for diagnostics
    private boolean lastActivationHadStateLeak = false;

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
        // Trace entry
        trace(String.format("playMana ENTRY: unpaid=%s, excludeSource=%s, tapReservations=%d",
            unpaid.getText(), abilitySourceToExcludeFromMana, tapTargetCostReservations.size()));
        
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "PLAYMANA: Called for unpaid=" + unpaid.getText()
                    + ", ability=" + (ability != null ? ability.getRule() : "null")
                    + ", excludingSource=" + abilitySourceToExcludeFromMana);
        }
        int prevDepth = playManaDepth.get();
        String prevUnpaid = currentUnpaidManaText.get();
        playManaDepth.set(prevDepth + 1);
        currentUnpaidManaText.set(unpaid != null ? unpaid.getText() : "");
        boolean result;
        try {
            result = super.playMana(ability, unpaid, promptText, game);
        } finally {
            if (prevDepth <= 0) {
                playManaDepth.remove();
            } else {
                playManaDepth.set(prevDepth);
            }
            if (prevUnpaid == null) {
                currentUnpaidManaText.remove();
            } else {
                currentUnpaidManaText.set(prevUnpaid);
            }
        }
        if (ACTIVATION_DIAG) {
            RLTrainer.threadLocalLogger.get().info(
                    "PLAYMANA: Result=" + result + ", unpaid remaining=" + unpaid.getText());
        }
        
        // Trace exit
        trace(String.format("playMana EXIT: result=%s, remaining=%s", result, unpaid.getText()));
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

        if (inPlayManaContext() && producers.size() > 1) {
            producers.sort(Comparator.comparingInt(obj -> manaProducerUsePenalty(obj, game)));
            if (ACTIVATION_DIAG) {
                String ranked = producers.stream()
                        .limit(6)
                        .map(obj -> {
                            String name = obj != null ? obj.getName() : "unknown";
                            int pen = manaProducerUsePenalty(obj, game);
                            return name + "(pen=" + pen + ")";
                        })
                        .collect(Collectors.joining(", "));
                RLTrainer.threadLocalLogger.get().info("MANA-PRODUCER-RANK: " + ranked);
            }
        }

        return producers;
    }

    /**
     * Override cast() to add safety bookmarks that catch engine state leaks.
     * 
     * When PlayerImpl.cast() fails (e.g., spell == null after card.cast()), it doesn't
     * call restoreState(bookmark), leaving the card on the stack. This override detects
     * stack growth on failure and restores to the safety bookmark.
     */
    @Override
    public boolean cast(mage.abilities.SpellAbility ability, mage.game.Game game, boolean noMana, mage.ApprovingObject approvingObject) {
        if (ability == null || game == null) {
            trace("cast() EXIT: refused null ability/game");
            return false;
        }

        int preStackSize = game.getStack().size();
        int safetyBookmark = game.bookmarkState();
        
        trace(String.format("cast() ENTRY: stackSize=%d, source=%s, zone=%s",
            preStackSize, ability.getSourceId(),
            game.getState().getZone(ability.getSourceId())));
        
        boolean result = super.cast(ability, game, noMana, approvingObject);
        
        if (result) {
            game.removeBookmark(safetyBookmark);
            trace("cast() SUCCESS");
            lastActivationHadStateLeak = false;
        } else {
            int postStackSize = game.getStack().size();
            trace(String.format("cast() FAILED: postStackSize=%d (was %d)", postStackSize, preStackSize));
            
            if (postStackSize > preStackSize) {
                trace("cast() STATE LEAK: spell left on stack, restoring safety bookmark");
                game.restoreState(safetyBookmark, "RL safety restore after cast leak");
                lastActivationHadStateLeak = true;
            } else {
                game.removeBookmark(safetyBookmark);
                lastActivationHadStateLeak = false;
            }
        }
        return result;
    }

    /**
     * Override playAbility() to add safety bookmarks that catch engine state leaks.
     * 
     * When PlayerImpl.playAbility() fails after ability.activate(), the bookmark restore
     * may fail if inner restores invalidated it, leaving stack abilities behind. This
     * override detects stack growth on failure and restores to the safety bookmark.
     */
    @Override
    protected boolean playAbility(mage.abilities.ActivatedAbility ability, mage.game.Game game) {
        int preStackSize = game.getStack().size();
        int safetyBookmark = game.bookmarkState();
        
        trace(String.format("playAbility() ENTRY: stackSize=%d, ability=%s",
            preStackSize, ability.getRule()));
        
        boolean result = super.playAbility(ability, game);
        
        if (result) {
            game.removeBookmark(safetyBookmark);
            trace("playAbility() SUCCESS");
            lastActivationHadStateLeak = false;
        } else {
            int postStackSize = game.getStack().size();
            trace(String.format("playAbility() FAILED: postStackSize=%d (was %d)", postStackSize, preStackSize));
            
            if (postStackSize > preStackSize) {
                trace("playAbility() STATE LEAK: ability left on stack, restoring safety bookmark");
                game.restoreState(safetyBookmark, "RL safety restore after playAbility leak");
                lastActivationHadStateLeak = true;
            } else {
                game.removeBookmark(safetyBookmark);
                lastActivationHadStateLeak = false;
            }
        }
        return result;
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

    // -----------------------------------------------------------------------
    // Draft: pickCard override
    // -----------------------------------------------------------------------

    /**
     * Called by the draft engine for each pick. If a draft bridge is available,
     * uses the draft model; otherwise falls back to heuristic (super).
     */
    @Override
    public void pickCard(List<Card> cards, Deck deck, Draft draft) {
        if (draftBridge == null || cards.isEmpty()) {
            super.pickCard(cards, deck, draft);
            // Record seen cards (heuristic pick path - no RL data)
            int packNum = draft.getBoosterNum();
            int pickNum = draft.getCardNum();
            for (Card c : cards) {
                draftSeenCards.add(new DraftPickRecord(c, packNum, pickNum, false));
            }
            return;
        }

        int packNum = draft.getBoosterNum();
        int pickNum = draft.getCardNum();

        try {
            DraftStateBuilder.DraftStateOutput state = DraftStateBuilder.buildState(
                    draftPool, draftSeenCards, packNum, pickNum);

            float[] candFeats = DraftStateBuilder.buildCandidateFeatures(cards);
            int[] candIds = DraftStateBuilder.buildCandidateIds(cards);

            float[] scores = draftBridge.scoreDraftPick(
                    state.sequence, state.mask, state.tokenIds,
                    candFeats, candIds,
                    cards.size(), DraftStateBuilder.MAX_LEN, DraftStateBuilder.DIM_PER_TOKEN
            );

            // Last element is the value estimate
            float value = scores.length > cards.size() ? scores[cards.size()] : 0.0f;

            // Sample from softmax scores (or greedy in eval)
            int chosenIdx = sampleFromScores(scores, cards.size());
            Card chosenCard = cards.get(chosenIdx);

            // Compute log-prob for PPO
            float chosenScore = scores[chosenIdx];
            float logProb = (float) Math.log(Math.max(chosenScore, 1e-8f));

            // Candidate mask (all real = 0)
            int[] candMask = new int[cards.size()];

            draftSteps.add(new DraftStepData(
                    state.sequence, state.mask, state.tokenIds,
                    candFeats, candIds, candMask,
                    chosenIdx, logProb, value, 0 // headIdx=0 for pick
            ));

            // Log pick decision
            if (draftLogger != null && draftLogger.isEnabled()) {
                draftLogger.logPick(packNum, pickNum, cards, scores, chosenIdx, value, draftPool.size());
            }

            // Record seen-but-not-picked cards
            for (int i = 0; i < cards.size(); i++) {
                if (i != chosenIdx) {
                    draftSeenCards.add(new DraftPickRecord(cards.get(i), packNum, pickNum, false));
                }
            }

            // Add chosen card to pool
            draftPool.add(chosenCard);
            draftSeenCards.add(new DraftPickRecord(chosenCard, packNum, pickNum, true));

            draft.addPick(playerId, chosenCard.getId(), null);

        } catch (Exception e) {
            System.err.println("[DraftRL] pickCard error: " + e.getMessage());
            // Fallback: pick highest-rated card
            super.pickCard(cards, deck, draft);
        }
    }

    private int sampleFromScores(float[] scores, int n) {
        // Greedy: pick highest score
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (scores[i] > scores[best]) best = i;
        }
        return best;
    }

    // -----------------------------------------------------------------------
    // Draft: deck construction (called by DraftTrainer, not tournament system)
    // -----------------------------------------------------------------------

    /**
     * Uses the construction head to select land count and non-land cards,
     * then deterministically materializes lands to always produce a 40-card deck.
     */
    public Deck constructDraftDeck() {
        List<Card> pool = new ArrayList<>(draftPool);
        List<Card> nonLands = pool.stream().filter(c -> !c.isLand()).collect(Collectors.toList());
        List<Card> lands = pool.stream().filter(Card::isLand).collect(Collectors.toList());

        LandCountDecision landCountDecision = chooseLandCount(pool);
        int targetNonLandCount = Math.max(0, DRAFT_TARGET_DECK_SIZE - landCountDecision.appliedLandCount);

        // Hard playability repair for malformed pools: maintain 40 if target nonlands exceed available.
        List<String> repairActions = new ArrayList<>();
        if (targetNonLandCount > nonLands.size()) {
            int adjusted = nonLands.size();
            repairActions.add(String.format(
                    "adjust_target_nonlands old=%d new=%d reason=nonland_pool_exhausted",
                    targetNonLandCount, adjusted));
            targetNonLandCount = adjusted;
        }

        NonLandSelectionResult nonLandSelection = selectNonLandsSequentially(nonLands, targetNonLandCount);
        List<Card> selectedNonLands = nonLandSelection.selectedCards;
        int targetLandCount = DRAFT_TARGET_DECK_SIZE - selectedNonLands.size();

        LandSelectionResult landSelection = selectLandsSequentially(lands, selectedNonLands, targetLandCount);
        List<Card> selectedLands = landSelection.selectedCards;
        while (selectedLands.size() < targetLandCount) {
            String basicName = BASIC_LAND_NAMES[selectedLands.size() % BASIC_LAND_NAMES.length];
            Card extra = createBasicLand(basicName);
            if (extra == null) {
                break;
            }
            selectedLands.add(extra);
            repairActions.add("fill_missing_land synthetic=" + basicName);
        }
        if (selectedLands.size() > targetLandCount) {
            selectedLands = new ArrayList<>(selectedLands.subList(0, targetLandCount));
            repairActions.add("trim_land_overflow to_target=" + targetLandCount);
        }

        repairActions.addAll(applyMinimalSourceRepair(selectedNonLands, selectedLands));
        LandOriginBreakdown landOrigins = computeLandOriginBreakdown(selectedLands, lands);
        String landSourceSummary = buildLandSourceSummary(selectedLands);

        Deck deck = new Deck();
        for (Card c : selectedNonLands) {
            deck.getCards().add(c);
        }
        for (Card c : selectedLands) {
            deck.getCards().add(c);
        }

        if (draftLogger != null) {
            draftLogger.logConstruction(
                    pool,
                    selectedNonLands,
                    selectedLands,
                    landCountDecision.rawLandCount,
                    landCountDecision.floorLandCount,
                    targetLandCount,
                    targetNonLandCount,
                    landCountDecision.forcedByFloor,
                    selectedNonLands.size(),
                    selectedLands.size(),
                    deck.getCards().size(),
                    landOrigins.draftedNonBasicsUsed,
                    landOrigins.draftedBasicsUsed,
                    landOrigins.syntheticBasicsAdded,
                    nonLandSelection.stepLogs,
                    landSelection.stepLogs,
                    repairActions,
                    landSourceSummary
            );
        }

        return deck;
    }

    private LandCountDecision chooseLandCount(List<Card> fullPool) {
        int floor = computeCurrentLandFloor();
        int raw = floor;
        float rawLogProb = 0.0f;
        float value = 0.0f;

        if (draftBridge != null) {
            try {
                DraftStateBuilder.DraftStateOutput state = DraftStateBuilder.buildState(
                        fullPool, draftSeenCards, 3, 15);
                int numCands = DRAFT_LAND_COUNT_MAX - DRAFT_LAND_COUNT_MIN + 1;
                float[] candFeats = new float[numCands * DraftStateBuilder.DIM_PER_TOKEN];
                int[] candIds = new int[numCands];
                int[] candMask = new int[numCands];
                Arrays.fill(candMask, 1);
                for (int i = 0; i < numCands; i++) {
                    int landCount = DRAFT_LAND_COUNT_MIN + i;
                    int base = i * DraftStateBuilder.DIM_PER_TOKEN;
                    candFeats[base + 9] = 1.0f; // mark as "land-ish" candidate
                    candFeats[base + 17] = landCount / (float) DRAFT_LAND_COUNT_MAX;
                    candFeats[base + 2] = landCount / (float) DRAFT_LAND_COUNT_MAX;
                    candIds[i] = ("land_count_" + landCount).hashCode() & 0x7fff_ffff;
                }

                float[] scores = draftBridge.scoreDraftConstruction(
                        state.sequence, state.mask, state.tokenIds,
                        candFeats, candIds,
                        numCands, DraftStateBuilder.MAX_LEN, DraftStateBuilder.DIM_PER_TOKEN
                );
                value = scores.length > numCands ? scores[numCands] : 0.0f;
                int bestIdx = sampleFromScores(scores, numCands);
                raw = DRAFT_LAND_COUNT_MIN + bestIdx;

                int applied = Math.max(raw, floor);
                boolean forced = applied != raw;
                int appliedIdx = applied - DRAFT_LAND_COUNT_MIN;
                rawLogProb = (float) Math.log(Math.max(scores[appliedIdx], 1e-8f));
                draftSteps.add(new DraftStepData(
                        state.sequence, state.mask, state.tokenIds,
                        candFeats, candIds, candMask,
                        appliedIdx, rawLogProb, value, 1,
                        forced, "land_count"
                ));

                return new LandCountDecision(raw, floor, applied, forced);
            } catch (Exception e) {
                System.err.println("[DraftRL] land-count decision error: " + e.getMessage());
            }
        }

        int applied = Math.max(raw, floor);
        return new LandCountDecision(raw, floor, applied, applied != raw);
    }

    private NonLandSelectionResult selectNonLandsSequentially(List<Card> nonLands, int targetNonLandCount) {
        NonLandSelectionResult result = new NonLandSelectionResult();
        if (targetNonLandCount <= 0 || nonLands.isEmpty()) {
            return result;
        }

        List<Card> remaining = new ArrayList<>(nonLands);
        for (int step = 0; step < targetNonLandCount && !remaining.isEmpty(); step++) {
            int chosenIdx;
            float[] scores = null;
            float value = 0.0f;
            float[] candFeats = null;
            int[] candIds = null;
            int[] candMask = null;
            DraftStateBuilder.DraftStateOutput state = null;

            if (draftBridge != null) {
                try {
                    List<Card> stateCards = new ArrayList<>(result.selectedCards);
                    state = DraftStateBuilder.buildState(stateCards, draftSeenCards, 3, 15);
                    candFeats = DraftStateBuilder.buildCandidateFeatures(remaining);
                    candIds = DraftStateBuilder.buildCandidateIds(remaining);
                    candMask = new int[remaining.size()];
                    Arrays.fill(candMask, 1);
                    scores = draftBridge.scoreDraftConstruction(
                            state.sequence, state.mask, state.tokenIds,
                            candFeats, candIds,
                            remaining.size(), DraftStateBuilder.MAX_LEN, DraftStateBuilder.DIM_PER_TOKEN
                    );
                    value = scores.length > remaining.size() ? scores[remaining.size()] : 0.0f;
                    chosenIdx = sampleFromScores(scores, remaining.size());
                    recordConstructionStep(state, candFeats, candIds, candMask, chosenIdx, scores, value, "spell_pick");
                } catch (Exception e) {
                    System.err.println("[DraftRL] spell-pick step error: " + e.getMessage());
                    chosenIdx = 0;
                }
            } else {
                remaining.sort(Comparator.comparingDouble(Card::getManaValue));
                chosenIdx = 0;
            }

            Card chosen = remaining.remove(Math.max(0, Math.min(chosenIdx, remaining.size() - 1)));
            result.selectedCards.add(chosen);
            float chosenScore = (scores != null && chosenIdx < scores.length) ? scores[chosenIdx] : -1.0f;
            result.stepLogs.add(String.format(
                    "step=%d chosen_idx=%d card=%s candidates=%d score=%.4f",
                    step + 1, chosenIdx, chosen.getName(), (scores != null ? scores.length - 1 : remaining.size() + 1), chosenScore));
        }
        return result;
    }

    private LandSelectionResult selectLandsSequentially(List<Card> draftedLands, List<Card> selectedSpells, int targetLandCount) {
        LandSelectionResult result = new LandSelectionResult();
        if (targetLandCount <= 0) {
            return result;
        }

        List<Card> remainingDrafted = new ArrayList<>(draftedLands);
        List<Card> syntheticBasics = createSyntheticBasicCandidates();

        for (int step = 0; step < targetLandCount; step++) {
            List<Card> candidates = new ArrayList<>(remainingDrafted.size() + syntheticBasics.size());
            candidates.addAll(remainingDrafted);
            candidates.addAll(syntheticBasics);
            if (candidates.isEmpty()) {
                break;
            }

            int chosenIdx;
            float[] scores = null;
            float value = 0.0f;
            DraftStateBuilder.DraftStateOutput state = null;
            float[] candFeats = null;
            int[] candIds = null;
            int[] candMask = null;

            if (draftBridge != null) {
                try {
                    List<Card> stateCards = new ArrayList<>(selectedSpells.size() + result.selectedCards.size());
                    stateCards.addAll(selectedSpells);
                    stateCards.addAll(result.selectedCards);
                    state = DraftStateBuilder.buildState(stateCards, draftSeenCards, 3, 15);
                    candFeats = DraftStateBuilder.buildCandidateFeatures(candidates);
                    candIds = DraftStateBuilder.buildCandidateIds(candidates);
                    candMask = new int[candidates.size()];
                    Arrays.fill(candMask, 1);
                    scores = draftBridge.scoreDraftConstruction(
                            state.sequence, state.mask, state.tokenIds,
                            candFeats, candIds,
                            candidates.size(), DraftStateBuilder.MAX_LEN, DraftStateBuilder.DIM_PER_TOKEN
                    );
                    value = scores.length > candidates.size() ? scores[candidates.size()] : 0.0f;
                    chosenIdx = sampleFromScores(scores, candidates.size());
                    recordConstructionStep(state, candFeats, candIds, candMask, chosenIdx, scores, value, "land_pick");
                } catch (Exception e) {
                    System.err.println("[DraftRL] land-pick step error: " + e.getMessage());
                    chosenIdx = 0;
                }
            } else {
                int[] demand = computeColorDemand(selectedSpells, targetLandCount);
                String wanted = pickBasicForDemand(demand, step);
                int found = -1;
                for (int i = 0; i < candidates.size(); i++) {
                    if (wanted.equals(candidates.get(i).getName())) {
                        found = i;
                        break;
                    }
                }
                chosenIdx = found >= 0 ? found : 0;
            }

            chosenIdx = Math.max(0, Math.min(chosenIdx, candidates.size() - 1));
            Card chosenCandidate = candidates.get(chosenIdx);
            Card deckCard;
            String origin;
            if (chosenIdx < remainingDrafted.size()) {
                deckCard = remainingDrafted.remove(chosenIdx);
                origin = isBasicLandName(deckCard.getName()) ? "drafted_basic" : "drafted_nonbasic";
            } else {
                deckCard = createBasicLand(chosenCandidate.getName());
                if (deckCard == null) {
                    deckCard = chosenCandidate.copy();
                }
                origin = "synthetic_basic";
            }
            result.selectedCards.add(deckCard);
            float chosenScore = (scores != null && chosenIdx < scores.length) ? scores[chosenIdx] : -1.0f;
            result.stepLogs.add(String.format(
                    "step=%d chosen_idx=%d card=%s origin=%s candidates=%d score=%.4f",
                    step + 1, chosenIdx, deckCard.getName(), origin, (scores != null ? scores.length - 1 : candidates.size()), chosenScore));
        }

        return result;
    }

    private void recordConstructionStep(
            DraftStateBuilder.DraftStateOutput state,
            float[] candFeats,
            int[] candIds,
            int[] candMask,
            int chosenIdx,
            float[] scores,
            float value,
            String kind
    ) {
        if (state == null || candFeats == null || candIds == null || candMask == null
                || scores == null || chosenIdx < 0 || chosenIdx >= candIds.length) {
            return;
        }
        float logProb = (float) Math.log(Math.max(scores[chosenIdx], 1e-8f));
        draftSteps.add(new DraftStepData(
                state.sequence, state.mask, state.tokenIds,
                candFeats, candIds, candMask,
                chosenIdx, logProb, value, 1,
                false, kind
        ));
    }

    private List<Card> createSyntheticBasicCandidates() {
        List<Card> basics = new ArrayList<>();
        for (String basic : BASIC_LAND_NAMES) {
            Card c = createBasicLand(basic);
            if (c != null) {
                basics.add(c);
            }
        }
        return basics;
    }

    private List<String> applyMinimalSourceRepair(List<Card> selectedSpells, List<Card> selectedLands) {
        List<String> repairs = new ArrayList<>();
        if (selectedLands.isEmpty()) {
            return repairs;
        }
        boolean[] demanded = computeDemandedColors(selectedSpells);
        int[] sourceCounts = computeLandSourceCounts(selectedLands);

        for (int color = 0; color < 5; color++) {
            if (!demanded[color] || sourceCounts[color] > 0) {
                continue;
            }

            int donorIdx = findBasicSwapDonorIndex(selectedLands, sourceCounts, demanded);
            if (donorIdx < 0) {
                repairs.add("source_repair_failed missing=" + BASIC_LAND_NAMES[color]);
                continue;
            }

            Card replacement = createBasicLand(BASIC_LAND_NAMES[color]);
            if (replacement == null) {
                repairs.add("source_repair_failed missing=" + BASIC_LAND_NAMES[color] + " reason=create_basic_failed");
                continue;
            }

            Card old = selectedLands.set(donorIdx, replacement);
            sourceCounts = computeLandSourceCounts(selectedLands);
            repairs.add(String.format(
                    "source_repair swap_out=%s swap_in=%s",
                    old.getName(), replacement.getName()));
        }
        return repairs;
    }

    private int[] computeColorDemand(List<Card> selectedSpells, int numLands) {
        int[] colorCounts = new int[5]; // W U B R G
        for (Card c : selectedSpells) {
            try {
                Mana mana = new Mana();
                for (mage.abilities.costs.mana.ManaCost mc : c.getManaCost()) {
                    mana.add(mc.getMana());
                }
                colorCounts[0] += mana.getWhite();
                colorCounts[1] += mana.getBlue();
                colorCounts[2] += mana.getBlack();
                colorCounts[3] += mana.getRed();
                colorCounts[4] += mana.getGreen();
            } catch (Exception ignored) {
            }
        }

        int total = 0;
        for (int count : colorCounts) {
            total += count;
        }

        int[] demand = new int[5];
        if (total <= 0) {
            demand[0] = numLands;
            return demand;
        }

        int assigned = 0;
        for (int i = 0; i < 5; i++) {
            demand[i] = (int) Math.floor(colorCounts[i] * numLands / (double) total);
            assigned += demand[i];
        }
        int remainder = numLands - assigned;
        int idx = 0;
        while (remainder > 0) {
            if (colorCounts[idx] > 0) {
                demand[idx]++;
                remainder--;
            }
            idx = (idx + 1) % 5;
        }
        return demand;
    }

    private boolean[] computeDemandedColors(List<Card> selectedSpells) {
        boolean[] demanded = new boolean[5];
        for (Card c : selectedSpells) {
            try {
                Mana mana = new Mana();
                for (mage.abilities.costs.mana.ManaCost mc : c.getManaCost()) {
                    mana.add(mc.getMana());
                }
                if (mana.getWhite() > 0) demanded[0] = true;
                if (mana.getBlue() > 0) demanded[1] = true;
                if (mana.getBlack() > 0) demanded[2] = true;
                if (mana.getRed() > 0) demanded[3] = true;
                if (mana.getGreen() > 0) demanded[4] = true;
            } catch (Exception ignored) {
            }
        }
        return demanded;
    }

    private int[] computeLandSourceCounts(List<Card> lands) {
        int[] counts = new int[5];
        for (Card land : lands) {
            for (int c = 0; c < 5; c++) {
                if (landProvidesColor(land, c)) {
                    counts[c]++;
                }
            }
        }
        return counts;
    }

    private boolean landProvidesColor(Card land, int colorIdx) {
        if (land == null) {
            return false;
        }
        String name = land.getName();
        if ("Plains".equals(name)) return colorIdx == 0;
        if ("Island".equals(name)) return colorIdx == 1;
        if ("Swamp".equals(name)) return colorIdx == 2;
        if ("Mountain".equals(name)) return colorIdx == 3;
        if ("Forest".equals(name)) return colorIdx == 4;

        try {
            String rules = String.join(" ", land.getRules()).toUpperCase();
            if (rules.contains("ANY COLOR")) {
                return true;
            }
            switch (colorIdx) {
                case 0:
                    return rules.contains("{W}");
                case 1:
                    return rules.contains("{U}");
                case 2:
                    return rules.contains("{B}");
                case 3:
                    return rules.contains("{R}");
                case 4:
                    return rules.contains("{G}");
                default:
                    return false;
            }
        } catch (Exception e) {
            return false;
        }
    }

    private int findBasicSwapDonorIndex(List<Card> lands, int[] sourceCounts, boolean[] demanded) {
        int bestIdx = -1;
        int bestScore = Integer.MIN_VALUE;

        for (int i = 0; i < lands.size(); i++) {
            Card c = lands.get(i);
            if (!isBasicLandName(c.getName())) {
                continue;
            }
            int donorColor = colorIndexForBasic(c.getName());
            if (donorColor < 0 || sourceCounts[donorColor] <= 0) {
                continue;
            }
            int score;
            if (!demanded[donorColor]) {
                score = 1000 + sourceCounts[donorColor];
            } else if (sourceCounts[donorColor] > 1) {
                score = 100 + sourceCounts[donorColor];
            } else {
                score = sourceCounts[donorColor] - 10;
            }
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private String buildLandSourceSummary(List<Card> lands) {
        int[] source = computeLandSourceCounts(lands);
        return String.format("W=%d U=%d B=%d R=%d G=%d", source[0], source[1], source[2], source[3], source[4]);
    }

    private LandOriginBreakdown computeLandOriginBreakdown(List<Card> selectedLands, List<Card> draftedLands) {
        Set<UUID> draftedIds = draftedLands.stream().map(Card::getId).collect(Collectors.toSet());
        LandOriginBreakdown out = new LandOriginBreakdown();
        for (Card c : selectedLands) {
            if (draftedIds.contains(c.getId())) {
                if (isBasicLandName(c.getName())) {
                    out.draftedBasicsUsed++;
                } else {
                    out.draftedNonBasicsUsed++;
                }
            } else if (isBasicLandName(c.getName())) {
                out.syntheticBasicsAdded++;
            }
        }
        return out;
    }

    private int computeCurrentLandFloor() {
        if (!DRAFT_MIN_LANDS_ENABLE) {
            return 0;
        }
        int episode = Math.max(0, currentDraftEpisode);
        double progress = Math.min(1.0, Math.max(0.0, episode / (double) DRAFT_MIN_LANDS_DECAY_EPISODES));
        int floor = (int) Math.round(DRAFT_MIN_LANDS_START
                + (DRAFT_MIN_LANDS_END - DRAFT_MIN_LANDS_START) * progress);
        return Math.max(DRAFT_LAND_COUNT_MIN, Math.min(DRAFT_LAND_COUNT_MAX, floor));
    }

    private static final String[] BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest"};

    private String pickBasicForDemand(int[] demand, int cycleIdx) {
        int bestIdx = -1;
        int bestVal = Integer.MIN_VALUE;
        for (int i = 0; i < demand.length; i++) {
            if (demand[i] > bestVal) {
                bestVal = demand[i];
                bestIdx = i;
            }
        }
        if (bestIdx >= 0 && bestVal > 0) {
            return BASIC_LAND_NAMES[bestIdx];
        }
        return BASIC_LAND_NAMES[cycleIdx % BASIC_LAND_NAMES.length];
    }

    private void decrementColorDemand(int[] demand, String basicName) {
        int idx = colorIndexForBasic(basicName);
        if (idx >= 0 && demand[idx] > 0) {
            demand[idx]--;
        }
    }

    private boolean isBasicLandName(String name) {
        return "Plains".equals(name)
                || "Island".equals(name)
                || "Swamp".equals(name)
                || "Mountain".equals(name)
                || "Forest".equals(name);
    }

    private int colorIndexForBasic(String basicName) {
        switch (basicName) {
            case "Plains":
                return 0;
            case "Island":
                return 1;
            case "Swamp":
                return 2;
            case "Mountain":
                return 3;
            case "Forest":
                return 4;
            default:
                return -1;
        }
    }

    private Card createBasicLand(String basicName) {
        try {
            CardInfo cardInfo = CardRepository.instance.findCard(basicName);
            return cardInfo != null ? cardInfo.createCard() : null;
        } catch (Exception e) {
            return null;
        }
    }

    private static class LandCountDecision {
        final int rawLandCount;
        final int floorLandCount;
        final int appliedLandCount;
        final boolean forcedByFloor;

        LandCountDecision(int rawLandCount, int floorLandCount, int appliedLandCount, boolean forcedByFloor) {
            this.rawLandCount = rawLandCount;
            this.floorLandCount = floorLandCount;
            this.appliedLandCount = appliedLandCount;
            this.forcedByFloor = forcedByFloor;
        }
    }

    private static class LandBaseResult {
        final List<Card> lands = new ArrayList<>();
        int draftedNonBasicsUsed = 0;
        int draftedBasicsUsed = 0;
        int syntheticBasicsAdded = 0;
    }

    private static class NonLandSelectionResult {
        final List<Card> selectedCards = new ArrayList<>();
        final List<String> stepLogs = new ArrayList<>();
    }

    private static class LandSelectionResult {
        final List<Card> selectedCards = new ArrayList<>();
        final List<String> stepLogs = new ArrayList<>();
    }

    private static class LandOriginBreakdown {
        int draftedNonBasicsUsed = 0;
        int draftedBasicsUsed = 0;
        int syntheticBasicsAdded = 0;
    }

    /**
     * Candidate wrapper for combat decisions (attacking and blocking).
     * When creature is null, this represents the DONE sentinel (stop selecting).
     */
    static class CombatCandidate {
        final Permanent creature;  // the attacker (Phase 1) or blocker making the decision; null = DONE
        final Object context;      // Phase 1: null. Phase 2 attack target: defender UUID. Block: attacker Permanent.

        CombatCandidate(Permanent creature, Object context) {
            this.creature = creature;
            this.context = context;
        }

        boolean isDone() {
            return creature == null;
        }
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
