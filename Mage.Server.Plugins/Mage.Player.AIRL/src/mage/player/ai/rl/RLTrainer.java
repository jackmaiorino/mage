package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameException;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.player.ai.ComputerPlayer;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

public class RLTrainer {

    private static final Logger logger = Logger.getLogger(RLTrainer.class);

    // Local training command:
    // $env:TOTAL_EPISODES='1'; $env:DECKS_DIR='src/mage/player/ai/decks/Pauper'; mvn -q compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
    private static final int NUM_EPISODES = EnvConfig.i32("TOTAL_EPISODES", 10000);
    private static final int NUM_EVAL_EPISODES = EnvConfig.i32("EVAL_EPISODES", 5);

    // Defaults assume running from repo root. Override via DECKS_DIR if needed.
    public static final String DECKS_DIRECTORY = EnvConfig.str("DECKS_DIR",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper");
    // Optional: explicit deck list file (one path per line, relative to CWD unless absolute)
    public static final String DECK_LIST_FILE = EnvConfig.str("DECK_LIST_FILE", "");
    // Optional: separate deck list for RL agent (if empty, agent uses DECK_LIST_FILE)
    public static final String RL_AGENT_DECK_LIST_FILE = EnvConfig.str("RL_AGENT_DECK_LIST", "");

    // Model and episode counter paths -- all derived from MODEL_PROFILE via RLLogPaths.
    public static final String MODEL_FILE_PATH        = RLLogPaths.MODEL_FILE_PATH;
    // Episode-level statistics will be appended here (CSV)
    public static final String STATS_FILE_PATH        = RLLogPaths.TRAINING_STATS_PATH;
    // Path that stores the cumulative number of episodes trained so far (persisted across runs)
    public static final String EPISODE_COUNT_PATH     = RLLogPaths.EPISODE_COUNT_PATH;
    // Auto-detect optimal number of threads based on CPU cores
    private static final int DEFAULT_GAME_RUNNERS = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    public static final int NUM_THREADS = EnvConfig.i32("NUM_THREADS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_GAME_RUNNERS = EnvConfig.i32("NUM_GAME_RUNNERS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_EPISODES_PER_GAME_RUNNER = EnvConfig.i32("EPISODES_PER_WORKER", 500);
    public static final int EVAL_EVERY = EnvConfig.i32("EVAL_EVERY", 5000);
    private static final boolean EVAL_AT_START = EnvConfig.bool("EVAL_AT_START", false);
    private static final int EVAL_CP7_SKILL = EnvConfig.i32("EVAL_CP7_SKILL", 7);
    private static final int EVAL_GAMES_PER_DECK = EnvConfig.i32("EVAL_GAMES_PER_DECK", 5);

    private static final boolean SHARED_GPU_MODE = PythonModelFactory.isSharedGpuMode();
    private static final DeckTemplateCache DECK_TEMPLATE_CACHE = new DeckTemplateCache();
    private static final AsyncLineWriter ASYNC_LINE_WRITER = new AsyncLineWriter("RLTrainer-AsyncLineWriter", logger);
    public static final PythonModel sharedModel = new LazyPythonModel(PythonModelFactory::getInstance);
    public static final MetricsCollector metrics = MetricsCollector.getInstance();
    private static final boolean ACTOR_LEARNER_ASYNC = EnvConfig.bool("ACTOR_LEARNER_ASYNC", true);
    private static final int ACTOR_LEARNER_QUEUE_MAX = Math.max(1, EnvConfig.i32("ACTOR_LEARNER_QUEUE_MAX", 512));
    private static final int ACTOR_LEARNER_WORKERS = Math.max(1, EnvConfig.i32("ACTOR_LEARNER_WORKERS", 2));
    private static final String ACTOR_LEARNER_BACKPRESSURE =
            EnvConfig.str("ACTOR_LEARNER_BACKPRESSURE", "block").trim().toLowerCase();
    private static final int ACTOR_LEARNER_OFFER_TIMEOUT_MS =
            Math.max(1, EnvConfig.i32("ACTOR_LEARNER_OFFER_TIMEOUT_MS", 30000));
    private static final int ACTOR_LEARNER_DRAIN_TIMEOUT_MS =
            Math.max(0, EnvConfig.i32("ACTOR_LEARNER_DRAIN_TIMEOUT_MS", 10000));
    private static final ActorLearnerDispatcher ACTOR_LEARNER = new ActorLearnerDispatcher(
            ACTOR_LEARNER_ASYNC, ACTOR_LEARNER_QUEUE_MAX, ACTOR_LEARNER_WORKERS,
            ACTOR_LEARNER_BACKPRESSURE, ACTOR_LEARNER_OFFER_TIMEOUT_MS);

    // Global episode counter to track total episodes across all threads
    private static final AtomicInteger EPISODE_COUNTER = new AtomicInteger(0);
    private static final AtomicInteger ACTIVE_EPISODES = new AtomicInteger(0);
    private static final boolean TRAIN_DIAG = EnvConfig.bool("TRAIN_DIAG", false);
    private static final int TRAIN_DIAG_EVERY = EnvConfig.i32("TRAIN_DIAG_EVERY", 50);
    private static final int TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER =
            Math.max(0, EnvConfig.i32("TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER", 0));
    private static final boolean AWR_SELECTED_ACTION_TARGETS_ENABLE =
            EnvConfig.bool("RL_AWR_SELECTED_ACTION_TARGETS_ENABLE", false);
    private static final double AWR_GAMMA =
            Math.max(0.0, Math.min(1.0, EnvConfig.f64("RL_AWR_GAMMA", 0.99)));
    private static final double AWR_TEMPERATURE =
            Math.max(1e-6, EnvConfig.f64("RL_AWR_TEMPERATURE", 0.50));
    private static final double AWR_MIN_WEIGHT =
            Math.max(0.0, EnvConfig.f64("RL_AWR_MIN_WEIGHT", 0.05));
    private static final double AWR_MAX_WEIGHT =
            Math.max(AWR_MIN_WEIGHT, EnvConfig.f64("RL_AWR_MAX_WEIGHT", 5.0));
    private static final boolean AWR_POSITIVE_ADVANTAGE_ONLY =
            EnvConfig.bool("RL_AWR_POSITIVE_ADVANTAGE_ONLY", false);
    private static final long RL_BASE_SEED = EnvConfig.i64("RL_BASE_SEED", -1L);
    private static final boolean EVAL_REPLAY_METADATA = EnvConfig.bool("EVAL_REPLAY_METADATA", false);
    private static final long EVAL_REPLAY_SEED_BASE = EnvConfig.i64("EVAL_REPLAY_SEED_BASE", 7777L);

    private static Random newSeededRandom(long salt) {
        if (RL_BASE_SEED < 0L) {
            return new Random();
        }
        long mixed = RL_BASE_SEED ^ (salt * 0x5DEECE66DL);
        return new Random(mixed);
    }

    private static long evalReplaySeed(int gameIndex) {
        return EVAL_REPLAY_SEED_BASE + 7919L * (long) (gameIndex + 1);
    }

    private static long replayRandomUtilSeed(long replaySeed) {
        return replaySeed ^ 0x6A09E667F3BCC909L;
    }

    private static void setReplayTraceContext(int scenario, long replaySeed, String scope) {
        System.setProperty("xmage.replay.scenario", String.valueOf(scenario));
        System.setProperty("xmage.replay.seed", String.valueOf(replaySeed));
        System.setProperty("xmage.replay.random_util_seed", String.valueOf(replayRandomUtilSeed(replaySeed)));
        System.setProperty("xmage.replay.scope", scope == null ? "" : scope);
    }

    private static Deck replayShuffledCopy(Deck source, long seed) {
        Deck deck = source.copy();
        List<Card> cards = new ArrayList<>(deck.getCards());
        Collections.shuffle(cards, new Random(seed));
        deck.getCards().clear();
        for (Card card : cards) {
            deck.getCards().add(card);
        }
        return deck;
    }

    private static void forceReplayLibraryOrder(Player player, Deck deck, Game game) {
        if (player == null || deck == null || game == null) {
            return;
        }
        LinkedHashSet<Card> ordered = new LinkedHashSet<>();
        for (Card card : deck.getCards()) {
            if (card != null && !card.isExtraDeckCard()) {
                ordered.add(card);
            }
        }
        player.getLibrary().clear();
        player.getLibrary().addAll(ordered, game);
    }

    // ============================================================
    // Adaptive Curriculum Learning Configuration
    // ============================================================
    private static final boolean ADAPTIVE_CURRICULUM = EnvConfig.bool("ADAPTIVE_CURRICULUM", true);

    // Game logging: log every N episodes (0 = disabled, includes episode 0)
    private static final int GAME_LOG_FREQUENCY = EnvConfig.i32("GAME_LOG_FREQUENCY", SHARED_GPU_MODE ? 0 : 200);
    private static final int WINRATE_WINDOW = EnvConfig.i32("WINRATE_WINDOW", 100);

    // Matchup-balanced RL-deck sampling: track per-deck recent wins over a rolling
    // window and sample agent deck with probability inversely proportional to
    // winrate. Oversamples weak decks (Wildfire) and undersamples dominant ones
    // (Rally). Disabled by default; enable with MATCHUP_BALANCED_SAMPLING=1.
    private static final boolean MATCHUP_BALANCED_SAMPLING =
            "1".equals(System.getenv().getOrDefault("MATCHUP_BALANCED_SAMPLING", "0"));
    // When rolling value accuracy crosses this, we log a MCTS-THRESHOLD alert
    // so the operator (or an external watchdog) knows to schedule MCTS eval.
    private static final double VALUE_ACCURACY_MCTS_THRESHOLD =
            Double.parseDouble(System.getenv().getOrDefault("VALUE_ACCURACY_MCTS_THRESHOLD", "0.70"));
    private static final java.util.concurrent.atomic.AtomicBoolean MCTS_THRESHOLD_FIRED =
            new java.util.concurrent.atomic.AtomicBoolean(false);
    private static final int DECK_WINRATE_WINDOW = EnvConfig.i32("DECK_WINRATE_WINDOW", 200);
    // Weight = 1 / (winrate + floor). Smaller floor => more aggressive oversampling.
    private static final double DECK_SAMPLE_FLOOR =
            Double.parseDouble(System.getenv().getOrDefault("DECK_SAMPLE_FLOOR", "0.15"));
    private static final java.util.concurrent.ConcurrentHashMap<String, java.util.Deque<Boolean>>
            DECK_RECENT_WINS = new java.util.concurrent.ConcurrentHashMap<>();

    /** Record a training game outcome for the given RL-piloted deck. */
    static void recordDeckTrainingOutcome(String deckKey, boolean rlWon) {
        java.util.Deque<Boolean> q = DECK_RECENT_WINS.computeIfAbsent(deckKey,
                k -> new java.util.concurrent.ConcurrentLinkedDeque<>());
        q.addLast(rlWon);
        while (q.size() > DECK_WINRATE_WINDOW) q.pollFirst();
    }

    private static double deckRecentWinrate(String deckKey) {
        java.util.Deque<Boolean> q = DECK_RECENT_WINS.get(deckKey);
        if (q == null || q.isEmpty()) return 0.5;
        int wins = 0, n = 0;
        for (Boolean b : q) { if (Boolean.TRUE.equals(b)) wins++; n++; }
        return n == 0 ? 0.5 : (double) wins / n;
    }

    /** Pick an RL-piloted deck. Weighted inverse-winrate when
     *  MATCHUP_BALANCED_SAMPLING=1 and we have data; uniform otherwise. */
    static java.nio.file.Path pickAgentDeck(java.util.List<java.nio.file.Path> pool, java.util.Random rand) {
        if (!MATCHUP_BALANCED_SAMPLING || pool.size() <= 1) {
            return pool.get(rand.nextInt(pool.size()));
        }
        double[] weights = new double[pool.size()];
        double sum = 0.0;
        for (int i = 0; i < pool.size(); i++) {
            String key = pool.get(i).getFileName().toString();
            double wr = deckRecentWinrate(key);
            weights[i] = 1.0 / (wr + DECK_SAMPLE_FLOOR);
            sum += weights[i];
        }
        double pick = rand.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < pool.size(); i++) {
            acc += weights[i];
            if (pick <= acc) return pool.get(i);
        }
        return pool.get(pool.size() - 1);
    }
    // When false, suppresses training_stats.csv and agent_status.json writes (auxiliary JVMs in multi-node)
    private static final boolean GAME_STATS_WRITER = EnvConfig.bool("GAME_STATS_WRITER", true);

    // Minimum games at current difficulty before allowing level change
    // Prevents premature promotion/demotion based on mixed difficulty data
    private static final int MIN_GAMES_PER_DIFFICULTY = EnvConfig.i32("MIN_GAMES_PER_DIFFICULTY", 100);

    // Opponent difficulty thresholds with hysteresis to prevent oscillation
    // Updated: removed useless ComputerPlayer, start at CP7 skill=1
    // Upgrade thresholds (need this winrate to move up)
    private static final double WEAK_TO_MEDIUM_THRESHOLD = EnvConfig.f64("THRESHOLD_WEAK_MEDIUM", 0.50);
    private static final double MEDIUM_TO_STRONG_THRESHOLD = EnvConfig.f64("THRESHOLD_MEDIUM_STRONG", 0.55);
    private static final double STRONG_TO_SELFPLAY_THRESHOLD = EnvConfig.f64("THRESHOLD_STRONG_SELFPLAY", 0.60);

    // Downgrade thresholds (need to drop below this to move down)
    // Default: 5% gap to prevent oscillation at boundaries
    private static final double MEDIUM_TO_WEAK_THRESHOLD = EnvConfig.f64("THRESHOLD_MEDIUM_WEAK", 0.45);
    private static final double STRONG_TO_MEDIUM_THRESHOLD = EnvConfig.f64("THRESHOLD_STRONG_MEDIUM", 0.50);
    private static final double SELFPLAY_TO_STRONG_THRESHOLD = EnvConfig.f64("THRESHOLD_SELFPLAY_STRONG", 0.55);

    // Episode boundaries for fixed curriculum (if not using adaptive)
    private static final int FIXED_WEAK_UNTIL = EnvConfig.i32("FIXED_WEAK_UNTIL", 10000);
    private static final int FIXED_MEDIUM_UNTIL = EnvConfig.i32("FIXED_MEDIUM_UNTIL", 20000);
    private static final int FIXED_STRONG_UNTIL = EnvConfig.i32("FIXED_STRONG_UNTIL", 30000);

    // Thread-safe circular buffer for tracking recent wins
    private static final ConcurrentLinkedQueue<Boolean> recentWins = new ConcurrentLinkedQueue<>();
    private static final AtomicInteger winCount = new AtomicInteger(0);

    // Track current opponent level for hysteresis
    enum OpponentLevel {
        WEAK, MEDIUM, STRONG, SELFPLAY
    }
    private static volatile OpponentLevel currentOpponentLevel = OpponentLevel.WEAK;
    private static String lastOpponentType = "UNKNOWN";

    // Track games played at current difficulty level
    private static final AtomicInteger gamesAtCurrentLevel = new AtomicInteger(0);

    // ============================================================
    // League-style opponent sampling (bots never go to zero)
    // ============================================================
    private static final String OPPONENT_SAMPLER = EnvConfig.str("OPPONENT_SAMPLER", "league"); // league|adaptive|fixed|meta|ladder|skillmix|hybrid|meta_hybrid|self
    private static final String LEAGUE_MODE = EnvConfig.str("LEAGUE_MODE", ""); // "rl_only" = no CP7 fallback

    // Eval benchmark settings
    private static final String EVAL_OPPONENT_DECK = EnvConfig.str("EVAL_OPPONENT_DECK", "");
    private static final int EVAL_OPPONENT_SKILL = EnvConfig.i32("EVAL_OPPONENT_SKILL", 1);
    private static final int EVAL_NUM_GAMES = EnvConfig.i32("EVAL_NUM_GAMES", 50);
    private static final String EVAL_RESULTS_FILE = EnvConfig.str("EVAL_RESULTS_FILE", "");

    // Self-play probability ramps from START->END over RAMP_EPISODES, but is capped to (1 - BOT_FLOOR).
    private static final double BOT_FLOOR_P = EnvConfig.f64("BOT_FLOOR", 0.25);
    private static final double SELFPLAY_START_P = EnvConfig.f64("SELFPLAY_START_P", 0.20);
    private static final double SELFPLAY_END_P = EnvConfig.f64("SELFPLAY_END_P", 0.80);
    private static final int SELFPLAY_RAMP_EPISODES = EnvConfig.i32("SELFPLAY_RAMP_EPISODES", 10000);
    // Bot mix among CP7(skill=1/2/3). Format: "w1,w2,w3"
    private static final String BOT_MIX = EnvConfig.str("BOT_MIX", "0.25,0.35,0.40");

    private static final String SNAPSHOT_DIR = EnvConfig.str("SNAPSHOT_DIR",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/snapshots");
    private static final double SNAPSHOT_OPPONENT_PROB = EnvConfig.f64("SNAPSHOT_OPPONENT_PROB", 0.20);
    private static final int SNAPSHOT_START_EPISODE = EnvConfig.i32("SNAPSHOT_START_EPISODE", 2000);

    // ============================================================
    // League v2 (replaces prior league behavior)
    // ============================================================
    private static final int LEAGUE_TICK_EPISODES = EnvConfig.i32("LEAGUE_TICK_EPISODES", 5000);
    private static final int LEAGUE_BASELINE_GAMES_PER_MATCHUP = EnvConfig.i32("LEAGUE_BASELINE_GAMES_PER_MATCHUP", 3);
    private static final String LEAGUE_BASELINE_DECKLIST_FILE = EnvConfig.str("LEAGUE_BASELINE_DECKLIST_FILE", DECK_LIST_FILE);
    private static final int LEAGUE_BASELINE_BOT_SKILL = EnvConfig.i32("LEAGUE_BASELINE_BOT_SKILL", 1);
    private static final double LEAGUE_PROMOTE_WR = EnvConfig.f64("LEAGUE_PROMOTE_WR", 0.55);
    private static final double LEAGUE_POOL_FLOOR_WR = EnvConfig.f64("LEAGUE_POOL_FLOOR_WR", 0.50);
    private static final double LEAGUE_CHAMPION_PROMOTE_WR = EnvConfig.f64("LEAGUE_CHAMPION_PROMOTE_WR", 0.55);
    private static final int LEAGUE_POOL_MAX = EnvConfig.i32("LEAGUE_POOL_MAX", 30);
    private static final int LEAGUE_POOL_CHAMPIONS = EnvConfig.i32("LEAGUE_POOL_CHAMPIONS", 8);
    private static final int LEAGUE_POOL_RECENT = EnvConfig.i32("LEAGUE_POOL_RECENT", 8);
    private static final double LEAGUE_POST_HEURISTIC_P = EnvConfig.f64("LEAGUE_POST_HEURISTIC_P", 0.20);
    private static final double LEAGUE_POST_LOCAL_P = EnvConfig.f64("LEAGUE_POST_LOCAL_P", 0.40);
    private static final double LEAGUE_POST_CROSS_P = EnvConfig.f64("LEAGUE_POST_CROSS_P", 0.40);
    private static final int LEAGUE_POST_HEURISTIC_SKILL = EnvConfig.i32("LEAGUE_POST_HEURISTIC_SKILL", 1);
    private static final int LEAGUE_CROSS_PROFILE_REFRESH_MS = EnvConfig.i32("LEAGUE_CROSS_PROFILE_REFRESH_MS", 30000);
    private static final String LEAGUE_REGISTRY_PATH = EnvConfig.str("LEAGUE_REGISTRY_PATH",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_league_registry.json");
    private static final String LEAGUE_REPORTS_DIR = EnvConfig.str("LEAGUE_REPORTS_DIR",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper");
    private static final int LEAGUE_ELO_GAMES_PER_DIRECTION = EnvConfig.i32("LEAGUE_ELO_GAMES_PER_DIRECTION", 6);
    private static final double LEAGUE_ELO_K_FACTOR = EnvConfig.f64("LEAGUE_ELO_K_FACTOR", 20.0);
    private static final boolean LEAGUE_ANCHOR_ENABLE = EnvConfig.bool("LEAGUE_ANCHOR_ENABLE", true);
    private static final int LEAGUE_ANCHOR_GAMES = EnvConfig.i32("LEAGUE_ANCHOR_GAMES", 20);
    private static final int LEAGUE_EVAL_CADENCE_EPISODES = EnvConfig.i32("LEAGUE_EVAL_CADENCE_EPISODES", 5000);
    private static final boolean LEAGUE_EVAL_FORCE = EnvConfig.bool("LEAGUE_EVAL_FORCE", false);
    private static final int LEAGUE_BENCHMARK_THREADS = EnvConfig.i32("LEAGUE_BENCHMARK_THREADS",
            Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
    private static final int LEAGUE_BENCHMARK_LOG_EVERY = EnvConfig.i32("LEAGUE_BENCHMARK_LOG_EVERY", 25);
    private static final int LEAGUE_BENCHMARK_HEARTBEAT_SEC = EnvConfig.i32("LEAGUE_BENCHMARK_HEARTBEAT_SEC", 30);
    private static final int LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC = EnvConfig.i32("LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC", 900);
    private static final boolean LEAGUE_EVAL_GAME_LOGGING = EnvConfig.bool("LEAGUE_EVAL_GAME_LOGGING", false);

    // ============================================================
    // Ladder mode (skill-only progression, no self-play)
    // ============================================================
    private static final String LADDER_SKILLS = EnvConfig.str("LADDER_SKILLS", "0,1,2,3");
    private static final double LADDER_PROMOTE_WR = EnvConfig.f64("LADDER_PROMOTE_WR", 0.55);
    private static final int LADDER_TICK_EPISODES = EnvConfig.i32("LADDER_TICK_EPISODES", 5000);
    private static final int LADDER_GAMES_PER_MATCHUP = EnvConfig.i32("LADDER_GAMES_PER_MATCHUP", 6);
    private static final double LADDER_MIX_LOWER_P = EnvConfig.f64("LADDER_MIX_LOWER_P", 0.20);
    private static final String SKILL_MIX = EnvConfig.str("SKILL_MIX", "1:1.0");
    private static final double HYBRID_SELFPLAY_P = EnvConfig.f64("HYBRID_SELFPLAY_P", 0.25);
    private static final double META_HYBRID_META_P = EnvConfig.f64("META_HYBRID_META_P", 0.75);
    private static final boolean SELFPLAY_OPPONENT_TRAINING = EnvConfig.bool("SELFPLAY_OPPONENT_TRAINING", true);

    private static final Object LEAGUE_LOCK = new Object();
    private static LeagueState LEAGUE_STATE = null;
    private static final AtomicInteger LEAGUE_LAST_TICK_EP = new AtomicInteger(0);
    private static final ThreadLocal<Path> THREAD_LOCAL_OPPONENT_DECK_OVERRIDE = new ThreadLocal<>();
    private static final Object CROSS_PROFILE_CACHE_LOCK = new Object();
    private static volatile long CROSS_PROFILE_CACHE_AT_MS = 0L;
    private static volatile java.util.List<CrossProfileSnapshot> CROSS_PROFILE_CACHE = java.util.Collections.emptyList();
    private static final Object LEAGUE_META_CACHE_LOCK = new Object();
    private static volatile long LEAGUE_META_CACHE_AT_MS = 0L;
    private static volatile java.util.List<LeagueMetaOpponentCandidate> LEAGUE_META_CACHE = java.util.Collections.emptyList();
    private static final String MODEL_PROFILE_NAME = EnvConfig.str("MODEL_PROFILE", "").trim();
    
    private static final Object LADDER_LOCK = new Object();
    private static LadderState LADDER_STATE = null;
    private static final AtomicInteger LADDER_LAST_TICK_EP = new AtomicInteger(0);

    static {
        // Ensure we have at least a console appender so INFO logs are visible
        try {
            if (!LogManager.getRootLogger().getAllAppenders().hasMoreElements()) {
                BasicConfigurator.configure();
            }
        } catch (Exception ignored) {
            // ignore
        }

        // Keep console output readable:
        // - default to WARN everywhere (XMage can be very chatty),
        // - allow override via MTG_AI_LOG_LEVEL (OFF/ERROR/WARN/INFO/DEBUG/TRACE).
        Level baseLevel = parseLog4jLevel(System.getenv("MTG_AI_LOG_LEVEL"), Level.WARN);
        Logger root = LogManager.getRootLogger();
        root.setLevel(baseLevel);

        // Our own components: respect MTG_AI_LOG_LEVEL (default WARN).
        Logger.getLogger(RLTrainer.class).setLevel(baseLevel);
        Logger.getLogger(MetricsCollector.class).setLevel(baseLevel);

        // Noisy engine components: keep to ERROR unless user opts into verbose logs.
        Logger.getLogger("mage.game").setLevel(Level.ERROR);
        Logger.getLogger("mage.game.GameImpl").setLevel(Level.ERROR);
        Logger.getLogger("mage.server").setLevel(Level.ERROR);

        // Suppress "AI player thinks too long" warnings from opponent AIs during training
        Logger.getLogger("mage.player.ai.ComputerPlayer6").setLevel(Level.ERROR);
    }

    static final class LeagueState {

        boolean promoted;
        int lastTickEpisode;
        String championPolicyKey; // e.g. "snap:league_ep_5000.pt"
        final java.util.LinkedList<String> recent = new java.util.LinkedList<>();
        final java.util.LinkedList<String> pool = new java.util.LinkedList<>(); // all pool members (includes recent/champions)
        final java.util.HashMap<String, Double> baselineWr = new java.util.HashMap<>();

        String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"promoted\": ").append(promoted ? "true" : "false").append(",\n");
            sb.append("  \"lastTickEpisode\": ").append(lastTickEpisode).append(",\n");
            sb.append("  \"championPolicyKey\": ").append(jsonString(championPolicyKey)).append(",\n");
            sb.append("  \"recent\": ").append(jsonStringArray(recent)).append(",\n");
            sb.append("  \"pool\": ").append(jsonStringArray(pool)).append(",\n");
            sb.append("  \"baselineWr\": ").append(jsonDoubleMap(baselineWr)).append("\n");
            sb.append("}\n");
            return sb.toString();
        }

        static LeagueState fromJson(String s) {
            LeagueState st = new LeagueState();
            if (s == null) {
                return st;
            }
            st.promoted = jsonBool(s, "promoted", false);
            st.lastTickEpisode = jsonInt(s, "lastTickEpisode", 0);
            st.championPolicyKey = jsonStringField(s, "championPolicyKey", null);
            st.recent.addAll(jsonStringArrayField(s, "recent"));
            st.pool.addAll(jsonStringArrayField(s, "pool"));
            st.baselineWr.putAll(jsonDoubleMapField(s, "baselineWr"));
            // ensure pool contains recent (best-effort)
            for (String r : st.recent) {
                if (r != null && !r.isEmpty() && !st.pool.contains(r)) {
                    st.pool.add(r);
                }
            }
            return st;
        }
    }

    static final class LadderState {
        int currentTier;
        int lastTickEpisode;
        final java.util.HashMap<Integer, Double> tierWinrates = new java.util.HashMap<>();
        final java.util.ArrayList<String> evalHistory = new java.util.ArrayList<>();

        String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"currentTier\": ").append(currentTier).append(",\n");
            sb.append("  \"lastTickEpisode\": ").append(lastTickEpisode).append(",\n");
            sb.append("  \"tierWinrates\": ").append(jsonIntDoubleMap(tierWinrates)).append(",\n");
            sb.append("  \"evalHistory\": ").append(jsonStringArray(evalHistory)).append("\n");
            sb.append("}\n");
            return sb.toString();
        }

        static LadderState fromJson(String s) {
            LadderState ls = new LadderState();
            if (s == null) {
                return ls;
            }
            ls.currentTier = jsonInt(s, "currentTier", 0);
            ls.lastTickEpisode = jsonInt(s, "lastTickEpisode", 0);
            ls.tierWinrates.putAll(jsonIntDoubleMapField(s, "tierWinrates"));
            ls.evalHistory.addAll(jsonStringArrayField(s, "evalHistory"));
            return ls;
        }
    }

    private static final class LeagueRegistryEntry {
        String profile;
        String deckPath;
        boolean active;
        String notes;
    }

    private static final class CrossProfileSnapshot {
        final String profile;
        final Path deckPath;
        final Path snapshotPath;
        final int episode;

        private CrossProfileSnapshot(String profile, Path deckPath, Path snapshotPath, int episode) {
            this.profile = profile;
            this.deckPath = deckPath;
            this.snapshotPath = snapshotPath;
            this.episode = episode;
        }
    }

    private static final class LeagueMetaOpponentCandidate {
        final String profile;
        final Path deckPath;
        final Path snapshotPath; // nullable when not yet qualified
        final int episode;
        final boolean promoted;
        final double baselineWr;
        final boolean qualified;

        private LeagueMetaOpponentCandidate(
                String profile,
                Path deckPath,
                Path snapshotPath,
                int episode,
                boolean promoted,
                double baselineWr,
                boolean qualified
        ) {
            this.profile = profile;
            this.deckPath = deckPath;
            this.snapshotPath = snapshotPath;
            this.episode = episode;
            this.promoted = promoted;
            this.baselineWr = baselineWr;
            this.qualified = qualified;
        }
    }

    private static final class LeagueEvalEntrant {
        final String profile;
        final Path deckPath;
        final Path snapshotPath;
        final int episode;

        private LeagueEvalEntrant(String profile, Path deckPath, Path snapshotPath, int episode) {
            this.profile = profile;
            this.deckPath = deckPath;
            this.snapshotPath = snapshotPath;
            this.episode = episode;
        }
    }

    private static String jsonString(String s) {
        if (s == null) {
            return "null";
        }
        String v = s.replace("\\", "\\\\").replace("\"", "\\\"");
        return "\"" + v + "\"";
    }

    private static String jsonStringArray(java.util.List<String> items) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        boolean first = true;
        for (String it : items) {
            if (it == null) {
                continue;
            }
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append(jsonString(it));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String jsonDoubleMap(java.util.Map<String, Double> m) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        
        // Sort entries by episode number extracted from key (e.g., "snap:league_ep_5000.pt" -> 5000)
        java.util.List<java.util.Map.Entry<String, Double>> entries = new java.util.ArrayList<>(m.entrySet());
        entries.sort((a, b) -> {
            int epA = extractEpisodeNumber(a.getKey());
            int epB = extractEpisodeNumber(b.getKey());
            return Integer.compare(epA, epB);
        });
        
        boolean first = true;
        for (java.util.Map.Entry<String, Double> e : entries) {
            if (e.getKey() == null) {
                continue;
            }
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append(jsonString(e.getKey())).append(": ").append(e.getValue() == null ? "0.0" : e.getValue());
        }
        sb.append("}");
        return sb.toString();
    }

    private static int extractEpisodeNumber(String key) {
        // Extract episode number from keys like "snap:league_ep_5000.pt"
        if (key == null || !key.contains("_ep_")) {
            return Integer.MAX_VALUE; // Put non-episode keys at the end
        }
        try {
            int start = key.indexOf("_ep_") + 4;
            int end = key.indexOf(".", start);
            if (end < 0) {
                end = key.length();
            }
            return Integer.parseInt(key.substring(start, end));
        } catch (Exception e) {
            return Integer.MAX_VALUE;
        }
    }

    private static boolean jsonBool(String json, String key, boolean def) {
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return def;
            }
            int c = json.indexOf(":", i);
            if (c < 0) {
                return def;
            }
            String tail = json.substring(c + 1).trim();
            if (tail.startsWith("true")) {
                return true;
            }
            if (tail.startsWith("false")) {
                return false;
            }
            return def;
        } catch (Exception ignored) {
            return def;
        }
    }

    private static int jsonInt(String json, String key, int def) {
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return def;
            }
            int c = json.indexOf(":", i);
            if (c < 0) {
                return def;
            }
            int end = c + 1;
            while (end < json.length() && (json.charAt(end) == ' ' || json.charAt(end) == '\t')) {
                end++;
            }
            int j = end;
            while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '-')) {
                j++;
            }
            if (j <= end) {
                return def;
            }
            return Integer.parseInt(json.substring(end, j));
        } catch (Exception ignored) {
            return def;
        }
    }

    private static double jsonDouble(String json, String key, double def) {
        try {
            String pattern = "\\\"" + java.util.regex.Pattern.quote(key)
                    + "\\\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)";
            java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
            if (!m.find()) {
                return def;
            }
            return Double.parseDouble(m.group(1));
        } catch (Exception ignored) {
            return def;
        }
    }

    private static String jsonStringField(String json, String key, String def) {
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return def;
            }
            int c = json.indexOf(":", i);
            if (c < 0) {
                return def;
            }
            int q1 = json.indexOf("\"", c + 1);
            if (q1 < 0) {
                // null?
                String tail = json.substring(c + 1).trim();
                return tail.startsWith("null") ? null : def;
            }
            int q2 = json.indexOf("\"", q1 + 1);
            if (q2 < 0) {
                return def;
            }
            return json.substring(q1 + 1, q2).replace("\\\"", "\"").replace("\\\\", "\\");
        } catch (Exception ignored) {
            return def;
        }
    }

    private static java.util.List<String> jsonStringArrayField(String json, String key) {
        java.util.ArrayList<String> out = new java.util.ArrayList<>();
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return out;
            }
            int c = json.indexOf("[", i);
            if (c < 0) {
                return out;
            }
            int e = json.indexOf("]", c);
            if (e < 0) {
                return out;
            }
            String body = json.substring(c + 1, e);
            int p = 0;
            while (p < body.length()) {
                int q1 = body.indexOf("\"", p);
                if (q1 < 0) {
                    break;
                }
                int q2 = body.indexOf("\"", q1 + 1);
                if (q2 < 0) {
                    break;
                }
                out.add(body.substring(q1 + 1, q2).replace("\\\"", "\"").replace("\\\\", "\\"));
                p = q2 + 1;
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    private static java.util.Map<String, Double> jsonDoubleMapField(String json, String key) {
        java.util.HashMap<String, Double> out = new java.util.HashMap<>();
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return out;
            }
            int c = json.indexOf("{", i);
            if (c < 0) {
                return out;
            }
            int e = json.indexOf("}", c);
            if (e < 0) {
                return out;
            }
            String body = json.substring(c + 1, e);
            // Very simple parser: "key": number
            int p = 0;
            while (p < body.length()) {
                int q1 = body.indexOf("\"", p);
                if (q1 < 0) {
                    break;
                }
                int q2 = body.indexOf("\"", q1 + 1);
                if (q2 < 0) {
                    break;
                }
                String mk = body.substring(q1 + 1, q2).replace("\\\"", "\"").replace("\\\\", "\\");
                int colon = body.indexOf(":", q2 + 1);
                if (colon < 0) {
                    break;
                }
                int n0 = colon + 1;
                while (n0 < body.length() && (body.charAt(n0) == ' ' || body.charAt(n0) == '\t')) {
                    n0++;
                }
                int n1 = n0;
                while (n1 < body.length() && ("-0123456789.eE".indexOf(body.charAt(n1)) >= 0)) {
                    n1++;
                }
                if (n1 > n0) {
                    try {
                        out.put(mk, Double.parseDouble(body.substring(n0, n1)));
                    } catch (NumberFormatException ignored) {
                    }
                }
                p = n1 + 1;
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    private static String jsonIntDoubleMap(java.util.Map<Integer, Double> m) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        java.util.List<Integer> keys = new java.util.ArrayList<>(m.keySet());
        keys.sort(Integer::compareTo);
        boolean first = true;
        for (Integer k : keys) {
            if (k == null) {
                continue;
            }
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append("\"").append(k).append("\": ").append(m.get(k) == null ? "0.0" : m.get(k));
        }
        sb.append("}");
        return sb.toString();
    }

    private static java.util.Map<Integer, Double> jsonIntDoubleMapField(String json, String key) {
        java.util.HashMap<Integer, Double> out = new java.util.HashMap<>();
        try {
            String k = "\"" + key + "\"";
            int i = json.indexOf(k);
            if (i < 0) {
                return out;
            }
            int c = json.indexOf("{", i);
            if (c < 0) {
                return out;
            }
            int e = json.indexOf("}", c);
            if (e < 0) {
                return out;
            }
            String body = json.substring(c + 1, e);
            // Simple parser: "intKey": number
            int p = 0;
            while (p < body.length()) {
                int q1 = body.indexOf("\"", p);
                if (q1 < 0) {
                    break;
                }
                int q2 = body.indexOf("\"", q1 + 1);
                if (q2 < 0) {
                    break;
                }
                String mk = body.substring(q1 + 1, q2);
                int colon = body.indexOf(":", q2 + 1);
                if (colon < 0) {
                    break;
                }
                int n0 = colon + 1;
                while (n0 < body.length() && (body.charAt(n0) == ' ' || body.charAt(n0) == '\t')) {
                    n0++;
                }
                int n1 = n0;
                while (n1 < body.length() && ("-0123456789.eE".indexOf(body.charAt(n1)) >= 0)) {
                    n1++;
                }
                if (n1 > n0) {
                    try {
                        int intKey = Integer.parseInt(mk);
                        double value = Double.parseDouble(body.substring(n0, n1));
                        out.put(intKey, value);
                    } catch (NumberFormatException ignored) {
                    }
                }
                p = n1 + 1;
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    private Path leagueStatePath() {
        return Paths.get(profileContext != null ? profileContext.paths.leagueStatePath : RLLogPaths.LEAGUE_STATE_PATH);
    }

    private Path leagueEventsLogPath() {
        return Paths.get(profileContext != null ? profileContext.paths.leagueEventsLogPath : RLLogPaths.LEAGUE_EVENTS_LOG_PATH);
    }

    private Path leagueStatusPath() {
        return Paths.get(profileContext != null ? profileContext.paths.leagueStatusPath : RLLogPaths.LEAGUE_STATUS_PATH);
    }

    private void appendLeagueEvent(String line) {
        if (line == null || line.trim().isEmpty()) {
            return;
        }
        try {
            Path p = leagueEventsLogPath();
            if (p.getParent() != null) {
                Files.createDirectories(p.getParent());
            }
            String stamped = java.time.LocalDateTime.now().toString() + " " + line.trim() + System.lineSeparator();
            Files.write(p, stamped.getBytes(StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND);
        } catch (Exception ignored) {
        }
    }

    private void writeLeagueStatus(LeagueState st) {
        if (st == null) {
            return;
        }
        try {
            Path p = leagueStatusPath();
            if (p.getParent() != null) {
                Files.createDirectories(p.getParent());
            }
            java.util.List<String> champs = computeChampions(st);
            StringBuilder sb = new StringBuilder();
            sb.append("LEAGUE_STATUS\n");
            sb.append("updated=").append(java.time.LocalDateTime.now().toString()).append('\n');
            sb.append("promoted=").append(st.promoted).append('\n');
            sb.append("lastTickEpisode=").append(st.lastTickEpisode).append('\n');
            sb.append("champion=").append(st.championPolicyKey == null ? "" : st.championPolicyKey).append('\n');
            sb.append("poolSize=").append(st.pool.size()).append('\n');
            sb.append("recent=").append(st.recent.toString()).append('\n');
            sb.append("champions=").append(champs.toString()).append('\n');
            sb.append("baselineWrTop\n");
            for (String k : champs) {
                Double wr = st.baselineWr.get(k);
                sb.append("  ").append(k).append(" wr=").append(wr == null ? 0.0 : wr).append('\n');
            }
            Files.write(p, sb.toString().getBytes(StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception ignored) {
        }
    }

    private static Path leagueRegistryPath() {
        return Paths.get(LEAGUE_REGISTRY_PATH);
    }

    private static Path profileAgentStatusPath(String profile) {
        String p = profile == null ? "" : profile.trim();
        if (p.isEmpty()) {
            return Paths.get(RLLogPaths.LOGS_BASE_DIR, "league", "agent_status.json");
        }
        return Paths.get("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles",
                p, "logs", "league", "agent_status.json");
    }

    private static Path profileSnapshotsDir(String profile) {
        String p = profile == null ? "" : profile.trim();
        if (p.isEmpty()) {
            return Paths.get(RLLogPaths.SNAPSHOT_DIR);
        }
        return Paths.get("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles",
                p, "models", "snapshots");
    }

    private static Path profileModelsDir(String profile) {
        String p = profile == null ? "" : profile.trim();
        if (p.isEmpty()) {
            return Paths.get(RLLogPaths.MODELS_BASE_DIR);
        }
        return Paths.get("Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles",
                p, "models");
    }

    private static Path findLatestLeagueSnapshotPathForProfile(String profile) {
        try {
            Path dir = profileSnapshotsDir(profile);
            if (!Files.isDirectory(dir)) {
                return null;
            }
            java.util.List<Path> snaps;
            try (java.util.stream.Stream<Path> stream = Files.list(dir)) {
                snaps = stream
                        .filter(Files::isRegularFile)
                        .filter(p -> {
                            String n = p.getFileName().toString().toLowerCase();
                            return n.endsWith(".pt") && n.contains("league_ep_");
                        })
                        .collect(Collectors.toList());
            }
            if (snaps.isEmpty()) {
                return null;
            }
            snaps.sort((a, b) -> {
                int ea = extractEpisodeNumber("snap:" + a.getFileName().toString());
                int eb = extractEpisodeNumber("snap:" + b.getFileName().toString());
                if (ea != eb) {
                    return Integer.compare(eb, ea);
                }
                try {
                    long ta = Files.getLastModifiedTime(a).toMillis();
                    long tb = Files.getLastModifiedTime(b).toMillis();
                    return Long.compare(tb, ta);
                } catch (Exception ignored) {
                    return 0;
                }
            });
            return snaps.get(0).toAbsolutePath().normalize();
        } catch (Exception ignored) {
            return null;
        }
    }

    private static Path findLatestPolicyPathForProfile(String profile) {
        Path leagueSnapshot = findLatestLeagueSnapshotPathForProfile(profile);
        if (leagueSnapshot != null && Files.isRegularFile(leagueSnapshot)) {
            return leagueSnapshot.toAbsolutePath().normalize();
        }
        Path modelsDir = profileModelsDir(profile);
        Path latest = modelsDir.resolve("model_latest.pt");
        if (Files.isRegularFile(latest)) {
            return latest.toAbsolutePath().normalize();
        }
        Path model = modelsDir.resolve("model.pt");
        if (Files.isRegularFile(model)) {
            return model.toAbsolutePath().normalize();
        }
        return null;
    }

    private void writeAgentStatus(int episodeNum) {
        try {
            LeagueState st = getLeagueState();
            String profile = profileName().isEmpty() ? "default" : profileName();
            String latestKey = null;
            int latestEp = Integer.MIN_VALUE;
            synchronized (leagueLock()) {
                for (String k : st.baselineWr.keySet()) {
                    int ep = extractEpisodeNumber(k);
                    if (ep > latestEp) {
                        latestEp = ep;
                        latestKey = k;
                    }
                }
            }
            double baselineWr = 0.0;
            if (latestKey != null) {
                Double wr = st.baselineWr.get(latestKey);
                baselineWr = wr == null ? 0.0 : wr;
            }
            Path latestSnapshot = findLatestPolicyPathForProfile(profile);
            Path statusPath = Paths.get(logsBaseDir(), "league", "agent_status.json");
            if (statusPath.getParent() != null) {
                Files.createDirectories(statusPath.getParent());
            }
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"profile\": ").append(jsonString(profile)).append(",\n");
            sb.append("  \"episode\": ").append(episodeNum).append(",\n");
            sb.append("  \"promoted\": ").append(st.promoted ? "true" : "false").append(",\n");
            sb.append("  \"baseline_wr\": ").append(String.format(java.util.Locale.US, "%.6f", baselineWr)).append(",\n");
            sb.append("  \"latest_snapshot_path\": ").append(jsonString(latestSnapshot == null ? "" : latestSnapshot.toString())).append(",\n");
            sb.append("  \"updated_at\": ").append(jsonString(java.time.Instant.now().toString())).append("\n");
            sb.append("}\n");
            Files.write(statusPath, sb.toString().getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Failed to write agent_status.json: " + e.getMessage());
        }
    }

    private static java.util.List<LeagueRegistryEntry> loadLeagueRegistryEntries() {
        java.util.ArrayList<LeagueRegistryEntry> out = new java.util.ArrayList<>();
        try {
            Path p = leagueRegistryPath();
            if (!Files.exists(p)) {
                return out;
            }
            String s = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("\\{[^\\{\\}]*\\}").matcher(s);
            while (m.find()) {
                String obj = m.group();
                String profile = jsonStringField(obj, "profile", "").trim();
                String deckPath = jsonStringField(obj, "deck_path", "").trim();
                boolean active = jsonBool(obj, "active", true);
                if (profile.isEmpty() || deckPath.isEmpty()) {
                    continue;
                }
                LeagueRegistryEntry e = new LeagueRegistryEntry();
                e.profile = profile;
                e.deckPath = deckPath;
                e.active = active;
                e.notes = jsonStringField(obj, "notes", "");
                out.add(e);
            }
        } catch (Exception e) {
            logger.warn("Failed to load league registry: " + e.getMessage());
        }
        return out;
    }

    private static Path resolveDeckPath(String rawDeckPath, Path baseDir) {
        if (rawDeckPath == null || rawDeckPath.trim().isEmpty()) {
            return null;
        }
        Path p = Paths.get(rawDeckPath.trim());
        if (p.isAbsolute()) {
            return p.normalize();
        }

        // First try relative to provided base (registry or status file location).
        Path baseResolved = (baseDir != null ? baseDir : Paths.get(System.getProperty("user.dir")))
                .resolve(p).normalize();
        if (Files.exists(baseResolved)) {
            return baseResolved;
        }

        // Fallback for repo-root relative paths in config files.
        Path cwdResolved = Paths.get(System.getProperty("user.dir")).resolve(p).normalize();
        if (Files.exists(cwdResolved)) {
            return cwdResolved;
        }

        // Return the base-resolved value for diagnostics even if missing.
        return baseResolved;
    }

    private static java.util.List<CrossProfileSnapshot> getCrossProfileCandidates() {
        long now = System.currentTimeMillis();
        if (now - CROSS_PROFILE_CACHE_AT_MS < Math.max(1000, LEAGUE_CROSS_PROFILE_REFRESH_MS)) {
            return CROSS_PROFILE_CACHE;
        }
        synchronized (CROSS_PROFILE_CACHE_LOCK) {
            now = System.currentTimeMillis();
            if (now - CROSS_PROFILE_CACHE_AT_MS < Math.max(1000, LEAGUE_CROSS_PROFILE_REFRESH_MS)) {
                return CROSS_PROFILE_CACHE;
            }
            java.util.ArrayList<CrossProfileSnapshot> loaded = new java.util.ArrayList<>();
            Path registry = leagueRegistryPath();
            Path baseDir = registry.toAbsolutePath().getParent();
            for (LeagueRegistryEntry e : loadLeagueRegistryEntries()) {
                if (!e.active) {
                    continue;
                }
                if (!MODEL_PROFILE_NAME.isEmpty() && MODEL_PROFILE_NAME.equalsIgnoreCase(e.profile)) {
                    continue;
                }
                Path statusPath = profileAgentStatusPath(e.profile);
                if (!Files.exists(statusPath)) {
                    continue;
                }
                try {
                    String status = new String(Files.readAllBytes(statusPath), StandardCharsets.UTF_8);
                    String snapPathRaw = jsonStringField(status, "latest_snapshot_path", "").trim();
                    int episode = jsonInt(status, "episode", 0);
                    Path snapPath = resolveDeckPath(snapPathRaw, statusPath.toAbsolutePath().getParent());
                    if (snapPath == null || !Files.isRegularFile(snapPath)) {
                        snapPath = findLatestPolicyPathForProfile(e.profile);
                    }
                    Path deckPath = resolveDeckPath(e.deckPath, baseDir);
                    if (snapPath == null || deckPath == null || !Files.isRegularFile(snapPath) || !Files.isRegularFile(deckPath)) {
                        continue;
                    }
                    loaded.add(new CrossProfileSnapshot(
                            e.profile,
                            deckPath.toAbsolutePath().normalize(),
                            snapPath.toAbsolutePath().normalize(),
                            episode
                    ));
                } catch (Exception ignored) {
                }
            }
            CROSS_PROFILE_CACHE = java.util.Collections.unmodifiableList(loaded);
            CROSS_PROFILE_CACHE_AT_MS = now;
            return CROSS_PROFILE_CACHE;
        }
    }

    private static java.util.List<LeagueMetaOpponentCandidate> getLeagueMetaOpponentCandidates() {
        long now = System.currentTimeMillis();
        if (now - LEAGUE_META_CACHE_AT_MS < Math.max(1000, LEAGUE_CROSS_PROFILE_REFRESH_MS)) {
            return LEAGUE_META_CACHE;
        }
        synchronized (LEAGUE_META_CACHE_LOCK) {
            now = System.currentTimeMillis();
            if (now - LEAGUE_META_CACHE_AT_MS < Math.max(1000, LEAGUE_CROSS_PROFILE_REFRESH_MS)) {
                return LEAGUE_META_CACHE;
            }
            java.util.ArrayList<LeagueMetaOpponentCandidate> loaded = new java.util.ArrayList<>();
            Path registry = leagueRegistryPath();
            Path baseDir = registry.toAbsolutePath().getParent();
            for (LeagueRegistryEntry e : loadLeagueRegistryEntries()) {
                if (!e.active) {
                    continue;
                }
                String profile = e.profile == null ? "" : e.profile.trim();
                if (profile.isEmpty()) {
                    continue;
                }

                Path deckPath = resolveDeckPath(e.deckPath, baseDir);
                if (deckPath == null || !Files.isRegularFile(deckPath)) {
                    continue;
                }

                Path statusPath = profileAgentStatusPath(profile);
                boolean promoted = false;
                double baselineWr = 0.0;
                int episode = 0;
                Path snapPath = null;
                if (Files.exists(statusPath)) {
                    try {
                        String status = new String(Files.readAllBytes(statusPath), StandardCharsets.UTF_8);
                        promoted = jsonBool(status, "promoted", false);
                        baselineWr = jsonDouble(status, "baseline_wr", 0.0);
                        episode = jsonInt(status, "episode", 0);
                        String snapPathRaw = jsonStringField(status, "latest_snapshot_path", "").trim();
                        snapPath = resolveDeckPath(snapPathRaw, statusPath.toAbsolutePath().getParent());
                    } catch (Exception ignored) {
                    }
                }
                if (snapPath == null || !Files.isRegularFile(snapPath)) {
                    snapPath = findLatestPolicyPathForProfile(profile);
                }
                boolean hasSnapshot = snapPath != null && Files.isRegularFile(snapPath);
                boolean qualified = hasSnapshot && (promoted || baselineWr >= LEAGUE_PROMOTE_WR);
                loaded.add(new LeagueMetaOpponentCandidate(
                        profile,
                        deckPath.toAbsolutePath().normalize(),
                        hasSnapshot ? snapPath.toAbsolutePath().normalize() : null,
                        episode,
                        promoted,
                        baselineWr,
                        qualified
                ));
            }
            loaded.sort((a, b) -> a.profile.compareToIgnoreCase(b.profile));
            LEAGUE_META_CACHE = java.util.Collections.unmodifiableList(loaded);
            LEAGUE_META_CACHE_AT_MS = now;
            return LEAGUE_META_CACHE;
        }
    }

    // ============================================================
    // Ladder state management
    // ============================================================
    
    private Path ladderStatePath() {
        return Paths.get(logsBaseDir(), "league", "ladder_state.json");
    }

    private Path ladderStatusPath() {
        return Paths.get(logsBaseDir(), "league", "ladder_status.txt");
    }

    private LadderState getLadderState() {
        synchronized (ladderLock()) {
            LadderState ls;
            if (profileContext != null) {
                ls = profileContext.ladderState;
            } else {
                ls = LADDER_STATE;
            }
            if (ls != null) {
                return ls;
            }
            ls = new LadderState();
            try {
                Path p = ladderStatePath();
                if (Files.exists(p)) {
                    String s = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                    ls = LadderState.fromJson(s);
                }
            } catch (Exception ignored) {
            }
            if (profileContext != null) {
                profileContext.ladderState = ls;
            } else {
                LADDER_STATE = ls;
            }
            return ls;
        }
    }

    private void saveLadderState() {
        synchronized (ladderLock()) {
            LadderState ls = profileContext != null ? profileContext.ladderState : LADDER_STATE;
            if (ls == null) {
                return;
            }
            try {
                Path p = ladderStatePath();
                if (p.getParent() != null) {
                    Files.createDirectories(p.getParent());
                }
                Files.write(p, ls.toJson().getBytes(StandardCharsets.UTF_8),
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
            } catch (Exception ignored) {
            }
        }
    }

    private void writeLadderStatus(LadderState ls, int[] tiers) {
        if (ls == null || tiers == null) {
            return;
        }
        try {
            Path p = ladderStatusPath();
            if (p.getParent() != null) {
                Files.createDirectories(p.getParent());
            }
            StringBuilder sb = new StringBuilder();
            sb.append("LADDER_STATUS\n");
            sb.append("updated=").append(java.time.LocalDateTime.now().toString()).append('\n');
            sb.append("currentTier=").append(ls.currentTier).append('\n');
            sb.append("currentSkill=").append(ls.currentTier < tiers.length ? tiers[ls.currentTier] : -1).append('\n');
            sb.append("lastTickEpisode=").append(ls.lastTickEpisode).append('\n');
            sb.append("tierWinrates:\n");
            for (int i = 0; i < tiers.length; i++) {
                Double wr = ls.tierWinrates.get(i);
                sb.append("  tier=").append(i).append(" skill=").append(tiers[i])
                  .append(" wr=").append(wr == null ? "n/a" : String.format("%.3f", wr)).append('\n');
            }
            sb.append("evalHistory:\n");
            for (String entry : ls.evalHistory) {
                sb.append("  ").append(entry).append('\n');
            }
            Files.write(p, sb.toString().getBytes(StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception ignored) {
        }
    }

    private LeagueState getLeagueState() {
        synchronized (leagueLock()) {
            LeagueState st;
            if (profileContext != null) {
                st = profileContext.leagueState;
            } else {
                st = LEAGUE_STATE;
            }
            if (st != null) {
                return st;
            }
            st = new LeagueState();
            try {
                Path p = leagueStatePath();
                if (Files.exists(p)) {
                    String s = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                    st = LeagueState.fromJson(s);
                }
            } catch (Exception ignored) {
            }
            if (profileContext != null) {
                profileContext.leagueState = st;
            } else {
                LEAGUE_STATE = st;
            }
            return st;
        }
    }

    private void saveLeagueState() {
        synchronized (leagueLock()) {
            LeagueState st = profileContext != null ? profileContext.leagueState : LEAGUE_STATE;
            if (st == null) {
                return;
            }
            try {
                Path p = leagueStatePath();
                if (p.getParent() != null) {
                    Files.createDirectories(p.getParent());
                }
                Files.write(p, st.toJson().getBytes(StandardCharsets.UTF_8),
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
            } catch (Exception ignored) {
            }
        }
    }

    private String leagueSnapshotFileName(int episode) {
        String profile = profileName() == null ? "" : profileName().trim();
        if (profile.isEmpty()) {
            return "league_ep_" + episode + ".pt";
        }
        String safe = profile.replaceAll("[^A-Za-z0-9._-]", "_");
        return safe + "_league_ep_" + episode + ".pt";
    }

    private String leagueSnapshotPolicyKey(int episode) {
        return "snap:" + leagueSnapshotFileName(episode);
    }

    private static void addRecent(LeagueState st, String snapKey) {
        if (st == null || snapKey == null || snapKey.trim().isEmpty()) {
            return;
        }
        String k = snapKey.trim();
        st.recent.remove(k);
        st.recent.addFirst(k);
        while (st.recent.size() > Math.max(0, LEAGUE_POOL_RECENT)) {
            st.recent.removeLast();
        }
    }

    private static void addToPool(LeagueState st, String snapKey) {
        if (st == null || snapKey == null || snapKey.trim().isEmpty()) {
            return;
        }
        String k = snapKey.trim();
        if (!st.pool.contains(k)) {
            st.pool.addFirst(k);
        }
    }

    private static java.util.List<String> computeChampions(LeagueState st) {
        java.util.ArrayList<java.util.Map.Entry<String, Double>> entries = new java.util.ArrayList<>(st.baselineWr.entrySet());
        java.util.Collections.sort(entries, (a, b) -> {
            double da = a.getValue() == null ? 0.0 : a.getValue();
            double db = b.getValue() == null ? 0.0 : b.getValue();
            return Double.compare(db, da);
        });
        java.util.ArrayList<String> champs = new java.util.ArrayList<>();
        int k = Math.max(0, LEAGUE_POOL_CHAMPIONS);
        for (int i = 0; i < entries.size() && champs.size() < k; i++) {
            String key = entries.get(i).getKey();
            if (key != null && st.pool.contains(key)) {
                champs.add(key);
            }
        }
        return champs;
    }

    private static void prunePool(LeagueState st, Random rand) {
        if (st == null) {
            return;
        }
        int max = Math.max(0, LEAGUE_POOL_MAX);
        if (st.pool.size() <= max) {
            return;
        }
        java.util.HashSet<String> keep = new java.util.HashSet<>();
        // Champions
        for (String c : computeChampions(st)) {
            if (c != null) {
                keep.add(c);
            }
        }
        // Recent
        for (String r : st.recent) {
            if (r != null) {
                keep.add(r);
            }
        }
        // Champion policyKey should survive if set
        if (st.championPolicyKey != null) {
            keep.add(st.championPolicyKey);
        }

        // Fill remaining slots with random survivors from the rest
        java.util.ArrayList<String> rest = new java.util.ArrayList<>();
        for (String k : st.pool) {
            if (k != null && !keep.contains(k)) {
                rest.add(k);
            }
        }
        java.util.Collections.shuffle(rest, rand);
        for (String k : rest) {
            if (keep.size() >= max) {
                break;
            }
            keep.add(k);
        }

        // Rebuild pool in a stable-ish order: champions, recent, then remaining keepers
        java.util.LinkedList<String> newPool = new java.util.LinkedList<>();
        java.util.HashSet<String> seen = new java.util.HashSet<>();
        for (String c : computeChampions(st)) {
            if (c != null && keep.contains(c) && seen.add(c)) {
                newPool.add(c);
            }
        }
        for (String r : st.recent) {
            if (r != null && keep.contains(r) && seen.add(r)) {
                newPool.add(r);
            }
        }
        for (String k : keep) {
            if (k != null && seen.add(k)) {
                newPool.add(k);
            }
        }
        st.pool.clear();
        st.pool.addAll(newPool);
    }

    private void maybeRunLeagueTick(int episodeNum) {
        if (LEAGUE_TICK_EPISODES <= 0) {
            return;
        }
        if (episodeNum <= 0 || (episodeNum % LEAGUE_TICK_EPISODES) != 0) {
            return;
        }
        // Only in league mode
        String mode = OPPONENT_SAMPLER == null ? "league" : OPPONENT_SAMPLER.trim().toLowerCase();
        if (!"league".equals(mode)) {
            return;
        }
        // Single-thread the tick across runners
        int prev = leagueLastTickEp().get();
        if (episodeNum <= prev) {
            return;
        }
        if (!leagueLastTickEp().compareAndSet(prev, episodeNum)) {
            return;
        }

        final long tickStartMs = System.currentTimeMillis();
        final Random rand = newSeededRandom(100L + episodeNum);
        LeagueState st = getLeagueState();

        logger.info("League tick start: ep=" + episodeNum + " tickEvery=" + LEAGUE_TICK_EPISODES
                + " decklist=" + (LEAGUE_BASELINE_DECKLIST_FILE == null ? "" : LEAGUE_BASELINE_DECKLIST_FILE)
                + " gpm=" + LEAGUE_BASELINE_GAMES_PER_MATCHUP
                + " promoteWR=" + String.format("%.3f", LEAGUE_PROMOTE_WR)
                + " poolFloorWR=" + String.format("%.3f", LEAGUE_POOL_FLOOR_WR)
                + " champPromoteWR=" + String.format("%.3f", LEAGUE_CHAMPION_PROMOTE_WR));
        appendLeagueEvent("tick_start ep=" + episodeNum
                + " decklist=" + (LEAGUE_BASELINE_DECKLIST_FILE == null ? "" : LEAGUE_BASELINE_DECKLIST_FILE)
                + " gpm=" + LEAGUE_BASELINE_GAMES_PER_MATCHUP);

        // 1) Save snapshot S_t (stable policyKey = file name)
        String snapFile = leagueSnapshotFileName(episodeNum);
        Path snapPath = Paths.get(snapshotDir(), snapFile);
        try {
            if (snapPath.getParent() != null) {
                Files.createDirectories(snapPath.getParent());
            }
        } catch (Exception ignored) {
        }
        // Snapshot save handled autonomously by Python snapshot_manager
        String snapKey = "snap:" + snapFile;
        logger.info("League tick snapshot saved: ep=" + episodeNum + " key=" + snapKey + " path=" + snapPath);
        appendLeagueEvent("snapshot_saved ep=" + episodeNum + " key=" + snapKey + " path=" + snapPath.toString());

        // 2) Baseline eval: S_t vs CP7Skill1 using deck-matrix benchmark (GamesPerMatchup=3 effect)
        double baselineWr = runLeagueBenchmarkPolicy(
                snapKey,
                LeagueOpponentSpec.bot(Math.max(1, LEAGUE_BASELINE_BOT_SKILL)),
                null,
                LEAGUE_BASELINE_DECKLIST_FILE,
                Math.max(1, LEAGUE_BASELINE_GAMES_PER_MATCHUP),
                "baseline_ep=" + episodeNum
        );
        logger.info(String.format("League tick baseline: ep=%d key=%s vs CP7Skill%d wr=%.3f",
                episodeNum, snapKey, Math.max(1, LEAGUE_BASELINE_BOT_SKILL), baselineWr));
        appendLeagueEvent(String.format("baseline_eval ep=%d key=%s opp=CP7Skill%d wr=%.6f",
                episodeNum, snapKey, Math.max(1, LEAGUE_BASELINE_BOT_SKILL), baselineWr));

        synchronized (leagueLock()) {
            st.lastTickEpisode = episodeNum;
            st.baselineWr.put(snapKey, baselineWr);
        }

        // 3) Promotion gate (bootstrap -> league)
        boolean promotedNow = false;
        synchronized (leagueLock()) {
            if (!st.promoted && baselineWr >= LEAGUE_PROMOTE_WR) {
                st.promoted = true;
                promotedNow = true;
            }
        }
        if (promotedNow) {
            logger.info(String.format("League tick PROMOTED: ep=%d key=%s baselineWR=%.3f >= %.3f",
                    episodeNum, snapKey, baselineWr, LEAGUE_PROMOTE_WR));
            appendLeagueEvent(String.format("promoted ep=%d key=%s wr=%.6f threshold=%.6f",
                    episodeNum, snapKey, baselineWr, LEAGUE_PROMOTE_WR));
        }

        // 4) Pool admission + recency tracking
        boolean admitted = false;
        synchronized (leagueLock()) {
            if (baselineWr >= LEAGUE_POOL_FLOOR_WR) {
                addToPool(st, snapKey);
                addRecent(st, snapKey);
                admitted = true;
            }
        }
        logger.info(String.format("League tick pool_admit: ep=%d key=%s admitted=%s baselineWR=%.3f floor=%.3f",
                episodeNum, snapKey, admitted, baselineWr, LEAGUE_POOL_FLOOR_WR));
        appendLeagueEvent(String.format("pool_admit ep=%d key=%s admitted=%s wr=%.6f floor=%.6f",
                episodeNum, snapKey, admitted, baselineWr, LEAGUE_POOL_FLOOR_WR));

        // 5) Champion update (S_t vs S_best) once promoted & we have a champion
        boolean championUpdated = false;
        Double champMatchWr = null;
        String championBefore;
        synchronized (leagueLock()) {
            championBefore = st.championPolicyKey;
            if (st.championPolicyKey == null && st.promoted && admitted) {
                st.championPolicyKey = snapKey;
                championUpdated = true;
            }
        }
        if (championUpdated) {
            logger.info("League tick champion init: ep=" + episodeNum + " champion=" + snapKey);
            appendLeagueEvent("champion_init ep=" + episodeNum + " champion=" + snapKey);
        }
        String championAfter;
        synchronized (leagueLock()) {
            championAfter = st.championPolicyKey;
        }
        if (!championUpdated && st.promoted && championAfter != null && !championAfter.equals(snapKey)) {
            logger.info("League tick champion match start: ep=" + episodeNum + " challenger=" + snapKey + " vs champion=" + championAfter);
            appendLeagueEvent("champion_match_start ep=" + episodeNum + " challenger=" + snapKey + " champion=" + championAfter);
            champMatchWr = runLeagueBenchmarkPolicy(
                    snapKey,
                    LeagueOpponentSpec.snapshot(championAfter),
                    null,
                    LEAGUE_BASELINE_DECKLIST_FILE,
                    Math.max(1, LEAGUE_BASELINE_GAMES_PER_MATCHUP),
                    "champion_match_ep=" + episodeNum
            );
            logger.info(String.format("League tick champion match done: ep=%d challenger=%s vs champion=%s wr=%.3f threshold=%.3f",
                    episodeNum, snapKey, championAfter, champMatchWr, LEAGUE_CHAMPION_PROMOTE_WR));
            appendLeagueEvent(String.format("champion_match_done ep=%d challenger=%s champion=%s wr=%.6f threshold=%.6f",
                    episodeNum, snapKey, championAfter, champMatchWr, LEAGUE_CHAMPION_PROMOTE_WR));
            synchronized (leagueLock()) {
                if (champMatchWr != null && champMatchWr >= LEAGUE_CHAMPION_PROMOTE_WR) {
                    st.championPolicyKey = snapKey;
                    championUpdated = true;
                }
            }
        }
        if (championUpdated && championBefore != null && !championBefore.equals(st.championPolicyKey)) {
            logger.info("League tick champion UPDATED: ep=" + episodeNum + " old=" + championBefore + " new=" + st.championPolicyKey);
            appendLeagueEvent("champion_updated ep=" + episodeNum + " old=" + championBefore + " new=" + st.championPolicyKey);
        }

        // 6) Prune pool
        synchronized (leagueLock()) {
            prunePool(st, rand);
            saveLeagueState();
            writeLeagueStatus(st);
        }

        long tickMs = Math.max(1, System.currentTimeMillis() - tickStartMs);
        // One-line summary for easy grepping.
        logger.info(String.format(
                "League tick ep=%d snap=%s baselineWR=%.3f promoted=%s admitted=%s championUpdated=%s champMatchWR=%s pool=%d time=%.1fs",
                episodeNum,
                snapKey,
                baselineWr,
                st.promoted,
                admitted,
                championUpdated,
                champMatchWr == null ? "n/a" : String.format("%.3f", champMatchWr),
                st.pool.size(),
                tickMs / 1000.0
        ));
        appendLeagueEvent(String.format("tick_done ep=%d key=%s baselineWR=%.6f promoted=%s admitted=%s champion=%s pool=%d time_s=%.3f",
                episodeNum, snapKey, baselineWr, st.promoted, admitted,
                st.championPolicyKey == null ? "" : st.championPolicyKey,
                st.pool.size(), tickMs / 1000.0));
        if (GAME_STATS_WRITER) writeAgentStatus(episodeNum);
    }

    private void maybeRunLadderTick(int episodeNum) {
        if (LADDER_TICK_EPISODES <= 0) {
            return;
        }
        if (episodeNum <= 0 || (episodeNum % LADDER_TICK_EPISODES) != 0) {
            return;
        }
        // Only in ladder mode
        String mode = OPPONENT_SAMPLER == null ? "league" : OPPONENT_SAMPLER.trim().toLowerCase();
        if (!"ladder".equals(mode)) {
            return;
        }
        // Single-thread the tick across runners
        int prev = ladderLastTickEp().get();
        if (episodeNum <= prev) {
            return;
        }
        if (!ladderLastTickEp().compareAndSet(prev, episodeNum)) {
            return;
        }

        final long tickStartMs = System.currentTimeMillis();
        LadderState ls = getLadderState();
        int[] tiers = parseSkillList(LADDER_SKILLS, new int[]{0, 1, 2, 3});
        
        if (tiers.length == 0) {
            logger.warn("Ladder tick: no tiers defined in LADDER_SKILLS, skipping");
            return;
        }

        int currentTier = Math.min(ls.currentTier, tiers.length - 1);
        int currentSkill = tiers[currentTier];

        logger.info("Ladder tick start: ep=" + episodeNum + " tickEvery=" + LADDER_TICK_EPISODES
                + " currentTier=" + currentTier + " currentSkill=" + currentSkill
                + " gpm=" + LADDER_GAMES_PER_MATCHUP
                + " promoteWR=" + String.format("%.3f", LADDER_PROMOTE_WR));
        appendLeagueEvent("ladder_tick_start ep=" + episodeNum 
                + " tier=" + currentTier + " skill=" + currentSkill
                + " gpm=" + LADDER_GAMES_PER_MATCHUP);

        // Benchmark current policy vs current tier bot.
        // Agent uses its own deck list (RL_AGENT_DECK_LIST if set, else full pool).
        String ladderAgentDeckList = (RL_AGENT_DECK_LIST_FILE != null && !RL_AGENT_DECK_LIST_FILE.trim().isEmpty())
                ? RL_AGENT_DECK_LIST_FILE : null;
        double benchmarkWr = runLeagueBenchmarkPolicy(
                "train",
                LeagueOpponentSpec.bot(currentSkill),
                ladderAgentDeckList,
                DECK_LIST_FILE,
                Math.max(1, LADDER_GAMES_PER_MATCHUP),
                "ladder_ep=" + episodeNum + "_tier=" + currentTier
        );
        
        logger.info(String.format("Ladder tick benchmark: ep=%d tier=%d skill=%d wr=%.3f",
                episodeNum, currentTier, currentSkill, benchmarkWr));
        appendLeagueEvent(String.format("ladder_benchmark ep=%d tier=%d skill=%d wr=%.6f",
                episodeNum, currentTier, currentSkill, benchmarkWr));

        synchronized (ladderLock()) {
            ls.lastTickEpisode = episodeNum;
            ls.tierWinrates.put(currentTier, benchmarkWr);
            ls.evalHistory.add(String.format("ep=%d,tier=%d,skill=%d,wr=%.6f",
                    episodeNum, currentTier, currentSkill, benchmarkWr));
        }

        // Promotion check
        boolean promotedNow = false;
        synchronized (ladderLock()) {
            if (benchmarkWr >= LADDER_PROMOTE_WR && currentTier < tiers.length - 1) {
                ls.currentTier++;
                promotedNow = true;
            }
        }

        if (promotedNow) {
            int newTier = ls.currentTier;
            int newSkill = tiers[newTier];
            logger.info(String.format("Ladder tick PROMOTED: ep=%d tier %d->%d skill %d->%d benchmarkWR=%.3f >= %.3f",
                    episodeNum, currentTier, newTier, currentSkill, newSkill, benchmarkWr, LADDER_PROMOTE_WR));
            appendLeagueEvent(String.format("ladder_promoted ep=%d oldTier=%d newTier=%d oldSkill=%d newSkill=%d wr=%.6f threshold=%.6f",
                    episodeNum, currentTier, newTier, currentSkill, newSkill, benchmarkWr, LADDER_PROMOTE_WR));
        }

        synchronized (ladderLock()) {
            saveLadderState();
            writeLadderStatus(ls, tiers);
        }

        long tickMs = Math.max(1, System.currentTimeMillis() - tickStartMs);
        logger.info(String.format(
                "Ladder tick ep=%d tier=%d skill=%d wr=%.3f promoted=%s time=%.1fs",
                episodeNum,
                ls.currentTier,
                ls.currentTier < tiers.length ? tiers[ls.currentTier] : -1,
                benchmarkWr,
                promotedNow,
                tickMs / 1000.0
        ));
        appendLeagueEvent(String.format("ladder_tick_done ep=%d tier=%d skill=%d wr=%.6f promoted=%s time_s=%.3f",
                episodeNum, ls.currentTier, 
                ls.currentTier < tiers.length ? tiers[ls.currentTier] : -1,
                benchmarkWr, promotedNow, tickMs / 1000.0));
    }

    private static Level parseLog4jLevel(String s, Level fallback) {
        if (s == null) {
            return fallback;
        }
        String v = s.trim().toUpperCase();
        if (v.isEmpty()) {
            return fallback;
        }
        switch (v) {
            case "OFF":
                return Level.OFF;
            case "FATAL":
                return Level.FATAL;
            case "ERROR":
                return Level.ERROR;
            case "WARN":
            case "WARNING":
                return Level.WARN;
            case "INFO":
                return Level.INFO;
            case "DEBUG":
                return Level.DEBUG;
            case "TRACE":
                return Level.TRACE;
            default:
                return fallback;
        }
    }

    // ThreadLocal logger
    public static final ThreadLocal<Logger> threadLocalLogger = ThreadLocal.withInitial(() -> {
        Logger threadLogger = Logger.getLogger("Thread-" + Thread.currentThread().getId());
        // Per-decision logging is extremely verbose; keep it off by default.
        // - Enable with RL_VERBOSE_DECISIONS=1 (forces INFO)
        // - Or set RL_THREAD_LOG_LEVEL=OFF|ERROR|WARN|INFO|DEBUG|TRACE
        // - Or hard-silence via RL_SILENCE_THREAD_LOGGER=1 (forces OFF)
        boolean silence = "1".equals(System.getenv().getOrDefault("RL_SILENCE_THREAD_LOGGER", "0"))
                || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_SILENCE_THREAD_LOGGER", "0"));
        boolean verboseDecisions = "1".equals(System.getenv().getOrDefault("RL_VERBOSE_DECISIONS", "0"))
                || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_VERBOSE_DECISIONS", "0"));
        if (silence) {
            threadLogger.setLevel(Level.OFF);
        } else if (verboseDecisions) {
            threadLogger.setLevel(Level.INFO);
        } else {
            threadLogger.setLevel(parseLog4jLevel(System.getenv("RL_THREAD_LOG_LEVEL"), Level.WARN));
        }
        return threadLogger;
    });

    // ThreadLocal game logger for detailed analysis of eval games
    // Enable with GAME_LOGGING=1
    public static final ThreadLocal<GameLogger> threadLocalGameLogger = ThreadLocal.withInitial(
            () -> GameLogger.create(false) // Default: disabled
    );

    private static final class ActorLearnerDispatcher {

        private enum TaskType {
            TRAINING,
            GAME_RESULT
        }

        private static final class Task {
            final TaskType type;
            final ProfileContext context;
            final List<StateSequenceBuilder.TrainingData> trainingData;
            final List<Double> rewards;
            final float lastValue;
            final boolean won;
            final String source;

            Task(ProfileContext context,
                 List<StateSequenceBuilder.TrainingData> trainingData,
                 List<Double> rewards,
                 String source) {
                this.type = TaskType.TRAINING;
                this.context = context;
                this.trainingData = new ArrayList<>(trainingData);
                this.rewards = new ArrayList<>(rewards);
                this.lastValue = 0.0f;
                this.won = false;
                this.source = source;
            }

            Task(ProfileContext context, float lastValue, boolean won, String source) {
                this.type = TaskType.GAME_RESULT;
                this.context = context;
                this.trainingData = Collections.emptyList();
                this.rewards = Collections.emptyList();
                this.lastValue = lastValue;
                this.won = won;
                this.source = source;
            }
        }

        private final boolean async;
        private final int capacity;
        private final int workerCount;
        private final String backpressureMode;
        private final int offerTimeoutMs;
        private final BlockingQueue<Task> queue;
        private final List<Thread> workers = new ArrayList<>();
        private final AtomicBoolean started = new AtomicBoolean(false);
        private final AtomicBoolean shutdown = new AtomicBoolean(false);

        ActorLearnerDispatcher(boolean async,
                               int capacity,
                               int workerCount,
                               String backpressureMode,
                               int offerTimeoutMs) {
            this.async = async;
            this.capacity = capacity;
            this.workerCount = workerCount;
            this.backpressureMode = normalizeBackpressureMode(backpressureMode);
            this.offerTimeoutMs = offerTimeoutMs;
            this.queue = async ? new ArrayBlockingQueue<>(capacity) : null;
            metrics.setActorLearnerQueueDepth(0, async ? capacity : 0);
        }

        void enqueueTraining(ProfileContext context,
                             List<StateSequenceBuilder.TrainingData> trainingData,
                             List<Double> rewards,
                             String source) {
            if (trainingData == null || trainingData.isEmpty()) {
                return;
            }
            enqueue(new Task(context, trainingData, rewards, source));
        }

        void enqueueGameResult(ProfileContext context, float lastValue, boolean won, String source) {
            enqueue(new Task(context, lastValue, won, source));
        }

        private void enqueue(Task task) {
            if (!async) {
                metrics.recordActorLearnerEnqueued(0, 0);
                runTask(task);
                return;
            }
            startWorkers();
            if (shutdown.get()) {
                metrics.recordActorLearnerDropped(queue.size(), capacity);
                return;
            }
            if (queue.offer(task)) {
                metrics.recordActorLearnerEnqueued(queue.size(), capacity);
                return;
            }
            if ("drop_oldest".equals(backpressureMode)) {
                queue.poll();
                metrics.recordActorLearnerDropped(queue.size(), capacity);
                if (queue.offer(task)) {
                    metrics.recordActorLearnerEnqueued(queue.size(), capacity);
                } else {
                    metrics.recordActorLearnerDropped(queue.size(), capacity);
                }
                return;
            }
            if ("drop_newest".equals(backpressureMode)) {
                metrics.recordActorLearnerDropped(queue.size(), capacity);
                return;
            }
            enqueueWithBackpressure(task);
        }

        private void enqueueWithBackpressure(Task task) {
            long waitStartNanos = System.nanoTime();
            while (!shutdown.get()) {
                try {
                    if (queue.offer(task, offerTimeoutMs, TimeUnit.MILLISECONDS)) {
                        long waitMs = (System.nanoTime() - waitStartNanos) / 1_000_000L;
                        metrics.recordActorLearnerBackpressureWait(waitMs, queue.size(), capacity);
                        metrics.recordActorLearnerEnqueued(queue.size(), capacity);
                        return;
                    }
                    long waitMs = (System.nanoTime() - waitStartNanos) / 1_000_000L;
                    logger.warn("Actor learner queue has been full for " + waitMs
                            + "ms; backpressuring game runners instead of dropping training data");
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    metrics.recordActorLearnerDropped(queue.size(), capacity);
                    return;
                }
            }
            metrics.recordActorLearnerDropped(queue.size(), capacity);
        }

        private void startWorkers() {
            if (!started.compareAndSet(false, true)) {
                return;
            }
            for (int i = 0; i < workerCount; i++) {
                Thread worker = new Thread(this::workerLoop, "ACTOR-LEARNER-" + i);
                worker.setDaemon(true);
                worker.setPriority(Thread.NORM_PRIORITY);
                workers.add(worker);
                worker.start();
            }
        }

        private void workerLoop() {
            while (!shutdown.get() || !queue.isEmpty()) {
                try {
                    Task task = queue.poll(250, TimeUnit.MILLISECONDS);
                    if (task != null) {
                        runTask(task);
                    } else {
                        metrics.setActorLearnerQueueDepth(queue.size(), capacity);
                    }
                } catch (InterruptedException e) {
                    if (shutdown.get()) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
            metrics.setActorLearnerQueueDepth(queue.size(), capacity);
        }

        private void runTask(Task task) {
            ProfileContext previous = ProfileContext.current();
            try {
                ProfileContext.setCurrent(task.context);
                if (task.type == TaskType.TRAINING) {
                    sharedModel.enqueueTraining(task.trainingData, task.rewards);
                } else {
                    sharedModel.recordGameResult(task.lastValue, task.won);
                }
                metrics.recordActorLearnerSent(queueDepth(), queueCapacity());
            } catch (Throwable t) {
                metrics.recordActorLearnerFailed(queueDepth(), queueCapacity());
                logger.warn("Actor learner task failed"
                        + (task.source == null || task.source.isEmpty() ? "" : " source=" + task.source)
                        + ": " + t.getMessage(), t);
            } finally {
                ProfileContext.setCurrent(previous);
            }
        }

        private int queueDepth() {
            return async ? queue.size() : 0;
        }

        private int queueCapacity() {
            return async ? capacity : 0;
        }

        private static String normalizeBackpressureMode(String value) {
            String mode = value == null ? "" : value.trim().toLowerCase();
            if ("drop".equals(mode)) {
                return "drop_oldest";
            }
            if ("drop_oldest".equals(mode) || "drop_newest".equals(mode) || "block".equals(mode)) {
                return mode;
            }
            return "block";
        }

        void shutdownAndDrain(long timeoutMs) {
            if (!async) {
                return;
            }
            shutdown.set(true);
            long deadline = System.currentTimeMillis() + Math.max(0L, timeoutMs);
            for (Thread worker : workers) {
                long remaining = Math.max(0L, deadline - System.currentTimeMillis());
                if (remaining <= 0L && worker.isAlive()) {
                    break;
                }
                try {
                    worker.join(remaining);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            if (!queue.isEmpty()) {
                int remaining = queue.size();
                queue.clear();
                for (int i = 0; i < remaining; i++) {
                    metrics.recordActorLearnerDropped(0, capacity);
                }
                logger.warn("Actor learner shutdown dropped " + remaining + " queued tasks after drain timeout");
            }
            metrics.setActorLearnerQueueDepth(queue.size(), capacity);
        }
    }

    // Multi-profile support: when set, per-profile state comes from this context
    // instead of the static fields.  Null means legacy single-profile mode.
    private final ProfileContext profileContext;

    public RLTrainer() {
        this.profileContext = null;
    }

    RLTrainer(ProfileContext ctx) {
        this.profileContext = ctx;
    }

    // ---- Per-profile state accessors (context-first, fallback to static) ----

    private java.util.concurrent.atomic.AtomicInteger episodeCounter() {
        return profileContext != null ? profileContext.episodeCounter : EPISODE_COUNTER;
    }

    private java.util.concurrent.atomic.AtomicInteger activeEps() {
        return profileContext != null ? profileContext.activeEpisodes : ACTIVE_EPISODES;
    }

    private ConcurrentLinkedQueue<Boolean> recentWinsQueue() {
        return profileContext != null ? profileContext.recentWins : recentWins;
    }

    private java.util.concurrent.atomic.AtomicInteger winCounter() {
        return profileContext != null ? profileContext.winCount : winCount;
    }

    private java.util.concurrent.atomic.AtomicInteger gamesAtLevel() {
        return profileContext != null ? profileContext.gamesAtCurrentLevel : gamesAtCurrentLevel;
    }

    private String profileName() {
        return profileContext != null ? profileContext.profileName : MODEL_PROFILE_NAME;
    }

    private String statsFilePath() {
        return profileContext != null ? profileContext.paths.trainingStatsPath : STATS_FILE_PATH;
    }

    private String modelFilePath() {
        return profileContext != null ? profileContext.paths.modelFilePath : MODEL_FILE_PATH;
    }

    private String modelsBaseDir() {
        return profileContext != null ? profileContext.paths.modelsBaseDir : RLLogPaths.MODELS_BASE_DIR;
    }

    private String logsBaseDir() {
        return profileContext != null ? profileContext.paths.logsBaseDir : RLLogPaths.LOGS_BASE_DIR;
    }

    private String snapshotDir() {
        return profileContext != null ? profileContext.paths.snapshotDir : SNAPSHOT_DIR;
    }

    private OpponentLevel currentOpponentLevel() {
        return profileContext != null ? profileContext.currentOpponentLevel : currentOpponentLevel;
    }

    private void setCurrentOpponentLevel(OpponentLevel level) {
        if (profileContext != null) {
            profileContext.currentOpponentLevel = level;
        } else {
            currentOpponentLevel = level;
        }
    }

    private String lastOpponentType() {
        return profileContext != null ? profileContext.lastOpponentType : lastOpponentType;
    }

    private void setLastOpponentType(String type) {
        if (profileContext != null) {
            profileContext.lastOpponentType = type;
        } else {
            lastOpponentType = type;
        }
    }

    private Object leagueLock() {
        return profileContext != null ? profileContext.leagueLock : LEAGUE_LOCK;
    }

    private java.util.concurrent.atomic.AtomicInteger leagueLastTickEp() {
        return profileContext != null ? profileContext.leagueLastTickEp : LEAGUE_LAST_TICK_EP;
    }

    private Object ladderLock() {
        return profileContext != null ? profileContext.ladderLock : LADDER_LOCK;
    }

    private java.util.concurrent.atomic.AtomicInteger ladderLastTickEp() {
        return profileContext != null ? profileContext.ladderLastTickEp : LADDER_LAST_TICK_EP;
    }

    /* ================================================================
     *  Simple CLI entry point: java RLTrainer train|eval
     * ============================================================ */
    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : EnvConfig.str("MODE", "train");
        int exitCode = 0;

        int metricsPort = EnvConfig.i32("METRICS_PORT", 9090);
        try {
            if (!"trainAll".equalsIgnoreCase(mode)) {
                logger.info("Python device info: " + sharedModel.getDeviceInfo());
            }
            metrics.startMetricsServer(metricsPort);
            logger.info("Metrics server started on port " + metricsPort);
            if ("eval".equalsIgnoreCase(mode)) {
                runEvaluation(NUM_EVAL_EPISODES);
            } else if ("benchmark".equalsIgnoreCase(mode)) {
                int gamesPerMatchup = EnvConfig.i32("GAMES_PER_MATCHUP", 20);
                new RLTrainer().runBenchmark(gamesPerMatchup);
            } else if ("league_eval".equalsIgnoreCase(mode)) {
                new RLTrainer().runLeagueEvaluation();
            } else if ("league_bench".equalsIgnoreCase(mode)) {
                new RLTrainer().runLeagueBench();
            } else if ("trainAll".equalsIgnoreCase(mode)) {
                // Multi-profile: single JVM, shared card DB
                List<String> profiles = new ArrayList<>();
                for (int i = 1; i < args.length; i++) {
                    profiles.add(args[i]);
                }
                if (profiles.isEmpty()) {
                    String list = EnvConfig.str("TRAIN_PROFILES_LIST", "");
                    for (String p : list.split(",")) {
                        if (!p.trim().isEmpty()) profiles.add(p.trim());
                    }
                }
                if (profiles.isEmpty()) {
                    logger.error("trainAll mode requires profile names as args or TRAIN_PROFILES_LIST env var");
                    System.exit(1);
                }
                trainMultiProfile(profiles);
            } else {
                new RLTrainer().train();
            }
        } catch (Throwable e) {
            exitCode = 1;
            logger.error("RLTrainer main failed", e);
        } finally {
            // Ensure CLI invocations exit cleanly (no lingering metrics scheduler / py4j ports).
            try {
                ACTOR_LEARNER.shutdownAndDrain(ACTOR_LEARNER_DRAIN_TIMEOUT_MS);
            } catch (Exception ignored) {
            }
            try {
                metrics.stop();
            } catch (Exception ignored) {
            }
            try {
                sharedModel.shutdown();
            } catch (Exception ignored) {
            }
        }
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    public void train() {
        System.out.println("DEBUG: Starting train() method");
        System.out.println("DEBUG: DECKS_DIRECTORY = " + DECKS_DIRECTORY);
        System.out.println("DEBUG: Working directory = " + System.getProperty("user.dir"));

        // In multi-profile mode, card DB and deck pools are pre-loaded by trainMultiProfile()
        final boolean multiProfile = profileContext != null;
        if (!multiProfile) {
            System.out.println("DEBUG: Initializing card database...");
            mage.cards.repository.CardScanner.scan();
            System.out.println("DEBUG: Card database initialized");
        } else {
            System.out.println("DEBUG: Multi-profile mode, card DB already loaded. Profile: " + profileContext.profileName);
        }

        try {
            List<Path> deckFiles;
            List<Path> agentDeckFiles;
            if (multiProfile) {
                deckFiles = profileContext.deckFiles;
                agentDeckFiles = profileContext.agentDeckFiles;
            } else {
                deckFiles = loadDeckPool();
                if (RL_AGENT_DECK_LIST_FILE != null && !RL_AGENT_DECK_LIST_FILE.trim().isEmpty()) {
                    agentDeckFiles = loadDeckPoolWithOverride(RL_AGENT_DECK_LIST_FILE);
                    logger.info("Agent deck pool: " + agentDeckFiles.size() + " decks (from RL_AGENT_DECK_LIST)");
                } else {
                    agentDeckFiles = deckFiles;
                    logger.info("Agent deck pool: same as opponent pool (" + deckFiles.size() + " decks)");
                }
            }
            System.out.println("DEBUG: Found " + deckFiles.size() + " deck files in deck pool");

            // Reset health stats and failure counters at start of training
            TrainingHealthStats healthStats = TrainingHealthStats.getInstance();
            healthStats.reset();
            String pName = profileName();
            if (!pName.isEmpty()) {
                logger.info("Model profile: " + pName
                        + "  models=" + modelsBaseDir()
                        + "  logs=" + logsBaseDir());
            }
            logger.info("Starting training - health stats reset (games_killed=0, rl_failures=0)");

            // Start health stats logging
            healthStats.start();

            // Episode counters start from 0 each JVM launch; Python owns persistence.

            // Log system resource utilization
            int cpuCores = Runtime.getRuntime().availableProcessors();
            long maxMemory = Runtime.getRuntime().maxMemory() / 1024 / 1024; // MB
            RLTrainer.threadLocalLogger.get().info("=== SYSTEM RESOURCES ===");
            RLTrainer.threadLocalLogger.get().info("CPU Cores Available: " + cpuCores);
            RLTrainer.threadLocalLogger.get().info("Max JVM Memory: " + maxMemory + " MB");
            int loggedRunners = multiProfile ? profileContext.numGameRunners : NUM_GAME_RUNNERS;
            RLTrainer.threadLocalLogger.get().info("Game Runners: " + loggedRunners + " (using " + (loggedRunners * 100.0 / cpuCores) + "% of CPU cores)");
            RLTrainer.threadLocalLogger.get().info("Episodes per runner: " + NUM_EPISODES_PER_GAME_RUNNER);
            RLTrainer.threadLocalLogger.get().info("Total episodes target: " + NUM_EPISODES);
            RLTrainer.threadLocalLogger.get().info("========================");

            // Record start time
            long startTime = System.nanoTime();
            final long startMs = System.currentTimeMillis();
            final int startEpisodeCountSnapshot = episodeCounter().get();
            final int targetEpisodeCount = NUM_EPISODES;
            final int trainLogEvery = EnvConfig.i32("TRAIN_LOG_EVERY", 10);
            final int trainHeartbeatSec = EnvConfig.i32("TRAIN_HEARTBEAT_SEC", 30);
            final java.util.concurrent.atomic.AtomicInteger lastLoggedEpisode = new java.util.concurrent.atomic.AtomicInteger(startEpisodeCountSnapshot);

            // Heartbeat so the console doesn't look stuck during long games.
            java.util.concurrent.ScheduledExecutorService heartbeat = null;
            if (trainHeartbeatSec > 0) {
                heartbeat = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
                    Thread t = new Thread(r, "TRAIN-HEARTBEAT");
                    t.setDaemon(true);
                    return t;
                });
                final java.util.concurrent.ScheduledExecutorService hbRef = heartbeat;
                heartbeat.scheduleAtFixedRate(() -> {
                    int epNow = episodeCounter().get();
                    int done = Math.max(0, epNow - startEpisodeCountSnapshot);
                    int remaining = Math.max(0, targetEpisodeCount - epNow);
                    long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                    double epsPerSec = done / (elapsedMs / 1000.0);
                    long etaSec = epsPerSec > 0 ? (long) (remaining / epsPerSec) : -1;
                    logger.info(String.format(
                            "Training heartbeat: episode=%d/%d (run=%d, %.3f eps/s), ETA %ds",
                            epNow, targetEpisodeCount, done, epsPerSec, etaSec
                    ));
                    if (epNow >= targetEpisodeCount) {
                        hbRef.shutdown();
                    }
                }, trainHeartbeatSec, trainHeartbeatSec, java.util.concurrent.TimeUnit.SECONDS);
            }

            final int numRunners = multiProfile ? profileContext.numGameRunners : NUM_GAME_RUNNERS;
            ExecutorService executor = Executors.newFixedThreadPool(numRunners, runnable -> {
                Thread thread = new Thread(runnable);
                thread.setPriority(Thread.MIN_PRIORITY);
                return thread;
            });
            logger.info("Using platform threads for " + numRunners + " game runners");

            List<Future<Void>> futures = new ArrayList<>();
            final Object lock = new Object(); // Lock object for synchronization
            final boolean[] isFirstThread = {true}; // Flag to track the first thread

            final ProfileContext parentCtx = profileContext;
            for (int i = 0; i < numRunners; i++) {
                final int runnerIndex = i;
                Future<Void> future = executor.submit(() -> {
                    if (parentCtx != null) ProfileContext.setCurrent(parentCtx);
                    boolean isFirst;
                    synchronized (lock) {
                        isFirst = isFirstThread[0];
                        isFirstThread[0] = false;
                    }

                    Logger currentLogger = threadLocalLogger.get();
                    // All threads will now log at INFO level by default
                    currentLogger.info("Starting Game Runner");

                    Thread.currentThread().setName("GAME");
                    Random threadRand = newSeededRandom(1_000L + runnerIndex);

                    while (episodeCounter().get() < NUM_EPISODES) {
                      try {
                        StateSequenceBuilder.clearThreadLocalKnownArchetypeLabels();
                        int epNumber = episodeCounter().incrementAndGet();
                        if (epNumber > NUM_EPISODES) {
                            break; // Another thread reached the target
                        }
                        metrics.recordEpisodeStarted();
                        long episodeStartNanos = System.nanoTime();
                        activeEps().incrementAndGet();
                        metrics.setActiveEpisodes(activeEps().get());
                        Path rlPlayerDeckPath = pickAgentDeck(agentDeckFiles, threadRand);
                        Deck rlPlayerDeckThread = loadDeck(rlPlayerDeckPath.toString());
                        if (rlPlayerDeckThread == null) {
                            logger.warn("Train: failed to load deck(s) for game, skipping.");
                            activeEps().decrementAndGet();
                            metrics.setActiveEpisodes(activeEps().get());
                            continue;
                        }

                        // Periodic eval checkpoint: play batch of games vs CP7 across all decks (background)
                        boolean evalDue = EVAL_EVERY > 0 && (
                                epNumber % EVAL_EVERY == 0
                                        || (EVAL_AT_START && firstEvalDone.compareAndSet(false, true)));
                        if (evalDue) {
                            logger.info("EVAL TRIGGER: ep=" + epNumber + " EVAL_EVERY=" + EVAL_EVERY
                                    + " EVAL_AT_START=" + EVAL_AT_START
                                    + " deckFiles=" + deckFiles.size() + " agentDecks=" + agentDeckFiles.size());
                            submitEvalCheckpoint(epNumber, deckFiles, agentDeckFiles);
                        }

                        THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.remove();
                        Player opponentPlayer = createTrainingOpponent(epNumber, threadRand);
                        Path opponentDeckPath = THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.get();
                        if (opponentDeckPath == null) {
                            opponentDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
                        }
                        // Load opponent deck independently (not from cache) to avoid shared card references
                        // when both players use the same deck file in self-play
                        Deck opponentDeckThread = loadDeckFresh(opponentDeckPath.toString());
                        if (opponentDeckThread == null && THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.get() != null) {
                            logger.warn("Train: failed to load override opponent deck " + opponentDeckPath + ", falling back to pool random deck.");
                            opponentDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
                            opponentDeckThread = loadDeck(opponentDeckPath.toString());
                        }
                        THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.remove();
                        if (opponentDeckThread == null) {
                            logger.warn("Train: failed to load deck(s) for game, skipping.");
                            activeEps().decrementAndGet();
                            metrics.setActiveEpisodes(activeEps().get());
                            continue;
                        }

                        // ------------------ Build a full Match so players have MatchPlayer objects ------------------
                        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));

                        // Start an empty game so we can attach players
                        match.startGame();
                        Game game = match.getGames().get(0);

                        // Log first 20 games immediately for sanity checking, then every GAME_LOG_FREQUENCY
                        boolean enableGameLogging = (epNumber <= 20)
                                || (GAME_LOG_FREQUENCY > 0 && epNumber % GAME_LOG_FREQUENCY == 0);
                        // In multi-profile mode, direct game logs to the profile's log directory
                        if (multiProfile && enableGameLogging && profileContext != null) {
                            System.setProperty("GAME_LOG_DIR_OVERRIDE", profileContext.paths.trainingGameLogsDir);
                        }
                        GameLogger gameLogger = enableGameLogging && multiProfile && profileContext != null
                                ? GameLogger.createInProfileDir(profileContext.paths.trainingGameLogsDir, 20)
                                : GameLogger.create(enableGameLogging);
                        threadLocalGameLogger.set(gameLogger);

                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel);
                        rlPlayer.setCurrentEpisode(epNumber); // Set main model episode for logging
                        rlPlayer.setAttachedGameLogger(gameLogger);
                        if (opponentPlayer instanceof ComputerPlayerRL) {
                            ((ComputerPlayerRL) opponentPlayer).setCurrentEpisode(epNumber);
                        }
                        game.addPlayer(rlPlayer, rlPlayerDeckThread);
                        match.addPlayer(rlPlayer, rlPlayerDeckThread);

                        // Opponent already selected before deck binding.
                        game.addPlayer(opponentPlayer, opponentDeckThread);
                        match.addPlayer(opponentPlayer, opponentDeckThread);

                        java.util.Map<UUID, Integer> knownArchetypeLabels = new java.util.HashMap<>();
                        knownArchetypeLabels.put(rlPlayer.getId(),
                                StateSequenceBuilder.computeArchetypeLabelFromDeckName(rlPlayerDeckPath == null ? "" : rlPlayerDeckPath.toString()));
                        knownArchetypeLabels.put(opponentPlayer.getId(),
                                StateSequenceBuilder.computeArchetypeLabelFromDeckName(opponentDeckPath == null ? "" : opponentDeckPath.toString()));
                        StateSequenceBuilder.setThreadLocalKnownArchetypeLabels(knownArchetypeLabels);

                        String opponentTag = formatOpponentTag(opponentPlayer);
                        if (gameLogger.isEnabled()) {
                            gameLogger.log("OPPONENT: " + opponentTag + " (name=" + opponentPlayer.getName() + ")");
                        }

                        logger.info("Players added to game. RL player library size: " + rlPlayer.getLibrary().size() + ", Opponent library size: " + opponentPlayer.getLibrary().size());

                        game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeckThread.getCards(), opponentPlayer.getId());

                        GameOptions options = new GameOptions();
                        options.rollbackTurnsAllowed = false;
                        game.setGameOptions(options);

                        // Start health monitor to detect stuck/infinite-loop games
                        GameHealthMonitor healthMonitor = GameHealthMonitor.createAndStart(game);

                        long startGameNanos = System.nanoTime();
                        metrics.recordEpisodeSetupMs((startGameNanos - episodeStartNanos) / 1_000_000L);
                        // Restart game now that players are added
                        game.start(rlPlayer.getId());
                        long endGameNanos = System.nanoTime();
                        metrics.recordEpisodeGameMs((endGameNanos - startGameNanos) / 1_000_000L);

                        // Stop health monitor
                        healthMonitor.stop();

                        // Log game duration
                        long gameDurationSec = (endGameNanos - startGameNanos) / 1_000_000_000L;
                        int turns = game.getTurnNum();

                        // Check if game was killed by health monitor
                        boolean rlPlayerWon;
                        if (healthMonitor.wasKilled()) {
                            logger.warn(String.format("Episode %d KILLED after %ds (%d turns): %s", 
                                epNumber, gameDurationSec, turns, healthMonitor.getKillReason()));
                            System.err.println(String.format("!!! GAME KILLED: Episode %d after %ds (%d turns): %s", 
                                epNumber, gameDurationSec, turns, healthMonitor.getKillReason()));
                            // Treat killed games as losses to discourage problematic states
                            rlPlayerWon = false;
                        } else {
                            // Record outcome for adaptive curriculum - returns snapshot to avoid race condition
                            rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
                            // Log successful game duration occasionally for comparison
                            if (epNumber % 10 == 0) {
                                logger.info(String.format("Episode %d completed in %ds (%d turns)", 
                                    epNumber, gameDurationSec, turns));
                            }
                        }
                        double[] outcomeSnapshot = recordGameOutcome(rlPlayerWon);
                        double snapshotWinrate = outcomeSnapshot[0];
                        int snapshotSampleSize = (int) outcomeSnapshot[1];

                        // Log game outcome to detailed game log if enabled
                        if (gameLogger.isEnabled()) {
                            String winner = rlPlayerWon ? rlPlayer.getName() : opponentPlayer.getName();
                            String loser = rlPlayerWon ? opponentPlayer.getName() : rlPlayer.getName();
                            String reason = String.format("Episode %d - Win", epNumber);
                            gameLogger.logOutcome(winner, loser, turns, reason);
                            logger.info("GAME LOGS stats: " + opponentTag + " gameId=" + gameLogger.getGameId());
                            gameLogger.close();
                        }

                        // Log head usage statistics
                        logHeadUsageStats(epNumber, rlPlayer, opponentPlayer, turns, rlPlayerWon,
                                rlPlayerDeckPath, opponentDeckPath);

                        logGameResult(game, rlPlayer);
                        long rewardStartNanos = System.nanoTime();
                        RewardDiag rewardDiag = updateModelBasedOnOutcome(game, rlPlayer, opponentPlayer);
                        double finalReward = rewardDiag.finalReward;
                        long rewardEndNanos = System.nanoTime();
                        metrics.recordRewardUpdateMs((rewardEndNanos - rewardStartNanos) / 1_000_000L);

                        // Defer statistics writing until after we compute episodeSeconds below
                        // -------- Episode duration & counter logging --------
                        long episodeDurationNanos = System.nanoTime() - episodeStartNanos;
                        metrics.recordEpisodeTotalMs(episodeDurationNanos / 1_000_000L);
                        double episodeSeconds = episodeDurationNanos / 1_000_000_000.0;
                        RLTrainer.threadLocalLogger.get().info(String.format("Episode %d completed in %.2f seconds", epNumber, episodeSeconds));

                        if (GAME_STATS_WRITER && epNumber % 100 == 0) {
                            writeAgentStatus(epNumber);
                        }
                        if (TRAIN_DIAG && (TRAIN_DIAG_EVERY <= 1 || epNumber % TRAIN_DIAG_EVERY == 0)) {
                            double totalMs = (System.nanoTime() - episodeStartNanos) / 1_000_000.0;
                            double gameMs = (endGameNanos - startGameNanos) / 1_000_000.0;
                            double rewardMs = (rewardEndNanos - rewardStartNanos) / 1_000_000.0;
                            int active = activeEps().get();
                            logger.info(String.format(
                                    "Episode %d timing: total=%.1fms game=%.1fms reward=%.1fms activeEpisodes=%d thread=%s",
                                    epNumber, totalMs, gameMs, rewardMs, active, Thread.currentThread().getName()
                            ));
                        }

                        // ------------------ Progress logging ------------------
                        if (trainLogEvery > 0 && (epNumber % trainLogEvery == 0 || epNumber == targetEpisodeCount)) {
                            // Avoid multiple threads spamming the same log line for the same epNumber.
                            int prev = lastLoggedEpisode.get();
                            if (epNumber > prev && lastLoggedEpisode.compareAndSet(prev, epNumber)) {
                                int done = Math.max(0, epNumber - startEpisodeCountSnapshot);
                                int remaining = Math.max(0, targetEpisodeCount - epNumber);
                                long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                                double epsPerSec = done / (elapsedMs / 1000.0);
                                long etaSec = epsPerSec > 0 ? (long) (remaining / epsPerSec) : -1;
                                int rlFailures = ComputerPlayerRL.getRLActivationFailureCount();
                                int simTrainSkipped = ComputerPlayerRL.getSimulationTrainingSkippedCount();

                                // Get training stats
                                java.util.Map<String, Integer> mainStats = sharedModel.getMainModelTrainingStats();

                                logger.info(String.format(
                                        "Training progress: episode=%d/%d (run=%d, %.3f eps/s), ETA %ds, RL_activation_failures=%d, sim_training_skipped=%d, games_killed=%d",
                                        epNumber, targetEpisodeCount, done, epsPerSec, etaSec, rlFailures, simTrainSkipped, GameHealthMonitor.getGamesKilled()
                                ));
                                logger.info(String.format(
                                        "  Main model: %d train steps, %d samples",
                                        mainStats.get("train_steps"), mainStats.get("train_samples")
                                ));
                                // Value head quality metrics
                                double valueAccuracy = metrics.getValueAccuracy();
                                double avgWinValue = metrics.getAverageValueForWins();
                                double avgLossValue = metrics.getAverageValueForLosses();
                                logger.info(String.format(
                                        "  Value head: accuracy=%.1f%%, avg_win=%.3f, avg_loss=%.3f (target: +1/-1)",
                                        valueAccuracy * 100, avgWinValue, avgLossValue
                                ));
                                // Periodic CSV for value accuracy trend
                                if (GAME_STATS_WRITER) {
                                    String statsDir2 = profileContext != null
                                            ? profileContext.paths.logsBaseDir + "/stats"
                                            : logsBaseDir() + "/stats";
                                    Path vaPath = Paths.get(statsDir2, "value_accuracy.csv");
                                    String vaHeader = "episode,timestamp,value_accuracy,avg_win_value,avg_loss_value\n";
                                    String vaLine = String.format("%d,%s,%.4f,%.4f,%.4f%n",
                                            epNumber,
                                            java.time.Instant.now().toString(),
                                            valueAccuracy, avgWinValue, avgLossValue);
                                    ASYNC_LINE_WRITER.append(vaPath, vaHeader, vaLine);
                                }
                                // MCTS-threshold trigger: when value_accuracy crosses VALUE_ACCURACY_MCTS_THRESHOLD,
                                // log a one-shot ALERT so we know to schedule a MCTS A/B run.
                                if (valueAccuracy >= VALUE_ACCURACY_MCTS_THRESHOLD
                                        && MCTS_THRESHOLD_FIRED.compareAndSet(false, true)) {
                                    logger.info(String.format(
                                            "[MCTS-THRESHOLD] value_accuracy=%.3f >= threshold %.2f at episode %d -- ready for MCTS A/B eval",
                                            valueAccuracy, VALUE_ACCURACY_MCTS_THRESHOLD, epNumber));
                                    System.out.println(String.format(
                                            "[MCTS-THRESHOLD] value_accuracy=%.3f >= %.2f at ep=%d -- time to run MCTS eval",
                                            valueAccuracy, VALUE_ACCURACY_MCTS_THRESHOLD, epNumber));
                                }
                                logger.info(String.format(
                                        "  Reward diag (last ep): won=%s steps=%d/%d opp_steps=%d/%d finalReward=%.3f mc_return0=%.3f sum_rewards=%.3f last_reward=%.3f",
                                        rewardDiag.won, rewardDiag.steps, rewardDiag.rawSteps,
                                        rewardDiag.opponentSteps, rewardDiag.opponentRawSteps,
                                        rewardDiag.finalReward, rewardDiag.mcReturn0,
                                        rewardDiag.sumRewards, rewardDiag.lastReward
                                ));
                            }
                        }

                        // ------------------ Statistics ------------------
                        String opponentType = opponentTag;
                        logger.info(String.format("Episode %d summary: turns=%d, reward=%.3f, opponent=%s, winrate=%.3f (%d games)",
                                epNumber, turns, finalReward, opponentType, snapshotWinrate, snapshotSampleSize));

                        // League tick (runs only on specific ep boundaries, single-threaded)
                        try {
                            maybeRunLeagueTick(epNumber);
                            maybeRunLadderTick(epNumber);
                        } catch (Exception e) {
                            logger.warn("League/Ladder tick failed at ep " + epNumber, e);
                        }

                        if (GAME_STATS_WRITER) {
                            Path statsPath = Paths.get(statsFilePath());
                            String statsHeader = "episode,turns,final_reward,opponent_type,winrate,episode_seconds\n";
                            String statsLine = new StringBuilder()
                                    .append(epNumber).append(',').append(turns).append(',')
                                    .append(String.format("%.3f", finalReward)).append(',')
                                    .append(opponentType).append(',')
                                    .append(String.format("%.3f", snapshotWinrate)).append(',')
                                    .append(String.format("%.2f", episodeSeconds)).append('\n')
                                    .toString();
                            ASYNC_LINE_WRITER.append(statsPath, statsHeader, statsLine);

                            // Also log per-deck-pairing outcome to training_cells.csv so we
                            // can reconstruct per-deck rolling training winrate later.
                            // Strip directory and extension, keep just the filename.
                            String rlDeckName = rlPlayerDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "");
                            String oppDeckName = opponentDeckPath != null
                                    ? opponentDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "")
                                    : "unknown";
                            String statsDir = profileContext != null
                                    ? profileContext.paths.logsBaseDir + "/stats"
                                    : logsBaseDir() + "/stats";
                            Path cellsPath = Paths.get(statsDir, "training_cells.csv");
                            String cellsHeader = "episode,rl_deck,opp_deck,opponent_type,final_reward,turns,episode_seconds\n";
                            String cellsLine = new StringBuilder()
                                    .append(epNumber).append(',')
                                    .append(rlDeckName).append(',')
                                    .append(oppDeckName).append(',')
                                    .append(opponentType).append(',')
                                    .append(String.format("%.3f", finalReward)).append(',')
                                    .append(turns).append(',')
                                    .append(String.format("%.2f", episodeSeconds)).append('\n')
                                    .toString();
                            ASYNC_LINE_WRITER.append(cellsPath, cellsHeader, cellsLine);

                            // Update per-deck rolling winrate for matchup-balanced sampling.
                            recordDeckTrainingOutcome(rlPlayerDeckPath.getFileName().toString(), rlPlayerWon);
                        }

                        // Record opponent side stats for meta RL opponents
                        if (opponentPlayer instanceof ComputerPlayerRL
                                && (opponentTag.startsWith("META(") || "MIRROR".equals(opponentTag))) {
                            String oppProfileName = "MIRROR".equals(opponentTag)
                                    ? profileName()
                                    : opponentTag.substring(5, opponentTag.length() - 1);
                            ProfileContext oppCtx = ProfileContext.byName(oppProfileName);
                            if (oppCtx != null) {
                                boolean oppWon = !rlPlayerWon;
                                // Record to opponent's rolling winrate
                                ConcurrentLinkedQueue<Boolean> oppWins = oppCtx.recentWins;
                                AtomicInteger oppWc = oppCtx.winCount;
                                oppWins.add(oppWon);
                                if (oppWon) oppWc.incrementAndGet();
                                while (oppWins.size() > WINRATE_WINDOW) {
                                    Boolean removed = oppWins.poll();
                                    if (removed != null && removed) oppWc.decrementAndGet();
                                }
                                double oppWinrate = oppWins.isEmpty() ? 0.0 : oppWc.get() / (double) oppWins.size();
                                // Write to opponent's CSV
                                if (GAME_STATS_WRITER) {
                                    String myProfile = profileName();
                                    String oppTag;
                                    if ("MIRROR".equals(opponentTag)) {
                                        oppTag = "MIRROR";
                                    } else {
                                        oppTag = myProfile.isEmpty() ? "META(unknown)" : "META(" + myProfile + ")";
                                    }
                                    Path oppStatsPath = Paths.get(oppCtx.paths.trainingStatsPath);
                                    double oppReward = oppWon ? 1.0 : -1.0;
                                    String oppLine = new StringBuilder()
                                            .append(epNumber).append(',').append(turns).append(',')
                                            .append(String.format("%.3f", oppReward)).append(',')
                                            .append(oppTag).append(',')
                                            .append(String.format("%.3f", oppWinrate)).append(',')
                                            .append(String.format("%.2f", episodeSeconds)).append('\n')
                                            .toString();
                                    String oppHeader = "episode,turns,final_reward,opponent_type,winrate,episode_seconds\n";
                                    ASYNC_LINE_WRITER.append(oppStatsPath, oppHeader, oppLine);
                                }
                            }
                        }

                        metrics.recordEpisodeCompleted();
                        activeEps().decrementAndGet();
                        metrics.setActiveEpisodes(activeEps().get());

                        // Release game objects so GC can reclaim them.
                        // The normal server path calls match.getGames().clear()
                        // during cleanup -- without this, Match.games holds a
                        // reference to Game, preventing collection of the entire
                        // game object graph (GameStates, cards, LKI maps, etc.)
                        game.end();
                        match.getGames().clear();
                        match = null;
                        game = null;
                        rlPlayer = null;
                        opponentPlayer = null;
                        healthMonitor = null;
                        gameLogger = null;
                        threadLocalGameLogger.remove();
                        StateSequenceBuilder.clearThreadLocalKnownArchetypeLabels();
                      } catch (Exception e) {
                        // Log but don't kill the runner thread -- keep playing games
                        logger.error("Game runner exception (continuing): " + e.getMessage());
                        activeEps().decrementAndGet();
                        metrics.setActiveEpisodes(activeEps().get());
                        StateSequenceBuilder.clearThreadLocalKnownArchetypeLabels();
                      }
                    }
                    return null;
                });
                futures.add(future);
            }

            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

            for (Future<Void> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    logger.error("Error in thread execution", e.getCause());
                    throw new RuntimeException(e.getCause());
                }
            }

            ASYNC_LINE_WRITER.close();

            // Record end time and log statistics
            long endTime = System.nanoTime();
            long totalTime = endTime - startTime;
            double totalTimeInMinutes = totalTime / 1_000_000_000.0 / 60.0;
            int episodesRun = episodeCounter().get();
            double gamesRunPerMinute = totalTimeInMinutes > 0 ? episodesRun / totalTimeInMinutes : 0;

            // Stop health stats logging
            healthStats.stop();

            logger.info("Training completed:");
            logger.info("Total Games Run: " + episodesRun);
            logger.info("Games Run Per Minute: " + gamesRunPerMinute);
            logger.info("Total Training Time: " + (totalTime / 1_000_000_000.0) + " seconds");
            logger.info("Health summary: " + healthStats.getSummary());
            logger.info("MCTS_GATE: " + ComputerPlayerRL.getMctsGateStats());
            logger.info(PolicyValueMCTS.getMctsTimingStats());

            if (!multiProfile) {
                ACTOR_LEARNER.shutdownAndDrain(ACTOR_LEARNER_DRAIN_TIMEOUT_MS);
                sharedModel.shutdown();
            }
            if (heartbeat != null) {
                heartbeat.shutdownNow();
            }
        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
    }

    // ============================================================
    // Multi-profile training: load card DB once, run N profiles
    // ============================================================

    public static void trainMultiProfile(List<String> profileNames) {
        if (profileNames == null || profileNames.isEmpty()) {
            logger.error("trainMultiProfile: no profile names provided");
            return;
        }
        logger.info("Multi-profile training: " + profileNames.size() + " profiles: " + profileNames);

        // 1) Load card database ONCE (the whole point of this refactor)
        System.out.println("Multi-profile: Initializing card database...");
        mage.cards.repository.CardScanner.scan();
        System.out.println("Multi-profile: Card database initialized");

        // 2) Load shared deck pools once
        List<Path> sharedDeckPool;
        List<Path> sharedAgentDeckPool;
        try {
            sharedDeckPool = loadDeckPool();
            if (RL_AGENT_DECK_LIST_FILE != null && !RL_AGENT_DECK_LIST_FILE.trim().isEmpty()) {
                sharedAgentDeckPool = loadDeckPoolWithOverride(RL_AGENT_DECK_LIST_FILE);
            } else {
                sharedAgentDeckPool = sharedDeckPool;
            }
        } catch (IOException e) {
            logger.error("Multi-profile: failed to load deck pool", e);
            return;
        }
        logger.info("Multi-profile: deck pool loaded (" + sharedDeckPool.size() + " decks)");

        // 3) Create profile contexts
        String artifactsRoot = EnvConfig.str("RL_ARTIFACTS_ROOT",
                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl");
        int totalRunners = NUM_GAME_RUNNERS;
        int runnersPerProfile = Math.max(1, totalRunners / profileNames.size());
        logger.info("Multi-profile: " + totalRunners + " total runners, "
                + runnersPerProfile + " per profile");

        // Load per-profile agent deck overrides from registry
        java.util.Map<String, String> profileAgentDecks = new java.util.HashMap<>();
        try {
            Path registryPath = leagueRegistryPath();
            if (Files.exists(registryPath)) {
                String regJson = new String(Files.readAllBytes(registryPath), java.nio.charset.StandardCharsets.UTF_8);
                com.google.gson.JsonArray entries = com.google.gson.JsonParser.parseString(regJson).getAsJsonArray();
                for (com.google.gson.JsonElement el : entries) {
                    com.google.gson.JsonObject obj = el.getAsJsonObject();
                    String pName = obj.get("profile").getAsString().trim();
                    if (obj.has("train_env")) {
                        com.google.gson.JsonObject trainEnv = obj.getAsJsonObject("train_env");
                        if (trainEnv.has("RL_AGENT_DECK_LIST")) {
                            profileAgentDecks.put(pName, trainEnv.get("RL_AGENT_DECK_LIST").getAsString().trim());
                        }
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("Multi-profile: failed to load registry for per-profile decks: " + e.getMessage());
        }

        List<RLTrainer> trainers = new ArrayList<>();
        for (String name : profileNames) {
            ProfilePaths paths = new ProfilePaths(name, artifactsRoot);
            // Ensure profile directories exist
            new java.io.File(paths.modelsBaseDir).mkdirs();
            new java.io.File(paths.logsBaseDir).mkdirs();
            new java.io.File(paths.snapshotDir).mkdirs();
            new java.io.File(paths.pythonLogsDir).mkdirs();
            new java.io.File(paths.trainingGameLogsDir).mkdirs();
            new java.io.File(paths.evalGameLogsDir).mkdirs();
            ProfileContext ctx = new ProfileContext(name, paths);
            // Resume episode counter from existing training stats CSV
            int resumedEp = readMaxEpisodeFromCsv(paths.trainingStatsPath);
            if (resumedEp > 0) {
                ctx.episodeCounter.set(resumedEp);
                logger.info("Resumed episode counter for " + name + " at " + resumedEp);
            }
            ctx.deckFiles = sharedDeckPool;
            // Per-profile agent deck from registry RL_AGENT_DECK_LIST
            String agentDeckOverride = profileAgentDecks.get(name);
            if (agentDeckOverride != null && !agentDeckOverride.isEmpty()) {
                try {
                    ctx.agentDeckFiles = loadDeckPoolWithOverride(agentDeckOverride);
                    logger.info("Profile " + name + ": agent deck = " + agentDeckOverride
                            + " (" + ctx.agentDeckFiles.size() + " decks)");
                } catch (IOException e) {
                    logger.warn("Profile " + name + ": failed to load agent deck " + agentDeckOverride + ", using shared pool");
                    ctx.agentDeckFiles = sharedAgentDeckPool;
                }
            } else {
                ctx.agentDeckFiles = sharedAgentDeckPool;
                logger.info("Profile " + name + ": using shared agent deck pool (" + sharedAgentDeckPool.size() + " decks)");
            }
            ctx.numGameRunners = runnersPerProfile;
            ProfileContext.register(ctx);
            RLTrainer trainer = new RLTrainer(ctx);
            trainers.add(trainer);
        }

        // 4) Launch all profiles in parallel (one thread per profile coordinator)
        ExecutorService profileExecutor = Executors.newFixedThreadPool(profileNames.size(), r -> {
            Thread t = new Thread(r, "PROFILE-COORD");
            t.setDaemon(false);
            return t;
        });
        List<Future<?>> profileFutures = new ArrayList<>();
        for (RLTrainer trainer : trainers) {
            profileFutures.add(profileExecutor.submit(() -> {
                Thread.currentThread().setName("PROFILE-" + trainer.profileContext.profileName);
                ProfileContext.setCurrent(trainer.profileContext);
                try {
                    trainer.train();
                } catch (Exception e) {
                    logger.error("Profile " + trainer.profileContext.profileName + " failed", e);
                }
            }));
        }

        // Wait for all profiles to finish
        profileExecutor.shutdown();
        for (Future<?> f : profileFutures) {
            try {
                f.get();
            } catch (Exception e) {
                logger.error("Profile future failed", e);
            }
        }
        ACTOR_LEARNER.shutdownAndDrain(ACTOR_LEARNER_DRAIN_TIMEOUT_MS);
        logger.info("Multi-profile training complete for all " + profileNames.size() + " profiles");
        String mctsGate = "MCTS_GATE: " + ComputerPlayerRL.getMctsGateStats();
        String mctsTiming = PolicyValueMCTS.getMctsTimingStats();
        logger.info(mctsGate);
        logger.info(mctsTiming);
        System.out.println(mctsGate);
        System.out.println(mctsTiming);
    }

    private String formatOpponentTag(Player opponentPlayer) {
        if (opponentPlayer instanceof ComputerPlayerRL) {
            String name = opponentPlayer.getName();
            if (name != null && name.startsWith("Meta-")) {
                String oppProfile = name.substring(5);
                String self = profileName();
                if (!self.isEmpty() && self.equals(oppProfile)) {
                    return "MIRROR";
                }
                return "META(" + oppProfile + ")";
            }
            return "SELFPLAY";
        }
        if (opponentPlayer instanceof ComputerPlayer7) {
            String name;
            try {
                name = opponentPlayer.getName();
            } catch (Exception ignored) {
                name = "";
            }
            if (name == null) {
                name = "";
            }
            String n = name.toLowerCase();
            int skill = 0;
            if (n.contains("weak")) {
                skill = 1;
            } else if (n.contains("medium")) {
                skill = 2;
            } else if (n.contains("strong")) {
                skill = 3;
            } else {
                int idx = n.indexOf("skill");
                if (idx >= 0) {
                    int j = idx + 5;
                    int start = j;
                    while (j < n.length() && Character.isDigit(n.charAt(j))) {
                        j++;
                    }
                    if (j > start) {
                        try {
                            skill = Integer.parseInt(n.substring(start, j));
                        } catch (NumberFormatException ignored) {
                            skill = 0;
                        }
                    }
                }
            }
            return "CP7-Skill " + skill;
        }
        try {
            return opponentPlayer == null ? "UNKNOWN" : opponentPlayer.getName();
        } catch (Exception ignored) {
            return "UNKNOWN";
        }
    }

    public void eval(int numEpisodesPerThread) {
        runEvaluation(numEpisodesPerThread);
    }

    public static double runEvaluation(int numEpisodesPerThread) {
        List<Path> deckFiles;
        List<Path> agentDeckFiles;
        try {
            deckFiles = loadDeckPool();
            // Load agent deck pool (may be separate from opponent pool)
            if (RL_AGENT_DECK_LIST_FILE != null && !RL_AGENT_DECK_LIST_FILE.trim().isEmpty()) {
                agentDeckFiles = loadDeckPoolWithOverride(RL_AGENT_DECK_LIST_FILE);
                logger.info("Eval: Agent deck pool: " + agentDeckFiles.size() + " decks (from RL_AGENT_DECK_LIST)");
            } else {
                agentDeckFiles = deckFiles;  // default: same pool
            }
        } catch (Exception e) {
            logger.error("Error during evaluation", e);
            return 0.0;
        }

        // Reset failure counter at start of evaluation
        ComputerPlayerRL.resetRLActivationFailureCount();
        logger.info("Starting evaluation - RL activation failure counter reset to 0");

        final int evalLogEvery = EnvConfig.i32("EVAL_LOG_EVERY", 50);
        final int totalEvalGames = Math.max(1, numEpisodesPerThread * NUM_GAME_RUNNERS);
        final long evalStartMs = System.currentTimeMillis();
        final AtomicInteger evalCounter = new AtomicInteger(0);
        final AtomicInteger lastLoggedEval = new AtomicInteger(0);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS, runnable -> {
            Thread thread = new Thread(runnable);
            thread.setPriority(Thread.MIN_PRIORITY);
            return thread;
        });
        final Object lock = new Object();
        final boolean[] isFirstThread = {true};

        List<Future<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
            final int evalRunnerIndex = i;
            Future<Integer> future = executor.submit(() -> {
                boolean isFirst;
                synchronized (lock) {
                    isFirst = isFirstThread[0];
                    isFirstThread[0] = false;
                }

                Logger currentLogger = threadLocalLogger.get();
                if (isFirst) {
                    currentLogger.setLevel(Level.INFO);
                }

                int localWinsAgainstComputerPlayer7 = 0;

                Thread.currentThread().setName("GAME");
                Random threadRand = newSeededRandom(2_000L + evalRunnerIndex);

                for (int evalEpisode = 0; evalEpisode < numEpisodesPerThread; evalEpisode++) {
                    Path rlPlayerDeckPath = agentDeckFiles.get(threadRand.nextInt(agentDeckFiles.size()));
                    Deck rlPlayerDeckThread = new RLTrainer().loadDeck(rlPlayerDeckPath.toString());
                    Path opponentDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
                    Deck opponentDeckThread = new RLTrainer().loadDeck(opponentDeckPath.toString());
                    if (rlPlayerDeckThread == null || opponentDeckThread == null) {
                        logger.warn("Eval: failed to load deck(s) for game, skipping.");
                        continue;
                    }

                    TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));
                    try {
                        match.startGame();
                    } catch (GameException e) {
                        logger.error("Error starting game", e);
                        continue;
                    }
                    Game game = match.getGames().get(0);

                    // Increment counter first to get unique episode number (avoid race condition in parallel execution)
                    int currentEvalGame = evalCounter.incrementAndGet();

                    // Enable game logging for eval: either GAME_LOGGING=1 or GAME_LOG_FREQUENCY applies
                    boolean enableGameLogging = "1".equals(System.getenv().getOrDefault("GAME_LOGGING", "0"))
                            || "true".equalsIgnoreCase(System.getenv().getOrDefault("GAME_LOGGING", "0"))
                            || (GAME_LOG_FREQUENCY > 0 && (currentEvalGame == 1 || currentEvalGame % GAME_LOG_FREQUENCY == 0));
                    GameLogger gameLogger = GameLogger.create(enableGameLogging);
                    threadLocalGameLogger.set(gameLogger);

                    // Greedy evaluation: use deterministic arg-max player, training DISABLED so
                    // the MCTS-eval-override gate (ISMCTS_ENABLE && !trainingEnabled) actually fires.
                    ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel, true, false, "train");
                    rlPlayer.setCurrentEpisode(-currentEvalGame); // Negative for eval = deterministic mulligan
                    rlPlayer.setAttachedGameLogger(gameLogger);
                    game.addPlayer(rlPlayer, rlPlayerDeckThread);
                    match.addPlayer(rlPlayer, rlPlayerDeckThread);

                    // Use stronger opponent for evaluation (skill=6) - reserved for benchmarking
                    int evalSkill = EnvConfig.i32("EVAL_OPPONENT_SKILL", 6);
                    ComputerPlayer7 opponent = new ComputerPlayer7("EvalBot-Skill" + evalSkill, RangeOfInfluence.ALL, evalSkill);
                    game.addPlayer(opponent, opponentDeckThread);
                    match.addPlayer(opponent, opponentDeckThread);

                    game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                    game.loadCards(opponentDeckThread.getCards(), opponent.getId());

                    GameOptions options = new GameOptions();
                    game.setGameOptions(options);

                    // Start health monitor for eval games
                    GameHealthMonitor healthMonitor = GameHealthMonitor.createAndStart(game);

                    game.start(rlPlayer.getId());

                    healthMonitor.stop();

                    boolean rlPlayerWon;
                    if (healthMonitor.wasKilled()) {
                        logger.warn("Eval game " + currentEvalGame + " killed by health monitor: " + healthMonitor.getKillReason());
                        rlPlayerWon = false;
                    } else {
                        rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
                    }

                    // Log game outcome to detailed game log if enabled
                    if (gameLogger.isEnabled()) {
                        String winner = rlPlayerWon ? rlPlayer.getName() : opponent.getName();
                        String loser = rlPlayerWon ? opponent.getName() : rlPlayer.getName();
                        int turns = game.getTurnNum();
                        String reason = String.format("Eval game %d", currentEvalGame);
                        gameLogger.logOutcome(winner, loser, turns, reason);
                        gameLogger.close();
                    }

                    if (isFirst) {
                        logStaticGameResult(game, rlPlayer);
                    }

                    if (rlPlayerWon) {
                        localWinsAgainstComputerPlayer7++;
                    }

                    int done = currentEvalGame; // Already incremented before game started
                    if (evalLogEvery > 0 && done % evalLogEvery == 0) {
                        int prev = lastLoggedEval.get();
                        if (done > prev && lastLoggedEval.compareAndSet(prev, done)) {
                            long elapsedMs = Math.max(1, System.currentTimeMillis() - evalStartMs);
                            double epsPerSec = done / (elapsedMs / 1000.0);
                            int remaining = Math.max(0, totalEvalGames - done);
                            long etaSec = epsPerSec > 0 ? (long) (remaining / epsPerSec) : -1;
                            logger.info(String.format(
                                    "Eval progress: %d/%d (%.3f eps/s), ETA %ds",
                                    done, totalEvalGames, epsPerSec, etaSec
                            ));
                        }
                    }
                }
                return localWinsAgainstComputerPlayer7;
            });
            futures.add(future);
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            logger.error("Evaluation interrupted", e);
            Thread.currentThread().interrupt();
        }

        int totalWins = 0;
        for (Future<Integer> future : futures) {
            try {
                totalWins += future.get();
            } catch (InterruptedException | ExecutionException e) {
                logger.error("Error getting evaluation results", e);
            }
        }
        double winRate = (double) totalWins / (numEpisodesPerThread * NUM_GAME_RUNNERS);
        int rlFailures = ComputerPlayerRL.getRLActivationFailureCount();
        int mctsActivations = ComputerPlayerRL.getMctsActivationCount();
        logger.info(String.format("Evaluation win rate: %.4f, RL_activation_failures=%d, mcts_activations=%d", winRate, rlFailures, mctsActivations));
        System.out.println(String.format("EVAL_SUMMARY: wins=%d played=%d winrate=%.4f mcts_activations=%d",
                totalWins, numEpisodesPerThread * NUM_GAME_RUNNERS, winRate, mctsActivations));
        System.out.println("MCTS_GATE: " + ComputerPlayerRL.getMctsGateStats());
        System.out.println(PolicyValueMCTS.getMctsTimingStats());
        return winRate;
    }

    /**
     * Benchmark: round-robin RL-vs-ComputerPlayer7 over an explicit deck pool.
     * Writes a simple CSV to the trajectories dir if configured, otherwise
     * logs.
     */
    public void runBenchmark(int gamesPerMatchup) {
        try {
            mage.cards.repository.CardScanner.scan();
            List<Path> decks = loadDeckPool();
            if (decks.size() < 2) {
                logger.warn("Benchmark requires at least 2 decks in the deck pool; found " + decks.size());
                return;
            }

            final int benchThreads = EnvConfig.i32("BENCHMARK_THREADS", Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
            final int logEvery = EnvConfig.i32("BENCHMARK_LOG_EVERY", 5);
            final int heartbeatSec = EnvConfig.i32("BENCHMARK_HEARTBEAT_SEC", 30);
            final int gameTimeoutSec = EnvConfig.i32("BENCHMARK_GAME_TIMEOUT_SEC", 900);
            final int totalPlannedGames = decks.size() * (decks.size() - 1) * gamesPerMatchup;

            final AtomicLong completed = new AtomicLong(0);
            final AtomicLong started = new AtomicLong(0);
            final AtomicLong winsTotal = new AtomicLong(0);
            final long startMs = System.currentTimeMillis();

            final ConcurrentHashMap<String, AtomicLong> matchupWins = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> matchupGames = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> deckWinsAsRL = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> deckGamesAsRL = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> deckWinsVsOpp = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> deckGamesVsOpp = new ConcurrentHashMap<>();

            final String liveReportPathStr = EnvConfig.str("BENCHMARK_LIVE_REPORT_PATH",
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/benchmark_live_report.txt");
            final Path liveReportPath = Paths.get(liveReportPathStr);
            final AtomicLong lastLiveReportDone = new AtomicLong(0);

            ExecutorService exec = Executors.newFixedThreadPool(benchThreads, r -> {
                Thread t = new Thread(r);
                // XMage requires game code to run in threads named with GAME prefix
                t.setName("GAME-BENCH");
                t.setPriority(Thread.NORM_PRIORITY);
                return t;
            });

            List<Future<Void>> futures = new ArrayList<>();

            logger.info(String.format(
                    "Benchmark started: decks=%d, gamesPerMatchup=%d, plannedGames=%d, threads=%d (logEvery=%d, heartbeat=%ds)",
                    decks.size(), gamesPerMatchup, totalPlannedGames, benchThreads, logEvery, heartbeatSec
            ));
            logger.info("Python device info: " + sharedModel.getDeviceInfo());

            // Heartbeat so users aren't staring at a blank console while the first games warm up.
            // (ETA requires at least 1 completed game.)
            java.util.concurrent.ScheduledExecutorService heartbeat = null;
            if (heartbeatSec > 0) {
                heartbeat = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
                    Thread t = new Thread(r, "BENCH-HEARTBEAT");
                    t.setDaemon(true);
                    return t;
                });
                final java.util.concurrent.ScheduledExecutorService hbRef = heartbeat;
                heartbeat.scheduleAtFixedRate(() -> {
                    long done = completed.get();
                    if (done >= totalPlannedGames) {
                        hbRef.shutdown();
                        return;
                    }
                    if (done == 0) {
                        logger.info(String.format(
                                "Benchmark heartbeat: %d/%d games done (started=%d; warming up; ETA after first completion)",
                                done, totalPlannedGames, started.get()));
                    } else {
                        long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                        double gamesPerSec = done / (elapsedMs / 1000.0);
                        long remaining = Math.max(0, totalPlannedGames - done);
                        long etaSec = gamesPerSec > 0 ? (long) (remaining / gamesPerSec) : -1;
                        logger.info(String.format(
                                "Benchmark heartbeat: %d/%d games done (started=%d; %.2f games/s), ETA %ds",
                                done, totalPlannedGames, started.get(), gamesPerSec, etaSec));
                    }
                }, heartbeatSec, heartbeatSec, java.util.concurrent.TimeUnit.SECONDS);
            }

            for (int i = 0; i < decks.size(); i++) {
                for (int j = 0; j < decks.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    final Path p1 = decks.get(i);
                    final Path p2 = decks.get(j);
                    final String matchupKey = p1.getFileName() + " vs " + p2.getFileName();
                    final String rlDeckKey = String.valueOf(p1.getFileName());
                    final String oppDeckKey = String.valueOf(p2.getFileName());

                    matchupWins.putIfAbsent(matchupKey, new AtomicLong(0));
                    matchupGames.putIfAbsent(matchupKey, new AtomicLong(0));
                    deckWinsAsRL.putIfAbsent(rlDeckKey, new AtomicLong(0));
                    deckGamesAsRL.putIfAbsent(rlDeckKey, new AtomicLong(0));
                    deckWinsVsOpp.putIfAbsent(oppDeckKey, new AtomicLong(0));
                    deckGamesVsOpp.putIfAbsent(oppDeckKey, new AtomicLong(0));

                    for (int g = 0; g < gamesPerMatchup; g++) {
                        futures.add(exec.submit(() -> {
                            Thread.currentThread().setName("GAME-BENCH");
                            long s = started.incrementAndGet();
                            if (s % logEvery == 0 || s == totalPlannedGames) {
                                logger.info(String.format("Benchmark started games: %d/%d", s, totalPlannedGames));
                            }

                            boolean win = runSingleBenchmarkGame(p1, p2, gameTimeoutSec);
                            deckGamesAsRL.get(rlDeckKey).incrementAndGet();
                            deckGamesVsOpp.get(oppDeckKey).incrementAndGet();
                            matchupGames.get(matchupKey).incrementAndGet();
                            if (win) {
                                matchupWins.get(matchupKey).incrementAndGet();
                                winsTotal.incrementAndGet();
                                deckWinsAsRL.get(rlDeckKey).incrementAndGet();
                                deckWinsVsOpp.get(oppDeckKey).incrementAndGet();
                            }

                            long done = completed.incrementAndGet();
                            if (done % logEvery == 0 || done == totalPlannedGames) {
                                long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                                double gamesPerSec = done / (elapsedMs / 1000.0);
                                long remaining = Math.max(0, totalPlannedGames - done);
                                long etaSec = gamesPerSec > 0 ? (long) (remaining / gamesPerSec) : -1;
                                logger.info(String.format(
                                        "Benchmark progress: %d/%d games done (%.2f games/s), ETA %ds",
                                        done, totalPlannedGames, gamesPerSec, etaSec
                                ));

                                // Live report file (overwrite) so you can tail it during eval/benchmark.
                                long prev = lastLiveReportDone.get();
                                if (done > prev && lastLiveReportDone.compareAndSet(prev, done)) {
                                    writeBenchmarkLiveReport(
                                            liveReportPath,
                                            done,
                                            totalPlannedGames,
                                            startMs,
                                            started.get(),
                                            winsTotal.get(),
                                            matchupWins,
                                            matchupGames,
                                            deckWinsAsRL,
                                            deckGamesAsRL,
                                            deckWinsVsOpp,
                                            deckGamesVsOpp
                                    );
                                }
                            }
                            return null;
                        }));
                    }
                }
            }

            exec.shutdown();
            exec.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    logger.warn("Benchmark game task failed", e);
                }
            }
            if (heartbeat != null) {
                heartbeat.shutdownNow();
            }

            // Per-matchup summary
            java.util.List<String> matchupKeys = new java.util.ArrayList<>(matchupGames.keySet());
            java.util.Collections.sort(matchupKeys);
            for (String matchupKey : matchupKeys) {
                long games = matchupGames.get(matchupKey) == null ? 0L : matchupGames.get(matchupKey).get();
                long wins = matchupWins.get(matchupKey) == null ? 0L : matchupWins.get(matchupKey).get();
                double wr = games > 0 ? (double) wins / games : 0.0;
                logger.info("Benchmark matchup: " + matchupKey + " winRate=" + String.format("%.3f", wr)
                        + " (" + wins + "/" + games + ")");
            }

            long totalGames = completed.get();
            long totalWins = winsTotal.get();
            double overall = totalGames > 0 ? (double) totalWins / totalGames : 0.0;
            logger.info("Benchmark overall win rate vs heuristic across pool: " + String.format("%.3f", overall)
                    + " (" + totalWins + "/" + totalGames + ")");

            // Per-deck summaries
            java.util.List<String> deckKeys = new java.util.ArrayList<>(deckGamesAsRL.keySet());
            java.util.Collections.sort(deckKeys);

            logger.info("Benchmark winrate WITH each deck (RL piloting deck):");
            for (String dk : deckKeys) {
                long games = deckGamesAsRL.get(dk) == null ? 0L : deckGamesAsRL.get(dk).get();
                long wins = deckWinsAsRL.get(dk) == null ? 0L : deckWinsAsRL.get(dk).get();
                double wr = games > 0 ? (double) wins / games : 0.0;
                logger.info("  WITH " + dk + ": winRate=" + String.format("%.3f", wr) + " (" + wins + "/" + games + ")");
            }

            java.util.List<String> oppKeys = new java.util.ArrayList<>(deckGamesVsOpp.keySet());
            java.util.Collections.sort(oppKeys);
            logger.info("Benchmark winrate AGAINST each deck (opponent deck):");
            for (String dk : oppKeys) {
                long games = deckGamesVsOpp.get(dk) == null ? 0L : deckGamesVsOpp.get(dk).get();
                long wins = deckWinsVsOpp.get(dk) == null ? 0L : deckWinsVsOpp.get(dk).get();
                double wr = games > 0 ? (double) wins / games : 0.0;
                logger.info("  VS " + dk + ": winRate=" + String.format("%.3f", wr) + " (" + wins + "/" + games + ")");
            }

            // Final write (ensures report exists even if logEvery is huge)
            writeBenchmarkLiveReport(
                    liveReportPath,
                    totalGames,
                    totalPlannedGames,
                    startMs,
                    started.get(),
                    totalWins,
                    matchupWins,
                    matchupGames,
                    deckWinsAsRL,
                    deckGamesAsRL,
                    deckWinsVsOpp,
                    deckGamesVsOpp
            );
        } catch (Exception e) {
            logger.error("Benchmark failed", e);
        }
    }

    public void runLeagueEvaluation() {
        try {
            mage.cards.repository.CardScanner.scan();
            java.util.ArrayList<String> skipped = new java.util.ArrayList<>();
            java.util.List<LeagueEvalEntrant> entrants = loadLeagueEvalEntrants(skipped);
            if (entrants.isEmpty()) {
                logger.warn("League eval: no valid entrants from registry/status; nothing to do.");
                writeLeagueEvalReport(Paths.get(LEAGUE_REPORTS_DIR), "no_run", java.time.Instant.now().toString(),
                        java.util.Collections.emptyList(), java.util.Collections.emptyMap(),
                        java.util.Collections.emptyMap(), java.util.Collections.emptyMap(),
                        skipped, java.util.Collections.emptyList());
                return;
            }

            int minEpisode = entrants.stream().mapToInt(e -> e.episode).min().orElse(0);
            int maxEpisode = entrants.stream().mapToInt(e -> e.episode).max().orElse(0);
            String timestamp = java.time.Instant.now().toString();

            Path reportsDir = Paths.get(LEAGUE_REPORTS_DIR);
            Files.createDirectories(reportsDir);
            Path evalStatePath = reportsDir.resolve("eval_state.json");
            int lastMinEpisode = -1;
            try {
                if (Files.exists(evalStatePath)) {
                    String s = new String(Files.readAllBytes(evalStatePath), StandardCharsets.UTF_8);
                    lastMinEpisode = jsonInt(s, "last_min_episode", -1);
                }
            } catch (Exception ignored) {
            }
            if (!LEAGUE_EVAL_FORCE && lastMinEpisode >= 0 && (minEpisode - lastMinEpisode) < Math.max(1, LEAGUE_EVAL_CADENCE_EPISODES)) {
                logger.info(String.format(
                        "League eval skipped by cadence: minEpisode=%d lastMinEpisode=%d cadence=%d force=%s",
                        minEpisode, lastMinEpisode, LEAGUE_EVAL_CADENCE_EPISODES, LEAGUE_EVAL_FORCE
                ));
                return;
            }

            String evalRunId = buildEvalRunId(minEpisode, maxEpisode);
            logger.info(String.format("League eval start: runId=%s entrants=%d gpd=%d k=%.2f",
                    evalRunId, entrants.size(), LEAGUE_ELO_GAMES_PER_DIRECTION, LEAGUE_ELO_K_FACTOR));

            Path matchesCsv = reportsDir.resolve("elo_matches.csv");
            Path ratingsCsv = reportsDir.resolve("elo_ratings.csv");
            Path historyCsv = reportsDir.resolve("elo_history.csv");
            Path pairwiseCsv = reportsDir.resolve("pairwise_matrix.csv");
            Path anchorCsv = reportsDir.resolve("anchor_cp7_skill1.csv");
            Path currentRatingsJson = reportsDir.resolve("elo_current_ratings.json");

            ensureCsvHeader(matchesCsv, "eval_run_id,timestamp,profile_a,profile_b,deck_a,deck_b,wins_a,wins_b,games,wr_a\n");
            ensureCsvHeader(ratingsCsv, "eval_run_id,timestamp,profile,rating,rank,games_played\n");
            ensureCsvHeader(historyCsv, "eval_run_id,timestamp,profile,rating,delta_from_prev\n");
            ensureCsvHeader(pairwiseCsv, "eval_run_id,profile_a,profile_b,wr_a,games\n");
            ensureCsvHeader(anchorCsv, "eval_run_id,timestamp,profile,wins,losses,win_rate,games\n");

            java.util.Map<String, Double> prevRatings = loadCurrentRatings(currentRatingsJson);
            java.util.Map<String, Double> ratings = new java.util.HashMap<>();
            java.util.Map<String, Integer> gamesPlayed = new java.util.HashMap<>();
            for (LeagueEvalEntrant e : entrants) {
                ratings.put(e.profile, prevRatings.getOrDefault(e.profile, 1500.0));
                gamesPlayed.put(e.profile, 0);
            }

            java.util.Map<String, double[]> pairwise = new java.util.LinkedHashMap<>();
            int timeoutSec = Math.max(60, LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC);
            int gpd = Math.max(1, LEAGUE_ELO_GAMES_PER_DIRECTION);
            for (LeagueEvalEntrant a : entrants) {
                for (LeagueEvalEntrant b : entrants) {
                    if (a.profile.equals(b.profile)) {
                        continue;
                    }
                    long winsA = 0L;
                    for (int g = 0; g < gpd; g++) {
                        boolean winA = runSingleLeagueBenchmarkGame(
                                a.deckPath,
                                b.deckPath,
                                timeoutSec,
                                "snap:" + a.snapshotPath.toString(),
                                LeagueOpponentSpec.snapshot("snap:" + b.snapshotPath.toString()),
                                g == 0 && LEAGUE_EVAL_GAME_LOGGING
                        );
                        if (winA) {
                            winsA++;
                        }
                    }
                    double wrA = gpd > 0 ? (double) winsA / (double) gpd : 0.0;
                    appendCsv(matchesCsv, String.format(java.util.Locale.US,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%d,%.6f\n",
                            csv(evalRunId),
                            csv(timestamp),
                            csv(a.profile),
                            csv(b.profile),
                            csv(a.deckPath.toString()),
                            csv(b.deckPath.toString()),
                            winsA,
                            (gpd - winsA),
                            gpd,
                            wrA
                    ));
                    pairwise.put(a.profile + "||" + b.profile, new double[]{wrA, gpd});

                    gamesPlayed.put(a.profile, gamesPlayed.getOrDefault(a.profile, 0) + gpd);
                    gamesPlayed.put(b.profile, gamesPlayed.getOrDefault(b.profile, 0) + gpd);

                    double ra = ratings.getOrDefault(a.profile, 1500.0);
                    double rb = ratings.getOrDefault(b.profile, 1500.0);
                    double ea = 1.0 / (1.0 + Math.pow(10.0, (rb - ra) / 400.0));
                    double delta = LEAGUE_ELO_K_FACTOR * (wrA - ea);
                    ratings.put(a.profile, ra + delta);
                    ratings.put(b.profile, rb - delta);
                }
            }

            java.util.List<String> ranked = new java.util.ArrayList<>(ratings.keySet());
            ranked.sort((x, y) -> Double.compare(ratings.getOrDefault(y, 1500.0), ratings.getOrDefault(x, 1500.0)));

            java.util.Map<String, Double> deltas = new java.util.HashMap<>();
            int rank = 1;
            for (String profile : ranked) {
                double rating = ratings.getOrDefault(profile, 1500.0);
                double prev = prevRatings.getOrDefault(profile, 1500.0);
                double delta = rating - prev;
                deltas.put(profile, delta);
                appendCsv(ratingsCsv, String.format(java.util.Locale.US,
                        "%s,%s,%s,%.6f,%d,%d\n",
                        csv(evalRunId),
                        csv(timestamp),
                        csv(profile),
                        rating,
                        rank,
                        gamesPlayed.getOrDefault(profile, 0)
                ));
                appendCsv(historyCsv, String.format(java.util.Locale.US,
                        "%s,%s,%s,%.6f,%.6f\n",
                        csv(evalRunId),
                        csv(timestamp),
                        csv(profile),
                        rating,
                        delta
                ));
                rank++;
            }

            for (java.util.Map.Entry<String, double[]> e : pairwise.entrySet()) {
                String[] parts = e.getKey().split("\\|\\|", 2);
                String pa = parts.length > 0 ? parts[0] : "";
                String pb = parts.length > 1 ? parts[1] : "";
                double wr = e.getValue()[0];
                int games = (int) e.getValue()[1];
                appendCsv(pairwiseCsv, String.format(java.util.Locale.US,
                        "%s,%s,%s,%.6f,%d\n",
                        csv(evalRunId), csv(pa), csv(pb), wr, games
                ));
            }

            java.util.List<String> anchorSummary = new java.util.ArrayList<>();
            if (LEAGUE_ANCHOR_ENABLE) {
                int anchorGames = Math.max(1, LEAGUE_ANCHOR_GAMES);
                for (LeagueEvalEntrant e : entrants) {
                    int wins = 0;
                    for (int g = 0; g < anchorGames; g++) {
                        boolean win = runSingleLeagueBenchmarkGame(
                                e.deckPath,
                                e.deckPath,
                                timeoutSec,
                                "snap:" + e.snapshotPath.toString(),
                                LeagueOpponentSpec.bot(1),
                                g == 0 && LEAGUE_EVAL_GAME_LOGGING
                        );
                        if (win) {
                            wins++;
                        }
                    }
                    int losses = anchorGames - wins;
                    double wr = (double) wins / (double) anchorGames;
                    appendCsv(anchorCsv, String.format(java.util.Locale.US,
                            "%s,%s,%s,%d,%d,%.6f,%d\n",
                            csv(evalRunId),
                            csv(timestamp),
                            csv(e.profile),
                            wins,
                            losses,
                            wr,
                            anchorGames
                    ));
                    anchorSummary.add(String.format(java.util.Locale.US, "%s %.3f (%d/%d)", e.profile, wr, wins, anchorGames));
                }
            }

            writeCurrentRatings(currentRatingsJson, ratings, evalRunId, minEpisode, maxEpisode, timestamp);
            String evalState = "{\n"
                    + "  \"last_eval_run_id\": " + jsonString(evalRunId) + ",\n"
                    + "  \"last_min_episode\": " + minEpisode + ",\n"
                    + "  \"last_max_episode\": " + maxEpisode + ",\n"
                    + "  \"updated_at\": " + jsonString(timestamp) + "\n"
                    + "}\n";
            Files.write(evalStatePath, evalState.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

            writeLeagueEvalReport(reportsDir, evalRunId, timestamp, ranked, ratings, deltas, pairwise, skipped, anchorSummary);
            logger.info(String.format("League eval done: runId=%s entrants=%d", evalRunId, entrants.size()));
        } catch (Exception e) {
            logger.error("League eval failed", e);
        }
    }

    private static String buildEvalRunId(int minEpisode, int maxEpisode) {
        String ts = java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")
                .withZone(java.time.ZoneOffset.UTC)
                .format(java.time.Instant.now());
        return ts + "_" + minEpisode + "_" + maxEpisode;
    }

    private static void ensureCsvHeader(Path path, String header) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        if (!Files.exists(path)) {
            Files.write(path, header.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        }
    }

    private static void appendCsv(Path path, String line) throws IOException {
        Files.write(path, line.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static String csv(String s) {
        if (s == null) {
            return "\"\"";
        }
        String v = s.replace("\"", "\"\"");
        return "\"" + v + "\"";
    }

    private java.util.List<LeagueEvalEntrant> loadLeagueEvalEntrants(java.util.List<String> skipped) {
        java.util.ArrayList<LeagueEvalEntrant> out = new java.util.ArrayList<>();
        java.util.HashSet<String> seenProfiles = new java.util.HashSet<>();
        Path registryPath = leagueRegistryPath();
        Path baseDir = registryPath.toAbsolutePath().getParent();
        for (LeagueRegistryEntry e : loadLeagueRegistryEntries()) {
            if (!e.active) {
                continue;
            }
            String profile = e.profile == null ? "" : e.profile.trim();
            if (profile.isEmpty() || seenProfiles.contains(profile)) {
                continue;
            }
            Path deckPath = resolveDeckPath(e.deckPath, baseDir);
            if (deckPath == null || !Files.isRegularFile(deckPath)) {
                skipped.add("skip profile=" + profile + " reason=deck_missing path=" + e.deckPath);
                continue;
            }
            Path statusPath = profileAgentStatusPath(profile);
            if (!Files.exists(statusPath)) {
                skipped.add("skip profile=" + profile + " reason=status_missing path=" + statusPath);
                continue;
            }
            try {
                String s = new String(Files.readAllBytes(statusPath), StandardCharsets.UTF_8);
                int episode = jsonInt(s, "episode", 0);
                String snapRaw = jsonStringField(s, "latest_snapshot_path", "").trim();
                Path snapPath = resolveDeckPath(snapRaw, statusPath.toAbsolutePath().getParent());
                if (snapPath == null || !Files.isRegularFile(snapPath)) {
                    snapPath = findLatestPolicyPathForProfile(profile);
                }
                if (snapPath == null || !Files.isRegularFile(snapPath)) {
                    skipped.add("skip profile=" + profile + " reason=snapshot_missing");
                    continue;
                }
                out.add(new LeagueEvalEntrant(profile,
                        deckPath.toAbsolutePath().normalize(),
                        snapPath.toAbsolutePath().normalize(),
                        episode));
                seenProfiles.add(profile);
            } catch (Exception ex) {
                skipped.add("skip profile=" + profile + " reason=status_parse_error");
            }
        }
        out.sort((a, b) -> a.profile.compareToIgnoreCase(b.profile));
        return out;
    }

    private static java.util.Map<String, Double> loadCurrentRatings(Path currentRatingsJson) {
        java.util.HashMap<String, Double> out = new java.util.HashMap<>();
        try {
            if (!Files.exists(currentRatingsJson)) {
                return out;
            }
            String s = new String(Files.readAllBytes(currentRatingsJson), StandardCharsets.UTF_8);
            out.putAll(jsonDoubleMapField(s, "ratings"));
        } catch (Exception ignored) {
        }
        return out;
    }

    private static void writeCurrentRatings(Path currentRatingsJson, java.util.Map<String, Double> ratings,
                                            String evalRunId, int minEpisode, int maxEpisode, String timestamp) {
        try {
            if (currentRatingsJson.getParent() != null) {
                Files.createDirectories(currentRatingsJson.getParent());
            }
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"eval_run_id\": ").append(jsonString(evalRunId)).append(",\n");
            sb.append("  \"min_episode\": ").append(minEpisode).append(",\n");
            sb.append("  \"max_episode\": ").append(maxEpisode).append(",\n");
            sb.append("  \"updated_at\": ").append(jsonString(timestamp)).append(",\n");
            sb.append("  \"ratings\": ").append(jsonDoubleMap(ratings)).append("\n");
            sb.append("}\n");
            Files.write(currentRatingsJson, sb.toString().getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Failed to write current Elo ratings: " + e.getMessage());
        }
    }

    private static void writeLeagueEvalReport(
            Path reportsDir,
            String evalRunId,
            String timestamp,
            java.util.List<String> ranked,
            java.util.Map<String, Double> ratings,
            java.util.Map<String, Double> deltas,
            java.util.Map<String, double[]> pairwise,
            java.util.List<String> skipped,
            java.util.List<String> anchorSummary
    ) {
        try {
            Files.createDirectories(reportsDir);
            Path p = reportsDir.resolve("league_report_latest.md");
            StringBuilder sb = new StringBuilder();
            sb.append("# Pauper League Report\n\n");
            sb.append("- eval_run_id: ").append(evalRunId).append('\n');
            sb.append("- timestamp: ").append(timestamp).append('\n');
            sb.append("- entrants: ").append(ranked.size()).append("\n\n");

            sb.append("## Current Ranking (Snapshot Elo)\n\n");
            sb.append("| Rank | Profile | Elo | Delta |\n");
            sb.append("|---:|---|---:|---:|\n");
            int r = 1;
            for (String profile : ranked) {
                double elo = ratings.getOrDefault(profile, 1500.0);
                double d = deltas.getOrDefault(profile, 0.0);
                sb.append("| ").append(r).append(" | ").append(profile).append(" | ")
                        .append(String.format(java.util.Locale.US, "%.2f", elo)).append(" | ")
                        .append(String.format(java.util.Locale.US, "%+.2f", d)).append(" |\n");
                r++;
            }
            sb.append('\n');

            sb.append("## Top Elo Movers\n\n");
            java.util.List<String> movers = new java.util.ArrayList<>(deltas.keySet());
            movers.sort((a, b) -> Double.compare(Math.abs(deltas.getOrDefault(b, 0.0)), Math.abs(deltas.getOrDefault(a, 0.0))));
            int shown = 0;
            for (String profile : movers) {
                if (shown >= 5) {
                    break;
                }
                sb.append("- ").append(profile).append(": ")
                        .append(String.format(java.util.Locale.US, "%+.2f", deltas.getOrDefault(profile, 0.0))).append('\n');
                shown++;
            }
            sb.append('\n');

            sb.append("## Difficult Pairwise Matchups (Lowest WR for profile_a)\n\n");
            java.util.List<java.util.Map.Entry<String, double[]>> pairs = new java.util.ArrayList<>(pairwise.entrySet());
            pairs.sort((x, y) -> Double.compare(x.getValue()[0], y.getValue()[0]));
            int badN = 0;
            for (java.util.Map.Entry<String, double[]> e : pairs) {
                if (badN >= 8) {
                    break;
                }
                String[] parts = e.getKey().split("\\|\\|", 2);
                String pa = parts.length > 0 ? parts[0] : "";
                String pb = parts.length > 1 ? parts[1] : "";
                sb.append("- ").append(pa).append(" vs ").append(pb)
                        .append(": wr=").append(String.format(java.util.Locale.US, "%.3f", e.getValue()[0]))
                        .append(" games=").append((int) e.getValue()[1]).append('\n');
                badN++;
            }
            sb.append('\n');

            sb.append("## CP7 Skill1 Anchor\n\n");
            if (anchorSummary.isEmpty()) {
                sb.append("- disabled\n\n");
            } else {
                for (String line : anchorSummary) {
                    sb.append("- ").append(line).append('\n');
                }
                sb.append('\n');
            }

            sb.append("## Coverage\n\n");
            if (skipped == null || skipped.isEmpty()) {
                sb.append("- no skipped profiles\n");
            } else {
                for (String s : skipped) {
                    sb.append("- ").append(s).append('\n');
                }
            }

            Files.write(p, sb.toString().getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Failed to write league report: " + e.getMessage());
        }
    }

    private enum LeagueOpponentKind {
        BOT,
        SNAPSHOT
    }

    private static final class LeagueOpponentSpec {

        final LeagueOpponentKind kind;
        final int botSkill; // if BOT
        final String policyKey; // if SNAPSHOT (e.g. "snap:league_ep_5000.pt")

        private LeagueOpponentSpec(LeagueOpponentKind kind, int botSkill, String policyKey) {
            this.kind = kind;
            this.botSkill = botSkill;
            this.policyKey = policyKey;
        }

        static LeagueOpponentSpec bot(int skill) {
            return new LeagueOpponentSpec(LeagueOpponentKind.BOT, skill, null);
        }

        static LeagueOpponentSpec snapshot(String policyKey) {
            return new LeagueOpponentSpec(LeagueOpponentKind.SNAPSHOT, 0, policyKey);
        }
    }

    /**
     * Run a benchmark between the RL policy and an opponent across a deck matrix.
     *
     * @param agentDeckListFile deck list the RL agent pilots; null/empty = same pool as oppDeckListFile
     *                          (old symmetric round-robin, skipping mirror matchups)
     * @param oppDeckListFile   deck list the opponent uses
     * @param logContext        short label written to league events for per-matchup rows
     */
    private double runLeagueBenchmarkPolicy(
            String rlPolicyKey,
            LeagueOpponentSpec opponent,
            String agentDeckListFile,
            String oppDeckListFile,
            int gamesPerMatchup,
            String logContext
    ) {
        try {
            mage.cards.repository.CardScanner.scan();

            // Load deck pools
            final boolean samePool = (agentDeckListFile == null || agentDeckListFile.trim().isEmpty());
            List<Path> oppDecks = loadDeckPoolWithOverride(oppDeckListFile);
            List<Path> agentDecks = samePool ? oppDecks : loadDeckPoolWithOverride(agentDeckListFile);

            if (oppDecks.isEmpty()) {
                logger.warn("League benchmark: opponent deck pool is empty");
                return 0.0;
            }
            if (agentDecks.isEmpty()) {
                logger.warn("League benchmark: agent deck pool is empty");
                return 0.0;
            }
            if (samePool && oppDecks.size() < 2) {
                logger.warn("League benchmark requires at least 2 decks in shared pool; found " + oppDecks.size());
                return 0.0;
            }

            final int benchThreads = LEAGUE_BENCHMARK_THREADS;
            final int logEvery = Math.max(1, LEAGUE_BENCHMARK_LOG_EVERY);
            final int heartbeatSec = LEAGUE_BENCHMARK_HEARTBEAT_SEC;
            final int gameTimeoutSec = LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC;

            // Build the list of (agentDeck, oppDeck) pairs
            final List<Path[]> pairs = new ArrayList<>();
            for (Path a : agentDecks) {
                for (Path o : oppDecks) {
                    if (samePool && a.equals(o)) {
                        continue; // skip mirror matchups when using shared pool
                    }
                    pairs.add(new Path[]{a, o});
                }
            }
            if (pairs.isEmpty()) {
                logger.warn("League benchmark: no valid deck pairs");
                return 0.0;
            }

            final int totalPlannedGames = pairs.size() * Math.max(1, gamesPerMatchup);
            final AtomicLong completed = new AtomicLong(0);
            final AtomicLong started = new AtomicLong(0);
            final AtomicLong winsTotal = new AtomicLong(0);
            final ConcurrentHashMap<String, AtomicLong> matchupWins = new ConcurrentHashMap<>();
            final ConcurrentHashMap<String, AtomicLong> matchupGames = new ConcurrentHashMap<>();
            final long startMs = System.currentTimeMillis();

            for (Path[] pair : pairs) {
                String key = pair[0].getFileName() + " vs " + pair[1].getFileName();
                matchupWins.put(key, new AtomicLong(0));
                matchupGames.put(key, new AtomicLong(0));
            }

            ExecutorService exec = Executors.newFixedThreadPool(benchThreads, r -> {
                Thread t = new Thread(r);
                t.setName("GAME-BENCH");
                t.setPriority(Thread.NORM_PRIORITY);
                return t;
            });

            logger.info(String.format(
                    "League benchmark started: context=%s rlPolicy=%s vs %s, agentDecks=%d oppDecks=%d pairs=%d gpm=%d plannedGames=%d threads=%d",
                    logContext,
                    rlPolicyKey,
                    opponent.kind == LeagueOpponentKind.BOT ? ("CP7Skill" + opponent.botSkill) : opponent.policyKey,
                    agentDecks.size(),
                    oppDecks.size(),
                    pairs.size(),
                    gamesPerMatchup,
                    totalPlannedGames,
                    benchThreads
            ));

            java.util.concurrent.ScheduledExecutorService heartbeat = null;
            if (heartbeatSec > 0) {
                heartbeat = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
                    Thread t = new Thread(r, "LEAGUE-BENCH-HEARTBEAT");
                    t.setDaemon(true);
                    return t;
                });
                final java.util.concurrent.ScheduledExecutorService hbRef = heartbeat;
                heartbeat.scheduleAtFixedRate(() -> {
                    long done = completed.get();
                    if (done >= totalPlannedGames) {
                        hbRef.shutdown();
                        return;
                    }
                    long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                    double gamesPerSec = done / (elapsedMs / 1000.0);
                    logger.info(String.format(
                            "League benchmark heartbeat: %d/%d games done (started=%d; %.2f games/s)",
                            done, totalPlannedGames, started.get(), gamesPerSec
                    ));
                }, heartbeatSec, heartbeatSec, java.util.concurrent.TimeUnit.SECONDS);
            }

            List<Future<Void>> futures = new ArrayList<>();
            for (Path[] pair : pairs) {
                final Path p1 = pair[0];
                final Path p2 = pair[1];
                final String matchupKey = p1.getFileName() + " vs " + p2.getFileName();
                for (int g = 0; g < gamesPerMatchup; g++) {
                    final boolean logThisGame = (g == 0);
                    futures.add(exec.submit(() -> {
                        Thread.currentThread().setName("GAME-BENCH");
                        long s = started.incrementAndGet();
                        if (s % logEvery == 0 || s == totalPlannedGames) {
                            logger.info(String.format("League benchmark started games: %d/%d", s, totalPlannedGames));
                        }

                        boolean win = runSingleLeagueBenchmarkGame(
                                p1, p2, gameTimeoutSec,
                                rlPolicyKey, opponent, logThisGame
                        );
                        matchupGames.get(matchupKey).incrementAndGet();
                        if (win) {
                            winsTotal.incrementAndGet();
                            matchupWins.get(matchupKey).incrementAndGet();
                        }

                        long done = completed.incrementAndGet();
                        if (done % logEvery == 0 || done == totalPlannedGames) {
                            long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
                            double gamesPerSec = done / (elapsedMs / 1000.0);
                            logger.info(String.format(
                                    "League benchmark progress: %d/%d games done (%.2f games/s)",
                                    done, totalPlannedGames, gamesPerSec
                            ));
                        }
                        return null;
                    }));
                }
            }

            exec.shutdown();
            exec.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    logger.warn("League benchmark game task failed", e);
                }
            }
            if (heartbeat != null) {
                heartbeat.shutdownNow();
            }

            long totalGames = completed.get();
            long totalWins = winsTotal.get();
            double overall = totalGames > 0 ? (double) totalWins / totalGames : 0.0;
            logger.info(String.format(
                    "League benchmark done: context=%s winrate=%.3f (%d/%d)",
                    logContext, overall, totalWins, totalGames
            ));

            // Log per-matchup winrates to league events
            java.util.List<String> sortedMatchups = new java.util.ArrayList<>(matchupGames.keySet());
            java.util.Collections.sort(sortedMatchups);
            for (String mk : sortedMatchups) {
                long mg = matchupGames.get(mk).get();
                long mw = matchupWins.get(mk).get();
                double mwr = mg > 0 ? (double) mw / mg : 0.0;
                appendLeagueEvent(String.format("matchup_result context=%s matchup=[%s] wr=%.4f wins=%d games=%d",
                        logContext, mk, mwr, mw, mg));
            }

            return overall;
        } catch (Exception e) {
            logger.error("League benchmark failed", e);
            return 0.0;
        }
    }

    /**
     * League bench: play N games against CP7 on a specified deck, write results to file.
     * Used by the orchestrator's eval benchmark to get absolute performance numbers.
     */
    public void runLeagueBench() {
        if (EVAL_OPPONENT_DECK.isEmpty()) {
            logger.error("league_bench requires EVAL_OPPONENT_DECK to be set");
            return;
        }
        try {
            mage.cards.repository.CardScanner.scan();
        } catch (Exception e) {
            logger.error("Card scan failed", e);
            return;
        }

        List<Path> agentDecks;
        try {
            if (RL_AGENT_DECK_LIST_FILE != null && !RL_AGENT_DECK_LIST_FILE.trim().isEmpty()) {
                agentDecks = loadDeckPoolWithOverride(RL_AGENT_DECK_LIST_FILE);
            } else {
                agentDecks = loadDeckPool();
            }
        } catch (Exception e) {
            logger.error("Failed to load agent decks", e);
            return;
        }

        Path oppDeckPath = Paths.get(EVAL_OPPONENT_DECK);
        Deck oppDeck = loadDeck(oppDeckPath.toString());
        if (oppDeck == null) {
            logger.error("Failed to load opponent deck: " + EVAL_OPPONENT_DECK);
            return;
        }

        logger.info(String.format("league_bench: %d games vs CP7-Skill%d deck=%s profile=%s",
                EVAL_NUM_GAMES, EVAL_OPPONENT_SKILL, EVAL_OPPONENT_DECK, MODEL_PROFILE_NAME));

        ComputerPlayerRL.resetRLActivationFailureCount();
        int wins = 0;
        int total = 0;
        Random rand = newSeededRandom(7777);

        for (int i = 0; i < EVAL_NUM_GAMES; i++) {
            try {
                long replaySeed = evalReplaySeed(i);
                Path agentDeckPath = agentDecks.get(rand.nextInt(agentDecks.size()));
                Deck agentDeck = loadDeck(agentDeckPath.toString());
                Deck oppDeckCopy = loadDeckFresh(oppDeckPath.toString());
                if (agentDeck == null || oppDeckCopy == null) continue;
                if (EVAL_REPLAY_METADATA) {
                    RandomUtil.setSeed(replayRandomUtilSeed(replaySeed));
                    setReplayTraceContext(i + 1, replaySeed, "league_bench");
                    agentDeck = replayShuffledCopy(agentDeck, replaySeed ^ 0x5DEECE66DL);
                    oppDeckCopy = replayShuffledCopy(oppDeckCopy, replaySeed ^ 0xC0FFEE1234L);
                }

                // Set up game logger -- GAME_LOG_DIR env var directs output.
                // Default off for benchmark sweeps; enable with EVAL_GAME_LOGGING=1.
                GameLogger gameLogger = GameLogger.create(EnvConfig.bool("EVAL_GAME_LOGGING", false));
                threadLocalGameLogger.set(gameLogger);
                if (gameLogger.isEnabled()) {
                    gameLogger.log("MODE=league_bench_eval");
                    gameLogger.log("MATCHUP: agent=" + agentDeckPath.getFileName()
                            + " vs opp=" + oppDeckPath.getFileName());
                    if (EVAL_REPLAY_METADATA) {
                        gameLogger.log("REPLAY: scenario=" + (i + 1)
                                + " seed=" + replaySeed
                                + " agent_deck=" + agentDeckPath.getFileName()
                                + " opp_deck=" + oppDeckPath.getFileName()
                                + " action_counterfactual_compatible=true");
                        gameLogger.log("REPLAY_RANDOM: scenario=" + (i + 1)
                                + " seed=" + replaySeed
                                + " random_util_seed=" + replayRandomUtilSeed(replaySeed)
                                + " scope=league_bench");
                    }
                }

                TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("BenchMatch", "BenchMatch", false));
                match.startGame();
                Game game = match.getGames().get(0);

                // Evaluation must disable training so ISMCTS_ENABLE can trigger
                // eval-time MCTS override instead of the training distillation path.
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel,
                        true, false, "train");
                rlPlayer.setCurrentEpisode(-(i + 1));
                rlPlayer.setAttachedGameLogger(gameLogger);
                game.addPlayer(rlPlayer, agentDeck);
                match.addPlayer(rlPlayer, agentDeck);

                ComputerPlayer7 opponent = ReplayOpponentDecisionPlayer.create(
                        "EvalBot-Skill" + EVAL_OPPONENT_SKILL, RangeOfInfluence.ALL, EVAL_OPPONENT_SKILL,
                        i + 1, replaySeed, gameLogger);
                game.addPlayer(opponent, oppDeckCopy);
                match.addPlayer(opponent, oppDeckCopy);

                game.loadCards(agentDeck.getCards(), rlPlayer.getId());
                game.loadCards(oppDeckCopy.getCards(), opponent.getId());
                if (EVAL_REPLAY_METADATA) {
                    forceReplayLibraryOrder(rlPlayer, agentDeck, game);
                    forceReplayLibraryOrder(opponent, oppDeckCopy, game);
                }
                GameOptions options = new GameOptions();
                if (EVAL_REPLAY_METADATA) {
                    options.skipInitShuffling = true;
                }
                game.setGameOptions(options);

                GameHealthMonitor healthMonitor = GameHealthMonitor.createAndStart(game);
                int evalJoinTimeoutSec = Math.max(60,
                        EnvConfig.i32("EVAL_GAME_THREAD_TIMEOUT_SEC", EnvConfig.i32("GAME_TIMEOUT_SEC", 300))) + 30;
                startGameInGameThread(game, rlPlayer.getId(), evalJoinTimeoutSec);
                healthMonitor.stop();

                boolean won;
                if (healthMonitor.wasKilled()) {
                    won = false;
                } else {
                    won = game.getWinner().contains(rlPlayer.getName());
                }
                if (gameLogger.isEnabled()) {
                    gameLogger.log("RESULT: " + (won ? "WIN" : "LOSS"));
                }
                if (won) wins++;
                total++;
            } catch (Exception e) {
                logger.error("Eval game " + i + " failed: " + e.getMessage());
            }
        }

        double winrate = total > 0 ? (double) wins / total : 0.0;
        int mctsActivations = ComputerPlayerRL.getMctsActivationCount();
        String result = String.format("EVAL_RESULT: wins=%d total=%d winrate=%.4f profile=%s mcts_activations=%d",
                wins, total, winrate, MODEL_PROFILE_NAME, mctsActivations);
        logger.info(result);
        System.out.println(result);
        System.out.println("MCTS_GATE: " + ComputerPlayerRL.getMctsGateStats());
        System.out.println(PolicyValueMCTS.getMctsTimingStats());

        if (!EVAL_RESULTS_FILE.isEmpty()) {
            try (java.io.FileWriter fw = new java.io.FileWriter(EVAL_RESULTS_FILE)) {
                fw.write(String.format("%d,%d,%.4f,%s,%d\n", wins, total, winrate, MODEL_PROFILE_NAME, mctsActivations));
            } catch (Exception e) {
                logger.error("Failed to write eval results: " + e.getMessage());
            }
        }
    }

    private boolean runSingleLeagueBenchmarkGame(
            Path rlDeckPath,
            Path oppDeckPath,
            int gameTimeoutSec,
            String rlPolicyKey,
            LeagueOpponentSpec opponent,
            boolean logThisGame
    ) {
        Deck d1 = loadDeck(rlDeckPath.toString());
        Deck d2 = loadDeck(oppDeckPath.toString());
        if (d1 == null || d2 == null) {
            return false;
        }

        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));
        try {
            match.startGame();
        } catch (GameException e) {
            logger.error("Error starting league benchmark game", e);
            return false;
        }
        Game game = match.getGames().get(0);

        GameLogger gameLogger = GameLogger.createForEval(logThisGame);
        threadLocalGameLogger.set(gameLogger);
        if (gameLogger.isEnabled()) {
            gameLogger.log("MODE=league_benchmark");
            gameLogger.log("MATCHUP: rlDeck=" + rlDeckPath.getFileName() + " vs oppDeck=" + oppDeckPath.getFileName());
            gameLogger.log("RL_POLICY_KEY=" + rlPolicyKey);
            gameLogger.log("OPPONENT_KIND=" + opponent.kind);
            if (opponent.kind == LeagueOpponentKind.BOT) {
                gameLogger.log("OPPONENT_BOT_SKILL=" + opponent.botSkill);
            } else {
                gameLogger.log("OPPONENT_POLICY_KEY=" + opponent.policyKey);
            }
        }

        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("RL", RangeOfInfluence.ALL, sharedModel, true, false, rlPolicyKey);
        rlPlayer.setCurrentEpisode(-1);
        rlPlayer.setAttachedGameLogger(gameLogger);

        Player opp;
        if (opponent.kind == LeagueOpponentKind.BOT) {
            int skill = Math.max(1, opponent.botSkill);
            opp = new ComputerPlayer7("Benchmark-skill" + skill, RangeOfInfluence.ALL, skill);
        } else {
            String pk = opponent.policyKey == null ? "train" : opponent.policyKey;
            opp = new ComputerPlayerRL("SnapshotOpp", RangeOfInfluence.ALL, sharedModel, true, false, pk);
            ((ComputerPlayerRL) opp).setAttachedGameLogger(gameLogger);
        }

        Deck rlDeck = d1.copy();
        Deck oppDeck = d2.copy();
        game.addPlayer(rlPlayer, rlDeck);
        match.addPlayer(rlPlayer, rlDeck);
        game.addPlayer(opp, oppDeck);
        match.addPlayer(opp, oppDeck);

        game.loadCards(rlDeck.getCards(), rlPlayer.getId());
        game.loadCards(oppDeck.getCards(), opp.getId());
        game.setGameOptions(new GameOptions());

        final Thread watchdog = new Thread(() -> {
            try {
                Thread.sleep(Math.max(1, gameTimeoutSec) * 1000L);
                if (game.getState() != null && !game.getState().isGameOver()) {
                    logger.warn("League benchmark game timed out after " + gameTimeoutSec + "s; forcing end: "
                            + rlDeckPath.getFileName() + " vs " + oppDeckPath.getFileName());
                    try {
                        game.end();
                    } catch (Exception e) {
                        logger.warn("Failed to force-end timed out league benchmark game", e);
                    }
                }
            } catch (InterruptedException ignored) {
            } catch (Exception e) {
                logger.warn("League benchmark watchdog error", e);
            }
        }, "LEAGUE-BENCH-WATCHDOG");
        watchdog.setDaemon(true);
        watchdog.start();

        startGameInGameThread(game, rlPlayer.getId(), Math.max(1, gameTimeoutSec) + 30);
        watchdog.interrupt();

        boolean win = false;
        try {
            win = game.getWinner().contains(rlPlayer.getName());
        } catch (Exception ignored) {
            win = false;
        }

        if (gameLogger.isEnabled()) {
            try {
                String winner = win ? rlPlayer.getName() : opp.getName();
                String loser = win ? opp.getName() : rlPlayer.getName();
                int turns = game.getTurnNum();
                String reason = "LeagueBenchmark: " + rlDeckPath.getFileName() + " vs " + oppDeckPath.getFileName();
                gameLogger.logOutcome(winner, loser, turns, reason);
            } finally {
                gameLogger.close();
            }
        }
        return win;
    }

    private static void writeBenchmarkLiveReport(
            Path path,
            long done,
            long planned,
            long startMs,
            long started,
            long winsTotal,
            ConcurrentHashMap<String, AtomicLong> matchupWins,
            ConcurrentHashMap<String, AtomicLong> matchupGames,
            ConcurrentHashMap<String, AtomicLong> deckWinsAsRL,
            ConcurrentHashMap<String, AtomicLong> deckGamesAsRL,
            ConcurrentHashMap<String, AtomicLong> deckWinsVsOpp,
            ConcurrentHashMap<String, AtomicLong> deckGamesVsOpp
    ) {
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }

            long elapsedMs = Math.max(1, System.currentTimeMillis() - startMs);
            double gamesPerSec = done / (elapsedMs / 1000.0);
            long remaining = Math.max(0, planned - done);
            long etaSec = gamesPerSec > 0 ? (long) (remaining / gamesPerSec) : -1;
            double overall = done > 0 ? (double) winsTotal / done : 0.0;

            StringBuilder sb = new StringBuilder();
            sb.append("BENCHMARK LIVE REPORT\n");
            sb.append("updated_ms=").append(System.currentTimeMillis()).append('\n');
            sb.append("progress=").append(done).append('/').append(planned)
                    .append(" started=").append(started)
                    .append(" done=").append(done)
                    .append(" wins=").append(winsTotal)
                    .append(" winrate=").append(String.format("%.4f", overall))
                    .append(" games_per_s=").append(String.format("%.3f", gamesPerSec))
                    .append(" eta_s=").append(etaSec)
                    .append('\n');
            sb.append('\n');

            // WITH each deck (RL pilots)
            java.util.List<String> deckKeys = new java.util.ArrayList<>(deckGamesAsRL.keySet());
            java.util.Collections.sort(deckKeys);
            sb.append("WITH_DECK\n");
            for (String dk : deckKeys) {
                long g = deckGamesAsRL.get(dk) == null ? 0L : deckGamesAsRL.get(dk).get();
                long w = deckWinsAsRL.get(dk) == null ? 0L : deckWinsAsRL.get(dk).get();
                double wr = g > 0 ? (double) w / g : 0.0;
                sb.append(dk).append(",winrate=").append(String.format("%.4f", wr))
                        .append(",wins=").append(w).append(",games=").append(g).append('\n');
            }
            sb.append('\n');

            // VS each deck (opponent deck)
            java.util.List<String> oppKeys = new java.util.ArrayList<>(deckGamesVsOpp.keySet());
            java.util.Collections.sort(oppKeys);
            sb.append("VS_DECK\n");
            for (String dk : oppKeys) {
                long g = deckGamesVsOpp.get(dk) == null ? 0L : deckGamesVsOpp.get(dk).get();
                long w = deckWinsVsOpp.get(dk) == null ? 0L : deckWinsVsOpp.get(dk).get();
                double wr = g > 0 ? (double) w / g : 0.0;
                sb.append(dk).append(",winrate=").append(String.format("%.4f", wr))
                        .append(",wins=").append(w).append(",games=").append(g).append('\n');
            }
            sb.append('\n');

            // Per matchup
            java.util.List<String> matchupKeys = new java.util.ArrayList<>(matchupGames.keySet());
            java.util.Collections.sort(matchupKeys);
            sb.append("MATCHUP\n");
            for (String mk : matchupKeys) {
                long g = matchupGames.get(mk) == null ? 0L : matchupGames.get(mk).get();
                long w = matchupWins.get(mk) == null ? 0L : matchupWins.get(mk).get();
                double wr = g > 0 ? (double) w / g : 0.0;
                sb.append(mk).append(",winrate=").append(String.format("%.4f", wr))
                        .append(",wins=").append(w).append(",games=").append(g).append('\n');
            }

            Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception ignored) {
            // Don't fail benchmark for reporting.
        }
    }

    private boolean runSingleBenchmarkGame(Path rlDeckPath, Path oppDeckPath, int gameTimeoutSec) {
        Deck d1 = loadDeck(rlDeckPath.toString());
        Deck d2 = loadDeck(oppDeckPath.toString());
        if (d1 == null || d2 == null) {
            return false;
        }

        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));
        try {
            match.startGame();
        } catch (GameException e) {
            logger.error("Error starting benchmark game", e);
            return false;
        }
        Game game = match.getGames().get(0);

        // Benchmark gamelogs: allow full per-game logging like training/eval.
        boolean enableGameLogging = "1".equals(System.getenv().getOrDefault("GAME_LOGGING", "0"))
                || "true".equalsIgnoreCase(System.getenv().getOrDefault("GAME_LOGGING", "0"));
        GameLogger gameLogger = GameLogger.create(enableGameLogging);
        threadLocalGameLogger.set(gameLogger);
        if (gameLogger.isEnabled()) {
            gameLogger.log("MODE=benchmark");
            gameLogger.log("MATCHUP: rlDeck=" + rlDeckPath.getFileName() + " vs oppDeck=" + oppDeckPath.getFileName());
        }

        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("RL", RangeOfInfluence.ALL, sharedModel, true);
        rlPlayer.setCurrentEpisode(-1); // -1 indicates benchmark game
        rlPlayer.setAttachedGameLogger(gameLogger);
        // Use strong opponent for benchmarking
        int benchSkill = EnvConfig.i32("BENCHMARK_OPPONENT_SKILL", 6);
        Player opponent = new ComputerPlayer7("Benchmark-skill" + benchSkill, RangeOfInfluence.ALL, benchSkill);

        Deck rlDeck = d1.copy();
        Deck oppDeck = d2.copy();

        game.addPlayer(rlPlayer, rlDeck);
        match.addPlayer(rlPlayer, rlDeck);
        game.addPlayer(opponent, oppDeck);
        match.addPlayer(opponent, oppDeck);

        game.loadCards(rlDeck.getCards(), rlPlayer.getId());
        game.loadCards(oppDeck.getCards(), opponent.getId());
        game.setGameOptions(new GameOptions());
        // Watchdog: end runaway/stuck games so benchmark doesn't stall forever.
        final Thread watchdog = new Thread(() -> {
            try {
                Thread.sleep(Math.max(1, gameTimeoutSec) * 1000L);
                if (game.getState() != null && !game.getState().isGameOver()) {
                    logger.warn("Benchmark game timed out after " + gameTimeoutSec + "s; forcing end: "
                            + rlDeckPath.getFileName() + " vs " + oppDeckPath.getFileName());
                    try {
                        game.end();
                    } catch (Exception e) {
                        logger.warn("Failed to force-end timed out game", e);
                    }
                }
            } catch (InterruptedException ignored) {
                // normal
            } catch (Exception e) {
                logger.warn("Benchmark watchdog error", e);
            }
        }, "BENCH-WATCHDOG");
        watchdog.setDaemon(true);
        watchdog.start();

        startGameInGameThread(game, rlPlayer.getId(), Math.max(1, gameTimeoutSec) + 30);
        watchdog.interrupt();

        boolean win = game.getWinner().contains(rlPlayer.getName());
        if (gameLogger.isEnabled()) {
            try {
                String winner = win ? rlPlayer.getName() : opponent.getName();
                String loser = win ? opponent.getName() : rlPlayer.getName();
                int turns = game.getTurnNum();
                String reason = "Benchmark: " + rlDeckPath.getFileName() + " vs " + oppDeckPath.getFileName();
                gameLogger.logOutcome(winner, loser, turns, reason);
            } finally {
                gameLogger.close();
            }
        }
        return win;
    }

    private static int readMaxEpisodeFromCsv(String csvPath) {
        java.io.File f = new java.io.File(csvPath);
        if (!f.exists()) return 0;
        int maxEp = 0;
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String line = reader.readLine(); // skip header
            while ((line = reader.readLine()) != null) {
                int comma = line.indexOf(',');
                if (comma > 0) {
                    try {
                        int ep = Integer.parseInt(line.substring(0, comma));
                        if (ep > maxEp) maxEp = ep;
                    } catch (NumberFormatException ignored) {}
                }
            }
        } catch (Exception e) {
            logger.warn("Failed to read episode count from " + csvPath + ": " + e.getMessage());
        }
        return maxEp;
    }

    private static void startGameInGameThread(Game game, UUID startingPlayerId, int joinTimeoutSec) {
        // In some modes (e.g., league_eval) the caller runs on RLTrainer.main, which fails strict game-thread checks.
        if (ThreadUtils.isRunGameThread()) {
            game.start(startingPlayerId);
            return;
        }

        final AtomicReference<Throwable> error = new AtomicReference<>(null);
        final GameLogger callerLogger = threadLocalGameLogger.get();
        Thread gameThread = new Thread(() -> {
            try {
                if (callerLogger != null) threadLocalGameLogger.set(callerLogger);
                game.start(startingPlayerId);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-LEAGUE-EVAL");
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for game thread", ie);
        }

        if (gameThread.isAlive()) {
            try {
                game.end();
            } catch (Exception ignored) {
            }
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            if (gameThread.isAlive()) {
                throw new IllegalStateException("Game thread did not finish after timeout and forced end.");
            }
        }

        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        }
        throw new IllegalStateException("Error while running game on GAME thread", t);
    }

    public static List<Path> loadDeckPool() throws IOException {
        // If explicit list is provided, use it
        if (DECK_LIST_FILE != null && !DECK_LIST_FILE.trim().isEmpty()) {
            Path listPath = Paths.get(DECK_LIST_FILE);
            if (Files.exists(listPath) && Files.isRegularFile(listPath)) {
                String fileName = listPath.getFileName().toString().toLowerCase();
                if (fileName.endsWith(".dek") || fileName.endsWith(".dck")) {
                    return Collections.singletonList(listPath.toAbsolutePath().normalize());
                }
            }
            Path base = listPath.toAbsolutePath().getParent();
            if (base == null) {
                base = Paths.get(System.getProperty("user.dir"));
            }
            List<String> lines = Files.readAllLines(listPath, StandardCharsets.UTF_8);
            List<Path> decks = new ArrayList<>();
            for (String raw : lines) {
                String line = raw.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                Path p = Paths.get(line);
                if (!p.isAbsolute()) {
                    p = base.resolve(p).normalize();
                }
                if (Files.exists(p) && Files.isRegularFile(p)) {
                    decks.add(p);
                } else {
                    logger.warn("Deck list entry not found: " + p);
                }
            }
            return decks;
        }

        // Fallback: scan directory
        return Files.list(Paths.get(DECKS_DIRECTORY))
                .filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.getFileName().toString().toLowerCase();
                    return name.endsWith(".dek") || name.endsWith(".dck");
                })
                .collect(Collectors.toList());
    }

    private static List<Path> loadDeckPoolWithOverride(String deckListFileOverride) throws IOException {
        String dl = deckListFileOverride == null ? "" : deckListFileOverride.trim();
        if (!dl.isEmpty()) {
            Path listPath = Paths.get(dl);
            if (Files.exists(listPath) && Files.isRegularFile(listPath)) {
                String fileName = listPath.getFileName().toString().toLowerCase();
                if (fileName.endsWith(".dek") || fileName.endsWith(".dck")) {
                    return Collections.singletonList(listPath.toAbsolutePath().normalize());
                }
            }
            Path base = listPath.toAbsolutePath().getParent();
            if (base == null) {
                base = Paths.get(System.getProperty("user.dir"));
            }
            List<String> lines = Files.readAllLines(listPath, StandardCharsets.UTF_8);
            List<Path> decks = new ArrayList<>();
            for (String raw : lines) {
                String line = raw.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                Path p = Paths.get(line);
                if (!p.isAbsolute()) {
                    p = base.resolve(p).normalize();
                }
                if (Files.exists(p) && Files.isRegularFile(p)) {
                    decks.add(p);
                } else {
                    logger.warn("Deck list entry not found: " + p);
                }
            }
            return decks;
        }
        return loadDeckPool();
    }

    private static void logEvaluationResult(int updateStep, double winRate) {
        try {
            Path statsPath = Paths.get(RLLogPaths.EVALUATION_STATS_PATH);
            if (statsPath.getParent() != null) {
                Files.createDirectories(statsPath.getParent());
            }
            boolean writeHeader = !Files.exists(statsPath);
            StringBuilder sb = new StringBuilder();
            if (writeHeader) {
                sb.append("update_step,win_rate\n");
            }
            sb.append(updateStep).append(',').append(String.format("%.4f", winRate)).append('\n');
            Files.write(statsPath, sb.toString().getBytes(StandardCharsets.UTF_8), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write evaluation stats CSV", e);
        }
    }

    private void logGameResult(Game game, ComputerPlayerRL rlPlayer) {
        logStaticGameResult(game, rlPlayer);
    }

    /**
     * Log head usage statistics to CSV for tracking how each decision head is being trained.
     * Groups ActionTypes by actual neural network head.
     */
    private void logHeadUsageStats(int episodeNum, ComputerPlayerRL rlPlayer, Player opponentPlayer,
                                   int turns, boolean rlPlayerWon, Path rlDeckPath, Path oppDeckPath) {
        try {
            Path logPath = Paths.get(profileContext != null
                    ? profileContext.paths.headUsageLogPath
                    : RLLogPaths.HEAD_USAGE_LOG_PATH);

            // Get decision counts for both players
            java.util.Map<StateSequenceBuilder.ActionType, Integer> rlCounts = rlPlayer.getDecisionCountsByHead();
            java.util.Map<StateSequenceBuilder.ActionType, Integer> oppCounts = new java.util.HashMap<>();
            if (opponentPlayer instanceof ComputerPlayerRL) {
                oppCounts = ((ComputerPlayerRL) opponentPlayer).getDecisionCountsByHead();
            }

            // Group by actual head (action, target, card_select)
            int rlActionHead = 0;
            int rlTargetHead = 0;
            int rlCardSelectHead = 0;
            for (java.util.Map.Entry<StateSequenceBuilder.ActionType, Integer> entry : rlCounts.entrySet()) {
                StateSequenceBuilder.ActionType type = entry.getKey();
                int count = entry.getValue();
                if (type == StateSequenceBuilder.ActionType.SELECT_TARGETS) {
                    rlTargetHead += count;
                } else if (type == StateSequenceBuilder.ActionType.SELECT_CARD) {
                    rlCardSelectHead += count;
                } else {
                    rlActionHead += count;
                }
            }

            int oppActionHead = 0;
            int oppTargetHead = 0;
            int oppCardSelectHead = 0;
            for (java.util.Map.Entry<StateSequenceBuilder.ActionType, Integer> entry : oppCounts.entrySet()) {
                StateSequenceBuilder.ActionType type = entry.getKey();
                int count = entry.getValue();
                if (type == StateSequenceBuilder.ActionType.SELECT_TARGETS) {
                    oppTargetHead += count;
                } else if (type == StateSequenceBuilder.ActionType.SELECT_CARD) {
                    oppCardSelectHead += count;
                } else {
                    oppActionHead += count;
                }
            }

            int rlTotal = rlActionHead + rlTargetHead + rlCardSelectHead;
            int oppTotal = oppActionHead + oppTargetHead + oppCardSelectHead;

            // Get opponent type
            String opponentType = "unknown";
            if (opponentPlayer instanceof ComputerPlayerRL) {
                opponentType = "RL";
            } else if (opponentPlayer.getClass().getSimpleName().startsWith("ComputerPlayer")) {
                opponentType = opponentPlayer.getClass().getSimpleName();
            }

            String rlDeck = rlDeckPath != null
                    ? rlDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "")
                    : "unknown";
            String oppDeck = oppDeckPath != null
                    ? oppDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "")
                    : "unknown";

            // Build CSV line
            String line = String.format("%d,%s,%s,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    episodeNum, rlDeck, oppDeck, turns, rlPlayerWon ? 1 : 0, opponentType,
                    rlTotal, rlActionHead, rlTargetHead, rlCardSelectHead,
                    oppTotal, oppActionHead, oppTargetHead, oppCardSelectHead);
            String header = "episode,rl_deck,opp_deck,turns,won,opponent_type," +
                    "rl_total,rl_action_head,rl_target_head,rl_card_select_head," +
                    "opp_total,opp_action_head,opp_target_head,opp_card_select_head\n";
            ASYNC_LINE_WRITER.append(logPath, header, line);

        } catch (Exception e) {
            logger.warn("Failed to write head usage stats: " + e.getMessage());
        }
    }

    private static void logStaticGameResult(Game game, ComputerPlayerRL rlPlayer) {
        if (game.getWinner().contains(rlPlayer.getName())) {
            logger.info("Game finished. Winner: " + rlPlayer.getName());
        } else {
            logger.info("Game finished. Loser: " + rlPlayer.getName());
        }
    }

    public Deck loadDeck(String filePath) {
        return DECK_TEMPLATE_CACHE.load(filePath, logger);
    }

    /**
     * Load a deck without using the cache -- guarantees fully independent card instances.
     * Use for the opponent deck in self-play to avoid shared card references.
     */
    public Deck loadDeckFresh(String filePath) {
        try {
            StringBuilder warnings = new StringBuilder();
            mage.cards.decks.DeckCardLists lists = mage.cards.decks.importer.DeckImporter.importDeckFromFile(filePath, warnings, false);
            if (lists == null || lists.getCards().isEmpty()) {
                String detail = warnings.length() > 0 ? " warnings=" + warnings : "";
                logger.error("Error loading fresh deck: empty maindeck for " + filePath + detail);
                return null;
            }
            Deck deck = Deck.load(lists, false, false, null);
            if (deck == null || deck.getCards().isEmpty()) {
                logger.error("Error loading fresh deck: no playable cards for " + filePath);
                return null;
            }
            return deck;
        } catch (Exception e) {
            logger.error("Error loading fresh deck: " + filePath, e);
            return null;
        }
    }

    private RewardDiag updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, Player opponentPlayer) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());
        ProfileContext mainTrainingContext = profileContext;

        // ------------------------------------------------------------------
        // 1.  Terminal win / loss reward (ground-truth)
        // Scaling up via TERMINAL_REWARD_SCALE pushes value predictions harder
        // toward the ±1 clip bounds (larger MSE target = stronger gradient),
        // which increases the win/loss magnitude separation that MCTS needs.
        // ------------------------------------------------------------------
        double rewardScale = Double.parseDouble(System.getenv().getOrDefault("TERMINAL_REWARD_SCALE", "1.0"));
        double finalReward = rlPlayerWon ? rewardScale : -rewardScale;

        if (Double.isNaN(finalReward) || Double.isInfinite(finalReward)) {
            finalReward = rlPlayerWon ? rewardScale : -rewardScale;
            logger.warn("Reward was NaN/Inf – reverted to ±" + rewardScale + " fallback");
        }

        // Get training data for RL player
        List<StateSequenceBuilder.TrainingData> rlPlayerTrainingData = rlPlayer.getTrainingBuffer();

        List<StateSequenceBuilder.TrainingData> opponentTrainingData = new ArrayList<>();
        ProfileContext opponentTrainingContext = mainTrainingContext;
        boolean trainOpponent = false;
        if (opponentPlayer instanceof ComputerPlayerRL) {
            ComputerPlayerRL rlOpponent = (ComputerPlayerRL) opponentPlayer;
            trainOpponent = rlOpponent.isTrainingEnabled() && !isSnapshotPolicyKey(rlOpponent.getPolicyKey());
            if (trainOpponent) {
                opponentTrainingContext = trainingContextForPolicyKey(rlOpponent.getPolicyKey(), mainTrainingContext);
                opponentTrainingData = rlOpponent.getTrainingBuffer();
            }
        }

        int rlRawSteps = rlPlayerTrainingData.size();
        int opponentRawSteps = opponentTrainingData.size();
        rlPlayerTrainingData = capTrajectorySuffix(rlPlayerTrainingData, TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER);
        opponentTrainingData = capTrajectorySuffix(opponentTrainingData, TRAIN_MAX_TRAJECTORY_STEPS_PER_PLAYER);

        // --- Calculate immediate rewards for each step (GAE will compute advantages in Python) ---
        List<Double> rlPlayerRewards = calculateImmediateRewards(rlPlayerTrainingData, finalReward);
        List<Double> opponentRewards = calculateImmediateRewards(opponentTrainingData, -finalReward); // Opposite reward for opponent
        if (AWR_SELECTED_ACTION_TARGETS_ENABLE) {
            attachAwrSelectedActionTargets(rlPlayerTrainingData, rlPlayerRewards);
            attachAwrSelectedActionTargets(opponentTrainingData, opponentRewards);
        }

        // Update the model with all states and immediate rewards (Python will apply GAE)
        if (!rlPlayerTrainingData.isEmpty()) {
            ACTOR_LEARNER.enqueueTraining(mainTrainingContext, rlPlayerTrainingData, rlPlayerRewards, "main");
        }
        if (trainOpponent && !opponentTrainingData.isEmpty()) {
            ACTOR_LEARNER.enqueueTraining(opponentTrainingContext, opponentTrainingData, opponentRewards, "opponent");
        }

        // Record value head prediction for auto-GAE tracking
        float lastValue = rlPlayer.getLastValueScore();
        metrics.recordValuePrediction(lastValue, rlPlayerWon);
        if (trainOpponent && opponentPlayer instanceof ComputerPlayerRL) {
            float opponentLastValue = ((ComputerPlayerRL) opponentPlayer).getLastValueScore();
            metrics.recordValuePrediction(opponentLastValue, !rlPlayerWon);
        }

        RewardDiag diag = new RewardDiag();
        diag.won = rlPlayerWon;
        diag.rawSteps = rlRawSteps;
        diag.opponentRawSteps = opponentRawSteps;
        diag.opponentSteps = opponentRewards.size();
        diag.finalReward = finalReward;
        diag.steps = rlPlayerRewards.size();
        diag.sumRewards = rlPlayerRewards.stream().mapToDouble(d -> d).sum();
        diag.lastReward = rlPlayerRewards.isEmpty() ? 0.0 : rlPlayerRewards.get(rlPlayerRewards.size() - 1);
        diag.mcReturn0 = computeDiscountedReturn0(rlPlayerRewards, 0.99);
        return diag;
    }

    private static List<StateSequenceBuilder.TrainingData> capTrajectorySuffix(
            List<StateSequenceBuilder.TrainingData> trajectory,
            int maxSteps
    ) {
        if (trajectory == null || trajectory.isEmpty() || maxSteps <= 0 || trajectory.size() <= maxSteps) {
            return trajectory == null ? new ArrayList<>() : trajectory;
        }
        return new ArrayList<>(trajectory.subList(trajectory.size() - maxSteps, trajectory.size()));
    }

    private static void attachAwrSelectedActionTargets(
            List<StateSequenceBuilder.TrainingData> trajectory,
            List<Double> rewards
    ) {
        if (trajectory == null || trajectory.isEmpty() || rewards == null || rewards.size() != trajectory.size()) {
            return;
        }
        double runningReturn = 0.0;
        for (int i = trajectory.size() - 1; i >= 0; i--) {
            Double reward = rewards.get(i);
            runningReturn = (reward == null ? 0.0 : reward) + AWR_GAMMA * runningReturn;
            StateSequenceBuilder.TrainingData td = trajectory.get(i);
            if (td == null || td.chosenCount <= 0 || td.candidateCount <= 0) {
                continue;
            }
            double advantage = runningReturn - td.oldValue;
            if (AWR_POSITIVE_ADVANTAGE_ONLY && advantage <= 0.0) {
                continue;
            }
            double weight = Math.exp(advantage / AWR_TEMPERATURE);
            if (!Double.isFinite(weight)) {
                weight = advantage >= 0.0 ? AWR_MAX_WEIGHT : AWR_MIN_WEIGHT;
            }
            weight = Math.max(AWR_MIN_WEIGHT, Math.min(AWR_MAX_WEIGHT, weight));

            float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
            int count = 0;
            int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            for (int j = 0; j < Math.min(td.chosenCount, td.chosenIndices.length); j++) {
                int idx = td.chosenIndices[j];
                if (idx >= 0 && idx < max && td.candidateMask[idx] == 1) {
                    count++;
                }
            }
            if (count <= 0) {
                continue;
            }
            float perChoice = (float) (weight / count);
            for (int j = 0; j < Math.min(td.chosenCount, td.chosenIndices.length); j++) {
                int idx = td.chosenIndices[j];
                if (idx >= 0 && idx < max && td.candidateMask[idx] == 1) {
                    target[idx] += perChoice;
                }
            }
            td.setMctsVisitTargets(target);
        }
    }

    private static boolean isSnapshotPolicyKey(String policyKey) {
        return policyKey != null && policyKey.trim().startsWith("snap:");
    }

    private static ProfileContext trainingContextForPolicyKey(String policyKey, ProfileContext fallback) {
        String key = policyKey == null ? "" : policyKey.trim();
        if (key.isEmpty() || "train".equals(key)) {
            return fallback;
        }
        if (key.startsWith("profile:")) {
            String name = key.substring("profile:".length()).trim();
            ProfileContext ctx = ProfileContext.byName(name);
            return ctx != null ? ctx : fallback;
        }
        ProfileContext ctx = ProfileContext.byName(key);
        return ctx != null ? ctx : fallback;
    }

    private static final class RewardDiag {

        boolean won;
        int rawSteps;
        int steps;
        int opponentRawSteps;
        int opponentSteps;
        double finalReward;
        double mcReturn0;
        double sumRewards;
        double lastReward;
    }

    private static double computeDiscountedReturn0(List<Double> rewards, double gamma) {
        if (rewards == null || rewards.isEmpty()) {
            return 0.0;
        }
        double ret = 0.0;
        double g = 1.0;
        for (double r : rewards) {
            ret += g * r;
            g *= gamma;
        }
        return ret;
    }

    /**
     * Calculate immediate rewards for each step in the trajectory. GAE
     * (Generalized Advantage Estimation) will be computed in Python using these
     * rewards.
     *
     * @param trajectory List of training data for one episode
     * @param finalReward Terminal reward (+1 for win, -1 for loss)
     * @return List of immediate rewards (one per step)
     */
    public static List<Double> calculateImmediateRewards(List<StateSequenceBuilder.TrainingData> trajectory, double finalReward) {
        if (trajectory.isEmpty()) {
            return new ArrayList<>();
        }

        List<Double> immediateRewards = new ArrayList<>(trajectory.size());

        for (int i = 0; i < trajectory.size(); i++) {
            double reward = trajectory.get(i).stepReward;

            // Add terminal reward to the last step
            if (i == trajectory.size() - 1) {
                reward += finalReward;
            }

            immediateRewards.add(reward);
        }

        return immediateRewards;
    }

    /**
     * DEPRECATED: Old Monte Carlo returns calculation. Kept for reference but
     * no longer used (replaced by GAE in Python).
     */
    @Deprecated
    public static List<Double> calculateDiscountedReturns(List<StateSequenceBuilder.TrainingData> trajectory, double finalReward, double gamma) {
        if (trajectory.isEmpty()) {
            return new ArrayList<>();
        }

        List<Double> discountedReturns = new ArrayList<>(Collections.nCopies(trajectory.size(), 0.0));
        double cumulativeReturn = 0.0;

        // Apply terminal reward plus any per-step shaping reward.
        // We iterate backwards from the end of the game.
        for (int i = trajectory.size() - 1; i >= 0; i--) {
            double immediateReward = trajectory.get(i).stepReward;
            if (i == trajectory.size() - 1) {
                immediateReward += finalReward;
            }

            // Removed step penalty - let value head learn from win/loss signals only
            final double STEP_PENALTY = 0.0;
            cumulativeReturn = (immediateReward + STEP_PENALTY) + gamma * cumulativeReturn;
            discountedReturns.set(i, cumulativeReturn);
        }

        // Normalize returns for stability (optional but good practice)
        double mean = discountedReturns.stream().mapToDouble(d -> d).average().orElse(0.0);
        double std = Math.sqrt(discountedReturns.stream().mapToDouble(d -> Math.pow(d - mean, 2)).average().orElse(0.0));
        if (std > 1e-6) { // Avoid division by zero
            for (int i = 0; i < discountedReturns.size(); i++) {
                discountedReturns.set(i, (discountedReturns.get(i) - mean) / std);
            }
        }

        return discountedReturns;
    }

    public static PythonModel getSharedModel() {
        return sharedModel;
    }

    /**
     * Records a game outcome and updates rolling winrate. Thread-safe for
     * concurrent training.
     *
     * @return Array [winrate, sampleSize] at the moment this outcome was
     * recorded
     */
    private double[] recordGameOutcome(boolean rlPlayerWon) {
        ConcurrentLinkedQueue<Boolean> wins = recentWinsQueue();
        AtomicInteger wc = winCounter();
        wins.add(rlPlayerWon);
        if (rlPlayerWon) {
            wc.incrementAndGet();
        }
        gamesAtLevel().incrementAndGet();
        while (wins.size() > WINRATE_WINDOW) {
            Boolean removed = wins.poll();
            if (removed != null && removed) {
                wc.decrementAndGet();
            }
        }
        int size = wins.size();
        double winrate = size == 0 ? 0.0 : wc.get() / (double) size;
        return new double[]{winrate, size};
    }

    private double getCurrentWinrate() {
        ConcurrentLinkedQueue<Boolean> wins = recentWinsQueue();
        int size = wins.size();
        if (size == 0) {
            return 0.0;
        }
        return winCounter().get() / (double) size;
    }

    private final java.util.concurrent.ExecutorService evalExecutor =
            java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "GAME");
                t.setDaemon(true);
                return t;
            });
    private final java.util.concurrent.atomic.AtomicBoolean evalRunning =
            new java.util.concurrent.atomic.AtomicBoolean(false);
    private final java.util.concurrent.atomic.AtomicBoolean firstEvalDone =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    private void submitEvalCheckpoint(int triggerEpisode, List<Path> opponentDeckPool,
                                      List<Path> agentDeckPool) {
        if (!evalRunning.compareAndSet(false, true)) {
            logger.info("EVAL checkpoint skipped (previous eval still running) at episode " + triggerEpisode);
            return;
        }
        // Snapshot deck pools (they're immutable lists, but be safe)
        List<Path> oppDecks = new ArrayList<>(opponentDeckPool);
        List<Path> agentDecks = new ArrayList<>(agentDeckPool);
        evalExecutor.submit(() -> {
            try {
                runEvalCheckpoint(triggerEpisode, oppDecks, agentDecks, new Random(triggerEpisode));
            } finally {
                evalRunning.set(false);
            }
        });
    }

    /**
     * Run a batch of eval games vs CP7 across all opponent decks. No training
     * data is generated and training_stats.csv is not touched, so rolling
     * training winrate remains a pure training signal.
     */
    private void runEvalCheckpoint(int triggerEpisode, List<Path> opponentDeckPool,
                                   List<Path> agentDeckPool, Random rand) {
        int skill = EVAL_CP7_SKILL;
        int gamesPerDeck = EVAL_GAMES_PER_DECK;
        int totalGames = opponentDeckPool.size() * gamesPerDeck * agentDeckPool.size();
        int wins = 0;
        int played = 0;
        // Per opponent deck: aggregate across agent decks (backward-compatible CSV).
        java.util.Map<String, int[]> perDeck = new java.util.LinkedHashMap<>();
        // Per (agent_deck, opp_deck) cell for detailed breakdown.
        java.util.Map<String, int[]> perCell = new java.util.LinkedHashMap<>();

        logger.info(String.format("EVAL checkpoint at episode %d: %d games (%d agent x %d opp x %d games) vs CP7(skill=%d)",
                triggerEpisode, totalGames, agentDeckPool.size(), opponentDeckPool.size(), gamesPerDeck, skill));

        for (Path agentDeckPath : agentDeckPool) {
            String agentName = agentDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "");
        for (Path oppDeckPath : opponentDeckPool) {
            String deckName = oppDeckPath.getFileName().toString().replaceAll("\\.[^.]+$", "");
            perDeck.putIfAbsent(deckName, new int[]{0, 0}); // [wins, total]
            String cellKey = agentName + "::" + deckName;
            perCell.putIfAbsent(cellKey, new int[]{0, 0});
            for (int g = 0; g < gamesPerDeck; g++) {
                try {
                    Deck agentDeck = loadDeckFresh(agentDeckPath.toString());
                    Deck oppDeck = loadDeckFresh(oppDeckPath.toString());
                    if (agentDeck == null || oppDeck == null) {
                        logger.warn("EVAL: failed to load decks, skipping game");
                        continue;
                    }

                    // Eval uses greedy mode (no exploration) and training disabled
                    ComputerPlayerRL rlPlayer = new ComputerPlayerRL("EvalRL", RangeOfInfluence.ALL, sharedModel, true, false, "train");
                    rlPlayer.setCurrentEpisode(-1);  // -1 disables exploration in isMainExplorationEnabled
                    Player cp7 = new ComputerPlayer7("EvalCP7Skill" + skill, RangeOfInfluence.ALL, skill);

                    TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("EvalMatch", "EvalMatch", false));
                    match.startGame();
                    Game game = match.getGames().get(0);

                    // Log eval games to profile eval directory
                    String evalLogDir = profileContext != null
                            ? profileContext.paths.evalGameLogsDir
                            : logsBaseDir() + "/eval_games";
                    new java.io.File(evalLogDir).mkdirs();
                    GameLogger evalLogger = GameLogger.createInProfileDir(evalLogDir, 50);
                    threadLocalGameLogger.set(evalLogger);
                    rlPlayer.setAttachedGameLogger(evalLogger);

                    game.addPlayer(rlPlayer, agentDeck);
                    match.addPlayer(rlPlayer, agentDeck);
                    game.addPlayer(cp7, oppDeck);
                    match.addPlayer(cp7, oppDeck);

                    game.loadCards(agentDeck.getCards(), rlPlayer.getId());
                    game.loadCards(oppDeck.getCards(), cp7.getId());

                    GameOptions options = new GameOptions();
                    options.rollbackTurnsAllowed = false;
                    game.setGameOptions(options);

                    GameHealthMonitor hm = GameHealthMonitor.createAndStart(game);
                    long startNanos = System.nanoTime();
                    game.start(rlPlayer.getId());
                    long endNanos = System.nanoTime();
                    hm.stop();

                    boolean won = hm.wasKilled() ? false : game.getWinner().contains(rlPlayer.getName());
                    int turns = game.getTurnNum();
                    double secs = (endNanos - startNanos) / 1_000_000_000.0;
                    played++;
                    if (won) wins++;
                    int[] dc = perDeck.get(deckName);
                    dc[1]++;
                    if (won) dc[0]++;
                    int[] cc = perCell.get(cellKey);
                    cc[1]++;
                    if (won) cc[0]++;
                    String evalTag = String.format("EVAL-CP7(skill=%d,agent=%s,opp=%s)", skill, agentName, deckName);

                    // Close eval game log
                    if (evalLogger.isEnabled()) {
                        String winner = won ? rlPlayer.getName() : cp7.getName();
                        String loser = won ? cp7.getName() : rlPlayer.getName();
                        evalLogger.logOutcome(winner, loser, turns,
                                String.format("Eval ep=%d vs %s", triggerEpisode, evalTag));
                        evalLogger.close();
                    }
                    threadLocalGameLogger.remove();

                    game.end();
                    match.getGames().clear();
                } catch (Exception e) {
                    logger.warn("EVAL game failed: " + e.getMessage());
                }
            }
        }
        }  // end agent-deck loop

        double evalWinrate = played > 0 ? (double) wins / played : 0.0;
        logger.info(String.format("EVAL checkpoint ep=%d complete: %d/%d wins (%.1f%%) vs CP7(skill=%d)",
                triggerEpisode, wins, played, evalWinrate * 100, skill));
        // Per-cell matchup breakdown: lets us tell "agent deck X vs opp deck Y" apart.
        if (agentDeckPool.size() > 1) {
            for (java.util.Map.Entry<String, int[]> e : perCell.entrySet()) {
                int[] cc = e.getValue();
                if (cc[1] > 0) {
                    logger.info(String.format("  CELL %s -> %d/%d (%.1f%%)",
                            e.getKey(), cc[0], cc[1], 100.0 * cc[0] / cc[1]));
                }
            }
        }

        // Write aggregate to eval_history.csv
        if (GAME_STATS_WRITER && played > 0) {
            String statsDir = profileContext != null
                    ? profileContext.paths.logsBaseDir + "/stats"
                    : logsBaseDir() + "/stats";
            new java.io.File(statsDir).mkdirs();
            Path evalPath = Paths.get(statsDir, "eval_history.csv");
            StringBuilder deckCols = new StringBuilder();
            StringBuilder deckHeader = new StringBuilder();
            for (java.util.Map.Entry<String, int[]> e : perDeck.entrySet()) {
                deckHeader.append(",wr_").append(e.getKey().replace(' ', '_'));
                int[] dc = e.getValue();
                double dwr = dc[1] > 0 ? (double) dc[0] / dc[1] : 0.0;
                deckCols.append(',').append(String.format("%.3f", dwr));
            }
            String header = "episode,wins,played,winrate" + deckHeader + "\n";
            String line = new StringBuilder()
                    .append(triggerEpisode).append(',')
                    .append(wins).append(',')
                    .append(played).append(',')
                    .append(String.format("%.3f", evalWinrate))
                    .append(deckCols).append('\n')
                    .toString();
            ASYNC_LINE_WRITER.append(evalPath, header, line);

            // Multi-agent-deck profiles: also write per-cell (agent x opp) table
            // to eval_cells.csv so we can see generalist's per-archetype performance.
            if (agentDeckPool.size() > 1) {
                Path cellsPath = Paths.get(statsDir, "eval_cells.csv");
                StringBuilder cellHeader = new StringBuilder("episode");
                StringBuilder cellLine = new StringBuilder().append(triggerEpisode);
                for (java.util.Map.Entry<String, int[]> e : perCell.entrySet()) {
                    int[] cc = e.getValue();
                    double cwr = cc[1] > 0 ? (double) cc[0] / cc[1] : 0.0;
                    cellHeader.append(",wr_").append(e.getKey().replace(' ', '_').replace("::", "__vs__"));
                    cellLine.append(',').append(String.format("%.3f", cwr));
                }
                cellHeader.append('\n');
                cellLine.append('\n');
                ASYNC_LINE_WRITER.append(cellsPath, cellHeader.toString(), cellLine.toString());
            }
        }
    }

    private Player createTrainingOpponent(int episodeNum, Random rand) {
        String mode = OPPONENT_SAMPLER == null ? "league" : OPPONENT_SAMPLER.trim().toLowerCase();
        switch (mode) {
            case "self":
                return createSelfPlayOpponent();
            case "meta":
                return createMetaOpponent(rand);
            case "adaptive":
                return createAdaptiveOpponent(episodeNum, rand);
            case "fixed":
                // Force fixed schedule by temporarily disabling adaptive logic.
                // (This preserves existing behavior with FIXED_* env vars.)
                return createFixedOpponent(episodeNum, rand);
            case "ladder":
                return createLadderOpponent(episodeNum, rand);
            case "skillmix":
                return createSkillMixOpponent(rand);
            case "hybrid":
                return createHybridOpponent(rand);
            case "meta_hybrid":
            case "metahybrid":
            case "profile_hybrid":
            case "profilehybrid":
                return createMetaHybridOpponent(rand);
            case "league":
            default:
                return createLeagueOpponent(episodeNum, rand);
        }
    }

    private Player createSelfPlayOpponent() {
        lastOpponentType = "SELFPLAY";
        return newSelfPlayOpponent("SelfPlay");
    }

    private Player newSelfPlayOpponent(String name) {
        return new ComputerPlayerRL(name, RangeOfInfluence.ALL, sharedModel,
                false, SELFPLAY_OPPONENT_TRAINING, "train");
    }

    private Player createMetaOpponent(Random rand) {
        // Pick a random profile from the registry and play against its model + deck
        try {
            Path registryPath = leagueRegistryPath();
            if (!Files.exists(registryPath)) {
                logger.warn("Meta opponent: registry not found, falling back to self-play");
                return createSelfPlayOpponent();
            }
            String json = new String(Files.readAllBytes(registryPath), java.nio.charset.StandardCharsets.UTF_8);
            com.google.gson.JsonArray entries = com.google.gson.JsonParser.parseString(json).getAsJsonArray();
            // Filter to profiles that are active, train-enabled, and in the same
            // population group as the current TRAIN_PROFILES_LIST
            String trainProfilesList = System.getenv("TRAIN_PROFILES_LIST");
            java.util.Set<String> trainProfiles = new java.util.HashSet<>();
            if (trainProfilesList != null) {
                for (String p : trainProfilesList.split(",")) {
                    if (!p.trim().isEmpty()) trainProfiles.add(p.trim());
                }
            }
            List<com.google.gson.JsonObject> active = new ArrayList<>();
            for (com.google.gson.JsonElement e : entries) {
                com.google.gson.JsonObject obj = e.getAsJsonObject();
                if (obj.has("active") && obj.get("active").getAsBoolean()
                        && obj.has("train_enabled") && obj.get("train_enabled").getAsBoolean()) {
                    // If we have a train profiles list, only pick from those profiles
                    if (!trainProfiles.isEmpty()) {
                        String profile = obj.get("profile").getAsString().trim();
                        if (!trainProfiles.contains(profile)) continue;
                    }
                    active.add(obj);
                }
            }
            if (active.isEmpty()) {
                return createSelfPlayOpponent();
            }
            // Pick random profile (can be self -- that's fine, acts as self-play fraction)
            com.google.gson.JsonObject chosen = active.get(rand.nextInt(active.size()));
            String oppProfile = chosen.get("profile").getAsString().trim();
            // Prefer RL_AGENT_DECK_LIST from train_env (the profile's specific deck)
            // over deck_path (the opponent pool)
            String oppDeck = "";
            if (chosen.has("train_env")) {
                com.google.gson.JsonObject trainEnv = chosen.getAsJsonObject("train_env");
                if (trainEnv.has("RL_AGENT_DECK_LIST")) {
                    oppDeck = trainEnv.get("RL_AGENT_DECK_LIST").getAsString().trim();
                }
            }
            if (oppDeck.isEmpty() && chosen.has("deck_path")) {
                oppDeck = chosen.get("deck_path").getAsString().trim();
            }
            if (!oppDeck.isEmpty()) {
                THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.set(Paths.get(oppDeck));
            }
            // policyKey = opponent profile name, routes inference to that profile's model on GPU
            // trainingEnabled=true so opponent exploration + training data is recorded
            return new ComputerPlayerRL("Meta-" + oppProfile, RangeOfInfluence.ALL, sharedModel, false, true, oppProfile);
        } catch (Exception e) {
            logger.warn("Meta opponent: failed to load registry: " + e.getMessage() + ", falling back to self-play");
            return createSelfPlayOpponent();
        }
    }

    private Player createFixedOpponent(int episodeNum, Random rand) {
        // Legacy fixed schedule, but keep a permanent bot floor by mixing bots even after self-play.
        if (episodeNum < FIXED_WEAK_UNTIL) {
            return new ComputerPlayer7("WeakBot", RangeOfInfluence.ALL, 1);
        } else if (episodeNum < FIXED_MEDIUM_UNTIL) {
            return new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 4);
        } else if (episodeNum < FIXED_STRONG_UNTIL) {
            return new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 7);
        } else {
            double botFloor = Math.max(0.0, Math.min(1.0, BOT_FLOOR_P));
            if (rand.nextDouble() < botFloor) {
                return pickBotFromMix(rand);
            }
            return newSelfPlayOpponent("SelfPlay");
        }
    }

    private Player createLeagueOpponent(int episodeNum, Random rand) {
        LeagueState st = getLeagueState();
        THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.remove();

        // Full-meta league sampling:
        // - Sample opponent profile from active registry entries (excluding self when possible).
        // - If that profile is qualified (promoted and has snapshot), play vs its RL snapshot.
        // - Otherwise play heuristic CP7 on that profile's deck.
        java.util.List<LeagueMetaOpponentCandidate> allMeta = getLeagueMetaOpponentCandidates();
        if (!allMeta.isEmpty()) {
            java.util.ArrayList<LeagueMetaOpponentCandidate> meta = new java.util.ArrayList<>(allMeta);
            if (!profileName().isEmpty() && meta.size() > 1) {
                final String pn = profileName();
                meta.removeIf(c -> pn.equalsIgnoreCase(c.profile));
            }
            if (!meta.isEmpty()) {
                LeagueMetaOpponentCandidate pick = meta.get(rand.nextInt(meta.size()));
                THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.set(pick.deckPath);
                boolean rlOnly = "rl_only".equalsIgnoreCase(LEAGUE_MODE);
                if (pick.qualified && pick.snapshotPath != null) {
                    String policyKey = "snap:" + pick.snapshotPath.toString();
                    lastOpponentType = String.format(java.util.Locale.US,
                            "META-RL(profile=%s,ep=%d,wr=%.3f,promoted=%s)",
                            pick.profile, pick.episode, pick.baselineWr, pick.promoted);
                    return new ComputerPlayerRL("MetaSnapshotOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
                }
                if (rlOnly) {
                    // No snapshot yet -- use current training policy (random weights at cold start)
                    String policyKey = "profile:" + pick.profile;
                    lastOpponentType = String.format(java.util.Locale.US,
                            "META-RL-LIVE(profile=%s,wr=%.3f)",
                            pick.profile, pick.baselineWr);
                    return new ComputerPlayerRL("LiveRLOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
                }
                int skill = Math.max(1, LEAGUE_POST_HEURISTIC_SKILL);
                lastOpponentType = String.format(java.util.Locale.US,
                        "META-H(profile=%s,skill=%d,wr=%.3f,promoted=%s)",
                        pick.profile, skill, pick.baselineWr, pick.promoted);
                return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
            }
        }

        // Legacy fallback if registry/meta candidates are unavailable.
        return createLeagueOpponentLegacy(st, rand);
    }

    private Player createLeagueOpponentLegacy(LeagueState st, Random rand) {
        // Stage 0 (bootstrap): train against CP7Skill1 only until promoted.
        if (!st.promoted) {
            int skill = Math.max(1, LEAGUE_BASELINE_BOT_SKILL);
            lastOpponentType = "H-CP7(skill=" + skill + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        }

        // After promotion:
        // 20% heuristic (CP7 skill1),
        // 40% local profile snapshot/self-play,
        // 40% cross-profile snapshot opponent.
        double pH = clamp01(LEAGUE_POST_HEURISTIC_P);
        double pL = clamp01(LEAGUE_POST_LOCAL_P);
        double pX = clamp01(LEAGUE_POST_CROSS_P);
        double sum = pH + pL + pX;
        if (sum <= 0) {
            pH = 0.20;
            pL = 0.40;
            pX = 0.40;
            sum = 1.0;
        }
        pH /= sum;
        pL /= sum;
        pX /= sum;

        double r = rand.nextDouble();
        if (r < pH) {
            int skill = Math.max(1, LEAGUE_POST_HEURISTIC_SKILL);
            lastOpponentType = "H-CP7(skill=" + skill + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        } else if (r < pH + pL) {
            String snapKey = pickLeagueSnapshotPolicyKey(st, rand);
            boolean useSnapshot = snapKey != null && rand.nextBoolean();
            if (useSnapshot) {
                lastOpponentType = "LOCAL-SNAP(" + snapKey + ")";
                return new ComputerPlayerRL("LocalSnapshotOpp", RangeOfInfluence.ALL, sharedModel, false, false, snapKey);
            }
            lastOpponentType = "LOCAL-SELFPLAY";
            return new ComputerPlayerRL("SelfPlayLocal", RangeOfInfluence.ALL, sharedModel, false, false, "train");
        } else {
            java.util.List<CrossProfileSnapshot> candidates = getCrossProfileCandidates();
            if (!candidates.isEmpty()) {
                CrossProfileSnapshot pick = candidates.get(rand.nextInt(candidates.size()));
                THREAD_LOCAL_OPPONENT_DECK_OVERRIDE.set(pick.deckPath);
                String policyKey = "snap:" + pick.snapshotPath.toString();
                lastOpponentType = "CROSS(" + pick.profile + ",ep=" + pick.episode + ")";
                return new ComputerPlayerRL("CrossProfileOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
            }

            // Fallback if no cross-profile candidates are available.
            String snapKey = pickLeagueSnapshotPolicyKey(st, rand);
            if (snapKey != null) {
                lastOpponentType = "LOCAL-SNAP-FALLBACK(" + snapKey + ")";
                return new ComputerPlayerRL("LocalSnapshotFallback", RangeOfInfluence.ALL, sharedModel, false, false, snapKey);
            }
            int skill = Math.max(1, LEAGUE_POST_HEURISTIC_SKILL);
            lastOpponentType = "H-CP7-FALLBACK(skill=" + skill + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        }
    }

    private Player createLadderOpponent(int episodeNum, Random rand) {
        int[] tiers = parseSkillList(LADDER_SKILLS, new int[]{0, 1, 2, 3});
        if (tiers.length == 0) {
            // Fallback to skill 1 if no tiers defined
            lastOpponentType = "L-CP7(skill=1,tier=0)";
            return new ComputerPlayer7("Bot-Skill1", RangeOfInfluence.ALL, 1);
        }
        
        LadderState ls = getLadderState();
        int currentTier = Math.min(ls.currentTier, tiers.length - 1);

        // Mix in lower-tier bots to prevent forgetting
        if (currentTier > 0 && rand.nextDouble() < LADDER_MIX_LOWER_P) {
            int lowerTier = rand.nextInt(currentTier); // random lower tier
            int skill = tiers[lowerTier];
            lastOpponentType = "L-CP7(skill=" + skill + ",tier=" + lowerTier + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        }

        int skill = tiers[currentTier];
        lastOpponentType = "L-CP7(skill=" + skill + ",tier=" + currentTier + ")";
        return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
    }

    private Player createSkillMixOpponent(Random rand) {
        java.util.ArrayList<Integer> skills = new java.util.ArrayList<>();
        java.util.ArrayList<Double> weights = new java.util.ArrayList<>();
        if (SKILL_MIX != null) {
            for (String part : SKILL_MIX.split(",")) {
                String token = part == null ? "" : part.trim();
                if (token.isEmpty()) {
                    continue;
                }
                String[] pieces = token.split("[:=]", 2);
                try {
                    int skill = Integer.parseInt(pieces[0].trim());
                    double weight = pieces.length > 1 ? Double.parseDouble(pieces[1].trim()) : 1.0;
                    if (skill >= 1 && weight > 0.0 && Double.isFinite(weight)) {
                        skills.add(skill);
                        weights.add(weight);
                    }
                } catch (NumberFormatException ignored) {
                }
            }
        }
        if (skills.isEmpty()) {
            skills.add(1);
            weights.add(1.0);
        }
        double total = 0.0;
        for (double w : weights) {
            total += w;
        }
        double r = rand.nextDouble() * total;
        int pick = skills.get(skills.size() - 1);
        double seen = 0.0;
        for (int i = 0; i < skills.size(); i++) {
            seen += weights.get(i);
            if (r <= seen) {
                pick = skills.get(i);
                break;
            }
        }
        lastOpponentType = "MIX-CP7(skill=" + pick + ")";
        return new ComputerPlayer7("Bot-Skill" + pick, RangeOfInfluence.ALL, pick);
    }

    private Player createHybridOpponent(Random rand) {
        double pSelf = clamp01(HYBRID_SELFPLAY_P);
        if (rand.nextDouble() < pSelf) {
            return createSelfPlayOpponent();
        }
        return createSkillMixOpponent(rand);
    }

    private Player createMetaHybridOpponent(Random rand) {
        double pMeta = clamp01(META_HYBRID_META_P);
        if (rand.nextDouble() < pMeta) {
            return createMetaOpponent(rand);
        }
        return createSkillMixOpponent(rand);
    }

    private static double clamp01(double v) {
        if (v < 0.0) {
            return 0.0;
        }
        if (v > 1.0) {
            return 1.0;
        }
        return v;
    }

    private static int[] parseSkillList(String s, int[] def) {
        if (s == null || s.trim().isEmpty()) {
            return def;
        }
        String[] parts = s.split(",");
        java.util.ArrayList<Integer> out = new java.util.ArrayList<>();
        for (String p : parts) {
            String t = p.trim();
            if (t.isEmpty()) {
                continue;
            }
            try {
                int v = Integer.parseInt(t);
                if (v >= 0) {
                    out.add(v);
                }
            } catch (NumberFormatException ignored) {
            }
        }
        if (out.isEmpty()) {
            return def;
        }
        int[] arr = new int[out.size()];
        for (int i = 0; i < out.size(); i++) {
            arr[i] = out.get(i);
        }
        return arr;
    }

    private static int pickFrom(int[] arr, Random rand, int def) {
        if (arr == null || arr.length == 0) {
            return def;
        }
        return arr[rand.nextInt(arr.length)];
    }

    private static String pickLeagueSnapshotPolicyKey(LeagueState st, Random rand) {
        if (st == null) {
            return null;
        }
        java.util.ArrayList<String> champs = new java.util.ArrayList<>();
        if (st.championPolicyKey != null && !st.championPolicyKey.trim().isEmpty()) {
            champs.add(st.championPolicyKey.trim());
        }
        java.util.ArrayList<String> recent = new java.util.ArrayList<>(st.recent);
        // Ensure recent entries exist in pool
        recent.removeIf(x -> x == null || x.trim().isEmpty());
        champs.removeIf(x -> x == null || x.trim().isEmpty());

        boolean takeRecent = rand.nextBoolean(); // 50/50 recent vs champions
        if (takeRecent && !recent.isEmpty()) {
            return recent.get(rand.nextInt(recent.size()));
        }
        if (!champs.isEmpty()) {
            return champs.get(rand.nextInt(champs.size()));
        }
        // fallback: any pool member
        java.util.ArrayList<String> pool = new java.util.ArrayList<>(st.pool);
        pool.removeIf(x -> x == null || x.trim().isEmpty());
        if (pool.isEmpty()) {
            return null;
        }
        return pool.get(rand.nextInt(pool.size()));
    }

    private String pickSnapshotPolicyKey(Random rand) {
        try {
            java.nio.file.Path dir = java.nio.file.Paths.get(SNAPSHOT_DIR);
            if (!java.nio.file.Files.isDirectory(dir)) {
                return null;
            }
            java.util.List<java.nio.file.Path> snaps;
            try (java.util.stream.Stream<java.nio.file.Path> stream = java.nio.file.Files.list(dir)) {
                snaps = stream
                        .filter(p -> p.getFileName() != null && p.getFileName().toString().endsWith(".pt"))
                        .collect(java.util.stream.Collectors.toList());
            }
            if (snaps.isEmpty()) {
                return null;
            }
            java.nio.file.Path pick = snaps.get(rand.nextInt(snaps.size()));
            return "snap:" + pick.getFileName().toString();
        } catch (Exception ignored) {
            return null;
        }
    }

    private Player pickBotFromMix(Random rand) {
        double[] w = parseBotMix(BOT_MIX);
        double r = rand.nextDouble() * (w[0] + w[1] + w[2]);
        int skill;
        if (r < w[0]) {
            skill = 1;
        } else if (r < w[0] + w[1]) {
            skill = 2;
        } else {
            skill = 3;
        }
        lastOpponentType = "BOT-CP7(skill=" + skill + ")";
        return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
    }

    private static double[] parseBotMix(String s) {
        // default: [0.25, 0.35, 0.40]
        double[] def = new double[]{0.25, 0.35, 0.40};
        if (s == null || s.trim().isEmpty()) {
            return def;
        }
        String[] parts = s.split(",");
        if (parts.length != 3) {
            return def;
        }
        try {
            double a = Double.parseDouble(parts[0].trim());
            double b = Double.parseDouble(parts[1].trim());
            double c = Double.parseDouble(parts[2].trim());
            if (a < 0 || b < 0 || c < 0) {
                return def;
            }
            double sum = a + b + c;
            if (sum <= 0) {
                return def;
            }
            return new double[]{a, b, c};
        } catch (NumberFormatException ignored) {
            return def;
        }
    }

    /**
     * Creates an opponent based on adaptive curriculum with hysteresis.
     *
     * Adaptive Strategy with Hysteresis: - Start with weak opponent (no
     * simulation) - Upgrade to medium at 40% winrate, downgrade at 35% -
     * Upgrade to strong at 55% winrate, downgrade at 50% - Upgrade to self-play
     * at 65% winrate, downgrade at 60%
     *
     * Hysteresis prevents oscillation at threshold boundaries. Reserve skill=6+
     * for evaluation/benchmarking only.
     */
    private Player createAdaptiveOpponent(int episodeNum, Random rand) {
        if (ADAPTIVE_CURRICULUM) {
            double winRate = getCurrentWinrate();
            int sampleSize = recentWins.size();
            int currentLevelGames = gamesAtCurrentLevel.get();

            // Bootstrap phase: not enough data for promotion decisions
            // But still use the current difficulty level (don't revert to WEAK after promotion)
            boolean isBootstrapping = sampleSize < Math.min(20, WINRATE_WINDOW / 5);

            if (isBootstrapping && currentLevelGames % 20 == 0 && currentLevelGames < MIN_GAMES_PER_DIFFICULTY) {
                // Log bootstrap progress at current level
                logger.info(String.format("Bootstrap: %d/%d games at %s (need %d for transitions)",
                        currentLevelGames, Math.min(20, WINRATE_WINDOW / 5),
                        currentOpponentLevel, MIN_GAMES_PER_DIFFICULTY));
            }

            // Determine new opponent level using hysteresis
            // IMPORTANT: Only allow transitions after minimum games at current level
            // This prevents contamination from previous difficulty levels
            OpponentLevel newLevel = currentOpponentLevel;
            boolean hasEnoughGames = currentLevelGames >= MIN_GAMES_PER_DIFFICULTY && !isBootstrapping;

            // Only check for level changes if we have enough games at current difficulty
            if (hasEnoughGames) {
                switch (currentOpponentLevel) {
                    case WEAK:
                        // Can only upgrade from WEAK
                        if (winRate >= WEAK_TO_MEDIUM_THRESHOLD) {
                            newLevel = OpponentLevel.MEDIUM;
                        }
                        break;

                    case MEDIUM:
                        // Can upgrade or downgrade from MEDIUM
                        if (winRate >= MEDIUM_TO_STRONG_THRESHOLD) {
                            newLevel = OpponentLevel.STRONG;
                        } else if (winRate < MEDIUM_TO_WEAK_THRESHOLD) {
                            newLevel = OpponentLevel.WEAK;
                        }
                        break;

                    case STRONG:
                        // Can upgrade or downgrade from STRONG
                        if (winRate >= STRONG_TO_SELFPLAY_THRESHOLD) {
                            newLevel = OpponentLevel.SELFPLAY;
                        } else if (winRate < STRONG_TO_MEDIUM_THRESHOLD) {
                            newLevel = OpponentLevel.MEDIUM;
                        }
                        break;

                    case SELFPLAY:
                        // Can only downgrade from SELFPLAY
                        if (winRate < SELFPLAY_TO_STRONG_THRESHOLD) {
                            newLevel = OpponentLevel.STRONG;
                        }
                        break;
                }
            }

            // Create opponent based on level
            String opType;
            Player opponent;

            // CP7 skill levels are configurable via env because skill ~= search depth:
            // skill=9 is a 9-ply alpha-beta search that can take 15-30s per decision,
            // crushing training throughput. Default training skills are 4/5/6 instead
            // of 7/8/9 — still challenging but ~100x faster. Eval still uses skill=7
            // (EVAL_CP7_SKILL) for consistent benchmarking.
            int skillWeak = EnvConfig.i32("CURRICULUM_SKILL_WEAK", 4);
            int skillMedium = EnvConfig.i32("CURRICULUM_SKILL_MEDIUM", 5);
            int skillStrong = EnvConfig.i32("CURRICULUM_SKILL_STRONG", 6);

            switch (newLevel) {
                case WEAK:
                    opType = "WEAK-CP7(skill=" + skillWeak + ")";
                    opponent = new ComputerPlayer7("WeakBot", RangeOfInfluence.ALL, skillWeak);
                    break;

                case MEDIUM:
                    opType = "MEDIUM-CP7(skill=" + skillMedium + ")";
                    opponent = new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, skillMedium);
                    break;

                case STRONG:
                    opType = "STRONG-CP7(skill=" + skillStrong + ")";
                    opponent = new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, skillStrong);
                    break;

                case SELFPLAY:
                    // Mostly self-play with occasional strong heuristic for stability
                    if (rand.nextDouble() < 0.9) {
                        opType = "SELFPLAY";
                        opponent = newSelfPlayOpponent("SelfPlay");
                    } else {
                        opType = "STRONG-CP7(skill=3)";
                        opponent = new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
                    }
                    break;

                default:
                    opType = "WEAK-CP7(skill=1)";
                    opponent = new ComputerPlayer7("WeakBot", RangeOfInfluence.ALL, 1);
            }

            // Log opponent transitions and reset tracking when level changes
            if (newLevel != currentOpponentLevel || !opType.equals(lastOpponentType)) {
                String action = newLevel.ordinal() > currentOpponentLevel.ordinal() ? "upgraded"
                        : (newLevel.ordinal() < currentOpponentLevel.ordinal() ? "downgraded" : "changed");
                logger.info(String.format("Opponent %s to: %s (winrate=%.3f over %d games at %s, episode=%d)",
                        action, opType, winRate, currentLevelGames, currentOpponentLevel, episodeNum));

                // Reset tracking when level changes to avoid contamination
                if (newLevel != currentOpponentLevel) {
                    recentWins.clear();
                    winCount.set(0);
                    gamesAtCurrentLevel.set(0);
                    logger.info(String.format("Reset winrate tracking for new difficulty level"));
                }

                lastOpponentType = opType;
                currentOpponentLevel = newLevel;
            } else if (!hasEnoughGames && currentLevelGames % 20 == 0) {
                // Log progress toward minimum games requirement (every 20 games)
                logger.info(String.format("Opponent: %s (winrate=%.3f, %d/%d games at %s, episode=%d)",
                        opType, winRate, currentLevelGames, MIN_GAMES_PER_DIFFICULTY, currentOpponentLevel, episodeNum));
            }

            return opponent;

        } else {
            // Fixed schedule (legacy behavior for comparison)
            // Updated: start at skill=1 instead of useless ComputerPlayer
            if (episodeNum < FIXED_WEAK_UNTIL) {
                return new ComputerPlayer7("WeakBot", RangeOfInfluence.ALL, 1);
            } else if (episodeNum < FIXED_MEDIUM_UNTIL) {
                return new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 2);
            } else if (episodeNum < FIXED_STRONG_UNTIL) {
                return new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
            } else {
                // Pure self-play after threshold
                return rand.nextDouble() < 0.9
                        ? newSelfPlayOpponent("SelfPlay")
                        : new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
            }
        }
    }

    /**
     * @deprecated Use createAdaptiveOpponent() instead for better curriculum
     * learning
     */
    @Deprecated
    private boolean shouldUseHeuristicOpponent(int episodeIdx, Random rand) {
        if (episodeIdx < 5000) {
            return true;
        } else if (episodeIdx < 15000) {
            return rand.nextDouble() < 0.5;
        } else {
            return false;
        }
    }
}
