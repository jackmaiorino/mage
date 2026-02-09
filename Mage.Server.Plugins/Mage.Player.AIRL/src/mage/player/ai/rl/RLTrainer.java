package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

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

    // Prefer MTG_MODEL_PATH (Python-side convention), but keep MODEL_PATH for backward compatibility.
    public static final String MODEL_FILE_PATH = EnvConfig.str("MTG_MODEL_PATH",
            EnvConfig.str("MODEL_PATH", "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt"));
    // Episode-level statistics will be appended here (CSV)
    public static final String STATS_FILE_PATH = RLLogPaths.TRAINING_STATS_PATH;
    // Path that stores the cumulative number of episodes trained so far (persisted across runs)
    public static final String EPISODE_COUNT_PATH = EnvConfig.str("EPISODE_COUNTER_PATH",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/episodes.txt");
    // Auto-detect optimal number of threads based on CPU cores
    private static final int DEFAULT_GAME_RUNNERS = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    public static final int NUM_THREADS = EnvConfig.i32("NUM_THREADS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_GAME_RUNNERS = EnvConfig.i32("NUM_GAME_RUNNERS", DEFAULT_GAME_RUNNERS);
    public static final int NUM_EPISODES_PER_GAME_RUNNER = EnvConfig.i32("EPISODES_PER_WORKER", 500);
    public static final int EVAL_EVERY = EnvConfig.i32("EVAL_EVERY", 100);

    public static final PythonModel sharedModel = PythonMLService.getInstance();
    public static final MetricsCollector metrics = MetricsCollector.getInstance();

    // Global episode counter to track total episodes across all threads
    private static final AtomicInteger EPISODE_COUNTER = new AtomicInteger(0);
    private static final AtomicInteger ACTIVE_EPISODES = new AtomicInteger(0);
    private static final boolean TRAIN_DIAG = EnvConfig.bool("TRAIN_DIAG", false);
    private static final int TRAIN_DIAG_EVERY = EnvConfig.i32("TRAIN_DIAG_EVERY", 50);

    // ============================================================
    // Adaptive Curriculum Learning Configuration
    // ============================================================
    private static final boolean ADAPTIVE_CURRICULUM = EnvConfig.bool("ADAPTIVE_CURRICULUM", true);

    // Game logging: log every N episodes (0 = disabled, includes episode 0)
    private static final int GAME_LOG_FREQUENCY = EnvConfig.i32("GAME_LOG_FREQUENCY", 200);
    private static final int WINRATE_WINDOW = EnvConfig.i32("WINRATE_WINDOW", 100);

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
    private static enum OpponentLevel {
        WEAK, MEDIUM, STRONG, SELFPLAY
    }
    private static volatile OpponentLevel currentOpponentLevel = OpponentLevel.WEAK;
    private static String lastOpponentType = "UNKNOWN";

    // Track games played at current difficulty level
    private static final AtomicInteger gamesAtCurrentLevel = new AtomicInteger(0);

    // ============================================================
    // League-style opponent sampling (bots never go to zero)
    // ============================================================
    private static final String OPPONENT_SAMPLER = EnvConfig.str("OPPONENT_SAMPLER", "league"); // league|adaptive|fixed

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
    private static final double LEAGUE_P_H = EnvConfig.f64("LEAGUE_P_H", 0.30);
    private static final double LEAGUE_P_S = EnvConfig.f64("LEAGUE_P_S", 0.50);
    private static final double LEAGUE_P_C = EnvConfig.f64("LEAGUE_P_C", 0.20);
    private static final String LEAGUE_HEURISTIC_SKILLS_PRE = EnvConfig.str("LEAGUE_HEURISTIC_SKILLS_PRE", "1");
    private static final String LEAGUE_HEURISTIC_SKILLS_POST = EnvConfig.str("LEAGUE_HEURISTIC_SKILLS_POST", "1,2,3");
    private static final int LEAGUE_BENCHMARK_THREADS = EnvConfig.i32("LEAGUE_BENCHMARK_THREADS",
            Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
    private static final int LEAGUE_BENCHMARK_LOG_EVERY = EnvConfig.i32("LEAGUE_BENCHMARK_LOG_EVERY", 25);
    private static final int LEAGUE_BENCHMARK_HEARTBEAT_SEC = EnvConfig.i32("LEAGUE_BENCHMARK_HEARTBEAT_SEC", 30);
    private static final int LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC = EnvConfig.i32("LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC", 900);
    private static final boolean LEAGUE_EVAL_GAME_LOGGING = EnvConfig.bool("LEAGUE_EVAL_GAME_LOGGING", false);

    private static final Object LEAGUE_LOCK = new Object();
    private static LeagueState LEAGUE_STATE = null;
    private static final AtomicInteger LEAGUE_LAST_TICK_EP = new AtomicInteger(0);

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

    private static final class LeagueState {

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

    private static Path leagueStatePath() {
        return Paths.get(RLLogPaths.LEAGUE_STATE_PATH);
    }

    private static Path leagueEventsLogPath() {
        return Paths.get(RLLogPaths.LEAGUE_EVENTS_LOG_PATH);
    }

    private static Path leagueStatusPath() {
        return Paths.get(RLLogPaths.LEAGUE_STATUS_PATH);
    }

    private static void appendLeagueEvent(String line) {
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

    private static void writeLeagueStatus(LeagueState st) {
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

    private static LeagueState getLeagueState() {
        synchronized (LEAGUE_LOCK) {
            if (LEAGUE_STATE != null) {
                return LEAGUE_STATE;
            }
            LeagueState st = new LeagueState();
            try {
                Path p = leagueStatePath();
                if (Files.exists(p)) {
                    String s = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                    st = LeagueState.fromJson(s);
                }
            } catch (Exception ignored) {
                // ignore load failures; start fresh
            }
            LEAGUE_STATE = st;
            return LEAGUE_STATE;
        }
    }

    private static void saveLeagueState() {
        synchronized (LEAGUE_LOCK) {
            if (LEAGUE_STATE == null) {
                return;
            }
            try {
                Path p = leagueStatePath();
                if (p.getParent() != null) {
                    Files.createDirectories(p.getParent());
                }
                Files.write(p, LEAGUE_STATE.toJson().getBytes(StandardCharsets.UTF_8),
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
            } catch (Exception ignored) {
                // ignore
            }
        }
    }

    private static String leagueSnapshotFileName(int episode) {
        return "league_ep_" + episode + ".pt";
    }

    private static String leagueSnapshotPolicyKey(int episode) {
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
        int prev = LEAGUE_LAST_TICK_EP.get();
        if (episodeNum <= prev) {
            return;
        }
        if (!LEAGUE_LAST_TICK_EP.compareAndSet(prev, episodeNum)) {
            return;
        }

        final long tickStartMs = System.currentTimeMillis();
        final Random rand = new Random();
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
        Path snapPath = Paths.get(SNAPSHOT_DIR, snapFile);
        try {
            if (snapPath.getParent() != null) {
                Files.createDirectories(snapPath.getParent());
            }
        } catch (Exception ignored) {
        }
        try {
            sharedModel.saveModel(snapPath.toString());
        } catch (Exception e) {
            logger.warn("League tick: failed to save snapshot at ep " + episodeNum + " -> " + snapPath, e);
            return;
        }
        String snapKey = "snap:" + snapFile;
        logger.info("League tick snapshot saved: ep=" + episodeNum + " key=" + snapKey + " path=" + snapPath);
        appendLeagueEvent("snapshot_saved ep=" + episodeNum + " key=" + snapKey + " path=" + snapPath.toString());

        // 2) Baseline eval: S_t vs CP7Skill1 using deck-matrix benchmark (GamesPerMatchup=3 effect)
        double baselineWr = runLeagueBenchmarkPolicy(
                snapKey,
                LeagueOpponentSpec.bot(Math.max(1, LEAGUE_BASELINE_BOT_SKILL)),
                LEAGUE_BASELINE_DECKLIST_FILE,
                Math.max(1, LEAGUE_BASELINE_GAMES_PER_MATCHUP)
        );
        logger.info(String.format("League tick baseline: ep=%d key=%s vs CP7Skill%d wr=%.3f",
                episodeNum, snapKey, Math.max(1, LEAGUE_BASELINE_BOT_SKILL), baselineWr));
        appendLeagueEvent(String.format("baseline_eval ep=%d key=%s opp=CP7Skill%d wr=%.6f",
                episodeNum, snapKey, Math.max(1, LEAGUE_BASELINE_BOT_SKILL), baselineWr));

        synchronized (LEAGUE_LOCK) {
            st.lastTickEpisode = episodeNum;
            st.baselineWr.put(snapKey, baselineWr);
        }

        // 3) Promotion gate (bootstrap -> league)
        boolean promotedNow = false;
        synchronized (LEAGUE_LOCK) {
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
        synchronized (LEAGUE_LOCK) {
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
        synchronized (LEAGUE_LOCK) {
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
        synchronized (LEAGUE_LOCK) {
            championAfter = st.championPolicyKey;
        }
        if (!championUpdated && st.promoted && championAfter != null && !championAfter.equals(snapKey)) {
            logger.info("League tick champion match start: ep=" + episodeNum + " challenger=" + snapKey + " vs champion=" + championAfter);
            appendLeagueEvent("champion_match_start ep=" + episodeNum + " challenger=" + snapKey + " champion=" + championAfter);
            champMatchWr = runLeagueBenchmarkPolicy(
                    snapKey,
                    LeagueOpponentSpec.snapshot(championAfter),
                    LEAGUE_BASELINE_DECKLIST_FILE,
                    Math.max(1, LEAGUE_BASELINE_GAMES_PER_MATCHUP)
            );
            logger.info(String.format("League tick champion match done: ep=%d challenger=%s vs champion=%s wr=%.3f threshold=%.3f",
                    episodeNum, snapKey, championAfter, champMatchWr, LEAGUE_CHAMPION_PROMOTE_WR));
            appendLeagueEvent(String.format("champion_match_done ep=%d challenger=%s champion=%s wr=%.6f threshold=%.6f",
                    episodeNum, snapKey, championAfter, champMatchWr, LEAGUE_CHAMPION_PROMOTE_WR));
            synchronized (LEAGUE_LOCK) {
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
        synchronized (LEAGUE_LOCK) {
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

    public RLTrainer() {
    }

    /* ================================================================
     *  Simple CLI entry point: java RLTrainer train|eval
     * ============================================================ */
    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : EnvConfig.str("MODE", "train");

        // Start metrics collection server
        int metricsPort = EnvConfig.i32("METRICS_PORT", 9090);
        metrics.startMetricsServer(metricsPort);
        logger.info("Metrics server started on port " + metricsPort);
        try {
            if ("eval".equalsIgnoreCase(mode)) {
                runEvaluation(NUM_EVAL_EPISODES);
            } else if ("benchmark".equalsIgnoreCase(mode)) {
                int gamesPerMatchup = EnvConfig.i32("GAMES_PER_MATCHUP", 20);
                new RLTrainer().runBenchmark(gamesPerMatchup);
            } else {
                new RLTrainer().train();
            }
        } catch (Exception e) {
            logger.error("RLTrainer main failed", e);
        } finally {
            // Ensure CLI invocations exit cleanly (no lingering metrics scheduler / py4j ports).
            try {
                metrics.stop();
            } catch (Exception ignored) {
            }
            try {
                sharedModel.shutdown();
            } catch (Exception ignored) {
            }
        }
    }

    public void train() {
        System.out.println("DEBUG: Starting train() method");
        System.out.println("DEBUG: DECKS_DIRECTORY = " + DECKS_DIRECTORY);
        System.out.println("DEBUG: Working directory = " + System.getProperty("user.dir"));

        // Initialize the card database - this is crucial for deck loading to work
        System.out.println("DEBUG: Initializing card database...");
        mage.cards.repository.CardScanner.scan();
        System.out.println("DEBUG: Card database initialized");

        try {
            List<Path> deckFiles = loadDeckPool();
            System.out.println("DEBUG: Found " + deckFiles.size() + " deck files in deck pool");

            // Reset health stats and failure counters at start of training
            TrainingHealthStats healthStats = TrainingHealthStats.getInstance();
            healthStats.reset();
            logger.info("Starting training - health stats reset (games_killed=0, rl_failures=0)");

            // Start health stats logging
            healthStats.start();

            // ------------------ 1.  Load persisted episode count ------------------
            int initialEpisodeCount = 0;
            try {
                Path epPath = Paths.get(EPISODE_COUNT_PATH);
                if (Files.exists(epPath)) {
                    String content = new String(Files.readAllBytes(epPath), StandardCharsets.UTF_8).trim();
                    int persisted = Integer.parseInt(content);
                    EPISODE_COUNTER.set(persisted);
                    initialEpisodeCount = persisted;
                    logger.info("Loaded episode counter from file: " + persisted);
                }
            } catch (Exception e) {
                logger.warn("Failed to read episode counter, starting from 0", e);
            }
            boolean resetEpisodeCounter = EnvConfig.bool("RESET_EPISODE_COUNTER", false);
            if (resetEpisodeCounter) {
                EPISODE_COUNTER.set(0);
                initialEpisodeCount = 0;
                logger.info("Episode counter reset via RESET_EPISODE_COUNTER");
            }

            if (EPISODE_COUNTER.get() >= NUM_EPISODES) {
                logger.warn("No episodes to run: TOTAL_EPISODES=" + NUM_EPISODES
                        + " <= persisted counter=" + EPISODE_COUNTER.get()
                        + ". Increase TOTAL_EPISODES, set RESET_EPISODE_COUNTER=1, or delete "
                        + EPISODE_COUNT_PATH);
                return;
            }

            // Add shutdown hook to save episode counter on Ctrl+C
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    int finalCount = EPISODE_COUNTER.get();
                    Files.write(Paths.get(EPISODE_COUNT_PATH),
                            String.valueOf(finalCount).getBytes(StandardCharsets.UTF_8));
                    System.out.println("Shutdown hook: Saved episode counter = " + finalCount);
                } catch (IOException e) {
                    System.err.println("Shutdown hook: Failed to save episode counter: " + e.getMessage());
                }
            }, "Episode-Counter-Shutdown-Hook"));
            logger.info("Registered shutdown hook for episode counter persistence");

            // Log system resource utilization
            int cpuCores = Runtime.getRuntime().availableProcessors();
            long maxMemory = Runtime.getRuntime().maxMemory() / 1024 / 1024; // MB
            RLTrainer.threadLocalLogger.get().info("=== SYSTEM RESOURCES ===");
            RLTrainer.threadLocalLogger.get().info("CPU Cores Available: " + cpuCores);
            RLTrainer.threadLocalLogger.get().info("Max JVM Memory: " + maxMemory + " MB");
            RLTrainer.threadLocalLogger.get().info("Game Runners: " + NUM_GAME_RUNNERS + " (using " + (NUM_GAME_RUNNERS * 100.0 / cpuCores) + "% of CPU cores)");
            RLTrainer.threadLocalLogger.get().info("Episodes per runner: " + NUM_EPISODES_PER_GAME_RUNNER);
            RLTrainer.threadLocalLogger.get().info("Total episodes target: " + NUM_EPISODES);
            RLTrainer.threadLocalLogger.get().info("========================");

            // Record start time
            long startTime = System.nanoTime();
            final long startMs = System.currentTimeMillis();
            final int startEpisodeCountSnapshot = EPISODE_COUNTER.get();
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
                    int epNow = EPISODE_COUNTER.get();
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

            ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS, runnable -> {
                Thread thread = new Thread(runnable);
                thread.setPriority(Thread.MAX_PRIORITY - 1);
                return thread;
            });

            List<Future<Void>> futures = new ArrayList<>();
            final Object lock = new Object(); // Lock object for synchronization
            final boolean[] isFirstThread = {true}; // Flag to track the first thread

            for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
                Future<Void> future = executor.submit(() -> {
                    boolean isFirst;
                    synchronized (lock) {
                        isFirst = isFirstThread[0];
                        isFirstThread[0] = false;
                    }

                    Logger currentLogger = threadLocalLogger.get();
                    // All threads will now log at INFO level by default
                    currentLogger.info("Starting Game Runner");

                    Thread.currentThread().setName("GAME");
                    Random threadRand = new Random();

                    while (EPISODE_COUNTER.get() < NUM_EPISODES) {
                        int epNumber = EPISODE_COUNTER.incrementAndGet();
                        if (epNumber > NUM_EPISODES) {
                            break; // Another thread reached the target
                        }
                        metrics.recordEpisodeStarted();
                        long episodeStartNanos = System.nanoTime();
                        ACTIVE_EPISODES.incrementAndGet();
                        metrics.setActiveEpisodes(ACTIVE_EPISODES.get());
                        Path rlPlayerDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
                        Deck rlPlayerDeckThread = loadDeck(rlPlayerDeckPath.toString());
                        Path opponentDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
                        Deck opponentDeckThread = loadDeck(opponentDeckPath.toString());
                        if (rlPlayerDeckThread == null || opponentDeckThread == null) {
                            logger.warn("Train: failed to load deck(s) for game, skipping.");
                            ACTIVE_EPISODES.decrementAndGet();
                            metrics.setActiveEpisodes(ACTIVE_EPISODES.get());
                            continue;
                        }

                        // ------------------ Build a full Match so players have MatchPlayer objects ------------------
                        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("TwoPlayerMatch", "TwoPlayerMatch", false));

                        // Start an empty game so we can attach players
                        match.startGame();
                        Game game = match.getGames().get(0);

                        // Enable game logging if this episode should be logged
                        boolean enableGameLogging = GAME_LOG_FREQUENCY > 0
                                && (epNumber == 0 || epNumber % GAME_LOG_FREQUENCY == 0);
                        GameLogger gameLogger = GameLogger.create(enableGameLogging);
                        threadLocalGameLogger.set(gameLogger);

                        ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel);
                        rlPlayer.setCurrentEpisode(epNumber); // Set episode for mulligan logging
                        game.addPlayer(rlPlayer, rlPlayerDeckThread);
                        match.addPlayer(rlPlayer, rlPlayerDeckThread);

                        // Opponent selection
                        Player opponentPlayer = createTrainingOpponent(epNumber, threadRand);
                        game.addPlayer(opponentPlayer, opponentDeckThread);
                        match.addPlayer(opponentPlayer, opponentDeckThread);

                        String opponentTag = formatOpponentTag(opponentPlayer);
                        if (gameLogger.isEnabled()) {
                            gameLogger.log("OPPONENT: " + opponentTag + " (name=" + opponentPlayer.getName() + ")");
                        }

                        logger.info("Players added to game. RL player library size: " + rlPlayer.getLibrary().size() + ", Opponent library size: " + opponentPlayer.getLibrary().size());
                        System.out.println("DEBUG: Players added to game. RL player library size: " + rlPlayer.getLibrary().size() + ", Opponent library size: " + opponentPlayer.getLibrary().size());

                        game.loadCards(rlPlayerDeckThread.getCards(), rlPlayer.getId());
                        game.loadCards(opponentDeckThread.getCards(), opponentPlayer.getId());

                        GameOptions options = new GameOptions();
                        game.setGameOptions(options);

                        // Start health monitor to detect stuck/infinite-loop games
                        GameHealthMonitor healthMonitor = GameHealthMonitor.createAndStart(game);

                        long startGameNanos = System.nanoTime();
                        // Restart game now that players are added
                        game.start(rlPlayer.getId());
                        long endGameNanos = System.nanoTime();

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

                        // Train mulligan model from this game's decisions
                        trainMulliganModel(rlPlayer, rlPlayerWon, turns);
                        maybeSaveMulliganModel();

                        // Log head usage statistics
                        logHeadUsageStats(epNumber, rlPlayer, opponentPlayer, turns, rlPlayerWon);

                        logGameResult(game, rlPlayer);
                        long rewardStartNanos = System.nanoTime();
                        RewardDiag rewardDiag = updateModelBasedOnOutcome(game, rlPlayer, opponentPlayer);
                        double finalReward = rewardDiag.finalReward;
                        long rewardEndNanos = System.nanoTime();

                        // Defer statistics writing until after we compute episodeSeconds below
                        // -------- Episode duration & counter logging --------
                        long episodeDurationNanos = System.nanoTime() - episodeStartNanos;
                        double episodeSeconds = episodeDurationNanos / 1_000_000_000.0;
                        RLTrainer.threadLocalLogger.get().info(String.format("Episode %d completed in %.2f seconds", epNumber, episodeSeconds));

                        // Periodically save episode counter (every 100 episodes)
                        if (epNumber % 100 == 0) {
                            try {
                                Files.write(Paths.get(EPISODE_COUNT_PATH),
                                        String.valueOf(epNumber).getBytes(StandardCharsets.UTF_8));
                                logger.info("Episode counter saved: " + epNumber);
                            } catch (IOException e) {
                                logger.error("Failed to save episode counter at episode " + epNumber, e);
                            }
                        }
                        if (TRAIN_DIAG && (TRAIN_DIAG_EVERY <= 1 || epNumber % TRAIN_DIAG_EVERY == 0)) {
                            double totalMs = (System.nanoTime() - episodeStartNanos) / 1_000_000.0;
                            double gameMs = (endGameNanos - startGameNanos) / 1_000_000.0;
                            double rewardMs = (rewardEndNanos - rewardStartNanos) / 1_000_000.0;
                            int active = ACTIVE_EPISODES.get();
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
                                java.util.Map<String, Integer> mulliganStats = sharedModel.getMulliganModelTrainingStats();

                                logger.info(String.format(
                                        "Training progress: episode=%d/%d (run=%d, %.3f eps/s), ETA %ds, RL_activation_failures=%d, sim_training_skipped=%d, games_killed=%d",
                                        epNumber, targetEpisodeCount, done, epsPerSec, etaSec, rlFailures, simTrainSkipped, GameHealthMonitor.getGamesKilled()
                                ));
                                logger.info(String.format(
                                        "  Main model: %d train steps, %d samples | Mulligan model: %d train steps, %d samples",
                                        mainStats.get("train_steps"), mainStats.get("train_samples"),
                                        mulliganStats.get("train_steps"), mulliganStats.get("train_samples")
                                ));
                                // Value head quality metrics
                                logger.info(String.format(
                                        "  Value head: accuracy=%.1f%%, avg_win=%.3f, avg_loss=%.3f (target: +1/-1)",
                                        metrics.getValueAccuracy() * 100,
                                        metrics.getAverageValueForWins(),
                                        metrics.getAverageValueForLosses()
                                ));
                                logger.info(String.format(
                                        "  Reward diag (last ep): won=%s steps=%d finalReward=%.3f mc_return0=%.3f sum_rewards=%.3f last_reward=%.3f",
                                        rewardDiag.won, rewardDiag.steps, rewardDiag.finalReward,
                                        rewardDiag.mcReturn0, rewardDiag.sumRewards, rewardDiag.lastReward
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
                        } catch (Exception e) {
                            logger.warn("League tick failed at ep " + epNumber, e);
                        }

                        try {
                            Path statsPath = Paths.get(STATS_FILE_PATH);
                            if (statsPath.getParent() != null) {
                                Files.createDirectories(statsPath.getParent());
                            }
                            boolean writeHeader = !Files.exists(statsPath);
                            StringBuilder sb = new StringBuilder();
                            if (writeHeader) {
                                sb.append("episode,turns,final_reward,opponent_type,winrate,episode_seconds\n");
                            }
                            sb.append(epNumber).append(',').append(turns).append(',')
                                    .append(String.format("%.3f", finalReward)).append(',')
                                    .append(opponentType).append(',')
                                    .append(String.format("%.3f", snapshotWinrate)).append(',')
                                    .append(String.format("%.2f", episodeSeconds)).append('\n');
                            Files.write(statsPath, sb.toString().getBytes(StandardCharsets.UTF_8), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                        } catch (IOException e) {
                            logger.warn("Failed to write stats CSV", e);
                        }
                        metrics.recordEpisodeCompleted();
                        ACTIVE_EPISODES.decrementAndGet();
                        metrics.setActiveEpisodes(ACTIVE_EPISODES.get());
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

            // Record end time and log statistics
            long endTime = System.nanoTime();
            long totalTime = endTime - startTime;
            double totalTimeInMinutes = totalTime / 1_000_000_000.0 / 60.0;
            int episodesRun = EPISODE_COUNTER.get() - initialEpisodeCount;
            double gamesRunPerMinute = totalTimeInMinutes > 0 ? episodesRun / totalTimeInMinutes : 0;

            // Stop health stats logging
            healthStats.stop();

            logger.info("Training completed:");
            logger.info("Total Games Run: " + episodesRun);
            logger.info("Games Run Per Minute: " + gamesRunPerMinute);
            logger.info("Total Training Time: " + (totalTime / 1_000_000_000.0) + " seconds");
            logger.info("Health summary: " + healthStats.getSummary());

            // Save the trained model
            sharedModel.saveModel(MODEL_FILE_PATH);
            sharedModel.shutdown();
            if (heartbeat != null) {
                heartbeat.shutdownNow();
            }
            // ------------------ 2.  Persist updated episode counter ------------------
            try {
                Files.write(Paths.get(EPISODE_COUNT_PATH), String.valueOf(EPISODE_COUNTER.get()).getBytes(StandardCharsets.UTF_8));
            } catch (IOException e) {
                logger.error("Failed to persist episode counter", e);
            }

        } catch (IOException | InterruptedException e) {
            logger.error("Error during training", e);
        }
    }

    private static String formatOpponentTag(Player opponentPlayer) {
        if (opponentPlayer instanceof ComputerPlayerRL) {
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
            return (skill > 0) ? ("CP7-Skill " + skill) : "CP7";
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
        try {
            deckFiles = loadDeckPool();
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

        ExecutorService executor = Executors.newFixedThreadPool(NUM_GAME_RUNNERS);
        final Object lock = new Object();
        final boolean[] isFirstThread = {true};

        List<Future<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < NUM_GAME_RUNNERS; i++) {
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
                Random threadRand = new Random();

                for (int evalEpisode = 0; evalEpisode < numEpisodesPerThread; evalEpisode++) {
                    Path rlPlayerDeckPath = deckFiles.get(threadRand.nextInt(deckFiles.size()));
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

                    // Greedy evaluation: use deterministic arg-max player
                    ComputerPlayerRL rlPlayer = new ComputerPlayerRL("PlayerRL1", RangeOfInfluence.ALL, sharedModel, true);
                    rlPlayer.setCurrentEpisode(-currentEvalGame); // Negative for eval = deterministic mulligan
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
        logger.info(String.format("Evaluation win rate: %.4f, RL_activation_failures=%d", winRate, rlFailures));
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

    private double runLeagueBenchmarkPolicy(
            String rlPolicyKey,
            LeagueOpponentSpec opponent,
            String deckListFileOverride,
            int gamesPerMatchup
    ) {
        try {
            mage.cards.repository.CardScanner.scan();
            List<Path> decks = loadDeckPoolWithOverride(deckListFileOverride);
            if (decks.size() < 2) {
                logger.warn("League benchmark requires at least 2 decks; found " + decks.size());
                return 0.0;
            }

            final int benchThreads = LEAGUE_BENCHMARK_THREADS;
            final int logEvery = Math.max(1, LEAGUE_BENCHMARK_LOG_EVERY);
            final int heartbeatSec = LEAGUE_BENCHMARK_HEARTBEAT_SEC;
            final int gameTimeoutSec = LEAGUE_BENCHMARK_GAME_TIMEOUT_SEC;
            final int totalPlannedGames = decks.size() * (decks.size() - 1) * Math.max(1, gamesPerMatchup);

            final AtomicLong completed = new AtomicLong(0);
            final AtomicLong started = new AtomicLong(0);
            final AtomicLong winsTotal = new AtomicLong(0);
            final long startMs = System.currentTimeMillis();

            ExecutorService exec = Executors.newFixedThreadPool(benchThreads, r -> {
                Thread t = new Thread(r);
                t.setName("GAME-BENCH");
                t.setPriority(Thread.NORM_PRIORITY);
                return t;
            });

            logger.info(String.format(
                    "League benchmark started: rlPolicy=%s vs %s, decks=%d, gamesPerMatchup=%d, plannedGames=%d, threads=%d",
                    rlPolicyKey,
                    opponent.kind == LeagueOpponentKind.BOT ? ("CP7Skill" + opponent.botSkill) : opponent.policyKey,
                    decks.size(),
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
            for (int i = 0; i < decks.size(); i++) {
                for (int j = 0; j < decks.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    final Path p1 = decks.get(i);
                    final Path p2 = decks.get(j);
                    for (int g = 0; g < gamesPerMatchup; g++) {
                        futures.add(exec.submit(() -> {
                            Thread.currentThread().setName("GAME-BENCH");
                            long s = started.incrementAndGet();
                            if (s % logEvery == 0 || s == totalPlannedGames) {
                                logger.info(String.format("League benchmark started games: %d/%d", s, totalPlannedGames));
                            }

                            boolean win = runSingleLeagueBenchmarkGame(
                                    p1, p2, gameTimeoutSec,
                                    rlPolicyKey, opponent
                            );
                            if (win) {
                                winsTotal.incrementAndGet();
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
                    "League benchmark done: winrate=%.3f (%d/%d)",
                    overall, totalWins, totalGames
            ));
            return overall;
        } catch (Exception e) {
            logger.error("League benchmark failed", e);
            return 0.0;
        }
    }

    private boolean runSingleLeagueBenchmarkGame(
            Path rlDeckPath,
            Path oppDeckPath,
            int gameTimeoutSec,
            String rlPolicyKey,
            LeagueOpponentSpec opponent
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

        GameLogger gameLogger = GameLogger.create(LEAGUE_EVAL_GAME_LOGGING);
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

        Player opp;
        if (opponent.kind == LeagueOpponentKind.BOT) {
            int skill = Math.max(1, opponent.botSkill);
            opp = new ComputerPlayer7("Benchmark-skill" + skill, RangeOfInfluence.ALL, skill);
        } else {
            String pk = opponent.policyKey == null ? "train" : opponent.policyKey;
            opp = new ComputerPlayerRL("SnapshotOpp", RangeOfInfluence.ALL, sharedModel, true, false, pk);
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

        game.start(rlPlayer.getId());
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

        game.start(rlPlayer.getId());
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

    public static List<Path> loadDeckPool() throws IOException {
        // If explicit list is provided, use it
        if (DECK_LIST_FILE != null && !DECK_LIST_FILE.trim().isEmpty()) {
            Path listPath = Paths.get(DECK_LIST_FILE);
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
    private static void logHeadUsageStats(int episodeNum, ComputerPlayerRL rlPlayer, Player opponentPlayer, int turns, boolean rlPlayerWon) {
        try {
            Path logPath = Paths.get(RLLogPaths.HEAD_USAGE_LOG_PATH);
            Files.createDirectories(logPath.getParent());

            // Initialize with header if file doesn't exist
            if (!Files.exists(logPath)) {
                String header = "episode,turns,won,opponent_type," +
                        "rl_total,rl_action_head,rl_target_head,rl_card_select_head," +
                        "opp_total,opp_action_head,opp_target_head,opp_card_select_head\n";
                Files.write(logPath, header.getBytes(StandardCharsets.UTF_8));
            }

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

            // Build CSV line
            String line = String.format("%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    episodeNum, turns, rlPlayerWon ? 1 : 0, opponentType,
                    rlTotal, rlActionHead, rlTargetHead, rlCardSelectHead,
                    oppTotal, oppActionHead, oppTargetHead, oppCardSelectHead);

            Files.write(logPath, line.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);

        } catch (IOException e) {
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
        try {
            StringBuilder importWarnings = new StringBuilder();
            DeckCardLists deckCardLists = DeckImporter.importDeckFromFile(filePath, importWarnings, false);

            if (importWarnings.length() > 0) {
                // Most common reason for <60 cards: DeckImporter couldn't find some card names in this XMage build,
                // so those entries are dropped during import.
                logger.warn("Deck import warnings for " + filePath + ":\n" + importWarnings);
            }

            Deck deck = Deck.load(deckCardLists, false, false, null);
            if (deck != null) {
                int mainCount = deck.getCards().size();
                int sideCount = deck.getSideboard().size();
                if (mainCount != 60) {
                    logger.warn("Deck mainboard size is " + mainCount + " (expected 60) for: " + filePath
                            + " (sideboard=" + sideCount + ")");
                }
            }
            return deck;
        } catch (GameException e) {
            logger.error("Error loading deck: " + filePath, e);
            return null;
        }
    }

    private RewardDiag updateModelBasedOnOutcome(Game game, ComputerPlayerRL rlPlayer, Player opponentPlayer) {
        boolean rlPlayerWon = game.getWinner().contains(rlPlayer.getName());

        // ------------------------------------------------------------------
        // 1.  Terminal win / loss reward (ground-truth)
        // ------------------------------------------------------------------
        double finalReward = rlPlayerWon ? 1.0 : -1.0;

        if (Double.isNaN(finalReward) || Double.isInfinite(finalReward)) {
            finalReward = rlPlayerWon ? 1.0 : -1.0;
            logger.warn("Reward was NaN/Inf – reverted to ±1 fallback");
        }

        // Get training data for RL player
        List<StateSequenceBuilder.TrainingData> rlPlayerTrainingData = rlPlayer.getTrainingBuffer();

        List<StateSequenceBuilder.TrainingData> opponentTrainingData = new ArrayList<>();
        if (opponentPlayer instanceof ComputerPlayerRL) {
            opponentTrainingData = ((ComputerPlayerRL) opponentPlayer).getTrainingBuffer();
        }

        // --- Calculate immediate rewards for each step (GAE will compute advantages in Python) ---
        List<Double> rlPlayerRewards = calculateImmediateRewards(rlPlayerTrainingData, finalReward);
        List<Double> opponentRewards = calculateImmediateRewards(opponentTrainingData, -finalReward); // Opposite reward for opponent

        // Update the model with all states and immediate rewards (Python will apply GAE)
        if (!rlPlayerTrainingData.isEmpty()) {
            sharedModel.enqueueTraining(rlPlayerTrainingData, rlPlayerRewards);
        }
        if (!opponentTrainingData.isEmpty()) {
            sharedModel.enqueueTraining(opponentTrainingData, opponentRewards); // Opposite reward for opponent
        }

        // Record value head prediction for auto-GAE tracking
        float lastValue = rlPlayer.getLastValueScore();
        metrics.recordValuePrediction(lastValue, rlPlayerWon);
        sharedModel.recordGameResult(lastValue, rlPlayerWon);

        RewardDiag diag = new RewardDiag();
        diag.won = rlPlayerWon;
        diag.finalReward = finalReward;
        diag.steps = rlPlayerRewards.size();
        diag.sumRewards = rlPlayerRewards.stream().mapToDouble(d -> d).sum();
        diag.lastReward = rlPlayerRewards.isEmpty() ? 0.0 : rlPlayerRewards.get(rlPlayerRewards.size() - 1);
        diag.mcReturn0 = computeDiscountedReturn0(rlPlayerRewards, 0.99);
        return diag;
    }

    private static final class RewardDiag {

        boolean won;
        int steps;
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
    private static double[] recordGameOutcome(boolean rlPlayerWon) {
        // Add to queue
        recentWins.add(rlPlayerWon);
        if (rlPlayerWon) {
            winCount.incrementAndGet();
        }

        // Track games at current difficulty level
        gamesAtCurrentLevel.incrementAndGet();

        // Trim queue to window size
        while (recentWins.size() > WINRATE_WINDOW) {
            Boolean removed = recentWins.poll();
            if (removed != null && removed) {
                winCount.decrementAndGet();
            }
        }

        // Return snapshot at this exact moment (avoids race condition in logging)
        int size = recentWins.size();
        double winrate = size == 0 ? 0.0 : winCount.get() / (double) size;
        return new double[]{winrate, size};
    }

    /**
     * Calculates current rolling winrate. Thread-safe for concurrent training.
     */
    private static double getCurrentWinrate() {
        int size = recentWins.size();
        if (size == 0) {
            return 0.0;
        }
        return winCount.get() / (double) size;
    }

    /**
     * Train mulligan model from a completed game.
     *
     * @param rlPlayer The RL player whose mulligans to train on
     * @param won Whether the player won the game
     */
    private static void trainMulliganModel(ComputerPlayerRL rlPlayer, boolean won, int gameTurns) {
        try {
            List<float[]> features = rlPlayer.getMulliganFeatures();
            List<Float> decisions = rlPlayer.getMulliganDecisions();

            if (features.isEmpty()) {
                return; // No mulligan decisions this game
            }

            int batchSize = features.size();
            float outcome = won ? 1.0f : 0.0f;
            int featureSize = features.get(0).length;

            // Pack pre-built feature vectors directly
            java.nio.ByteBuffer featuresBuf = java.nio.ByteBuffer.allocate(batchSize * featureSize * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (float[] feat : features) {
                for (float f : feat) {
                    featuresBuf.putFloat(f);
                }
            }

            // Pack decisions
            java.nio.ByteBuffer decisionsBuf = java.nio.ByteBuffer.allocate(batchSize * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (Float decision : decisions) {
                decisionsBuf.putFloat(decision);
            }

            // Pack outcomes (same outcome for all decisions in this game)
            java.nio.ByteBuffer outcomesBuf = java.nio.ByteBuffer.allocate(batchSize * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < batchSize; i++) {
                outcomesBuf.putFloat(outcome);
            }

            // Pack game lengths for survival-based reward shaping (same for all decisions)
            java.nio.ByteBuffer gameLengthsBuf = java.nio.ByteBuffer.allocate(batchSize * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < batchSize; i++) {
                gameLengthsBuf.putInt(Math.max(1, gameTurns));
            }

            long keepN = decisions.stream().filter(d -> d != null && d > 0.5f).count();
            logger.info(String.format("MULLIGAN TRAIN batch=%d keepN=%d mullN=%d turns=%d won=%s", batchSize, keepN, batchSize - keepN, gameTurns, won));

            // Train the model
            sharedModel.trainMulligan(
                    featuresBuf.array(),
                    decisionsBuf.array(),
                    outcomesBuf.array(),
                    gameLengthsBuf.array(),
                    batchSize
            );

            // Clear player's mulligan buffer
            rlPlayer.clearMulliganData();

        } catch (Exception e) {
            logger.error("Error training mulligan model: " + e.getMessage(), e);
        }
    }

    // Mulligan model save counter
    private static final AtomicInteger mulliganTrainCount = new AtomicInteger(0);
    private static final int MULLIGAN_SAVE_INTERVAL = EnvConfig.i32("MULLIGAN_SAVE_INTERVAL", 100);

    /**
     * Periodically save mulligan model.
     */
    private static void maybeSaveMulliganModel() {
        if (mulliganTrainCount.incrementAndGet() % MULLIGAN_SAVE_INTERVAL == 0) {
            try {
                sharedModel.saveMulliganModel();
                logger.info("Mulligan model saved (after " + mulliganTrainCount.get() + " training updates)");
            } catch (Exception e) {
                logger.error("Failed to save mulligan model: " + e.getMessage(), e);
            }
        }
    }

    private Player createTrainingOpponent(int episodeNum, Random rand) {
        String mode = OPPONENT_SAMPLER == null ? "league" : OPPONENT_SAMPLER.trim().toLowerCase();
        switch (mode) {
            case "adaptive":
                return createAdaptiveOpponent(episodeNum, rand);
            case "fixed":
                // Force fixed schedule by temporarily disabling adaptive logic.
                // (This preserves existing behavior with FIXED_* env vars.)
                return createFixedOpponent(episodeNum, rand);
            case "league":
            default:
                return createLeagueOpponent(episodeNum, rand);
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
            return new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel);
        }
    }

    private Player createLeagueOpponent(int episodeNum, Random rand) {
        LeagueState st = getLeagueState();

        // Stage 0 (bootstrap): train against CP7Skill1 only until promoted.
        if (!st.promoted) {
            int[] skills = parseSkillList(LEAGUE_HEURISTIC_SKILLS_PRE, new int[]{1});
            int skill = pickFrom(skills, rand, 1);
            lastOpponentType = "H-CP7(skill=" + skill + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        }

        // After promotion: mixed league training (H/S/C)
        double pH = clamp01(LEAGUE_P_H);
        double pS = clamp01(LEAGUE_P_S);
        double pC = clamp01(LEAGUE_P_C);
        double sum = pH + pS + pC;
        if (sum <= 0) {
            pH = 0.30;
            pS = 0.50;
            pC = 0.20;
            sum = 1.0;
        }
        pH /= sum;
        pS /= sum;
        pC /= sum;

        double r = rand.nextDouble();
        if (r < pH) {
            int[] skills = parseSkillList(LEAGUE_HEURISTIC_SKILLS_POST, new int[]{1, 2, 3});
            int skill = pickFrom(skills, rand, 1);
            lastOpponentType = "H-CP7(skill=" + skill + ")";
            return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
        } else if (r < pH + pS) {
            String snapKey = pickLeagueSnapshotPolicyKey(st, rand);
            if (snapKey == null) {
                int[] skills = parseSkillList(LEAGUE_HEURISTIC_SKILLS_POST, new int[]{1, 2, 3});
                int skill = pickFrom(skills, rand, 1);
                lastOpponentType = "H-CP7(skill=" + skill + ")";
                return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
            }
            lastOpponentType = "S-" + snapKey;
            return new ComputerPlayerRL("SnapshotOpp", RangeOfInfluence.ALL, sharedModel, false, false, snapKey);
        } else {
            // Self-play bucket: C vs C. Opponent is off-policy from the learner's perspective.
            lastOpponentType = "SELFPLAY";
            return new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel, false, false, "train");
        }
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
                if (v > 0) {
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

            switch (newLevel) {
                case WEAK:
                    opType = "WEAK-CP7(skill=7)";
                    opponent = new ComputerPlayer7("WeakBot", RangeOfInfluence.ALL, 7);
                    break;

                case MEDIUM:
                    opType = "MEDIUM-CP7(skill=8)";
                    opponent = new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 8);
                    break;

                case STRONG:
                    opType = "STRONG-CP7(skill=9)";
                    opponent = new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 9);
                    break;

                case SELFPLAY:
                    // Mostly self-play with occasional strong heuristic for stability
                    if (rand.nextDouble() < 0.9) {
                        opType = "SELFPLAY";
                        opponent = new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel);
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
                        ? new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel)
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
