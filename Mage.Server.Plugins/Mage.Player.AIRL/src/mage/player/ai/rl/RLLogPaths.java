package mage.player.ai.rl;

/**
 * Centralized path configuration for RL training artifacts (models + logs).
 *
 * When MODEL_PROFILE is set, all paths are rooted under:
 *   rl/profiles/<profile>/models/
 *   rl/profiles/<profile>/logs/
 *
 * Without MODEL_PROFILE the legacy flat layout is used (backward compat):
 *   rl/models/
 *   rl/logs/
 *
 * Directory structure (per profile):
 * models/
 * ├── model.pt
 * ├── model_latest.pt
 * ├── mulligan_model.pt
 * ├── episodes.txt
 * ├── mulligan_episodes.txt
 * └── snapshots/
 * logs/
 * ├── health/
 * │   ├── training_health.csv
 * │   ├── game_kills.log
 * │   └── activation_failures.log
 * ├── stats/
 * │   ├── training_stats.csv
 * │   ├── evaluation_stats.csv
 * │   ├── mulligan_stats.csv
 * │   ├── head_usage.csv
 * │   └── training_losses.csv
 * ├── games/
 * │   ├── training/        (max 50 files)
 * │   └── evaluation/
 * └── league/
 *     ├── events.log
 *     ├── status.txt
 *     └── league_state.json
 */
public final class RLLogPaths {

    private RLLogPaths() {
    }

    private static final String RL_BASE =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl";

    private static final String PROFILE_NAME = EnvConfig.str("MODEL_PROFILE", "");

    public static final String MODELS_BASE_DIR;
    public static final String LOGS_BASE_DIR;

    static {
        if (!PROFILE_NAME.isEmpty()) {
            String profileRoot = RL_BASE + "/profiles/" + PROFILE_NAME;
            MODELS_BASE_DIR = EnvConfig.str("RL_MODELS_DIR", profileRoot + "/models");
            LOGS_BASE_DIR   = EnvConfig.str("RL_LOGS_DIR",   profileRoot + "/logs");
        } else {
            MODELS_BASE_DIR = EnvConfig.str("RL_MODELS_DIR", RL_BASE + "/models");
            LOGS_BASE_DIR   = EnvConfig.str("RL_LOGS_DIR",   RL_BASE + "/logs");
        }
    }

    // Model artifacts
    public static final String MODEL_FILE_PATH = EnvConfig.str("MTG_MODEL_PATH",
            EnvConfig.str("MODEL_PATH", MODELS_BASE_DIR + "/model.pt"));

    public static final String EPISODE_COUNT_PATH = EnvConfig.str("EPISODE_COUNTER_PATH",
            MODELS_BASE_DIR + "/episodes.txt");

    public static final String MULLIGAN_EPISODE_COUNT_PATH = EnvConfig.str("MULLIGAN_EPISODE_COUNTER_PATH",
            MODELS_BASE_DIR + "/mulligan_episodes.txt");

    public static final String MULLIGAN_MODEL_PATH = EnvConfig.str("MULLIGAN_MODEL_PATH",
            MODELS_BASE_DIR + "/mulligan_model.pt");

    public static final String SNAPSHOT_DIR = EnvConfig.str("SNAPSHOT_DIR",
            MODELS_BASE_DIR + "/snapshots");

    // Health monitoring
    public static final String HEALTH_LOG_PATH = EnvConfig.str("HEALTH_LOG_PATH",
            LOGS_BASE_DIR + "/health/training_health.csv");

    public static final String GAME_KILLS_LOG_PATH = EnvConfig.str("GAME_KILLS_LOG_PATH",
            LOGS_BASE_DIR + "/health/game_kills.log");

    public static final String ACTIVATION_FAILURES_LOG_PATH = EnvConfig.str("ACTIVATION_FAILURES_LOG_PATH",
            LOGS_BASE_DIR + "/health/activation_failures.log");

    public static final String HEAD_USAGE_LOG_PATH = EnvConfig.str("HEAD_USAGE_LOG_PATH",
            LOGS_BASE_DIR + "/stats/head_usage.csv");

    public static final String TRAINING_LOSSES_PATH = EnvConfig.str("TRAINING_LOSSES_PATH",
            LOGS_BASE_DIR + "/stats/training_losses.csv");

    // Episode statistics
    public static final String TRAINING_STATS_PATH = EnvConfig.str("STATS_PATH",
            LOGS_BASE_DIR + "/stats/training_stats.csv");

    public static final String EVALUATION_STATS_PATH = EnvConfig.str("EVAL_STATS_PATH",
            LOGS_BASE_DIR + "/stats/evaluation_stats.csv");

    public static final String MULLIGAN_STATS_PATH = EnvConfig.str("MULLIGAN_STATS_PATH",
            LOGS_BASE_DIR + "/stats/mulligan_stats.csv");

    // Game logs
    public static final String TRAINING_GAME_LOGS_DIR = EnvConfig.str("TRAINING_GAME_LOGS_DIR",
            LOGS_BASE_DIR + "/games/training");

    public static final String EVAL_GAME_LOGS_DIR = EnvConfig.str("EVAL_GAME_LOGS_DIR",
            LOGS_BASE_DIR + "/games/evaluation");

    // League training
    public static final String LEAGUE_EVENTS_LOG_PATH = EnvConfig.str("LEAGUE_EVENTS_LOG_PATH",
            LOGS_BASE_DIR + "/league/events.log");

    public static final String LEAGUE_STATUS_PATH = EnvConfig.str("LEAGUE_STATUS_PATH",
            LOGS_BASE_DIR + "/league/status.txt");

    public static final String LEAGUE_STATE_PATH = EnvConfig.str("LEAGUE_STATE_PATH",
            LOGS_BASE_DIR + "/league/league_state.json");

    // Python logs (kept in MLPythonCode for proximity to Python scripts)
    public static final String PYTHON_MULLIGAN_TRACE_PATH
            = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mulligan_trace.jsonl";

    // -----------------------------------------------------------------------
    // Draft model paths (DRAFT_MODEL_PROFILE, defaults to MODEL_PROFILE + "-Draft")
    // -----------------------------------------------------------------------
    private static final String DRAFT_PROFILE_NAME;
    public static final String DRAFT_MODELS_BASE_DIR;
    public static final String DRAFT_LOGS_BASE_DIR;

    static {
        String dp = EnvConfig.str("DRAFT_MODEL_PROFILE", "");
        if (dp.isEmpty()) {
            dp = PROFILE_NAME.isEmpty() ? "VintageCube-Draft" : PROFILE_NAME + "-Draft";
        }
        DRAFT_PROFILE_NAME = dp;
        String draftRoot = RL_BASE + "/profiles/" + DRAFT_PROFILE_NAME;
        DRAFT_MODELS_BASE_DIR = EnvConfig.str("DRAFT_MODELS_DIR", draftRoot + "/models");
        DRAFT_LOGS_BASE_DIR   = EnvConfig.str("DRAFT_LOGS_DIR",   draftRoot + "/logs");
    }

    public static final String DRAFT_MODEL_FILE_PATH = EnvConfig.str("DRAFT_MODEL_PATH",
            DRAFT_MODELS_BASE_DIR + "/draft_model.pt");

    public static final String DRAFT_EPISODE_COUNT_PATH = EnvConfig.str("DRAFT_EPISODE_COUNTER_PATH",
            DRAFT_MODELS_BASE_DIR + "/draft_episodes.txt");

    public static final String DRAFT_GAME_LOGS_DIR = EnvConfig.str("DRAFT_GAME_LOGS_DIR",
            DRAFT_LOGS_BASE_DIR + "/games/drafts");

    public static final String DRAFT_STATS_PATH = EnvConfig.str("DRAFT_STATS_PATH",
            DRAFT_LOGS_BASE_DIR + "/stats/draft_stats.csv");

    public static final String DRAFT_LOSSES_PATH = EnvConfig.str("DRAFT_LOSSES_PATH",
            DRAFT_LOGS_BASE_DIR + "/stats/draft_losses.csv");
}
