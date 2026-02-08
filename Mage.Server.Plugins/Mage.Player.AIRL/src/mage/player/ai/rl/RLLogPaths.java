package mage.player.ai.rl;

/**
 * Centralized log path configuration for RL training.
 *
 * Directory structure:
 * logs/
 * ├── health/              # Training health monitoring
 * │   ├── training_health.csv
 * │   └── game_kills.log
 * ├── stats/               # Episode-level statistics
 * │   ├── training_stats.csv
 * │   ├── evaluation_stats.csv
 * │   ├── mulligan_stats.csv
 * │   └── head_usage.csv
 * ├── games/               # Detailed game logs
 * │   ├── training/        (max 50 files)
 * │   └── evaluation/
 * └── league/              # League training logs
 *     ├── events.log
 *     ├── status.txt
 *     └── league_state.json
 */
public final class RLLogPaths {

    private RLLogPaths() {
    }

    // Base directory for all RL logs
    public static final String LOGS_BASE_DIR = EnvConfig.str("RL_LOGS_DIR",
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/logs");

    // Health monitoring
    public static final String HEALTH_LOG_PATH = EnvConfig.str("HEALTH_LOG_PATH",
            LOGS_BASE_DIR + "/health/training_health.csv");

    public static final String GAME_KILLS_LOG_PATH = EnvConfig.str("GAME_KILLS_LOG_PATH",
            LOGS_BASE_DIR + "/health/game_kills.log");

    public static final String HEAD_USAGE_LOG_PATH = EnvConfig.str("HEAD_USAGE_LOG_PATH",
            LOGS_BASE_DIR + "/stats/head_usage.csv");

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
}
