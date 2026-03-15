package mage.player.ai.rl;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Instance-based path configuration for a single training profile.
 * Mirrors the fields in {@link RLLogPaths} but can be constructed per-profile
 * so multiple profiles can coexist in the same JVM.
 */
public final class ProfilePaths {

    public final String profileName;
    public final String modelsBaseDir;
    public final String logsBaseDir;

    // Model artifacts
    public final String modelFilePath;
    public final String episodeCountPath;
    public final String mulliganEpisodeCountPath;
    public final String mulliganModelPath;
    public final String snapshotDir;

    // Health monitoring
    public final String healthLogPath;
    public final String gameKillsLogPath;
    public final String activationFailuresLogPath;
    public final String headUsageLogPath;
    public final String trainingLossesPath;

    // Episode statistics
    public final String trainingStatsPath;
    public final String evaluationStatsPath;
    public final String mulliganStatsPath;

    // Game logs
    public final String trainingGameLogsDir;
    public final String evalGameLogsDir;

    // League training
    public final String leagueEventsLogPath;
    public final String leagueStatusPath;
    public final String leagueStatePath;

    // Python logs
    public final String pythonLogsDir;
    public final String pythonMainLogPath;
    public final String pythonMulliganLogPath;
    public final String pythonVramDiagnosticsLogPath;
    public final String pythonMulliganTracePath;

    public ProfilePaths(String profileName, String artifactsRoot) {
        this.profileName = profileName;
        String profileRoot = artifactsRoot + "/profiles/" + profileName;
        this.modelsBaseDir = profileRoot + "/models";
        this.logsBaseDir = profileRoot + "/logs";

        // Model artifacts
        this.modelFilePath = modelsBaseDir + "/model.pt";
        this.episodeCountPath = modelsBaseDir + "/episodes.txt";
        this.mulliganEpisodeCountPath = modelsBaseDir + "/mulligan_episodes.txt";
        this.mulliganModelPath = modelsBaseDir + "/mulligan_model.pt";
        this.snapshotDir = modelsBaseDir + "/snapshots";

        // Health monitoring
        this.healthLogPath = logsBaseDir + "/health/training_health.csv";
        this.gameKillsLogPath = logsBaseDir + "/health/game_kills.log";
        this.activationFailuresLogPath = logsBaseDir + "/health/activation_failures.log";
        this.headUsageLogPath = logsBaseDir + "/stats/head_usage.csv";
        this.trainingLossesPath = logsBaseDir + "/stats/training_losses.csv";

        // Episode statistics
        this.trainingStatsPath = logsBaseDir + "/stats/training_stats.csv";
        this.evaluationStatsPath = logsBaseDir + "/stats/evaluation_stats.csv";
        this.mulliganStatsPath = logsBaseDir + "/stats/mulligan_stats.csv";

        // Game logs
        this.trainingGameLogsDir = logsBaseDir + "/games/training";
        this.evalGameLogsDir = logsBaseDir + "/games/evaluation";

        // League training
        this.leagueEventsLogPath = logsBaseDir + "/league/events.log";
        this.leagueStatusPath = logsBaseDir + "/league/status.txt";
        this.leagueStatePath = logsBaseDir + "/league/league_state.json";

        // Python logs
        this.pythonLogsDir = logsBaseDir + "/python";
        this.pythonMainLogPath = pythonLogsDir + "/mtg_ai.log";
        this.pythonMulliganLogPath = pythonLogsDir + "/mulligan_training.log";
        this.pythonVramDiagnosticsLogPath = pythonLogsDir + "/VRAM_diagnostics.log";
        this.pythonMulliganTracePath = pythonLogsDir + "/mulligan_trace.jsonl";
    }

    public Path agentStatusPath() {
        return Paths.get(logsBaseDir, "league", "agent_status.json");
    }

    public Path snapshotsPath() {
        return Paths.get(snapshotDir);
    }

    public Path modelsPath() {
        return Paths.get(modelsBaseDir);
    }
}
