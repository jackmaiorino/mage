package mage.player.ai.rl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Logger for detailed game analysis. Writes each game to a separate file.
 * Captures game state, candidate actions, model scores, and decisions.
 */
public class GameLogger {
    
    private static final AtomicInteger gameCounter = new AtomicInteger(0);
    private static final String LOG_DIR = System.getenv().getOrDefault(
        "GAME_LOG_DIR", 
        "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/gamelogs"
    );
    
    private final BufferedWriter writer;
    private final String gameId;
    private final Path logFile;
    private int decisionNumber = 0;
    private final boolean enabled;
    
    private GameLogger(String gameId, Path logFile, BufferedWriter writer, boolean enabled) {
        this.gameId = gameId;
        this.logFile = logFile;
        this.writer = writer;
        this.enabled = enabled;
    }
    
    /**
     * Create a new game logger. Returns null if logging is disabled.
     */
    public static GameLogger create(boolean enableLogging) {
        if (!enableLogging) {
            return createNoOp();
        }
        
        try {
            // Create log directory if it doesn't exist
            Path logDir = Paths.get(LOG_DIR);
            Files.createDirectories(logDir);
            
            // Generate unique game ID
            int gameNum = gameCounter.incrementAndGet();
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String gameId = String.format("game_%s_%04d", timestamp, gameNum);
            
            // Create log file
            Path logFile = logDir.resolve(gameId + ".txt");
            BufferedWriter writer = Files.newBufferedWriter(
                logFile, 
                StandardOpenOption.CREATE, 
                StandardOpenOption.TRUNCATE_EXISTING
            );
            
            GameLogger logger = new GameLogger(gameId, logFile, writer, true);
            logger.writeHeader();
            return logger;
            
        } catch (IOException e) {
            System.err.println("Failed to create game logger: " + e.getMessage());
            return createNoOp();
        }
    }
    
    /**
     * Create a no-op logger that does nothing.
     */
    private static GameLogger createNoOp() {
        return new GameLogger("noop", null, null, false);
    }
    
    private void writeHeader() {
        if (!enabled) return;
        try {
            writer.write("=" .repeat(80));
            writer.newLine();
            writer.write(String.format("GAME LOG: %s", gameId));
            writer.newLine();
            writer.write(String.format("Started: %s", LocalDateTime.now()));
            writer.newLine();
            writer.write("=".repeat(80));
            writer.newLine();
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error writing header: " + e.getMessage());
        }
    }
    
    /**
     * Log a decision point with game state, options, and scores.
     */
    public void logDecision(
        String playerName,
        String phase,
        int turn,
        String gameState,
        List<String> options,
        float[] actionProbs,
        float valueScore,
        String selectedAction
    ) {
        if (!enabled) return;
        
        try {
            decisionNumber++;
            
            writer.write("-".repeat(80));
            writer.newLine();
            writer.write(String.format("DECISION #%d - Turn %d, %s - %s", 
                decisionNumber, turn, phase, playerName));
            writer.newLine();
            writer.write("-".repeat(80));
            writer.newLine();
            writer.newLine();
            
            // Game state
            writer.write("GAME STATE:");
            writer.newLine();
            writer.write(gameState);
            writer.newLine();
            writer.newLine();
            
            // Options and scores
            writer.write("OPTIONS & SCORES:");
            writer.newLine();
            for (int i = 0; i < options.size() && i < actionProbs.length; i++) {
                String marker = options.get(i).equals(selectedAction) ? " >>> " : "     ";
                writer.write(String.format("%s[%d] %.6f - %s", 
                    marker, i, actionProbs[i], options.get(i)));
                writer.newLine();
            }
            writer.newLine();
            
            writer.write(String.format("VALUE SCORE: %.6f", valueScore));
            writer.newLine();
            writer.write(String.format("SELECTED: %s", selectedAction));
            writer.newLine();
            writer.newLine();
            
            writer.flush();
            
        } catch (IOException e) {
            System.err.println("Error logging decision: " + e.getMessage());
        }
    }
    
    /**
     * Log game outcome.
     */
    public void logOutcome(String winner, String loser, int turns, String reason) {
        if (!enabled) return;
        
        try {
            writer.write("=".repeat(80));
            writer.newLine();
            writer.write("GAME OUTCOME");
            writer.newLine();
            writer.write("=".repeat(80));
            writer.newLine();
            writer.write(String.format("Winner: %s", winner));
            writer.newLine();
            writer.write(String.format("Loser: %s", loser));
            writer.newLine();
            writer.write(String.format("Turns: %d", turns));
            writer.newLine();
            writer.write(String.format("Reason: %s", reason));
            writer.newLine();
            writer.write(String.format("Total Decisions Logged: %d", decisionNumber));
            writer.newLine();
            writer.write(String.format("Ended: %s", LocalDateTime.now()));
            writer.newLine();
            writer.write("=".repeat(80));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging outcome: " + e.getMessage());
        }
    }
    
    /**
     * Log an action taken by any player (e.g., opponent actions).
     * Simpler than logDecision - just records what happened.
     */
    public void logAction(String playerName, String phase, int turn, String action) {
        if (!enabled) return;
        
        try {
            writer.write(String.format("[Turn %d, %s] %s: %s", 
                turn, phase, playerName, action));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging action: " + e.getMessage());
        }
    }
    
    /**
     * Log turn start with full game state summary (shows what happened since last turn).
     */
    public void logTurnStart(int turn, String activePlayer, String fullGameState) {
        if (!enabled) return;
        
        try {
            writer.newLine();
            writer.write("*".repeat(80));
            writer.newLine();
            writer.write(String.format("*** TURN %d START - Active Player: %s ***", turn, activePlayer));
            writer.newLine();
            writer.write("*".repeat(80));
            writer.newLine();
            writer.write(fullGameState);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging turn start: " + e.getMessage());
        }
    }
    
    /**
     * Log arbitrary text.
     */
    public void log(String message) {
        if (!enabled) return;
        
        try {
            writer.write(message);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("Error logging message: " + e.getMessage());
        }
    }
    
    /**
     * Close the logger.
     */
    public void close() {
        if (!enabled || writer == null) return;
        
        try {
            writer.close();
        } catch (IOException e) {
            System.err.println("Error closing game logger: " + e.getMessage());
        }
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public String getGameId() {
        return gameId;
    }
    
    public Path getLogFile() {
        return logFile;
    }
}
