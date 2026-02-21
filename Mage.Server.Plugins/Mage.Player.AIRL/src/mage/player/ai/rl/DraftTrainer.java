package mage.player.ai.rl;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.ComputerPlayerRL.DraftStepData;

/**
 * Co-training loop for the Vintage Cube Draft Model.
 *
 * Each episode:
 *   1. Run an 8-player draft (RL agent + 7 heuristic bots)
 *   2. RL agent constructs a deck (construction head + heuristic land base)
 *   3. Self-play games: RL game model plays both sides
 *      - RL-drafted deck vs heuristic-drafted deck
 *   4. Draft reward = RL win rate across evaluation games
 *   5. Train draft model via PPO (per-pick + construction advantages)
 *   6. Game model also trains from self-play game data (handled by RLTrainer/shared model)
 *
 * Run as: mvn exec:java -Dexec.mainClass=mage.player.ai.rl.DraftTrainer -Dexec.args=train
 */
public class DraftTrainer {

    private static final Logger logger = Logger.getLogger(DraftTrainer.class);

    // Configuration
    private static final int NUM_DRAFT_EPISODES   = EnvConfig.i32("DRAFT_TOTAL_EPISODES", 5000);
    private static final int GAMES_PER_EPISODE    = EnvConfig.i32("DRAFT_GAMES_PER_EPISODE", 3);
    private static final int SAVE_EVERY           = EnvConfig.i32("DRAFT_SAVE_EVERY", 50);
    private static final int DRAFT_LOG_FREQUENCY  = EnvConfig.i32("DRAFT_LOG_FREQUENCY", 10);
    private static final int STATS_EVERY          = EnvConfig.i32("DRAFT_STATS_EVERY", 10);
    private static final int WINRATE_WINDOW       = EnvConfig.i32("DRAFT_WINRATE_WINDOW", 100);

    // Paths
    private static final String DRAFT_MODEL_PATH  = RLLogPaths.DRAFT_MODEL_FILE_PATH;
    private static final String EPISODE_COUNT_PATH = RLLogPaths.DRAFT_EPISODE_COUNT_PATH;
    private static final String STATS_PATH        = RLLogPaths.DRAFT_STATS_PATH;

    private final DraftPythonMLBridge draftBridge;
    private final PythonModel gameModel;
    private final DraftRunner draftRunner;

    // Rolling win tracking
    private final java.util.Deque<Boolean> recentWins = new java.util.ArrayDeque<>();
    private final AtomicInteger totalWins = new AtomicInteger(0);
    private final AtomicInteger totalGames = new AtomicInteger(0);

    private int startEpisode = 0;

    public DraftTrainer() {
        this.draftBridge = DraftPythonMLBridge.getInstance();
        this.gameModel = PythonMLService.getInstance();
        this.draftRunner = new DraftRunner();
    }

    // -----------------------------------------------------------------------
    // Main training loop
    // -----------------------------------------------------------------------

    public void train() {
        startEpisode = loadEpisodeCount();
        logger.info(String.format("Starting draft training at episode %d (target: %d)",
                startEpisode, NUM_DRAFT_EPISODES));

        // Ensure output directories exist
        try {
            Files.createDirectories(Paths.get(RLLogPaths.DRAFT_LOGS_BASE_DIR));
            Files.createDirectories(Paths.get(RLLogPaths.DRAFT_MODELS_BASE_DIR));
        } catch (IOException e) {
            logger.warn("Could not create draft directories: " + e.getMessage());
        }

        Random rand = new Random();
        int episode = startEpisode;

        while (episode < NUM_DRAFT_EPISODES) {
            long t0 = System.currentTimeMillis();

            try {
                boolean logThis = (DRAFT_LOG_FREQUENCY > 0)
                        && (episode == 0 || episode % DRAFT_LOG_FREQUENCY == 0);
                DraftLogger draftLogger = DraftLogger.create(logThis);

                // 1. Create RL player and reset draft state
                ComputerPlayerRL rlPlayer = new ComputerPlayerRL(
                        "DraftRL", RangeOfInfluence.ALL, gameModel);
                rlPlayer.resetDraftState(draftBridge, draftLogger, episode, logThis);

                // 2. Run draft (RL agent uses pick_head via pickCard override)
                DraftRunner.DraftResult draftResult = draftRunner.runDraft(rlPlayer);
                if (draftResult == null) {
                    logger.warn("Draft failed for episode " + episode + ", skipping");
                    episode++;
                    continue;
                }

                // 3. Construct deck with construction head
                Deck rlDeck = rlPlayer.constructDraftDeck();
                if (rlDeck == null || rlDeck.getCards().isEmpty()) {
                    logger.warn("Empty RL deck at episode " + episode + ", skipping");
                    episode++;
                    continue;
                }

                // Ensure minimum 40 cards - pad with basic lands if needed
                rlDeck = padDeckToMinimum(rlDeck, 40);

                // Pick a heuristic-drafted deck for the opponent side
                Deck heuristicDeck = draftResult.getRandomHeuristicDeck(rand);
                if (heuristicDeck == null || heuristicDeck.getCards().isEmpty()) {
                    logger.warn("No heuristic deck at episode " + episode);
                    episode++;
                    continue;
                }
                // Build a playable heuristic deck from sideboard
                Deck playableHeuristicDeck = buildHeuristicDeckFromSideboard(heuristicDeck);

                // 4. Self-play evaluation games
                int wins = 0;
                int losses = 0;
                for (int g = 0; g < GAMES_PER_EPISODE; g++) {
                    boolean rlWon = playEvaluationGame(rlDeck, playableHeuristicDeck);
                    if (rlWon) { wins++; } else { losses++; }
                }

                float winRate = wins / (float) GAMES_PER_EPISODE;
                float reward  = winRate * 2.0f - 1.0f; // [-1, +1]

                // 5. Update rolling win stats
                recentWins.add(winRate > 0.5f);
                if (recentWins.size() > WINRATE_WINDOW) recentWins.poll();
                totalGames.addAndGet(GAMES_PER_EPISODE);
                totalWins.addAndGet(wins);

                // 6. Train draft model
                List<DraftStepData> steps = rlPlayer.getDraftSteps();
                if (!steps.isEmpty()) {
                    trainDraftModel(steps, reward);
                }

                // 7. Log
                if (logThis) {
                    draftLogger.logEpisodeResults(wins, losses, reward, null);
                    draftLogger.close();
                }

                long elapsed = System.currentTimeMillis() - t0;
                double rollingWr = getRollingWinRate();

                if (episode % STATS_EVERY == 0) {
                    logger.info(String.format(
                            "[Draft ep %d] W=%d L=%d reward=%.3f rolling_wr=%.3f (%dms)",
                            episode, wins, losses, reward, rollingWr, elapsed));
                    appendEpisodeStats(episode, wins, losses, reward, rollingWr);
                }

                // 8. Periodic save
                if (episode > 0 && episode % SAVE_EVERY == 0) {
                    draftBridge.saveDraftModel(DRAFT_MODEL_PATH);
                    saveEpisodeCount(episode);
                    logger.info("Saved draft model at episode " + episode);
                }

                episode++;

            } catch (Exception e) {
                logger.error("Error at draft episode " + episode + ": " + e.getMessage(), e);
                episode++;
            }
        }

        // Final save
        draftBridge.saveDraftModel(DRAFT_MODEL_PATH);
        saveEpisodeCount(episode);
        logger.info("Draft training complete. Total episodes: " + episode);
    }

    // -----------------------------------------------------------------------
    // Game self-play
    // -----------------------------------------------------------------------

    /**
     * Play one evaluation game: RL game model plays both sides.
     * RL-drafted deck on one side, heuristic-drafted deck on the other.
     *
     * @return true if RL-drafted deck side won
     */
    private boolean playEvaluationGame(Deck rlDeck, Deck heuristicDeck) {
        try {
            TwoPlayerMatch match = new TwoPlayerMatch(
                    new MatchOptions("DraftEval", "DraftEval", false));
            match.startGame();
            Game game = match.getGames().get(0);

            // RL player uses game model (no draft model needed here)
            ComputerPlayerRL rlSide = new ComputerPlayerRL(
                    "RLSide", RangeOfInfluence.ALL, gameModel);
            // Opponent also uses game model (self-play)
            ComputerPlayerRL heuristicSide = new ComputerPlayerRL(
                    "HeuristicSide", RangeOfInfluence.ALL, gameModel);

            game.addPlayer(rlSide, rlDeck);
            match.addPlayer(rlSide, rlDeck);
            game.addPlayer(heuristicSide, heuristicDeck);
            match.addPlayer(heuristicSide, heuristicDeck);

            game.loadCards(rlDeck.getCards(), rlSide.getId());
            game.loadCards(heuristicDeck.getCards(), heuristicSide.getId());

            GameOptions opts = new GameOptions();
            game.setGameOptions(opts);

            GameHealthMonitor monitor = GameHealthMonitor.createAndStart(game);
            game.start(rlSide.getId());
            monitor.stop();

            if (monitor.wasKilled()) {
                return false; // killed games count as losses
            }

            // Determine winner
            String winner = game.getWinner();
            return winner != null && winner.contains(rlSide.getName());

        } catch (Exception e) {
            logger.warn("Error in evaluation game: " + e.getMessage());
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // Draft model training
    // -----------------------------------------------------------------------

    private void trainDraftModel(List<DraftStepData> steps, float terminalReward) {
        int n = steps.size();
        if (n == 0) return;

        int seqLen = DraftStateBuilder.MAX_LEN;
        int dim = DraftStateBuilder.DIM_PER_TOKEN;
        int maxCands = computeMaxCands(steps);

        // Allocate flat byte arrays
        float[] stateArr  = new float[n * seqLen * dim];
        int[]   maskArr   = new int[n * seqLen];
        int[]   tokArr    = new int[n * seqLen];
        float[] cfArr     = new float[n * maxCands * dim];
        int[]   cidArr    = new int[n * maxCands];
        int[]   cmaskArr  = new int[n * maxCands]; // 1=real, 0=padding (matches Java convention)
        int[]   chosenArr = new int[n];
        float[] logpArr   = new float[n];
        float[] valArr    = new float[n];
        float[] rewArr    = new float[n];
        int[]   doneArr   = new int[n];
        int[]   headArr   = new int[n];

        for (int i = 0; i < n; i++) {
            DraftStepData s = steps.get(i);

            System.arraycopy(s.stateFlat, 0, stateArr, i * seqLen * dim, seqLen * dim);
            System.arraycopy(s.maskFlat,  0, maskArr,  i * seqLen, seqLen);
            System.arraycopy(s.tokenIds,  0, tokArr,   i * seqLen, seqLen);

            int nc = s.candIds.length;
            // copy candidate features (pad with zeros to maxCands)
            System.arraycopy(s.candFeats, 0, cfArr, i * maxCands * dim, nc * dim);
            System.arraycopy(s.candIds,   0, cidArr, i * maxCands, nc);
            // candidate mask: 1=real, 0=padding
            for (int j = 0; j < nc; j++) cmaskArr[i * maxCands + j] = 1;

            chosenArr[i] = s.chosenIdx;
            logpArr[i]   = s.logProb;
            valArr[i]    = s.value;

            // Reward: 0 for all intermediate steps, terminal reward for last step
            rewArr[i] = (i == n - 1) ? terminalReward : 0.0f;
            doneArr[i] = (i == n - 1) ? 1 : 0;
            headArr[i] = s.headIdx;
        }

        byte[] stateBytes  = toBytes(stateArr);
        byte[] maskBytes   = toBytes(maskArr);
        byte[] tokBytes    = toBytes(tokArr);
        byte[] cfBytes     = toBytes(cfArr);
        byte[] cidBytes    = toBytes(cidArr);
        byte[] cmaskBytes  = toBytes(cmaskArr);
        byte[] chosenBytes = toBytes(chosenArr);
        byte[] logpBytes   = toBytes(logpArr);
        byte[] valBytes    = toBytes(valArr);
        byte[] rewBytes    = toBytes(rewArr);
        byte[] doneBytes   = toBytes(doneArr);
        byte[] headBytes   = toBytes(headArr);

        draftBridge.trainDraftBatch(
                stateBytes, maskBytes, tokBytes,
                cfBytes, cidBytes, cmaskBytes,
                chosenBytes, logpBytes, valBytes,
                rewBytes, doneBytes, headBytes,
                n, seqLen, dim, maxCands);
    }

    private int computeMaxCands(List<DraftStepData> steps) {
        int max = 1;
        for (DraftStepData s : steps) {
            if (s.candIds.length > max) max = s.candIds.length;
        }
        return max;
    }

    // -----------------------------------------------------------------------
    // Deck construction helpers
    // -----------------------------------------------------------------------

    /**
     * Build a playable 40-card deck from a heuristic drafter's sideboard
     * (all drafted cards), using the heuristic AI's own logic.
     */
    private Deck buildHeuristicDeckFromSideboard(Deck sideboardDeck) {
        List<Card> pool = new ArrayList<>(sideboardDeck.getSideboard());
        if (pool.isEmpty()) {
            pool.addAll(sideboardDeck.getCards());
        }

        // Sort by CMC and take 23 non-lands
        List<Card> nonLands = new ArrayList<>();
        List<Card> lands = new ArrayList<>();
        for (Card c : pool) {
            if (c.isLand()) { lands.add(c); }
            else { nonLands.add(c); }
        }

        nonLands.sort((a, b) -> Double.compare(a.getManaValue(), b.getManaValue()));
        List<Card> mainNonLands = nonLands.subList(0, Math.min(23, nonLands.size()));

        // Heuristic land base from drafted lands + basics
        List<Card> landBase = new ArrayList<>(lands.subList(0, Math.min(17, lands.size())));
        // If not enough lands, just use what we have
        int needed = 40 - mainNonLands.size() - landBase.size();
        // Add extra non-lands if short on lands
        if (needed > 0 && nonLands.size() > mainNonLands.size()) {
            List<Card> extras = nonLands.subList(mainNonLands.size(),
                    Math.min(nonLands.size(), mainNonLands.size() + needed));
            landBase.addAll(extras);
        }

        Deck deck = new Deck();
        for (Card c : mainNonLands) deck.getCards().add(c);
        for (Card c : landBase) deck.getCards().add(c);
        return deck;
    }

    /**
     * Pad deck to minimum size with land cards from sideboard, or just leave as-is.
     * In cube, players should always have 40 cards, but handle edge cases.
     */
    private Deck padDeckToMinimum(Deck deck, int minimum) {
        int current = deck.getCards().size();
        if (current >= minimum) return deck;

        // Try to find extra cards from sideboard or just duplicate existing lands
        List<Card> sideCards = new ArrayList<>(deck.getSideboard());
        for (Card c : sideCards) {
            if (deck.getCards().size() >= minimum) break;
            deck.getCards().add(c);
        }
        return deck;
    }

    // -----------------------------------------------------------------------
    // Episode counter persistence
    // -----------------------------------------------------------------------

    private int loadEpisodeCount() {
        try {
            Path p = Paths.get(EPISODE_COUNT_PATH);
            if (Files.exists(p)) {
                return Integer.parseInt(new String(Files.readAllBytes(p)).trim());
            }
        } catch (Exception e) {
            logger.warn("Could not read draft episode count: " + e.getMessage());
        }
        return 0;
    }

    private void saveEpisodeCount(int count) {
        try {
            Path p = Paths.get(EPISODE_COUNT_PATH);
            Files.createDirectories(p.getParent());
            Files.write(p, String.valueOf(count).getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            logger.warn("Could not save draft episode count: " + e.getMessage());
        }
    }

    private double getRollingWinRate() {
        if (recentWins.isEmpty()) return 0.0;
        long wins = recentWins.stream().filter(b -> b).count();
        return wins / (double) recentWins.size();
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    private void appendEpisodeStats(int episode, int wins, int losses,
            float reward, double rollingWr) {
        try {
            Path p = Paths.get(STATS_PATH);
            Files.createDirectories(p.getParent());

            boolean exists = Files.exists(p);
            String header = "episode,wins,losses,reward,rolling_winrate\n";
            String row = String.format("%d,%d,%d,%.4f,%.4f%n",
                    episode, wins, losses, reward, rollingWr);

            if (!exists) {
                Files.write(p, header.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE);
            }
            Files.write(p, row.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write draft stats: " + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Byte conversion helpers
    // -----------------------------------------------------------------------

    private static byte[] toBytes(float[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : arr) buf.putFloat(v);
        return buf.array();
    }

    private static byte[] toBytes(int[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : arr) buf.putInt(v);
        return buf.array();
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        BasicConfigurator.configure();
        Level logLevel = parseLogLevel(System.getenv("MTG_AI_LOG_LEVEL"), Level.WARN);
        LogManager.getRootLogger().setLevel(logLevel);
        Logger.getLogger("mage.game").setLevel(Level.ERROR);
        Logger.getLogger("mage.server").setLevel(Level.ERROR);
        Logger.getLogger("mage.player.ai.ComputerPlayer6").setLevel(Level.ERROR);

        logger.info("Initializing DraftTrainer...");
        DraftTrainer trainer = new DraftTrainer();
        trainer.train();
    }

    private static Level parseLogLevel(String s, Level def) {
        if (s == null) return def;
        switch (s.trim().toUpperCase()) {
            case "DEBUG": return Level.DEBUG;
            case "INFO":  return Level.INFO;
            case "WARN":  return Level.WARN;
            case "ERROR": return Level.ERROR;
            case "OFF":   return Level.OFF;
            default:      return def;
        }
    }
}
