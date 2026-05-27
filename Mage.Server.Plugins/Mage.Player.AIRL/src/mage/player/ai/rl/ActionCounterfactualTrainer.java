package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.costs.mana.ManaCost;
import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.repository.TokenRepository;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.combat.CombatGroup;
import mage.game.match.MatchOptions;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;
import mage.target.Target;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Collection;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;

/**
 * Terminal-only post-mulligan action counterfactual trainer.
 *
 * A baseline greedy rollout records early eligible gameplay decisions. For each
 * recorded decision, branch rollouts replay the same seed and previous greedy
 * choices, force one candidate at the target decision, and continue to terminal.
 * Terminal branch outcomes become policy-distillation targets on the original
 * decision tensor.
 */
public final class ActionCounterfactualTrainer {

    private static final String DEFAULT_AGENT_DECK_LIST =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Combo.dek";
    private static final String DEFAULT_OPP_DECK_LIST =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt";
    private static final String DEFAULT_OUT_ROOT =
            "local-training/local_pbt/action_counterfactual";
    private static final float BRANCH_RETURN_UNOBSERVED = -2.0f;
    private static final Object STACK_PRIORITY_TRACE_LOCK = new Object();
    private static final boolean STACK_PRIORITY_TRACE_ENABLED = EnvConfig.bool(
            "EVAL_REPLAY_STACK_PRIORITY_TRACE_JSON",
            EnvConfig.bool("EVAL_REPLAY_STACK_PRIORITY_TRACE", false));
    private static final boolean CP7_TOKEN_METADATA_PARITY_ENABLED = EnvConfig.bool(
            "EVAL_REPLAY_CP7_TOKEN_METADATA_PARITY",
            EnvConfig.bool("EVAL_REPLAY_CP7_TOKEN_METADATA_RNG_PARITY", false));
    private static final Path STACK_PRIORITY_TRACE_FILE = stackPriorityTracePath();
    private static final Pattern REPLAY_ACTOR_TARGET_SUFFIX = Pattern.compile(
            "\\s*\\((?:evalbot-skill\\d+|acf-[^)]+|player:acf-[^)]+|playerrl\\d+)\\)$",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern REPLAY_OBJECT_ID_MARKER = Pattern.compile(
            "\\s+\\[[0-9a-f]{2,}\\]",
            Pattern.CASE_INSENSITIVE);
    private static final String OPPONENT_DECISION_JSON_PREFIX = "REPLAY_OPPONENT_DECISION_JSON:";
    private static final long REPLAY_RANDOM_UTIL_SALT = 0x6A09E667F3BCC909L;
    private static final Object REPLAY_STARTUP_DIAGNOSTIC_LOCK = new Object();
    private static final Path REPLAY_STARTUP_DIAGNOSTIC_FILE = replayStartupDiagnosticPath();

    private static long replayRandomUtilSeed(long replaySeed) {
        return replaySeed ^ REPLAY_RANDOM_UTIL_SALT;
    }

    private static void seedReplayRandomUtil(Args args, long replaySeed) {
        if (args != null && args.replayFile != null) {
            RandomUtil.setSeed(replayRandomUtilSeed(replaySeed));
        }
    }

    private static void setReplayTraceContext(ScenarioJob job, String scope) {
        if (job == null) {
            return;
        }
        System.setProperty("xmage.replay.scenario", String.valueOf(job.scenario));
        System.setProperty("xmage.replay.seed", String.valueOf(job.seed));
        System.setProperty("xmage.replay.random_util_seed", String.valueOf(replayRandomUtilSeed(job.seed)));
        System.setProperty("xmage.replay.scope", scope == null ? "" : scope);
    }

    private static Path replayStartupDiagnosticPath() {
        String raw = EnvConfig.str("EVAL_REPLAY_STARTUP_DIAGNOSTIC_FILE", "").trim();
        if (raw.isEmpty()) {
            return null;
        }
        return Paths.get(raw).toAbsolutePath().normalize();
    }

    private static void appendReplayStartupDiagnostic(
            String scope,
            String stage,
            ScenarioJob job,
            Deck agentDeck,
            Deck oppDeck,
            Player rlPlayer,
            Player opponent,
            Game game
    ) {
        if (REPLAY_STARTUP_DIAGNOSTIC_FILE == null) {
            return;
        }
        try {
            if (REPLAY_STARTUP_DIAGNOSTIC_FILE.getParent() != null) {
                Files.createDirectories(REPLAY_STARTUP_DIAGNOSTIC_FILE.getParent());
            }
            String header = "scope,stage,scenario,seed,turn,phase,step,active,priority,"
                    + "agent_deck_size,agent_deck_top,opp_deck_size,opp_deck_top,"
                    + "rl_hand_size,rl_hand,rl_library_size,rl_library_top,rl_graveyard_size,rl_battlefield_size,"
                    + "opp_hand_size,opp_library_size,expected_hand_size,expected_hand,expected_library_size,expected_library_top\n";
            String line = csv(scope)
                    + "," + csv(stage)
                    + "," + (job == null ? "" : String.valueOf(job.scenario))
                    + "," + (job == null ? "" : String.valueOf(job.seed))
                    + "," + safeGameTurn(game)
                    + "," + csv(game == null ? "" : String.valueOf(game.getTurnPhaseType()))
                    + "," + csv(game == null ? "" : String.valueOf(game.getTurnStepType()))
                    + "," + csv(gamePlayerName(game, game == null ? null : game.getActivePlayerId()))
                    + "," + csv(gamePlayerName(game, game == null ? null : game.getPriorityPlayerId()))
                    + "," + deckCardCount(agentDeck)
                    + "," + csv(deckCardNames(agentDeck, 12))
                    + "," + deckCardCount(oppDeck)
                    + "," + csv(deckCardNames(oppDeck, 12))
                    + "," + playerHandSize(rlPlayer)
                    + "," + csv(playerHandNames(rlPlayer, game, 12))
                    + "," + playerLibrarySize(rlPlayer)
                    + "," + csv(playerLibraryNames(rlPlayer, game, 12))
                    + "," + playerGraveyardSize(rlPlayer)
                    + "," + playerBattlefieldSize(rlPlayer, game)
                    + "," + playerHandSize(opponent)
                    + "," + playerLibrarySize(opponent)
                    + "," + (job == null ? "" : String.valueOf(job.agentOpeningHandNames.size()))
                    + "," + csv(job == null ? "" : String.join("|", job.agentOpeningHandNames))
                    + "," + (job == null ? "" : String.valueOf(job.agentOpeningLibraryNames.size()))
                    + "," + csv(job == null ? "" : limitNames(job.agentOpeningLibraryNames, 12))
                    + "\n";
            synchronized (REPLAY_STARTUP_DIAGNOSTIC_LOCK) {
                boolean writeHeader = !Files.exists(REPLAY_STARTUP_DIAGNOSTIC_FILE)
                        || Files.size(REPLAY_STARTUP_DIAGNOSTIC_FILE) == 0;
                try (BufferedWriter out = Files.newBufferedWriter(REPLAY_STARTUP_DIAGNOSTIC_FILE, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND)) {
                    if (writeHeader) {
                        out.write(header);
                    }
                    out.write(line);
                }
            }
        } catch (Exception ignored) {
            // Diagnostic-only path. Never affect replay behavior.
        }
    }

    private static int safeGameTurn(Game game) {
        try {
            return game == null ? -1 : game.getTurnNum();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static String gamePlayerName(Game game, UUID playerId) {
        try {
            Player player = game == null || playerId == null ? null : game.getPlayer(playerId);
            return player == null ? "" : player.getName();
        } catch (Exception ignored) {
            return "";
        }
    }

    private static int deckCardCount(Deck deck) {
        try {
            return deck == null ? -1 : deck.getCards().size();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static String deckCardNames(Deck deck, int limit) {
        try {
            return cardNameList(deck == null ? null : deck.getCards(), limit);
        } catch (Exception ignored) {
            return "";
        }
    }

    private static int playerHandSize(Player player) {
        try {
            return player == null || player.getHand() == null ? -1 : player.getHand().size();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static String playerHandNames(Player player, Game game, int limit) {
        try {
            return cardNameList(player == null || player.getHand() == null ? null : player.getHand().getCards(game), limit);
        } catch (Exception ignored) {
            return "";
        }
    }

    private static int playerLibrarySize(Player player) {
        try {
            return player == null || player.getLibrary() == null ? -1 : player.getLibrary().size();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static String playerLibraryNames(Player player, Game game, int limit) {
        try {
            return cardNameList(player == null || player.getLibrary() == null ? null : player.getLibrary().getCards(game), limit);
        } catch (Exception ignored) {
            return "";
        }
    }

    private static int playerGraveyardSize(Player player) {
        try {
            return player == null || player.getGraveyard() == null ? -1 : player.getGraveyard().size();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static int playerBattlefieldSize(Player player, Game game) {
        try {
            return player == null || game == null ? -1 : game.getBattlefield().getAllActivePermanents(player.getId()).size();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static String cardNameList(Collection<Card> cards, int limit) {
        List<String> names = new ArrayList<>();
        if (cards != null) {
            for (Card card : cards) {
                if (card != null && card.getName() != null && !card.getName().isEmpty()) {
                    names.add(card.getName());
                }
                if (names.size() >= limit) {
                    break;
                }
            }
        }
        return String.join("|", names);
    }

    private static String limitNames(List<String> names, int limit) {
        if (names == null || names.isEmpty()) {
            return "";
        }
        return names.stream().limit(limit).collect(Collectors.joining("|"));
    }

    private enum TerminalMode {
        WIN,
        SPY_COMBO_MILESTONE,
        SPY_COMBO_MILESTONE_ONLY,
        SPY_BALUSTRADE_REACHED,
        SPY_LANDLESS_COMBO_WIN
    }

    private ActionCounterfactualTrainer() {
    }

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);
        Path outDir = parsed.outDir;
        if (outDir == null) {
            String stamp = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").format(LocalDateTime.now());
            outDir = Paths.get(DEFAULT_OUT_ROOT, stamp).toAbsolutePath().normalize();
        }
        Files.createDirectories(outDir);
        parsed.outDir = outDir;
        if (parsed.forceOpponentTranscript && !"cp7".equals(parsed.opponentMode)) {
            throw new IllegalArgumentException("--force-opponent-transcript requires --opponent=cp7; opponent="
                    + parsed.opponentMode + " would not replay the transcript");
        }
        if (parsed.forceOpponentTranscript && parsed.opponentTranscriptMismatchPath == null) {
            parsed.opponentTranscriptMismatchPath = outDir.resolve("opponent_transcript_mismatch.csv");
        }
        if (parsed.opponentTranscriptMismatchPath != null) {
            initializeOpponentTranscriptMismatch(parsed.opponentTranscriptMismatchPath);
        }
        if (parsed.policyInputDump) {
            if (parsed.policyInputDumpPath == null) {
                parsed.policyInputDumpPath = outDir.resolve("policy_input_dump.csv");
            }
            initializePolicyInputDump(parsed.policyInputDumpPath);
        }
        if (parsed.policyInferenceProbe) {
            if (parsed.policyInferenceProbePath == null) {
                parsed.policyInferenceProbePath = outDir.resolve("policy_inference_probe.csv");
            }
            initializePolicyInferenceProbe(parsed.policyInferenceProbePath);
        }

        mage.cards.repository.CardScanner.scan();

        List<Path> agentDecks = loadDeckListOrSingle(parsed.agentDeckList);
        List<Path> oppDecks = loadDeckListOrSingle(parsed.oppDeckList);
        if (agentDecks.isEmpty() || oppDecks.isEmpty()) {
            throw new IllegalArgumentException("No decks found for action counterfactual run");
        }
        if (parsed.fitScoreProbe) {
            runImportedFitScoreProbe(parsed, outDir);
            RLTrainer.sharedModel.shutdown();
            return;
        }
        if (parsed.scoreTrainingDataPath != null) {
            runImportedScoreProbe(parsed, outDir);
            RLTrainer.sharedModel.shutdown();
            return;
        }
        if (parsed.importTrajectoryDataPath != null) {
            runImportedTrajectoryTraining(parsed, outDir);
            RLTrainer.sharedModel.shutdown();
            return;
        }
        if (parsed.importTrainingDataPath != null) {
            runImportedTraining(parsed, outDir);
            RLTrainer.sharedModel.shutdown();
            return;
        }
        if (parsed.replayFile != null) {
            runReplayProbe(parsed, outDir, agentDecks, oppDecks);
            RLTrainer.sharedModel.shutdown();
            return;
        }

        List<ScenarioJob> jobs = new ArrayList<>();
        Random rand = new Random(parsed.seed);
        for (int i = 1; i <= parsed.scenarios; i++) {
            Path agentDeck = agentDecks.get(rand.nextInt(agentDecks.size()));
            Path oppDeck = oppDecks.get(rand.nextInt(oppDecks.size()));
            long scenarioSeed = parsed.seed + 7919L * i;
            List<String> agentHand = selectOpeningHand(parsed.agentOpeningHandPool, parsed.agentOpeningHandNames, i);
            List<String> oppHand = selectOpeningHand(parsed.oppOpeningHandPool, parsed.oppOpeningHandNames, i);
            jobs.add(new ScenarioJob(i, agentDeck, oppDeck, scenarioSeed, agentHand,
                    Collections.emptyList(), oppHand));
        }

        List<DecisionRecord> records = new ArrayList<>();
        List<BranchValueProbeRecord> valueProbeRecords = new ArrayList<>();
        List<PrefixRecord> prefixRecords = new ArrayList<>();
        List<TrainingExample> examples = new ArrayList<>();
        List<WinningTrajectoryRecord> winningTrajectories = new ArrayList<>();
        List<TrajectoryTrainingEpisode> trajectoryEpisodes = new ArrayList<>();
        long started = System.currentTimeMillis();
        int completed = 0;
        int trainedScenarios = 0;
        int skippedScenarios = 0;

        ExecutorService executor = Executors.newFixedThreadPool(parsed.workers, new ActionWorkerThreadFactory());
        CompletionService<ScenarioOutcome> completion = new ExecutorCompletionService<>(executor);
        ThreadLocal<RLTrainer> workerTrainer = ThreadLocal.withInitial(RLTrainer::new);
        int submitted = 0;
        int maxInFlight = Math.max(1, Math.min(parsed.workers, jobs.size()));
        while (submitted < jobs.size() && submitted < maxInFlight) {
            ScenarioJob job = jobs.get(submitted++);
            completion.submit(() -> processScenario(workerTrainer.get(), job, parsed));
        }
        boolean stoppingEarly = false;

        try {
            while (completed < submitted) {
                ScenarioOutcome outcome;
                try {
                    outcome = completion.take().get();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while waiting for action counterfactual workers", ie);
                } catch (ExecutionException ee) {
                    throw new IllegalStateException("Action counterfactual worker failed", ee);
                }
                completed++;
                records.addAll(outcome.records);
                valueProbeRecords.addAll(outcome.valueProbeRecords);
                prefixRecords.addAll(outcome.prefixRecords);
                examples.addAll(outcome.trainingExamples);
                winningTrajectories.addAll(outcome.winningTrajectories);
                trajectoryEpisodes.addAll(outcome.trajectoryEpisodes);
                int winningTrajectoryCount = countWinningTrajectories(winningTrajectories);
                if (outcome.trainingExamples.isEmpty()) {
                    skippedScenarios++;
                } else {
                    trainedScenarios++;
                }
                if (completed % Math.max(1, parsed.reportEvery) == 0) {
                    writeDecisionRecords(outDir.resolve("action_branch_samples.csv"), records);
                    if (parsed.winningPrefixMode) {
                        writePrefixRecords(outDir.resolve("prefix_search_samples.csv"), prefixRecords);
                        writeWinningTrajectoryRecords(outDir.resolve("winning_trajectories.csv"), winningTrajectories);
                    }
                    writeReadme(outDir.resolve("README.md"), parsed, agentDecks, oppDecks,
                            completed, trainedScenarios, skippedScenarios, examples.size(), winningTrajectoryCount, 0,
                            System.currentTimeMillis() - started);
                    System.out.println(String.format(Locale.US,
                            "scenarios=%d/%d trainedScenarios=%d skippedScenarios=%d records=%d trajectories=%d selected=%d last=%s",
                            completed, parsed.scenarios, trainedScenarios, skippedScenarios,
                            records.size(), winningTrajectoryCount, examples.size(), outcome.label));
                }
                if (parsed.stopAfterExamples > 0 && examples.size() >= parsed.stopAfterExamples) {
                    System.out.println(String.format(Locale.US,
                            "earlyStop=stopAfterExamples scenarios=%d/%d selected=%d threshold=%d",
                            completed, parsed.scenarios, examples.size(), parsed.stopAfterExamples));
                    stoppingEarly = true;
                    break;
                }
                if (parsed.stopAfterWinningTrajectories > 0
                        && winningTrajectoryCount >= parsed.stopAfterWinningTrajectories) {
                    System.out.println(String.format(Locale.US,
                            "earlyStop=stopAfterWinningTrajectories scenarios=%d/%d trajectories=%d threshold=%d",
                            completed, parsed.scenarios, winningTrajectoryCount, parsed.stopAfterWinningTrajectories));
                    stoppingEarly = true;
                    break;
                }
                if (submitted < jobs.size()) {
                    ScenarioJob job = jobs.get(submitted++);
                    completion.submit(() -> processScenario(workerTrainer.get(), job, parsed));
                }
            }
        } finally {
            if (stoppingEarly) {
                executor.shutdownNow();
            } else {
                executor.shutdown();
            }
            try {
                long waitSec = stoppingEarly ? Math.max(15L, parsed.timeoutSec + 10L) : 15L;
                if (!executor.awaitTermination(waitSec, TimeUnit.SECONDS)) {
                    System.out.println("earlyStopWarning=workersStillTerminating");
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }

        List<TrainingExample> selected = selectExamples(examples, parsed);
        writeTrainingExamples(outDir.resolve("action_training_samples.csv"), selected);
        if (parsed.exportTrainingDataFile != null) {
            writeSerializedTrainingData(parsed.exportTrainingDataFile, selected);
        }
        if (parsed.exportTrajectoryDataFile != null) {
            writeSerializedTrajectoryData(parsed.exportTrajectoryDataFile, trajectoryEpisodes);
        }
        if (selected.isEmpty()) {
            writeTensorReplayRecords(outDir.resolve("tensor_replay_samples.csv"), Collections.emptyList());
            writeTensorReplayReadme(outDir.resolve("tensor_replay_summary.md"), Collections.emptyList());
            writeDecisionRecords(outDir.resolve("action_branch_samples.csv"), records);
            if (parsed.branchValueProbe) {
                writeBranchValueProbeRecords(outDir.resolve("branch_value_probe_samples.csv"), valueProbeRecords);
                writeBranchValueProbeReadme(outDir.resolve("branch_value_probe_summary.md"), valueProbeRecords);
            }
            if (parsed.winningPrefixMode) {
                writePrefixRecords(outDir.resolve("prefix_search_samples.csv"), prefixRecords);
                writeWinningTrajectoryRecords(outDir.resolve("winning_trajectories.csv"), winningTrajectories);
            }
            writeReadme(outDir.resolve("README.md"), parsed, agentDecks, oppDecks,
                    completed, trainedScenarios, skippedScenarios, examples.size(),
                    countWinningTrajectories(winningTrajectories), 0,
                    System.currentTimeMillis() - started);
            RLTrainer.sharedModel.shutdown();
            System.out.println("Action counterfactual output: " + outDir);
            System.out.println("trainedScenarios=" + trainedScenarios
                    + " skippedScenarios=" + skippedScenarios
                    + " candidateExamples=" + examples.size()
                    + " winningTrajectories=" + countWinningTrajectories(winningTrajectories)
                    + " selectedExamples=0 trainPassSamples=0 tensorTop1=0/0 tensorAccuracy=0.0000 stats={}");
            return;
        }
        if (parsed.collectOnly) {
            writeTensorReplayRecords(outDir.resolve("tensor_replay_samples.csv"), Collections.emptyList());
            writeTensorReplayReadme(outDir.resolve("tensor_replay_summary.md"), Collections.emptyList());
            writeDecisionRecords(outDir.resolve("action_branch_samples.csv"), records);
            if (parsed.branchValueProbe) {
                writeBranchValueProbeRecords(outDir.resolve("branch_value_probe_samples.csv"), valueProbeRecords);
                writeBranchValueProbeReadme(outDir.resolve("branch_value_probe_summary.md"), valueProbeRecords);
            }
            if (parsed.winningPrefixMode) {
                writePrefixRecords(outDir.resolve("prefix_search_samples.csv"), prefixRecords);
                writeWinningTrajectoryRecords(outDir.resolve("winning_trajectories.csv"), winningTrajectories);
            }
            writeReadme(outDir.resolve("README.md"), parsed, agentDecks, oppDecks,
                    completed, trainedScenarios, skippedScenarios, examples.size(),
                    countWinningTrajectories(winningTrajectories), 0,
                    System.currentTimeMillis() - started);
            RLTrainer.sharedModel.shutdown();
            System.out.println("Action counterfactual output: " + outDir);
            System.out.println("collectOnly=true"
                    + " trainedScenarios=" + trainedScenarios
                    + " skippedScenarios=" + skippedScenarios
                    + " candidateExamples=" + examples.size()
                    + " winningTrajectories=" + countWinningTrajectories(winningTrajectories)
                    + " selectedExamples=" + selected.size()
                    + " trainPassSamples=0 tensorTop1=0/0 tensorAccuracy=0.0000 stats={}");
            return;
        }
        trainSelectedExamples(selected, parsed);

        saveSharedModelLatest();
        Map<String, Integer> stats = RLTrainer.sharedModel.getMainModelTrainingStats();
        List<TensorReplayRecord> tensorReplayRecords = runTensorReplay(selected);
        writeTensorReplayRecords(outDir.resolve("tensor_replay_samples.csv"), tensorReplayRecords);
        writeTensorReplayReadme(outDir.resolve("tensor_replay_summary.md"), tensorReplayRecords);
        TensorReplayStats tensorStats = TensorReplayStats.from(tensorReplayRecords);

        writeDecisionRecords(outDir.resolve("action_branch_samples.csv"), records);
        if (parsed.branchValueProbe) {
            writeBranchValueProbeRecords(outDir.resolve("branch_value_probe_samples.csv"), valueProbeRecords);
            writeBranchValueProbeReadme(outDir.resolve("branch_value_probe_summary.md"), valueProbeRecords);
        }
        if (parsed.winningPrefixMode) {
            writePrefixRecords(outDir.resolve("prefix_search_samples.csv"), prefixRecords);
            writeWinningTrajectoryRecords(outDir.resolve("winning_trajectories.csv"), winningTrajectories);
        }
        writeTrainingExamples(outDir.resolve("action_training_samples.csv"), selected);
        writeReadme(outDir.resolve("README.md"), parsed, agentDecks, oppDecks,
                completed, trainedScenarios, skippedScenarios, examples.size(),
                countWinningTrajectories(winningTrajectories),
                selected.size() * parsed.trainEpochs * parsed.candidatePermutations,
                System.currentTimeMillis() - started);
        RLTrainer.sharedModel.shutdown();

        System.out.println("Action counterfactual output: " + outDir);
        System.out.println("trainedScenarios=" + trainedScenarios
                + " skippedScenarios=" + skippedScenarios
                + " candidateExamples=" + examples.size()
                + " winningTrajectories=" + countWinningTrajectories(winningTrajectories)
                + " selectedExamples=" + selected.size()
                + " trainPassSamples=" + (selected.size() * parsed.trainEpochs * parsed.candidatePermutations)
                + " tensorTop1=" + tensorStats.top1Matches + "/" + tensorStats.total
                + " tensorAccuracy=" + String.format(Locale.US, "%.4f", tensorStats.accuracy())
                + " stats=" + stats);
    }

    private static ScenarioOutcome processScenario(RLTrainer trainer, ScenarioJob job, Args args) {
        if (args.winningPrefixMode) {
            return processWinningPrefixScenario(trainer, job, args);
        }

        Pattern includeActionTextPattern = compileOptionalPattern(
                args.includeActionTextRegex,
                "--include-action-text-regex");
        Pattern avoidLosingActionTextPattern = compileOptionalPattern(
                args.avoidLosingActionTextRegex,
                "--avoid-losing-action-text-regex");
        long deadlineMs = scenarioDeadlineMs(args);
        BaselineResult baseline = runBaseline(trainer, job, args);
        List<DecisionRecord> records = new ArrayList<>();
        List<BranchValueProbeRecord> valueProbeRecords = new ArrayList<>();
        List<TrainingExample> examples = new ArrayList<>();
        List<TrajectoryTrainingEpisode> trajectoryEpisodes = new ArrayList<>();
        if (baseline.timedOut || !baseline.error.isEmpty() || baseline.decisions.isEmpty()) {
            return new ScenarioOutcome(records, valueProbeRecords, examples, Collections.emptyList(),
                    Collections.emptyList(), Collections.emptyList(),
                    baseline.timedOut ? "BASELINE_TIMEOUT" : "NO_DECISIONS");
        }

        for (DecisionPoint point : baseline.decisions) {
            if (scenarioDeadlineReached(deadlineMs)) {
                break;
            }
            if (!decisionPointMatchesActionText(point, includeActionTextPattern)) {
                continue;
            }
            List<Integer> branches = branchCandidates(point, args, job.seed + point.ordinal * 101L);
            if (branches.size() < 2) {
                continue;
            }

            List<BranchResult> branchResults = new ArrayList<>();
            List<TrajectoryTrainingEpisode> pointTrajectoryEpisodes = new ArrayList<>();
            Map<String, PrefixRunResult> subtreeRunCache = args.branchSubtreeSearchNodes > 0
                    ? new HashMap<>()
                    : Collections.emptyMap();
            for (Integer forcedIdx : branches) {
                if (scenarioDeadlineReached(deadlineMs)) {
                    break;
                }
                BranchResult br = args.branchSubtreeSearchNodes > 0
                        ? runForcedBranchSubtree(trainer, job, args, baseline, point, forcedIdx,
                                deadlineMs, subtreeRunCache)
                        : runForcedBranch(trainer, job, args,
                                baseline.prefixChoices, baseline.prefixChoiceTexts,
                                point.ordinal, forcedIdx, Collections.singletonList(point.candidateText(forcedIdx)));
                branchResults.add(br);
                records.add(DecisionRecord.from(job, point, br));
                if (args.branchTrajectoryMode
                        && !args.branchTrajectoryPairMode
                        && br.trajectoryEpisode != null
                        && !br.trajectoryEpisode.records.isEmpty()) {
                    pointTrajectoryEpisodes.add(br.trajectoryEpisode);
                }
            }

            TrainingExample example = buildTrainingExample(job, point, branchResults, args, avoidLosingActionTextPattern);
            if (args.branchTrajectoryMode && args.branchTrajectoryPairMode
                    && (!args.branchTrajectoryRequireTrainingExample || example != null)) {
                trajectoryEpisodes.addAll(buildBranchTrajectoryPairEpisodes(job, point, branchResults, args));
            }

            if (args.branchTrajectoryMode && !args.branchTrajectoryPairMode
                    && (!args.branchTrajectoryRequireTrainingExample || example != null)) {
                trajectoryEpisodes.addAll(pointTrajectoryEpisodes);
            }
            if (example != null) {
                examples.add(example);
                if (args.branchValueProbe) {
                    valueProbeRecords.addAll(runBranchValueProbesForExample(
                            trainer, job, args, baseline, point, branchResults, example, deadlineMs));
                }
            }
        }

        String label = examples.isEmpty() ? "NO_ACTIONABLE_LABEL" : "TRAIN";
        return new ScenarioOutcome(records, valueProbeRecords, examples, Collections.emptyList(),
                Collections.emptyList(), trajectoryEpisodes, label);
    }

    private static ScenarioOutcome processWinningPrefixScenario(RLTrainer trainer, ScenarioJob job, Args args) {
        List<PrefixRecord> prefixRecords = new ArrayList<>();
        List<TrainingExample> examples = new ArrayList<>();
        List<WinningTrajectoryRecord> winningTrajectories = new ArrayList<>();
        List<TrajectoryTrainingEpisode> trajectoryEpisodes = new ArrayList<>();
        List<PrefixNode> queue = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int nextNodeId = 1;
        PrefixNode root = new PrefixNode(nextNodeId++, 0,
                copyPrefix(args.initialPrefixChoices), Collections.emptyList(), "ROOT");
        queue.add(root);
        seen.add(prefixKey(root.prefixChoices));

        long deadlineMs = scenarioDeadlineMs(args);
        int cursor = 0;
        int nodesTried = 0;
        int winningPrefixes = 0;
        boolean scenarioTimedOut = false;
        Map<String, PrefixRunResult> runCache = new HashMap<>();
        while (cursor < queue.size()
                && nodesTried < args.maxSearchNodes
                && winningPrefixes < args.maxWinningPrefixesPerScenario) {
            if (scenarioDeadlineReached(deadlineMs)) {
                scenarioTimedOut = true;
                break;
            }
            PrefixNode node = queue.get(cursor++);
            nodesTried++;

            PrefixRunResult run = cachedPrefixRun(trainer, job, args, node, runCache);
            PrefixRecord prefixRecord = PrefixRecord.from(job, node, run);
            prefixRecords.add(prefixRecord);
            if (scenarioDeadlineReached(deadlineMs)) {
                scenarioTimedOut = true;
            }
            if (!run.timedOut && !run.error.isEmpty()) {
                continue;
            }
            if (run.forcedAppliedCount < node.prefixChoices.size()) {
                continue;
            }

            if (run.won) {
                List<List<Integer>> trainPrefix = winningTrainingPrefix(node.prefixChoices.size(), run.decisions, args);
                List<TrainingExample> found = args.trainRootMulliganOnly
                        ? buildRootMulliganFromWinningPrefix(job, run.decisions, trainPrefix, args)
                        : buildPrefixTrainingExamples(job, run.decisions, trainPrefix, args);
                if (!found.isEmpty()) {
                    examples.addAll(found);
                    winningTrajectories.addAll(buildWinningTrajectoryRecords(
                            job, run, trainPrefix, node.prefixChoices, winningPrefixes + 1));
                    TrajectoryTrainingEpisode episode = buildTrajectoryTrainingEpisode(
                            job, run, trainPrefix, node.prefixChoices, winningPrefixes + 1, args);
                    if (episode != null && !episode.records.isEmpty()) {
                        trajectoryEpisodes.add(episode);
                    }
                    winningPrefixes++;
                }
                continue;
            }

            if (node.prefixChoices.size() >= args.maxPrefixDepth) {
                continue;
            }
            if (run.decisions.size() <= node.prefixChoices.size()) {
                continue;
            }

            DecisionPoint next = run.decisions.get(node.prefixChoices.size());
            List<List<Integer>> branches = branchChoiceLists(next, args,
                    job.seed + 104729L * Math.max(1, nodesTried) + 997L * node.prefixChoices.size());
            prefixRecord.branchCount = branches.size();
            prefixRecord.branchOrder = branchOrderText(next, branches);
            List<PrefixNode> children = new ArrayList<>();
            for (List<Integer> branchChoice : branches) {
                List<List<Integer>> childPrefix = copyPrefix(node.prefixChoices);
                List<List<String>> childTexts = copyPrefixTexts(node.prefixChoiceTexts);
                childPrefix.add(branchChoice);
                childTexts.add(prefixChoiceTexts(next, branchChoice));
                String key = prefixKey(childPrefix);
                if (!seen.add(key)) {
                    continue;
                }
                children.add(new PrefixNode(nextNodeId++, node.nodeId, childPrefix, childTexts,
                        prefixChoiceText(next, branchChoice)));
            }
            if (args.prefixSiblingContrast && !children.isEmpty()) {
                List<PrefixBranchResult> siblingResults = new ArrayList<>();
                for (PrefixNode child : children) {
                    if (scenarioDeadlineReached(deadlineMs)) {
                        scenarioTimedOut = true;
                        break;
                    }
                    String childKey = prefixKey(child.prefixChoices);
                    PrefixRunResult childRun = cachedPrefixRun(trainer, job, args, child, runCache);
                    PrefixRunResult contrastRun = childRun;
                    if (args.prefixSiblingContrastSearchNodes > 0) {
                        PrefixSubtreeResult subtree = searchWinningPrefixSubtree(
                                trainer, job, args, child, args.prefixSiblingContrastSearchNodes,
                                deadlineMs, runCache);
                        if (subtree.won) {
                            contrastRun = new PrefixRunResult(Collections.emptyList(), true, false, "",
                                    node.prefixChoices.size() + 1, subtree.turns, "", "", StateSnapshot.EMPTY);
                        }
                    }
                    List<Integer> branchChoice = child.prefixChoices.isEmpty()
                            ? Collections.emptyList()
                            : child.prefixChoices.get(child.prefixChoices.size() - 1);
                    siblingResults.add(new PrefixBranchResult(branchChoice, contrastRun));
                }
                TrainingExample contrast = buildPrefixSiblingContrastExample(
                        job, next, siblingResults, node.prefixChoices.size() + 1, args);
                if (contrast != null) {
                    examples.add(contrast);
                }
            }
            enqueueChildren(queue, cursor, children, args.depthFirstSearch);
            appendPassMacroBranches(queue, seen, node, next, args, nextNodeId);
            nextNodeId += countNewPassMacroBranches(node, next, args, seen);
        }

        if (examples.isEmpty() && args.trainRootMulliganOnNoWin) {
            PrefixRunResult rootRun = runCache.get(prefixKey(Collections.emptyList()));
            TrainingExample mulliganExample = buildRootMulliganNoWinExample(job, rootRun, args);
            if (mulliganExample != null) {
                examples.add(mulliganExample);
            }
        }

        String label = examples.isEmpty()
                ? (prefixRecords.isEmpty() ? "NO_SEARCH" : (scenarioTimedOut ? "SCENARIO_TIMEOUT" : "NO_WINNING_PREFIX"))
                : "WINNING_PREFIX";
        return new ScenarioOutcome(Collections.emptyList(), Collections.emptyList(), examples, prefixRecords,
                winningTrajectories, trajectoryEpisodes, label);
    }

    private static PrefixRunResult cachedPrefixRun(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            PrefixNode node,
            Map<String, PrefixRunResult> runCache
    ) {
        String key = prefixKey(node.prefixChoices);
        PrefixRunResult run = runCache.get(key);
        if (run == null) {
            run = runPrefixBranch(trainer, job, args, node.prefixChoices, node.prefixChoiceTexts);
            runCache.put(key, run);
        }
        return run;
    }

    private static PrefixSubtreeResult searchWinningPrefixSubtree(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            PrefixNode root,
            int maxNodes,
            long deadlineMs,
            Map<String, PrefixRunResult> runCache
    ) {
        if (maxNodes <= 0 || root == null) {
            return PrefixSubtreeResult.noWin();
        }
        List<PrefixNode> queue = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        queue.add(root);
        seen.add(prefixKey(root.prefixChoices));
        int cursor = 0;
        int nodes = 0;
        while (cursor < queue.size() && nodes < maxNodes && !scenarioDeadlineReached(deadlineMs)) {
            PrefixNode node = queue.get(cursor++);
            nodes++;
            PrefixRunResult run = cachedPrefixRun(trainer, job, args, node, runCache);
            if (run == null || run.timedOut || !run.error.isEmpty()) {
                continue;
            }
            if (run.forcedAppliedCount < node.prefixChoices.size()) {
                continue;
            }
            if (run.won) {
                return new PrefixSubtreeResult(true, Math.max(1, run.turns), nodes);
            }
            if (node.prefixChoices.size() >= args.maxPrefixDepth) {
                continue;
            }
            if (run.decisions.size() <= node.prefixChoices.size()) {
                continue;
            }
            DecisionPoint next = run.decisions.get(node.prefixChoices.size());
            List<List<Integer>> branches = branchChoiceLists(next, args,
                    job.seed + 32452843L * Math.max(1, nodes) + 499L * node.prefixChoices.size());
            List<PrefixNode> children = new ArrayList<>();
            for (List<Integer> branchChoice : branches) {
                List<List<Integer>> childPrefix = copyPrefix(node.prefixChoices);
                List<List<String>> childTexts = copyPrefixTexts(node.prefixChoiceTexts);
                childPrefix.add(branchChoice);
                childTexts.add(prefixChoiceTexts(next, branchChoice));
                String key = prefixKey(childPrefix);
                if (!seen.add(key)) {
                    continue;
                }
                children.add(new PrefixNode(0, node.nodeId, childPrefix, childTexts,
                        prefixChoiceText(next, branchChoice)));
            }
            enqueueChildren(queue, cursor, children, args.depthFirstSearch);
            appendPassMacroBranches(queue, seen, node, next, args, 0);
        }
        return PrefixSubtreeResult.noWin();
    }

    private static BranchResult runForcedBranchSubtree(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            BaselineResult baseline,
            DecisionPoint point,
            int forcedIdx,
            long deadlineMs,
            Map<String, PrefixRunResult> runCache
    ) {
        if (point == null || baseline == null) {
            return BranchResult.failed(forcedIdx, "missing_baseline_or_point", false);
        }
        List<List<Integer>> prefix = new ArrayList<>();
        List<List<String>> prefixTexts = new ArrayList<>();
        int prior = Math.max(0, Math.min(point.ordinal, baseline.prefixChoices.size()));
        for (int i = 0; i < prior; i++) {
            prefix.add(new ArrayList<>(baseline.prefixChoices.get(i)));
            if (i < baseline.prefixChoiceTexts.size()) {
                prefixTexts.add(new ArrayList<>(baseline.prefixChoiceTexts.get(i)));
            } else {
                prefixTexts.add(Collections.emptyList());
            }
        }
        prefix.add(Collections.singletonList(forcedIdx));
        prefixTexts.add(Collections.singletonList(point.candidateText(forcedIdx)));

        PrefixNode root = new PrefixNode(0, 0, prefix, prefixTexts, point.candidateText(forcedIdx));
        PrefixSubtreeResult subtree = searchWinningPrefixSubtree(
                trainer, job, args, root, args.branchSubtreeSearchNodes, deadlineMs, runCache);
        PrefixRunResult rootRun = runCache.get(prefixKey(prefix));
        boolean forcedApplied = rootRun != null && rootRun.forcedAppliedCount >= prefix.size();
        boolean timedOut = scenarioDeadlineReached(deadlineMs)
                || (rootRun != null && rootRun.timedOut);
        String error = rootRun == null ? "subtree_root_not_run" : rootRun.error;
        int turns = subtree.won
                ? subtree.turns
                : (rootRun == null ? -1 : rootRun.turns);
            return new BranchResult(forcedIdx, subtree.won, timedOut, error, forcedApplied, turns);
        }

    private static void enqueueChildren(
            List<PrefixNode> queue,
            int cursor,
            List<PrefixNode> children,
            boolean depthFirst
    ) {
        if (children == null || children.isEmpty()) {
            return;
        }
        if (!depthFirst) {
            queue.addAll(children);
            return;
        }
        int insertAt = Math.max(0, Math.min(cursor, queue.size()));
        for (int i = children.size() - 1; i >= 0; i--) {
            queue.add(insertAt, children.get(i));
        }
    }

    private static List<String> selectOpeningHand(List<List<String>> pool, List<String> fallback, int scenario) {
        if (pool != null && !pool.isEmpty()) {
            int idx = Math.floorMod(Math.max(0, scenario - 1), pool.size());
            return new ArrayList<>(pool.get(idx));
        }
        return fallback == null ? Collections.emptyList() : new ArrayList<>(fallback);
    }

    private static long scenarioDeadlineMs(Args args) {
        if (args.scenarioTimeoutSec <= 0) {
            return Long.MAX_VALUE;
        }
        return System.currentTimeMillis() + args.scenarioTimeoutSec * 1000L;
    }

    private static boolean scenarioDeadlineReached(long deadlineMs) {
        return deadlineMs != Long.MAX_VALUE && System.currentTimeMillis() >= deadlineMs;
    }

    private static void runReplayProbe(
            Args args,
            Path outDir,
            List<Path> agentDecks,
            List<Path> oppDecks
    ) throws Exception {
        List<ReplayExpectation> expectations = loadReplayExpectations(args.replayFile, agentDecks, oppDecks);
        if (expectations.isEmpty()) {
            throw new IllegalArgumentException("Replay file contained no expectations: " + args.replayFile);
        }

        int maxOrdinal = expectations.stream().mapToInt(e -> e.ordinal).max().orElse(0);
        args.maxDecisionDepth = Math.max(args.maxDecisionDepth, maxOrdinal + 1);

        LinkedHashMap<String, ReplayGroup> groups = new LinkedHashMap<>();
        for (ReplayExpectation expectation : expectations) {
            String key = expectation.scenario + "|" + expectation.seed + "|"
                    + expectation.agentDeck + "|" + expectation.oppDeck;
            groups.computeIfAbsent(key, ignored -> new ReplayGroup(expectation)).expectations.add(expectation);
        }

        List<ReplayGroup> groupList = new ArrayList<>(groups.values());
        if (args.replayMaxScenarios > 0 && groupList.size() > args.replayMaxScenarios) {
            groupList = new ArrayList<>(groupList.subList(0, args.replayMaxScenarios));
        }
        if (args.checkpointBranchProbe) {
            runCheckpointBranchProbe(args, outDir, groupList);
            return;
        }
        if (args.forcedPrefixReplay) {
            args.livePrefixTracePath = outDir.resolve("forced_prefix_trace.csv");
            initializePrefixTraceRecords(args.livePrefixTracePath);
        }

        ExecutorService executor = Executors.newFixedThreadPool(
                Math.max(1, Math.min(args.workers, groupList.size())),
                new ActionWorkerThreadFactory());
        CompletionService<ReplayGroupResult> completion = new ExecutorCompletionService<>(executor);
        ThreadLocal<RLTrainer> workerTrainer = ThreadLocal.withInitial(RLTrainer::new);

        long started = System.currentTimeMillis();
        for (ReplayGroup group : groupList) {
            completion.submit(() -> replayGroup(workerTrainer.get(), group, args));
        }

        List<ReplayRecord> records = new ArrayList<>();
        List<PrefixTraceRecord> prefixTraceRecords = new ArrayList<>();
        List<TrainingExample> deviationExamples = new ArrayList<>();
        List<TrainingExample> daggerExamples = new ArrayList<>();
        int completed = 0;
        try {
            while (completed < groupList.size()) {
                try {
                    ReplayGroupResult result = completion.take().get();
                    records.addAll(result.records);
                    prefixTraceRecords.addAll(result.prefixTraceRecords);
                    deviationExamples.addAll(result.deviationExamples);
                    daggerExamples.addAll(result.daggerExamples);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while waiting for replay probe workers", ie);
                } catch (ExecutionException ee) {
                    throw new IllegalStateException("Replay probe worker failed", ee);
                }
                completed++;
                if (completed % Math.max(1, args.reportEvery) == 0) {
                    writeReplayRecords(outDir.resolve("replay_samples.csv"), records);
                    if (!prefixTraceRecords.isEmpty()) {
                        writePrefixTraceRecords(outDir.resolve("forced_prefix_trace.csv"), prefixTraceRecords);
                    }
                    writeReplayReadme(outDir.resolve("README.md"), args, records, deviationExamples.size(),
                            daggerExamples.size(),
                            completed, groupList.size(),
                            System.currentTimeMillis() - started);
                    ReplayStats stats = ReplayStats.from(records);
                    System.out.println(String.format(Locale.US,
                            "replay=%d/%d matched=%d/%d accuracy=%.3f deviations=%d dagger=%d",
                            completed, groupList.size(), stats.matched, stats.total, stats.accuracy(),
                            deviationExamples.size(), daggerExamples.size()));
                }
            }
        } finally {
            executor.shutdownNow();
        }

        writeReplayRecords(outDir.resolve("replay_samples.csv"), records);
        if (!prefixTraceRecords.isEmpty()) {
            writePrefixTraceRecords(outDir.resolve("forced_prefix_trace.csv"), prefixTraceRecords);
        }
        if (args.replayDeviationTrainingDataFile != null) {
            writeSerializedTrainingData(args.replayDeviationTrainingDataFile, deviationExamples);
            writeTrainingExamples(outDir.resolve("deviation_training_samples.csv"), deviationExamples);
        }
        if (args.replayDaggerTrainingDataFile != null) {
            writeSerializedTrainingData(args.replayDaggerTrainingDataFile, daggerExamples);
            writeTrainingExamples(outDir.resolve("dagger_training_samples.csv"), daggerExamples);
        }
        writeReplayReadme(outDir.resolve("README.md"), args, records, deviationExamples.size(),
                daggerExamples.size(),
                completed, groupList.size(),
                System.currentTimeMillis() - started);
        ReplayStats stats = ReplayStats.from(records);
        System.out.println("Replay probe output: " + outDir);
        System.out.println(String.format(Locale.US,
                "replayMatched=%d replayTotal=%d replayAccuracy=%.4f scenarioWins=%d scenarioTotal=%d deviationExamples=%d daggerExamples=%d",
                stats.matched, stats.total, stats.accuracy(), stats.wonGroups, stats.totalGroups,
                deviationExamples.size(), daggerExamples.size()));
    }

    private static void runCheckpointBranchProbe(
            Args args,
            Path outDir,
            List<ReplayGroup> groupList
    ) throws IOException {
        RLTrainer trainer = new RLTrainer();
        List<CheckpointBranchRecord> records = new ArrayList<>();
        List<EngineDecisionCheckpoint> capturedCheckpoints = new ArrayList<>();
        long started = System.currentTimeMillis();
        boolean previousForcedPrefixReplay = args.forcedPrefixReplay;
        args.forcedPrefixReplay = true;
        try {
            int completed = 0;
            for (ReplayGroup group : groupList) {
                ScenarioJob job = new ScenarioJob(group.scenario, group.agentDeck, group.oppDeck, group.seed,
                        group.agentOpeningHandNames.isEmpty() ? args.agentOpeningHandNames : group.agentOpeningHandNames,
                        group.agentOpeningLibraryNames,
                        group.oppOpeningHandNames.isEmpty() ? args.oppOpeningHandNames : group.oppOpeningHandNames);
                Map<Integer, ReplayExpectation> byOrdinal = new HashMap<>();
                for (ReplayExpectation expectation : group.expectations) {
                    byOrdinal.put(expectation.ordinal, expectation);
                }
                for (ReplayExpectation expected : replayTargets(group)) {
                    List<List<Integer>> prefix = new ArrayList<>();
                    List<List<String>> prefixTexts = new ArrayList<>();
                    List<StateSequenceBuilder.ActionType> prefixActionTypes = new ArrayList<>();
                    List<ReplayExpectation> prefixExpectations = new ArrayList<>();
                    for (int ordinal = 0; ordinal < expected.ordinal; ordinal++) {
                        ReplayExpectation prior = byOrdinal.get(ordinal);
                        if (prior == null || prior.expectedIndices.isEmpty()) {
                            prefix.add(Collections.emptyList());
                            prefixTexts.add(Collections.emptyList());
                            prefixActionTypes.add(null);
                            prefixExpectations.add(null);
                        } else {
                            prefix.add(new ArrayList<>(prior.expectedIndices));
                            prefixTexts.add(prior.expectedTexts == null
                                    ? Collections.emptyList()
                                    : new ArrayList<>(prior.expectedTexts));
                            prefixActionTypes.add(prior.actionType);
                            prefixExpectations.add(prior);
                        }
                    }
                    PrefixRunResult sourceRun = runPrefixBranch(
                            trainer, job, args,
                            prefix, prefixTexts, prefixActionTypes, prefixExpectations, expected);
                    capturedCheckpoints.addAll(sourceRun.checkpoints);
                    records.add(probeCheckpointTarget(job, expected, sourceRun, args));
                }
                completed++;
                if (completed % Math.max(1, args.reportEvery) == 0) {
                    writeCheckpointBranchRecords(outDir.resolve("checkpoint_branch_probe.csv"), records);
                    writeCapturedCheckpointManifest(outDir.resolve("captured_checkpoints.csv"), capturedCheckpoints);
                    writeCheckpointBranchReadme(outDir.resolve("README.md"), args, records,
                            completed, groupList.size(), System.currentTimeMillis() - started);
                    System.out.println("checkpointBranchProbe=" + completed + "/" + groupList.size()
                            + " records=" + records.size());
                }
            }
            writeCheckpointBranchRecords(outDir.resolve("checkpoint_branch_probe.csv"), records);
            writeCapturedCheckpointManifest(outDir.resolve("captured_checkpoints.csv"), capturedCheckpoints);
            writeCheckpointBranchReadme(outDir.resolve("README.md"), args, records,
                    groupList.size(), groupList.size(), System.currentTimeMillis() - started);
            System.out.println("Checkpoint branch probe output: " + outDir);
            System.out.println("checkpointBranchRecords=" + records.size());
        } finally {
            args.forcedPrefixReplay = previousForcedPrefixReplay;
        }
    }

    private static CheckpointBranchRecord probeCheckpointTarget(
            ScenarioJob job,
            ReplayExpectation expected,
            PrefixRunResult sourceRun,
            Args args
    ) {
        EngineDecisionCheckpoint raw = checkpointForExpected(sourceRun.checkpoints, expected);
        if (raw == null) {
            return CheckpointBranchRecord.noCheckpoint(job, expected, sourceRun);
        }
        DecisionPoint sourcePoint = raw.ordinal >= 0 && raw.ordinal < sourceRun.decisions.size()
                ? sourceRun.decisions.get(raw.ordinal)
                : null;
        EngineDecisionCheckpoint checkpoint = raw.withSourceDecision(sourcePoint, expected);
        boolean sourceChoiceMatchesExpected = checkpointSourceChoiceMatchesExpected(checkpoint, expected);

        CheckpointContinuationResult sourceChoiceA = runCheckpointContinuation(
                checkpoint, args, checkpoint.sourceChosenIndices, checkpoint.sourceChosenTexts,
                true, "source_choice_a");
        CheckpointContinuationResult sourceChoiceB = runCheckpointContinuation(
                checkpoint, args, checkpoint.sourceChosenIndices, checkpoint.sourceChosenTexts,
                true, "source_choice_b");
        boolean sourceChoiceReentryMatched = sourceChoiceA.reentryMatches(checkpoint)
                && sourceChoiceB.reentryMatches(checkpoint);

        CheckpointContinuationResult sourceTerminal = CheckpointContinuationResult.skipped("source_choice_reentry_failed");
        CheckpointContinuationResult alternateTerminal = CheckpointContinuationResult.skipped("source_choice_reentry_failed");
        int alternateIndex = alternateIndex(checkpoint);
        if (sourceChoiceReentryMatched && sourceChoiceMatchesExpected) {
            sourceTerminal = runCheckpointContinuation(
                    checkpoint, args, checkpoint.sourceChosenIndices, checkpoint.sourceChosenTexts,
                    false, "source_terminal");
            if (alternateIndex >= 0) {
                alternateTerminal = runCheckpointContinuation(
                        checkpoint, args, Collections.singletonList(alternateIndex),
                        Collections.singletonList(checkpoint.candidateText(alternateIndex)),
                        false, "alternate_terminal");
            } else {
                alternateTerminal = CheckpointContinuationResult.skipped("no_alternate_candidate");
            }
        }
        return new CheckpointBranchRecord(
                job,
                expected,
                checkpoint,
                sourceRun,
                sourceChoiceMatchesExpected,
                sourceChoiceReentryMatched,
                sourceChoiceA,
                sourceChoiceB,
                sourceTerminal,
                alternateIndex,
                alternateTerminal);
    }

    private static EngineDecisionCheckpoint checkpointForExpected(
            List<EngineDecisionCheckpoint> checkpoints,
            ReplayExpectation expected
    ) {
        if (checkpoints == null) {
            return null;
        }
        EngineDecisionCheckpoint ordinalFallback = null;
        for (EngineDecisionCheckpoint checkpoint : checkpoints) {
            if (checkpoint == null) {
                continue;
            }
            if (expected != null && checkpoint.ordinal == expected.ordinal) {
                ordinalFallback = checkpoint;
            }
            EngineDecisionCheckpoint sourceAnnotated = checkpoint.withSourceDecision(null, expected);
            if (checkpointSourceChoiceMatchesExpected(sourceAnnotated, expected)) {
                return checkpoint;
            }
        }
        if (expected == null || expected.sourceCandidateTexts.isEmpty()) {
            return ordinalFallback;
        }
        return null;
    }

    private static boolean checkpointSourceChoiceMatchesExpected(
            EngineDecisionCheckpoint checkpoint,
            ReplayExpectation expected
    ) {
        if (checkpoint == null || expected == null) {
            return false;
        }
        if (checkpoint.actionType != expected.actionType) {
            return false;
        }
        if (!checkpoint.sourceChosenIndices.equals(expected.expectedIndices)) {
            return false;
        }
        if (!expected.expectedTexts.isEmpty()
                && !textListsMatchExpected(expected.expectedTexts, checkpoint.sourceChosenTexts)) {
            return false;
        }
        return expected.sourceCandidateTexts.isEmpty()
                || textListsMatchExpected(expected.sourceCandidateTexts, checkpoint.candidateTexts);
    }

    private static int alternateIndex(EngineDecisionCheckpoint checkpoint) {
        if (checkpoint == null || checkpoint.candidateTexts.size() < 2) {
            return -1;
        }
        Set<Integer> source = new HashSet<>(checkpoint.sourceChosenIndices);
        for (int i = 0; i < checkpoint.candidateTexts.size(); i++) {
            if (!source.contains(i)) {
                return i;
            }
        }
        return -1;
    }

    private static CheckpointContinuationResult runCheckpointContinuation(
            EngineDecisionCheckpoint checkpoint,
            Args args,
            List<Integer> forcedIndices,
            List<String> forcedTexts,
            boolean stopAtReentry,
            String label
    ) {
        if (checkpoint == null) {
            return CheckpointContinuationResult.skipped("missing_checkpoint");
        }
        RandomUtil.State previousRandomState = RandomUtil.captureState();
        Game game = null;
        ActionPlayer player = null;
        try {
            RandomUtil.restoreState(checkpoint.randomState);
            setReplayTraceContext(checkpoint.job, "checkpoint_branch_probe_" + label);
            game = checkpoint.copyGame();
            Player copied = game.getPlayer(checkpoint.playerId);
            if (!(copied instanceof ActionPlayer)) {
                String copiedType = copied == null ? "null" : copied.getClass().getName();
                return CheckpointContinuationResult.failed(
                        "checkpoint_player_copy_type_mismatch:" + copiedType, false, null, game);
            }
            player = (ActionPlayer) copied;
            player.setCheckpointForcedChoice(checkpoint.ordinal, forcedIndices, forcedTexts, stopAtReentry);
            try {
                resumeGameInGameThread(game, args.timeoutSec + 5);
            } catch (Throwable t) {
                if (!containsCheckpointProbeTerminated(t)) {
                    throw t;
                }
            }
            List<DecisionPoint> points = buildDecisionPoints(
                    player.getTrainingBuffer(),
                    player.getCandidateTextsByOrdinal(),
                    player.getStateSnapshotsByOrdinal(),
                    args);
            Float terminalValue = terminalValueFor(game, player);
            boolean timedOut = false;
            return new CheckpointContinuationResult(
                    label,
                    player.getLastCheckpointReentryProbe(),
                    points,
                    terminalValue,
                    timedOut,
                    "",
                    StateSnapshot.capture(game, player),
                    safeTurn(game));
        } catch (Throwable t) {
            boolean timedOut = String.valueOf(t.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            Float terminalValue = terminalValueFor(game, player);
            return new CheckpointContinuationResult(
                    label,
                    player == null ? null : player.getLastCheckpointReentryProbe(),
                    Collections.emptyList(),
                    terminalValue,
                    timedOut,
                    exceptionSummary(asException(t)),
                    StateSnapshot.capture(game, player),
                    game == null ? -1 : safeTurn(game));
        } finally {
            cleanup(game, null);
            RandomUtil.restoreState(previousRandomState);
        }
    }

    private static ReplayGroupResult replayGroup(RLTrainer trainer, ReplayGroup group, Args args) {
        ScenarioJob job = new ScenarioJob(group.scenario, group.agentDeck, group.oppDeck, group.seed,
                group.agentOpeningHandNames.isEmpty() ? args.agentOpeningHandNames : group.agentOpeningHandNames,
                group.agentOpeningLibraryNames,
                group.oppOpeningHandNames.isEmpty() ? args.oppOpeningHandNames : group.oppOpeningHandNames);
        if (args.forcedPrefixReplay) {
            return replayGroupWithForcedPrefixes(trainer, job, group, args);
        }
        PrefixRunResult run = runPrefixBranch(trainer, job, args, Collections.emptyList(), Collections.emptyList());
        List<ReplayRecord> out = new ArrayList<>();
        for (ReplayExpectation expected : replayTargets(group)) {
            DecisionPoint point = expected.ordinal >= 0 && expected.ordinal < run.decisions.size()
                    ? run.decisions.get(expected.ordinal)
                    : null;
            out.add(ReplayRecord.from(expected, point, run));
        }
        List<TrainingExample> deviation = Collections.emptyList();
        if (args.replayDeviationTrainingDataFile != null && !run.won) {
            TrainingExample example = buildFirstDeviationTrainingExample(job, group, run, args);
            if (example != null) {
                deviation = Collections.singletonList(example);
            }
        }
        List<TrainingExample> dagger = args.replayDaggerTrainingDataFile == null
                ? Collections.emptyList()
                : buildReplayDaggerTrainingExamples(job, group, run, args);
        return new ReplayGroupResult(out, Collections.emptyList(), deviation, dagger);
    }

    private static ReplayGroupResult replayGroupWithForcedPrefixes(
            RLTrainer trainer,
            ScenarioJob job,
            ReplayGroup group,
            Args args
    ) {
        Map<Integer, ReplayExpectation> byOrdinal = new HashMap<>();
        for (ReplayExpectation expected : group.expectations) {
            byOrdinal.put(expected.ordinal, expected);
        }
        List<ReplayRecord> out = new ArrayList<>();
        List<PrefixTraceRecord> traces = new ArrayList<>();
        for (ReplayExpectation expected : replayTargets(group)) {
            List<List<Integer>> prefix = new ArrayList<>();
            List<List<String>> prefixTexts = new ArrayList<>();
            List<StateSequenceBuilder.ActionType> prefixActionTypes = new ArrayList<>();
            List<ReplayExpectation> prefixExpectations = new ArrayList<>();
            for (int ordinal = 0; ordinal < expected.ordinal; ordinal++) {
                ReplayExpectation prior = byOrdinal.get(ordinal);
                if (prior == null || prior.expectedIndices.isEmpty()) {
                    prefix.add(Collections.emptyList());
                    prefixTexts.add(Collections.emptyList());
                    prefixActionTypes.add(null);
                    prefixExpectations.add(null);
                } else {
                    prefix.add(new ArrayList<>(prior.expectedIndices));
                    prefixTexts.add(prior.expectedTexts == null
                            ? Collections.emptyList()
                            : new ArrayList<>(prior.expectedTexts));
                    prefixActionTypes.add(prior.actionType);
                    prefixExpectations.add(prior);
                }
            }
            PrefixRunResult run = runPrefixBranch(trainer, job, args,
                    prefix, prefixTexts, prefixActionTypes, prefixExpectations, expected);
            DecisionPoint point = expected.ordinal >= 0 && expected.ordinal < run.decisions.size()
                    ? run.decisions.get(expected.ordinal)
                    : null;
            out.add(ReplayRecord.from(expected, point, run));
            traces.addAll(PrefixTraceRecord.from(job, expected, prefixExpectations, run));
        }
        return new ReplayGroupResult(out, traces, Collections.emptyList(), Collections.emptyList());
    }

    private static List<ReplayExpectation> replayTargets(ReplayGroup group) {
        if (group == null || group.expectations.isEmpty()) {
            return Collections.emptyList();
        }
        List<ReplayExpectation> targets = group.expectations.stream()
                .filter(e -> e.replayTarget)
                .collect(Collectors.toList());
        return targets.isEmpty() ? group.expectations : targets;
    }

    private static TrainingExample buildFirstDeviationTrainingExample(
            ScenarioJob job,
            ReplayGroup group,
            PrefixRunResult run,
            Args args
    ) {
        if (group == null || group.expectations.isEmpty() || run == null || run.decisions.isEmpty()) {
            return null;
        }
        List<ReplayExpectation> ordered = new ArrayList<>(replayTargets(group));
        ordered.sort(Comparator.comparingInt((ReplayExpectation e) -> e.ordinal));
        for (ReplayExpectation expected : ordered) {
            DecisionPoint point = expected.ordinal >= 0 && expected.ordinal < run.decisions.size()
                    ? run.decisions.get(expected.ordinal)
                    : null;
            if (point == null || point.trainingData.actionType != expected.actionType) {
                return null;
            }
            if (expectationChoiceMatchesPoint(expected, point)) {
                continue;
            }
            List<Integer> targetChoice = remapExpectedChoice(expected, point);
            targetChoice = sanitizeChoice(targetChoice, point.trainingData);
            if (targetChoice.isEmpty()) {
                return null;
            }
            if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
                return null;
            }
            String targetText = point.candidateText(targetChoice.get(0));
            if (args.skipPassTraining && isPassText(targetText)) {
                return null;
            }
            if (args.skipBlankTraining && targetText.trim().isEmpty()) {
                return null;
            }
            float[] target = rankDistribution(targetChoice);
            StateSequenceBuilder.TrainingData td = targetChoice.size() == 1
                    ? copyWithTarget(point.trainingData, targetChoice.get(0), target)
                    : copyWithFullChoiceTarget(point.trainingData, targetChoice, target);
            return new TrainingExample(job, point, td, 1, true);
        }
        return null;
    }

    private static List<TrainingExample> buildReplayDaggerTrainingExamples(
            ScenarioJob job,
            ReplayGroup group,
            PrefixRunResult run,
            Args args
    ) {
        if (group == null || group.expectations.isEmpty() || run == null || run.decisions.isEmpty()) {
            return Collections.emptyList();
        }
        List<ReplayExpectation> ordered = new ArrayList<>(replayTargets(group));
        ordered.sort(Comparator.comparingInt((ReplayExpectation e) -> e.ordinal));
        List<TrainingExample> out = new ArrayList<>();
        for (ReplayExpectation expected : ordered) {
            DecisionPoint point = expected.ordinal >= 0 && expected.ordinal < run.decisions.size()
                    ? run.decisions.get(expected.ordinal)
                    : null;
            if (point == null || point.trainingData.actionType != expected.actionType) {
                break;
            }
            if (!run.won && !expectationChoiceMatchesPoint(expected, point)) {
                List<Integer> targetChoice = remapExpectedChoice(expected, point);
                TrainingExample repair = buildChoiceTargetTrainingExample(job, point, targetChoice, args);
                if (repair != null) {
                    int repeat = Math.max(1, args.replayDeviationRepeat);
                    for (int i = 0; i < repeat; i++) {
                        out.add(repair);
                    }
                }
                break;
            }
            TrainingExample anchor = buildChoiceTargetTrainingExample(job, point, point.chosenIndices, args);
            if (anchor != null) {
                out.add(anchor);
            }
        }
        return out;
    }

    private static TrainingExample buildChoiceTargetTrainingExample(
            ScenarioJob job,
            DecisionPoint point,
            List<Integer> rawChoice,
            Args args
    ) {
        if (job == null || point == null) {
            return null;
        }
        List<Integer> targetChoice = sanitizeChoice(rawChoice, point.trainingData);
        if (targetChoice.isEmpty()) {
            return null;
        }
        if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
            return null;
        }
        String targetText = point.candidateText(targetChoice.get(0));
        if (args.skipPassTraining && isPassText(targetText)) {
            return null;
        }
        if (args.skipBlankTraining && targetText.trim().isEmpty()) {
            return null;
        }
        float[] target = rankDistribution(targetChoice);
        StateSequenceBuilder.TrainingData td = targetChoice.size() == 1
                ? copyWithTarget(point.trainingData, targetChoice.get(0), target)
                : copyWithFullChoiceTarget(point.trainingData, targetChoice, target);
        return new TrainingExample(job, point, td, 1, true);
    }

    private static boolean expectationChoiceMatchesPoint(ReplayExpectation expected, DecisionPoint point) {
        if (expected == null || point == null) {
            return false;
        }
        List<Integer> actualIndices = new ArrayList<>(point.chosenIndices);
        if (expected.expectedTexts != null && !expected.expectedTexts.isEmpty()) {
            return textListsMatchExpected(expected.expectedTexts, candidateTexts(point, actualIndices), point);
        }
        return actualIndices.equals(expected.expectedIndices);
    }

    private static List<Integer> remapExpectedChoice(ReplayExpectation expected, DecisionPoint point) {
        if (expected == null || point == null) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        int requestedCount = expected.expectedIndices == null ? 0 : expected.expectedIndices.size();
        int textCount = expected.expectedTexts == null ? 0 : expected.expectedTexts.size();
        int ranks = Math.max(requestedCount, textCount);
        int max = Math.min(point.trainingData.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        for (int rank = 0; rank < ranks; rank++) {
            Integer idx = rank < requestedCount ? expected.expectedIndices.get(rank) : null;
            String expectedText = rank < textCount ? normalizeText(expected.expectedTexts.get(rank)) : "";
            if (idx != null && idx >= 0 && idx < max && !out.contains(idx)) {
                String actualText = normalizeText(point.candidateText(idx));
                if (expectedText.isEmpty() || expectedText.equals(actualText)) {
                    out.add(idx);
                    continue;
                }
            }
            if (!expectedText.isEmpty()) {
                int byText = findCandidateByNormalizedText(point, expectedText, out);
                if (byText >= 0) {
                    out.add(byText);
                    continue;
                }
            }
            return Collections.emptyList();
        }
        return out;
    }

    private static int findCandidateByNormalizedText(DecisionPoint point, String normalizedText, List<Integer> used) {
        if (point == null || normalizedText == null || normalizedText.isEmpty()) {
            return -1;
        }
        int max = Math.min(point.trainingData.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        for (int i = 0; i < max; i++) {
            if (used != null && used.contains(i)) {
                continue;
            }
            if (point.trainingData.candidateMask[i] == 0) {
                continue;
            }
            if (textMatchesExpected(normalizedText, point.candidateText(i), point)) {
                return i;
            }
        }
        return -1;
    }

    private static List<String> candidateTexts(DecisionPoint point, List<Integer> indices) {
        List<String> out = new ArrayList<>();
        if (point == null || indices == null) {
            return out;
        }
        for (Integer idx : indices) {
            out.add(point.candidateText(idx == null ? -1 : idx));
        }
        return out;
    }

    private static List<String> normalizedTexts(List<String> texts) {
        if (texts == null) {
            return Collections.emptyList();
        }
        return texts.stream().map(ActionCounterfactualTrainer::normalizeText).collect(Collectors.toList());
    }

    private static boolean textListsMatchExpected(List<String> expectedTexts, List<String> actualTexts) {
        if (expectedTexts == null || actualTexts == null || expectedTexts.size() != actualTexts.size()) {
            return false;
        }
        for (int i = 0; i < expectedTexts.size(); i++) {
            if (!textMatchesExpected(expectedTexts.get(i), actualTexts.get(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean textListsMatchExpected(
            List<String> expectedTexts,
            List<String> actualTexts,
            DecisionPoint point
    ) {
        if (expectedTexts == null || actualTexts == null || expectedTexts.size() != actualTexts.size()) {
            return false;
        }
        for (int i = 0; i < expectedTexts.size(); i++) {
            if (!textMatchesExpected(expectedTexts.get(i), actualTexts.get(i), point)) {
                return false;
            }
        }
        return true;
    }

    private static BaselineResult runBaseline(RLTrainer trainer, ScenarioJob job, Args args) {
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("ActionCFBaseline", "ActionCFBaseline", false));
        Game game = null;
        ActionPlayer rlPlayer = null;
        Player opponent = null;
        try {
            Deck agentBase = trainer.loadDeckFresh(job.agentDeck.toString());
            Deck oppBase = trainer.loadDeckFresh(job.oppDeck.toString());
            if (agentBase == null || oppBase == null) {
                return BaselineResult.failed("deck_load_failed");
            }
            Deck agentDeck = shuffledCopy(agentBase, job.seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, job.seed ^ 0xC0FFEE1234L);
            stackOpeningState(agentDeck, job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            stackOpeningHand(oppDeck, job.oppOpeningHandNames);
            seedReplayRandomUtil(args, job.seed);
            setReplayTraceContext(job, "action_counterfactual_baseline");

            match.startGame();
            game = match.getGames().get(0);
            rlPlayer = new ActionPlayer("ACF-Base", args.targetTypes,
                    Collections.emptyList(), Collections.emptyList(), -1, -1, Collections.emptyList(), true,
                    args.terminalMode, args.maxGameTurns, args.tacticAutopilot);
            rlPlayer.setForceKeepOpeningHand(args.replayFile != null && !job.agentOpeningHandNames.isEmpty());
            rlPlayer.setExactFirstPriorityOpeningState(job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = ReplayOpponentDecisionPlayer.create(
                        "ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill, job.scenario, job.seed, null);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("ACF-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, true, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);

            List<DecisionPoint> points = buildDecisionPoints(
                    rlPlayer.getTrainingBuffer(),
                    rlPlayer.getCandidateTextsByOrdinal(),
                    rlPlayer.getStateSnapshotsByOrdinal(),
                    args);
            List<List<Integer>> prefix = points.stream()
                    .map(p -> new ArrayList<>(p.chosenIndices))
                    .collect(Collectors.toList());
            List<List<String>> prefixTexts = points.stream()
                    .map(p -> candidateTexts(p, p.chosenIndices))
                    .collect(Collectors.toList());
            boolean won = terminalSuccess(game, rlPlayer, args);
            return new BaselineResult(points, prefix, prefixTexts, won, false, "", safeTurn(game));
        } catch (Exception e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            List<DecisionPoint> points = Collections.emptyList();
            if (timedOut && rlPlayer != null) {
                try {
                    points = buildDecisionPoints(
                            rlPlayer.getTrainingBuffer(),
                            rlPlayer.getCandidateTextsByOrdinal(),
                            rlPlayer.getStateSnapshotsByOrdinal(),
                            args);
                } catch (Exception ignored) {
                    points = Collections.emptyList();
                }
            }
            List<List<Integer>> prefix = points.stream()
                    .map(p -> new ArrayList<>(p.chosenIndices))
                    .collect(Collectors.toList());
            List<List<String>> prefixTexts = points.stream()
                    .map(p -> candidateTexts(p, p.chosenIndices))
                    .collect(Collectors.toList());
            return new BaselineResult(points, prefix, prefixTexts, false, timedOut,
                    exceptionSummary(e), game == null ? -1 : safeTurn(game));
        } finally {
            cleanup(game, match);
        }
    }

    private static PrefixRunResult runPrefixBranch(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> prefixChoices,
            List<List<String>> prefixChoiceTexts
    ) {
        return runPrefixBranch(trainer, job, args, prefixChoices, prefixChoiceTexts,
                Collections.emptyList(), Collections.emptyList());
    }

    private static PrefixRunResult runPrefixBranch(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> prefixChoices,
            List<List<String>> prefixChoiceTexts,
            List<StateSequenceBuilder.ActionType> prefixActionTypes
    ) {
        return runPrefixBranch(trainer, job, args, prefixChoices, prefixChoiceTexts,
                prefixActionTypes, Collections.emptyList());
    }

    private static PrefixRunResult runPrefixBranch(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> prefixChoices,
            List<List<String>> prefixChoiceTexts,
            List<StateSequenceBuilder.ActionType> prefixActionTypes,
            List<ReplayExpectation> prefixExpectations
    ) {
        return runPrefixBranch(trainer, job, args, prefixChoices, prefixChoiceTexts,
                prefixActionTypes, prefixExpectations, null);
    }

    private static PrefixRunResult runPrefixBranch(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> prefixChoices,
            List<List<String>> prefixChoiceTexts,
            List<StateSequenceBuilder.ActionType> prefixActionTypes,
            List<ReplayExpectation> prefixExpectations,
            ReplayExpectation liveTraceTarget
    ) {
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("ActionCFPrefix", "ActionCFPrefix", false));
        Game game = null;
        ActionPlayer rlPlayer = null;
        Player opponent = null;
        List<EngineDecisionCheckpoint> checkpointSink = args.checkpointBranchProbe
                ? new ArrayList<>()
                : Collections.emptyList();
        try {
            Deck agentBase = trainer.loadDeckFresh(job.agentDeck.toString());
            Deck oppBase = trainer.loadDeckFresh(job.oppDeck.toString());
            if (agentBase == null || oppBase == null) {
                return PrefixRunResult.failed("deck_load_failed");
            }
            appendReplayStartupDiagnostic("prefix", "base_loaded", job, agentBase, oppBase, null, null, null);
            Deck agentDeck = shuffledCopy(agentBase, job.seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, job.seed ^ 0xC0FFEE1234L);
            appendReplayStartupDiagnostic("prefix", "shuffled_copy", job, agentDeck, oppDeck, null, null, null);
            stackOpeningState(agentDeck, job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            stackOpeningHand(oppDeck, job.oppOpeningHandNames);
            appendReplayStartupDiagnostic("prefix", "opening_state_stacked", job, agentDeck, oppDeck, null, null, null);
            seedReplayRandomUtil(args, job.seed);
            setReplayTraceContext(job, "action_counterfactual_prefix");

            match.startGame();
            game = match.getGames().get(0);
            rlPlayer = new ActionPlayer("ACF-Prefix", args.targetTypes,
                    prefixChoices, prefixChoiceTexts, -1, -1, Collections.emptyList(), true,
                    args.terminalMode, args.maxGameTurns, args.tacticAutopilot);
            rlPlayer.setForceKeepOpeningHand(args.replayFile != null && !job.agentOpeningHandNames.isEmpty());
            rlPlayer.setExactFirstPriorityOpeningState(job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            rlPlayer.setPrefixActionTypes(prefixActionTypes);
            rlPlayer.setPrefixExpectations(prefixExpectations);
            if (args.checkpointBranchProbe) {
                rlPlayer.setCheckpointCaptureSink(job, checkpointSink);
            }
            OpponentTranscriptCursor opponentTranscript = args.opponentTranscriptCursor(job);
            rlPlayer.setLivePrefixTrace(job, liveTraceTarget, args.livePrefixTracePath);
            rlPlayer.setOpponentTranscriptCursor(opponentTranscript);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = args.forcedPrefixReplay
                        ? args.forceOpponentTranscript
                        ? new TranscriptReplayingComputerPlayer7(
                                "ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill, opponentTranscript)
                        : new StackResolvingComputerPlayer7("ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill)
                        : ReplayOpponentDecisionPlayer.create(
                                "ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill, job.scenario, job.seed, null);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("ACF-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, true, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            appendReplayStartupDiagnostic("prefix", "players_added", job, agentDeck, oppDeck, rlPlayer, opponent, game);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            appendReplayStartupDiagnostic("prefix", "cards_loaded", job, agentDeck, oppDeck, rlPlayer, opponent, game);
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);
            appendReplayStartupDiagnostic("prefix", "library_forced", job, agentDeck, oppDeck, rlPlayer, opponent, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);
            appendReplayStartupDiagnostic("prefix", "before_start", job, agentDeck, oppDeck, rlPlayer, opponent, game);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);
            appendReplayStartupDiagnostic("prefix", "after_start", job, agentDeck, oppDeck, rlPlayer, opponent, game);
            List<DecisionPoint> points = buildDecisionPoints(
                    rlPlayer.getTrainingBuffer(),
                    rlPlayer.getCandidateTextsByOrdinal(),
                    rlPlayer.getStateSnapshotsByOrdinal(),
                    args);
            boolean won = terminalSuccess(game, rlPlayer, args);
            StateSnapshot finalState = StateSnapshot.capture(game, rlPlayer);
            return new PrefixRunResult(points, won, false, "",
                    rlPlayer.getForcedPrefixCount(), safeTurn(game),
                    rlPlayer.getFirstMulliganHandText(), rlPlayer.getFirstPriorityHandText(),
                    finalState, rlPlayer.prefixDivergenceOrEnd(points.size(), finalState),
                    checkpointSink);
        } catch (Exception e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            List<DecisionPoint> points = Collections.emptyList();
            if (timedOut && rlPlayer != null) {
                try {
                    points = buildDecisionPoints(
                            rlPlayer.getTrainingBuffer(),
                            rlPlayer.getCandidateTextsByOrdinal(),
                            rlPlayer.getStateSnapshotsByOrdinal(),
                            args);
                } catch (Exception ignored) {
                    points = Collections.emptyList();
                }
            }
            boolean won = !timedOut && terminalSuccess(game, rlPlayer, args);
            StateSnapshot finalState = StateSnapshot.capture(game, rlPlayer);
            return new PrefixRunResult(points, won, timedOut,
                    exceptionSummary(e),
                    rlPlayer == null ? 0 : rlPlayer.getForcedPrefixCount(),
                    game == null ? -1 : safeTurn(game),
                    rlPlayer == null ? "" : rlPlayer.getFirstMulliganHandText(),
                    rlPlayer == null ? "" : rlPlayer.getFirstPriorityHandText(),
                    finalState,
                    rlPlayer == null ? null : rlPlayer.prefixDivergenceOrEnd(points.size(), finalState),
                    checkpointSink);
        } finally {
            cleanup(game, match);
        }
    }

    private static BranchResult runForcedBranch(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> baselinePrefix,
            List<List<String>> baselinePrefixTexts,
            int targetOrdinal,
            int forcedIdx,
            List<String> forcedChoiceTexts
    ) {
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("ActionCFBranch", "ActionCFBranch", false));
        Game game = null;
        ActionPlayer rlPlayer = null;
        Player opponent = null;
        try {
            Deck agentBase = trainer.loadDeckFresh(job.agentDeck.toString());
            Deck oppBase = trainer.loadDeckFresh(job.oppDeck.toString());
            if (agentBase == null || oppBase == null) {
                return BranchResult.failed(forcedIdx, "deck_load_failed", false);
            }
            Deck agentDeck = shuffledCopy(agentBase, job.seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, job.seed ^ 0xC0FFEE1234L);
            stackOpeningState(agentDeck, job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            stackOpeningHand(oppDeck, job.oppOpeningHandNames);

            match.startGame();
            game = match.getGames().get(0);
            rlPlayer = new ActionPlayer("ACF-Branch-" + forcedIdx, args.targetTypes,
                    baselinePrefix, baselinePrefixTexts, targetOrdinal, forcedIdx, forcedChoiceTexts, true,
                    args.terminalMode, args.maxGameTurns, args.tacticAutopilot);
            rlPlayer.setForceKeepOpeningHand(args.replayFile != null && !job.agentOpeningHandNames.isEmpty());
            rlPlayer.setExactFirstPriorityOpeningState(job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = ReplayOpponentDecisionPlayer.create(
                        "ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill, job.scenario, job.seed, null);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("ACF-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, true, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);
            boolean won = terminalSuccess(game, rlPlayer, args);
            BranchTrajectoryBuildResult trajectory = buildBranchTrajectoryResult(
                    job, targetOrdinal, forcedIdx, won, false, rlPlayer, args);
            return new BranchResult(forcedIdx, won, false, "", rlPlayer.wasTargetForced(), safeTurn(game),
                    trajectory.episode, trajectory.rawRecords, trajectory.keptRecords, trajectory.dropReason);
        } catch (Exception e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            boolean won = !timedOut && terminalSuccess(game, rlPlayer, args);
            BranchTrajectoryBuildResult trajectory = buildBranchTrajectoryResult(
                    job, targetOrdinal, forcedIdx, won, timedOut, rlPlayer, args);
            return new BranchResult(forcedIdx, won, timedOut,
                    exceptionSummary(e),
                    rlPlayer != null && rlPlayer.wasTargetForced(),
                    game == null ? -1 : safeTurn(game),
                    trajectory.episode, trajectory.rawRecords, trajectory.keptRecords, trajectory.dropReason);
        } finally {
            cleanup(game, match);
        }
    }

    private static BranchTrajectoryBuildResult buildBranchTrajectoryResult(
            ScenarioJob job,
            int targetOrdinal,
            int forcedIdx,
            boolean won,
            boolean timedOut,
            ActionPlayer player,
            Args args
    ) {
        if (!args.branchTrajectoryMode) {
            return BranchTrajectoryBuildResult.empty("disabled");
        }
        if (timedOut) {
            return BranchTrajectoryBuildResult.empty("timed_out");
        }
        if (player == null) {
            return BranchTrajectoryBuildResult.empty("no_player");
        }
        if (!player.wasTargetForced()) {
            return BranchTrajectoryBuildResult.empty("target_not_forced");
        }
        List<StateSequenceBuilder.TrainingData> source = player.getTrainingBuffer();
        if (source == null || source.isEmpty()) {
            return BranchTrajectoryBuildResult.empty("no_training_records");
        }
        List<StateSequenceBuilder.TrainingData> records = new ArrayList<>();
        if (args.branchTrajectoryFirstPostTargetOnly) {
            for (int i = Math.max(0, targetOrdinal + 1); i < source.size(); i++) {
                StateSequenceBuilder.TrainingData td = source.get(i);
                if (td == null || !args.targetTypes.contains(td.actionType)) {
                    continue;
                }
                if (td.actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
                    continue;
                }
                if (args.skipPassTraining && isLikelyPassTarget(td)) {
                    continue;
                }
                records.add(td);
                break;
            }
        } else {
            for (StateSequenceBuilder.TrainingData td : source) {
                if (td == null || !args.targetTypes.contains(td.actionType)) {
                    continue;
                }
                if (args.skipPassTraining && isLikelyPassTarget(td)) {
                    continue;
                }
                records.add(td);
            }
        }
        if (records.isEmpty()) {
            return BranchTrajectoryBuildResult.dropped(source.size(), "filtered_all_records");
        }
        List<Double> rewards = new ArrayList<>(Collections.nCopies(records.size(), 0.0));
        rewards.set(rewards.size() - 1, won ? args.trajectoryFinalReward : -args.trajectoryFinalReward);
        String key = "branch|scenario=" + job.scenario
                + "|seed=" + job.seed
                + "|targetOrdinal=" + targetOrdinal
                + "|forcedIdx=" + forcedIdx
                + "|won=" + won;
        return BranchTrajectoryBuildResult.built(new TrajectoryTrainingEpisode(key, records, rewards),
                source.size(), records.size());
    }

    private static List<TrajectoryTrainingEpisode> buildBranchTrajectoryPairEpisodes(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> branchResults,
            Args args
    ) {
        if (!args.branchTrajectoryMode || !args.branchTrajectoryPairMode
                || point == null || branchResults == null || branchResults.isEmpty()) {
            return Collections.emptyList();
        }
        List<BranchResult> wins = new ArrayList<>();
        List<BranchResult> losses = new ArrayList<>();
        for (BranchResult result : branchResults) {
            if (result == null
                    || result.timedOut
                    || !result.error.isEmpty()
                    || result.trajectoryEpisode == null
                    || result.trajectoryEpisode.records.isEmpty()) {
                continue;
            }
            if (args.skipPassTraining && isPassText(point.candidateText(result.forcedIdx))) {
                continue;
            }
            if (args.skipBlankTraining && point.candidateText(result.forcedIdx).trim().isEmpty()) {
                continue;
            }
            if (result.won) {
                wins.add(result);
            } else {
                losses.add(result);
            }
        }
        if (wins.isEmpty() || losses.isEmpty()) {
            return Collections.emptyList();
        }

        BranchResult loss = null;
        int baselineIdx = point.baselineIdx();
        for (BranchResult candidate : losses) {
            if (candidate.forcedIdx == baselineIdx) {
                loss = candidate;
                break;
            }
        }
        if (loss == null) {
            loss = losses.get(0);
        }

        List<TrajectoryTrainingEpisode> out = new ArrayList<>();
        for (BranchResult win : wins) {
            StateSequenceBuilder.TrainingData winRecord = win.trajectoryEpisode.records.get(0);
            StateSequenceBuilder.TrainingData lossRecord = loss.trajectoryEpisode.records.get(0);
            if (winRecord == null || lossRecord == null) {
                continue;
            }
            List<StateSequenceBuilder.TrainingData> records = new ArrayList<>(2);
            records.add(winRecord);
            records.add(lossRecord);
            List<Double> rewards = new ArrayList<>(2);
            rewards.add(args.trajectoryFinalReward);
            rewards.add(-args.trajectoryFinalReward);
            String key = "branchPair|scenario=" + job.scenario
                    + "|seed=" + job.seed
                    + "|targetOrdinal=" + point.ordinal
                    + "|winIdx=" + win.forcedIdx
                    + "|lossIdx=" + loss.forcedIdx;
            out.add(new TrajectoryTrainingEpisode(key, records, rewards));
            break;
        }
        return out;
    }

    private static List<BranchValueProbeRecord> runBranchValueProbesForExample(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            BaselineResult baseline,
            DecisionPoint point,
            List<BranchResult> branchResults,
            TrainingExample example,
            long deadlineMs
    ) {
        if (point == null
                || point.trainingData == null
                || point.trainingData.actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
            return Collections.emptyList();
        }
        Map<Integer, BranchResult> byIdx = new HashMap<>();
        for (BranchResult result : branchResults) {
            byIdx.put(result.forcedIdx, result);
        }
        LinkedHashSet<Integer> probeIndices = new LinkedHashSet<>();
        probeIndices.add(point.baselineIdx());
        probeIndices.addAll(example.choiceIndices);
        List<BranchValueProbeRecord> out = new ArrayList<>();
        for (Integer idx : probeIndices) {
            if (idx == null || idx < 0 || idx >= point.trainingData.candidateCount) {
                continue;
            }
            if (scenarioDeadlineReached(deadlineMs)) {
                break;
            }
            BranchResult terminal = byIdx.get(idx);
            BranchValueProbeResult valueProbe = runForcedBranchValueProbe(
                    trainer, job, args,
                    baseline.prefixChoices, baseline.prefixChoiceTexts,
                    point.ordinal, idx, Collections.singletonList(point.candidateText(idx)));
            out.add(BranchValueProbeRecord.from(job, point, terminal, valueProbe));
        }
        return out;
    }

    private static BranchValueProbeResult runForcedBranchValueProbe(
            RLTrainer trainer,
            ScenarioJob job,
            Args args,
            List<List<Integer>> baselinePrefix,
            List<List<String>> baselinePrefixTexts,
            int targetOrdinal,
            int forcedIdx,
            List<String> forcedChoiceTexts
    ) {
        TwoPlayerMatch match = new TwoPlayerMatch(new MatchOptions("ActionCFValueProbe", "ActionCFValueProbe", false));
        Game game = null;
        ActionPlayer rlPlayer = null;
        Player opponent = null;
        try {
            Deck agentBase = trainer.loadDeckFresh(job.agentDeck.toString());
            Deck oppBase = trainer.loadDeckFresh(job.oppDeck.toString());
            if (agentBase == null || oppBase == null) {
                return BranchValueProbeResult.failed("deck_load_failed", false);
            }
            Deck agentDeck = shuffledCopy(agentBase, job.seed ^ 0x5DEECE66DL);
            Deck oppDeck = shuffledCopy(oppBase, job.seed ^ 0xC0FFEE1234L);
            stackOpeningState(agentDeck, job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            stackOpeningHand(oppDeck, job.oppOpeningHandNames);

            match.startGame();
            game = match.getGames().get(0);
            rlPlayer = new ActionPlayer("ACF-ValueProbe-" + forcedIdx, args.targetTypes,
                    baselinePrefix, baselinePrefixTexts, targetOrdinal, forcedIdx, forcedChoiceTexts, true,
                    args.terminalMode, args.maxGameTurns, args.tacticAutopilot, true);
            rlPlayer.setForceKeepOpeningHand(args.replayFile != null && !job.agentOpeningHandNames.isEmpty());
            rlPlayer.setExactFirstPriorityOpeningState(job.agentOpeningHandNames, job.agentOpeningLibraryNames);
            rlPlayer.setCurrentEpisode(-1);

            if ("cp7".equals(args.opponentMode)) {
                opponent = ReplayOpponentDecisionPlayer.create(
                        "ACF-CP7", RangeOfInfluence.ALL, args.cp7Skill, job.scenario, job.seed, null);
            } else {
                ComputerPlayerRL oppRl = new ComputerPlayerRL("ACF-OppRL", RangeOfInfluence.ALL,
                        RLTrainer.sharedModel, true, false, "train");
                oppRl.setCurrentEpisode(-1);
                opponent = oppRl;
            }

            game.addPlayer(rlPlayer, agentDeck);
            match.addPlayer(rlPlayer, agentDeck);
            game.addPlayer(opponent, oppDeck);
            match.addPlayer(opponent, oppDeck);
            game.loadCards(agentDeck.getCards(), rlPlayer.getId());
            game.loadCards(oppDeck.getCards(), opponent.getId());
            forceLibraryOrder(rlPlayer, agentDeck, game);
            forceLibraryOrder(opponent, oppDeck, game);

            GameOptions options = new GameOptions();
            options.rollbackTurnsAllowed = false;
            options.skipInitShuffling = true;
            game.setGameOptions(options);

            startGameInGameThread(game, rlPlayer.getId(), args.timeoutSec + 5);
            if (rlPlayer.wasValueProbeCaptured()) {
                return rlPlayer.valueProbeResult(false, "");
            }
            Float terminalValue = terminalValueFor(game, rlPlayer);
            boolean terminal = terminalValue != null;
            return new BranchValueProbeResult(false, terminal, terminalValue == null ? Float.NaN : terminalValue,
                    -1, StateSnapshot.capture(game, rlPlayer), "", rlPlayer.wasTargetForced(), false);
        } catch (Throwable e) {
            boolean timedOut = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT).contains("timeout");
            if (rlPlayer != null && rlPlayer.wasValueProbeCaptured() && containsValueProbeTerminated(e)) {
                return rlPlayer.valueProbeResult(false, "");
            }
            return BranchValueProbeResult.failed(exceptionSummary(asException(e)), timedOut);
        } finally {
            cleanup(game, match);
        }
    }

    private static boolean terminalSuccess(Game game, ActionPlayer player, Args args) {
        if (game == null || player == null) {
            return false;
        }
        if (player.turnLimitReached()) {
            return false;
        }
        if (args.terminalMode == TerminalMode.SPY_COMBO_MILESTONE_ONLY
                || args.terminalMode == TerminalMode.SPY_BALUSTRADE_REACHED) {
            return player.terminalMilestoneReached()
                    || terminalMilestoneReached(game, player.getId(), args.terminalMode);
        }
        if (args.terminalMode == TerminalMode.SPY_LANDLESS_COMBO_WIN) {
            return actualWin(game, player) && spyLandlessComboWinReached(game, player);
        }
        if (actualWin(game, player)) {
            return true;
        }
        if (args.terminalMode == TerminalMode.SPY_COMBO_MILESTONE) {
            return player.terminalMilestoneReached()
                    || terminalMilestoneReached(game, player.getId(), args.terminalMode);
        }
        return false;
    }

    private static Float terminalValueFor(Game game, Player player) {
        if (game == null || player == null) {
            return null;
        }
        try {
            if (player.hasWon()) {
                return 1.0f;
            }
            if (player.hasLost()) {
                return -1.0f;
            }
            if (game.hasEnded()) {
                String name = player.getName();
                return game.getWinner() != null && name != null && game.getWinner().contains(name) ? 1.0f : -1.0f;
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    private static boolean containsValueProbeTerminated(Throwable t) {
        Throwable cur = t;
        while (cur != null) {
            if (cur instanceof ValueProbeTerminated) {
                return true;
            }
            cur = cur.getCause();
        }
        return false;
    }

    private static boolean containsCheckpointProbeTerminated(Throwable t) {
        Throwable cur = t;
        while (cur != null) {
            if (cur instanceof CheckpointProbeTerminated) {
                return true;
            }
            cur = cur.getCause();
        }
        return false;
    }

    private static final class ValueProbeTerminated extends Error {
        static final ValueProbeTerminated INSTANCE = new ValueProbeTerminated();
        private ValueProbeTerminated() {
            super(null, null, false, false);
        }
    }

    private static final class CheckpointProbeTerminated extends Error {
        static final CheckpointProbeTerminated INSTANCE = new CheckpointProbeTerminated();
        private CheckpointProbeTerminated() {
            super(null, null, false, false);
        }
    }

    private static boolean terminalMilestoneReached(Game game, UUID playerId, TerminalMode mode) {
        if (mode == TerminalMode.SPY_BALUSTRADE_REACHED) {
            return controlledPermanentNamed(game, playerId, "Balustrade Spy")
                    || controlledStackObjectNamed(game, playerId, "Balustrade Spy");
        }
        if (mode == TerminalMode.SPY_COMBO_MILESTONE
                || mode == TerminalMode.SPY_COMBO_MILESTONE_ONLY) {
            return spyComboMilestoneReached(game, playerId);
        }
        return false;
    }

    private static boolean actualWin(Game game, Player player) {
        try {
            return game != null && player != null && game.getWinner().contains(player.getName());
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean spyComboMilestoneReached(Game game, UUID playerId) {
        if (game == null || playerId == null) {
            return false;
        }
        Player player = game.getPlayer(playerId);
        if (player == null || player.hasLost()) {
            return false;
        }
        try {
            for (UUID opponentId : game.getOpponents(playerId, true)) {
                Player opponent = game.getPlayer(opponentId);
                if (opponent != null && (opponent.hasLost() || opponent.getLife() <= 0)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }

        if (controlledPermanentNamed(game, playerId, "Lotleth Giant")
                || controlledStackObjectNamed(game, playerId, "Lotleth Giant")) {
            return true;
        }

        if (!controlledPermanentNamed(game, playerId, "Balustrade Spy")) {
            return false;
        }

        int graveyardCreatures = graveyardCreatureCount(player, game);
        int battlefieldCreatures = controlledCreatureCount(game, playerId);
        boolean graveyardHasDreadReturn = graveyardContains(player, game, "Dread Return");
        boolean graveyardHasLotleth = graveyardContains(player, game, "Lotleth Giant");

        return graveyardHasDreadReturn
                && graveyardHasLotleth
                && (graveyardCreatures >= 8 || battlefieldCreatures >= 3);
    }

    private static boolean spyLandlessComboWinReached(Game game, Player player) {
        if (game == null || player == null || player.hasLost()) {
            return false;
        }
        try {
            if (countLibraryTrueLands(player, game) > 0) {
                return false;
            }
            boolean lotlethResolvedOrPending =
                    controlledPermanentNamed(game, player.getId(), "Lotleth Giant")
                            || controlledStackObjectNamed(game, player.getId(), "Lotleth Giant")
                            || controlledStackAbilityContains(game, player.getId(), "Undergrowth");
            if (!lotlethResolvedOrPending) {
                return false;
            }
            return graveyardContains(player, game, "Dread Return")
                    || exileContains(player, game, "Dread Return")
                    || controlledStackObjectNamed(game, player.getId(), "Dread Return");
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean controlledPermanentNamed(Game game, UUID playerId, String name) {
        try {
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(playerId)) {
                if (nameEquals(permanent == null ? null : permanent.getName(), name)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static boolean controlledStackObjectNamed(Game game, UUID playerId, String name) {
        try {
            for (StackObject stackObject : game.getStack()) {
                if (stackObject != null
                        && stackObject.isControlledBy(playerId)
                        && nameEquals(stackObject.getName(), name)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static boolean controlledStackAbilityContains(Game game, UUID playerId, String text) {
        try {
            String needle = text == null ? "" : text.toLowerCase(Locale.ROOT);
            if (needle.isEmpty()) {
                return false;
            }
            for (StackObject stackObject : game.getStack()) {
                if (stackObject != null
                        && stackObject.isControlledBy(playerId)
                        && String.valueOf(stackObject.getName()).toLowerCase(Locale.ROOT).contains(needle)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static int graveyardCreatureCount(Player player, Game game) {
        int count = 0;
        try {
            for (Card card : player.getGraveyard().getCards(game)) {
                if (card != null && card.isCreature(game)) {
                    count++;
                }
            }
        } catch (Exception ignored) {
        }
        return count;
    }

    private static int controlledCreatureCount(Game game, UUID playerId) {
        int count = 0;
        try {
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(playerId)) {
                if (permanent != null && permanent.isCreature(game)) {
                    count++;
                }
            }
        } catch (Exception ignored) {
        }
        return count;
    }

    private static int opponentLife(Game game, UUID playerId) {
        int best = 20;
        try {
            for (UUID opponentId : game.getOpponents(playerId, true)) {
                Player opponent = game.getPlayer(opponentId);
                if (opponent != null) {
                    best = Math.min(best, opponent.getLife());
                }
            }
        } catch (Exception ignored) {
        }
        return best;
    }

    private static boolean graveyardContains(Player player, Game game, String name) {
        try {
            for (Card card : player.getGraveyard().getCards(game)) {
                if (card != null && nameEquals(card.getName(), name)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static boolean exileContains(Player player, Game game, String name) {
        try {
            for (Card card : game.getExile().getCardsOwned(game, player.getId())) {
                if (card != null && nameEquals(card.getName(), name)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static int countLibraryTrueLands(Player player, Game game) {
        int count = 0;
        try {
            for (Card card : player.getLibrary().getCards(game)) {
                if (card != null && card.isLand(game)) {
                    count++;
                }
            }
        } catch (Exception ignored) {
        }
        return count;
    }

    private static String stackNames(Game game, UUID playerId, int max) {
        List<String> names = new ArrayList<>();
        try {
            for (StackObject stackObject : game.getStack()) {
                if (stackObject != null && stackObject.isControlledBy(playerId)) {
                    names.add(stackObject.getName() == null ? "" : stackObject.getName());
                    if (names.size() >= max) {
                        break;
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return String.join("|", names);
    }

    private static boolean nameEquals(String actual, String expected) {
        return actual != null && expected != null && actual.equalsIgnoreCase(expected);
    }

    private static String exceptionSummary(Exception e) {
        if (e == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        Throwable current = e;
        int depth = 0;
        while (current != null && depth < 4) {
            if (sb.length() > 0) {
                sb.append(" <- ");
            }
            sb.append(current.getClass().getSimpleName()).append(": ").append(current.getMessage());
            current = current.getCause();
            depth++;
        }
        return sb.toString();
    }

    private static Exception asException(Throwable t) {
        if (t instanceof Exception) {
            return (Exception) t;
        }
        return new RuntimeException(t);
    }

    private static List<DecisionPoint> buildDecisionPoints(
            List<StateSequenceBuilder.TrainingData> data,
            List<List<String>> candidateTexts,
            List<StateSnapshot> stateSnapshots,
            Args args
    ) {
        List<DecisionPoint> out = new ArrayList<>();
        int ordinal = 0;
        for (StateSequenceBuilder.TrainingData td : data) {
            if (td == null || !args.targetTypes.contains(td.actionType)) {
                continue;
            }
            if (td.candidateCount < 2 || td.chosenCount < 1) {
                continue;
            }
            PolicyScoreResult score = scoreTrainingDataDetailed(td, args);
            float[] probs = score.policyScores;
            List<String> texts = ordinal < candidateTexts.size() ? candidateTexts.get(ordinal) : Collections.emptyList();
            StateSnapshot snapshot = stateSnapshots != null && ordinal < stateSnapshots.size()
                    ? stateSnapshots.get(ordinal)
                    : StateSnapshot.EMPTY;
            appendPolicyInputDump(args, ordinal, td, probs, texts, snapshot);
            appendPolicyInferenceProbe(args, ordinal, td, score, texts, snapshot);
            out.add(new DecisionPoint(ordinal, td, probs, texts, snapshot));
            ordinal++;
            if (out.size() >= args.maxDecisionDepth) {
                break;
            }
        }
        return out;
    }

    private static PolicyScoreResult scoreTrainingDataDetailed(StateSequenceBuilder.TrainingData td, Args args) {
        String policyKey = "train";
        String headId = headForActionType(td == null ? null : td.actionType);
        int pickIndex = 0;
        int minTargets = 1;
        int maxTargets = td == null ? 1 : Math.max(1, td.chosenCount);
        boolean captureProbeMetadata = args != null && args.policyInferenceProbe;
        String modelClass = captureProbeMetadata ? modelClassName() : "";
        String deviceInfo = captureProbeMetadata ? modelDeviceInfo() : "";
        String callerThreadName = Thread.currentThread().getName();
        long callerThreadId = Thread.currentThread().getId();
        if (args != null && args.noSearchModelScoring) {
            float[] uniform = uniformScores(td);
            return PolicyScoreResult.fallback(uniform, policyKey, headId, pickIndex, minTargets, maxTargets,
                    modelClass, deviceInfo, callerThreadName, callerThreadId, "no_search_model_scoring", "");
        }
        try {
            PythonMLBatchManager.PredictionResult result = RLTrainer.sharedModel.scoreCandidates(
                    td.state,
                    td.candidateActionIds,
                    td.candidateFeatures,
                    td.candidateMask,
                    policyKey,
                    headId,
                    pickIndex,
                    minTargets,
                    maxTargets
            );
            float[] scores = result.policyScores == null
                    ? new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES]
                    : Arrays.copyOf(result.policyScores, result.policyScores.length);
            return PolicyScoreResult.from(scores, result, policyKey, headId, pickIndex, minTargets, maxTargets,
                    modelClass, deviceInfo, callerThreadName, callerThreadId);
        } catch (Exception ignored) {
            float[] uniform = uniformScores(td);
            return PolicyScoreResult.fallback(uniform, policyKey, headId, pickIndex, minTargets, maxTargets,
                    modelClass, deviceInfo, callerThreadName, callerThreadId, "uniform_fallback",
                    exceptionSummary(ignored));
        }
    }

    private static float[] scoreTrainingData(StateSequenceBuilder.TrainingData td, Args args) {
        return scoreTrainingDataDetailed(td, args).policyScores;
    }

    private static float[] scoreTrainingData(StateSequenceBuilder.TrainingData td) {
        return scoreTrainingData(td, null);
    }

    private static String modelClassName() {
        try {
            return RLTrainer.sharedModel == null ? "" : RLTrainer.sharedModel.getClass().getName();
        } catch (Exception e) {
            return "unavailable:" + e.getClass().getSimpleName();
        }
    }

    private static String modelDeviceInfo() {
        try {
            return RLTrainer.sharedModel == null ? "" : safeText(RLTrainer.sharedModel.getDeviceInfo());
        } catch (Exception e) {
            return "unavailable:" + exceptionSummary(e);
        }
    }

    private static final class PolicyScoreResult {
        final float[] policyScores;
        final float[] rawScores;
        final float[] candidateQScores;
        final float valueScore;
        final String policyKey;
        final String headId;
        final int pickIndex;
        final int minTargets;
        final int maxTargets;
        final String modelClass;
        final String deviceInfo;
        final String backendPath;
        final String requestId;
        final String batchId;
        final int batchIndex;
        final int batchSize;
        final String callerThreadName;
        final long callerThreadId;
        final String backendThreadName;
        final String rawScoreKind;
        final boolean fallback;
        final String backendDetails;
        final String error;

        private PolicyScoreResult(
                float[] policyScores,
                float[] rawScores,
                float[] candidateQScores,
                float valueScore,
                String policyKey,
                String headId,
                int pickIndex,
                int minTargets,
                int maxTargets,
                String modelClass,
                String deviceInfo,
                String backendPath,
                String requestId,
                String batchId,
                int batchIndex,
                int batchSize,
                String callerThreadName,
                long callerThreadId,
                String backendThreadName,
                String rawScoreKind,
                boolean fallback,
                String backendDetails,
                String error
        ) {
            this.policyScores = policyScores == null ? new float[0] : Arrays.copyOf(policyScores, policyScores.length);
            this.rawScores = rawScores == null ? Arrays.copyOf(this.policyScores, this.policyScores.length)
                    : Arrays.copyOf(rawScores, rawScores.length);
            this.candidateQScores = candidateQScores == null ? new float[0] : Arrays.copyOf(candidateQScores, candidateQScores.length);
            this.valueScore = valueScore;
            this.policyKey = safeText(policyKey);
            this.headId = safeText(headId);
            this.pickIndex = pickIndex;
            this.minTargets = minTargets;
            this.maxTargets = maxTargets;
            this.modelClass = safeText(modelClass);
            this.deviceInfo = safeText(deviceInfo);
            this.backendPath = safeText(backendPath);
            this.requestId = safeText(requestId);
            this.batchId = safeText(batchId);
            this.batchIndex = batchIndex;
            this.batchSize = batchSize;
            this.callerThreadName = safeText(callerThreadName);
            this.callerThreadId = callerThreadId;
            this.backendThreadName = safeText(backendThreadName);
            this.rawScoreKind = safeText(rawScoreKind);
            this.fallback = fallback;
            this.backendDetails = safeText(backendDetails);
            this.error = safeText(error);
        }

        static PolicyScoreResult from(
                float[] scores,
                PythonMLBatchManager.PredictionResult result,
                String policyKey,
                String headId,
                int pickIndex,
                int minTargets,
                int maxTargets,
                String modelClass,
                String deviceInfo,
                String callerThreadName,
                long callerThreadId
        ) {
            return new PolicyScoreResult(
                    scores,
                    result == null ? scores : result.policyScores,
                    result == null ? null : result.candidateQScores,
                    result == null ? 0.0f : result.valueScores,
                    policyKey,
                    headId,
                    pickIndex,
                    minTargets,
                    maxTargets,
                    modelClass,
                    deviceInfo,
                    result == null ? "" : result.backendPath,
                    result == null ? "" : result.requestId,
                    result == null ? "" : result.batchId,
                    result == null ? -1 : result.batchIndex,
                    result == null ? -1 : result.batchSize,
                    callerThreadName,
                    callerThreadId,
                    result == null ? "" : result.backendThreadName,
                    result == null ? "policy_scores" : result.rawScoreKind,
                    result != null && result.fallback,
                    result == null ? "" : result.backendDetails,
                    ""
            );
        }

        static PolicyScoreResult fallback(
                float[] scores,
                String policyKey,
                String headId,
                int pickIndex,
                int minTargets,
                int maxTargets,
                String modelClass,
                String deviceInfo,
                String callerThreadName,
                long callerThreadId,
                String backendPath,
                String error
        ) {
            return new PolicyScoreResult(scores, scores, null, 0.0f, policyKey, headId, pickIndex, minTargets, maxTargets,
                    modelClass, deviceInfo, backendPath, "", "", -1, -1, callerThreadName, callerThreadId,
                    Thread.currentThread().getName(), "uniform_fallback", true, "", error);
        }
    }

    private static float[] uniformScores(StateSequenceBuilder.TrainingData td) {
        float[] fallback = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        if (td == null) {
            return fallback;
        }
        int valid = 0;
        for (int i = 0; i < Math.min(td.candidateCount, fallback.length); i++) {
            if (td.candidateMask[i] != 0) {
                valid++;
            }
        }
        float p = valid <= 0 ? 0.0f : 1.0f / valid;
        for (int i = 0; i < Math.min(td.candidateCount, fallback.length); i++) {
            fallback[i] = td.candidateMask[i] != 0 ? p : 0.0f;
        }
        return fallback;
    }

    private static List<Integer> branchCandidates(DecisionPoint point, Args args, long seed) {
        List<Integer> valid = validCandidateIndices(point);
        valid.sort((a, b) -> Float.compare(point.probs[b], point.probs[a]));
        LinkedHashSet<Integer> chosenSet = new LinkedHashSet<>();
        for (Integer idx : preferredBranchCandidates(point, valid, args)) {
            chosenSet.add(idx);
        }
        for (Integer idx : valid) {
            if (chosenSet.size() >= args.topK) {
                break;
            }
            chosenSet.add(idx);
        }
        for (Integer idx : point.chosenIndices) {
            if (idx != null && idx >= 0 && valid.contains(idx)) {
                chosenSet.add(idx);
            }
        }
        List<Integer> chosen = new ArrayList<>(chosenSet);
        if (args.randomExtra > 0 && chosen.size() < valid.size()) {
            List<Integer> rest = valid.stream().filter(i -> !chosen.contains(i)).collect(Collectors.toCollection(ArrayList::new));
            Collections.shuffle(rest, new Random(seed));
            for (Integer idx : rest) {
                if (chosen.size() >= args.topK + args.randomExtra) {
                    break;
                }
                chosen.add(idx);
            }
        }
        chosen.sort(Comparator
                .comparingInt((Integer idx) -> branchOrderPriority(point, idx, args))
                .thenComparing((Integer a, Integer b) -> Float.compare(point.probs[b], point.probs[a]))
                .thenComparingInt(Integer::intValue));
        return chosen;
    }

    private static int branchOrderPriority(DecisionPoint point, int idx, Args args) {
        int preferred = preferredBranchPriority(point, idx, args);
        if (preferred < Integer.MAX_VALUE) {
            return preferred;
        }
        return isPassText(point.candidateText(idx)) ? 100_000 : 50_000;
    }

    private static List<Integer> preferredBranchCandidates(DecisionPoint point, List<Integer> valid, Args args) {
        if (args != null && args.genericBranchOrder) {
            return Collections.emptyList();
        }
        List<Integer> preferred = new ArrayList<>();
        for (Integer idx : valid) {
            if (preferredBranchPriority(point, idx, args) < Integer.MAX_VALUE) {
                preferred.add(idx);
            }
        }
        preferred.sort(Comparator
                .comparingInt((Integer idx) -> preferredBranchPriority(point, idx, args))
                .thenComparing((Integer a, Integer b) -> Float.compare(point.probs[b], point.probs[a]))
                .thenComparingInt(Integer::intValue));
        return preferred;
    }

    private static int preferredBranchPriority(DecisionPoint point, int idx, Args args) {
        if (args != null && args.genericBranchOrder) {
            return Integer.MAX_VALUE;
        }
        return preferredTacticPriority(point.trainingData.actionType, point.candidateText(idx), point.stateSnapshot);
    }

    private static int preferredTacticPriority(
            StateSequenceBuilder.ActionType actionType,
            String candidateText,
            StateSnapshot snapshot
    ) {
        String text = normalizeText(candidateText);
        if (text.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        if (actionType == StateSequenceBuilder.ActionType.MULLIGAN && "keep".equals(text)) {
            return 0;
        }
        if (actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS) {
            String state = snapshot == null ? "" : snapshot.toCompactText();
            boolean lotlethTargetPrompt = stateContainsName(state, "battlefield", "Lotleth Giant")
                    || stateContainsName(state, "stack", "Lotleth Giant");
            boolean spyTargetPrompt = stateContainsName(state, "battlefield", "Balustrade Spy")
                    || stateContainsName(state, "stack", "Balustrade Spy");
            if (text.contains("lotleth giant")) {
                return 0;
            }
            if (isSelfTargetText(text)) {
                if (lotlethTargetPrompt) {
                    return 90_000;
                }
                return spyTargetPrompt ? 1 : 10;
            }
            if (isPlayerTargetText(text)) {
                if (lotlethTargetPrompt) {
                    return 1;
                }
                if (spyTargetPrompt) {
                    return 90_000;
                }
                return 20;
            }
            return Integer.MAX_VALUE;
        }
        if (text.contains("dread return: flashback")) {
            return dreadReturnPriority(snapshot == null ? "" : snapshot.toCompactText());
        }
        if (isPassText(candidateText) && shouldPreferSpyTriggerPass(snapshot == null ? "" : snapshot.toCompactText())) {
            return 5;
        }
        String state = snapshot == null ? "" : snapshot.toCompactText();
        int battlefieldCreatures = stateInt(state, "battlefieldCreatures", 0);
        int libraryTrueLands = stateInt(state, "libraryTrueLands", 0);
        if (libraryTrueLands > 0 && text.contains("land grant") && text.contains("cast")) {
            return 1;
        }
        if (libraryTrueLands > 0 && isLandSearchText(text)) {
            return 2;
        }
        if (text.contains(": play forest")) {
            return 3;
        }
        if (text.contains(": play swamp")) {
            return 4;
        }
        if (text.contains(": play ")) {
            return 5;
        }
        if (text.contains("land grant") && text.contains("cast")) {
            return 11;
        }
        if (text.contains("balustrade spy") && text.contains("cast")) {
            if (libraryTrueLands > 0) {
                return 90_000;
            }
            return battlefieldCreatures >= 2 ? 6 : 90_000;
        }
        if (isCreatureCastText(text)) {
            return creatureCastPriority(text);
        }
        if (text.contains("winding way") && text.contains("cast")) {
            return 25;
        }
        if (text.contains("lead the stampede") && text.contains("cast")) {
            return 26;
        }
        if (text.contains("lotus petal") && text.contains("cast")) {
            return 24;
        }
        if (text.contains(": {t}: add")) {
            return 40;
        }
        if (text.contains("saruli caretaker") && text.contains("add one mana")) {
            return 41;
        }
        if (text.contains("tinder wall") && text.contains("sacrifice") && text.contains("add")) {
            return battlefieldCreatures >= 4 ? 45 : 90_500;
        }
        return Integer.MAX_VALUE;
    }

    private static boolean isLandSearchText(String text) {
        return text.contains("forestcycling")
                || text.contains("swampcycling")
                || text.contains("basic landcycling")
                || text.contains("search your library for a forest")
                || text.contains("search your library for a swamp")
                || text.contains("search your library for a basic land");
    }

    private static int dreadReturnPriority(String state) {
        int graveyardCreatures = stateInt(state, "graveyardCreatures", 0);
        int opponentLife = stateInt(state, "opponentLife", 20);
        boolean hasLotleth = "true".equalsIgnoreCase(stateField(state, "graveyardHasLotleth"));
        if (hasLotleth && graveyardCreatures + 2 >= opponentLife) {
            return 0;
        }
        return 90_200;
    }

    private static boolean isCreatureCastText(String text) {
        return text.contains("cast")
                && (text.contains("saruli caretaker")
                || text.contains("tinder wall")
                || text.contains("wall of roots")
                || text.contains("overgrown battlement")
                || text.contains("gatecreeper vine")
                || text.contains("quirion ranger")
                || text.contains("elves of deep shadow")
                || text.contains("mesmeric fiend")
                || text.contains("masked vandal")
                || text.contains("sagu wildling")
                || text.contains("roost seek"));
    }

    private static int creatureCastPriority(String text) {
        if (text.contains("saruli caretaker")) {
            return 12;
        }
        if (text.contains("tinder wall")) {
            return 13;
        }
        if (text.contains("wall of roots")) {
            return 14;
        }
        if (text.contains("overgrown battlement")) {
            return 15;
        }
        if (text.contains("gatecreeper vine")) {
            return 16;
        }
        if (text.contains("elves of deep shadow")) {
            return 17;
        }
        if (text.contains("quirion ranger")) {
            return 18;
        }
        if (text.contains("mesmeric fiend")) {
            return 19;
        }
        if (text.contains("masked vandal")) {
            return 20;
        }
        return 21;
    }

    private static boolean shouldPreferSpyTriggerPass(String state) {
        if (state == null || state.isEmpty()) {
            return false;
        }
        return stateField(state, "battlefield").contains("Balustrade Spy")
                && state.contains("graveyardHasDread=false")
                && !state.contains("librarySize=0");
    }

    private static boolean isSelfTargetText(String normalizedText) {
        return normalizedText != null
                && (normalizedText.equals("player:acf-prefix")
                || normalizedText.equals("acf-prefix")
                || normalizedText.contains("player:acf-prefix"));
    }

    private static boolean isPlayerTargetText(String normalizedText) {
        return normalizedText != null
                && (normalizedText.startsWith("player:")
                || normalizedText.contains("player:acf-")
                || normalizedText.equals("acf-prefix")
                || normalizedText.equals("acf-cp7"));
    }

    private static boolean stateContainsName(String compactState, String field, String name) {
        return stateField(compactState, field).toLowerCase(Locale.ROOT)
                .contains(name.toLowerCase(Locale.ROOT));
    }

    private static int stateInt(DecisionPoint point, String key, int fallback) {
        if (point == null || point.stateSnapshot == null) {
            return fallback;
        }
        return stateInt(point.stateSnapshot.toCompactText(), key, fallback);
    }

    private static int stateInt(String compactState, String key, int fallback) {
        String value = stateField(compactState, key);
        if (value.isEmpty()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static String stateField(String compactState, String key) {
        if (compactState == null || key == null || key.isEmpty()) {
            return "";
        }
        String needle = key + "=";
        int start = compactState.indexOf(needle);
        if (start < 0) {
            return "";
        }
        start += needle.length();
        int end = compactState.indexOf(';', start);
        return end < 0 ? compactState.substring(start) : compactState.substring(start, end);
    }

    private static List<List<Integer>> branchChoiceLists(DecisionPoint point, Args args, long seed) {
        List<Integer> candidates = branchCandidates(point, args, seed);
        List<List<Integer>> out = new ArrayList<>();
        if (point.trainingData.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN) {
            for (Integer idx : candidates) {
                List<Integer> ranking = rankingWithCandidateLast(point, idx);
                if (!ranking.isEmpty()) {
                    out.add(ranking);
                }
            }
            return out;
        }
        for (Integer idx : candidates) {
            out.add(Collections.singletonList(idx));
        }
        return out;
    }

    private static List<Integer> validCandidateIndices(DecisionPoint point) {
        List<Integer> valid = new ArrayList<>();
        for (int i = 0; i < Math.min(point.trainingData.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES); i++) {
            valid.add(i);
        }
        return valid;
    }

    private static void appendPassMacroBranches(
            List<PrefixNode> queue,
            Set<String> seen,
            PrefixNode node,
            DecisionPoint next,
            Args args,
            int nextNodeId
    ) {
        if (args.passMacroDepths.isEmpty() || next == null) {
            return;
        }
        int passIdx = passCandidateIndex(next);
        if (passIdx < 0) {
            return;
        }
        int allocated = 0;
        for (Integer depthValue : args.passMacroDepths) {
            int depth = depthValue == null ? 0 : depthValue;
            if (depth <= 1 || node.prefixChoices.size() + depth > args.maxPrefixDepth) {
                continue;
            }
            List<List<Integer>> childPrefix = copyPrefix(node.prefixChoices);
            List<List<String>> childTexts = copyPrefixTexts(node.prefixChoiceTexts);
            List<String> passText = Collections.singletonList(next.candidateText(passIdx));
            for (int i = 0; i < depth; i++) {
                childPrefix.add(Collections.singletonList(passIdx));
                childTexts.add(passText);
            }
            String key = prefixKey(childPrefix);
            if (!seen.add(key)) {
                continue;
            }
            queue.add(new PrefixNode(nextNodeId + allocated, node.nodeId, childPrefix, childTexts, "PASS x" + depth));
            allocated++;
        }
    }

    private static int countNewPassMacroBranches(
            PrefixNode node,
            DecisionPoint next,
            Args args,
            Set<String> seen
    ) {
        if (args.passMacroDepths.isEmpty() || next == null || passCandidateIndex(next) < 0) {
            return 0;
        }
        int count = 0;
        int passIdx = passCandidateIndex(next);
        for (Integer depthValue : args.passMacroDepths) {
            int depth = depthValue == null ? 0 : depthValue;
            if (depth <= 1 || node.prefixChoices.size() + depth > args.maxPrefixDepth) {
                continue;
            }
            List<List<Integer>> childPrefix = copyPrefix(node.prefixChoices);
            for (int i = 0; i < depth; i++) {
                childPrefix.add(Collections.singletonList(passIdx));
            }
            if (seen.contains(prefixKey(childPrefix))) {
                count++;
            }
        }
        return count;
    }

    private static int passCandidateIndex(DecisionPoint point) {
        if (point == null || point.candidateTexts == null) {
            return -1;
        }
        for (int i = 0; i < point.candidateTexts.size(); i++) {
            if (isPassText(point.candidateText(i))) {
                return i;
            }
        }
        return -1;
    }

    private static List<Integer> rankingWithCandidateLast(DecisionPoint point, int bottomIdx) {
        List<Integer> valid = validCandidateIndices(point);
        if (!valid.contains(bottomIdx)) {
            return Collections.emptyList();
        }
        List<Integer> ranking = new ArrayList<>();
        for (Integer idx : point.chosenIndices) {
            if (idx != null && valid.contains(idx) && idx != bottomIdx && !ranking.contains(idx)) {
                ranking.add(idx);
            }
        }
        List<Integer> rest = new ArrayList<>(valid);
        rest.sort((a, b) -> Float.compare(point.probs[b], point.probs[a]));
        for (Integer idx : rest) {
            if (idx != bottomIdx && !ranking.contains(idx)) {
                ranking.add(idx);
            }
        }
        ranking.add(bottomIdx);
        return ranking;
    }

    private static String prefixChoiceText(DecisionPoint point, List<Integer> choice) {
        if (choice == null || choice.isEmpty()) {
            return "";
        }
        if (point.trainingData.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN && choice.size() > 1) {
            int bottomIdx = choice.get(choice.size() - 1);
            return "BOTTOM " + point.candidateText(bottomIdx);
        }
        return point.candidateText(choice.get(0));
    }

    private static List<String> prefixChoiceTexts(DecisionPoint point, List<Integer> choice) {
        if (point == null || choice == null || choice.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Integer idx : choice) {
            out.add(idx == null ? "" : point.candidateText(idx));
        }
        return out;
    }

    private static String branchOrderText(DecisionPoint point, List<List<Integer>> branches) {
        if (point == null || branches == null || branches.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (List<Integer> branch : branches) {
            if (branch == null || branch.isEmpty()) {
                continue;
            }
            if (point.trainingData.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN && branch.size() > 1) {
                int bottomIdx = branch.get(branch.size() - 1);
                out.add(bottomIdx + ":BOTTOM " + point.candidateText(bottomIdx));
            } else {
                int idx = branch.get(0);
                out.add(idx + ":" + point.candidateText(idx));
            }
        }
        return joinTexts(out);
    }

    private static TrainingExample buildTrainingExample(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> branchResults,
            Args args,
            Pattern avoidLosingActionTextPattern
    ) {
        List<BranchResult> usable = branchResults.stream()
                .filter(r -> r != null && r.forcedApplied && !r.timedOut && r.error.isEmpty())
                .collect(Collectors.toList());
        if (usable.size() < 2) {
            return null;
        }
        if (args.branchReturnTargets) {
            return buildBranchReturnTrainingExample(job, point, usable, args);
        }
        if (args.baselineLosingAlternativeOnly) {
            return buildBaselineLosingAlternativeExample(job, point, usable, args);
        }
        boolean anyWin = usable.stream().anyMatch(r -> r.won);
        if (!anyWin && args.lossTurnBonus <= 0.0) {
            return null;
        }

        if (avoidLosingActionTextPattern != null) {
            TrainingExample avoidExample = buildAvoidLosingActionExample(
                    job, point, usable, args, avoidLosingActionTextPattern);
            if (avoidExample != null) {
                return avoidExample;
            }
            return null;
        }

        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        double maxScore = usable.stream().mapToDouble(r -> branchScore(r, args)).max().orElse(0.0);
        double minScore = usable.stream().mapToDouble(r -> branchScore(r, args)).min().orElse(0.0);
        if (maxScore - minScore < 1e-6) {
            return null;
        }
        double sum = 0.0;
        for (BranchResult r : usable) {
            double score = Math.exp((branchScore(r, args) - maxScore) / Math.max(1e-6, args.targetTemperature));
            if (r.forcedIdx >= 0 && r.forcedIdx < target.length) {
                target[r.forcedIdx] += (float) score;
                sum += score;
            }
        }
        if (sum <= 0.0) {
            return null;
        }
        int bestIdx = 0;
        float best = -1.0f;
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
            return null;
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }
        return new TrainingExample(job, point, td, usable.size(), anyWin);
    }

    private static TrainingExample buildBranchReturnTrainingExample(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> usable,
            Args args
    ) {
        if (point == null || usable == null || usable.size() < 2) {
            return null;
        }
        if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
            return null;
        }
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int limit = Math.min(point.trainingData.candidateCount, max);
        float[] target = new float[max];
        Arrays.fill(target, BRANCH_RETURN_UNOBSERVED);

        int observed = 0;
        int bestIdx = -1;
        float best = Float.NEGATIVE_INFINITY;
        for (BranchResult r : usable) {
            if (r.forcedIdx < 0 || r.forcedIdx >= limit || point.trainingData.candidateMask[r.forcedIdx] == 0) {
                continue;
            }
            float value = branchReturnTarget(r);
            target[r.forcedIdx] = value;
            observed++;
            if (value > best) {
                best = value;
                bestIdx = r.forcedIdx;
            }
        }
        if (observed <= 0 || bestIdx < 0) {
            return null;
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        boolean anyWin = usable.stream().anyMatch(r -> r.won);
        return new TrainingExample(job, point, td, observed, anyWin);
    }

    private static TrainingExample buildBaselineLosingAlternativeExample(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> usable,
            Args args
    ) {
        if (point == null || usable == null || usable.size() < 2) {
            return null;
        }
        if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
            return null;
        }
        int baselineIdx = point.baselineIdx();
        if (baselineIdx < 0 || baselineIdx >= point.trainingData.candidateCount) {
            return null;
        }
        BranchResult baseline = usable.stream()
                .filter(r -> r.forcedIdx == baselineIdx)
                .findFirst()
                .orElse(null);
        if (baseline == null || baseline.won) {
            return null;
        }

        List<BranchResult> winningAlternatives = usable.stream()
                .filter(r -> r.forcedIdx != baselineIdx && r.won)
                .collect(Collectors.toList());
        if (winningAlternatives.isEmpty()) {
            return null;
        }

        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        double maxScore = winningAlternatives.stream()
                .mapToDouble(r -> branchScore(r, args))
                .max()
                .orElse(0.0);
        double sum = 0.0;
        for (BranchResult r : winningAlternatives) {
            if (r.forcedIdx < 0 || r.forcedIdx >= target.length) {
                continue;
            }
            if (args.skipPassTraining && isPassText(point.candidateText(r.forcedIdx))) {
                continue;
            }
            if (args.skipBlankTraining && point.candidateText(r.forcedIdx).trim().isEmpty()) {
                continue;
            }
            double score = Math.exp((branchScore(r, args) - maxScore) / Math.max(1e-6, args.targetTemperature));
            target[r.forcedIdx] += (float) score;
            sum += score;
        }
        if (sum <= 1e-6) {
            return null;
        }

        int bestIdx = 0;
        float best = -1.0f;
        int targetNonZero = 0;
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
            if (target[i] > 1e-6f) {
                targetNonZero++;
            }
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        if (targetNonZero <= 0 || best <= 1e-6f) {
            return null;
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }

        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        return new TrainingExample(job, point, td, usable.size(), true);
    }

    private static TrainingExample buildAvoidLosingActionExample(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> usable,
            Args args,
            Pattern avoidPattern
    ) {
        if (point == null || usable == null || usable.size() < 2 || avoidPattern == null) {
            return null;
        }
        if (args.avoidLosingStrictNegative) {
            return buildStrictAvoidLosingActionExample(job, point, usable, args, avoidPattern);
        }
        List<BranchResult> critical = usable.stream()
                .filter(r -> matches(avoidPattern, point.candidateText(r.forcedIdx)))
                .collect(Collectors.toList());
        if (critical.isEmpty()) {
            return null;
        }

        List<BranchResult> criticalWins = critical.stream()
                .filter(r -> r.won)
                .collect(Collectors.toList());
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        int targetNonZero = 0;

        if (!criticalWins.isEmpty()) {
            double maxScore = criticalWins.stream().mapToDouble(r -> branchScore(r, args)).max().orElse(0.0);
            double sum = 0.0;
            for (BranchResult r : criticalWins) {
                if (r.forcedIdx < 0 || r.forcedIdx >= target.length) {
                    continue;
                }
                double score = Math.exp((branchScore(r, args) - maxScore) / Math.max(1e-6, args.targetTemperature));
                target[r.forcedIdx] += (float) score;
                sum += score;
            }
            if (sum <= 0.0) {
                return null;
            }
            for (int i = 0; i < target.length; i++) {
                target[i] = (float) (target[i] / sum);
                if (target[i] > 1e-6f) {
                    targetNonZero++;
                }
            }
        } else {
            Set<Integer> losingCritical = critical.stream()
                    .filter(r -> !r.won)
                    .map(r -> r.forcedIdx)
                    .collect(Collectors.toSet());
            if (losingCritical.isEmpty()) {
                return null;
            }
            double sum = 0.0;
            int limit = Math.min(point.trainingData.candidateCount, target.length);
            for (int i = 0; i < limit; i++) {
                if (point.trainingData.candidateMask[i] == 0 || losingCritical.contains(i)) {
                    continue;
                }
                float p = i < point.probs.length ? point.probs[i] : 0.0f;
                if (Float.isFinite(p) && p > 0.0f) {
                    target[i] = p;
                    sum += p;
                }
            }
            if (sum <= 1e-6) {
                for (int i = 0; i < limit; i++) {
                    if (point.trainingData.candidateMask[i] != 0 && !losingCritical.contains(i)) {
                        target[i] = 1.0f;
                        sum += 1.0;
                    }
                }
            }
            if (sum <= 1e-6) {
                return null;
            }
            for (int i = 0; i < target.length; i++) {
                target[i] = (float) (target[i] / sum);
                if (target[i] > 1e-6f) {
                    targetNonZero++;
                }
            }
        }

        int bestIdx = 0;
        float best = -1.0f;
        for (int i = 0; i < target.length; i++) {
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        if (targetNonZero <= 0 || best <= 1e-6f) {
            return null;
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        return new TrainingExample(job, point, td, usable.size(), true);
    }

    private static TrainingExample buildStrictAvoidLosingActionExample(
            ScenarioJob job,
            DecisionPoint point,
            List<BranchResult> usable,
            Args args,
            Pattern avoidPattern
    ) {
        int baselineIdx = point.baselineIdx();
        if (baselineIdx < 0 || !matches(avoidPattern, point.candidateText(baselineIdx))) {
            return null;
        }
        BranchResult baseline = usable.stream()
                .filter(r -> r.forcedIdx == baselineIdx)
                .findFirst()
                .orElse(null);
        if (baseline == null || baseline.won) {
            return null;
        }

        List<BranchResult> winningAlternatives = usable.stream()
                .filter(r -> r.forcedIdx != baselineIdx && r.won)
                .collect(Collectors.toList());
        if (winningAlternatives.isEmpty()) {
            return null;
        }

        if (args.avoidLosingMaskBaselineOnly) {
            return buildMaskBaselineOnlyExample(job, point, usable.size(), args, baselineIdx);
        }

        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        double maxScore = winningAlternatives.stream().mapToDouble(r -> branchScore(r, args)).max().orElse(0.0);
        double sum = 0.0;
        for (BranchResult r : winningAlternatives) {
            if (r.forcedIdx < 0 || r.forcedIdx >= target.length) {
                continue;
            }
            double score = Math.exp((branchScore(r, args) - maxScore) / Math.max(1e-6, args.targetTemperature));
            target[r.forcedIdx] += (float) score;
            sum += score;
        }
        if (sum <= 1e-6) {
            return null;
        }
        int bestIdx = 0;
        float best = -1.0f;
        int targetNonZero = 0;
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
            if (target[i] > 1e-6f) {
                targetNonZero++;
            }
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        if (targetNonZero <= 0 || best <= 1e-6f) {
            return null;
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        return new TrainingExample(job, point, td, usable.size(), true);
    }

    private static TrainingExample buildMaskBaselineOnlyExample(
            ScenarioJob job,
            DecisionPoint point,
            int branchCount,
            Args args,
            int baselineIdx
    ) {
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        int limit = Math.min(point.trainingData.candidateCount, target.length);
        double sum = 0.0;
        for (int i = 0; i < limit; i++) {
            if (i == baselineIdx || point.trainingData.candidateMask[i] == 0) {
                continue;
            }
            float p = i < point.probs.length ? point.probs[i] : 0.0f;
            if (Float.isFinite(p) && p > 0.0f) {
                target[i] = p;
                sum += p;
            }
        }
        if (sum <= 1e-6) {
            for (int i = 0; i < limit; i++) {
                if (i != baselineIdx && point.trainingData.candidateMask[i] != 0) {
                    target[i] = 1.0f;
                    sum += 1.0;
                }
            }
        }
        if (sum <= 1e-6) {
            return null;
        }
        int bestIdx = 0;
        float best = -1.0f;
        int targetNonZero = 0;
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
            if (target[i] > 1e-6f) {
                targetNonZero++;
            }
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        if (targetNonZero <= 0 || best <= 1e-6f) {
            return null;
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        return new TrainingExample(job, point, td, branchCount, true);
    }

    private static List<TrainingExample> buildPrefixTrainingExamples(
            ScenarioJob job,
            List<DecisionPoint> decisions,
            List<List<Integer>> prefixChoices,
            Args args
    ) {
        List<TrainingExample> out = new ArrayList<>();
        int n = Math.min(prefixChoices.size(), decisions.size());
        for (int ordinal = 0; ordinal < n; ordinal++) {
            DecisionPoint point = decisions.get(ordinal);
            if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
                continue;
            }
            List<Integer> choice = prefixChoices.get(ordinal);
            if (choice == null || choice.isEmpty()) {
                continue;
            }
            List<Integer> sanitized = sanitizeChoice(choice, point.trainingData);
            if (sanitized.isEmpty()) {
                continue;
            }
            int firstIdx = sanitized.get(0);
            if (args.skipPassTraining && isPassText(point.candidateText(firstIdx))) {
                continue;
            }
            if (args.skipBlankTraining && point.candidateText(firstIdx).trim().isEmpty()) {
                continue;
            }
            StateSequenceBuilder.TrainingData td;
            if (point.trainingData.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN
                    && sanitized.size() > 1) {
                td = copyWithFullChoiceTarget(point.trainingData, sanitized, rankDistribution(sanitized));
            } else {
                float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
                target[firstIdx] = 1.0f;
                td = copyWithTarget(point.trainingData, firstIdx, target);
            }
            out.add(new TrainingExample(job, point, td, 1, true));
        }
        return out;
    }

    private static List<WinningTrajectoryRecord> buildWinningTrajectoryRecords(
            ScenarioJob job,
            PrefixRunResult run,
            List<List<Integer>> trainPrefix,
            List<List<Integer>> searchPrefix,
            int winIndex
    ) {
        if (run == null || run.decisions.isEmpty() || trainPrefix == null || trainPrefix.isEmpty()) {
            return Collections.emptyList();
        }
        String trajectoryKey = job.scenario + "|" + job.seed + "|" + winIndex + "|"
                + prefixKey(searchPrefix == null ? Collections.emptyList() : searchPrefix);
        List<WinningTrajectoryRecord> out = new ArrayList<>();
        int n = Math.min(trainPrefix.size(), run.decisions.size());
        for (int ordinal = 0; ordinal < n; ordinal++) {
            DecisionPoint point = run.decisions.get(ordinal);
            List<Integer> choice = sanitizeChoice(trainPrefix.get(ordinal), point.trainingData);
            if (choice.isEmpty()) {
                continue;
            }
            out.add(new WinningTrajectoryRecord(
                    trajectoryKey,
                    job,
                    point,
                    choice,
                    run.turns,
                    run.firstMulliganHand,
                    run.firstPriorityHand,
                    run.finalState.toCompactText()));
        }
        return out;
    }

    private static TrajectoryTrainingEpisode buildTrajectoryTrainingEpisode(
            ScenarioJob job,
            PrefixRunResult run,
            List<List<Integer>> trainPrefix,
            List<List<Integer>> searchPrefix,
            int winIndex,
            Args args
    ) {
        if (run == null || run.decisions.isEmpty() || trainPrefix == null || trainPrefix.isEmpty()) {
            return null;
        }
        String trajectoryKey = job.scenario + "|" + job.seed + "|" + winIndex + "|"
                + prefixKey(searchPrefix == null ? Collections.emptyList() : searchPrefix);
        List<StateSequenceBuilder.TrainingData> records = new ArrayList<>();
        int n = Math.min(trainPrefix.size(), run.decisions.size());
        for (int ordinal = 0; ordinal < n; ordinal++) {
            DecisionPoint point = run.decisions.get(ordinal);
            if (point == null || point.trainingData == null) {
                continue;
            }
            if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
                continue;
            }
            List<Integer> choice = sanitizeChoice(trainPrefix.get(ordinal), point.trainingData);
            if (choice.isEmpty()) {
                continue;
            }
            int firstIdx = choice.get(0);
            if (args.skipPassTraining && isPassText(point.candidateText(firstIdx))) {
                continue;
            }
            if (args.skipBlankTraining && point.candidateText(firstIdx).trim().isEmpty()) {
                continue;
            }
            records.add(copyWithChoiceForTrajectory(point.trainingData, choice));
        }
        if (records.isEmpty()) {
            return null;
        }
        List<Double> rewards = new ArrayList<>(Collections.nCopies(records.size(), 0.0));
        rewards.set(rewards.size() - 1, args.trajectoryFinalReward);
        return new TrajectoryTrainingEpisode(trajectoryKey, records, rewards);
    }

    private static TrainingExample buildPrefixSiblingContrastExample(
            ScenarioJob job,
            DecisionPoint point,
            List<PrefixBranchResult> branchResults,
            int requiredForcedDepth,
            Args args
    ) {
        if (point == null || branchResults == null || branchResults.size() < 2) {
            return null;
        }
        if (args.skipMulliganTraining && isMulliganAction(point.trainingData.actionType)) {
            return null;
        }
        if (point.trainingData.actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN) {
            return null;
        }
        List<PrefixBranchResult> usable = branchResults.stream()
                .filter(r -> r != null
                        && r.run != null
                        && r.run.forcedAppliedCount >= requiredForcedDepth
                        && !r.run.timedOut
                        && r.run.error.isEmpty()
                        && !sanitizeChoice(r.choice, point.trainingData).isEmpty())
                .collect(Collectors.toList());
        if (usable.size() < 2) {
            return null;
        }
        boolean anyWin = usable.stream().anyMatch(r -> r.run.won);
        if (!anyWin && args.lossTurnBonus <= 0.0) {
            return null;
        }

        double maxScore = usable.stream()
                .mapToDouble(r -> branchScore(r.run.won, r.run.turns, args))
                .max()
                .orElse(0.0);
        double minScore = usable.stream()
                .mapToDouble(r -> branchScore(r.run.won, r.run.turns, args))
                .min()
                .orElse(0.0);
        if (maxScore - minScore < 1e-6) {
            return null;
        }

        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        double sum = 0.0;
        for (PrefixBranchResult r : usable) {
            List<Integer> sanitized = sanitizeChoice(r.choice, point.trainingData);
            if (sanitized.isEmpty()) {
                continue;
            }
            int idx = sanitized.get(0);
            if (args.skipPassTraining && isPassText(point.candidateText(idx))) {
                continue;
            }
            if (args.skipBlankTraining && point.candidateText(idx).trim().isEmpty()) {
                continue;
            }
            double score = Math.exp((branchScore(r.run.won, r.run.turns, args) - maxScore)
                    / Math.max(1e-6, args.targetTemperature));
            target[idx] += (float) score;
            sum += score;
        }
        if (sum <= 0.0) {
            return null;
        }

        int bestIdx = 0;
        float best = -1.0f;
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
            if (target[i] > best) {
                best = target[i];
                bestIdx = i;
            }
        }
        if (args.skipPassBest && isPassText(point.candidateText(bestIdx))) {
            return null;
        }
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, bestIdx, target);
        return new TrainingExample(job, point, td, usable.size(), anyWin);
    }

    private static TrainingExample buildRootMulliganNoWinExample(
            ScenarioJob job,
            PrefixRunResult rootRun,
            Args args
    ) {
        if (rootRun == null || rootRun.decisions.isEmpty()) {
            return null;
        }
        DecisionPoint point = rootRun.decisions.get(0);
        if (point.trainingData.actionType != StateSequenceBuilder.ActionType.MULLIGAN) {
            return null;
        }
        int mulliganIdx = -1;
        for (int i = 0; i < Math.min(point.trainingData.candidateCount, point.candidateTexts.size()); i++) {
            if ("mulligan".equalsIgnoreCase(point.candidateText(i).trim())) {
                mulliganIdx = i;
                break;
            }
        }
        if (mulliganIdx < 0) {
            return null;
        }
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        target[mulliganIdx] = 1.0f;
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, mulliganIdx, target);
        return new TrainingExample(job, point, td, 1, false);
    }

    private static List<TrainingExample> buildRootMulliganFromWinningPrefix(
            ScenarioJob job,
            List<DecisionPoint> decisions,
            List<List<Integer>> prefixChoices,
            Args args
    ) {
        if (decisions == null || decisions.isEmpty() || prefixChoices == null || prefixChoices.isEmpty()) {
            return Collections.emptyList();
        }
        DecisionPoint point = decisions.get(0);
        if (point.trainingData.actionType != StateSequenceBuilder.ActionType.MULLIGAN) {
            return Collections.emptyList();
        }
        List<Integer> sanitized = sanitizeChoice(prefixChoices.get(0), point.trainingData);
        if (sanitized.isEmpty()) {
            return Collections.emptyList();
        }
        int idx = sanitized.get(0);
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        target[idx] = 1.0f;
        StateSequenceBuilder.TrainingData td = copyWithTarget(point.trainingData, idx, target);
        return Collections.singletonList(new TrainingExample(job, point, td, 1, true));
    }

    private static boolean isMulliganAction(StateSequenceBuilder.ActionType actionType) {
        return actionType == StateSequenceBuilder.ActionType.MULLIGAN
                || actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN;
    }

    private static boolean isLikelyPassTarget(StateSequenceBuilder.TrainingData td) {
        if (td == null || td.candidateMask == null) {
            return false;
        }
        int idx = targetIndex(td);
        if (idx < 0 || idx >= Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES)
                || td.candidateMask[idx] == 0) {
            return false;
        }
        if (td.candidateActionIds != null && idx < td.candidateActionIds.length
                && td.candidateActionIds[idx] == passVocabId()) {
            return true;
        }
        return td.candidateFeatures != null
                && idx < td.candidateFeatures.length
                && td.candidateFeatures[idx] != null
                && td.candidateFeatures[idx].length > 1
                && td.candidateFeatures[idx][1] > 0.5f;
    }

    private static int targetIndex(StateSequenceBuilder.TrainingData td) {
        int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        int best = -1;
        float bestMass = 0.0f;
        if (td.mctsVisitTargets != null) {
            for (int i = 0; i < max && i < td.mctsVisitTargets.length; i++) {
                if (td.candidateMask[i] != 0 && td.mctsVisitTargets[i] > bestMass) {
                    bestMass = td.mctsVisitTargets[i];
                    best = i;
                }
            }
        }
        if (best >= 0) {
            return best;
        }
        return td.chosenIndices != null && td.chosenIndices.length > 0 ? td.chosenIndices[0] : -1;
    }

    private static int passVocabId() {
        int h = "PASS".hashCode();
        int mod = Math.floorMod(h, StateSequenceBuilder.TOKEN_ID_VOCAB - 1);
        return 1 + mod;
    }

    private static List<Integer> sanitizeChoice(List<Integer> choice, StateSequenceBuilder.TrainingData td) {
        if (choice == null || td == null) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        for (Integer idx : choice) {
            if (idx == null || idx < 0 || idx >= max || td.candidateMask[idx] == 0 || out.contains(idx)) {
                continue;
            }
            out.add(idx);
        }
        return out;
    }

    private static float[] rankDistribution(List<Integer> choice) {
        float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        int n = choice == null ? 0 : Math.min(choice.size(), target.length);
        if (n <= 0) {
            return target;
        }
        double sum = 0.0;
        for (int rank = 0; rank < n; rank++) {
            int idx = choice.get(rank);
            if (idx < 0 || idx >= target.length) {
                continue;
            }
            double weight = Math.max(0.0, n - rank - 1.0);
            if (n == 1) {
                weight = 1.0;
            }
            target[idx] = (float) weight;
            sum += weight;
        }
        if (sum <= 0.0) {
            int idx = choice.get(0);
            if (idx >= 0 && idx < target.length) {
                target[idx] = 1.0f;
            }
            return target;
        }
        for (int i = 0; i < target.length; i++) {
            target[i] = (float) (target[i] / sum);
        }
        return target;
    }

    private static List<List<Integer>> winningTrainingPrefix(
            int forcedPrefixDepth,
            List<DecisionPoint> decisions,
            Args args
    ) {
        List<List<Integer>> out = new ArrayList<>();
        int n = Math.min(args.trainPrefixDepth, decisions.size());
        for (int ordinal = 0; ordinal < n; ordinal++) {
            out.add(new ArrayList<>(decisions.get(ordinal).chosenIndices));
        }
        return out;
    }

    private static List<List<Integer>> copyPrefix(List<List<Integer>> prefix) {
        List<List<Integer>> out = new ArrayList<>();
        for (List<Integer> choice : prefix) {
            out.add(choice == null ? Collections.emptyList() : new ArrayList<>(choice));
        }
        return out;
    }

    private static List<List<String>> copyPrefixTexts(List<List<String>> prefixTexts) {
        List<List<String>> out = new ArrayList<>();
        if (prefixTexts == null) {
            return out;
        }
        for (List<String> choice : prefixTexts) {
            out.add(choice == null ? Collections.emptyList() : new ArrayList<>(choice));
        }
        return out;
    }

    private static String prefixKey(List<List<Integer>> prefix) {
        StringBuilder sb = new StringBuilder();
        for (List<Integer> choice : prefix) {
            if (sb.length() > 0) {
                sb.append('/');
            }
            if (choice == null || choice.isEmpty()) {
                sb.append('_');
            } else {
                for (int i = 0; i < choice.size(); i++) {
                    if (i > 0) {
                        sb.append('+');
                    }
                    sb.append(choice.get(i));
                }
            }
        }
        return sb.toString();
    }

    private static List<List<Integer>> parsePrefixKey(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return Collections.emptyList();
        }
        List<List<Integer>> out = new ArrayList<>();
        for (String step : raw.trim().split("/")) {
            String s = step.trim();
            if (s.isEmpty() || "_".equals(s)) {
                out.add(Collections.emptyList());
                continue;
            }
            List<Integer> indices = new ArrayList<>();
            for (String token : s.split("\\+")) {
                String t = token.trim();
                if (!t.isEmpty()) {
                    indices.add(Integer.parseInt(t));
                }
            }
            out.add(indices);
        }
        return out;
    }

    private static List<Integer> parseIntList(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        for (String token : raw.split("[,;]")) {
            String s = token.trim();
            if (!s.isEmpty()) {
                out.add(Integer.parseInt(s));
            }
        }
        return out;
    }

    private static boolean isPassText(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.trim().toLowerCase(Locale.ROOT);
        return "pass".equals(normalized)
                || "ability: pass".equals(normalized)
                || normalized.endsWith(": pass");
    }

    private static String normalizeText(String text) {
        if (text == null) {
            return "";
        }
        String normalized = text.trim().replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
        normalized = REPLAY_OBJECT_ID_MARKER.matcher(normalized).replaceAll("");
        return REPLAY_ACTOR_TARGET_SUFFIX.matcher(normalized).replaceFirst("").trim();
    }

    private static boolean textMatchesExpected(String expectedText, String actualText) {
        String expected = normalizeText(expectedText);
        String actual = normalizeText(actualText);
        if (expected.isEmpty()) {
            return true;
        }
        if (expected.equals(actual)) {
            return true;
        }
        String expectedPlayer = replayPlayerTargetRole(expected);
        String actualPlayer = replayPlayerTargetRole(actual);
        if (!expectedPlayer.isEmpty() && expectedPlayer.equals(actualPlayer)) {
            return true;
        }
        if (replayPlayerTargetSuffixMatches(expected, actual)) {
            return true;
        }
        return actual.endsWith(": " + expected) || actual.endsWith(expected);
    }

    private static boolean textMatchesExpected(String expectedText, String actualText, DecisionPoint point) {
        if (textMatchesExpected(expectedText, actualText)) {
            return true;
        }
        String role = replayRoleFromState(expectedText, actualText, point == null ? null : point.stateSnapshot);
        return !role.isEmpty() && role.equals(standaloneRoleSuffix(expectedText).role);
    }

    private static boolean replayPlayerTargetSuffixMatches(String expected, String actual) {
        int expectedArrow = expected == null ? -1 : expected.lastIndexOf("->");
        int actualArrow = actual == null ? -1 : actual.lastIndexOf("->");
        if (expectedArrow < 0 || actualArrow < 0) {
            return false;
        }
        String expectedPrefix = expected.substring(0, expectedArrow).trim();
        String actualPrefix = actual.substring(0, actualArrow).trim();
        if (expectedPrefix.isEmpty() || !expectedPrefix.equals(actualPrefix)) {
            return false;
        }
        String expectedRole = replayPlayerTargetRole(expected.substring(expectedArrow + 2).trim());
        String actualRole = replayPlayerTargetRole(actual.substring(actualArrow + 2).trim());
        return !expectedRole.isEmpty() && expectedRole.equals(actualRole);
    }

    private static String replayPlayerTargetRole(String normalizedText) {
        if (normalizedText == null || normalizedText.isEmpty()) {
            return "";
        }
        String text = normalizedText;
        if (text.startsWith("player:")) {
            text = text.substring("player:".length()).trim();
        }
        if ("playerrl1".equals(text) || "acf-prefix".equals(text) || "acf-base".equals(text)) {
            return "agent";
        }
        if ("acf-cp7".equals(text) || text.startsWith("evalbot-skill")) {
            return "opponent";
        }
        return "";
    }

    private static String replayRoleFromState(String expectedText, String actualText, StateSnapshot state) {
        StandaloneRoleSuffix expected = standaloneRoleSuffix(expectedText);
        if (!expected.valid()) {
            return "";
        }
        String actual = normalizeText(actualText);
        StandaloneRoleSuffix actualSuffix = standaloneRoleSuffix(actualText);
        String actualBase = actualSuffix.valid() ? actualSuffix.base : actual;
        if (!expected.base.equals(actualBase) && !actualBase.endsWith(": " + expected.base)) {
            return "";
        }
        if (state == null) {
            return "";
        }
        String compact = state.toCompactText();
        int selfCount = countStateName(compact, expected.base, "battlefield", "battlefieldDetail", "hand", "graveyard", "exile");
        int opponentCount = countStateName(compact, expected.base,
                "opponentBattlefield", "opponentGraveyard", "opponentExile");
        if (selfCount > 0 && opponentCount == 0) {
            return "agent";
        }
        if (opponentCount > 0 && selfCount == 0) {
            return "opponent";
        }
        return "";
    }

    private static int countStateName(String compactState, String normalizedName, String... fields) {
        if (compactState == null || compactState.isEmpty() || normalizedName == null || normalizedName.isEmpty()) {
            return 0;
        }
        int count = 0;
        for (String field : fields) {
            String value = compactStateField(compactState, field);
            if (value.isEmpty()) {
                continue;
            }
            for (String token : value.split("\\|")) {
                String normalized = normalizeStateObjectName(token);
                if (normalizedName.equals(normalized)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static String compactStateField(String compactState, String key) {
        if (compactState == null || key == null || key.isEmpty()) {
            return "";
        }
        String prefix = key + "=";
        for (String part : compactState.split(";")) {
            if (part.startsWith(prefix)) {
                return part.substring(prefix.length());
            }
        }
        return "";
    }

    private static String normalizeStateObjectName(String text) {
        if (text == null) {
            return "";
        }
        String normalized = text.replaceAll("\\[[^\\]]*\\]", " ");
        return normalizeText(normalized);
    }

    private static StandaloneRoleSuffix standaloneRoleSuffix(String text) {
        String normalized = normalizeText(text);
        int close = normalized.endsWith(")") ? normalized.lastIndexOf(')') : -1;
        int open = close > 0 ? normalized.lastIndexOf('(', close) : -1;
        if (open <= 0 || close != normalized.length() - 1) {
            return StandaloneRoleSuffix.EMPTY;
        }
        String base = normalized.substring(0, open).trim();
        String suffix = normalized.substring(open + 1, close).trim();
        String role = replayStandaloneRole(suffix);
        if (base.isEmpty() || role.isEmpty()) {
            return StandaloneRoleSuffix.EMPTY;
        }
        return new StandaloneRoleSuffix(base, role);
    }

    private static String replayStandaloneRole(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }
        String normalized = text.trim().toLowerCase(Locale.ROOT);
        if ("you".equals(normalized)
                || "playerrl1".equals(normalized)
                || "acf-prefix".equals(normalized)
                || "acf-base".equals(normalized)
                || "player:acf-prefix".equals(normalized)
                || "player:acf-base".equals(normalized)) {
            return "agent";
        }
        if ("acf-cp7".equals(normalized)
                || "player:acf-cp7".equals(normalized)
                || normalized.startsWith("evalbot-skill")) {
            return "opponent";
        }
        return "";
    }

    private static final class StandaloneRoleSuffix {
        static final StandaloneRoleSuffix EMPTY = new StandaloneRoleSuffix("", "");

        final String base;
        final String role;

        private StandaloneRoleSuffix(String base, String role) {
            this.base = base == null ? "" : base;
            this.role = role == null ? "" : role;
        }

        boolean valid() {
            return !base.isEmpty() && !role.isEmpty();
        }
    }

    private static String contextKey(String text) {
        if (text == null) {
            return "";
        }
        String normalized = text.trim().toUpperCase(Locale.ROOT).replaceAll("[^A-Z0-9]+", "_");
        return normalized.replaceAll("^_+", "").replaceAll("_+$", "");
    }

    private static String sourceStepKey(String sourcePhase) {
        String key = contextKey(sourcePhase);
        if (key.contains("PRECOMBAT_MAIN")) {
            return "PRECOMBAT_MAIN";
        }
        if (key.contains("POSTCOMBAT_MAIN")) {
            return "POSTCOMBAT_MAIN";
        }
        if (key.contains("DECLARE_ATTACKERS") || key.contains("DECLARE_ATTACKS")) {
            return "DECLARE_ATTACKERS";
        }
        if (key.contains("DECLARE_BLOCKERS") || key.contains("DECLARE_BLOCKS")) {
            return "DECLARE_BLOCKERS";
        }
        if (key.contains("BEGIN_COMBAT")) {
            return "BEGIN_COMBAT";
        }
        if (key.contains("COMBAT_DAMAGE")) {
            return "COMBAT_DAMAGE";
        }
        if (key.contains("END_COMBAT")) {
            return "END_COMBAT";
        }
        if (key.contains("UPKEEP")) {
            return "UPKEEP";
        }
        if (key.contains("DRAW")) {
            return "DRAW";
        }
        if (key.contains("END_TURN") || key.equals("END")) {
            return "END_TURN";
        }
        if (key.contains("CLEANUP")) {
            return "CLEANUP";
        }
        return "";
    }

    private static String sourcePromptKey(String sourcePhase) {
        String key = contextKey(sourcePhase);
        if (key.contains("TARGET_PICK")) {
            return "TARGET_PICK";
        }
        if (key.contains("CARD_PICK") || key.contains("SELECT_CARD")) {
            return "CARD_PICK";
        }
        if (key.contains("CHOOSE_USE")) {
            return "CHOOSE_USE";
        }
        if (key.contains("CHOOSE_MODE")) {
            return "CHOOSE_MODE";
        }
        if (key.contains("ANNOUNCE_X")) {
            return "ANNOUNCE_X";
        }
        if (key.contains("DECLARE_BLOCKS")) {
            return "DECLARE_BLOCKS";
        }
        if (key.contains("DECLARE_ATTACKS")) {
            return "DECLARE_ATTACKS";
        }
        return "";
    }

    private static String actualPromptKey(StateSequenceBuilder.ActionType actionType, StateSnapshot state) {
        if (actionType == null) {
            return "";
        }
        if (actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS) {
            return "TARGET_PICK";
        }
        if (actionType == StateSequenceBuilder.ActionType.SELECT_CARD) {
            return "CARD_PICK";
        }
        if (actionType == StateSequenceBuilder.ActionType.CHOOSE_USE) {
            return "CHOOSE_USE";
        }
        if (actionType == StateSequenceBuilder.ActionType.CHOOSE_MODE) {
            return "CHOOSE_MODE";
        }
        if (actionType == StateSequenceBuilder.ActionType.ANNOUNCE_X) {
            return "ANNOUNCE_X";
        }
        String actualStep = contextKey(state == null ? "" : state.stepText());
        if (actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS || "DECLARE_BLOCKERS".equals(actualStep)) {
            return "DECLARE_BLOCKS";
        }
        if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS || "DECLARE_ATTACKERS".equals(actualStep)) {
            return "DECLARE_ATTACKS";
        }
        return "";
    }

    private static boolean promptCompatible(String expectedPrompt, String actualPrompt) {
        return expectedPrompt == null
                || expectedPrompt.isEmpty()
                || expectedPrompt.equals(actualPrompt);
    }

    private static boolean isObjectSensitiveReplayAction(StateSequenceBuilder.ActionType actionType) {
        return actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                || actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS
                || actionType == StateSequenceBuilder.ActionType.SELECT_CARD;
    }

    private static String sourceContextDetail(
            ReplayExpectation expected,
            StateSequenceBuilder.ActionType actualActionType,
            StateSnapshot actualState
    ) {
        return "source_decision=" + (expected == null ? "" : expected.sourceDecisionNumber)
                + " source_turn=" + (expected == null ? "" : expected.sourceTurn)
                + " source_phase=" + (expected == null ? "" : expected.sourcePhase)
                + " source_actor=" + (expected == null ? "" : expected.sourceActor)
                + " expected_step=" + (expected == null ? "" : sourceStepKey(expected.sourcePhase))
                + " expected_prompt=" + (expected == null ? "" : sourcePromptKey(expected.sourcePhase))
                + " expected_stack_count=" + (expected == null ? -1 : expected.sourceStackCount)
                + " expected_stack_top=" + (expected == null ? "" : expected.sourceStackTop)
                + " actual_turn=" + (actualState == null ? -1 : actualState.turn())
                + " actual_source_turn=" + (actualState == null ? -1 : compactSourceTurn(actualState.turn()))
                + " actual_phase=" + (actualState == null ? "" : actualState.phaseText())
                + " actual_step=" + (actualState == null ? "" : actualState.stepText())
                + " actual_stack_count=" + (actualState == null ? -1 : actualState.stackCount())
                + " actual_stack_top=" + (actualState == null ? "" : actualState.stackTop())
                + " actual_action_type=" + (actualActionType == null ? "" : actualActionType)
                + " actual_prompt=" + actualPromptKey(actualActionType, actualState);
    }

    private static String stackTopKey(String text) {
        if (text == null) {
            return "";
        }
        String withoutController = text.replaceAll("\\s*\\(controller=.*$", "");
        return contextKey(withoutController);
    }

    private static boolean replayStackTopCompatible(String expectedTop, String actualTop) {
        String expected = stackTopKey(expectedTop);
        String actual = stackTopKey(actualTop);
        if (expected.isEmpty() || actual.isEmpty()) {
            return true;
        }
        return expected.equals(actual) || expected.contains(actual) || actual.contains(expected);
    }

    private static String appendDetail(String base, String extra) {
        if (extra == null || extra.trim().isEmpty()) {
            return base == null ? "" : base;
        }
        if (base == null || base.trim().isEmpty()) {
            return extra;
        }
        return base + " " + extra;
    }

    private static String combatTrace(
            Game game,
            Player player,
            String stage,
            int expectedOrdinal,
            ReplayExpectation expected
    ) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("stage=").append(stage);
            sb.append(";expectedOrdinal=").append(expectedOrdinal);
            if (expected != null) {
                sb.append(";expectedSourceDecision=").append(expected.sourceDecisionNumber);
                sb.append(";expectedSourcePhase=").append(expected.sourcePhase);
            }
            sb.append(";turn=").append(safeTurn(game));
            sb.append(";sourceTurn=").append(compactSourceTurn(safeTurn(game)));
            sb.append(";phase=").append(game == null ? "" : String.valueOf(game.getTurnPhaseType()));
            sb.append(";step=").append(game == null ? "" : String.valueOf(game.getTurnStepType()));
            sb.append(";active=").append(safePlayerName(game, game == null ? null : game.getActivePlayerId()));
            sb.append(";priority=").append(safePlayerName(game, game == null ? null : game.getPriorityPlayerId()));
            sb.append(";attackers=").append(combatSetText(game, true));
            sb.append(";blockers=").append(combatSetText(game, false));
            sb.append(";groups=").append(combatGroupsText(game));
            sb.append(";legalBlockers=").append(legalBlockersText(game, player));
            sb.append(";blockerLegality=").append(blockerLegalityText(game, player));
            return sb.toString();
        } catch (Exception e) {
            return "combat_trace_error=" + exceptionSummary(e);
        }
    }

    private static String combatSetText(Game game, boolean attackers) {
        if (game == null || game.getCombat() == null) {
            return "";
        }
        try {
            Set<UUID> ids = attackers ? game.getCombat().getAttackers() : game.getCombat().getBlockers();
            List<String> names = new ArrayList<>();
            for (UUID id : ids) {
                names.add(permanentDebugName(game, id));
            }
            return String.join("|", names);
        } catch (Exception e) {
            return "error=" + exceptionSummary(e);
        }
    }

    private static String combatGroupsText(Game game) {
        if (game == null || game.getCombat() == null) {
            return "";
        }
        try {
            List<String> groups = new ArrayList<>();
            for (CombatGroup group : game.getCombat().getGroups()) {
                List<String> attackers = new ArrayList<>();
                for (UUID attackerId : group.getAttackers()) {
                    attackers.add(permanentDebugDetail(game, attackerId));
                }
                List<String> blockers = new ArrayList<>();
                for (UUID blockerId : group.getBlockers()) {
                    blockers.add(permanentDebugDetail(game, blockerId));
                }
                groups.add("att=" + String.join("+", attackers)
                        + ">def=" + safePlayerName(game, group.getDefenderId())
                        + "(" + String.valueOf(group.getDefenderId()) + ")"
                        + ">defPlayer=" + safePlayerName(game, group.getDefendingPlayerId())
                        + "(" + String.valueOf(group.getDefendingPlayerId()) + ")"
                        + ">blk=" + String.join("+", blockers));
            }
            return String.join("|", groups);
        } catch (Exception e) {
            return "error=" + exceptionSummary(e);
        }
    }

    private static String legalBlockersText(Game game, Player player) {
        if (game == null || player == null || game.getCombat() == null || game.getBattlefield() == null) {
            return "";
        }
        try {
            List<String> rows = new ArrayList<>();
            List<Permanent> blockers = new ArrayList<>(game.getBattlefield().getAllActivePermanents(player.getId()));
            for (CombatGroup group : game.getCombat().getGroups()) {
                for (UUID attackerId : group.getAttackers()) {
                    Permanent attacker = game.getPermanent(attackerId);
                    if (attacker == null) {
                        continue;
                    }
                    List<String> legal = new ArrayList<>();
                    for (Permanent blocker : blockers) {
                        if (blocker != null && blocker.isCreature(game) && blocker.canBlock(attackerId, game)) {
                            legal.add(permanentDebugName(blocker, game));
                        }
                    }
                    legal.add("DONE");
                    rows.add(permanentDebugName(attacker, game) + "<-" + String.join("+", legal));
                }
            }
            return String.join("|", rows);
        } catch (Exception e) {
            return "error=" + exceptionSummary(e);
        }
    }

    private static String blockerLegalityText(Game game, Player player) {
        if (game == null || player == null || game.getCombat() == null || game.getBattlefield() == null) {
            return "";
        }
        try {
            List<Permanent> candidates = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(player.getId())) {
                if (permanent != null && permanent.isCreature(game)) {
                    candidates.add(permanent);
                }
            }
            List<String> rows = new ArrayList<>();
            int groupIndex = 0;
            for (CombatGroup group : game.getCombat().getGroups()) {
                for (UUID attackerId : group.getAttackers()) {
                    Permanent attacker = game.getPermanent(attackerId);
                    for (Permanent blocker : candidates) {
                        boolean permanentCanBlock = false;
                        boolean groupCanBlock = false;
                        try {
                            permanentCanBlock = blocker.canBlock(attackerId, game);
                        } catch (Exception ignored) {
                        }
                        try {
                            groupCanBlock = group.canBlock(blocker, game);
                        } catch (Exception ignored) {
                        }
                        rows.add("group=" + groupIndex
                                + ">defender=" + safePlayerName(game, group.getDefenderId())
                                + "(" + String.valueOf(group.getDefenderId()) + ")"
                                + ">defendingPlayer=" + safePlayerName(game, group.getDefendingPlayerId())
                                + "(" + String.valueOf(group.getDefendingPlayerId()) + ")"
                                + ">attacker=" + permanentDebugDetail(attacker, game)
                                + ">blocker=" + permanentDebugDetail(blocker, game)
                                + ">Permanent.canBlock=" + permanentCanBlock
                                + ">CombatGroup.canBlock=" + groupCanBlock
                                + ">reason=" + blockerLegalityReason(group, attackerId, blocker, game,
                                permanentCanBlock, groupCanBlock));
                    }
                }
                groupIndex++;
            }
            return rows.isEmpty() ? "no_attackers_or_no_candidate_blockers" : String.join("|", rows);
        } catch (Exception e) {
            return "error=" + exceptionSummary(e);
        }
    }

    private static String blockerLegalityReason(
            CombatGroup group,
            UUID attackerId,
            Permanent blocker,
            Game game,
            boolean permanentCanBlock,
            boolean groupCanBlock
    ) {
        if (blocker == null) {
            return "missing_blocker";
        }
        if (group == null) {
            return "missing_combat_group";
        }
        if (group.getDefendingPlayerId() == null
                || !group.getDefendingPlayerId().equals(blocker.getControllerId())) {
            return "wrong_controller";
        }
        try {
            if (!blocker.isCreature(game)) {
                return "not_creature";
            }
        } catch (Exception ignored) {
            return "creature_check_error";
        }
        try {
            if (blocker.isTapped()) {
                return "tapped";
            }
        } catch (Exception ignored) {
        }
        try {
            if (blocker.isAttacking()) {
                return "attacking";
            }
        } catch (Exception ignored) {
        }
        try {
            if (blocker.getBlocking() > 0) {
                return "already_blocking";
            }
        } catch (Exception ignored) {
        }
        if (permanentCanBlock && groupCanBlock) {
            return "legal";
        }
        if (!permanentCanBlock) {
            return "permanent_canBlock_false";
        }
        if (!groupCanBlock) {
            return "combat_group_canBlock_false";
        }
        return "unknown";
    }

    private static String safePlayerName(Game game, UUID playerId) {
        try {
            Player p = game == null || playerId == null ? null : game.getPlayer(playerId);
            return p == null ? String.valueOf(playerId) : p.getName();
        } catch (Exception e) {
            return "";
        }
    }

    private static String permanentDebugName(Game game, UUID permanentId) {
        try {
            Permanent permanent = game == null || permanentId == null ? null : game.getPermanent(permanentId);
            return permanent == null ? String.valueOf(permanentId) : permanentDebugName(permanent, game);
        } catch (Exception e) {
            return String.valueOf(permanentId);
        }
    }

    private static String permanentDebugName(Permanent permanent, Game game) {
        if (permanent == null) {
            return "";
        }
        List<String> flags = new ArrayList<>();
        try {
            if (permanent.isTapped()) {
                flags.add("tapped");
            }
        } catch (Exception ignored) {
        }
        try {
            if (permanent.isAttacking()) {
                flags.add("attacking");
            }
        } catch (Exception ignored) {
        }
        try {
            if (permanent.getBlocking() > 0) {
                flags.add("blocking");
            }
        } catch (Exception ignored) {
        }
        return permanent.getName() + (flags.isEmpty() ? "" : "[" + String.join(",", flags) + "]");
    }

    private static String permanentDebugDetail(Game game, UUID permanentId) {
        try {
            Permanent permanent = game == null || permanentId == null ? null : game.getPermanent(permanentId);
            return permanent == null ? String.valueOf(permanentId) : permanentDebugDetail(permanent, game);
        } catch (Exception e) {
            return String.valueOf(permanentId);
        }
    }

    private static String permanentDebugDetail(Permanent permanent, Game game) {
        if (permanent == null) {
            return "";
        }
        List<String> flags = new ArrayList<>();
        try {
            if (permanent.isTapped()) {
                flags.add("tapped");
            }
        } catch (Exception ignored) {
        }
        try {
            if (permanent.hasSummoningSickness()) {
                flags.add("summoning_sick");
            }
        } catch (Exception ignored) {
        }
        try {
            if (permanent.isAttacking()) {
                flags.add("attacking");
            }
        } catch (Exception ignored) {
        }
        try {
            if (permanent.getBlocking() > 0) {
                flags.add("blocking");
            }
        } catch (Exception ignored) {
        }
        return permanent.getName()
                + "{id=" + permanent.getId()
                + ",controller=" + safePlayerName(game, permanent.getControllerId())
                + "(" + permanent.getControllerId() + ")"
                + ",status=" + (flags.isEmpty() ? "ready" : String.join("+", flags))
                + "}";
    }

    private static int compactSourceTurn(int rawTurn) {
        return rawTurn < 0 ? rawTurn : (rawTurn + 1) / 2;
    }

    private static double branchScore(BranchResult result, Args args) {
        return branchScore(result.won, result.turns, args);
    }

    private static float branchReturnTarget(BranchResult result) {
        return result != null && result.won ? 1.0f : -1.0f;
    }

    private static boolean usesBranchReturnTargets(StateSequenceBuilder.TrainingData td) {
        if (td == null || td.mctsVisitTargets == null) {
            return false;
        }
        int max = Math.min(td.candidateCount, td.mctsVisitTargets.length);
        for (int i = 0; i < max; i++) {
            if (td.candidateMask != null && i < td.candidateMask.length && td.candidateMask[i] == 0) {
                continue;
            }
            float v = td.mctsVisitTargets[i];
            if (v <= BRANCH_RETURN_UNOBSERVED + 0.25f || v < -1e-6f) {
                return true;
            }
        }
        return false;
    }

    private static boolean observedTargetValue(float value, boolean branchReturnTargets) {
        if (!Float.isFinite(value)) {
            return false;
        }
        if (branchReturnTargets) {
            return value > BRANCH_RETURN_UNOBSERVED + 0.25f;
        }
        return value > 1e-6f;
    }

    private static double branchScore(boolean won, int turns, Args args) {
        if (!won) {
            if (args.lossTurnBonus <= 0.0 || turns <= 0) {
                return 0.0;
            }
            double survival = Math.min(1.0, turns / 100.0);
            return Math.min(0.99, args.lossTurnBonus * survival);
        }
        double turnBonus = turns > 0 ? 1.0 / Math.sqrt(turns) : 0.0;
        return 1.0 + args.winTurnBonus * turnBonus;
    }

    private static List<TrainingExample> selectExamples(List<TrainingExample> examples, Args args) {
        Pattern includeActionTextPattern = compileOptionalPattern(
                args.includeActionTextRegex,
                "--include-action-text-regex");
        Map<String, TrainingExample> unique = new LinkedHashMap<>();
        for (TrainingExample example : examples) {
            if (!exampleMatchesActionText(example, includeActionTextPattern)) {
                continue;
            }
            if (!passesPolicyMissFilter(example.trainingData, args)) {
                continue;
            }
            unique.putIfAbsent(trainingExampleKey(example), example);
        }
        List<TrainingExample> selected = new ArrayList<>(unique.values());
        selected.sort(Comparator.comparingInt((TrainingExample e) -> e.scenario)
                .thenComparingInt(e -> e.ordinal));
        if (args.maxTrainExamples > 0 && selected.size() > args.maxTrainExamples) {
            return new ArrayList<>(selected.subList(0, args.maxTrainExamples));
        }
        return selected;
    }

    private static Pattern compileOptionalPattern(String raw, String argName) {
        if (raw == null || raw.trim().isEmpty()) {
            return null;
        }
        try {
            return Pattern.compile(raw, Pattern.CASE_INSENSITIVE);
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("Invalid regex for " + argName + ": " + raw, e);
        }
    }

    private static boolean exampleMatchesActionText(TrainingExample example, Pattern pattern) {
        if (pattern == null) {
            return true;
        }
        if (example == null) {
            return false;
        }
        if (matches(pattern, example.bestText) || matches(pattern, example.baselineText)) {
            return true;
        }
        for (String text : example.choiceTexts) {
            if (matches(pattern, text)) {
                return true;
            }
        }
        int count = Math.min(example.candidateCount, example.candidateTexts.size());
        for (int i = 0; i < count; i++) {
            if (matches(pattern, example.candidateTexts.get(i))) {
                return true;
            }
        }
        return false;
    }

    private static boolean decisionPointMatchesActionText(DecisionPoint point, Pattern pattern) {
        if (pattern == null) {
            return true;
        }
        if (point == null || point.candidateTexts == null) {
            return false;
        }
        int count = Math.min(point.trainingData.candidateCount, point.candidateTexts.size());
        for (int i = 0; i < count; i++) {
            if (matches(pattern, point.candidateTexts.get(i))) {
                return true;
            }
        }
        return false;
    }

    private static boolean matches(Pattern pattern, String text) {
        return pattern != null && text != null && pattern.matcher(text).find();
    }

    private static String trainingExampleKey(TrainingExample example) {
        return example.scenario
                + "|" + example.seed
                + "|" + example.agentDeck
                + "|" + example.oppDeck
                + "|" + example.ordinal
                + "|" + example.actionType;
    }

    private static void trainSelectedExamples(List<TrainingExample> selected, Args args) {
        List<StateSequenceBuilder.TrainingData> data = new ArrayList<>(selected.size());
        for (TrainingExample example : selected) {
            data.add(example.trainingData);
        }
        trainTrainingData(data, args);
    }

    private static void trainTrainingData(List<StateSequenceBuilder.TrainingData> selected, Args args) {
        long trainPasses = (long) selected.size()
                * Math.max(1, args.trainEpochs)
                * Math.max(1, args.candidatePermutations);
        List<StateSequenceBuilder.TrainingData> batch = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();
        Random rand = new Random(args.seed ^ 0xAC7100D5EEDL);
        for (int epoch = 0; epoch < args.trainEpochs; epoch++) {
            List<StateSequenceBuilder.TrainingData> epochExamples = new ArrayList<>(selected);
            Collections.shuffle(epochExamples, rand);
            for (StateSequenceBuilder.TrainingData example : epochExamples) {
                for (int variant = 0; variant < args.candidatePermutations; variant++) {
                    StateSequenceBuilder.TrainingData td = variant == 0
                            ? example
                            : permuteCandidates(example, rand);
                    batch.add(td);
                    rewards.add(0.0);
                    if (batch.size() >= args.batchSize) {
                        RLTrainer.sharedModel.enqueueTraining(batch, rewards);
                        batch = new ArrayList<>();
                        rewards = new ArrayList<>();
                    }
                }
            }
        }
        if (!batch.isEmpty()) {
            RLTrainer.sharedModel.enqueueTraining(batch, rewards);
        }
        long drainTimeoutMs = scaledTrainingDrainTimeoutMs(args.postTrainWaitMs, trainPasses);
        if (!RLTrainer.sharedModel.awaitTrainingDrained(drainTimeoutMs)) {
            throw new IllegalStateException("Timed out waiting for action counterfactual training queue to drain"
                    + " after " + drainTimeoutMs + " ms for " + trainPasses + " train passes");
        }
    }

    private static long scaledTrainingDrainTimeoutMs(int configuredTimeoutMs, long trainPasses) {
        long base = Math.max(0L, configuredTimeoutMs);
        // Large terminal-prefix runs can enqueue tens of thousands of small
        // supervised updates. A fixed timeout makes good search data fail
        // before the learner has had a realistic chance to consume the queue.
        long scaled = 60_000L + Math.max(0L, trainPasses) * 25L;
        return Math.max(base, scaled);
    }

    private static void runImportedTraining(Args args, Path outDir) throws Exception {
        long started = System.currentTimeMillis();
        List<Path> files = serializedTrainingDataFiles(args.importTrainingDataPath);
        if (files.isEmpty()) {
            throw new IllegalArgumentException("No serialized training data found at " + args.importTrainingDataPath);
        }
        ImportTrainingStats importStats = args.importFlatAsTerminalEpisodes
                ? ImportTrainingStats.unbalanced(
                        trainSerializedFlatTerminalEpisodes(files, args),
                        args.trainEpochs,
                        args.candidatePermutations)
                : trainSerializedTrainingData(files, args);
        saveSharedModelLatest();
        Map<String, Integer> modelStats = RLTrainer.sharedModel.getMainModelTrainingStats();
        writeTensorReplayRecords(outDir.resolve("tensor_replay_samples.csv"), Collections.emptyList());
        writeTensorReplayReadme(outDir.resolve("tensor_replay_summary.md"), Collections.emptyList());
        writeImportedTrainingReadme(outDir.resolve("README.md"), args, importStats,
                System.currentTimeMillis() - started);
        System.out.println("Action counterfactual import output: " + outDir);
        System.out.println("importedTrainingExamples=" + importStats.trainingExamples
                + " importFlatAsTerminalEpisodes=" + args.importFlatAsTerminalEpisodes
                + " trainPassSamples=" + importStats.trainPassSamples
                + branchReturnBalanceStatsLine(importStats.branchReturnBalanceStats)
                + " stats=" + modelStats);
    }

    private static void runImportedTrajectoryTraining(Args args, Path outDir) throws Exception {
        long started = System.currentTimeMillis();
        List<Path> files = serializedTrainingDataFiles(args.importTrajectoryDataPath);
        if (files.isEmpty()) {
            throw new IllegalArgumentException("No serialized trajectory data found at " + args.importTrajectoryDataPath);
        }
        TrajectoryImportStats importStats = trainSerializedTrajectoryData(files, args);
        saveSharedModelLatest();
        Map<String, Integer> stats = RLTrainer.sharedModel.getMainModelTrainingStats();
        writeTensorReplayRecords(outDir.resolve("tensor_replay_samples.csv"), Collections.emptyList());
        writeTensorReplayReadme(outDir.resolve("tensor_replay_summary.md"), Collections.emptyList());
        writeImportedTrajectoryReadme(outDir.resolve("README.md"), args, importStats,
                System.currentTimeMillis() - started);
        System.out.println("Action counterfactual trajectory import output: " + outDir);
        System.out.println("importedTrajectoryEpisodes=" + importStats.episodes
                + " importedTrajectorySteps=" + importStats.steps
                + " trainPassSamples=" + importStats.trainPassSamples
                + " stats=" + stats);
    }

    private static void runImportedScoreProbe(Args args, Path outDir) throws Exception {
        long started = System.currentTimeMillis();
        List<Path> files = serializedTrainingDataFiles(args.scoreTrainingDataPath);
        if (files.isEmpty()) {
            throw new IllegalArgumentException("No serialized training data found at " + args.scoreTrainingDataPath);
        }
        Files.createDirectories(outDir);
        int maxExamples = args.scoreMaxExamples;
        List<StateSequenceBuilder.TrainingData> records = loadFilteredSerializedTrainingData(files, args, maxExamples);
        ScoreProbeStats stats = scoreTrainingRecords(records, outDir.resolve("import_score_samples.csv"));
        StringBuilder sb = new StringBuilder();
        sb.append("# Imported Score Probe\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("score_training_data_path: ").append(args.scoreTrainingDataPath).append('\n');
        sb.append("score_max_examples: ").append(maxExamples).append('\n');
        appendScoreProbeStats(sb, stats);
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", (System.currentTimeMillis() - started) / 1000.0)).append('\n');
        Files.write(outDir.resolve("README.md"), sb.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("Imported score probe output: " + outDir);
        System.out.println(scoreProbeStatsLine("score", stats));
    }

    private static void runImportedFitScoreProbe(Args args, Path outDir) throws Exception {
        long started = System.currentTimeMillis();
        Path dataPath = args.importTrainingDataPath != null ? args.importTrainingDataPath : args.scoreTrainingDataPath;
        if (dataPath == null) {
            throw new IllegalArgumentException("--fit-score-probe requires --import-training-data-path or --score-training-data-path");
        }
        List<Path> files = serializedTrainingDataFiles(dataPath);
        if (files.isEmpty()) {
            throw new IllegalArgumentException("No serialized training data found at " + dataPath);
        }
        Files.createDirectories(outDir);
        int maxExamples = args.maxTrainExamples > 0 ? args.maxTrainExamples : args.scoreMaxExamples;
        List<StateSequenceBuilder.TrainingData> records = loadFilteredSerializedTrainingData(files, args, maxExamples);
        ScoreProbeStats before = scoreTrainingRecords(records, outDir.resolve("fit_score_before.csv"));
        long trainPasses = trainSerializedTrainingRecords(records, args);
        ScoreProbeStats after = scoreTrainingRecords(records, outDir.resolve("fit_score_after.csv"));
        saveSharedModelLatest();

        StringBuilder sb = new StringBuilder();
        sb.append("# Imported Fit Score Probe\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("training_data_path: ").append(dataPath).append('\n');
        sb.append("max_examples: ").append(maxExamples).append('\n');
        sb.append("train_epochs: ").append(args.trainEpochs).append('\n');
        sb.append("batch_size: ").append(args.batchSize).append('\n');
        sb.append("candidate_permutations: ").append(args.candidatePermutations).append('\n');
        sb.append("train_pass_samples: ").append(trainPasses).append('\n');
        sb.append("skip_pass_training: ").append(args.skipPassTraining).append('\n');
        sb.append("\n## Before\n\n");
        appendScoreProbeStats(sb, before);
        sb.append("\n## After\n\n");
        appendScoreProbeStats(sb, after);
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", (System.currentTimeMillis() - started) / 1000.0)).append('\n');
        Files.write(outDir.resolve("README.md"), sb.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("Imported fit score probe output: " + outDir);
        System.out.println(scoreProbeStatsLine("before", before));
        System.out.println("trainPassSamples=" + trainPasses);
        System.out.println(scoreProbeStatsLine("after", after));
    }

    private static void saveSharedModelLatest() throws IOException {
        Path modelPath = Paths.get(RLLogPaths.MODEL_FILE_PATH).toAbsolutePath().normalize();
        if (modelPath.getParent() != null) {
            Files.createDirectories(modelPath.getParent());
        }
        RLTrainer.sharedModel.saveModel(modelPath.toString());
        Path latestModelPath = modelPath.resolveSibling("model_latest.pt");
        Files.copy(modelPath, latestModelPath, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void writeSerializedTrainingData(Path path, List<TrainingExample> selected) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<StateSequenceBuilder.TrainingData> data = new ArrayList<>(selected.size());
        for (TrainingExample example : selected) {
            data.add(example.trainingData);
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(new SerializedTrainingDataFile(data));
        }
    }

    private static void writeSerializedTrajectoryData(Path path, List<TrajectoryTrainingEpisode> episodes)
            throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(new SerializedTrajectoryDataFile(
                    episodes == null ? Collections.emptyList() : episodes));
        }
    }

    private static List<StateSequenceBuilder.TrainingData> loadSerializedTrainingData(Path path)
            throws IOException, ClassNotFoundException {
        List<Path> files = serializedTrainingDataFiles(path);
        List<StateSequenceBuilder.TrainingData> data = new ArrayList<>();
        for (Path file : files) {
            data.addAll(loadSerializedTrainingDataFile(file));
        }
        return data;
    }

    private static ImportTrainingStats trainSerializedTrainingData(List<Path> files, Args args)
            throws IOException, ClassNotFoundException {
        Random rand = new Random(args.seed ^ 0x51A7EBC0DEL);
        int baseRecords = 0;
        long totalTrainPasses = 0L;
        int maxBaseRecords = args.maxTrainExamples > 0 ? args.maxTrainExamples : Integer.MAX_VALUE;
        BranchReturnBalanceStats branchStats = args.branchReturnBalance
                ? countEligibleBranchReturnLabels(files, args)
                : null;
        if (branchStats != null) {
            System.out.println("branchReturnBalanceScan"
                    + " positive=" + branchStats.eligiblePositive
                    + " negative=" + branchStats.eligibleNegative
                    + " none=" + branchStats.eligibleNone
                    + " maxNegativesPerPositive=" + args.branchReturnMaxNegativesPerPositive);
        }
        for (int epoch = 0; epoch < args.trainEpochs; epoch++) {
            List<Path> epochFiles = new ArrayList<>(files);
            Collections.shuffle(epochFiles, rand);
            int epochBaseRecords = 0;
            int fileOrdinal = 0;
            long epochTrainPasses = 0L;
            int maxEpochNegatives = branchStats == null
                    ? Integer.MAX_VALUE
                    : branchStats.maxAcceptedNegative(args.branchReturnMaxNegativesPerPositive);
            int remainingEpochNegatives = branchStats == null ? 0 : branchStats.eligibleNegative;
            int acceptedEpochNegatives = 0;
            List<StateSequenceBuilder.TrainingData> batch = new ArrayList<>();
            List<Double> rewards = new ArrayList<>();
            for (Path file : epochFiles) {
                if (epochBaseRecords >= maxBaseRecords) {
                    break;
                }
                fileOrdinal++;
                List<StateSequenceBuilder.TrainingData> records = loadSerializedTrainingDataFile(file);
                Collections.shuffle(records, rand);
                long fileTrainPasses = 0L;
                for (StateSequenceBuilder.TrainingData record : records) {
                    if (epochBaseRecords >= maxBaseRecords) {
                        break;
                    }
                    if (!passesSerializedImportFilters(record, args)) {
                        continue;
                    }
                    BranchReturnLabel branchLabel = BranchReturnLabel.NONE;
                    if (branchStats != null) {
                        branchLabel = branchReturnLabel(record, args);
                        if (branchLabel == BranchReturnLabel.NONE) {
                            if (epoch == 0) {
                                branchStats.skippedNone++;
                            }
                            continue;
                        }
                        if (branchLabel == BranchReturnLabel.NEGATIVE) {
                            boolean acceptNegative = acceptBalancedNegative(
                                    rand,
                                    acceptedEpochNegatives,
                                    maxEpochNegatives,
                                    remainingEpochNegatives);
                            remainingEpochNegatives--;
                            if (!acceptNegative) {
                                if (epoch == 0) {
                                    branchStats.skippedNegative++;
                                }
                                continue;
                            }
                            acceptedEpochNegatives++;
                        }
                        if (epoch == 0) {
                            branchStats.accept(branchLabel);
                        }
                    }
                    if (epoch == 0) {
                        baseRecords++;
                    }
                    epochBaseRecords++;
                    for (int variant = 0; variant < args.candidatePermutations; variant++) {
                        StateSequenceBuilder.TrainingData td = variant == 0
                                ? record
                                : permuteCandidates(record, rand);
                        batch.add(td);
                        rewards.add(0.0);
                        fileTrainPasses++;
                        epochTrainPasses++;
                        totalTrainPasses++;
                        if (batch.size() >= args.batchSize) {
                            RLTrainer.sharedModel.enqueueTraining(batch, rewards);
                            batch = new ArrayList<>();
                            rewards = new ArrayList<>();
                        }
                    }
                }
                if (fileOrdinal % 10 == 0 || fileOrdinal == epochFiles.size()) {
                    System.out.println(String.format(Locale.US,
                            "importProgress epoch=%d/%d files=%d/%d baseRecords=%d totalTrainPasses=%d lastFilePasses=%d",
                            epoch + 1, args.trainEpochs, fileOrdinal, epochFiles.size(),
                            baseRecords, totalTrainPasses, fileTrainPasses));
                }
                records.clear();
                System.gc();
            }
            if (!batch.isEmpty()) {
                RLTrainer.sharedModel.enqueueTraining(batch, rewards);
            }
            long drainTimeoutMs = scaledTrainingDrainTimeoutMs(args.postTrainWaitMs, epochTrainPasses);
            if (!RLTrainer.sharedModel.awaitTrainingDrained(drainTimeoutMs)) {
                throw new IllegalStateException("Timed out waiting for imported action training queue to drain"
                        + " after " + drainTimeoutMs + " ms for " + epochTrainPasses
                        + " train passes in epoch " + (epoch + 1));
            }
            System.out.println(String.format(Locale.US,
                    "importEpochDrain epoch=%d/%d epochTrainPasses=%d totalTrainPasses=%d",
                    epoch + 1, args.trainEpochs, epochTrainPasses, totalTrainPasses));
        }
        return new ImportTrainingStats(baseRecords, totalTrainPasses, branchStats);
    }

    private static int trainSerializedFlatTerminalEpisodes(List<Path> files, Args args)
            throws IOException, ClassNotFoundException {
        Random rand = new Random(args.seed ^ 0x5EED7711L);
        int baseRecords = 0;
        long totalTrainPasses = 0L;
        int maxBaseRecords = args.maxTrainExamples > 0 ? args.maxTrainExamples : Integer.MAX_VALUE;
        for (int epoch = 0; epoch < args.trainEpochs; epoch++) {
            List<Path> epochFiles = new ArrayList<>(files);
            Collections.shuffle(epochFiles, rand);
            int epochBaseRecords = 0;
            int fileOrdinal = 0;
            long epochTrainPasses = 0L;
            for (Path file : epochFiles) {
                if (epochBaseRecords >= maxBaseRecords) {
                    break;
                }
                fileOrdinal++;
                List<StateSequenceBuilder.TrainingData> records = loadSerializedTrainingDataFile(file);
                Collections.shuffle(records, rand);
                long fileTrainPasses = 0L;
                for (StateSequenceBuilder.TrainingData record : records) {
                    if (epochBaseRecords >= maxBaseRecords) {
                        break;
                    }
                    if (!passesSerializedImportFilters(record, args)) {
                        continue;
                    }
                    if (epoch == 0) {
                        baseRecords++;
                    }
                    epochBaseRecords++;
                    for (int variant = 0; variant < args.candidatePermutations; variant++) {
                        StateSequenceBuilder.TrainingData td = variant == 0
                                ? record
                                : permuteCandidates(record, rand);
                        RLTrainer.sharedModel.enqueueTraining(
                                Collections.singletonList(td),
                                Collections.singletonList(args.trajectoryFinalReward));
                        fileTrainPasses++;
                        epochTrainPasses++;
                        totalTrainPasses++;
                    }
                }
                if (fileOrdinal % 10 == 0 || fileOrdinal == epochFiles.size()) {
                    System.out.println(String.format(Locale.US,
                            "flatTerminalImportProgress epoch=%d/%d files=%d/%d baseRecords=%d totalTrainPasses=%d lastFilePasses=%d",
                            epoch + 1, args.trainEpochs, fileOrdinal, epochFiles.size(),
                            baseRecords, totalTrainPasses, fileTrainPasses));
                }
                records.clear();
                System.gc();
            }
            long drainTimeoutMs = scaledTrainingDrainTimeoutMs(args.postTrainWaitMs, epochTrainPasses);
            if (!RLTrainer.sharedModel.awaitTrainingDrained(drainTimeoutMs)) {
                throw new IllegalStateException("Timed out waiting for flat terminal import queue to drain"
                        + " after " + drainTimeoutMs + " ms for " + epochTrainPasses
                        + " train passes in epoch " + (epoch + 1));
            }
            System.out.println(String.format(Locale.US,
                    "flatTerminalImportEpochDrain epoch=%d/%d epochTrainPasses=%d totalTrainPasses=%d",
                    epoch + 1, args.trainEpochs, epochTrainPasses, totalTrainPasses));
        }
        return baseRecords;
    }

    private static TrajectoryImportStats trainSerializedTrajectoryData(List<Path> files, Args args)
            throws IOException, ClassNotFoundException {
        Random rand = new Random(args.seed ^ 0x77A7EBC0DEL);
        int baseEpisodes = 0;
        int baseSteps = 0;
        long totalTrainPasses = 0L;
        int maxBaseSteps = args.maxTrainExamples > 0 ? args.maxTrainExamples : Integer.MAX_VALUE;
        for (int epoch = 0; epoch < args.trainEpochs; epoch++) {
            List<TrajectoryTrainingEpisode> epochEpisodes = new ArrayList<>();
            int fileOrdinal = 0;
            for (Path file : files) {
                fileOrdinal++;
                epochEpisodes.addAll(loadSerializedTrajectoryDataFile(file));
                if (fileOrdinal % 10 == 0 || fileOrdinal == files.size()) {
                    System.out.println(String.format(Locale.US,
                            "trajectoryImportLoad epoch=%d/%d files=%d/%d episodes=%d",
                            epoch + 1, args.trainEpochs, fileOrdinal, files.size(), epochEpisodes.size()));
                }
            }
            Collections.shuffle(epochEpisodes, rand);

            long epochTrainPasses = 0L;
            int epochEpisodesUsed = 0;
            for (TrajectoryTrainingEpisode episode : epochEpisodes) {
                if (baseSteps >= maxBaseSteps) {
                    break;
                }
                TrajectoryTrainingEpisode filtered = filterTrajectoryEpisode(episode, args);
                if (filtered.records.isEmpty()) {
                    continue;
                }
                if (epoch == 0) {
                    baseEpisodes++;
                    baseSteps += filtered.records.size();
                }
                for (int variant = 0; variant < args.candidatePermutations; variant++) {
                    TrajectoryTrainingEpisode trainEpisode = variant == 0
                            ? filtered
                            : permuteTrajectoryEpisode(filtered, rand);
                    RLTrainer.sharedModel.enqueueTraining(trainEpisode.records, trainEpisode.rewards);
                    epochTrainPasses += trainEpisode.records.size();
                    totalTrainPasses += trainEpisode.records.size();
                }
                epochEpisodesUsed++;
            }
            long drainTimeoutMs = scaledTrainingDrainTimeoutMs(args.postTrainWaitMs, epochTrainPasses);
            if (!RLTrainer.sharedModel.awaitTrainingDrained(drainTimeoutMs)) {
                throw new IllegalStateException("Timed out waiting for imported trajectory training queue to drain"
                        + " after " + drainTimeoutMs + " ms for " + epochTrainPasses
                        + " train passes in epoch " + (epoch + 1));
            }
            System.out.println(String.format(Locale.US,
                    "trajectoryImportEpochDrain epoch=%d/%d episodes=%d epochTrainPasses=%d totalTrainPasses=%d",
                    epoch + 1, args.trainEpochs, epochEpisodesUsed, epochTrainPasses, totalTrainPasses));
        }
        return new TrajectoryImportStats(baseEpisodes, baseSteps, totalTrainPasses);
    }

    private static TrajectoryTrainingEpisode filterTrajectoryEpisode(TrajectoryTrainingEpisode episode, Args args) {
        if (episode == null || episode.records == null || episode.records.isEmpty()) {
            return new TrajectoryTrainingEpisode("", Collections.emptyList(), Collections.emptyList());
        }
        List<StateSequenceBuilder.TrainingData> records = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();
        for (int i = 0; i < episode.records.size(); i++) {
            StateSequenceBuilder.TrainingData record = episode.records.get(i);
            if (record == null || !args.targetTypes.contains(record.actionType)) {
                continue;
            }
            if (args.skipPassTraining && isLikelyPassTarget(record)) {
                continue;
            }
            if (!passesTargetMargin(record, args)) {
                continue;
            }
            if (!passesPolicyMissFilter(record, args)) {
                continue;
            }
            records.add(record);
            rewards.add(i < episode.rewards.size() ? episode.rewards.get(i) : 0.0);
        }
        if (!records.isEmpty()) {
            boolean anyTerminalReward = false;
            for (Double reward : rewards) {
                if (reward != null && Math.abs(reward) > 1e-9) {
                    anyTerminalReward = true;
                    break;
                }
            }
            if (!anyTerminalReward) {
                rewards.set(rewards.size() - 1, args.trajectoryFinalReward);
            }
        }
        if (args.branchTrajectoryPairMode) {
            if (records.size() != 2 || rewards.size() != 2) {
                return new TrajectoryTrainingEpisode(episode.key, Collections.emptyList(), Collections.emptyList());
            }
            double first = rewards.get(0) == null ? 0.0 : rewards.get(0);
            double second = rewards.get(1) == null ? 0.0 : rewards.get(1);
            if (first <= second) {
                return new TrajectoryTrainingEpisode(episode.key, Collections.emptyList(), Collections.emptyList());
            }
        }
        return new TrajectoryTrainingEpisode(episode.key, records, rewards);
    }

    private static TrajectoryTrainingEpisode permuteTrajectoryEpisode(
            TrajectoryTrainingEpisode episode,
            Random rand
    ) {
        List<StateSequenceBuilder.TrainingData> records = new ArrayList<>(episode.records.size());
        for (StateSequenceBuilder.TrainingData record : episode.records) {
            records.add(permuteCandidates(record, rand));
        }
        return new TrajectoryTrainingEpisode(episode.key, records, episode.rewards);
    }

    private static List<StateSequenceBuilder.TrainingData> loadFilteredSerializedTrainingData(
            List<Path> files,
            Args args,
            int maxExamples
    ) throws IOException, ClassNotFoundException {
        List<StateSequenceBuilder.TrainingData> out = new ArrayList<>();
        int max = Math.max(1, maxExamples);
        for (Path file : files) {
            if (out.size() >= max) {
                break;
            }
            List<StateSequenceBuilder.TrainingData> records = loadSerializedTrainingDataFile(file);
            for (StateSequenceBuilder.TrainingData record : records) {
                if (out.size() >= max) {
                    break;
                }
                if (!passesSerializedImportFilters(record, args)) {
                    continue;
                }
                out.add(record);
            }
            records.clear();
            System.gc();
        }
        return out;
    }

    private static boolean passesSerializedImportFilters(StateSequenceBuilder.TrainingData record, Args args) {
        if (record == null || args == null) {
            return false;
        }
        if (!args.targetTypes.contains(record.actionType)) {
            return false;
        }
        if (args.skipPassTraining && isLikelyPassTarget(record)) {
            return false;
        }
        if (!passesTargetMargin(record, args)) {
            return false;
        }
        return passesPolicyMissFilter(record, args);
    }

    private static BranchReturnBalanceStats countEligibleBranchReturnLabels(List<Path> files, Args args)
            throws IOException, ClassNotFoundException {
        BranchReturnBalanceStats stats = new BranchReturnBalanceStats();
        for (Path file : files) {
            List<StateSequenceBuilder.TrainingData> records = loadSerializedTrainingDataFile(file);
            for (StateSequenceBuilder.TrainingData record : records) {
                if (!passesSerializedImportFilters(record, args)) {
                    continue;
                }
                stats.addEligible(branchReturnLabel(record, args));
            }
            records.clear();
            System.gc();
        }
        return stats;
    }

    private static boolean acceptBalancedNegative(
            Random rand,
            int acceptedNegatives,
            int maxNegatives,
            int remainingNegativesIncludingCurrent
    ) {
        if (maxNegatives <= acceptedNegatives || remainingNegativesIncludingCurrent <= 0) {
            return false;
        }
        int quota = maxNegatives - acceptedNegatives;
        if (quota >= remainingNegativesIncludingCurrent) {
            return true;
        }
        return rand.nextDouble() < ((double) quota / (double) remainingNegativesIncludingCurrent);
    }

    private static BranchReturnLabel branchReturnLabel(StateSequenceBuilder.TrainingData record, Args args) {
        if (record == null || record.mctsVisitTargets == null || record.candidateCount <= 0) {
            return BranchReturnLabel.NONE;
        }
        boolean branchTargets = (args != null && args.branchReturnTargets) || usesBranchReturnTargets(record);
        int limit = Math.min(record.candidateCount, record.mctsVisitTargets.length);
        boolean observed = false;
        float bestObserved = -Float.MAX_VALUE;
        for (int i = 0; i < limit; i++) {
            if (record.candidateMask != null && i < record.candidateMask.length && record.candidateMask[i] == 0) {
                continue;
            }
            float v = record.mctsVisitTargets[i];
            if (!observedTargetValue(v, branchTargets)) {
                continue;
            }
            observed = true;
            if (v > bestObserved) {
                bestObserved = v;
            }
        }
        if (!observed) {
            return BranchReturnLabel.NONE;
        }
        return bestObserved > 0.0f ? BranchReturnLabel.POSITIVE : BranchReturnLabel.NEGATIVE;
    }

    private static String branchReturnBalanceStatsLine(BranchReturnBalanceStats stats) {
        if (stats == null) {
            return "";
        }
        return " branchReturnBalance="
                + "eligiblePositive=" + stats.eligiblePositive
                + ",eligibleNegative=" + stats.eligibleNegative
                + ",eligibleNone=" + stats.eligibleNone
                + ",acceptedPositive=" + stats.acceptedPositive
                + ",acceptedNegative=" + stats.acceptedNegative
                + ",skippedNegative=" + stats.skippedNegative
                + ",skippedNone=" + stats.skippedNone;
    }

    private static boolean passesTargetMargin(StateSequenceBuilder.TrainingData record, Args args) {
        if (record == null || args == null || args.minTargetMargin <= 0.0) {
            return true;
        }
        if (record.mctsVisitTargets == null || record.candidateCount < 2) {
            return false;
        }
        int limit = Math.min(record.candidateCount, record.mctsVisitTargets.length);
        float best = -1.0f;
        float second = -1.0f;
        for (int i = 0; i < limit; i++) {
            if (record.candidateMask != null && i < record.candidateMask.length && record.candidateMask[i] == 0) {
                continue;
            }
            float v = record.mctsVisitTargets[i];
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                continue;
            }
            if (v > best) {
                second = best;
                best = v;
            } else if (v > second) {
                second = v;
            }
        }
        return best >= 0.0f && second >= 0.0f && (best - second) >= args.minTargetMargin;
    }

    private static boolean passesPolicyMissFilter(StateSequenceBuilder.TrainingData record, Args args) {
        if (record == null || args == null || !args.policyMissOnly) {
            return true;
        }
        if (record.mctsVisitTargets == null || record.candidateCount < 2) {
            return false;
        }
        float[] policy = scoreTrainingData(record, args);
        ScoreProbeRecord probe = ScoreProbeRecord.from(0, record, policy);
        return !probe.targetSetMatch;
    }

    private static long trainSerializedTrainingRecords(
            List<StateSequenceBuilder.TrainingData> records,
            Args args
    ) {
        Random rand = new Random(args.seed ^ 0x51A7EBC0DEL);
        long totalTrainPasses = 0L;
        for (int epoch = 0; epoch < args.trainEpochs; epoch++) {
            List<StateSequenceBuilder.TrainingData> epochRecords = new ArrayList<>(records);
            Collections.shuffle(epochRecords, rand);
            List<StateSequenceBuilder.TrainingData> batch = new ArrayList<>();
            List<Double> rewards = new ArrayList<>();
            long epochTrainPasses = 0L;
            for (StateSequenceBuilder.TrainingData record : epochRecords) {
                for (int variant = 0; variant < args.candidatePermutations; variant++) {
                    StateSequenceBuilder.TrainingData td = variant == 0
                            ? record
                            : permuteCandidates(record, rand);
                    batch.add(td);
                    rewards.add(0.0);
                    epochTrainPasses++;
                    totalTrainPasses++;
                    if (batch.size() >= args.batchSize) {
                        RLTrainer.sharedModel.enqueueTraining(batch, rewards);
                        batch = new ArrayList<>();
                        rewards = new ArrayList<>();
                    }
                }
            }
            if (!batch.isEmpty()) {
                RLTrainer.sharedModel.enqueueTraining(batch, rewards);
            }
            long drainTimeoutMs = scaledTrainingDrainTimeoutMs(args.postTrainWaitMs, epochTrainPasses);
            if (!RLTrainer.sharedModel.awaitTrainingDrained(drainTimeoutMs)) {
                throw new IllegalStateException("Timed out waiting for fixed-subset training queue to drain"
                        + " after " + drainTimeoutMs + " ms for " + epochTrainPasses
                        + " train passes");
            }
            System.out.println(String.format(Locale.US,
                    "fitProbeProgress epoch=%d/%d records=%d totalTrainPasses=%d lastEpochPasses=%d",
                    epoch + 1, args.trainEpochs, records.size(), totalTrainPasses, epochTrainPasses));
        }
        return totalTrainPasses;
    }

    private static ScoreProbeStats scoreTrainingRecords(
            List<StateSequenceBuilder.TrainingData> records,
            Path samplesPath
    ) throws IOException {
        int total = 0;
        int top1 = 0;
        int targetSetTop1 = 0;
        int ranked = 0;
        int missingTarget = 0;
        double targetProbSum = 0.0;
        double topProbSum = 0.0;
        double topTargetMassSum = 0.0;
        double rankSum = 0.0;
        try (BufferedWriter samples = Files.newBufferedWriter(samplesPath, StandardCharsets.UTF_8)) {
            samples.write("ordinal,action_type,candidate_count,target_idx,top_idx,target_prob,top_prob,target_rank,top1_match,valid_count,state_hash,candidate_hash,mcts_target_sum,cand0_id,cand1_id,cand0_feat_hash,cand1_feat_hash,cand01_feat_equal,target_positive_count,top_target_mass,target_set_match,target_q,top_q,q_top_idx,q_target_rank,q_top_target_mass,q_top1_match\n");
            for (StateSequenceBuilder.TrainingData td : records) {
                PolicyScoreResult scored = scoreTrainingDataDetailed(td, null);
                ScoreProbeRecord record = ScoreProbeRecord.from(total, td, scored.policyScores, scored.candidateQScores);
                total++;
                if (record.targetIdx < 0) {
                    missingTarget++;
                }
                if (record.top1Match) {
                    top1++;
                }
                if (record.targetSetMatch) {
                    targetSetTop1++;
                }
                if (record.targetRank > 0) {
                    ranked++;
                    rankSum += record.targetRank;
                }
                targetProbSum += record.targetProb;
                topProbSum += record.topProb;
                topTargetMassSum += record.topTargetMass;
                samples.write(record.toCsv());
                samples.write('\n');
            }
        }
        double accuracy = total > 0 ? top1 / (double) total : 0.0;
        double avgTargetProb = total > 0 ? targetProbSum / total : 0.0;
        double avgTopProb = total > 0 ? topProbSum / total : 0.0;
        double targetSetAccuracy = total > 0 ? targetSetTop1 / (double) total : 0.0;
        double avgTopTargetMass = total > 0 ? topTargetMassSum / total : 0.0;
        double avgRank = ranked > 0 ? rankSum / ranked : 0.0;
        return new ScoreProbeStats(total, top1, accuracy, targetSetTop1, targetSetAccuracy,
                missingTarget, avgTargetProb, avgTopProb, avgTopTargetMass, avgRank);
    }

    private static void appendScoreProbeStats(StringBuilder sb, ScoreProbeStats stats) {
        sb.append("examples_scored: ").append(stats.total).append('\n');
        sb.append("top1_matches: ").append(stats.top1).append('\n');
        sb.append("top1_accuracy: ").append(String.format(Locale.US, "%.4f", stats.accuracy)).append('\n');
        sb.append("target_set_top1_matches: ").append(stats.targetSetTop1).append('\n');
        sb.append("target_set_top1_accuracy: ").append(String.format(Locale.US, "%.4f", stats.targetSetAccuracy)).append('\n');
        sb.append("missing_targets: ").append(stats.missingTarget).append('\n');
        sb.append("avg_target_prob: ").append(String.format(Locale.US, "%.6f", stats.avgTargetProb)).append('\n');
        sb.append("avg_top_prob: ").append(String.format(Locale.US, "%.6f", stats.avgTopProb)).append('\n');
        sb.append("avg_top_target_mass: ").append(String.format(Locale.US, "%.6f", stats.avgTopTargetMass)).append('\n');
        sb.append("avg_target_rank: ").append(String.format(Locale.US, "%.4f", stats.avgRank)).append('\n');
    }

    private static String scoreProbeStatsLine(String label, ScoreProbeStats stats) {
        return label + "ScoreExamples=" + stats.total
                + " top1=" + stats.top1 + "/" + stats.total
                + " accuracy=" + String.format(Locale.US, "%.4f", stats.accuracy)
                + " targetSetTop1=" + stats.targetSetTop1 + "/" + stats.total
                + " targetSetAccuracy=" + String.format(Locale.US, "%.4f", stats.targetSetAccuracy)
                + " avgTargetProb=" + String.format(Locale.US, "%.6f", stats.avgTargetProb)
                + " avgRank=" + String.format(Locale.US, "%.4f", stats.avgRank);
    }

    private static List<StateSequenceBuilder.TrainingData> loadSerializedTrainingDataFile(Path file)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(file))) {
            Object obj = in.readObject();
            if (obj instanceof SerializedTrainingDataFile) {
                return new ArrayList<>(((SerializedTrainingDataFile) obj).records);
            }
            if (obj instanceof List<?>) {
                List<StateSequenceBuilder.TrainingData> records = new ArrayList<>();
                for (Object item : (List<?>) obj) {
                    if (item instanceof StateSequenceBuilder.TrainingData) {
                        records.add((StateSequenceBuilder.TrainingData) item);
                    }
                }
                return records;
            }
            throw new IOException("Unsupported serialized training data object in " + file
                    + ": " + obj.getClass().getName());
        }
    }

    private static List<TrajectoryTrainingEpisode> loadSerializedTrajectoryDataFile(Path file)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(file))) {
            Object obj = in.readObject();
            if (obj instanceof SerializedTrajectoryDataFile) {
                return new ArrayList<>(((SerializedTrajectoryDataFile) obj).episodes);
            }
            if (obj instanceof TrajectoryTrainingEpisode) {
                return Collections.singletonList((TrajectoryTrainingEpisode) obj);
            }
            if (obj instanceof List<?>) {
                List<TrajectoryTrainingEpisode> episodes = new ArrayList<>();
                for (Object item : (List<?>) obj) {
                    if (item instanceof TrajectoryTrainingEpisode) {
                        episodes.add((TrajectoryTrainingEpisode) item);
                    }
                }
                return episodes;
            }
            throw new IOException("Unsupported serialized trajectory data object in " + file
                    + ": " + obj.getClass().getName());
        }
    }

    private static List<Path> serializedTrainingDataFiles(Path path) throws IOException {
        if (Files.isRegularFile(path)) {
            return Collections.singletonList(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException("Serialized training data path does not exist: " + path);
        }
        try (java.util.stream.Stream<Path> stream = Files.walk(path)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return name.endsWith(".ser") || name.endsWith(".bin") || name.endsWith(".dat");
                    })
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    private static StateSequenceBuilder.TrainingData permuteCandidates(
            StateSequenceBuilder.TrainingData source,
            Random rand
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        List<Integer> valid = new ArrayList<>();
        for (int i = 0; i < max; i++) {
            if (source.candidateMask[i] != 0) {
                valid.add(i);
            }
        }
        if (valid.size() <= 1) {
            return source;
        }
        Collections.shuffle(valid, rand);
        int[] oldToNew = new int[max];
        Arrays.fill(oldToNew, -1);
        int[] ids = new int[max];
        float[][] features = new float[max][StateSequenceBuilder.TrainingData.CAND_FEAT_DIM];
        int[] mask = new int[max];
        float[] targets = new float[max];
        for (int newIdx = 0; newIdx < valid.size(); newIdx++) {
            int oldIdx = valid.get(newIdx);
            oldToNew[oldIdx] = newIdx;
            ids[newIdx] = source.candidateActionIds[oldIdx];
            features[newIdx] = Arrays.copyOf(source.candidateFeatures[oldIdx], source.candidateFeatures[oldIdx].length);
            mask[newIdx] = source.candidateMask[oldIdx];
            if (source.mctsVisitTargets != null && oldIdx < source.mctsVisitTargets.length) {
                targets[newIdx] = source.mctsVisitTargets[oldIdx];
            }
        }
        int[] chosen = new int[source.chosenIndices.length];
        Arrays.fill(chosen, -1);
        for (int i = 0; i < Math.min(source.chosenCount, source.chosenIndices.length); i++) {
            int oldIdx = source.chosenIndices[i];
            if (oldIdx >= 0 && oldIdx < oldToNew.length && oldToNew[oldIdx] >= 0) {
                chosen[i] = oldToNew[oldIdx];
            }
        }
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                ids,
                features,
                mask,
                source.chosenCount,
                chosen,
                source.oldLogpTotal,
                source.oldValue,
                source.actionType,
                source.stepReward
        );
        td.setMctsVisitTargets(targets);
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        return td;
    }

    private static List<TensorReplayRecord> runTensorReplay(List<TrainingExample> selected) {
        List<TensorReplayRecord> records = new ArrayList<>();
        for (TrainingExample example : selected) {
            records.add(TensorReplayRecord.from(example, scoreTrainingData(example.trainingData)));
        }
        return records;
    }

    private static StateSequenceBuilder.TrainingData copyWithTarget(
            StateSequenceBuilder.TrainingData source,
            int bestIdx,
            float[] target
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int[] chosen = Arrays.copyOf(source.chosenIndices, source.chosenIndices.length);
        Arrays.fill(chosen, -1);
        chosen[0] = Math.max(0, Math.min(bestIdx, max - 1));
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                Arrays.copyOf(source.candidateActionIds, source.candidateActionIds.length),
                copy2d(source.candidateFeatures),
                Arrays.copyOf(source.candidateMask, source.candidateMask.length),
                1,
                chosen,
                0.0f,
                0.0f,
                source.actionType,
                0.0
        );
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        td.setMctsVisitTargets(target);
        return td;
    }

    private static StateSequenceBuilder.TrainingData copyWithChoiceForTrajectory(
            StateSequenceBuilder.TrainingData source,
            List<Integer> choice
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int[] chosen = new int[source.chosenIndices.length];
        Arrays.fill(chosen, -1);
        int count = 0;
        for (Integer idx : choice) {
            if (idx == null || idx < 0 || idx >= max || count >= chosen.length) {
                continue;
            }
            chosen[count++] = idx;
        }
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                Arrays.copyOf(source.candidateActionIds, source.candidateActionIds.length),
                copy2d(source.candidateFeatures),
                Arrays.copyOf(source.candidateMask, source.candidateMask.length),
                count,
                chosen,
                source.oldLogpTotal,
                source.oldValue,
                source.actionType,
                source.stepReward
        );
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        return td;
    }

    private static StateSequenceBuilder.TrainingData copyWithFullChoiceTarget(
            StateSequenceBuilder.TrainingData source,
            List<Integer> choice,
            float[] target
    ) {
        int max = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int[] chosen = new int[source.chosenIndices.length];
        Arrays.fill(chosen, -1);
        int count = 0;
        for (Integer idx : choice) {
            if (idx == null || idx < 0 || idx >= max || count >= chosen.length) {
                continue;
            }
            chosen[count++] = idx;
        }
        StateSequenceBuilder.TrainingData td = new StateSequenceBuilder.TrainingData(
                source.state,
                source.candidateCount,
                Arrays.copyOf(source.candidateActionIds, source.candidateActionIds.length),
                copy2d(source.candidateFeatures),
                Arrays.copyOf(source.candidateMask, source.candidateMask.length),
                count,
                chosen,
                0.0f,
                0.0f,
                source.actionType,
                0.0
        );
        td.setBeliefArchetypeLabel(source.beliefArchetypeLabel);
        td.setMctsVisitTargets(target);
        return td;
    }

    private static float[][] copy2d(float[][] src) {
        float[][] out = new float[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = Arrays.copyOf(src[i], src[i].length);
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
            case MULLIGAN:
                return "mulligan";
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

    private static Deck shuffledCopy(Deck source, long seed) {
        Deck deck = source.copy();
        List<Card> cards = new ArrayList<>(deck.getCards());
        Collections.shuffle(cards, new Random(seed));
        deck.getCards().clear();
        for (Card card : cards) {
            deck.getCards().add(card);
        }
        return deck;
    }

    private static void stackOpeningHand(Deck deck, List<String> requestedNames) {
        if (deck == null || requestedNames == null || requestedNames.isEmpty()) {
            return;
        }
        List<Card> remaining = new ArrayList<>(deck.getCards());
        List<Card> stacked = new ArrayList<>();
        appendRequestedCards(remaining, stacked, requestedNames);
        if (stacked.isEmpty()) {
            return;
        }
        deck.getCards().clear();
        for (Card card : stacked) {
            deck.getCards().add(card);
        }
        for (Card card : remaining) {
            deck.getCards().add(card);
        }
    }

    private static void stackOpeningState(Deck deck, List<String> handNames, List<String> libraryNames) {
        if (deck == null || libraryNames == null || libraryNames.isEmpty()) {
            stackOpeningHand(deck, handNames);
            return;
        }
        List<Card> remaining = new ArrayList<>(deck.getCards());
        List<Card> stacked = new ArrayList<>();
        appendRequestedCards(remaining, stacked, handNames);
        appendRequestedCards(remaining, stacked, libraryNames);
        if (stacked.isEmpty()) {
            return;
        }
        deck.getCards().clear();
        for (Card card : stacked) {
            deck.getCards().add(card);
        }
        for (Card card : remaining) {
            deck.getCards().add(card);
        }
    }

    private static void appendRequestedCards(List<Card> remaining, List<Card> stacked, List<String> requestedNames) {
        if (remaining == null || stacked == null || requestedNames == null || requestedNames.isEmpty()) {
            return;
        }
        for (String requestedName : requestedNames) {
            String wanted = normalizeName(requestedName);
            if (wanted.isEmpty()) {
                continue;
            }
            for (int i = 0; i < remaining.size(); i++) {
                Card card = remaining.get(i);
                if (card != null && normalizeName(card.getName()).equals(wanted)) {
                    stacked.add(card);
                    remaining.remove(i);
                    break;
                }
            }
        }
    }

    private static void forceLibraryOrder(Player player, Deck deck, Game game) {
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

    private static String normalizeName(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest((value == null ? "" : value).getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return out.toString();
        } catch (Exception e) {
            return Integer.toHexString(String.valueOf(value).hashCode());
        }
    }

    private static void startGameInGameThread(Game game, UUID startingPlayerId, int joinTimeoutSec) {
        if (ThreadUtils.isRunGameThread()) {
            game.start(startingPlayerId);
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                game.start(startingPlayerId);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-ACTION-CF");
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            try {
                game.end();
            } catch (Exception ignored) {
            }
            boolean restoreInterrupt = Thread.interrupted();
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ignored) {
                restoreInterrupt = true;
            }
            if (restoreInterrupt) {
                Thread.currentThread().interrupt();
            }
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
                throw new IllegalStateException("Game thread timeout; worker did not stop after cleanup");
            }
            throw new IllegalStateException("Game thread timeout");
        }

        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        }
        throw new IllegalStateException("Error while running game", t);
    }

    private static void resumeGameInGameThread(Game game, int joinTimeoutSec) {
        if (ThreadUtils.isRunGameThread()) {
            game.resume();
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                game.resume();
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-ACTION-CF-CHECKPOINT");
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            try {
                game.end();
            } catch (Exception ignored) {
            }
            boolean restoreInterrupt = Thread.interrupted();
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ignored) {
                restoreInterrupt = true;
            }
            if (restoreInterrupt) {
                Thread.currentThread().interrupt();
            }
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for checkpoint game thread", ie);
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
                throw new IllegalStateException("Checkpoint game thread timeout; worker did not stop after cleanup");
            }
            throw new IllegalStateException("Checkpoint game thread timeout");
        }

        Throwable t = error.get();
        if (t == null) {
            return;
        }
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        }
        if (t instanceof Error) {
            throw (Error) t;
        }
        throw new IllegalStateException("Error while resuming checkpoint game", t);
    }

    private static void cleanup(Game game, TwoPlayerMatch match) {
        try {
            if (game != null) {
                game.end();
                game.cleanUp();
            }
        } catch (Exception ignored) {
        }
        try {
            if (match != null) {
                match.getGames().clear();
            }
        } catch (Exception ignored) {
        }
    }

    private static int safeTurn(Game game) {
        try {
            return game.getTurnNum();
        } catch (Exception ignored) {
            return -1;
        }
    }

    private static List<Path> loadDeckListOrSingle(String deckListPath) throws IOException {
        Path p = Paths.get(deckListPath).toAbsolutePath().normalize();
        if (Files.isRegularFile(p) && !p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".txt")) {
            return Collections.singletonList(p);
        }
        return loadDeckList(deckListPath);
    }

    private static List<Path> loadDeckList(String deckListPath) throws IOException {
        Path list = Paths.get(deckListPath).toAbsolutePath().normalize();
        Path base = list.getParent();
        List<Path> decks = new ArrayList<>();
        for (String raw : Files.readAllLines(list, StandardCharsets.UTF_8)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Path p = Paths.get(line);
            if (!p.isAbsolute()) {
                p = base.resolve(p);
            }
            p = p.toAbsolutePath().normalize();
            if (Files.isRegularFile(p)) {
                decks.add(p);
            } else {
                Path cwdRelative = Paths.get(line).toAbsolutePath().normalize();
                if (Files.isRegularFile(cwdRelative)) {
                    decks.add(cwdRelative);
                } else {
                    throw new IllegalArgumentException("Deck path not found: " + line + " from " + deckListPath);
                }
            }
        }
        return decks;
    }

    private static List<ReplayExpectation> loadReplayExpectations(
            Path csvFile,
            List<Path> agentDecks,
            List<Path> oppDecks
    ) throws IOException {
        Map<String, Path> agentByName = deckMap(agentDecks);
        Map<String, Path> oppByName = deckMap(oppDecks);
        List<String> lines = Files.readAllLines(csvFile, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> header = parseCsvLine(lines.get(0));
        Map<String, Integer> columns = new HashMap<>();
        for (int i = 0; i < header.size(); i++) {
            columns.put(header.get(i), i);
        }
        List<ReplayExpectation> out = new ArrayList<>();
        for (int lineNo = 1; lineNo < lines.size(); lineNo++) {
            String raw = lines.get(lineNo);
            if (raw.trim().isEmpty()) {
                continue;
            }
            List<String> cols = parseCsvLine(raw);
            String agentName = csvValue(cols, columns, "agent_deck");
            String oppName = csvValue(cols, columns, "opp_deck");
            Path agentDeck = agentByName.get(agentName);
            Path oppDeck = oppByName.get(oppName);
            if (agentDeck == null || oppDeck == null) {
                throw new IllegalArgumentException("Replay row references deck not in configured deck lists at line "
                        + (lineNo + 1) + ": " + agentName + " vs " + oppName);
            }
            StateSequenceBuilder.ActionType actionType = StateSequenceBuilder.ActionType.valueOf(
                    csvValue(cols, columns, "action_type"));
            List<Integer> expectedIndices = parseIntList(csvValue(cols, columns, "chosen_indices"));
            if (expectedIndices.isEmpty()) {
                expectedIndices = Collections.singletonList(Integer.parseInt(csvValue(cols, columns, "best_idx")));
            }
            List<String> expectedTexts = parseTextList(csvValue(cols, columns, "chosen_texts"));
            if (expectedTexts.isEmpty()) {
                expectedTexts = Collections.singletonList(csvValue(cols, columns, "best_text"));
            }
            List<String> agentOpeningHand = parseTextList(csvValue(cols, columns, "first_priority_hand"));
            if (agentOpeningHand.isEmpty()) {
                agentOpeningHand = parseTextList(csvValue(cols, columns, "first_mulligan_hand"));
            }
            List<String> sourceHandNames = parseTextList(csvValue(cols, columns, "source_hand"));
            if (agentOpeningHand.isEmpty()) {
                agentOpeningHand = sourceHandNames;
            }
            List<String> agentOpeningLibrary = parseTextList(csvValue(cols, columns, "source_library"));
            List<String> sourceLibraryTopNames = parseTextList(csvValue(cols, columns, "source_library_top"));
            List<String> sourceCandidateTexts = parseTextList(csvValue(cols, columns, "source_candidate_texts"));
            List<String> sourceSelectedObjectIds = parseTextList(csvValue(cols, columns, "source_selected_object_ids"));
            List<String> sourceCandidateObjectIds = parseTextList(csvValue(cols, columns, "source_candidate_object_ids"));
            String sourceIdentityStatus = csvValue(cols, columns, "source_identity_status");
            boolean replayTarget = replayTargetValue(csvValue(cols, columns, "target_marker"));
            String sourceDecisionNumber = csvValue(cols, columns, "source_decision_number");
            String sourceAnchorId = csvValue(cols, columns, "source_anchor_id");
            String sourceTurn = csvValue(cols, columns, "turn");
            String sourcePhase = csvValue(cols, columns, "phase");
            String sourceActor = csvValue(cols, columns, "actor");
            String sourceSelectedText = csvValue(cols, columns, "source_selected_text");
            long sourceRandomUtilCountBeforeSearch = parseOptionalLong(
                    csvValue(cols, columns, "source_random_util_count_before_search"), -1L);
            int sourceStackCount = parseOptionalInt(csvValue(cols, columns, "source_stack_count"), -1);
            String sourceStackTop = csvValue(cols, columns, "source_stack_top");
            out.add(new ReplayExpectation(
                    Integer.parseInt(csvValue(cols, columns, "scenario")),
                    agentDeck,
                    oppDeck,
                    Long.parseLong(csvValue(cols, columns, "seed")),
                    Integer.parseInt(csvValue(cols, columns, "ordinal")),
                    actionType,
                    expectedIndices,
                    expectedTexts,
                    agentOpeningHand,
                    agentOpeningLibrary,
                    Collections.emptyList(),
                    replayTarget,
                    sourceDecisionNumber,
                    sourceAnchorId,
                    sourceTurn,
                    sourcePhase,
                    sourceActor,
                    sourceSelectedText,
                    sourceHandNames,
                    sourceLibraryTopNames,
                    sourceCandidateTexts,
                    sourceSelectedObjectIds,
                    sourceCandidateObjectIds,
                    sourceIdentityStatus,
                    sourceRandomUtilCountBeforeSearch,
                    sourceStackCount,
                    sourceStackTop
            ));
        }
        return out;
    }

    private static Map<String, Path> deckMap(List<Path> decks) {
        Map<String, Path> out = new HashMap<>();
        for (Path deck : decks) {
            out.put(deck.getFileName().toString(), deck);
        }
        return out;
    }

    private static String csvValue(List<String> row, Map<String, Integer> columns, String name) {
        Integer idx = columns.get(name);
        if (idx == null || idx < 0 || idx >= row.size()) {
            return "";
        }
        return row.get(idx);
    }

    private static long parseOptionalLong(String value, long fallback) {
        if (value == null || value.trim().isEmpty()) {
            return fallback;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static int parseOptionalInt(String value, int fallback) {
        if (value == null || value.trim().isEmpty()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static List<String> parseCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        out.add(cur.toString());
        return out;
    }

    private static List<String> parseTextList(String value) {
        if (value == null || value.trim().isEmpty()) {
            return Collections.emptyList();
        }
        return Arrays.stream(value.split("\\|\\|", -1))
                .map(String::trim)
                .collect(Collectors.toList());
    }

    private static boolean replayTargetValue(String value) {
        if (value == null || value.trim().isEmpty()) {
            return true;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return !(normalized.equals("prefix")
                || normalized.equals("prefix_only")
                || normalized.equals("false")
                || normalized.equals("0")
                || normalized.equals("no"));
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        return values.stream().map(String::valueOf).collect(Collectors.joining(";"));
    }

    private static String joinTexts(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        return values.stream()
                .map(v -> v == null ? "" : v.replace("||", " "))
                .collect(Collectors.joining("||"));
    }

    private static String indexedTexts(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            out.add(i + ":" + (values.get(i) == null ? "" : values.get(i)));
        }
        return joinTexts(out);
    }

    private static List<Integer> candidateIndices(List<String> values) {
        if (values == null || values.isEmpty()) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            out.add(i);
        }
        return out;
    }

    private static List<String> actualSelectedTexts(List<Integer> indices, List<String> actualCandidateTexts) {
        if (indices == null || indices.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Integer idx : indices) {
            if (idx != null && actualCandidateTexts != null && idx >= 0 && idx < actualCandidateTexts.size()) {
                out.add(actualCandidateTexts.get(idx));
            } else {
                out.add("");
            }
        }
        return out;
    }

    private static void writeDecisionRecords(Path path, List<DecisionRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<DecisionRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((DecisionRecord r) -> r.scenario)
                .thenComparingInt(r -> r.ordinal)
                .thenComparingInt(r -> r.forcedIdx));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,action_type,candidate_count,baseline_idx,forced_idx,forced_text,policy_prob,won,timed_out,forced_applied,turns,error,trajectory_raw_records,trajectory_kept_records,trajectory_drop_reason\n");
            for (DecisionRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeBranchValueProbeRecords(Path path, List<BranchValueProbeRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<BranchValueProbeRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((BranchValueProbeRecord r) -> r.scenario)
                .thenComparingInt(r -> r.ordinal)
                .thenComparingInt(r -> r.forcedIdx));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,action_type,candidate_count,baseline_idx,forced_idx,forced_text,policy_prob,terminal_won,terminal_timed_out,terminal_forced_applied,value_captured,value_terminal,value_score,value_ordinal,value_timed_out,value_forced_applied,state,error\n");
            for (BranchValueProbeRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeBranchValueProbeReadme(Path path, List<BranchValueProbeRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        int captured = 0;
        int terminal = 0;
        int winning = 0;
        int losing = 0;
        double winningValue = 0.0;
        double losingValue = 0.0;
        Map<String, List<BranchValueProbeRecord>> byDecision = new LinkedHashMap<>();
        for (BranchValueProbeRecord r : records) {
            String decisionKey = r.scenario + ":" + r.ordinal;
            byDecision.computeIfAbsent(decisionKey, k -> new ArrayList<>()).add(r);
            if (r.valueCaptured) {
                captured++;
            }
            if (r.valueTerminal) {
                terminal++;
            }
            if (Float.isFinite(r.valueScore)) {
                if (r.terminalWon) {
                    winning++;
                    winningValue += r.valueScore;
                } else {
                    losing++;
                    losingValue += r.valueScore;
                }
            }
        }
        int pairedDecisions = 0;
        int pairedWinnerPreferred = 0;
        int pairedWinnerTied = 0;
        int pairedCaptured = 0;
        for (List<BranchValueProbeRecord> decisionRecords : byDecision.values()) {
            BranchValueProbeRecord bestWin = null;
            BranchValueProbeRecord bestLoss = null;
            for (BranchValueProbeRecord r : decisionRecords) {
                if (!Float.isFinite(r.valueScore)) {
                    continue;
                }
                if (r.terminalWon) {
                    if (bestWin == null || r.valueScore > bestWin.valueScore) {
                        bestWin = r;
                    }
                } else {
                    if (bestLoss == null || r.valueScore > bestLoss.valueScore) {
                        bestLoss = r;
                    }
                }
            }
            if (bestWin == null || bestLoss == null) {
                continue;
            }
            pairedDecisions++;
            if (bestWin.valueCaptured && bestLoss.valueCaptured) {
                pairedCaptured++;
            }
            if (bestWin.valueScore > bestLoss.valueScore) {
                pairedWinnerPreferred++;
            } else if (bestWin.valueScore == bestLoss.valueScore) {
                pairedWinnerTied++;
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# Branch Value Probe\n\n");
        sb.append("records: ").append(records.size()).append('\n');
        sb.append("captured_next_state: ").append(captured).append('\n');
        sb.append("terminal_before_next_state: ").append(terminal).append('\n');
        sb.append("winning_value_mean: ")
                .append(String.format(Locale.US, "%.6f", winning == 0 ? 0.0 : winningValue / winning)).append('\n');
        sb.append("losing_value_mean: ")
                .append(String.format(Locale.US, "%.6f", losing == 0 ? 0.0 : losingValue / losing)).append('\n');
        sb.append("paired_win_loss_decisions: ").append(pairedDecisions).append('\n');
        sb.append("paired_captured_decisions: ").append(pairedCaptured).append('\n');
        sb.append("paired_value_prefers_winner: ").append(pairedWinnerPreferred).append('\n');
        sb.append("paired_value_ties_winner: ").append(pairedWinnerTied).append('\n');
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static void writeReplayRecords(Path path, List<ReplayRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<ReplayRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((ReplayRecord r) -> r.scenario)
                .thenComparingInt(r -> r.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,source_decision_number,source_anchor_id,source_turn,source_phase,source_actor,action_type,actual_action_type,expected_idx,expected_text,actual_idx,actual_text,expected_indices,actual_indices,expected_texts,actual_texts,expected_prob,actual_prob,index_match,text_match,matched,replay_won,timed_out,turns,actual_decisions,forced_prefix_count,prefix_failure_ordinal,prefix_failure_reason,prefix_failure_detail,prefix_expected_action_type,prefix_actual_action_type,prefix_expected_indices,prefix_expected_texts,prefix_actual_candidate_indices,prefix_actual_candidate_texts,prefix_actual_selected_indices,prefix_actual_selected_texts,prefix_failure_forced_count,prefix_source_decision_number,prefix_source_anchor_id,prefix_source_turn,prefix_source_phase,prefix_source_actor,prefix_state,error\n");
            for (ReplayRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeCheckpointBranchRecords(Path path, List<CheckpointBranchRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<CheckpointBranchRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((CheckpointBranchRecord r) -> r.job == null ? -1 : r.job.scenario)
                .thenComparingInt(r -> r.expected == null ? -1 : r.expected.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,source_decision_number,source_anchor_id,expected_action_type,"
                    + "checkpoint_captured,captured_checkpoint_count,source_actual_decisions,source_run_timed_out,source_run_error,"
                    + "source_prefix_failure_ordinal,source_prefix_failure_reason,source_prefix_failure_detail,"
                    + "source_prefix_expected_action_type,source_prefix_actual_action_type,"
                    + "source_prefix_expected_indices,source_prefix_expected_texts,"
                    + "source_prefix_actual_candidate_indices,source_prefix_actual_candidate_texts,"
                    + "source_prefix_actual_selected_indices,source_prefix_actual_selected_texts,"
                    + "source_prefix_failure_forced_count,source_prefix_source_decision_number,"
                    + "source_prefix_source_anchor_id,source_prefix_source_turn,source_prefix_source_phase,"
                    + "source_prefix_source_actor,source_prefix_state,"
                    + "source_choice_matches_expected,source_choice_reentry_matched,classification,checkpoint_action_type,"
                    + "checkpoint_candidate_count,source_indices,source_texts,checkpoint_candidates,candidate_hash,state_hash,rng_state_hash,"
                    + "source_choice_a_candidate_hash,source_choice_b_candidate_hash,"
                    + "source_terminal,source_won,source_lost,source_terminal_value,source_turns,source_error,source_final_state,"
                    + "alternate_index,alternate_text,"
                    + "alternate_terminal,alternate_won,alternate_lost,alternate_terminal_value,alternate_turns,alternate_error,alternate_final_state\n");
            for (CheckpointBranchRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeCapturedCheckpointManifest(
            Path path,
            List<EngineDecisionCheckpoint> checkpoints
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<EngineDecisionCheckpoint> ordered = checkpoints == null
                ? Collections.emptyList()
                : new ArrayList<>(checkpoints);
        ordered.sort(Comparator.comparingInt((EngineDecisionCheckpoint c) -> c.job == null ? -1 : c.job.scenario)
                .thenComparingInt(c -> c.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,checkpoint_ordinal,action_type,candidate_count,"
                    + "candidate_hash,state_hash,rng_state_hash,candidates\n");
            for (EngineDecisionCheckpoint checkpoint : ordered) {
                if (checkpoint == null) {
                    continue;
                }
                out.write((checkpoint.job == null ? -1 : checkpoint.job.scenario)
                        + "," + csv(checkpoint.job == null ? "" : checkpoint.job.agentDeck.getFileName().toString())
                        + "," + csv(checkpoint.job == null ? "" : checkpoint.job.oppDeck.getFileName().toString())
                        + "," + (checkpoint.job == null ? -1L : checkpoint.job.seed)
                        + "," + checkpoint.ordinal
                        + "," + csv(checkpoint.actionType == null ? "" : checkpoint.actionType.name())
                        + "," + checkpoint.candidateTexts.size()
                        + "," + csv(checkpoint.candidateHash)
                        + "," + csv(checkpoint.stateHash)
                        + "," + csv(checkpoint.randomStateHash)
                        + "," + csv(indexedTexts(checkpoint.candidateTexts))
                        + "\n");
            }
        }
    }

    private static void writeCheckpointBranchReadme(
            Path path,
            Args args,
            List<CheckpointBranchRecord> records,
            int completedGroups,
            int totalGroups,
            long elapsedMs
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        long captured = records.stream().filter(r -> r.checkpointCaptured).count();
        long reentryMatched = records.stream().filter(r -> r.sourceChoiceReentryMatched).count();
        long correctionCandidates = records.stream()
                .filter(r -> "correction_candidate".equals(r.classification))
                .count();
        long cleanNegatives = records.stream()
                .filter(r -> "clean_negative".equals(r.classification))
                .count();
        StringBuilder sb = new StringBuilder();
        sb.append("# Checkpoint Branch Probe\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("replay_file: ").append(args.replayFile).append('\n');
        sb.append("checkpoint_branch_probe: ").append(args.checkpointBranchProbe).append('\n');
        sb.append("checkpoint_capture_uses_forced_prefix: true\n");
        sb.append("checkpoint_continuations_replay_prefix: false\n");
        sb.append("random_util_seeded_from_replay_seed: ").append(args.replayFile != null).append('\n');
        sb.append("completed_groups: ").append(completedGroups).append('/').append(totalGroups).append('\n');
        sb.append("records: ").append(records.size()).append('\n');
        sb.append("checkpoints_captured: ").append(captured).append('\n');
        sb.append("source_choice_reentry_matched: ").append(reentryMatched).append('\n');
        sb.append("correction_candidates: ").append(correctionCandidates).append('\n');
        sb.append("clean_negatives: ").append(cleanNegatives).append('\n');
        sb.append("elapsed_ms: ").append(elapsedMs).append('\n');
        sb.append("artifact: checkpoint_branch_probe.csv\n");
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static final Object POLICY_INPUT_DUMP_LOCK = new Object();
    private static final String POLICY_INPUT_DUMP_HEADER =
            "ordinal,action_type,candidate_count,chosen_indices,state_tensor_hash,state_mask_hash,state_token_ids_hash,"
                    + "state_shape,candidate_vector_hash,policy_vector,battlefield_order,opponent_battlefield_order,"
                    + "hand,library_top,state_snapshot_hash,state_snapshot,candidate_index,candidate_mask,"
                    + "candidate_action_id,candidate_feature_hash,candidate_features,policy_score,chosen,candidate_text\n";

    private static void initializePolicyInputDump(Path path) throws IOException {
        if (path == null) {
            return;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        synchronized (POLICY_INPUT_DUMP_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE)) {
                out.write(POLICY_INPUT_DUMP_HEADER);
                out.flush();
            }
        }
    }

    private static void appendPolicyInputDump(
            Args args,
            int ordinal,
            StateSequenceBuilder.TrainingData td,
            float[] policyScores,
            List<String> candidateTexts,
            StateSnapshot snapshot
    ) {
        if (args == null || !args.policyInputDump || args.policyInputDumpPath == null || td == null) {
            return;
        }
        StateSequenceBuilder.SequenceOutput state = td.state;
        String stateTensorHash = hashFloats2d(state == null ? null : state.getSequence());
        String stateMaskHash = hashInts(state == null ? null : state.getMask());
        String stateTokenIdsHash = hashInts(state == null ? null : state.getTokenIds());
        String stateShape = stateShape(state);
        String candidateVectorHash = hashCandidateVector(td);
        String policyVector = indexedPolicyScores(policyScores, td.candidateMask, td.candidateCount);
        String chosenIndices = chosenIndicesText(td);
        String stateText = snapshot == null ? "" : snapshot.toCompactText();
        String battlefieldOrder = snapshotField(stateText, "battlefieldDetail");
        String opponentBattlefieldOrder = snapshotField(stateText, "opponentBattlefield");
        String hand = snapshotField(stateText, "hand");
        String libraryTop = snapshotField(stateText, "libraryTop");
        String stateSnapshotHash = hexHashString(stateText);
        int limit = Math.min(Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES),
                td.candidateMask == null ? 0 : td.candidateMask.length);
        List<String> rows = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            int mask = td.candidateMask == null || i >= td.candidateMask.length ? 0 : td.candidateMask[i];
            int actionId = td.candidateActionIds != null && i < td.candidateActionIds.length
                    ? td.candidateActionIds[i] : 0;
            float[] features = td.candidateFeatures != null && i < td.candidateFeatures.length
                    ? td.candidateFeatures[i] : null;
            float policy = policyScores != null && i < policyScores.length ? policyScores[i] : 0.0f;
            String text = candidateTexts != null && i < candidateTexts.size() ? candidateTexts.get(i) : "";
            rows.add(ordinal
                    + "," + csv(td.actionType == null ? "" : td.actionType.name())
                    + "," + td.candidateCount
                    + "," + csv(chosenIndices)
                    + "," + csv(stateTensorHash)
                    + "," + csv(stateMaskHash)
                    + "," + csv(stateTokenIdsHash)
                    + "," + csv(stateShape)
                    + "," + csv(candidateVectorHash)
                    + "," + csv(policyVector)
                    + "," + csv(battlefieldOrder)
                    + "," + csv(opponentBattlefieldOrder)
                    + "," + csv(hand)
                    + "," + csv(libraryTop)
                    + "," + csv(stateSnapshotHash)
                    + "," + csv(stateText)
                    + "," + i
                    + "," + mask
                    + "," + actionId
                    + "," + csv(hashFloats(features))
                    + "," + csv(floatArrayText(features))
                    + "," + csv(formatFloat(policy))
                    + "," + (isChosenIndex(td, i) ? "1" : "0")
                    + "," + csv(text));
        }
        if (rows.isEmpty()) {
            return;
        }
        synchronized (POLICY_INPUT_DUMP_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(args.policyInputDumpPath, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE)) {
                for (String row : rows) {
                    out.write(row);
                    out.write('\n');
                }
                out.flush();
            } catch (IOException ignored) {
                // Validation diagnostics must not alter replay behavior.
            }
        }
    }

    private static final Object POLICY_INFERENCE_PROBE_LOCK = new Object();
    private static final String POLICY_INFERENCE_PROBE_HEADER =
            "ordinal,action_type,candidate_count,chosen_indices,state_tensor_hash,state_mask_hash,state_token_ids_hash,"
                    + "state_shape,candidate_vector_hash,state_snapshot_hash,policy_key,head_id,pick_index,min_targets,"
                    + "max_targets,model_class,model_device,backend_path,request_id,batch_id,batch_index,batch_size,"
                    + "caller_thread,caller_thread_id,backend_thread,raw_vector_kind,raw_policy_vector,"
                    + "normalized_policy_vector,policy_sum,top_index,top_score,value_score,score_fallback,score_error,"
                    + "backend_details,py_service_mode,py_backend_mode,shared_gpu_endpoint,gpu_service_num_channels,"
                    + "py_batch_max_size,py_batch_timeout_ms,policy_order_key,candidate_row_order,candidate_text_order\n";

    private static void initializePolicyInferenceProbe(Path path) throws IOException {
        if (path == null) {
            return;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        synchronized (POLICY_INFERENCE_PROBE_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE)) {
                out.write(POLICY_INFERENCE_PROBE_HEADER);
                out.flush();
            }
        }
    }

    private static void appendPolicyInferenceProbe(
            Args args,
            int ordinal,
            StateSequenceBuilder.TrainingData td,
            PolicyScoreResult score,
            List<String> candidateTexts,
            StateSnapshot snapshot
    ) {
        if (args == null || !args.policyInferenceProbe || args.policyInferenceProbePath == null
                || td == null || score == null) {
            return;
        }
        StateSequenceBuilder.SequenceOutput state = td.state;
        String stateTensorHash = hashFloats2d(state == null ? null : state.getSequence());
        String stateMaskHash = hashInts(state == null ? null : state.getMask());
        String stateTokenIdsHash = hashInts(state == null ? null : state.getTokenIds());
        String stateShape = stateShape(state);
        String candidateVectorHash = hashCandidateVector(td);
        String chosenIndices = chosenIndicesText(td);
        String stateText = snapshot == null ? "" : snapshot.toCompactText();
        String stateSnapshotHash = hexHashString(stateText);
        float policySum = validPolicySum(score.rawScores, td.candidateMask, td.candidateCount);
        int topIndex = topPolicyIndex(score.rawScores, td.candidateMask, td.candidateCount);
        float topScore = topIndex >= 0 && topIndex < score.rawScores.length ? score.rawScores[topIndex] : 0.0f;
        String row = ordinal
                + "," + csv(td.actionType == null ? "" : td.actionType.name())
                + "," + td.candidateCount
                + "," + csv(chosenIndices)
                + "," + csv(stateTensorHash)
                + "," + csv(stateMaskHash)
                + "," + csv(stateTokenIdsHash)
                + "," + csv(stateShape)
                + "," + csv(candidateVectorHash)
                + "," + csv(stateSnapshotHash)
                + "," + csv(score.policyKey)
                + "," + csv(score.headId)
                + "," + score.pickIndex
                + "," + score.minTargets
                + "," + score.maxTargets
                + "," + csv(score.modelClass)
                + "," + csv(score.deviceInfo)
                + "," + csv(score.backendPath)
                + "," + csv(score.requestId)
                + "," + csv(score.batchId)
                + "," + score.batchIndex
                + "," + score.batchSize
                + "," + csv(score.callerThreadName)
                + "," + score.callerThreadId
                + "," + csv(score.backendThreadName)
                + "," + csv(score.rawScoreKind)
                + "," + csv(indexedPolicyScores(score.rawScores, td.candidateMask, td.candidateCount))
                + "," + csv(indexedNormalizedPolicyScores(score.rawScores, td.candidateMask, td.candidateCount))
                + "," + csv(formatFloat(policySum))
                + "," + topIndex
                + "," + csv(formatFloat(topScore))
                + "," + csv(formatFloat(score.valueScore))
                + "," + (score.fallback ? "1" : "0")
                + "," + csv(score.error)
                + "," + csv(score.backendDetails)
                + "," + csv(envText("PY_SERVICE_MODE"))
                + "," + csv(envText("PY_BACKEND_MODE"))
                + "," + csv(envText("SHARED_GPU_ENDPOINT"))
                + "," + csv(envText("GPU_SERVICE_NUM_CHANNELS"))
                + "," + csv(envText("PY_BATCH_MAX_SIZE"))
                + "," + csv(envText("PY_BATCH_TIMEOUT_MS"))
                + "," + csv(score.policyKey + "|" + score.headId + "|" + score.pickIndex + "|"
                + score.minTargets + "|" + score.maxTargets)
                + "," + csv(candidateRowOrder(td, candidateTexts))
                + "," + csv(candidateTextOrder(candidateTexts, td.candidateCount));
        synchronized (POLICY_INFERENCE_PROBE_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(args.policyInferenceProbePath, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE)) {
                out.write(row);
                out.write('\n');
                out.flush();
            } catch (IOException ignored) {
                // Validation diagnostics must not alter replay behavior.
            }
        }
    }

    private static String stateShape(StateSequenceBuilder.SequenceOutput state) {
        if (state == null || state.getSequence() == null) {
            return "0x0";
        }
        float[][] seq = state.getSequence();
        int dim = seq.length == 0 || seq[0] == null ? 0 : seq[0].length;
        return seq.length + "x" + dim;
    }

    private static String chosenIndicesText(StateSequenceBuilder.TrainingData td) {
        if (td == null || td.chosenIndices == null || td.chosenCount <= 0) {
            return "";
        }
        List<Integer> chosen = new ArrayList<>();
        for (int i = 0; i < Math.min(td.chosenCount, td.chosenIndices.length); i++) {
            int idx = td.chosenIndices[i];
            if (idx >= 0) {
                chosen.add(idx);
            }
        }
        return joinInts(chosen);
    }

    private static boolean isChosenIndex(StateSequenceBuilder.TrainingData td, int candidateIndex) {
        if (td == null || td.chosenIndices == null || td.chosenCount <= 0) {
            return false;
        }
        for (int i = 0; i < Math.min(td.chosenCount, td.chosenIndices.length); i++) {
            if (td.chosenIndices[i] == candidateIndex) {
                return true;
            }
        }
        return false;
    }

    private static String indexedPolicyScores(float[] policyScores, int[] candidateMask, int candidateCount) {
        if (policyScores == null || candidateCount <= 0) {
            return "";
        }
        int limit = Math.min(candidateCount, policyScores.length);
        List<String> values = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            if (candidateMask != null && (i >= candidateMask.length || candidateMask[i] == 0)) {
                continue;
            }
            values.add(i + ":" + formatFloat(policyScores[i]));
        }
        return String.join(";", values);
    }

    private static String indexedNormalizedPolicyScores(float[] policyScores, int[] candidateMask, int candidateCount) {
        if (policyScores == null || candidateCount <= 0) {
            return "";
        }
        float sum = validPolicySum(policyScores, candidateMask, candidateCount);
        int limit = Math.min(candidateCount, policyScores.length);
        List<String> values = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            if (candidateMask != null && (i >= candidateMask.length || candidateMask[i] == 0)) {
                continue;
            }
            float raw = policyScores[i];
            float normalized = sum == 0.0f ? 0.0f : raw / sum;
            values.add(i + ":" + formatFloat(normalized));
        }
        return String.join(";", values);
    }

    private static float validPolicySum(float[] policyScores, int[] candidateMask, int candidateCount) {
        if (policyScores == null || candidateCount <= 0) {
            return 0.0f;
        }
        int limit = Math.min(candidateCount, policyScores.length);
        float sum = 0.0f;
        for (int i = 0; i < limit; i++) {
            if (candidateMask != null && (i >= candidateMask.length || candidateMask[i] == 0)) {
                continue;
            }
            float value = policyScores[i];
            if (!Float.isNaN(value) && !Float.isInfinite(value)) {
                sum += value;
            }
        }
        return sum;
    }

    private static int topPolicyIndex(float[] policyScores, int[] candidateMask, int candidateCount) {
        if (policyScores == null || candidateCount <= 0) {
            return -1;
        }
        int limit = Math.min(candidateCount, policyScores.length);
        int best = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < limit; i++) {
            if (candidateMask != null && (i >= candidateMask.length || candidateMask[i] == 0)) {
                continue;
            }
            float value = policyScores[i];
            if (Float.isNaN(value)) {
                continue;
            }
            if (best < 0 || value > bestScore) {
                best = i;
                bestScore = value;
            }
        }
        return best;
    }

    private static String candidateRowOrder(StateSequenceBuilder.TrainingData td, List<String> candidateTexts) {
        if (td == null || td.candidateCount <= 0) {
            return "";
        }
        int limit = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        List<String> values = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            int mask = td.candidateMask != null && i < td.candidateMask.length ? td.candidateMask[i] : 0;
            int actionId = td.candidateActionIds != null && i < td.candidateActionIds.length
                    ? td.candidateActionIds[i] : 0;
            String text = candidateTexts != null && i < candidateTexts.size() ? candidateTexts.get(i) : "";
            values.add(i + ":mask=" + mask + ":id=" + actionId + ":text=" + text);
        }
        return String.join(" | ", values);
    }

    private static String candidateTextOrder(List<String> candidateTexts, int candidateCount) {
        if (candidateTexts == null || candidateTexts.isEmpty() || candidateCount <= 0) {
            return "";
        }
        int limit = Math.min(candidateCount, candidateTexts.size());
        List<String> values = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            values.add(i + ":" + candidateTexts.get(i));
        }
        return String.join(" | ", values);
    }

    private static String envText(String key) {
        return System.getenv().getOrDefault(key, "");
    }

    private static String snapshotField(String snapshot, String field) {
        if (snapshot == null || snapshot.isEmpty() || field == null || field.isEmpty()) {
            return "";
        }
        String prefix = field + "=";
        int start = snapshot.indexOf(prefix);
        if (start < 0) {
            return "";
        }
        start += prefix.length();
        int end = snapshot.indexOf(';', start);
        return end < 0 ? snapshot.substring(start) : snapshot.substring(start, end);
    }

    private static String hashCandidateVector(StateSequenceBuilder.TrainingData td) {
        if (td == null) {
            return "0";
        }
        long hash = 1469598103934665603L;
        hash = mixInt(hash, td.candidateCount);
        int limit = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        for (int i = 0; i < limit; i++) {
            int mask = td.candidateMask != null && i < td.candidateMask.length ? td.candidateMask[i] : 0;
            hash = mixInt(hash, mask);
            if (mask == 0) {
                continue;
            }
            hash = mixInt(hash, td.candidateActionIds != null && i < td.candidateActionIds.length
                    ? td.candidateActionIds[i] : 0);
            hash = mixFloats(hash, td.candidateFeatures != null && i < td.candidateFeatures.length
                    ? td.candidateFeatures[i] : null);
        }
        return Long.toHexString(hash);
    }

    private static String hashFloats2d(float[][] values) {
        long hash = 1469598103934665603L;
        if (values != null) {
            hash = mixInt(hash, values.length);
            for (float[] row : values) {
                hash = mixFloats(hash, row);
            }
        }
        return Long.toHexString(hash);
    }

    private static String hashFloats(float[] values) {
        return Long.toHexString(mixFloats(1469598103934665603L, values));
    }

    private static String hashInts(int[] values) {
        long hash = 1469598103934665603L;
        if (values != null) {
            hash = mixInt(hash, values.length);
            for (int value : values) {
                hash = mixInt(hash, value);
            }
        }
        return Long.toHexString(hash);
    }

    private static String hexHashString(String value) {
        long hash = 1469598103934665603L;
        String text = value == null ? "" : value;
        hash = mixInt(hash, text.length());
        for (int i = 0; i < text.length(); i++) {
            hash = mixInt(hash, text.charAt(i));
        }
        return Long.toHexString(hash);
    }

    private static long mixFloats(long hash, float[] values) {
        if (values == null) {
            return mixInt(hash, -1);
        }
        hash = mixInt(hash, values.length);
        for (float value : values) {
            hash = mixInt(hash, Float.floatToIntBits(value));
        }
        return hash;
    }

    private static long mixInt(long hash, int value) {
        long out = hash ^ (value & 0xffffffffL);
        return out * 1099511628211L;
    }

    private static String floatArrayText(float[] values) {
        if (values == null || values.length == 0) {
            return "";
        }
        List<String> out = new ArrayList<>(values.length);
        for (float value : values) {
            out.add(formatFloat(value));
        }
        return String.join(";", out);
    }

    private static String formatFloat(float value) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            return Float.toString(value);
        }
        return String.format(Locale.US, "%.9g", value);
    }

    private static final Object PREFIX_TRACE_FILE_LOCK = new Object();
    private static final String PREFIX_TRACE_HEADER = "scenario,agent_deck,opp_deck,seed,target_ordinal,target_source_decision_number,target_source_anchor_id,ordinal,source_decision_number,source_anchor_id,source_turn,source_phase,source_actor,expected_action_type,actual_action_type,expected_indices,expected_texts,actual_selected_indices,actual_selected_texts,actual_nonselected_texts,actual_candidate_indices,actual_candidate_texts,revealed_or_pickable_names,library_top_before,library_top_after,hand_before,hand_after,graveyard_before,graveyard_after,stack_before,stack_after,state\n";

    private static void initializePrefixTraceRecords(Path path) throws IOException {
        if (path == null) {
            return;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        synchronized (PREFIX_TRACE_FILE_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE)) {
                out.write(PREFIX_TRACE_HEADER);
                out.flush();
            }
        }
    }

    private static void writePrefixTraceRecords(Path path, List<PrefixTraceRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<PrefixTraceRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((PrefixTraceRecord r) -> r.scenario)
                .thenComparingInt(r -> r.targetOrdinal)
                .thenComparingInt(r -> r.ordinal));
        synchronized (PREFIX_TRACE_FILE_LOCK) {
            try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                out.write(PREFIX_TRACE_HEADER);
                for (PrefixTraceRecord r : ordered) {
                    out.write(r.toCsv());
                    out.write('\n');
                }
                out.flush();
            }
        }
    }

    private static void appendPrefixTraceLine(Path path, String csvLine) throws IOException {
        if (path == null || csvLine == null || csvLine.isEmpty()) {
            return;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        synchronized (PREFIX_TRACE_FILE_LOCK) {
            boolean needsHeader = !Files.exists(path) || Files.size(path) == 0L;
            try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND)) {
                if (needsHeader) {
                    out.write(PREFIX_TRACE_HEADER);
                }
                out.write(csvLine);
                out.write('\n');
                out.flush();
            }
        }
    }

    private static void appendLivePrefixTraceRecord(
            Path path,
            ScenarioJob job,
            ReplayExpectation target,
            ReplayExpectation expected,
            int ordinal,
            StateSequenceBuilder.ActionType actualActionType,
            List<Integer> actualSelectedIndices,
            List<String> actualCandidateTexts,
            StateSnapshot state
    ) throws IOException {
        if (path == null || job == null) {
            return;
        }
        List<Integer> safeActualSelectedIndices = actualSelectedIndices == null
                ? Collections.emptyList()
                : new ArrayList<>(actualSelectedIndices);
        List<String> safeActualCandidateTexts = actualCandidateTexts == null
                ? Collections.emptyList()
                : new ArrayList<>(actualCandidateTexts);
        String line = job.scenario
                + "," + csv(job.agentDeck.getFileName().toString())
                + "," + csv(job.oppDeck.getFileName().toString())
                + "," + job.seed
                + "," + (target == null ? -1 : target.ordinal)
                + "," + csv(target == null ? "" : target.sourceDecisionNumber)
                + "," + csv(target == null ? "" : target.sourceAnchorId)
                + "," + ordinal
                + "," + csv(expected == null ? "" : expected.sourceDecisionNumber)
                + "," + csv(expected == null ? "" : expected.sourceAnchorId)
                + "," + csv(expected == null ? "" : expected.sourceTurn)
                + "," + csv(expected == null ? "" : expected.sourcePhase)
                + "," + csv(expected == null ? "" : expected.sourceActor)
                + "," + (expected == null ? "" : expected.actionType)
                + "," + (actualActionType == null ? "" : actualActionType)
                + "," + csv(expected == null ? "" : joinInts(expected.expectedIndices))
                + "," + csv(expected == null ? "" : joinTexts(expected.expectedTexts))
                + "," + csv(joinInts(safeActualSelectedIndices))
                + "," + csv(joinTexts(actualSelectedTexts(safeActualSelectedIndices, safeActualCandidateTexts)))
                + "," + csv(nonselectedTexts(safeActualCandidateTexts, safeActualSelectedIndices))
                + "," + csv(joinInts(candidateIndices(safeActualCandidateTexts)))
                + "," + csv(indexedTexts(safeActualCandidateTexts))
                + "," + csv(actualActionType == StateSequenceBuilder.ActionType.SELECT_CARD
                ? revealedOrPickableNames(safeActualCandidateTexts) : "")
                + "," + csv(stateField(state, "libraryTop"))
                + "," + csv("")
                + "," + csv(stateField(state, "hand"))
                + "," + csv("")
                + "," + csv(stateField(state, "graveyard"))
                + "," + csv("")
                + "," + csv(stateField(state, "stack"))
                + "," + csv("")
                + "," + csv(state == null ? "" : state.toCompactText());
        appendPrefixTraceLine(path, line);
    }

    private static String stateField(StateSnapshot state, String key) {
        if (state == null || key == null || key.isEmpty()) {
            return "";
        }
        String prefix = key + "=";
        String text = state.toCompactText();
        if (text == null || text.isEmpty()) {
            return "";
        }
        for (String part : text.split(";")) {
            if (part.startsWith(prefix)) {
                return part.substring(prefix.length());
            }
        }
        return "";
    }

    private static String nonselectedTexts(List<String> candidateTexts, List<Integer> selectedIndices) {
        if (candidateTexts == null || candidateTexts.isEmpty()) {
            return "";
        }
        Set<Integer> selected = new HashSet<>();
        if (selectedIndices != null) {
            selected.addAll(selectedIndices);
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < candidateTexts.size(); i++) {
            if (selected.contains(i)) {
                continue;
            }
            String text = candidateTexts.get(i);
            if (PrefixTraceRecord.isStopCandidateText(text)) {
                continue;
            }
            out.add(text);
        }
        return joinTexts(out);
    }

    private static String revealedOrPickableNames(List<String> candidateTexts) {
        if (candidateTexts == null || candidateTexts.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (String text : candidateTexts) {
            if (!PrefixTraceRecord.isStopCandidateText(text)) {
                out.add(text);
            }
        }
        return joinTexts(out);
    }

    private static void writeReplayReadme(
            Path path,
            Args args,
            List<ReplayRecord> records,
            int deviationExampleCount,
            int daggerExampleCount,
            int completedGroups,
            int totalGroups,
            long elapsedMs
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        ReplayStats stats = ReplayStats.from(records);
        StringBuilder sb = new StringBuilder();
        sb.append("# Line Search Replay Probe\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("replay_file: ").append(args.replayFile).append('\n');
        sb.append("forced_prefix_replay: ").append(args.forcedPrefixReplay).append('\n');
        sb.append("force_opponent_transcript: ").append(args.forceOpponentTranscript).append('\n');
        if (args.opponentTranscriptFile != null) {
            sb.append("opponent_transcript_file: ").append(args.opponentTranscriptFile).append('\n');
        }
        if (args.opponentTranscriptMismatchPath != null) {
            sb.append("opponent_transcript_mismatch_file: ").append(args.opponentTranscriptMismatchPath).append('\n');
        }
        sb.append("random_util_seeded_from_replay_seed: ").append(args.replayFile != null).append('\n');
        sb.append("random_util_seed_salt: ").append(REPLAY_RANDOM_UTIL_SALT).append('\n');
        if (args.policyInputDumpPath != null) {
            sb.append("policy_input_dump_file: ").append(args.policyInputDumpPath).append('\n');
        }
        if (args.policyInferenceProbePath != null) {
            sb.append("policy_inference_probe_file: ").append(args.policyInferenceProbePath).append('\n');
        }
        sb.append("completed_groups: ").append(completedGroups).append('\n');
        sb.append("total_groups: ").append(totalGroups).append('\n');
        sb.append("records: ").append(stats.total).append('\n');
        sb.append("matched: ").append(stats.matched).append('\n');
        sb.append("accuracy: ").append(String.format(Locale.US, "%.4f", stats.accuracy())).append('\n');
        sb.append("index_matches: ").append(stats.indexMatches).append('\n');
        sb.append("text_matches: ").append(stats.textMatches).append('\n');
        sb.append("scenario_wins: ").append(stats.wonGroups).append('\n');
        sb.append("scenario_total: ").append(stats.totalGroups).append('\n');
        sb.append("deviation_training_examples: ").append(deviationExampleCount).append('\n');
        if (args.replayDeviationTrainingDataFile != null) {
            sb.append("deviation_training_data_file: ").append(args.replayDeviationTrainingDataFile).append('\n');
        }
        sb.append("dagger_training_examples: ").append(daggerExampleCount).append('\n');
        if (args.replayDaggerTrainingDataFile != null) {
            sb.append("dagger_training_data_file: ").append(args.replayDaggerTrainingDataFile).append('\n');
            sb.append("dagger_deviation_repeat: ").append(args.replayDeviationRepeat).append('\n');
        }
        sb.append("workers: ").append(args.workers).append('\n');
        sb.append("timeout_sec: ").append(args.timeoutSec).append('\n');
        sb.append("max_decision_depth: ").append(args.maxDecisionDepth).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeTensorReplayRecords(Path path, List<TensorReplayRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<TensorReplayRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((TensorReplayRecord r) -> r.scenario)
                .thenComparingInt(r -> r.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,action_type,candidate_count,target_idx,target_text,top_idx,top_text,target_prob,top_prob,target_rank,top1_match,valid_count\n");
            for (TensorReplayRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeTensorReplayReadme(Path path, List<TensorReplayRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        TensorReplayStats stats = TensorReplayStats.from(records);
        StringBuilder sb = new StringBuilder();
        sb.append("# Tensor Replay Probe\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("records: ").append(stats.total).append('\n');
        sb.append("top1_matches: ").append(stats.top1Matches).append('\n');
        sb.append("top1_accuracy: ").append(String.format(Locale.US, "%.4f", stats.accuracy())).append('\n');
        sb.append("mean_target_prob: ").append(String.format(Locale.US, "%.6f", stats.meanTargetProb())).append('\n');
        sb.append("mean_top_prob: ").append(String.format(Locale.US, "%.6f", stats.meanTopProb())).append('\n');
        sb.append("mean_target_rank: ").append(String.format(Locale.US, "%.3f", stats.meanTargetRank())).append('\n');
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeTrainingExamples(Path path, List<TrainingExample> examples) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<TrainingExample> ordered = new ArrayList<>(examples);
        ordered.sort(Comparator.comparingInt((TrainingExample e) -> e.scenario).thenComparingInt(e -> e.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,ordinal,action_type,candidate_count,baseline_idx,baseline_text,best_idx,best_text,chosen_indices,chosen_texts,branch_count,target_nonzero,target_observed,target_positive,target_negative,target_min,target_max\n");
            for (TrainingExample e : ordered) {
                out.write(e.toCsv());
                out.write('\n');
            }
        }
    }

    private static int countWinningTrajectories(List<WinningTrajectoryRecord> records) {
        if (records == null || records.isEmpty()) {
            return 0;
        }
        Set<String> keys = new HashSet<>();
        for (WinningTrajectoryRecord r : records) {
            if (r != null && !r.trajectoryKey.isEmpty()) {
                keys.add(r.trajectoryKey);
            }
        }
        return keys.size();
    }

    private static void writeWinningTrajectoryRecords(Path path, List<WinningTrajectoryRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<WinningTrajectoryRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparing((WinningTrajectoryRecord r) -> r.trajectoryKey)
                .thenComparingInt(r -> r.ordinal));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("trajectory_key,scenario,agent_deck,opp_deck,seed,ordinal,action_type,candidate_count,chosen_indices,chosen_texts,state,turns,first_mulligan_hand,first_priority_hand,final_state\n");
            for (WinningTrajectoryRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writePrefixRecords(Path path, List<PrefixRecord> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        List<PrefixRecord> ordered = new ArrayList<>(records);
        ordered.sort(Comparator.comparingInt((PrefixRecord r) -> r.scenario)
                .thenComparingInt(r -> r.nodeId));
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("scenario,agent_deck,opp_deck,seed,node_id,parent_id,prefix_len,forced_applied,won,timed_out,turns,first_mulligan_hand,first_priority_hand,next_ordinal,next_action_type,next_candidate_count,next_state,final_state,next_choices,branch_count,branch_order,expanded_text,prefix,error\n");
            for (PrefixRecord r : ordered) {
                out.write(r.toCsv());
                out.write('\n');
            }
        }
    }

    private static void writeReadme(
            Path path,
            Args args,
            List<Path> agentDecks,
            List<Path> oppDecks,
            int completedScenarios,
            int trainedScenarios,
            int skippedScenarios,
            int candidateExamples,
            int winningTrajectories,
            int trainPassSamples,
            long elapsedMs
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# Action Counterfactual Trainer\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("scenarios: ").append(args.scenarios).append('\n');
        sb.append("completed_scenarios: ").append(completedScenarios).append('\n');
        sb.append("collect_only: ").append(args.collectOnly).append('\n');
        if (args.policyInputDumpPath != null) {
            sb.append("policy_input_dump_file: ").append(args.policyInputDumpPath).append('\n');
        }
        if (args.policyInferenceProbePath != null) {
            sb.append("policy_inference_probe_file: ").append(args.policyInferenceProbePath).append('\n');
        }
        if (args.exportTrainingDataFile != null) {
            sb.append("export_training_data_file: ").append(args.exportTrainingDataFile).append('\n');
        }
        if (args.exportTrajectoryDataFile != null) {
            sb.append("export_trajectory_data_file: ").append(args.exportTrajectoryDataFile).append('\n');
            sb.append("trajectory_final_reward: ")
                    .append(String.format(Locale.US, "%.4f", args.trajectoryFinalReward)).append('\n');
        }
        if (!args.agentOpeningHandNames.isEmpty()) {
            sb.append("agent_opening_hand: ").append(args.agentOpeningHandNames).append('\n');
        }
        if (!args.agentOpeningHandPool.isEmpty() && args.agentOpeningHandPoolFile == null) {
            sb.append("agent_opening_hand_pool: ").append(args.agentOpeningHandPool).append('\n');
        }
        if (args.agentOpeningHandPoolFile != null) {
            sb.append("agent_opening_hand_pool_file: ").append(args.agentOpeningHandPoolFile).append('\n');
        }
        if (!args.oppOpeningHandNames.isEmpty()) {
            sb.append("opp_opening_hand: ").append(args.oppOpeningHandNames).append('\n');
        }
        if (!args.oppOpeningHandPool.isEmpty() && args.oppOpeningHandPoolFile == null) {
            sb.append("opp_opening_hand_pool: ").append(args.oppOpeningHandPool).append('\n');
        }
        if (args.oppOpeningHandPoolFile != null) {
            sb.append("opp_opening_hand_pool_file: ").append(args.oppOpeningHandPoolFile).append('\n');
        }
        sb.append("stop_after_examples: ").append(args.stopAfterExamples).append('\n');
        sb.append("stop_after_winning_trajectories: ").append(args.stopAfterWinningTrajectories).append('\n');
        sb.append("max_decision_depth: ").append(args.maxDecisionDepth).append('\n');
        sb.append("winning_prefix_mode: ").append(args.winningPrefixMode).append('\n');
        if (!args.initialPrefixChoices.isEmpty()) {
            sb.append("initial_prefix: ").append(prefixKey(args.initialPrefixChoices)).append('\n');
        }
        if (args.winningPrefixMode) {
            sb.append("max_prefix_depth: ").append(args.maxPrefixDepth).append('\n');
            sb.append("train_prefix_depth: ").append(args.trainPrefixDepth).append('\n');
            sb.append("max_search_nodes: ").append(args.maxSearchNodes).append('\n');
            sb.append("depth_first_search: ").append(args.depthFirstSearch).append('\n');
            sb.append("prefix_sibling_contrast: ").append(args.prefixSiblingContrast).append('\n');
            sb.append("prefix_sibling_contrast_search_nodes: ").append(args.prefixSiblingContrastSearchNodes).append('\n');
            sb.append("train_root_mulligan_on_no_win: ").append(args.trainRootMulliganOnNoWin).append('\n');
            sb.append("train_root_mulligan_only: ").append(args.trainRootMulliganOnly).append('\n');
            sb.append("generic_branch_order: ").append(args.genericBranchOrder).append('\n');
            sb.append("tactic_autopilot: ").append(args.tacticAutopilot).append('\n');
            sb.append("no_search_model_scoring: ").append(args.noSearchModelScoring).append('\n');
            sb.append("max_winning_prefixes_per_scenario: ").append(args.maxWinningPrefixesPerScenario).append('\n');
            if (!args.passMacroDepths.isEmpty()) {
                sb.append("pass_macro_depths: ").append(args.passMacroDepths).append('\n');
            }
            sb.append("skip_pass_training: ").append(args.skipPassTraining).append('\n');
            sb.append("skip_blank_training: ").append(args.skipBlankTraining).append('\n');
            sb.append("skip_mulligan_training: ").append(args.skipMulliganTraining).append('\n');
        }
        if (args.branchSubtreeSearchNodes > 0) {
            sb.append("branch_subtree_search_nodes: ").append(args.branchSubtreeSearchNodes).append('\n');
            sb.append("max_prefix_depth: ").append(args.maxPrefixDepth).append('\n');
        }
        sb.append("baseline_losing_alternative_only: ").append(args.baselineLosingAlternativeOnly).append('\n');
        if (!args.includeActionTextRegex.trim().isEmpty()) {
            sb.append("include_action_text_regex: ").append(args.includeActionTextRegex).append('\n');
        }
        if (!args.avoidLosingActionTextRegex.trim().isEmpty()) {
            sb.append("avoid_losing_action_text_regex: ").append(args.avoidLosingActionTextRegex).append('\n');
            sb.append("avoid_losing_strict_negative: ").append(args.avoidLosingStrictNegative).append('\n');
            sb.append("avoid_losing_mask_baseline_only: ").append(args.avoidLosingMaskBaselineOnly).append('\n');
        }
        sb.append("branch_return_targets: ").append(args.branchReturnTargets).append('\n');
        sb.append("branch_value_probe: ").append(args.branchValueProbe).append('\n');
        sb.append("branch_trajectory_mode: ").append(args.branchTrajectoryMode).append('\n');
        sb.append("branch_trajectory_first_post_target_only: ")
                .append(args.branchTrajectoryFirstPostTargetOnly).append('\n');
        sb.append("branch_trajectory_pair_mode: ").append(args.branchTrajectoryPairMode).append('\n');
        sb.append("top_k: ").append(args.topK).append('\n');
        sb.append("random_extra: ").append(args.randomExtra).append('\n');
        sb.append("target_temperature: ").append(String.format(Locale.US, "%.4f", args.targetTemperature)).append('\n');
        sb.append("win_turn_bonus: ").append(String.format(Locale.US, "%.4f", args.winTurnBonus)).append('\n');
        sb.append("loss_turn_bonus: ").append(String.format(Locale.US, "%.4f", args.lossTurnBonus)).append('\n');
        sb.append("skip_pass_best: ").append(args.skipPassBest).append('\n');
        sb.append("terminal_mode: ").append(args.terminalMode).append('\n');
        sb.append("train_epochs: ").append(args.trainEpochs).append('\n');
        sb.append("candidate_permutations: ").append(args.candidatePermutations).append('\n');
        sb.append("trained_scenarios: ").append(trainedScenarios).append('\n');
        sb.append("skipped_scenarios: ").append(skippedScenarios).append('\n');
        sb.append("candidate_examples: ").append(candidateExamples).append('\n');
        sb.append("winning_trajectories: ").append(winningTrajectories).append('\n');
        sb.append("train_pass_samples: ").append(trainPassSamples).append('\n');
        sb.append("opponent_mode: ").append(args.opponentMode).append('\n');
        sb.append("action_types: ").append(args.targetTypes).append('\n');
        sb.append("batch_size: ").append(args.batchSize).append('\n');
        sb.append("workers: ").append(args.workers).append('\n');
        sb.append("timeout_sec: ").append(args.timeoutSec).append('\n');
        sb.append("scenario_timeout_sec: ").append(args.scenarioTimeoutSec).append('\n');
        sb.append("max_game_turns: ").append(args.maxGameTurns).append('\n');
        sb.append("seed: ").append(args.seed).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        sb.append("agent_decks:\n");
        for (Path deck : agentDecks) {
            sb.append("- ").append(deck).append('\n');
        }
        sb.append("opponent_decks:\n");
        for (Path deck : oppDecks) {
            sb.append("- ").append(deck).append('\n');
        }
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeImportedTrainingReadme(
            Path path,
            Args args,
            ImportTrainingStats importStats,
            long elapsedMs
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# Imported Action Training\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("import_training_data_path: ").append(args.importTrainingDataPath).append('\n');
        sb.append("import_flat_as_terminal_episodes: ").append(args.importFlatAsTerminalEpisodes).append('\n');
        if (args.importFlatAsTerminalEpisodes) {
            sb.append("trajectory_final_reward: ")
                    .append(String.format(Locale.US, "%.4f", args.trajectoryFinalReward)).append('\n');
        }
        sb.append("training_examples: ").append(importStats.trainingExamples).append('\n');
        sb.append("train_epochs: ").append(args.trainEpochs).append('\n');
        sb.append("candidate_permutations: ").append(args.candidatePermutations).append('\n');
        sb.append("batch_size: ").append(args.batchSize).append('\n');
        sb.append("branch_return_targets: ").append(args.branchReturnTargets).append('\n');
        sb.append("branch_return_balance: ").append(args.branchReturnBalance).append('\n');
        sb.append("branch_return_max_negatives_per_positive: ")
                .append(args.branchReturnMaxNegativesPerPositive).append('\n');
        if (importStats.branchReturnBalanceStats != null) {
            BranchReturnBalanceStats balance = importStats.branchReturnBalanceStats;
            sb.append("branch_return_eligible_positive: ").append(balance.eligiblePositive).append('\n');
            sb.append("branch_return_eligible_negative: ").append(balance.eligibleNegative).append('\n');
            sb.append("branch_return_eligible_none: ").append(balance.eligibleNone).append('\n');
            sb.append("branch_return_accepted_positive: ").append(balance.acceptedPositive).append('\n');
            sb.append("branch_return_accepted_negative: ").append(balance.acceptedNegative).append('\n');
            sb.append("branch_return_skipped_negative: ").append(balance.skippedNegative).append('\n');
            sb.append("branch_return_skipped_none: ").append(balance.skippedNone).append('\n');
        }
        sb.append("train_pass_samples: ").append(importStats.trainPassSamples).append('\n');
        sb.append("post_train_wait_ms: ").append(args.postTrainWaitMs).append('\n');
        sb.append("seed: ").append(args.seed).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeImportedTrajectoryReadme(
            Path path,
            Args args,
            TrajectoryImportStats stats,
            long elapsedMs
    ) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# Imported Trajectory RL Training\n\n");
        sb.append("date: ").append(LocalDateTime.now()).append('\n');
        sb.append("import_trajectory_data_path: ").append(args.importTrajectoryDataPath).append('\n');
        sb.append("trajectory_episodes: ").append(stats.episodes).append('\n');
        sb.append("trajectory_steps: ").append(stats.steps).append('\n');
        sb.append("trajectory_final_reward: ").append(String.format(Locale.US, "%.4f", args.trajectoryFinalReward)).append('\n');
        sb.append("train_epochs: ").append(args.trainEpochs).append('\n');
        sb.append("candidate_permutations: ").append(args.candidatePermutations).append('\n');
        sb.append("train_pass_samples: ").append(stats.trainPassSamples).append('\n');
        sb.append("post_train_wait_ms: ").append(args.postTrainWaitMs).append('\n');
        sb.append("seed: ").append(args.seed).append('\n');
        sb.append("elapsed_sec: ").append(String.format(Locale.US, "%.1f", elapsedMs / 1000.0)).append('\n');
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static String csv(String value) {
        String s = value == null ? "" : value;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private static String safeText(String value) {
        return value == null ? "" : value;
    }

    private static Path stackPriorityTracePath() {
        String raw = EnvConfig.str("EVAL_REPLAY_STACK_PRIORITY_TRACE_FILE", "").trim();
        if (raw.isEmpty()) {
            return null;
        }
        return Paths.get(raw).toAbsolutePath().normalize();
    }

    private static void initializeOpponentTranscriptMismatch(Path path) throws IOException {
        if (path == null) {
            return;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        Files.write(path,
                "scenario,seed,decision_index,reason,expected_event,actual_event,expected_source_turn,actual_source_turn,expected_phase,actual_phase,expected_action_text,actual_candidate_texts,detail,state\n"
                        .getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static void appendStackPriorityTrace(
            Game game,
            Player player,
            String event,
            boolean stackShortCircuit,
            boolean result,
            long before,
            long after,
            long directBefore,
            long directAfter,
            long directAccessBefore,
            long directAccessAfter,
            int decisionIndex
    ) {
        if (!STACK_PRIORITY_TRACE_ENABLED || STACK_PRIORITY_TRACE_FILE == null || game == null || player == null) {
            return;
        }
        try {
            StringBuilder sb = new StringBuilder(1536);
            sb.append('{');
            appendJsonString(sb, "event", event);
            sb.append(',');
            appendJsonNumber(sb, "decision_index", decisionIndex);
            sb.append(',');
            appendJsonString(sb, "actor", player.getName());
            sb.append(',');
            appendJsonString(sb, "actor_class", player.getClass().getName());
            sb.append(',');
            appendJsonNumber(sb, "turn", safeTurn(game));
            sb.append(',');
            appendJsonNumber(sb, "source_turn", compactSourceTurn(safeTurn(game)));
            sb.append(',');
            appendJsonString(sb, "phase", String.valueOf(game.getTurnStepType()));
            sb.append(',');
            appendJsonString(sb, "active", tracePlayerName(game, game.getActivePlayerId()));
            sb.append(',');
            appendJsonString(sb, "priority", tracePlayerName(game, game.getPriorityPlayerId()));
            sb.append(',');
            appendJsonBoolean(sb, "stack_short_circuit", stackShortCircuit);
            sb.append(',');
            appendJsonBoolean(sb, "priority_result", result);
            sb.append(',');
            appendJsonNumber(sb, "stack_size", game.getStack() == null ? 0 : game.getStack().size());
            sb.append(',');
            appendJsonString(sb, "stack_top", traceStackTop(game));
            sb.append(',');
            appendJsonNumber(sb, "random_util_count_before", before);
            sb.append(',');
            appendJsonNumber(sb, "random_util_count_after", after);
            sb.append(',');
            appendJsonNumber(sb, "random_util_delta", Math.max(0L, after - before));
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_count_before", directBefore);
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_count_after", directAfter);
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_delta", Math.max(0L, directAfter - directBefore));
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_access_before", directAccessBefore);
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_access_after", directAccessAfter);
            sb.append(',');
            appendJsonNumber(sb, "random_util_direct_access_delta", Math.max(0L, directAccessAfter - directAccessBefore));
            sb.append(',');
            appendJsonString(sb, "state", StateSnapshot.capture(game, player).toCompactText());
            sb.append('}');
            Path parent = STACK_PRIORITY_TRACE_FILE.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            synchronized (STACK_PRIORITY_TRACE_LOCK) {
                Files.write(STACK_PRIORITY_TRACE_FILE,
                        ("REPLAY_STACK_PRIORITY_JSON: " + sb.toString() + System.lineSeparator())
                                .getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            }
        } catch (Throwable ignored) {
            // Trace durability must not affect replay behavior.
        }
    }

    private static void appendPreTargetPassTrace(
            Game game,
            Player player,
            String event,
            ReplayExpectation expected,
            int eligibleOrdinal,
            int previousSourceDecisionNumber,
            int sourceDecisionGap,
            int syntheticPassCountBefore,
            String reason,
            OpponentTranscriptCursor transcript
    ) {
        if (!STACK_PRIORITY_TRACE_ENABLED || STACK_PRIORITY_TRACE_FILE == null || game == null || player == null) {
            return;
        }
        try {
            OpponentDecision next = transcript == null ? null : transcript.peek();
            StringBuilder sb = new StringBuilder(2048);
            sb.append('{');
            appendJsonString(sb, "event", event);
            sb.append(',');
            appendJsonString(sb, "reason", reason == null ? "" : reason);
            sb.append(',');
            appendJsonNumber(sb, "eligible_ordinal", eligibleOrdinal);
            sb.append(',');
            appendJsonNumber(sb, "previous_source_decision_number", previousSourceDecisionNumber);
            sb.append(',');
            appendJsonNumber(sb, "expected_source_decision_number",
                    sourceDecisionNumberForTrace(expected));
            sb.append(',');
            appendJsonString(sb, "expected_source_anchor_id", expected == null ? "" : expected.sourceAnchorId);
            sb.append(',');
            appendJsonString(sb, "expected_source_turn", expected == null ? "" : expected.sourceTurn);
            sb.append(',');
            appendJsonString(sb, "expected_source_phase", expected == null ? "" : expected.sourcePhase);
            sb.append(',');
            appendJsonString(sb, "expected_source_actor", expected == null ? "" : expected.sourceActor);
            sb.append(',');
            appendJsonString(sb, "expected_text", expected == null ? "" : expected.expectedText);
            sb.append(',');
            appendJsonNumber(sb, "source_decision_gap", sourceDecisionGap);
            sb.append(',');
            appendJsonNumber(sb, "synthetic_pre_target_passes_before", syntheticPassCountBefore);
            sb.append(',');
            appendJsonNumber(sb, "synthetic_pre_target_pass_number", syntheticPassCountBefore + 1L);
            sb.append(',');
            appendJsonNumber(sb, "turn", safeTurn(game));
            sb.append(',');
            appendJsonNumber(sb, "source_turn", compactSourceTurn(safeTurn(game)));
            sb.append(',');
            appendJsonString(sb, "phase", String.valueOf(game.getTurnStepType()));
            sb.append(',');
            appendJsonString(sb, "active", tracePlayerName(game, game.getActivePlayerId()));
            sb.append(',');
            appendJsonString(sb, "priority", tracePlayerName(game, game.getPriorityPlayerId()));
            sb.append(',');
            appendJsonNumber(sb, "stack_size", game.getStack() == null ? 0 : game.getStack().size());
            sb.append(',');
            appendJsonString(sb, "stack_top", traceStackTop(game));
            sb.append(',');
            appendJsonNumber(sb, "opponent_transcript_index", transcript == null ? -1 : transcript.currentIndex());
            sb.append(',');
            appendJsonNumber(sb, "opponent_transcript_size", transcript == null ? -1 : transcript.size());
            sb.append(',');
            appendJsonNumber(sb, "opponent_next_decision_index", next == null ? -1 : next.decisionIndex);
            sb.append(',');
            appendJsonString(sb, "opponent_next_event", next == null ? "" : next.event);
            sb.append(',');
            appendJsonNumber(sb, "opponent_next_source_turn", next == null ? -1 : next.sourceTurn);
            sb.append(',');
            appendJsonString(sb, "opponent_next_phase", next == null ? "" : next.phase);
            sb.append(',');
            appendJsonString(sb, "opponent_next_action_text", next == null ? "" : next.chosenActionText);
            sb.append(',');
            appendJsonString(sb, "state", StateSnapshot.capture(game, player).toCompactText());
            sb.append('}');
            Path parent = STACK_PRIORITY_TRACE_FILE.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            synchronized (STACK_PRIORITY_TRACE_LOCK) {
                Files.write(STACK_PRIORITY_TRACE_FILE,
                        ("REPLAY_PRE_TARGET_PASS_JSON: " + sb.toString() + System.lineSeparator())
                                .getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            }
        } catch (Throwable ignored) {
            // Trace durability must not affect gameplay or replay control flow.
        }
    }

    private static int sourceDecisionNumberForTrace(ReplayExpectation expected) {
        if (expected == null || expected.sourceDecisionNumber == null
                || !expected.sourceDecisionNumber.matches("\\d+")) {
            return -1;
        }
        try {
            return Integer.parseInt(expected.sourceDecisionNumber);
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private static String tracePlayerName(Game game, UUID playerId) {
        try {
            Player player = playerId == null ? null : game.getPlayer(playerId);
            return player == null ? "" : player.getName();
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static String traceStackTop(Game game) {
        try {
            StackObject top = game == null || game.getStack() == null ? null : game.getStack().getFirstOrNull();
            return top == null ? "" : top.toString();
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static final class OpponentTranscript {

        private static final OpponentTranscript EMPTY =
                new OpponentTranscript(Collections.emptyMap(), null);

        private final Map<String, List<OpponentDecision>> byScenarioSeed;
        private final Path mismatchPath;

        private OpponentTranscript(Map<String, List<OpponentDecision>> byScenarioSeed, Path mismatchPath) {
            this.byScenarioSeed = byScenarioSeed == null ? Collections.emptyMap() : byScenarioSeed;
            this.mismatchPath = mismatchPath;
        }

        static OpponentTranscript load(Path path, Path mismatchPath) throws IOException {
            if (path == null) {
                return new OpponentTranscript(Collections.emptyMap(), mismatchPath);
            }
            Map<String, List<OpponentDecision>> out = new LinkedHashMap<>();
            for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
                int idx = line.indexOf(OPPONENT_DECISION_JSON_PREFIX);
                if (idx < 0) {
                    continue;
                }
                String json = line.substring(idx + OPPONENT_DECISION_JSON_PREFIX.length()).trim();
                if (json.isEmpty()) {
                    continue;
                }
                OpponentDecision decision = OpponentDecision.parse(json);
                if (decision == null) {
                    continue;
                }
                out.computeIfAbsent(key(decision.scenario, decision.seed), ignored -> new ArrayList<>())
                        .add(decision);
            }
            for (List<OpponentDecision> decisions : out.values()) {
                decisions.sort(Comparator.comparingInt((OpponentDecision d) -> d.decisionIndex));
            }
            return new OpponentTranscript(out, mismatchPath);
        }

        OpponentTranscriptCursor cursor(ScenarioJob job) {
            if (job == null) {
                return OpponentTranscriptCursor.empty();
            }
            List<OpponentDecision> decisions = byScenarioSeed.get(key(job.scenario, job.seed));
            return new OpponentTranscriptCursor(job.scenario, job.seed, decisions, mismatchPath);
        }

        private static String key(int scenario, long seed) {
            return scenario + "|" + seed;
        }
    }

    private static final class OpponentDecision {

        final int scenario;
        final long seed;
        final int decisionIndex;
        final String actor;
        final String event;
        final int sourceTurn;
        final String phase;
        final String chosenActionText;
        final List<String> candidateTexts;
        final String visibleState;
        final String hiddenStateProvenance;
        final List<String> actorHandNames;
        final List<String> actorLibraryNames;
        final List<String> actorLibraryTopNames;
        final List<String> candidateObjectIds;
        final List<String> chosenObjectIds;
        final List<String> selectedObjectIds;
        final List<String> sourceObjectIds;
        final List<String> targetObjectIds;
        final List<String> tapSourceObjectIds;
        final List<String> manaPaymentSourceObjectIds;
        final List<String> plannedCandidateTexts;
        final List<String> legalCandidateTexts;
        final List<String> plannedCandidateObjectIds;
        final List<String> legalCandidateObjectIds;
        final List<String> visibleSelfBattlefieldNames;
        final List<String> visibleOpponentBattlefieldNames;
        final List<String> visibleSelfBattlefieldObjectIds;
        final List<String> visibleOpponentBattlefieldObjectIds;
        final List<String> visibleBattlefieldObjectIds;
        final List<String> attachmentObjectIds;
        final List<String> attachedToObjectIds;
        final List<String> attachmentContext;
        final List<String> equippedToObjectIds;
        final List<String> equipmentContext;
        final List<String> combatAttackerObjectIds;
        final List<String> combatBlockerObjectIds;

        private OpponentDecision(int scenario, long seed, int decisionIndex, String actor, String event,
                                 int sourceTurn, String phase, String chosenActionText,
                                 List<String> candidateTexts, String visibleState,
                                 String hiddenStateProvenance,
                                 List<String> actorHandNames,
                                 List<String> actorLibraryNames,
                                 List<String> actorLibraryTopNames,
                                 List<String> candidateObjectIds,
                                 List<String> chosenObjectIds,
                                 List<String> selectedObjectIds,
                                 List<String> sourceObjectIds,
                                 List<String> targetObjectIds,
                                 List<String> tapSourceObjectIds,
                                 List<String> manaPaymentSourceObjectIds,
                                  List<String> plannedCandidateTexts,
                                  List<String> legalCandidateTexts,
                                  List<String> plannedCandidateObjectIds,
                                  List<String> legalCandidateObjectIds,
                                  List<String> visibleSelfBattlefieldNames,
                                  List<String> visibleOpponentBattlefieldNames,
                                  List<String> visibleSelfBattlefieldObjectIds,
                                  List<String> visibleOpponentBattlefieldObjectIds,
                                  List<String> visibleBattlefieldObjectIds,
                                 List<String> attachmentObjectIds,
                                 List<String> attachedToObjectIds,
                                 List<String> attachmentContext,
                                 List<String> equippedToObjectIds,
                                 List<String> equipmentContext,
                                 List<String> combatAttackerObjectIds,
                                 List<String> combatBlockerObjectIds) {
            this.scenario = scenario;
            this.seed = seed;
            this.decisionIndex = decisionIndex;
            this.actor = actor == null ? "" : actor;
            this.event = event == null ? "" : event;
            this.sourceTurn = sourceTurn;
            this.phase = phase == null ? "" : phase;
            this.chosenActionText = chosenActionText == null ? "" : chosenActionText;
            this.candidateTexts = candidateTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(candidateTexts);
            this.visibleState = visibleState == null ? "" : visibleState;
            this.hiddenStateProvenance = hiddenStateProvenance == null ? "" : hiddenStateProvenance;
            this.actorHandNames = actorHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actorHandNames);
            this.actorLibraryNames = actorLibraryNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actorLibraryNames);
            this.actorLibraryTopNames = actorLibraryTopNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actorLibraryTopNames);
            this.candidateObjectIds = copyList(candidateObjectIds);
            this.chosenObjectIds = copyList(chosenObjectIds);
            this.selectedObjectIds = copyList(selectedObjectIds);
            this.sourceObjectIds = copyList(sourceObjectIds);
            this.targetObjectIds = copyList(targetObjectIds);
            this.tapSourceObjectIds = copyList(tapSourceObjectIds);
            this.manaPaymentSourceObjectIds = copyList(manaPaymentSourceObjectIds);
            this.plannedCandidateTexts = copyList(plannedCandidateTexts);
            this.legalCandidateTexts = copyList(legalCandidateTexts);
            this.plannedCandidateObjectIds = copyList(plannedCandidateObjectIds);
            this.legalCandidateObjectIds = copyList(legalCandidateObjectIds);
            this.visibleSelfBattlefieldNames = copyList(visibleSelfBattlefieldNames);
            this.visibleOpponentBattlefieldNames = copyList(visibleOpponentBattlefieldNames);
            this.visibleSelfBattlefieldObjectIds = copyList(visibleSelfBattlefieldObjectIds);
            this.visibleOpponentBattlefieldObjectIds = copyList(visibleOpponentBattlefieldObjectIds);
            this.visibleBattlefieldObjectIds = copyList(visibleBattlefieldObjectIds);
            this.attachmentObjectIds = copyList(attachmentObjectIds);
            this.attachedToObjectIds = copyList(attachedToObjectIds);
            this.attachmentContext = copyList(attachmentContext);
            this.equippedToObjectIds = copyList(equippedToObjectIds);
            this.equipmentContext = copyList(equipmentContext);
            this.combatAttackerObjectIds = copyList(combatAttackerObjectIds);
            this.combatBlockerObjectIds = copyList(combatBlockerObjectIds);
        }

        static OpponentDecision parse(String json) {
            try {
                com.google.gson.JsonObject obj = com.google.gson.JsonParser.parseString(json).getAsJsonObject();
                int sourceTurn = jsonInt(obj, "source_turn", -1);
                if (sourceTurn < 0) {
                    sourceTurn = compactSourceTurn(jsonInt(obj, "turn", -1));
                }
                return new OpponentDecision(
                        jsonInt(obj, "scenario", -1),
                        jsonLong(obj, "seed", -1L),
                        jsonInt(obj, "decision_index", -1),
                        jsonString(obj, "actor"),
                        jsonString(obj, "event"),
                        sourceTurn,
                        jsonString(obj, "phase"),
                        jsonString(obj, "chosen_action_text"),
                        jsonStringArray(obj, "candidate_texts"),
                        jsonString(obj, "visible_state"),
                        jsonString(obj, "hidden_state_provenance"),
                        jsonStringArray(obj, "actor_hand"),
                        jsonStringArray(obj, "actor_library"),
                        jsonStringArray(obj, "actor_library_top"),
                        jsonStringArray(obj, "candidate_object_ids"),
                        jsonStringArray(obj, "chosen_object_ids"),
                        jsonStringArray(obj, "selected_object_ids"),
                        objectIdsFromScalarOrArray(obj, "source_object_id", "source_object_ids"),
                        objectIdsFromScalarOrArray(obj, "target_object_id", "target_object_ids"),
                        jsonStringArray(obj, "tap_source_object_ids"),
                        jsonStringArray(obj, "mana_payment_source_object_ids"),
                        jsonStringArray(obj, "planned_candidate_texts"),
                        jsonStringArray(obj, "legal_candidate_texts"),
                        jsonStringArray(obj, "planned_candidate_object_ids"),
                        jsonStringArray(obj, "legal_candidate_object_ids"),
                        jsonStringArray(obj, "visible_self_battlefield_names"),
                        jsonStringArray(obj, "visible_opponent_battlefield_names"),
                        jsonStringArray(obj, "visible_self_battlefield_object_ids"),
                        jsonStringArray(obj, "visible_opponent_battlefield_object_ids"),
                        jsonStringArray(obj, "visible_battlefield_object_ids"),
                        jsonStringArray(obj, "attachment_object_ids"),
                        jsonStringArray(obj, "attached_to_object_ids"),
                        jsonStringArray(obj, "attachment_context"),
                        jsonStringArray(obj, "equipped_to_object_ids"),
                        jsonStringArray(obj, "equipment_context"),
                        jsonStringArray(obj, "combat_attacker_object_ids"),
                        jsonStringArray(obj, "combat_blocker_object_ids"));
            } catch (Exception ignored) {
                return null;
            }
        }

        HiddenStateFailure verifyHiddenStateForReplay(Game game) {
            if (!requiresHiddenHandProvenance()) {
                return HiddenStateFailure.none();
            }
            String sourceName = actionSourceName(chosenActionText);
            if (sourceName.isEmpty()) {
                return HiddenStateFailure.none();
            }
            if (hiddenStateProvenance.isEmpty()
                    || actorHandNames.isEmpty()
                    || (actorLibraryNames.isEmpty() && actorLibraryTopNames.isEmpty())) {
                return HiddenStateFailure.failed(
                        "opponent_opening_state_unverifiable",
                        "hidden-hand transcript action lacks actor hand/library provenance"
                                + " source=" + sourceName
                                + " provenance=" + hiddenStateProvenance
                                + " actor_hand_count=" + actorHandNames.size()
                                + " actor_library_count=" + actorLibraryNames.size()
                                + " actor_library_top_count=" + actorLibraryTopNames.size());
            }
            if (!containsCardName(actorHandNames, sourceName)) {
                return HiddenStateFailure.failed(
                        "opponent_opening_state_mismatch",
                        "source transcript actor hand did not contain hidden-action source"
                                + " source=" + sourceName
                                + " actor_hand=" + joinTexts(actorHandNames));
            }
            Player liveActor = liveActor(game);
            if (liveActor == null) {
                return HiddenStateFailure.failed(
                        "opponent_opening_state_unverifiable",
                        "could not resolve live opponent actor for hidden-state verification"
                                + " source=" + sourceName);
            }
            List<String> liveHand = cardNames(liveActor.getHand().getCards(game), 0);
            if (!nameMultisetMatches(actorHandNames, liveHand)) {
                return HiddenStateFailure.failed(
                        "opponent_opening_state_mismatch",
                        "source/live opponent hand mismatch before transcript action"
                                + " source=" + sourceName
                                + " source_hand=" + joinTexts(actorHandNames)
                                + " live_hand=" + joinTexts(liveHand));
            }
            List<String> expectedLibraryTop = actorLibraryTopNames.isEmpty()
                    ? prefix(actorLibraryNames, 12)
                    : actorLibraryTopNames;
            List<String> liveLibraryTop = cardNames(liveActor.getLibrary().getCards(game), expectedLibraryTop.size());
            if (!libraryPrefixMatches(expectedLibraryTop, liveLibraryTop)) {
                return HiddenStateFailure.failed(
                        "opponent_opening_state_mismatch",
                        "source/live opponent library-top mismatch before transcript action"
                                + " source=" + sourceName
                                + " source_library_top=" + joinTexts(expectedLibraryTop)
                                + " live_library_top=" + joinTexts(liveLibraryTop));
            }
            return HiddenStateFailure.none();
        }

        ObjectContextFailure verifyObjectContextForReplay(Game game) {
            if (!requiresObjectContextProvenance()) {
                return ObjectContextFailure.none();
            }
            if (visibleBattlefieldObjectIds.isEmpty()
                    && visibleSelfBattlefieldObjectIds.isEmpty()
                    && visibleOpponentBattlefieldObjectIds.isEmpty()) {
                return ObjectContextFailure.failed(
                        "opponent_object_context_unverifiable",
                        "opponent transcript row lacks visible battlefield object ids"
                                + " event=" + event
                                + " action=" + chosenActionText);
            }
            Player actorPlayer = liveActorByName(game);
            if (actorPlayer == null) {
                return ObjectContextFailure.failed(
                        "opponent_object_context_unverifiable",
                        "could not resolve live opponent actor for object-context verification"
                                + " actor=" + actor);
            }
            ObjectContextFailure battlefieldFailure = verifyBattlefieldObjectIds(game, actorPlayer);
            if (battlefieldFailure.failed()) {
                return battlefieldFailure;
            }
            if (!sourceObjectIds.isEmpty() && !sourceObjectHasNameProvenance()) {
                ObjectContextFailure sourceFailure = verifyResolvableObjectIds(game, sourceObjectIds,
                        "source_object_id_mismatch", "source object id missing in live replay");
                if (sourceFailure.failed()) {
                    return sourceFailure;
                }
            }
            if (sourceObjectIds.isEmpty() && candidateObjectIds.isEmpty() && chosenObjectIds.isEmpty()
                    && selectedObjectIds.isEmpty() && isPriorityObjectSensitiveRow()) {
                return ObjectContextFailure.failed(
                        "opponent_object_context_unverifiable",
                        "opponent priority transcript row lacks structured source/chosen object ids"
                                + " action=" + chosenActionText);
            }
            if (requiresTargetObjectIds() && targetObjectIds.isEmpty()) {
                return ObjectContextFailure.failed(
                        "opponent_object_context_unverifiable",
                        "opponent transcript target-bearing action lacks structured target object ids"
                                + " action=" + chosenActionText);
            }
            if (!targetObjectIds.isEmpty()) {
                ObjectContextFailure targetFailure = verifyResolvableObjectIds(game, targetObjectIds,
                        "target_object_id_mismatch", "target object id missing in live replay");
                if (targetFailure.failed()) {
                    return targetFailure;
                }
            }
            if (!tapSourceObjectIds.isEmpty()) {
                ObjectContextFailure tapFailure = verifyResolvableObjectIds(game, tapSourceObjectIds,
                        "tap_source_object_id_mismatch", "tap source object id missing in live replay");
                if (tapFailure.failed()) {
                    return tapFailure;
                }
            }
            if (!manaPaymentSourceObjectIds.isEmpty()) {
                ObjectContextFailure manaFailure = verifyResolvableObjectIds(game, manaPaymentSourceObjectIds,
                        "mana_payment_source_object_id_mismatch",
                        "mana payment source object id missing in live replay");
                if (manaFailure.failed()) {
                    return manaFailure;
                }
            }
            if (isCombatObjectSensitiveRow()) {
                List<String> expectedCombatIds = "DECLARE_ATTACKERS".equalsIgnoreCase(event)
                        ? combatAttackerObjectIds
                        : combatBlockerObjectIds;
                if (expectedCombatIds.isEmpty() && selectedObjectIds.isEmpty() && chosenObjectIds.isEmpty()) {
                    return ObjectContextFailure.failed(
                            "opponent_object_context_unverifiable",
                            "opponent combat transcript row lacks selected combat permanent object ids"
                                    + " event=" + event
                                    + " action=" + chosenActionText);
                }
            }
            return ObjectContextFailure.none();
        }

        private boolean requiresHiddenHandProvenance() {
            if (!"PRIORITY".equalsIgnoreCase(event) || isPassText(chosenActionText)) {
                return false;
            }
            String sourceName = actionSourceName(chosenActionText);
            return !sourceName.isEmpty() && !visibleStateContainsSource(sourceName);
        }

        private boolean requiresObjectContextProvenance() {
            return isPriorityObjectSensitiveRow() || isCombatObjectSensitiveRow();
        }

        private boolean isPriorityObjectSensitiveRow() {
            return "PRIORITY".equalsIgnoreCase(event) && !isPassText(chosenActionText);
        }

        private boolean isCombatObjectSensitiveRow() {
            return "DECLARE_ATTACKERS".equalsIgnoreCase(event)
                    || "DECLARE_BLOCKERS".equalsIgnoreCase(event);
        }

        private boolean requiresTargetObjectIds() {
            String text = chosenActionText == null ? "" : chosenActionText;
            return text.contains("->") || text.toLowerCase(Locale.ROOT).contains("target");
        }

        private ObjectContextFailure verifyBattlefieldObjectIds(Game game, Player actorPlayer) {
            if (!visibleSelfBattlefieldObjectIds.isEmpty()) {
                if (!visibleSelfBattlefieldNames.isEmpty()) {
                    List<String> liveSelfNames = battlefieldNames(game, actorPlayer.getId());
                    if (!nameMultisetMatches(visibleSelfBattlefieldNames, liveSelfNames)) {
                        return ObjectContextFailure.failed(
                                "opponent_battlefield_name_context_mismatch",
                                "source/live opponent self battlefield names differ"
                                        + " expected=" + joinTexts(visibleSelfBattlefieldNames)
                                        + " live=" + joinTexts(liveSelfNames));
                    }
                } else {
                    List<String> liveSelfIds = battlefieldObjectIds(game, actorPlayer.getId());
                    if (!idSet(visibleSelfBattlefieldObjectIds).equals(idSet(liveSelfIds))) {
                        return ObjectContextFailure.failed(
                                "opponent_battlefield_object_context_mismatch",
                                "source/live opponent self battlefield ids differ"
                                        + " expected=" + joinTexts(visibleSelfBattlefieldObjectIds)
                                        + " live=" + joinTexts(liveSelfIds));
                    }
                }
            }
            if (!visibleOpponentBattlefieldObjectIds.isEmpty()) {
                if (!visibleOpponentBattlefieldNames.isEmpty()) {
                    List<String> liveOpponentNames = opponentBattlefieldNames(game, actorPlayer.getId());
                    if (!nameMultisetMatches(visibleOpponentBattlefieldNames, liveOpponentNames)) {
                        return ObjectContextFailure.failed(
                                "opponent_battlefield_name_context_mismatch",
                                "source/live opponent-view opposing battlefield names differ"
                                        + " expected=" + joinTexts(visibleOpponentBattlefieldNames)
                                        + " live=" + joinTexts(liveOpponentNames));
                    }
                } else {
                    List<String> liveOpponentIds = opponentBattlefieldObjectIds(game, actorPlayer.getId());
                    if (!idSet(visibleOpponentBattlefieldObjectIds).equals(idSet(liveOpponentIds))) {
                        return ObjectContextFailure.failed(
                                "opponent_battlefield_object_context_mismatch",
                                "source/live opponent-view opposing battlefield ids differ"
                                        + " expected=" + joinTexts(visibleOpponentBattlefieldObjectIds)
                                        + " live=" + joinTexts(liveOpponentIds));
                    }
                }
            } else if (!visibleBattlefieldObjectIds.isEmpty()) {
                List<String> liveAllIds = new ArrayList<>();
                liveAllIds.addAll(battlefieldObjectIds(game, actorPlayer.getId()));
                liveAllIds.addAll(opponentBattlefieldObjectIds(game, actorPlayer.getId()));
                if (!idSet(visibleBattlefieldObjectIds).equals(idSet(liveAllIds))) {
                    return ObjectContextFailure.failed(
                            "opponent_battlefield_object_context_mismatch",
                            "source/live visible battlefield ids differ"
                                    + " expected=" + joinTexts(visibleBattlefieldObjectIds)
                                    + " live=" + joinTexts(liveAllIds));
                }
            }
            return ObjectContextFailure.none();
        }

        private boolean sourceObjectHasNameProvenance() {
            String sourceName = actionSourceName(chosenActionText);
            return !sourceName.isEmpty()
                    && (containsCardName(actorHandNames, sourceName) || visibleStateContainsSource(sourceName));
        }

        private ObjectContextFailure verifyResolvableObjectIds(Game game, List<String> ids,
                                                               String reason, String message) {
            for (String raw : ids) {
                UUID id = parseUuid(raw);
                if (id == null) {
                    continue;
                }
                if (!isObjectVisibleOrKnown(game, id)) {
                    return ObjectContextFailure.failed(reason, message + " id=" + id);
                }
            }
            return ObjectContextFailure.none();
        }

        private Player liveActorByName(Game game) {
            if (game == null) {
                return null;
            }
            if (actor != null && !actor.isEmpty()) {
                try {
                    for (Player player : game.getPlayers().values()) {
                        if (player != null && actor.equals(player.getName())) {
                            return player;
                        }
                    }
                } catch (Exception ignored) {
                }
            }
            return liveActor(game);
        }

        private static List<String> battlefieldObjectIds(Game game, UUID controllerId) {
            if (game == null || game.getBattlefield() == null || controllerId == null) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
                if (permanent != null && controllerId.equals(permanent.getControllerId())) {
                    out.add(permanent.getId().toString());
                }
            }
            return out;
        }

        private static List<String> battlefieldNames(Game game, UUID controllerId) {
            if (game == null || game.getBattlefield() == null || controllerId == null) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
                if (permanent != null && controllerId.equals(permanent.getControllerId())) {
                    out.add(permanent.getName());
                }
            }
            return out;
        }

        private static List<String> opponentBattlefieldObjectIds(Game game, UUID actorId) {
            if (game == null || actorId == null) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            try {
                for (UUID opponentId : game.getOpponents(actorId)) {
                    out.addAll(battlefieldObjectIds(game, opponentId));
                }
            } catch (Exception ignored) {
            }
            return out;
        }

        private static List<String> opponentBattlefieldNames(Game game, UUID actorId) {
            if (game == null || actorId == null) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            try {
                for (UUID opponentId : game.getOpponents(actorId)) {
                    out.addAll(battlefieldNames(game, opponentId));
                }
            } catch (Exception ignored) {
            }
            return out;
        }

        private static boolean isObjectVisibleOrKnown(Game game, UUID id) {
            if (game == null || id == null) {
                return false;
            }
            try {
                return game.getObject(id) != null
                        || game.getCard(id) != null
                        || game.getPermanent(id) != null
                        || game.getPlayer(id) != null
                        || (game.getStack() != null && game.getStack().getStackObject(id) != null);
            } catch (Exception ignored) {
                return false;
            }
        }

        private static Set<String> idSet(List<String> ids) {
            Set<String> out = new LinkedHashSet<>();
            if (ids == null) {
                return out;
            }
            for (String raw : ids) {
                String id = raw == null ? "" : raw.trim();
                if (!id.isEmpty()) {
                    out.add(id);
                }
            }
            return out;
        }

        private static UUID parseUuid(String raw) {
            try {
                String value = raw == null ? "" : raw.trim();
                if (value.isEmpty() || value.startsWith("sentinel:")) {
                    return null;
                }
                return UUID.fromString(value);
            } catch (Exception ignored) {
                return null;
            }
        }

        private boolean visibleStateContainsSource(String sourceName) {
            return stateZoneContains(visibleState, "selfBattlefield", sourceName)
                    || stateZoneContains(visibleState, "selfGraveyard", sourceName)
                    || stateZoneContains(visibleState, "selfExile", sourceName);
        }

        private static boolean stateZoneContains(String stateText, String key, String sourceName) {
            String zone = stateField(stateText, key);
            if (zone.isEmpty()) {
                return false;
            }
            for (String item : zone.split("\\|", -1)) {
                if (normalizeZoneItem(item).equals(normalizeText(sourceName))) {
                    return true;
                }
            }
            return false;
        }

        private static String normalizeZoneItem(String text) {
            if (text == null) {
                return "";
            }
            String withoutFlags = text.replaceAll("\\[[^\\]]*\\]", "").replaceAll("\\{.*\\}", "");
            return normalizeText(withoutFlags);
        }

        private static String actionSourceName(String actionText) {
            if (actionText == null) {
                return "";
            }
            int idx = actionText.indexOf(':');
            if (idx <= 0) {
                return "";
            }
            return normalizeText(actionText.substring(0, idx));
        }

        private static boolean containsCardName(List<String> names, String sourceName) {
            for (String name : names) {
                if (normalizeText(name).equals(normalizeText(sourceName))) {
                    return true;
                }
            }
            return false;
        }

        private static boolean nameMultisetMatches(List<String> expectedNames, List<String> actualNames) {
            return nameCounts(expectedNames).equals(nameCounts(actualNames));
        }

        private static Map<String, Integer> nameCounts(List<String> names) {
            Map<String, Integer> out = new LinkedHashMap<>();
            if (names == null) {
                return out;
            }
            for (String raw : names) {
                String name = normalizeText(raw);
                if (name.isEmpty()) {
                    continue;
                }
                Integer cur = out.get(name);
                out.put(name, cur == null ? 1 : cur + 1);
            }
            return out;
        }

        private static boolean libraryPrefixMatches(List<String> expectedNames, List<String> actualNames) {
            if (expectedNames == null || expectedNames.isEmpty()) {
                return false;
            }
            if (actualNames == null || actualNames.size() < expectedNames.size()) {
                return false;
            }
            for (int i = 0; i < expectedNames.size(); i++) {
                if (!normalizeText(expectedNames.get(i)).equals(normalizeText(actualNames.get(i)))) {
                    return false;
                }
            }
            return true;
        }

        private static List<String> prefix(List<String> names, int limit) {
            if (names == null || names.isEmpty()) {
                return Collections.emptyList();
            }
            return new ArrayList<>(names.subList(0, Math.min(limit, names.size())));
        }

        private static Player liveActor(Game game) {
            if (game == null) {
                return null;
            }
            UUID priority = game.getPriorityPlayerId();
            Player player = priority == null ? null : game.getPlayer(priority);
            if (player != null) {
                return player;
            }
            UUID active = game.getActivePlayerId();
            return active == null ? null : game.getPlayer(active);
        }

        private static List<String> cardNames(Collection<Card> cards, int limit) {
            List<String> out = new ArrayList<>();
            if (cards == null) {
                return out;
            }
            for (Card card : cards) {
                if (card != null && card.getName() != null && !card.getName().isEmpty()) {
                    out.add(card.getName());
                }
                if (limit > 0 && out.size() >= limit) {
                    break;
                }
            }
            return out;
        }

        private static int jsonInt(com.google.gson.JsonObject obj, String key, int fallback) {
            try {
                return obj.has(key) && !obj.get(key).isJsonNull() ? obj.get(key).getAsInt() : fallback;
            } catch (Exception ignored) {
                return fallback;
            }
        }

        private static long jsonLong(com.google.gson.JsonObject obj, String key, long fallback) {
            try {
                return obj.has(key) && !obj.get(key).isJsonNull() ? obj.get(key).getAsLong() : fallback;
            } catch (Exception ignored) {
                return fallback;
            }
        }

        private static String jsonString(com.google.gson.JsonObject obj, String key) {
            try {
                return obj.has(key) && !obj.get(key).isJsonNull() ? obj.get(key).getAsString() : "";
            } catch (Exception ignored) {
                return "";
            }
        }

        private static List<String> jsonStringArray(com.google.gson.JsonObject obj, String key) {
            if (!obj.has(key) || obj.get(key).isJsonNull() || !obj.get(key).isJsonArray()) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (com.google.gson.JsonElement element : obj.get(key).getAsJsonArray()) {
                if (element != null && !element.isJsonNull()) {
                    out.add(element.getAsString());
                }
            }
            return out;
        }

        private static List<String> objectIdsFromScalarOrArray(com.google.gson.JsonObject obj,
                                                               String scalarKey,
                                                               String arrayKey) {
            List<String> out = new ArrayList<>(jsonStringArray(obj, arrayKey));
            String scalar = jsonString(obj, scalarKey);
            if (out.isEmpty() && scalar != null && !scalar.isEmpty()) {
                out.add(scalar);
            }
            return out;
        }

        private static List<String> copyList(List<String> values) {
            return values == null ? Collections.emptyList() : new ArrayList<>(values);
        }
    }

    private static final class HiddenStateFailure {
        final String reason;
        final String detail;

        private HiddenStateFailure(String reason, String detail) {
            this.reason = reason == null ? "" : reason;
            this.detail = detail == null ? "" : detail;
        }

        static HiddenStateFailure none() {
            return new HiddenStateFailure("", "");
        }

        static HiddenStateFailure failed(String reason, String detail) {
            return new HiddenStateFailure(reason, detail);
        }

        boolean failed() {
            return !reason.isEmpty();
        }
    }

    private static final class ObjectContextFailure {
        final String reason;
        final String detail;

        private ObjectContextFailure(String reason, String detail) {
            this.reason = reason == null ? "" : reason;
            this.detail = detail == null ? "" : detail;
        }

        static ObjectContextFailure none() {
            return new ObjectContextFailure("", "");
        }

        static ObjectContextFailure failed(String reason, String detail) {
            return new ObjectContextFailure(reason, detail);
        }

        boolean failed() {
            return !reason.isEmpty();
        }
    }

    private static final class OpponentTranscriptCursor {

        private final int scenario;
        private final long seed;
        private final List<OpponentDecision> decisions;
        private final Path mismatchPath;
        private int index = 0;
        private boolean mismatchRecorded = false;
        private int agentOrdinalContext = -1;
        private OpponentTranscriptMismatch firstMismatch = null;

        private OpponentTranscriptCursor(int scenario, long seed, List<OpponentDecision> decisions, Path mismatchPath) {
            this.scenario = scenario;
            this.seed = seed;
            this.decisions = decisions == null ? Collections.emptyList() : new ArrayList<>(decisions);
            this.mismatchPath = mismatchPath;
        }

        static OpponentTranscriptCursor empty() {
            return new OpponentTranscriptCursor(-1, -1L, Collections.emptyList(), null);
        }

        OpponentTranscriptCursor copy() {
            OpponentTranscriptCursor out = new OpponentTranscriptCursor(scenario, seed, decisions, mismatchPath);
            out.index = index;
            out.mismatchRecorded = mismatchRecorded;
            out.agentOrdinalContext = agentOrdinalContext;
            out.firstMismatch = firstMismatch;
            return out;
        }

        boolean isEmpty() {
            return decisions.isEmpty();
        }

        OpponentDecision consumeForContext(String expectedEvent, Game game,
                                           List<String> actualCandidateTexts, String actualEvent) {
            if (decisions.isEmpty()) {
                recordMismatch(null, game, actualEvent, actualCandidateTexts,
                        "transcript_empty", "no opponent transcript rows for scenario/seed");
                return null;
            }
            if (index >= decisions.size()) {
                recordMismatch(null, game, actualEvent, actualCandidateTexts,
                        "transcript_exhausted", "opponent transcript ended before replay did");
                return null;
            }
            OpponentDecision expected = decisions.get(index);
            int actualSourceTurn = compactSourceTurn(safeTurn(game));
            String actualPhase = game == null || game.getTurnStepType() == null
                    ? ""
                    : String.valueOf(game.getTurnStepType());
            if (!eventMatches(expected.event, expectedEvent)
                    || (expected.sourceTurn >= 0 && expected.sourceTurn != actualSourceTurn)
                    || !phaseMatches(expected.phase, actualPhase)) {
                recordMismatch(expected, game, actualEvent, actualCandidateTexts,
                        "opponent_transcript_context_mismatch",
                        "expected transcript row did not match replay opponent context");
                return null;
            }
            HiddenStateFailure hiddenStateFailure = expected.verifyHiddenStateForReplay(game);
            if (hiddenStateFailure.failed()) {
                recordMismatch(expected, game, actualEvent, actualCandidateTexts,
                        hiddenStateFailure.reason, hiddenStateFailure.detail);
                return null;
            }
            ObjectContextFailure objectContextFailure = expected.verifyObjectContextForReplay(game);
            if (objectContextFailure.failed()) {
                recordMismatch(expected, game, actualEvent, actualCandidateTexts,
                        objectContextFailure.reason, objectContextFailure.detail);
                return null;
            }
            index++;
            return expected;
        }

        OpponentDecision peek() {
            if (decisions.isEmpty() || index < 0 || index >= decisions.size()) {
                return null;
            }
            return decisions.get(index);
        }

        int currentIndex() {
            return index;
        }

        int size() {
            return decisions.size();
        }

        void setAgentOrdinalContext(int ordinal) {
            this.agentOrdinalContext = ordinal;
        }

        boolean agentOrdinalAtLeast(int minOrdinal) {
            return agentOrdinalContext >= minOrdinal;
        }

        boolean hasMismatchAtOrBefore(int ordinalLimit) {
            if (firstMismatch == null) {
                return false;
            }
            return firstMismatch.agentOrdinalContext < 0
                    || firstMismatch.agentOrdinalContext <= ordinalLimit;
        }

        int mismatchOrdinalOr(int fallback) {
            if (firstMismatch == null || firstMismatch.agentOrdinalContext < 0) {
                return fallback;
            }
            return firstMismatch.agentOrdinalContext;
        }

        PrefixDivergence toPrefixDivergence(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType,
                List<Integer> expectedIndices,
                List<String> expectedTexts,
                int forcedPrefixCount,
                StateSnapshot fallbackState,
                ReplayExpectation sourceExpectation
        ) {
            if (firstMismatch == null) {
                return null;
            }
            return new PrefixDivergence(
                    ordinal,
                    firstMismatch.reason,
                    expectedActionType,
                    null,
                    expectedIndices,
                    expectedTexts,
                    firstMismatch.actualCandidateTexts,
                    Collections.emptyList(),
                    Collections.emptyList(),
                    forcedPrefixCount,
                    firstMismatch.state == null ? fallbackState : firstMismatch.state,
                    firstMismatch.detail,
                    sourceExpectation);
        }

        void recordChoiceMismatch(OpponentDecision expected, Game game,
                                  List<String> actualCandidateTexts, String detail) {
            recordMismatch(expected, game, "choice", actualCandidateTexts,
                    "opponent_transcript_choice_mismatch", detail);
        }

        void recordSkippedBlockerMismatch(OpponentDecision expected, Game game,
                                          List<String> actualCandidateTexts, String detail) {
            recordMismatch(expected, game, "declare_blockers", actualCandidateTexts,
                    "missing_blocker_transcript", detail);
        }

        private void recordMismatch(OpponentDecision expected, Game game, String actualEvent,
                                    List<String> actualCandidateTexts, String reason, String detail) {
            try {
                Player player = null;
                if (game != null) {
                    UUID priority = game.getPriorityPlayerId();
                    player = priority == null ? null : game.getPlayer(priority);
                    if (player == null) {
                        UUID active = game.getActivePlayerId();
                        player = active == null ? null : game.getPlayer(active);
                    }
                }
                StateSnapshot state = StateSnapshot.capture(game, player);
                String actualPhase = game == null || game.getTurnStepType() == null
                        ? ""
                        : String.valueOf(game.getTurnStepType());
                if (firstMismatch == null) {
                    firstMismatch = new OpponentTranscriptMismatch(
                            agentOrdinalContext,
                            reason,
                            "opponent_decision_index=" + (expected == null ? -1 : expected.decisionIndex)
                                    + " expected_event=" + (expected == null ? "" : expected.event)
                                    + " actual_event=" + actualEvent
                                    + " expected_source_turn=" + (expected == null ? -1 : expected.sourceTurn)
                                    + " actual_source_turn=" + compactSourceTurn(safeTurn(game))
                                    + " expected_phase=" + (expected == null ? "" : expected.phase)
                                    + " actual_phase=" + actualPhase
                                    + " expected_action=" + (expected == null ? "" : expected.chosenActionText)
                                    + " actual_candidates=" + indexedTexts(actualCandidateTexts)
                                    + " detail=" + detail,
                            actualCandidateTexts,
                            state);
                }
                if (mismatchRecorded || mismatchPath == null) {
                    return;
                }
                mismatchRecorded = true;
                String row = scenario
                        + "," + seed
                        + "," + (expected == null ? -1 : expected.decisionIndex)
                        + "," + csv(reason)
                        + "," + csv(expected == null ? "" : expected.event)
                        + "," + csv(actualEvent)
                        + "," + (expected == null ? -1 : expected.sourceTurn)
                        + "," + compactSourceTurn(safeTurn(game))
                        + "," + csv(expected == null ? "" : expected.phase)
                        + "," + csv(actualPhase)
                        + "," + csv(expected == null ? "" : expected.chosenActionText)
                        + "," + csv(indexedTexts(actualCandidateTexts))
                        + "," + csv(detail)
                        + "," + csv(state.toCompactText())
                        + System.lineSeparator();
                Files.write(mismatchPath, row.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.APPEND);
            } catch (Exception ignored) {
                // Diagnostics must not affect replay control flow.
            }
        }

        private boolean eventMatches(String expected, String actual) {
            return normalizeEvent(expected).equals(normalizeEvent(actual));
        }

        private boolean phaseMatches(String expected, String actual) {
            String lhs = normalizeEvent(expected);
            String rhs = normalizeEvent(actual);
            return lhs.isEmpty() || rhs.isEmpty() || lhs.equals(rhs);
        }

        private String normalizeEvent(String value) {
            if (value == null) {
                return "";
            }
            return value.trim().replaceAll("[^A-Za-z0-9]+", "_")
                    .replaceAll("_+", "_")
                    .replaceAll("^_|_$", "")
                    .toUpperCase(Locale.ROOT);
        }
    }

    private static final class OpponentTranscriptMismatch {

        final int agentOrdinalContext;
        final String reason;
        final String detail;
        final List<String> actualCandidateTexts;
        final StateSnapshot state;

        private OpponentTranscriptMismatch(int agentOrdinalContext, String reason, String detail,
                                           List<String> actualCandidateTexts, StateSnapshot state) {
            this.agentOrdinalContext = agentOrdinalContext;
            this.reason = reason == null ? "" : reason;
            this.detail = detail == null ? "" : detail;
            this.actualCandidateTexts = actualCandidateTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actualCandidateTexts);
            this.state = state == null ? StateSnapshot.EMPTY : state;
        }
    }

    private static void appendJsonNumber(StringBuilder sb, String key, long value) {
        appendJsonString(sb, key, null);
        sb.append(value);
    }

    private static void appendJsonBoolean(StringBuilder sb, String key, boolean value) {
        appendJsonString(sb, key, null);
        sb.append(value);
    }

    private static void appendJsonString(StringBuilder sb, String key, String value) {
        sb.append('"').append(escapeJson(key)).append('"').append(':');
        if (value == null) {
            return;
        }
        sb.append('"').append(escapeJson(value)).append('"');
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder out = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
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
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                    break;
            }
        }
        return out.toString();
    }

    private static class StackResolvingComputerPlayer7 extends ComputerPlayer7 {
        private int stackPriorityTraceIndex = 0;
        private int foodTokenMetadataParityBurns = 0;
        private int heroTokenMetadataParityBurns = 0;

        private StackResolvingComputerPlayer7(String name, RangeOfInfluence range, int skill) {
            super(name, range, skill);
        }

        private StackResolvingComputerPlayer7(StackResolvingComputerPlayer7 player) {
            super(player);
            this.stackPriorityTraceIndex = player.stackPriorityTraceIndex;
            this.foodTokenMetadataParityBurns = player.foodTokenMetadataParityBurns;
            this.heroTokenMetadataParityBurns = player.heroTokenMetadataParityBurns;
        }

        @Override
        public StackResolvingComputerPlayer7 copy() {
            return new StackResolvingComputerPlayer7(this);
        }

        @Override
        public boolean priority(Game game) {
            int traceIndex = ++stackPriorityTraceIndex;
            long before = RandomUtil.getConsumptionCount();
            long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
            long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
            if (game != null && game.getStack() != null && !game.getStack().isEmpty()) {
                consumeSourceCompatibleTokenMetadata(game);
                pass(game);
                appendStackPriorityTrace(game, this, "stack_priority_pass", true, false,
                        before,
                        RandomUtil.getConsumptionCount(),
                        directBefore,
                        RandomUtil.getDirectGetRandomConsumptionCount(),
                        directAccessBefore,
                        RandomUtil.getDirectGetRandomAccessCount(),
                        traceIndex);
                return false;
            }
            boolean result = super.priority(game);
            appendStackPriorityTrace(game, this, "cp7_priority", false, result,
                    before,
                    RandomUtil.getConsumptionCount(),
                    directBefore,
                    RandomUtil.getDirectGetRandomConsumptionCount(),
                    directAccessBefore,
                    RandomUtil.getDirectGetRandomAccessCount(),
                    traceIndex);
            return result;
        }

        protected void consumeSourceCompatibleTokenMetadata(Game game) {
            if (!CP7_TOKEN_METADATA_PARITY_ENABLED || game == null || game.getStack() == null
                    || game.getStack().isEmpty()) {
                return;
            }
            String stackTop = traceStackTop(game);
            if (stackTop == null || stackTop.isEmpty()) {
                return;
            }
            String normalized = stackTop.toLowerCase(Locale.ENGLISH);
            if (foodTokenMetadataParityBurns == 0 && normalized.contains("food token")) {
                TokenRepository.instance.consumePreferredTokenInfoForReplayParity(
                        "mage.game.permanent.token.FoodToken", "LTR", 5);
                foodTokenMetadataParityBurns += 5;
                return;
            }
            if (heroTokenMetadataParityBurns == 0 && normalized.contains("hero creature token")) {
                TokenRepository.instance.consumePreferredTokenInfoForReplayParity(
                        "mage.game.permanent.token.HeroToken", "FIN", 10);
                heroTokenMetadataParityBurns += 10;
            }
        }
    }

    private static final class TranscriptReplayingComputerPlayer7 extends StackResolvingComputerPlayer7 {

        private static final int STRICT_TRANSCRIPT_REPLAY_MIN_AGENT_ORDINAL = 26;

        private final OpponentTranscriptCursor transcript;
        private List<String> pendingTranscriptSacrificeNames = Collections.emptyList();
        private OpponentDecision pendingTranscriptSacrificeDecision = null;
        private List<String> pendingTranscriptTargetNames = Collections.emptyList();
        private OpponentDecision pendingTranscriptTargetDecision = null;

        private TranscriptReplayingComputerPlayer7(String name, RangeOfInfluence range, int skill,
                                                   OpponentTranscriptCursor transcript) {
            super(name, range, skill);
            this.transcript = transcript == null ? OpponentTranscriptCursor.empty() : transcript;
        }

        private TranscriptReplayingComputerPlayer7(TranscriptReplayingComputerPlayer7 player) {
            super(player);
            this.transcript = player.transcript.copy();
            this.pendingTranscriptSacrificeNames = new ArrayList<>(player.pendingTranscriptSacrificeNames);
            this.pendingTranscriptSacrificeDecision = player.pendingTranscriptSacrificeDecision;
            this.pendingTranscriptTargetNames = new ArrayList<>(player.pendingTranscriptTargetNames);
            this.pendingTranscriptTargetDecision = player.pendingTranscriptTargetDecision;
        }

        @Override
        public TranscriptReplayingComputerPlayer7 copy() {
            return new TranscriptReplayingComputerPlayer7(this);
        }

        @Override
        public boolean priority(Game game) {
            if (game != null && !game.isSimulation() && game.getStack() != null && !game.getStack().isEmpty()) {
                OpponentDecision expected = transcript.consumeForContext(
                        "PRIORITY", game, Collections.singletonList("Pass"), "stack_priority");
                if (expected != null) {
                    if (isPassText(expected.chosenActionText)) {
                        long before = RandomUtil.getConsumptionCount();
                        long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
                        long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
                        consumeSourceCompatibleTokenMetadata(game);
                        pass(game);
                        appendStackPriorityTrace(game, this, "transcript_stack_priority_pass", true, false,
                                before,
                                RandomUtil.getConsumptionCount(),
                                directBefore,
                                RandomUtil.getDirectGetRandomConsumptionCount(),
                                directAccessBefore,
                                RandomUtil.getDirectGetRandomAccessCount(),
                                expected.decisionIndex);
                        return false;
                    }
                    List<Ability> playable = transcriptPlayableActions(game);
                    List<String> playableTexts = playable.stream()
                            .map(ability -> safeAbilityTextForTranscript(game, ability))
                            .collect(Collectors.toList());
                    int playableForced = findTranscriptCandidate(expected.chosenActionText, playableTexts);
                    if (playableForced >= 0 && playableForced < playable.size()) {
                        long before = RandomUtil.getConsumptionCount();
                        long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
                        long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
                        consumeSourceCompatibleTokenMetadata(game);
                        actions.clear();
                        actions.add(playable.get(playableForced));
                        executeForcedTranscriptAction(game, expected);
                        appendStackPriorityTrace(game, this, "transcript_stack_priority_action", true, true,
                                before,
                                RandomUtil.getConsumptionCount(),
                                directBefore,
                                RandomUtil.getDirectGetRandomConsumptionCount(),
                                directAccessBefore,
                                RandomUtil.getDirectGetRandomAccessCount(),
                                expected.decisionIndex);
                        return true;
                    }
                    TranscriptTargetSpec deferredTarget = inlineTranscriptTargetSpec(expected.chosenActionText);
                    int deferredForced = findTranscriptCandidateWithDeferredTarget(deferredTarget, playableTexts);
                    if (deferredForced >= 0 && deferredForced < playable.size()) {
                        long before = RandomUtil.getConsumptionCount();
                        long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
                        long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
                        consumeSourceCompatibleTokenMetadata(game);
                        actions.clear();
                        actions.add(playable.get(deferredForced));
                        executeForcedTranscriptAction(game, expected, deferredTarget.targetNames);
                        appendStackPriorityTrace(game, this, "transcript_stack_priority_deferred_target_action",
                                true, true,
                                before,
                                RandomUtil.getConsumptionCount(),
                                directBefore,
                                RandomUtil.getDirectGetRandomConsumptionCount(),
                                directAccessBefore,
                                RandomUtil.getDirectGetRandomAccessCount(),
                                expected.decisionIndex);
                        return true;
                    }
                    transcript.recordChoiceMismatch(expected, game,
                            transcriptCandidateDebug(Collections.singletonList("Pass"), playableTexts),
                            "opponent stack-priority transcript action was not present in legal playable candidates");
                }
            }
            if (maybeForceBoundedD116OpponentPrecombatAction(game)) {
                return true;
            }
            if (maybePassBoundedD119SourceSkippedPreAttackPriority(game)) {
                return false;
            }
            return super.priority(game);
        }

        private boolean maybeForceBoundedD116OpponentPrecombatAction(Game game) {
            OpponentDecision expected = transcript.peek();
            if (!boundedD116OpponentPrecombatContext(expected, game)) {
                return false;
            }
            List<Ability> playable = transcriptPlayableActions(game);
            List<String> playableTexts = playable.stream()
                    .map(ability -> safeAbilityTextForTranscript(game, ability))
                    .collect(Collectors.toList());
            int forced = findTranscriptCandidate(expected.chosenActionText, playableTexts);
            List<String> pendingTargets = Collections.emptyList();
            if (forced < 0 || forced >= playable.size()) {
                TranscriptTargetSpec deferredTarget = inlineTranscriptTargetSpec(expected.chosenActionText);
                forced = findTranscriptCandidateWithDeferredTarget(deferredTarget, playableTexts);
                if (forced >= 0 && forced < playable.size()) {
                    pendingTargets = deferredTarget.targetNames;
                }
            }
            if (forced < 0 || forced >= playable.size()) {
                transcript.recordChoiceMismatch(expected, game,
                        transcriptCandidateDebug(Collections.emptyList(), playableTexts),
                        "bounded D116 opponent precombat transcript action was not legal before CP7 pass/attack advance");
                return false;
            }
            OpponentDecision consumed = transcript.consumeForContext(
                    "PRIORITY", game, playableTexts, "priority_action");
            if (consumed == null) {
                return false;
            }
            long before = RandomUtil.getConsumptionCount();
            long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
            long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
            consumeSourceCompatibleTokenMetadata(game);
            actions.clear();
            actions.add(playable.get(forced));
            executeForcedTranscriptAction(game, consumed, pendingTargets);
            appendStackPriorityTrace(game, this, "transcript_d116_precombat_action", false, true,
                    before,
                    RandomUtil.getConsumptionCount(),
                    directBefore,
                    RandomUtil.getDirectGetRandomConsumptionCount(),
                    directAccessBefore,
                    RandomUtil.getDirectGetRandomAccessCount(),
                    consumed.decisionIndex);
            return true;
        }

        private boolean boundedD116OpponentPrecombatContext(OpponentDecision expected, Game game) {
            if (expected == null || game == null || game.isSimulation()) {
                return false;
            }
            if (expected.scenario != 1 || expected.seed != 204727199L
                    || expected.sourceTurn != 8
                    || expected.decisionIndex < 106 || expected.decisionIndex > 108
                    || !"PRIORITY".equals(contextKey(expected.event))
                    || isPassText(expected.chosenActionText)
                    || !"PRECOMBAT_MAIN".equals(sourceStepKey(expected.phase))) {
                return false;
            }
            if (game.getStack() != null && !game.getStack().isEmpty()) {
                return false;
            }
            if (game.getActivePlayerId() == null || !game.getActivePlayerId().equals(getId())) {
                return false;
            }
            if (compactSourceTurn(safeTurn(game)) != expected.sourceTurn) {
                return false;
            }
            String actualStep = game.getTurnStepType() == null ? "" : String.valueOf(game.getTurnStepType());
            return "PRECOMBAT_MAIN".equals(sourceStepKey(actualStep));
        }

        private boolean maybePassBoundedD119SourceSkippedPreAttackPriority(Game game) {
            OpponentDecision expected = transcript.peek();
            if (!boundedD119SourceSkippedPreAttackPassContext(expected, game)) {
                return false;
            }
            List<Ability> playable = transcriptPlayableActions(game);
            List<String> playableTexts = playable.stream()
                    .map(ability -> safeAbilityTextForTranscript(game, ability))
                    .collect(Collectors.toList());
            for (String text : playableTexts) {
                if (!isPassText(text)) {
                    return false;
                }
            }
            long before = RandomUtil.getConsumptionCount();
            long directBefore = RandomUtil.getDirectGetRandomConsumptionCount();
            long directAccessBefore = RandomUtil.getDirectGetRandomAccessCount();
            pass(game);
            appendStackPriorityTrace(game, this, "transcript_d119_pre_attack_source_skipped_pass", false, false,
                    before,
                    RandomUtil.getConsumptionCount(),
                    directBefore,
                    RandomUtil.getDirectGetRandomConsumptionCount(),
                    directAccessBefore,
                    RandomUtil.getDirectGetRandomAccessCount(),
                    expected.decisionIndex);
            return true;
        }

        private boolean boundedD119SourceSkippedPreAttackPassContext(OpponentDecision expected, Game game) {
            if (expected == null || game == null || game.isSimulation()) {
                return false;
            }
            if (expected.scenario != 1 || expected.seed != 204727199L
                    || expected.sourceTurn != 8
                    || expected.decisionIndex != 109
                    || !"DECLARE_ATTACKERS".equals(contextKey(expected.event))
                    || !"DECLARE_ATTACKERS".equals(sourceStepKey(expected.phase))) {
                return false;
            }
            if (game.getStack() != null && !game.getStack().isEmpty()) {
                return false;
            }
            if (game.getActivePlayerId() == null || !game.getActivePlayerId().equals(getId())) {
                return false;
            }
            if (game.getPriorityPlayerId() == null || !game.getPriorityPlayerId().equals(getId())) {
                return false;
            }
            if (compactSourceTurn(safeTurn(game)) != expected.sourceTurn) {
                return false;
            }
            String actualStep = game.getTurnStepType() == null ? "" : String.valueOf(game.getTurnStepType());
            return "PRECOMBAT_MAIN".equals(sourceStepKey(actualStep));
        }

        @Override
        protected void act(Game game) {
            if (game == null || game.isSimulation()) {
                super.act(game);
                return;
            }
            List<Ability> planned = new ArrayList<>(actions);
            List<String> actionTexts = planned.stream()
                    .map(ability -> safeAbilityTextForTranscript(game, ability))
                    .collect(Collectors.toList());
            OpponentDecision expected = transcript.consumeForContext(
                    "PRIORITY", game, actionTexts, "priority_action");
            if (expected == null) {
                if (strictTranscriptReplay(game)) {
                    passForTranscriptGap(game);
                    return;
                }
                super.act(game);
                return;
            }
            if (isPassText(expected.chosenActionText)) {
                actions.clear();
                pass(game);
                return;
            }
            int forced = findTranscriptCandidate(expected.chosenActionText, actionTexts);
            if (forced >= 0 && forced < planned.size()) {
                actions.clear();
                actions.add(planned.get(forced));
                executeForcedTranscriptAction(game, expected);
                return;
            }
            TranscriptTargetSpec deferredTarget = inlineTranscriptTargetSpec(expected.chosenActionText);
            int deferredForced = findTranscriptCandidateWithDeferredTarget(deferredTarget, actionTexts);
            if (deferredForced >= 0 && deferredForced < planned.size()) {
                actions.clear();
                actions.add(planned.get(deferredForced));
                executeForcedTranscriptAction(game, expected, deferredTarget.targetNames);
                return;
            }
            List<Ability> playable = transcriptPlayableActions(game);
            List<String> playableTexts = playable.stream()
                    .map(ability -> safeAbilityTextForTranscript(game, ability))
                    .collect(Collectors.toList());
            int playableForced = findTranscriptCandidate(expected.chosenActionText, playableTexts);
            if (playableForced >= 0 && playableForced < playable.size()) {
                actions.clear();
                actions.add(playable.get(playableForced));
                executeForcedTranscriptAction(game, expected);
                return;
            }
            int playableDeferredForced = findTranscriptCandidateWithDeferredTarget(deferredTarget, playableTexts);
            if (playableDeferredForced >= 0 && playableDeferredForced < playable.size()) {
                actions.clear();
                actions.add(playable.get(playableDeferredForced));
                executeForcedTranscriptAction(game, expected, deferredTarget.targetNames);
                return;
            }
            transcript.recordChoiceMismatch(expected, game, transcriptCandidateDebug(actionTexts, playableTexts),
                    "opponent priority transcript action was not present in CP7 planned or legal playable candidates");
            if (strictTranscriptReplay(game)) {
                passForTranscriptGap(game);
                return;
            }
            super.act(game);
        }

        @Override
        public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
            Boolean forced = forcePendingTranscriptSacrificeChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            forced = forcePendingTranscriptTargetChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            return super.choose(outcome, target, source, game);
        }

        @Override
        public boolean choose(Outcome outcome, Target target, Ability source, Game game,
                              Map<String, Serializable> options) {
            Boolean forced = forcePendingTranscriptSacrificeChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            forced = forcePendingTranscriptTargetChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            return super.choose(outcome, target, source, game, options);
        }

        @Override
        public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
            Boolean forced = forcePendingTranscriptSacrificeChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            forced = forcePendingTranscriptTargetChoice(outcome, target, source, game);
            if (forced != null) {
                return forced;
            }
            return super.chooseTarget(outcome, target, source, game);
        }

        @Override
        public void selectAttackers(Game game, UUID attackingPlayerId) {
            if (game == null || game.isSimulation()) {
                super.selectAttackers(game, attackingPlayerId);
                return;
            }
            List<String> candidateNames = controlledCreatureNames(game, attackingPlayerId);
            OpponentDecision expected = transcript.consumeForContext(
                    "DECLARE_ATTACKERS", game, candidateNames, "declare_attackers");
            if (expected == null) {
                super.selectAttackers(game, attackingPlayerId);
                return;
            }
            List<String> attackers = parseCombatNames(expected.chosenActionText, "Declare attackers:");
            if (attackers.isEmpty()) {
                return;
            }
            UUID defenderId = firstOpponentId(game);
            Player attackingPlayer = game.getPlayer(attackingPlayerId);
            if (defenderId == null || attackingPlayer == null) {
                transcript.recordChoiceMismatch(expected, game, candidateNames,
                        "could not resolve attacking player or defender for opponent transcript");
                super.selectAttackers(game, attackingPlayerId);
                return;
            }
            int declared = 0;
            List<String> remaining = new ArrayList<>(attackers);
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(attackingPlayerId)) {
                if (permanent == null || !permanent.isCreature(game)) {
                    continue;
                }
                int idx = findNameIndex(remaining, permanent.getName());
                if (idx < 0) {
                    continue;
                }
                attackingPlayer.declareAttacker(permanent.getId(), defenderId, game, true);
                remaining.remove(idx);
                declared++;
            }
            if (!remaining.isEmpty() || declared == 0) {
                transcript.recordChoiceMismatch(expected, game, candidateNames,
                        "could not declare every attacker named by opponent transcript: " + joinTexts(remaining));
            }
        }

        @Override
        public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
            if (game == null || game.isSimulation()) {
                super.selectBlockers(source, game, defendingPlayerId);
                return;
            }
            List<Permanent> legalBlockers = legalBlockerPermanents(game, defendingPlayerId);
            List<String> candidateNames = blockerCandidateNames(game, defendingPlayerId, legalBlockers);
            OpponentDecision next = transcript.peek();
            if (sourceSkippedNoBlockerPrompt(next, game)) {
                if (onlyNoBlockerCandidates(candidateNames, legalBlockers)) {
                    return;
                }
                String trace = combatTrace(game, this, "skipped_blocker_mismatch", -1, null);
                transcript.recordSkippedBlockerMismatch(next, game, candidateNames,
                        "source transcript advanced past declare blockers but replay has legal blocker candidates "
                                + "combat_trace=" + trace);
                super.selectBlockers(source, game, defendingPlayerId);
                return;
            }
            OpponentDecision expected = transcript.consumeForContext(
                    "DECLARE_BLOCKERS", game, candidateNames, "declare_blockers");
            if (expected == null) {
                super.selectBlockers(source, game, defendingPlayerId);
                return;
            }
            List<String> blockers = parseCombatNames(expected.chosenActionText, "Declare blockers:");
            if (blockers.isEmpty()) {
                return;
            }
            transcript.recordChoiceMismatch(expected, game, candidateNames,
                    "non-empty blocker transcript forcing is not implemented in this validation hook");
            super.selectBlockers(source, game, defendingPlayerId);
        }

        private boolean strictTranscriptReplay(Game game) {
            return game != null
                    && !game.isSimulation()
                    && !transcript.isEmpty()
                    && transcript.agentOrdinalAtLeast(STRICT_TRANSCRIPT_REPLAY_MIN_AGENT_ORDINAL)
                    && compactSourceTurn(safeTurn(game)) >= 4;
        }

        private void passForTranscriptGap(Game game) {
            actions.clear();
            pass(game);
        }

        private boolean sourceSkippedNoBlockerPrompt(OpponentDecision next, Game game) {
            if (next == null || game == null || !isReplayDeclareBlockersStep(game)) {
                return false;
            }
            int actualSourceTurn = compactSourceTurn(safeTurn(game));
            if (next.sourceTurn >= 0 && next.sourceTurn != actualSourceTurn) {
                return false;
            }
            if (!isPassText(next.chosenActionText)) {
                return false;
            }
            String nextStep = sourceStepKey(next.phase);
            return "COMBAT_DAMAGE".equals(nextStep)
                    || "END_COMBAT".equals(nextStep)
                    || "POSTCOMBAT_MAIN".equals(nextStep);
        }

        private boolean isReplayDeclareBlockersStep(Game game) {
            if (game == null || game.getTurnStepType() == null) {
                return false;
            }
            String step = sourceStepKey(String.valueOf(game.getTurnStepType()));
            if ("DECLARE_BLOCKERS".equals(step)) {
                return true;
            }
            String key = contextKey(String.valueOf(game.getTurnStepType()));
            return key.contains("DECLARE_BLOCKERS") || key.contains("DECLARE_BLOCKS");
        }

        private List<Permanent> legalBlockerPermanents(Game game, UUID defendingPlayerId) {
            if (game == null || defendingPlayerId == null || game.getCombat() == null) {
                return Collections.emptyList();
            }
            List<Permanent> blockers;
            try {
                blockers = new ArrayList<>(getAvailableBlockers(game));
            } catch (Exception e) {
                blockers = Collections.emptyList();
            }
            if (blockers.isEmpty()) {
                return Collections.emptyList();
            }
            List<Permanent> legal = new ArrayList<>();
            for (CombatGroup group : game.getCombat().getGroups()) {
                for (UUID attackerId : group.getAttackers()) {
                    for (Permanent blocker : blockers) {
                        if (blocker != null
                                && defendingPlayerId.equals(blocker.getControllerId())
                                && blocker.isCreature(game)
                                && blocker.canBlock(attackerId, game)
                                && !legal.contains(blocker)) {
                            legal.add(blocker);
                        }
                    }
                }
            }
            return legal;
        }

        private List<String> blockerCandidateNames(Game game, UUID defendingPlayerId, List<Permanent> legalBlockers) {
            List<String> names = controlledCreatureNames(game, defendingPlayerId);
            if (names.isEmpty() && legalBlockers != null && !legalBlockers.isEmpty()) {
                for (Permanent blocker : legalBlockers) {
                    if (blocker != null) {
                        names.add(blocker.getName());
                    }
                }
            }
            if (names.isEmpty()) {
                names.add("DECLARE_BLOCKERS: NONE");
            }
            return names;
        }

        private boolean onlyNoBlockerCandidates(List<String> candidateNames, List<Permanent> legalBlockers) {
            if (legalBlockers == null || legalBlockers.isEmpty()) {
                return true;
            }
            if (candidateNames == null || candidateNames.isEmpty()) {
                return false;
            }
            for (String candidate : candidateNames) {
                if (!isNoBlockerCandidate(candidate)) {
                    return false;
                }
            }
            return true;
        }

        private boolean isNoBlockerCandidate(String text) {
            String normalized = text == null
                    ? ""
                    : text.trim().replaceAll("[^A-Za-z0-9]+", "_")
                    .replaceAll("_+", "_")
                    .replaceAll("^_|_$", "")
                    .toUpperCase(Locale.ROOT);
            return normalized.isEmpty()
                    || "PASS".equals(normalized)
                    || "DONE".equals(normalized)
                    || "NONE".equals(normalized)
                    || "NO_BLOCK".equals(normalized)
                    || "NO_BLOCKS".equals(normalized)
                    || "NO_BLOCKERS".equals(normalized)
                    || "DECLARE_BLOCKERS_NONE".equals(normalized);
        }

        private String safeAbilityTextForTranscript(Game game, Ability ability) {
            try {
                return getAbilityAndSourceInfo(game, ability, true);
            } catch (Exception e) {
                return String.valueOf(ability);
            }
        }

        private List<String> controlledCreatureNames(Game game, UUID controllerId) {
            if (game == null || controllerId == null || game.getBattlefield() == null) {
                return Collections.emptyList();
            }
            List<String> names = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(controllerId)) {
                if (permanent != null && permanent.isCreature(game)) {
                    names.add(permanent.getName());
                }
            }
            return names;
        }

        private UUID firstOpponentId(Game game) {
            if (game == null) {
                return null;
            }
            for (UUID opponentId : game.getOpponents(getId())) {
                return opponentId;
            }
            return null;
        }

        private List<String> parseCombatNames(String text, String prefix) {
            String value = text == null ? "" : text.trim();
            if (!prefix.isEmpty() && value.toLowerCase(Locale.ROOT).startsWith(prefix.toLowerCase(Locale.ROOT))) {
                value = value.substring(prefix.length()).trim();
            }
            if (value.isEmpty() || "NONE".equalsIgnoreCase(value)) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (String part : value.split("\\|")) {
                String name = part.trim();
                if (!name.isEmpty() && !"NONE".equalsIgnoreCase(name)) {
                    out.add(name);
                }
            }
            return out;
        }

        private void executeForcedTranscriptAction(Game game, OpponentDecision expected) {
            executeForcedTranscriptAction(game, expected, Collections.emptyList());
        }

        private void executeForcedTranscriptAction(Game game, OpponentDecision expected,
                                                   List<String> pendingTargetNames) {
            List<String> previousSacrificeNames = pendingTranscriptSacrificeNames;
            OpponentDecision previousSacrificeDecision = pendingTranscriptSacrificeDecision;
            List<String> previousTargetNames = pendingTranscriptTargetNames;
            OpponentDecision previousTargetDecision = pendingTranscriptTargetDecision;
            pendingTranscriptSacrificeNames = inferSourceSacrificeNames(expected, transcript.peek());
            pendingTranscriptSacrificeDecision = expected;
            pendingTranscriptTargetNames = pendingTargetNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(pendingTargetNames);
            pendingTranscriptTargetDecision = expected;
            try {
                super.act(game);
            } finally {
                pendingTranscriptSacrificeNames = previousSacrificeNames;
                pendingTranscriptSacrificeDecision = previousSacrificeDecision;
                pendingTranscriptTargetNames = previousTargetNames;
                pendingTranscriptTargetDecision = previousTargetDecision;
            }
        }

        private Boolean forcePendingTranscriptSacrificeChoice(Outcome outcome, Target target,
                                                              Ability source, Game game) {
            if (game == null || game.isSimulation() || target == null
                    || pendingTranscriptSacrificeNames == null || pendingTranscriptSacrificeNames.isEmpty()
                    || outcome != Outcome.Sacrifice) {
                return null;
            }
            UUID controllerId = target.getAffectedAbilityControllerId(getId());
            if (controllerId == null) {
                controllerId = getId();
            }
            Set<UUID> possibleIds;
            try {
                possibleIds = target.possibleTargets(controllerId, source, game);
            } catch (Exception e) {
                possibleIds = Collections.emptySet();
            }
            List<String> candidates = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents(controllerId)) {
                if (permanent == null || !possibleIds.contains(permanent.getId())) {
                    continue;
                }
                candidates.add(permanent.getName());
            }
            for (String desiredName : pendingTranscriptSacrificeNames) {
                for (Permanent permanent : game.getBattlefield().getAllActivePermanents(controllerId)) {
                    if (permanent == null || !possibleIds.contains(permanent.getId())) {
                        continue;
                    }
                    if (!sameVisiblePermanentName(desiredName, permanent.getName())) {
                        continue;
                    }
                    target.add(permanent.getId(), game);
                    return target.isChosen(game) || target.contains(permanent.getId());
                }
            }
            transcript.recordChoiceMismatch(pendingTranscriptSacrificeDecision, game, candidates,
                    "source-inferred sacrifice choice was not legal: "
                            + joinTexts(pendingTranscriptSacrificeNames));
            return Boolean.FALSE;
        }

        private Boolean forcePendingTranscriptTargetChoice(Outcome outcome, Target target,
                                                           Ability source, Game game) {
            if (game == null || game.isSimulation() || target == null
                    || pendingTranscriptTargetNames == null || pendingTranscriptTargetNames.isEmpty()) {
                return null;
            }
            UUID controllerId = target.getAffectedAbilityControllerId(getId());
            if (controllerId == null) {
                controllerId = getId();
            }
            Set<UUID> possibleIds;
            try {
                possibleIds = target.possibleTargets(controllerId, source, game);
            } catch (Exception e) {
                possibleIds = Collections.emptySet();
            }
            List<UUID> matches = new ArrayList<>();
            List<String> candidates = new ArrayList<>();
            for (UUID id : possibleIds) {
                String candidateName = transcriptTargetName(id, game);
                if (candidateName.isEmpty()) {
                    continue;
                }
                candidates.add(candidateName);
                for (String desiredName : pendingTranscriptTargetNames) {
                    if (sameTranscriptTargetName(desiredName, candidateName)) {
                        matches.add(id);
                        break;
                    }
                }
            }
            if (matches.size() == 1) {
                target.add(matches.get(0), game);
                return target.isChosen(game) || target.contains(matches.get(0));
            }
            String detail = matches.isEmpty()
                    ? "source inline target choice was not legal: "
                    : "source inline target choice was ambiguous: ";
            transcript.recordChoiceMismatch(pendingTranscriptTargetDecision, game, candidates,
                    detail + joinTexts(pendingTranscriptTargetNames));
            return Boolean.FALSE;
        }

        private List<String> inferSourceSacrificeNames(OpponentDecision before, OpponentDecision after) {
            if (before == null || after == null
                    || before.chosenActionText == null
                    || !before.chosenActionText.toLowerCase(Locale.ROOT).contains("sacrifice")) {
                return Collections.emptyList();
            }
            Map<String, Integer> beforeBattlefield = visibleZoneCounts(before.visibleState, "selfBattlefield");
            Map<String, Integer> afterBattlefield = visibleZoneCounts(after.visibleState, "selfBattlefield");
            Map<String, Integer> beforeGraveyard = visibleZoneCounts(before.visibleState, "selfGraveyard");
            Map<String, Integer> afterGraveyard = visibleZoneCounts(after.visibleState, "selfGraveyard");
            List<String> out = new ArrayList<>();
            for (Map.Entry<String, Integer> entry : beforeBattlefield.entrySet()) {
                String name = entry.getKey();
                int removed = entry.getValue() - countOf(afterBattlefield, name);
                int graveyardAdded = countOf(afterGraveyard, name) - countOf(beforeGraveyard, name);
                for (int i = 0; i < Math.min(removed, graveyardAdded); i++) {
                    out.add(name);
                }
            }
            return out;
        }

        private Map<String, Integer> visibleZoneCounts(String visibleState, String key) {
            Map<String, Integer> counts = new LinkedHashMap<>();
            if (visibleState == null || visibleState.isEmpty() || key == null || key.isEmpty()) {
                return counts;
            }
            String prefix = key + "=";
            for (String part : visibleState.split(";")) {
                String trimmed = part == null ? "" : part.trim();
                if (!trimmed.startsWith(prefix)) {
                    continue;
                }
                String value = trimmed.substring(prefix.length());
                if (value.isEmpty()) {
                    return counts;
                }
                for (String rawName : value.split("\\|")) {
                    String name = normalizeVisiblePermanentName(rawName);
                    if (!name.isEmpty()) {
                        counts.put(name, countOf(counts, name) + 1);
                    }
                }
                return counts;
            }
            return counts;
        }

        private int countOf(Map<String, Integer> counts, String name) {
            Integer value = counts == null ? null : counts.get(name);
            return value == null ? 0 : value;
        }

        private String normalizeVisiblePermanentName(String value) {
            String name = value == null ? "" : value.trim();
            int marker = name.indexOf('[');
            if (marker >= 0) {
                name = name.substring(0, marker).trim();
            }
            return name;
        }

        private boolean sameVisiblePermanentName(String expected, String actual) {
            String lhs = normalizeVisiblePermanentName(expected);
            String rhs = normalizeVisiblePermanentName(actual);
            return !lhs.isEmpty()
                    && (lhs.equalsIgnoreCase(rhs)
                    || textMatchesExpected(lhs, rhs)
                    || textMatchesExpected(rhs, lhs));
        }

        private String transcriptTargetName(UUID id, Game game) {
            if (id == null || game == null) {
                return "";
            }
            Permanent permanent = game.getPermanent(id);
            if (permanent != null) {
                return permanent.getName();
            }
            Card card = game.getCard(id);
            if (card != null) {
                return card.getName();
            }
            MageObject object = game.getObject(id);
            if (object != null) {
                return object.getName();
            }
            Player player = game.getPlayer(id);
            return player == null ? "" : player.getName();
        }

        private boolean sameTranscriptTargetName(String expected, String actual) {
            return sameVisiblePermanentName(expected, actual);
        }

        private int findNameIndex(List<String> names, String target) {
            for (int i = 0; i < names.size(); i++) {
                if (textMatchesExpected(names.get(i), target)) {
                    return i;
                }
            }
            return -1;
        }

        private int findTranscriptCandidate(String expectedText, List<String> actualTexts) {
            if (actualTexts == null || actualTexts.isEmpty()) {
                return -1;
            }
            for (int i = 0; i < actualTexts.size(); i++) {
                if (textMatchesExpected(expectedText, actualTexts.get(i))) {
                    return i;
                }
            }
            return -1;
        }

        private int findTranscriptCandidateWithDeferredTarget(TranscriptTargetSpec expected,
                                                              List<String> actualTexts) {
            if (expected == null || expected.targetNames.isEmpty()
                    || actualTexts == null || actualTexts.isEmpty()) {
                return -1;
            }
            int match = -1;
            for (int i = 0; i < actualTexts.size(); i++) {
                if (!textMatchesExpected(expected.actionText, actualTexts.get(i))) {
                    continue;
                }
                if (match >= 0) {
                    return -1;
                }
                match = i;
            }
            return match;
        }

        private TranscriptTargetSpec inlineTranscriptTargetSpec(String expectedText) {
            String text = expectedText == null ? "" : expectedText.trim();
            int arrow = text.lastIndexOf("->");
            if (arrow < 0) {
                return null;
            }
            String actionText = text.substring(0, arrow).trim();
            String targetText = text.substring(arrow + 2).trim();
            if (actionText.isEmpty() || targetText.isEmpty()) {
                return null;
            }
            List<String> targets = Collections.singletonList(targetText);
            return new TranscriptTargetSpec(actionText, targets);
        }

        private static final class TranscriptTargetSpec {
            final String actionText;
            final List<String> targetNames;

            private TranscriptTargetSpec(String actionText, List<String> targetNames) {
                this.actionText = actionText == null ? "" : actionText;
                this.targetNames = targetNames == null
                        ? Collections.emptyList()
                        : new ArrayList<>(targetNames);
            }
        }

        private List<Ability> transcriptPlayableActions(Game game) {
            if (game == null) {
                return Collections.emptyList();
            }
            try {
                return new ArrayList<>(getPlayableFast(game, true, Zone.ALL, false));
            } catch (Exception e) {
                return Collections.emptyList();
            }
        }

        private List<String> transcriptCandidateDebug(List<String> plannedTexts, List<String> playableTexts) {
            List<String> out = new ArrayList<>();
            if (plannedTexts != null) {
                for (String text : plannedTexts) {
                    out.add("planned:" + text);
                }
            }
            if (playableTexts != null) {
                for (String text : playableTexts) {
                    out.add("legal:" + text);
                }
            }
            return out;
        }
    }

    private static final class ActionPlayer extends ComputerPlayerRL {

        private final Set<StateSequenceBuilder.ActionType> targetTypes;
        private final List<List<Integer>> prefixChoices;
        private final List<List<String>> prefixChoiceTexts;
        private List<StateSequenceBuilder.ActionType> prefixActionTypes = Collections.emptyList();
        private List<ReplayExpectation> prefixExpectations = Collections.emptyList();
        private final int targetOrdinal;
        private final int forcedIndex;
        private final List<String> forcedChoiceTexts;
        private final TerminalMode terminalMode;
        private final int maxGameTurns;
        private final boolean tacticAutopilot;
        private final boolean valueProbeAfterTarget;
        private final List<List<String>> candidateTextsByOrdinal = new ArrayList<>();
        private final List<StateSnapshot> stateSnapshotsByOrdinal = new ArrayList<>();
        private int eligibleOrdinal = 0;
        private boolean targetForced = false;
        private int forcedPrefixCount = 0;
        private boolean terminalMilestoneReached = false;
        private boolean turnLimitReached = false;
        private String firstMulliganHandText = "";
        private String firstPriorityHandText = "";
        private boolean valueProbeCaptured = false;
        private boolean valueProbeTerminal = false;
        private float valueProbeScore = Float.NaN;
        private int valueProbeOrdinal = -1;
        private StateSnapshot valueProbeState = StateSnapshot.EMPTY;
        private PrefixDivergence firstPrefixDivergence = null;
        private String lastCombatTrace = "";
        private int lastCombatTraceOrdinal = -1;
        private boolean forceKeepOpeningHand = false;
        private List<String> exactOpeningHandNames = Collections.emptyList();
        private List<String> exactOpeningLibraryNames = Collections.emptyList();
        private boolean exactOpeningStateRestored = false;
        private ScenarioJob livePrefixTraceJob = null;
        private ReplayExpectation livePrefixTraceTarget = null;
        private Path livePrefixTracePath = null;
        private boolean firstPriorityStartupDiagnosticWritten = false;
        private int skippedStackPassOrdinal = -1;
        private int skippedStackPassCount = 0;
        private int skippedPreTargetPassOrdinal = -1;
        private int skippedPreTargetPassCount = 0;
        private int d079PhaseAdvancePassOrdinal = -1;
        private int d079PhaseAdvancePassCount = 0;
        private int d070InsertedBlockersDoneOrdinal = -1;
        private int d070InsertedBlockersDoneCount = 0;
        private int d035InsertedBlockersDoneOrdinal = -1;
        private int d035InsertedBlockersDoneCount = 0;
        private OpponentTranscriptCursor opponentTranscriptCursor = null;
        private ScenarioJob checkpointCaptureJob = null;
        private List<EngineDecisionCheckpoint> checkpointCaptureSink = null;
        private int checkpointForcedOrdinal = -1;
        private List<Integer> checkpointForcedIndices = Collections.emptyList();
        private List<String> checkpointForcedTexts = Collections.emptyList();
        private boolean checkpointStopAtReentry = false;
        private CheckpointReentryProbe lastCheckpointReentryProbe = null;
        private int pendingReplayPaymentOrdinal = -1;
        private Set<String> activeReplayPaymentReservedManaSourceIds = Collections.emptySet();
        private static final int REPLAY_PAYMENT_RESERVATION_MIN_ORDINAL = 26;

        private ActionPlayer(
                String name,
                Set<StateSequenceBuilder.ActionType> targetTypes,
                List<List<Integer>> prefixChoices,
                List<List<String>> prefixChoiceTexts,
                int targetOrdinal,
                int forcedIndex,
                List<String> forcedChoiceTexts,
                boolean greedy,
                TerminalMode terminalMode,
                int maxGameTurns,
                boolean tacticAutopilot
        ) {
            this(name, targetTypes, prefixChoices, prefixChoiceTexts, targetOrdinal, forcedIndex,
                    forcedChoiceTexts, greedy, terminalMode, maxGameTurns, tacticAutopilot, false);
        }

        private ActionPlayer(
                String name,
                Set<StateSequenceBuilder.ActionType> targetTypes,
                List<List<Integer>> prefixChoices,
                List<List<String>> prefixChoiceTexts,
                int targetOrdinal,
                int forcedIndex,
                List<String> forcedChoiceTexts,
                boolean greedy,
                TerminalMode terminalMode,
                int maxGameTurns,
                boolean tacticAutopilot,
                boolean valueProbeAfterTarget
        ) {
            super(name, RangeOfInfluence.ALL, RLTrainer.sharedModel, greedy, true, "train");
            this.targetTypes = targetTypes == null ? EnumSet.noneOf(StateSequenceBuilder.ActionType.class) : targetTypes;
            this.prefixChoices = prefixChoices == null ? Collections.emptyList() : prefixChoices;
            this.prefixChoiceTexts = prefixChoiceTexts == null ? Collections.emptyList() : prefixChoiceTexts;
            this.targetOrdinal = targetOrdinal;
            this.forcedIndex = forcedIndex;
            this.forcedChoiceTexts = forcedChoiceTexts == null ? Collections.emptyList() : forcedChoiceTexts;
            this.terminalMode = terminalMode == null ? TerminalMode.WIN : terminalMode;
            this.maxGameTurns = Math.max(0, maxGameTurns);
            this.tacticAutopilot = tacticAutopilot;
            this.valueProbeAfterTarget = valueProbeAfterTarget;
        }

        private ActionPlayer(ActionPlayer player) {
            super(player);
            this.targetTypes = player.targetTypes == null || player.targetTypes.isEmpty()
                    ? EnumSet.noneOf(StateSequenceBuilder.ActionType.class)
                    : EnumSet.copyOf(player.targetTypes);
            this.prefixChoices = copyPrefix(player.prefixChoices);
            this.prefixChoiceTexts = copyPrefixTexts(player.prefixChoiceTexts);
            this.prefixActionTypes = player.prefixActionTypes == null
                    ? Collections.emptyList()
                    : new ArrayList<>(player.prefixActionTypes);
            this.prefixExpectations = player.prefixExpectations == null
                    ? Collections.emptyList()
                    : new ArrayList<>(player.prefixExpectations);
            this.targetOrdinal = player.targetOrdinal;
            this.forcedIndex = player.forcedIndex;
            this.forcedChoiceTexts = player.forcedChoiceTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(player.forcedChoiceTexts);
            this.terminalMode = player.terminalMode;
            this.maxGameTurns = player.maxGameTurns;
            this.tacticAutopilot = player.tacticAutopilot;
            this.valueProbeAfterTarget = player.valueProbeAfterTarget;
            this.eligibleOrdinal = player.eligibleOrdinal;
            this.targetForced = player.targetForced;
            this.forcedPrefixCount = player.forcedPrefixCount;
            this.terminalMilestoneReached = player.terminalMilestoneReached;
            this.turnLimitReached = player.turnLimitReached;
            this.firstMulliganHandText = player.firstMulliganHandText;
            this.firstPriorityHandText = player.firstPriorityHandText;
            this.valueProbeCaptured = player.valueProbeCaptured;
            this.valueProbeTerminal = player.valueProbeTerminal;
            this.valueProbeScore = player.valueProbeScore;
            this.valueProbeOrdinal = player.valueProbeOrdinal;
            this.valueProbeState = player.valueProbeState;
            this.firstPrefixDivergence = player.firstPrefixDivergence;
            this.lastCombatTrace = player.lastCombatTrace;
            this.lastCombatTraceOrdinal = player.lastCombatTraceOrdinal;
            this.forceKeepOpeningHand = player.forceKeepOpeningHand;
            this.exactOpeningHandNames = player.exactOpeningHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(player.exactOpeningHandNames);
            this.exactOpeningLibraryNames = player.exactOpeningLibraryNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(player.exactOpeningLibraryNames);
            this.exactOpeningStateRestored = player.exactOpeningStateRestored;
            this.livePrefixTraceJob = null;
            this.livePrefixTraceTarget = null;
            this.livePrefixTracePath = null;
            this.firstPriorityStartupDiagnosticWritten = player.firstPriorityStartupDiagnosticWritten;
            this.skippedStackPassOrdinal = player.skippedStackPassOrdinal;
            this.skippedStackPassCount = player.skippedStackPassCount;
            this.skippedPreTargetPassOrdinal = player.skippedPreTargetPassOrdinal;
            this.skippedPreTargetPassCount = player.skippedPreTargetPassCount;
            this.d079PhaseAdvancePassOrdinal = player.d079PhaseAdvancePassOrdinal;
            this.d079PhaseAdvancePassCount = player.d079PhaseAdvancePassCount;
            this.d070InsertedBlockersDoneOrdinal = player.d070InsertedBlockersDoneOrdinal;
            this.d070InsertedBlockersDoneCount = player.d070InsertedBlockersDoneCount;
            this.d035InsertedBlockersDoneOrdinal = player.d035InsertedBlockersDoneOrdinal;
            this.d035InsertedBlockersDoneCount = player.d035InsertedBlockersDoneCount;
            this.opponentTranscriptCursor = player.opponentTranscriptCursor == null
                    ? null
                    : player.opponentTranscriptCursor.copy();
            this.pendingReplayPaymentOrdinal = player.pendingReplayPaymentOrdinal;
            this.activeReplayPaymentReservedManaSourceIds = player.activeReplayPaymentReservedManaSourceIds == null
                    ? Collections.emptySet()
                    : new LinkedHashSet<>(player.activeReplayPaymentReservedManaSourceIds);
        }

        @Override
        public ActionPlayer copy() {
            return new ActionPlayer(this);
        }

        void setPrefixActionTypes(List<StateSequenceBuilder.ActionType> prefixActionTypes) {
            this.prefixActionTypes = prefixActionTypes == null
                    ? Collections.emptyList()
                    : new ArrayList<>(prefixActionTypes);
        }

        void setPrefixExpectations(List<ReplayExpectation> prefixExpectations) {
            this.prefixExpectations = prefixExpectations == null
                    ? Collections.emptyList()
                    : new ArrayList<>(prefixExpectations);
        }

        void setForceKeepOpeningHand(boolean forceKeepOpeningHand) {
            this.forceKeepOpeningHand = forceKeepOpeningHand;
        }

        void setExactFirstPriorityOpeningState(List<String> handNames, List<String> libraryNames) {
            this.exactOpeningHandNames = handNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(handNames);
            this.exactOpeningLibraryNames = libraryNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(libraryNames);
        }

        void setLivePrefixTrace(ScenarioJob job, ReplayExpectation target, Path path) {
            this.livePrefixTraceJob = job;
            this.livePrefixTraceTarget = target;
            this.livePrefixTracePath = path;
        }

        void setOpponentTranscriptCursor(OpponentTranscriptCursor opponentTranscriptCursor) {
            this.opponentTranscriptCursor = opponentTranscriptCursor;
            updateOpponentTranscriptOrdinalContext();
        }

        void setCheckpointCaptureSink(ScenarioJob job, List<EngineDecisionCheckpoint> sink) {
            this.checkpointCaptureJob = job;
            this.checkpointCaptureSink = sink;
        }

        void setCheckpointForcedChoice(
                int ordinal,
                List<Integer> forcedIndices,
                List<String> forcedTexts,
                boolean stopAtReentry
        ) {
            this.checkpointForcedOrdinal = ordinal;
            this.checkpointForcedIndices = forcedIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(forcedIndices);
            this.checkpointForcedTexts = forcedTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(forcedTexts);
            this.checkpointStopAtReentry = stopAtReentry;
            this.lastCheckpointReentryProbe = null;
        }

        CheckpointReentryProbe getLastCheckpointReentryProbe() {
            return lastCheckpointReentryProbe;
        }

        @Override
        public boolean playMana(Ability ability, ManaCost unpaid, String promptText, Game game) {
            Set<String> previousReserved = activeReplayPaymentReservedManaSourceIds;
            if (shouldApplyReplayPaymentReservation(ability)) {
                activeReplayPaymentReservedManaSourceIds =
                        replayPaymentReservedManaSourceIds(pendingReplayPaymentOrdinal);
            }
            try {
                return super.playMana(ability, unpaid, promptText, game);
            } finally {
                activeReplayPaymentReservedManaSourceIds = previousReserved;
            }
        }

        @Override
        public List<MageObject> getAvailableManaProducers(Game game) {
            return filterReplayReservedManaSources(super.getAvailableManaProducers(game));
        }

        @Override
        public List<Permanent> getAvailableManaProducersWithCost(Game game) {
            return filterReplayReservedManaSources(super.getAvailableManaProducersWithCost(game));
        }

        @Override
        public boolean priority(Game game) {
            maybeWriteFirstPriorityStartupDiagnostic(game);
            updateOpponentTranscriptOrdinalContext();
            if (maybePassD079PhaseAdvanceBridge(game)) {
                return false;
            }
            if (maybePassSourceSkippedPreTargetGap(game)) {
                return false;
            }
            if (maybePassSourceSkippedStackPriority(game)) {
                return false;
            }
            if (valueProbeAfterTarget) {
                maybeRestoreExactFirstPriorityOpeningState(game);
                captureFirstPriorityHand(game);
                if (valueProbeCaptured || checkTerminalMilestone(game) || checkTurnLimit(game)) {
                    return false;
                }
                game.resumeTimer(getTurnControlledBy());
                try {
                    boolean result;
                    try {
                        result = priorityPlay(game);
                    } catch (ValueProbeTerminated ignored) {
                        result = false;
                    } catch (Throwable ignored) {
                        pass(game);
                        result = false;
                    }
                    checkTerminalMilestone(game);
                    return result;
                } finally {
                    game.pauseTimer(getTurnControlledBy());
                }
            }
            maybeRestoreExactFirstPriorityOpeningState(game);
            captureFirstPriorityHand(game);
            if (checkTerminalMilestone(game)) {
                return false;
            }
            if (checkTurnLimit(game)) {
                return false;
            }
            boolean result = super.priority(game);
            checkTerminalMilestone(game);
            return result;
        }

        private void maybeWriteFirstPriorityStartupDiagnostic(Game game) {
            if (firstPriorityStartupDiagnosticWritten) {
                return;
            }
            firstPriorityStartupDiagnosticWritten = true;
            appendReplayStartupDiagnostic("prefix", "first_priority_entry",
                    livePrefixTraceJob, null, null, this, null, game);
        }

        @Override
        public void selectAttackers(Game game, UUID attackingPlayerId) {
            if (sourceSkippedNoAttackersPrompt(game, attackingPlayerId)) {
                return;
            }
            super.selectAttackers(game, attackingPlayerId);
        }

        @Override
        public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
            maybeRecordCombatTrace(game, "before_selectBlockers");
            if (sourceD070InsertedDeclareBlockersNoAdditionalBlockers(game)) {
                maybeRecordCombatTrace(game, "d070_inserted_blockers_done");
                return;
            }
            super.selectBlockers(source, game, defendingPlayerId);
            maybeRecordCombatTrace(game, "after_selectBlockers");
        }

        @Override
        protected Integer forcedMulliganChoiceIndex(Game game, int handSize, int landCount) {
            updateOpponentTranscriptOrdinalContext();
            if (!targetTypes.contains(StateSequenceBuilder.ActionType.MULLIGAN)) {
                if (forceKeepOpeningHand) {
                    captureFirstMulliganHand(game);
                    return 0;
                }
                if (tacticAutopilot) {
                    return 0;
                }
                return null;
            }
            captureFirstMulliganHand(game);
            int ordinal = eligibleOrdinal++;
            List<String> texts = Arrays.asList("KEEP", "MULLIGAN");
            StateSnapshot state = StateSnapshot.capture(game, this);
            candidateTextsByOrdinal.add(texts);
            stateSnapshotsByOrdinal.add(state);
            if (ordinal < prefixChoices.size() && (targetOrdinal < 0 || ordinal < targetOrdinal)) {
                StateSequenceBuilder.ActionType expectedActionType = prefixActionTypeAt(ordinal);
                if (expectedActionType != null && expectedActionType != StateSequenceBuilder.ActionType.MULLIGAN) {
                    recordPrefixDivergence(ordinal, "action_type_mismatch", expectedActionType,
                            StateSequenceBuilder.ActionType.MULLIGAN, prefixChoices.get(ordinal),
                            prefixChoiceTextsAt(ordinal), texts, Collections.emptyList(),
                            state,
                            "expected action type does not match replay action type");
                    appendLivePrefixTrace(ordinal, StateSequenceBuilder.ActionType.MULLIGAN,
                            Collections.emptyList(), texts, state);
                    return null;
                }
                List<Integer> forced = sanitize(prefixChoices.get(ordinal), prefixChoiceTextsAt(ordinal),
                        texts, texts, 1, 1, null);
                if (forced != null && !forced.isEmpty()) {
                    forcedPrefixCount++;
                    updateOpponentTranscriptOrdinalContext();
                    appendLivePrefixTrace(ordinal, StateSequenceBuilder.ActionType.MULLIGAN,
                            forced, texts, state);
                    return forced.get(0);
                }
                recordPrefixDivergence(ordinal, prefixFailureReason(prefixChoices.get(ordinal),
                                prefixChoiceTextsAt(ordinal), texts, 1, 1),
                        expectedActionType, StateSequenceBuilder.ActionType.MULLIGAN,
                        prefixChoices.get(ordinal), prefixChoiceTextsAt(ordinal), texts,
                        Collections.emptyList(), state,
                        prefixFailureDetail(prefixChoices.get(ordinal), prefixChoiceTextsAt(ordinal), texts, 1, 1));
                appendLivePrefixTrace(ordinal, StateSequenceBuilder.ActionType.MULLIGAN,
                        Collections.emptyList(), texts, state);
                return null;
            }
            if (ordinal == targetOrdinal && forcedIndex >= 0) {
                targetForced = true;
                List<Integer> forced = sanitize(Collections.singletonList(forcedIndex), forcedChoiceTexts,
                        texts, texts, 1, 1, null);
                return forced == null || forced.isEmpty() ? null : forced.get(0);
            }
            if (tacticAutopilot && (targetOrdinal < 0 || ordinal > targetOrdinal)) {
                return 0;
            }
            return null;
        }

        @Override
        protected <T> List<Integer> forcedChoiceIndices(
                StateSequenceBuilder.ActionType actionType,
                List<T> candidates,
                int maxTargets,
                int minTargets,
                Game game,
                mage.abilities.Ability source
        ) {
            updateOpponentTranscriptOrdinalContext();
            if (isExcludedDecisionStep(game)) {
                return null;
            }
            boolean tracked = targetTypes.contains(actionType);
            int count = candidates == null ? 0 : candidates.size();
            if (count < 2) {
                if (tracked && eligibleOrdinal < prefixChoices.size()) {
                    List<String> texts = snapshotCandidateTexts(candidates, game);
                    StateSequenceBuilder.ActionType expectedActionType = prefixActionTypeAt(eligibleOrdinal);
                    List<String> expectedTexts = prefixChoiceTextsForReplay(eligibleOrdinal, expectedActionType);
                    StateSnapshot state = StateSnapshot.capture(game, this);
                    List<Integer> actualSelected = count == 1
                            ? Collections.singletonList(0)
                            : Collections.emptyList();
                    PrefixContextFailure contextFailure = prefixContextFailure(eligibleOrdinal, actionType, state);
                    String reason = contextFailure != null ? contextFailure.reason
                            : expectedActionType != null && expectedActionType != actionType
                            ? "action_type_mismatch"
                            : "state_divergence";
                    String detail = contextFailure != null ? contextFailure.detail
                            : expectedActionType != null && expectedActionType != actionType
                            ? "expected action type does not match replay action type"
                            : "replay action has fewer than two candidates and is skipped by ActionCounterfactual ordinal tracking";
                    recordPrefixDivergence(eligibleOrdinal, reason, expectedActionType, actionType,
                            prefixChoices.get(eligibleOrdinal), expectedTexts,
                            texts, actualSelected,
                            state, detail);
                    appendLivePrefixTrace(eligibleOrdinal, actionType, actualSelected, texts, state);
                }
                return null;
            }
            List<String> texts = snapshotCandidateTexts(candidates, game);
            StateSnapshot state = StateSnapshot.capture(game, this);
            List<Integer> d070InsertedDone = d070InsertedDeclareBlockersDoneChoice(actionType, texts, state, game);
            if (d070InsertedDone != null) {
                appendPreTargetPassTrace(game, this, "synthetic_d070_inserted_declare_blockers_done",
                        activePrefixExpectationAt(eligibleOrdinal), eligibleOrdinal,
                        previousSourceDecisionNumberBefore(eligibleOrdinal),
                        sourceDecisionGapBefore(eligibleOrdinal, activePrefixExpectationAt(eligibleOrdinal)),
                        d070InsertedBlockersDoneCount, "d067_declare_blockers_consumed",
                        opponentTranscriptCursor);
                d070InsertedBlockersDoneCount++;
                return d070InsertedDone;
            }
            List<Integer> d035InsertedDone = d035InsertedDeclareBlockersDoneChoice(actionType, texts, state, game);
            if (d035InsertedDone != null) {
                appendPreTargetPassTrace(game, this, "synthetic_d035_inserted_declare_blockers_done",
                        activePrefixExpectationAt(eligibleOrdinal), eligibleOrdinal,
                        previousSourceDecisionNumberBefore(eligibleOrdinal),
                        sourceDecisionGapBefore(eligibleOrdinal, activePrefixExpectationAt(eligibleOrdinal)),
                        d035InsertedBlockersDoneCount, "d034_declare_blockers_consumed",
                        opponentTranscriptCursor);
                d035InsertedBlockersDoneCount++;
                return d035InsertedDone;
            }
            if (!tracked && !tacticAutopilot) {
                return null;
            }
            if (!tracked) {
                return tacticChoice(actionType, texts, state, maxTargets, minTargets);
            }
            maybeCaptureEngineDecisionCheckpoint(eligibleOrdinal, actionType, texts, state, game);
            int ordinal = eligibleOrdinal++;
            updateOpponentTranscriptOrdinalContext();
            candidateTextsByOrdinal.add(texts);
            stateSnapshotsByOrdinal.add(state);
            List<Integer> checkpointForced = checkpointForcedChoice(
                    ordinal, actionType, candidates, texts, maxTargets, minTargets, state, game);
            if (checkpointForced != null) {
                return checkpointForced;
            }
            if (valueProbeAfterTarget
                    && targetForced
                    && ordinal > targetOrdinal
                    && actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
                captureValueProbe(game, ordinal);
                throw ValueProbeTerminated.INSTANCE;
            }
            if (ordinal < prefixChoices.size() && (targetOrdinal < 0 || ordinal < targetOrdinal)) {
                StateSequenceBuilder.ActionType expectedActionType = prefixActionTypeAt(ordinal);
                List<String> expectedTexts = prefixChoiceTextsForReplay(ordinal, expectedActionType);
                if (expectedActionType != null && expectedActionType != actionType) {
                    List<Integer> d046DiscardChoice = d046RefurbishedFamiliarDiscardChoice(
                            ordinal, expectedActionType, actionType, candidates, texts,
                            maxTargets, minTargets, state, game);
                    if (d046DiscardChoice != null) {
                        forcedPrefixCount++;
                        appendLivePrefixTrace(ordinal, actionType, d046DiscardChoice, texts, state);
                        return d046DiscardChoice;
                    }
                    List<Integer> repeatedBlockersPass = repeatedDeclareBlockersPassChoice(
                            ordinal, expectedActionType, actionType, texts, state, game);
                    if (repeatedBlockersPass != null) {
                        forcedPrefixCount++;
                        appendLivePrefixTrace(ordinal, actionType, repeatedBlockersPass, texts, state);
                        return repeatedBlockersPass;
                    }
                    recordPrefixDivergence(ordinal, "action_type_mismatch", expectedActionType, actionType,
                            prefixChoices.get(ordinal), expectedTexts, texts,
                            Collections.emptyList(), state,
                            "expected action type does not match replay action type");
                    appendLivePrefixTrace(ordinal, actionType, Collections.emptyList(), texts, state);
                    return null;
                }
                PrefixContextFailure contextFailure = prefixContextFailure(ordinal, actionType, state);
                if (contextFailure != null) {
                    recordPrefixDivergence(ordinal, contextFailure.reason, expectedActionType, actionType,
                            prefixChoices.get(ordinal), expectedTexts, texts,
                            Collections.emptyList(), state, contextFailure.detail);
                    appendLivePrefixTrace(ordinal, actionType, Collections.emptyList(), texts, state);
                    return null;
                }
                PrefixContextFailure identityFailure = prefixIdentityFailure(
                        ordinal, expectedActionType, actionType, expectedTexts);
                if (identityFailure != null) {
                    recordPrefixDivergence(ordinal, identityFailure.reason, expectedActionType, actionType,
                            prefixChoices.get(ordinal), expectedTexts, texts,
                            Collections.emptyList(), state, identityFailure.detail);
                    appendLivePrefixTrace(ordinal, actionType, Collections.emptyList(), texts, state);
                    return null;
                }
                PrefixObjectChoice objectChoice = objectAwarePrefixChoice(
                        ordinal, expectedActionType, actionType, candidates, texts, maxTargets, minTargets, game);
                if (objectChoice != null) {
                    if (objectChoice.failure != null) {
                        recordPrefixDivergence(ordinal, objectChoice.failure.reason, expectedActionType, actionType,
                                prefixChoices.get(ordinal), expectedTexts, texts,
                                Collections.emptyList(), state, objectChoice.failure.detail);
                        appendLivePrefixTrace(ordinal, actionType, Collections.emptyList(), texts, state);
                        return null;
                    }
                    if (objectChoice.indices != null && !objectChoice.indices.isEmpty()) {
                        maybeAdvanceRandomUtilToSourceSearchCount(prefixExpectationAt(ordinal));
                        forcedPrefixCount++;
                        markPendingReplayPayment(ordinal, expectedActionType);
                        appendLivePrefixTrace(ordinal, actionType, objectChoice.indices, texts, state);
                        return objectChoice.indices;
                    }
                }
                List<Integer> forced = sanitize(prefixChoices.get(ordinal), expectedTexts,
                        candidates, texts, maxTargets, minTargets, game);
                if (forced != null && !forced.isEmpty()) {
                    maybeAdvanceRandomUtilToSourceSearchCount(prefixExpectationAt(ordinal));
                    forcedPrefixCount++;
                    markPendingReplayPayment(ordinal, expectedActionType);
                    appendLivePrefixTrace(ordinal, actionType, forced, texts, state);
                } else {
                    recordPrefixDivergence(ordinal, prefixFailureReason(prefixChoices.get(ordinal),
                                    expectedTexts, texts, maxTargets, minTargets),
                            expectedActionType, actionType, prefixChoices.get(ordinal),
                            expectedTexts, texts, Collections.emptyList(),
                            state,
                            prefixFailureDetail(prefixChoices.get(ordinal), expectedTexts,
                                    texts, maxTargets, minTargets));
                    appendLivePrefixTrace(ordinal, actionType, Collections.emptyList(), texts, state);
                }
                return forced;
            }
            if (ordinal == targetOrdinal && forcedIndex >= 0) {
                List<Integer> forced = sanitize(Collections.singletonList(forcedIndex), forcedChoiceTexts,
                        candidates, texts, maxTargets, minTargets, game);
                if (forced != null && !forced.isEmpty()) {
                    maybeAdvanceRandomUtilToSourceSearchCount(prefixExpectationAt(ordinal));
                    targetForced = true;
                    markPendingReplayPayment(ordinal, actionType);
                }
                return forced;
            }
            if (tacticAutopilot && (targetOrdinal < 0 || ordinal > targetOrdinal)) {
                List<Integer> tactic = tacticChoice(actionType, texts, StateSnapshot.capture(game, this),
                        maxTargets, minTargets);
                if (tactic != null && !tactic.isEmpty()) {
                    return tactic;
                }
            }
            return null;
        }

        private void markPendingReplayPayment(int ordinal, StateSequenceBuilder.ActionType actionType) {
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
                return;
            }
            if (ordinal < REPLAY_PAYMENT_RESERVATION_MIN_ORDINAL) {
                clearPendingReplayPayment();
                return;
            }
            if (!isReplaySpellCastExpectation(actionType, expected)) {
                clearPendingReplayPayment();
                return;
            }
            Set<String> reserved = replayPaymentReservedManaSourceIds(ordinal);
            if (reserved.isEmpty()) {
                clearPendingReplayPayment();
                return;
            }
            pendingReplayPaymentOrdinal = ordinal;
        }

        private void clearPendingReplayPayment() {
            pendingReplayPaymentOrdinal = -1;
        }

        private boolean shouldApplyReplayPaymentReservation(Ability ability) {
            return pendingReplayPaymentOrdinal >= 0
                    && !replayPaymentReservedManaSourceIds(pendingReplayPaymentOrdinal).isEmpty();
        }

        private Set<String> replayPaymentReservedManaSourceIds(int ordinal) {
            ReplayExpectation current = prefixExpectationAt(ordinal);
            if (current == null || current.sourceTurn.isEmpty()) {
                return Collections.emptySet();
            }
            Set<String> out = new LinkedHashSet<>();
            for (int i = ordinal + 1; i < prefixExpectations.size(); i++) {
                if (targetOrdinal >= 0 && i > targetOrdinal) {
                    break;
                }
                ReplayExpectation future = prefixExpectationAt(i);
                if (future == null) {
                    continue;
                }
                if (!future.sourceTurn.isEmpty() && !current.sourceTurn.equals(future.sourceTurn)) {
                    break;
                }
                int limit = Math.min(future.sourceCandidateTexts.size(), future.sourceCandidateObjectIds.size());
                for (int j = 0; j < limit; j++) {
                    if (!isReplayManaAbilityText(future.sourceCandidateTexts.get(j))) {
                        continue;
                    }
                    String sourceId = normalizeObjectId(future.sourceCandidateObjectIds.get(j));
                    if (!sourceId.isEmpty()) {
                        out.add(sourceId);
                    }
                }
            }
            return out;
        }

        private boolean isReplaySpellCastExpectation(
                StateSequenceBuilder.ActionType actionType,
                ReplayExpectation expected
        ) {
            if (actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL || expected == null) {
                return false;
            }
            if (isReplaySpellCastText(expected.sourceSelectedText) || isReplaySpellCastText(expected.expectedText)) {
                return true;
            }
            for (String text : expected.expectedTexts) {
                if (isReplaySpellCastText(text)) {
                    return true;
                }
            }
            return false;
        }

        private boolean isReplaySpellCastText(String text) {
            String normalized = normalizeText(text);
            return normalized.startsWith("cast ") || normalized.contains(": cast ");
        }

        private boolean isReplayManaAbilityText(String text) {
            String normalized = normalizeText(text);
            return normalized.contains("add {") || normalized.contains("add one mana");
        }

        private <T extends MageObject> List<T> filterReplayReservedManaSources(List<T> producers) {
            if (producers == null || producers.isEmpty()
                    || activeReplayPaymentReservedManaSourceIds == null
                    || activeReplayPaymentReservedManaSourceIds.isEmpty()) {
                return producers;
            }
            List<T> filtered = new ArrayList<>(producers.size());
            for (T producer : producers) {
                String id = producer == null ? "" : normalizeObjectId(producer.getId());
                if (!activeReplayPaymentReservedManaSourceIds.contains(id)) {
                    filtered.add(producer);
                }
            }
            return filtered;
        }

        private void maybeCaptureEngineDecisionCheckpoint(
                int ordinal,
                StateSequenceBuilder.ActionType actionType,
                List<String> texts,
                StateSnapshot state,
                Game game
        ) {
            if (checkpointCaptureSink == null || checkpointCaptureJob == null || game == null) {
                return;
            }
            if (texts == null || texts.size() < 2 || !targetTypes.contains(actionType)) {
                return;
            }
            for (EngineDecisionCheckpoint checkpoint : checkpointCaptureSink) {
                if (checkpoint != null && checkpoint.ordinal == ordinal) {
                    return;
                }
            }
            try {
                RandomUtil.State randomState = RandomUtil.captureState();
                Game snapshot = game.createSimulationForAI();
                checkpointCaptureSink.add(new EngineDecisionCheckpoint(
                        checkpointCaptureJob,
                        snapshot,
                        getId(),
                        ordinal,
                        "",
                        "",
                        actionType,
                        texts,
                        Collections.emptyList(),
                        Collections.emptyList(),
                        state,
                        randomState));
            } catch (Throwable t) {
                System.err.println("[CHECKPOINT_CAPTURE_FAILED] ordinal=" + ordinal
                        + " action_type=" + actionType
                        + " error=" + exceptionSummary(asException(t)));
            }
        }

        private <T> List<Integer> checkpointForcedChoice(
                int ordinal,
                StateSequenceBuilder.ActionType actionType,
                List<T> candidates,
                List<String> texts,
                int maxTargets,
                int minTargets,
                StateSnapshot state,
                Game game
        ) {
            if (checkpointForcedOrdinal < 0 || ordinal != checkpointForcedOrdinal) {
                return null;
            }
            List<Integer> forced = sanitize(checkpointForcedIndices, checkpointForcedTexts,
                    candidates, texts, maxTargets, minTargets, game);
            if (forced == null || forced.isEmpty()) {
                forced = Collections.emptyList();
            }
            List<String> selectedTexts = selectedTexts(texts, forced);
            lastCheckpointReentryProbe = new CheckpointReentryProbe(
                    actionType, texts, forced, selectedTexts, state);
            if (!forced.isEmpty()) {
                targetForced = true;
            }
            if (checkpointStopAtReentry) {
                throw CheckpointProbeTerminated.INSTANCE;
            }
            return forced.isEmpty() ? null : forced;
        }

        private List<String> selectedTexts(List<String> texts, List<Integer> indices) {
            List<String> out = new ArrayList<>();
            if (texts == null || indices == null) {
                return out;
            }
            for (Integer index : indices) {
                int i = index == null ? -1 : index;
                out.add(i >= 0 && i < texts.size() ? texts.get(i) : "");
            }
            return out;
        }

        @Override
        protected boolean shouldRecordTrainingData(StateSequenceBuilder.TrainingData td, Game game) {
            return !isExcludedDecisionStep(game);
        }

        private boolean isExcludedDecisionStep(Game game) {
            if (game == null || game.getStep() == null || game.getStep().getType() == null) {
                return false;
            }
            String step = game.getStep().getType().toString().toUpperCase(Locale.ROOT);
            return step.contains("CLEANUP");
        }

        private <T> List<Integer> d046RefurbishedFamiliarDiscardChoice(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<T> candidates,
                List<String> actualTexts,
                int maxTargets,
                int minTargets,
                StateSnapshot state,
                Game game
        ) {
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected == null
                    || expectedActionType != StateSequenceBuilder.ActionType.SELECT_TARGETS
                    || actualActionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || expected.sourceAnchorId == null
                    || !expected.sourceAnchorId.endsWith("_D046")
                    || !"46".equals(expected.sourceDecisionNumber)
                    || !"TARGET_PICK".equals(sourcePromptKey(expected.sourcePhase))) {
                return null;
            }
            String stateText = state == null ? "" : state.toCompactText();
            if (!stateField(stateText, "opponentBattlefield").contains("Refurbished Familiar")) {
                return null;
            }
            if (!"PRECOMBAT_MAIN".equals(sourceStepKey(expected.sourcePhase))
                    || !"PRECOMBAT_MAIN".equals(contextKey(state == null ? "" : state.stepText()))) {
                return null;
            }
            String hand = stateField(stateText, "hand");
            List<String> expectedTexts = prefixChoiceTextsForReplay(ordinal, expectedActionType);
            if (hand.isEmpty()
                    || !zoneContainsAnyText(hand, expectedTexts)
                    || !zoneContainsAllActualCandidateTexts(hand, actualTexts)) {
                return null;
            }
            return sanitize(prefixChoices.get(ordinal), expectedTexts,
                    candidates, actualTexts, maxTargets, minTargets, game);
        }

        private boolean zoneContainsAnyText(String zoneText, List<String> names) {
            if (names == null) {
                return false;
            }
            for (String name : names) {
                String normalized = normalizeText(name);
                if (!normalized.isEmpty() && zoneContainsNormalizedName(zoneText, normalized)) {
                    return true;
                }
            }
            return false;
        }

        private boolean zoneContainsAllActualCandidateTexts(String zoneText, List<String> actualTexts) {
            if (actualTexts == null) {
                return false;
            }
            for (String text : actualTexts) {
                String normalized = normalizeText(text);
                if (normalized.isEmpty()) {
                    continue;
                }
                if (!zoneContainsNormalizedName(zoneText, normalized)) {
                    return false;
                }
            }
            return true;
        }

        private boolean zoneContainsNormalizedName(String zoneText, String normalizedName) {
            if (zoneText == null || zoneText.isEmpty() || normalizedName == null || normalizedName.isEmpty()) {
                return false;
            }
            for (String raw : zoneText.split("\\|")) {
                if (normalizeText(raw).equals(normalizedName)) {
                    return true;
                }
            }
            return false;
        }

        private List<String> prefixChoiceTextsAt(int ordinal) {
            if (ordinal < 0 || ordinal >= prefixChoiceTexts.size()) {
                return Collections.emptyList();
            }
            List<String> texts = prefixChoiceTexts.get(ordinal);
            return texts == null ? Collections.emptyList() : texts;
        }

        private List<String> prefixChoiceTextsForReplay(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType
        ) {
            List<String> texts = prefixChoiceTextsAt(ordinal);
            if (hasNonBlankText(texts)) {
                return texts;
            }
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected == null
                    || expected.sourceSelectedText == null
                    || expected.sourceSelectedText.trim().isEmpty()) {
                return texts;
            }
            if (expectedActionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS) {
                return isNoAttackersText(expected.sourceSelectedText)
                        ? texts
                        : Collections.singletonList(expected.sourceSelectedText);
            }
            if (expectedActionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || expectedActionType == StateSequenceBuilder.ActionType.CHOOSE_USE) {
                return Collections.singletonList(expected.sourceSelectedText);
            }
            return texts;
        }

        private boolean hasNonBlankText(List<String> texts) {
            if (texts == null) {
                return false;
            }
            for (String text : texts) {
                if (text != null && !text.trim().isEmpty()) {
                    return true;
                }
            }
            return false;
        }

        private StateSequenceBuilder.ActionType prefixActionTypeAt(int ordinal) {
            if (ordinal < 0 || ordinal >= prefixActionTypes.size()) {
                return null;
            }
            return prefixActionTypes.get(ordinal);
        }

        private ReplayExpectation prefixExpectationAt(int ordinal) {
            if (ordinal < 0 || ordinal >= prefixExpectations.size()) {
                return null;
            }
            return prefixExpectations.get(ordinal);
        }

        private ReplayExpectation activePrefixExpectationAt(int ordinal) {
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected != null) {
                return expected;
            }
            if (livePrefixTraceTarget != null && livePrefixTraceTarget.ordinal == ordinal) {
                return livePrefixTraceTarget;
            }
            return null;
        }

        private List<Integer> expectedIndicesAt(int ordinal, ReplayExpectation expected) {
            if (ordinal >= 0 && ordinal < prefixChoices.size()) {
                return prefixChoices.get(ordinal);
            }
            return expected == null ? Collections.emptyList() : expected.expectedIndices;
        }

        private List<String> expectedTextsAt(int ordinal, ReplayExpectation expected) {
            if (ordinal >= 0 && ordinal < prefixChoiceTexts.size()) {
                return prefixChoiceTextsAt(ordinal);
            }
            return expected == null ? Collections.emptyList() : expected.expectedTexts;
        }

        private void maybeAdvanceRandomUtilToSourceSearchCount(ReplayExpectation expected) {
            if (expected == null || expected.sourceRandomUtilCountBeforeSearch < 0) {
                return;
            }
            RandomUtil.advanceGlobalToConsumptionCount(expected.sourceRandomUtilCountBeforeSearch);
        }

        private void appendLivePrefixTrace(
                int ordinal,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> actualSelectedIndices,
                List<String> actualCandidateTexts,
                StateSnapshot state
        ) {
            if (livePrefixTracePath == null || livePrefixTraceJob == null) {
                return;
            }
            try {
                appendLivePrefixTraceRecord(livePrefixTracePath, livePrefixTraceJob, livePrefixTraceTarget,
                        prefixExpectationAt(ordinal), ordinal, actualActionType, actualSelectedIndices,
                        actualCandidateTexts, state);
            } catch (Throwable ignored) {
                // Trace durability must not affect gameplay or replay control flow.
            }
        }

        private PrefixContextFailure prefixContextFailure(
                int ordinal,
                StateSequenceBuilder.ActionType actualActionType,
                StateSnapshot actualState
        ) {
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected == null || !expected.hasSourceContext()) {
                return null;
            }
            if (expected.sourceStackCount >= 0) {
                if (actualState == null || actualState.stackCount() < 0) {
                    return new PrefixContextFailure("stack_context_mismatch",
                            sourceContextDetailWithCombatTrace(expected, actualActionType, actualState));
                }
                if (expected.sourceStackCount != actualState.stackCount()
                        || (expected.sourceStackCount > 0
                        && !replayStackTopCompatible(expected.sourceStackTop, actualState.stackTop()))) {
                    return new PrefixContextFailure("stack_context_mismatch",
                            sourceContextDetailWithCombatTrace(expected, actualActionType, actualState));
                }
            }
            String expectedStep = sourceStepKey(expected.sourcePhase);
            String actualStep = contextKey(actualState == null ? "" : actualState.stepText());
            if (!expectedStep.isEmpty() && !actualStep.isEmpty() && !expectedStep.equals(actualStep)) {
                return new PrefixContextFailure("phase_mismatch",
                        sourceContextDetailWithCombatTrace(expected, actualActionType, actualState));
            }
            String expectedPrompt = sourcePromptKey(expected.sourcePhase);
            String actualPrompt = actualPromptKey(actualActionType, actualState);
            if (!expectedPrompt.isEmpty() && !actualPrompt.isEmpty() && !promptCompatible(expectedPrompt, actualPrompt)) {
                return new PrefixContextFailure("source_context_mismatch",
                        sourceContextDetailWithCombatTrace(expected, actualActionType, actualState));
            }
            if (!expected.sourceTurn.isEmpty()
                    && expected.sourceTurn.matches("\\d+")
                    && actualState != null
                    && actualState.turn() >= 0
                    && !expected.sourceTurn.equals(String.valueOf(compactSourceTurn(actualState.turn())))) {
                return new PrefixContextFailure("source_context_mismatch",
                        sourceContextDetailWithCombatTrace(expected, actualActionType, actualState));
            }
            return null;
        }

        private PrefixContextFailure prefixIdentityFailure(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<String> expectedTexts
        ) {
            StateSequenceBuilder.ActionType checkedType = expectedActionType == null
                    ? actualActionType
                    : expectedActionType;
            if (!isObjectSensitiveReplayAction(checkedType)
                    && !isObjectSensitiveReplayAction(actualActionType)) {
                return null;
            }
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected != null && expected.hasStableObjectIdentity()) {
                return null;
            }
            String detail = "object-sensitive prefix choice lacks stable object ids"
                    + " source_decision=" + (expected == null ? "" : expected.sourceDecisionNumber)
                    + " action_type=" + checkedType
                    + " expected_texts=" + joinTexts(expectedTexts)
                    + " identity_status=" + (expected == null || expected.sourceIdentityStatus.isEmpty()
                    ? "unverifiable_display_text_only"
                    : expected.sourceIdentityStatus);
            return new PrefixContextFailure("identity_unverifiable", detail);
        }

        private <T> PrefixObjectChoice objectAwarePrefixChoice(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<T> candidates,
                List<String> actualTexts,
                int maxTargets,
                int minTargets,
                Game game
        ) {
            StateSequenceBuilder.ActionType checkedType = expectedActionType == null
                    ? actualActionType
                    : expectedActionType;
            if (checkedType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || actualActionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
                return null;
            }
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected == null || expected.sourceSelectedObjectIds.isEmpty()) {
                return null;
            }
            List<String> liveObjectIds = candidateObjectIds(candidates, game);
            List<Integer> forced = new ArrayList<>();
            for (String rawExpectedId : expected.sourceSelectedObjectIds) {
                String expectedId = normalizeObjectId(rawExpectedId);
                if (expectedId.isEmpty()) {
                    continue;
                }
                int liveIdx = indexOfObjectId(liveObjectIds, expectedId);
                if (liveIdx < 0) {
                    liveIdx = uniqueSourceSelectedTextIndex(expected, actualTexts);
                }
                if (liveIdx < 0) {
                    return PrefixObjectChoice.failure(objectIdMismatchFailure(
                            expected, expectedId, liveObjectIds, actualTexts));
                }
                if (!forced.contains(liveIdx)) {
                    forced.add(liveIdx);
                }
                if (maxTargets > 0 && forced.size() >= maxTargets) {
                    break;
                }
            }
            if (forced.size() < Math.max(0, minTargets)) {
                return PrefixObjectChoice.failure(objectIdMismatchFailure(
                        expected, joinTexts(expected.sourceSelectedObjectIds), liveObjectIds, actualTexts));
            }
            return forced.isEmpty() ? null : PrefixObjectChoice.forced(forced);
        }

        private int uniqueSourceSelectedTextIndex(ReplayExpectation expected, List<String> actualTexts) {
            if (expected == null || expected.sourceSelectedText == null || expected.sourceSelectedText.trim().isEmpty()
                    || actualTexts == null || actualTexts.isEmpty()) {
                return -1;
            }
            int match = -1;
            for (int i = 0; i < actualTexts.size(); i++) {
                if (!textMatchesExpected(expected.sourceSelectedText, actualTexts.get(i))) {
                    continue;
                }
                if (match >= 0) {
                    return -1;
                }
                match = i;
            }
            return match;
        }

        private PrefixContextFailure objectIdMismatchFailure(
                ReplayExpectation expected,
                String missingObjectId,
                List<String> liveObjectIds,
                List<String> actualTexts
        ) {
            String detail = "object-aware ACTIVATE_ABILITY_OR_SPELL prefix could not match source selected object id"
                    + " source_decision=" + (expected == null ? "" : expected.sourceDecisionNumber)
                    + " source_anchor_id=" + (expected == null ? "" : expected.sourceAnchorId)
                    + " missing_object_id=" + missingObjectId
                    + " source_selected_object_ids=" + (expected == null ? "" : joinTexts(expected.sourceSelectedObjectIds))
                    + " source_candidate_object_ids=" + (expected == null ? "" : joinTexts(expected.sourceCandidateObjectIds))
                    + " live_candidate_object_ids=" + joinTexts(liveObjectIds)
                    + " live_candidate_texts=" + joinTexts(actualTexts);
            return new PrefixContextFailure("object_id_mismatch", detail);
        }

        private <T> List<String> candidateObjectIds(List<T> candidates, Game game) {
            if (candidates == null || candidates.isEmpty()) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (T candidate : candidates) {
                out.add(candidateObjectId(candidate, game));
            }
            return out;
        }

        private String candidateObjectId(Object candidate, Game game) {
            if (candidate == null) {
                return "";
            }
            if (candidate instanceof Ability) {
                return normalizeObjectId(((Ability) candidate).getSourceId());
            }
            if (candidate instanceof Permanent) {
                return normalizeObjectId(((Permanent) candidate).getId());
            }
            if (candidate instanceof Card) {
                return normalizeObjectId(((Card) candidate).getId());
            }
            if (candidate instanceof UUID) {
                return normalizeObjectId((UUID) candidate);
            }
            if (candidate instanceof String) {
                return normalizeObjectId((String) candidate);
            }
            if (game != null) {
                try {
                    UUID id = UUID.fromString(String.valueOf(candidate).trim());
                    return normalizeObjectId(id);
                } catch (IllegalArgumentException ignored) {
                    return "";
                }
            }
            return "";
        }

        private String normalizeObjectId(UUID id) {
            return id == null ? "" : id.toString().toLowerCase(Locale.ROOT);
        }

        private String normalizeObjectId(String raw) {
            if (raw == null) {
                return "";
            }
            String text = raw.trim();
            if (text.isEmpty()) {
                return "";
            }
            try {
                return UUID.fromString(text).toString().toLowerCase(Locale.ROOT);
            } catch (IllegalArgumentException ignored) {
                return text.toLowerCase(Locale.ROOT);
            }
        }

        private int indexOfObjectId(List<String> objectIds, String expectedObjectId) {
            if (objectIds == null || expectedObjectId == null || expectedObjectId.isEmpty()) {
                return -1;
            }
            for (int i = 0; i < objectIds.size(); i++) {
                if (expectedObjectId.equals(normalizeObjectId(objectIds.get(i)))) {
                    return i;
                }
            }
            return -1;
        }

        private void updateOpponentTranscriptOrdinalContext() {
            if (opponentTranscriptCursor != null) {
                opponentTranscriptCursor.setAgentOrdinalContext(eligibleOrdinal);
            }
        }

        private void maybeRecordCombatTrace(Game game, String stage) {
            if (game == null || eligibleOrdinal < 0 || eligibleOrdinal >= prefixExpectations.size()) {
                return;
            }
            ReplayExpectation expected = prefixExpectationAt(eligibleOrdinal);
            if (expected == null || !expected.hasSourceContext()) {
                return;
            }
            String expectedStep = sourceStepKey(expected.sourcePhase);
            String expectedPrompt = sourcePromptKey(expected.sourcePhase);
            if (!"DECLARE_BLOCKERS".equals(expectedStep) && !"DECLARE_BLOCKS".equals(expectedPrompt)) {
                return;
            }
            lastCombatTraceOrdinal = eligibleOrdinal;
            lastCombatTrace = combatTrace(game, this, stage, eligibleOrdinal, expected);
        }

        private String sourceContextDetailWithCombatTrace(
                ReplayExpectation expected,
                StateSequenceBuilder.ActionType actualActionType,
                StateSnapshot actualState
        ) {
            String detail = sourceContextDetail(expected, actualActionType, actualState);
            if (lastCombatTraceOrdinal >= 0
                    && expected != null
                    && lastCombatTraceOrdinal == expected.ordinal) {
                return appendDetail(detail, "combat_trace=" + lastCombatTrace);
            }
            return detail;
        }

        private boolean maybePassSourceSkippedStackPriority(Game game) {
            if (game == null || game.isSimulation() || game.getStack() == null || game.getStack().isEmpty()) {
                return false;
            }
            if (eligibleOrdinal < 0 || eligibleOrdinal >= prefixChoices.size()
                    || (targetOrdinal >= 0 && eligibleOrdinal >= targetOrdinal)) {
                return false;
            }
            ReplayExpectation expected = prefixExpectationAt(eligibleOrdinal);
            int skippedRows = sourceDecisionGapBefore(eligibleOrdinal, expected);
            if (skippedRows <= 0 || !sourceStackResolvedPassExpected(expected, game)) {
                return false;
            }
            if (skippedStackPassOrdinal != eligibleOrdinal) {
                skippedStackPassOrdinal = eligibleOrdinal;
                skippedStackPassCount = 0;
            }
            StateSnapshot state = StateSnapshot.capture(game, this);
            if (skippedStackPassCount >= skippedRows) {
                recordPrefixDivergence(eligibleOrdinal, "skipped_stack_priority_unresolved",
                        expected.actionType, StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL,
                        prefixChoices.get(eligibleOrdinal), prefixChoiceTextsAt(eligibleOrdinal),
                        Collections.singletonList("Ability: Pass"), Collections.emptyList(), state,
                        appendDetail(sourceContextDetail(expected,
                                        StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL, state),
                                "source_decision_gap=" + skippedRows
                                        + " synthetic_stack_passes=" + skippedStackPassCount
                                        + " stack_top=" + traceStackTop(game)));
                return false;
            }
            skippedStackPassCount++;
            pass(game);
            return true;
        }

        private int sourceDecisionGapBefore(int ordinal, ReplayExpectation expected) {
            int current = sourceDecisionNumber(expected);
            if (current < 0 || ordinal <= 0) {
                return 0;
            }
            int previous = previousSourceDecisionNumberBefore(ordinal);
            return previous >= 0 ? Math.max(0, current - previous - 1) : 0;
        }

        private int previousSourceDecisionNumberBefore(int ordinal) {
            for (int i = ordinal - 1; i >= 0; i--) {
                int previous = sourceDecisionNumber(prefixExpectationAt(i));
                if (previous >= 0) {
                    return previous;
                }
            }
            return -1;
        }

        private int sourceDecisionNumber(ReplayExpectation expected) {
            if (expected == null || expected.sourceDecisionNumber == null
                    || !expected.sourceDecisionNumber.matches("\\d+")) {
                return -1;
            }
            try {
                return Integer.parseInt(expected.sourceDecisionNumber);
            } catch (NumberFormatException ignored) {
                return -1;
            }
        }

        private boolean maybePassSourceSkippedPreTargetGap(Game game) {
            if (game == null || game.isSimulation()
                    || eligibleOrdinal < 0) {
                return false;
            }
            ReplayExpectation expected = activePrefixExpectationAt(eligibleOrdinal);
            int previousSourceDecisionNumber = previousSourceDecisionNumberBefore(eligibleOrdinal);
            int skippedRows = sourceDecisionGapBefore(eligibleOrdinal, expected);
            if (!isSourceProvenD076D088PassGap(expected, previousSourceDecisionNumber, skippedRows)
                    || !sourceContextBeforeExpected(game, expected)) {
                return false;
            }
            if (skippedPreTargetPassOrdinal != eligibleOrdinal) {
                skippedPreTargetPassOrdinal = eligibleOrdinal;
                skippedPreTargetPassCount = 0;
            }
            StateSnapshot state = StateSnapshot.capture(game, this);
            int passLimit = skippedPreTargetPassLimit(skippedRows);
            if (skippedPreTargetPassCount >= passLimit) {
                appendPreTargetPassTrace(game, this, "synthetic_pre_target_stop",
                        expected, eligibleOrdinal, previousSourceDecisionNumber, skippedRows,
                        skippedPreTargetPassCount, "skipped_pre_target_pass_gap_unresolved",
                        opponentTranscriptCursor);
                recordPrefixDivergence(eligibleOrdinal, "skipped_pre_target_pass_gap_unresolved",
                        expected.actionType, StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL,
                        expectedIndicesAt(eligibleOrdinal, expected), expectedTextsAt(eligibleOrdinal, expected),
                        Collections.singletonList("Ability: Pass"), Collections.emptyList(), state,
                        appendDetail(sourceContextDetail(expected,
                                        StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL, state),
                                "source_decision_gap=" + skippedRows
                                        + " synthetic_pre_target_passes=" + skippedPreTargetPassCount
                                        + " synthetic_pre_target_pass_limit=" + passLimit
                                        + " stack_top=" + traceStackTop(game)),
                        expected);
                return false;
            }
            appendPreTargetPassTrace(game, this, "synthetic_pre_target_pass",
                    expected, eligibleOrdinal, previousSourceDecisionNumber, skippedRows,
                    skippedPreTargetPassCount, "source_context_before_expected",
                    opponentTranscriptCursor);
            skippedPreTargetPassCount++;
            pass(game);
            return true;
        }

        private int skippedPreTargetPassLimit(int skippedRows) {
            // v103 proved this bridge needs priority passes for opponent transcript and stack side effects,
            // not one pass per omitted compact source row. Keep the larger allowance local to this gap.
            return isSourceProvenD076D088PassGap(activePrefixExpectationAt(eligibleOrdinal),
                    previousSourceDecisionNumberBefore(eligibleOrdinal), skippedRows)
                    ? 64
                    : skippedRows;
        }

        private boolean maybePassD079PhaseAdvanceBridge(Game game) {
            if (game == null || game.isSimulation()
                    || eligibleOrdinal < 0
                    || eligibleOrdinal >= prefixChoices.size()
                    || (targetOrdinal >= 0 && eligibleOrdinal >= targetOrdinal)) {
                return false;
            }
            ReplayExpectation expected = prefixExpectationAt(eligibleOrdinal);
            if (!isSourceD079PostBlockersPass(expected)
                    || !sourceTurnMatchesExpected(expected, game)
                    || !"DECLARE_BLOCKERS".equals(contextKey(String.valueOf(game.getTurnStepType())))
                    || (game.getStack() != null && !game.getStack().isEmpty())
                    || !sourceD078RepeatedBlockersPassConsumed(eligibleOrdinal)) {
                return false;
            }
            if (d079PhaseAdvancePassOrdinal != eligibleOrdinal) {
                d079PhaseAdvancePassOrdinal = eligibleOrdinal;
                d079PhaseAdvancePassCount = 0;
            }
            if (d079PhaseAdvancePassCount >= 1) {
                return false;
            }
            appendPreTargetPassTrace(game, this, "synthetic_d079_phase_advance_pass",
                    expected, eligibleOrdinal, 78, 0, d079PhaseAdvancePassCount,
                    "d078_repeated_blockers_done_consumed", opponentTranscriptCursor);
            d079PhaseAdvancePassCount++;
            pass(game);
            return true;
        }

        private boolean isSourceD079PostBlockersPass(ReplayExpectation expected) {
            return expected != null
                    && expected.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && sourceDecisionNumber(expected) == 79
                    && expected.sourceAnchorId != null
                    && expected.sourceAnchorId.endsWith("_D079")
                    && "PlayerRL1".equals(expected.sourceActor)
                    && "POSTCOMBAT_MAIN".equals(sourceStepKey(expected.sourcePhase))
                    && isPassOrNoMoreActionExpected(expected);
        }

        private boolean sourceD078RepeatedBlockersPassConsumed(int ordinal) {
            if (ordinal < 2 || forcedPrefixCount < ordinal) {
                return false;
            }
            ReplayExpectation previous = prefixExpectationAt(ordinal - 1);
            ReplayExpectation beforePrevious = prefixExpectationAt(ordinal - 2);
            return previous != null
                    && beforePrevious != null
                    && sourceDecisionNumber(previous) == 78
                    && previous.sourceAnchorId != null
                    && previous.sourceAnchorId.endsWith("_D078")
                    && previous.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && "DECLARE_BLOCKERS".equals(sourceStepKey(previous.sourcePhase))
                    && "PlayerRL1".equals(previous.sourceActor)
                    && isPassOrNoMoreActionExpected(previous)
                    && sourceDecisionNumber(beforePrevious) == 77
                    && beforePrevious.sourceAnchorId != null
                    && beforePrevious.sourceAnchorId.endsWith("_D077")
                    && beforePrevious.actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    && "DECLARE_BLOCKERS".equals(sourceStepKey(beforePrevious.sourcePhase))
                    && "DECLARE_BLOCKS".equals(sourcePromptKey(beforePrevious.sourcePhase))
                    && "PlayerRL1".equals(beforePrevious.sourceActor);
        }

        private List<Integer> d070InsertedDeclareBlockersDoneChoice(
                StateSequenceBuilder.ActionType actualActionType,
                List<String> actualTexts,
                StateSnapshot actualState,
                Game game
        ) {
            int bridgeTargetOrdinal = d070BridgeTargetOrdinal();
            if (game == null || game.isSimulation()
                    || firstPrefixDivergence != null
                    || eligibleOrdinal < 1
                    || bridgeTargetOrdinal < 0
                    || eligibleOrdinal != bridgeTargetOrdinal
                    || actualActionType != StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    || !"DECLARE_BLOCKS".equals(actualPromptKey(actualActionType, actualState))
                    || !"DECLARE_BLOCKERS".equals(contextKey(String.valueOf(game.getTurnStepType())))
                    || (game.getStack() != null && !game.getStack().isEmpty())) {
                return null;
            }
            ReplayExpectation expected = activePrefixExpectationAt(eligibleOrdinal);
            ReplayExpectation previous = prefixExpectationAt(eligibleOrdinal - 1);
            if (!isSourceD070DreadReturnTarget(expected)
                    || !sourceD067DeclareBlockersConsumedBeforeD070(eligibleOrdinal, previous, expected)
                    || !d070InsertedBlockerSurfaceMatches(previous, actualTexts, actualState)) {
                return null;
            }
            if (d070InsertedBlockersDoneOrdinal != eligibleOrdinal) {
                d070InsertedBlockersDoneOrdinal = eligibleOrdinal;
                d070InsertedBlockersDoneCount = 0;
            }
            if (d070InsertedBlockersDoneCount >= 1) {
                return null;
            }
            int doneIdx = uniqueDoneCandidateIndex(actualTexts);
            return doneIdx < 0 ? null : Collections.singletonList(doneIdx);
        }

        private List<Integer> d035InsertedDeclareBlockersDoneChoice(
                StateSequenceBuilder.ActionType actualActionType,
                List<String> actualTexts,
                StateSnapshot actualState,
                Game game
        ) {
            if (game == null || game.isSimulation()
                    || firstPrefixDivergence != null
                    || eligibleOrdinal < 1
                    || actualActionType != StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    || !"DECLARE_BLOCKS".equals(actualPromptKey(actualActionType, actualState))
                    || !"DECLARE_BLOCKERS".equals(contextKey(String.valueOf(game.getTurnStepType())))
                    || (game.getStack() != null && !game.getStack().isEmpty())) {
                return null;
            }
            ReplayExpectation expected = activePrefixExpectationAt(eligibleOrdinal);
            ReplayExpectation previous = prefixExpectationAt(eligibleOrdinal - 1);
            if (!sourceD034DeclareBlockersConsumedBeforeD035Pass(previous, expected)
                    || !sourceTurnMatchesExpected(expected, game)
                    || !d035InsertedBlockerSurfaceMatches(previous, actualTexts, actualState)) {
                return null;
            }
            if (d035InsertedBlockersDoneOrdinal != eligibleOrdinal) {
                d035InsertedBlockersDoneOrdinal = eligibleOrdinal;
                d035InsertedBlockersDoneCount = 0;
            }
            if (d035InsertedBlockersDoneCount >= 1) {
                return null;
            }
            int doneIdx = uniqueDoneCandidateIndex(actualTexts);
            return doneIdx < 0 ? null : Collections.singletonList(doneIdx);
        }

        private int d070BridgeTargetOrdinal() {
            int activeTargetOrdinal = livePrefixTraceTarget != null && livePrefixTraceTarget.ordinal >= 0
                    ? livePrefixTraceTarget.ordinal
                    : targetOrdinal;
            int prefixOrdinal = d070BridgePrefixOrdinalBefore(activeTargetOrdinal);
            if (prefixOrdinal >= 0) {
                return prefixOrdinal;
            }
            return activeTargetOrdinal;
        }

        private int d070BridgePrefixOrdinalBefore(int activeTargetOrdinal) {
            if (activeTargetOrdinal < 0) {
                return -1;
            }
            int start = Math.min(activeTargetOrdinal, prefixExpectations.size() - 1);
            for (int ordinal = start; ordinal >= 0; ordinal--) {
                ReplayExpectation expected = prefixExpectationAt(ordinal);
                ReplayExpectation previous = prefixExpectationAt(ordinal - 1);
                if (isSourceD070DreadReturnTarget(expected)
                        && sourceD067DeclareBlockersConsumedBeforeD070(ordinal, previous, expected)) {
                    return ordinal;
                }
            }
            return -1;
        }

        private boolean sourceD070InsertedDeclareBlockersNoAdditionalBlockers(Game game) {
            int bridgeTargetOrdinal = d070BridgeTargetOrdinal();
            if (game == null || game.isSimulation()
                    || firstPrefixDivergence != null
                    || eligibleOrdinal < 1
                    || bridgeTargetOrdinal < 0
                    || eligibleOrdinal != bridgeTargetOrdinal
                    || !"DECLARE_BLOCKERS".equals(contextKey(String.valueOf(game.getTurnStepType())))
                    || (game.getStack() != null && !game.getStack().isEmpty())) {
                return false;
            }
            ReplayExpectation expected = activePrefixExpectationAt(eligibleOrdinal);
            ReplayExpectation previous = prefixExpectationAt(eligibleOrdinal - 1);
            if (!isSourceD070DreadReturnTarget(expected)
                    || !sourceD067DeclareBlockersConsumedBeforeD070(eligibleOrdinal, previous, expected)) {
                return false;
            }
            StateSnapshot state = StateSnapshot.capture(game, this);
            String compact = state.toCompactText();
            if (!compact.contains(previous.sourceSelectedText + "[blocking]")
                    || !compact.contains("Overgrown Battlement")
                    || compact.contains("Overgrown Battlement[blocking]")) {
                return false;
            }
            if (d070InsertedBlockersDoneOrdinal != eligibleOrdinal) {
                d070InsertedBlockersDoneOrdinal = eligibleOrdinal;
                d070InsertedBlockersDoneCount = 0;
            }
            if (d070InsertedBlockersDoneCount >= 1) {
                return false;
            }
            appendPreTargetPassTrace(game, this, "synthetic_d070_inserted_declare_blockers_done",
                    expected, eligibleOrdinal, previousSourceDecisionNumberBefore(eligibleOrdinal),
                    sourceDecisionGapBefore(eligibleOrdinal, expected), d070InsertedBlockersDoneCount,
                    "d067_declare_blockers_consumed_selectBlockers", opponentTranscriptCursor);
            d070InsertedBlockersDoneCount++;
            return true;
        }

        private boolean isSourceD070DreadReturnTarget(ReplayExpectation expected) {
            return expected != null
                    && expected.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && sourceDecisionNumber(expected) == 70
                    && expected.sourceAnchorId != null
                    && expected.sourceAnchorId.endsWith("_D070")
                    && "PlayerRL1".equals(expected.sourceActor)
                    && "PRECOMBAT_MAIN".equals(sourceStepKey(expected.sourcePhase))
                    && (textMatchesExpected("Cast Dread Return", expected.sourceSelectedText)
                    || textMatchesExpected("Cast Dread Return", expected.expectedText));
        }

        private boolean sourceD067DeclareBlockersConsumedBeforeD070(
                int ordinal,
                ReplayExpectation previous,
                ReplayExpectation expected
        ) {
            if (previous == null || expected == null) {
                return false;
            }
            return sourceDecisionNumber(previous) == 67
                    && previous.sourceAnchorId != null
                    && previous.sourceAnchorId.endsWith("_D067")
                    && previous.actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    && "DECLARE_BLOCKERS".equals(sourceStepKey(previous.sourcePhase))
                    && "DECLARE_BLOCKS".equals(sourcePromptKey(previous.sourcePhase))
                    && "PlayerRL1".equals(previous.sourceActor)
                    && textMatchesExpected("Gatecreeper Vine", previous.sourceSelectedText)
                    && sourceDecisionNumber(expected) == 70
                    && "6".equals(expected.sourceTurn)
                    && "5".equals(previous.sourceTurn);
        }

        private boolean sourceD034DeclareBlockersConsumedBeforeD035Pass(
                ReplayExpectation previous,
                ReplayExpectation expected
        ) {
            return previous != null
                    && expected != null
                    && sourceDecisionNumber(previous) == 34
                    && previous.sourceAnchorId != null
                    && previous.sourceAnchorId.endsWith("_D034")
                    && previous.actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    && "DECLARE_BLOCKERS".equals(sourceStepKey(previous.sourcePhase))
                    && "DECLARE_BLOCKS".equals(sourcePromptKey(previous.sourcePhase))
                    && "PlayerRL1".equals(previous.sourceActor)
                    && textMatchesExpected("Saruli Caretaker", previous.sourceSelectedText)
                    && sourceDecisionNumber(expected) == 35
                    && expected.sourceAnchorId != null
                    && expected.sourceAnchorId.endsWith("_D035")
                    && expected.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && "DECLARE_BLOCKERS".equals(sourceStepKey(expected.sourcePhase))
                    && "PlayerRL1".equals(expected.sourceActor)
                    && isPassOrNoMoreActionExpected(expected)
                    && "4".equals(previous.sourceTurn)
                    && "4".equals(expected.sourceTurn);
        }

        private boolean d070InsertedBlockerSurfaceMatches(
                ReplayExpectation previous,
                List<String> actualTexts,
                StateSnapshot actualState
        ) {
            if (previous == null || actualTexts == null || actualTexts.size() != 2) {
                return false;
            }
            int doneIdx = uniqueDoneCandidateIndex(actualTexts);
            if (doneIdx < 0) {
                return false;
            }
            int blockerIdx = doneIdx == 0 ? 1 : 0;
            String blockerText = actualTexts.get(blockerIdx);
            if (!textMatchesExpected("Overgrown Battlement", blockerText)) {
                return false;
            }
            return true;
        }

        private boolean d035InsertedBlockerSurfaceMatches(
                ReplayExpectation previous,
                List<String> actualTexts,
                StateSnapshot actualState
        ) {
            if (previous == null || actualTexts == null || actualTexts.size() != 2) {
                return false;
            }
            if (uniqueDoneCandidateIndex(actualTexts) < 0) {
                return false;
            }
            boolean blockerCandidatePresent = false;
            for (String text : actualTexts) {
                if (textMatchesExpected(previous.sourceSelectedText, text)) {
                    blockerCandidatePresent = true;
                    break;
                }
            }
            if (!blockerCandidatePresent || actualState == null || previous.sourceSelectedText == null) {
                return false;
            }
            return actualState.toCompactText().contains(previous.sourceSelectedText + "[blocking]");
        }

        private boolean isSourceProvenD076D088PassGap(
                ReplayExpectation expected,
                int previousSourceDecisionNumber,
                int skippedRows
        ) {
            // v101 proved this exact omitted source interval is pass-only; keep the bridge fail-closed.
            return expected != null
                    && expected.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && previousSourceDecisionNumber == 75
                    && sourceDecisionNumber(expected) == 89
                    && skippedRows == 13
                    && "PlayerRL1".equals(expected.sourceActor)
                    && (expected.sourceAnchorId.endsWith("_D089")
                    || textMatchesExpected("Cast Masked Vandal", expected.sourceSelectedText)
                    || textMatchesExpected("Cast Masked Vandal", expected.expectedText));
        }

        private boolean sourceContextBeforeExpected(Game game, ReplayExpectation expected) {
            if (game == null || expected == null || expected.sourceTurn == null
                    || !expected.sourceTurn.matches("\\d+")) {
                return false;
            }
            int expectedTurn;
            try {
                expectedTurn = Integer.parseInt(expected.sourceTurn);
            } catch (NumberFormatException ignored) {
                return false;
            }
            int actualTurn = compactSourceTurn(safeTurn(game));
            if (actualTurn < 0) {
                return false;
            }
            if (actualTurn < expectedTurn) {
                return true;
            }
            if (actualTurn > expectedTurn) {
                return false;
            }
            String expectedStep = sourceStepKey(expected.sourcePhase);
            String actualStep = contextKey(String.valueOf(game.getTurnStepType()));
            int expectedOrder = sourceStepOrder(expectedStep);
            int actualOrder = sourceStepOrder(actualStep);
            return expectedOrder >= 0 && actualOrder >= 0 && actualOrder < expectedOrder;
        }

        private int sourceStepOrder(String step) {
            if ("UPKEEP".equals(step)) {
                return 0;
            }
            if ("DRAW".equals(step)) {
                return 1;
            }
            if ("PRECOMBAT_MAIN".equals(step)) {
                return 2;
            }
            if ("BEGIN_COMBAT".equals(step)) {
                return 3;
            }
            if ("DECLARE_ATTACKERS".equals(step)) {
                return 4;
            }
            if ("DECLARE_BLOCKERS".equals(step)) {
                return 5;
            }
            if ("COMBAT_DAMAGE".equals(step)) {
                return 6;
            }
            if ("END_COMBAT".equals(step)) {
                return 7;
            }
            if ("POSTCOMBAT_MAIN".equals(step)) {
                return 8;
            }
            if ("END_TURN".equals(step)) {
                return 9;
            }
            if ("CLEANUP".equals(step)) {
                return 10;
            }
            return -1;
        }

        private boolean sourceStackResolvedPassExpected(ReplayExpectation expected, Game game) {
            if (expected == null
                    || expected.actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
                return false;
            }
            if (!expected.sourceTurn.isEmpty()
                    && expected.sourceTurn.matches("\\d+")
                    && !expected.sourceTurn.equals(String.valueOf(compactSourceTurn(safeTurn(game))))) {
                return false;
            }
            if (isPassText(expected.sourceSelectedText) || isPassText(expected.expectedText)) {
                return true;
            }
            if (expected.expectedTexts != null) {
                for (String text : expected.expectedTexts) {
                    if (isPassText(text)) {
                        return true;
                    }
                }
            }
            return expected.expectedIdx == 0
                    && (expected.sourceSelectedText.isEmpty() || isPassText(expected.sourceSelectedText))
                    && (expected.expectedText.isEmpty() || isPassText(expected.expectedText));
        }

        private boolean sourceSkippedNoAttackersPrompt(Game game, UUID attackingPlayerId) {
            if (game == null || game.isSimulation() || attackingPlayerId == null
                    || !attackingPlayerId.equals(getId())) {
                return false;
            }
            if (eligibleOrdinal < 0 || eligibleOrdinal >= prefixChoices.size()
                    || (targetOrdinal >= 0 && eligibleOrdinal >= targetOrdinal)) {
                return false;
            }
            ReplayExpectation expected = prefixExpectationAt(eligibleOrdinal);
            if (!sourceNoAttackersExpected(expected, game)) {
                return false;
            }
            if (combatHasAttackers(game)) {
                StateSnapshot state = StateSnapshot.capture(game, this);
                recordPrefixDivergence(eligibleOrdinal, "state_divergence", expected.actionType,
                        StateSequenceBuilder.ActionType.DECLARE_ATTACKS,
                        prefixChoices.get(eligibleOrdinal), prefixChoiceTextsAt(eligibleOrdinal),
                        Collections.emptyList(), Collections.emptyList(), state,
                        "source expected no attackers, but replay combat already had attackers before selectAttackers");
            }
            return true;
        }

        private boolean sourceNoAttackersExpected(ReplayExpectation expected, Game game) {
            if (expected == null || expected.actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || !"DECLARE_ATTACKERS".equals(sourceStepKey(expected.sourcePhase))) {
                return false;
            }
            if (!expected.sourceTurn.isEmpty()
                    && expected.sourceTurn.matches("\\d+")
                    && !expected.sourceTurn.equals(String.valueOf(compactSourceTurn(safeTurn(game))))) {
                return false;
            }
            if (isNoAttackersText(expected.sourceSelectedText)
                    || isNoAttackersText(expected.expectedText)) {
                return true;
            }
            if (expected.expectedTexts != null) {
                for (String text : expected.expectedTexts) {
                    if (isNoAttackersText(text)) {
                        return true;
                    }
                }
            }
            return expected.expectedIdx == 0
                    && (expected.sourceSelectedText.isEmpty() || isNoAttackersText(expected.sourceSelectedText))
                    && (expected.expectedText.isEmpty() || isNoAttackersText(expected.expectedText));
        }

        private List<Integer> repeatedDeclareBlockersPassChoice(
                int ordinal,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<String> actualTexts,
                StateSnapshot actualState,
                Game game
        ) {
            ReplayExpectation expected = prefixExpectationAt(ordinal);
            if (expected == null
                    || firstPrefixDivergence != null
                    || expectedActionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || actualActionType != StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    || !"DECLARE_BLOCKERS".equals(sourceStepKey(expected.sourcePhase))
                    || !"DECLARE_BLOCKS".equals(actualPromptKey(actualActionType, actualState))
                    || !sourceTurnMatchesExpected(expected, game)
                    || !isPassOrNoMoreActionExpected(expected)
                    || !sourceDeclareBlocksConsumedBeforePass(ordinal, expected)) {
                return null;
            }
            int doneIdx = uniqueDoneCandidateIndex(actualTexts);
            if (doneIdx < 0) {
                return null;
            }
            return Collections.singletonList(doneIdx);
        }

        private boolean sourceTurnMatchesExpected(ReplayExpectation expected, Game game) {
            if (expected == null || game == null || expected.sourceTurn.isEmpty()) {
                return false;
            }
            if (!expected.sourceTurn.matches("\\d+")) {
                return false;
            }
            return expected.sourceTurn.equals(String.valueOf(compactSourceTurn(safeTurn(game))));
        }

        private boolean sourceDeclareBlocksConsumedBeforePass(int ordinal, ReplayExpectation expected) {
            if (ordinal <= 0 || forcedPrefixCount < ordinal) {
                return false;
            }
            ReplayExpectation previous = prefixExpectationAt(ordinal - 1);
            if (previous == null
                    || previous.actionType != StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    || !"DECLARE_BLOCKERS".equals(sourceStepKey(previous.sourcePhase))
                    || !"DECLARE_BLOCKS".equals(sourcePromptKey(previous.sourcePhase))) {
                return false;
            }
            int previousSourceDecision = sourceDecisionNumber(previous);
            int expectedSourceDecision = sourceDecisionNumber(expected);
            return previousSourceDecision >= 0
                    && expectedSourceDecision == previousSourceDecision + 1;
        }

        private boolean isPassOrNoMoreActionExpected(ReplayExpectation expected) {
            if (expected == null) {
                return false;
            }
            if (isPassOrNoMoreActionText(expected.sourceSelectedText)
                    || isPassOrNoMoreActionText(expected.expectedText)) {
                return true;
            }
            if (expected.expectedTexts != null) {
                for (String text : expected.expectedTexts) {
                    if (isPassOrNoMoreActionText(text)) {
                        return true;
                    }
                }
            }
            return expected.expectedIdx == 0
                    && (expected.sourceSelectedText.isEmpty()
                    || isPassOrNoMoreActionText(expected.sourceSelectedText))
                    && (expected.expectedText.isEmpty()
                    || isPassOrNoMoreActionText(expected.expectedText));
        }

        private boolean isPassOrNoMoreActionText(String text) {
            if (isPassText(text)) {
                return true;
            }
            String normalized = contextKey(text);
            return "DONE".equals(normalized)
                    || "NO_MORE_ACTIONS".equals(normalized)
                    || "NO_MORE_BLOCKERS".equals(normalized)
                    || "NO_BLOCKERS".equals(normalized);
        }

        private int uniqueDoneCandidateIndex(List<String> actualTexts) {
            int doneIdx = -1;
            if (actualTexts == null) {
                return doneIdx;
            }
            for (int i = 0; i < actualTexts.size(); i++) {
                if (!"DONE".equals(contextKey(actualTexts.get(i)))) {
                    continue;
                }
                if (doneIdx >= 0) {
                    return -1;
                }
                doneIdx = i;
            }
            return doneIdx;
        }

        private boolean isNoAttackersText(String text) {
            if (isPassText(text)) {
                return true;
            }
            String normalized = text == null
                    ? ""
                    : text.trim().replaceAll("[^A-Za-z0-9]+", "_")
                    .replaceAll("_+", "_")
                    .replaceAll("^_|_$", "")
                    .toUpperCase(Locale.ROOT);
            return "DONE".equals(normalized)
                    || "NONE".equals(normalized)
                    || "NO_ATTACK".equals(normalized)
                    || "NO_ATTACKS".equals(normalized)
                    || "NO_ATTACKERS".equals(normalized)
                    || "DECLARE_ATTACKERS_NONE".equals(normalized);
        }

        private boolean combatHasAttackers(Game game) {
            if (game == null || game.getCombat() == null) {
                return false;
            }
            for (CombatGroup group : game.getCombat().getGroups()) {
                if (group != null && group.getAttackers() != null && !group.getAttackers().isEmpty()) {
                    return true;
                }
            }
            return false;
        }

        private void recordPrefixDivergence(
                int ordinal,
                String reason,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> expectedIndices,
                List<String> expectedTexts,
                List<String> actualCandidateTexts,
                List<Integer> actualSelectedIndices,
                StateSnapshot state,
                String detail
        ) {
            recordPrefixDivergence(ordinal, reason, expectedActionType, actualActionType, expectedIndices,
                    expectedTexts, actualCandidateTexts, actualSelectedIndices, state, detail,
                    prefixExpectationAt(ordinal));
        }

        private void recordPrefixDivergence(
                int ordinal,
                String reason,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> expectedIndices,
                List<String> expectedTexts,
                List<String> actualCandidateTexts,
                List<Integer> actualSelectedIndices,
                StateSnapshot state,
                String detail,
                ReplayExpectation sourceExpectation
        ) {
            if (firstPrefixDivergence != null) {
                return;
            }
            firstPrefixDivergence = new PrefixDivergence(
                    ordinal,
                    reason,
                    expectedActionType,
                    actualActionType,
                    expectedIndices,
                    expectedTexts,
                    actualCandidateTexts,
                    actualSelectedIndices,
                    actualSelectedTexts(actualSelectedIndices, actualCandidateTexts),
                    forcedPrefixCount,
                    state,
                    detail,
                    sourceExpectation);
        }

        private PrefixDivergence prefixDivergenceOrEnd(int actualDecisionCount, StateSnapshot finalState) {
            if (firstPrefixDivergence != null) {
                return firstPrefixDivergence;
            }
            PrefixDivergence opponentMismatch = opponentTranscriptDivergenceOrEnd(actualDecisionCount, finalState);
            if (opponentMismatch != null) {
                return opponentMismatch;
            }
            if (prefixChoices.isEmpty() || actualDecisionCount >= prefixChoices.size()) {
                return null;
            }
            return new PrefixDivergence(
                    actualDecisionCount,
                    "state_divergence",
                    prefixActionTypeAt(actualDecisionCount),
                    null,
                    actualDecisionCount >= 0 && actualDecisionCount < prefixChoices.size()
                            ? prefixChoices.get(actualDecisionCount)
                            : Collections.emptyList(),
                    prefixChoiceTextsAt(actualDecisionCount),
                    Collections.emptyList(),
                    Collections.emptyList(),
                    Collections.emptyList(),
                    forcedPrefixCount,
                    finalState,
                    "replay ended before the next forced-prefix ordinal",
                    prefixExpectationAt(actualDecisionCount));
        }

        private PrefixDivergence opponentTranscriptDivergenceOrEnd(int actualDecisionCount, StateSnapshot finalState) {
            if (opponentTranscriptCursor == null) {
                return null;
            }
            int ordinalLimit = prefixChoices.size();
            if (!opponentTranscriptCursor.hasMismatchAtOrBefore(ordinalLimit)) {
                return null;
            }
            int ordinal = opponentTranscriptCursor.mismatchOrdinalOr(ordinalLimit);
            if (ordinal < 0) {
                ordinal = Math.max(0, Math.min(actualDecisionCount, ordinalLimit));
            }
            ReplayExpectation expected = activePrefixExpectationAt(ordinal);
            StateSequenceBuilder.ActionType expectedActionType = expected == null
                    ? prefixActionTypeAt(ordinal)
                    : expected.actionType;
            return opponentTranscriptCursor.toPrefixDivergence(
                    ordinal,
                    expectedActionType,
                    expectedIndicesAt(ordinal, expected),
                    expectedTextsAt(ordinal, expected),
                    forcedPrefixCount,
                    finalState,
                    expected);
        }

        private String prefixFailureReason(List<Integer> requested, List<String> requestedTexts,
                                           List<String> actualTexts, int maxTargets, int minTargets) {
            int count = actualTexts == null ? 0 : actualTexts.size();
            int requestedCount = requested == null ? 0 : requested.size();
            int textCount = requestedTexts == null ? 0 : requestedTexts.size();
            boolean hasExpectedText = false;
            boolean missingExpectedText = false;
            boolean invalidIndex = false;
            boolean textMismatchAtIndex = false;
            int ranks = Math.max(requestedCount, textCount);
            for (int rank = 0; rank < ranks; rank++) {
                Integer idx = rank < requestedCount ? requested.get(rank) : null;
                String expectedText = rank < textCount ? normalizeText(requestedTexts.get(rank)) : "";
                if (idx == null || idx < 0 || idx >= count) {
                    invalidIndex = true;
                }
                if (!expectedText.isEmpty()) {
                    hasExpectedText = true;
                    boolean found = false;
                    for (String actualText : actualTexts == null ? Collections.<String>emptyList() : actualTexts) {
                        if (textMatchesExpected(expectedText, actualText)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        missingExpectedText = true;
                    }
                    if (idx != null && idx >= 0 && idx < count
                            && !textMatchesExpected(expectedText, actualTexts.get(idx))) {
                        textMismatchAtIndex = true;
                    }
                }
            }
            if (invalidIndex && !hasExpectedText) {
                return "invalid_index";
            }
            if (missingExpectedText) {
                return "missing_candidate";
            }
            if (textMismatchAtIndex) {
                return "text_mismatch";
            }
            if (invalidIndex) {
                return "invalid_index";
            }
            if (Math.max(0, minTargets) > 0) {
                return "missing_candidate";
            }
            return "state_divergence";
        }

        private String prefixFailureDetail(List<Integer> requested, List<String> requestedTexts,
                                           List<String> actualTexts, int maxTargets, int minTargets) {
            int count = actualTexts == null ? 0 : actualTexts.size();
            return "candidate_count=" + count
                    + " requested_indices=" + joinInts(requested)
                    + " requested_texts=" + joinTexts(requestedTexts)
                    + " min_targets=" + minTargets
                    + " max_targets=" + maxTargets;
        }

        private <T> List<Integer> sanitize(List<Integer> requested, List<String> requestedTexts,
                                            List<T> candidates, List<String> actualTexts,
                                            int maxTargets, int minTargets, Game game) {
            int count = candidates == null ? 0 : candidates.size();
            List<Integer> out = new ArrayList<>();
            int requestedCount = requested == null ? 0 : requested.size();
            int textCount = requestedTexts == null ? 0 : requestedTexts.size();
            int ranks = Math.max(requestedCount, textCount);
            for (int rank = 0; rank < ranks; rank++) {
                Integer idx = rank < requestedCount ? requested.get(rank) : null;
                String expectedText = rank < textCount ? normalizeText(requestedTexts.get(rank)) : "";
                if (idx != null && idx >= 0 && idx < count && !out.contains(idx)) {
                    String actualText = idx < actualTexts.size() ? actualTexts.get(idx) : "";
                    T candidate = candidates.get(idx);
                    if (textMatchesExpectedForCandidate(expectedText, actualText, candidate, game)) {
                        out.add(idx);
                    }
                }
                if (!expectedText.isEmpty() && (idx == null || !out.contains(idx))) {
                    for (int candidateIdx = 0; candidateIdx < count; candidateIdx++) {
                        if (out.contains(candidateIdx)) {
                            continue;
                        }
                        String actualText = candidateIdx < actualTexts.size() ? actualTexts.get(candidateIdx) : "";
                        T candidate = candidates.get(candidateIdx);
                        if (textMatchesExpectedForCandidate(expectedText, actualText, candidate, game)) {
                            out.add(candidateIdx);
                            break;
                        }
                    }
                }
                if (out.size() >= Math.max(1, maxTargets)) {
                    break;
                }
            }
            return out.size() >= Math.max(0, minTargets) ? out : null;
        }

        private boolean textMatchesExpectedForCandidate(
                String expectedText,
                String actualText,
                Object candidate,
                Game game
        ) {
            if (ActionCounterfactualTrainer.textMatchesExpected(expectedText, actualText)) {
                return true;
            }
            StandaloneRoleSuffix expected = standaloneRoleSuffix(expectedText);
            if (!expected.valid()) {
                return false;
            }
            String actual = normalizeText(actualText);
            StandaloneRoleSuffix actualSuffix = standaloneRoleSuffix(actualText);
            String actualBase = actualSuffix.valid() ? actualSuffix.base : actual;
            if (!expected.base.equals(actualBase) && !actualBase.endsWith(": " + expected.base)) {
                return false;
            }
            String role = candidateControllerRole(candidate, game);
            return expected.role.equals(role);
        }

        private String candidateControllerRole(Object candidate, Game game) {
            UUID controllerId = candidateControllerId(candidate, game);
            if (controllerId == null) {
                return "";
            }
            if (controllerId.equals(getId())) {
                return "agent";
            }
            return "opponent";
        }

        private UUID candidateControllerId(Object candidate, Game game) {
            if (candidate == null || game == null) {
                return null;
            }
            if (candidate instanceof Permanent) {
                return ((Permanent) candidate).getControllerId();
            }
            if (candidate instanceof Card) {
                return ((Card) candidate).getOwnerId();
            }
            if (candidate instanceof Player) {
                return ((Player) candidate).getId();
            }
            UUID id = null;
            if (candidate instanceof UUID) {
                id = (UUID) candidate;
            } else if (candidate instanceof String) {
                try {
                    id = UUID.fromString(((String) candidate).trim());
                } catch (IllegalArgumentException ignored) {
                    id = null;
                }
            }
            if (id == null) {
                return null;
            }
            Permanent permanent = game.getPermanent(id);
            if (permanent != null) {
                return permanent.getControllerId();
            }
            Player player = game.getPlayer(id);
            if (player != null) {
                return player.getId();
            }
            Card card = game.getCard(id);
            return card == null ? null : card.getOwnerId();
        }

        private <T> List<String> snapshotCandidateTexts(List<T> candidates, Game game) {
            List<String> out = new ArrayList<>();
            if (candidates == null) {
                return out;
            }
            for (T candidate : candidates) {
                String text = describeCandidate(candidate, game)
                        .replace('\r', ' ')
                        .replace('\n', ' ')
                        .trim();
                if (text.length() > 300) {
                    text = text.substring(0, 300);
                }
                out.add(text);
            }
            return out;
        }

        private List<Integer> tacticChoice(
                StateSequenceBuilder.ActionType actionType,
                List<String> texts,
                StateSnapshot snapshot,
                int maxTargets,
                int minTargets
        ) {
            if (texts == null || texts.isEmpty()) {
                return null;
            }
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < texts.size(); i++) {
                if (preferredTacticPriority(actionType, texts.get(i), snapshot) < Integer.MAX_VALUE) {
                    indices.add(i);
                }
            }
            if (indices.isEmpty()) {
                return null;
            }
            indices.sort(Comparator
                    .comparingInt((Integer idx) -> preferredTacticPriority(actionType, texts.get(idx), snapshot))
                    .thenComparingInt(Integer::intValue));
            int wanted = Math.max(1, Math.max(0, minTargets));
            if (maxTargets > 0) {
                wanted = Math.min(wanted, maxTargets);
            }
            List<Integer> out = new ArrayList<>();
            for (Integer idx : indices) {
                if (out.size() >= wanted) {
                    break;
                }
                out.add(idx);
            }
            return out.size() >= Math.max(0, minTargets) ? out : null;
        }

        private String describeCandidate(Object candidate, Game game) {
            if (candidate == null) {
                return "STOP";
            }
            if (candidate instanceof Boolean) {
                return Boolean.TRUE.equals(candidate) ? "YES" : "NO";
            }
            if (candidate instanceof Integer) {
                return "X=" + candidate;
            }
            if (candidate instanceof UUID && game != null) {
                UUID id = (UUID) candidate;
                Player player = game.getPlayer(id);
                if (player != null) {
                    return "Player:" + player.getName();
                }
                MageObject object = game.getObject(id);
                if (object != null) {
                    return object.getName();
                }
                Card card = game.getCard(id);
                if (card != null) {
                    return card.getName();
                }
                return id.toString();
            }
            if (candidate instanceof String && game != null) {
                String text = ((String) candidate).trim();
                try {
                    UUID id = UUID.fromString(text);
                    MageObject object = game.getObject(id);
                    if (object != null) {
                        return object.getName();
                    }
                    Card card = game.getCard(id);
                    if (card != null) {
                        return card.getName();
                    }
                } catch (IllegalArgumentException ignored) {
                }
            }
            if (candidate instanceof Card) {
                return ((Card) candidate).getName();
            }
            if (candidate instanceof mage.abilities.Mode) {
                mage.abilities.Mode mode = (mage.abilities.Mode) candidate;
                try {
                    String text = mode.getEffects().getText(mode);
                    if (text != null && !text.trim().isEmpty()) {
                        return text;
                    }
                } catch (Exception ignored) {
                }
                return "Mode:" + mode.getId();
            }
            if (candidate instanceof mage.abilities.Ability) {
                mage.abilities.Ability ability = (mage.abilities.Ability) candidate;
                MageObject source = game == null ? null : game.getObject(ability.getSourceId());
                String sourceName = source == null ? "Ability" : source.getName();
                return sourceName + ": " + ability.toString();
            }
            return String.valueOf(candidate);
        }

        boolean wasTargetForced() {
            return targetForced;
        }

        private void captureValueProbe(Game game, int ordinal) {
            valueProbeCaptured = true;
            valueProbeTerminal = false;
            valueProbeOrdinal = ordinal;
            valueProbeState = StateSnapshot.capture(game, this);
            valueProbeScore = scoreStateValue(game);
        }

        private float scoreStateValue(Game game) {
            try {
                StateSequenceBuilder.SequenceOutput state =
                        StateSequenceBuilder.buildBaseState(
                                game,
                                game.getPhase() != null ? game.getPhase().getType()
                                        : mage.constants.TurnPhase.BEGINNING,
                                StateSequenceBuilder.MAX_LEN);
                int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
                int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
                int[] dummyIds = new int[maxCand];
                float[][] dummyFeats = new float[maxCand][candDim];
                int[] dummyMask = new int[maxCand];
                dummyMask[0] = 1;
                PythonMLBatchManager.PredictionResult pred = RLTrainer.sharedModel.scoreCandidates(
                        state, dummyIds, dummyFeats, dummyMask,
                        "branch_value_probe", "action", 0, 0, 0);
                float v = pred == null ? 0.0f : pred.valueScores;
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    return 0.0f;
                }
                return Math.max(-1.0f, Math.min(1.0f, v));
            } catch (Throwable ignored) {
                return Float.NaN;
            }
        }

        boolean wasValueProbeCaptured() {
            return valueProbeCaptured;
        }

        BranchValueProbeResult valueProbeResult(boolean timedOut, String error) {
            return new BranchValueProbeResult(
                    valueProbeCaptured,
                    valueProbeTerminal,
                    valueProbeScore,
                    valueProbeOrdinal,
                    valueProbeState,
                    error,
                    targetForced,
                    timedOut);
        }

        List<List<String>> getCandidateTextsByOrdinal() {
            return new ArrayList<>(candidateTextsByOrdinal);
        }

        List<StateSnapshot> getStateSnapshotsByOrdinal() {
            return new ArrayList<>(stateSnapshotsByOrdinal);
        }

        int getForcedPrefixCount() {
            return forcedPrefixCount;
        }

        String getFirstMulliganHandText() {
            return firstMulliganHandText;
        }

        boolean terminalMilestoneReached() {
            return terminalMilestoneReached;
        }

        boolean turnLimitReached() {
            return turnLimitReached;
        }

        String getFirstPriorityHandText() {
            return firstPriorityHandText;
        }

        private void maybeRestoreExactFirstPriorityOpeningState(Game game) {
            if (exactOpeningStateRestored
                    || game == null
                    || exactOpeningHandNames.isEmpty()
                    || exactOpeningLibraryNames.isEmpty()
                    || exactOpeningHandNames.size() >= 7) {
                return;
            }
            exactOpeningStateRestored = true;
            try {
                List<Card> currentHand = new ArrayList<>(getHand().getCards(game));
                List<Card> currentLibrary = new ArrayList<>(getLibrary().getCards(game));
                if (namesMatch(currentHand, exactOpeningHandNames)
                        && namesMatch(currentLibrary, exactOpeningLibraryNames)) {
                    return;
                }

                List<Card> remaining = new ArrayList<>();
                remaining.addAll(currentHand);
                remaining.addAll(currentLibrary);
                List<Card> restoredHand = takeCardsByName(remaining, exactOpeningHandNames);
                List<Card> restoredLibrary = takeCardsByName(remaining, exactOpeningLibraryNames);
                if (restoredHand.size() != exactOpeningHandNames.size()
                        || restoredLibrary.size() != exactOpeningLibraryNames.size()) {
                    return;
                }

                getHand().clear();
                for (Card card : restoredHand) {
                    card.setZone(Zone.HAND, game);
                    getHand().add(card);
                }
                LinkedHashSet<Card> orderedLibrary = new LinkedHashSet<>();
                orderedLibrary.addAll(restoredLibrary);
                orderedLibrary.addAll(remaining);
                getLibrary().clear();
                getLibrary().addAll(orderedLibrary, game);
                firstPriorityHandText = "";
            } catch (Exception ignored) {
                // Replay diagnostics should not change normal gameplay failure modes.
            }
        }

        private static boolean namesMatch(List<Card> cards, List<String> names) {
            if (cards == null || names == null || cards.size() != names.size()) {
                return false;
            }
            for (int i = 0; i < names.size(); i++) {
                Card card = cards.get(i);
                if (card == null || !normalizeName(card.getName()).equals(normalizeName(names.get(i)))) {
                    return false;
                }
            }
            return true;
        }

        private static List<Card> takeCardsByName(List<Card> remaining, List<String> names) {
            List<Card> out = new ArrayList<>();
            if (remaining == null || names == null) {
                return out;
            }
            for (String name : names) {
                String wanted = normalizeName(name);
                if (wanted.isEmpty()) {
                    continue;
                }
                int found = -1;
                for (int i = 0; i < remaining.size(); i++) {
                    Card card = remaining.get(i);
                    if (card != null && normalizeName(card.getName()).equals(wanted)) {
                        found = i;
                        break;
                    }
                }
                if (found < 0) {
                    return Collections.emptyList();
                }
                out.add(remaining.remove(found));
            }
            return out;
        }

        private void captureFirstPriorityHand(Game game) {
            if (!firstPriorityHandText.isEmpty() || game == null) {
                return;
            }
            try {
                firstPriorityHandText = getHand().getCards(game).stream()
                        .map(card -> card == null ? "" : card.getName())
                        .filter(name -> name != null && !name.isEmpty())
                        .collect(Collectors.joining("|"));
            } catch (Exception ignored) {
                firstPriorityHandText = "";
            }
        }

        private void captureFirstMulliganHand(Game game) {
            if (!firstMulliganHandText.isEmpty() || game == null) {
                return;
            }
            try {
                firstMulliganHandText = getHand().getCards(game).stream()
                        .map(card -> card == null ? "" : card.getName())
                        .filter(name -> name != null && !name.isEmpty())
                        .collect(Collectors.joining("|"));
            } catch (Exception ignored) {
                firstMulliganHandText = "";
            }
        }

        private boolean checkTerminalMilestone(Game game) {
            if (!ActionCounterfactualTrainer.terminalMilestoneReached(game, getId(), terminalMode)) {
                return false;
            }
            terminalMilestoneReached = true;
            try {
                game.end();
            } catch (Exception ignored) {
            }
            return true;
        }

        private boolean checkTurnLimit(Game game) {
            if (maxGameTurns <= 0 || game == null) {
                return false;
            }
            try {
                if (game.getTurnNum() <= maxGameTurns) {
                    return false;
                }
                turnLimitReached = true;
                game.end();
                return true;
            } catch (Exception ignored) {
                return false;
            }
        }
    }

    private static final class DecisionPoint {

        final int ordinal;
        final StateSequenceBuilder.TrainingData trainingData;
        final float[] probs;
        final List<Integer> chosenIndices;
        final List<String> candidateTexts;
        final StateSnapshot stateSnapshot;

        private DecisionPoint(int ordinal, StateSequenceBuilder.TrainingData trainingData,
                              float[] probs, List<String> candidateTexts, StateSnapshot stateSnapshot) {
            this.ordinal = ordinal;
            this.trainingData = trainingData;
            this.probs = probs == null ? new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES] : probs;
            this.candidateTexts = candidateTexts == null ? Collections.emptyList() : new ArrayList<>(candidateTexts);
            this.stateSnapshot = stateSnapshot == null ? StateSnapshot.EMPTY : stateSnapshot;
            this.chosenIndices = new ArrayList<>();
            for (int i = 0; i < Math.min(trainingData.chosenCount, trainingData.chosenIndices.length); i++) {
                int idx = trainingData.chosenIndices[i];
                if (idx >= 0) {
                    chosenIndices.add(idx);
                }
            }
        }

        int baselineIdx() {
            return chosenIndices.isEmpty() ? -1 : chosenIndices.get(0);
        }

        String candidateText(int idx) {
            return idx >= 0 && idx < candidateTexts.size() ? candidateTexts.get(idx) : "";
        }
    }

    private static final class BaselineResult {

        final List<DecisionPoint> decisions;
        final List<List<Integer>> prefixChoices;
        final List<List<String>> prefixChoiceTexts;
        final boolean won;
        final boolean timedOut;
        final String error;
        final int turns;

        private BaselineResult(List<DecisionPoint> decisions, List<List<Integer>> prefixChoices,
                               List<List<String>> prefixChoiceTexts,
                               boolean won, boolean timedOut, String error, int turns) {
            this.decisions = decisions;
            this.prefixChoices = prefixChoices;
            this.prefixChoiceTexts = prefixChoiceTexts == null ? Collections.emptyList() : prefixChoiceTexts;
            this.won = won;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.turns = turns;
        }

        static BaselineResult failed(String error) {
            return new BaselineResult(Collections.emptyList(), Collections.emptyList(), Collections.emptyList(),
                    false, false, error, -1);
        }
    }

    private static final class BranchResult {

        final int forcedIdx;
        final boolean won;
        final boolean timedOut;
        final String error;
        final boolean forcedApplied;
        final int turns;
        final TrajectoryTrainingEpisode trajectoryEpisode;
        final int trajectoryRawRecords;
        final int trajectoryKeptRecords;
        final String trajectoryDropReason;

        private BranchResult(int forcedIdx, boolean won, boolean timedOut, String error, boolean forcedApplied, int turns) {
            this(forcedIdx, won, timedOut, error, forcedApplied, turns, null, 0, 0, "");
        }

        private BranchResult(int forcedIdx, boolean won, boolean timedOut, String error,
                             boolean forcedApplied, int turns,
                             TrajectoryTrainingEpisode trajectoryEpisode) {
            this(forcedIdx, won, timedOut, error, forcedApplied, turns, trajectoryEpisode, 0, 0, "");
        }

        private BranchResult(int forcedIdx, boolean won, boolean timedOut, String error,
                             boolean forcedApplied, int turns,
                             TrajectoryTrainingEpisode trajectoryEpisode,
                             int trajectoryRawRecords,
                             int trajectoryKeptRecords,
                             String trajectoryDropReason) {
            this.forcedIdx = forcedIdx;
            this.won = won;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.forcedApplied = forcedApplied;
            this.turns = turns;
            this.trajectoryEpisode = trajectoryEpisode;
            this.trajectoryRawRecords = Math.max(0, trajectoryRawRecords);
            this.trajectoryKeptRecords = Math.max(0, trajectoryKeptRecords);
            this.trajectoryDropReason = trajectoryDropReason == null ? "" : trajectoryDropReason;
        }

        static BranchResult failed(int forcedIdx, String error, boolean timedOut) {
            return new BranchResult(forcedIdx, false, timedOut, error, false, -1);
        }
    }

    private static final class BranchTrajectoryBuildResult {

        final TrajectoryTrainingEpisode episode;
        final int rawRecords;
        final int keptRecords;
        final String dropReason;

        private BranchTrajectoryBuildResult(
                TrajectoryTrainingEpisode episode,
                int rawRecords,
                int keptRecords,
                String dropReason
        ) {
            this.episode = episode;
            this.rawRecords = Math.max(0, rawRecords);
            this.keptRecords = Math.max(0, keptRecords);
            this.dropReason = dropReason == null ? "" : dropReason;
        }

        static BranchTrajectoryBuildResult empty(String reason) {
            return new BranchTrajectoryBuildResult(null, 0, 0, reason);
        }

        static BranchTrajectoryBuildResult dropped(int rawRecords, String reason) {
            return new BranchTrajectoryBuildResult(null, rawRecords, 0, reason);
        }

        static BranchTrajectoryBuildResult built(
                TrajectoryTrainingEpisode episode,
                int rawRecords,
                int keptRecords
        ) {
            return new BranchTrajectoryBuildResult(episode, rawRecords, keptRecords, "");
        }
    }

    private static final class StateSnapshot {

        static final StateSnapshot EMPTY = new StateSnapshot("", -1, "", "", -1, "");

        private final String compactText;
        private final int turn;
        private final String phaseText;
        private final String stepText;
        private final int stackCount;
        private final String stackTop;

        private StateSnapshot(String compactText) {
            this(compactText, -1, "", "", -1, "");
        }

        private StateSnapshot(String compactText, int turn, String phaseText, String stepText,
                              int stackCount, String stackTop) {
            this.compactText = compactText == null ? "" : compactText;
            this.turn = turn;
            this.phaseText = phaseText == null ? "" : phaseText;
            this.stepText = stepText == null ? "" : stepText;
            this.stackCount = stackCount;
            this.stackTop = stackTop == null ? "" : stackTop;
        }

        static StateSnapshot capture(Game game, Player player) {
            if (game == null || player == null) {
                return EMPTY;
            }
            try {
                int turn = safeTurn(game);
                String phase = String.valueOf(game.getTurnPhaseType());
                String step = String.valueOf(game.getTurnStepType());
                int stackCount = game.getStack() == null ? -1 : game.getStack().size();
                String stackTop = traceStackTop(game);
                StringBuilder sb = new StringBuilder();
                sb.append("turn=").append(turn);
                sb.append(";phase=").append(phase);
                sb.append(";step=").append(step);
                sb.append(";active=").append(playerName(game, game.getActivePlayerId()));
                sb.append(";priority=").append(playerName(game, game.getPriorityPlayerId()));
                sb.append(";life=").append(player.getLife());
                sb.append(";opponentLife=").append(opponentLife(game, player.getId()));
                sb.append(";lands=").append(player.getLandsPlayed()).append('/').append(player.getLandsPerTurn());
                sb.append(";canPlayLand=").append(player.canPlayLand());
                sb.append(";mana=").append(manaText(player));
                sb.append(";hand=").append(cardNames(player.getHand().getCards(game), game, 12));
                sb.append(";battlefield=").append(permanentNames(game, player.getId(), 16));
                sb.append(";battlefieldDetail=").append(permanentNamesDetailed(game, player.getId(), 16));
                sb.append(";stackObjectCount=").append(stackCount);
                sb.append(";stackTop=").append(compactStateValue(stackTop));
                sb.append(";stack=").append(stackNames(game, player.getId(), 8));
                sb.append(";battlefieldCreatures=").append(controlledCreatureCount(game, player.getId()));
                sb.append(";graveyard=").append(cardNames(player.getGraveyard().getCards(game), game, 20));
                sb.append(";graveyardCreatures=").append(graveyardCreatureCount(player, game));
                sb.append(";graveyardHasDread=").append(graveyardContains(player, game, "Dread Return"));
                sb.append(";graveyardHasLotleth=").append(graveyardContains(player, game, "Lotleth Giant"));
                Player opponent = firstOpponent(game, player.getId());
                if (opponent != null) {
                    sb.append(";opponentHandSize=").append(opponent.getHand().size());
                    sb.append(";opponentBattlefield=").append(permanentNamesDetailed(game, opponent.getId(), 20));
                    sb.append(";opponentGraveyard=").append(cardNames(opponent.getGraveyard().getCards(game), game, 20));
                    sb.append(";opponentExile=").append(cardNames(game.getExile().getCardsOwned(game, opponent.getId()), game, 20));
                }
                List<Card> library = player.getLibrary().getCards(game);
                sb.append(";librarySize=").append(library.size());
                sb.append(";libraryTrueLands=").append(countLands(library, game));
                sb.append(";libraryLandGrants=").append(countNamed(library, "Land Grant"));
                sb.append(";handLandGrants=").append(countNamed(player.getHand().getCards(game), "Land Grant"));
                sb.append(";libraryTop=").append(cardNames(library, game, 12));
                return new StateSnapshot(sb.toString(), turn, phase, step, stackCount, stackTop);
            } catch (Exception e) {
                return new StateSnapshot("snapshot_error=" + exceptionSummary(e));
            }
        }

        String toCompactText() {
            return compactText;
        }

        int turn() {
            return turn;
        }

        String phaseText() {
            return phaseText;
        }

        String stepText() {
            return stepText;
        }

        int stackCount() {
            return stackCount;
        }

        String stackTop() {
            return stackTop;
        }

        private static String compactStateValue(String value) {
            if (value == null) {
                return "";
            }
            return value.replace('\r', ' ').replace('\n', ' ').replace(';', ',');
        }

        private static String playerName(Game game, UUID playerId) {
            try {
                Player p = playerId == null ? null : game.getPlayer(playerId);
                return p == null ? "" : p.getName();
            } catch (Exception ignored) {
                return "";
            }
        }

        private static String manaText(Player player) {
            try {
                mage.players.ManaPool pool = player.getManaPool();
                return "R" + pool.getRed()
                        + "|G" + pool.getGreen()
                        + "|B" + pool.getBlack()
                        + "|U" + pool.getBlue()
                        + "|W" + pool.getWhite()
                        + "|C" + pool.getColorless();
            } catch (Exception ignored) {
                return "";
            }
        }

        private static String permanentNames(Game game, UUID playerId, int limit) {
            try {
                List<String> names = new ArrayList<>();
                for (Permanent permanent : game.getBattlefield().getAllActivePermanents(playerId)) {
                    if (permanent != null && permanent.getName() != null && !permanent.getName().isEmpty()) {
                        names.add(permanent.getName());
                    }
                    if (names.size() >= limit) {
                        break;
                    }
                }
                return String.join("|", names);
            } catch (Exception ignored) {
                return "";
            }
        }

        private static String permanentNamesDetailed(Game game, UUID playerId, int limit) {
            try {
                List<String> names = new ArrayList<>();
                for (Permanent permanent : game.getBattlefield().getAllActivePermanents(playerId)) {
                    if (permanent != null && permanent.getName() != null && !permanent.getName().isEmpty()) {
                        StringBuilder name = new StringBuilder(permanent.getName());
                        List<String> flags = new ArrayList<>();
                        if (permanent.isTapped()) {
                            flags.add("tapped");
                        }
                        if (permanent.isAttacking()) {
                            flags.add("attacking");
                        }
                        if (permanent.getBlocking() > 0) {
                            flags.add("blocking");
                        }
                        if (!flags.isEmpty()) {
                            name.append('[').append(String.join(",", flags)).append(']');
                        }
                        names.add(name.toString());
                    }
                    if (names.size() >= limit) {
                        break;
                    }
                }
                return String.join("|", names);
            } catch (Exception ignored) {
                return "";
            }
        }

        private static Player firstOpponent(Game game, UUID playerId) {
            try {
                for (UUID opponentId : game.getOpponents(playerId, true)) {
                    Player opponent = game.getPlayer(opponentId);
                    if (opponent != null) {
                        return opponent;
                    }
                }
            } catch (Exception ignored) {
            }
            return null;
        }

        private static String cardNames(Collection<Card> cards, Game game, int limit) {
            try {
                List<String> names = new ArrayList<>();
                if (cards != null) {
                    for (Card card : cards) {
                        if (card != null && card.getName() != null && !card.getName().isEmpty()) {
                            names.add(card.getName());
                        }
                        if (names.size() >= limit) {
                            break;
                        }
                    }
                }
                return String.join("|", names);
            } catch (Exception ignored) {
                return "";
            }
        }

        private static int countLands(Collection<Card> cards, Game game) {
            int count = 0;
            try {
                if (cards != null) {
                    for (Card card : cards) {
                        if (card != null && card.isLand(game)) {
                            count++;
                        }
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }

        private static int countNamed(Collection<Card> cards, String name) {
            int count = 0;
            try {
                if (cards != null) {
                    for (Card card : cards) {
                        if (card != null && nameEquals(card.getName(), name)) {
                            count++;
                        }
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }
    }

    private static final class PrefixDivergence {

        final int ordinal;
        final String reason;
        final StateSequenceBuilder.ActionType expectedActionType;
        final StateSequenceBuilder.ActionType actualActionType;
        final List<Integer> expectedIndices;
        final List<String> expectedTexts;
        final List<Integer> actualCandidateIndices;
        final List<String> actualCandidateTexts;
        final List<Integer> actualSelectedIndices;
        final List<String> actualSelectedTexts;
        final int forcedPrefixCount;
        final StateSnapshot state;
        final String detail;
        final String sourceDecisionNumber;
        final String sourceAnchorId;
        final String sourceTurn;
        final String sourcePhase;
        final String sourceActor;

        private PrefixDivergence(
                int ordinal,
                String reason,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> expectedIndices,
                List<String> expectedTexts,
                List<String> actualCandidateTexts,
                List<Integer> actualSelectedIndices,
                List<String> actualSelectedTexts,
                int forcedPrefixCount,
                StateSnapshot state,
                String detail
        ) {
            this(ordinal, reason, expectedActionType, actualActionType, expectedIndices, expectedTexts,
                    actualCandidateTexts, actualSelectedIndices, actualSelectedTexts, forcedPrefixCount,
                    state, detail, null);
        }

        private PrefixDivergence(
                int ordinal,
                String reason,
                StateSequenceBuilder.ActionType expectedActionType,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> expectedIndices,
                List<String> expectedTexts,
                List<String> actualCandidateTexts,
                List<Integer> actualSelectedIndices,
                List<String> actualSelectedTexts,
                int forcedPrefixCount,
                StateSnapshot state,
                String detail,
                ReplayExpectation sourceExpectation
        ) {
            this.ordinal = ordinal;
            this.reason = reason == null ? "" : reason;
            this.expectedActionType = expectedActionType;
            this.actualActionType = actualActionType;
            this.expectedIndices = expectedIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(expectedIndices);
            this.expectedTexts = expectedTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(expectedTexts);
            this.actualCandidateTexts = actualCandidateTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actualCandidateTexts);
            this.actualCandidateIndices = candidateIndices(this.actualCandidateTexts);
            this.actualSelectedIndices = actualSelectedIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actualSelectedIndices);
            this.actualSelectedTexts = actualSelectedTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(actualSelectedTexts);
            this.forcedPrefixCount = forcedPrefixCount;
            this.state = state == null ? StateSnapshot.EMPTY : state;
            this.detail = detail == null ? "" : detail;
            this.sourceDecisionNumber = sourceExpectation == null ? "" : sourceExpectation.sourceDecisionNumber;
            this.sourceAnchorId = sourceExpectation == null ? "" : sourceExpectation.sourceAnchorId;
            this.sourceTurn = sourceExpectation == null ? "" : sourceExpectation.sourceTurn;
            this.sourcePhase = sourceExpectation == null ? "" : sourceExpectation.sourcePhase;
            this.sourceActor = sourceExpectation == null ? "" : sourceExpectation.sourceActor;
        }
    }

    private static final class PrefixContextFailure {

        final String reason;
        final String detail;

        private PrefixContextFailure(String reason, String detail) {
            this.reason = reason == null ? "" : reason;
            this.detail = detail == null ? "" : detail;
        }
    }

    private static final class PrefixObjectChoice {

        final List<Integer> indices;
        final PrefixContextFailure failure;

        private PrefixObjectChoice(List<Integer> indices, PrefixContextFailure failure) {
            this.indices = indices == null ? Collections.emptyList() : new ArrayList<>(indices);
            this.failure = failure;
        }

        static PrefixObjectChoice forced(List<Integer> indices) {
            return new PrefixObjectChoice(indices, null);
        }

        static PrefixObjectChoice failure(PrefixContextFailure failure) {
            return new PrefixObjectChoice(Collections.emptyList(), failure);
        }
    }

    private static final class PrefixRunResult {

        final List<DecisionPoint> decisions;
        final boolean won;
        final boolean timedOut;
        final String error;
        final int forcedAppliedCount;
        final int turns;
        final String firstMulliganHand;
        final String firstPriorityHand;
        final StateSnapshot finalState;
        final PrefixDivergence prefixDivergence;
        final List<EngineDecisionCheckpoint> checkpoints;

        private PrefixRunResult(List<DecisionPoint> decisions, boolean won, boolean timedOut,
                                String error, int forcedAppliedCount, int turns,
                                String firstMulliganHand, String firstPriorityHand,
                                StateSnapshot finalState) {
            this(decisions, won, timedOut, error, forcedAppliedCount, turns,
                    firstMulliganHand, firstPriorityHand, finalState, null, Collections.emptyList());
        }

        private PrefixRunResult(List<DecisionPoint> decisions, boolean won, boolean timedOut,
                                String error, int forcedAppliedCount, int turns,
                                String firstMulliganHand, String firstPriorityHand,
                                StateSnapshot finalState, PrefixDivergence prefixDivergence) {
            this(decisions, won, timedOut, error, forcedAppliedCount, turns,
                    firstMulliganHand, firstPriorityHand, finalState, prefixDivergence, Collections.emptyList());
        }

        private PrefixRunResult(List<DecisionPoint> decisions, boolean won, boolean timedOut,
                                String error, int forcedAppliedCount, int turns,
                                String firstMulliganHand, String firstPriorityHand,
                                StateSnapshot finalState, PrefixDivergence prefixDivergence,
                                List<EngineDecisionCheckpoint> checkpoints) {
            this.decisions = decisions == null ? Collections.emptyList() : decisions;
            this.won = won;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.forcedAppliedCount = forcedAppliedCount;
            this.turns = turns;
            this.firstMulliganHand = firstMulliganHand == null ? "" : firstMulliganHand;
            this.firstPriorityHand = firstPriorityHand == null ? "" : firstPriorityHand;
            this.finalState = finalState == null ? StateSnapshot.EMPTY : finalState;
            this.prefixDivergence = prefixDivergence;
            this.checkpoints = checkpoints == null ? Collections.emptyList() : new ArrayList<>(checkpoints);
        }

        static PrefixRunResult failed(String error) {
            return new PrefixRunResult(Collections.emptyList(), false, false, error, 0, -1, "", "", StateSnapshot.EMPTY);
        }
    }

    private static final class EngineDecisionCheckpoint {
        final ScenarioJob job;
        final Game gameSnapshot;
        final UUID playerId;
        final int ordinal;
        final String sourceDecisionNumber;
        final String sourceAnchorId;
        final StateSequenceBuilder.ActionType actionType;
        final List<String> candidateTexts;
        final List<Integer> sourceChosenIndices;
        final List<String> sourceChosenTexts;
        final StateSnapshot stateSnapshot;
        final RandomUtil.State randomState;
        final String candidateHash;
        final String stateHash;
        final String randomStateHash;

        private EngineDecisionCheckpoint(
                ScenarioJob job,
                Game gameSnapshot,
                UUID playerId,
                int ordinal,
                String sourceDecisionNumber,
                String sourceAnchorId,
                StateSequenceBuilder.ActionType actionType,
                List<String> candidateTexts,
                List<Integer> sourceChosenIndices,
                List<String> sourceChosenTexts,
                StateSnapshot stateSnapshot,
                RandomUtil.State randomState
        ) {
            this.job = job;
            this.gameSnapshot = gameSnapshot;
            this.playerId = playerId;
            this.ordinal = ordinal;
            this.sourceDecisionNumber = sourceDecisionNumber == null ? "" : sourceDecisionNumber;
            this.sourceAnchorId = sourceAnchorId == null ? "" : sourceAnchorId;
            this.actionType = actionType;
            this.candidateTexts = candidateTexts == null ? Collections.emptyList() : new ArrayList<>(candidateTexts);
            this.sourceChosenIndices = sourceChosenIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceChosenIndices);
            this.sourceChosenTexts = sourceChosenTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceChosenTexts);
            this.stateSnapshot = stateSnapshot == null ? StateSnapshot.EMPTY : stateSnapshot;
            this.randomState = randomState;
            this.candidateHash = sha256(String.join("\n", this.candidateTexts));
            this.stateHash = sha256(this.stateSnapshot.toCompactText());
            this.randomStateHash = randomState == null ? "" : randomState.fingerprint();
        }

        EngineDecisionCheckpoint withSourceDecision(DecisionPoint sourcePoint, ReplayExpectation expected) {
            List<Integer> chosen = expected == null || expected.expectedIndices.isEmpty()
                    ? Collections.emptyList()
                    : new ArrayList<>(expected.expectedIndices);
            List<String> chosenTexts = expected == null || expected.expectedTexts.isEmpty()
                    ? Collections.emptyList()
                    : new ArrayList<>(expected.expectedTexts);
            if (chosen.isEmpty() && sourcePoint != null) {
                chosen = new ArrayList<>(sourcePoint.chosenIndices);
            }
            if (chosenTexts.isEmpty() && sourcePoint != null) {
                chosenTexts = candidateTexts(sourcePoint, chosen);
            }
            if (chosenTexts.isEmpty() && expected != null && !expected.sourceSelectedText.isEmpty()) {
                chosenTexts = Collections.singletonList(expected.sourceSelectedText);
            }
            return new EngineDecisionCheckpoint(
                    job,
                    gameSnapshot,
                    playerId,
                    ordinal,
                    expected == null ? sourceDecisionNumber : expected.sourceDecisionNumber,
                    expected == null ? sourceAnchorId : expected.sourceAnchorId,
                    actionType,
                    candidateTexts,
                    chosen,
                    chosenTexts,
                    stateSnapshot,
                    randomState);
        }

        Game copyGame() {
            return gameSnapshot.createSimulationForAI();
        }

        String candidateText(int index) {
            return index >= 0 && index < candidateTexts.size() ? candidateTexts.get(index) : "";
        }
    }

    private static final class CheckpointReentryProbe {
        final StateSequenceBuilder.ActionType actionType;
        final List<String> candidateTexts;
        final List<Integer> selectedIndices;
        final List<String> selectedTexts;
        final StateSnapshot stateSnapshot;
        final String candidateHash;
        final String stateHash;

        private CheckpointReentryProbe(
                StateSequenceBuilder.ActionType actionType,
                List<String> candidateTexts,
                List<Integer> selectedIndices,
                List<String> selectedTexts,
                StateSnapshot stateSnapshot
        ) {
            this.actionType = actionType;
            this.candidateTexts = candidateTexts == null ? Collections.emptyList() : new ArrayList<>(candidateTexts);
            this.selectedIndices = selectedIndices == null ? Collections.emptyList() : new ArrayList<>(selectedIndices);
            this.selectedTexts = selectedTexts == null ? Collections.emptyList() : new ArrayList<>(selectedTexts);
            this.stateSnapshot = stateSnapshot == null ? StateSnapshot.EMPTY : stateSnapshot;
            this.candidateHash = sha256(String.join("\n", this.candidateTexts));
            this.stateHash = sha256(this.stateSnapshot.toCompactText());
        }
    }

    private static final class CheckpointContinuationResult {
        final String label;
        final CheckpointReentryProbe reentryProbe;
        final List<DecisionPoint> decisions;
        final Float terminalValue;
        final boolean timedOut;
        final String error;
        final StateSnapshot finalState;
        final int turns;

        private CheckpointContinuationResult(
                String label,
                CheckpointReentryProbe reentryProbe,
                List<DecisionPoint> decisions,
                Float terminalValue,
                boolean timedOut,
                String error,
                StateSnapshot finalState,
                int turns
        ) {
            this.label = label == null ? "" : label;
            this.reentryProbe = reentryProbe;
            this.decisions = decisions == null ? Collections.emptyList() : new ArrayList<>(decisions);
            this.terminalValue = terminalValue;
            this.timedOut = timedOut;
            this.error = error == null ? "" : error;
            this.finalState = finalState == null ? StateSnapshot.EMPTY : finalState;
            this.turns = turns;
        }

        static CheckpointContinuationResult skipped(String reason) {
            return new CheckpointContinuationResult("skipped", null, Collections.emptyList(),
                    null, false, reason, StateSnapshot.EMPTY, -1);
        }

        static CheckpointContinuationResult failed(String reason, boolean timedOut, ActionPlayer player, Game game) {
            return new CheckpointContinuationResult("failed",
                    player == null ? null : player.getLastCheckpointReentryProbe(),
                    Collections.emptyList(), terminalValueFor(game, player), timedOut, reason,
                    StateSnapshot.capture(game, player), game == null ? -1 : safeTurn(game));
        }

        boolean terminal() {
            return terminalValue != null && !timedOut;
        }

        boolean lost() {
            return terminal() && terminalValue < 0.0f;
        }

        boolean won() {
            return terminal() && terminalValue > 0.0f;
        }

        boolean reentryMatches(EngineDecisionCheckpoint checkpoint) {
            if (checkpoint == null || reentryProbe == null) {
                return false;
            }
            return reentryProbe.actionType == checkpoint.actionType
                    && normalizedTexts(reentryProbe.candidateTexts).equals(normalizedTexts(checkpoint.candidateTexts))
                    && reentryProbe.selectedIndices.equals(checkpoint.sourceChosenIndices)
                    && selectedTextsMatch(checkpoint);
        }

        private boolean selectedTextsMatch(EngineDecisionCheckpoint checkpoint) {
            return checkpoint.sourceChosenTexts.isEmpty()
                    || textListsMatchExpected(checkpoint.sourceChosenTexts, reentryProbe.selectedTexts);
        }
    }

    private static final class CheckpointBranchRecord {
        final ScenarioJob job;
        final ReplayExpectation expected;
        final EngineDecisionCheckpoint checkpoint;
        final int capturedCheckpointCount;
        final int sourceActualDecisions;
        final boolean sourceRunTimedOut;
        final String sourceRunError;
        final PrefixDivergence sourcePrefixDivergence;
        final boolean checkpointCaptured;
        final boolean sourceChoiceMatchesExpected;
        final boolean sourceChoiceReentryMatched;
        final CheckpointContinuationResult sourceChoiceA;
        final CheckpointContinuationResult sourceChoiceB;
        final CheckpointContinuationResult sourceTerminal;
        final int alternateIndex;
        final CheckpointContinuationResult alternateTerminal;
        final String classification;

        private CheckpointBranchRecord(
                ScenarioJob job,
                ReplayExpectation expected,
                EngineDecisionCheckpoint checkpoint,
                PrefixRunResult sourceRun,
                boolean sourceChoiceMatchesExpected,
                boolean sourceChoiceReentryMatched,
                CheckpointContinuationResult sourceChoiceA,
                CheckpointContinuationResult sourceChoiceB,
                CheckpointContinuationResult sourceTerminal,
                int alternateIndex,
                CheckpointContinuationResult alternateTerminal
        ) {
            this.job = job;
            this.expected = expected;
            this.checkpoint = checkpoint;
            this.capturedCheckpointCount = sourceRun == null ? 0 : sourceRun.checkpoints.size();
            this.sourceActualDecisions = sourceRun == null ? 0 : sourceRun.decisions.size();
            this.sourceRunTimedOut = sourceRun != null && sourceRun.timedOut;
            this.sourceRunError = sourceRun == null ? "" : sourceRun.error;
            this.sourcePrefixDivergence = sourceRun == null ? null : sourceRun.prefixDivergence;
            this.checkpointCaptured = checkpoint != null;
            this.sourceChoiceMatchesExpected = sourceChoiceMatchesExpected;
            this.sourceChoiceReentryMatched = sourceChoiceReentryMatched;
            this.sourceChoiceA = sourceChoiceA;
            this.sourceChoiceB = sourceChoiceB;
            this.sourceTerminal = sourceTerminal;
            this.alternateIndex = alternateIndex;
            this.alternateTerminal = alternateTerminal;
            this.classification = classify();
        }

        static CheckpointBranchRecord noCheckpoint(ScenarioJob job, ReplayExpectation expected, PrefixRunResult sourceRun) {
            return new CheckpointBranchRecord(job, expected, null, sourceRun,
                    false, false,
                    CheckpointContinuationResult.skipped("missing_checkpoint"),
                    CheckpointContinuationResult.skipped("missing_checkpoint"),
                    CheckpointContinuationResult.skipped("missing_checkpoint"),
                    -1,
                    CheckpointContinuationResult.skipped("missing_checkpoint"));
        }

        private String classify() {
            if (!checkpointCaptured) {
                if (sourcePrefixDivergence != null) {
                    return "source_prefix_divergence";
                }
                return "no_checkpoint";
            }
            if (!sourceChoiceReentryMatched) {
                return "checkpoint_reentry_mismatch";
            }
            if (!sourceChoiceMatchesExpected) {
                return "source_choice_mismatch";
            }
            if (sourceTerminal == null || !sourceTerminal.lost()) {
                return "source_terminal_not_loss";
            }
            if (alternateTerminal == null || !alternateTerminal.terminal()) {
                return "alternate_not_terminal";
            }
            if (alternateTerminal.won()) {
                return "correction_candidate";
            }
            if (alternateTerminal.lost()) {
                return "clean_negative";
            }
            return "clean_branch_pair";
        }

        String toCsv() {
            return (job == null ? -1 : job.scenario)
                    + "," + csv(job == null ? "" : job.agentDeck.getFileName().toString())
                    + "," + csv(job == null ? "" : job.oppDeck.getFileName().toString())
                    + "," + (job == null ? -1L : job.seed)
                    + "," + (expected == null ? -1 : expected.ordinal)
                    + "," + csv(expected == null ? "" : expected.sourceDecisionNumber)
                    + "," + csv(expected == null ? "" : expected.sourceAnchorId)
                    + "," + csv(expected == null || expected.actionType == null ? "" : expected.actionType.name())
                    + "," + checkpointCaptured
                    + "," + capturedCheckpointCount
                    + "," + sourceActualDecisions
                    + "," + sourceRunTimedOut
                    + "," + csv(sourceRunError)
                    + "," + prefixDivergenceFields(sourcePrefixDivergence)
                    + "," + sourceChoiceMatchesExpected
                    + "," + sourceChoiceReentryMatched
                    + "," + csv(classification)
                    + "," + csv(checkpoint == null || checkpoint.actionType == null ? "" : checkpoint.actionType.name())
                    + "," + (checkpoint == null ? -1 : checkpoint.candidateTexts.size())
                    + "," + csv(checkpoint == null ? "" : joinInts(checkpoint.sourceChosenIndices))
                    + "," + csv(checkpoint == null ? "" : joinTexts(checkpoint.sourceChosenTexts))
                    + "," + csv(checkpoint == null ? "" : indexedTexts(checkpoint.candidateTexts))
                    + "," + csv(checkpoint == null ? "" : checkpoint.candidateHash)
                    + "," + csv(checkpoint == null ? "" : checkpoint.stateHash)
                    + "," + csv(checkpoint == null ? "" : checkpoint.randomStateHash)
                    + "," + csv(reentryHash(sourceChoiceA))
                    + "," + csv(reentryHash(sourceChoiceB))
                    + "," + resultFields(sourceTerminal)
                    + "," + alternateIndex
                    + "," + csv(checkpoint == null ? "" : checkpoint.candidateText(alternateIndex))
                    + "," + resultFields(alternateTerminal);
        }

        private static String prefixDivergenceFields(PrefixDivergence divergence) {
            if (divergence == null) {
                return "-1"
                        + "," + csv("")
                        + "," + csv("")
                        + ","
                        + ","
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + ",-1"
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("")
                        + "," + csv("");
            }
            return divergence.ordinal
                    + "," + csv(divergence.reason)
                    + "," + csv(divergence.detail)
                    + "," + (divergence.expectedActionType == null ? "" : divergence.expectedActionType)
                    + "," + (divergence.actualActionType == null ? "" : divergence.actualActionType)
                    + "," + csv(joinInts(divergence.expectedIndices))
                    + "," + csv(joinTexts(divergence.expectedTexts))
                    + "," + csv(joinInts(divergence.actualCandidateIndices))
                    + "," + csv(indexedTexts(divergence.actualCandidateTexts))
                    + "," + csv(joinInts(divergence.actualSelectedIndices))
                    + "," + csv(joinTexts(divergence.actualSelectedTexts))
                    + "," + divergence.forcedPrefixCount
                    + "," + csv(divergence.sourceDecisionNumber)
                    + "," + csv(divergence.sourceAnchorId)
                    + "," + csv(divergence.sourceTurn)
                    + "," + csv(divergence.sourcePhase)
                    + "," + csv(divergence.sourceActor)
                    + "," + csv(divergence.state == null ? "" : divergence.state.toCompactText());
        }

        private static String reentryHash(CheckpointContinuationResult result) {
            return result == null || result.reentryProbe == null ? "" : result.reentryProbe.candidateHash;
        }

        private static String resultFields(CheckpointContinuationResult result) {
            if (result == null) {
                return "false,false,false,,-1," + csv("") + "," + csv("");
            }
            return result.terminal()
                    + "," + result.won()
                    + "," + result.lost()
                    + "," + (result.terminalValue == null
                    ? ""
                    : String.format(Locale.US, "%.6f", result.terminalValue))
                    + "," + result.turns
                    + "," + csv(result.error)
                    + "," + csv(result.finalState.toCompactText());
        }
    }

    private static final class PrefixNode {

        final int nodeId;
        final int parentId;
        final List<List<Integer>> prefixChoices;
        final List<List<String>> prefixChoiceTexts;
        final String expandedText;

        private PrefixNode(int nodeId, int parentId, List<List<Integer>> prefixChoices,
                           List<List<String>> prefixChoiceTexts, String expandedText) {
            this.nodeId = nodeId;
            this.parentId = parentId;
            this.prefixChoices = copyPrefix(prefixChoices);
            this.prefixChoiceTexts = copyPrefixTexts(prefixChoiceTexts);
            this.expandedText = expandedText == null ? "" : expandedText;
        }
    }

    private static final class PrefixBranchResult {

        final List<Integer> choice;
        final PrefixRunResult run;

        private PrefixBranchResult(List<Integer> choice, PrefixRunResult run) {
            this.choice = choice == null ? Collections.emptyList() : new ArrayList<>(choice);
            this.run = run;
        }
    }

    private static final class PrefixSubtreeResult {

        final boolean won;
        final int turns;
        final int nodes;

        private PrefixSubtreeResult(boolean won, int turns, int nodes) {
            this.won = won;
            this.turns = turns;
            this.nodes = nodes;
        }

        static PrefixSubtreeResult noWin() {
            return new PrefixSubtreeResult(false, -1, 0);
        }
    }

    private static final class SerializedTrainingDataFile implements Serializable {

        private static final long serialVersionUID = 1L;

        final List<StateSequenceBuilder.TrainingData> records;

        private SerializedTrainingDataFile(List<StateSequenceBuilder.TrainingData> records) {
            this.records = new ArrayList<>(records);
        }
    }

    private static final class SerializedTrajectoryDataFile implements Serializable {

        private static final long serialVersionUID = 1L;

        final List<TrajectoryTrainingEpisode> episodes;

        private SerializedTrajectoryDataFile(List<TrajectoryTrainingEpisode> episodes) {
            this.episodes = new ArrayList<>(episodes);
        }
    }

    private static final class TrajectoryTrainingEpisode implements Serializable {

        private static final long serialVersionUID = 1L;

        final String key;
        final List<StateSequenceBuilder.TrainingData> records;
        final List<Double> rewards;

        private TrajectoryTrainingEpisode(
                String key,
                List<StateSequenceBuilder.TrainingData> records,
                List<Double> rewards
        ) {
            this.key = key == null ? "" : key;
            this.records = records == null ? Collections.emptyList() : new ArrayList<>(records);
            this.rewards = rewards == null ? Collections.emptyList() : new ArrayList<>(rewards);
        }
    }

    private static final class TrajectoryImportStats {

        final int episodes;
        final int steps;
        final long trainPassSamples;

        private TrajectoryImportStats(int episodes, int steps, long trainPassSamples) {
            this.episodes = episodes;
            this.steps = steps;
            this.trainPassSamples = trainPassSamples;
        }
    }

    private enum BranchReturnLabel {
        POSITIVE,
        NEGATIVE,
        NONE
    }

    private static final class BranchReturnBalanceStats {

        int eligiblePositive;
        int eligibleNegative;
        int eligibleNone;
        int acceptedPositive;
        int acceptedNegative;
        int skippedNegative;
        int skippedNone;

        void addEligible(BranchReturnLabel label) {
            switch (label) {
                case POSITIVE:
                    eligiblePositive++;
                    break;
                case NEGATIVE:
                    eligibleNegative++;
                    break;
                default:
                    eligibleNone++;
                    break;
            }
        }

        void accept(BranchReturnLabel label) {
            switch (label) {
                case POSITIVE:
                    acceptedPositive++;
                    break;
                case NEGATIVE:
                    acceptedNegative++;
                    break;
                default:
                    break;
            }
        }

        int maxAcceptedNegative(int maxNegativePerPositive) {
            if (maxNegativePerPositive < 0) {
                return eligibleNegative;
            }
            long max = (long) eligiblePositive * (long) maxNegativePerPositive;
            return (int) Math.min((long) eligibleNegative, Math.max(0L, max));
        }
    }

    private static final class ImportTrainingStats {

        final int trainingExamples;
        final long trainPassSamples;
        final BranchReturnBalanceStats branchReturnBalanceStats;

        private ImportTrainingStats(
                int trainingExamples,
                long trainPassSamples,
                BranchReturnBalanceStats branchReturnBalanceStats
        ) {
            this.trainingExamples = trainingExamples;
            this.trainPassSamples = trainPassSamples;
            this.branchReturnBalanceStats = branchReturnBalanceStats;
        }

        static ImportTrainingStats unbalanced(int trainingExamples, int trainEpochs, int candidatePermutations) {
            long trainPassSamples = (long) trainingExamples
                    * (long) Math.max(1, trainEpochs)
                    * (long) Math.max(1, candidatePermutations);
            return new ImportTrainingStats(trainingExamples, trainPassSamples, null);
        }
    }

    private static final class TrainingExample {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final int baselineIdx;
        final String baselineText;
        final int bestIdx;
        final String bestText;
        final int branchCount;
        final StateSequenceBuilder.TrainingData trainingData;
        final List<String> candidateTexts;
        final List<Integer> choiceIndices;
        final List<String> choiceTexts;

        private TrainingExample(ScenarioJob job, DecisionPoint point,
                                StateSequenceBuilder.TrainingData trainingData,
                                int branchCount, boolean anyWin) {
            this.scenario = job.scenario;
            this.agentDeck = job.agentDeck;
            this.oppDeck = job.oppDeck;
            this.seed = job.seed;
            this.ordinal = point.ordinal;
            this.actionType = point.trainingData.actionType;
            this.candidateCount = point.trainingData.candidateCount;
            this.baselineIdx = point.baselineIdx();
            this.baselineText = point.candidateText(this.baselineIdx);
            this.bestIdx = trainingData.chosenIndices[0];
            this.bestText = point.candidateText(this.bestIdx);
            this.branchCount = branchCount;
            this.trainingData = trainingData;
            this.candidateTexts = new ArrayList<>(point.candidateTexts);
            this.choiceIndices = new ArrayList<>();
            this.choiceTexts = new ArrayList<>();
            for (int i = 0; i < Math.min(trainingData.chosenCount, trainingData.chosenIndices.length); i++) {
                int idx = trainingData.chosenIndices[i];
                if (idx >= 0) {
                    this.choiceIndices.add(idx);
                    this.choiceTexts.add(point.candidateText(idx));
                }
            }
        }

        String candidateText(int idx) {
            return idx >= 0 && idx < candidateTexts.size() ? candidateTexts.get(idx) : "";
        }

        String toCsv() {
            int nonzero = 0;
            int observed = 0;
            int positive = 0;
            int negative = 0;
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            boolean branchTargets = usesBranchReturnTargets(trainingData);
            if (trainingData.mctsVisitTargets != null) {
                for (float v : trainingData.mctsVisitTargets) {
                    if (v > 1e-6f) {
                        nonzero++;
                    }
                    if (observedTargetValue(v, branchTargets)) {
                        observed++;
                        if (v > 1e-6f) {
                            positive++;
                        } else if (v < -1e-6f) {
                            negative++;
                        }
                        min = Math.min(min, v);
                        max = Math.max(max, v);
                    }
                }
            }
            String minText = observed > 0 ? String.format(Locale.US, "%.4f", min) : "";
            String maxText = observed > 0 ? String.format(Locale.US, "%.4f", max) : "";
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + baselineIdx
                    + "," + csv(baselineText)
                    + "," + bestIdx
                    + "," + csv(bestText)
                    + "," + csv(joinInts(choiceIndices))
                    + "," + csv(joinTexts(choiceTexts))
                    + "," + branchCount
                    + "," + nonzero
                    + "," + observed
                    + "," + positive
                    + "," + negative
                    + "," + minText
                    + "," + maxText;
        }
    }

    private static final class WinningTrajectoryRecord {

        final String trajectoryKey;
        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final List<Integer> choiceIndices;
        final List<String> choiceTexts;
        final String state;
        final int turns;
        final String firstMulliganHand;
        final String firstPriorityHand;
        final String finalState;

        private WinningTrajectoryRecord(
                String trajectoryKey,
                ScenarioJob job,
                DecisionPoint point,
                List<Integer> choiceIndices,
                int turns,
                String firstMulliganHand,
                String firstPriorityHand,
                String finalState
        ) {
            this.trajectoryKey = trajectoryKey == null ? "" : trajectoryKey;
            this.scenario = job.scenario;
            this.agentDeck = job.agentDeck;
            this.oppDeck = job.oppDeck;
            this.seed = job.seed;
            this.ordinal = point.ordinal;
            this.actionType = point.trainingData.actionType;
            this.candidateCount = point.trainingData.candidateCount;
            this.choiceIndices = choiceIndices == null ? Collections.emptyList() : new ArrayList<>(choiceIndices);
            this.choiceTexts = new ArrayList<>();
            for (Integer idx : this.choiceIndices) {
                this.choiceTexts.add(idx == null ? "" : point.candidateText(idx));
            }
            this.state = point.stateSnapshot.toCompactText();
            this.turns = turns;
            this.firstMulliganHand = firstMulliganHand == null ? "" : firstMulliganHand;
            this.firstPriorityHand = firstPriorityHand == null ? "" : firstPriorityHand;
            this.finalState = finalState == null ? "" : finalState;
        }

        String toCsv() {
            return csv(trajectoryKey)
                    + "," + scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + csv(joinInts(choiceIndices))
                    + "," + csv(joinTexts(choiceTexts))
                    + "," + csv(state)
                    + "," + turns
                    + "," + csv(firstMulliganHand)
                    + "," + csv(firstPriorityHand)
                    + "," + csv(finalState);
        }
    }

    private static final class TensorReplayRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final int targetIdx;
        final String targetText;
        final int topIdx;
        final String topText;
        final float targetProb;
        final float topProb;
        final int targetRank;
        final boolean top1Match;
        final int validCount;
        final int stateHash;
        final int candidateHash;
        final float mctsTargetSum;
        final int cand0Id;
        final int cand1Id;
        final int cand0FeatHash;
        final int cand1FeatHash;
        final boolean cand01FeatEqual;

        private TensorReplayRecord(
                TrainingExample example,
                int targetIdx,
                int topIdx,
                float targetProb,
                float topProb,
                int targetRank,
                int validCount,
                int stateHash,
                int candidateHash,
                float mctsTargetSum,
                int cand0Id,
                int cand1Id,
                int cand0FeatHash,
                int cand1FeatHash,
                boolean cand01FeatEqual
        ) {
            this.scenario = example.scenario;
            this.agentDeck = example.agentDeck;
            this.oppDeck = example.oppDeck;
            this.seed = example.seed;
            this.ordinal = example.ordinal;
            this.actionType = example.actionType;
            this.candidateCount = example.candidateCount;
            this.targetIdx = targetIdx;
            this.targetText = example.candidateText(targetIdx);
            this.topIdx = topIdx;
            this.topText = example.candidateText(topIdx);
            this.targetProb = targetProb;
            this.topProb = topProb;
            this.targetRank = targetRank;
            this.top1Match = targetIdx >= 0 && targetIdx == topIdx;
            this.validCount = validCount;
            this.stateHash = stateHash;
            this.candidateHash = candidateHash;
            this.mctsTargetSum = mctsTargetSum;
            this.cand0Id = cand0Id;
            this.cand1Id = cand1Id;
            this.cand0FeatHash = cand0FeatHash;
            this.cand1FeatHash = cand1FeatHash;
            this.cand01FeatEqual = cand01FeatEqual;
        }

        static TensorReplayRecord from(TrainingExample example, float[] policy) {
            StateSequenceBuilder.TrainingData td = example.trainingData;
            int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            int targetIdx = -1;
            float targetMass = Float.NEGATIVE_INFINITY;
            float targetSum = 0.0f;
            boolean branchTargets = usesBranchReturnTargets(td);
            if (td.mctsVisitTargets != null) {
                for (int i = 0; i < max && i < td.mctsVisitTargets.length; i++) {
                    if (td.candidateMask[i] == 0 || !observedTargetValue(td.mctsVisitTargets[i], branchTargets)) {
                        continue;
                    }
                    targetSum += td.mctsVisitTargets[i];
                    if (td.mctsVisitTargets[i] > targetMass) {
                        targetMass = td.mctsVisitTargets[i];
                        targetIdx = i;
                    }
                }
            }
            if (targetIdx < 0 && td.chosenIndices.length > 0 && td.chosenIndices[0] >= 0) {
                targetIdx = td.chosenIndices[0];
            }

            int topIdx = -1;
            float topProb = Float.NEGATIVE_INFINITY;
            int validCount = 0;
            for (int i = 0; i < max && i < policy.length; i++) {
                if (td.candidateMask[i] == 0) {
                    continue;
                }
                validCount++;
                if (policy[i] > topProb) {
                    topProb = policy[i];
                    topIdx = i;
                }
            }
            float targetProb = targetIdx >= 0 && targetIdx < policy.length ? policy[targetIdx] : 0.0f;
            if (!Float.isFinite(topProb)) {
                topProb = 0.0f;
            }
            int rank = 0;
            if (targetIdx >= 0) {
                rank = 1;
                for (int i = 0; i < max && i < policy.length; i++) {
                    if (td.candidateMask[i] != 0 && i != targetIdx && policy[i] > targetProb) {
                        rank++;
                    }
                }
            }
            return new TensorReplayRecord(example, targetIdx, topIdx, targetProb, topProb, rank, validCount,
                    0, 0, targetSum, 0, 0, 0, 0, false);
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + targetIdx
                    + "," + csv(targetText)
                    + "," + topIdx
                    + "," + csv(topText)
                    + "," + String.format(Locale.US, "%.6f", targetProb)
                    + "," + String.format(Locale.US, "%.6f", topProb)
                    + "," + targetRank
                    + "," + top1Match
                    + "," + validCount;
        }
    }

    private static final class ScoreProbeStats {
        final int total;
        final int top1;
        final double accuracy;
        final int targetSetTop1;
        final double targetSetAccuracy;
        final int missingTarget;
        final double avgTargetProb;
        final double avgTopProb;
        final double avgTopTargetMass;
        final double avgRank;

        private ScoreProbeStats(
                int total,
                int top1,
                double accuracy,
                int targetSetTop1,
                double targetSetAccuracy,
                int missingTarget,
                double avgTargetProb,
                double avgTopProb,
                double avgTopTargetMass,
                double avgRank
        ) {
            this.total = total;
            this.top1 = top1;
            this.accuracy = accuracy;
            this.targetSetTop1 = targetSetTop1;
            this.targetSetAccuracy = targetSetAccuracy;
            this.missingTarget = missingTarget;
            this.avgTargetProb = avgTargetProb;
            this.avgTopProb = avgTopProb;
            this.avgTopTargetMass = avgTopTargetMass;
            this.avgRank = avgRank;
        }
    }

    private static final class ScoreProbeRecord {

        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final int targetIdx;
        final int topIdx;
        final float targetProb;
        final float topProb;
        final int targetRank;
        final boolean top1Match;
        final int validCount;
        final int stateHash;
        final int candidateHash;
        final float mctsTargetSum;
        final int cand0Id;
        final int cand1Id;
        final int cand0FeatHash;
        final int cand1FeatHash;
        final boolean cand01FeatEqual;
        final int targetPositiveCount;
        final float topTargetMass;
        final boolean targetSetMatch;
        final float targetQ;
        final float topQ;
        final int qTopIdx;
        final int qTargetRank;
        final float qTopTargetMass;
        final boolean qTop1Match;

        private ScoreProbeRecord(
                int ordinal,
                StateSequenceBuilder.ActionType actionType,
                int candidateCount,
                int targetIdx,
                int topIdx,
                float targetProb,
                float topProb,
                int targetRank,
                int validCount,
                int stateHash,
                int candidateHash,
                float mctsTargetSum,
                int cand0Id,
                int cand1Id,
                int cand0FeatHash,
                int cand1FeatHash,
                boolean cand01FeatEqual,
                int targetPositiveCount,
                float topTargetMass,
                boolean targetSetMatch,
                float targetQ,
                float topQ,
                int qTopIdx,
                int qTargetRank,
                float qTopTargetMass
        ) {
            this.ordinal = ordinal;
            this.actionType = actionType;
            this.candidateCount = candidateCount;
            this.targetIdx = targetIdx;
            this.topIdx = topIdx;
            this.targetProb = targetProb;
            this.topProb = topProb;
            this.targetRank = targetRank;
            this.top1Match = targetIdx >= 0 && targetIdx == topIdx;
            this.validCount = validCount;
            this.stateHash = stateHash;
            this.candidateHash = candidateHash;
            this.mctsTargetSum = mctsTargetSum;
            this.cand0Id = cand0Id;
            this.cand1Id = cand1Id;
            this.cand0FeatHash = cand0FeatHash;
            this.cand1FeatHash = cand1FeatHash;
            this.cand01FeatEqual = cand01FeatEqual;
            this.targetPositiveCount = targetPositiveCount;
            this.topTargetMass = topTargetMass;
            this.targetSetMatch = targetSetMatch;
            this.targetQ = targetQ;
            this.topQ = topQ;
            this.qTopIdx = qTopIdx;
            this.qTargetRank = qTargetRank;
            this.qTopTargetMass = qTopTargetMass;
            this.qTop1Match = targetIdx >= 0 && targetIdx == qTopIdx;
        }

        static ScoreProbeRecord from(int ordinal, StateSequenceBuilder.TrainingData td, float[] policy) {
            return from(ordinal, td, policy, null);
        }

        static ScoreProbeRecord from(int ordinal, StateSequenceBuilder.TrainingData td, float[] policy, float[] candidateQ) {
            int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            int targetIdx = -1;
            float targetMass = Float.NEGATIVE_INFINITY;
            float targetSum = 0.0f;
            int targetPositiveCount = 0;
            boolean branchTargets = usesBranchReturnTargets(td);
            if (td.mctsVisitTargets != null) {
                for (int i = 0; i < max && i < td.mctsVisitTargets.length; i++) {
                    if (td.candidateMask[i] == 0 || !observedTargetValue(td.mctsVisitTargets[i], branchTargets)) {
                        continue;
                    }
                    targetSum += td.mctsVisitTargets[i];
                    if (td.mctsVisitTargets[i] > 1e-6f) {
                        targetPositiveCount++;
                    }
                    if (td.mctsVisitTargets[i] > targetMass) {
                        targetMass = td.mctsVisitTargets[i];
                        targetIdx = i;
                    }
                }
            }
            if (targetIdx < 0 && td.chosenIndices.length > 0 && td.chosenIndices[0] >= 0) {
                targetIdx = td.chosenIndices[0];
            }
            int topIdx = -1;
            float topProb = Float.NEGATIVE_INFINITY;
            int validCount = 0;
            for (int i = 0; i < max && i < policy.length; i++) {
                if (td.candidateMask[i] == 0) {
                    continue;
                }
                validCount++;
                if (policy[i] > topProb) {
                    topProb = policy[i];
                    topIdx = i;
                }
            }
            if (!Float.isFinite(topProb)) {
                topProb = 0.0f;
            }
            float targetProb = targetIdx >= 0 && targetIdx < policy.length ? policy[targetIdx] : 0.0f;
            int rank = 0;
            if (targetIdx >= 0) {
                rank = 1;
                for (int i = 0; i < max && i < policy.length; i++) {
                    if (td.candidateMask[i] != 0 && i != targetIdx && policy[i] > targetProb) {
                        rank++;
                    }
                }
            }
            float topTargetMass = 0.0f;
            if (td.mctsVisitTargets != null && topIdx >= 0 && topIdx < td.mctsVisitTargets.length) {
                topTargetMass = td.mctsVisitTargets[topIdx];
            } else if (targetIdx >= 0 && topIdx == targetIdx) {
                topTargetMass = 1.0f;
            }
            boolean targetSetMatch = observedTargetValue(topTargetMass, branchTargets);
            int qTopIdx = -1;
            float topQ = Float.NEGATIVE_INFINITY;
            if (candidateQ != null) {
                for (int i = 0; i < max && i < candidateQ.length; i++) {
                    if (td.candidateMask[i] == 0) {
                        continue;
                    }
                    float q = candidateQ[i];
                    if (Float.isFinite(q) && q > topQ) {
                        topQ = q;
                        qTopIdx = i;
                    }
                }
            }
            if (!Float.isFinite(topQ)) {
                topQ = 0.0f;
            }
            float targetQ = targetIdx >= 0 && candidateQ != null && targetIdx < candidateQ.length
                    ? candidateQ[targetIdx]
                    : 0.0f;
            int qRank = 0;
            if (targetIdx >= 0 && candidateQ != null && targetIdx < candidateQ.length) {
                qRank = 1;
                for (int i = 0; i < max && i < candidateQ.length; i++) {
                    if (td.candidateMask[i] != 0 && i != targetIdx && candidateQ[i] > targetQ) {
                        qRank++;
                    }
                }
            }
            float qTopTargetMass = 0.0f;
            if (td.mctsVisitTargets != null && qTopIdx >= 0 && qTopIdx < td.mctsVisitTargets.length) {
                qTopTargetMass = td.mctsVisitTargets[qTopIdx];
            }
            return new ScoreProbeRecord(ordinal, td.actionType, td.candidateCount,
                    targetIdx, topIdx, targetProb, topProb, rank, validCount,
                    hashState(td), hashCandidates(td), targetSum,
                    candidateId(td, 0), candidateId(td, 1),
                    candidateFeatureHash(td, 0), candidateFeatureHash(td, 1),
                    candidateFeaturesEqual(td, 0, 1), targetPositiveCount,
                    topTargetMass, targetSetMatch,
                    targetQ, topQ, qTopIdx, qRank, qTopTargetMass);
        }

        private static int candidateId(StateSequenceBuilder.TrainingData td, int idx) {
            return td != null && td.candidateActionIds != null && idx >= 0 && idx < td.candidateActionIds.length
                    ? td.candidateActionIds[idx]
                    : 0;
        }

        private static int candidateFeatureHash(StateSequenceBuilder.TrainingData td, int idx) {
            return td != null && td.candidateFeatures != null && idx >= 0 && idx < td.candidateFeatures.length
                    ? Arrays.hashCode(td.candidateFeatures[idx])
                    : 0;
        }

        private static boolean candidateFeaturesEqual(StateSequenceBuilder.TrainingData td, int a, int b) {
            return td != null
                    && td.candidateFeatures != null
                    && a >= 0
                    && b >= 0
                    && a < td.candidateFeatures.length
                    && b < td.candidateFeatures.length
                    && Arrays.equals(td.candidateFeatures[a], td.candidateFeatures[b]);
        }

        private static int hashState(StateSequenceBuilder.TrainingData td) {
            if (td == null || td.state == null || td.state.getTokenIds() == null) {
                return 0;
            }
            return Arrays.hashCode(td.state.getTokenIds());
        }

        private static int hashCandidates(StateSequenceBuilder.TrainingData td) {
            if (td == null || td.candidateFeatures == null || td.candidateActionIds == null || td.candidateMask == null) {
                return 0;
            }
            int hash = 17;
            int max = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            for (int i = 0; i < max; i++) {
                if (td.candidateMask[i] == 0) {
                    continue;
                }
                hash = 31 * hash + td.candidateActionIds[i];
                hash = 31 * hash + Arrays.hashCode(td.candidateFeatures[i]);
            }
            return hash;
        }

        String toCsv() {
            return ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + targetIdx
                    + "," + topIdx
                    + "," + String.format(Locale.US, "%.6f", targetProb)
                    + "," + String.format(Locale.US, "%.6f", topProb)
                    + "," + targetRank
                    + "," + top1Match
                    + "," + validCount
                    + "," + stateHash
                    + "," + candidateHash
                    + "," + String.format(Locale.US, "%.6f", mctsTargetSum)
                    + "," + cand0Id
                    + "," + cand1Id
                    + "," + cand0FeatHash
                    + "," + cand1FeatHash
                    + "," + cand01FeatEqual
                    + "," + targetPositiveCount
                    + "," + String.format(Locale.US, "%.6f", topTargetMass)
                    + "," + targetSetMatch
                    + "," + String.format(Locale.US, "%.6f", targetQ)
                    + "," + String.format(Locale.US, "%.6f", topQ)
                    + "," + qTopIdx
                    + "," + qTargetRank
                    + "," + String.format(Locale.US, "%.6f", qTopTargetMass)
                    + "," + qTop1Match;
        }
    }

    private static final class TensorReplayStats {

        final int total;
        final int top1Matches;
        final double targetProbSum;
        final double topProbSum;
        final double targetRankSum;
        final int ranked;

        private TensorReplayStats(int total, int top1Matches,
                                  double targetProbSum, double topProbSum,
                                  double targetRankSum, int ranked) {
            this.total = total;
            this.top1Matches = top1Matches;
            this.targetProbSum = targetProbSum;
            this.topProbSum = topProbSum;
            this.targetRankSum = targetRankSum;
            this.ranked = ranked;
        }

        static TensorReplayStats from(List<TensorReplayRecord> records) {
            int total = records == null ? 0 : records.size();
            int matches = 0;
            double targetProb = 0.0;
            double topProb = 0.0;
            double rank = 0.0;
            int ranked = 0;
            if (records != null) {
                for (TensorReplayRecord r : records) {
                    if (r.top1Match) {
                        matches++;
                    }
                    targetProb += r.targetProb;
                    topProb += r.topProb;
                    if (r.targetRank > 0) {
                        rank += r.targetRank;
                        ranked++;
                    }
                }
            }
            return new TensorReplayStats(total, matches, targetProb, topProb, rank, ranked);
        }

        double accuracy() {
            return total == 0 ? 0.0 : top1Matches / (double) total;
        }

        double meanTargetProb() {
            return total == 0 ? 0.0 : targetProbSum / total;
        }

        double meanTopProb() {
            return total == 0 ? 0.0 : topProbSum / total;
        }

        double meanTargetRank() {
            return ranked == 0 ? 0.0 : targetRankSum / ranked;
        }
    }

    private static final class BranchValueProbeResult {

        final boolean captured;
        final boolean terminal;
        final float valueScore;
        final int valueOrdinal;
        final StateSnapshot state;
        final String error;
        final boolean forcedApplied;
        final boolean timedOut;

        private BranchValueProbeResult(boolean captured, boolean terminal, float valueScore,
                                       int valueOrdinal, StateSnapshot state, String error,
                                       boolean forcedApplied, boolean timedOut) {
            this.captured = captured;
            this.terminal = terminal;
            this.valueScore = valueScore;
            this.valueOrdinal = valueOrdinal;
            this.state = state == null ? StateSnapshot.EMPTY : state;
            this.error = error == null ? "" : error;
            this.forcedApplied = forcedApplied;
            this.timedOut = timedOut;
        }

        static BranchValueProbeResult failed(String error, boolean timedOut) {
            return new BranchValueProbeResult(false, false, Float.NaN, -1,
                    StateSnapshot.EMPTY, error, false, timedOut);
        }
    }

    private static final class BranchValueProbeRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final int baselineIdx;
        final int forcedIdx;
        final String forcedText;
        final float policyProb;
        final boolean terminalWon;
        final boolean terminalTimedOut;
        final boolean terminalForcedApplied;
        final boolean valueCaptured;
        final boolean valueTerminal;
        final float valueScore;
        final int valueOrdinal;
        final boolean valueTimedOut;
        final boolean valueForcedApplied;
        final String state;
        final String error;

        private BranchValueProbeRecord(ScenarioJob job, DecisionPoint point,
                                       BranchResult terminal,
                                       BranchValueProbeResult valueProbe) {
            this.scenario = job.scenario;
            this.agentDeck = job.agentDeck;
            this.oppDeck = job.oppDeck;
            this.seed = job.seed;
            this.ordinal = point.ordinal;
            this.actionType = point.trainingData.actionType;
            this.candidateCount = point.trainingData.candidateCount;
            this.baselineIdx = point.baselineIdx();
            this.forcedIdx = terminal == null ? -1 : terminal.forcedIdx;
            this.forcedText = point.candidateText(this.forcedIdx);
            this.policyProb = forcedIdx >= 0 && forcedIdx < point.probs.length ? point.probs[forcedIdx] : 0.0f;
            this.terminalWon = terminal != null && terminal.won;
            this.terminalTimedOut = terminal != null && terminal.timedOut;
            this.terminalForcedApplied = terminal != null && terminal.forcedApplied;
            this.valueCaptured = valueProbe != null && valueProbe.captured;
            this.valueTerminal = valueProbe != null && valueProbe.terminal;
            this.valueScore = valueProbe == null ? Float.NaN : valueProbe.valueScore;
            this.valueOrdinal = valueProbe == null ? -1 : valueProbe.valueOrdinal;
            this.valueTimedOut = valueProbe != null && valueProbe.timedOut;
            this.valueForcedApplied = valueProbe != null && valueProbe.forcedApplied;
            this.state = valueProbe == null || valueProbe.state == null ? "" : valueProbe.state.toCompactText();
            this.error = valueProbe == null ? "" : valueProbe.error;
        }

        static BranchValueProbeRecord from(ScenarioJob job, DecisionPoint point,
                                           BranchResult terminal,
                                           BranchValueProbeResult valueProbe) {
            return new BranchValueProbeRecord(job, point, terminal, valueProbe);
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + baselineIdx
                    + "," + forcedIdx
                    + "," + csv(forcedText)
                    + "," + String.format(Locale.US, "%.6f", policyProb)
                    + "," + terminalWon
                    + "," + terminalTimedOut
                    + "," + terminalForcedApplied
                    + "," + valueCaptured
                    + "," + valueTerminal
                    + "," + (Float.isFinite(valueScore) ? String.format(Locale.US, "%.6f", valueScore) : "")
                    + "," + valueOrdinal
                    + "," + valueTimedOut
                    + "," + valueForcedApplied
                    + "," + csv(state)
                    + "," + csv(error);
        }
    }

    private static final class DecisionRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int candidateCount;
        final int baselineIdx;
        final int forcedIdx;
        final String forcedText;
        final float policyProb;
        final boolean won;
        final boolean timedOut;
        final boolean forcedApplied;
        final int turns;
        final String error;
        final int trajectoryRawRecords;
        final int trajectoryKeptRecords;
        final String trajectoryDropReason;

        private DecisionRecord(int scenario, Path agentDeck, Path oppDeck, long seed,
                               int ordinal, StateSequenceBuilder.ActionType actionType,
                               int candidateCount, int baselineIdx, int forcedIdx,
                               String forcedText, float policyProb, boolean won, boolean timedOut,
                               boolean forcedApplied, int turns, String error,
                               int trajectoryRawRecords, int trajectoryKeptRecords,
                               String trajectoryDropReason) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.ordinal = ordinal;
            this.actionType = actionType;
            this.candidateCount = candidateCount;
            this.baselineIdx = baselineIdx;
            this.forcedIdx = forcedIdx;
            this.forcedText = forcedText == null ? "" : forcedText;
            this.policyProb = policyProb;
            this.won = won;
            this.timedOut = timedOut;
            this.forcedApplied = forcedApplied;
            this.turns = turns;
            this.error = error == null ? "" : error;
            this.trajectoryRawRecords = Math.max(0, trajectoryRawRecords);
            this.trajectoryKeptRecords = Math.max(0, trajectoryKeptRecords);
            this.trajectoryDropReason = trajectoryDropReason == null ? "" : trajectoryDropReason;
        }

        static DecisionRecord from(ScenarioJob job, DecisionPoint point, BranchResult result) {
            float p = result.forcedIdx >= 0 && result.forcedIdx < point.probs.length ? point.probs[result.forcedIdx] : 0.0f;
            return new DecisionRecord(job.scenario, job.agentDeck, job.oppDeck, job.seed,
                    point.ordinal, point.trainingData.actionType, point.trainingData.candidateCount,
                    point.baselineIdx(), result.forcedIdx, point.candidateText(result.forcedIdx), p, result.won, result.timedOut,
                    result.forcedApplied, result.turns, result.error,
                    result.trajectoryRawRecords, result.trajectoryKeptRecords, result.trajectoryDropReason);
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + actionType
                    + "," + candidateCount
                    + "," + baselineIdx
                    + "," + forcedIdx
                    + "," + csv(forcedText)
                    + "," + String.format(Locale.US, "%.6f", policyProb)
                    + "," + won
                    + "," + timedOut
                    + "," + forcedApplied
                    + "," + turns
                    + "," + csv(error)
                    + "," + trajectoryRawRecords
                    + "," + trajectoryKeptRecords
                    + "," + csv(trajectoryDropReason);
        }
    }

    private static final class PrefixRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int nodeId;
        final int parentId;
        final int prefixLen;
        final int forcedApplied;
        final boolean won;
        final boolean timedOut;
        final int turns;
        final String firstMulliganHand;
        final String firstPriorityHand;
        final int nextOrdinal;
        final String nextActionType;
        final int nextCandidateCount;
        final String nextState;
        final String finalState;
        final String nextChoices;
        int branchCount;
        String branchOrder;
        final String expandedText;
        final String prefix;
        final String error;

        private PrefixRecord(int scenario, Path agentDeck, Path oppDeck, long seed,
                             int nodeId, int parentId, int prefixLen, int forcedApplied,
                             boolean won, boolean timedOut, int turns, String firstMulliganHand, String firstPriorityHand, int nextOrdinal,
                             String nextActionType, int nextCandidateCount, String nextState, String finalState, String nextChoices, int branchCount, String branchOrder, String expandedText,
                             String prefix, String error) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.nodeId = nodeId;
            this.parentId = parentId;
            this.prefixLen = prefixLen;
            this.forcedApplied = forcedApplied;
            this.won = won;
            this.timedOut = timedOut;
            this.turns = turns;
            this.firstMulliganHand = firstMulliganHand == null ? "" : firstMulliganHand;
            this.firstPriorityHand = firstPriorityHand == null ? "" : firstPriorityHand;
            this.nextOrdinal = nextOrdinal;
            this.nextActionType = nextActionType == null ? "" : nextActionType;
            this.nextCandidateCount = nextCandidateCount;
            this.nextState = nextState == null ? "" : nextState;
            this.finalState = finalState == null ? "" : finalState;
            this.nextChoices = nextChoices == null ? "" : nextChoices;
            this.branchCount = branchCount;
            this.branchOrder = branchOrder == null ? "" : branchOrder;
            this.expandedText = expandedText == null ? "" : expandedText;
            this.prefix = prefix == null ? "" : prefix;
            this.error = error == null ? "" : error;
        }

        static PrefixRecord from(ScenarioJob job, PrefixNode node, PrefixRunResult run) {
            int nextOrdinal = -1;
            String nextActionType = "";
            int nextCandidateCount = 0;
            String nextState = "";
            String nextChoices = "";
            if (run != null && run.decisions.size() > node.prefixChoices.size()) {
                DecisionPoint next = run.decisions.get(node.prefixChoices.size());
                nextOrdinal = next.ordinal;
                nextActionType = String.valueOf(next.trainingData.actionType);
                nextCandidateCount = next.trainingData.candidateCount;
                nextState = next.stateSnapshot.toCompactText();
                nextChoices = joinTexts(next.candidateTexts);
            }
            return new PrefixRecord(job.scenario, job.agentDeck, job.oppDeck, job.seed,
                    node.nodeId, node.parentId, node.prefixChoices.size(),
                    run == null ? 0 : run.forcedAppliedCount,
                    run != null && run.won,
                    run != null && run.timedOut,
                    run == null ? -1 : run.turns,
                    run == null ? "" : run.firstMulliganHand,
                    run == null ? "" : run.firstPriorityHand,
                    nextOrdinal,
                    nextActionType,
                    nextCandidateCount,
                    nextState,
                    run == null ? "" : run.finalState.toCompactText(),
                    nextChoices,
                    0,
                    "",
                    node.expandedText,
                    prefixKey(node.prefixChoices),
                    run == null ? "missing_run" : run.error);
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + nodeId
                    + "," + parentId
                    + "," + prefixLen
                    + "," + forcedApplied
                    + "," + won
                    + "," + timedOut
                    + "," + turns
                    + "," + csv(firstMulliganHand)
                    + "," + csv(firstPriorityHand)
                    + "," + nextOrdinal
                    + "," + csv(nextActionType)
                    + "," + nextCandidateCount
                    + "," + csv(nextState)
                    + "," + csv(finalState)
                    + "," + csv(nextChoices)
                    + "," + branchCount
                    + "," + csv(branchOrder)
                    + "," + csv(expandedText)
                    + "," + csv(prefix)
                    + "," + csv(error);
        }
    }

    private static final class ScenarioOutcome {

        final List<DecisionRecord> records;
        final List<BranchValueProbeRecord> valueProbeRecords;
        final List<TrainingExample> trainingExamples;
        final List<PrefixRecord> prefixRecords;
        final List<WinningTrajectoryRecord> winningTrajectories;
        final List<TrajectoryTrainingEpisode> trajectoryEpisodes;
        final String label;

        private ScenarioOutcome(List<DecisionRecord> records, List<BranchValueProbeRecord> valueProbeRecords,
                                List<TrainingExample> trainingExamples,
                                List<PrefixRecord> prefixRecords,
                                List<WinningTrajectoryRecord> winningTrajectories,
                                List<TrajectoryTrainingEpisode> trajectoryEpisodes,
                                String label) {
            this.records = records;
            this.valueProbeRecords = valueProbeRecords == null ? Collections.emptyList() : valueProbeRecords;
            this.trainingExamples = trainingExamples;
            this.prefixRecords = prefixRecords == null ? Collections.emptyList() : prefixRecords;
            this.winningTrajectories = winningTrajectories == null ? Collections.emptyList() : winningTrajectories;
            this.trajectoryEpisodes = trajectoryEpisodes == null ? Collections.emptyList() : trajectoryEpisodes;
            this.label = label;
        }
    }

    private static final class ScenarioJob {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final List<String> agentOpeningHandNames;
        final List<String> agentOpeningLibraryNames;
        final List<String> oppOpeningHandNames;

        private ScenarioJob(int scenario, Path agentDeck, Path oppDeck, long seed,
                            List<String> agentOpeningHandNames, List<String> agentOpeningLibraryNames,
                            List<String> oppOpeningHandNames) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.agentOpeningHandNames = agentOpeningHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(agentOpeningHandNames);
            this.agentOpeningLibraryNames = agentOpeningLibraryNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(agentOpeningLibraryNames);
            this.oppOpeningHandNames = oppOpeningHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(oppOpeningHandNames);
        }
    }

    private static final class ReplayExpectation {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final StateSequenceBuilder.ActionType actionType;
        final int expectedIdx;
        final String expectedText;
        final List<Integer> expectedIndices;
        final List<String> expectedTexts;
        final List<String> agentOpeningHandNames;
        final List<String> agentOpeningLibraryNames;
        final List<String> oppOpeningHandNames;
        final boolean replayTarget;
        final String sourceDecisionNumber;
        final String sourceAnchorId;
        final String sourceTurn;
        final String sourcePhase;
        final String sourceActor;
        final String sourceSelectedText;
        final List<String> sourceHandNames;
        final List<String> sourceLibraryTopNames;
        final List<String> sourceCandidateTexts;
        final List<String> sourceSelectedObjectIds;
        final List<String> sourceCandidateObjectIds;
        final String sourceIdentityStatus;
        final long sourceRandomUtilCountBeforeSearch;
        final int sourceStackCount;
        final String sourceStackTop;

        private ReplayExpectation(int scenario, Path agentDeck, Path oppDeck, long seed,
                                  int ordinal, StateSequenceBuilder.ActionType actionType,
                                  List<Integer> expectedIndices, List<String> expectedTexts,
                                  List<String> agentOpeningHandNames, List<String> agentOpeningLibraryNames,
                                  List<String> oppOpeningHandNames,
                                    boolean replayTarget,
                                    String sourceDecisionNumber, String sourceAnchorId,
                                    String sourceTurn, String sourcePhase, String sourceActor,
                                    String sourceSelectedText,
                                    List<String> sourceHandNames,
                                    List<String> sourceLibraryTopNames,
                                    List<String> sourceCandidateTexts,
                                    List<String> sourceSelectedObjectIds,
                                    List<String> sourceCandidateObjectIds,
                                    String sourceIdentityStatus,
                                    long sourceRandomUtilCountBeforeSearch,
                                    int sourceStackCount,
                                    String sourceStackTop) {
            this.scenario = scenario;
            this.agentDeck = agentDeck;
            this.oppDeck = oppDeck;
            this.seed = seed;
            this.ordinal = ordinal;
            this.actionType = actionType;
            this.expectedIndices = expectedIndices == null ? Collections.emptyList() : new ArrayList<>(expectedIndices);
            this.expectedTexts = expectedTexts == null ? Collections.emptyList() : new ArrayList<>(expectedTexts);
            this.expectedIdx = this.expectedIndices.isEmpty() ? -1 : this.expectedIndices.get(0);
            this.expectedText = this.expectedTexts.isEmpty() ? "" : this.expectedTexts.get(0);
            this.agentOpeningHandNames = agentOpeningHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(agentOpeningHandNames);
            this.agentOpeningLibraryNames = agentOpeningLibraryNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(agentOpeningLibraryNames);
            this.oppOpeningHandNames = oppOpeningHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(oppOpeningHandNames);
            this.replayTarget = replayTarget;
            this.sourceDecisionNumber = sourceDecisionNumber == null ? "" : sourceDecisionNumber.trim();
            this.sourceAnchorId = sourceAnchorId == null ? "" : sourceAnchorId.trim();
            this.sourceTurn = sourceTurn == null ? "" : sourceTurn.trim();
            this.sourcePhase = sourcePhase == null ? "" : sourcePhase.trim();
            this.sourceActor = sourceActor == null ? "" : sourceActor.trim();
            this.sourceSelectedText = sourceSelectedText == null ? "" : sourceSelectedText.trim();
            this.sourceHandNames = sourceHandNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceHandNames);
            this.sourceLibraryTopNames = sourceLibraryTopNames == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceLibraryTopNames);
            this.sourceCandidateTexts = sourceCandidateTexts == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceCandidateTexts);
            this.sourceSelectedObjectIds = sourceSelectedObjectIds == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceSelectedObjectIds);
            this.sourceCandidateObjectIds = sourceCandidateObjectIds == null
                    ? Collections.emptyList()
                    : new ArrayList<>(sourceCandidateObjectIds);
            this.sourceIdentityStatus = sourceIdentityStatus == null ? "" : sourceIdentityStatus.trim();
            this.sourceRandomUtilCountBeforeSearch = sourceRandomUtilCountBeforeSearch;
            this.sourceStackCount = sourceStackCount;
            this.sourceStackTop = sourceStackTop == null ? "" : sourceStackTop.trim();
        }

        boolean hasSourceContext() {
            return !sourceDecisionNumber.isEmpty()
                    || !sourceTurn.isEmpty()
                    || !sourcePhase.isEmpty()
                    || !sourceActor.isEmpty()
                    || sourceStackCount >= 0;
        }

        boolean hasStableObjectIdentity() {
            if (!sourceIdentityStatus.isEmpty()) {
                return "stable_object_ids".equals(sourceIdentityStatus);
            }
            return !sourceSelectedObjectIds.isEmpty() || !sourceCandidateObjectIds.isEmpty();
        }

        boolean hasRequiredTargetSourceState() {
            return !sourceHandNames.isEmpty() && !sourceLibraryTopNames.isEmpty();
        }
    }

    private static final class ReplayGroup {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final List<String> agentOpeningHandNames;
        final List<String> agentOpeningLibraryNames;
        final List<String> oppOpeningHandNames;
        final List<ReplayExpectation> expectations = new ArrayList<>();

        private ReplayGroup(ReplayExpectation first) {
            this.scenario = first.scenario;
            this.agentDeck = first.agentDeck;
            this.oppDeck = first.oppDeck;
            this.seed = first.seed;
            this.agentOpeningHandNames = first.agentOpeningHandNames;
            this.agentOpeningLibraryNames = first.agentOpeningLibraryNames;
            this.oppOpeningHandNames = first.oppOpeningHandNames;
        }
    }

    private static final class ReplayGroupResult {

        final List<ReplayRecord> records;
        final List<PrefixTraceRecord> prefixTraceRecords;
        final List<TrainingExample> deviationExamples;
        final List<TrainingExample> daggerExamples;

        private ReplayGroupResult(
                List<ReplayRecord> records,
                List<PrefixTraceRecord> prefixTraceRecords,
                List<TrainingExample> deviationExamples,
                List<TrainingExample> daggerExamples
        ) {
            this.records = records == null ? Collections.emptyList() : records;
            this.prefixTraceRecords = prefixTraceRecords == null ? Collections.emptyList() : prefixTraceRecords;
            this.deviationExamples = deviationExamples == null ? Collections.emptyList() : deviationExamples;
            this.daggerExamples = daggerExamples == null ? Collections.emptyList() : daggerExamples;
        }
    }

    private static final class PrefixTraceRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int targetOrdinal;
        final String targetSourceDecisionNumber;
        final String targetSourceAnchorId;
        final int ordinal;
        final String sourceDecisionNumber;
        final String sourceAnchorId;
        final String sourceTurn;
        final String sourcePhase;
        final String sourceActor;
        final StateSequenceBuilder.ActionType expectedActionType;
        final StateSequenceBuilder.ActionType actualActionType;
        final String expectedIndicesText;
        final String expectedTextsJoined;
        final String actualSelectedIndicesText;
        final String actualSelectedTextsJoined;
        final String actualNonselectedTextsJoined;
        final String actualCandidateIndicesText;
        final String actualCandidateTextsJoined;
        final String revealedOrPickableNamesJoined;
        final String libraryTopBefore;
        final String libraryTopAfter;
        final String handBefore;
        final String handAfter;
        final String graveyardBefore;
        final String graveyardAfter;
        final String stackBefore;
        final String stackAfter;
        final String state;

        private PrefixTraceRecord(
                ScenarioJob job,
                ReplayExpectation target,
                ReplayExpectation expected,
                DecisionPoint point,
                DecisionPoint nextPoint
        ) {
            this.scenario = job.scenario;
            this.agentDeck = job.agentDeck;
            this.oppDeck = job.oppDeck;
            this.seed = job.seed;
            this.targetOrdinal = target == null ? -1 : target.ordinal;
            this.targetSourceDecisionNumber = target == null ? "" : target.sourceDecisionNumber;
            this.targetSourceAnchorId = target == null ? "" : target.sourceAnchorId;
            this.ordinal = point == null ? -1 : point.ordinal;
            this.sourceDecisionNumber = expected == null ? "" : expected.sourceDecisionNumber;
            this.sourceAnchorId = expected == null ? "" : expected.sourceAnchorId;
            this.sourceTurn = expected == null ? "" : expected.sourceTurn;
            this.sourcePhase = expected == null ? "" : expected.sourcePhase;
            this.sourceActor = expected == null ? "" : expected.sourceActor;
            this.expectedActionType = expected == null ? null : expected.actionType;
            this.actualActionType = point == null ? null : point.trainingData.actionType;
            this.expectedIndicesText = expected == null ? "" : joinInts(expected.expectedIndices);
            this.expectedTextsJoined = expected == null ? "" : joinTexts(expected.expectedTexts);
            this.actualSelectedIndicesText = point == null ? "" : joinInts(point.chosenIndices);
            this.actualSelectedTextsJoined = point == null ? "" : joinTexts(candidateTexts(point, point.chosenIndices));
            this.actualNonselectedTextsJoined = nonselectedTexts(point);
            this.actualCandidateIndicesText = point == null ? "" : joinInts(candidateIndices(point.candidateTexts));
            this.actualCandidateTextsJoined = point == null ? "" : indexedTexts(point.candidateTexts);
            this.revealedOrPickableNamesJoined = revealedOrPickableNames(point);
            this.libraryTopBefore = stateField(point, "libraryTop");
            this.libraryTopAfter = stateField(nextPoint, "libraryTop");
            this.handBefore = stateField(point, "hand");
            this.handAfter = stateField(nextPoint, "hand");
            this.graveyardBefore = stateField(point, "graveyard");
            this.graveyardAfter = stateField(nextPoint, "graveyard");
            this.stackBefore = stateField(point, "stack");
            this.stackAfter = stateField(nextPoint, "stack");
            this.state = point == null || point.stateSnapshot == null ? "" : point.stateSnapshot.toCompactText();
        }

        static List<PrefixTraceRecord> from(
                ScenarioJob job,
                ReplayExpectation target,
                List<ReplayExpectation> prefixExpectations,
                PrefixRunResult run
        ) {
            if (run == null || run.decisions.isEmpty()) {
                return Collections.emptyList();
            }
            List<PrefixTraceRecord> out = new ArrayList<>();
            for (int i = 0; i < run.decisions.size(); i++) {
                DecisionPoint point = run.decisions.get(i);
                DecisionPoint nextPoint = i + 1 < run.decisions.size() ? run.decisions.get(i + 1) : null;
                ReplayExpectation expected = point.ordinal >= 0
                        && prefixExpectations != null
                        && point.ordinal < prefixExpectations.size()
                        ? prefixExpectations.get(point.ordinal)
                        : null;
                out.add(new PrefixTraceRecord(job, target, expected, point, nextPoint));
            }
            return out;
        }

        private static String stateField(DecisionPoint point, String key) {
            if (point == null || point.stateSnapshot == null || key == null || key.isEmpty()) {
                return "";
            }
            String prefix = key + "=";
            String text = point.stateSnapshot.toCompactText();
            if (text == null || text.isEmpty()) {
                return "";
            }
            for (String part : text.split(";")) {
                if (part.startsWith(prefix)) {
                    return part.substring(prefix.length());
                }
            }
            return "";
        }

        private static String nonselectedTexts(DecisionPoint point) {
            if (point == null || point.candidateTexts == null || point.candidateTexts.isEmpty()) {
                return "";
            }
            Set<Integer> selected = new HashSet<>();
            if (point.chosenIndices != null) {
                selected.addAll(point.chosenIndices);
            }
            List<String> out = new ArrayList<>();
            for (int i = 0; i < point.candidateTexts.size(); i++) {
                if (selected.contains(i)) {
                    continue;
                }
                String text = point.candidateTexts.get(i);
                if (isStopCandidateText(text)) {
                    continue;
                }
                out.add(text);
            }
            return joinTexts(out);
        }

        private static String revealedOrPickableNames(DecisionPoint point) {
            if (point == null
                    || point.trainingData == null
                    || point.trainingData.actionType != StateSequenceBuilder.ActionType.SELECT_CARD
                    || point.candidateTexts == null) {
                return "";
            }
            List<String> out = new ArrayList<>();
            for (String text : point.candidateTexts) {
                if (!isStopCandidateText(text)) {
                    out.add(text);
                }
            }
            return joinTexts(out);
        }

        private static boolean isStopCandidateText(String text) {
            return text == null || text.trim().isEmpty() || "STOP".equalsIgnoreCase(text.trim());
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + targetOrdinal
                    + "," + csv(targetSourceDecisionNumber)
                    + "," + csv(targetSourceAnchorId)
                    + "," + ordinal
                    + "," + csv(sourceDecisionNumber)
                    + "," + csv(sourceAnchorId)
                    + "," + csv(sourceTurn)
                    + "," + csv(sourcePhase)
                    + "," + csv(sourceActor)
                    + "," + (expectedActionType == null ? "" : expectedActionType)
                    + "," + (actualActionType == null ? "" : actualActionType)
                    + "," + csv(expectedIndicesText)
                    + "," + csv(expectedTextsJoined)
                    + "," + csv(actualSelectedIndicesText)
                    + "," + csv(actualSelectedTextsJoined)
                    + "," + csv(actualNonselectedTextsJoined)
                    + "," + csv(actualCandidateIndicesText)
                    + "," + csv(actualCandidateTextsJoined)
                    + "," + csv(revealedOrPickableNamesJoined)
                    + "," + csv(libraryTopBefore)
                    + "," + csv(libraryTopAfter)
                    + "," + csv(handBefore)
                    + "," + csv(handAfter)
                    + "," + csv(graveyardBefore)
                    + "," + csv(graveyardAfter)
                    + "," + csv(stackBefore)
                    + "," + csv(stackAfter)
                    + "," + csv(state);
        }
    }

    private static final class ReplayRecord {

        final int scenario;
        final Path agentDeck;
        final Path oppDeck;
        final long seed;
        final int ordinal;
        final String sourceDecisionNumber;
        final String sourceAnchorId;
        final String sourceTurn;
        final String sourcePhase;
        final String sourceActor;
        final StateSequenceBuilder.ActionType actionType;
        final StateSequenceBuilder.ActionType actualActionType;
        final int expectedIdx;
        final String expectedText;
        final int actualIdx;
        final String actualText;
        final String expectedIndicesText;
        final String actualIndicesText;
        final String expectedTextsJoined;
        final String actualTextsJoined;
        final float expectedProb;
        final float actualProb;
        final boolean indexMatch;
        final boolean textMatch;
        final boolean matched;
        final boolean replayWon;
        final boolean timedOut;
        final int turns;
        final int actualDecisions;
        final int forcedPrefixCount;
        final int prefixFailureOrdinal;
        final String prefixFailureReason;
        final String prefixFailureDetail;
        final StateSequenceBuilder.ActionType prefixExpectedActionType;
        final StateSequenceBuilder.ActionType prefixActualActionType;
        final String prefixExpectedIndicesText;
        final String prefixExpectedTextsJoined;
        final String prefixActualCandidateIndicesText;
        final String prefixActualCandidateTextsJoined;
        final String prefixActualSelectedIndicesText;
        final String prefixActualSelectedTextsJoined;
        final int prefixFailureForcedCount;
        final String prefixSourceDecisionNumber;
        final String prefixSourceAnchorId;
        final String prefixSourceTurn;
        final String prefixSourcePhase;
        final String prefixSourceActor;
        final String prefixFailureState;
        final String error;

        private ReplayRecord(ReplayExpectation expected, StateSequenceBuilder.ActionType actualActionType,
                             List<Integer> actualIndices, List<String> actualTexts,
                             float expectedProb, float actualProb, boolean indexMatch,
                             boolean textMatch, boolean matched, boolean replayWon,
                             boolean timedOut, int turns, int actualDecisions,
                             int forcedPrefixCount, PrefixDivergence prefixDivergence,
                             List<Integer> prefixActualSelectedIndices,
                             List<String> prefixActualSelectedTexts,
                             String error) {
            this.scenario = expected.scenario;
            this.agentDeck = expected.agentDeck;
            this.oppDeck = expected.oppDeck;
            this.seed = expected.seed;
            this.ordinal = expected.ordinal;
            this.sourceDecisionNumber = expected.sourceDecisionNumber;
            this.sourceAnchorId = expected.sourceAnchorId;
            this.sourceTurn = expected.sourceTurn;
            this.sourcePhase = expected.sourcePhase;
            this.sourceActor = expected.sourceActor;
            this.actionType = expected.actionType;
            this.actualActionType = actualActionType;
            this.expectedIdx = expected.expectedIdx;
            this.expectedText = expected.expectedText;
            this.actualIdx = actualIndices == null || actualIndices.isEmpty() ? -1 : actualIndices.get(0);
            this.actualText = actualTexts == null || actualTexts.isEmpty() ? "" : actualTexts.get(0);
            this.expectedIndicesText = joinInts(expected.expectedIndices);
            this.actualIndicesText = joinInts(actualIndices);
            this.expectedTextsJoined = joinTexts(expected.expectedTexts);
            this.actualTextsJoined = joinTexts(actualTexts);
            this.expectedProb = expectedProb;
            this.actualProb = actualProb;
            this.indexMatch = indexMatch;
            this.textMatch = textMatch;
            this.matched = matched;
            this.replayWon = replayWon;
            this.timedOut = timedOut;
            this.turns = turns;
            this.actualDecisions = actualDecisions;
            this.forcedPrefixCount = forcedPrefixCount;
            this.prefixFailureOrdinal = prefixDivergence == null ? -1 : prefixDivergence.ordinal;
            this.prefixFailureReason = prefixDivergence == null ? "" : prefixDivergence.reason;
            this.prefixFailureDetail = prefixDivergence == null ? "" : prefixDivergence.detail;
            this.prefixExpectedActionType = prefixDivergence == null ? null : prefixDivergence.expectedActionType;
            this.prefixActualActionType = prefixDivergence == null ? null : prefixDivergence.actualActionType;
            this.prefixExpectedIndicesText = prefixDivergence == null ? "" : joinInts(prefixDivergence.expectedIndices);
            this.prefixExpectedTextsJoined = prefixDivergence == null ? "" : joinTexts(prefixDivergence.expectedTexts);
            this.prefixActualCandidateIndicesText = prefixDivergence == null ? "" : joinInts(prefixDivergence.actualCandidateIndices);
            this.prefixActualCandidateTextsJoined = prefixDivergence == null ? "" : indexedTexts(prefixDivergence.actualCandidateTexts);
            this.prefixActualSelectedIndicesText = prefixDivergence == null ? "" : joinInts(prefixActualSelectedIndices);
            this.prefixActualSelectedTextsJoined = prefixDivergence == null ? "" : joinTexts(prefixActualSelectedTexts);
            this.prefixFailureForcedCount = prefixDivergence == null ? -1 : prefixDivergence.forcedPrefixCount;
            this.prefixSourceDecisionNumber = prefixDivergence == null ? "" : prefixDivergence.sourceDecisionNumber;
            this.prefixSourceAnchorId = prefixDivergence == null ? "" : prefixDivergence.sourceAnchorId;
            this.prefixSourceTurn = prefixDivergence == null ? "" : prefixDivergence.sourceTurn;
            this.prefixSourcePhase = prefixDivergence == null ? "" : prefixDivergence.sourcePhase;
            this.prefixSourceActor = prefixDivergence == null ? "" : prefixDivergence.sourceActor;
            this.prefixFailureState = prefixDivergence == null ? "" : prefixDivergence.state.toCompactText();
            this.error = error == null ? "" : error;
        }

        static ReplayRecord from(ReplayExpectation expected, DecisionPoint point, PrefixRunResult run) {
            int actualIdx = -1;
            String actualText = "";
            float expectedProb = 0.0f;
            float actualProb = 0.0f;
            boolean indexMatch = false;
            boolean textMatch = false;
            boolean matched = false;
            List<Integer> actualIndices = Collections.emptyList();
            List<String> actualTexts = Collections.emptyList();
            String error = run.error;
            StateSequenceBuilder.ActionType actualActionType = null;
            ObservedReplayChoice observedChoice = null;
            if (point == null) {
                if (error.isEmpty()) {
                    error = "missing_ordinal";
                }
            } else {
                DecisionPoint nextPoint = expected.ordinal >= 0 && expected.ordinal + 1 < run.decisions.size()
                        ? run.decisions.get(expected.ordinal + 1)
                        : null;
                observedChoice = observedReplayChoice(expected, point, nextPoint);
                actualActionType = observedChoice == null
                        ? replayActualActionType(expected, point)
                        : observedChoice.actionType;
            }
            if (point != null && observedChoice != null) {
                actualIndices = new ArrayList<>(observedChoice.indices);
                actualTexts = new ArrayList<>(observedChoice.texts);
                actualIdx = actualIndices.isEmpty() ? -1 : actualIndices.get(0);
                actualText = actualTexts.isEmpty() ? "" : actualTexts.get(0);
                expectedProb = observedChoice.probabilityIndex >= 0 && observedChoice.probabilityIndex < point.probs.length
                        ? point.probs[observedChoice.probabilityIndex]
                        : 0.0f;
                actualProb = expectedProb;
                indexMatch = actualIndices.equals(expected.expectedIndices);
                textMatch = ActionCounterfactualTrainer.textListsMatchExpected(expected.expectedTexts, actualTexts, point);
                matched = indexMatch && (expected.expectedTexts.isEmpty() || textMatch);
            } else if (point != null && actualActionType != expected.actionType) {
                actualIndices = new ArrayList<>(point.chosenIndices);
                actualTexts = candidateTexts(point, actualIndices);
                actualIdx = actualIndices.isEmpty() ? -1 : actualIndices.get(0);
                actualText = actualTexts.isEmpty() ? "" : actualTexts.get(0);
                if (error.isEmpty()) {
                    error = "action_type_mismatch";
                }
            } else if (point != null) {
                actualIndices = new ArrayList<>(point.chosenIndices);
                actualTexts = candidateTexts(point, actualIndices);
                actualIdx = actualIndices.isEmpty() ? -1 : actualIndices.get(0);
                actualText = actualTexts.isEmpty() ? "" : actualTexts.get(0);
                expectedProb = expected.expectedIdx >= 0 && expected.expectedIdx < point.probs.length
                        ? point.probs[expected.expectedIdx]
                        : 0.0f;
                actualProb = actualIdx >= 0 && actualIdx < point.probs.length ? point.probs[actualIdx] : 0.0f;
                indexMatch = actualIndices.equals(expected.expectedIndices);
                textMatch = ActionCounterfactualTrainer.textListsMatchExpected(expected.expectedTexts, actualTexts, point);
                matched = indexMatch && (expected.expectedTexts.isEmpty() || textMatch);
            }
            PrefixDivergence targetStateDivergence = targetSourceStateDivergence(
                    expected, point, actualActionType, actualIndices, actualTexts, run.forcedAppliedCount);
            if (targetStateDivergence != null) {
                matched = false;
                error = error.isEmpty()
                        ? targetStateDivergence.reason
                        : appendDetail(error, targetStateDivergence.reason);
            }
            PrefixDivergence prefixDivergence = run.prefixDivergence == null
                    ? targetStateDivergence
                    : run.prefixDivergence;
            List<Integer> prefixActualSelectedIndices = Collections.emptyList();
            List<String> prefixActualSelectedTexts = Collections.emptyList();
            if (prefixDivergence != null) {
                prefixActualSelectedIndices = new ArrayList<>(prefixDivergence.actualSelectedIndices);
                prefixActualSelectedTexts = new ArrayList<>(prefixDivergence.actualSelectedTexts);
                if (prefixActualSelectedIndices.isEmpty()
                        && prefixDivergence.ordinal >= 0
                        && prefixDivergence.ordinal < run.decisions.size()) {
                    DecisionPoint divergencePoint = run.decisions.get(prefixDivergence.ordinal);
                    prefixActualSelectedIndices = new ArrayList<>(divergencePoint.chosenIndices);
                    prefixActualSelectedTexts = candidateTexts(divergencePoint, prefixActualSelectedIndices);
                }
            }
            return new ReplayRecord(expected, actualActionType, actualIndices, actualTexts, expectedProb, actualProb,
                    indexMatch, textMatch, matched, run.won, run.timedOut, run.turns, run.decisions.size(),
                    run.forcedAppliedCount, prefixDivergence, prefixActualSelectedIndices, prefixActualSelectedTexts,
                    error);
        }

        private static PrefixDivergence targetSourceStateDivergence(
                ReplayExpectation expected,
                DecisionPoint point,
                StateSequenceBuilder.ActionType actualActionType,
                List<Integer> actualIndices,
                List<String> actualTexts,
                int forcedPrefixCount
        ) {
            if (expected == null || !expected.replayTarget || point == null) {
                return null;
            }
            String liveHand = stateField(point.stateSnapshot, "hand");
            String liveLibraryTop = stateField(point.stateSnapshot, "libraryTop");
            boolean metadataMissing = !expected.hasRequiredTargetSourceState();
            boolean handMismatch = !metadataMissing && !zoneMultisetMatches(expected.sourceHandNames, liveHand);
            boolean libraryMismatch = !metadataMissing && !libraryPrefixMatches(expected.sourceLibraryTopNames, liveLibraryTop);
            if (!metadataMissing && !handMismatch && !libraryMismatch) {
                return null;
            }
            String reason = metadataMissing ? "target_source_state_unverifiable" : "target_source_state_mismatch";
            String expectedAction = !expected.sourceSelectedText.isEmpty()
                    ? expected.sourceSelectedText
                    : joinTexts(expected.expectedTexts);
            String detail = "expected_target_action=" + expectedAction
                    + " source_hand=" + joinTexts(expected.sourceHandNames)
                    + " source_library_top=" + joinTexts(expected.sourceLibraryTopNames)
                    + " live_target_action=" + joinTexts(actualTexts)
                    + " live_candidates=" + indexedTexts(point.candidateTexts)
                    + " live_hand=" + liveHand
                    + " live_library_top=" + liveLibraryTop;
            if (metadataMissing) {
                detail = appendDetail(detail, "required_source_metadata=source_hand,source_library_top");
            }
            return new PrefixDivergence(
                    expected.ordinal,
                    reason,
                    expected.actionType,
                    actualActionType,
                    expected.expectedIndices,
                    expected.expectedTexts,
                    point.candidateTexts,
                    actualIndices,
                    actualTexts,
                    forcedPrefixCount,
                    point.stateSnapshot,
                    detail,
                    expected);
        }

        private static boolean zoneMultisetMatches(List<String> expectedNames, String actualZoneText) {
            Map<String, Integer> expected = countsFromList(expectedNames);
            Map<String, Integer> actual = zoneCounts(actualZoneText);
            return expected.equals(actual);
        }

        private static boolean libraryPrefixMatches(List<String> expectedNames, String actualZoneText) {
            List<String> actual = zoneItems(actualZoneText);
            if (expectedNames == null || expectedNames.isEmpty() || actual.size() < expectedNames.size()) {
                return false;
            }
            for (int i = 0; i < expectedNames.size(); i++) {
                if (!normalizeText(expectedNames.get(i)).equals(normalizeText(actual.get(i)))) {
                    return false;
                }
            }
            return true;
        }

        private static Map<String, Integer> countsFromList(List<String> names) {
            Map<String, Integer> out = new LinkedHashMap<>();
            if (names == null) {
                return out;
            }
            for (String raw : names) {
                String name = normalizeText(raw);
                if (name.isEmpty()) {
                    continue;
                }
                out.put(name, countOf(out, name) + 1);
            }
            return out;
        }

        private static List<String> zoneItems(String zoneText) {
            if (zoneText == null || zoneText.trim().isEmpty()) {
                return Collections.emptyList();
            }
            return Arrays.stream(zoneText.split("\\|", -1))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
        }

        private static ObservedReplayChoice observedReplayChoice(
                ReplayExpectation expected,
                DecisionPoint point,
                DecisionPoint nextPoint
        ) {
            if (!isD046RefurbishedFamiliarDiscardPrompt(expected, point, nextPoint)) {
                return null;
            }
            String discarded = uniqueHandToGraveyardMove(point.stateSnapshot, nextPoint.stateSnapshot);
            if (discarded.isEmpty()) {
                return null;
            }
            int expectedTextIndex = indexOfNormalized(expected.expectedTexts, discarded);
            List<Integer> indices = expectedTextIndex >= 0
                    ? new ArrayList<>(expected.expectedIndices)
                    : candidateTextIndices(point, discarded);
            if (indices.isEmpty()) {
                return null;
            }
            int probabilityIndex = point.chosenIndices == null || point.chosenIndices.isEmpty()
                    ? indices.get(0)
                    : point.chosenIndices.get(0);
            return new ObservedReplayChoice(
                    StateSequenceBuilder.ActionType.SELECT_TARGETS,
                    indices,
                    Collections.singletonList(discarded),
                    probabilityIndex);
        }

        private static boolean isD046RefurbishedFamiliarDiscardPrompt(
                ReplayExpectation expected,
                DecisionPoint point,
                DecisionPoint nextPoint
        ) {
            if (expected == null
                    || point == null
                    || nextPoint == null
                    || expected.actionType != StateSequenceBuilder.ActionType.SELECT_TARGETS
                    || point.trainingData == null
                    || point.trainingData.actionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    || expected.sourceAnchorId == null
                    || !expected.sourceAnchorId.endsWith("_D046")
                    || !"46".equals(expected.sourceDecisionNumber)
                    || sourcePromptKey(expected.sourcePhase) == null
                    || !"TARGET_PICK".equals(sourcePromptKey(expected.sourcePhase))) {
                return false;
            }
            String state = point.stateSnapshot == null ? "" : point.stateSnapshot.toCompactText();
            if (!stateField(state, "opponentBattlefield").contains("Refurbished Familiar")) {
                return false;
            }
            String hand = stateField(state, "hand");
            if (hand.isEmpty() || !zoneContainsAllCandidateTexts(hand, point.candidateTexts)) {
                return false;
            }
            return !expected.expectedTexts.isEmpty()
                    && zoneContainsAny(hand, expected.expectedTexts);
        }

        private static boolean zoneContainsAllCandidateTexts(String zoneText, List<String> candidateTexts) {
            Map<String, Integer> zone = zoneCounts(zoneText);
            for (String candidate : candidateTexts) {
                if (candidate == null || candidate.trim().isEmpty()) {
                    continue;
                }
                if (countOf(zone, normalizeText(candidate)) <= 0) {
                    return false;
                }
            }
            return true;
        }

        private static boolean zoneContainsAny(String zoneText, List<String> names) {
            Map<String, Integer> zone = zoneCounts(zoneText);
            for (String name : names) {
                if (countOf(zone, normalizeText(name)) > 0) {
                    return true;
                }
            }
            return false;
        }

        private static String uniqueHandToGraveyardMove(StateSnapshot before, StateSnapshot after) {
            String beforeState = before == null ? "" : before.toCompactText();
            String afterState = after == null ? "" : after.toCompactText();
            Map<String, Integer> beforeHand = zoneCounts(stateField(beforeState, "hand"));
            Map<String, Integer> afterHand = zoneCounts(stateField(afterState, "hand"));
            Map<String, Integer> beforeGraveyard = zoneCounts(stateField(beforeState, "graveyard"));
            Map<String, Integer> afterGraveyard = zoneCounts(stateField(afterState, "graveyard"));
            List<String> moved = new ArrayList<>();
            for (Map.Entry<String, Integer> entry : beforeHand.entrySet()) {
                String name = entry.getKey();
                int removed = entry.getValue() - countOf(afterHand, name);
                int addedToGraveyard = countOf(afterGraveyard, name) - countOf(beforeGraveyard, name);
                for (int i = 0; i < Math.min(removed, addedToGraveyard); i++) {
                    moved.add(name);
                }
            }
            return moved.size() == 1 ? moved.get(0) : "";
        }

        private static Map<String, Integer> zoneCounts(String zoneText) {
            Map<String, Integer> out = new LinkedHashMap<>();
            if (zoneText == null || zoneText.trim().isEmpty()) {
                return out;
            }
            for (String raw : zoneText.split("\\|")) {
                String name = normalizeText(raw);
                if (name.isEmpty()) {
                    continue;
                }
                out.put(name, countOf(out, name) + 1);
            }
            return out;
        }

        private static int countOf(Map<String, Integer> counts, String name) {
            Integer value = counts.get(name);
            return value == null ? 0 : value;
        }

        private static int indexOfNormalized(List<String> texts, String wanted) {
            String normalized = normalizeText(wanted);
            for (int i = 0; i < texts.size(); i++) {
                if (normalizeText(texts.get(i)).equals(normalized)) {
                    return i;
                }
            }
            return -1;
        }

        private static List<Integer> candidateTextIndices(DecisionPoint point, String wanted) {
            String normalized = normalizeText(wanted);
            List<Integer> out = new ArrayList<>();
            for (int i = 0; i < point.candidateTexts.size(); i++) {
                if (normalizeText(point.candidateTexts.get(i)).equals(normalized)) {
                    out.add(i);
                }
            }
            return out;
        }

        private static StateSequenceBuilder.ActionType replayActualActionType(
                ReplayExpectation expected,
                DecisionPoint point
        ) {
            StateSequenceBuilder.ActionType actual = point == null || point.trainingData == null
                    ? null
                    : point.trainingData.actionType;
            if (actual == StateSequenceBuilder.ActionType.DECLARE_BLOCKS
                    && expected != null
                    && expected.actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                    && "70".equals(expected.sourceDecisionNumber)
                    && expected.sourceAnchorId != null
                    && expected.sourceAnchorId.endsWith("_D070")
                    && "PRECOMBAT_MAIN".equals(sourceStepKey(expected.sourcePhase))
                    && point != null
                    && point.stateSnapshot != null
                    && "PRECOMBAT_MAIN".equals(sourceStepKey(point.stateSnapshot.stepText()))
                    && textListsMatchExpected(expected.expectedTexts, candidateTexts(point, point.chosenIndices), point)) {
                return expected.actionType;
            }
            return actual;
        }

        private static final class ObservedReplayChoice {
            final StateSequenceBuilder.ActionType actionType;
            final List<Integer> indices;
            final List<String> texts;
            final int probabilityIndex;

            private ObservedReplayChoice(
                    StateSequenceBuilder.ActionType actionType,
                    List<Integer> indices,
                    List<String> texts,
                    int probabilityIndex
            ) {
                this.actionType = actionType;
                this.indices = indices == null ? Collections.emptyList() : indices;
                this.texts = texts == null ? Collections.emptyList() : texts;
                this.probabilityIndex = probabilityIndex;
            }
        }

        private static List<String> candidateTexts(DecisionPoint point, List<Integer> indices) {
            List<String> out = new ArrayList<>();
            if (indices == null) {
                return out;
            }
            for (Integer idx : indices) {
                out.add(point.candidateText(idx == null ? -1 : idx));
            }
            return out;
        }

        private static List<String> normalizedTexts(List<String> texts) {
            if (texts == null) {
                return Collections.emptyList();
            }
            return texts.stream().map(ActionCounterfactualTrainer::normalizeText).collect(Collectors.toList());
        }

        String toCsv() {
            return scenario
                    + "," + csv(agentDeck.getFileName().toString())
                    + "," + csv(oppDeck.getFileName().toString())
                    + "," + seed
                    + "," + ordinal
                    + "," + csv(sourceDecisionNumber)
                    + "," + csv(sourceAnchorId)
                    + "," + csv(sourceTurn)
                    + "," + csv(sourcePhase)
                    + "," + csv(sourceActor)
                    + "," + actionType
                    + "," + (actualActionType == null ? "" : actualActionType)
                    + "," + expectedIdx
                    + "," + csv(expectedText)
                    + "," + actualIdx
                    + "," + csv(actualText)
                    + "," + csv(expectedIndicesText)
                    + "," + csv(actualIndicesText)
                    + "," + csv(expectedTextsJoined)
                    + "," + csv(actualTextsJoined)
                    + "," + String.format(Locale.US, "%.6f", expectedProb)
                    + "," + String.format(Locale.US, "%.6f", actualProb)
                    + "," + indexMatch
                    + "," + textMatch
                    + "," + matched
                    + "," + replayWon
                    + "," + timedOut
                    + "," + turns
                    + "," + actualDecisions
                    + "," + forcedPrefixCount
                    + "," + prefixFailureOrdinal
                    + "," + csv(prefixFailureReason)
                    + "," + csv(prefixFailureDetail)
                    + "," + (prefixExpectedActionType == null ? "" : prefixExpectedActionType)
                    + "," + (prefixActualActionType == null ? "" : prefixActualActionType)
                    + "," + csv(prefixExpectedIndicesText)
                    + "," + csv(prefixExpectedTextsJoined)
                    + "," + csv(prefixActualCandidateIndicesText)
                    + "," + csv(prefixActualCandidateTextsJoined)
                    + "," + csv(prefixActualSelectedIndicesText)
                    + "," + csv(prefixActualSelectedTextsJoined)
                    + "," + prefixFailureForcedCount
                    + "," + csv(prefixSourceDecisionNumber)
                    + "," + csv(prefixSourceAnchorId)
                    + "," + csv(prefixSourceTurn)
                    + "," + csv(prefixSourcePhase)
                    + "," + csv(prefixSourceActor)
                    + "," + csv(prefixFailureState)
                    + "," + csv(error);
        }
    }

    private static final class ReplayStats {

        final int total;
        final int matched;
        final int indexMatches;
        final int textMatches;
        final int wonGroups;
        final int totalGroups;

        private ReplayStats(int total, int matched, int indexMatches, int textMatches,
                            int wonGroups, int totalGroups) {
            this.total = total;
            this.matched = matched;
            this.indexMatches = indexMatches;
            this.textMatches = textMatches;
            this.wonGroups = wonGroups;
            this.totalGroups = totalGroups;
        }

        static ReplayStats from(List<ReplayRecord> records) {
            int total = records.size();
            int matched = 0;
            int indexMatches = 0;
            int textMatches = 0;
            Set<String> groups = new HashSet<>();
            Set<String> wonGroups = new HashSet<>();
            for (ReplayRecord r : records) {
                if (r.matched) {
                    matched++;
                }
                if (r.indexMatch) {
                    indexMatches++;
                }
                if (r.textMatch) {
                    textMatches++;
                }
                String key = r.scenario + "|" + r.seed;
                groups.add(key);
                if (r.replayWon) {
                    wonGroups.add(key);
                }
            }
            return new ReplayStats(total, matched, indexMatches, textMatches,
                    wonGroups.size(), groups.size());
        }

        double accuracy() {
            return total == 0 ? 0.0 : ((double) matched / (double) total);
        }
    }

    private static final class ActionWorkerThreadFactory implements ThreadFactory {

        private final AtomicInteger idx = new AtomicInteger(0);

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "ACTION-CF-" + idx.incrementAndGet());
            t.setDaemon(false);
            t.setPriority(Thread.NORM_PRIORITY);
            return t;
        }
    }

    private static final class Args {

        String agentDeckList = DEFAULT_AGENT_DECK_LIST;
        String oppDeckList = DEFAULT_OPP_DECK_LIST;
        List<String> agentOpeningHandNames = Collections.emptyList();
        List<String> oppOpeningHandNames = Collections.emptyList();
        List<List<String>> agentOpeningHandPool = Collections.emptyList();
        List<List<String>> oppOpeningHandPool = Collections.emptyList();
        Path agentOpeningHandPoolFile = null;
        Path oppOpeningHandPoolFile = null;
        Path outDir = null;
        Path replayFile = null;
        Path exportTrainingDataFile = null;
        Path exportTrajectoryDataFile = null;
        Path importTrainingDataPath = null;
        Path importTrajectoryDataPath = null;
        Path scoreTrainingDataPath = null;
        boolean importFlatAsTerminalEpisodes = false;
        Path replayDeviationTrainingDataFile = null;
        Path replayDaggerTrainingDataFile = null;
        Path livePrefixTracePath = null;
        Path opponentTranscriptFile = null;
        Path opponentTranscriptMismatchPath = null;
        Path policyInputDumpPath = null;
        Path policyInferenceProbePath = null;
        boolean policyInputDump = false;
        boolean policyInferenceProbe = false;
        boolean collectOnly = false;
        int scoreMaxExamples = 10000;
        boolean fitScoreProbe = false;
        int replayMaxScenarios = 0;
        boolean forcedPrefixReplay = false;
        boolean checkpointBranchProbe = false;
        boolean forceOpponentTranscript = false;
        OpponentTranscript opponentTranscript = null;
        int replayDeviationRepeat = 1;
        List<List<Integer>> initialPrefixChoices = Collections.emptyList();
        int scenarios = 32;
        int batchSize = 32;
        int timeoutSec = 60;
        int scenarioTimeoutSec = 0;
        int maxGameTurns = 0;
        int reportEvery = 4;
        int workers = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        int postTrainWaitMs = 300000;
        long seed = System.currentTimeMillis();
        String opponentMode = "rl";
        int cp7Skill = 7;
        int maxDecisionDepth = 6;
        int topK = 4;
        int randomExtra = 1;
        int trainEpochs = 4;
        int candidatePermutations = 1;
        int maxTrainExamples = 0;
        double minTargetMargin = 0.0;
        boolean policyMissOnly = false;
        int stopAfterExamples = 0;
        int stopAfterWinningTrajectories = 0;
        double trajectoryFinalReward = 1.0;
        boolean winningPrefixMode = false;
        int maxPrefixDepth = 6;
        int trainPrefixDepth = 6;
        int maxSearchNodes = 64;
        int branchSubtreeSearchNodes = 0;
        int maxWinningPrefixesPerScenario = 1;
        boolean skipPassTraining = false;
        boolean skipBlankTraining = false;
        boolean skipMulliganTraining = false;
        String includeActionTextRegex = "";
        String avoidLosingActionTextRegex = "";
        boolean avoidLosingStrictNegative = false;
        boolean avoidLosingMaskBaselineOnly = false;
        boolean branchReturnTargets = false;
        boolean branchReturnBalance = false;
        int branchReturnMaxNegativesPerPositive = 1;
        boolean baselineLosingAlternativeOnly = false;
        boolean branchValueProbe = false;
        boolean branchTrajectoryMode = false;
        boolean branchTrajectoryFirstPostTargetOnly = false;
        boolean branchTrajectoryPairMode = false;
        boolean branchTrajectoryRequireTrainingExample = false;
        double targetTemperature = 0.25;
        double winTurnBonus = 1.00;
        double lossTurnBonus = 0.00;
        boolean skipPassBest = false;
        boolean depthFirstSearch = true;
        boolean prefixSiblingContrast = false;
        boolean trainRootMulliganOnNoWin = false;
        boolean trainRootMulliganOnly = false;
        boolean genericBranchOrder = false;
        boolean tacticAutopilot = false;
        boolean noSearchModelScoring = false;
        int prefixSiblingContrastSearchNodes = 0;
        List<Integer> passMacroDepths = Collections.emptyList();
        TerminalMode terminalMode = TerminalMode.WIN;
        Set<StateSequenceBuilder.ActionType> targetTypes = EnumSet.of(
                StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL,
                StateSequenceBuilder.ActionType.SELECT_TARGETS,
                StateSequenceBuilder.ActionType.SELECT_CARD,
                StateSequenceBuilder.ActionType.CHOOSE_USE,
                StateSequenceBuilder.ActionType.CHOOSE_MODE,
                StateSequenceBuilder.ActionType.ANNOUNCE_X
        );

        OpponentTranscriptCursor opponentTranscriptCursor(ScenarioJob job) {
            if (!forceOpponentTranscript) {
                return OpponentTranscriptCursor.empty();
            }
            if (opponentTranscript == null) {
                try {
                    opponentTranscript = OpponentTranscript.load(opponentTranscriptFile, opponentTranscriptMismatchPath);
                } catch (IOException e) {
                    throw new IllegalStateException("failed to load opponent transcript: " + opponentTranscriptFile, e);
                }
            }
            return opponentTranscript.cursor(job);
        }

        static Args parse(String[] args) {
            Args out = new Args();
            Map<String, String> kv = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if (!a.startsWith("--")) {
                    continue;
                }
                String key = a.substring(2);
                String value = "true";
                int eq = key.indexOf('=');
                if (eq >= 0) {
                    value = key.substring(eq + 1);
                    key = key.substring(0, eq);
                } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    value = args[++i];
                }
                kv.put(key, value);
            }
            if (kv.containsKey("agent-deck-list")) out.agentDeckList = kv.get("agent-deck-list");
            if (kv.containsKey("opp-deck-list")) out.oppDeckList = kv.get("opp-deck-list");
            if (kv.containsKey("agent-opening-hand")) out.agentOpeningHandNames = parseCardNames(kv.get("agent-opening-hand"));
            if (kv.containsKey("opp-opening-hand")) out.oppOpeningHandNames = parseCardNames(kv.get("opp-opening-hand"));
            if (kv.containsKey("agent-opening-hand-pool")) out.agentOpeningHandPool = parseCardNamePool(kv.get("agent-opening-hand-pool"));
            if (kv.containsKey("opp-opening-hand-pool")) out.oppOpeningHandPool = parseCardNamePool(kv.get("opp-opening-hand-pool"));
            if (kv.containsKey("agent-opening-hand-pool-file")) {
                out.agentOpeningHandPoolFile = Paths.get(kv.get("agent-opening-hand-pool-file")).toAbsolutePath().normalize();
                out.agentOpeningHandPool = parseCardNamePoolFile(out.agentOpeningHandPoolFile);
            }
            if (kv.containsKey("opp-opening-hand-pool-file")) {
                out.oppOpeningHandPoolFile = Paths.get(kv.get("opp-opening-hand-pool-file")).toAbsolutePath().normalize();
                out.oppOpeningHandPool = parseCardNamePoolFile(out.oppOpeningHandPoolFile);
            }
            if (kv.containsKey("out")) out.outDir = Paths.get(kv.get("out")).toAbsolutePath().normalize();
            if (kv.containsKey("replay-file")) out.replayFile = Paths.get(kv.get("replay-file")).toAbsolutePath().normalize();
            if (kv.containsKey("export-training-data-file")) {
                out.exportTrainingDataFile = Paths.get(kv.get("export-training-data-file")).toAbsolutePath().normalize();
            }
            if (kv.containsKey("export-trajectory-data-file")) {
                out.exportTrajectoryDataFile = Paths.get(kv.get("export-trajectory-data-file")).toAbsolutePath().normalize();
            }
            if (kv.containsKey("import-training-data-path")) {
                out.importTrainingDataPath = Paths.get(kv.get("import-training-data-path")).toAbsolutePath().normalize();
            }
            if (kv.containsKey("import-trajectory-data-path")) {
                out.importTrajectoryDataPath = Paths.get(kv.get("import-trajectory-data-path")).toAbsolutePath().normalize();
            }
            if (kv.containsKey("import-flat-as-terminal-episodes")) {
                out.importFlatAsTerminalEpisodes = Boolean.parseBoolean(kv.get("import-flat-as-terminal-episodes"));
            }
            if (kv.containsKey("score-training-data-path")) {
                out.scoreTrainingDataPath = Paths.get(kv.get("score-training-data-path")).toAbsolutePath().normalize();
            }
            if (kv.containsKey("replay-deviation-training-data-file")) {
                out.replayDeviationTrainingDataFile = Paths.get(kv.get("replay-deviation-training-data-file"))
                        .toAbsolutePath().normalize();
            }
            if (kv.containsKey("replay-dagger-training-data-file")) {
                out.replayDaggerTrainingDataFile = Paths.get(kv.get("replay-dagger-training-data-file"))
                        .toAbsolutePath().normalize();
            }
            out.forceOpponentTranscript = envFlag("EVAL_REPLAY_FORCE_OPPONENT_TRANSCRIPT");
            String envOpponentTranscriptFile = System.getenv()
                    .getOrDefault("EVAL_REPLAY_OPPONENT_TRANSCRIPT_FILE", "").trim();
            if (envOpponentTranscriptFile.isEmpty()) {
                envOpponentTranscriptFile = System.getenv()
                        .getOrDefault("EVAL_OPPONENT_TRANSCRIPT_FILE", "").trim();
            }
            if (!envOpponentTranscriptFile.isEmpty()) {
                out.opponentTranscriptFile = Paths.get(envOpponentTranscriptFile).toAbsolutePath().normalize();
            }
            String envOpponentTranscriptMismatchFile = System.getenv()
                    .getOrDefault("EVAL_REPLAY_OPPONENT_TRANSCRIPT_MISMATCH_FILE", "").trim();
            if (!envOpponentTranscriptMismatchFile.isEmpty()) {
                out.opponentTranscriptMismatchPath = Paths.get(envOpponentTranscriptMismatchFile)
                        .toAbsolutePath().normalize();
            }
            out.policyInputDump = envFlag("RL_POLICY_INPUT_DUMP")
                    || envFlag("EVAL_REPLAY_POLICY_INPUT_DUMP");
            String envPolicyDumpFile = System.getenv().getOrDefault("RL_POLICY_INPUT_DUMP_FILE", "").trim();
            if (envPolicyDumpFile.isEmpty()) {
                envPolicyDumpFile = System.getenv().getOrDefault("EVAL_REPLAY_POLICY_INPUT_DUMP_FILE", "").trim();
            }
            if (!envPolicyDumpFile.isEmpty()) {
                out.policyInputDumpPath = Paths.get(envPolicyDumpFile).toAbsolutePath().normalize();
            }
            if (kv.containsKey("policy-input-dump")) out.policyInputDump = Boolean.parseBoolean(kv.get("policy-input-dump"));
            if (kv.containsKey("policy-input-dump-file")) {
                out.policyInputDumpPath = Paths.get(kv.get("policy-input-dump-file")).toAbsolutePath().normalize();
            }
            out.policyInferenceProbe = envFlag("RL_POLICY_INFERENCE_PROBE")
                    || envFlag("EVAL_REPLAY_POLICY_INFERENCE_PROBE");
            String envPolicyInferenceProbeFile = System.getenv()
                    .getOrDefault("RL_POLICY_INFERENCE_PROBE_FILE", "").trim();
            if (envPolicyInferenceProbeFile.isEmpty()) {
                envPolicyInferenceProbeFile = System.getenv()
                        .getOrDefault("EVAL_REPLAY_POLICY_INFERENCE_PROBE_FILE", "").trim();
            }
            if (!envPolicyInferenceProbeFile.isEmpty()) {
                out.policyInferenceProbePath = Paths.get(envPolicyInferenceProbeFile).toAbsolutePath().normalize();
            }
            if (kv.containsKey("policy-inference-probe")) {
                out.policyInferenceProbe = Boolean.parseBoolean(kv.get("policy-inference-probe"));
            }
            if (kv.containsKey("policy-inference-probe-file")) {
                out.policyInferenceProbePath = Paths.get(kv.get("policy-inference-probe-file"))
                        .toAbsolutePath().normalize();
            }
            if (kv.containsKey("score-max-examples")) out.scoreMaxExamples = Integer.parseInt(kv.get("score-max-examples"));
            if (kv.containsKey("fit-score-probe")) out.fitScoreProbe = Boolean.parseBoolean(kv.get("fit-score-probe"));
            if (kv.containsKey("collect-only")) out.collectOnly = Boolean.parseBoolean(kv.get("collect-only"));
            if (kv.containsKey("replay-max-scenarios")) out.replayMaxScenarios = Integer.parseInt(kv.get("replay-max-scenarios"));
            if (kv.containsKey("forced-prefix-replay")) out.forcedPrefixReplay = Boolean.parseBoolean(kv.get("forced-prefix-replay"));
            if (kv.containsKey("checkpoint-branch-probe")) {
                out.checkpointBranchProbe = Boolean.parseBoolean(kv.get("checkpoint-branch-probe"));
            }
            if (kv.containsKey("force-opponent-transcript")) {
                out.forceOpponentTranscript = Boolean.parseBoolean(kv.get("force-opponent-transcript"));
            }
            if (kv.containsKey("opponent-transcript-file")) {
                out.opponentTranscriptFile = Paths.get(kv.get("opponent-transcript-file"))
                        .toAbsolutePath().normalize();
            }
            if (kv.containsKey("opponent-transcript-mismatch-file")) {
                out.opponentTranscriptMismatchPath = Paths.get(kv.get("opponent-transcript-mismatch-file"))
                        .toAbsolutePath().normalize();
            }
            if (kv.containsKey("replay-deviation-repeat")) out.replayDeviationRepeat = Integer.parseInt(kv.get("replay-deviation-repeat"));
            if (kv.containsKey("initial-prefix")) out.initialPrefixChoices = parsePrefixKey(kv.get("initial-prefix"));
            if (kv.containsKey("scenarios")) out.scenarios = Integer.parseInt(kv.get("scenarios"));
            if (kv.containsKey("batch-size")) out.batchSize = Integer.parseInt(kv.get("batch-size"));
            if (kv.containsKey("timeout-sec")) out.timeoutSec = Integer.parseInt(kv.get("timeout-sec"));
            if (kv.containsKey("scenario-timeout-sec")) out.scenarioTimeoutSec = Integer.parseInt(kv.get("scenario-timeout-sec"));
            if (kv.containsKey("max-game-turns")) out.maxGameTurns = Integer.parseInt(kv.get("max-game-turns"));
            if (kv.containsKey("report-every")) out.reportEvery = Integer.parseInt(kv.get("report-every"));
            if (kv.containsKey("workers")) out.workers = Integer.parseInt(kv.get("workers"));
            if (kv.containsKey("post-train-wait-ms")) out.postTrainWaitMs = Integer.parseInt(kv.get("post-train-wait-ms"));
            if (kv.containsKey("seed")) out.seed = Long.parseLong(kv.get("seed"));
            if (kv.containsKey("opponent")) out.opponentMode = kv.get("opponent").trim().toLowerCase(Locale.ROOT);
            if (kv.containsKey("cp7-skill")) out.cp7Skill = Integer.parseInt(kv.get("cp7-skill"));
            if (kv.containsKey("max-decision-depth")) out.maxDecisionDepth = Integer.parseInt(kv.get("max-decision-depth"));
            if (kv.containsKey("top-k")) out.topK = Integer.parseInt(kv.get("top-k"));
            if (kv.containsKey("random-extra")) out.randomExtra = Integer.parseInt(kv.get("random-extra"));
            if (kv.containsKey("train-epochs")) out.trainEpochs = Integer.parseInt(kv.get("train-epochs"));
            if (kv.containsKey("candidate-permutations")) out.candidatePermutations = Integer.parseInt(kv.get("candidate-permutations"));
            if (kv.containsKey("max-train-examples")) out.maxTrainExamples = Integer.parseInt(kv.get("max-train-examples"));
            if (kv.containsKey("min-target-margin")) out.minTargetMargin = Double.parseDouble(kv.get("min-target-margin"));
            if (kv.containsKey("policy-miss-only")) out.policyMissOnly = Boolean.parseBoolean(kv.get("policy-miss-only"));
            if (kv.containsKey("stop-after-examples")) out.stopAfterExamples = Integer.parseInt(kv.get("stop-after-examples"));
            if (kv.containsKey("stop-after-winning-trajectories")) out.stopAfterWinningTrajectories = Integer.parseInt(kv.get("stop-after-winning-trajectories"));
            if (kv.containsKey("trajectory-final-reward")) out.trajectoryFinalReward = Double.parseDouble(kv.get("trajectory-final-reward"));
            if (kv.containsKey("winning-prefix-mode")) out.winningPrefixMode = Boolean.parseBoolean(kv.get("winning-prefix-mode"));
            if (kv.containsKey("max-prefix-depth")) out.maxPrefixDepth = Integer.parseInt(kv.get("max-prefix-depth"));
            if (kv.containsKey("train-prefix-depth")) out.trainPrefixDepth = Integer.parseInt(kv.get("train-prefix-depth"));
            if (kv.containsKey("max-search-nodes")) out.maxSearchNodes = Integer.parseInt(kv.get("max-search-nodes"));
            if (kv.containsKey("branch-subtree-search-nodes")) out.branchSubtreeSearchNodes = Integer.parseInt(kv.get("branch-subtree-search-nodes"));
            if (kv.containsKey("max-winning-prefixes-per-scenario")) out.maxWinningPrefixesPerScenario = Integer.parseInt(kv.get("max-winning-prefixes-per-scenario"));
            if (kv.containsKey("skip-pass-training")) out.skipPassTraining = Boolean.parseBoolean(kv.get("skip-pass-training"));
            if (kv.containsKey("skip-blank-training")) out.skipBlankTraining = Boolean.parseBoolean(kv.get("skip-blank-training"));
            if (kv.containsKey("skip-mulligan-training")) out.skipMulliganTraining = Boolean.parseBoolean(kv.get("skip-mulligan-training"));
            if (kv.containsKey("include-action-text-regex")) out.includeActionTextRegex = kv.get("include-action-text-regex");
            if (kv.containsKey("avoid-losing-action-text-regex")) out.avoidLosingActionTextRegex = kv.get("avoid-losing-action-text-regex");
            if (kv.containsKey("avoid-losing-strict-negative")) out.avoidLosingStrictNegative = Boolean.parseBoolean(kv.get("avoid-losing-strict-negative"));
            if (kv.containsKey("avoid-losing-mask-baseline-only")) out.avoidLosingMaskBaselineOnly = Boolean.parseBoolean(kv.get("avoid-losing-mask-baseline-only"));
            if (kv.containsKey("branch-return-targets")) out.branchReturnTargets = Boolean.parseBoolean(kv.get("branch-return-targets"));
            if (kv.containsKey("candidate-q-branch-returns")) out.branchReturnTargets = Boolean.parseBoolean(kv.get("candidate-q-branch-returns"));
            if (kv.containsKey("branch-return-balance")) out.branchReturnBalance = Boolean.parseBoolean(kv.get("branch-return-balance"));
            if (kv.containsKey("branch-return-max-negatives-per-positive")) {
                out.branchReturnMaxNegativesPerPositive = Integer.parseInt(kv.get("branch-return-max-negatives-per-positive"));
            }
            if (kv.containsKey("baseline-losing-alternative-only")) out.baselineLosingAlternativeOnly = Boolean.parseBoolean(kv.get("baseline-losing-alternative-only"));
            if (kv.containsKey("branch-value-probe")) out.branchValueProbe = Boolean.parseBoolean(kv.get("branch-value-probe"));
            if (kv.containsKey("branch-trajectory-mode")) out.branchTrajectoryMode = Boolean.parseBoolean(kv.get("branch-trajectory-mode"));
            if (kv.containsKey("branch-trajectory-first-post-target-only")) {
                out.branchTrajectoryFirstPostTargetOnly = Boolean.parseBoolean(kv.get("branch-trajectory-first-post-target-only"));
            }
            if (kv.containsKey("branch-trajectory-pair-mode")) {
                out.branchTrajectoryPairMode = Boolean.parseBoolean(kv.get("branch-trajectory-pair-mode"));
            }
            if (kv.containsKey("branch-trajectory-require-training-example")) {
                out.branchTrajectoryRequireTrainingExample = Boolean.parseBoolean(
                        kv.get("branch-trajectory-require-training-example"));
            }
            if (kv.containsKey("target-temperature")) out.targetTemperature = Double.parseDouble(kv.get("target-temperature"));
            if (kv.containsKey("win-turn-bonus")) out.winTurnBonus = Double.parseDouble(kv.get("win-turn-bonus"));
            if (kv.containsKey("loss-turn-bonus")) out.lossTurnBonus = Double.parseDouble(kv.get("loss-turn-bonus"));
            if (kv.containsKey("skip-pass-best")) out.skipPassBest = Boolean.parseBoolean(kv.get("skip-pass-best"));
            if (kv.containsKey("depth-first-search")) out.depthFirstSearch = Boolean.parseBoolean(kv.get("depth-first-search"));
            if (kv.containsKey("prefix-sibling-contrast")) out.prefixSiblingContrast = Boolean.parseBoolean(kv.get("prefix-sibling-contrast"));
            if (kv.containsKey("prefix-sibling-contrast-search-nodes")) out.prefixSiblingContrastSearchNodes = Integer.parseInt(kv.get("prefix-sibling-contrast-search-nodes"));
            if (kv.containsKey("train-root-mulligan-on-no-win")) out.trainRootMulliganOnNoWin = Boolean.parseBoolean(kv.get("train-root-mulligan-on-no-win"));
            if (kv.containsKey("train-root-mulligan-only")) out.trainRootMulliganOnly = Boolean.parseBoolean(kv.get("train-root-mulligan-only"));
            if (kv.containsKey("generic-branch-order")) out.genericBranchOrder = Boolean.parseBoolean(kv.get("generic-branch-order"));
            if (kv.containsKey("tactic-autopilot")) out.tacticAutopilot = Boolean.parseBoolean(kv.get("tactic-autopilot"));
            if (kv.containsKey("no-search-model-scoring")) out.noSearchModelScoring = Boolean.parseBoolean(kv.get("no-search-model-scoring"));
            if (kv.containsKey("pass-macro-depths")) out.passMacroDepths = parseIntList(kv.get("pass-macro-depths"));
            if (kv.containsKey("terminal-mode")) out.terminalMode = parseTerminalMode(kv.get("terminal-mode"));
            if (kv.containsKey("action-types")) out.targetTypes = parseActionTypes(kv.get("action-types"));

            out.scenarios = Math.max(1, out.scenarios);
            out.batchSize = Math.max(1, out.batchSize);
            out.timeoutSec = Math.max(1, out.timeoutSec);
            out.scenarioTimeoutSec = Math.max(0, out.scenarioTimeoutSec);
            out.maxGameTurns = Math.max(0, Math.min(200, out.maxGameTurns));
            out.reportEvery = Math.max(1, out.reportEvery);
            out.workers = Math.max(1, Math.min(out.workers, out.scenarios));
            out.maxDecisionDepth = Math.max(1, Math.min(32, out.maxDecisionDepth));
            out.topK = Math.max(1, Math.min(StateSequenceBuilder.TrainingData.MAX_CANDIDATES, out.topK));
            out.randomExtra = Math.max(0, Math.min(StateSequenceBuilder.TrainingData.MAX_CANDIDATES, out.randomExtra));
            out.trainEpochs = Math.max(1, Math.min(100, out.trainEpochs));
            out.candidatePermutations = Math.max(1, Math.min(16, out.candidatePermutations));
            out.maxTrainExamples = Math.max(0, out.maxTrainExamples);
            out.minTargetMargin = Math.max(0.0, Math.min(1.0, out.minTargetMargin));
            out.stopAfterExamples = Math.max(0, out.stopAfterExamples);
            out.stopAfterWinningTrajectories = Math.max(0, out.stopAfterWinningTrajectories);
            out.trajectoryFinalReward = Math.max(-1.0, Math.min(1.0, out.trajectoryFinalReward));
            out.scoreMaxExamples = Math.max(1, out.scoreMaxExamples);
            out.replayMaxScenarios = Math.max(0, out.replayMaxScenarios);
            out.replayDeviationRepeat = Math.max(1, Math.min(1000, out.replayDeviationRepeat));
            out.maxPrefixDepth = Math.max(1, Math.min(32, out.maxPrefixDepth));
            out.trainPrefixDepth = Math.max(1, Math.min(32, out.trainPrefixDepth));
            out.maxSearchNodes = Math.max(1, Math.min(100000, out.maxSearchNodes));
            out.branchSubtreeSearchNodes = Math.max(0, Math.min(10000, out.branchSubtreeSearchNodes));
            out.prefixSiblingContrastSearchNodes = Math.max(0, Math.min(10000, out.prefixSiblingContrastSearchNodes));
            out.maxWinningPrefixesPerScenario = Math.max(1, Math.min(1000, out.maxWinningPrefixesPerScenario));
            if (out.winningPrefixMode && out.maxDecisionDepth < out.trainPrefixDepth) {
                out.maxDecisionDepth = out.trainPrefixDepth;
            }
            out.targetTemperature = Math.max(0.01, Math.min(10.0, out.targetTemperature));
            out.winTurnBonus = Math.max(0.0, Math.min(1.0, out.winTurnBonus));
            out.lossTurnBonus = Math.max(0.0, Math.min(0.99, out.lossTurnBonus));
            out.branchReturnMaxNegativesPerPositive = Math.max(0, Math.min(1000, out.branchReturnMaxNegativesPerPositive));
            if (out.branchReturnBalance && !out.branchReturnTargets) {
                throw new IllegalArgumentException("--branch-return-balance requires --branch-return-targets");
            }
            if (!"rl".equals(out.opponentMode) && !"cp7".equals(out.opponentMode)) {
                throw new IllegalArgumentException("--opponent must be rl or cp7");
            }
            if (out.targetTypes.isEmpty()) {
                throw new IllegalArgumentException("--action-types produced an empty set");
            }
            return out;
        }

        private static boolean envFlag(String key) {
            String value = System.getenv().getOrDefault(key, "").trim();
            return "1".equals(value)
                    || "true".equalsIgnoreCase(value)
                    || "yes".equalsIgnoreCase(value)
                    || "on".equalsIgnoreCase(value);
        }

        private static Set<StateSequenceBuilder.ActionType> parseActionTypes(String raw) {
            EnumSet<StateSequenceBuilder.ActionType> out = EnumSet.noneOf(StateSequenceBuilder.ActionType.class);
            for (String token : raw.split(",")) {
                String s = token.trim();
                if (s.isEmpty()) {
                    continue;
                }
                out.add(StateSequenceBuilder.ActionType.valueOf(s));
            }
            return out;
        }

        private static List<String> parseCardNames(String raw) {
            if (raw == null || raw.trim().isEmpty()) {
                return Collections.emptyList();
            }
            List<String> out = new ArrayList<>();
            for (String token : raw.split("[;,]")) {
                String name = token.trim().replace('_', ' ');
                if (!name.isEmpty()) {
                    out.add(name);
                }
            }
            return out;
        }

        private static List<List<String>> parseCardNamePool(String raw) {
            if (raw == null || raw.trim().isEmpty()) {
                return Collections.emptyList();
            }
            List<List<String>> out = new ArrayList<>();
            for (String group : raw.split("\\|")) {
                List<String> hand = parseCardNames(group);
                if (!hand.isEmpty()) {
                    out.add(hand);
                }
            }
            return out;
        }

        private static List<List<String>> parseCardNamePoolFile(Path path) {
            if (path == null) {
                return Collections.emptyList();
            }
            List<List<String>> out = new ArrayList<>();
            try {
                for (String raw : Files.readAllLines(path, StandardCharsets.UTF_8)) {
                    String line = raw == null ? "" : raw.trim();
                    if (line.isEmpty() || line.startsWith("#")) {
                        continue;
                    }
                    List<String> hand = parseCardNames(line);
                    if (!hand.isEmpty()) {
                        out.add(hand);
                    }
                }
            } catch (IOException e) {
                throw new IllegalArgumentException("Could not read hand pool file: " + path, e);
            }
            return out;
        }

        private static TerminalMode parseTerminalMode(String raw) {
            String s = raw == null ? "" : raw.trim()
                    .replace('-', '_')
                    .toUpperCase(Locale.ROOT);
            if ("SPY".equals(s) || "SPY_COMBO".equals(s) || "SPY_MILESTONE".equals(s)) {
                return TerminalMode.SPY_COMBO_MILESTONE;
            }
            if ("SPY_ONLY".equals(s) || "SPY_COMBO_ONLY".equals(s)
                    || "SPY_MILESTONE_ONLY".equals(s) || "SPY_COMBO_MILESTONE_STRICT".equals(s)) {
                return TerminalMode.SPY_COMBO_MILESTONE_ONLY;
            }
            if ("SPY_BALUSTRADE".equals(s) || "SPY_BALUSTRADE_REACHED".equals(s)
                    || "BALUSTRADE".equals(s) || "BALUSTRADE_SPY".equals(s)) {
                return TerminalMode.SPY_BALUSTRADE_REACHED;
            }
            if ("SPY_LANDLESS_COMBO_WIN".equals(s) || "SPY_LANDLESS_WIN".equals(s)
                    || "LANDLESS_COMBO_WIN".equals(s)) {
                return TerminalMode.SPY_LANDLESS_COMBO_WIN;
            }
            return TerminalMode.valueOf(s);
        }
    }
}
