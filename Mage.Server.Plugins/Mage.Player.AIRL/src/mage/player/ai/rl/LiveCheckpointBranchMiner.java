package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import mage.player.ai.ComputerPlayerRL;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * Validates and branches durable live-eval checkpoints without reconstructing
 * the game from the original replay prefix.
 */
public final class LiveCheckpointBranchMiner {

    private static final String CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,selected_indices,selected_texts,"
                    + "candidate_hash,state_hash,rng_state_hash,source_reentry_a_matched,source_reentry_b_matched,"
                    + "classification,source_terminal,source_won,source_lost,source_error,alternate_indices,"
                    + "alternate_texts,alternate_terminal,alternate_won,alternate_lost,alternate_error,"
                    + "alternate_attempt_count,alternate_terminal_count,alternate_win_count,alternate_outcomes,"
                    + "positive_confirmation_count,positive_confirmation_pass_count,positive_confirmation_outcomes,"
                    + "reentry_a_candidate_hash,reentry_b_candidate_hash,reentry_a_state_hash,reentry_b_state_hash,"
                    + "reentry_a_reason,reentry_b_reason\n";
    private static final String SELECTION_CSV_HEADER =
            "rank,snapshot_path,score,game_key,ordinal,decision_number,action_type,candidate_count,"
                    + "selected_indices,selected_texts,selected_prob,value_score,nonpass_candidate_count,"
                    + "nonpass_alternate_count,spell_candidate_count,mana_candidate_count,pass_candidate_count,"
                    + "turn,own_life,own_graveyard_count,opponent_permanent_count,score_reasons,load_error\n";
    private static final String VALUE_TREE_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,candidate_hash,state_hash,rng_state_hash,"
                    + "action_indices,action_texts,is_source,rollouts,terminal_count,win_count,loss_count,draw_count,"
                    + "error_count,not_terminal_count,win_rate,loss_rate,terminal_rate,value_mean,delta_vs_source,"
                    + "importance_score,outcomes\n";
    private static final String VALUE_TREE_SUMMARY_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,candidate_hash,state_hash,rng_state_hash,"
                    + "classification,source_indices,source_texts,source_win_rate,source_loss_rate,source_terminal_rate,"
                    + "best_indices,best_texts,best_win_rate,best_loss_rate,best_terminal_rate,delta_win_rate,"
                    + "importance_score,actions_evaluated,total_rollouts,terminal_rollouts,reentry_a_matched,"
                    + "reentry_b_matched,reentry_a_candidate_hash,reentry_b_candidate_hash,reentry_a_state_hash,"
                    + "reentry_b_state_hash,reentry_a_reason,reentry_b_reason,error\n";

    private LiveCheckpointBranchMiner() {
    }

    public static void main(String[] args) throws Exception {
        Config cfg = Config.parse(args);
        Files.createDirectories(cfg.outDir);
        Selection selection = selectSnapshots(cfg);
        writeSelectionManifest(cfg, selection.selected);
        if (cfg.valueTree) {
            runValueTreeMode(cfg, selection);
            return;
        }

        Path csvPath = cfg.outDir.resolve("live_checkpoint_branch_probe.csv");
        Files.write(csvPath, CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        int processed = 0;
        Counts counts = new Counts();
        for (SnapshotCandidate candidate : selection.selected) {
            if (cfg.maxSnapshots > 0 && processed >= cfg.maxSnapshots) {
                break;
            }
            if (candidate.loadError != null && !candidate.loadError.isEmpty()) {
                BranchRow row = BranchRow.loadError(candidate.path, candidate.loadError);
                appendRow(csvPath, row);
                counts.add(row.classification);
                processed++;
                continue;
            }
            BranchRow row = probeSnapshot(candidate.path, candidate.snapshot, cfg);
            appendRow(csvPath, row);
            counts.add(row.classification);
            processed++;
        }
        writeReadme(cfg, csvPath, processed, selection.discoveredPathCount, selection.eligibleCount, counts);
        System.out.println("live checkpoint branch miner wrote " + processed + " row(s) to " + csvPath);
        System.out.println("classification counts: " + counts.values);
    }

    private static void runValueTreeMode(Config cfg, Selection selection) throws Exception {
        Path actionCsv = cfg.outDir.resolve("counterfactual_value_tree.csv");
        Path summaryCsv = cfg.outDir.resolve("counterfactual_value_tree_summary.csv");
        Files.write(actionCsv, VALUE_TREE_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        Files.write(summaryCsv, VALUE_TREE_SUMMARY_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        int processed = 0;
        int actionRows = 0;
        Counts counts = new Counts();
        for (SnapshotCandidate candidate : selection.selected) {
            if (cfg.maxSnapshots > 0 && processed >= cfg.maxSnapshots) {
                break;
            }
            ValueTreeSummary summary;
            if (candidate.loadError != null && !candidate.loadError.isEmpty()) {
                summary = ValueTreeSummary.loadError(candidate.path, candidate.loadError);
            } else {
                ValueTreeResult result = probeValueTree(candidate.path, candidate.snapshot, cfg);
                summary = result.summary;
                actionRows += result.actions.size();
                appendValueActionRows(actionCsv, result.actions);
            }
            appendValueSummary(summaryCsv, summary);
            counts.add(summary.classification);
            processed++;
        }
        writeValueTreeReadme(cfg, actionCsv, summaryCsv, processed,
                selection.discoveredPathCount, selection.eligibleCount, actionRows, counts);
        System.out.println("counterfactual value tree wrote " + actionRows + " action row(s) to " + actionCsv);
        System.out.println("counterfactual value tree wrote " + processed + " summary row(s) to " + summaryCsv);
        System.out.println("classification counts: " + counts.values);
    }

    private static ValueTreeResult probeValueTree(Path snapshotPath, LiveCheckpointRecorder.Snapshot snapshot, Config cfg) {
        ValueTreeSummary summary = ValueTreeSummary.fromSnapshot(snapshotPath, snapshot);
        List<ValueActionStats> actions = new ArrayList<>();
        if (snapshot == null || snapshot.gameSnapshot == null) {
            summary.classification = "snapshot_missing_game";
            summary.error = "snapshot_missing_game";
            return new ValueTreeResult(summary, actions);
        }
        if (snapshot.candidateTexts == null || snapshot.candidateTexts.size() < 2) {
            summary.classification = "snapshot_too_few_candidates";
            summary.error = "snapshot_too_few_candidates";
            return new ValueTreeResult(summary, actions);
        }
        List<Integer> sourceIndices = sanitizeIndices(snapshot.selectedIndices, snapshot.candidateTexts.size());
        if (sourceIndices.isEmpty()) {
            summary.classification = "snapshot_missing_source_choice";
            summary.error = "snapshot_missing_source_choice";
            return new ValueTreeResult(summary, actions);
        }
        BranchOutcome reentryA = runProbe(snapshot, sourceIndices, true, true,
                "value_tree_reentry_a", cfg.timeoutSec, false);
        BranchOutcome reentryB = runProbe(snapshot, sourceIndices, true, true,
                "value_tree_reentry_b", cfg.timeoutSec, false);
        summary.applyReentry(reentryA, reentryB);
        if (!reentryA.reentryMatched || !reentryB.reentryMatched) {
            summary.classification = "checkpoint_reentry_mismatch";
            summary.error = "checkpoint_reentry_mismatch";
            return new ValueTreeResult(summary, actions);
        }

        List<List<Integer>> choices = valueTreeChoices(snapshot, sourceIndices, cfg);
        if (choices.isEmpty()) {
            summary.classification = "no_tree_actions";
            summary.error = "no_tree_actions";
            return new ValueTreeResult(summary, actions);
        }
        for (List<Integer> choice : choices) {
            boolean isSource = choice.equals(sourceIndices);
            ValueActionStats stats = ValueActionStats.fromSnapshot(snapshotPath, snapshot, choice, isSource);
            int rollouts = Math.max(1, cfg.treeRollouts);
            for (int rollout = 0; rollout < rollouts; rollout++) {
                long seed = treeRolloutSeed(cfg, snapshot, choice, rollout);
                BranchOutcome outcome = runProbe(
                        snapshot,
                        choice,
                        false,
                        isSource,
                        "value_tree_" + joinInts(choice) + "_r" + rollout,
                        cfg.treeTimeoutSec,
                        cfg.postBranchAutopilot,
                        cfg.treeContinuationPolicy,
                        seed);
                stats.add(outcome);
            }
            actions.add(stats);
        }
        ValueActionStats source = null;
        ValueActionStats best = null;
        int terminalRollouts = 0;
        int totalRollouts = 0;
        for (ValueActionStats stats : actions) {
            totalRollouts += stats.rollouts;
            terminalRollouts += stats.terminalCount;
            if (stats.source) {
                source = stats;
            }
            if (best == null || valueActionRank(stats) > valueActionRank(best)) {
                best = stats;
            }
        }
        if (source == null) {
            summary.classification = "source_action_missing";
            summary.error = "source_action_missing";
            return new ValueTreeResult(summary, actions);
        }
        double sourceWinRate = source.winRate();
        double sourceLossRate = source.lossRate();
        double sourceTerminalRate = source.terminalRate();
        double bestWinRate = best == null ? 0.0 : best.winRate();
        double bestLossRate = best == null ? 0.0 : best.lossRate();
        double bestTerminalRate = best == null ? 0.0 : best.terminalRate();
        double delta = bestWinRate - sourceWinRate;
        double importance = Math.max(0.0, delta)
                * Math.max(0.0, sourceLossRate)
                * Math.min(sourceTerminalRate, bestTerminalRate);
        for (ValueActionStats stats : actions) {
            stats.deltaVsSource = stats.winRate() - sourceWinRate;
            stats.importanceScore = Math.max(0.0, stats.deltaVsSource)
                    * Math.max(0.0, sourceLossRate)
                    * Math.min(sourceTerminalRate, stats.terminalRate());
        }

        summary.sourceIndices = source.actionIndices;
        summary.sourceTexts = source.actionTexts;
        summary.sourceWinRate = sourceWinRate;
        summary.sourceLossRate = sourceLossRate;
        summary.sourceTerminalRate = sourceTerminalRate;
        if (best != null) {
            summary.bestIndices = best.actionIndices;
            summary.bestTexts = best.actionTexts;
        }
        summary.bestWinRate = bestWinRate;
        summary.bestLossRate = bestLossRate;
        summary.bestTerminalRate = bestTerminalRate;
        summary.deltaWinRate = delta;
        summary.importanceScore = importance;
        summary.actionsEvaluated = actions.size();
        summary.totalRollouts = totalRollouts;
        summary.terminalRollouts = terminalRollouts;
        summary.classification = valueTreeClassification(source, best, delta, importance);
        return new ValueTreeResult(summary, actions);
    }

    private static BranchRow probeSnapshot(Path snapshotPath, LiveCheckpointRecorder.Snapshot snapshot, Config cfg) {
        BranchRow row = BranchRow.fromSnapshot(snapshotPath, snapshot);
        if (snapshot == null || snapshot.gameSnapshot == null) {
            row.classification = "snapshot_missing_game";
            return row;
        }
        if (snapshot.candidateTexts == null || snapshot.candidateTexts.size() < 2) {
            row.classification = "snapshot_too_few_candidates";
            return row;
        }
        List<Integer> sourceIndices = sanitizeIndices(snapshot.selectedIndices, snapshot.candidateTexts.size());
        if (sourceIndices.isEmpty()) {
            row.classification = "snapshot_missing_source_choice";
            return row;
        }

        BranchOutcome reentryA = runProbe(snapshot, sourceIndices, true, true, "source_reentry_a", cfg.timeoutSec, false);
        BranchOutcome reentryB = runProbe(snapshot, sourceIndices, true, true, "source_reentry_b", cfg.timeoutSec, false);
        row.applyReentry(reentryA, reentryB);
        if (!reentryA.reentryMatched || !reentryB.reentryMatched) {
            row.classification = "checkpoint_reentry_mismatch";
            return row;
        }
        if (cfg.reentryOnly) {
            row.classification = "reentry_matched";
            return row;
        }

        BranchOutcome source = runProbe(
                snapshot,
                sourceIndices,
                false,
                true,
                "source_terminal",
                cfg.timeoutSec,
                cfg.postBranchAutopilot);
        row.applySource(source);
        if (!source.error.isEmpty()) {
            row.classification = "source_error";
            return row;
        }
        if (!source.terminal) {
            row.classification = "source_not_terminal";
            return row;
        }
        if (source.won) {
            row.classification = "source_terminal_not_loss";
            return row;
        }

        List<List<Integer>> alternateChoices = alternateChoices(snapshot, sourceIndices, cfg.maxAlternates);
        if (alternateChoices.isEmpty()) {
            row.classification = "alternate_unavailable";
            return row;
        }
        applyBestAlternate(row, snapshot, sourceIndices, alternateChoices, cfg);
        if ("clean_positive".equals(row.classification) && cfg.snapshotPath == null && cfg.requireIsolatedPositiveReprobe) {
            row.classification = "clean_positive_needs_isolated_reprobe";
        }
        return row;
    }

    private static BranchOutcome runProbe(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> forcedIndices,
            boolean stopAtReentry,
            boolean requireSourceChoiceMatch,
            String label,
            int timeoutSec,
            boolean postBranchAutopilot
    ) {
        return runProbe(snapshot, forcedIndices, stopAtReentry, requireSourceChoiceMatch,
                label, timeoutSec, postBranchAutopilot, ContinuationPolicy.STABLE, 0L);
    }

    private static BranchOutcome runProbe(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> forcedIndices,
            boolean stopAtReentry,
            boolean requireSourceChoiceMatch,
            String label,
            int timeoutSec,
            boolean postBranchAutopilot,
            ContinuationPolicy continuationPolicy,
            long rolloutSeed
    ) {
        RandomUtil.State previousRandom = RandomUtil.captureState();
        Game game = null;
        SnapshotBranchController controller =
                new SnapshotBranchController(
                        snapshot,
                        forcedIndices,
                        stopAtReentry,
                        requireSourceChoiceMatch,
                        postBranchAutopilot,
                        continuationPolicy,
                        rolloutSeed);
        BranchOutcome outcome = new BranchOutcome(label);
        try {
            RandomUtil.restoreState(snapshot.randomState);
            game = snapshot.gameSnapshot.createSimulationForAI();
            Player player = game.getPlayer(snapshot.playerId);
            if (!(player instanceof ComputerPlayerRL)) {
                outcome.error = "checkpoint_player_copy_type_mismatch:"
                        + (player == null ? "null" : player.getClass().getName());
                return outcome;
            }
            installBranchControllers(game, (ComputerPlayerRL) player, controller, postBranchAutopilot);
            try {
                resumeGameInGameThread(game, timeoutSec, label);
            } catch (EngineDecisionBranchController.BranchTerminated terminated) {
                outcome.terminationReason = terminated.getReason();
            }
            outcome.captureController(controller);
            outcome.captureTerminal(game, snapshot.playerName);
            return outcome;
        } catch (Throwable t) {
            outcome.captureController(controller);
            outcome.error = errorSummary(t);
            return outcome;
        } finally {
            if (game != null) {
                try {
                    game.end();
                } catch (Throwable ignored) {
                    // ignore cleanup failures
                }
                try {
                    game.cleanUp();
                } catch (Throwable ignored) {
                    // ignore cleanup failures
                }
            }
            RandomUtil.restoreState(previousRandom);
        }
    }

    private static void installBranchControllers(
            Game game,
            ComputerPlayerRL sourcePlayer,
            SnapshotBranchController controller,
            boolean postBranchAutopilot
    ) {
        if (!postBranchAutopilot || game == null || game.getPlayers() == null) {
            sourcePlayer.setEngineDecisionBranchController(controller);
            return;
        }
        for (Player player : game.getPlayers().values()) {
            if (player instanceof ComputerPlayerRL) {
                ((ComputerPlayerRL) player).setEngineDecisionBranchController(controller);
            }
        }
    }

    private static void resumeGameInGameThread(Game game, int joinTimeoutSec, String label) {
        long deadlineNanos = System.nanoTime() + Math.max(1L, joinTimeoutSec) * 1_000_000_000L;
        if (ThreadUtils.isRunGameThread()) {
            resumeUntilTerminal(game, deadlineNanos);
            return;
        }
        AtomicReference<Throwable> error = new AtomicReference<>(null);
        Thread gameThread = new Thread(() -> {
            try {
                resumeUntilTerminal(game, deadlineNanos);
            } catch (Throwable t) {
                error.set(t);
            }
        }, "GAME-LIVE-CHECKPOINT-" + label);
        gameThread.setDaemon(true);
        gameThread.start();

        long timeoutMs = Math.max(1L, joinTimeoutSec) * 1000L;
        try {
            gameThread.join(timeoutMs);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            try {
                game.end();
            } catch (Throwable ignored) {
                // ignore cleanup failures
            }
            throw new IllegalStateException("Interrupted while waiting for live checkpoint branch", ie);
        }
        if (gameThread.isAlive()) {
            try {
                game.end();
            } catch (Throwable ignored) {
                // ignore cleanup failures
            }
            try {
                gameThread.join(5000L);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            throw new IllegalStateException("Live checkpoint branch timeout");
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
        throw new IllegalStateException("Error while resuming live checkpoint branch", t);
    }

    private static void resumeUntilTerminal(Game game, long deadlineNanos) {
        while (!game.hasEnded()) {
            game.resume();
            if (!game.hasEnded() && System.nanoTime() >= deadlineNanos) {
                throw new IllegalStateException("Live checkpoint branch returned before terminal");
            }
        }
    }

    private static LiveCheckpointRecorder.Snapshot loadSnapshot(Path path) throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            Object obj = in.readObject();
            if (!(obj instanceof LiveCheckpointRecorder.Snapshot)) {
                throw new IllegalArgumentException("Unexpected snapshot object: "
                        + (obj == null ? "null" : obj.getClass().getName()));
            }
            return (LiveCheckpointRecorder.Snapshot) obj;
        }
    }

    private static Selection selectSnapshots(Config cfg) throws Exception {
        List<Path> paths = discoverSnapshotPaths(cfg);
        List<SnapshotCandidate> eligible = new ArrayList<>();
        for (Path path : paths) {
            SnapshotCandidate candidate = new SnapshotCandidate(path);
            try {
                candidate.snapshot = loadSnapshot(path);
                if (!cfg.actionTypes.isEmpty() && !cfg.actionTypes.contains(candidate.snapshot.actionType)) {
                    continue;
                }
                scoreSnapshot(candidate);
            } catch (Throwable t) {
                candidate.loadError = errorSummary(t);
                candidate.score = Integer.MIN_VALUE;
                candidate.scoreReasons = "load_error";
                if ("ranked".equalsIgnoreCase(cfg.selectionMode)) {
                    continue;
                }
            }
            eligible.add(candidate);
        }
        if ("ranked".equalsIgnoreCase(cfg.selectionMode)) {
            eligible.sort(Comparator
                    .comparingInt((SnapshotCandidate c) -> c.score).reversed()
                    .thenComparing(c -> c.path == null ? "" : c.path.toString()));
        }
        List<SnapshotCandidate> selected = new ArrayList<>();
        Map<String, Integer> perGame = new LinkedHashMap<>();
        int limit = cfg.maxSnapshots <= 0 ? Integer.MAX_VALUE : cfg.maxSnapshots;
        for (SnapshotCandidate candidate : eligible) {
            if (selected.size() >= limit) {
                break;
            }
            if ("ranked".equalsIgnoreCase(cfg.selectionMode) && cfg.rankedMaxPerGame > 0) {
                String key = candidate.gameKey;
                int count = perGame.getOrDefault(key, 0);
                if (count >= cfg.rankedMaxPerGame) {
                    continue;
                }
                perGame.put(key, count + 1);
            }
            candidate.rank = selected.size() + 1;
            selected.add(candidate);
        }
        return new Selection(paths.size(), eligible.size(), selected);
    }

    private static List<Path> discoverSnapshotPaths(Config cfg) throws Exception {
        List<Path> out = new ArrayList<>();
        if (cfg.snapshotPath != null) {
            out.add(cfg.snapshotPath);
            return out;
        }
        if (cfg.checkpointRoot == null) {
            throw new IllegalArgumentException("--checkpoint-root or --snapshot is required");
        }
        try (Stream<Path> stream = Files.walk(cfg.checkpointRoot)) {
            out.addAll(stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName() != null && p.getFileName().toString().endsWith(".ser.gz"))
                    .sorted()
                    .collect(Collectors.toList()));
        }
        return out;
    }

    private static void scoreSnapshot(SnapshotCandidate candidate) {
        LiveCheckpointRecorder.Snapshot snapshot = candidate.snapshot;
        if (snapshot == null) {
            candidate.score = Integer.MIN_VALUE;
            candidate.scoreReasons = "snapshot_missing";
            return;
        }
        List<String> candidates = snapshot.candidateTexts == null ? Collections.emptyList() : snapshot.candidateTexts;
        int candidateCount = candidates.size();
        List<Integer> sourceIndices = sanitizeIndices(snapshot.selectedIndices, candidateCount);
        Set<Integer> source = new HashSet<>(sourceIndices);
        String sourceText = sourceIndices.isEmpty()
                ? ""
                : candidates.get(sourceIndices.get(0));
        int nonPassCandidateCount = 0;
        int nonPassAlternateCount = 0;
        int spellCandidateCount = 0;
        int manaCandidateCount = 0;
        int passCandidateCount = 0;
        for (int i = 0; i < candidateCount; i++) {
            String text = candidates.get(i);
            if (isPassLike(text)) {
                passCandidateCount++;
            } else {
                nonPassCandidateCount++;
                if (!source.contains(i)) {
                    nonPassAlternateCount++;
                }
            }
            if (isSpellLike(text)) {
                spellCandidateCount++;
            }
            if (isManaLike(text)) {
                manaCandidateCount++;
            }
        }

        candidate.gameKey = gameKey(candidate.path);
        candidate.nonPassCandidateCount = nonPassCandidateCount;
        candidate.nonPassAlternateCount = nonPassAlternateCount;
        candidate.spellCandidateCount = spellCandidateCount;
        candidate.manaCandidateCount = manaCandidateCount;
        candidate.passCandidateCount = passCandidateCount;
        candidate.turn = parseIntAfter(snapshot.compactState, "turn=");
        candidate.ownLife = parsePerspectiveInt(snapshot.compactState, "life");
        candidate.ownGraveyardCount = parsePerspectiveInt(snapshot.compactState, "graveyard");
        candidate.opponentPermanentCount = countOpponentPermanents(snapshot.compactState);

        List<String> reasons = new ArrayList<>();
        int score = 0;
        score += addScore(reasons, "candidate_count", candidateCount * 4);
        score += addScore(reasons, "nonpass_alternates", nonPassAlternateCount * 18);
        score += addScore(reasons, "spell_candidates", spellCandidateCount * 8);
        score += addScore(reasons, "decision_depth", Math.min(Math.max(snapshot.decisionNumber, 0), 120) / 2);
        score += addScore(reasons, "turn_depth", Math.min(Math.max(candidate.turn, 0), 12) * 6);
        if (!isPassLike(sourceText)) {
            score += addScore(reasons, "source_nonpass", 18);
        } else {
            score += addScore(reasons, "source_pass_penalty", -35);
        }
        if (isSpellLike(sourceText)) {
            score += addScore(reasons, "source_spell", 20);
        }
        if (isManaLike(sourceText)) {
            score += addScore(reasons, "source_mana_penalty", -30);
        }
        if (isLandPlayLike(sourceText)) {
            score += addScore(reasons, "source_land_play_penalty", -14);
        }
        if (nonPassAlternateCount <= 0) {
            score += addScore(reasons, "no_nonpass_alternates_penalty", -80);
        }
        if (nonPassCandidateCount > 0 && nonPassCandidateCount == manaCandidateCount) {
            score += addScore(reasons, "all_mana_nonpass_penalty", -35);
        }
        if (candidateCount <= 2) {
            score += addScore(reasons, "small_candidate_set_penalty", -10);
        }
        if (!Float.isNaN(snapshot.selectedProb)) {
            if (snapshot.selectedProb < 0.35f) {
                score += addScore(reasons, "low_policy_confidence", 14);
            } else if (snapshot.selectedProb < 0.55f) {
                score += addScore(reasons, "medium_policy_confidence", 8);
            }
        }
        if (!Float.isNaN(snapshot.valueScore) && snapshot.valueScore < 0.0f) {
            score += addScore(reasons, "negative_value", Math.min(20, Math.round(-snapshot.valueScore * 80.0f)));
        }
        if (candidate.ownLife > 0 && candidate.ownLife <= 8) {
            score += addScore(reasons, "low_life_pressure", 16);
        } else if (candidate.ownLife > 0 && candidate.ownLife <= 14) {
            score += addScore(reasons, "life_pressure", 8);
        }
        if (candidate.ownGraveyardCount >= 5) {
            score += addScore(reasons, "graveyard_pressure", 8);
        }
        if (candidate.opponentPermanentCount >= 8) {
            score += addScore(reasons, "opponent_board_pressure", 10);
        } else if (candidate.opponentPermanentCount >= 5) {
            score += addScore(reasons, "opponent_board", 5);
        }

        candidate.score = score;
        candidate.scoreReasons = joinStrings(reasons);
    }

    private static int addScore(List<String> reasons, String name, int value) {
        if (value != 0) {
            reasons.add(name + "=" + value);
        }
        return value;
    }

    private static void applyBestAlternate(
            BranchRow row,
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> sourceIndices,
            List<List<Integer>> alternateChoices,
            Config cfg
    ) {
        BranchOutcome best = null;
        List<Integer> bestIndices = Collections.emptyList();
        List<String> outcomeSummaries = new ArrayList<>();
        int terminalCount = 0;
        int winCount = 0;
        int attempt = 0;
        for (List<Integer> alternateIndices : alternateChoices) {
            attempt++;
            BranchOutcome alternate = runProbe(
                    snapshot,
                    alternateIndices,
                    false,
                    false,
                    "alternate_terminal_" + attempt,
                    cfg.alternateTimeoutSec,
                    cfg.postBranchAutopilot);
            if (alternate.terminal) {
                terminalCount++;
            }
            if (alternate.won) {
                winCount++;
            }
            outcomeSummaries.add(joinInts(alternateIndices)
                    + ":"
                    + joinStrings(selectedTexts(snapshot.candidateTexts, alternateIndices))
                    + ":"
                    + alternate.shortClassification());
            if (best == null || alternateRank(alternate) > alternateRank(best)) {
                best = alternate;
                bestIndices = alternateIndices;
            }
            if (alternate.terminal && alternate.won) {
                break;
            }
        }
        row.alternateAttemptCount = attempt;
        row.alternateTerminalCount = terminalCount;
        row.alternateWinCount = winCount;
        row.alternateOutcomes = joinStrings(outcomeSummaries);
        row.alternateIndices = joinInts(bestIndices);
        row.alternateTexts = joinStrings(selectedTexts(snapshot.candidateTexts, bestIndices));
        row.applyAlternate(best);
        if (best == null) {
            row.classification = "alternate_unavailable";
        } else if (best.terminal && best.won) {
            PositiveConfirmation confirmation = confirmPositive(snapshot, sourceIndices, bestIndices, cfg);
            row.positiveConfirmationCount = confirmation.requested;
            row.positiveConfirmationPassCount = confirmation.passed;
            row.positiveConfirmationOutcomes = confirmation.outcomes;
            row.classification = confirmation.isConfirmed() ? "clean_positive" : "clean_positive_unstable";
        } else if (best.terminal && best.lost) {
            row.classification = "clean_negative";
        } else if (best.terminal) {
            row.classification = "alternate_terminal_draw";
        } else if (!best.error.isEmpty()) {
            row.classification = "alternate_error";
        } else {
            row.classification = "alternate_not_terminal";
        }
    }

    private static PositiveConfirmation confirmPositive(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> sourceIndices,
            List<Integer> alternateIndices,
            Config cfg
    ) {
        int requested = Math.max(0, cfg.confirmPositiveRepeats);
        if (requested <= 0) {
            return new PositiveConfirmation(0, 0, "");
        }
        int passed = 0;
        List<String> summaries = new ArrayList<>();
        for (int i = 1; i <= requested; i++) {
            BranchOutcome source = runProbe(
                    snapshot,
                    sourceIndices,
                    false,
                    true,
                    "positive_confirm_source_" + i,
                    cfg.timeoutSec,
                    cfg.postBranchAutopilot);
            BranchOutcome alternate = runProbe(
                    snapshot,
                    alternateIndices,
                    false,
                    false,
                    "positive_confirm_alternate_" + i,
                    cfg.alternateTimeoutSec,
                    cfg.postBranchAutopilot);
            boolean ok = source.error.isEmpty()
                    && alternate.error.isEmpty()
                    && source.reentryMatched
                    && alternate.reentryMatched
                    && source.terminal
                    && source.lost
                    && alternate.terminal
                    && alternate.won;
            if (ok) {
                passed++;
            }
            summaries.add("repeat" + i
                    + ":source=" + source.shortClassification()
                    + ";alternate=" + alternate.shortClassification()
                    + ";pass=" + ok);
        }
        return new PositiveConfirmation(requested, passed, joinStrings(summaries));
    }

    private static int alternateRank(BranchOutcome outcome) {
        if (outcome == null) {
            return 0;
        }
        if (outcome.terminal && outcome.won) {
            return 50;
        }
        if (outcome.terminal && outcome.lost) {
            return 40;
        }
        if (outcome.terminal) {
            return 30;
        }
        if (outcome.error.isEmpty()) {
            return 20;
        }
        return 10;
    }

    private static List<List<Integer>> alternateChoices(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> sourceIndices,
            int maxAlternates
    ) {
        List<List<Integer>> out = new ArrayList<>();
        int candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
        Set<Integer> source = new HashSet<>(sourceIndices == null ? Collections.emptyList() : sourceIndices);
        for (int i = 0; i < candidateCount; i++) {
            if (!source.contains(i) && !isPassLike(snapshot.candidateTexts.get(i))) {
                out.add(alternateReplacingFirstSource(sourceIndices, i, candidateCount));
                if (maxAlternates > 0 && out.size() >= maxAlternates) {
                    return out;
                }
            }
        }
        for (int i = 0; i < candidateCount; i++) {
            if (!source.contains(i) && isPassLike(snapshot.candidateTexts.get(i))) {
                out.add(alternateReplacingFirstSource(sourceIndices, i, candidateCount));
                if (maxAlternates > 0 && out.size() >= maxAlternates) {
                    return out;
                }
            }
        }
        return out;
    }

    private static List<Integer> alternateReplacingFirstSource(List<Integer> sourceIndices, int alternate, int candidateCount) {
        List<Integer> out = new ArrayList<>(sourceIndices == null ? Collections.emptyList() : sourceIndices);
        if (out.isEmpty()) {
            out.add(alternate);
        } else {
            out.set(0, alternate);
        }
        return sanitizeIndices(out, candidateCount);
    }

    private static boolean isPassLike(String text) {
        String value = text == null ? "" : text.trim();
        return "Pass".equalsIgnoreCase(value) || "PASS".equals(value);
    }

    private static boolean isDoneLike(String text) {
        String value = text == null ? "" : text.trim();
        return "DONE".equalsIgnoreCase(value);
    }

    private static boolean isStopLike(String text) {
        String value = text == null ? "" : text.trim();
        return "STOP".equalsIgnoreCase(value);
    }

    private static boolean isSpellLike(String text) {
        String value = text == null ? "" : text.trim();
        if (value.isEmpty() || isPassLike(value) || isManaLike(value) || isLandPlayLike(value)) {
            return false;
        }
        return value.startsWith("Cast ")
                || value.startsWith("Activate ")
                || value.startsWith("Attack")
                || value.startsWith("Block")
                || value.contains(": Cast ");
    }

    private static boolean isManaLike(String text) {
        String value = text == null ? "" : text.trim().toLowerCase(Locale.US);
        return value.contains("add {")
                || value.contains("add one mana")
                || value.contains("add one mana of any color")
                || (value.startsWith("{t}") && value.contains("add"));
    }

    private static boolean isLandPlayLike(String text) {
        String value = text == null ? "" : text.trim();
        return value.startsWith("Play ");
    }

    private static String gameKey(Path path) {
        if (path == null) {
            return "";
        }
        String fileName = path.getFileName() == null ? "" : path.getFileName().toString();
        int ord = fileName.indexOf("_ord");
        String game = ord > 0 ? fileName.substring(0, ord) : fileName;
        Path parent = path.getParent();
        String matchup = parent == null || parent.getFileName() == null ? "" : parent.getFileName().toString();
        return matchup + "/" + game;
    }

    private static int parseIntAfter(String text, String key) {
        if (text == null || key == null || key.isEmpty()) {
            return -1;
        }
        int start = text.indexOf(key);
        if (start < 0) {
            return -1;
        }
        start += key.length();
        int end = start;
        while (end < text.length() && (Character.isDigit(text.charAt(end)) || text.charAt(end) == '-')) {
            end++;
        }
        if (end <= start) {
            return -1;
        }
        try {
            return Integer.parseInt(text.substring(start, end));
        } catch (NumberFormatException nfe) {
            return -1;
        }
    }

    private static int parsePerspectiveInt(String compactState, String fieldName) {
        if (compactState == null || compactState.isEmpty()) {
            return -1;
        }
        String perspective = textBetween(compactState, "perspective=", ";");
        if (perspective.isEmpty()) {
            return -1;
        }
        String playerPrefix = ";player=" + perspective + ":";
        int playerStart = compactState.indexOf(playerPrefix);
        if (playerStart < 0) {
            return -1;
        }
        int playerEnd = compactState.indexOf(";player=", playerStart + 1);
        int battlefieldStart = compactState.indexOf(";battlefield=", playerStart + 1);
        int end = playerEnd >= 0 ? playerEnd : (battlefieldStart >= 0 ? battlefieldStart : compactState.length());
        String player = compactState.substring(playerStart, Math.max(playerStart, end));
        return parseIntAfter(player, ":" + fieldName + "=");
    }

    private static int countOpponentPermanents(String compactState) {
        if (compactState == null || compactState.isEmpty()) {
            return 0;
        }
        String perspective = textBetween(compactState, "perspective=", ";");
        int start = compactState.indexOf(";battlefield=");
        if (start < 0) {
            return 0;
        }
        int end = compactState.indexOf(";stack=", start + 1);
        String battlefield = compactState.substring(start, end >= 0 ? end : compactState.length());
        int count = 0;
        for (String permanent : battlefield.split("\\|")) {
            if (permanent.contains(":ctrl=") && (perspective.isEmpty() || !permanent.contains(":ctrl=" + perspective))) {
                count++;
            }
        }
        return count;
    }

    private static String textBetween(String text, String startToken, String endToken) {
        if (text == null || startToken == null || endToken == null) {
            return "";
        }
        int start = text.indexOf(startToken);
        if (start < 0) {
            return "";
        }
        start += startToken.length();
        int end = text.indexOf(endToken, start);
        if (end < 0) {
            end = text.length();
        }
        return text.substring(start, end);
    }

    private static List<Integer> sanitizeIndices(List<Integer> values, int candidateCount) {
        if (values == null || values.isEmpty() || candidateCount <= 0) {
            return Collections.emptyList();
        }
        List<Integer> out = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();
        for (Integer value : values) {
            if (value == null || value < 0 || value >= candidateCount || seen.contains(value)) {
                continue;
            }
            out.add(value);
            seen.add(value);
        }
        return out;
    }

    private static List<String> selectedTexts(List<String> candidates, List<Integer> indices) {
        if (candidates == null || indices == null || indices.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Integer idx : indices) {
            if (idx != null && idx >= 0 && idx < candidates.size()) {
                out.add(candidates.get(idx));
            }
        }
        return out;
    }

    private static void appendRow(Path csvPath, BranchRow row) throws Exception {
        Files.write(csvPath, row.toCsvLine().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void writeSelectionManifest(Config cfg, List<SnapshotCandidate> selected) throws Exception {
        Path manifest = cfg.outDir.resolve("selected_snapshots.csv");
        Files.write(manifest, SELECTION_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        if (selected == null || selected.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(selected.size() * 256);
        for (SnapshotCandidate candidate : selected) {
            LiveCheckpointRecorder.Snapshot snapshot = candidate.snapshot;
            sb.append(candidate.rank)
                    .append(",").append(csv(candidate.path == null ? "" : candidate.path.toString()))
                    .append(",").append(candidate.score)
                    .append(",").append(csv(candidate.gameKey))
                    .append(",").append(snapshot == null ? -1 : snapshot.ordinal)
                    .append(",").append(snapshot == null ? -1 : snapshot.decisionNumber)
                    .append(",").append(csv(snapshot == null ? "" : snapshot.actionType))
                    .append(",").append(snapshot == null || snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size())
                    .append(",").append(csv(snapshot == null ? "" : joinInts(snapshot.selectedIndices)))
                    .append(",").append(csv(snapshot == null ? "" : joinStrings(snapshot.selectedTexts)))
                    .append(",").append(String.format(Locale.US, "%.8f", snapshot == null ? 0.0f : snapshot.selectedProb))
                    .append(",").append(String.format(Locale.US, "%.8f", snapshot == null ? 0.0f : snapshot.valueScore))
                    .append(",").append(candidate.nonPassCandidateCount)
                    .append(",").append(candidate.nonPassAlternateCount)
                    .append(",").append(candidate.spellCandidateCount)
                    .append(",").append(candidate.manaCandidateCount)
                    .append(",").append(candidate.passCandidateCount)
                    .append(",").append(candidate.turn)
                    .append(",").append(candidate.ownLife)
                    .append(",").append(candidate.ownGraveyardCount)
                    .append(",").append(candidate.opponentPermanentCount)
                    .append(",").append(csv(candidate.scoreReasons))
                    .append(",").append(csv(candidate.loadError))
                    .append("\n");
        }
        Files.write(manifest, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void writeReadme(
            Config cfg,
            Path csvPath,
            int processed,
            int discovered,
            int eligible,
            Counts counts
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Live Checkpoint Branch Miner\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- eligible: ").append(eligible).append("\n");
        sb.append("- selection_mode: ").append(cfg.selectionMode).append("\n");
        sb.append("- ranked_max_per_game: ").append(cfg.rankedMaxPerGame).append("\n");
        sb.append("- reentry_only: ").append(cfg.reentryOnly).append("\n");
        sb.append("- post_branch_autopilot: ").append(cfg.postBranchAutopilot).append("\n");
        sb.append("- confirm_positive_repeats: ").append(cfg.confirmPositiveRepeats).append("\n");
        sb.append("- require_isolated_positive_reprobe: ").append(cfg.requireIsolatedPositiveReprobe).append("\n");
        sb.append("- timeout_sec: ").append(cfg.timeoutSec).append("\n");
        sb.append("- alternate_timeout_sec: ").append(cfg.alternateTimeoutSec).append("\n");
        sb.append("- max_alternates: ").append(cfg.maxAlternates).append("\n");
        sb.append("- selected_snapshots_csv: ").append(cfg.outDir.resolve("selected_snapshots.csv")).append("\n");
        sb.append("- csv: ").append(csvPath).append("\n");
        sb.append("- classification_counts: ").append(counts.values).append("\n");
        Files.write(cfg.outDir.resolve("README.md"), sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static void appendValueActionRows(Path csvPath, List<ValueActionStats> rows) throws Exception {
        if (rows == null || rows.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(rows.size() * 256);
        for (ValueActionStats row : rows) {
            sb.append(row.toCsvLine());
        }
        Files.write(csvPath, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void appendValueSummary(Path csvPath, ValueTreeSummary summary) throws Exception {
        Files.write(csvPath, summary.toCsvLine().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void writeValueTreeReadme(
            Config cfg,
            Path actionCsv,
            Path summaryCsv,
            int processed,
            int discovered,
            int eligible,
            int actionRows,
            Counts counts
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Counterfactual Value Tree Miner\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- eligible: ").append(eligible).append("\n");
        sb.append("- action_rows: ").append(actionRows).append("\n");
        sb.append("- selection_mode: ").append(cfg.selectionMode).append("\n");
        sb.append("- ranked_max_per_game: ").append(cfg.rankedMaxPerGame).append("\n");
        sb.append("- tree_rollouts: ").append(cfg.treeRollouts).append("\n");
        sb.append("- tree_max_actions: ").append(cfg.treeMaxActions).append("\n");
        sb.append("- tree_include_pass: ").append(cfg.treeIncludePass).append("\n");
        sb.append("- tree_continuation_policy: ").append(cfg.treeContinuationPolicy.name().toLowerCase(Locale.US)).append("\n");
        sb.append("- tree_timeout_sec: ").append(cfg.treeTimeoutSec).append("\n");
        sb.append("- tree_seed: ").append(cfg.treeSeed).append("\n");
        sb.append("- post_branch_autopilot: ").append(cfg.postBranchAutopilot).append("\n");
        sb.append("- selected_snapshots_csv: ").append(cfg.outDir.resolve("selected_snapshots.csv")).append("\n");
        sb.append("- action_csv: ").append(actionCsv).append("\n");
        sb.append("- summary_csv: ").append(summaryCsv).append("\n");
        sb.append("- classification_counts: ").append(counts.values).append("\n\n");
        sb.append("This mode estimates action importance at each serialized checkpoint by forcing each selected root action, ")
                .append("running configurable continuations, and comparing action win rates against the accepted-policy source action. ")
                .append("It is a bounded sampled tree, not an exhaustive Magic game tree.\n");
        Files.write(cfg.outDir.resolve("README.md"), sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static List<List<Integer>> valueTreeChoices(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> sourceIndices,
            Config cfg
    ) {
        int candidateCount = snapshot == null || snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
        if (candidateCount <= 0) {
            return Collections.emptyList();
        }
        List<List<Integer>> out = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        List<Integer> source = sanitizeIndices(sourceIndices, candidateCount);
        addValueTreeChoice(out, seen, source);
        for (int i = 0; i < candidateCount; i++) {
            if (cfg.treeMaxActions > 0 && out.size() >= cfg.treeMaxActions) {
                break;
            }
            String text = snapshot.candidateTexts.get(i);
            if (!cfg.treeIncludePass && isTerminalChoiceText(text) && !(source.size() == 1 && source.get(0) == i)) {
                continue;
            }
            List<Integer> choice = alternateReplacingFirstSource(source, i, candidateCount);
            addValueTreeChoice(out, seen, choice);
        }
        return out;
    }

    private static void addValueTreeChoice(List<List<Integer>> out, Set<String> seen, List<Integer> choice) {
        List<Integer> sanitized = choice == null ? Collections.emptyList() : new ArrayList<>(choice);
        String key = joinInts(sanitized);
        if (key.isEmpty() || seen.contains(key)) {
            return;
        }
        seen.add(key);
        out.add(sanitized);
    }

    private static boolean isTerminalChoiceText(String text) {
        return isPassLike(text) || isDoneLike(text) || isStopLike(text);
    }

    private static long treeRolloutSeed(
            Config cfg,
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> choice,
            int rollout
    ) {
        long seed = cfg.treeSeed;
        seed = 31L * seed + (snapshot == null || snapshot.candidateHash == null ? 0 : snapshot.candidateHash.hashCode());
        seed = 31L * seed + (snapshot == null || snapshot.stateHash == null ? 0 : snapshot.stateHash.hashCode());
        seed = 31L * seed + joinInts(choice).hashCode();
        seed = 31L * seed + rollout;
        return seed;
    }

    private static double valueActionRank(ValueActionStats stats) {
        if (stats == null) {
            return Double.NEGATIVE_INFINITY;
        }
        return stats.winRate() * 10_000.0
                + stats.terminalRate() * 100.0
                - stats.lossRate()
                - (stats.source ? 0.0001 : 0.0);
    }

    private static String valueTreeClassification(
            ValueActionStats source,
            ValueActionStats best,
            double delta,
            double importance
    ) {
        if (source == null || best == null) {
            return "no_value_tree_evidence";
        }
        if (source.terminalCount <= 0 && best.terminalCount <= 0) {
            return "no_terminal_evidence";
        }
        if (best == source || delta <= 0.0) {
            return "no_better_action";
        }
        if (source.lossRate() >= 0.999 && best.winRate() >= 0.999) {
            return "dominant_correction";
        }
        if (importance >= 0.50 || delta >= 0.50) {
            return "strong_correction";
        }
        if (importance >= 0.20 || delta >= 0.20) {
            return "moderate_correction";
        }
        return "weak_correction";
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (Integer value : values) {
            out.add(String.valueOf(value == null ? -1 : value));
        }
        return String.join("|", out);
    }

    private static String joinStrings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        return String.join("|", values);
    }

    private static String csv(String value) {
        String s = value == null ? "" : value;
        return "\"" + s.replace("\"", "\"\"").replace("\r", " ").replace("\n", " ") + "\"";
    }

    private static String formatDouble(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return "0.000000";
        }
        return String.format(Locale.US, "%.6f", value);
    }

    private static String errorSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String message = t.getMessage();
        return t.getClass().getSimpleName() + (message == null || message.isEmpty() ? "" : ": " + message);
    }

    private static final class SnapshotBranchController implements EngineDecisionBranchController {
        private final LiveCheckpointRecorder.Snapshot snapshot;
        private final List<Integer> forcedIndices;
        private final boolean stopAtReentry;
        private final boolean requireSourceChoiceMatch;
        private final boolean postBranchAutopilot;
        private final ContinuationPolicy continuationPolicy;
        private final Random rolloutRandom;

        private boolean seen;
        private boolean reentryMatched;
        private String reason = "";
        private String actualActionType = "";
        private String actualCandidateHash = "";
        private String actualStateHash = "";
        private List<String> actualCandidateTexts = Collections.emptyList();
        private List<String> selectedTexts = Collections.emptyList();

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> forcedIndices,
                boolean stopAtReentry,
                boolean requireSourceChoiceMatch,
                boolean postBranchAutopilot,
                ContinuationPolicy continuationPolicy,
                long rolloutSeed
        ) {
            this.snapshot = snapshot;
            this.forcedIndices = forcedIndices == null
                    ? Collections.emptyList()
                    : new ArrayList<>(forcedIndices);
            this.stopAtReentry = stopAtReentry;
            this.requireSourceChoiceMatch = requireSourceChoiceMatch;
            this.postBranchAutopilot = postBranchAutopilot;
            this.continuationPolicy = continuationPolicy == null ? ContinuationPolicy.STABLE : continuationPolicy;
            this.rolloutRandom = new Random(rolloutSeed);
        }

        @Override
        public boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
            return true;
        }

        @Override
        public <T> Choice onDecision(DecisionContext<T> context) {
            if (seen) {
                return postBranchAutopilotChoice(context);
            }
            if (context == null
                    || context.player == null
                    || snapshot == null
                    || snapshot.playerId == null
                    || !snapshot.playerId.equals(context.player.getId())) {
                return Choice.none();
            }
            seen = true;
            actualActionType = context.actionType == null ? "" : context.actionType.name();
            actualCandidateHash = context.candidateHash;
            actualStateHash = context.stateHash;
            actualCandidateTexts = new ArrayList<>(context.candidateTexts);

            List<Integer> sanitized = sanitizeIndices(forcedIndices, context.candidateCount);
            selectedTexts = LiveCheckpointBranchMiner.selectedTexts(context.candidateTexts, sanitized);
            boolean actionMatched = actualActionType.equals(snapshot.actionType);
            boolean candidatesMatched = context.candidateTexts.equals(snapshot.candidateTexts);
            boolean selectedIndicesMatched = sanitized.equals(sanitizeIndices(snapshot.selectedIndices, context.candidateCount));
            boolean selectedTextsMatched = selectedTexts.equals(snapshot.selectedTexts);
            reentryMatched = actionMatched
                    && candidatesMatched
                    && (!requireSourceChoiceMatch || (selectedIndicesMatched && selectedTextsMatched))
                    && !sanitized.isEmpty();
            if (!actionMatched) {
                reason = "action_type_mismatch";
            } else if (!candidatesMatched) {
                reason = "candidate_text_mismatch";
            } else if (sanitized.isEmpty()) {
                reason = "forced_indices_invalid";
            } else if (requireSourceChoiceMatch && (!selectedIndicesMatched || !selectedTextsMatched)) {
                reason = "source_choice_mismatch";
            } else if (!requireSourceChoiceMatch) {
                reason = "alternate_choice_matched";
            } else {
                reason = "reentry_matched";
            }

            if (stopAtReentry || !reentryMatched) {
                return Choice.chooseAndTerminate(sanitized, reason);
            }
            return Choice.choose(sanitized);
        }

        private <T> Choice postBranchAutopilotChoice(DecisionContext<T> context) {
            if (!postBranchAutopilot || context == null || context.candidateCount <= 0) {
                return Choice.none();
            }
            List<Integer> indices = continuationPolicy == ContinuationPolicy.SAMPLE
                    ? sampledAutopilotIndices(context, rolloutRandom)
                    : deterministicAutopilotIndices(context);
            if (indices.isEmpty()) {
                return Choice.none();
            }
            return Choice.choose(indices);
        }
    }

    private static <T> List<Integer> deterministicAutopilotIndices(EngineDecisionBranchController.DecisionContext<T> context) {
        int candidateCount = context == null ? 0 : context.candidateCount;
        if (candidateCount <= 0) {
            return Collections.emptyList();
        }
        int pickLimit = context.maxTargets <= 0
                ? candidateCount
                : Math.min(context.maxTargets, candidateCount);
        int minTargets = Math.max(0, context.minTargets);
        StateSequenceBuilder.ActionType actionType = context.actionType;
        if (actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
            int spell = stableBestIndex(context, true, false, false, false);
            if (spell >= 0) {
                return Collections.singletonList(spell);
            }
            int pass = stableTerminalIndex(context);
            if (pass >= 0) {
                return Collections.singletonList(pass);
            }
            return Collections.singletonList(0);
        }
        if (actionType == StateSequenceBuilder.ActionType.DECLARE_ATTACKS
                || actionType == StateSequenceBuilder.ActionType.DECLARE_BLOCKS) {
            List<Integer> combat = stableNonTerminalIndices(context, pickLimit);
            if (!combat.isEmpty()) {
                return combat;
            }
            int done = stableTerminalIndex(context);
            return done >= 0 ? Collections.singletonList(done) : Collections.emptyList();
        }
        if (pickLimit == 1) {
            int nonStop = stableBestIndex(context, false, true, true, false);
            return Collections.singletonList(nonStop >= 0 ? nonStop : 0);
        }
        List<Integer> indices = stableNonTerminalIndices(context, pickLimit);
        if (indices.size() < minTargets) {
            for (int i = 0; i < candidateCount && indices.size() < Math.max(minTargets, 1); i++) {
                if (!indices.contains(i)) {
                    indices.add(i);
                }
            }
        }
        return indices;
    }

    private static <T> List<Integer> sampledAutopilotIndices(
            EngineDecisionBranchController.DecisionContext<T> context,
            Random random
    ) {
        int candidateCount = context == null ? 0 : context.candidateCount;
        if (candidateCount <= 0) {
            return Collections.emptyList();
        }
        Random rng = random == null ? new Random(0L) : random;
        int pickLimit = context.maxTargets <= 0
                ? candidateCount
                : Math.min(context.maxTargets, candidateCount);
        int minTargets = Math.max(0, context.minTargets);
        List<Integer> pool = stableNonTerminalIndices(context, candidateCount);
        if (pool.isEmpty()) {
            for (int i = 0; i < candidateCount; i++) {
                pool.add(i);
            }
        }
        Collections.shuffle(pool, rng);
        int required = Math.max(minTargets, 1);
        int maxPick = Math.max(required, pickLimit);
        int count = Math.min(pool.size(), maxPick <= required ? required : required + rng.nextInt(maxPick - required + 1));
        List<Integer> out = new ArrayList<>(pool.subList(0, count));
        if (out.size() < minTargets) {
            for (int i = 0; i < candidateCount && out.size() < minTargets; i++) {
                if (!out.contains(i)) {
                    out.add(i);
                }
            }
        }
        if (pickLimit >= 0 && out.size() > pickLimit) {
            out = new ArrayList<>(out.subList(0, pickLimit));
        }
        return out;
    }

    private static int stableBestIndex(
            EngineDecisionBranchController.DecisionContext<?> context,
            boolean requireNonPass,
            boolean allowMana,
            boolean allowLandPlay,
            boolean allowTerminal
    ) {
        int candidateCount = context == null ? 0 : context.candidateCount;
        int best = -1;
        String bestKey = "";
        for (int i = 0; i < candidateCount; i++) {
            String text = candidateText(context, i);
            if (requireNonPass && isPassLike(text)) {
                continue;
            }
            if (!allowTerminal && (isPassLike(text) || isDoneLike(text) || isStopLike(text))) {
                continue;
            }
            if (!allowMana && isManaLike(text)) {
                continue;
            }
            if (!allowLandPlay && isLandPlayLike(text)) {
                continue;
            }
            String key = candidateSortKey(context, i);
            if (best < 0 || key.compareTo(bestKey) < 0) {
                best = i;
                bestKey = key;
            }
        }
        return best;
    }

    private static List<Integer> stableNonTerminalIndices(
            EngineDecisionBranchController.DecisionContext<?> context,
            int maxTargets
    ) {
        List<Integer> indices = new ArrayList<>();
        int candidateCount = context == null ? 0 : context.candidateCount;
        for (int i = 0; i < candidateCount; i++) {
            String text = candidateText(context, i);
            if (isPassLike(text) || isDoneLike(text) || isStopLike(text)) {
                continue;
            }
            indices.add(i);
        }
        indices.sort((a, b) -> candidateSortKey(context, a).compareTo(candidateSortKey(context, b)));
        if (maxTargets >= 0 && indices.size() > maxTargets) {
            return new ArrayList<>(indices.subList(0, maxTargets));
        }
        return indices;
    }

    private static int stableTerminalIndex(EngineDecisionBranchController.DecisionContext<?> context) {
        int candidateCount = context == null ? 0 : context.candidateCount;
        int best = -1;
        String bestKey = "";
        for (int i = 0; i < candidateCount; i++) {
            String text = candidateText(context, i);
            if (isPassLike(text) || isDoneLike(text) || isStopLike(text)) {
                String key = candidateSortKey(context, i);
                if (best < 0 || key.compareTo(bestKey) < 0) {
                    best = i;
                    bestKey = key;
                }
            }
        }
        return best;
    }

    private static String candidateText(EngineDecisionBranchController.DecisionContext<?> context, int index) {
        if (context == null || context.candidateTexts == null || index < 0 || index >= context.candidateTexts.size()) {
            return "";
        }
        String text = context.candidateTexts.get(index);
        return text == null ? "" : text;
    }

    private static String candidateObjectId(EngineDecisionBranchController.DecisionContext<?> context, int index) {
        if (context == null || context.candidateObjectIds == null || index < 0 || index >= context.candidateObjectIds.size()) {
            return "";
        }
        String id = context.candidateObjectIds.get(index);
        return id == null ? "" : id;
    }

    private static String candidateSortKey(EngineDecisionBranchController.DecisionContext<?> context, int index) {
        return candidateText(context, index).trim().toLowerCase(Locale.US)
                + "\u0001"
                + candidateObjectId(context, index).trim().toLowerCase(Locale.US)
                + "\u0001"
                + String.format(Locale.US, "%04d", index);
    }

    private enum ContinuationPolicy {
        STABLE,
        SAMPLE;

        private static ContinuationPolicy parse(String raw) {
            if (raw == null || raw.trim().isEmpty()) {
                return STABLE;
            }
            String value = raw.trim().toUpperCase(Locale.US);
            if ("SAMPLED".equals(value) || "RANDOM".equals(value)) {
                return SAMPLE;
            }
            return ContinuationPolicy.valueOf(value);
        }
    }

    private static final class ValueTreeResult {
        private final ValueTreeSummary summary;
        private final List<ValueActionStats> actions;

        private ValueTreeResult(ValueTreeSummary summary, List<ValueActionStats> actions) {
            this.summary = summary == null ? new ValueTreeSummary() : summary;
            this.actions = actions == null ? Collections.emptyList() : actions;
        }
    }

    private static final class ValueActionStats {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private String actionIndices = "";
        private String actionTexts = "";
        private boolean source = false;
        private int rollouts = 0;
        private int terminalCount = 0;
        private int winCount = 0;
        private int lossCount = 0;
        private int drawCount = 0;
        private int errorCount = 0;
        private int notTerminalCount = 0;
        private double deltaVsSource = 0.0;
        private double importanceScore = 0.0;
        private final List<String> outcomes = new ArrayList<>();

        private static ValueActionStats fromSnapshot(
                Path path,
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> actionIndices,
                boolean source
        ) {
            ValueActionStats stats = new ValueActionStats();
            stats.snapshotPath = path == null ? "" : path.toString();
            if (snapshot != null) {
                stats.ordinal = snapshot.ordinal;
                stats.decisionNumber = snapshot.decisionNumber;
                stats.actionType = snapshot.actionType;
                stats.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                stats.candidateHash = snapshot.candidateHash;
                stats.stateHash = snapshot.stateHash;
                stats.randomStateHash = snapshot.randomStateHash;
                stats.actionIndices = joinInts(actionIndices);
                stats.actionTexts = joinStrings(selectedTexts(snapshot.candidateTexts, actionIndices));
            }
            stats.source = source;
            return stats;
        }

        private void add(BranchOutcome outcome) {
            rollouts++;
            if (outcome == null) {
                errorCount++;
                outcomes.add("null_outcome");
                return;
            }
            if (!outcome.error.isEmpty()) {
                errorCount++;
            } else if (outcome.terminal && outcome.won) {
                terminalCount++;
                winCount++;
            } else if (outcome.terminal && outcome.lost) {
                terminalCount++;
                lossCount++;
            } else if (outcome.terminal) {
                terminalCount++;
                drawCount++;
            } else {
                notTerminalCount++;
            }
            outcomes.add(outcome.shortClassification());
        }

        private double winRate() {
            return rollouts <= 0 ? 0.0 : ((double) winCount) / rollouts;
        }

        private double lossRate() {
            return rollouts <= 0 ? 0.0 : ((double) lossCount) / rollouts;
        }

        private double terminalRate() {
            return rollouts <= 0 ? 0.0 : ((double) terminalCount) / rollouts;
        }

        private double valueMean() {
            return rollouts <= 0 ? 0.0 : ((double) winCount + 0.5d * drawCount) / rollouts;
        }

        private String toCsvLine() {
            List<String> cells = new ArrayList<>();
            cells.add(csv(snapshotPath));
            cells.add(String.valueOf(ordinal));
            cells.add(String.valueOf(decisionNumber));
            cells.add(csv(actionType));
            cells.add(String.valueOf(candidateCount));
            cells.add(csv(candidateHash));
            cells.add(csv(stateHash));
            cells.add(csv(randomStateHash));
            cells.add(csv(actionIndices));
            cells.add(csv(actionTexts));
            cells.add(String.valueOf(source));
            cells.add(String.valueOf(rollouts));
            cells.add(String.valueOf(terminalCount));
            cells.add(String.valueOf(winCount));
            cells.add(String.valueOf(lossCount));
            cells.add(String.valueOf(drawCount));
            cells.add(String.valueOf(errorCount));
            cells.add(String.valueOf(notTerminalCount));
            cells.add(formatDouble(winRate()));
            cells.add(formatDouble(lossRate()));
            cells.add(formatDouble(terminalRate()));
            cells.add(formatDouble(valueMean()));
            cells.add(formatDouble(deltaVsSource));
            cells.add(formatDouble(importanceScore));
            cells.add(csv(joinStrings(outcomes)));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class ValueTreeSummary {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private String classification = "";
        private String sourceIndices = "";
        private String sourceTexts = "";
        private double sourceWinRate = 0.0;
        private double sourceLossRate = 0.0;
        private double sourceTerminalRate = 0.0;
        private String bestIndices = "";
        private String bestTexts = "";
        private double bestWinRate = 0.0;
        private double bestLossRate = 0.0;
        private double bestTerminalRate = 0.0;
        private double deltaWinRate = 0.0;
        private double importanceScore = 0.0;
        private int actionsEvaluated = 0;
        private int totalRollouts = 0;
        private int terminalRollouts = 0;
        private boolean reentryAMatched = false;
        private boolean reentryBMatched = false;
        private String reentryACandidateHash = "";
        private String reentryBCandidateHash = "";
        private String reentryAStateHash = "";
        private String reentryBStateHash = "";
        private String reentryAReason = "";
        private String reentryBReason = "";
        private String error = "";

        private static ValueTreeSummary fromSnapshot(Path path, LiveCheckpointRecorder.Snapshot snapshot) {
            ValueTreeSummary summary = new ValueTreeSummary();
            summary.snapshotPath = path == null ? "" : path.toString();
            if (snapshot != null) {
                summary.ordinal = snapshot.ordinal;
                summary.decisionNumber = snapshot.decisionNumber;
                summary.actionType = snapshot.actionType;
                summary.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                summary.candidateHash = snapshot.candidateHash;
                summary.stateHash = snapshot.stateHash;
                summary.randomStateHash = snapshot.randomStateHash;
                summary.sourceIndices = joinInts(snapshot.selectedIndices);
                summary.sourceTexts = joinStrings(snapshot.selectedTexts);
            }
            return summary;
        }

        private static ValueTreeSummary loadError(Path path, String error) {
            ValueTreeSummary summary = new ValueTreeSummary();
            summary.snapshotPath = path == null ? "" : path.toString();
            summary.classification = "snapshot_load_error";
            summary.error = error == null ? "" : error;
            return summary;
        }

        private void applyReentry(BranchOutcome a, BranchOutcome b) {
            reentryAMatched = a != null && a.reentryMatched;
            reentryBMatched = b != null && b.reentryMatched;
            reentryACandidateHash = a == null ? "" : a.candidateHash;
            reentryBCandidateHash = b == null ? "" : b.candidateHash;
            reentryAStateHash = a == null ? "" : a.stateHash;
            reentryBStateHash = b == null ? "" : b.stateHash;
            reentryAReason = a == null ? "" : (a.reason.isEmpty() ? a.error : a.reason);
            reentryBReason = b == null ? "" : (b.reason.isEmpty() ? b.error : b.reason);
        }

        private String toCsvLine() {
            List<String> cells = new ArrayList<>();
            cells.add(csv(snapshotPath));
            cells.add(String.valueOf(ordinal));
            cells.add(String.valueOf(decisionNumber));
            cells.add(csv(actionType));
            cells.add(String.valueOf(candidateCount));
            cells.add(csv(candidateHash));
            cells.add(csv(stateHash));
            cells.add(csv(randomStateHash));
            cells.add(csv(classification));
            cells.add(csv(sourceIndices));
            cells.add(csv(sourceTexts));
            cells.add(formatDouble(sourceWinRate));
            cells.add(formatDouble(sourceLossRate));
            cells.add(formatDouble(sourceTerminalRate));
            cells.add(csv(bestIndices));
            cells.add(csv(bestTexts));
            cells.add(formatDouble(bestWinRate));
            cells.add(formatDouble(bestLossRate));
            cells.add(formatDouble(bestTerminalRate));
            cells.add(formatDouble(deltaWinRate));
            cells.add(formatDouble(importanceScore));
            cells.add(String.valueOf(actionsEvaluated));
            cells.add(String.valueOf(totalRollouts));
            cells.add(String.valueOf(terminalRollouts));
            cells.add(String.valueOf(reentryAMatched));
            cells.add(String.valueOf(reentryBMatched));
            cells.add(csv(reentryACandidateHash));
            cells.add(csv(reentryBCandidateHash));
            cells.add(csv(reentryAStateHash));
            cells.add(csv(reentryBStateHash));
            cells.add(csv(reentryAReason));
            cells.add(csv(reentryBReason));
            cells.add(csv(error));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class BranchOutcome {
        private final String label;
        private boolean firstDecisionSeen;
        private boolean reentryMatched;
        private boolean terminal;
        private boolean won;
        private boolean lost;
        private String error = "";
        private String terminationReason = "";
        private String actionType = "";
        private String candidateHash = "";
        private String stateHash = "";
        private String reason = "";

        private BranchOutcome(String label) {
            this.label = label == null ? "" : label;
        }

        private void captureController(SnapshotBranchController controller) {
            if (controller == null) {
                return;
            }
            firstDecisionSeen = controller.seen;
            reentryMatched = controller.reentryMatched;
            actionType = controller.actualActionType;
            candidateHash = controller.actualCandidateHash;
            stateHash = controller.actualStateHash;
            reason = controller.reason;
            if (!firstDecisionSeen && error.isEmpty()) {
                error = "checkpoint_no_reentry_decision";
            }
        }

        private void captureTerminal(Game game, String perspectiveName) {
            try {
                terminal = game != null && game.hasEnded();
                String winner = game == null ? "" : game.getWinner();
                String name = perspectiveName == null ? "" : perspectiveName;
                won = terminal && winner != null && !winner.isEmpty() && !name.isEmpty() && winner.contains(name);
                lost = terminal && winner != null && !winner.isEmpty() && !won;
            } catch (Throwable t) {
                error = error.isEmpty() ? errorSummary(t) : error;
            }
        }

        private String shortClassification() {
            if (!error.isEmpty()) {
                return "error=" + error;
            }
            if (terminal && won) {
                return "terminal_win";
            }
            if (terminal && lost) {
                return "terminal_loss";
            }
            if (terminal) {
                return "terminal_draw";
            }
            if (!reason.isEmpty()) {
                return "not_terminal=" + reason;
            }
            return "not_terminal";
        }
    }

    private static final class BranchRow {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String selectedIndices = "";
        private String selectedTexts = "";
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private boolean sourceReentryAMatched = false;
        private boolean sourceReentryBMatched = false;
        private String classification = "";
        private boolean sourceTerminal = false;
        private boolean sourceWon = false;
        private boolean sourceLost = false;
        private String sourceError = "";
        private String alternateIndices = "";
        private String alternateTexts = "";
        private boolean alternateTerminal = false;
        private boolean alternateWon = false;
        private boolean alternateLost = false;
        private String alternateError = "";
        private int alternateAttemptCount = 0;
        private int alternateTerminalCount = 0;
        private int alternateWinCount = 0;
        private String alternateOutcomes = "";
        private int positiveConfirmationCount = 0;
        private int positiveConfirmationPassCount = 0;
        private String positiveConfirmationOutcomes = "";
        private String reentryACandidateHash = "";
        private String reentryBCandidateHash = "";
        private String reentryAStateHash = "";
        private String reentryBStateHash = "";
        private String reentryAReason = "";
        private String reentryBReason = "";

        private static BranchRow fromSnapshot(Path path, LiveCheckpointRecorder.Snapshot snapshot) {
            BranchRow row = new BranchRow();
            row.snapshotPath = path == null ? "" : path.toString();
            if (snapshot != null) {
                row.ordinal = snapshot.ordinal;
                row.decisionNumber = snapshot.decisionNumber;
                row.actionType = snapshot.actionType;
                row.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                row.selectedIndices = joinInts(snapshot.selectedIndices);
                row.selectedTexts = joinStrings(snapshot.selectedTexts);
                row.candidateHash = snapshot.candidateHash;
                row.stateHash = snapshot.stateHash;
                row.randomStateHash = snapshot.randomStateHash;
            }
            return row;
        }

        private static BranchRow loadError(Path path, String error) {
            BranchRow row = new BranchRow();
            row.snapshotPath = path == null ? "" : path.toString();
            row.classification = "snapshot_load_error";
            row.sourceError = error;
            return row;
        }

        private void applyReentry(BranchOutcome a, BranchOutcome b) {
            sourceReentryAMatched = a != null && a.reentryMatched;
            sourceReentryBMatched = b != null && b.reentryMatched;
            reentryACandidateHash = a == null ? "" : a.candidateHash;
            reentryBCandidateHash = b == null ? "" : b.candidateHash;
            reentryAStateHash = a == null ? "" : a.stateHash;
            reentryBStateHash = b == null ? "" : b.stateHash;
            reentryAReason = a == null ? "" : (a.reason.isEmpty() ? a.error : a.reason);
            reentryBReason = b == null ? "" : (b.reason.isEmpty() ? b.error : b.reason);
        }

        private void applySource(BranchOutcome source) {
            sourceTerminal = source != null && source.terminal;
            sourceWon = source != null && source.won;
            sourceLost = source != null && source.lost;
            sourceError = source == null ? "" : source.error;
        }

        private void applyAlternate(BranchOutcome alternate) {
            alternateTerminal = alternate != null && alternate.terminal;
            alternateWon = alternate != null && alternate.won;
            alternateLost = alternate != null && alternate.lost;
            alternateError = alternate == null ? "" : alternate.error;
        }

        private String toCsvLine() {
            List<String> cells = new ArrayList<>();
            cells.add(csv(snapshotPath));
            cells.add(String.valueOf(ordinal));
            cells.add(String.valueOf(decisionNumber));
            cells.add(csv(actionType));
            cells.add(String.valueOf(candidateCount));
            cells.add(csv(selectedIndices));
            cells.add(csv(selectedTexts));
            cells.add(csv(candidateHash));
            cells.add(csv(stateHash));
            cells.add(csv(randomStateHash));
            cells.add(String.valueOf(sourceReentryAMatched));
            cells.add(String.valueOf(sourceReentryBMatched));
            cells.add(csv(classification));
            cells.add(String.valueOf(sourceTerminal));
            cells.add(String.valueOf(sourceWon));
            cells.add(String.valueOf(sourceLost));
            cells.add(csv(sourceError));
            cells.add(csv(alternateIndices));
            cells.add(csv(alternateTexts));
            cells.add(String.valueOf(alternateTerminal));
            cells.add(String.valueOf(alternateWon));
            cells.add(String.valueOf(alternateLost));
            cells.add(csv(alternateError));
            cells.add(String.valueOf(alternateAttemptCount));
            cells.add(String.valueOf(alternateTerminalCount));
            cells.add(String.valueOf(alternateWinCount));
            cells.add(csv(alternateOutcomes));
            cells.add(String.valueOf(positiveConfirmationCount));
            cells.add(String.valueOf(positiveConfirmationPassCount));
            cells.add(csv(positiveConfirmationOutcomes));
            cells.add(csv(reentryACandidateHash));
            cells.add(csv(reentryBCandidateHash));
            cells.add(csv(reentryAStateHash));
            cells.add(csv(reentryBStateHash));
            cells.add(csv(reentryAReason));
            cells.add(csv(reentryBReason));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class PositiveConfirmation {
        private final int requested;
        private final int passed;
        private final String outcomes;

        private PositiveConfirmation(int requested, int passed, String outcomes) {
            this.requested = requested;
            this.passed = passed;
            this.outcomes = outcomes == null ? "" : outcomes;
        }

        private boolean isConfirmed() {
            return requested <= 0 || requested == passed;
        }
    }

    private static final class Selection {
        private final int discoveredPathCount;
        private final int eligibleCount;
        private final List<SnapshotCandidate> selected;

        private Selection(int discoveredPathCount, int eligibleCount, List<SnapshotCandidate> selected) {
            this.discoveredPathCount = discoveredPathCount;
            this.eligibleCount = eligibleCount;
            this.selected = selected == null ? Collections.emptyList() : selected;
        }
    }

    private static final class SnapshotCandidate {
        private final Path path;
        private LiveCheckpointRecorder.Snapshot snapshot;
        private String loadError = "";
        private int rank = 0;
        private int score = 0;
        private String scoreReasons = "";
        private String gameKey = "";
        private int nonPassCandidateCount = 0;
        private int nonPassAlternateCount = 0;
        private int spellCandidateCount = 0;
        private int manaCandidateCount = 0;
        private int passCandidateCount = 0;
        private int turn = -1;
        private int ownLife = -1;
        private int ownGraveyardCount = -1;
        private int opponentPermanentCount = 0;

        private SnapshotCandidate(Path path) {
            this.path = path;
            this.gameKey = gameKey(path);
        }
    }

    private static final class Counts {
        private final Map<String, Integer> values = new LinkedHashMap<>();

        private void add(String classification) {
            String key = classification == null || classification.isEmpty() ? "unknown" : classification;
            values.put(key, values.getOrDefault(key, 0) + 1);
        }
    }

    private static final class Config {
        private Path checkpointRoot;
        private Path snapshotPath;
        private Path outDir = defaultOutDir();
        private int maxSnapshots = 0;
        private int timeoutSec = 30;
        private int alternateTimeoutSec = 30;
        private int maxAlternates = 1;
        private String selectionMode = "path";
        private int rankedMaxPerGame = 0;
        private boolean postBranchAutopilot = true;
        private int confirmPositiveRepeats = 1;
        private boolean requireIsolatedPositiveReprobe = true;
        private boolean reentryOnly = false;
        private Set<String> actionTypes = Collections.emptySet();
        private boolean valueTree = false;
        private int treeRollouts = 1;
        private int treeMaxActions = 0;
        private boolean treeIncludePass = true;
        private int treeTimeoutSec = 30;
        private long treeSeed = 1337L;
        private ContinuationPolicy treeContinuationPolicy = ContinuationPolicy.STABLE;

        private static Config parse(String[] args) {
            Config cfg = new Config();
            Map<String, String> values = parseArgs(args);
            if (values.containsKey("checkpoint-root")) {
                cfg.checkpointRoot = Paths.get(values.get("checkpoint-root"));
            }
            if (values.containsKey("snapshot")) {
                cfg.snapshotPath = Paths.get(values.get("snapshot"));
            }
            if (values.containsKey("out")) {
                cfg.outDir = Paths.get(values.get("out"));
            }
            if (values.containsKey("max-snapshots")) {
                cfg.maxSnapshots = Integer.parseInt(values.get("max-snapshots"));
            }
            if (values.containsKey("timeout-sec")) {
                cfg.timeoutSec = Integer.parseInt(values.get("timeout-sec"));
            }
            if (values.containsKey("alternate-timeout-sec")) {
                cfg.alternateTimeoutSec = Integer.parseInt(values.get("alternate-timeout-sec"));
            } else {
                cfg.alternateTimeoutSec = cfg.timeoutSec;
            }
            if (values.containsKey("max-alternates")) {
                cfg.maxAlternates = Integer.parseInt(values.get("max-alternates"));
            }
            if (values.containsKey("selection-mode")) {
                cfg.selectionMode = values.get("selection-mode").trim().toLowerCase(Locale.US);
            }
            if (values.containsKey("ranked")) {
                cfg.selectionMode = Boolean.parseBoolean(values.get("ranked")) ? "ranked" : "path";
            }
            if (values.containsKey("ranked-max-per-game")) {
                cfg.rankedMaxPerGame = Integer.parseInt(values.get("ranked-max-per-game"));
            }
            if (values.containsKey("post-branch-autopilot")) {
                cfg.postBranchAutopilot = Boolean.parseBoolean(values.get("post-branch-autopilot"));
            }
            if (values.containsKey("confirm-positive-repeats")) {
                cfg.confirmPositiveRepeats = Integer.parseInt(values.get("confirm-positive-repeats"));
            }
            if (values.containsKey("positive-confirmations")) {
                cfg.confirmPositiveRepeats = Integer.parseInt(values.get("positive-confirmations"));
            }
            if (values.containsKey("require-isolated-positive-reprobe")) {
                cfg.requireIsolatedPositiveReprobe = Boolean.parseBoolean(values.get("require-isolated-positive-reprobe"));
            }
            if (values.containsKey("reentry-only")) {
                cfg.reentryOnly = Boolean.parseBoolean(values.get("reentry-only"));
            }
            if (values.containsKey("action-types")) {
                cfg.actionTypes = parseSet(values.get("action-types"));
            }
            if (values.containsKey("value-tree")) {
                cfg.valueTree = Boolean.parseBoolean(values.get("value-tree"));
            }
            if (values.containsKey("tree-rollouts")) {
                cfg.treeRollouts = Math.max(1, Integer.parseInt(values.get("tree-rollouts")));
            }
            if (values.containsKey("tree-max-actions")) {
                cfg.treeMaxActions = Math.max(0, Integer.parseInt(values.get("tree-max-actions")));
            }
            if (values.containsKey("tree-include-pass")) {
                cfg.treeIncludePass = Boolean.parseBoolean(values.get("tree-include-pass"));
            }
            if (values.containsKey("tree-timeout-sec")) {
                cfg.treeTimeoutSec = Math.max(1, Integer.parseInt(values.get("tree-timeout-sec")));
            } else {
                cfg.treeTimeoutSec = cfg.alternateTimeoutSec;
            }
            if (values.containsKey("tree-seed")) {
                cfg.treeSeed = Long.parseLong(values.get("tree-seed"));
            }
            if (values.containsKey("tree-continuation-policy")) {
                cfg.treeContinuationPolicy = ContinuationPolicy.parse(values.get("tree-continuation-policy"));
            }
            return cfg;
        }

        private static Path defaultOutDir() {
            String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss", Locale.US));
            return Paths.get("local-training", "local_pbt", "live_checkpoint_branch_miner", ts);
        }

        private static Map<String, String> parseArgs(String[] args) {
            Map<String, String> out = new LinkedHashMap<>();
            if (args == null) {
                return out;
            }
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg == null || !arg.startsWith("--")) {
                    continue;
                }
                String key;
                String value;
                int eq = arg.indexOf('=');
                if (eq >= 0) {
                    key = arg.substring(2, eq);
                    value = arg.substring(eq + 1);
                } else {
                    key = arg.substring(2);
                    if (i + 1 < args.length && args[i + 1] != null && !args[i + 1].startsWith("--")) {
                        value = args[++i];
                    } else {
                        value = "true";
                    }
                }
                out.put(key, value);
            }
            return out;
        }

        private static Set<String> parseSet(String raw) {
            if (raw == null || raw.trim().isEmpty() || "*".equals(raw.trim())) {
                return Collections.emptySet();
            }
            Set<String> out = new HashSet<>();
            for (String item : raw.split(",")) {
                String value = item.trim();
                if (!value.isEmpty()) {
                    out.add(value);
                }
            }
            return out;
        }
    }
}
