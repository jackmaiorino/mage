package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import mage.player.ai.ComputerPlayerRL;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidClassException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
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
import java.util.Objects;
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
    private static final String SEQUENCE_TREE_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,candidate_hash,state_hash,rng_state_hash,"
                    + "pair_key,order_label,rollout,sequence_indices,sequence_texts,forced_steps_requested,"
                    + "forced_steps_completed,prefix_complete,prefix_reason,post_prefix_state_hash,"
                    + "post_prefix_candidate_hash,post_prefix_action_type,terminal,won,lost,error,outcome\n";
    private static final String SEQUENCE_TREE_SUMMARY_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,candidate_hash,state_hash,rng_state_hash,"
                    + "pair_key,first_indices,first_texts,second_indices,second_texts,classification,"
                    + "forward_rollouts,reverse_rollouts,forward_prefix_complete,reverse_prefix_complete,"
                    + "converged_post_prefix_count,compared_post_prefix_count,forward_terminal_count,"
                    + "reverse_terminal_count,forward_win_count,reverse_win_count,forward_loss_count,"
                    + "reverse_loss_count,forward_error_count,reverse_error_count,forward_post_prefix_hashes,"
                    + "reverse_post_prefix_hashes,forward_outcomes,reverse_outcomes\n";
    private static final String TERMINAL_LINE_SEARCH_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,action_type,candidate_count,candidate_hash,state_hash,rng_state_hash,"
                    + "attempt,continuation_sample,continuation_seed,root_indices,root_texts,terminal,won,lost,error,outcome,decision_count,"
                    + "forced_steps_requested,forced_steps_completed,prefix_complete,final_state_hash,line_trace\n";
    private static final String TERMINAL_LINE_TRAINING_DATA_SUMMARY_CSV_HEADER =
            "snapshot_path,ordinal,decision_number,attempt,continuation_sample,root_indices,root_texts,outcome,"
                    + "terminal_return,captured_records,written_records,skipped_records\n";
    private static final int MAX_DECISION_TRACE_ROWS = 256;
    private static final float BRANCH_RETURN_UNOBSERVED = -2.0f;

    private LiveCheckpointBranchMiner() {
    }

    public static void main(String[] args) throws Exception {
        Config cfg = Config.parse(args);
        Files.createDirectories(cfg.outDir);
        Selection selection = selectSnapshots(cfg);
        writeSelectionManifest(cfg, selection.selected);
        if (cfg.terminalLineSearch) {
            runTerminalLineSearchMode(cfg, selection);
            return;
        }
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
        writeReadme(cfg, csvPath, processed, selection.discoveredPathCount, selection.eligibleCount,
                selection.loadErrorCount, selection.loadErrorExamples, counts);
        System.out.println("live checkpoint branch miner wrote " + processed + " row(s) to " + csvPath);
        System.out.println("classification counts: " + counts.values);
    }

    private static void runValueTreeMode(Config cfg, Selection selection) throws Exception {
        Path actionCsv = cfg.outDir.resolve("counterfactual_value_tree.csv");
        Path summaryCsv = cfg.outDir.resolve("counterfactual_value_tree_summary.csv");
        Path sequenceCsv = cfg.outDir.resolve("counterfactual_sequence_tree.csv");
        Path sequenceSummaryCsv = cfg.outDir.resolve("counterfactual_sequence_tree_summary.csv");
        Files.write(actionCsv, VALUE_TREE_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        Files.write(summaryCsv, VALUE_TREE_SUMMARY_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        if (cfg.sequenceTree) {
            Files.write(sequenceCsv, SEQUENCE_TREE_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            Files.write(sequenceSummaryCsv, SEQUENCE_TREE_SUMMARY_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }

        int processed = 0;
        int actionRows = 0;
        int sequenceRows = 0;
        int sequenceSummaryRows = 0;
        Counts counts = new Counts();
        Counts sequenceCounts = new Counts();
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
                if (cfg.sequenceTree) {
                    sequenceRows += result.sequenceRows.size();
                    sequenceSummaryRows += result.sequenceSummaries.size();
                    appendSequenceRows(sequenceCsv, result.sequenceRows);
                    appendSequenceSummaries(sequenceSummaryCsv, result.sequenceSummaries);
                    for (SequencePairSummary sequenceSummary : result.sequenceSummaries) {
                        sequenceCounts.add(sequenceSummary.classification);
                    }
                }
            }
            appendValueSummary(summaryCsv, summary);
            counts.add(summary.classification);
            processed++;
        }
        writeValueTreeReadme(cfg, actionCsv, summaryCsv, processed,
                selection.discoveredPathCount, selection.eligibleCount,
                selection.loadErrorCount, selection.loadErrorExamples, actionRows,
                sequenceRows, sequenceSummaryRows, counts, sequenceCounts);
        System.out.println("counterfactual value tree wrote " + actionRows + " action row(s) to " + actionCsv);
        System.out.println("counterfactual value tree wrote " + processed + " summary row(s) to " + summaryCsv);
        System.out.println("classification counts: " + counts.values);
        if (cfg.sequenceTree) {
            System.out.println("counterfactual sequence tree wrote " + sequenceRows + " row(s) to " + sequenceCsv);
            System.out.println("counterfactual sequence tree wrote " + sequenceSummaryRows + " summary row(s) to " + sequenceSummaryCsv);
            System.out.println("sequence classification counts: " + sequenceCounts.values);
        }
    }

    private static void runTerminalLineSearchMode(Config cfg, Selection selection) throws Exception {
        Path csvPath = cfg.outDir.resolve("terminal_line_search.csv");
        Files.write(csvPath, TERMINAL_LINE_SEARCH_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        Path trainingDataPath = cfg.outDir.resolve("terminal_line_training_data.ser");
        Path trainingSummaryPath = cfg.outDir.resolve("terminal_line_training_data_summary.csv");
        List<StateSequenceBuilder.TrainingData> trainingRecords = cfg.lineCaptureTrainingData
                ? new ArrayList<>()
                : Collections.emptyList();
        if (cfg.lineCaptureTrainingData) {
            Files.write(trainingSummaryPath, TERMINAL_LINE_TRAINING_DATA_SUMMARY_CSV_HEADER.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }
        int processed = 0;
        int attempts = 0;
        int wins = 0;
        int trainingRows = 0;
        Counts counts = new Counts();
        for (SnapshotCandidate candidate : selection.selected) {
            if (cfg.maxSnapshots > 0 && processed >= cfg.maxSnapshots) {
                break;
            }
            if (candidate.loadError != null && !candidate.loadError.isEmpty()) {
                TerminalLineRow row = TerminalLineRow.loadError(candidate.path, candidate.loadError);
                appendTerminalLineRow(csvPath, row);
                counts.add(row.classification());
                processed++;
                continue;
            }
            LiveCheckpointRecorder.Snapshot snapshot = candidate.snapshot;
            if (snapshot == null || snapshot.gameSnapshot == null) {
                TerminalLineRow row = TerminalLineRow.failure(candidate.path, snapshot, "snapshot_missing_game");
                appendTerminalLineRow(csvPath, row);
                counts.add(row.classification());
                processed++;
                continue;
            }
            if (snapshot.candidateTexts == null || snapshot.candidateTexts.size() < 1) {
                TerminalLineRow row = TerminalLineRow.failure(candidate.path, snapshot, "snapshot_missing_candidates");
                appendTerminalLineRow(csvPath, row);
                counts.add(row.classification());
                processed++;
                continue;
            }
            List<Integer> sourceIndices = sanitizeIndices(snapshot.selectedIndices, snapshot.candidateTexts.size());
            if (sourceIndices.isEmpty()) {
                TerminalLineRow row = TerminalLineRow.failure(candidate.path, snapshot, "snapshot_missing_source_choice");
                appendTerminalLineRow(csvPath, row);
                counts.add(row.classification());
                processed++;
                continue;
            }

            List<List<Integer>> rootChoices = valueTreeChoices(snapshot, sourceIndices, cfg);
            int rootLimit = cfg.lineMaxRootActions <= 0
                    ? rootChoices.size()
                    : Math.min(cfg.lineMaxRootActions, rootChoices.size());
            if (rootLimit <= 0) {
                TerminalLineRow row = TerminalLineRow.failure(candidate.path, snapshot, "no_root_actions");
                appendTerminalLineRow(csvPath, row);
                counts.add(row.classification());
                processed++;
                continue;
            }

            boolean foundWin = false;
            int lineAttempts = Math.max(1, cfg.lineAttempts);
            for (int attempt = 0; attempt < lineAttempts; attempt++) {
                List<Integer> rootChoice = rootChoices.get(attempt % rootLimit);
                boolean isSource = rootChoice.equals(sourceIndices);
                int continuationSample = cfg.lineCommonContinuationSeeds ? attempt / rootLimit : attempt;
                long seed = terminalLineSeed(cfg, snapshot, rootChoice, attempt, continuationSample);
                BranchOutcome outcome = runProbe(
                        snapshot,
                        rootChoice,
                        false,
                        isSource,
                        "terminal_line_" + processed + "_" + attempt,
                        cfg.lineTimeoutSec,
                        cfg.postBranchAutopilot,
                        cfg.treeContinuationPolicy,
                        seed,
                        cfg.lineCaptureTrainingData,
                        cfg.lineTrainingMaxRecordsPerBranch);
                TerminalLineRow row = TerminalLineRow.fromSnapshot(
                        candidate.path, snapshot, attempt, continuationSample, seed, rootChoice);
                row.apply(outcome);
                appendTerminalLineRow(csvPath, row);
                if (cfg.lineCaptureTrainingData) {
                    trainingRows += appendTerminalLineTrainingData(
                            trainingDataPath,
                            trainingSummaryPath,
                            trainingRecords,
                            candidate.path,
                            snapshot,
                            attempt,
                            continuationSample,
                            rootChoice,
                            row,
                            outcome);
                }
                counts.add(row.classification());
                attempts++;
                if (outcome != null && outcome.terminal && outcome.won) {
                    wins++;
                    foundWin = true;
                    if (cfg.lineStopOnWin) {
                        break;
                    }
                }
            }
            processed++;
            if (foundWin && cfg.lineStopOnWinAll) {
                break;
            }
        }
        if (cfg.lineCaptureTrainingData) {
            writeTrainingData(trainingDataPath, trainingRecords);
        }
        writeTerminalLineSearchReadme(cfg, csvPath, processed,
                selection.discoveredPathCount, selection.eligibleCount,
                selection.loadErrorCount, selection.loadErrorExamples, attempts, wins, counts,
                cfg.lineCaptureTrainingData, trainingDataPath, trainingRows, trainingRecords.size());
        System.out.println("terminal line search wrote " + attempts + " attempt row(s) to " + csvPath);
        System.out.println("terminal line wins: " + wins);
        if (cfg.lineCaptureTrainingData) {
            System.out.println("terminal line training records: " + trainingRecords.size()
                    + " to " + trainingDataPath);
        }
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
        SequenceTreeResult sequences = cfg.sequenceTree
                ? probeSequenceTree(snapshotPath, snapshot, cfg, choices)
                : SequenceTreeResult.empty();
        return new ValueTreeResult(summary, actions, sequences.rows, sequences.summaries);
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
        return runProbe(snapshot, forcedIndices, stopAtReentry, requireSourceChoiceMatch,
                label, timeoutSec, postBranchAutopilot, continuationPolicy, rolloutSeed, false, 0);
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
            long rolloutSeed,
            boolean captureTrainingData,
            int maxTrainingRecords
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
                        rolloutSeed,
                        captureTrainingData,
                        maxTrainingRecords);
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
        try {
            return loadSnapshot(path, false);
        } catch (InvalidClassException e) {
            return loadSnapshot(path, true);
        }
    }

    private static LiveCheckpointRecorder.Snapshot loadSnapshot(Path path, boolean tolerateMageSerialVersionDrift) throws Exception {
        try (ObjectInputStream in = snapshotInputStream(path, tolerateMageSerialVersionDrift)) {
            Object obj = in.readObject();
            if (!(obj instanceof LiveCheckpointRecorder.Snapshot)) {
                throw new IllegalArgumentException("Unexpected snapshot object: "
                        + (obj == null ? "null" : obj.getClass().getName()));
            }
            return (LiveCheckpointRecorder.Snapshot) obj;
        }
    }

    private static ObjectInputStream snapshotInputStream(Path path, boolean tolerateMageSerialVersionDrift) throws IOException {
        InputStream gzip = new GZIPInputStream(Files.newInputStream(path));
        if (tolerateMageSerialVersionDrift) {
            return new MageSnapshotObjectInputStream(gzip);
        }
        return new ObjectInputStream(gzip);
    }

    private static final class MageSnapshotObjectInputStream extends ObjectInputStream {

        MageSnapshotObjectInputStream(InputStream in) throws IOException {
            super(in);
        }

        @Override
        protected ObjectStreamClass readClassDescriptor() throws IOException, ClassNotFoundException {
            ObjectStreamClass streamDescriptor = super.readClassDescriptor();
            String className = streamDescriptor == null ? "" : streamDescriptor.getName();
            if (!isMageSnapshotClass(className)) {
                return streamDescriptor;
            }
            Class<?> localClass;
            try {
                localClass = Class.forName(className, false, Thread.currentThread().getContextClassLoader());
            } catch (ClassNotFoundException e) {
                return streamDescriptor;
            }
            ObjectStreamClass localDescriptor = ObjectStreamClass.lookup(localClass);
            if (localDescriptor != null
                    && localDescriptor.getSerialVersionUID() != streamDescriptor.getSerialVersionUID()) {
                return localDescriptor;
            }
            return streamDescriptor;
        }

        private static boolean isMageSnapshotClass(String className) {
            return className != null && (className.startsWith("mage.") || className.startsWith("[Lmage."));
        }
    }

    private static Selection selectSnapshots(Config cfg) throws Exception {
        List<Path> paths = discoverSnapshotPaths(cfg);
        int discoveredPathCount = paths.size();
        paths = applyPathShard(paths, cfg);
        List<SnapshotCandidate> eligible = new ArrayList<>();
        int loadErrorCount = 0;
        List<String> loadErrorExamples = new ArrayList<>();
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
                loadErrorCount++;
                if (loadErrorExamples.size() < 5) {
                    loadErrorExamples.add((candidate.path == null ? "" : candidate.path.toString())
                            + " :: " + candidate.loadError);
                }
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
        int limit = selectionLimitForShard(cfg);
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
        return new Selection(discoveredPathCount, eligible.size(), loadErrorCount, loadErrorExamples, selected);
    }

    private static List<Path> applyPathShard(List<Path> paths, Config cfg) {
        if (paths == null || paths.isEmpty() || cfg.selectionShards <= 1) {
            return paths;
        }
        List<Path> sharded = new ArrayList<>();
        for (int i = 0; i < paths.size(); i++) {
            if (i % cfg.selectionShards == cfg.selectionShardIndex) {
                sharded.add(paths.get(i));
            }
        }
        return sharded;
    }

    private static int selectionLimitForShard(Config cfg) {
        if (cfg.maxSnapshots <= 0) {
            return Integer.MAX_VALUE;
        }
        if (cfg.selectionShards <= 1) {
            return cfg.maxSnapshots;
        }
        return Math.max(1, (cfg.maxSnapshots + cfg.selectionShards - 1) / cfg.selectionShards);
    }

    private static List<Path> discoverSnapshotPaths(Config cfg) throws Exception {
        List<Path> out = new ArrayList<>();
        if (cfg.snapshotPath != null) {
            out.add(cfg.snapshotPath);
            return out;
        }
        if (cfg.snapshotListPath != null) {
            for (String line : Files.readAllLines(cfg.snapshotListPath, StandardCharsets.UTF_8)) {
                String value = line == null ? "" : line.trim();
                if (!value.isEmpty() && value.charAt(0) == '\uFEFF') {
                    value = value.substring(1).trim();
                }
                if (value.isEmpty() || value.startsWith("#")) {
                    continue;
                }
                out.add(Paths.get(value));
            }
            return out;
        }
        if (cfg.checkpointRoot == null) {
            throw new IllegalArgumentException("--checkpoint-root, --snapshot, or --snapshot-list is required");
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

    private static void appendTerminalLineRow(Path csvPath, TerminalLineRow row) throws Exception {
        Files.write(csvPath, row.toCsvLine().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static int appendTerminalLineTrainingData(
            Path trainingDataPath,
            Path summaryPath,
            List<StateSequenceBuilder.TrainingData> out,
            Path snapshotPath,
            LiveCheckpointRecorder.Snapshot snapshot,
            int attempt,
            int continuationSample,
            List<Integer> rootChoice,
            TerminalLineRow row,
            BranchOutcome outcome
    ) throws Exception {
        int captured = outcome == null || outcome.trainingData == null ? 0 : outcome.trainingData.size();
        float terminalReturn = outcome != null && outcome.won ? 1.0f : (outcome != null && outcome.lost ? -1.0f : 0.0f);
        int written = 0;
        int skipped = 0;
        if (terminalReturn == 0.0f || captured <= 0) {
            skipped = captured;
        } else {
            for (StateSequenceBuilder.TrainingData td : outcome.trainingData) {
                if (td == null || td.chosenCount <= 0 || td.chosenIndices == null) {
                    skipped++;
                    continue;
                }
                float[] target = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
                for (int i = 0; i < target.length; i++) {
                    target[i] = BRANCH_RETURN_UNOBSERVED;
                }
                int limit = Math.min(td.candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
                int marked = 0;
                for (int i = 0; i < Math.min(td.chosenCount, td.chosenIndices.length); i++) {
                    int idx = td.chosenIndices[i];
                    if (idx >= 0
                            && idx < limit
                            && td.candidateMask != null
                            && idx < td.candidateMask.length
                            && td.candidateMask[idx] != 0) {
                        target[idx] = terminalReturn;
                        marked++;
                    }
                }
                if (marked <= 0) {
                    skipped++;
                    continue;
                }
                td.setMctsVisitTargets(target);
                out.add(td);
                written++;
            }
        }
        StringBuilder sb = new StringBuilder(512);
        sb.append(csv(snapshotPath == null ? "" : snapshotPath.toString()))
                .append(",").append(snapshot == null ? -1 : snapshot.ordinal)
                .append(",").append(snapshot == null ? -1 : snapshot.decisionNumber)
                .append(",").append(attempt)
                .append(",").append(continuationSample)
                .append(",").append(csv(joinInts(rootChoice)))
                .append(",").append(csv(row == null ? "" : row.rootTexts))
                .append(",").append(csv(row == null ? "" : row.outcome))
                .append(",").append(String.format(Locale.US, "%.1f", terminalReturn))
                .append(",").append(captured)
                .append(",").append(written)
                .append(",").append(skipped)
                .append("\n");
        Files.write(summaryPath, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        return written;
    }

    private static void writeTrainingData(Path path, List<StateSequenceBuilder.TrainingData> records) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(new ArrayList<>(records));
        }
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
            int loadErrors,
            List<String> loadErrorExamples,
            Counts counts
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Live Checkpoint Branch Miner\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- eligible: ").append(eligible).append("\n");
        sb.append("- load_errors: ").append(loadErrors).append("\n");
        appendLoadErrorExamples(sb, loadErrorExamples);
        sb.append("- selection_mode: ").append(cfg.selectionMode).append("\n");
        sb.append("- ranked_max_per_game: ").append(cfg.rankedMaxPerGame).append("\n");
        sb.append("- selection_shards: ").append(cfg.selectionShards).append("\n");
        sb.append("- selection_shard_index: ").append(cfg.selectionShardIndex).append("\n");
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

    private static void appendLoadErrorExamples(StringBuilder sb, List<String> examples) {
        if (examples == null || examples.isEmpty()) {
            return;
        }
        int index = 1;
        for (String example : examples) {
            sb.append("- load_error_example_").append(index).append(": ").append(example).append("\n");
            index++;
        }
    }

    private static void writeTerminalLineSearchReadme(
            Config cfg,
            Path csvPath,
            int processed,
            int discovered,
            int eligible,
            int loadErrors,
            List<String> loadErrorExamples,
            int attempts,
            int wins,
            Counts counts,
            boolean captureTrainingData,
            Path trainingDataPath,
            int trainingAttemptRows,
            int trainingRecords
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Terminal Line Search Proof\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- eligible: ").append(eligible).append("\n");
        sb.append("- load_errors: ").append(loadErrors).append("\n");
        appendLoadErrorExamples(sb, loadErrorExamples);
        sb.append("- attempts: ").append(attempts).append("\n");
        sb.append("- wins: ").append(wins).append("\n");
        sb.append("- selection_mode: ").append(cfg.selectionMode).append("\n");
        sb.append("- ranked_max_per_game: ").append(cfg.rankedMaxPerGame).append("\n");
        sb.append("- line_attempts: ").append(cfg.lineAttempts).append("\n");
        sb.append("- line_max_root_actions: ").append(cfg.lineMaxRootActions).append("\n");
        sb.append("- line_timeout_sec: ").append(cfg.lineTimeoutSec).append("\n");
        sb.append("- line_stop_on_win: ").append(cfg.lineStopOnWin).append("\n");
        sb.append("- line_stop_on_win_all: ").append(cfg.lineStopOnWinAll).append("\n");
        sb.append("- line_common_continuation_seeds: ").append(cfg.lineCommonContinuationSeeds).append("\n");
        sb.append("- post_branch_autopilot: ").append(cfg.postBranchAutopilot).append("\n");
        sb.append("- tree_max_actions: ").append(cfg.treeMaxActions).append("\n");
        sb.append("- tree_include_pass: ").append(cfg.treeIncludePass).append("\n");
        sb.append("- tree_continuation_policy: ").append(cfg.treeContinuationPolicy.name().toLowerCase(Locale.US)).append("\n");
        sb.append("- tree_seed: ").append(cfg.treeSeed).append("\n");
        sb.append("- selected_snapshots_csv: ").append(cfg.outDir.resolve("selected_snapshots.csv")).append("\n");
        sb.append("- terminal_line_search_csv: ").append(csvPath).append("\n");
        sb.append("- line_capture_training_data: ").append(captureTrainingData).append("\n");
        if (captureTrainingData) {
            sb.append("- terminal_line_training_data_ser: ").append(trainingDataPath).append("\n");
            sb.append("- terminal_line_training_attempt_rows: ").append(trainingAttemptRows).append("\n");
            sb.append("- terminal_line_training_records: ").append(trainingRecords).append("\n");
            sb.append("- line_training_max_records_per_branch: ").append(cfg.lineTrainingMaxRecordsPerBranch).append("\n");
        }
        sb.append("- classification_counts: ").append(counts.values).append("\n\n");
        sb.append("This mode is a bounded proof search over serialized checkpoints. It forces a root action, ")
                .append("uses either branch-controller autopilot or the normal model path for subsequent continuations, ")
                .append("records the resulting decision trace, ")
                .append("and stops when a terminal win is found if configured. It uses terminal win/loss only; it does not add combo-specific rewards.\n");
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

    private static void appendSequenceRows(Path csvPath, List<SequenceTreeRow> rows) throws Exception {
        if (rows == null || rows.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(rows.size() * 256);
        for (SequenceTreeRow row : rows) {
            sb.append(row.toCsvLine());
        }
        Files.write(csvPath, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void appendSequenceSummaries(Path csvPath, List<SequencePairSummary> rows) throws Exception {
        if (rows == null || rows.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(rows.size() * 256);
        for (SequencePairSummary row : rows) {
            sb.append(row.toCsvLine());
        }
        Files.write(csvPath, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void writeValueTreeReadme(
            Config cfg,
            Path actionCsv,
            Path summaryCsv,
            int processed,
            int discovered,
            int eligible,
            int loadErrors,
            List<String> loadErrorExamples,
            int actionRows,
            int sequenceRows,
            int sequenceSummaryRows,
            Counts counts,
            Counts sequenceCounts
    ) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("# Counterfactual Value Tree Miner\n\n");
        sb.append("- processed: ").append(processed).append("\n");
        sb.append("- discovered: ").append(discovered).append("\n");
        sb.append("- eligible: ").append(eligible).append("\n");
        sb.append("- load_errors: ").append(loadErrors).append("\n");
        appendLoadErrorExamples(sb, loadErrorExamples);
        sb.append("- action_rows: ").append(actionRows).append("\n");
        sb.append("- sequence_tree: ").append(cfg.sequenceTree).append("\n");
        if (cfg.sequenceTree) {
            sb.append("- sequence_rows: ").append(sequenceRows).append("\n");
            sb.append("- sequence_summary_rows: ").append(sequenceSummaryRows).append("\n");
            sb.append("- tree_sequence_depth: ").append(cfg.treeSequenceDepth).append("\n");
            sb.append("- tree_sequence_beam: ").append(cfg.treeSequenceBeam).append("\n");
            sb.append("- tree_sequence_rollouts: ").append(cfg.treeSequenceRollouts).append("\n");
        }
        sb.append("- selection_mode: ").append(cfg.selectionMode).append("\n");
        sb.append("- ranked_max_per_game: ").append(cfg.rankedMaxPerGame).append("\n");
        sb.append("- selection_shards: ").append(cfg.selectionShards).append("\n");
        sb.append("- selection_shard_index: ").append(cfg.selectionShardIndex).append("\n");
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
        if (cfg.sequenceTree) {
            sb.append("- sequence_csv: ").append(cfg.outDir.resolve("counterfactual_sequence_tree.csv")).append("\n");
            sb.append("- sequence_summary_csv: ").append(cfg.outDir.resolve("counterfactual_sequence_tree_summary.csv")).append("\n");
            sb.append("- sequence_classification_counts: ").append(sequenceCounts.values).append("\n\n");
        }
        sb.append("This mode estimates action importance at each serialized checkpoint by forcing each selected root action, ")
                .append("running configurable continuations, and comparing action win rates against the accepted-policy source action. ")
                .append("It is a bounded sampled tree, not an exhaustive Magic game tree.\n");
        if (cfg.sequenceTree) {
            sb.append("\nSequence mode additionally forces short ordered prefixes such as A then B and B then A, ")
                    .append("records post-prefix state hashes when another decision is reached, and flags converged versus order-sensitive pairs.\n");
        }
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

    private static long terminalLineSeed(
            Config cfg,
            LiveCheckpointRecorder.Snapshot snapshot,
            List<Integer> rootChoice,
            int attempt,
            int continuationSample
    ) {
        long seed;
        if (cfg.lineCommonContinuationSeeds) {
            seed = cfg.treeSeed;
            seed = 31L * seed + (snapshot == null || snapshot.candidateHash == null ? 0 : snapshot.candidateHash.hashCode());
            seed = 31L * seed + (snapshot == null || snapshot.stateHash == null ? 0 : snapshot.stateHash.hashCode());
            seed = 31L * seed + continuationSample;
        } else {
            seed = treeRolloutSeed(cfg, snapshot, rootChoice, attempt);
        }
        seed = 31L * seed + 0x54_4c_49_4eL;
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
        if (source.terminalCount <= 0) {
            return "source_not_terminal";
        }
        if (source.lossCount <= 0) {
            return "source_terminal_not_loss";
        }
        if (best.terminalCount <= 0) {
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

    private static SequenceTreeResult probeSequenceTree(
            Path snapshotPath,
            LiveCheckpointRecorder.Snapshot snapshot,
            Config cfg,
            List<List<Integer>> rootChoices
    ) {
        if (snapshot == null || snapshot.candidateTexts == null || rootChoices == null || rootChoices.size() < 2) {
            return SequenceTreeResult.empty();
        }
        int beam = cfg.treeSequenceBeam <= 0
                ? rootChoices.size()
                : Math.min(cfg.treeSequenceBeam, rootChoices.size());
        int depth = Math.max(2, cfg.treeSequenceDepth);
        int rollouts = Math.max(1, cfg.treeSequenceRollouts);
        List<SequenceTreeRow> rows = new ArrayList<>();
        List<SequencePairSummary> summaries = new ArrayList<>();
        for (int i = 0; i < beam; i++) {
            for (int j = i + 1; j < beam; j++) {
                List<Integer> first = rootChoices.get(i);
                List<Integer> second = rootChoices.get(j);
                SequencePairSummary pair = SequencePairSummary.fromSnapshot(snapshotPath, snapshot, first, second);
                String pairKey = sequenceKey(first, second);
                for (int rollout = 0; rollout < rollouts; rollout++) {
                    SequenceTreeRow forward = runSequenceRow(
                            snapshotPath,
                            snapshot,
                            pairKey,
                            "forward",
                            rollout,
                            sequenceChoices(first, second, depth),
                            cfg);
                    SequenceTreeRow reverse = runSequenceRow(
                            snapshotPath,
                            snapshot,
                            pairKey,
                            "reverse",
                            rollout,
                            sequenceChoices(second, first, depth),
                            cfg);
                    rows.add(forward);
                    rows.add(reverse);
                    pair.add(forward, reverse);
                }
                pair.classify();
                summaries.add(pair);
            }
        }
        return new SequenceTreeResult(rows, summaries);
    }

    private static SequenceTreeRow runSequenceRow(
            Path snapshotPath,
            LiveCheckpointRecorder.Snapshot snapshot,
            String pairKey,
            String orderLabel,
            int rollout,
            List<List<Integer>> sequenceChoices,
            Config cfg
    ) {
        SequenceTreeRow row = SequenceTreeRow.fromSnapshot(snapshotPath, snapshot, pairKey, orderLabel, rollout, sequenceChoices);
        long seed = sequenceRolloutSeed(cfg, snapshot, sequenceChoices, rollout);
        BranchOutcome outcome = runSequenceProbe(
                snapshot,
                sequenceChoices,
                "sequence_tree_" + pairKey + "_" + orderLabel + "_r" + rollout,
                cfg.treeTimeoutSec,
                cfg.postBranchAutopilot,
                cfg.treeContinuationPolicy,
                seed);
        row.apply(outcome);
        return row;
    }

    private static BranchOutcome runSequenceProbe(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<List<Integer>> sequenceChoices,
            String label,
            int timeoutSec,
            boolean postBranchAutopilot,
            ContinuationPolicy continuationPolicy,
            long rolloutSeed
    ) {
        List<ForcedStep> steps = forcedSteps(snapshot, sequenceChoices);
        RandomUtil.State previousRandom = RandomUtil.captureState();
        Game game = null;
        SnapshotBranchController controller =
                new SnapshotBranchController(
                        snapshot,
                        steps,
                        true,
                        false,
                        false,
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

    private static List<List<Integer>> sequenceChoices(List<Integer> first, List<Integer> second, int depth) {
        List<List<Integer>> out = new ArrayList<>();
        out.add(first == null ? Collections.emptyList() : new ArrayList<>(first));
        if (depth >= 2) {
            out.add(second == null ? Collections.emptyList() : new ArrayList<>(second));
        }
        return out;
    }

    private static List<ForcedStep> forcedSteps(
            LiveCheckpointRecorder.Snapshot snapshot,
            List<List<Integer>> sequenceChoices
    ) {
        if (snapshot == null || sequenceChoices == null || sequenceChoices.isEmpty()) {
            return Collections.emptyList();
        }
        List<ForcedStep> steps = new ArrayList<>();
        for (int i = 0; i < sequenceChoices.size(); i++) {
            List<Integer> indices = sanitizeIndices(sequenceChoices.get(i),
                    snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size());
            List<String> texts = selectedTexts(snapshot.candidateTexts, indices);
            if (i == 0) {
                steps.add(ForcedStep.root(indices, texts, snapshot.actionType));
            } else {
                steps.add(ForcedStep.byText(texts, snapshot.actionType));
            }
        }
        return steps;
    }

    private static long sequenceRolloutSeed(
            Config cfg,
            LiveCheckpointRecorder.Snapshot snapshot,
            List<List<Integer>> sequenceChoices,
            int rollout
    ) {
        long seed = cfg.treeSeed;
        seed = 31L * seed + (snapshot == null || snapshot.candidateHash == null ? 0 : snapshot.candidateHash.hashCode());
        seed = 31L * seed + (snapshot == null || snapshot.stateHash == null ? 0 : snapshot.stateHash.hashCode());
        seed = 31L * seed + sequenceKey(sequenceChoices).hashCode();
        seed = 31L * seed + rollout;
        return seed;
    }

    private static String sequenceKey(List<List<Integer>> sequenceChoices) {
        if (sequenceChoices == null || sequenceChoices.isEmpty()) {
            return "";
        }
        return sequenceChoices.stream().map(LiveCheckpointBranchMiner::joinInts).collect(Collectors.joining(">"));
    }

    private static String sequenceKey(List<Integer> first, List<Integer> second) {
        return joinInts(first) + "_vs_" + joinInts(second);
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

    private static String candidateTextMismatchReason(List<String> expected, List<String> actual) {
        List<String> left = expected == null ? Collections.emptyList() : expected;
        List<String> right = actual == null ? Collections.emptyList() : actual;
        int limit = Math.min(left.size(), right.size());
        int firstMismatch = -1;
        for (int i = 0; i < limit; i++) {
            if (!Objects.equals(left.get(i), right.get(i))) {
                firstMismatch = i;
                break;
            }
        }
        if (firstMismatch < 0 && left.size() != right.size()) {
            firstMismatch = limit;
        }
        String expectedText = firstMismatch >= 0 && firstMismatch < left.size() ? left.get(firstMismatch) : "<missing>";
        String actualText = firstMismatch >= 0 && firstMismatch < right.size() ? right.get(firstMismatch) : "<missing>";
        return "candidate_text_mismatch"
                + ":expected_size=" + left.size()
                + ":actual_size=" + right.size()
                + ":first=" + firstMismatch
                + ":expected=" + compactReasonText(expectedText)
                + ":actual=" + compactReasonText(actualText);
    }

    private static String compactReasonText(String value) {
        if (value == null) {
            return "";
        }
        String compact = value.replace('\r', ' ').replace('\n', ' ').replace('|', '/');
        return compact.length() <= 80 ? compact : compact.substring(0, 80) + "...";
    }

    private static final class ForcedStep {
        private final List<Integer> rootIndices;
        private final List<String> expectedTexts;
        private final String actionType;
        private final boolean root;

        private ForcedStep(List<Integer> rootIndices, List<String> expectedTexts, String actionType, boolean root) {
            this.rootIndices = rootIndices == null ? Collections.emptyList() : new ArrayList<>(rootIndices);
            this.expectedTexts = expectedTexts == null ? Collections.emptyList() : new ArrayList<>(expectedTexts);
            this.actionType = actionType == null ? "" : actionType;
            this.root = root;
        }

        private static ForcedStep root(List<Integer> indices, List<String> texts, String actionType) {
            return new ForcedStep(indices, texts, actionType, true);
        }

        private static ForcedStep byText(List<String> texts, String actionType) {
            return new ForcedStep(Collections.emptyList(), texts, actionType, false);
        }

        private boolean actionTypeMatches(StateSequenceBuilder.ActionType actual) {
            String actualType = actual == null ? "" : actual.name();
            return actionType.isEmpty() || actionType.equals(actualType);
        }

        private <T> List<Integer> resolve(EngineDecisionBranchController.DecisionContext<T> context) {
            if (context == null || context.candidateCount <= 0) {
                return Collections.emptyList();
            }
            if (root) {
                return sanitizeIndices(rootIndices, context.candidateCount);
            }
            if (!actionTypeMatches(context.actionType) || expectedTexts.isEmpty()) {
                return Collections.emptyList();
            }
            List<Integer> out = new ArrayList<>();
            Set<Integer> used = new HashSet<>();
            for (String expected : expectedTexts) {
                int found = -1;
                for (int i = 0; i < context.candidateCount; i++) {
                    if (used.contains(i)) {
                        continue;
                    }
                    String actual = candidateText(context, i);
                    if (choiceTextMatches(expected, actual)) {
                        found = i;
                        break;
                    }
                }
                if (found < 0) {
                    return Collections.emptyList();
                }
                used.add(found);
                out.add(found);
            }
            return out;
        }
    }

    private static final class SnapshotBranchController implements EngineDecisionBranchController {
        private final LiveCheckpointRecorder.Snapshot snapshot;
        private final List<ForcedStep> forcedSteps;
        private final boolean stopAtReentry;
        private final boolean requireSourceChoiceMatch;
        private final boolean postBranchAutopilot;
        private final ContinuationPolicy continuationPolicy;
        private final Random rolloutRandom;
        private final boolean captureTrainingData;
        private final int maxTrainingRecords;

        private boolean seen;
        private boolean reentryMatched;
        private int decisionCount;
        private int forcedStepIndex;
        private int forcedStepsCompleted;
        private boolean prefixComplete;
        private boolean postPrefixCaptured;
        private String reason = "";
        private String prefixReason = "";
        private String actualActionType = "";
        private String actualCandidateHash = "";
        private String actualStateHash = "";
        private String postPrefixActionType = "";
        private String postPrefixCandidateHash = "";
        private String postPrefixStateHash = "";
        private List<String> actualCandidateTexts = Collections.emptyList();
        private List<String> selectedTexts = Collections.emptyList();
        private final List<String> decisionTrace = new ArrayList<>();
        private final List<StateSequenceBuilder.TrainingData> trainingData = new ArrayList<>();

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> forcedIndices,
                boolean stopAtReentry,
                boolean requireSourceChoiceMatch,
                boolean postBranchAutopilot,
                ContinuationPolicy continuationPolicy,
                long rolloutSeed
        ) {
            this(snapshot, forcedIndices, stopAtReentry, requireSourceChoiceMatch,
                    postBranchAutopilot, continuationPolicy, rolloutSeed, false, 0);
        }

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> forcedIndices,
                boolean stopAtReentry,
                boolean requireSourceChoiceMatch,
                boolean postBranchAutopilot,
                ContinuationPolicy continuationPolicy,
                long rolloutSeed,
                boolean captureTrainingData,
                int maxTrainingRecords
        ) {
            this(
                    snapshot,
                    Collections.singletonList(ForcedStep.root(
                            forcedIndices,
                            Collections.emptyList(),
                            snapshot == null ? "" : snapshot.actionType)),
                    false,
                    stopAtReentry,
                    requireSourceChoiceMatch,
                    postBranchAutopilot,
                    continuationPolicy,
                    rolloutSeed,
                    captureTrainingData,
                    maxTrainingRecords);
        }

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<ForcedStep> forcedSteps,
                boolean sequenceController,
                boolean stopAtReentry,
                boolean requireSourceChoiceMatch,
                boolean postBranchAutopilot,
                ContinuationPolicy continuationPolicy,
                long rolloutSeed
        ) {
            this(snapshot, forcedSteps, sequenceController, stopAtReentry, requireSourceChoiceMatch,
                    postBranchAutopilot, continuationPolicy, rolloutSeed, false, 0);
        }

        private SnapshotBranchController(
                LiveCheckpointRecorder.Snapshot snapshot,
                List<ForcedStep> forcedSteps,
                boolean sequenceController,
                boolean stopAtReentry,
                boolean requireSourceChoiceMatch,
                boolean postBranchAutopilot,
                ContinuationPolicy continuationPolicy,
                long rolloutSeed,
                boolean captureTrainingData,
                int maxTrainingRecords
        ) {
            this.snapshot = snapshot;
            this.forcedSteps = forcedSteps == null
                    ? Collections.emptyList()
                    : new ArrayList<>(forcedSteps);
            this.stopAtReentry = stopAtReentry;
            this.requireSourceChoiceMatch = requireSourceChoiceMatch;
            this.postBranchAutopilot = postBranchAutopilot;
            this.continuationPolicy = continuationPolicy == null ? ContinuationPolicy.STABLE : continuationPolicy;
            this.rolloutRandom = new Random(rolloutSeed);
            this.captureTrainingData = captureTrainingData;
            this.maxTrainingRecords = Math.max(0, maxTrainingRecords);
        }

        @Override
        public boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
            return true;
        }

        @Override
        public boolean shouldBypassModelInference() {
            return postBranchAutopilot;
        }

        @Override
        public boolean shouldCaptureTrainingData() {
            return captureTrainingData;
        }

        @Override
        public void onTrainingData(StateSequenceBuilder.TrainingData data) {
            if (!captureTrainingData || data == null || !prefixComplete) {
                return;
            }
            if (maxTrainingRecords > 0 && trainingData.size() >= maxTrainingRecords) {
                return;
            }
            trainingData.add(data);
        }

        @Override
        public <T> Choice onDecision(DecisionContext<T> context) {
            if (seen) {
                return onPostRootDecision(context);
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

            ForcedStep rootStep = forcedSteps.isEmpty()
                    ? ForcedStep.root(Collections.emptyList(), Collections.emptyList(), snapshot.actionType)
                    : forcedSteps.get(0);
            List<Integer> sanitized = rootStep.resolve(context);
            selectedTexts = LiveCheckpointBranchMiner.selectedTexts(context.candidateTexts, sanitized);
            boolean actionMatched = actualActionType.equals(snapshot.actionType);
            boolean candidatesMatched = context.candidateTexts.equals(snapshot.candidateTexts);
            boolean selectedIndicesMatched = sanitized.equals(sanitizeIndices(snapshot.selectedIndices, context.candidateCount));
            boolean selectedTextsMatched = selectedTexts.equals(snapshot.selectedTexts);
            boolean forcedTextsMatched = forcedTextsMatch(snapshot, rootStep, sanitized, selectedTexts);
            boolean anchoredCandidateMatch = snapshot.candidateTexts != null
                    && context.candidateTexts != null
                    && snapshot.candidateTexts.size() == context.candidateTexts.size()
                    && forcedTextsMatched;
            reentryMatched = actionMatched
                    && (candidatesMatched || anchoredCandidateMatch)
                    && (!requireSourceChoiceMatch || (selectedIndicesMatched && selectedTextsMatched))
                    && !sanitized.isEmpty();
            if (!actionMatched) {
                reason = "action_type_mismatch";
            } else if (sanitized.isEmpty()) {
                reason = "forced_indices_invalid";
            } else if (!forcedTextsMatched) {
                reason = "forced_text_mismatch";
            } else if (!candidatesMatched) {
                reason = "candidate_anchor_matched:" + candidateTextMismatchReason(snapshot.candidateTexts, context.candidateTexts);
            } else if (requireSourceChoiceMatch && (!selectedIndicesMatched || !selectedTextsMatched)) {
                reason = "source_choice_mismatch";
            } else if (!requireSourceChoiceMatch) {
                reason = "alternate_choice_matched";
            } else {
                reason = "reentry_matched";
            }

            if (stopAtReentry || !reentryMatched) {
                recordDecision("root_terminate", context, sanitized, reason);
                return Choice.chooseAndTerminate(sanitized, reason);
            }
            forcedStepIndex = 1;
            forcedStepsCompleted = 1;
            if (forcedStepIndex >= forcedSteps.size()) {
                prefixComplete = true;
                prefixReason = "prefix_complete";
            }
            recordDecision("root", context, sanitized, reason);
            return Choice.choose(sanitized);
        }

        private static boolean forcedTextsMatch(
                LiveCheckpointRecorder.Snapshot snapshot,
                ForcedStep rootStep,
                List<Integer> sanitized,
                List<String> actualSelectedTexts
        ) {
            if (snapshot == null || rootStep == null || sanitized == null || sanitized.isEmpty()) {
                return false;
            }
            List<String> expectedTexts = rootStep.root
                    ? LiveCheckpointBranchMiner.selectedTexts(snapshot.candidateTexts, sanitized)
                    : rootStep.expectedTexts;
            return choiceTextsMatch(expectedTexts, actualSelectedTexts);
        }

        private <T> Choice onPostRootDecision(DecisionContext<T> context) {
            if (forcedStepIndex >= forcedSteps.size()) {
                capturePostPrefix(context);
                return postBranchAutopilotChoice(context);
            }
            if (context == null
                    || context.player == null
                    || snapshot == null
                    || snapshot.playerId == null
                    || !snapshot.playerId.equals(context.player.getId())) {
                return postBranchAutopilotChoice(context);
            }
            ForcedStep step = forcedSteps.get(forcedStepIndex);
            if (!step.actionTypeMatches(context.actionType)) {
                return postBranchAutopilotChoice(context);
            }
            List<Integer> indices = step.resolve(context);
            if (indices.isEmpty()) {
                prefixReason = "prefix_step_" + forcedStepIndex + "_unavailable";
                reason = prefixReason;
                recordDecision("forced_unavailable", context, Collections.emptyList(), prefixReason);
                return Choice.chooseAndTerminate(Collections.emptyList(), prefixReason);
            }
            forcedStepIndex++;
            forcedStepsCompleted++;
            if (forcedStepIndex >= forcedSteps.size()) {
                prefixComplete = true;
                prefixReason = "prefix_complete";
            }
            recordDecision("forced", context, indices, "prefix_step_" + (forcedStepIndex - 1));
            return Choice.choose(indices);
        }

        private <T> void capturePostPrefix(DecisionContext<T> context) {
            if (postPrefixCaptured || !prefixComplete || context == null) {
                return;
            }
            postPrefixCaptured = true;
            postPrefixActionType = context.actionType == null ? "" : context.actionType.name();
            postPrefixCandidateHash = context.candidateHash;
            postPrefixStateHash = context.stateHash;
        }

        private <T> Choice postBranchAutopilotChoice(DecisionContext<T> context) {
            if (!postBranchAutopilot || context == null || context.candidateCount <= 0) {
                return Choice.none();
            }
            List<Integer> indices;
            if (continuationPolicy == ContinuationPolicy.EXPLORE) {
                indices = exploratoryAutopilotIndices(context, rolloutRandom);
            } else if (continuationPolicy == ContinuationPolicy.SAMPLE) {
                indices = sampledAutopilotIndices(context, rolloutRandom);
            } else {
                indices = deterministicAutopilotIndices(context);
            }
            if (indices.isEmpty()) {
                return Choice.none();
            }
            recordDecision("autopilot", context, indices, continuationPolicy.name().toLowerCase(Locale.US));
            return Choice.choose(indices);
        }

        private <T> void recordDecision(
                String role,
                DecisionContext<T> context,
                List<Integer> indices,
                String decisionReason
        ) {
            decisionCount++;
            if (decisionTrace.size() >= MAX_DECISION_TRACE_ROWS) {
                return;
            }
            String actionType = context == null || context.actionType == null ? "" : context.actionType.name();
            String candidateHash = context == null ? "" : context.candidateHash;
            String stateHash = context == null ? "" : context.stateHash;
            int candidateCount = context == null ? 0 : context.candidateCount;
            List<String> texts = context == null
                    ? Collections.emptyList()
                    : LiveCheckpointBranchMiner.selectedTexts(context.candidateTexts, indices);
            decisionTrace.add(String.format(Locale.US,
                    "%03d:%s:action=%s:indices=%s:texts=%s:candidates=%d:candidate_hash=%s:state_hash=%s:reason=%s",
                    decisionCount,
                    role == null ? "" : role,
                    actionType,
                    joinInts(indices),
                    joinStrings(texts),
                    candidateCount,
                    candidateHash == null ? "" : candidateHash,
                    stateHash == null ? "" : stateHash,
                    decisionReason == null ? "" : decisionReason));
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

    private static <T> List<Integer> exploratoryAutopilotIndices(
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
        StateSequenceBuilder.ActionType actionType = context.actionType;
        if (actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL || pickLimit == 1) {
            List<Integer> weighted = new ArrayList<>();
            for (int i = 0; i < candidateCount; i++) {
                String text = candidateText(context, i);
                int weight;
                if (isStopLike(text)) {
                    weight = 1;
                } else if (isPassLike(text) || isDoneLike(text)) {
                    weight = 3;
                } else if (isSpellLike(text)) {
                    weight = 5;
                } else if (isLandPlayLike(text)) {
                    weight = 4;
                } else if (isManaLike(text)) {
                    weight = 3;
                } else {
                    weight = 2;
                }
                for (int j = 0; j < weight; j++) {
                    weighted.add(i);
                }
            }
            if (weighted.isEmpty()) {
                return Collections.singletonList(0);
            }
            return Collections.singletonList(weighted.get(rng.nextInt(weighted.size())));
        }
        return sampledAutopilotIndices(context, rng);
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

    private static boolean choiceTextMatches(String expected, String actual) {
        String left = normalizeChoiceText(expected);
        String right = normalizeChoiceText(actual);
        if (left.isEmpty() || right.isEmpty()) {
            return false;
        }
        return left.equals(right)
                || left.endsWith(": " + right)
                || right.endsWith(": " + left);
    }

    private static boolean choiceTextsMatch(List<String> expected, List<String> actual) {
        if (expected == null || actual == null || expected.size() != actual.size()) {
            return false;
        }
        for (int i = 0; i < expected.size(); i++) {
            if (!choiceTextMatches(expected.get(i), actual.get(i))) {
                return false;
            }
        }
        return !expected.isEmpty();
    }

    private static String normalizeChoiceText(String text) {
        if (text == null) {
            return "";
        }
        return text.trim().replaceAll("\\s+", " ").toLowerCase(Locale.US);
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
        SAMPLE,
        EXPLORE;

        private static ContinuationPolicy parse(String raw) {
            if (raw == null || raw.trim().isEmpty()) {
                return STABLE;
            }
            String value = raw.trim().toUpperCase(Locale.US);
            if ("SAMPLED".equals(value) || "RANDOM".equals(value)) {
                return SAMPLE;
            }
            if ("EXPLORATORY".equals(value) || "SEARCH".equals(value)) {
                return EXPLORE;
            }
            return ContinuationPolicy.valueOf(value);
        }
    }

    private static final class SequenceTreeResult {
        private final List<SequenceTreeRow> rows;
        private final List<SequencePairSummary> summaries;

        private SequenceTreeResult(List<SequenceTreeRow> rows, List<SequencePairSummary> summaries) {
            this.rows = rows == null ? Collections.emptyList() : rows;
            this.summaries = summaries == null ? Collections.emptyList() : summaries;
        }

        private static SequenceTreeResult empty() {
            return new SequenceTreeResult(Collections.emptyList(), Collections.emptyList());
        }
    }

    private static final class SequenceTreeRow {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private String pairKey = "";
        private String orderLabel = "";
        private int rollout = 0;
        private String sequenceIndices = "";
        private String sequenceTexts = "";
        private int forcedStepsRequested = 0;
        private int forcedStepsCompleted = 0;
        private boolean prefixComplete = false;
        private String prefixReason = "";
        private String postPrefixStateHash = "";
        private String postPrefixCandidateHash = "";
        private String postPrefixActionType = "";
        private boolean terminal = false;
        private boolean won = false;
        private boolean lost = false;
        private String error = "";
        private String outcome = "";

        private static SequenceTreeRow fromSnapshot(
                Path path,
                LiveCheckpointRecorder.Snapshot snapshot,
                String pairKey,
                String orderLabel,
                int rollout,
                List<List<Integer>> sequenceChoices
        ) {
            SequenceTreeRow row = new SequenceTreeRow();
            row.snapshotPath = path == null ? "" : path.toString();
            row.pairKey = pairKey == null ? "" : pairKey;
            row.orderLabel = orderLabel == null ? "" : orderLabel;
            row.rollout = rollout;
            row.sequenceIndices = sequenceKey(sequenceChoices);
            if (snapshot != null) {
                row.ordinal = snapshot.ordinal;
                row.decisionNumber = snapshot.decisionNumber;
                row.actionType = snapshot.actionType;
                row.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                row.candidateHash = snapshot.candidateHash;
                row.stateHash = snapshot.stateHash;
                row.randomStateHash = snapshot.randomStateHash;
                List<String> texts = new ArrayList<>();
                if (sequenceChoices != null) {
                    for (List<Integer> choice : sequenceChoices) {
                        texts.add(joinStrings(selectedTexts(snapshot.candidateTexts, choice)));
                    }
                }
                row.sequenceTexts = joinStrings(texts);
            }
            return row;
        }

        private void apply(BranchOutcome outcome) {
            if (outcome == null) {
                error = "null_outcome";
                this.outcome = "error=null_outcome";
                return;
            }
            forcedStepsRequested = outcome.forcedStepsRequested;
            forcedStepsCompleted = outcome.forcedStepsCompleted;
            prefixComplete = outcome.prefixComplete;
            prefixReason = outcome.prefixReason.isEmpty() ? outcome.reason : outcome.prefixReason;
            postPrefixStateHash = outcome.postPrefixStateHash;
            postPrefixCandidateHash = outcome.postPrefixCandidateHash;
            postPrefixActionType = outcome.postPrefixActionType;
            terminal = outcome.terminal;
            won = outcome.won;
            lost = outcome.lost;
            error = outcome.error;
            this.outcome = outcome.shortClassification();
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
            cells.add(csv(pairKey));
            cells.add(csv(orderLabel));
            cells.add(String.valueOf(rollout));
            cells.add(csv(sequenceIndices));
            cells.add(csv(sequenceTexts));
            cells.add(String.valueOf(forcedStepsRequested));
            cells.add(String.valueOf(forcedStepsCompleted));
            cells.add(String.valueOf(prefixComplete));
            cells.add(csv(prefixReason));
            cells.add(csv(postPrefixStateHash));
            cells.add(csv(postPrefixCandidateHash));
            cells.add(csv(postPrefixActionType));
            cells.add(String.valueOf(terminal));
            cells.add(String.valueOf(won));
            cells.add(String.valueOf(lost));
            cells.add(csv(error));
            cells.add(csv(outcome));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class SequencePairSummary {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private String pairKey = "";
        private String firstIndices = "";
        private String firstTexts = "";
        private String secondIndices = "";
        private String secondTexts = "";
        private String classification = "";
        private int forwardRollouts = 0;
        private int reverseRollouts = 0;
        private int forwardPrefixComplete = 0;
        private int reversePrefixComplete = 0;
        private int convergedPostPrefixCount = 0;
        private int comparedPostPrefixCount = 0;
        private int forwardTerminalCount = 0;
        private int reverseTerminalCount = 0;
        private int forwardWinCount = 0;
        private int reverseWinCount = 0;
        private int forwardLossCount = 0;
        private int reverseLossCount = 0;
        private int forwardErrorCount = 0;
        private int reverseErrorCount = 0;
        private final List<String> forwardPostPrefixHashes = new ArrayList<>();
        private final List<String> reversePostPrefixHashes = new ArrayList<>();
        private final List<String> forwardOutcomes = new ArrayList<>();
        private final List<String> reverseOutcomes = new ArrayList<>();

        private static SequencePairSummary fromSnapshot(
                Path path,
                LiveCheckpointRecorder.Snapshot snapshot,
                List<Integer> first,
                List<Integer> second
        ) {
            SequencePairSummary row = new SequencePairSummary();
            row.snapshotPath = path == null ? "" : path.toString();
            row.pairKey = sequenceKey(first, second);
            row.firstIndices = joinInts(first);
            row.secondIndices = joinInts(second);
            if (snapshot != null) {
                row.ordinal = snapshot.ordinal;
                row.decisionNumber = snapshot.decisionNumber;
                row.actionType = snapshot.actionType;
                row.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                row.candidateHash = snapshot.candidateHash;
                row.stateHash = snapshot.stateHash;
                row.randomStateHash = snapshot.randomStateHash;
                row.firstTexts = joinStrings(selectedTexts(snapshot.candidateTexts, first));
                row.secondTexts = joinStrings(selectedTexts(snapshot.candidateTexts, second));
            }
            return row;
        }

        private void add(SequenceTreeRow forward, SequenceTreeRow reverse) {
            forwardRollouts++;
            reverseRollouts++;
            if (forward != null) {
                if (forward.prefixComplete) {
                    forwardPrefixComplete++;
                }
                if (forward.terminal) {
                    forwardTerminalCount++;
                }
                if (forward.won) {
                    forwardWinCount++;
                }
                if (forward.lost) {
                    forwardLossCount++;
                }
                if (!forward.error.isEmpty()) {
                    forwardErrorCount++;
                }
                if (!forward.postPrefixStateHash.isEmpty()) {
                    forwardPostPrefixHashes.add(forward.postPrefixStateHash);
                }
                forwardOutcomes.add(forward.outcome);
            }
            if (reverse != null) {
                if (reverse.prefixComplete) {
                    reversePrefixComplete++;
                }
                if (reverse.terminal) {
                    reverseTerminalCount++;
                }
                if (reverse.won) {
                    reverseWinCount++;
                }
                if (reverse.lost) {
                    reverseLossCount++;
                }
                if (!reverse.error.isEmpty()) {
                    reverseErrorCount++;
                }
                if (!reverse.postPrefixStateHash.isEmpty()) {
                    reversePostPrefixHashes.add(reverse.postPrefixStateHash);
                }
                reverseOutcomes.add(reverse.outcome);
            }
            if (forward != null
                    && reverse != null
                    && !forward.postPrefixStateHash.isEmpty()
                    && !reverse.postPrefixStateHash.isEmpty()) {
                comparedPostPrefixCount++;
                if (forward.postPrefixStateHash.equals(reverse.postPrefixStateHash)) {
                    convergedPostPrefixCount++;
                }
            }
        }

        private void classify() {
            if (forwardRollouts <= 0 || reverseRollouts <= 0) {
                classification = "sequence_not_run";
                return;
            }
            if (forwardPrefixComplete <= 0 || reversePrefixComplete <= 0) {
                classification = "sequence_incomplete";
                return;
            }
            if (forwardErrorCount > 0 || reverseErrorCount > 0) {
                classification = "sequence_error";
                return;
            }
            if (comparedPostPrefixCount > 0 && convergedPostPrefixCount == comparedPostPrefixCount) {
                classification = "order_converged";
                return;
            }
            double forwardWinRate = ((double) forwardWinCount) / Math.max(1, forwardRollouts);
            double reverseWinRate = ((double) reverseWinCount) / Math.max(1, reverseRollouts);
            if (forwardWinRate > reverseWinRate) {
                classification = "order_sensitive_forward_better";
                return;
            }
            if (reverseWinRate > forwardWinRate) {
                classification = "order_sensitive_reverse_better";
                return;
            }
            if (forwardTerminalCount <= 0 && reverseTerminalCount <= 0) {
                classification = "order_diverged_no_terminal";
                return;
            }
            classification = "order_diverged_same_value";
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
            cells.add(csv(pairKey));
            cells.add(csv(firstIndices));
            cells.add(csv(firstTexts));
            cells.add(csv(secondIndices));
            cells.add(csv(secondTexts));
            cells.add(csv(classification));
            cells.add(String.valueOf(forwardRollouts));
            cells.add(String.valueOf(reverseRollouts));
            cells.add(String.valueOf(forwardPrefixComplete));
            cells.add(String.valueOf(reversePrefixComplete));
            cells.add(String.valueOf(convergedPostPrefixCount));
            cells.add(String.valueOf(comparedPostPrefixCount));
            cells.add(String.valueOf(forwardTerminalCount));
            cells.add(String.valueOf(reverseTerminalCount));
            cells.add(String.valueOf(forwardWinCount));
            cells.add(String.valueOf(reverseWinCount));
            cells.add(String.valueOf(forwardLossCount));
            cells.add(String.valueOf(reverseLossCount));
            cells.add(String.valueOf(forwardErrorCount));
            cells.add(String.valueOf(reverseErrorCount));
            cells.add(csv(joinStrings(forwardPostPrefixHashes)));
            cells.add(csv(joinStrings(reversePostPrefixHashes)));
            cells.add(csv(joinStrings(forwardOutcomes)));
            cells.add(csv(joinStrings(reverseOutcomes)));
            return String.join(",", cells) + "\n";
        }
    }

    private static final class ValueTreeResult {
        private final ValueTreeSummary summary;
        private final List<ValueActionStats> actions;
        private final List<SequenceTreeRow> sequenceRows;
        private final List<SequencePairSummary> sequenceSummaries;

        private ValueTreeResult(ValueTreeSummary summary, List<ValueActionStats> actions) {
            this(summary, actions, Collections.emptyList(), Collections.emptyList());
        }

        private ValueTreeResult(
                ValueTreeSummary summary,
                List<ValueActionStats> actions,
                List<SequenceTreeRow> sequenceRows,
                List<SequencePairSummary> sequenceSummaries
        ) {
            this.summary = summary == null ? new ValueTreeSummary() : summary;
            this.actions = actions == null ? Collections.emptyList() : actions;
            this.sequenceRows = sequenceRows == null ? Collections.emptyList() : sequenceRows;
            this.sequenceSummaries = sequenceSummaries == null ? Collections.emptyList() : sequenceSummaries;
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
        private int decisionCount;
        private int forcedStepsRequested;
        private int forcedStepsCompleted;
        private boolean prefixComplete;
        private String error = "";
        private String terminationReason = "";
        private String actionType = "";
        private String candidateHash = "";
        private String stateHash = "";
        private String prefixReason = "";
        private String postPrefixActionType = "";
        private String postPrefixCandidateHash = "";
        private String postPrefixStateHash = "";
        private String finalStateHash = "";
        private String reason = "";
        private List<String> decisionTrace = Collections.emptyList();
        private List<StateSequenceBuilder.TrainingData> trainingData = Collections.emptyList();

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
            decisionCount = controller.decisionCount;
            decisionTrace = new ArrayList<>(controller.decisionTrace);
            forcedStepsRequested = controller.forcedSteps.size();
            forcedStepsCompleted = controller.forcedStepsCompleted;
            prefixComplete = controller.prefixComplete;
            prefixReason = controller.prefixReason;
            postPrefixActionType = controller.postPrefixActionType;
            postPrefixCandidateHash = controller.postPrefixCandidateHash;
            postPrefixStateHash = controller.postPrefixStateHash;
            if (!firstDecisionSeen && error.isEmpty()) {
                error = "checkpoint_no_reentry_decision";
            }
            trainingData = new ArrayList<>(controller.trainingData);
        }

        private void captureTerminal(Game game, String perspectiveName) {
            try {
                terminal = game != null && game.hasEnded();
                String winner = game == null ? "" : game.getWinner();
                String name = perspectiveName == null ? "" : perspectiveName;
                won = terminal && winner != null && !winner.isEmpty() && !name.isEmpty() && winner.contains(name);
                lost = terminal && winner != null && !winner.isEmpty() && !won;
                Player perspective = null;
                if (game != null && game.getPlayers() != null) {
                    for (Player player : game.getPlayers().values()) {
                        String playerName = player == null || player.getName() == null ? "" : player.getName();
                        if (player != null && (name.isEmpty() || playerName.contains(name) || name.contains(playerName))) {
                            perspective = player;
                            break;
                        }
                    }
                }
                if (game != null) {
                    finalStateHash = LiveCheckpointRecorder.sha256(LiveCheckpointRecorder.compactState(game, perspective));
                }
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

    private static final class TerminalLineRow {
        private String snapshotPath = "";
        private int ordinal = -1;
        private int decisionNumber = -1;
        private String actionType = "";
        private int candidateCount = 0;
        private String candidateHash = "";
        private String stateHash = "";
        private String randomStateHash = "";
        private int attempt = 0;
        private int continuationSample = -1;
        private long continuationSeed = 0L;
        private String rootIndices = "";
        private String rootTexts = "";
        private boolean terminal = false;
        private boolean won = false;
        private boolean lost = false;
        private String error = "";
        private String outcome = "";
        private int decisionCount = 0;
        private int forcedStepsRequested = 0;
        private int forcedStepsCompleted = 0;
        private boolean prefixComplete = false;
        private String finalStateHash = "";
        private String lineTrace = "";

        private static TerminalLineRow fromSnapshot(
                Path path,
                LiveCheckpointRecorder.Snapshot snapshot,
                int attempt,
                int continuationSample,
                long continuationSeed,
                List<Integer> rootChoice
        ) {
            TerminalLineRow row = new TerminalLineRow();
            row.snapshotPath = path == null ? "" : path.toString();
            row.attempt = attempt;
            row.continuationSample = continuationSample;
            row.continuationSeed = continuationSeed;
            if (snapshot != null) {
                row.ordinal = snapshot.ordinal;
                row.decisionNumber = snapshot.decisionNumber;
                row.actionType = snapshot.actionType;
                row.candidateCount = snapshot.candidateTexts == null ? 0 : snapshot.candidateTexts.size();
                row.candidateHash = snapshot.candidateHash;
                row.stateHash = snapshot.stateHash;
                row.randomStateHash = snapshot.randomStateHash;
                row.rootTexts = joinStrings(selectedTexts(snapshot.candidateTexts, rootChoice));
            }
            row.rootIndices = joinInts(rootChoice);
            return row;
        }

        private static TerminalLineRow loadError(Path path, String error) {
            TerminalLineRow row = new TerminalLineRow();
            row.snapshotPath = path == null ? "" : path.toString();
            row.error = error == null ? "" : error;
            row.outcome = "error=" + row.error;
            return row;
        }

        private static TerminalLineRow failure(Path path, LiveCheckpointRecorder.Snapshot snapshot, String error) {
            TerminalLineRow row = fromSnapshot(path, snapshot, 0, -1, 0L, Collections.emptyList());
            row.error = error == null ? "" : error;
            row.outcome = "error=" + row.error;
            return row;
        }

        private void apply(BranchOutcome branch) {
            if (branch == null) {
                error = "null_outcome";
                outcome = "error=null_outcome";
                return;
            }
            terminal = branch.terminal;
            won = branch.won;
            lost = branch.lost;
            error = branch.error;
            outcome = branch.shortClassification();
            decisionCount = branch.decisionCount;
            forcedStepsRequested = branch.forcedStepsRequested;
            forcedStepsCompleted = branch.forcedStepsCompleted;
            prefixComplete = branch.prefixComplete;
            finalStateHash = branch.finalStateHash;
            lineTrace = joinStrings(branch.decisionTrace);
        }

        private String classification() {
            if (error != null && !error.isEmpty()) {
                return "error";
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
            return "not_terminal";
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
            cells.add(String.valueOf(attempt));
            cells.add(String.valueOf(continuationSample));
            cells.add(String.valueOf(continuationSeed));
            cells.add(csv(rootIndices));
            cells.add(csv(rootTexts));
            cells.add(String.valueOf(terminal));
            cells.add(String.valueOf(won));
            cells.add(String.valueOf(lost));
            cells.add(csv(error));
            cells.add(csv(outcome));
            cells.add(String.valueOf(decisionCount));
            cells.add(String.valueOf(forcedStepsRequested));
            cells.add(String.valueOf(forcedStepsCompleted));
            cells.add(String.valueOf(prefixComplete));
            cells.add(csv(finalStateHash));
            cells.add(csv(lineTrace));
            return String.join(",", cells) + "\n";
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
        private final int loadErrorCount;
        private final List<String> loadErrorExamples;
        private final List<SnapshotCandidate> selected;

        private Selection(
                int discoveredPathCount,
                int eligibleCount,
                int loadErrorCount,
                List<String> loadErrorExamples,
                List<SnapshotCandidate> selected
        ) {
            this.discoveredPathCount = discoveredPathCount;
            this.eligibleCount = eligibleCount;
            this.loadErrorCount = loadErrorCount;
            this.loadErrorExamples = loadErrorExamples == null ? Collections.emptyList() : loadErrorExamples;
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
        private Path snapshotListPath;
        private Path outDir = defaultOutDir();
        private int maxSnapshots = 0;
        private int timeoutSec = 30;
        private int alternateTimeoutSec = 30;
        private int maxAlternates = 1;
        private String selectionMode = "path";
        private int rankedMaxPerGame = 0;
        private int selectionShards = 1;
        private int selectionShardIndex = 0;
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
        private boolean terminalLineSearch = false;
        private int lineAttempts = 64;
        private int lineMaxRootActions = 0;
        private int lineTimeoutSec = 30;
        private boolean lineStopOnWin = true;
        private boolean lineStopOnWinAll = true;
        private boolean lineCommonContinuationSeeds = false;
        private boolean lineCaptureTrainingData = false;
        private int lineTrainingMaxRecordsPerBranch = 64;
        private boolean sequenceTree = false;
        private int treeSequenceDepth = 2;
        private int treeSequenceBeam = 4;
        private int treeSequenceRollouts = 1;

        private static Config parse(String[] args) {
            Config cfg = new Config();
            Map<String, String> values = parseArgs(args);
            if (values.containsKey("checkpoint-root")) {
                cfg.checkpointRoot = Paths.get(values.get("checkpoint-root"));
            }
            if (values.containsKey("snapshot")) {
                cfg.snapshotPath = Paths.get(values.get("snapshot"));
            }
            if (values.containsKey("snapshot-list")) {
                cfg.snapshotListPath = Paths.get(values.get("snapshot-list"));
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
            if (values.containsKey("selection-shards")) {
                cfg.selectionShards = Integer.parseInt(values.get("selection-shards"));
            }
            if (values.containsKey("shards")) {
                cfg.selectionShards = Integer.parseInt(values.get("shards"));
            }
            if (values.containsKey("selection-shard-index")) {
                cfg.selectionShardIndex = Integer.parseInt(values.get("selection-shard-index"));
            }
            if (values.containsKey("shard-index")) {
                cfg.selectionShardIndex = Integer.parseInt(values.get("shard-index"));
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
            if (values.containsKey("terminal-line-search")) {
                cfg.terminalLineSearch = Boolean.parseBoolean(values.get("terminal-line-search"));
            }
            if (values.containsKey("line-attempts")) {
                cfg.lineAttempts = Math.max(1, Integer.parseInt(values.get("line-attempts")));
            }
            if (values.containsKey("line-max-root-actions")) {
                cfg.lineMaxRootActions = Math.max(0, Integer.parseInt(values.get("line-max-root-actions")));
            }
            if (values.containsKey("line-timeout-sec")) {
                cfg.lineTimeoutSec = Math.max(1, Integer.parseInt(values.get("line-timeout-sec")));
            } else {
                cfg.lineTimeoutSec = cfg.treeTimeoutSec;
            }
            if (values.containsKey("line-stop-on-win")) {
                cfg.lineStopOnWin = Boolean.parseBoolean(values.get("line-stop-on-win"));
            }
            if (values.containsKey("line-stop-on-win-all")) {
                cfg.lineStopOnWinAll = Boolean.parseBoolean(values.get("line-stop-on-win-all"));
            }
            if (values.containsKey("line-common-continuation-seeds")) {
                cfg.lineCommonContinuationSeeds = Boolean.parseBoolean(values.get("line-common-continuation-seeds"));
            }
            if (values.containsKey("line-capture-training-data")) {
                cfg.lineCaptureTrainingData = Boolean.parseBoolean(values.get("line-capture-training-data"));
            }
            if (values.containsKey("line-training-max-records-per-branch")) {
                cfg.lineTrainingMaxRecordsPerBranch = Math.max(0,
                        Integer.parseInt(values.get("line-training-max-records-per-branch")));
            }
            if (values.containsKey("sequence-tree")) {
                cfg.sequenceTree = Boolean.parseBoolean(values.get("sequence-tree"));
            }
            if (values.containsKey("tree-sequence-depth")) {
                cfg.treeSequenceDepth = Math.max(2, Integer.parseInt(values.get("tree-sequence-depth")));
            }
            if (values.containsKey("tree-sequence-beam")) {
                cfg.treeSequenceBeam = Math.max(0, Integer.parseInt(values.get("tree-sequence-beam")));
            }
            if (values.containsKey("tree-sequence-rollouts")) {
                cfg.treeSequenceRollouts = Math.max(1, Integer.parseInt(values.get("tree-sequence-rollouts")));
            }
            if (cfg.selectionShards < 1) {
                throw new IllegalArgumentException("--selection-shards must be >= 1");
            }
            if (cfg.selectionShardIndex < 0 || cfg.selectionShardIndex >= cfg.selectionShards) {
                throw new IllegalArgumentException("--selection-shard-index must be between 0 and selection-shards - 1");
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
