package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.common.PassAbility;
import mage.cards.Card;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * Eval-only terminal prefix search for Spy-style combo turns.
 *
 * <p>This intentionally mirrors the successful offline terminal-prefix teacher.
 * It is opt-in and never records training data. The default branch ordering
 * retains the historical diagnostic tactic ordering; {@link Config#genericBranchOrder}
 * disables those card-name preferences for thesis-clean search probes.</p>
 */
public final class TerminalPrefixSearch {

    private TerminalPrefixSearch() {
    }

    public static final class Config {
        public int maxNodes = 7;
        public int maxDepth = 6;
        public int topK = 3;
        public int maxGameTurns = 0;
        public int selectedIndex = -1;
        public long totalTimeoutMs = 1500L;
        public long branchTimeoutMs = 350L;
        public boolean log = false;
        public boolean returnSameActionWins = false;
        public boolean modelGuidedFallback = false;
        public boolean genericBranchOrder = false;
    }

    public static final class Result {
        public final int actionIndex;
        public final int nodesRun;
        public final int queued;
        public final long wallMs;
        public final boolean foundWin;
        public final boolean timedOut;
        public final String reason;
        public final String actionText;
        public final List<String> prefixTexts;

        private Result(int actionIndex, int nodesRun, int queued, long wallMs,
                       boolean foundWin, boolean timedOut, String reason, String actionText) {
            this(actionIndex, nodesRun, queued, wallMs, foundWin, timedOut, reason, actionText, Collections.emptyList());
        }

        private Result(int actionIndex, int nodesRun, int queued, long wallMs,
                       boolean foundWin, boolean timedOut, String reason, String actionText,
                       List<String> prefixTexts) {
            this.actionIndex = actionIndex;
            this.nodesRun = nodesRun;
            this.queued = queued;
            this.wallMs = wallMs;
            this.foundWin = foundWin;
            this.timedOut = timedOut;
            this.reason = reason == null ? "" : reason;
            this.actionText = actionText == null ? "" : actionText;
            this.prefixTexts = Collections.unmodifiableList(
                    prefixTexts == null ? Collections.emptyList() : new ArrayList<>(prefixTexts));
        }

        public boolean hasAction() {
            return foundWin && actionIndex >= 0;
        }
    }

    public static <T> Result search(
            Game liveGame,
            UUID selfId,
            List<T> rootCandidates,
            StateSequenceBuilder.ActionType rootActionType,
            Ability rootSource,
            float[] rootPolicyProbs,
            PythonModel model,
            Config config
    ) {
        long searchStart = System.nanoTime();
        Config cfg = config == null ? new Config() : config;
        if (liveGame == null || selfId == null || rootCandidates == null || rootCandidates.size() < 2) {
            return new Result(-1, 0, 0, 0L, false, false, "invalid_input", "");
        }
        if (rootActionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
            return new Result(-1, 0, 0, 0L, false, false, "unsupported_root_type", "");
        }
        Player self = liveGame.getPlayer(selfId);
        if (self == null) {
            return new Result(-1, 0, 0, 0L, false, false, "missing_player", "");
        }
        int candidateCount = Math.min(rootCandidates.size(), StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        List<String> rootTexts = describeCandidates(rootCandidates, liveGame, candidateCount);
        StateSnapshot rootSnapshot = StateSnapshot.capture(liveGame, self);
        List<Integer> rootOrder = branchCandidates(
                rootActionType,
                rootTexts,
                rootPolicyProbs,
                cfg.selectedIndex,
                rootSnapshot,
                Math.max(1, cfg.topK),
                cfg.genericBranchOrder);
        if (rootOrder.isEmpty()) {
            return new Result(-1, 0, 0, 0L, false, false, "empty_root_order", "");
        }

        int maxNodes = Math.max(1, cfg.maxNodes);
        int maxDepth = Math.max(1, cfg.maxDepth);
        long totalDeadline = searchStart + TimeUnit.MILLISECONDS.toNanos(Math.max(1L, cfg.totalTimeoutMs));
        Queue<PrefixNode> queue = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        for (Integer idx : rootOrder) {
            if (idx == null || idx < 0 || idx >= candidateCount) {
                continue;
            }
            PrefixNode node = new PrefixNode(idx, Collections.singletonList(rootTexts.get(idx)));
            if (seen.add(node.key())) {
                queue.add(node);
            }
        }

        int nodesRun = 0;
        boolean anyTimedOut = false;
        boolean sameActionWinOnly = false;
        String lastReason = "";
        while (!queue.isEmpty()
                && nodesRun < maxNodes
                && System.nanoTime() < totalDeadline) {
            PrefixNode node = queue.poll();
            long branchDeadline = Math.min(
                    totalDeadline,
                    System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(1L, cfg.branchTimeoutMs)));
            RunResult run = runPrefixBranch(liveGame, selfId, model, node.prefixTexts, cfg, branchDeadline);
            nodesRun++;
            anyTimedOut |= run.timedOut;
            lastReason = run.reason;
            if (run.won) {
                if (node.rootIndex == cfg.selectedIndex && !cfg.returnSameActionWins) {
                    sameActionWinOnly = true;
                    continue;
                }
                long wallMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - searchStart);
                String text = node.rootIndex >= 0 && node.rootIndex < rootTexts.size()
                        ? rootTexts.get(node.rootIndex)
                        : "";
                return new Result(node.rootIndex, nodesRun, queue.size(), wallMs,
                        true, anyTimedOut, "win", text, run.executedTexts);
            }
            if (node.prefixTexts.size() >= maxDepth || run.branchPoint == null) {
                continue;
            }
            for (Integer childIdx : branchCandidates(
                    run.branchPoint.actionType,
                    run.branchPoint.candidateTexts,
                    run.branchPoint.probs,
                    run.branchPoint.chosenIndex,
                    run.branchPoint.snapshot,
                    Math.max(1, cfg.topK),
                    cfg.genericBranchOrder)) {
                if (childIdx == null || childIdx < 0 || childIdx >= run.branchPoint.candidateTexts.size()) {
                    continue;
                }
                List<String> childPrefix = new ArrayList<>(node.prefixTexts);
                childPrefix.add(run.branchPoint.candidateTexts.get(childIdx));
                PrefixNode child = new PrefixNode(node.rootIndex, childPrefix);
                if (seen.add(child.key())) {
                    queue.add(child);
                }
            }
        }
        long wallMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - searchStart);
        String reason = System.nanoTime() >= totalDeadline
                ? "search_timeout"
                : (sameActionWinOnly ? "same_action_win_only" : (lastReason.isEmpty() ? "no_win" : lastReason));
        return new Result(-1, nodesRun, queue.size(), wallMs, false,
                anyTimedOut || System.nanoTime() >= totalDeadline, reason, "");
    }

    /** Config for the in-loop per-candidate terminal win-rate estimator. */
    public static final class WinRateConfig {
        public int topK = 3;
        public int playoutsPerCandidate = 2;
        public long perPlayoutTimeoutMs = 1500L;
        public long totalTimeoutMs = 8000L;
        public int maxGameTurns = 0;
        public boolean genericBranchOrder = true;
        public boolean modelGuidedFallback = false;
        public long baseSeed = 0L;
        public boolean log = false;
    }

    /** Per-candidate terminal win-rate result. winRate[i] is NaN for un-branched slots. */
    public static final class WinRateResult {
        public final float[] winRate;
        public final int[] terminals;
        public final int[] wins;
        public final int bestIndex;
        public final int observedCount;
        public final long wallMs;
        public final int totalPlayouts;
        public final int totalTerminals;

        WinRateResult(float[] winRate, int[] terminals, int[] wins, int bestIndex,
                      int observedCount, long wallMs, int totalPlayouts, int totalTerminals) {
            this.winRate = winRate;
            this.terminals = terminals;
            this.wins = wins;
            this.bestIndex = bestIndex;
            this.observedCount = observedCount;
            this.wallMs = wallMs;
            this.totalPlayouts = totalPlayouts;
            this.totalTerminals = totalTerminals;
        }
    }

    /**
     * In-loop terminal-win-rate estimator for the policy-distillation operator
     * (scope_B). Branches the top-K root candidates using thesis-clean generic
     * ordering, plays each forward to a REAL terminal under per-thread RNG
     * isolation, and returns the per-candidate terminal win-rate. Pure terminal
     * win/lose signal -- no value-leaf substitution, no card-name preferences.
     * Every playout is a simulation copy, so no training data is recorded inside.
     */
    public static <T> WinRateResult estimateCandidateWinRates(
            Game liveGame,
            UUID selfId,
            List<T> rootCandidates,
            StateSequenceBuilder.ActionType rootActionType,
            float[] rootPolicyProbs,
            PythonModel model,
            WinRateConfig wcfg
    ) {
        long start = System.nanoTime();
        int maxC = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        float[] winRate = new float[maxC];
        Arrays.fill(winRate, Float.NaN);
        int[] terminals = new int[maxC];
        int[] wins = new int[maxC];
        WinRateConfig cfgw = wcfg == null ? new WinRateConfig() : wcfg;
        if (liveGame == null || selfId == null || rootCandidates == null || rootCandidates.size() < 2
                || rootActionType != StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL) {
            return new WinRateResult(winRate, terminals, wins, -1, 0, 0L, 0, 0);
        }
        Player self = liveGame.getPlayer(selfId);
        if (self == null) {
            return new WinRateResult(winRate, terminals, wins, -1, 0, 0L, 0, 0);
        }
        int candidateCount = Math.min(rootCandidates.size(), maxC);
        List<String> rootTexts = describeCandidates(rootCandidates, liveGame, candidateCount);
        StateSnapshot snapshot = StateSnapshot.capture(liveGame, self);
        Config cfg = new Config();
        cfg.genericBranchOrder = cfgw.genericBranchOrder;
        cfg.modelGuidedFallback = cfgw.modelGuidedFallback;
        cfg.maxGameTurns = cfgw.maxGameTurns;
        cfg.log = cfgw.log;
        List<Integer> order = branchCandidates(
                rootActionType, rootTexts, rootPolicyProbs, -1, snapshot,
                Math.max(1, cfgw.topK), cfgw.genericBranchOrder);
        long totalDeadline = start + TimeUnit.MILLISECONDS.toNanos(Math.max(1L, cfgw.totalTimeoutMs));
        int totalPlayouts = 0;
        int totalTerminals = 0;
        for (Integer idx : order) {
            if (idx == null || idx < 0 || idx >= candidateCount) {
                continue;
            }
            if (System.nanoTime() >= totalDeadline) {
                break;
            }
            List<String> prefix = Collections.singletonList(rootTexts.get(idx));
            int t = 0;
            int w = 0;
            for (int p = 0; p < Math.max(1, cfgw.playoutsPerCandidate); p++) {
                if (System.nanoTime() >= totalDeadline) {
                    break;
                }
                long playoutDeadline = Math.min(totalDeadline,
                        System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(1L, cfgw.perPlayoutTimeoutMs)));
                long seed = cfgw.baseSeed ^ (0x9E3779B97F4A7C15L * (idx * 131L + p + 1L));
                Outcome o;
                try (mage.util.RandomUtil.RandomIsolation ignored =
                             mage.util.RandomUtil.isolateThreadLocalRandom(seed)) {
                    o = runWinRatePlayout(liveGame, selfId, model, prefix, cfg, playoutDeadline);
                }
                if (!o.prefixApplied) {
                    continue;
                }
                totalPlayouts++;
                if (o.terminal) {
                    t++;
                    totalTerminals++;
                    if (o.won) {
                        w++;
                    }
                }
            }
            if (t > 0) {
                winRate[idx] = (float) w / (float) t;
                terminals[idx] = t;
                wins[idx] = w;
            }
        }
        int best = -1;
        float bestWr = Float.NEGATIVE_INFINITY;
        int observed = 0;
        for (int i = 0; i < candidateCount; i++) {
            if (!Float.isNaN(winRate[i])) {
                observed++;
                if (winRate[i] > bestWr) {
                    bestWr = winRate[i];
                    best = i;
                }
            }
        }
        long wallMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
        return new WinRateResult(winRate, terminals, wins, best, observed, wallMs, totalPlayouts, totalTerminals);
    }

    private static final class Outcome {
        final boolean prefixApplied;
        final boolean terminal;
        final boolean won;
        final boolean timedOut;

        Outcome(boolean prefixApplied, boolean terminal, boolean won, boolean timedOut) {
            this.prefixApplied = prefixApplied;
            this.terminal = terminal;
            this.won = won;
            this.timedOut = timedOut;
        }

        static Outcome failed() {
            return new Outcome(false, false, false, false);
        }
    }

    /** One playout: fork the live game, force the prefix, resume to terminal, read outcome.
     *  Mirrors {@link #runPrefixBranch} but reports terminal-reached so a win-rate
     *  denominator can exclude deadline-truncated (non-terminal) playouts. */
    private static Outcome runWinRatePlayout(
            Game liveGame,
            UUID selfId,
            PythonModel model,
            List<String> prefixTexts,
            Config cfg,
            long deadlineNanos
    ) {
        Game sim;
        try {
            sim = liveGame.createSimulationForAI();
        } catch (Throwable t) {
            return Outcome.failed();
        }
        SearchPlayer player = replaceSelf(sim, selfId, model, prefixTexts, cfg, deadlineNanos);
        if (player == null) {
            return Outcome.failed();
        }
        boolean timedOut = false;
        try {
            sim.resume();
        } catch (SearchTerminated t) {
            timedOut = "deadline".equals(String.valueOf(t.getMessage()));
        } catch (Throwable ignored) {
            // playout failed mid-resolution; treat as non-terminal
        }
        if (System.nanoTime() >= deadlineNanos) {
            timedOut = true;
        }
        boolean prefixApplied = false;
        boolean terminal = false;
        boolean won = false;
        try {
            prefixApplied = player.prefixAppliedCount() >= prefixTexts.size();
            terminal = sim.hasEnded();
            won = terminal && sim.getWinner() != null && sim.getWinner().contains(player.getName());
        } catch (Throwable ignored) {
            // leave defaults on read failure
        }
        return new Outcome(prefixApplied, terminal, won, timedOut);
    }

    public static boolean isControllableActionType(StateSequenceBuilder.ActionType actionType) {
        return actionType == StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL
                || actionType == StateSequenceBuilder.ActionType.SELECT_TARGETS
                || actionType == StateSequenceBuilder.ActionType.SELECT_CARD
                || actionType == StateSequenceBuilder.ActionType.CHOOSE_USE
                || actionType == StateSequenceBuilder.ActionType.CHOOSE_MODE
                || actionType == StateSequenceBuilder.ActionType.ANNOUNCE_X
                || actionType == StateSequenceBuilder.ActionType.MULLIGAN
                || actionType == StateSequenceBuilder.ActionType.LONDON_MULLIGAN;
    }

    public static <T> int findMatchingCandidateIndex(
            List<T> candidates,
            int candidateCount,
            Game game,
            Ability source,
            String expectedText
    ) {
        int count = Math.min(candidates == null ? 0 : candidates.size(),
                Math.min(candidateCount, StateSequenceBuilder.TrainingData.MAX_CANDIDATES));
        if (count <= 0 || expectedText == null || expectedText.trim().isEmpty()) {
            return -1;
        }
        List<String> texts = describeCandidates(candidates, game, count);
        for (int i = 0; i < texts.size(); i++) {
            if (textMatches(expectedText, texts.get(i))) {
                return i;
            }
        }
        return -1;
    }

    private static RunResult runPrefixBranch(
            Game liveGame,
            UUID selfId,
            PythonModel model,
            List<String> prefixTexts,
            Config cfg,
            long branchDeadlineNanos
    ) {
        Game sim;
        try {
            sim = liveGame.createSimulationForAI();
        } catch (Throwable t) {
            return RunResult.failed("clone_failed:" + exceptionSummary(t), false);
        }
        SearchPlayer player = replaceSelf(sim, selfId, model, prefixTexts, cfg, branchDeadlineNanos);
        if (player == null) {
            return RunResult.failed("replace_failed", false);
        }

        boolean timedOut = false;
        String reason = "";
        try {
            sim.resume();
        } catch (SearchTerminated t) {
            reason = String.valueOf(t.getMessage());
            timedOut = "deadline".equals(reason);
        } catch (Throwable t) {
            reason = exceptionSummary(t);
        }
        if (System.nanoTime() >= branchDeadlineNanos) {
            timedOut = true;
            if (reason.isEmpty()) {
                reason = "branch_timeout";
            }
        }
        boolean won = false;
        try {
            won = player.prefixAppliedCount() >= prefixTexts.size()
                    && sim.hasEnded()
                    && sim.getWinner() != null
                    && sim.getWinner().contains(player.getName());
        } catch (Throwable ignored) {
            won = false;
        }
        if (player.invalidPrefix() && reason.isEmpty()) {
            reason = "prefix_not_found";
        }
        return new RunResult(won, timedOut, reason, player.firstBranchPoint(), player.executedTrace());
    }

    private static SearchPlayer replaceSelf(
            Game sim,
            UUID selfId,
            PythonModel model,
            List<String> prefixTexts,
            Config cfg,
            long deadlineNanos
    ) {
        try {
            Player oldPlayer = sim.getState().getPlayers().get(selfId);
            if (oldPlayer == null) {
                return null;
            }
            if (!(oldPlayer instanceof ComputerPlayerRL)) {
                return null;
            }
            SearchPlayer replacement = new SearchPlayer((ComputerPlayerRL) oldPlayer, prefixTexts, cfg, deadlineNanos);
            sim.getState().getPlayers().put(selfId, replacement);
            return replacement;
        } catch (Throwable t) {
            return null;
        }
    }

    private static List<Integer> branchCandidates(
            StateSequenceBuilder.ActionType actionType,
            List<String> candidateTexts,
            float[] probs,
            int selectedIndex,
            StateSnapshot snapshot,
            int topK,
            boolean genericBranchOrder
    ) {
        int count = candidateTexts == null ? 0
                : Math.min(candidateTexts.size(), StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
        if (count <= 0) {
            return Collections.emptyList();
        }
        List<Integer> valid = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            valid.add(i);
        }
        valid.sort((a, b) -> Float.compare(probAt(probs, b), probAt(probs, a)));

        LinkedHashSet<Integer> chosen = new LinkedHashSet<>();
        if (!genericBranchOrder) {
            for (Integer idx : preferredBranchCandidates(actionType, candidateTexts, probs, snapshot, valid)) {
                chosen.add(idx);
            }
        }
        for (Integer idx : valid) {
            if (chosen.size() >= topK) {
                break;
            }
            chosen.add(idx);
        }
        if (selectedIndex >= 0 && selectedIndex < count) {
            chosen.add(selectedIndex);
        }
        List<Integer> out = new ArrayList<>(chosen);
        out.sort(Comparator
                .comparingInt((Integer idx) -> genericBranchOrder
                        ? genericBranchOrderPriority(candidateTexts.get(idx))
                        : branchOrderPriority(actionType, candidateTexts.get(idx), snapshot))
                .thenComparing((Integer a, Integer b) -> Float.compare(probAt(probs, b), probAt(probs, a)))
                .thenComparingInt(Integer::intValue));
        return out;
    }

    private static int genericBranchOrderPriority(String candidateText) {
        return isPassText(candidateText) ? 100_000 : 50_000;
    }

    private static List<Integer> preferredBranchCandidates(
            StateSequenceBuilder.ActionType actionType,
            List<String> candidateTexts,
            float[] probs,
            StateSnapshot snapshot,
            List<Integer> valid
    ) {
        List<Integer> preferred = new ArrayList<>();
        for (Integer idx : valid) {
            if (preferredTacticPriority(actionType, candidateTexts.get(idx), snapshot) < Integer.MAX_VALUE) {
                preferred.add(idx);
            }
        }
        preferred.sort(Comparator
                .comparingInt((Integer idx) -> preferredTacticPriority(actionType, candidateTexts.get(idx), snapshot))
                .thenComparing((Integer a, Integer b) -> Float.compare(probAt(probs, b), probAt(probs, a)))
                .thenComparingInt(Integer::intValue));
        return preferred;
    }

    private static int branchOrderPriority(
            StateSequenceBuilder.ActionType actionType,
            String candidateText,
            StateSnapshot snapshot
    ) {
        int preferred = preferredTacticPriority(actionType, candidateText, snapshot);
        if (preferred < Integer.MAX_VALUE) {
            return preferred;
        }
        return isPassText(candidateText) ? 100_000 : 50_000;
    }

    private static float probAt(float[] probs, int idx) {
        if (probs == null || idx < 0 || idx >= probs.length) {
            return 0.0f;
        }
        float p = probs[idx];
        return Float.isNaN(p) || Float.isInfinite(p) ? 0.0f : p;
    }

    private static float[] uniformScores(int candidateCount) {
        float[] out = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
        int count = Math.max(0, Math.min(candidateCount, out.length));
        float p = count <= 0 ? 0.0f : 1.0f / count;
        for (int i = 0; i < count; i++) {
            out[i] = p;
        }
        return out;
    }

    /** Public accessor: canonical action text for a single candidate, matching the
     *  exact string the autopilot matches against (describeCandidate + cleanCandidateText).
     *  Used by Go-Explore trace capture / prefix replay. */
    public static String describeOne(Object candidate, Game game) {
        return cleanCandidateText(describeCandidate(candidate, game));
    }

    private static <T> List<String> describeCandidates(List<T> candidates, Game game, int limit) {
        List<String> out = new ArrayList<>();
        int count = Math.min(limit, candidates == null ? 0 : candidates.size());
        for (int i = 0; i < count; i++) {
            out.add(cleanCandidateText(describeCandidate(candidates.get(i), game)));
        }
        return out;
    }

    private static String cleanCandidateText(String text) {
        String out = text == null ? "" : text.replace('\r', ' ').replace('\n', ' ').trim();
        return out.length() > 300 ? out.substring(0, 300) : out;
    }

    private static String describeCandidate(Object candidate, Game game) {
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
        if (candidate instanceof Ability) {
            Ability ability = (Ability) candidate;
            MageObject source = game == null ? null : game.getObject(ability.getSourceId());
            if (source == null && game != null) {
                source = game.getCard(ability.getSourceId());
            }
            String sourceName = source == null ? "Ability" : source.getName();
            return sourceName + ": " + ability.toString();
        }
        return String.valueOf(candidate);
    }

    private static String normalizeText(String text) {
        if (text == null) {
            return "";
        }
        return text.toLowerCase(Locale.ROOT).replace('\r', ' ').replace('\n', ' ').trim();
    }

    private static boolean textMatches(String expected, String actual) {
        String e = normalizeText(expected);
        String a = normalizeText(actual);
        return Objects.equals(e, a);
    }

    private static boolean isPassText(String candidateText) {
        String text = normalizeText(candidateText);
        return text.equals("pass")
                || text.endsWith(": pass")
                || text.contains("pass priority")
                || text.contains("passability");
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
        if (text.contains("saruli caretaker")) return 12;
        if (text.contains("tinder wall")) return 13;
        if (text.contains("wall of roots")) return 14;
        if (text.contains("overgrown battlement")) return 15;
        if (text.contains("gatecreeper vine")) return 16;
        if (text.contains("elves of deep shadow")) return 17;
        if (text.contains("quirion ranger")) return 18;
        if (text.contains("mesmeric fiend")) return 19;
        if (text.contains("masked vandal")) return 20;
        return 21;
    }

    private static boolean shouldPreferSpyTriggerPass(String state) {
        return state != null
                && !state.isEmpty()
                && stateField(state, "battlefield").contains("Balustrade Spy")
                && state.contains("graveyardHasDread=false")
                && !state.contains("librarySize=0");
    }

    private static boolean isSelfTargetText(String normalizedText) {
        return normalizedText != null
                && (normalizedText.equals("player:acf-prefix")
                || normalizedText.equals("acf-prefix")
                || normalizedText.contains("player:acf-prefix")
                || normalizedText.contains("player:rl")
                || normalizedText.contains("player:spy")
                || normalizedText.contains("online-prefix"));
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

    private static String exceptionSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String s = t.getClass().getSimpleName() + ":" + String.valueOf(t.getMessage());
        return s.length() > 180 ? s.substring(0, 180) : s;
    }

    private static int safeTurn(Game game) {
        try {
            return game == null ? -1 : game.getTurnNum();
        } catch (Throwable ignored) {
            return -1;
        }
    }

    private static final class PrefixNode {
        final int rootIndex;
        final List<String> prefixTexts;

        PrefixNode(int rootIndex, List<String> prefixTexts) {
            this.rootIndex = rootIndex;
            this.prefixTexts = prefixTexts == null ? Collections.emptyList() : new ArrayList<>(prefixTexts);
        }

        String key() {
            List<String> normalized = new ArrayList<>();
            for (String text : prefixTexts) {
                normalized.add(normalizeText(text));
            }
            return rootIndex + ":" + String.join("\u0001", normalized);
        }
    }

    private static final class RunResult {
        final boolean won;
        final boolean timedOut;
        final String reason;
        final DecisionPoint branchPoint;
        final List<String> executedTexts;

        RunResult(boolean won, boolean timedOut, String reason, DecisionPoint branchPoint, List<String> executedTexts) {
            this.won = won;
            this.timedOut = timedOut;
            this.reason = reason == null ? "" : reason;
            this.branchPoint = branchPoint;
            this.executedTexts = Collections.unmodifiableList(
                    executedTexts == null ? Collections.emptyList() : new ArrayList<>(executedTexts));
        }

        static RunResult failed(String reason, boolean timedOut) {
            return new RunResult(false, timedOut, reason, null, Collections.emptyList());
        }
    }

    private static final class DecisionPoint {
        final StateSequenceBuilder.ActionType actionType;
        final List<String> candidateTexts;
        float[] probs;
        final StateSnapshot snapshot;
        int chosenIndex;

        DecisionPoint(StateSequenceBuilder.ActionType actionType, List<String> candidateTexts,
                      float[] probs, StateSnapshot snapshot, int chosenIndex) {
            this.actionType = actionType;
            this.candidateTexts = candidateTexts == null ? Collections.emptyList() : new ArrayList<>(candidateTexts);
            this.probs = probs == null ? new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES]
                    : Arrays.copyOf(probs, probs.length);
            this.snapshot = snapshot == null ? StateSnapshot.EMPTY : snapshot;
            this.chosenIndex = chosenIndex;
        }
    }

    private static final class SearchTerminated extends Error {
        SearchTerminated(String reason) {
            super(reason, null, false, false);
        }
    }

    private static final class InlinePrefixController implements MCTSSimPlayer.MCTSInlineController {
        private final Game sim;
        private final UUID selfId;
        private final List<String> prefixTexts;
        private final int maxGameTurns;
        private final long deadlineNanos;
        private final boolean genericBranchOrder;
        private int prefixApplied = 0;
        private boolean invalidPrefix = false;
        private boolean timedOut = false;
        private String terminatedReason = "";
        private DecisionPoint firstBranchPoint = null;

        InlinePrefixController(Game sim, UUID selfId, List<String> prefixTexts, Config cfg, long deadlineNanos) {
            this.sim = sim;
            this.selfId = selfId;
            this.prefixTexts = prefixTexts == null ? Collections.emptyList() : new ArrayList<>(prefixTexts);
            this.maxGameTurns = cfg == null ? 0 : Math.max(0, cfg.maxGameTurns);
            this.genericBranchOrder = cfg != null && cfg.genericBranchOrder;
            this.deadlineNanos = deadlineNanos;
        }

        @Override
        public int handleRequest(MCTSSimPlayer.DecisionRequest req) {
            checkStop();
            if (req == null || req.options == null || req.options.isEmpty()) {
                return 0;
            }
            int candidateCount = Math.min(req.options.size(), StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            StateSequenceBuilder.ActionType actionType = actionType(req.choiceType);
            List<String> texts = describeCandidates(req.options, sim, candidateCount);
            StateSnapshot snapshot = StateSnapshot.capture(sim, sim.getPlayer(selfId));

            if (prefixApplied < prefixTexts.size()) {
                String expected = prefixTexts.get(prefixApplied);
                for (int i = 0; i < texts.size(); i++) {
                    if (textMatches(expected, texts.get(i))) {
                        prefixApplied++;
                        return i;
                    }
                }
                invalidPrefix = true;
                terminatedReason = "prefix_not_found";
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }

            DecisionPoint pending = null;
            if (firstBranchPoint == null && candidateCount >= 2) {
                pending = new DecisionPoint(actionType, texts, uniformScores(candidateCount), snapshot, -1);
            }

            int choice = chooseSelf(actionType, texts, snapshot, candidateCount, genericBranchOrder);
            if (pending != null) {
                pending.chosenIndex = choice;
                firstBranchPoint = pending;
            }
            return Math.max(0, Math.min(choice, candidateCount - 1));
        }

        int prefixAppliedCount() {
            return prefixApplied;
        }

        boolean invalidPrefix() {
            return invalidPrefix;
        }

        boolean timedOut() {
            return timedOut;
        }

        String terminatedReason() {
            return terminatedReason == null ? "" : terminatedReason;
        }

        DecisionPoint firstBranchPoint() {
            return firstBranchPoint;
        }

        private void checkStop() {
            if (System.nanoTime() >= deadlineNanos) {
                timedOut = true;
                terminatedReason = "deadline";
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }
            if (maxGameTurns > 0 && safeTurn(sim) > maxGameTurns) {
                terminatedReason = "turn_cap";
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }
        }

        private static StateSequenceBuilder.ActionType actionType(MCTSSimPlayer.ChoiceType choiceType) {
            return choiceType == MCTSSimPlayer.ChoiceType.SELECT_TARGET
                    ? StateSequenceBuilder.ActionType.SELECT_TARGETS
                    : StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL;
        }

        private static int chooseSelf(
                StateSequenceBuilder.ActionType actionType,
                List<String> texts,
                StateSnapshot snapshot,
                int candidateCount,
                boolean genericBranchOrder
        ) {
            List<Integer> ordered = branchCandidates(
                    actionType,
                    texts,
                    uniformScores(candidateCount),
                    -1,
                    snapshot,
                    1,
                    genericBranchOrder);
            return ordered.isEmpty() ? 0 : ordered.get(0);
        }
    }

    private static final class SearchPlayer extends ComputerPlayerRL {
        private final List<String> prefixTexts;
        private final int maxGameTurns;
        private final long deadlineNanos;
        private final boolean modelGuidedFallback;
        private final boolean genericBranchOrder;
        private final List<String> executedTexts = new ArrayList<>();
        private volatile boolean halted = false;
        private int prefixApplied = 0;
        private boolean invalidPrefix = false;
        private DecisionPoint firstBranchPoint = null;

        SearchPlayer(ComputerPlayerRL original, List<String> prefixTexts, Config cfg, long deadlineNanos) {
            super(original);
            this.prefixTexts = prefixTexts == null ? Collections.emptyList() : new ArrayList<>(prefixTexts);
            this.maxGameTurns = cfg == null ? 0 : Math.max(0, cfg.maxGameTurns);
            this.modelGuidedFallback = cfg != null && cfg.modelGuidedFallback;
            this.genericBranchOrder = cfg != null && cfg.genericBranchOrder;
            this.deadlineNanos = deadlineNanos;
            setFastFailInTestMode(false);
            setTestMode(false);
        }

        void halt() {
            halted = true;
        }

        int prefixAppliedCount() {
            return prefixApplied;
        }

        boolean invalidPrefix() {
            return invalidPrefix;
        }

        DecisionPoint firstBranchPoint() {
            return firstBranchPoint;
        }

        List<String> executedTrace() {
            return new ArrayList<>(executedTexts);
        }

        @Override
        public boolean priority(Game game) {
            if (game != null) {
                game.resumeTimer(getTurnControlledBy());
            }
            checkStop(game);
            return priorityPlay(game);
        }

        @Override
        public <T> List<Integer> genericChoose(
                List<T> candidates,
                int maxTargets,
                int minTargets,
                StateSequenceBuilder.ActionType actionType,
                Game game,
                Ability source
        ) {
            checkStop(game);
            int candidateCount = Math.min(candidates == null ? 0 : candidates.size(),
                    StateSequenceBuilder.TrainingData.MAX_CANDIDATES);
            if (candidateCount <= 1 || !isSearchActionType(actionType)) {
                return super.genericChoose(candidates, maxTargets, minTargets, actionType, game, source);
            }

            List<String> texts = describeCandidates(candidates, game, candidateCount);
            StateSnapshot snapshot = StateSnapshot.capture(game, this);

            if (prefixApplied < prefixTexts.size()) {
                String expected = prefixTexts.get(prefixApplied);
                for (int i = 0; i < texts.size(); i++) {
                    if (textMatches(expected, texts.get(i))) {
                        prefixApplied++;
                        executedTexts.add(texts.get(i));
                        return Collections.singletonList(i);
                    }
                }
                invalidPrefix = true;
                throw new SearchTerminated("prefix_not_found");
            }

            DecisionPoint pending = null;
            if (firstBranchPoint == null) {
                pending = new DecisionPoint(actionType, texts, uniformScores(candidateCount), snapshot, -1);
            }

            List<Integer> tactic = genericBranchOrder
                    ? null
                    : tacticChoice(actionType, texts, snapshot, maxTargets, minTargets);
            List<Integer> chosen;
            float[] branchProbs = uniformScores(candidateCount);
            if (tactic != null && !tactic.isEmpty()) {
                chosen = tactic;
            } else if (modelGuidedFallback) {
                chosen = super.genericChoose(candidates, maxTargets, minTargets, actionType, game, source);
                float[] lastProbs = getLastActionProbsSnapshot();
                if (lastProbs != null && lastProbs.length > 0) {
                    branchProbs = lastProbs;
                }
            } else {
                chosen = defaultChoice(actionType, texts, snapshot, candidateCount, maxTargets, minTargets,
                        genericBranchOrder);
            }
            if (chosen != null && !chosen.isEmpty()) {
                int idx = chosen.get(0);
                if (idx >= 0 && idx < texts.size()) {
                    executedTexts.add(texts.get(idx));
                }
            }
            if (pending != null) {
                pending.probs = Arrays.copyOf(branchProbs, branchProbs.length);
                pending.chosenIndex = chosen == null || chosen.isEmpty() ? -1 : chosen.get(0);
                firstBranchPoint = pending;
            }
            return chosen;
        }

        private void checkStop(Game game) {
            if (halted || System.nanoTime() >= deadlineNanos) {
                throw new SearchTerminated("deadline");
            }
            if (maxGameTurns > 0 && safeTurn(game) > maxGameTurns) {
                throw new SearchTerminated("turn_cap");
            }
        }

        private static boolean isSearchActionType(StateSequenceBuilder.ActionType actionType) {
            return isControllableActionType(actionType);
        }

        private static float[] uniformScores(int candidateCount) {
            float[] out = new float[StateSequenceBuilder.TrainingData.MAX_CANDIDATES];
            int count = Math.max(0, Math.min(candidateCount, out.length));
            float p = count <= 0 ? 0.0f : 1.0f / count;
            for (int i = 0; i < count; i++) {
                out[i] = p;
            }
            return out;
        }

        private static List<Integer> defaultChoice(
                StateSequenceBuilder.ActionType actionType,
                List<String> texts,
                StateSnapshot snapshot,
                int candidateCount,
                int maxTargets,
                int minTargets,
                boolean genericBranchOrder
        ) {
            List<Integer> ordered = branchCandidates(
                    actionType,
                    texts,
                    uniformScores(candidateCount),
                    -1,
                    snapshot,
                    Math.max(1, candidateCount),
                    genericBranchOrder);
            int wanted = Math.max(1, Math.max(0, minTargets));
            if (maxTargets > 0) {
                wanted = Math.min(wanted, maxTargets);
            }
            List<Integer> out = new ArrayList<>();
            for (Integer idx : ordered) {
                if (idx == null || idx < 0 || idx >= candidateCount || out.contains(idx)) {
                    continue;
                }
                out.add(idx);
                if (out.size() >= wanted) {
                    break;
                }
            }
            if (out.isEmpty() && candidateCount > 0) {
                out.add(0);
            }
            return out;
        }

        private static List<Integer> tacticChoice(
                StateSequenceBuilder.ActionType actionType,
                List<String> texts,
                StateSnapshot snapshot,
                int maxTargets,
                int minTargets
        ) {
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
    }

    private static final class StateSnapshot {
        static final StateSnapshot EMPTY = new StateSnapshot("");

        private final String compactText;

        private StateSnapshot(String compactText) {
            this.compactText = compactText == null ? "" : compactText;
        }

        static StateSnapshot capture(Game game, Player player) {
            if (game == null || player == null) {
                return EMPTY;
            }
            try {
                StringBuilder sb = new StringBuilder();
                sb.append("turn=").append(safeTurn(game));
                sb.append(";phase=").append(game.getTurnPhaseType());
                sb.append(";step=").append(game.getTurnStepType());
                sb.append(";active=").append(playerName(game, game.getActivePlayerId()));
                sb.append(";priority=").append(playerName(game, game.getPriorityPlayerId()));
                sb.append(";life=").append(player.getLife());
                sb.append(";opponentLife=").append(opponentLife(game, player.getId()));
                sb.append(";lands=").append(player.getLandsPlayed()).append('/').append(player.getLandsPerTurn());
                sb.append(";canPlayLand=").append(player.canPlayLand());
                sb.append(";mana=").append(manaText(player));
                sb.append(";hand=").append(cardNames(player.getHand().getCards(game), 12));
                sb.append(";battlefield=").append(permanentNames(game, player.getId(), 16));
                sb.append(";stack=").append(stackNames(game, player.getId(), 8));
                sb.append(";battlefieldCreatures=").append(controlledCreatureCount(game, player.getId()));
                sb.append(";graveyard=").append(cardNames(player.getGraveyard().getCards(game), 20));
                sb.append(";graveyardCreatures=").append(graveyardCreatureCount(player, game));
                sb.append(";graveyardHasDread=").append(graveyardContains(player, game, "Dread Return"));
                sb.append(";graveyardHasLotleth=").append(graveyardContains(player, game, "Lotleth Giant"));
                List<Card> library = player.getLibrary().getCards(game);
                sb.append(";librarySize=").append(library.size());
                sb.append(";libraryTrueLands=").append(countLands(library, game));
                sb.append(";libraryLandGrants=").append(countNamed(library, "Land Grant"));
                sb.append(";handLandGrants=").append(countNamed(player.getHand().getCards(game), "Land Grant"));
                sb.append(";libraryTop=").append(cardNames(library, 12));
                return new StateSnapshot(sb.toString());
            } catch (Exception e) {
                return new StateSnapshot("snapshot_error=" + exceptionSummary(e));
            }
        }

        String toCompactText() {
            return compactText;
        }

        private static String playerName(Game game, UUID playerId) {
            try {
                Player p = playerId == null ? null : game.getPlayer(playerId);
                return p == null ? "" : p.getName();
            } catch (Exception ignored) {
                return "";
            }
        }

        private static int opponentLife(Game game, UUID playerId) {
            try {
                for (Player p : game.getState().getPlayers().values()) {
                    if (p != null && !p.getId().equals(playerId)) {
                        return p.getLife();
                    }
                }
            } catch (Exception ignored) {
            }
            return 20;
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

        private static String stackNames(Game game, UUID playerId, int limit) {
            try {
                List<String> names = new ArrayList<>();
                for (StackObject object : game.getStack()) {
                    if (object != null && object.getControllerId() != null && object.getControllerId().equals(playerId)) {
                        names.add(object.getName());
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

        private static String cardNames(Collection<Card> cards, int limit) {
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

        private static boolean nameEquals(String left, String right) {
            return left != null && right != null && left.equalsIgnoreCase(right);
        }
    }
}
