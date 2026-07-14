package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.Ability;
import mage.counters.CounterType;
import mage.cards.Card;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.ComputerPlayerRL;
import mage.players.Player;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

/**
 * Branch differential oracle wrapper (external reviewer protocol, Sol #89/#91).
 *
 * Env-gated (RL_BRANCH_ORACLE=1), test-only, and never touches production
 * decision paths: it replays a burn_mirror game from scratch under the exact
 * same deterministic recipe that generated the corpus, forces exactly one
 * *unchosen* alternate action at a single pinned decision point via the
 * existing {@link EngineDecisionBranchController} hook (this class does not
 * add a new hook mechanism, generate candidates, or change any candidate
 * generation code -- it only forces a choice among candidates the engine
 * already produced), then emits full canonical state
 * ({@code BRANCH_ORACLE_JSON}) at the next rules decision and, optionally,
 * a fixed number of further decisions under a deterministic (non-random,
 * "seeded" in the sense of reproducible-by-construction) continuation
 * policy. There is no new H(arness) version: this is a replay-and-force
 * wrapper around the same decision surface the corpus already exercises.
 */
public final class BranchOracle {

    public static final int BRANCH_ORACLE_VERSION = 1;

    public static final boolean ENABLED = EnvConfig.bool("RL_BRANCH_ORACLE", false);
    public static final int TARGET_EPISODE = EnvConfig.i32("RL_BRANCH_ORACLE_TARGET_EPISODE", -1);

    /**
     * Provenance: the reference engine's own commit, for the same reason
     * the burn_mirror corpus's manifest.json records `java_oracle_commit`
     * -- an external reviewer needs to know exactly which Java source this
     * run's BRANCH_ORACLE_JSON output was produced by. Prefers an
     * explicitly supplied value (the orchestration script's own call to
     * `git rev-parse HEAD`, avoiding a per-JVM subprocess); falls back to
     * asking git directly, once, if not supplied. Never throws; an empty
     * string means "could not be determined", not "unknown commit".
     */
    private static final String JAVA_ORACLE_COMMIT = resolveJavaOracleCommit();

    private static String resolveJavaOracleCommit() {
        String fromEnv = EnvConfig.str("RL_BRANCH_ORACLE_JAVA_COMMIT", "");
        if (!fromEnv.isEmpty()) {
            return fromEnv;
        }
        try {
            Process p = new ProcessBuilder("git", "rev-parse", "HEAD").redirectErrorStream(true).start();
            try (java.io.BufferedReader reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream(), java.nio.charset.StandardCharsets.UTF_8))) {
                String line = reader.readLine();
                p.waitFor();
                return (line == null || p.exitValue() != 0) ? "" : line.trim();
            }
        } catch (Throwable ignored) {
            return "";
        }
    }

    private BranchOracle() {
    }

    /**
     * Installs one shared controller on both seats for the single targeted
     * episode. Called from RLTrainer's per-episode setup; a no-op unless
     * {@link #ENABLED} and the episode number matches {@link #TARGET_EPISODE}.
     */
    public static void maybeInstall(Player rlPlayer, Player opponentPlayer) {
        if (!ENABLED) {
            return;
        }
        Controller controller = new Controller();
        if (rlPlayer instanceof ComputerPlayerRL) {
            ((ComputerPlayerRL) rlPlayer).setEngineDecisionBranchController(controller);
        }
        if (opponentPlayer instanceof ComputerPlayerRL) {
            ((ComputerPlayerRL) opponentPlayer).setEngineDecisionBranchController(controller);
        }
    }

    /**
     * Removes the controller installed by {@link #maybeInstall}. The caller
     * must invoke this from a {@code finally} block so a branch controller
     * is never left attached to a (possibly pooled/reused) player instance
     * after an exception -- a stale controller would silently intercept and
     * force decisions in a later, unrelated episode.
     */
    public static void uninstall(Player rlPlayer, Player opponentPlayer) {
        if (!ENABLED) {
            return;
        }
        if (rlPlayer instanceof ComputerPlayerRL) {
            ((ComputerPlayerRL) rlPlayer).setEngineDecisionBranchController(null);
        }
        if (opponentPlayer instanceof ComputerPlayerRL) {
            ((ComputerPlayerRL) opponentPlayer).setEngineDecisionBranchController(null);
        }
    }

    private static final class Config {
        final String targetPlayer = EnvConfig.str("RL_BRANCH_ORACLE_TARGET_PLAYER", "");
        // Sol #101 upgrade: record_id (GameLogger's genuinely-monotonic
        // per-game counter, unique across BOTH players and every decision
        // type -- the same join key used throughout the Sol #100 checkpoint
        // campaign) replaces the old per-player forced-call-index addressing.
        // The prior scheme required the branch-point selector to reproduce
        // this controller's exact call-counting rules (mulligan exclusion,
        // isTraceVisible whitelist) by hand; record_id is read directly off
        // the live GameLogger via peekNextRecordId(), so there is nothing to
        // keep in sync -- the same trace-derived record_id used to select a
        // combat point from burn_mirror_v5's REPLAY_DECISION_JSON records is
        // used verbatim here, matching build_trace_suffix_specs.py's
        // record_id-joined discipline applied from ordinal 0.
        final int targetRecordId = EnvConfig.i32("RL_BRANCH_ORACLE_TARGET_RECORD_ID", -1);
        final String targetActionType = EnvConfig.str("RL_BRANCH_ORACLE_TARGET_ACTION_TYPE", "");
        final int targetCandidateCount = EnvConfig.i32("RL_BRANCH_ORACLE_TARGET_CANDIDATE_COUNT", -1);
        final int altIndex = EnvConfig.i32("RL_BRANCH_ORACLE_ALT_INDEX", -1);
        final int continueSteps = Math.max(0, EnvConfig.i32("RL_BRANCH_ORACLE_CONTINUE_STEPS", 0));
        // Sol #101 item (b), the CONTROL check: force the branch point's
        // OWN historically-recorded index (altIndex == the original chosen
        // index, not a real alternate) and run every subsequent decision via
        // the shared-semantic-policy continuation all the way to
        // game.hasEnded(), instead of stopping after continueSteps -- proves
        // the from-scratch replay + forced-choice + continuation machinery
        // itself is faithful (reproduces the recorded winner) before trusting
        // it to evaluate a real alternate.
        final boolean runToTerminal = EnvConfig.bool("RL_BRANCH_ORACLE_RUN_TO_TERMINAL", false);
        final int terminalStepCap = Math.max(1, EnvConfig.i32("RL_BRANCH_ORACLE_TERMINAL_STEP_CAP", 400));
        final String branchId = EnvConfig.str("RL_BRANCH_ORACLE_BRANCH_ID", "branch");
    }

    /**
     * One instance is shared by both seats so it observes the whole game's
     * decision sequence in game-wide order, not just one player's.
     */
    static final class Controller implements EngineDecisionBranchController {
        private final Config cfg = new Config();
        private volatile boolean branched = false;
        private volatile int stepsEmitted = 0;
        private volatile boolean done = false;
        // Provenance (Sol #89/#91 amendment): the state hash at the moment
        // of forcing (known from BRANCHED onward) and the state hash
        // immediately after the forced action resolved (known from
        // POST_BRANCH_1 onward). Fixed once known and repeated on every
        // later record in this branch episode, so any single line is a
        // self-contained audit anchor back to both endpoints.
        private volatile String preForceStateHash = "";
        private volatile String postForceStateHash = "";

        @Override
        public boolean shouldEvaluateBeforeModel(StateSequenceBuilder.ActionType actionType) {
            // Root-caused during pilot validation: several call sites are
            // "if preModelForcedChoiceIndices() returned nothing, THEN score
            // normally and call the plain forcedChoiceIndices() as a second,
            // final check" (an if/else, e.g. ComputerPlayerRL.java's
            // CHOOSE_MODE handling ~line 8020). Returning true here made
            // onDecision fire TWICE per real decision whenever this
            // controller had not yet forced anything (once via the
            // pre-model path, again via the post-scoring path) -- silently
            // doubling this controller's own call count relative to the
            // trace's real decision count, and the drift compounded with
            // every decision before the target. false relies on the single,
            // unconditional forcedChoiceIndices() call every decision type
            // already makes (confirmed: every action type this pilot
            // targets reaches it), which is exactly one call per real
            // decision -- and forcing still works identically through that
            // same call.
            return false;
        }

        @Override
        public boolean shouldBypassModelInference() {
            return false;
        }

        @Override
        public <T> Choice onDecision(DecisionContext<T> context) {
            if (done) {
                return Choice.none();
            }
            // Internal legality/lookahead probes clone the game and replay
            // candidate construction on the clone (alt-cost testing,
            // playable checks, etc); those clones still reach this hook,
            // but never reach logReplayDecision (guarded there by the same
            // check), so they are invisible to the corpus trace and to the
            // branch-point selector's per-player position count. Counting
            // them here would silently inflate this controller's count
            // past what the trace (and the kernel's own per-player cursor)
            // expects -- exactly the kind of drift the branch-point
            // selection script's forced_call_index assumes cannot happen.
            if (context.game != null && context.game.isSimulation()) {
                return Choice.none();
            }
            // Root-caused during pilot validation: several decision kinds
            // reach this hook exactly like any other (ChooseOptionalCost's
            // CHOOSE_USE prompt -- Highway Robbery's "discard a card or
            // sacrifice a land?" offer; ChooseCastMode's CHOOSE_MODE
            // prompt; likely others) but the golden-trace corpus records
            // only 5 action types plus one narrow CHOOSE_USE shape (the
            // Madness yes/no offer, candidates exactly ["Yes","No"]; see
            // trace.rs's MADNESS_CHOOSE_USE_MARKER doc) -- confirmed
            // empirically against the full v4 corpus (grep across all 40
            // files). Whitelisting what the trace can possibly contain,
            // rather than blacklisting each discovered silent decision
            // kind one at a time, is the closed, not reopen-prone form of
            // this fix.
            if (!isTraceVisible(context)) {
                return Choice.none();
            }
            String playerName = context.player == null ? "" : context.player.getName();

            if (!branched) {
                GameLogger logger = RLTrainer.threadLocalGameLogger.get();
                // Fail-closed: without a live logger this controller cannot know
                // which record_id the current decision is about to receive, so
                // it must never guess by falling through to some other check.
                if (logger == null) {
                    return Choice.none();
                }
                int nextRecordId = logger.peekNextRecordId();
                if (nextRecordId != cfg.targetRecordId) {
                    return Choice.none();
                }
                // Fail-closed alignment checks: a silent mismatch here would force
                // the wrong decision and corrupt every downstream comparison.
                if (!cfg.targetPlayer.isEmpty() && !cfg.targetPlayer.equals(playerName)) {
                    String reason = "branch_oracle_alignment_error";
                    logJson(record(context, "ALIGNMENT_ERROR",
                            "player_mismatch:expected=" + cfg.targetPlayer + ":actual=" + playerName, reason));
                    done = true;
                    return Choice.chooseAndTerminate(Collections.emptyList(), reason);
                }
                String actual = context.actionType == null ? "" : context.actionType.name();
                if (!cfg.targetActionType.isEmpty() && !cfg.targetActionType.equals(actual)) {
                    String reason = "branch_oracle_alignment_error";
                    logJson(record(context, "ALIGNMENT_ERROR",
                            "action_type_mismatch:expected=" + cfg.targetActionType + ":actual=" + actual, reason));
                    done = true;
                    return Choice.chooseAndTerminate(Collections.emptyList(), reason);
                }
                if (cfg.targetCandidateCount >= 0 && context.candidateCount != cfg.targetCandidateCount) {
                    String reason = "branch_oracle_alignment_error";
                    logJson(record(context, "ALIGNMENT_ERROR",
                            "candidate_count_mismatch:expected=" + cfg.targetCandidateCount
                                    + ":actual=" + context.candidateCount, reason));
                    done = true;
                    return Choice.chooseAndTerminate(Collections.emptyList(), reason);
                }
                if (cfg.altIndex < 0 || cfg.altIndex >= context.candidateCount) {
                    String reason = "branch_oracle_alignment_error";
                    logJson(record(context, "ALIGNMENT_ERROR",
                            "alt_index_out_of_range:altIndex=" + cfg.altIndex
                                    + ":candidateCount=" + context.candidateCount, reason));
                    done = true;
                    return Choice.chooseAndTerminate(Collections.emptyList(), reason);
                }
                branched = true;
                preForceStateHash = LiveCheckpointRecorder.sha256(canonicalState(context.game));
                logJson(record(context, "BRANCHED", "", "", Collections.singletonList(cfg.altIndex)));
                return Choice.choose(Collections.singletonList(cfg.altIndex));
            }

            // Already branched: every subsequent onDecision call (either seat) is
            // the next rules decision in game-wide order.
            stepsEmitted++;
            if (postForceStateHash.isEmpty()) {
                postForceStateHash = LiveCheckpointRecorder.sha256(canonicalState(context.game));
            }
            boolean gameOver = context.game != null && context.game.hasEnded();
            boolean stepCapHit = cfg.runToTerminal
                    ? stepsEmitted > cfg.terminalStepCap
                    : stepsEmitted > cfg.continueSteps;
            if (gameOver || stepCapHit) {
                String reason = gameOver ? "branch_oracle_terminal_reached" : "branch_oracle_capture_complete";
                logJson(record(context, "POST_BRANCH_" + stepsEmitted, gameOver ? terminalSummary(context.game) : "", reason));
                done = true;
                return Choice.chooseAndTerminate(Collections.emptyList(), reason);
            }
            // Compute the continuation BEFORE logging, so this step's record
            // carries the text it is actually about to force -- not a value
            // derived after the fact, which would drift if the policy or the
            // engine's own candidate order ever changed between the two calls.
            List<Integer> continuation = sharedSemanticPolicyIndex(context);
            if (continuation.isEmpty()) {
                logJson(record(context, "POST_BRANCH_" + stepsEmitted, "no_legal_continuation", "branch_oracle_no_continuation"));
                done = true;
                return Choice.chooseAndTerminate(Collections.emptyList(), "branch_oracle_no_continuation");
            }
            logJson(record(context, "POST_BRANCH_" + stepsEmitted, "", "", continuation));
            return Choice.choose(continuation);
        }

        /**
         * Winner/loser summary for the CONTROL check (Sol #101 item b): must
         * match the recorded corpus game's outcome exactly for the
         * from-scratch replay + forced-choice + continuation path to be
         * trusted for evaluating a real alternate at this same point.
         */
        private String terminalSummary(Game game) {
            if (game == null) {
                return "";
            }
            StringBuilder sb = new StringBuilder(128);
            sb.append("terminal:");
            for (UUID id : game.getPlayerList()) {
                Player p = game.getPlayer(id);
                if (p != null) {
                    sb.append(p.getName()).append("(hasLost=").append(p.hasLost())
                            .append(",life=").append(p.getLife()).append(")");
                }
            }
            return sb.toString();
        }

        /**
         * True only for the decision kinds the burn_mirror_v4 corpus can
         * possibly contain a record for: the 5 real REPLAY_DECISION_JSON
         * action types, plus the one narrow CHOOSE_USE shape trace.rs
         * synthesizes (Madness yes/no, candidates exactly ["Yes","No"]).
         * Everything else this hook ever sees (ChooseOptionalCost's
         * CHOOSE_USE, ChooseCastMode's CHOOSE_MODE, and any other kind
         * with no trace counterpart) must be skipped without counting,
         * or this controller's call count silently drifts ahead of the
         * trace-derived forced_call_index it is supposed to match.
         */
        private <T> boolean isTraceVisible(DecisionContext<T> context) {
            String actionType = context.actionType == null ? "" : context.actionType.name();
            switch (actionType) {
                case "ACTIVATE_ABILITY_OR_SPELL":
                case "SELECT_TARGETS":
                case "SELECT_CARD":
                case "DECLARE_ATTACKS":
                case "DECLARE_BLOCKS":
                    return true;
                case "CHOOSE_USE":
                    List<String> texts = context.candidateTexts;
                    return texts.size() == 2 && "Yes".equals(texts.get(0)) && "No".equals(texts.get(1));
                default:
                    return false;
            }
        }

        /**
         * Sol #101 upgrade / Sol #102: shared semantic policy -- the
         * lexicographically-first canonical candidate text, ties broken by
         * index. Identical to LiveCheckpointBranchMiner's
         * sharedSemanticPolicyIndices (Sol #100 checkpoint campaign) and to
         * the kernel's own shared_semantic_policy_index (walk_diff.rs):
         * deliberately engine-independent, not "first-legal by this
         * engine's native candidate order" (an implementation detail that
         * is not guaranteed comparable across two independent engines).
         * Replaces the pilot's placeholder "always Pass / index 0" rule so
         * the from-scratch combat-strata protocol uses the exact same
         * continuation the rest of the Sol #100/#101 campaign already relies
         * on for boundary-by-boundary comparison against the kernel.
         */
        private <T> List<Integer> sharedSemanticPolicyIndex(DecisionContext<T> context) {
            if (context == null || context.candidateCount <= 0 || context.candidateTexts == null) {
                return Collections.emptyList();
            }
            int best = -1;
            String bestText = null;
            for (int i = 0; i < context.candidateCount && i < context.candidateTexts.size(); i++) {
                String text = context.candidateTexts.get(i);
                if (text == null) {
                    text = "";
                }
                if (bestText == null || text.compareTo(bestText) < 0) {
                    bestText = text;
                    best = i;
                }
            }
            return best < 0 ? Collections.emptyList() : Collections.singletonList(best);
        }

        private <T> String record(DecisionContext<T> context, String marker, String detail, String terminationReason) {
            return record(context, marker, detail, terminationReason, Collections.<Integer>emptyList());
        }

        /**
         * chosenIndices: the index (or indices) THIS step is about to force
         * (root: cfg.altIndex; continuation steps: the shared-semantic-policy
         * pick). Recorded alongside candidate_texts so a comparator can
         * reconstruct chosen_text without re-deriving the policy -- mirrors
         * the exact fix applied earlier this campaign to
         * LiveCheckpointBranchMiner's PreprobeController, where the root
         * step's recorded text was originally taken from the policy's
         * hypothetical pick instead of the actually-forced index.
         */
        private <T> String record(DecisionContext<T> context, String marker, String detail, String terminationReason,
                                   List<Integer> chosenIndices) {
            String stateJson = context.game != null ? canonicalState(context.game) : "";
            List<String> texts = context.candidateTexts;
            List<String> chosenTexts = new ArrayList<>();
            for (Integer i : chosenIndices) {
                if (texts != null && i != null && i >= 0 && i < texts.size()) {
                    chosenTexts.add(texts.get(i));
                }
            }
            StringBuilder sb = new StringBuilder(4096);
            sb.append("{\"schema_version\":\"branch-oracle-v1\"")
                    .append(",\"branch_oracle_version\":").append(BRANCH_ORACLE_VERSION)
                    .append(",\"java_oracle_commit\":").append(jsonStr(JAVA_ORACLE_COMMIT))
                    .append(",\"branch_id\":").append(jsonStr(cfg.branchId))
                    .append(",\"marker\":").append(jsonStr(marker))
                    .append(",\"detail\":").append(jsonStr(detail))
                    .append(",\"termination_reason\":").append(jsonStr(terminationReason))
                    .append(",\"actor\":").append(jsonStr(context.player == null ? "" : context.player.getName()))
                    .append(",\"action_type\":").append(jsonStr(context.actionType == null ? "" : context.actionType.name()))
                    .append(",\"source_name\":").append(jsonStr(sourceNameOf(context.source, context.game)))
                    .append(",\"candidate_count\":").append(context.candidateCount)
                    .append(",\"candidate_texts\":").append(jsonStrList(context.candidateTexts))
                    .append(",\"candidate_object_ids\":").append(jsonStrList(context.candidateObjectIds))
                    .append(",\"target_record_id\":").append(cfg.targetRecordId)
                    .append(",\"forced_alt_index\":").append("BRANCHED".equals(marker) ? cfg.altIndex : -1)
                    .append(",\"chosen_indices\":").append(chosenIndices)
                    .append(",\"chosen_texts\":").append(jsonStrList(chosenTexts))
                    .append(",\"state_hash\":").append(jsonStr(LiveCheckpointRecorder.sha256(stateJson)))
                    .append(",\"pre_force_state_hash\":").append(jsonStr(preForceStateHash))
                    .append(",\"post_force_state_hash\":").append(jsonStr(postForceStateHash));
            if (!stateJson.isEmpty()) {
                sb.append(",").append(stateJson);
            }
            sb.append("}");
            return sb.toString();
        }

        private void logJson(String json) {
            try {
                GameLogger logger = RLTrainer.threadLocalGameLogger.get();
                if (logger != null) {
                    logger.log("BRANCH_ORACLE_JSON: " + json);
                }
            } catch (Throwable ignored) {
                // Diagnostic emission must never affect game execution.
            }
        }
    }

    /**
     * Full two-player canonical state: life, turn, phase, active/priority
     * player, both players' battlefield/graveyard/hand/library (sorted
     * multisets of card names; battlefield entries also carry
     * tapped/summoning-sick/damage/+1+1 counters), and the stack in order.
     * Field names and per-permanent formatting deliberately mirror the
     * kernel's own `canonical_snapshot()` (kernel/mtg-kernel/examples/
     * replay_burn_v2.rs) so the comparator can diff sorted string lists
     * directly without a translation layer.
     */
    static String canonicalState(Game game) {
        StringBuilder sb = new StringBuilder(4096);
        try {
            sb.append("\"turn\":").append(game.getTurnNum())
                    .append(",\"phase\":").append(jsonStr(game.getPhase() != null && game.getPhase().getType() != null
                            ? game.getPhase().getType().toString() : ""))
                    .append(",\"active_player\":").append(jsonStr(nameOf(game, game.getActivePlayerId())))
                    .append(",\"priority_player\":").append(jsonStr(nameOf(game, game.getPriorityPlayerId())));

            List<Player> players = new ArrayList<>();
            for (UUID id : game.getPlayerList()) {
                Player p = game.getPlayer(id);
                if (p != null) {
                    players.add(p);
                }
            }
            players.sort(Comparator.comparing(Player::getName));

            sb.append(",\"players\":[");
            for (int i = 0; i < players.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                appendPlayer(sb, game, players.get(i));
            }
            sb.append("]");

            List<StackObject> stack = new ArrayList<>(game.getStack());
            sb.append(",\"stack\":[");
            for (int i = 0; i < stack.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                StackObject so = stack.get(i);
                sb.append(jsonStr(so.getName() + "(controller=" + nameOf(game, so.getControllerId()) + ")"));
            }
            sb.append("]");
        } catch (Throwable t) {
            sb.append("\"canonical_state_error\":").append(jsonStr(errorSummary(t)));
        }
        return sb.toString();
    }

    private static void appendPlayer(StringBuilder sb, Game game, Player p) {
        List<String> battlefield = new ArrayList<>();
        for (Permanent perm : game.getBattlefield().getAllActivePermanents(p.getId())) {
            if (perm == null) {
                continue;
            }
            // Canonicalized as controlled_since_turn_start (Sol #89/#91
            // amendment), not the creature-only "sick" flag: control tenure
            // matters even for a non-creature permanent that could later
            // become a creature. hasSummoningSickness() is deliberately NOT
            // used here -- it ORs in haste (`!(controlledFromStart ||
            // hasHaste)`), which would make this field mean "can this
            // attack/tap right now" instead of "has this been continuously
            // controlled since the turn started". wasControlledFromStart
            // OfControllerTurn() is the raw, haste-independent flag, tracked
            // identically for every permanent type (PermanentImpl.java's
            // entersBattlefield()/beginningOfTurn() never check creature-
            // ness) -- matching the kernel's own type-agnostic
            // GameObject::summoning_sick storage (see branch_diff.rs's
            // canonical_state_json doc for the kernel side, including the
            // CreateToken bug this pilot found and fixed there).
            battlefield.add(String.format(Locale.US, "%s(tapped=%s,controlled_since_turn_start=%s,dmg=%d,+1/+1=%d)",
                    perm.getName(), perm.isTapped(), perm.wasControlledFromStartOfControllerTurn(), perm.getDamage(),
                    perm.getCounters(game).getCount(CounterType.P1P1)));
        }
        Collections.sort(battlefield);

        List<String> graveyard = cardNames(p.getGraveyard() == null ? Collections.emptyList() : p.getGraveyard().getCards(game));
        List<String> hand = cardNames(p.getHand() == null ? Collections.emptyList() : p.getHand().getCards(game));
        List<String> library = cardNames(p.getLibrary() == null ? Collections.emptyList() : p.getLibrary().getCards(game));

        sb.append("{\"seat\":").append(jsonStr(p.getName()))
                .append(",\"life\":").append(p.getLife())
                .append(",\"has_lost\":").append(p.hasLost())
                .append(",\"battlefield\":").append(jsonStrList(battlefield))
                .append(",\"graveyard\":").append(jsonStrList(graveyard))
                .append(",\"hand\":").append(jsonStrList(hand))
                .append(",\"library_multiset\":").append(jsonStrList(library))
                .append("}");
    }

    private static List<String> cardNames(Iterable<Card> cards) {
        List<String> out = new ArrayList<>();
        if (cards != null) {
            for (Card c : cards) {
                if (c != null) {
                    out.add(c.getName());
                }
            }
        }
        Collections.sort(out);
        return out;
    }

    private static String sourceNameOf(Ability source, Game game) {
        if (source == null || game == null) {
            return "";
        }
        try {
            MageObject obj = source.getSourceObject(game);
            if (obj != null) {
                return obj.getName();
            }
        } catch (Throwable ignored) {
            // fall through
        }
        return "";
    }

    private static String nameOf(Game game, UUID playerId) {
        if (playerId == null) {
            return "";
        }
        Player p = game.getPlayer(playerId);
        return p == null ? "" : p.getName();
    }

    private static String jsonStrList(List<String> values) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        if (values != null) {
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append(jsonStr(values.get(i)));
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private static String jsonStr(String value) {
        String v = value == null ? "" : value;
        StringBuilder sb = new StringBuilder(v.length() + 2);
        sb.append('"');
        for (int i = 0; i < v.length(); i++) {
            char c = v.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append(' ');
                    break;
                case '\r':
                    sb.append(' ');
                    break;
                default:
                    sb.append(c);
            }
        }
        sb.append('"');
        return sb.toString();
    }

    private static String errorSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String message = t.getMessage();
        return t.getClass().getSimpleName() + (message == null || message.isEmpty() ? "" : ": " + message);
    }
}
