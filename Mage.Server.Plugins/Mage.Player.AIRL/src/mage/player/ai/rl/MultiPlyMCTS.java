package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.game.Game;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Multi-ply MCTS with factored decisions.
 *
 * Each "iteration" clones the live game, applies a determinization of opp's
 * hidden info, then walks the coroutine-paused engine prompt-by-prompt,
 * using PUCT selection at each level of the search tree. Leaves are reached
 * when the engine returns to a new ACTIVATE_ABILITY prompt for our player
 * (the compound action we're evaluating has resolved) — at that point we
 * evaluate V(s) with the value net and back up along the traversal path.
 *
 * Branching is per sub-decision: the first child level is one-per-legal-ability;
 * if the chosen ability asks for a target, the second level is one-per-legal-target;
 * etc. This gives MCTS proper visibility into target choice (which was the
 * key missing signal in the old flat PolicyValueMCTS).
 *
 * Opponent decisions during our compound action (e.g. opp has priority to
 * counter our spell) go through opp's MCTSSimPlayer with random choice —
 * rollouts don't model opp's strategic response yet. Search quality comes
 * from our own decisions being value-guided and determinization-averaged.
 */
public final class MultiPlyMCTS {

    private static final float C_PUCT = Float.parseFloat(
            System.getenv().getOrDefault("MCTS_C_PUCT", "1.5"));
    private static final int DEFAULT_ITERATIONS = EnvConfig.i32("MCTS_ITERATIONS", 16);
    /** Phase 3b: run engine inline on main thread via MCTSInlineController
     *  instead of simThread+queues. Defaults to 1 (on). Set MCTS_INLINE=0 to
     *  revert to threaded path if issues emerge. */
    private static final boolean USE_INLINE =
            !"0".equals(System.getenv().getOrDefault("MCTS_INLINE", "1"));
    private static final int DEFAULT_DETERMINIZATIONS = EnvConfig.i32("MCTS_DETERMINIZATIONS", 1);
    private static final long SEARCH_TIMEOUT_MS = EnvConfig.i32("MCTS_TIMEOUT_MS", 100);
    private static final long ITER_TIMEOUT_MS = EnvConfig.i32("MCTS_ITER_TIMEOUT_MS", 75);
    private static final boolean DEEP_POLICY_PRIORS =
            EnvConfig.bool("MCTS_DEEP_POLICY_PRIORS", true);
    /** How many of our consecutive priority-activated actions each rollout walks
     *  through before halting and evaluating. 1 = single compound action (old
     *  behavior). Higher = plan further into the future at the cost of more
     *  engine steps per iteration. 0 (or negative) = UNLIMITED — each iteration
     *  walks until the game ends or the per-iteration deadline fires, letting
     *  complex positions naturally get deeper walks. Opp's responses are
     *  randomized during the walk, so the extra depth is most useful when opp
     *  has few interactions. */
    private static final int MAX_OUR_ACTIONS =
            EnvConfig.i32("MCTS_MAX_OUR_ACTIONS", 1);
    private static final String FINAL_SELECTION =
            EnvConfig.str("MCTS_FINAL_SELECTION", "visits").trim().toLowerCase();

    /** If non-zero, stop iterating once the top root child's visit share
     *  exceeds this threshold (and at least MIN_ITERS iterations have run).
     *  Lets easy positions exit quickly so more budget is left for the
     *  positions that actually need deep search. Typical value: 0.7. */
    private static final float EARLY_STOP_CONCENTRATION =
            Float.parseFloat(System.getenv().getOrDefault("MCTS_EARLY_STOP_CONCENTRATION", "0.0"));
    private static final int EARLY_STOP_MIN_ITERS =
            EnvConfig.i32("MCTS_EARLY_STOP_MIN_ITERS", 4);

    /** When enabled, each MCTS tree node remembers an engine-state clone
     *  captured the first time the walker reached that decision point.
     *  Subsequent iterations that walk through the same node can resume from
     *  the snapshot instead of replaying all actions from root, eliminating
     *  the walk-phase cost of shared-prefix iterations. Unsafe when
     *  MCTS_DETERMINIZATIONS>1 (a snapshot is tied to a specific
     *  determinization). Set MCTS_ENGINE_SNAPSHOTS=1 to enable. */
    private static final boolean ENGINE_SNAPSHOTS =
            "1".equals(System.getenv().getOrDefault("MCTS_ENGINE_SNAPSHOTS", "0"));

    /** Intra-search parallelism: run this many rollouts concurrently per round,
     *  batching their leaf evaluations together so ONNX inference sees K-way
     *  batches instead of 1-way per-iter calls. When >1 and MAX_OUR_ACTIONS=1,
     *  uses the virtual-loss-aware root expansion path. Default 1 (serial,
     *  original behavior). Set via MCTS_PARALLEL_ROLLOUTS env var. */
    private static final int PARALLEL_ROLLOUTS = EnvConfig.i32("MCTS_PARALLEL_ROLLOUTS", 1);
    /** Virtual-loss penalty per pending visit when planning a round. Higher =
     *  stronger pressure to pick distinct children. Standard value -1.0
     *  (pessimistic leaf estimate). */
    private static final float VLOSS_VALUE =
            Float.parseFloat(System.getenv().getOrDefault("MCTS_VLOSS_VALUE", "-1.0"));
    /** Rollout pool is shared across all runners. Default caps at ~2× logical
     *  cores so the walk phase (CPU-bound) doesn't thrash, while still
     *  oversubscribing enough to overlap eval waits (I/O-bound on ONNX TCP).
     *  Walk/eval ratio is roughly 15/95ms, so most workers are sleeping on
     *  I/O; a 2× cap keeps ~30% of threads CPU-active and balances context-
     *  switch overhead. Env MCTS_PARALLEL_POOL_SIZE overrides. */
    private static final int PARALLEL_POOL_SIZE = computePoolSize();
    private static int computePoolSize() {
        int explicit = EnvConfig.i32("MCTS_PARALLEL_POOL_SIZE", 0);
        if (explicit > 0) return explicit;
        int runners = EnvConfig.i32("NUM_GAME_RUNNERS", 16);
        int cores = Runtime.getRuntime().availableProcessors();
        int naturalMax = runners * PARALLEL_ROLLOUTS;
        int cap = cores * 2;
        int pool = Math.max(PARALLEL_ROLLOUTS, Math.min(naturalMax, cap));
        System.out.println("[MCTS] rollout pool size=" + pool
                + " (K=" + PARALLEL_ROLLOUTS + " runners=" + runners
                + " cores=" + cores + " cap=" + cap + ")");
        return pool;
    }
    /** Thread prefix MUST start with AI-SIM-MCTS so ThreadUtils.isRunGameThread
     *  returns true; otherwise engine throws "Wrong code usage: game related
     *  code must run in GAME thread". */
    private static final ExecutorService ROLLOUT_POOL = PARALLEL_ROLLOUTS > 1
            ? Executors.newFixedThreadPool(PARALLEL_POOL_SIZE, new ThreadFactory() {
                private final AtomicInteger c = new AtomicInteger();
                @Override public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "AI-SIM-MCTS-parallel-" + c.incrementAndGet());
                    t.setDaemon(true);
                    return t;
                }
            })
            : null;

    /** One-shot flag that measures Game.copy() cost + size the first time MCTS
     *  runs. Enabled via MCTS_MEASURE_CLONE=1. Feasibility data for the tree-
     *  snapshot optimization (store Game state on each MCTSNode). */
    private static final boolean CLONE_MEASURE_ENABLED =
            "1".equals(System.getenv().getOrDefault("MCTS_MEASURE_CLONE", "0"));
    private static final java.util.concurrent.atomic.AtomicBoolean CLONE_MEASURE_DONE =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    private static void measureGameCopyCostOnce(Game liveGame) {
        final int N = 100;
        Runtime rt = Runtime.getRuntime();
        Game[] holders = new Game[N];
        // Warmup: trigger JIT, force class init
        for (int i = 0; i < 10; i++) holders[i] = liveGame.createSimulationForAI();
        java.util.Arrays.fill(holders, null);

        // Pre-measure: quiesce heap
        for (int i = 0; i < 3; i++) { rt.gc(); try { Thread.sleep(20); } catch (InterruptedException ignored) {} }
        long baseMem = rt.totalMemory() - rt.freeMemory();

        long tStart = System.nanoTime();
        for (int i = 0; i < N; i++) holders[i] = liveGame.createSimulationForAI();
        long tEnd = System.nanoTime();
        long cloneNs = (tEnd - tStart) / N;

        // Measure retained memory after holding N copies (force full GC to cleanup scratch)
        for (int i = 0; i < 3; i++) { rt.gc(); try { Thread.sleep(20); } catch (InterruptedException ignored) {} }
        long postMem = rt.totalMemory() - rt.freeMemory();
        long bytesPerCopy = (postMem - baseMem) / N;

        // Release holders and re-measure floor (sanity)
        java.util.Arrays.fill(holders, null);
        for (int i = 0; i < 3; i++) { rt.gc(); try { Thread.sleep(20); } catch (InterruptedException ignored) {} }
        long floorMem = rt.totalMemory() - rt.freeMemory();

        String msg = "[MCTS_CLONE_MEASURE] n=" + N
                + " avg_us=" + (cloneNs / 1000)
                + " bytes_per_copy=" + bytesPerCopy
                + " (baseHeap=" + (baseMem / 1024 / 1024) + "MB"
                + " retained=" + ((postMem - baseMem) / 1024 / 1024) + "MB"
                + " floor=" + (floorMem / 1024 / 1024) + "MB)";
        System.out.println(msg);
        System.err.println(msg);
    }

    private MultiPlyMCTS() {}

    /** Result bundle mirroring PolicyValueMCTS.SearchResult's shape so it
     *  plugs in at the ComputerPlayerRL call site without changes. */
    public static final class Result {
        public final int bestActionIndex;
        public final int[] aggregateVisits;
        public final float[] aggregateValues; // mean Q per root candidate
        public final int iterationsRun;
        public final long wallMs;
        public final String predictedArchetype;

        Result(int best, int[] visits, float[] values, int iters, long ms, String arch) {
            this.bestActionIndex = best;
            this.aggregateVisits = visits;
            this.aggregateValues = values;
            this.iterationsRun = iters;
            this.wallMs = ms;
            this.predictedArchetype = arch;
        }
    }

    /**
     * Run multi-ply MCTS from the given live game state.
     *
     * @param liveGame       real game state (not modified)
     * @param selfId         our player's UUID
     * @param rootCandidates legal ActivatedAbilities at this outer priority call
     * @param rootPriors     P(a) from the action head, one per root candidate
     * @param model          value net for leaf evaluation
     * @param sampler        determinization sampler for opp's hidden info
     * @return best action index + diagnostics
     */
    public static Result search(
            Game liveGame,
            UUID selfId,
            List<? extends Ability> rootCandidates,
            float[] rootPriors,
            PythonModel model,
            DeterminizationSampler sampler
    ) {
        return search(liveGame, selfId, rootCandidates, rootPriors, model, sampler, null);
    }

    /**
     * Full entry point with optional persistent session for tree reuse across
     * priority calls. When session is non-null and its root is compatible
     * (matches numCandidates), we inherit its accumulated Q/N and continue
     * search from there. When incompatible, the session is reset.
     */
    public static Result search(
            Game liveGame,
            UUID selfId,
            List<? extends Ability> rootCandidates,
            float[] rootPriors,
            PythonModel model,
            DeterminizationSampler sampler,
            MCTSSession session
    ) {
        long started = System.currentTimeMillis();
        int numCandidates = rootCandidates.size();

        MCTSNode root = null;
        if (session != null) {
            // Phase 3: if opp has acted since our last search, the sampled
            // determinizations backing the inherited tree may be wrong.
            // Conservatively drop the tree; a more precise implementation
            // could keep subtrees whose det is still consistent.
            if (session.hasPendingObservations()) {
                session.reset("opp observation invalidates inherited tree");
            }
            MCTSNode inherited = session.getRoot();
            if (inherited != null
                    && inherited.kind() == MCTSNode.Kind.DECISION
                    && inherited.numChildren() == numCandidates
                    && rootRefsMatch(inherited, rootCandidates)) {
                root = inherited;
            } else if (inherited != null) {
                session.reset("root mismatch: width/refs differ ("
                        + inherited.numChildren() + " vs " + numCandidates + ")");
            }
        }

        if (root == null) {
            // Build the root decision node up front. Priors are the action-head
            // logits for our outer priority; child candidates are the abilities.
            root = new MCTSNode();
            Object[] rootRefs = new Object[numCandidates];
            float[] rootPriorsCopy = new float[numCandidates];
            for (int i = 0; i < numCandidates; i++) {
                rootRefs[i] = rootCandidates.get(i);
                rootPriorsCopy[i] = i < rootPriors.length ? rootPriors[i] : 1.0f / numCandidates;
            }
            // Normalize priors (in case input is not a proper distribution)
            float sum = 0f;
            for (float p : rootPriorsCopy) sum += p;
            if (sum > 1e-6f) {
                for (int i = 0; i < numCandidates; i++) rootPriorsCopy[i] /= sum;
            } else {
                for (int i = 0; i < numCandidates; i++) rootPriorsCopy[i] = 1.0f / numCandidates;
            }
            root.initializeAsDecision(MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY,
                    rootRefs, rootPriorsCopy);
            if (session != null) session.setRoot(root);
        }

        // Opponent UUID for determinization
        UUID oppId = null;
        for (UUID pid : liveGame.getOpponents(selfId)) {
            oppId = pid;
            break;
        }

        // Pre-sample determinizations; iterations pick round-robin
        Random rng = new Random();
        String predictedArch = "";
        List<DeterminizationSampler.Determinization> dets = new ArrayList<>();
        if (sampler != null && oppId != null) {
            for (int d = 0; d < DEFAULT_DETERMINIZATIONS; d++) {
                DeterminizationSampler.Determinization det = sampler.sample(liveGame, oppId, rng);
                if (det != null) dets.add(det);
                if (d == 0 && det != null) predictedArch = det.archetype;
            }
        }

        // Phase 4: register root in transposition table so parallel paths
        // that reach the same outer state can share Q/N.
        if (session != null && root.stateFingerprint() == null) {
            try {
                long fp = StateFingerprint.compute(liveGame, selfId);
                if (fp != 0L) {
                    root.setStateFingerprint(fp);
                    session.registerTransposition(fp, root);
                }
            } catch (Throwable ignored) {}
        }

        // Engine snapshots are per-search (captured on first visit to each
        // node within a single MCTS call); drop any carried over from a prior
        // search via the session — live game has moved forward since then.
        if (ENGINE_SNAPSHOTS) {
            clearSubtreeSnapshots(root);
        }

        int iterationsCompleted = 0;
        boolean earlyStopped = false;
        boolean useParallel = PARALLEL_ROLLOUTS > 1 && MAX_OUR_ACTIONS == 1 && ROLLOUT_POOL != null;

        if (useParallel) {
            // Parallel rollout path: run rounds of K iterations with forced
            // root child selection via virtual loss. Batches K leaf evals
            // concurrently so the shared ONNX batch manager sees K-way
            // requests per round instead of one-at-a-time per iter.
            final MCTSNode rootRef = root;
            final UUID oppIdRef = oppId;
            int K = PARALLEL_ROLLOUTS;
            int remaining = DEFAULT_ITERATIONS;
            int detCursor = 0;
            while (remaining > 0) {
                if ((System.currentTimeMillis() - started) >= SEARCH_TIMEOUT_MS) break;
                int roundSize = Math.min(K, remaining);
                int[] picks = selectKChildrenWithVLoss(rootRef, roundSize);
                List<Future<Float>> futures = new ArrayList<>(roundSize);
                for (int k = 0; k < roundSize; k++) {
                    final int forcedIdx = picks[k];
                    final DeterminizationSampler.Determinization det =
                            dets.isEmpty() ? null
                                    : dets.get((detCursor++) % dets.size());
                    futures.add(ROLLOUT_POOL.submit(() -> runForcedRootChildIter(
                            liveGame, selfId, oppIdRef, rootRef, forcedIdx, det, model)));
                }
                for (int k = 0; k < roundSize; k++) {
                    try {
                        long remainingMs = SEARCH_TIMEOUT_MS
                                - (System.currentTimeMillis() - started);
                        if (remainingMs <= 0) {
                            futures.get(k).cancel(false);
                            continue;
                        }
                        float v = futures.get(k).get(remainingMs, TimeUnit.MILLISECONDS);
                        root.backup(picks[k], v);
                        iterationsCompleted++;
                    } catch (Throwable t) {
                        futures.get(k).cancel(false);
                    }
                }
                remaining -= roundSize;

                if (EARLY_STOP_CONCENTRATION > 0f
                        && iterationsCompleted >= EARLY_STOP_MIN_ITERS) {
                    int[] visits = root.visits();
                    if (visits != null && visits.length > 0) {
                        int total = 0, max = 0;
                        for (int v : visits) { total += v; if (v > max) max = v; }
                        if (total > 0 && (float) max / total >= EARLY_STOP_CONCENTRATION) {
                            earlyStopped = true;
                            break;
                        }
                    }
                }
            }
        } else {
            for (int iter = 0; iter < DEFAULT_ITERATIONS; iter++) {
                if ((System.currentTimeMillis() - started) >= SEARCH_TIMEOUT_MS) break;
                DeterminizationSampler.Determinization det =
                        dets.isEmpty() ? null : dets.get(iter % dets.size());
                try {
                    if (USE_INLINE) {
                        runIterationInline(liveGame, selfId, oppId, root, det, model, session);
                    } else {
                        runIteration(liveGame, selfId, oppId, root, det, model, session);
                    }
                    iterationsCompleted++;
                } catch (Throwable t) {
                    // Iteration failed -- continue with remaining budget
                }

                // Early stop when one root action clearly dominates. Saves budget
                // on easy positions; hard ones burn through all iterations.
                if (EARLY_STOP_CONCENTRATION > 0f
                        && iterationsCompleted >= EARLY_STOP_MIN_ITERS) {
                    int[] visits = root.visits();
                    if (visits != null && visits.length > 0) {
                        int total = 0, max = 0;
                        for (int v : visits) { total += v; if (v > max) max = v; }
                        if (total > 0 && (float) max / total >= EARLY_STOP_CONCENTRATION) {
                            earlyStopped = true;
                            break;
                        }
                    }
                }
            }
        }

        int bestIdx = "value".equals(FINAL_SELECTION)
                ? root.bestActionByMeanValue()
                : root.bestActionByVisits();
        long wallMs = System.currentTimeMillis() - started;
        // Aggregate stats logged every N searches so we can see cost-vs-depth
        // without spamming the log.
        long callNum = SEARCH_CALL_COUNT.incrementAndGet();
        TOTAL_SEARCH_MS.addAndGet(wallMs);
        TOTAL_ITERATIONS.addAndGet(iterationsCompleted);
        if (earlyStopped) EARLY_STOP_COUNT.incrementAndGet();
        if (callNum % 100 == 0) {
            long avgMs = TOTAL_SEARCH_MS.get() / Math.max(1, callNum);
            long avgIters = TOTAL_ITERATIONS.get() / Math.max(1, callNum);
            int trans = (session != null) ? session.transpositionCount() : 0;
            int inval = (session != null) ? session.invalidations() : 0;
            long earlyStops = EARLY_STOP_COUNT.get();
            long iters = Math.max(1, ITER_COUNT.get());
            long cloneMs = ITER_CLONE_NS.get() / 1_000_000 / iters;
            long setupMs = ITER_SETUP_NS.get() / 1_000_000 / iters;
            long walkMs  = ITER_WALK_NS.get()  / 1_000_000 / iters;
            long evalMs  = ITER_EVAL_NS.get()  / 1_000_000 / iters;
            System.out.println("[MCTS_STATS] calls=" + callNum
                    + " avg_wallMs=" + avgMs
                    + " avg_iters=" + avgIters
                    + " max_our_actions=" + MAX_OUR_ACTIONS
                    + " early_stop_frac=" + String.format("%.2f", earlyStops / (double) callNum)
                    + " per_iter_ms[clone=" + cloneMs
                    + " setup=" + setupMs
                    + " walk=" + walkMs
                    + " eval=" + evalMs + "]"
                    + " root_map=" + ROOT_MAP_HITS.get() + "/" + (ROOT_MAP_HITS.get() + ROOT_MAP_MISSES.get())
                    + " transpositions=" + trans
                    + " session_resets=" + inval);
        }
        return new Result(bestIdx, root.visits(), root.meanValues(),
                iterationsCompleted, wallMs, predictedArch);
    }

    private static final java.util.concurrent.atomic.AtomicLong SEARCH_CALL_COUNT =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TOTAL_SEARCH_MS =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TOTAL_ITERATIONS =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong EARLY_STOP_COUNT =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ROOT_MAP_HITS =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ROOT_MAP_MISSES =
            new java.util.concurrent.atomic.AtomicLong();

    /** Phase 3b inline walk: run engine on the main thread, drive priority
     *  decisions via an inline callback, no simThread, no queues. Walker state
     *  (current tree node, path, counter) lives on closures inside the
     *  {@link MCTSSimPlayer.MCTSInlineController}. Uses
     *  {@link MCTSSimPlayer.WalkTerminated} to unwind the engine on leaf. */
    private static void runIterationInline(
            Game liveGame, UUID selfId, UUID oppId,
            MCTSNode root,
            DeterminizationSampler.Determinization det,
            PythonModel model,
            MCTSSession session
    ) {
        long tSetupStart = System.nanoTime();

        // Pre-walk tree via PUCT to find the deepest snapshot on this iter's
        // planned path. If found, clone from that snapshot instead of
        // re-playing all actions from root. Only enabled with
        // MCTS_ENGINE_SNAPSHOTS=1 and single determinization (snapshots tied
        // to the det that was active when captured).
        MCTSNode resumeFromNode = null;
        Deque<PathStep> resumePath = null;
        int resumeOurActions = 0;
        if (ENGINE_SNAPSHOTS && DEFAULT_DETERMINIZATIONS <= 1) {
            MCTSNode node = root;
            int ourActions = 0;
            Deque<PathStep> path = new ArrayDeque<>();
            while (node.kind() == MCTSNode.Kind.DECISION) {
                if (MAX_OUR_ACTIONS > 0
                        && node.choiceType() == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY
                        && ourActions >= MAX_OUR_ACTIONS) break;
                int childIdx;
                try {
                    childIdx = node.selectChildPUCT(C_PUCT);
                } catch (Throwable t) {
                    break;
                }
                MCTSNode child = node.getChild(childIdx);
                if (child == null) break;
                path.push(new PathStep(node, childIdx));
                if (node.choiceType() == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY) {
                    ourActions++;
                }
                if (child.engineSnapshot() != null) {
                    resumeFromNode = child;
                    resumePath = new ArrayDeque<>(path);
                    resumeOurActions = ourActions;
                }
                node = child;
            }
        }

        Game sim;
        boolean resumedFromSnapshot = false;
        try {
            if (resumeFromNode != null) {
                sim = resumeFromNode.engineSnapshot().createSimulationForAI();
                resumedFromSnapshot = true;
            } else {
                sim = liveGame.createSimulationForAI();
            }
        } catch (Throwable t) {
            return;
        }
        if (!resumedFromSnapshot && det != null && oppId != null) {
            try {
                DeterminizationSampler.applyToClone(sim, selfId, oppId, det);
            } catch (Throwable ignored) {}
        }
        long tCloneDone = System.nanoTime();

        long iterDeadlineNanos = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(ITER_TIMEOUT_MS);
        MCTSSimPlayer.ReplacementResult rep;
        if (resumedFromSnapshot) {
            // Snapshot clone already has MCTSSimPlayer instances in it (from
            // the iter that captured the snapshot). Build a lighter-weight
            // ReplacementResult that references them directly so the existing
            // inline-controller setup works without re-running the heavier
            // replaceAllPlayers (which calls restore() and would re-init
            // player state from the live game, corrupting mid-walk state).
            rep = new MCTSSimPlayer.ReplacementResult();
            for (mage.players.Player p : sim.getState().getPlayers().values()) {
                if (p instanceof MCTSSimPlayer) {
                    rep.players.put(p.getId(), (MCTSSimPlayer) p);
                }
            }
        } else {
            rep = MCTSSimPlayer.replaceAllPlayers(sim, sim, iterDeadlineNanos);
        }

        // Walker state held in a holder so the controller lambda can mutate it.
        final MCTSNode walkerStart = resumeFromNode != null ? resumeFromNode : root;
        final Deque<PathStep> walkerPath = resumePath != null ? resumePath : new ArrayDeque<>();
        final int walkerOurActions = resumeOurActions;
        final class WalkerState {
            MCTSNode currentNode = walkerStart;
            final Deque<PathStep> path = walkerPath;
            int ourActionsSeen = walkerOurActions;
            boolean reachedLeaf = false;
            boolean gameEnded = false;
        }
        WalkerState st = new WalkerState();

        // Self controller: does PUCT selection + tree descent inline.
        MCTSSimPlayer.MCTSInlineController selfCtrl = req -> {
            if (System.nanoTime() >= iterDeadlineNanos) {
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }
            // Depth cap
            if (MAX_OUR_ACTIONS > 0
                    && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY
                    && st.ourActionsSeen >= MAX_OUR_ACTIONS) {
                st.reachedLeaf = true;
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }
            // Expand current node if first visit
            boolean firstVisit = (st.currentNode.kind() == null);
            if (firstVisit) {
                int n = req.options.size();
                if (n == 0) {
                    throw MCTSSimPlayer.WalkTerminated.INSTANCE;
                }
                Object[] refs = req.options.toArray();
                float[] priors = priorsForRequest(sim, model, req);
                st.currentNode.initializeAsDecision(req.choiceType, refs, priors);

                // Transposition splice
                if (session != null) {
                    try {
                        long fp = StateFingerprint.compute(sim, selfId);
                        if (fp != 0L) {
                            MCTSNode existing = session.getTransposition(fp);
                            if (existing != null
                                    && existing != st.currentNode
                                    && existing.kind() == MCTSNode.Kind.DECISION
                                    && existing.numChildren() == n) {
                                if (!st.path.isEmpty()) {
                                    PathStep top = st.path.peek();
                                    top.node.setChild(top.childIdx, existing);
                                }
                                st.currentNode = existing;
                                // Spliced node's snapshot is from a different
                                // sim path and may not match this walker's
                                // engine state; drop it to be safe.
                                st.currentNode.clearEngineSnapshot();
                                firstVisit = false;
                            } else {
                                st.currentNode.setStateFingerprint(fp);
                                session.registerTransposition(fp, st.currentNode);
                            }
                        }
                    } catch (Throwable ignored) {}
                }

                // Capture engine-state snapshot for reuse on subsequent iters
                // that walk back to this same node via PUCT. Snapshot is
                // tied to the current determinization; only valid when
                // MCTS_DETERMINIZATIONS<=1 (enforced in pre-walk).
                if (ENGINE_SNAPSHOTS && firstVisit && st.currentNode.engineSnapshot() == null) {
                    try {
                        st.currentNode.setEngineSnapshot(sim.createSimulationForAI());
                    } catch (Throwable ignored) {}
                }
            }

            if (st.currentNode.kind() != MCTSNode.Kind.DECISION) {
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }

            boolean atRootActivate = st.currentNode == root
                    && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY;
            int childIdx = st.currentNode.selectChildPUCT(C_PUCT);
            int responseIdx = childIdx;
            if (atRootActivate) {
                int mapped = requestIndexForRootChild(root, childIdx, req.options);
                if (mapped >= 0) {
                    responseIdx = mapped;
                    ROOT_MAP_HITS.incrementAndGet();
                } else {
                    ROOT_MAP_MISSES.incrementAndGet();
                }
            }
            st.path.push(new PathStep(st.currentNode, childIdx));

            MCTSNode child = st.currentNode.getChild(childIdx);
            if (child == null) {
                child = new MCTSNode();
                st.currentNode.setChild(childIdx, child);
            }
            st.currentNode = child;
            if (req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY) {
                st.ourActionsSeen++;
            }
            return responseIdx;
        };

        // Opp controller: random. Never throws (opp plays the whole sim out
        // until our MAX_OUR_ACTIONS is hit or the game ends).
        MCTSSimPlayer.MCTSInlineController oppCtrl = req -> {
            int n = req.options.size();
            return n > 0 ? ThreadLocalRandom().nextInt(n) : 0;
        };

        for (MCTSSimPlayer p : rep.players.values()) {
            if (p.getId().equals(selfId)) {
                p.setInlineController(selfCtrl);
            } else {
                p.setInlineController(oppCtrl);
            }
        }

        long tWalkStart = System.nanoTime();
        ITER_CLONE_NS.addAndGet(tCloneDone - tSetupStart);
        ITER_SETUP_NS.addAndGet(tWalkStart - tCloneDone);

        try {
            try {
                sim.resume();
                // If sim returns normally, the game ended within the walk.
                st.gameEnded = true;
            } catch (MCTSSimPlayer.WalkTerminated ignored) {
                // Walker hit leaf / depth cap / deadline.
            } catch (Throwable t) {
                // Other engine error — treat as leaf and let eval handle.
            }

            long tWalkDone = System.nanoTime();
            ITER_WALK_NS.addAndGet(tWalkDone - tWalkStart);

            float leafValue;
            if (st.gameEnded) {
                leafValue = terminalValueFor(sim, selfId);
            } else {
                leafValue = evaluateLeafValue(sim, selfId, model);
            }
            long tEvalDone = System.nanoTime();
            ITER_EVAL_NS.addAndGet(tEvalDone - tWalkDone);

            while (!st.path.isEmpty()) {
                PathStep step = st.path.pop();
                step.node.backup(step.childIdx, leafValue);
            }
            ITER_COUNT.incrementAndGet();
        } finally {
            long tFinStart = System.nanoTime();
            // Detach inline controllers — sim is being abandoned.
            for (MCTSSimPlayer p : rep.players.values()) {
                p.setInlineController(null);
                p.halt();
            }
            ITER_FIN_NS.addAndGet(System.nanoTime() - tFinStart);
        }
    }

    /** One PUCT walk from root to leaf with backup. */
    private static void runIteration(
            Game liveGame, UUID selfId, UUID oppId,
            MCTSNode root,
            DeterminizationSampler.Determinization det,
            PythonModel model,
            MCTSSession session
    ) {
        long tSetupStart = System.nanoTime();
        // Clone + apply determinization
        Game sim;
        try {
            sim = liveGame.createSimulationForAI();
        } catch (Throwable t) {
            return;
        }
        if (det != null && oppId != null) {
            try {
                DeterminizationSampler.applyToClone(sim, selfId, oppId, det);
            } catch (Throwable ignored) {}
        }
        long tCloneDone = System.nanoTime();

        long iterDeadlineNanos = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(ITER_TIMEOUT_MS);
        MCTSSimPlayer.ReplacementResult rep =
                MCTSSimPlayer.replaceAllPlayers(sim, sim, iterDeadlineNanos);

        // Start sim thread
        AtomicReference<Throwable> simError = new AtomicReference<>();
        Thread simThread = new Thread(() -> {
            try {
                sim.resume();
            } catch (Throwable t) {
                simError.set(t);
            }
        }, "AI-SIM-MCTS-multiply");
        simThread.setDaemon(true);
        simThread.start();
        long tWalkStart = System.nanoTime();
        ITER_CLONE_NS.addAndGet(tCloneDone - tSetupStart);
        ITER_SETUP_NS.addAndGet(tWalkStart - tCloneDone);

        // Walk the tree
        Deque<PathStep> path = new ArrayDeque<>();
        MCTSNode currentNode = root;
        int ourActionsSeen = 0;
        boolean reachedLeaf = false;
        boolean gameEnded = false;

        try {
            while (simThread.isAlive()) {
                if (System.nanoTime() >= iterDeadlineNanos) break;

                MCTSSimPlayer.DecisionRequest req = pollAnyChannel(rep, iterDeadlineNanos);
                if (req == null) {
                    // No request before deadline — sim may have completed or hung
                    break;
                }

                // Non-self requests (opp priority, etc.) → random response, don't touch tree
                if (!req.playerId.equals(selfId)) {
                    int n = req.options.size();
                    int idx = n > 0 ? ThreadLocalRandom().nextInt(n) : 0;
                    MCTSSimPlayer.sendResponse(responseQueueFor(rep, req.playerId), idx);
                    continue;
                }

                // Self's request
                // Depth cap: stop at the next ACTIVATE_ABILITY prompt after we've
                // already seen MAX_OUR_ACTIONS of them. If MAX_OUR_ACTIONS <= 0,
                // there's no cap — the walk continues until the game ends or the
                // iteration deadline (ITER_TIMEOUT_MS) fires below.
                if (MAX_OUR_ACTIONS > 0
                        && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY
                        && ourActionsSeen >= MAX_OUR_ACTIONS) {
                    reachedLeaf = true;
                    MCTSSimPlayer.sendResponse(
                            responseQueueFor(rep, req.playerId),
                            -1);
                    break;
                }

                // Expand if needed
                if (currentNode.kind() == null) {
                    // Not yet initialized — build candidate refs + uniform priors
                    int n = req.options.size();
                    if (n == 0) {
                        // Nothing to choose — halt
                        MCTSSimPlayer.sendResponse(
                                responseQueueFor(rep, req.playerId), -1);
                        break;
                    }
                    Object[] refs = req.options.toArray();
                    float[] priors = priorsForRequest(sim, model, req);
                    currentNode.initializeAsDecision(req.choiceType, refs, priors);

                    // Phase 4: fingerprint this freshly-initialized state and
                    // splice in an existing transposition if one exists.
                    if (session != null) {
                        try {
                            long fp = StateFingerprint.compute(sim, selfId);
                            if (fp != 0L) {
                                MCTSNode existing = session.getTransposition(fp);
                                if (existing != null
                                        && existing != currentNode
                                        && existing.kind() == MCTSNode.Kind.DECISION
                                        && existing.numChildren() == n) {
                                    // Redirect parent's child pointer to the
                                    // existing node so its Q/N is shared. The
                                    // parent's visits[]/values[] arrays are
                                    // unchanged — those are per-edge not per-node.
                                    if (!path.isEmpty()) {
                                        PathStep top = path.peek();
                                        top.node.setChild(top.childIdx, existing);
                                    }
                                    currentNode = existing;
                                } else {
                                    currentNode.setStateFingerprint(fp);
                                    session.registerTransposition(fp, currentNode);
                                }
                            }
                        } catch (Throwable ignored) {}
                    }
                }

                if (currentNode.kind() != MCTSNode.Kind.DECISION) {
                    // Reached a LEAF or TERMINAL node during walk; no more decisions.
                    // Halt and evaluate.
                    MCTSSimPlayer.sendResponse(
                            responseQueueFor(rep, req.playerId), -1);
                    break;
                }

                boolean atRootActivate = currentNode == root
                        && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY;
                int childIdx = currentNode.selectChildPUCT(C_PUCT);
                int responseIdx = childIdx;
                if (atRootActivate) {
                    int mapped = requestIndexForRootChild(root, childIdx, req.options);
                    if (mapped >= 0) {
                        responseIdx = mapped;
                        ROOT_MAP_HITS.incrementAndGet();
                    } else {
                        ROOT_MAP_MISSES.incrementAndGet();
                    }
                }
                path.push(new PathStep(currentNode, childIdx));

                // Send response for the engine to proceed
                MCTSSimPlayer.sendResponse(
                        responseQueueFor(rep, req.playerId), responseIdx);

                // Move to child (create if absent)
                MCTSNode child = currentNode.getChild(childIdx);
                if (child == null) {
                    child = new MCTSNode();
                    currentNode.setChild(childIdx, child);
                }
                currentNode = child;

                if (req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY) {
                    ourActionsSeen++;
                }
            }

            long tWalkDone = System.nanoTime();
            ITER_WALK_NS.addAndGet(tWalkDone - tWalkStart);

            if (!simThread.isAlive()) {
                gameEnded = true;
            }

            // Evaluate leaf value
            float leafValue;
            if (gameEnded) {
                leafValue = terminalValueFor(sim, selfId);
            } else if (reachedLeaf || currentNode != root) {
                leafValue = evaluateLeafValue(sim, selfId, model);
            } else {
                leafValue = evaluateLeafValue(sim, selfId, model);
            }
            long tEvalDone = System.nanoTime();
            ITER_EVAL_NS.addAndGet(tEvalDone - tWalkDone);

            // Backup along path (top-down stack = bottom-up backup order)
            while (!path.isEmpty()) {
                PathStep step = path.pop();
                step.node.backup(step.childIdx, leafValue);
            }
            ITER_COUNT.incrementAndGet();
        } finally {
            long tFinStart = System.nanoTime();
            // Stop the sim thread cleanly.
            for (MCTSSimPlayer p : rep.players.values()) {
                p.halt();
                MCTSSimPlayer.Channel ch = rep.channels.get(p.getId());
                if (ch != null) {
                    MCTSSimPlayer.sendResponse(ch.responseQueue, -1);
                }
            }
            // If the sim is stuck in a long call (e.g. getPlayable's nested
            // clone), the halt flag doesn't get checked until that call
            // returns. Interrupt forcibly so blocking I/O / sleeps unblock,
            // then join with a real timeout so the zombie sim doesn't overlap
            // the next iteration and race on shared state during clone.
            // Without this, XMage's non-thread-safe ArrayList iteration can
            // throw ConcurrentModificationException during clone-within-clone.
            // Interrupt immediately — simThread is almost always blocked on
            // the response queue and halt flag checks are rare in engine code.
            // Prior code used join(50) first, which ate ~50ms/iter waiting for
            // a thread that couldn't check halt anyway. Profile showed this
            // dominated 82% of per-iter cost.
            if (simThread.isAlive()) {
                simThread.interrupt();
                try {
                    simThread.join(50);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }
            ITER_FIN_NS.addAndGet(System.nanoTime() - tFinStart);
        }
    }

    // Per-iteration phase timers; reset via [MCTS_STATS] emission path.
    private static final java.util.concurrent.atomic.AtomicLong ITER_CLONE_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ITER_SETUP_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ITER_WALK_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ITER_EVAL_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ITER_FIN_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong ITER_COUNT = new java.util.concurrent.atomic.AtomicLong();

    /** Recursively clear engineSnapshot on a tree. Called at the start of each
     *  search when MCTS_ENGINE_SNAPSHOTS is enabled so stale snapshots from
     *  prior searches don't leak across decision points. */
    private static void clearSubtreeSnapshots(MCTSNode node) {
        if (node == null) return;
        node.clearEngineSnapshot();
        if (node.kind() == MCTSNode.Kind.DECISION) {
            int n = node.numChildren();
            for (int i = 0; i < n; i++) {
                MCTSNode child = node.getChild(i);
                if (child != null && child != node) {
                    clearSubtreeSnapshots(child);
                }
            }
        }
    }

    /** Pick K root children to visit this round using a snapshot of root's
     *  visits/values/priors plus per-pick virtual-loss bookkeeping. Virtual
     *  loss creates exploration pressure so parallel workers pick distinct
     *  children even though they all see the same tree snapshot. Tree is
     *  unchanged; real backups happen after worker values return. */
    private static int[] selectKChildrenWithVLoss(MCTSNode root, int k) {
        int n = root.numChildren();
        int[] picks = new int[k];
        if (n == 0) {
            return picks;
        }
        int[] vloss = new int[n];
        // Snapshot root stats (backup is synchronized, but other workers may
        // have already backed up during prior rounds — those updates are
        // committed before this round starts).
        int[] visits = root.visits();
        float[] values = root.values();
        float[] priors = root.priors();
        int totalVisits = root.totalVisits();
        int[] visitsSnap = visits != null ? visits.clone() : new int[n];
        float[] valuesSnap = values != null ? values.clone() : new float[n];
        float[] priorsSnap = priors != null ? priors.clone() : new float[n];

        for (int pi = 0; pi < k; pi++) {
            int effTotal = totalVisits;
            for (int v : vloss) effTotal += v;
            float sqrtTotal = (float) Math.sqrt(Math.max(1, effTotal));
            int bestIdx = 0;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                int effVisits = visitsSnap[i] + vloss[i];
                float effValue = valuesSnap[i] + vloss[i] * VLOSS_VALUE;
                float q = effVisits > 0 ? effValue / effVisits : 0f;
                float u = C_PUCT * priorsSnap[i] * sqrtTotal / (1 + effVisits);
                float score = q + u;
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = i;
                }
            }
            picks[pi] = bestIdx;
            vloss[bestIdx]++;
        }
        return picks;
    }

    /** Parallel-path iteration: clone sim, apply det, force the first
     *  ACTIVATE_ABILITY decision to {@code forcedRootChildIdx}, walk until
     *  depth cap / game end / timeout, then evaluate and return the leaf
     *  value. Does not mutate the tree below root — depth 1 only.
     *  Throws RuntimeException on clone failure so the outer loop can skip
     *  the spurious backup (bogus 0f value would otherwise pollute Q). */
    private static float runForcedRootChildIter(
            Game liveGame, UUID selfId, UUID oppId,
            MCTSNode root, int forcedRootChildIdx,
            DeterminizationSampler.Determinization det,
            PythonModel model
    ) {
        long tSetupStart = System.nanoTime();
        Game sim;
        try {
            sim = liveGame.createSimulationForAI();
        } catch (Throwable t) {
            throw new RuntimeException("MCTS clone failed", t);
        }
        if (det != null && oppId != null) {
            try {
                DeterminizationSampler.applyToClone(sim, selfId, oppId, det);
            } catch (Throwable ignored) {}
        }
        long tCloneDone = System.nanoTime();

        long iterDeadlineNanos = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(ITER_TIMEOUT_MS);
        MCTSSimPlayer.ReplacementResult rep =
                MCTSSimPlayer.replaceAllPlayers(sim, sim, iterDeadlineNanos);

        // Walker state per rollout (thread-local; each parallel worker has
        // its own WalkerState + sim clone).
        final class WalkerState {
            int ourActionsSeen = 0;
            boolean forcedConsumed = false;
            boolean reachedLeaf = false;
            boolean gameEnded = false;
        }
        WalkerState st = new WalkerState();

        MCTSSimPlayer.MCTSInlineController selfCtrl = req -> {
            if (System.nanoTime() >= iterDeadlineNanos) {
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }
            // Depth cap (MAX_OUR_ACTIONS=1 means we only do one of OUR activations)
            if (MAX_OUR_ACTIONS > 0
                    && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY
                    && st.ourActionsSeen >= MAX_OUR_ACTIONS) {
                st.reachedLeaf = true;
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }

            int n = req.options.size();
            if (n == 0) {
                throw MCTSSimPlayer.WalkTerminated.INSTANCE;
            }

            if (!st.forcedConsumed
                    && req.choiceType == MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY) {
                st.forcedConsumed = true;
                st.ourActionsSeen++;
                int mapped = requestIndexForRootChild(root, forcedRootChildIdx, req.options);
                if (mapped >= 0) {
                    ROOT_MAP_HITS.incrementAndGet();
                    return mapped;
                }
                ROOT_MAP_MISSES.incrementAndGet();
                if (forcedRootChildIdx >= 0 && forcedRootChildIdx < n) {
                    return forcedRootChildIdx;
                }
                // Out-of-range (can happen if candidate list shifted mid-search);
                // fall through to random to avoid hard failure.
            }

            // Sub-decisions (targets, card selects, multi-ply opp responses):
            // random sample. Keeps this path simple; tree updates below root
            // are avoided in v1 parallel mode.
            return java.util.concurrent.ThreadLocalRandom.current().nextInt(n);
        };

        MCTSSimPlayer.MCTSInlineController oppCtrl = req -> {
            int n = req.options.size();
            return n > 0 ? java.util.concurrent.ThreadLocalRandom.current().nextInt(n) : 0;
        };

        for (MCTSSimPlayer p : rep.players.values()) {
            if (p.getId().equals(selfId)) {
                p.setInlineController(selfCtrl);
            } else {
                p.setInlineController(oppCtrl);
            }
        }

        long tWalkStart = System.nanoTime();
        ITER_CLONE_NS.addAndGet(tCloneDone - tSetupStart);
        ITER_SETUP_NS.addAndGet(tWalkStart - tCloneDone);

        try {
            try {
                sim.resume();
                st.gameEnded = true;
            } catch (MCTSSimPlayer.WalkTerminated ignored) {
                // Normal leaf / depth cap termination
            } catch (Throwable t) {
                // Other engine error — treat as leaf
            }
            long tWalkDone = System.nanoTime();
            ITER_WALK_NS.addAndGet(tWalkDone - tWalkStart);

            float leafValue;
            if (st.gameEnded) {
                leafValue = terminalValueFor(sim, selfId);
            } else {
                leafValue = evaluateLeafValue(sim, selfId, model);
            }
            long tEvalDone = System.nanoTime();
            ITER_EVAL_NS.addAndGet(tEvalDone - tWalkDone);

            ITER_COUNT.incrementAndGet();
            return leafValue;
        } finally {
            long tFinStart = System.nanoTime();
            for (MCTSSimPlayer p : rep.players.values()) {
                p.setInlineController(null);
                p.halt();
            }
            ITER_FIN_NS.addAndGet(System.nanoTime() - tFinStart);
        }
    }

    private static float[] priorsForRequest(Game sim, PythonModel model, MCTSSimPlayer.DecisionRequest req) {
        int n = req.options == null ? 0 : req.options.size();
        if (n <= 0) {
            return new float[0];
        }
        if (!DEEP_POLICY_PRIORS
                || model == null
                || req.choiceType != MCTSSimPlayer.ChoiceType.ACTIVATE_ABILITY) {
            return uniformPriors(n);
        }
        try {
            List<ActivatedAbility> abilities = req.activateOptions();
            int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            int slots = Math.min(n, maxCand);
            int[] ids = new int[maxCand];
            float[][] feats = new float[maxCand][candDim];
            int[] mask = new int[maxCand];
            for (int i = 0; i < slots; i++) {
                ids[i] = abilityActionId(abilities.get(i));
                mask[i] = 1;
            }
            StateSequenceBuilder.SequenceOutput state =
                    StateSequenceBuilder.buildBaseState(
                            sim,
                            sim.getPhase() != null ? sim.getPhase().getType()
                                    : mage.constants.TurnPhase.BEGINNING,
                            StateSequenceBuilder.MAX_LEN);
            PythonMLBatchManager.PredictionResult pred = model.scoreCandidates(
                    state, ids, feats, mask,
                    "mcts_node_prior", "action", 0, 0, 0);
            if (pred == null || pred.policyScores == null) {
                return uniformPriors(n);
            }
            float[] priors = new float[n];
            float floor = 1.0f / Math.max(1, n * 100);
            for (int i = 0; i < n; i++) {
                priors[i] = floor;
            }
            for (int i = 0; i < slots; i++) {
                float p = pred.policyScores[i];
                if (!Float.isNaN(p) && !Float.isInfinite(p) && p > 0f) {
                    priors[i] = p;
                }
            }
            return normalizePriors(priors);
        } catch (Throwable t) {
            return uniformPriors(n);
        }
    }

    private static float[] uniformPriors(int n) {
        float[] priors = new float[n];
        if (n <= 0) {
            return priors;
        }
        float p = 1.0f / n;
        for (int i = 0; i < n; i++) {
            priors[i] = p;
        }
        return priors;
    }

    private static float[] normalizePriors(float[] priors) {
        float sum = 0f;
        for (float p : priors) {
            if (!Float.isNaN(p) && !Float.isInfinite(p) && p > 0f) {
                sum += p;
            }
        }
        if (sum <= 1e-6f) {
            return uniformPriors(priors.length);
        }
        for (int i = 0; i < priors.length; i++) {
            priors[i] = priors[i] > 0f && !Float.isNaN(priors[i]) && !Float.isInfinite(priors[i])
                    ? priors[i] / sum
                    : 0f;
        }
        return priors;
    }

    private static int abilityActionId(Ability ability) {
        if (ability == null) {
            return 0;
        }
        String key = ability.getClass().getName() + ":"
                + ability.getSourceId() + ":"
                + ability.getRule() + ":"
                + ability.toString();
        int h = key.hashCode();
        final int vocab = 65536;
        return 1 + Math.floorMod(h, vocab - 1);
    }

    private static float evaluateLeafValue(Game sim, UUID selfId, PythonModel model) {
        if (model == null) return 0f;
        try {
            sim.checkStateAndTriggered();
            sim.applyEffects();
        } catch (Throwable ignored) {}
        try {
            StateSequenceBuilder.SequenceOutput state =
                    StateSequenceBuilder.buildBaseState(
                            sim,
                            sim.getPhase() != null ? sim.getPhase().getType()
                                    : mage.constants.TurnPhase.BEGINNING,
                            StateSequenceBuilder.MAX_LEN);
            int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
            int candDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;
            int[] dummyIds = new int[maxCand];
            float[][] dummyFeats = new float[maxCand][candDim];
            int[] dummyMask = new int[maxCand];
            dummyMask[0] = 1;
            PythonMLBatchManager.PredictionResult pred = model.scoreCandidates(
                    state, dummyIds, dummyFeats, dummyMask,
                    "mcts_leaf", "action", 0, 0, 0);
            float v = pred != null ? pred.valueScores : 0f;
            if (Float.isNaN(v) || Float.isInfinite(v)) return 0f;
            return Math.max(-1f, Math.min(1f, v));
        } catch (Throwable t) {
            return 0f;
        }
    }

    private static float terminalValueFor(Game sim, UUID selfId) {
        try {
            if (sim.hasEnded()) {
                if (sim.getWinner().contains(sim.getPlayer(selfId).getName())) return 1f;
                return -1f;
            }
        } catch (Throwable ignored) {}
        return 0f;
    }

    /** Block on the shared request queue until a request arrives or deadline. */
    private static MCTSSimPlayer.DecisionRequest pollAnyChannel(
            MCTSSimPlayer.ReplacementResult rep,
            long deadlineNanos) {
        long remaining = deadlineNanos - System.nanoTime();
        if (remaining <= 0) return null;
        try {
            return rep.sharedRequestQueue.poll(remaining, TimeUnit.NANOSECONDS);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    private static BlockingQueue<MCTSSimPlayer.DecisionResponse> responseQueueFor(
            MCTSSimPlayer.ReplacementResult rep, UUID playerId) {
        MCTSSimPlayer.Channel ch = rep.channels.get(playerId);
        if (ch == null) return null;
        return ch.responseQueue;
    }

    /**
     * Root candidates come from ComputerPlayerRL's validated/deduped live list
     * (with Pass at slot 0). The simulation player re-enumerates legal options
     * from a clone, so raw indices are not guaranteed to line up. Search values
     * must be backed up to the live root child, while the engine response must
     * use the matching index in the simulation request.
     */
    private static int requestIndexForRootChild(MCTSNode root, int rootChildIdx, List<?> requestOptions) {
        if (root == null || requestOptions == null || rootChildIdx < 0) {
            return -1;
        }
        Object[] rootRefs = root.candidateRefs();
        if (rootRefs == null || rootChildIdx >= rootRefs.length) {
            return -1;
        }
        Object wanted = rootRefs[rootChildIdx];
        if (!(wanted instanceof Ability)) {
            return -1;
        }
        Ability wantedAbility = (Ability) wanted;
        int looseMatch = -1;
        int looseMatches = 0;
        for (int i = 0; i < requestOptions.size(); i++) {
            Object option = requestOptions.get(i);
            if (!(option instanceof Ability)) {
                continue;
            }
            Ability optionAbility = (Ability) option;
            if (sameAbilityForMcts(wantedAbility, optionAbility, true)) {
                return i;
            }
            if (sameAbilityForMcts(wantedAbility, optionAbility, false)) {
                looseMatch = i;
                looseMatches++;
            }
        }
        return looseMatches == 1 ? looseMatch : -1;
    }

    private static boolean sameAbilityForMcts(Ability a, Ability b, boolean requireSourceId) {
        if (a == b) {
            return true;
        }
        if (a == null || b == null) {
            return false;
        }
        boolean aPass = a instanceof PassAbility;
        boolean bPass = b instanceof PassAbility;
        if (aPass || bPass) {
            return aPass && bPass;
        }
        if (requireSourceId && !Objects.equals(a.getSourceId(), b.getSourceId())) {
            return false;
        }
        return Objects.equals(String.valueOf(a), String.valueOf(b));
    }

    private static Random ThreadLocalRandom() {
        return java.util.concurrent.ThreadLocalRandom.current();
    }

    /** Verify that an inherited root's candidate refs still correspond to the
     *  legal abilities at this priority call. If any ref has been replaced
     *  (abilities regenerated, spells resolved, etc.) the stored priors and
     *  accumulated Q/N may refer to decisions that no longer exist. */
    private static boolean rootRefsMatch(MCTSNode inherited, List<? extends Ability> rootCandidates) {
        Object[] refs = inherited.candidateRefs();
        if (refs == null || refs.length != rootCandidates.size()) return false;
        for (int i = 0; i < refs.length; i++) {
            Object stored = refs[i];
            Object live = rootCandidates.get(i);
            if (stored == live) continue;
            if (stored == null || live == null) return false;
            try {
                if (!stored.equals(live)) return false;
            } catch (Throwable t) {
                return false;
            }
        }
        return true;
    }

    /** (node, chosen child index) pair pushed onto traversal stack during a walk. */
    private static final class PathStep {
        final MCTSNode node;
        final int childIdx;
        PathStep(MCTSNode node, int childIdx) {
            this.node = node;
            this.childIdx = childIdx;
        }
    }

    static {
        // Always dump aggregate phase breakdown at JVM exit so benchmark runs
        // with <100 searches still produce a per-iter us breakdown.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            long calls = SEARCH_CALL_COUNT.get();
            if (calls == 0) return;
            long iters = Math.max(1, ITER_COUNT.get());
            long avgMs = TOTAL_SEARCH_MS.get() / Math.max(1, calls);
            long avgIters = TOTAL_ITERATIONS.get() / Math.max(1, calls);
            long cloneUs = ITER_CLONE_NS.get() / 1_000 / iters;
            long setupUs = ITER_SETUP_NS.get() / 1_000 / iters;
            long walkUs  = ITER_WALK_NS.get()  / 1_000 / iters;
            long evalUs  = ITER_EVAL_NS.get()  / 1_000 / iters;
            long finUs   = ITER_FIN_NS.get()   / 1_000 / iters;
            long totalUs = cloneUs + setupUs + walkUs + evalUs + finUs;
            String msg = "[MCTS_STATS_FINAL] calls=" + calls
                    + " avg_wallMs=" + avgMs
                    + " avg_iters=" + avgIters
                    + " total_iters=" + iters
                    + " per_iter_us[clone=" + cloneUs
                    + " setup=" + setupUs
                    + " walk=" + walkUs
                    + " eval=" + evalUs
                    + " fin=" + finUs
                    + " sum=" + totalUs + "]"
                    + " root_map=" + ROOT_MAP_HITS.get() + "/" + (ROOT_MAP_HITS.get() + ROOT_MAP_MISSES.get())
                    + " early_stops=" + EARLY_STOP_COUNT.get();
            System.out.println(msg);
            System.err.println(msg);
            try (java.io.PrintWriter pw = new java.io.PrintWriter(
                    new java.io.FileWriter("mcts-profile.log", true))) {
                pw.println(new java.util.Date() + " " + msg);
            } catch (java.io.IOException ignored) {
            }
        }, "mcts-profile-telemetry"));
    }
}
