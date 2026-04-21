package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.game.Game;
import mage.players.Player;
import mage.player.ai.SimulatedPlayerMCTS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * AlphaZero-style MCTS with policy priors and value-net leaf evaluation.
 * Uses determinization sampling for imperfect information (ISMCTS).
 * <p>
 * Unlike the vanilla MCTS in XMage's ComputerPlayerMCTS (which uses random
 * rollouts), this implementation:
 * <ul>
 *   <li>Uses the trained policy net as action prior P(a) for PUCT selection</li>
 *   <li>Uses the trained value net V(s) for leaf evaluation (no rollouts)</li>
 *   <li>Samples hidden info from a {@link DeterminizationSampler} (Bayesian
 *       archetype classification + canonical decklist)</li>
 * </ul>
 * <p>
 * Current limitation: 1-ply search. After applying each candidate action on
 * the clone, we evaluate the resulting state with the value net immediately.
 * Full multi-ply tree expansion (handling both players' interleaved decisions
 * via a coroutine-style Player) is the target architecture but requires
 * intercepting the game engine's decision loop. This 1-ply version already
 * uses the correct tree structure and PUCT math, so extending to multi-ply
 * means plugging in deeper expansion — the node/backprop/selection code stays.
 */
public final class PolicyValueMCTS {

    private static final float C_PUCT = Float.parseFloat(
            System.getenv().getOrDefault("MCTS_C_PUCT", "1.5"));
    private static final int DEFAULT_ITERATIONS = EnvConfig.i32("MCTS_ITERATIONS", 8);
    private static final int DEFAULT_DETERMINIZATIONS = EnvConfig.i32("MCTS_DETERMINIZATIONS", 8);
    private static final long SEARCH_TIMEOUT_MS = EnvConfig.i32("MCTS_TIMEOUT_MS", 5000);
    /** How many of OUR decisions to run forward before value-net leaf eval.
     *  0 = no rollout (1-ply snapshot, old behavior). 3-5 is the target range. */
    private static final int ROLLOUT_DEPTH = EnvConfig.i32("MCTS_ROLLOUT_DEPTH", 3);
    /** Per-rollout wall-time cap (ms). Tune so K rollouts fit in search budget. */
    private static final long ROLLOUT_TIMEOUT_MS = EnvConfig.i32("MCTS_ROLLOUT_TIMEOUT_MS", 1500);
    /** 1 = drive rollouts with the policy net (slower, realistic).
     *  0 = uniform random play inside rollouts (faster, noisy). */
    private static final boolean ROLLOUT_POLICY_DRIVEN = "1".equals(
            System.getenv().getOrDefault("MCTS_POLICY_ROLLOUTS", "1"));
    /** Parallel rollout workers. Each worker drives one rollout. ONNX
     *  inference calls from multiple workers fan into the shared batch
     *  manager and get batched naturally. */
    private static final int PARALLEL_ROLLOUTS = EnvConfig.i32("MCTS_PARALLEL_ROLLOUTS", 8);
    private static final ExecutorService ROLLOUT_POOL = Executors.newFixedThreadPool(
            Math.max(1, PARALLEL_ROLLOUTS),
            new ThreadFactory() {
                private final AtomicInteger counter = new AtomicInteger();
                @Override public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "AI-SIM-MCTS-rollout-" + counter.incrementAndGet());
                    t.setDaemon(true);
                    return t;
                }
            });

    private PolicyValueMCTS() {}

    /** Holds a pending parallel rollout so we can correlate its future
     *  result back to the candidate it evaluates. */
    private static final class RolloutTask {
        final int candidateIndex;
        final Future<Float> future;
        RolloutTask(int candidateIndex, Future<Float> future) {
            this.candidateIndex = candidateIndex;
            this.future = future;
        }
    }

    /**
     * Result of an MCTS search: the chosen action index into the original
     * candidate list, plus diagnostic info.
     */
    public static final class SearchResult {
        public final int bestActionIndex;
        public final int[] aggregateVisits;  // per candidate
        public final float[] aggregateValues; // mean Q per candidate
        public final int determinizationsRun;
        public final long wallMs;
        public final String predictedArchetype;

        SearchResult(int best, int[] visits, float[] values, int dets, long ms, String arch) {
            this.bestActionIndex = best;
            this.aggregateVisits = visits;
            this.aggregateValues = values;
            this.determinizationsRun = dets;
            this.wallMs = ms;
            this.predictedArchetype = arch;
        }
    }

    /**
     * Run ISMCTS search: sample determinizations, build a 1-ply search tree
     * per determinization using policy priors + value-net leaf eval, aggregate
     * visit counts across determinizations, return the best action.
     *
     * @param game          live game state (NOT modified)
     * @param selfId        our player UUID
     * @param candidates    legal candidate abilities at this decision point
     * @param policyProbs   policy net's prior probability per candidate [len = candidates.size()]
     * @param currentValue  value net's estimate of current state V(s)
     * @param model         model for evaluating child states
     * @param sampler       determinization sampler
     * @return search result with best action index + diagnostics
     */
    public static SearchResult search(
            Game game,
            UUID selfId,
            List<? extends Ability> candidates,
            float[] policyProbs,
            float currentValue,
            PythonModel model,
            DeterminizationSampler sampler
    ) {
        int numCandidates = candidates.size();
        int numDets = DEFAULT_DETERMINIZATIONS;
        long started = System.currentTimeMillis();

        int[] totalVisits = new int[numCandidates];
        float[] totalValues = new float[numCandidates];
        int detsCompleted = 0;
        String predictedArch = "";

        UUID oppId = null;
        for (UUID pid : game.getOpponents(selfId)) {
            oppId = pid;
            break;
        }

        Random rng = new Random();

        // Shallow-branching fast path: when candidateCount x K_min <= budget,
        // skip PUCT entirely and enumerate each candidate uniformly. At small
        // branching (our common case: 2-10 candidates), this gives tighter
        // per-candidate value estimates than PUCT's selection dynamics.
        // PUCT's advantage is only meaningful when the budget is much smaller
        // than the branching factor and we need the prior to focus exploration.
        boolean useExhaustive = numCandidates <= DEFAULT_ITERATIONS;
        int visitsPerCandidate = useExhaustive
                ? Math.max(1, DEFAULT_ITERATIONS / numCandidates)
                : 1;

        for (int d = 0; d < numDets; d++) {
            if ((System.currentTimeMillis() - started) >= SEARCH_TIMEOUT_MS) break;

            // 1. Sample a determinization of opp's hidden info.
            DeterminizationSampler.Determinization det = null;
            if (sampler != null && oppId != null) {
                det = sampler.sample(game, oppId, rng);
                if (d == 0) predictedArch = det.archetype;
            }

            if (useExhaustive) {
                // Exhaustive enumeration: visit each candidate K times per determinization,
                // launching all rollouts in parallel. ONNX inference from each worker
                // shares the batch manager so the per-decision policy calls pipeline.
                final DeterminizationSampler.Determinization detFinal = det;
                final UUID oppIdFinal = oppId;
                List<RolloutTask> tasks = new ArrayList<>();
                for (int c = 0; c < numCandidates; c++) {
                    for (int k = 0; k < visitsPerCandidate; k++) {
                        if ((System.currentTimeMillis() - started) >= SEARCH_TIMEOUT_MS) break;
                        final int candIdx = c;
                        RolloutTask task = new RolloutTask(candIdx, ROLLOUT_POOL.submit(() ->
                                evaluateAction(game, selfId, oppIdFinal, candidates, candIdx,
                                        detFinal, model)));
                        tasks.add(task);
                    }
                }
                // Each rollout enforces its own wall-time via MCTS_ROLLOUT_TIMEOUT_MS.
                // Wait for each to complete up to the remaining global budget.
                // Use cancel(false) so we don't interrupt in-flight ONNX calls
                // on a stale/slow rollout — let them finish cleanly in the pool.
                for (RolloutTask task : tasks) {
                    long elapsed = System.currentTimeMillis() - started;
                    long waitMs = SEARCH_TIMEOUT_MS - elapsed;
                    if (waitMs <= 0) {
                        task.future.cancel(false);
                        continue;
                    }
                    try {
                        float leafValue = task.future.get(waitMs, TimeUnit.MILLISECONDS);
                        totalVisits[task.candidateIndex]++;
                        totalValues[task.candidateIndex] += leafValue;
                    } catch (java.util.concurrent.TimeoutException te) {
                        task.future.cancel(false);
                    } catch (Throwable t) {
                        task.future.cancel(false);
                    }
                }
            } else {
                // PUCT tree search for high-branching cases.
                PolicyValueMCTSNode root = new PolicyValueMCTSNode(null, -1, 1.0f);
                root.valueEstimate = currentValue;
                root.expand(policyProbs);
                for (int iter = 0; iter < DEFAULT_ITERATIONS; iter++) {
                    if ((System.currentTimeMillis() - started) >= SEARCH_TIMEOUT_MS) break;
                    PolicyValueMCTSNode selected = root.selectChild(C_PUCT);
                    if (selected == null) break;
                    float leafValue = evaluateAction(
                            game, selfId, oppId, candidates, selected.actionIndex,
                            det, model);
                    selected.visitCount++;
                    selected.totalValue += leafValue;
                    root.visitCount++;
                }
                if (root.children != null) {
                    for (PolicyValueMCTSNode child : root.children) {
                        int idx = child.actionIndex;
                        if (idx >= 0 && idx < numCandidates) {
                            totalVisits[idx] += child.visitCount;
                            totalValues[idx] += child.totalValue;
                        }
                    }
                }
            }
            detsCompleted++;
        }

        // Compute mean values per candidate.
        float[] meanValues = new float[numCandidates];
        for (int i = 0; i < numCandidates; i++) {
            meanValues[i] = totalVisits[i] > 0 ? totalValues[i] / totalVisits[i] : 0;
        }

        // 5. Pick action: primary = highest mean value (expected winrate);
        // tiebreaker = most visits (PUCT correlates visits with promise).
        // This handles the exhaustive case (all visits tied) cleanly and
        // doesn't regress the PUCT case (visits track value closely anyway).
        int bestIdx = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        int bestVisits = -1;
        for (int i = 0; i < numCandidates; i++) {
            if (totalVisits[i] == 0) continue;
            if (meanValues[i] > bestValue
                    || (meanValues[i] == bestValue && totalVisits[i] > bestVisits)) {
                bestValue = meanValues[i];
                bestVisits = totalVisits[i];
                bestIdx = i;
            }
        }

        long elapsed = System.currentTimeMillis() - started;
        return new SearchResult(bestIdx, totalVisits, meanValues, detsCompleted, elapsed, predictedArch);
    }

    /**
     * Clone the game, apply determinization, execute one candidate action,
     * then evaluate the resulting state with the value net (no random rollout).
     * <p>
     * We do NOT run {@code sim.resume()} to game completion — random-play
     * rollouts add more noise than signal when we already have a trained
     * value net. Instead, after the action is activated and the stack
     * resolves (via a targeted single-step), we read the value head's
     * estimate of the resulting state.
     */
    // Simple atomic counters for MCTS time breakdown (ns)
    private static final java.util.concurrent.atomic.AtomicLong TIME_CLONE_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TIME_DET_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TIME_ACTIVATE_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TIME_ROLLOUT_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong TIME_LEAF_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicInteger COUNT_CLONES = new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger COUNT_ROLLOUTS = new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger COUNT_LEAVES = new java.util.concurrent.atomic.AtomicInteger();
    public static String getMctsTimingStats() {
        long c = COUNT_CLONES.get();
        long r = COUNT_ROLLOUTS.get();
        long l = COUNT_LEAVES.get();
        double toMs = 1e6;
        return String.format(
                "MCTS_TIMING clones=%d(%.1fms avg) dets=%.1fms activates=%.1fms rollouts=%d(%.1fms avg) leaves=%d(%.1fms avg)",
                c, c > 0 ? TIME_CLONE_NS.get() / toMs / c : 0,
                TIME_DET_NS.get() / toMs,
                TIME_ACTIVATE_NS.get() / toMs,
                r, r > 0 ? TIME_ROLLOUT_NS.get() / toMs / r : 0,
                l, l > 0 ? TIME_LEAF_NS.get() / toMs / l : 0
        );
    }

    private static float evaluateAction(
            Game liveGame, UUID selfId, UUID oppId,
            List<? extends Ability> candidates, int actionIndex,
            DeterminizationSampler.Determinization det,
            PythonModel model
    ) {
        try {
            long t_clone_start = System.nanoTime();
            Game sim = liveGame.createSimulationForAI();
            TIME_CLONE_NS.addAndGet(System.nanoTime() - t_clone_start);
            COUNT_CLONES.incrementAndGet();

            // Apply determinization to opp's hidden zones if available.
            long t_det_start = System.nanoTime();
            if (det != null && oppId != null) {
                DeterminizationSampler.applyToClone(sim, selfId, oppId, det);
            }
            TIME_DET_NS.addAndGet(System.nanoTime() - t_det_start);

            // Execute the candidate action on the sim. This puts the spell
            // on the stack (or plays the land, or activates the mana ability).
            long t_act_start = System.nanoTime();
            Player simSelf = sim.getPlayer(selfId);
            if (simSelf != null && actionIndex >= 0 && actionIndex < candidates.size()) {
                Ability chosen = candidates.get(actionIndex);
                if (chosen instanceof ActivatedAbility) {
                    try {
                        simSelf.activateAbility((ActivatedAbility) chosen, sim);
                    } catch (Throwable t) {
                        // Activation failed — fall through to evaluate current state.
                    }
                }
            }
            TIME_ACTIVATE_NS.addAndGet(System.nanoTime() - t_act_start);

            // Now advance the sim N OUR-priority decisions forward using the
            // coroutine infrastructure, then evaluate either terminal outcome
            // or value-net leaf. Puts the action's consequences through real
            // engine resolution (including opp responses) before scoring.
            if (ROLLOUT_DEPTH > 0) {
                long t_roll_start = System.nanoTime();
                CoroutineRollout.Result result = ROLLOUT_POLICY_DRIVEN
                        ? CoroutineRollout.runPolicyTruncatedRollout(sim, selfId, ROLLOUT_DEPTH, model, ROLLOUT_TIMEOUT_MS)
                        : CoroutineRollout.runTruncatedRollout(sim, selfId, ROLLOUT_DEPTH, model, ROLLOUT_TIMEOUT_MS);
                TIME_ROLLOUT_NS.addAndGet(System.nanoTime() - t_roll_start);
                COUNT_ROLLOUTS.incrementAndGet();
                return result.valueFor(selfId);
            }

            // Fallback (ROLLOUT_DEPTH=0): 1-ply snapshot value-net eval.
            // This was the original behavior that showed the distribution-shift
            // bug; kept as a toggle for A/B comparison.
            try {
                sim.checkStateAndTriggered();
                sim.applyEffects();
            } catch (Throwable ignored) {}
            if (model == null) return 0f;
            long t_leaf_start = System.nanoTime();
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
                    "mcts_eval", "action", 0, 0, 0);
            TIME_LEAF_NS.addAndGet(System.nanoTime() - t_leaf_start);
            COUNT_LEAVES.incrementAndGet();
            float v = pred != null ? pred.valueScores : 0f;
            if (Float.isNaN(v) || Float.isInfinite(v)) return 0f;
            return Math.max(-1f, Math.min(1f, v));

        } catch (Throwable t) {
            return 0f;
        }
    }
}
