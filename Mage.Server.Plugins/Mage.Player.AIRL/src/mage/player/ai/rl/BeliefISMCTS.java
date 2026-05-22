package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import mage.players.Player;
import mage.player.ai.SimulatedPlayerMCTS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Phase 2: Information Set Monte Carlo rollouts for belief-informed
 * action selection.
 * <p>
 * Uses {@code game.createSimulationForAI()} (same pattern as the existing
 * {@code ComputerPlayerMCTS}) to deep-copy the game and run random-policy
 * rollouts to completion. Hidden information (opponent hand / library order)
 * is replaced with a shuffled draw from a pool -- by default the mixed
 * current hand+library, matching vanilla MCTS. If an archetype belief is
 * provided, future work can substitute a canonical decklist for that
 * archetype here (true ISMCTS determinization).
 * <p>
 * This class is intentionally side-effect free with respect to the live
 * game: every operation happens on the simulation clone. Callers MUST NOT
 * confuse the returned rollout stats with real-game outcomes.
 */
public final class BeliefISMCTS {

    private static final long ROLLOUT_TIMEOUT_MS = EnvConfig.i32("ISMCTS_ROLLOUT_TIMEOUT_MS", 5000);
    private static final int ROOT_DETERMINIZATIONS = EnvConfig.i32(
            "ISMCTS_ROOT_DETERMINIZATIONS", EnvConfig.i32("ISMCTS_DETERMINIZATIONS", 1));
    private static final int ROOT_ROLLOUTS_PER_ACTION = EnvConfig.i32("ISMCTS_ROOT_ROLLOUTS_PER_ACTION", 1);
    private static final long ROOT_SEARCH_TIMEOUT_MS = EnvConfig.i64("ISMCTS_ROOT_SEARCH_TIMEOUT_MS", 3000L);
    private static final int ROLLOUT_WIN = 1;
    private static final int ROLLOUT_LOSS = -1;
    private static final int ROLLOUT_DRAW_OR_ERROR = 0;

    private BeliefISMCTS() {
    }

    public static final class RolloutStats {
        public final int wins;
        public final int losses;
        public final int drawsOrErrors;
        public final long totalMillis;

        RolloutStats(int wins, int losses, int drawsOrErrors, long totalMillis) {
            this.wins = wins;
            this.losses = losses;
            this.drawsOrErrors = drawsOrErrors;
            this.totalMillis = totalMillis;
        }

        public int completed() {
            return wins + losses;
        }

        public double winRate() {
            int n = completed();
            return n == 0 ? 0.0 : (double) wins / (double) n;
        }
    }

    public static final class SearchResult {
        public final int bestActionIndex;
        public final int[] aggregateVisits;
        public final float[] aggregateValues;
        public final int determinizationsRun;
        public final long wallMs;
        public final String predictedArchetype;

        SearchResult(int bestActionIndex, int[] aggregateVisits, float[] aggregateValues,
                     int determinizationsRun, long wallMs, String predictedArchetype) {
            this.bestActionIndex = bestActionIndex;
            this.aggregateVisits = aggregateVisits;
            this.aggregateValues = aggregateValues;
            this.determinizationsRun = determinizationsRun;
            this.wallMs = wallMs;
            this.predictedArchetype = predictedArchetype;
        }
    }

    /**
     * Run {@code numRollouts} randomized rollouts of the given game from the
     * current state and report aggregate win/loss counts for {@code selfId}.
     * <p>
     * Each rollout:
     *   1. deep-copies the game via {@link Game#createSimulationForAI}
     *   2. replaces every player with a {@link SimulatedPlayerMCTS} so both
     *      sides auto-play (random-ish, fast)
     *   3. shuffles all libraries + re-draws opponent hands from library
     *      (standard MCTS hidden-info handling)
     *   4. calls {@code sim.resume()} to run the game to completion
     *   5. checks whether {@code selfId} has won.
     * <p>
     * Any exception in a rollout is counted as a draw-or-error (i.e. doesn't
     * skew win rate).
     */
    public static RolloutStats runRollouts(Game liveGame, UUID selfId, int numRollouts) {
        if (liveGame == null || selfId == null || numRollouts <= 0) {
            return new RolloutStats(0, 0, 0, 0L);
        }
        int wins = 0, losses = 0, other = 0;
        long started = System.currentTimeMillis();
        for (int r = 0; r < numRollouts; r++) {
            if ((System.currentTimeMillis() - started) >= ROLLOUT_TIMEOUT_MS) {
                break;
            }
            int result;
            try {
                result = runSingleRollout(liveGame, selfId);
            } catch (Throwable t) {
                result = ROLLOUT_DRAW_OR_ERROR;
            }
            if (result == ROLLOUT_WIN) wins++;
            else if (result == ROLLOUT_LOSS) losses++;
            else other++;
        }
        return new RolloutStats(wins, losses, other, System.currentTimeMillis() - started);
    }

    /**
     * Run ISMCTS rollouts with belief-driven determinization: archetype is
     * inferred from visible cards, then opp's hand/library are sampled from
     * the canonical decklist for that archetype.
     */
    public static RolloutStats runBeliefRollouts(Game liveGame, UUID selfId,
                                                 DeterminizationSampler sampler,
                                                 int numRollouts) {
        if (liveGame == null || selfId == null || sampler == null || numRollouts <= 0) {
            return new RolloutStats(0, 0, 0, 0L);
        }
        UUID oppId = null;
        for (UUID pid : liveGame.getOpponents(selfId)) {
            oppId = pid;
            break;
        }
        if (oppId == null) return new RolloutStats(0, 0, 0, 0L);

        int wins = 0, losses = 0, other = 0;
        long started = System.currentTimeMillis();
        Random rng = new Random();
        for (int r = 0; r < numRollouts; r++) {
            if ((System.currentTimeMillis() - started) >= ROLLOUT_TIMEOUT_MS) break;
            int result;
            try {
                DeterminizationSampler.Determinization det = sampler.sample(liveGame, oppId, rng);
                result = runSingleBeliefRollout(liveGame, selfId, oppId, det);
            } catch (Throwable t) {
                result = ROLLOUT_DRAW_OR_ERROR;
            }
            if (result == ROLLOUT_WIN) wins++;
            else if (result == ROLLOUT_LOSS) losses++;
            else other++;
        }
        return new RolloutStats(wins, losses, other, System.currentTimeMillis() - started);
    }

    /**
     * Eval-only root selector using belief determinization plus terminal random
     * rollouts. This deliberately avoids the value head used by PolicyValueMCTS:
     * each candidate is force-applied in a simulation clone, then the rest of
     * the game is played out by {@link SimulatedPlayerMCTS}. The result is too
     * expensive/noisy for training, but it is a useful diagnostic for whether
     * imperfect-information rollout search has any policy-improvement signal at
     * critical Spy decisions.
     */
    public static SearchResult searchRoot(Game liveGame, UUID selfId,
                                          List<? extends Ability> candidates,
                                          float[] policyPriors,
                                          DeterminizationSampler sampler) {
        int numCandidates = candidates == null ? 0 : candidates.size();
        if (liveGame == null || selfId == null || sampler == null || numCandidates <= 0) {
            return new SearchResult(0, new int[Math.max(0, numCandidates)],
                    new float[Math.max(0, numCandidates)], 0, 0L, "");
        }

        UUID oppId = null;
        for (UUID pid : liveGame.getOpponents(selfId)) {
            oppId = pid;
            break;
        }
        if (oppId == null) {
            return new SearchResult(0, new int[numCandidates], new float[numCandidates], 0, 0L, "");
        }

        int[] visits = new int[numCandidates];
        float[] sums = new float[numCandidates];
        long started = System.currentTimeMillis();
        Random rng = new Random();
        int detsCompleted = 0;
        String predictedArch = "";

        int dets = Math.max(1, ROOT_DETERMINIZATIONS);
        int rolloutsPerAction = Math.max(1, ROOT_ROLLOUTS_PER_ACTION);
        for (int d = 0; d < dets; d++) {
            if ((System.currentTimeMillis() - started) >= ROOT_SEARCH_TIMEOUT_MS) {
                break;
            }
            DeterminizationSampler.Determinization det;
            try {
                det = sampler.sample(liveGame, oppId, rng);
                if (d == 0 && det != null) {
                    predictedArch = det.archetype;
                }
            } catch (Throwable t) {
                continue;
            }
            for (int c = 0; c < numCandidates; c++) {
                for (int k = 0; k < rolloutsPerAction; k++) {
                    if ((System.currentTimeMillis() - started) >= ROOT_SEARCH_TIMEOUT_MS) {
                        break;
                    }
                    int result = runSingleBeliefActionRollout(liveGame, selfId, oppId, det, candidates.get(c));
                    visits[c]++;
                    if (result == ROLLOUT_WIN) {
                        sums[c] += 1.0f;
                    } else if (result == ROLLOUT_LOSS) {
                        sums[c] -= 1.0f;
                    }
                }
            }
            detsCompleted++;
        }

        float[] means = new float[numCandidates];
        for (int i = 0; i < numCandidates; i++) {
            means[i] = visits[i] > 0 ? sums[i] / visits[i] : 0.0f;
        }
        int bestIdx = bestMeanIndex(means, visits, policyPriors);
        return new SearchResult(bestIdx, visits, means, detsCompleted,
                System.currentTimeMillis() - started, predictedArch);
    }

    private static int bestMeanIndex(float[] means, int[] visits, float[] policyPriors) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        int bestVisits = -1;
        float bestPrior = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < means.length; i++) {
            if (visits[i] <= 0) {
                continue;
            }
            float prior = policyPriors != null && i < policyPriors.length ? policyPriors[i] : 0.0f;
            if (means[i] > bestValue
                    || (means[i] == bestValue && visits[i] > bestVisits)
                    || (means[i] == bestValue && visits[i] == bestVisits && prior > bestPrior)) {
                best = i;
                bestValue = means[i];
                bestVisits = visits[i];
                bestPrior = prior;
            }
        }
        return best;
    }

    private static int runSingleBeliefActionRollout(Game liveGame, UUID selfId,
                                                    UUID oppId,
                                                    DeterminizationSampler.Determinization det,
                                                    Ability candidate) {
        Game sim = liveGame.createSimulationForAI();

        for (Player oldPlayer : new ArrayList<>(sim.getState().getPlayers().values())) {
            Player livePlayer = liveGame.getState().getPlayers().get(oldPlayer.getId());
            if (livePlayer == null) {
                continue;
            }
            Player origPlayer = livePlayer.copy();
            SimulatedPlayerMCTS newPlayer = new SimulatedPlayerMCTS(oldPlayer, true);
            newPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(oldPlayer.getId(), newPlayer);
        }

        DeterminizationSampler.applyToClone(sim, selfId, oppId, det);

        Player simSelf = sim.getPlayer(selfId);
        if (simSelf == null || !(candidate instanceof ActivatedAbility)) {
            return ROLLOUT_DRAW_OR_ERROR;
        }
        try {
            boolean activated = simSelf.activateAbility((ActivatedAbility) candidate.copy(), sim);
            if (!activated) {
                return ROLLOUT_DRAW_OR_ERROR;
            }
        } catch (Throwable t) {
            return ROLLOUT_DRAW_OR_ERROR;
        }

        try {
            sim.resume();
        } catch (Throwable t) {
            return ROLLOUT_DRAW_OR_ERROR;
        }
        Player me = sim.getPlayer(selfId);
        if (me == null) return ROLLOUT_DRAW_OR_ERROR;
        if (me.hasWon()) return ROLLOUT_WIN;
        if (me.hasLost()) return ROLLOUT_LOSS;
        return ROLLOUT_DRAW_OR_ERROR;
    }

    private static int runSingleBeliefRollout(Game liveGame, UUID selfId,
                                              UUID oppId,
                                              DeterminizationSampler.Determinization det) {
        Game sim = liveGame.createSimulationForAI();

        // Replace players with SimulatedPlayerMCTS for auto-play.
        for (Player oldPlayer : new ArrayList<>(sim.getState().getPlayers().values())) {
            Player origPlayer = liveGame.getState().getPlayers().get(oldPlayer.getId()).copy();
            SimulatedPlayerMCTS newPlayer = new SimulatedPlayerMCTS(oldPlayer, true);
            newPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(oldPlayer.getId(), newPlayer);
        }

        // Apply belief-driven determinization: replace opp's hand+library
        // with a sample drawn from the predicted archetype's canonical decklist.
        DeterminizationSampler.applyToClone(sim, selfId, oppId, det);

        try {
            sim.resume();
        } catch (Throwable t) {
            return ROLLOUT_DRAW_OR_ERROR;
        }
        Player me = sim.getPlayer(selfId);
        if (me == null) return ROLLOUT_DRAW_OR_ERROR;
        if (me.hasWon()) return ROLLOUT_WIN;
        if (me.hasLost()) return ROLLOUT_LOSS;
        return ROLLOUT_DRAW_OR_ERROR;
    }

    private static int runSingleRollout(Game liveGame, UUID selfId) {
        Game sim = liveGame.createSimulationForAI();

        // Replace each player with a SimulatedPlayerMCTS so both sides auto-play.
        for (Player oldPlayer : new ArrayList<>(sim.getState().getPlayers().values())) {
            Player origPlayer = liveGame.getState().getPlayers().get(oldPlayer.getId()).copy();
            SimulatedPlayerMCTS newPlayer = new SimulatedPlayerMCTS(oldPlayer, true);
            newPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(oldPlayer.getId(), newPlayer);
        }

        // Randomize hidden info: opp's exact library order and hand contents
        // aren't known to the real agent, so shuffle them. For our player we
        // only shuffle the library (the hand is public to us).
        for (Player p : sim.getState().getPlayers().values()) {
            if (!p.getId().equals(selfId)) {
                int handSize = p.getHand().size();
                p.getLibrary().addAll(p.getHand().getCards(sim), sim);
                p.getHand().clear();
                p.getLibrary().shuffle();
                for (int i = 0; i < handSize; i++) {
                    Card card = p.getLibrary().drawFromTop(sim);
                    if (card != null) {
                        card.setZone(Zone.HAND, sim);
                        p.getHand().add(card);
                    }
                }
            } else {
                p.getLibrary().shuffle();
            }
        }

        try {
            sim.resume();
        } catch (Throwable t) {
            return ROLLOUT_DRAW_OR_ERROR;
        }

        Player me = sim.getPlayer(selfId);
        if (me == null) return ROLLOUT_DRAW_OR_ERROR;
        if (me.hasWon()) return ROLLOUT_WIN;
        if (me.hasLost()) return ROLLOUT_LOSS;
        return ROLLOUT_DRAW_OR_ERROR;
    }
}
