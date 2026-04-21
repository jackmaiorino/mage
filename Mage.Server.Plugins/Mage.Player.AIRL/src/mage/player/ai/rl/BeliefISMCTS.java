package mage.player.ai.rl;

import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import mage.players.Player;
import mage.player.ai.SimulatedPlayerMCTS;

import java.util.ArrayList;
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
