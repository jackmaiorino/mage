package mage.player.ai.rl;

import mage.cards.Card;
import mage.game.Game;
import mage.players.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Separate neural network model for mulligan decisions.
 *
 * Features considered: - Hand composition (lands, creatures, spells, combo
 * pieces) - Mulligan number (1st, 2nd, 3rd, etc.) - Deck composition - Special
 * cards (e.g., Land Grant can fetch lands)
 */
public class MulliganModel {

    private static final Logger logger = LoggerFactory.getLogger(MulliganModel.class);

    private final PythonModel pythonBridge;
    private static final int MAX_HAND_SIZE = 7;
    private static final int MAX_DECK_SIZE = 60;
    private static final int NUM_EXPLICIT = 3; // land_count, creature_count, avg_cmc
    private static final int FEATURE_SIZE = 1 + NUM_EXPLICIT + MAX_HAND_SIZE + MAX_DECK_SIZE; // 71
    private static final int TOKEN_ID_VOCAB = 65536; // Same as StateSequenceBuilder

    // Epsilon-greedy exploration parameters (linear decay)
    private static final double EPSILON_START = EnvConfig.f64("MULLIGAN_EPSILON_START", 0.3);
    private static final double EPSILON_END = EnvConfig.f64("MULLIGAN_EPSILON_END", 0.05);
    private static final int EPSILON_DECAY_EPISODES = EnvConfig.i32("MULLIGAN_EPSILON_DECAY_EPISODES", 20000);

    // Optional keep floor early in training to avoid always-mull collapse
    private static final int KEEP_FLOOR_EPISODES = EnvConfig.i32("MULLIGAN_KEEP_FLOOR_EPISODES", 200);
    private static final double KEEP_FLOOR_P = EnvConfig.f64("MULLIGAN_KEEP_FLOOR_P", 0.7);

    public MulliganModel(PythonModel pythonBridge) {
        this.pythonBridge = pythonBridge;
    }

    /**
     * Calculate epsilon for epsilon-greedy exploration during training.
     * Linear decay from EPSILON_START to EPSILON_END over EPSILON_DECAY_EPISODES.
     */
    private double calculateEpsilon(int episodeNum) {
        if (episodeNum >= EPSILON_DECAY_EPISODES) {
            return EPSILON_END;
        }
        double t = (double) episodeNum / (double) EPSILON_DECAY_EPISODES;
        return EPSILON_START + (EPSILON_END - EPSILON_START) * t;
    }

    /**
     * Result of mulligan decision with card IDs for training.
     */
    public static class MulliganDecision {

        public final boolean shouldMulligan; // Final decision (after overrides)
        public final boolean wasOverridden; // Whether an override was applied
        public final boolean originalModelDecision; // What the model wanted before override
        public final int mulliganNum;
        public final int[] handCardIds;
        public final int[] deckCardIds;
        public final float qKeep;
        public final float qMull;
        public final float[] features; // Full feature vector for training

        public MulliganDecision(boolean shouldMulligan, boolean wasOverridden, boolean originalModelDecision,
                int mulliganNum, int[] handCardIds, int[] deckCardIds, float qKeep, float qMull, float[] features) {
            this.shouldMulligan = shouldMulligan;
            this.wasOverridden = wasOverridden;
            this.originalModelDecision = originalModelDecision;
            this.mulliganNum = mulliganNum;
            this.handCardIds = handCardIds;
            this.deckCardIds = deckCardIds;
            this.qKeep = qKeep;
            this.qMull = qMull;
            this.features = features;
        }
    }

    /**
     * Decide whether to mulligan the current hand.
     *
     * @param player The player making the decision
     * @param game The game state
     * @param mulliganCount How many times we've already mulliganed (0 for
     * opening hand)
     * @return true to mulligan, false to keep
     */
    public boolean shouldMulligan(Player player, Game game, int mulliganCount) {
        return shouldMulligan(player, game, mulliganCount, -1);
    }

    /**
     * Decide whether to mulligan and return card IDs + features for training.
     */
    public MulliganDecision shouldMulliganWithFeatures(Player player, Game game, int mulliganCount, int episodeNum) {
        // Extract card IDs from hand and deck
        int[] handCardIds = extractHandCardIds(player, game);
        int[] deckCardIds = extractDeckCardIds(player, game);

        // Build feature vector with explicit hand features
        float[] features = buildFeatureVector(player, game, mulliganCount, handCardIds, deckCardIds);

        float[] scores = pythonBridge.predictMulliganScores(features);
        float qKeep = scores != null && scores.length > 0 ? scores[0] : 0.0f;
        float qMull = scores != null && scores.length > 1 ? scores[1] : 0.0f;
        boolean shouldMulligan;

        if (episodeNum < 0) {
            // Evaluation/benchmark: fully deterministic argmax
            shouldMulligan = qKeep < qMull;
        } else {
            // Training: epsilon-greedy exploration with stochastic sampling
            double epsilon = calculateEpsilon(episodeNum);

            if (ThreadLocalRandom.current().nextDouble() < epsilon) {
                // Explore: random action
                shouldMulligan = ThreadLocalRandom.current().nextBoolean();
            } else {
                // Exploit: deterministic argmax
                shouldMulligan = qKeep < qMull;
            }

            // Early training keep-floor (only for opening hand, and only for "reasonable" land counts)
            if (KEEP_FLOOR_EPISODES > 0 && episodeNum < KEEP_FLOOR_EPISODES && mulliganCount == 0 && shouldMulligan) {
                int landCount = (int) features[1]; // land_count is at index 1
                if (landCount >= 2 && landCount <= 4) {
                    double t = Math.max(0.0, Math.min(1.0, episodeNum / (double) KEEP_FLOOR_EPISODES));
                    double p = KEEP_FLOOR_P * (1.0 - t);
                    if (ThreadLocalRandom.current().nextDouble() < p) {
                        shouldMulligan = false;
                    }
                }
            }
        }

        // Save the model's original decision before any hard overrides
        boolean originalModelDecision = shouldMulligan;
        boolean wasOverridden = false;

        // Training-only hard override: force mulligan on trivially unkeepable hands.
        // This protects the game/main model while the mulligan model learns.
        // We record the ORIGINAL model decision for training, so bad keeps get punished.
        // Overrides apply ONLY during training (episodeNum >= 0), not eval.
        // - 0 lands: never keepable
        // - All lands (no spells): never keepable
        // - 1 land: allowed (Ponder/Brainstorm/Lorien can find lands)
        if (episodeNum >= 0 && !shouldMulligan) {
            int landCount = (int) features[1];
            int handSize = 0;
            for (int i = 0; i < MAX_HAND_SIZE; i++) {
                if (features[1 + NUM_EXPLICIT + i] != 0) {
                    handSize++;
                }
            }
            // Override if: 0 lands OR all-lands (landCount == handSize, meaning no spells)
            if (handSize >= 4 && (landCount == 0 || landCount >= handSize)) {
                shouldMulligan = true;
                wasOverridden = true;
            }
        }

        return new MulliganDecision(shouldMulligan, wasOverridden, originalModelDecision, mulliganCount, 
                handCardIds, deckCardIds, qKeep, qMull, features);
    }

    /**
     * Decide whether to mulligan the current hand (with episode tracking).
     *
     * @param player The player making the decision
     * @param game The game state
     * @param mulliganCount How many times we've already mulliganed (0 for
     * opening hand)
     * @param episodeNum Current episode number for logging (-1 if unknown)
     * @return true to mulligan, false to keep
     */
    public boolean shouldMulligan(Player player, Game game, int mulliganCount, int episodeNum) {
        try {
            MulliganDecision decision = shouldMulliganWithFeatures(player, game, mulliganCount, episodeNum);
            return decision.shouldMulligan;
        } catch (Exception e) {
            logger.error("Error in mulligan model, defaulting to keep", e);
            return getDefaultMulliganDecision(player);
        }
    }

    /**
     * Build the full feature vector for mulligan model input.
     * Format: [mulligan_num, land_count, creature_count, avg_cmc, hand_ids[7], deck_ids[60]]
     */
    private float[] buildFeatureVector(Player player, Game game, int mulliganCount,
            int[] handCardIds, int[] deckCardIds) {
        // Compute explicit hand features
        int landCount = 0;
        int creatureCount = 0;
        float totalCmc = 0;
        int nonLandCount = 0;
        for (Card card : player.getHand().getCards(game)) {
            if (card == null) {
                continue;
            }
            if (card.isLand(game)) {
                landCount++;
            } else {
                if (card.isCreature(game)) {
                    creatureCount++;
                }
                totalCmc += card.getManaValue();
                nonLandCount++;
            }
        }
        float avgCmc = nonLandCount > 0 ? totalCmc / nonLandCount : 0.0f;

        float[] features = new float[FEATURE_SIZE];
        features[0] = mulliganCount;
        features[1] = landCount;
        features[2] = creatureCount;
        features[3] = avgCmc;
        for (int i = 0; i < handCardIds.length; i++) {
            features[1 + NUM_EXPLICIT + i] = handCardIds[i];
        }
        for (int i = 0; i < deckCardIds.length; i++) {
            features[1 + NUM_EXPLICIT + MAX_HAND_SIZE + i] = deckCardIds[i];
        }
        return features;
    }

    /**
     * Extract card IDs from hand (padded to MAX_HAND_SIZE with 0s).
     */
    public int[] extractHandCardIds(Player player, Game game) {
        int[] handIds = new int[MAX_HAND_SIZE];
        List<Card> hand = new ArrayList<>(player.getHand().getCards(game));

        for (int i = 0; i < Math.min(hand.size(), MAX_HAND_SIZE); i++) {
            handIds[i] = cardNameToTokenId(hand.get(i));
        }
        // Remaining slots are already 0 (padding)

        return handIds;
    }

    /**
     * Extract card IDs from library/deck (padded to MAX_DECK_SIZE with 0s).
     */
    public int[] extractDeckCardIds(Player player, Game game) {
        int[] deckIds = new int[MAX_DECK_SIZE];

        if (player.getLibrary() != null) {
            List<Card> deck = new ArrayList<>(player.getLibrary().getCards(game));

            for (int i = 0; i < Math.min(deck.size(), MAX_DECK_SIZE); i++) {
                deckIds[i] = cardNameToTokenId(deck.get(i));
            }
        }
        // Remaining slots are already 0 (padding)

        return deckIds;
    }

    /**
     * Convert card name to token ID (same as StateSequenceBuilder).
     */
    private int cardNameToTokenId(Card card) {
        if (card == null || card.getName() == null) {
            return 0;
        }
        String key = card.getName();
        int h = key.hashCode();
        int mod = Math.floorMod(h, TOKEN_ID_VOCAB - 1);
        return 1 + mod; // Map to [1, TOKEN_ID_VOCAB-1]
    }

    // Logging is now handled entirely in ComputerPlayerRL.java:
    // - chooseMulligan() logs KEEP decisions immediately
    // - chooseLondonMulliganCards() logs MULLIGAN decisions after card selection
    /**
     * Fallback heuristic if model fails
     */
    private boolean getDefaultMulliganDecision(Player player) {
        int handSize = player.getHand().size();

        // Always keep if we're down to 5 or fewer cards
        if (handSize <= 5) {
            return false;
        }

        // Count lands
        int landCount = 0;
        boolean hasLandGrant = false;
        for (Card card : player.getHand().getCards(null)) {
            if (card.isLand(null)) {
                landCount++;
            }
            if ("Land Grant".equals(card.getName())) {
                hasLandGrant = true;
            }
        }

        // Mulligan if: 0-1 lands (unless Land Grant), or 6+ lands
        if (landCount == 0 && !hasLandGrant) {
            return true; // Mulligan: no mana sources
        }
        if (landCount == 1) {
            return true; // Mulligan: too risky with only 1 land
        }
        if (landCount >= 6) {
            return true; // Mulligan: too many lands
        }

        // Keep 2-5 lands, or 0 lands with Land Grant
        return false;
    }
}
