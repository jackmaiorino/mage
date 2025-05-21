package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import mage.Mana;
import mage.abilities.LoyaltyAbility;
import mage.abilities.costs.common.PayLoyaltyCost;
import mage.abilities.costs.mana.ManaCost;
import mage.cards.Card;
import mage.constants.TurnPhase;
import mage.constants.Zone;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;

/**
 * Helper that converts a {@link Game} snapshot into a padded token sequence +
 * attention mask for a Transformer encoder. Can also prepend action tokens to
 * the sequence for prediction.
 */
public class StateSequenceBuilder {

    private static final Logger logger = Logger.getLogger(StateSequenceBuilder.class.getName());

    /* === CONFIGURATION ================================================= */
    public static final int DIM_PER_TOKEN = 128; // feature dim per token
    public static final int MAX_LEN = 256; // hard cap for sequence length

    // special‑token IDs (for simple one‑hot placeholder embedding)
    private static final int CLS_ID = 0;
    private static final int PHASE_BASE = 10; // + phase.ordinal()
    private static final int ACTION_BASE = 30; // + actionType.ordinal()

    public enum ZoneType {
        HAND,
        BATTLEFIELD,
        GRAVEYARD,
        EXILE,
        LIBRARY,
        STACK,
        REFERENCE
    }

    public enum ActionType {
        ACTIVATE_ABILITY_OR_SPELL,
        ACTIVATE_MANA_ABILITY,
        ACTIVATE_LOYALTY_ABILITY,
        CAST_SPELL,
        SELECT_TARGETS,
        DECLARE_ATTACKS,
        DECLARE_BLOCKS,
        MULLIGAN,
        SELECT_CHOICE,
        SELECT_TRIGGERED_ABILITY,
        SELECT_CARD
    }

    /* === PUBLIC API ===================================================== */
    /**
     * Build base sequence + mask for the given game state.
     *
     * @param game current game
     * @param phase current phase of the game
     * @param maxLen padding / truncation length (≤ {@link #MAX_LEN})
     * @return sequence + mask
     */
    public static SequenceOutput buildBaseState(Game game, TurnPhase phase, int maxLen) {
        if (maxLen > MAX_LEN) {
            throw new IllegalArgumentException("maxLen exceeds hard cap: " + MAX_LEN);
        }

        /*
         * ------------------------------------------------------------
         * 1. Resolve active player (mirrors original logic)
         * ----------------------------------------------------------
         */
        Player player = game.getPlayer(game.getActivePlayerId());
        if (player == null) {
            player = game.getPlayer(game.getState().getChoosingPlayerId());
            if (player == null) {
                logger.severe("No active player found in game " + game.getId());
                throw new IllegalStateException("Cannot build state sequence: no active player");
            }
        }

        /*
         * ------------------------------------------------------------
         * 2. Assemble token list + mask
         * ----------------------------------------------------------
         */
        List<float[]> tokens = new ArrayList<>();
        List<Integer> mask = new ArrayList<>();

        // (a) Special tokens -----------------------------------------
        tokens.add(embedSpecial(CLS_ID));
        mask.add(0);  // Don't mask CLS token
        if (phase != null) {
            tokens.add(embedSpecial(PHASE_BASE + phase.ordinal()));
        } else {
            // Special case for mulligan
            tokens.add(embedSpecial(PHASE_BASE + TurnPhase.values().length + 1));
        }
        mask.add(0);  // Don't mask phase token

        // (b) Player + opponent stats tokens -------------------------
        tokens.add(embedPlayerStats(player, game));
        mask.add(0);  // Don't mask player stats
        Player opponent = game.getPlayer(game.getOpponents(player.getId()).iterator().next());
        if (opponent == null) {
            logger.severe("No opponent found in game " + game.getId());
            throw new IllegalStateException("Cannot build state sequence: no opponent");
        }
        tokens.add(embedPlayerStats(opponent, game));
        mask.add(0);  // Don't mask opponent stats

        // (c) Variable-length entity tokens --------------------------
        addCardList(tokens, mask, player.getHand().getCards(game), Zone.HAND, game);
        addPermList(tokens, mask, game.getBattlefield().getAllActivePermanents(player.getId()), Zone.BATTLEFIELD, game);
        addCardList(tokens, mask, player.getGraveyard().getCards(game), Zone.GRAVEYARD, game);
        addCardList(tokens, mask, player.getLibrary().getCards(game), Zone.LIBRARY, game);

        // Opponent perms
        addPermList(tokens, mask, game.getBattlefield().getAllActivePermanents(opponent.getId()), Zone.BATTLEFIELD, game);
        addCardList(tokens, mask, opponent.getGraveyard().getCards(game), Zone.GRAVEYARD, game);

        for (ExileZone ez : game.getExile().getExileZones()) {
            addCardList(tokens, mask, ez.getCards(game), Zone.EXILED, game);
        }
        for (StackObject so : game.getStack()) {
            if (so instanceof Card) {
                addCard(tokens, mask, (Card) so, Zone.STACK, game);
            }
        }

        /*
         * ------------------------------------------------------------
         * 3. Pad / truncate to maxLen
         * ----------------------------------------------------------
         */
        while (tokens.size() < maxLen) {
            tokens.add(new float[DIM_PER_TOKEN]); // zero padding token
            mask.add(1);  // Only mask padding tokens
        }
        if (tokens.size() > maxLen) {
            tokens = tokens.subList(0, maxLen);
            mask = mask.subList(0, maxLen);
        }

        return new SequenceOutput(tokens, mask);
    }

    /**
     * Create a new sequence with an action token prepended to the base state
     * sequence.
     *
     * @param baseState The base state sequence
     * @param actionType The action to prepend
     * @param abilityEncoding The encoded ability to prepend
     * @return A new sequence with the action prepended
     */
    public static SequenceOutput prependAction(SequenceOutput baseState, ActionType actionType, float[] abilityEncoding) {
        // Create new sequence with action token prepended
        List<float[]> tokens = new ArrayList<>();
        List<Integer> mask = new ArrayList<>();

        // Prepend action tokens
        tokens.add(0, abilityEncoding);
        tokens.add(0, embedSpecial(ACTION_BASE + actionType.ordinal()));
        mask.add(0, 0);  // Don't mask ability encoding
        mask.add(0, 0);  // Don't mask action token

        // Add base state tokens
        tokens.addAll(baseState.tokens);
        mask.addAll(baseState.mask);

        return new SequenceOutput(tokens, mask);
    }

    /* === EMBEDDING HELPERS ============================================== */
    private static float[] embedSpecial(int tokenId) {
        float[] v = new float[DIM_PER_TOKEN];
        v[tokenId % DIM_PER_TOKEN] = 1.0f; // simple one‑hot placeholder
        return v;
    }

    private static float[] embedPlayerStats(Player p, Game g) {
        float[] v = new float[DIM_PER_TOKEN];
        v[0] = (float) p.getLife() / g.getStartingLife();
        v[1] = (float) p.getHand().size() / 7f;
        v[2] = (float) p.getLibrary().size() / 60f;
        v[3] = (float) p.getGraveyard().size() / 10f;
        v[4] = (float) p.getLandsPlayed() / 10f;
        return v;
    }

    private static void addCardList(List<float[]> t, List<Integer> m,
            java.util.Collection<? extends Card> cards, Zone z, Game g) {
        for (Card c : cards) {
            addCard(t, m, c, z, g);
        }
    }

    private static void addPermList(List<float[]> t, List<Integer> m,
            List<? extends Permanent> perms, Zone z, Game g) {
        for (Permanent p : perms) {
            t.add(embedCard(p, z, g));
            m.add(0);  // Don't mask permanent tokens
        }
    }

    private static void addCard(List<float[]> t, List<Integer> m,
            Card c, Zone z, Game g) {
        t.add(embedCard(c, z, g));
        m.add(0);  // Don't mask card tokens
    }

    private static float[] embedCard(Card card, Zone zone, Game game) {
        float[] v = new float[DIM_PER_TOKEN];
        int index = 0;

        // --- 1. One‑hot zone (wrap if outbound) ------------------
        v[zone.ordinal() % DIM_PER_TOKEN] = 1.0f;
        index = Zone.values().length;

        // --- 2. Owner flag --------------------------------------
        if (index < DIM_PER_TOKEN) {
            v[index++] = card.getOwnerId().equals(game.getActivePlayerId()) ? 1.0f : 0.0f;
        }

        // --- 3. Basic stats -------------------------------------
        if (index + 3 < DIM_PER_TOKEN) {
            v[index++] = (float) card.getPower().getValue();
            v[index++] = (float) card.getToughness().getValue();
            v[index++] = (float) card.getManaValue();
        }

        // --- 4. Mana cost breakdown -----------------------------
        Mana total = new Mana();
        if (!card.getManaCost().isEmpty()) {
            for (Object obj : card.getManaCost()) {
                if (obj instanceof ManaCost) {
                    ManaCost mc = (ManaCost) obj;
                    Mana m = mc.getMana();
                    total.setWhite(total.getWhite() + m.getWhite());
                    total.setBlue(total.getBlue() + m.getBlue());
                    total.setGreen(total.getGreen() + m.getGreen());
                    total.setBlack(total.getBlack() + m.getBlack());
                    total.setRed(total.getRed() + m.getRed());
                    total.setColorless(total.getColorless() + m.getColorless());
                    total.setGeneric(total.getGeneric() + m.getGeneric());
                }
            }
        } else {
            total = null;
        }
        float[] manaFields = new float[]{
            total != null ? total.getWhite() : 0.0f,
            total != null ? total.getBlue() : 0.0f,
            total != null ? total.getGreen() : 0.0f,
            total != null ? total.getBlack() : 0.0f,
            total != null ? total.getRed() : 0.0f,
            total != null ? total.getColorless() : 0.0f,
            total != null ? total.getGeneric() : 0.0f
        };
        for (int i = 0; i < manaFields.length && index < DIM_PER_TOKEN; i++) {
            v[index++] = manaFields[i];
        }

        // --- 5. Card type flags ---------------------------------
        boolean[] flags = new boolean[]{
            card.isCreature(), card.isArtifact(), card.isEnchantment(), card.isLand(),
            card.isPlaneswalker(), card.isPermanent(), card.isInstant(), card.isSorcery()
        };
        for (int i = 0; i < flags.length && index < DIM_PER_TOKEN; i++) {
            v[index++] = flags[i] ? 1.0f : 0.0f;
        }

        // --- 6. Battlefield‑only properties ---------------------
        if (zone == Zone.BATTLEFIELD && card instanceof Permanent) {
            Permanent p = (Permanent) card;
            float[] bf = new float[]{
                p.isTapped() ? 1.0f : 0.0f,
                p.isAttacking() ? 1.0f : 0.0f,
                p.isBlocked(game) ? 1.0f : 0.0f,
                p.hasSummoningSickness() ? 1.0f : 0.0f,
                (float) p.getDamage()
            };
            for (int i = 0; i < bf.length && index < DIM_PER_TOKEN; i++) {
                v[index++] = bf[i];
            }
        } else {
            int pad = Math.min(5, DIM_PER_TOKEN - index);
            for (int i = 0; i < pad; i++) {
                v[index++] = 0.0f;
            }
        }

        // --- 7. Text embedding ----------------------------------
        if (index < DIM_PER_TOKEN) {
            String cardText = "";
            try {
                cardText = org.apache.commons.lang3.StringUtils.join(card.getRules(), ' ');
            } catch (Exception e) {
                // ignore
            }
            float[] textEmb = EmbeddingManager.getEmbedding(cardText);
            for (int i = 0; i < textEmb.length && index < DIM_PER_TOKEN; i++) {
                v[index++] = textEmb[i];
            }
        }

        // any remaining slots are already 0 by default
        return v;
    }

    /**
     * Encodes an activated ability into a vector representation. The encoding
     * includes: - Source type (permanent, card, etc.) - Ability type (mana,
     * loyalty, etc.) - Cost information - Effect information
     *
     * @param ability The ability to encode
     * @param source The source of the ability
     * @param game The current game state
     * @return A float array encoding the ability
     */
    public static float[] encodeActivatedAbility(mage.abilities.Ability ability, mage.game.permanent.Permanent source, Game game) {
        float[] encoding = new float[DIM_PER_TOKEN];

        // Source type encoding (first 10 dimensions)
        if (source != null) {
            encoding[0] = 1.0f; // Is permanent
            encoding[1] = source.isCreature() ? 1.0f : 0.0f;
            encoding[2] = source.isArtifact() ? 1.0f : 0.0f;
            encoding[3] = source.isEnchantment() ? 1.0f : 0.0f;
            encoding[4] = source.isPlaneswalker() ? 1.0f : 0.0f;
            encoding[5] = source.isLand() ? 1.0f : 0.0f;
        }

        // Ability type encoding (next 10 dimensions)
        if (ability instanceof mage.abilities.mana.ManaAbility) {
            encoding[10] = 1.0f; // Is mana ability
        } else if (ability instanceof LoyaltyAbility) {
            encoding[11] = 1.0f; // Is loyalty ability
            // Get loyalty cost from PayLoyaltyCost
            for (mage.abilities.costs.Cost cost : ability.getCosts()) {
                if (cost instanceof PayLoyaltyCost) {
                    encoding[12] = ((PayLoyaltyCost) cost).getAmount() / 10.0f; // Normalize loyalty cost
                    break;
                }
            }
        } else {
            encoding[13] = 1.0f; // Is regular activated ability
        }

        // Cost encoding (next 20 dimensions)
        int costIndex = 20;
        for (mage.abilities.costs.Cost cost : ability.getCosts()) {
            if (cost instanceof mage.abilities.costs.mana.ManaCost) {
                ManaCost manaCost = (ManaCost) cost;
                encoding[costIndex] = manaCost.getMana().getGeneric() / 10.0f;
                encoding[costIndex + 1] = manaCost.getMana().getWhite() / 10.0f;
                encoding[costIndex + 2] = manaCost.getMana().getBlue() / 10.0f;
                encoding[costIndex + 3] = manaCost.getMana().getBlack() / 10.0f;
                encoding[costIndex + 4] = manaCost.getMana().getRed() / 10.0f;
                encoding[costIndex + 5] = manaCost.getMana().getGreen() / 10.0f;
                costIndex += 6;
            } else if (cost instanceof mage.abilities.costs.common.TapSourceCost) {
                encoding[costIndex] = 1.0f;
                costIndex++;
            } else if (cost instanceof mage.abilities.costs.common.SacrificeSourceCost) {
                encoding[costIndex + 1] = 1.0f;
                costIndex++;
            }
        }

        // Effect encoding (remaining dimensions)
        String abilityText = ability.toString();
        if (abilityText != null) {
            float[] textEmb = EmbeddingManager.getEmbedding(abilityText);
            // Copy the embedding into the remaining dimensions
            for (int i = 0; i < textEmb.length && (40 + i) < DIM_PER_TOKEN; i++) {
                encoding[40 + i] = textEmb[i];
            }
        }

        return encoding;
    }

    /* === SIMPLE CONTAINER ============================================== */
    public static class SequenceOutput {

        public final List<float[]> tokens;
        public final List<Integer> mask;

        public SequenceOutput(List<float[]> tokens, List<Integer> mask) {
            this.tokens = tokens;
            this.mask = mask;
        }

        public List<float[]> getSequence() {
            return this.tokens;
        }

        public List<Integer> getMask() {
            return this.mask;
        }
    }

    public static class TrainingData {

        public final SequenceOutput stateActionPair;
        public final double policyScore;
        public final double valueScore;
        public final List<Integer> actionCombo;
        public final ActionType actionType;

        public TrainingData(SequenceOutput stateActionPair, double policyScore, double valueScore,
                List<Integer> actionCombo, ActionType actionType) {
            this.stateActionPair = stateActionPair;
            this.policyScore = policyScore;
            this.valueScore = valueScore;
            this.actionCombo = actionCombo;
            this.actionType = actionType;
        }
    }
}
