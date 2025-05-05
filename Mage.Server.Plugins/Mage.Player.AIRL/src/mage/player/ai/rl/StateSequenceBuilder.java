package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import mage.Mana;
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
 * Java‑8‑compatible helper that converts a {@link Game} snapshot into a padded
 * token sequence + attention mask for a Transformer encoder.
 * <p>
 * <ul>
 * <li>Each token = {@code float[D]} feature vector.</li>
 * <li>{@code sequence} tensor shape: {@code [1, D, maxLen]} (DL4J format).</li>
 * <li>{@code mask} tensor shape: {@code [1, maxLen]} &nbsp;(&#x201C;1&#x201D;
 * =&nbsp;real, &#x201C;0&#x201D; =&nbsp;padding).</li>
 * </ul>
 */
public class StateSequenceBuilder {

    private static final Logger logger = Logger.getLogger(StateSequenceBuilder.class.getName());

    /* === CONFIGURATION ================================================= */

    public static final int DIM_PER_TOKEN = 128; // feature dim per token
    public static final int MAX_LEN = 256; // hard cap for sequence length

    // special‑token IDs (for simple one‑hot placeholder embedding)
    private static final int CLS_ID = 0;
    private static final int PHASE_BASE = 10; // + phase.ordinal()
    private static final int ASK_BASE = 30; // + actionType.ordinal()

    public enum ZoneType {
        HAND,
        BATTLEFIELD,
        GRAVEYARD,
        EXILE,
        LIBRARY,
        STACK,
        // This is a special case for the source card
        REFERENCE
    }

    public enum ActionType {
        ACTIVATE_ABILITY_OR_SPELL,
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
     * Build sequence + mask for the given game state.
     *
     * @param game       current game
     * @param actionType what decision the agent is being asked to make
     * @param phase      current phase of the game
     * @param maxLen     padding / truncation length (≤ {@link #MAX_LEN})
     * @param numOptions number of available options
     * @return           sequence + mask
     */
    public static SequenceOutput build(Game game, ActionType actionType, TurnPhase phase, int maxLen, int numOptions) {
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
        mask.add(1);
        if (phase != null){
            tokens.add(embedSpecial(PHASE_BASE + phase.ordinal()));
        } else {
            // Special case for mulligan
            tokens.add(embedSpecial(PHASE_BASE + TurnPhase.values().length + 1));
        }
        mask.add(1);
        tokens.add(embedSpecial(ASK_BASE + actionType.ordinal()));
        mask.add(1);

        // (b) Player + opponent stats tokens -------------------------
        tokens.add(embedPlayerStats(player, game));
        mask.add(1);
        Player opponent = game.getPlayer(game.getOpponents(player.getId()).iterator().next());
        if (opponent == null) {
            logger.severe("No opponent found in game " + game.getId());
            throw new IllegalStateException("Cannot build state sequence: no opponent");
        }
        tokens.add(embedPlayerStats(opponent, game));
        mask.add(1);

        // (c) Variable-length entity tokens --------------------------
        addCardList(tokens, mask, player.getHand().getCards(game), Zone.HAND, game);
        addPermList(tokens, mask, game.getBattlefield().getAllActivePermanents(player.getId()), Zone.BATTLEFIELD, game);
        addCardList(tokens, mask, player.getGraveyard().getCards(game), Zone.GRAVEYARD, game);
        addCardList(tokens, mask, player.getLibrary().getCards(game), Zone.LIBRARY, game);

        // Opponent perms
        addPermList(tokens, mask, game.getBattlefield().getAllActivePermanents(opponent.getId()), Zone.BATTLEFIELD,
                game);
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
            mask.add(0);
        }
        if (tokens.size() > maxLen) {
            tokens = tokens.subList(0, maxLen);
            mask = mask.subList(0, maxLen);
        }

        /*
         * ------------------------------------------------------------
         * 4. Convert to NDArrays (DL4J expects [B, D, L])
         * ----------------------------------------------------------
         */
        float[][] seq2d = new float[tokens.size()][DIM_PER_TOKEN];
        for (int i = 0; i < tokens.size(); i++) {
            seq2d[i] = tokens.get(i);
        }
        INDArray seq = Nd4j.create(seq2d).transpose(); // [D, L]
        seq = seq.reshape(1, DIM_PER_TOKEN, maxLen); // batch dim =1

        // Ensure the sequence is padded to maxLen
        if (tokens.size() < maxLen) {
            INDArray padded = Nd4j.zeros(1, DIM_PER_TOKEN, maxLen);
            padded.putSlice(0, seq);
            seq = padded;
        } else if (tokens.size() > maxLen) {
            System.out.println("Truncating sequence from " + tokens.size() + " to " + maxLen);
            seq = seq.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, maxLen));
        }

        float[] maskFloats = new float[maxLen]; // Always use maxLen for mask
        for (int i = 0; i < mask.size(); i++) {
            maskFloats[i] = mask.get(i);
        }
        // Pad the rest with zeros
        for (int i = mask.size(); i < maxLen; i++) {
            maskFloats[i] = 0;
        }
        INDArray maskArr = Nd4j.create(maskFloats).reshape(1, maxLen);

        return new SequenceOutput(seq, maskArr, new ArrayList<>(), numOptions, mapActionTypeToAskType(actionType));
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
            m.add(1);
        }
    }

    private static void addCard(List<float[]> t, List<Integer> m,
            Card c, Zone z, Game g) {
        t.add(embedCard(c, z, g));
        m.add(1);
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
        float[] manaFields = new float[] {
                total != null ? total.getWhite()      : 0.0f,
                total != null ? total.getBlue()       : 0.0f,
                total != null ? total.getGreen()      : 0.0f,
                total != null ? total.getBlack()      : 0.0f,
                total != null ? total.getRed()        : 0.0f,
                total != null ? total.getColorless()  : 0.0f,
                total != null ? total.getGeneric()    : 0.0f
        };
        for (int i = 0; i < manaFields.length && index < DIM_PER_TOKEN; i++) {
            v[index++] = manaFields[i];
        }

        // --- 5. Card type flags ---------------------------------
        boolean[] flags = new boolean[] {
                card.isCreature(), card.isArtifact(), card.isEnchantment(), card.isLand(),
                card.isPlaneswalker(), card.isPermanent(), card.isInstant(), card.isSorcery()
        };
        for (int i = 0; i < flags.length && index < DIM_PER_TOKEN; i++) {
            v[index++] = flags[i] ? 1.0f : 0.0f;
        }

        // --- 6. Battlefield‑only properties ---------------------
        if (zone == Zone.BATTLEFIELD && card instanceof Permanent) {
            Permanent p = (Permanent) card;
            float[] bf = new float[] {
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

    private static TransformerNeuralNetwork.AskType mapActionTypeToAskType(ActionType actionType) {
        switch (actionType) {
            case ACTIVATE_ABILITY_OR_SPELL:
            case SELECT_TARGETS:
            case SELECT_CHOICE:
            case SELECT_CARD:
                return TransformerNeuralNetwork.AskType.CAST;
            case DECLARE_ATTACKS:
                return TransformerNeuralNetwork.AskType.ATTACK;
            case DECLARE_BLOCKS:
                return TransformerNeuralNetwork.AskType.BLOCK;
            case MULLIGAN:
            case SELECT_TRIGGERED_ABILITY:
            default:
                return TransformerNeuralNetwork.AskType.CAST;
        }
    }

    /* === SIMPLE CONTAINER (Java 8 version) =============================== */
    public static class SequenceOutput {
        public final INDArray sequence;
        public final INDArray mask;
        public final List<Integer> targetIndices;
        public final int numOptions;
        public INDArray currentQValues;
        public final TransformerNeuralNetwork.AskType askType;

        public SequenceOutput(INDArray sequence, INDArray mask, List<Integer> targetIndices, int numOptions, TransformerNeuralNetwork.AskType askType) {
            this.sequence = sequence;
            this.mask = mask;
            this.targetIndices = targetIndices;
            this.numOptions = numOptions;
            this.currentQValues = null;
            this.askType = askType;
        }

        // Copy constructor that preserves Q-values
        public SequenceOutput(SequenceOutput original, List<Integer> newTargetIndices) {
            this.sequence = original.sequence;
            this.mask = original.mask;
            this.targetIndices = newTargetIndices;
            this.currentQValues = original.currentQValues;
            this.numOptions = original.numOptions;
            this.askType = original.askType;
        }

        public INDArray getSequence() {
            return sequence;
        }

        public INDArray getMask() {
            return mask;
        }

        public List<Integer> getTargetIndices() {
            return targetIndices;
        }

        public int getNumOptions() {
            return numOptions;
        }
    }
}
