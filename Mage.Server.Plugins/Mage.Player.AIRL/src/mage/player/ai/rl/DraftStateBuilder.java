package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import mage.Mana;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.keyword.DeathtouchAbility;
import mage.abilities.keyword.DefenderAbility;
import mage.abilities.keyword.DoubleStrikeAbility;
import mage.abilities.keyword.FirstStrikeAbility;
import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HasteAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.LifelinkAbility;
import mage.abilities.keyword.MenaceAbility;
import mage.abilities.keyword.ReachAbility;
import mage.abilities.keyword.TrampleAbility;
import mage.abilities.keyword.VigilanceAbility;
import mage.cards.Card;

/**
 * Encodes draft state into float token sequences for the draft Transformer model.
 *
 * State sequence layout:
 *   [CLS] [DraftMeta] [PoolCard_1..P] [SeenCard_1..S]
 *
 * DIM_PER_TOKEN = 32:
 *   [0]    zone flag: is pool card (1=pool, 0=seen)
 *   [1]    zone flag: is seen card (0=pool, 1=seen)
 *   [2]    pack number (1-3, normalized to 0-1)
 *   [3]    pick number (1-15, normalized to 0-1)
 *   [4-10] type flags: creature, instant, sorcery, enchantment, artifact, land, planeswalker
 *   [11-17] mana cost: W, U, B, R, G, C, CMC (all normalized)
 *   [18-19] power, toughness (normalized, 0 for non-creatures)
 *   [20-31] keyword abilities: flying, trample, haste, deathtouch, lifelink,
 *           first strike, double strike, reach, vigilance, menace, defender, hexproof
 *
 * CLS token: all zeros (special token)
 * DraftMeta token uses:
 *   [0] = pack number normalized
 *   [1] = pick number normalized
 *   [2-6] = WUBRG color density in current pool (0-1)
 *   [7-13] = CMC distribution in pool (0,1,2,3,4,5,6+ normalized)
 */
public class DraftStateBuilder {

    private static final Logger logger = Logger.getLogger(DraftStateBuilder.class.getName());

    public static final int DIM_PER_TOKEN = 32;
    public static final int MAX_LEN = 420;
    public static final int TOKEN_ID_VOCAB = 65536; // same hashing vocab as game model

    // Max cards per category
    private static final int MAX_POOL = 45;
    private static final int MAX_SEEN = 315;

    /**
     * Output of state encoding: token sequence + attention mask + token IDs.
     */
    public static class DraftStateOutput {
        public final float[] sequence;  // [MAX_LEN * DIM_PER_TOKEN] flat float array
        public final int[] mask;        // [MAX_LEN] 0=real, 1=padding
        public final int[] tokenIds;    // [MAX_LEN] vocab IDs for embedding table
        public final int seqLen;

        public DraftStateOutput(float[] sequence, int[] mask, int[] tokenIds, int seqLen) {
            this.sequence = sequence;
            this.mask = mask;
            this.tokenIds = tokenIds;
            this.seqLen = seqLen;
        }
    }

    /**
     * Build the state sequence for a draft decision point.
     *
     * @param poolCards       cards already drafted (in pool)
     * @param seenCards       cards seen but not picked, with pack/pick metadata
     * @param currentPackNum  1-3
     * @param currentPickNum  1-15
     * @return encoded state
     */
    public static DraftStateOutput buildState(
            List<Card> poolCards,
            List<DraftPickRecord> seenCards,
            int currentPackNum,
            int currentPickNum
    ) {
        List<float[]> tokens = new ArrayList<>();
        List<Integer> mask = new ArrayList<>();
        List<Integer> tokenIds = new ArrayList<>();

        // --- CLS token ---
        tokens.add(new float[DIM_PER_TOKEN]);
        mask.add(0);
        tokenIds.add(0);

        // --- DraftMeta token ---
        tokens.add(buildMetaToken(poolCards, currentPackNum, currentPickNum));
        mask.add(0);
        tokenIds.add(0);

        // --- Pool card tokens ---
        int poolCount = Math.min(poolCards.size(), MAX_POOL);
        for (int i = 0; i < poolCount; i++) {
            Card card = poolCards.get(i);
            tokens.add(buildCardToken(card, true, currentPackNum, currentPickNum));
            mask.add(0);
            tokenIds.add(cardTokenId(card));
        }

        // --- Seen-but-not-picked tokens ---
        int seenCount = Math.min(seenCards.size(), MAX_SEEN);
        for (int i = 0; i < seenCount; i++) {
            DraftPickRecord rec = seenCards.get(i);
            tokens.add(buildCardToken(rec.card, false, rec.packNumber, rec.pickNumber));
            mask.add(0);
            tokenIds.add(cardTokenId(rec.card));
        }

        // --- Pad to MAX_LEN ---
        while (tokens.size() < MAX_LEN) {
            tokens.add(new float[DIM_PER_TOKEN]);
            mask.add(1);
            tokenIds.add(0);
        }
        if (tokens.size() > MAX_LEN) {
            tokens = tokens.subList(0, MAX_LEN);
            mask = mask.subList(0, MAX_LEN);
            tokenIds = tokenIds.subList(0, MAX_LEN);
        }

        // Flatten
        float[] flatSeq = new float[MAX_LEN * DIM_PER_TOKEN];
        for (int i = 0; i < MAX_LEN; i++) {
            float[] tok = tokens.get(i);
            System.arraycopy(tok, 0, flatSeq, i * DIM_PER_TOKEN, DIM_PER_TOKEN);
        }
        int[] flatMask = mask.stream().mapToInt(Integer::intValue).toArray();
        int[] flatIds = tokenIds.stream().mapToInt(Integer::intValue).toArray();

        return new DraftStateOutput(flatSeq, flatMask, flatIds, tokens.size());
    }

    /**
     * Encodes a pack of cards as candidate features for the pick head.
     *
     * @param packCards cards currently in the pack
     * @return flat float array [numCards * DIM_PER_TOKEN]
     */
    public static float[] buildCandidateFeatures(List<Card> packCards) {
        float[] features = new float[packCards.size() * DIM_PER_TOKEN];
        for (int i = 0; i < packCards.size(); i++) {
            float[] tok = buildCardToken(packCards.get(i), false, 0, 0);
            System.arraycopy(tok, 0, features, i * DIM_PER_TOKEN, DIM_PER_TOKEN);
        }
        return features;
    }

    /**
     * Returns the token ID for each candidate card (for embedding lookup).
     */
    public static int[] buildCandidateIds(List<Card> packCards) {
        int[] ids = new int[packCards.size()];
        for (int i = 0; i < packCards.size(); i++) {
            ids[i] = cardTokenId(packCards.get(i));
        }
        return ids;
    }

    private static float[] buildMetaToken(List<Card> pool, int packNum, int pickNum) {
        float[] v = new float[DIM_PER_TOKEN];
        v[0] = (packNum - 1) / 2.0f;       // pack 1-3 → 0, 0.5, 1
        v[1] = (pickNum - 1) / 14.0f;      // pick 1-15 → 0..1

        // Color density: total colored mana symbols in pool / pool size
        int w = 0, u = 0, b = 0, r = 0, g = 0;
        for (Card card : pool) {
            Mana mana = totalMana(card);
            w += mana.getWhite();
            u += mana.getBlue();
            b += mana.getBlack();
            r += mana.getRed();
            g += mana.getGreen();
        }
        int total = w + u + b + r + g;
        if (total > 0) {
            v[2] = w / (float) total;
            v[3] = u / (float) total;
            v[4] = b / (float) total;
            v[5] = r / (float) total;
            v[6] = g / (float) total;
        }

        // CMC distribution: count of cards at each CMC bucket
        int[] cmcBuckets = new int[7]; // 0,1,2,3,4,5,6+
        for (Card card : pool) {
            int cmc = Math.min(6, (int) card.getManaValue());
            cmcBuckets[cmc]++;
        }
        int poolSize = Math.max(1, pool.size());
        for (int i = 0; i < 7 && 7 + i < DIM_PER_TOKEN; i++) {
            v[7 + i] = cmcBuckets[i] / (float) poolSize;
        }

        return v;
    }

    private static float[] buildCardToken(Card card, boolean isPool, int packNum, int pickNum) {
        float[] v = new float[DIM_PER_TOKEN];

        // Zone flags [0-1]
        v[0] = isPool ? 1.0f : 0.0f;
        v[1] = isPool ? 0.0f : 1.0f;

        // Pack / pick metadata [2-3]
        v[2] = packNum > 0 ? (packNum - 1) / 2.0f : 0.0f;
        v[3] = pickNum > 0 ? (pickNum - 1) / 14.0f : 0.0f;

        // Type flags [4-10]: creature, instant, sorcery, enchantment, artifact, land, planeswalker
        v[4]  = card.isCreature()     ? 1.0f : 0.0f;
        v[5]  = card.isInstant()      ? 1.0f : 0.0f;
        v[6]  = card.isSorcery()      ? 1.0f : 0.0f;
        v[7]  = card.isEnchantment()  ? 1.0f : 0.0f;
        v[8]  = card.isArtifact()     ? 1.0f : 0.0f;
        v[9]  = card.isLand()         ? 1.0f : 0.0f;
        v[10] = card.isPlaneswalker() ? 1.0f : 0.0f;

        // Mana cost [11-17]: W, U, B, R, G, C, CMC
        Mana mana = totalMana(card);
        float cmc = (float) card.getManaValue();
        float cmcNorm = Math.min(cmc / 10.0f, 1.0f);
        v[11] = Math.min(mana.getWhite() / 5.0f, 1.0f);
        v[12] = Math.min(mana.getBlue()  / 5.0f, 1.0f);
        v[13] = Math.min(mana.getBlack() / 5.0f, 1.0f);
        v[14] = Math.min(mana.getRed()   / 5.0f, 1.0f);
        v[15] = Math.min(mana.getGreen() / 5.0f, 1.0f);
        v[16] = Math.min(mana.getColorless() / 5.0f, 1.0f);
        v[17] = cmcNorm;

        // Power / toughness [18-19]
        if (card.isCreature()) {
            try {
                String ps = card.getPower().toString();
                String ts = card.getToughness().toString();
                int power = parseInt(ps, 0);
                int toughness = parseInt(ts, 0);
                v[18] = Math.min(Math.max(power, 0) / 10.0f, 1.0f);
                v[19] = Math.min(Math.max(toughness, 0) / 10.0f, 1.0f);
            } catch (Exception e) {
                // leave as 0
            }
        }

        // Keyword abilities [20-31]
        v[20] = card.getAbilities().containsClass(FlyingAbility.class)      ? 1.0f : 0.0f;
        v[21] = card.getAbilities().containsClass(TrampleAbility.class)     ? 1.0f : 0.0f;
        v[22] = card.getAbilities().containsClass(HasteAbility.class)       ? 1.0f : 0.0f;
        v[23] = card.getAbilities().containsClass(DeathtouchAbility.class)  ? 1.0f : 0.0f;
        v[24] = card.getAbilities().containsClass(LifelinkAbility.class)    ? 1.0f : 0.0f;
        v[25] = card.getAbilities().containsClass(FirstStrikeAbility.class) ? 1.0f : 0.0f;
        v[26] = card.getAbilities().containsClass(DoubleStrikeAbility.class)? 1.0f : 0.0f;
        v[27] = card.getAbilities().containsClass(ReachAbility.class)       ? 1.0f : 0.0f;
        v[28] = card.getAbilities().containsClass(VigilanceAbility.class)   ? 1.0f : 0.0f;
        v[29] = card.getAbilities().containsClass(MenaceAbility.class)      ? 1.0f : 0.0f;
        v[30] = card.getAbilities().containsClass(DefenderAbility.class)    ? 1.0f : 0.0f;
        v[31] = card.getAbilities().containsClass(HexproofAbility.class)    ? 1.0f : 0.0f;

        return v;
    }

    private static Mana totalMana(Card card) {
        Mana total = new Mana();
        try {
            for (ManaCost cost : card.getManaCost()) {
                total.add(cost.getMana());
            }
        } catch (Exception e) {
            // ignore
        }
        return total;
    }

    private static int parseInt(String s, int def) {
        if (s == null) return def;
        try {
            return Integer.parseInt(s.trim().replaceAll("[^0-9\\-]", ""));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** Hash card name to a vocab ID in [1, TOKEN_ID_VOCAB). */
    public static int cardTokenId(Card card) {
        if (card == null) return 0;
        String key = card.getName().toLowerCase().replace(" ", "_");
        return (key.hashCode() & 0x7FFF_FFFF) % TOKEN_ID_VOCAB + 1;
    }
}
