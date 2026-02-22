package mage.player.ai.rl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

import mage.Mana;
import mage.abilities.Ability;
import mage.abilities.LoyaltyAbility;
import mage.abilities.Mode;
import mage.abilities.common.EntersBattlefieldTriggeredAbility;
import mage.abilities.costs.common.PayLoyaltyCost;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.effects.Effect;
import mage.abilities.effects.SearchEffect;
import mage.abilities.effects.common.CopyPermanentEffect;
import mage.abilities.effects.common.CopyTargetStackObjectEffect;
import mage.abilities.effects.common.CounterTargetEffect;
import mage.abilities.effects.common.CreateTokenEffect;
import mage.abilities.effects.common.DamagePlayersEffect;
import mage.abilities.effects.common.DamageTargetEffect;
import mage.abilities.effects.common.DestroyAllEffect;
import mage.abilities.effects.common.DestroyTargetEffect;
import mage.abilities.effects.common.DrawCardSourceControllerEffect;
import mage.abilities.effects.common.DrawCardTargetEffect;
import mage.abilities.effects.common.ExileAllEffect;
import mage.abilities.effects.common.ExileTargetEffect;
import mage.abilities.effects.common.GainLifeEffect;
import mage.abilities.effects.common.LoseLifeSourceControllerEffect;
import mage.abilities.effects.common.LoseLifeTargetEffect;
import mage.abilities.effects.common.MillCardsControllerEffect;
import mage.abilities.effects.common.MillCardsTargetEffect;
import mage.abilities.effects.common.PreventDamageToTargetEffect;
import mage.abilities.effects.common.ReturnToHandFromBattlefieldAllEffect;
import mage.abilities.effects.common.ReturnToHandTargetEffect;
import mage.abilities.effects.common.SacrificeEffect;
import mage.abilities.effects.common.SacrificeSourceEffect;
import mage.abilities.effects.common.TransformSourceEffect;
import mage.abilities.effects.common.continuous.BoostControlledEffect;
import mage.abilities.effects.common.continuous.BoostSourceEffect;
import mage.abilities.effects.common.continuous.BoostTargetEffect;
import mage.abilities.effects.common.continuous.GainAbilityControlledEffect;
import mage.abilities.effects.common.continuous.GainAbilityTargetEffect;
import mage.abilities.effects.common.cost.CostModificationEffectImpl;
import mage.abilities.effects.common.counter.AddCountersSourceEffect;
import mage.abilities.effects.common.counter.AddCountersTargetEffect;
import mage.abilities.effects.common.discard.DiscardControllerEffect;
import mage.abilities.effects.common.discard.DiscardTargetEffect;
import mage.abilities.effects.common.turn.AddExtraTurnControllerEffect;
import mage.abilities.effects.mana.ManaEffect;
import mage.abilities.keyword.DeathtouchAbility;
import mage.abilities.keyword.DefenderAbility;
import mage.abilities.keyword.DoubleStrikeAbility;
import mage.abilities.keyword.FirstStrikeAbility;
import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HasteAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.IndestructibleAbility;
import mage.abilities.keyword.LifelinkAbility;
import mage.abilities.keyword.MenaceAbility;
import mage.abilities.keyword.ReachAbility;
import mage.abilities.keyword.TrampleAbility;
import mage.abilities.keyword.VigilanceAbility;
import mage.cards.Card;
import mage.constants.TurnPhase;
import mage.constants.Zone;
import mage.counters.CounterType;
import mage.filter.common.FilterArtifactPermanent;
import mage.filter.common.FilterCreaturePermanent;
import mage.filter.common.FilterEnchantmentPermanent;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.Target;
import mage.target.common.TargetAnyTarget;
import mage.target.common.TargetCreatureOrPlayer;

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

    /**
     * Token-ID vocabulary size for learned embeddings on the Python side.
     * <p>
     * We intentionally keep this bounded (hashing trick) to avoid huge
     * embedding tables while still providing stable IDs for the local Pauper
     * milestone.
     */
    public static final int TOKEN_ID_VOCAB = 65536;

    // special‑token IDs (for simple one‑hot placeholder embedding)
    private static final int CLS_ID = 0;
    private static final int PHASE_BASE = 10; // + phase.ordinal()
    private static final int ACTION_BASE = 30; // + actionType.ordinal()

    // First slot in a stack-token vector reserved for target linkage.
    // Layout: [0..Zone.len-1] zone one-hot, [Zone.len] owner, [+3] stats,
    //         [+5] mana colors, [+8] type flags, [+5] bf-props/pad = Zone.len+22.
    // Each target takes 2 slots: has_target flag + normalized token index.
    // Supports up to 4 targets (8 slots) before hitting free space.
    private static final int STACK_TARGET_SLOT_START = Zone.values().length + 22;

    // Slots 39-50: keyword ability flags (all zones).
    // Slot 31 = STACK_TARGET_SLOT_START; 31+8=39 starts after 4 target pairs.
    private static final int KW_SLOT_START = 39;
    // Slots 51-55: extra BF-only properties (counters, attachments, blocking, token).
    private static final int EXTRA_SLOT_START = 51;
    // Slots 56-82: effect-type flags (27 flags, all zones).
    private static final int EFFECT_SLOT_START = 56;
    // Slots 83-114: pre-computed text embeddings (32 dims, filled by CardTextEmbeddings).
    private static final int TEXT_EMBED_SLOT_START = 83;
    public static final int TEXT_EMBED_DIM = 32;

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
        LONDON_MULLIGAN, // Choosing which cards to put on bottom after mulligan
        SELECT_CHOICE,
        SELECT_TRIGGERED_ABILITY,
        SELECT_CARD,
        CHOOSE_USE,          // Binary yes/no decisions (kicker, optional costs, etc.)
        DECLARE_ATTACK_TARGET, // Phase 2: choosing which defender to attack (player/PW/battle)
        CHOOSE_MODE,         // Selecting a mode from a modal spell/ability
        ANNOUNCE_X           // Announcing an X value for spells/abilities with variable X cost
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
        List<Integer> tokenIds = new ArrayList<>();
        Map<UUID, Integer> uuidMap = new HashMap<>();

        // (a) Special tokens -----------------------------------------
        tokens.add(embedSpecial(CLS_ID));
        mask.add(0);  // Don't mask CLS token
        tokenIds.add(0);
        if (phase != null) {
            tokens.add(embedSpecial(PHASE_BASE + phase.ordinal()));
        } else {
            // Special case for mulligan
            tokens.add(embedSpecial(PHASE_BASE + TurnPhase.values().length + 1));
        }
        mask.add(0);  // Don't mask phase token
        tokenIds.add(0);

        // (b) Player + opponent stats tokens -------------------------
        tokens.add(embedPlayerStats(player, game));
        mask.add(0);  // Don't mask player stats
        tokenIds.add(0);
        Player opponent = game.getPlayer(game.getOpponents(player.getId()).iterator().next());
        if (opponent == null) {
            logger.severe("No opponent found in game " + game.getId());
            throw new IllegalStateException("Cannot build state sequence: no opponent");
        }
        tokens.add(embedPlayerStats(opponent, game));
        mask.add(0);  // Don't mask opponent stats
        tokenIds.add(0);

        // (c) Variable-length entity tokens --------------------------
        // All non-stack entities are added first so the UUID map is fully
        // populated before stack objects (which may reference them as targets).
        addCardList(tokens, mask, tokenIds, uuidMap, player.getHand().getCards(game), Zone.HAND, game);
        addPermList(tokens, mask, tokenIds, uuidMap, game.getBattlefield().getAllActivePermanents(player.getId()), Zone.BATTLEFIELD, game);
        addCardList(tokens, mask, tokenIds, uuidMap, player.getGraveyard().getCards(game), Zone.GRAVEYARD, game);
        addCardList(tokens, mask, tokenIds, uuidMap, player.getLibrary().getCards(game), Zone.LIBRARY, game);

        // Opponent perms
        addPermList(tokens, mask, tokenIds, uuidMap, game.getBattlefield().getAllActivePermanents(opponent.getId()), Zone.BATTLEFIELD, game);
        addCardList(tokens, mask, tokenIds, uuidMap, opponent.getGraveyard().getCards(game), Zone.GRAVEYARD, game);

        for (ExileZone ez : game.getExile().getExileZones()) {
            addCardList(tokens, mask, tokenIds, uuidMap, ez.getCards(game), Zone.EXILED, game);
        }

        // Stack objects: added last so the UUID map is complete, enabling target
        // token index linkage (e.g., knowing which permanent a buff spell targets).
        for (StackObject so : game.getStack()) {
            if (so instanceof Card) {
                int soTokenIdx = tokens.size();
                float[] stackTok = embedCard((Card) so, Zone.STACK, game);
                // Patch target linkage into reserved slots of the stack token.
                try {
                    Ability sa = so.getStackAbility();
                    if (sa != null && sa.getModes() != null) {
                        int slot = STACK_TARGET_SLOT_START;
                        outer:
                        for (UUID modeId : sa.getModes().getSelectedModes()) {
                            Mode mode = sa.getModes().get(modeId);
                            if (mode == null) continue;
                            for (Target target : mode.getTargets()) {
                                for (UUID targetId : target.getTargets()) {
                                    Integer tIdx = uuidMap.get(targetId);
                                    if (tIdx != null && slot + 1 < DIM_PER_TOKEN) {
                                        stackTok[slot]     = 1.0f;
                                        stackTok[slot + 1] = tIdx / (float) MAX_LEN;
                                        slot += 2;
                                        if (slot + 1 >= DIM_PER_TOKEN) break outer;
                                    }
                                }
                            }
                        }
                    }
                } catch (Exception e) {
                    // Non-critical: target linkage is best-effort
                }
                uuidMap.put(((Card) so).getId(), soTokenIdx);
                tokens.add(stackTok);
                mask.add(0);
                tokenIds.add(cardTokenId((Card) so));
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
            tokenIds.add(0);
        }
        if (tokens.size() > maxLen) {
            tokens = tokens.subList(0, maxLen);
            mask = mask.subList(0, maxLen);
            tokenIds = tokenIds.subList(0, maxLen);
        }

        // Validate final sequence for NaN or Inf values
        for (int i = 0; i < tokens.size(); i++) {
            float[] token = tokens.get(i);
            for (int j = 0; j < token.length; j++) {
                if (Float.isNaN(token[j]) || Float.isInfinite(token[j])) {
                    logger.severe(String.format("Invalid value detected in final sequence at token %d, index %d: %f", i, j, token[j]));
                    token[j] = 0.0f; // Replace NaN/Inf with 0
                }
            }
        }

        return new SequenceOutput(tokens, mask, tokenIds, uuidMap);
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
        List<Integer> tokenIds = new ArrayList<>();

        // Prepend action tokens (add at beginning)
        tokens.add(embedSpecial(ACTION_BASE + actionType.ordinal()));
        tokens.add(abilityEncoding);
        mask.add(0);  // Don't mask action token
        mask.add(0);  // Don't mask ability encoding
        tokenIds.add(0);
        tokenIds.add(0);

        // Add base state tokens
        for (float[] token : baseState.tokens) {
            tokens.add(token);
        }
        for (int maskValue : baseState.mask) {
            mask.add(maskValue);
        }
        for (int tokenId : baseState.tokenIds) {
            tokenIds.add(tokenId);
        }

        return new SequenceOutput(tokens, mask, tokenIds);
    }

    /* === EMBEDDING HELPERS ============================================== */
    private static int toVocabId(String key) {
        if (key == null || key.isEmpty()) {
            return 0;
        }
        // Reserve 0 for "no-id"; map into [1, TOKEN_ID_VOCAB-1]
        int h = key.hashCode();
        int mod = Math.floorMod(h, TOKEN_ID_VOCAB - 1);
        return 1 + mod;
    }

    private static int cardTokenId(Card card) {
        // For the local Pauper milestone, name-based ID is sufficient and stable.
        // If we later want more granularity, we can incorporate expansion/number.
        return toVocabId(card != null ? card.getName() : null);
    }

    private static float[] embedSpecial(int tokenId) {
        float[] v = new float[DIM_PER_TOKEN];
        v[tokenId % DIM_PER_TOKEN] = 1.0f; // simple one‑hot placeholder
        return v;
    }

    private static float[] embedPlayerStats(Player p, Game g) {
        float[] v = new float[DIM_PER_TOKEN];

        v[0] = (float) p.getLife() / Math.max(1, g.getStartingLife());
        v[1] = p.getHand().size() / 7.0f;
        v[2] = p.getLibrary().size() / 60.0f;
        v[3] = p.getGraveyard().size() / 30.0f;
        // v[4]: lands played this turn (normalized, 0-1 scale)
        v[4] = (float) p.getLandsPlayed() / Math.max(1, p.getLandsPerTurn());
        // v[5]: can still play a land (binary: 1.0 = yes)
        v[5] = p.getLandsPlayed() < p.getLandsPerTurn() ? 1.0f : 0.0f;
        // v[6]: turn number (normalized, caps at 20)
        v[6] = Math.min(g.getTurnNum(), 20) / 20.0f;
        // v[7]: is active player (binary: 1.0 = it's this player's turn)
        v[7] = g.getActivePlayerId() != null && g.getActivePlayerId().equals(p.getId()) ? 1.0f : 0.0f;

        // v[8-13]: current mana pool contents (RGUBC)
        mage.players.ManaPool pool = p.getManaPool();
        v[8]  = pool.getRed()       / 10.0f;
        v[9]  = pool.getGreen()     / 10.0f;
        v[10] = pool.getBlue()      / 10.0f;
        v[11] = pool.getWhite()     / 10.0f;
        v[12] = pool.getBlack()     / 10.0f;
        v[13] = pool.getColorless() / 10.0f;

        // Validate normalized values
        for (int i = 0; i < 14; i++) {
            if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                logger.severe(String.format("Invalid normalized value at index %d: %f", i, v[i]));
                v[i] = 0.0f;
            }
        }

        return v;
    }

    private static void addCardList(List<float[]> t, List<Integer> m, List<Integer> ids,
            Map<UUID, Integer> uuidMap, java.util.Collection<? extends Card> cards, Zone z, Game g) {
        for (Card c : cards) {
            addCard(t, m, ids, uuidMap, c, z, g);
        }
    }

    private static void addPermList(List<float[]> t, List<Integer> m, List<Integer> ids,
            Map<UUID, Integer> uuidMap, List<? extends Permanent> perms, Zone z, Game g) {
        for (Permanent p : perms) {
            uuidMap.put(p.getId(), t.size()); // record index BEFORE appending
            t.add(embedCard(p, z, g));
            m.add(0);  // Don't mask permanent tokens
            ids.add(cardTokenId(p));
        }
    }

    private static void addCard(List<float[]> t, List<Integer> m, List<Integer> ids,
            Map<UUID, Integer> uuidMap, Card c, Zone z, Game g) {
        uuidMap.put(c.getId(), t.size()); // record index BEFORE appending
        t.add(embedCard(c, z, g));
        m.add(0);  // Don't mask card tokens
        ids.add(cardTokenId(c));
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
            // Add validation for power/toughness
            int power = card.getPower().getValue();
            int toughness = card.getToughness().getValue();
            int manaValue = card.getManaValue();

            // Ensure values are non-negative
            power = Math.max(0, power);
            toughness = Math.max(0, toughness);
            manaValue = Math.max(0, manaValue);

            v[index++] = (float) power;
            v[index++] = (float) toughness;
            v[index++] = (float) manaValue;

            // Validate values
            for (int i = index - 3; i < index; i++) {
                if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                    logger.severe(String.format("Invalid card stat value at index %d: %f", i, v[i]));
                    v[i] = 0.0f; // Replace NaN/Inf with 0
                }
            }
        }

        // --- 4. Mana cost breakdown -----------------------------
        Mana total = new Mana();
        for (ManaCost cost : card.getManaCost()) {
            total.add(cost.getMana());
        }

        // Add validation for mana values
        if (index + 5 < DIM_PER_TOKEN) {
            v[index++] = Math.max(0, total.getWhite());
            v[index++] = Math.max(0, total.getBlue());
            v[index++] = Math.max(0, total.getBlack());
            v[index++] = Math.max(0, total.getRed());
            v[index++] = Math.max(0, total.getGreen());

            // Validate mana values
            for (int i = index - 5; i < index; i++) {
                if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                    logger.severe(String.format("Invalid mana value at index %d: %f", i, v[i]));
                    v[i] = 0.0f; // Replace NaN/Inf with 0
                }
            }
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

        // --- 7. Keyword ability flags (all zones, fixed slots 39-50) -----------
        // For BF permanents use getAbilities(game) so granted/removed abilities
        // are reflected. For cards in other zones use static getAbilities().
        boolean isPerm = (zone == Zone.BATTLEFIELD && card instanceof Permanent);
        Permanent kwPerm = isPerm ? (Permanent) card : null;
        boolean[] kws = new boolean[]{
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(FlyingAbility.getInstance().getId())
                : card.getAbilities().containsClass(FlyingAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(FirstStrikeAbility.getInstance().getId())
                : card.getAbilities().containsClass(FirstStrikeAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(DoubleStrikeAbility.getInstance().getId())
                : card.getAbilities().containsClass(DoubleStrikeAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(TrampleAbility.getInstance().getId())
                : card.getAbilities().containsClass(TrampleAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(HasteAbility.getInstance().getId())
                : card.getAbilities().containsClass(HasteAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(DeathtouchAbility.getInstance().getId())
                : card.getAbilities().containsClass(DeathtouchAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(LifelinkAbility.getInstance().getId())
                : card.getAbilities().containsClass(LifelinkAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsClass(MenaceAbility.class)
                : card.getAbilities().containsClass(MenaceAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(HexproofAbility.getInstance().getId())
                : card.getAbilities().containsClass(HexproofAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(DefenderAbility.getInstance().getId())
                : card.getAbilities().containsClass(DefenderAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(ReachAbility.getInstance().getId())
                : card.getAbilities().containsClass(ReachAbility.class),
            kwPerm != null
                ? kwPerm.getAbilities(game).containsKey(VigilanceAbility.getInstance().getId())
                : card.getAbilities().containsClass(VigilanceAbility.class),
        };
        for (int i = 0; i < kws.length && KW_SLOT_START + i < DIM_PER_TOKEN; i++) {
            v[KW_SLOT_START + i] = kws[i] ? 1.0f : 0.0f;
        }

        // --- 8. Extra BF-only properties (slots 51-55) ------------------
        if (isPerm) {
            Permanent ep = (Permanent) card;
            if (EXTRA_SLOT_START < DIM_PER_TOKEN)
                v[EXTRA_SLOT_START]     = Math.min(ep.getCounters(game).getCount(CounterType.P1P1), 10) / 10.0f;
            if (EXTRA_SLOT_START + 1 < DIM_PER_TOKEN)
                v[EXTRA_SLOT_START + 1] = Math.min(ep.getCounters(game).getCount(CounterType.M1M1), 10) / 10.0f;
            if (EXTRA_SLOT_START + 2 < DIM_PER_TOKEN)
                v[EXTRA_SLOT_START + 2] = Math.min(ep.getAttachments().size(), 5) / 5.0f;
            if (EXTRA_SLOT_START + 3 < DIM_PER_TOKEN)
                v[EXTRA_SLOT_START + 3] = ep.getBlocking() > 0 ? 1.0f : 0.0f;
            if (EXTRA_SLOT_START + 4 < DIM_PER_TOKEN)
                v[EXTRA_SLOT_START + 4] = (ep instanceof PermanentToken) ? 1.0f : 0.0f;
        }

        // --- 10. Effect-type flags (slots 56-82) ----------------
        // Walk all abilities and their effects to detect functional categories.
        // Uses static getAbilities() - reflects card text across all zones.
        {
            boolean fDealsDamage = false;
            boolean fDestroys = false;
            boolean fExiles = false;
            boolean fBounces = false;
            boolean fDraws = false;
            boolean fGainsLife = false;
            boolean fLosesLife = false;
            boolean fTokens = false;
            boolean fCounters = false;
            boolean fAddCounters = false;
            boolean fSacrifices = false;
            boolean fDiscards = false;
            boolean fMills = false;
            boolean fTutors = false;
            boolean fBoostsPT = false;
            boolean fGrantsAbility = false;
            boolean fAddsMana = false;
            boolean fCostReduction = false;
            boolean fPreventsDamage = false;
            boolean fExtraTurn = false;
            boolean fCopies = false;
            boolean fTransforms = false;
            boolean fTargetsAny = false;
            boolean fTargetsCreature = false;
            boolean fTargetsArtifact = false;
            boolean fTargetsEnchantment = false;
            boolean fHasETB = false;

            Iterable<Ability> abilities = isPerm
                    ? kwPerm.getAbilities(game)
                    : card.getAbilities();

            for (Ability ability : abilities) {
                if (ability instanceof EntersBattlefieldTriggeredAbility) {
                    fHasETB = true;
                }
                for (Effect effect : ability.getAllEffects()) {
                    if (effect instanceof DamageTargetEffect || effect instanceof DamagePlayersEffect)
                        fDealsDamage = true;
                    if (effect instanceof DestroyTargetEffect || effect instanceof DestroyAllEffect)
                        fDestroys = true;
                    if (effect instanceof ExileTargetEffect || effect instanceof ExileAllEffect)
                        fExiles = true;
                    if (effect instanceof ReturnToHandTargetEffect || effect instanceof ReturnToHandFromBattlefieldAllEffect)
                        fBounces = true;
                    if (effect instanceof DrawCardSourceControllerEffect || effect instanceof DrawCardTargetEffect)
                        fDraws = true;
                    if (effect instanceof GainLifeEffect)
                        fGainsLife = true;
                    if (effect instanceof LoseLifeTargetEffect || effect instanceof LoseLifeSourceControllerEffect)
                        fLosesLife = true;
                    if (effect instanceof CreateTokenEffect)
                        fTokens = true;
                    if (effect instanceof CounterTargetEffect)
                        fCounters = true;
                    if (effect instanceof AddCountersTargetEffect || effect instanceof AddCountersSourceEffect)
                        fAddCounters = true;
                    if (effect instanceof SacrificeEffect || effect instanceof SacrificeSourceEffect)
                        fSacrifices = true;
                    if (effect instanceof DiscardTargetEffect || effect instanceof DiscardControllerEffect)
                        fDiscards = true;
                    if (effect instanceof MillCardsTargetEffect || effect instanceof MillCardsControllerEffect)
                        fMills = true;
                    if (effect instanceof SearchEffect)
                        fTutors = true;
                    if (effect instanceof BoostTargetEffect || effect instanceof BoostSourceEffect || effect instanceof BoostControlledEffect)
                        fBoostsPT = true;
                    if (effect instanceof GainAbilityTargetEffect || effect instanceof GainAbilityControlledEffect)
                        fGrantsAbility = true;
                    if (effect instanceof ManaEffect)
                        fAddsMana = true;
                    if (effect instanceof CostModificationEffectImpl)
                        fCostReduction = true;
                    if (effect instanceof PreventDamageToTargetEffect)
                        fPreventsDamage = true;
                    if (effect instanceof AddExtraTurnControllerEffect)
                        fExtraTurn = true;
                    if (effect instanceof CopyTargetStackObjectEffect || effect instanceof CopyPermanentEffect)
                        fCopies = true;
                    if (effect instanceof TransformSourceEffect)
                        fTransforms = true;
                }
                for (Target target : ability.getTargets()) {
                    if (target instanceof TargetAnyTarget || target instanceof TargetCreatureOrPlayer)
                        fTargetsAny = true;
                    if (target.getFilter() instanceof FilterCreaturePermanent)
                        fTargetsCreature = true;
                    if (target.getFilter() instanceof FilterArtifactPermanent)
                        fTargetsArtifact = true;
                    if (target.getFilter() instanceof FilterEnchantmentPermanent)
                        fTargetsEnchantment = true;
                }
            }

            boolean[] ef = {
                fDealsDamage, fDestroys, fExiles, fBounces, fDraws, fGainsLife, fLosesLife,
                fTokens, fCounters, fAddCounters, fSacrifices, fDiscards, fMills, fTutors,
                fBoostsPT, fGrantsAbility, fAddsMana, fCostReduction, fPreventsDamage,
                fExtraTurn, fCopies, fTransforms,
                fTargetsAny, fTargetsCreature, fTargetsArtifact, fTargetsEnchantment, fHasETB
            };
            for (int i = 0; i < ef.length && EFFECT_SLOT_START + i < DIM_PER_TOKEN; i++) {
                v[EFFECT_SLOT_START + i] = ef[i] ? 1.0f : 0.0f;
            }
        }

        // --- 11. Pre-computed text embeddings (slots 83-114) ----
        // Filled by CardTextEmbeddings singleton if a card_embeddings.json is present
        // for the current profile. Falls back to zeros if not available.
        {
            float[] textEmbed = CardTextEmbeddings.getInstance().getEmbedding(card.getName());
            for (int i = 0; i < TEXT_EMBED_DIM && TEXT_EMBED_SLOT_START + i < DIM_PER_TOKEN; i++) {
                v[TEXT_EMBED_SLOT_START + i] = textEmbed[i];
            }
        }

        // Validate final vector
        for (int i = 0; i < v.length; i++) {
            if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                logger.severe(String.format("Invalid value in final vector for card %s at index %d: %f",
                        card.getName(), i, v[i]));
                v[i] = 0.0f;
            }
        }

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
                    // Add validation for loyalty cost
                    int loyaltyCost = ((PayLoyaltyCost) cost).getAmount();
                    encoding[12] = Math.max(0, loyaltyCost) / 10.0f; // Normalize loyalty cost
                    break;
                }
            }
        } else {
            encoding[13] = 1.0f; // Is regular activated ability
        }

        // Cost encoding (next 20 dimensions)
        int costIndex = 20;
        for (mage.abilities.costs.Cost cost : ability.getCosts()) {
            if (cost instanceof ManaCost) {
                Mana mana = ((ManaCost) cost).getMana();
                // Add validation for mana values
                encoding[costIndex++] = Math.max(0, mana.getWhite());
                encoding[costIndex++] = Math.max(0, mana.getBlue());
                encoding[costIndex++] = Math.max(0, mana.getBlack());
                encoding[costIndex++] = Math.max(0, mana.getRed());
                encoding[costIndex++] = Math.max(0, mana.getGreen());
                encoding[costIndex++] = Math.max(0, mana.getColorless());
            }
        }

        // Effect encoding (remaining dimensions)
        // For the local Pauper milestone we run fully offline:
        // - Do NOT embed ability text via external APIs.
        // - Candidate/action IDs will be embedded on the Python side once we switch
        //   to action-conditional scoring.
        // Validate final vector
        for (int i = 0; i < encoding.length; i++) {
            if (Float.isNaN(encoding[i]) || Float.isInfinite(encoding[i])) {
                logger.severe(String.format("Invalid value in ability encoding at index %d: %f", i, encoding[i]));
                encoding[i] = 0.0f; // Replace NaN/Inf with 0
            }
        }

        return encoding;
    }

    /* === SIMPLE CONTAINER ============================================== */
    public static class SequenceOutput implements Serializable {

        private static final long serialVersionUID = 2L;

        public final float[][] tokens;
        public final int[] mask;
        public final int[] tokenIds;
        /** Maps each card/permanent UUID to its token index in the sequence. */
        public final Map<UUID, Integer> uuidToTokenIndex;

        public SequenceOutput(List<float[]> tokenList, List<Integer> maskList,
                List<Integer> tokenIdsList, Map<UUID, Integer> uuidToTokenIndex) {
            this.tokens = tokenList.toArray(new float[tokenList.size()][]);
            this.mask = maskList.stream().mapToInt(Integer::intValue).toArray();
            this.tokenIds = tokenIdsList.stream().mapToInt(Integer::intValue).toArray();
            this.uuidToTokenIndex = Collections.unmodifiableMap(new HashMap<>(uuidToTokenIndex));
        }

        /** Backward-compatible constructor (no UUID map). */
        public SequenceOutput(List<float[]> tokenList, List<Integer> maskList, List<Integer> tokenIdsList) {
            this(tokenList, maskList, tokenIdsList, Collections.emptyMap());
        }

        public float[][] getSequence() {
            return this.tokens;
        }

        public int[] getMask() {
            return this.mask;
        }

        public int[] getTokenIds() {
            return this.tokenIds;
        }
    }

    public static class TrainingData implements Serializable {

        private static final long serialVersionUID = 1L;

        /**
         * Max number of candidates we will score per decision step.
         * <p>
         * NOTE: This is a fixed-size padding limit for the Java<->Python
         * bridge. We can tune it later via env/config once the pipeline is
         * stable.
         */
        public static final int MAX_CANDIDATES = 64;

        /**
         * Fixed-dimensional numeric feature vector per candidate.
         */
        public static final int CAND_FEAT_DIM = 48;

        public final SequenceOutput state;
        public final int candidateCount;
        public final int[] candidateActionIds;     // [MAX_CANDIDATES]
        public final float[][] candidateFeatures;  // [MAX_CANDIDATES][CAND_FEAT_DIM]
        public final int[] candidateMask;          // [MAX_CANDIDATES] 1=valid,0=pad
        public final int chosenCount;              // number of picks in chosenIndices (<= MAX_CANDIDATES)
        public final int[] chosenIndices;          // [MAX_CANDIDATES], padded with -1
        public final float oldLogpTotal;           // joint log-prob of chosenIndices under behavior policy
        public final float oldValue;               // V_old(s) at rollout time
        public final ActionType actionType;
        public final double stepReward;

        public TrainingData(SequenceOutput state,
                int candidateCount,
                int[] candidateActionIds,
                float[][] candidateFeatures,
                int[] candidateMask,
                int chosenCount,
                int[] chosenIndices,
                float oldLogpTotal,
                float oldValue,
                ActionType actionType,
                double stepReward) {
            this.state = state;
            this.candidateCount = candidateCount;
            this.candidateActionIds = candidateActionIds;
            this.candidateFeatures = candidateFeatures;
            this.candidateMask = candidateMask;
            this.chosenCount = chosenCount;
            this.chosenIndices = chosenIndices;
            this.oldLogpTotal = oldLogpTotal;
            this.oldValue = oldValue;
            this.actionType = actionType;
            this.stepReward = stepReward;
        }
    }
}
