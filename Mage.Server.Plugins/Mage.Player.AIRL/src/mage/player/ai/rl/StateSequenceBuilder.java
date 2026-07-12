package mage.player.ai.rl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

import mage.Mana;
import mage.MageObject;
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
import mage.abilities.effects.common.DamageAllEffect;
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
import mage.abilities.keyword.AffinityForArtifactsAbility;
import mage.abilities.keyword.DeathtouchAbility;
import mage.abilities.keyword.DefenderAbility;
import mage.abilities.keyword.DoubleStrikeAbility;
import mage.abilities.keyword.FirstStrikeAbility;
import mage.abilities.keyword.FlashAbility;
import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HasteAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.IndestructibleAbility;
import mage.abilities.keyword.LifelinkAbility;
import mage.abilities.keyword.MenaceAbility;
import mage.abilities.keyword.ProwessAbility;
import mage.abilities.keyword.ReachAbility;
import mage.abilities.keyword.TrampleAbility;
import mage.abilities.keyword.VigilanceAbility;
import mage.abilities.keyword.WardAbility;
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
    private static final java.util.concurrent.atomic.AtomicLong TRUNCATION_COUNT = new java.util.concurrent.atomic.AtomicLong(0);

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
    private static final boolean ZONE_COUNT_FEATURES_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_ZONE_COUNT_FEATURES_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ZONE_COUNT_FEATURES_ENABLE", "0"));
    private static final boolean LIBRARY_COUNT_FEATURES_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_LIBRARY_COUNT_FEATURES_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_LIBRARY_COUNT_FEATURES_ENABLE", "0"));
    // One-shot engagement check: prove the zone-count scalars are actually nonzero at runtime
    // (RL_ZONE_COUNT_DIAG=1). Guards against silently-zero features when flags don't propagate.
    private static final boolean ZONE_COUNT_DIAG =
            "1".equals(System.getenv().getOrDefault("RL_ZONE_COUNT_DIAG", "0"));
    private static final java.util.concurrent.atomic.AtomicInteger ZONE_COUNT_DIAG_N =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private static final boolean PUBLIC_BOARD_FEATURES_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_PUBLIC_BOARD_FEATURES_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_PUBLIC_BOARD_FEATURES_ENABLE", "0"));
    // Designation-state features (initiative/Undercity + monarch): public symmetric game
    // state the engine tracks but the encoder was blind to. Required for decks whose win
    // path runs through the initiative (e.g. Avenging Hunter). Opt-in: old models were
    // trained with these slots at zero. Slots v[23-26] on the player-stats token.
    private static final boolean INITIATIVE_FEATURES_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_INITIATIVE_FEATURES_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_INITIATIVE_FEATURES_ENABLE", "0"));
    private static final int INITIATIVE_SLOT_START = 23;
    // One-shot engagement check (RL_INITIATIVE_DIAG=1): prints the first few nonzero
    // designation-feature vectors to prove the slots engage at runtime.
    private static final boolean INITIATIVE_DIAG =
            "1".equals(System.getenv().getOrDefault("RL_INITIATIVE_DIAG", "0"));
    private static final java.util.concurrent.atomic.AtomicInteger INITIATIVE_DIAG_N =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private static final int PUBLIC_BOARD_SLOT_START = 56;
    private static final int PUBLIC_BOARD_SLOT_COUNT = 12;
    private static final boolean EXTENDED_EFFECT_FLAGS_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_EXTENDED_EFFECT_FLAGS_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_EXTENDED_EFFECT_FLAGS_ENABLE", "0"));
    private static final boolean ARCHETYPE_BELIEF_FEATURES_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_ARCHETYPE_BELIEF_FEATURES_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_ARCHETYPE_BELIEF_FEATURES_ENABLE", "0"));
    private static final int ARCHETYPE_BELIEF_SLOT_START = 40;
    private static final DeterminizationSampler ARCHETYPE_BELIEF_SAMPLER =
            ARCHETYPE_BELIEF_FEATURES_ENABLED ? loadArchetypeBeliefSampler() : null;
    private static final boolean TRUE_ARCHETYPE_LABELS_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_BELIEF_TRUE_ARCHETYPE_LABELS_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_BELIEF_TRUE_ARCHETYPE_LABELS_ENABLE", "0"));
    private static final ThreadLocal<Map<UUID, Integer>> THREAD_LOCAL_KNOWN_ARCHETYPE_LABELS = new ThreadLocal<>();
    private static final boolean CARD_BELIEF_LABELS_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_CARD_BELIEF_LABELS_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_CARD_BELIEF_LABELS_ENABLE", "0"));
    // Self-supervised world-model labels: predict the agent's OWN future
    // zone-counts at fixed decision horizons. Thesis-clean (future-state, not
    // a value/good-bad belief). Opt-in; old models trained with slots absent.
    private static final boolean WORLD_MODEL_LABELS_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_WORLD_MODEL_LABELS_ENABLE", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_WORLD_MODEL_LABELS_ENABLE", "0"));
    public static final int WORLD_MODEL_FEATURES = 6;
    public static final int[] WORLD_MODEL_HORIZONS = {8, 24, 64};
    private static final int CARD_BELIEF_MAX_CARDS =
            Math.max(0, parseEnvInt("RL_CARD_BELIEF_MAX_CARDS", 256));
    private static final CardBeliefVocab CARD_BELIEF_VOCAB =
            CARD_BELIEF_LABELS_ENABLED ? loadCardBeliefVocab() : CardBeliefVocab.empty();

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
        UUID choosingPlayerId = game.getState() == null ? null : game.getState().getChoosingPlayerId();
        Player player = choosingPlayerId == null ? null : game.getPlayer(choosingPlayerId);
        if (player == null) {
            player = game.getPlayer(game.getActivePlayerId());
        }
        if (player == null) {
            logger.severe("No perspective player found in game " + game.getId());
            throw new IllegalStateException("Cannot build state sequence: no perspective player");
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
        // Include non-card stack abilities so target-pick decisions can "see" triggered
        // abilities currently on stack, not only spells.
        for (StackObject so : game.getStack()) {
            int soTokenIdx = tokens.size();
            float[] stackTok;
            int tokenId;
            if (so instanceof Card) {
                stackTok = embedCard((Card) so, Zone.STACK, game);
                tokenId = cardTokenId((Card) so);
            } else {
                stackTok = embedStackObject(so, game);
                tokenId = stackObjectTokenId(so);
            }
            patchStackTargetLinks(stackTok, so.getStackAbility(), uuidMap);
            uuidMap.put(so.getId(), soTokenIdx);
            tokens.add(stackTok);
            mask.add(0);
            tokenIds.add(tokenId);
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
            // Log truncation for diagnostics (sampled to avoid spam)
            long truncCount = TRUNCATION_COUNT.incrementAndGet();
            if (truncCount <= 5 || truncCount % 1000 == 0) {
                System.out.println("[STATE] TRUNCATION: " + tokens.size() + " tokens -> " + maxLen
                    + " (count=" + truncCount + ")");
            }
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

    /**
     * Return a sequence whose player-stats token carries compact, generic
     * recent-action history. This does not add tokens or shift entity indexes,
     * so candidate token-link features remain valid.
     */
    public static SequenceOutput withActionHistoryFeatures(SequenceOutput baseState, float[] historyFeatures) {
        if (baseState == null || historyFeatures == null || historyFeatures.length == 0
                || baseState.tokens == null || baseState.tokens.length <= 2) {
            return baseState;
        }
        List<float[]> tokens = new ArrayList<>(baseState.tokens.length);
        Collections.addAll(tokens, baseState.tokens);
        float[] playerStats = tokens.get(2).clone();
        int start = 23;
        int limit = Math.min(historyFeatures.length, playerStats.length - start);
        for (int i = 0; i < limit; i++) {
            float value = historyFeatures[i];
            if (Float.isNaN(value) || Float.isInfinite(value)) {
                value = 0.0f;
            }
            playerStats[start + i] = value;
        }
        tokens.set(2, playerStats);

        List<Integer> mask = new ArrayList<>(baseState.mask.length);
        for (int value : baseState.mask) {
            mask.add(value);
        }
        List<Integer> tokenIds = new ArrayList<>(baseState.tokenIds.length);
        for (int tokenId : baseState.tokenIds) {
            tokenIds.add(tokenId);
        }
        return new SequenceOutput(tokens, mask, tokenIds, baseState.uuidToTokenIndex);
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

        // v[14]: artifact count on battlefield (normalized, for Metalcraft/Affinity)
        int artifactCount = 0;
        int elfCount = 0;
        int creatureCount = 0;
        for (Permanent perm : g.getBattlefield().getAllActivePermanents(p.getId())) {
            if (perm.isArtifact(g)) artifactCount++;
            if (perm.isCreature(g)) creatureCount++;
            if (perm.hasSubtype(mage.constants.SubType.ELF, g)) elfCount++;
        }
        v[14] = artifactCount / 10.0f;
        v[15] = elfCount / 10.0f;
        v[16] = creatureCount / 10.0f;

        if (PUBLIC_BOARD_FEATURES_ENABLED) {
            writePublicBoardFeatures(v, p, g);
        }

        // Optional generic zone-count features. These expose counts the player
        // can derive from known zones/deck contents without adding deck-specific
        // rules, rewards, or action constraints. Kept opt-in because old models
        // were trained with these slots at zero.
        if (ZONE_COUNT_FEATURES_ENABLED || LIBRARY_COUNT_FEATURES_ENABLED) {
            int handLands = 0;
            int handCreatures = 0;
            if (ZONE_COUNT_FEATURES_ENABLED) {
                for (Card c : p.getHand().getCards(g)) {
                    if (c.isLand(g)) handLands++;
                    if (c.isCreature(g)) handCreatures++;
                }
            }
            int libraryLands = 0;
            int libraryCreatures = 0;
            for (Card c : p.getLibrary().getCards(g)) {
                if (c.isLand(g)) libraryLands++;
                if (c.isCreature(g)) libraryCreatures++;
            }
            int graveyardLands = 0;
            int graveyardCreatures = 0;
            if (ZONE_COUNT_FEATURES_ENABLED) {
                for (Card c : p.getGraveyard().getCards(g)) {
                    if (c.isLand(g)) graveyardLands++;
                    if (c.isCreature(g)) graveyardCreatures++;
                }
            }
            if (ZONE_COUNT_FEATURES_ENABLED) {
                v[17] = handLands / 10.0f;
                v[18] = handCreatures / 10.0f;
            }
            v[19] = libraryLands / 20.0f;
            v[20] = libraryCreatures / 60.0f;
            if (ZONE_COUNT_FEATURES_ENABLED) {
                v[21] = graveyardLands / 20.0f;
                v[22] = graveyardCreatures / 30.0f;
            }
            if (ZONE_COUNT_DIAG && ZONE_COUNT_DIAG_N.getAndIncrement() < 6) {
                System.err.println("[ZONE_COUNT_DIAG] libLands(v19)=" + v[19] + " libCreat(v20)=" + v[20]
                        + " gyCreat(v22)=" + v[22] + " libSize=" + p.getLibrary().size());
            }
        }
        if (INITIATIVE_FEATURES_ENABLED) {
            // v[23]: this player holds the initiative; v[24]: has ventured (dungeon in
            // progress); v[25]: current room is terminal (payoff room reached);
            // v[26]: this player is the monarch. All public, symmetric for both players.
            try {
                v[INITIATIVE_SLOT_START] = p.getId().equals(g.getInitiativeId()) ? 1.0f : 0.0f;
                mage.game.command.Dungeon dungeon = g.getPlayerDungeon(p.getId());
                if (dungeon != null) {
                    v[INITIATIVE_SLOT_START + 1] = 1.0f;
                    mage.game.command.DungeonRoom room = dungeon.getCurrentRoom();
                    v[INITIATIVE_SLOT_START + 2] = (room != null && !room.hasNextRoom()) ? 1.0f : 0.0f;
                }
                v[INITIATIVE_SLOT_START + 3] = p.getId().equals(g.getMonarchId()) ? 1.0f : 0.0f;
                if (INITIATIVE_DIAG
                        && (v[INITIATIVE_SLOT_START] > 0 || v[INITIATIVE_SLOT_START + 1] > 0
                            || v[INITIATIVE_SLOT_START + 3] > 0)
                        && INITIATIVE_DIAG_N.getAndIncrement() < 6) {
                    System.err.println("[INITIATIVE_DIAG] hasInitiative(v23)=" + v[23]
                            + " ventured(v24)=" + v[24] + " terminalRoom(v25)=" + v[25]
                            + " monarch(v26)=" + v[26] + " turn=" + g.getTurnNum());
                }
            } catch (Throwable ignored) {
                // Designation features are optional observation inputs; never let a
                // transient game-state issue disrupt gameplay.
            }
        }
        if (ARCHETYPE_BELIEF_FEATURES_ENABLED && ARCHETYPE_BELIEF_SAMPLER != null) {
            try {
                Map<String, Float> posterior = ARCHETYPE_BELIEF_SAMPLER.classifyArchetype(g, p.getId());
                List<String> archetypes = ARCHETYPE_BELIEF_SAMPLER.getArchetypes();
                int limit = Math.min(TrainingData.NUM_ARCHETYPES, archetypes.size());
                for (int i = 0; i < limit && ARCHETYPE_BELIEF_SLOT_START + i < v.length; i++) {
                    Float prob = posterior.get(archetypes.get(i));
                    v[ARCHETYPE_BELIEF_SLOT_START + i] = prob == null ? 0.0f : sanitizeUnit(prob);
                }
            } catch (Throwable ignored) {
                // Belief features are optional diagnostics/training inputs; never
                // let a decklist or transient game-state issue disrupt gameplay.
            }
        }

        // Validate normalized values
        int validateLimit = INITIATIVE_FEATURES_ENABLED ? 27 : 23;
        if (PUBLIC_BOARD_FEATURES_ENABLED) {
            validateLimit = Math.max(validateLimit,
                    Math.min(v.length, PUBLIC_BOARD_SLOT_START + PUBLIC_BOARD_SLOT_COUNT));
        }
        if (ARCHETYPE_BELIEF_FEATURES_ENABLED) {
            validateLimit = Math.max(validateLimit,
                    Math.min(v.length, ARCHETYPE_BELIEF_SLOT_START + TrainingData.NUM_ARCHETYPES));
        }
        for (int i = 0; i < validateLimit; i++) {
            if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                logger.severe(String.format("Invalid normalized value at index %d: %f", i, v[i]));
                v[i] = 0.0f;
            }
        }

        return v;
    }

    private static void writePublicBoardFeatures(float[] v, Player p, Game g) {
        int permanents = 0;
        int nonlandPermanents = 0;
        int untappedCreatures = 0;
        int tappedCreatures = 0;
        int attackingCreatures = 0;
        int blockingCreatures = 0;
        int summoningSickCreatures = 0;
        int tokenPermanents = 0;
        int totalPower = 0;
        int untappedPower = 0;
        int attackingPower = 0;
        int maxPower = 0;

        for (Permanent perm : g.getBattlefield().getAllActivePermanents(p.getId())) {
            permanents++;
            if (!perm.isLand(g)) {
                nonlandPermanents++;
            }
            if (perm instanceof PermanentToken) {
                tokenPermanents++;
            }
            if (!perm.isCreature(g)) {
                continue;
            }
            int power = Math.max(0, perm.getPower().getValue());
            totalPower += power;
            maxPower = Math.max(maxPower, power);
            if (perm.isTapped()) {
                tappedCreatures++;
            } else {
                untappedCreatures++;
                untappedPower += power;
            }
            if (perm.isAttacking()) {
                attackingCreatures++;
                attackingPower += power;
            }
            if (perm.isBlocked(g)) {
                blockingCreatures++;
            }
            if (perm.hasSummoningSickness()) {
                summoningSickCreatures++;
            }
        }

        int slot = PUBLIC_BOARD_SLOT_START;
        if (slot < v.length) v[slot++] = permanents / 20.0f;
        if (slot < v.length) v[slot++] = nonlandPermanents / 20.0f;
        if (slot < v.length) v[slot++] = untappedCreatures / 20.0f;
        if (slot < v.length) v[slot++] = tappedCreatures / 20.0f;
        if (slot < v.length) v[slot++] = attackingCreatures / 10.0f;
        if (slot < v.length) v[slot++] = blockingCreatures / 10.0f;
        if (slot < v.length) v[slot++] = summoningSickCreatures / 10.0f;
        if (slot < v.length) v[slot++] = tokenPermanents / 20.0f;
        if (slot < v.length) v[slot++] = totalPower / 30.0f;
        if (slot < v.length) v[slot++] = untappedPower / 30.0f;
        if (slot < v.length) v[slot++] = attackingPower / 30.0f;
        if (slot < v.length) v[slot] = maxPower / 15.0f;
    }

    private static DeterminizationSampler loadArchetypeBeliefSampler() {
        try {
            return DeterminizationSampler.pauperDefaults();
        } catch (Throwable t) {
            logger.warning("RL_ARCHETYPE_BELIEF_FEATURES_ENABLE requested but sampler failed to load: "
                    + t.getMessage());
            return null;
        }
    }

    private static float sanitizeUnit(Float value) {
        if (value == null || Float.isNaN(value) || Float.isInfinite(value)) {
            return 0.0f;
        }
        if (value < 0.0f) {
            return 0.0f;
        }
        if (value > 1.0f) {
            return 1.0f;
        }
        return value;
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

    private static int stackObjectTokenId(StackObject stackObject) {
        if (stackObject == null) {
            return 0;
        }
        try {
            Ability stackAbility = stackObject.getStackAbility();
            String name = stackObject.getName() != null ? stackObject.getName() : "unknown";
            String kind = stackAbility != null
                    ? stackAbility.getClass().getSimpleName()
                    : stackObject.getClass().getSimpleName();
            return toVocabId("STACK:" + name + ":" + kind);
        } catch (Exception e) {
            return toVocabId("STACK:unknown");
        }
    }

    private static void patchStackTargetLinks(float[] stackTok, Ability stackAbility, Map<UUID, Integer> uuidMap) {
        if (stackTok == null || stackAbility == null || uuidMap == null) {
            return;
        }
        try {
            int slot = STACK_TARGET_SLOT_START;
            boolean linkedAny = false;
            if (stackAbility.getModes() != null) {
                List<Mode> modesToScan = new ArrayList<>();
                if (stackAbility.getModes().getSelectedModes() != null
                        && !stackAbility.getModes().getSelectedModes().isEmpty()) {
                    for (UUID modeId : stackAbility.getModes().getSelectedModes()) {
                        Mode mode = stackAbility.getModes().get(modeId);
                        if (mode != null) {
                            modesToScan.add(mode);
                        }
                    }
                } else {
                    modesToScan.addAll(stackAbility.getModes().values());
                }
                for (Mode mode : modesToScan) {
                    slot = writeTargetLinks(stackTok, slot, mode.getTargets(), uuidMap);
                    if (slot > STACK_TARGET_SLOT_START) {
                        linkedAny = true;
                    }
                    if (slot + 1 >= DIM_PER_TOKEN) {
                        return;
                    }
                }
            }
            // Some abilities expose chosen targets directly on the root ability.
            if (!linkedAny && stackAbility.getTargets() != null) {
                writeTargetLinks(stackTok, slot, stackAbility.getTargets(), uuidMap);
            }
        } catch (Exception e) {
            // Non-critical: target linkage is best-effort.
        }
    }

    private static int writeTargetLinks(float[] stackTok, int slot, Iterable<Target> targets, Map<UUID, Integer> uuidMap) {
        if (targets == null) {
            return slot;
        }
        for (Target target : targets) {
            if (target == null || target.getTargets() == null) {
                continue;
            }
            for (UUID targetId : target.getTargets()) {
                if (slot + 1 >= DIM_PER_TOKEN) {
                    return slot;
                }
                Integer targetTokenIdx = uuidMap.get(targetId);
                if (targetTokenIdx != null) {
                    stackTok[slot] = 1.0f;
                    stackTok[slot + 1] = targetTokenIdx / (float) MAX_LEN;
                    slot += 2;
                }
            }
        }
        return slot;
    }

    private static float[] embedStackObject(StackObject stackObject, Game game) {
        float[] v = new float[DIM_PER_TOKEN];
        Ability stackAbility = stackObject != null ? stackObject.getStackAbility() : null;
        Card sourceCard = null;

        try {
            if (stackObject != null && stackObject.getSourceId() != null) {
                MageObject sourceObj = game.getObject(stackObject.getSourceId());
                if (sourceObj instanceof Card) {
                    sourceCard = (Card) sourceObj;
                }
            }
        } catch (Exception ignored) {
            // Best effort only.
        }

        if (sourceCard != null) {
            v = embedCard(sourceCard, Zone.STACK, game);
        } else {
            v[Zone.STACK.ordinal() % DIM_PER_TOKEN] = 1.0f;
            int index = Zone.values().length;
            if (index < DIM_PER_TOKEN) {
                UUID controllerId = stackObject != null ? stackObject.getControllerId() : null;
                v[index++] = controllerId != null && controllerId.equals(game.getActivePlayerId()) ? 1.0f : 0.0f;
            }

            // Basic stats slots are not meaningful for non-card stack objects.
            if (index + 3 < DIM_PER_TOKEN) {
                v[index++] = 0.0f;
                v[index++] = 0.0f;
                int mv = (stackAbility != null && stackAbility.getManaCostsToPay() != null)
                        ? Math.max(0, stackAbility.getManaCostsToPay().manaValue())
                        : 0;
                v[index++] = mv / 10.0f;
            }

            Mana total = new Mana();
            if (stackAbility != null && stackAbility.getManaCostsToPay() != null) {
                for (ManaCost cost : stackAbility.getManaCostsToPay()) {
                    total.add(cost.getMana());
                }
            }
            if (index + 5 < DIM_PER_TOKEN) {
                v[index++] = Math.max(0, total.getWhite()) / 10.0f;
                v[index++] = Math.max(0, total.getBlue()) / 10.0f;
                v[index++] = Math.max(0, total.getBlack()) / 10.0f;
                v[index++] = Math.max(0, total.getRed()) / 10.0f;
                v[index++] = Math.max(0, total.getGreen()) / 10.0f;
            }
        }

        applyAbilitySemanticFlags(v, stackAbility);

        if (stackObject != null && stackObject.getName() != null) {
            float[] textEmbed = CardTextEmbeddings.getInstance().getEmbedding(stackObject.getName());
            for (int i = 0; i < TEXT_EMBED_DIM && TEXT_EMBED_SLOT_START + i < DIM_PER_TOKEN; i++) {
                v[TEXT_EMBED_SLOT_START + i] = textEmbed[i];
            }
        }

        for (int i = 0; i < v.length; i++) {
            if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                v[i] = 0.0f;
            }
        }
        return v;
    }

    private static void applyAbilitySemanticFlags(float[] v, Ability ability) {
        if (v == null || ability == null) {
            return;
        }
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
        boolean fHasETB = ability instanceof EntersBattlefieldTriggeredAbility;

        for (Effect effect : ability.getAllEffects()) {
            if (isDamageEffect(effect)) fDealsDamage = true;
            if (effect instanceof DestroyTargetEffect || effect instanceof DestroyAllEffect) fDestroys = true;
            if (isExileEffect(effect)) fExiles = true;
            if (effect instanceof ReturnToHandTargetEffect || effect instanceof ReturnToHandFromBattlefieldAllEffect) fBounces = true;
            if (effect instanceof DrawCardSourceControllerEffect || effect instanceof DrawCardTargetEffect) fDraws = true;
            if (effect instanceof GainLifeEffect) fGainsLife = true;
            if (effect instanceof LoseLifeTargetEffect || effect instanceof LoseLifeSourceControllerEffect) fLosesLife = true;
            if (effect instanceof CreateTokenEffect) fTokens = true;
            if (effect instanceof CounterTargetEffect) fCounters = true;
            if (effect instanceof AddCountersTargetEffect || effect instanceof AddCountersSourceEffect) fAddCounters = true;
            if (effect instanceof SacrificeEffect || effect instanceof SacrificeSourceEffect) fSacrifices = true;
            if (effect instanceof DiscardTargetEffect || effect instanceof DiscardControllerEffect) fDiscards = true;
            if (effect instanceof MillCardsTargetEffect || effect instanceof MillCardsControllerEffect) fMills = true;
            if (effect instanceof SearchEffect) fTutors = true;
            if (effect instanceof BoostTargetEffect || effect instanceof BoostSourceEffect || effect instanceof BoostControlledEffect) fBoostsPT = true;
            if (effect instanceof GainAbilityTargetEffect || effect instanceof GainAbilityControlledEffect) fGrantsAbility = true;
            if (effect instanceof ManaEffect) fAddsMana = true;
            if (effect instanceof CostModificationEffectImpl) fCostReduction = true;
            if (effect instanceof PreventDamageToTargetEffect) fPreventsDamage = true;
            if (effect instanceof AddExtraTurnControllerEffect) fExtraTurn = true;
            if (effect instanceof CopyTargetStackObjectEffect || effect instanceof CopyPermanentEffect) fCopies = true;
            if (effect instanceof TransformSourceEffect) fTransforms = true;
        }

        for (Target target : ability.getTargets()) {
            if (target instanceof TargetAnyTarget || target instanceof TargetCreatureOrPlayer) fTargetsAny = true;
            if (target.getFilter() instanceof FilterCreaturePermanent) fTargetsCreature = true;
            if (target.getFilter() instanceof FilterArtifactPermanent) fTargetsArtifact = true;
            if (target.getFilter() instanceof FilterEnchantmentPermanent) fTargetsEnchantment = true;
        }

        boolean[] ef = {
            fDealsDamage, fDestroys, fExiles, fBounces, fDraws, fGainsLife, fLosesLife,
            fTokens, fCounters, fAddCounters, fSacrifices, fDiscards, fMills, fTutors,
            fBoostsPT, fGrantsAbility, fAddsMana, fCostReduction, fPreventsDamage,
            fExtraTurn, fCopies, fTransforms,
            fTargetsAny, fTargetsCreature, fTargetsArtifact, fTargetsEnchantment, fHasETB
        };
        for (int i = 0; i < ef.length && EFFECT_SLOT_START + i < DIM_PER_TOKEN; i++) {
            // Merge (OR) with existing flags so stack abilities add to card-level flags
            // rather than overwriting them
            if (ef[i]) v[EFFECT_SLOT_START + i] = 1.0f;
        }
    }

    private static boolean isDamageEffect(Effect effect) {
        if (effect == null) {
            return false;
        }
        if (effect instanceof DamageTargetEffect
                || effect instanceof DamagePlayersEffect) {
            return true;
        }
        if (!EXTENDED_EFFECT_FLAGS_ENABLED) {
            return false;
        }
        if (effect instanceof DamageAllEffect) {
            return true;
        }
        String simpleName = effect.getClass().getSimpleName();
        return simpleName != null && simpleName.startsWith("Damage");
    }

    private static boolean isExileEffect(Effect effect) {
        if (effect == null) {
            return false;
        }
        if (effect instanceof ExileTargetEffect || effect instanceof ExileAllEffect) {
            return true;
        }
        if (!EXTENDED_EFFECT_FLAGS_ENABLED) {
            return false;
        }
        String simpleName = effect.getClass().getSimpleName();
        if (simpleName == null) {
            return false;
        }
        String lower = simpleName.toLowerCase(Locale.ROOT);
        return simpleName.startsWith("Exile")
                || lower.contains("andexile")
                || (lower.contains("graveyard") && lower.contains("exile"));
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

            v[index++] = power / 10.0f;
            v[index++] = toughness / 10.0f;
            v[index++] = manaValue / 10.0f;

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
            v[index++] = Math.max(0, total.getWhite()) / 10.0f;
            v[index++] = Math.max(0, total.getBlue()) / 10.0f;
            v[index++] = Math.max(0, total.getBlack()) / 10.0f;
            v[index++] = Math.max(0, total.getRed()) / 10.0f;
            v[index++] = Math.max(0, total.getGreen()) / 10.0f;

            // Validate mana values
            for (int i = index - 5; i < index; i++) {
                if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
                    logger.severe(String.format("Invalid mana value at index %d: %f", i, v[i]));
                    v[i] = 0.0f; // Replace NaN/Inf with 0
                }
            }
        }

        // --- 5. Card type flags ---------------------------------
        // Use game-aware type checks for permanents (reflects continuous effects)
        boolean[] flags = new boolean[]{
            card.isCreature(game), card.isArtifact(game), card.isEnchantment(game), card.isLand(game),
            card.isPlaneswalker(game), card.isPermanent(), card.isInstant(game), card.isSorcery(game)
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
                p.getDamage() / 10.0f
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

        // --- 7b. Extended keywords + subtype flags (free slots 115+) ---
        int extSlot = 115;
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] = (kwPerm != null
            ? kwPerm.getAbilities(game).containsKey(IndestructibleAbility.getInstance().getId())
            : card.getAbilities().containsClass(IndestructibleAbility.class)) ? 1.0f : 0.0f;
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] = (kwPerm != null
            ? kwPerm.getAbilities(game).containsKey(FlashAbility.getInstance().getId())
            : card.getAbilities().containsClass(FlashAbility.class)) ? 1.0f : 0.0f;
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] =
            card.getAbilities().containsClass(AffinityForArtifactsAbility.class) ? 1.0f : 0.0f;
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] =
            card.getAbilities().containsClass(WardAbility.class) ? 1.0f : 0.0f;
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] =
            card.getAbilities().containsClass(ProwessAbility.class) ? 1.0f : 0.0f;
        // Elf subtype
        if (extSlot < DIM_PER_TOKEN) v[extSlot++] =
            card.hasSubtype(mage.constants.SubType.ELF, game) ? 1.0f : 0.0f;
        // Controller flag (distinct from owner at slot 9)
        if (extSlot < DIM_PER_TOKEN && isPerm) {
            v[extSlot] = ((Permanent) card).getControllerId().equals(game.getActivePlayerId()) ? 1.0f : 0.0f;
        }
        extSlot++;

        // --- 7c. Temporal features (BF permanents only) ----------
        // These fill previously-zero slots 122+ so existing models keep loading
        // but acquire tempo signal over further training.
        //   122: turns-on-battlefield (normalized, caps at 10 turns)
        //   123: controlled-from-start-of-controller-turn (complement of summoning
        //        sickness: "has been under controller's control since start of their turn")
        //   124: attacking-this-combat (redundant with BF slot 25 but explicit)
        //   125: has-counters (boolean: any counters at all -- supports proliferate,
        //        +1/+1 synergies, etc.)
        if (isPerm) {
            Permanent tp = (Permanent) card;
            if (extSlot < DIM_PER_TOKEN) {
                v[extSlot++] = Math.min(tp.getTurnsOnBattlefield(), 10) / 10.0f;
            }
            if (extSlot < DIM_PER_TOKEN) {
                // hasSummoningSickness returns true when NOT controlled since start of turn;
                // store the complement so "1.0 = ready to act (can attack/tap for mana)".
                v[extSlot++] = tp.hasSummoningSickness() ? 0.0f : 1.0f;
            }
            if (extSlot < DIM_PER_TOKEN) {
                v[extSlot++] = tp.isAttacking() ? 1.0f : 0.0f;
            }
            if (extSlot < DIM_PER_TOKEN) {
                v[extSlot++] = tp.getCounters(game).size() > 0 ? 1.0f : 0.0f;
            }
        } else {
            extSlot = Math.min(DIM_PER_TOKEN, extSlot + 4);
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
                    if (isDamageEffect(effect))
                        fDealsDamage = true;
                    if (effect instanceof DestroyTargetEffect || effect instanceof DestroyAllEffect)
                        fDestroys = true;
                    if (isExileEffect(effect))
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

        /**
         * Number of deck archetypes predicted by the belief head (Phase 1).
         * Order must match {@link #computeArchetypeLabel} and
         * {@link DeterminizationSampler#pauperDefaults()}:
         *   0 = Wildfire, 1 = Rally, 2 = Affinity, 3 = Elves,
         *   4 = SpyCombo, 5 = Burn, 6 = Terror, 7 = CawGates, 8 = Faeries.
         */
        public static final int NUM_ARCHETYPES = 9;

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

        // Phase 1 belief-loss ground truth: opponent's deck archetype id,
        // or -1 if unknown (python side skips the loss when label < 0).
        public int beliefArchetypeLabel;

        // AlphaZero-style policy distillation target: MCTS visit distribution
        // over the candidate list, normalized to sum to 1.0 across valid slots.
        // All-zero array signals "no MCTS target for this step" -- python skips
        // the KL-divergence loss in that case and falls back to PPO only.
        public float[] mctsVisitTargets;  // [MAX_CANDIDATES]

        // Generic hidden-card belief target: normalized opponent hand+library
        // counts over the deck-pool vocabulary. Null means absent; the bridge
        // serializes absent rows as -1 so Python skips them.
        public float[] cardBeliefLabels;

        // Self-supervised world-model: present-state snapshot of the agent's
        // OWN zone-counts (WORLD_MODEL_FEATURES) captured at this decision.
        // Transient -- RLTrainer reads future snapshots to fill worldModelLabels.
        public float[] worldModelSnapshot;
        // Back-filled target: future snapshots at WORLD_MODEL_HORIZONS flattened
        // (features x horizons). Null/-1 means horizon ran past the episode end;
        // the bridge serializes absent rows as -1 so Python skips them.
        public float[] worldModelLabels;

        // Window-gated self-imitation eligibility: true if this decision is in
        // the combo finisher window (set by the agent from game state, e.g. a
        // large graveyard = post-Spy-mill). Restricts SIL to the short finisher
        // horizon so it doesn't over-fit whole winning trajectories (SIL v1).
        public boolean silEligible;

        // Per-row multiplier on the scheduled entropy coefficient. Population
        // league mode sets this per episode by opponent source (anchor vs
        // population) for source-conditional exploration pressure.
        public float entropyScale = 1.0f;

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
            this.beliefArchetypeLabel = -1;  // unknown until populated
            this.mctsVisitTargets = new float[MAX_CANDIDATES];
            // -2 sentinel = no target; signed candidate-Q targets live in [-1,1] and the Python
            // mask (>= -1.0) excludes the sentinel. A zero fill would train Q toward 0 on every
            // non-search step under CANDIDATE_Q_MCTS_SIGNED_TARGETS=1.
            java.util.Arrays.fill(this.mctsVisitTargets, -2.0f);
            this.cardBeliefLabels = null;
            this.worldModelSnapshot = null;
            this.worldModelLabels = null;
            this.silEligible = false;
        }

        public void setBeliefArchetypeLabel(int label) {
            this.beliefArchetypeLabel = label;
        }

        public void setMctsVisitTargets(float[] targets) {
            if (targets != null && targets.length == MAX_CANDIDATES) {
                this.mctsVisitTargets = targets;
            }
        }

        public void setCardBeliefLabels(float[] labels) {
            int dim = StateSequenceBuilder.cardBeliefDim();
            if (labels != null && dim > 0 && labels.length == dim) {
                this.cardBeliefLabels = labels;
            }
        }

        public void setWorldModelLabels(float[] labels) {
            int dim = StateSequenceBuilder.worldModelDim();
            if (labels != null && dim > 0 && labels.length == dim) {
                this.worldModelLabels = labels;
            }
        }
    }

    private static final class CardBeliefVocab {
        final String[] cardNames;
        final Map<String, Integer> indexByName;
        final float[] maxCounts;

        CardBeliefVocab(String[] cardNames, Map<String, Integer> indexByName, float[] maxCounts) {
            this.cardNames = cardNames;
            this.indexByName = indexByName;
            this.maxCounts = maxCounts;
        }

        static CardBeliefVocab empty() {
            return new CardBeliefVocab(new String[0], Collections.emptyMap(), new float[0]);
        }
    }

    public static int cardBeliefDim() {
        return CARD_BELIEF_VOCAB.cardNames.length;
    }

    /** World-model label dimension = features x horizons (0 when disabled). */
    public static int worldModelDim() {
        return WORLD_MODEL_LABELS_ENABLED
                ? WORLD_MODEL_FEATURES * WORLD_MODEL_HORIZONS.length
                : 0;
    }

    /**
     * Present-state snapshot of the agent's OWN observable zone-counts, in the
     * exact order the world-model head predicts. Normalized to roughly [0,1].
     * Captured per decision; RLTrainer back-fills future snapshots as labels.
     */
    public static float[] worldModelSnapshot(Player p, Game g) {
        float[] s = new float[WORLD_MODEL_FEATURES];
        if (p == null || g == null) {
            return s;
        }
        int gyCreatures = 0;
        for (Card c : p.getGraveyard().getCards(g)) {
            if (c.isCreature(g)) {
                gyCreatures++;
            }
        }
        int boardCreatures = 0;
        for (Permanent perm : g.getBattlefield().getAllActivePermanents(p.getId())) {
            if (perm.isCreature(g)) {
                boardCreatures++;
            }
        }
        s[0] = p.getLibrary().size() / 60.0f;
        s[1] = p.getGraveyard().size() / 30.0f;
        s[2] = gyCreatures / 30.0f;
        s[3] = p.getHand().size() / 7.0f;
        s[4] = (float) p.getLife() / Math.max(1, g.getStartingLife());
        s[5] = boardCreatures / 10.0f;
        return s;
    }

    public static List<String> cardBeliefVocab() {
        List<String> names = new ArrayList<>();
        Collections.addAll(names, CARD_BELIEF_VOCAB.cardNames);
        return Collections.unmodifiableList(names);
    }

    public static float[] cardBeliefMaxCounts() {
        return CARD_BELIEF_VOCAB.maxCounts.clone();
    }

    // Signature cards that uniquely identify each archetype. Ordered to match
    // TrainingData.NUM_ARCHETYPES and DeterminizationSampler.pauperDefaults().
    // If any of these appears anywhere in the opponent's zones, we label the
    // opponent with the corresponding archetype id.
    private static final String[][] ARCHETYPE_SIGNATURE_CARDS = {
            {"Cleansing Wildfire"},                              // Wildfire (Jund Wildfire: cleansing-target artifact lands)
            {"Kuldotha Rebirth", "Goblin Bushwhacker"},         // Rally (Mono Red Rally: aggro go-wide)
            {"Refurbished Familiar", "Drossforge Bridge"},      // Affinity (Grixis Affinity: artifact synergy)
            {"Nettle Sentinel", "Elvish Mystic", "Lys Alana Huntmaster"}, // Elves (tribal)
            {"Balustrade Spy", "Dread Return", "Lotleth Giant"}, // SpyCombo
            {"Guttersnipe", "Fireblast", "Lava Dart"},          // Burn
            {"Tolarian Terror", "Cryptic Serpent"},              // Terror
            {"Basilisk Gate", "Squadron Hawk", "The Modern Age"}, // CawGates
            {"Spellstutter Sprite", "Faerie Seer", "Moon-Circuit Hacker"} // Faeries
    };

    public static void setThreadLocalKnownArchetypeLabels(Map<UUID, Integer> labels) {
        if (labels == null || labels.isEmpty()) {
            THREAD_LOCAL_KNOWN_ARCHETYPE_LABELS.remove();
            return;
        }
        THREAD_LOCAL_KNOWN_ARCHETYPE_LABELS.set(new HashMap<>(labels));
    }

    public static void clearThreadLocalKnownArchetypeLabels() {
        THREAD_LOCAL_KNOWN_ARCHETYPE_LABELS.remove();
    }

    public static int computeArchetypeLabelFromDeckName(String deckName) {
        String normalized = deckName == null ? "" : deckName.toLowerCase(java.util.Locale.ROOT);
        if (normalized.contains("wildfire")) return 0;
        if (normalized.contains("rally")) return 1;
        if (normalized.contains("affinity")) return 2;
        if (normalized.contains("elves")) return 3;
        if (normalized.contains("spy")) return 4;
        if (normalized.contains("burn")) return 5;
        if (normalized.contains("terror")) return 6;
        if (normalized.contains("caw")) return 7;
        if (normalized.contains("faerie")) return 8;
        return -1;
    }

    /**
     * Determine the opponent's deck archetype from public information only.
     * Returns the archetype id (0..NUM_ARCHETYPES-1), or -1 if no visible
     * signature matches (e.g., unknown/off-meta deck; python skips loss).
     */
    public static int computeArchetypeLabel(Game game, UUID viewerId) {
        if (game == null || viewerId == null) return -1;
        UUID oppId = null;
        for (UUID pid : game.getOpponents(viewerId)) {
            oppId = pid;
            break;
        }
        if (oppId == null) return -1;

        if (TRUE_ARCHETYPE_LABELS_ENABLED) {
            Map<UUID, Integer> knownLabels = THREAD_LOCAL_KNOWN_ARCHETYPE_LABELS.get();
            if (knownLabels != null) {
                Integer known = knownLabels.get(oppId);
                if (known != null && known >= 0 && known < TrainingData.NUM_ARCHETYPES) {
                    return known;
                }
            }
        }

        Player opp = game.getPlayer(oppId);
        if (opp == null) return -1;

        java.util.Set<String> names = new java.util.HashSet<>();
        try {
            for (Permanent p : game.getBattlefield().getAllActivePermanents(oppId)) if (p != null) names.add(p.getName());
            for (Card c : opp.getGraveyard().getCards(game)) if (c != null) names.add(c.getName());
            for (Card c : game.getExile().getAllCards(game)) {
                if (c != null && oppId.equals(c.getOwnerId())) {
                    names.add(c.getName());
                }
            }
            for (StackObject so : game.getStack()) {
                if (so != null && oppId.equals(so.getControllerId())) {
                    String name = so.getName();
                    if (name != null && !name.isEmpty()) {
                        names.add(name);
                    }
                }
            }
        } catch (Throwable t) {
            return -1;
        }
        for (int i = 0; i < ARCHETYPE_SIGNATURE_CARDS.length; i++) {
            for (String sig : ARCHETYPE_SIGNATURE_CARDS[i]) {
                if (names.contains(sig)) return i;
            }
        }
        return -1;
    }

    public static float[] computeCardBeliefLabels(Game game, UUID viewerId) {
        if (!CARD_BELIEF_LABELS_ENABLED || CARD_BELIEF_VOCAB.cardNames.length == 0
                || game == null || viewerId == null) {
            return null;
        }
        UUID oppId = null;
        for (UUID pid : game.getOpponents(viewerId)) {
            oppId = pid;
            break;
        }
        if (oppId == null) {
            return null;
        }
        Player opp = game.getPlayer(oppId);
        if (opp == null) {
            return null;
        }

        float[] labels = new float[CARD_BELIEF_VOCAB.cardNames.length];
        try {
            for (Card c : opp.getHand().getCards(game)) {
                addCardBeliefCount(labels, c);
            }
            for (Card c : opp.getLibrary().getCards(game)) {
                addCardBeliefCount(labels, c);
            }
        } catch (Throwable t) {
            return null;
        }

        for (int i = 0; i < labels.length; i++) {
            float maxCount = i < CARD_BELIEF_VOCAB.maxCounts.length
                    ? CARD_BELIEF_VOCAB.maxCounts[i]
                    : 1.0f;
            labels[i] = Math.max(0.0f, Math.min(1.0f, labels[i] / Math.max(1.0f, maxCount)));
        }
        return labels;
    }

    private static void addCardBeliefCount(float[] labels, Card card) {
        if (labels == null || card == null) {
            return;
        }
        Integer idx = CARD_BELIEF_VOCAB.indexByName.get(card.getName());
        if (idx != null && idx >= 0 && idx < labels.length) {
            labels[idx] += 1.0f;
        }
    }

    private static CardBeliefVocab loadCardBeliefVocab() {
        String deckList = System.getenv().getOrDefault("RL_CARD_BELIEF_DECK_LIST", "").trim();
        if (deckList.isEmpty()) {
            deckList = System.getenv().getOrDefault("DECK_LIST_FILE", "").trim();
        }
        DeterminizationSampler sampler = deckList.isEmpty()
                ? DeterminizationSampler.pauperDefaults()
                : DeterminizationSampler.loadFromDeckListFile(deckList);
        if (sampler == null || sampler.getArchetypes().isEmpty()) {
            return CardBeliefVocab.empty();
        }

        Map<String, Integer> maxCounts = new java.util.TreeMap<>();
        for (String archetype : sampler.getArchetypes()) {
            Map<String, Integer> counts = sampler.decklistCounts(archetype);
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                String name = e.getKey();
                int count = e.getValue() == null ? 0 : e.getValue();
                if (name == null || name.trim().isEmpty() || count <= 0) {
                    continue;
                }
                Integer prev = maxCounts.get(name);
                if (prev == null || count > prev) {
                    maxCounts.put(name, count);
                }
            }
        }
        if (maxCounts.isEmpty()) {
            return CardBeliefVocab.empty();
        }

        List<Map.Entry<String, Integer>> entries = new ArrayList<>(maxCounts.entrySet());
        Collections.sort(entries, (a, b) -> {
            int byCount = Integer.compare(b.getValue(), a.getValue());
            return byCount != 0 ? byCount : a.getKey().compareTo(b.getKey());
        });
        int limit = CARD_BELIEF_MAX_CARDS <= 0 ? entries.size() : Math.min(CARD_BELIEF_MAX_CARDS, entries.size());
        String[] names = new String[limit];
        float[] max = new float[limit];
        Map<String, Integer> index = new HashMap<>();
        for (int i = 0; i < limit; i++) {
            Map.Entry<String, Integer> e = entries.get(i);
            names[i] = e.getKey();
            max[i] = Math.max(1, e.getValue());
            index.put(names[i], i);
        }
        logger.info("Loaded generic card-belief vocabulary: dim=" + names.length
                + " deckList=" + (deckList.isEmpty() ? "<pauperDefaults>" : deckList));
        return new CardBeliefVocab(names, Collections.unmodifiableMap(index), max);
    }

    private static int parseEnvInt(String key, int fallback) {
        try {
            return Integer.parseInt(System.getenv().getOrDefault(key, Integer.toString(fallback)).trim());
        } catch (Exception e) {
            return fallback;
        }
    }
}
