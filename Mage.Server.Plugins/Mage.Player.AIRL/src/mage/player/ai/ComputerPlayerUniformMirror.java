package mage.player.ai;

import mage.Mana;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.Mode;
import mage.abilities.Modes;
import mage.abilities.SpellAbility;
import mage.abilities.TriggeredAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.common.DiscardCardCost;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.effects.common.DrawCardSourceControllerEffect;
import mage.abilities.mana.RedManaAbility;
import mage.abilities.triggers.BeginningOfUpkeepTriggeredAbility;
import mage.abilities.mana.ManaOptions;
import mage.cards.Card;
import mage.cards.CardSetInfo;
import mage.cards.Cards;
import mage.cards.basiclands.Mountain;
import mage.cards.c.ChainLightning;
import mage.cards.g.GoblinBushwhacker;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.constants.Rarity;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.token.BloodToken;
import mage.game.stack.StackObject;
import mage.player.ai.rl.PythonMLBatchManager;
import mage.player.ai.rl.PythonModel;
import mage.player.ai.rl.SeededUniformMirrorPolicy;
import mage.player.ai.rl.StateSequenceBuilder;
import mage.players.Player;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.target.TargetCard;
import mage.target.common.TargetCardInHand;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Benchmark-only XMage player with an explicit seeded-uniform policy surface.
 *
 * <p>This class deliberately does not alter {@link ComputerPlayerRL}'s legacy
 * random or trainer behavior. Every strategic callback used by the Rally
 * mirror is implemented here and any unsupported canonicalization fails the
 * game instead of falling through to a heuristic or model.</p>
 */
public final class ComputerPlayerUniformMirror extends ComputerPlayerRL {

    private static final long serialVersionUID = 1L;
    private static final PythonModel FAIL_FAST_MODEL = new FailFastPythonModel();
    private static final int MAX_TRIGGER_ORDER_OBJECTS = 7;
    private static final String CHAIN_LIGHTNING_PROMPT = "Pay {R}{R} to copy the spell?";
    private static final String CHAIN_LIGHTNING_CLASS = "mage.cards.c.ChainLightning";
    private static final java.util.regex.Pattern CHAIN_LIGHTNING_RETARGET_PROMPT =
            java.util.regex.Pattern.compile(
                    "^Change this 1 of 1 target: [^\\r\\n?]{1,256}\\?$");

    private final SeededUniformMirrorPolicy mirrorPolicy;
    private final String physicalSeat;
    private long canonicalEncounterTieBreaks;
    private long explicitConcedeAttempts;
    private long forcedNoPolicySelections;
    private TriggerOrderCache triggerOrderCache;

    public ComputerPlayerUniformMirror(String name, RangeOfInfluence range,
                                       long baseSeed, long episodeId, String physicalSeat) {
        super(name, range, FAIL_FAST_MODEL, true, false, SeededUniformMirrorPolicy.POLICY_ID);
        this.physicalSeat = requireSeat(physicalSeat);
        this.mirrorPolicy = new SeededUniformMirrorPolicy(baseSeed, episodeId, physicalSeat);
    }

    private ComputerPlayerUniformMirror(ComputerPlayerUniformMirror source) {
        super(source);
        this.physicalSeat = source.physicalSeat;
        this.mirrorPolicy = source.mirrorPolicy.copy();
        this.canonicalEncounterTieBreaks = source.canonicalEncounterTieBreaks;
        this.explicitConcedeAttempts = source.explicitConcedeAttempts;
        this.forcedNoPolicySelections = source.forcedNoPolicySelections;
        this.triggerOrderCache = source.triggerOrderCache == null
                ? null : source.triggerOrderCache.copyAsSnapshotDrift();
    }

    @Override
    public ComputerPlayerUniformMirror copy() {
        return new ComputerPlayerUniformMirror(this);
    }

    /** Fail closed instead of ComputerPlayerRL's catch-and-pass production fallback. */
    @Override
    public boolean priority(Game game) {
        game.resumeTimer(getTurnControlledBy());
        try {
            return priorityPlay(game);
        } finally {
            game.pauseTimer(getTurnControlledBy());
        }
    }

    /**
     * Build the benchmark priority menu directly from XMage's public playable
     * surface. This intentionally bypasses ComputerPlayerRL's model-feature and
     * secondary activation-validation machinery: neither is part of a uniform
     * policy, and the former can persist fallback text embeddings even when the
     * model itself is fail-fast disabled.
     */
    @Override
    protected Ability calculateRLAction(Game game) {
        // The default two-argument surface collapses equal-looking battlefield
        // activations across distinct objects. Rust exposes each source object,
        // so use XMage's public non-deduplicating form. Its empty/nonempty
        // predicate is identical to getPlayable(game, true).
        List<ActivatedAbility> playable = querySourceDistinctPlayable(
                (hidden, fromZone, hideDuplicates) ->
                        getPlayable(game, hidden, fromZone, hideDuplicates));
        for (ActivatedAbility ability : playable) {
            if (ability == null) {
                throw violation("XMage playable menu contains null ability");
            }
        }
        Ability forcedPass = passOnlyPriorityResult(playable, mirrorPolicy);
        if (forcedPass != null) {
            forcedNoPolicySelections++;
            return forcedPass;
        }
        playable.add(0, new PassAbility());
        List<Integer> selected = genericChoose(
                playable,
                1,
                1,
                StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL,
                game,
                null);
        if (selected.size() != 1 || selected.get(0) < 0 || selected.get(0) >= playable.size()) {
            throw violation("uniform priority menu returned invalid selection " + selected);
        }
        return playable.get(selected.get(0));
    }

    @Override
    public <T> List<Integer> genericChoose(List<T> candidates, int maxTargets, int minTargets,
                                           StateSequenceBuilder.ActionType actionType,
                                           Game game, Ability source) {
        if (candidates == null || candidates.isEmpty()) {
            if (minTargets > 0) {
                throw violation("empty mandatory generic menu " + actionType);
            }
            return Collections.emptyList();
        }
        int boundedMax = Math.max(0, Math.min(maxTargets, candidates.size()));
        int boundedMin = Math.max(0, Math.min(minTargets, candidates.size()));
        if (boundedMin > boundedMax) {
            throw violation("generic menu min exceeds max for " + actionType);
        }
        if (boundedMax == 0) {
            return Collections.emptyList();
        }
        List<IndexedCandidate<T>> ordered = canonicalize(candidates, game, source);
        int[] selectedRanks = mirrorPolicy.chooseNoncombatWithoutReplacement(
                category(actionType), ordered.size(), boundedMax);
        List<Integer> selected = new ArrayList<>(selectedRanks.length);
        for (int rank : selectedRanks) {
            selected.add(ordered.get(rank).originalIndex);
        }
        if (selected.size() < boundedMin) {
            throw violation("uniform generic menu selected fewer than mandatory minimum");
        }
        return selected;
    }

    /** Keep seven is a fixed setup rule and consumes no policy decision. */
    @Override
    public boolean chooseMulligan(Game game) {
        return false;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        return chooseTarget(outcome, target, source, game);
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game,
                          Map<String, Serializable> options) {
        return chooseTarget(outcome, target, source, game);
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        if (target == null) {
            return false;
        }
        if ("starting player".equalsIgnoreCase(target.getTargetName())) {
            throw violation("starting-player prompt reached policy despite fixed P0 setup");
        }
        UUID abilityControllerId = getId();
        if (target.getTargetController() != null && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }
        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());
        int alreadyChosen = target.getSize();
        Set<UUID> existing = new HashSet<>(target.getTargets());
        Set<UUID> chosen = new HashSet<>();
        while (alreadyChosen + chosen.size() < maxTargets) {
            List<UUID> legal = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
            final UUID controller = abilityControllerId;
            legal.removeIf(id -> id == null || existing.contains(id) || chosen.contains(id)
                    || !target.canTarget(controller, id, source, game));
            int outstandingMandatory = Math.max(0, minTargets - alreadyChosen - chosen.size());
            List<UUID> forcedCandidates = mandatoryForcedCandidates(
                    legal, outstandingMandatory);
            if (forcedCandidates != null) {
                for (IndexedCandidate<UUID> forced : canonicalize(forcedCandidates, game, source)) {
                    target.addTarget(forced.candidate, source, game);
                    chosen.add(forced.candidate);
                    forcedNoPolicySelections++;
                }
                break;
            }
            if (alreadyChosen + chosen.size() >= minTargets) {
                legal.add(null); // canonical STOP action
            }
            if (legal.isEmpty()) {
                break;
            }
            UUID selected = chooseCanonicalOne("target", legal, game, source);
            if (selected == null) {
                break;
            }
            target.addTarget(selected, source, game);
            chosen.add(selected);
        }
        if (alreadyChosen + chosen.size() < minTargets) {
            throw violation("target menu ended below mandatory minimum");
        }
        return alreadyChosen + chosen.size() > 0 || minTargets == 0;
    }

    @Override
    public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        return chooseTarget(outcome, cards, target, source, game);
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target,
                                Ability source, Game game) {
        if (cards == null || target == null) {
            return false;
        }
        List<Card> legal = target.getFilter() == null
                ? new ArrayList<>(cards.getCards(game))
                : new ArrayList<>(cards.getCards(target.getFilter(), getId(), source, game));
        return chooseCardTargetCallbackFromLegal(
                legal, target, source, game,
                card -> target.addTarget(card.getId(), source, game));
    }

    @FunctionalInterface
    interface CardTargetCommitter {
        void commit(Card card);
    }

    boolean chooseCardTargetCallbackFromLegal(
            List<Card> legal, TargetCard target, Ability source, Game game,
            CardTargetCommitter committer) {
        if (legal == null || target == null || committer == null) {
            throw violation("card-target callback seam received null input");
        }
        int minTargets = Math.max(0, target.getMinNumberOfTargets());
        int maxTargets = Math.max(0, target.getMaxNumberOfTargets());
        int alreadyChosen = target.getSize();
        Set<UUID> existing = new HashSet<>(target.getTargets());
        Set<UUID> chosen = new HashSet<>();
        while (alreadyChosen + chosen.size() < maxTargets) {
            List<Card> remaining = new ArrayList<>();
            for (Card card : legal) {
                if (card != null && !existing.contains(card.getId())
                        && !chosen.contains(card.getId())) {
                    remaining.add(card);
                }
            }
            int outstandingMandatory = Math.max(0, minTargets - alreadyChosen - chosen.size());
            List<Card> forcedCandidates = mandatoryForcedCandidates(
                    remaining, outstandingMandatory);
            if (forcedCandidates != null) {
                List<Card> orderedForced;
                if (forcedCandidates.size() == 1) {
                    orderedForced = forcedCandidates;
                } else {
                    orderedForced = new ArrayList<>(forcedCandidates.size());
                    for (IndexedCandidate<Card> forced : canonicalize(
                            forcedCandidates, game, source)) {
                        orderedForced.add(forced.candidate);
                    }
                }
                for (Card forced : orderedForced) {
                    committer.commit(forced);
                    chosen.add(forced.getId());
                    forcedNoPolicySelections++;
                }
                break;
            }
            List<Card> menu = new ArrayList<>(remaining);
            if (alreadyChosen + chosen.size() >= minTargets) {
                menu.add(null);
            }
            if (menu.isEmpty()) {
                break;
            }
            Card selected = chooseCanonicalOne("card_target", menu, game, source);
            if (selected == null) {
                break;
            }
            committer.commit(selected);
            chosen.add(selected.getId());
        }
        if (alreadyChosen + chosen.size() < minTargets) {
            throw violation("card target menu ended below mandatory minimum");
        }
        return alreadyChosen + chosen.size() > 0 || minTargets == 0;
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        if (choice == null) {
            return false;
        }
        if (choice.isKeyChoice()) {
            List<String> keys = new ArrayList<>(choice.getKeyChoices().keySet());
            if (keys.isEmpty()) {
                return false;
            }
            choice.setChoiceByKey(chooseCanonicalOne("choice_key", keys, game, null));
        } else {
            List<String> values = new ArrayList<>(choice.getChoices());
            if (values.isEmpty()) {
                return false;
            }
            choice.setChoice(chooseCanonicalOne("choice", values, game, null));
        }
        return choice.isChosen();
    }

    @Override
    public Mode chooseMode(Modes modes, Ability source, Game game) {
        if (modes == null) {
            return null;
        }
        List<Mode> available = new ArrayList<>(modes.getAvailableModes(source, game));
        if (available.isEmpty()) {
            return null;
        }
        return chooseCanonicalOne("mode", available, game, source);
    }

    @Override
    public int announceX(int min, int max, String message, Game game, Ability source, boolean isManaPay) {
        if (max < min) {
            throw violation("announceX max is below min");
        }
        long count = (long) max - (long) min + 1L;
        if (count > Integer.MAX_VALUE) {
            throw violation("announceX legal domain exceeds exact benchmark implementation");
        }
        return min + mirrorPolicy.chooseNoncombat("announce_x", (int) count);
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, Ability source, Game game) {
        return chooseUse(outcome, message, null, null, null, source, game);
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage,
                             String trueText, String falseText, Ability source, Game game) {
        return chooseUseCallback(
                outcome, message, secondMessage, trueText, falseText, source,
                isExactChainLightningSource(source, game), getManaAvailable(game));
    }

    boolean chooseUseCallback(
            Outcome outcome, String message, String secondMessage,
            String trueText, String falseText, Ability source,
            boolean exactChainSource, ManaOptions available) {
        if (available == null) {
            throw violation("chooseUse callback seam received null mana surface");
        }
        boolean chainPaymentOutcome = outcome == Outcome.Copy;
        boolean chainPaymentPrompt = CHAIN_LIGHTNING_PROMPT.equals(message);
        boolean chainRetargetOutcome = outcome == Outcome.Damage;
        boolean chainRetargetPrompt = message != null
                && CHAIN_LIGHTNING_RETARGET_PROMPT.matcher(message).matches();
        boolean standardLabels = secondMessage == null && trueText == null && falseText == null;

        if (exactChainSource && (chainRetargetOutcome || chainRetargetPrompt)) {
            if (!(chainRetargetOutcome && chainRetargetPrompt && standardLabels)) {
                throw violation("unsupported near-match to Chain Lightning retarget contract");
            }
            return chainLightningChoiceForRank(
                    mirrorPolicy.chooseNoncombat("chain_lightning_copy_retarget", 2), true);
        }
        if (chainRetargetPrompt) {
            throw violation("Chain Lightning retarget prompt came from a non-Chain source");
        }
        if (chainPaymentOutcome || chainPaymentPrompt || exactChainSource) {
            if (!(chainPaymentOutcome && chainPaymentPrompt
                    && exactChainSource && standardLabels)) {
                throw violation("unsupported near-match to Chain Lightning copy contract");
            }
            // Matched Rust action order is [pay/copy, decline], even when the
            // eventual mana payment cannot succeed.
            boolean affordable = available.enough(manaFromMessage(message));
            return chainLightningChoiceForRank(
                    mirrorPolicy.chooseNoncombat("chain_lightning_copy", 2), affordable);
        }
        // XMage can offer an optional additional cost before checking whether
        // base + additional mana is jointly affordable. In that state YES is
        // not a legal policy action (Goblin Bushwhacker kicker is the Rally
        // case). Rust removes the unavailable branch before policy dispatch,
        // so this forced result consumes zero seeded decisions.
        if (outcome == Outcome.AIDontUseIt && source instanceof SpellAbility) {
            Mana additional = manaFromMessage(message);
            if (additional.count() > 0) {
                if (isJointlyUnaffordable(
                        source.getManaCostsToPay().getMana(), additional,
                        available)) {
                    forcedNoPolicySelections++;
                    return false;
                }
            }
        }
        if (message != null && message.startsWith("Pay ") && message.contains("{")
                && !message.toLowerCase(java.util.Locale.ROOT).contains("instead of")) {
            Mana optional = manaFromMessage(message);
            if (optional.count() > 0 && !available.enough(optional)) {
                mirrorPolicy.chooseNoncombat("choose_use_forced_false", 1);
                return false;
            }
        }
        return mirrorPolicy.chooseNoncombat("choose_use", 2) == 1;
    }

    private static boolean isExactChainLightningSource(Ability source, Game game) {
        if (!(source instanceof SpellAbility) || game == null || source.getSourceId() == null) {
            return false;
        }
        Card sourceCard = game.getCard(source.getSourceId());
        return sourceCard != null
                && CHAIN_LIGHTNING_CLASS.equals(sourceCard.getClass().getName())
                && "Chain Lightning".equals(sourceCard.getName());
    }

    static boolean isJointlyUnaffordable(Mana baseCost, Mana additionalCost,
                                         ManaOptions available) {
        if (baseCost == null || additionalCost == null || available == null) {
            throw violation("joint affordability seam received null input");
        }
        Mana total = baseCost.copy();
        total.add(additionalCost);
        return !available.enough(total);
    }

    static boolean chainLightningChoiceForRank(int rank, boolean affordable) {
        // Affordability is intentionally bound into this production seam but
        // does not prune either matched action rank.
        if (rank == 0) {
            return true;
        }
        if (rank == 1) {
            return false;
        }
        throw violation("Chain Lightning policy rank is outside [0,1]");
    }

    /** Bind benchmark semantics independently of RL_ENGINE_CHOICES. */
    @Override
    public boolean getStrictChooseMode() {
        return false;
    }

    /**
     * XMage asks repeatedly while moving a simultaneous trigger group to the
     * stack. Rust exposes one action containing a full permutation. Cache that
     * one sampled permutation and replay it across XMage callbacks without
     * consuming additional physical decisions.
     */
    @Override
    public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        if (game == null || abilities == null || abilities.size() <= 1) {
            clearTriggerOrderCache();
            throw violation("trigger-order callback requires a game and at least two triggers");
        }
        if (triggerOrderCache != null && triggerOrderCache.snapshotDrift) {
            clearTriggerOrderCache();
            throw violation("active trigger-order cache crossed a player snapshot");
        }
        if (triggerOrderCache == null) {
            if (abilities.size() > MAX_TRIGGER_ORDER_OBJECTS) {
                throw violation("trigger-order group exceeds exact benchmark cap");
            }
            long tiesBefore = canonicalEncounterTieBreaks;
            List<IndexedCandidate<TriggeredAbility>> canonical = canonicalize(abilities, game, null);
            if (canonicalEncounterTieBreaks != tiesBefore) {
                throw violation("trigger-order group has non-canonical semantic duplicates");
            }
            triggerOrderCache = beginTriggerOrderCache(
                    game.getId(), triggerOffers(canonical), mirrorPolicy);
        }
        return consumeTriggerOrder(abilities, game);
    }

    /** Explicit concession is never a natural benchmark terminal. */
    @Override
    public void concede(Game game) {
        explicitConcedeAttempts++;
        throw violation("explicit concession is prohibited in uniform mirror benchmark");
    }

    private static Mana manaFromMessage(String message) {
        if (message == null) {
            return new Mana();
        }
        java.util.regex.Matcher matcher = java.util.regex.Pattern
                .compile("\\{([^}]+)\\}").matcher(message);
        StringBuilder symbols = new StringBuilder();
        while (matcher.find()) {
            symbols.append('{').append(matcher.group(1)).append('}');
        }
        if (symbols.length() == 0) {
            return new Mana();
        }
        try {
            return new ManaCostsImpl<>(symbols.toString()).getMana();
        } catch (RuntimeException e) {
            throw violation("cannot parse optional mana prompt: " + message);
        }
    }

    /** Benchmark activation failure is an invalid trial, never a forced pass. */
    @Override
    protected boolean failClosedOnActivationFailure() {
        return true;
    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        if (target == null || source == null) {
            return false;
        }
        List<? extends TargetAmount> options = target.getTargetOptions(source, game);
        if (options.isEmpty()) {
            return false;
        }
        List<TargetAmount> mutable = new ArrayList<>(options);
        TargetAmount selected = chooseCanonicalOne("target_amount", mutable, game, source);
        for (UUID id : selected.getTargets()) {
            target.addTarget(id, selected.getTargetAmount(id), source, game, true);
        }
        return target.isChosen(game);
    }

    @Override
    public boolean choosePile(Outcome outcome, String message,
                              List<? extends Card> pile1, List<? extends Card> pile2, Game game) {
        return mirrorPolicy.chooseNoncombat("choose_pile", 2) == 0;
    }

    @Override
    public int chooseReplacementEffect(Map<String, String> effectsMap,
                                       Map<String, MageObject> objectsMap, Game game) {
        if (effectsMap == null || effectsMap.isEmpty()) {
            throw violation("empty replacement-effect menu");
        }
        List<String> original = new ArrayList<>(effectsMap.keySet());
        List<IndexedCandidate<String>> ordered = canonicalize(original, game, null);
        int rank = mirrorPolicy.chooseNoncombat("replacement_effect", ordered.size());
        return ordered.get(rank).originalIndex;
    }

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        if (game.isSimulation()) {
            return;
        }
        List<Permanent> eligible = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllActivePermanents(attackingPlayerId)) {
            if (permanent != null && permanent.isCreature() && permanent.canAttack(null, game)) {
                eligible.add(permanent);
            }
        }
        if (eligible.isEmpty()) {
            return;
        }
        List<IndexedCandidate<Permanent>> canonical = canonicalize(eligible, game, null);
        boolean[] include = mirrorPolicy.chooseAttackers(canonical.size());
        List<UUID> defenders = new ArrayList<>(game.getCombat().getDefenders());
        if (defenders.isEmpty()) {
            throw violation("attack declaration has no legal defender");
        }
        boolean any = false;
        for (int i = 0; i < canonical.size(); i++) {
            if (!include[i]) {
                continue;
            }
            Permanent attacker = canonical.get(i).candidate;
            List<UUID> legalDefenders = new ArrayList<>();
            for (UUID defender : defenders) {
                if (attacker.canAttack(defender, game)) {
                    legalDefenders.add(defender);
                }
            }
            if (legalDefenders.isEmpty()) {
                throw violation("sampled attacker has no legal defender");
            }
            UUID defender = legalDefenders.size() == 1
                    ? legalDefenders.get(0)
                    : chooseCanonicalOne("attack_defender", legalDefenders, game, null);
            declareAttacker(attacker.getId(), defender, game, false);
            any = true;
        }
        if (any) {
            game.getPlayers().resetPassed();
        }
    }

    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        if (game.isSimulation()) {
            return;
        }
        List<Permanent> attackers = new ArrayList<>();
        for (UUID attackerId : game.getCombat().getAttackers()) {
            Permanent attacker = game.getPermanent(attackerId);
            if (attacker != null) {
                attackers.add(attacker);
            }
        }
        if (attackers.isEmpty()) {
            return;
        }
        List<Permanent> available = new ArrayList<>(getAvailableBlockers(game));
        List<IndexedCandidate<Permanent>> orderedAttackers = canonicalize(attackers, game, source);
        boolean any = false;
        for (IndexedCandidate<Permanent> indexedAttacker : orderedAttackers) {
            Permanent attacker = indexedAttacker.candidate;
            List<Permanent> legal = new ArrayList<>();
            for (Permanent blocker : available) {
                if (blocker != null && blocker.canBlock(attacker.getId(), game)) {
                    legal.add(blocker);
                }
            }
            if (legal.isEmpty()) {
                continue;
            }
            List<IndexedCandidate<Permanent>> orderedBlockers = canonicalize(legal, game, source);
            int selectedRank = mirrorPolicy.chooseBlocker(orderedBlockers.size());
            if (selectedRank >= 0) {
                Permanent blocker = orderedBlockers.get(selectedRank).candidate;
                declareBlocker(getId(), blocker.getId(), attacker.getId(), game);
                available.remove(blocker);
                any = true;
            }
        }
        if (any) {
            game.getPlayers().resetPassed();
        }
    }

    public SeededUniformMirrorPolicy getMirrorPolicySnapshot() {
        return mirrorPolicy.copy();
    }

    public long getCanonicalEncounterTieBreaks() {
        return canonicalEncounterTieBreaks;
    }

    public long getExplicitConcedeAttempts() {
        return explicitConcedeAttempts;
    }

    public long getForcedNoPolicySelections() {
        return forcedNoPolicySelections;
    }

    public String getPhysicalSeat() {
        return physicalSeat;
    }

    private TriggeredAbility consumeTriggerOrder(List<TriggeredAbility> abilities, Game game) {
        try {
            List<TriggerOffer> offered = new ArrayList<>(abilities.size());
            for (TriggeredAbility ability : abilities) {
                offered.add(new TriggerOffer(
                        ability == null ? null : ability.getId(),
                        canonicalKey(ability, game, null), ability));
            }
            TriggerConsumeResult result = consumeTriggerOrderCache(
                    triggerOrderCache, offered, game.getId());
            // GameImpl auto-stacks the final remaining trigger without another
            // callback, so no cache may survive once only that tail remains.
            if (result.complete) {
                clearTriggerOrderCache();
            }
            return result.ability;
        } catch (RuntimeException e) {
            clearTriggerOrderCache();
            throw e;
        }
    }

    private static List<TriggerOffer> triggerOffers(
            List<IndexedCandidate<TriggeredAbility>> canonical) {
        List<TriggerOffer> offers = new ArrayList<>(canonical.size());
        for (IndexedCandidate<TriggeredAbility> item : canonical) {
            TriggeredAbility ability = item.candidate;
            offers.add(new TriggerOffer(
                    ability == null ? null : ability.getId(), item.key, ability));
        }
        return offers;
    }

    static TriggerOrderCache beginTriggerOrderCache(
            UUID gameId, List<TriggerOffer> canonicalOffers,
            SeededUniformMirrorPolicy policy) {
        if (gameId == null || canonicalOffers == null || policy == null
                || canonicalOffers.size() < 2
                || canonicalOffers.size() > MAX_TRIGGER_ORDER_OBJECTS) {
            throw violation("invalid trigger-order cache input");
        }
        Set<UUID> identities = new HashSet<>();
        for (TriggerOffer offer : canonicalOffers) {
            if (offer == null || offer.identity == null || offer.canonicalKey == null
                    || offer.instance == null || !identities.add(offer.identity)) {
                throw violation("trigger-order group lacks unique stable ability identities");
            }
        }
        int selectedRank = policy.chooseNoncombat(
                "order_triggers", factorialExact(canonicalOffers.size()));
        int[] order = triggerPermutations(canonicalOffers.size()).get(selectedRank);
        List<TriggerOffer> selected = new ArrayList<>(order.length);
        for (int index : order) {
            selected.add(canonicalOffers.get(index));
        }
        return new TriggerOrderCache(gameId, selected, false);
    }

    static TriggerConsumeResult consumeTriggerOrderCache(
            TriggerOrderCache cache, List<TriggerOffer> offered, UUID gameId) {
        if (cache == null || cache.snapshotDrift || gameId == null
                || !cache.gameId.equals(gameId)) {
            throw violation("trigger-order cache is absent, copied, or belongs to another game");
        }
        if (offered == null
                || cache.selectedOrder.size() - cache.cursor != offered.size()) {
            throw violation("trigger-order callback changed the remaining group size");
        }
        Map<UUID, TriggerOffer> offeredByIdentity = new LinkedHashMap<>();
        for (TriggerOffer offer : offered) {
            if (offer == null || offer.identity == null || offer.canonicalKey == null
                    || offer.instance == null
                    || offeredByIdentity.put(offer.identity, offer) != null) {
                throw violation("trigger-order callback contains null or duplicate identity");
            }
        }
        for (int i = cache.cursor; i < cache.selectedOrder.size(); i++) {
            TriggerOffer expected = cache.selectedOrder.get(i);
            TriggerOffer current = offeredByIdentity.get(expected.identity);
            if (current == null || current.instance != expected.instance
                    || !expected.canonicalKey.equals(current.canonicalKey)) {
                throw violation("trigger-order callback changed identity, key, or object instance");
            }
        }
        TriggerOffer next = cache.selectedOrder.get(cache.cursor);
        TriggeredAbility result = offeredByIdentity.get(next.identity).instance;
        cache.cursor++;
        return new TriggerConsumeResult(
                result, cache.selectedOrder.size() - cache.cursor <= 1);
    }

    private void clearTriggerOrderCache() {
        triggerOrderCache = null;
    }

    private static int factorialExact(int n) {
        if (n < 2 || n > MAX_TRIGGER_ORDER_OBJECTS) {
            throw violation("unsupported trigger-order factorial domain");
        }
        int value = 1;
        for (int i = 2; i <= n; i++) {
            if (value > Integer.MAX_VALUE / i) {
                throw violation("trigger-order factorial overflows int action domain");
            }
            value *= i;
        }
        return value;
    }

    /** Exact Rust permute_from swap order. */
    private static List<int[]> triggerPermutations(int n) {
        int expected = factorialExact(n);
        int[] current = new int[n];
        for (int i = 0; i < n; i++) {
            current[i] = i;
        }
        List<int[]> result = new ArrayList<>(expected);
        permuteFrom(0, current, result);
        if (result.size() != expected) {
            throw violation("trigger permutation enumeration cardinality mismatch");
        }
        return result;
    }

    private static void permuteFrom(int start, int[] current, List<int[]> out) {
        if (start == current.length) {
            out.add(current.clone());
            return;
        }
        for (int i = start; i < current.length; i++) {
            int value = current[start];
            current[start] = current[i];
            current[i] = value;
            permuteFrom(start + 1, current, out);
            value = current[start];
            current[start] = current[i];
            current[i] = value;
        }
    }

    /** Pure failure shields exercised by the benchmark --self-test path. */
    public static void runUniformMirrorSelfTest() {
        if (ComputerPlayerRL.activationFailureDiagnosticsAllowed(true)
                || !ComputerPlayerRL.activationFailureDiagnosticsAllowed(false)) {
            throw new IllegalStateException("strict activation diagnostic gate failed");
        }
        assertPermutationOrder(triggerPermutations(2), new int[][]{{0, 1}, {1, 0}}, "n=2");
        assertPermutationOrder(triggerPermutations(3), new int[][]{
                {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                {1, 2, 0}, {2, 1, 0}, {2, 0, 1}}, "n=3");

        assertProductionTriggerCacheSeam();
        assertPassOnlyPrioritySeam();
        assertSourceDistinctPlayableSeam();
        assertForcedBloodTargetCallback();
        assertBushwhackerChooseUseCallback();
        assertChainLightningChooseUseCallbacks();

        String exile0 = cardKeyComponents("exile", "p0", 0, "Twin", "TST", "1", "CardClass");
        String exile1 = cardKeyComponents("exile", "p0", 1, "Twin", "TST", "1", "CardClass");
        if (exile0.equals(exile1)) {
            throw new IllegalStateException("duplicate exiled card identities collided");
        }
    }

    private static void assertProductionTriggerCacheSeam() {
        UUID gameId = UUID.randomUUID();
        BeginningOfUpkeepTriggeredAbility first = new BeginningOfUpkeepTriggeredAbility(
                new DrawCardSourceControllerEffect(1));
        BeginningOfUpkeepTriggeredAbility second = new BeginningOfUpkeepTriggeredAbility(
                new DrawCardSourceControllerEffect(2));
        BeginningOfUpkeepTriggeredAbility third = new BeginningOfUpkeepTriggeredAbility(
                new DrawCardSourceControllerEffect(3));
        first.setSourceId(UUID.randomUUID());
        second.setSourceId(UUID.randomUUID());
        third.setSourceId(UUID.randomUUID());
        List<TriggerOffer> canonical = Arrays.asList(
                new TriggerOffer(first.getId(), "trigger|first", first),
                new TriggerOffer(second.getId(), "trigger|second", second),
                new TriggerOffer(third.getId(), "trigger|third", third));
        SeededUniformMirrorPolicy policy = new SeededUniformMirrorPolicy(71_501L, 17L, "p0");
        TriggerOrderCache cache = beginTriggerOrderCache(gameId, canonical, policy);
        if (policy.getPhysicalDecisionCount() != 1L
                || policy.getPolicyActionSelections() != 1L
                || policy.getPolicyLeafEvaluations() != 1L) {
            throw new IllegalStateException("trigger-order physical decision accounting failed");
        }

        List<TriggerOffer> allOffered = new ArrayList<>(canonical);
        Collections.reverse(allOffered);
        TriggerConsumeResult firstResult = consumeTriggerOrderCache(cache, allOffered, gameId);
        if (firstResult.ability != cache.selectedOrder.get(0).instance || firstResult.complete) {
            throw new IllegalStateException("trigger-order first production consumption failed");
        }
        List<TriggerOffer> remaining = new ArrayList<>(cache.selectedOrder.subList(1, 3));
        Collections.reverse(remaining);
        TriggerConsumeResult secondResult = consumeTriggerOrderCache(cache, remaining, gameId);
        if (secondResult.ability != cache.selectedOrder.get(1).instance || !secondResult.complete
                || policy.getPhysicalDecisionCount() != 1L
                || policy.getPolicyActionSelections() != 1L) {
            throw new IllegalStateException("trigger-order cached consumption spent another decision");
        }

        TriggerOrderCache repeated = new TriggerOrderCache(gameId, cache.selectedOrder, false);
        consumeTriggerOrderCache(repeated, allOffered, gameId);
        expectUniformMirrorViolation(
                () -> consumeTriggerOrderCache(repeated, allOffered, gameId),
                "repeated trigger callback");

        List<TriggerOffer> keyDrift = new ArrayList<>(canonical);
        TriggerOffer original = keyDrift.get(0);
        keyDrift.set(0, new TriggerOffer(
                original.identity, original.canonicalKey + "|drift", original.instance));
        expectUniformMirrorViolation(
                () -> consumeTriggerOrderCache(
                        new TriggerOrderCache(gameId, cache.selectedOrder, false), keyDrift, gameId),
                "trigger canonical-key drift");

        BeginningOfUpkeepTriggeredAbility replacement = new BeginningOfUpkeepTriggeredAbility(
                new DrawCardSourceControllerEffect(1));
        replacement.setSourceId(first.getSourceId());
        List<TriggerOffer> objectDrift = new ArrayList<>(canonical);
        objectDrift.set(0, new TriggerOffer(first.getId(), "trigger|first", replacement));
        expectUniformMirrorViolation(
                () -> consumeTriggerOrderCache(
                        new TriggerOrderCache(gameId, cache.selectedOrder, false), objectDrift, gameId),
                "trigger object-instance drift");
        expectUniformMirrorViolation(
                () -> consumeTriggerOrderCache(cache.copyAsSnapshotDrift(), allOffered, gameId),
                "trigger cache player-snapshot drift");
    }

    private static void assertPassOnlyPrioritySeam() {
        SeededUniformMirrorPolicy policy = new SeededUniformMirrorPolicy(71_501L, 18L, "p0");
        long decisionsBefore = policy.getPhysicalDecisionCount();
        Ability result = passOnlyPriorityResult(Collections.emptyList(), policy);
        if (!(result instanceof PassAbility)
                || policy.getPhysicalDecisionCount() != decisionsBefore
                || policy.getPolicyActionSelections() != 0L) {
            throw new IllegalStateException("empty priority did not force a zero-policy pass");
        }
    }

    private static void assertSourceDistinctPlayableSeam() {
        Mountain firstMountain = selfTestMountain("1");
        Mountain secondMountain = selfTestMountain("2");
        ActivatedAbility firstMountainMana = firstActivatedAbility(firstMountain);
        ActivatedAbility secondMountainMana = firstActivatedAbility(secondMountain);
        RedManaAbility furnaceMana = new RedManaAbility();
        furnaceMana.setSourceId(UUID.randomUUID());

        BloodToken firstBlood = new BloodToken();
        BloodToken secondBlood = new BloodToken();
        ActivatedAbility firstBloodAbility = firstActivatedAbility(firstBlood);
        ActivatedAbility secondBloodAbility = firstActivatedAbility(secondBlood);
        firstBloodAbility.setSourceId(firstBlood.getId());
        secondBloodAbility.setSourceId(secondBlood.getId());

        List<ActivatedAbility> actualAbilities = Arrays.asList(
                firstMountainMana, secondMountainMana, furnaceMana,
                firstBloodAbility, secondBloodAbility);
        List<ActivatedAbility> queried = querySourceDistinctPlayable(
                (hidden, fromZone, hideDuplicates) -> {
                    if (!hidden || fromZone != Zone.ALL || hideDuplicates) {
                        throw new IllegalStateException(
                                "production playable query flags changed");
                    }
                    return actualAbilities;
                });
        Set<UUID> sourceIds = new HashSet<>();
        for (ActivatedAbility ability : queried) {
            sourceIds.add(ability.getSourceId());
        }
        if (queried.size() != 5 || sourceIds.size() != 5
                || queried.get(0) != firstMountainMana
                || queried.get(1) != secondMountainMana
                || queried.get(2) != furnaceMana
                || queried.get(3) != firstBloodAbility
                || queried.get(4) != secondBloodAbility) {
            throw new IllegalStateException(
                    "source-distinct actual playable abilities were deduplicated or reordered");
        }
    }

    private static void assertForcedBloodTargetCallback() {
        BloodToken blood = new BloodToken();
        ActivatedAbility bloodAbility = firstActivatedAbility(blood);
        bloodAbility.setSourceId(blood.getId());
        boolean hasDiscardCost = bloodAbility.getCosts().stream()
                .anyMatch(DiscardCardCost.class::isInstance);
        if (!hasDiscardCost) {
            throw new IllegalStateException("Blood token fixture lacks its discard cost");
        }
        ComputerPlayerUniformMirror player = selfTestPlayer(19L, "p0");
        Card onlyCard = selfTestMountain("3");
        TargetCardInHand target = new TargetCardInHand();
        List<Card> committed = new ArrayList<>();
        SeededUniformMirrorPolicy before = player.getMirrorPolicySnapshot();
        long forcedBefore = player.getForcedNoPolicySelections();
        boolean selected = player.chooseCardTargetCallbackFromLegal(
                Collections.singletonList(onlyCard), target, bloodAbility, null,
                committed::add);
        SeededUniformMirrorPolicy after = player.getMirrorPolicySnapshot();
        if (!selected || committed.size() != 1 || committed.get(0) != onlyCard
                || after.getPhysicalDecisionCount() != before.getPhysicalDecisionCount()
                || after.getPolicyActionSelections() != before.getPolicyActionSelections()
                || player.getForcedNoPolicySelections() != forcedBefore + 1L) {
            throw new IllegalStateException(
                    "one-card Blood discard callback was not a zero-policy forced target");
        }
    }

    private static void assertBushwhackerChooseUseCallback() {
        GoblinBushwhacker bushwhacker = new GoblinBushwhacker(
                UUID.randomUUID(),
                new CardSetInfo("Goblin Bushwhacker", "TST", "4", Rarity.COMMON));
        SpellAbility source = bushwhacker.getSpellAbility();
        source.setSourceId(bushwhacker.getId());
        ManaOptions oneRed = new ManaOptions();
        oneRed.add(redMana(1));
        ManaOptions twoRed = new ManaOptions();
        twoRed.add(redMana(2));

        ComputerPlayerUniformMirror unpayable = selfTestPlayer(20L, "p0");
        SeededUniformMirrorPolicy unpayableBefore = unpayable.getMirrorPolicySnapshot();
        long forcedBefore = unpayable.getForcedNoPolicySelections();
        boolean unpayableChoice = unpayable.chooseUseCallback(
                Outcome.AIDontUseIt, "Pay Kicker {R} ?", null, null, null,
                source, false, oneRed);
        SeededUniformMirrorPolicy unpayableAfter = unpayable.getMirrorPolicySnapshot();
        if (unpayableChoice
                || unpayableAfter.getPhysicalDecisionCount()
                != unpayableBefore.getPhysicalDecisionCount()
                || unpayableAfter.getPolicyActionSelections()
                != unpayableBefore.getPolicyActionSelections()
                || unpayable.getForcedNoPolicySelections() != forcedBefore + 1L) {
            throw new IllegalStateException(
                    "jointly unaffordable Bushwhacker callback was not zero-policy false");
        }

        ComputerPlayerUniformMirror affordable = selfTestPlayer(21L, "p0");
        SeededUniformMirrorPolicy expected = affordable.getMirrorPolicySnapshot();
        boolean expectedChoice = expected.chooseNoncombat("choose_use", 2) == 1;
        long affordableForcedBefore = affordable.getForcedNoPolicySelections();
        boolean affordableChoice = affordable.chooseUseCallback(
                Outcome.AIDontUseIt, "Pay Kicker {R} ?", null, null, null,
                source, false, twoRed);
        SeededUniformMirrorPolicy affordableAfter = affordable.getMirrorPolicySnapshot();
        if (affordableChoice != expectedChoice
                || affordableAfter.getPhysicalDecisionCount() != 1L
                || affordableAfter.getPolicyActionSelections() != 1L
                || affordable.getForcedNoPolicySelections() != affordableForcedBefore) {
            throw new IllegalStateException(
                    "jointly affordable Bushwhacker callback did not dispatch one policy group");
        }
    }

    private static void assertChainLightningChooseUseCallbacks() {
        ChainLightning chain = new ChainLightning(
                UUID.randomUUID(),
                new CardSetInfo("Chain Lightning", "TST", "5", Rarity.COMMON));
        SpellAbility source = chain.getSpellAbility();
        source.setSourceId(chain.getId());
        ManaOptions oneRed = new ManaOptions();
        oneRed.add(redMana(1));
        ManaOptions twoRed = new ManaOptions();
        twoRed.add(redMana(2));

        if (!chainLightningChoiceForRank(0, true)
                || !chainLightningChoiceForRank(0, false)
                || chainLightningChoiceForRank(1, true)
                || chainLightningChoiceForRank(1, false)) {
            throw new IllegalStateException("Chain Lightning rank mapping depends on affordability");
        }
        for (int i = 0; i < 2; i++) {
            ComputerPlayerUniformMirror player = selfTestPlayer(22L + i, "p1");
            SeededUniformMirrorPolicy expected = player.getMirrorPolicySnapshot();
            int rank = expected.chooseNoncombat("chain_lightning_copy", 2);
            boolean choice = player.chooseUseCallback(
                    Outcome.Copy, CHAIN_LIGHTNING_PROMPT, null, null, null,
                    source, true, i == 0 ? oneRed : twoRed);
            SeededUniformMirrorPolicy after = player.getMirrorPolicySnapshot();
            if (choice != (rank == 0)
                    || after.getPhysicalDecisionCount() != 1L
                    || after.getPolicyActionSelections() != 1L
                    || after.getPolicyLeafEvaluations() != 1L) {
                throw new IllegalStateException(
                        "Chain Lightning payment callback rank/accounting fixture failed");
            }
        }

        String retargetPrompt = "Change this 1 of 1 target: uniform-p0?";
        ManaOptions noMana = new ManaOptions();
        for (int rank = 0; rank < 2; rank++) {
            ComputerPlayerUniformMirror player = selfTestPlayerForRank(
                    "chain_lightning_copy_retarget", rank);
            boolean choice = player.chooseUseCallback(
                    Outcome.Damage, retargetPrompt, null, null, null,
                    source, true, noMana);
            SeededUniformMirrorPolicy after = player.getMirrorPolicySnapshot();
            if (choice != (rank == 0)
                    || after.getPhysicalDecisionCount() != 1L
                    || after.getPolicyActionSelections() != 1L
                    || after.getPolicyLeafEvaluations() != 1L) {
                throw new IllegalStateException(
                        "Chain Lightning retarget callback rank/accounting fixture failed");
            }
        }

        expectUniformMirrorViolation(
                () -> selfTestPlayer(30L, "p0").chooseUseCallback(
                        Outcome.Damage, "Change this 1 of 2 targets?", null, null, null,
                        source, true, noMana),
                "malformed Chain Lightning retarget callback");
        expectUniformMirrorViolation(
                () -> selfTestPlayer(31L, "p0").chooseUseCallback(
                        Outcome.Damage, retargetPrompt, null, null, null,
                        source, false, noMana),
                "Chain Lightning retarget prompt with non-Chain source");
        expectUniformMirrorViolation(
                () -> selfTestPlayer(32L, "p0").chooseUseCallback(
                        Outcome.Copy, CHAIN_LIGHTNING_PROMPT, null, null, null,
                        source, false, twoRed),
                "Chain Lightning payment prompt with non-Chain source");
    }

    private static ComputerPlayerUniformMirror selfTestPlayer(long episodeId, String seat) {
        return new ComputerPlayerUniformMirror(
                "uniform-self-test-" + seat, RangeOfInfluence.ALL,
                71_501L, episodeId, seat);
    }

    private static ComputerPlayerUniformMirror selfTestPlayerForRank(
            String category, int desiredRank) {
        for (long episodeId = 40L; episodeId < 296L; episodeId++) {
            ComputerPlayerUniformMirror player = selfTestPlayer(episodeId, "p0");
            SeededUniformMirrorPolicy expected = player.getMirrorPolicySnapshot();
            if (expected.chooseNoncombat(category, 2) == desiredRank) {
                return player;
            }
        }
        throw new IllegalStateException(
                "unable to synthesize uniform self-test rank " + desiredRank
                        + " for " + category);
    }

    private static Mountain selfTestMountain(String cardNumber) {
        Mountain mountain = new Mountain(
                UUID.randomUUID(),
                new CardSetInfo("Mountain", "TST", cardNumber, Rarity.COMMON));
        firstActivatedAbility(mountain).setSourceId(mountain.getId());
        return mountain;
    }

    private static ActivatedAbility firstActivatedAbility(MageObject object) {
        for (Ability ability : object.getAbilities()) {
            if (ability instanceof ActivatedAbility) {
                return (ActivatedAbility) ability;
            }
        }
        throw new IllegalStateException(
                "self-test object lacks an activated ability: " + object.getClass().getName());
    }

    private static Mana redMana(int amount) {
        return new Mana(0, 0, 0, amount, 0, 0, 0, 0);
    }

    private static void expectUniformMirrorViolation(Runnable action, String label) {
        try {
            action.run();
        } catch (UniformMirrorPolicyViolation expected) {
            return;
        }
        throw new IllegalStateException(label + " did not fail closed");
    }

    static Ability passOnlyPriorityResult(List<? extends ActivatedAbility> playable,
                                          SeededUniformMirrorPolicy policy) {
        if (playable == null || policy == null) {
            throw violation("priority seam received null input");
        }
        return playable.isEmpty() ? new PassAbility() : null;
    }

    @FunctionalInterface
    interface SourceDistinctPlayableQuery<T> {
        List<T> query(boolean hidden, Zone fromZone, boolean hideDuplicates);
    }

    static <T> List<T> querySourceDistinctPlayable(SourceDistinctPlayableQuery<T> query) {
        if (query == null) {
            throw violation("source-distinct playable query is null");
        }
        return preserveSourceDistinctPlayable(query.query(true, Zone.ALL, false));
    }

    private static <T> List<T> preserveSourceDistinctPlayable(List<T> raw) {
        if (raw == null) {
            throw violation("source-distinct playable surface returned null");
        }
        List<T> preserved = new ArrayList<>(raw);
        if (preserved.size() != raw.size()) {
            throw violation("source-distinct playable surface changed cardinality");
        }
        return preserved;
    }

    static <T> List<T> mandatoryForcedCandidates(List<T> remaining,
                                                 int outstandingMandatory) {
        if (remaining == null || outstandingMandatory < 0) {
            throw violation("mandatory target seam received invalid input");
        }
        return outstandingMandatory > 0 && remaining.size() == outstandingMandatory
                ? new ArrayList<>(remaining) : null;
    }

    private static void assertPermutationOrder(List<int[]> actual, int[][] expected, String label) {
        if (actual.size() != expected.length) {
            throw new IllegalStateException("trigger permutation count failed for " + label);
        }
        for (int i = 0; i < expected.length; i++) {
            if (!Arrays.equals(actual.get(i), expected[i])) {
                throw new IllegalStateException("trigger permutation order failed for " + label);
            }
        }
    }

    private <T> T chooseCanonicalOne(String category, List<T> candidates, Game game, Ability source) {
        List<IndexedCandidate<T>> ordered = canonicalize(candidates, game, source);
        int rank = mirrorPolicy.chooseNoncombat(category, ordered.size());
        return ordered.get(rank).candidate;
    }

    private <T> List<IndexedCandidate<T>> canonicalize(List<T> candidates, Game game, Ability source) {
        if (candidates == null || candidates.isEmpty()) {
            throw violation("cannot canonicalize an empty candidate list");
        }
        List<IndexedCandidate<T>> indexed = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            T candidate = candidates.get(i);
            indexed.add(new IndexedCandidate<>(candidate, i, canonicalKey(candidate, game, source)));
        }
        indexed.sort(Comparator
                .comparing((IndexedCandidate<T> item) -> item.key)
                .thenComparingInt(item -> item.originalIndex));
        for (int i = 1; i < indexed.size(); i++) {
            if (indexed.get(i - 1).key.equals(indexed.get(i).key)) {
                canonicalEncounterTieBreaks++;
            }
        }
        return indexed;
    }

    private String canonicalKey(Object candidate, Game game, Ability source) {
        if (candidate == null) {
            return "00|sentinel|stop";
        }
        if (candidate instanceof PassAbility) {
            return "01|ability|pass";
        }
        if (candidate instanceof UUID) {
            return canonicalUuid((UUID) candidate, game);
        }
        if (candidate instanceof Permanent) {
            return canonicalPermanent((Permanent) candidate, game);
        }
        if (candidate instanceof Card) {
            return canonicalCard((Card) candidate, game);
        }
        if (candidate instanceof Ability) {
            Ability ability = (Ability) candidate;
            return "20|ability|" + canonicalUuid(ability.getSourceId(), game)
                    + "|" + candidate.getClass().getName()
                    + "|" + safe(ability.getRule())
                    + "|" + safe(String.valueOf(ability));
        }
        if (candidate instanceof Mode) {
            return "30|mode|" + candidate.getClass().getName() + "|" + safe(String.valueOf(candidate));
        }
        if (candidate instanceof String || candidate instanceof Number
                || candidate instanceof Boolean || candidate instanceof Enum) {
            return "40|scalar|" + candidate.getClass().getName() + "|" + safe(String.valueOf(candidate));
        }
        if (candidate instanceof TargetAmount) {
            TargetAmount amount = (TargetAmount) candidate;
            List<String> targets = new ArrayList<>();
            for (UUID id : amount.getTargets()) {
                targets.add(canonicalUuid(id, game) + "=" + amount.getTargetAmount(id));
            }
            Collections.sort(targets);
            return "50|target_amount|" + targets;
        }
        if (candidate instanceof MageObject) {
            MageObject object = (MageObject) candidate;
            return "60|mage_object|" + candidate.getClass().getName()
                    + "|" + safe(object.getName()) + "|" + canonicalUuid(object.getId(), game);
        }
        throw violation("unsupported canonical candidate type " + candidate.getClass().getName()
                + (source == null ? "" : " for " + source.getClass().getName()));
    }

    private String canonicalUuid(UUID id, Game game) {
        if (id == null) {
            return "uuid|null";
        }
        if (game == null) {
            throw violation("UUID candidate cannot be canonicalized without a game");
        }
        Player player = game.getPlayer(id);
        if (player != null) {
            return "10|player|" + seatFor(player.getId(), game);
        }
        Permanent permanent = game.getPermanent(id);
        if (permanent != null) {
            return canonicalPermanent(permanent, game);
        }
        int stackIndex = 0;
        for (StackObject stackObject : game.getStack()) {
            if (id.equals(stackObject.getId()) || id.equals(stackObject.getSourceId())) {
                return "13|stack|" + stackIndex + "|" + safe(stackObject.getName())
                        + "|" + stackObject.getClass().getName();
            }
            stackIndex++;
        }
        Card card = game.getCard(id);
        if (card != null) {
            return canonicalCard(card, game);
        }
        throw violation("UUID is not a player, permanent, card, or stack object");
    }

    private String canonicalPermanent(Permanent permanent, Game game) {
        int battlefieldIndex = 0;
        for (Permanent current : game.getBattlefield().getAllPermanents()) {
            if (current != null && current.getId().equals(permanent.getId())) {
                return "11|permanent|" + seatFor(permanent.getControllerId(), game)
                        + "|" + battlefieldIndex + "|" + safe(permanent.getName())
                        + "|" + permanent.getClass().getName();
            }
            battlefieldIndex++;
        }
        throw violation("permanent candidate is absent from battlefield order");
    }

    private String canonicalCard(Card card, Game game) {
        List<Player> players = playersInSeatOrder(game);
        for (Player player : players) {
            String seat = seatFor(player.getId(), game);
            int index = 0;
            for (Card current : player.getHand().getCards(game)) {
                if (sameCard(current, card)) {
                    return cardKey("hand", seat, index, card);
                }
                index++;
            }
            index = 0;
            for (UUID currentId : player.getLibrary().getCardList()) {
                if (card.getId().equals(currentId)) {
                    return cardKey("library", seat, index, card);
                }
                index++;
            }
            index = 0;
            for (Card current : player.getGraveyard().getCards(game)) {
                if (sameCard(current, card)) {
                    return cardKey("graveyard", seat, index, card);
                }
                index++;
            }
            index = 0;
            for (Card current : game.getExile().getCardsOwned(game, player.getId())) {
                if (sameCard(current, card)) {
                    return cardKey("exile", seat, index, card);
                }
                index++;
            }
        }
        int stackIndex = 0;
        for (StackObject stackObject : game.getStack()) {
            if (card.getId().equals(stackObject.getId())
                    || card.getId().equals(stackObject.getSourceId())) {
                return cardKey("stack", seatFor(card.getOwnerId(), game), stackIndex, card);
            }
            stackIndex++;
        }
        throw violation("card candidate is absent from every supported deterministic zone order");
    }

    private static boolean sameCard(Card left, Card right) {
        return left != null && right != null && left.getId().equals(right.getId());
    }

    private static String cardKey(String zone, String seat, int index, Card card) {
        return cardKeyComponents(zone, seat, index, card.getName(),
                card.getExpansionSetCode(), card.getCardNumber(), card.getClass().getName());
    }

    private static String cardKeyComponents(String zone, String seat, int index,
                                            String name, String setCode, String cardNumber,
                                            String className) {
        return "12|card|" + seat + "|" + zone + "|" + index
                + "|" + safe(name)
                + "|" + safe(setCode)
                + "|" + safe(cardNumber)
                + "|" + safe(className);
    }

    private List<Player> playersInSeatOrder(Game game) {
        Player self = game.getPlayer(getId());
        if (self == null) {
            throw violation("uniform player is absent from game");
        }
        Player p0 = "p0".equals(physicalSeat) ? self : onlyOpponent(game);
        Player p1 = "p1".equals(physicalSeat) ? self : onlyOpponent(game);
        List<Player> result = new ArrayList<>(2);
        result.add(p0);
        result.add(p1);
        return result;
    }

    private Player onlyOpponent(Game game) {
        Player opponent = null;
        for (Player player : game.getPlayers().values()) {
            if (player == null || player.getId().equals(getId())) {
                continue;
            }
            if (opponent != null) {
                throw violation("mirror benchmark encountered more than one opponent");
            }
            opponent = player;
        }
        if (opponent == null) {
            throw violation("mirror benchmark has no opponent");
        }
        return opponent;
    }

    private String seatFor(UUID playerId, Game game) {
        if (playerId == null) {
            return "none";
        }
        if (playerId.equals(getId())) {
            return physicalSeat;
        }
        for (Player player : game.getPlayers().values()) {
            if (player != null && player.getId().equals(playerId)) {
                return "p0".equals(physicalSeat) ? "p1" : "p0";
            }
        }
        return "none";
    }

    private static String category(StateSequenceBuilder.ActionType actionType) {
        return "noncombat_" + (actionType == null ? "unknown" : actionType.name().toLowerCase());
    }

    private static String requireSeat(String seat) {
        if (!"p0".equals(seat) && !"p1".equals(seat)) {
            throw new IllegalArgumentException("physicalSeat must be exactly p0 or p1");
        }
        return seat;
    }

    private static String safe(String value) {
        return value == null ? "" : value.replace('|', '/');
    }

    private static UniformMirrorPolicyViolation violation(String message) {
        return new UniformMirrorPolicyViolation(message);
    }

    private static final class IndexedCandidate<T> {
        final T candidate;
        final int originalIndex;
        final String key;

        IndexedCandidate(T candidate, int originalIndex, String key) {
            this.candidate = candidate;
            this.originalIndex = originalIndex;
            this.key = key;
        }
    }

    static final class TriggerOrderCache implements Serializable {
        private static final long serialVersionUID = 1L;

        final UUID gameId;
        final List<TriggerOffer> selectedOrder;
        final boolean snapshotDrift;
        int cursor;

        TriggerOrderCache(UUID gameId, List<TriggerOffer> selectedOrder, boolean snapshotDrift) {
            if (gameId == null || selectedOrder == null || selectedOrder.size() < 2) {
                throw violation("invalid trigger-order cache construction");
            }
            this.gameId = gameId;
            this.selectedOrder = new ArrayList<>(selectedOrder);
            this.snapshotDrift = snapshotDrift;
        }

        private TriggerOrderCache(TriggerOrderCache source, boolean snapshotDrift) {
            this.gameId = source.gameId;
            this.selectedOrder = new ArrayList<>(source.selectedOrder);
            this.snapshotDrift = snapshotDrift;
            this.cursor = source.cursor;
        }

        TriggerOrderCache copyAsSnapshotDrift() {
            return new TriggerOrderCache(this, true);
        }
    }

    static final class TriggerOffer implements Serializable {
        private static final long serialVersionUID = 1L;

        final UUID identity;
        final String canonicalKey;
        final TriggeredAbility instance;

        TriggerOffer(UUID identity, String canonicalKey, TriggeredAbility instance) {
            this.identity = identity;
            this.canonicalKey = canonicalKey;
            this.instance = instance;
        }
    }

    static final class TriggerConsumeResult {
        final TriggeredAbility ability;
        final boolean complete;

        TriggerConsumeResult(TriggeredAbility ability, boolean complete) {
            this.ability = ability;
            this.complete = complete;
        }
    }

    public static final class UniformMirrorPolicyViolation extends RuntimeException {
        private static final long serialVersionUID = 1L;

        UniformMirrorPolicyViolation(String message) {
            super(message);
        }
    }

    private static final class FailFastPythonModel implements PythonModel {
        @Override
        public PythonMLBatchManager.PredictionResult scoreCandidates(
                StateSequenceBuilder.SequenceOutput state, int[] candidateActionIds,
                float[][] candidateFeatures, int[] candidateMask, String policyKey,
                String headId, int pickIndex, int minTargets, int maxTargets) {
            throw new AssertionError("uniform mirror attempted model inference");
        }

        @Override
        public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData,
                                    List<Double> rewards) {
            throw new AssertionError("uniform mirror attempted training persistence");
        }

        @Override
        public void saveModel(String path) {
            throw new AssertionError("uniform mirror attempted model persistence");
        }

        @Override
        public String getDeviceInfo() {
            return "fail-fast-uniform-mirror";
        }

        @Override
        public Map<String, Integer> getMainModelTrainingStats() {
            return Collections.emptyMap();
        }

        @Override
        public Map<String, Integer> getHealthStats() {
            return Collections.emptyMap();
        }

        @Override
        public void resetHealthStats() {
        }

        @Override
        public void recordGameResult(float lastValuePrediction, boolean won) {
            throw new AssertionError("uniform mirror attempted result persistence");
        }

        @Override
        public Map<String, Object> getValueHeadMetrics() {
            return Collections.emptyMap();
        }

        @Override
        public void shutdown() {
        }
    }
}
