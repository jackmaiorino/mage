package mage.player.ai.rl;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.Mana;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.PlayLandAbility;
import mage.abilities.SpellAbility;
import mage.abilities.TriggeredAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.Cost;
import mage.abilities.costs.common.DiscardTargetCost;
import mage.abilities.mana.ManaAbility;
import mage.abilities.mana.ManaOptions;
import mage.cards.Card;
import mage.constants.Outcome;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.Target;
import mage.target.common.TargetCardInHand;
import mage.target.common.TargetDiscard;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/**
 * Applies real CP7 Rally decisions to the deterministic Rust shadow episode.
 *
 * <p>This is intentionally a Rally-only engineering bridge. It covers the six
 * decision families observed in the retained Rally corpus and fails closed on
 * every other CP7 callback. Original card identity is bound from the exact
 * opening hand plus library order. Token identity is learned only from an
 * exact ordered candidate correspondence or an unambiguous priority menu.</p>
 *
 * <p>CP7 simulation copies share the observer dispatch state, but
 * {@link RallyCp7DecisionObserver.DispatchState} suppresses all events from a
 * {@code game.isSimulation()} copy. Consequently this class is reached only
 * by the real game and no CP7 search copy can advance the live Rust process.</p>
 */
public final class RallyCp7KernelShadowMapper implements RallyCp7DecisionObserver {

    private static final int RALLY_DECK_SIZE = 60;
    private static final int OPENING_HAND_SIZE = 7;
    private static final int INITIAL_OBJECT_COUNT = RALLY_DECK_SIZE * 2;

    private static final Set<String> TARGET_KINDS = immutableSet(
            "choose_target", "choose_effect_target");
    private static final Set<String> CARD_TARGET_KINDS = immutableSet(
            "choose_target", "choose_cost_target", "choose_effect_target", "discard");
    private static final Set<String> FINISH_TARGET_KINDS = immutableSet(
            "finish_target_selection", "finish_effect_selection");
    private static final Set<String> BOOLEAN_KINDS = immutableSet(
            "choose_kicker", "choose_effect_boolean", "choose_optional_cost_use",
            "choose_madness_cast", "choose_spell_copy_payment",
            "choose_spell_copy_retarget");
    private static final Set<String> PRIORITY_KINDS = immutableSet(
            "pass", "play_land", "cast_spell", "activate_mana_ability",
            "activate_ability");

    /** Frozen card-definition ids used to verify the opening identity join. */
    private static final Map<String, Integer> RALLY_CARD_IDS = rallyCardIds();

    private final XMageRallyBridgeProcessClient bridge;
    private final long episodeId;
    private final XMageRallyBridgeProtocol.Seat physicalSeat;
    private final String expectedPlayerName;
    private final Map<UUID, Integer> initialArenaIds;
    private final Map<UUID, StableBinding> uuidBindings = new LinkedHashMap<>();
    private final Map<Integer, UUID> arenaBindings = new HashMap<>();
    private final Map<String, Long> appliedKinds = new LinkedHashMap<>();
    private final Set<Long> appliedPhysicalDecisionIds = new LinkedHashSet<>();

    private UUID physicalPlayerId;
    private UUID p0PlayerId;
    private UUID p1PlayerId;
    private long lastObserverSequence;
    private long requestOrdinal;
    private long appliedPolicySteps;
    private long forcedNoPolicyEvents;
    private String lastAppliedActionKind;
    private Integer lastAppliedSourceArenaId;
    private TriggerOrderReplay triggerOrderReplay;
    private boolean openingIdentityBound;
    private boolean failed;
    private String firstFailure;

    public RallyCp7KernelShadowMapper(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            String physicalSeat,
            String expectedPlayerName) {
        this(bridge, episodeId, parseSeat(physicalSeat), expectedPlayerName,
                Collections.emptyMap());
    }

    public RallyCp7KernelShadowMapper(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            XMageRallyBridgeProtocol.Seat physicalSeat,
            String expectedPlayerName) {
        this(bridge, episodeId, physicalSeat, expectedPlayerName,
                Collections.emptyMap());
    }

    public RallyCp7KernelShadowMapper(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            XMageRallyBridgeProtocol.Seat physicalSeat,
            String expectedPlayerName,
            Map<UUID, Integer> initialArenaIds) {
        if (bridge == null) {
            throw new IllegalArgumentException("bridge must not be null");
        }
        if (episodeId < 0L) {
            throw new IllegalArgumentException("episodeId must be nonnegative");
        }
        if (physicalSeat == null) {
            throw new IllegalArgumentException("physicalSeat must not be null");
        }
        if (expectedPlayerName == null || expectedPlayerName.trim().isEmpty()) {
            throw new IllegalArgumentException("expectedPlayerName must be nonempty");
        }
        this.initialArenaIds = validateInitialArenaIds(initialArenaIds);
        try {
            requireExternalDeterministicCp7Configuration();
        } catch (MapperViolation error) {
            bridge.close();
            throw error;
        }
        if (!bridge.isUsable()) {
            throw new MapperViolation("bridge is not usable");
        }
        if (!Long.valueOf(episodeId).equals(bridge.getActiveEpisodeId())) {
            bridge.close();
            throw new MapperViolation("bridge active episode does not match CP7 mapper episode");
        }
        if (bridge.getCandidateSeat() == physicalSeat) {
            bridge.close();
            throw new MapperViolation("CP7 mapper must be bound to the noncandidate seat");
        }
        this.bridge = bridge;
        this.episodeId = episodeId;
        this.physicalSeat = physicalSeat;
        this.expectedPlayerName = expectedPlayerName.trim();
    }

    /** Optional stronger binding once the observed player has been constructed. */
    public synchronized void bindPlayer(RallyCp7ObservedPlayer player) {
        if (player == null) {
            throw fail("cannot bind a null CP7 player", null);
        }
        bindPlayerId(player.getId());
    }

    public synchronized void bindPlayerId(UUID playerId) {
        requireLive();
        if (playerId == null) {
            throw fail("cannot bind a null CP7 player id", null);
        }
        if (physicalPlayerId != null && !physicalPlayerId.equals(playerId)) {
            throw fail("CP7 player id was rebound", null);
        }
        physicalPlayerId = playerId;
    }

    @Override
    public synchronized void onDecision(RallyCp7DecisionObserver.Decision observed) {
        requireLive();
        if (observed == null) {
            throw fail("CP7 observer delivered a null decision", null);
        }
        if (observed.getSequence() != lastObserverSequence + 1L) {
            throw fail("CP7 observer sequence is not contiguous: expected "
                    + (lastObserverSequence + 1L) + " got " + observed.getSequence(), null);
        }
        lastObserverSequence = observed.getSequence();
        Game game = observed.getGame();
        if (game == null || game.isSimulation()) {
            throw fail("CP7 mapper received a missing or simulated game", null);
        }
        bindOpeningIdentity(game);

        try {
            switch (observed.getKind()) {
                case MULLIGAN:
                    applyFixedMulligan(observed);
                    break;
                case PRIORITY_PASS:
                    applyPriorityPass(observed);
                    break;
                case PRIORITY_ACTION:
                    applyPriorityAction(observed);
                    break;
                case PRESELECTED_TARGETS:
                    applyObjectTargets(observed, TARGET_KINDS);
                    break;
                case TARGET:
                case CARD_TARGET:
                    applyObjectTargets(observed, CARD_TARGET_KINDS);
                    break;
                case CHOOSE_USE:
                    applyBooleanChoice(observed);
                    break;
                case DECLARE_ATTACKERS:
                    applyAttackers(observed);
                    break;
                case DECLARE_BLOCKERS:
                    applyBlockers(observed);
                    break;
                case TRIGGER_ORDER:
                    applyTriggerOrder(observed);
                    break;
                default:
                    throw fail("unsupported retained-Rally CP7 callback: "
                            + observed.getKind(), null);
            }
        } catch (MapperViolation error) {
            if (failed) {
                throw error;
            }
            throw fail(error.getMessage(), error);
        } catch (RuntimeException error) {
            throw fail("CP7 mapping failed at " + observed.getKind(), error);
        }
    }

    public synchronized long getAppliedPolicySteps() {
        return appliedPolicySteps;
    }

    public synchronized long getAppliedPhysicalDecisionCount() {
        return appliedPhysicalDecisionIds.size();
    }

    public synchronized long getForcedNoPolicyEvents() {
        return forcedNoPolicyEvents;
    }

    public synchronized Map<String, Long> getAppliedKinds() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(appliedKinds));
    }

    public synchronized boolean isFailed() {
        return failed;
    }

    public synchronized String getFirstFailure() {
        return firstFailure;
    }

    private void applyFixedMulligan(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted()
                || observed.getSelected().size() != 1
                || !Boolean.FALSE.equals(observed.getSelected().get(0))) {
            throw fail("Rally CP7 did not keep its fixed opening seven", null);
        }
        forcedNoPolicyEvents++;
        increment(appliedKinds, "fixed_mulligan");
    }

    private void applyPriorityPass(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted()
                || observed.getCandidates().isEmpty()
                || !observed.getSelected().isEmpty()) {
            throw fail("CP7 priority pass has an invalid observer shape", null);
        }
        if (bridge.getCurrentDecision() == null && bridge.getTerminal() != null) {
            recordForcedNoPolicy("forced_priority_pass_after_native_terminal");
            return;
        }
        XMageRallyBridgeProtocol.DecisionBody current = currentForCp7OrNull();
        if (current == null) {
            tracePriorityPass("forced_other_actor", observed,
                    bridge.getCurrentDecision());
            recordForcedNoPolicy("forced_priority_pass");
            return;
        }
        if (!matchesPriorityMenu(current, observed.getCandidates(), observed.getGame())) {
            tracePriorityPass("forced_menu_mismatch", observed, current);
            recordForcedNoPolicy("forced_priority_pass_menu_mismatch");
            return;
        }
        int selected = -1;
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            if (!"pass".equals(current.getActionSemantics().get(i).getActionKind())) {
                continue;
            }
            if (selected >= 0) {
                throw fail("Rust priority decision has multiple pass actions", null);
            }
            selected = i;
        }
        if (selected < 0) {
            recordForcedNoPolicy("forced_priority_pass_nonpriority_cursor");
            return;
        }
        requireSurface(current, "priority pass");
        tracePriorityPass("step", observed, current);
        step(current, selected, "priority_pass", observed.getGame());
    }

    private void tracePriorityPass(
            String outcome,
            RallyCp7DecisionObserver.Decision observed,
            XMageRallyBridgeProtocol.DecisionBody current) {
        if (!Boolean.getBoolean("xmage.rally.traceCp7Mapper")) {
            return;
        }
        List<String> xmage = new ArrayList<>();
        for (Object raw : observed.getCandidates()) {
            if (!(raw instanceof Ability)) {
                xmage.add("non_ability");
                continue;
            }
            Ability ability = (Ability) raw;
            UUID sourceId = ability.getSourceId();
            StableBinding binding = sourceId == null ? null : uuidBindings.get(sourceId);
            MageObject object = sourceId == null
                    ? null : observed.getGame().getObject(sourceId);
            xmage.add((ability instanceof PassAbility ? "pass" : priorityKind(ability))
                    + "@" + bindingSummary(binding)
                    + ":" + (object == null ? "missing" : object.getName())
                    + ":" + (sourceId == null ? "none"
                    : observed.getGame().getState().getZone(sourceId)));
        }
        List<String> rust = new ArrayList<>();
        if (current != null && current.getActionSemantics() != null) {
            for (XMageRallyBridgeProtocol.ActionSemantic semantic
                    : current.getActionSemantics()) {
                JsonObject row = parseSemantic(semantic);
                Integer arena = row.has("source")
                        ? requiredInt(requiredObject(
                        row, "source", semantic.getActionKind()),
                        "arena_id", semantic.getActionKind()) : null;
                rust.add(semantic.getActionKind() + "@" + arena);
            }
        }
        String rustSelected = "none";
        if (current != null && current.getSelectedActionIndex() != null) {
            int selectedIndex = current.getSelectedActionIndex();
            if (current.getActionSemantics() == null
                    || selectedIndex < 0
                    || selectedIndex >= current.getActionSemantics().size()) {
                rustSelected = selectedIndex + ":invalid";
            } else {
                rustSelected = selectedIndex + ":"
                        + current.getActionSemantics().get(selectedIndex).getActionKind();
            }
        }
        System.err.println("XMAGE_RALLY_CP7_PASS_TRACE"
                + " episode=" + episodeId
                + " outcome=" + outcome
                + " turn=" + observed.getGame().getTurnNum()
                + " phase=" + observed.getGame().getTurnStepType()
                + " stack=" + observed.getGame().getStack().size()
                + " rust_actor=" + (current == null ? "none"
                : current.getActingPlayer().wire())
                + " rust_step=" + (current == null ? -1 : current.getStep())
                + " rust_kind=" + (current == null ? "none"
                : current.getDecisionKind())
                + " rust_selected=" + rustSelected
                + " xmage=" + xmage
                + " rust=" + rust);
    }

    private boolean matchesPriorityMenu(
            XMageRallyBridgeProtocol.DecisionBody current,
            List<Object> observedCandidates,
            Game game) {
        requireCommon(current, "priority pass menu");
        if (!"surface".equals(current.getDecisionKind())
                || current.getSubstepIndex() != 0
                || current.getSubstepCount() != 1
                || current.getActionSemantics().size() != observedCandidates.size()) {
            return false;
        }
        List<Ability> xmage = new ArrayList<>(observedCandidates.size());
        int xmagePasses = 0;
        for (Object raw : observedCandidates) {
            if (!(raw instanceof Ability)) {
                throw fail("CP7 priority pass candidate is not an ability", null);
            }
            Ability ability = (Ability) raw;
            if (ability instanceof PassAbility) {
                xmagePasses++;
            } else if (ability.getSourceId() == null) {
                throw fail("CP7 priority pass candidate has no source id", null);
            }
            xmage.add(ability);
        }
        if (xmagePasses != 1) {
            throw fail("CP7 priority pass menu must contain exactly one pass", null);
        }
        prebindPriorityTokenSources(current, game);
        boolean[] used = new boolean[xmage.size()];
        int rustPasses = 0;
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : current.getActionSemantics()) {
            String kind = semantic.getActionKind();
            if (!PRIORITY_KINDS.contains(kind)) {
                return false;
            }
            JsonObject row = parseSemantic(semantic);
            JsonObject source = "pass".equals(kind)
                    ? null : requiredObject(row, "source", kind);
            int match = -1;
            for (int i = 0; i < xmage.size(); i++) {
                if (used[i]) {
                    continue;
                }
                Ability ability = xmage.get(i);
                boolean matches = "pass".equals(kind)
                        ? ability instanceof PassAbility
                        : !(ability instanceof PassAbility)
                        && kind.equals(priorityKind(ability))
                        && matchesBoundUuid(ability.getSourceId(), source, game);
                if (matches) {
                    match = i;
                    break;
                }
            }
            if (match < 0) {
                return false;
            }
            used[match] = true;
            if ("pass".equals(kind)) {
                rustPasses++;
            }
        }
        return rustPasses == 1;
    }

    private void applyPriorityAction(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted() || observed.getSelected().size() != 1
                || !(observed.getSelected().get(0) instanceof Ability)) {
            throw fail("CP7 priority action has an invalid observer shape", null);
        }
        Ability ability = (Ability) observed.getSelected().get(0);
        if (ability.getSourceId() == null) {
            throw fail("CP7 priority action has no source id", null);
        }
        XMageRallyBridgeProtocol.DecisionBody current = requireCurrentCp7(
                observed.getGame(), "priority action");
        requireSurface(current, "priority action");
        String expectedKind = priorityKind(ability);
        prebindPriorityTokenSources(current, observed.getGame());

        List<Integer> matches = new ArrayList<>();
        List<String> expectedRows = new ArrayList<>();
        List<String> actionKinds = new ArrayList<>();
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            XMageRallyBridgeProtocol.ActionSemantic semantic =
                    current.getActionSemantics().get(i);
            actionKinds.add(semantic.getActionKind());
            if (!expectedKind.equals(semantic.getActionKind())) {
                continue;
            }
            JsonObject row = parseSemantic(semantic);
            JsonObject source = requiredObject(row, "source", expectedKind);
            boolean sourceMatches = matchesBoundUuid(
                    ability.getSourceId(), source, observed.getGame());
            expectedRows.add("arena=" + requiredInt(source, "arena_id", expectedKind)
                    + "/card=" + requiredInt(source, "card_db_id", expectedKind)
                    + "/matches=" + sourceMatches);
            if (sourceMatches) {
                matches.add(i);
            }
        }
        if (matches.size() != 1) {
            MageObject sourceObject = observed.getGame().getObject(ability.getSourceId());
            StableBinding bound = uuidBindings.get(ability.getSourceId());
            throw fail("CP7 priority action did not identify exactly one Rust action: kind="
                    + expectedKind + " matches=" + matches.size()
                    + " phase=" + observed.getGame().getTurnStepType()
                    + " rust_step=" + current.getStep()
                    + " rust_decision=" + current.getDecisionKind()
                    + " rust_kinds=" + actionKinds
                    + " xmage_source="
                    + (sourceObject == null ? "missing" : sourceObject.getName())
                    + " xmage_uuid=" + ability.getSourceId()
                    + " bound=" + (bound == null ? "none"
                    : bound.arenaId + "/" + bound.cardDbId)
                    + " expected_rows=" + expectedRows, null);
        }
        step(current, matches.get(0), "priority_action:" + expectedKind,
                observed.getGame());
    }

    private void applyObjectTargets(
            RallyCp7DecisionObserver.Decision observed,
            Set<String> allowedKinds) {
        if (!observed.isAccepted()) {
            throw fail(observed.getKind() + " was not accepted", null);
        }
        List<UUID> selected = uuidValues(observed.getSelected(), "selected target");
        List<UUID> candidates = objectIds(observed.getCandidates(), "target candidate");
        if (observed.getKind() == RallyCp7DecisionObserver.Kind.PRESELECTED_TARGETS
                && observed.getSubject() instanceof Target) {
            Target target = (Target) observed.getSubject();
            Set<UUID> reconstructed = new LinkedHashSet<>(target.possibleTargets(
                    target.getAffectedAbilityControllerId(physicalPlayerId),
                    observed.getSource(), observed.getGame()));
            reconstructed.addAll(selected);
            candidates = new ArrayList<>(reconstructed);
        }
        if (selected.isEmpty()) {
            XMageRallyBridgeProtocol.DecisionBody current = currentForCp7OrNull();
            if (current == null) {
                recordForcedNoPolicy("forced_empty_target");
                return;
            }
            requireSurface(current, "empty target");
            int finish = uniqueKindIndex(current, FINISH_TARGET_KINDS);
            step(current, finish, "finish_target_selection", observed.getGame());
            return;
        }

        Set<UUID> remainingCandidates = new LinkedHashSet<>(candidates);
        for (UUID target : selected) {
            XMageRallyBridgeProtocol.DecisionBody current = currentForCp7OrNull();
            if (current == null) {
                if (isForcedSingleBloodDiscard(observed, selected, candidates, current)) {
                    recordForcedNoPolicy("forced_single_blood_discard");
                    return;
                }
                throw fail("Rust has no CP7 target decision for an observed selection", null);
            }
            if (isForcedSingleBloodDiscard(observed, selected, candidates, current)) {
                recordForcedNoPolicy("forced_single_blood_discard");
                return;
            }
            requireSurface(current, "target");
            prebindCommonSource(current, observed.getSource(), observed.getGame());
            prebindOrderedTargetCandidates(
                    current, remainingCandidates, allowedKinds, observed.getGame());
            requireTargetMenuBijection(
                    current, remainingCandidates, allowedKinds,
                    observed.getSource(), observed.getGame());
            int index = selectedTargetIndex(
                    current, target, allowedKinds, observed.getSource(), observed.getGame());
            UUID appliedTarget = canonicalEquivalentDiscardTarget(
                    observed, current, target, remainingCandidates, index);
            if (!appliedTarget.equals(target)) {
                replaceObservedSingleCardTarget(observed, target, appliedTarget);
                index = selectedTargetIndex(
                        current, appliedTarget, allowedKinds,
                        observed.getSource(), observed.getGame());
                if (!"discard".equals(
                        current.getActionSemantics().get(index).getActionKind())) {
                    throw fail("canonical discard target did not map to a discard action", null);
                }
            }
            step(current, index,
                    "target:" + current.getActionSemantics().get(index).getActionKind(),
                    observed.getGame());
            remainingCandidates.remove(appliedTarget);
        }
    }

    /**
     * CP7's final comparator uses process-local UUIDs to break ties between
     * otherwise identical hand cards. The Rust surface already supplies an
     * exact opening arena identity, so use its lowest occurrence only when the
     * selected action is a one-card discard and every game-visible property of
     * the alternative copy matches. This changes physical identity, not CP7's
     * semantic card choice.
     */
    private UUID canonicalEquivalentDiscardTarget(
            RallyCp7DecisionObserver.Decision observed,
            XMageRallyBridgeProtocol.DecisionBody current,
            UUID selected,
            Collection<UUID> candidates,
            int selectedIndex) {
        if ((!(observed.getSubject() instanceof TargetCardInHand)
                && !(observed.getSubject() instanceof TargetDiscard))
                || observed.getSelected().size() != 1
                || !"discard".equals(current.getActionSemantics()
                .get(selectedIndex).getActionKind())) {
            return selected;
        }
        StableBinding selectedBinding = uuidBindings.get(selected);
        Card selectedCard = observed.getGame().getCard(selected);
        if (selectedBinding == null || selectedBinding.arenaId >= INITIAL_OBJECT_COUNT
                || selectedCard == null
                || observed.getGame().getState().getZone(selected) != mage.constants.Zone.HAND) {
            return selected;
        }

        UUID canonical = selected;
        int canonicalArenaId = selectedBinding.arenaId;
        int selectedZoneChangeCount = selectedCard.getZoneChangeCounter(observed.getGame());
        for (UUID candidateId : candidates) {
            StableBinding candidateBinding = uuidBindings.get(candidateId);
            Card candidateCard = observed.getGame().getCard(candidateId);
            if (candidateBinding == null
                    || candidateBinding.arenaId >= INITIAL_OBJECT_COUNT
                    || candidateBinding.cardDbId != selectedBinding.cardDbId
                    || candidateCard == null
                    || !selectedCard.getClass().equals(candidateCard.getClass())
                    || !selectedCard.getName().equals(candidateCard.getName())
                    || !selectedCard.getOwnerId().equals(candidateCard.getOwnerId())
                    || candidateCard.getZoneChangeCounter(observed.getGame())
                    != selectedZoneChangeCount
                    || observed.getGame().getState().getZone(candidateId)
                    != mage.constants.Zone.HAND) {
                continue;
            }
            if (candidateBinding.arenaId < canonicalArenaId) {
                canonical = candidateId;
                canonicalArenaId = candidateBinding.arenaId;
            }
        }
        return canonical;
    }

    private void replaceObservedSingleCardTarget(
            RallyCp7DecisionObserver.Decision observed,
            UUID selected,
            UUID canonical) {
        Target target = (Target) observed.getSubject();
        List<UUID> pending = target.getTargets();
        if (pending.size() != 1 || !selected.equals(pending.get(0))
                || target.getTargetAmount(selected) != 0) {
            throw fail("cannot canonicalize a non-singleton discard target", null);
        }
        target.remove(selected);
        target.add(canonical, observed.getGame());
        List<UUID> replaced = target.getTargets();
        if (replaced.size() != 1 || !canonical.equals(replaced.get(0))) {
            throw fail("canonical discard target replacement was not retained", null);
        }
    }

    private boolean isForcedSingleBloodDiscard(
            RallyCp7DecisionObserver.Decision observed,
            List<UUID> selected,
            List<UUID> candidates,
            XMageRallyBridgeProtocol.DecisionBody current) {
        if (selected.size() != 1 || candidates.size() != 1
                || !selected.get(0).equals(candidates.get(0))
                || !(observed.getSubject() instanceof TargetCardInHand)
                || observed.getSource() == null
                || observed.getSource().getSourceId() == null
                || current != null && hasAnyKind(current, CARD_TARGET_KINDS)) {
            return false;
        }
        TargetCardInHand target = (TargetCardInHand) observed.getSubject();
        if (target.getMinNumberOfTargets() != 1
                || target.getMaxNumberOfTargets() != 1) {
            return false;
        }
        boolean hasDiscardTargetCost = false;
        for (Cost cost : observed.getSource().getCosts()) {
            if (cost instanceof DiscardTargetCost) {
                hasDiscardTargetCost = true;
                break;
            }
        }
        UUID sourceId = observed.getSource().getSourceId();
        StableBinding sourceBinding = uuidBindings.get(sourceId);
        MageObject sourceObject = observed.getGame().getObject(sourceId);
        return hasDiscardTargetCost
                && sourceBinding != null && sourceBinding.cardDbId == 132
                && sourceObject != null && "Blood Token".equals(sourceObject.getName())
                && "activate_ability".equals(lastAppliedActionKind)
                && Objects.equals(lastAppliedSourceArenaId, sourceBinding.arenaId)
                && observed.getGame().getState().getZone(selected.get(0))
                == mage.constants.Zone.HAND;
    }

    private static boolean hasAnyKind(
            XMageRallyBridgeProtocol.DecisionBody current,
            Set<String> kinds) {
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : current.getActionSemantics()) {
            if (kinds.contains(semantic.getActionKind())) {
                return true;
            }
        }
        return false;
    }

    private void applyBooleanChoice(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted() || observed.getSelected().size() != 1
                || !(observed.getSelected().get(0) instanceof Boolean)) {
            throw fail("chooseUse has an invalid observer shape", null);
        }
        boolean selectedValue = (Boolean) observed.getSelected().get(0);
        XMageRallyBridgeProtocol.DecisionBody current = currentForCp7OrNull();
        if (current == null) {
            if (!selectedValue) {
                recordForcedNoPolicy("forced_false_choose_use");
                return;
            }
            throw fail("Rust has no CP7 boolean decision for observed true", null);
        }
        boolean hasRustBoolean = false;
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : current.getActionSemantics()) {
            if (BOOLEAN_KINDS.contains(semantic.getActionKind())) {
                hasRustBoolean = true;
                break;
            }
        }
        if (!hasRustBoolean
                && isForcedJointlyUnaffordableBushwhackerKicker(
                observed, selectedValue)) {
            recordForcedNoPolicy("forced_false_unaffordable_kicker");
            return;
        }
        requireSurface(current, "chooseUse");
        prebindCommonSource(current, observed.getSource(), observed.getGame());
        List<Integer> matches = new ArrayList<>();
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            XMageRallyBridgeProtocol.ActionSemantic semantic =
                    current.getActionSemantics().get(i);
            if (!BOOLEAN_KINDS.contains(semantic.getActionKind())) {
                continue;
            }
            JsonObject row = parseSemantic(semantic);
            String field = booleanField(semantic.getActionKind());
            if (requiredBoolean(row, field, semantic.getActionKind()) != selectedValue) {
                continue;
            }
            if (observed.getSource() != null && row.has("source")
                    && !matchesBoundUuid(observed.getSource().getSourceId(),
                    requiredObject(row, "source", semantic.getActionKind()),
                    observed.getGame())) {
                continue;
            }
            matches.add(i);
        }
        if (matches.size() != 1) {
            throw fail("chooseUse did not identify exactly one Rust action; matches="
                    + matches.size(), null);
        }
        step(current, matches.get(0), "choose_use", observed.getGame());
    }

    private boolean isForcedJointlyUnaffordableBushwhackerKicker(
            RallyCp7DecisionObserver.Decision observed,
            boolean selectedValue) {
        Ability source = observed.getSource();
        if (selectedValue
                || observed.getSubject() != Outcome.AIDontUseIt
                || !"Pay Kicker {R} ?".equals(observed.getPrompt())
                || !(source instanceof SpellAbility)
                || source.getSourceId() == null) {
            return false;
        }
        StableBinding binding = uuidBindings.get(source.getSourceId());
        MageObject object = observed.getGame().getObject(source.getSourceId());
        if (binding == null || binding.cardDbId != 44
                || object == null || !"Goblin Bushwhacker".equals(object.getName())) {
            return false;
        }
        Player player = requirePhysicalPlayer(observed.getGame());
        Mana additional = new Mana();
        additional.increaseRed();
        return isJointlyUnaffordable(
                source.getManaCostsToPay().getMana(), additional,
                player.getManaAvailable(observed.getGame()));
    }

    private static boolean isJointlyUnaffordable(
            Mana baseCost,
            Mana additionalCost,
            ManaOptions available) {
        if (baseCost == null || additionalCost == null || available == null) {
            throw new MapperViolation("joint affordability seam received null input");
        }
        Mana total = baseCost.copy();
        total.add(additionalCost);
        return !available.enough(total);
    }

    private void applyTriggerOrder(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted() || observed.getSelected().size() != 1
                || !(observed.getSelected().get(0) instanceof TriggeredAbility)) {
            throw fail("trigger order has an invalid observer shape", null);
        }
        List<TriggeredAbility> offered = triggeredAbilities(
                observed.getCandidates(), "trigger-order candidate");
        TriggeredAbility selected = (TriggeredAbility) observed.getSelected().get(0);
        if (triggerOrderReplay != null) {
            boolean complete = triggerOrderReplay.consume(
                    offered, selected, observed.getGame().getId());
            recordForcedNoPolicy("trigger_order_replay");
            if (complete) {
                triggerOrderReplay = null;
            }
            return;
        }
        if (offered.size() < 2 || selected != offered.get(0)) {
            throw fail("CP7 trigger ordering is not the exact first-offered contract", null);
        }

        XMageRallyBridgeProtocol.DecisionBody current =
                requireCurrentCp7(observed.getGame(), "trigger order");
        requireSurface(current, "trigger order");
        int expectedWidth = factorialExact(offered.size());
        if (current.getLegalActionCount() != expectedWidth) {
            throw fail("trigger-order factorial width differs between XMage and Rust: "
                    + "xmage=" + expectedWidth
                    + " rust=" + current.getLegalActionCount(), null);
        }

        JsonObject firstRow = parseSemantic(current.getActionSemantics().get(0));
        if (!"order_triggers".equals(
                current.getActionSemantics().get(0).getActionKind())) {
            throw fail("Rust trigger order has a non-order semantic", null);
        }
        List<JsonObject> pendingSources = requiredObjectArray(
                firstRow, "pending_sources", "trigger order");
        if (pendingSources.size() != offered.size()) {
            throw fail("trigger-order pending source width differs from XMage", null);
        }

        List<Integer> desiredOrder = new ArrayList<>(offered.size());
        Set<Integer> usedPending = new HashSet<>();
        Set<UUID> abilityIds = new HashSet<>();
        Set<Integer> sourceArenas = new HashSet<>();
        for (TriggeredAbility ability : offered) {
            if (ability == null || ability.getId() == null
                    || ability.getSourceId() == null
                    || !abilityIds.add(ability.getId())) {
                throw fail("trigger-order candidates lack unique live ability ids", null);
            }
            StableBinding sourceBinding = uuidBindings.get(ability.getSourceId());
            if (sourceBinding == null
                    || !isPromptFreeRallyTriggerSource(sourceBinding.cardDbId)
                    || !sourceArenas.add(sourceBinding.arenaId)) {
                throw fail("trigger-order candidates lack unique bound source arenas", null);
            }
            int matched = -1;
            for (int i = 0; i < pendingSources.size(); i++) {
                JsonObject pending = pendingSources.get(i);
                if (requiredInt(pending, "arena_id", "trigger pending source")
                        == sourceBinding.arenaId
                        && matchesBoundUuid(ability.getSourceId(), pending,
                        observed.getGame())) {
                    if (matched >= 0) {
                        throw fail("trigger source maps to multiple Rust pending rows", null);
                    }
                    matched = i;
                }
            }
            if (matched < 0 || !usedPending.add(matched)) {
                throw fail("trigger source does not map one-to-one to Rust pending rows", null);
            }
            desiredOrder.add(matched);
        }

        int selectedIndex = -1;
        Set<String> seenOrders = new HashSet<>();
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            XMageRallyBridgeProtocol.ActionSemantic semantic =
                    current.getActionSemantics().get(i);
            if (!"order_triggers".equals(semantic.getActionKind())) {
                throw fail("Rust trigger-order menu contains "
                        + semantic.getActionKind(), null);
            }
            JsonObject row = parseSemantic(semantic);
            if (!pendingSources.equals(requiredObjectArray(
                    row, "pending_sources", "trigger order"))) {
                throw fail("Rust trigger-order rows disagree on pending sources", null);
            }
            List<Integer> order = requiredIntArray(
                    row, "order", "trigger order");
            requirePermutation(order, offered.size(), "trigger order");
            if (!seenOrders.add(order.toString())) {
                throw fail("Rust trigger-order menu repeats a permutation", null);
            }
            if (order.equals(desiredOrder)) {
                if (selectedIndex >= 0) {
                    throw fail("CP7 trigger order matches multiple Rust actions", null);
                }
                selectedIndex = i;
            }
        }
        if (seenOrders.size() != expectedWidth || selectedIndex < 0) {
            throw fail("Rust trigger-order menu is incomplete or lacks the CP7 order", null);
        }

        if (Boolean.getBoolean("xmage.rally.traceCp7Mapper")) {
            System.err.println("XMAGE_RALLY_CP7_TRIGGER_TRACE"
                    + " episode=" + episodeId
                    + " turn=" + observed.getGame().getTurnNum()
                    + " phase=" + observed.getGame().getTurnStepType()
                    + " rust_step=" + current.getStep()
                    + " desired_order=" + desiredOrder
                    + " selected_index=" + selectedIndex
                    + " pending_sources=" + pendingSources);
        }
        step(current, selectedIndex, "trigger_order", observed.getGame());
        if (offered.size() > 2) {
            triggerOrderReplay = new TriggerOrderReplay(
                    observed.getGame().getId(), offered, 1);
        }
    }

    private static boolean isPromptFreeRallyTriggerSource(int cardDbId) {
        return cardDbId == 10   // Burning-Tree Emissary
                || cardDbId == 16   // Clockwork Percussionist
                || cardDbId == 30   // Experimental Synthesizer
                || cardDbId == 44   // Goblin Bushwhacker
                || cardDbId == 127; // Voldaren Epicure
    }

    private static List<TriggeredAbility> triggeredAbilities(
            Collection<?> values,
            String label) {
        List<TriggeredAbility> result = new ArrayList<>();
        for (Object value : values) {
            if (!(value instanceof TriggeredAbility)) {
                throw new MapperViolation(label + " is not a triggered ability");
            }
            result.add((TriggeredAbility) value);
        }
        return result;
    }

    private static int factorialExact(int value) {
        if (value < 2) {
            throw new MapperViolation("trigger-order factorial domain is below two");
        }
        int result = 1;
        for (int i = 2; i <= value; i++) {
            if (result > Integer.MAX_VALUE / i) {
                throw new MapperViolation("trigger-order factorial width overflows int");
            }
            result *= i;
        }
        return result;
    }

    private static void requirePermutation(
            List<Integer> order,
            int width,
            String label) {
        if (order.size() != width) {
            throw new MapperViolation(label + " permutation has the wrong width");
        }
        boolean[] seen = new boolean[width];
        for (int index : order) {
            if (index < 0 || index >= width || seen[index]) {
                throw new MapperViolation(label + " contains a malformed permutation");
            }
            seen[index] = true;
        }
    }

    private void applyAttackers(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted()) {
            throw fail("attacker declaration was not accepted", null);
        }
        Set<UUID> selected = new LinkedHashSet<>(
                uuidValues(observed.getSelected(), "selected attacker"));
        List<Permanent> eligible = new ArrayList<>();
        for (Object raw : observed.getCandidates()) {
            if (!(raw instanceof Permanent)) {
                throw fail("attacker candidate is not a permanent", null);
            }
            Permanent permanent = (Permanent) raw;
            // The observer snapshots all controlled creatures. After CP7
            // commits, selected attackers are normally tapped; retain them and
            // recompute eligibility only for the unselected remainder.
            if (selected.contains(permanent.getId())
                    || permanent.canAttack(null, observed.getGame())) {
                eligible.add(permanent);
            }
        }
        sortPermanents(eligible, observed.getGame());
        if (eligible.isEmpty()) {
            if (!selected.isEmpty()) {
                throw fail("selected attackers are absent from inferred eligible set", null);
            }
            recordForcedNoPolicy("zero_candidate_attackers");
            return;
        }

        long physicalDecisionId = -1L;
        Set<UUID> used = new HashSet<>();
        for (int substep = 0; substep < eligible.size(); substep++) {
            XMageRallyBridgeProtocol.DecisionBody current =
                    requireCurrentCp7(observed.getGame(), "attacker inclusion");
            requireBinaryShape(current, "attacker_inclusion", substep, eligible.size(),
                    physicalDecisionId);
            if (substep == 0) {
                physicalDecisionId = current.getPhysicalDecisionId();
            }
            BinaryRefs refs = binaryRefs(current, "choose_attacker_inclusion", false);
            Permanent attacker = resolveCombatPermanent(
                    refs.primary, eligible, used, observed.getGame(),
                    "attacker substep " + substep);
            if (!used.add(attacker.getId())) {
                throw fail("attacker aggregate repeats an XMage identity", null);
            }
            int selectedIndex = selected.contains(attacker.getId()) ? 1 : 0;
            step(current, selectedIndex, "attacker_inclusion", observed.getGame());
        }
        if (used.size() != eligible.size()) {
            throw fail("attacker mapping did not consume the complete XMage menu", null);
        }
    }

    private void applyBlockers(RallyCp7DecisionObserver.Decision observed) {
        if (!observed.isAccepted()) {
            throw fail("blocker declaration was not accepted", null);
        }
        Game game = observed.getGame();
        List<Permanent> available = new ArrayList<>();
        for (Object raw : observed.getCandidates()) {
            if (!(raw instanceof Permanent)) {
                throw fail("blocker candidate is not a permanent", null);
            }
            available.add((Permanent) raw);
        }
        sortPermanents(available, game);
        Set<BlockAssignment> selected = blockAssignments(observed.getSelected());

        List<Permanent> attackers = new ArrayList<>();
        for (UUID attackerId : game.getCombat().getAttackers()) {
            Permanent attacker = game.getPermanent(attackerId);
            if (attacker != null) {
                attackers.add(attacker);
            }
        }
        sortPermanents(attackers, game);

        Set<BlockAssignment> applied = new HashSet<>();
        while (hasLegalBlockerPair(attackers, available, game)) {
            XMageRallyBridgeProtocol.DecisionBody first =
                    requireCurrentCp7(game, "blocker inclusion");
            requireCommon(first, "blocker inclusion");
            if (!"blocker_inclusion".equals(first.getDecisionKind())
                    || first.getSubstepIndex() != 0
                    || first.getLegalActionCount() != 2) {
                throw fail("blocker group does not start at a binary substep zero", null);
            }
            BinaryRefs firstRefs = binaryRefs(
                    first, "choose_blocker_inclusion", true);
            Permanent attacker = resolveCombatPermanent(
                    firstRefs.primary,
                    attackersWithLegalBlockers(attackers, available, game),
                    Collections.emptySet(), game, "blocker group attacker");
            List<Permanent> legal = legalBlockersFor(attacker, available, game);
            sortPermanents(legal, game);

            long physicalDecisionId = -1L;
            Set<UUID> usedBlockers = new HashSet<>();
            List<Permanent> selectedForAttacker = new ArrayList<>();
            for (int substep = 0; substep < legal.size(); substep++) {
                XMageRallyBridgeProtocol.DecisionBody current = substep == 0
                        ? first : requireCurrentCp7(game, "blocker inclusion");
                requireBinaryShape(current, "blocker_inclusion", substep, legal.size(),
                        physicalDecisionId);
                if (substep == 0) {
                    physicalDecisionId = current.getPhysicalDecisionId();
                }
                BinaryRefs refs = binaryRefs(current, "choose_blocker_inclusion", true);
                Permanent boundAttacker = resolveCombatPermanent(
                        refs.primary, Collections.singletonList(attacker),
                        Collections.emptySet(), game,
                        "blocker attacker substep " + substep);
                if (!boundAttacker.getId().equals(attacker.getId())) {
                    throw fail("blocker aggregate changed its fixed attacker", null);
                }
                Permanent blocker = resolveCombatPermanent(
                        refs.secondary, legal, usedBlockers, game,
                        "blocker substep " + substep);
                if (!usedBlockers.add(blocker.getId())) {
                    throw fail("blocker aggregate repeats an XMage identity", null);
                }
                BlockAssignment pair = new BlockAssignment(blocker.getId(), attacker.getId());
                boolean include = selected.contains(pair);
                if (include) {
                    applied.add(pair);
                    selectedForAttacker.add(blocker);
                }
                step(current, include ? 1 : 0, "blocker_inclusion", game);
            }
            if (usedBlockers.size() != legal.size()) {
                throw fail("blocker mapping did not consume the complete XMage menu", null);
            }
            available.removeAll(selectedForAttacker);
            attackers.remove(attacker);
        }
        if (!applied.equals(selected)) {
            throw fail("XMage blocker assignments were not represented exactly in Rust: selected="
                    + selected.size() + " applied=" + applied.size(), null);
        }
    }

    private static boolean hasLegalBlockerPair(
            List<Permanent> attackers, List<Permanent> blockers, Game game) {
        for (Permanent attacker : attackers) {
            for (Permanent blocker : blockers) {
                if (blocker.canBlock(attacker.getId(), game)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static List<Permanent> attackersWithLegalBlockers(
            List<Permanent> attackers, List<Permanent> blockers, Game game) {
        List<Permanent> result = new ArrayList<>();
        for (Permanent attacker : attackers) {
            for (Permanent blocker : blockers) {
                if (blocker.canBlock(attacker.getId(), game)) {
                    result.add(attacker);
                    break;
                }
            }
        }
        return result;
    }

    private static List<Permanent> legalBlockersFor(
            Permanent attacker, List<Permanent> blockers, Game game) {
        List<Permanent> result = new ArrayList<>();
        for (Permanent blocker : blockers) {
            if (blocker.canBlock(attacker.getId(), game)) {
                result.add(blocker);
            }
        }
        return result;
    }

    private Permanent resolveCombatPermanent(
            JsonObject stable,
            List<? extends Permanent> candidates,
            Set<UUID> used,
            Game game,
            String label) {
        CombatStableRef reference = combatStableRef(stable, label);
        List<UUID> candidateIds = new ArrayList<>(candidates.size());
        for (Permanent candidate : candidates) {
            if (candidate == null) {
                throw fail(label + " contains a null XMage candidate", null);
            }
            candidateIds.add(candidate.getId());
        }

        UUID bound = mapBoundCombatUuid(
                reference.binding.arenaId, candidateIds, arenaBindings, used, label);
        if (bound != null) {
            Permanent permanent = game.getPermanent(bound);
            validateCombatPermanent(permanent, reference, game, label);
            return permanent;
        }
        if (reference.binding.arenaId < INITIAL_OBJECT_COUNT) {
            throw fail(label + " has no XMage binding for opening arena id "
                    + reference.binding.arenaId, null);
        }

        List<Permanent> compatible = new ArrayList<>();
        for (Permanent candidate : candidates) {
            UUID id = candidate.getId();
            Integer cardId = RALLY_CARD_IDS.get(candidate.getName());
            if (used.contains(id) || uuidBindings.containsKey(id)
                    || cardId == null || cardId != reference.binding.cardDbId
                    || !reference.controller.equals(seatFor(candidate.getControllerId()))
                    || !reference.owner.equals(seatFor(candidate.getOwnerId()))) {
                continue;
            }
            compatible.add(candidate);
        }
        sortPermanents(compatible, game);
        if (compatible.isEmpty()) {
            throw fail(label + " cannot bind generated arena id "
                    + reference.binding.arenaId + " card_db_id="
                    + reference.binding.cardDbId, null);
        }

        Permanent resolved = compatible.get(0);
        validateCombatPermanent(resolved, reference, game, label);
        putBinding(resolved.getId(), reference.binding, label);
        return resolved;
    }

    private static UUID mapBoundCombatUuid(
            int arenaId,
            List<UUID> candidateIds,
            Map<Integer, UUID> bindings,
            Set<UUID> used,
            String label) {
        UUID bound = bindings.get(arenaId);
        if (bound == null) {
            return null;
        }
        int matches = 0;
        for (UUID candidateId : candidateIds) {
            if (bound.equals(candidateId)) {
                matches++;
            }
        }
        if (matches != 1) {
            throw new MapperViolation(label + " bound arena id " + arenaId
                    + " maps to " + matches + " XMage candidates");
        }
        if (used.contains(bound)) {
            throw new MapperViolation(
                    label + " repeats an already consumed XMage candidate");
        }
        return bound;
    }

    private void validateCombatPermanent(
            Permanent permanent, CombatStableRef reference, Game game, String label) {
        if (permanent == null || game.getPermanent(permanent.getId()) == null) {
            throw fail(label + " resolved a stale permanent", null);
        }
        Integer cardId = RALLY_CARD_IDS.get(permanent.getName());
        if (cardId == null || cardId != reference.binding.cardDbId) {
            throw fail(label + " card id does not match Rust", null);
        }
        if (!reference.controller.equals(seatFor(permanent.getControllerId()))) {
            throw fail(label + " controller does not match Rust", null);
        }
        if (!reference.owner.equals(seatFor(permanent.getOwnerId()))) {
            throw fail(label + " owner does not match Rust", null);
        }
        StableBinding actual = uuidBindings.get(permanent.getId());
        if (actual != null && !actual.equals(reference.binding)) {
            throw fail(label + " changed the Rust identity of an XMage object", null);
        }
    }

    private CombatStableRef combatStableRef(JsonObject stable, String label) {
        if (stable == null) {
            throw new MapperViolation(label + " lacks a stable reference");
        }
        StableBinding binding = stableBinding(stable, label);
        String owner = requiredString(stable, "owner", label);
        String controller = requiredString(stable, "controller", label);
        if (!("p0".equals(owner) || "p1".equals(owner))
                || !("p0".equals(controller) || "p1".equals(controller))) {
            throw new MapperViolation(label + " has an invalid owner or controller seat");
        }
        if (!"Battlefield".equals(requiredString(stable, "zone", label))) {
            throw new MapperViolation(label + " is not a battlefield reference");
        }
        if (requiredInt(stable, "zone_change_count", label) < 0) {
            throw new MapperViolation(label + " has a negative zone change count");
        }
        return new CombatStableRef(binding, owner, controller);
    }

    private void bindOpeningIdentity(Game game) {
        if (openingIdentityBound) {
            requirePhysicalPlayer(game);
            return;
        }
        Player physical = requirePhysicalPlayer(game);
        Player opponent = null;
        for (Player player : game.getPlayers().values()) {
            if (player == null || player.getId().equals(physical.getId())) {
                continue;
            }
            if (opponent != null) {
                throw fail("CP7 mapper requires exactly two players", null);
            }
            opponent = player;
        }
        if (opponent == null) {
            throw fail("CP7 mapper cannot find the opponent", null);
        }
        p0PlayerId = physicalSeat == XMageRallyBridgeProtocol.Seat.P0
                ? physical.getId() : opponent.getId();
        p1PlayerId = physicalSeat == XMageRallyBridgeProtocol.Seat.P1
                ? physical.getId() : opponent.getId();

        List<List<Integer>> rustLibraries = bridge.getInitialLibraryCardDefinitionIds();
        if (rustLibraries == null || rustLibraries.size() != 2
                || rustLibraries.get(0).size() != RALLY_DECK_SIZE
                || rustLibraries.get(1).size() != RALLY_DECK_SIZE) {
            throw fail("bridge lacks two exact Rally opening libraries", null);
        }
        if (initialArenaIds.isEmpty()) {
            bindOpeningSeat(game, game.getPlayer(p0PlayerId), rustLibraries.get(0), 0, "p0");
            bindOpeningSeat(game, game.getPlayer(p1PlayerId), rustLibraries.get(1),
                    RALLY_DECK_SIZE, "p1");
        } else {
            bindExactOpeningIdentity(game, rustLibraries);
        }
        openingIdentityBound = true;
    }

    private void bindExactOpeningIdentity(
            Game game,
            List<List<Integer>> rustLibraries) {
        for (Map.Entry<UUID, Integer> entry : initialArenaIds.entrySet()) {
            UUID id = entry.getKey();
            int arenaId = entry.getValue();
            MageObject object = game.getObject(id);
            if (!(object instanceof Card)) {
                throw fail("exact opening object is not a card at arena " + arenaId, null);
            }
            Card card = (Card) object;
            int seatIndex = arenaId < RALLY_DECK_SIZE ? 0 : 1;
            int libraryIndex = arenaId % RALLY_DECK_SIZE;
            int rustCardId = rustLibraries.get(seatIndex).get(libraryIndex);
            Integer expectedCardId = RALLY_CARD_IDS.get(card.getName());
            if (expectedCardId == null || expectedCardId != rustCardId) {
                throw fail("exact opening card identity mismatch at arena " + arenaId
                        + ": XMage=" + card.getName() + " RustId=" + rustCardId, null);
            }
            UUID expectedOwner = seatIndex == 0 ? p0PlayerId : p1PlayerId;
            if (!expectedOwner.equals(card.getOwnerId())) {
                throw fail("exact opening card owner mismatch at arena " + arenaId, null);
            }
            putBinding(id, new StableBinding(arenaId, rustCardId),
                    "exact opening arena " + arenaId);
        }
        requireKeepSevenShape(game.getPlayer(p0PlayerId), game, "p0");
        requireKeepSevenShape(game.getPlayer(p1PlayerId), game, "p1");
    }

    private void requireKeepSevenShape(Player player, Game game, String seat) {
        if (player == null || player.getHand().size() != OPENING_HAND_SIZE
                || player.getLibrary().size() != RALLY_DECK_SIZE - OPENING_HAND_SIZE) {
            throw fail("exact opening " + seat + " is not fixed keep-seven shape", null);
        }
        for (Card card : player.getHand().getCards(game)) {
            if (!initialArenaIds.containsKey(card.getId())) {
                throw fail("exact opening " + seat + " contains an unknown hand card", null);
            }
        }
    }

    private void bindOpeningSeat(
            Game game,
            Player player,
            List<Integer> rustLibrary,
            int arenaOffset,
            String seat) {
        if (player == null) {
            throw fail("opening " + seat + " player is missing", null);
        }
        List<Card> hand = new ArrayList<>(player.getHand().getCards(game));
        List<UUID> library = player.getLibrary().getCardList();
        if (hand.size() != OPENING_HAND_SIZE
                || hand.size() + library.size() != RALLY_DECK_SIZE) {
            throw fail("opening " + seat + " is not fixed keep-seven Rally shape: hand="
                    + hand.size() + " library=" + library.size(), null);
        }
        List<UUID> ordered = new ArrayList<>(RALLY_DECK_SIZE);
        for (Card card : hand) {
            ordered.add(card.getId());
        }
        ordered.addAll(library);
        for (int index = 0; index < ordered.size(); index++) {
            UUID id = ordered.get(index);
            MageObject object = game.getObject(id);
            if (object == null) {
                throw fail("opening " + seat + " object is missing at " + index, null);
            }
            Integer expectedCardId = RALLY_CARD_IDS.get(object.getName());
            int rustCardId = rustLibrary.get(index);
            if (expectedCardId == null || expectedCardId != rustCardId) {
                throw fail("opening " + seat + " card identity mismatch at " + index
                        + ": XMage=" + object.getName() + " RustId=" + rustCardId, null);
            }
            putBinding(id, new StableBinding(arenaOffset + index, rustCardId),
                    "opening " + seat + " index " + index);
        }
    }

    private Player requirePhysicalPlayer(Game game) {
        Player byId = physicalPlayerId == null ? null : game.getPlayer(physicalPlayerId);
        if (byId != null) {
            if (!expectedPlayerName.equals(byId.getName())) {
                throw fail("bound CP7 player name changed", null);
            }
            return byId;
        }
        Player found = null;
        for (Player player : game.getPlayers().values()) {
            if (player != null && expectedPlayerName.equals(player.getName())) {
                if (found != null) {
                    throw fail("CP7 player name is not unique", null);
                }
                found = player;
            }
        }
        if (found == null) {
            throw fail("expected CP7 player is absent: " + expectedPlayerName, null);
        }
        physicalPlayerId = found.getId();
        return found;
    }

    private void prebindPriorityTokenSources(
            XMageRallyBridgeProtocol.DecisionBody current,
            Game game) {
        Map<Integer, Map<Integer, JsonObject>> refsByCard = new LinkedHashMap<>();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : current.getActionSemantics()) {
            if (!"activate_ability".equals(semantic.getActionKind())) {
                continue;
            }
            JsonObject source = requiredObject(parseSemantic(semantic), "source",
                    "activate_ability");
            int cardId = requiredInt(source, "card_db_id", "priority source");
            int arenaId = requiredInt(source, "arena_id", "priority source");
            if (arenaId < INITIAL_OBJECT_COUNT || arenaBindings.containsKey(arenaId)) {
                continue;
            }
            refsByCard.computeIfAbsent(cardId, ignored -> new LinkedHashMap<>())
                    .put(arenaId, source);
        }
        for (Map.Entry<Integer, Map<Integer, JsonObject>> entry : refsByCard.entrySet()) {
            int cardId = entry.getKey();
            List<Permanent> matching = new ArrayList<>();
            for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
                if (permanent == null || uuidBindings.containsKey(permanent.getId())
                        || !seatFor(permanent.getControllerId()).equals(physicalSeat.wire())) {
                    continue;
                }
                Integer expected = RALLY_CARD_IDS.get(permanent.getName());
                if (expected == null || expected != cardId) {
                    continue;
                }
                if (cardId == 132 && permanent.isTapped()) {
                    continue;
                }
                matching.add(permanent);
            }
            List<JsonObject> refs = new ArrayList<>(entry.getValue().values());
            refs.sort(Comparator.comparingInt(ref -> requiredInt(
                    ref, "arena_id", "priority token source")));
            sortPermanents(matching, game);
            if (matching.size() == refs.size()) {
                for (int i = 0; i < matching.size(); i++) {
                    bindExpected(matching.get(i).getId(), refs.get(i), game,
                            "priority token source rank " + i);
                }
            }
        }
    }

    private void prebindCommonSource(
            XMageRallyBridgeProtocol.DecisionBody current,
            Ability observedSource,
            Game game) {
        if (observedSource == null || observedSource.getSourceId() == null) {
            return;
        }
        Map<Integer, JsonObject> distinct = new LinkedHashMap<>();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : current.getActionSemantics()) {
            JsonObject row = parseSemantic(semantic);
            if (!row.has("source")) {
                continue;
            }
            JsonObject source = requiredObject(row, "source", semantic.getActionKind());
            distinct.put(requiredInt(source, "arena_id", "common source"), source);
        }
        if (distinct.size() == 1) {
            bindExpected(observedSource.getSourceId(), distinct.values().iterator().next(), game,
                    "common decision source");
        }
    }

    private void prebindOrderedTargetCandidates(
            XMageRallyBridgeProtocol.DecisionBody current,
            Collection<UUID> candidates,
            Set<String> allowedKinds,
            Game game) {
        Map<String, Map<Integer, JsonObject>> rustByIdentity = new LinkedHashMap<>();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : current.getActionSemantics()) {
            if (!allowedKinds.contains(semantic.getActionKind())) {
                continue;
            }
            JsonObject ref = actionObjectRef(parseSemantic(semantic), semantic.getActionKind());
            if (ref == null || "player".equals(optionalString(ref, "target_kind"))) {
                continue;
            }
            JsonObject stable = "object".equals(optionalString(ref, "target_kind"))
                    ? requiredObject(ref, "object", "target object") : ref;
            int arenaId = requiredInt(stable, "arena_id", "target token object");
            if (arenaId < INITIAL_OBJECT_COUNT || arenaBindings.containsKey(arenaId)) {
                continue;
            }
            int cardId = requiredInt(stable, "card_db_id", "target token object");
            String controller = requiredString(
                    stable, "controller", "target token object");
            String identity = controller + "|" + cardId;
            JsonObject previous = rustByIdentity
                    .computeIfAbsent(identity, ignored -> new LinkedHashMap<>())
                    .put(arenaId, stable);
            if (previous != null) {
                throw fail("target token arena id is repeated", null);
            }
        }

        for (Map.Entry<String, Map<Integer, JsonObject>> group
                : rustByIdentity.entrySet()) {
            List<JsonObject> rustTargets = new ArrayList<>(group.getValue().values());
            rustTargets.sort(Comparator.comparingInt(target -> requiredInt(
                    target, "arena_id", "target token object")));
            List<Permanent> xmageTargets = new ArrayList<>();
            Set<UUID> seen = new HashSet<>();
            for (UUID id : candidates) {
                if (id == null || !seen.add(id) || uuidBindings.containsKey(id)) {
                    continue;
                }
                Permanent permanent = game.getPermanent(id);
                Integer cardId = permanent == null
                        ? null : RALLY_CARD_IDS.get(permanent.getName());
                if (permanent != null && cardId != null
                        && group.getKey().equals(
                        seatFor(permanent.getControllerId()) + "|" + cardId)) {
                    xmageTargets.add(permanent);
                }
            }
            sortPermanents(xmageTargets, game);
            if (xmageTargets.size() != rustTargets.size()) {
                throw fail("target token identity count differs between XMage and Rust", null);
            }
            for (int i = 0; i < rustTargets.size(); i++) {
                bindExpected(xmageTargets.get(i).getId(), rustTargets.get(i), game,
                        "target token object rank " + i);
            }
        }
    }

    private void requireTargetMenuBijection(
            XMageRallyBridgeProtocol.DecisionBody current,
            Collection<UUID> candidates,
            Set<String> allowedKinds,
            Ability observedSource,
            Game game) {
        Set<UUID> xmage = new LinkedHashSet<>();
        for (UUID candidate : candidates) {
            if (candidate == null || !xmage.add(candidate)) {
                throw fail("XMage target menu contains a null or duplicate candidate", null);
            }
        }
        Set<UUID> used = new HashSet<>();
        boolean usedStop = false;
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : current.getActionSemantics()) {
            String kind = semantic.getActionKind();
            JsonObject row = parseSemantic(semantic);
            if (observedSource != null && row.has("source")
                    && !matchesBoundUuid(observedSource.getSourceId(),
                    requiredObject(row, "source", kind), game)) {
                throw fail("Rust target menu source does not match XMage", null);
            }
            if (FINISH_TARGET_KINDS.contains(kind)) {
                if (usedStop) {
                    throw fail("Rust target menu repeats STOP", null);
                }
                usedStop = true;
                continue;
            }
            if (!allowedKinds.contains(kind)) {
                throw fail("Rust target menu contains unsupported action kind " + kind, null);
            }
            JsonObject ref = actionObjectRef(row, kind);
            UUID unique = null;
            int matches = 0;
            for (UUID candidate : xmage) {
                if (targetRefMatches(candidate, ref, game)) {
                    unique = candidate;
                    matches++;
                }
            }
            if (matches != 1 || used.contains(unique)) {
                throw fail("Rust target row did not map to exactly one unused XMage target: "
                        + "kind=" + kind + " matches=" + matches
                        + " json=" + semantic.getCanonicalJson(), null);
            }
            used.add(unique);
        }
        int mappedWidth = used.size() + (usedStop ? 1 : 0);
        int xmageWidth = xmage.size() + (usedStop ? 1 : 0);
        if (mappedWidth != xmageWidth
                || current.getActionSemantics().size() != xmageWidth) {
            throw fail("Rust and XMage target menus are not a complete bijection: "
                    + "rust=" + current.getActionSemantics().size()
                    + " xmage=" + xmageWidth + " mapped=" + mappedWidth, null);
        }
    }

    private int selectedTargetIndex(
            XMageRallyBridgeProtocol.DecisionBody current,
            UUID selected,
            Set<String> allowedKinds,
            Ability observedSource,
            Game game) {
        List<Integer> matches = new ArrayList<>();
        List<String> rowDetails = new ArrayList<>();
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            XMageRallyBridgeProtocol.ActionSemantic semantic =
                    current.getActionSemantics().get(i);
            if (!allowedKinds.contains(semantic.getActionKind())) {
                rowDetails.add(i + ":kind=" + semantic.getActionKind()
                        + "/allowed=false");
                continue;
            }
            JsonObject row = parseSemantic(semantic);
            boolean sourceMatches = observedSource == null || !row.has("source")
                    || matchesBoundUuid(observedSource.getSourceId(),
                    requiredObject(row, "source", semantic.getActionKind()), game);
            if (!sourceMatches) {
                rowDetails.add(i + ":kind=" + semantic.getActionKind()
                        + "/source=false/json=" + semantic.getCanonicalJson());
                continue;
            }
            JsonObject ref = actionObjectRef(row, semantic.getActionKind());
            boolean targetMatches = ref != null && targetRefMatches(selected, ref, game);
            rowDetails.add(i + ":kind=" + semantic.getActionKind()
                    + "/source=true/target=" + targetMatches
                    + "/json=" + semantic.getCanonicalJson());
            if (targetMatches) {
                matches.add(i);
            }
        }
        if (matches.size() != 1) {
            MageObject selectedObject = game.getObject(selected);
            StableBinding selectedBinding = uuidBindings.get(selected);
            UUID sourceId = observedSource == null ? null : observedSource.getSourceId();
            MageObject sourceObject = sourceId == null ? null : game.getObject(sourceId);
            StableBinding sourceBinding = sourceId == null ? null : uuidBindings.get(sourceId);
            throw fail("selected target did not identify exactly one Rust action; matches="
                    + matches.size()
                    + " step=" + current.getStep()
                    + " decision_kind=" + current.getDecisionKind()
                    + " selected=" + selected
                    + "/" + (selectedObject == null ? "missing" : selectedObject.getName())
                    + "/binding=" + bindingSummary(selectedBinding)
                    + "/zone=" + game.getState().getZone(selected)
                    + " source=" + sourceId
                    + "/" + (sourceObject == null ? "missing" : sourceObject.getName())
                    + "/binding=" + bindingSummary(sourceBinding)
                    + " rows=" + rowDetails, null);
        }
        return matches.get(0);
    }

    private static String bindingSummary(StableBinding binding) {
        return binding == null ? "none" : binding.arenaId + "/" + binding.cardDbId;
    }

    private boolean targetRefMatches(UUID id, JsonObject ref, Game game) {
        String targetKind = optionalString(ref, "target_kind");
        if ("player".equals(targetKind)) {
            return seatFor(id).equals(requiredString(ref, "player", "target player"));
        }
        JsonObject stable = "object".equals(targetKind)
                ? requiredObject(ref, "object", "target object") : ref;
        return matchesBoundUuid(id, stable, game);
    }

    private JsonObject actionObjectRef(JsonObject row, String kind) {
        if ("choose_target".equals(kind) || "choose_effect_target".equals(kind)) {
            return requiredObject(row, "target", kind);
        }
        if ("choose_cost_target".equals(kind)) {
            return requiredObject(row, "candidate", kind);
        }
        if ("discard".equals(kind)) {
            JsonElement cards = row.get("cards");
            if (cards == null || !cards.isJsonArray() || cards.getAsJsonArray().size() != 1
                    || !cards.getAsJsonArray().get(0).isJsonObject()) {
                throw fail("discard semantic is not a singleton", null);
            }
            return cards.getAsJsonArray().get(0).getAsJsonObject();
        }
        return null;
    }

    private BinaryRefs binaryRefs(
            XMageRallyBridgeProtocol.DecisionBody current,
            String expectedKind,
            boolean blocker) {
        if (current.getActionSemantics().size() != 2) {
            throw fail(expectedKind + " is not binary", null);
        }
        JsonObject no = parseSemantic(current.getActionSemantics().get(0));
        JsonObject yes = parseSemantic(current.getActionSemantics().get(1));
        if (!expectedKind.equals(current.getActionSemantics().get(0).getActionKind())
                || !expectedKind.equals(current.getActionSemantics().get(1).getActionKind())
                || requiredBoolean(no, "include", expectedKind)
                || !requiredBoolean(yes, "include", expectedKind)) {
            throw fail(expectedKind + " rows are not ordered false then true", null);
        }
        JsonObject noAttacker = requiredObject(no, "attacker", expectedKind);
        JsonObject yesAttacker = requiredObject(yes, "attacker", expectedKind);
        requireSameStable(noAttacker, yesAttacker, expectedKind + " attacker");
        if (!blocker) {
            return new BinaryRefs(noAttacker, null);
        }
        JsonObject noBlocker = requiredObject(no, "blocker", expectedKind);
        JsonObject yesBlocker = requiredObject(yes, "blocker", expectedKind);
        requireSameStable(noBlocker, yesBlocker, expectedKind + " blocker");
        return new BinaryRefs(noAttacker, noBlocker);
    }

    private void requireBinaryShape(
            XMageRallyBridgeProtocol.DecisionBody current,
            String decisionKind,
            int substep,
            int substepCount,
            long physicalDecisionId) {
        requireCommon(current, decisionKind);
        if (!decisionKind.equals(current.getDecisionKind())
                || current.getLegalActionCount() != 2
                || current.getSubstepIndex() != substep
                || current.getSubstepCount() != substepCount
                || substep > 0 && current.getPhysicalDecisionId() != physicalDecisionId) {
            throw fail(decisionKind + " aggregate shape mismatch at substep " + substep, null);
        }
    }

    private void requireSurface(
            XMageRallyBridgeProtocol.DecisionBody current,
            String label) {
        requireCommon(current, label);
        if (!"surface".equals(current.getDecisionKind())
                || current.getSubstepIndex() != 0
                || current.getSubstepCount() != 1) {
            List<String> actionKinds = new ArrayList<>();
            for (XMageRallyBridgeProtocol.ActionSemantic semantic
                    : current.getActionSemantics()) {
                actionKinds.add(semantic.getActionKind());
            }
            throw fail(label + " expected one Rust surface decision"
                    + "; step=" + current.getStep()
                    + " decision_kind=" + current.getDecisionKind()
                    + " substep=" + current.getSubstepIndex()
                    + "/" + current.getSubstepCount()
                    + " physical_decision_id=" + current.getPhysicalDecisionId()
                    + " action_kinds=" + actionKinds, null);
        }
    }

    private void requireCommon(
            XMageRallyBridgeProtocol.DecisionBody current,
            String label) {
        if (current == null || current.getEpisodeId() != episodeId
                || current.getActingPlayer() != physicalSeat
                || current.getCandidateSeat() == physicalSeat
                || current.isCandidateControlsCurrentActor()
                || current.getSelectedActionIndex() != null
                || current.getCandidateActionSeedU64Hex() != null
                || current.getLegalActionCount() <= 0
                || current.getActionSemantics() == null
                || current.getActionSemantics().size() != current.getLegalActionCount()
                || current.getLogitsF32Bits() == null
                || current.getLogitsF32Bits().size() != current.getLegalActionCount()) {
            throw fail(label + " has an invalid Rust CP7 decision binding", null);
        }
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : current.getActionSemantics()) {
            JsonObject row = parseSemantic(semantic);
            if (!physicalSeat.wire().equals(requiredString(row, "actor", label))) {
                throw fail(label + " semantic actor does not match the CP7 seat", null);
            }
        }
    }

    private XMageRallyBridgeProtocol.DecisionBody requireCurrentCp7(
            Game game, String label) {
        XMageRallyBridgeProtocol.DecisionBody current = currentForCp7OrNull();
        if (current == null) {
            throw fail("Rust has no CP7 decision for " + label, null);
        }
        requireClockMatch(current, game, label);
        return current;
    }

    private XMageRallyBridgeProtocol.DecisionBody currentForCp7OrNull() {
        XMageRallyBridgeProtocol.DecisionBody current = bridge.getCurrentDecision();
        if (current == null) {
            if (bridge.getTerminal() != null) {
                throw fail("Rust reached terminal before XMage CP7", null);
            }
            throw fail("bridge has neither decision nor terminal", null);
        }
        if (current.getEpisodeId() != episodeId) {
            throw fail("Rust current episode changed", null);
        }
        // Cursor lookup is observational. Same-seat future decisions can be
        // rejected by menu shape without consuming them. Required callbacks
        // admit in requireCurrentCp7; optional callbacks admit in step().
        return current.getActingPlayer() == physicalSeat ? current : null;
    }

    private int uniqueKindIndex(
            XMageRallyBridgeProtocol.DecisionBody current,
            Set<String> kinds) {
        int selected = -1;
        for (int i = 0; i < current.getActionSemantics().size(); i++) {
            if (!kinds.contains(current.getActionSemantics().get(i).getActionKind())) {
                continue;
            }
            if (selected >= 0) {
                throw fail("Rust decision has multiple actions in " + kinds, null);
            }
            selected = i;
        }
        if (selected < 0) {
            throw fail("Rust decision has no action in " + kinds, null);
        }
        return selected;
    }

    private void step(
            XMageRallyBridgeProtocol.DecisionBody current,
            int selectedIndex,
            String kind,
            Game game) {
        if (selectedIndex < 0 || selectedIndex >= current.getLegalActionCount()) {
            throw fail("selected Rust action is outside the legal width", null);
        }
        XMageRallyBridgeProtocol.ExpectedClock expectedClock =
                requireClockMatch(current, game, kind + ":step");
        String requestId = "xmage-cp7-shadow-" + episodeId + "-" + requestOrdinal;
        if (requestOrdinal == Long.MAX_VALUE || appliedPolicySteps == Long.MAX_VALUE) {
            throw fail("CP7 mapper counter exhausted", null);
        }
        requestOrdinal++;
        XMageRallyBridgeProtocol.ActionSemantic selectedSemantic =
                current.getActionSemantics().get(selectedIndex);
        String selectedActionKind = selectedSemantic.getActionKind();
        Integer selectedSourceArenaId = null;
        if ("activate_ability".equals(selectedActionKind)) {
            JsonObject selectedRow = parseSemantic(selectedSemantic);
            JsonObject source = requiredObject(
                    selectedRow, "source", selectedActionKind);
            selectedSourceArenaId = requiredInt(
                    source, "arena_id", selectedActionKind + " source");
        }
        try {
            bridge.step(requestId, episodeId, current.getStep(), selectedIndex,
                    expectedClock);
        } catch (XMageRallyBridgeProcessClient.BridgeFailure error) {
            throw fail(bridgeStepFailureMessage(
                    episodeId, current.getStep(), error.getMessage()), error);
        }
        appliedPhysicalDecisionIds.add(current.getPhysicalDecisionId());
        appliedPolicySteps++;
        increment(appliedKinds, kind);
        lastAppliedActionKind = selectedActionKind;
        lastAppliedSourceArenaId = selectedSourceArenaId;
    }

    private static String bridgeStepFailureMessage(
            long episodeId, long step, String bridgeMessage) {
        return "CP7_KERNEL_SHADOW_MAPPER_BRIDGE_STEP_FAILURE"
                + " episode=" + episodeId
                + " step=" + step
                + " " + (bridgeMessage == null ? "bridge_error=unknown" : bridgeMessage);
    }

    private XMageRallyBridgeProtocol.ExpectedClock requireClockMatch(
            XMageRallyBridgeProtocol.DecisionBody decision,
            Game game,
            String label) {
        try {
            // Policy step 0 is still the reset decision, so this is the first
            // and only point where the reset clock meets the live game.
            return decision != null && decision.getStep() == 0L
                    ? XMageRallyClockComparator.requireResetBinding(
                    decision, game, "cp7_kernel_shadow_mapper", label)
                    : XMageRallyClockComparator.requireMatch(
                            decision, game, "cp7_kernel_shadow_mapper", label);
        } catch (XMageRallyClockComparator.ClockMismatch mismatch) {
            throw fail("CP7_KERNEL_SHADOW_MAPPER_CLOCK_MISMATCH "
                    + mismatch.getMessage(), mismatch);
        }
    }

    private boolean matchesBoundUuid(UUID id, JsonObject stable, Game game) {
        StableBinding expected = stableBinding(stable, "semantic stable reference");
        StableBinding actual = uuidBindings.get(id);
        if (actual != null) {
            return actual.equals(expected);
        }
        UUID boundUuid = arenaBindings.get(expected.arenaId);
        if (boundUuid != null) {
            return boundUuid.equals(id);
        }
        MageObject object = game.getObject(id);
        Integer expectedCardId = object == null ? null : RALLY_CARD_IDS.get(object.getName());
        if (expected.arenaId < INITIAL_OBJECT_COUNT || expectedCardId == null
                || expectedCardId != expected.cardDbId) {
            return false;
        }
        // Unknown tokens are bound only when this semantic is the sole
        // compatible object at the call site. Callers with an exact ordered
        // correspondence use bindExpected directly.
        int compatible = 0;
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            Integer cardId = permanent == null ? null : RALLY_CARD_IDS.get(permanent.getName());
            if (permanent != null && !uuidBindings.containsKey(permanent.getId())
                    && cardId != null && cardId == expected.cardDbId
                    && seatFor(permanent.getControllerId())
                    .equals(requiredString(stable, "controller", "token controller"))) {
                compatible++;
            }
        }
        if (compatible == 1) {
            bindExpected(id, stable, game, "unambiguous dynamic object");
            return true;
        }
        return false;
    }

    private void bindExpected(UUID id, JsonObject stable, Game game, String label) {
        StableBinding expected = stableBinding(stable, label);
        MageObject object = game.getObject(id);
        if (object == null) {
            throw fail(label + " XMage object is missing", null);
        }
        Integer cardId = RALLY_CARD_IDS.get(object.getName());
        if (cardId == null || cardId != expected.cardDbId) {
            throw fail(label + " card id mismatch: " + object.getName()
                    + " vs Rust " + expected.cardDbId, null);
        }
        if (expected.arenaId >= INITIAL_OBJECT_COUNT) {
            String controller = requiredString(stable, "controller", label);
            UUID controllerId = object instanceof Permanent
                    ? ((Permanent) object).getControllerId()
                    : object instanceof Card ? ((Card) object).getOwnerId() : null;
            if (controllerId != null && !seatFor(controllerId).equals(controller)) {
                throw fail(label + " controller mismatch", null);
            }
        }
        putBinding(id, expected, label);
    }

    private void putBinding(UUID id, StableBinding binding, String label) {
        StableBinding previous = uuidBindings.get(id);
        if (previous != null && !previous.equals(binding)) {
            throw fail(label + " changed the Rust identity of an XMage object", null);
        }
        UUID previousUuid = arenaBindings.get(binding.arenaId);
        if (previousUuid != null && !previousUuid.equals(id)) {
            throw fail(label + " reused Rust arena id " + binding.arenaId, null);
        }
        uuidBindings.put(id, binding);
        arenaBindings.put(binding.arenaId, id);
    }

    private StableBinding stableBinding(JsonObject stable, String label) {
        int arenaId = requiredInt(stable, "arena_id", label);
        int cardDbId = requiredInt(stable, "card_db_id", label);
        if (arenaId < 0 || cardDbId < 0) {
            throw fail(label + " contains a negative stable id", null);
        }
        return new StableBinding(arenaId, cardDbId);
    }

    private String canonicalUuidKey(UUID id, Game game) {
        if (id == null) {
            return "99|null";
        }
        String seat = seatFor(id);
        if (!"none".equals(seat)) {
            return "10|player|" + seat;
        }
        int battlefieldIndex = 0;
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent != null && id.equals(permanent.getId())) {
                return String.format(Locale.ROOT, "11|permanent|%s|%06d|%s",
                        seatFor(permanent.getControllerId()), battlefieldIndex,
                        safe(permanent.getName()));
            }
            battlefieldIndex++;
        }
        int stackIndex = 0;
        for (StackObject stackObject : game.getStack()) {
            if (id.equals(stackObject.getId()) || id.equals(stackObject.getSourceId())) {
                return String.format(Locale.ROOT, "13|stack|%06d|%s",
                        stackIndex, safe(stackObject.getName()));
            }
            stackIndex++;
        }
        for (Player player : playersInSeatOrder(game)) {
            String playerSeat = seatFor(player.getId());
            int index = 0;
            for (Card card : player.getHand().getCards(game)) {
                if (id.equals(card.getId())) {
                    return cardKey("hand", playerSeat, index, card);
                }
                index++;
            }
            index = 0;
            for (UUID cardId : player.getLibrary().getCardList()) {
                if (id.equals(cardId)) {
                    Card card = game.getCard(id);
                    return cardKey("library", playerSeat, index, card);
                }
                index++;
            }
            index = 0;
            for (Card card : player.getGraveyard().getCards(game)) {
                if (id.equals(card.getId())) {
                    return cardKey("graveyard", playerSeat, index, card);
                }
                index++;
            }
            index = 0;
            for (Card card : game.getExile().getCardsOwned(game, player.getId())) {
                if (id.equals(card.getId())) {
                    return cardKey("exile", playerSeat, index, card);
                }
                index++;
            }
        }
        throw fail("cannot canonicalize XMage UUID " + id, null);
    }

    private void sortPermanents(List<Permanent> permanents, Game game) {
        permanents.sort(Comparator.comparing(permanent ->
                canonicalUuidKey(permanent.getId(), game)));
    }

    private List<Player> playersInSeatOrder(Game game) {
        if (p0PlayerId == null || p1PlayerId == null) {
            throw fail("seat identities are not bound", null);
        }
        return Arrays.asList(game.getPlayer(p0PlayerId), game.getPlayer(p1PlayerId));
    }

    private String seatFor(UUID id) {
        if (id == null) {
            return "none";
        }
        if (id.equals(p0PlayerId)) {
            return "p0";
        }
        if (id.equals(p1PlayerId)) {
            return "p1";
        }
        return "none";
    }

    private static String cardKey(String zone, String seat, int index, Card card) {
        return String.format(Locale.ROOT, "12|card|%s|%s|%06d|%s|%s|%s|%s",
                seat, zone, index,
                card == null ? "" : safe(card.getName()),
                card == null ? "" : safe(card.getExpansionSetCode()),
                card == null ? "" : safe(card.getCardNumber()),
                card == null ? "" : card.getClass().getName());
    }

    private static String priorityKind(Ability ability) {
        if (ability instanceof PlayLandAbility) {
            return "play_land";
        }
        if (ability instanceof SpellAbility) {
            return "cast_spell";
        }
        if (ability instanceof ManaAbility) {
            return "activate_mana_ability";
        }
        return "activate_ability";
    }

    private static String booleanField(String kind) {
        if ("choose_kicker".equals(kind) || "choose_spell_copy_payment".equals(kind)) {
            return "pay";
        }
        if ("choose_effect_boolean".equals(kind)) {
            return "value";
        }
        if ("choose_optional_cost_use".equals(kind)) {
            return "use_cost";
        }
        if ("choose_madness_cast".equals(kind)) {
            return "cast_it";
        }
        if ("choose_spell_copy_retarget".equals(kind)) {
            return "change_target";
        }
        throw new MapperViolation("unsupported boolean semantic " + kind);
    }

    private Set<BlockAssignment> blockAssignments(Collection<?> values) {
        Set<BlockAssignment> result = new LinkedHashSet<>();
        for (Object value : values) {
            if (!(value instanceof String)) {
                throw fail("block assignment is not a string", null);
            }
            String text = (String) value;
            int split = text.indexOf("->");
            if (split <= 0 || split != text.lastIndexOf("->")
                    || split + 2 >= text.length()) {
                throw fail("malformed block assignment " + text, null);
            }
            try {
                BlockAssignment assignment = new BlockAssignment(
                        UUID.fromString(text.substring(0, split)),
                        UUID.fromString(text.substring(split + 2)));
                if (!result.add(assignment)) {
                    throw fail("duplicate block assignment " + text, null);
                }
            } catch (IllegalArgumentException error) {
                throw fail("malformed block UUID in " + text, error);
            }
        }
        return result;
    }

    private List<UUID> uuidValues(Collection<?> values, String label) {
        List<UUID> result = new ArrayList<>();
        for (Object value : values) {
            if (!(value instanceof UUID)) {
                throw fail(label + " is not a UUID", null);
            }
            result.add((UUID) value);
        }
        return result;
    }

    private List<UUID> objectIds(Collection<?> values, String label) {
        List<UUID> result = new ArrayList<>();
        for (Object value : values) {
            if (value instanceof UUID) {
                result.add((UUID) value);
            } else if (value instanceof MageObject) {
                result.add(((MageObject) value).getId());
            } else {
                throw fail(label + " has no XMage object id", null);
            }
        }
        return result;
    }

    private static JsonObject parseSemantic(XMageRallyBridgeProtocol.ActionSemantic semantic) {
        if (semantic == null) {
            throw new MapperViolation("Rust semantic is null");
        }
        JsonElement parsed;
        try {
            parsed = JsonParser.parseString(semantic.getCanonicalJson());
        } catch (RuntimeException error) {
            throw new MapperViolation("Rust semantic JSON is invalid", error);
        }
        if (!parsed.isJsonObject()) {
            throw new MapperViolation("Rust semantic is not an object");
        }
        JsonObject object = parsed.getAsJsonObject();
        if (!semantic.getActionKind().equals(
                requiredString(object, "action_kind", "semantic"))) {
            throw new MapperViolation("Rust semantic discriminator mismatch");
        }
        return object;
    }

    private static JsonObject requiredObject(JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonObject()) {
            throw new MapperViolation(label + " lacks object field " + field);
        }
        return value.getAsJsonObject();
    }

    private static List<JsonObject> requiredObjectArray(
            JsonObject object,
            String field,
            String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonArray()) {
            throw new MapperViolation(label + " lacks object array " + field);
        }
        List<JsonObject> result = new ArrayList<>();
        for (JsonElement element : value.getAsJsonArray()) {
            if (!element.isJsonObject()) {
                throw new MapperViolation(label + " has a non-object in " + field);
            }
            result.add(element.getAsJsonObject());
        }
        return result;
    }

    private static List<Integer> requiredIntArray(
            JsonObject object,
            String field,
            String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonArray()) {
            throw new MapperViolation(label + " lacks integer array " + field);
        }
        List<Integer> result = new ArrayList<>();
        for (JsonElement element : value.getAsJsonArray()) {
            if (!element.isJsonPrimitive()
                    || !element.getAsJsonPrimitive().isNumber()) {
                throw new MapperViolation(label + " has a non-integer in " + field);
            }
            try {
                result.add(element.getAsInt());
            } catch (RuntimeException error) {
                throw new MapperViolation(label + " has an invalid integer in " + field,
                        error);
            }
        }
        return result;
    }

    private static String requiredString(JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isString()) {
            throw new MapperViolation(label + " lacks string field " + field);
        }
        return value.getAsString();
    }

    private static String optionalString(JsonObject object, String field) {
        JsonElement value = object.get(field);
        return value != null && value.isJsonPrimitive()
                && value.getAsJsonPrimitive().isString() ? value.getAsString() : null;
    }

    private static int requiredInt(JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isNumber()) {
            throw new MapperViolation(label + " lacks integer field " + field);
        }
        try {
            return value.getAsInt();
        } catch (RuntimeException error) {
            throw new MapperViolation(label + " has invalid integer field " + field, error);
        }
    }

    private static boolean requiredBoolean(JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isBoolean()) {
            throw new MapperViolation(label + " lacks boolean field " + field);
        }
        return value.getAsBoolean();
    }

    private static void requireSameStable(JsonObject left, JsonObject right, String label) {
        if (!left.equals(right)) {
            throw new MapperViolation(label + " changes across binary rows");
        }
    }

    private void recordForcedNoPolicy(String kind) {
        if (forcedNoPolicyEvents == Long.MAX_VALUE) {
            throw fail("forced no-policy counter exhausted", null);
        }
        forcedNoPolicyEvents++;
        increment(appliedKinds, kind);
    }

    private void requireLive() {
        if (failed) {
            throw new MapperViolation("CP7 mapper is failed closed after: " + firstFailure);
        }
        if (!bridge.isUsable()) {
            throw fail("bridge is no longer usable", null);
        }
    }

    private MapperViolation fail(String message, Throwable cause) {
        if (!failed) {
            failed = true;
            firstFailure = message;
            bridge.close();
        }
        return cause == null ? new MapperViolation(message) : new MapperViolation(message, cause);
    }

    /**
     * CP7 reads these values during class initialization. The harness must set
     * them on the process, not mutate them after constructing the player.
     */
    public static void requireExternalDeterministicCp7Configuration() {
        requireEnabled("AI_DETERMINISTIC_TIEBREAKS");
        requireEnabled("AI_DETERMINISTIC_SEARCH");
        requireExactInt("AI_DETERMINISTIC_MAX_NODES", 5000);
        requireExactInt("AI_MAX_THREADS_FOR_SIMULATIONS", 1);
    }

    private static void requireEnabled(String name) {
        String value = configured(name);
        if (value == null || !("1".equals(value.trim())
                || "true".equalsIgnoreCase(value.trim())
                || "yes".equalsIgnoreCase(value.trim()))) {
            throw new MapperViolation(name + " must be enabled externally before CP7 loads");
        }
    }

    private static void requireExactInt(String name, int expected) {
        String value = configured(name);
        try {
            if (value == null || Integer.parseInt(value.trim()) != expected) {
                throw new MapperViolation(name + " must be exactly " + expected);
            }
        } catch (NumberFormatException error) {
            throw new MapperViolation(name + " must be exactly " + expected, error);
        }
    }

    private static String configured(String name) {
        String value = System.getProperty(name);
        return value == null || value.trim().isEmpty() ? System.getenv(name) : value;
    }

    private static XMageRallyBridgeProtocol.Seat parseSeat(String seat) {
        if ("p0".equals(seat)) {
            return XMageRallyBridgeProtocol.Seat.P0;
        }
        if ("p1".equals(seat)) {
            return XMageRallyBridgeProtocol.Seat.P1;
        }
        throw new IllegalArgumentException("physicalSeat must be exactly p0 or p1");
    }

    private static Map<String, Integer> rallyCardIds() {
        Map<String, Integer> ids = new HashMap<>();
        ids.put("Clockwork Percussionist", 16);
        ids.put("Voldaren Epicure", 127);
        ids.put("Goblin Bushwhacker", 44);
        ids.put("Goblin Tomb Raider", 45);
        ids.put("Burning-Tree Emissary", 10);
        ids.put("Galvanic Blast", 41);
        ids.put("Experimental Synthesizer", 30);
        ids.put("Lightning Bolt", 66);
        ids.put("Reckless Impulse", 93);
        ids.put("Rally at the Hornburg", 92);
        ids.put("Great Furnace", 48);
        ids.put("Mountain", 76);
        ids.put("Chain Lightning", 13);
        ids.put("End the Festivities", 27);
        ids.put("Blood Token", 132);
        ids.put("Human Soldier Token", 133);
        ids.put("Samurai Token", 134);
        return Collections.unmodifiableMap(ids);
    }

    private static Map<UUID, Integer> validateInitialArenaIds(
            Map<UUID, Integer> initialArenaIds) {
        if (initialArenaIds == null || initialArenaIds.isEmpty()) {
            return Collections.emptyMap();
        }
        if (initialArenaIds.size() != INITIAL_OBJECT_COUNT) {
            throw new IllegalArgumentException(
                    "initialArenaIds must cover exactly 120 cards");
        }
        Map<UUID, Integer> copy = new LinkedHashMap<>();
        Set<Integer> seenArenaIds = new HashSet<>();
        for (Map.Entry<UUID, Integer> entry : initialArenaIds.entrySet()) {
            UUID id = entry.getKey();
            Integer arenaId = entry.getValue();
            if (id == null || arenaId == null || arenaId < 0
                    || arenaId >= INITIAL_OBJECT_COUNT || !seenArenaIds.add(arenaId)) {
                throw new IllegalArgumentException(
                        "initialArenaIds must be a UUID bijection onto [0,120)");
            }
            copy.put(id, arenaId);
        }
        return Collections.unmodifiableMap(copy);
    }

    private static Set<String> immutableSet(String... values) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(values)));
    }

    private static String safe(String value) {
        return value == null ? "" : value.replace('|', '/').replace('\n', ' ');
    }

    private static void increment(Map<String, Long> values, String key) {
        Long previous = values.get(key);
        if (previous != null && previous == Long.MAX_VALUE) {
            throw new MapperViolation("counter exhausted for " + key);
        }
        values.put(key, previous == null ? 1L : previous + 1L);
    }

    /** Pure semantic checks; no XMage game or Rust process is launched. */
    public static void runFocusedSelfTest() {
        XMageRallyBridgeProtocol.ActionSemantic no =
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "choose_blocker_inclusion",
                        "{\"action_kind\":\"choose_blocker_inclusion\","
                                + "\"actor\":\"p1\","
                                + "\"attacker\":{\"arena_id\":3,\"card_db_id\":44},"
                                + "\"blocker\":{\"arena_id\":64,\"card_db_id\":127},"
                                + "\"include\":false}");
        XMageRallyBridgeProtocol.ActionSemantic yes =
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "choose_blocker_inclusion",
                        "{\"action_kind\":\"choose_blocker_inclusion\","
                                + "\"actor\":\"p1\","
                                + "\"attacker\":{\"arena_id\":3,\"card_db_id\":44},"
                                + "\"blocker\":{\"arena_id\":64,\"card_db_id\":127},"
                                + "\"include\":true}");
        JsonObject noRow = parseSemantic(no);
        JsonObject yesRow = parseSemantic(yes);
        if (requiredBoolean(noRow, "include", "self-test")
                || !requiredBoolean(yesRow, "include", "self-test")) {
            throw new IllegalStateException("binary include order self-test failed");
        }
        requireSameStable(requiredObject(noRow, "attacker", "self-test"),
                requiredObject(yesRow, "attacker", "self-test"), "self-test attacker");
        if (RALLY_CARD_IDS.get("Rally at the Hornburg") != 92
                || !"change_target".equals(booleanField("choose_spell_copy_retarget"))) {
            throw new IllegalStateException("Rally mapper frozen-vector self-test failed");
        }
        Mana oneRedMana = new Mana();
        oneRedMana.increaseRed();
        Mana twoRedMana = new Mana();
        twoRedMana.setRed(2);
        ManaOptions oneRedAvailable = new ManaOptions();
        oneRedAvailable.add(oneRedMana.copy());
        ManaOptions twoRedAvailable = new ManaOptions();
        twoRedAvailable.add(twoRedMana);
        if (!isJointlyUnaffordable(oneRedMana, oneRedMana, oneRedAvailable)
                || isJointlyUnaffordable(oneRedMana, oneRedMana, twoRedAvailable)) {
            throw new IllegalStateException(
                    "Bushwhacker joint-affordability self-test failed");
        }

        UUID firstCandidate = UUID.fromString("00000000-0000-0000-0000-000000000001");
        UUID stableCandidate = UUID.fromString("00000000-0000-0000-0000-000000000002");
        Map<Integer, UUID> combatBindings = new HashMap<>();
        combatBindings.put(134, stableCandidate);
        UUID resolved = mapBoundCombatUuid(
                134, Arrays.asList(firstCandidate, stableCandidate),
                combatBindings, Collections.emptySet(), "combat self-test");
        if (!stableCandidate.equals(resolved)) {
            throw new IllegalStateException(
                    "combat stable identity mapping used XMage candidate rank");
        }
        try {
            mapBoundCombatUuid(
                    134, Arrays.asList(stableCandidate, firstCandidate),
                    combatBindings, Collections.singleton(stableCandidate),
                    "combat duplicate self-test");
            throw new IllegalStateException(
                    "combat stable identity mapping accepted a repeated candidate");
        } catch (MapperViolation expected) {
            // Expected fail-closed duplicate rejection.
        }
        String scorerFailure = bridgeStepFailureMessage(
                2L, 3L, "XMAGE_RALLY_SCORER_ERROR error_code=clock_mismatch");
        if (!scorerFailure.contains("CP7_KERNEL_SHADOW_MAPPER_BRIDGE_STEP_FAILURE")
                || !scorerFailure.contains(
                "XMAGE_RALLY_SCORER_ERROR error_code=clock_mismatch")) {
            throw new IllegalStateException(
                    "CP7 bridge failure marker self-test failed");
        }
        requirePermutation(Arrays.asList(2, 0, 1), 3, "trigger self-test");
        if (factorialExact(3) != 6) {
            throw new IllegalStateException("trigger-order factorial self-test failed");
        }
        try {
            requirePermutation(Arrays.asList(0, 0), 2, "trigger duplicate self-test");
            throw new IllegalStateException(
                    "trigger-order duplicate permutation was accepted");
        } catch (MapperViolation expected) {
            // Expected fail-closed duplicate rejection.
        }
    }

    public static void main(String[] args) {
        runFocusedSelfTest();
        System.out.println("RallyCp7KernelShadowMapper focused self-test PASS");
    }

    private static final class TriggerOrderReplay {
        private final UUID gameId;
        private final List<TriggeredAbility> ordered;
        private int cursor;

        private TriggerOrderReplay(
                UUID gameId,
                List<TriggeredAbility> ordered,
                int cursor) {
            if (gameId == null || ordered == null || ordered.size() < 3
                    || cursor != 1) {
                throw new MapperViolation("invalid trigger-order replay construction");
            }
            this.gameId = gameId;
            this.ordered = new ArrayList<>(ordered);
            this.cursor = cursor;
        }

        private boolean consume(
                List<TriggeredAbility> offered,
                TriggeredAbility selected,
                UUID currentGameId) {
            if (!gameId.equals(currentGameId)
                    || offered == null
                    || offered.size() != ordered.size() - cursor) {
                throw new MapperViolation(
                        "trigger-order replay changed game or remaining width");
            }
            for (int i = 0; i < offered.size(); i++) {
                TriggeredAbility expected = ordered.get(cursor + i);
                TriggeredAbility actual = offered.get(i);
                if (actual != expected || actual.getId() == null
                        || !actual.getId().equals(expected.getId())) {
                    throw new MapperViolation(
                            "trigger-order replay changed identity or object order");
                }
            }
            if (selected != ordered.get(cursor)) {
                throw new MapperViolation(
                        "CP7 trigger-order replay did not choose the first offered trigger");
            }
            cursor++;
            return ordered.size() - cursor <= 1;
        }
    }

    private static final class StableBinding {
        private final int arenaId;
        private final int cardDbId;

        private StableBinding(int arenaId, int cardDbId) {
            this.arenaId = arenaId;
            this.cardDbId = cardDbId;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof StableBinding)) {
                return false;
            }
            StableBinding that = (StableBinding) other;
            return arenaId == that.arenaId && cardDbId == that.cardDbId;
        }

        @Override
        public int hashCode() {
            return 31 * arenaId + cardDbId;
        }
    }

    private static final class CombatStableRef {
        private final StableBinding binding;
        private final String owner;
        private final String controller;

        private CombatStableRef(
                StableBinding binding, String owner, String controller) {
            this.binding = binding;
            this.owner = owner;
            this.controller = controller;
        }
    }

    private static final class BinaryRefs {
        private final JsonObject primary;
        private final JsonObject secondary;

        private BinaryRefs(JsonObject primary, JsonObject secondary) {
            this.primary = primary;
            this.secondary = secondary;
        }
    }

    private static final class BlockAssignment {
        private final UUID blocker;
        private final UUID attacker;

        private BlockAssignment(UUID blocker, UUID attacker) {
            this.blocker = Objects.requireNonNull(blocker, "blocker");
            this.attacker = Objects.requireNonNull(attacker, "attacker");
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof BlockAssignment)) {
                return false;
            }
            BlockAssignment that = (BlockAssignment) other;
            return blocker.equals(that.blocker) && attacker.equals(that.attacker);
        }

        @Override
        public int hashCode() {
            return 31 * blocker.hashCode() + attacker.hashCode();
        }
    }

    public static final class MapperViolation extends IllegalStateException {
        private static final long serialVersionUID = 1L;

        public MapperViolation(String message) {
            super(message);
        }

        public MapperViolation(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
