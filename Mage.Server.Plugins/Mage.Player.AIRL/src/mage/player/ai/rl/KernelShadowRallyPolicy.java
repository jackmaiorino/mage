package mage.player.ai.rl;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.MageObject;
import mage.abilities.ActivatedAbility;
import mage.abilities.PlayLandAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.SimpleActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.effects.common.DrawCardSourceControllerEffect;
import mage.abilities.mana.ManaAbility;
import mage.abilities.mana.RedManaAbility;
import mage.cards.Card;
import mage.game.Game;
import mage.game.permanent.Permanent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Fail-closed live Rally policy backed by the promoted checkpoint shadow.
 *
 * <p>Exactly one instance is bound to one physical XMage seat. The two live
 * seat instances may share one process client, but snapshots never do:
 * {@link #copy()} returns an independent seeded-uniform simulation policy.</p>
 */
public final class KernelShadowRallyPolicy implements RallyCanonicalDecisionPolicy {

    private static final long serialVersionUID = 1L;

    private static final Set<String> PRIORITY_KINDS = kinds(
            "pass", "play_land", "cast_spell", "activate_mana_ability",
            "activate_ability", "plot_spell");
    private static final Set<String> TARGET_KINDS = kinds(
            "choose_target", "choose_effect_target", "finish_target_selection",
            "finish_effect_selection");
    private static final Set<String> CARD_TARGET_KINDS = kinds(
            "choose_target", "choose_cost_target", "choose_effect_target",
            "finish_target_selection", "finish_effect_selection", "discard");
    private static final Set<String> CHOICE_KINDS = kinds(
            "choose_effect_option", "choose_effect_color", "choose_effect_number",
            "choose_effect_boolean", "choose_optional_cost_which");
    private static final Set<String> MODE_KINDS = kinds(
            "choose_cast_mode", "choose_spell_mode");
    private static final Set<String> USE_KINDS = kinds(
            "choose_kicker", "choose_effect_boolean", "choose_optional_cost_use",
            "choose_madness_cast");
    private static final Map<String, Integer> RALLY_TOKEN_CARD_IDS = rallyTokenCardIds();
    private static final Map<String, Integer> RALLY_CARD_IDS = rallyCardIds();

    private transient final XMageRallyBridgeProcessClient bridge;
    private final long episodeId;
    private final XMageRallyBridgeProtocol.Seat physicalSeat;
    private final boolean modelControlled;
    private final RallyCanonicalDecisionPolicy delegate;
    private final SeededUniformMirrorPolicy simulationPolicyTemplate;
    private final Map<UUID, Integer> initialArenaIds;
    private final RallyCp7CounterfactualTeacher counterfactualTeacher;
    private final Map<UUID, Integer> dynamicArenaIds = new LinkedHashMap<>();
    private final Map<Integer, UUID> dynamicArenaUuids = new LinkedHashMap<>();

    private long requestOrdinal;
    private long physicalDecisionCount;
    private long policyActionSelections;
    private long policyLeafEvaluations;
    private long selectedPriorityProjectionCount;
    private final Map<String, Long> physicalDecisionCategories = new LinkedHashMap<>();
    private final Map<String, Long> outcomeHistogram = new LinkedHashMap<>();
    private boolean failed;
    private String firstFailure;

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            String physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy) {
        this(bridge, episodeId, parseSeat(physicalSeat), modelControlled,
                delegate, simulationPolicy, Collections.emptyMap());
    }

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            String physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy,
            Map<UUID, Integer> initialArenaIds) {
        this(bridge, episodeId, physicalSeat, modelControlled, delegate,
                simulationPolicy, initialArenaIds, null);
    }

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            String physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy,
            Map<UUID, Integer> initialArenaIds,
            RallyCp7CounterfactualTeacher counterfactualTeacher) {
        this(bridge, episodeId, parseSeat(physicalSeat), modelControlled,
                delegate, simulationPolicy, initialArenaIds, counterfactualTeacher);
    }

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            XMageRallyBridgeProtocol.Seat physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy) {
        this(bridge, episodeId, physicalSeat, modelControlled, delegate,
                simulationPolicy, Collections.emptyMap());
    }

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            XMageRallyBridgeProtocol.Seat physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy,
            Map<UUID, Integer> initialArenaIds) {
        this(bridge, episodeId, physicalSeat, modelControlled, delegate,
                simulationPolicy, initialArenaIds, null);
    }

    public KernelShadowRallyPolicy(
            XMageRallyBridgeProcessClient bridge,
            long episodeId,
            XMageRallyBridgeProtocol.Seat physicalSeat,
            boolean modelControlled,
            RallyCanonicalDecisionPolicy delegate,
            SeededUniformMirrorPolicy simulationPolicy,
            Map<UUID, Integer> initialArenaIds,
            RallyCp7CounterfactualTeacher counterfactualTeacher) {
        if (bridge == null) {
            throw new IllegalArgumentException("bridge must not be null");
        }
        if (episodeId < 0L) {
            throw new IllegalArgumentException("episodeId must be nonnegative");
        }
        if (physicalSeat == null) {
            throw new IllegalArgumentException("physicalSeat must not be null");
        }
        if (simulationPolicy == null) {
            throw new IllegalArgumentException("simulationPolicy must not be null");
        }
        if (modelControlled && delegate != null) {
            throw new IllegalArgumentException(
                    "model-controlled policy must not have an opponent delegate");
        }
        if (!modelControlled && delegate == null) {
            throw new IllegalArgumentException(
                    "noncandidate policy requires an opponent delegate");
        }
        if (delegate instanceof KernelShadowRallyPolicy) {
            throw new IllegalArgumentException(
                    "opponent delegate must not own another live shadow bridge");
        }

        this.bridge = bridge;
        this.episodeId = episodeId;
        this.physicalSeat = physicalSeat;
        this.modelControlled = modelControlled;
        this.delegate = delegate;
        this.simulationPolicyTemplate = simulationPolicy.copy();
        this.initialArenaIds = validateInitialArenaIds(initialArenaIds);
        this.counterfactualTeacher = counterfactualTeacher;

        validateActiveBindingAtConstruction();
    }

    /**
     * Return a bridge-free policy for XMage snapshots and simulations.
     */
    @Override
    public synchronized SeededUniformMirrorPolicy copy() {
        return simulationPolicyTemplate.copy();
    }

    @Override
    public synchronized int chooseNoncombat(String category, int canonicalLegalCount) {
        throw fail("live noncombat selection requires an XMage game clock", null);
    }

    public synchronized int chooseNoncombat(
            String category, int canonicalLegalCount, Game game) {
        requireLive();
        String checkedCategory = requireCategory(category);
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision =
                requireCurrentDecision(game, checkedCategory);
        validateSurfaceDecision(decision, checkedCategory, canonicalLegalCount, game);

        int selected;
        if (modelControlled) {
            selected = requireModelSelection(decision);
        } else {
            try {
                selected = delegate.chooseNoncombat(checkedCategory, canonicalLegalCount);
            } catch (RuntimeException error) {
                throw fail("opponent delegate failed for " + checkedCategory, error);
            }
        }
        validateSelectedIndex(selected, canonicalLegalCount);
        stepExactlyOnce(decision, selected, game, checkedCategory);
        recordSurfaceOutcome(checkedCategory, canonicalLegalCount, selected);
        return selected;
    }

    /**
     * Select a live XMage priority ability by the Rust semantic row rather
     * than assuming the two engines use the same menu order.
     */
    public synchronized ActivatedAbility choosePriorityAbility(
            List<? extends ActivatedAbility> xmageAbilities, Game game) {
        return choosePriorityAbility(xmageAbilities, game, null);
    }

    public synchronized ActivatedAbility choosePriorityAbility(
            List<? extends ActivatedAbility> xmageAbilities,
            Game game,
            UUID physicalPlayerId) {
        requireLive();
        if (xmageAbilities == null || xmageAbilities.isEmpty()) {
            throw fail("priority ability menu must be nonempty", null);
        }
        if (game == null) {
            throw fail("priority ability selection requires an XMage game", null);
        }
        String category = "noncombat_activate_ability_or_spell";
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision =
                requireCurrentDecision(game, category);
        try {
            bindPriorityTokenSources(
                    xmageAbilities, decision.getActionSemantics(), game);
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }
        if (decision.getLegalActionCount() != xmageAbilities.size()) {
            throw fail(category + " legal width mismatch: XMage=" + xmageAbilities.size()
                    + " kernel=" + decision.getLegalActionCount()
                    + " phase=" + game.getTurnStepType()
                    + " xmage_rows=" + xmagePrioritySummary(xmageAbilities, game)
                    + " kernel_rows=" + rustPrioritySummary(decision.getActionSemantics())
                    + " bound_sources=" + boundSourceSummary(
                    decision.getActionSemantics(), game)
                    + " xmage_lands=" + xmageLandSummary(game), null);
        }
        validateSurfaceDecision(decision, category, xmageAbilities.size(), game);

        List<ActivatedAbility> abilitiesByRustRow;
        try {
            abilitiesByRustRow = mapPriorityRows(
                    xmageAbilities, decision.getActionSemantics(), allArenaIds());
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }

        if (counterfactualTeacher != null) {
            if (!modelControlled || physicalPlayerId == null) {
                throw fail("counterfactual teacher lacks the model-controlled player id", null);
            }
            counterfactualTeacher.capturePriority(
                    decision, abilitiesByRustRow, game, physicalPlayerId);
        }

        int selected;
        if (modelControlled) {
            selected = requireModelSelection(decision);
        } else {
            try {
                selected = delegate.chooseNoncombat(category, abilitiesByRustRow.size());
            } catch (RuntimeException error) {
                throw fail("opponent delegate failed for " + category, error);
            }
        }
        validateSelectedIndex(selected, abilitiesByRustRow.size());
        ActivatedAbility result = abilitiesByRustRow.get(selected);
        stepExactlyOnce(decision, selected, game, category);
        recordSurfaceOutcome(category, abilitiesByRustRow.size(), selected);
        return result;
    }

    /**
     * Return true only when a legacy XMage auto-pass window corresponds to
     * this seat's current native priority menu. Expected phase-skipping
     * mismatches are observational and do not fail or advance the bridge.
     */
    public synchronized boolean matchesCurrentPriorityMenu(
            List<? extends ActivatedAbility> xmageAbilities, Game game) {
        requireLive();
        if (xmageAbilities == null || xmageAbilities.isEmpty() || game == null) {
            return false;
        }
        XMageRallyBridgeProtocol.DecisionBody decision = bridge.getCurrentDecision();
        if (decision == null || decision.getActingPlayer() != physicalSeat
                || !"surface".equals(decision.getDecisionKind())
                || decision.getLegalActionCount() != xmageAbilities.size()
                || decision.getActionSemantics() == null) {
            tracePriorityMenuMatch("shape_mismatch", xmageAbilities, game, decision);
            return false;
        }
        for (XMageRallyBridgeProtocol.ActionSemantic semantic
                : decision.getActionSemantics()) {
            if (semantic == null || !PRIORITY_KINDS.contains(semantic.getActionKind())) {
                tracePriorityMenuMatch(
                        "nonpriority_semantic", xmageAbilities, game, decision);
                return false;
            }
        }
        Map<UUID, Integer> dynamicArenaIdsBefore =
                new LinkedHashMap<>(dynamicArenaIds);
        Map<Integer, UUID> dynamicArenaUuidsBefore =
                new LinkedHashMap<>(dynamicArenaUuids);
        try {
            bindPriorityTokenSources(
                    xmageAbilities, decision.getActionSemantics(), game);
            mapPriorityRows(
                    xmageAbilities, decision.getActionSemantics(), allArenaIds());
        } catch (KernelShadowPolicyViolation expectedPhaseMismatch) {
            dynamicArenaIds.clear();
            dynamicArenaIds.putAll(dynamicArenaIdsBefore);
            dynamicArenaUuids.clear();
            dynamicArenaUuids.putAll(dynamicArenaUuidsBefore);
            tracePriorityMenuMatch("mapping_mismatch", xmageAbilities, game, decision);
            return false;
        }
        try {
            XMageRallyClockComparator.requireMatch(
                    decision, game, "kernel_shadow_policy", "priority_menu_match");
        } catch (XMageRallyClockComparator.ClockMismatch expectedPhaseMismatch) {
            dynamicArenaIds.clear();
            dynamicArenaIds.putAll(dynamicArenaIdsBefore);
            dynamicArenaUuids.clear();
            dynamicArenaUuids.putAll(dynamicArenaUuidsBefore);
            tracePriorityMenuMatch("clock_mismatch", xmageAbilities, game, decision);
            return false;
        }
        return true;
    }

    /**
     * Consume the Rust-selected priority row at the caller-verified active,
     * empty-stack postcombat rendezvous when XMage exposes that exact stable
     * action inside a non-identical menu. A missing or ambiguous selected row
     * fails closed, and no substitute action is ever submitted to Rust.
     */
    public synchronized ActivatedAbility chooseSelectedPriorityAbilityAtPostcombatRendezvous(
            List<? extends ActivatedAbility> xmageAbilities, Game game) {
        requireLive();
        if (!modelControlled) {
            return null;
        }
        if (xmageAbilities == null || xmageAbilities.isEmpty() || game == null) {
            throw fail("postcombat selected-action projection lacks a live XMage menu", null);
        }
        if (game.getTurnStepType() != mage.constants.PhaseStep.POSTCOMBAT_MAIN
                || !game.getStack().isEmpty()
                || !physicalSeat.wire().equals(
                seatFor(game.getActivePlayerId(), game))) {
            throw fail("selected-action projection escaped the active empty-stack"
                    + " postcombat rendezvous", null);
        }
        XMageRallyBridgeProtocol.DecisionBody decision = bridge.getCurrentDecision();
        if (decision == null) {
            throw fail(bridge.getTerminal() == null
                    ? "bridge has neither decision nor terminal at postcombat rendezvous"
                    : "kernel reached terminal before the postcombat rendezvous", null);
        }
        if (decision.getEpisodeId() != episodeId) {
            throw fail("postcombat rendezvous crossed the bound episode", null);
        }
        if (decision.getActingPlayer() != physicalSeat) {
            return null;
        }
        if (!"surface".equals(decision.getDecisionKind())
                || decision.getSubstepIndex() != 0
                || decision.getSubstepCount() != 1
                || decision.getActionSemantics() == null) {
            tracePriorityMenuMatch(
                    "selected_projection_not_priority_surface",
                    xmageAbilities, game, decision);
            return null;
        }
        validateCommonDecisionBinding(decision, "postcombat_selected_projection");
        try {
            for (XMageRallyBridgeProtocol.ActionSemantic semantic
                    : decision.getActionSemantics()) {
                if (semantic == null) {
                    throw new KernelShadowPolicyViolation(
                            "candidate postcombat rendezvous contains a null semantic");
                }
                validateSemanticActor(semantic, physicalSeat.wire());
                if (!PRIORITY_KINDS.contains(semantic.getActionKind())) {
                    tracePriorityMenuMatch(
                            "selected_projection_not_priority_semantic",
                            xmageAbilities, game, decision);
                    return null;
                }
            }
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }

        int selectedIndex = requireModelSelection(decision);
        XMageRallyBridgeProtocol.ActionSemantic selectedSemantic =
                decision.getActionSemantics().get(selectedIndex);
        if (!"pass".equals(selectedSemantic.getActionKind())
                && priorityArenaId(selectedSemantic) >= 120) {
            throw fail("selected-action projection requires an original deck object"
                    + " or pass; generated token identity lacks full-menu authority", null);
        }
        ActivatedAbility selected;
        try {
            selected = mapSelectedPriorityRow(
                    xmageAbilities,
                    selectedSemantic,
                    allArenaIds());
            validateSelectedPriorityProjection(selectedSemantic, selected, game);
        } catch (KernelShadowPolicyViolation expectedPhaseMismatch) {
            tracePriorityMenuMatch(
                    "selected_row_absent", xmageAbilities, game, decision);
            throw fail(
                    "Rust-selected priority row is absent at the active empty-stack"
                            + " postcombat rendezvous",
                    expectedPhaseMismatch);
        }

        if (selectedPriorityProjectionCount == Long.MAX_VALUE) {
            throw fail("selected-priority projection counter exhausted", null);
        }
        ensureCounterCapacity(1);
        tracePriorityMenuMatch(
                "selected_row_catchup", xmageAbilities, game, decision);
        stepExactlyOnce(
                decision, selectedIndex, game, "postcombat_selected_projection");
        selectedPriorityProjectionCount++;
        recordSurfaceOutcome(
                "noncombat_activate_ability_or_spell",
                decision.getLegalActionCount(), selectedIndex);
        return selected;
    }

    private void validateSelectedPriorityProjection(
            XMageRallyBridgeProtocol.ActionSemantic semantic,
            ActivatedAbility selected,
            Game game) {
        String kind = semantic.getActionKind();
        if ("pass".equals(kind)) {
            if (!(selected instanceof PassAbility)) {
                throw new KernelShadowPolicyViolation(
                        "selected pass projection did not resolve to XMage pass");
            }
            return;
        }
        JsonObject row = parseSemantic(semantic, kind);
        JsonObject source = requiredObject(row, "source", "selected projection source");
        int arenaId = requiredUnsignedInt(
                source, "arena_id", "selected projection source");
        UUID sourceId = selected.getSourceId();
        MageObject live = sourceId == null ? null : game.getObject(sourceId);
        Integer mappedArena = sourceId == null ? null : initialArenaIds.get(sourceId);
        Integer liveCardDbId = live == null ? null : RALLY_CARD_IDS.get(live.getName());
        String expectedZone = requiredStringValue(
                source, "zone", "selected projection source");
        mage.constants.Zone liveZone = sourceId == null
                ? null : game.getState().getZone(sourceId);
        int expectedZoneChange = requiredUnsignedInt(
                source, "zone_change_count", "selected projection source");
        if (expectedZoneChange == Integer.MAX_VALUE) {
            throw new KernelShadowPolicyViolation(
                    "selected projection source zone-change count cannot be translated");
        }
        // XMage materializes the loaded deck in Library before the first
        // hand move; Rust starts those same original objects in Library at
        // zone-change count zero. Their stable transport relation is +1.
        int expectedXMageZoneChange = expectedZoneChange + 1;
        int expectedCardDbId = requiredUnsignedInt(
                source, "card_db_id", "selected projection source");
        String expectedOwner = requiredStringValue(
                source, "owner", "selected projection source");
        String expectedController = requiredStringValue(
                source, "controller", "selected projection source");
        int liveZoneChange = sourceId == null
                ? -1 : game.getState().getZoneChangeCounter(sourceId);
        String liveOwner = live == null ? "none" : seatFor(game.getOwnerId(live), game);
        String liveController = sourceId == null
                ? "none" : seatFor(game.getControllerId(sourceId), game);
        if (live == null || mappedArena == null || mappedArena != arenaId
                || liveCardDbId == null
                || liveCardDbId != expectedCardDbId
                || liveZone == null
                || !expectedZone.equalsIgnoreCase(liveZone.name())
                || liveZoneChange != expectedXMageZoneChange
                || !expectedOwner.equals(liveOwner)
                || !expectedController.equals(liveController)) {
            throw new KernelShadowPolicyViolation(
                    "selected projection source does not match live XMage identity:"
                            + " expected=" + arenaId + "/" + expectedCardDbId
                            + "/" + expectedZone + "/zcc" + expectedZoneChange
                            + "(xmage=" + expectedXMageZoneChange + ")"
                            + "/" + expectedOwner + "/" + expectedController
                            + " actual=" + mappedArena + "/" + liveCardDbId
                            + "/" + (liveZone == null ? "none" : liveZone.name())
                            + "/zcc" + liveZoneChange + "/" + liveOwner
                            + "/" + liveController);
        }
        if ("cast_spell".equals(kind)) {
            if (!(selected instanceof SpellAbility) || liveZone != mage.constants.Zone.HAND) {
                throw new KernelShadowPolicyViolation(
                        "selected cast projection is not an ordinary hand spell");
            }
            return;
        }
        if ("play_land".equals(kind)) {
            if (!(selected instanceof PlayLandAbility)
                    || liveZone != mage.constants.Zone.HAND) {
                throw new KernelShadowPolicyViolation(
                        "selected land projection is not an ordinary hand land");
            }
            return;
        }
        if ("activate_mana_ability".equals(kind)) {
            String manaChoice = requiredStringValue(
                    row, "mana_choice", "selected mana projection");
            if (!(selected instanceof ManaAbility)
                    || liveZone != mage.constants.Zone.BATTLEFIELD
                    || liveCardDbId != 76 || !"Mountain".equals(live.getName())
                    || !"R".equals(manaChoice)) {
                throw new KernelShadowPolicyViolation(
                        "selected mana projection is not the proven red Mountain form");
            }
            return;
        }
        throw new KernelShadowPolicyViolation(
                "selected projection rejects unproved priority kind " + kind);
    }

    private void tracePriorityMenuMatch(
            String outcome,
            List<? extends ActivatedAbility> xmageAbilities,
            Game game,
            XMageRallyBridgeProtocol.DecisionBody decision) {
        if (!Boolean.getBoolean("xmage.rally.traceCp7Mapper")) {
            return;
        }
        String selected = "none";
        String rustRows = "none";
        if (decision != null) {
            Integer index = decision.getSelectedActionIndex();
            if (index != null) {
                if (decision.getActionSemantics() == null
                        || index < 0 || index >= decision.getActionSemantics().size()) {
                    selected = index + ":invalid";
                } else {
                    selected = index + ":"
                            + decision.getActionSemantics().get(index).getCanonicalJson();
                }
            }
            rustRows = rustCombatSummary(decision.getActionSemantics());
        }
        System.err.println("XMAGE_RALLY_CANDIDATE_PRIORITY_TRACE"
                + " episode=" + episodeId
                + " outcome=" + outcome
                + " turn=" + (game == null ? -1 : game.getTurnNum())
                + " phase=" + (game == null ? "none" : game.getTurnStepType())
                + " stack=" + (game == null ? -1 : game.getStack().size())
                + " rust_actor=" + (decision == null ? "none"
                : decision.getActingPlayer().wire())
                + " rust_step=" + (decision == null ? -1 : decision.getStep())
                + " rust_kind=" + (decision == null ? "none"
                : decision.getDecisionKind())
                + " rust_selected=" + selected
                + " xmage=" + xmagePrioritySummary(xmageAbilities, game)
                + " rust=" + rustRows);
    }

    /** Select a card callback by Rust stable arena identity, not XMage rank. */
    public synchronized Card chooseCardTarget(
            List<? extends Card> xmageMenu, Game game) {
        requireLive();
        if (xmageMenu == null || xmageMenu.isEmpty() || game == null) {
            throw fail("card-target menu and game must be nonempty", null);
        }
        String category = "card_target";
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision =
                requireCurrentDecision(game, category);
        validateSurfaceDecision(decision, category, xmageMenu.size(), game);

        List<UUID> xmageIds = new ArrayList<>(xmageMenu.size());
        Map<UUID, Card> cardsById = new LinkedHashMap<>();
        for (Card card : xmageMenu) {
            UUID id = card == null ? null : card.getId();
            if (card != null && cardsById.put(id, card) != null) {
                throw fail("card-target menu repeats a card UUID", null);
            }
            xmageIds.add(id);
        }

        List<UUID> idsByRustRow;
        try {
            idsByRustRow = mapCardTargetRows(
                    xmageIds, decision.getActionSemantics(), allArenaIds());
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }

        int selected;
        if (modelControlled) {
            selected = requireModelSelection(decision);
        } else {
            try {
                selected = delegate.chooseNoncombat(category, idsByRustRow.size());
            } catch (RuntimeException error) {
                throw fail("opponent delegate failed for " + category, error);
            }
        }
        validateSelectedIndex(selected, idsByRustRow.size());
        UUID selectedId = idsByRustRow.get(selected);
        Card result = selectedId == null ? null : cardsById.get(selectedId);
        if (selectedId != null && result == null) {
            throw fail("selected card-target identity is absent from XMage menu", null);
        }
        if (Boolean.getBoolean("xmage.rally.traceActions")) {
            System.err.println("XMAGE_RALLY_CARD_TARGET_TRACE episode=" + episodeId
                    + " step=" + decision.getStep()
                    + " rust=" + decision.getActionSemantics().get(selected).getCanonicalJson()
                    + " xmage_arena_id=" + allArenaIds().get(selectedId)
                    + " xmage_card=" + (result == null ? "STOP" : result.getName()));
        }
        stepExactlyOnce(decision, selected, game, category);
        recordSurfaceOutcome(category, idsByRustRow.size(), selected);
        return result;
    }

    /** Select a live object, player, or STOP target by Rust stable identity. */
    public synchronized UUID chooseTarget(List<UUID> xmageMenu, Game game) {
        requireLive();
        if (xmageMenu == null || xmageMenu.isEmpty() || game == null) {
            throw fail("target menu and game must be nonempty", null);
        }
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision =
                requireCurrentDecision(game, "target");
        boolean cardIdentity = decision.getActionSemantics().stream()
                .anyMatch(semantic -> semantic != null
                        && ("discard".equals(semantic.getActionKind())
                        || "choose_cost_target".equals(semantic.getActionKind())));
        String category = cardIdentity ? "card_target" : "target";
        validateSurfaceDecision(decision, category, xmageMenu.size(), game);

        List<UUID> idsByRustRow;
        try {
            if (cardIdentity) {
                idsByRustRow = mapCardTargetRows(
                        xmageMenu, decision.getActionSemantics(), allArenaIds());
            } else {
                bindTargetTokenObjects(
                        xmageMenu, decision.getActionSemantics(), game);
                idsByRustRow = mapTargetRows(
                        xmageMenu, decision.getActionSemantics(), allArenaIds(),
                        targetPlayerIds(game));
            }
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }

        int selected;
        if (modelControlled) {
            selected = requireModelSelection(decision);
        } else {
            try {
                selected = delegate.chooseNoncombat(category, idsByRustRow.size());
            } catch (RuntimeException error) {
                throw fail("opponent delegate failed for " + category, error);
            }
        }
        validateSelectedIndex(selected, idsByRustRow.size());
        UUID result = idsByRustRow.get(selected);
        if (Boolean.getBoolean("xmage.rally.traceActions")) {
            System.err.println("XMAGE_RALLY_TARGET_TRACE episode=" + episodeId
                    + " step=" + decision.getStep()
                    + " rust=" + decision.getActionSemantics().get(selected).getCanonicalJson()
                    + " xmage_uuid=" + (result == null ? "STOP" : result));
        }
        stepExactlyOnce(decision, selected, game, category);
        recordSurfaceOutcome(category, idsByRustRow.size(), selected);
        return result;
    }

    @Override
    public synchronized int[] chooseNoncombatWithoutReplacement(
            String category, int canonicalLegalCount, int picks) {
        throw fail("live aggregate selection requires an XMage game clock", null);
    }

    public synchronized int[] chooseNoncombatWithoutReplacement(
            String category, int canonicalLegalCount, int picks, Game game) {
        requireLive();
        if (picks != 1) {
            throw fail("surface callback must contain exactly one Rust substep; picks=" + picks,
                    null);
        }
        String checkedCategory = requireCategory(category);
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision =
                requireCurrentDecision(game, checkedCategory);
        validateSurfaceDecision(decision, checkedCategory, canonicalLegalCount, game);

        int selected;
        if (modelControlled) {
            selected = requireModelSelection(decision);
        } else {
            int[] delegated;
            try {
                delegated = delegate.chooseNoncombatWithoutReplacement(
                        checkedCategory, canonicalLegalCount, picks);
            } catch (RuntimeException error) {
                throw fail("opponent delegate failed for " + checkedCategory, error);
            }
            if (delegated == null || delegated.length != 1) {
                throw fail("opponent delegate returned a non-singleton surface selection", null);
            }
            selected = delegated[0];
        }
        validateSelectedIndex(selected, canonicalLegalCount);
        stepExactlyOnce(decision, selected, game, checkedCategory);
        recordSurfaceOutcome(checkedCategory, canonicalLegalCount, selected);
        return new int[]{selected};
    }

    @Override
    public synchronized boolean[] chooseAttackers(int canonicalEligibleCount) {
        throw fail("live attacker selection requires XMage permanent identities", null);
    }

    @Override
    public synchronized boolean[] chooseBlockers(int canonicalLegalBlockerCount) {
        throw fail("live blocker selection requires XMage permanent identities", null);
    }

    @Override
    public synchronized int chooseBlocker(int canonicalLegalBlockerCount) {
        throw fail("live blocker selection requires XMage permanent identities", null);
    }

    /**
     * Consume one native attacker scan and return the selected XMage
     * permanent identities in native scan order.
     */
    public synchronized List<UUID> chooseAttackerIds(
            List<? extends Permanent> xmageEligible, Game game) {
        requireLive();
        List<Permanent> eligible = requireCombatPermanents(
                xmageEligible, game, "attacker");
        if (eligible.isEmpty()) {
            return Collections.emptyList();
        }

        String category = "declare_attackers";
        int candidateCount = eligible.size();
        ensureCounterCapacity(candidateCount);
        XMageRallyBridgeProtocol.DecisionBody firstDecision =
                requireCurrentDecision(game, category);
        boolean[] delegated = delegatedCombatSelection(category, candidateCount, false);
        List<UUID> selected = new ArrayList<>();
        Set<UUID> used = new HashSet<>();
        Long physicalDecisionId = null;

        for (int substep = 0; substep < candidateCount; substep++) {
            XMageRallyBridgeProtocol.DecisionBody decision = substep == 0
                    ? firstDecision : requireCurrentDecision(game, category);
            BinaryShape shape = validateCombatDecision(
                    decision, category, "attacker_inclusion",
                    "choose_attacker_inclusion", candidateCount,
                    substep, physicalDecisionId, game);
            if (physicalDecisionId == null) {
                physicalDecisionId = decision.getPhysicalDecisionId();
            }
            Permanent attacker;
            try {
                attacker = resolveCombatPermanent(
                        shape.attacker, eligible, used, game,
                        "attacker substep " + substep);
            } catch (KernelShadowPolicyViolation error) {
                throw fail(error.getMessage(), error);
            }
            if (!used.add(attacker.getId())) {
                throw fail("attacker aggregate repeats an XMage identity", null);
            }

            int selectedIndex = selectedCombatIndex(decision, delegated, substep, category);
            if (selectedIndex == 1) {
                selected.add(attacker.getId());
            }
            traceCombatSelection(decision, shape.attacker, attacker, selectedIndex, null);
            stepExactlyOnce(decision, selectedIndex, game, category);
            policyActionSelections++;
            policyLeafEvaluations++;
        }
        if (used.size() != candidateCount) {
            throw fail("attacker mapping did not consume the complete XMage menu", null);
        }
        recordCombatOutcome(category, candidateCount, selected.size());
        return Collections.unmodifiableList(selected);
    }

    /**
     * Consume every consecutive native blocker group for this declaration.
     * Native stable refs choose both the attacker group and each blocker, so
     * neither engine's iteration order becomes policy meaning.
     */
    public synchronized List<BlockAssignment> chooseBlockAssignments(
            List<? extends Permanent> xmageAttackers,
            List<? extends Permanent> xmageAvailableBlockers,
            Game game) {
        requireLive();
        List<Permanent> pendingAttackers = requireCombatPermanents(
                xmageAttackers, game, "blocking attacker");
        List<Permanent> availableBlockers = requireCombatPermanents(
                xmageAvailableBlockers, game, "available blocker");
        if (!hasLegalBlockerPair(pendingAttackers, availableBlockers, game)) {
            return Collections.emptyList();
        }

        List<BlockAssignment> assignments = new ArrayList<>();
        while (hasLegalBlockerPair(pendingAttackers, availableBlockers, game)) {
            XMageRallyBridgeProtocol.DecisionBody firstDecision =
                    requireCurrentDecision(game, "declare_blocker_for_attacker");
            if (!"blocker_inclusion".equals(firstDecision.getDecisionKind())) {
                throw fail("XMage has a blocker group but kernel decision kind is "
                        + firstDecision.getDecisionKind(), null);
            }
            validateCommonDecision(
                    firstDecision, game, "declare_blocker_for_attacker");

            BinaryShape firstShape;
            try {
                firstShape = validateBinarySemanticOrder(
                        firstDecision, "choose_blocker_inclusion");
            } catch (KernelShadowPolicyViolation error) {
                throw fail(error.getMessage(), error);
            }
            List<Permanent> attackersWithChoices = attackersWithLegalBlockers(
                    pendingAttackers, availableBlockers, game);
            Permanent attacker;
            try {
                attacker = resolveCombatPermanent(
                        firstShape.attacker, attackersWithChoices,
                        Collections.emptySet(), game, "blocker group attacker");
            } catch (KernelShadowPolicyViolation error) {
                throw fail(error.getMessage(), error);
            }

            List<Permanent> legalBlockers = legalBlockersFor(
                    attacker, availableBlockers, game);
            int candidateCount = legalBlockers.size();
            String category = "declare_blocker_for_attacker";
            ensureCounterCapacity(candidateCount);
            boolean[] delegated = delegatedCombatSelection(
                    category, candidateCount, true);
            Set<UUID> usedBlockers = new HashSet<>();
            List<Permanent> selectedForAttacker = new ArrayList<>();
            Long physicalDecisionId = null;

            for (int substep = 0; substep < candidateCount; substep++) {
                XMageRallyBridgeProtocol.DecisionBody decision = substep == 0
                        ? firstDecision : requireCurrentDecision(game, category);
                BinaryShape shape = validateCombatDecision(
                        decision, category, "blocker_inclusion",
                        "choose_blocker_inclusion", candidateCount,
                        substep, physicalDecisionId, game);
                if (physicalDecisionId == null) {
                    physicalDecisionId = decision.getPhysicalDecisionId();
                }
                Permanent boundAttacker;
                Permanent blocker;
                try {
                    boundAttacker = resolveCombatPermanent(
                            shape.attacker, Collections.singletonList(attacker),
                            Collections.emptySet(), game,
                            "blocker attacker substep " + substep);
                    blocker = resolveCombatPermanent(
                            shape.blocker, legalBlockers, usedBlockers, game,
                            "blocker substep " + substep);
                } catch (KernelShadowPolicyViolation error) {
                    throw fail(error.getMessage(), error);
                }
                if (!boundAttacker.getId().equals(attacker.getId())) {
                    throw fail("blocker aggregate changed its fixed attacker", null);
                }
                if (!usedBlockers.add(blocker.getId())) {
                    throw fail("blocker aggregate repeats an XMage identity", null);
                }

                int selectedIndex = selectedCombatIndex(
                        decision, delegated, substep, category);
                if (selectedIndex == 1) {
                    selectedForAttacker.add(blocker);
                    assignments.add(new BlockAssignment(
                            blocker.getId(), attacker.getId()));
                }
                traceCombatSelection(
                        decision, shape.blocker, blocker, selectedIndex, attacker);
                stepExactlyOnce(decision, selectedIndex, game, category);
                policyActionSelections++;
                policyLeafEvaluations++;
            }
            if (usedBlockers.size() != candidateCount) {
                throw fail("blocker mapping did not consume the complete XMage menu", null);
            }
            availableBlockers.removeAll(selectedForAttacker);
            pendingAttackers.remove(attacker);
            recordCombatOutcome(category, candidateCount, selectedForAttacker.size());
        }

        XMageRallyBridgeProtocol.DecisionBody remaining = bridge.getCurrentDecision();
        if (remaining != null && "blocker_inclusion".equals(remaining.getDecisionKind())) {
            throw fail("kernel retains a blocker group after XMage exhausted legal pairs", null);
        }
        return Collections.unmodifiableList(assignments);
    }

    @Override
    public synchronized long getPhysicalDecisionCount() {
        return physicalDecisionCount;
    }

    @Override
    public synchronized long getPolicyActionSelections() {
        return policyActionSelections;
    }

    @Override
    public synchronized long getPolicyLeafEvaluations() {
        return policyLeafEvaluations;
    }

    public synchronized long getSelectedPriorityProjectionCount() {
        return selectedPriorityProjectionCount;
    }

    @Override
    public synchronized Map<String, Long> getPhysicalDecisionCategories() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(physicalDecisionCategories));
    }

    @Override
    public synchronized Map<String, Long> getOutcomeHistogram() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(outcomeHistogram));
    }

    public long getEpisodeId() {
        return episodeId;
    }

    public String getPhysicalSeat() {
        return physicalSeat.wire();
    }

    public boolean isModelControlled() {
        return modelControlled;
    }

    public synchronized boolean isFailed() {
        return failed;
    }

    public synchronized String getFirstFailure() {
        return firstFailure;
    }

    /** Opt-in live diagnostic used by the Rally spike's mana trace. */
    public synchronized String describeXmageLands(Game game) {
        return xmageLandSummary(game);
    }

    private List<Permanent> requireCombatPermanents(
            List<? extends Permanent> candidates, Game game, String label) {
        if (candidates == null || game == null) {
            throw fail(label + " candidate surface requires a live game", null);
        }
        List<Permanent> live = new ArrayList<>(candidates.size());
        Set<UUID> seen = new HashSet<>();
        for (Permanent candidate : candidates) {
            UUID id = candidate == null ? null : candidate.getId();
            Permanent permanent = id == null ? null : game.getPermanent(id);
            if (permanent == null || !seen.add(id)) {
                throw fail(label + " candidate surface contains a null, stale, or duplicate identity",
                        null);
            }
            live.add(permanent);
        }
        return live;
    }

    private boolean[] delegatedCombatSelection(
            String category, int candidateCount, boolean blockerGroup) {
        if (modelControlled) {
            return null;
        }
        boolean[] delegated;
        try {
            delegated = blockerGroup
                    ? delegate.chooseBlockers(candidateCount)
                    : delegate.chooseAttackers(candidateCount);
        } catch (RuntimeException error) {
            throw fail("opponent delegate failed for " + category, error);
        }
        if (delegated == null || delegated.length != candidateCount) {
            throw fail("opponent delegate returned an invalid " + category + " vector", null);
        }
        return delegated;
    }

    private int selectedCombatIndex(
            XMageRallyBridgeProtocol.DecisionBody decision,
            boolean[] delegated,
            int substep,
            String category) {
        int selectedIndex = modelControlled
                ? requireModelSelection(decision)
                : delegated[substep] ? 1 : 0;
        if (selectedIndex != 0 && selectedIndex != 1) {
            throw fail(category + " selected a non-binary action index", null);
        }
        return selectedIndex;
    }

    private void recordCombatOutcome(String category, int legalCount, int includedCount) {
        physicalDecisionCount++;
        increment(physicalDecisionCategories, category);
        increment(outcomeHistogram, category + "|legal=" + legalCount
                + "|included=" + includedCount);
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
        StableCombatRef reference = stableCombatRef(stable, label);
        List<UUID> candidateIds = new ArrayList<>(candidates.size());
        for (Permanent candidate : candidates) {
            candidateIds.add(candidate.getId());
        }
        UUID bound = mapBoundCombatUuid(
                reference.arenaId, candidateIds, allArenaIds(), used, label);
        if (bound != null) {
            Permanent permanent = game.getPermanent(bound);
            validateCombatPermanent(permanent, reference, game, label);
            return permanent;
        }
        if (reference.arenaId < 120) {
            throw new KernelShadowPolicyViolation(
                    label + " has no XMage binding for opening arena id "
                            + reference.arenaId);
        }

        List<Permanent> compatible = new ArrayList<>();
        for (Permanent candidate : candidates) {
            UUID id = candidate.getId();
            if (used.contains(id) || initialArenaIds.containsKey(id)
                    || dynamicArenaIds.containsKey(id)) {
                continue;
            }
            Integer cardDbId = RALLY_TOKEN_CARD_IDS.get(candidate.getName());
            if (cardDbId != null && cardDbId == reference.cardDbId
                    && reference.controller.equals(
                    seatFor(candidate.getControllerId(), game))) {
                compatible.add(candidate);
            }
        }
        compatible.sort(Comparator.comparingInt(permanent ->
                battlefieldIndex(permanent.getId(), game)));
        if (compatible.isEmpty()) {
            throw new KernelShadowPolicyViolation(
                    label + " cannot bind generated arena id " + reference.arenaId
                            + " card_db_id=" + reference.cardDbId);
        }
        // Native attacker/blocker candidates filter the battlefield Vec in
        // place, while XMage's Battlefield exposes LinkedHashMap insertion
        // order. Pair equal unbound tokens by that shared creation rank.
        Permanent selected = compatible.get(0);
        putDynamicArenaBinding(selected.getId(), reference.arenaId, label);
        validateCombatPermanent(selected, reference, game, label);
        return selected;
    }

    private static UUID mapBoundCombatUuid(
            int arenaId,
            List<UUID> candidateIds,
            Map<UUID, Integer> arenaIds,
            Set<UUID> used,
            String label) {
        UUID matched = null;
        int matches = 0;
        for (UUID id : candidateIds) {
            Integer candidateArena = arenaIds.get(id);
            if (candidateArena != null && candidateArena == arenaId) {
                matched = id;
                matches++;
            }
        }
        if (matches > 1) {
            throw new KernelShadowPolicyViolation(
                    label + " maps one arena id to multiple XMage candidates");
        }
        if (matched != null && used.contains(matched)) {
            throw new KernelShadowPolicyViolation(
                    label + " repeats an already consumed XMage candidate");
        }
        return matched;
    }

    private void validateCombatPermanent(
            Permanent permanent, StableCombatRef reference, Game game, String label) {
        if (permanent == null || game.getPermanent(permanent.getId()) == null) {
            throw new KernelShadowPolicyViolation(label + " resolved a stale permanent");
        }
        if (!reference.controller.equals(seatFor(permanent.getControllerId(), game))) {
            throw new KernelShadowPolicyViolation(label + " controller does not match Rust");
        }
        if (!reference.owner.equals(seatFor(permanent.getOwnerId(), game))) {
            throw new KernelShadowPolicyViolation(label + " owner does not match Rust");
        }
        if (reference.arenaId >= 120) {
            Integer cardDbId = RALLY_TOKEN_CARD_IDS.get(permanent.getName());
            if (cardDbId == null || cardDbId != reference.cardDbId) {
                throw new KernelShadowPolicyViolation(
                        label + " generated token definition does not match Rust");
            }
        }
    }

    private void putDynamicArenaBinding(UUID id, int arenaId, String label) {
        if (id == null || arenaId < 120 || initialArenaIds.containsKey(id)
                || initialArenaIds.containsValue(arenaId)) {
            throw new KernelShadowPolicyViolation(label + " has an invalid dynamic binding");
        }
        Integer previousArena = dynamicArenaIds.get(id);
        UUID previousUuid = dynamicArenaUuids.get(arenaId);
        if ((previousArena != null && previousArena != arenaId)
                || (previousUuid != null && !previousUuid.equals(id))) {
            throw new KernelShadowPolicyViolation(label + " dynamic binding is not one-to-one");
        }
        dynamicArenaIds.put(id, arenaId);
        dynamicArenaUuids.put(arenaId, id);
    }

    private static StableCombatRef stableCombatRef(JsonObject stable, String label) {
        if (stable == null) {
            throw new KernelShadowPolicyViolation(label + " lacks a stable reference");
        }
        int arenaId = requiredUnsignedInt(stable, "arena_id", label);
        int cardDbId = requiredUnsignedInt(stable, "card_db_id", label);
        String owner = requiredStringValue(stable, "owner", label);
        String controller = requiredStringValue(stable, "controller", label);
        requireSeatWire(owner, label + " owner");
        requireSeatWire(controller, label + " controller");
        requireString(stable, "zone", "Battlefield", label);
        requiredUnsignedInt(stable, "zone_change_count", label);
        return new StableCombatRef(arenaId, cardDbId, owner, controller);
    }

    private static void requireSeatWire(String seat, String label) {
        if (!"p0".equals(seat) && !"p1".equals(seat)) {
            throw new KernelShadowPolicyViolation(label + " is not p0 or p1");
        }
    }

    private void traceCombatSelection(
            XMageRallyBridgeProtocol.DecisionBody decision,
            JsonObject stable,
            Permanent permanent,
            int selectedIndex,
            Permanent fixedAttacker) {
        if (!Boolean.getBoolean("xmage.rally.traceActions")) {
            return;
        }
        System.err.println("XMAGE_RALLY_COMBAT_TRACE episode=" + episodeId
                + " step=" + decision.getStep()
                + " kind=" + decision.getDecisionKind()
                + " rust_arena_id=" + requiredUnsignedInt(
                stable, "arena_id", "combat trace")
                + " xmage=" + permanent.getName()
                + " xmage_uuid=" + permanent.getId()
                + " fixed_attacker=" + (fixedAttacker == null
                ? "none" : fixedAttacker.getId())
                + " include=" + (selectedIndex == 1));
    }

    private void validateActiveBindingAtConstruction() {
        if (!bridge.isUsable()) {
            throw constructionFailure("bridge is not usable");
        }
        Long activeEpisode = bridge.getActiveEpisodeId();
        if (activeEpisode == null || activeEpisode.longValue() != episodeId) {
            throw constructionFailure("bridge active episode does not match policy episode");
        }
        XMageRallyBridgeProtocol.DecisionBody current = bridge.getCurrentDecision();
        if (current == null) {
            if (bridge.getTerminal() != null) {
                throw constructionFailure("kernel was terminal before XMage policy binding");
            }
            throw constructionFailure("bridge has neither a current decision nor terminal");
        }
        if (current.getEpisodeId() != episodeId) {
            throw constructionFailure("current decision episode does not match policy episode");
        }
        boolean candidateIsPhysical = current.getCandidateSeat() == physicalSeat;
        if (candidateIsPhysical != modelControlled) {
            throw constructionFailure("physical seat has the wrong model/delegate role");
        }
    }

    private void validateSurfaceDecision(
            XMageRallyBridgeProtocol.DecisionBody decision,
            String category,
            int canonicalLegalCount,
            Game game) {
        validateCommonDecision(decision, game, category);
        if (canonicalLegalCount <= 0) {
            throw fail("canonicalLegalCount must be positive", null);
        }
        if (!"surface".equals(decision.getDecisionKind())
                || decision.getSubstepIndex() != 0
                || decision.getSubstepCount() != 1) {
            throw fail(category + " expected a single surface substep", null);
        }
        if (decision.getLegalActionCount() != canonicalLegalCount) {
            throw fail(category + " legal width mismatch: XMage=" + canonicalLegalCount
                    + " kernel=" + decision.getLegalActionCount(), null);
        }
        validateDecisionWidth(decision);
        try {
            for (XMageRallyBridgeProtocol.ActionSemantic semantic
                    : decision.getActionSemantics()) {
                String kind = semantic.getActionKind();
                if (!isSurfaceSemanticAllowed(category, kind)) {
                    throw new KernelShadowPolicyViolation(
                            category + " rejects kernel semantic kind " + kind);
                }
                validateSemanticActor(semantic, physicalSeat.wire());
            }
            validateCategorySpecificOrder(category, decision.getActionSemantics());
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }
    }

    private BinaryShape validateCombatDecision(
            XMageRallyBridgeProtocol.DecisionBody decision,
            String category,
            String decisionKind,
            String semanticKind,
            int candidateCount,
            int substep,
            Long physicalDecisionId,
            Game game) {
        validateCommonDecision(decision, game, category);
        if (!decisionKind.equals(decision.getDecisionKind())) {
            throw fail(category + " expected decision kind " + decisionKind
                    + " but got " + decision.getDecisionKind(), null);
        }
        if (decision.getSubstepIndex() != substep
                || decision.getSubstepCount() != candidateCount
                || decision.getLegalActionCount() != 2) {
            throw fail(category + " aggregate substep shape mismatch at " + substep
                    + ": xmage_candidates=" + candidateCount
                    + " kernel_substep=" + decision.getSubstepIndex()
                    + "/" + decision.getSubstepCount()
                    + " kernel_legal=" + decision.getLegalActionCount()
                    + " semantics=" + rustCombatSummary(decision.getActionSemantics()), null);
        }
        if (substep == 0 && physicalDecisionId != null) {
            throw fail(category + " received an invalid initial physical binding", null);
        }
        if (substep > 0 && (physicalDecisionId == null
                || decision.getPhysicalDecisionId() != physicalDecisionId.longValue())) {
            throw fail(category + " crossed a physical decision boundary", null);
        }
        try {
            return validateBinarySemanticOrder(decision, semanticKind);
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
        }
    }

    private void validateCommonDecision(
            XMageRallyBridgeProtocol.DecisionBody decision,
            Game game,
            String label) {
        requireClockMatch(decision, game, label);
        validateCommonDecisionBinding(decision, label);
    }

    private void validateCommonDecisionBinding(
            XMageRallyBridgeProtocol.DecisionBody decision,
            String label) {
        if (decision == null) {
            throw fail("current kernel decision is null", null);
        }
        if (decision.getEpisodeId() != episodeId) {
            throw fail("kernel decision episode mismatch", null);
        }
        if (decision.getActingPlayer() != physicalSeat) {
            throw fail("kernel acting seat " + decision.getActingPlayer().wire()
                    + " does not match XMage seat " + physicalSeat.wire()
                    + ": step=" + decision.getStep()
                    + " kind=" + decision.getDecisionKind()
                    + " substep=" + decision.getSubstepIndex()
                    + "/" + decision.getSubstepCount()
                    + " semantics=" + rustCombatSummary(
                    decision.getActionSemantics()), null);
        }
        if (decision.getCandidateSeat() == physicalSeat != modelControlled
                || decision.isCandidateControlsCurrentActor() != modelControlled) {
            throw fail("kernel decision has the wrong model/delegate role", null);
        }
        if (modelControlled) {
            if (decision.getSelectedActionIndex() == null
                    || decision.getCandidateActionSeedU64Hex() == null) {
                throw fail("model-controlled decision lacks Rust-side selection", null);
            }
        } else if (decision.getSelectedActionIndex() != null
                || decision.getCandidateActionSeedU64Hex() != null) {
            throw fail("delegate-controlled decision contains a model selection", null);
        }
        validateDecisionWidth(decision);
    }

    private void validateDecisionWidth(XMageRallyBridgeProtocol.DecisionBody decision) {
        int width = decision.getLegalActionCount();
        if (width <= 0
                || decision.getActionSemantics() == null
                || decision.getLogitsF32Bits() == null
                || decision.getActionSemantics().size() != width
                || decision.getLogitsF32Bits().size() != width) {
            throw fail("kernel decision width or score row is invalid", null);
        }
    }

    private int requireModelSelection(XMageRallyBridgeProtocol.DecisionBody decision) {
        if (!modelControlled || !decision.isCandidateControlsCurrentActor()) {
            throw fail("attempted to read a model selection for a delegate seat", null);
        }
        Integer selected = decision.getSelectedActionIndex();
        if (selected == null) {
            throw fail("Rust omitted the candidate model selection", null);
        }
        validateSelectedIndex(selected, decision.getLegalActionCount());
        return selected;
    }

    private void stepExactlyOnce(
            XMageRallyBridgeProtocol.DecisionBody decision,
            int selectedIndex,
            Game game,
            String label) {
        XMageRallyBridgeProtocol.ExpectedClock expectedClock =
                requireClockMatch(decision, game, label + ":step");
        String requestId = nextRequestId();
        try {
            XMageRallyBridgeProtocol.Response response = bridge.step(
                    requestId, episodeId, decision.getStep(), selectedIndex,
                    expectedClock);
            XMageRallyBridgeProtocol.AppliedAction applied;
            if (response.getBody() instanceof XMageRallyBridgeProtocol.DecisionResponseBody) {
                applied = ((XMageRallyBridgeProtocol.DecisionResponseBody) response.getBody())
                        .getAppliedAction();
            } else if (response.getBody()
                    instanceof XMageRallyBridgeProtocol.TerminalResponseBody) {
                applied = ((XMageRallyBridgeProtocol.TerminalResponseBody) response.getBody())
                        .getAppliedAction();
            } else {
                throw fail("bridge returned a non-success body from step", null);
            }
            if (applied == null) {
                throw fail("bridge step response omitted applied_action", null);
            }
            if (Boolean.getBoolean("xmage.rally.traceActions")) {
                System.err.println("XMAGE_RALLY_ACTION_TRACE episode=" + episodeId
                        + " step=" + decision.getStep()
                        + " semantic=" + applied.getSemantic().getCanonicalJson());
            }
        } catch (XMageRallyBridgeProcessClient.BridgeFailure error) {
            throw fail(bridgeStepFailureMessage(
                    episodeId, decision.getStep(), error.getMessage()), error);
        }
    }

    private static String bridgeStepFailureMessage(
            long episodeId, long step, String bridgeMessage) {
        return "KERNEL_SHADOW_POLICY_BRIDGE_STEP_FAILURE"
                + " episode=" + episodeId
                + " step=" + step
                + " " + (bridgeMessage == null ? "bridge_error=unknown" : bridgeMessage);
    }

    private XMageRallyBridgeProtocol.DecisionBody requireCurrentDecision(
            Game game, String label) {
        if (bridge == null) {
            throw fail("deserialized live policy has no bridge", null);
        }
        XMageRallyBridgeProtocol.DecisionBody decision = bridge.getCurrentDecision();
        if (decision != null) {
            requireClockMatch(decision, game, label);
            return decision;
        }
        if (bridge.getTerminal() != null) {
            throw fail("kernel reached terminal before XMage requested decision", null);
        }
        throw fail("bridge has no current decision", null);
    }

    private XMageRallyBridgeProtocol.ExpectedClock requireClockMatch(
            XMageRallyBridgeProtocol.DecisionBody decision,
            Game game,
            String label) {
        try {
            return XMageRallyClockComparator.requireMatch(
                    decision, game, "kernel_shadow_policy", label);
        } catch (XMageRallyClockComparator.ClockMismatch mismatch) {
            throw fail("KERNEL_SHADOW_POLICY_CLOCK_MISMATCH "
                    + mismatch.getMessage(), mismatch);
        }
    }

    private void recordSurfaceOutcome(String category, int legalCount, int selected) {
        ensureCounterCapacity(1);
        physicalDecisionCount++;
        policyActionSelections++;
        policyLeafEvaluations++;
        increment(physicalDecisionCategories, category);
        increment(outcomeHistogram, category + "|legal=" + legalCount
                + "|selected=" + selected);
    }

    private void ensureCounterCapacity(int addedSelections) {
        if (physicalDecisionCount == Long.MAX_VALUE
                || policyActionSelections > Long.MAX_VALUE - addedSelections
                || policyLeafEvaluations > Long.MAX_VALUE - addedSelections) {
            throw fail("policy accounting counter exhausted", null);
        }
    }

    private String nextRequestId() {
        if (requestOrdinal == Long.MAX_VALUE) {
            throw fail("bridge request ordinal exhausted", null);
        }
        String id = "xmage-shadow-" + episodeId + "-"
                + physicalSeat.wire() + "-" + requestOrdinal;
        requestOrdinal++;
        return id;
    }

    private void requireLive() {
        if (failed) {
            throw new KernelShadowPolicyViolation(
                    "kernel shadow policy is failed closed after: " + firstFailure);
        }
        if (bridge == null) {
            throw fail("live bridge is unavailable", null);
        }
        if (!bridge.isUsable()) {
            throw fail("bridge is no longer usable", null);
        }
    }

    private KernelShadowPolicyViolation fail(String message, Throwable cause) {
        if (!failed) {
            failed = true;
            firstFailure = message;
            if (bridge != null) {
                bridge.close();
            }
        }
        return cause == null
                ? new KernelShadowPolicyViolation(message)
                : new KernelShadowPolicyViolation(message, cause);
    }

    private KernelShadowPolicyViolation constructionFailure(String message) {
        bridge.close();
        return new KernelShadowPolicyViolation(message);
    }

    private void validateSelectedIndex(int selected, int width) {
        if (selected < 0 || selected >= width) {
            throw fail("selected index " + selected + " is outside width " + width, null);
        }
    }

    private static BinaryShape validateBinarySemanticOrder(
            XMageRallyBridgeProtocol.DecisionBody decision,
            String expectedKind) {
        List<XMageRallyBridgeProtocol.ActionSemantic> semantics =
                decision.getActionSemantics();
        if (semantics == null || semantics.size() != 2) {
            throw new KernelShadowPolicyViolation(
                    expectedKind + " requires exactly two semantics");
        }
        JsonObject no = parseSemantic(semantics.get(0), expectedKind);
        JsonObject yes = parseSemantic(semantics.get(1), expectedKind);
        requireBoolean(no, "include", false, expectedKind + " row zero");
        requireBoolean(yes, "include", true, expectedKind + " row one");
        requireString(no, "actor", decision.getActingPlayer().wire(), expectedKind);
        requireString(yes, "actor", decision.getActingPlayer().wire(), expectedKind);

        JsonObject noAttacker = requiredObject(no, "attacker", expectedKind);
        JsonObject yesAttacker = requiredObject(yes, "attacker", expectedKind);
        if (!noAttacker.equals(yesAttacker)) {
            throw new KernelShadowPolicyViolation(
                    expectedKind + " binary rows bind different attackers");
        }
        JsonObject blocker = null;
        if ("choose_blocker_inclusion".equals(expectedKind)) {
            JsonObject noBlocker = requiredObject(no, "blocker", expectedKind);
            JsonObject yesBlocker = requiredObject(yes, "blocker", expectedKind);
            if (!noBlocker.equals(yesBlocker)) {
                throw new KernelShadowPolicyViolation(
                        expectedKind + " binary rows bind different blockers");
            }
            blocker = noBlocker;
        }
        return new BinaryShape(noAttacker, blocker);
    }

    private static void validateCategorySpecificOrder(
            String category,
            List<XMageRallyBridgeProtocol.ActionSemantic> semantics) {
        if (semantics.size() == 2 && "chain_lightning_copy".equals(category)) {
            validateBooleanRows(semantics, "choose_spell_copy_payment", "pay", true, false);
        } else if (semantics.size() == 2
                && "chain_lightning_copy_retarget".equals(category)) {
            validateBooleanRows(
                    semantics, "choose_spell_copy_retarget", "change_target", true, false);
        } else if (semantics.size() == 2 && "choose_use".equals(category)) {
            String kind = semantics.get(0).getActionKind();
            if (!kind.equals(semantics.get(1).getActionKind())) {
                throw new KernelShadowPolicyViolation(
                        "choose_use binary rows have different semantic kinds");
            }
            if ("choose_kicker".equals(kind)) {
                validateBooleanRows(semantics, kind, "pay", false, true);
            } else if ("choose_effect_boolean".equals(kind)) {
                validateBooleanRows(semantics, kind, "value", false, true);
            } else if ("choose_optional_cost_use".equals(kind)) {
                validateBooleanRows(semantics, kind, "use_cost", false, true);
            } else if ("choose_madness_cast".equals(kind)) {
                validateBooleanRows(semantics, kind, "cast_it", false, true);
            }
        } else if (semantics.size() == 1 && "choose_use_forced_false".equals(category)) {
            JsonObject row = parseSemantic(semantics.get(0), semantics.get(0).getActionKind());
            boolean foundFalse = hasBoolean(row, "pay", false)
                    || hasBoolean(row, "value", false)
                    || hasBoolean(row, "use_cost", false)
                    || hasBoolean(row, "cast_it", false);
            if (!foundFalse) {
                throw new KernelShadowPolicyViolation(
                        "forced-false callback did not expose a false semantic");
            }
        }
    }

    private static void validateBooleanRows(
            List<XMageRallyBridgeProtocol.ActionSemantic> semantics,
            String kind,
            String field,
            boolean first,
            boolean second) {
        JsonObject firstRow = parseSemantic(semantics.get(0), kind);
        JsonObject secondRow = parseSemantic(semantics.get(1), kind);
        requireBoolean(firstRow, field, first, kind + " row zero");
        requireBoolean(secondRow, field, second, kind + " row one");
    }

    private static void validateSemanticActor(
            XMageRallyBridgeProtocol.ActionSemantic semantic,
            String expectedSeat) {
        JsonObject object = parseSemantic(semantic, semantic.getActionKind());
        requireString(object, "actor", expectedSeat, semantic.getActionKind());
    }

    private static JsonObject parseSemantic(
            XMageRallyBridgeProtocol.ActionSemantic semantic,
            String expectedKind) {
        if (semantic == null || !expectedKind.equals(semantic.getActionKind())) {
            throw new KernelShadowPolicyViolation(
                    "semantic kind mismatch; expected " + expectedKind);
        }
        JsonElement parsed;
        try {
            parsed = JsonParser.parseString(semantic.getCanonicalJson());
        } catch (RuntimeException error) {
            throw new KernelShadowPolicyViolation("semantic JSON is invalid", error);
        }
        if (!parsed.isJsonObject()) {
            throw new KernelShadowPolicyViolation("semantic must be a JSON object");
        }
        JsonObject object = parsed.getAsJsonObject();
        requireString(object, "action_kind", expectedKind, expectedKind);
        return object;
    }

    private static void requireString(
            JsonObject object, String field, String expected, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isString()
                || !expected.equals(value.getAsString())) {
            throw new KernelShadowPolicyViolation(
                    label + " has an invalid " + field + " field");
        }
    }

    private static void requireBoolean(
            JsonObject object, String field, boolean expected, String label) {
        if (!hasBoolean(object, field, expected)) {
            throw new KernelShadowPolicyViolation(
                    label + " has an invalid " + field + " field");
        }
    }

    private static boolean hasBoolean(JsonObject object, String field, boolean expected) {
        JsonElement value = object.get(field);
        return value != null
                && value.isJsonPrimitive()
                && value.getAsJsonPrimitive().isBoolean()
                && value.getAsBoolean() == expected;
    }

    private static boolean isSurfaceSemanticAllowed(String category, String kind) {
        if ("noncombat_activate_ability_or_spell".equals(category)) {
            return PRIORITY_KINDS.contains(kind);
        }
        if ("noncombat_activate_mana_ability".equals(category)) {
            return "activate_mana_ability".equals(kind);
        }
        if ("noncombat_activate_loyalty_ability".equals(category)) {
            return "activate_ability".equals(kind);
        }
        if ("noncombat_cast_spell".equals(category)) {
            return "cast_spell".equals(kind) || "plot_spell".equals(kind);
        }
        if ("target".equals(category)
                || "target_amount".equals(category)
                || "attack_defender".equals(category)
                || "noncombat_select_targets".equals(category)
                || "noncombat_declare_attack_target".equals(category)) {
            return TARGET_KINDS.contains(kind);
        }
        if ("card_target".equals(category)
                || "noncombat_select_card".equals(category)) {
            return CARD_TARGET_KINDS.contains(kind);
        }
        if ("choice".equals(category)
                || "choice_key".equals(category)
                || "choose_pile".equals(category)
                || "replacement_effect".equals(category)
                || "noncombat_select_choice".equals(category)) {
            return CHOICE_KINDS.contains(kind);
        }
        if ("mode".equals(category) || "noncombat_choose_mode".equals(category)) {
            return MODE_KINDS.contains(kind);
        }
        if ("announce_x".equals(category) || "noncombat_announce_x".equals(category)) {
            return "choose_effect_number".equals(kind);
        }
        if ("choose_use".equals(category)
                || "choose_use_forced_false".equals(category)
                || "noncombat_choose_use".equals(category)) {
            return USE_KINDS.contains(kind);
        }
        if ("chain_lightning_copy".equals(category)) {
            return "choose_spell_copy_payment".equals(kind);
        }
        if ("chain_lightning_copy_retarget".equals(category)) {
            return "choose_spell_copy_retarget".equals(kind);
        }
        if ("order_triggers".equals(category)
                || "noncombat_select_triggered_ability".equals(category)) {
            return "order_triggers".equals(kind);
        }
        if ("noncombat_london_mulligan".equals(category)) {
            return "discard".equals(kind) || CARD_TARGET_KINDS.contains(kind);
        }
        return false;
    }

    private static Map<UUID, Integer> validateInitialArenaIds(
            Map<UUID, Integer> bindings) {
        if (bindings == null) {
            throw new IllegalArgumentException("initialArenaIds must not be null");
        }
        if (bindings.isEmpty()) {
            return Collections.emptyMap();
        }
        if (bindings.size() != 120) {
            throw new IllegalArgumentException(
                    "initialArenaIds must bind all 120 Rally cards");
        }
        Map<UUID, Integer> copy = new LinkedHashMap<>();
        Set<Integer> seen = new HashSet<>();
        for (Map.Entry<UUID, Integer> entry : bindings.entrySet()) {
            UUID id = entry.getKey();
            Integer arenaId = entry.getValue();
            if (id == null || arenaId == null || arenaId < 0 || arenaId >= 120
                    || !seen.add(arenaId)) {
                throw new IllegalArgumentException(
                        "initialArenaIds is not a bijection onto 0..119");
            }
            copy.put(id, arenaId);
        }
        if (seen.size() != 120) {
            throw new IllegalArgumentException(
                    "initialArenaIds does not cover 0..119");
        }
        return Collections.unmodifiableMap(copy);
    }

    private Map<UUID, Integer> allArenaIds() {
        if (dynamicArenaIds.isEmpty()) {
            return initialArenaIds;
        }
        Map<UUID, Integer> combined = new LinkedHashMap<>(initialArenaIds);
        combined.putAll(dynamicArenaIds);
        return combined;
    }

    private Map<String, UUID> targetPlayerIds(Game game) {
        UUID startingPlayer = game.getStartingPlayerId();
        if (startingPlayer == null || !game.getPlayers().containsKey(startingPlayer)) {
            throw new KernelShadowPolicyViolation(
                    "target mapping requires the live starting player");
        }
        UUID otherPlayer = null;
        for (UUID playerId : game.getPlayers().keySet()) {
            if (playerId == null || playerId.equals(startingPlayer)) {
                continue;
            }
            if (otherPlayer != null) {
                throw new KernelShadowPolicyViolation(
                        "target mapping requires exactly two players");
            }
            otherPlayer = playerId;
        }
        if (otherPlayer == null) {
            throw new KernelShadowPolicyViolation(
                    "target mapping requires exactly two players");
        }
        Map<String, UUID> result = new LinkedHashMap<>();
        result.put("p0", startingPlayer);
        result.put("p1", otherPlayer);
        return result;
    }

    /** Bulk-bind every unbound generated token visible in one target menu. */
    private void bindTargetTokenObjects(
            List<UUID> xmageIds,
            List<XMageRallyBridgeProtocol.ActionSemantic> rustRows,
            Game game) {
        Map<String, Map<Integer, StableCombatRef>> rustByIdentity = new LinkedHashMap<>();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : rustRows) {
            TargetSemanticRef target = targetSemanticRef(semantic);
            if (!"object".equals(target.kind) || target.arenaId < 120
                    || dynamicArenaUuids.containsKey(target.arenaId)) {
                continue;
            }
            StableCombatRef stable = stableCombatRef(
                    target.stable, "target token object");
            String identity = stable.controller + "|" + stable.cardDbId;
            StableCombatRef previous = rustByIdentity
                    .computeIfAbsent(identity, ignored -> new LinkedHashMap<>())
                    .put(stable.arenaId, stable);
            if (previous != null) {
                throw new KernelShadowPolicyViolation(
                        "target token arena id is repeated");
            }
        }

        for (Map.Entry<String, Map<Integer, StableCombatRef>> group
                : rustByIdentity.entrySet()) {
            List<StableCombatRef> rustTargets = new ArrayList<>(group.getValue().values());
            rustTargets.sort(Comparator.comparingInt(target -> target.arenaId));

            List<Permanent> xmageTargets = new ArrayList<>();
            Set<UUID> seen = new HashSet<>();
            for (UUID id : xmageIds) {
                if (id == null || !seen.add(id) || initialArenaIds.containsKey(id)
                        || dynamicArenaIds.containsKey(id)) {
                    continue;
                }
                Permanent permanent = game.getPermanent(id);
                Integer cardDbId = permanent == null
                        ? null : RALLY_TOKEN_CARD_IDS.get(permanent.getName());
                if (permanent != null && cardDbId != null
                        && group.getKey().equals(
                        seatFor(permanent.getControllerId(), game) + "|" + cardDbId)) {
                    xmageTargets.add(permanent);
                }
            }
            xmageTargets.sort(Comparator.comparingInt(permanent ->
                    battlefieldIndex(permanent.getId(), game)));
            if (xmageTargets.size() != rustTargets.size()) {
                throw new KernelShadowPolicyViolation(
                        "target token identity count differs between XMage and Rust");
            }
            for (int i = 0; i < rustTargets.size(); i++) {
                putDynamicArenaBinding(
                        xmageTargets.get(i).getId(), rustTargets.get(i).arenaId,
                        "target token object");
            }
        }
    }

    /**
     * Bind generated Rally tokens from the only stable correspondence shared
     * by both engines: token definition, controller seat, creation order, and
     * the Rust arena id. Original deck cards remain fixed by the opening
     * 0..119 bijection.
     */
    private void bindPriorityTokenSources(
            List<? extends ActivatedAbility> xmageAbilities,
            List<XMageRallyBridgeProtocol.ActionSemantic> rustRows,
            Game game) {
        if (game == null || rustRows == null || xmageAbilities == null) {
            throw new KernelShadowPolicyViolation(
                    "priority token binding requires live XMage state");
        }

        Map<String, Map<Integer, JsonObject>> rustByIdentity = new LinkedHashMap<>();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : rustRows) {
            if (semantic == null || "pass".equals(semantic.getActionKind())) {
                continue;
            }
            JsonObject row = parseSemantic(semantic, semantic.getActionKind());
            JsonObject source = requiredObject(row, "source", "priority source");
            int arenaId = requiredUnsignedInt(source, "arena_id", "priority source");
            if (arenaId < 120 || initialArenaIds.containsValue(arenaId)
                    || dynamicArenaUuids.containsKey(arenaId)) {
                continue;
            }
            int cardDbId = requiredUnsignedInt(source, "card_db_id", "priority source");
            String controller = requiredStringValue(source, "controller", "priority source");
            requireString(source, "zone", "Battlefield", "priority token source");
            String identity = controller + "|" + cardDbId;
            JsonObject previous = rustByIdentity
                    .computeIfAbsent(identity, ignored -> new LinkedHashMap<>())
                    .put(arenaId, source);
            if (previous != null && !previous.equals(source)) {
                throw new KernelShadowPolicyViolation(
                        "priority token arena id has conflicting semantics");
            }
        }

        for (Map.Entry<String, Map<Integer, JsonObject>> group : rustByIdentity.entrySet()) {
            List<JsonObject> rustSources = new ArrayList<>(group.getValue().values());
            rustSources.sort(Comparator.comparingInt(source ->
                    requiredUnsignedInt(source, "arena_id", "priority token source")));

            List<Permanent> xmageSources = new ArrayList<>();
            Set<UUID> seen = new HashSet<>();
            for (ActivatedAbility ability : xmageAbilities) {
                if (ability == null || ability instanceof PassAbility
                        || ability.getSourceId() == null || !seen.add(ability.getSourceId())
                        || initialArenaIds.containsKey(ability.getSourceId())
                        || dynamicArenaIds.containsKey(ability.getSourceId())) {
                    continue;
                }
                Permanent permanent = game.getPermanent(ability.getSourceId());
                Integer cardDbId = permanent == null
                        ? null : RALLY_TOKEN_CARD_IDS.get(permanent.getName());
                if (permanent == null || cardDbId == null) {
                    continue;
                }
                String identity = seatFor(permanent.getControllerId(), game)
                        + "|" + cardDbId;
                if (group.getKey().equals(identity)) {
                    xmageSources.add(permanent);
                }
            }
            xmageSources.sort(Comparator.comparingInt(permanent ->
                    battlefieldIndex(permanent.getId(), game)));
            if (xmageSources.size() != rustSources.size()) {
                continue;
            }
            for (int i = 0; i < xmageSources.size(); i++) {
                Permanent permanent = xmageSources.get(i);
                JsonObject rustSource = rustSources.get(i);
                int arenaId = requiredUnsignedInt(
                        rustSource, "arena_id", "priority token source");
                int cardDbId = requiredUnsignedInt(
                        rustSource, "card_db_id", "priority token source");
                Integer actualCardDbId = RALLY_TOKEN_CARD_IDS.get(permanent.getName());
                if (actualCardDbId == null || actualCardDbId != cardDbId) {
                    throw new KernelShadowPolicyViolation(
                            "priority token definition changed during binding");
                }
                UUID previousUuid = dynamicArenaUuids.put(arenaId, permanent.getId());
                Integer previousArena = dynamicArenaIds.put(permanent.getId(), arenaId);
                if ((previousUuid != null && !previousUuid.equals(permanent.getId()))
                        || (previousArena != null && previousArena != arenaId)) {
                    throw new KernelShadowPolicyViolation(
                            "priority token binding is not one-to-one");
                }
            }
        }
    }

    private String seatFor(UUID playerId, Game game) {
        if (playerId == null || game == null || game.getStartingPlayerId() == null) {
            return "none";
        }
        if (playerId.equals(game.getStartingPlayerId())) {
            return "p0";
        }
        if (game.getPlayers().containsKey(playerId)) {
            return "p1";
        }
        return "none";
    }

    private static int battlefieldIndex(UUID permanentId, Game game) {
        int index = 0;
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent != null && permanent.getId().equals(permanentId)) {
                return index;
            }
            index++;
        }
        throw new KernelShadowPolicyViolation(
                "priority token source is absent from the battlefield");
    }

    private static JsonObject requiredObject(
            JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonObject()) {
            throw new KernelShadowPolicyViolation(label + " lacks object field " + field);
        }
        return value.getAsJsonObject();
    }

    private static int requiredUnsignedInt(
            JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isNumber()) {
            throw new KernelShadowPolicyViolation(label + " lacks integer field " + field);
        }
        String wire = value.getAsString();
        if (!wire.matches("0|[1-9][0-9]*")) {
            throw new KernelShadowPolicyViolation(label + " has invalid integer field " + field);
        }
        try {
            long parsed = Long.parseLong(wire);
            if (parsed > Integer.MAX_VALUE) {
                throw new KernelShadowPolicyViolation(
                        label + " integer field exceeds Java range: " + field);
            }
            return (int) parsed;
        } catch (NumberFormatException error) {
            throw new KernelShadowPolicyViolation(
                    label + " has invalid integer field " + field, error);
        }
    }

    private static String requiredStringValue(
            JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || !value.isJsonPrimitive()
                || !value.getAsJsonPrimitive().isString()) {
            throw new KernelShadowPolicyViolation(label + " lacks string field " + field);
        }
        return value.getAsString();
    }

    private static Map<String, Integer> rallyTokenCardIds() {
        Map<String, Integer> ids = new LinkedHashMap<>();
        ids.put("Blood Token", 132);
        ids.put("Human Soldier Token", 133);
        ids.put("Samurai Token", 134);
        return Collections.unmodifiableMap(ids);
    }

    private static Map<String, Integer> rallyCardIds() {
        Map<String, Integer> ids = new LinkedHashMap<>();
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
        ids.putAll(RALLY_TOKEN_CARD_IDS);
        return Collections.unmodifiableMap(ids);
    }

    private String xmagePrioritySummary(
            List<? extends ActivatedAbility> abilities, Game game) {
        List<String> rows = new ArrayList<>(abilities.size());
        Map<UUID, Integer> bindings = allArenaIds();
        for (ActivatedAbility ability : abilities) {
            if (ability instanceof PassAbility) {
                rows.add("pass");
                continue;
            }
            UUID sourceId = ability == null ? null : ability.getSourceId();
            MageObject source = game == null || sourceId == null
                    ? null : game.getObject(sourceId);
            rows.add(priorityAbilityKind(ability) + "@"
                    + bindings.get(sourceId) + ":"
                    + (source == null ? "unknown" : source.getName()) + ":"
                    + (sourceId == null || game == null
                    ? "unknown" : String.valueOf(game.getState().getZone(sourceId))));
        }
        return rows.toString();
    }

    private static String rustPrioritySummary(
            List<XMageRallyBridgeProtocol.ActionSemantic> semantics) {
        List<String> rows = new ArrayList<>(semantics.size());
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : semantics) {
            if (semantic == null || "pass".equals(semantic.getActionKind())) {
                rows.add("pass");
                continue;
            }
            JsonObject source = requiredObject(
                    parseSemantic(semantic, semantic.getActionKind()),
                    "source", "priority source");
            rows.add(semantic.getActionKind() + "@"
                    + requiredUnsignedInt(source, "arena_id", "priority source")
                    + ":db" + requiredUnsignedInt(
                    source, "card_db_id", "priority source")
                    + ":" + requiredStringValue(source, "zone", "priority source"));
        }
        return rows.toString();
    }

    private static String rustCombatSummary(
            List<XMageRallyBridgeProtocol.ActionSemantic> semantics) {
        List<String> rows = new ArrayList<>();
        if (semantics != null) {
            for (XMageRallyBridgeProtocol.ActionSemantic semantic : semantics) {
                rows.add(semantic == null ? "null" : semantic.getCanonicalJson());
            }
        }
        return rows.toString();
    }

    private String boundSourceSummary(
            List<XMageRallyBridgeProtocol.ActionSemantic> semantics, Game game) {
        List<String> rows = new ArrayList<>();
        Map<UUID, Integer> bindings = allArenaIds();
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : semantics) {
            if (semantic == null || "pass".equals(semantic.getActionKind())) {
                continue;
            }
            JsonObject source = requiredObject(
                    parseSemantic(semantic, semantic.getActionKind()),
                    "source", "priority source");
            int arenaId = requiredUnsignedInt(source, "arena_id", "priority source");
            UUID sourceId = null;
            for (Map.Entry<UUID, Integer> binding : bindings.entrySet()) {
                if (binding.getValue() == arenaId) {
                    sourceId = binding.getKey();
                    break;
                }
            }
            MageObject object = sourceId == null || game == null
                    ? null : game.getObject(sourceId);
            rows.add(arenaId + ":"
                    + (object == null ? "missing" : object.getName()) + ":"
                    + (sourceId == null || game == null
                    ? "unknown" : String.valueOf(game.getState().getZone(sourceId))));
        }
        return rows.toString();
    }

    private String xmageLandSummary(Game game) {
        List<String> rows = new ArrayList<>();
        if (game == null) {
            return rows.toString();
        }
        Map<UUID, Integer> bindings = allArenaIds();
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent != null && permanent.isLand(game)) {
                rows.add(bindings.get(permanent.getId()) + ":"
                        + seatFor(permanent.getControllerId(), game) + ":"
                        + permanent.getName() + ":tapped=" + permanent.isTapped());
            }
        }
        return rows.toString();
    }

    private static String priorityAbilityKind(ActivatedAbility ability) {
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

    private static List<UUID> mapCardTargetRows(
            List<UUID> xmageIds,
            List<XMageRallyBridgeProtocol.ActionSemantic> rustRows,
            Map<UUID, Integer> arenaIds) {
        if (xmageIds == null || rustRows == null || arenaIds == null
                || xmageIds.size() != rustRows.size()) {
            throw new KernelShadowPolicyViolation(
                    "card-target semantic and XMage widths differ");
        }
        List<UUID> mapped = new ArrayList<>(rustRows.size());
        Set<UUID> used = new HashSet<>();
        boolean usedStop = false;
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : rustRows) {
            Integer arenaId = cardTargetArenaId(semantic);
            UUID unique = null;
            int matches = 0;
            for (UUID id : xmageIds) {
                boolean match = arenaId == null
                        ? id == null
                        : id != null && arenaId.equals(arenaIds.get(id));
                if (match) {
                    unique = id;
                    matches++;
                }
            }
            if (matches != 1 || (unique == null ? usedStop : used.contains(unique))) {
                throw new KernelShadowPolicyViolation(
                        "card-target row did not map to exactly one unused XMage card: kind="
                                + semantic.getActionKind() + " arena_id=" + arenaId
                                + " matches=" + matches);
            }
            if (unique == null) {
                usedStop = true;
            } else {
                used.add(unique);
            }
            mapped.add(unique);
        }
        if (used.size() + (usedStop ? 1 : 0) != xmageIds.size()) {
            throw new KernelShadowPolicyViolation(
                    "card-target mapping did not consume the complete XMage menu");
        }
        return mapped;
    }

    private static List<UUID> mapTargetRows(
            List<UUID> xmageIds,
            List<XMageRallyBridgeProtocol.ActionSemantic> rustRows,
            Map<UUID, Integer> arenaIds,
            Map<String, UUID> playerIds) {
        if (xmageIds == null || rustRows == null || arenaIds == null
                || playerIds == null || xmageIds.size() != rustRows.size()) {
            throw new KernelShadowPolicyViolation(
                    "target semantic and XMage widths differ");
        }
        List<UUID> mapped = new ArrayList<>(rustRows.size());
        Set<UUID> used = new HashSet<>();
        boolean usedStop = false;
        for (XMageRallyBridgeProtocol.ActionSemantic semantic : rustRows) {
            TargetSemanticRef target = targetSemanticRef(semantic);
            UUID unique = null;
            int matches = 0;
            for (UUID id : xmageIds) {
                boolean match;
                if ("stop".equals(target.kind)) {
                    match = id == null;
                } else if ("player".equals(target.kind)) {
                    match = id != null && id.equals(playerIds.get(target.player));
                } else {
                    match = id != null && target.arenaId.equals(arenaIds.get(id));
                }
                if (match) {
                    unique = id;
                    matches++;
                }
            }
            if (matches != 1 || (unique == null ? usedStop : used.contains(unique))) {
                throw new KernelShadowPolicyViolation(
                        "target row did not map to exactly one unused XMage identity: kind="
                                + target.kind + " player=" + target.player
                                + " arena_id=" + target.arenaId + " matches=" + matches);
            }
            if (unique == null) {
                usedStop = true;
            } else {
                used.add(unique);
            }
            mapped.add(unique);
        }
        if (used.size() + (usedStop ? 1 : 0) != xmageIds.size()) {
            throw new KernelShadowPolicyViolation(
                    "target mapping did not consume the complete XMage menu");
        }
        return mapped;
    }

    private static TargetSemanticRef targetSemanticRef(
            XMageRallyBridgeProtocol.ActionSemantic semantic) {
        if (semantic == null) {
            throw new KernelShadowPolicyViolation("target semantic row is null");
        }
        String kind = semantic.getActionKind();
        JsonObject row = parseSemantic(semantic, kind);
        if ("finish_target_selection".equals(kind)
                || "finish_effect_selection".equals(kind)) {
            return TargetSemanticRef.stop();
        }
        if (!"choose_target".equals(kind) && !"choose_effect_target".equals(kind)) {
            throw new KernelShadowPolicyViolation(
                    "unsupported target semantic kind " + kind);
        }
        JsonObject target = requiredObject(row, "target", kind);
        String targetKind = requiredStringValue(target, "target_kind", kind);
        if ("player".equals(targetKind)) {
            String player = requiredStringValue(target, "player", kind);
            requireSeatWire(player, kind + " player target");
            return TargetSemanticRef.player(player);
        }
        if ("object".equals(targetKind)) {
            JsonObject stable = requiredObject(target, "object", kind);
            return TargetSemanticRef.object(
                    requiredUnsignedInt(stable, "arena_id", kind + " object"), stable);
        }
        throw new KernelShadowPolicyViolation(
                "unsupported target_kind " + targetKind);
    }

    private static Integer cardTargetArenaId(
            XMageRallyBridgeProtocol.ActionSemantic semantic) {
        String kind = semantic.getActionKind();
        JsonObject row = parseSemantic(semantic, kind);
        if ("finish_target_selection".equals(kind)
                || "finish_effect_selection".equals(kind)) {
            return null;
        }
        JsonObject stable;
        if ("discard".equals(kind)) {
            JsonElement cards = row.get("cards");
            if (cards == null || !cards.isJsonArray()
                    || cards.getAsJsonArray().size() != 1
                    || !cards.getAsJsonArray().get(0).isJsonObject()) {
                throw new KernelShadowPolicyViolation(
                        "discard card-target row must contain one stable card");
            }
            stable = cards.getAsJsonArray().get(0).getAsJsonObject();
        } else if ("choose_cost_target".equals(kind)) {
            stable = requiredObject(row, "candidate", kind);
        } else if ("choose_target".equals(kind)
                || "choose_effect_target".equals(kind)) {
            JsonObject target = requiredObject(row, "target", kind);
            requireString(target, "target_kind", "object", kind);
            stable = requiredObject(target, "object", kind);
        } else {
            throw new KernelShadowPolicyViolation(
                    "unsupported card-target semantic kind " + kind);
        }
        return requiredUnsignedInt(stable, "arena_id", kind + " card");
    }

    private static List<ActivatedAbility> mapPriorityRows(
            List<? extends ActivatedAbility> xmageAbilities,
            List<XMageRallyBridgeProtocol.ActionSemantic> rustRows,
            Map<UUID, Integer> initialArenaIds) {
        if (xmageAbilities == null || rustRows == null
                || xmageAbilities.size() != rustRows.size()) {
            throw new KernelShadowPolicyViolation(
                    "priority semantic and XMage widths differ");
        }
        List<ActivatedAbility> mapped = new ArrayList<>(rustRows.size());
        Set<ActivatedAbility> used = Collections.newSetFromMap(new IdentityHashMap<>());
        for (XMageRallyBridgeProtocol.ActionSemantic row : rustRows) {
            String kind = row.getActionKind();
            Integer arenaId = "pass".equals(kind) ? null : priorityArenaId(row);
            ActivatedAbility unique = null;
            int matches = 0;
            for (ActivatedAbility ability : xmageAbilities) {
                if (priorityAbilityMatches(ability, kind, arenaId, initialArenaIds)) {
                    unique = ability;
                    matches++;
                }
            }
            if (matches != 1 || unique == null || !used.add(unique)) {
                throw new KernelShadowPolicyViolation(
                        "priority row did not map to exactly one unused XMage ability: kind="
                                + kind + " arena_id=" + arenaId + " matches=" + matches);
            }
            mapped.add(unique);
        }
        if (used.size() != xmageAbilities.size()) {
            throw new KernelShadowPolicyViolation(
                    "priority mapping did not consume the complete XMage menu");
        }
        return mapped;
    }

    private static ActivatedAbility mapSelectedPriorityRow(
            List<? extends ActivatedAbility> xmageAbilities,
            XMageRallyBridgeProtocol.ActionSemantic rustRow,
            Map<UUID, Integer> arenaIds) {
        if (xmageAbilities == null || rustRow == null || arenaIds == null) {
            throw new KernelShadowPolicyViolation(
                    "selected priority mapping requires nonnull inputs");
        }
        String kind = rustRow.getActionKind();
        if (!PRIORITY_KINDS.contains(kind)) {
            throw new KernelShadowPolicyViolation(
                    "selected priority row has nonpriority kind " + kind);
        }
        Integer arenaId = "pass".equals(kind) ? null : priorityArenaId(rustRow);
        ActivatedAbility selected = null;
        int matches = 0;
        for (ActivatedAbility ability : xmageAbilities) {
            if (priorityAbilityMatches(ability, kind, arenaId, arenaIds)) {
                selected = ability;
                matches++;
            }
        }
        if (matches != 1 || selected == null) {
            throw new KernelShadowPolicyViolation(
                    "selected priority row did not map to exactly one XMage ability: kind="
                            + kind + " arena_id=" + arenaId + " matches=" + matches);
        }
        return selected;
    }

    private static boolean priorityAbilityMatches(
            ActivatedAbility ability,
            String kind,
            Integer arenaId,
            Map<UUID, Integer> initialArenaIds) {
        if (ability == null) {
            return false;
        }
        if ("pass".equals(kind)) {
            return ability instanceof PassAbility;
        }
        if (ability instanceof PassAbility || ability.getSourceId() == null
                || arenaId == null || !arenaId.equals(initialArenaIds.get(ability.getSourceId()))) {
            return false;
        }
        if ("play_land".equals(kind)) {
            return ability instanceof PlayLandAbility;
        }
        if ("cast_spell".equals(kind)) {
            return ability instanceof SpellAbility;
        }
        if ("activate_mana_ability".equals(kind)) {
            return ability instanceof ManaAbility;
        }
        return "activate_ability".equals(kind)
                && !(ability instanceof PlayLandAbility)
                && !(ability instanceof SpellAbility)
                && !(ability instanceof ManaAbility);
    }

    private static int priorityArenaId(
            XMageRallyBridgeProtocol.ActionSemantic semantic) {
        JsonObject row = parseSemantic(semantic, semantic.getActionKind());
        JsonElement sourceElement = row.get("source");
        if (sourceElement == null || !sourceElement.isJsonObject()) {
            throw new KernelShadowPolicyViolation(
                    semantic.getActionKind() + " priority semantic lacks source");
        }
        JsonElement arenaElement = sourceElement.getAsJsonObject().get("arena_id");
        if (arenaElement == null || !arenaElement.isJsonPrimitive()
                || !arenaElement.getAsJsonPrimitive().isNumber()) {
            throw new KernelShadowPolicyViolation(
                    semantic.getActionKind() + " priority source lacks arena_id");
        }
        String wire = arenaElement.getAsString();
        if (!wire.matches("0|[1-9][0-9]*")) {
            throw new KernelShadowPolicyViolation("priority arena_id is not an unsigned integer");
        }
        try {
            long value = Long.parseLong(wire);
            if (value > Integer.MAX_VALUE) {
                throw new KernelShadowPolicyViolation("priority arena_id exceeds Java range");
            }
            return (int) value;
        } catch (NumberFormatException error) {
            throw new KernelShadowPolicyViolation("priority arena_id is invalid", error);
        }
    }

    private String requireCategory(String category) {
        if (category == null || category.trim().isEmpty()) {
            throw fail("category must be nonempty", null);
        }
        return category.trim();
    }

    private static XMageRallyBridgeProtocol.Seat parseSeat(String seat) {
        if (XMageRallyBridgeProtocol.Seat.P0.wire().equals(seat)) {
            return XMageRallyBridgeProtocol.Seat.P0;
        }
        if (XMageRallyBridgeProtocol.Seat.P1.wire().equals(seat)) {
            return XMageRallyBridgeProtocol.Seat.P1;
        }
        throw new IllegalArgumentException("physicalSeat must be exactly p0 or p1");
    }

    private static Set<String> kinds(String... values) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(values)));
    }

    private static void increment(Map<String, Long> map, String key) {
        Long previous = map.get(key);
        if (previous != null && previous == Long.MAX_VALUE) {
            throw new KernelShadowPolicyViolation("histogram counter exhausted for " + key);
        }
        map.put(key, previous == null ? 1L : previous + 1L);
    }

    /** Pure protocol-shape checks runnable without launching XMage or Rust. */
    public static void runFocusedSelfTest() {
        XMageRallyBridgeProtocol.ActionSemantic attackNo =
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "choose_attacker_inclusion",
                        "{\"action_kind\":\"choose_attacker_inclusion\","
                                + "\"actor\":\"p0\",\"attacker\":{\"arena_id\":1},"
                                + "\"include\":false}");
        XMageRallyBridgeProtocol.ActionSemantic attackYes =
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "choose_attacker_inclusion",
                        "{\"action_kind\":\"choose_attacker_inclusion\","
                                + "\"actor\":\"p0\",\"attacker\":{\"arena_id\":1},"
                                + "\"include\":true}");
        List<XMageRallyBridgeProtocol.ActionSemantic> rows =
                new ArrayList<>(Arrays.asList(attackNo, attackYes));
        validateBooleanRows(rows, "choose_attacker_inclusion", "include", false, true);
        if (!isSurfaceSemanticAllowed("noncombat_activate_ability_or_spell", "pass")
                || isSurfaceSemanticAllowed("announce_x", "pass")) {
            throw new IllegalStateException("surface category semantic self-test failed");
        }
        assertPrioritySemanticMapping();
        assertCardTargetSemanticMapping();
        assertTargetSemanticMapping();
        assertCombatStableIdentityMapping();
        String scorerFailure = bridgeStepFailureMessage(
                2L, 3L, "XMAGE_RALLY_SCORER_ERROR error_code=clock_mismatch");
        if (!scorerFailure.contains("KERNEL_SHADOW_POLICY_BRIDGE_STEP_FAILURE")
                || !scorerFailure.contains(
                "XMAGE_RALLY_SCORER_ERROR error_code=clock_mismatch")) {
            throw new IllegalStateException(
                    "policy bridge failure marker self-test failed");
        }
    }

    private static void assertPrioritySemanticMapping() {
        UUID landId = UUID.randomUUID();
        UUID spellId = UUID.randomUUID();
        UUID manaId = UUID.randomUUID();
        UUID activatedId = UUID.randomUUID();

        PlayLandAbility land = new PlayLandAbility("Mountain");
        land.setSourceId(landId);
        SpellAbility spell = new SpellAbility(new ManaCostsImpl<>("{R}"), "Fixture");
        spell.setSourceId(spellId);
        RedManaAbility mana = new RedManaAbility();
        mana.setSourceId(manaId);
        SimpleActivatedAbility activated = new SimpleActivatedAbility(
                new DrawCardSourceControllerEffect(1), new ManaCostsImpl<>(""));
        activated.setSourceId(activatedId);
        PassAbility pass = new PassAbility();

        Map<UUID, Integer> bindings = new LinkedHashMap<>();
        bindings.put(landId, 3);
        bindings.put(spellId, 8);
        bindings.put(manaId, 17);
        bindings.put(activatedId, 23);
        List<ActivatedAbility> xmage = Arrays.asList(pass, activated, mana, land, spell);
        List<XMageRallyBridgeProtocol.ActionSemantic> rust = Arrays.asList(
                prioritySemantic("play_land", 3),
                prioritySemantic("cast_spell", 8),
                prioritySemantic("activate_mana_ability", 17),
                prioritySemantic("activate_ability", 23),
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "pass", "{\"action_kind\":\"pass\",\"actor\":\"p0\"}"));
        List<ActivatedAbility> mapped = mapPriorityRows(xmage, rust, bindings);
        if (mapped.size() != 5 || mapped.get(0) != land || mapped.get(1) != spell
                || mapped.get(2) != mana || mapped.get(3) != activated
                || mapped.get(4) != pass) {
            throw new IllegalStateException("priority semantic row mapping failed");
        }

        List<ActivatedAbility> singleton = mapPriorityRows(
                Collections.singletonList(pass),
                Collections.singletonList(rust.get(4)), bindings);
        if (singleton.size() != 1 || singleton.get(0) != pass) {
            throw new IllegalStateException("singleton Rust pass mapping failed");
        }

        ActivatedAbility selectedOnly = mapSelectedPriorityRow(
                Arrays.asList(pass, activated, mana, land, spell), rust.get(1), bindings);
        if (selectedOnly != spell) {
            throw new IllegalStateException("selected priority row mapping failed");
        }
        try {
            mapSelectedPriorityRow(
                    Collections.singletonList(pass), rust.get(1), bindings);
            throw new IllegalStateException(
                    "selected priority row mapping accepted an absent action");
        } catch (KernelShadowPolicyViolation expected) {
            // Expected fail-closed absence.
        }
        try {
            mapSelectedPriorityRow(
                    Arrays.asList(spell, spell, pass), rust.get(1), bindings);
            throw new IllegalStateException(
                    "selected priority row mapping accepted an ambiguous action");
        } catch (KernelShadowPolicyViolation expected) {
            // Expected fail-closed ambiguity.
        }
    }

    private static XMageRallyBridgeProtocol.ActionSemantic prioritySemantic(
            String kind, int arenaId) {
        return new XMageRallyBridgeProtocol.ActionSemantic(
                kind,
                "{\"action_kind\":\"" + kind + "\",\"actor\":\"p0\","
                        + "\"source\":{\"arena_id\":" + arenaId + "}}");
    }

    private static void assertCardTargetSemanticMapping() {
        UUID first = UUID.randomUUID();
        UUID second = UUID.randomUUID();
        Map<UUID, Integer> bindings = new LinkedHashMap<>();
        bindings.put(first, 3);
        bindings.put(second, 8);
        List<XMageRallyBridgeProtocol.ActionSemantic> rust = Arrays.asList(
                discardSemantic(3), discardSemantic(8));
        List<UUID> mapped = mapCardTargetRows(
                Arrays.asList(second, first), rust, bindings);
        if (!mapped.equals(Arrays.asList(first, second))) {
            throw new IllegalStateException("card-target semantic row mapping failed");
        }
    }

    private static XMageRallyBridgeProtocol.ActionSemantic discardSemantic(int arenaId) {
        return new XMageRallyBridgeProtocol.ActionSemantic(
                "discard",
                "{\"action_kind\":\"discard\",\"actor\":\"p0\","
                        + "\"cards\":[{\"arena_id\":" + arenaId + "}]}");
    }

    private static void assertTargetSemanticMapping() {
        UUID object = UUID.randomUUID();
        UUID p0 = UUID.randomUUID();
        UUID p1 = UUID.randomUUID();
        Map<UUID, Integer> bindings = new LinkedHashMap<>();
        bindings.put(object, 121);
        Map<String, UUID> players = new LinkedHashMap<>();
        players.put("p0", p0);
        players.put("p1", p1);
        List<XMageRallyBridgeProtocol.ActionSemantic> rust = Arrays.asList(
                targetPlayerSemantic("choose_target", "p1"),
                targetObjectSemantic("choose_target", 121),
                targetPlayerSemantic("choose_effect_target", "p0"),
                new XMageRallyBridgeProtocol.ActionSemantic(
                        "finish_effect_selection",
                        "{\"action_kind\":\"finish_effect_selection\","
                                + "\"actor\":\"p0\",\"selected_count\":1}"));
        List<UUID> mapped = mapTargetRows(
                Arrays.asList(object, null, p0, p1), rust, bindings, players);
        if (!mapped.equals(Arrays.asList(p1, object, p0, null))) {
            throw new IllegalStateException("target semantic row mapping failed");
        }
    }

    private static XMageRallyBridgeProtocol.ActionSemantic targetPlayerSemantic(
            String kind, String player) {
        return new XMageRallyBridgeProtocol.ActionSemantic(
                kind,
                "{\"action_kind\":\"" + kind + "\",\"actor\":\"p0\","
                        + "\"target\":{\"target_kind\":\"player\","
                        + "\"player\":\"" + player + "\"}}");
    }

    private static XMageRallyBridgeProtocol.ActionSemantic targetObjectSemantic(
            String kind, int arenaId) {
        return new XMageRallyBridgeProtocol.ActionSemantic(
                kind,
                "{\"action_kind\":\"" + kind + "\",\"actor\":\"p0\","
                        + "\"target\":{\"target_kind\":\"object\","
                        + "\"object\":{\"arena_id\":" + arenaId + "}}}");
    }

    private static void assertCombatStableIdentityMapping() {
        UUID battlefieldIndexTwo = UUID.randomUUID();
        UUID battlefieldIndexTen = UUID.randomUUID();
        Map<UUID, Integer> bindings = new LinkedHashMap<>();
        bindings.put(battlefieldIndexTwo, 2);
        bindings.put(battlefieldIndexTen, 10);
        JsonObject stable = JsonParser.parseString(
                "{\"arena_id\":10,\"card_db_id\":44,"
                        + "\"owner\":\"p0\",\"controller\":\"p0\","
                        + "\"zone\":\"Battlefield\",\"zone_change_count\":1}")
                .getAsJsonObject();
        StableCombatRef reference = stableCombatRef(stable, "combat self-test");
        UUID mapped = mapBoundCombatUuid(
                reference.arenaId,
                Arrays.asList(battlefieldIndexTen, battlefieldIndexTwo),
                bindings, Collections.emptySet(), "combat self-test");
        if (!battlefieldIndexTen.equals(mapped) || reference.cardDbId != 44) {
            throw new IllegalStateException(
                    "combat stable identity mapping used XMage candidate rank");
        }
        try {
            mapBoundCombatUuid(
                    reference.arenaId,
                    Arrays.asList(battlefieldIndexTwo, battlefieldIndexTen),
                    bindings, Collections.singleton(battlefieldIndexTen),
                    "combat self-test duplicate");
            throw new IllegalStateException(
                    "combat stable identity mapping accepted a repeated candidate");
        } catch (KernelShadowPolicyViolation expected) {
            // Expected fail-closed duplicate rejection.
        }
    }

    public static void main(String[] args) {
        runFocusedSelfTest();
        System.out.println("KernelShadowRallyPolicy focused self-test PASS");
    }

    private static final class BinaryShape {
        private final JsonObject attacker;
        private final JsonObject blocker;

        private BinaryShape(JsonObject attacker, JsonObject blocker) {
            this.attacker = attacker;
            this.blocker = blocker;
        }
    }

    private static final class StableCombatRef {
        private final int arenaId;
        private final int cardDbId;
        private final String owner;
        private final String controller;

        private StableCombatRef(
                int arenaId, int cardDbId, String owner, String controller) {
            this.arenaId = arenaId;
            this.cardDbId = cardDbId;
            this.owner = owner;
            this.controller = controller;
        }
    }

    public static final class BlockAssignment {
        private final UUID blockerId;
        private final UUID attackerId;

        private BlockAssignment(UUID blockerId, UUID attackerId) {
            this.blockerId = blockerId;
            this.attackerId = attackerId;
        }

        public UUID getBlockerId() {
            return blockerId;
        }

        public UUID getAttackerId() {
            return attackerId;
        }
    }

    public static final class KernelShadowPolicyViolation extends IllegalStateException {
        private static final long serialVersionUID = 1L;

        public KernelShadowPolicyViolation(String message) {
            super(message);
        }

        public KernelShadowPolicyViolation(String message, Throwable cause) {
            super(message, cause);
        }
    }

    private static final class TargetSemanticRef {
        private final String kind;
        private final String player;
        private final Integer arenaId;
        private final JsonObject stable;

        private TargetSemanticRef(
                String kind, String player, Integer arenaId, JsonObject stable) {
            this.kind = kind;
            this.player = player;
            this.arenaId = arenaId;
            this.stable = stable;
        }

        private static TargetSemanticRef stop() {
            return new TargetSemanticRef("stop", null, null, null);
        }

        private static TargetSemanticRef player(String player) {
            return new TargetSemanticRef("player", player, null, null);
        }

        private static TargetSemanticRef object(int arenaId, JsonObject stable) {
            return new TargetSemanticRef("object", null, arenaId, stable);
        }
    }
}
