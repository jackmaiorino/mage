package mage.player.ai.rl;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.abilities.ActivatedAbility;
import mage.abilities.PlayLandAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.SimpleActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.effects.common.DrawCardSourceControllerEffect;
import mage.abilities.mana.ManaAbility;
import mage.abilities.mana.RedManaAbility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

    private transient final XMageRallyBridgeProcessClient bridge;
    private final long episodeId;
    private final XMageRallyBridgeProtocol.Seat physicalSeat;
    private final boolean modelControlled;
    private final RallyCanonicalDecisionPolicy delegate;
    private final SeededUniformMirrorPolicy simulationPolicyTemplate;
    private final Map<UUID, Integer> initialArenaIds;

    private long requestOrdinal;
    private long physicalDecisionCount;
    private long policyActionSelections;
    private long policyLeafEvaluations;
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
        this(bridge, episodeId, parseSeat(physicalSeat), modelControlled,
                delegate, simulationPolicy, initialArenaIds);
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
        requireLive();
        String checkedCategory = requireCategory(category);
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision = requireCurrentDecision();
        validateSurfaceDecision(decision, checkedCategory, canonicalLegalCount);

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
        stepExactlyOnce(decision, selected);
        recordSurfaceOutcome(checkedCategory, canonicalLegalCount, selected);
        return selected;
    }

    /**
     * Select a live XMage priority ability by the Rust semantic row rather
     * than assuming the two engines use the same menu order.
     */
    public synchronized ActivatedAbility choosePriorityAbility(
            List<? extends ActivatedAbility> xmageAbilities) {
        requireLive();
        if (xmageAbilities == null || xmageAbilities.isEmpty()) {
            throw fail("priority ability menu must be nonempty", null);
        }
        String category = "noncombat_activate_ability_or_spell";
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision = requireCurrentDecision();
        validateSurfaceDecision(decision, category, xmageAbilities.size());

        List<ActivatedAbility> abilitiesByRustRow;
        try {
            abilitiesByRustRow = mapPriorityRows(
                    xmageAbilities, decision.getActionSemantics(), initialArenaIds);
        } catch (KernelShadowPolicyViolation error) {
            throw fail(error.getMessage(), error);
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
        stepExactlyOnce(decision, selected);
        recordSurfaceOutcome(category, abilitiesByRustRow.size(), selected);
        return result;
    }

    @Override
    public synchronized int[] chooseNoncombatWithoutReplacement(
            String category, int canonicalLegalCount, int picks) {
        requireLive();
        if (picks != 1) {
            throw fail("surface callback must contain exactly one Rust substep; picks=" + picks,
                    null);
        }
        String checkedCategory = requireCategory(category);
        ensureCounterCapacity(1);
        XMageRallyBridgeProtocol.DecisionBody decision = requireCurrentDecision();
        validateSurfaceDecision(decision, checkedCategory, canonicalLegalCount);

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
        stepExactlyOnce(decision, selected);
        recordSurfaceOutcome(checkedCategory, canonicalLegalCount, selected);
        return new int[]{selected};
    }

    @Override
    public synchronized boolean[] chooseAttackers(int canonicalEligibleCount) {
        return chooseCombatGroup(
                "declare_attackers", "attacker_inclusion",
                "choose_attacker_inclusion", canonicalEligibleCount, false);
    }

    @Override
    public synchronized boolean[] chooseBlockers(int canonicalLegalBlockerCount) {
        return chooseCombatGroup(
                "declare_blocker_for_attacker", "blocker_inclusion",
                "choose_blocker_inclusion", canonicalLegalBlockerCount, true);
    }

    @Override
    public synchronized int chooseBlocker(int canonicalLegalBlockerCount) {
        boolean[] included = chooseBlockers(canonicalLegalBlockerCount);
        int selected = -1;
        for (int i = 0; i < included.length; i++) {
            if (!included[i]) {
                continue;
            }
            if (selected >= 0) {
                throw fail("single-blocker callback received a multiple-blocker selection", null);
            }
            selected = i;
        }
        return selected;
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

    private boolean[] chooseCombatGroup(
            String category,
            String decisionKind,
            String semanticKind,
            int candidateCount,
            boolean blockerGroup) {
        requireLive();
        if (candidateCount <= 0) {
            throw fail(category + " candidate count must be positive", null);
        }
        ensureCounterCapacity(candidateCount);

        boolean[] delegated = null;
        XMageRallyBridgeProtocol.DecisionBody firstDecision = requireCurrentDecision();
        validateCombatDecision(
                firstDecision, category, decisionKind, semanticKind,
                candidateCount, 0, null);
        if (!modelControlled) {
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
        }

        boolean[] selected = new boolean[candidateCount];
        Long physicalDecisionId = null;
        String fixedAttacker = null;
        Set<String> candidateObjects = new HashSet<>();
        for (int substep = 0; substep < candidateCount; substep++) {
            XMageRallyBridgeProtocol.DecisionBody decision = substep == 0
                    ? firstDecision : requireCurrentDecision();
            BinaryShape shape = validateCombatDecision(
                    decision, category, decisionKind, semanticKind,
                    candidateCount, substep, physicalDecisionId);
            if (physicalDecisionId == null) {
                physicalDecisionId = decision.getPhysicalDecisionId();
            }
            if (blockerGroup) {
                if (shape.blocker == null) {
                    throw fail("blocker inclusion semantic lacks a blocker binding", null);
                }
                if (fixedAttacker == null) {
                    fixedAttacker = shape.attacker;
                } else if (!fixedAttacker.equals(shape.attacker)) {
                    throw fail("blocker aggregate spans multiple attackers", null);
                }
                if (!candidateObjects.add(shape.blocker)) {
                    throw fail("blocker aggregate repeats a blocker binding", null);
                }
            } else {
                if (!candidateObjects.add(shape.attacker)) {
                    throw fail("attacker aggregate repeats an attacker binding", null);
                }
            }

            int selectedIndex = modelControlled
                    ? requireModelSelection(decision)
                    : (delegated[substep] ? 1 : 0);
            if (selectedIndex != 0 && selectedIndex != 1) {
                throw fail(category + " selected a non-binary action index", null);
            }
            selected[substep] = selectedIndex == 1;
            stepExactlyOnce(decision, selectedIndex);
            policyActionSelections++;
            policyLeafEvaluations++;
        }

        int includedCount = 0;
        for (boolean include : selected) {
            if (include) {
                includedCount++;
            }
        }
        physicalDecisionCount++;
        increment(physicalDecisionCategories, category);
        increment(outcomeHistogram, category + "|legal=" + candidateCount
                + "|included=" + includedCount);
        return selected;
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
            int canonicalLegalCount) {
        validateCommonDecision(decision);
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
            Long physicalDecisionId) {
        validateCommonDecision(decision);
        if (!decisionKind.equals(decision.getDecisionKind())) {
            throw fail(category + " expected decision kind " + decisionKind
                    + " but got " + decision.getDecisionKind(), null);
        }
        if (decision.getSubstepIndex() != substep
                || decision.getSubstepCount() != candidateCount
                || decision.getLegalActionCount() != 2) {
            throw fail(category + " aggregate substep shape mismatch at " + substep, null);
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

    private void validateCommonDecision(XMageRallyBridgeProtocol.DecisionBody decision) {
        if (decision == null) {
            throw fail("current kernel decision is null", null);
        }
        if (decision.getEpisodeId() != episodeId) {
            throw fail("kernel decision episode mismatch", null);
        }
        if (decision.getActingPlayer() != physicalSeat) {
            throw fail("kernel acting seat " + decision.getActingPlayer().wire()
                    + " does not match XMage seat " + physicalSeat.wire(), null);
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
            int selectedIndex) {
        String requestId = nextRequestId();
        try {
            XMageRallyBridgeProtocol.Response response = bridge.step(
                    requestId, episodeId, decision.getStep(), selectedIndex);
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
        } catch (XMageRallyBridgeProcessClient.BridgeFailure error) {
            throw fail("bridge step failed for episode " + episodeId
                    + " step " + decision.getStep(), error);
        }
    }

    private XMageRallyBridgeProtocol.DecisionBody requireCurrentDecision() {
        if (bridge == null) {
            throw fail("deserialized live policy has no bridge", null);
        }
        XMageRallyBridgeProtocol.DecisionBody decision = bridge.getCurrentDecision();
        if (decision != null) {
            return decision;
        }
        if (bridge.getTerminal() != null) {
            throw fail("kernel reached terminal before XMage requested decision", null);
        }
        throw fail("bridge has no current decision", null);
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
        String id = "xmage-shadow-" + episodeId + "-" + requestOrdinal;
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

        String noAttacker = requiredCanonicalField(no, "attacker", expectedKind);
        String yesAttacker = requiredCanonicalField(yes, "attacker", expectedKind);
        if (!noAttacker.equals(yesAttacker)) {
            throw new KernelShadowPolicyViolation(
                    expectedKind + " binary rows bind different attackers");
        }
        String blocker = null;
        if ("choose_blocker_inclusion".equals(expectedKind)) {
            String noBlocker = requiredCanonicalField(no, "blocker", expectedKind);
            String yesBlocker = requiredCanonicalField(yes, "blocker", expectedKind);
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

    private static String requiredCanonicalField(
            JsonObject object, String field, String label) {
        JsonElement value = object.get(field);
        if (value == null || value.isJsonNull()) {
            throw new KernelShadowPolicyViolation(label + " lacks " + field);
        }
        return value.toString();
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
    }

    private static XMageRallyBridgeProtocol.ActionSemantic prioritySemantic(
            String kind, int arenaId) {
        return new XMageRallyBridgeProtocol.ActionSemantic(
                kind,
                "{\"action_kind\":\"" + kind + "\",\"actor\":\"p0\","
                        + "\"source\":{\"arena_id\":" + arenaId + "}}");
    }

    public static void main(String[] args) {
        runFocusedSelfTest();
        System.out.println("KernelShadowRallyPolicy focused self-test PASS");
    }

    private static final class BinaryShape {
        private final String attacker;
        private final String blocker;

        private BinaryShape(String attacker, String blocker) {
            this.attacker = attacker;
            this.blocker = blocker;
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
}
