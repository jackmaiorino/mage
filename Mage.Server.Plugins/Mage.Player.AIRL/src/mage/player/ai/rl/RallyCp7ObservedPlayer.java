package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.Mode;
import mage.abilities.Modes;
import mage.abilities.TriggeredAbility;
import mage.abilities.common.PassAbility;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.ComputerPlayer7;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.target.TargetCard;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/** A deterministic CP7 seat whose real Rally decisions are synchronously observed. */
public final class RallyCp7ObservedPlayer extends ComputerPlayer7 {

    private static final long serialVersionUID = 1L;

    private final transient RallyCp7DecisionObserver.DispatchState observerState;

    public RallyCp7ObservedPlayer(String name, RangeOfInfluence range, int skill,
                                  RallyCp7DecisionObserver observer) {
        super(name, range, skill);
        this.observerState = new RallyCp7DecisionObserver.DispatchState(observer);
    }

    private RallyCp7ObservedPlayer(RallyCp7ObservedPlayer player) {
        super(player);
        this.observerState = player.requireObserverState();
    }

    @Override
    public RallyCp7ObservedPlayer copy() {
        return new RallyCp7ObservedPlayer(this);
    }

    @Override
    public boolean chooseMulligan(Game game) {
        List<Card> candidates = new ArrayList<>(getHand().getCards(game));
        // The native Rally environment has no mulligan decision. Both seats
        // keep the exact seven cards produced by the bound library order.
        boolean result = false;
        emit(game, RallyCp7DecisionObserver.Kind.MULLIGAN, null, getHand(),
                "", candidates, Collections.singletonList(result), true);
        return result;
    }

    /**
     * CP7 can plan several activations at one priority point. Report each
     * chosen activation separately, immediately before XMage commits it.
     */
    @Override
    protected void act(Game game) {
        requireLiveGameOrSimulation(game);
        if (actions == null || actions.isEmpty()) {
            pass(game);
            return;
        }
        boolean usedStack = false;
        while (actions.peek() != null) {
            Ability ability = actions.poll();
            if (!(ability instanceof PassAbility)) {
                emit(game, RallyCp7DecisionObserver.Kind.PRIORITY_ACTION,
                        ability, ability, "", Collections.singletonList(ability),
                        Collections.singletonList(ability), true);
            }
            if (!ability.getTargets().isEmpty()) {
                for (Target target : ability.getTargets()) {
                    emit(game, RallyCp7DecisionObserver.Kind.PRESELECTED_TARGETS,
                            ability, target, target.getTargetName(), target.getTargets(),
                            target.getTargets(), true);
                    for (UUID id : target.getTargets()) {
                        target.updateTarget(id, game);
                        if (!target.isNotTarget()) {
                            game.addSimultaneousEvent(GameEvent.getEvent(
                                    GameEvent.EventType.TARGETED, id, ability,
                                    ability.getControllerId()));
                        }
                    }
                }
            }
            activateAbility((ActivatedAbility) ability, game);
            if (ability.isUsesStack()) {
                usedStack = true;
            }
        }
        if (usedStack) {
            pass(game);
        }
    }

    /** Observe both explicit CP7 passes and passes performed by PassAbility. */
    @Override
    public void pass(Game game) {
        requireLiveGameOrSimulation(game);
        emit(game, RallyCp7DecisionObserver.Kind.PRIORITY_PASS,
                null, null, "", Collections.emptyList(),
                Collections.emptyList(), true);
        super.pass(game);
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game,
                          Map<String, Serializable> options) {
        List<UUID> candidates = possibleTargets(target, source, game);
        boolean result = super.choose(outcome, target, source, game, options);
        emitTarget(game, source, target, candidates, result);
        return result;
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        List<UUID> candidates = possibleTargets(target, source, game);
        boolean result = super.chooseTarget(outcome, target, source, game);
        emitTarget(game, source, target, candidates, result);
        return result;
    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target,
                                      Ability source, Game game) {
        List<? extends TargetAmount> candidates = target == null || source == null
                ? Collections.emptyList() : new ArrayList<>(target.getTargetOptions(source, game));
        boolean result = super.chooseTargetAmount(outcome, target, source, game);
        emit(game, RallyCp7DecisionObserver.Kind.TARGET_AMOUNT, source, target,
                target == null ? "" : target.getTargetName(), candidates,
                targetAmountSelection(target), result);
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Cards cards, TargetCard target,
                          Ability source, Game game) {
        List<Card> candidates = cardCandidates(cards, game);
        boolean result = super.choose(outcome, cards, target, source, game);
        emitCardTarget(game, source, target, candidates, result);
        return result;
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target,
                                Ability source, Game game) {
        List<Card> candidates = cardCandidates(cards, game);
        boolean result = super.chooseTarget(outcome, cards, target, source, game);
        emitCardTarget(game, source, target, candidates, result);
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        List<String> candidates = choiceCandidates(choice);
        boolean result = super.choose(outcome, choice, game);
        List<Object> selected = new ArrayList<>();
        if (choice != null && choice.isChosen()) {
            if (choice.isKeyChoice()) {
                selected.add(choice.getChoiceKey());
            } else {
                selected.add(choice.getChoice());
            }
        }
        emit(game, RallyCp7DecisionObserver.Kind.CHOICE, null, choice,
                choice == null ? "" : choice.getMessage(), candidates, selected, result);
        return result;
    }

    /** The one-argument overload delegates here, so this produces exactly one event. */
    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage,
                             String trueText, String falseText, Ability source, Game game) {
        boolean result = super.chooseUse(outcome, message, secondMessage,
                trueText, falseText, source, game);
        List<Boolean> candidates = new ArrayList<>();
        candidates.add(Boolean.TRUE);
        candidates.add(Boolean.FALSE);
        emit(game, RallyCp7DecisionObserver.Kind.CHOOSE_USE, source, outcome,
                message, candidates, Collections.singletonList(result), true);
        return result;
    }

    @Override
    public Mode chooseMode(Modes modes, Ability source, Game game) {
        List<Mode> candidates = modes == null
                ? Collections.emptyList()
                : new ArrayList<>(modes.getAvailableModes(source, game));
        Mode selected = super.chooseMode(modes, source, game);
        emit(game, RallyCp7DecisionObserver.Kind.MODE, source, modes, "",
                candidates, singletonOrEmpty(selected), selected != null);
        return selected;
    }

    @Override
    public int announceX(int min, int max, String message, Game game,
                         Ability source, boolean isManaPay) {
        int selected = super.announceX(min, max, message, game, source, isManaPay);
        emit(game, RallyCp7DecisionObserver.Kind.ANNOUNCE_X, source, isManaPay,
                message, integerDomain(min, max), Collections.singletonList(selected), true);
        return selected;
    }

    @Override
    public int getAmount(int min, int max, String message, Ability source, Game game) {
        int selected = super.getAmount(min, max, message, source, game);
        emit(game, RallyCp7DecisionObserver.Kind.AMOUNT, source, null, message,
                integerDomain(min, max), Collections.singletonList(selected), true);
        return selected;
    }

    @Override
    public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        List<TriggeredAbility> candidates = abilities == null
                ? Collections.emptyList() : new ArrayList<>(abilities);
        TriggeredAbility selected = super.chooseTriggeredAbility(abilities, game);
        emit(game, RallyCp7DecisionObserver.Kind.TRIGGER_ORDER, selected, abilities,
                "", candidates, singletonOrEmpty(selected), selected != null);
        return selected;
    }

    @Override
    public int chooseReplacementEffect(Map<String, String> effectsMap,
                                       Map<String, MageObject> objectsMap, Game game) {
        List<String> candidates = effectsMap == null
                ? Collections.emptyList() : new ArrayList<>(effectsMap.keySet());
        int selected = super.chooseReplacementEffect(effectsMap, objectsMap, game);
        List<Object> selection = selected >= 0 && selected < candidates.size()
                ? Collections.<Object>singletonList(candidates.get(selected))
                : Collections.<Object>singletonList(selected);
        emit(game, RallyCp7DecisionObserver.Kind.REPLACEMENT, null, effectsMap,
                "", candidates, selection, selected >= 0 && selected < candidates.size());
        return selected;
    }

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        List<Permanent> candidates = controlledCreatures(game, attackingPlayerId);
        super.selectAttackers(game, attackingPlayerId);
        List<UUID> selected = game.getCombat() == null
                ? Collections.emptyList() : new ArrayList<>(game.getCombat().getAttackers());
        emit(game, RallyCp7DecisionObserver.Kind.DECLARE_ATTACKERS, null,
                attackingPlayerId, "", candidates, selected, true);
    }

    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        List<Permanent> candidates = game == null
                ? Collections.emptyList() : new ArrayList<>(getAvailableBlockers(game));
        super.selectBlockers(source, game, defendingPlayerId);
        emit(game, RallyCp7DecisionObserver.Kind.DECLARE_BLOCKERS, source,
                defendingPlayerId, "", candidates, combatAssignments(game), true);
    }

    RallyCp7DecisionObserver.DispatchState observerStateForTest() {
        return requireObserverState();
    }

    private void emitTarget(Game game, Ability source, Target target,
                            Collection<?> candidates, boolean result) {
        emit(game, RallyCp7DecisionObserver.Kind.TARGET, source, target,
                target == null ? "" : target.getTargetName(), candidates,
                target == null ? Collections.emptyList() : target.getTargets(), result);
    }

    private void emitCardTarget(Game game, Ability source, TargetCard target,
                                Collection<?> candidates, boolean result) {
        emit(game, RallyCp7DecisionObserver.Kind.CARD_TARGET, source, target,
                target == null ? "" : target.getTargetName(), candidates,
                target == null ? Collections.emptyList() : target.getTargets(), result);
    }

    private void emit(Game game, RallyCp7DecisionObserver.Kind kind, Ability source,
                      Object subject, String prompt, Collection<?> candidates,
                      Collection<?> selected, boolean accepted) {
        requireLiveGameOrSimulation(game);
        requireObserverState().emit(game.isSimulation(), kind, game, source,
                subject, prompt, candidates, selected, accepted);
    }

    private RallyCp7DecisionObserver.DispatchState requireObserverState() {
        if (observerState == null) {
            throw new IllegalStateException("Rally CP7 observer is not installed");
        }
        return observerState;
    }

    private static void requireLiveGameOrSimulation(Game game) {
        if (game == null) {
            throw new IllegalArgumentException("Rally CP7 decision has no game");
        }
    }

    private List<UUID> possibleTargets(Target target, Ability source, Game game) {
        if (target == null) {
            return Collections.emptyList();
        }
        UUID controllerId = target.getAffectedAbilityControllerId(getId());
        return new ArrayList<>(target.possibleTargets(controllerId, source, game));
    }

    private static List<Card> cardCandidates(Cards cards, Game game) {
        return cards == null ? Collections.emptyList() : new ArrayList<>(cards.getCards(game));
    }

    private static List<String> choiceCandidates(Choice choice) {
        if (choice == null) {
            return Collections.emptyList();
        }
        return choice.isKeyChoice()
                ? new ArrayList<>(choice.getKeyChoices().keySet())
                : new ArrayList<>(choice.getChoices());
    }

    private static List<Integer> integerDomain(int min, int max) {
        long count = (long) max - (long) min + 1L;
        if (count <= 0L || count > 4096L) {
            return Collections.emptyList();
        }
        List<Integer> values = new ArrayList<>((int) count);
        for (int value = min; value <= max; value++) {
            values.add(value);
            if (value == Integer.MAX_VALUE) {
                break;
            }
        }
        return values;
    }

    private static List<Object> singletonOrEmpty(Object value) {
        return value == null
                ? Collections.emptyList() : Collections.singletonList(value);
    }

    private static List<String> targetAmountSelection(TargetAmount target) {
        if (target == null || target.getTargets().isEmpty()) {
            return Collections.emptyList();
        }
        List<String> selected = new ArrayList<>();
        for (UUID id : target.getTargets()) {
            selected.add(id + "=" + target.getTargetAmount(id));
        }
        return selected;
    }

    private static List<Permanent> controlledCreatures(Game game, UUID playerId) {
        if (game == null || game.getBattlefield() == null) {
            return Collections.emptyList();
        }
        List<Permanent> creatures = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllActivePermanents(playerId)) {
            if (permanent != null && permanent.isCreature()) {
                creatures.add(permanent);
            }
        }
        return creatures;
    }

    private static List<String> combatAssignments(Game game) {
        if (game == null || game.getCombat() == null) {
            return Collections.emptyList();
        }
        Set<String> assignments = new LinkedHashSet<>();
        for (CombatGroup group : game.getCombat().getGroups()) {
            for (UUID attacker : group.getAttackers()) {
                for (UUID blocker : group.getBlockers()) {
                    assignments.add(blocker + "->" + attacker);
                }
            }
        }
        return new ArrayList<>(assignments);
    }
}
