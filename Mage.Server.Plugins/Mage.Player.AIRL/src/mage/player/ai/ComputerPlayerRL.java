package mage.player.ai;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;

import mage.ConditionalMana;
import mage.MageObject;
import mage.Mana;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.Mode;
import mage.abilities.Modes;
import mage.abilities.TriggeredAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.VariableCost;
import mage.abilities.costs.mana.GenericManaCost;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.costs.mana.VariableManaCost;
import mage.abilities.mana.ManaOptions;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.ColoredManaSymbol;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.filter.StaticFilters;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.rl.QValueEntry;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLState;
import mage.player.ai.rl.RLTrainer;
import mage.player.ai.util.CombatUtil;
import mage.players.Player;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.target.TargetCard;

public class ComputerPlayerRL extends ComputerPlayer {
    public RLModel model;
    protected RLState currentState;
    private final List<RLState> stateBuffer;
    private Ability currentAbility;

    public ComputerPlayerRL(String name, RangeOfInfluence range, RLModel model) {
        super(name, range);
        this.model = model;
        this.stateBuffer = new ArrayList<>();
        RLTrainer.threadLocalLogger.get().info("ComputerPlayerRL initialized for " + name);
    }

    public ComputerPlayerRL(final ComputerPlayerRL player) {
        super(player);
        this.stateBuffer = new ArrayList<>();
    }

    @Override
    public ComputerPlayerRL copy() {
        return new ComputerPlayerRL(this);
    }

    @Override
    public boolean priority(Game game) {
        game.resumeTimer(getTurnControlledBy());
        boolean result = priorityPlay(game);
        game.pauseTimer(getTurnControlledBy());
        return result;
    }

    public INDArray genericChoose(int numOptions, RLState.ActionType actionType, Game game, Ability source) {
        currentState = new RLState(game, actionType, source);
        int numRows = numOptions / RLModel.MAX_OPTIONS;
        int remainingOptions = numOptions % RLModel.MAX_OPTIONS;
        for(int i = 0; i < numRows; i++){
            currentState.exploreDimensions.add(RLModel.MAX_OPTIONS);
        }
        if(remainingOptions > 0){
            currentState.exploreDimensions.add(remainingOptions);
        }

        stateBuffer.add(currentState);
        INDArray qValues = model.predictDistribution(currentState, true);
        return qValues;
    }

    // Stuff like Opp agent? Investigate further how to handle. Just choosing how to handle multiple replacement effects?
    // TODO
//    @Override
//    public int chooseReplacementEffect(Map<String, String> effectsMap, Map<String, MageObject> objectsMap, Game game) {
//        log.debug("chooseReplacementEffect");

    // Stuff like sheoldred's edict, kozilek's command
    @Override
    public Mode chooseMode(Modes modes, Ability source, Game game) {
        Modes availableModes = modes.copy();

        INDArray qValues = genericChoose(availableModes.size() + 1, RLState.ActionType.SELECT_CHOICE, game, source);
        // Create a list to store Q-values with their indices
        List<QValueWithIndex> qValueList = new ArrayList<>();
        for (int i = 0; i < availableModes.size() + 1; i++) {
            qValueList.add(new QValueWithIndex(qValues.getFloat(i), i));
        }

        // Sort the list by Q-value in descending order
        qValueList.sort((a, b) -> Float.compare(b.qValue, a.qValue));

        // Use the sorted Q-values and their indices as needed
        int selectedModes = 0;
        List<Integer> selectedModesList = new ArrayList<>();
        boolean stopChoosing = false;
        for (QValueWithIndex qValueWithIndex : qValueList) {
            // Stop if we've reached the maximum number of modes
            if (selectedModes >= modes.getMaxModes(game, source) && modes.getMaxModes(game, source) > 0) {
                break;
            }

            // Stop if we've reached minimum number of modes
            if (stopChoosing && selectedModes >= modes.getMinModes()) {
                break;
            }

            // Stop if we've reached the minimum and the do nothing option is the best option
            if (qValueWithIndex.index == availableModes.size()) {
                //TODO: Good design to mark this as target?
                currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, qValueWithIndex.index));
                if (selectedModes >= modes.getMinModes()) {
                    break;
                }else{
                    stopChoosing = true;
                    continue;
                }
            }

            // Access the original index and Q-value
            int originalIndex = qValueWithIndex.index;
            currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, originalIndex));
            selectedModes++;
            selectedModesList.add(originalIndex);
        }

        if (selectedModes == 0) {
            return null;
        } else {
            // Add all selected modes except the last one
            for(int i = 0; i < selectedModesList.size() - 1; i++){
                Mode mode = (Mode) availableModes.values().toArray()[selectedModesList.get(i)];
                modes.addSelectedMode(mode.getId());
            }
            // Return the last selected mode to let outer loop handle it
            return (Mode) availableModes.values().toArray()[selectedModesList.get(selectedModesList.size() - 1)];
        }

    }

    // TODO: Make this an AI decision
    @Override
    public int announceXMana(int min, int max, String message, Game game, Ability ability) {
        VariableManaCost variableManaCost = null;
        for (ManaCost cost : ability.getManaCostsToPay()) {
            if (cost instanceof VariableManaCost) {
                if (variableManaCost == null) {
                    variableManaCost = (VariableManaCost) cost;
                } else {
                    throw new RuntimeException("More than one VariableManaCost in spell");
                }
            }
        }
        if (variableManaCost == null) {
            throw new RuntimeException("No VariableManaCost in spell");
        }

        // Get all possible mana combinations
        ManaOptions manaOptions = getManaAvailable(game);
        if (manaOptions.isEmpty() && min == 0) {
            return 0;
        }
        // Use a Set to ensure unique X values
        Set<Integer> possibleXValuesSet = new HashSet<>();
        for (Mana mana : manaOptions) {
            //TODO: Make this work, it will never hit
            if (mana instanceof ConditionalMana && !((ConditionalMana) mana).apply(ability, game, getId(), ability.getManaCosts())) {
                continue;
            }
            int availableMana = mana.count() - ability.getManaCostsToPay().manaValue();

            for (int x = min; x <= max; x++) {
                if (variableManaCost.getXInstancesCount() * x <= availableMana) {
                    possibleXValuesSet.add(x);
                } else {
                    break;
                }
            }
        }

        // Convert the Set to a List
        List<Integer> possibleXValues = new ArrayList<>(possibleXValuesSet);

        // Select the best X value using Q-values
        if (!possibleXValues.isEmpty() && possibleXValues.size() > 1) {
            INDArray qValues = genericChoose(possibleXValues.size(), RLState.ActionType.SELECT_CHOICE, game, ability);
            int bestChoice = 0;
            double bestQVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < possibleXValues.size(); i++) {
                double qVal = qValues.getDouble(i);
                if (qVal > bestQVal) {
                    bestQVal = qVal;
                    bestChoice = i;
                }
            }
            return possibleXValues.get(bestChoice);
        } else if (possibleXValues.size() == 1) {
            // No need to query model for only 1 option
            return possibleXValues.get(0);
        }

        return 0; // Default to 0 if no valid options are found
    }

    //TODO: Implement
    //TODO: I don't know when this is used?
    @Override
    public int announceXCost(int min, int max, String message, Game game, Ability ability, VariableCost variableCost) {
        return super.announceXCost(min, max, message, game, ability, variableCost);
    }

    // Deciding to use FOW alt cast, choosing creaturetype for cavern of souls
    // TODO: Implement
    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        // TODO: Allow RLModel to handle this logic
        // choose the correct color to pay a spell (use last unpaid ability for color hint)
        ManaCost unpaid = null;
        if (!getLastUnpaidMana().isEmpty()) {
            unpaid = new ArrayList<>(getLastUnpaidMana().values()).get(getLastUnpaidMana().size() - 1);
        }
        if (outcome == Outcome.PutManaInPool && unpaid != null && choice.isManaColorChoice()) {
            if (unpaid.containsColor(ColoredManaSymbol.W) && choice.getChoices().contains("White")) {
                choice.setChoice("White");
                return true;
            }
            if (unpaid.containsColor(ColoredManaSymbol.R) && choice.getChoices().contains("Red")) {
                choice.setChoice("Red");
                return true;
            }
            if (unpaid.containsColor(ColoredManaSymbol.G) && choice.getChoices().contains("Green")) {
                choice.setChoice("Green");
                return true;
            }
            if (unpaid.containsColor(ColoredManaSymbol.U) && choice.getChoices().contains("Blue")) {
                choice.setChoice("Blue");
                return true;
            }
            if (unpaid.containsColor(ColoredManaSymbol.B) && choice.getChoices().contains("Black")) {
                choice.setChoice("Black");
                return true;
            }
            if (unpaid.getMana().getColorless() > 0 && choice.getChoices().contains("Colorless")) {
                choice.setChoice("Colorless");
                return true;
            }
        }

        // choose by RLModel
        if (!choice.isChosen()) {
            if (choice.getKeyChoices() != null && !choice.getKeyChoices().isEmpty()) {
                //Keychoice
                if(choice.getKeyChoices().size() > 1){
                    Ability source;
                    if (game.getStack().isEmpty()) {
                        source = currentAbility;
                    }else{
                        source = game.getStack().getFirst().getStackAbility();
                    }
                    INDArray qValues = genericChoose(choice.getKeyChoices().size(), RLState.ActionType.SELECT_CHOICE, game, source);
                    int bestChoice = 0;
                    double bestQVal = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < choice.getKeyChoices().size(); i++) {
                        double qVal = qValues.getDouble(i);
                        if (qVal > bestQVal) {
                            bestQVal = qVal;
                            bestChoice = i;
                        }
                    }
                    choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[bestChoice].toString());
                    return true;
                } else {
                    // Only one choice
                    choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[0].toString());
                    return true;
                }
            } else if(choice.getChoices() != null && !choice.getChoices().isEmpty()) {
                // Normal Choice
                if (choice.getChoices().size() > 1) {
                    INDArray qValues = genericChoose(choice.getChoices().size(), RLState.ActionType.SELECT_CHOICE, game, game.getStack().getFirst().getStackAbility());
                    int bestChoice = 0;
                    double bestQVal = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < choice.getChoices().size(); i++) {
                        double qVal = qValues.getDouble(i);
                        if (qVal > bestQVal) {
                            bestQVal = qVal;
                            bestChoice = i;
                        }
                    }
                    choice.setChoice(choice.getChoices().toArray()[bestChoice].toString());
                    return true;
                } else {
                    choice.setChoice(choice.getChoices().toArray()[0].toString());
                    return true;
                }
            }
        }
        throw new RuntimeException("No choice made");
//        return super.choose(outcome, choice, game);
    }

    // Deciding ponder cards, exile card from opponent's hand
    //Choose2
    @Override
    public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        if (cards == null || cards.isEmpty()) {
            return true;
        }

        // sometimes a target selection can be made from a player that does not control the ability
        UUID abilityControllerId = playerId;
        if (target.getTargetController() != null
                && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }

        List<Card> cardChoices = new ArrayList<>(cards.getCards(target.getFilter(), abilityControllerId, source, game));
        if (cardChoices.isEmpty()) {
            return true;
        }

        INDArray qValues = genericChoose(cardChoices.size(), RLState.ActionType.SELECT_CARD, game, source);
        
        // Create sorted indices based on q-values
        List<Integer> sortedIndices = new ArrayList<>();
        for (int i = 0; i < cardChoices.size(); i++) {
            sortedIndices.add(i);
        }
        sortedIndices.sort((a, b) -> Double.compare(qValues.getDouble(b), qValues.getDouble(a)));
        
        int currentIndex = 0;
        while (!target.doneChoosing(game)) {
            if (currentIndex >= sortedIndices.size()) {
                return target.getTargets().size() >= target.getNumberOfTargets();
            }

            Card card = cardChoices.get(sortedIndices.get(currentIndex));
            if (target.canTarget(abilityControllerId, card.getId(), source, game)) {
                target.add(card.getId(), game);
                cardChoices.remove((int)sortedIndices.get(currentIndex));
            }
            currentIndex++;

            if (outcome == Outcome.Neutral && target.getTargets().size() > target.getNumberOfTargets() + (target.getMaxNumberOfTargets() - target.getNumberOfTargets()) / 2) {
                return true;
            }
        }
        return true;
    }

    // TODO
//    @Override
//    public boolean chooseMulligan(Game game) {
//        log.debug("chooseMulligan");
//        if (hand.size() < 6
//                || isTestsMode() // ignore mulligan in tests
//                || game.getClass().getName().contains("Momir") // ignore mulligan in Momir games
//        ) {
//            return false;
//        }
//        Set<Card> lands = hand.getCards(new FilterLandCard(), game);
//        return lands.size() < 2
//                || lands.size() > hand.size() - 2;
//    }

    // Choosing which stack ability from the stack you want to resolve
    @Override
    public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        if (!abilities.isEmpty()) {
            if (abilities.size() == 1) {
                return abilities.get(0);
            }
            INDArray qValues = genericChoose(abilities.size(), RLState.ActionType.SELECT_TRIGGERED_ABILITY, game, null);
            int bestChoice = 0;
            double bestQVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < abilities.size(); i++) {
                double qVal = qValues.getDouble(i);
                if (qVal > bestQVal) {
                    bestQVal = qVal;
                    bestChoice = i;
                }
            }
            currentState.targetQValues.add(new QValueEntry((float)bestQVal, bestChoice));
            return abilities.get(bestChoice);
        }
        return null;
    }

    // Examples:
    // Damage assignment from fury
    // ((TargetCreatureOrPlaneswalkerAmount) target).getAmountRemaining()
    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        // TODO: Investigate what calls this
        return super.chooseTargetAmount(outcome, target, source, game);
        //return choose(outcome, target, source, game, null);
    }

    // TODO: This breaks on mulligans? Because there is no active player?
    // Examples: Return card from graveyard to hand,
    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        return choose(outcome, target, source, game, null);
    }

    // Examples:
    // Discarding to hand size, Choosing to keep which legend for legend rule
    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        return choose(outcome, target, source, game, null);
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
        UUID abilityControllerId = playerId;
        if (target.getTargetController() != null && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }

        // Not really sure why we are prompted to choose the starting player
        // Just hand it off to the superclass
        // TODO: Investigate why this is called
        if (Objects.equals(target.getTargetName(), "starting player")) {
            return super.choose(outcome, target, source, game, null);
        }

        List<UUID> possibleTargetsList = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));

        // If we can't choose any targets, pass
        if (possibleTargetsList.isEmpty()) {
            return false;
        }

        int maxTargets = target.getMaxNumberOfTargets();
        int minTargets = target.getMinNumberOfTargets();

        // +1 allows option to not select target or select fewer targets
        INDArray qValues = genericChoose(possibleTargetsList.size()+1, RLState.ActionType.SELECT_TARGETS, game, source);

        // Create a list to store Q-values with their indices
        List<QValueWithIndex> qValueList = new ArrayList<>();
        for (int i = 0; i < possibleTargetsList.size() + 1; i++) {
            qValueList.add(new QValueWithIndex(qValues.getFloat(i), i));
        }

        // Sort the list by Q-value in descending order
        qValueList.sort((a, b) -> Float.compare(b.qValue, a.qValue));

        // Use the sorted Q-values and their indices as needed
        int selectedTargets = 0;
        List<UUID> selectedTargetsList = new ArrayList<>();
        boolean stopChoosing = false;
        for (QValueWithIndex qValueWithIndex : qValueList) {
            // Stop if we've tried to stop choosing or we've reached the maximum number of targets
            if (maxTargets != 0 && selectedTargets >= maxTargets) {
                break;
            }

            // Stop if we've reached the minimum number of targets
            if (stopChoosing && selectedTargets >= minTargets) {
                break;
            }

            // Stop if we've reached the minimum number of targets and the "do nothing" option is the best option
            if (qValueWithIndex.index == possibleTargetsList.size()){
                //TODO: Good design to mark this as target?
                currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, qValueWithIndex.index));
                // Can we stop? Or do we need more targets?
                if (selectedTargets >= minTargets) {
                    break;
                }else{
                    stopChoosing = true;
                    continue;
                }
            }
            // Access the original index and Q-value
            int originalIndex = qValueWithIndex.index;
            target.add(possibleTargetsList.get(originalIndex), game);
            currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, originalIndex));
            selectedTargets++;

            // For logging purposes
            selectedTargetsList.add(possibleTargetsList.get(originalIndex));
        }

        List<String> selectedTargetNames = new ArrayList<>();
        for (UUID targetId : selectedTargetsList) {
            MageObject mageObject = game.getObject(targetId);
            if (mageObject != null) {
                selectedTargetNames.add(mageObject.getName());
            } else {
                Player player = game.getPlayer(targetId);
                if (player != null) {
                    selectedTargetNames.add(player.getName());
                }
            }
        }
        // Retrieve the source name
        String sourceName = "Unknown Source";
        if (source != null) {
            MageObject sourceMageObject = game.getObject(source.getSourceId());
            if (sourceMageObject != null) {
                sourceName = sourceMageObject.getName();
            } else {
                Player sourcePlayer = game.getPlayer(source.getSourceId());
                if (sourcePlayer != null) {
                    sourceName = sourcePlayer.getName();
                }
            }
        } else {
            sourceName = outcome.name();
        }
        RLTrainer.threadLocalLogger.get().info("Selected targets: " + selectedTargetNames + " for source: " + sourceName);

        // Return true if the minimum number of targets is selected
        return selectedTargets >= minTargets;
    }

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) {
            // Generate list of possible attackers
            List<Permanent> allAttackers = game.getBattlefield().getAllActivePermanents(
                StaticFilters.FILTER_PERMANENT_CREATURE,
                attackingPlayerId,
                game
            );
            List<Permanent> possibleAttackers = new ArrayList<>();

            for (Permanent creature : allAttackers) {
                if (creature.canAttack(null, game)) {
                    possibleAttackers.add(creature);
                }
            }

            if (possibleAttackers.isEmpty()) {
                return;
            }

            currentState = new RLState(game, RLState.ActionType.DECLARE_ATTACKS);
            stateBuffer.add(currentState);
            // Generate list of attack targets (Player, planeswalkers, battles)
            List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
            if (possibleAttackers.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }
            if (possibleAttackTargets.size() > RLModel.MAX_OPTIONS - 1) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attack targets than max options, Model truncating");
            }
            int numAttackers = Math.min(RLModel.MAX_ACTIONS, possibleAttackers.size());
            // -1 to reserve the option to not attack
            int numAttackTargets = Math.min(RLModel.MAX_OPTIONS-1, possibleAttackTargets.size());

            for(int i = 0; i < numAttackers; i++){
                // +1 to explore the option to not attack
                currentState.exploreDimensions.add(numAttackTargets+1);
            }

            // Predict on game state
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);

            // For each attacker
            // Attacker = X, Attack Target = Y
            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
                Permanent attacker = possibleAttackers.get(attackerIndex);

                // Create a list of defender indices with their Q-values for this attacker
                List<AttackOption> attackOptions = new ArrayList<>();
                for (int attackTargetIndex = 0; attackTargetIndex < RLModel.MAX_OPTIONS; attackTargetIndex++) {
                    float qValue = qValues.getFloat(attackerIndex, attackTargetIndex);
                    attackOptions.add(new AttackOption(attackTargetIndex, attackerIndex, qValue));
                }

                // Sort attack options by Q-value in descending order
                attackOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));

                // Declare attacks based on sorted Q-values
                for (AttackOption option : attackOptions) {
                    if (option.attackTargetIndex >= numAttackTargets) {
                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
                        currentState.targetQValues.add(new QValueEntry(option.qValue, index));
                        break; // Skip this attacker if the first choice is to not attack

                    }
                    UUID attackTargetId = possibleAttackTargets.get(option.attackTargetIndex);
                    if (attacker.canAttack(attackTargetId, game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring attacker: " + attacker.getName() + " for attack target: " + attackTargetId.toString());
                        this.declareAttacker(attacker.getId(), attackTargetId, game, false);
                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
                        currentState.targetQValues.add(new QValueEntry(option.qValue, index));
                        break; // Once an attack is declared, move to the next attacker

                    }
                }
            }
            if (currentState.targetQValues.isEmpty()){
                stateBuffer.remove(currentState);
            }
        }
    }

    // Don't need to override?
    // TODO: Do we need to pass the action vector a reference for WHICH creature its declaring blocks?
    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) {
            List<Permanent> attackers = getAttackers(game);
            if (attackers == null) {
                return;
            }

            List<Permanent> possibleBlockers = super.getAvailableBlockers(game);
            possibleBlockers = filterOutNonblocking(game, attackers, possibleBlockers);
            if (possibleBlockers.isEmpty()) {
                return;
            }

            RLTrainer.threadLocalLogger.get().info("possibleBlockers: " + possibleBlockers);

            attackers = filterOutUnblockable(game, attackers, possibleBlockers);
            if (attackers.isEmpty()) {
                return;
            }

            currentState = new RLState(game, RLState.ActionType.DECLARE_BLOCKS);
            stateBuffer.add(currentState);
            // -1 to reserve the option to not block nothing no a creature. Essentially an attacker that is "nothing"
            int numAttackers = Math.min(RLModel.MAX_ACTIONS - 1, attackers.size());
            int numBlockers = Math.min(RLModel.MAX_OPTIONS, possibleBlockers.size());
            if (attackers.size() > RLModel.MAX_ACTIONS - 1) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }
            if (possibleBlockers.size() > RLModel.MAX_OPTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More blockers than max actions, Model truncating");
            }

            // Build exploration dimensions
            // +1 to explore the option to not block
            for(int i = 0; i < numAttackers + 1; i++){
                currentState.exploreDimensions.add(numBlockers);
            }
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);

            boolean blockerDeclared = false;

            // Iterate over blockers first
            // Attacker = X, Blockers = Y
            for (int blockerIndex = 0; blockerIndex < numBlockers; blockerIndex++) {
                Permanent blocker = possibleBlockers.get(blockerIndex);

                // Create a list of blocker indices with their Q-values for this attacker
                List<BlockOption> blockOptions = new ArrayList<>();
                // We use the full MAX_OPTIONS because we need to reserve the option to not block
                for (int attackerIndex = 0; attackerIndex < RLModel.MAX_ACTIONS; attackerIndex++) {
                    float qValue = qValues.getFloat(attackerIndex, blockerIndex);
                    blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
                }

                // Sort block options by Q-value in descending order
                blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));

                // Declare blocks based on sorted Q-values
                for (BlockOption option : blockOptions) {
                    if (option.attackerIndex >= numAttackers) {
                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
                        currentState.targetQValues.add(new QValueEntry(option.qValue, index));
                        break; // Skip this blocker if the first choice is to not block
                    }

                    Permanent attacker = attackers.get(option.attackerIndex);
                    if (blocker.canBlock(attacker.getId(), game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
                        currentState.targetQValues.add(new QValueEntry(option.qValue, index));
                        blockerDeclared = true;
                        // TODO: implement multiblock
                        break;

                    }
                }
            }
            if (blockerDeclared) {
                game.getPlayers().resetPassed();
            }
            if (currentState.targetQValues.isEmpty()){
                stateBuffer.remove(currentState);
            }
        }
    }

    private List<Permanent> filterOutNonblocking(Game game, List<Permanent> attackers, List<Permanent> blockers) {
        List<Permanent> blockersLeft = new ArrayList<>();
        for (Permanent blocker : blockers) {
            for (Permanent attacker : attackers) {
                if (blocker.canBlock(attacker.getId(), game)) {
                    blockersLeft.add(blocker);
                    break;
                }
            }
        }
        return blockersLeft;
    }

    private List<Permanent> filterOutUnblockable(Game game, List<Permanent> attackers, List<Permanent> blockers) {
        List<Permanent> attackersLeft = new ArrayList<>();
        for (Permanent attacker : attackers) {
            if (CombatUtil.canBeBlocked(game, attacker, blockers)) {
                attackersLeft.add(attacker);
            }
        }
        return attackersLeft;
    }

    private List<Permanent> getAttackers(Game game) {
        Set<UUID> attackersUUID = game.getCombat().getAttackers();
        if (attackersUUID.isEmpty()) {
            return null;
        }

        List<Permanent> attackers = new ArrayList<>();
        for (UUID attackerId : attackersUUID) {
            Permanent permanent = game.getPermanent(attackerId);
            attackers.add(permanent);
        }
        return attackers;
    }

    //    @Override
//    public boolean chooseMulligan(Game game) {
//        currentState = new RLState(game);
//        RLAction action = model.getAction(currentState);
//        return action != null && action.getType() == RLAction.ActionType.MULLIGAN;
//    }

    protected boolean priorityPlay(Game game) {
        game.getState().setPriorityPlayerId(playerId);
        game.firePriorityEvent(playerId);
        Ability ability;
        switch (game.getTurnStepType()) {
            case UPKEEP:
            case DRAW:
                pass(game);
                return false;
            case PRECOMBAT_MAIN:
                printBattleField(game, "Sim PRIORITY on MAIN 1");
                currentAbility = calculateActions(game);
                act(game, (ActivatedAbility) currentAbility);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                currentAbility = calculateActions(game);
                act(game, (ActivatedAbility) currentAbility);
                pass(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE BLOCKERS");
                currentAbility = calculateActions(game);
                act(game, (ActivatedAbility) currentAbility);
                pass(game);
                return true;
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
            case END_COMBAT:
                pass(game);
                return false;
            case POSTCOMBAT_MAIN:
                printBattleField(game, "Sim PRIORITY on MAIN 2");
                currentAbility = calculateActions(game);
                act(game, (ActivatedAbility) currentAbility);
                return true;
            case END_TURN:  
            case CLEANUP:
                pass(game);
                return false;
        }
        return false;
    }

    protected void printBattleField(Game game, String info) {
        if (RLTrainer.threadLocalLogger.get().isInfoEnabled()) {
            // Clear the console line
            System.out.print("\033[2K"); // ANSI escape code to clear the current line
            // Move the cursor up one line
            System.out.print("\033[1A");

            // Print the battlefield information
            System.out.println("=================== " + info + ", turn " + game.getTurnNum() + ", " + game.getPlayer(game.getPriorityPlayerId()).getName() + " ===================");
            System.out.println("[Stack]: " + game.getStack());
            printBattleField(game, playerId);
            for (UUID opponentId : game.getOpponents(playerId)) {
                printBattleField(game, opponentId);
            }
        }
    }

    protected void printBattleField(Game game, UUID playerId) {
        Player player = game.getPlayer(playerId);
        System.out.println(new StringBuilder("[").append(game.getPlayer(playerId).getName()).append("]")
                .append(", life = ").append(player.getLife())
                .toString());
        String cardsInfo = player.getHand().getCards(game).stream()
                .map(card -> card.getName()) // Removed card score
                .collect(Collectors.joining("; "));
        StringBuilder sb = new StringBuilder("-> Hand: [")
                .append(cardsInfo)
                .append("]");
        System.out.println(sb.toString());

        // battlefield
        sb.setLength(0);
        String ownPermanentsInfo = game.getBattlefield().getAllPermanents().stream()
                .filter(p -> p.isOwnedBy(player.getId()))
                .map(p -> p.getName()
                        + (p.isTapped() ? ",tapped" : "")
                        + (p.isAttacking() ? ",attacking" : "")
                        + (p.getBlocking() > 0 ? ",blocking" : ""))
                .collect(Collectors.joining("; "));
        sb.append("-> Permanents: [").append(ownPermanentsInfo).append("]");
        System.out.println(sb.toString());
    }

    // I'm changing the design here to not use an actions queue.
    // Instead, I'm passing the ability to the act method.
    // We don't calculate lists of actions, but instead just one action at a time.
    // NOTE: I think the way computerplayer6 does this is because it implements the idea
    // of holding priority
    protected void act(Game game, ActivatedAbility ability) {
        if (ability == null) {
            RLTrainer.threadLocalLogger.get().info("Model opted to pass priority");
            pass(game);
        } else {
            RLTrainer.threadLocalLogger.get().info(String.format("===> SELECTED ACTION for %s: %s", getName(), ability));
            //TODO: Need to look into target selection. 
            if (!ability.getTargets().isEmpty()) {
                for (Target target : ability.getTargets()) {
                    for (UUID id : target.getTargets()) {
                        target.updateTarget(id, game);
                        if (!target.isNotTarget()) {
                            game.addSimultaneousEvent(GameEvent.getEvent(GameEvent.EventType.TARGETED, id, ability, ability.getControllerId()));
                        }
                    }
                }
            }
            if (!this.activateAbility(ability, game)){
                //TODO: if we are here it is because the ComputerPlayerRL chose an invalid subaction (choose likely)
                throw new RuntimeException("Failed to activate ability: " + ability);
            }
            //TODO: Implement holding priority for abilities that don't use the stack
            if (ability.isUsesStack()){
                pass(game);
            }
        }
    }

    protected List<ActivatedAbility> getPlayableAbilities(Game game) {
        List<ActivatedAbility> playables = getPlayable(game, true);
        playables.add(new PassAbility());
        return playables;
    }

    public List<List<Ability>> getPlayableOptions(Game game) {
        List<List<Ability>> allOptions = new ArrayList<>();
        List<ActivatedAbility> playables = getPlayableAbilities(game);

        for (ActivatedAbility ability : playables) {
            List<Ability> options = game.getPlayer(playerId).getPlayableOptions(ability, game);
            if (options.isEmpty()) {
                if (!ability.getManaCosts().getVariableCosts().isEmpty()) {
                    options = simulateVariableCosts(ability, game);
                } else {
                    options.add(ability);
                }
            } else {
                List<Ability> expandedOptions = new ArrayList<>();
                for (Ability option : options) {
                    if (!option.getManaCosts().getVariableCosts().isEmpty()) {
                        expandedOptions.addAll(simulateVariableCosts(option, game));
                    } else {
                        expandedOptions.add(option);
                    }
                }
                options = expandedOptions;
            }
            allOptions.add(options);
        }

        // Ensure the list does not exceed RLModel.MAX_ACTIONS x RLModel.MAX_OPTIONS
//        if (allOptions.size() > RLModel.MAX_ACTIONS) {
//            RLTrainer.threadLocalLogger.get().error("ERROR: More actions than max actions, Model truncating");
//            allOptions = allOptions.subList(0, RLModel.MAX_ACTIONS);
//        }
//        for (int i = 0; i < allOptions.size(); i++) {
//            List<Ability> optionList = allOptions.get(i);
//
//
//            if (optionList.size() > RLModel.MAX_OPTIONS) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More options than max options, Model truncating");
//                allOptions.set(i, optionList.subList(0, RLModel.MAX_OPTIONS));
//            }
//        }

        return allOptions;
    }

    // TODO: This doesn't work for XX costs, i think it will suggest spending 1 mana on an XX
    private List<Ability> simulateVariableCosts(Ability ability, Game game) {
        List<Ability> options = new ArrayList<>();
        // TODO: This is wrong. getavailproducers returns 1 if you have an ancient tomb which can produce mana values of 2
        int numAvailable = getAvailableManaProducers(game).size() - ability.getManaCosts().manaValue();
        int start = 0;
        for (int i = 0; i < numAvailable; i++) {
            Ability newAbility = ability.copy();
            newAbility.addManaCostsToPay(new GenericManaCost(i));
            options.add(newAbility);
        }
        return options;
    }

    //TODO: Add this to calc?
        //     //Filter playables to remove tapping lands down.
        // // I thought this was what was breaking some spells(manifold mouse) but it still doesn't work
        // List<Ability> filtered = new ArrayList<Ability>();
        // for(int i=0;i<playableOptions.size();i++){
        //     MageObject source=playableOptions.get(i).getSourceObjectIfItStillExists(game);
        //     if(source!=null && source instanceof Permanent && source.isLand()){
        //         //Don't allow just tapping a land to be an action
        //         //May break lands with activated abilities
        //         continue;
        //     }
        //     filtered.add(playableOptions.get(i));
        // }
        // playableOptions=filtered;

    protected Game createSimulation(Game game) {
        Game sim = game.createSimulationForAI();
        for (Player oldPlayer : sim.getState().getPlayers().values()) {
            // replace original player by simulated player and find result (execute/resolve current action)
            Player origPlayer = game.getState().getPlayers().get(oldPlayer.getId()).copy();
            SimulatedPlayer2 simPlayer = new SimulatedPlayer2(oldPlayer, oldPlayer.getId().equals(playerId));
            simPlayer.restore(origPlayer);
            sim.getState().getPlayers().put(oldPlayer.getId(), simPlayer);
        }
        return sim;
    }

    protected Ability calculateActions(Game game) {
        Game sim = createSimulation(game);
        SimulatedPlayer2 currentPlayer = (SimulatedPlayer2) sim.getPlayer(game.getPlayerList().get());
        List<Ability> flattenedOptions = currentPlayer.simulatePriority(sim);
        List<Ability> validOptions = new ArrayList<>();
        for(Ability ability : flattenedOptions){
            Game tmpGame = createSimulation(game);
            SimulatedPlayer2 tmpPlayer = (SimulatedPlayer2) tmpGame.getPlayer(game.getPlayerList().get());
            ActivatedAbility tmpAbility = (ActivatedAbility) ability.copy();
            if (tmpPlayer.activateAbility(tmpAbility, tmpGame)){
                validOptions.add(ability);
            } else{
                RLTrainer.threadLocalLogger.get().info("Invalid ability: " + ability);
            }
        }

        flattenedOptions = validOptions;

        // Remove duplicate spell abilities with the same name
        List<Ability> uniqueOptions = new ArrayList<>();
        Set<String> seenNames = new HashSet<>();

        // Remove duplicate spell abilities with the same name
        // TODO: Investigate if this is what we want. I did this because despite "setting targets" during selection. we still get prompted for choices later anyway
        for (Ability ability : flattenedOptions) {
            String name = ability.toString();
            if (!seenNames.contains(name)) {
                seenNames.add(name);
                uniqueOptions.add(ability);
            }
        }
        flattenedOptions = uniqueOptions;

        // If we can only pass, don't query model
        if (flattenedOptions.size() == 1) {
            return flattenedOptions.get(0);
        }

        // Set the exploration dimensions to the size of the flattened list
        currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
        stateBuffer.add(currentState);
        int numRows = (flattenedOptions.size()) / RLModel.MAX_OPTIONS;
        int numCols = (flattenedOptions.size()) % RLModel.MAX_OPTIONS;
        for (int i = 0; i < numRows; i++) {
            currentState.exploreDimensions.add(RLModel.MAX_OPTIONS);
        }
        if (numCols > 0) {
            currentState.exploreDimensions.add(numCols);
        }

        // Evaluate each action using the model
        INDArray qValues = model.predictDistribution(currentState, true);
        float maxQValue = Float.NEGATIVE_INFINITY;
        int bestIndex = 0;

        // Find the index with the highest Q-value
        for (int i = 0; i < RLModel.OUTPUT_SIZE; i++) {
            float qValue = qValues.getFloat(i);
            if (qValue > maxQValue) {
                maxQValue = qValue;
                bestIndex = i;
            }
        }

        currentState.targetQValues.add(new QValueEntry(maxQValue, bestIndex));
        RLTrainer.threadLocalLogger.get().info("Playable options: " + flattenedOptions);
        RLTrainer.threadLocalLogger.get().info("Best action index: " + bestIndex);
        RLTrainer.threadLocalLogger.get().info("Best Q-value: " + maxQValue);


        // Return the ability corresponding to the best index
        if (bestIndex < flattenedOptions.size()) {
            return flattenedOptions.get(bestIndex);
        } else {
            throw new RuntimeException("Best index out of bounds");
        }
    }

    public List<RLState> getStateBuffer() {
        return stateBuffer;
    }
}

// Helper class to store block options
class BlockOption {
    int attackerIndex;
    int blockerIndex;
    float qValue;

    BlockOption(int attackerIndex, int blockerIndex, float qValue) {
        this.attackerIndex = attackerIndex;
        this.blockerIndex = blockerIndex;
        this.qValue = qValue;
    }
}

// Helper class to store attack options
class AttackOption {
    int attackTargetIndex;
    int attackerIndex;
    float qValue;

    AttackOption(int attackTargetIndex, int attackerIndex, float qValue) {
        this.attackTargetIndex = attackTargetIndex;
        this.attackerIndex = attackerIndex;
        this.qValue = qValue;
    }
}

// Helper class to store Q-value with its index
class QValueWithIndex {
    float qValue;
    int index;

    QValueWithIndex(float qValue, int index) {
        this.qValue = qValue;
        this.index = index;
    }
}
