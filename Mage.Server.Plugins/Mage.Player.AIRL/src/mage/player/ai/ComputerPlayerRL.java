package mage.player.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.constants.RangeOfInfluence;
import mage.constants.TurnPhase;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.rl.PythonMLBridge;
import mage.player.ai.rl.RLTrainer;
import mage.player.ai.rl.StateSequenceBuilder;
import mage.player.ai.util.CombatUtil;
import mage.players.Player;
import mage.target.Target;

public class ComputerPlayerRL extends ComputerPlayer {

    private PythonMLBridge model;
    protected StateSequenceBuilder.SequenceOutput currentState;
    private final List<StateSequenceBuilder.TrainingData> trainingBuffer;
    private Ability currentAbility;

    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonMLBridge model) {
        super(name, range);
        this.model = model;
        this.trainingBuffer = new ArrayList<>();
        RLTrainer.threadLocalLogger.get().info("ComputerPlayerRL initialized for " + name);
    }

    // The default constructor for ComputerPlayerRL used by server to create
    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill) {
        this(name, range, PythonMLBridge.getInstance());
    }

    public ComputerPlayerRL(final ComputerPlayerRL player) {
        super(player);
        this.model = player.model;
        this.trainingBuffer = new ArrayList<>();
        this.currentAbility = player.currentAbility;
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

    public List<Integer> genericChoose(int possibleTargetsSize, int maxTargets, int minTargets, StateSequenceBuilder.ActionType actionType, Game game, Ability source) {
        // Don't query the model if only one/no option
        if (possibleTargetsSize == 1) {
            return Arrays.asList(0);
        } else if (possibleTargetsSize == 0) {
            return Arrays.asList();
        }

        // Get current phase
        TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;

        // Build base state
        StateSequenceBuilder.SequenceOutput baseState = StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);

        // Create action mask - 1 for valid actions (up to possibleTargetsSize), 0 for invalid
        float[] actionMask = new float[15]; // Using 15 as the fixed number of actions
        for (int i = 0; i < possibleTargetsSize; i++) {
            actionMask[i] = 1.0f;
        }

        // Get model predictions
        INDArray[] predictions = model.predict(baseState);
        INDArray actionProbs = predictions[0];
        INDArray valueScore = predictions[1];

        // Convert action probabilities to Java array and apply mask
        float[] maskedProbs = new float[possibleTargetsSize];
        float maxProb = Float.NEGATIVE_INFINITY;

        // First pass: apply mask and find max for numerical stability
        for (int i = 0; i < possibleTargetsSize; i++) {
            float prob = actionProbs.getFloat(i);
            // Handle numerical stability issues
            if (Float.isNaN(prob) || Float.isInfinite(prob)) {
                prob = 0.0f;
            }
            maskedProbs[i] = prob * actionMask[i];
            maxProb = Math.max(maxProb, maskedProbs[i]);
        }

        // Second pass: subtract max and apply softmax for numerical stability
        float sum = 0.0f;
        for (int i = 0; i < possibleTargetsSize; i++) {
            maskedProbs[i] = (float) Math.exp(maskedProbs[i] - maxProb);
            sum += maskedProbs[i];
        }

        // Normalize probabilities
        if (sum > 0) {
            for (int i = 0; i < possibleTargetsSize; i++) {
                maskedProbs[i] /= sum;
            }
        } else {
            // If all probabilities are 0, use uniform distribution
            float uniformProb = 1.0f / possibleTargetsSize;
            for (int i = 0; i < possibleTargetsSize; i++) {
                maskedProbs[i] = uniformProb;
            }
        }

        RLTrainer.threadLocalLogger.get().info("Action probabilities: " + Arrays.toString(maskedProbs));
        RLTrainer.threadLocalLogger.get().info("Value score: " + valueScore.toString());

        // Sample from the probability distribution
        List<Integer> selectedIndices = new ArrayList<>();
        float[] cumulativeProbs = new float[possibleTargetsSize];
        cumulativeProbs[0] = maskedProbs[0];
        for (int i = 1; i < possibleTargetsSize; i++) {
            cumulativeProbs[i] = cumulativeProbs[i - 1] + maskedProbs[i];
        }

        // Select indices based on probabilities
        Random random = new Random();
        while (selectedIndices.size() < maxTargets) {
            float r = random.nextFloat();
            for (int i = 0; i < cumulativeProbs.length; i++) {
                if (r <= cumulativeProbs[i]) {
                    if (!selectedIndices.contains(i)) {
                        selectedIndices.add(i);
                    }
                    break;
                }
            }
        }

        // Ensure we have at least minTargets
        while (selectedIndices.size() < minTargets) {
            int remaining = possibleTargetsSize - selectedIndices.size();
            if (remaining <= 0) {
                break;
            }

            // Find indices not yet selected
            List<Integer> available = new ArrayList<>();
            for (int i = 0; i < possibleTargetsSize; i++) {
                if (!selectedIndices.contains(i)) {
                    available.add(i);
                }
            }

            // Randomly select from remaining indices
            if (!available.isEmpty()) {
                selectedIndices.add(available.get(random.nextInt(available.size())));
            }
        }

        return selectedIndices;
    }

    // Helper method to generate all valid combinations of targets
    private void generateCombinations(List<List<Integer>> result, List<Integer> current, int start, int n, int minSize, int maxSize) {
        // If current combination is valid size, add it
        if (current.size() >= minSize && current.size() <= maxSize) {
            result.add(new ArrayList<>(current));
        }

        // If we've reached max size, stop
        if (current.size() >= maxSize) {
            return;
        }

        // Try adding each remaining number
        for (int i = start; i < n; i++) {
            current.add(i);
            generateCombinations(result, current, i + 1, n, minSize, maxSize);
            current.remove(current.size() - 1);
        }
    }

    // Stuff like Opp agent? Investigate further how to handle. Just choosing how to handle multiple replacement effects?
    // TODO
//    @Override
//    public int chooseReplacementEffect(Map<String, String> effectsMap, Map<String, MageObject> objectsMap, Game game) {
//        log.debug("chooseReplacementEffect");
    // Stuff like sheoldred's edict, kozilek's command
    // @Override
    // public Mode chooseMode(Modes modes, Ability source, Game game) {
    //     //TODO: Testing if we can make this not a copy.
    //     ArrayList<UUID> modeIds = new ArrayList<>(modes.values().stream().map(Mode::getId).collect(Collectors.toList()));
    //     for (UUID modeId : modeIds) {
    //         Mode mode = modes.get(modeId);
    //         // Need to do this so target validation is correct
    //         modes.addSelectedMode(mode.getId());
    //         source.getModes().setActiveMode(modeId);
    //         if (!source.getAbilityType().isTriggeredAbility()) {
    //             source.adjustTargets(game);
    //         }
    //         if ((!mode.getTargets().isEmpty() && !mode.getTargets().canChoose(source.getControllerId(), source, game)) || (mode.getCost() != null && !mode.getCost().canPay(source, source, playerId, game))) {
    //             modes.remove(modeId);
    //         }
    //         modes.removeSelectedMode(modeId);
    //     }
    //     int maxTargets = Math.min(modes.getMaxModes(game, source), modes.size());
    //     int minTargets = modes.getMinModes();
    //     List<Integer> targetsToSet = genericChoose(modes.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //     if (targetsToSet.size() == 1){
    //         return (Mode) modes.values().toArray()[0];
    //     } else if(targetsToSet.isEmpty()){
    //         return null;
    //     }else {
    //         // Add all selected modes except the last one
    //         for(int i = 0; i < targetsToSet.size() - 1; i++){
    //             Mode mode = (Mode) modes.values().toArray()[targetsToSet.get(i)];
    //             modes.addSelectedMode(mode.getId());
    //         }
    //         // Return the last selected mode to let outer loop handle it
    //         return (Mode) modes.values().toArray()[targetsToSet.get(targetsToSet.size() - 1)];
    //     }
    // }
    // @Override
    // public int announceXMana(int min, int max, String message, Game game, Ability ability) {
    //     VariableManaCost variableManaCost = null;
    //     for (ManaCost cost : ability.getManaCostsToPay()) {
    //         if (cost instanceof VariableManaCost) {
    //             if (variableManaCost == null) {
    //                 variableManaCost = (VariableManaCost) cost;
    //             } else {
    //                 throw new RuntimeException("More than one VariableManaCost in spell");
    //             }
    //         }
    //     }
    //     if (variableManaCost == null) {
    //         throw new RuntimeException("No VariableManaCost in spell");
    //     }
    //     // Get all possible mana combinations
    //     ManaOptions manaOptions = getManaAvailable(game);
    //     if (manaOptions.isEmpty() && min == 0) {
    //         return 0;
    //     }
    //     // Use a Set to ensure unique X values
    //     Set<Integer> possibleXValuesSet = new HashSet<>();
    //     possibleXValuesSet.add(0); // Always allow X=0
    //     for (Mana mana : manaOptions) {
    //         if (mana instanceof ConditionalMana && !((ConditionalMana) mana).apply(ability, game, getId(), ability.getManaCosts())) {
    //             continue;
    //         }
    //         int availableMana = mana.count() - ability.getManaCostsToPay().manaValue();
    //         for (int x = min; x <= max; x++) {
    //             if (variableManaCost.getXInstancesCount() * x <= availableMana) {
    //                 possibleXValuesSet.add(x);
    //             } else {
    //                 break;
    //             }
    //         }
    //     }
    //     // Convert the Set to a List
    //     List<Integer> possibleXValues = new ArrayList<>(possibleXValuesSet);
    //     // Select the best X value using Q-values
    //     if (!possibleXValues.isEmpty() && possibleXValues.size() > 1) {
    //         List<Integer> targetsToSet = genericChoose(possibleXValues.size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, ability);
    //         return possibleXValues.get(targetsToSet.get(0));
    //     } else if (possibleXValues.size() == 1) {
    //         // No need to query model for only 1 option
    //         return possibleXValues.get(0);
    //     }
    //     return 0; // Default to 0 if no valid options are found
    // }
    // //TODO: Implement
    // //TODO: I don't know when this is used?
    // @Override
    // public int announceXCost(int min, int max, String message, Game game, Ability ability, VariableCost variableCost) {
    //     return super.announceXCost(min, max, message, game, ability, variableCost);
    // }
    // // Deciding to use FOW alt cast, choosing creaturetype for cavern of souls
    // // TODO: Implement
    // @Override
    // public boolean choose(Outcome outcome, Choice choice, Game game) {
    //     // TODO: Allow RLModel to handle this logic
    //     // choose the correct color to pay a spell (use last unpaid ability for color hint)
    //     ManaCost unpaid = null;
    //     if (!getLastUnpaidMana().isEmpty()) {
    //         unpaid = new ArrayList<>(getLastUnpaidMana().values()).get(getLastUnpaidMana().size() - 1);
    //     }
    //     if (outcome == Outcome.PutManaInPool && unpaid != null && choice.isManaColorChoice()) {
    //         if (unpaid.containsColor(ColoredManaSymbol.W) && choice.getChoices().contains("White")) {
    //             choice.setChoice("White");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.R) && choice.getChoices().contains("Red")) {
    //             choice.setChoice("Red");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.G) && choice.getChoices().contains("Green")) {
    //             choice.setChoice("Green");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.U) && choice.getChoices().contains("Blue")) {
    //             choice.setChoice("Blue");
    //             return true;
    //         }
    //         if (unpaid.containsColor(ColoredManaSymbol.B) && choice.getChoices().contains("Black")) {
    //             choice.setChoice("Black");
    //             return true;
    //         }
    //         if (unpaid.getMana().getColorless() > 0 && choice.getChoices().contains("Colorless")) {
    //             choice.setChoice("Colorless");
    //             return true;
    //         }
    //     }
    //     // choose by RLModel
    //     Ability source;
    //     if (game.getStack().isEmpty()) {
    //         source = currentAbility;
    //     }else{
    //         source = game.getStack().getFirst().getStackAbility();
    //     }
    //     if (!choice.isChosen()) {
    //         if (choice.getKeyChoices() != null && !choice.getKeyChoices().isEmpty()) {
    //             for (Map.Entry<String, String> entry : choice.getKeyChoices().entrySet()) {
    //                 if (choice.getChoice() == null) {
    //                     choice.setChoice(entry.getKey());
    //                 }
    //             }
    //             //Keychoice
    //             if(choice.getKeyChoices().size() > 1){
    //                 List<Integer> targetsToSet = genericChoose(choice.getKeyChoices().size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //                 choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[targetsToSet.get(0)].toString());
    //                 return true;
    //             } else {
    //                 // Only one choice
    //                 choice.setChoiceByKey(choice.getKeyChoices().keySet().toArray()[0].toString());
    //                 return true;
    //             }
    //         } else if(choice.getChoices() != null && !choice.getChoices().isEmpty()) {
    //             // Normal Choice
    //             if (choice.getChoices().size() > 1) {
    //                 List<Integer> targetsToSet = genericChoose(choice.getChoices().size(),1,1, StateSequenceBuilder.ActionType.SELECT_CHOICE, game, source);
    //                 choice.setChoice(choice.getChoices().toArray()[targetsToSet.get(0)].toString());
    //                 return true;
    //             } else {
    //                 choice.setChoice(choice.getChoices().toArray()[0].toString());
    //                 return true;
    //             }
    //         }
    //     }
    //     throw new RuntimeException("No choice made");
    // }
    // Deciding ponder cards, exile card from opponent's hand
    //Choose2
    // @Override
    // public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
    //     if (cards == null || cards.isEmpty()) {
    //         return true;
    //     }
    //     // sometimes a target selection can be made from a player that does not control the ability
    //     UUID abilityControllerId = playerId;
    //     if (target.getTargetController() != null
    //             && target.getAbilityController() != null) {
    //         abilityControllerId = target.getAbilityController();
    //     }
    //     List<Card> cardChoices = new ArrayList<>(cards.getCards(target.getFilter(), abilityControllerId, source, game));
    //     int maxTargets = Math.min(target.getMaxNumberOfTargets(), cardChoices.size());
    //     int minTargets = target.getMinNumberOfTargets();
    //     List<Integer> targetsToSet = genericChoose(cardChoices.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_CARD, game, source);
    //     for (int i = 0; i < targetsToSet.size(); i++) {
    //         target.add(cardChoices.get(targetsToSet.get(i)).getId(), game);
    //     }
    //     return true;
    // }
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
    // @Override
    // public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
    //     if (!abilities.isEmpty()) {
    //         if (abilities.size() == 1) {
    //             return abilities.get(0);
    //         }
    //         List<Integer> targetsToSet = genericChoose(abilities.size(),1,1, StateSequenceBuilder.ActionType.SELECT_TRIGGERED_ABILITY, game, null);
    //         return abilities.get(targetsToSet.get(0));
    //     }
    //     return null;
    // }
    // Examples:
    // Damage assignment from fury
    // ((TargetCreatureOrPlaneswalkerAmount) target).getAmountRemaining()
    // @Override
    // public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
    //     // TODO: Investigate what calls this
    //     return super.chooseTargetAmount(outcome, target, source, game);
    //     //return choose(outcome, target, source, game, null);
    // }
    // TODO: This breaks on mulligans? Because there is no active player?
    // Examples: Return card from graveyard to hand,
    // @Override
    // public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
    //     return choose(outcome, target, source, game, null);
    // }
    // Examples: Choosing when searching library. Fetch lands
//     @Override
//     public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
//         if (cards == null || cards.isEmpty()) {
//             return target.isRequired(source);
//         }
//         // sometimes a target selection can be made from a player that does not control the ability
//         UUID abilityControllerId = playerId;
//         if (target.getTargetController() != null
//                 && target.getAbilityController() != null) {
//             abilityControllerId = target.getAbilityController();
//         }
//         // we still use playerId when getting cards even if they don't control the search
//         List<Card> cardChoices = new ArrayList<>(cards.getCards(target.getFilter(), playerId, source, game));
//         // TODO: Fetchlands incorrectly state mintargets = 1 but you can "fail to find"
//         int maxTargets = target.getMaxNumberOfTargets();
//         int minTargets = target.getMinNumberOfTargets();
//         List<Integer> targetsToSet = genericChoose(cardChoices.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source);
//         for (int i = 0; i < targetsToSet.size(); i++) {
//             // TODO: For some reason this always fails because the card zone is OUTSIDE
//             // Pretty important to fix this for computerPlayer because I think they always fail to find
//             // so they will be rly bad, could just be with how I'm setting the game up?
// //            if (target.canTarget(abilityControllerId, card.getId(), source, game)) {
//             target.add(cardChoices.get(targetsToSet.get(i)).getId(), game);
//         }
//         return true;
//     }
    // Examples:
    // Discarding to hand size, Choosing to keep which legend for legend rule
    // @Override
    // public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
    //     return choose(outcome, target, source, game, null);
    // }
    // @Override
    // public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
    //     UUID abilityControllerId = playerId;
    //     if (target.getTargetController() != null && target.getAbilityController() != null) {
    //         abilityControllerId = target.getAbilityController();
    //     }
    //     // TODO: I guess we can make this an ai decision?
    //     if (Objects.equals(target.getTargetName(), "starting player")) {
    //         return super.choose(outcome, target, source, game, null);
    //     }
    //     List<UUID> possibleTargetsList = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
    //     // Remove targets that can't be targeted
    //     for (UUID possibleTarget : possibleTargetsList) {
    //         if (!target.canTarget(abilityControllerId, possibleTarget, source, game)) {
    //             possibleTargetsList.remove(possibleTarget);
    //         }
    //     }
    //     int maxTargets = Math.min(target.getMaxNumberOfTargets(), possibleTargetsList.size());
    //     int minTargets = target.getMinNumberOfTargets();
    //     List<Integer> qValues = genericChoose(possibleTargetsList.size(), maxTargets, minTargets, StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source);
    //     for (int i = 0; i < qValues.size(); i++) {
    //         target.add(possibleTargetsList.get(qValues.get(i)), game);
    //     }
    //     return true;
    // }
//    @Override
//    public void selectAttackers(Game game, UUID attackingPlayerId) {
//        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
//        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) {
//            // Generate list of possible attackers
//            List<Permanent> allAttackers = game.getBattlefield().getAllActivePermanents(
//                StaticFilters.FILTER_PERMANENT_CREATURE,
//                attackingPlayerId,
//                game
//            );
//            List<Permanent> possibleAttackers = new ArrayList<>();
//
//            for (Permanent creature : allAttackers) {
//                if (creature.canAttack(null, game)) {
//                    possibleAttackers.add(creature);
//                }
//            }
//
//            if (possibleAttackers.isEmpty()) {
//                return;
//            }
//
//            currentState = StateSequenceBuilder.build(game,
//                                                      StateSequenceBuilder.ActionType.DECLARE_ATTACKS,
//                                                      game.getPhase().getType(),
//                                                      StateSequenceBuilder.MAX_LEN);
//            stateBuffer.add(currentState);
//            // Generate list of attack targets (Player, planeswalkers, battles)
//            List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
//            if (possibleAttackers.size() > RLModel.MAX_ACTIONS) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
//            }
//            if (possibleAttackTargets.size() > RLModel.MAX_OPTIONS - 1) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attack targets than max options, Model truncating");
//            }
//            int numAttackers = Math.min(RLModel.MAX_ACTIONS, possibleAttackers.size());
//            // -1 to reserve the option to not attack
//            int numAttackTargets = Math.min(RLModel.MAX_OPTIONS-1, possibleAttackTargets.size());
//
//            // predict logits once for the whole batch
//            INDArray qValues = model.predictDistribution(currentState, true)
//                                    .reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
//
//            // for each attacker we'll record its chosen index separately
//            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
//                Permanent attacker = possibleAttackers.get(attackerIndex);
//
//                // Create a list of defender indices with their Q-values for this attacker
//                List<AttackOption> attackOptions = new ArrayList<>();
//                for (int attackTargetIndex = 0; attackTargetIndex < RLModel.MAX_OPTIONS; attackTargetIndex++) {
//                    float qValue = qValues.getFloat(attackerIndex, attackTargetIndex);
//                    attackOptions.add(new AttackOption(attackTargetIndex, attackerIndex, qValue));
//                }
//
//                // Sort attack options by Q-value in descending order
//                attackOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
//
//                // Declare attacks based on sorted Q-values
//                for (AttackOption option : attackOptions) {
//                    if (option.attackTargetIndex >= numAttackTargets) {
//                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this attacker if the first choice is to not attack
//                    }
//                    UUID attackTargetId = possibleAttackTargets.get(option.attackTargetIndex);
//                    if (attacker.canAttack(attackTargetId, game)) {
//                        RLTrainer.threadLocalLogger.get().info("Declaring attacker: " + attacker.getName() + " for attack target: " + attackTargetId.toString());
//                        this.declareAttacker(attacker.getId(), attackTargetId, game, false);
//                        int index = attackerIndex * (numAttackTargets + 1) + option.attackTargetIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Once an attack is declared, move to the next attacker
//                    }
//                }
//            }
//        }
//    }
//    @Override
//    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
//        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
//        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) {
//            List<Permanent> attackers = getAttackers(game);
//            if (attackers == null) {
//                return;
//            }
//
//            List<Permanent> possibleBlockers = super.getAvailableBlockers(game);
//            possibleBlockers = filterOutNonblocking(game, attackers, possibleBlockers);
//            if (possibleBlockers.isEmpty()) {
//                return;
//            }
//
//            RLTrainer.threadLocalLogger.get().info("possibleBlockers: " + possibleBlockers);
//
//            attackers = filterOutUnblockable(game, attackers, possibleBlockers);
//            if (attackers.isEmpty()) {
//                return;
//            }
//
//            currentState = StateSequenceBuilder.build(game,
//                                                      StateSequenceBuilder.ActionType.DECLARE_BLOCKS,
//                                                      game.getPhase().getType(),
//                                                      StateSequenceBuilder.MAX_LEN);
//            stateBuffer.add(currentState);
//            // -1 to reserve the option to not block nothing no a creature. Essentially an attacker that is "nothing"
//            int numAttackers = Math.min(RLModel.MAX_ACTIONS - 1, attackers.size());
//            int numBlockers = Math.min(RLModel.MAX_OPTIONS, possibleBlockers.size());
//            if (attackers.size() > RLModel.MAX_ACTIONS - 1) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
//            }
//            if (possibleBlockers.size() > RLModel.MAX_OPTIONS) {
//                RLTrainer.threadLocalLogger.get().error("ERROR: More blockers than max actions, Model truncating");
//            }
//
//            // Build exploration dimensions
//            // +1 to explore the option to not block
//            for(int i = 0; i < numAttackers + 1; i++){
//                // exploration metadata skipped
//            }
//            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
//
//            boolean blockerDeclared = false;
//
//            // Iterate over blockers first
//            // Attacker = X, Blockers = Y
//            for (int blockerIndex = 0; blockerIndex < numBlockers; blockerIndex++) {
//                Permanent blocker = possibleBlockers.get(blockerIndex);
//
//                // Create a list of blocker indices with their Q-values for this attacker
//                List<BlockOption> blockOptions = new ArrayList<>();
//                // We use the full MAX_OPTIONS because we need to reserve the option to not block
//                for (int attackerIndex = 0; attackerIndex < RLModel.MAX_ACTIONS; attackerIndex++) {
//                    float qValue = qValues.getFloat(attackerIndex, blockerIndex);
//                    blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
//                }
//
//                // Sort block options by Q-value in descending order
//                blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
//
//                // Declare blocks based on sorted Q-values
//                for (BlockOption option : blockOptions) {
//                    if (option.attackerIndex >= numAttackers) {
//                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this blocker if the first choice is to not block
//                    }
//
//                    Permanent attacker = attackers.get(option.attackerIndex);
//                    if (blocker.canBlock(attacker.getId(), game)) {
//                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
//                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
//                        int index = option.attackerIndex * numBlockers + option.blockerIndex;
//                        stateBuffer.add(new StateSequenceBuilder.SequenceOutput(currentState.sequence, currentState.mask, index));
//                        break; // Skip this blocker if the first choice is to not block
//                    }
//                }
//            }
//            if (blockerDeclared) {
//                game.getPlayers().resetPassed();
//            }
//            // skip training metadata cleanup
//        }
//    }
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

    // TODO: Implement
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
            if (!this.activateAbility(ability, game)) {
                //TODO: if we are here it is because the ComputerPlayerRL chose an invalid subaction (choose likely)
                throw new RuntimeException("Failed to activate ability: " + ability);
            }
            //TODO: Implement holding priority for abilities that don't use the stack
            if (ability.isUsesStack()) {
                pass(game);
            }
        }
    }

    protected Ability calculateActions(Game game) {
        List<ActivatedAbility> flattenedOptions = getPlayable(game, true);
        flattenedOptions.add(new PassAbility());

        // Remove duplicate spell abilities with the same name
        List<ActivatedAbility> uniqueOptions = new ArrayList<>();
        Set<String> seenNames = new HashSet<>();

        // Remove duplicate spell abilities with the same name
        // TODO: Investigate if this is what we want. I did this because despite "setting targets" during selection. we still get prompted for choices later anyway
        for (ActivatedAbility ability : flattenedOptions) {
            String name = ability.toString();
            if (!seenNames.contains(name)) {
                seenNames.add(name);
                uniqueOptions.add(ability);
            }
        }
        flattenedOptions = uniqueOptions;

        // Get model's choice of actions
        List<Integer> targetsToSet = genericChoose(flattenedOptions.size(), 1, 1, StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL, game, null);

        RLTrainer.threadLocalLogger.get().info("Playable options: " + flattenedOptions);

        // Return the ability corresponding to the best index
        return flattenedOptions.get(targetsToSet.get(0));
    }

    public List<StateSequenceBuilder.TrainingData> getTrainingBuffer() {
        return new ArrayList<>(trainingBuffer);
    }

    public PythonMLBridge getModel() {
        return model;
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
