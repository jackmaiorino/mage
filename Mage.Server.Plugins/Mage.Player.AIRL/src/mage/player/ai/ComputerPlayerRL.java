package mage.player.ai;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
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
import mage.target.TargetAmount;
import mage.target.TargetCard;

public class ComputerPlayerRL extends ComputerPlayer7 {

    private PythonMLBridge model;
    protected StateSequenceBuilder.SequenceOutput currentState;
    private final List<StateSequenceBuilder.TrainingData> trainingBuffer;
    private Ability currentAbility;

    /**
     * If true, the agent will act greedily (arg-max) instead of sampling.
     * Useful for evaluation where we want a deterministic policy.
     */
    private final boolean greedyMode;

    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonMLBridge model) {
        this(name, range, model, false);
    }

    // Convenience ctor with greedy flag
    public ComputerPlayerRL(String name, RangeOfInfluence range, PythonMLBridge model, boolean greedy) {
        super(name, range, 10);
        this.model = model;
        this.trainingBuffer = new ArrayList<>();
        this.greedyMode = greedy;
        // Auto-targeting disabled via getStrictChooseMode override
        RLTrainer.threadLocalLogger.get().info("ComputerPlayerRL initialized for " + name + (greedy ? " [GREEDY]" : ""));
    }

    // The default constructor for ComputerPlayerRL used by server to create
    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill) {
        this(name, range, PythonMLBridge.getInstance(), false);
    }

    public ComputerPlayerRL(final ComputerPlayerRL player) {
        super(player);
        this.model = player.model;
        this.trainingBuffer = new ArrayList<>();
        this.currentAbility = player.currentAbility;
        this.greedyMode = player.greedyMode;
        // strict choose mode enforced via method override
    }

    @Override
    public ComputerPlayerRL copy() {
        return new ComputerPlayerRL(this);
    }

    @Override
    public boolean priority(Game game) {
        game.resumeTimer(getTurnControlledBy());
        boolean result;
        try {
            result = priorityPlay(game);
        } catch (Throwable t) {
            // Never let RL decision logic crash the game engine.
            // A single bad activation/choice can otherwise trigger "too many errors" and end the whole game.
            try {
                RLTrainer.threadLocalLogger.get().warn("RL priority() caught exception; forcing pass: " + t.getMessage());
            } catch (Exception ignored) {
                // ignore
            }
            pass(game);
            result = false;
        }
        game.pauseTimer(getTurnControlledBy());
        return result;
    }

    public <T> List<Integer> genericChoose(List<T> candidates, int maxTargets, int minTargets, StateSequenceBuilder.ActionType actionType, Game game, Ability source) {
        // Candidate-based policy: score up to MAX_CANDIDATES candidates per decision.
        final int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        final int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

        int candidateCount = Math.min(candidates.size(), maxCandidates);
        if (candidates.size() > maxCandidates) {
            RLTrainer.threadLocalLogger.get().warn(
                    "genericChoose: received " + candidates.size() + " options, truncating to " + maxCandidates);
        }

        // Prevent infinite loops if the engine requests more picks than we kept after truncation
        maxTargets = Math.min(maxTargets, candidateCount);
        minTargets = Math.min(minTargets, candidateCount);

        if (candidateCount == 1) {
            return Arrays.asList(0);
        } else if (candidateCount == 0) {
            return Arrays.asList();
        }

        // Get current phase
        TurnPhase turnPhase = game.getPhase() != null ? game.getPhase().getType() : null;

        // Build base state and cache as current
        StateSequenceBuilder.SequenceOutput baseState = StateSequenceBuilder.buildBaseState(game, turnPhase, StateSequenceBuilder.MAX_LEN);
        this.currentState = baseState;

        // Build padded candidate tensors
        int[] candidateActionIds = new int[maxCandidates];
        float[][] candidateFeatures = new float[maxCandidates][candFeatDim];
        int[] candidateMask = new int[maxCandidates]; // 1 valid, 0 padding

        for (int i = 0; i < candidateCount; i++) {
            candidateMask[i] = 1;
            T cand = candidates.get(i);
            candidateActionIds[i] = computeCandidateActionId(actionType, game, source, cand);
            candidateFeatures[i] = computeCandidateFeatures(actionType, game, source, cand, candFeatDim);
        }

        // Get model predictions - single call for both policy and value
        RLTrainer.threadLocalLogger.get().info("About to call model.scoreCandidates() with candidateCount: " + candidateCount);
        mage.player.ai.rl.PythonMLBatchManager.PredictionResult prediction
                = model.scoreCandidates(baseState, candidateActionIds, candidateFeatures, candidateMask);
        RLTrainer.threadLocalLogger.get().info("Successfully received candidate scoring result");
        float[] actionProbs = prediction.policyScores; // length = maxCandidates
        float valueScore = prediction.valueScores;

        // Apply mask and normalize over valid candidates only
        float[] maskedProbs = new float[candidateCount];

        // Apply mask and sum up probabilities
        float sum = 0.0f;
        for (int i = 0; i < candidateCount; i++) {
            float prob = actionProbs[i];
            // Handle numerical stability issues
            if (Float.isNaN(prob) || Float.isInfinite(prob)) {
                prob = 0.0f;
            }
            maskedProbs[i] = prob * (candidateMask[i] == 1 ? 1.0f : 0.0f);
            sum += maskedProbs[i];
        }

        // Normalize probabilities
        if (sum > 0) {
            for (int i = 0; i < candidateCount; i++) {
                maskedProbs[i] /= sum;
            }
        } else {
            // If all probabilities are 0, use uniform distribution
            float uniformProb = 1.0f / candidateCount;
            for (int i = 0; i < candidateCount; i++) {
                maskedProbs[i] = uniformProb;
            }
        }

        RLTrainer.threadLocalLogger.get().info("Action probabilities: " + Arrays.toString(maskedProbs));
        RLTrainer.threadLocalLogger.get().info("Value score: " + valueScore);

        // Choose indices ---------------------------------------------
        List<Integer> selectedIndices = new ArrayList<>();
        Random random = new Random();
        if (greedyMode) {
            // Deterministic arg-max
            int bestIdx = 0;
            float bestProb = maskedProbs[0];
            for (int i = 1; i < candidateCount; i++) {
                if (maskedProbs[i] > bestProb) {
                    bestProb = maskedProbs[i];
                    bestIdx = i;
                }
            }
            selectedIndices.add(bestIdx);
        } else {
            // Stochastic sampling as before
            float[] cumulativeProbs = new float[candidateCount];
            cumulativeProbs[0] = maskedProbs[0];
            for (int i = 1; i < candidateCount; i++) {
                cumulativeProbs[i] = cumulativeProbs[i - 1] + maskedProbs[i];
            }

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
        }

        // Ensure we have at least minTargets
        while (selectedIndices.size() < minTargets) {
            int remaining = candidateCount - selectedIndices.size();
            if (remaining <= 0) {
                break;
            }

            // Find indices not yet selected
            List<Integer> available = new ArrayList<>();
            for (int i = 0; i < candidateCount; i++) {
                if (!selectedIndices.contains(i)) {
                    available.add(i);
                }
            }

            // Randomly select from remaining indices
            if (!available.isEmpty()) {
                selectedIndices.add(available.get(random.nextInt(available.size())));
            }
        }

        // after you compute logits/probs and make a choice
        // Record training only for simple single-choice decisions for now.
        if (minTargets == 1 && maxTargets == 1 && !selectedIndices.isEmpty()) {
            int chosen = selectedIndices.get(0);
            trainingBuffer.add(new StateSequenceBuilder.TrainingData(
                    baseState,
                    candidateCount,
                    candidateActionIds,
                    candidateFeatures,
                    candidateMask,
                    chosen,
                    actionType));
        }

        return selectedIndices;
    }

    private static int toVocabId(String key) {
        if (key == null || key.isEmpty()) {
            return 0;
        }
        int h = key.hashCode();
        int mod = Math.floorMod(h, StateSequenceBuilder.TOKEN_ID_VOCAB - 1);
        return 1 + mod;
    }

    private int computeCandidateActionId(StateSequenceBuilder.ActionType actionType, Game game, Ability source, Object candidate) {
        try {
            String base = actionType.name();
            if (candidate instanceof PassAbility) {
                return toVocabId("PASS");
            }
            if (candidate instanceof mage.abilities.Ability) {
                mage.abilities.Ability ab = (mage.abilities.Ability) candidate;
                MageObject srcObj = game.getObject(ab.getSourceId());
                String srcName = srcObj != null ? srcObj.getName() : "unknown";
                base += ":ABILITY:" + srcName + ":" + ab.getClass().getSimpleName();
            } else if (candidate instanceof java.util.UUID) {
                java.util.UUID tid = (java.util.UUID) candidate;
                MageObject obj = game.getObject(tid);
                if (obj != null) {
                    base += ":TARGET:" + obj.getName();
                } else if (game.getPlayer(tid) != null) {
                    base += ":TARGET:PLAYER";
                } else {
                    base += ":TARGET:UNKNOWN";
                }
            } else if (candidate != null) {
                base += ":" + candidate.getClass().getSimpleName();
            }
            return toVocabId(base);
        } catch (Exception e) {
            return 0;
        }
    }

    private float[] computeCandidateFeatures(StateSequenceBuilder.ActionType actionType, Game game, Ability source, Object candidate, int dim) {
        float[] f = new float[dim];
        try {
            // Generic context
            f[0] = actionType.ordinal() / 16.0f;

            if (candidate instanceof PassAbility) {
                f[1] = 1.0f; // is_pass
                return f;
            }

            // Ability-based candidate features
            if (candidate instanceof mage.abilities.Ability) {
                mage.abilities.Ability ab = (mage.abilities.Ability) candidate;
                MageObject srcObj = game.getObject(ab.getSourceId());
                if (srcObj instanceof Permanent) {
                    Permanent p = (Permanent) srcObj;
                    f[2] = p.isCreature() ? 1.0f : 0.0f;
                    f[3] = p.isLand() ? 1.0f : 0.0f;
                    f[4] = p.isTapped() ? 1.0f : 0.0f;
                    f[5] = p.getPower().getValue() / 10.0f;
                    f[6] = p.getToughness().getValue() / 10.0f;
                }
                // rough target count
                f[7] = ab.getTargets() != null ? ab.getTargets().size() / 5.0f : 0.0f;
                f[8] = ab.isUsesStack() ? 1.0f : 0.0f;
            }

            // Target candidate features
            if (candidate instanceof java.util.UUID) {
                java.util.UUID tid = (java.util.UUID) candidate;
                Player pl = game.getPlayer(tid);
                if (pl != null) {
                    f[10] = 1.0f; // is_player
                    f[11] = pl.getId().equals(this.getId()) ? 1.0f : 0.0f; // is_you
                    f[12] = pl.getLife() / 20.0f;
                } else {
                    Permanent perm = game.getPermanent(tid);
                    if (perm != null) {
                        f[13] = 1.0f; // is_permanent
                        f[14] = perm.isCreature() ? 1.0f : 0.0f;
                        f[15] = perm.isTapped() ? 1.0f : 0.0f;
                        f[16] = perm.getPower().getValue() / 10.0f;
                        f[17] = perm.getToughness().getValue() / 10.0f;
                    }
                }
            }
        } catch (Exception e) {
            // leave zeros
        }
        return f;
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
    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        // Mulligan and some game setup flows call chooseTarget with a null source;
        // defer to base implementation until we add RL support for those cases.
        if (source == null) {
            return super.chooseTarget(outcome, target, source, game);
        }
        // Determine which player must choose the targets (abilityController may differ)
        UUID abilityControllerId = playerId;
        if (target.getTargetController() != null && target.getAbilityController() != null) {
            abilityControllerId = target.getAbilityController();
        }

        // Build list of possible targets that are legal
        List<UUID> possibleTargetsList = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
        final UUID ctrlId = abilityControllerId; // make effectively final for lambda
        possibleTargetsList.removeIf(id -> !target.canTarget(ctrlId, id, source, game));

        // Reorder targets deterministically: self first (index 0), primary opponent second (index 1)
        if (!possibleTargetsList.isEmpty()) {
            UUID selfId = this.getId();
            UUID primaryOpponentId = null;
            if (!game.getOpponents(selfId).isEmpty()) {
                primaryOpponentId = game.getOpponents(selfId).iterator().next();
            }

            List<UUID> reordered = new ArrayList<>(possibleTargetsList.size());
            if (possibleTargetsList.contains(selfId)) {
                reordered.add(selfId);
            }
            if (primaryOpponentId != null && possibleTargetsList.contains(primaryOpponentId)) {
                reordered.add(primaryOpponentId);
            }
            for (UUID tid : possibleTargetsList) {
                if (!reordered.contains(tid)) {
                    reordered.add(tid);
                }
            }
            possibleTargetsList = reordered;
        }

        // Log all possible legal targets before the RL decision, similar to ability logging
        if (RLTrainer.threadLocalLogger.get().isInfoEnabled()) {
            StringBuilder sb = new StringBuilder("Possible targets: ");
            for (int i = 0; i < possibleTargetsList.size(); i++) {
                UUID tid = possibleTargetsList.get(i);
                sb.append("[").append(i).append("] ").append(describeTargetWithOwner(tid, game));
                if (i < possibleTargetsList.size() - 1) {
                    sb.append("; ");
                }
            }
            RLTrainer.threadLocalLogger.get().info(sb.toString());
        }
        // If engine thinks no choice is required, fall back to original behaviour
        if (possibleTargetsList.isEmpty()) {
            return super.chooseTarget(outcome, target, source, game);
        }

        int maxTargets = Math.min(target.getMaxNumberOfTargets(), possibleTargetsList.size());
        int minTargets = target.getMinNumberOfTargets();

        // If only one option or zero required, just pick automatically
        if (possibleTargetsList.size() == 1 || maxTargets == 0) {
            target.addTarget(possibleTargetsList.get(0), source, game);
        } else {
            // Ask the RL policy which indices to choose
            List<Integer> chosenIdx = genericChoose(possibleTargetsList, maxTargets, minTargets,
                    StateSequenceBuilder.ActionType.SELECT_TARGETS, game, source);
            for (int idx : chosenIdx) {
                target.addTarget(possibleTargetsList.get(idx), source, game);
            }
        }

        // Logging
        for (UUID targetId : target.getTargets()) {
            String targetName = describeTargetWithOwner(targetId, game);
            RLTrainer.threadLocalLogger.get().info(
                    "Player " + getName() + " chose target: " + targetName + " (" + targetId + ")"
                    + " for ability: " + (source != null ? source.toString() : "null source")
            );
        }
        return true;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        boolean result = super.choose(outcome, target, source, game);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().info(
                        "Player " + getName() + " chose target (from choose): " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        boolean result = super.chooseTarget(outcome, cards, target, source, game);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().info(
                        "Player " + getName() + " chose target from cards: " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        boolean result = super.choose(outcome, choice, game);
        if (result && choice.isChosen()) {
            RLTrainer.threadLocalLogger.get().info(
                    "Player " + getName() + " chose (from Choice): " + choice.getChoiceKey() + " -> " + choice.getChoice()
            );
        }
        return result;
    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        boolean result = super.chooseTargetAmount(outcome, target, source, game);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                int amount = target.getTargetAmount(targetId);
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().info(
                        "Player " + getName() + " chose target amount: " + targetName + " (" + targetId + "), amount: " + amount
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game, Map<String, Serializable> options) {
        boolean result = super.choose(outcome, target, source, game, options);
        if (result && !target.getTargets().isEmpty()) {
            for (UUID targetId : target.getTargets()) {
                String targetName = describeTargetWithOwner(targetId, game);
                RLTrainer.threadLocalLogger.get().info(
                        "Player " + getName() + " chose target (from choose with options): " + targetName + " (" + targetId + ")"
                        + " for ability: " + (source != null ? source.toString() : "null source")
                );
            }
        }
        return result;
    }

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
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                currentAbility = calculateRLAction(game);
                act(game, (ActivatedAbility) currentAbility);
                pass(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE BLOCKERS");
                currentAbility = calculateRLAction(game);
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
                currentAbility = calculateRLAction(game);
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
                // If we are here it is because the RL player chose an action that isn't actually executable
                // (often due to hidden costs/choices not captured in our candidate features).
                // Treat this as a forced pass rather than crashing the engine.
                RLTrainer.threadLocalLogger.get().warn("Failed to activate ability; forcing pass: " + ability);
                pass(game);
                return;
            }

            // Log all resolved targets for the ability (covers auto-chosen single targets)
            logAbilityTargets(ability, game);

            //TODO: Implement holding priority for abilities that don't use the stack
            if (ability.isUsesStack()) {
                pass(game);
            }
        }
    }

    /**
     * Logs every target that has been recorded on the provided ability. This is
     * called *after* the ability has been activated so it catches cases where
     * the engine auto-selected a single possible target (which bypasses the
     * usual choose/chooseTarget callbacks).
     */
    private void logAbilityTargets(Ability ability, Game game) {
        if (ability == null) {
            return;
        }

        boolean logged = false;

        for (Target tgt : ability.getTargets()) {
            if (!tgt.getTargets().isEmpty()) {
                for (UUID targetId : tgt.getTargets()) {
                    String targetName = describeTargetWithOwner(targetId, game);
                    RLTrainer.threadLocalLogger.get().info(
                            "Player " + getName() + " resolved target: " + targetName + " (" + targetId + ")"
                            + " for ability: " + ability.toString()
                    );
                    logged = true;
                }
            }
        }

        // Fallback: look at the newest stack object (in case targets were set on the stack copy)
        if (!logged && ability.isUsesStack() && !game.getStack().isEmpty()) {
            mage.game.stack.StackObject top = game.getStack().getFirst();
            if (top != null && top.getSourceId().equals(ability.getSourceId())) {
                for (Target tgt : top.getStackAbility().getTargets()) {
                    if (!tgt.getTargets().isEmpty()) {
                        for (UUID targetId : tgt.getTargets()) {
                            String targetName = describeTargetWithOwner(targetId, game);
                            RLTrainer.threadLocalLogger.get().info(
                                    "Player " + getName() + " resolved target (stack): " + targetName + " (" + targetId + ")"
                                    + " for ability: " + ability.toString()
                            );
                            logged = true;
                        }
                    }
                }
            }
        }
    }

    protected Ability calculateRLAction(Game game) {
        List<ActivatedAbility> flattenedOptions = getPlayable(game, true);
        // (PassAbility will be added later after duplicate removal)

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

        // Finally, ensure PassAbility maps to index 0
        flattenedOptions.add(0, new PassAbility());

        // Get model's choice of actions
        List<Integer> targetsToSet = genericChoose(flattenedOptions, 1, 1, StateSequenceBuilder.ActionType.ACTIVATE_ABILITY_OR_SPELL, game, null);

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

    // Disables engine auto-targeting heuristics by forcing strict choose mode
    @Override
    public boolean getStrictChooseMode() {
        return true;
    }

    private String describeTargetWithOwner(UUID targetId, Game game) {
        // Players: return their name (self/opponent distinction already obvious)
        Player player = game.getPlayer(targetId);
        if (player != null) {
            return player.getName();
        }

        MageObject obj = game.getObject(targetId);
        if (obj == null) {
            return "unknown target";
        }

        // Default base name is the card/permanent name
        StringBuilder sb = new StringBuilder(obj.getName());

        // If it's a permanent on the battlefield we can check controller
        if (obj instanceof Permanent) {
            UUID ctrl = ((Permanent) obj).getControllerId();
            if (ctrl != null) {
                if (ctrl.equals(this.getId())) {
                    sb.append(" (you)");
                } else {
                    Player ownerP = game.getPlayer(ctrl);
                    if (ownerP != null) {
                        sb.append(" (").append(ownerP.getName()).append(")");
                    }
                }
            }
        }
        return sb.toString();
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
