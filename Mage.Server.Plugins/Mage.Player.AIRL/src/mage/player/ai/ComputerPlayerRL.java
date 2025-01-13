package mage.player.ai;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import mage.abilities.*;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.filter.common.FilterLandCard;
import mage.target.TargetCard;
import org.nd4j.linalg.api.ndarray.INDArray;

import mage.MageObject;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.mana.GenericManaCost;
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

public class ComputerPlayerRL extends ComputerPlayer {
    public RLModel model;
    protected RLState currentState;
    private final List<RLState> stateBuffer;

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

    // Stuff like Opp agent? Investigate further how to handle. Just choosing how to handle multiple replacement effects?
    // TODO
//    @Override
//    public int chooseReplacementEffect(Map<String, String> effectsMap, Map<String, MageObject> objectsMap, Game game) {
//        log.debug("chooseReplacementEffect");

    // Stuff like sheoldred's edict (I think this can just intercept a set mode. Same with set mana cost.)
    // TODO: How do we preset modes and manacost.
    // TODO
//    @Override
//    public Mode chooseMode(Modes modes, Ability source, Game game) {
//        log.debug("chooseMode");

    // TODO
//    @Override
//    public int announceXMana(int min, int max, String message, Game game, Ability ability) {
//        log.debug("announceXMana");

    // Deciding to use FOW alt cast
    // TODO
//    @Override
//    public boolean choose(Outcome outcome, Choice choice, Game game) {
//        log.debug("choose 3");

    // Deciding ponder cards
    // TODO
//    @Override
//    public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
//        log.debug("choose 2");

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
    // TODO
//    @Override
//    public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
//        log.debug("chooseTriggeredAbility: " + abilities.toString());
//        //TODO: improve this
//        if (!abilities.isEmpty()) {
//            return abilities.get(0);
//        }
//        return null;
//    }

    @Override
    public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        return choose(outcome, target, source, game, null);
    }

    // Will this work?
    // TODO: This breaks on mulligans? Because there is no active player?
    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        return choose(outcome, target, source, game, null);
    }

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
        UUID sourceId = source != null ? source.getSourceId() : null;

        // Not really sure why we are prompted to choose the starting player
        // Just hand it off to the superclass
        if (Objects.equals(target.getTargetName(), "starting player")) {
            return super.choose(outcome, target, source, game, null);
        }

        boolean required = target.isRequired(sourceId, game);
        List<UUID> possibleTargetsList = new ArrayList<>(target.possibleTargets(abilityControllerId, source, game));
        if (possibleTargetsList.isEmpty() || target.getTargets().size() >= target.getNumberOfTargets()) {
            required = false;
        }

        // If we can't choose any targets, pass
        if (possibleTargetsList.isEmpty()) {
            return false;
        }

        int maxTargets = target.getMaxNumberOfTargets();
        int minTargets = target.getMinNumberOfTargets();

        // TODO: Differentiate between ACTIVATE_ABILITY_OR_SPELL and CHOOSE_TARGET
        currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
        stateBuffer.add(currentState);

        // Initialize explore dimensions as a 2D list
        List<Integer> exploreDimensions = new ArrayList<>();
        int totalTargets = possibleTargetsList.size();
        int fullRows = totalTargets / RLModel.MAX_OPTIONS;
        int remainingTargets = totalTargets % RLModel.MAX_OPTIONS;

        // Fill the 2D list with indices
        for (int i = 0; i < fullRows; i++) {
            exploreDimensions.add(RLModel.MAX_OPTIONS);
        }
        if (remainingTargets > 0) {
            exploreDimensions.add(remainingTargets);
        }

        currentState.exploreDimensions = exploreDimensions;

        INDArray qValues = model.predictDistribution(currentState, true);

        // Create a list to store Q-values with their indices
        List<QValueWithIndex> qValueList = new ArrayList<>();
        for (int i = 0; i < qValues.length(); i++) {
            qValueList.add(new QValueWithIndex(qValues.getFloat(i), i));
        }

        // Sort the list by Q-value in descending order
        qValueList.sort((a, b) -> Float.compare(b.qValue, a.qValue));

        // Use the sorted Q-values and their indices as needed
        int selectedTargets = 0;
        List<UUID> selectedTargetsList = new ArrayList<>();
        for (QValueWithIndex qValueWithIndex : qValueList) {
            // Access the original index and Q-value
            int originalIndex = qValueWithIndex.index;

            // Check if the index is valid
            if (originalIndex < possibleTargetsList.size()) {
                target.add(possibleTargetsList.get(originalIndex), game);
                // Convert 1D index to 2D index
                currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, originalIndex % RLModel.MAX_ACTIONS, originalIndex / RLModel.MAX_ACTIONS));
                selectedTargetsList.add(possibleTargetsList.get(originalIndex));
                selectedTargets++;
            }

            // Stop if we've reached the minimum and encounter an invalid index
            if (selectedTargets >= minTargets && originalIndex >= possibleTargetsList.size()) {
                break;
            }

            // Stop if we've reached the maximum number of targets
            if (selectedTargets >= maxTargets && maxTargets != 0) {
                break;
            }
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

        if (currentState.targetQValues.isEmpty()){
            // If no targets are selected, but minTargets is 0, the selected action was the "non-action"
            if (minTargets == 0) {
                QValueWithIndex qValueWithIndex = qValueList.get(0);
                int originalIndex = qValueWithIndex.index;
                currentState.targetQValues.add(new QValueEntry(qValueWithIndex.qValue, originalIndex % RLModel.MAX_ACTIONS, originalIndex / RLModel.MAX_ACTIONS));
            }else{
                stateBuffer.remove(currentState);
            }
        }
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
                        currentState.targetQValues.add(new QValueEntry(option.qValue, attackerIndex, option.attackTargetIndex));
                        break; // Skip this attacker if the first choice is to not attack
                    }
                    UUID attackTargetId = possibleAttackTargets.get(option.attackTargetIndex);
                    if (attacker.canAttack(attackTargetId, game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring attacker: " + attacker.getName() + " for attack target: " + attackTargetId.toString());
                        this.declareAttacker(attacker.getId(), attackTargetId, game, false);
                        currentState.targetQValues.add(new QValueEntry(option.qValue, attackerIndex, option.attackTargetIndex));
                        break; // Once an attack is declared, move to the next attacker
                    }
                }
            }
        }
        if (currentState.targetQValues.isEmpty()){
            stateBuffer.remove(currentState);
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
                        currentState.targetQValues.add(new QValueEntry(option.qValue, option.attackerIndex, option.blockerIndex));
                        break; // Skip this blocker if the first choice is to not block
                    }
                    Permanent attacker = attackers.get(option.attackerIndex);
                    if (blocker.canBlock(attacker.getId(), game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                        currentState.targetQValues.add(new QValueEntry(option.qValue, option.attackerIndex, option.blockerIndex));
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
                ability = calculateActions(game);
                act(game, (ActivatedAbility) ability);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                ability = calculateActions(game);
                act(game, (ActivatedAbility) ability);
                pass(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattleField(game, "Sim PRIORITY on DECLARE BLOCKERS");
                ability = calculateActions(game);
                act(game, (ActivatedAbility) ability);
                pass(game);
                return true;
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
            case END_COMBAT:
                pass(game);
                return false;
            case POSTCOMBAT_MAIN:
                printBattleField(game, "Sim PRIORITY on MAIN 2");
                ability = calculateActions(game);
                act(game, (ActivatedAbility) ability);
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
            this.activateAbility(ability, game);
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
        if (allOptions.size() > RLModel.MAX_ACTIONS) {
            RLTrainer.threadLocalLogger.get().error("ERROR: More actions than max actions, Model truncating");
            allOptions = allOptions.subList(0, RLModel.MAX_ACTIONS);
        }
        for (int i = 0; i < allOptions.size(); i++) {
            List<Ability> optionList = allOptions.get(i);


            if (optionList.size() > RLModel.MAX_OPTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More options than max options, Model truncating");
                allOptions.set(i, optionList.subList(0, RLModel.MAX_OPTIONS));
            }
        }

        return allOptions;
    }

    private List<Ability> simulateVariableCosts(Ability ability, Game game) {
        List<Ability> options = new ArrayList<>();
        int numAvailable = getAvailableManaProducers(game).size() - ability.getManaCosts().manaValue();
        int start = 0;
        if (!(ability instanceof SpellAbility)) {
            if (numAvailable == 0) {
                return options;
            } else {
                start = 1;
            }
        }
        for (int i = start; i < numAvailable; i++) {
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

    protected Ability calculateActions(Game game) {
        // Get the 2D list of playable options
        List<List<Ability>> playableOptions = getPlayableOptions(game);
        // If we can only pass, don't query model
        if (playableOptions.size() == 1){
            return null;
        }

        // Set the exploration columns so it only selects playableOptions
        currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
        stateBuffer.add(currentState);
        for (List<Ability> options : playableOptions) {
            currentState.exploreDimensions.add(options.size());
        }

        // Evaluate each action using the model
        INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);
        float maxQValue = Float.NEGATIVE_INFINITY;
        int bestRowIndex = 0;
        int bestColIndex = 0;

        // Find the index with the highest Q-value
        for (int i = 0; i < RLModel.MAX_ACTIONS; i++) {
            for (int j = 0; j < RLModel.MAX_OPTIONS; j++) {
                float qValue = qValues.getFloat(i, j);
                if (qValue > maxQValue) {
                    maxQValue = qValue;
                    bestRowIndex = i;
                    bestColIndex = j;
                }
            }
        }

        currentState.targetQValues.add(new QValueEntry(maxQValue, bestRowIndex, bestColIndex));
        RLTrainer.threadLocalLogger.get().info("Best action index: (" + bestRowIndex + ", " + bestColIndex + ")");
        RLTrainer.threadLocalLogger.get().info("Playable options: " + playableOptions);

        // Check if the selected index is valid and return the corresponding ability
        if (bestRowIndex < playableOptions.size() && bestColIndex < playableOptions.get(bestRowIndex).size()) {
            return playableOptions.get(bestRowIndex).get(bestColIndex);
        } else {
            // Masking invalid choice to passing
            return null;
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