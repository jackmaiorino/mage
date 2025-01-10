package mage.player.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.mana.GenericManaCost;
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

            if (possibleAttackers.size() > RLModel.MAX_OPTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }

            currentState = new RLState(game, RLState.ActionType.DECLARE_ATTACKS);
            stateBuffer.add(currentState);
            int numAttackers = currentState.exploreYCol = Math.min(RLModel.MAX_OPTIONS, possibleAttackers.size());
            // Generate list of attack targets (Player, planeswalkers, battles)
            List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
            if (possibleAttackTargets.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attack targets than max actions, Model truncating");
            }
            int numAttackTargets = currentState.exploreXCol = Math.min(RLModel.MAX_ACTIONS, possibleAttackTargets.size());

            // Predict on game state
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);

            // For each attacker
            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
                Permanent attacker = possibleAttackers.get(attackerIndex);

                // Create a list of defender indices with their Q-values for this attacker
                List<AttackOption> attackOptions = new ArrayList<>();
                for (int defenderIndex = 0; defenderIndex < RLModel.MAX_OPTIONS; defenderIndex++) {
                    float qValue = qValues.getFloat(attackerIndex, defenderIndex);
                    attackOptions.add(new AttackOption(defenderIndex, attackerIndex, qValue));
                }

                // Sort attack options by Q-value in descending order
                attackOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));

                // Declare attacks based on sorted Q-values
                for (AttackOption option : attackOptions) {
                    if (option.defenderIndex >= numAttackTargets) {
                        currentState.targetQValues.add(new QValueEntry(option.qValue, attackerIndex, option.defenderIndex));
                        break; // Skip this attacker if the first choice is to not attack
                    }
                    UUID defenderId = possibleAttackTargets.get(option.defenderIndex);
                    if (attacker.canAttack(defenderId, game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring attacker: " + attacker.getName() + " for defender: " + defenderId.toString());
                        this.declareAttacker(attacker.getId(), defenderId, game, false);
                        currentState.targetQValues.add(new QValueEntry(option.qValue, attackerIndex, option.defenderIndex));
                        break; // Once an attack is declared, move to the next attacker
                    }
                }
            }
        }
    }

    // Don't need to override?
    // TODO: Do we need to pass the action vector a reference for WHICH creature its declaring blocks?
    private void declareBlockers(Game game, UUID activePlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, activePlayerId));
        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, activePlayerId, activePlayerId))) {
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
            int numAttackers = currentState.exploreXCol = Math.min(RLModel.MAX_ACTIONS, attackers.size());
            // -1 to reserve the option to not block
            int numBlockers = currentState.exploreYCol = Math.min(RLModel.MAX_OPTIONS - 1, possibleBlockers.size());
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_OPTIONS);

            if (attackers.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }
            if (possibleBlockers.size() > RLModel.MAX_OPTIONS - 1) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More blockers than max actions, Model truncating");
            }

            boolean blockerDeclared = false;

            // Iterate over attackers first
            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
                Permanent attacker = attackers.get(attackerIndex);

                // Create a list of blocker indices with their Q-values for this attacker
                List<BlockOption> blockOptions = new ArrayList<>();
                // We use the full MAX_OPTIONS because we need to reserve the option to not block
                for (int blockerIndex = 0; blockerIndex < RLModel.MAX_OPTIONS; blockerIndex++) {
                    float qValue = qValues.getFloat(attackerIndex, blockerIndex);
                    blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
                }

                // Sort block options by Q-value in descending order
                blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));

                // Declare blocks based on sorted Q-values
                for (BlockOption option : blockOptions) {
                    if (option.blockerIndex >= numBlockers) {
                        currentState.targetQValues.add(new QValueEntry(option.qValue, option.blockerIndex, option.attackerIndex));
                        break; // Skip this attacker if the first choice is to not block
                    }
                    Permanent blocker = possibleBlockers.get(option.blockerIndex);
                    if (blocker.canBlock(attacker.getId(), game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                        currentState.targetQValues.add(new QValueEntry(option.qValue, option.blockerIndex, option.attackerIndex));
                        blockerDeclared = true;
                        // TODO: implement multiblock
                        break;
                    }
                }
            }
            if (blockerDeclared) {
                game.getPlayers().resetPassed();
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
                if (game.isActivePlayer(playerId)) {
                    printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                    selectAttackers(game, playerId);
                }
                // TODO: We can also perform actions here but lets simplify for now
                pass(game);
                //act(game);
                return true;
            case DECLARE_BLOCKERS:
                if (!game.isActivePlayer(playerId)) {
                    printBattleField(game, "Sim PRIORITY on DECLARE BLOCKERS");
                    declareBlockers(game, playerId);
                }
                // TODO: We can also perform actions here but lets simplify for now
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

        // Ensure the list is of size RLModel.MAX_ACTIONS x RLModel.MAX_OPTIONS
        while (allOptions.size() < RLModel.MAX_ACTIONS) {
            allOptions.add(new ArrayList<>());
        }
        for (List<Ability> optionList : allOptions) {
            while (optionList.size() < RLModel.MAX_OPTIONS) {
                optionList.add(null); // or some placeholder if needed
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

        // Set the exploration columns so it only selects playableOptions
        currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
        stateBuffer.add(currentState);
        currentState.exploreXCol = Math.min(RLModel.MAX_ACTIONS, playableOptions.size());
        currentState.exploreYCol = Math.min(RLModel.MAX_OPTIONS, playableOptions.stream().mapToInt(List::size).max().orElse(0));

        // Evaluate each action using the model
        INDArray qValues = model.predictDistribution(currentState, true);
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
    int defenderIndex;
    int attackerIndex;
    float qValue;

    AttackOption(int defenderIndex, int attackerIndex, float qValue) {
        this.defenderIndex = defenderIndex;
        this.attackerIndex = attackerIndex;
        this.qValue = qValue;
    }
}