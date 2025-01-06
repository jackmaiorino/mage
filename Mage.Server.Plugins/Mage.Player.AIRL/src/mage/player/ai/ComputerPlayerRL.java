package mage.player.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;

import mage.abilities.ActivatedAbility;
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

public class ComputerPlayerRL extends ComputerPlayer6 {
    public RLModel model;
    protected RLState currentState;
    private final List<RLState> stateBuffer;

    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill, RLModel model) {
        super(name, range, skill);
        this.model = model;
        this.stateBuffer = new ArrayList<>();
        RLTrainer.threadLocalLogger.get().info("ComputerPlayerRL initialized for " + name);
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
            if (possibleAttackers.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }

            int numAttackers = currentState.exploreYCol = Math.min(RLModel.MAX_ACTIONS, possibleAttackers.size());
            // Generate list of attack targets (Player, planeswalkers, battles)
            List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
            if (possibleAttackTargets.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attack targets than max actions, Model truncating");
            }
            int numAttackTargets = currentState.exploreXCol = Math.min(RLModel.MAX_ACTIONS, possibleAttackTargets.size());

            // Predict on game state
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_ACTIONS + 1);

            // For each attacker
            for (int attackerIndex = 0; attackerIndex < numAttackers; attackerIndex++) {
                Permanent attacker = possibleAttackers.get(attackerIndex);

                // Create a list of defender indices with their Q-values for this attacker
                List<AttackOption> attackOptions = new ArrayList<>();
                // +1 to reserve the option to not attack
                for (int defenderIndex = 0; defenderIndex < RLModel.MAX_ACTIONS + 1; defenderIndex++) {
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
            int numBlockers = currentState.exploreYCol = Math.min(RLModel.MAX_ACTIONS, possibleBlockers.size());
            int numAttackers = currentState.exploreXCol = Math.min(RLModel.MAX_ACTIONS, attackers.size());
            INDArray qValues = model.predictDistribution(currentState, true).reshape(RLModel.MAX_ACTIONS, RLModel.MAX_ACTIONS + 1);
            
            if (possibleBlockers.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More blockers than max actions, Model truncating");
            }
            if (attackers.size() > RLModel.MAX_ACTIONS) {
                RLTrainer.threadLocalLogger.get().error("ERROR: More attackers than max actions, Model truncating");
            }

            boolean blockerDeclared = false;
            
            for (int blockerIndex = 0; blockerIndex < numBlockers; blockerIndex++) {
                Permanent blocker = possibleBlockers.get(blockerIndex);

                // Create a list of attacker indices with their Q-values for this blocker
                List<BlockOption> blockOptions = new ArrayList<>();
                for (int attackerIndex = 0; attackerIndex < RLModel.MAX_ACTIONS + 1; attackerIndex++) {
                    float qValue = qValues.getFloat(blockerIndex, attackerIndex);
                    blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
                }

                // Sort block options by Q-value in descending order
                blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
    
                // Declare blocks based on sorted Q-values
                for (BlockOption option : blockOptions) {
                    if (option.attackerIndex >= numAttackers) {
                        // TODO: Investigate, is storing this "Pass value" correct?
                        currentState.targetQValues.add(new QValueEntry(option.qValue, blockerIndex, option.attackerIndex));
                        break; // Skip this blocker if the first choice is to not block
                    }
                    Permanent attacker = attackers.get(option.attackerIndex);
                    if (blocker.canBlock(attacker.getId(), game)) {
                        RLTrainer.threadLocalLogger.get().info("Declaring blocker: " + blocker.getName() + " for attacker: " + attacker.getName());
                        this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                        currentState.targetQValues.add(new QValueEntry(option.qValue, blockerIndex, option.attackerIndex));
                        blockerDeclared = true;
                        // TODO: Implement creatures that can multiblock
                        break;
                        // Remove the attacker if it can't be blocked anymore
                        // if (!attacker.canBeBlocked(game)) {
                        //     attackers.remove(option.attackerIndex);
                        // }
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
        ActivatedAbility ability;
        switch (game.getTurnStepType()) {
            case UPKEEP:
            case DRAW:
                pass(game);
                return false;
            case PRECOMBAT_MAIN:
                currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
                stateBuffer.add(currentState);
                printBattleField(game, "Sim PRIORITY on MAIN 1");
                do {
                    ability = calculateActions(game);
                    act(game, ability);
                } while (ability != null);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                currentState = new RLState(game, RLState.ActionType.DECLARE_ATTACKS);
                stateBuffer.add(currentState);
                if (game.isActivePlayer(playerId)) {    
                    printBattleField(game, "Sim PRIORITY on DECLARE ATTACKERS");
                    selectAttackers(game, playerId);
                }
                // TODO: We can also perform actions here but lets simplify for now
                pass(game);
                //act(game);
                return true;
            case DECLARE_BLOCKERS:
                currentState = new RLState(game, RLState.ActionType.DECLARE_BLOCKS);
                stateBuffer.add(currentState);

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
                currentState = new RLState(game, RLState.ActionType.ACTIVATE_ABILITY_OR_SPELL);
                stateBuffer.add(currentState);
                printBattleField(game, "Sim PRIORITY on MAIN 2");
                do {
                    ability = calculateActions(game);
                    act(game, ability);
                    printBattleField(game, "Sim PRIORITY on MAIN 2");
                } while (ability != null);
                return true;
            case END_TURN:  
            case CLEANUP:
                actionCache.clear();
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
            pass(game);
        }
    }

    protected ActivatedAbility calculateActions(Game game) {
        boolean isSimulatedPlayer = true;
        // Returns a list of all available spells and abilities the player can currently cast/activate with his available resources.
        // Without target validation.
        List<ActivatedAbility> playables = game.getPlayer(playerId).getPlayable(game, isSimulatedPlayer);

        // Set the exploration columns so it only selects playables
        currentState.exploreXCol = 1;
        currentState.exploreYCol = Math.min(RLModel.MAX_ACTIONS, playables.size());
        // Evaluate each action using the model
        INDArray qValues = model.predictDistribution(currentState, true);
        float maxQValue = Float.NEGATIVE_INFINITY;
        int bestIndex = 0;
        
        // Find index with highest Q-value
        long max = qValues.data().length();
        for (int i = 0; i < max; i++) {
            if (qValues.getFloat(i) > maxQValue) {
                maxQValue = qValues.getFloat(i);
                bestIndex = i;
            }
        }

        RLTrainer.threadLocalLogger.get().info("bestIndex: " + bestIndex);
        RLTrainer.threadLocalLogger.get().info("playables: " + playables);

        // Convert 1D index to 2D coordinates on 10x11 matrix
        int row = bestIndex / (RLModel.MAX_ACTIONS + 1); 
        int col = bestIndex % (RLModel.MAX_ACTIONS + 1);
        currentState.targetQValues.add(new QValueEntry(maxQValue, row, col));

        // Get the corresponding ability and add it to actions queue
        // -1 on range is to reserve the option to do nothing
        if (bestIndex < Math.min(RLModel.OUTPUT_SIZE - 1, playables.size())) {
            return playables.get(bestIndex);
        }else{
            //Model chose to pass
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