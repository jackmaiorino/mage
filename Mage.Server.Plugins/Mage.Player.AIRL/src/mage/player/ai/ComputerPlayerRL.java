package mage.player.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import mage.abilities.ActivatedAbility;
import mage.constants.RangeOfInfluence;
import mage.filter.StaticFilters;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.rl.Experience;
import mage.player.ai.rl.RLAction;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLState;
import mage.player.ai.util.CombatUtil;
import mage.target.Target;

public class ComputerPlayerRL extends ComputerPlayer6 {
    private static final Logger logger = Logger.getLogger(ComputerPlayerRL.class);
    public RLModel model;
    protected RLState currentState;
    protected RLAction currentAction;
    private final List<Experience> experienceBuffer;

    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill, RLModel model) {
        super(name, range, skill);
        this.model = model;
        this.experienceBuffer = new ArrayList<>();
        logger.info("ComputerPlayerRL initialized for " + name);
    }

    @Override
    public boolean priority(Game game) {
        logger.info("priority called for " + getName());
        game.resumeTimer(getTurnControlledBy());
        boolean result = priorityPlay(game);
        game.pauseTimer(getTurnControlledBy());
        return result;
    }

    //TODO: Implement ability to attack planeswalkers
    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        logger.info("selectAttackers called for " + getName());

        currentState = new RLState(game);
        List<Permanent> allAttackers = game.getBattlefield().getAllActivePermanents(
            StaticFilters.FILTER_PERMANENT_CREATURE,
            playerId,
            game
        );
        List<Permanent> possibleAttackers = new ArrayList<>();

        for (Permanent creature : allAttackers) {
            if (creature.canAttack(null, game)) {
                possibleAttackers.add(creature);
            }
        }
        currentAction = new RLAction(RLAction.ActionType.DECLARE_ATTACKS, null, possibleAttackers, game);
        INDArray qValues = model.predictDistribution(currentState, currentAction, true);

        if (possibleAttackers.size() > RLAction.MAX_ACTIONS) {
            logger.error("ERROR: More attackers than max actions, Model truncating");
        }

        List<UUID> possibleAttackTargets = new ArrayList<>(game.getCombat().getDefenders());
        if (possibleAttackTargets.size() > RLAction.MAX_ACTIONS) {
            logger.error("ERROR: More attack targets than max actions, Model truncating");
        }

        for (int attackerIndex = 0; attackerIndex < RLAction.MAX_ACTIONS; attackerIndex++) {
            Permanent attacker = possibleAttackers.get(attackerIndex);

            // Create a list of defender indices with their Q-values for this attacker
            List<AttackOption> attackOptions = new ArrayList<>();
            // +1 to reserve the option to not attack
            for (int defenderIndex = 0; defenderIndex < RLAction.MAX_ACTIONS + 1; defenderIndex++) {
                double qValue = qValues.getDouble(attackerIndex, defenderIndex);
                attackOptions.add(new AttackOption(defenderIndex, attackerIndex, qValue));
            }

            // Sort attack options by Q-value in descending order
            attackOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));

            // Declare attacks based on sorted Q-values
            for (AttackOption option : attackOptions) {
                if (option.defenderIndex >= Math.min(RLAction.MAX_ACTIONS, possibleAttackTargets.size())) {
                    continue; // Skip this attacker if the first choice is to not attack
                }
                UUID defenderId = possibleAttackTargets.get(option.defenderIndex);
                if (attacker.canAttack(defenderId, game)) {
                    this.declareAttacker(attacker.getId(), defenderId, game, false);
                    break; // Once an attack is declared, move to the next attacker
                }
            }
        }
    }

    // Don't need to override?
    // TODO: Do we need to pass the action vector a reference for WHICH creature its declaring blocks?
    private void declareBlockers(Game game) {
        currentState = new RLState(game);

        List<Permanent> attackers = getAttackers(game);
        if (attackers == null) {
            return;
        }

        List<Permanent> possibleBlockers = super.getAvailableBlockers(game);
        possibleBlockers = filterOutNonblocking(game, attackers, possibleBlockers);
        if (possibleBlockers.isEmpty()) {
            return;
        }
        logger.info("possibleBlockers: " + possibleBlockers);

        attackers = filterOutUnblockable(game, attackers, possibleBlockers);
        if (attackers.isEmpty()) {
            return;
        }
        currentAction = new RLAction(RLAction.ActionType.DECLARE_BLOCKS, null, null, game);
        INDArray qValues = model.predictDistribution(currentState, currentAction, true);
        
        if (possibleBlockers.size() > RLAction.MAX_ACTIONS) {
            logger.error("ERROR: More blockers than max actions, Model truncating");
        }
        if (attackers.size() > RLAction.MAX_ACTIONS) {
            logger.error("ERROR: More attackers than max actions, Model truncating");
        }

        for (int blockerIndex = 0; blockerIndex < RLAction.MAX_ACTIONS; blockerIndex++) {
            Permanent blocker = possibleBlockers.get(blockerIndex);

            // Create a list of attacker indices with their Q-values for this blocker
            List<BlockOption> blockOptions = new ArrayList<>();
            for (int attackerIndex = 0; attackerIndex < RLAction.MAX_ACTIONS + 1; attackerIndex++) {
                double qValue = qValues.getDouble(blockerIndex, attackerIndex);
                blockOptions.add(new BlockOption(attackerIndex, blockerIndex, qValue));
            }

            // Sort block options by Q-value in descending order
            blockOptions.sort((a, b) -> Double.compare(b.qValue, a.qValue));
 
            // Declare blocks based on sorted Q-values
            for (BlockOption option : blockOptions) {
                if (option.attackerIndex >= Math.min(RLAction.MAX_ACTIONS, attackers.size())) {
                    continue; // Skip this blocker if the first choice is to not block
                }
                Permanent attacker = attackers.get(option.attackerIndex);
                if (blocker.canBlock(attacker.getId(), game)) {
                    this.declareBlocker(playerId, blocker.getId(), attacker.getId(), game);
                    // Remove the attacker if it can't be blocked anymore
                    // if (!attacker.canBeBlocked(game)) {
                    //     attackers.remove(option.attackerIndex);
                    // }
                }
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
        logger.info("priorityPlay called for " + getName() + " during " + game.getTurnStepType());
        game.getState().setPriorityPlayerId(playerId);
        game.firePriorityEvent(playerId);
        RLState nextState;
        switch (game.getTurnStepType()) {
            case UPKEEP:
            case DRAW:
                pass(game);
                return false;
            case PRECOMBAT_MAIN:
                printBattlefieldScore(game, "Sim PRIORITY on MAIN 1");
                ActivatedAbility ability = calculateActions(game);
                if (ability == null) {
                    logger.info("Model opted to pass priority");
                    pass(game);
                    return false;
                }
                act(game, ability);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                printBattlefieldScore(game, "Sim PRIORITY on DECLARE ATTACKERS");
                //TODO: only selectattackers if its your turn
                selectAttackers(game, playerId);
                // TODO: We can also perform actions here but lets simplify for now
                pass(game);
                nextState = new RLState(game);
                addExperience(currentState, currentAction, nextState);
                //act(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattlefieldScore(game, "Sim PRIORITY on DECLARE BLOCKERS");
                //TODO: only declareblockers if its your turn
                declareBlockers(game);
                // TODO: We can also perform actions here but lets simplify for now
                pass(game);
                nextState = new RLState(game);
                addExperience(currentState, currentAction, nextState);
                return true;
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
            case END_COMBAT:
                pass(game);
                return false;
            case POSTCOMBAT_MAIN:
                printBattlefieldScore(game, "Sim PRIORITY on MAIN 2");
                calculateActions(game);
                act(game);
                return true;
            case END_TURN:
            case CLEANUP:
                actionCache.clear();
                pass(game);
                return false;
        }
        return false;
    }

    // I'm changing the design here to not use an actions queue.
    // Instead, I'm passing the ability to the act method.
    // We don't calculate lists of actions, but instead just one action at a time.
    // NOTE: I think the way computerplayer6 does this is because it implements the idea
    // of holding priority
    protected void act(Game game, ActivatedAbility ability) {
        logger.info("act called for " + getName());
        if (ability == null) {
            pass(game);
        } else {
            logger.info(String.format("===> SELECTED ACTION for %s: %s", getName(), ability));
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
            this.activateAbility((ActivatedAbility) ability, game);

            // Store the experience with the updated nextState
            RLState nextState = new RLState(game);
            addExperience(currentState, currentAction, nextState);
        }
    }

    protected ActivatedAbility calculateActions(Game game) {
        logger.info("calculateActions called for " + getName());
        boolean isSimulatedPlayer = true;
        // Returns a list of all available spells and abilities the player can currently cast/activate with his available resources.
        // Without target validation.
        List<ActivatedAbility> playables = game.getPlayer(playerId).getPlayable(game, isSimulatedPlayer);
        currentAction = new RLAction(RLAction.ActionType.ACTIVATE_ABILITY_OR_SPELL, playables, null, game);
        currentState = new RLState(game);

        // Evaluate each action using the model
        INDArray qValues = model.predictDistribution(currentState, currentAction, true);
        double maxQValue = Double.NEGATIVE_INFINITY;
        int bestIndex = 0;
        
        // Find index with highest Q-value
        for (int i = 0; i < qValues.length(); i++) {
            if (qValues.getDouble(i) > maxQValue) {
                maxQValue = qValues.getDouble(i);
                bestIndex = i;
            }
        }

        logger.info("playables: " + playables);
        logger.info("qValues: " + qValues);

        // Get the corresponding ability and add it to actions queue
        // -1 on range is to reserve the option to do nothing
        if (bestIndex < Math.min(RLModel.OUTPUT_SIZE - 1, playables.size())) {
            return playables.get(bestIndex);
        }else{
            //Model chose to pass
            return null;
        }
    }

    public void addExperience(RLState state, RLAction action, RLState nextState) {
        experienceBuffer.add(new Experience(state, action, 0, nextState)); // Reward is set to 0 initially
    }

    public List<Experience> getExperienceBuffer() {
        return experienceBuffer;
    }

    public void clearExperienceBuffer() {
        experienceBuffer.clear();
    }
}

// Helper class to store block options
class BlockOption {
    int attackerIndex;
    int blockerIndex;
    double qValue;

    BlockOption(int attackerIndex, int blockerIndex, double qValue) {
        this.attackerIndex = attackerIndex;
        this.blockerIndex = blockerIndex;
        this.qValue = qValue;
    }
}

// Helper class to store attack options
class AttackOption {
    int defenderIndex;
    int attackerIndex;
    double qValue;

    AttackOption(int defenderIndex, int attackerIndex, double qValue) {
        this.defenderIndex = defenderIndex;
        this.attackerIndex = attackerIndex;
        this.qValue = qValue;
    }
}