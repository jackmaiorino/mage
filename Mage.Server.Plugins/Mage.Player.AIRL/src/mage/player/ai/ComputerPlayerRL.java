package mage.player.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import org.apache.log4j.Logger;

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
import mage.player.ai.util.CombatInfo;
import mage.player.ai.util.CombatUtil;
import mage.players.Player;
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
        currentAction = new RLAction(RLAction.ActionType.DECLARE_ATTACKS, null, possibleAttackers, game, -1);
        float[] qValues = model.predictDistribution(currentState, currentAction, true);
        // We need to not even consider the case where there are no attacker
        logger.info("possibleAttackers: " + possibleAttackers);
        logger.info("qValues: " + Arrays.toString(qValues));
        for (int i = 0; i < possibleAttackers.size(); i++) {
            if (qValues[i] > model.getActionThreshold()) {
                // Declare the creature as an attacker
                this.declareAttacker(possibleAttackers.get(i).getId(), game.getCombat().getDefenders().iterator().next(), game, false);
            }
        }   
    }

    // Don't need to override?
    // TODO: Do we need to pass the action vector a reference for WHICH creature its declaring blocks?
    private void declareBlockers(Game game, UUID activePlayerId) {
        currentState = new RLState(game);
        //TODO: Implement blocker selection
        //TODO: Investigate comp player 6 firing these game events. Do we need them?

        List<Permanent> attackers = getAttackers(game);
        if (attackers == null) {
            return;
        }

        List<Permanent> possibleBlockers = super.getAvailableBlockers(game);
        possibleBlockers = filterOutNonblocking(game, attackers, possibleBlockers);
        if (possibleBlockers.isEmpty()) {
            return;
        }

        attackers = filterOutUnblockable(game, attackers, possibleBlockers);
        if (attackers.isEmpty()) {
            return;
        }

        int cardReferenceIndex = 0;
        for (Permanent attacker : attackers) {
            currentAction = new RLAction(RLAction.ActionType.DECLARE_BLOCKS, null, null, game, cardReferenceIndex);
            float[] qValues = model.predictDistribution(currentState, currentAction, true);
            for (int i = 0; i < possibleBlockers.size(); i++) {
                if (qValues[i] > model.getActionThreshold()) {
                    // Declare the creature as an attacker
                    this.declareBlocker(playerId, possibleBlockers.get(i).getId(), attacker.getId(), game);
                    // TODO: Add support for creatures that can block multiple attackers
                    possibleBlockers.remove(i);
                }
            }   
            cardReferenceIndex++;
        }

        CombatUtil.sortByPower(attackers, false);

        CombatInfo combatInfo = CombatUtil.blockWithGoodTrade2(game, attackers, possibleBlockers);
        Player player = game.getPlayer(playerId);

        boolean blocked = false;
        for (Map.Entry<Permanent, List<Permanent>> entry : combatInfo.getCombat().entrySet()) {
            UUID attackerId = entry.getKey().getId();
            List<Permanent> blockers = entry.getValue();
            if (blockers != null) {
                for (Permanent blocker : blockers) {
                    player.declareBlocker(player.getId(), blocker.getId(), attackerId, game);
                    blocked = true;
                }
            }
        }
        if (blocked) {
            game.getPlayers().resetPassed();
        }


        pass(game);
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
                selectAttackers(game, playerId);
                // TODO: We can also perform actions here but lets simplify for now
                pass(game);
                nextState = new RLState(game);
                addExperience(currentState, currentAction, nextState);
                //act(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattlefieldScore(game, "Sim PRIORITY on DECLARE BLOCKERS");
                declareBlockers(game, playerId);
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
        currentAction = new RLAction(RLAction.ActionType.ACTIVATE_ABILITY_OR_SPELL, playables, null, game, -1);
        currentState = new RLState(game);

        // Evaluate each action using the model
        float[] qValues = model.predictDistribution(currentState, currentAction, true);
        float maxQValue = Float.NEGATIVE_INFINITY;
        int bestIndex = 0;
        
        // Find index with highest Q-value
        for (int i = 0; i < qValues.length; i++) {
            if (qValues[i] > maxQValue) {
                maxQValue = qValues[i];
                bestIndex = i;
            }
        }

        logger.info("playables: " + playables);
        logger.info("qValues: " + Arrays.toString(qValues));

        // Get the corresponding ability and add it to actions queue
        if (bestIndex < playables.size()) {
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