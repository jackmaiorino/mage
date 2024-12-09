package mage.player.ai;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.player.ai.rl.RLState;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLAction;
import org.apache.log4j.Logger;

import java.util.UUID;
import java.util.List;
import java.util.LinkedList;

import mage.target.Target;
import mage.game.events.GameEvent;
import mage.players.Player;
import mage.filter.StaticFilters;
import mage.game.permanent.Permanent;

public class ComputerPlayerRL extends ComputerPlayer6 {
    private static final Logger logger = Logger.getLogger(ComputerPlayerRL.class);
    public RLModel model;
    protected LinkedList<Ability> actions = new LinkedList<>();
    protected RLState currentState;

    public ComputerPlayerRL(String name, RangeOfInfluence range, int skill) {
        super(name, range, skill);
        this.model = new RLModel(playerId);
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

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        logger.info("selectAttackers called for " + getName());
        
        currentState = new RLState(game);
        List<Permanent> possibleAttackers = game.getBattlefield().getAllActivePermanents(
            StaticFilters.FILTER_PERMANENT_CREATURE,
            playerId,
            game
        );
        
        for (Permanent creature : possibleAttackers) {
            if (creature.canAttack(null, game)) {
                RLAction action = new RLAction(RLAction.ActionType.ATTACK, creature.getId());
                float score = model.predictQValue(currentState, action);
                if (score > model.getActionThreshold()) {
                    // Declare the creature as an attacker
                    this.declareAttacker(creature.getId(), game.getCombat().getDefenders().iterator().next(), game, false);
                }
            }
        }
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
        switch (game.getTurnStepType()) {
            case UPKEEP:
            case DRAW:
                pass(game);
                return false;
            case PRECOMBAT_MAIN:
                printBattlefieldScore(game, "Sim PRIORITY on MAIN 1");
                calculateActions(game);
                act(game);
                return true;
            case BEGIN_COMBAT:
                pass(game);
                return false;
            case DECLARE_ATTACKERS:
                printBattlefieldScore(game, "Sim PRIORITY on DECLARE ATTACKERS");
                calculateActions(game);
                act(game);
                return true;
            case DECLARE_BLOCKERS:
                printBattlefieldScore(game, "Sim PRIORITY on DECLARE BLOCKERS");
                calculateActions(game);
                act(game);
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

    protected void act(Game game) {
        logger.info("act called for " + getName());
        if (actions == null || actions.isEmpty()) {
            pass(game);
        } else {
            boolean usedStack = false;
            while (actions.peek() != null) {
                Ability ability = actions.poll();
                logger.info(String.format("===> SELECTED ACTION for %s: %s", getName(), ability));
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
                if (ability.isUsesStack()) {
                    usedStack = true;
                }
            }
            if (usedStack) {
                pass(game);
            }
        }
    }

    protected void calculateActions(Game game) {
        logger.info("calculateActions called for " + getName());
        // Get all playable actions
        boolean isSimulatedPlayer = true;
        List<ActivatedAbility> playables = game.getPlayer(playerId).getPlayable(game, isSimulatedPlayer);
        RLAction bestAction = null;
        float bestScore = Float.NEGATIVE_INFINITY;

        // Evaluate each action using the model
        for (ActivatedAbility ability : playables) {
            RLState state = new RLState(game);
            RLAction action = new RLAction(RLAction.ActionType.ACTIVATE_ABILITY, ability);
            float score = model.predictQValue(state, action);

            // Select the action with the highest score
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
        }

        // Add the best action to the actions list
        if (bestAction != null) {
            actions.clear();
            actions.add(bestAction.getAbility());
        } else {
            logger.info("No valid actions found.");
        }
    }
}