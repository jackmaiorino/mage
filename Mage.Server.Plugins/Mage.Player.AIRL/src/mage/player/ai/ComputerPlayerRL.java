package mage.player.ai;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.common.PassAbility;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.combat.Combat;
import mage.player.ai.rl.RLState;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLAction;
import mage.players.Player;
import org.apache.log4j.Logger;

import java.util.UUID;
import java.util.List;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;

public class ComputerPlayerRL extends ComputerPlayer6 {
    private static final Logger logger = Logger.getLogger(ComputerPlayerRL.class);
    public RLModel model;
    protected RLState currentState;
    
    public ComputerPlayerRL(UUID id) {
        super("Computer - RL " + id.toString().substring(0, 3), RangeOfInfluence.ALL);
        this.model = new RLModel(id);
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
        currentState = new RLState(game);
        RLAction action = model.getAction(currentState);
        if (action != null && action.getType() == RLAction.ActionType.ATTACK) {
            declareAttacker(action.getTargetId(), game.getCombat().getDefenders().iterator().next(), game, false);
        }
    }

    @Override
    public boolean chooseMulligan(Game game) {
        currentState = new RLState(game);
        RLAction action = model.getAction(currentState);
        return action != null && action.getType() == RLAction.ActionType.MULLIGAN;
    }

    protected boolean priorityPlay(Game game) {
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

    protected List<ActivatedAbility> getPlayableAbilities(Game game) {
        List<ActivatedAbility> playables = getPlayable(game, true);
        
        // Create state and get model prediction for filtering abilities
        currentState = new RLState(game);
        List<Float> abilityScores = model.evaluateAbilities(currentState, playables);
        
        // Filter out low-scoring abilities (optional)
        List<ActivatedAbility> filteredPlayables = new ArrayList<>();
        for (int i = 0; i < playables.size(); i++) {
            if (abilityScores.get(i) > model.getActionThreshold()) {
                filteredPlayables.add(playables.get(i));
            }
        }
        
        // Always include pass ability as an option
        filteredPlayables.add(new PassAbility());
        
        return filteredPlayables;
    }

    public static List<RLAction> getPlayableActions(Game game, ComputerPlayerRL player) {
        List<RLAction> actions = new ArrayList<>();
        
        // Always add PASS as a possible action
        actions.add(new RLAction(RLAction.ActionType.PASS));

        // Get all activated abilities that can be used
        for (Ability ability : player.getPlayable(game, true)) {
            if (ability instanceof ActivatedAbility && !(ability instanceof PassAbility)) {
                actions.add(new RLAction(RLAction.ActionType.ACTIVATE_ABILITY, ability));
            }
        }

        return actions;
    }

    protected void act(Game game) {
        RLAction action = model.getAction(new RLState(game));
        if (action != null) {
            logger.info(String.format("===> SELECTED ACTION for %s: %s", getName(), action));
            action.execute(game, playerId);
        } else {
            pass(game);
        }
    }

    protected void calculateActions(Game game) {
        // Check if there are already actions calculated
        if (!getNextAction(game)) {
            // Log the start of action calculation
            Date startTime = new Date();
            
            // Evaluate the current game state
            currentScore = GameStateEvaluator2.evaluate(playerId, game).getTotalScore();
            
            // Create a simulation of the current game state
            Game sim = createSimulation(game);
            
            // Reset the simulation node count
            SimulationNode2.resetCount();
            
            // Initialize the root of the decision tree
            root = new SimulationNode2(null, sim, maxDepth, playerId);
            
            // Add actions to the decision tree
            addActionsTimed();
            
            // Check if the root has children (possible actions)
            if (root != null && root.children != null && !root.children.isEmpty()) {
                root = root.children.get(0);

                // Prevent repeating the same action with no cost
                boolean doThis = true;
                if (root.abilities.size() == 1) {
                    for (Ability ability : root.abilities) {
                        if (ability.getManaCosts().manaValue() == 0 && ability.getCosts().isEmpty()) {
                            if (actionCache.contains(ability.getRule() + '_' + ability.getSourceId())) {
                                doThis = false; // Don't do it again
                            }
                        }
                    }
                }

                // If valid, set actions and combat
                if (doThis) {
                    actions = new LinkedList<>(root.abilities);
                    combat = root.combat;
                    for (Ability ability : actions) {
                        actionCache.add(ability.getRule() + '_' + ability.getSourceId());
                    }
                }
            } else {
                logger.info('[' + game.getPlayer(playerId).getName() + "][pre] Action: skip");
            }
            
            // Log the end of action calculation
            Date endTime = new Date();
            this.setLastThinkTime((endTime.getTime() - startTime.getTime()));
        } else {
            logger.debug("Next Action exists!");
        }
    }
}