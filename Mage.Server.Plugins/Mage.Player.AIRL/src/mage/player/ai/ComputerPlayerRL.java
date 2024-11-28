package mage.player.ai;

import mage.abilities.Ability;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.combat.Combat;
import mage.player.ai.rl.RLState;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLAction;
import org.apache.log4j.Logger;

import java.util.UUID;

import mage.cards.Cards;
import mage.cards.CardsImpl;

public class ComputerPlayerRL extends ComputerPlayer {

    private static final Logger logger = Logger.getLogger(ComputerPlayerRL.class);
    
    public RLModel model;
    
    public ComputerPlayerRL(UUID id) {
        super("Computer - RL " + id.toString().substring(0, 3), RangeOfInfluence.ALL);
        this.model = new RLModel();
        this.hand = new CardsImpl();  // Initialize hand
    }

    @Override
    public boolean priority(Game game) {
        RLState state = new RLState(game);
        
        RLAction action = model.getAction(state);
        
        if (action != null) {
            return executeAction(action, game);
        }
        
        return false;
    }

    protected boolean executeAction(RLAction action, Game game) {
        return action.execute(game, this.getId());
    }
} 