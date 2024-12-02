package mage.player.ai;

import mage.abilities.Ability;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.combat.Combat;
import mage.player.ai.rl.RLState;
import mage.player.ai.rl.RLModel;
import mage.player.ai.rl.RLAction;
import mage.player.ai.rl.RLTrainer;
import mage.players.Player;
import org.apache.log4j.Logger;

import java.util.UUID;
import java.util.List;
import java.util.ArrayList;

import mage.cards.Cards;
import mage.cards.CardsImpl;

public class ComputerPlayerRL extends ComputerPlayer {

    private static final Logger logger = Logger.getLogger(ComputerPlayerRL.class);
    
    public RLModel model;
    
    public ComputerPlayerRL(UUID id) {
        super("Computer - RL " + id.toString().substring(0, 3), RangeOfInfluence.ALL);
        this.model = new RLModel(id);
        this.hand = new CardsImpl();
    }

    @Override
    public boolean priority(Game game) {
        if (game == null) {
            logger.error("Game is null in priority()");
            return false;
        }

        // Validate player still exists
        Player self = game.getPlayer(this.getId());
        if (self == null) {
            logger.error("Player " + this.getId() + " not found in game during priority. Active player: " + 
                        game.getActivePlayerId() + ", Players: " + game.getPlayers().keySet());
            return false;
        }
        
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

    public static List<RLAction> getPlayableActions(Game game, ComputerPlayerRL player) {
        RLTrainer trainer = new RLTrainer();
        return trainer.getPlayableActions(game, player);
    }

    public List<RLAction> getPlayableActions(Game game) {
        RLTrainer trainer = new RLTrainer();
        return trainer.getPlayableActions(game, this);
    }
} 