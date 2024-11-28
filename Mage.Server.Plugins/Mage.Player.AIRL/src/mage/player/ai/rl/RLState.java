package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import java.util.ArrayList;
import java.util.List;

public class RLState {
    private Game game;
    private List<Float> stateVector;

    public RLState(Game game) {
        this.game = game;
        this.stateVector = new ArrayList<>();
        buildStateVector();
    }

    private void buildStateVector() {
        Player player = game.getPlayer(game.getActivePlayerId());
        if (player != null) {
            stateVector.add((float) player.getLife());
            stateVector.add((float) player.getHand().size());
            // Add more state information as needed
        }
    }

    public Game getGame() {
        return game;
    }

    public List<Float> getStateVector() {
        return stateVector;
    }
} 