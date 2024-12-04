package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;

public class RLState {
    private static final Logger logger = Logger.getLogger(RLState.class);
    private Game game;
    private List<Float> stateVector;

    public RLState(Game game) {
        this.game = game;
        this.stateVector = new ArrayList<>();
        buildStateVector();
    }

    private void buildStateVector() {
        Player player = game.getPlayer(game.getActivePlayerId());
        if (player == null) {
            logger.error("No active player found in game " + game.getId());
            throw new IllegalStateException("Cannot build state vector: no active player");
        }

        // Player state (5 values)
        stateVector.add((float) player.getLife());
        stateVector.add((float) player.getHand().size());
        stateVector.add((float) player.getLibrary().size());
        stateVector.add((float) player.getGraveyard().size());
        stateVector.add((float) player.getLandsPlayed());
        
        // Battlefield state (pad to 40 values)
        game.getBattlefield().getAllPermanents().forEach(permanent -> {
            stateVector.add((float) permanent.getPower().getValue());
            stateVector.add((float) permanent.getToughness().getValue());
        });
        
        // Pad remaining battlefield slots with zeros
        while (stateVector.size() < 45) {  // 5 player state + 40 battlefield state
            stateVector.add(0.0f);
        }
    }

    public Game getGame() {
        return game;
    }

    public List<Float> getStateVector() {
        return stateVector;
    }

    public float[] toFeatureVector() {
        // Convert game state to feature vector
        float[] features = new float[RLModel.STATE_SIZE];
        // TODO: Implement state feature extraction
        return features;
    }
} 