package mage.player.ai.rl;

import mage.game.Game;
import mage.players.Player;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;
import mage.abilities.ActivatedAbility;
import mage.abilities.SpellAbility;
import mage.abilities.ClueAbility;
import mage.abilities.SomeOtherAbilityType; // Replace with actual ability types
import mage.abilities.effects.Effect;

public class RLState {
    private static final Logger logger = Logger.getLogger(RLState.class);
    private List<Float> stateVector;
    private List<RLAction> possibleActions;

    public RLState(Game game, List<RLAction> possibleActions) {
        this.stateVector = new ArrayList<>();
        this.possibleActions = new ArrayList<>(possibleActions);
        buildStateVector(game);
    }

    public RLState(Game game) {
        this.stateVector = new ArrayList<>();
        this.possibleActions = new ArrayList<>();
        buildStateVector(game);
    }

    private void buildStateVector(Game game) {
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
        while (stateVector.size() < 45) { // 5 player state + 40 battlefield state
            stateVector.add(0.0f);
        }
    }

    public List<Float> getStateVector() {
        return stateVector;
    }

    public float[] toFeatureVector() {
        float[] features = new float[RLModel.STATE_SIZE];
        for (int i = 0; i < Math.min(stateVector.size(), RLModel.STATE_SIZE); i++) {
            features[i] = stateVector.get(i);
        }
        return features;
    }

    public List<RLAction> getPossibleActions() {
        return possibleActions;
    }

    public void setPossibleActions(List<RLAction> possibleActions) {
        this.possibleActions = new ArrayList<>(possibleActions);
    }

    public float[] convertPermanentToFeatureVector(Permanent permanent) {
        float[] featureVector = new float[40];

        // Example features
        featureVector[0] = (float) permanent.getPower().getValue();
        featureVector[1] = (float) permanent.getToughness().getValue();
        featureVector[2] = permanent.isTapped() ? 1.0f : 0.0f;
        featureVector[3] = permanent.isAttacking() ? 1.0f : 0.0f;
        featureVector[4] = permanent.isBlocking() ? 1.0f : 0.0f;
        featureVector[5] = permanent.isCreature() ? 1.0f : 0.0f;
        featureVector[6] = permanent.isArtifact() ? 1.0f : 0.0f;
        featureVector[7] = permanent.isEnchantment() ? 1.0f : 0.0f;
        featureVector[8] = permanent.isLand() ? 1.0f : 0.0f;
        featureVector[9] = permanent.isPlaneswalker() ? 1.0f : 0.0f;

        // Add more features as needed, such as abilities, counters, etc.

        // Zero padding for remaining slots
        for (int i = 10; i < featureVector.length; i++) {
            featureVector[i] = 0.0f;
        }

        return featureVector;
    }

    private float[] getTextEmbedding(String text) {
        // Implement text embedding logic here, possibly using a pre-trained NLP model
        // TODO: Implement text embedding logic
        return new float[10]; // Example: return a fixed-size embedding
    }
}