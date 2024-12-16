package mage.player.ai.rl;

import mage.cards.Card;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.players.Player;
import java.util.List;
import java.util.Set;

import org.apache.log4j.Logger;

public class RLState {
    private static final Logger logger = Logger.getLogger(RLState.class);
    private float[] stateVector;
    private RLAction currentAction;
    public static int STATE_VECTOR_SIZE = 0;
    public static final int MAX_PERMANENTS = 15;
    public static final int MAX_CARDS_IN_HAND = 10;
    public static final int MAX_CARDS_IN_GRAVEYARD = 20;
    public static int EMBEDDING_SIZE = 0;
    public static final int CARD_STATS_SIZE = 8;

    public RLState(Game game, RLAction action) {
        this.stateVector = new float[STATE_VECTOR_SIZE];
        this.currentAction = new RLAction(action);
        EMBEDDING_SIZE = EmbeddingManager.REDUCED_EMBEDDING_SIZE + CARD_STATS_SIZE;
        STATE_VECTOR_SIZE = 5 // Player 1 Numerical Stats
        + (MAX_CARDS_IN_HAND * EMBEDDING_SIZE) // Player 1 Cards in Hand
        + (MAX_PERMANENTS * EMBEDDING_SIZE) // Player 1 Permanents
        + (MAX_CARDS_IN_GRAVEYARD * EMBEDDING_SIZE) // Player 1 Graveyard
        + 5 // Opponent Numerical Stats
        + (MAX_CARDS_IN_HAND * EMBEDDING_SIZE) // Opponent Cards in Hand
        + (MAX_PERMANENTS * EMBEDDING_SIZE) // Opponent Permanents
        + (MAX_CARDS_IN_GRAVEYARD * EMBEDDING_SIZE); // Opponent Graveyard
        stateVector = new float[STATE_VECTOR_SIZE];
        buildStateVector(game);
    }

    private void buildStateVector(Game game) {
        Player player = game.getPlayer(game.getActivePlayerId());
        if (player == null) {
            logger.error("No active player found in game " + game.getId());
            throw new IllegalStateException("Cannot build state vector: no active player");
        }

        // Numerical Stats
        int index = 0;
        stateVector[index++] = (float) player.getLife();
        stateVector[index++] = (float) player.getHand().size();
        stateVector[index++] = (float) player.getLibrary().size();
        stateVector[index++] = (float) player.getGraveyard().size();
        stateVector[index++] = (float) player.getLandsPlayed();

        // Player 1 Cards in Hand
        Set<Card> playerCardsInHand = player.getHand().getCards(game);
        for (Card card : playerCardsInHand) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player cards in hand exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }
        index += (MAX_CARDS_IN_HAND - playerCardsInHand.size()) * EMBEDDING_SIZE; // Padding

        // Player 1 Permanents on Battlefield
        List<Permanent> playerPermanents = game.getBattlefield().getAllActivePermanents(player.getId());
        for (Permanent permanent : playerPermanents) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player permanents exceed the maximum allowed size");
                break;
            }
            float[] permanentFeatures = convertCardToFeatureVector(permanent);
            System.arraycopy(permanentFeatures, 0, stateVector, index, permanentFeatures.length);
            index += permanentFeatures.length;
        }
        index += (MAX_PERMANENTS - playerPermanents.size()) * EMBEDDING_SIZE; // Padding

        // Player 1 Cards in Graveyard
        Set<Card> playerCardsInGraveyard = player.getGraveyard().getCards(game);
        for (Card card : playerCardsInGraveyard) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player cards in graveyard exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }
        index += (MAX_CARDS_IN_GRAVEYARD - playerCardsInGraveyard.size()) * EMBEDDING_SIZE; // Padding

        // Opponent State
        Player opponent = game.getPlayer(game.getOpponents(player.getId()).iterator().next());
        if (opponent == null) {
            logger.error("No opponent found in game " + game.getId());
            throw new IllegalStateException("Cannot build state vector: no opponent");
        }
        stateVector[index++] = (float) opponent.getLife();
        stateVector[index++] = (float) opponent.getHand().size();
        stateVector[index++] = (float) opponent.getLibrary().size();
        stateVector[index++] = (float) opponent.getGraveyard().size();
        stateVector[index++] = (float) opponent.getLandsPlayed();

        // Opponent Cards in Hand
        Set<Card> opponentCardsInHand = opponent.getHand().getCards(game);
        for (Card card : opponentCardsInHand) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Opponent cards in hand exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }
        index += (MAX_CARDS_IN_HAND - opponentCardsInHand.size()) * EMBEDDING_SIZE; // Padding

        // Opponent Permanents on Battlefield
        List<Permanent> opponentPermanents = game.getBattlefield().getAllActivePermanents(opponent.getId());
        for (Permanent permanent : opponentPermanents) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Opponent permanents exceed the maximum allowed size");
                break;
            }
            float[] permanentFeatures = convertCardToFeatureVector(permanent);
            System.arraycopy(permanentFeatures, 0, stateVector, index, permanentFeatures.length);
            index += permanentFeatures.length;
        }
        index += (MAX_PERMANENTS - opponentPermanents.size()) * EMBEDDING_SIZE; // Padding

        // Opponent Cards in Graveyard
        Set<Card> opponentCardsInGraveyard = opponent.getGraveyard().getCards(game);
        for (Card card : opponentCardsInGraveyard) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Opponent cards in graveyard exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }
        index += (MAX_CARDS_IN_GRAVEYARD - opponentCardsInGraveyard.size()) * EMBEDDING_SIZE; // Padding
    }

    public float[] getStateVector() {
        return stateVector;
    }

    public RLAction getCurrentAction() {
        return currentAction;
    }

    public void setCurrentAction(RLAction action) {
        this.currentAction = action;
    }

    public float[] convertCardToFeatureVector(Card card) {
        float[] featureVector = new float[EMBEDDING_SIZE];

        // Example features
        featureVector[0] = (float) card.getPower().getValue();
        featureVector[1] = (float) card.getToughness().getValue();
        featureVector[2] = card.isCreature() ? 1.0f : 0.0f;
        featureVector[3] = card.isArtifact() ? 1.0f : 0.0f;
        featureVector[4] = card.isEnchantment() ? 1.0f : 0.0f;
        featureVector[5] = card.isLand() ? 1.0f : 0.0f;
        featureVector[6] = card.isPlaneswalker() ? 1.0f : 0.0f;
        featureVector[7] = card.isPermanent() ? 1.0f : 0.0f;
        // Add more features as needed, such as abilities, counters, etc.

        // Add the text embedding of the text
        String cardText = String.join(" ", card.getRules());
        float[] textEmbedding = EmbeddingManager.getEmbedding(cardText);
        System.arraycopy(textEmbedding, 0, featureVector, 8, textEmbedding.length);

        return featureVector;
    }

    // Might to break up permanent and spell into separate functions
    // protected void permanentToString(Permanent permanent) {
    //     StringBuilder sb = new StringBuilder();
    //     sb.append(permanent.getControllerId()).append(permanent.getName()).append(permanent.isTapped()).append(permanent.getDamage());
    //     sb.append(permanent.getSubtype().toString()).append(permanent.getSupertype().toString()).append(permanent.getPower().getValue()).append(permanent.getToughness().getValue());
    //     sb.append(permanent.getAbilities().toString());
    //     for (Counter counter : permanent.getCounters().values()) {
    //         sb.append(counter.getName()).append(counter.getCount());
    //     }
    //     return sb.toString();
    // }

    // protected void CardToString (Card card) {
    //     StringBuilder sb = new StringBuilder();
    //     sb.append(card.getControllerId()).append(card.getName()).append(card.getCost());
    //     sb.append(card.getAbilities().toString());
    //     return sb.toString();
    // }
}
