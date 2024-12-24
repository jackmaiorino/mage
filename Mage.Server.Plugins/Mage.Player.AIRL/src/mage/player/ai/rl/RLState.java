package mage.player.ai.rl;

import java.util.List;
import java.util.Set;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import mage.Mana;
import mage.abilities.costs.mana.ManaCost;
import mage.cards.Card;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.players.Player;


public class RLState {
    private static final Logger logger = Logger.getLogger(RLState.class);
    private float[] stateVector;
    public INDArray targetQValues;
    public static final int CARD_STATS_SIZE = ZoneType.values().length + 24; // ZoneType.values().length for one-hot encoding + 24 for other features
    public static final int NUM_PLAYER_STATS = 5;   
    public static final int NUM_CARDS = 60;
    public static final int EMBEDDING_SIZE = EmbeddingManager.EMBEDDING_SIZE + CARD_STATS_SIZE;
    public static final int ACTION_TYPE_SIZE = ActionType.values().length;
    // The possibility of having tons of tokens may break this
    public static final int STATE_VECTOR_SIZE = ACTION_TYPE_SIZE +
                                                NUM_PLAYER_STATS + (NUM_CARDS * EMBEDDING_SIZE) // Player
                                                + NUM_PLAYER_STATS + (NUM_CARDS * EMBEDDING_SIZE); // Opponent
    public ActionType actionType;
    // TODO: Might not need these anymore
    public int numAttackers;
    public int numAttackTargets;
    public int numBlockers;

    public enum ZoneType {
        HAND,
        BATTLEFIELD, 
        GRAVEYARD,
        EXILE,
        LIBRARY,
        //TODO: IMPLEMENT STACK AWARENESS?
        STACK
    }

    public enum ActionType {
        ACTIVATE_ABILITY_OR_SPELL,
        SELECT_TARGETS,
        DECLARE_ATTACKS,
        DECLARE_BLOCKS,
        MULLIGAN
    }
                                        
    public RLState(Game game, ActionType actionType) {
        stateVector = new float[STATE_VECTOR_SIZE];
        this.actionType = actionType;
        this.targetQValues = Nd4j.zeros(RLModel.MAX_ACTIONS, RLModel.MAX_ACTIONS + 1);
        buildStateVector(game);
    }

    private void buildStateVector(Game game) {
        Player player = game.getPlayer(game.getActivePlayerId());
        if (player == null) {
            logger.error("No active player found in game " + game.getId());
            throw new IllegalStateException("Cannot build state vector: no active player");
        }
        int index = 0;
        // Action Type
        stateVector[actionType.ordinal()] = 1.0f;
        index += ActionType.values().length;

        // Player Numerical Stats Normalized (Some values like graveyard and land are arbitrary)
        stateVector[index++] = (float) player.getLife() / game.getStartingLife();
        stateVector[index++] = (float) player.getHand().size() / 7;
        stateVector[index++] = (float) player.getLibrary().size() / 60;
        stateVector[index++] = (float) player.getGraveyard().size() / 10;
        stateVector[index++] = (float) player.getLandsPlayed() / 10;

        // Player 1 Cards in Hand
        Set<Card> playerCardsInHand = player.getHand().getCards(game);
        for (Card card : playerCardsInHand) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player cards in hand exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.HAND, game);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }

        // Player 1 Permanents on Battlefield
        List<Permanent> playerPermanents = game.getBattlefield().getAllActivePermanents(player.getId());
        for (Permanent permanent : playerPermanents) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player permanents exceed the maximum allowed size");
                break;
            }
            float[] permanentFeatures = convertCardToFeatureVector(permanent, ZoneType.BATTLEFIELD, game);
            System.arraycopy(permanentFeatures, 0, stateVector, index, permanentFeatures.length);
            index += permanentFeatures.length;
        }

        // Player 1 Cards in Graveyard
        Set<Card> playerCardsInGraveyard = player.getGraveyard().getCards(game);
        for (Card card : playerCardsInGraveyard) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player cards in graveyard exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.GRAVEYARD, game);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }

        // Player 1 Library
        // TODO: Will the ai know what cards it is going to draw?
        List<Card> playerCardsInLibrary = player.getLibrary().getCards(game);
        for (Card card : playerCardsInLibrary) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Player cards in library exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.LIBRARY, game);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }

        // Opponent State
        Player opponent = game.getPlayer(game.getOpponents(player.getId()).iterator().next());
        if (opponent == null) {
            logger.error("No opponent found in game " + game.getId());
            throw new IllegalStateException("Cannot build state vector: no opponent");
        }
        stateVector[index++] = (float) opponent.getLife() / game.getStartingLife();
        stateVector[index++] = (float) opponent.getHand().size() / 7;
        stateVector[index++] = (float) opponent.getLibrary().size() / 60;
        stateVector[index++] = (float) opponent.getGraveyard().size() / 10;
        stateVector[index++] = (float) opponent.getLandsPlayed() / 10;

        // We can't know what cards are in the opponent's hand, unless we have special reveal effects
        // Opponent Cards in Hand
        // Set<Card> opponentCardsInHand = opponent.getHand().getCards(game);
        // for (Card card : opponentCardsInHand) {
        //     if (index >= STATE_VECTOR_SIZE) {
        //         logger.error("Opponent cards in hand exceed the maximum allowed size");
        //         break;
        //     }
        //     float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.HAND, game);
        //     System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
        //     index += cardFeatures.length;
        // }

        // Opponent Permanents on Battlefield
        List<Permanent> opponentPermanents = game.getBattlefield().getAllActivePermanents(opponent.getId());
        for (Permanent permanent : opponentPermanents) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Opponent permanents exceed the maximum allowed size");
                break;
            }
            float[] permanentFeatures = convertCardToFeatureVector(permanent, ZoneType.BATTLEFIELD, game);
            System.arraycopy(permanentFeatures, 0, stateVector, index, permanentFeatures.length);
            index += permanentFeatures.length;
        }

        // Opponent Cards in Graveyard
        Set<Card> opponentCardsInGraveyard = opponent.getGraveyard().getCards(game);
        for (Card card : opponentCardsInGraveyard) {
            if (index >= STATE_VECTOR_SIZE) {
                logger.error("Opponent cards in graveyard exceed the maximum allowed size");
                break;
            }
            float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.GRAVEYARD, game);
            System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
            index += cardFeatures.length;
        }

        // Cards in Exile
        for (ExileZone exile : game.getExile().getExileZones()) {
            for (Card card : exile.getCards(game)) {
                if (index >= STATE_VECTOR_SIZE) {
                    logger.error("Cards in exile exceed the maximum allowed size");
                    break;
                }
                float[] cardFeatures = convertCardToFeatureVector(card, ZoneType.EXILE, game);
                System.arraycopy(cardFeatures, 0, stateVector, index, cardFeatures.length);
                index += cardFeatures.length;
            }
        }
    }

    public float[] getStateVector() {
        return stateVector;
    }

    public ActionType getActionType() {
        return actionType;
    }

    public float[] convertCardToFeatureVector(Card card, ZoneType zoneType, Game game) {
        float[] featureVector = new float[EMBEDDING_SIZE];

        // One-hot encode the zone type
        int index = 0;
        featureVector[zoneType.ordinal()] = 1.0f;
        index += ZoneType.values().length;
        featureVector[index++] = card.getOwnerId().equals(game.getActivePlayerId()) ? 1.0f : 0.0f;
        featureVector[index++] = (float) card.getPower().getValue();
        featureVector[index++] = (float) card.getToughness().getValue();
        featureVector[index++] = (float) card.getManaValue();
        // TODO: represent all possible mana costs
        // getMana returns all possible mana costs, this is just one for now
        // TODO: represent phyrexian mana

        // if (card instanceof Spell) {
        //     // Mana it costs to cast
        //     currManaCost = card.getManaCost().get(0).getManaOptions().get(0);
        // } else {
        //      // This is for what mana it produces
        //      // TODO: some spell cards can produce mana, need to handle that
        //     currManaProduced = card.getMana().get(0);
        // }
        Mana currManaCost = new Mana();
        if (!card.getManaCost().isEmpty()) {
            for (ManaCost manaCost : card.getManaCost()) {
                Mana TempManaCost = manaCost.getMana();
                currManaCost.setWhite(TempManaCost.getWhite() + currManaCost.getWhite());
                currManaCost.setBlue(TempManaCost.getBlue() + currManaCost.getBlue());
                currManaCost.setGreen(TempManaCost.getGreen() + currManaCost.getGreen());
                currManaCost.setBlack(TempManaCost.getBlack() + currManaCost.getBlack());
                currManaCost.setRed(TempManaCost.getRed() + currManaCost.getRed());
                currManaCost.setColorless(TempManaCost.getColorless() + currManaCost.getColorless());
                currManaCost.setGeneric(TempManaCost.getGeneric() + currManaCost.getGeneric());
            }
        }else{
            currManaCost = null;
        }
        featureVector[index++] = currManaCost != null ? currManaCost.getWhite() : 0.0f;
        featureVector[index++] = currManaCost != null ? currManaCost.getBlue() : 0.0f;
        featureVector[index++] = currManaCost != null ? currManaCost.getGreen() : 0.0f;
        featureVector[index++] = currManaCost != null ? currManaCost.getBlack() : 0.0f;
        featureVector[index++] = currManaCost != null ? currManaCost.getRed() : 0.0f;   
        featureVector[index++] = currManaCost != null ? currManaCost.getColorless() : 0.0f;
        featureVector[index++] = currManaCost != null ? currManaCost.getGeneric() : 0.0f;

        featureVector[index++] = card.isCreature() ? 1.0f : 0.0f;
        featureVector[index++] = card.isArtifact() ? 1.0f : 0.0f;
        featureVector[index++] = card.isEnchantment() ? 1.0f : 0.0f;
        featureVector[index++] = card.isLand() ? 1.0f : 0.0f;
        featureVector[index++] = card.isPlaneswalker() ? 1.0f : 0.0f;
        featureVector[index++] = card.isPermanent() ? 1.0f : 0.0f;
        featureVector[index++] = card.isInstant() ? 1.0f : 0.0f;
        featureVector[index++] = card.isSorcery() ? 1.0f : 0.0f;
        // Is the card tapped?
        if (zoneType == ZoneType.BATTLEFIELD) {
            featureVector[index++] = (card instanceof Permanent) ? ((Permanent)card).isTapped() ? 1.0f : 0.0f : 0.0f;
            featureVector[index++] = (card instanceof Permanent) ? ((Permanent)card).isAttacking() ? 1.0f : 0.0f : 0.0f;
            featureVector[index++] = (card instanceof Permanent) ? ((Permanent)card).isBlocked(game) ? 1.0f : 0.0f : 0.0f;
            featureVector[index++] = (card instanceof Permanent) ? ((Permanent)card).hasSummoningSickness() ? 1.0f : 0.0f : 0.0f;
            featureVector[index++] = (card instanceof Permanent) ? ((Permanent)card).getDamage() : 0.0f;
        }else{
            featureVector[index++] = 0.0f;
            featureVector[index++] = 0.0f;
            featureVector[index++] = 0.0f;
            featureVector[index++] = 0.0f;
            featureVector[index++] = 0.0f;
        }

        // Add the text embedding of the text
        String cardAbilities = card.getAbilities().toString();
        String cardText = String.join(" ", card.getRules());
        float[] textEmbedding = EmbeddingManager.getEmbedding(cardText);
        System.arraycopy(textEmbedding, 0, featureVector, index, textEmbedding.length);

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
