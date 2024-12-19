package mage.player.ai.rl;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.TargetAmount;
import mage.util.CardUtil;

public class RLAction {
    private static final Logger logger = Logger.getLogger(RLAction.class);

    public enum ActionType {
        ACTIVATE_ABILITY_OR_SPELL,
        SELECT_TARGETS,
        DECLARE_ATTACKS,
        DECLARE_BLOCKS,
        MULLIGAN
    }

    private final ActionType type;
    private final List<ActivatedAbility> abilities;
    private final List<Permanent> creatures;
    private final Game game;
    private final int cardReferenceIndex;
    public static final int REFERENCE_INDEX_SIZE = 10;
    private float[] featureVector;
    public static final int MAX_ACTIONS = 10;
    public static final int EMBEDDING_SIZE = EmbeddingManager.EMBEDDING_SIZE + 4; // 4 for basic features
    //public static final int FEATURE_VECTOR_SIZE = MAX_ACTIONS * EMBEDDING_SIZE + ActionType.values().length;  Include space for one-hot encoding
    public static final int FEATURE_VECTOR_SIZE = ActionType.values().length + REFERENCE_INDEX_SIZE;


    public RLAction(ActionType type, List<ActivatedAbility> abilities, List<Permanent> creatures, Game game, int cardReferenceIndex) {
        this.type = type;
        this.abilities = abilities;
        this.creatures = creatures;
        this.game = game;
        this.featureVector = toFeatureVector();
        this.cardReferenceIndex = cardReferenceIndex;
    }

    public List<ActivatedAbility> getAbilities() {
        return abilities;
    }

    public float[] getFeatureVector() {
        return featureVector;
    }

    public List<Permanent> getCreatures() {
        return creatures;
    }

    public ActionType getType() {
        return type;
    }

    public Game getGame() {
        return game;
    }

    // Temporarily going to try only passing the action type
    private float[] toFeatureVector() {
        featureVector = new float[FEATURE_VECTOR_SIZE];
        int index = 0;

        // One-hot encode action type
        int typeIndex = type.ordinal();
        featureVector[typeIndex] = 1.0f;
        index += ActionType.values().length;

        // Add the card reference index
        // This will be used to reference the card in the game state
        // For example: If we are deciding on how to block a creature, if the (attacking creature) that appear
        // in the game state is at index 5, then the card reference index will be 5

        //Right now we are going to allow the option to pass -1 as the index to not specify a card
        if (cardReferenceIndex != -1) {
            featureVector[index + cardReferenceIndex] = 1.0f;
        }

        // Type-specific features
        // switch (type) {
        //     case ACTIVATE_ABILITY_OR_SPELL:
        //         if (abilities != null) {
        //             for (ActivatedAbility ability : abilities) {
        //                 if (index >= FEATURE_VECTOR_SIZE) {
        //                     logger.error("Too many abilities, truncating");
        //                     break;
        //                 }
        //                 float[] abilityFeatures = convertAbilityOrSpellToFeatureVector(ability);
        //                 System.arraycopy(abilityFeatures, 0, featureVector, index, abilityFeatures.length);
        //                 index += abilityFeatures.length;
        //             }
        //         }
        //         break;
        //     case DECLARE_ATTACKS:
        //         //TODO: I'm sure there is a better way to do this. I don't like duplicating the game state info here
        //         if (creatures != null) {
        //             for (Permanent creature : creatures) {
        //                 if (index >= FEATURE_VECTOR_SIZE) {
        //                     logger.error("Too many creatures, truncating");
        //                     break;
        //                 }
        //                 float[] creatureFeatures = convertCardToFeatureVector(creature, ZoneType.BATTLEFIELD, game);
        //                 System.arraycopy(creatureFeatures, 0, featureVector, index, creatureFeatures.length);
        //                 index += creatureFeatures.length;
        //             }
        //         }
        //         break;
        //     case DECLARE_BLOCKS:
        //         // Add block-specific features
        //         // TODO: Implement block-specific features
        //         break;
        //     case MULLIGAN:
        //         // TODO: Implement mulligan-specific features
        //         break;
        //     case PASS_PRIORITY:
        //         // TODO: Implement pass priority-specific features
        //         break;
        //     default:
        //         break;
        // }

        return featureVector;
    }

    private float[] convertAbilityOrSpellToFeatureVector(ActivatedAbility ability) {
        // TODO: We're setting a class global but also returning?!
        featureVector = new float[EMBEDDING_SIZE];

        // Add basic ability features
        int index = 0;
        featureVector[index++] = ability.getManaCosts().manaValue(); // Total mana cost
        featureVector[index++] = ability.getEffects().size(); // Number of effects
        featureVector[index++] = ability.getTargets().size(); // Number of targets
        featureVector[index++] = ability.getCosts().size(); // Number of costs


        // Get class name embedding and add it to the feature vector
        String className = ability.getClass().getSimpleName();

        // Encode Card Text
        // TODO: This will likely not include all card text for casting creatures
        // i.e. it will say something like "Cast a 2/2 red Dragon"
        // but not "Cast a red Dragon with 2/2 with flying and lifeline"
        // not sure will need to test

        //Might want to use this at some point
        //String abilityText = getAbilityAndSourceInfo(game, ability, false);
        float[] textEmbedding = EmbeddingManager.getEmbedding(className + " " + ability);
        System.arraycopy(textEmbedding, 0, featureVector, index, textEmbedding.length);

        // Zero padding for remaining slots
        for (int i = index + textEmbedding.length; i < featureVector.length; i++) {
            featureVector[i] = 0.0f;
        }

        return featureVector;
    }

    protected String getAbilityAndSourceInfo(Game game, Ability ability, boolean showTargets) {
        // ability
        // TODO: add modal info
        // + (action.isModal() ? " Mode = " + action.getModes().getMode().toString() : "")
        if (ability.isModal()) {
            //throw new IllegalStateException("TODO: need implement");
        }
        MageObject sourceObject = ability.getSourceObject(game);
        String abilityInfo = (sourceObject == null ? "" : sourceObject.getIdName() + ": ") + CardUtil.substring(ability.toString(), 30, "...");
        // targets
        String targetsInfo = "";
        if (showTargets) {
            List<String> allTargetsInfo = new ArrayList<>();
            ability.getAllSelectedTargets().forEach(target -> {
                target.getTargets().forEach(selectedId -> {
                    String xInfo = "";
                    if (target instanceof TargetAmount) {
                        xInfo = "x" + target.getTargetAmount(selectedId) + " ";
                    }

                    String targetInfo = null;
                    Player player = game.getPlayer(selectedId);
                    if (player != null) {
                        targetInfo = player.getName();
                    }
                    if (targetInfo == null) {
                        MageObject object = game.getObject(selectedId);
                        if (object != null) {
                            targetInfo = object.getIdName();
                        }
                    }
                    if (targetInfo == null) {
                        StackObject stackObject = game.getState().getStack().getStackObject(selectedId);
                        if (stackObject != null) {
                            targetInfo = CardUtil.substring(stackObject.toString(), 20, "...");
                        }
                    }
                    if (targetInfo == null) {
                        targetInfo = "unknown";
                    }
                    allTargetsInfo.add(xInfo + targetInfo);
                });
            });
            targetsInfo = String.join(" + ", allTargetsInfo);
        }
        return abilityInfo + (targetsInfo.isEmpty() ? "" : " -> " + targetsInfo);
    }

    public float[] convertCreatureToFeatureVector(Permanent creature, Game game) {
        // TODO: We're setting a class global but also returning?!
        featureVector = new float[EMBEDDING_SIZE];

        // One-hot encode the zone type
        int index = 0;  
        featureVector[index++] = creature.getOwnerId().equals(game.getActivePlayerId()) ? 1.0f : 0.0f;
        featureVector[index++] = (float) creature.getPower().getValue();
        featureVector[index++] = (float) creature.getToughness().getValue();
        featureVector[index++] = (float) creature.getManaValue();
        featureVector[index++] = creature.isCreature() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isArtifact() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isEnchantment() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isLand() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isPlaneswalker() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isPermanent() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isInstant() ? 1.0f : 0.0f;
        featureVector[index++] = creature.isSorcery() ? 1.0f : 0.0f;
        // Is the card tapped?
        featureVector[index++] = creature.isTapped() ? 1.0f : 0.0f;

        // Add the text embedding of the text
        String cardText = String.join(" ", creature.getRules());
        float[] textEmbedding = EmbeddingManager.getEmbedding(cardText);
        System.arraycopy(textEmbedding, 0, featureVector, index, textEmbedding.length);

        return featureVector;
    }
} 