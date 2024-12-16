package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.ActivatedAbility;
import mage.abilities.Ability;
import mage.game.Game;

import java.util.ArrayList;
import java.util.List;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.TargetAmount;
import mage.util.CardUtil;

public class RLAction {
    public enum ActionType {
        PASS_PRIORITY,
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
    private float[] featureVector;
    public static final int MAX_ACTIONS = 10;
    public static int EMBEDDING_SIZE = 0;
    public static int FEATURE_VECTOR_SIZE = 0;

    public RLAction(ActionType type, List<ActivatedAbility> abilities, List<Permanent> creatures, Game game) {
        this.type = type;
        this.abilities = abilities;
        this.creatures = creatures;
        this.game = game;
        EMBEDDING_SIZE = EmbeddingManager.REDUCED_EMBEDDING_SIZE + 4; // 4 for basic features
        FEATURE_VECTOR_SIZE = MAX_ACTIONS * EMBEDDING_SIZE + ActionType.values().length; // Include space for one-hot encoding
        
        this.featureVector = toFeatureVector(game);
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

    private float[] toFeatureVector(Game game) {
        float[] featureVector = new float[FEATURE_VECTOR_SIZE];
        int index = 0;

        // One-hot encode action type
        int typeIndex = type.ordinal();
        featureVector[typeIndex] = 1.0f;
        index += ActionType.values().length;

        // Type-specific features
        switch (type) {
            case ACTIVATE_ABILITY_OR_SPELL:
                if (abilities != null) {
                    for (ActivatedAbility ability : abilities) {
                        if (index >= FEATURE_VECTOR_SIZE) {
                            throw new IllegalStateException("Too many abilities");
                        }
                        float[] abilityFeatures = convertAbilityOrSpellToFeatureVector(ability);
                        System.arraycopy(abilityFeatures, 0, featureVector, index, abilityFeatures.length);
                        index += abilityFeatures.length;
                    }
                }
                break;
            case DECLARE_ATTACKS:
                if (creatures != null) {
                    // TODO: implement
                }
                break;
            case DECLARE_BLOCKS:
                // Add block-specific features
                // TODO: Implement block-specific features
                break;
            case MULLIGAN:
                // TODO: Implement mulligan-specific features
                break;
            case PASS_PRIORITY:
                // TODO: Implement pass priority-specific features
                break;
            default:
                break;
        }

        // Zero padding for remaining slots
        for (int i = index; i < featureVector.length; i++) {
            featureVector[i] = 0.0f;
        }

        return featureVector;
    }

    private float[] convertAbilityOrSpellToFeatureVector(ActivatedAbility ability) {
        float[] featureVector = new float[EMBEDDING_SIZE];

        // Get class name embedding and add it to the feature vector
        String className = ability.getClass().getSimpleName();
        float[] classEmbedding = EmbeddingManager.getEmbedding(className);
        System.arraycopy(classEmbedding, 0, featureVector, 0, classEmbedding.length);

        // Add basic ability features
        int index = classEmbedding.length;
        featureVector[index++] = ability.getManaCosts().manaValue(); // Total mana cost
        featureVector[index++] = ability.getEffects().size(); // Number of effects
        featureVector[index++] = ability.getTargets().size(); // Number of targets
        featureVector[index++] = ability.getCosts().size(); // Number of costs

        // Encode Card Text
        // TODO: This will likely not include all card text for casting creatures
        // i.e. it will say something like "Cast a 2/2 red Dragon"
        // but not "Cast a red Dragon with 2/2 with flying and lifeline"
        // not sure will need to test
        String abilityText = getAbilityAndSourceInfo(game, ability, false);
        float[] textEmbedding = EmbeddingManager.getEmbedding(abilityText);
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
} 