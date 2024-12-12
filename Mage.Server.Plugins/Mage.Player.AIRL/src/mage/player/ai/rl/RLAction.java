package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.game.Game;
import mage.game.permanent.Permanent;

import java.util.UUID;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RLAction {
    private static Word2Vec word2Vec;
    
    static {
        try {
            // Load pre-trained Word2Vec model
            word2Vec = WordVectorSerializer.readWord2VecModel("path/to/pretrained/word2vec.bin");
        } catch (Exception e) {
            throw new RuntimeException("Failed to load Word2Vec model", e);
        }
    }

    public enum ActionType {
        PASS,
        ACTIVATE_ABILITY,
        ATTACK,
        BLOCK,
        MULLIGAN
    }

    private final ActionType type;
    private final Ability ability;
    private final UUID targetId;
    private float cost;

    public RLAction(ActionType type) {
        this(type, null, null);
    }

    public RLAction(ActionType type, Ability ability) {
        this(type, ability, null);
    }

    public RLAction(ActionType type, UUID targetId) {
        this(type, null, targetId);
    }

    private RLAction(ActionType type, Ability ability, UUID targetId, float cost) {
        this.type = type;
        this.ability = ability;
        this.targetId = targetId;
        this.cost = cost;
    }

    public Ability getAbility() {
        return ability;
    }

    public ActionType getType() {
        return type;
    }

    public UUID getTargetId() {
        return targetId;
    }

    public boolean execute(Game game, UUID playerId) {
        if (type == ActionType.PASS) {
            return false;
        }
        return true; // TODO: Implement actual execution
    }

    public float[] toFeatureVector(Game game) {
        float[] featureVector = new float[RLModel.ACTION_SIZE];

        // One-hot encode action type
        int typeIndex = type.ordinal();
        featureVector[typeIndex] = 1.0f;

        // Type-specific features
        switch (type) {
            case ACTIVATE_ABILITY:
                if (ability != null) {
                    // One-hot encode ability type
                    if (ability instanceof SpellAbility) {
                        featureVector[5] = 1.0f;
                    } else if (ability instanceof SomeOtherAbilityType) {
                        featureVector[6] = 1.0f;
                    }
                    // Add more ability types as needed

                    featureVector[7] = ability.getManaCosts().manaValue(); // Example: mana cost
                    featureVector[8] = ability.getEffects().size(); // Example: number of effects

                    // Example: encode specific effects
                    for (Effect effect : ability.getEffects()) {
                        if (effect instanceof DrawCardEffect) {
                            featureVector[9] += ((DrawCardEffect) effect).getAmount();
                        }
                        if (effect instanceof MillCardsEffect) {
                            featureVector[10] += ((MillCardsEffect) effect).getAmount();
                        }
                        // Add more effect-based features as needed
                    }

                    // Use text embeddings for complex ability descriptions
                    String abilityText = ability.getText();
                    float[] textEmbedding = getTextEmbedding(abilityText);
                    System.arraycopy(textEmbedding, 0, featureVector, 11, textEmbedding.length);
                }
                break;
            case ATTACK:
                if (targetId != null) {
                    // Example: encode target's power and toughness
                    Permanent target = game.getPermanent(targetId);
                    if (target != null) {
                        featureVector[5] = target.getPower().getValue();
                        featureVector[6] = target.getToughness().getValue();
                    }
                }
                break;
            case BLOCK:
                // Add block-specific features
                // TODO: Implement block-specific features
                break;
            case MULLIGAN:
                // Add mulligan-specific features
                // TODO: Implement mulligan-specific features
                break;
            default:
                break;
        }

        // Ensure the vector is fully populated
        for (int i = 11 + textEmbedding.length; i < featureVector.length; i++) {
            featureVector[i] = 0.0f; // Zero padding
        }

        return featureVector;
    }

    public float[] convertActivatedAbilityToFeatureVector(ActivatedAbility ability) {
        float[] featureVector = new float[74]; // 10 for class embedding + 4 for basic features + 10 for text embedding + padding
        
        // Get class name embedding and add it to the feature vector
        String className = ability.getClass().getSimpleName();
        float[] classEmbedding = getClassNameEmbedding(className);
        System.arraycopy(classEmbedding, 0, featureVector, 0, classEmbedding.length);
        
        // Add basic ability features
        int index = classEmbedding.length;
        featureVector[index++] = ability.getManaCosts().manaValue(); // Total mana cost
        featureVector[index++] = ability.getEffects().size(); // Number of effects
        
        // Encode specific effects
        for (Effect effect : ability.getEffects()) {
            if (effect instanceof DrawCardEffect) {
                featureVector[index++] += ((DrawCardEffect) effect).getAmount();
            }
            if (effect instanceof MillCardsEffect) {
                featureVector[index++] += ((MillCardsEffect) effect).getAmount();
            }
            // Add more effect-based features as needed
        }
        
        // Use text embeddings for ability text
        String abilityText = ability.getText();
        float[] textEmbedding = getTextEmbedding(abilityText);
        System.arraycopy(textEmbedding, 0, featureVector, index, textEmbedding.length);
        
        // Zero padding for remaining slots
        for (int i = index + textEmbedding.length; i < featureVector.length; i++) {
            featureVector[i] = 0.0f;
        }
        
        return featureVector;
    }

    private float[] getTextEmbedding(String text) {
        // Implement text embedding logic here, possibly using a pre-trained NLP model
        // TODO: Implement text embedding logic
        return new float[10]; // Example: return a fixed-size embedding
    }

    private float[] getClassNameEmbedding(String className) {
        // Split camelCase into words
        String[] words = className.split("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])");
        
        // Get embedding for each word and average them
        INDArray sum = Nd4j.zeros(word2Vec.getLayerSize());
        int count = 0;
        
        for (String word : words) {
            if (word2Vec.hasWord(word.toLowerCase())) {
                sum.addi(word2Vec.getWordVectorMatrix(word.toLowerCase()));
                count++;
            }
        }
        
        // If no words were found in the vocabulary, return zero vector
        if (count == 0) {
            return new float[word2Vec.getLayerSize()];
        }
        
        // Average the word vectors
        INDArray averaged = sum.div(count);
        return averaged.toFloatVector();
    }
} 