package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.game.Game;
import java.util.UUID;

public class RLAction {
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

    public float[] toFeatureVector() {
        float[] featureVector = new float[RLModel.ACTION_SIZE];

        // Encode action type
        featureVector[0] = type.ordinal();

        // Type-specific features
        switch (type) {
            case ACTIVATE_ABILITY:
                if (ability != null) {
                    featureVector[1] = ability.getManaCosts().manaValue(); // Example: mana cost
                    featureVector[2] = ability.getEffects().size(); // Example: number of effects
                }
                break;
            case ATTACK:
                if (targetId != null) {
                    // Example: encode target's power and toughness
                    Permanent target = game.getPermanent(targetId);
                    if (target != null) {
                        featureVector[1] = target.getPower().getValue();
                        featureVector[2] = target.getToughness().getValue();
                    }
                }
                break;
            case BLOCK:
                // Add block-specific features
                break;
            case MULLIGAN:
                // Add mulligan-specific features
                break;
            default:
                break;
        }

        // Ensure the vector is fully populated
        for (int i = 3; i < featureVector.length; i++) {
            featureVector[i] = 0.0f; // Zero padding
        }

        return featureVector;
    }
} 