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

    public RLAction(ActionType type) {
        this(type, null, null);
    }

    public RLAction(ActionType type, Ability ability) {
        this(type, ability, null);
    }

    public RLAction(ActionType type, UUID targetId) {
        this(type, null, targetId);
    }

    private RLAction(ActionType type, Ability ability, UUID targetId) {
        this.type = type;
        this.ability = ability;
        this.targetId = targetId;
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
        // Convert action to feature vector
        float[] features = new float[RLModel.ACTION_SIZE];
        // TODO: Implement action feature extraction
        return features;
    }
} 