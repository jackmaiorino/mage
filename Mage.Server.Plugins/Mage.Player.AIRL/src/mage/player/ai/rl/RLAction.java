package mage.player.ai.rl;

import mage.ApprovingObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.SpellAbility;
import mage.game.Game;
import mage.game.combat.Combat;
import mage.game.permanent.Permanent;
import mage.players.Player;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class RLAction {
    public enum ActionType {
        CAST_SPELL,
        ATTACK,
        BLOCK,
        ACTIVATE_ABILITY,
        PASS
    }

    private ActionType type;
    private UUID sourceId;
    private UUID targetId;
    private Ability ability;

    public RLAction(ActionType type) {
        this.type = type;
    }

    public RLAction(ActionType type, UUID sourceId) {
        this.type = type;
        this.sourceId = sourceId;
    }

    public RLAction(ActionType type, UUID sourceId, UUID targetId) {
        this.type = type;
        this.sourceId = sourceId;
        this.targetId = targetId;
    }

    public RLAction(ActionType type, Ability ability) {
        this.type = type;
        this.ability = ability;
    }

    public boolean execute(Game game, UUID playerId) {
        Player player = game.getPlayer(playerId);
        if (player == null) return false;

        switch (type) {
            case CAST_SPELL:
                if (ability instanceof SpellAbility) {
                    return player.cast((SpellAbility)ability, game, false, new ApprovingObject(ability, game));
                }
                return false;
            case ATTACK:
                Combat combat = game.getCombat();
                if (combat != null) {
                    return combat.declareAttacker(sourceId, targetId, playerId, game);
                }
                return false;
            case BLOCK:
                combat = game.getCombat();
                if (combat != null) {
                    List<UUID> attackers = new ArrayList<>(combat.getAttackers());
                    if (!attackers.isEmpty()) {
                        player.declareBlocker(playerId, sourceId, attackers.get(0), game);
                        return true;
                    }
                }
                return false;
            case ACTIVATE_ABILITY:
                if (ability instanceof ActivatedAbility) {
                    return player.activateAbility((ActivatedAbility)ability, game);
                }
                return false;
            case PASS:
                return true;
            default:
                return false;
        }
    }

    public void setAbility(Ability ability) {
        this.ability = ability;
    }

    public ActionType getType() {
        return type;
    }

    public UUID getSourceId() {
        return sourceId;
    }

    public UUID getTargetId() {
        return targetId;
    }
} 