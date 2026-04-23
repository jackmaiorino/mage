
package mage.abilities.common;

import mage.MageObjectReference;
import mage.abilities.TriggeredAbilityImpl;
import mage.abilities.effects.Effect;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.watchers.common.AttackedThisTurnWatcher;

import java.util.EnumSet;
import java.util.Set;

/**
 * @author TheElk801
 */
public class AttacksFirstTimeTriggeredAbility extends TriggeredAbilityImpl {

    private static final Set<GameEvent.EventType> WATCHED_EVENT_TYPES =
            EnumSet.of(GameEvent.EventType.ATTACKER_DECLARED);

    public AttacksFirstTimeTriggeredAbility(Effect effect, boolean optional) {
        super(Zone.BATTLEFIELD, effect, optional);
        setTriggerPhrase("Whenever {this} attacks for the first time each turn, ");
    }

    protected AttacksFirstTimeTriggeredAbility(final AttacksFirstTimeTriggeredAbility ability) {
        super(ability);
    }

    @Override
    public Set<GameEvent.EventType> getWatchedEventTypes() {
        return WATCHED_EVENT_TYPES;
    }

    @Override
    public boolean checkEventType(GameEvent event, Game game) {
        return event.getType() == GameEvent.EventType.ATTACKER_DECLARED;
    }

    @Override
    public boolean checkTrigger(GameEvent event, Game game) {
        if (!event.getSourceId().equals(this.getSourceId())) {
            return false;
        }
        AttackedThisTurnWatcher watcher = game.getState().getWatcher(AttackedThisTurnWatcher.class);
        if (watcher == null) {
            return false;
        }
        Permanent sourcePerm = game.getPermanentOrLKIBattlefield(event.getSourceId());
        if (sourcePerm == null) {
            return false;
        }
        for (MageObjectReference mor : watcher.getAttackedThisTurnCreaturesCounts().keySet()) {
            if (mor.refersTo(sourcePerm, game)
                    && watcher.getAttackedThisTurnCreaturesCounts().get(mor) > 1) {
                return false;
            }
        }
        return true;
    }

    @Override
    public AttacksFirstTimeTriggeredAbility copy() {
        return new AttacksFirstTimeTriggeredAbility(this);
    }
}
