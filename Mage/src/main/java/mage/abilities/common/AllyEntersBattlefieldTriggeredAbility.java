
package mage.abilities.common;

import mage.abilities.TriggeredAbilityImpl;
import mage.abilities.effects.Effect;
import mage.constants.AbilityWord;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.events.EntersTheBattlefieldEvent;
import mage.game.events.GameEvent;

import java.util.EnumSet;
import java.util.Set;

/**
 * @author North
 */
public class AllyEntersBattlefieldTriggeredAbility extends TriggeredAbilityImpl {

    private static final Set<GameEvent.EventType> WATCHED_EVENT_TYPES =
            EnumSet.of(GameEvent.EventType.ENTERS_THE_BATTLEFIELD);

    public AllyEntersBattlefieldTriggeredAbility(Effect effect, boolean optional) {
        super(Zone.BATTLEFIELD, effect, optional);
        this.setAbilityWord(AbilityWord.RALLY);
        setTriggerPhrase("Whenever {this} or another Ally you control enters, ");
    }

    public AllyEntersBattlefieldTriggeredAbility(AllyEntersBattlefieldTriggeredAbility ability) {
        super(ability);
    }

    @Override
    public Set<GameEvent.EventType> getWatchedEventTypes() {
        return WATCHED_EVENT_TYPES;
    }

    @Override
    public boolean checkEventType(GameEvent event, Game game) {
        return event.getType() == GameEvent.EventType.ENTERS_THE_BATTLEFIELD;
    }

    @Override
    public boolean checkTrigger(GameEvent event, Game game) {
        EntersTheBattlefieldEvent ebe = (EntersTheBattlefieldEvent) event;
        return ebe.getTarget().isControlledBy(this.controllerId)
                && (event.getTargetId().equals(this.getSourceId())
                || (ebe.getTarget().hasSubtype(SubType.ALLY, game) && !event.getTargetId().equals(this.getSourceId())));
    }

    @Override
    public AllyEntersBattlefieldTriggeredAbility copy() {
        return new AllyEntersBattlefieldTriggeredAbility(this);
    }
}
