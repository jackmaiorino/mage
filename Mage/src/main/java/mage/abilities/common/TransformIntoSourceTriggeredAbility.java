package mage.abilities.common;

import mage.abilities.TriggeredAbilityImpl;
import mage.abilities.effects.Effect;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.events.GameEvent;

import java.util.EnumSet;
import java.util.Set;

/**
 * @author TheElk801
 */
public class TransformIntoSourceTriggeredAbility extends TriggeredAbilityImpl {

    private static final Set<GameEvent.EventType> WATCHED_EVENT_TYPES =
            EnumSet.of(GameEvent.EventType.TRANSFORMED);

    public TransformIntoSourceTriggeredAbility(Effect effect) {
        this(effect, false);
    }

    public TransformIntoSourceTriggeredAbility(Effect effect, boolean optional) {
        this(effect, optional, false);
    }

    public TransformIntoSourceTriggeredAbility(Effect effect, boolean optional, boolean whenever) {
        super(Zone.BATTLEFIELD, effect, optional);
        setTriggerPhrase("When" + (whenever ? "ever" : "") + " this creature transforms into {this}, ");
    }

    private TransformIntoSourceTriggeredAbility(final TransformIntoSourceTriggeredAbility ability) {
        super(ability);
    }

    @Override
    public TransformIntoSourceTriggeredAbility copy() {
        return new TransformIntoSourceTriggeredAbility(this);
    }

    @Override
    public Set<GameEvent.EventType> getWatchedEventTypes() {
        return WATCHED_EVENT_TYPES;
    }

    @Override
    public boolean checkEventType(GameEvent event, Game game) {
        return event.getType() == GameEvent.EventType.TRANSFORMED;
    }

    @Override
    public boolean checkTrigger(GameEvent event, Game game) {
        return event.getTargetId().equals(this.getSourceId());
    }
}
