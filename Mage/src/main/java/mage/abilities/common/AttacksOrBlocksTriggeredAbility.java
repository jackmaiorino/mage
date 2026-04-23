package mage.abilities.common;

import mage.abilities.TriggeredAbilityImpl;
import mage.abilities.effects.Effect;
import mage.abilities.effects.common.CreateDelayedTriggeredAbilityEffect;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.events.GameEvent;

import java.util.EnumSet;
import java.util.Set;

public class AttacksOrBlocksTriggeredAbility extends TriggeredAbilityImpl {

    private static final Set<GameEvent.EventType> WATCHED_EVENT_TYPES =
            EnumSet.of(GameEvent.EventType.ATTACKER_DECLARED, GameEvent.EventType.CREATURE_BLOCKS);

    public AttacksOrBlocksTriggeredAbility(Effect effect, boolean optional) {
        super(Zone.BATTLEFIELD, effect, optional);
        if (effect instanceof CreateDelayedTriggeredAbilityEffect) {
            setTriggerPhrase("When {this} attacks or blocks, ");
        } else {
            setTriggerPhrase("Whenever {this} attacks or blocks, ");
        }
        this.withRuleTextReplacement(true); // default true to replace "{this}" with "it"
    }

    protected AttacksOrBlocksTriggeredAbility(final AttacksOrBlocksTriggeredAbility ability) {
        super(ability);
    }

    @Override
    public AttacksOrBlocksTriggeredAbility copy() {
        return new AttacksOrBlocksTriggeredAbility(this);
    }

    @Override
    public Set<GameEvent.EventType> getWatchedEventTypes() {
        return WATCHED_EVENT_TYPES;
    }

    @Override
    public boolean checkEventType(GameEvent event, Game game) {
        return event.getType() == GameEvent.EventType.ATTACKER_DECLARED || event.getType() == GameEvent.EventType.CREATURE_BLOCKS;
    }

    @Override
    public boolean checkTrigger(GameEvent event, Game game) {
        return getSourceId().equals((event.getType() == GameEvent.EventType.ATTACKER_DECLARED) ? event.getSourceId() : event.getTargetId());
    }
}
