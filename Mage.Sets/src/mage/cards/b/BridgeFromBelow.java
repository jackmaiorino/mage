package mage.cards.b;

import mage.abilities.TriggeredAbilityImpl;
import mage.abilities.condition.common.SourceInGraveyardCondition;
import mage.abilities.effects.Effect;
import mage.abilities.effects.common.CreateTokenEffect;
import mage.abilities.effects.common.ExileSourceEffect;
import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.constants.CardType;
import mage.constants.TargetController;
import mage.constants.Zone;
import mage.filter.common.FilterCreaturePermanent;
import mage.filter.predicate.permanent.TokenPredicate;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.events.ZoneChangeEvent;
import mage.game.permanent.Permanent;
import mage.game.permanent.token.ZombieToken;

import java.util.UUID;

/**
 *
 * @author Plopman
 */
public final class BridgeFromBelow extends CardImpl {

    private static final FilterCreaturePermanent filter1 = new FilterCreaturePermanent("Whenever a nontoken creature is put into your graveyard from the battlefield, ");
    private static final FilterCreaturePermanent filter2 = new FilterCreaturePermanent("When a creature is put into an opponent's graveyard from the battlefield, ");

    static {
        filter1.add(TargetController.YOU.getOwnerPredicate());
        filter1.add(TokenPredicate.FALSE);
        filter2.add(TargetController.OPPONENT.getOwnerPredicate());
    }

    public BridgeFromBelow(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId, setInfo, new CardType[]{CardType.ENCHANTMENT}, "{B}{B}{B}");

        // Whenever a nontoken creature is put into your graveyard from the battlefield, if Bridge from Below is in your graveyard, create a 2/2 black Zombie creature token.
        this.addAbility(new BridgeFromBelowAbility(new CreateTokenEffect(new ZombieToken()), filter1));

        // When a creature is put into an opponent's graveyard from the battlefield, if Bridge from Below is in your graveyard, exile Bridge from Below.
        this.addAbility(new BridgeFromBelowAbility(new ExileSourceEffect(), filter2));

    }

    private BridgeFromBelow(final BridgeFromBelow card) {
        super(card);
    }

    @Override
    public BridgeFromBelow copy() {
        return new BridgeFromBelow(this);
    }
}

class BridgeFromBelowAbility extends TriggeredAbilityImpl {

    private final FilterCreaturePermanent filter;

    BridgeFromBelowAbility(Effect effect, FilterCreaturePermanent filter) {
        super(Zone.GRAVEYARD, effect, false);
        this.filter = filter;
        this.withInterveningIf(SourceInGraveyardCondition.instance);
        setTriggerPhrase(filter.getMessage());
    }

    private BridgeFromBelowAbility(final BridgeFromBelowAbility ability) {
        super(ability);
        this.filter = ability.filter;
    }

    @Override
    public BridgeFromBelowAbility copy() {
        return new BridgeFromBelowAbility(this);
    }

    @Override
    public boolean checkEventType(GameEvent event, Game game) {
        return event.getType() == GameEvent.EventType.ZONE_CHANGE;
    }

    @Override
    public boolean checkTrigger(GameEvent event, Game game) {
        Permanent permanent = (Permanent) game.getLastKnownInformation(event.getTargetId(), Zone.BATTLEFIELD);
        if (permanent == null || !((ZoneChangeEvent) event).isDiesEvent()) {
            return false;
        }
        return filter.match(permanent, controllerId, this, game);
    }

}
