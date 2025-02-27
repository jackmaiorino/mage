package mage.cards.a;

import mage.abilities.common.BecomesTargetControllerTriggeredAbility;
import mage.abilities.common.SimpleStaticAbility;
import mage.abilities.costs.mana.GenericManaCost;
import mage.abilities.effects.common.CounterUnlessPaysEffect;
import mage.abilities.effects.common.continuous.BoostAllEffect;
import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.constants.CardType;
import mage.constants.Duration;
import mage.constants.SetTargetPointer;
import mage.constants.Zone;
import mage.filter.StaticFilters;

import java.util.UUID;

/**
 *
 * @author TheElk801
 */
public final class AmuletOfSafekeeping extends CardImpl {

    public AmuletOfSafekeeping(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId, setInfo, new CardType[]{CardType.ARTIFACT}, "{2}");

        // Whenever you become the target of a spell or ability an opponent controls, counter that spell or ability unless its controller pays {1}.
        this.addAbility(new BecomesTargetControllerTriggeredAbility(new CounterUnlessPaysEffect(new GenericManaCost(1)),
                null, StaticFilters.FILTER_SPELL_OR_ABILITY_OPPONENTS, SetTargetPointer.SPELL, false));

        // Creature tokens get -1/-0.
        this.addAbility(new SimpleStaticAbility(
                new BoostAllEffect(
                        -1, 0, Duration.WhileOnBattlefield,
                        StaticFilters.FILTER_CREATURE_TOKENS, false
                )
        ));
    }

    private AmuletOfSafekeeping(final AmuletOfSafekeeping card) {
        super(card);
    }

    @Override
    public AmuletOfSafekeeping copy() {
        return new AmuletOfSafekeeping(this);
    }
}
