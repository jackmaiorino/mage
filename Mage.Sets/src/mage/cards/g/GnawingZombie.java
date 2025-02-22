package mage.cards.g;

import mage.MageInt;
import mage.abilities.Ability;
import mage.abilities.common.SimpleActivatedAbility;
import mage.abilities.costs.common.SacrificeTargetCost;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.effects.common.GainLifeEffect;
import mage.abilities.effects.common.LoseLifeTargetEffect;
import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.constants.CardType;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.target.TargetPlayer;
import mage.target.common.TargetControlledCreaturePermanent;

import java.util.UUID;

import mage.filter.StaticFilters;

/**
 * @author LevelX2
 */
public final class GnawingZombie extends CardImpl {

    public GnawingZombie(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId, setInfo, new CardType[]{CardType.CREATURE}, "{1}{B}");
        this.subtype.add(SubType.ZOMBIE);

        this.power = new MageInt(1);
        this.toughness = new MageInt(3);

        // {1}{B}, Sacrifice a creature: Target player loses 1 life and you gain 1 life.
        Ability ability = new SimpleActivatedAbility(new LoseLifeTargetEffect(1), new ManaCostsImpl<>("{1}{B}"));
        ability.addEffect(new GainLifeEffect(1).concatBy("and"));
        ability.addTarget(new TargetPlayer());
        ability.addCost(new SacrificeTargetCost(StaticFilters.FILTER_PERMANENT_CREATURE));
        this.addAbility(ability);
    }

    private GnawingZombie(final GnawingZombie card) {
        super(card);
    }

    @Override
    public GnawingZombie copy() {
        return new GnawingZombie(this);
    }
}
