package mage.cards.p;

import mage.abilities.Ability;
import mage.abilities.triggers.BeginningOfUpkeepTriggeredAbility;
import mage.abilities.effects.OneShotEffect;
import mage.abilities.effects.common.AttachEffect;
import mage.abilities.effects.common.CreateTokenEffect;
import mage.abilities.keyword.EnchantAbility;
import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.constants.CardType;
import mage.constants.Outcome;
import mage.constants.SubType;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.token.PhyrexianMyrToken;
import mage.target.TargetPermanent;
import mage.target.common.TargetCreaturePermanent;

import java.util.UUID;

/**
 * @author Loki
 */
public final class ParasiticImplant extends CardImpl {

    public ParasiticImplant(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId, setInfo, new CardType[]{CardType.ENCHANTMENT}, "{3}{B}");
        this.subtype.add(SubType.AURA);

        TargetPermanent auraTarget = new TargetCreaturePermanent();
        this.getSpellAbility().addTarget(auraTarget);
        this.getSpellAbility().addEffect(new AttachEffect(Outcome.Sacrifice));
        Ability ability = new EnchantAbility(auraTarget);
        this.addAbility(ability);
        ability = new BeginningOfUpkeepTriggeredAbility(new ParasiticImplantEffect());
        ability.addEffect(new CreateTokenEffect(new PhyrexianMyrToken()).concatBy("and you"));
        this.addAbility(ability);
    }

    private ParasiticImplant(final ParasiticImplant card) {
        super(card);
    }

    @Override
    public ParasiticImplant copy() {
        return new ParasiticImplant(this);
    }
}

class ParasiticImplantEffect extends OneShotEffect {
    ParasiticImplantEffect() {
        super(Outcome.Sacrifice);
        staticText = "enchanted creature's controller sacrifices it";
    }

    private ParasiticImplantEffect(final ParasiticImplantEffect effect) {
        super(effect);
    }


    @Override
    public boolean apply(Game game, Ability source) {
        Permanent enchantment = game.getPermanent(source.getSourceId());
        if (enchantment != null && enchantment.getAttachedTo() != null) {
            Permanent creature = game.getPermanent(enchantment.getAttachedTo());
            if (creature != null) {
                return creature.sacrifice(source, game);
            }
        }
        return false;
    }

    @Override
    public ParasiticImplantEffect copy() {
        return new ParasiticImplantEffect(this);
    }
}
