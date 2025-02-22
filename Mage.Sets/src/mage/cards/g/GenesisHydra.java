package mage.cards.g;

import mage.MageInt;
import mage.abilities.Ability;
import mage.abilities.SpellAbility;
import mage.abilities.common.EntersBattlefieldAbility;
import mage.abilities.effects.OneShotEffect;
import mage.abilities.effects.common.CastSourceTriggeredAbility;
import mage.abilities.effects.common.EntersBattlefieldWithXCountersEffect;
import mage.cards.*;
import mage.constants.*;
import mage.counters.CounterType;
import mage.filter.FilterCard;
import mage.filter.common.FilterPermanentCard;
import mage.filter.predicate.Predicates;
import mage.filter.predicate.mageobject.ManaValuePredicate;
import mage.game.Game;
import mage.players.Player;
import mage.target.TargetCard;
import mage.util.CardUtil;

import java.util.UUID;

/**
 *
 * @author LevelX2
 */
public final class GenesisHydra extends CardImpl {

    public GenesisHydra(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId, setInfo, new CardType[]{CardType.CREATURE}, "{X}{G}{G}");
        this.subtype.add(SubType.PLANT);
        this.subtype.add(SubType.HYDRA);

        this.power = new MageInt(0);
        this.toughness = new MageInt(0);

        // When you cast Genesis Hydra, reveal the top X cards of your library. You may put a nonland permanent card with converted mana cost X or less from among them onto the battlefield. Then shuffle the rest into your library.
        this.addAbility(new CastSourceTriggeredAbility(new GenesisHydraPutOntoBattlefieldEffect(), false));

        // Genesis Hydra enters the battlefield with X +1/+1 counters on it.
        this.addAbility(new EntersBattlefieldAbility(new EntersBattlefieldWithXCountersEffect(CounterType.P1P1.createInstance())));
    }

    private GenesisHydra(final GenesisHydra card) {
        super(card);
    }

    @Override
    public GenesisHydra copy() {
        return new GenesisHydra(this);
    }
}

class GenesisHydraPutOntoBattlefieldEffect extends OneShotEffect {

    GenesisHydraPutOntoBattlefieldEffect() {
        super(Outcome.PutCardInPlay);
        staticText = "reveal the top X cards of your library. You may put a nonland permanent card with mana value X or less from among them onto the battlefield. Then shuffle the rest into your library";
    }

    private GenesisHydraPutOntoBattlefieldEffect(final GenesisHydraPutOntoBattlefieldEffect effect) {
        super(effect);
    }

    @Override
    public boolean apply(Game game, Ability source) {
        Player controller = game.getPlayer(source.getControllerId());
        Object obj = getValue(CastSourceTriggeredAbility.SOURCE_CAST_SPELL_ABILITY);
        if (controller != null && obj instanceof SpellAbility) {
            int count = CardUtil.getSourceCostsTag(game, ((SpellAbility) obj), "X", 0);
            if (count > 0) {
                Cards cards = new CardsImpl(controller.getLibrary().getTopCards(game, count));
                controller.revealCards(source, cards, game);
                FilterCard filter = new FilterPermanentCard("a nonland permanent card with mana value " + count + " or less to put onto the battlefield");
                filter.add(Predicates.not(CardType.LAND.getPredicate()));
                filter.add(new ManaValuePredicate(ComparisonType.FEWER_THAN, count + 1));
                TargetCard target1 = new TargetCard(0, 1, Zone.LIBRARY, filter);
                if (cards.count(filter, source.getSourceId(), source, game) > 0) {
                    if (controller.choose(Outcome.PutCardInPlay, cards, target1, source, game)) {
                        Card card = cards.get(target1.getFirstTarget(), game);
                        if (card != null) {
                            cards.remove(card);
                            controller.moveCards(card, Zone.BATTLEFIELD, source, game);
                        }
                        target1.clearChosen();
                    } else {
                        game.informPlayers(controller.getLogName() + " didn't choose anything");
                    }
                } else {
                    game.informPlayers("No nonland permanent card with mana value " + count + " or less to choose.");
                }
                if (!cards.isEmpty()) {
                    controller.moveCards(cards, Zone.LIBRARY, source, game);
                    controller.shuffleLibrary(source, game);
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public GenesisHydraPutOntoBattlefieldEffect copy() {
        return new GenesisHydraPutOntoBattlefieldEffect(this);
    }

}
