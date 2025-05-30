
package mage.cards.b;

import java.util.UUID;
import mage.abilities.Ability;
import mage.abilities.triggers.BeginningOfUpkeepTriggeredAbility;
import mage.abilities.effects.common.CreateTokenEffect;
import mage.abilities.effects.common.LoseLifeSourceControllerEffect;
import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.constants.CardType;
import mage.constants.SubType;
import mage.game.permanent.token.FaerieRogueToken;

/**
 *
 * @author Loki
 */
public final class Bitterblossom extends CardImpl {

    public Bitterblossom(UUID ownerId, CardSetInfo setInfo) {
        super(ownerId,setInfo,new CardType[]{CardType.KINDRED,CardType.ENCHANTMENT},"{1}{B}");
        this.subtype.add(SubType.FAERIE);

        // At the beginning of your upkeep, you lose 1 life and create a 1/1 black Faerie Rogue creature token with flying.
        Ability ability = new BeginningOfUpkeepTriggeredAbility(new LoseLifeSourceControllerEffect(1));
        ability.addEffect(new CreateTokenEffect(new FaerieRogueToken(), 1).concatBy("and"));
        this.addAbility(ability);
    }

    private Bitterblossom(final Bitterblossom card) {
        super(card);
    }

    @Override
    public Bitterblossom copy() {
        return new Bitterblossom(this);
    }
}
