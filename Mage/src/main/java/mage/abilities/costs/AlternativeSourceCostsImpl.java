package mage.abilities.costs;

import mage.ConditionalMana;
import mage.Mana;
import mage.abilities.Ability;
import mage.abilities.StaticAbility;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.costs.mana.ManaCosts;
import mage.abilities.costs.mana.ManaCostsImpl;
import mage.abilities.mana.ManaOptions;
import mage.constants.AbilityType;
import mage.constants.Zone;
import mage.game.Game;
import mage.players.Player;
import mage.players.PlayerImpl;
import mage.util.CardUtil;

import java.util.Iterator;

/**
 * @author TheElk801
 */
public abstract class AlternativeSourceCostsImpl extends StaticAbility implements AlternativeSourceCosts {

    protected final AlternativeCost alternativeCost;
    protected final String reminderText;
    protected final String activationKey;

    protected static String getActivationKey(String name) {
        return name + "ActivationKey";
    }

    protected AlternativeSourceCostsImpl(String name, String reminderText, String manaString) {
        this(name, reminderText, new ManaCostsImpl<>(manaString));
    }

    protected AlternativeSourceCostsImpl(String name, String reminderText, Cost cost) {
        this(name, reminderText, cost, name);
    }

    protected AlternativeSourceCostsImpl(String name, String reminderText, Cost cost, String activationKey) {
        super(Zone.ALL, null);
        this.name = name;
        this.reminderText = reminderText;
        this.alternativeCost = new AlternativeCostImpl<>(name, reminderText, cost);
        this.activationKey = getActivationKey(activationKey);
    }

    protected AlternativeSourceCostsImpl(final AlternativeSourceCostsImpl ability) {
        super(ability);
        this.alternativeCost = ability.alternativeCost.copy();
        this.reminderText = ability.reminderText;
        this.activationKey = ability.activationKey;
    }


    @Override
    public boolean canActivateAlternativeCostsNow(Ability ability, Game game) {
        if (ability == null || !AbilityType.SPELL.equals(ability.getAbilityType())) {
            return isActivated(ability, game);
        }
        Player player = game.getPlayer(ability.getControllerId());
        boolean canPayAllCosts = false;
        // TODO: There can likely be alternative costs that are not mana costs, so this will need to be updated
        // TODO: alternative cost is actually a list and getCost returns the first cost, we just need to iterate over all costs
        if (player != null) {
            ManaOptions availableMana = ((PlayerImpl)player).getManaAvailable(game, ability.getSourceObject(game));
            canPayAllCosts = false;
            for (Mana mana : availableMana) {
                if (!(alternativeCost.getCost() instanceof ManaCostsImpl)){
                    throw new RuntimeException("Alternative cost is not a ManaCostsImpl");
                }
                if (((ManaCostsImpl) alternativeCost.getCost()).getMana().enough(mana)){
                    if (mana instanceof ConditionalMana) {
                        if (((ConditionalMana)mana).apply(ability,game,ability.getOriginalId(),alternativeCost.getCost())) {
                            canPayAllCosts = true;
                            break;
                        }
                    } else {
                        canPayAllCosts = true;
                        break;
                    }
                }
            }

        }
        return ((ManaCostsImpl) alternativeCost.getCost()).isEmpty() || canPayAllCosts;
    }

    @Override
    public String getAlternativeCostText(Ability ability, Game game) {
        return "Cast with " + this.name + " alternative cost: " + alternativeCost.getText(true) + CardUtil.getSourceLogName(game, this);
    }

    @Override
    public boolean activateAlternativeCosts(Ability ability, Game game) {
        this.resetCost();
        ability.setCostsTag(activationKey, null);
        alternativeCost.activate();

        ability.clearManaCostsToPay();
        ability.clearCosts();
        for (Iterator<Cost> it = ((Costs<Cost>) alternativeCost).iterator(); it.hasNext(); ) {
            Cost cost = it.next();
            if (cost instanceof ManaCost) {
                ability.addManaCostsToPay((ManaCost) cost.copy());
            } else {
                ability.addCost(cost.copy());
            }
        }
        return true;
    }

    @Override
    public boolean isActivated(Ability ability, Game game) {
        return CardUtil.checkSourceCostsTagExists(game, ability, activationKey);
    }

    @Override
    public Costs<Cost> getCosts() {
        return (Costs<Cost>) alternativeCost;
    }

    @Override
    public String getRule() {
        return alternativeCost.getText(false) + ' ' + alternativeCost.getReminderText();
    }

    @Override
    public void resetCost() {
        alternativeCost.reset();
    }

    @Override
    public boolean isAvailable(Ability source, Game game) {
        return true;
    }

    public String getCastMessageSuffix(Game game) {
        return alternativeCost.getCastSuffixMessage(0);
    }
}
