package mage.player.ai.rl;

import mage.cards.Card;
import mage.game.permanent.Permanent;

public class CardState {
    private int attack;
    private int defense;
    private int life;
    private boolean isTapped;

    public CardState(Card card) {
        if (card instanceof Permanent) {
            Permanent permanent = (Permanent) card;
            this.attack = permanent.getPower().getValue();
            this.defense = permanent.getToughness().getValue();
            this.life = permanent.getDamage();
            this.isTapped = permanent.isTapped();
        } else {
            this.attack = 0;
            this.defense = 0;
            this.life = 0;
            this.isTapped = false;
        }
    }

    public int getAttack() { return attack; }
    public int getDefense() { return defense; }
    public int getLife() { return life; }
    public boolean isTapped() { return isTapped; }
} 