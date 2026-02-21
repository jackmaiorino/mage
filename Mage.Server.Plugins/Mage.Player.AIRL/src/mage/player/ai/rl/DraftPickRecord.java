package mage.player.ai.rl;

import mage.cards.Card;

/**
 * Tracks a single pick event during a draft: either a card picked by the RL
 * agent, or a card seen in a pack but not picked (for signal reading).
 */
public class DraftPickRecord {

    public final Card card;
    public final int packNumber;  // 1-3
    public final int pickNumber;  // 1-15
    public final boolean picked;  // true = drafted into pool, false = seen/passed

    public DraftPickRecord(Card card, int packNumber, int pickNumber, boolean picked) {
        this.card = card;
        this.packNumber = packNumber;
        this.pickNumber = pickNumber;
        this.picked = picked;
    }
}
