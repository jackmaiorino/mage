package mage.game;

import mage.cards.Card;
import mage.constants.Zone;
import mage.players.Player;
import org.apache.log4j.Logger;

import java.util.UUID;

/**
 * Fail-fast zone/container consistency checks (ReferenceRules v2, Sol #106).
 * <p>
 * Root bug this guards against: {@link GameState#addCard(mage.cards.Card)} zones every
 * card OUTSIDE at load time (deliberate, rules-modeled for sideboard cards that never
 * enter a real zone). Historically nothing re-zoned an un-drawn library card from
 * OUTSIDE to LIBRARY before some caller order could leave it that way, and
 * {@code CardImpl.removeFromZone}'s OUTSIDE branch silently reported success without
 * touching the physical {@code Library} container - so a zone-routed effect (mill,
 * impulse-draw exile, surveil, search) would duplicate the card: still physically in
 * the library, also placed in the destination zone. {@link GameImpl#init} now
 * reconciles every library card's recorded zone to LIBRARY once, before first
 * untap/hand-dealing, which is the real fix. This class is the belt-and-suspenders
 * check: any future code path that re-introduces a zone/container mismatch trips
 * loudly (an exception) instead of silently duplicating a card.
 * <p>
 * Scope: only the zones where a card's own {@link UUID} is tracked directly inside a
 * per-player (or global) container - LIBRARY, HAND, GRAVEYARD, EXILED. BATTLEFIELD,
 * STACK and COMMAND are intentionally excluded: those hold distinct Permanent/Spell/
 * CommandObject identities (CR 400.7 "new object" rule), not the plain card id, so a
 * naive containment check there would false-positive on ordinary play.
 * <p>
 * Only checks for actual duplication (a card physically present in 2+ tracked
 * containers, or present in exactly one but recorded as a different zone) - NOT for
 * "recorded zone X but not physically present anywhere". That weaker check was tried
 * and reverted: it false-positives on two long-standing, legitimate engine idioms that
 * predate this fix and aren't the bug being guarded against here - (1) double-faced/
 * modal cards, where {@code ZonesHandler.placeInDestinationZone} stamps BOTH halves'
 * ids with the destination zone but only ever adds the single active-face card object
 * to the real container (Suppression Ray, Grist, Tamiyo, Sorin of House Markov all hit
 * this), and (2) "exile {this}, then return it transformed" self-effects (e.g.
 * {@code ExileAndReturnSourceEffect}), where the permanent briefly gets tagged EXILED
 * without ever actually being added to an exile zone container before immediately
 * returning to the battlefield. Neither case is a duplication risk (nothing is ever
 * findable in two places at once for them), so it's out of scope for this check.
 * <p>
 * Enabled by default (checks only run at zone-change commit points, not per-decision;
 * each is O(library/hand/graveyard/exile size), negligible next to per-game cost).
 * Set {@code MAGE_ZONE_INVARIANTS=0} (system property or environment variable) to
 * disable, e.g. if a future card outside the audited pool trips a false positive
 * before it can be fixed.
 */
public final class ZoneInvariants {

    private static final Logger logger = Logger.getLogger(ZoneInvariants.class);

    private static final boolean ENABLED = computeEnabled();

    private ZoneInvariants() {
    }

    private static boolean computeEnabled() {
        String v = System.getProperty("MAGE_ZONE_INVARIANTS");
        if (v == null) {
            v = System.getenv("MAGE_ZONE_INVARIANTS");
        }
        return v == null || !"0".equals(v);
    }

    public static boolean isEnabled() {
        return ENABLED;
    }

    /**
     * Verify that {@code cardId}'s recorded zone agrees with exactly the one physical
     * container it's actually sitting in, among the tracked zones. A no-op for
     * anything outside those zones (battlefield permanents, stack objects, sideboard
     * cards, tokens, cards not yet loaded).
     *
     * @param checkpoint short label identifying the call site, for diagnostics only
     */
    public static void checkCard(Game game, UUID cardId, String checkpoint) {
        if (!ENABLED || cardId == null || game == null) {
            return;
        }
        Card card = game.getCard(cardId);
        if (card == null) {
            return; // not a plain card (token/permanent-only object/unloaded): out of scope
        }
        Player owner = game.getPlayer(card.getOwnerId());
        if (owner == null) {
            return;
        }

        Zone recordedZone = game.getState().getZone(cardId);

        Zone foundIn = null;
        int foundCount = 0;
        if (owner.getLibrary().contains(cardId)) {
            foundIn = Zone.LIBRARY;
            foundCount++;
        }
        if (owner.getHand().contains(cardId)) {
            foundIn = Zone.HAND;
            foundCount++;
        }
        if (owner.getGraveyard().contains(cardId)) {
            foundIn = Zone.GRAVEYARD;
            foundCount++;
        }
        if (game.getExile().containsId(cardId, game)) {
            foundIn = Zone.EXILED;
            foundCount++;
        }

        if (foundCount > 1) {
            fail(card, checkpoint, "card is physically present in " + foundCount
                    + " zone containers simultaneously (recorded zone=" + recordedZone + ")");
        } else if (foundCount == 1 && recordedZone != foundIn) {
            fail(card, checkpoint, "recorded zone=" + recordedZone
                    + " disagrees with the physical container it's actually in=" + foundIn);
        }
    }

    private static void fail(Card card, String checkpoint, String detail) {
        String message = "ZONE INVARIANT VIOLATION [" + checkpoint + "] card=" + card.getIdName()
                + " (" + card.getId() + ") owner=" + card.getOwnerId() + ": " + detail;
        logger.fatal(message);
        throw new IllegalStateException(message);
    }
}
