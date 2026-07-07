package mage.player.ai.rl;

import mage.constants.WatcherScope;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.watchers.Watcher;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.UUID;

/**
 * Records a rolling buffer of the last-K PUBLIC game events for the temporal /
 * event-history falsifier (Step 1). Perspective-agnostic: stores the actor's
 * player name; the RL agent maps name -> me/opp when it reads the buffer at a
 * decision point. Only active when RL_CORPUS_DUMP is truthy, and only records on
 * the real game (never simulated clones), so normal training is untouched and
 * deep CP7 opponent search does not pollute the real buffer.
 */
public class PublicEventHistoryWatcher extends Watcher {

    private static final boolean CORPUS_ENABLED =
            "1".equals(System.getenv().getOrDefault("RL_CORPUS_DUMP", "0"))
                    || "true".equalsIgnoreCase(System.getenv().getOrDefault("RL_CORPUS_DUMP", "0"));

    private static final int MAX_EVENTS;
    static {
        int k = 96;
        try { k = Integer.parseInt(System.getenv().getOrDefault("RL_CORPUS_EVENT_WINDOW", "96")); } catch (Exception ignored) {}
        MAX_EVENTS = Math.max(8, k);
    }

    /** Compact, serializable, deep-copyable event record. */
    public static final class EventRecord implements Serializable {
        private static final long serialVersionUID = 1L;
        public final int turn;
        public final String actor;   // player name
        public final String type;    // short event code
        public final String card;    // public source/card name ("" if unknown)
        public final String extra;   // target name / zone / defender ("" if n/a)
        public final int amount;

        public EventRecord(int turn, String actor, String type, String card, String extra, int amount) {
            this.turn = turn;
            this.actor = actor == null ? "" : actor;
            this.type = type == null ? "" : type;
            this.card = card == null ? "" : card;
            this.extra = extra == null ? "" : extra;
            this.amount = amount;
        }
    }

    private ArrayDeque<EventRecord> events = new ArrayDeque<>();

    public PublicEventHistoryWatcher() {
        super(WatcherScope.GAME);
    }

    /**
     * Override the base reflective deep-copy: it routes each field through
     * CardUtil.deepCopyObject, which does not handle ArrayDeque and throws during
     * CP7's game-clone simulations. EventRecord is immutable, so a fresh deque that
     * shares the records is a correct and cheap copy.
     */
    @Override
    @SuppressWarnings("unchecked")
    public <T extends Watcher> T copy() {
        PublicEventHistoryWatcher w = new PublicEventHistoryWatcher();
        w.setControllerId(this.getControllerId());
        w.setSourceId(this.getSourceId());
        w.events = new ArrayDeque<>(this.events);
        return (T) w;
    }

    @Override
    public void watch(GameEvent event, Game game) {
        if (!CORPUS_ENABLED || event == null || game == null) {
            return;
        }
        try {
            if (game.isSimulation()) {
                return; // never record on cloned/simulated games
            }
            String type = classify(event.getType());
            if (type == null) {
                return; // not an event we track
            }
            int turn = game.getTurnNum();
            String actor = playerName(game, event.getPlayerId());
            String card = objectName(game, event.getSourceId());
            String extra = "";
            int amount = event.getAmount();
            switch (event.getType()) {
                case ZONE_CHANGE:
                    extra = zoneChangeExtra(event);
                    if (card.isEmpty()) card = objectName(game, event.getTargetId());
                    break;
                case ATTACKER_DECLARED:
                case BLOCKER_DECLARED:
                    if (card.isEmpty()) card = objectName(game, event.getSourceId());
                    extra = playerName(game, event.getTargetId()); // defender / creature-owner target
                    break;
                case DAMAGED_PLAYER:
                    extra = playerName(game, event.getTargetId());
                    break;
                case COUNTERED:
                    if (card.isEmpty()) card = objectName(game, event.getTargetId());
                    break;
                default:
                    break;
            }
            if (actor.isEmpty()) {
                actor = playerName(game, event.getTargetId());
            }
            events.addLast(new EventRecord(turn, actor, type, card, extra, amount));
            while (events.size() > MAX_EVENTS) {
                events.removeFirst();
            }
        } catch (Exception ignored) {
            // a corpus dumper must never affect game correctness
        }
    }

    /** Snapshot of recent events (oldest -> newest), for the agent to read at a decision. */
    public List<EventRecord> getRecent() {
        return new ArrayList<>(events);
    }

    private static String classify(GameEvent.EventType t) {
        if (t == null) return null;
        switch (t) {
            case CAST_SPELL: return "cast";
            case SPELL_CAST: return "spellcast";
            case LAND_PLAYED: return "land";
            case ATTACKER_DECLARED: return "attack";
            case BLOCKER_DECLARED: return "block";
            case DAMAGED_PLAYER: return "dmg";
            case COUNTERED: return "counter";
            case ZONE_CHANGE: return "zone";
            default: return null;
        }
    }

    private static String zoneChangeExtra(GameEvent event) {
        try {
            Zone from = ((mage.game.events.ZoneChangeEvent) event).getFromZone();
            Zone to = ((mage.game.events.ZoneChangeEvent) event).getToZone();
            return (from == null ? "?" : from.name()) + ">" + (to == null ? "?" : to.name());
        } catch (Exception e) {
            return "";
        }
    }

    private static String playerName(Game game, UUID id) {
        if (id == null) return "";
        try {
            mage.players.Player p = game.getPlayer(id);
            return p == null ? "" : p.getName();
        } catch (Exception e) {
            return "";
        }
    }

    private static String objectName(Game game, UUID id) {
        if (id == null) return "";
        try {
            mage.MageObject o = game.getObject(id);
            if (o != null && o.getName() != null && !o.getName().isEmpty()) {
                return o.getName();
            }
            mage.cards.Card c = game.getCard(id);
            return c == null ? "" : c.getName();
        } catch (Exception e) {
            return "";
        }
    }
}
