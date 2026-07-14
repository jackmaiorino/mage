package mage.player.ai.util;

import mage.game.permanent.Permanent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author noxx
 */
public class CombatInfo {

    // Determinism: Permanent/PermanentImpl never overrides equals()/hashCode(),
    // so a plain HashMap<Permanent,...> buckets by Java's default identity
    // hashCode (memory-address-derived), which differs across independent JVM
    // runs even for byte-identical game states. getCombat().entrySet() below
    // drives the declareBlocker() call order in ComputerPlayer6.declareBlockers,
    // so this must be insertion-ordered (deterministic candidate order fix,
    // same class of bug as Combat.getAttackers()/getBlockers()).
    private Map<Permanent, List<Permanent>> combat = new LinkedHashMap<>();

    public void addPair(Permanent attacker, Permanent blocker) {
        List<Permanent> blockers = combat.computeIfAbsent(attacker, k -> new ArrayList<>());
        blockers.add(blocker);
    }

    public Map<Permanent, List<Permanent>> getCombat() {
        return combat;
    }
}
