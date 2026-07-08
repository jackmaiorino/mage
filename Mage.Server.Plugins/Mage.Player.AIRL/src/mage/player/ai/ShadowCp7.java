package mage.player.ai;

import mage.abilities.Ability;
import mage.game.Game;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Seat-bound one-shot CP7 planner for shadow teacher labels (DAgger-style):
 * answers "what would CP7 do from this exact state for this seat" WITHOUT
 * executing anything.
 *
 * calculateActions searches on its own internal simulation copy
 * (createSimulation -> game.createSimulationForAI()) and the recursive search
 * runs on the shared AI-SIM-MAD pool, so the live game is never mutated and the
 * calling thread needs no renaming -- as long as act()/pass() are never called
 * on this instance. This object is NOT a seated player; never add it to a game.
 */
final class ShadowCp7 extends ComputerPlayer7 {

    ShadowCp7(UUID seatId, int skill) {
        super(seatId, skill);
    }

    /**
     * One-shot plan from the given live state. Returns the planned ability
     * sequence (empty = CP7 would pass). Fresh instances should be used per
     * query; the clears below are belt-and-braces against plan reuse.
     */
    List<Ability> planOnce(Game game) {
        actions.clear();
        root = null;
        calculateActions(game);
        return new ArrayList<>(actions);
    }
}
