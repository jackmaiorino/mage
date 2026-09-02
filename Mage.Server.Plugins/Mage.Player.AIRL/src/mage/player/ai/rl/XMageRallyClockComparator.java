package mage.player.ai.rl;

import mage.constants.PhaseStep;
import mage.game.Game;
import mage.players.Player;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/** Exact, side-effect-free comparison of a Rust decision clock to live XMage. */
final class XMageRallyClockComparator {

    static final String MARKER = "XMAGE_RALLY_KERNEL_CLOCK_MISMATCH";

    private XMageRallyClockComparator() {
    }

    static XMageRallyBridgeProtocol.ExpectedClock requireMatch(
            XMageRallyBridgeProtocol.DecisionBody decision,
            Game game,
            String consumer,
            String label) {
        return match(decision, game, false, consumer, label);
    }

    /**
     * Bind an episode's reset decision (policy step 0) to the live XMage game.
     *
     * <p>The kernel's {@code turn} is a ROUND counter that spans both seats, not
     * an XMage turn number: kernel round {@code r} covers XMage turn
     * {@code 2r - 1} (p0 active) and XMage turn {@code 2r} (p1 active). The
     * kernel's starting player is always p0, but the first surfaced decision of
     * an episode legitimately lands on p1 in Main1 of round 1 whenever p0's
     * opening turn surfaces no decision at all. A reset therefore binds kernel
     * round 1, the starting-player identity, and the normalized turn; it never
     * requires {@code active_player == p0}.
     */
    static XMageRallyBridgeProtocol.ExpectedClock requireResetBinding(
            XMageRallyBridgeProtocol.DecisionBody resetDecision,
            Game game,
            String consumer,
            String label) {
        return match(resetDecision, game, true, consumer, label);
    }

    /**
     * Pure reset binding for the same rule, for callers that already hold the
     * live XMage turn, phase, active seat and the seat XMage's starting player
     * binds to. See {@link #requireResetBinding(
     * XMageRallyBridgeProtocol.DecisionBody, Game, String, String)} for the
     * round-counter semantics.
     */
    static XMageRallyBridgeProtocol.ExpectedClock requireResetBinding(
            XMageRallyBridgeProtocol.KernelClock resetClock,
            long xmageGlobalTurn,
            PhaseStep xmagePhase,
            XMageRallyBridgeProtocol.Seat xmageActivePlayer,
            XMageRallyBridgeProtocol.Seat xmageStartingPlayer,
            String consumer,
            String label) {
        requireResetRound(resetClock, xmageStartingPlayer, consumer, label);
        return requireMatch(resetClock, xmageGlobalTurn, xmagePhase, xmageActivePlayer,
                consumer, label);
    }

    private static XMageRallyBridgeProtocol.ExpectedClock match(
            XMageRallyBridgeProtocol.DecisionBody decision,
            Game game,
            boolean reset,
            String consumer,
            String label) {
        if (decision == null) {
            throw mismatch(consumer, label, "decision is null");
        }
        XMageRallyBridgeProtocol.KernelClock kernel = decision.getKernelClock();
        if (kernel == null) {
            throw mismatch(consumer, label, "decision has no kernel_clock");
        }
        Snapshot xmage = snapshot(game, consumer, label);
        if (reset) {
            requireResetRound(kernel, xmage.startingPlayer, consumer, label);
        }
        return compare(kernel, xmage.globalTurn, xmage.phaseStep, xmage.activePlayer,
                consumer, label,
                "episode=" + decision.getEpisodeId()
                        + " policy_step=" + decision.getStep() + " ");
    }

    private static void requireResetRound(
            XMageRallyBridgeProtocol.KernelClock resetClock,
            XMageRallyBridgeProtocol.Seat xmageStartingPlayer,
            String consumer,
            String label) {
        if (resetClock == null) {
            throw mismatch(consumer, label, "reset decision has no kernel_clock");
        }
        if (resetClock.getTurn() != 1L) {
            throw mismatch(consumer, label,
                    "reset decision does not bind kernel round 1: kernel_turn="
                            + resetClock.getTurn());
        }
        if (xmageStartingPlayer != XMageRallyBridgeProtocol.Seat.P0) {
            throw mismatch(consumer, label,
                    "reset decision does not bind kernel starting player p0:"
                            + " xmage_starting_player="
                            + seatWire(xmageStartingPlayer));
        }
    }

    static XMageRallyBridgeProtocol.ExpectedClock requireMatch(
            XMageRallyBridgeProtocol.KernelClock kernel,
            long xmageGlobalTurn,
            PhaseStep xmagePhase,
            XMageRallyBridgeProtocol.Seat xmageActivePlayer,
            String consumer,
            String label) {
        if (kernel == null || xmagePhase == null || xmageActivePlayer == null) {
            throw mismatch(consumer, label,
                    "pure clock comparison received a null clock field");
        }
        XMageRallyBridgeProtocol.KernelPhaseStep mappedPhase = phase(xmagePhase);
        if (mappedPhase == null) {
            throw mismatch(consumer, label,
                    "unsupported XMage phase " + xmagePhase);
        }
        return compare(kernel, xmageGlobalTurn, mappedPhase, xmageActivePlayer,
                consumer, label, "");
    }

    private static XMageRallyBridgeProtocol.ExpectedClock compare(
            XMageRallyBridgeProtocol.KernelClock kernel,
            long xmageGlobalTurn,
            XMageRallyBridgeProtocol.KernelPhaseStep xmagePhase,
            XMageRallyBridgeProtocol.Seat xmageActivePlayer,
            String consumer,
            String label,
            String identity) {
        long normalizedKernelTurn = 2L * kernel.getTurn() - 1L
                + (kernel.getActivePlayer() == XMageRallyBridgeProtocol.Seat.P1 ? 1L : 0L);
        List<String> differences = new ArrayList<>();
        if (xmageGlobalTurn != normalizedKernelTurn) {
            differences.add("turn expected=" + normalizedKernelTurn
                    + " actual=" + xmageGlobalTurn);
        }
        if (xmagePhase != kernel.getPhaseStep()) {
            differences.add("phase expected=" + kernel.getPhaseStep().wire()
                    + " actual=" + phaseWire(xmagePhase));
        }
        if (xmageActivePlayer != kernel.getActivePlayer()) {
            differences.add("active expected=" + kernel.getActivePlayer().wire()
                    + " actual=" + seatWire(xmageActivePlayer));
        }
        if (!differences.isEmpty()) {
            throw mismatch(consumer, label,
                    identity
                            + " kernel_turn=" + kernel.getTurn()
                            + " normalized_turn=" + normalizedKernelTurn
                            + " differences=" + String.join(",", differences));
        }
        boolean parityMatches = xmageActivePlayer == XMageRallyBridgeProtocol.Seat.P0
                ? (xmageGlobalTurn & 1L) == 1L : (xmageGlobalTurn & 1L) == 0L;
        if (xmageGlobalTurn <= 0L || !parityMatches) {
            throw mismatch(consumer, label,
                    "XMage turn parity does not match active player: turn="
                            + xmageGlobalTurn + " active=" + xmageActivePlayer.wire());
        }
        long xmageKernelTurn = xmageActivePlayer == XMageRallyBridgeProtocol.Seat.P0
                ? (xmageGlobalTurn + 1L) / 2L : xmageGlobalTurn / 2L;
        return new XMageRallyBridgeProtocol.ExpectedClock(
                xmageKernelTurn, xmagePhase, xmageActivePlayer);
    }

    private static Snapshot snapshot(Game game, String consumer, String label) {
        if (game == null) {
            throw mismatch(consumer, label, "game is null");
        }
        UUID starting = game.getStartingPlayerId();
        if (starting == null || game.getPlayers() == null
                || game.getPlayers().size() != 2
                || !game.getPlayers().containsKey(starting)) {
            throw mismatch(consumer, label,
                    "XMage does not have exactly two players with a starting player");
        }
        UUID other = null;
        for (Player player : game.getPlayers().values()) {
            if (player == null) {
                throw mismatch(consumer, label, "XMage player map contains null");
            }
            if (!player.getId().equals(starting)) {
                if (other != null) {
                    throw mismatch(consumer, label, "XMage has multiple p1 players");
                }
                other = player.getId();
            }
        }
        if (other == null) {
            throw mismatch(consumer, label, "XMage p1 player is missing");
        }
        XMageRallyBridgeProtocol.Seat active = seat(
                game.getActivePlayerId(), starting, other);
        XMageRallyBridgeProtocol.Seat startingSeat = seat(starting, starting, other);
        XMageRallyBridgeProtocol.KernelPhaseStep phase = phase(game.getTurnStepType());
        if (active == null || startingSeat == null || phase == null) {
            throw mismatch(consumer, label,
                    "XMage clock has unknown active player, starting player or phase");
        }
        long globalTurn = game.getTurnNum();
        return new Snapshot(globalTurn, phase, active, startingSeat);
    }

    private static XMageRallyBridgeProtocol.Seat seat(
            UUID id, UUID starting, UUID other) {
        if (id == null) {
            return null;
        }
        if (id.equals(starting)) {
            return XMageRallyBridgeProtocol.Seat.P0;
        }
        if (id.equals(other)) {
            return XMageRallyBridgeProtocol.Seat.P1;
        }
        return null;
    }

    private static XMageRallyBridgeProtocol.KernelPhaseStep phase(PhaseStep phase) {
        if (phase == null) {
            return null;
        }
        switch (phase) {
            case UNTAP:
                return XMageRallyBridgeProtocol.KernelPhaseStep.UNTAP;
            case UPKEEP:
                return XMageRallyBridgeProtocol.KernelPhaseStep.UPKEEP;
            case DRAW:
                return XMageRallyBridgeProtocol.KernelPhaseStep.DRAW;
            case PRECOMBAT_MAIN:
                return XMageRallyBridgeProtocol.KernelPhaseStep.MAIN1;
            case BEGIN_COMBAT:
                return XMageRallyBridgeProtocol.KernelPhaseStep.BEGIN_COMBAT;
            case DECLARE_ATTACKERS:
                return XMageRallyBridgeProtocol.KernelPhaseStep.DECLARE_ATTACKERS;
            case DECLARE_BLOCKERS:
                return XMageRallyBridgeProtocol.KernelPhaseStep.DECLARE_BLOCKERS;
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
                return XMageRallyBridgeProtocol.KernelPhaseStep.COMBAT_DAMAGE;
            case END_COMBAT:
                return XMageRallyBridgeProtocol.KernelPhaseStep.END_COMBAT;
            case POSTCOMBAT_MAIN:
                return XMageRallyBridgeProtocol.KernelPhaseStep.MAIN2;
            case END_TURN:
                return XMageRallyBridgeProtocol.KernelPhaseStep.END;
            case CLEANUP:
                return XMageRallyBridgeProtocol.KernelPhaseStep.CLEANUP;
            default:
                return null;
        }
    }

    private static String phaseWire(XMageRallyBridgeProtocol.KernelPhaseStep phase) {
        return phase == null ? "unknown" : phase.wire();
    }

    private static String seatWire(XMageRallyBridgeProtocol.Seat seat) {
        return seat == null ? "unknown" : seat.wire();
    }

    private static ClockMismatch mismatch(String consumer, String label, String detail) {
        return new ClockMismatch(MARKER
                + " consumer=" + consumer
                + " label=" + label
                + " " + detail);
    }

    static final class ClockMismatch extends RuntimeException {
        private static final long serialVersionUID = 1L;

        ClockMismatch(String message) {
            super(message);
        }
    }

    private static final class Snapshot {
        private final long globalTurn;
        private final XMageRallyBridgeProtocol.KernelPhaseStep phaseStep;
        private final XMageRallyBridgeProtocol.Seat activePlayer;
        private final XMageRallyBridgeProtocol.Seat startingPlayer;

        private Snapshot(long globalTurn,
                         XMageRallyBridgeProtocol.KernelPhaseStep phaseStep,
                         XMageRallyBridgeProtocol.Seat activePlayer,
                         XMageRallyBridgeProtocol.Seat startingPlayer) {
            this.globalTurn = globalTurn;
            this.phaseStep = phaseStep;
            this.activePlayer = activePlayer;
            this.startingPlayer = startingPlayer;
        }
    }
}
