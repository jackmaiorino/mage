package mage.player.ai.rl;

import mage.constants.RangeOfInfluence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Standalone, engine-free checks for observer ordering and fail-closed behavior. */
public final class RallyCp7ObservedPlayerSelfTest {

    private RallyCp7ObservedPlayerSelfTest() {
    }

    public static void main(String[] args) {
        testOrderingCardinalityAndSimulationSuppression();
        testCopySharesOneSequence();
        testObserverFailurePropagates();
        System.out.println("RallyCp7ObservedPlayerSelfTest: PASS");
    }

    private static void testOrderingCardinalityAndSimulationSuppression() {
        List<RallyCp7DecisionObserver.Decision> seen = new ArrayList<>();
        RallyCp7DecisionObserver.DispatchState state =
                new RallyCp7DecisionObserver.DispatchState(seen::add);
        state.emit(false, RallyCp7DecisionObserver.Kind.PRIORITY_ACTION,
                null, null, "action", "", Arrays.asList("pass", "bolt"),
                Collections.singletonList("bolt"), true);
        state.emit(true, RallyCp7DecisionObserver.Kind.TARGET,
                null, null, "target", "", Arrays.asList("self", "other"),
                Collections.singletonList("other"), true);
        state.emit(false, RallyCp7DecisionObserver.Kind.PRIORITY_PASS,
                null, null, "pass", "", Collections.emptyList(),
                Collections.emptyList(), true);

        check(seen.size() == 2, "simulation event must be suppressed");
        check(seen.get(0).getSequence() == 1L, "first sequence");
        check(seen.get(1).getSequence() == 2L, "suppressed event must not consume sequence");
        check(seen.get(0).getCandidates().size() == 2, "candidate cardinality");
        check(seen.get(0).getSelected().size() == 1, "selection cardinality");
        boolean immutable = false;
        try {
            seen.get(0).getSelected().add("other");
        } catch (UnsupportedOperationException expected) {
            immutable = true;
        }
        check(immutable, "event lists must be immutable");
    }

    private static void testCopySharesOneSequence() {
        List<Long> sequences = new ArrayList<>();
        RallyCp7ObservedPlayer original = new RallyCp7ObservedPlayer(
                "cp7", RangeOfInfluence.ALL, 5,
                event -> sequences.add(event.getSequence()));
        RallyCp7ObservedPlayer copy = original.copy();
        RallyCp7DecisionObserver.DispatchState originalState = original.observerStateForTest();
        RallyCp7DecisionObserver.DispatchState copyState = copy.observerStateForTest();
        check(originalState == copyState, "copy must share observer dispatch state");
        originalState.emit(false, RallyCp7DecisionObserver.Kind.CHOICE,
                null, null, null, "", Collections.singletonList("a"),
                Collections.singletonList("a"), true);
        copyState.emit(false, RallyCp7DecisionObserver.Kind.MODE,
                null, null, null, "", Collections.singletonList("b"),
                Collections.singletonList("b"), true);
        check(sequences.equals(Arrays.asList(1L, 2L)), "copies must preserve one total order");
    }

    private static void testObserverFailurePropagates() {
        RallyCp7DecisionObserver.DispatchState state =
                new RallyCp7DecisionObserver.DispatchState(event -> {
                    throw new IllegalStateException("sink failed");
                });
        boolean propagated = false;
        try {
            state.emit(false, RallyCp7DecisionObserver.Kind.CHOOSE_USE,
                    null, null, null, "", Arrays.asList(true, false),
                    Collections.singletonList(true), true);
        } catch (IllegalStateException expected) {
            propagated = "sink failed".equals(expected.getMessage());
        }
        check(propagated, "observer failure must propagate without fallback");
        check(state.getSequence() == 1L, "failed event remains sequence-accounted");
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
