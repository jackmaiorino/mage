package mage.player.ai.rl;

import mage.abilities.Ability;
import mage.game.Game;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Synchronous observation seam for decisions made by the Rally CP7 anchor.
 *
 * <p>The observer runs on the game thread. It may inspect the supplied XMage
 * objects only for the duration of {@link #onDecision(Decision)}. Implementors
 * that need to retain data must make their own value snapshots. Throwing from
 * the observer is deliberately fail-closed and aborts the current game path.</p>
 */
@FunctionalInterface
public interface RallyCp7DecisionObserver {

    void onDecision(Decision decision);

    enum Kind {
        PRIORITY_ACTION,
        PRIORITY_PASS,
        MULLIGAN,
        PRESELECTED_TARGETS,
        TARGET,
        TARGET_AMOUNT,
        CARD_TARGET,
        CHOICE,
        CHOOSE_USE,
        MODE,
        ANNOUNCE_X,
        AMOUNT,
        TRIGGER_ORDER,
        REPLACEMENT,
        DECLARE_ATTACKERS,
        DECLARE_BLOCKERS
    }

    final class Decision {
        private final long sequence;
        private final Kind kind;
        private final Game game;
        private final Ability source;
        private final Object subject;
        private final String prompt;
        private final List<Object> candidates;
        private final List<Object> selected;
        private final boolean accepted;

        private Decision(long sequence, Kind kind, Game game, Ability source,
                         Object subject, String prompt, Collection<?> candidates,
                         Collection<?> selected, boolean accepted) {
            this.sequence = sequence;
            this.kind = Objects.requireNonNull(kind, "kind");
            this.game = game;
            this.source = source;
            this.subject = subject;
            this.prompt = prompt == null ? "" : prompt;
            this.candidates = immutableCopy(candidates);
            this.selected = immutableCopy(selected);
            this.accepted = accepted;
        }

        public long getSequence() {
            return sequence;
        }

        public Kind getKind() {
            return kind;
        }

        public Game getGame() {
            return game;
        }

        public Ability getSource() {
            return source;
        }

        public Object getSubject() {
            return subject;
        }

        public String getPrompt() {
            return prompt;
        }

        public List<Object> getCandidates() {
            return candidates;
        }

        public List<Object> getSelected() {
            return selected;
        }

        public boolean isAccepted() {
            return accepted;
        }

        private static List<Object> immutableCopy(Collection<?> values) {
            if (values == null || values.isEmpty()) {
                return Collections.emptyList();
            }
            return Collections.unmodifiableList(new ArrayList<Object>(values));
        }
    }

    /** Shared by live copies so all observed decisions have one total order. */
    final class DispatchState {
        private final RallyCp7DecisionObserver observer;
        private final AtomicLong sequence = new AtomicLong();

        public DispatchState(RallyCp7DecisionObserver observer) {
            this.observer = Objects.requireNonNull(observer, "observer");
        }

        public void emit(boolean simulation, Kind kind, Game game, Ability source,
                         Object subject, String prompt, Collection<?> candidates,
                         Collection<?> selected, boolean accepted) {
            if (simulation) {
                return;
            }
            long next = sequence.incrementAndGet();
            observer.onDecision(new Decision(next, kind, game, source, subject,
                    prompt, candidates, selected, accepted));
        }

        long getSequence() {
            return sequence.get();
        }
    }
}
