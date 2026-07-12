//! Copy-on-write snapshots.
//!
//! v1 is deliberately a full clone: `GameState` is plain-old-data (Vecs of
//! small structs, no interior mutability, no pointers), so `Clone` already
//! gives correct value semantics for snapshot/restore. What matters for
//! callers (search, PBT rollback, training-loop branching) is the API
//! boundary below; swapping the clone for a real copy-on-write or
//! delta-based representation later is an implementation change behind this
//! same boundary, not an API change.

use crate::state::GameState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot(GameState);

impl GameState {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot(self.clone())
    }

    /// Overwrites `self` with the state captured in `snapshot`.
    pub fn restore(&mut self, snapshot: &Snapshot) {
        *self = snapshot.0.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;
    use std::time::{Duration, Instant};

    fn two_card_libraries() -> (Vec<u16>, Vec<u16>) {
        (vec![1, 2, 3], vec![4, 5, 6, 7])
    }

    #[test]
    fn snapshot_restore_round_trip_is_exact() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, |c| format!("card-{c}"), 5);
        state.draw_card(PlayerId::P0);
        let snap = state.snapshot();

        state.draw_card(PlayerId::P0);
        state.draw_card(PlayerId::P1);
        state.players[0].life -= 3;
        assert_ne!(state, snap.0);

        state.restore(&snap);
        assert_eq!(state, snap.0);
        assert_eq!(state.state_hash(), snap.0.state_hash());
    }

    /// ~80 objects: two 40-card libraries, mid-game (some drawn, some on
    /// battlefield). Object count is fixed at construction (zone moves
    /// mutate in place, never allocate new objects), so this is a
    /// representative object-count shape for any point in the game.
    fn mid_game_state() -> GameState {
        let lib0: Vec<u16> = (0..40).collect();
        let lib1: Vec<u16> = (0..40).collect();
        let mut state =
            GameState::new_from_libraries(&lib0, &lib1, |c| format!("card-{c}"), 123);
        for _ in 0..12 {
            if let Some(id) = state.draw_card(PlayerId::P0) {
                if id.0 % 2 == 0 {
                    state.move_hand_to_battlefield(PlayerId::P0, id);
                }
            }
            if let Some(id) = state.draw_card(PlayerId::P1) {
                if id.0 % 2 == 0 {
                    state.move_hand_to_battlefield(PlayerId::P1, id);
                }
            }
        }
        state
    }

    #[test]
    fn snapshot_clone_cost_is_bounded() {
        let state = mid_game_state();
        assert_eq!(state.objects.len(), 80);

        for _ in 0..200 {
            std::hint::black_box(state.snapshot());
        }

        let iterations = 2000u32;
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(state.snapshot());
        }
        let per_call = start.elapsed() / iterations;
        println!("snapshot clone: {per_call:?}/call over {iterations} iterations, {} objects", state.objects.len());

        // Release-build budget from the design doc. Debug builds (no
        // inlining/LTO) are allowed a much looser bound so `cargo test`
        // without --release still passes; the real bar is `cargo test
        // --release`.
        let budget = if cfg!(debug_assertions) {
            Duration::from_micros(2000)
        } else {
            Duration::from_micros(40)
        };
        assert!(
            per_call < budget,
            "snapshot clone took {per_call:?}/call, budget {budget:?}"
        );
    }
}
