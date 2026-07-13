//! mtg-kernel: fast, resumable, copy-on-write game core for a fixed Pauper pool.
//!
//! Scope: exactly the 132-card 8-deck pool (see the rules-surface matrix). The
//! Java XMage engine remains the reference implementation and claim surface;
//! this kernel is a training/search accelerator whose behavior is validated by
//! golden-trace replay equivalence and branch differential testing against the
//! reference. Unsupported mechanics FAIL CLOSED (a deck that needs them cannot
//! be simulated here and must run on the reference engine).
//!
//! Architectural invariants (Sol #84):
//! - Deterministic transitions; no unordered-map iteration anywhere.
//! - Stable arena ids for all game objects; canonical identity across snapshots.
//! - O(1) snapshot/branch via copy-on-write or reversible deltas.
//! - Batched `reset/step/legal_actions/snapshot/restore` as the public API.

pub mod card_def;
pub mod effect;
pub mod engine;
pub mod event;
pub mod ids;
pub mod mana;
pub mod snapshot;
pub mod state;
pub mod surface;
pub mod surface_v2;
pub mod trace;
pub mod trigger;

pub const KERNEL_VERSION: &str = "0.0.1-spike";
