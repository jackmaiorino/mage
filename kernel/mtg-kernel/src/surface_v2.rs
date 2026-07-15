//! `HarnessSurfaceV2`: the H2 decision-visibility surface, driven against
//! corpus v4 (`local-training/kernel_oracle/burn_mirror_v4_run1/`, LOCKED).
//!
//! ## This is an EXPLICIT CONTRACT DIFF from H1 (Sol #88)
//!
//! `HarnessSurfaceV1` (`surface.rs`) is frozen as of increment 9 and stays
//! that way: this module adds a new, separate type alongside it rather than
//! editing it in place -- not one line of `surface.rs` changes for H2. The
//! diff, precisely:
//!
//! 1. **All four H1 suppression rules are preserved, unchanged.** Silent
//!    step-gated priority passes, forced empty-candidate auto-resolutions,
//!    the `DeclareAttackers`/`DeclareBlockers` one-action-per-round combat
//!    throttle, and the same-caster force-pass after a stack-using
//!    cast/activation (`StackTopIsCastersOwn`) all fire under exactly the
//!    same conditions here as in `HarnessSurfaceV1::next_decision` -- see
//!    that function's doc and `surface.rs`'s module doc (predicate points
//!    1-4) for the full citation-backed derivation, which this module does
//!    not repeat. Nothing in `ComputerPlayerRL.priorityPlay`/`act`/
//!    `selectAttackers`/`selectBlockers` changed between the v3 and v4
//!    corpus-generating commits (see `H2_JAVA_ORACLE_COMMIT`'s doc for the
//!    exact diff checked), so there is no *predicate* content to change --
//!    this type exists to be independently versioned, not because the
//!    suppression logic itself needed to move.
//!
//!    The state machine below (`HarnessSurfaceV2::next_decision`/`apply`
//!    and its private blockers-reshape bookkeeping) is therefore a
//!    deliberate, byte-for-byte duplicate of `HarnessSurfaceV1`'s, under a
//!    new name -- not a refactor, not a shared-generics abstraction. Data
//!    types that carry no predicate logic of their own
//!    (`SuppressionReason`, `Suppression`, `SurfaceDecision`,
//!    `SurfaceAction`, `harness_never_offers_priority`) are reused directly
//!    from `surface.rs` rather than re-declared: importing a plain enum/
//!    struct is not "editing `HarnessSurfaceV1`", and re-typing them here
//!    would only create a second, driftable copy of data that has no H1-
//!    vs-H2 distinction to express. If a future increment needs `V2` to
//!    diverge from `V1`'s predicate, the fix is to stop sharing those types
//!    then -- not preemptively today.
//!
//! 2. **The Java `chooseTarget` same-name-dedup shortcut is gone, with no
//!    replacement.** That shortcut (`java_reference_target_shortcut` in
//!    `examples/replay_burn.rs`) was never part of `HarnessSurfaceV1` to
//!    begin with -- it lived entirely in the v3 comparator, because it was
//!    comparator-local special-casing of a Java reference *bug*
//!    (`ComputerPlayerRL.chooseTarget`'s `allSameName` null-name mishandling,
//!    see that function's doc for the full root-cause). Corpus v4 was
//!    generated *after* that bug was fixed on the Java side (`ea2620fc938`,
//!    "Sol #87, remove chooseTarget same-name shortcut" -- see
//!    `H2_JAVA_ORACLE_COMMIT`'s doc), so the shortcut it used to emulate no
//!    longer exists in the reference at all: every `AnyTarget` window in v4
//!    is a real, logged `SELECT_TARGETS` decision (this is the entire
//!    reason v4 has roughly double v3's `SELECT_TARGETS` count -- 1107 vs
//!    555, confirmed by direct corpus count). `examples/replay_burn_v2.rs`
//!    (the H2 comparator) has no function playing that role and must not
//!    grow one: an unexplained `ChooseTargets` window against v4 is a real
//!    divergence to report, never a silent guess.
//!
//! 3. **Provenance consts**, below: `H2_PREDICATE_VERSION`,
//!    `H2_JAVA_ORACLE_COMMIT`, and `verify_corpus_provenance` (the runtime
//!    "does this corpus's own recorded generating commit match what this
//!    predicate was verified against" gate the H2 contract requires,
//!    "fail loudly if not"). The card-DB hash gate reuses
//!    `card_def::KERNEL_CARDDB_HASH`, already generated at build time; no
//!    new const needed for it.
//!
//! 4. **No fuzzy H1 fallback anywhere in H2 replay.** `HarnessSurfaceV1` is
//!    not imported or invoked anywhere in this module or in
//!    `examples/replay_burn_v2.rs`. If a v4 decision doesn't match what the
//!    trace expects, that is reported as a divergence with a specific
//!    reason string -- never silently rescued by falling back to a V1-era
//!    behavior (the target shortcut above being the one such rescue that
//!    existed in the v3 comparator).

use crate::engine::{self, Action, Decision, OptionalCostChoice};
use crate::ids::{ObjectId, PlayerId};
use crate::state::{GameState, Step};
pub use crate::surface::{
    harness_never_offers_priority, Suppression, SuppressionReason, SurfaceAction, SurfaceDecision,
};
use serde::{Deserialize, Serialize};

/// Predicate version. `HarnessSurfaceV1` is version 1 (`H1_PREDICATE_VERSION`,
/// `surface.rs`); this is the second, versioned independently per that
/// module's own doc ("never an edit to `HarnessSurfaceV1` itself... a new
/// type living alongside this one").
pub const H2_PREDICATE_VERSION: u32 = 2;

/// `ComputerPlayerRL.java`'s commit hash this predicate's citations (shared
/// with `HarnessSurfaceV1`'s -- see this module's doc, point 1) were last
/// verified against, and the exact commit corpus v4 was generated at
/// (`local-training/kernel_oracle/burn_mirror_v4_run1/manifest.json`'s own
/// `java_oracle_commit` field: `6de2528fada1c740ceb5fdda0f273bdb9ee28b79`).
/// Pinned to that value, matching `H1_JAVA_ORACLE_COMMIT`'s own convention
/// of naming the commit the predicate was verified against, not a live
/// `git log` re-run at every build -- `verify_corpus_provenance` is the
/// runtime check that a corpus being replayed actually claims this same
/// commit.
///
/// As of this increment (2026-07-13), `git log --format=%H -1 -- \
/// Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerRL.java`
/// (run from the repo root) actually returns
/// `5f5305503ae2159bdad02502967ae20ad7dff847` -- one commit ahead of this
/// pin. That commit ("RL: force-pass opportunity telemetry (Sol #88, H3
/// groundwork)") is additive-only: it inserts a call to a new
/// `maybeLogForcePassOpportunity` helper immediately before the
/// `pass(game)` predicate point 4 cites, gated behind
/// `RL_FORCEPASS_TELEMETRY` (default `false`), and logs via `gameLogger.log`
/// (a plain human-readable line, not `logReplayDecision`) -- it emits no
/// `REPLAY_DECISION_JSON` record and does not touch `priorityPlay`,
/// `chooseTarget`, `genericChoose`, `selectAttackers`, or `selectBlockers`.
/// It is therefore inert for every citation this predicate (H1's four
/// points, unchanged per this module's doc) depends on, and this pin
/// remains accurate. `verify_corpus_provenance` still checks the *corpus's*
/// claimed commit against this pin, not live `HEAD` -- see that function's
/// doc for why.
///
/// **Re-verified and re-pinned for the Rally promotion increment** (corpora
/// `rally_mirror_v1`/`rally_vs_burn_v1`,
/// `local-training/kernel_oracle/rally_mirror_v1/manifest.json`'s
/// `java_oracle_commit`: `9e92e240193170626ea4530ab048873889911b68`).
/// `git log --oneline 6de2528fada1..9e92e240193 -- .../ComputerPlayerRL.java`
/// shows 6 commits (the one above, `5f5305503ae`, plus 5 more); every one
/// audited individually and confirmed inert for this predicate's citations:
/// - `3e966e3c4f5` ("fix telemetry compile"): null-guards the same
///   `RL_FORCEPASS_TELEMETRY`-gated diagnostic line `5f5305503ae` added,
///   before formatting it. No behavior change, nothing new touched.
/// - `e3e29e92a48` ("determinism closure batch, Sol #97"): two fixes, both
///   *determinism-improving, not decision-changing*: an isolation-seed hash
///   switches from `ability.getSourceId().hashCode()` (a per-run-random
///   UUID) to `ability.getRule().hashCode()` (content-stable), and
///   `SEARCH_OP`'s rollout seed switches from `System.identityHashCode(...)`
///   to `RL_BASE_SEED`-derived *only when `RL_BASE_SEED` is set* (this
///   increment's own corpora set it, `=5151`) -- neither touches
///   `chooseTarget`/`genericChoose`/`priorityPlay`/`selectAttackers`/
///   `selectBlockers`, or what candidates are legal/offered, only what a
///   downstream *rollout* RNG (unrelated to the 4 predicate points, which
///   are all pure priority/combat-window suppressions) is seeded with.
/// - `05845190d9e` ("canonical stack-position ids, Sol #96"): changes how a
///   stack-object candidate/target *id is stringified* for logging
///   (`UUID.toString()` -> `canonicalStackObjectId`, a reproducible
///   "stack#N" form) -- a representation change for reproducibility, not a
///   legality or choice-content change; no card in either Rally corpus ever
///   targets the stack (no `TargetSpell`/counterspell-shaped card in the
///   18-card pool), so this path is not even exercised by these corpora.
/// - `d25a540e917` ("per-game record_id, v5 schema"): adds a new,
///   purely-additive `record_id` field to `REPLAY_DECISION_JSON`/
///   checkpoint-manifest rows (this is the commit that *committed* what
///   `burn_mirror_v5`'s own manifest had documented as an uncommitted
///   prerequisite diff) -- an identity/join key for tooling, consumed by
///   this increment's `identity_check_v6.py` actor/player invariant, not
///   read by this driver or the H1/H2 predicate at all.
/// - `88a77c625ab` ("official 24-pt gate hardening"): adds
///   `REAL_INFERENCE_CALLS`, a pure `AtomicLong` counter incremented at the
///   one real model-inference call site, for an unrelated Stage-C
///   checkpoint-reentry invariant. Read by no code this predicate or driver
///   depends on.
///
/// The prior session's own uncommitted prerequisite diff on top of
/// `9e92e240193` (see `rally_mirror_v1/manifest.json`'s `working_tree_diff`:
/// `ComputerPlayerRL.stableBattlefieldPosition`, a same-createOrder-batch
/// tie-break fix in `sortTargetsForStableChoice`) is the same "determinism-
/// improving, not decision-content-changing" category as `e3e29e92a48`
/// above: it changes *which specific same-name candidate a tied index maps
/// to* in a run-stable way, never what candidates are offered or how many.
/// That diff was committed as part of `9e92e240193` itself (see its own
/// commit body); nothing further to audit for it.
///
/// **Re-pinned again for ReferenceRules v2** (`burn_mirror_v6`,
/// `local-training/kernel_oracle/burn_mirror_v6/manifest.json`'s
/// `java_oracle_commit`: `0723fc0c2be922af47b0ef0539f28114cc23b998`, "Mage:
/// zone-reconciliation fix + fail-fast invariants (Sol #106 / ReferenceRules
/// v2)"). Unlike every prior re-pin above, this commit touches NONE of
/// `ComputerPlayerRL.java`'s `chooseTarget`/`genericChoose`/`priorityPlay`/
/// `selectAttackers`/`selectBlockers` (confirmed: `git log --format=%H -1 -1
/// -- .../ComputerPlayerRL.java` from `9e92e240193` through this commit is
/// unchanged at `9e92e240193` itself -- ComputerPlayerRL.java was not
/// touched), so it would not even surface via this doc's usual audit method
/// of walking that file's own history. It instead changes three core-engine
/// files this predicate does NOT cite directly but whose correctness this
/// driver silently assumes: `GameImpl.java` (a new `init()` reconciliation
/// step that re-zones every physically-present, still-undrawn library card
/// to `Zone.LIBRARY`, order-independent of `addPlayer`/`loadCards` call
/// order), `ZonesHandler.java` (wires a new fail-fast check at the
/// zone-change commit point), and the new `ZoneInvariants.java` (the
/// fail-fast utility itself, diagnostic-only -- see its own doc). Root bug
/// fixed: `GameState.addCard` zones every loaded card `OUTSIDE`, and nothing
/// previously re-zoned an undrawn library card to `LIBRARY` before the RL
/// harness's own (reversed, `addPlayer` then `loadCards`) call order left it
/// stuck there; any zone-routed effect moving such a card out of the library
/// (impulse-draw exile, mill, search-into-hand, land fetch) then hit
/// `CardImpl.removeFromZone`'s `OUTSIDE` branch, which no-ops the physical
/// container removal -- duplicating the card. Plain draws
/// (`Library.drawFromTop`/`drawFromBottom`) never consult the recorded zone
/// at all (direct `Deque.pollFirst()`/`pollLast()`, zone set to `HAND`
/// explicitly afterward), so they were always immune regardless of this bug.
///
/// **Whether this re-pin is inert for THIS predicate is corpus-dependent, not
/// universally true like the re-pins above** -- this is the one commit in
/// this doc's history that can change decision content, for any deck whose
/// pool contains a card that routes a library card out via `ZonesHandler`
/// (see `local-training/kernel_oracle/reference_rules_v2_addendum.md`'s
/// pool-wide static audit table). `burn_mirror_v6`'s own H2 replay-gate
/// result (39/40 replayed clean, 1 pre-existing classified divergence,
/// identical scoreboard shape to `burn_mirror_v5`) confirms it is inert for
/// **Mono-Red Burn specifically**: that deck's pool has zero cards on the
/// audit table's "Affected" list (no impulse-draw/mill/search-into-hand/land-
/// fetch effect anywhere in `Deck - Mono-Red Burn.dek`), so the reconciled
/// zone value is never queried before a card is drawn and this fix is a
/// pure no-op for Burn's own decision stream. It is explicitly NOT expected
/// to be inert for Mono Red Rally (Reckless Impulse, Experimental
/// Synthesizer, Clockwork Percussionist all on the "Affected" list) --
/// re-verify this predicate's citations again before trusting an H2 replay
/// against any `rally_mirror_v2`/`rally_vs_burn_v2`-class corpus minted at
/// this commit or later; do not assume this re-pin's Burn-clean result
/// transfers.
pub const H2_JAVA_ORACLE_COMMIT: &str = "0723fc0c2be922af47b0ef0539f28114cc23b998";

/// Checks a corpus's own `manifest.json`-recorded `java_oracle_commit`
/// against `H2_JAVA_ORACLE_COMMIT`, "failing loudly" (a descriptive `Err`,
/// never a silent pass-through) on any mismatch -- the H2 contract's
/// provenance gate.
///
/// Deliberately compares against the *pinned* constant (the commit this
/// predicate was verified against and the commit v4 itself was generated
/// at), not a freshly re-run `git log` at replay time: this driver's job is
/// to catch "someone pointed H2 at a corpus generated by a *different*,
/// unverified Java commit" (a real bug class -- H1's own increment-report
/// history includes exactly this failure mode, root-caused in the v4
/// manifest's own `identity_check` section as a stale-jar reactor bug), not
/// to fail every single build the moment `ComputerPlayerRL.java` gets any
/// unrelated commit (see `H2_JAVA_ORACLE_COMMIT`'s doc for today's example
/// of exactly such a commit). Re-deriving live `HEAD` here would make this
/// assertion permanently red for reasons unrelated to corpus validity,
/// which is worse than not checking at all.
pub fn verify_corpus_provenance(manifest_java_oracle_commit: &str) -> Result<(), String> {
    if manifest_java_oracle_commit != H2_JAVA_ORACLE_COMMIT {
        return Err(format!(
            "H2 corpus provenance mismatch: manifest.json java_oracle_commit={manifest_java_oracle_commit:?} \
             but HarnessSurfaceV2 was verified against H2_JAVA_ORACLE_COMMIT={H2_JAVA_ORACLE_COMMIT:?}. \
             This corpus was not generated by the Java commit this predicate's citations were checked \
             against -- do not trust replay results against it until re-verified."
        ));
    }
    Ok(())
}

#[derive(Default)]
struct BlockersReshape {
    remaining: std::collections::VecDeque<(ObjectId, Vec<ObjectId>)>,
    accumulated: Vec<(ObjectId, ObjectId)>,
    current_attacker: Option<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HarnessSurfacePublicContextV2 {
    pub blockers: Option<BlockersReshapePublicV2>,
    pub discard: Option<DiscardReshapePublicV2>,
    pub optional_cost: Option<OptionalCostReshapePublicV2>,
    pub combat_priority_spent: [bool; 2],
    pub combat_priority_round_seen: Option<u64>,
    pub combat_priority_stack_len_seen: usize,
    pub combat_priority_mana_count_seen: u64,
    pub combat_round_opening_mana_count: u64,
    pub round_opening_stack_len: usize,
    pub stack_len_round_seen: Option<u64>,
    pub last_seen_stack_len: Option<usize>,
    pub mana_count_at_last_stack_change: u64,
    pub madness_cast_reprompt_exemption: Option<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockersReshapePublicV2 {
    pub current_attacker: Option<ObjectId>,
    pub accumulated: Vec<(ObjectId, ObjectId)>,
    pub remaining: Vec<(ObjectId, Vec<ObjectId>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiscardReshapePublicV2 {
    pub player: PlayerId,
    pub chosen: Vec<ObjectId>,
    pub remaining_choices: Vec<ObjectId>,
    pub remaining_needed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionalCostStagePublicV2 {
    Use,
    Which,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OptionalCostReshapePublicV2 {
    pub player: PlayerId,
    pub discard_payable: bool,
    pub sacrifice_payable: bool,
    pub stage: OptionalCostStagePublicV2,
}

/// H2's decomposition of the engine's one-shot `Decision::Discard { count,
/// choices }` (pick `count` cards from `choices` in a single answer) into
/// `count` sequential single-card picks -- Java's real shape for *every*
/// discard (a cost's, an effect's, cleanup's), confirmed against the live
/// cross-engine oracle: even a 1-card discard surfaces as a real
/// `SELECT_TARGETS` window there, and a 2-card discard (Faithless Looting)
/// is two, with the second's candidate pool missing exactly the first's
/// pick -- never one `SELECT_CARD`-shaped batch. `next_decision` re-presents
/// the *same* `Decision::Discard` variant for each pick (so the shape stays
/// a plain, engine-native type -- no new `Decision` variant needed), just
/// with `count` pinned to `1` and `choices` narrowed to whatever's left;
/// `apply` accumulates real answers here and only calls the engine's own
/// (still-batched) `Action::Discard` once every card is chosen. The
/// engine's own aggregate "no real choice left" pre-check
/// (`drain_pending_discard_or_decide`'s `choices.len() <= count`) still runs
/// before a `Decision::Discard` is ever raised at all, which guarantees
/// `remaining_choices.len() > remaining_needed` at every step below (the
/// slack between them is invariant, since each pick removes exactly one
/// from both) -- so, unlike the sibling `ChooseCostTargets`/`SacrificeLands`
/// reshape, no per-pick "down to 1 candidate, auto-finish" shortcut is ever
/// reachable here and none is implemented.
struct DiscardReshape {
    player: PlayerId,
    remaining_choices: Vec<ObjectId>,
    chosen: Vec<ObjectId>,
    remaining_needed: u32,
}

/// H2's decomposition of the engine's one-shot, 3-way `Decision::
/// ChooseOptionalCost` (Decline / Discard / SacrificeLand, all in a single
/// answer) into Java's real two-stage shape (Highway Robbery's own
/// `DoIfCostPaid`+`OrCost`, read in full this session):
///
/// 1. **`Use`**: a binary "pay this cost at all?" gate
///    (`DoIfCostPaid.apply`'s own `player.chooseUse(...)` -- plain `Yes`/
///    `No`, confirmed against the live oracle: every Highway-Robbery-shaped
///    window it captured this round starts with exactly this gate).
/// 2. **`Which`**: only reached when *both* `discard_payable` and
///    `sacrifice_payable` are true (`OrCost.pay`'s own `usable.size() == 2`
///    branch) -- a second binary pick between the two payable sub-costs'
///    own texts (`"Discard a card"` / `"Sacrifice a land"`). When only one
///    sub-cost is payable, `OrCost.pay`'s `usable.size() == 1` branch
///    auto-selects it with **no** decision at all -- matched here by
///    resolving straight through without ever presenting `Which`.
///
/// Represented by re-presenting the same `Decision::ChooseOptionalCost`
/// variant (no new `Decision` needed) with a sentinel field combination
/// `next_decision` never sees from the engine itself (which always sets at
/// least one of `discard_payable`/`sacrifice_payable`, per that decision's
/// own doc): `(false, false)` for the `Use` gate, `(true, true)` for
/// `Which`. `apply` still accepts the engine's original one-shot
/// `Action::ChooseOptionalCost(OptionalCostChoice)` too, as a direct
/// "resolve the whole reshape right now" bypass -- every pre-existing H2
/// caller (`bench_kernel.rs`'s random policy, `branch_diff.rs`'s fixed
/// continuation/force helpers, `replay_burn_v2.rs`'s own best-effort
/// look-ahead guess, none of which have real per-stage ground truth to
/// begin with) keeps constructing that single answer unchanged.
#[derive(Clone, Copy)]
struct OptionalCostReshape {
    player: crate::ids::PlayerId,
    discard_payable: bool,
    sacrifice_payable: bool,
    stage: OptionalCostStage,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OptionalCostStage {
    Use,
    Which,
}

/// See the module doc. Structurally identical to `HarnessSurfaceV1` (same
/// four suppression rules, same blockers-reshape bookkeeping) but a
/// separate type, per the H2 contract.
#[derive(Default)]
pub struct HarnessSurfaceV2 {
    suppressions: Vec<Suppression>,
    blockers: Option<BlockersReshape>,
    combat_priority_spent: [bool; 2],
    combat_priority_round_seen: Option<u64>,
    /// `state.stack.len()` as of the last time the `DeclareAttackers`/
    /// `DeclareBlockers` one-action-per-round throttle was evaluated --
    /// deliberately *not* reset only at `priority_round` boundaries (see
    /// `combat_priority_round_seen`), because a fresh cast/activation
    /// mid-round does not bump `priority_round` (`finalize_cast`/
    /// `finalize_activation` intentionally don't -- see their own doc) but
    /// *does* call Java's `PlayerImpl.activateAbility`, whose last lines
    /// unconditionally call `game.getPlayers().resetPassed()` on *any*
    /// successful action (cast, activation, land, mana ability alike) --
    /// clearing *every* player's passed flag, not just the actor's. Fixed
    /// (increment 13) against `game_20260713_002152_0007.txt` decision 24:
    /// SelfPlay flashback-casts Lava Dart targeting themselves while both
    /// players had already spent this round's one action (`combat_priority_
    /// spent == [true, true]` going stale); the kernel's `next_decision`
    /// then force-passed *both* players back-to-back off that stale flag,
    /// resolving Lava Dart a full priority round before the reference (whose
    /// `PlayerRL1` gets one genuine extra ask here, per `resetPassed`'s
    /// effect on *all* players) ever does -- a one-point life-total
    /// desync from that point on. See `next_decision`'s combat branch for
    /// how this field is used to re-arm `combat_priority_spent` for both
    /// players when the stack size changes mid-round, while still routing
    /// the *acting* player's own reopened window through the pre-existing
    /// `stack_top_is_fresh_own_item` check (so their own immediate reprompt
    /// stays silently suppressed, unchanged from before this fix).
    combat_priority_stack_len_seen: usize,
    /// `state.engine.mana_ability_activations` as of the last time the
    /// combat throttle was (re-)armed -- the analogous re-arm signal to
    /// `combat_priority_stack_len_seen`, for the one action type that
    /// resets `priority_passes` mid-round without ever touching
    /// `state.stack` (a mana ability, 605.3b). See `EngineState::
    /// mana_ability_activations`'s doc for the full root-cause
    /// (`game_20260713_002148_0003.txt` decision 34,
    /// `game_20260713_002202_0024.txt` decision 179): without this,
    /// a mid-round mana ability activation left both players' `combat_
    /// priority_spent` flags stale-true (the stack-length-only re-arm
    /// below never fires for it), silently force-passing the rest of the
    /// combat phase's priority windows all the way into `Main2`.
    ///
    /// This alone is *not* sufficient, though: unlike the stack-length
    /// re-arm (safe to reset *both* players' flags unconditionally, because
    /// the acting caster's own immediate reprompt is separately, durably
    /// re-suppressed every time by `stack_top_is_fresh_own_item` for as
    /// long as their item sits unresolved on top of the stack), a mana
    /// ability activation leaves nothing on `state.stack` to re-check --
    /// so blindly resetting *both* flags here would also incorrectly
    /// un-suppress the activator's own next ask. See
    /// `combat_round_opening_mana_count`'s doc for the durable
    /// self-suppression counterpart this pairs with.
    combat_priority_mana_count_seen: u64,
    /// `state.engine.mana_ability_activations` as of the start of the
    /// current `DeclareAttackers`/`DeclareBlockers` round (captured once,
    /// alongside `combat_priority_round_seen`, and never updated again
    /// until the next round -- the "did this grow *since the round
    /// opened*" baseline, mirroring `round_opening_stack_len`'s role for
    /// `stack_top_is_fresh_own_item`). Paired with `state.engine.
    /// last_mana_ability_activator` in `next_decision`'s combat branch to
    /// durably re-suppress the activator's own immediate reprompt after
    /// their own mana ability ("mana_ability_is_fresh_own_action", the
    /// mana-ability analogue of `stack_top_is_fresh_own_item`) -- without
    /// this, an over-broad fix that resets *both* players' `combat_
    /// priority_spent` flags on every mana ability (matching
    /// `combat_priority_mana_count_seen`'s re-arm) would also let the
    /// activator themselves get re-asked repeatedly (e.g. tapping several
    /// lands in a row), stalling the kernel *behind* the reference instead
    /// of racing ahead of it -- confirmed by an intermediate, over-broad
    /// version of this fix regressing the corpus from 21/40 to 6/40
    /// complete traces before this field was added.
    combat_round_opening_mana_count: u64,
    /// Stack length when this surface first observes the current
    /// `priority_round`. The lazy timing is semantic: resolution-created
    /// triggers are already present and therefore belong to the baseline
    /// (Java resets passed flags after resolution), while casts and their
    /// cast-time triggers grow the stack later in the same round and remain
    /// covered by the originating `act()` call's trailing force-pass.
    round_opening_stack_len: usize,
    stack_len_round_seen: Option<u64>,
    /// `state.stack.len()` as of the last time the plain (non-combat)
    /// `stack_top_is_fresh_own_item` suppression was evaluated -- deliberately
    /// keyed off the stack's own length changing, *not* off
    /// `priority_round` the way `round_opening_stack_len` itself is (see
    /// that field's doc for why a fresh cast mid-round doesn't bump
    /// `priority_round`): a `priority_round`-keyed baseline stays fixed for
    /// the *entire* round, so it can't tell "the opponent responded before
    /// my item was even cast, earlier this same round" apart from "the
    /// opponent responded after my item was cast" -- and only the second
    /// case is a real reason to un-suppress. Paired with
    /// `mana_count_at_last_stack_change`, below.
    last_seen_stack_len: Option<usize>,
    /// `state.engine.mana_ability_activations` as of the moment
    /// `last_seen_stack_len` was last (re-)captured, i.e. as of the last
    /// time `state.stack.len()` actually changed -- closes the same
    /// `resetPassed()` gap `combat_priority_mana_count_seen`'s doc
    /// describes, but for the plain Main1/Main2 `stack_top_is_fresh_own_
    /// item` suppression rather than the combat one-action-per-round
    /// throttle. Root-caused against `game_20260713_002146_0001.txt`
    /// decision 221: SelfPlay casts Highway Robbery (Main1), gets silently
    /// re-suppressed by `stack_top_is_fresh_own_item` every time it's their
    /// priority again -- correct for the *immediate* follow-up, but this
    /// check has no expiry, so it stays true for as long as Highway
    /// Robbery sits unresolved, including *after* PlayerRL1 responds with
    /// two mana-ability activations in between (each of which calls Java's
    /// `PlayerImpl.activateAbility` -> unconditional `resetPassed()` on
    /// *every* player, the same signal `combat_priority_mana_count_seen`
    /// already exists to catch for combat). The reference's very next
    /// SelfPlay record after those is a real, fully-logged `Pass` with a
    /// genuine `{T}: Add {R}.` option alongside it -- not silently
    /// suppressed -- so the kernel's version instead resolves Highway
    /// Robbery a full ask early, silently declining its optional cost
    /// (guessed from a stale peek at that same not-yet-consumed record)
    /// where the reference actually pays it, one-graveyard-card off from
    /// that point on.
    ///
    /// A first attempt keyed this baseline the same way as
    /// `round_opening_stack_len` (reset once per `priority_round`) --
    /// caught by the mandatory full-corpus diff: it regressed
    /// `game_20260713_002202_0024.txt` (21/40 -> still passing count-wise,
    /// but this trace flipped from complete to `decision-kind-mismatch`).
    /// There, SelfPlay taps a mana ability *before* PlayerRL1 casts
    /// Faithless Looting; a `priority_round`-keyed baseline had already
    /// captured that stale tap as "activity", so PlayerRL1's own post-cast
    /// reprompt was wrongly un-suppressed even though *nothing* happened
    /// between the cast and its discard-resolution in the real trace --
    /// the reference keeps PlayerRL1 silently suppressed there, same as
    /// this suppression's original (pre-fix) behavior. Re-keying the
    /// baseline to "since the stack length itself last changed" (this
    /// field) instead of "since this `priority_round` opened" fixes both
    /// traces at once: the baseline only resets when a *new* item lands on
    /// the stack (or one resolves off it), which is exactly the "since
    /// this specific stack-top became fresh" scope `stack_top_is_fresh_own_
    /// item` is supposed to have.
    ///
    /// See `combat_round_opening_mana_count`'s doc for why the
    /// activator-identity check (below) is required and not optional --
    /// the earlier over-broad combat fix attempt (Sol #90-era) that reset
    /// unconditionally on any mana-ability activity regressed 21/40 to
    /// 6/40 by also un-suppressing the activator's *own* immediate
    /// reprompt.
    mana_count_at_last_stack_change: u64,
    /// Set to `Some(card)` the instant `Action::ChooseMadnessCast(true)` is
    /// applied (see `apply`) -- `card` is `state.engine.pending_cast`'s
    /// spell right after that step, i.e. the card now being cast for its
    /// madness cost. Consumed (checked, then cleared unconditionally) the
    /// next time `next_decision` would otherwise apply `StackTopIsCastersOwn`
    /// to this same card, at which point the suppression is skipped instead
    /// -- a real, logged decision reaches the caller.
    ///
    /// Root-caused (increment 13) against `game_20260713_002213_0040.txt`
    /// decision 45: SelfPlay discards Fiery Temper (Madness) to Blood
    /// Token's cost, attempts the madness cast targeting PlayerRL1
    /// (decision 44), and the reference's *very next* record for SelfPlay
    /// is a genuine, fully-logged `ACTIVATE_ABILITY_OR_SPELL` choice between
    /// `Pass` and `Cast Fireblast` -- with Fiery Temper still unresolved
    /// (`opp_life` unchanged; it only drops 3 turns later at decision 47).
    /// The kernel, applying the ordinary `StackTopIsCastersOwn` suppression
    /// (fresh stack top, controlled by the player being asked), silently
    /// force-passed SelfPlay instead, resolving Fiery Temper a full round
    /// early. Confirmed against the real Java reference
    /// (`Mage/src/main/java/mage/abilities/keyword/MadnessAbility.java`'s
    /// `MadnessCastEffect.apply()`): a madness cast is issued via a direct
    /// `owner.cast(castByMadness, game, false, ...)` call from *inside* a
    /// triggered ability's own resolution (`MadnessTriggeredAbility.
    /// resolve()`), never through `ComputerPlayerRL.act()`'s normal
    /// priority-window dispatch -- so the "immediately `pass(game)` after
    /// any stack-using activation" behavior `StackTopIsCastersOwn` exists to
    /// emulate (predicate point 4, `surface.rs`'s module doc) structurally
    /// cannot fire for this specific path; the caster gets a fully genuine
    /// re-ask, exactly once, same as this field's own doc says. Every other
    /// cast/activation (including a *later*, unrelated cast by the same
    /// player while this item is still on the stack) is unaffected -- this
    /// exemption is a single-use flag, not a standing carve-out for the
    /// player or the card.
    madness_cast_reprompt_exemption: Option<ObjectId>,
    /// See `DiscardReshape`'s doc.
    discard: Option<DiscardReshape>,
    /// See `OptionalCostReshape`'s doc.
    optional_cost: Option<OptionalCostReshape>,
}

/// `REPLAY_DEBUG_SURFACE_WALK=1` diagnostic (ReferenceRules v2 grind, Sol
/// #107 continuation): every engine-state field relevant to the stuck-
/// trigger investigation (`rally/coverage_ledger.md`'s "castability-gap"
/// entries), in one line, so a full walk from a trigger landing to the
/// eventual divergence can be read start to finish without cross-
/// referencing multiple print sites by hand.
fn walk_state_snapshot(state: &GameState) -> String {
    format!(
        "step={:?} turn={} active={:?} priority_player={:?} passes={:?} round={} stack_len={} stack={:?}",
        state.step,
        state.turn,
        state.active_player,
        state.priority_player,
        state.engine.priority_passes,
        state.engine.priority_round,
        state.stack.len(),
        state
            .stack
            .iter()
            .map(|si| format!(
                "{}({}) ctrl={:?} inline={} madness={}",
                state.objects.get(si.source).name,
                si.source.0,
                si.controller,
                si.inline_effect.is_some(),
                si.madness_offer
            ))
            .collect::<Vec<_>>()
    )
}

/// Short tag for `Decision`, for the same diagnostic -- a full `{:?}` on
/// `Decision::CastSpellOrPass` dumps every candidate vector (castable
/// spells, mana abilities, land drops, activatable abilities), which is
/// unreadable noise at the "what KIND of decision, for whom" granularity
/// this walk needs; the existing `NOT-IN-BUCKET` diagnostic already prints
/// full candidate detail at the one specific point that needs it.
fn walk_decision_tag(decision: &Decision) -> String {
    match decision {
        Decision::CastSpellOrPass {
            player,
            castable_spells,
            mana_abilities,
            land_drops,
            activatable_abilities,
            plot_actions,
        } => format!(
            "CastSpellOrPass player={player:?} castable={} mana={} land={} activatable={} plot={}",
            castable_spells.len(),
            mana_abilities.len(),
            land_drops.len(),
            activatable_abilities.len(),
            plot_actions.len()
        ),
        Decision::DeclareAttackers { player, eligible } => format!(
            "DeclareAttackers player={player:?} eligible={}",
            eligible.len()
        ),
        Decision::DeclareBlockers {
            player, attackers, ..
        } => format!(
            "DeclareBlockers player={player:?} attackers={}",
            attackers.len()
        ),
        other => format!("{other:?}"),
    }
}

impl HarnessSurfaceV2 {
    pub fn new() -> HarnessSurfaceV2 {
        HarnessSurfaceV2::default()
    }

    pub fn public_context(&self) -> HarnessSurfacePublicContextV2 {
        HarnessSurfacePublicContextV2 {
            blockers: self.blockers.as_ref().map(|b| BlockersReshapePublicV2 {
                current_attacker: b.current_attacker,
                accumulated: b.accumulated.clone(),
                remaining: b.remaining.iter().cloned().collect(),
            }),
            discard: self.discard.as_ref().map(|d| DiscardReshapePublicV2 {
                player: d.player,
                chosen: d.chosen.clone(),
                remaining_choices: d.remaining_choices.clone(),
                remaining_needed: d.remaining_needed,
            }),
            optional_cost: self.optional_cost.map(|o| OptionalCostReshapePublicV2 {
                player: o.player,
                discard_payable: o.discard_payable,
                sacrifice_payable: o.sacrifice_payable,
                stage: match o.stage {
                    OptionalCostStage::Use => OptionalCostStagePublicV2::Use,
                    OptionalCostStage::Which => OptionalCostStagePublicV2::Which,
                },
            }),
            combat_priority_spent: self.combat_priority_spent,
            combat_priority_round_seen: self.combat_priority_round_seen,
            combat_priority_stack_len_seen: self.combat_priority_stack_len_seen,
            combat_priority_mana_count_seen: self.combat_priority_mana_count_seen,
            combat_round_opening_mana_count: self.combat_round_opening_mana_count,
            round_opening_stack_len: self.round_opening_stack_len,
            stack_len_round_seen: self.stack_len_round_seen,
            last_seen_stack_len: self.last_seen_stack_len,
            mana_count_at_last_stack_change: self.mana_count_at_last_stack_change,
            madness_cast_reprompt_exemption: self.madness_cast_reprompt_exemption,
        }
    }

    /// Every auto-resolution performed so far, in the order they happened.
    pub fn suppressions(&self) -> &[Suppression] {
        &self.suppressions
    }

    /// `(discard_payable, sacrifice_payable)` for a genuinely pending
    /// `ChooseOptionalCost` (`state.engine.pending_optional_cost` -- `None`
    /// if none is pending; untouched by this surface's own reshape, which
    /// only ever mutates it via the real, terminal `Action::
    /// ChooseOptionalCost` once the reshape fully resolves, so this stays
    /// accurate for the reshape's *entire* duration, both stages). The
    /// presented `Decision::ChooseOptionalCost` itself carries a *sentinel*
    /// field combination at the `Use` stage (`(false, false)`, chosen so
    /// `decision_texts`-style renderers can tell the two stages apart from
    /// the decision value alone -- see `OptionalCostReshape`'s doc), which
    /// is exactly wrong for a caller that wants the *real* payable flags to
    /// build a one-shot `Action::ChooseOptionalCost` bypass answer
    /// (`bench_kernel.rs`'s random policy, `branch_diff.rs`'s fixed
    /// continuation/force helpers, `replay_burn_v2.rs`'s own look-ahead
    /// guess) regardless of which stage happens to be presented. This
    /// accessor is that real signal -- a pure read of engine state, not
    /// this surface's own bookkeeping, so it needs no `&self` at all.
    pub fn pending_optional_cost_payable(state: &GameState) -> Option<(bool, bool)> {
        state
            .engine
            .pending_optional_cost
            .as_ref()
            .map(|p| (p.discard_payable, p.sacrifice_payable))
    }

    /// The *total* number of cards a genuinely pending `Decision::Discard`
    /// obligation needs (`state.engine.pending_discard`'s own `count` --
    /// `None` if none is pending). The presented `Decision::Discard` itself
    /// always shows `count: 1` (one real card per surfaced pick -- see
    /// `DiscardReshape`'s doc), which tells a caller "how many cards does
    /// *this* pick need" but not "how many *total* picks does this whole
    /// obligation need" -- exactly the distinction a driver walking the
    /// reference's own per-card record sequence needs to loop the *right*
    /// number of times, rather than naively continuing for as long as
    /// `next_decision` keeps returning `Decision::Discard` (ambiguous: nothing
    /// distinguishes "one more pick of this same batch" from "an unrelated,
    /// later single-card discard that happens to immediately follow" --
    /// see `replay_burn_v2.rs::apply_discard`'s doc for the root-caused
    /// regression this fixes). Stable for the reshape's entire duration:
    /// this surface's own bookkeeping never mutates `state.engine.
    /// pending_discard` except via the real, terminal `Action::Discard`
    /// once every card is chosen.
    pub fn pending_discard_total(state: &GameState) -> Option<u32> {
        state.engine.pending_discard.as_ref().map(|pd| pd.count)
    }

    fn record(
        &mut self,
        reason: SuppressionReason,
        auto_action: impl Into<String>,
        before: u64,
        state: &GameState,
    ) {
        let auto_action = auto_action.into();
        if std::env::var("REPLAY_DEBUG_SURFACE_WALK").is_ok() {
            eprintln!(
                "SURFACE_WALK SUPPRESS reason={reason:?} auto_action={auto_action:?} {}",
                walk_state_snapshot(state)
            );
        }
        self.suppressions.push(Suppression {
            reason,
            auto_action,
            state_hash_before: before,
            state_hash_after: state.state_hash(),
        });
    }

    /// See `HarnessSurfaceV1::next_decision` (`surface.rs`) -- identical
    /// logic, duplicated per this module's doc.
    ///
    /// Thin logging wrapper around `next_decision_inner` (ReferenceRules v2
    /// grind, Sol #107 continuation): `REPLAY_DEBUG_SURFACE_WALK=1` logs the
    /// raw `engine::advance_until_decision` result on every loop iteration
    /// (via `next_decision_inner`'s own per-iteration print, see there) and
    /// the final decision actually surfaced to the caller here -- together
    /// with `record`'s own per-suppression logging, this gives a complete,
    /// ordered walk of every internal engine decision the surface consumed
    /// between any two calls, including which suppression rule (if any) ate
    /// which one. Wrapping (rather than logging at every `return` site
    /// inside `next_decision_inner`) is deliberate: that function has many
    /// early returns (sub-decision reshapes, several suppression branches,
    /// the final fall-through) and duplicating a log call at each one would
    /// be exactly the kind of easy-to-miss-one-spot instrumentation this
    /// investigation doesn't want.
    pub fn next_decision(&mut self, state: &mut GameState) -> SurfaceDecision {
        let sd = self.next_decision_inner(state);
        if std::env::var("REPLAY_DEBUG_SURFACE_WALK").is_ok() {
            eprintln!(
                "SURFACE_WALK SURFACED {sd:?} {}",
                walk_state_snapshot(state)
            );
        }
        sd
    }

    fn next_decision_inner(&mut self, state: &mut GameState) -> SurfaceDecision {
        loop {
            if let Some(sd) = self.next_blockers_subdecision(state) {
                return sd;
            }
            if let Some(sd) = self.next_discard_subdecision() {
                return sd;
            }
            if let Some(sd) = self.next_optional_cost_subdecision() {
                return sd;
            }

            let before = state.state_hash();
            let decision = engine::advance_until_decision(state);
            if std::env::var("REPLAY_DEBUG_SURFACE_WALK").is_ok() {
                eprintln!(
                    "SURFACE_WALK RAW_DECISION {} {}",
                    walk_decision_tag(&decision),
                    walk_state_snapshot(state)
                );
            }

            match &decision {
                Decision::CastSpellOrPass {
                    player,
                    castable_spells,
                    mana_abilities,
                    land_drops,
                    activatable_abilities,
                    ..
                } => {
                    if self.stack_len_round_seen != Some(state.engine.priority_round) {
                        self.round_opening_stack_len = state.stack.len();
                        self.stack_len_round_seen = Some(state.engine.priority_round);
                    }
                    // `resetPassed()` gap (see `mana_count_at_last_stack_
                    // change`'s doc): activating a mana ability since
                    // `player`'s own item last landed on the stack is a
                    // real, fresh reason for `player` to be asked again for
                    // real, even though that item is still sitting
                    // unresolved -- only suppress the "my own item,
                    // nothing's happened since it landed" case when the
                    // mana-ability count hasn't moved since the stack was
                    // last this size.
                    //
                    // Root-caused (increment 15) against
                    // `game_20260713_002207_0031.txt` decision 197: this
                    // used to also require `last_mana_ability_activator !=
                    // Some(*player)` -- i.e. only the *opponent's* mana tap
                    // counted, not `player`'s own -- which is too narrow.
                    // `ComputerPlayerRL.act`'s `if (ability.isUsesStack())
                    // pass(game)` (predicate point 4, `surface.rs`'s module
                    // doc) is a one-shot tied to the exact act() call that
                    // completed the stack-using cast/activation; it does
                    // not re-fire on any *later*, unrelated act() call by
                    // that same player, mana ability included (a mana
                    // ability is never itself stack-using, so its own
                    // act() call never appends the extra pass). In 0031,
                    // SelfPlay activates the Blood Token (fresh stack item,
                    // correctly suppressed once), then -- still holding
                    // priority after PlayerRL1's own intervening mana taps
                    // reset both `priority_passes` flags -- activates a
                    // mana ability *themselves* (decision 196); the
                    // reference's very next SelfPlay record (197) is a
                    // real, fully-logged `Pass` alongside a genuine
                    // `{T}: Add {R}.` option, not a silent re-suppression.
                    // Requiring the activator to be the *opponent* left
                    // this exact window silently force-passed instead,
                    // permanently desyncing SelfPlay's trace cursor by one
                    // real record -- the Blood Token's own `Draw a card`
                    // then resolved (via the ordinary two-consecutive-
                    // passes path) one priority window earlier than the
                    // reference, applying the draw before `check_state`
                    // for what the driver mistook for decision 197 (really
                    // the reference's decision 201) expected it.
                    if self.last_seen_stack_len != Some(state.stack.len()) {
                        self.last_seen_stack_len = Some(state.stack.len());
                        self.mana_count_at_last_stack_change =
                            state.engine.mana_ability_activations;
                    }
                    let mana_activity_since_stack_change = state.engine.mana_ability_activations
                        != self.mana_count_at_last_stack_change;
                    let stack_top_is_fresh_own_item = state.stack.len()
                        > self.round_opening_stack_len
                        && state
                            .stack
                            .last()
                            .is_some_and(|item| item.controller == *player)
                        && !mana_activity_since_stack_change;
                    // One-shot madness-cast exemption -- see
                    // `madness_cast_reprompt_exemption`'s doc. Consumed
                    // (cleared) the moment we reach *any* `CastSpellOrPass`
                    // check after a madness cast was attempted, whether or
                    // not it still matches the current stack top (an
                    // unaffordable attempt that fizzled via `abort_cast`
                    // leaves a *different* item exposed, which must still
                    // get ordinary suppression treatment, not a leaked
                    // exemption).
                    let stack_top_is_fresh_own_item =
                        match self.madness_cast_reprompt_exemption.take() {
                            Some(card)
                                if state.stack.last().is_some_and(|item| item.source == card) =>
                            {
                                false
                            }
                            _ => stack_top_is_fresh_own_item,
                        };

                    if matches!(state.step, Step::DeclareAttackers | Step::DeclareBlockers) {
                        if self.combat_priority_round_seen != Some(state.engine.priority_round) {
                            self.combat_priority_spent = [false, false];
                            self.combat_priority_round_seen = Some(state.engine.priority_round);
                            self.combat_priority_stack_len_seen = state.stack.len();
                            self.combat_priority_mana_count_seen =
                                state.engine.mana_ability_activations;
                            self.combat_round_opening_mana_count =
                                state.engine.mana_ability_activations;
                        }
                        // The mana-ability analogue of `stack_top_is_fresh_
                        // own_item`, below: a mana ability never appears on
                        // `state.stack`, so it needs its own durable "was the
                        // thing that reopened this round mine" signal --
                        // see `combat_round_opening_mana_count`'s doc.
                        //
                        // The `state.stack.len() == self.combat_priority_
                        // stack_len_seen` clause guards a gap in that signal:
                        // it only remembers *whether my own mana tap was the
                        // last reopening event*, never checking if a *later,
                        // stack-based* reopening (someone activating a real
                        // ability) has since superseded it. Root-caused
                        // against `game_20260713_002148_0003.txt` decisions
                        // 34-38: SelfPlay taps mana (their own reopen),
                        // *then* PlayerRL1 activates Masked Meower's ability
                        // (a real, stack-landing activation -- not a mana
                        // ability, so `last_mana_ability_activator` still
                        // stales-points at SelfPlay). `combat_priority_stack_
                        // len_seen` hasn't caught up to the new stack length
                        // yet either (its own re-arm runs *after* this check,
                        // and never at all for a self-suppressed player --
                        // see the `continue` a few lines down), so without
                        // this clause `mana_ability_is_fresh_own_action`
                        // stays wrongly true off SelfPlay's now-stale mana
                        // tap, self-suppressing them right through the
                        // window where the reference lets them respond to
                        // Masked Meower's ability with a real `Cast Lava
                        // Dart` -- permanently, since a self-suppressed
                        // player's `continue` also skips the stack-length
                        // re-arm block that would otherwise have corrected
                        // `combat_priority_stack_len_seen` for them.
                        let mana_ability_is_fresh_own_action =
                            state.engine.mana_ability_activations
                                > self.combat_round_opening_mana_count
                                && state.engine.last_mana_ability_activator == Some(*player)
                                && state.stack.len() == self.combat_priority_stack_len_seen;
                        // The acting player's own reopened window from a
                        // cast/activation that just landed on the stack this
                        // round is *always* silently suppressed here, same as
                        // the non-combat branch below -- checked *before*
                        // consulting/mutating `combat_priority_spent` so it
                        // never gets a chance to look "un-spent" for them
                        // (see `combat_priority_stack_len_seen`'s doc).
                        if stack_top_is_fresh_own_item || mana_ability_is_fresh_own_action {
                            engine::step(state, Action::Pass)
                                .expect("Pass is always legal in an offered priority window");
                            self.record(
                                SuppressionReason::StackTopIsCastersOwn,
                                "Pass (forced: caster's own cast/activation/mana-ability still the last thing that happened this round)",
                                before,
                                state,
                            );
                            continue;
                        }
                        // A new stack item appeared this round (either
                        // player's cast/activation) -- Java's `PlayerImpl.
                        // activateAbility` calls `game.getPlayers().
                        // resetPassed()` unconditionally on any successful
                        // action, clearing *every* player's passed flag, not
                        // just the actor's -- so the other player earns a
                        // fresh priority ask here even though they already
                        // "spent" theirs earlier this same `priority_round`.
                        // See `combat_priority_stack_len_seen`'s doc for the
                        // full root-cause (`game_20260713_002152_0007.txt`
                        // decision 24). A mana ability triggers the exact
                        // same `resetPassed()` call but never touches the
                        // stack (605.3b), so it needs its own parallel
                        // signal -- `mana_ability_activations` -- checked
                        // alongside the stack-length one; see
                        // `combat_priority_mana_count_seen`'s doc for the
                        // root-cause this second condition fixes.
                        if state.stack.len() != self.combat_priority_stack_len_seen
                            || state.engine.mana_ability_activations
                                != self.combat_priority_mana_count_seen
                        {
                            self.combat_priority_spent = [false, false];
                            self.combat_priority_stack_len_seen = state.stack.len();
                            self.combat_priority_mana_count_seen =
                                state.engine.mana_ability_activations;
                        }
                        if self.combat_priority_spent[player.index()] {
                            engine::step(state, Action::Pass)
                                .expect("Pass is always legal in an offered priority window");
                            self.record(
                                SuppressionReason::CombatPriorityActionSpent,
                                "Pass (forced: one action per round already taken)",
                                before,
                                state,
                            );
                            continue;
                        }
                        self.combat_priority_spent[player.index()] = true;
                    } else if stack_top_is_fresh_own_item {
                        engine::step(state, Action::Pass)
                            .expect("Pass is always legal in an offered priority window");
                        self.record(
                            SuppressionReason::StackTopIsCastersOwn,
                            "Pass (forced: caster's own cast/activation still unresolved on the stack this round)",
                            before,
                            state,
                        );
                        continue;
                    }
                    let no_real_option = castable_spells.is_empty()
                        && mana_abilities.is_empty()
                        && land_drops.is_empty()
                        && activatable_abilities.is_empty();
                    let step_gated = harness_never_offers_priority(state.step);
                    if step_gated || no_real_option {
                        engine::step(state, Action::Pass)
                            .expect("Pass is always legal in an offered priority window");
                        self.record(
                            if step_gated {
                                SuppressionReason::StepGated
                            } else {
                                SuppressionReason::NoRealOption
                            },
                            "Pass",
                            before,
                            state,
                        );
                        continue;
                    }
                }
                Decision::DeclareAttackers { eligible, .. } => {
                    if eligible.is_empty() {
                        engine::step(state, Action::DeclareAttackers(Vec::new()))
                            .expect("declaring zero attackers is always legal here");
                        self.record(
                            SuppressionReason::NoEligibleAttacker,
                            "DeclareAttackers([])",
                            before,
                            state,
                        );
                        continue;
                    }
                }
                Decision::DeclareBlockers { legal_blockers, .. } => {
                    self.begin_blockers_reshape(legal_blockers.clone(), before, state);
                    // `next_blockers_subdecision` finishes the reshape
                    // itself when its queue drains (whether originally
                    // empty or emptied by filtering), so there is no
                    // separate `finish_blockers_reshape` call needed here
                    // -- see that function's doc.
                    if let Some(sd) = self.next_blockers_subdecision(state) {
                        return sd;
                    }
                    continue;
                }
                Decision::Discard {
                    player,
                    count,
                    choices,
                } => {
                    // See `DiscardReshape`'s doc: begin the per-card
                    // sequence; the loop's top-of-iteration check re-presents
                    // it (one card at a time) on the next pass.
                    self.discard = Some(DiscardReshape {
                        player: *player,
                        remaining_choices: choices.clone(),
                        chosen: Vec::new(),
                        remaining_needed: *count,
                    });
                    continue;
                }
                Decision::ChooseOptionalCost {
                    player,
                    discard_payable,
                    sacrifice_payable,
                } => {
                    // See `OptionalCostReshape`'s doc: begin the two-stage
                    // sequence at the `Use` gate.
                    self.optional_cost = Some(OptionalCostReshape {
                        player: *player,
                        discard_payable: *discard_payable,
                        sacrifice_payable: *sacrifice_payable,
                        stage: OptionalCostStage::Use,
                    });
                    continue;
                }
                _ => {}
            }

            return SurfaceDecision::Decision(decision);
        }
    }

    /// See `HarnessSurfaceV1::apply`.
    pub fn apply(&mut self, state: &mut GameState, action: SurfaceAction) -> Result<(), String> {
        match action {
            SurfaceAction::Action(Action::ChooseMadnessCast(true)) => {
                let result = engine::step(state, Action::ChooseMadnessCast(true));
                if result.is_ok() {
                    // Arm the one-shot exemption -- see
                    // `madness_cast_reprompt_exemption`'s doc. `pending_cast`
                    // is guaranteed `Some` here: `apply_choose_madness_cast`
                    // (the `cast_it == true` branch) unconditionally calls
                    // `begin_cast_ex`, which always sets it.
                    self.madness_cast_reprompt_exemption =
                        state.engine.pending_cast.as_ref().map(|p| p.spell);
                }
                result
            }
            SurfaceAction::Action(Action::OrderTriggers(perm)) => {
                // Snapshot *before* applying: was the stack top, right now,
                // the exact card `madness_cast_reprompt_exemption` is still
                // armed for? Only that specific case (a same-round Madness
                // cast's own resulting triggers) gets the bump below --
                // see the doc block right after this match arm for why an
                // unconditional bump (every `OrderTriggers`, regardless of
                // what it followed) regresses ordinary act()-driven casts.
                let madness_exempt_card_still_on_top = self
                    .madness_cast_reprompt_exemption
                    .is_some_and(|card| state.stack.last().is_some_and(|item| item.source == card));
                let result = engine::step(state, Action::OrderTriggers(perm));
                // A triggered ability landing on the stack (`engine::
                // apply_order_triggers` -> `push_trigger_onto_stack`) never
                // goes through `ComputerPlayerRL.act()` on its own -- the
                // reference's `GameImpl.checkTriggered` places it there
                // automatically, with no `pass(game)`-appending player
                // action anywhere in that path (`chooseTriggeredAbility`'s
                // order pick is itself silent -- see `Decision::
                // OrderTriggers`'s own handling in the driver, no trace
                // record consumed). But that alone does *not* mean every
                // `OrderTriggers` application should bump `round_opening_
                // stack_len` past what it just pushed: for an *ordinary*,
                // act()-driven cast/activation (`finalize_cast`/`finalize_
                // activation`, not a Madness cast), `ComputerPlayerRL.act`'s
                // `pass(game)` already fired for that action itself, and
                // the correct behavior is for that one-shot suppression to
                // cover the very next ask *regardless* of whatever triggers
                // happened to piggyback on the same stack-growth event --
                // root-caused (increment 15) against `game_20260713_
                // 002158_0018.txt` decision 302 (a plain, non-Madness cast
                // whose own 2 same-controller triggers land right after):
                // an earlier, unconditional version of this bump wrongly
                // let PlayerRL1's own immediate reprompt through as a real
                // ask there, when the reference silently force-passes it
                // and resolves both triggers first (`opp_life` already
                // reflecting both by the reference's own next real ask).
                //
                // The one case that *does* need the bump -- root-caused
                // against `game_20260713_002212_0039.txt` decision 151 --
                // is a *Madness* cast's own triggers: `apply_choose_
                // madness_cast`'s `cast_it == true` branch never goes
                // through `act()` either (`MadnessCastEffect.apply()` casts
                // from *inside* a triggered ability's own resolution --
                // see `madness_cast_reprompt_exemption`'s doc), so nothing
                // about that cast *or* its own resulting triggers (here,
                // Guttersnipe's "whenever you cast an instant or sorcery"
                // firing twice off SelfPlay's own Madness-cast Fiery
                // Temper) should ever engage predicate point 4's one-shot
                // at all. `madness_exempt_card_still_on_top` (snapshotted
                // above, *before* this application, since the exemption's
                // own card is what OrderTriggers pushes on top of) tells
                // the two cases apart: only bump when the triggers being
                // ordered right now are landing directly on top of the
                // still-armed Madness-cast card itself.
                //
                // Also stamps `stack_len_round_seen` (Sol #107 fix, found
                // while wiring up `EngineState::stack_len_at_round_open`):
                // without this, the very next `CastSpellOrPass` ask's lazy
                // `if self.stack_len_round_seen != Some(priority_round)`
                // check in `next_decision_inner` still reads `stack_len_
                // round_seen` as unset/stale for this round (a Madness cast
                // never goes through `reset_priority`, so `priority_round`
                // never actually changes across this whole sequence -- this
                // is often the *first* `CastSpellOrPass` this surface has
                // ever seen, so `round_opening_stack_len` was still sitting
                // at its type default, not a real baseline) -- re-firing the
                // general capture path and silently clobbering the bump this
                // block just made. Root-caused via `madness_cast_reprompt_
                // is_not_silently_suppressed_by_its_own_resulting_triggers`
                // regressing the instant `stack_len_at_round_open` was wired
                // in below: that field's own real value (correctly `0`,
                // since `reset_priority` genuinely never ran yet) is a
                // legitimate value for a *different* round, not a
                // trustworthy fallback for a round this method has already
                // hand-corrected.
                if result.is_ok() && madness_exempt_card_still_on_top {
                    self.round_opening_stack_len = state.stack.len();
                    self.stack_len_round_seen = Some(state.engine.priority_round);
                }
                result
            }
            SurfaceAction::Action(Action::Discard(picked)) => {
                // See `DiscardReshape`'s doc: accept either a single-card
                // answer to the currently-presented pick (the per-card
                // sequence a driver walks alongside the reference's own
                // per-card `SELECT_TARGETS` records), or the whole
                // remaining batch at once (every pre-existing H2 caller
                // that predates this reshape and still constructs the
                // engine's original, un-decomposed answer).
                let reshape = self
                    .discard
                    .as_ref()
                    .ok_or("no Discard decision is pending")?;
                if !picked
                    .iter()
                    .all(|id| reshape.remaining_choices.contains(id))
                {
                    return Err("discarded card is not among the legal candidates".to_string());
                }
                let remaining_needed = reshape.remaining_needed;
                if picked.len() as u32 == remaining_needed {
                    let reshape = self.discard.take().expect("checked Some above");
                    let mut chosen = reshape.chosen;
                    chosen.extend(picked);
                    engine::step(state, Action::Discard(chosen))
                } else if picked.len() == 1 {
                    let reshape = self.discard.as_mut().expect("checked Some above");
                    let id = picked[0];
                    reshape.remaining_choices.retain(|&c| c != id);
                    reshape.chosen.push(id);
                    reshape.remaining_needed -= 1;
                    Ok(())
                } else {
                    Err(format!(
                        "Action::Discard must supply either exactly 1 card (answer the current pick) or exactly the {remaining_needed} still needed (resolve the whole reshape at once), got {}",
                        picked.len()
                    ))
                }
            }
            SurfaceAction::Action(Action::ChooseOptionalCost(choice)) => {
                // One-shot bypass: resolve the whole reshape (whichever
                // stage it's currently presenting, if any) immediately --
                // see `OptionalCostReshape`'s doc.
                self.optional_cost = None;
                engine::step(state, Action::ChooseOptionalCost(choice))
            }
            SurfaceAction::Action(Action::ChooseOptionalCostStage(use_it)) => {
                let reshape = self
                    .optional_cost
                    .ok_or("no ChooseOptionalCost decision is pending")?;
                match reshape.stage {
                    OptionalCostStage::Use => {
                        if !use_it {
                            self.optional_cost = None;
                            return engine::step(
                                state,
                                Action::ChooseOptionalCost(OptionalCostChoice::Decline),
                            );
                        }
                        match (reshape.discard_payable, reshape.sacrifice_payable) {
                            (true, true) => {
                                self.optional_cost.as_mut().expect("checked Some above").stage = OptionalCostStage::Which;
                                Ok(())
                            }
                            (true, false) => {
                                self.optional_cost = None;
                                engine::step(state, Action::ChooseOptionalCost(OptionalCostChoice::Discard))
                            }
                            (false, true) => {
                                self.optional_cost = None;
                                engine::step(state, Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand))
                            }
                            (false, false) => Err("ChooseOptionalCostStage(Use) answered yes but neither discard nor sacrifice is payable".to_string()),
                        }
                    }
                    OptionalCostStage::Which => {
                        let choice = if use_it {
                            OptionalCostChoice::Discard
                        } else {
                            OptionalCostChoice::SacrificeLand
                        };
                        self.optional_cost = None;
                        engine::step(state, Action::ChooseOptionalCost(choice))
                    }
                }
            }
            SurfaceAction::Action(a) => engine::step(state, a),
            SurfaceAction::DeclareBlockersForAttacker(blockers) => {
                let reshape = self
                    .blockers
                    .as_mut()
                    .ok_or("no DeclareBlockersForAttacker decision is pending")?;
                let attacker = reshape
                    .current_attacker
                    .take()
                    .ok_or("no DeclareBlockersForAttacker decision is pending")?;
                for b in blockers {
                    reshape.accumulated.push((b, attacker));
                }
                if reshape.remaining.is_empty() {
                    self.finish_blockers_reshape(state);
                }
                Ok(())
            }
        }
    }

    fn begin_blockers_reshape(
        &mut self,
        legal_blockers: Vec<(ObjectId, Vec<ObjectId>)>,
        before: u64,
        state: &GameState,
    ) {
        let mut remaining = std::collections::VecDeque::new();
        for (attacker, blockers) in legal_blockers {
            if blockers.is_empty() {
                self.record(
                    SuppressionReason::NoEligibleBlockersForAttacker,
                    format!("skip attacker {attacker}"),
                    before,
                    state,
                );
                continue;
            }
            remaining.push_back((attacker, blockers));
        }
        self.blockers = Some(BlockersReshape {
            remaining,
            accumulated: Vec::new(),
            current_attacker: None,
        });
    }

    /// Pops the next attacker with a *still-legal* blocker candidate,
    /// filtering out any blocker already committed (`reshape.accumulated`)
    /// to an earlier attacker this same combat -- 509.1c-family rule: one
    /// creature can block at most one attacker. Neither the engine's
    /// one-shot `legal_blockers_for` batch computation (`advance_until_
    /// decision`'s `Decision::DeclareBlockers` arm) nor the original
    /// version of this function accounted for that, so a blocker already
    /// assigned to attacker A was still offered (and, per `check_state`'s
    /// candidate-multiset check against the trace, *expected* to be
    /// offered) as a legal choice for attacker B -- something only the
    /// commit-time check in `apply_declare_blockers` ever caught, and only
    /// as a hard error, not a silent narrowing. The reference's own
    /// per-attacker `selectBlockers` loop evidently narrows sequentially
    /// (never re-offering an already-used blocker), so it silently skips
    /// (logs nothing) any attacker left with zero real candidates once its
    /// only blocker(s) are already spoken for -- same shape as
    /// `begin_blockers_reshape`'s original "zero blockers from the start"
    /// case, just discovered lazily instead of up front. Root-caused
    /// against `game_20260713_002203_0025.txt` decision 278: SelfPlay
    /// declares two identically-named attackers, PlayerRL1 controls a
    /// single eligible blocker, the reference logs exactly one
    /// `DECLARE_BLOCKS` record (for the first attacker) and nothing for
    /// the second; the kernel, offering the same blocker again for
    /// attacker two, asked a phantom second sub-decision.
    ///
    /// A drained-by-filtering queue is finished here too (not left for the
    /// caller), so callers only ever need to check this function's return
    /// value, not separately re-inspect `self.blockers`.
    fn next_blockers_subdecision(&mut self, state: &mut GameState) -> Option<SurfaceDecision> {
        loop {
            let popped = self.blockers.as_mut()?.remaining.pop_front();
            let Some((attacker, legal_blockers)) = popped else {
                self.finish_blockers_reshape(state);
                return None;
            };
            let already_used: Vec<ObjectId> = self
                .blockers
                .as_ref()?
                .accumulated
                .iter()
                .map(|&(b, _)| b)
                .collect();
            let filtered: Vec<ObjectId> = legal_blockers
                .into_iter()
                .filter(|b| !already_used.contains(b))
                .collect();
            if filtered.is_empty() {
                let before = state.state_hash();
                self.record(
                    SuppressionReason::NoEligibleBlockersForAttacker,
                    format!("skip attacker {attacker} (every legal blocker already assigned to an earlier attacker this combat)"),
                    before,
                    state,
                );
                continue;
            }
            self.blockers.as_mut()?.current_attacker = Some(attacker);
            return Some(SurfaceDecision::DeclareBlockersForAttacker {
                attacker,
                legal_blockers: filtered,
            });
        }
    }

    fn finish_blockers_reshape(&mut self, state: &mut GameState) {
        let reshape = self
            .blockers
            .take()
            .expect("finish_blockers_reshape requires an in-progress reshape");
        debug_assert!(
            reshape.remaining.is_empty(),
            "finish_blockers_reshape called before every attacker was asked"
        );
        engine::step(state, Action::DeclareBlockers(reshape.accumulated))
            .expect("accumulated blocks were already checked legal one attacker at a time");
    }

    /// See `DiscardReshape`'s doc. Re-presents the in-progress reshape's
    /// *current* pick as a plain, single-card `Decision::Discard` (`count:
    /// 1`); `None` when no discard reshape is in progress, letting
    /// `next_decision`'s loop fall through to the engine as usual.
    fn next_discard_subdecision(&self) -> Option<SurfaceDecision> {
        let reshape = self.discard.as_ref()?;
        Some(SurfaceDecision::Decision(Decision::Discard {
            player: reshape.player,
            count: 1,
            choices: reshape.remaining_choices.clone(),
        }))
    }

    /// See `OptionalCostReshape`'s doc. Re-presents the in-progress
    /// reshape's current stage as a `Decision::ChooseOptionalCost` with the
    /// stage's own sentinel field combination.
    fn next_optional_cost_subdecision(&self) -> Option<SurfaceDecision> {
        let reshape = self.optional_cost?;
        let (discard_payable, sacrifice_payable) = match reshape.stage {
            OptionalCostStage::Use => (false, false),
            OptionalCostStage::Which => (true, true),
        };
        Some(SurfaceDecision::Decision(Decision::ChooseOptionalCost {
            player: reshape.player,
            discard_payable,
            sacrifice_payable,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::{self, CARD_DEFS};
    use crate::ids::PlayerId;
    use crate::state::{GameState, Target, Zone};

    fn empty_game() -> GameState {
        GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1)
    }

    fn put_on_battlefield(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id = card_def::card_id_by_name(card_name)
            .unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Battlefield,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[player.index()].battlefield.push(obj_id);
        obj_id
    }

    #[test]
    fn provenance_consts_are_pinned() {
        assert_eq!(H2_PREDICATE_VERSION, 2);
        assert_eq!(H2_JAVA_ORACLE_COMMIT.len(), 40, "should be a full git sha");
        assert_ne!(
            H2_JAVA_ORACLE_COMMIT,
            crate::surface::H1_JAVA_ORACLE_COMMIT,
            "H2 must pin its own commit, not reuse H1's"
        );
    }

    #[test]
    fn verify_corpus_provenance_passes_on_exact_match() {
        verify_corpus_provenance(H2_JAVA_ORACLE_COMMIT).expect("identical commit must pass");
    }

    #[test]
    fn verify_corpus_provenance_fails_loudly_on_mismatch() {
        let err = verify_corpus_provenance("deadbeef00000000000000000000000000000000").unwrap_err();
        assert!(err.contains("provenance mismatch"), "got: {err}");
        assert!(
            err.contains(H2_JAVA_ORACLE_COMMIT),
            "error must name the pinned commit: {err}"
        );
    }

    /// Same shape as `surface::tests::step_gated_window_is_suppressed_not_surfaced`
    /// -- proves predicate point 1 (step-gated priority passes) survived the
    /// V1->V2 duplication unchanged.
    #[test]
    fn step_gated_window_is_suppressed_not_surfaced() {
        let mut state = empty_game();
        state.step = Step::Upkeep;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let decision = surface.next_decision(&mut state);

        assert!(matches!(decision, SurfaceDecision::Decision(_)));
        assert!(!surface.suppressions().is_empty());
        let s = &surface.suppressions()[0];
        assert_eq!(s.reason, SuppressionReason::StepGated);
        assert_eq!(s.auto_action, "Pass");
        assert_ne!(s.state_hash_before, 0);
    }

    /// Mirrors `surface::tests::declare_blockers_is_reshaped_per_attacker_and_skips_the_ineligible_one`.
    #[test]
    fn declare_blockers_is_reshaped_per_attacker_and_skips_the_ineligible_one() {
        let mut state = empty_game();
        let attacker_a = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let blocker = put_on_battlefield(&mut state, PlayerId::P1, "Masked Meower");
        state.objects.get_mut(attacker_a).controller = PlayerId::P0;

        state.active_player = PlayerId::P0;
        state.step = Step::DeclareBlockers;
        state.engine.combat.attackers_declared = true;
        state.engine.combat.attackers = vec![attacker_a];

        let mut surface = HarnessSurfaceV2::new();
        let decision = surface.next_decision(&mut state);
        match decision {
            SurfaceDecision::DeclareBlockersForAttacker {
                attacker,
                legal_blockers,
            } => {
                assert_eq!(attacker, attacker_a);
                assert_eq!(legal_blockers, vec![blocker]);
            }
            other => panic!("expected DeclareBlockersForAttacker, got {other:?}"),
        }

        surface
            .apply(
                &mut state,
                SurfaceAction::DeclareBlockersForAttacker(vec![blocker]),
            )
            .unwrap();
        assert!(
            state.engine.combat.blockers_declared,
            "the combined DeclareBlockers action should have been applied automatically"
        );
        assert_eq!(
            state.engine.combat.blocked_by,
            vec![(attacker_a, vec![blocker])]
        );
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002203_0025.txt` decision 278, see
    /// `next_blockers_subdecision`'s doc): two attackers share a single
    /// legal blocker (509.1c-family "one creature blocks at most one
    /// attacker") -- once that blocker is assigned to the first attacker,
    /// the second attacker must be silently skipped (zero real candidates
    /// left), not re-offered the same, now-already-spoken-for blocker.
    #[test]
    fn a_blocker_already_assigned_is_not_re_offered_to_a_later_attacker() {
        let mut state = empty_game();
        let attacker_a = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let attacker_b = put_on_battlefield(&mut state, PlayerId::P0, "Voldaren Epicure");
        let blocker = put_on_battlefield(&mut state, PlayerId::P1, "Masked Meower");
        state.objects.get_mut(attacker_a).controller = PlayerId::P0;
        state.objects.get_mut(attacker_b).controller = PlayerId::P0;

        state.active_player = PlayerId::P0;
        state.step = Step::DeclareBlockers;
        state.engine.combat.attackers_declared = true;
        state.engine.combat.attackers = vec![attacker_a, attacker_b];
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let first = surface.next_decision(&mut state);
        match first {
            SurfaceDecision::DeclareBlockersForAttacker {
                attacker,
                legal_blockers,
            } => {
                assert_eq!(attacker, attacker_a);
                assert_eq!(legal_blockers, vec![blocker]);
            }
            other => panic!("expected DeclareBlockersForAttacker for attacker_a, got {other:?}"),
        }
        surface
            .apply(
                &mut state,
                SurfaceAction::DeclareBlockersForAttacker(vec![blocker]),
            )
            .unwrap();

        // attacker_b's only legal blocker is already spoken for -- must be
        // silently skipped, landing straight on the post-blocks priority
        // window (or beyond, if that also has nothing real to do -- this
        // `empty_game` has empty libraries on both sides, so the engine can
        // legitimately auto-play all the way to a deck-out `GameOver` from
        // here) instead of a second (phantom) DeclareBlockersForAttacker.
        let second = surface.next_decision(&mut state);
        assert!(
            !matches!(&second, SurfaceDecision::DeclareBlockersForAttacker { .. }),
            "attacker_b must not get a second real blockers ask, got {second:?}"
        );
        assert!(state.engine.combat.blockers_declared);
        assert_eq!(
            state.engine.combat.blocked_by,
            vec![(attacker_a, vec![blocker])],
            "attacker_b must end up unblocked, not double-assigned the same blocker"
        );

        let skipped = surface.suppressions().iter().find(|s| {
            s.reason == SuppressionReason::NoEligibleBlockersForAttacker
                && s.auto_action.contains(&attacker_b.to_string())
        });
        assert!(
            skipped.is_some(),
            "expected a NoEligibleBlockersForAttacker suppression naming attacker_b, got {:?}",
            surface.suppressions()
        );
    }

    #[test]
    fn no_real_option_priority_window_is_suppressed() {
        let mut state = empty_game();
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        surface.next_decision(&mut state);

        assert!(!surface.suppressions().is_empty());
        assert_eq!(
            surface.suppressions()[0].reason,
            SuppressionReason::NoRealOption
        );
    }

    #[test]
    fn real_option_priority_window_is_surfaced() {
        let mut state = empty_game();
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let decision = surface.next_decision(&mut state);
        assert!(matches!(
            decision,
            SurfaceDecision::Decision(Decision::CastSpellOrPass { .. })
        ));
        assert!(surface.suppressions().is_empty());
        let _ = CARD_DEFS;
        let _ = Target::Player(PlayerId::P0);
    }

    /// Predicate point 4 (`StackTopIsCastersOwn`), same scenario as
    /// `surface::tests::same_caster_reprompt_after_own_cast_is_suppressed`.
    #[test]
    fn same_caster_reprompt_after_own_cast_is_suppressed() {
        let mut state = empty_game();
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let first = surface.next_decision(&mut state);
        assert!(matches!(
            first,
            SurfaceDecision::Decision(Decision::CastSpellOrPass { .. })
        ));
        assert!(surface.suppressions().is_empty());

        state.players[0].hand.retain(|&h| h != id);
        state.objects.get_mut(id).zone = Zone::Stack;
        state.stack.push(crate::state::StackItem {
            source: id,
            controller: PlayerId::P0,
            targets: vec![Target::Player(PlayerId::P1)],
            inline_effect: None,
            discarded: Vec::new(),
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
        });

        let _second = surface.next_decision(&mut state);
        let suppressions = surface.suppressions();
        assert_eq!(
            suppressions[0].reason,
            SuppressionReason::StackTopIsCastersOwn,
            "got {suppressions:?}"
        );
        assert!(suppressions[0].auto_action.starts_with("Pass"));
    }

    /// Regression test for the increment-14 fix (root-caused against
    /// `game_20260713_002146_0001.txt` decision 221, see `mana_count_at_
    /// last_stack_change`'s doc): outside combat, once the *opponent*
    /// activates a mana ability while the caster's own item still sits
    /// unresolved on the stack, the caster's next ask must be a genuine
    /// `CastSpellOrPass` -- not silently re-suppressed by
    /// `StackTopIsCastersOwn` forever, the way `same_caster_reprompt_after_
    /// own_cast_is_suppressed` (just above) correctly suppresses only the
    /// *immediate* first reprompt.
    #[test]
    fn main_phase_reprompt_is_un_suppressed_after_opponent_mana_activity_since_the_cast() {
        let mut state = empty_game();
        // P0: Lightning Bolt plus *two* Mountains -- one pays for the bolt,
        // the other keeps P0 with a genuine option afterward (so their
        // reprompt is a real ask to inspect, not masked by `NoRealOption`).
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;
        let p0_second_mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(p0_second_mountain).tapped = false;
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;

        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        let d0 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            castable_spells,
            ..
        }) = &d0
        else {
            panic!("expected CastSpellOrPass, got {d0:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert!(castable_spells.contains(&bolt_id));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();

        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P0),
            "got {d1:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
            )
            .unwrap();

        // P0's own immediate follow-up is silently suppressed (their own
        // fresh item); priority lands on P1 for a genuine ask.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(mana_abilities.contains(&p1_mountain));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p1_mountain)),
            )
            .unwrap();

        // P1's mana ability resets *everyone's* passed flag (605.3b +
        // Java's unconditional `resetPassed()`), so P0 must now get a real,
        // un-suppressed ask -- even though Lightning Bolt is still their
        // own unresolved item on top of the stack.
        let d3 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d3
        else {
            panic!("P0 must get a genuine fresh ask after P1's mana ability, got {d3:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert!(mana_abilities.contains(&p0_second_mountain));
        assert_eq!(
            state.stack.len(),
            1,
            "Lightning Bolt must still be unresolved when P0 is asked"
        );

        // P0's *immediate* first reprompt is suppressed exactly once
        // (`StackTopIsCastersOwn`) -- P1's own subsequent auto-pass
        // (`NoRealOption`, once they've spent their only mana source) is
        // expected and irrelevant to what this test proves; the point is
        // that `StackTopIsCastersOwn` does not fire a *second* time for P0.
        let suppressions = surface.suppressions();
        let stack_top_is_casters_own_count = suppressions
            .iter()
            .filter(|s| s.reason == SuppressionReason::StackTopIsCastersOwn)
            .count();
        assert_eq!(stack_top_is_casters_own_count, 1, "P0 must be suppressed exactly once (their immediate reprompt), not re-suppressed after P1's mana ability, got {suppressions:?}");
    }

    /// Regression test (increment 15) for `mana_activity_since_stack_
    /// change`'s own `!= Some(*player)` activator restriction, removed this
    /// increment -- root-caused against `game_20260713_002207_0031.txt`
    /// decision 197. Continues the exact scenario `main_phase_reprompt_is_
    /// un_suppressed_after_opponent_mana_activity_since_the_cast` (above)
    /// already proves un-suppresses P0 (P1's mana ability, an *opponent*
    /// activity, correctly un-suppresses P0's next ask even with Lightning
    /// Bolt still unresolved) one step further: once P0 is legitimately
    /// asked for real, if P0 *themselves* also taps a mana ability (not a
    /// stack-using action -- `Action::ActivateManaAbility` never re-arms
    /// `StackTopIsCastersOwn`, matching `ComputerPlayerRL.act`'s `if
    /// (ability.isUsesStack())` gate never firing for one), P0's *next* ask
    /// must still be genuine too. The removed restriction required `last_
    /// mana_ability_activator != Some(*player)` -- i.e. only credited
    /// *someone else's* mana tap as "activity since the cast" -- so the
    /// instant the *most recent* tap happened to be P0's own, the un-
    /// suppression this test's setup already earned was wrongly withdrawn,
    /// re-arming `StackTopIsCastersOwn` for this exact ask even though
    /// nothing about *whose* mana ability it is changes Java's real
    /// mechanics here (605.3b: no mana ability, anyone's, ever re-engages
    /// predicate point 4's one-shot). In the 0031 trace this permanently
    /// desynced SelfPlay's cursor by one record, letting the kernel resolve
    /// the Blood Token's `Draw a card` a full priority window early.
    #[test]
    fn main_phase_reprompt_is_still_un_suppressed_after_the_casters_own_mana_activity_follows_the_opponents(
    ) {
        let mut state = empty_game();
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(bolt_id);
        // P0: one Mountain pays for the bolt, a second and third stay
        // untapped -- one to be P0's *own* mana activity under test, one
        // left over so P0's final ask is a genuine option, not a trivial
        // `NoRealOption` Pass that would prove nothing about
        // `StackTopIsCastersOwn` specifically.
        for _ in 0..3 {
            let m = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
            state.objects.get_mut(m).tapped = false;
        }
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;

        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Establish the pre-cast baseline (`round_opening_stack_len` etc.)
        // via a real `next_decision` call *before* casting -- same
        // requirement as `main_phase_reprompt_is_un_suppressed_after_
        // opponent_mana_activity_since_the_cast`'s own `d0`: skipping this
        // means the first resync happens only once `next_decision` is next
        // called (after the cast already grew the stack), wrongly
        // capturing the *post*-growth length as the baseline.
        let d0 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            castable_spells,
            ..
        }) = &d0
        else {
            panic!("expected CastSpellOrPass, got {d0:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert!(castable_spells.contains(&bolt_id));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();
        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P0),
            "got {d1:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
            )
            .unwrap();

        // P0's own immediate reprompt: suppressed. P1 asked next.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(mana_abilities.contains(&p1_mountain));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p1_mountain)),
            )
            .unwrap();

        // P1's mana ability (opponent activity) legitimately un-suppresses
        // P0 -- already covered by the test above, re-established here as
        // this test's own starting point.
        let d3 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d3
        else {
            panic!("expected a genuine ask for P0 after P1's mana activity, got {d3:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert_eq!(
            mana_abilities.len(),
            2,
            "both of P0's remaining Mountains should be offered"
        );
        let p0_mountain_to_tap = mana_abilities[0];
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p0_mountain_to_tap)),
            )
            .unwrap();

        // The critical assertion: P0's *own* mana tap must not withdraw the
        // un-suppression P1's earlier activity already earned -- P0's next
        // ask must still be genuine, with Lightning Bolt still unresolved.
        let d4 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d4
        else {
            panic!("expected a genuine ask for P0 after their own mana tap, got {d4:?} -- StackTopIsCastersOwn must not re-arm merely because the most recent tap was P0's own")
        };
        assert_eq!(*player, PlayerId::P0);
        assert_eq!(
            mana_abilities.len(),
            1,
            "P0's one remaining Mountain should still be offered"
        );
        assert_eq!(
            state.stack.len(),
            1,
            "Lightning Bolt must still be unresolved throughout"
        );

        let suppressions = surface.suppressions();
        let stack_top_is_casters_own_count = suppressions
            .iter()
            .filter(|s| s.reason == SuppressionReason::StackTopIsCastersOwn)
            .count();
        assert_eq!(stack_top_is_casters_own_count, 1, "P0 must be suppressed exactly once total (their immediate reprompt), got {suppressions:?}");
    }

    /// Regression test for the increment-14 fix's own false-positive trap
    /// (root-caused against `game_20260713_002202_0024.txt`: an intermediate
    /// version of the fix above keyed its "did the opponent do something"
    /// baseline off `priority_round` rather than off the stack's own length
    /// changing, so an opponent mana ability *before* the cast -- still
    /// earlier in the same `priority_round` -- was wrongly read as "activity
    /// since the cast," un-suppressing a reprompt the reference actually
    /// keeps silent). Mana activity that happened *before* `player`'s item
    /// even landed on the stack must not count -- only activity *since*.
    #[test]
    fn main_phase_reprompt_stays_suppressed_when_the_opponent_mana_activity_predates_the_cast() {
        let mut state = empty_game();
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;
        // A second Mountain for P1 -- tapping the first one (before P0's
        // cast) must not leave P1 with zero options later, or P1's own
        // later reprompt would itself get silently `NoRealOption`-passed
        // straight into full resolution (and, with both test libraries
        // empty, on into an unrelated empty-library loss) instead of
        // surfacing the real ask this test means to inspect.
        let p1_second_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_second_mountain).tapped = false;
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;

        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P1;

        let mut surface = HarnessSurfaceV2::new();

        // P1 taps one Mountain for mana *before* P0 casts anything -- still
        // the same `priority_round` (nothing has bumped it yet) -- then
        // explicitly passes (a mana ability doesn't hand away priority by
        // itself; the second Mountain is only there so this Pass is a real
        // choice, not a `NoRealOption` auto-pass indistinguishable from it).
        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P1),
            "got {d1:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p1_mountain)),
            )
            .unwrap();

        let d1b = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d1b
        else {
            panic!("got {d1b:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert_eq!(
            mana_abilities,
            &[p1_second_mountain],
            "the tapped Mountain must no longer be offered"
        );
        surface
            .apply(&mut state, SurfaceAction::Action(Action::Pass))
            .unwrap();

        // Priority returns to P0, who casts Lightning Bolt at P1.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            castable_spells,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert!(castable_spells.contains(&bolt_id));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();

        let d3 = surface.next_decision(&mut state);
        assert!(
            matches!(&d3, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P0),
            "got {d3:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
            )
            .unwrap();

        // Nothing has happened *since* Lightning Bolt landed on the stack --
        // P1's only mana activity was earlier, before the cast -- so P0's
        // own immediate reprompt must stay silently suppressed, exactly as
        // `same_caster_reprompt_after_own_cast_is_suppressed` establishes
        // for the no-intervening-activity case.
        let d4 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d4
        else {
            panic!("priority must skip straight past P0's own reprompt to P1, got {d4:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(mana_abilities.contains(&p1_second_mountain));
        assert_eq!(
            state.stack.len(),
            1,
            "Lightning Bolt must still be unresolved when P1 is asked"
        );
        let suppressions = surface.suppressions();
        let last = suppressions
            .last()
            .expect("P0's own reprompt must have been suppressed");
        assert_eq!(
            last.reason,
            SuppressionReason::StackTopIsCastersOwn,
            "got {suppressions:?}"
        );
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002152_0007.txt` decision 24, see `HarnessSurfaceV2`'s
    /// `combat_priority_stack_len_seen` doc): once *both* players have spent
    /// their one `DeclareAttackers`/`DeclareBlockers` priority action this
    /// round, a fresh cast by one of them must still grant the *other*
    /// player one genuine extra ask (Java's `PlayerImpl.activateAbility`
    /// resets *every* player's passed flag on any successful action) --
    /// while the caster's own reopened window stays silently suppressed via
    /// `StackTopIsCastersOwn`, unchanged from before this fix.
    #[test]
    fn combat_throttle_regrants_the_other_player_a_fresh_ask_after_a_mid_round_cast() {
        let mut state = empty_game();
        // P0: a mana source only (no spell) -- enough to be a "real option"
        // (not `NoRealOption`) without giving them anything to cast.
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;

        // P1: Lightning Bolt + a Mountain to cast it with.
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P1,
            controller: PlayerId::P1,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[1].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state
            .objects
            .get_mut(state.players[1].battlefield[0])
            .tapped = false;

        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers_declared = true;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Round 1: P0 spends their one action (Pass)...
        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0),
            "got {d1:?}"
        );
        surface
            .apply(&mut state, SurfaceAction::Action(Action::Pass))
            .unwrap();

        // ...then P1 spends theirs by casting Lightning Bolt at P0.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            castable_spells,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(castable_spells.contains(&bolt_id));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();

        let d3 = surface.next_decision(&mut state);
        assert!(
            matches!(&d3, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P1),
            "got {d3:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P0))),
            )
            .unwrap();

        // Both players' `combat_priority_spent` flags are now stale-true
        // from round 1, but Lightning Bolt just landed on the stack this
        // same `priority_round` (`finalize_cast` doesn't bump it). Before
        // the fix, the next call would silently force-pass *both* P1 (via
        // `StackTopIsCastersOwn`, correct) *and* P0 (via the stale
        // `CombatPriorityActionSpent`, wrong), resolving the spell without
        // ever giving P0 a real chance to respond to it.
        let d4 = surface.next_decision(&mut state);
        assert!(
            matches!(&d4, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0),
            "P0 must get a genuine fresh ask in response to P1's new spell, got {d4:?}"
        );
        assert_eq!(
            state.stack.len(),
            1,
            "Lightning Bolt must still be unresolved when P0 is asked"
        );

        let suppressions = surface.suppressions();
        let last = suppressions
            .last()
            .expect("P1's own reprompt must have been suppressed");
        assert_eq!(
            last.reason,
            SuppressionReason::StackTopIsCastersOwn,
            "got {suppressions:?}"
        );
        assert!(
            !suppressions
                .iter()
                .any(|s| s.reason == SuppressionReason::CombatPriorityActionSpent),
            "got {suppressions:?}"
        );
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002148_0003.txt` decision 34 and
    /// `game_20260713_002202_0024.txt` decision 179, see
    /// `combat_round_opening_mana_count`'s doc): a mid-round *mana ability*
    /// (not a cast/activation) must have the exact same two effects a cast
    /// does in `combat_throttle_regrants_the_other_player_a_fresh_ask_
    /// after_a_mid_round_cast` above -- (1) it re-arms the *other* player's
    /// stale-spent throttle flag, and (2) the activator's own immediate
    /// reprompt stays silently suppressed -- even though a mana ability
    /// never touches `state.stack`, so neither effect can be detected via
    /// stack length the way the cast case is.
    #[test]
    fn combat_throttle_regrants_the_other_player_a_fresh_ask_after_a_mid_round_mana_ability() {
        let mut state = empty_game();
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state
            .objects
            .get_mut(state.players[0].battlefield[0])
            .tapped = false;
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;

        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers_declared = true;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Round 1: P0 spends their one action (Pass)...
        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0),
            "got {d1:?}"
        );
        surface
            .apply(&mut state, SurfaceAction::Action(Action::Pass))
            .unwrap();

        // ...then P1 spends theirs by tapping their Mountain for mana --
        // no cast, no stack growth at all.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(mana_abilities.contains(&p1_mountain));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p1_mountain)),
            )
            .unwrap();

        // Both players' `combat_priority_spent` flags are now stale-true
        // from round 1, and the stack is still empty (a mana ability never
        // touches it) -- so the pre-fix stack-length-only re-arm would see
        // nothing at all here and silently force-pass straight through the
        // rest of combat. The fix must (a) genuinely re-ask P0, not P1
        // again, and (b) not need any stack growth to do it.
        let d3 = surface.next_decision(&mut state);
        assert!(
            matches!(&d3, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0),
            "P0 must get a genuine fresh ask in response to P1's mana ability, got {d3:?}"
        );
        assert!(
            state.stack.is_empty(),
            "a mana ability never puts anything on the stack"
        );

        let suppressions = surface.suppressions();
        let last = suppressions
            .last()
            .expect("P1's own reprompt after their mana ability must have been suppressed");
        assert_eq!(
            last.reason,
            SuppressionReason::StackTopIsCastersOwn,
            "got {suppressions:?}"
        );
        assert!(
            !suppressions
                .iter()
                .any(|s| s.reason == SuppressionReason::CombatPriorityActionSpent),
            "got {suppressions:?}"
        );
    }

    /// Regression test for the increment-14 fix (root-caused against
    /// `game_20260713_002148_0003.txt` decisions 34-38, see
    /// `mana_ability_is_fresh_own_action`'s doc): once a player's own mana
    /// tap has been superseded by a *later* stack-landing action (someone's
    /// real cast/activation, not another mana ability), that mana tap must
    /// no longer count as "the last thing that happened was mine" --
    /// `mana_ability_is_fresh_own_action` tracked only "was my mana ability
    /// the most recent one," never checking whether a later, non-mana
    /// reopening event had since superseded it, so this player stayed
    /// wrongly self-suppressed straight through the window where the
    /// reference lets them respond for real.
    #[test]
    fn combat_throttle_stale_own_mana_tap_is_superseded_by_a_later_cast() {
        let mut state = empty_game();
        let p0_mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(p0_mountain).tapped = false;
        // A second Mountain for P0 -- tapping the first one for their round-
        // opening action must not leave them with zero options later, or
        // their later reprompt would itself get silently `NoRealOption`-
        // passed straight into full resolution (and, with both test
        // libraries empty, on into an unrelated empty-library loss) instead
        // of surfacing the real ask this test means to inspect.
        let p0_second_mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(p0_second_mountain).tapped = false;

        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P1,
            controller: PlayerId::P1,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[1].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state
            .objects
            .get_mut(state.players[1].battlefield[0])
            .tapped = false;

        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers_declared = true;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Round 1: P0 spends their one action by tapping mana (not
        // casting/passing) -- this is what stales `last_mana_ability_
        // activator` to P0.
        let d1 = surface.next_decision(&mut state);
        assert!(
            matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0),
            "got {d1:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ActivateManaAbility(p0_mountain)),
            )
            .unwrap();

        // ...then P1 spends theirs by casting Lightning Bolt at P0 -- a
        // real, stack-landing action, *not* a mana ability.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            castable_spells,
            ..
        }) = &d2
        else {
            panic!("expected CastSpellOrPass, got {d2:?}")
        };
        assert_eq!(*player, PlayerId::P1);
        assert!(castable_spells.contains(&bolt_id));
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();

        let d3 = surface.next_decision(&mut state);
        assert!(
            matches!(&d3, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P1),
            "got {d3:?}"
        );
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P0))),
            )
            .unwrap();

        // P1's cast is a real, later reopening event -- P0's stale mana tap
        // must not keep them self-suppressed through this window; they must
        // get a genuine fresh ask, with Lightning Bolt still unresolved.
        let d4 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass {
            player,
            mana_abilities,
            ..
        }) = &d4
        else {
            panic!("P0 must get a genuine fresh ask after P1's cast, not stay suppressed off their own stale mana tap, got {d4:?}")
        };
        assert_eq!(*player, PlayerId::P0);
        assert!(mana_abilities.contains(&p0_second_mountain));
        assert_eq!(
            state.stack.len(),
            1,
            "Lightning Bolt must still be unresolved when P0 is asked"
        );

        let suppressions = surface.suppressions();
        assert!(
            !suppressions
                .iter()
                .any(|s| s.reason == SuppressionReason::CombatPriorityActionSpent),
            "got {suppressions:?}"
        );
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002213_0040.txt` decision 45, see
    /// `madness_cast_reprompt_exemption`'s doc): the caster's own reprompt
    /// right after finishing a madness cast must be a genuine, un-suppressed
    /// decision -- unlike an ordinary cast/activation's reprompt, which
    /// `StackTopIsCastersOwn` correctly does silently suppress (see
    /// `same_caster_reprompt_after_own_cast_is_suppressed`, just above).
    #[test]
    fn madness_cast_reprompt_is_not_silently_suppressed() {
        let mut state = empty_game();
        // Fiery Temper, already exiled by Madness's replacement effect
        // (702.33a) -- this test starts from the madness offer already
        // sitting on the stack, not from a full discard flow.
        let temper_def = card_def::card_id_by_name("Fiery Temper").unwrap();
        let temper = state.objects.push(crate::state::GameObject {
            card_def: temper_def,
            name: "Fiery Temper".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Exile,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });

        // A second real spell in hand, so the post-cast reprompt (if it
        // isn't wrongly suppressed) has a genuine alternative to offer, not
        // just a trivial "Pass, nothing else to do" window.
        let lava_dart_def = card_def::card_id_by_name("Lava Dart").unwrap();
        let lava_dart = state.objects.push(crate::state::GameObject {
            card_def: lava_dart_def,
            name: "Lava Dart".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(lava_dart);

        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");

        state.stack.push(crate::state::StackItem {
            source: temper,
            controller: PlayerId::P0,
            targets: vec![],
            inline_effect: None,
            discarded: Vec::new(),
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: true,
            kicked: false,
        });
        state.engine.priority_passes = [true, true];
        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, card }) => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast, got {other:?}"),
        }
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseMadnessCast(true)),
            )
            .unwrap();

        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseTargets {
                player,
                legal_targets,
                ..
            }) => {
                assert_eq!(player, PlayerId::P0);
                assert!(legal_targets.contains(&Target::Player(PlayerId::P1)));
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
                    )
                    .unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        // The critical assertion: the caster's own reprompt after finishing
        // the madness cast must be genuine (Lava Dart + the second Mountain
        // both still offered), not a silent forced Pass.
        let reprompt = surface.next_decision(&mut state);
        match &reprompt {
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                player,
                castable_spells,
                ..
            }) => {
                assert_eq!(*player, PlayerId::P0);
                assert!(
                    castable_spells.contains(&lava_dart),
                    "expected Lava Dart still offered, got {reprompt:?}"
                );
            }
            other => panic!(
                "expected a real CastSpellOrPass reprompt for the madness caster, got {other:?}"
            ),
        }
        assert!(
            !surface
                .suppressions()
                .iter()
                .any(|s| s.reason == SuppressionReason::StackTopIsCastersOwn),
            "the madness cast's own reprompt must not be silently suppressed, got {:?}",
            surface.suppressions()
        );
        assert_eq!(
            state.stack.len(),
            1,
            "Fiery Temper must still be unresolved on the stack at the reprompt"
        );
    }

    /// Regression test (increment 15) for the `Action::OrderTriggers`
    /// `round_opening_stack_len` bump in `apply` -- root-caused against
    /// `game_20260713_002212_0039.txt` decision 151. Same shape as
    /// `madness_cast_reprompt_is_not_silently_suppressed` (above), but with
    /// *two* Guttersnipes on P0's own battlefield: casting Fiery Temper (an
    /// instant/sorcery) via Madness fires both "whenever you cast an
    /// instant or sorcery" triggers, a same-controller 2+ group that goes
    /// through `Decision::OrderTriggers`/`Action::OrderTriggers` -- unlike
    /// a single such trigger, which `engine::drain_pending_triggers_or_
    /// decide` auto-pushes with no surfaced decision at all (see `apply`'s
    /// `OrderTriggers` doc for that gap). Both triggers land on the stack
    /// immediately on top of the still-armed `madness_cast_reprompt_
    /// exemption`'s own card; without the bump, they were `round_opening_
    /// stack_len`-fresh and P0-controlled, so the reprompt right after
    /// ordering them was wrongly re-suppressed by `StackTopIsCastersOwn`
    /// (the exemption itself, keyed to the *exact* stack top matching the
    /// Madness-cast card, no longer matched once a trigger sat on top of
    /// it) -- letting the kernel resolve the first trigger (2 damage to
    /// the opponent) a full priority window before the reference does.
    #[test]
    fn madness_cast_reprompt_is_not_silently_suppressed_by_its_own_resulting_triggers() {
        let mut state = empty_game();
        let temper_def = card_def::card_id_by_name("Fiery Temper").unwrap();
        let temper = state.objects.push(crate::state::GameObject {
            card_def: temper_def,
            name: "Fiery Temper".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Exile,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });

        let lava_dart_def = card_def::card_id_by_name("Lava Dart").unwrap();
        let lava_dart = state.objects.push(crate::state::GameObject {
            card_def: lava_dart_def,
            name: "Lava Dart".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(lava_dart);

        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");

        state.stack.push(crate::state::StackItem {
            source: temper,
            controller: PlayerId::P0,
            targets: vec![],
            inline_effect: None,
            discarded: Vec::new(),
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: true,
            kicked: false,
        });
        state.engine.priority_passes = [true, true];
        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;
        let opponent_life_before = state.players[1].life;

        let mut surface = HarnessSurfaceV2::new();
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, card }) => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast, got {other:?}"),
        }
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::ChooseMadnessCast(true)),
            )
            .unwrap();

        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseTargets {
                player,
                legal_targets,
                ..
            }) => {
                assert_eq!(player, PlayerId::P0);
                assert!(legal_targets.contains(&Target::Player(PlayerId::P1)));
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
                    )
                    .unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        // Both Guttersnipe triggers, same controller -> a real
        // `OrderTriggers` decision (unlike a lone trigger, silently
        // auto-pushed with no decision at all).
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::OrderTriggers { player, pending }) => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(
                    pending.len(),
                    2,
                    "both Guttersnipes' triggers must be in the same same-controller group"
                );
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())),
                    )
                    .unwrap();
            }
            other => panic!("expected OrderTriggers for both Guttersnipe triggers, got {other:?}"),
        }

        // The critical assertion: P0's reprompt right after ordering their
        // own triggers must still be genuine, and neither trigger may have
        // resolved yet (both still sit on the stack, above Fiery Temper).
        let reprompt = surface.next_decision(&mut state);
        match &reprompt {
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                player,
                castable_spells,
                ..
            }) => {
                assert_eq!(*player, PlayerId::P0);
                assert!(
                    castable_spells.contains(&lava_dart),
                    "expected Lava Dart still offered, got {reprompt:?}"
                );
            }
            other => panic!("expected a real CastSpellOrPass reprompt for P0, got {other:?}"),
        }
        assert!(
            !surface.suppressions().iter().any(|s| s.reason == SuppressionReason::StackTopIsCastersOwn),
            "P0's reprompt must not be silently suppressed by their own just-ordered triggers, got {:?}",
            surface.suppressions()
        );
        assert_eq!(
            state.stack.len(),
            3,
            "Fiery Temper plus both unresolved Guttersnipe triggers must still be on the stack"
        );
        assert_eq!(
            state.players[1].life, opponent_life_before,
            "neither Guttersnipe trigger may have resolved yet"
        );
    }

    /// Companion regression guard for the fix above -- root-caused against
    /// `game_20260713_002158_0018.txt` decision 302, where an earlier,
    /// *unconditional* version of the `OrderTriggers` bump (applied to
    /// every `Action::OrderTriggers`, not just ones landing on a still-
    /// armed Madness-cast card) wrongly let PlayerRL1's own immediate
    /// reprompt through as a real ask right after an *ordinary* (non-
    /// Madness) cast whose own same-controller trigger pair fires --
    /// exactly the mechanism family this increment's own full-corpus diff
    /// exists to catch. `ComputerPlayerRL.act`'s `pass(game)` really does
    /// fire once for a plain stack-using cast (predicate point 4,
    /// unaffected by whatever the cast happens to also trigger), so the
    /// reference silently force-passes PlayerRL1 here and resolves both
    /// triggers before ever asking again -- the bump must stay scoped to
    /// the Madness-cast case the fix above targets, not fire for this one.
    #[test]
    fn ordinary_cast_reprompt_stays_suppressed_by_its_own_resulting_triggers() {
        let mut state = empty_game();
        let bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let bolt_id = state.objects.push(crate::state::GameObject {
            card_def: bolt,
            name: "Lightning Bolt".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let p0_second_mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(p0_second_mountain).tapped = false;
        put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        // P1 needs a real option too -- otherwise their own post-cast ask
        // auto-passes via `NoRealOption` (transparent, not interceptable),
        // both players end up passed, and *one* Guttersnipe trigger
        // resolves before `next_decision` ever returns control here --
        // which then opens a genuinely fresh round and un-suppresses P0
        // for an unrelated reason (correct reference behavior, but not
        // what this test means to isolate: whether P0's *immediate*
        // reprompt, right after the triggers are ordered, stays
        // suppressed).
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;

        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        // Establish the pre-cast baseline via a real `next_decision` call
        // first -- see the companion fix's own test, above, for why
        // skipping this wrongly captures the *post*-cast stack length as
        // the "round opened this tall" baseline.
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                player,
                castable_spells,
                ..
            }) => {
                assert_eq!(player, PlayerId::P0);
                assert!(castable_spells.contains(&bolt_id));
            }
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
        surface
            .apply(
                &mut state,
                SurfaceAction::Action(Action::CastSpell(bolt_id)),
            )
            .unwrap();
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) => {
                assert_eq!(player, PlayerId::P0);
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1))),
                    )
                    .unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::OrderTriggers { player, pending }) => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(pending.len(), 2);
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())),
                    )
                    .unwrap();
            }
            other => panic!("expected OrderTriggers for both Guttersnipe triggers, got {other:?}"),
        }

        // Unlike the Madness case above: this is a plain act()-driven cast,
        // so the one-shot suppression still applies here, regardless of
        // the triggers riding along on the same stack-growth event -- P1
        // must be asked next (with a genuine option, so the ask stops
        // right here rather than cascading past an unrelated auto-pass),
        // not P0 again.
        let reprompt = surface.next_decision(&mut state);
        match &reprompt {
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                player,
                mana_abilities,
                ..
            }) => {
                assert_eq!(
                    *player,
                    PlayerId::P1,
                    "P0's reprompt must stay suppressed; P1 should be asked next"
                );
                assert!(mana_abilities.contains(&p1_mountain));
            }
            other => panic!("expected a CastSpellOrPass ask for P1, got {other:?}"),
        }
        assert_eq!(state.stack.len(), 3, "Lightning Bolt plus both unresolved Guttersnipe triggers must still be on the stack when P1 is asked");
        assert_eq!(
            surface
                .suppressions()
                .iter()
                .filter(|s| s.reason == SuppressionReason::StackTopIsCastersOwn)
                .count(),
            1,
            "P0's immediate reprompt must still be suppressed exactly once, got {:?}",
            surface.suppressions()
        );
    }

    /// Kicked-path guard retained from the ReferenceRules v2 investigation.
    /// The original corpus diagnosis was wrong: in
    /// `game_20260714_200336_0016.txt`, Java's log says `Pay Kicker {R} ? ...
    /// decision=NO`. Java's intervening-if gate therefore creates no
    /// Bushwhacker trigger at all, while the kernel used to create one and
    /// defer the Kicker check until its effect resolved. The actual fix is
    /// `trigger::TriggeredAbilityDef::intervening_if_kicked`; this test still
    /// proves that the distinct, genuinely kicked path resolves normally
    /// through the surface even when the opponent has a live mana option.
    #[test]
    fn goblin_bushwhacker_kicked_trigger_resolves_in_the_minimal_single_active_player_case() {
        let mut state = empty_game();
        for _ in 0..4 {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let bushwhacker_def = card_def::card_id_by_name("Goblin Bushwhacker").unwrap();
        let bushwhacker = state.objects.push(crate::state::GameObject {
            card_def: bushwhacker_def,
            name: "Goblin Bushwhacker".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(bushwhacker);

        let raider_def = card_def::card_id_by_name("Goblin Tomb Raider").unwrap();
        let raider = state.objects.push(crate::state::GameObject {
            card_def: raider_def,
            name: "Goblin Tomb Raider".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[0].hand.push(raider);
        // A real P1 (a live Mountain, so `mana_abilities` is non-empty for
        // them too) -- matches every sampled corpus divergence's own
        // `kernel_mana` field always showing something, unlike a totally
        // inert opponent with nothing to ever choose between but Pass.
        put_on_battlefield(&mut state, PlayerId::P1, "Mountain");

        let mut surface = HarnessSurfaceV2::new();
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                castable_spells, ..
            }) => {
                assert!(
                    castable_spells.contains(&bushwhacker),
                    "Bushwhacker should be castable turn 1 with 4 Mountains up"
                );
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::CastSpell(bushwhacker)),
                    )
                    .unwrap();
            }
            other => panic!("expected CastSpellOrPass offering Bushwhacker, got {other:?}"),
        }
        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseKicker { .. }) => {
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::ChooseKicker(true)),
                    )
                    .unwrap();
            }
            other => panic!("expected ChooseKicker, got {other:?}"),
        }

        let mut raider_became_castable = false;
        let mut iterations = 0;
        for _ in 0..40 {
            iterations += 1;
            match surface.next_decision(&mut state) {
                SurfaceDecision::Decision(Decision::CastSpellOrPass {
                    castable_spells, ..
                }) => {
                    if castable_spells.contains(&raider) {
                        raider_became_castable = true;
                        break;
                    }
                    surface
                        .apply(&mut state, SurfaceAction::Action(Action::Pass))
                        .unwrap();
                }
                other => panic!("unexpected decision while draining priority: {other:?}"),
            }
        }

        assert!(
            raider_became_castable,
            "stack never emptied after Bushwhacker's kicked ETB trigger after {iterations} Pass-only iterations \
             -- stack len is now {} (contents: {:?}), i.e. the surface layer (not the raw engine) is where this gets stuck",
            state.stack.len(),
            state.stack.iter().map(|si| si.source).collect::<Vec<_>>()
        );
    }
}
