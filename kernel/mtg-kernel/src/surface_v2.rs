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

use crate::engine::{self, Action, Decision};
use crate::ids::ObjectId;
use crate::state::{GameState, Step};
pub use crate::surface::{harness_never_offers_priority, Suppression, SuppressionReason, SurfaceAction, SurfaceDecision};

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
pub const H2_JAVA_ORACLE_COMMIT: &str = "6de2528fada1c740ceb5fdda0f273bdb9ee28b79";

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
    round_opening_stack_len: usize,
    stack_len_round_seen: Option<u64>,
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
}

impl HarnessSurfaceV2 {
    pub fn new() -> HarnessSurfaceV2 {
        HarnessSurfaceV2::default()
    }

    /// Every auto-resolution performed so far, in the order they happened.
    pub fn suppressions(&self) -> &[Suppression] {
        &self.suppressions
    }

    fn record(&mut self, reason: SuppressionReason, auto_action: impl Into<String>, before: u64, state: &GameState) {
        self.suppressions.push(Suppression { reason, auto_action: auto_action.into(), state_hash_before: before, state_hash_after: state.state_hash() });
    }

    /// See `HarnessSurfaceV1::next_decision` (`surface.rs`) -- identical
    /// logic, duplicated per this module's doc.
    pub fn next_decision(&mut self, state: &mut GameState) -> SurfaceDecision {
        loop {
            if let Some(sd) = self.next_blockers_subdecision(state) {
                return sd;
            }

            let before = state.state_hash();
            let decision = engine::advance_until_decision(state);

            match &decision {
                Decision::CastSpellOrPass { player, castable_spells, mana_abilities, land_drops, activatable_abilities, .. } => {
                    if self.stack_len_round_seen != Some(state.engine.priority_round) {
                        self.round_opening_stack_len = state.stack.len();
                        self.stack_len_round_seen = Some(state.engine.priority_round);
                    }
                    let stack_top_is_fresh_own_item = state.stack.len() > self.round_opening_stack_len
                        && state.stack.last().is_some_and(|item| item.controller == *player);
                    // One-shot madness-cast exemption -- see
                    // `madness_cast_reprompt_exemption`'s doc. Consumed
                    // (cleared) the moment we reach *any* `CastSpellOrPass`
                    // check after a madness cast was attempted, whether or
                    // not it still matches the current stack top (an
                    // unaffordable attempt that fizzled via `abort_cast`
                    // leaves a *different* item exposed, which must still
                    // get ordinary suppression treatment, not a leaked
                    // exemption).
                    let stack_top_is_fresh_own_item = match self.madness_cast_reprompt_exemption.take() {
                        Some(card) if state.stack.last().is_some_and(|item| item.source == card) => false,
                        _ => stack_top_is_fresh_own_item,
                    };

                    if matches!(state.step, Step::DeclareAttackers | Step::DeclareBlockers) {
                        if self.combat_priority_round_seen != Some(state.engine.priority_round) {
                            self.combat_priority_spent = [false, false];
                            self.combat_priority_round_seen = Some(state.engine.priority_round);
                            self.combat_priority_stack_len_seen = state.stack.len();
                            self.combat_priority_mana_count_seen = state.engine.mana_ability_activations;
                            self.combat_round_opening_mana_count = state.engine.mana_ability_activations;
                        }
                        // The mana-ability analogue of `stack_top_is_fresh_
                        // own_item`, below: a mana ability never appears on
                        // `state.stack`, so it needs its own durable "was the
                        // thing that reopened this round mine" signal --
                        // see `combat_round_opening_mana_count`'s doc.
                        let mana_ability_is_fresh_own_action = state.engine.mana_ability_activations > self.combat_round_opening_mana_count
                            && state.engine.last_mana_ability_activator == Some(*player);
                        // The acting player's own reopened window from a
                        // cast/activation that just landed on the stack this
                        // round is *always* silently suppressed here, same as
                        // the non-combat branch below -- checked *before*
                        // consulting/mutating `combat_priority_spent` so it
                        // never gets a chance to look "un-spent" for them
                        // (see `combat_priority_stack_len_seen`'s doc).
                        if stack_top_is_fresh_own_item || mana_ability_is_fresh_own_action {
                            engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
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
                            || state.engine.mana_ability_activations != self.combat_priority_mana_count_seen
                        {
                            self.combat_priority_spent = [false, false];
                            self.combat_priority_stack_len_seen = state.stack.len();
                            self.combat_priority_mana_count_seen = state.engine.mana_ability_activations;
                        }
                        if self.combat_priority_spent[player.index()] {
                            engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
                            self.record(SuppressionReason::CombatPriorityActionSpent, "Pass (forced: one action per round already taken)", before, state);
                            continue;
                        }
                        self.combat_priority_spent[player.index()] = true;
                    } else if stack_top_is_fresh_own_item {
                        engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
                        self.record(
                            SuppressionReason::StackTopIsCastersOwn,
                            "Pass (forced: caster's own cast/activation still unresolved on the stack this round)",
                            before,
                            state,
                        );
                        continue;
                    }
                    let no_real_option = castable_spells.is_empty() && mana_abilities.is_empty() && land_drops.is_empty() && activatable_abilities.is_empty();
                    let step_gated = harness_never_offers_priority(state.step);
                    if step_gated || no_real_option {
                        engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
                        self.record(if step_gated { SuppressionReason::StepGated } else { SuppressionReason::NoRealOption }, "Pass", before, state);
                        continue;
                    }
                }
                Decision::DeclareAttackers { eligible, .. } => {
                    if eligible.is_empty() {
                        engine::step(state, Action::DeclareAttackers(Vec::new())).expect("declaring zero attackers is always legal here");
                        self.record(SuppressionReason::NoEligibleAttacker, "DeclareAttackers([])", before, state);
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
                    self.madness_cast_reprompt_exemption = state.engine.pending_cast.as_ref().map(|p| p.spell);
                }
                result
            }
            SurfaceAction::Action(a) => engine::step(state, a),
            SurfaceAction::DeclareBlockersForAttacker(blockers) => {
                let reshape = self.blockers.as_mut().ok_or("no DeclareBlockersForAttacker decision is pending")?;
                let attacker = reshape.current_attacker.take().ok_or("no DeclareBlockersForAttacker decision is pending")?;
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

    fn begin_blockers_reshape(&mut self, legal_blockers: Vec<(ObjectId, Vec<ObjectId>)>, before: u64, state: &GameState) {
        let mut remaining = std::collections::VecDeque::new();
        for (attacker, blockers) in legal_blockers {
            if blockers.is_empty() {
                self.record(SuppressionReason::NoEligibleBlockersForAttacker, format!("skip attacker {attacker}"), before, state);
                continue;
            }
            remaining.push_back((attacker, blockers));
        }
        self.blockers = Some(BlockersReshape { remaining, accumulated: Vec::new(), current_attacker: None });
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
            let already_used: Vec<ObjectId> = self.blockers.as_ref()?.accumulated.iter().map(|&(b, _)| b).collect();
            let filtered: Vec<ObjectId> = legal_blockers.into_iter().filter(|b| !already_used.contains(b)).collect();
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
            return Some(SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers: filtered });
        }
    }

    fn finish_blockers_reshape(&mut self, state: &mut GameState) {
        let reshape = self.blockers.take().expect("finish_blockers_reshape requires an in-progress reshape");
        debug_assert!(reshape.remaining.is_empty(), "finish_blockers_reshape called before every attacker was asked");
        engine::step(state, Action::DeclareBlockers(reshape.accumulated)).expect("accumulated blocks were already checked legal one attacker at a time");
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
        let card_id = card_def::card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
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
        });
        state.players[player.index()].battlefield.push(obj_id);
        obj_id
    }

    #[test]
    fn provenance_consts_are_pinned() {
        assert_eq!(H2_PREDICATE_VERSION, 2);
        assert_eq!(H2_JAVA_ORACLE_COMMIT.len(), 40, "should be a full git sha");
        assert_ne!(H2_JAVA_ORACLE_COMMIT, crate::surface::H1_JAVA_ORACLE_COMMIT, "H2 must pin its own commit, not reuse H1's");
    }

    #[test]
    fn verify_corpus_provenance_passes_on_exact_match() {
        verify_corpus_provenance(H2_JAVA_ORACLE_COMMIT).expect("identical commit must pass");
    }

    #[test]
    fn verify_corpus_provenance_fails_loudly_on_mismatch() {
        let err = verify_corpus_provenance("deadbeef00000000000000000000000000000000").unwrap_err();
        assert!(err.contains("provenance mismatch"), "got: {err}");
        assert!(err.contains(H2_JAVA_ORACLE_COMMIT), "error must name the pinned commit: {err}");
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
            SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers } => {
                assert_eq!(attacker, attacker_a);
                assert_eq!(legal_blockers, vec![blocker]);
            }
            other => panic!("expected DeclareBlockersForAttacker, got {other:?}"),
        }

        surface.apply(&mut state, SurfaceAction::DeclareBlockersForAttacker(vec![blocker])).unwrap();
        assert!(state.engine.combat.blockers_declared, "the combined DeclareBlockers action should have been applied automatically");
        assert_eq!(state.engine.combat.blocked_by, vec![(attacker_a, vec![blocker])]);
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
            SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers } => {
                assert_eq!(attacker, attacker_a);
                assert_eq!(legal_blockers, vec![blocker]);
            }
            other => panic!("expected DeclareBlockersForAttacker for attacker_a, got {other:?}"),
        }
        surface.apply(&mut state, SurfaceAction::DeclareBlockersForAttacker(vec![blocker])).unwrap();

        // attacker_b's only legal blocker is already spoken for -- must be
        // silently skipped, landing straight on the post-blocks priority
        // window (or beyond, if that also has nothing real to do -- this
        // `empty_game` has empty libraries on both sides, so the engine can
        // legitimately auto-play all the way to a deck-out `GameOver` from
        // here) instead of a second (phantom) DeclareBlockersForAttacker.
        let second = surface.next_decision(&mut state);
        assert!(!matches!(&second, SurfaceDecision::DeclareBlockersForAttacker { .. }), "attacker_b must not get a second real blockers ask, got {second:?}");
        assert!(state.engine.combat.blockers_declared);
        assert_eq!(state.engine.combat.blocked_by, vec![(attacker_a, vec![blocker])], "attacker_b must end up unblocked, not double-assigned the same blocker");

        let skipped = surface
            .suppressions()
            .iter()
            .find(|s| s.reason == SuppressionReason::NoEligibleBlockersForAttacker && s.auto_action.contains(&attacker_b.to_string()));
        assert!(skipped.is_some(), "expected a NoEligibleBlockersForAttacker suppression naming attacker_b, got {:?}", surface.suppressions());
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
        assert_eq!(surface.suppressions()[0].reason, SuppressionReason::NoRealOption);
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
        });
        state.players[0].hand.push(id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(state.players[0].battlefield[0]).tapped = false;
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let decision = surface.next_decision(&mut state);
        assert!(matches!(decision, SurfaceDecision::Decision(Decision::CastSpellOrPass { .. })));
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
        });
        state.players[0].hand.push(id);
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.objects.get_mut(state.players[0].battlefield[0]).tapped = false;
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();
        let first = surface.next_decision(&mut state);
        assert!(matches!(first, SurfaceDecision::Decision(Decision::CastSpellOrPass { .. })));
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
        });

        let _second = surface.next_decision(&mut state);
        let suppressions = surface.suppressions();
        assert_eq!(suppressions[0].reason, SuppressionReason::StackTopIsCastersOwn, "got {suppressions:?}");
        assert!(suppressions[0].auto_action.starts_with("Pass"));
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
        state.objects.get_mut(state.players[0].battlefield[0]).tapped = false;

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
        });
        state.players[1].hand.push(bolt_id);
        put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(state.players[1].battlefield[0]).tapped = false;

        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers_declared = true;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Round 1: P0 spends their one action (Pass)...
        let d1 = surface.next_decision(&mut state);
        assert!(matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0), "got {d1:?}");
        surface.apply(&mut state, SurfaceAction::Action(Action::Pass)).unwrap();

        // ...then P1 spends theirs by casting Lightning Bolt at P0.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass { player, castable_spells, .. }) = &d2 else { panic!("expected CastSpellOrPass, got {d2:?}") };
        assert_eq!(*player, PlayerId::P1);
        assert!(castable_spells.contains(&bolt_id));
        surface.apply(&mut state, SurfaceAction::Action(Action::CastSpell(bolt_id))).unwrap();

        let d3 = surface.next_decision(&mut state);
        assert!(matches!(&d3, SurfaceDecision::Decision(Decision::ChooseTargets { player, .. }) if *player == PlayerId::P1), "got {d3:?}");
        surface.apply(&mut state, SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P0)))).unwrap();

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
        assert_eq!(state.stack.len(), 1, "Lightning Bolt must still be unresolved when P0 is asked");

        let suppressions = surface.suppressions();
        let last = suppressions.last().expect("P1's own reprompt must have been suppressed");
        assert_eq!(last.reason, SuppressionReason::StackTopIsCastersOwn, "got {suppressions:?}");
        assert!(!suppressions.iter().any(|s| s.reason == SuppressionReason::CombatPriorityActionSpent), "got {suppressions:?}");
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
        state.objects.get_mut(state.players[0].battlefield[0]).tapped = false;
        let p1_mountain = put_on_battlefield(&mut state, PlayerId::P1, "Mountain");
        state.objects.get_mut(p1_mountain).tapped = false;

        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers_declared = true;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV2::new();

        // Round 1: P0 spends their one action (Pass)...
        let d1 = surface.next_decision(&mut state);
        assert!(matches!(&d1, SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. }) if *player == PlayerId::P0), "got {d1:?}");
        surface.apply(&mut state, SurfaceAction::Action(Action::Pass)).unwrap();

        // ...then P1 spends theirs by tapping their Mountain for mana --
        // no cast, no stack growth at all.
        let d2 = surface.next_decision(&mut state);
        let SurfaceDecision::Decision(Decision::CastSpellOrPass { player, mana_abilities, .. }) = &d2 else { panic!("expected CastSpellOrPass, got {d2:?}") };
        assert_eq!(*player, PlayerId::P1);
        assert!(mana_abilities.contains(&p1_mountain));
        surface.apply(&mut state, SurfaceAction::Action(Action::ActivateManaAbility(p1_mountain))).unwrap();

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
        assert!(state.stack.is_empty(), "a mana ability never puts anything on the stack");

        let suppressions = surface.suppressions();
        let last = suppressions.last().expect("P1's own reprompt after their mana ability must have been suppressed");
        assert_eq!(last.reason, SuppressionReason::StackTopIsCastersOwn, "got {suppressions:?}");
        assert!(!suppressions.iter().any(|s| s.reason == SuppressionReason::CombatPriorityActionSpent), "got {suppressions:?}");
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
        surface.apply(&mut state, SurfaceAction::Action(Action::ChooseMadnessCast(true))).unwrap();

        match surface.next_decision(&mut state) {
            SurfaceDecision::Decision(Decision::ChooseTargets { player, legal_targets, .. }) => {
                assert_eq!(player, PlayerId::P0);
                assert!(legal_targets.contains(&Target::Player(PlayerId::P1)));
                surface.apply(&mut state, SurfaceAction::Action(Action::ChooseTarget(Target::Player(PlayerId::P1)))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        // The critical assertion: the caster's own reprompt after finishing
        // the madness cast must be genuine (Lava Dart + the second Mountain
        // both still offered), not a silent forced Pass.
        let reprompt = surface.next_decision(&mut state);
        match &reprompt {
            SurfaceDecision::Decision(Decision::CastSpellOrPass { player, castable_spells, .. }) => {
                assert_eq!(*player, PlayerId::P0);
                assert!(castable_spells.contains(&lava_dart), "expected Lava Dart still offered, got {reprompt:?}");
            }
            other => panic!("expected a real CastSpellOrPass reprompt for the madness caster, got {other:?}"),
        }
        assert!(
            !surface.suppressions().iter().any(|s| s.reason == SuppressionReason::StackTopIsCastersOwn),
            "the madness cast's own reprompt must not be silently suppressed, got {:?}",
            surface.suppressions()
        );
        assert_eq!(state.stack.len(), 1, "Fiery Temper must still be unresolved on the stack at the reprompt");
    }
}
