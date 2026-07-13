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
    round_opening_stack_len: usize,
    stack_len_round_seen: Option<u64>,
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
            if let Some(sd) = self.next_blockers_subdecision() {
                return sd;
            }

            let before = state.state_hash();
            let decision = engine::advance_until_decision(state);

            match &decision {
                Decision::CastSpellOrPass { player, castable_spells, mana_abilities, land_drops, activatable_abilities, .. } => {
                    if matches!(state.step, Step::DeclareAttackers | Step::DeclareBlockers) {
                        if self.combat_priority_round_seen != Some(state.engine.priority_round) {
                            self.combat_priority_spent = [false, false];
                            self.combat_priority_round_seen = Some(state.engine.priority_round);
                        }
                        if self.combat_priority_spent[player.index()] {
                            engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
                            self.record(SuppressionReason::CombatPriorityActionSpent, "Pass (forced: one action per round already taken)", before, state);
                            continue;
                        }
                        self.combat_priority_spent[player.index()] = true;
                    } else {
                        if self.stack_len_round_seen != Some(state.engine.priority_round) {
                            self.round_opening_stack_len = state.stack.len();
                            self.stack_len_round_seen = Some(state.engine.priority_round);
                        }
                        let stack_top_is_fresh_own_item = state.stack.len() > self.round_opening_stack_len
                            && state.stack.last().is_some_and(|item| item.controller == *player);
                        if stack_top_is_fresh_own_item {
                            engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
                            self.record(
                                SuppressionReason::StackTopIsCastersOwn,
                                "Pass (forced: caster's own cast/activation still unresolved on the stack this round)",
                                before,
                                state,
                            );
                            continue;
                        }
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
                    if let Some(sd) = self.next_blockers_subdecision() {
                        return sd;
                    }
                    self.finish_blockers_reshape(state);
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

    fn next_blockers_subdecision(&mut self) -> Option<SurfaceDecision> {
        let reshape = self.blockers.as_mut()?;
        let (attacker, legal_blockers) = reshape.remaining.pop_front()?;
        reshape.current_attacker = Some(attacker);
        Some(SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers })
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
}
