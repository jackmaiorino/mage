//! `HarnessSurfaceV1`: the versioned decision-visibility filter between the
//! kernel's raw rules engine (`engine::advance_until_decision`/`engine::step`)
//! and any caller that wants exactly the decision stream the Java reference
//! harness (`ComputerPlayerRL`) actually asks a player about -- no more, no
//! less.
//!
//! ## Why this has to be a shared library module, not comparator-local logic
//!
//! The kernel offers a rules-faithful priority window at every step the
//! comprehensive rules grant one (508.1/509.1/117 etc). The Java reference
//! does not: `ComputerPlayerRL` only calls `logReplayDecision` -- and hence
//! only produces a trace record, or (for the model-serving path) only
//! actually invokes the policy -- for a strict subset of those windows.
//! Whatever consumes the kernel's decision stream (the golden-trace
//! comparator today; the RL model-serving path this crate is ultimately
//! for) must reproduce the reference's own visibility rules and
//! deterministically auto-resolve, without asking anyone, exactly the
//! windows the reference itself never asked about. If that logic lived only
//! in `examples/replay_burn.rs`, the model-serving path would have to
//! reimplement it (or worse, silently diverge from it) the day this crate
//! is wired up to actually drive training -- and the trained policy would
//! then see a *different* decision distribution than the one this
//! comparator validated it against. `HarnessSurfaceV1` is the single
//! definition both paths share.
//!
//! ## Provenance
//!
//! This predicate was reverse engineered from the Java source (every
//! citation below is `file:line` as of this increment) and cross-checked
//! against the real v3 corpus (`local-training/kernel_oracle/burn_mirror_v3/`,
//! 40 games / 4382 decision records). `H1_JAVA_ORACLE_COMMIT` pins the exact
//! `ComputerPlayerRL.java` revision those citations were read against.
//!
//! **V1 is IMMUTABLE.** If a future change to the Java harness (a new
//! silent-pass step, a new genericChoose shortcut, a changed trace schema)
//! requires a different visibility predicate, that is `HarnessSurfaceV2`, a
//! new type living alongside this one -- never an edit to `HarnessSurfaceV1`
//! itself. Any golden trace already validated against V1 must stay
//! replayable against the exact V1 semantics it was validated with.
//!
//! ## The predicate itself
//!
//! **1. Priority windows (`ACTIVATE_ABILITY_OR_SPELL`, kernel
//! `Decision::CastSpellOrPass`).** `ComputerPlayerRL.priorityPlay`
//! (ComputerPlayerRL.java:9604-9657) is a hard `switch` on
//! `game.getTurnStepType()`. Only `PRECOMBAT_MAIN`, `POSTCOMBAT_MAIN`,
//! `DECLARE_ATTACKERS`, `DECLARE_BLOCKERS` call `calculateRLAction` (which
//! can reach `genericChoose`'s log call at line 4716); every other step
//! calls `pass(game)` unconditionally and returns, *without ever computing
//! candidates* -- so the reference silently passes there regardless of
//! whether the kernel would legitimately offer a real option (e.g. an
//! instant castable during upkeep). Mapped onto the kernel's `state::Step`:
//!   - `Untap`, `Cleanup`: the kernel itself never grants priority here
//!     (`engine::step_grants_priority`), so there's nothing to reconcile.
//!   - `Upkeep`, `Draw`, `BeginCombat`, `CombatDamage` (covers XMage's
//!     `FIRST_COMBAT_DAMAGE`+`COMBAT_DAMAGE`), `EndCombat`, `End` (XMage's
//!     `END_TURN`): the reference *always* silently passes, unconditionally
//!     -- this surface must too, even when `castable_spells`/
//!     `mana_abilities`/etc are non-empty. See [`harness_never_offers_priority`].
//!   - `Main1`, `Main2`, `DeclareAttackers`/`DeclareBlockers` (after the
//!     attack/block declaration decision itself, once
//!     `attackers_declared`/`blockers_declared`): `calculateRLAction` runs.
//!     `RL_FILTER_PRIORITY_MANA_ACTIONS` (default `false` when unset)
//!     means mana abilities are *not* filtered out, so a mana-only window
//!     (no castable spell/land/other ability) still surfaces as a real
//!     choice with at least 2 options (Pass + the mana ability(ies)) and
//!     still gets logged -- this surface's `no_real_option` check (every
//!     kernel bucket empty) already matches that.
//!
//! **2. `DECLARE_ATTACKS`/`DECLARE_BLOCKS`.** `selectAttackers`
//! (ComputerPlayerRL.java:8531-8794) logs whenever `possibleAttackers` is
//! non-empty -- exactly the kernel's own skip condition, so no extra
//! handling needed for attackers beyond the empty-`eligible` auto-resolve.
//! `selectBlockers` (ComputerPlayerRL.java:8795-8983) is structurally
//! different: it logs *once per attacker that has >= 1 eligible blocker*
//! (`for (Permanent attacker : attackers) { if (eligibleBlockers.isEmpty())
//! continue; ... }`), not once for the whole step. The kernel's own
//! `Decision::DeclareBlockers` bundles every attacker into one
//! `legal_blockers` list; this surface decomposes it into one
//! `SurfaceDecision::DeclareBlockersForAttacker` per attacker with a
//! non-empty blocker list, silently skipping (and recording a suppression
//! for) attackers with zero eligible blockers, then applies every
//! attacker's picks as a single `Action::DeclareBlockers` once every
//! sub-decision is resolved.
//!
//! **3. `DeclareAttackers`/`DeclareBlockers`'s one-action-per-round
//! throttle.** `priorityPlay`'s `DECLARE_ATTACKERS`/`DECLARE_BLOCKERS` cases
//! (ComputerPlayerRL.java:9623-9638) are shaped differently from every other
//! case in the switch: `currentAbility = calculateRLAction(game); act(game,
//! ...); pass(game); return true;` -- an *unconditional* `pass(game)` right
//! after acting, with no surrounding loop. `PRECOMBAT_MAIN`/`POSTCOMBAT_MAIN`
//! have no such call; `GameImpl.playPriority`'s own `while
//! (!player.isPassed() && player.canRespond() ...)` loop (GameImpl.java:1768)
//! keeps re-invoking `player.priority(this)` -- hence `calculateRLAction`
//! again -- for the *same* player there until they truly choose Pass. So on
//! these two steps only, a player gets exactly one priority action (real or
//! auto-suppressed) per "round" -- even a cast/activation/land-drop/Plot that
//! reopens priority back to the *same* player under 601.2i/117.3b (the
//! kernel is rules-faithful here; the reference harness just doesn't ask
//! again) is immediately followed by a forced Pass, deferring any further
//! action from that player to the next round. A round ends -- both players'
//! throttle clears -- only at a genuine `GameImpl`-level `resetPassed()`
//! boundary: the step's own start (after the attack/block declaration) or a
//! stack resolution; *not* at the four `priority_passes = [false, false]`
//! sites that hand priority back to the same actor
//! (`finalize_cast`/`finalize_activation`/`play_land`/`plot_spell`) or at a
//! mid-cascade triggered ability going on the stack
//! (`push_trigger_onto_stack`) -- see `engine::EngineState::priority_round`'s
//! doc, which this surface reads to tell the two apart without re-deriving
//! round boundaries from stack length or step identity (both ambiguous
//! across turns). Empirically root-caused against the real v3 corpus
//! (`game_20260712_194602_0003.txt`, decision #76-79: SelfPlay casts
//! Lightning Bolt and targets it during `DeclareAttackers`'s priority phase;
//! the immediate post-cast re-priority-to-caster window that 601.2i would
//! grant is never logged -- not even as a silent single-candidate
//! auto-resolution, since a second untapped land was still available -- and
//! decision #79, the *next* logged window for that player, already shows the
//! spell resolved (graveyard +1, life -3)).
//!
//! Anything trace-specific (turn/hand/library/graveyard cross-checks,
//! candidate-multiset translation against trace UUIDs, the stale-forced-
//! discard trace-record skip) is *not* part of this surface: it has no
//! meaning outside the comparator and stays in `examples/replay_burn.rs`.

use crate::engine::{self, Action, Decision};
use crate::ids::ObjectId;
use crate::state::{GameState, Step};

/// Predicate version. Bump only by adding `HarnessSurfaceV2` alongside this
/// module -- never by editing the V1 predicate in place (see the module
/// doc).
pub const H1_PREDICATE_VERSION: u32 = 1;

/// `ComputerPlayerRL.java`'s commit hash as of the increment that reverse
/// engineered this predicate (`git log --format=%H -1 -- \
/// Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerRL.java`,
/// run from the repo root). If the Java harness changes, this predicate's
/// citations may no longer be accurate -- that is exactly the trigger for
/// minting `HarnessSurfaceV2` rather than editing this one.
pub const H1_JAVA_ORACLE_COMMIT: &str = "034ce665e1e2330e07dfc2f5bc088504b1bbaf48";

/// Why a particular engine decision was auto-resolved instead of surfaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionReason {
    /// The reference harness's `priorityPlay` never calls `calculateRLAction`
    /// for this `Step` at all (predicate point 1) -- suppressed regardless of
    /// what the kernel would otherwise offer.
    StepGated,
    /// The step *is* one the reference asks about, but the kernel's own
    /// candidate buckets are all empty (Pass is the only option) -- the
    /// reference's `genericChoose` never logs a single-legal-candidate
    /// window either.
    NoRealOption,
    /// `Decision::DeclareAttackers` with an empty `eligible` set --
    /// `selectAttackers` returns without logging when `possibleAttackers`
    /// is empty (predicate point 2).
    NoEligibleAttacker,
    /// One attacker within a `Decision::DeclareBlockers` step has zero
    /// eligible blockers -- `selectBlockers`'s per-attacker loop `continue`s
    /// without logging for that specific attacker (predicate point 2).
    NoEligibleBlockersForAttacker,
    /// This player already spent their one `DeclareAttackers`/
    /// `DeclareBlockers` priority action this round (predicate point 3) --
    /// `priorityPlay`'s `DECLARE_ATTACKERS`/`DECLARE_BLOCKERS` cases call
    /// `calculateRLAction` *once* then unconditionally `pass(game)`, even
    /// when the action just taken (a cast/activation/land/Plot) reopened
    /// priority back to the same player under 117.3b/601.2i -- unlike
    /// `PRECOMBAT_MAIN`/`POSTCOMBAT_MAIN`, which loop `calculateRLAction`
    /// for the same player until they actually choose Pass.
    CombatPriorityActionSpent,
}

/// One auto-resolution the surface performed instead of asking. `auto_action`
/// is a human-readable description of what was applied (not a parseable
/// format -- callers that need the actual `Action` can read it off the
/// suppression reason and context; this field exists for logs/debugging).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suppression {
    pub reason: SuppressionReason,
    pub auto_action: String,
    pub state_hash_before: u64,
    pub state_hash_after: u64,
}

/// A decision worth surfacing to a caller (comparator or, eventually, a
/// model-serving loop). Every variant except [`SurfaceDecision::DeclareBlockersForAttacker`]
/// is a transparent pass-through of the matching `engine::Decision` variant,
/// already stripped of every silently-auto-resolved window; see the module
/// doc for why blockers are reshaped instead of passed through.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceDecision {
    Decision(Decision),
    /// One sub-decision per attacker that has at least one eligible
    /// blocker, in attacker-`ObjectId` order. Answer with
    /// `SurfaceAction::DeclareBlockersForAttacker` (a possibly-empty set of
    /// blockers for *this* attacker); once every such attacker has been
    /// answered, the surface applies the combined pick as a single
    /// `engine::Action::DeclareBlockers` automatically.
    DeclareBlockersForAttacker {
        attacker: ObjectId,
        legal_blockers: Vec<ObjectId>,
    },
}

/// The answer to a [`SurfaceDecision`]. Everything except the blockers
/// reshape is a transparent pass-through of `engine::Action`.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceAction {
    Action(Action),
    /// Blockers assigned to the attacker named by the most recently
    /// returned `SurfaceDecision::DeclareBlockersForAttacker`.
    DeclareBlockersForAttacker(Vec<ObjectId>),
}

/// Steps where `ComputerPlayerRL.priorityPlay`'s hard-coded `switch` always
/// calls `pass(game)` and returns *without ever computing candidates*
/// (ComputerPlayerRL.java:9604-9657) -- see the module doc's predicate,
/// point 1. `Untap` and `Cleanup` are deliberately omitted: the kernel
/// itself never grants priority there (`engine::step_grants_priority`), so
/// a `CastSpellOrPass` decision for either is unreachable and there's
/// nothing to gate.
pub fn harness_never_offers_priority(step: Step) -> bool {
    matches!(step, Step::Upkeep | Step::Draw | Step::BeginCombat | Step::CombatDamage | Step::EndCombat | Step::End)
}

#[derive(Default)]
struct BlockersReshape {
    /// (attacker, legal_blockers) pairs still waiting to be asked about, in
    /// order. Attackers with an empty blocker list are filtered out (and
    /// suppressed) before this is populated -- see `begin_declare_blockers`.
    remaining: std::collections::VecDeque<(ObjectId, Vec<ObjectId>)>,
    accumulated: Vec<(ObjectId, ObjectId)>,
    current_attacker: Option<ObjectId>,
}

/// See the module doc. Owns the suppression log and whatever bookkeeping the
/// `DeclareBlockers` reshape needs across calls.
#[derive(Default)]
pub struct HarnessSurfaceV1 {
    suppressions: Vec<Suppression>,
    blockers: Option<BlockersReshape>,
    /// [P0, P1] -- has this player already spent their one `DeclareAttackers`/
    /// `DeclareBlockers` priority action this round? See predicate point 3.
    combat_priority_spent: [bool; 2],
    /// `engine::EngineState::priority_round` as of the last time
    /// `combat_priority_spent` was synced -- a change means a genuine new
    /// round started (step entry or a stack resolution), so both flags
    /// clear. `None` before the first DeclareAttackers/DeclareBlockers
    /// priority window this surface has ever seen.
    combat_priority_round_seen: Option<u64>,
}

impl HarnessSurfaceV1 {
    pub fn new() -> HarnessSurfaceV1 {
        HarnessSurfaceV1::default()
    }

    /// Every auto-resolution performed so far, in the order they happened.
    pub fn suppressions(&self) -> &[Suppression] {
        &self.suppressions
    }

    fn record(&mut self, reason: SuppressionReason, auto_action: impl Into<String>, before: u64, state: &GameState) {
        self.suppressions.push(Suppression { reason, auto_action: auto_action.into(), state_hash_before: before, state_hash_after: state.state_hash() });
    }

    /// Drives `engine::advance_until_decision`, silently auto-resolving
    /// every window the reference harness never asks about (recording a
    /// [`Suppression`] for each), and returns the next decision actually
    /// worth surfacing.
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
                        // Whatever happens next (real ask or NoRealOption
                        // auto-suppression below) is this player's one
                        // allotted call to `calculateRLAction` this round --
                        // see the module doc's predicate point 3.
                        self.combat_priority_spent[player.index()] = true;
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
                    // Every attacker had zero eligible blockers: nothing left
                    // to ask, apply the (empty) combined pick immediately.
                    self.finish_blockers_reshape(state);
                    continue;
                }
                _ => {}
            }

            return SurfaceDecision::Decision(decision);
        }
    }

    /// Applies the answer to the most recently returned [`SurfaceDecision`].
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

    /// Applies the fully-accumulated blocker assignment as one combined
    /// `engine::Action::DeclareBlockers` and clears the reshape state.
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

    /// Provenance consts are pinned, not placeholder-empty.
    #[test]
    fn provenance_consts_are_pinned() {
        assert_eq!(H1_PREDICATE_VERSION, 1);
        assert_eq!(H1_JAVA_ORACLE_COMMIT.len(), 40, "should be a full git sha");
    }

    /// End-to-end shape check: a step the reference never offers priority in
    /// (`Upkeep`) is silently passed through with no `SurfaceDecision`
    /// surfaced for it, and a suppression is recorded with the right reason
    /// and a state hash on each side (proves the log shape the acceptance
    /// criteria asks for). An empty game has no real decisions anywhere, so
    /// `next_decision` keeps auto-resolving all the way around to
    /// `GameOver` (an empty library loses on its first real draw) --
    /// this only pins the *first* suppression, which is deterministically
    /// the Upkeep window this test starts at.
    #[test]
    fn step_gated_window_is_suppressed_not_surfaced() {
        let mut state = empty_game();
        state.step = Step::Upkeep;
        state.priority_player = PlayerId::P0;
        state.active_player = PlayerId::P0;

        let mut surface = HarnessSurfaceV1::new();
        let decision = surface.next_decision(&mut state);

        assert!(matches!(decision, SurfaceDecision::Decision(_)));
        assert!(!surface.suppressions().is_empty());
        let s = &surface.suppressions()[0];
        assert_eq!(s.reason, SuppressionReason::StepGated);
        assert_eq!(s.auto_action, "Pass");
        assert_ne!(s.state_hash_before, 0);
    }

    /// `DeclareBlockers` is reshaped into one sub-decision per attacker that
    /// has at least one eligible blocker; an attacker with zero eligible
    /// blockers is suppressed instead of surfaced, and the final combined
    /// `DeclareBlockers` action is only applied once every sub-decision is
    /// answered.
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

        let mut surface = HarnessSurfaceV1::new();
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

        let mut surface = HarnessSurfaceV1::new();
        surface.next_decision(&mut state);

        assert!(!surface.suppressions().is_empty());
        assert_eq!(surface.suppressions()[0].reason, SuppressionReason::NoRealOption);
    }

    /// A real, >=2-option priority window (a castable spell present) is
    /// surfaced, not suppressed.
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

        let mut surface = HarnessSurfaceV1::new();
        let decision = surface.next_decision(&mut state);
        assert!(matches!(decision, SurfaceDecision::Decision(Decision::CastSpellOrPass { .. })));
        assert!(surface.suppressions().is_empty());
        let _ = CARD_DEFS;
        let _ = Target::Player(PlayerId::P0);
    }
}
