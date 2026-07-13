//! Priority / stack / state-based-action turn engine.
//!
//! Two entry points, both fully general -- no "no response expected"
//! shortcuts and no auto-pass:
//!
//! - [`advance_until_decision`] drives the turn structure (the 12-step
//!   sequence in `state::Step`), runs SBAs/trigger-collection after every
//!   resolution, and returns the next real [`Decision`] a player must make.
//! - [`step`] applies a chosen [`Action`] in response to the last
//!   `Decision`, mutating `GameState`.
//!
//! All transient engine state that needs to survive between an
//! `advance_until_decision` call and the matching `step` call (whose spell
//! is mid-cast, who has already passed this priority round, the event log,
//! active replacements, queued triggers, in-progress combat) lives in
//! `GameState::engine` ([`EngineState`]) so both functions can stay pure
//! `&mut GameState` signatures with no separate engine object.
//!
//! Card-effect mutation still only ever happens through
//! `event::propose_and_commit`/`propose_and_commit_batch` (see
//! `effect.rs`/`event.rs`); everything in this module that mutates
//! `GameState` directly is turn/stack/priority/cost-payment bookkeeping,
//! not card behavior.
//!
//! ## Multi-stage decisions (casting, activating, discarding)
//!
//! Casting a spell, activating a non-mana ability, and discarding cards
//! are each potentially multi-step processes with real decision points in
//! the middle (choose targets, choose a cast mode, choose which cards to
//! discard). Each has a `Pending*` struct parked in `EngineState` that
//! `advance_until_decision`'s loop re-examines every pass: if a stage
//! isn't resolved yet, it returns the `Decision` for that stage (or, if
//! there's only one legal answer, resolves it automatically and
//! `continue`s -- same "don't ask when there's no real choice" pattern
//! increment 2 established for `OrderTriggers` singleton groups). Once
//! every stage is resolved, `finalize_cast`/`finalize_activation` pays the
//! remaining (non-discard) costs and pushes the stack item -- except a
//! cast's stack item: 601.2a puts a spell on the stack at *announcement*
//! (`begin_cast`), before targets/modes/costs, so `finalize_cast` only
//! fills in the placeholder `begin_cast` already pushed.

use crate::card_def::{self, CardType, CostComponent, FlashbackCost, Keywords, TargetSpec};
use crate::effect::{self, EffectOp, ExecCtx, ObjectRef};
use crate::event::{self, ActiveReplacement, CommittedEvent, ProposedEvent};
use crate::ids::{ObjectId, PlayerId};
use crate::mana::{self, Cost};
use crate::state::{GameState, Step, StackItem, Target, Zone};
use crate::trigger::{self, PendingTrigger};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngineState {
    /// Whether [P0, P1] has passed priority since the last time priority
    /// was reset (new step, a cast/activation/land-drop, or a resolution).
    pub priority_passes: [bool; 2],
    /// Bumped by `reset_priority` alone -- i.e. only on the two rules-level
    /// "everyone's pass streak clears and the active player is asked
    /// first" boundaries (a new step via `advance_step`, or a stack
    /// resolution) and the declare-attackers/declare-blockers transition
    /// into that step's own priority phase (`apply_declare_attackers`/
    /// `apply_declare_blockers`, which route through the same helper).
    /// Deliberately *not* bumped by the other four `priority_passes =
    /// [false, false]` sites (`finalize_cast`, `finalize_activation`,
    /// `play_land`, `plot_spell`, `push_trigger_onto_stack`): those hand
    /// priority back to the *same* player who just acted (601.2i/117.3b),
    /// which is a real fresh priority window under the comprehensive rules
    /// but not a new "round" in the sense this counter tracks. Exists
    /// purely so `mtg_kernel::surface::HarnessSurfaceV1` can detect the
    /// DeclareAttackers/DeclareBlockers one-action-per-round throttle
    /// (`ComputerPlayerRL.priorityPlay`'s hard-coded `act(); pass();` for
    /// those two steps) without re-deriving round boundaries from stack
    /// length or step identity, both of which are ambiguous across turns.
    pub priority_round: u64,
    /// A spell that has been announced (601.2a: already moved to the stack
    /// by `begin_cast`) but not yet finished being targeted, mode-chosen,
    /// or cost-paid.
    pub pending_cast: Option<PendingCast>,
    /// A non-mana ability that has begun activating but isn't fully
    /// targeted/cost-paid yet (Masked Meower's, the Blood token's).
    pub pending_activation: Option<PendingActivation>,
    /// A discard obligation -- from a cast/activation's cost, from a
    /// resolving effect (`EffectOp::DiscardCards`), or from cleanup --
    /// waiting on `Decision::Discard`/`Action::Discard`. See `card_def`'s
    /// `CostComponent::DiscardCards` and the `EffectOp::DiscardCards` doc
    /// for why this needs its own pending-state slot instead of being
    /// solved synchronously like every other cost/effect leaf.
    pub pending_discard: Option<PendingDiscard>,
    /// A resolution-time optional cost (Highway Robbery's "you may discard
    /// a card or sacrifice a land") staged by `EffectOp::MayPayCostThen`,
    /// waiting on `Decision::ChooseOptionalCost`/`Action::ChooseOptionalCost`.
    pub pending_optional_cost: Option<PendingOptionalCost>,
    /// Once `Action::ChooseOptionalCost(SacrificeLand)` is chosen, *which*
    /// land(s) to sacrifice -- a real decision when more than one is legal,
    /// same `Decision::ChooseCostTargets`/`ChooseCastMode`-mode-Alternative
    /// shape as Fireblast's alt cost and Lava Dart's flashback cost (see
    /// `sacrifice_lands_needed`'s doc and `drain_pending_optional_cost_
    /// sacrifice_or_decide`). Previously auto-solved silently by
    /// `sacrifice_lowest_id_lands` with no `Decision` at all -- same bug
    /// class as the other two, root-caused the same way (Sol #90,
    /// increment 11).
    pub pending_optional_cost_sacrifice: Option<PendingOptionalCostSacrifice>,
    /// Cards exiled by Madness (`card_def::CardDef::madness_cost`) since
    /// their controlling discard, in discard order, each waiting on
    /// `Decision::ChooseMadnessCast`/`Action::ChooseMadnessCast`. A `Vec`
    /// (not a single `Option`) because a single `Decision::Discard` can
    /// discard more than one Madness card at once in general, even though
    /// no card pair in this pool can -- `drain_pending_madness_or_decide`
    /// processes it front-to-back, one real decision (or silent
    /// auto-resolve-to-graveyard, if unaffordable) at a time.
    pub pending_madness: Vec<ObjectId>,
    /// Transient buffer for the *current* resolution: `event::commit`
    /// appends here, `trigger::collect_and_process` drains it after every
    /// resolution to match triggers. Empty between resolutions.
    pub event_log: Vec<CommittedEvent>,
    /// Full permanent record of every committed event this game, in
    /// commit order. Never drained; this is what
    /// `event::commit`/`trigger::collect_and_process`'s draining of
    /// `event_log` would otherwise make unobservable after the fact (game
    /// replay / RL trace logging / the acceptance tests' event-log
    /// assertions all read this instead).
    pub event_history: Vec<CommittedEvent>,
    pub active_replacements: Vec<ActiveReplacement>,
    pub next_replacement_id: u32,
    /// Triggered abilities collected but not yet placed on the stack,
    /// grouped APNAP (active player's group first); see
    /// `drain_pending_triggers_or_decide`.
    pub pending_triggers: Vec<PendingTrigger>,
    /// This turn's combat, reset fresh at every `Step::BeginCombat`.
    pub combat: CombatState,
    /// "Until end of turn" continuous effects, cleared at every
    /// `Step::Cleanup` (514.2). No card in this increment's pool creates
    /// one; this is a proven-but-unused shape, same role
    /// `event::ReplacementEffectKind::PreventNextDamage` plays for the
    /// replacement pipeline -- see
    /// `tests::until_end_of_turn_effects_are_cleared_at_cleanup`.
    pub until_end_of_turn: Vec<UntilEndOfTurnEffect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UntilEndOfTurnEffect {
    /// Synthetic placeholder -- see the `until_end_of_turn` field doc.
    SyntheticMarker(ObjectId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastMode {
    /// Pay the card's printed mana cost.
    Normal,
    /// Pay its `CardDef::alt_cost` instead (Fireblast: sacrifice 2 Mountains).
    Alternative,
}

/// Which cost `Decision::ChooseCostTargets` is picking permanents for.
/// One variant today (this pool's only "choose which permanents" cost
/// shape); kept as its own type rather than inlined so a future cost kind
/// (e.g. a generic sacrifice-a-creature cost) doesn't have to overload
/// this one's meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostKind {
    SacrificeLands,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingCast {
    pub spell: ObjectId,
    pub controller: PlayerId,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    /// True iff `spell` is being cast from the graveyard via flashback
    /// (`CardDef::flashback`), which uses a wholly different cost (never
    /// the printed mana cost) and exiles instead of going to the
    /// graveyard on resolution.
    pub is_flashback: bool,
    /// `None` until resolved. Pre-seeded to `Some(CastMode::Normal)` at
    /// `begin_cast` for every card without an `alt_cost` (i.e. almost
    /// everything), so the "which mode" decision stage is skipped
    /// entirely unless the card is Fireblast.
    pub cast_mode: Option<CastMode>,
    /// `None` until this cast's mandatory additional cost (if any) has
    /// been paid; pre-seeded to `Some(vec![])` at `begin_cast` for every
    /// card without an `additional_cost`. Holds whichever cards were
    /// discarded to pay it (Grab the Prize), read by
    /// `EffectCond::DiscardedNonLandForCost` via `ExecCtx::discarded`.
    pub additional_cost_discarded: Option<Vec<ObjectId>>,
    /// `Some(cost)` overrides the normal/alt-cost/flashback payment branch
    /// in `finalize_cast` entirely, paying exactly `cost` instead: `Cost::
    /// zero()` for casting a Plotted card for free (`begin_cast`, `zone ==
    /// Exile`), or a card's `madness_cost` for a Madness cast
    /// (`apply_choose_madness_cast`). `None` for every ordinary cast.
    /// `#[serde(skip)]`: `mana::Cost` can't derive `Deserialize` (see its
    /// doc) -- harmless since nothing serializes mid-cast engine state
    /// today; a real cast is entirely synchronous within one `step` call.
    #[serde(skip)]
    pub cost_override: Option<Cost>,
    /// `None` until resolved, only meaningful for a modal card
    /// (`CardDef::mode2.is_some()`) -- pre-seeded to `Some(0)` at
    /// `begin_cast` for every non-modal card, so the "which mode" decision
    /// stage is skipped entirely unless the card is Pyroblast/Red Elemental
    /// Blast. `0` = the card's primary `target_spec`/`spell_effect`, `1` =
    /// `mode2`.
    pub mode_chosen: Option<u8>,
    /// Which zone this cast was announced from (Hand, Graveyard for
    /// flashback, or Exile for Plot/Madness) -- `begin_cast` captures the
    /// spell's zone *before* `move_to_stack` changes it, purely so
    /// `abort_cast` (unreachable this increment, but kept in shape rather
    /// than papered over) knows where to return the card if its cost turns
    /// out to be unpayable.
    pub origin_zone: Zone,
    /// Which lands have been chosen so far to pay a `SacrificeLands`
    /// sub-cost of this cast (Fireblast's alt cost, once `cast_mode`
    /// resolves to `Alternative`; Lava Dart's flashback cost,
    /// unconditionally) -- see `sacrifice_lands_needed`/
    /// `Decision::ChooseCostTargets`. Always empty for a cast that doesn't
    /// need one (`sacrifice_lands_needed` returns 0), same "just stays at
    /// its zero value, never consulted" shape `additional_cost_discarded`
    /// has for a card with no additional cost.
    pub sacrifice_chosen: Vec<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingActivation {
    pub source: ObjectId,
    pub controller: PlayerId,
    pub ability_index: u8,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    /// Same shape/rationale as `PendingCast::additional_cost_discarded`.
    pub cost_discard_paid: Option<Vec<ObjectId>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscardResume {
    /// Nothing further to do once the discard lands (cleanup's
    /// discard-to-7).
    None,
    /// Write the discarded cards back into `EngineState::pending_cast`'s
    /// `additional_cost_discarded` and let the cast staging continue.
    FinishCast,
    /// Same, but for `EngineState::pending_activation`'s
    /// `cost_discard_paid`.
    FinishActivation,
    /// A resolving instant/sorcery's own effect discarded as part of its
    /// resolution (Faithless Looting: "draw two, then discard two").
    /// `EffectOp::DiscardCards` stages `pending_discard` and returns
    /// *synchronously*, before the discard is actually answered (see that
    /// leaf's doc) -- so by the time `execute` returns to
    /// `resolve_top_of_stack`, the discard hasn't happened yet. 608.2m: the
    /// spell can only move to its post-resolution zone as the *last* part
    /// of its resolution, which isn't done until this discard lands -- so
    /// `resolve_top_of_stack` defers that move here instead of doing it
    /// immediately, and `apply_discard` performs it once the discard is
    /// actually resolved.
    FinishSpellResolution { source: ObjectId, to_zone: Zone },
    /// The discard sub-cost of a resolution-time optional cost (Highway
    /// Robbery's `EffectOp::MayPayCostThen` -> `Action::ChooseOptionalCost
    /// (OptionalCostChoice::Discard)`) just landed -- run `then` now.
    /// `spell_resume`, if `Some`, is `PendingOptionalCost::spell_resume`
    /// carried over -- the deferred move to apply once `then` runs (see
    /// that field's doc).
    FinishOptionalCost { source: ObjectId, controller: PlayerId, then: Box<EffectOp>, spell_resume: Option<(ObjectId, Zone)> },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingDiscard {
    pub player: PlayerId,
    pub count: u32,
    pub resume: DiscardResume,
}

/// A resolution-time optional cost (`effect::EffectOp::MayPayCostThen`),
/// waiting on `Decision::ChooseOptionalCost`. `discard_payable`/
/// `sacrifice_payable` are snapshotted at stage time (by `execute`, which
/// already checked at least one is true before staging this at all) so
/// `Action::ChooseOptionalCost` can validate the chosen option without
/// recomputing legality against however state has (not) changed since.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCost {
    pub player: PlayerId,
    pub source: ObjectId,
    pub discard: u8,
    pub sacrifice_lands: u8,
    pub discard_payable: bool,
    pub sacrifice_payable: bool,
    pub then: EffectOp,
    /// `Some((source, to_zone))` iff this optional cost is itself part of
    /// `source`'s own spell resolution (Highway Robbery's "you may... if
    /// you do, draw two cards" -- `EffectOp::MayPayCostThen` staged this
    /// synchronously, from inside `resolve_top_of_stack`'s own `execute`
    /// call) and that spell hasn't moved to `to_zone` (its post-resolution
    /// zone) yet. See `resolve_top_of_stack`'s doc for why this can't just
    /// move there immediately the way it does for every other instant/
    /// sorcery -- same 608.2m "moves to its zone only as the very last
    /// part of resolution" deferral `DiscardResume::FinishSpellResolution`
    /// already handles for `EffectOp::DiscardCards`, just at the optional-
    /// cost layer instead. `None` for the normal case: an optional cost
    /// staged by something *other* than a spell's own top-level resolution
    /// has nothing waiting on it (no card in this pool triggers
    /// `MayPayCostThen` from anywhere but a spell resolving, but this
    /// isn't assumed -- `resolve_top_of_stack` is the only place that ever
    /// sets it `Some`).
    pub spell_resume: Option<(ObjectId, Zone)>,
}

/// Staged once `Action::ChooseOptionalCost(SacrificeLand)` is chosen --
/// see `EngineState::pending_optional_cost_sacrifice`'s doc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCostSacrifice {
    pub player: PlayerId,
    pub source: ObjectId,
    pub remaining: u8,
    pub chosen: Vec<ObjectId>,
    pub then: EffectOp,
    /// Carried over from `PendingOptionalCost::spell_resume` -- see that
    /// field's doc.
    pub spell_resume: Option<(ObjectId, Zone)>,
}

/// The answer to a `Decision::ChooseOptionalCost`. Declining is always
/// legal (matches `DoIfCostPaid`'s optional "may" framing); the other two
/// are only legal when the matching `PendingOptionalCost` field is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionalCostChoice {
    Decline,
    Discard,
    SacrificeLand,
}

/// This turn's combat. Reset at every `Step::BeginCombat`. An attacker
/// with no entry in `blocked_by` is unblocked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CombatState {
    pub attackers_declared: bool,
    pub blockers_declared: bool,
    pub attackers: Vec<ObjectId>,
    /// Attacker -> blockers, in the attacking player's damage-assignment
    /// order (509.2). This increment always sorts by `ObjectId` --
    /// deterministic, but a stand-in for a real
    /// `Decision::AssignDamageOrder` a future increment can slot in here
    /// without changing `blocked_by`'s shape.
    pub blocked_by: Vec<(ObjectId, Vec<ObjectId>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    CastSpellOrPass {
        player: PlayerId,
        /// Hand-castable spells, graveyard flashback-castable cards
        /// (`CardDef::flashback`), and exiled Plotted cards castable for
        /// free (`CardDef::plot_cost`, cast on a later turn); `step()`
        /// tells them apart by the object's current zone.
        castable_spells: Vec<ObjectId>,
        mana_abilities: Vec<ObjectId>,
        land_drops: Vec<ObjectId>,
        /// (source, ability_index) pairs for every non-mana activated
        /// ability currently affordable (Masked Meower's, the Blood
        /// token's).
        activatable_abilities: Vec<(ObjectId, u8)>,
        /// Hand cards with `CardDef::plot_cost` currently affordable at
        /// sorcery speed (Highway Robbery's `PlotAbility`, a `SpecialAction`
        /// -- see `Action::PlotSpell`). Disjoint from `castable_spells`:
        /// Plotting is a separate action from casting, offered alongside it
        /// in the same priority window (the real trace's "Plot {1}{R}"
        /// candidate, distinct from "Cast Highway Robbery").
        plot_actions: Vec<ObjectId>,
    },
    ChooseTargets {
        player: PlayerId,
        /// The spell or (non-mana-ability) source this targeting belongs
        /// to.
        spell: ObjectId,
        remaining: u8,
        legal_targets: Vec<Target>,
    },
    /// A cost component whose payment requires choosing WHICH permanents
    /// pay it, not merely how many -- Fireblast's alt cost (sacrifice 2
    /// Mountains, `CostComponent::SacrificeLands`) and Lava Dart's
    /// flashback cost (sacrifice 1 Mountain, `FlashbackCost::
    /// SacrificeLands`), this increment. Previously auto-solved silently
    /// by `sacrifice_lowest_id_lands`'s tapped-status heuristic with no
    /// `Decision` at all; the reference logs a real `SELECT_TARGETS`
    /// record for this pick (increment 11 characterization), so it's a
    /// real decision here too. Asked one permanent at a time (`remaining`
    /// counts down, `candidates` excludes whatever was already picked for
    /// this same cost) -- same shape as `ChooseTargets`'s `remaining`/
    /// `legal_targets`. Auto-resolved (no `Decision` returned, every
    /// remaining candidate silently sacrificed) whenever `candidates.len()
    /// <= remaining` -- no real choice left -- matching every other "don't
    /// ask when there's one legal answer" shortcut in this module
    /// (`ChooseCastMode`, `Discard`).
    ChooseCostTargets {
        player: PlayerId,
        /// The spell (cast or flashback-cast) this cost belongs to.
        source: ObjectId,
        cost_kind: CostKind,
        remaining: u8,
        candidates: Vec<ObjectId>,
    },
    /// Fireblast only, this increment: whether to pay its printed mana
    /// cost or sacrifice 2 Mountains instead. Only asked when *both* are
    /// currently legal (601.2b) -- if just one is affordable, that one is
    /// silently used, no decision.
    ChooseCastMode {
        player: PlayerId,
        spell: ObjectId,
        options: Vec<CastMode>,
    },
    /// Pyroblast/Red Elemental Blast only, this increment: which of the
    /// spell's 2 modes to resolve (`CardDef::mode2`). Asked before
    /// targeting, since the two modes have different `TargetSpec`s. Unlike
    /// `ChooseCastMode`, always asked when `mode2.is_some()` -- both modes
    /// are always legal to *choose* regardless of what's currently on the
    /// battlefield/stack (601.2b: mode is chosen before targets, so
    /// there's no "only one is affordable" shortcut to take here).
    ChooseSpellMode {
        player: PlayerId,
        spell: ObjectId,
        mode_count: u8,
    },
    /// Highway Robbery only, this increment: a resolution-time optional
    /// cost (`effect::EffectOp::MayPayCostThen`). Always a real choice with
    /// at least 2 options (`Decline` plus whichever of `Discard`/
    /// `SacrificeLand` `PendingOptionalCost` marked payable) -- declining
    /// is always legal, so this is never auto-resolved for "no real
    /// option" the way `CastSpellOrPass` is.
    ChooseOptionalCost {
        player: PlayerId,
        discard_payable: bool,
        sacrifice_payable: bool,
    },
    /// Fiery Temper only, this increment: whether to cast a just-discarded
    /// Madness card for its madness cost (`CardDef::madness_cost`) instead
    /// of letting it go to the graveyard. Only asked when the madness cost
    /// is currently affordable -- see `drain_pending_madness_or_decide`
    /// (same "don't ask when there's only one legal answer" shortcut
    /// `ChooseCastMode` uses for Fireblast).
    ChooseMadnessCast {
        player: PlayerId,
        card: ObjectId,
    },
    /// Choose exactly `count` cards from `choices` to discard. Backs
    /// cleanup's discard-to-7, Faithless Looting's "draw two, then
    /// discard two", and the discard costs of Grab the Prize / Masked
    /// Meower / the Blood token. Only asked when `choices.len() >
    /// count` (otherwise there's no real choice -- discard everything,
    /// silently).
    Discard {
        player: PlayerId,
        count: u32,
        choices: Vec<ObjectId>,
    },
    /// 508.1: choose a (possibly empty) subset of `eligible` to attack
    /// with. Always asked whenever `Step::DeclareAttackers` is reached --
    /// `eligible` itself can be empty (no creature able to attack), and
    /// this is still asked rather than auto-passed: 508.1 makes Declare
    /// Attackers a turn-based action that always happens, even to declare
    /// zero attackers (see `advance_step`'s doc). Callers that mirror a
    /// reference implementation which *does* skip logging an empty-eligible
    /// window (as the Java harness does) need their own silent-auto-resolve
    /// handling for that case -- this decision itself makes no such
    /// distinction.
    DeclareAttackers {
        player: PlayerId,
        eligible: Vec<ObjectId>,
    },
    /// 509.1: choose blocks. `legal_blockers` is given per attacker
    /// (flying/reach constrain which of the defending player's untapped
    /// creatures may block which attacker -- see
    /// `card_def::Keywords::FLYING`/`REACH`). Always asked whenever
    /// `attackers` is non-empty, even if `legal_blockers` is empty for
    /// every attacker.
    DeclareBlockers {
        player: PlayerId,
        attackers: Vec<ObjectId>,
        legal_blockers: Vec<(ObjectId, Vec<ObjectId>)>,
    },
    /// Stub per the design brief: fixed APNAP grouping always happens;
    /// this decision only fires when one player controls 2+ simultaneous
    /// triggers and must choose an order for them (603.3b). No card in
    /// this increment's pool triggers 2+ simultaneously, so it's
    /// unreachable from either acceptance test; see
    /// `tests::order_triggers_decision_exists` for a synthetic proof it
    /// works.
    OrderTriggers {
        player: PlayerId,
        pending: Vec<PendingTrigger>,
    },
    GameOver {
        winner: Option<PlayerId>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    PlayLand(ObjectId),
    CastSpell(ObjectId),
    ActivateManaAbility(ObjectId),
    ActivateAbility(ObjectId, u8),
    Pass,
    ChooseTarget(Target),
    /// Answers one `Decision::ChooseCostTargets` pick (one permanent at a
    /// time -- see that decision's doc).
    ChooseCostTarget(ObjectId),
    ChooseCastMode(CastMode),
    Discard(Vec<ObjectId>),
    DeclareAttackers(Vec<ObjectId>),
    /// (blocker, attacker) pairs. A blocker may appear at most once; an
    /// attacker may appear multiple times (gang-blocked).
    DeclareBlockers(Vec<(ObjectId, ObjectId)>),
    /// Indices into the current `OrderTriggers` decision's `pending`, in
    /// the order they should be placed on the stack (last index resolves
    /// first -- stack is LIFO).
    OrderTriggers(Vec<usize>),
    /// Plots a hand card (`PlotAbility`, a `SpecialAction`): pays
    /// `CardDef::plot_cost`, exiles it, and marks it castable for free on a
    /// later turn. Doesn't use the stack and doesn't pass priority (605.3b-
    /// like: same shape as `PlayLand`).
    PlotSpell(ObjectId),
    ChooseSpellMode(u8),
    ChooseOptionalCost(OptionalCostChoice),
    /// `true` = cast the pending Madness card for its madness cost; `false`
    /// = let it go to the graveyard.
    ChooseMadnessCast(bool),
}

const STEP_ORDER: [Step; 12] = [
    Step::Untap,
    Step::Upkeep,
    Step::Draw,
    Step::Main1,
    Step::BeginCombat,
    Step::DeclareAttackers,
    Step::DeclareBlockers,
    Step::CombatDamage,
    Step::EndCombat,
    Step::Main2,
    Step::End,
    Step::Cleanup,
];

fn step_grants_priority(step: Step) -> bool {
    !matches!(step, Step::Untap | Step::Cleanup)
}

fn target_count(spec: TargetSpec) -> u8 {
    match spec {
        TargetSpec::None => 0,
        TargetSpec::AnyTarget
        | TargetSpec::AnySpellOnStack
        | TargetSpec::BlueSpellOnStack
        | TargetSpec::AnyPermanent
        | TargetSpec::BluePermanent => 1,
        TargetSpec::PlayerThenTheirCreature => 2,
    }
}

fn is_blue(state: &GameState, id: ObjectId) -> bool {
    let def_idx = state.objects.get(id).card_def;
    card_def::CARD_DEFS[def_idx as usize].colors.contains(&mana::ManaColor::U)
}

fn battlefield_objects(state: &GameState) -> impl Iterator<Item = ObjectId> + '_ {
    [PlayerId::P0, PlayerId::P1].into_iter().flat_map(|p| state.players[p.index()].battlefield.iter().copied())
}

/// `targets_chosen` is the *already-picked* prefix for this same targeting
/// pass -- needed for `PlayerThenTheirCreature`'s second, dependent pick
/// (any other spec's legal pool doesn't vary with what's already chosen, so
/// they simply ignore it).
pub fn legal_targets_for(spec: TargetSpec, targets_chosen: &[Target], state: &GameState) -> Vec<Target> {
    match spec {
        TargetSpec::None => Vec::new(),
        TargetSpec::AnyTarget => {
            let mut out = vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)];
            for p in [PlayerId::P0, PlayerId::P1] {
                for &id in &state.players[p.index()].battlefield {
                    let obj = state.objects.get(id);
                    if card_def::CARD_DEFS[obj.card_def as usize].has_type(CardType::Creature) {
                        out.push(Target::Object(id));
                    }
                }
            }
            out
        }
        TargetSpec::PlayerThenTheirCreature => {
            if targets_chosen.is_empty() {
                vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)]
            } else if let Some(Target::Player(p)) = targets_chosen.first() {
                state.players[p.index()]
                    .battlefield
                    .iter()
                    .copied()
                    .filter(|&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].has_type(CardType::Creature))
                    .map(Target::Object)
                    .collect()
            } else {
                Vec::new()
            }
        }
        TargetSpec::AnySpellOnStack => state.stack.iter().map(|item| Target::Object(item.source)).collect(),
        TargetSpec::BlueSpellOnStack => state.stack.iter().map(|item| item.source).filter(|&id| is_blue(state, id)).map(Target::Object).collect(),
        TargetSpec::AnyPermanent => battlefield_objects(state).map(Target::Object).collect(),
        TargetSpec::BluePermanent => battlefield_objects(state).filter(|&id| is_blue(state, id)).map(Target::Object).collect(),
    }
}

// ------------------------------------------------------------------ query

/// A component of a non-mana cost that only makes sense in an already-
/// affordable / already-decided context (a `DiscardCards` component whose
/// actual discard already happened via `Decision::Discard`). Both
/// `can_pay_components` and `pay_cost_components` treat it as trivially
/// satisfied/a no-op; the real legality/payment lives in the
/// `pending_discard` staging.
fn discard_count_in(components: &[CostComponent]) -> Option<u8> {
    components.iter().find_map(|c| match c {
        CostComponent::DiscardCards(n) => Some(*n),
        _ => None,
    })
}

/// `pub(crate)`: also read by `effect::execute`'s `MayPayCostThen` handler
/// to check whether the sacrifice-a-land sub-cost is currently payable.
pub(crate) fn count_controlled_lands(player: PlayerId, state: &GameState) -> u32 {
    state.players[player.index()]
        .battlefield
        .iter()
        .filter(|&&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land)
        .count() as u32
}

/// Whether `player` can currently pay every component of `components`,
/// where `source` is the spell being cast (still nominally in hand) or
/// the permanent whose ability is being activated (on the battlefield --
/// so the hand-exclusion below is a no-op for abilities).
fn can_pay_components(components: &[CostComponent], player: PlayerId, source: ObjectId, state: &GameState) -> bool {
    for c in components {
        let ok = match c {
            CostComponent::Tap => {
                let obj = state.objects.get(source);
                let def = &card_def::CARD_DEFS[obj.card_def as usize];
                // 302.6: a *creature's* tap-cost ability needs continuous
                // control since the turn began. Irrelevant to every
                // tap-cost ability in this pool (Blood is an artifact),
                // kept for correctness if a future card needs it.
                !(obj.tapped || (def.has_type(CardType::Creature) && obj.summoning_sick))
            }
            CostComponent::SacrificeSelf | CostComponent::ExileSelf => true,
            CostComponent::DiscardCards(n) => {
                let hand_other = state.players[player.index()].hand.iter().filter(|&&id| id != source).count();
                hand_other >= *n as usize
            }
            CostComponent::SacrificeLands(n) => count_controlled_lands(player, state) >= *n as u32,
            CostComponent::Mana(cost) => mana::can_pay(cost, 0, player, state).is_some(),
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Pays every component of `components` except `DiscardCards` (already
/// paid via the `pending_discard` staging by the time this runs -- see
/// `EngineState::pending_discard`'s doc). `sacrifice_chosen` is the
/// already-decided answer to any `CostComponent::SacrificeLands(n)` in
/// `components` (`PendingCast::sacrifice_chosen`, staged by
/// `Decision::ChooseCostTargets` before this ever runs -- see
/// `sacrifice_lands_needed`'s doc); pass `&[]` for a `components` list
/// that's statically known never to contain `SacrificeLands` (every
/// activated-ability cost and every `additional_cost` in this pool), where
/// the `debug_assert_eq!` below is the fail-loud guard against that
/// assumption silently going stale.
fn pay_cost_components(state: &mut GameState, player: PlayerId, source: ObjectId, components: &[CostComponent], sacrifice_chosen: &[ObjectId]) {
    for c in components {
        match c {
            CostComponent::Tap => event::propose_and_commit(state, ProposedEvent::tap(source)),
            CostComponent::SacrificeSelf => event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Graveyard)),
            CostComponent::ExileSelf => event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Exile)),
            CostComponent::SacrificeLands(n) => {
                debug_assert_eq!(sacrifice_chosen.len(), *n as usize, "sacrifice_chosen must be exactly this component's already-decided picks");
                commit_sacrifice(state, sacrifice_chosen);
            }
            CostComponent::Mana(cost) => {
                let plan = mana::can_pay(cost, 0, player, state).expect("legality already checked by can_pay_components");
                pay_plan(state, player, &plan);
            }
            CostComponent::DiscardCards(_) => {}
        }
    }
}

/// Zone-changes exactly `chosen` to the graveyard -- the actual payment
/// half of a `SacrificeLands` sub-cost, once `Decision::ChooseCostTargets`
/// has already decided *which* permanents (`PendingCast::sacrifice_chosen`).
fn commit_sacrifice(state: &mut GameState, chosen: &[ObjectId]) {
    for &id in chosen {
        event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
    }
}

/// How many lands (0 if none) the cast currently staged in `pending`
/// still needs sacrificed to pay its cost: Fireblast's alt cost, once
/// `cast_mode` has resolved to `Alternative`; Lava Dart's flashback cost,
/// unconditionally (a flashback cast has no alternative payment path to
/// resolve first). A bare `u8` rather than `Option<u8>` -- no card in
/// this pool has `SacrificeLands(0)`, so "not applicable" and "applicable
/// but zero" are not a real ambiguity here -- mirrors `target_count`'s own
/// shape.
fn sacrifice_lands_needed(pending: &PendingCast, def: &card_def::CardDef) -> u8 {
    if pending.is_flashback {
        return match def.flashback.as_ref().map(|fb| fb.cost) {
            Some(FlashbackCost::SacrificeLands(n)) => n,
            _ => 0,
        };
    }
    if pending.cast_mode == Some(CastMode::Alternative) {
        if let Some(alt) = def.alt_cost {
            for c in alt {
                if let CostComponent::SacrificeLands(n) = c {
                    return *n;
                }
            }
        }
    }
    0
}

/// `player`'s currently-controlled lands, minus whichever have already
/// been picked this same `Decision::ChooseCostTargets` sequence -- the
/// candidate pool for the next pick (see that decision's doc for why a
/// picked land disappears from the next ask's candidates).
fn sacrificeable_lands(player: PlayerId, state: &GameState, already_chosen: &[ObjectId]) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land && !already_chosen.contains(&id))
        .collect()
}

/// Whether `id` (from hand or graveyard) is castable right now, given
/// sorcery-speed timing and every cost path (`is_flashback` selects
/// between the normal cost/alt-cost pair and the flashback cost).
fn is_castable_now(player: PlayerId, id: ObjectId, is_flashback: bool, state: &GameState) -> bool {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    if !is_flashback && !def.is_castable() {
        return false;
    }
    let sorcery_speed_ok = if def.has_type(CardType::Sorcery) || def.has_type(CardType::Creature) {
        sorcery_speed_timing_ok(player, state)
    } else {
        true // instants: castable any time the caster has priority
    };
    if !sorcery_speed_ok {
        return false;
    }

    if is_flashback {
        let fb = def.flashback.as_ref().expect("caller only passes is_flashback=true for cards with a flashback cost");
        match fb.cost {
            FlashbackCost::Mana(cost) => mana::can_pay(&cost, 0, player, state).is_some(),
            FlashbackCost::SacrificeLands(n) => count_controlled_lands(player, state) >= n as u32,
        }
    } else {
        let normal_ok = mana::can_pay(&def.cost, 0, player, state).is_some();
        let alt_ok = def.alt_cost.map(|c| can_pay_components(c, player, id, state)).unwrap_or(false);
        if !normal_ok && !alt_ok {
            return false;
        }
        if let Some(add) = def.additional_cost {
            if !can_pay_components(add, player, id, state) {
                return false;
            }
        }
        true
    }
}

fn castable_spells(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    let mut out = Vec::new();
    for &id in &state.players[player.index()].hand {
        if is_castable_now(player, id, false, state) {
            out.push(id);
        }
    }
    for &id in &state.players[player.index()].graveyard {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        if def.flashback.is_some() && is_castable_now(player, id, true, state) {
            out.push(id);
        }
    }
    for &id in &state.exile {
        if state.objects.get(id).owner == player && is_plotted_castable_now(player, id, state) {
            out.push(id);
        }
    }
    out
}

/// Sorcery-speed timing (508.1a's "any time you could cast a sorcery"),
/// shared by an ordinary sorcery-speed cast (`is_castable_now`), a Plotted
/// card cast from exile, and Plotting itself (`plot_action_candidates`) --
/// 702.163a/`PlotAbility.setTiming(TimingRule.SORCERY)` grant that timing
/// regardless of the card's own type.
fn sorcery_speed_timing_ok(player: PlayerId, state: &GameState) -> bool {
    player == state.active_player && state.stack.is_empty() && matches!(state.step, Step::Main1 | Step::Main2)
}

/// Whether `id` (in `state.exile`, owned by `player`) was Plotted on an
/// earlier turn and is therefore castable for free right now (702.163a: at
/// sorcery speed, never the turn it was Plotted). `def.is_castable()` keeps
/// this fail-closed if a future Plot-able card is added to the JSON pool
/// before its `spell_effect` is implemented.
fn is_plotted_castable_now(player: PlayerId, id: ObjectId, state: &GameState) -> bool {
    let obj = state.objects.get(id);
    let Some(plotted_turn) = obj.plotted_turn else { return false };
    if plotted_turn == state.turn {
        return false;
    }
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    def.is_castable() && sorcery_speed_timing_ok(player, state)
}

/// Hand cards `player` can currently afford to Plot (`CardDef::plot_cost`).
fn plot_action_candidates(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    if !sorcery_speed_timing_ok(player, state) {
        return Vec::new();
    }
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .filter(|&id| {
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            def.plot_cost.is_some_and(|cost| mana::can_pay(&cost, 0, player, state).is_some())
        })
        .collect()
}

fn available_mana_abilities(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| {
            let obj = state.objects.get(id);
            !obj.tapped && (card_def::CARD_DEFS[obj.card_def as usize].mana_ability)().is_some()
        })
        .collect()
}

fn available_activatable_abilities(player: PlayerId, state: &GameState) -> Vec<(ObjectId, u8)> {
    let mut out = Vec::new();
    for &id in &state.players[player.index()].battlefield {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        for (i, ability) in def.activated_abilities.iter().enumerate() {
            if can_pay_components(ability.cost, player, id, state) {
                out.push((id, i as u8));
            }
        }
    }
    out
}

fn land_drop_candidates(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    if player != state.active_player
        || state.players[player.index()].lands_played_this_turn > 0
        || !matches!(state.step, Step::Main1 | Step::Main2)
        || !state.stack.is_empty()
    {
        return Vec::new();
    }
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .filter(|&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land)
        .collect()
}

fn can_attack(state: &GameState, id: ObjectId) -> bool {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    def.has_type(CardType::Creature) && !obj.tapped && (!obj.summoning_sick || def.keywords.has(Keywords::HASTE))
}

fn eligible_attackers(state: &GameState) -> Vec<ObjectId> {
    state.players[state.active_player.index()].battlefield.iter().copied().filter(|&id| can_attack(state, id)).collect()
}

/// Which of the defending player's untapped creatures may legally block
/// `attacker` (509.1b: flying attackers can only be blocked by
/// flying/reach).
fn legal_blockers_for(state: &GameState, attacker: ObjectId) -> Vec<ObjectId> {
    let attacker_obj = state.objects.get(attacker);
    let defender = attacker_obj.controller.opponent();
    let attacker_flying = card_def::CARD_DEFS[attacker_obj.card_def as usize].keywords.has(Keywords::FLYING);
    state.players[defender.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| {
            let obj = state.objects.get(id);
            if obj.tapped {
                return false;
            }
            let def = &card_def::CARD_DEFS[obj.card_def as usize];
            if !def.has_type(CardType::Creature) {
                return false;
            }
            if attacker_flying && !def.keywords.has(Keywords::FLYING) && !def.keywords.has(Keywords::REACH) {
                return false;
            }
            true
        })
        .collect()
}

fn check_game_over(state: &GameState) -> Option<Decision> {
    match (state.players[0].has_lost, state.players[1].has_lost) {
        (false, false) => None,
        (true, false) => Some(Decision::GameOver { winner: Some(PlayerId::P1) }),
        (false, true) => Some(Decision::GameOver { winner: Some(PlayerId::P0) }),
        (true, true) => Some(Decision::GameOver { winner: None }),
    }
}

// -------------------------------------------------------------- the loop

/// Drives the state machine forward until a real decision point is
/// reached. Never auto-passes and never skips a priority window that the
/// comprehensive rules would actually grant.
pub fn advance_until_decision(state: &mut GameState) -> Decision {
    loop {
        if let Some(d) = check_game_over(state) {
            return d;
        }

        if let Some(d) = drain_pending_discard_or_decide(state) {
            return d;
        }

        // A discard just processed above may have queued a Madness card
        // (`apply_discard` exiles it instead of graveyarding it, on the way
        // in) -- ask (or silently auto-resolve to graveyard, if
        // unaffordable) before anything else gets a chance to offer a
        // priority window.
        if let Some(d) = drain_pending_madness_or_decide(state) {
            return d;
        }

        if let Some(d) = drain_pending_cast_or_decide(state) {
            return d;
        }
        // A cast's additional-cost discard (Grab the Prize) may have just
        // been staged here (`pending_discard = Some(...); return None`,
        // per that branch's doc) -- restart from the top so
        // `drain_pending_discard_or_decide` picks it up *before* anything
        // below gets a chance to fall through to a priority offer. Without
        // this, the loop would fall straight through the declare-attackers/
        // blockers checks and the priority-window return at the bottom of
        // this same pass, wrongly offering a `CastSpellOrPass` decision
        // (including the very ability/cast that's still mid-cost-payment)
        // before the discard the player owes has ever been asked for.
        if state.engine.pending_discard.is_some() {
            continue;
        }

        if let Some(d) = drain_pending_activation_or_decide(state) {
            return d;
        }
        // Same reasoning as above, for an activated ability's discard cost
        // (Masked Meower, the Blood token).
        if state.engine.pending_discard.is_some() {
            continue;
        }

        // Unlike `pending_discard`/`pending_cast`/`pending_activation`,
        // `pending_optional_cost` never auto-resolves (declining is always
        // legal, so there's always a real choice once staged) -- no
        // `continue`-and-recheck dance needed here: if `Action::
        // ChooseOptionalCost(Discard)` stages a fresh `pending_discard`,
        // that's picked up at the very top of the *next* call to this
        // function (the loop only reaches this far after already
        // confirming `pending_discard` is `None`).
        if let Some(d) = drain_pending_optional_cost_or_decide(state) {
            return d;
        }

        if let Some(d) = drain_pending_optional_cost_sacrifice_or_decide(state) {
            return d;
        }
        // Same reasoning as the two `pending_discard` re-checks above:
        // once every land is chosen, this runs `poc.then` (Highway
        // Robbery's own "draw two cards"), which could in principle stage
        // a fresh `pending_discard` of its own.
        if state.engine.pending_discard.is_some() {
            continue;
        }

        if let Some(d) = drain_pending_triggers_or_decide(state) {
            return d;
        }

        if state.step == Step::DeclareAttackers && !state.engine.combat.attackers_declared {
            return Decision::DeclareAttackers { player: state.active_player, eligible: eligible_attackers(state) };
        }
        if state.step == Step::DeclareBlockers && !state.engine.combat.blockers_declared {
            let attackers = state.engine.combat.attackers.clone();
            let legal_blockers = attackers.iter().map(|&a| (a, legal_blockers_for(state, a))).collect();
            return Decision::DeclareBlockers { player: state.active_player.opponent(), attackers, legal_blockers };
        }

        if !step_grants_priority(state.step) {
            advance_step(state);
            continue;
        }

        if state.engine.priority_passes == [true, true] {
            if !state.stack.is_empty() {
                resolve_top_of_stack(state);
                collect_and_queue_triggers(state);
                reset_priority(state);
            } else {
                advance_step(state);
            }
            continue;
        }

        return Decision::CastSpellOrPass {
            player: state.priority_player,
            castable_spells: castable_spells(state.priority_player, state),
            mana_abilities: available_mana_abilities(state.priority_player, state),
            land_drops: land_drop_candidates(state.priority_player, state),
            activatable_abilities: available_activatable_abilities(state.priority_player, state),
            plot_actions: plot_action_candidates(state.priority_player, state),
        };
    }
}

fn reset_priority(state: &mut GameState) {
    state.engine.priority_passes = [false, false];
    state.priority_player = state.active_player;
    state.engine.priority_round += 1;
}

fn collect_and_queue_triggers(state: &mut GameState) {
    let triggers = trigger::collect_and_process(state);
    state.engine.pending_triggers.extend(triggers);
}

/// If a discard is pending, either auto-resolves it (no real choice: the
/// legal pool is already <= the required count) or returns
/// `Decision::Discard`.
fn drain_pending_discard_or_decide(state: &mut GameState) -> Option<Decision> {
    let pd = state.engine.pending_discard.clone()?;
    let exclude = if pd.resume == DiscardResume::FinishCast { state.engine.pending_cast.as_ref().map(|p| p.spell) } else { None };
    let choices: Vec<ObjectId> = state.players[pd.player.index()].hand.iter().copied().filter(|&id| Some(id) != exclude).collect();

    if choices.len() <= pd.count as usize {
        state.engine.pending_discard = None;
        apply_discard(state, choices, pd.resume);
        return None;
    }
    Some(Decision::Discard { player: pd.player, count: pd.count, choices })
}

fn apply_discard(state: &mut GameState, chosen: Vec<ObjectId>, resume: DiscardResume) {
    for &id in &chosen {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        if def.madness_cost.is_some() {
            // 702.33a: a discarded Madness card is exiled instead of
            // graveyarded, and its owner is later offered the chance to
            // cast it for its madness cost -- see
            // `drain_pending_madness_or_decide`.
            event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Exile));
            state.engine.pending_madness.push(id);
        } else {
            event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
        }
    }
    match resume {
        DiscardResume::None => collect_and_queue_triggers(state),
        DiscardResume::FinishCast => {
            if let Some(p) = state.engine.pending_cast.as_mut() {
                p.additional_cost_discarded = Some(chosen);
            }
        }
        DiscardResume::FinishActivation => {
            // 602.2b-h: every component of an activation's cost is paid
            // together, as one atomic action, the instant the interactive
            // part (which card(s) to discard) is determined -- not
            // deferred to `finalize_activation` (pushing the ability's
            // `StackItem`), which can run arbitrarily later (after a
            // Madness trigger this same discard caused has already been
            // fully offered and attempted, in particular). Root-caused
            // against `game_20260713_002156_0014.txt` decision 45: Masked
            // Meower's `[DiscardCards(1), SacrificeSelf]` ability discards
            // a Fiery Temper, and the reference's own graveyard snapshot
            // already has Masked Meower in it *before* Fiery Temper's
            // madness-cast target pick is even offered -- i.e. the
            // sacrifice happens right here, alongside the discard,
            // not after the triggered Madness attempt resolves. The Blood
            // Token's ability has the same `[..., DiscardCards(1),
            // SacrificeSelf]` shape and would hit the identical ordering
            // bug were it ever exercised this way.
            if let Some(p) = state.engine.pending_activation.clone() {
                let def = &card_def::CARD_DEFS[state.objects.get(p.source).card_def as usize];
                let ability = &def.activated_abilities[p.ability_index as usize];
                pay_cost_components(state, p.controller, p.source, ability.cost, &[]);
                // 111.8/704.5d: paying `SacrificeSelf` here (a token's own
                // ability, e.g. the Blood Token's) can put a token into the
                // graveyard well before `finalize_activation` ever runs
                // (pending_madness -- see `drain_pending_madness_or_decide`
                // -- is checked *before* `pending_activation` in
                // `advance_until_decision`'s loop, so a Madness sub-flow
                // this same discard triggers can fully play out first).
                // `sba_fixed_point` normally only runs inside
                // `collect_and_queue_triggers`, which `finalize_activation`
                // doesn't call until this activation is fully resolved --
                // too late to make the token cease to exist before the
                // comparator's `check_state` observes an intervening
                // decision's graveyard size. Run it right here instead,
                // same "don't leave a token sitting somewhere it can never
                // really be" fix `sba_fixed_point`'s own 111.8/704.5d rule
                // already applies everywhere else.
                trigger::sba_fixed_point(state);
            }
            if let Some(p) = state.engine.pending_activation.as_mut() {
                p.cost_discard_paid = Some(chosen);
            }
        }
        DiscardResume::FinishSpellResolution { source, to_zone } => {
            // See `DiscardResume::FinishSpellResolution`'s doc: this is the
            // "move to post-resolution zone" step `resolve_top_of_stack`
            // deferred until the resolution-effect discard it triggered
            // (Faithless Looting) actually landed.
            event::propose_and_commit(state, ProposedEvent::zone_change(source, to_zone));
            collect_and_queue_triggers(state);
        }
        DiscardResume::FinishOptionalCost { source, controller, then, spell_resume } => {
            // See `EffectOp::MayPayCostThen`'s doc: the discard sub-cost
            // just landed, so run the effect it unlocked (Highway Robbery:
            // draw two cards).
            let ctx = ExecCtx::no_targets(source, controller);
            effect::execute(&then, &ctx, state);
            // See `PendingOptionalCost::spell_resume`'s doc: only now is
            // the spell this optional cost belongs to *actually* fully
            // resolved -- perform the move `resolve_top_of_stack` deferred.
            if let Some((spell, to_zone)) = spell_resume {
                event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
            }
            collect_and_queue_triggers(state);
        }
    }
}

/// If a Madness card is queued (`EngineState::pending_madness`, populated by
/// `apply_discard`), returns `Decision::ChooseMadnessCast` -- unconditionally,
/// with **no affordability pre-check**.
///
/// An earlier version of this function pre-filtered on `mana::can_pay`,
/// silently sending an unaffordable card to the graveyard with no `Decision`
/// at all (the same "don't ask when there's only one legal answer" shortcut
/// `drain_pending_cast_or_decide` uses for Fireblast's cast mode). That was
/// root-caused as wrong (Sol #90, increment 11) against the real Java
/// reference: `MadnessTriggeredAbility.resolve()` (`MadnessAbility.java`)
/// is a plain *optional* triggered ability -- it always calls
/// `player.chooseUse(...)` first, with no discrete `canPay()` gate anywhere
/// in that path. "Affordable" in the reference means nothing more than "the
/// real `cast()` call happened to succeed"; an unaffordable attempt still
/// gets offered, still begins a real cast (its own target-selection decision
/// genuinely logged), and only fails -- reverting the card to the graveyard
/// -- at the cost-payment step. Confirmed against two golden-trace games
/// (`game_20260713_002149_0004.txt` decision 26, `game_20260713_002156_0014.txt`
/// decision 45): both show a real `SELECT_TARGETS` record for the Madness
/// card's own target at a moment when the discarding player has zero
/// available mana (every battlefield permanent already tapped this same
/// turn paying for earlier, unrelated spells) -- and in the second game, the
/// very next trace record for that player shows the card sitting in the
/// graveyard, never on the stack/battlefield again, proving the attempt was
/// offered, attempted, and reverted, not silently declined upfront. See
/// `apply_choose_madness_cast`'s `cast_it == true` branch and `abort_cast`'s
/// Madness-specific graveyard redirect for the other half of this fix.
fn drain_pending_madness_or_decide(state: &mut GameState) -> Option<Decision> {
    let &card = state.engine.pending_madness.first()?;
    let owner = state.objects.get(card).owner;
    Some(Decision::ChooseMadnessCast { player: owner, card })
}

/// If a resolution-time optional cost is pending (`EngineState::
/// pending_optional_cost`, staged by `EffectOp::MayPayCostThen`), returns
/// `Decision::ChooseOptionalCost`. Always asked, never auto-resolved:
/// declining is always a legal answer, so this is never a "no real choice"
/// situation the way a forced discard or a single-affordable-cast-mode is.
fn drain_pending_optional_cost_or_decide(state: &mut GameState) -> Option<Decision> {
    let poc = state.engine.pending_optional_cost.as_ref()?;
    Some(Decision::ChooseOptionalCost { player: poc.player, discard_payable: poc.discard_payable, sacrifice_payable: poc.sacrifice_payable })
}

/// If `Action::ChooseOptionalCost(SacrificeLand)` was just chosen, stages
/// *which* land(s) through the same per-pick `Decision::ChooseCostTargets`
/// flow `drain_pending_cast_or_decide` uses for Fireblast/Lava Dart (see
/// `sacrifice_lands_needed`'s doc for the per-pick auto-resolve rule this
/// mirrors: ask unless this single pick's own candidate pool is `<= 1`).
/// Once fully chosen, commits the sacrifice and runs `then` (Highway
/// Robbery's "draw two cards").
fn drain_pending_optional_cost_sacrifice_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_optional_cost_sacrifice.clone()?;
    if (pending.chosen.len() as u8) < pending.remaining {
        let candidates = sacrificeable_lands(pending.player, state, &pending.chosen);
        let remaining = pending.remaining - pending.chosen.len() as u8;
        if candidates.len() <= 1 {
            state.engine.pending_optional_cost_sacrifice.as_mut().unwrap().chosen.extend(candidates);
            return drain_pending_optional_cost_sacrifice_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets { player: pending.player, source: pending.source, cost_kind: CostKind::SacrificeLands, remaining, candidates });
    }
    let pending = state.engine.pending_optional_cost_sacrifice.take().expect("checked Some above");
    commit_sacrifice(state, &pending.chosen);
    let ctx = ExecCtx::no_targets(pending.source, pending.player);
    effect::execute(&pending.then, &ctx, state);
    // See `PendingOptionalCost::spell_resume`'s doc: only now is the spell
    // this optional cost belongs to actually fully resolved.
    if let Some((spell, to_zone)) = pending.spell_resume {
        event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
    }
    None
}

/// Stages `PendingCast` through its targets -> cast-mode -> additional-
/// cost-discard -> finalize pipeline, one stage per call (each stage that
/// makes progress `continue`s the outer loop instead of looping here, so
/// `pending_discard`/triggers/etc. staged along the way always get
/// checked first).
fn drain_pending_cast_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_cast.clone()?;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];

    // 601.2b: mode is chosen before targets (the two modes can have
    // different target shapes) -- always asked when the card is modal,
    // never auto-resolved (both modes are always legal to *choose*
    // regardless of the battlefield/stack; see `Decision::ChooseSpellMode`'s
    // doc).
    if def.mode2.is_some() && pending.mode_chosen.is_none() {
        return Some(Decision::ChooseSpellMode { player: pending.controller, spell: pending.spell, mode_count: 2 });
    }
    let active_target_spec = match pending.mode_chosen {
        Some(1) => def.mode2.as_ref().expect("mode_chosen == 1 only when mode2 is Some").target_spec,
        _ => def.target_spec,
    };

    let need = target_count(active_target_spec);
    if (pending.targets_chosen.len() as u8) < need {
        return Some(Decision::ChooseTargets {
            player: pending.controller,
            spell: pending.spell,
            remaining: need - pending.targets_chosen.len() as u8,
            legal_targets: legal_targets_for(active_target_spec, &pending.targets_chosen, state),
        });
    }

    if pending.cast_mode.is_none() {
        let alt = def.alt_cost.expect("cast_mode is None only when begin_cast saw an alt_cost");
        let normal_ok = mana::can_pay(&def.cost, 0, pending.controller, state).is_some();
        let alt_ok = can_pay_components(alt, pending.controller, pending.spell, state);
        if normal_ok && alt_ok {
            return Some(Decision::ChooseCastMode { player: pending.controller, spell: pending.spell, options: vec![CastMode::Normal, CastMode::Alternative] });
        }
        state.engine.pending_cast.as_mut().unwrap().cast_mode = Some(if normal_ok { CastMode::Normal } else { CastMode::Alternative });
        return drain_pending_cast_or_decide(state);
    }

    let sacrifice_needed = sacrifice_lands_needed(&pending, def);
    if (pending.sacrifice_chosen.len() as u8) < sacrifice_needed {
        let candidates = sacrificeable_lands(pending.controller, state, &pending.sacrifice_chosen);
        let remaining = sacrifice_needed - pending.sacrifice_chosen.len() as u8;
        if candidates.len() <= 1 {
            // No real choice for *this single pick* (0 or 1 legal
            // candidate) -- auto-resolve just this one and let the next
            // loop pass re-derive whether the *following* pick (if any)
            // is still real. Deliberately per-pick, not "auto-resolve the
            // whole remaining batch whenever candidates.len() <=
            // remaining": empirically, the reference always logs a real
            // decision for every pick whose own candidate pool has 2+
            // legal choices, even when the *aggregate* choice is forced
            // (e.g. exactly 2 Mountains for Fireblast's 2-land cost still
            // logs one real 2-candidate pick before the final,
            // now-1-candidate pick goes silent) -- root-caused against the
            // v4 corpus's own `(candidate_count...)` sequence per Fireblast/
            // Lava Dart episode (e.g. Fireblast's dominant shape is exactly
            // 2 post-target records, `(target=N, sac1=2)`, never a 3rd
            // `sac2=1` record; Lava Dart with exactly 1 controlled Mountain
            // logs *zero* post-target records at all -- both match "ask
            // until this pick's own pool is <= 1", not "ask until the
            // aggregate is forced"). A first version of this auto-resolve
            // used the aggregate `candidates.len() <= remaining` test and
            // silently over-suppressed exactly this shape (verified against
            // `game_20260713_002152_0008.txt` decision 42: kernel skipped
            // straight past a real 2-candidate Fireblast sacrifice pick the
            // trace logs).
            state.engine.pending_cast.as_mut().unwrap().sacrifice_chosen.extend(candidates);
            return drain_pending_cast_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets { player: pending.controller, source: pending.spell, cost_kind: CostKind::SacrificeLands, remaining, candidates });
    }

    if pending.additional_cost_discarded.is_none() {
        let add = def.additional_cost.expect("additional_cost_discarded is None only when begin_cast saw an additional_cost");
        match discard_count_in(add) {
            Some(n) => {
                state.engine.pending_discard = Some(PendingDiscard { player: pending.controller, count: n as u32, resume: DiscardResume::FinishCast });
                return None; // let the caller's next loop pass see pending_discard
            }
            None => {
                state.engine.pending_cast.as_mut().unwrap().additional_cost_discarded = Some(vec![]);
                return drain_pending_cast_or_decide(state);
            }
        }
    }

    finalize_cast(state);
    None
}

fn drain_pending_activation_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_activation.clone()?;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];

    let need = target_count(pending.target_spec);
    if (pending.targets_chosen.len() as u8) < need {
        return Some(Decision::ChooseTargets {
            player: pending.controller,
            spell: pending.source,
            remaining: need - pending.targets_chosen.len() as u8,
            legal_targets: legal_targets_for(pending.target_spec, &pending.targets_chosen, state),
        });
    }

    if pending.cost_discard_paid.is_none() {
        let n = discard_count_in(ability.cost).expect("cost_discard_paid is None only when begin_activation saw a DiscardCards component");
        state.engine.pending_discard = Some(PendingDiscard { player: pending.controller, count: n as u32, resume: DiscardResume::FinishActivation });
        return None;
    }

    finalize_activation(state);
    None
}

/// If there are pending triggers, either auto-places a singleton group (no
/// real choice to make) or returns `Decision::OrderTriggers` for a group of
/// 2+ sharing a controller.
fn drain_pending_triggers_or_decide(state: &mut GameState) -> Option<Decision> {
    if state.engine.pending_triggers.is_empty() {
        return None;
    }
    let controller = state.engine.pending_triggers[0].controller;
    let group_len = state.engine.pending_triggers.iter().take_while(|t| t.controller == controller).count();

    if group_len >= 2 {
        let pending = state.engine.pending_triggers[..group_len].to_vec();
        return Some(Decision::OrderTriggers { player: controller, pending });
    }

    let group: Vec<_> = state.engine.pending_triggers.drain(..group_len).collect();
    for t in group {
        push_trigger_onto_stack(state, t);
    }
    None
}

fn push_trigger_onto_stack(state: &mut GameState, t: PendingTrigger) {
    state.stack.push(StackItem {
        source: t.source,
        controller: t.controller,
        targets: vec![],
        inline_effect: Some(t.effect),
        discarded: vec![],
        is_flashback: false,
        mode_chosen: 0,
    });
    // Same `priority_passes`/`priority_player` reset as `reset_priority`
    // (117.5: priority passes to the active player once a triggered
    // ability is put on the stack), but deliberately inlined instead of
    // calling that shared helper: a trigger firing mid-cascade off another
    // action (e.g. Guttersnipe off a cast) is not a rules-level "everyone's
    // pass streak clears" boundary in the same sense `advance_step`/a
    // resolution/declare-attackers-or-blockers are, so it must not bump
    // `EngineState::priority_round` -- see that field's doc.
    state.engine.priority_passes = [false, false];
    state.priority_player = state.active_player;
}

/// 704.5g/h creature death, 704.5a life-loss, 704.5c empty-draw-loss all
/// happen inside `resolve_top_of_stack`'s `collect_and_queue_triggers`
/// call; this just pops and executes.
fn resolve_top_of_stack(state: &mut GameState) {
    let item = state.stack.pop().expect("resolve_top_of_stack called with an empty stack");
    let ctx = ExecCtx { source: item.source, controller: item.controller, targets: item.targets, discarded: item.discarded };

    if let Some(effect) = item.inline_effect {
        effect::execute(&effect, &ctx, state);
        return;
    }

    let card_def_idx = state.objects.get(item.source).card_def;
    let def = &card_def::CARD_DEFS[card_def_idx as usize];
    // 601.2b: a modal spell resolves whichever mode was chosen at cast time
    // (`PendingCast::mode_chosen`, threaded onto the `StackItem` by
    // `finalize_cast`) -- `mode_chosen == 1` only for Pyroblast/Red
    // Elemental Blast's destroy mode, everything else always resolves its
    // primary `spell_effect`.
    let program = if item.mode_chosen == 1 {
        def.mode2.as_ref().map(|m| (m.effect)())
    } else {
        (def.spell_effect)()
    };
    if let Some(program) = program {
        effect::execute(&program, &ctx, state);
    }

    // 608.2m: instants/sorceries go to the graveyard as the last part of
    // resolution -- or to exile instead, if this was a flashback cast
    // (702.10e). Creatures/artifacts/enchantments handle entering the
    // battlefield themselves, via their own MoveObject effect.
    if def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery) {
        let to_zone = if item.is_flashback { Zone::Exile } else { Zone::Graveyard };
        if let Some(pd) = state.engine.pending_discard.as_mut() {
            // The effect just resolved into `EffectOp::DiscardCards`
            // (Faithless Looting: "draw two, then discard two"), which
            // stages `pending_discard` and returns *before* the discard is
            // actually answered -- see that leaf's doc and
            // `DiscardResume::FinishSpellResolution`'s. The spell can't
            // reach its post-resolution zone until that discard (part of
            // its own resolution) is done, so defer the move instead of
            // doing it here.
            pd.resume = DiscardResume::FinishSpellResolution { source: item.source, to_zone };
        } else if let Some(poc) = state.engine.pending_optional_cost.as_mut() {
            // Same 608.2m deferral, one layer further out: the effect
            // resolved into `EffectOp::MayPayCostThen` (Highway Robbery:
            // "you may... if you do, draw two cards"), which stages
            // `pending_optional_cost` and returns before the "may pay?"
            // question is even answered, let alone `then` run. Root-caused
            // (Sol #90, increment 11) against several golden-trace games
            // (e.g. `game_20260713_002147_0002.txt` decision 115): moving
            // Highway Robbery to the graveyard *here* put it there several
            // decisions before the reference's own graveyard snapshot ever
            // shows it -- the reference doesn't finish resolving Highway
            // Robbery until its own optional-cost choice (and whichever
            // sub-cost it leads to) is fully paid. See
            // `PendingOptionalCost::spell_resume`'s doc for where the
            // deferred move actually happens once that's all done.
            poc.spell_resume = Some((item.source, to_zone));
        } else {
            event::propose_and_commit(state, ProposedEvent::zone_change(item.source, to_zone));
        }
    }
}

/// Moves `state.step`/`state.active_player`/`state.turn` to the next step,
/// clearing both players' mana pools (500.4: unused mana empties at the end
/// of every step and phase -- a turn-based action, unconditional on whether
/// the step just ended ever granted priority, so this single choke point
/// covers Untap/Cleanup's transitions too, not just the priority-bearing
/// steps), running that step's turn-based entry action (untap, draw,
/// cleanup, combat damage), and resetting priority. Only skips
/// declare-blockers/combat-damage, and only when the active player declared
/// zero attackers (509.4/510.4-ish -- no card in this pool changes that
/// "zero attackers" trigger, so this increment doesn't need the exact rule
/// number to get it right). Declare Attackers itself is *never* skipped,
/// even when no creature could possibly attack: 508.1 makes it a
/// turn-based action that always happens (it's how "the active player
/// declares no attackers" gets decided in the first place), still followed
/// by its own priority round -- confirmed against the real corpus: e.g.
/// `game_20260712_194609_0010.txt` decision #4 is a real, 2-candidate
/// `ACTIVATE_ABILITY_OR_SPELL` record with `phase="Combat"` on turn 1,
/// before either player has a creature on the battlefield (so `eligible`
/// is empty both sides -- yet the reference still asks). An earlier version
/// of this function skipped the whole step whenever `eligible` was empty,
/// which silently ate that priority window entirely -- the resulting
/// "missing decision" left the *next* real trace record unconsumed until
/// some later, unrelated kernel decision, at which point the kernel's
/// `state.step` (already advanced past combat) no longer matched what that
/// stale record was captured against, manifesting downstream as spurious
/// extra/missing `ACTIVATE_ABILITY_OR_SPELL` candidates rather than as an
/// obviously-combat-shaped divergence. See `Decision::DeclareAttackers`'s
/// doc and the replay comparator's handling of an empty `eligible`.
fn advance_step(state: &mut GameState) {
    let cur_idx = STEP_ORDER.iter().position(|&s| s == state.step).expect("state.step is always a STEP_ORDER member");
    let mut next_idx = cur_idx + 1;

    if next_idx >= STEP_ORDER.len() {
        state.active_player = state.active_player.opponent();
        if state.active_player == PlayerId::P0 {
            state.turn += 1;
        }
        next_idx = 0;
    }

    let mut next = STEP_ORDER[next_idx];
    if next == Step::DeclareBlockers && state.engine.combat.attackers.is_empty() {
        next = Step::EndCombat;
    }

    state.step = next;
    state.players[0].mana_pool = [0; 6];
    state.players[1].mana_pool = [0; 6];
    run_step_entry_action(state, next);
    reset_priority(state);
}

fn run_step_entry_action(state: &mut GameState, step: Step) {
    match step {
        Step::Untap => {
            let p = state.active_player;
            let permanents = state.players[p.index()].battlefield.clone();
            for id in permanents {
                let obj = state.objects.get_mut(id);
                obj.tapped = false;
                obj.summoning_sick = false;
            }
            state.players[0].draws_this_turn = 0;
            state.players[1].draws_this_turn = 0;
        }
        Step::Draw => {
            let p = state.active_player;
            // 103.8a: the starting player skips the draw step of their
            // very first turn. `turn == 1 && p == P0` uniquely identifies
            // that turn under this kernel's round-based turn numbering
            // (see module doc): `turn` only becomes 1 again... it never
            // does, it's monotonic, so this combination occurs exactly
            // once, at the start of the game.
            let is_first_turn_of_the_game = state.turn == 1 && p == PlayerId::P0;
            if !is_first_turn_of_the_game {
                event::propose_and_commit(state, ProposedEvent::draw(p));
                collect_and_queue_triggers(state);
            }
        }
        Step::BeginCombat => {
            state.engine.combat = CombatState::default();
        }
        Step::CombatDamage => {
            deal_combat_damage(state);
        }
        Step::Cleanup => {
            // 514.1/514.2: reset damage, "until end of turn" effects end,
            // then discard down to the maximum hand size, then reset the
            // land-drop counter for the player whose turn just ended.
            for (_, obj) in state.objects.iter_mut() {
                obj.damage = 0;
            }
            state.engine.until_end_of_turn.clear();
            let p = state.active_player;
            state.players[p.index()].lands_played_this_turn = 0;
            let hand_size = state.players[p.index()].hand.len();
            if hand_size > 7 {
                state.engine.pending_discard = Some(PendingDiscard { player: p, count: (hand_size - 7) as u32, resume: DiscardResume::None });
            }
        }
        _ => {}
    }
}

// --------------------------------------------------------------- combat

fn effective_power(state: &GameState, id: ObjectId) -> i32 {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    def.power.unwrap_or(0) as i32 + obj.counters.plus1_plus1 as i32
}

fn participates_in_wave(state: &GameState, id: ObjectId, first_strike_wave: bool) -> bool {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    let has_fs = def.keywords.has(Keywords::FIRST_STRIKE);
    let has_ds = def.keywords.has(Keywords::DOUBLE_STRIKE);
    if first_strike_wave {
        has_fs || has_ds
    } else {
        !has_fs || has_ds
    }
}

fn combat_has_first_or_double_strike(state: &GameState) -> bool {
    let all_combatants = state.engine.combat.attackers.iter().copied().chain(state.engine.combat.blocked_by.iter().flat_map(|(_, bs)| bs.iter().copied()));
    all_combatants.into_iter().any(|id| {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        def.keywords.has(Keywords::FIRST_STRIKE) || def.keywords.has(Keywords::DOUBLE_STRIKE)
    })
}

/// 510: all combat damage is dealt in one simultaneous batch, via
/// `event::propose_and_commit_batch` -- unless first/double strike is in
/// play, in which case there are two such batches (510.5), with an SBA
/// check between them so a first-strike kill actually removes its victim
/// before the normal wave. No creature in this increment's pool has
/// either keyword, so `combat_has_first_or_double_strike` is always
/// false here and this always takes the single-wave path; the branch
/// exists so the next increment that adds a first-strike creature only
/// needs to flip a keyword bit, not restructure this function.
fn deal_combat_damage(state: &mut GameState) {
    if combat_has_first_or_double_strike(state) {
        combat_damage_wave(state, true);
        trigger::sba_fixed_point(state);
        combat_damage_wave(state, false);
    } else {
        combat_damage_wave(state, false);
    }
}

fn combat_damage_wave(state: &mut GameState, first_strike_wave: bool) {
    let attackers = state.engine.combat.attackers.clone();
    let blocked_by = state.engine.combat.blocked_by.clone();
    let mut events = Vec::new();

    for &attacker in &attackers {
        if !participates_in_wave(state, attacker, first_strike_wave) {
            continue;
        }
        let power = effective_power(state, attacker);
        if power <= 0 {
            continue;
        }
        if let Some((_, blockers)) = blocked_by.iter().find(|(a, _)| *a == attacker) {
            assign_attacker_damage_to_blockers(state, attacker, power, blockers, &mut events);
        } else {
            let defender = state.objects.get(attacker).controller.opponent();
            events.push(ProposedEvent::damage(attacker, Target::Player(defender), power));
        }
    }
    for (attacker, blockers) in &blocked_by {
        for &blocker in blockers {
            if !participates_in_wave(state, blocker, first_strike_wave) {
                continue;
            }
            let power = effective_power(state, blocker);
            if power > 0 {
                events.push(ProposedEvent::damage(blocker, Target::Object(*attacker), power));
            }
        }
    }

    event::propose_and_commit_batch(state, events);
    collect_and_queue_triggers(state);
}

/// 510.1c, no-trample simplification: lethal damage (toughness minus
/// damage already marked) goes to each blocker in `blockers`' order
/// except the last, which absorbs whatever power remains (there being no
/// trample in this pool, that's the only legal recipient once the
/// attacker itself has already been assigned to blockers rather than the
/// player). A single blocker just gets it all directly. The order itself
/// is `CombatState::blocked_by`'s fixed deterministic sort -- see that
/// field's doc for why this is a stubbed decision point, not a real one,
/// this increment.
fn assign_attacker_damage_to_blockers(state: &GameState, attacker: ObjectId, power: i32, blockers: &[ObjectId], events: &mut Vec<ProposedEvent>) {
    if blockers.len() == 1 {
        events.push(ProposedEvent::damage(attacker, Target::Object(blockers[0]), power));
        return;
    }
    let mut remaining = power;
    for (i, &blocker) in blockers.iter().enumerate() {
        if remaining <= 0 {
            break;
        }
        let is_last = i + 1 == blockers.len();
        let assign = if is_last {
            remaining
        } else {
            let bdef = &card_def::CARD_DEFS[state.objects.get(blocker).card_def as usize];
            let toughness = bdef.toughness.unwrap_or(0) as i32 + state.objects.get(blocker).counters.plus1_plus1 as i32;
            let already = state.objects.get(blocker).damage as i32;
            let lethal_needed = (toughness - already).max(0);
            remaining.min(lethal_needed)
        };
        if assign > 0 {
            events.push(ProposedEvent::damage(attacker, Target::Object(blocker), assign));
        }
        remaining -= assign;
    }
}

// ---------------------------------------------------------------- actions

/// Applies `action` in response to the last `Decision` returned by
/// `advance_until_decision`. Returns `Err` for an action that isn't
/// currently legal (caller bug); never silently no-ops.
pub fn step(state: &mut GameState, action: Action) -> Result<(), String> {
    match action {
        Action::Pass => {
            let p = state.priority_player;
            state.engine.priority_passes[p.index()] = true;
            state.priority_player = p.opponent();
            Ok(())
        }
        Action::PlayLand(id) => {
            let p = state.priority_player;
            if !land_drop_candidates(p, state).contains(&id) {
                return Err(format!("{id} is not a legal land drop for {p:?}"));
            }
            play_land(state, p, id);
            Ok(())
        }
        Action::ActivateManaAbility(id) => {
            let p = state.priority_player;
            if !available_mana_abilities(p, state).contains(&id) {
                return Err(format!("{id} has no available mana ability for {p:?}"));
            }
            let program = (card_def::CARD_DEFS[state.objects.get(id).card_def as usize].mana_ability)()
                .expect("checked available_mana_abilities above");
            let ctx = ExecCtx::no_targets(id, p);
            effect::execute(&program, &ctx, state);
            // 605.3b: mana abilities don't use the stack and don't cause a
            // new priority round.
            Ok(())
        }
        Action::ActivateAbility(source, index) => {
            let p = state.priority_player;
            if !available_activatable_abilities(p, state).contains(&(source, index)) {
                return Err(format!("ability {index} on {source} is not activatable by {p:?} right now"));
            }
            begin_activation(state, p, source, index);
            Ok(())
        }
        Action::CastSpell(id) => {
            let p = state.priority_player;
            if !castable_spells(p, state).contains(&id) {
                return Err(format!("{id} is not castable by {p:?} right now"));
            }
            begin_cast(state, p, id);
            Ok(())
        }
        Action::ChooseTarget(t) => {
            let (spec, chosen_so_far) = pending_target_spec_and_chosen(state).ok_or("no spell or ability is currently being targeted")?;
            if !legal_targets_for(spec, &chosen_so_far, state).contains(&t) {
                return Err(format!("{t:?} is not a legal target"));
            }
            if let Some(p) = state.engine.pending_cast.as_mut() {
                p.targets_chosen.push(t);
            } else if let Some(p) = state.engine.pending_activation.as_mut() {
                p.targets_chosen.push(t);
            }
            Ok(())
        }
        Action::ChooseCostTarget(id) => apply_choose_cost_target(state, id),
        Action::ChooseCastMode(mode) => {
            let pending = state.engine.pending_cast.as_mut().ok_or("no spell is currently being cast")?;
            if pending.cast_mode.is_some() {
                return Err("this cast's mode has already been chosen".to_string());
            }
            pending.cast_mode = Some(mode);
            Ok(())
        }
        Action::ChooseSpellMode(mode) => {
            let p = state.priority_player;
            let pending = state.engine.pending_cast.as_mut().ok_or("no spell is currently being cast")?;
            if pending.mode_chosen.is_some() {
                return Err("this cast's mode has already been chosen".to_string());
            }
            let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
            if def.mode2.is_none() || mode > 1 {
                return Err(format!("{mode} is not a legal spell mode for {p:?}'s cast"));
            }
            pending.mode_chosen = Some(mode);
            Ok(())
        }
        Action::ChooseOptionalCost(choice) => apply_choose_optional_cost(state, choice),
        Action::ChooseMadnessCast(cast_it) => apply_choose_madness_cast(state, cast_it),
        Action::PlotSpell(id) => {
            let p = state.priority_player;
            if !plot_action_candidates(p, state).contains(&id) {
                return Err(format!("{id} is not a legal Plot action for {p:?}"));
            }
            plot_spell(state, p, id);
            Ok(())
        }
        Action::Discard(chosen) => apply_discard_action(state, chosen),
        Action::DeclareAttackers(attackers) => apply_declare_attackers(state, attackers),
        Action::DeclareBlockers(blocks) => apply_declare_blockers(state, blocks),
        Action::OrderTriggers(perm) => apply_order_triggers(state, perm),
    }
}

/// The active `TargetSpec` for whatever is currently being targeted
/// (mode-aware for a modal cast) and the targets already picked so far
/// this same targeting pass -- see `legal_targets_for`'s doc for why the
/// second pick of a dependent spec (`PlayerThenTheirCreature`) needs both.
fn pending_target_spec_and_chosen(state: &GameState) -> Option<(TargetSpec, Vec<Target>)> {
    if let Some(p) = &state.engine.pending_cast {
        let def = &card_def::CARD_DEFS[state.objects.get(p.spell).card_def as usize];
        let spec = match p.mode_chosen {
            Some(1) => def.mode2.as_ref().expect("mode_chosen == 1 only when mode2 is Some").target_spec,
            _ => p.target_spec,
        };
        return Some((spec, p.targets_chosen.clone()));
    }
    if let Some(p) = &state.engine.pending_activation {
        return Some((p.target_spec, p.targets_chosen.clone()));
    }
    None
}

/// Answers one `Decision::ChooseCostTargets` pick -- see that decision's
/// doc. Two pending shapes stage this: `PendingCast` (Fireblast's alt cost,
/// Lava Dart's flashback cost) and `PendingOptionalCostSacrifice` (Highway
/// Robbery's `SacrificeLand` optional cost); no activated ability in this
/// pool has a `SacrificeLands` cost component, unlike `ChooseTarget` which
/// also answers `PendingActivation`'s targeting.
fn apply_choose_cost_target(state: &mut GameState, id: ObjectId) -> Result<(), String> {
    if let Some(pending) = state.engine.pending_cast.as_ref() {
        let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
        let needed = sacrifice_lands_needed(pending, def);
        if (pending.sacrifice_chosen.len() as u8) < needed {
            if !sacrificeable_lands(pending.controller, state, &pending.sacrifice_chosen).contains(&id) {
                return Err(format!("{id} is not a legal cost-sacrifice candidate"));
            }
            state.engine.pending_cast.as_mut().unwrap().sacrifice_chosen.push(id);
            return Ok(());
        }
    }
    if let Some(pending) = state.engine.pending_optional_cost_sacrifice.as_ref() {
        if (pending.chosen.len() as u8) < pending.remaining {
            if !sacrificeable_lands(pending.player, state, &pending.chosen).contains(&id) {
                return Err(format!("{id} is not a legal cost-sacrifice candidate"));
            }
            state.engine.pending_optional_cost_sacrifice.as_mut().unwrap().chosen.push(id);
            return Ok(());
        }
    }
    Err("no sacrifice-cost-target decision is pending".to_string())
}

fn apply_discard_action(state: &mut GameState, chosen: Vec<ObjectId>) -> Result<(), String> {
    let pd = state.engine.pending_discard.clone().ok_or("no discard is pending")?;
    if chosen.len() as u32 != pd.count {
        return Err(format!("must discard exactly {} card(s), got {}", pd.count, chosen.len()));
    }
    let mut dedup = chosen.clone();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != chosen.len() {
        return Err("duplicate card in discard selection".to_string());
    }
    let exclude = if pd.resume == DiscardResume::FinishCast { state.engine.pending_cast.as_ref().map(|p| p.spell) } else { None };
    let hand = &state.players[pd.player.index()].hand;
    if !chosen.iter().all(|id| hand.contains(id) && Some(*id) != exclude) {
        return Err("illegal discard selection".to_string());
    }
    state.engine.pending_discard = None;
    apply_discard(state, chosen, pd.resume);
    Ok(())
}

fn apply_choose_optional_cost(state: &mut GameState, choice: OptionalCostChoice) -> Result<(), String> {
    let poc = state.engine.pending_optional_cost.take().ok_or("no optional cost is pending")?;
    match choice {
        OptionalCostChoice::Decline => {
            // See `PendingOptionalCost::spell_resume`'s doc: declining
            // still means the spell this cost belongs to is now fully
            // resolved (there's no `then` to run either way), so its
            // deferred move happens right here.
            if let Some((spell, to_zone)) = poc.spell_resume {
                event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
            }
            Ok(())
        }
        OptionalCostChoice::Discard => {
            if !poc.discard_payable {
                return Err("discard is not currently a payable option for this optional cost".to_string());
            }
            state.engine.pending_discard = Some(PendingDiscard {
                player: poc.player,
                count: poc.discard as u32,
                resume: DiscardResume::FinishOptionalCost { source: poc.source, controller: poc.player, then: Box::new(poc.then), spell_resume: poc.spell_resume },
            });
            Ok(())
        }
        OptionalCostChoice::SacrificeLand => {
            if !poc.sacrifice_payable {
                return Err("sacrificing a land is not currently a payable option for this optional cost".to_string());
            }
            state.engine.pending_optional_cost_sacrifice = Some(PendingOptionalCostSacrifice {
                player: poc.player,
                source: poc.source,
                remaining: poc.sacrifice_lands,
                chosen: vec![],
                then: poc.then,
                spell_resume: poc.spell_resume,
            });
            Ok(())
        }
    }
}

/// `cast_it == true` no longer pre-verifies affordability (see
/// `drain_pending_madness_or_decide`'s doc) -- it unconditionally stages a
/// real cast via `begin_cast_ex`, exactly as if the card were being cast
/// normally. If the madness cost genuinely can't be paid, that surfaces
/// naturally at `finalize_cast`'s existing `cost_override` affordability
/// check, which now (see `abort_cast`'s doc) reverts a failed Madness
/// attempt to the graveyard instead of erroring -- matching the reference's
/// observed "offer it, attempt it, let it fizzle to the graveyard" behavior
/// rather than a hard failure this function used to return.
fn apply_choose_madness_cast(state: &mut GameState, cast_it: bool) -> Result<(), String> {
    let &card = state.engine.pending_madness.first().ok_or("no Madness decision is pending")?;
    state.engine.pending_madness.remove(0);
    let owner = state.objects.get(card).owner;
    if !cast_it {
        event::propose_and_commit(state, ProposedEvent::zone_change(card, Zone::Graveyard));
        collect_and_queue_triggers(state);
        return Ok(());
    }
    let def = &card_def::CARD_DEFS[state.objects.get(card).card_def as usize];
    let cost = def.madness_cost.expect("only Madness cards are ever queued in pending_madness");
    begin_cast_ex(state, owner, card, Some(cost));
    Ok(())
}

fn apply_declare_attackers(state: &mut GameState, attackers: Vec<ObjectId>) -> Result<(), String> {
    if state.step != Step::DeclareAttackers || state.engine.combat.attackers_declared {
        return Err("no declare-attackers decision is pending".to_string());
    }
    let eligible = eligible_attackers(state);
    if !attackers.iter().all(|a| eligible.contains(a)) {
        return Err("one or more declared attackers is not an eligible attacker".to_string());
    }
    let mut dedup = attackers.clone();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != attackers.len() {
        return Err("duplicate attacker in declaration".to_string());
    }

    for &id in &attackers {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        if !def.keywords.has(Keywords::VIGILANCE) {
            event::propose_and_commit(state, ProposedEvent::tap(id));
        }
    }
    state.engine.combat.attackers = attackers;
    state.engine.combat.attackers_declared = true;
    collect_and_queue_triggers(state);
    reset_priority(state);
    Ok(())
}

fn apply_declare_blockers(state: &mut GameState, blocks: Vec<(ObjectId, ObjectId)>) -> Result<(), String> {
    if state.step != Step::DeclareBlockers || state.engine.combat.blockers_declared {
        return Err("no declare-blockers decision is pending".to_string());
    }
    let attackers = state.engine.combat.attackers.clone();
    let mut used_blockers = Vec::new();
    for &(blocker, attacker) in &blocks {
        if !attackers.contains(&attacker) {
            return Err(format!("{attacker} is not an attacker this combat"));
        }
        if !legal_blockers_for(state, attacker).contains(&blocker) {
            return Err(format!("{blocker} cannot legally block {attacker}"));
        }
        if used_blockers.contains(&blocker) {
            return Err(format!("{blocker} is assigned to block more than one attacker"));
        }
        used_blockers.push(blocker);
    }

    let mut blocked_by: Vec<(ObjectId, Vec<ObjectId>)> = Vec::new();
    for &(blocker, attacker) in &blocks {
        match blocked_by.iter_mut().find(|(a, _)| *a == attacker) {
            Some((_, bs)) => bs.push(blocker),
            None => blocked_by.push((attacker, vec![blocker])),
        }
    }
    for (_, bs) in blocked_by.iter_mut() {
        bs.sort_unstable();
    }

    state.engine.combat.blocked_by = blocked_by;
    state.engine.combat.blockers_declared = true;
    collect_and_queue_triggers(state);
    reset_priority(state);
    Ok(())
}

fn apply_order_triggers(state: &mut GameState, perm: Vec<usize>) -> Result<(), String> {
    if state.engine.pending_triggers.is_empty() {
        return Err("no pending triggers to order".to_string());
    }
    let controller = state.engine.pending_triggers[0].controller;
    let group_len = state.engine.pending_triggers.iter().take_while(|t| t.controller == controller).count();

    let mut sorted = perm.clone();
    sorted.sort_unstable();
    if sorted != (0..group_len).collect::<Vec<_>>() {
        return Err(format!("OrderTriggers action must be a permutation of 0..{group_len}"));
    }

    let group: Vec<_> = state.engine.pending_triggers.drain(..group_len).collect();
    for i in perm {
        push_trigger_onto_stack(state, group[i].clone());
    }
    Ok(())
}

fn play_land(state: &mut GameState, player: PlayerId, id: ObjectId) {
    let ctx = ExecCtx::no_targets(id, player);
    effect::execute(&EffectOp::MoveObject { object: ObjectRef::ThisSource, to_zone: Zone::Battlefield }, &ctx, state);
    state.players[player.index()].lands_played_this_turn += 1;
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = player;
}

/// Plots a hand card (`PlotAbility`): pays `CardDef::plot_cost`, exiles it,
/// and stamps `GameObject::plotted_turn` so `is_plotted_castable_now` can
/// recognize it later. A `SpecialAction` (`usesStack = false` in the Java
/// source): doesn't touch the stack and doesn't pass priority, same shape
/// as `play_land`.
fn plot_spell(state: &mut GameState, player: PlayerId, id: ObjectId) {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    let cost = def.plot_cost.expect("plot_action_candidates only offers cards with a plot_cost");
    let plan = mana::can_pay(&cost, 0, player, state).expect("legality already checked by plot_action_candidates");
    pay_plan(state, player, &plan);
    event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Exile));
    state.objects.get_mut(id).plotted_turn = Some(state.turn);
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = player;
}

/// 601.2a: announcing a cast moves the spell from hand (or graveyard, for a
/// flashback cast) onto the stack immediately -- *before* modes/targets are
/// chosen (601.2b/601.2c) or costs are paid (601.2f-h), which is why
/// `PendingCast`'s later stages (see `drain_pending_cast_or_decide`) mutate
/// the `StackItem` this pushes in place rather than building one from
/// scratch at `finalize_cast`. Pre-resolves the cast-mode/additional-cost/
/// spell-mode stages when there's no real choice to make -- see
/// `PendingCast`'s field docs.
fn begin_cast(state: &mut GameState, player: PlayerId, spell_id: ObjectId) {
    begin_cast_ex(state, player, spell_id, None);
}

/// `forced_cost_override`, when `Some`, is a Madness cast (`Action::
/// ChooseMadnessCast(true)`): the card is already sitting in `state.exile`
/// (moved there by `apply_discard`'s Madness interception), and is cast for
/// exactly this cost rather than any zone-inferred cost. `None` covers every
/// ordinary `Action::CastSpell`, where the cost (if overridden at all) is
/// inferred from the spell's zone instead (`Cost::zero()` for a Plotted
/// card cast from exile).
fn begin_cast_ex(state: &mut GameState, player: PlayerId, spell_id: ObjectId, forced_cost_override: Option<Cost>) {
    let origin_zone = state.objects.get(spell_id).zone;
    let is_flashback = forced_cost_override.is_none() && origin_zone == Zone::Graveyard;
    let is_plotted = forced_cost_override.is_none() && origin_zone == Zone::Exile;
    let def = &card_def::CARD_DEFS[state.objects.get(spell_id).card_def as usize];
    let target_spec = def.target_spec;
    let cost_override = forced_cost_override.or(if is_plotted { Some(Cost::zero()) } else { None });
    let cast_mode = if is_flashback || cost_override.is_some() || def.alt_cost.is_none() { Some(CastMode::Normal) } else { None };
    let additional_cost_discarded = if def.additional_cost.is_none() { Some(vec![]) } else { None };
    let mode_chosen = if def.mode2.is_none() { Some(0) } else { None };

    move_to_stack(state, spell_id, origin_zone);
    state.stack.push(StackItem {
        source: spell_id,
        controller: player,
        targets: vec![],
        inline_effect: None,
        discarded: vec![],
        is_flashback,
        mode_chosen: 0,
    });

    state.engine.pending_cast = Some(PendingCast {
        spell: spell_id,
        controller: player,
        target_spec,
        targets_chosen: vec![],
        is_flashback,
        cast_mode,
        additional_cost_discarded,
        cost_override,
        mode_chosen,
        origin_zone,
        sacrifice_chosen: vec![],
    });
}

/// Stages a `PendingActivation`. The cost itself isn't paid here -- see
/// `apply_discard`'s `DiscardResume::FinishActivation` arm (the discard
/// case) and `finalize_activation` (the no-discard case) for exactly when
/// each component pays, and why that split exists.
fn begin_activation(state: &mut GameState, player: PlayerId, source: ObjectId, ability_index: u8) {
    let def = &card_def::CARD_DEFS[state.objects.get(source).card_def as usize];
    let ability = &def.activated_abilities[ability_index as usize];
    state.engine.pending_activation = Some(PendingActivation {
        source,
        controller: player,
        ability_index,
        target_spec: ability.target_spec,
        targets_chosen: vec![],
        cost_discard_paid: if discard_count_in(ability.cost).is_none() { Some(vec![]) } else { None },
    });
}

/// Pays whichever cost this cast settled on (flashback cost; or the
/// printed mana cost / alt cost, plus any mandatory additional cost) and
/// fills in the targets/discards on the `StackItem` `begin_cast` already
/// pushed (601.2a put the spell there before this ran). 117.3c: the caster
/// retains priority afterward.
fn finalize_cast(state: &mut GameState) {
    let pending = state.engine.pending_cast.take().expect("finalize_cast requires a pending cast");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];

    if let Some(cost) = pending.cost_override {
        // Plot (free) or Madness (its own alternative cost) -- see
        // `PendingCast::cost_override`'s doc. Same unreachable-`abort_cast`
        // shape as every other cost branch here: `Cost::zero()` always
        // trivially pays, and a Madness cast's affordability is re-checked
        // immediately before `begin_cast_ex` is even called
        // (`apply_choose_madness_cast`).
        let Some(plan) = mana::can_pay(&cost, 0, pending.controller, state) else {
            return abort_cast(state, pending);
        };
        pay_plan(state, pending.controller, &plan);
    } else if pending.is_flashback {
        let fb = def.flashback.as_ref().expect("is_flashback implies CardDef::flashback is Some");
        match fb.cost {
            FlashbackCost::Mana(cost) => {
                // 601.2h: legality (including affordability) is fully
                // pre-checked by `is_castable_now` before `Action::CastSpell`
                // is even accepted, and nothing can interleave between
                // `begin_cast`'s announcement and this payment (no priority
                // window opens mid-cast), so `can_pay` returning `None` here
                // is unreachable today. Handled via `abort_cast` (601.2a's
                // "returns to its prior zone" case), not `.expect()`, so a
                // future increment that adds cost-modifying replacement
                // effects or interposed priority doesn't have to rediscover
                // this shape.
                let Some(plan) = mana::can_pay(&cost, 0, pending.controller, state) else {
                    return abort_cast(state, pending);
                };
                pay_plan(state, pending.controller, &plan);
            }
            FlashbackCost::SacrificeLands(n) => {
                debug_assert_eq!(pending.sacrifice_chosen.len(), n as usize, "Decision::ChooseCostTargets must have fully resolved this flashback's sacrifice cost by now");
                commit_sacrifice(state, &pending.sacrifice_chosen);
            }
        }
    } else {
        match pending.cast_mode.expect("resolved by drain_pending_cast_or_decide before finalize_cast is reached") {
            CastMode::Normal => {
                let Some(plan) = mana::can_pay(&def.cost, 0, pending.controller, state) else {
                    return abort_cast(state, pending);
                };
                pay_plan(state, pending.controller, &plan);
            }
            CastMode::Alternative => {
                let alt = def.alt_cost.expect("Alternative mode only chosen when alt_cost is Some");
                pay_cost_components(state, pending.controller, pending.spell, alt, &pending.sacrifice_chosen);
            }
        }
    }
    if let Some(add) = def.additional_cost {
        pay_cost_components(state, pending.controller, pending.spell, add, &[]);
    }

    let discarded = pending.additional_cost_discarded.unwrap_or_default();
    let item = state.stack.last_mut().expect("begin_cast pushed this spell's StackItem and nothing can push another item while a cast is pending");
    debug_assert_eq!(item.source, pending.spell, "the top of the stack must still be this cast's own placeholder");
    item.targets = pending.targets_chosen;
    item.discarded = discarded;
    item.mode_chosen = pending.mode_chosen.unwrap_or(0);
    event::log_spell_cast(state, pending.spell, pending.controller);

    // 601.2i/603.3: casting is complete the instant costs are paid --
    // triggered abilities that saw it happen (Guttersnipe) go on the stack
    // *before* anyone gets priority again, same as `play_land`'s land-drop
    // trigger check.
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
}

/// 601.2a's flip side: if an announced cast turns out to be unpayable, the
/// spell returns to whichever zone it was announced from -- *except* a
/// Madness-cost cast (`PendingCast::cost_override` set from
/// `CardDef::madness_cost`, `apply_choose_madness_cast`'s `cast_it == true`
/// branch), which goes to the graveyard instead of back to exile.
///
/// Root-caused against two golden-trace games (Sol #90, increment 11 --
/// see `drain_pending_madness_or_decide`'s doc for the full citation): the
/// reference lets a player attempt an unaffordable Madness cast (a real,
/// logged target-selection decision happens) and the card ends up in the
/// graveyard once the attempt fails, not back in exile still waiting to be
/// offered again. 702.33a's own wording ("cast it... If you don't, it goes
/// to your graveyard") already frames the *whole* Madness window as
/// resolving to "graveyard" on any non-cast outcome; the reference
/// evidently applies that same ultimate destination to a cast that was
/// *attempted* but couldn't complete, rather than rules-textually
/// re-offering the exiled card later. A Plotted cast's `cost_override`
/// (`Cost::zero()`) can never actually fail this check (paying nothing is
/// always affordable), so this branch is unreachable for Plot -- the
/// madness-cost check is the only real discriminator needed here.
fn abort_cast(state: &mut GameState, pending: PendingCast) {
    let item = state.stack.pop();
    debug_assert!(item.is_some_and(|i| i.source == pending.spell), "abort_cast expects its spell's placeholder to be the top of the stack");
    let owner = state.objects.get(pending.spell).owner;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let to_zone = if pending.origin_zone == Zone::Exile && def.madness_cost.is_some() { Zone::Graveyard } else { pending.origin_zone };
    match to_zone {
        Zone::Hand => state.players[owner.index()].hand.push(pending.spell),
        Zone::Graveyard => state.players[owner.index()].graveyard.push(pending.spell),
        Zone::Exile => state.exile.push(pending.spell),
        _ => unreachable!("origin_zone is always Hand, Graveyard, or Exile"),
    }
    state.objects.get_mut(pending.spell).zone = to_zone;
}

/// Pushes the ability's `StackItem`. If this ability's cost has a
/// `DiscardCards` component, the *entire* cost (including this component)
/// was already paid atomically back when that discard resolved
/// (`apply_discard`'s `DiscardResume::FinishActivation` arm -- see its doc
/// for why paying it there, not here, matters) -- paying again here would
/// double-tap/double-sacrifice. Only an ability with no discard component
/// at all (none in this pool, but not assumed away) still needs its cost
/// paid at this point, same as always.
fn finalize_activation(state: &mut GameState) {
    let pending = state.engine.pending_activation.take().expect("finalize_activation requires a pending activation");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];
    if discard_count_in(ability.cost).is_none() {
        pay_cost_components(state, pending.controller, pending.source, ability.cost, &[]);
    }

    let effect = (ability.effect)();
    state.stack.push(StackItem {
        source: pending.source,
        controller: pending.controller,
        targets: pending.targets_chosen,
        inline_effect: Some(effect),
        discarded: pending.cost_discard_paid.unwrap_or_default(),
        is_flashback: false,
        mode_chosen: 0,
    });

    // No ability in this increment's pool triggers off another ability
    // being activated, but see `finalize_cast`'s identical call for why
    // this has to happen before priority is handed back regardless.
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
}

/// Hand-or-graveyard -> Stack zone bookkeeping. Not routed through
/// `event::commit` (which explicitly panics on a Stack destination):
/// casting is an engine action, not a `MoveObject` effect leaf any card
/// program emits.
fn move_to_stack(state: &mut GameState, id: ObjectId, from_zone: Zone) {
    let owner = state.objects.get(id).owner;
    match from_zone {
        Zone::Graveyard => state.players[owner.index()].graveyard.retain(|&x| x != id),
        Zone::Exile => state.exile.retain(|&x| x != id),
        _ => state.players[owner.index()].hand.retain(|&x| x != id),
    }
    state.objects.get_mut(id).zone = Zone::Stack;
}

fn pay_plan(state: &mut GameState, player: PlayerId, plan: &mana::PaymentPlan) {
    for &(id, color) in &plan.taps {
        event::propose_and_commit(state, ProposedEvent::tap(id));
        event::propose_and_commit(state, ProposedEvent::mana_add(player, vec![color]));
    }
    // Spend: every newly-tapped mana is fully consumed by this cost by
    // construction (the solver only taps what it needs), plus whatever
    // floating pool the plan says to use.
    for &(_, color) in &plan.taps {
        state.players[player.index()].mana_pool[color.pool_index()] -= 1;
    }
    for (i, &amt) in plan.pool_used.iter().enumerate() {
        state.players[player.index()].mana_pool[i] -= amt;
    }
    state.players[player.index()].life -= plan.life_paid;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::card_id_by_name;
    use crate::effect::PlayerRef;

    fn empty_game() -> GameState {
        GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1)
    }

    fn put_on_battlefield(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id = card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
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

    fn put_in_hand(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id = card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
        });
        state.players[player.index()].hand.push(obj_id);
        obj_id
    }

    fn put_in_graveyard(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id = card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Graveyard,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
        });
        state.players[player.index()].graveyard.push(obj_id);
        obj_id
    }

    /// Fireblast's alternative cost (Sol #85: alt costs are payment
    /// *choices*) surfaces a real `Decision::ChooseCastMode` when both the
    /// printed mana cost and sacrificing 2 Mountains are legal, and (Sol
    /// #90, increment 11) a real `Decision::ChooseCostTargets` for *which*
    /// 2 of the 6 Mountains, asked one at a time with the already-picked
    /// one excluded from the second ask's candidates.
    #[test]
    fn fireblast_asks_to_choose_between_mana_cost_and_sacrificing_mountains() {
        let mut state = empty_game();
        for _ in 0..6 {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        let target = Target::Player(PlayerId::P1);
        step(&mut state, Action::ChooseTarget(target)).unwrap();

        match advance_until_decision(&mut state) {
            Decision::ChooseCastMode { player, spell, options } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(spell, fireblast);
                assert_eq!(options, vec![CastMode::Normal, CastMode::Alternative]);
            }
            other => panic!("expected ChooseCastMode, got {other:?}"),
        }

        step(&mut state, Action::ChooseCastMode(CastMode::Alternative)).unwrap();

        let first_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets { player, source, cost_kind, remaining, candidates } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, fireblast);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 2);
                assert_eq!(candidates.len(), 6, "all 6 Mountains are candidates before any pick");
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(first_pick)).unwrap();

        let second_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets { remaining, candidates, .. } => {
                assert_eq!(remaining, 1);
                assert_eq!(candidates.len(), 5, "the first pick is excluded from the second ask");
                assert!(!candidates.contains(&first_pick));
                candidates[0]
            }
            other => panic!("expected second ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(second_pick)).unwrap();

        advance_until_decision(&mut state); // drives the remaining cast stages (cost payment, stack push)
        // Alternative mode: exactly the 2 chosen Mountains sacrificed, no
        // mana tapped, and (since none were tapped) all 6 Mountains minus
        // the 2 sacrificed remain untapped.
        assert_eq!(state.players[0].graveyard.len(), 2, "should have sacrificed exactly 2 Mountains");
        assert!(state.players[0].graveyard.contains(&first_pick));
        assert!(state.players[0].graveyard.contains(&second_pick));
        assert_eq!(state.players[0].battlefield.len(), 4);
        assert!(state.players[0].battlefield.iter().all(|&id| !state.objects.get(id).tapped), "alt cost shouldn't tap any Mountain");
        assert_eq!(state.stack.len(), 1);
    }

    /// When only one of Fireblast's two cost paths is actually payable,
    /// there's no real choice of *mode* -- same "don't ask when there's
    /// only one legal answer" treatment `OrderTriggers` gets for a
    /// singleton group. But *which* Mountain still asks once: the
    /// reference logs a real decision for the first sacrifice pick even
    /// when the aggregate choice is forced (2 candidates for 2 needed),
    /// only going silent on the final pick once exactly 1 candidate
    /// remains (`sacrifice_lands_needed`'s doc / `drain_pending_cast_or_
    /// decide`'s per-pick auto-resolve comment -- root-caused against the
    /// v4 corpus's own candidate-count sequences).
    #[test]
    fn fireblast_auto_resolves_cast_mode_but_still_asks_the_first_sacrifice_pick() {
        let mut state = empty_game();
        // Only 2 Mountains: nowhere near {4}{R}{R}, but exactly enough to
        // sacrifice for the alt cost.
        let m1 = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let m2 = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();

        // No ChooseCastMode decision (only the alt cost is affordable),
        // but a real ChooseCostTargets for the first of the 2 Mountains --
        // both are legal candidates even though both will end up
        // sacrificed either way.
        let first_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCastMode { .. } => panic!("only the alt cost is affordable, so there's nothing to choose"),
            Decision::ChooseCostTargets { remaining, candidates, .. } => {
                assert_eq!(remaining, 2);
                let mut sorted = candidates.clone();
                sorted.sort_unstable();
                let mut expected = vec![m1, m2];
                expected.sort_unstable();
                assert_eq!(sorted, expected);
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(first_pick)).unwrap();

        // The second (and final) pick is now forced to the one remaining
        // Mountain -- silently auto-resolved, no second ChooseCostTargets.
        let decision = advance_until_decision(&mut state);
        assert!(!matches!(decision, Decision::ChooseCostTargets { .. }), "exactly 1 Mountain left for the last pick is no real choice");
        assert_eq!(state.players[0].graveyard.len(), 2);
        assert_eq!(state.stack.len(), 1);
    }

    /// Lava Dart's flashback cost (`FlashbackCost::SacrificeLands(1)`) is
    /// unconditional (no mana alternative to choose between first, unlike
    /// Fireblast) but still asks a real `Decision::ChooseCostTargets` for
    /// *which* Mountain when more than 1 is controlled.
    #[test]
    fn lava_dart_flashback_asks_which_mountain_to_sacrifice() {
        let mut state = empty_game();
        for _ in 0..3 {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        let lava_dart = put_in_graveyard(&mut state, PlayerId::P0, "Lava Dart");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(lava_dart)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();

        let pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets { player, source, cost_kind, remaining, candidates } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, lava_dart);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 1);
                assert_eq!(candidates.len(), 3);
                candidates[1]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(pick)).unwrap();

        advance_until_decision(&mut state); // drives the remaining cast stages (cost payment, stack push, exile-on-resolve)
        assert_eq!(state.players[0].graveyard.len(), 1, "the sacrificed Mountain, not Lava Dart itself (flashback exiles on resolution)");
        assert!(state.players[0].graveyard.contains(&pick));
        assert_eq!(state.players[0].battlefield.len(), 2);
        assert_eq!(state.stack.len(), 1);
    }

    /// Exactly 1 controlled Mountain for a 1-Mountain flashback cost is no
    /// real choice -- same auto-resolve shortcut as
    /// `fireblast_auto_resolves_to_the_only_affordable_mode`.
    #[test]
    fn lava_dart_flashback_auto_resolves_with_exactly_one_mountain() {
        let mut state = empty_game();
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let lava_dart = put_in_graveyard(&mut state, PlayerId::P0, "Lava Dart");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(lava_dart)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();

        let decision = advance_until_decision(&mut state);
        assert!(!matches!(decision, Decision::ChooseCostTargets { .. }), "exactly 1 Mountain for a 1-Mountain cost is no real choice");
        assert_eq!(state.players[0].graveyard.len(), 1);
        assert_eq!(state.players[0].battlefield.len(), 0);
        assert_eq!(state.stack.len(), 1);
    }

    #[test]
    fn illegal_action_returns_err_not_a_silent_noop() {
        let mut state = empty_game();
        let bogus = ObjectId(999);
        let err = step(&mut state, Action::PlayLand(bogus)).unwrap_err();
        assert!(err.contains("not a legal land drop"));
    }

    #[test]
    fn order_triggers_decision_exists_and_is_reachable() {
        let mut state = empty_game();
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(0),
            effect: EffectOp::GainLife { player: PlayerRef::Controller, amount: 1 },
        });
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(1),
            effect: EffectOp::GainLife { player: PlayerRef::Controller, amount: 2 },
        });

        let decision = advance_until_decision(&mut state);
        let pending = match decision {
            Decision::OrderTriggers { player, pending } => {
                assert_eq!(player, PlayerId::P0);
                pending
            }
            other => panic!("expected OrderTriggers, got {other:?}"),
        };
        assert_eq!(pending.len(), 2);

        // Choose to place them reversed: pending[1] (source ObjectId(1))
        // pushed first (bottom), pending[0] pushed last (top) -- so
        // ObjectId(0)'s trigger resolves first once the stack is popped.
        step(&mut state, Action::OrderTriggers(vec![1, 0])).unwrap();
        assert!(state.engine.pending_triggers.is_empty());
        assert_eq!(state.stack.len(), 2);
        assert_eq!(state.stack[0].source, ObjectId(1));
        assert_eq!(state.stack[1].source, ObjectId(0));
    }

    #[test]
    fn order_triggers_rejects_a_non_permutation() {
        let mut state = empty_game();
        state.engine.pending_triggers.push(PendingTrigger { controller: PlayerId::P0, source: ObjectId(0), effect: EffectOp::Sequence(vec![]) });
        state.engine.pending_triggers.push(PendingTrigger { controller: PlayerId::P0, source: ObjectId(1), effect: EffectOp::Sequence(vec![]) });
        let err = step(&mut state, Action::OrderTriggers(vec![0, 0])).unwrap_err();
        assert!(err.contains("permutation"));
    }

    #[test]
    fn lethal_damage_kills_creature_via_sba() {
        let mut state = empty_game();
        let guttersnipe = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        event::propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Object(guttersnipe), 2));
        trigger::sba_fixed_point(&mut state);
        assert_eq!(state.objects.get(guttersnipe).zone, Zone::Graveyard);
        assert!(!state.players[0].battlefield.contains(&guttersnipe));
        assert!(state.players[0].graveyard.contains(&guttersnipe));
    }

    #[test]
    fn zero_toughness_after_counters_also_dies() {
        // Sanity check on the toughness-<=0 leg of 704.5g independent of
        // damage marking.
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower"); // 1/1
        state.objects.get_mut(id).counters.plus1_plus1 = -1;
        trigger::sba_fixed_point(&mut state);
        assert_eq!(state.objects.get(id).zone, Zone::Graveyard);
    }

    #[test]
    fn until_end_of_turn_effects_are_cleared_at_cleanup() {
        let mut state = empty_game();
        state.engine.until_end_of_turn.push(UntilEndOfTurnEffect::SyntheticMarker(ObjectId(0)));
        run_step_entry_action(&mut state, Step::Cleanup);
        assert!(state.engine.until_end_of_turn.is_empty());
    }

    #[test]
    fn haste_creature_can_attack_the_turn_it_enters() {
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower");
        state.objects.get_mut(id).summoning_sick = true; // just entered
        state.active_player = PlayerId::P0;
        assert!(can_attack(&state, id), "haste creature should be able to attack despite summoning sickness");
    }

    #[test]
    fn non_haste_summoning_sick_creature_cannot_attack() {
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.objects.get_mut(id).summoning_sick = true;
        state.active_player = PlayerId::P0;
        assert!(!can_attack(&state, id));
    }

    #[test]
    fn flying_attacker_can_only_be_blocked_by_flying_or_reach() {
        let mut state = empty_game();
        let attacker = put_on_battlefield(&mut state, PlayerId::P0, "Sneaky Snacker"); // flying
        let _grounded_blocker = put_on_battlefield(&mut state, PlayerId::P1, "Guttersnipe");
        state.objects.get_mut(attacker).controller = PlayerId::P0;
        let legal = legal_blockers_for(&state, attacker);
        assert!(legal.is_empty(), "a non-flying, non-reach creature should not be able to block a flyer");
    }

    // ================================================================
    // Increment 7: Highway Robbery (+ Plot), Fiery Temper (Madness),
    // Searing Blaze, Pyroblast / Red Elemental Blast.
    // ================================================================

    fn put_on_stack(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id = card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
        });
        state.stack.push(StackItem { source: obj_id, controller: player, targets: vec![], inline_effect: None, discarded: vec![], is_flashback: false, mode_chosen: 0 });
        obj_id
    }

    /// Both players passing priority (117.3c: the caster keeps priority
    /// after finishing a cast, so this always starts with them) is what
    /// actually triggers `resolve_top_of_stack` -- finishing targeting
    /// alone only finalizes the cast and hands priority back. Repeatedly
    /// passes a `CastSpellOrPass` window as long as the stack is still
    /// non-empty, then returns whatever decision comes next -- which may
    /// itself be a resolution-triggered `Decision` (`ChooseOptionalCost`,
    /// `ChooseMadnessCast`) rather than another `CastSpellOrPass`, since a
    /// resolving spell's own effect can stage one synchronously.
    fn pass_until_stack_resolves(state: &mut GameState) -> Decision {
        loop {
            let decision = advance_until_decision(state);
            match &decision {
                Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => step(state, Action::Pass).unwrap(),
                _ => return decision,
            }
        }
    }

    /// Same idea, but stops as soon as the stack has shrunk by at least one
    /// item, instead of draining it completely -- for tests that put a
    /// second, synthetic (targetless) item under the one actually being
    /// exercised, which would panic if it were ever resolved for real.
    fn pass_until_one_stack_item_resolves(state: &mut GameState) {
        let starting_len = state.stack.len();
        loop {
            match advance_until_decision(state) {
                Decision::CastSpellOrPass { .. } => step(state, Action::Pass).unwrap(),
                other => panic!("unexpected decision while resolving one stack item: {other:?}"),
            }
            if state.stack.len() < starting_len {
                return;
            }
        }
    }

    fn ready_game_in_main1(p0_mountains: u32) -> GameState {
        let mut state = empty_game();
        for _ in 0..p0_mountains {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;
        state
    }

    #[test]
    fn highway_robbery_decline_draws_nothing() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        assert_eq!(state.players[0].hand.len(), 0, "Highway Robbery itself already left the hand once announced");

        match pass_until_stack_resolves(&mut state) {
            Decision::ChooseOptionalCost { player, discard_payable, sacrifice_payable } => {
                assert_eq!(player, PlayerId::P0);
                assert!(!discard_payable, "hand is empty (only Highway Robbery was in it)");
                assert!(sacrifice_payable, "2 Mountains are in play to sacrifice");
            }
            other => panic!("expected ChooseOptionalCost, got {other:?}"),
        }
        step(&mut state, Action::ChooseOptionalCost(OptionalCostChoice::Decline)).unwrap();
        assert_eq!(state.players[0].hand.len(), 0, "declining pays nothing and draws nothing");
        assert_eq!(state.players[0].battlefield.len(), 2, "declining doesn't sacrifice a land either");
    }

    #[test]
    fn highway_robbery_discard_draws_two() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        // 2 discardable cards left in hand (not just 1) so the discard is a
        // real choice -- `Decision::Discard` -- rather than silently
        // auto-resolving the same way a genuinely-forced 1-candidate
        // discard would (see `drain_pending_discard_or_decide`'s doc).
        let lava_dart = put_in_hand(&mut state, PlayerId::P0, "Lava Dart");
        let _lightning_bolt = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert!(matches!(decision, Decision::ChooseOptionalCost { discard_payable: true, .. }));
        step(&mut state, Action::ChooseOptionalCost(OptionalCostChoice::Discard)).unwrap();

        match advance_until_decision(&mut state) {
            Decision::Discard { player, count, choices } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(count, 1);
                assert_eq!(choices.len(), 2);
                assert!(choices.contains(&lava_dart));
                step(&mut state, Action::Discard(vec![lava_dart])).unwrap();
            }
            other => panic!("expected Discard, got {other:?}"),
        }
        // 608.2m: Highway Robbery moves to the graveyard only as the very
        // *last* part of its own resolution -- after the discard its own
        // "you may... if you do" text triggered -- so the discarded card
        // lands in the graveyard first, Highway Robbery itself second (see
        // `PendingOptionalCost::spell_resume`'s doc).
        assert_eq!(state.players[0].graveyard, vec![lava_dart, robbery], "the discarded card, then Highway Robbery itself, in that order");
        assert_eq!(state.players[0].hand.len(), 1, "the undiscarded Lightning Bolt is still in hand; the 2 drawn cards came from an empty library");
    }

    #[test]
    fn highway_robbery_sacrifice_land_draws_two() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        for i in 0..2 {
            state.draw_card(PlayerId::P0); // nothing to draw (empty library) -- just proves the count via hand growth below
            let _ = i;
        }
        state.players[0].hand.clear(); // isolate: only Highway Robbery matters for this test
        state.players[0].hand.push(robbery);

        step(&mut state, Action::CastSpell(robbery)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert!(matches!(decision, Decision::ChooseOptionalCost { sacrifice_payable: true, .. }));
        step(&mut state, Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand)).unwrap();

        // 2 Mountains for a 1-Mountain cost is a real choice (candidates >
        // 1) -- see `sacrifice_lands_needed`'s doc for the per-pick
        // auto-resolve rule this is *not* hitting.
        let picked = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets { player, source, cost_kind, remaining, candidates } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, robbery);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 1);
                assert_eq!(candidates.len(), 2);
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(picked)).unwrap();
        advance_until_decision(&mut state); // drives the sacrifice + "draw two" resolution

        assert_eq!(state.players[0].battlefield.len(), 1, "exactly 1 of the 2 Mountains should have been sacrificed");
        assert!(state.players[0].graveyard.contains(&picked));
        assert_eq!(state.players[0].graveyard.len(), 2, "Highway Robbery + the sacrificed Mountain");
    }

    #[test]
    fn plot_then_free_cast_on_a_later_turn() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");

        assert_eq!(plot_action_candidates(PlayerId::P0, &state), vec![robbery]);
        step(&mut state, Action::PlotSpell(robbery)).unwrap();
        assert_eq!(state.objects.get(robbery).zone, Zone::Exile);
        assert_eq!(state.objects.get(robbery).plotted_turn, Some(state.turn));
        assert_eq!(state.players[0].battlefield.iter().filter(|&&id| !state.objects.get(id).tapped).count(), 0, "Plot pays real mana");

        // Same turn: not castable yet (702.163a).
        assert!(!castable_spells(PlayerId::P0, &state).contains(&robbery));

        // A later turn, still Main1: castable for free.
        state.turn += 1;
        state.players[0].mana_pool = [0; 6];
        assert!(castable_spells(PlayerId::P0, &state).contains(&robbery), "Plotted card should be castable for free on a later turn");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, robbery);
        // Free: no Mountain should have been tapped for this cast (both were
        // already tapped paying the Plot cost and stay that way).
        assert_eq!(state.players[0].battlefield.iter().filter(|&&id| state.objects.get(id).tapped).count(), 2);
    }

    #[test]
    fn fiery_temper_discard_offers_madness_cast_for_r() {
        let mut state = ready_game_in_main1(1);
        let temper = put_in_hand(&mut state, PlayerId::P0, "Fiery Temper");
        state.engine.pending_discard = Some(PendingDiscard { player: PlayerId::P0, count: 1, resume: DiscardResume::None });
        step(&mut state, Action::Discard(vec![temper])).unwrap();
        assert_eq!(state.objects.get(temper).zone, Zone::Exile, "Madness exiles instead of graveyarding");
        assert!(state.players[0].graveyard.is_empty());

        match advance_until_decision(&mut state) {
            Decision::ChooseMadnessCast { player, card } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast, got {other:?}"),
        }
        step(&mut state, Action::ChooseMadnessCast(true)).unwrap();
        assert_eq!(state.stack.len(), 1, "the cast is announced (601.2a) before its cost is paid or its target chosen");
        assert!(!state.players[0].battlefield.iter().any(|&id| state.objects.get(id).tapped), "cost isn't paid until targeting finishes");

        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                step(&mut state, Action::ChooseTarget(target)).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        // `finalize_cast` (and hence cost payment) only actually runs on
        // the *next* pass through `advance_until_decision`'s loop, not
        // synchronously inside `step(ChooseTarget(..))` itself.
        pass_until_stack_resolves(&mut state);
        assert!(state.players[0].battlefield.iter().all(|&id| state.objects.get(id).tapped), "madness cost {{R}} should have tapped the only Mountain");
        assert_eq!(state.players[1].life, 17, "Fiery Temper still deals its normal 3 damage when cast via madness");
        assert_eq!(state.objects.get(temper).zone, Zone::Graveyard, "resolves to the graveyard as normal (madness doesn't change that)");
    }

    /// Root-caused against `game_20260713_002149_0004.txt` (decision 26) and
    /// `game_20260713_002156_0014.txt` (decision 45): the reference offers
    /// `ChooseMadnessCast` unconditionally -- no affordability pre-check --
    /// then lets the attempt begin a real cast (a genuine, trace-logged
    /// target pick) that only fails at cost payment, reverting the card to
    /// the graveyard. See `drain_pending_madness_or_decide`'s doc for the
    /// full citation; this replaces a prior (wrong) version of this test
    /// that asserted the opposite -- a silent, no-decision auto-resolve.
    #[test]
    fn fiery_temper_madness_attempt_fizzles_to_the_graveyard_when_unaffordable() {
        let mut state = empty_game(); // no Mountains: {R} is never payable
        let temper = put_in_hand(&mut state, PlayerId::P0, "Fiery Temper");
        state.engine.pending_discard = Some(PendingDiscard { player: PlayerId::P0, count: 1, resume: DiscardResume::None });
        step(&mut state, Action::Discard(vec![temper])).unwrap();
        assert_eq!(state.objects.get(temper).zone, Zone::Exile, "Madness exiles instead of graveyarding, same as the affordable case");

        match advance_until_decision(&mut state) {
            Decision::ChooseMadnessCast { player, card } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast (always offered, even when unaffordable), got {other:?}"),
        }
        step(&mut state, Action::ChooseMadnessCast(true)).unwrap();

        // The attempt proceeds through a real target pick, exactly like any
        // other cast -- this is the decision the reference's own
        // SELECT_TARGETS record captures for an ultimately-unpayable
        // attempt.
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { player, spell, legal_targets, .. } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(spell, temper);
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                step(&mut state, Action::ChooseTarget(target)).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        // Only now, at cost payment (no Mountain exists to tap), does the
        // attempt fail -- reverting to the graveyard (not back to exile,
        // and never landing on the stack/resolving).
        advance_until_decision(&mut state);
        assert_eq!(state.objects.get(temper).zone, Zone::Graveyard, "a failed madness attempt goes to the graveyard, not back to exile");
        assert!(state.stack.is_empty(), "the failed cast must not leave a stack item behind");
        assert_eq!(state.players[1].life, 20, "no damage: the cast never actually resolved");
    }

    #[test]
    fn searing_blaze_landfall_triples_damage_to_player_and_creature() {
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure"); // 2/1
        state.players[0].lands_played_this_turn = 1; // landfall satisfied

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(legal_targets, vec![Target::Object(victim)], "only P1's own creature is legal for the second target");
                step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.players[1].life, 17, "landfall: 3 damage to the player");
        assert_eq!(state.objects.get(victim).damage, 3, "landfall: 3 damage to the creature");
    }

    #[test]
    fn searing_blaze_deals_only_1_without_landfall() {
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.players[1].life, 19);
        assert_eq!(state.objects.get(victim).damage, 1);
    }

    #[test]
    fn searing_blaze_still_hits_the_player_when_the_creature_target_dies_first() {
        // 608.2b partial fizzle: the player target is always legal, so
        // Searing Blaze never fully fizzles in this pool -- only the
        // creature-damage leaf should be skipped once its target is gone.
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
        // The creature leaves the battlefield before Searing Blaze resolves
        // (simulating e.g. an in-response removal spell).
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(victim, Zone::Graveyard));
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.players[1].life, 19, "the player is still hit for the full amount");
    }

    #[test]
    fn pyroblast_counters_a_blue_spell_on_the_stack() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell"); // blue instant

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseSpellMode { mode_count, .. } => {
                assert_eq!(mode_count, 2);
                step(&mut state, Action::ChooseSpellMode(0)).unwrap(); // counter mode
            }
            other => panic!("expected ChooseSpellMode, got {other:?}"),
        }
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(counterspell)));
                step(&mut state, Action::ChooseTarget(Target::Object(counterspell))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.objects.get(counterspell).zone, Zone::Graveyard, "a blue spell should be countered (moved to its owner's graveyard)");
        assert_eq!(state.objects.get(pyroblast).zone, Zone::Graveyard, "Pyroblast itself resolves to the graveyard as normal");
    }

    #[test]
    fn pyroblast_does_not_counter_a_non_blue_spell() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let bolt = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt"); // red, not blue

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(bolt))).unwrap();
        // Only resolve Pyroblast itself (the top of the stack) -- `bolt` was
        // planted directly via `put_on_stack` with no real targets of its
        // own, so actually letting it resolve too would panic on an empty
        // `ctx.targets`, unrelated to what this test is checking.
        pass_until_one_stack_item_resolves(&mut state);
        assert_eq!(state.objects.get(bolt).zone, Zone::Stack, "Pyroblast's counter-mode targets any spell but only actually counters a blue one");
        assert_eq!(state.objects.get(pyroblast).zone, Zone::Graveyard, "Pyroblast itself still resolves normally even when its effect no-ops");
    }

    #[test]
    fn pyroblast_mode2_destroys_a_blue_permanent() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let serpent = put_on_battlefield(&mut state, PlayerId::P1, "Cryptic Serpent"); // blue creature

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(1)).unwrap(); // destroy mode
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(serpent)));
                step(&mut state, Action::ChooseTarget(Target::Object(serpent))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.objects.get(serpent).zone, Zone::Graveyard);
    }

    #[test]
    fn red_elemental_blast_targeting_is_prefiltered_to_blue_only() {
        let mut state = ready_game_in_main1(1);
        let reb = put_in_hand(&mut state, PlayerId::P0, "Red Elemental Blast");
        let bolt = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt"); // red: illegal target
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell"); // blue: legal target

        step(&mut state, Action::CastSpell(reb)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(!legal_targets.contains(&Target::Object(bolt)), "REB's counter mode should never even offer a non-blue spell as a target");
                assert!(legal_targets.contains(&Target::Object(counterspell)));
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
    }

    #[test]
    fn fizzle_when_the_targeted_spell_already_left_the_stack() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell");

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(counterspell))).unwrap();
        // The target resolves and leaves the stack before Pyroblast does
        // (simulated directly here; the kernel's real priority/stack
        // ordering can't actually reach this shape with this pool's cards,
        // but the guard should hold regardless of how it's reached).
        state.stack.retain(|item| item.source != counterspell);
        state.objects.get_mut(counterspell).zone = Zone::Graveyard;
        pass_until_stack_resolves(&mut state);
        // No crash, no double-move: Pyroblast itself still resolves to the
        // graveyard normally, and the fizzle guard is exercised via
        // EffectCond::TargetInZone, not a panic on a stale ObjectId.
        assert_eq!(state.objects.get(pyroblast).zone, Zone::Graveyard);
    }
}
