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
//! remaining (non-discard) costs and pushes the stack item.

use crate::card_def::{self, CardType, CostComponent, FlashbackCost, Keywords, TargetSpec};
use crate::effect::{self, EffectOp, ExecCtx, ObjectRef};
use crate::event::{self, ActiveReplacement, CommittedEvent, ProposedEvent};
use crate::ids::{ObjectId, PlayerId};
use crate::mana;
use crate::state::{GameState, Step, StackItem, Target, Zone};
use crate::trigger::{self, PendingTrigger};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngineState {
    /// Whether [P0, P1] has passed priority since the last time priority
    /// was reset (new step, a cast/activation/land-drop, or a resolution).
    pub priority_passes: [bool; 2],
    /// A spell that has begun casting but not yet finished being targeted,
    /// mode-chosen, or cost-paid (and therefore not yet on the stack).
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscardResume {
    /// Nothing further to do once the discard lands (Faithless Looting's
    /// resolution-effect discard; cleanup's discard-to-7).
    None,
    /// Write the discarded cards back into `EngineState::pending_cast`'s
    /// `additional_cost_discarded` and let the cast staging continue.
    FinishCast,
    /// Same, but for `EngineState::pending_activation`'s
    /// `cost_discard_paid`.
    FinishActivation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingDiscard {
    pub player: PlayerId,
    pub count: u32,
    pub resume: DiscardResume,
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
        /// Both hand-castable spells and graveyard flashback-castable
        /// cards (see `CardDef::flashback`); `step()` tells them apart by
        /// the object's current zone.
        castable_spells: Vec<ObjectId>,
        mana_abilities: Vec<ObjectId>,
        land_drops: Vec<ObjectId>,
        /// (source, ability_index) pairs for every non-mana activated
        /// ability currently affordable (Masked Meower's, the Blood
        /// token's).
        activatable_abilities: Vec<(ObjectId, u8)>,
    },
    ChooseTargets {
        player: PlayerId,
        /// The spell or (non-mana-ability) source this targeting belongs
        /// to.
        spell: ObjectId,
        remaining: u8,
        legal_targets: Vec<Target>,
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
    /// with. Always asked whenever `eligible` is non-empty, even if the
    /// only sane answer is the empty set -- no auto-pass.
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
        TargetSpec::AnyTarget => 1,
    }
}

pub fn legal_targets_for(spec: TargetSpec, state: &GameState) -> Vec<Target> {
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

fn count_controlled_lands(player: PlayerId, state: &GameState) -> u32 {
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
/// `EngineState::pending_discard`'s doc).
fn pay_cost_components(state: &mut GameState, player: PlayerId, source: ObjectId, components: &[CostComponent]) {
    for c in components {
        match c {
            CostComponent::Tap => event::propose_and_commit(state, ProposedEvent::tap(source)),
            CostComponent::SacrificeSelf => event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Graveyard)),
            CostComponent::ExileSelf => event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Exile)),
            CostComponent::SacrificeLands(n) => sacrifice_lowest_id_lands(state, player, *n),
            CostComponent::Mana(cost) => {
                let plan = mana::can_pay(cost, 0, player, state).expect("legality already checked by can_pay_components");
                pay_plan(state, player, &plan);
            }
            CostComponent::DiscardCards(_) => {}
        }
    }
}

/// Sacrifices the `n` lowest-`ObjectId` lands `player` controls. Which
/// specific lands are picked is not a real decision in this pool (every
/// land is a Mountain, fully interchangeable) -- same "auto-solve, don't
/// ask" treatment `mana::solve` gives ordinary tap sources.
fn sacrifice_lowest_id_lands(state: &mut GameState, player: PlayerId, n: u8) {
    let mut lands: Vec<ObjectId> = state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land)
        .collect();
    lands.sort_unstable();
    for &id in lands.iter().take(n as usize) {
        event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
    }
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
        player == state.active_player && state.stack.is_empty() && matches!(state.step, Step::Main1 | Step::Main2)
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
    out
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

fn has_eligible_attacker(state: &GameState) -> bool {
    state.players[state.active_player.index()].battlefield.iter().any(|&id| can_attack(state, id))
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

        if let Some(d) = drain_pending_cast_or_decide(state) {
            return d;
        }

        if let Some(d) = drain_pending_activation_or_decide(state) {
            return d;
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
        };
    }
}

fn reset_priority(state: &mut GameState) {
    state.engine.priority_passes = [false, false];
    state.priority_player = state.active_player;
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
        event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
    }
    match resume {
        DiscardResume::None => collect_and_queue_triggers(state),
        DiscardResume::FinishCast => {
            if let Some(p) = state.engine.pending_cast.as_mut() {
                p.additional_cost_discarded = Some(chosen);
            }
        }
        DiscardResume::FinishActivation => {
            if let Some(p) = state.engine.pending_activation.as_mut() {
                p.cost_discard_paid = Some(chosen);
            }
        }
    }
}

/// Stages `PendingCast` through its targets -> cast-mode -> additional-
/// cost-discard -> finalize pipeline, one stage per call (each stage that
/// makes progress `continue`s the outer loop instead of looping here, so
/// `pending_discard`/triggers/etc. staged along the way always get
/// checked first).
fn drain_pending_cast_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_cast.clone()?;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];

    let need = target_count(pending.target_spec);
    if (pending.targets_chosen.len() as u8) < need {
        return Some(Decision::ChooseTargets {
            player: pending.controller,
            spell: pending.spell,
            remaining: need - pending.targets_chosen.len() as u8,
            legal_targets: legal_targets_for(pending.target_spec, state),
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
            legal_targets: legal_targets_for(pending.target_spec, state),
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
    });
    reset_priority(state);
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
    if let Some(program) = (def.spell_effect)() {
        effect::execute(&program, &ctx, state);
    }

    // 608.2m: instants/sorceries go to the graveyard as the last part of
    // resolution -- or to exile instead, if this was a flashback cast
    // (702.10e). Creatures/artifacts/enchantments handle entering the
    // battlefield themselves, via their own MoveObject effect.
    if def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery) {
        let to_zone = if item.is_flashback { Zone::Exile } else { Zone::Graveyard };
        event::propose_and_commit(state, ProposedEvent::zone_change(item.source, to_zone));
    }
}

/// Moves `state.step`/`state.active_player`/`state.turn` to the next step,
/// running that step's turn-based entry action (untap, draw, cleanup,
/// combat damage) and resetting priority. Skips the declare-attackers/
/// blockers/combat-damage steps when there's no possible attack (508.1/
/// 508.7): entirely, if no eligible attacker exists at all; past declare-
/// blockers/combat-damage, if attackers were declared but the set is
/// empty.
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
    if next == Step::DeclareAttackers && !has_eligible_attacker(state) {
        next = Step::EndCombat;
    }
    if next == Step::DeclareBlockers && state.engine.combat.attackers.is_empty() {
        next = Step::EndCombat;
    }

    state.step = next;
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
            let spec = pending_target_spec(state).ok_or("no spell or ability is currently being targeted")?;
            if !legal_targets_for(spec, state).contains(&t) {
                return Err(format!("{t:?} is not a legal target"));
            }
            if let Some(p) = state.engine.pending_cast.as_mut() {
                p.targets_chosen.push(t);
            } else if let Some(p) = state.engine.pending_activation.as_mut() {
                p.targets_chosen.push(t);
            }
            Ok(())
        }
        Action::ChooseCastMode(mode) => {
            let pending = state.engine.pending_cast.as_mut().ok_or("no spell is currently being cast")?;
            if pending.cast_mode.is_some() {
                return Err("this cast's mode has already been chosen".to_string());
            }
            pending.cast_mode = Some(mode);
            Ok(())
        }
        Action::Discard(chosen) => apply_discard_action(state, chosen),
        Action::DeclareAttackers(attackers) => apply_declare_attackers(state, attackers),
        Action::DeclareBlockers(blocks) => apply_declare_blockers(state, blocks),
        Action::OrderTriggers(perm) => apply_order_triggers(state, perm),
    }
}

fn pending_target_spec(state: &GameState) -> Option<TargetSpec> {
    if let Some(p) = &state.engine.pending_cast {
        return Some(p.target_spec);
    }
    if let Some(p) = &state.engine.pending_activation {
        return Some(p.target_spec);
    }
    None
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

/// 601.2c-601.2h: targets are chosen before costs are paid. `begin_cast`
/// records the pending cast (pre-resolving the cast-mode/additional-cost
/// stages when there's no real choice to make -- see `PendingCast`'s
/// field docs); `drain_pending_cast_or_decide` walks the remaining stages.
fn begin_cast(state: &mut GameState, player: PlayerId, spell_id: ObjectId) {
    let is_flashback = state.objects.get(spell_id).zone == Zone::Graveyard;
    let def = &card_def::CARD_DEFS[state.objects.get(spell_id).card_def as usize];
    state.engine.pending_cast = Some(PendingCast {
        spell: spell_id,
        controller: player,
        target_spec: def.target_spec,
        targets_chosen: vec![],
        is_flashback,
        cast_mode: if is_flashback || def.alt_cost.is_none() { Some(CastMode::Normal) } else { None },
        additional_cost_discarded: if def.additional_cost.is_none() { Some(vec![]) } else { None },
    });
}

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
/// moves the spell from hand-or-graveyard onto the stack. 117.3c: the
/// caster retains priority afterward.
fn finalize_cast(state: &mut GameState) {
    let pending = state.engine.pending_cast.take().expect("finalize_cast requires a pending cast");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];

    if pending.is_flashback {
        let fb = def.flashback.as_ref().expect("is_flashback implies CardDef::flashback is Some");
        match fb.cost {
            FlashbackCost::Mana(cost) => {
                let plan = mana::can_pay(&cost, 0, pending.controller, state).expect("castable_spells already verified affordability");
                pay_plan(state, pending.controller, &plan);
            }
            FlashbackCost::SacrificeLands(n) => sacrifice_lowest_id_lands(state, pending.controller, n),
        }
    } else {
        match pending.cast_mode.expect("resolved by drain_pending_cast_or_decide before finalize_cast is reached") {
            CastMode::Normal => {
                let plan = mana::can_pay(&def.cost, 0, pending.controller, state).expect("castable_spells already verified affordability before begin_cast");
                pay_plan(state, pending.controller, &plan);
            }
            CastMode::Alternative => {
                let alt = def.alt_cost.expect("Alternative mode only chosen when alt_cost is Some");
                pay_cost_components(state, pending.controller, pending.spell, alt);
            }
        }
    }
    if let Some(add) = def.additional_cost {
        pay_cost_components(state, pending.controller, pending.spell, add);
    }

    let discarded = pending.additional_cost_discarded.unwrap_or_default();
    move_to_stack(state, pending.spell, pending.is_flashback);
    event::log_spell_cast(state, pending.spell, pending.controller);
    state.stack.push(StackItem {
        source: pending.spell,
        controller: pending.controller,
        targets: pending.targets_chosen,
        inline_effect: None,
        discarded,
        is_flashback: pending.is_flashback,
    });

    // 601.2i/603.3: casting is complete the instant the spell is on the
    // stack, costs are paid, etc. -- triggered abilities that saw it
    // happen (Guttersnipe) go on the stack *before* anyone gets priority
    // again, same as `play_land`'s land-drop trigger check.
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
}

fn finalize_activation(state: &mut GameState) {
    let pending = state.engine.pending_activation.take().expect("finalize_activation requires a pending activation");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];
    pay_cost_components(state, pending.controller, pending.source, ability.cost);

    let effect = (ability.effect)();
    state.stack.push(StackItem {
        source: pending.source,
        controller: pending.controller,
        targets: pending.targets_chosen,
        inline_effect: Some(effect),
        discarded: pending.cost_discard_paid.unwrap_or_default(),
        is_flashback: false,
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
fn move_to_stack(state: &mut GameState, id: ObjectId, from_graveyard: bool) {
    let owner = state.objects.get(id).owner;
    if from_graveyard {
        state.players[owner.index()].graveyard.retain(|&x| x != id);
    } else {
        state.players[owner.index()].hand.retain(|&x| x != id);
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
        });
        state.players[player.index()].hand.push(obj_id);
        obj_id
    }

    /// Fireblast's alternative cost (Sol #85: alt costs are payment
    /// *choices*) surfaces a real `Decision::ChooseCastMode` when both the
    /// printed mana cost and sacrificing 2 Mountains are legal.
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
        advance_until_decision(&mut state); // drives the remaining cast stages (cost payment, stack push)
        // Alternative mode: 2 Mountains sacrificed, no mana tapped, and
        // (since none were tapped) all 6 Mountains minus the 2 sacrificed
        // remain untapped.
        assert_eq!(state.players[0].graveyard.len(), 2, "should have sacrificed exactly 2 Mountains");
        assert_eq!(state.players[0].battlefield.len(), 4);
        assert!(state.players[0].battlefield.iter().all(|&id| !state.objects.get(id).tapped), "alt cost shouldn't tap any Mountain");
        assert_eq!(state.stack.len(), 1);
    }

    /// When only one of Fireblast's two cost paths is actually payable,
    /// there's no real choice -- same "don't ask when there's only one
    /// legal answer" treatment `OrderTriggers` gets for a singleton group.
    #[test]
    fn fireblast_auto_resolves_to_the_only_affordable_mode() {
        let mut state = empty_game();
        // Only 2 Mountains: nowhere near {4}{R}{R}, but exactly enough to
        // sacrifice for the alt cost.
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();

        // No ChooseCastMode decision: straight through to the spell
        // landing on the stack, paid via the alt cost.
        let decision = advance_until_decision(&mut state);
        assert!(!matches!(decision, Decision::ChooseCastMode { .. }), "only the alt cost is affordable, so there's nothing to choose");
        assert_eq!(state.players[0].graveyard.len(), 2);
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
}
