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
//! active replacements, queued triggers) lives in `GameState::engine`
//! ([`EngineState`]) so both functions can stay pure `&mut GameState`
//! signatures with no separate engine object.
//!
//! Card-effect mutation still only ever happens through
//! `event::propose_and_commit` (see `effect.rs`/`event.rs`); everything in
//! this module that mutates `GameState` directly is turn/stack/priority
//! bookkeeping, not card behavior.

use crate::card_def::{self, CardType, TargetSpec};
use crate::effect::{self, EffectOp, ExecCtx, ObjectRef};
use crate::event::{self, ActiveReplacement, CommittedEvent};
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
    /// A spell that has begun casting but not yet finished being targeted
    /// (and therefore not yet paid for or placed on the stack).
    pub pending_cast: Option<PendingCast>,
    /// Transient buffer for the *current* resolution: `event::commit`
    /// appends here, `trigger::collect_and_process` drains it after every
    /// resolution to match triggers. Empty between resolutions.
    pub event_log: Vec<CommittedEvent>,
    /// Full permanent record of every committed event this game, in
    /// commit order. Never drained; this is what
    /// `event::commit`/`trigger::collect_and_process`'s draining of
    /// `event_log` would otherwise make unobservable after the fact (game
    /// replay / RL trace logging / the acceptance test's event-log
    /// assertions all read this instead).
    pub event_history: Vec<CommittedEvent>,
    pub active_replacements: Vec<ActiveReplacement>,
    pub next_replacement_id: u32,
    /// Triggered abilities collected but not yet placed on the stack,
    /// grouped APNAP (active player's group first); see
    /// `drain_pending_triggers_or_decide`.
    pub pending_triggers: Vec<PendingTrigger>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingCast {
    pub spell: ObjectId,
    pub controller: PlayerId,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    CastSpellOrPass {
        player: PlayerId,
        castable_spells: Vec<ObjectId>,
        mana_abilities: Vec<ObjectId>,
        land_drops: Vec<ObjectId>,
    },
    ChooseTargets {
        player: PlayerId,
        spell: ObjectId,
        remaining: u8,
        legal_targets: Vec<Target>,
    },
    /// Stub per the design brief: fixed APNAP grouping always happens;
    /// this decision only fires when one player controls 2+ simultaneous
    /// triggers and must choose an order for them (603.3b). No card in
    /// this increment's pool triggers, so it's unreachable from the
    /// acceptance goldfish; see `tests::order_triggers_decision_exists`
    /// for a synthetic proof it works.
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
    Pass,
    ChooseTarget(Target),
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

fn castable_spells(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    let mut out = Vec::new();
    for &id in &state.players[player.index()].hand {
        let obj = state.objects.get(id);
        let def = &card_def::CARD_DEFS[obj.card_def as usize];
        if !def.is_castable() {
            continue;
        }
        let sorcery_speed_ok = if def.has_type(CardType::Sorcery) || def.has_type(CardType::Creature) {
            player == state.active_player && state.stack.is_empty() && matches!(state.step, Step::Main1 | Step::Main2)
        } else {
            true // instants: castable any time the caster has priority
        };
        if !sorcery_speed_ok {
            continue;
        }
        if mana::can_pay(&def.cost, 0, player, state).is_none() {
            continue;
        }
        out.push(id);
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

fn has_eligible_attacker(state: &GameState) -> bool {
    state.players[state.active_player.index()].battlefield.iter().any(|&id| {
        let obj = state.objects.get(id);
        card_def::CARD_DEFS[obj.card_def as usize].has_type(CardType::Creature) && !obj.tapped && !obj.summoning_sick
    })
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

        if let Some(pending) = &state.engine.pending_cast {
            let need = target_count(pending.target_spec);
            if (pending.targets_chosen.len() as u8) < need {
                return Decision::ChooseTargets {
                    player: pending.controller,
                    spell: pending.spell,
                    remaining: need - pending.targets_chosen.len() as u8,
                    legal_targets: legal_targets_for(pending.target_spec, state),
                };
            }
            finalize_cast(state);
            continue;
        }

        if let Some(d) = drain_pending_triggers_or_decide(state) {
            return d;
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
    state.stack.push(StackItem { source: t.source, controller: t.controller, targets: vec![], trigger_effect: Some(t.effect) });
    reset_priority(state);
}

/// 704.5g/h creature death, 704.5a life-loss, 704.5c empty-draw-loss all
/// happen inside `resolve_top_of_stack`'s `collect_and_queue_triggers`
/// call; this just pops and executes.
fn resolve_top_of_stack(state: &mut GameState) {
    let item = state.stack.pop().expect("resolve_top_of_stack called with an empty stack");
    let ctx = ExecCtx { source: item.source, controller: item.controller, targets: item.targets };

    if let Some(effect) = item.trigger_effect {
        effect::execute(&effect, &ctx, state);
        return;
    }

    let card_def_idx = state.objects.get(item.source).card_def;
    let def = &card_def::CARD_DEFS[card_def_idx as usize];
    if let Some(program) = (def.spell_effect)() {
        effect::execute(&program, &ctx, state);
    }

    // 608.2m: instants/sorceries go to the graveyard as the last part of
    // resolution. Creatures/artifacts/enchantments handle entering the
    // battlefield themselves, via their own MoveObject effect.
    if def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery) {
        event::propose_and_commit(state, event::ProposedEvent::zone_change(item.source, Zone::Graveyard));
    }
}

/// Moves `state.step`/`state.active_player`/`state.turn` to the next step,
/// running that step's turn-based entry action (untap, draw) and resetting
/// priority. Skips the declare-attackers/blockers/combat-damage steps
/// entirely when there is no possible attacker (507.1/508.1); panics if
/// there *is* one, since attacking/blocking decisions are the next
/// increment's scope, not this one's.
fn advance_step(state: &mut GameState) {
    let cur_idx = STEP_ORDER.iter().position(|&s| s == state.step).expect("state.step is always a STEP_ORDER member");
    let mut next_idx = cur_idx + 1;

    if next_idx >= STEP_ORDER.len() {
        run_cleanup(state);
        state.active_player = state.active_player.opponent();
        if state.active_player == PlayerId::P0 {
            state.turn += 1;
        }
        next_idx = 0;
    }

    let mut next = STEP_ORDER[next_idx];
    if next == Step::DeclareAttackers {
        if has_eligible_attacker(state) {
            panic!("next increment: combat (declare attackers)");
        }
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
                event::propose_and_commit(state, event::ProposedEvent::draw(p));
                collect_and_queue_triggers(state);
            }
        }
        _ => {}
    }
}

/// 514: reset damage, discard to hand size (panics past this increment's
/// scope if it would ever fire -- no test exercises it), reset the land
/// drop counter for the player whose turn just ended.
fn run_cleanup(state: &mut GameState) {
    let p = state.active_player;

    for (_, obj) in state.objects.iter_mut() {
        obj.damage = 0;
    }

    if state.players[p.index()].hand.len() > 7 {
        panic!("next increment: cleanup discard (hand size > 7)");
    }

    state.players[p.index()].lands_played_this_turn = 0;
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
        Action::CastSpell(id) => {
            let p = state.priority_player;
            if !castable_spells(p, state).contains(&id) {
                return Err(format!("{id} is not castable by {p:?} right now"));
            }
            begin_cast(state, p, id);
            Ok(())
        }
        Action::ChooseTarget(t) => {
            let spec = state
                .engine
                .pending_cast
                .as_ref()
                .ok_or("no spell is currently being cast")?
                .target_spec;
            if !legal_targets_for(spec, state).contains(&t) {
                return Err(format!("{t:?} is not a legal target"));
            }
            state.engine.pending_cast.as_mut().unwrap().targets_chosen.push(t);
            Ok(())
        }
        Action::OrderTriggers(perm) => apply_order_triggers(state, perm),
    }
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
/// records the pending cast; `advance_until_decision` detects it needs
/// targets (or, for a 0-target spell, needs none) and either asks via
/// `Decision::ChooseTargets` or calls `finalize_cast` immediately.
fn begin_cast(state: &mut GameState, player: PlayerId, spell_id: ObjectId) {
    let def_idx = state.objects.get(spell_id).card_def;
    let target_spec = card_def::CARD_DEFS[def_idx as usize].target_spec;
    state.engine.pending_cast = Some(PendingCast { spell: spell_id, controller: player, target_spec, targets_chosen: vec![] });
}

/// Pays the cost (auto-solved via the exact backtracking mana solver
/// against currently-untapped sources + floating pool) and moves the spell
/// from hand onto the stack. 117.3c: the caster retains priority
/// afterward.
fn finalize_cast(state: &mut GameState) {
    let pending = state.engine.pending_cast.take().expect("finalize_cast requires a pending cast");
    let def_idx = state.objects.get(pending.spell).card_def;
    let cost = card_def::CARD_DEFS[def_idx as usize].cost;
    let plan = mana::can_pay(&cost, 0, pending.controller, state)
        .expect("castable_spells already verified affordability before begin_cast");
    pay_plan(state, pending.controller, &plan);

    move_hand_to_stack(state, pending.spell);
    state.stack.push(StackItem {
        source: pending.spell,
        controller: pending.controller,
        targets: pending.targets_chosen,
        trigger_effect: None,
    });

    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
}

/// Hand -> Stack zone bookkeeping. Not routed through `event::commit`
/// (which explicitly panics on a Stack destination): casting is an engine
/// action, not a `MoveObject` effect leaf any card program emits.
fn move_hand_to_stack(state: &mut GameState, id: ObjectId) {
    let owner = state.objects.get(id).owner;
    state.players[owner.index()].hand.retain(|&x| x != id);
    state.objects.get_mut(id).zone = Zone::Stack;
}

fn pay_plan(state: &mut GameState, player: PlayerId, plan: &mana::PaymentPlan) {
    for &(id, color) in &plan.taps {
        event::propose_and_commit(state, event::ProposedEvent::tap(id));
        event::propose_and_commit(state, event::ProposedEvent::mana_add(player, vec![color]));
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
            effect: EffectOp::GainLife { player: crate::effect::PlayerRef::Controller, amount: 1 },
        });
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(1),
            effect: EffectOp::GainLife { player: crate::effect::PlayerRef::Controller, amount: 2 },
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
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(0),
            effect: EffectOp::Sequence(vec![]),
        });
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(1),
            effect: EffectOp::Sequence(vec![]),
        });
        let err = step(&mut state, Action::OrderTriggers(vec![0, 0])).unwrap_err();
        assert!(err.contains("permutation"));
    }

    #[test]
    fn lethal_damage_kills_creature_via_sba() {
        let mut state = empty_game();
        let guttersnipe = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        event::propose_and_commit(&mut state, event::ProposedEvent::damage(ObjectId(0), Target::Object(guttersnipe), 2));
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
}