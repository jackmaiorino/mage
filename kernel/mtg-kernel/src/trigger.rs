//! State-based actions and trigger collection.
//!
//! After every resolution (a spell/ability resolving, a land entering, a
//! turn-based draw -- anything that produced `CommittedEvent`s), the engine
//! calls `collect_and_process`: it drains the event log, matches triggered
//! abilities, runs SBAs to a fixed point, and returns any newly-pending
//! triggers in APNAP order for `engine.rs` to place on the stack (or, if
//! 2+ share a controller, to ask that controller to order via
//! `engine::Decision::OrderTriggers`).

use crate::effect::{EffectCond, EffectOp, PlayerRef, TargetRef};
use crate::event::CommittedEvent;
use crate::ids::{ObjectId, PlayerId};
use crate::state::{GameState, Zone};
use serde::{Deserialize, Serialize};

/// Trigger conditions this increment's kernel can match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerCondition {
    /// The permanent itself enters the battlefield (Voldaren Epicure).
    Etb,
    /// The permanent itself deals damage. Unused by any Burn 16 card this
    /// increment (kept from increment 2's shape -- see `trigger_matches`).
    DealsDamage,
    /// The controller casts an instant or sorcery spell -- any such
    /// spell, not just ones this permanent's controller controls the
    /// *source* of (Guttersnipe: "whenever you cast an instant or sorcery
    /// spell"). Matched against `CommittedEvent::SpellCast`, logged by
    /// `engine::finalize_cast`.
    CastInstantOrSorcery,
    /// The controller draws their `n`th card since the current turn began
    /// (Sneaky Snacker: "whenever you draw your third card in a turn").
    /// This ability functions from the graveyard (`home_zone: Graveyard`
    /// on its `TriggeredAbilityDef`), so unlike every other condition
    /// here it's checked with the source in the graveyard, not the
    /// battlefield.
    DrawNth(u32),
    /// This permanent moves from the battlefield to its owner's graveyard
    /// (700.4 "dies", generalized to any battlefield -> graveyard
    /// transition regardless of card type -- Clockwork Percussionist's
    /// death and Experimental Synthesizer's own sacrifice-cost ability both
    /// have this exact event shape, and neither card's Java source
    /// distinguishes "dies" from "put into a graveyard from the
    /// battlefield" in a way this pool needs to model separately). Checked
    /// with the source *currently in the graveyard* (`home_zone: Graveyard`
    /// -- see `Etb`'s doc for why the outer gate uses the post-event zone,
    /// not a "functions from" zone: by the time `collect_and_process` looks,
    /// the object has already moved there).
    LeftBattlefieldToGraveyard,
}

pub struct TriggeredAbilityDef {
    pub condition: TriggerCondition,
    /// Which zone the source must be in for this ability to function.
    /// Every keyworded/triggered ability in MTG works from the
    /// battlefield unless its reminder text says otherwise (like Sneaky
    /// Snacker's "from your graveyard").
    pub home_zone: Zone,
    /// 603.4 intervening-if gate checked when the event happens. Goblin
    /// Bushwhacker is the only card in this pool with one: an unkicked
    /// Bushwhacker must not create a stack item at all (as opposed to
    /// creating a trigger whose effect later resolves to a no-op).
    pub intervening_if_kicked: bool,
    pub effect: fn() -> EffectOp,
}

fn guttersnipe_effect() -> EffectOp {
    // Guttersnipe deals 2 damage to each opponent.
    EffectOp::DealDamage { target: TargetRef::Opponent, amount: 2 }
}

fn voldaren_epicure_effect() -> EffectOp {
    // It deals 1 damage to each opponent. Create a Blood token.
    let blood_token = crate::card_def::card_id_by_name("Blood Token").expect("Blood Token in CARD_DEFS");
    EffectOp::Sequence(vec![
        EffectOp::DealDamage { target: TargetRef::Opponent, amount: 1 },
        EffectOp::CreateToken { token_def: blood_token, controller: PlayerRef::Controller },
    ])
}

fn sneaky_snacker_effect() -> EffectOp {
    // Return Sneaky Snacker from your graveyard to the battlefield tapped.
    EffectOp::Sequence(vec![
        EffectOp::MoveObject { object: crate::effect::ObjectRef::ThisSource, to_zone: Zone::Battlefield },
        EffectOp::TapObject { object: crate::effect::ObjectRef::ThisSource },
    ])
}

const GUTTERSNIPE_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::CastInstantOrSorcery, home_zone: Zone::Battlefield, intervening_if_kicked: false, effect: guttersnipe_effect }];
const VOLDAREN_EPICURE_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::Etb, home_zone: Zone::Battlefield, intervening_if_kicked: false, effect: voldaren_epicure_effect }];
const SNEAKY_SNACKER_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::DrawNth(3), home_zone: Zone::Graveyard, intervening_if_kicked: false, effect: sneaky_snacker_effect }];

fn burning_tree_emissary_effect() -> EffectOp {
    // When Burning-Tree Emissary enters the battlefield, add {R}{G}.
    EffectOp::AddMana { player: PlayerRef::Controller, colors: vec![crate::mana::ManaColor::R, crate::mana::ManaColor::G] }
}

fn clockwork_percussionist_dies_effect() -> EffectOp {
    // When Clockwork Percussionist dies, exile the top card of your
    // library. You may play it until the end of your next turn.
    EffectOp::ImpulseDraw { count: 1, duration: crate::effect::ImpulseDuration::UntilOwnersNextTurn }
}

fn experimental_synthesizer_impulse_effect() -> EffectOp {
    // When Experimental Synthesizer enters or leaves the battlefield, exile
    // the top card of your library. Until end of turn, you may play that
    // card. Both triggers share this one effect -- the card's text is
    // identical either way.
    EffectOp::ImpulseDraw { count: 1, duration: crate::effect::ImpulseDuration::EndOfTurn }
}

fn goblin_bushwhacker_effect() -> EffectOp {
    // When this creature enters, if it was kicked, creatures you control
    // get +1/+0 and gain haste until end of turn.
    EffectOp::Conditional {
        cond: EffectCond::WasKicked,
        then: Box::new(EffectOp::PumpControlled {
            filter: crate::effect::CreatureFilter::AnyControlled,
            power: 1,
            toughness: 0,
            grant_haste: true,
        }),
        else_: Box::new(EffectOp::Sequence(vec![])),
    }
}

const BURNING_TREE_EMISSARY_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::Etb, home_zone: Zone::Battlefield, intervening_if_kicked: false, effect: burning_tree_emissary_effect }];
const CLOCKWORK_PERCUSSIONIST_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::LeftBattlefieldToGraveyard,
    home_zone: Zone::Graveyard,
    intervening_if_kicked: false,
    effect: clockwork_percussionist_dies_effect,
}];
const EXPERIMENTAL_SYNTHESIZER_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef { condition: TriggerCondition::Etb, home_zone: Zone::Battlefield, intervening_if_kicked: false, effect: experimental_synthesizer_impulse_effect },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefieldToGraveyard,
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        effect: experimental_synthesizer_impulse_effect,
    },
];
const GOBLIN_BUSHWHACKER_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::Etb, home_zone: Zone::Battlefield, intervening_if_kicked: true, effect: goblin_bushwhacker_effect }];

/// The Burn 16's and Mono Red Rally's real triggered abilities, matched by
/// card name (ids are codegen-assigned from `cards_v1.json`'s array order
/// and not worth duplicating as constants here -- see `build.rs`'s module
/// doc on id stability). Every other card in the pool has no triggered
/// ability implemented and falls through to `&[]`.
pub fn triggers_for(card_def: u16) -> &'static [TriggeredAbilityDef] {
    match crate::card_def::CARD_DEFS[card_def as usize].name {
        "Guttersnipe" => &GUTTERSNIPE_TRIGGERS,
        "Voldaren Epicure" => &VOLDAREN_EPICURE_TRIGGERS,
        "Sneaky Snacker" => &SNEAKY_SNACKER_TRIGGERS,
        "Burning-Tree Emissary" => &BURNING_TREE_EMISSARY_TRIGGERS,
        "Clockwork Percussionist" => &CLOCKWORK_PERCUSSIONIST_TRIGGERS,
        "Experimental Synthesizer" => &EXPERIMENTAL_SYNTHESIZER_TRIGGERS,
        "Goblin Bushwhacker" => &GOBLIN_BUSHWHACKER_TRIGGERS,
        _ => &[],
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingTrigger {
    pub controller: PlayerId,
    pub source: ObjectId,
    pub effect: EffectOp,
    /// True iff this is a Madness triggered-ability offer (`engine::
    /// apply_discard`'s Madness branch), not one of this module's
    /// card-def-matched triggers (`triggers_for`). Threaded through the
    /// same APNAP grouping/`Decision::OrderTriggers` machinery as any other
    /// trigger (603.3b makes no distinction), but `effect` is a meaningless
    /// placeholder (`EffectOp::Sequence(vec![])`) for one of these --
    /// `engine::push_trigger_onto_stack` reads this flag to leave the
    /// resulting `StackItem`'s `inline_effect` as `None` and set its own
    /// `madness_offer` instead (see that field's doc). Always `false` for a
    /// real `triggers_for`-matched trigger.
    pub is_madness_offer: bool,
    /// True iff this trigger's own `source` is the object that just
    /// resolved from a kicked cast (`engine::EngineState::
    /// pending_kicked_source`, consumed by `collect_and_process`) -- Goblin
    /// Bushwhacker's ETB trigger reads this via `engine::
    /// push_trigger_onto_stack` copying it onto the trigger's own
    /// `state::StackItem::kicked`, then `effect::ExecCtx::kicked` at
    /// resolution. `false` for every other trigger.
    pub kicked: bool,
}

/// Runs state-based actions to a fixed point: repeat the full SBA sweep
/// until one pass makes no change (704.3).
pub fn sba_fixed_point(state: &mut GameState) {
    loop {
        let mut changed = false;

        // 704.5g: a creature with toughness 0 or less is put into its
        // owner's graveyard. 704.5h: a creature with lethal damage marked
        // is destroyed.
        let mut dying = Vec::new();
        for (id, obj) in state.objects.iter() {
            if obj.zone != Zone::Battlefield {
                continue;
            }
            let def = &crate::card_def::CARD_DEFS[obj.card_def as usize];
            if !def.has_type(crate::card_def::CardType::Creature) {
                continue;
            }
            let toughness = crate::engine::effective_toughness(state, id);
            if toughness <= 0 || obj.damage as i32 >= toughness {
                dying.push(id);
            }
        }
        for id in dying {
            crate::event::commit(state, crate::event::ProposedEvent::zone_change(id, Zone::Graveyard));
            changed = true;
        }

        // 704.5a: a player with 0 or less life loses. 704.5c: a player who
        // attempted to draw from an empty library loses.
        for p in [PlayerId::P0, PlayerId::P1] {
            let ps = &mut state.players[p.index()];
            if !ps.has_lost && (ps.life <= 0 || ps.drew_from_empty) {
                ps.has_lost = true;
                changed = true;
            }
        }

        // 111.8/704.5d: a token in any zone other than the battlefield
        // ceases to exist -- most commonly a sacrificed/died Blood Token
        // ending up back in the graveyard's card-count for the rest of the
        // game, which it never does in a real game (root-caused against
        // the real v3 corpus: `kernel_gy` carrying a stray "Blood Token"
        // entry the trace's own graveyard snapshot never has, many turns
        // after the token was created and then activated/sacrificed).
        let leaving: Vec<ObjectId> = state
            .objects
            .iter()
            .filter(|(_, obj)| obj.zone != Zone::Battlefield && crate::card_def::CARD_DEFS[obj.card_def as usize].is_token)
            .map(|(id, _)| id)
            .collect();
        for id in leaving {
            if crate::event::cease_to_exist(state, id) {
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

/// Drains `state.engine.event_log`, matches triggers, runs SBAs to a fixed
/// point, and returns any newly-triggered abilities in APNAP order (active
/// player's triggers first).
pub fn collect_and_process(state: &mut GameState) -> Vec<PendingTrigger> {
    let events: Vec<CommittedEvent> = state.engine.event_log.drain(..).collect();
    let draws_this_turn_at = draws_this_turn_snapshot(&events, state);
    // Single-shot: `engine::resolve_top_of_stack` set this immediately
    // before the resolution whose events we're about to match, explicitly
    // (`Some`/`None`) every single time -- taking it here means it can never
    // carry over into a later, unrelated `collect_and_process` call (see
    // `EngineState::pending_kicked_source`'s doc).
    let kicked_source = state.engine.pending_kicked_source.take();

    let mut new_triggers = Vec::new();
    for (id, obj) in state.objects.iter() {
        for def in triggers_for(obj.card_def) {
            if obj.zone != def.home_zone {
                continue;
            }
            for (i, ev) in events.iter().enumerate() {
                if trigger_matches(def.condition, ev, id, obj.controller, state, draws_this_turn_at[i]) {
                    let kicked = Some(id) == kicked_source;
                    if def.intervening_if_kicked && !kicked {
                        continue;
                    }
                    new_triggers.push(PendingTrigger {
                        controller: obj.controller,
                        source: id,
                        effect: (def.effect)(),
                        is_madness_offer: false,
                        kicked,
                    });
                }
            }
        }
    }

    sba_fixed_point(state);

    order_apnap(new_triggers, state.active_player)
}

/// For each event in `events` (already committed, in commit order), the
/// value `draws_this_turn` genuinely held *at the moment that specific
/// event was committed* -- not `state`'s current (post-batch) value.
///
/// Root-caused against `game_20260713_002147_0002.txt`: `EffectOp::
/// DrawCards`'s loop (Faithless Looting's "draw two cards") commits both
/// draws into `event_log` before `collect_and_process` ever runs (both
/// resolve as one atomic ability, 608.2h -- correctly so; a trigger check
/// belongs *after* the whole ability resolves, not spliced mid-resolution,
/// 603.3/704.3), so by the time this function inspects `state`, `draws_
/// this_turn` already reflects *both* draws. Checking every event in the
/// batch against that single final value made `TriggerCondition::
/// DrawNth(3)` (Sneaky Snacker) miss entirely whenever a 2-draw batch
/// jumped straight over 3 (2 -> 4): neither event's *own* moment (3, then
/// 4) was ever actually tested, only the batch's final value (4) tested
/// twice. 608.2h itself settles that each drawn card is still a distinct,
/// sequential event ("the player draws that many cards, in that order"),
/// so a "your Nth draw this turn" condition must see each one's own true
/// historical count -- reconstructed here (not threaded through
/// `CommittedEvent::Draw` itself, which stays a plain, serializable
/// snapshot-free record) by walking `events` forward per player: the
/// count immediately *before* this batch is `state`'s current value minus
/// however many of this player's own `Draw` events are in this same
/// batch, then each of that player's `Draw` events in commit order adds
/// exactly 1.
fn draws_this_turn_snapshot(events: &[CommittedEvent], state: &GameState) -> Vec<u32> {
    let batch_draws = |p: PlayerId| events.iter().filter(|e| matches!(e, CommittedEvent::Draw { player, object: Some(_) } if *player == p)).count() as u32;
    let mut running = [state.players[0].draws_this_turn - batch_draws(PlayerId::P0), state.players[1].draws_this_turn - batch_draws(PlayerId::P1)];
    events
        .iter()
        .map(|ev| match ev {
            CommittedEvent::Draw { player, object: Some(_) } => {
                running[player.index()] += 1;
                running[player.index()]
            }
            _ => 0, // unused by any non-DrawNth trigger_matches arm
        })
        .collect()
}

/// `draws_this_turn_at_event`: only meaningful for a `CommittedEvent::Draw`
/// (see `draws_this_turn_snapshot`'s doc for why this can't just read
/// `state` live) -- unused, and irrelevant, for every other event/condition
/// pairing.
fn trigger_matches(cond: TriggerCondition, ev: &CommittedEvent, source: ObjectId, controller: PlayerId, state: &GameState, draws_this_turn_at_event: u32) -> bool {
    match (cond, ev) {
        (TriggerCondition::Etb, CommittedEvent::ZoneChange { object, to: Zone::Battlefield, .. }) => *object == source,
        (TriggerCondition::DealsDamage, CommittedEvent::Damage { source: s, .. }) => *s == source,
        (TriggerCondition::CastInstantOrSorcery, CommittedEvent::SpellCast { spell, controller: caster }) => {
            *caster == controller && {
                let def = &crate::card_def::CARD_DEFS[state.objects.get(*spell).card_def as usize];
                def.has_type(crate::card_def::CardType::Instant) || def.has_type(crate::card_def::CardType::Sorcery)
            }
        }
        (TriggerCondition::DrawNth(n), CommittedEvent::Draw { player, object: Some(_) }) => *player == controller && draws_this_turn_at_event == n,
        (TriggerCondition::LeftBattlefieldToGraveyard, CommittedEvent::ZoneChange { object, from: Zone::Battlefield, to: Zone::Graveyard }) => {
            *object == source
        }
        _ => false,
    }
}

/// 603.3b: each player puts triggered abilities they control on the stack
/// in an order of their choice, in APNAP order (active player's group
/// first). This only groups; a real per-player choice within a group of 2+
/// is `engine::Decision::OrderTriggers`.
pub fn order_apnap(triggers: Vec<PendingTrigger>, active_player: PlayerId) -> Vec<PendingTrigger> {
    let (mut active, mut other): (Vec<_>, Vec<_>) = triggers.into_iter().partition(|t| t.controller == active_player);
    active.append(&mut other);
    active
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;
    use crate::state::GameState;

    // Lethal-damage creature death is exercised end-to-end in
    // `engine::tests::lethal_damage_kills_creature_via_sba`, using a real
    // `CARD_DEFS` creature (card-def ids here are synthetic and don't map
    // to real cards).

    #[test]
    fn sba_declares_loss_at_zero_life() {
        let mut state = GameState::new_from_libraries(&[1], &[2], |c| format!("card-{c}"), 1);
        state.players[0].life = 0;
        sba_fixed_point(&mut state);
        assert!(state.players[0].has_lost);
    }

    #[test]
    fn sba_declares_loss_on_drew_from_empty() {
        let mut state = GameState::new_from_libraries(&[1], &[2], |c| format!("card-{c}"), 1);
        state.players[1].drew_from_empty = true;
        sba_fixed_point(&mut state);
        assert!(state.players[1].has_lost);
    }

    #[test]
    fn apnap_orders_active_player_triggers_first() {
        let a = PendingTrigger { controller: PlayerId::P1, source: ObjectId(1), effect: EffectOp::Sequence(vec![]), is_madness_offer: false, kicked: false };
        let b = PendingTrigger { controller: PlayerId::P0, source: ObjectId(2), effect: EffectOp::Sequence(vec![]), is_madness_offer: false, kicked: false };
        let ordered = order_apnap(vec![a.clone(), b.clone()], PlayerId::P0);
        assert_eq!(ordered, vec![b, a]);
    }
}
