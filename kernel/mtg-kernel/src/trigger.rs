//! State-based actions and trigger collection.
//!
//! After every resolution (a spell/ability resolving, a land entering, a
//! turn-based draw -- anything that produced `CommittedEvent`s), the engine
//! calls `collect_and_process`: it drains the event log, matches triggered
//! abilities, runs SBAs to a fixed point, and returns any newly-pending
//! triggers in APNAP order for `engine.rs` to place on the stack (or, if
//! 2+ share a controller, to ask that controller to order via
//! `engine::Decision::OrderTriggers`).

use crate::effect::{EffectOp, PlayerRef, TargetRef};
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
}

pub struct TriggeredAbilityDef {
    pub condition: TriggerCondition,
    /// Which zone the source must be in for this ability to function.
    /// Every keyworded/triggered ability in MTG works from the
    /// battlefield unless its reminder text says otherwise (like Sneaky
    /// Snacker's "from your graveyard").
    pub home_zone: Zone,
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
    [TriggeredAbilityDef { condition: TriggerCondition::CastInstantOrSorcery, home_zone: Zone::Battlefield, effect: guttersnipe_effect }];
const VOLDAREN_EPICURE_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::Etb, home_zone: Zone::Battlefield, effect: voldaren_epicure_effect }];
const SNEAKY_SNACKER_TRIGGERS: [TriggeredAbilityDef; 1] =
    [TriggeredAbilityDef { condition: TriggerCondition::DrawNth(3), home_zone: Zone::Graveyard, effect: sneaky_snacker_effect }];

/// The Burn 16's real triggered abilities, matched by card name (ids are
/// codegen-assigned from `cards_v1.json`'s array order and not worth
/// duplicating as constants here -- see `build.rs`'s module doc on id
/// stability). Every other card in the 132-card pool has no triggered
/// ability implemented this increment and falls through to `&[]`.
pub fn triggers_for(card_def: u16) -> &'static [TriggeredAbilityDef] {
    match crate::card_def::CARD_DEFS[card_def as usize].name {
        "Guttersnipe" => &GUTTERSNIPE_TRIGGERS,
        "Voldaren Epicure" => &VOLDAREN_EPICURE_TRIGGERS,
        "Sneaky Snacker" => &SNEAKY_SNACKER_TRIGGERS,
        _ => &[],
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingTrigger {
    pub controller: PlayerId,
    pub source: ObjectId,
    pub effect: EffectOp,
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
            let toughness = def.toughness.unwrap_or(0) as i32 + obj.counters.plus1_plus1 as i32;
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

    let mut new_triggers = Vec::new();
    for (id, obj) in state.objects.iter() {
        for def in triggers_for(obj.card_def) {
            if obj.zone != def.home_zone {
                continue;
            }
            for ev in &events {
                if trigger_matches(def.condition, ev, id, obj.controller, state) {
                    new_triggers.push(PendingTrigger { controller: obj.controller, source: id, effect: (def.effect)() });
                }
            }
        }
    }

    sba_fixed_point(state);

    order_apnap(new_triggers, state.active_player)
}

fn trigger_matches(cond: TriggerCondition, ev: &CommittedEvent, source: ObjectId, controller: PlayerId, state: &GameState) -> bool {
    match (cond, ev) {
        (TriggerCondition::Etb, CommittedEvent::ZoneChange { object, to: Zone::Battlefield, .. }) => *object == source,
        (TriggerCondition::DealsDamage, CommittedEvent::Damage { source: s, .. }) => *s == source,
        (TriggerCondition::CastInstantOrSorcery, CommittedEvent::SpellCast { spell, controller: caster }) => {
            *caster == controller && {
                let def = &crate::card_def::CARD_DEFS[state.objects.get(*spell).card_def as usize];
                def.has_type(crate::card_def::CardType::Instant) || def.has_type(crate::card_def::CardType::Sorcery)
            }
        }
        (TriggerCondition::DrawNth(n), CommittedEvent::Draw { player, object: Some(_) }) => {
            *player == controller && state.players[player.index()].draws_this_turn == n
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
        let a = PendingTrigger { controller: PlayerId::P1, source: ObjectId(1), effect: EffectOp::Sequence(vec![]) };
        let b = PendingTrigger { controller: PlayerId::P0, source: ObjectId(2), effect: EffectOp::Sequence(vec![]) };
        let ordered = order_apnap(vec![a.clone(), b.clone()], PlayerId::P0);
        assert_eq!(ordered, vec![b, a]);
    }
}
