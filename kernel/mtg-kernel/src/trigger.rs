//! State-based actions and trigger collection.
//!
//! After every resolution (a spell/ability resolving, a land entering, a
//! turn-based draw -- anything that produced `CommittedEvent`s), the engine
//! calls `collect_and_process`: it drains the event log, matches triggered
//! abilities, runs SBAs to a fixed point, and returns any newly-pending
//! triggers in APNAP order for `engine.rs` to place on the stack (or, if
//! 2+ share a controller, to ask that controller to order via
//! `engine::Decision::OrderTriggers`).

use crate::effect::EffectOp;
use crate::event::CommittedEvent;
use crate::ids::{ObjectId, PlayerId};
use crate::state::{GameState, Zone};
use serde::{Deserialize, Serialize};

/// Trigger conditions this increment's kernel can match. Mono-Red Burn's
/// pool has zero implemented triggers this increment (Guttersnipe and
/// Voldaren Epicure are vanilla bodies -- see `card_def.rs`), so
/// `triggers_for` always returns `&[]` today. The shape exists so a future
/// increment can wire real triggered abilities into `CARD_DEFS` without
/// touching the collection loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerCondition {
    Etb,
    DealsDamage,
}

pub struct TriggeredAbilityDef {
    pub condition: TriggerCondition,
    pub effect: fn() -> EffectOp,
}

pub fn triggers_for(_card_def: u16) -> &'static [TriggeredAbilityDef] {
    &[]
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
        if obj.zone != Zone::Battlefield {
            continue;
        }
        for def in triggers_for(obj.card_def) {
            for ev in &events {
                if trigger_matches(def.condition, ev, id) {
                    new_triggers.push(PendingTrigger { controller: obj.controller, source: id, effect: (def.effect)() });
                }
            }
        }
    }

    sba_fixed_point(state);

    order_apnap(new_triggers, state.active_player)
}

fn trigger_matches(cond: TriggerCondition, ev: &CommittedEvent, source: ObjectId) -> bool {
    match (cond, ev) {
        (TriggerCondition::Etb, CommittedEvent::ZoneChange { object, to: Zone::Battlefield, .. }) => *object == source,
        (TriggerCondition::DealsDamage, CommittedEvent::Damage { source: s, .. }) => *s == source,
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
