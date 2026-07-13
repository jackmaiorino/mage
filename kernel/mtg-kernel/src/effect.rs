//! Interpreted effect programs.
//!
//! `EffectOp` is the only representation of card behavior: composition
//! primitives (`Sequence`, `Conditional`, `Choice`) plus a fixed
//! leaf-op vocabulary (`DealDamage`, `GainLife`, `LoseLife`, `DrawCards`,
//! `MoveObject`, `TapObject`, `AddMana`). There is no card-shaped op --
//! "Lightning Bolt" is not a variant, `DealDamage { amount: 3, .. }` is
//! (see `card_def.rs` / the generated `CARD_DEFS` table for how card
//! behavior handlers are wired up).
//!
//! `execute` is the *only* function that runs an `EffectOp`, and every leaf
//! goes through `event::propose_and_commit`, so nothing but the commit
//! pipeline (`event::commit`) ever mutates `GameState` in response to card
//! behavior (see the crate-level invariants in `lib.rs`).

use crate::event;
use crate::ids::{ObjectId, PlayerId};
use crate::mana::ManaColor;
use crate::state::{GameState, Target, Zone};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectRef {
    /// The permanent/spell this effect program belongs to.
    ThisSource,
    /// A target resolved at cast/activation time, by index.
    Target(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerRef {
    Controller,
    Target(u8),
    /// The controller of a target object (e.g. "that creature's
    /// controller").
    ObjectController(ObjectRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetRef {
    ThisSource,
    Target(u8),
    /// The controller's one opponent. The kernel only ever simulates 1v1
    /// games (see `lib.rs`), so "deal N damage to each opponent"
    /// (Guttersnipe, Voldaren Epicure, Grab the Prize) never needs a
    /// chosen target -- it's always exactly `ctx.controller.opponent()`.
    Opponent,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectCond {
    Always,
    Never,
    /// True iff this cast's mandatory additional cost discarded a
    /// non-land card (Grab the Prize). Reads `ExecCtx::discarded`, which
    /// `engine::finalize_cast` populates from the additional-cost payment
    /// before pushing the spell onto the stack.
    DiscardedNonLandForCost,
    /// True iff `ctx.controller` had a land enter the battlefield under
    /// their control this turn (Searing Blaze's landfall clause). Reads
    /// `PlayerState::lands_played_this_turn`, which only tracks land
    /// *drops* -- an accurate proxy for this pool, since nothing in it puts
    /// a land onto the battlefield any other way.
    LandfallThisTurn,
    /// True iff `ctx.targets[idx]` is `Target::Object(id)` and that
    /// object is currently in `zone`. `false` for a `Target::Player` (no
    /// card in this pool needs that combination) or an out-of-range index.
    /// The general-purpose 608.2b "is this target still legal" fizzle
    /// check: a creature that died, a spell that already left the stack, or
    /// a permanent that's no longer on the battlefield all read `false`
    /// here, and the guarded effect they'd otherwise feed is skipped
    /// instead of misfiring against a stale `ObjectId`.
    TargetInZone(u8, Zone),
    /// True iff `ctx.targets[idx]` is a `Target::Object(id)` whose card
    /// definition's `colors` includes `color` (105.1/202.2). `false` for a
    /// `Target::Player`. Pyroblast's/Red Elemental Blast's "if it's
    /// blue"/"blue spell"/"blue permanent" checks.
    TargetIsColor(u8, crate::mana::ManaColor),
    /// Both sub-conditions must hold.
    And(Box<EffectCond>, Box<EffectCond>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectOp {
    Sequence(Vec<EffectOp>),
    Conditional {
        cond: EffectCond,
        then: Box<EffectOp>,
        else_: Box<EffectOp>,
    },
    /// The controller picks one of `options`. No card in this increment's
    /// pool is modal, so the resolver is a deterministic stand-in (always
    /// runs `options[0]`) until a real decision kind routes controller
    /// choice through `engine::Decision` (see module docs there).
    Choice {
        controller: PlayerRef,
        options: Vec<EffectOp>,
    },
    DealDamage {
        target: TargetRef,
        amount: i32,
    },
    GainLife {
        player: PlayerRef,
        amount: i32,
    },
    LoseLife {
        player: PlayerRef,
        amount: i32,
    },
    DrawCards {
        player: PlayerRef,
        count: u32,
    },
    /// Discard `count` cards from `player`'s hand, chosen by that player.
    /// Unlike every other leaf, this one doesn't necessarily mutate state
    /// synchronously: `execute` stages `EngineState::pending_discard` and
    /// returns, and `engine::advance_until_decision` asks
    /// `Decision::Discard`. Because of that, **this must be the last leaf
    /// in any `Sequence` it appears in** (see `engine.rs`'s
    /// `pending_discard` doc for why: nothing after it in the same
    /// resolution would run before the decision is answered). The only
    /// user this increment, Faithless Looting ("draw two, then discard
    /// two"), satisfies this by construction.
    DiscardCards {
        player: PlayerRef,
        count: u32,
    },
    MoveObject {
        object: ObjectRef,
        to_zone: Zone,
    },
    TapObject {
        object: ObjectRef,
    },
    AddMana {
        player: PlayerRef,
        colors: Vec<ManaColor>,
    },
    /// Creates a fresh token permanent (e.g. Blood) directly on the
    /// battlefield under `controller`'s control. `token_def` indexes
    /// `card_def::CARD_DEFS` same as any other object -- tokens are real
    /// `GameObject`s, not a separate representation (see
    /// `event::ProposedEvent::create_token`).
    CreateToken {
        token_def: u16,
        controller: PlayerRef,
    },
    /// The controller may pay ONE of {discard `discard` cards, sacrifice
    /// `sacrifice_lands` lands} -- only whichever options are currently
    /// legal are offered, and declining is always legal too (Highway
    /// Robbery's `DoIfCostPaid(OrCost(DiscardCardCost, SacrificeTargetCost))`).
    /// If they do, `then` runs. Like `DiscardCards`, this is deferred:
    /// `execute` stages `EngineState::pending_optional_cost` and returns
    /// without knowing the outcome yet (`engine::Decision::ChooseOptionalCost`
    /// asks), so **this must be the last leaf in any `Sequence` it appears
    /// in**, same constraint and same reason as `DiscardCards`. A future
    /// card that needs both sub-costs simultaneously payable (not this
    /// pool) is out of scope: `discard`/`sacrifice_lands` are mutually
    /// exclusive choices, never both paid.
    MayPayCostThen {
        discard: u8,
        sacrifice_lands: u8,
        then: Box<EffectOp>,
    },
}

/// Everything an effect program needs to resolve symbolic refs against a
/// concrete game: which object it's running for, who controls it, and the
/// targets chosen when it was cast/activated.
pub struct ExecCtx {
    pub source: ObjectId,
    pub controller: PlayerId,
    pub targets: Vec<Target>,
    /// Cards discarded to pay this cast's mandatory additional cost (Grab
    /// the Prize), if any. Empty for everything else. Read by
    /// `EffectCond::DiscardedNonLandForCost`.
    pub discarded: Vec<ObjectId>,
}

impl ExecCtx {
    pub fn no_targets(source: ObjectId, controller: PlayerId) -> ExecCtx {
        ExecCtx { source, controller, targets: Vec::new(), discarded: Vec::new() }
    }

    fn resolve_object(&self, r: ObjectRef) -> ObjectId {
        match r {
            ObjectRef::ThisSource => self.source,
            ObjectRef::Target(i) => match self.targets[i as usize] {
                Target::Object(id) => id,
                Target::Player(_) => panic!("effect expected an object target at index {i}"),
            },
        }
    }

    fn resolve_target(&self, r: TargetRef) -> Target {
        match r {
            TargetRef::ThisSource => Target::Object(self.source),
            TargetRef::Target(i) => self.targets[i as usize],
            TargetRef::Opponent => Target::Player(self.controller.opponent()),
        }
    }

    fn resolve_player(&self, r: PlayerRef, state: &GameState) -> PlayerId {
        match r {
            PlayerRef::Controller => self.controller,
            PlayerRef::Target(i) => match self.targets[i as usize] {
                Target::Player(p) => p,
                Target::Object(_) => panic!("effect expected a player target at index {i}"),
            },
            PlayerRef::ObjectController(oref) => state.objects.get(self.resolve_object(oref)).controller,
        }
    }
}

pub fn execute(op: &EffectOp, ctx: &ExecCtx, state: &mut GameState) {
    match op {
        EffectOp::Sequence(ops) => {
            for inner in ops {
                execute(inner, ctx, state);
            }
        }
        EffectOp::Conditional { cond, then, else_ } => {
            let taken = eval_cond(cond, ctx, state);
            execute(if taken { then } else { else_ }, ctx, state);
        }
        EffectOp::Choice { options, .. } => {
            if let Some(first) = options.first() {
                execute(first, ctx, state);
            }
        }
        EffectOp::DealDamage { target, amount } => {
            let target = ctx.resolve_target(*target);
            event::propose_and_commit(state, event::ProposedEvent::damage(ctx.source, target, *amount));
        }
        EffectOp::GainLife { player, amount } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(state, event::ProposedEvent::life_gain(player, *amount));
        }
        EffectOp::LoseLife { player, amount } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(state, event::ProposedEvent::life_loss(player, *amount));
        }
        EffectOp::DrawCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            for _ in 0..*count {
                event::propose_and_commit(state, event::ProposedEvent::draw(player));
            }
        }
        EffectOp::MoveObject { object, to_zone } => {
            let object = ctx.resolve_object(*object);
            event::propose_and_commit(state, event::ProposedEvent::zone_change(object, *to_zone));
        }
        EffectOp::TapObject { object } => {
            let object = ctx.resolve_object(*object);
            event::propose_and_commit(state, event::ProposedEvent::tap(object));
        }
        EffectOp::AddMana { player, colors } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(state, event::ProposedEvent::mana_add(player, colors.clone()));
        }
        EffectOp::DiscardCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            state.engine.pending_discard = Some(crate::engine::PendingDiscard {
                player,
                count: *count,
                resume: crate::engine::DiscardResume::None,
            });
        }
        EffectOp::CreateToken { token_def, controller } => {
            let controller = ctx.resolve_player(*controller, state);
            event::propose_and_commit(state, event::ProposedEvent::create_token(*token_def, controller));
        }
        EffectOp::MayPayCostThen { discard, sacrifice_lands, then } => {
            let discard_payable = *discard > 0 && state.players[ctx.controller.index()].hand.len() >= *discard as usize;
            let sacrifice_payable = *sacrifice_lands > 0 && crate::engine::count_controlled_lands(ctx.controller, state) >= *sacrifice_lands as u32;
            if !discard_payable && !sacrifice_payable {
                // Nothing payable: DoIfCostPaid's own `cost.canPay(...)`
                // gate is false too, so the reference never even offers the
                // "may pay?" prompt here -- matches, no-op.
                return;
            }
            state.engine.pending_optional_cost = Some(crate::engine::PendingOptionalCost {
                player: ctx.controller,
                source: ctx.source,
                discard: *discard,
                sacrifice_lands: *sacrifice_lands,
                discard_payable,
                sacrifice_payable,
                then: (**then).clone(),
                // `resolve_top_of_stack` fills this in right after this
                // call returns, if it's resolving this same spell -- see
                // `PendingOptionalCost::spell_resume`'s doc.
                spell_resume: None,
            });
        }
    }
}

fn eval_cond(cond: &EffectCond, ctx: &ExecCtx, state: &GameState) -> bool {
    match cond {
        EffectCond::Always => true,
        EffectCond::Never => false,
        EffectCond::DiscardedNonLandForCost => ctx.discarded.iter().any(|&id| {
            let def_idx = state.objects.get(id).card_def;
            !crate::card_def::CARD_DEFS[def_idx as usize].is_land
        }),
        EffectCond::LandfallThisTurn => state.players[ctx.controller.index()].lands_played_this_turn > 0,
        EffectCond::TargetInZone(idx, zone) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) => state.objects.get(*id).zone == *zone,
            _ => false,
        },
        EffectCond::TargetIsColor(idx, color) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) => {
                let def_idx = state.objects.get(*id).card_def;
                crate::card_def::CARD_DEFS[def_idx as usize].colors.contains(color)
            }
            _ => false,
        },
        EffectCond::And(a, b) => eval_cond(a, ctx, state) && eval_cond(b, ctx, state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;

    fn two_card_libraries() -> GameState {
        GameState::new_from_libraries(&[1, 2], &[3, 4], |c| format!("card-{c}"), 1)
    }

    #[test]
    fn sequence_runs_every_leaf_in_order() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::LoseLife { player: PlayerRef::Controller, amount: 2 },
            EffectOp::GainLife { player: PlayerRef::Controller, amount: 5 },
        ]);
        execute(&op, &ctx, &mut state);
        assert_eq!(state.players[0].life, 20 - 2 + 5);
    }

    /// Proves the `Conditional` composition primitive works end-to-end even
    /// though no card in this increment's pool needs it.
    #[test]
    fn conditional_picks_then_or_else_branch() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);

        let taken = EffectOp::Conditional {
            cond: EffectCond::Always,
            then: Box::new(EffectOp::LoseLife { player: PlayerRef::Controller, amount: 3 }),
            else_: Box::new(EffectOp::Sequence(vec![])),
        };
        execute(&taken, &ctx, &mut state);
        assert_eq!(state.players[0].life, 17);

        let not_taken = EffectOp::Conditional {
            cond: EffectCond::Never,
            then: Box::new(EffectOp::LoseLife { player: PlayerRef::Controller, amount: 100 }),
            else_: Box::new(EffectOp::Sequence(vec![])),
        };
        execute(&not_taken, &ctx, &mut state);
        assert_eq!(state.players[0].life, 17, "else branch is a no-op here");
    }

    #[test]
    fn deal_damage_to_target_player_reduces_life() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx {
            source: ObjectId(0),
            controller: PlayerId::P0,
            targets: vec![Target::Player(PlayerId::P1)],
            discarded: Vec::new(),
        };
        execute(&EffectOp::DealDamage { target: TargetRef::Target(0), amount: 3 }, &ctx, &mut state);
        assert_eq!(state.players[1].life, 17);
    }

    #[test]
    fn deal_damage_to_target_object_marks_damage() {
        let mut state = two_card_libraries();
        let creature = state.draw_card(PlayerId::P1).unwrap();
        state.move_hand_to_battlefield(PlayerId::P1, creature);
        let ctx = ExecCtx {
            source: ObjectId(0),
            controller: PlayerId::P0,
            targets: vec![Target::Object(creature)],
            discarded: Vec::new(),
        };
        execute(&EffectOp::DealDamage { target: TargetRef::Target(0), amount: 4 }, &ctx, &mut state);
        assert_eq!(state.objects.get(creature).damage, 4);
    }

    #[test]
    fn draw_cards_leaf_draws_the_requested_count() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        execute(&EffectOp::DrawCards { player: PlayerRef::Controller, count: 2 }, &ctx, &mut state);
        assert_eq!(state.players[0].hand.len(), 2);
    }

    #[test]
    fn tap_and_add_mana_leaves_compose_a_mana_ability() {
        let mut state = two_card_libraries();
        let land = state.draw_card(PlayerId::P0).unwrap();
        state.move_hand_to_battlefield(PlayerId::P0, land);
        let ctx = ExecCtx::no_targets(land, PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::TapObject { object: ObjectRef::ThisSource },
            EffectOp::AddMana { player: PlayerRef::Controller, colors: vec![ManaColor::R] },
        ]);
        execute(&op, &ctx, &mut state);
        assert!(state.objects.get(land).tapped);
        assert_eq!(state.players[0].mana_pool[ManaColor::R.pool_index()], 1);
    }
}
