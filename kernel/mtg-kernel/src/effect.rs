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

use crate::card_def::Subtype;
use crate::event;
use crate::ids::{ObjectId, PlayerId};
use crate::mana::ManaColor;
use crate::state::{GameState, Target, Zone};
use serde::{Deserialize, Serialize};

/// Which of a controller's creatures a team-wide pump/keyword effect
/// affects (`EffectOp::PumpControlled`). A closed, tiny enum rather than a
/// general subtype query: only the two shapes Rally's cards need exist
/// today (Goblin Bushwhacker's unfiltered "creatures you control", Rally at
/// the Hornburg's "Humans you control") -- a future card needing a
/// different subtype filter adds a `CreatureFilter` variant, reusing
/// `card_def::Subtype` (typed, not a string-contains check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreatureFilter {
    AnyControlled,
    ControlledWithSubtype(Subtype),
}

/// How long an impulse-drawn card (`EffectOp::ImpulseDraw`) stays playable
/// from exile. `EndOfTurn` is cleared unconditionally at the very next
/// `Step::Cleanup`, whoever's; `UntilOwnersNextTurn` survives through the
/// rest of this turn, the opponent's turn, and the owner's own next turn,
/// expiring at *that* turn's cleanup -- tracked via `engine::
/// PlayPermissionExpiry`, since a plain turn-number comparison can't tell
/// the owner's turn apart from the opponent's turn sharing the same kernel
/// round number (see that type's doc). Both durations are carried by an
/// `engine::PlayPermission`, not a pseudo-hand-zone membership list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImpulseDuration {
    EndOfTurn,
    UntilOwnersNextTurn,
}

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
    /// Metalcraft (Galvanic Blast): true iff `ctx.controller` currently
    /// controls at least `n` permanents with `CardType::Artifact` (Great
    /// Furnace, Clockwork Percussionist, and Experimental Synthesizer are
    /// this pool's artifacts). A plain board-state count, not persisted
    /// anywhere -- recomputed fresh every time it's checked, same as
    /// `LandfallThisTurn`.
    ControlsArtifactCount(u8),
    /// True iff the cast this resolution's ETB trigger followed from was
    /// kicked (`card_def::CardDef::kicker_cost`, Goblin Bushwhacker's "if it
    /// was kicked" intervening-if). Reads `ExecCtx::kicked` -- cast-time
    /// metadata carried on the spell's `StackItem`, propagated to the ETB
    /// trigger's own `PendingTrigger`/`StackItem`/`ExecCtx` when it's
    /// queued, and gone once that trigger resolves (CR 702.33/601.2f: Kicker
    /// is a property of *this casting*, not a durable fact stored anywhere
    /// keyed by stable object id -- CR 400.7 zone changes create new
    /// objects, so a persistent id-keyed marker could falsely survive a
    /// later, unkicked cast of the same physical card if it were ever
    /// cleared incorrectly. Reading cast-scoped context instead of a lookup
    /// table makes that failure mode structurally impossible rather than
    /// merely avoided by careful clearing).
    WasKicked,
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
    /// "Deals `amount` damage to each opponent and each creature they
    /// control" (End the Festivities). The kernel only ever simulates 1v1
    /// games (see `lib.rs`), so "each opponent" is always exactly one
    /// player, same rationale as `TargetRef::Opponent`; no planeswalker
    /// card exists in the 132-card pool this increment adds to, so the
    /// "and each planeswalker they control" half of the real card's text
    /// is vacuously covered (there is never one to hit).
    DamageOpponentAndTheirCreatures {
        amount: i32,
    },
    /// A team-wide, until-end-of-turn pump/haste grant (Goblin Bushwhacker's
    /// kicked ETB, Rally at the Hornburg's token haste). Snapshots exactly
    /// which of `ctx.controller`'s current creatures match `filter` *at the
    /// moment this executes* (611.2c: the affected-objects set is locked in
    /// when the effect begins, not re-evaluated later) and stages an
    /// `engine::UntilEndOfTurnEffect::ResolvedSetEffect` naming those
    /// specific `ObjectId`s, cleared unconditionally at the next
    /// `Step::Cleanup` --
    /// see that variant's doc. Sequencing matters for Rally at the Hornburg:
    /// its own two `CreateToken` calls run *before* this in the same
    /// `Sequence`, so the freshly-created Human Soldier tokens are already
    /// on the battlefield (and therefore in the snapshot) by the time this
    /// leaf runs.
    PumpControlled {
        filter: CreatureFilter,
        power: i32,
        toughness: i32,
        grant_haste: bool,
    },
    /// Exiles the top `count` cards of `ctx.controller`'s library (silently
    /// stopping short if the library runs out first -- this is not a draw,
    /// so an empty library here is not a loss condition) and marks each one
    /// playable by its owner for `duration` -- Clockwork Percussionist's
    /// dies trigger, Experimental Synthesizer's enters-or-leaves trigger,
    /// and Reckless Impulse all reduce to this, differing only in `count`/
    /// `duration`. See `ImpulseDuration`'s doc for how each duration is
    /// tracked/expired (`engine::PlayPermission`), and `engine::
    /// castable_spells`/`engine::land_drop_candidates` for where the
    /// resulting exiled cards become legally castable/playable again,
    /// through the *ordinary* timing/cost/land-quota checks.
    ImpulseDraw {
        count: u8,
        duration: ImpulseDuration,
    },
    /// Chain Lightning's "Then that player or that permanent's controller
    /// may pay {R}{R}. If the player does, they may copy this spell and may
    /// choose a new target for that copy": this kernel has no spell-copy
    /// primitive at all (copying a spell, retargeting the copy, and the
    /// copy's own susceptibility to being re-copied are all unmodeled). The
    /// mandatory "deals 3 damage" leaf always runs first (unconditionally,
    /// same generated function Lightning Bolt/Fiery Temper share); *this*
    /// leaf runs after it and checks whether `affected` (whichever single
    /// target the damage actually hit -- a player directly, or a
    /// permanent's controller) could currently pay {R}{R}. If they
    /// couldn't, declining is the *only* legal choice, so resolution simply
    /// finishes (matches the real game exactly, not an approximation). If
    /// they could, this is a genuine, live decision this kernel cannot
    /// simulate -- rather than silently guessing "declines" (which would be
    /// a real, hidden divergence from a possible real game), it halts the
    /// walk explicitly via `EngineState::halted` / `Decision::Halted` with
    /// `UnsupportedMechanic::SpellCopy`. Per external review: corpus
    /// non-occurrence alone doesn't justify skipping this check, since
    /// off-trace/search-driven play can reach board states where the
    /// payment *is* affordable even if no recorded game ever cast this into
    /// one.
    HaltIfAffectedCanPayCopyCost {
        affected: TargetRef,
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
    /// True iff the spell/ability this resolution belongs to was kicked
    /// (`card_def::CardDef::kicker_cost`) -- carried on `state::StackItem::
    /// kicked` and copied in here by `engine::resolve_top_of_stack`, and
    /// (for the ETB trigger this spell's own resolution queues) propagated
    /// onto that trigger's own `trigger::PendingTrigger`/`StackItem` in turn.
    /// Read by `EffectCond::WasKicked`. `false` for every card without
    /// Kicker (the overwhelming majority), and for any ability/trigger not
    /// downstream of a kicked cast.
    pub kicked: bool,
}

impl ExecCtx {
    pub fn no_targets(source: ObjectId, controller: PlayerId) -> ExecCtx {
        ExecCtx { source, controller, targets: Vec::new(), discarded: Vec::new(), kicked: false }
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
        EffectOp::DamageOpponentAndTheirCreatures { amount } => {
            // 611.2c-adjacent simultaneity: one instance of this effect
            // hits the opponent and every creature they control at once,
            // same pattern `engine::combat_damage_wave` already uses for
            // combat damage -- a single propose/replace/commit batch, not
            // sequential individual commits that could see each other's
            // side effects mid-resolution.
            let opponent = ctx.controller.opponent();
            let mut events = vec![event::ProposedEvent::damage(ctx.source, Target::Player(opponent), *amount)];
            events.extend(state.players[opponent.index()].battlefield.iter().copied().filter_map(|id| {
                let def = &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                def.has_type(crate::card_def::CardType::Creature).then(|| event::ProposedEvent::damage(ctx.source, Target::Object(id), *amount))
            }));
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::PumpControlled { filter, power, toughness, grant_haste } => {
            let object_ids: Vec<ObjectId> = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .copied()
                .filter(|&id| {
                    let def = &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                    if !def.has_type(crate::card_def::CardType::Creature) {
                        return false;
                    }
                    match filter {
                        CreatureFilter::AnyControlled => true,
                        CreatureFilter::ControlledWithSubtype(sub) => def.subtypes.contains(sub),
                    }
                })
                .collect();
            if !object_ids.is_empty() {
                let mut layer = crate::engine::Layers::NONE;
                if *power != 0 || *toughness != 0 {
                    layer = layer | crate::engine::Layers::POWER_TOUGHNESS;
                }
                if *grant_haste {
                    layer = layer | crate::engine::Layers::ABILITY_ADDING;
                }
                let timestamp = crate::engine::next_timestamp(state);
                state.engine.until_end_of_turn.push(crate::engine::UntilEndOfTurnEffect::ResolvedSetEffect {
                    object_ids,
                    layer,
                    timestamp,
                    duration: crate::engine::EffectDuration::EndOfTurn,
                    power: *power,
                    toughness: *toughness,
                    grant_haste: *grant_haste,
                });
            }
        }
        EffectOp::ImpulseDraw { count, duration } => {
            for _ in 0..*count {
                let Some(&top) = state.players[ctx.controller.index()].library.first() else {
                    break; // library ran dry partway through -- not a draw, no loss condition
                };
                if std::env::var("REPLAY_DEBUG_IMPULSE").is_ok() {
                    eprintln!(
                        "IMPULSE_DRAW controller={:?} source={:?} exiling id={} name={:?} turn={} round_step={:?} lib_remaining_before={} hand_size={} priority_round={}",
                        ctx.controller,
                        state.objects.get(ctx.source).name,
                        top.0,
                        state.objects.get(top).name,
                        state.turn,
                        state.step,
                        state.players[ctx.controller.index()].library.len(),
                        state.players[ctx.controller.index()].hand.len(),
                        state.engine.priority_round,
                    );
                }
                event::propose_and_commit(state, event::ProposedEvent::zone_change(top, Zone::Exile));
                let expiry = match duration {
                    ImpulseDuration::EndOfTurn => crate::engine::PlayPermissionExpiry::EndOfTurn,
                    ImpulseDuration::UntilOwnersNextTurn => crate::engine::PlayPermissionExpiry::UntilHoldersNextTurn { holder_turn_started: false },
                };
                let def = &crate::card_def::CARD_DEFS[state.objects.get(top).card_def as usize];
                let play_or_cast = if def.is_land { crate::engine::PlayOrCast::Play } else { crate::engine::PlayOrCast::Cast };
                state.engine.exile_play_permissions.push(crate::engine::PlayPermission {
                    object: top,
                    holder: ctx.controller,
                    // Snapshot *after* the exile move above, so this
                    // permission's own creating zone change isn't what
                    // immediately invalidates it -- see `PlayPermission::
                    // zone_change_generation`'s doc.
                    zone_change_generation: state.objects.get(top).zone_change_count,
                    play_or_cast,
                    expiry,
                });
            }
        }
        EffectOp::HaltIfAffectedCanPayCopyCost { affected } => {
            let target = ctx.resolve_target(*affected);
            let decider = match target {
                Target::Player(p) => p,
                Target::Object(id) => state.objects.get(id).controller,
            };
            let pay_rr = crate::mana::Cost { pips: &[crate::mana::Pip::Colored(ManaColor::R), crate::mana::Pip::Colored(ManaColor::R)], generic: 0, x_count: 0 };
            if crate::mana::can_pay(&pay_rr, 0, decider, state).is_some() {
                state.engine.halted = Some((crate::engine::UnsupportedMechanic::SpellCopy, ctx.source));
            }
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
        EffectCond::ControlsArtifactCount(n) => {
            let count = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .filter(|&&id| {
                    let def = &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                    def.has_type(crate::card_def::CardType::Artifact)
                })
                .count();
            count >= *n as usize
        }
        EffectCond::WasKicked => ctx.kicked,
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
            kicked: false,
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
            kicked: false,
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
