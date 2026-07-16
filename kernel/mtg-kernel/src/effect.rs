//! Interpreted effect programs.
//!
//! `EffectOp` is the only representation of card behavior: composition
//! primitives (`Sequence`, `Conditional`, `Choice`) plus a fixed
//! leaf-op vocabulary (`DealDamage`, `DrawCards`, `MoveObject`, library
//! partitioning, token creation, and other reusable state transitions).
//! There is no card-shaped op --
//! "Lightning Bolt" is not a variant, `DealDamage { amount: 3, .. }` is
//! (see `card_def.rs` / the generated `CARD_DEFS` table for how card
//! behavior handlers are wired up).
//!
//! `execute` and the resumable interpreter are the only paths that run an
//! `EffectOp`, and every leaf mutation goes through `event::propose_and_commit`
//! or `event::propose_and_commit_batch`, so nothing but the commit pipeline
//! (`event::commit`) mutates `GameState` in response to card behavior (see the
//! crate-level invariants in `lib.rs`).

use crate::card_def::{CardType, Subtype};
use crate::event;
use crate::ids::{ObjectId, PlayerId};
use crate::mana::ManaColor;
use crate::state::{GameState, StackItem, Target, Zone};
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
    /// The selected player picks one of `options` during resolution. The
    /// generic resumable interpreter preserves printed option order and
    /// yields a policy-visible decision without opening a priority, SBA, or
    /// trigger window.
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
    /// Publicly reveals a fixed snapshot of the top `count` cards of
    /// `player`'s library, then moves every card with `card_type` to
    /// `matching_to` and the remainder to `rest_to`. The matching group
    /// moves first and each group is one replacement-evaluated batch. A
    /// 2+ card graveyard group is explicitly ordered by its owner (the
    /// forced final card auto-completes); other groups retain snapshot order.
    /// This is not a draw: a short/empty library simply contributes fewer
    /// cards and never sets the draw-from-empty loss marker.
    RevealTopAndPartitionByType {
        player: PlayerRef,
        count: u8,
        card_type: CardType,
        matching_to: Zone,
        rest_to: Zone,
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
    /// Chain Lightning's post-damage "that player or that permanent's
    /// controller may pay {R}{R}" branch. XMage asks this choice before it
    /// attempts payment, even when payment will fail, so this always
    /// suspends the current resolution in the engine's dedicated spell-copy
    /// state machine; no SBA, trigger placement, zone
    /// move, or priority window can occur until payment and optional
    /// retargeting finish. This leaf must remain last in Chain Lightning's
    /// generated sequence for the same reason `DiscardCards` must remain
    /// last in its sequence.
    OfferAffectedPlayerSpellCopy {
        affected: TargetRef,
    },
}

/// One owned interpreter frame. `path` is the structural route through the
/// original effect program (sequence/branch/choice/group ordinals), making a
/// suspended continuation deterministic, hashable, and auditable without
/// storing closures or card-definition function pointers. Dynamic batch
/// frames are interpreter-owned: generated card programs only contain
/// `EffectOp`s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectFrame {
    Program {
        op: EffectOp,
        path: Vec<u16>,
    },
    MoveObjectsBatch {
        objects: Vec<EffectObjectBinding>,
        to_zone: Zone,
        preserve_known_identity: bool,
        order_resolved: bool,
        path: Vec<u16>,
    },
}

/// Binds a physical arena id to the exact public incarnation selected when
/// an effect snapshotted it. A restored/stale continuation must never move a
/// later incarnation that happens to reuse the same stable `ObjectId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectObjectBinding {
    pub object: ObjectId,
    pub expected_zone: Zone,
    pub expected_zone_change_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectTargetCandidate {
    pub target: Target,
    pub expected_object: Option<EffectObjectBinding>,
}

/// Internal reason/completion for a generic target-selection continuation.
/// Public schema-v4 projects this graveyard-ordering use as the already
/// reserved `TargetSelectionPurposeV4::CardSelection`; no card-specific
/// state or action identity is introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectTargetSelectionPurpose {
    OrderIntoGraveyard { preserve_known_identity: bool },
}

/// A policy-visible choice yielded by the generic effect interpreter. This is
/// intentionally typed and extensible: later library ordering, subset, Ward,
/// and Escape choices add variants here instead of adding card-specific
/// `EngineState::pending_*` fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PendingEffectChoice {
    ChooseOption {
        player: PlayerId,
        path: Vec<u16>,
        options: Vec<EffectOp>,
    },
    SelectTargets {
        player: PlayerId,
        path: Vec<u16>,
        selected: Vec<EffectTargetCandidate>,
        legal: Vec<EffectTargetCandidate>,
        min_targets: u16,
        max_targets: u16,
        ordered: bool,
        purpose: EffectTargetSelectionPurpose,
    },
}

impl PendingEffectChoice {
    pub fn player(&self) -> PlayerId {
        match self {
            PendingEffectChoice::ChooseOption { player, .. } => *player,
            PendingEffectChoice::SelectTargets { player, .. } => *player,
        }
    }

    pub fn structural_path(&self) -> &[u16] {
        match self {
            PendingEffectChoice::ChooseOption { path, .. }
            | PendingEffectChoice::SelectTargets { path, .. } => path,
        }
    }
}

/// Full in-state continuation for one resolving stack item. The complete
/// `StackItem`, execution context, remaining frames, and active typed choice
/// all participate in clone/equality/hash/serde, so snapshot/restore cannot
/// lose or alias a mid-resolution decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectContinuation {
    pub resolving_item: StackItem,
    pub ctx: ExecCtx,
    pub frames: Vec<EffectFrame>,
    pub choice: Option<PendingEffectChoice>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumableProgress {
    Complete(StackItem),
    Suspended,
}

/// Everything an effect program needs to resolve symbolic refs against a
/// concrete game: which object it's running for, who controls it, and the
/// targets chosen when it was cast/activated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Whether this program contains a policy-visible `Choice` anywhere in its
/// tree. Existing Burn/Rally programs stay on their frozen synchronous/legacy
/// continuation paths; only genuinely choice-bearing programs enter the v4
/// interpreter in this first migration slice.
pub fn contains_player_choice(op: &EffectOp) -> bool {
    match op {
        EffectOp::Sequence(ops) => ops.iter().any(contains_player_choice),
        EffectOp::Conditional { then, else_, .. } => {
            contains_player_choice(then) || contains_player_choice(else_)
        }
        EffectOp::Choice { options, .. } => {
            options.len() > 1 || options.iter().any(contains_player_choice)
        }
        // Whether the decision is needed depends on the revealed cards, but
        // the program must enter the resumable interpreter so a 2+ card
        // graveyard batch can yield its owner's ordering choice.
        EffectOp::RevealTopAndPartitionByType { .. } => true,
        _ => false,
    }
}

/// Starts a choice-bearing stack resolution and runs synchronously until it
/// either completes or yields a real player choice. Legacy-suspending leaves
/// are rejected up front in this first v4 slice: mixing them with remaining
/// frames would change the already-certified Burn/Rally completion timing.
pub fn begin_resumable_resolution(
    op: &EffectOp,
    ctx: &ExecCtx,
    resolving_item: StackItem,
    state: &mut GameState,
) -> Result<ResumableProgress, String> {
    if state.engine.pending_effect.is_some() {
        return Err("cannot begin an effect while another continuation is pending".to_string());
    }
    validate_resumable_program(op)?;
    state.engine.pending_effect = Some(EffectContinuation {
        resolving_item,
        ctx: ctx.clone(),
        frames: vec![EffectFrame::Program {
            op: op.clone(),
            path: Vec::new(),
        }],
        choice: None,
    });
    drive_resumable(state)
}

/// Resumes after `choose_resumable_option` installed the selected branch.
pub fn resume_resumable_resolution(state: &mut GameState) -> Result<ResumableProgress, String> {
    if state.engine.pending_effect.is_none() {
        return Err("no effect continuation is pending".to_string());
    }
    drive_resumable(state)
}

/// Records one selected option without executing it. The next engine advance
/// owns resumption, preserving the usual step/advance separation and making a
/// snapshot taken immediately after the action deterministic too.
pub fn choose_resumable_option(state: &mut GameState, option_index: u16) -> Result<(), String> {
    let continuation = state
        .engine
        .pending_effect
        .as_mut()
        .ok_or("no effect continuation is pending")?;
    let choice = continuation
        .choice
        .take()
        .ok_or("the pending effect is not waiting for a choice")?;
    match choice {
        PendingEffectChoice::ChooseOption {
            player,
            mut path,
            options,
        } => {
            let index = option_index as usize;
            let Some(selected) = options.get(index).cloned() else {
                continuation.choice = Some(PendingEffectChoice::ChooseOption {
                    player,
                    path,
                    options,
                });
                return Err(format!(
                    "effect option {option_index} is outside the available range"
                ));
            };
            path.push(option_index);
            continuation
                .frames
                .push(EffectFrame::Program { op: selected, path });
            Ok(())
        }
        PendingEffectChoice::SelectTargets { .. } => {
            continuation.choice = Some(choice);
            Err("the pending effect is not waiting for an option".to_string())
        }
    }
}

/// Records one target in an effect-owned selection. Exact-count selections
/// auto-append every forced remaining target, so an N-card ordering exposes
/// only N-1 picks. Validation precedes mutation, making stale, duplicate, or
/// wrong-shape actions byte-for-byte nonmutating.
pub fn choose_resumable_target(state: &mut GameState, target: Target) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    let choice = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        legal,
        selected,
        max_targets,
        ..
    } = choice
    else {
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    if selected.len() >= usize::from(*max_targets) {
        return Err("the pending effect target selection is already full".to_string());
    }
    let Some(position) = legal
        .iter()
        .position(|candidate| candidate.target == target)
    else {
        return Err(format!("{target:?} is not a legal remaining effect target"));
    };
    validate_effect_target_candidate(state, &legal[position])?;

    let continuation = state.engine.pending_effect.as_mut().unwrap();
    let PendingEffectChoice::SelectTargets {
        selected,
        legal,
        min_targets,
        max_targets,
        ordered,
        ..
    } = continuation.choice.as_mut().unwrap()
    else {
        unreachable!("validated target-selection choice above")
    };
    selected.push(legal.remove(position));

    let required = usize::from(*min_targets).saturating_sub(selected.len());
    if (*ordered && required == 1 && legal.len() == 1)
        || (!*ordered && required > 0 && required == legal.len())
    {
        selected.append(legal);
    }
    if selected.len() == usize::from(*max_targets) {
        complete_resumable_target_selection(continuation)?;
    }
    Ok(())
}

/// Finishes a generic variable-count selection once its minimum has been
/// met. Winding Way's graveyard ordering has `min == max`, so its forced
/// final card auto-completes and this action is never legal there.
pub fn finish_resumable_target_selection(state: &mut GameState) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    let choice = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        selected,
        min_targets,
        ..
    } = choice
    else {
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    if selected.len() < usize::from(*min_targets) {
        return Err("the pending effect target selection has not reached its minimum".to_string());
    }
    complete_resumable_target_selection(state.engine.pending_effect.as_mut().unwrap())
}

fn complete_resumable_target_selection(
    continuation: &mut EffectContinuation,
) -> Result<(), String> {
    let choice = continuation
        .choice
        .take()
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        path,
        selected,
        purpose,
        ..
    } = choice
    else {
        continuation.choice = Some(choice);
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    let objects = selected
        .into_iter()
        .map(|candidate| {
            let binding = candidate.expected_object.ok_or_else(|| {
                "zone-order selection target lacks an object-incarnation binding".to_string()
            })?;
            if candidate.target != Target::Object(binding.object) {
                return Err("zone-order selection target/binding mismatch".to_string());
            }
            Ok(binding)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (to_zone, preserve_known_identity) = match purpose {
        EffectTargetSelectionPurpose::OrderIntoGraveyard {
            preserve_known_identity,
        } => (Zone::Graveyard, preserve_known_identity),
    };
    continuation.frames.push(EffectFrame::MoveObjectsBatch {
        objects,
        to_zone,
        preserve_known_identity,
        order_resolved: true,
        path,
    });
    Ok(())
}

fn validate_effect_target_candidate(
    state: &GameState,
    candidate: &EffectTargetCandidate,
) -> Result<(), String> {
    let Some(binding) = candidate.expected_object else {
        return Ok(());
    };
    if candidate.target != Target::Object(binding.object) {
        return Err("effect target/binding mismatch".to_string());
    }
    let object = state
        .objects
        .try_get(binding.object)
        .ok_or_else(|| format!("effect target object {} no longer exists", binding.object.0))?;
    if object.zone != binding.expected_zone
        || object.zone_change_count != binding.expected_zone_change_count
    {
        return Err(format!(
            "effect target object {} changed incarnation: expected {:?}/{} but found {:?}/{}",
            binding.object.0,
            binding.expected_zone,
            binding.expected_zone_change_count,
            object.zone,
            object.zone_change_count
        ));
    }
    Ok(())
}

pub fn validate_pending_effect_choice(state: &GameState) -> Result<(), String> {
    let Some(choice) = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
    else {
        return Ok(());
    };
    if let PendingEffectChoice::SelectTargets {
        selected, legal, ..
    } = choice
    {
        for candidate in selected.iter().chain(legal) {
            validate_effect_target_candidate(state, candidate)?;
        }
    }
    Ok(())
}

fn validate_resumable_program(op: &EffectOp) -> Result<(), String> {
    match op {
        EffectOp::Sequence(ops) => {
            for inner in ops {
                validate_resumable_program(inner)?;
            }
        }
        EffectOp::Conditional { then, else_, .. } => {
            validate_resumable_program(then)?;
            validate_resumable_program(else_)?;
        }
        EffectOp::Choice { options, .. } => {
            for option in options {
                validate_resumable_program(option)?;
            }
        }
        EffectOp::DiscardCards { .. }
        | EffectOp::MayPayCostThen { .. }
        | EffectOp::OfferAffectedPlayerSpellCopy { .. } => {
            return Err(
                "choice-bearing programs cannot yet mix legacy-suspending effect leaves"
                    .to_string(),
            );
        }
        _ => {}
    }
    Ok(())
}

fn drive_resumable(state: &mut GameState) -> Result<ResumableProgress, String> {
    let mut continuation = state
        .engine
        .pending_effect
        .take()
        .ok_or("no effect continuation is pending")?;
    if continuation.choice.is_some() {
        state.engine.pending_effect = Some(continuation);
        return Ok(ResumableProgress::Suspended);
    }

    while let Some(frame) = continuation.frames.pop() {
        let EffectFrame::Program { op, path } = frame else {
            let EffectFrame::MoveObjectsBatch {
                objects,
                to_zone,
                preserve_known_identity,
                order_resolved,
                path,
            } = frame
            else {
                unreachable!()
            };
            if to_zone == Zone::Graveyard && objects.len() >= 2 && !order_resolved {
                for binding in &objects {
                    validate_effect_object_binding(state, *binding)?;
                }
                let player = state.objects.get(objects[0].object).owner;
                assert!(
                    objects
                        .iter()
                        .all(|binding| state.objects.get(binding.object).owner == player),
                    "one graveyard-order batch must contain cards from one owner"
                );
                let count = objects
                    .len()
                    .try_into()
                    .expect("effect target count fits the u16 public contract");
                continuation.choice = Some(PendingEffectChoice::SelectTargets {
                    player,
                    path,
                    selected: Vec::new(),
                    legal: objects
                        .into_iter()
                        .map(|binding| EffectTargetCandidate {
                            target: Target::Object(binding.object),
                            expected_object: Some(binding),
                        })
                        .collect(),
                    min_targets: count,
                    max_targets: count,
                    ordered: true,
                    purpose: EffectTargetSelectionPurpose::OrderIntoGraveyard {
                        preserve_known_identity,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            commit_zone_change_batch(state, &objects, to_zone, preserve_known_identity)?;
            continue;
        };
        match op {
            EffectOp::Sequence(ops) => {
                for (index, inner) in ops.into_iter().enumerate().rev() {
                    let mut inner_path = path.clone();
                    inner_path.push(index as u16);
                    continuation.frames.push(EffectFrame::Program {
                        op: inner,
                        path: inner_path,
                    });
                }
            }
            EffectOp::Conditional { cond, then, else_ } => {
                let branch = if eval_cond(&cond, &continuation.ctx, state) {
                    0
                } else {
                    1
                };
                let mut branch_path = path;
                branch_path.push(branch);
                continuation.frames.push(EffectFrame::Program {
                    op: if branch == 0 { *then } else { *else_ },
                    path: branch_path,
                });
            }
            EffectOp::Choice {
                controller,
                mut options,
            } => match options.len() {
                0 => {}
                1 => {
                    let mut option_path = path;
                    option_path.push(0);
                    continuation.frames.push(EffectFrame::Program {
                        op: options.remove(0),
                        path: option_path,
                    });
                }
                _ => {
                    let player = continuation.ctx.resolve_player(controller, state);
                    continuation.choice = Some(PendingEffectChoice::ChooseOption {
                        player,
                        path,
                        options,
                    });
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            },
            EffectOp::RevealTopAndPartitionByType {
                player,
                count,
                card_type,
                matching_to,
                rest_to,
            } => {
                let player = continuation.ctx.resolve_player(player, state);
                let (matching, rest) =
                    reveal_top_and_partition(state, player, count, card_type, matching_to, rest_to);
                // Frames are LIFO: push the rest group first so the matching
                // group is committed/ordered first, mirroring XMage's two
                // separate moveCards calls.
                for (group_index, objects, to_zone) in
                    [(1_u16, rest, rest_to), (0_u16, matching, matching_to)]
                {
                    let mut group_path = path.clone();
                    group_path.push(group_index);
                    continuation.frames.push(EffectFrame::MoveObjectsBatch {
                        objects,
                        to_zone,
                        preserve_known_identity: true,
                        order_resolved: false,
                        path: group_path,
                    });
                }
            }
            leaf => execute(&leaf, &continuation.ctx, state),
        }
    }

    Ok(ResumableProgress::Complete(continuation.resolving_item))
}

impl ExecCtx {
    pub fn no_targets(source: ObjectId, controller: PlayerId) -> ExecCtx {
        ExecCtx {
            source,
            controller,
            targets: Vec::new(),
            discarded: Vec::new(),
            kicked: false,
        }
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
            PlayerRef::ObjectController(oref) => {
                state.objects.get(self.resolve_object(oref)).controller
            }
        }
    }
}

fn reveal_top_and_partition(
    state: &mut GameState,
    player: PlayerId,
    count: u8,
    card_type: CardType,
    matching_to: Zone,
    rest_to: Zone,
) -> (Vec<EffectObjectBinding>, Vec<EffectObjectBinding>) {
    assert!(
        !matches!(matching_to, Zone::Library | Zone::Stack)
            && !matches!(rest_to, Zone::Library | Zone::Stack),
        "library partition destinations must be ordinary nonlibrary card zones"
    );
    let revealed = state.players[player.index()].library
        [..usize::from(count).min(state.players[player.index()].library.len())]
        .to_vec();
    let mut matching = Vec::new();
    let mut rest = Vec::new();
    for object in revealed.iter().copied() {
        let live = state.objects.get(object);
        let binding = EffectObjectBinding {
            object,
            expected_zone: Zone::Library,
            expected_zone_change_count: live.zone_change_count,
        };
        let def = &crate::card_def::CARD_DEFS[live.card_def as usize];
        if def.has_type(card_type) {
            matching.push(binding);
        } else {
            rest.push(binding);
        }
    }

    // "Reveal" is public, unlike a private look. Record the exact prefix
    // for both perspectives before any member leaves and shifts the
    // remaining position facts.
    for observer in [PlayerId::P0, PlayerId::P1] {
        state.reveal_library_top(observer, player, revealed.len());
    }
    (matching, rest)
}

fn validate_effect_object_binding(
    state: &GameState,
    binding: EffectObjectBinding,
) -> Result<(), String> {
    let object = state
        .objects
        .try_get(binding.object)
        .ok_or_else(|| format!("effect object {} no longer exists", binding.object.0))?;
    if object.zone != binding.expected_zone
        || object.zone_change_count != binding.expected_zone_change_count
    {
        return Err(format!(
            "effect object {} changed incarnation: expected {:?}/{} but found {:?}/{}",
            binding.object.0,
            binding.expected_zone,
            binding.expected_zone_change_count,
            object.zone,
            object.zone_change_count
        ));
    }
    Ok(())
}

fn commit_zone_change_batch(
    state: &mut GameState,
    objects: &[EffectObjectBinding],
    to_zone: Zone,
    preserve_known_identity: bool,
) -> Result<(), String> {
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
    }
    let events = objects
        .iter()
        .map(|binding| {
            if preserve_known_identity {
                event::ProposedEvent::zone_change_preserving_known_identity(binding.object, to_zone)
            } else {
                event::ProposedEvent::zone_change(binding.object, to_zone)
            }
        })
        .collect();
    event::propose_and_commit_batch(state, events);
    Ok(())
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
            event::propose_and_commit(
                state,
                event::ProposedEvent::damage(ctx.source, target, *amount),
            );
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
        EffectOp::RevealTopAndPartitionByType {
            player,
            count,
            card_type,
            matching_to,
            rest_to,
        } => {
            let player = ctx.resolve_player(*player, state);
            let (matching, rest) =
                reveal_top_and_partition(state, player, *count, *card_type, *matching_to, *rest_to);
            assert!(
                !(*matching_to == Zone::Graveyard && matching.len() >= 2
                    || *rest_to == Zone::Graveyard && rest.len() >= 2),
                "a multi-card graveyard partition must use the resumable interpreter"
            );
            for (objects, destination) in [(&matching, *matching_to), (&rest, *rest_to)] {
                commit_zone_change_batch(state, objects, destination, true)
                    .expect("freshly revealed batch bindings remain valid");
            }
        }
        EffectOp::MoveObject { object, to_zone } => {
            let object = ctx.resolve_object(*object);
            let live_stack_spell = state
                .stack
                .iter()
                .find(|item| {
                    item.source == object && item.kind == crate::state::StackItemKind::Spell
                })
                .cloned();
            if *to_zone != Zone::Stack {
                if let Some(item) = live_stack_spell {
                    if item.is_copy {
                        // A copied spell is not a card and never enters a
                        // destination zone when it leaves the stack (707.10a).
                        event::cease_to_exist(state, object);
                        return;
                    }
                    if item.is_flashback {
                        // Flashback's replacement applies to every attempted
                        // move away from the stack, including being countered.
                        event::propose_and_commit(
                            state,
                            event::ProposedEvent::zone_change(object, Zone::Exile),
                        );
                        return;
                    }
                }
            }
            event::propose_and_commit(state, event::ProposedEvent::zone_change(object, *to_zone));
        }
        EffectOp::TapObject { object } => {
            let object = ctx.resolve_object(*object);
            event::propose_and_commit(state, event::ProposedEvent::tap(object));
        }
        EffectOp::AddMana { player, colors } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(
                state,
                event::ProposedEvent::mana_add(player, colors.clone()),
            );
        }
        EffectOp::DiscardCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            state.engine.pending_discard = Some(crate::engine::PendingDiscard {
                player,
                count: *count,
                resume: crate::engine::DiscardResume::None,
            });
        }
        EffectOp::CreateToken {
            token_def,
            controller,
        } => {
            let token = crate::card_def::CARD_DEFS
                .get(*token_def as usize)
                .unwrap_or_else(|| panic!("CreateToken references unknown definition {token_def}"));
            assert!(
                token.is_token && token.is_executable() && token.has_full_support(),
                "CreateToken requires a fully supported executable token definition, got {} ({:?}, is_token={})",
                token.name,
                token.capability,
                token.is_token
            );
            let controller = ctx.resolve_player(*controller, state);
            event::propose_and_commit(
                state,
                event::ProposedEvent::create_token(*token_def, controller),
            );
        }
        EffectOp::MayPayCostThen {
            discard,
            sacrifice_lands,
            then,
        } => {
            let discard_payable = *discard > 0
                && state.players[ctx.controller.index()].hand.len() >= *discard as usize;
            let sacrifice_payable = *sacrifice_lands > 0
                && crate::engine::count_controlled_lands(ctx.controller, state)
                    >= *sacrifice_lands as u32;
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
            let mut events = vec![event::ProposedEvent::damage(
                ctx.source,
                Target::Player(opponent),
                *amount,
            )];
            events.extend(
                state.players[opponent.index()]
                    .battlefield
                    .iter()
                    .copied()
                    .filter_map(|id| {
                        let def =
                            &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                        def.has_type(crate::card_def::CardType::Creature).then(|| {
                            event::ProposedEvent::damage(ctx.source, Target::Object(id), *amount)
                        })
                    }),
            );
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::PumpControlled {
            filter,
            power,
            toughness,
            grant_haste,
        } => {
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
                state.engine.until_end_of_turn.push(
                    crate::engine::UntilEndOfTurnEffect::ResolvedSetEffect {
                        object_ids,
                        layer,
                        timestamp,
                        duration: crate::engine::EffectDuration::EndOfTurn,
                        power: *power,
                        toughness: *toughness,
                        grant_haste: *grant_haste,
                    },
                );
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
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::zone_change(top, Zone::Exile),
                );
                let expiry = match duration {
                    ImpulseDuration::EndOfTurn => crate::engine::PlayPermissionExpiry::EndOfTurn,
                    ImpulseDuration::UntilOwnersNextTurn => {
                        crate::engine::PlayPermissionExpiry::UntilHoldersNextTurn {
                            holder_turn_started: false,
                        }
                    }
                };
                let def = &crate::card_def::CARD_DEFS[state.objects.get(top).card_def as usize];
                let play_or_cast = if def.is_playable_land() {
                    crate::engine::PlayOrCast::Play
                } else if def.is_castable() {
                    crate::engine::PlayOrCast::Cast
                } else {
                    // Exiling still happened, but an unsupported definition
                    // never receives an executable permission. This is the
                    // runtime half of the fail-closed deck preflight.
                    continue;
                };
                state
                    .engine
                    .exile_play_permissions
                    .push(crate::engine::PlayPermission {
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
        EffectOp::OfferAffectedPlayerSpellCopy { affected } => {
            let target = ctx.resolve_target(*affected);
            let decider = match target {
                Target::Player(p) => p,
                Target::Object(id) => state.objects.get(id).controller,
            };
            state.engine.pending_spell_copy = Some(crate::engine::PendingSpellCopy {
                resolving_source: ctx.source,
                player: decider,
                inherited_target: target,
                stage: crate::engine::SpellCopyStage::Payment,
                copy_source: None,
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
        EffectCond::LandfallThisTurn => {
            state.players[ctx.controller.index()].lands_played_this_turn > 0
        }
        EffectCond::TargetInZone(idx, zone) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) if *zone == Zone::Stack => {
                state.stack.iter().any(|item| item.source == *id)
            }
            Some(Target::Object(id)) => state.objects.get(*id).zone == *zone,
            _ => false,
        },
        EffectCond::TargetIsColor(idx, color) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) => {
                let def_idx = state.objects.get(*id).card_def;
                crate::card_def::CARD_DEFS[def_idx as usize]
                    .colors
                    .contains(color)
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
    use crate::event::CommittedEvent;
    use crate::ids::PlayerId;

    fn two_card_libraries() -> GameState {
        GameState::new_from_libraries(&[1, 2], &[3, 4], |c| format!("card-{c}"), 1)
    }

    #[test]
    fn sequence_runs_every_leaf_in_order() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 2,
            },
            EffectOp::GainLife {
                player: PlayerRef::Controller,
                amount: 5,
            },
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
            then: Box::new(EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 3,
            }),
            else_: Box::new(EffectOp::Sequence(vec![])),
        };
        execute(&taken, &ctx, &mut state);
        assert_eq!(state.players[0].life, 17);

        let not_taken = EffectOp::Conditional {
            cond: EffectCond::Never,
            then: Box::new(EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 100,
            }),
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
        execute(
            &EffectOp::DealDamage {
                target: TargetRef::Target(0),
                amount: 3,
            },
            &ctx,
            &mut state,
        );
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
        execute(
            &EffectOp::DealDamage {
                target: TargetRef::Target(0),
                amount: 4,
            },
            &ctx,
            &mut state,
        );
        assert_eq!(state.objects.get(creature).damage, 4);
    }

    #[test]
    fn draw_cards_leaf_draws_the_requested_count() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        execute(
            &EffectOp::DrawCards {
                player: PlayerRef::Controller,
                count: 2,
            },
            &ctx,
            &mut state,
        );
        assert_eq!(state.players[0].hand.len(), 2);
    }

    fn card_ids(names: &[&str]) -> Vec<u16> {
        names
            .iter()
            .map(|name| crate::card_def::card_id_by_name(name).unwrap())
            .collect()
    }

    #[test]
    fn reveal_top_partition_is_public_ordered_and_not_a_draw() {
        let definitions = card_ids(&[
            "Elvish Mystic",
            "Quirion Ranger",
            "Llanowar Elves",
            "Lightning Bolt",
            "Island",
        ]);
        let mut state = GameState::new_from_libraries(
            &definitions,
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            9,
        );
        let original = state.players[0].library.clone();
        state.reveal_library_top(PlayerId::P1, PlayerId::P0, 5);
        let ctx = ExecCtx::no_targets(original[0], PlayerId::P0);

        execute(
            &EffectOp::RevealTopAndPartitionByType {
                player: PlayerRef::Controller,
                count: 4,
                card_type: CardType::Creature,
                matching_to: Zone::Hand,
                rest_to: Zone::Graveyard,
            },
            &ctx,
            &mut state,
        );

        assert_eq!(
            state.players[0].hand,
            vec![original[0], original[1], original[2]]
        );
        assert_eq!(state.players[0].graveyard, vec![original[3]]);
        assert_eq!(state.players[0].library, vec![original[4]]);
        assert_eq!(
            state
                .known_hand_cards(PlayerId::P1, PlayerId::P0)
                .iter()
                .map(|entry| entry.object)
                .collect::<Vec<_>>(),
            vec![original[0], original[1], original[2]]
        );
        assert!(state
            .known_hand_cards(PlayerId::P0, PlayerId::P0)
            .is_empty());
        assert_eq!(
            state
                .known_library_cards(PlayerId::P1, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, original[4])]
        );
        assert_eq!(state.players[0].draws_this_turn, 0);
        assert!(!state.players[0].drew_from_empty);
        assert!(state
            .engine
            .event_history
            .iter()
            .all(|event| !matches!(event, CommittedEvent::Draw { .. })));
        assert_eq!(
            state
                .engine
                .event_history
                .iter()
                .filter_map(|event| match event {
                    CommittedEvent::ZoneChange { object, .. } => Some(*object),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![original[0], original[1], original[2], original[3]]
        );
        for &object in &original[..4] {
            assert_eq!(state.objects.get(object).zone_change_count, 1);
        }
    }

    #[test]
    fn reveal_top_partition_handles_short_empty_and_zero_hit_libraries() {
        let definitions = card_ids(&["Lightning Bolt"]);
        let mut short = GameState::new_from_libraries(
            &definitions,
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            1,
        );
        let original = short.players[0].library.clone();
        let ctx = ExecCtx::no_targets(original[0], PlayerId::P0);
        let op = EffectOp::RevealTopAndPartitionByType {
            player: PlayerRef::Controller,
            count: 4,
            card_type: CardType::Creature,
            matching_to: Zone::Hand,
            rest_to: Zone::Graveyard,
        };
        execute(&op, &ctx, &mut short);
        assert!(short.players[0].hand.is_empty());
        assert_eq!(short.players[0].graveyard, original);
        assert!(!short.players[0].drew_from_empty);

        let mut empty = GameState::new_from_libraries(&[], &[], |_| String::new(), 1);
        let before = empty.clone();
        execute(
            &op,
            &ExecCtx::no_targets(ObjectId(0), PlayerId::P0),
            &mut empty,
        );
        assert_eq!(empty, before);
    }

    #[test]
    fn tap_and_add_mana_leaves_compose_a_mana_ability() {
        let mut state = two_card_libraries();
        let land = state.draw_card(PlayerId::P0).unwrap();
        state.move_hand_to_battlefield(PlayerId::P0, land);
        let ctx = ExecCtx::no_targets(land, PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::TapObject {
                object: ObjectRef::ThisSource,
            },
            EffectOp::AddMana {
                player: PlayerRef::Controller,
                colors: vec![ManaColor::R],
            },
        ]);
        execute(&op, &ctx, &mut state);
        assert!(state.objects.get(land).tapped);
        assert_eq!(state.players[0].mana_pool[ManaColor::R.pool_index()], 1);
    }

    #[test]
    fn create_token_requires_and_materializes_a_full_token_definition() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        let blood = crate::card_def::card_id_by_name("Blood Token").unwrap();
        execute(
            &EffectOp::CreateToken {
                token_def: blood,
                controller: PlayerRef::Controller,
            },
            &ctx,
            &mut state,
        );
        let created = *state.players[0].battlefield.last().unwrap();
        assert_eq!(state.objects.get(created).card_def, blood);
        assert!(crate::card_def::CARD_DEFS[blood as usize].has_full_support());
    }

    #[test]
    #[should_panic(expected = "CreateToken requires a fully supported executable token definition")]
    fn create_token_fails_loudly_for_a_nontoken_definition() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        execute(
            &EffectOp::CreateToken {
                token_def: crate::card_def::card_id_by_name("Island").unwrap(),
                controller: PlayerRef::Controller,
            },
            &ctx,
            &mut state,
        );
    }

    #[test]
    fn impulse_draw_exiles_but_does_not_authorize_an_unsupported_card() {
        let landscape = crate::card_def::card_id_by_name("Twisted Landscape").unwrap();
        let mut state = GameState::new_from_libraries(
            &[landscape],
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            1,
        );
        let card = state.players[0].library[0];
        let ctx = ExecCtx::no_targets(card, PlayerId::P0);

        execute(
            &EffectOp::ImpulseDraw {
                count: 1,
                duration: ImpulseDuration::EndOfTurn,
            },
            &ctx,
            &mut state,
        );

        assert!(state.players[0].library.is_empty());
        assert_eq!(state.objects.get(card).zone, Zone::Exile);
        assert!(state.exile.contains(&card));
        assert!(state.engine.exile_play_permissions.is_empty());
    }
}
