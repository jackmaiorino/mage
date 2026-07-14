//! propose -> replace/prevent -> commit.
//!
//! `effect::execute` never mutates `GameState` directly: every leaf op
//! builds a `ProposedEvent` and calls `propose_and_commit`, which runs the
//! replacement/prevention pass (`apply_replacements`) and then, if
//! anything survived, `commit`. `commit` is the *only* function that
//! mutates `GameState` in response to a game event, and it appends the
//! resulting `CommittedEvent` to `state.engine.event_log` for
//! `trigger::collect_and_process` to drain after the resolution finishes.

use crate::ids::{ObjectId, PlayerId};
use crate::mana::ManaColor;
use crate::state::{GameState, Target, Zone};
use serde::{Deserialize, Serialize};

pub type ReplacementId = u32;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplacementEffectKind {
    /// Prevent the next `remaining` damage that would be dealt to `target`.
    /// Synthetic -- no pool card grants this yet -- but proves the
    /// replacement pipeline shape end-to-end; see
    /// `tests::prevention_shield_absorbs_then_expires`.
    PreventNextDamage { target: Target, remaining: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActiveReplacement {
    pub id: ReplacementId,
    pub source: ObjectId,
    pub kind: ReplacementEffectKind,
}

// ---------------------------------------------------------------- proposed

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DamageProposed {
    pub source: ObjectId,
    pub target: Target,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneChangeProposed {
    pub object: ObjectId,
    pub to_zone: Zone,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeLossProposed {
    pub player: PlayerId,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeGainProposed {
    pub player: PlayerId,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawProposed {
    pub player: PlayerId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TapProposed {
    pub object: ObjectId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManaAddProposed {
    pub player: PlayerId,
    pub colors: Vec<ManaColor>,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTokenProposed {
    pub token_def: u16,
    pub controller: PlayerId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposedEvent {
    Damage(DamageProposed),
    ZoneChange(ZoneChangeProposed),
    LifeLoss(LifeLossProposed),
    LifeGain(LifeGainProposed),
    Draw(DrawProposed),
    Tap(TapProposed),
    ManaAdd(ManaAddProposed),
    CreateToken(CreateTokenProposed),
}

impl ProposedEvent {
    pub fn damage(source: ObjectId, target: Target, amount: i32) -> ProposedEvent {
        ProposedEvent::Damage(DamageProposed { source, target, amount, touched_by: Vec::new() })
    }
    pub fn zone_change(object: ObjectId, to_zone: Zone) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed { object, to_zone, touched_by: Vec::new() })
    }
    pub fn life_loss(player: PlayerId, amount: i32) -> ProposedEvent {
        ProposedEvent::LifeLoss(LifeLossProposed { player, amount, touched_by: Vec::new() })
    }
    pub fn life_gain(player: PlayerId, amount: i32) -> ProposedEvent {
        ProposedEvent::LifeGain(LifeGainProposed { player, amount, touched_by: Vec::new() })
    }
    pub fn draw(player: PlayerId) -> ProposedEvent {
        ProposedEvent::Draw(DrawProposed { player, touched_by: Vec::new() })
    }
    pub fn tap(object: ObjectId) -> ProposedEvent {
        ProposedEvent::Tap(TapProposed { object, touched_by: Vec::new() })
    }
    pub fn mana_add(player: PlayerId, colors: Vec<ManaColor>) -> ProposedEvent {
        ProposedEvent::ManaAdd(ManaAddProposed { player, colors, touched_by: Vec::new() })
    }
    pub fn create_token(token_def: u16, controller: PlayerId) -> ProposedEvent {
        ProposedEvent::CreateToken(CreateTokenProposed { token_def, controller, touched_by: Vec::new() })
    }

    fn touched_by(&self) -> &[ReplacementId] {
        match self {
            ProposedEvent::Damage(e) => &e.touched_by,
            ProposedEvent::ZoneChange(e) => &e.touched_by,
            ProposedEvent::LifeLoss(e) => &e.touched_by,
            ProposedEvent::LifeGain(e) => &e.touched_by,
            ProposedEvent::Draw(e) => &e.touched_by,
            ProposedEvent::Tap(e) => &e.touched_by,
            ProposedEvent::ManaAdd(e) => &e.touched_by,
            ProposedEvent::CreateToken(e) => &e.touched_by,
        }
    }

    fn mark_touched(&mut self, id: ReplacementId) {
        let v = match self {
            ProposedEvent::Damage(e) => &mut e.touched_by,
            ProposedEvent::ZoneChange(e) => &mut e.touched_by,
            ProposedEvent::LifeLoss(e) => &mut e.touched_by,
            ProposedEvent::LifeGain(e) => &mut e.touched_by,
            ProposedEvent::Draw(e) => &mut e.touched_by,
            ProposedEvent::Tap(e) => &mut e.touched_by,
            ProposedEvent::ManaAdd(e) => &mut e.touched_by,
            ProposedEvent::CreateToken(e) => &mut e.touched_by,
        };
        v.push(id);
    }
}

// --------------------------------------------------------------- committed

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommittedEvent {
    Damage { source: ObjectId, target: Target, amount: i32 },
    ZoneChange { object: ObjectId, from: Zone, to: Zone },
    LifeLoss { player: PlayerId, amount: i32 },
    LifeGain { player: PlayerId, amount: i32 },
    /// `object` is `None` when the draw was attempted against an empty
    /// library; SBA picks that up as a loss condition (704.5c).
    Draw { player: PlayerId, object: Option<ObjectId> },
    Tap { object: ObjectId },
    ManaAdded { player: PlayerId, colors: Vec<ManaColor> },
    CreateToken { object: ObjectId, token_def: u16, controller: PlayerId },
    /// Logged by `engine::finalize_cast` the moment a spell is placed on
    /// the stack (not routed through `propose_and_commit`: casting a spell
    /// is an engine action with no replaceable "cast" event in this
    /// increment's scope, same rationale as the Hand->Stack zone move
    /// itself -- see `commit_zone_change`'s doc). Exists purely so
    /// `trigger::TriggerCondition::CastInstantOrSorcery` (Guttersnipe) has
    /// something to match against; it is still appended to both
    /// `event_log` (drained by `trigger::collect_and_process`) and
    /// `event_history` for consistency with every other committed event.
    SpellCast { spell: ObjectId, controller: PlayerId },
}

/// Runs the replace/prevent pass to a fixed point: repeatedly finds an
/// active replacement that applies to `event` and hasn't already touched it
/// (loop-prevention via `touched_by`), applies it, and marks it touched.
/// Returns `None` if the event ends up fully prevented.
pub fn apply_replacements(state: &mut GameState, mut proposed: ProposedEvent) -> Option<ProposedEvent> {
    loop {
        let hit = state
            .engine
            .active_replacements
            .iter()
            .find(|r| !proposed.touched_by().contains(&r.id) && replacement_applies(r, &proposed))
            .cloned();

        let Some(repl) = hit else {
            return Some(proposed);
        };

        proposed.mark_touched(repl.id);
        match replacement_apply(&repl, proposed, state) {
            Some(rewritten) => proposed = rewritten,
            None => return None,
        }
    }
}

fn replacement_applies(repl: &ActiveReplacement, proposed: &ProposedEvent) -> bool {
    match (&repl.kind, proposed) {
        (ReplacementEffectKind::PreventNextDamage { target, remaining }, ProposedEvent::Damage(d)) => {
            *remaining > 0 && d.target == *target
        }
        _ => false,
    }
}

/// Applies `repl` to `proposed`, mutating the replacement's own bookkeeping
/// in `state.engine.active_replacements` (e.g. decrementing/expiring a
/// prevention shield's remaining count) as a side effect.
fn replacement_apply(
    repl: &ActiveReplacement,
    proposed: ProposedEvent,
    state: &mut GameState,
) -> Option<ProposedEvent> {
    match (&repl.kind, proposed) {
        (ReplacementEffectKind::PreventNextDamage { remaining, .. }, ProposedEvent::Damage(mut d)) => {
            let prevented = (*remaining).min(d.amount);
            d.amount -= prevented;

            if let Some(slot) = state.engine.active_replacements.iter_mut().find(|r| r.id == repl.id) {
                // Only variant today, but matched explicitly (rather than
                // destructured directly) so this stays a real pattern match
                // -- and a compile error, not a silent no-op -- the moment
                // a second `ReplacementEffectKind` is added.
                #[allow(irrefutable_let_patterns)]
                if let ReplacementEffectKind::PreventNextDamage { remaining, .. } = &mut slot.kind {
                    *remaining -= prevented;
                }
            }
            state.engine.active_replacements.retain(|r| {
                !matches!(&r.kind, ReplacementEffectKind::PreventNextDamage { remaining, .. } if *remaining <= 0)
            });

            if d.amount <= 0 {
                None
            } else {
                Some(ProposedEvent::Damage(d))
            }
        }
        (_, other) => Some(other),
    }
}

/// Convenience: replace/prevent then commit if anything survived.
pub fn propose_and_commit(state: &mut GameState, event: ProposedEvent) {
    if let Some(final_event) = apply_replacements(state, event) {
        commit(state, final_event);
    }
}

/// Applies the (possibly rewritten) proposal to `GameState` and appends the
/// resulting `CommittedEvent` to the event log for this resolution.
pub fn commit(state: &mut GameState, event: ProposedEvent) {
    let committed = match event {
        ProposedEvent::Damage(d) => {
            match d.target {
                Target::Object(id) => {
                    let obj = state.objects.get_mut(id);
                    obj.damage = obj.damage.saturating_add(d.amount.max(0) as u16);
                }
                Target::Player(p) => {
                    state.players[p.index()].life -= d.amount;
                }
            }
            CommittedEvent::Damage { source: d.source, target: d.target, amount: d.amount }
        }
        ProposedEvent::ZoneChange(z) => {
            let from = state.objects.get(z.object).zone;
            commit_zone_change(state, z.object, z.to_zone);
            CommittedEvent::ZoneChange { object: z.object, from, to: z.to_zone }
        }
        ProposedEvent::LifeLoss(l) => {
            state.players[l.player.index()].life -= l.amount;
            CommittedEvent::LifeLoss { player: l.player, amount: l.amount }
        }
        ProposedEvent::LifeGain(g) => {
            state.players[g.player.index()].life += g.amount;
            CommittedEvent::LifeGain { player: g.player, amount: g.amount }
        }
        ProposedEvent::Draw(d) => {
            let empty_before = state.players[d.player.index()].library.is_empty();
            let drawn = state.draw_card(d.player);
            if empty_before {
                state.players[d.player.index()].drew_from_empty = true;
            }
            if drawn.is_some() {
                state.players[d.player.index()].draws_this_turn += 1;
            }
            CommittedEvent::Draw { player: d.player, object: drawn }
        }
        ProposedEvent::Tap(t) => {
            state.objects.get_mut(t.object).tapped = true;
            CommittedEvent::Tap { object: t.object }
        }
        ProposedEvent::ManaAdd(m) => {
            for &c in &m.colors {
                state.players[m.player.index()].mana_pool[c.pool_index()] += 1;
            }
            CommittedEvent::ManaAdded { player: m.player, colors: m.colors }
        }
        ProposedEvent::CreateToken(t) => {
            let name = crate::card_def::CARD_DEFS[t.token_def as usize].name.to_string();
            let object = state.objects.push(crate::state::GameObject {
                card_def: t.token_def,
                name,
                owner: t.controller,
                controller: t.controller,
                zone: Zone::Battlefield,
                tapped: false,
                // A token entering the battlefield is exactly as summoning-
                // sick as any other permanent that just entered (see
                // `commit_zone_change`'s identical `= true` a few lines
                // down for the ordinary cast/move path) -- this was
                // hardcoded `false` and never flipped, the one "enters
                // battlefield" path that skipped setting it. Found via the
                // branch-differential pilot (Sol #89/#91): a Blood Token's
                // controlled-since-turn-start flag disagreed with the
                // reference engine's `wasControlledFromStartOfControllerTurn()`
                // immediately after Voldaren Epicure's ETB created it.
                summoning_sick: true,
                damage: 0,
                counters: Default::default(),
                attachments: Vec::new(),
                plotted_turn: None,
                zone_change_count: 0,
            });
            state.players[t.controller.index()].battlefield.push(object);
            CommittedEvent::CreateToken { object, token_def: t.token_def, controller: t.controller }
        }
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Runs the replace/prevent pass independently on every event in `events`
/// (each is evaluated against the currently-active replacements as if it
/// were the only proposal in flight -- true simultaneity: none of them can
/// see or react to one another), then commits every survivor back-to-back
/// with no SBA/trigger check interleaved. Used for combat damage (510.2:
/// all of it happens at once); the caller is responsible for running SBAs
/// / trigger collection exactly once after the whole batch (see
/// `engine::deal_combat_damage`), not per event.
pub fn propose_and_commit_batch(state: &mut GameState, events: Vec<ProposedEvent>) {
    let survivors: Vec<ProposedEvent> = events.into_iter().filter_map(|e| apply_replacements(state, e)).collect();
    for e in survivors {
        commit(state, e);
    }
}

/// Logs a `SpellCast` marker with no accompanying state mutation (casting
/// itself -- moving hand to stack -- is handled by
/// `engine::move_hand_to_stack`; this is purely a trigger-matching hook).
/// Not named `commit_*` and not routed through `propose_and_commit`
/// because there is no proposed/replaceable form of "a spell was cast" in
/// this increment's scope (countering a spell removes it from the stack
/// later, it doesn't replace the cast event itself).
pub fn log_spell_cast(state: &mut GameState, spell: ObjectId, controller: PlayerId) {
    let committed = CommittedEvent::SpellCast { spell, controller };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Zone bookkeeping shared by every `MoveObject` effect leaf. "Hand ->
/// Stack" (casting) is deliberately not reachable here: putting a spell on
/// the stack is an engine action (see `engine::begin_cast`), never
/// something a card's own effect program does.
fn commit_zone_change(state: &mut GameState, id: ObjectId, to_zone: Zone) {
    let owner = state.objects.get(id).owner;
    let from_zone = state.objects.get(id).zone;

    remove_from_zone(state, owner, id, from_zone);

    match to_zone {
        Zone::Library => state.players[owner.index()].library.insert(0, id),
        Zone::Hand => state.players[owner.index()].hand.push(id),
        Zone::Battlefield => state.players[owner.index()].battlefield.push(id),
        Zone::Graveyard => state.players[owner.index()].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Command => state.command.push(id),
        Zone::Stack => panic!("MoveObject to Stack is an engine action, not an effect leaf"),
    }

    let obj = state.objects.get_mut(id);
    obj.zone = to_zone;
    // CR 400.7's zone-change identity: bumped on *every* zone change,
    // regardless of which zones, so `engine::PlayPermission::
    // zone_change_generation` can tell "still sitting where it was granted"
    // apart from "moved since, for any reason" without needing a
    // zone-specific special case.
    obj.zone_change_count += 1;
    if to_zone == Zone::Battlefield {
        obj.tapped = false;
        obj.summoning_sick = true;
        obj.damage = 0;
        obj.counters = Default::default();
        obj.attachments.clear();
    }
}

/// 111.8/704.5d: "If a token is in a zone other than the battlefield, it
/// ceases to exist. This is a state-based action." Removes `id` from
/// whichever zone list it's currently tracked in (its owner's hand/library/
/// graveyard, `state.exile`/`command`, or the stack) without adding it
/// anywhere -- unlike every other zone transition, a token leaving the
/// battlefield doesn't go *to* another real zone, it just stops being
/// tracked. Called only by `trigger::sba_fixed_point`, and only for objects
/// `CardDef::is_token` marks as a token -- see that field's doc.
///
/// Returns whether `id` was actually still present (an already-ceased token
/// is a legal, idempotent no-op call) -- `sba_fixed_point`'s fixed-point
/// loop needs this to know whether the sweep made progress; unconditionally
/// reporting "changed" here would loop forever re-"removing" the same
/// already-gone token every pass.
///
/// Deliberately does *not* touch `GameObject::zone` (left as whatever
/// non-battlefield zone the token most recently moved to, e.g.
/// `Zone::Graveyard` for a sacrificed Blood Token): every other read of an
/// object's zone reaches it by first scanning a zone's own list (`ps.hand`,
/// `ps.graveyard`, `state.exile`, ...), which this function already empties
/// the token out of, so a stale `.zone` on an unreachable `ObjectId` the
/// arena still holds (ids are never freed -- see `ids.rs`'s module doc) is
/// inert, not a live correctness gap.
pub fn cease_to_exist(state: &mut GameState, id: ObjectId) -> bool {
    let owner = state.objects.get(id).owner;
    let zone = state.objects.get(id).zone;
    remove_from_zone(state, owner, id, zone)
}

/// Returns whether `id` was actually present in `zone`'s list before being
/// removed -- see `cease_to_exist`'s doc for why that matters to callers.
fn remove_from_zone(state: &mut GameState, owner: PlayerId, id: ObjectId, zone: Zone) -> bool {
    fn drop_from(v: &mut Vec<ObjectId>, id: ObjectId) -> bool {
        let before = v.len();
        v.retain(|&x| x != id);
        before != v.len()
    }
    match zone {
        Zone::Library => drop_from(&mut state.players[owner.index()].library, id),
        Zone::Hand => drop_from(&mut state.players[owner.index()].hand, id),
        Zone::Battlefield => drop_from(&mut state.players[owner.index()].battlefield, id),
        Zone::Graveyard => drop_from(&mut state.players[owner.index()].graveyard, id),
        Zone::Exile => drop_from(&mut state.exile, id),
        Zone::Command => drop_from(&mut state.command, id),
        Zone::Stack => {
            let before = state.stack.len();
            state.stack.retain(|item| item.source != id);
            before != state.stack.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;

    fn fresh_state() -> GameState {
        GameState::new_from_libraries(&[1, 2, 3], &[4, 5, 6], |c| format!("card-{c}"), 1)
    }

    #[test]
    fn commit_damage_to_player_reduces_life() {
        let mut state = fresh_state();
        propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 3));
        assert_eq!(state.players[1].life, 17);
        assert_eq!(
            state.engine.event_log,
            vec![CommittedEvent::Damage { source: ObjectId(0), target: Target::Player(PlayerId::P1), amount: 3 }]
        );
    }

    #[test]
    fn zone_change_moves_between_owner_zones_and_updates_object() {
        let mut state = fresh_state();
        let card = state.draw_card(PlayerId::P0).unwrap();
        propose_and_commit(&mut state, ProposedEvent::zone_change(card, Zone::Battlefield));
        assert_eq!(state.objects.get(card).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&card));
        assert!(!state.players[0].hand.contains(&card));
    }

    #[test]
    fn draw_from_empty_library_sets_drew_from_empty() {
        let mut state = GameState::new_from_libraries(&[], &[1], |c| format!("card-{c}"), 1);
        propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
        assert!(state.players[0].drew_from_empty);
    }

    /// End-to-end proof of the replacement pipeline shape required by the
    /// design: a prevention shield partially absorbs one hit, then expires
    /// and lets a subsequent hit through in full.
    #[test]
    fn prevention_shield_absorbs_then_expires() {
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 1,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage { target: Target::Player(PlayerId::P1), remaining: 2 },
        });

        // First hit: 5 damage, shield absorbs 2 -> 3 gets through.
        propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 5));
        assert_eq!(state.players[1].life, 17);
        assert!(state.engine.active_replacements.is_empty(), "shield should be fully consumed");

        // Second hit: shield is gone, full damage applies.
        propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 4));
        assert_eq!(state.players[1].life, 13);
    }

    #[test]
    fn prevention_shield_can_fully_prevent_small_hits() {
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 7,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage { target: Target::Player(PlayerId::P1), remaining: 10 },
        });
        propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 3));
        assert_eq!(state.players[1].life, 20, "fully prevented, no event should mutate life");
        assert_eq!(state.engine.active_replacements[0].kind, ReplacementEffectKind::PreventNextDamage {
            target: Target::Player(PlayerId::P1),
            remaining: 7,
        });
    }

    #[test]
    fn a_replacement_never_touches_the_same_proposal_twice() {
        // A replacement that always "applies" but rewrites to something it
        // would also match would loop forever without touched_by tracking.
        // PreventNextDamage never re-matches its own rewritten (smaller)
        // event because it fully consumes itself in one hit; this test
        // just pins that a single shield only ever fires once per event.
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 3,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage { target: Target::Player(PlayerId::P1), remaining: 100 },
        });
        propose_and_commit(&mut state, ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 5));
        assert_eq!(state.players[1].life, 20);
        assert_eq!(state.engine.active_replacements[0].kind, ReplacementEffectKind::PreventNextDamage {
            target: Target::Player(PlayerId::P1),
            remaining: 95,
        });
    }
}
