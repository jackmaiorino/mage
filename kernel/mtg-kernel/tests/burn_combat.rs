//! Combat acceptance test: a scripted Mono-Red Burn "mirror" (P0 plays
//! Guttersnipe and Lightning Bolt; P1 plays Masked Meower as a one-shot
//! chump blocker) driven entirely through the public
//! `engine::advance_until_decision`/`engine::step` state machine,
//! extending `burn_goldfish.rs`'s acceptance shape into full combat.
//!
//! Script: both players play lands on curve. P0 casts Guttersnipe turn 3
//! (no attack that turn: summoning sick). P1 casts Masked Meower (haste)
//! turn 1 but its controller-side policy never attacks with it -- it exists
//! purely as a one-time chump blocker. Starting turn 4, P0 attacks with
//! Guttersnipe every turn; P1 blocks with Masked Meower exactly once (the
//! first opportunity), which dies to combat damage via SBA (2 damage vs 1
//! toughness) while Guttersnipe survives (1 damage vs 2 toughness). Every
//! subsequent attack goes through unblocked (P1 has no more creatures).
//! Starting turn 4, P0 also casts one Lightning Bolt per turn at P1, which
//! fires Guttersnipe's "whenever you cast an instant or sorcery spell,
//! Guttersnipe deals 2 damage to each opponent" trigger. The game ends when
//! P1's life reaches <=0 -- a mix of unblocked combat damage and burn.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, CommittedEvent, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};

fn debug_name(card_id: u16) -> String {
    CARD_DEFS[card_id as usize].name.to_string()
}

fn build_library(names: &[&str]) -> Vec<u16> {
    names
        .iter()
        .map(|n| card_id_by_name(n).unwrap_or_else(|| panic!("card {n:?} not found in CARD_DEFS")))
        .collect()
}

fn deal_opening_hands(state: &mut GameState, n: usize) {
    for _ in 0..n {
        event::propose_and_commit(state, ProposedEvent::draw(PlayerId::P0));
        event::propose_and_commit(state, ProposedEvent::draw(PlayerId::P1));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    CastOrPass(PlayerId),
    ChooseTargets(PlayerId),
    ChooseCastMode(PlayerId),
    Discard(PlayerId),
    DeclareAttackers(PlayerId),
    DeclareBlockers(PlayerId),
    OrderTriggers(PlayerId),
    GameOver,
}

fn kind_of(d: &Decision) -> Kind {
    match d {
        Decision::CastSpellOrPass { player, .. } => Kind::CastOrPass(*player),
        Decision::ChooseTargets { player, .. } => Kind::ChooseTargets(*player),
        Decision::ChooseCastMode { player, .. } => Kind::ChooseCastMode(*player),
        Decision::Discard { player, .. } => Kind::Discard(*player),
        Decision::DeclareAttackers { player, .. } => Kind::DeclareAttackers(*player),
        Decision::DeclareBlockers { player, .. } => Kind::DeclareBlockers(*player),
        Decision::OrderTriggers { player, .. } => Kind::OrderTriggers(*player),
        Decision::GameOver { .. } => Kind::GameOver,
        // This script's card pool (Mountain, Guttersnipe, Lightning Bolt,
        // Masked Meower) has no Plot/Madness/modal card, so none of these
        // are ever reachable here.
        Decision::ChooseSpellMode { .. }
        | Decision::ChooseOptionalCost { .. }
        | Decision::ChooseSpellCopyPayment { .. }
        | Decision::ChooseSpellCopyRetarget { .. }
        | Decision::ChooseMadnessCast { .. } => {
            unreachable!("no card in this script is Plotted, Madness, or modal")
        }
        // Neither Fireblast nor Lava Dart is in this script's card pool, so
        // no cast ever needs a sacrifice-cost-target pick.
        Decision::ChooseCostTargets { .. } => {
            unreachable!("no card in this script has a SacrificeLands cost")
        }
        // Goblin Bushwhacker (the only Kicker card) isn't in this script's
        // card pool.
        Decision::ChooseKicker { .. } => unreachable!("no card in this script has Kicker"),
        // Chain Lightning (the only card that can produce this) isn't in
        // this script's card pool.
        Decision::Halted { .. } => unreachable!("no card in this script can halt the walk"),
    }
}

fn setup() -> GameState {
    // P0: 3 Mountains (for Guttersnipe's {2}{R}) + Guttersnipe + 3 Lightning
    // Bolts in the opening hand, then padding Mountains so the draw step
    // never empties the library before the game ends.
    let mut p0 = vec!["Mountain", "Mountain", "Mountain", "Guttersnipe"];
    p0.extend(std::iter::repeat_n("Lightning Bolt", 3));
    p0.extend(std::iter::repeat_n("Mountain", 30));

    // P1: 1 Mountain + Masked Meower (haste, {R}) in the opening hand, then
    // padding Mountains. Never draws or casts anything else.
    let mut p1 = vec!["Mountain", "Masked Meower"];
    p1.extend(std::iter::repeat_n("Mountain", 40));

    let p0_lib = build_library(&p0);
    let p1_lib = build_library(&p1);
    let mut state = GameState::new_from_libraries(&p0_lib, &p1_lib, debug_name, 0xC0FFEE);
    deal_opening_hands(&mut state, 7);
    state
}

fn card_in_hand(state: &GameState, player: PlayerId, def_id: u16) -> Option<ObjectId> {
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == def_id)
}

/// Drives the whole scripted game to completion, logging every decision
/// kind observed and applying a small reactive policy:
/// - always play a land when one is available;
/// - P0 casts Guttersnipe as soon as it's affordable, then casts at most
///   one Lightning Bolt (at P1) per turn;
/// - P0 attacks with Guttersnipe whenever it's an eligible attacker;
/// - P1 casts Masked Meower once, never attacks with it, and blocks
///   Guttersnipe with it the first time that's a legal block;
/// - every other decision (discard, trigger ordering) is answered
///   minimally/deterministically; anything the script's card pool cannot
///   produce is `unreachable!()`.
///
/// Returns the decision-kind log plus the count of `DeclareAttackers(P0)`
/// decisions that actually declared a non-empty attacking set. Since
/// increment 6's step-lag fix (508.1: Declare Attackers is a turn-based
/// action that always happens, never skipped just because nothing's
/// eligible -- see `engine::advance_step`'s doc), P0 gets a
/// `DeclareAttackers` decision *every* one of its own turns, including the
/// turns before Guttersnipe exists or while it's summoning sick -- so raw
/// `Kind::DeclareAttackers(P0)` occurrences in the log are no longer the
/// same thing as "P0 actually attacked", and callers that need the latter
/// (the `DeclareBlockers(P1)` cross-check, the Guttersnipe-trigger damage
/// arithmetic) need this separate count instead.
fn run_combat_game(state: &mut GameState) -> (Vec<Kind>, u32) {
    let guttersnipe_def = card_id_by_name("Guttersnipe").unwrap();
    let bolt_def = card_id_by_name("Lightning Bolt").unwrap();
    let meower_def = card_id_by_name("Masked Meower").unwrap();

    let mut log = Vec::new();
    let mut real_p0_attacks = 0u32;
    let mut last_bolt_turn = 0u32;
    let mut iterations = 0u32;

    loop {
        iterations += 1;
        assert!(
            iterations < 5000,
            "scripted game did not terminate; policy or engine logic is likely wrong"
        );

        let decision = engine::advance_until_decision(state);
        log.push(kind_of(&decision));

        match decision {
            Decision::GameOver { winner } => {
                assert_eq!(
                    winner,
                    Some(PlayerId::P0),
                    "P0 should win via a mix of combat and burn"
                );
                break;
            }
            Decision::OrderTriggers { .. } => {
                unreachable!(
                    "only Guttersnipe ever triggers in this script, and only once per event batch"
                )
            }
            Decision::ChooseCastMode { .. } => {
                unreachable!(
                    "no card in this script has an alt_cost (Fireblast isn't in either library)"
                )
            }
            Decision::ChooseCostTargets { .. } => {
                unreachable!("no card in this script has a SacrificeLands cost (Fireblast/Lava Dart aren't in either library)")
            }
            Decision::Discard {
                player,
                count,
                choices,
            } => {
                // Defensive, not scripted: discard the lowest-id `count`
                // cards. Never actually exercised by this script's hand
                // sizes (see the increment-3 report), but keeps the test
                // robust instead of asserting unreachable on a hand-size
                // computation this test doesn't want to hand-verify.
                let mut sorted = choices.clone();
                sorted.sort_unstable();
                let chosen: Vec<ObjectId> = sorted.into_iter().take(count as usize).collect();
                assert_eq!(
                    chosen.len() as u32,
                    count,
                    "player {player:?} had fewer legal discards than required"
                );
                engine::step(state, Action::Discard(chosen)).unwrap();
            }
            Decision::ChooseTargets {
                player,
                remaining,
                legal_targets,
                ..
            } => {
                assert_eq!(
                    player,
                    PlayerId::P0,
                    "only P0 ever casts a targeted spell (Lightning Bolt) in this script"
                );
                assert_eq!(remaining, 1);
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                engine::step(state, Action::ChooseTarget(target)).unwrap();
            }
            Decision::DeclareAttackers { player, eligible } => {
                let attackers: Vec<ObjectId> = if player == PlayerId::P0 {
                    // Attack with Guttersnipe whenever it's eligible.
                    eligible
                        .iter()
                        .copied()
                        .filter(|&id| state.objects.get(id).card_def == guttersnipe_def)
                        .collect()
                } else {
                    // P1's Masked Meower has haste and so is technically
                    // eligible from the turn it's cast, but the script
                    // never attacks with it -- it's a dedicated blocker.
                    Vec::new()
                };
                if player == PlayerId::P0 && !attackers.is_empty() {
                    real_p0_attacks += 1;
                }
                engine::step(state, Action::DeclareAttackers(attackers)).unwrap();
            }
            Decision::DeclareBlockers {
                player,
                attackers,
                legal_blockers,
            } => {
                assert_eq!(
                    player,
                    PlayerId::P1,
                    "P0 is never the defending player in this script (P1 never attacks)"
                );
                let blocks: Vec<(ObjectId, ObjectId)> = attackers
                    .iter()
                    .filter_map(|&attacker| {
                        let (_, blockers) = legal_blockers.iter().find(|(a, _)| *a == attacker)?;
                        blockers
                            .iter()
                            .copied()
                            .find(|&b| state.objects.get(b).card_def == meower_def)
                            .map(|b| (b, attacker))
                    })
                    .collect();
                engine::step(state, Action::DeclareBlockers(blocks)).unwrap();
            }
            Decision::CastSpellOrPass {
                player,
                castable_spells,
                land_drops,
                ..
            } => {
                if !land_drops.is_empty() {
                    engine::step(state, Action::PlayLand(land_drops[0])).unwrap();
                } else if player == PlayerId::P0
                    && card_in_hand(state, PlayerId::P0, guttersnipe_def)
                        .is_some_and(|g| castable_spells.contains(&g))
                {
                    let guttersnipe = card_in_hand(state, PlayerId::P0, guttersnipe_def).unwrap();
                    engine::step(state, Action::CastSpell(guttersnipe)).unwrap();
                } else if player == PlayerId::P0
                    && state.active_player == PlayerId::P0
                    && matches!(
                        state.step,
                        mtg_kernel::state::Step::Main1 | mtg_kernel::state::Step::Main2
                    )
                    && state.turn != last_bolt_turn
                    && state.players[0]
                        .battlefield
                        .iter()
                        .any(|&id| state.objects.get(id).card_def == guttersnipe_def)
                    && card_in_hand(state, PlayerId::P0, bolt_def)
                        .is_some_and(|b| castable_spells.contains(&b))
                {
                    // Bolt is an instant and could legally be cast any time
                    // P0 has priority (including during P1's turn, or P0's
                    // own upkeep, or before Guttersnipe has resolved) --
                    // restricted here to P0's own main phases *after*
                    // Guttersnipe is on the battlefield purely so this
                    // script's narrative (every Bolt fires Guttersnipe's
                    // cast-trigger) is deterministic, not because the
                    // engine requires it.
                    let bolt = card_in_hand(state, PlayerId::P0, bolt_def).unwrap();
                    last_bolt_turn = state.turn;
                    engine::step(state, Action::CastSpell(bolt)).unwrap();
                } else if player == PlayerId::P1
                    && card_in_hand(state, PlayerId::P1, meower_def)
                        .is_some_and(|m| castable_spells.contains(&m))
                {
                    let meower = card_in_hand(state, PlayerId::P1, meower_def).unwrap();
                    engine::step(state, Action::CastSpell(meower)).unwrap();
                } else {
                    engine::step(state, Action::Pass).unwrap();
                }
            }
            Decision::ChooseSpellMode { .. }
            | Decision::ChooseOptionalCost { .. }
            | Decision::ChooseSpellCopyPayment { .. }
            | Decision::ChooseSpellCopyRetarget { .. }
            | Decision::ChooseMadnessCast { .. } => {
                unreachable!("no card in this script is Plotted, Madness, or modal")
            }
            Decision::ChooseKicker { .. } => unreachable!("no card in this script has Kicker"),
            Decision::Halted { .. } => unreachable!("no card in this script can halt the walk"),
        }
    }

    (log, real_p0_attacks)
}

#[test]
fn combat_and_burn_together_end_the_game() {
    let mut state = setup();
    let guttersnipe_def = card_id_by_name("Guttersnipe").unwrap();
    let meower_def = card_id_by_name("Masked Meower").unwrap();

    let (log, real_p0_attacks) = run_combat_game(&mut state);

    // ---- terminal state -------------------------------------------------
    assert!(state.players[1].has_lost);
    assert!(!state.players[0].has_lost);
    assert!(state.players[1].life <= 0);

    // ---- Guttersnipe survived combat; Masked Meower died to it ----------
    let guttersnipe_id = state.players[0]
        .battlefield
        .iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == guttersnipe_def);
    assert!(
        guttersnipe_id.is_some(),
        "Guttersnipe should still be on P0's battlefield"
    );
    let meower_id = state.players[1]
        .graveyard
        .iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == meower_def);
    assert!(
        meower_id.is_some(),
        "Masked Meower should have died blocking and be in P1's graveyard"
    );
    assert!(!state.players[1]
        .battlefield
        .iter()
        .any(|&id| state.objects.get(id).card_def == meower_def));

    // ---- decision-kind sequence: real DeclareAttackers/DeclareBlockers
    // windows happened on both sides, no shortcuts taken. Declare Attackers
    // is a turn-based action that's *never* skipped (508.1 -- see
    // `run_combat_game`'s doc): P0 gets one every one of its own turns, even
    // before Guttersnipe exists, so raw `DeclareAttackers(P0)` count is
    // strictly greater than `real_p0_attacks` (the turns it actually had
    // something to attack with). P1's Masked Meower has haste, so P1 gets a
    // DeclareAttackers decision every one of its own turns too (turns 1-3,
    // whether or not it declares -- it never does). DeclareBlockers, by
    // contrast, *is* skipped when the just-completed DeclareAttackers
    // declared nothing (509.4-ish), so P1 only gets a DeclareBlockers
    // decision on the subset of P0's turns that were `real_p0_attacks`.
    let count = |k: Kind| log.iter().filter(|&&d| d == k).count();
    assert_eq!(count(Kind::GameOver), 1);
    assert_eq!(count(Kind::OrderTriggers(PlayerId::P0)), 0);
    assert_eq!(count(Kind::ChooseCastMode(PlayerId::P0)), 0);
    assert!(real_p0_attacks >= 1, "P0 must have attacked at least once");
    assert!(
        count(Kind::DeclareAttackers(PlayerId::P0)) > real_p0_attacks as usize,
        "P0 should also get DeclareAttackers decisions on turns before Guttersnipe exists (508.1: the step is never skipped, even with nothing eligible)"
    );
    assert!(count(Kind::DeclareAttackers(PlayerId::P1)) >= 1, "P1's hasty Masked Meower should have produced attack decisions even though it never attacks");
    assert_eq!(
        count(Kind::DeclareBlockers(PlayerId::P1)),
        real_p0_attacks as usize,
        "P1 should be asked to declare blockers exactly once per turn P0 actually declared a non-empty attack (509.4-ish: skipped when zero attackers were declared)"
    );
    assert_eq!(
        count(Kind::DeclareBlockers(PlayerId::P0)),
        0,
        "P0 is never the defending player (P1 never attacks)"
    );
    assert!(
        count(Kind::ChooseTargets(PlayerId::P0)) >= 1,
        "at least one Lightning Bolt should have been cast"
    );

    // ---- event log: exact combat-damage exchange between Guttersnipe and
    // Masked Meower (the one turn they fought).
    let guttersnipe_id = guttersnipe_id.unwrap();
    let meower_id = meower_id.unwrap();
    let object_damage: Vec<_> = state
        .engine
        .event_history
        .iter()
        .filter(|e| {
            matches!(
                e,
                CommittedEvent::Damage {
                    target: Target::Object(_),
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        object_damage,
        vec![
            &CommittedEvent::Damage {
                source: guttersnipe_id,
                target: Target::Object(meower_id),
                amount: 2
            },
            &CommittedEvent::Damage {
                source: meower_id,
                target: Target::Object(guttersnipe_id),
                amount: 1
            },
        ],
        "exactly one combat exchange should have happened, in the same simultaneous-damage batch"
    );

    // ---- event log: every Lightning Bolt actually resolved for exactly 3
    // damage to P1 (unambiguous -- nothing else in this script deals 3).
    let bolt_def = card_id_by_name("Lightning Bolt").unwrap();
    let bolts_cast = state
        .engine
        .event_history
        .iter()
        .filter(|e| matches!(e, CommittedEvent::SpellCast { spell, controller: PlayerId::P0 } if state.objects.get(*spell).card_def == bolt_def))
        .count();
    let bolt_damage = state
        .engine
        .event_history
        .iter()
        .filter(|e| {
            matches!(
                e,
                CommittedEvent::Damage {
                    target: Target::Player(PlayerId::P1),
                    amount: 3,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        bolts_cast, bolt_damage,
        "one SpellCast per Lightning Bolt, one 3-damage event per resolved Lightning Bolt"
    );
    assert!(bolts_cast >= 1);

    // ---- Guttersnipe's cast-trigger + unblocked-combat damage to P1 both
    // deal exactly 2 -- indistinguishable by (source, amount, target)
    // alone (both are `source: guttersnipe_id, amount: 2, target:
    // Player(P1)`), so cross-check by total-damage arithmetic instead:
    // every point of P1's life loss is accounted for by exactly these
    // three sources (unblocked combat, Guttersnipe triggers, Bolts).
    let damage_to_p1: i32 = state
        .engine
        .event_history
        .iter()
        .filter_map(|e| match e {
            CommittedEvent::Damage {
                target: Target::Player(PlayerId::P1),
                amount,
                ..
            } => Some(*amount),
            _ => None,
        })
        .sum();
    assert_eq!(
        damage_to_p1,
        20 - state.players[1].life,
        "every point of P1's life loss should be accounted for by a logged Damage event"
    );

    let two_damage_events_to_p1 = state
        .engine
        .event_history
        .iter()
        .filter(|e| matches!(e, CommittedEvent::Damage { target: Target::Player(PlayerId::P1), amount: 2, source, .. } if *source == guttersnipe_id))
        .count();
    // One 2-damage hit to P1 per Guttersnipe cast-trigger firing, plus one
    // per unblocked combat attack (every *real* P0 attack after the single
    // blocked one -- see `real_p0_attacks`'s doc, not raw
    // `DeclareAttackers(P0)` count, which now also includes the
    // nothing-eligible turns before Guttersnipe exists).
    let unblocked_attacks = real_p0_attacks as usize - 1;
    assert_eq!(two_damage_events_to_p1, bolts_cast + unblocked_attacks, "Guttersnipe's trigger should have fired exactly once per instant/sorcery P0 cast while it was alive");

    // ---- no shortcuts: P1 got a real priority window somewhere between
    // every ChooseTargets decision (mirrors burn_goldfish's equivalent
    // check), proving instants/sorceries weren't auto-resolved.
    let choose_target_positions: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d, Kind::ChooseTargets(_)))
        .map(|(i, _)| i)
        .collect();
    let mut segment_start = 0;
    for &pos in &choose_target_positions {
        let segment = &log[segment_start..pos];
        assert!(
            segment
                .iter()
                .any(|d| matches!(d, Kind::CastOrPass(PlayerId::P1))),
            "segment [{segment_start}..{pos}) never offered P1 a decision"
        );
        segment_start = pos + 1;
    }
}

/// Zone-transition sanity check on the same scripted game: Masked Meower's
/// journey is Library -> Hand -> Stack -> Battlefield -> Graveyard (the
/// Hand -> Stack leg is an engine action, not a `MoveObject` effect leaf --
/// see `engine::move_to_stack`'s doc -- so it isn't in this event log; the
/// object's zone really is `Stack` at the moment its own `MoveObject`
/// effect fires on resolution). It stays at the same stable `ObjectId`
/// throughout (no card is ever "replaced" by a zone change).
#[test]
fn masked_meower_zone_history_is_stack_battlefield_graveyard() {
    let mut state = setup();
    run_combat_game(&mut state);

    let meower_def = card_id_by_name("Masked Meower").unwrap();
    let meower_id = state.players[1]
        .graveyard
        .iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == meower_def)
        .expect("Masked Meower should be dead");

    let history: Vec<_> = state
        .engine
        .event_history
        .iter()
        .filter_map(|e| match e {
            CommittedEvent::ZoneChange { object, from, to } if *object == meower_id => {
                Some((*from, *to))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        history,
        vec![
            (Zone::Stack, Zone::Battlefield),
            (Zone::Battlefield, Zone::Graveyard)
        ]
    );
    assert_eq!(state.objects.get(meower_id).zone, Zone::Graveyard);
}
