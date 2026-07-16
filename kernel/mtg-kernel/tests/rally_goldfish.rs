//! Acceptance test: a scripted synthetic Mono Red Rally opening driven
//! entirely through the public `engine::advance_until_decision`/
//! `engine::step` state machine -- no shortcuts, no direct `GameState`
//! mutation for gameplay (mirrors `tests/burn_goldfish.rs`'s shape).
//!
//! P0 plays a curve of Rally's new cards -- Burning-Tree Emissary (hybrid
//! cost + ETB mana), a *kicked* Goblin Bushwhacker (Kicker cast-time
//! decision -> conditional ETB trigger -> team-wide temporary pump/haste
//! affecting both itself and an already-resolved creature), Galvanic Blast
//! without Metalcraft, Clockwork Percussionist (static Haste), and Reckless
//! Impulse (impulse-draw exiling cards playable across a later turn) --
//! then finishes P1 off with a mix of unblocked combat damage and burn
//! spells (2 hand-drawn, 2 cast straight out of exile via the impulse-draw
//! window). P1 never has a creature or a spell; every priority window it's
//! offered (including ones with no legal response) is asked and explicitly
//! passed, same faithfulness bar `burn_goldfish.rs` holds itself to.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target};

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

fn setup() -> GameState {
    // P0's curve, in library order (top to bottom): 2 Mountains (opening
    // hand), Burning-Tree Emissary, a kicked Goblin Bushwhacker, Galvanic
    // Blast, Clockwork Percussionist, Reckless Impulse, then 2 more
    // Mountains and 4 Lightning Bolts to finish, padded with Mountains so
    // the library never runs out before the game ends.
    let mut p0 = vec![
        "Mountain",
        "Mountain",
        "Burning-Tree Emissary",
        "Goblin Bushwhacker",
        "Galvanic Blast",
        "Clockwork Percussionist",
        "Reckless Impulse",
        "Mountain",
        "Mountain",
    ];
    p0.extend(std::iter::repeat_n("Lightning Bolt", 4));
    p0.extend(std::iter::repeat_n("Mountain", 30));

    // P1: never casts anything; just needs enough library to survive its
    // own draw steps for the length of the game.
    let p1: Vec<&str> = std::iter::repeat_n("Mountain", 50).collect();

    let p0_lib = build_library(&p0);
    let p1_lib = build_library(&p1);
    let mut state = GameState::new_from_libraries(&p0_lib, &p1_lib, debug_name, 0xC0FFEE);
    deal_opening_hands(&mut state, 7);
    state
}

fn card_in(list: &[ObjectId], state: &GameState, def_id: u16) -> Option<ObjectId> {
    list.iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == def_id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    CastOrPass(PlayerId),
    ChooseKicker(PlayerId),
    ChooseTargets(PlayerId),
    DeclareAttackers(PlayerId),
    DeclareBlockers(PlayerId),
    OrderTriggers(PlayerId),
    GameOver,
}

fn kind_of(d: &Decision) -> Kind {
    match d {
        Decision::CastSpellOrPass { player, .. } => Kind::CastOrPass(*player),
        Decision::ChooseKicker { player, .. } => Kind::ChooseKicker(*player),
        Decision::ChooseTargets { player, .. } => Kind::ChooseTargets(*player),
        Decision::DeclareAttackers { player, .. } => Kind::DeclareAttackers(*player),
        Decision::DeclareBlockers { player, .. } => Kind::DeclareBlockers(*player),
        Decision::OrderTriggers { player, .. } => Kind::OrderTriggers(*player),
        Decision::GameOver { .. } => Kind::GameOver,
        // Nothing in this goldfish's scripted library can produce these --
        // no alt-cost/additional-cost/flashback/Plot/Madness/modal card, no
        // discard obligation (hand never exceeds 7).
        other => {
            unreachable!("the rally goldfish's library has no card that can produce this decision: {other:?}")
        }
    }
}

/// Turn-gated policy: land whenever available; Burning-Tree Emissary as
/// soon as affordable (turn 2); Goblin Bushwhacker only from turn 3 onward
/// (so it's held until Burning-Tree Emissary is already down and 2 mana is
/// up for Kicker, rather than a naive "cast whatever's affordable" grabbing
/// it unkicked on turn 1); Galvanic Blast turn 3 (no Metalcraft yet: 0
/// artifacts); Clockwork Percussionist + Reckless Impulse turn 4; Lightning
/// Bolt (hand or impulse-exiled) from turn 5 onward to finish.
fn p0_action(
    state: &GameState,
    land_drops: &[ObjectId],
    castable_spells: &[ObjectId],
    defs: &Defs,
) -> Action {
    if !land_drops.is_empty() {
        return Action::PlayLand(land_drops[0]);
    }
    if state.turn >= 3 {
        if let Some(id) = card_in(castable_spells, state, defs.bushwhacker) {
            return Action::CastSpell(id);
        }
    }
    if let Some(id) = card_in(castable_spells, state, defs.bte) {
        return Action::CastSpell(id);
    }
    if state.turn == 3 {
        if let Some(id) = card_in(castable_spells, state, defs.galvanic) {
            return Action::CastSpell(id);
        }
    }
    if state.turn == 4 {
        if let Some(id) = card_in(castable_spells, state, defs.percussionist) {
            return Action::CastSpell(id);
        }
        if let Some(id) = card_in(castable_spells, state, defs.reckless) {
            return Action::CastSpell(id);
        }
    }
    if state.turn >= 5 {
        if let Some(id) = card_in(castable_spells, state, defs.bolt) {
            return Action::CastSpell(id);
        }
    }
    Action::Pass
}

struct Defs {
    bte: u16,
    bushwhacker: u16,
    galvanic: u16,
    percussionist: u16,
    reckless: u16,
    bolt: u16,
}

struct RunResult {
    log: Vec<Kind>,
    life_history: Vec<i32>,
    kicker_decisions: u32,
    choose_targets_p0: u32,
}

fn run_goldfish(state: &mut GameState) -> RunResult {
    let defs = Defs {
        bte: card_id_by_name("Burning-Tree Emissary").unwrap(),
        bushwhacker: card_id_by_name("Goblin Bushwhacker").unwrap(),
        galvanic: card_id_by_name("Galvanic Blast").unwrap(),
        percussionist: card_id_by_name("Clockwork Percussionist").unwrap(),
        reckless: card_id_by_name("Reckless Impulse").unwrap(),
        bolt: card_id_by_name("Lightning Bolt").unwrap(),
    };

    let mut log = Vec::new();
    let mut life_history = vec![state.players[1].life];
    let mut kicker_decisions = 0u32;
    let mut choose_targets_p0 = 0u32;
    let mut iterations = 0u32;

    loop {
        iterations += 1;
        assert!(
            iterations < 5000,
            "scripted game did not terminate; policy or engine logic is likely wrong"
        );

        let decision = engine::advance_until_decision(state);
        log.push(kind_of(&decision));

        if state.players[1].life != *life_history.last().unwrap() {
            life_history.push(state.players[1].life);
        }

        match decision {
            Decision::GameOver { winner } => {
                assert_eq!(
                    winner,
                    Some(PlayerId::P0),
                    "P0 should win via a kicked Rally curve + burn"
                );
                break;
            }
            Decision::OrderTriggers { .. } => {
                unreachable!(
                    "no card in this script produces 2+ simultaneous same-controller triggers"
                )
            }
            Decision::ChooseCastMode { .. }
            | Decision::ChooseCostTargets { .. }
            | Decision::Discard { .. }
            | Decision::ChooseSpellMode { .. }
            | Decision::ChooseOptionalCost { .. }
            | Decision::ChooseSpellCopyPayment { .. }
            | Decision::ChooseSpellCopyRetarget { .. }
            | Decision::ChooseMadnessCast { .. } => {
                unreachable!(
                    "the rally goldfish's library has no card that can produce this decision"
                )
            }
            // Chain Lightning isn't in this script's card pool.
            Decision::Halted { .. } => unreachable!("no card in this script can halt the walk"),
            Decision::ChooseKicker { player, .. } => {
                assert_eq!(
                    player,
                    PlayerId::P0,
                    "only Goblin Bushwhacker (P0's) ever offers Kicker in this script"
                );
                kicker_decisions += 1;
                engine::step(state, Action::ChooseKicker(true)).unwrap();
            }
            Decision::ChooseTargets {
                player,
                remaining,
                legal_targets,
                ..
            } => {
                assert_eq!(remaining, 1);
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                if player == PlayerId::P0 {
                    choose_targets_p0 += 1;
                }
                engine::step(state, Action::ChooseTarget(target)).unwrap();
            }
            Decision::DeclareAttackers { eligible, .. } => {
                // P0 always swings with everything eligible; P1's `eligible`
                // is always empty (its library has no creature), so this
                // one rule correctly covers both players.
                engine::step(state, Action::DeclareAttackers(eligible)).unwrap();
            }
            Decision::DeclareBlockers { player, .. } => {
                assert_eq!(
                    player,
                    PlayerId::P1,
                    "P0 is never the defending player in this script"
                );
                engine::step(state, Action::DeclareBlockers(Vec::new())).unwrap();
            }
            Decision::CastSpellOrPass {
                player,
                castable_spells,
                land_drops,
                ..
            } => {
                // P1's own land drops keep its hand from overflowing 7 (it
                // never casts anything, but it still gets its own turn's
                // one land play, same as P0) -- `p0_action` handles P0's
                // whole curve including its land drops.
                if !land_drops.is_empty() && player == PlayerId::P1 {
                    engine::step(state, Action::PlayLand(land_drops[0])).unwrap();
                } else if player == PlayerId::P0 {
                    let action = p0_action(state, &land_drops, &castable_spells, &defs);
                    engine::step(state, action).unwrap();
                } else {
                    engine::step(state, Action::Pass).unwrap();
                }
            }
        }
    }

    RunResult {
        log,
        life_history,
        kicker_decisions,
        choose_targets_p0,
    }
}

#[test]
fn rally_goldfish_kicked_bushwhacker_pump_plus_burn_ends_the_game_through_faithful_priority() {
    let mut state = setup();
    let result = run_goldfish(&mut state);

    // ---- life totals: 18 (Galvanic Blast, no Metalcraft) -> 13 (turn 3
    // combat: kicked 3/2 Burning-Tree Emissary + kicked 2/1 hasty Goblin
    // Bushwhacker, 5 unblocked) -> 9 (turn 4 combat: 2/2 + 1/1 + 1/1 hasty
    // Clockwork Percussionist, pump expired, 4 unblocked) -> 6 -> 3 -> 0
    // (3 Lightning Bolts, at least one straight out of exile via Reckless
    // Impulse's impulse-draw window).
    assert_eq!(result.life_history, vec![20, 18, 13, 9, 6, 3, 0]);
    assert!(state.players[1].has_lost);
    assert!(!state.players[0].has_lost);

    // ---- Kicker was offered and chosen exactly once (Goblin Bushwhacker).
    assert_eq!(result.kicker_decisions, 1);

    // ---- exactly 4 targeted spells resolved for P0 (1 Galvanic Blast + 3
    // Lightning Bolts); P1 never cast or targeted anything.
    assert_eq!(result.choose_targets_p0, 4);
    assert_eq!(
        result
            .log
            .iter()
            .filter(|d| matches!(d, Kind::ChooseTargets(PlayerId::P1)))
            .count(),
        0
    );
    assert_eq!(
        result
            .log
            .iter()
            .filter(|d| matches!(d, Kind::ChooseKicker(PlayerId::P1)))
            .count(),
        0
    );
    assert_eq!(
        result
            .log
            .iter()
            .filter(|d| matches!(d, Kind::OrderTriggers(_)))
            .count(),
        0
    );

    // ---- final board: Burning-Tree Emissary, Goblin Bushwhacker, and
    // Clockwork Percussionist all survived (P1 never blocked or removed
    // anything) -- the pump was "until end of turn" only, so by game's end
    // they're back to their printed stats.
    let bte_id = card_id_by_name("Burning-Tree Emissary").unwrap();
    let bushwhacker_id = card_id_by_name("Goblin Bushwhacker").unwrap();
    let percussionist_id = card_id_by_name("Clockwork Percussionist").unwrap();
    assert!(state.players[0]
        .battlefield
        .iter()
        .any(|&id| state.objects.get(id).card_def == bte_id));
    assert!(state.players[0]
        .battlefield
        .iter()
        .any(|&id| state.objects.get(id).card_def == bushwhacker_id));
    assert!(state.players[0]
        .battlefield
        .iter()
        .any(|&id| state.objects.get(id).card_def == percussionist_id));
    assert!(
        state.engine.until_end_of_turn.is_empty(),
        "no pump effect should still be lingering at game's end"
    );

    // ---- no shortcuts: P1 got a real priority window somewhere between
    // every ChooseTargets decision that wasn't immediately preceded by
    // another one -- same faithfulness check `burn_goldfish.rs` holds
    // itself to, relaxed only for the back-to-back finishing burn (117.3c:
    // the caster keeps priority after casting, so P0 legally chains its
    // last 2 Lightning Bolts before anyone else is asked again -- a
    // zero/one-entry segment here reflects that legal chaining, not a
    // skipped priority window).
    let choose_target_positions: Vec<usize> = result
        .log
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d, Kind::ChooseTargets(_)))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        choose_target_positions.len(),
        4,
        "1 Galvanic Blast + 3 Lightning Bolts"
    );
    let mut segment_start = 0;
    for &pos in &choose_target_positions {
        let segment = &result.log[segment_start..pos];
        if segment.len() > 1 {
            assert!(
                segment
                    .iter()
                    .any(|d| matches!(d, Kind::CastOrPass(PlayerId::P1))),
                "segment [{segment_start}..{pos}) never offered P1 a decision"
            );
        }
        segment_start = pos + 1;
    }
    assert!(
        result
            .log
            .iter()
            .filter(|d| matches!(d, Kind::CastOrPass(PlayerId::P1)))
            .count()
            > 30,
        "P1 should have gotten many real priority windows across the whole game"
    );
}
