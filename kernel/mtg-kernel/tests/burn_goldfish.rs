//! Acceptance test: a scripted synthetic Mono-Red Burn "goldfish" driven
//! entirely through the public `engine::advance_until_decision` /
//! `engine::step` state machine -- no shortcuts, no direct `GameState`
//! mutation for gameplay. P0 plays a Mountain on turn 1 and passes; P1
//! plays a Mountain on turn 1 and passes; starting turn 2, P0 plays a land
//! (while it has one) and casts one Lightning Bolt at P1 per turn until
//! P1's life reaches <=0 and state-based actions declare P0 the winner.
//!
//! This asserts the *faithful* priority-window sequence (every window is
//! logged and checked, including the ones offered to P1 that have no
//! legal response), not just the final outcome: see
//! `narrated_prefix_matches_turn_1_and_turn_2_exactly` for turn 1/2
//! decision-by-decision, and the aggregate counts/log-content assertions
//! in the main test for the rest of the game.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, CommittedEvent, ProposedEvent};
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

/// Draw `n` cards for each player via the real propose/commit draw event --
/// this is opening-hand dealing (draw 7), not gameplay. There is no
/// mulligan decision kind this increment (see the design brief), so
/// dealing the fixed opening hand is test setup, done directly rather than
/// through `engine::step`.
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
    OrderTriggers(PlayerId),
    // Neither player ever has a creature in this goldfish's library, but
    // 508.1 makes Declare Attackers a turn-based action that always
    // happens regardless -- see `engine::advance_step`'s doc -- so the
    // active player still gets asked (with an always-empty `eligible`)
    // every single one of its own combats.
    DeclareAttackers(PlayerId),
    GameOver,
}

fn kind_of(d: &Decision) -> Kind {
    match d {
        Decision::CastSpellOrPass { player, .. } => Kind::CastOrPass(*player),
        Decision::ChooseTargets { player, .. } => Kind::ChooseTargets(*player),
        Decision::OrderTriggers { player, .. } => Kind::OrderTriggers(*player),
        Decision::DeclareAttackers { player, .. } => Kind::DeclareAttackers(*player),
        Decision::GameOver { .. } => Kind::GameOver,
        // Nothing in this goldfish's scripted library (Mountain + Lightning
        // Bolt only) can produce these -- no alt-cost/additional-cost card,
        // no discard obligation (hand never exceeds 7), no creature ever
        // cast so no real block/attack declaration (DeclareBlockers is
        // never reached: it's skipped whenever Declare Attackers just
        // declared zero attackers, which is always true here). See
        // `tests/burn_combat.rs` for these.
        Decision::ChooseCastMode { .. }
        | Decision::ChooseKicker { .. }
        | Decision::ChooseCostTargets { .. }
        | Decision::Discard { .. }
        | Decision::DeclareBlockers { .. }
        | Decision::ChooseSpellMode { .. }
        | Decision::ChooseEffectOption { .. }
        | Decision::ChooseEffectTargets { .. }
        | Decision::ChooseOptionalCost { .. }
        | Decision::ChooseSpellCopyPayment { .. }
        | Decision::ChooseSpellCopyRetarget { .. }
        | Decision::ChooseMadnessCast { .. }
        | Decision::Halted { .. } => {
            unreachable!("the burn goldfish's library has no card that can produce this decision")
        }
    }
}

fn setup() -> GameState {
    // P0: opening hand = 3 Mountain + 4 Lightning Bolt, then a long run of
    // Lightning Bolt so every subsequent draw keeps a bolt in hand, padded
    // with Mountains so the library never runs out before the game ends.
    let mut p0 = vec!["Mountain", "Mountain", "Mountain"];
    p0.extend(std::iter::repeat_n("Lightning Bolt", 4));
    p0.extend(std::iter::repeat_n("Lightning Bolt", 10));
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

fn bolt_in_hand(state: &GameState, player: PlayerId, bolt_def: u16) -> Option<ObjectId> {
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .find(|&id| state.objects.get(id).card_def == bolt_def)
}

/// Drives the whole goldfish to completion, logging every decision kind and
/// every P1 life total observed right after it changes. Policy: play a
/// land whenever available; starting turn 2, cast exactly one Lightning
/// Bolt at P1 per P0 turn (tracked via `last_cast_turn` so a second
/// untapped Mountain on the same turn is deliberately *not* spent --
/// keeps the life sequence exactly 20,17,14,...); otherwise pass.
fn run_goldfish(state: &mut GameState) -> (Vec<Kind>, Vec<i32>) {
    let bolt_def = card_id_by_name("Lightning Bolt").unwrap();
    let mut log = Vec::new();
    let mut life_history = vec![state.players[1].life];
    let mut last_cast_turn = 0u32;

    loop {
        let decision = engine::advance_until_decision(state);
        log.push(kind_of(&decision));

        // `state` already reflects any resolution `advance_until_decision`
        // just ran (including the one that produces `GameOver`), so check
        // for a life change before branching on/consuming the decision.
        if state.players[1].life != *life_history.last().unwrap() {
            life_history.push(state.players[1].life);
        }

        match decision {
            Decision::GameOver { winner } => {
                assert_eq!(winner, Some(PlayerId::P0), "P0 should win via burn only");
                break;
            }
            Decision::OrderTriggers { .. } => {
                unreachable!("no card in this increment's pool has an implemented trigger")
            }
            Decision::ChooseCastMode { .. }
            | Decision::ChooseKicker { .. }
            | Decision::ChooseCostTargets { .. }
            | Decision::Discard { .. }
            | Decision::DeclareBlockers { .. }
            | Decision::ChooseSpellMode { .. }
            | Decision::ChooseEffectOption { .. }
            | Decision::ChooseEffectTargets { .. }
            | Decision::ChooseOptionalCost { .. }
            | Decision::ChooseSpellCopyPayment { .. }
            | Decision::ChooseSpellCopyRetarget { .. }
            | Decision::ChooseMadnessCast { .. }
            | Decision::Halted { .. } => {
                unreachable!(
                    "the burn goldfish's library has no card that can produce this decision"
                )
            }
            Decision::DeclareAttackers { eligible, .. } => {
                assert!(
                    eligible.is_empty(),
                    "neither player's library has a creature in it"
                );
                engine::step(state, Action::DeclareAttackers(Vec::new())).unwrap();
            }
            Decision::ChooseTargets {
                player,
                spell: _,
                remaining,
                legal_targets,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(remaining, 1);
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                engine::step(state, Action::ChooseTarget(target)).unwrap();
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
                    && state.turn >= 2
                    && state.turn != last_cast_turn
                    && bolt_in_hand(state, player, bolt_def).is_some()
                {
                    let bolt = bolt_in_hand(state, player, bolt_def).unwrap();
                    assert!(
                        castable_spells.contains(&bolt),
                        "bolt should be affordable off 1 untapped Mountain"
                    );
                    last_cast_turn = state.turn;
                    let stack_len_before = state.stack.len();
                    engine::step(state, Action::CastSpell(bolt)).unwrap();
                    // 601.2a: CastSpell immediately *announces* the cast,
                    // moving the spell onto the stack before targets are
                    // chosen or costs are paid.
                    assert_eq!(state.stack.len(), stack_len_before + 1);
                } else {
                    engine::step(state, Action::Pass).unwrap();
                }
            }
        }
    }

    (log, life_history)
}

/// Turn 1 (P0 plays Mountain + passes, both steps of priority; P1 gets a
/// real response window at every step even though it never has a legal
/// response) through the start of P1's turn 1 (P1's Upkeep/Draw, its own
/// Mountain + pass in Main1, Begin Combat, and its own empty Declare
/// Attackers decision, with P0's response window right after). Hand-derived
/// step by step against the turn structure in `engine.rs`'s `STEP_ORDER` +
/// APNAP priority rules and cross-checked against the real log.
///
/// 508.1: Declare Attackers is a turn-based action that always happens, so
/// every combat gets a `DeclareAttackers(<active player>)` decision (always
/// declaring nobody -- this goldfish's library has no creature in it)
/// followed by its own priority round, even though nothing was declared --
/// see `engine::advance_step`'s doc. Declare Blockers/Combat Damage *are*
/// still skipped afterward (zero attackers were actually declared).
const NARRATED_PREFIX: [Kind; 30] = {
    use Kind::{CastOrPass, DeclareAttackers};
    const P0: PlayerId = PlayerId::P0;
    const P1: PlayerId = PlayerId::P1;
    [
        CastOrPass(P0),
        CastOrPass(P1), // Upkeep
        CastOrPass(P0),
        CastOrPass(P1), // Draw (P0's first-turn draw skipped)
        CastOrPass(P0),
        CastOrPass(P0), // Main1: P0 plays Mountain, then passes
        CastOrPass(P1), // Main1: P1's response window, passes
        CastOrPass(P0),
        CastOrPass(P1),       // Begin Combat
        DeclareAttackers(P0), // Declare Attackers: P0 (active) declares nobody
        CastOrPass(P0),
        CastOrPass(P1), // Declare Attackers step's own priority round
        CastOrPass(P0),
        CastOrPass(P1), // End Combat (Declare Blockers/Combat Damage
        // skipped: zero attackers were declared)
        CastOrPass(P0),
        CastOrPass(P1), // Main2
        CastOrPass(P0),
        CastOrPass(P1), // End step
        // Cleanup + Untap grant no priority (no decisions), turn passes to P1.
        CastOrPass(P1),
        CastOrPass(P0), // P1's Upkeep (active player first)
        CastOrPass(P1),
        CastOrPass(P0), // P1's Draw (P1 does NOT skip its own first draw)
        CastOrPass(P1),
        CastOrPass(P1), // Main1: P1 plays Mountain, then passes
        CastOrPass(P0), // Main1: P0's response window, passes
        CastOrPass(P1),
        CastOrPass(P0),       // P1's Begin Combat
        DeclareAttackers(P1), // P1's Declare Attackers: declares nobody
        CastOrPass(P1),
        CastOrPass(P0), // P1's Declare Attackers step's priority round
    ]
};

#[test]
fn narrated_prefix_matches_turn_1_and_start_of_p1_turn_1_exactly() {
    let mut state = setup();
    let (log, _life_history) = run_goldfish(&mut state);
    assert_eq!(&log[..30], &NARRATED_PREFIX[..]);
}

#[test]
fn burn_goldfish_kills_p1_via_repeated_lightning_bolt_through_faithful_priority() {
    let mut state = setup();
    let (log, life_history) = run_goldfish(&mut state);

    // ---- life totals: exactly 7 casts of 3 damage, the first is the
    // narrated 20 -> 17.
    assert_eq!(life_history, vec![20, 17, 14, 11, 8, 5, 2, -1]);

    // ---- final has_lost: only P1, via burn (no library-out, no combat).
    assert!(state.players[1].has_lost);
    assert!(!state.players[0].has_lost);

    // ---- decision sequence, counted: no shortcuts were taken anywhere in
    // the game, not just the narrated prefix. 7 casts means exactly 7
    // ChooseTargets decisions (all P0) and exactly 1 terminal GameOver;
    // every other decision is a real CastSpellOrPass priority window,
    // split close to evenly between the two players (P1 gets slightly
    // fewer because P0 sometimes acts twice in a window: play a land,
    // *then* get asked again before passing) -- plus one
    // `DeclareAttackers(<active player>)` per combat, always declaring
    // nobody (508.1: never skipped -- see `engine::advance_step`'s doc and
    // `NARRATED_PREFIX`'s). The game ends mid-round (P0's 7th Bolt kills
    // P1 before P1 gets an 8th turn), so both players get exactly 7 of
    // their own combats.
    let count = |k: Kind| log.iter().filter(|&&d| d == k).count();
    assert_eq!(count(Kind::DeclareAttackers(PlayerId::P0)), 7);
    assert_eq!(count(Kind::DeclareAttackers(PlayerId::P1)), 7);
    assert_eq!(count(Kind::CastOrPass(PlayerId::P0)), 129);
    assert_eq!(count(Kind::CastOrPass(PlayerId::P1)), 126);
    assert_eq!(count(Kind::ChooseTargets(PlayerId::P0)), 7);
    assert_eq!(count(Kind::GameOver), 1);
    assert_eq!(log.len(), 277);
    assert_eq!(
        log.iter()
            .filter(|d| matches!(d, Kind::ChooseTargets(PlayerId::P1)))
            .count(),
        0
    );
    assert_eq!(
        log.iter()
            .filter(|d| matches!(d, Kind::OrderTriggers(_)))
            .count(),
        0
    );

    // ---- the specific claim in the design brief: P1 gets a real priority
    // window *before* every single resolution, not just the first one.
    // Split the log into the 7 segments between consecutive ChooseTargets
    // decisions (a cast finalizing) and confirm P1 was offered (and had
    // to explicitly pass) a CastSpellOrPass decision in every segment.
    let choose_target_positions: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d, Kind::ChooseTargets(_)))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(choose_target_positions.len(), 7);
    let mut segment_start = 0;
    for &pos in &choose_target_positions {
        let segment = &log[segment_start..pos];
        // Only relevant for segments 2..7 (the first has no prior cast to
        // resolve); still true for segment 1, which covers P1's whole
        // first-turn priority windows.
        assert!(
            segment
                .iter()
                .any(|d| matches!(d, Kind::CastOrPass(PlayerId::P1))),
            "segment [{segment_start}..{pos}) never offered P1 a decision"
        );
        segment_start = pos + 1;
    }

    // ---- event log contents (the permanent history; `event_log` itself
    // is drained to empty after every resolution by
    // `trigger::collect_and_process`, by design -- see `event_history`'s
    // doc comment in `engine::EngineState`).
    let damage_to_p1: Vec<_> = state
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
        .collect();
    assert_eq!(
        damage_to_p1.len(),
        7,
        "exactly 7 Lightning Bolts should have dealt 3 damage to P1"
    );

    let bolts_to_graveyard = state
        .engine
        .event_history
        .iter()
        .filter(|e| {
            matches!(
                e,
                CommittedEvent::ZoneChange {
                    to: mtg_kernel::state::Zone::Graveyard,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        bolts_to_graveyard, 7,
        "each resolved Lightning Bolt goes to the graveyard, nothing else dies"
    );

    let taps = state
        .engine
        .event_history
        .iter()
        .filter(|e| matches!(e, CommittedEvent::Tap { .. }))
        .count();
    assert_eq!(taps, 7, "each cast pays with exactly 1 Mountain");

    let p0_lands_played = state
        .engine
        .event_history
        .iter()
        .filter(|e| matches!(e, CommittedEvent::ZoneChange { to: mtg_kernel::state::Zone::Battlefield, object, .. } if state.objects.get(*object).owner == PlayerId::P0))
        .count();
    assert_eq!(
        p0_lands_played, 3,
        "P0's library only had 3 Mountains before the game ended"
    );

    // ---- final board: P0's battlefield has exactly the 3 Mountains it
    // played (Lightning Bolt never sticks around -- it's a spell, not a
    // permanent).
    assert_eq!(state.players[0].battlefield.len(), 3);
    assert_eq!(state.players[0].graveyard.len(), 7);
}
