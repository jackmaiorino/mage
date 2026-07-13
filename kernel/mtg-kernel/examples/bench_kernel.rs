//! PERFORMANCE-ONLY kernel benchmark suite (external review Sol #93 scope:
//! measure raw engine speed, do not claim correctness/training/search value
//! -- that's a separate, already-tested concern covered by the ~150 green
//! tests elsewhere in this crate). Every number this binary prints is a raw
//! wall-clock measurement of the mtg-kernel engine alone: no NN inference,
//! no Python, no training loop.
//!
//! Sections (see the design brief this was commissioned against):
//!   1. snapshot/restore cost at ~80/200/350/500-object states
//!   2. step throughput: (a) the deterministic Burn goldfish script,
//!      (b) full Burn-mirror games via a seeded random-legal-action policy,
//!      both driven through the raw `engine::advance_until_decision`/`step`
//!      API (every priority window counted, none suppressed)
//!   3. self-play throughput: 1/4/8/16 threads, each running seeded
//!      random-policy Burn-mirror games through `HarnessSurfaceV2` (the
//!      H-visible decision surface a training loop would actually consume)
//!   4. legal-actions enumeration cost on a captured mid-game cast window
//!   5. allocator profile (only with `--features count_allocs`)
//!
//! Run: cargo run --release --example bench_kernel
//! Alloc profile: cargo run --release --example bench_kernel --features count_allocs

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision, OptionalCostChoice};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{Counters, GameObject, GameState, SplitMix64, Target, Zone};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SurfaceAction, SurfaceDecision};

use std::collections::HashSet;
use std::time::{Duration, Instant};

// ------------------------------------------------------------- tuning knobs

const SNAPSHOT_WARMUP_ITERS: u64 = 1_000;
const SNAPSHOT_TIMED_ITERS: u64 = 20_000;
const GOLDFISH_DURATION: Duration = Duration::from_millis(1_500);
const RANDOM_GAME_DURATION: Duration = Duration::from_millis(2_500);
const SELFPLAY_DURATION_PER_CONFIG: Duration = Duration::from_millis(4_000);
const SELFPLAY_THREAD_COUNTS: [usize; 4] = [1, 4, 8, 16];
const LEGAL_ACTIONS_TIMED_ITERS: u64 = 50_000;
#[cfg(feature = "count_allocs")]
const ALLOC_PROFILE_GAMES: u64 = 20;
/// Safety valve only: a bug in the random policy (or a real engine stall)
/// aborts the game instead of hanging the whole suite. Never expected to
/// fire -- typical games run a few hundred to a few thousand decisions.
const DECISION_SAFETY_CAP: u64 = 200_000;

const ATTACK_INCLUDE_CHANCE: (u64, u64) = (1, 2);
const BLOCK_CHANCE: (u64, u64) = (35, 100);

fn main() {
    println!("=== mtg-kernel PERFORMANCE-ONLY benchmark suite (Sol #93 scope) ===");
    println!("Every number below is engine-only wall-clock time: no NN inference, no training.\n");

    section1_snapshot_scaling();
    section2_step_throughput();
    section3_selfplay_threading();
    section4_legal_actions_cost();
    section5_alloc_profile();
}

// ------------------------------------------------------ shared: burn deck

/// Mono-Red Burn mainboard, exactly as played in the golden-corpus mirror
/// (`Mage.Server.Plugins/.../decks/Pauper/Deck - Mono-Red Burn.dek`,
/// `Sideboard="false"` entries only). All 12 of these have a real, fully
/// implemented `spell_effect`/`mana_ability` program in this kernel (see
/// `card_def.rs`'s module doc), so a full random-legal-action self-play
/// game never touches an unimplemented mechanic.
const BURN_MAINBOARD: &[(&str, u32)] = &[
    ("Sneaky Snacker", 4),
    ("Faithless Looting", 2),
    ("Highway Robbery", 4),
    ("Masked Meower", 4),
    ("Lightning Bolt", 4),
    ("Mountain", 18),
    ("Grab the Prize", 4),
    ("Fireblast", 4),
    ("Guttersnipe", 4),
    ("Fiery Temper", 4),
    ("Voldaren Epicure", 4),
    ("Lava Dart", 4),
];

fn burn_deck_ids() -> Vec<u16> {
    let mut ids = Vec::with_capacity(60);
    for &(name, qty) in BURN_MAINBOARD {
        let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} missing from CARD_DEFS"));
        for _ in 0..qty {
            ids.push(id);
        }
    }
    assert_eq!(ids.len(), 60, "Mono-Red Burn mainboard should be exactly 60 cards");
    ids
}

fn shuffled(ids: &[u16], rng: &mut SplitMix64) -> Vec<u16> {
    let mut v = ids.to_vec();
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn debug_name(card_id: u16) -> String {
    CARD_DEFS[card_id as usize].name.to_string()
}

/// A fresh Burn-mirror game: both players shuffle the same 60-card Burn
/// list independently (seeded), then draw their opening 7.
fn build_mirror_state(seed: u64) -> GameState {
    let deck = burn_deck_ids();
    let mut shuffle_rng = SplitMix64::seed(seed);
    let lib0 = shuffled(&deck, &mut shuffle_rng);
    let lib1 = shuffled(&deck, &mut shuffle_rng);
    let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_name, seed);
    for _ in 0..7 {
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P1));
    }
    state
}

// -------------------------------------------------- shared: random policy

fn rng_below(rng: &mut SplitMix64, n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (rng.next_u64() % n as u64) as usize
    }
}

fn rng_chance(rng: &mut SplitMix64, num: u64, den: u64) -> bool {
    rng.next_u64() % den < num
}

/// Uniformly random pick among every legal action `decision` currently
/// offers (Pass included as one of the candidates for `CastSpellOrPass`).
/// Never called on `Decision::GameOver` (the driver loops break before
/// reaching here for that variant).
fn random_action_for_decision(decision: &Decision, rng: &mut SplitMix64) -> Action {
    match decision {
        Decision::CastSpellOrPass { castable_spells, mana_abilities, land_drops, activatable_abilities, plot_actions, .. } => {
            let mut candidates: Vec<Action> = Vec::with_capacity(
                castable_spells.len() + mana_abilities.len() + land_drops.len() + activatable_abilities.len() + plot_actions.len() + 1,
            );
            candidates.extend(castable_spells.iter().map(|&id| Action::CastSpell(id)));
            candidates.extend(mana_abilities.iter().map(|&id| Action::ActivateManaAbility(id)));
            candidates.extend(land_drops.iter().map(|&id| Action::PlayLand(id)));
            candidates.extend(activatable_abilities.iter().map(|&(id, idx)| Action::ActivateAbility(id, idx)));
            candidates.extend(plot_actions.iter().map(|&id| Action::PlotSpell(id)));
            candidates.push(Action::Pass);
            let i = rng_below(rng, candidates.len());
            candidates.swap_remove(i)
        }
        Decision::ChooseTargets { legal_targets, .. } => {
            assert!(!legal_targets.is_empty(), "a real ChooseTargets window must have at least one legal target");
            Action::ChooseTarget(legal_targets[rng_below(rng, legal_targets.len())])
        }
        Decision::ChooseCostTargets { candidates, .. } => Action::ChooseCostTarget(candidates[rng_below(rng, candidates.len())]),
        Decision::ChooseCastMode { options, .. } => Action::ChooseCastMode(options[rng_below(rng, options.len())]),
        Decision::ChooseSpellMode { mode_count, .. } => Action::ChooseSpellMode(rng_below(rng, *mode_count as usize) as u8),
        Decision::ChooseOptionalCost { discard_payable, sacrifice_payable, .. } => {
            let mut options = vec![OptionalCostChoice::Decline];
            if *discard_payable {
                options.push(OptionalCostChoice::Discard);
            }
            if *sacrifice_payable {
                options.push(OptionalCostChoice::SacrificeLand);
            }
            Action::ChooseOptionalCost(options[rng_below(rng, options.len())])
        }
        Decision::ChooseMadnessCast { .. } => Action::ChooseMadnessCast(rng_chance(rng, 1, 2)),
        Decision::Discard { count, choices, .. } => {
            let mut pool = choices.clone();
            let mut picked = Vec::new();
            for _ in 0..*count {
                if pool.is_empty() {
                    break;
                }
                picked.push(pool.swap_remove(rng_below(rng, pool.len())));
            }
            Action::Discard(picked)
        }
        Decision::DeclareAttackers { eligible, .. } => {
            let (num, den) = ATTACK_INCLUDE_CHANCE;
            Action::DeclareAttackers(eligible.iter().copied().filter(|_| rng_chance(rng, num, den)).collect())
        }
        Decision::DeclareBlockers { legal_blockers, .. } => {
            let mut used: HashSet<ObjectId> = HashSet::new();
            let mut pairs = Vec::new();
            let (num, den) = BLOCK_CHANCE;
            for (attacker, blockers) in legal_blockers {
                if !rng_chance(rng, num, den) {
                    continue;
                }
                let avail: Vec<ObjectId> = blockers.iter().copied().filter(|b| !used.contains(b)).collect();
                if !avail.is_empty() {
                    let b = avail[rng_below(rng, avail.len())];
                    used.insert(b);
                    pairs.push((b, *attacker));
                }
            }
            Action::DeclareBlockers(pairs)
        }
        Decision::OrderTriggers { pending, .. } => {
            let mut idx: Vec<usize> = (0..pending.len()).collect();
            for i in (1..idx.len()).rev() {
                let j = rng_below(rng, i + 1);
                idx.swap(i, j);
            }
            Action::OrderTriggers(idx)
        }
        Decision::GameOver { .. } => unreachable!("caller must check for GameOver before requesting an action"),
    }
}

fn warn_safety_cap_once() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("WARNING: a game hit DECISION_SAFETY_CAP ({DECISION_SAFETY_CAP}) and was aborted early -- this should not happen for the Burn mainboard; throughput numbers may be skewed.");
    }
}

/// Plays one full Burn-mirror game through the raw `engine` API (every
/// priority window counted, nothing suppressed). Returns decisions taken.
fn play_one_game_raw(seed: u64) -> u64 {
    let mut state = build_mirror_state(seed);
    let mut rng = SplitMix64::seed(seed ^ 0x5EED_1234_ABCD_0001);
    let mut decisions = 0u64;
    loop {
        let decision = engine::advance_until_decision(&mut state);
        decisions += 1;
        if matches!(decision, Decision::GameOver { .. }) {
            break;
        }
        let action = random_action_for_decision(&decision, &mut rng);
        engine::step(&mut state, action).expect("random policy only picks actions the decision itself listed as legal");
        if decisions >= DECISION_SAFETY_CAP {
            warn_safety_cap_once();
            break;
        }
    }
    decisions
}

/// Plays one full Burn-mirror game through `HarnessSurfaceV2` (the
/// H-visible decision surface). Returns H-visible decisions taken.
fn play_one_game_surface(seed: u64) -> u64 {
    let mut state = build_mirror_state(seed);
    let mut rng = SplitMix64::seed(seed ^ 0x5EED_1234_ABCD_0002);
    let mut surface = HarnessSurfaceV2::new();
    let mut decisions = 0u64;
    loop {
        let sd = surface.next_decision(&mut state);
        decisions += 1;
        match &sd {
            SurfaceDecision::Decision(Decision::GameOver { .. }) => break,
            SurfaceDecision::Decision(d) => {
                let action = random_action_for_decision(d, &mut rng);
                surface.apply(&mut state, SurfaceAction::Action(action)).expect("random policy only picks legal actions");
            }
            SurfaceDecision::DeclareBlockersForAttacker { legal_blockers, .. } => {
                let (num, den) = BLOCK_CHANCE;
                let picks = if !legal_blockers.is_empty() && rng_chance(&mut rng, num, den) {
                    vec![legal_blockers[rng_below(&mut rng, legal_blockers.len())]]
                } else {
                    Vec::new()
                };
                surface.apply(&mut state, SurfaceAction::DeclareBlockersForAttacker(picks)).expect("blockers reshape accepts a subset of the offered legal_blockers");
            }
        }
        if decisions >= DECISION_SAFETY_CAP {
            warn_safety_cap_once();
            break;
        }
    }
    decisions
}

// ---------------------------------------------------------- measurement

fn measure_games<F: FnMut(u64) -> u64>(mut play_one: F, warmup: Duration, target: Duration) -> (u64, u64, Duration) {
    let mut seed = 0u64;
    let warm_start = Instant::now();
    while warm_start.elapsed() < warmup {
        play_one(seed);
        seed += 1;
    }
    let start = Instant::now();
    let mut games = 0u64;
    let mut decisions = 0u64;
    while start.elapsed() < target {
        decisions += play_one(seed);
        games += 1;
        seed += 1;
    }
    (games, decisions, start.elapsed())
}

// ---------------------------------------------------- section 1: snapshot

fn move_hand_to_graveyard(state: &mut GameState, player: PlayerId, id: ObjectId) {
    let ps = &mut state.players[player.index()];
    if let Some(pos) = ps.hand.iter().position(|&h| h == id) {
        ps.hand.remove(pos);
        ps.graveyard.push(id);
        state.objects.get_mut(id).zone = Zone::Graveyard;
    }
}

/// Builds a synthetic state with roughly `total_objects` objects, spread
/// plausibly across library/hand/battlefield/graveyard for both players,
/// plus a couple of fabricated token objects per player (standing in for
/// Voldaren Epicure-style ETB tokens). Object count is fixed at
/// construction (zone moves mutate in place; see `snapshot.rs`'s module
/// doc), so this is a representative shape for a mid-game state of that
/// size -- construction is synthetic (direct zone assignment, not played
/// out turn by turn); plausibility over realism, per the design brief.
fn synthetic_state_of_size(total_objects: u32, seed: u64) -> GameState {
    let cycle = burn_deck_ids();
    let token_card_def = card_id_by_name("Mountain").unwrap();
    let base = total_objects.saturating_sub(4); // reserve 4 slots for fabricated tokens below
    let per_p0 = base / 2;
    let per_p1 = base - per_p0;
    let lib0: Vec<u16> = (0..per_p0).map(|i| cycle[i as usize % cycle.len()]).collect();
    let lib1: Vec<u16> = (0..per_p1).map(|i| cycle[i as usize % cycle.len()]).collect();
    let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_name, seed);

    for player in [PlayerId::P0, PlayerId::P1] {
        for _ in 0..7 {
            state.draw_card(player);
        }
        let hand_snapshot: Vec<ObjectId> = state.players[player.index()].hand.clone();
        for (i, &id) in hand_snapshot.iter().enumerate() {
            if i % 3 == 0 {
                state.move_hand_to_battlefield(player, id);
            }
        }
        for _ in 0..5 {
            if let Some(id) = state.draw_card(player) {
                move_hand_to_graveyard(&mut state, player, id);
            }
        }
        for _ in 0..2 {
            let id = state.objects.push(GameObject {
                card_def: token_card_def,
                name: "Food Token".to_string(),
                owner: player,
                controller: player,
                zone: Zone::Battlefield,
                tapped: false,
                summoning_sick: false,
                damage: 0,
                counters: Counters::default(),
                attachments: Vec::new(),
                plotted_turn: None,
            });
            state.players[player.index()].battlefield.push(id);
        }
    }
    state
}

fn time_iters<F: FnMut()>(mut f: F, warmup: u64, timed: u64) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..timed {
        f();
    }
    start.elapsed()
}

fn section1_snapshot_scaling() {
    println!("--- Section 1: snapshot/restore scaling (PERFORMANCE-ONLY) ---");
    println!("{:>10} {:>12} {:>14} {:>14}", "objects", "actual_objs", "snapshot ns/op", "restore ns/op");
    for &target in &[80u32, 200, 350, 500] {
        let state = synthetic_state_of_size(target, 0xABCD_0000 + target as u64);
        let actual_objects = state.objects.len();

        let snap_elapsed = time_iters(
            || {
                std::hint::black_box(state.snapshot());
            },
            SNAPSHOT_WARMUP_ITERS,
            SNAPSHOT_TIMED_ITERS,
        );
        let snapshot_ns = snap_elapsed.as_nanos() as f64 / SNAPSHOT_TIMED_ITERS as f64;

        let snap = state.snapshot();
        let mut scratch = state.clone();
        let restore_elapsed = time_iters(
            || {
                scratch.restore(&snap);
                std::hint::black_box(&scratch);
            },
            SNAPSHOT_WARMUP_ITERS,
            SNAPSHOT_TIMED_ITERS,
        );
        let restore_ns = restore_elapsed.as_nanos() as f64 / SNAPSHOT_TIMED_ITERS as f64;

        println!("{target:>10} {actual_objects:>12} {snapshot_ns:>14.1} {restore_ns:>14.1}");
    }
    println!();
}

// -------------------------------------------------- section 2: step throughput

fn goldfish_library(names: &[&str]) -> Vec<u16> {
    names.iter().map(|n| card_id_by_name(n).unwrap_or_else(|| panic!("card {n:?} not found"))).collect()
}

/// Deterministic scripted Mono-Red Burn goldfish (see `tests/burn_goldfish.rs`,
/// duplicated here per this file's own "no cross-target sharing" convention --
/// tests/ isn't a library target an example can import from anyway):
/// P0 plays lands then bolts P1 to death; P1 never casts anything.
fn play_goldfish_once(_seed: u64) -> u64 {
    let mut p0 = vec!["Mountain", "Mountain", "Mountain"];
    p0.extend(std::iter::repeat_n("Lightning Bolt", 14));
    p0.extend(std::iter::repeat_n("Mountain", 30));
    let p1: Vec<&str> = std::iter::repeat_n("Mountain", 50).collect();
    let p0_lib = goldfish_library(&p0);
    let p1_lib = goldfish_library(&p1);
    let mut state = GameState::new_from_libraries(&p0_lib, &p1_lib, debug_name, 0xC0FFEE);
    for _ in 0..7 {
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P1));
    }

    let bolt_def = card_id_by_name("Lightning Bolt").unwrap();
    let mut last_cast_turn = 0u32;
    let mut decisions = 0u64;
    loop {
        let decision = engine::advance_until_decision(&mut state);
        decisions += 1;
        match &decision {
            Decision::GameOver { .. } => break,
            Decision::DeclareAttackers { .. } => {
                engine::step(&mut state, Action::DeclareAttackers(Vec::new())).unwrap();
            }
            Decision::ChooseTargets { .. } => {
                engine::step(&mut state, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();
            }
            Decision::CastSpellOrPass { player, land_drops, .. } => {
                let player = *player;
                if !land_drops.is_empty() {
                    let land = land_drops[0];
                    engine::step(&mut state, Action::PlayLand(land)).unwrap();
                } else if player == PlayerId::P0 && state.turn >= 2 && state.turn != last_cast_turn {
                    let bolt_in_hand = state.players[0].hand.iter().copied().find(|&id| state.objects.get(id).card_def == bolt_def);
                    if let Some(bolt) = bolt_in_hand {
                        last_cast_turn = state.turn;
                        engine::step(&mut state, Action::CastSpell(bolt)).unwrap();
                    } else {
                        engine::step(&mut state, Action::Pass).unwrap();
                    }
                } else {
                    engine::step(&mut state, Action::Pass).unwrap();
                }
            }
            _ => unreachable!("the burn goldfish's library (Mountain + Lightning Bolt only) cannot produce this decision"),
        }
        if decisions >= DECISION_SAFETY_CAP {
            warn_safety_cap_once();
            break;
        }
    }
    decisions
}

fn section2_step_throughput() {
    println!("--- Section 2: step throughput, single-threaded (PERFORMANCE-ONLY) ---");

    let (games, decisions, elapsed) = measure_games(play_goldfish_once, Duration::from_millis(200), GOLDFISH_DURATION);
    let secs = elapsed.as_secs_f64();
    let games_per_sec = games as f64 / secs;
    let decisions_per_sec = decisions as f64 / secs;
    println!(
        "(a) Burn goldfish script, replayed:  {games_per_sec:>8.1} games/sec   {decisions_per_sec:>10.1} decisions/sec   ({games} games, {decisions} decisions, {secs:.2}s)"
    );

    let (games, decisions, elapsed) = measure_games(play_one_game_raw, Duration::from_millis(300), RANDOM_GAME_DURATION);
    let secs = elapsed.as_secs_f64();
    let games_per_sec = games as f64 / secs;
    let decisions_per_sec = decisions as f64 / secs;
    println!(
        "(b) full Burn-mirror, random policy: {games_per_sec:>8.1} games/sec   {decisions_per_sec:>10.1} decisions/sec   ({games} games, {decisions} decisions, {secs:.2}s)"
    );
    println!();
}

// ------------------------------------------------- section 3: self-play threading

fn section3_selfplay_threading() {
    println!("--- Section 3 (HEADLINE): self-play throughput via HarnessSurfaceV2, N threads (PERFORMANCE-ONLY) ---");
    println!("Compare against Java's ~3 eps/sec (24-core box, 48 runners, FULL training stack incl. NN inference+training --");
    println!("this section is ENGINE+SURFACE ONLY, no NN, so this is an upper bound on what the engine alone could feed a learner.");
    println!("{:>8} {:>14} {:>18} {:>18}", "threads", "games/sec", "H-visible dec/sec", "games/sec/thread");

    for &n in &SELFPLAY_THREAD_COUNTS {
        let start = Instant::now();
        let handles: Vec<_> = (0..n)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut seed = 0x9E37_0000u64 + (t as u64) * 7_919;
                    let warm_deadline = Instant::now() + Duration::from_millis(150);
                    while Instant::now() < warm_deadline {
                        play_one_game_surface(seed);
                        seed = seed.wrapping_add(1);
                    }
                    let deadline = Instant::now() + SELFPLAY_DURATION_PER_CONFIG;
                    let mut games = 0u64;
                    let mut decisions = 0u64;
                    while Instant::now() < deadline {
                        decisions += play_one_game_surface(seed);
                        games += 1;
                        seed = seed.wrapping_add(1);
                    }
                    (games, decisions)
                })
            })
            .collect();

        let mut total_games = 0u64;
        let mut total_decisions = 0u64;
        for h in handles {
            let (g, d) = h.join().expect("worker thread panicked");
            total_games += g;
            total_decisions += d;
        }
        let wall = start.elapsed().as_secs_f64();
        let games_per_sec = total_games as f64 / wall;
        let decisions_per_sec = total_decisions as f64 / wall;
        println!("{n:>8} {games_per_sec:>14.2} {decisions_per_sec:>18.1} {:>18.3}", games_per_sec / n as f64);
    }
    println!();
}

// -------------------------------------------------- section 4: legal actions

/// Plays a raw random-policy game up to `max_decisions`, remembering the
/// richest `CastSpellOrPass` window seen (castable spells + mana abilities
/// + a land drop all simultaneously available -- the brief's "cast windows
///   with several castable spells + mana abilities + land drop" shape) and
///   returns a snapshot of the state at that point.
fn capture_rich_cast_window(seed: u64, max_decisions: u64) -> GameState {
    let mut state = build_mirror_state(seed);
    let mut rng = SplitMix64::seed(seed ^ 0x5EED_1234_ABCD_0003);
    let mut best: Option<GameState> = None;
    let mut best_score = 0usize;
    for _ in 0..max_decisions {
        let decision = engine::advance_until_decision(&mut state);
        if let Decision::CastSpellOrPass { castable_spells, mana_abilities, land_drops, .. } = &decision {
            if !castable_spells.is_empty() && !mana_abilities.is_empty() && !land_drops.is_empty() {
                let score = castable_spells.len() + mana_abilities.len();
                if score > best_score {
                    best_score = score;
                    best = Some(state.clone());
                }
            }
        }
        if matches!(decision, Decision::GameOver { .. }) {
            break;
        }
        let action = random_action_for_decision(&decision, &mut rng);
        if engine::step(&mut state, action).is_err() {
            break;
        }
    }
    best.unwrap_or(state)
}

fn section4_legal_actions_cost() {
    println!("--- Section 4: legal-actions enumeration cost (PERFORMANCE-ONLY) ---");
    let captured = capture_rich_cast_window(0x1357_9BDF, 2_000);
    let decision = engine::advance_until_decision(&mut captured.clone());
    let (castable, mana, land) = match &decision {
        Decision::CastSpellOrPass { castable_spells, mana_abilities, land_drops, .. } => (castable_spells.len(), mana_abilities.len(), land_drops.len()),
        _ => (0, 0, 0),
    };
    println!("captured window: {castable} castable spells, {mana} mana abilities, {land} land drops, {} total objects", captured.objects.len());

    let mut probe = captured.clone();
    let elapsed = time_iters(
        || {
            std::hint::black_box(engine::advance_until_decision(&mut probe));
        },
        1_000,
        LEGAL_ACTIONS_TIMED_ITERS,
    );
    let ns_per_call = elapsed.as_nanos() as f64 / LEGAL_ACTIONS_TIMED_ITERS as f64;
    assert_eq!(probe, captured, "advance_until_decision must not mutate an already-idle decision state");
    println!("advance_until_decision (idle re-enumeration): {ns_per_call:.1} ns/op over {LEGAL_ACTIONS_TIMED_ITERS} iterations");

    // Section 3's HarnessSurfaceV2::next_decision computes `state.state_hash()`
    // unconditionally at the top of every internal loop iteration (surface_v2.rs:350),
    // even for iterations that end up surfaced (not suppressed) -- where the hash is
    // computed and then never used. Measured here directly to attribute section 3's
    // per-decision cost relative to section 2's hash-free raw driver.
    let hash_elapsed = time_iters(
        || {
            std::hint::black_box(captured.state_hash());
        },
        1_000,
        LEGAL_ACTIONS_TIMED_ITERS,
    );
    let hash_ns_per_call = hash_elapsed.as_nanos() as f64 / LEGAL_ACTIONS_TIMED_ITERS as f64;
    println!("state_hash() on the same captured state:      {hash_ns_per_call:.1} ns/op over {LEGAL_ACTIONS_TIMED_ITERS} iterations");
    println!();
}

// --------------------------------------------------- section 5: allocator

#[cfg(feature = "count_allocs")]
mod counting_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
    pub static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

    pub struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    pub fn snapshot() -> (u64, u64) {
        (ALLOC_COUNT.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
    }
}

#[cfg(feature = "count_allocs")]
#[global_allocator]
static GLOBAL: counting_alloc::CountingAlloc = counting_alloc::CountingAlloc;

#[cfg(feature = "count_allocs")]
fn section5_alloc_profile() {
    println!("--- Section 5: allocator profile (PERFORMANCE-ONLY, count_allocs feature) ---");
    let mut seed = 0x4110_C000u64;
    // warm up allocator/heap so the first game's one-time setup cost doesn't skew the average
    play_one_game_raw(seed);
    seed += 1;

    let (count_before, bytes_before) = counting_alloc::snapshot();
    let mut total_decisions = 0u64;
    for _ in 0..ALLOC_PROFILE_GAMES {
        total_decisions += play_one_game_raw(seed);
        seed += 1;
    }
    let (count_after, bytes_after) = counting_alloc::snapshot();

    let allocs = count_after - count_before;
    let bytes = bytes_after - bytes_before;
    println!(
        "{ALLOC_PROFILE_GAMES} games, {total_decisions} decisions: {:.0} allocations/game, {:.0} bytes/game, {:.1} allocations/decision",
        allocs as f64 / ALLOC_PROFILE_GAMES as f64,
        bytes as f64 / ALLOC_PROFILE_GAMES as f64,
        allocs as f64 / total_decisions as f64
    );
    println!();
}

#[cfg(not(feature = "count_allocs"))]
fn section5_alloc_profile() {
    println!("--- Section 5: allocator profile (SKIPPED) ---");
    println!("Run with: cargo run --release --example bench_kernel --features count_allocs");
    println!();
}
