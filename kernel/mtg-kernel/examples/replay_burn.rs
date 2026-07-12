//! Golden-trace replay: drives every trace in a corpus through the kernel
//! engine, gate-checking each decision against the trace's logged
//! candidates/choice, and reports an honest scoreboard (not a pass rate to
//! maximize -- partial success this increment is expected).
//!
//! Run: cargo run --release --example replay_burn -- <corpus dir>
//!
//! ## Two non-obvious trace-format facts this driver depends on
//!
//! 1. `DECLARE_ATTACKS`/`DECLARE_BLOCKS` log `chosen_indices` as a full
//!    permutation of *every* candidate (including `sentinel:DONE`), not a
//!    shrinking iterative pick list. The real applied action is the
//!    *prefix* of that permutation up to (excluding) the first `DONE`
//!    entry -- confirmed against the free-text `DECLARE_ATTACKS:
//!    selected=[...] from N possible` lines and the post-decision
//!    `tapped,attacking` permanent markers in real trace files. Taking
//!    "every non-DONE id" instead (as a naive reading of the format
//!    might suggest) silently over-attacks/over-blocks.
//! 2. `DecisionRecord::turn` is XMage's *global* absolute turn counter
//!    (P0 on odd turns, P1 on even, alternating every individual
//!    player-turn) -- not the kernel's `GameState::turn`, which is a
//!    *round* counter (increments once per round, when play returns to
//!    P0). `expected_round = rec.turn.div_ceil(2)`, cross-checked against
//!    the free-text `DECISION #N - Turn R (<player> turn)` headers (e.g.
//!    json turn=13 <-> "Turn 7 (PlayerRL1 turn)"; json turn=26 <-> "Turn
//!    13 (SelfPlay turn)").
//! 3. `ACTIVATE_ABILITY_OR_SPELL` candidates are deduplicated by XMage
//!    itself: 2 untapped Mountains show up as a single "Play Mountain" /
//!    "{T}: Add {R}." candidate, not two (confirmed empirically: 0 of
//!    3159 real records in the corpus ever carry a duplicate
//!    `candidate_text`). The kernel's `land_drops`/`mana_abilities`/etc.
//!    are per-object and don't dedupe, so this driver collapses them by
//!    equivalence class (bucket + card name [+ ability index / hand-vs-
//!    flashback]) before comparing -- see `apply_cast_spell_or_pass`.
//!    `DECLARE_ATTACKS`/`DECLARE_BLOCKS`/`SELECT_TARGETS` do NOT dedupe
//!    (two same-named creatures keep distinct candidates there, since
//!    which specific one attacks/blocks/gets targeted is a real choice).

use mtg_kernel::card_def::{self, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

const DONE: &str = "sentinel:DONE";

fn main() {
    let root = std::env::args().nth(1).map(PathBuf::from).expect("usage: replay_burn <corpus dir>");
    let (traces, errors) = trace::load_corpus(&root);
    println!("traces parsed: {}   parse errors: {}", traces.len(), errors.len());
    for e in errors.iter().take(5) {
        println!("  ERR {e}");
    }

    let mut attempted = 0usize;
    let mut replayed_to_end = 0usize;
    let mut winner_matched = 0usize;
    let mut diverged = 0usize;
    let mut trace_exhausted_pass_total = 0usize;
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut phantom_total = 0usize;
    let mut decisions_consumed_total = 0usize;
    let mut decisions_total_total = 0usize;
    // (reason, source_path) per diverged trace, for `REPLAY_DEBUG`'s
    // per-trace listing below -- lets a divergence be traced back to its
    // exact file without re-running with `REPLAY_TRACE_FILTER` per
    // candidate first.
    let mut per_trace_divergence: Vec<(String, String)> = Vec::new();

    for t in &traces {
        attempted += 1;
        phantom_total += t.phantom_decisions_skipped;
        let outcome = replay_trace(t);
        trace_exhausted_pass_total += outcome.trace_exhausted_passes;
        decisions_consumed_total += outcome.decisions_consumed;
        decisions_total_total += outcome.decisions_total;
        if outcome.reached_game_over {
            replayed_to_end += 1;
            if outcome.winner_matched {
                winner_matched += 1;
            }
        }
        if let Some(reason) = outcome.divergence {
            diverged += 1;
            *histogram.entry(reason.clone()).or_default() += 1;
            per_trace_divergence.push((reason, t.source_path.clone()));
        }
    }

    println!("\nphantom (episode<0) decision records skipped across corpus: {phantom_total}");
    println!("\n--- scoreboard ---");
    println!("traces attempted:                 {attempted}");
    println!("replayed to end (GameOver seen):  {replayed_to_end}");
    println!("winner matched:                   {winner_matched}");
    println!("diverged:                         {diverged}");
    println!("trace-exhausted-pass occurrences (informational, not a failure): {trace_exhausted_pass_total}");
    // A softer signal than the binary reached/diverged split: how much of
    // each trace's real decision stream validated cleanly before either
    // GameOver or the first divergence.
    let pct = if decisions_total_total > 0 { 100.0 * decisions_consumed_total as f64 / decisions_total_total as f64 } else { 0.0 };
    println!(
        "real (non-mulligan) decisions gate-checked and applied before GameOver/divergence: {decisions_consumed_total} / {decisions_total_total} ({pct:.1}%)"
    );
    println!("\nfirst-divergence-reason histogram:");
    if histogram.is_empty() {
        println!("  (none)");
    }
    for (reason, n) in &histogram {
        println!("  {n:>4}  {reason}");
    }

    // Per-trace detail, gated behind the same `REPLAY_DEBUG` flag
    // `debug_verbose` uses: not part of the default scoreboard output (no
    // script parses this example's stdout -- see the increment-5 report),
    // but pinpoints exactly which file to hand to `REPLAY_TRACE_FILTER`
    // for a given divergence reason.
    if std::env::var("REPLAY_DEBUG").is_ok() {
        println!("\nper-trace divergence (REPLAY_DEBUG):");
        let mut sorted = per_trace_divergence.clone();
        sorted.sort();
        for (reason, path) in &sorted {
            println!("  {reason:<45} {path}");
        }
    }
}

#[derive(Default)]
struct ReplayOutcome {
    reached_game_over: bool,
    winner_matched: bool,
    trace_exhausted_passes: usize,
    divergence: Option<String>,
    /// Real (non-mulligan) trace decisions successfully gate-checked and
    /// applied before either reaching `GameOver` or hitting the first
    /// divergence. A more nuanced signal than the binary reached/diverged
    /// split: two traces that both "diverge" can still differ hugely in
    /// how much of the game replayed cleanly first.
    decisions_consumed: usize,
    /// Total real (non-mulligan) decisions in the trace, for computing
    /// `decisions_consumed`'s share.
    decisions_total: usize,
}

fn replay_trace(t: &GoldenTrace) -> ReplayOutcome {
    let mut outcome = ReplayOutcome::default();
    if let Err(reason) = run(t, &mut outcome) {
        outcome.divergence = Some(reason);
    }
    outcome
}

/// Per-trace replay context: everything derived once at setup time and
/// held immutably for the rest of the replay (id/player maps, per-seat
/// decision queues).
struct ReplayCtx<'a> {
    id_map: HashMap<String, ObjectId>,
    seat_uuid: [Option<String>; 2],
    queues: [Vec<&'a DecisionRecord>; 2],
    cursors: [usize; 2],
}

impl<'a> ReplayCtx<'a> {
    fn next(&self, seat: PlayerId) -> Option<&&'a DecisionRecord> {
        self.queues[seat.index()].get(self.cursors[seat.index()])
    }

    fn advance(&mut self, seat: PlayerId) {
        self.cursors[seat.index()] += 1;
    }
}

fn run(t: &GoldenTrace, outcome: &mut ReplayOutcome) -> Result<(), String> {
    if std::env::var("REPLAY_DEBUG").is_ok() {
        eprintln!("=== {} ===", t.source_path);
    }
    let (p0_name, p1_name) = seat_names(t)?;
    let opening0 = t.opening_hand_for(&p0_name).ok_or("setup:no-opening-hand-record:p0")?;
    let opening1 = t.opening_hand_for(&p1_name).ok_or("setup:no-opening-hand-record:p1")?;

    let lib0 = card_ids_for(opening0.hand.iter().chain(opening0.library.iter()))?;
    let lib1 = card_ids_for(opening1.hand.iter().chain(opening1.library.iter()))?;

    let mut state = GameState::new_from_libraries(&lib0, &lib1, |id| CARD_DEFS[id as usize].name.to_string(), t.header.seed);
    for _ in 0..opening0.hand.len() {
        state.draw_card(PlayerId::P0);
    }
    for _ in 0..opening1.hand.len() {
        state.draw_card(PlayerId::P1);
    }

    let id_map = build_id_map(&opening0, &opening1, lib0.len() as u32);
    let seat_uuid = find_player_uuids(t, &p0_name, &p1_name);

    let queue_for = |name: &str| -> Vec<&DecisionRecord> {
        t.decisions.iter().filter(|d| d.player == name && d.action_type != "MULLIGAN" && d.action_type != "LONDON_MULLIGAN").collect()
    };
    let mut ctx = ReplayCtx { id_map, seat_uuid, queues: [queue_for(&p0_name), queue_for(&p1_name)], cursors: [0, 0] };
    outcome.decisions_total = ctx.queues[0].len() + ctx.queues[1].len();

    loop {
        let decision = engine::advance_until_decision(&mut state);
        match decision {
            Decision::GameOver { winner } => {
                outcome.reached_game_over = true;
                let winner_name = winner.map(|p| if p == PlayerId::P0 { p0_name.clone() } else { p1_name.clone() });
                outcome.winner_matched = matches!((&winner_name, &t.winner), (Some(a), Some(b)) if a == b);
                return Ok(());
            }
            Decision::CastSpellOrPass { player, castable_spells, mana_abilities, land_drops, activatable_abilities } => {
                // The Java reference itself only logs an
                // ACTIVATE_ABILITY_OR_SPELL decision when there's a real
                // alternative to Pass -- confirmed empirically: across
                // the whole corpus, zero decision records carry phase
                // Untap/Upkeep/Draw/Cleanup (the priority windows where
                // nothing is ever castable this early), and every logged
                // record has >= 2 candidates. So a kernel CastSpellOrPass
                // with no real option (Pass-only) is expected to have no
                // trace counterpart at all -- unlogged on the reference
                // side, not just missing from our cursor -- and gets
                // auto-passed without consuming the trace queue. This is
                // the faithful generalization of the design brief's
                // "trace-exhausted" carve-out (it undershot: the gap
                // isn't only at the tail of a seat's queue, it's every
                // Pass-only window throughout the game).
                let no_real_option = castable_spells.is_empty() && mana_abilities.is_empty() && land_drops.is_empty() && activatable_abilities.is_empty();
                if no_real_option {
                    engine::step(&mut state, Action::Pass).map_err(|e| format!("engine-step-error:CastSpellOrPass:{e}"))?;
                    outcome.trace_exhausted_passes += 1;
                    continue;
                }
                match ctx.next(player) {
                    None => return Err("trace-exhausted:CastSpellOrPass-with-real-options".to_string()),
                    Some(&rec) => {
                        debug_verbose(t, &state, player, rec, "CastSpellOrPass");
                        if rec.action_type != "ACTIVATE_ABILITY_OR_SPELL" {
                            if std::env::var("REPLAY_DEBUG").is_ok() {
                                eprintln!(
                                    "KIND MISMATCH decision_number={} player={} expected=CastSpellOrPass got={} kernel_castable={:?} kernel_land={:?} kernel_mana={:?}",
                                    rec.decision_number, rec.player, rec.action_type, castable_spells, land_drops, mana_abilities
                                );
                            }
                            return Err(format!("decision-kind-mismatch:CastSpellOrPass-vs-{}", rec.action_type));
                        }
                        check_state(&state, player, rec)?;
                        apply_cast_spell_or_pass(&mut state, rec, &castable_spells, &mana_abilities, &land_drops, &activatable_abilities, &ctx.id_map)?;
                        ctx.advance(player);
                        outcome.decisions_consumed += 1;
                    }
                }
            }
            Decision::ChooseTargets { player, legal_targets, .. } => {
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:ChooseTargets".to_string())?;
                debug_verbose(t, &state, player, rec, "ChooseTargets");
                if rec.action_type != "SELECT_TARGETS" {
                    return Err(format!("decision-kind-mismatch:ChooseTargets-vs-{}", rec.action_type));
                }
                // 601.2a: the kernel now moves a cast spell (or, for
                // flashback, the graveyard card) onto the stack at
                // announcement, before targets are chosen -- matching the
                // reference engine, so hand/graveyard sizes agree here with
                // no fudge factor needed (see `engine::begin_cast`).
                check_state(&state, player, rec)?;
                apply_choose_targets(&mut state, rec, &legal_targets, &ctx.id_map, &ctx.seat_uuid)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            Decision::DeclareAttackers { player, eligible } => {
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:DeclareAttackers".to_string())?;
                debug_verbose(t, &state, player, rec, "DeclareAttackers");
                if rec.action_type != "DECLARE_ATTACKS" {
                    return Err(format!("decision-kind-mismatch:DeclareAttackers-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec)?;
                apply_declare_attackers(&mut state, rec, &eligible, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            Decision::DeclareBlockers { player, legal_blockers, .. } => {
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:DeclareBlockers".to_string())?;
                debug_verbose(t, &state, player, rec, "DeclareBlockers");
                if rec.action_type != "DECLARE_BLOCKS" {
                    return Err(format!("decision-kind-mismatch:DeclareBlockers-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec)?;
                apply_declare_blockers(&mut state, rec, &legal_blockers, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            // Not observed anywhere in this corpus (verified by grep
            // across all 40 files); no sensible trace counterpart exists
            // to translate against, so this is a clean, named divergence
            // rather than a guess or a crash.
            Decision::ChooseCastMode { .. } => return Err("unhandled-decision:ChooseCastMode".to_string()),
            Decision::Discard { .. } => return Err("unhandled-decision:Discard".to_string()),
            Decision::OrderTriggers { .. } => return Err("unhandled-decision:OrderTriggers".to_string()),
        }
    }
}

/// P0 = the `player` of the trace's very first decision record; P1 = the
/// other of the two distinct `player` names seen in the trace.
///
/// This is *not* "`active_player` of the first non-mulligan decision" (a
/// reading the design brief suggested): that breaks the same way
/// `GoldenTrace::opening_hand_for` did, and worse -- if P0 has nothing
/// playable on turn 1, the first *logged* non-mulligan decision can
/// belong to P1's turn instead, so `active_player` there names P1, not
/// the true starting player, silently swapping the seats for the whole
/// replay (verified empirically: e.g. `game_20260712_183257_0002.txt`'s
/// first non-mulligan decision is SelfPlay's, on turn 2, even though
/// PlayerRL1 goes first). The mulligan phase, by contrast, always
/// happens for both players before any turn structure runs, in APNAP
/// order (the starting player decides first) -- so the very first
/// decision record in the file, mulligan or not, reliably names P0.
fn seat_names(t: &GoldenTrace) -> Result<(String, String), String> {
    let mut names: Vec<&str> = Vec::new();
    for d in &t.decisions {
        if !names.contains(&d.player.as_str()) {
            names.push(&d.player);
        }
    }
    if names.len() != 2 {
        return Err(format!("setup:player-name-count={}", names.len()));
    }
    let p0 = t.decisions.first().ok_or_else(|| "setup:no-decisions".to_string())?.player.clone();
    let p1 = names.into_iter().find(|&n| n != p0).ok_or_else(|| "setup:cannot-determine-p1-name".to_string())?.to_string();
    Ok((p0, p1))
}

fn debug_verbose(t: &GoldenTrace, state: &GameState, player: PlayerId, rec: &DecisionRecord, kind: &str) {
    let Ok(filter) = std::env::var("REPLAY_TRACE_FILTER") else { return };
    if !t.source_path.contains(&filter) {
        return;
    }
    let ps = &state.players[player.index()];
    eprintln!(
        "  [{kind}] decision_number={} player={} action={} rec_turn={} state_turn={} kernel_hand={} trace_hand={} kernel_lib={} trace_lib={}",
        rec.decision_number,
        rec.player,
        rec.action_type,
        rec.turn,
        state.turn,
        ps.hand.len(),
        rec.hand.len(),
        ps.library.len(),
        rec.library.len(),
    );
}

fn card_ids_for<'a>(names: impl Iterator<Item = &'a String>) -> Result<Vec<u16>, String> {
    names.map(|n| card_def::card_id_by_name(n).ok_or_else(|| format!("setup:unknown-card-name:{n}"))).collect()
}

/// Zips each seat's `hand_object_ids ++ library_object_ids` (trace UUIDs)
/// against the kernel `ObjectId`s `GameState::new_from_libraries` assigns
/// in that same order (0..len(lib0) for P0, offset thereafter for P1 --
/// see `state.rs`'s `new_from_libraries_assigns_ids_p0_first` test).
fn build_id_map(opening0: &trace::OpeningHand, opening1: &trace::OpeningHand, p0_object_count: u32) -> HashMap<String, ObjectId> {
    let mut id_map = HashMap::new();
    for (i, uuid) in opening0.hand_object_ids.iter().chain(opening0.library_object_ids.iter()).enumerate() {
        id_map.insert(uuid.clone(), ObjectId(i as u32));
    }
    for (i, uuid) in opening1.hand_object_ids.iter().chain(opening1.library_object_ids.iter()).enumerate() {
        id_map.insert(uuid.clone(), ObjectId(p0_object_count + i as u32));
    }
    id_map
}

/// Captures each seat's persistent player-object UUID (a distinct
/// namespace from card UUIDs) from the first `SELECT_TARGETS` record
/// whose `candidate_texts` mention a player's display name -- e.g.
/// Lightning Bolt's target list includes both players by name.
fn find_player_uuids(t: &GoldenTrace, p0_name: &str, p1_name: &str) -> [Option<String>; 2] {
    let mut found: HashMap<&str, String> = HashMap::new();
    for d in &t.decisions {
        if d.action_type != "SELECT_TARGETS" {
            continue;
        }
        for (text, uuid) in d.candidate_texts.iter().zip(d.candidate_object_ids.iter()) {
            if (text == p0_name || text == p1_name) && !found.contains_key(text.as_str()) {
                found.insert(if text == p0_name { p0_name } else { p1_name }, uuid.clone());
            }
        }
        if found.len() == 2 {
            break;
        }
    }
    [found.get(p0_name).cloned(), found.get(p1_name).cloned()]
}

/// State comparison at each decision boundary (this increment's floor per
/// the design brief): turn, then zone sizes for the *acting* player
/// (hand/library/graveyard counts). Turn is checked first because it's
/// the more diagnostic signal when the two diverge together -- a wrong
/// turn almost always *causes* a wrong hand size (a missed/extra
/// natural draw), not the other way around, and "turn-mismatch" says
/// that plainly where "zone-size-mismatch:hand" wouldn't. The opponent's
/// hand is hidden in the trace (only a card count would be knowable, and
/// only indirectly, from a later record where they act) so this
/// increment does not attempt to verify the non-acting player's zones.
/// Phase is intentionally NOT gated: kernel `Step` and XMage's `phase`
/// free-text label are related but distinct groupings (e.g. XMage's
/// "Combat" phase covers 5 kernel steps) and a hand-built mapping risks
/// manufacturing false divergences; life totals are a documented
/// stretch-goal gap (see the increment-4 report).
///
/// No fudge factor for hand/graveyard size during `ChooseTargets` anymore:
/// `engine::begin_cast` now follows 601.2a and moves the spell to the stack
/// at announcement, before targets are chosen, so the kernel's hand/
/// graveyard counts already agree with the trace's snapshot at every
/// decision boundary, cast-in-progress or not (previously this function
/// took a `+1`-while-mid-cast fudge from its callers to paper over the
/// kernel deferring that move to `finalize_cast`; removed along with the
/// bug it was working around).
fn check_state(state: &GameState, player: PlayerId, rec: &DecisionRecord) -> Result<(), String> {
    let ps = &state.players[player.index()];
    // See the module doc: `rec.turn` is XMage's global absolute turn
    // counter, not the kernel's round counter.
    let expected_round = rec.turn.div_ceil(2);
    if state.turn != expected_round {
        return Err("turn-mismatch".to_string());
    }
    if ps.hand.len() != rec.hand.len() {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "HAND MISMATCH decision_number={} player={} action={} rec_turn={} state_turn={} kernel_hand={} trace_hand={} kernel_names={:?} trace_names={:?}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                rec.turn,
                state.turn,
                ps.hand.len(),
                rec.hand.len(),
                ps.hand.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                rec.hand
            );
        }
        return Err("zone-size-mismatch:hand".to_string());
    }
    if ps.library.len() != rec.library.len() {
        return Err("zone-size-mismatch:library".to_string());
    }
    if ps.graveyard.len() != rec.graveyard.len() {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "GRAVEYARD MISMATCH decision_number={} player={} action={} kernel_gy={} trace_gy={} kernel_names={:?} trace_names={:?}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                ps.graveyard.len(),
                rec.graveyard.len(),
                ps.graveyard.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                rec.graveyard
            );
        }
        return Err("zone-size-mismatch:graveyard".to_string());
    }
    Ok(())
}

enum KernelChoice {
    Pass,
    PlayLand(ObjectId),
    CastSpell(ObjectId),
    ActivateMana(ObjectId),
    ActivateAbility(ObjectId, u8),
}

/// Equivalence-class key for an `ACTIVATE_ABILITY_OR_SPELL` candidate:
/// which bucket it's from, plus enough of its identity (card name,
/// hand-vs-flashback, ability index) to distinguish genuinely different
/// choices while collapsing fungible ones (2 untapped Mountains) into the
/// same key -- see the module doc's point 3. `None` means `id` isn't a
/// member of any of the 4 current buckets (a real divergence signal, not
/// swallowed by the caller).
fn candidate_key(
    state: &GameState,
    id: ObjectId,
    land_drops: &[ObjectId],
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
) -> Option<String> {
    let name = &state.objects.get(id).name;
    if land_drops.contains(&id) {
        return Some(format!("land:{name}"));
    }
    if castable_spells.contains(&id) {
        let is_flashback = state.objects.get(id).zone == Zone::Graveyard;
        return Some(format!("cast:{name}:{is_flashback}"));
    }
    if mana_abilities.contains(&id) {
        return Some(format!("mana:{name}"));
    }
    if let Some(&(_, idx)) = activatable_abilities.iter().find(|&&(oid, _)| oid == id) {
        return Some(format!("activate:{name}:{idx}"));
    }
    None
}

fn apply_cast_spell_or_pass(
    state: &mut GameState,
    rec: &DecisionRecord,
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    land_drops: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(), String> {
    // One representative KernelChoice per equivalence class (first one
    // encountered wins -- any representative works, they're fungible by
    // construction; see `candidate_key`'s doc and the module doc's point
    // 3). A given object can only land in one of the 4 buckets in this
    // card pool (land vs. castable are mutually exclusive by zone/type;
    // mana/activated abilities are battlefield-only, disjoint from
    // hand-only land/cast candidates), so this is well-defined.
    let mut by_key: BTreeMap<String, KernelChoice> = BTreeMap::new();
    by_key.insert("pass".to_string(), KernelChoice::Pass);
    for &id in land_drops {
        by_key.entry(format!("land:{}", state.objects.get(id).name)).or_insert(KernelChoice::PlayLand(id));
    }
    for &id in castable_spells {
        let is_flashback = state.objects.get(id).zone == Zone::Graveyard;
        by_key.entry(format!("cast:{}:{is_flashback}", state.objects.get(id).name)).or_insert(KernelChoice::CastSpell(id));
    }
    for &id in mana_abilities {
        by_key.entry(format!("mana:{}", state.objects.get(id).name)).or_insert(KernelChoice::ActivateMana(id));
    }
    for &(id, idx) in activatable_abilities {
        by_key.entry(format!("activate:{}:{idx}", state.objects.get(id).name)).or_insert(KernelChoice::ActivateAbility(id, idx));
    }
    let mut kernel_keys: Vec<String> = by_key.keys().cloned().collect();
    kernel_keys.sort();

    let trace_ids = translate_object_candidates(rec, id_map, "CastSpellOrPass")?;
    let mut trace_keys = Vec::with_capacity(trace_ids.len());
    for id in &trace_ids {
        let key = match id {
            None => "pass".to_string(),
            Some(oid) => candidate_key(state, *oid, land_drops, castable_spells, mana_abilities, activatable_abilities)
                .ok_or_else(|| "trace-candidate-not-in-any-kernel-bucket:CastSpellOrPass".to_string())?,
        };
        trace_keys.push(key);
    }
    let mut sorted_trace_keys = trace_keys.clone();
    sorted_trace_keys.sort();

    if kernel_keys != sorted_trace_keys {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "MISMATCH decision_number={} texts={:?} kernel={:?} trace={:?}",
                rec.decision_number, rec.candidate_texts, kernel_keys, sorted_trace_keys
            );
        }
        return Err("candidate-multiset-mismatch:CastSpellOrPass".to_string());
    }

    if rec.chosen_indices.len() != 1 {
        return Err("unexpected-chosen-count:CastSpellOrPass".to_string());
    }
    let idx = rec.chosen_indices[0] as usize;
    let chosen_key = trace_keys.get(idx).ok_or("chosen-index-out-of-range:CastSpellOrPass")?;

    let action = match by_key.get(chosen_key) {
        Some(KernelChoice::Pass) => Action::Pass,
        Some(KernelChoice::PlayLand(id)) => Action::PlayLand(*id),
        Some(KernelChoice::CastSpell(id)) => Action::CastSpell(*id),
        Some(KernelChoice::ActivateMana(id)) => Action::ActivateManaAbility(*id),
        Some(KernelChoice::ActivateAbility(id, idx)) => Action::ActivateAbility(*id, *idx),
        None => return Err("chosen-not-in-kernel-candidates:CastSpellOrPass".to_string()),
    };
    engine::step(state, action).map_err(|e| format!("engine-step-error:CastSpellOrPass:{e}"))
}

/// Translates every `(candidate_texts[i], candidate_object_ids[i])` pair
/// of an `ACTIVATE_ABILITY_OR_SPELL` record into `Option<ObjectId>`
/// (`None` = the implicit "Pass" candidate, `candidate_object_ids[i] ==
/// ""`), in original candidate order (index-addressable, so
/// `chosen_indices` can look items up directly).
fn translate_object_candidates(rec: &DecisionRecord, id_map: &HashMap<String, ObjectId>, kind: &str) -> Result<Vec<Option<ObjectId>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_texts.len());
    for (text, uuid) in rec.candidate_texts.iter().zip(rec.candidate_object_ids.iter()) {
        if text == "Pass" && uuid.is_empty() {
            out.push(None);
        } else {
            let id = id_map.get(uuid).copied().ok_or_else(|| format!("untranslatable-object-id:{kind}:{uuid}"))?;
            out.push(Some(id));
        }
    }
    Ok(out)
}

fn apply_choose_targets(
    state: &mut GameState,
    rec: &DecisionRecord,
    legal_targets: &[Target],
    id_map: &HashMap<String, ObjectId>,
    seat_uuid: &[Option<String>; 2],
) -> Result<(), String> {
    let translate = |uuid: &str| -> Option<Target> {
        if seat_uuid[0].as_deref() == Some(uuid) {
            return Some(Target::Player(PlayerId::P0));
        }
        if seat_uuid[1].as_deref() == Some(uuid) {
            return Some(Target::Player(PlayerId::P1));
        }
        id_map.get(uuid).map(|&id| Target::Object(id))
    };

    let mut kernel_keys: Vec<String> = legal_targets.iter().map(target_key).collect();
    kernel_keys.sort();

    let mut trace_targets: Vec<Target> = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        trace_targets.push(translate(uuid).ok_or_else(|| format!("untranslatable-target:ChooseTargets:{uuid}"))?);
    }
    let mut trace_keys: Vec<String> = trace_targets.iter().map(target_key).collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:ChooseTargets".to_string());
    }

    if rec.chosen_indices.len() != 1 {
        return Err("unexpected-chosen-count:ChooseTargets".to_string());
    }
    let idx = rec.chosen_indices[0] as usize;
    let target = *trace_targets.get(idx).ok_or("chosen-index-out-of-range:ChooseTargets")?;

    engine::step(state, Action::ChooseTarget(target)).map_err(|e| format!("engine-step-error:ChooseTargets:{e}"))
}

fn target_key(t: &Target) -> String {
    match t {
        Target::Player(p) => format!("P{}", p.index()),
        Target::Object(id) => format!("O{}", id.0),
    }
}

fn apply_declare_attackers(state: &mut GameState, rec: &DecisionRecord, eligible: &[ObjectId], id_map: &HashMap<String, ObjectId>) -> Result<(), String> {
    let mut kernel_keys: Vec<String> = eligible.iter().map(|id| format!("O{}", id.0)).collect();
    kernel_keys.push("DONE".to_string());
    kernel_keys.sort();

    let trace_candidates = translate_attacker_candidates(rec, id_map)?;
    let mut trace_keys: Vec<String> = trace_candidates
        .iter()
        .map(|c| match c {
            Some(id) => format!("O{}", id.0),
            None => "DONE".to_string(),
        })
        .collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:DeclareAttackers".to_string());
    }

    // `chosen_indices` is a full permutation over every candidate
    // (including DONE), not a shrinking iterative pick list -- see the
    // module doc. The real applied attacking set is the prefix before the
    // first DONE.
    let attackers = apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareAttackers")?;
    engine::step(state, Action::DeclareAttackers(attackers)).map_err(|e| format!("engine-step-error:DeclareAttackers:{e}"))
}

/// `chosen_indices` for `DECLARE_ATTACKS`/`DECLARE_BLOCKS` is a full
/// permutation of every candidate index (including the `sentinel:DONE`
/// entry, translated to `None` here) -- see the module doc's point 1.
/// The real applied picks are the *prefix* up to (excluding) the first
/// `None`; anything after it is unapplied ranking noise, not a second
/// round of picks.
fn apply_prefix_before_done<T: Copy>(chosen_indices: &[u32], candidates: &[Option<T>], kind: &str) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    for &idx in chosen_indices {
        match candidates.get(idx as usize).ok_or_else(|| format!("chosen-index-out-of-range:{kind}"))? {
            None => break,
            Some(v) => out.push(*v),
        }
    }
    Ok(out)
}

fn translate_attacker_candidates(rec: &DecisionRecord, id_map: &HashMap<String, ObjectId>) -> Result<Vec<Option<ObjectId>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
        } else {
            let id = id_map.get(uuid).copied().ok_or_else(|| format!("untranslatable-object-id:DeclareAttackers:{uuid}"))?;
            out.push(Some(id));
        }
    }
    Ok(out)
}

fn apply_declare_blockers(
    state: &mut GameState,
    rec: &DecisionRecord,
    legal_blockers: &[(ObjectId, Vec<ObjectId>)],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(), String> {
    // Kernel's `legal_blockers` is attacker-major: (attacker, [blockers]).
    // Normalize to blocker-major (blocker, attacker) to match the trace's
    // "blockerUuid->attackerUuid" text convention -- don't compare the two
    // conventions without normalizing first.
    let mut kernel_keys: Vec<String> = Vec::new();
    for (attacker, blockers) in legal_blockers {
        for blocker in blockers {
            kernel_keys.push(format!("{}->{}", blocker.0, attacker.0));
        }
    }
    kernel_keys.push("DONE".to_string());
    kernel_keys.sort();

    let trace_candidates = translate_blocker_candidates(rec, id_map)?;
    let mut trace_keys: Vec<String> = trace_candidates
        .iter()
        .map(|c| match c {
            Some((blocker, attacker)) => format!("{}->{}", blocker.0, attacker.0),
            None => "DONE".to_string(),
        })
        .collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:DeclareBlockers".to_string());
    }

    // Same prefix-before-DONE rule as DeclareAttackers.
    let blocks = apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareBlockers")?;
    engine::step(state, Action::DeclareBlockers(blocks)).map_err(|e| format!("engine-step-error:DeclareBlockers:{e}"))
}

fn translate_blocker_candidates(rec: &DecisionRecord, id_map: &HashMap<String, ObjectId>) -> Result<Vec<Option<(ObjectId, ObjectId)>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
            continue;
        }
        let (blocker_uuid, attacker_uuid) = uuid.split_once("->").ok_or_else(|| format!("malformed-block-pair:DeclareBlockers:{uuid}"))?;
        let blocker = id_map.get(blocker_uuid).copied().ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{blocker_uuid}"))?;
        let attacker = id_map.get(attacker_uuid).copied().ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{attacker_uuid}"))?;
        out.push(Some((blocker, attacker)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `DecisionRecord` from just the fields each test cares
    /// about; every other field uses its `#[serde(default)]` (matching
    /// how sparsely-populated real records -- e.g. mulligan JSON -- look).
    fn decision_record(action_type: &str, candidate_texts: &[&str], candidate_object_ids: &[&str], chosen_indices: &[u32]) -> DecisionRecord {
        let json = format!(
            "{{\"ordinal\":0,\"player\":\"P\",\"action_type\":\"{action_type}\",\"candidate_count\":{},\
             \"candidate_texts\":{},\"candidate_object_ids\":{},\"chosen_indices\":{:?}}}",
            candidate_texts.len(),
            serde_json::to_string(candidate_texts).unwrap(),
            serde_json::to_string(candidate_object_ids).unwrap(),
            chosen_indices
        );
        serde_json::from_str(&json).unwrap()
    }

    // ---- apply_prefix_before_done -----------------------------------

    #[test]
    fn prefix_before_done_stops_at_the_first_none_and_ignores_the_rest() {
        // Mirrors the real corpus shape: chosen_indices is a full
        // permutation of every candidate (attacker + DONE), and DONE can
        // land anywhere in it -- see the module doc's point 1.
        let candidates = vec![Some(10), Some(20), None]; // [attacker_a, attacker_b, DONE]

        // DONE first (index 2): nothing should be applied, regardless of
        // what follows it in the permutation.
        let picked = apply_prefix_before_done(&[2, 0, 1], &candidates, "test").unwrap();
        assert_eq!(picked, Vec::<i32>::new(), "DONE first means no attackers, even though more picks follow it in the array");

        // One real pick before DONE.
        let picked = apply_prefix_before_done(&[0, 2, 1], &candidates, "test").unwrap();
        assert_eq!(picked, vec![10]);

        // DONE last: every real candidate before it is applied.
        let picked = apply_prefix_before_done(&[1, 0, 2], &candidates, "test").unwrap();
        assert_eq!(picked, vec![20, 10]);
    }

    #[test]
    fn prefix_before_done_rejects_an_out_of_range_index() {
        let candidates = vec![Some(1), None];
        let err = apply_prefix_before_done(&[5], &candidates, "DeclareAttackers").unwrap_err();
        assert!(err.contains("chosen-index-out-of-range"));
    }

    // ---- attacker/blocker candidate translation ----------------------

    #[test]
    fn translate_attacker_candidates_maps_the_done_sentinel_to_none() {
        let mut id_map = HashMap::new();
        id_map.insert("attacker-uuid".to_string(), ObjectId(7));
        let rec = decision_record("DECLARE_ATTACKS", &["Guttersnipe", "DONE"], &["attacker-uuid", "sentinel:DONE"], &[0, 1]);

        let translated = translate_attacker_candidates(&rec, &id_map).unwrap();
        assert_eq!(translated, vec![Some(ObjectId(7)), None]);
    }

    #[test]
    fn translate_attacker_candidates_reports_an_untranslatable_uuid() {
        let id_map = HashMap::new(); // empty: nothing translates
        let rec = decision_record("DECLARE_ATTACKS", &["Guttersnipe", "DONE"], &["attacker-uuid", "sentinel:DONE"], &[0, 1]);
        let err = translate_attacker_candidates(&rec, &id_map).unwrap_err();
        assert!(err.contains("untranslatable-object-id:DeclareAttackers"));
    }

    #[test]
    fn translate_blocker_candidates_splits_blocker_attacker_pairs_blocker_major() {
        let mut id_map = HashMap::new();
        id_map.insert("blocker-uuid".to_string(), ObjectId(3));
        id_map.insert("attacker-uuid".to_string(), ObjectId(9));
        let rec = decision_record("DECLARE_BLOCKS", &["Guttersnipe", "DONE"], &["blocker-uuid->attacker-uuid", "sentinel:DONE"], &[0, 1]);

        let translated = translate_blocker_candidates(&rec, &id_map).unwrap();
        assert_eq!(translated, vec![Some((ObjectId(3), ObjectId(9))), None], "must be (blocker, attacker), not (attacker, blocker)");
    }

    #[test]
    fn translate_blocker_candidates_rejects_a_pair_missing_the_arrow() {
        let id_map = HashMap::new();
        let rec = decision_record("DECLARE_BLOCKS", &["Guttersnipe"], &["not-a-pair"], &[0]);
        let err = translate_blocker_candidates(&rec, &id_map).unwrap_err();
        assert!(err.contains("malformed-block-pair"));
    }

    // ---- ACTIVATE_ABILITY_OR_SPELL candidate dedup (candidate_key) ---

    fn two_mountains_in_hand() -> (GameState, ObjectId, ObjectId) {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain, mountain], &[], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        let a = state.draw_card(PlayerId::P0).unwrap();
        let b = state.draw_card(PlayerId::P0).unwrap();
        (state, a, b)
    }

    #[test]
    fn candidate_key_gives_two_untapped_mountains_the_same_land_drop_key() {
        let (state, a, b) = two_mountains_in_hand();
        let land_drops = [a, b];
        let key_a = candidate_key(&state, a, &land_drops, &[], &[], &[]).unwrap();
        let key_b = candidate_key(&state, b, &land_drops, &[], &[], &[]).unwrap();
        assert_eq!(key_a, key_b, "two interchangeable Mountains must dedup to one ACTIVATE_ABILITY_OR_SPELL candidate, matching the reference engine's own display-layer dedup");
        assert_eq!(key_a, "land:Mountain");
    }

    #[test]
    fn candidate_key_is_none_for_an_object_not_in_any_current_bucket() {
        let (state, a, _b) = two_mountains_in_hand();
        // `a` isn't a member of any of these (empty) buckets.
        assert_eq!(candidate_key(&state, a, &[], &[], &[], &[]), None);
    }

    // ---- turn conversion (json global turn -> kernel round) ----------

    #[test]
    fn expected_round_matches_free_text_examples_from_the_real_corpus() {
        // json turn -> "DECISION #N - Turn R (<player> turn)" cross-checks
        // recorded in the increment-4 report: 13<->7 (PlayerRL1), 23<->12
        // (PlayerRL1), 25<->13 (PlayerRL1), 26<->13 (SelfPlay).
        for (json_turn, expected_round) in [(1u32, 1u32), (2, 1), (13, 7), (23, 12), (25, 13), (26, 13)] {
            assert_eq!(json_turn.div_ceil(2), expected_round, "json_turn={json_turn}");
        }
    }
}
