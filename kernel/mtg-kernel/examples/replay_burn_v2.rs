//! H2 golden-trace replay: drives corpus v4 through
//! `mtg_kernel::surface_v2::HarnessSurfaceV2`, gate-checking each decision
//! against the trace's logged candidates/choice, and reports an honest
//! scoreboard -- same spirit as `examples/replay_burn.rs` (H1/v3, FROZEN),
//! but this is a genuinely separate, independent driver: see
//! `mtg_kernel::surface_v2`'s module doc for the full H1-vs-H2 contract
//! diff. Not one line of `replay_burn.rs` or `mtg_kernel::surface` changes
//! for this increment; the "H1/v3 path... invokable behind a flag" the H2
//! contract asks for is simply running that example directly --
//! `cargo run --release --example replay_burn -- <v3 corpus dir>` -- still
//! fully functional, still frozen, unchanged by this file's existence.
//!
//! Run: cargo run --release --example replay_burn_v2 -- <v4 corpus dir>
//!
//! ## What changed from the H1/v3 driver, and why
//!
//! 1. **`HarnessSurfaceV2` instead of `HarnessSurfaceV1`.** Same
//!    suppression predicate (see `surface_v2`'s doc); different type, so a
//!    corpus mismatch between the two can never silently compile.
//! 2. **`java_reference_target_shortcut` is gone, no replacement.** It
//!    emulated a Java reference *bug* (`chooseTarget`'s `allSameName`
//!    null-name mishandling) that was fixed on the Java side before v4 was
//!    generated -- see `surface_v2`'s doc, point 2. Every `ChooseTargets`
//!    window in v4 is a real, logged `SELECT_TARGETS` record; an
//!    unexplained one is a divergence, not a guess.
//! 3. **Strengthened state gate (`check_state`).** H1's four checks (turn,
//!    hand/library/graveyard zone sizes) are extended with life totals
//!    (`life`/`opp_life`, now parsed by `trace::DecisionRecord`) and a
//!    phase cross-check (kernel `Step` -> XMage phase text, empirically
//!    verified against the full v4 corpus -- see `expected_phase_strings`).
//!    The H2 contract also asked for stack size, pending-trigger count, and
//!    priority player. Priority player *is* checked here, via the
//!    `active_player` trace field (whose *turn* it is) cross-checked
//!    against `GameState::active_player` -- a genuinely new, independent
//!    signal from the existing turn/round check (round is a counter,
//!    active-player is *who* holds it). Stack size and pending-trigger
//!    count are **not checked, because `REPLAY_DECISION_JSON` does not
//!    carry either field anywhere in the v4 corpus** (verified: grepping
//!    every distinct top-level JSON key across all 40 files finds no
//!    `stack_size`/`stack`/`pending_trigger`/`trigger_count` key at all,
//!    on any `action_type`). This is a real, documented limit of what this
//!    trace format can parity-check, not an oversight -- extending the
//!    Java harness's `logReplayDecision` to also emit those two fields
//!    would be a Java change, out of scope for this increment (no Java
//!    modifications). Precisely what's compared, per decision: turn
//!    (round), hand/library/graveyard sizes, both players' life totals,
//!    phase (where the kernel `Step` maps unambiguously), and active
//!    player.
//! 4. **Corpus invariant validation pass**, run once before any replay
//!    begins (`validate_corpus_invariants`): confirms zero phantom records
//!    (matching the manifest's own "verified at lock" claim) and that
//!    every `SELECT_TARGETS` record has a unique semantic key
//!    `(episode_id, target_slot, accumulated_targets)` -- see that
//!    function's doc. Fails the whole run (non-zero exit, no replay
//!    attempted) if either check fails.
//! 5. **Corpus provenance gate**, also run before any replay: reads the
//!    corpus directory's own `manifest.json` and calls
//!    `surface_v2::verify_corpus_provenance` against `H2_JAVA_ORACLE_COMMIT`.
//!    Fails loudly (non-zero exit) on mismatch.
//!
//! Everything else -- candidate/UUID translation, the `DECLARE_ATTACKS`/
//! `DECLARE_BLOCKS` full-permutation-with-DONE-prefix format, the
//! `ACTIVATE_ABILITY_OR_SPELL` candidate-dedup-by-equivalence-class
//! handling, the `SELECT_CARD` name-multiset matching and its
//! `SELECT_TARGETS`-prefix wrinkle, the stale-forced-discard skip, mulligan
//! exclusion -- is unchanged from the H1/v3 driver and not re-derived here;
//! see `examples/replay_burn.rs`'s own module doc for the citations. This
//! file duplicates that machinery (not `#[path]`-shares it) for the same
//! reason `surface_v2` duplicates `HarnessSurfaceV1`'s state machine: H1's
//! driver is frozen and must not gain a runtime v4 code path bolted on.

use mtg_kernel::card_def::{self, CARD_DEFS};
use mtg_kernel::engine::{Action, Decision, OptionalCostChoice};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SuppressionReason, SurfaceAction, SurfaceDecision};
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const DONE: &str = "sentinel:DONE";

fn main() {
    if std::env::var("REPLAY_DEBUG").is_err() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let root = std::env::args().nth(1).map(PathBuf::from).expect("usage: replay_burn_v2 <v4 corpus dir>");

    // --- Gate 1: corpus provenance (see this file's module doc, point 5) ---
    match load_manifest(&root) {
        Some(manifest) => match mtg_kernel::surface_v2::verify_corpus_provenance(&manifest.java_oracle_commit) {
            Ok(()) => println!(
                "provenance check: PASS (manifest java_oracle_commit={} == H2_JAVA_ORACLE_COMMIT, corpus={} status={})",
                manifest.java_oracle_commit, manifest.corpus, manifest.status
            ),
            Err(e) => {
                eprintln!("PROVENANCE CHECK FAILED -- refusing to replay:\n{e}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!(
                "WARNING: no readable manifest.json at {}; skipping provenance check \
                 (not a hard failure -- but replay results against an unverified corpus carry no provenance guarantee).",
                root.display()
            );
        }
    }

    let (traces, errors) = trace::load_corpus(&root);
    println!("traces parsed: {}   parse errors: {}", traces.len(), errors.len());
    for e in errors.iter().take(5) {
        println!("  ERR {e}");
    }

    // --- Gate 2: corpus invariant validation (see this file's module doc, point 4) ---
    let invariants = validate_corpus_invariants(&traces);
    println!("\n--- corpus invariant validation ---");
    println!("phantom (episode<0) records found by this parser: {} (manifest claims 0 at lock)", invariants.phantom_total);
    println!("SELECT_TARGETS records scanned: {}", invariants.select_targets_total);
    println!("SELECT_TARGETS distinct semantic keys: {}", invariants.select_targets_distinct_keys);
    println!("SELECT_TARGETS semantic-key collisions (echo/duplicate-logging candidates): {}", invariants.violations.len());
    if invariants.phantom_total == 0 && invariants.violations.is_empty() {
        println!("corpus invariant validation: PASS");
    } else {
        println!("corpus invariant validation: FAIL -- refusing to replay");
        for v in invariants.violations.iter().take(20) {
            println!("  COLLISION file={} player={} key={} first_ordinal={} second_ordinal={}", v.file, v.player, v.key, v.first_occurrence_ordinal, v.second_occurrence_ordinal);
        }
        std::process::exit(1);
    }

    let mut attempted = 0usize;
    let mut replayed_to_end = 0usize;
    let mut winner_matched = 0usize;
    let mut diverged = 0usize;
    let mut trace_exhausted_pass_total = 0usize;
    let mut silent_window_step_gated_total = 0usize;
    let mut silent_window_no_eligible_attacker_total = 0usize;
    let mut declare_blocks_no_eligible_blockers_total = 0usize;
    let mut combat_priority_action_spent_total = 0usize;
    let mut stack_top_is_casters_own_total = 0usize;
    let mut forced_discard_records_skipped_total = 0usize;
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut phantom_total = 0usize;
    let mut decisions_consumed_total = 0usize;
    let mut decisions_total_total = 0usize;
    let mut per_trace_divergence: Vec<(String, String)> = Vec::new();
    let mut triage: Vec<(usize, usize, String, String)> = Vec::new();

    for t in &traces {
        attempted += 1;
        phantom_total += t.phantom_decisions_skipped;
        let outcome = replay_trace(t);
        trace_exhausted_pass_total += outcome.trace_exhausted_passes;
        silent_window_step_gated_total += outcome.silent_window_step_gated;
        silent_window_no_eligible_attacker_total += outcome.silent_window_no_eligible_attacker;
        declare_blocks_no_eligible_blockers_total += outcome.declare_blocks_no_eligible_blockers;
        combat_priority_action_spent_total += outcome.combat_priority_action_spent;
        stack_top_is_casters_own_total += outcome.stack_top_is_casters_own;
        forced_discard_records_skipped_total += outcome.forced_discard_records_skipped;
        decisions_consumed_total += outcome.decisions_consumed;
        decisions_total_total += outcome.decisions_total;
        if outcome.reached_game_over {
            replayed_to_end += 1;
            if outcome.winner_matched {
                winner_matched += 1;
            }
        }
        let reason_for_triage = if outcome.reached_game_over {
            if outcome.winner_matched { "COMPLETE:winner-matched".to_string() } else { "COMPLETE:winner-mismatch".to_string() }
        } else {
            outcome.divergence.clone().unwrap_or_else(|| "no-divergence-no-game-over(?)".to_string())
        };
        triage.push((outcome.decisions_consumed, outcome.decisions_total, reason_for_triage, t.source_path.clone()));
        if let Some(reason) = outcome.divergence {
            diverged += 1;
            *histogram.entry(reason.clone()).or_default() += 1;
            per_trace_divergence.push((reason, t.source_path.clone()));
        }
    }

    if std::env::var("TRIAGE").is_ok() {
        triage.sort_by(|a, b| b.0.cmp(&a.0));
        println!("\n--- triage (sorted by decisions_consumed desc) ---");
        for (consumed, total, reason, path) in &triage {
            let pct = if *total > 0 { 100.0 * *consumed as f64 / *total as f64 } else { 0.0 };
            println!("  {consumed:>4}/{total:<4} ({pct:>5.1}%)  {reason:<50} {path}");
        }
    }

    println!("\nphantom (episode<0) decision records skipped across corpus: {phantom_total}");
    println!("\n--- H2/v4 scoreboard ---");
    println!("traces attempted:                 {attempted}");
    println!("replayed to end (GameOver seen):  {replayed_to_end}");
    println!("winner matched:                   {winner_matched}");
    println!("diverged:                         {diverged}");
    println!("trace-exhausted-pass occurrences (informational, not a failure): {trace_exhausted_pass_total}");
    println!(
        "silent-window auto-resolutions, step-gated (reference never asks this step, informational): {silent_window_step_gated_total}"
    );
    println!(
        "silent-window auto-resolutions, DeclareAttackers with 0 eligible (declare nobody, informational): {silent_window_no_eligible_attacker_total}"
    );
    println!(
        "silent-window auto-resolutions, DECLARE_BLOCKS attacker had zero eligible blockers (informational): {declare_blocks_no_eligible_blockers_total}"
    );
    println!(
        "stale forced-discard trace records skipped (kernel already auto-applied, informational): {forced_discard_records_skipped_total}"
    );
    println!(
        "silent-window auto-resolutions, DeclareAttackers/DeclareBlockers one-action-per-round already spent (informational): {combat_priority_action_spent_total}"
    );
    println!(
        "silent-window auto-resolutions, same-caster reprompt after their own cast/activation this round (informational): {stack_top_is_casters_own_total}"
    );
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

    if std::env::var("REPLAY_DEBUG").is_ok() {
        println!("\nper-trace divergence (REPLAY_DEBUG):");
        let mut sorted = per_trace_divergence.clone();
        sorted.sort();
        for (reason, path) in &sorted {
            println!("  {reason:<45} {path}");
        }
    }
}

// ==================== corpus provenance ====================

struct CorpusManifest {
    java_oracle_commit: String,
    corpus: String,
    status: String,
}

fn load_manifest(root: &Path) -> Option<CorpusManifest> {
    let text = std::fs::read_to_string(root.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(CorpusManifest {
        java_oracle_commit: v.get("java_oracle_commit")?.as_str()?.to_string(),
        corpus: v.get("corpus").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
        status: v.get("status").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
    })
}

// ==================== corpus invariant validation ====================

struct TargetInvariantViolation {
    file: String,
    player: String,
    key: String,
    first_occurrence_ordinal: u32,
    second_occurrence_ordinal: u32,
}

#[derive(Default)]
struct CorpusInvariantReport {
    phantom_total: usize,
    select_targets_total: usize,
    select_targets_distinct_keys: usize,
    violations: Vec<TargetInvariantViolation>,
}

/// The H2 corpus invariant Sol #88 demands: every logged `SELECT_TARGETS`
/// record must have a unique semantic key `(decision_id, target_slot,
/// accumulated_targets)`, so that one real, semantic target choice can
/// never be double-counted as two trace decisions (a corpus-generation
/// echo/duplicate-logging bug, distinct from the phantom-clone-probe
/// records `trace::GoldenTrace::parse_file` already filters by
/// `episode < 0`).
///
/// `REPLAY_DECISION_JSON` carries no explicit `target_slot`/
/// `accumulated_targets`/`decision_id` field, so this derives them
/// independently from content, per player, walking each player's own
/// chronological decision stream (not the interleaved whole-file stream):
///
/// - **Episode**: a maximal run of consecutive `SELECT_TARGETS` records for
///   one player sharing the same `source_name`, nothing else interleaved
///   (any other real decision for that player breaks the run). This is
///   exactly the shape a single ability's sequential target/discard picks
///   take in this corpus -- e.g. Faithless Looting's 2-card discard cost:
///   two consecutive `source_name="Faithless Looting"` records, nothing
///   else between them (confirmed empirically against the v4 corpus).
/// - `decision_id` = `(player, this episode's first record's
///   `decision_number`)`.
/// - `target_slot` = the position of `rec.decision_number` among the
///   *distinct* `decision_number`s seen so far in this episode, in
///   first-occurrence order -- **not** a raw running count of records. A
///   `decision_number` repeated within the same episode (the corpus-echo
///   shape this validator exists to catch: the same underlying
///   `chooseTarget` call logged twice) maps to the *same* `target_slot` as
///   its first occurrence, rather than silently being treated as "the next
///   pick".
/// - `accumulated_targets` = the ordered `chosen_texts` of every *distinct*
///   `decision_number` strictly before this `target_slot` in the episode
///   (empty at `target_slot == 0`).
///
/// A genuine repeat -- same episode, same `decision_number` as an earlier
/// record in it -- reproduces that earlier record's exact `(decision_id,
/// target_slot, accumulated_targets)` triple (both are `target_slot`
/// `N`, `accumulated_targets` built from the same distinct predecessors),
/// so the two collide and get flagged; a real second pick always has a
/// `decision_number` unseen so far in the episode, so it always lands on a
/// fresh `target_slot`. Deliberately not "trust `decision_number`'s global
/// uniqueness" -- this only ever uses it as a *within-episode* identity
/// test, and independently re-derives the slot/accumulated content from
/// that structure, so the check still means something even if some future
/// corpus's numbering scheme were less well-behaved.
fn validate_corpus_invariants(traces: &[GoldenTrace]) -> CorpusInvariantReport {
    let mut report = CorpusInvariantReport::default();
    for t in traces {
        report.phantom_total += t.phantom_decisions_skipped;

        let mut players: Vec<&str> = Vec::new();
        for d in &t.decisions {
            if !players.contains(&d.player.as_str()) {
                players.push(&d.player);
            }
        }

        for player_name in players {
            let stream: Vec<&DecisionRecord> = t.decisions.iter().filter(|d| d.player == player_name).collect();
            let mut episode_start: Option<u32> = None;
            let mut episode_source: String = String::new();
            // Distinct decision_numbers seen so far in the current episode,
            // first-occurrence order, paired with the chosen text logged
            // the first time each was seen.
            let mut episode_calls: Vec<(u32, String)> = Vec::new();
            let mut seen_keys: HashMap<String, u32> = HashMap::new();

            for rec in &stream {
                if rec.action_type != "SELECT_TARGETS" {
                    episode_start = None;
                    episode_calls.clear();
                    continue;
                }
                report.select_targets_total += 1;

                let same_episode = episode_start.is_some() && episode_source == rec.source_name;
                if !same_episode {
                    episode_start = Some(rec.decision_number);
                    episode_source = rec.source_name.clone();
                    episode_calls.clear();
                }

                let slot = episode_calls.iter().position(|(dn, _)| *dn == rec.decision_number);
                let target_slot = slot.unwrap_or(episode_calls.len());
                let accumulated: Vec<&str> = episode_calls[..target_slot].iter().map(|(_, name)| name.as_str()).collect();
                let key = format!("{player_name}:{}:{target_slot}:[{}]", episode_start.expect("just set above"), accumulated.join("|"));

                match seen_keys.get(&key) {
                    Some(&first_ordinal) => {
                        report.violations.push(TargetInvariantViolation {
                            file: t.source_path.clone(),
                            player: player_name.to_string(),
                            key: key.clone(),
                            first_occurrence_ordinal: first_ordinal,
                            second_occurrence_ordinal: rec.ordinal,
                        });
                    }
                    None => {
                        seen_keys.insert(key, rec.ordinal);
                        report.select_targets_distinct_keys += 1;
                    }
                }

                if slot.is_none() {
                    let chosen_name = rec.chosen_indices.first().and_then(|&i| rec.candidate_texts.get(i as usize)).cloned().unwrap_or_default();
                    episode_calls.push((rec.decision_number, chosen_name));
                }
            }
        }
    }
    report
}

// ==================== replay driver (see module doc for the H1-vs-H2 diff) ====================

#[derive(Default)]
struct ReplayOutcome {
    reached_game_over: bool,
    winner_matched: bool,
    trace_exhausted_passes: usize,
    silent_window_step_gated: usize,
    silent_window_no_eligible_attacker: usize,
    declare_blocks_no_eligible_blockers: usize,
    combat_priority_action_spent: usize,
    stack_top_is_casters_own: usize,
    forced_discard_records_skipped: usize,
    divergence: Option<String>,
    decisions_consumed: usize,
    decisions_total: usize,
}

fn replay_trace(t: &GoldenTrace) -> ReplayOutcome {
    let mut outcome = ReplayOutcome::default();
    let mut surface = HarnessSurfaceV2::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(t, &mut surface, &mut outcome)));
    match result {
        Ok(Err(reason)) => outcome.divergence = Some(reason),
        Ok(Ok(())) => {}
        Err(payload) => {
            let msg = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "<non-string panic payload>".to_string());
            outcome.divergence = Some(format!("engine-panic:{msg}"));
        }
    }
    for s in surface.suppressions() {
        match s.reason {
            SuppressionReason::StepGated => outcome.silent_window_step_gated += 1,
            SuppressionReason::NoRealOption => outcome.trace_exhausted_passes += 1,
            SuppressionReason::NoEligibleAttacker => outcome.silent_window_no_eligible_attacker += 1,
            SuppressionReason::NoEligibleBlockersForAttacker => outcome.declare_blocks_no_eligible_blockers += 1,
            SuppressionReason::CombatPriorityActionSpent => outcome.combat_priority_action_spent += 1,
            SuppressionReason::StackTopIsCastersOwn => outcome.stack_top_is_casters_own += 1,
        }
    }
    outcome
}

struct ReplayCtx<'a> {
    id_map: HashMap<String, ObjectId>,
    pregame_object_count: u32,
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

fn run(t: &GoldenTrace, surface: &mut HarnessSurfaceV2, outcome: &mut ReplayOutcome) -> Result<(), String> {
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
    let pregame_object_count = (lib0.len() + lib1.len()) as u32;
    let seat_uuid = find_player_uuids(t, &p0_name, &p1_name);

    let queue_for = |name: &str| -> Vec<&DecisionRecord> {
        t.decisions.iter().filter(|d| d.player == name && d.action_type != "MULLIGAN" && d.action_type != "LONDON_MULLIGAN").collect()
    };
    let mut ctx = ReplayCtx { id_map, pregame_object_count, seat_uuid, queues: [queue_for(&p0_name), queue_for(&p1_name)], cursors: [0, 0] };
    outcome.decisions_total = ctx.queues[0].len() + ctx.queues[1].len();

    loop {
        let decision = surface.next_decision(&mut state);
        if let Some(player) = decision_player(&decision, &state) {
            skip_stale_forced_discards(&state, &mut ctx, player, outcome);
        }
        match decision {
            SurfaceDecision::Decision(Decision::GameOver { winner }) => {
                outcome.reached_game_over = true;
                let winner_name = winner.map(|p| if p == PlayerId::P0 { p0_name.clone() } else { p1_name.clone() });
                outcome.winner_matched = matches!((&winner_name, &t.winner), (Some(a), Some(b)) if a == b);
                return Ok(());
            }
            SurfaceDecision::Decision(Decision::CastSpellOrPass { player, castable_spells, mana_abilities, land_drops, activatable_abilities, plot_actions }) => {
                match ctx.next(player) {
                    None => return Err("trace-exhausted:CastSpellOrPass-with-real-options".to_string()),
                    Some(&rec) => {
                        debug_verbose(t, &state, player, rec, "CastSpellOrPass");
                        if rec.action_type != "ACTIVATE_ABILITY_OR_SPELL" {
                            if std::env::var("REPLAY_DEBUG").is_ok() {
                                let ps = &state.players[player.index()];
                                eprintln!(
                                    "KIND MISMATCH decision_number={} player={} expected=CastSpellOrPass got={} kernel_castable={:?} kernel_land={:?} kernel_mana={:?} state_step={:?} stack_len={} rec_source={:?} rec_texts={:?} kernel_hand={:?} kernel_gy={:?} kernel_exile={:?}",
                                    rec.decision_number, rec.player, rec.action_type, castable_spells, land_drops, mana_abilities, state.step, state.stack.len(), rec.source_name, rec.candidate_texts,
                                    ps.hand.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                                    ps.graveyard.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                                    state.objects.iter().filter(|(_, o)| o.zone == Zone::Exile && o.owner == player).map(|(_, o)| o.name.clone()).collect::<Vec<_>>(),
                                );
                            }
                            return Err(format!("decision-kind-mismatch:CastSpellOrPass-vs-{}", rec.action_type));
                        }
                        check_state(&state, player, rec, &p0_name, &p1_name)?;
                        learn_token_ids(&mut ctx, &state, rec);
                        apply_cast_spell_or_pass(surface, &mut state, rec, &castable_spells, &mana_abilities, &land_drops, &activatable_abilities, &plot_actions, &ctx.id_map)?;
                        ctx.advance(player);
                        outcome.decisions_consumed += 1;
                    }
                }
            }
            SurfaceDecision::Decision(Decision::ChooseTargets { player, legal_targets, .. }) => {
                // No `java_reference_target_shortcut` here -- see this
                // file's module doc, point 2. Every ChooseTargets window
                // consumes the next real trace record, unconditionally.
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:ChooseTargets".to_string())?;
                debug_verbose(t, &state, player, rec, "ChooseTargets");
                if rec.action_type != "SELECT_TARGETS" {
                    return Err(format!("decision-kind-mismatch:ChooseTargets-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec, &p0_name, &p1_name)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_choose_targets(surface, &mut state, rec, &legal_targets, &ctx.id_map, &ctx.seat_uuid)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::Decision(Decision::DeclareAttackers { player, eligible }) => {
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:DeclareAttackers".to_string())?;
                debug_verbose(t, &state, player, rec, "DeclareAttackers");
                if rec.action_type != "DECLARE_ATTACKS" {
                    return Err(format!("decision-kind-mismatch:DeclareAttackers-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec, &p0_name, &p1_name)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_declare_attackers(surface, &mut state, rec, &eligible, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers } => {
                let player = state.active_player.opponent();
                apply_declare_blockers_for_attacker(surface, &mut state, t, &mut ctx, outcome, player, attacker, &legal_blockers, &p0_name, &p1_name)?;
            }
            SurfaceDecision::Decision(Decision::Discard { player, count, choices }) => {
                apply_discard(surface, &mut state, t, &mut ctx, outcome, player, count, &choices, &p0_name, &p1_name)?;
            }
            SurfaceDecision::Decision(Decision::ChooseOptionalCost { player, discard_payable, sacrifice_payable }) => {
                apply_choose_optional_cost(surface, &mut state, &mut ctx, player, discard_payable, sacrifice_payable)?;
            }
            SurfaceDecision::Decision(Decision::ChooseMadnessCast { .. }) => {
                let mid_cost_payment = state.engine.pending_cast.is_some() || state.engine.pending_activation.is_some();
                surface
                    .apply(&mut state, SurfaceAction::Action(Action::ChooseMadnessCast(!mid_cost_payment)))
                    .map_err(|e| format!("engine-step-error:ChooseMadnessCast:{e}"))?;
            }
            SurfaceDecision::Decision(Decision::ChooseCastMode { .. }) => return Err("unhandled-decision:ChooseCastMode".to_string()),
            SurfaceDecision::Decision(Decision::OrderTriggers { .. }) => return Err("unhandled-decision:OrderTriggers".to_string()),
            SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => return Err("unhandled-decision:ChooseSpellMode".to_string()),
            SurfaceDecision::Decision(Decision::DeclareBlockers { .. }) => {
                return Err("unreachable-decision:DeclareBlockers-should-have-been-reshaped-by-the-surface".to_string());
            }
        }
    }
}

fn decision_player(d: &SurfaceDecision, state: &GameState) -> Option<PlayerId> {
    match d {
        SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseTargets { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareAttackers { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareBlockers { player, .. })
        | SurfaceDecision::Decision(Decision::Discard { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseSpellMode { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseOptionalCost { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, .. }) => Some(*player),
        SurfaceDecision::DeclareBlockersForAttacker { .. } => Some(state.active_player.opponent()),
        SurfaceDecision::Decision(Decision::GameOver { .. })
        | SurfaceDecision::Decision(Decision::ChooseCastMode { .. })
        | SurfaceDecision::Decision(Decision::OrderTriggers { .. }) => None,
    }
}

fn skip_stale_forced_discards(state: &GameState, ctx: &mut ReplayCtx, player: PlayerId, outcome: &mut ReplayOutcome) {
    loop {
        let Some(&rec) = ctx.next(player) else { return };
        if rec.action_type != "SELECT_CARD" || rec.chosen_object_ids.is_empty() {
            return;
        }
        let already_applied = rec.chosen_object_ids.iter().all(|uuid| match ctx.id_map.get(uuid) {
            Some(&id) => state.objects.get(id).zone == Zone::Graveyard,
            None => false,
        });
        if !already_applied {
            return;
        }
        ctx.advance(player);
        outcome.forced_discard_records_skipped += 1;
    }
}

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

fn learn_token_ids(ctx: &mut ReplayCtx, state: &GameState, rec: &DecisionRecord) {
    for raw in rec.candidate_object_ids.iter().chain(rec.chosen_object_ids.iter()) {
        for uuid in raw.split("->") {
            if uuid.is_empty() || uuid == DONE || ctx.id_map.contains_key(uuid) {
                continue;
            }
            let bound: std::collections::HashSet<ObjectId> = ctx.id_map.values().copied().collect();
            let Some(next) = state.objects.iter().map(|(id, _)| id).find(|id| id.0 >= ctx.pregame_object_count && !bound.contains(id)) else {
                continue;
            };
            ctx.id_map.insert(uuid.to_string(), next);
        }
    }
}

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

/// The XMage phase text a given kernel `Step` can appear under in
/// `REPLAY_DECISION_JSON`'s `phase` field, empirically confirmed by
/// cross-tabulating `(action_type, phase)` across every non-phantom record
/// in the full v4 corpus (40 games, 5622 records): `Precombat Main` only
/// ever pairs with `Main1`-shaped decisions, `Postcombat Main` only with
/// `Main2`-shaped ones, `Combat` covers all five kernel steps XMage groups
/// under its own Combat phase (`BeginCombat`/`DeclareAttackers`/
/// `DeclareBlockers`/`CombatDamage`/`EndCombat`), and `End` covers both
/// `End` and `Cleanup` (XMage's End phase; only cleanup-discard-shaped
/// `SELECT_CARD`/its `SELECT_TARGETS` prefix are ever seen there in this
/// corpus). `Untap`/`Upkeep`/`Draw` never produce a real (H2-visible)
/// decision at all -- predicate point 1 (`harness_never_offers_priority`)
/// silently passes through all three -- so there is no empirical mapping
/// to pin for them; returns `&[]` there, and `check_state` skips the phase
/// assertion whenever this returns empty, matching H1's own documented
/// caution against manufacturing false divergences from a hand-built
/// mapping (`examples/replay_burn.rs`'s `check_state` doc).
fn expected_phase_strings(step: mtg_kernel::state::Step) -> &'static [&'static str] {
    use mtg_kernel::state::Step;
    match step {
        Step::Main1 => &["Precombat Main"],
        Step::Main2 => &["Postcombat Main"],
        Step::BeginCombat | Step::DeclareAttackers | Step::DeclareBlockers | Step::CombatDamage | Step::EndCombat => &["Combat"],
        Step::End | Step::Cleanup => &["End"],
        Step::Untap | Step::Upkeep | Step::Draw => &[],
    }
}

/// State comparison at each decision boundary. Strengthened from H1's
/// `check_state` (`examples/replay_burn.rs`) -- see this file's module doc,
/// point 3, for exactly what's new and exactly what the trace format does
/// not expose (stack size, pending-trigger count).
fn check_state(state: &GameState, player: PlayerId, rec: &DecisionRecord, p0_name: &str, p1_name: &str) -> Result<(), String> {
    let ps = &state.players[player.index()];
    let expected_round = rec.turn.div_ceil(2);
    if state.turn != expected_round {
        return Err("turn-mismatch".to_string());
    }
    if ps.hand.len() != rec.hand.len() {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "HAND MISMATCH decision_number={} player={} action={} kernel_hand={} trace_hand={}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                ps.hand.len(),
                rec.hand.len(),
            );
        }
        return Err("zone-size-mismatch:hand".to_string());
    }
    if ps.library.len() != rec.library.len() {
        return Err("zone-size-mismatch:library".to_string());
    }
    if ps.graveyard.len() != rec.graveyard.len() {
        return Err("zone-size-mismatch:graveyard".to_string());
    }

    // NEW vs H1: life totals. `rec.life` is the acting (`player`)'s own
    // life; `rec.opp_life` is the other seat's -- confirmed empirically
    // (always a reciprocal pair against the other seat's own record at the
    // same instant).
    if ps.life != rec.life {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!("LIFE MISMATCH (own) decision_number={} player={} kernel_life={} trace_life={}", rec.decision_number, rec.player, ps.life, rec.life);
        }
        return Err("life-mismatch:own".to_string());
    }
    let opp = &state.players[player.opponent().index()];
    if opp.life != rec.opp_life {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!("LIFE MISMATCH (opp) decision_number={} player={} kernel_opp_life={} trace_opp_life={}", rec.decision_number, rec.player, opp.life, rec.opp_life);
        }
        return Err("life-mismatch:opponent".to_string());
    }

    // NEW vs H1: phase, only where the kernel Step -> XMage phase-text
    // mapping is unambiguous (see `expected_phase_strings`'s doc).
    let expected_phases = expected_phase_strings(state.step);
    if !expected_phases.is_empty() && !rec.phase.is_empty() && !expected_phases.contains(&rec.phase.as_str()) {
        return Err(format!("phase-mismatch:kernel_step={:?}:trace_phase={}", state.step, rec.phase));
    }

    // NEW vs H1: "priority player" -- the closest signal this trace format
    // exposes is `active_player` (whose *turn* it is, not literally who
    // currently holds priority; see this file's module doc, point 3, for
    // why that's the honest scope of this check). `player`/`rec.player`
    // itself is already `state.priority_player` by construction (the
    // kernel builds every `Decision::CastSpellOrPass`/`ChooseTargets`/etc.
    // with `player: state.priority_player`), so asserting that again here
    // would be tautological, not a new signal.
    if !rec.active_player.is_empty() {
        let expected_active = if rec.active_player == p0_name {
            Some(PlayerId::P0)
        } else if rec.active_player == p1_name {
            Some(PlayerId::P1)
        } else {
            None // unparsed name -- diagnostic gap, not a manufactured divergence
        };
        if let Some(expected) = expected_active {
            if state.active_player != expected {
                return Err("active-player-mismatch".to_string());
            }
        }
    }

    Ok(())
}

enum KernelChoice {
    Pass,
    PlayLand(ObjectId),
    CastSpell(ObjectId),
    ActivateMana(ObjectId),
    ActivateAbility(ObjectId, u8),
    PlotSpell(ObjectId),
}

fn cast_zone_tag(state: &GameState, id: ObjectId) -> &'static str {
    match state.objects.get(id).zone {
        Zone::Graveyard => "graveyard",
        Zone::Exile => "exile",
        _ => "hand",
    }
}

#[allow(clippy::too_many_arguments)]
fn candidate_key(
    state: &GameState,
    id: ObjectId,
    text: &str,
    land_drops: &[ObjectId],
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    plot_actions: &[ObjectId],
) -> Option<String> {
    let name = &state.objects.get(id).name;
    if text.starts_with("Plot ") && plot_actions.contains(&id) {
        return Some(format!("plot:{name}"));
    }
    if land_drops.contains(&id) {
        return Some(format!("land:{name}"));
    }
    if castable_spells.contains(&id) {
        return Some(format!("cast:{name}:{}", cast_zone_tag(state, id)));
    }
    if mana_abilities.contains(&id) {
        return Some(format!("mana:{name}"));
    }
    if let Some(&(_, idx)) = activatable_abilities.iter().find(|&&(oid, _)| oid == id) {
        return Some(format!("activate:{name}:{idx}"));
    }
    if plot_actions.contains(&id) {
        return Some(format!("plot:{name}"));
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn apply_cast_spell_or_pass(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    rec: &DecisionRecord,
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    land_drops: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    plot_actions: &[ObjectId],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(), String> {
    let mut by_key: BTreeMap<String, KernelChoice> = BTreeMap::new();
    by_key.insert("pass".to_string(), KernelChoice::Pass);
    for &id in land_drops {
        by_key.entry(format!("land:{}", state.objects.get(id).name)).or_insert(KernelChoice::PlayLand(id));
    }
    for &id in castable_spells {
        by_key.entry(format!("cast:{}:{}", state.objects.get(id).name, cast_zone_tag(state, id))).or_insert(KernelChoice::CastSpell(id));
    }
    for &id in mana_abilities {
        by_key.entry(format!("mana:{}", state.objects.get(id).name)).or_insert(KernelChoice::ActivateMana(id));
    }
    for &(id, idx) in activatable_abilities {
        by_key.entry(format!("activate:{}:{idx}", state.objects.get(id).name)).or_insert(KernelChoice::ActivateAbility(id, idx));
    }
    for &id in plot_actions {
        by_key.entry(format!("plot:{}", state.objects.get(id).name)).or_insert(KernelChoice::PlotSpell(id));
    }
    let mut kernel_keys: Vec<String> = by_key.keys().cloned().collect();
    kernel_keys.sort();

    let trace_ids = translate_object_candidates(rec, id_map, "CastSpellOrPass")?;
    let mut trace_keys = Vec::with_capacity(trace_ids.len());
    for (id, text) in trace_ids.iter().zip(rec.candidate_texts.iter()) {
        let key = match id {
            None => "pass".to_string(),
            Some(oid) => candidate_key(state, *oid, text, land_drops, castable_spells, mana_abilities, activatable_abilities, plot_actions).ok_or_else(|| {
                if std::env::var("REPLAY_DEBUG").is_ok() {
                    eprintln!(
                        "NOT-IN-BUCKET decision_number={} text={text:?} object_name={:?} kernel_castable={:?} kernel_land={:?} kernel_mana={:?}",
                        rec.decision_number,
                        state.objects.get(*oid).name,
                        castable_spells.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                        land_drops.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                        mana_abilities.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                    );
                }
                "trace-candidate-not-in-any-kernel-bucket:CastSpellOrPass".to_string()
            })?,
        };
        trace_keys.push(key);
    }
    let mut sorted_trace_keys = trace_keys.clone();
    sorted_trace_keys.sort();

    if kernel_keys != sorted_trace_keys {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!("MISMATCH decision_number={} texts={:?} kernel={:?} trace={:?}", rec.decision_number, rec.candidate_texts, kernel_keys, sorted_trace_keys);
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
        Some(KernelChoice::PlotSpell(id)) => Action::PlotSpell(*id),
        None => return Err("chosen-not-in-kernel-candidates:CastSpellOrPass".to_string()),
    };
    surface.apply(state, SurfaceAction::Action(action)).map_err(|e| format!("engine-step-error:CastSpellOrPass:{e}"))
}

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

/// Target identity: `candidate_object_ids`/`chosen_indices` only -- `text`
/// is never consulted to determine *which* target was chosen (target-port
/// hazard checklist item: "selected tuple mapped directly from
/// candidate_object_ids/chosen indices, never recovered from text").
fn apply_choose_targets(
    surface: &mut HarnessSurfaceV2,
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

    surface.apply(state, SurfaceAction::Action(Action::ChooseTarget(target))).map_err(|e| format!("engine-step-error:ChooseTargets:{e}"))
}

fn target_key(t: &Target) -> String {
    match t {
        Target::Player(p) => format!("P{}", p.index()),
        Target::Object(id) => format!("O{}", id.0),
    }
}

fn apply_declare_attackers(surface: &mut HarnessSurfaceV2, state: &mut GameState, rec: &DecisionRecord, eligible: &[ObjectId], id_map: &HashMap<String, ObjectId>) -> Result<(), String> {
    let mut kernel_keys: Vec<String> = eligible.iter().map(|id| format!("O{}", id.0)).collect();
    kernel_keys.push("DONE".to_string());
    kernel_keys.sort();

    let trace_candidates = translate_attacker_candidates(rec, id_map)?;
    let mut trace_keys: Vec<String> = trace_candidates.iter().map(|c| match c { Some(id) => format!("O{}", id.0), None => "DONE".to_string() }).collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:DeclareAttackers".to_string());
    }

    let attackers = apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareAttackers")?;
    surface.apply(state, SurfaceAction::Action(Action::DeclareAttackers(attackers))).map_err(|e| format!("engine-step-error:DeclareAttackers:{e}"))
}

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

#[allow(clippy::too_many_arguments)]
fn apply_declare_blockers_for_attacker(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    t: &GoldenTrace,
    ctx: &mut ReplayCtx,
    outcome: &mut ReplayOutcome,
    player: PlayerId,
    attacker: ObjectId,
    legal_blockers: &[ObjectId],
    p0_name: &str,
    p1_name: &str,
) -> Result<(), String> {
    let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:DeclareBlockers".to_string())?;
    debug_verbose(t, state, player, rec, "DeclareBlockers");
    if rec.action_type != "DECLARE_BLOCKS" {
        return Err(format!("decision-kind-mismatch:DeclareBlockers-vs-{}", rec.action_type));
    }
    check_state(state, player, rec, p0_name, p1_name)?;
    learn_token_ids(ctx, state, rec);

    let mut kernel_keys: Vec<String> = legal_blockers.iter().map(|b| format!("{}->{}", b.0, attacker.0)).collect();
    kernel_keys.push("DONE".to_string());
    kernel_keys.sort();

    let trace_candidates = translate_blocker_candidates(rec, &ctx.id_map)?;
    if trace_candidates.iter().any(|c| matches!(c, Some((_, a)) if a != &attacker)) {
        return Err("declare-blocks-attacker-mismatch".to_string());
    }
    let mut trace_keys: Vec<String> = trace_candidates.iter().map(|c| match c { Some((blocker, a)) => format!("{}->{}", blocker.0, a.0), None => "DONE".to_string() }).collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:DeclareBlockers".to_string());
    }

    let picks = apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareBlockers")?;
    let blockers_only: Vec<ObjectId> = picks.into_iter().map(|(blocker, _)| blocker).collect();

    ctx.advance(player);
    outcome.decisions_consumed += 1;
    surface.apply(state, SurfaceAction::DeclareBlockersForAttacker(blockers_only)).map_err(|e| format!("engine-step-error:DeclareBlockers:{e}"))
}

#[allow(clippy::too_many_arguments)]
fn apply_discard(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    t: &GoldenTrace,
    ctx: &mut ReplayCtx,
    outcome: &mut ReplayOutcome,
    player: PlayerId,
    count: u32,
    choices: &[ObjectId],
    p0_name: &str,
    p1_name: &str,
) -> Result<(), String> {
    let mut names_from_targets_prefix: Vec<String> = Vec::new();
    while let Some(&rec) = ctx.next(player) {
        if rec.action_type != "SELECT_TARGETS" {
            break;
        }
        debug_verbose(t, state, player, rec, "Discard(SELECT_TARGETS-prefix)");
        let &idx = rec.chosen_indices.first().ok_or("unexpected-chosen-count:Discard-SELECT_TARGETS-prefix")?;
        let name = rec.candidate_texts.get(idx as usize).ok_or("chosen-index-out-of-range:Discard-SELECT_TARGETS-prefix")?;
        names_from_targets_prefix.push(name.clone());
        ctx.advance(player);
        outcome.decisions_consumed += 1;
    }

    let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:Discard".to_string())?;
    debug_verbose(t, state, player, rec, "Discard");
    if rec.action_type != "SELECT_CARD" {
        return Err(format!("decision-kind-mismatch:Discard-vs-{}", rec.action_type));
    }
    if rec.chosen_indices.len() != count as usize {
        return Err("unexpected-chosen-count:Discard".to_string());
    }
    check_state(state, player, rec, p0_name, p1_name)?;

    let chosen_names: Vec<String> = rec.chosen_indices.iter().map(|&idx| rec.candidate_texts.get(idx as usize).cloned().ok_or_else(|| "chosen-index-out-of-range:Discard".to_string())).collect::<Result<_, _>>()?;
    if !names_from_targets_prefix.is_empty() && names_from_targets_prefix != chosen_names {
        return Err("discard-select-targets-prefix-mismatch".to_string());
    }

    let mut kernel_names: Vec<&str> = choices.iter().map(|&id| state.objects.get(id).name.as_str()).collect();
    kernel_names.sort_unstable();
    let mut trace_names: Vec<&str> = rec.candidate_texts.iter().map(String::as_str).collect();
    trace_names.sort_unstable();
    if kernel_names != trace_names {
        return Err("candidate-multiset-mismatch:Discard".to_string());
    }

    let mut pool: Vec<ObjectId> = choices.to_vec();
    let mut chosen: Vec<ObjectId> = Vec::with_capacity(chosen_names.len());
    for name in &chosen_names {
        let pos = pool.iter().position(|&id| state.objects.get(id).name == *name).ok_or_else(|| format!("chosen-name-not-in-pool:Discard:{name}"))?;
        chosen.push(pool.remove(pos));
    }

    ctx.advance(player);
    outcome.decisions_consumed += 1;
    surface.apply(state, SurfaceAction::Action(Action::Discard(chosen))).map_err(|e| format!("engine-step-error:Discard:{e}"))
}

fn apply_choose_optional_cost(surface: &mut HarnessSurfaceV2, state: &mut GameState, ctx: &mut ReplayCtx, player: PlayerId, discard_payable: bool, _sacrifice_payable: bool) -> Result<(), String> {
    let hand_len = state.players[player.index()].hand.len();
    let looks_like_discard_pick = discard_payable && matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && rec.candidate_texts.len() == hand_len);
    let choice = if looks_like_discard_pick { OptionalCostChoice::Discard } else { OptionalCostChoice::Decline };
    surface.apply(state, SurfaceAction::Action(Action::ChooseOptionalCost(choice))).map_err(|e| format!("engine-step-error:ChooseOptionalCost:{e}"))
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

    fn decision_record_ex(action_type: &str, candidate_texts: &[&str], candidate_object_ids: &[&str], chosen_indices: &[u32], extra: &str) -> DecisionRecord {
        let json = format!(
            "{{\"ordinal\":0,\"player\":\"P\",\"action_type\":\"{action_type}\",\"candidate_count\":{},\
             \"candidate_texts\":{},\"candidate_object_ids\":{},\"chosen_indices\":{:?}{extra}}}",
            candidate_texts.len(),
            serde_json::to_string(candidate_texts).unwrap(),
            serde_json::to_string(candidate_object_ids).unwrap(),
            chosen_indices
        );
        serde_json::from_str(&json).unwrap()
    }

    // ---- corpus invariant validator ----------------------------------

    fn fl_records(decision_number_start: u32) -> Vec<String> {
        // Faithless Looting-shaped: 2 consecutive SELECT_TARGETS records
        // for the same player+source, then a SELECT_CARD -- see the real
        // corpus shape documented in `validate_corpus_invariants`.
        vec![
            format!(
                "REPLAY_DECISION_JSON: {{\"ordinal\":0,\"decision_number\":{},\"player\":\"P\",\"action_type\":\"SELECT_TARGETS\",\"source_name\":\"Faithless Looting\",\
                 \"candidate_count\":2,\"candidate_texts\":[\"Mountain\",\"Bolt\"],\"chosen_indices\":[0],\"episode\":0}}",
                decision_number_start
            ),
            format!(
                "REPLAY_DECISION_JSON: {{\"ordinal\":1,\"decision_number\":{},\"player\":\"P\",\"action_type\":\"SELECT_TARGETS\",\"source_name\":\"Faithless Looting\",\
                 \"candidate_count\":1,\"candidate_texts\":[\"Bolt\"],\"chosen_indices\":[0],\"episode\":0}}",
                decision_number_start + 1
            ),
            format!(
                "REPLAY_DECISION_JSON: {{\"ordinal\":2,\"decision_number\":{},\"player\":\"P\",\"action_type\":\"SELECT_CARD\",\"source_name\":\"Faithless Looting\",\
                 \"candidate_count\":2,\"candidate_texts\":[\"Mountain\",\"Bolt\"],\"chosen_indices\":[0,1],\"episode\":0}}",
                decision_number_start + 2
            ),
        ]
    }

    #[test]
    fn corpus_invariant_validator_accepts_a_real_multi_slot_episode() {
        let text = fl_records(100).join("\n");
        let trace = trace::GoldenTrace::parse_file;
        let _ = trace; // keep import used regardless of parse path below
        let parsed = trace::load_corpus; // silence unused-import warnings if any
        let _ = parsed;
        let t = parse_fixture(&text);
        let report = validate_corpus_invariants(std::slice::from_ref(&t));
        assert_eq!(report.select_targets_total, 2);
        assert_eq!(report.select_targets_distinct_keys, 2, "the 2 sequential Faithless Looting picks have distinct target_slot/accumulated_targets and must not collide");
        assert!(report.violations.is_empty());
    }

    #[test]
    fn corpus_invariant_validator_flags_a_true_echo() {
        // Same record content logged twice back-to-back for the same
        // player/source/slot/accumulated-targets -- an echo/duplicate-log
        // bug, not a second real target pick.
        let mut lines = fl_records(200);
        lines.insert(1, lines[0].clone().replace("\"ordinal\":0", "\"ordinal\":1"));
        let text = lines.join("\n");
        let t = parse_fixture(&text);
        let report = validate_corpus_invariants(std::slice::from_ref(&t));
        assert_eq!(report.violations.len(), 1, "the injected duplicate first pick must be flagged");
    }

    #[test]
    fn corpus_invariant_validator_does_not_merge_two_separate_same_named_casts() {
        // Two independent single-target Lightning Bolt casts for the same
        // player, with an ACTIVATE_ABILITY_OR_SPELL between them (as every
        // real cast requires) -- must be 2 distinct target_slot=0 episodes,
        // not accidentally treated as one 2-slot episode.
        let text = [
            "REPLAY_DECISION_JSON: {\"ordinal\":0,\"decision_number\":10,\"player\":\"P\",\"action_type\":\"SELECT_TARGETS\",\"source_name\":\"Lightning Bolt\",\"candidate_count\":1,\"candidate_texts\":[\"Opp\"],\"chosen_indices\":[0],\"episode\":0}".to_string(),
            "REPLAY_DECISION_JSON: {\"ordinal\":1,\"decision_number\":11,\"player\":\"P\",\"action_type\":\"ACTIVATE_ABILITY_OR_SPELL\",\"candidate_count\":1,\"candidate_texts\":[\"Pass\"],\"chosen_indices\":[0],\"episode\":0}".to_string(),
            "REPLAY_DECISION_JSON: {\"ordinal\":2,\"decision_number\":12,\"player\":\"P\",\"action_type\":\"SELECT_TARGETS\",\"source_name\":\"Lightning Bolt\",\"candidate_count\":1,\"candidate_texts\":[\"Opp\"],\"chosen_indices\":[0],\"episode\":0}".to_string(),
        ]
        .join("\n");
        let t = parse_fixture(&text);
        let report = validate_corpus_invariants(std::slice::from_ref(&t));
        assert_eq!(report.select_targets_total, 2);
        assert_eq!(report.select_targets_distinct_keys, 2, "both casts are target_slot=0 of their own episode, and must not collide");
        assert!(report.violations.is_empty());
    }

    fn parse_fixture(text: &str) -> GoldenTrace {
        // Reuses `trace::parse_text` indirectly via a temp file, since that
        // function is private to `trace.rs`; `GoldenTrace::parse_file`
        // is the smallest public entry point that exercises the same
        // parser.
        let path = std::env::temp_dir().join(format!("replay_burn_v2_fixture_{:?}.txt", std::thread::current().id()));
        std::fs::write(&path, text).unwrap();
        let t = GoldenTrace::parse_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        t
    }

    // ---- expected_phase_strings ---------------------------------------

    #[test]
    fn expected_phase_strings_matches_the_empirical_v4_corpus_mapping() {
        use mtg_kernel::state::Step;
        assert_eq!(expected_phase_strings(Step::Main1), &["Precombat Main"]);
        assert_eq!(expected_phase_strings(Step::Main2), &["Postcombat Main"]);
        for step in [Step::BeginCombat, Step::DeclareAttackers, Step::DeclareBlockers, Step::CombatDamage, Step::EndCombat] {
            assert_eq!(expected_phase_strings(step), &["Combat"], "{step:?}");
        }
        assert_eq!(expected_phase_strings(Step::End), &["End"]);
        assert_eq!(expected_phase_strings(Step::Cleanup), &["End"]);
        assert!(expected_phase_strings(Step::Untap).is_empty());
        assert!(expected_phase_strings(Step::Upkeep).is_empty());
        assert!(expected_phase_strings(Step::Draw).is_empty());
    }

    // ---- translation/prefix helpers (same coverage as the H1 driver) --

    #[test]
    fn prefix_before_done_stops_at_the_first_none_and_ignores_the_rest() {
        let candidates = vec![Some(10), Some(20), None];
        let picked = apply_prefix_before_done(&[2, 0, 1], &candidates, "test").unwrap();
        assert_eq!(picked, Vec::<i32>::new());
        let picked = apply_prefix_before_done(&[0, 2, 1], &candidates, "test").unwrap();
        assert_eq!(picked, vec![10]);
        let picked = apply_prefix_before_done(&[1, 0, 2], &candidates, "test").unwrap();
        assert_eq!(picked, vec![20, 10]);
    }

    #[test]
    fn translate_blocker_candidates_splits_blocker_attacker_pairs_blocker_major() {
        let mut id_map = HashMap::new();
        id_map.insert("blocker-uuid".to_string(), ObjectId(3));
        id_map.insert("attacker-uuid".to_string(), ObjectId(9));
        let rec = decision_record_ex("DECLARE_BLOCKS", &["Guttersnipe", "DONE"], &["blocker-uuid->attacker-uuid", "sentinel:DONE"], &[0, 1], "");
        let translated = translate_blocker_candidates(&rec, &id_map).unwrap();
        assert_eq!(translated, vec![Some((ObjectId(3), ObjectId(9))), None]);
    }

    #[test]
    fn candidate_key_disambiguates_cast_vs_plot_for_the_same_plottable_card() {
        let robbery = card_def::card_id_by_name("Highway Robbery").unwrap();
        let mut state = GameState::new_from_libraries(&[robbery], &[], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        let id = state.draw_card(PlayerId::P0).unwrap();
        let castable_spells = [id];
        let plot_actions = [id];
        let cast_key = candidate_key(&state, id, "Cast Highway Robbery", &[], &castable_spells, &[], &[], &plot_actions).unwrap();
        let plot_key = candidate_key(&state, id, "Plot {1}{R}", &[], &castable_spells, &[], &[], &plot_actions).unwrap();
        assert_ne!(cast_key, plot_key);
    }

    // ---- check_state: the strengthened checks -------------------------

    #[test]
    fn check_state_catches_a_life_mismatch() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        let rec = decision_record_ex(
            "ACTIVATE_ABILITY_OR_SPELL",
            &["Pass"],
            &[""],
            &[0],
            ",\"hand\":[],\"library\":[\"Mountain\"],\"graveyard\":[],\"turn\":1,\"life\":19,\"opp_life\":20",
        );
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q").unwrap_err();
        assert_eq!(err, "life-mismatch:own");
    }

    #[test]
    fn check_state_catches_a_phase_mismatch() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        state.step = mtg_kernel::state::Step::Main1;
        let rec = decision_record_ex(
            "ACTIVATE_ABILITY_OR_SPELL",
            &["Pass"],
            &[""],
            &[0],
            ",\"hand\":[],\"library\":[\"Mountain\"],\"graveyard\":[],\"turn\":1,\"life\":20,\"opp_life\":20,\"phase\":\"Postcombat Main\"",
        );
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q").unwrap_err();
        assert!(err.starts_with("phase-mismatch"), "got {err}");
    }

    #[test]
    fn check_state_catches_an_active_player_mismatch() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        state.active_player = PlayerId::P0;
        let rec = decision_record_ex(
            "ACTIVATE_ABILITY_OR_SPELL",
            &["Pass"],
            &[""],
            &[0],
            ",\"hand\":[],\"library\":[\"Mountain\"],\"graveyard\":[],\"turn\":1,\"life\":20,\"opp_life\":20,\"active_player\":\"Q\"",
        );
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q").unwrap_err();
        assert_eq!(err, "active-player-mismatch");
    }

    #[test]
    fn check_state_passes_a_fully_consistent_record() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        state.active_player = PlayerId::P0;
        state.step = mtg_kernel::state::Step::Main1;
        let rec = decision_record_ex(
            "ACTIVATE_ABILITY_OR_SPELL",
            &["Pass"],
            &[""],
            &[0],
            ",\"hand\":[],\"library\":[\"Mountain\"],\"graveyard\":[],\"turn\":1,\"life\":20,\"opp_life\":20,\"phase\":\"Precombat Main\",\"active_player\":\"P\"",
        );
        check_state(&state, PlayerId::P0, &rec, "P", "Q").expect("everything agrees");
    }
}
