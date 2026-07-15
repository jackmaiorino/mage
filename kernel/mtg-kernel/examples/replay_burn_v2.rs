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
use mtg_kernel::engine::{self, Action, CastMode, Decision, OptionalCostChoice};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SuppressionReason, SurfaceAction, SurfaceDecision};
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};
use mtg_kernel::trigger::PendingTrigger;

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
        Some(manifest) => {
            match mtg_kernel::surface_v2::verify_corpus_provenance(&manifest.java_oracle_commit) {
                Ok(()) => println!(
                    "provenance check: PASS (manifest java_oracle_commit={} == H2_JAVA_ORACLE_COMMIT, corpus={} status={})",
                    manifest.java_oracle_commit, manifest.corpus, manifest.status
                ),
                Err(e) => {
                    eprintln!("PROVENANCE CHECK FAILED -- refusing to replay:\n{e}");
                    std::process::exit(1);
                }
            }
            let skip_compensation = manifest.reference_rules_version >= 2;
            SKIP_V1_EXILED_EVER_COMPENSATION.store(skip_compensation, std::sync::atomic::Ordering::Relaxed);
            println!(
                "reference_rules_version={} -> v1 exiled_ever library-size compensation {}",
                manifest.reference_rules_version,
                if skip_compensation { "SKIPPED (v2 corpus)" } else { "APPLIED (v1/legacy corpus)" }
            );
        }
        None => {
            eprintln!(
                "WARNING: no readable manifest.json at {}; skipping provenance check \
                 (not a hard failure -- but replay results against an unverified corpus carry no provenance guarantee). \
                 reference_rules_version unknown -- defaulting to v1 exiled_ever compensation APPLIED.",
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
    let mut halted_total = 0usize;
    let mut halted_histogram: BTreeMap<String, usize> = BTreeMap::new();
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
    let mut commutativity_total = CommutativityStats::default();

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
        commutativity_total.merge(outcome.commutativity.clone());
        if outcome.reached_game_over {
            replayed_to_end += 1;
            if outcome.winner_matched {
                winner_matched += 1;
            }
        }
        let reason_for_triage = if outcome.reached_game_over {
            if outcome.winner_matched { "COMPLETE:winner-matched".to_string() } else { "COMPLETE:winner-mismatch".to_string() }
        } else if let Some(reason) = &outcome.halted {
            format!("HALTED:{reason}")
        } else {
            outcome.divergence.clone().unwrap_or_else(|| "no-divergence-no-game-over(?)".to_string())
        };
        triage.push((outcome.decisions_consumed, outcome.decisions_total, reason_for_triage, t.source_path.clone()));
        if let Some(reason) = outcome.halted {
            // Classified, not a divergence -- see ReplayOutcome::halted's
            // doc. Quarantined into its own bucket, not the parity
            // denominator (diverged/replayed_to_end).
            halted_total += 1;
            *halted_histogram.entry(reason).or_default() += 1;
        } else if let Some(reason) = outcome.divergence {
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
    println!("halted (classified, NOT diverged): {halted_total}");
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

    println!("\nhalted-reason histogram (classified, not divergences):");
    if halted_histogram.is_empty() {
        println!("  (none)");
    }
    for (reason, n) in &halted_histogram {
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

    if std::env::var("CHECK_TRIGGER_COMMUTATIVITY").is_ok() {
        println!("\n=== trigger-order commutativity audit (CHECK_TRIGGER_COMMUTATIVITY) ===");
        println!("same-controller 2+-trigger groups checked: {}", commutativity_total.groups_checked);
        println!("  commutative:     {}", commutativity_total.commutative);
        println!("  NONCOMMUTATIVE:  {}", commutativity_total.noncommutative);
        if commutativity_total.skipped_too_large > 0 {
            println!("  skipped (group size > 7, factorial-blowup guard): {}", commutativity_total.skipped_too_large);
        }
        if commutativity_total.noncommutative > 0 {
            println!();
            println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            println!("!! NONCOMMUTATIVE TRIGGER ORDERING FOUND -- {} group(s) of {} checked !!", commutativity_total.noncommutative, commutativity_total.groups_checked);
            println!("!! At least one same-controller trigger group's final state DEPENDS on the  !!");
            println!("!! order chosen (603.3b) -- trigger-ordering correctness is no longer a      !!");
            println!("!! someday item; see the detail(s) below.                                    !!");
            println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            for (i, detail) in commutativity_total.noncommutative_details.iter().enumerate() {
                println!("\n  [{}] {detail}", i + 1);
            }
        } else if commutativity_total.groups_checked > 0 {
            println!("\nAll checked same-controller trigger groups are commutative in this corpus.");
        } else {
            println!("\nNo same-controller 2+-trigger groups were ever encountered in this corpus run.");
        }
    }
}

// ==================== corpus provenance ====================

struct CorpusManifest {
    java_oracle_commit: String,
    corpus: String,
    status: String,
    /// `reference_rules_version` (ReferenceRules v2 addendum, Sol #106/#107).
    /// Absent on every pre-v2 manifest (burn_mirror_v1-v5, rally_mirror_v1,
    /// rally_vs_burn_v1, rallymirror_gen1-3) -- absence means `1` (bug-
    /// compatible), per the addendum's own "safe default" rule. `2` means
    /// the Java-side library zone-duplication bug (Sol #106) is fixed:
    /// `rec.library`/`rec.library_size` now shrinks exactly when a real
    /// zone-change effect (impulse-draw exile, mill, search) removes a card,
    /// same instant, no lag -- see `SKIP_V1_EXILED_EVER_COMPENSATION`'s doc
    /// for why this changes `check_state`'s library-size check.
    reference_rules_version: i64,
}

fn load_manifest(root: &Path) -> Option<CorpusManifest> {
    let text = std::fs::read_to_string(root.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(CorpusManifest {
        java_oracle_commit: v.get("java_oracle_commit")?.as_str()?.to_string(),
        corpus: v.get("corpus").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
        status: v.get("status").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
        reference_rules_version: v.get("reference_rules_version").and_then(|x| x.as_i64()).unwrap_or(1),
    })
}

/// Set once in `main()` after reading the corpus's own `manifest.json`
/// (defaults to `false`, i.e. v1 compensation ON, for every call path that
/// never sets it -- unit tests in this file's own `tests` module construct
/// `ReplayCtx` directly without going through `main()`'s manifest gate, and
/// preserving their existing pre-v2 expectations is deliberate, not an
/// oversight).
///
/// `check_state`'s `zone-size-mismatch:library` check compensates for a
/// real, documented v1-era trace-format gap: `rec.library`/`rec.library_size`
/// (the Java trace's own logged field) did not shrink when an impulse-draw
/// effect exiled a card off the library, because the underlying Java bug
/// (Sol #106, `reference_rules_v2_addendum.md`) left that card's recorded
/// zone stuck at `Zone.OUTSIDE`, so `CardImpl.removeFromZone`'s `OUTSIDE`
/// branch silently no-op'd the physical `Library.remove` call the trace's
/// own zone-size logging depends on -- the card was gone from the kernel's
/// count but Java's own logged `library_size` never dropped to match. `ctx.
/// exiled_ever` compensates for exactly that gap by manually adding back
/// what Java's trace failed to subtract.
///
/// Once the Java bug is fixed (`reference_rules_version: 2`), `rec.library`
/// genuinely shrinks the instant a real zone-change effect removes a card --
/// there is no more gap to compensate for. Applying the v1 compensation to a
/// v2 corpus anyway DOUBLE-corrects: it adds back a count Java's own trace
/// already subtracted, producing a `kernel_lib + exiled_ever` sum that is
/// systematically too high by exactly the impulse-exiled count. Confirmed
/// empirically this session: uncompensated `rally_mirror_v2` (skip=true)
/// scores far closer to Burn-grade than compensated (skip=false) on the same
/// corpus -- see `rally/coverage_ledger.md`'s ReferenceRules v2 entry for the
/// exact before/after scoreboard. This is exactly the transition the
/// addendum's own "New manifest.json field" section warned a v2-aware
/// replay driver must make.
static SKIP_V1_EXILED_EVER_COMPENSATION: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
    /// Set instead of `divergence` when the walk hit `Decision::Halted` --
    /// a *classified*, not-a-divergence terminal (see that decision's doc
    /// and this file's `Decision::Halted` match arm): the kernel has
    /// deliberately given up on a resolution it cannot simulate (Chain
    /// Lightning's "may pay {R}{R} to copy" becoming a live choice), rather
    /// than silently guessing. Burn-grade bar treats this the same way as
    /// `ORACLE_UNSTABLE` elsewhere in this campaign's tooling: a proven,
    /// named limitation, quarantined out of the parity denominator instead
    /// of counted as a mismatch.
    halted: Option<String>,
    decisions_consumed: usize,
    decisions_total: usize,
    commutativity: CommutativityStats,
}

// ==================== trigger-order commutativity audit (increment 13) ====================
//
// Required deliverable, independent of the golden-trace scoreboard above:
// whenever 2+ triggered abilities share a controller (`engine::Decision::
// OrderTriggers`, the 603.3b same-controller-order choice -- see
// `trigger::collect_and_process`'s and `engine::drain_pending_triggers_or_
// decide`'s docs for where that group is actually assembled), this replays
// every legal ordering of that specific group on an independent clone and
// checks whether the reference's own always-"pick abilities.get(0)" choice
// (see the `Decision::OrderTriggers` match arm's comment, just above)
// happens to matter for the resulting game state. Gated behind
// `CHECK_TRIGGER_COMMUTATIVITY=1`; entirely a replay-time shadow audit --
// forks clones, discards them, never mutates the real replay's own `state`
// and never changes what action actually gets applied. Not engine
// behavior, so it lives here, not in `src/engine.rs`/`src/trigger.rs`.

#[derive(Default, Clone)]
struct CommutativityStats {
    groups_checked: usize,
    commutative: usize,
    noncommutative: usize,
    noncommutative_details: Vec<String>,
    skipped_too_large: usize,
}

impl CommutativityStats {
    fn merge(&mut self, other: CommutativityStats) {
        self.groups_checked += other.groups_checked;
        self.commutative += other.commutative;
        self.noncommutative += other.noncommutative;
        self.skipped_too_large += other.skipped_too_large;
        self.noncommutative_details.extend(other.noncommutative_details);
    }

    fn record(&mut self, outcome: CommutativityCheck) {
        match outcome {
            CommutativityCheck::Commutative => {
                self.groups_checked += 1;
                self.commutative += 1;
            }
            CommutativityCheck::Noncommutative(detail) => {
                self.groups_checked += 1;
                self.noncommutative += 1;
                self.noncommutative_details.push(detail);
            }
            CommutativityCheck::SkippedTooLarge => {
                self.skipped_too_large += 1;
            }
        }
    }
}

enum CommutativityCheck {
    Commutative,
    Noncommutative(String),
    SkippedTooLarge,
}

/// All permutations of `0..n`, smallest first. `n` is always a same-
/// controller `OrderTriggers` group size in practice (this 16-card pool
/// realistically never exceeds 2-3 simultaneous same-controller triggers),
/// but callers still guard the factorial blowup themselves.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    let mut items: Vec<usize> = (0..n).collect();
    permute_into(&mut items, 0, &mut out);
    out
}

fn permute_into(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute_into(items, k + 1, out);
        items.swap(k, i);
    }
}

/// Advances a *cloned* state forward from an `OrderTriggers` permutation
/// choice to the next quiescent decision boundary: auto-passes through
/// ordinary priority windows (both players declining, letting the stack
/// actually resolve) while the stack is non-empty; auto-resolves any
/// further same-controller trigger groups with the identity permutation
/// (matching the reference's own real behavior -- this audit's scope is
/// the *outer* group already in hand, not recursively branching into
/// cascades); and auto-declines any Madness offer that comes up mid-
/// cascade (`Action::ChooseMadnessCast(false)`) so that every permutation
/// keeps advancing *the same, deterministic way* regardless of how deep
/// the Madness item happens to sit on the stack for that particular
/// ordering -- without this, whichever permutation happens to put a
/// Madness offer on top *before* a sibling trigger underneath it has
/// resolved stops immediately (a real decision, `Decision::
/// ChooseMadnessCast`), while a permutation that resolves the sibling
/// *first* reaches the identical offer one step later, comparing two
/// genuinely different amounts of resolved cascade as if they were the
/// same boundary. (Root-caused empirically against this corpus's own
/// first commutativity run: every apparent `NONCOMMUTATIVE` case that
/// included a Fiery Temper trigger vanished once this was added -- the
/// prior boundary rule stopped at inconsistent cascade depths, not a real
/// game-state difference.) Any other real decision (a genuine target/mode/
/// attack/block choice, or `GameOver`) is treated as the boundary itself --
/// per this file's own module doc, "use your judgment on what a decision
/// boundary cleanly means" -- since anything past that point depends on
/// choices unrelated to this group's ordering.
fn advance_to_quiescent_boundary(state: &mut GameState) -> Decision {
    loop {
        let decision = engine::advance_until_decision(state);
        match &decision {
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => {
                engine::step(state, Action::Pass).expect("Pass is always legal in an offered priority window");
            }
            Decision::OrderTriggers { pending, .. } => {
                let n = pending.len();
                engine::step(state, Action::OrderTriggers((0..n).collect())).expect("identity permutation is always a legal ordering");
            }
            Decision::ChooseMadnessCast { .. } => {
                engine::step(state, Action::ChooseMadnessCast(false)).expect("declining a madness cast is always legal");
            }
            _ => return decision,
        }
    }
}

/// A *canonical*, rules-visible-only snapshot of `state`: life totals plus
/// each zone's contents, keyed only by what a player could actually
/// observe on the board (card name + the handful of per-permanent
/// attributes this pool ever mutates: tapped/summoning-sick/damage/
/// +1+1 counters). Every zone except the stack is sorted (multiset
/// semantics -- *which* physically-identical Guttersnipe is which has no
/// game-rules meaning), the stack is kept in order (resolution order is
/// exactly what this audit is testing).
///
/// Deliberately **not** `GameState::state_hash()` / `derive(Hash)` on the
/// raw struct: those also hash pure bookkeeping that legitimately differs
/// between two permutation walks without the *game* differing at all --
/// `EngineState::priority_round` (bumped by internal `reset_priority`
/// calls, not by anything a player experiences), `mana_ability_
/// activations`, `next_replacement_id`, and especially `event_history`
/// (a full, order-sensitive audit log of every committed event -- two
/// orderings of two damage triggers necessarily commit their events in a
/// different sequence even when the *net* damage total is identical).
/// Root-caused against this corpus's own first commutativity run: every
/// `Guttersnipe`+`Guttersnipe` / `Sneaky Snacker`+`Sneaky Snacker` group
/// this pool can produce "disagreed" under a raw `state_hash()` compare
/// while this function's own printed summary was byte-identical between
/// permutations -- proving the divergence was in bookkeeping/history, not
/// the game.
fn canonical_snapshot(state: &GameState) -> String {
    let describe_permanent = |&id: &ObjectId| {
        let o = state.objects.get(id);
        format!("{}(tapped={},sick={},dmg={},+1/+1={})", o.name, o.tapped, o.summoning_sick, o.damage, o.counters.plus1_plus1)
    };
    let describe_card = |&id: &ObjectId| state.objects.get(id).name.clone();
    let describe_player = |p: PlayerId| {
        let ps = &state.players[p.index()];
        let mut battlefield: Vec<String> = ps.battlefield.iter().map(describe_permanent).collect();
        battlefield.sort();
        let mut graveyard: Vec<String> = ps.graveyard.iter().map(describe_card).collect();
        graveyard.sort();
        let mut hand: Vec<String> = ps.hand.iter().map(describe_card).collect();
        hand.sort();
        let mut library: Vec<String> = ps.library.iter().map(describe_card).collect();
        library.sort();
        format!(
            "life={} has_lost={} battlefield={battlefield:?} graveyard={graveyard:?} hand={hand:?} library_multiset={library:?}",
            ps.life, ps.has_lost,
        )
    };
    let mut exile: Vec<String> = state.exile.iter().map(|&id| format!("{}(owner={:?})", state.objects.get(id).name, state.objects.get(id).owner)).collect();
    exile.sort();
    let stack: Vec<String> = state
        .stack
        .iter()
        .map(|s| format!("{}(controller={:?},madness_offer={})", state.objects.get(s.source).name, s.controller, s.madness_offer))
        .collect();
    format!(
        "turn={} step={:?} active_player={:?} exile={exile:?} stack={stack:?} P0[{}] P1[{}]",
        state.turn,
        state.step,
        state.active_player,
        describe_player(PlayerId::P0),
        describe_player(PlayerId::P1),
    )
}

/// The actual audit -- see this section's module doc. `state` is the game
/// exactly as of the moment `Decision::OrderTriggers` was raised for
/// `pending` (a same-controller group of 2+), before any permutation is
/// applied to the *real* replay.
fn check_trigger_commutativity(state: &GameState, pending: &[PendingTrigger]) -> CommutativityCheck {
    let n = pending.len();
    if n > 7 {
        // 7! = 5040 clones is already an absurd amount of work for a
        // same-controller trigger group in a 16-card pool; if some future
        // corpus ever manages more than this, skip rather than stall the
        // replay run -- informational only, tallied separately from a real
        // commutative/noncommutative verdict.
        return CommutativityCheck::SkippedTooLarge;
    }

    let mut results: Vec<(Vec<usize>, String)> = Vec::with_capacity(permutations(n).len());
    for perm in permutations(n) {
        let mut clone = state.clone();
        engine::step(&mut clone, Action::OrderTriggers(perm.clone())).expect("every generated permutation is a legal ordering of this group");
        advance_to_quiescent_boundary(&mut clone);
        results.push((perm, canonical_snapshot(&clone)));
    }

    let canonical = &results[0].1;
    if results.iter().all(|(_, snap)| snap == canonical) {
        return CommutativityCheck::Commutative;
    }

    let sources: Vec<String> = pending.iter().map(|t| state.objects.get(t.source).name.clone()).collect();
    let mut detail = format!("controller={:?} sources={sources:?} group_size={n} -- permutation outcomes:", pending[0].controller);
    for (perm, snapshot) in &results {
        detail.push_str(&format!("\n    perm={perm:?} {snapshot}"));
    }
    CommutativityCheck::Noncommutative(detail)
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
    /// Every `(ObjectId, owner)` ever seen in `state.exile`, across the
    /// whole replay so far, EXCLUDING flashback cards -- a monotonically-
    /// growing set (`check_state`'s own doc explains why: `rec.library`/
    /// `rec.library_size` never shrinks back down once an impulse-draw card
    /// is later actually *played* out of exile, so "currently in
    /// `state.exile`" alone under-compensates once that happens). Never
    /// removes an entry once inserted, even after the object leaves
    /// `state.exile` for real (played, or its permission window lapses) --
    /// that permanence is the entire point. Populated from raw `state.exile`
    /// membership rather than `state.engine.exile_play_permissions`
    /// specifically because a permission can be granted *and* expire
    /// between two `check_state` polls with no decision for this player in
    /// between, silently under-counting (root-caused: switching this to
    /// permission-sourced tracking regressed rally_mirror_v1 from 2 to 10
    /// library mismatches). Flashback cards (Lava Dart, Burn-side) are
    /// filtered out at the population site: they take the ordinary
    /// graveyard -> stack -> exile path, never impulse-draw's library ->
    /// exile path, and were never "library debt" to begin with.
    exiled_ever: std::collections::HashSet<(ObjectId, PlayerId)>,
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

    if std::env::var("REPLAY_DEBUG").is_ok() {
        eprintln!("=== TRACE {} ===", t.source_path);
    }
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
    let mut ctx = ReplayCtx { id_map, pregame_object_count, seat_uuid, queues: [queue_for(&p0_name), queue_for(&p1_name)], cursors: [0, 0], exiled_ever: std::collections::HashSet::new() };
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
                        check_state(&state, player, rec, &p0_name, &p1_name, &mut ctx)?;
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
                check_state(&state, player, rec, &p0_name, &p1_name, &mut ctx)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_choose_targets(surface, &mut state, rec, &legal_targets, &ctx.id_map, &ctx.seat_uuid)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::Decision(Decision::ChooseCostTargets { player, candidates, .. }) => {
                // Same shape as ChooseTargets above (a real, logged
                // SELECT_TARGETS record, one per pick) -- see
                // `mtg_kernel::engine::Decision::ChooseCostTargets`'s doc:
                // the reference's sacrifice-cost-target picks (Fireblast's
                // alt cost, Lava Dart's flashback cost) are real decisions,
                // not silently auto-solved.
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:ChooseCostTargets".to_string())?;
                debug_verbose(t, &state, player, rec, "ChooseCostTargets");
                if rec.action_type != "SELECT_TARGETS" {
                    return Err(format!("decision-kind-mismatch:ChooseCostTargets-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec, &p0_name, &p1_name, &mut ctx)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_choose_cost_targets(surface, &mut state, rec, &candidates, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::Decision(Decision::DeclareAttackers { player, eligible }) => {
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:DeclareAttackers".to_string())?;
                debug_verbose(t, &state, player, rec, "DeclareAttackers");
                if rec.action_type != "DECLARE_ATTACKS" {
                    return Err(format!("decision-kind-mismatch:DeclareAttackers-vs-{}", rec.action_type));
                }
                check_state(&state, player, rec, &p0_name, &p1_name, &mut ctx)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_declare_attackers(surface, &mut state, rec, &eligible, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::DeclareBlockersForAttacker { attacker, legal_blockers } => {
                let player = state.active_player.opponent();
                apply_declare_blockers_for_attacker(surface, &mut state, t, &mut ctx, outcome, player, attacker, &legal_blockers, &p0_name, &p1_name)?;
            }
            SurfaceDecision::Decision(Decision::Discard { player, choices, .. }) => {
                // The presented `count` is always 1 (one real pick at a
                // time -- see `DiscardReshape`'s doc); the *total* this
                // whole obligation needs comes from `state.engine.
                // pending_discard` directly (see `pending_discard_total`'s
                // doc for why that, not "keep going while `next_decision`
                // still says Discard", is the only correct loop bound).
                let count = HarnessSurfaceV2::pending_discard_total(&state).ok_or("no Discard decision is pending")?;
                apply_discard(surface, &mut state, t, &mut ctx, outcome, player, count, choices, &p0_name, &p1_name)?;
            }
            SurfaceDecision::Decision(Decision::ChooseOptionalCost { player, .. }) => {
                apply_choose_optional_cost(surface, &mut state, &mut ctx, player)?;
            }
            SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, .. }) => {
                // Ground truth, not a guess: `trace::parse_text` now
                // surfaces the reference's Madness `CHOOSE_USE` prompt
                // ("...instead of putting it into your graveyard") as a
                // real `action_type="CHOOSE_USE"` record in this player's
                // queue (see `MADNESS_CHOOSE_USE_MARKER`'s doc), so this
                // is a real, consumed trace record like any other decision
                // now -- not the blind "always attempt" default increment
                // 11 shipped (root-caused wrong this increment against
                // `game_20260713_002156_0015.txt` decision 322: the
                // reference actually said `NO` there, disproving "always
                // attempt matches every golden-trace case audited" -- that
                // claim was only ever checked against 2 games, both
                // incidentally `YES`; the corpus-wide split is close to
                // even, 25 `YES` / 25 `NO`).
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:ChooseMadnessCast".to_string())?;
                if rec.action_type != "CHOOSE_USE" {
                    return Err(format!("decision-kind-mismatch:ChooseMadnessCast-vs-{}", rec.action_type));
                }
                let attempt = rec.chosen_indices.first() == Some(&0); // 0=Yes, 1=No -- see MADNESS_CHOOSE_USE_MARKER's doc
                ctx.advance(player);
                outcome.decisions_consumed += 1;
                surface
                    .apply(&mut state, SurfaceAction::Action(Action::ChooseMadnessCast(attempt)))
                    .map_err(|e| format!("engine-step-error:ChooseMadnessCast:{e}"))?;
            }
            SurfaceDecision::Decision(Decision::ChooseCastMode { player, options, .. }) => {
                apply_choose_cast_mode(surface, &mut state, &mut ctx, player, &options)?;
            }
            SurfaceDecision::Decision(Decision::OrderTriggers { pending, .. }) => {
                // Ground truth, not a guess -- this is a genuinely *silent*
                // default on the reference side, not merely an unlogged
                // one: `ComputerPlayer.chooseTriggeredAbility` (the method
                // every AI in this lineage inherits -- `ComputerPlayerRL`
                // never overrides it; a real override exists in
                // `ComputerPlayerRL.java` but is dead, commented-out code)
                // is `return abilities.get(0);` -- literally "select first
                // trigger all the time", no RL policy call, no scoring, no
                // `logReplayDecision` anywhere on that path (confirmed:
                // `GameImpl.checkTriggered` repeatedly removes whichever
                // ability this always picks and stacks it, so the net
                // effect is simply "place them in original list order,
                // each one on top of the last" -- i.e. the identity
                // permutation into `Action::OrderTriggers`, exactly what
                // this applies). Every trigger this pool's cards can ever
                // put in the same `OrderTriggers` group this increment
                // (2+ Guttersnipes off one cast, 2+ Sneaky Snackers off
                // the same draw) has a fully symmetric, order-independent
                // effect anyway (a fixed damage/return-to-battlefield
                // program with no target choice), so which physical
                // permutation is chosen can never actually diverge game
                // state even if this pool ever grows a genuine tie-break
                // dependency -- but the *decision* itself still needs to
                // not be surfaced as an error, matching the reference's
                // real (silent, no-record) behavior.
                //
                // increment 13's required commutativity audit: gated
                // behind `CHECK_TRIGGER_COMMUTATIVITY=1`, a replay-time-only
                // shadow check that never changes what actually gets
                // applied below -- it snapshots `state` *before* the
                // identity permutation is applied, forks one independent
                // clone per legal permutation of this same-controller
                // group, and checks whether the claim in the comment above
                // ("can never actually diverge game state") actually holds.
                // See `check_trigger_commutativity`'s doc for the full
                // mechanism.
                if std::env::var("CHECK_TRIGGER_COMMUTATIVITY").is_ok() {
                    outcome.commutativity.record(check_trigger_commutativity(&state, &pending));
                }
                surface
                    .apply(&mut state, SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())))
                    .map_err(|e| format!("engine-step-error:OrderTriggers:{e}"))?;
            }
            SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => return Err("unhandled-decision:ChooseSpellMode".to_string()),
            SurfaceDecision::Decision(Decision::ChooseKicker { player, .. }) => {
                // Ground truth, not a guess (unlike ChooseCastMode/
                // ChooseOptionalCost/ChooseMadnessCast): Goblin Bushwhacker's
                // Kicker offer gets a real logged yes/no answer via
                // trace.rs's KICKER_CHOOSE_USE_MARKER parsing (see that
                // const's doc) -- "Pay Kicker {R} ?" is a genuine CHOOSE_USE
                // record like Fiery Temper's Madness offer, just for a
                // Rally-only card.
                let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:ChooseKicker".to_string())?;
                debug_verbose(t, &state, player, rec, "ChooseKicker");
                if rec.action_type != "CHOOSE_USE" {
                    return Err(format!("decision-kind-mismatch:ChooseKicker-vs-{}", rec.action_type));
                }
                let kicked = rec.chosen_indices.first() == Some(&0); // 0=Yes, 1=No -- see KICKER_CHOOSE_USE_MARKER's doc
                ctx.advance(player);
                outcome.decisions_consumed += 1;
                surface
                    .apply(&mut state, SurfaceAction::Action(Action::ChooseKicker(kicked)))
                    .map_err(|e| format!("engine-step-error:ChooseKicker:{e}"))?;
            }
            SurfaceDecision::Decision(Decision::Halted { mechanic, source }) => {
                // Terminal, same class as GameOver -- not a divergence (see
                // ReplayOutcome::halted's doc and engine::Decision::Halted's
                // own doc). The kernel has already proven, at this exact
                // board state, that the unmodeled branch (Chain Lightning's
                // spell-copy continuation) is live rather than vacuous, and
                // stopped rather than guess. Reference-side ground truth for
                // "was this really live" is cross-checkable against the
                // corpus's own "Pay {R}{R} to copy the spell?" CHOOSE_USE
                // lines (deliberately not parsed into a trace record --
                // Halted consumes none -- but real, and greppable for audit).
                let source_name = state.objects.get(source).name.clone();
                outcome.halted = Some(format!("{mechanic:?}:{source_name}"));
                return Ok(());
            }
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
        | SurfaceDecision::Decision(Decision::ChooseCostTargets { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareAttackers { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareBlockers { player, .. })
        | SurfaceDecision::Decision(Decision::Discard { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseSpellMode { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseOptionalCost { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, .. })
        // `OrderTriggers` itself consumes no trace record either (see that
        // arm's doc), same shape as `ChooseMadnessCast`/`ChooseOptionalCost`
        // -- but unlike those two, it was previously grouped under `None`
        // here, which meant `skip_stale_forced_discards` (called only
        // `if let Some(player) = decision_player(...)`) never ran ahead of
        // it. Root-caused against `game_20260713_002205_0028.txt` decision
        // 285: a `SELECT_CARD` trace record for a discard the *kernel*
        // already auto-resolved (`drain_pending_discard_or_decide`'s
        // `choices.len() <= count` shortcut, same as any other forced
        // discard) was left stale at the front of the queue precisely
        // because this decision kind skipped the stale-record check,
        // permanently desyncing the cursor by one real record.
        | SurfaceDecision::Decision(Decision::OrderTriggers { player, .. })
        // `ChooseCastMode` consumes no trace record either (`apply_choose_
        // cast_mode`'s doc) -- same latent stale-forced-discard hazard the
        // `OrderTriggers` comment above already root-caused once; grouped
        // here preemptively rather than waiting for a corpus trace to prove
        // it, since the shape (and the fix) are identical.
        | SurfaceDecision::Decision(Decision::ChooseCastMode { player, .. })
        // Not in this corpus (Goblin Bushwhacker/Kicker is Rally-only), but
        // the same "consumes no trace record" shape as `ChooseCastMode`
        // applies identically -- grouped here for the same reason.
        | SurfaceDecision::Decision(Decision::ChooseKicker { player, .. }) => Some(*player),
        SurfaceDecision::DeclareBlockersForAttacker { .. } => Some(state.active_player.opponent()),
        SurfaceDecision::Decision(Decision::GameOver { .. }) => None,
        // Not in this corpus (Chain Lightning is Rally-only).
        SurfaceDecision::Decision(Decision::Halted { .. }) => None,
    }
}

fn skip_stale_forced_discards(state: &GameState, ctx: &mut ReplayCtx, player: PlayerId, outcome: &mut ReplayOutcome) {
    loop {
        let Some(&rec) = ctx.next(player) else { return };
        if rec.action_type != "SELECT_CARD" || rec.chosen_object_ids.is_empty() {
            return;
        }
        // A discarded Madness card (`CardDef::madness_cost.is_some()`, this
        // pool's only one being Fiery Temper) lands in `Zone::Exile`, not
        // `Zone::Graveyard` -- `engine::apply_discard`'s own 702.83b branch
        // -- so a forced single-candidate discard the kernel auto-resolved
        // (`drain_pending_discard_or_decide`'s `choices.len() <= count`
        // shortcut) is *still* stale-but-unrecognized here if the discarded
        // card happened to have Madness, checking only `Graveyard`. Root-
        // caused against `game_20260713_002212_0039.txt` decision 148/149:
        // Blood Token's activation discards the controller's only card
        // (Fiery Temper) as its cost, the kernel auto-applies it (hand_len
        // == count == 1) and exiles it, but this check's `Graveyard`-only
        // zone test called it *not* already-applied -- leaving the stale
        // `SELECT_CARD` record at the front of the queue, where the
        // *following* real decision (`Decision::ChooseMadnessCast`, Fiery
        // Temper's own Madness offer this exile triggers) then peeked it
        // instead of the real `CHOOSE_USE` record synced right behind it.
        let already_applied = rec.chosen_object_ids.iter().all(|uuid| match ctx.id_map.get(uuid) {
            Some(&id) => matches!(state.objects.get(id).zone, Zone::Graveyard | Zone::Exile),
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
        "  [{kind}] decision_number={} player={} action={} rec_turn={} state_turn={} kernel_hand={} trace_hand={} kernel_lib={} trace_lib={} step={:?} phase={:?} priority_round={}",
        rec.decision_number,
        rec.player,
        rec.action_type,
        rec.turn,
        state.turn,
        ps.hand.len(),
        rec.hand.len(),
        ps.library.len(),
        rec.library.len(),
        state.step,
        rec.phase,
        state.engine.priority_round,
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

/// Learns a fresh (token) object's uuid the first time it's seen in any
/// candidate/chosen list, binding it to the lowest-`ObjectId` currently-
/// unbound post-pregame object -- see this function's original doc for the
/// mechanism.
///
/// Root-caused (Sol #90, increment 11) against
/// `game_20260713_002147_0002.txt` decision 159: a player's own seat uuid
/// (`ctx.seat_uuid`, e.g. the uuid `SELECT_TARGETS` candidate lists use for
/// "PlayerRL1"/"SelfPlay" themselves as `AnyTarget` picks) is *also* a uuid
/// this function had never seen before the first time it appeared -- and
/// this function used to have no way to tell "a player identity" apart
/// from "an object", so it happily "learned" the seat uuid too, binding it
/// to whatever Blood Token object was next in line. That silently
/// stole one token's object slot for a player uuid that was never a token
/// at all, permanently starving the *real* next token of anywhere to
/// bind -- `untranslatable-object-id` many decisions later, far from this
/// root cause. `ctx.seat_uuid` (populated once, up front, by
/// `find_player_uuids`) is the authoritative "is this uuid actually a
/// player, not an object" check; skip here, same as the sentinel `DONE`
/// and empty-string cases already were.
fn learn_token_ids(ctx: &mut ReplayCtx, state: &GameState, rec: &DecisionRecord) {
    for raw in rec.candidate_object_ids.iter().chain(rec.chosen_object_ids.iter()) {
        for uuid in raw.split("->") {
            if uuid.is_empty() || uuid == DONE || ctx.id_map.contains_key(uuid) || ctx.seat_uuid.iter().any(|s| s.as_deref() == Some(uuid)) {
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
fn check_state(state: &GameState, player: PlayerId, rec: &DecisionRecord, p0_name: &str, p1_name: &str, ctx: &mut ReplayCtx) -> Result<(), String> {
    // Track by raw `state.exile` membership (monotonic once filtered --
    // see `ReplayCtx::exiled_ever`'s doc), not `state.engine.
    // exile_play_permissions`: a permission can be granted *and* expire
    // entirely within a run of *silent, auto-resolved* windows (step-gated
    // passes, forced-empty auto-resolutions -- see this file's module doc,
    // point 1) with no real, `check_state`-gated decision for this player
    // anywhere in between, which would silently under-count. `state.exile`
    // itself never removes an object once added, so it doesn't have that
    // gap -- but it needs a correspondingly complete EXCLUDE list, since it
    // now also has to be told apart from every *other* way this shared
    // (Burn+Rally) card pool ever puts something in exile: flashback
    // (`def.flashback.is_some()`, Lava Dart, Burn-side -- a normal
    // graveyard -> stack -> exile path) and Madness discards
    // (`def.madness_cost.is_some()`, Fiery Temper, Burn-side -- 702.83b's
    // "exile instead of graveyard, then maybe cast it" detour, which lands
    // back in the graveyard, not exile, the instant Madness is declined).
    // Root-caused against a Burn-corpus regression check (burn_mirror_v4_
    // run1 dropped from 39/40 clean replays to 9/40 the first time this
    // filter only excluded flashback): every game with at least one Fiery
    // Temper discard picked up a permanent phantom +1 the moment that card
    // transiently touched `state.exile`, even on games where Madness was
    // declined and it moved straight back to the graveyard a moment later.
    for &id in &state.exile {
        let obj = state.objects.get(id);
        let owner = obj.owner;
        let def = &card_def::CARD_DEFS[obj.card_def as usize];
        // Plot (Highway Robbery, Burn-side): exiles face-down, cast free on
        // a later turn -- a normal, expected exile visit with nothing to do
        // with impulse-draw. `plotted_turn.is_some()` is set exactly when
        // `Action::PlotSpell` put it there (`is_plotted_castable_now`'s own
        // doc); same false-positive class as flashback/madness above.
        if def.flashback.is_none() && def.madness_cost.is_none() && obj.plotted_turn.is_none() {
            ctx.exiled_ever.insert((id, owner));
        }
    }
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
    // Rally-only wrinkle: `rec.library`/`rec.library_size` does not shrink
    // when an impulse-draw effect (Reckless Impulse, Experimental
    // Synthesizer, Clockwork Percussionist) exiles cards off the top of the
    // library, even though `ExileTopXMayPlayUntilEffect.apply` (the real
    // Java effect these cards use) genuinely calls `moveCardsToExile` --
    // confirmed empirically against rally_mirror_v1 game_20260714_144529_
    // 0001.txt record_id 31/32 (a real "Cast Reckless Impulse", hand
    // correctly drops 4->3 the very next record, but library stays 50->50
    // across the same turn boundary, well past when the sorcery must have
    // resolved). Not a kernel bug: this is a real, reproducible limit of
    // what this specific trace field reports for this mechanic -- same
    // "documented gap in the trace format" class as the module doc's
    // stack-size/pending-trigger note. Compensated via `ctx.exiled_ever`
    // (monotonic exile-zone membership, minus flashback cards -- see that
    // field's population site and doc for why raw `state.exile` membership,
    // not `state.engine.exile_play_permissions`, is the right monotonic
    // source: a permission can be granted *and* expire between two
    // `check_state` polls with no decision for this player in between,
    // silently under-counting).
    // ReferenceRules v2 (Sol #106/#107): once the Java-side library
    // zone-duplication bug is fixed, `rec.library` shrinks exactly when a
    // real zone-change effect removes a card -- no more compensation needed,
    // and applying the v1 compensation anyway double-corrects (adds back a
    // count Java's own trace already subtracted). See
    // `SKIP_V1_EXILED_EVER_COMPENSATION`'s doc for the full mechanism.
    let skip_compensation = SKIP_V1_EXILED_EVER_COMPENSATION.load(std::sync::atomic::Ordering::Relaxed);
    let player_exiled_ever = if skip_compensation {
        0
    } else {
        ctx.exiled_ever.iter().filter(|&&(_, owner)| owner == player).count()
    };
    if ps.library.len() + player_exiled_ever != rec.library.len() {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "LIBRARY MISMATCH decision_number={} player={} action={} kernel_lib={} kernel_exiled_ever_by_player={} (compensation {}) trace_lib={} kernel_exile_now={:?} trace_hand={:?}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                ps.library.len(),
                player_exiled_ever,
                if skip_compensation { "SKIPPED" } else { "APPLIED" },
                rec.library.len(),
                state.exile.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                rec.hand,
            );
        }
        return Err("zone-size-mismatch:library".to_string());
    }
    if ps.graveyard.len() != rec.graveyard.len() {
        if std::env::var("REPLAY_DEBUG").is_ok() {
            eprintln!(
                "GRAVEYARD MISMATCH decision_number={} player={} action={} kernel_gy={:?} trace_gy={:?}",
                rec.decision_number, rec.player, rec.action_type, ps.graveyard.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(), rec.graveyard,
            );
        }
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
        if std::env::var("REPLAY_TRACE_FILTER").is_ok() {
            eprintln!(
                "PHASE-MISMATCH-DEBUG decision_number={} player={player:?} lands_played_this_turn={} battlefield={:?} hand={:?}",
                rec.decision_number,
                ps.lands_played_this_turn,
                ps.battlefield.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                ps.hand.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
            );
        }
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

/// Mana-ability candidates are an equivalence class by *what mana they add*,
/// not by permanent name: `ComputerPlayerRL`'s own candidate-building
/// (`chooseTarget`'s "genericChoose"-fed option list) collapses multiple
/// untapped sources of the identical mana into one displayed
/// `"{T}: Add {R}."`-shaped option -- confirmed against Burn's own corpus,
/// where this was invisible only because Burn's mana base is Mountain-only
/// (name-keying and mana-keying coincide with exactly one land name).
/// Rally's Great Furnace (an artifact land, same "{T}: Add {R}." ability as
/// Mountain) is the first pool with two *differently-named* sources of the
/// identical mana -- keying by name alone (the prior behavior here) created
/// two kernel-side buckets ("mana:Mountain", "mana:Great Furnace") where the
/// reference only ever offers one, a real, reproducible
/// `candidate-multiset-mismatch:CastSpellOrPass` on every decision where
/// both are untapped. Keyed by `CardDef::produces_mana` (what the ability
/// actually adds), not by name -- so Mountain and Great Furnace, adding the
/// identical `[R]`, collapse into the same bucket.
fn mana_ability_key(state: &GameState, id: ObjectId) -> String {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    format!("mana:{:?}", def.produces_mana)
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
        return Some(mana_ability_key(state, id));
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
        by_key.entry(mana_ability_key(state, id)).or_insert(KernelChoice::ActivateMana(id));
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
    for ((id, text), uuid) in trace_ids.iter().zip(rec.candidate_texts.iter()).zip(rec.candidate_object_ids.iter()) {
        let key = match id {
            None => "pass".to_string(),
            Some(oid) => candidate_key(state, *oid, text, land_drops, castable_spells, mana_abilities, activatable_abilities, plot_actions).ok_or_else(|| {
                if std::env::var("REPLAY_DEBUG").is_ok() {
                    eprintln!(
                        "NOT-IN-BUCKET decision_number={} text={text:?} trace_uuid={uuid} object_id={} object_name={:?} object_zone={:?} object_owner={:?} object_tapped={} exile_perms={:?} kernel_castable={:?} kernel_land={:?} kernel_mana={:?}",
                        rec.decision_number,
                        oid.0,
                        state.objects.get(*oid).name,
                        state.objects.get(*oid).zone,
                        state.objects.get(*oid).owner,
                        state.objects.get(*oid).tapped,
                        state.engine.exile_play_permissions.iter().map(|p| format!("{}({})=holder:{:?},expiry:{:?}", state.objects.get(p.object).name, p.object.0, p.holder, p.expiry)).collect::<Vec<_>>(),
                        castable_spells.iter().map(|&id| format!("{}({})", state.objects.get(id).name, id.0)).collect::<Vec<_>>(),
                        land_drops.iter().map(|&id| format!("{}({})", state.objects.get(id).name, id.0)).collect::<Vec<_>>(),
                        mana_abilities.iter().map(|&id| format!("{}({})", state.objects.get(id).name, id.0)).collect::<Vec<_>>(),
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

/// Answers a `Decision::ChooseCostTargets` window (Fireblast's alt cost,
/// Lava Dart's flashback cost -- see that decision's doc). Same
/// candidate-multiset-then-chosen-index shape as `apply_choose_targets`,
/// simplified: every candidate is always a permanent (`ObjectId`), never a
/// player, so there's no `Target`/`seat_uuid` translation to do.
fn apply_choose_cost_targets(surface: &mut HarnessSurfaceV2, state: &mut GameState, rec: &DecisionRecord, candidates: &[ObjectId], id_map: &HashMap<String, ObjectId>) -> Result<(), String> {
    let mut kernel_keys: Vec<String> = candidates.iter().map(|id| format!("O{}", id.0)).collect();
    kernel_keys.sort();

    let trace_ids = translate_object_candidates(rec, id_map, "ChooseCostTargets")?;
    let mut trace_keys: Vec<String> = Vec::with_capacity(trace_ids.len());
    for id in &trace_ids {
        let oid = id.ok_or("choose-cost-targets-candidate-is-pass")?;
        trace_keys.push(format!("O{}", oid.0));
    }
    let mut sorted_trace_keys = trace_keys.clone();
    sorted_trace_keys.sort();

    if kernel_keys != sorted_trace_keys {
        return Err("candidate-multiset-mismatch:ChooseCostTargets".to_string());
    }

    if rec.chosen_indices.len() != 1 {
        return Err("unexpected-chosen-count:ChooseCostTargets".to_string());
    }
    let idx = rec.chosen_indices[0] as usize;
    let chosen = trace_ids.get(idx).copied().flatten().ok_or("chosen-index-out-of-range:ChooseCostTargets")?;

    surface.apply(state, SurfaceAction::Action(Action::ChooseCostTarget(chosen))).map_err(|e| format!("engine-step-error:ChooseCostTargets:{e}"))
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
    check_state(state, player, rec, p0_name, p1_name, ctx)?;
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

/// Drives `HarnessSurfaceV2`'s discard reshape (`DiscardReshape`'s doc) one
/// real, single-card pick at a time -- the reference's genuine shape for
/// *every* discard (a cost's, an effect's), confirmed against the live
/// cross-engine oracle this round: even a 1-card discard is one real
/// `SELECT_TARGETS` window; a 2-card discard (Faithless Looting) is two, in
/// sequence, the second's candidate pool missing exactly the first's pick.
/// This corpus (v4, recorded before the H2 surface decomposed the shape)
/// still carries that same real per-card `SELECT_TARGETS` sequence -- what
/// this function used to treat as an optional, purely-informational
/// "preview" prefix ahead of one batched terminal pick is, per this
/// increment's fix, the actual ground truth being replayed here.
///
/// The corpus also logs one further, *redundant* record right after: a
/// terminal `SELECT_CARD` summary of the whole batch, with empty
/// `candidate_probs` (not itself a decision -- `ComputerPlayerRL.
/// choose(Cards,TargetCard,...)`'s own post-hoc log of the `target` its
/// caller had already fully resolved via the real per-card picks above,
/// confirmed against `game_20260713_002147_0002.txt` decisions 169-171:
/// two real, model-scored `SELECT_TARGETS` records followed by one
/// zero-probability `SELECT_CARD` record naming the same two cards). This
/// function still consumes it (the cursor must land past it for the next
/// real decision) but no longer applies it as an action -- the real
/// `Discard` already fully landed via the per-card loop.
#[allow(clippy::too_many_arguments)]
fn apply_discard(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    t: &GoldenTrace,
    ctx: &mut ReplayCtx,
    outcome: &mut ReplayOutcome,
    player: PlayerId,
    count: u32,
    mut choices: Vec<ObjectId>,
    p0_name: &str,
    p1_name: &str,
) -> Result<(), String> {
    // Loop exactly `count` times, tracked locally -- NOT "keep going for as
    // long as `next_decision` still returns `Decision::Discard`": a second,
    // wholly unrelated single-card discard (a later turn's own cleanup, a
    // later cost) can immediately follow this one and is *also*
    // `Decision::Discard`-shaped, indistinguishable from "one more pick of
    // this same batch" by decision kind alone. Root-caused this session
    // against `game_20260713_002150_0005.txt` decisions 142-143: a correct
    // 1-card cleanup discard (hand 8 -> 7) was followed, many silent
    // decisions later, by a second, unrelated cleanup discard the very next
    // *logged* Cleanup step happened to also be exactly 8-cards-large --
    // continuing this loop off that peek re-presented its first real pick
    // as if it were this batch's (already-complete) second pick, expecting
    // another `SELECT_TARGETS` where the trace correctly had this batch's
    // own terminal `SELECT_CARD` summary instead.
    let mut chosen_names: Vec<String> = Vec::new();
    for _ in 0..count {
        let &rec = ctx.next(player).ok_or_else(|| "trace-exhausted:Discard".to_string())?;
        debug_verbose(t, state, player, rec, "Discard");
        if rec.action_type != "SELECT_TARGETS" {
            return Err(format!("decision-kind-mismatch:Discard-vs-{}", rec.action_type));
        }
        check_state(state, player, rec, p0_name, p1_name, ctx)?;
        learn_token_ids(ctx, state, rec);

        let mut kernel_names: Vec<&str> = choices.iter().map(|&id| state.objects.get(id).name.as_str()).collect();
        kernel_names.sort_unstable();
        let mut trace_names: Vec<&str> = rec.candidate_texts.iter().map(String::as_str).collect();
        trace_names.sort_unstable();
        if kernel_names != trace_names {
            return Err("candidate-multiset-mismatch:Discard".to_string());
        }

        let &idx = rec.chosen_indices.first().ok_or("unexpected-chosen-count:Discard")?;
        let name = rec.candidate_texts.get(idx as usize).cloned().ok_or("chosen-index-out-of-range:Discard")?;
        // Identity: translate the trace's own `chosen_object_ids` via
        // `id_map` -- same as every other decision kind (target-port hazard
        // checklist: "selected tuple mapped directly from
        // candidate_object_ids/chosen indices, never recovered from text").
        let uuid = rec.chosen_object_ids.first().ok_or("missing-chosen-object-id:Discard")?;
        let chosen_id = ctx.id_map.get(uuid).copied().ok_or_else(|| format!("untranslatable-object-id:Discard:{uuid}"))?;
        if !choices.contains(&chosen_id) {
            return Err("chosen-not-in-kernel-candidates:Discard".to_string());
        }

        ctx.advance(player);
        outcome.decisions_consumed += 1;
        chosen_names.push(name);
        surface.apply(state, SurfaceAction::Action(Action::Discard(vec![chosen_id]))).map_err(|e| format!("engine-step-error:Discard:{e}"))?;
        choices.retain(|&id| id != chosen_id);
    }

    if let Some(&rec) = ctx.next(player) {
        if rec.action_type == "SELECT_CARD" {
            let terminal_names: Vec<String> = rec
                .chosen_indices
                .iter()
                .map(|&idx| rec.candidate_texts.get(idx as usize).cloned().ok_or_else(|| "chosen-index-out-of-range:Discard-terminal-summary".to_string()))
                .collect::<Result<_, _>>()?;
            if terminal_names != chosen_names {
                return Err("discard-terminal-summary-mismatch".to_string());
            }
            ctx.advance(player);
            outcome.decisions_consumed += 1;
        }
    }
    Ok(())
}

/// Guesses which `CastMode` the reference's own policy chose -- same
/// "no ground truth for the offer itself, only for what follows" shape as
/// `ChooseOptionalCost`/`ChooseMadnessCast` (`Decision::ChooseCastMode`
/// itself consumes no trace record), so this can only look ahead at the
/// *next* record's shape. Only ever reached when *both* `CastMode::Normal`
/// and `CastMode::Alternative` are legally payable (`engine.rs`'s own
/// `ChooseCastMode` doc: with only one affordable, the engine never asks at
/// all) -- rare enough that this was the corpus's first real occurrence
/// (`game_20260713_002146_0001.txt`, a Fireblast cast with both its mana
/// cost and its "sacrifice two Mountains" alternative cost affordable).
/// Fireblast's alternative cost lands on the exact same bare
/// `"<name> (you)"`-shaped `SELECT_TARGETS` record `apply_choose_optional_
/// cost`'s sacrifice-land branch already recognizes for the same reason
/// (a land-sacrifice cost payment, picking *which* permanents); `Normal`
/// leaves no such extra interactive record before the spell's own targets
/// (if any) are asked.
fn apply_choose_cast_mode(surface: &mut HarnessSurfaceV2, state: &mut GameState, ctx: &mut ReplayCtx, player: PlayerId, options: &[CastMode]) -> Result<(), String> {
    let looks_like_alternative_cost_pick =
        matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && !rec.candidate_texts.is_empty() && rec.candidate_texts.iter().all(|t| t.ends_with(" (you)")));
    let mode = if looks_like_alternative_cost_pick && options.contains(&CastMode::Alternative) { CastMode::Alternative } else { CastMode::Normal };
    surface.apply(state, SurfaceAction::Action(Action::ChooseCastMode(mode))).map_err(|e| format!("engine-step-error:ChooseCastMode:{e}"))
}

/// Guesses which `OptionalCostChoice` the reference's own policy made --
/// `Decision::ChooseOptionalCost` itself consumes no trace record (same
/// "no ground truth for the offer itself, only for what follows" shape as
/// `ChooseMadnessCast`), so this can only look ahead at the *next* record's
/// shape. Both real payable sub-costs *usually* land on a `SELECT_TARGETS`
/// record (a discard's "which card" question uses the same
/// `SELECT_TARGETS`-then-`SELECT_CARD` prefix pattern `apply_discard`
/// already handles for every other discard cost -- see Masked Meower's
/// ability for the same shape) -- except the one-legal-candidate collapse
/// documented below.
///
/// Told apart primarily by candidate *text shape*, not count: every
/// sacrifice-land candidate in this corpus renders as `"<name> (you)"` (a
/// permanent-you-control reference -- there's more than one same-named land
/// you could mean), while every discard candidate is a bare card name (a
/// hand card is unambiguously yours, so the reference never qualifies it).
/// Candidate *count* against hand size / controlled-land count is kept only
/// as a fallback for when the stronger text-shape signal doesn't apply (an
/// empty/non-`SELECT_TARGETS` next record, i.e. a genuine `Decline`).
/// Root-caused against `game_20260713_002211_0037.txt` decision 131:
/// `hand_len == land_len == 3` by real coincidence, and the previous
/// count-only heuristic checked the discard shape *first*, unconditionally
/// picking `Discard` the instant the counts tied -- even though the next
/// record was three `"Mountain (you)"` land candidates (a `SacrificeLand`
/// pick, per the text-shape rule above). An even earlier version of this
/// function only ever checked the discard shape and fell back to `Decline`
/// otherwise -- silently wrong whenever the real choice was `SacrificeLand`
/// (Sol #90, increment 11).
///
/// **One-legal-candidate collapse** (increment 14): when only one card in
/// hand is a legal discard, the reference's own `TargetCardInHand` chooser
/// auto-selects it with no interactive prompt at all -- no `SELECT_TARGETS`
/// record is logged, just a lone `SELECT_CARD` naming that card directly.
/// Corpus-wide census of every Highway-Robbery-shaped discard pick (the
/// only card in this pool with a payable discard-or-sacrifice optional
/// cost): 13 occurrences with `hand_len >= 2` all log the
/// `SELECT_TARGETS`-then-`SELECT_CARD` pair this function already handled;
/// the corpus's *only* `hand_len == 1` occurrence
/// (`game_20260713_002153_0009.txt` decision 142, hand `["Guttersnipe"]`)
/// instead logs a bare one-candidate `SELECT_CARD` with no
/// `SELECT_TARGETS` at all -- which the `action_type == "SELECT_TARGETS"`-
/// only check above can never match, so this real `Discard` pick was
/// silently misread as `Decline`, desyncing every subsequent decision in
/// that trace. Same text-shape discriminator as the `SELECT_TARGETS` path
/// (bare name = discard, `"... (you)"` = sacrifice-land) applies here too,
/// so a symmetric single-legal-land collapse (no corpus example yet, but
/// the same reference chooser would behave identically) is handled the
/// same way rather than left as a latent gap.
fn apply_choose_optional_cost(surface: &mut HarnessSurfaceV2, state: &mut GameState, ctx: &mut ReplayCtx, player: PlayerId) -> Result<(), String> {
    // Real payable flags, not the H2 reshape's presentation-only sentinel
    // (`(false, false)` at the `Use` stage the caller just observed) -- see
    // `HarnessSurfaceV2::pending_optional_cost_payable`'s doc.
    let (discard_payable, sacrifice_payable) = HarnessSurfaceV2::pending_optional_cost_payable(state).ok_or("no ChooseOptionalCost decision is pending")?;
    let hand_len = state.players[player.index()].hand.len();
    let land_len = state.players[player.index()].battlefield.iter().filter(|&&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land).count();
    let next_is_select_targets_with_len = |n: usize| matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && rec.candidate_texts.len() == n);
    let next_looks_like_land_refs =
        matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && !rec.candidate_texts.is_empty() && rec.candidate_texts.iter().all(|t| t.ends_with(" (you)")));
    let next_is_lone_select_card_shaped = |want_land: bool| {
        matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_CARD" && rec.candidate_texts.len() == 1 && rec.candidate_texts[0].ends_with(" (you)") == want_land)
    };
    let looks_like_sacrifice_pick =
        sacrifice_payable && (next_looks_like_land_refs || (!discard_payable && next_is_select_targets_with_len(land_len)) || (land_len == 1 && next_is_lone_select_card_shaped(true)));
    let looks_like_discard_pick =
        discard_payable && !looks_like_sacrifice_pick && (next_is_select_targets_with_len(hand_len) || (hand_len == 1 && next_is_lone_select_card_shaped(false)));
    let choice = if looks_like_sacrifice_pick {
        OptionalCostChoice::SacrificeLand
    } else if looks_like_discard_pick {
        OptionalCostChoice::Discard
    } else {
        OptionalCostChoice::Decline
    };
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

    fn empty_ctx<'a>() -> ReplayCtx<'a> {
        ReplayCtx {
            id_map: HashMap::new(),
            pregame_object_count: 0,
            seat_uuid: [None, None],
            queues: [Vec::new(), Vec::new()],
            cursors: [0, 0],
            exiled_ever: std::collections::HashSet::new(),
        }
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
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q", &mut empty_ctx()).unwrap_err();
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
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q", &mut empty_ctx()).unwrap_err();
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
        let err = check_state(&state, PlayerId::P0, &rec, "P", "Q", &mut empty_ctx()).unwrap_err();
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
        check_state(&state, PlayerId::P0, &rec, "P", "Q", &mut empty_ctx()).expect("everything agrees");
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002211_0037.txt` decision 131, see
    /// `apply_choose_optional_cost`'s doc): hand size and controlled-land
    /// count tying (both 3 here) must not make the count-only heuristic
    /// misread a `SacrificeLand` pick as `Discard` -- the candidate text
    /// shape (`"Mountain (you)"` x3, a permanent-you-control reference)
    /// must win over the coincidental count match.
    #[test]
    fn choose_optional_cost_prefers_text_shape_over_a_coincidental_count_tie() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let lightning_bolt = card_def::card_id_by_name("Lightning Bolt").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);

        // 3 controlled Mountains (battlefield) and 3 hand cards -- the
        // exact coincidental tie that exposed the bug.
        for _ in 0..3 {
            let id = state.objects.push(mtg_kernel::state::GameObject {
                card_def: mountain,
                name: "Mountain".to_string(),
                owner: PlayerId::P0,
                controller: PlayerId::P0,
                zone: Zone::Battlefield,
                tapped: false,
                summoning_sick: false,
                damage: 0,
                counters: Default::default(),
                attachments: Vec::new(),
                plotted_turn: None,
                zone_change_count: 0,
            });
            state.players[0].battlefield.push(id);
        }
        for _ in 0..3 {
            let id = state.objects.push(mtg_kernel::state::GameObject {
                card_def: lightning_bolt,
                name: "Lightning Bolt".to_string(),
                owner: PlayerId::P0,
                controller: PlayerId::P0,
                zone: Zone::Hand,
                tapped: false,
                summoning_sick: false,
                damage: 0,
                counters: Default::default(),
                attachments: Vec::new(),
                plotted_turn: None,
                zone_change_count: 0,
            });
            state.players[0].hand.push(id);
        }

        let source = state.players[0].battlefield[0];
        state.engine.pending_optional_cost = Some(mtg_kernel::engine::PendingOptionalCost {
            player: PlayerId::P0,
            source,
            discard: 1,
            sacrifice_lands: 1,
            discard_payable: true,
            sacrifice_payable: true,
            then: mtg_kernel::effect::EffectOp::Sequence(vec![]),
            spell_resume: None,
        });

        let rec = decision_record_ex("SELECT_TARGETS", &["Mountain (you)", "Mountain (you)", "Mountain (you)"], &["a", "b", "c"], &[0], ",\"episode\":0");
        let queue = vec![&rec];
        let mut ctx = ReplayCtx { id_map: HashMap::new(), pregame_object_count: 0, seat_uuid: [None, None], queues: [queue, Vec::new()], cursors: [0, 0], exiled_ever: std::collections::HashSet::new() };
        let mut surface = HarnessSurfaceV2::new();

        apply_choose_optional_cost(&mut surface, &mut state, &mut ctx, PlayerId::P0).expect("ChooseOptionalCost must be legal here");

        assert!(state.engine.pending_optional_cost_sacrifice.is_some(), "expected SacrificeLand to be chosen (text shape), not Discard");
        assert!(state.engine.pending_discard.is_none(), "must not have staged a discard for a sacrifice-land pick");
    }

    /// Root-caused against `game_20260713_002153_0009.txt` decision 142:
    /// with exactly one card in hand, the reference's own `TargetCardInHand`
    /// chooser auto-selects it -- no `SELECT_TARGETS` record is logged at
    /// all, just a lone one-candidate `SELECT_CARD` naming that card. The
    /// pre-increment-14 heuristic only ever recognized `SELECT_TARGETS` as
    /// the "real pick" shape, so this real `Discard` was silently misread
    /// as `Decline`.
    #[test]
    fn choose_optional_cost_recognizes_the_lone_candidate_select_card_collapse() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let guttersnipe = card_def::card_id_by_name("Guttersnipe").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);

        let hand_card = state.objects.push(mtg_kernel::state::GameObject {
            card_def: guttersnipe,
            name: "Guttersnipe".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
                zone_change_count: 0,
        });
        state.players[0].hand.push(hand_card);

        let source = state.objects.push(mtg_kernel::state::GameObject {
            card_def: mountain,
            name: "Highway Robbery".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
                zone_change_count: 0,
        });
        state.engine.pending_optional_cost = Some(mtg_kernel::engine::PendingOptionalCost {
            player: PlayerId::P0,
            source,
            discard: 1,
            sacrifice_lands: 1,
            discard_payable: true,
            sacrifice_payable: false, // no lands controlled in this reduced repro, same as the real trace at that point
            then: mtg_kernel::effect::EffectOp::Sequence(vec![]),
            spell_resume: None,
        });

        let rec = decision_record_ex("SELECT_CARD", &["Guttersnipe"], &["a"], &[0], ",\"episode\":0");
        let queue = vec![&rec];
        let mut ctx = ReplayCtx { id_map: HashMap::new(), pregame_object_count: 0, seat_uuid: [None, None], queues: [queue, Vec::new()], cursors: [0, 0], exiled_ever: std::collections::HashSet::new() };
        let mut surface = HarnessSurfaceV2::new();

        apply_choose_optional_cost(&mut surface, &mut state, &mut ctx, PlayerId::P0).expect("ChooseOptionalCost must be legal here");

        assert!(state.engine.pending_discard.is_some(), "expected Discard to be chosen (lone-candidate SELECT_CARD collapse), not Decline");
    }

    /// Regression test for the increment-14 fix (root-caused against
    /// `game_20260713_002146_0001.txt`: a Fireblast cast with *both*
    /// `CastMode::Normal` and `CastMode::Alternative` legally payable --
    /// the corpus's first real `Decision::ChooseCastMode`, previously
    /// entirely unhandled by this driver). Same peek-ahead shape as
    /// `apply_choose_optional_cost`'s sacrifice-land branch: the next
    /// record being a bare `"<name> (you)"`-shaped `SELECT_TARGETS` (which
    /// Card in the deck), a land-sacrifice payment pick) means `Alternative`
    /// was chosen.
    #[test]
    fn choose_cast_mode_picks_alternative_when_the_next_record_is_a_land_reference_pick() {
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        let fireblast = card_def::card_id_by_name("Fireblast").unwrap();
        let mut state = GameState::new_from_libraries(&[mountain], &[mountain], |id| CARD_DEFS[id as usize].name.to_string(), 1);
        let spell = state.objects.push(mtg_kernel::state::GameObject {
            card_def: fireblast,
            name: "Fireblast".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
                zone_change_count: 0,
        });
        state.engine.pending_cast = Some(mtg_kernel::engine::PendingCast {
            spell,
            controller: PlayerId::P0,
            target_spec: mtg_kernel::card_def::TargetSpec::None,
            targets_chosen: Vec::new(),
            is_flashback: false,
            cast_mode: None,
            additional_cost_discarded: None,
            cost_override: None,
            mode_chosen: None,
            origin_zone: Zone::Hand,
            sacrifice_chosen: Vec::new(),
            kicked: Some(false),
        });

        let rec = decision_record_ex("SELECT_TARGETS", &["Mountain (you)", "Mountain (you)"], &["a", "b"], &[0], ",\"episode\":0");
        let queue = vec![&rec];
        let mut ctx = ReplayCtx { id_map: HashMap::new(), pregame_object_count: 0, seat_uuid: [None, None], queues: [queue, Vec::new()], cursors: [0, 0], exiled_ever: std::collections::HashSet::new() };
        let mut surface = HarnessSurfaceV2::new();

        apply_choose_cast_mode(&mut surface, &mut state, &mut ctx, PlayerId::P0, &[CastMode::Normal, CastMode::Alternative]).expect("ChooseCastMode must be legal here");

        assert_eq!(state.engine.pending_cast.as_ref().and_then(|p| p.cast_mode), Some(CastMode::Alternative), "expected Alternative, not Normal");
    }

    /// Regression test for the increment-14 fix (root-caused against
    /// `game_20260713_002212_0039.txt` decisions 148/149): a stale
    /// `SELECT_CARD` for a forced discard the kernel already auto-applied
    /// (`drain_pending_discard_or_decide`'s `choices.len() <= count`
    /// shortcut) must still be recognized when the discarded card has
    /// Madness and therefore landed in `Zone::Exile`, not `Zone::Graveyard`
    /// -- the zone-check this function used before only ever recognized
    /// the graveyard destination.
    #[test]
    fn skip_stale_forced_discards_recognizes_an_exiled_madness_card_as_already_applied() {
        let mut state = GameState::new_from_libraries(&[], &[], |id| format!("card-{id}"), 1);
        let fiery_temper = card_def::card_id_by_name("Fiery Temper").unwrap();
        let obj = state.objects.push(mtg_kernel::state::GameObject {
            card_def: fiery_temper,
            name: "Fiery Temper".to_string(),
            owner: PlayerId::P0,
            controller: PlayerId::P0,
            zone: Zone::Exile, // already auto-discarded via Madness by the kernel
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
                zone_change_count: 0,
        });
        let mut id_map = HashMap::new();
        id_map.insert("uuid-fiery-temper".to_string(), obj);

        let rec = decision_record_ex("SELECT_CARD", &["Fiery Temper"], &["uuid-fiery-temper"], &[0], ",\"episode\":0,\"chosen_object_ids\":[\"uuid-fiery-temper\"]");
        let queue = vec![&rec];
        let mut ctx = ReplayCtx { id_map, pregame_object_count: 0, seat_uuid: [None, None], queues: [queue, Vec::new()], cursors: [0, 0], exiled_ever: std::collections::HashSet::new() };
        let mut outcome = ReplayOutcome::default();

        skip_stale_forced_discards(&state, &mut ctx, PlayerId::P0, &mut outcome);

        assert_eq!(ctx.cursors[0], 1, "the stale SELECT_CARD record for the already-exiled Madness discard must be skipped");
        assert_eq!(outcome.forced_discard_records_skipped, 1);
    }

    // ---- trigger-order commutativity audit -----------------------------

    #[test]
    fn permutations_covers_every_distinct_ordering() {
        let perms = permutations(3);
        assert_eq!(perms.len(), 6);
        let mut sorted = perms.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 6, "all 6 permutations of 0..3 must be distinct, got {perms:?}");
        for p in &perms {
            let mut s = p.clone();
            s.sort_unstable();
            assert_eq!(s, vec![0, 1, 2]);
        }
    }

    fn dummy_object(state: &mut GameState, player: PlayerId, name: &str) -> ObjectId {
        let card_def = card_def::card_id_by_name(name).unwrap_or_else(|| panic!("{name} not in CARD_DEFS"));
        let id = state.objects.push(mtg_kernel::state::GameObject {
            card_def,
            name: name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Battlefield,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            plotted_turn: None,
                zone_change_count: 0,
        });
        state.players[player.index()].battlefield.push(id);
        id
    }

    #[test]
    fn canonical_snapshot_ignores_zone_insertion_order() {
        let mut a = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        dummy_object(&mut a, PlayerId::P0, "Guttersnipe");
        dummy_object(&mut a, PlayerId::P0, "Masked Meower");

        let mut b = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        dummy_object(&mut b, PlayerId::P0, "Masked Meower");
        dummy_object(&mut b, PlayerId::P0, "Guttersnipe");

        assert_eq!(canonical_snapshot(&a), canonical_snapshot(&b), "the same multiset of permanents, pushed in a different order, must snapshot identically");
    }

    #[test]
    fn canonical_snapshot_detects_a_real_difference() {
        let a = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        let mut b = a.clone();
        b.players[0].life -= 1;
        assert_ne!(canonical_snapshot(&a), canonical_snapshot(&b));
    }

    /// This 16-card pool's own real triggers (Guttersnipe/Voldaren Epicure/
    /// Sneaky Snacker) are already asserted commutative by construction
    /// (fixed damage/token/return-to-battlefield programs, no target
    /// choice, no dependence on which sibling resolved first) -- confirmed
    /// empirically by this test using the pool's own real trigger shape,
    /// and by the full corpus run (`CHECK_TRIGGER_COMMUTATIVITY=1`): 15
    /// same-controller 2-to-4-trigger groups, all commutative, zero
    /// noncommutative.
    #[test]
    fn two_guttersnipe_triggers_are_commutative() {
        let mut state = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        let g1 = dummy_object(&mut state, PlayerId::P0, "Guttersnipe");
        let g2 = dummy_object(&mut state, PlayerId::P0, "Guttersnipe");
        state.step = mtg_kernel::state::Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;

        let pending = vec![
            PendingTrigger { controller: PlayerId::P0, source: g1, effect: mtg_kernel::effect::EffectOp::DealDamage { target: mtg_kernel::effect::TargetRef::Opponent, amount: 2 }, is_madness_offer: false, kicked: false },
            PendingTrigger { controller: PlayerId::P0, source: g2, effect: mtg_kernel::effect::EffectOp::DealDamage { target: mtg_kernel::effect::TargetRef::Opponent, amount: 2 }, is_madness_offer: false, kicked: false },
        ];
        // `check_trigger_commutativity` (like its one real call site in
        // `run()`) applies `Action::OrderTriggers` per permutation, which
        // reads from `state.engine.pending_triggers` -- the real driver
        // gets this invariant for free (`pending` there is quite literally
        // `state.engine.pending_triggers[..group_len].to_vec()`), so a
        // synthetic test has to establish it explicitly.
        state.engine.pending_triggers = pending.clone();
        match check_trigger_commutativity(&state, &pending) {
            CommutativityCheck::Commutative => {}
            CommutativityCheck::Noncommutative(detail) => panic!("expected commutative, got: {detail}"),
            CommutativityCheck::SkippedTooLarge => panic!("group of 2 must never be skipped"),
        }
    }

    /// Proves the detector actually *works*, not just "always says yes":
    /// a synthetic same-controller pair (`LoseLife 20` then `GainLife 25`
    /// on the same player, starting at 20 life) is genuinely order-
    /// dependent through the 704.5a state-based loss check -- resolving
    /// the life-loss trigger *first* drops the player to 0 and marks
    /// `has_lost = true` (a real, and in actual play game-ending, SBA)
    /// before the life-gain trigger ever runs; resolving life-gain first
    /// means the player never dips to 0 at all. Both orders land on the
    /// same final life total (25) but disagree on `has_lost` -- exactly
    /// the kind of divergence `canonical_snapshot` (which captures
    /// `has_lost`) must catch and a life-total-only comparison would miss.
    #[test]
    fn a_synthetic_order_dependent_pair_is_correctly_flagged_noncommutative() {
        let mut state = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        let src = dummy_object(&mut state, PlayerId::P0, "Guttersnipe");
        state.step = mtg_kernel::state::Step::Main1;
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;
        assert_eq!(state.players[0].life, 20);

        let pending = vec![
            PendingTrigger { controller: PlayerId::P0, source: src, effect: mtg_kernel::effect::EffectOp::LoseLife { player: mtg_kernel::effect::PlayerRef::Controller, amount: 20 }, is_madness_offer: false, kicked: false },
            PendingTrigger { controller: PlayerId::P0, source: src, effect: mtg_kernel::effect::EffectOp::GainLife { player: mtg_kernel::effect::PlayerRef::Controller, amount: 25 }, is_madness_offer: false, kicked: false },
        ];
        state.engine.pending_triggers = pending.clone();
        match check_trigger_commutativity(&state, &pending) {
            CommutativityCheck::Noncommutative(detail) => {
                assert!(detail.contains("has_lost=true"), "expected one permutation to show the SBA-loss branch, got: {detail}");
                assert!(detail.contains("has_lost=false"), "expected the other permutation to show the no-loss branch, got: {detail}");
            }
            other => panic!("expected Noncommutative (this pair is genuinely order-dependent through 704.5a), got a different verdict: {}", matches!(other, CommutativityCheck::Commutative)),
        }
    }
}
