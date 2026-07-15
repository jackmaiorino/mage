//! Golden-trace replay: drives every trace in a corpus through the kernel
//! engine, gate-checking each decision against the trace's logged
//! candidates/choice, and reports an honest scoreboard (not a pass rate to
//! maximize -- partial success this increment is expected).
//!
//! Run: cargo run --release --example replay_burn -- <corpus dir>
//!
//! ## FROZEN (H1/v3, increment 9 -- per Sol #87)
//!
//! This is the final H1 increment against corpus v3
//! (`local-training/kernel_oracle/burn_mirror_v3/`). Acceptance was met:
//! this comparator reaches `GameOver` with a matched winner on 2/40 traces
//! (`game_20260712_194558_0001.txt` 73/73 decisions,
//! `game_20260712_194609_0009.txt` 63/63 decisions), and real-decision
//! consumption across the corpus rose from 29.3% to 31.9%
//! (1289/4046) -- see `mtg_kernel::surface`'s predicate point 4 for the fix
//! (a `HarnessSurfaceV1` gap: the reference hands priority to the *other*
//! player right after a cast/activation completes instead of re-asking the
//! same caster, `ComputerPlayerRL.java:10035-10038`). This driver and its
//! comparator-local wrinkles below are frozen as of this increment against
//! v3: do not extend them further or point them at a different corpus.
//! Further replay work is a new driver against `HarnessSurfaceV2`/corpus
//! v4, scoped to H2. The remaining 38/40 traces' divergences (see the
//! scoreboard's histogram) are open H2 material, not H1 debt.
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
//!
//! ## Decision visibility is now `mtg_kernel::surface::HarnessSurfaceV1`
//!
//! The kernel offers a rules-faithful priority window at every step that
//! grants one (508.1/509.1/117 etc), but the Java reference harness
//! (`ComputerPlayerRL`) only logs a trace record for a strict subset of
//! those windows. Reproducing the reference's own visibility rules --
//! auto-resolving, without consuming a trace record, exactly the windows
//! the reference itself never asked about -- used to live here as
//! comparator-local logic. It now lives in `mtg_kernel::surface`, a
//! versioned library module: the same surface must serve both this
//! comparator and the future model-serving path (so the trained policy
//! sees exactly the decision distribution it was trained on), and
//! comparator-local logic can't be reused there. See that module's doc for
//! the full predicate (priority-window step-gating, the `DeclareBlockers`
//! per-attacker reshape) and its citations into the Java source.
//!
//! What's left here is genuinely comparator-specific: matching each
//! surfaced `SurfaceDecision` against the next trace record, translating
//! candidates between the kernel's `ObjectId`s and the trace's UUIDs, and
//! the golden-trace-format wrinkles below that have no meaning outside a
//! replay (mulligan-phase exclusion, `SELECT_TARGETS`'s single-legal-target
//! shortcut never firing on this pool's only `TargetSpec`, and the stale
//! forced-discard trace-record skip).
//!
//! **`SELECT_TARGETS` (kernel `Decision::ChooseTargets`).** This pool's
//! only `TargetSpec` is `AnyTarget` (both players + creatures), so
//! `legal_targets` always has >= 2 entries (both players, always present,
//! always distinctly named) -- `chooseTarget`'s single-legal-target and
//! same-name-dedup shortcuts (ComputerPlayerRL.java:6706 onward) can never
//! fire for this card pool. No silent-window handling needed here; every
//! kernel `ChooseTargets` has a trace counterpart.
//!
//! **`SELECT_CARD` (kernel `Decision::Discard`).** `logReplayCardSelection`
//! (ComputerPlayerRL.java:7077-7122) logs the *entire* chosen discard set as
//! one record (not one record per card), matching the kernel's own
//! `Decision::Discard`/`Action::Discard(Vec<ObjectId>)` shape 1:1 -- see
//! `apply_discard`. Crucially, `logReplayCardSelection` logs *even when the
//! choice is forced* (its own candidate pool has exactly 1 legal card):
//! unlike `genericChoose`, it doesn't gate on candidate-pool size, only on
//! `target.getTargets()` being non-empty (ComputerPlayerRL.java:7079-7086)
//! -- 9 of the v3 corpus's 324 `SELECT_CARD` records have
//! `candidate_count==1`. The kernel's `drain_pending_discard_or_decide`, by
//! contrast, auto-resolves a forced discard silently (`choices.len() <=
//! count`) with no `Decision::Discard` at all -- so a genuinely-forced
//! discard produces a trace record the kernel never asks about. This driver
//! detects and skips exactly that shape (`skip_stale_forced_discards`):
//! whenever the front of a player's queue is a `SELECT_CARD` record whose
//! every chosen card is already sitting in that player's graveyard (proof
//! the kernel already applied it), it's consumed without a matching
//! `Decision::Discard`. This can't misfire on a genuine not-yet-applied
//! discard: its chosen cards are still in hand at that point, so the
//! graveyard check simply doesn't trigger.
//!
//! A second, easy-to-miss wrinkle discovered empirically (not obvious from
//! the Java source alone): every real discard is *also* preceded by one
//! `SELECT_TARGETS` record per card, from `choose(Outcome, Target, Ability,
//! Game)`'s own internal call to `chooseTarget` (ComputerPlayerRL.java:
//! 7023-7051, "This handles discard effects... etc") -- `chooseTarget`'s
//! per-pick loop is the *same* model-scoring machinery `SELECT_TARGETS`
//! itself uses, so it independently logs its own record for each card
//! before `logReplayCardSelection` logs the aggregate `SELECT_CARD`
//! (confirmed against the v3 corpus: Faithless Looting's 2-card discard
//! produces exactly 2 `SELECT_TARGETS` records with
//! `source_name="Faithless Looting"`, chosen one card each, immediately
//! followed by one `SELECT_CARD` covering both -- same shape for Masked
//! Meower/Blood Token's single-card cost discards). These `SELECT_TARGETS`
//! records have no kernel decision of their own (the kernel's discard model
//! is `Decision::Discard`/`Action::Discard`, never `ChooseTargets`), so
//! `apply_discard` consumes and discards any such prefix before the
//! terminal `SELECT_CARD` record, cross-checking their chosen-name sequence
//! against it as an integrity check.
//!
//! **Mulligan/`LONDON_MULLIGAN`.** Excluded from this driver's replay
//! queues entirely (`queue_for` filters both out) -- only
//! `GoldenTrace::opening_hand_for` (`trace.rs`) consumes them, to seed the
//! opening hand/library.

use mtg_kernel::card_def::{self, CARD_DEFS};
use mtg_kernel::engine::{Action, Decision, OptionalCostChoice};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::surface::{HarnessSurfaceV1, SuppressionReason, SurfaceAction, SurfaceDecision};
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

const DONE: &str = "sentinel:DONE";

fn main() {
    // `replay_trace` catches per-trace engine panics (see its doc) so one
    // buggy trace can't take down the whole corpus scoreboard -- silence the
    // default hook's own stderr dump so a caught panic doesn't look like an
    // uncaught crash in this example's output; `REPLAY_DEBUG=1` restores it
    // (the default hook, plus this driver's own richer per-decision dumps)
    // for whoever's actually chasing the panic down.
    if std::env::var("REPLAY_DEBUG").is_err() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: replay_burn <corpus dir>");
    let (traces, errors) = trace::load_corpus(&root);
    println!(
        "traces parsed: {}   parse errors: {}",
        traces.len(),
        errors.len()
    );
    for e in errors.iter().take(5) {
        println!("  ERR {e}");
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
    let mut java_target_shortcut_applied_total = 0usize;
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut phantom_total = 0usize;
    let mut decisions_consumed_total = 0usize;
    let mut decisions_total_total = 0usize;
    // (reason, source_path) per diverged trace, for `REPLAY_DEBUG`'s
    // per-trace listing below -- lets a divergence be traced back to its
    // exact file without re-running with `REPLAY_TRACE_FILTER` per
    // candidate first.
    let mut per_trace_divergence: Vec<(String, String)> = Vec::new();
    // Increment-8 triage: (decisions_consumed, decisions_total, reason,
    // source_path) per trace, gated behind `TRIAGE` -- picks out the single
    // most-tractable trace (deepest first divergence) to drill first. See
    // the increment-8 report for the selection this produced.
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
        java_target_shortcut_applied_total += outcome.java_target_shortcut_applied;
        decisions_consumed_total += outcome.decisions_consumed;
        decisions_total_total += outcome.decisions_total;
        if outcome.reached_game_over {
            replayed_to_end += 1;
            if outcome.winner_matched {
                winner_matched += 1;
            }
        }
        let reason_for_triage = if outcome.reached_game_over {
            if outcome.winner_matched {
                "COMPLETE:winner-matched".to_string()
            } else {
                "COMPLETE:winner-mismatch".to_string()
            }
        } else {
            outcome
                .divergence
                .clone()
                .unwrap_or_else(|| "no-divergence-no-game-over(?)".to_string())
        };
        triage.push((
            outcome.decisions_consumed,
            outcome.decisions_total,
            reason_for_triage,
            t.source_path.clone(),
        ));
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
            let pct = if *total > 0 {
                100.0 * *consumed as f64 / *total as f64
            } else {
                0.0
            };
            println!("  {consumed:>4}/{total:<4} ({pct:>5.1}%)  {reason:<50} {path}");
        }
    }

    println!("\nphantom (episode<0) decision records skipped across corpus: {phantom_total}");
    println!("\n--- scoreboard ---");
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
        "ChooseTargets windows auto-resolved via the Java allSameName-dedup bug (informational): {java_target_shortcut_applied_total}"
    );
    println!(
        "silent-window auto-resolutions, DeclareAttackers/DeclareBlockers one-action-per-round already spent (informational): {combat_priority_action_spent_total}"
    );
    println!(
        "silent-window auto-resolutions, same-caster reprompt after their own cast/activation this round (informational): {stack_top_is_casters_own_total}"
    );
    // A softer signal than the binary reached/diverged split: how much of
    // each trace's real decision stream validated cleanly before either
    // GameOver or the first divergence.
    let pct = if decisions_total_total > 0 {
        100.0 * decisions_consumed_total as f64 / decisions_total_total as f64
    } else {
        0.0
    };
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
    /// Priority windows auto-passed because the *step itself* is one the
    /// reference harness never asks about (`priorityPlay`'s hard-coded
    /// pass-only steps), regardless of whether the kernel had a real
    /// option -- `mtg_kernel::surface::SuppressionReason::StepGated`. A
    /// different category from `trace_exhausted_passes` (which fires
    /// inside an allowed step when every kernel candidate bucket happens
    /// to be empty, `SuppressionReason::NoRealOption`).
    silent_window_step_gated: usize,
    /// `Decision::DeclareAttackers` auto-resolved (declare nobody) because
    /// `eligible` was empty -- the kernel's own `Step::DeclareAttackers` is
    /// never skipped (508.1), but the Java reference's `selectAttackers`
    /// still doesn't log when `possibleAttackers.isEmpty()` --
    /// `SuppressionReason::NoEligibleAttacker`.
    silent_window_no_eligible_attacker: usize,
    /// `DECLARE_BLOCKS` sub-decisions silently skipped because that
    /// specific attacker has zero eligible blockers (the Java reference's
    /// per-attacker loop `continue`s with no log there) --
    /// `SuppressionReason::NoEligibleBlockersForAttacker`.
    declare_blocks_no_eligible_blockers: usize,
    /// DeclareAttackers/DeclareBlockers priority windows force-passed
    /// because this player already spent their one action this round --
    /// `SuppressionReason::CombatPriorityActionSpent` (predicate point 3).
    combat_priority_action_spent: usize,
    /// `CastSpellOrPass` windows force-passed because the same caster's own
    /// cast/activation just left an unresolved item on top of the stack
    /// this round -- `SuppressionReason::StackTopIsCastersOwn` (predicate
    /// point 4).
    stack_top_is_casters_own: usize,
    /// `SELECT_CARD` trace records skipped because the kernel already
    /// silently auto-applied that exact forced discard before any
    /// `Decision::Discard` was ever offered -- see
    /// `skip_stale_forced_discards`. Comparator-specific (not part of
    /// `HarnessSurfaceV1`): the kernel never asked a `Decision` here at
    /// all, so there's nothing for the surface to suppress -- this is a
    /// stale trace record being reconciled, not a hidden decision.
    forced_discard_records_skipped: usize,
    /// `ChooseTargets` windows silently auto-resolved via
    /// `java_reference_target_shortcut` -- see that function's doc.
    /// Comparator-specific for the same reason `forced_discard_records_skipped`
    /// is: the tie-break needs this trace's own player display-name strings.
    java_target_shortcut_applied: usize,
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
    let mut surface = HarnessSurfaceV1::new();
    // A genuine engine bug on one trace (a `.expect`/`panic!` inside
    // `mtg_kernel::engine`, not a driver-level `Result::Err`) must not take
    // down the whole corpus scoreboard -- every other trace's result is
    // still useful triage signal. `outcome`'s own progress fields (decisions
    // consumed so far, suppressions recorded so far) are lost on unwind
    // (they live on the stack inside `run`'s locals, not `outcome` itself,
    // until it returns) -- acceptable: a panic is already the loudest,
    // least-ambiguous signal a trace can produce, and `REPLAY_DEBUG` plus
    // `REPLAY_TRACE_FILTER` narrows it down same as any other divergence.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(t, &mut surface, &mut outcome)
    }));
    match result {
        Ok(Err(reason)) => outcome.divergence = Some(reason),
        Ok(Ok(())) => {}
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            outcome.divergence = Some(format!("engine-panic:{msg}"));
        }
    }
    // Tally every auto-resolution the surface performed, regardless of
    // whether `run` ultimately succeeded or diverged -- these counters are
    // purely informational (see `main`'s printouts), not part of the
    // pass/fail signal.
    for s in surface.suppressions() {
        match s.reason {
            SuppressionReason::StepGated => outcome.silent_window_step_gated += 1,
            SuppressionReason::NoRealOption => outcome.trace_exhausted_passes += 1,
            SuppressionReason::NoEligibleAttacker => {
                outcome.silent_window_no_eligible_attacker += 1
            }
            SuppressionReason::NoEligibleBlockersForAttacker => {
                outcome.declare_blocks_no_eligible_blockers += 1
            }
            SuppressionReason::CombatPriorityActionSpent => {
                outcome.combat_priority_action_spent += 1
            }
            SuppressionReason::StackTopIsCastersOwn => outcome.stack_top_is_casters_own += 1,
        }
    }
    outcome
}

/// Per-trace replay context: everything derived once at setup time and
/// held immutably for the rest of the replay (id/player maps, per-seat
/// decision queues), plus `id_map`'s one exception -- see `learn_token_ids`.
struct ReplayCtx<'a> {
    /// Pregame card UUIDs at setup, *extended in place* as tokens are
    /// discovered mid-game (`learn_token_ids`) -- every other field here
    /// really is setup-only/immutable, but folding token bindings into this
    /// same map (rather than a parallel one) means every existing
    /// UUID-translation call site in this file keeps reading a plain
    /// `&HashMap<String, ObjectId>`, unchanged.
    id_map: HashMap<String, ObjectId>,
    /// The first `ObjectId` the kernel could ever assign to a *token*:
    /// every id below this was handed out by `GameState::new_from_libraries`
    /// to a real pregame card (already covered by `id_map`); every id at or
    /// above it can only be a token, since no card in this pool copies or
    /// otherwise mints a non-token object -- see `learn_token_ids`.
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

fn run(
    t: &GoldenTrace,
    surface: &mut HarnessSurfaceV1,
    outcome: &mut ReplayOutcome,
) -> Result<(), String> {
    if std::env::var("REPLAY_DEBUG").is_ok() {
        eprintln!("=== {} ===", t.source_path);
    }
    let (p0_name, p1_name) = seat_names(t)?;
    let opening0 = t
        .opening_hand_for(&p0_name)
        .ok_or("setup:no-opening-hand-record:p0")?;
    let opening1 = t
        .opening_hand_for(&p1_name)
        .ok_or("setup:no-opening-hand-record:p1")?;

    let lib0 = card_ids_for(opening0.hand.iter().chain(opening0.library.iter()))?;
    let lib1 = card_ids_for(opening1.hand.iter().chain(opening1.library.iter()))?;

    let mut state = GameState::new_from_libraries(
        &lib0,
        &lib1,
        |id| CARD_DEFS[id as usize].name.to_string(),
        t.header.seed,
    );
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
        t.decisions
            .iter()
            .filter(|d| {
                d.player == name
                    && d.action_type != "MULLIGAN"
                    && d.action_type != "LONDON_MULLIGAN"
            })
            .collect()
    };
    let mut ctx = ReplayCtx {
        id_map,
        pregame_object_count,
        seat_uuid,
        queues: [queue_for(&p0_name), queue_for(&p1_name)],
        cursors: [0, 0],
    };
    outcome.decisions_total = ctx.queues[0].len() + ctx.queues[1].len();

    loop {
        let decision = surface.next_decision(&mut state);
        if let Some(player) = decision_player(&decision, &state) {
            skip_stale_forced_discards(&state, &mut ctx, player, outcome);
        }
        match decision {
            SurfaceDecision::Decision(Decision::GameOver { winner }) => {
                outcome.reached_game_over = true;
                let winner_name = winner.map(|p| {
                    if p == PlayerId::P0 {
                        p0_name.clone()
                    } else {
                        p1_name.clone()
                    }
                });
                outcome.winner_matched =
                    matches!((&winner_name, &t.winner), (Some(a), Some(b)) if a == b);
                return Ok(());
            }
            SurfaceDecision::Decision(Decision::CastSpellOrPass {
                player,
                castable_spells,
                mana_abilities,
                land_drops,
                activatable_abilities,
                plot_actions,
            }) => match ctx.next(player) {
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
                        return Err(format!(
                            "decision-kind-mismatch:CastSpellOrPass-vs-{}",
                            rec.action_type
                        ));
                    }
                    check_state(&state, player, rec)?;
                    learn_token_ids(&mut ctx, &state, rec);
                    apply_cast_spell_or_pass(
                        surface,
                        &mut state,
                        rec,
                        &castable_spells,
                        &mana_abilities,
                        &land_drops,
                        &activatable_abilities,
                        &plot_actions,
                        &ctx.id_map,
                    )?;
                    ctx.advance(player);
                    outcome.decisions_consumed += 1;
                }
            },
            SurfaceDecision::Decision(Decision::ChooseTargets {
                player,
                legal_targets,
                ..
            }) => {
                if let Some(winner) =
                    java_reference_target_shortcut(&state, &legal_targets, &p0_name, &p1_name)
                {
                    surface
                        .apply(
                            &mut state,
                            SurfaceAction::Action(Action::ChooseTarget(winner)),
                        )
                        .map_err(|e| {
                            format!("engine-step-error:ChooseTargets-java-target-shortcut:{e}")
                        })?;
                    outcome.java_target_shortcut_applied += 1;
                    continue;
                }
                let &rec = ctx
                    .next(player)
                    .ok_or_else(|| "trace-exhausted:ChooseTargets".to_string())?;
                debug_verbose(t, &state, player, rec, "ChooseTargets");
                if rec.action_type != "SELECT_TARGETS" {
                    if std::env::var("REPLAY_DEBUG").is_ok() {
                        eprintln!(
                            "CHOOSETARGETS KIND MISMATCH decision_number={} player={} expected=ChooseTargets got={} legal_targets={:?} names={:?}",
                            rec.decision_number,
                            rec.player,
                            rec.action_type,
                            legal_targets,
                            legal_targets.iter().map(|tg| match tg { Target::Player(p) => format!("P{}", p.index()), Target::Object(id) => state.objects.get(*id).name.clone() }).collect::<Vec<_>>()
                        );
                    }
                    return Err(format!(
                        "decision-kind-mismatch:ChooseTargets-vs-{}",
                        rec.action_type
                    ));
                }
                // 601.2a: the kernel now moves a cast spell (or, for
                // flashback, the graveyard card) onto the stack at
                // announcement, before targets are chosen -- matching the
                // reference engine, so hand/graveyard sizes agree here with
                // no fudge factor needed (see `engine::begin_cast`).
                check_state(&state, player, rec)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_choose_targets(
                    surface,
                    &mut state,
                    rec,
                    &legal_targets,
                    &ctx.id_map,
                    &ctx.seat_uuid,
                )?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::Decision(Decision::DeclareAttackers { player, eligible }) => {
                let &rec = ctx
                    .next(player)
                    .ok_or_else(|| "trace-exhausted:DeclareAttackers".to_string())?;
                debug_verbose(t, &state, player, rec, "DeclareAttackers");
                if rec.action_type != "DECLARE_ATTACKS" {
                    return Err(format!(
                        "decision-kind-mismatch:DeclareAttackers-vs-{}",
                        rec.action_type
                    ));
                }
                check_state(&state, player, rec)?;
                learn_token_ids(&mut ctx, &state, rec);
                apply_declare_attackers(surface, &mut state, rec, &eligible, &ctx.id_map)?;
                ctx.advance(player);
                outcome.decisions_consumed += 1;
            }
            SurfaceDecision::DeclareBlockersForAttacker {
                attacker,
                legal_blockers,
            } => {
                let player = state.active_player.opponent();
                apply_declare_blockers_for_attacker(
                    surface,
                    &mut state,
                    t,
                    &mut ctx,
                    outcome,
                    player,
                    attacker,
                    &legal_blockers,
                )?;
            }
            SurfaceDecision::Decision(Decision::Discard {
                player,
                count,
                choices,
            }) => {
                apply_discard(
                    surface, &mut state, t, &mut ctx, outcome, player, count, &choices,
                )?;
            }
            SurfaceDecision::Decision(Decision::ChooseOptionalCost {
                player,
                discard_payable,
                sacrifice_payable,
            }) => {
                apply_choose_optional_cost(
                    surface,
                    &mut state,
                    &mut ctx,
                    player,
                    discard_payable,
                    sacrifice_payable,
                )?;
            }
            SurfaceDecision::Decision(Decision::ChooseMadnessCast { .. }) => {
                // See `apply_choose_optional_cost`'s doc for the same
                // invisibility argument: `MadnessCastEffect.apply()` calls
                // `owner.cast(...)` directly, with no `logReplayDecision`
                // counterpart, so this decision itself consumes no trace
                // record either way -- there is no ground truth to read off
                // the trace here, only a default that must not desync the
                // replay. Default: cast whenever affordable (a
                // strictly-cheaper-cost burn spell is essentially always
                // correct value) -- the resulting cast's own targeting
                // decision (ChooseTargets) is picked up generically by the
                // next loop iteration, exactly like any other cast --
                // *unless* a cast or ability activation is already mid-cost-
                // payment (`pending_cast`/`pending_activation`, e.g. this
                // Madness discard was itself part of paying for something
                // else's `DiscardCards` cost): casting right now would
                // spend mana/resources the in-flight action's own
                // not-yet-paid cost may still need, which the real game's
                // hidden choice evidently avoided (root-caused against
                // `game_20260712_194617_0017.txt` decision #47-50: floats
                // {R} to activate the Blood Token, discards Fiery Temper --
                // a Madness card -- to pay its cost, and the trace's own
                // post-hoc graveyard snapshot at decision #69 shows Fiery
                // Temper sitting in the graveyard, never cast; greedily
                // casting it here instead panics `mtg_kernel::engine`'s
                // `pay_cost_components` when the Blood Token's own `{1}`
                // then finds the pool empty). Declining is always legal
                // (`Action::ChooseMadnessCast(false)` just lets the card
                // fall through to the graveyard, same as a non-Madness
                // discard), so this is a safe, conservative default, not a
                // guess that can desync anything further.
                let mid_cost_payment = state.engine.pending_cast.is_some()
                    || state.engine.pending_activation.is_some();
                surface
                    .apply(
                        &mut state,
                        SurfaceAction::Action(Action::ChooseMadnessCast(!mid_cost_payment)),
                    )
                    .map_err(|e| format!("engine-step-error:ChooseMadnessCast:{e}"))?;
            }
            // Not observed anywhere in this corpus (verified by grep
            // across all 40 files); no sensible trace counterpart exists
            // to translate against, so this is a clean, named divergence
            // rather than a guess or a crash.
            SurfaceDecision::Decision(Decision::ChooseCastMode { .. }) => {
                return Err("unhandled-decision:ChooseCastMode".to_string())
            }
            SurfaceDecision::Decision(Decision::ChooseKicker { .. }) => {
                return Err("unhandled-decision:ChooseKicker".to_string())
            }
            SurfaceDecision::Decision(Decision::Halted { .. }) => {
                return Err("unhandled-decision:Halted".to_string())
            }
            SurfaceDecision::Decision(Decision::OrderTriggers { .. }) => {
                return Err("unhandled-decision:OrderTriggers".to_string())
            }
            // `engine::Decision::ChooseCostTargets` is new as of increment
            // 11 (H2/v4 work, Sol #90 -- see that variant's doc), added to
            // the shared `engine.rs` so the sacrifice-cost-target picks
            // Fireblast/Lava Dart require are real, player-visible
            // decisions instead of a silent auto-solve. H1/v3 is frozen and
            // this driver is not being extended to handle it (no v3-side
            // characterization of this window was done this increment) --
            // this arm exists only so the match stays exhaustive against
            // `engine::Decision`'s new shape; same clean-named-divergence
            // treatment as `ChooseCastMode`/`OrderTriggers` above, not a
            // change to H1's comparison logic.
            SurfaceDecision::Decision(Decision::ChooseCostTargets { .. }) => {
                return Err("unhandled-decision:ChooseCostTargets".to_string())
            }
            // Pyroblast/Red Elemental Blast are both sideboard-only for
            // this pool's maindeck -- unobserved in this corpus (same
            // reasoning as ChooseCastMode/OrderTriggers above).
            SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => {
                return Err("unhandled-decision:ChooseSpellMode".to_string())
            }
            // `HarnessSurfaceV1::next_decision` always reshapes a raw
            // `Decision::DeclareBlockers` into `DeclareBlockersForAttacker`
            // sub-decisions before returning (see that module's doc) --
            // this arm only exists so the match stays exhaustive against
            // `engine::Decision`'s full shape.
            SurfaceDecision::Decision(Decision::DeclareBlockers { .. }) => {
                return Err(
                    "unreachable-decision:DeclareBlockers-should-have-been-reshaped-by-the-surface"
                        .to_string(),
                );
            }
        }
    }
}

/// The acting player for every `SurfaceDecision` kind that has a trace
/// counterpart to consume (`GameOver` has none). Used to run
/// `skip_stale_forced_discards` uniformly ahead of every real decision,
/// regardless of kind -- see that function's doc.
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
        // The defending player -- same as `Decision::DeclareBlockers`'s own
        // `player` field before the surface reshaped it per attacker.
        SurfaceDecision::DeclareBlockersForAttacker { .. } => Some(state.active_player.opponent()),
        SurfaceDecision::Decision(Decision::GameOver { .. })
        | SurfaceDecision::Decision(Decision::ChooseCastMode { .. })
        | SurfaceDecision::Decision(Decision::ChooseKicker { .. })
        | SurfaceDecision::Decision(Decision::ChooseCostTargets { .. })
        | SurfaceDecision::Decision(Decision::Halted { .. })
        | SurfaceDecision::Decision(Decision::OrderTriggers { .. }) => None,
    }
}

/// `logReplayCardSelection`
/// logs a `SELECT_CARD` record even when the discard is forced (its own
/// candidate pool has exactly 1 legal card), but the kernel's
/// `drain_pending_discard_or_decide` auto-applies a forced discard
/// (`choices.len() <= count`) silently, with no `Decision::Discard` at all.
/// That leaves a trace record the kernel will never ask about sitting at
/// the front of the player's queue, which would otherwise wrongly
/// kind-mismatch against whatever real decision comes next.
///
/// Detects and skips exactly that shape: whenever the front of `player`'s
/// queue is a `SELECT_CARD` record whose every `chosen_object_ids` entry is
/// both translatable and already sitting in `player`'s graveyard, the
/// discard it describes must already have happened (a not-yet-applied
/// discard's chosen cards are still in hand at this point, so this can't
/// misfire on a genuine pending decision) -- consume it without a matching
/// engine action, since the kernel already applied it internally.
fn skip_stale_forced_discards(
    state: &GameState,
    ctx: &mut ReplayCtx,
    player: PlayerId,
    outcome: &mut ReplayOutcome,
) {
    loop {
        let Some(&rec) = ctx.next(player) else { return };
        if rec.action_type != "SELECT_CARD" || rec.chosen_object_ids.is_empty() {
            return;
        }
        let already_applied = rec
            .chosen_object_ids
            .iter()
            .all(|uuid| match ctx.id_map.get(uuid) {
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
    let p0 = t
        .decisions
        .first()
        .ok_or_else(|| "setup:no-decisions".to_string())?
        .player
        .clone();
    let p1 = names
        .into_iter()
        .find(|&n| n != p0)
        .ok_or_else(|| "setup:cannot-determine-p1-name".to_string())?
        .to_string();
    Ok((p0, p1))
}

fn debug_verbose(
    t: &GoldenTrace,
    state: &GameState,
    player: PlayerId,
    rec: &DecisionRecord,
    kind: &str,
) {
    let Ok(filter) = std::env::var("REPLAY_TRACE_FILTER") else {
        return;
    };
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
    names
        .map(|n| card_def::card_id_by_name(n).ok_or_else(|| format!("setup:unknown-card-name:{n}")))
        .collect()
}

/// Zips each seat's `hand_object_ids ++ library_object_ids` (trace UUIDs)
/// against the kernel `ObjectId`s `GameState::new_from_libraries` assigns
/// in that same order (0..len(lib0) for P0, offset thereafter for P1 --
/// see `state.rs`'s `new_from_libraries_assigns_ids_p0_first` test).
fn build_id_map(
    opening0: &trace::OpeningHand,
    opening1: &trace::OpeningHand,
    p0_object_count: u32,
) -> HashMap<String, ObjectId> {
    let mut id_map = HashMap::new();
    for (i, uuid) in opening0
        .hand_object_ids
        .iter()
        .chain(opening0.library_object_ids.iter())
        .enumerate()
    {
        id_map.insert(uuid.clone(), ObjectId(i as u32));
    }
    for (i, uuid) in opening1
        .hand_object_ids
        .iter()
        .chain(opening1.library_object_ids.iter())
        .enumerate()
    {
        id_map.insert(uuid.clone(), ObjectId(p0_object_count + i as u32));
    }
    id_map
}

/// Learns token UUID<->`ObjectId` bindings the moment a trace record first
/// references one, extending `ctx.id_map` in place -- root-causing the
/// increment-8 "Blood Token candidates untranslatable" family (tokens
/// created mid-game have trace UUIDs absent from the pregame `id_map`,
/// which is built once, at setup, off both decklists).
///
/// Binding discipline: **positional match by creation order.** Any
/// `candidate_object_ids`/`chosen_object_ids` entry not already resolvable
/// via `ctx.id_map` can only be a token -- this card pool's only source of
/// new objects after setup (no copy/create-a-copy effect exists here), so
/// every non-token UUID is guaranteed present in the pregame map already.
/// Bind it to the *oldest still-unbound* kernel-side token: `Arena::push`
/// hands out `ObjectId`s in creation order (`ids.rs`'s
/// `ids_assigned_in_push_order` test), and every id at or above
/// `ctx.pregame_object_count` can only belong to a token (see that field's
/// doc) -- so the first such id with no existing entry in `ctx.id_map`'s
/// *values* is exactly the next token the kernel minted that the trace
/// hasn't referenced yet. This is safe (not just a heuristic) because both
/// engines are replaying the *same* decision sequence: they mint tokens in
/// the same causal order, and a token cannot be referenced by a trace
/// record before the event that creates it, so the two discovery orders --
/// "next kernel token by id" and "next trace UUID a record asks about" --
/// must line up one-to-one.
///
/// Split on `->` first for `DECLARE_BLOCKS`-shaped `blocker->attacker`
/// pair candidates (harmless for every other decision kind's plain UUIDs,
/// which never contain that substring) -- no card in this pool creates a
/// token that can attack or block, but this keeps the scan correct instead
/// of quietly relying on that.
fn learn_token_ids(ctx: &mut ReplayCtx, state: &GameState, rec: &DecisionRecord) {
    for raw in rec
        .candidate_object_ids
        .iter()
        .chain(rec.chosen_object_ids.iter())
    {
        for uuid in raw.split("->") {
            if uuid.is_empty() || uuid == DONE || ctx.id_map.contains_key(uuid) {
                continue;
            }
            let bound: std::collections::HashSet<ObjectId> = ctx.id_map.values().copied().collect();
            let Some(next) = state
                .objects
                .iter()
                .map(|(id, _)| id)
                .find(|id| id.0 >= ctx.pregame_object_count && !bound.contains(id))
            else {
                continue; // kernel hasn't minted this many tokens (yet) -- leave untranslatable, a real divergence surfaces downstream.
            };
            ctx.id_map.insert(uuid.to_string(), next);
        }
    }
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
                found.insert(
                    if text == p0_name { p0_name } else { p1_name },
                    uuid.clone(),
                );
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
                "HAND MISMATCH decision_number={} player={} action={} rec_turn={} state_turn={} kernel_hand={} trace_hand={} kernel_names={:?} trace_names={:?} stack_len={} stack_sources={:?}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                rec.turn,
                state.turn,
                ps.hand.len(),
                rec.hand.len(),
                ps.hand.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                rec.hand,
                state.stack.len(),
                state.stack.iter().map(|item| state.objects.get(item.source).name.clone()).collect::<Vec<_>>()
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
                "GRAVEYARD MISMATCH decision_number={} player={} action={} kernel_gy={} trace_gy={} kernel_names={:?} trace_names={:?} stack_len={} stack_sources={:?}",
                rec.decision_number,
                rec.player,
                rec.action_type,
                ps.graveyard.len(),
                rec.graveyard.len(),
                ps.graveyard.iter().map(|&id| state.objects.get(id).name.clone()).collect::<Vec<_>>(),
                rec.graveyard,
                state.stack.len(),
                state.stack.iter().map(|item| state.objects.get(item.source).name.clone()).collect::<Vec<_>>()
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
    PlotSpell(ObjectId),
}

/// A cast candidate's zone-derived tag: which of `castable_spells`'s three
/// sources (hand, graveyard flashback, exile Plot) `id` currently sits in --
/// folded into `candidate_key`'s `cast:` bucket key so a hand-cast, a
/// flashback cast, and a free Plot-cast of the *same* card (Highway
/// Robbery, at different points across a game) never collide.
fn cast_zone_tag(state: &GameState, id: ObjectId) -> &'static str {
    match state.objects.get(id).zone {
        Zone::Graveyard => "graveyard",
        Zone::Exile => "exile",
        _ => "hand",
    }
}

/// Equivalence-class key for an `ACTIVATE_ABILITY_OR_SPELL` candidate:
/// which bucket it's from, plus enough of its identity (card name,
/// hand-vs-flashback, ability index) to distinguish genuinely different
/// choices while collapsing fungible ones (2 untapped Mountains) into the
/// same key -- see the module doc's point 3. `None` means `id` isn't a
/// member of any of the 4 current buckets (a real divergence signal, not
/// swallowed by the caller).
/// `text` disambiguates a trace candidate whose `id` is a member of *two*
/// buckets at once: Highway Robbery in hand is simultaneously a real
/// `castable_spells` entry ("Cast Highway Robbery") and a real
/// `plot_actions` entry ("Plot {1}{R}") -- same underlying `ObjectId`, two
/// different actions on it. Once `translate_object_candidates` has already
/// reduced a trace candidate down to `Option<ObjectId>`, that distinction
/// is gone unless the candidate's own text (XMage's `PlotAbility.rule` is
/// always exactly `"Plot " + plotCost`, e.g. "Plot {1}{R}") is consulted
/// too -- so this checks `text` first for exactly that shape, before
/// falling through to the ordinary id-only bucket checks.
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

// `surface` is a mandatory plumbing parameter added on top of an
// already-wide argument list (one entry per kernel candidate bucket); not
// worth a bundling struct for 3 call sites.
#[allow(clippy::too_many_arguments)]
fn apply_cast_spell_or_pass(
    surface: &mut HarnessSurfaceV1,
    state: &mut GameState,
    rec: &DecisionRecord,
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    land_drops: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    plot_actions: &[ObjectId],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(), String> {
    // One representative KernelChoice per equivalence class (first one
    // encountered wins -- any representative works, they're fungible by
    // construction; see `candidate_key`'s doc and the module doc's point
    // 3). A given object can only land in one of the 5 buckets in this
    // card pool (land vs. castable are mutually exclusive by zone/type;
    // mana/activated abilities are battlefield-only, disjoint from
    // hand-only land/cast/plot candidates; Plotting and casting are
    // different actions on the same hand card, keyed apart by the `plot:`
    // prefix), so this is well-defined.
    let mut by_key: BTreeMap<String, KernelChoice> = BTreeMap::new();
    by_key.insert("pass".to_string(), KernelChoice::Pass);
    for &id in land_drops {
        by_key
            .entry(format!("land:{}", state.objects.get(id).name))
            .or_insert(KernelChoice::PlayLand(id));
    }
    for &id in castable_spells {
        by_key
            .entry(format!(
                "cast:{}:{}",
                state.objects.get(id).name,
                cast_zone_tag(state, id)
            ))
            .or_insert(KernelChoice::CastSpell(id));
    }
    for &id in mana_abilities {
        by_key
            .entry(format!("mana:{}", state.objects.get(id).name))
            .or_insert(KernelChoice::ActivateMana(id));
    }
    for &(id, idx) in activatable_abilities {
        by_key
            .entry(format!("activate:{}:{idx}", state.objects.get(id).name))
            .or_insert(KernelChoice::ActivateAbility(id, idx));
    }
    for &id in plot_actions {
        by_key
            .entry(format!("plot:{}", state.objects.get(id).name))
            .or_insert(KernelChoice::PlotSpell(id));
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
                        "NOT-IN-BUCKET decision_number={} text={text:?} object_zone={:?} object_name={:?} kernel_castable={:?} kernel_land={:?} kernel_mana={:?} kernel_plot={:?} kernel_step={:?} trace_phase={:?} lands_played_this_turn={:?}",
                        rec.decision_number,
                        state.objects.get(*oid).zone,
                        state.objects.get(*oid).name,
                        castable_spells,
                        land_drops,
                        mana_abilities,
                        plot_actions,
                        state.step,
                        rec.phase,
                        state.players.iter().map(|p| p.lands_played_this_turn).collect::<Vec<_>>()
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
    let chosen_key = trace_keys
        .get(idx)
        .ok_or("chosen-index-out-of-range:CastSpellOrPass")?;

    let action = match by_key.get(chosen_key) {
        Some(KernelChoice::Pass) => Action::Pass,
        Some(KernelChoice::PlayLand(id)) => Action::PlayLand(*id),
        Some(KernelChoice::CastSpell(id)) => Action::CastSpell(*id),
        Some(KernelChoice::ActivateMana(id)) => Action::ActivateManaAbility(*id),
        Some(KernelChoice::ActivateAbility(id, idx)) => Action::ActivateAbility(*id, *idx),
        Some(KernelChoice::PlotSpell(id)) => Action::PlotSpell(*id),
        None => return Err("chosen-not-in-kernel-candidates:CastSpellOrPass".to_string()),
    };
    surface
        .apply(state, SurfaceAction::Action(action))
        .map_err(|e| format!("engine-step-error:CastSpellOrPass:{e}"))
}

/// Translates every `(candidate_texts[i], candidate_object_ids[i])` pair
/// of an `ACTIVATE_ABILITY_OR_SPELL` record into `Option<ObjectId>`
/// (`None` = the implicit "Pass" candidate, `candidate_object_ids[i] ==
/// ""`), in original candidate order (index-addressable, so
/// `chosen_indices` can look items up directly).
fn translate_object_candidates(
    rec: &DecisionRecord,
    id_map: &HashMap<String, ObjectId>,
    kind: &str,
) -> Result<Vec<Option<ObjectId>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_texts.len());
    for (text, uuid) in rec
        .candidate_texts
        .iter()
        .zip(rec.candidate_object_ids.iter())
    {
        if text == "Pass" && uuid.is_empty() {
            out.push(None);
        } else {
            let id = id_map
                .get(uuid)
                .copied()
                .ok_or_else(|| format!("untranslatable-object-id:{kind}:{uuid}"))?;
            out.push(Some(id));
        }
    }
    Ok(out)
}

fn apply_choose_targets(
    surface: &mut HarnessSurfaceV1,
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
        trace_targets.push(
            translate(uuid).ok_or_else(|| format!("untranslatable-target:ChooseTargets:{uuid}"))?,
        );
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
    let target = *trace_targets
        .get(idx)
        .ok_or("chosen-index-out-of-range:ChooseTargets")?;

    surface
        .apply(state, SurfaceAction::Action(Action::ChooseTarget(target)))
        .map_err(|e| format!("engine-step-error:ChooseTargets:{e}"))
}

fn target_key(t: &Target) -> String {
    match t {
        Target::Player(p) => format!("P{}", p.index()),
        Target::Object(id) => format!("O{}", id.0),
    }
}

/// Root cause for the `decision-kind-mismatch:ChooseTargets-vs-*` family
/// (uncharacterized at increment-7): `ComputerPlayerRL.chooseTarget`'s
/// "all candidates share a name" dedup shortcut (ComputerPlayerRL.java:
/// 6682-6706, `allSameName`) has a latent bug for this pool's only real
/// target shape, `AnyTarget` (both players + any creatures). The shortcut
/// names candidates via `MageObject obj = game.getObject(id); String name =
/// obj != null ? obj.getName() : null;` -- but `GameImpl.getObject`
/// (GameImpl.java:465-509) only ever searches battlefield/entering
/// permanents, the stack, the command zone, and cards; it never resolves a
/// *player* UUID, so both players yield `name == null`. The loop's own
/// bookkeeping (`if (firstName == null) { firstName = name; } else if
/// (!Objects.equals(firstName, name)) { allSameName = false; }`) only ever
/// *compares* when `firstName` is already non-null -- a null-named entry
/// silently resets the comparison state instead of ever being compared
/// against. Since `sortTargetsForStableChoice` (ComputerPlayerRL.java:
/// 6990-6997) always sorts both players before any creature (`"0|" + name`
/// vs `"1|" + name`), the two players are always first in `possible` and
/// their null names never trip `allSameName` false -- only two *creatures*
/// with different names, both past the players, ever do. So: the shortcut
/// silently fires (`picked = possible.get(0)`, the alphabetically-first
/// player by display name -- the RL model is never called, and nothing is
/// ever logged) for *every* `AnyTarget` window unless 2+ differently-named
/// creatures are among the legal targets.
///
/// Confirmed empirically against `game_20260712_194623_0023.txt` decision
/// #139-143: SelfPlay casts Fireblast (`AnyTarget`) with only itself,
/// PlayerRL1, and a single Voldaren Epicure as legal targets (no second
/// distinctly-named creature) -- no `SELECT_TARGETS` record appears
/// anywhere between the cast and the next real decision (`DECLARE_ATTACKS`).
///
/// This is a reference-harness *bug* (not a comprehensive-rules behavior,
/// and not even really a training/inference concern -- a plain Java `null`
/// mishandling), so `mtg_kernel::engine` correctly does not reproduce it.
/// It's also not implemented in `mtg_kernel::surface::HarnessSurfaceV1`
/// despite being reference-*visibility* behavior in the same spirit as
/// that module's own predicate: the tie-break needs the two players'
/// *display-name strings* ("PlayerRL1" vs "SelfPlay") to replicate
/// `stableTargetSortName`'s ordering, and `GameState`/`PlayerId` carry no
/// such thing -- only this trace-parsing comparator does (`p0_name`/
/// `p1_name`, threaded in below). A future model-serving path driving a
/// real `Player` with a real name could implement this in its own
/// surface-equivalent; this kernel increment doesn't need to.
///
/// Returns the target this shortcut would silently pick, or `None` if a
/// real (loggable) decision is expected instead -- i.e. `legal_targets`
/// isn't this pool's `AnyTarget` shape (both players present) at all, or
/// the creature subset doesn't collapse to exactly one distinct name.
///
/// That second condition is deliberately *not* "<= 1": with **zero**
/// creatures present (candidates are just the two players, both null-named
/// per this function's own doc), Java's `firstName` never leaves `null`
/// either (the loop's `if (firstName == null) { firstName = name; }` branch
/// just keeps reassigning it `null`) -- so `if (allSameName && firstName !=
/// null)` is *false* and the shortcut does *not* fire, falling through to
/// the real, loggable RL-model branch instead. Only once at least one
/// creature is present does `firstName` ever pick up a real (non-null)
/// value, satisfying that guard. Getting this edge case wrong (treating 0
/// creatures the same as 1) was caught empirically: it silently ate real
/// `SELECT_TARGETS` records for early-game direct-damage-to-a-player casts
/// (no creatures on board yet), desyncing the trace cursor and spiking
/// `decision-kind-mismatch:CastSpellOrPass-vs-SELECT_TARGETS` corpus-wide.
fn java_reference_target_shortcut(
    state: &GameState,
    legal_targets: &[Target],
    p0_name: &str,
    p1_name: &str,
) -> Option<Target> {
    if !legal_targets.contains(&Target::Player(PlayerId::P0))
        || !legal_targets.contains(&Target::Player(PlayerId::P1))
    {
        return None; // not this pool's AnyTarget shape
    }
    let mut creature_names: Vec<&str> = legal_targets
        .iter()
        .filter_map(|t| match t {
            Target::Object(id) => Some(state.objects.get(*id).name.as_str()),
            Target::Player(_) => None,
        })
        .collect();
    creature_names.sort_unstable();
    creature_names.dedup();
    if creature_names.len() != 1 {
        return None; // 0 creatures (firstName stays null) or 2+ distinct names (allSameName trips false) -- both are real decisions
    }
    Some(if p0_name < p1_name {
        Target::Player(PlayerId::P0)
    } else {
        Target::Player(PlayerId::P1)
    })
}

fn apply_declare_attackers(
    surface: &mut HarnessSurfaceV1,
    state: &mut GameState,
    rec: &DecisionRecord,
    eligible: &[ObjectId],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(), String> {
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
    let attackers =
        apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareAttackers")?;
    surface
        .apply(
            state,
            SurfaceAction::Action(Action::DeclareAttackers(attackers)),
        )
        .map_err(|e| format!("engine-step-error:DeclareAttackers:{e}"))
}

/// `chosen_indices` for `DECLARE_ATTACKS`/`DECLARE_BLOCKS` is a full
/// permutation of every candidate index (including the `sentinel:DONE`
/// entry, translated to `None` here) -- see the module doc's point 1.
/// The real applied picks are the *prefix* up to (excluding) the first
/// `None`; anything after it is unapplied ranking noise, not a second
/// round of picks.
fn apply_prefix_before_done<T: Copy>(
    chosen_indices: &[u32],
    candidates: &[Option<T>],
    kind: &str,
) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    for &idx in chosen_indices {
        match candidates
            .get(idx as usize)
            .ok_or_else(|| format!("chosen-index-out-of-range:{kind}"))?
        {
            None => break,
            Some(v) => out.push(*v),
        }
    }
    Ok(out)
}

fn translate_attacker_candidates(
    rec: &DecisionRecord,
    id_map: &HashMap<String, ObjectId>,
) -> Result<Vec<Option<ObjectId>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
        } else {
            let id = id_map
                .get(uuid)
                .copied()
                .ok_or_else(|| format!("untranslatable-object-id:DeclareAttackers:{uuid}"))?;
            out.push(Some(id));
        }
    }
    Ok(out)
}

/// Answers one `SurfaceDecision::DeclareBlockersForAttacker` -- the surface
/// has already reshaped the kernel's single `Decision::DeclareBlockers` (and
/// its own attacker-with-zero-eligible-blockers suppression) into exactly
/// this per-attacker shape, matching `selectBlockers`'s own per-attacker
/// logging (`ComputerPlayerRL.java:8795-8983`) 1:1 -- see
/// `mtg_kernel::surface`'s module doc. Consumes exactly one `DECLARE_BLOCKS`
/// trace record and hands the picks back to the surface, which applies the
/// combined `Action::DeclareBlockers` automatically once every attacker has
/// been answered.
#[allow(clippy::too_many_arguments)]
fn apply_declare_blockers_for_attacker(
    surface: &mut HarnessSurfaceV1,
    state: &mut GameState,
    t: &GoldenTrace,
    ctx: &mut ReplayCtx,
    outcome: &mut ReplayOutcome,
    player: PlayerId,
    attacker: ObjectId,
    legal_blockers: &[ObjectId],
) -> Result<(), String> {
    let &rec = ctx
        .next(player)
        .ok_or_else(|| "trace-exhausted:DeclareBlockers".to_string())?;
    debug_verbose(t, state, player, rec, "DeclareBlockers");
    if rec.action_type != "DECLARE_BLOCKS" {
        return Err(format!(
            "decision-kind-mismatch:DeclareBlockers-vs-{}",
            rec.action_type
        ));
    }
    check_state(state, player, rec)?;
    learn_token_ids(ctx, state, rec);

    // Blocker-major (blocker, attacker) to match the trace's
    // "blockerUuid->attackerUuid" text convention, scoped to just this one
    // attacker (the kernel's `legal_blockers` was already attacker-major
    // before the surface split it apart).
    let mut kernel_keys: Vec<String> = legal_blockers
        .iter()
        .map(|b| format!("{}->{}", b.0, attacker.0))
        .collect();
    kernel_keys.push("DONE".to_string());
    kernel_keys.sort();

    let trace_candidates = translate_blocker_candidates(rec, &ctx.id_map)?;
    // Every non-DONE candidate in a real DECLARE_BLOCKS record names
    // exactly one attacker (0 of the v3 corpus's 37 records mix
    // attackers). A record naming a different attacker than the one the
    // surface expects next means this record doesn't belong here.
    if trace_candidates
        .iter()
        .any(|c| matches!(c, Some((_, a)) if a != &attacker))
    {
        return Err("declare-blocks-attacker-mismatch".to_string());
    }
    let mut trace_keys: Vec<String> = trace_candidates
        .iter()
        .map(|c| match c {
            Some((blocker, a)) => format!("{}->{}", blocker.0, a.0),
            None => "DONE".to_string(),
        })
        .collect();
    trace_keys.sort();

    if kernel_keys != trace_keys {
        return Err("candidate-multiset-mismatch:DeclareBlockers".to_string());
    }

    // Same prefix-before-DONE rule as DeclareAttackers.
    let picks =
        apply_prefix_before_done(&rec.chosen_indices, &trace_candidates, "DeclareBlockers")?;
    let blockers_only: Vec<ObjectId> = picks.into_iter().map(|(blocker, _)| blocker).collect();

    ctx.advance(player);
    outcome.decisions_consumed += 1;
    surface
        .apply(
            state,
            SurfaceAction::DeclareBlockersForAttacker(blockers_only),
        )
        .map_err(|e| format!("engine-step-error:DeclareBlockers:{e}"))
}

/// `SELECT_CARD` candidates are matched by *name* multiset (not object id,
/// unlike every other decision kind in this file) -- see the module doc's
/// `SELECT_CARD` section. `rec.candidate_texts`/`choices` must
/// agree as a name multiset; each `chosen_indices` entry names a card via
/// `candidate_texts[idx]`, resolved against the *remaining* kernel pool one
/// name at a time so duplicate names (two Guttersnipes) split correctly
/// across distinct objects without needing a UUID lookup.
///
/// Before the aggregate `SELECT_CARD` record, `chooseTarget`'s own per-card
/// model-scoring loop (the *same* mechanism backing `SELECT_TARGETS`)
/// independently logs one `SELECT_TARGETS` record per card chosen for this
/// exact discard -- confirmed empirically (e.g. Faithless Looting's 2-card
/// discard: two `SELECT_TARGETS` records with `source_name="Faithless
/// Looting"`, chosen one card each, immediately followed by one
/// `SELECT_CARD` record covering both). Those `SELECT_TARGETS` records have
/// no kernel decision of their own -- the kernel's discard model is
/// `Decision::Discard`/`Action::Discard`, never `ChooseTargets` -- so this
/// consumes and discards any such prefix here first, cross-checking their
/// chosen-name sequence against the terminal `SELECT_CARD` record's own
/// sequence as an integrity check (not just a blind skip). The prefix can
/// be shorter than `count` (or absent): `chooseTarget`'s per-pick loop has
/// its own single-legal-candidate shortcut that skips logging a pick once
/// the remaining pool narrows to 1, same as everywhere else in the
/// reference -- so this loop adapts to however many actually appear rather
/// than assuming exactly `count`.
#[allow(clippy::too_many_arguments)]
fn apply_discard(
    surface: &mut HarnessSurfaceV1,
    state: &mut GameState,
    t: &GoldenTrace,
    ctx: &mut ReplayCtx,
    outcome: &mut ReplayOutcome,
    player: PlayerId,
    count: u32,
    choices: &[ObjectId],
) -> Result<(), String> {
    let mut names_from_targets_prefix: Vec<String> = Vec::new();
    while let Some(&rec) = ctx.next(player) {
        if rec.action_type != "SELECT_TARGETS" {
            break;
        }
        debug_verbose(t, state, player, rec, "Discard(SELECT_TARGETS-prefix)");
        let &idx = rec
            .chosen_indices
            .first()
            .ok_or("unexpected-chosen-count:Discard-SELECT_TARGETS-prefix")?;
        let name = rec
            .candidate_texts
            .get(idx as usize)
            .ok_or("chosen-index-out-of-range:Discard-SELECT_TARGETS-prefix")?;
        names_from_targets_prefix.push(name.clone());
        ctx.advance(player);
        outcome.decisions_consumed += 1;
    }

    let &rec = ctx
        .next(player)
        .ok_or_else(|| "trace-exhausted:Discard".to_string())?;
    debug_verbose(t, state, player, rec, "Discard");
    if rec.action_type != "SELECT_CARD" {
        return Err(format!(
            "decision-kind-mismatch:Discard-vs-{}",
            rec.action_type
        ));
    }
    if rec.chosen_indices.len() != count as usize {
        return Err("unexpected-chosen-count:Discard".to_string());
    }
    check_state(state, player, rec)?;

    let chosen_names: Vec<String> = rec
        .chosen_indices
        .iter()
        .map(|&idx| {
            rec.candidate_texts
                .get(idx as usize)
                .cloned()
                .ok_or_else(|| "chosen-index-out-of-range:Discard".to_string())
        })
        .collect::<Result<_, _>>()?;
    if !names_from_targets_prefix.is_empty() && names_from_targets_prefix != chosen_names {
        return Err("discard-select-targets-prefix-mismatch".to_string());
    }

    let mut kernel_names: Vec<&str> = choices
        .iter()
        .map(|&id| state.objects.get(id).name.as_str())
        .collect();
    kernel_names.sort_unstable();
    let mut trace_names: Vec<&str> = rec.candidate_texts.iter().map(String::as_str).collect();
    trace_names.sort_unstable();
    if kernel_names != trace_names {
        return Err("candidate-multiset-mismatch:Discard".to_string());
    }

    let mut pool: Vec<ObjectId> = choices.to_vec();
    let mut chosen: Vec<ObjectId> = Vec::with_capacity(chosen_names.len());
    for name in &chosen_names {
        let pos = pool
            .iter()
            .position(|&id| state.objects.get(id).name == *name)
            .ok_or_else(|| format!("chosen-name-not-in-pool:Discard:{name}"))?;
        chosen.push(pool.remove(pos));
    }

    ctx.advance(player);
    outcome.decisions_consumed += 1;
    surface
        .apply(state, SurfaceAction::Action(Action::Discard(chosen)))
        .map_err(|e| format!("engine-step-error:Discard:{e}"))
}

/// Highway Robbery's `Decision::ChooseOptionalCost` ("You may discard a
/// card or sacrifice a land. If you do, draw two cards.") -- reverse
/// engineered from the Java source (`DoIfCostPaid.apply`/`OrCost.pay`, both
/// in `Mage/src/main/java/mage/abilities/`): the "may pay?" gate and the
/// "which of discard/sacrifice?" sub-choice are *both* routed through
/// `Player.chooseUse`, which `ComputerPlayerRL` never logs via
/// `logReplayDecision` (it only calls `GameLogger.logDecision`, a separate
/// human-readable log this driver doesn't parse) -- confirmed empirically:
/// `CHOOSE_USE` never appears as an `action_type` anywhere in the v3
/// corpus. So this decision consumes zero trace records either way.
///
/// The one outcome that *does* leave a footprint: paying via discard routes
/// through `DiscardCardCost.pay`'s own `chooseTarget` call, which is the
/// same visible `SELECT_TARGETS`-then-`SELECT_CARD` shape every other
/// discard cost in this pool uses (see the module doc's `SELECT_CARD`
/// section) -- so it shows up as the very next record in this player's
/// queue (nothing else can interleave: this fires synchronously inside
/// Highway Robbery's own resolution, before any priority is re-offered).
/// Paying via sacrificing a land leaves no footprint either (this pool's
/// only land, Mountain, is fungible, so `chooseTarget`'s same-name-dedup
/// shortcut -- the same one that makes Fireblast's alt cost silent --
/// applies), making it indistinguishable from Decline at this exact
/// decision boundary. Default: Decline, unless the next queued record
/// looks like a discard pick from the current hand -- see
/// `check_state`/`apply_discard`'s own zone-size checks for how a wrong
/// guess here still gets caught (as a downstream `zone-size-mismatch`),
/// not silently swallowed.
fn apply_choose_optional_cost(
    surface: &mut HarnessSurfaceV1,
    state: &mut GameState,
    ctx: &mut ReplayCtx,
    player: PlayerId,
    discard_payable: bool,
    _sacrifice_payable: bool,
) -> Result<(), String> {
    let hand_len = state.players[player.index()].hand.len();
    let looks_like_discard_pick = discard_payable
        && matches!(ctx.next(player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && rec.candidate_texts.len() == hand_len);
    let choice = if looks_like_discard_pick {
        OptionalCostChoice::Discard
    } else {
        OptionalCostChoice::Decline
    };
    surface
        .apply(
            state,
            SurfaceAction::Action(Action::ChooseOptionalCost(choice)),
        )
        .map_err(|e| format!("engine-step-error:ChooseOptionalCost:{e}"))
}

fn translate_blocker_candidates(
    rec: &DecisionRecord,
    id_map: &HashMap<String, ObjectId>,
) -> Result<Vec<Option<(ObjectId, ObjectId)>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
            continue;
        }
        let (blocker_uuid, attacker_uuid) = uuid
            .split_once("->")
            .ok_or_else(|| format!("malformed-block-pair:DeclareBlockers:{uuid}"))?;
        let blocker = id_map
            .get(blocker_uuid)
            .copied()
            .ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{blocker_uuid}"))?;
        let attacker = id_map
            .get(attacker_uuid)
            .copied()
            .ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{attacker_uuid}"))?;
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
    fn decision_record(
        action_type: &str,
        candidate_texts: &[&str],
        candidate_object_ids: &[&str],
        chosen_indices: &[u32],
    ) -> DecisionRecord {
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
        assert_eq!(
            picked,
            Vec::<i32>::new(),
            "DONE first means no attackers, even though more picks follow it in the array"
        );

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
        let rec = decision_record(
            "DECLARE_ATTACKS",
            &["Guttersnipe", "DONE"],
            &["attacker-uuid", "sentinel:DONE"],
            &[0, 1],
        );

        let translated = translate_attacker_candidates(&rec, &id_map).unwrap();
        assert_eq!(translated, vec![Some(ObjectId(7)), None]);
    }

    #[test]
    fn translate_attacker_candidates_reports_an_untranslatable_uuid() {
        let id_map = HashMap::new(); // empty: nothing translates
        let rec = decision_record(
            "DECLARE_ATTACKS",
            &["Guttersnipe", "DONE"],
            &["attacker-uuid", "sentinel:DONE"],
            &[0, 1],
        );
        let err = translate_attacker_candidates(&rec, &id_map).unwrap_err();
        assert!(err.contains("untranslatable-object-id:DeclareAttackers"));
    }

    #[test]
    fn translate_blocker_candidates_splits_blocker_attacker_pairs_blocker_major() {
        let mut id_map = HashMap::new();
        id_map.insert("blocker-uuid".to_string(), ObjectId(3));
        id_map.insert("attacker-uuid".to_string(), ObjectId(9));
        let rec = decision_record(
            "DECLARE_BLOCKS",
            &["Guttersnipe", "DONE"],
            &["blocker-uuid->attacker-uuid", "sentinel:DONE"],
            &[0, 1],
        );

        let translated = translate_blocker_candidates(&rec, &id_map).unwrap();
        assert_eq!(
            translated,
            vec![Some((ObjectId(3), ObjectId(9))), None],
            "must be (blocker, attacker), not (attacker, blocker)"
        );
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
        let mut state = GameState::new_from_libraries(
            &[mountain, mountain],
            &[],
            |id| CARD_DEFS[id as usize].name.to_string(),
            1,
        );
        let a = state.draw_card(PlayerId::P0).unwrap();
        let b = state.draw_card(PlayerId::P0).unwrap();
        (state, a, b)
    }

    #[test]
    fn candidate_key_gives_two_untapped_mountains_the_same_land_drop_key() {
        let (state, a, b) = two_mountains_in_hand();
        let land_drops = [a, b];
        let key_a =
            candidate_key(&state, a, "Play Mountain", &land_drops, &[], &[], &[], &[]).unwrap();
        let key_b =
            candidate_key(&state, b, "Play Mountain", &land_drops, &[], &[], &[], &[]).unwrap();
        assert_eq!(key_a, key_b, "two interchangeable Mountains must dedup to one ACTIVATE_ABILITY_OR_SPELL candidate, matching the reference engine's own display-layer dedup");
        assert_eq!(key_a, "land:Mountain");
    }

    #[test]
    fn candidate_key_is_none_for_an_object_not_in_any_current_bucket() {
        let (state, a, _b) = two_mountains_in_hand();
        // `a` isn't a member of any of these (empty) buckets.
        assert_eq!(
            candidate_key(&state, a, "Play Mountain", &[], &[], &[], &[], &[]),
            None
        );
    }

    /// Regression test for a real divergence found against the v3 corpus:
    /// Highway Robbery in hand is simultaneously a `castable_spells` member
    /// ("Cast Highway Robbery") and a `plot_actions` member ("Plot
    /// {1}{R}") -- same `ObjectId`, two different trace candidates. Without
    /// consulting `text`, `candidate_key` always returned the `cast:` key
    /// for both, collapsing them into one candidate and desyncing every
    /// game that ever offered both in the same window (2 of the v3
    /// corpus's 40 games).
    #[test]
    fn candidate_key_disambiguates_cast_vs_plot_for_the_same_plottable_card() {
        let robbery = card_def::card_id_by_name("Highway Robbery").unwrap();
        let mut state = GameState::new_from_libraries(
            &[robbery],
            &[],
            |id| CARD_DEFS[id as usize].name.to_string(),
            1,
        );
        let id = state.draw_card(PlayerId::P0).unwrap();
        let castable_spells = [id];
        let plot_actions = [id];

        let cast_key = candidate_key(
            &state,
            id,
            "Cast Highway Robbery",
            &[],
            &castable_spells,
            &[],
            &[],
            &plot_actions,
        )
        .unwrap();
        let plot_key = candidate_key(
            &state,
            id,
            "Plot {1}{R}",
            &[],
            &castable_spells,
            &[],
            &[],
            &plot_actions,
        )
        .unwrap();
        assert_ne!(
            cast_key, plot_key,
            "casting and Plotting the same card must produce distinct candidate keys"
        );
        assert_eq!(cast_key, "cast:Highway Robbery:hand");
        assert_eq!(plot_key, "plot:Highway Robbery");
    }

    // ---- turn conversion (json global turn -> kernel round) ----------

    #[test]
    fn expected_round_matches_free_text_examples_from_the_real_corpus() {
        // json turn -> "DECISION #N - Turn R (<player> turn)" cross-checks
        // recorded in the increment-4 report: 13<->7 (PlayerRL1), 23<->12
        // (PlayerRL1), 25<->13 (PlayerRL1), 26<->13 (SelfPlay).
        for (json_turn, expected_round) in
            [(1u32, 1u32), (2, 1), (13, 7), (23, 12), (25, 13), (26, 13)]
        {
            assert_eq!(
                json_turn.div_ceil(2),
                expected_round,
                "json_turn={json_turn}"
            );
        }
    }
}
