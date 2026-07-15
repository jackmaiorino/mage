//! Branch differential testing driver (external reviewer protocol, Sol
//! #89/#91 -- pilot). This is a genuinely separate, independent driver from
//! `examples/replay_burn_v2.rs` (H2/v4, frozen at 39/40): not one line of
//! that file changes for this increment, same convention `replay_burn_v2.rs`
//! itself already established relative to `replay_burn.rs` (H1/v3, FROZEN).
//! This file duplicates the pure helper functions it needs (candidate
//! bucketing, id-map building, state-gate checks, canonical snapshotting)
//! rather than sharing them, for the same reason.
//!
//! What this does, precisely: for one pinned decision point (identified by
//! trace file + acting player + a 0-based "forced-call index" -- the same
//! count `BranchOracle.java`'s controller uses, i.e. the player's Nth
//! non-mulligan decision), replay the named golden trace *exactly* (same
//! machinery as `replay_burn_v2.rs`) up to that decision, verify the
//! candidate set the kernel independently derives there still matches the
//! trace's own recorded candidate set (ground truth that nothing has
//! silently diverged before the branch), then force the caller-supplied
//! *alternate* candidate index instead of the trace's own choice. From that
//! point the trace is abandoned (we are off-trace by construction): the
//! kernel calls `HarnessSurfaceV2::next_decision` on its own to reach the
//! next rules decision and emits a canonical state snapshot plus that
//! decision's own candidate set. For a continuation window
//! (`continue_steps` > 0) it repeats this for further decisions under a
//! fixed, RNG-free continuation policy (see `fixed_continuation_action`'s
//! doc) -- the kernel-side half of the same policy `BranchOracle.java`
//! implements on the Java side, so both engines can be advanced identically
//! without a shared cross-language PRNG.
//!
//! Output: one JSON object to stdout (see `BranchDiffResult`). A companion
//! Python comparator (`local-training/kernel_oracle/branch_diff_compare.py`)
//! reads this alongside the Java run's `BRANCH_ORACLE_JSON` log lines and
//! reports parity per boundary.
//!
//! Usage: cargo run --release --example branch_diff -- <corpus_dir> <branch_spec.json>

use mtg_kernel::card_def::{self, CARD_DEFS};
use mtg_kernel::engine::{Action, CastMode, Decision, OptionalCostChoice};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::surface::{SurfaceAction, SurfaceDecision};
use mtg_kernel::surface_v2::HarnessSurfaceV2;
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

const DONE: &str = "sentinel:DONE";

#[derive(serde::Deserialize)]
struct BranchSpec {
    branch_id: String,
    trace_file: String,
    target_player: String,
    target_forced_call_index: usize,
    #[serde(default)]
    target_action_type: String,
    #[serde(default)]
    target_candidate_count: i64,
    alt_index: usize,
    #[serde(default)]
    continue_steps: usize,
}

#[derive(serde::Serialize, Default)]
struct Boundary {
    marker: String,
    action_type: String,
    candidate_count: usize,
    candidate_keys: Vec<String>,
    forced_key: String,
    state: serde_json::Value,
    /// FNV-1a hash (hex) of `state`'s canonical JSON string form -- a
    /// same-engine reproducibility/audit anchor, not a cross-engine
    /// equality check (Java hashes with SHA-256 over its own string
    /// serialization; the comparator's own field-by-field structural
    /// diff, not hash equality, is what proves cross-engine parity).
    state_hash: String,
}

#[derive(serde::Serialize)]
struct BranchDiffResult {
    branch_id: String,
    status: String,
    detail: String,
    kernel_version: &'static str,
    boundaries: Vec<Boundary>,
}

/// FNV-1a, 64-bit: std-only (no new crate dependency), deterministic,
/// matches this crate's own "no unordered-map iteration, deterministic
/// transitions" architectural invariant (lib.rs's module doc, Sol #84).
fn fnv1a_hex(s: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn state_hash_of(state_json: &serde_json::Value) -> String {
    fnv1a_hex(&state_json.to_string())
}

fn main() {
    if std::env::var("REPLAY_DEBUG").is_err() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let mut args = std::env::args().skip(1);
    let corpus_dir = PathBuf::from(
        args.next()
            .expect("usage: branch_diff <corpus_dir> <branch_spec.json>"),
    );
    let spec_path = PathBuf::from(
        args.next()
            .expect("usage: branch_diff <corpus_dir> <branch_spec.json>"),
    );
    let spec: BranchSpec =
        serde_json::from_str(&std::fs::read_to_string(&spec_path).expect("read branch spec"))
            .expect("parse branch spec");

    let (traces, errors) = trace::load_corpus(&corpus_dir);
    if !errors.is_empty() {
        eprintln!("WARNING: {} corpus parse errors", errors.len());
    }
    let Some(t) = traces
        .iter()
        .find(|t| t.source_path.ends_with(&spec.trace_file))
    else {
        print_result(BranchDiffResult {
            branch_id: spec.branch_id,
            status: "trace_not_found".to_string(),
            detail: spec.trace_file,
            kernel_version: mtg_kernel::KERNEL_VERSION,
            boundaries: vec![],
        });
        return;
    };

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| branch_and_diff(t, &spec)));
    match result {
        Ok(r) => print_result(r),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            print_result(BranchDiffResult {
                branch_id: spec.branch_id,
                status: "engine_panic".to_string(),
                detail: msg,
                kernel_version: mtg_kernel::KERNEL_VERSION,
                boundaries: vec![],
            });
        }
    }
}

fn print_result(r: BranchDiffResult) {
    println!("{}", serde_json::to_string(&r).unwrap());
}

// ==================== replay-to-branch-point (duplicated from replay_burn_v2.rs; see module doc) ====================

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

fn card_ids_for<'a>(names: impl Iterator<Item = &'a String>) -> Result<Vec<u16>, String> {
    names
        .map(|n| card_def::card_id_by_name(n).ok_or_else(|| format!("setup:unknown-card-name:{n}")))
        .collect()
}

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

fn learn_token_ids(ctx: &mut ReplayCtx, state: &GameState, rec: &DecisionRecord) {
    for raw in rec
        .candidate_object_ids
        .iter()
        .chain(rec.chosen_object_ids.iter())
    {
        for uuid in raw.split("->") {
            if uuid.is_empty()
                || uuid == DONE
                || ctx.id_map.contains_key(uuid)
                || ctx.seat_uuid.iter().any(|s| s.as_deref() == Some(uuid))
            {
                continue;
            }
            let bound: std::collections::HashSet<ObjectId> = ctx.id_map.values().copied().collect();
            let Some(next) = state
                .objects
                .iter()
                .map(|(id, _)| id)
                .find(|id| id.0 >= ctx.pregame_object_count && !bound.contains(id))
            else {
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

/// Skips trace `SELECT_CARD` records for a forced (single-legal-candidate)
/// discard the kernel already auto-applied silently (no real decision was
/// possible), so `ctx.cursors` doesn't desync by leaving a stale record at
/// the front of the queue. Ported unchanged from replay_burn_v2.rs's
/// function of the same name -- see that file for the Madness/Exile-zone
/// root-cause citation. Missing this was the root cause of an off-by-N
/// cursor drift found while validating this pilot (a forced discard
/// elsewhere in the game shifted every later branch-point lookup).
fn skip_stale_forced_discards(state: &GameState, ctx: &mut ReplayCtx, player: PlayerId) {
    loop {
        let Some(&rec) = ctx.next(player) else { return };
        if rec.action_type != "SELECT_CARD" || rec.chosen_object_ids.is_empty() {
            return;
        }
        let already_applied = rec
            .chosen_object_ids
            .iter()
            .all(|uuid| match ctx.id_map.get(uuid) {
                Some(&id) => matches!(state.objects.get(id).zone, Zone::Graveyard | Zone::Exile),
                None => false,
            });
        if !already_applied {
            return;
        }
        ctx.advance(player);
    }
}

fn expected_phase_strings(step: mtg_kernel::state::Step) -> &'static [&'static str] {
    use mtg_kernel::state::Step;
    match step {
        Step::Main1 => &["Precombat Main"],
        Step::Main2 => &["Postcombat Main"],
        Step::BeginCombat
        | Step::DeclareAttackers
        | Step::DeclareBlockers
        | Step::CombatDamage
        | Step::EndCombat => &["Combat"],
        Step::End | Step::Cleanup => &["End"],
        Step::Untap | Step::Upkeep | Step::Draw => &[],
    }
}

fn check_state(state: &GameState, player: PlayerId, rec: &DecisionRecord) -> Result<(), String> {
    let ps = &state.players[player.index()];
    let expected_round = rec.turn.div_ceil(2);
    if state.turn != expected_round {
        return Err("turn-mismatch".to_string());
    }
    if ps.hand.len() != rec.hand.len() {
        return Err("zone-size-mismatch:hand".to_string());
    }
    if ps.library.len() != rec.library.len() {
        return Err("zone-size-mismatch:library".to_string());
    }
    if ps.graveyard.len() != rec.graveyard.len() {
        return Err("zone-size-mismatch:graveyard".to_string());
    }
    if ps.life != rec.life {
        return Err("life-mismatch:own".to_string());
    }
    let opp = &state.players[player.opponent().index()];
    if opp.life != rec.opp_life {
        return Err("life-mismatch:opponent".to_string());
    }
    let expected_phases = expected_phase_strings(state.step);
    if !expected_phases.is_empty()
        && !rec.phase.is_empty()
        && !expected_phases.contains(&rec.phase.as_str())
    {
        return Err(format!(
            "phase-mismatch:kernel_step={:?}:trace_phase={}",
            state.step, rec.phase
        ));
    }
    Ok(())
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

fn target_key(t: &Target) -> String {
    match t {
        Target::Player(p) => format!("P{}", p.index()),
        Target::Object(id) => format!("O{}", id.0),
    }
}

/// Named (not opaque-id) rendering of a legal target, for cross-engine
/// comparison against Java's `candidate_texts` (which are always names).
fn target_name(state: &GameState, t: &Target, p0_name: &str, p1_name: &str) -> String {
    match t {
        Target::Player(p) => {
            if *p == PlayerId::P0 {
                p0_name.to_string()
            } else {
                p1_name.to_string()
            }
        }
        Target::Object(id) => state.objects.get(*id).name.clone(),
    }
}

enum KernelChoice {
    Pass,
    PlayLand(ObjectId),
    CastSpell(ObjectId),
    ActivateMana(ObjectId),
    ActivateAbility(ObjectId, u8),
    PlotSpell(ObjectId),
}

/// Builds (in trace-index order) the kernel's own canonical key for each of
/// the *trace's* candidates at a `CastSpellOrPass` decision, plus the
/// concrete `KernelChoice` action to take for a given index. Shared by the
/// prefix-replay path (index = `rec.chosen_indices[0]`) and the branch path
/// (index = the caller-supplied alternate).
#[allow(clippy::too_many_arguments)]
fn cast_spell_or_pass_candidates(
    state: &GameState,
    rec: &DecisionRecord,
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    land_drops: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    plot_actions: &[ObjectId],
    id_map: &HashMap<String, ObjectId>,
) -> Result<(BTreeMap<String, KernelChoice>, Vec<String>), String> {
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

    let trace_ids = translate_object_candidates(rec, id_map, "CastSpellOrPass")?;
    let mut trace_keys = Vec::with_capacity(trace_ids.len());
    for (id, text) in trace_ids.iter().zip(rec.candidate_texts.iter()) {
        let key = match id {
            None => "pass".to_string(),
            Some(oid) => candidate_key(
                state,
                *oid,
                text,
                land_drops,
                castable_spells,
                mana_abilities,
                activatable_abilities,
                plot_actions,
            )
            .ok_or_else(|| {
                "trace-candidate-not-in-any-kernel-bucket:CastSpellOrPass".to_string()
            })?,
        };
        trace_keys.push(key);
    }
    Ok((by_key, trace_keys))
}

// ==================== canonical state (mirrors BranchOracle.java's schema) ====================

fn canonical_state_json(state: &GameState, p0_name: &str, p1_name: &str) -> serde_json::Value {
    // Canonicalized as `controlled_since_turn_start` (not a creature-only
    // "sick" flag) per Sol #89/#91 amendment: the reviewer's reason is that
    // control tenure matters even for a non-creature permanent that could
    // later become a creature (e.g. an Equipment/Vehicle interaction, or a
    // continuous effect granting creature-ness). `summoning_sick` is
    // already tracked type-agnostically at the storage level on both
    // engines (kernel: `GameObject::summoning_sick`, set on every
    // "enters battlefield" path and cleared at its controller's Untap
    // step; Java: `PermanentImpl.controlledFromStartOfControllerTurn`,
    // queried via the public, haste-independent
    // `wasControlledFromStartOfControllerTurn()` -- deliberately NOT
    // `hasSummoningSickness()`, which ORs in haste and would make this
    // field mean something different from "has this object been
    // continuously controlled since your turn began"). This is the exact
    // inverse of `summoning_sick`.
    let describe_permanent = |&id: &ObjectId| {
        let o = state.objects.get(id);
        format!(
            "{}(tapped={},controlled_since_turn_start={},dmg={},+1/+1={})",
            o.name, o.tapped, !o.summoning_sick, o.damage, o.counters.plus1_plus1
        )
    };
    let describe_card = |&id: &ObjectId| state.objects.get(id).name.clone();
    let describe_player = |p: PlayerId, seat_name: &str| {
        let ps = &state.players[p.index()];
        let mut battlefield: Vec<String> = ps.battlefield.iter().map(describe_permanent).collect();
        battlefield.sort();
        let mut graveyard: Vec<String> = ps.graveyard.iter().map(describe_card).collect();
        graveyard.sort();
        let mut hand: Vec<String> = ps.hand.iter().map(describe_card).collect();
        hand.sort();
        let mut library: Vec<String> = ps.library.iter().map(describe_card).collect();
        library.sort();
        serde_json::json!({
            "seat": seat_name,
            "life": ps.life,
            "has_lost": ps.has_lost,
            "battlefield": battlefield,
            "graveyard": graveyard,
            "hand": hand,
            "library_multiset": library,
        })
    };
    let stack: Vec<String> = state
        .stack
        .iter()
        .map(|s| {
            let ctrl_name = if s.controller == PlayerId::P0 {
                p0_name
            } else {
                p1_name
            };
            format!(
                "{}(controller={})",
                state.objects.get(s.source).name,
                ctrl_name
            )
        })
        .collect();
    let active_name = if state.active_player == PlayerId::P0 {
        p0_name
    } else {
        p1_name
    };
    serde_json::json!({
        "turn": state.turn,
        "phase": format!("{:?}", state.step),
        "active_player": active_name,
        "players": [describe_player(PlayerId::P0, p0_name), describe_player(PlayerId::P1, p1_name)],
        "stack": stack,
    })
}

/// Renders a `SurfaceDecision`'s own candidate set as a sorted list of
/// canonical keys, plus its declared `action_type` in the same 6-value
/// vocabulary `BranchOracle.java`/the corpus already use. Returns `None` for
/// `GameOver` (handled by the caller).
fn decision_candidates(
    state: &GameState,
    decision: &SurfaceDecision,
    p0_name: &str,
    p1_name: &str,
) -> Option<(String, Vec<String>)> {
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass {
            castable_spells,
            mana_abilities,
            land_drops,
            activatable_abilities,
            plot_actions,
            ..
        }) => {
            let mut keys = vec!["pass".to_string()];
            for &id in land_drops {
                keys.push(format!("land:{}", state.objects.get(id).name));
            }
            for &id in castable_spells {
                keys.push(format!(
                    "cast:{}:{}",
                    state.objects.get(id).name,
                    cast_zone_tag(state, id)
                ));
            }
            for &id in mana_abilities {
                keys.push(format!("mana:{}", state.objects.get(id).name));
            }
            for &(id, idx) in activatable_abilities {
                keys.push(format!("activate:{}:{idx}", state.objects.get(id).name));
            }
            for &id in plot_actions {
                keys.push(format!("plot:{}", state.objects.get(id).name));
            }
            keys.sort();
            keys.dedup();
            Some(("ACTIVATE_ABILITY_OR_SPELL".to_string(), keys))
        }
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => {
            let mut keys: Vec<String> = legal_targets
                .iter()
                .map(|t| target_name(state, t, p0_name, p1_name))
                .collect();
            keys.sort();
            Some(("SELECT_TARGETS".to_string(), keys))
        }
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => {
            let mut keys: Vec<String> = candidates
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            keys.sort();
            Some(("SELECT_TARGETS".to_string(), keys))
        }
        SurfaceDecision::Decision(Decision::Discard { choices, .. }) => {
            let mut keys: Vec<String> = choices
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            keys.sort();
            Some(("SELECT_CARD".to_string(), keys))
        }
        SurfaceDecision::Decision(Decision::DeclareAttackers { eligible, .. }) => {
            let mut keys: Vec<String> = eligible
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            keys.push("DONE".to_string());
            keys.sort();
            Some(("DECLARE_ATTACKS".to_string(), keys))
        }
        SurfaceDecision::DeclareBlockersForAttacker { legal_blockers, .. } => {
            let mut keys: Vec<String> = legal_blockers
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            keys.push("DONE".to_string());
            keys.sort();
            Some(("DECLARE_BLOCKS".to_string(), keys))
        }
        SurfaceDecision::Decision(Decision::GameOver { .. }) => None,
        // Every other decision kind is a silent (unlogged) window on the
        // reference side (see replay_burn_v2.rs's `run()` doc); the fixed
        // continuation policy resolves these itself (see
        // `fixed_continuation_action`) without surfacing a boundary.
        _ => Some(("OTHER".to_string(), vec![])),
    }
}

/// Fixed, RNG-free continuation policy -- the kernel-side half of the same
/// rule `BranchOracle.Controller.fixedContinuationIndex` implements in Java:
/// unconditionally Pass at a `CastSpellOrPass` window (always legal, always
/// present, semantically unambiguous on both engines); for every other
/// decision kind encountered mid-continuation, pick this engine's own first
/// native candidate. Documented deviation from a literal "seeded"
/// continuation -- see `BranchOracle.java`'s doc for why.
fn fixed_continuation_action(decision: &SurfaceDecision) -> Result<SurfaceAction, String> {
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass { .. }) => {
            Ok(SurfaceAction::Action(Action::Pass))
        }
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => {
            let t = *legal_targets
                .first()
                .ok_or("continuation:no-legal-targets")?;
            Ok(SurfaceAction::Action(Action::ChooseTarget(t)))
        }
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => {
            let id = *candidates
                .first()
                .ok_or("continuation:no-cost-target-candidates")?;
            Ok(SurfaceAction::Action(Action::ChooseCostTarget(id)))
        }
        SurfaceDecision::Decision(Decision::Discard { choices, count, .. }) => {
            let picks: Vec<ObjectId> = choices.iter().take(*count as usize).copied().collect();
            Ok(SurfaceAction::Action(Action::Discard(picks)))
        }
        SurfaceDecision::Decision(Decision::DeclareAttackers { .. }) => {
            Ok(SurfaceAction::Action(Action::DeclareAttackers(vec![])))
        }
        SurfaceDecision::DeclareBlockersForAttacker { .. } => {
            Ok(SurfaceAction::DeclareBlockersForAttacker(vec![]))
        }
        SurfaceDecision::Decision(Decision::ChooseOptionalCost { .. }) => Ok(
            SurfaceAction::Action(Action::ChooseOptionalCost(OptionalCostChoice::Decline)),
        ),
        SurfaceDecision::Decision(Decision::ChooseMadnessCast { .. }) => {
            Ok(SurfaceAction::Action(Action::ChooseMadnessCast(false)))
        }
        SurfaceDecision::Decision(Decision::ChooseCastMode { .. }) => Ok(SurfaceAction::Action(
            Action::ChooseCastMode(CastMode::Normal),
        )),
        SurfaceDecision::Decision(Decision::ChooseKicker { .. }) => {
            Ok(SurfaceAction::Action(Action::ChooseKicker(false)))
        }
        SurfaceDecision::Decision(Decision::OrderTriggers { pending, .. }) => Ok(
            SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())),
        ),
        SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => {
            Err("continuation:unhandled-ChooseSpellMode".to_string())
        }
        SurfaceDecision::Decision(Decision::GameOver { .. }) => {
            Err("continuation:game-already-over".to_string())
        }
        SurfaceDecision::Decision(Decision::DeclareBlockers { .. }) => {
            Err("continuation:unreachable-DeclareBlockers".to_string())
        }
        SurfaceDecision::Decision(Decision::Halted { .. }) => {
            Err("continuation:halted".to_string())
        }
    }
}

// ==================== the branch-and-diff driver ====================

#[allow(clippy::too_many_arguments)]
fn branch_and_diff(t: &GoldenTrace, spec: &BranchSpec) -> BranchDiffResult {
    let mk_err = |status: &str, detail: String| BranchDiffResult {
        branch_id: spec.branch_id.clone(),
        status: status.to_string(),
        detail,
        kernel_version: mtg_kernel::KERNEL_VERSION,
        boundaries: vec![],
    };

    let (p0_name, p1_name) = match seat_names(t) {
        Ok(v) => v,
        Err(e) => return mk_err("setup_error", e),
    };
    let target_seat = if spec.target_player == p0_name {
        PlayerId::P0
    } else if spec.target_player == p1_name {
        PlayerId::P1
    } else {
        return mk_err(
            "setup_error",
            format!(
                "target_player {} not found in trace (p0={p0_name} p1={p1_name})",
                spec.target_player
            ),
        );
    };

    let (Some(opening0), Some(opening1)) =
        (t.opening_hand_for(&p0_name), t.opening_hand_for(&p1_name))
    else {
        return mk_err("setup_error", "no-opening-hand-record".to_string());
    };
    let lib0 = match card_ids_for(opening0.hand.iter().chain(opening0.library.iter())) {
        Ok(v) => v,
        Err(e) => return mk_err("setup_error", e),
    };
    let lib1 = match card_ids_for(opening1.hand.iter().chain(opening1.library.iter())) {
        Ok(v) => v,
        Err(e) => return mk_err("setup_error", e),
    };
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

    let mut surface = HarnessSurfaceV2::new();
    let mut boundaries: Vec<Boundary> = Vec::new();

    // ---- phase 1: follow the trace exactly up to the branch point ----
    loop {
        let decision = surface.next_decision(&mut state);
        let player = match decision_player_for(&decision, &state) {
            Some(p) => p,
            None => match decision {
                SurfaceDecision::Decision(Decision::GameOver { .. }) => {
                    return mk_err(
                        "trace_exhausted_before_branch",
                        "GameOver reached before the target decision".to_string(),
                    )
                }
                _ => {
                    if let Err(e) =
                        apply_silent_window(&mut surface, &mut state, &decision, &mut ctx)
                    {
                        return mk_err("prefix_replay_error", e);
                    }
                    continue;
                }
            },
        };
        skip_stale_forced_discards(&state, &mut ctx, player);

        if std::env::var("BRANCH_DIFF_DEBUG").is_ok() && player == target_seat {
            let kind = match &decision {
                SurfaceDecision::Decision(Decision::CastSpellOrPass { .. }) => "CastSpellOrPass",
                SurfaceDecision::Decision(Decision::ChooseTargets { .. }) => "ChooseTargets",
                SurfaceDecision::Decision(Decision::ChooseCostTargets { .. }) => {
                    "ChooseCostTargets"
                }
                SurfaceDecision::Decision(Decision::Discard { .. }) => "Discard",
                SurfaceDecision::Decision(Decision::DeclareAttackers { .. }) => "DeclareAttackers",
                SurfaceDecision::DeclareBlockersForAttacker { .. } => "DeclareBlockersForAttacker",
                _ => "other",
            };
            let peek = ctx
                .next(player)
                .map(|r| (r.decision_number, r.action_type.as_str(), r.candidate_count));
            eprintln!(
                "DEBUG cursor={} kind={kind} peek={peek:?}",
                ctx.cursors[player.index()]
            );
        }
        // Is this the branch point?
        let is_target_call =
            player == target_seat && ctx.cursors[player.index()] == spec.target_forced_call_index;
        if !is_target_call {
            if let Err(e) = apply_from_trace(&mut surface, &mut state, &mut ctx, player, &decision)
            {
                return mk_err("prefix_replay_error", e);
            }
            continue;
        }

        // Reached the presumed target cursor. For every decision kind this
        // pilot targets except Discard, the cursor position holds exactly
        // the trace record for this decision. A discard cost's "which
        // card(s)" question can be preceded by 0+ SELECT_TARGETS "preview"
        // trace records for this SAME kernel decision (see
        // `apply_from_trace`'s doc) -- the kernel raises exactly one
        // `Decision::Discard` for the whole episode, so the branch-point
        // selector's per-record cursor count lands here, at the first
        // preview, not at the real SELECT_CARD record. Consume any such
        // previews now (informational only, same as `apply_from_trace`) so
        // `rec` below always describes the actual decision.
        if let SurfaceDecision::Decision(Decision::Discard { .. }) = &decision {
            while let Some(&preview) = ctx.next(player) {
                if preview.action_type != "SELECT_TARGETS" {
                    break;
                }
                ctx.advance(player);
            }
        }
        // Cross-check against the trace before forcing anything (fail-closed
        // alignment check, mirrors `BranchOracle.Controller`'s Java-side check).
        let Some(&rec) = ctx.next(player) else {
            return mk_err(
                "trace_exhausted_before_branch",
                "no trace record at target cursor".to_string(),
            );
        };
        if !spec.target_action_type.is_empty() {
            let expected_kind = match &decision {
                SurfaceDecision::Decision(Decision::CastSpellOrPass { .. }) => {
                    "ACTIVATE_ABILITY_OR_SPELL"
                }
                SurfaceDecision::Decision(Decision::ChooseTargets { .. })
                | SurfaceDecision::Decision(Decision::ChooseCostTargets { .. }) => "SELECT_TARGETS",
                SurfaceDecision::Decision(Decision::Discard { .. }) => "SELECT_CARD",
                SurfaceDecision::Decision(Decision::DeclareAttackers { .. }) => "DECLARE_ATTACKS",
                SurfaceDecision::DeclareBlockersForAttacker { .. } => "DECLARE_BLOCKS",
                _ => "OTHER",
            };
            if expected_kind != spec.target_action_type {
                return mk_err(
                    "alignment_error",
                    format!(
                        "action_type_mismatch:expected={}:actual={expected_kind}",
                        spec.target_action_type
                    ),
                );
            }
            if rec.action_type != spec.target_action_type {
                return mk_err(
                    "alignment_error",
                    format!(
                        "trace_action_type_mismatch:expected={}:actual={}",
                        spec.target_action_type, rec.action_type
                    ),
                );
            }
        }
        if spec.target_candidate_count >= 0
            && rec.candidate_count as i64 != spec.target_candidate_count
        {
            return mk_err(
                "alignment_error",
                format!(
                    "candidate_count_mismatch:expected={}:actual={}",
                    spec.target_candidate_count, rec.candidate_count
                ),
            );
        }
        if spec.alt_index >= rec.candidate_count as usize {
            return mk_err(
                "alignment_error",
                format!(
                    "alt_index_out_of_range:altIndex={}:candidateCount={}",
                    spec.alt_index, rec.candidate_count
                ),
            );
        }

        // ---- phase 2: force the alternate ----
        if let Err(e) = check_state(&state, player, rec) {
            return mk_err("alignment_error", format!("pre_branch_state_mismatch:{e}"));
        }
        learn_token_ids(&mut ctx, &state, rec);
        let branched_state = canonical_state_json(&state, &p0_name, &p1_name);
        if let Err(e) = force_alternate(
            &mut surface,
            &mut state,
            rec,
            &decision,
            spec.alt_index,
            &ctx.id_map,
            &ctx.seat_uuid,
            &p0_name,
            &p1_name,
        ) {
            return mk_err("force_error", e);
        }
        boundaries.push(Boundary {
            marker: "BRANCHED".to_string(),
            action_type: rec.action_type.clone(),
            candidate_count: rec.candidate_count as usize,
            candidate_keys: vec![],
            forced_key: format!("alt_index={}", spec.alt_index),
            state_hash: state_hash_of(&branched_state),
            state: branched_state,
        });
        break;
    }

    // ---- phase 3: off-trace. capture the next decision, then optionally continue ----
    let total_boundaries = 1 + spec.continue_steps;
    for step in 1..=total_boundaries {
        let decision = surface.next_decision(&mut state);
        if let SurfaceDecision::Decision(Decision::GameOver { winner }) = &decision {
            let winner_name = winner.map(|p| {
                if p == PlayerId::P0 {
                    p0_name.clone()
                } else {
                    p1_name.clone()
                }
            });
            let over_state = canonical_state_json(&state, &p0_name, &p1_name);
            boundaries.push(Boundary {
                marker: format!("POST_BRANCH_{step}"),
                action_type: "GAME_OVER".to_string(),
                candidate_count: 0,
                candidate_keys: vec![],
                forced_key: winner_name.unwrap_or_else(|| "draw_or_unknown".to_string()),
                state_hash: state_hash_of(&over_state),
                state: over_state,
            });
            return BranchDiffResult {
                branch_id: spec.branch_id.clone(),
                status: "ok".to_string(),
                detail: "reached_game_over_during_capture".to_string(),
                kernel_version: mtg_kernel::KERNEL_VERSION,
                boundaries,
            };
        }
        let Some((action_type, mut keys)) =
            decision_candidates(&state, &decision, &p0_name, &p1_name)
        else {
            return mk_err(
                "capture_error",
                "unexpected None from decision_candidates".to_string(),
            );
        };
        keys.sort();
        let step_state = canonical_state_json(&state, &p0_name, &p1_name);
        boundaries.push(Boundary {
            marker: format!("POST_BRANCH_{step}"),
            action_type,
            candidate_count: keys.len(),
            candidate_keys: keys,
            forced_key: String::new(),
            state_hash: state_hash_of(&step_state),
            state: step_state,
        });
        if step == total_boundaries {
            break;
        }
        // Advance one more decision under the fixed continuation policy.
        let action = match fixed_continuation_action(&decision) {
            Ok(a) => a,
            Err(e) => {
                return BranchDiffResult {
                    branch_id: spec.branch_id.clone(),
                    status: "continuation_blocked".to_string(),
                    detail: e,
                    kernel_version: mtg_kernel::KERNEL_VERSION,
                    boundaries,
                }
            }
        };
        if let Err(e) = surface.apply(&mut state, action) {
            return BranchDiffResult {
                branch_id: spec.branch_id.clone(),
                status: "continuation_engine_error".to_string(),
                detail: e,
                kernel_version: mtg_kernel::KERNEL_VERSION,
                boundaries,
            };
        }
    }

    BranchDiffResult {
        branch_id: spec.branch_id.clone(),
        status: "ok".to_string(),
        detail: String::new(),
        kernel_version: mtg_kernel::KERNEL_VERSION,
        boundaries,
    }
}

fn decision_player_for(d: &SurfaceDecision, state: &GameState) -> Option<PlayerId> {
    match d {
        SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseTargets { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseCostTargets { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareAttackers { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareBlockers { player, .. })
        | SurfaceDecision::Decision(Decision::Discard { player, .. }) => Some(*player),
        SurfaceDecision::DeclareBlockersForAttacker { .. } => Some(state.active_player.opponent()),
        // Silent windows (no trace record ever consumed): treated as
        // "no target player" so the branch-search loop always routes them
        // to `apply_silent_window` regardless of whose turn it is.
        _ => None,
    }
}

/// Applies a decision that is *not* the branch target, by consuming the
/// next trace record exactly as `replay_burn_v2.rs::run()` does. Only the
/// four decision kinds this pilot supports as branch targets are handled
/// here (CastSpellOrPass, ChooseTargets/ChooseCostTargets, Discard,
/// DeclareAttackers/DeclareBlockersForAttacker) -- sufficient for the six
/// strata this pilot exercises (casting, targets, payment-modes,
/// sacrifice/discard, priority pass/act, trigger-adjacent windows all land
/// on one of these four kinds; combat declarations use the fifth).
fn apply_from_trace(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    ctx: &mut ReplayCtx,
    player: PlayerId,
    decision: &SurfaceDecision,
) -> Result<(), String> {
    let &rec = ctx
        .next(player)
        .ok_or_else(|| "trace-exhausted".to_string())?;
    check_state(state, player, rec)?;
    learn_token_ids(ctx, state, rec);
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass {
            castable_spells,
            mana_abilities,
            land_drops,
            activatable_abilities,
            plot_actions,
            ..
        }) => {
            if rec.action_type != "ACTIVATE_ABILITY_OR_SPELL" {
                return Err(format!(
                    "decision-kind-mismatch:CastSpellOrPass-vs-{}",
                    rec.action_type
                ));
            }
            let (by_key, trace_keys) = cast_spell_or_pass_candidates(
                state,
                rec,
                castable_spells,
                mana_abilities,
                land_drops,
                activatable_abilities,
                plot_actions,
                &ctx.id_map,
            )?;
            let mut kernel_keys: Vec<String> = by_key.keys().cloned().collect();
            kernel_keys.sort();
            let mut sorted_trace_keys = trace_keys.clone();
            sorted_trace_keys.sort();
            if kernel_keys != sorted_trace_keys {
                return Err("candidate-multiset-mismatch:CastSpellOrPass".to_string());
            }
            if rec.chosen_indices.len() != 1 {
                return Err("unexpected-chosen-count:CastSpellOrPass".to_string());
            }
            let chosen_key = trace_keys
                .get(rec.chosen_indices[0] as usize)
                .ok_or("chosen-index-out-of-range:CastSpellOrPass")?;
            let action = kernel_choice_to_action(
                by_key
                    .get(chosen_key)
                    .ok_or("chosen-not-in-kernel-candidates:CastSpellOrPass")?,
            );
            ctx.advance(player);
            surface
                .apply(state, SurfaceAction::Action(action))
                .map_err(|e| format!("engine-step-error:CastSpellOrPass:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => {
            if rec.action_type != "SELECT_TARGETS" {
                return Err(format!(
                    "decision-kind-mismatch:ChooseTargets-vs-{}",
                    rec.action_type
                ));
            }
            let target = resolve_trace_target(
                rec,
                legal_targets,
                &ctx.id_map,
                &ctx.seat_uuid,
                rec.chosen_indices.first().copied(),
            )?;
            ctx.advance(player);
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseTarget(target)))
                .map_err(|e| format!("engine-step-error:ChooseTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => {
            if rec.action_type != "SELECT_TARGETS" {
                return Err(format!(
                    "decision-kind-mismatch:ChooseCostTargets-vs-{}",
                    rec.action_type
                ));
            }
            let trace_ids = translate_object_candidates(rec, &ctx.id_map, "ChooseCostTargets")?;
            let mut kernel_keys: Vec<String> =
                candidates.iter().map(|id| format!("O{}", id.0)).collect();
            kernel_keys.sort();
            let mut trace_keys: Vec<String> = Vec::with_capacity(trace_ids.len());
            for id in &trace_ids {
                trace_keys.push(format!(
                    "O{}",
                    id.ok_or("choose-cost-targets-candidate-is-pass")?.0
                ));
            }
            let mut sorted = trace_keys.clone();
            sorted.sort();
            if kernel_keys != sorted {
                return Err("candidate-multiset-mismatch:ChooseCostTargets".to_string());
            }
            let idx = *rec
                .chosen_indices
                .first()
                .ok_or("unexpected-chosen-count:ChooseCostTargets")? as usize;
            let chosen = trace_ids
                .get(idx)
                .copied()
                .flatten()
                .ok_or("chosen-index-out-of-range:ChooseCostTargets")?;
            ctx.advance(player);
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::ChooseCostTarget(chosen)),
                )
                .map_err(|e| format!("engine-step-error:ChooseCostTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::Discard { choices, .. }) => {
            // H2's discard reshape (`surface_v2.rs::DiscardReshape`)
            // presents one real, single-card `SELECT_TARGETS`-shaped pick
            // at a time -- the reference's genuine shape for every discard,
            // confirmed against the live cross-engine oracle. This corpus
            // (recorded before that reshape existed) carries that same real
            // per-card `SELECT_TARGETS` sequence already; apply each pick
            // in turn, same pattern as `replay_burn_v2.rs::apply_discard`
            // (see that function's doc for the full citation, including
            // the corpus's redundant zero-probability terminal
            // `SELECT_CARD` summary consumed but not re-applied below).
            // Looped exactly `count` times (from `state.engine.
            // pending_discard`, stable for this whole obligation's
            // duration -- see `pending_discard_total`'s doc), *not* "while
            // `next_decision` still says Discard": a second, unrelated
            // single-card discard can immediately follow this one and is
            // just as `Decision::Discard`-shaped, indistinguishable from
            // "one more pick of this same batch" by decision kind alone
            // (root-caused this session, see `apply_discard`'s doc).
            let count = HarnessSurfaceV2::pending_discard_total(state)
                .ok_or("no Discard decision is pending")?;
            let mut choices = choices.clone();
            let mut chosen_names: Vec<String> = Vec::new();
            for _ in 0..count {
                let &rec = ctx
                    .next(player)
                    .ok_or_else(|| "trace-exhausted:Discard".to_string())?;
                if rec.action_type != "SELECT_TARGETS" {
                    return Err(format!(
                        "decision-kind-mismatch:Discard-vs-{}",
                        rec.action_type
                    ));
                }
                learn_token_ids(ctx, state, rec);
                let mut kernel_names: Vec<&str> = choices
                    .iter()
                    .map(|&id| state.objects.get(id).name.as_str())
                    .collect();
                kernel_names.sort_unstable();
                let mut trace_names: Vec<&str> =
                    rec.candidate_texts.iter().map(String::as_str).collect();
                trace_names.sort_unstable();
                if kernel_names != trace_names {
                    return Err("candidate-multiset-mismatch:Discard".to_string());
                }
                let &idx = rec
                    .chosen_indices
                    .first()
                    .ok_or("unexpected-chosen-count:Discard")?;
                let name = rec
                    .candidate_texts
                    .get(idx as usize)
                    .cloned()
                    .ok_or("chosen-index-out-of-range:Discard")?;
                let uuid = rec
                    .chosen_object_ids
                    .first()
                    .ok_or("missing-chosen-object-id:Discard")?;
                let chosen_id = ctx
                    .id_map
                    .get(uuid)
                    .copied()
                    .ok_or_else(|| format!("untranslatable-object-id:Discard:{uuid}"))?;
                if !choices.contains(&chosen_id) {
                    return Err("chosen-not-in-kernel-candidates:Discard".to_string());
                }
                ctx.advance(player);
                chosen_names.push(name);
                surface
                    .apply(
                        state,
                        SurfaceAction::Action(Action::Discard(vec![chosen_id])),
                    )
                    .map_err(|e| format!("engine-step-error:Discard:{e}"))?;
                choices.retain(|&id| id != chosen_id);
            }
            if let Some(&rec) = ctx.next(player) {
                if rec.action_type == "SELECT_CARD" {
                    let terminal_names: Vec<String> = rec
                        .chosen_indices
                        .iter()
                        .map(|&idx| {
                            rec.candidate_texts
                                .get(idx as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    "chosen-index-out-of-range:Discard-terminal-summary".to_string()
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    if terminal_names != chosen_names {
                        return Err("discard-terminal-summary-mismatch".to_string());
                    }
                    ctx.advance(player);
                }
            }
            Ok(())
        }
        SurfaceDecision::Decision(Decision::DeclareAttackers { eligible, .. }) => {
            if rec.action_type != "DECLARE_ATTACKS" {
                return Err(format!(
                    "decision-kind-mismatch:DeclareAttackers-vs-{}",
                    rec.action_type
                ));
            }
            let trace_candidates = translate_attacker_like(rec, &ctx.id_map)?;
            let attackers = prefix_before_done(&rec.chosen_indices, &trace_candidates)?;
            let _ = eligible;
            ctx.advance(player);
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::DeclareAttackers(attackers)),
                )
                .map_err(|e| format!("engine-step-error:DeclareAttackers:{e}"))
        }
        SurfaceDecision::DeclareBlockersForAttacker { legal_blockers, .. } => {
            if rec.action_type != "DECLARE_BLOCKS" {
                return Err(format!(
                    "decision-kind-mismatch:DeclareBlockers-vs-{}",
                    rec.action_type
                ));
            }
            let trace_candidates = translate_blocker_like(rec, &ctx.id_map)?;
            let picks: Vec<ObjectId> = prefix_before_done(&rec.chosen_indices, &trace_candidates)?
                .into_iter()
                .map(|(b, _)| b)
                .collect();
            let _ = legal_blockers;
            ctx.advance(player);
            surface
                .apply(state, SurfaceAction::DeclareBlockersForAttacker(picks))
                .map_err(|e| format!("engine-step-error:DeclareBlockers:{e}"))
        }
        SurfaceDecision::Decision(Decision::GameOver { .. }) => {
            Err("unreachable:GameOver-in-apply-from-trace".to_string())
        }
        _ => Err("unsupported-decision-kind-for-prefix-replay".to_string()),
    }
}

fn kernel_choice_to_action(c: &KernelChoice) -> Action {
    match c {
        KernelChoice::Pass => Action::Pass,
        KernelChoice::PlayLand(id) => Action::PlayLand(*id),
        KernelChoice::CastSpell(id) => Action::CastSpell(*id),
        KernelChoice::ActivateMana(id) => Action::ActivateManaAbility(*id),
        KernelChoice::ActivateAbility(id, idx) => Action::ActivateAbility(*id, *idx),
        KernelChoice::PlotSpell(id) => Action::PlotSpell(*id),
    }
}

fn resolve_trace_target(
    rec: &DecisionRecord,
    legal_targets: &[Target],
    id_map: &HashMap<String, ObjectId>,
    seat_uuid: &[Option<String>; 2],
    chosen_index: Option<u32>,
) -> Result<Target, String> {
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
    let idx = chosen_index.ok_or("unexpected-chosen-count:ChooseTargets")? as usize;
    trace_targets
        .get(idx)
        .copied()
        .ok_or_else(|| "chosen-index-out-of-range:ChooseTargets".to_string())
}

fn translate_attacker_like(
    rec: &DecisionRecord,
    id_map: &HashMap<String, ObjectId>,
) -> Result<Vec<Option<ObjectId>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
        } else {
            out.push(Some(id_map.get(uuid).copied().ok_or_else(|| {
                format!("untranslatable-object-id:DeclareAttackers:{uuid}")
            })?));
        }
    }
    Ok(out)
}

fn translate_blocker_like(
    rec: &DecisionRecord,
    id_map: &HashMap<String, ObjectId>,
) -> Result<Vec<Option<(ObjectId, ObjectId)>>, String> {
    let mut out = Vec::with_capacity(rec.candidate_object_ids.len());
    for uuid in &rec.candidate_object_ids {
        if uuid == DONE {
            out.push(None);
            continue;
        }
        let (b, a) = uuid
            .split_once("->")
            .ok_or_else(|| format!("malformed-block-pair:DeclareBlockers:{uuid}"))?;
        let blocker = id_map
            .get(b)
            .copied()
            .ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{b}"))?;
        let attacker = id_map
            .get(a)
            .copied()
            .ok_or_else(|| format!("untranslatable-object-id:DeclareBlockers:{a}"))?;
        out.push(Some((blocker, attacker)));
    }
    Ok(out)
}

fn prefix_before_done<T: Copy>(
    chosen_indices: &[u32],
    candidates: &[Option<T>],
) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    for &idx in chosen_indices {
        match candidates
            .get(idx as usize)
            .ok_or("chosen-index-out-of-range")?
        {
            None => break,
            Some(v) => out.push(*v),
        }
    }
    Ok(out)
}

/// Silent (unlogged) windows the trace never records a decision for --
/// same set `replay_burn_v2.rs::run()` handles, same defaults. See that
/// file's `run()` for the doc/root-cause citations; unchanged here.
fn apply_silent_window(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    decision: &SurfaceDecision,
    ctx: &mut ReplayCtx,
) -> Result<(), String> {
    match decision {
        SurfaceDecision::Decision(Decision::ChooseOptionalCost { player, .. }) => {
            // Real payable flags, not this decision's own -- the H2 surface
            // reshape re-presents `ChooseOptionalCost` with a presentation-
            // only sentinel at its `Use` stage (see `HarnessSurfaceV2::
            // pending_optional_cost_payable`'s doc).
            let (discard_payable, sacrifice_payable) =
                HarnessSurfaceV2::pending_optional_cost_payable(state)
                    .ok_or("no ChooseOptionalCost decision is pending")?;
            let hand_len = state.players[player.index()].hand.len();
            let land_len = state.players[player.index()]
                .battlefield
                .iter()
                .filter(|&&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land)
                .count();
            let next_is_select_targets_with_len = |n: usize| matches!(ctx.next(*player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && rec.candidate_texts.len() == n);
            let next_looks_like_land_refs = matches!(ctx.next(*player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && !rec.candidate_texts.is_empty() && rec.candidate_texts.iter().all(|t| t.ends_with(" (you)")));
            let next_is_lone_select_card_shaped = |want_land: bool| matches!(ctx.next(*player), Some(&rec) if rec.action_type == "SELECT_CARD" && rec.candidate_texts.len() == 1 && rec.candidate_texts[0].ends_with(" (you)") == want_land);
            let looks_like_sacrifice_pick = sacrifice_payable
                && (next_looks_like_land_refs
                    || (!discard_payable && next_is_select_targets_with_len(land_len))
                    || (land_len == 1 && next_is_lone_select_card_shaped(true)));
            let looks_like_discard_pick = discard_payable
                && !looks_like_sacrifice_pick
                && (next_is_select_targets_with_len(hand_len)
                    || (hand_len == 1 && next_is_lone_select_card_shaped(false)));
            let choice = if looks_like_sacrifice_pick {
                OptionalCostChoice::SacrificeLand
            } else if looks_like_discard_pick {
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
        SurfaceDecision::Decision(Decision::ChooseMadnessCast { player, .. }) => {
            let &rec = ctx
                .next(*player)
                .ok_or("trace-exhausted:ChooseMadnessCast")?;
            if rec.action_type != "CHOOSE_USE" {
                return Err(format!(
                    "decision-kind-mismatch:ChooseMadnessCast-vs-{}",
                    rec.action_type
                ));
            }
            let attempt = rec.chosen_indices.first() == Some(&0);
            ctx.advance(*player);
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::ChooseMadnessCast(attempt)),
                )
                .map_err(|e| format!("engine-step-error:ChooseMadnessCast:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseCastMode {
            player, options, ..
        }) => {
            let looks_like_alternative_cost_pick = matches!(ctx.next(*player), Some(&rec) if rec.action_type == "SELECT_TARGETS" && !rec.candidate_texts.is_empty() && rec.candidate_texts.iter().all(|t| t.ends_with(" (you)")));
            let mode =
                if looks_like_alternative_cost_pick && options.contains(&CastMode::Alternative) {
                    CastMode::Alternative
                } else {
                    CastMode::Normal
                };
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseCastMode(mode)))
                .map_err(|e| format!("engine-step-error:ChooseCastMode:{e}"))
        }
        SurfaceDecision::Decision(Decision::OrderTriggers { pending, .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())),
            )
            .map_err(|e| format!("engine-step-error:OrderTriggers:{e}")),
        SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => {
            Err("unhandled-decision:ChooseSpellMode".to_string())
        }
        SurfaceDecision::Decision(Decision::DeclareBlockers { .. }) => {
            Err("unreachable-decision:DeclareBlockers".to_string())
        }
        _ => Err("apply_silent_window-called-on-non-silent-decision".to_string()),
    }
}

/// Forces `spec.alt_index` instead of the trace's own choice at the branch
/// decision. Validates the candidate set against the trace first (same
/// multiset check as `apply_from_trace`) so a mismatch here is a genuine
/// pre-branch divergence, not a forcing bug.
#[allow(clippy::too_many_arguments)]
fn force_alternate(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    rec: &DecisionRecord,
    decision: &SurfaceDecision,
    alt_index: usize,
    id_map: &HashMap<String, ObjectId>,
    seat_uuid: &[Option<String>; 2],
    _p0_name: &str,
    _p1_name: &str,
) -> Result<(), String> {
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass {
            castable_spells,
            mana_abilities,
            land_drops,
            activatable_abilities,
            plot_actions,
            ..
        }) => {
            let (by_key, trace_keys) = cast_spell_or_pass_candidates(
                state,
                rec,
                castable_spells,
                mana_abilities,
                land_drops,
                activatable_abilities,
                plot_actions,
                id_map,
            )?;
            let mut kernel_keys: Vec<String> = by_key.keys().cloned().collect();
            kernel_keys.sort();
            let mut sorted_trace_keys = trace_keys.clone();
            sorted_trace_keys.sort();
            if kernel_keys != sorted_trace_keys {
                return Err("candidate-multiset-mismatch:CastSpellOrPass".to_string());
            }
            let alt_key = trace_keys
                .get(alt_index)
                .ok_or("alt-index-out-of-range:CastSpellOrPass")?;
            let action = kernel_choice_to_action(
                by_key
                    .get(alt_key)
                    .ok_or("alt-not-in-kernel-candidates:CastSpellOrPass")?,
            );
            surface
                .apply(state, SurfaceAction::Action(action))
                .map_err(|e| format!("engine-step-error:force:CastSpellOrPass:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => {
            let target = resolve_trace_target(
                rec,
                legal_targets,
                id_map,
                seat_uuid,
                Some(alt_index as u32),
            )?;
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseTarget(target)))
                .map_err(|e| format!("engine-step-error:force:ChooseTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => {
            let trace_ids = translate_object_candidates(rec, id_map, "ChooseCostTargets")?;
            let mut kernel_keys: Vec<String> =
                candidates.iter().map(|id| format!("O{}", id.0)).collect();
            kernel_keys.sort();
            let mut trace_keys: Vec<String> = Vec::with_capacity(trace_ids.len());
            for id in &trace_ids {
                trace_keys.push(format!(
                    "O{}",
                    id.ok_or("choose-cost-targets-candidate-is-pass")?.0
                ));
            }
            let mut sorted = trace_keys.clone();
            sorted.sort();
            if kernel_keys != sorted {
                return Err("candidate-multiset-mismatch:ChooseCostTargets".to_string());
            }
            let chosen = trace_ids
                .get(alt_index)
                .copied()
                .flatten()
                .ok_or("alt-index-out-of-range:ChooseCostTargets")?;
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::ChooseCostTarget(chosen)),
                )
                .map_err(|e| format!("engine-step-error:force:ChooseCostTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::Discard { choices, .. }) => {
            let mut kernel_names: Vec<&str> = choices
                .iter()
                .map(|&id| state.objects.get(id).name.as_str())
                .collect();
            kernel_names.sort_unstable();
            let mut trace_names: Vec<&str> =
                rec.candidate_texts.iter().map(String::as_str).collect();
            trace_names.sort_unstable();
            if kernel_names != trace_names {
                return Err("candidate-multiset-mismatch:Discard".to_string());
            }
            let uuid = rec
                .candidate_object_ids
                .get(alt_index)
                .ok_or("alt-index-out-of-range:Discard")?;
            let chosen = id_map
                .get(uuid)
                .copied()
                .ok_or_else(|| format!("untranslatable-object-id:Discard:{uuid}"))?;
            if !choices.contains(&chosen) {
                return Err("alt-not-in-kernel-candidates:Discard".to_string());
            }
            surface
                .apply(state, SurfaceAction::Action(Action::Discard(vec![chosen])))
                .map_err(|e| format!("engine-step-error:force:Discard:{e}"))
        }
        SurfaceDecision::Decision(Decision::DeclareAttackers { .. }) => {
            // alt_index may legally point at the DONE sentinel itself (the
            // selector's meaningful alternate to "attack with X" is often
            // "declare nobody") -- translate_attacker_like maps DONE to
            // None, which is the correct "no attackers" action, not an error.
            let trace_candidates = translate_attacker_like(rec, id_map)?;
            let picked: Option<ObjectId> = trace_candidates
                .get(alt_index)
                .copied()
                .ok_or("alt-index-out-of-range:DeclareAttackers")?;
            let attackers: Vec<ObjectId> = picked.into_iter().collect();
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::DeclareAttackers(attackers)),
                )
                .map_err(|e| format!("engine-step-error:force:DeclareAttackers:{e}"))
        }
        SurfaceDecision::DeclareBlockersForAttacker { .. } => {
            let trace_candidates = translate_blocker_like(rec, id_map)?;
            let picked: Option<(ObjectId, ObjectId)> = trace_candidates
                .get(alt_index)
                .copied()
                .ok_or("alt-index-out-of-range:DeclareBlockers")?;
            let blockers: Vec<ObjectId> = picked.map(|(b, _)| b).into_iter().collect();
            surface
                .apply(state, SurfaceAction::DeclareBlockersForAttacker(blockers))
                .map_err(|e| format!("engine-step-error:force:DeclareBlockers:{e}"))
        }
        _ => Err("unsupported-decision-kind-for-branch-target".to_string()),
    }
}
