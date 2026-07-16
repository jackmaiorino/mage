//! Branch differential campaign -- KERNEL SIDE (Sol #100 design). Companion
//! to `examples/branch_diff.rs` (the Sol #89/#91 pilot), but a genuinely
//! separate, independent driver: not one line of `branch_diff.rs` changes
//! for this file, same "duplicate rather than share" convention that file's
//! own doc already established relative to `replay_burn_v2.rs` (examples
//! are separate binary crates -- there is no way to `use` one example from
//! another even if we wanted to). This file duplicates the phase-1
//! prefix-replay machinery it needs from `branch_diff.rs` verbatim
//! (proven, frozen logic -- see that file's own citations) and adds three
//! genuinely new things:
//!
//! 1. **Canonical schema mirror** (`canonical_stack_entries`,
//!    `canonical_state_json`): mirrors `LiveCheckpointRecorder.
//!    appendCanonicalStack`/`canonicalStackObjectId`
//!    (`Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/
//!    LiveCheckpointRecorder.java:328-451`, read in full this session) --
//!    one `stack#N` entry per stack item, walked bottom-up, with
//!    source/controller/rule/targets/modes fields. See
//!    `canonical_stack_entries`'s own doc for the exact field-by-field
//!    fidelity accounting (what's a genuine mirror vs a documented,
//!    unavoidable gap). Battlefield canonicalization: confirmed this
//!    session (via `git log`/`git diff` on `LiveCheckpointRecorder.java`,
//!    currently clean -- nothing mid-edit) that Java has **not**
//!    implemented any battlefield canonicalization yet (still raw, sorted-
//!    by-random-UUID `id:name:ctrl:tap:zcc`, `LiveCheckpointRecorder.
//!    java:307-320`) -- there is nothing new to mirror there. This driver
//!    keeps `branch_diff.rs`'s existing name-keyed, UUID-free battlefield
//!    rendering (already more cross-engine-comparable than Java's current
//!    one). If the Java agent adds positional battlefield canonicalization
//!    later, `canonical_state_json`'s `describe_permanent` closure is the
//!    one place to update -- re-check `LiveCheckpointRecorder.java`'s
//!    `compactState` before this driver's next revision.
//!
//! 2. **Shared semantic sub-choice policy** (`shared_semantic_policy_index`):
//!    mirrors `LiveCheckpointBranchMiner.sharedSemanticPolicyIndices`
//!    (`LiveCheckpointBranchMiner.java:4172-4191`, read in full this
//!    session) exactly as an *algorithm* -- linear scan, lexicographically-
//!    first candidate text wins, strict `<` so a tie is broken by the
//!    FIRST (lowest-index) occurrence, never a later one. Applied over this
//!    driver's own canonical per-decision text (`decision_texts`), built to
//!    mirror Java's real `candidateTexts` shape as closely as the evidence
//!    gathered this session allows -- see `decision_texts`'s doc for the
//!    per-decision-kind confirmed-vs-best-effort accounting. This is now
//!    the *only* sub-choice policy this driver needs, applied uniformly to
//!    every decision kind: `sharedSemanticPolicyIndices` is hard-coded to
//!    return at most ONE index (`Collections.singletonList`), and every
//!    decision this driver can meet -- including `engine::Decision::Discard`
//!    -- is now genuinely single-pick too. Before the cross-engine campaign
//!    round-1 fix (Pattern A), `Decision::Discard` could carry `count > 1`
//!    in one batched answer (Faithless Looting's 2-card discard, cleanup's
//!    discard-to-7), which this driver used to answer with its own
//!    UNVERIFIED multi-index generalization (`shared_semantic_policy_top_n`,
//!    removed this round) rather than a confirmed mirror of real Java
//!    behavior for that shape; `HarnessSurfaceV2`'s `DiscardReshape` now
//!    decomposes that batch into `count` sequential single-pick decisions
//!    before it ever reaches here, closing the gap instead of working
//!    around it.
//!
//! 3. **Instrumented walk mode**: `(game, record_id)` addressing --
//!    `WalkSpec::record_id` is the v5-schema join key
//!    (`GameLogger.nextRecordId()`/`ComputerPlayerRL.java:3341-3375`,
//!    threaded into every `REPLAY_DECISION_JSON` line as `record_id` and
//!    into `LiveCheckpointRecorder`'s `manifest.csv`), not the old pilot's
//!    `target_forced_call_index`. This driver locates the root decision by
//!    searching the trace's per-player decision queues for a record whose
//!    `record_id` matches (`trace::DecisionRecord::record_id`, added this
//!    session with `#[serde(default)]` so pre-v5 fixtures still parse),
//!    derives `target_forced_call_index` from its position in that queue,
//!    then reuses `branch_diff.rs`'s exact replay-to-point mechanism
//!    (`ReplayCtx`, `apply_from_trace`, `force_alternate`, etc. --
//!    duplicated below) to reach it via `HarnessSurfaceV2`. From there it
//!    walks forward up to `WalkSpec::max_steps` decisions (default 6,
//!    matching `LiveCheckpointBranchMiner`'s own `--preprobe-max-steps`
//!    default), answering every decision kind it meets (not just the six
//!    `branch_diff.rs`'s `decision_candidates` covers) via the shared
//!    semantic policy, and recording one `WalkStep` per decision: sorted
//!    canonical legal-action multiset, chosen canonical text, and full
//!    canonical state.
//!
//! Output: one JSON object to stdout (`WalkDiffResult`). A companion Python
//! comparator (`local-training/kernel_oracle/walk_diff_compare.py`) reads
//! this alongside the Java run's `preprobe_rng_trace.csv`
//! (`LiveCheckpointBranchMiner`'s `--preprobe-rng-trace` mode output) and
//! reports parity per step.
//!
//! Usage: cargo run --release --example walk_diff -- <corpus_dir> <walk_spec.json>

use mtg_kernel::card_def::{self, FlashbackCost, CARD_DEFS};
use mtg_kernel::engine::{Action, CastMode, Decision, OptionalCostChoice};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{Cost, ManaColor, Pip};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::surface::{SurfaceAction, SurfaceDecision};
use mtg_kernel::surface_v2::HarnessSurfaceV2;
use mtg_kernel::trace::{self, DecisionRecord, GoldenTrace};

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

const DONE: &str = "sentinel:DONE";

#[derive(serde::Deserialize)]
struct WalkSpec {
    branch_id: String,
    trace_file: String,
    /// The root decision's `record_id` (see module doc, point 3) --
    /// addressing by `(game, record_id)`, not a forced-call-index.
    record_id: u32,
    alt_index: usize,
    #[serde(default = "default_max_steps")]
    max_steps: usize,
    #[serde(default)]
    target_action_type: String,
    #[serde(default = "neg_one")]
    target_candidate_count: i64,
}

fn default_max_steps() -> usize {
    6
}
fn neg_one() -> i64 {
    -1
}

#[derive(serde::Serialize, Default)]
struct WalkStep {
    step_index: usize,
    marker: String,
    action_type: String,
    /// `canonicalLegalActionMultiset` mirror: candidate texts, sorted.
    legal_multiset: Vec<String>,
    /// Joined with ", " when the policy picked more than one index (only
    /// `Discard` with `count > 1` -- see module doc point 2).
    chosen_text: String,
    chosen_indices: Vec<usize>,
    /// `true` only for step 0 (the root): its "choice" is the caller-forced
    /// `alt_index`, not the shared semantic policy's own pick.
    forced: bool,
    state: serde_json::Value,
    state_hash: String,
}

#[derive(serde::Serialize)]
struct WalkDiffResult {
    branch_id: String,
    status: String,
    detail: String,
    kernel_version: &'static str,
    record_id: u32,
    target_player: String,
    alt_index: usize,
    max_steps: usize,
    steps: Vec<WalkStep>,
}

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
            .expect("usage: walk_diff <corpus_dir> <walk_spec.json>"),
    );
    let spec_path = PathBuf::from(
        args.next()
            .expect("usage: walk_diff <corpus_dir> <walk_spec.json>"),
    );
    let spec: WalkSpec =
        serde_json::from_str(&std::fs::read_to_string(&spec_path).expect("read walk spec"))
            .expect("parse walk spec");

    let (traces, errors) = trace::load_corpus(&corpus_dir);
    if !errors.is_empty() {
        eprintln!("WARNING: {} corpus parse errors", errors.len());
    }
    let Some(t) = traces
        .iter()
        .find(|t| t.source_path.ends_with(&spec.trace_file))
    else {
        print_result(mk_err(
            &spec,
            "trace_not_found",
            spec.trace_file.clone(),
            String::new(),
        ));
        return;
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| walk_and_diff(t, &spec)));
    match result {
        Ok(r) => print_result(r),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            print_result(mk_err(&spec, "engine_panic", msg, String::new()));
        }
    }
}

fn print_result(r: WalkDiffResult) {
    println!("{}", serde_json::to_string(&r).unwrap());
}

fn mk_err(spec: &WalkSpec, status: &str, detail: String, target_player: String) -> WalkDiffResult {
    WalkDiffResult {
        branch_id: spec.branch_id.clone(),
        status: status.to_string(),
        detail,
        kernel_version: mtg_kernel::KERNEL_VERSION,
        record_id: spec.record_id,
        target_player,
        alt_index: spec.alt_index,
        max_steps: spec.max_steps,
        steps: vec![],
    }
}

// ==================== phase-1 replay-to-branch-point ====================
// Duplicated verbatim from `examples/branch_diff.rs` (proven, frozen logic
// -- see that file's own module doc for the full citation trail); see this
// file's module doc for why duplication (not sharing) is required here.

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

/// Learns kernel `ObjectId`s for trace uuids not yet in `ctx.id_map`
/// (tokens minted mid-game, e.g. Voldaren Epicure's Blood token -- see
/// `build_id_map`'s doc: only opening-hand/library objects are known
/// up front). Heuristic: the first UNBOUND kernel object at or past
/// `pregame_object_count`, since both engines mint/expose new objects
/// while replaying the identical forced trace and should therefore
/// reveal them in the same relative order.
///
/// `expected_controller`: root-caused this session
/// (`trace-candidate-not-in-any-kernel-bucket:CastSpellOrPass`, the
/// campaign's 3-point kernel coverage gap -- see `cast_spell_or_pass_candidates`'s
/// doc). The plain id-order heuristic breaks when two same-named tokens
/// controlled by DIFFERENT players exist unbound at once (e.g. both
/// players' own Voldaren Epicure Blood tokens): a P0 decision's
/// candidate list can expose P0's Blood Token uuid for the first time
/// while P1's, unrelated, Blood Token object already sits unbound too
/// (minted earlier in kernel's own timeline, not yet referenced by any
/// trace decision) -- "first unbound id" then silently grabs whichever
/// token happens to be numerically first, regardless of controller.
/// Every `CastSpellOrPass`/`Discard`/`ChooseCostTargets` candidate is,
/// by construction (`available_activatable_abilities`/hand-only costs),
/// controlled by the deciding player -- passing `Some(player)` for
/// those call sites restricts the search to that player's own objects,
/// which fully disambiguates the case above. `None` (unchanged
/// behavior) for decision kinds whose candidates can legitimately name
/// either player's objects (e.g. `ChooseTargets`).
fn learn_token_ids(
    ctx: &mut ReplayCtx,
    state: &GameState,
    rec: &DecisionRecord,
    expected_controller: Option<PlayerId>,
) {
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
            let Some(next) = state.objects.iter().map(|(id, _)| id).find(|id| {
                id.0 >= ctx.pregame_object_count
                    && !bound.contains(id)
                    && expected_controller.is_none_or(|c| state.objects.get(*id).controller == c)
            }) else {
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

#[allow(clippy::too_many_arguments)]
fn force_alternate(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    rec: &DecisionRecord,
    decision: &SurfaceDecision,
    alt_index: usize,
    id_map: &HashMap<String, ObjectId>,
    seat_uuid: &[Option<String>; 2],
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
    // Own-candidates-only decision kinds get the controller-restricted
    // lookup (see `learn_token_ids`'s doc); `ChooseTargets` can legally
    // name either player's objects, so it keeps the unrestricted search.
    let owns_candidates = matches!(
        decision,
        SurfaceDecision::Decision(
            Decision::CastSpellOrPass { .. }
                | Decision::ChooseCostTargets { .. }
                | Decision::Discard { .. }
        )
    );
    learn_token_ids(ctx, state, rec, owns_candidates.then_some(player));
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
            // at a time -- the reference's genuine shape for every discard.
            // This corpus (recorded before that reshape existed) carries
            // that same real per-card `SELECT_TARGETS` sequence already;
            // apply each pick in turn, same pattern as
            // `replay_burn_v2.rs::apply_discard` (see that function's doc
            // for the full citation, including the corpus's redundant
            // zero-probability terminal `SELECT_CARD` summary consumed but
            // not re-applied below). Looped exactly `count` times (from
            // `state.engine.pending_discard`, stable for this whole
            // obligation's duration -- see `pending_discard_total`'s doc),
            // *not* "while `next_decision` still says Discard": a second,
            // unrelated single-card discard can immediately follow this one
            // and is just as `Decision::Discard`-shaped, indistinguishable
            // from "one more pick of this same batch" by decision kind
            // alone (root-caused this session).
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
                learn_token_ids(ctx, state, rec, Some(player)); // own-hand-only, see learn_token_ids's doc
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

fn decision_player_for(d: &SurfaceDecision, state: &GameState) -> Option<PlayerId> {
    match d {
        SurfaceDecision::Decision(Decision::CastSpellOrPass { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseTargets { player, .. })
        | SurfaceDecision::Decision(Decision::ChooseCostTargets { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareAttackers { player, .. })
        | SurfaceDecision::Decision(Decision::DeclareBlockers { player, .. })
        | SurfaceDecision::Decision(Decision::Discard { player, .. }) => Some(*player),
        SurfaceDecision::DeclareBlockersForAttacker { .. } => Some(state.active_player.opponent()),
        _ => None,
    }
}

// ==================== canonical schema mirror (item 1) ====================

fn mana_symbol(c: ManaColor) -> &'static str {
    match c {
        ManaColor::W => "W",
        ManaColor::U => "U",
        ManaColor::B => "B",
        ManaColor::R => "R",
        ManaColor::G => "G",
        ManaColor::C => "C",
    }
}

/// MTG standard mana-cost notation (`{1}{R}`-style): generic first, then X
/// symbols, then each pip. Verified against real corpus evidence: Highway
/// Robbery's Plot cost (`generic=1, pips=[Colored(R)]`, confirmed by
/// `card_def.rs`'s own `highway_robbery_has_plot_cost` test) renders
/// exactly `"{1}{R}"`, matching the literal `"Plot {1}{R}"` candidate text
/// seen in `local-training/kernel_oracle/burn_mirror_v5_gate/
/// campaign_discovery/summary_offset1.csv`.
fn render_cost(cost: &Cost) -> String {
    let mut s = String::new();
    if cost.generic > 0 {
        s.push_str(&format!("{{{}}}", cost.generic));
    }
    for _ in 0..cost.x_count {
        s.push_str("{X}");
    }
    for pip in cost.pips {
        match pip {
            Pip::Colored(c) => s.push_str(&format!("{{{}}}", mana_symbol(*c))),
            Pip::Hybrid(a, b) => {
                s.push_str(&format!("{{{}/{}}}", mana_symbol(*a), mana_symbol(*b)))
            }
            Pip::Phyrexian(c) => s.push_str(&format!("{{{}/P}}", mana_symbol(*c))),
        }
    }
    if s.is_empty() {
        s.push_str("{0}");
    }
    s
}

/// `FlashbackCost::SacrificeLands(1)` -> `"sacrifice a Mountain"` is
/// confirmed against real evidence (Lava Dart's flashback candidate text,
/// same `summary_offset1.csv` sample: `"Flashback sacrifice a Mountain"`).
/// `SacrificeLands(n > 1)` never occurs in this pool (only Lava Dart has a
/// non-mana flashback cost, always 1) -- the plural rendering below is
/// UNVERIFIED best-effort, flagged for the Java-side agent if a future
/// card ever needs it.
fn render_flashback_cost(fb: FlashbackCost) -> String {
    match fb {
        FlashbackCost::Mana(cost) => render_cost(&cost),
        FlashbackCost::SacrificeLands(1) => "sacrifice a Mountain".to_string(),
        FlashbackCost::SacrificeLands(n) => format!("sacrifice {n} lands"),
    }
}

/// Mirrors `LiveCheckpointRecorder.appendCanonicalStack`/
/// `canonicalStackObjectId` (`LiveCheckpointRecorder.java:328-451`, read in
/// full this session). Kernel's `state.stack` is already stored bottom-up
/// (plain `Vec::push`/`pop` -- see `engine.rs`'s stack mutation call
/// sites), the same order Java derives via `getStack().descendingIterator()`
/// (`SpellStack#push` is `ArrayDeque#addFirst`, so its `descendingIterator`
/// visits tail-to-head = bottom-to-top) -- so `state.stack.iter().enumerate()`
/// needs no reversal to match `stack#N` numbering.
///
/// Field-by-field fidelity accounting against Java's real format
/// (`"stack#<i>:<name>:ctrl=<uuid>:src=<uuid>:rule=<text>:targets=<...>:modes=<...>|"`):
/// - `stack#<i>`, bottom-up ordering: EXACT mirror.
/// - `ctrl=`: Java emits a raw controller UUID (not cross-engine
///   comparable, not even reliably same-engine-reproducible per this
///   corpus's own `manifest.json` `identity_check` audit). This renders
///   the controller's SEAT NAME instead, per the "UUID-label fields
///   excluded, player identity by seat" discipline this pilot's comparator
///   already established -- genuinely comparable content, not a gap.
/// - `src=`: same treatment, rendered as the source object's NAME. Note
///   this collapses to the same value as the leading `<name>` field for
///   every stack item in this pool (kernel has no separate "ability
///   display name distinct from its source permanent" concept -- Java's
///   `StackObject.getName()` can differ from `sourceId`'s own name for a
///   triggered/activated ability, e.g. "Masked Meower's ability" vs
///   "Masked Meower"; this pool's abilities don't exercise that
///   distinction in a way this driver can detect without deeper access).
/// - `rule=`: ALWAYS EMPTY. The kernel has no rules-text templating
///   engine; Java's `Ability.getRule()` output cannot be reconstructed
///   here. Field is present for shape parity, never populated -- a real,
///   documented gap, not a silent guess.
/// - `targets=`: sorted, comma-joined NAME-based descriptors (`target_name`
///   -- the same convention `ChooseTargets` candidates already use)
///   instead of Java's recursive `canonicalStackObjectId` (stack-object
///   target -> `stack#M`, everything else -> raw UUID). This pool's only
///   modal targeting (Pyroblast/Red Elemental Blast's `mode2`) targets
///   permanents only, never a stack object, so the `stack#M`-target case
///   is structurally unreachable here; flagged in case that changes.
/// - `modes=`: `mode#<StackItem::mode_chosen>` unconditionally (`0` for
///   every non-modal card, `0`/`1` for Pyroblast/REB). UNVERIFIED against
///   Java: this session could not confirm whether
///   `ability.getModes().getSelectedModes()` is non-empty (and thus
///   renders `mode#0`) for a *non*-modal ability's single implicit mode,
///   or empty (`modes=` renders as `""`) -- flagged for the Java-side
///   agent to confirm against a real corpus sample; one-line fix either
///   way once known.
fn canonical_stack_entries(state: &GameState, p0_name: &str, p1_name: &str) -> Vec<String> {
    state
        .stack
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let name = &state.objects.get(item.source).name;
            let ctrl = if item.controller == PlayerId::P0 {
                p0_name
            } else {
                p1_name
            };
            let mut targets: Vec<String> = item
                .targets
                .iter()
                .map(|t| target_name(state, t, p0_name, p1_name))
                .collect();
            targets.sort();
            format!(
                "stack#{i}:{name}:ctrl={ctrl}:src={name}:rule=:targets={}:modes=mode#{}",
                targets.join(","),
                item.mode_chosen
            )
        })
        .collect()
}

/// Mirrors `LiveCheckpointRecorder.compactState`'s field set (players,
/// battlefield, stack) in the same shape `branch_diff.rs`'s pilot
/// established, with TWO updates made THIS SESSION after a late-session
/// `git diff` re-check of `LiveCheckpointRecorder.java` (per this module's
/// top doc, point 1's "periodic watch" instruction) caught the Java-side
/// agent landing real work mid-session:
///
/// - `stack`: uses `canonical_stack_entries` (item 1) instead of the
///   pilot's simpler `name(controller=seat)` rendering.
/// - `battlefield`/`hand`/`graveyard`: Java's `LiveCheckpointRecorder.
///   appendCanonicalBattlefield`/`cardsText` (added this session, commit
///   pending -- "Sol #100 canonicalization", diffed via `git diff` on
///   `LiveCheckpointRecorder.java` immediately before this driver's last
///   revision) switched from raw-UUID-keyed/sorted-by-id to POSITIONAL
///   labels: `bf#N` (N = index within the permanent's OWN CONTROLLER's
///   battlefield insertion order, plus newly-added `dmg=`/`counters=`
///   fields) and `hand#N`/`gy#N`/`lib#N` (N = index within that zone's own
///   iteration order). Root cause cited in Java's own doc comment: 3/40 of
///   their 40-point targeting campaign showed state-hash-only mismatches,
///   all Voldaren Epicure/Blood-Token games, traced to the SAME
///   `UUID.randomUUID()` non-reproducibility already fixed for stack
///   objects (Sol #93/#95) but not yet for permanents/tokens -- exactly
///   the "known adjacent risk, not yet investigated" the addendum
///   originally flagged.
///
///   Kernel's `PlayerState::battlefield`/`hand`/`graveyard` are ALREADY
///   plain insertion-ordered `Vec<ObjectId>` (see `state.rs`'s own field
///   docs: battlefield "order a permanent entered, not board position",
///   hand "insertion order, oldest first", graveyard "last element is
///   most-recently-added" -- i.e. oldest-first too, matching a
///   `LinkedHashSet`'s iteration order) -- no re-sort needed, just drop
///   the OLD sorted-by-descriptive-string rendering and use the raw
///   per-player index directly as `bf#i`/`hand#i`/`gy#i`.
///   `controlled_since_turn_start` is kept as this pilot's deliberate,
///   already-documented substitute for Java's raw `zcc=` (zone-change-
///   counter) field -- a different concept (see the ORIGINAL doc this
///   replaces, preserved in git history) -- not something this update
///   changes.
///
///   `library`: kept as a FULL sorted multiset (`library_multiset`,
///   unchanged), a deliberate DIVERGENCE from Java's `library_top`
///   (positionally-labelled but TRUNCATED to 12 cards, top-of-library
///   first). This pilot's full-multiset field is a strictly stronger
///   same-engine deck-consistency check than a 12-card peek would be;
///   changing it to match Java's truncation would lose that value for a
///   field this comparator doesn't even diff yet (Java's
///   `preprobe_rng_trace.csv` carries no per-step state content at all,
///   only an opaque hash -- see `walk_diff_compare.py`'s module doc).
///   Flagged explicitly rather than silently diverging.
fn canonical_state_json(state: &GameState, p0_name: &str, p1_name: &str) -> serde_json::Value {
    let describe_permanent = |i: usize, &id: &ObjectId| {
        let o = state.objects.get(id);
        format!(
            "bf#{i}:{}:tapped={}:controlled_since_turn_start={}:dmg={}:+1/+1={}",
            o.name, o.tapped, !o.summoning_sick, o.damage, o.counters.plus1_plus1
        )
    };
    let describe_player = |p: PlayerId, seat_name: &str| {
        let ps = &state.players[p.index()];
        let battlefield: Vec<String> = ps
            .battlefield
            .iter()
            .enumerate()
            .map(|(i, id)| describe_permanent(i, id))
            .collect();
        let graveyard: Vec<String> = ps
            .graveyard
            .iter()
            .enumerate()
            .map(|(i, &id)| format!("gy#{i}:{}", state.objects.get(id).name))
            .collect();
        let hand: Vec<String> = ps
            .hand
            .iter()
            .enumerate()
            .map(|(i, &id)| format!("hand#{i}:{}", state.objects.get(id).name))
            .collect();
        let mut library: Vec<String> = ps
            .library
            .iter()
            .map(|&id| state.objects.get(id).name.clone())
            .collect();
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
        "stack": canonical_stack_entries(state, p0_name, p1_name),
    })
}

// ==================== shared semantic sub-choice policy (item 2) ====================

/// Mirrors `LiveCheckpointBranchMiner.sharedSemanticPolicyIndices`
/// (`LiveCheckpointBranchMiner.java:4172-4191`) exactly: one linear scan,
/// the lexicographically-first text wins, strict `<` so the FIRST
/// occurrence of a tied minimum is kept (never overwritten by a later
/// equal-valued candidate) -- i.e. ties broken by lowest index. `None`
/// only for an empty candidate list (mirrors Java returning
/// `Collections.emptyList()` there).
fn shared_semantic_policy_index(texts: &[String]) -> Option<usize> {
    let mut best: Option<(usize, &str)> = None;
    for (i, t) in texts.iter().enumerate() {
        let take = match best {
            None => true,
            Some((_, bt)) => t.as_str() < bt,
        };
        if take {
            best = Some((i, t.as_str()));
        }
    }
    best.map(|(i, _)| i)
}

// ==================== uniform decision -> canonical text (item 3) ====================

/// Native (unsorted) candidate texts for a `CastSpellOrPass` window,
/// mirroring Java's real `candidateTexts` shape as closely as this
/// session's evidence allows. CONFIRMED against real corpus/comparator
/// evidence (`local-training/kernel_oracle/branch_diff_compare.py`'s
/// `bucket_activate_candidates`/`MANA_RE`, and literal samples in
/// `burn_mirror_v5_gate/campaign_discovery/summary_offset1.csv`):
/// `"Pass"`, `"Play <name>"`, `"Cast <name>"`, `"Cast <name> using Plot"`,
/// `"Flashback <cost-or-sacrifice-text>"`, `"Plot <cost>"`, and (Mountain
/// being this pool's only mana source) the fixed `"{T}: Add {R}."`.
/// UNVERIFIED / documented gap: non-mana activated abilities
/// (`activatable_abilities` -- Masked Meower's, the Blood token's) render
/// as a kernel-native `"activate:<name>:<idx>"` placeholder; Java's real
/// text for these cannot be reconstructed from evidence gathered this
/// session (the existing comparator already treats this same case as an
/// unbucketable "OTHER" free-text category, warning-only, not a hard
/// mismatch -- see `bucket_activate_candidates`'s own doc).
///
/// DEDUPES by produced text, first occurrence wins -- matching the
/// dedup granularity `cast_spell_or_pass_candidates` (phase-1's proven,
/// trace-validated builder, duplicated above from `branch_diff.rs`)
/// already applies via its `BTreeMap::or_insert`: two untapped Mountains
/// in hand are ONE "Play Mountain" candidate, not two, because which
/// specific (interchangeable) card object plays is not itself a Java-
/// visible decision. Bug found and fixed THIS session via a live
/// cross-check (`local-training/kernel_oracle/burn_mirror_v5/
/// game_20260714_062854_0001.txt` record_id=46: kernel produced 9 raw
/// candidates for a decision Java's own trace records as exactly 6,
/// because an earlier version of this function pushed one entry per
/// `ObjectId` with no dedup at all) -- without this, this driver's own
/// `legal_multiset` output would spuriously "mismatch" Java's real,
/// deduped candidate count/shape on every decision with 2+ same-named
/// lands or mana sources, a false positive with nothing to do with real
/// engine divergence.
fn cast_spell_or_pass_native(
    state: &GameState,
    land_drops: &[ObjectId],
    castable_spells: &[ObjectId],
    mana_abilities: &[ObjectId],
    activatable_abilities: &[(ObjectId, u8)],
    plot_actions: &[ObjectId],
) -> Vec<(String, KernelChoice)> {
    let mut out: Vec<(String, KernelChoice)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let push = |out: &mut Vec<(String, KernelChoice)>,
                seen: &mut std::collections::HashSet<String>,
                text: String,
                choice: KernelChoice| {
        if seen.insert(text.clone()) {
            out.push((text, choice));
        }
    };
    push(&mut out, &mut seen, "Pass".to_string(), KernelChoice::Pass);
    for &id in land_drops {
        push(
            &mut out,
            &mut seen,
            format!("Play {}", state.objects.get(id).name),
            KernelChoice::PlayLand(id),
        );
    }
    for &id in castable_spells {
        let name = state.objects.get(id).name.clone();
        let text = match state.objects.get(id).zone {
            Zone::Graveyard => {
                let cd = &CARD_DEFS[state.objects.get(id).card_def as usize];
                let fb = cd
                    .flashback
                    .as_ref()
                    .map(|f| f.cost)
                    .unwrap_or(FlashbackCost::SacrificeLands(0));
                format!("Flashback {}", render_flashback_cost(fb))
            }
            Zone::Exile => format!("Cast {name} using Plot"),
            _ => format!("Cast {name}"),
        };
        push(&mut out, &mut seen, text, KernelChoice::CastSpell(id));
    }
    for &id in mana_abilities {
        push(
            &mut out,
            &mut seen,
            "{T}: Add {R}.".to_string(),
            KernelChoice::ActivateMana(id),
        );
    }
    for &(id, idx) in activatable_abilities {
        push(
            &mut out,
            &mut seen,
            render_activated_ability_text(state, id, idx),
            KernelChoice::ActivateAbility(id, idx),
        );
    }
    for &id in plot_actions {
        let cd = &CARD_DEFS[state.objects.get(id).card_def as usize];
        let cost = cd.plot_cost.unwrap_or(Cost::zero());
        push(
            &mut out,
            &mut seen,
            format!("Plot {}", render_cost(&cost)),
            KernelChoice::PlotSpell(id),
        );
    }
    out
}

/// Real rules text for a non-mana activated ability (Masked Meower's,
/// Blood Token's -- the only two in this pool, `card_def::ActivatedAbilityDef`
/// has no text field of its own, per `build.rs`'s own codegen doc). Verified
/// against Java: `MaskedMeower.java` ("Discard a card, Sacrifice this
/// creature: Draw a card.") and `Mage/.../token/BloodToken.java` ("{1},
/// {T}, Discard a card, Sacrifice this artifact: Draw a card.") -- both
/// confirmed reachable verbatim in the live cross-engine oracle's own
/// `legal_multiset` output (Blood Token's ability renders correctly at the
/// walk's forced ROOT step, which reads its text straight from the trace
/// record rather than computing it -- see `walk_and_diff`'s doc). Rendered
/// here from `ActivatedAbilityDef::cost` (`CostComponent` already carries
/// everything a cost-side render needs) joined by `", "`, plus a hardcoded
/// `": Draw a card."` suffix: both abilities in this pool resolve to the
/// same effect (`card_def::ability_effect_draw_one`, per `build.rs`'s
/// `activated_abilities_for` doc: "both reduce to ... so both share
/// ability_effect_draw_one"), and `ActivatedAbilityDef::effect` is a bare
/// function pointer with no text of its own to derive a suffix from
/// generically -- a real gap if this pool ever grows a second activated-
/// ability effect, flagged here rather than silently assumed to generalize.
///
/// Previously this fell back to a kernel-native `"activate:<name>:<idx>"`
/// placeholder (cross-engine campaign round 1, Pattern C): every window
/// where Masked Meower's own ability competed against other real-text
/// candidates had its lexicographic shared-semantic-policy pick
/// (`shared_semantic_policy_index`'s doc) flipped by that placeholder's
/// wrong sort position relative to Java's real text.
fn render_activated_ability_text(state: &GameState, id: ObjectId, ability_idx: u8) -> String {
    let name = state.objects.get(id).name.as_str();
    let def = &CARD_DEFS[state.objects.get(id).card_def as usize];
    let cost = def.activated_abilities[ability_idx as usize].cost;
    let parts: Vec<String> = cost
        .iter()
        .map(|c| match c {
            card_def::CostComponent::Mana(cost) => render_cost(cost),
            card_def::CostComponent::Tap => "{T}".to_string(),
            // `mage.abilities.costs.common.SacrificeSourceCost`'s own
            // constructor sets a literal, UNRESOLVED default text --
            // `"sacrifice {this}"` (verified in Mage core: the `{this}`
            // template placeholder is never substituted unless the card
            // calls `.setText(...)` itself). Masked Meower's ability
            // (`MaskedMeower.java`) never does; Blood Token's does
            // (`BloodToken.java`: `.setText("Sacrifice this artifact")`).
            // Root-caused against the live cross-engine oracle this round:
            // an earlier, "nicer"-English version of this branch
            // (`has_type(Creature)` -> "Sacrifice this creature") was
            // plausible but wrong -- Java's own real text for Masked
            // Meower is the literal, un-templated `"Sacrifice {this}"`.
            card_def::CostComponent::SacrificeSelf if name == "Blood Token" => {
                "Sacrifice this artifact".to_string()
            }
            card_def::CostComponent::SacrificeSelf => "Sacrifice {this}".to_string(),
            card_def::CostComponent::ExileSelf => "Exile this".to_string(),
            card_def::CostComponent::DiscardCards(1) => "Discard a card".to_string(),
            card_def::CostComponent::DiscardCards(n) => format!("Discard {n} cards"),
            card_def::CostComponent::SacrificeLands(1) => "Sacrifice a land".to_string(),
            card_def::CostComponent::SacrificeLands(n) => format!("Sacrifice {n} lands"),
        })
        .collect();
    format!("{}: Draw a card.", parts.join(", "))
}

/// Uniform decision -> `(action_type, native-order candidate texts)`,
/// covering every `Decision`/`SurfaceDecision` variant (not just the six
/// `branch_diff.rs`'s `decision_candidates` renders) -- the post-branch
/// walk (item 3) must answer whatever it meets, including the "silent
/// window" kinds `branch_diff.rs`'s trace-driven prefix replay auto-
/// resolves with heuristics (`ChooseOptionalCost`/`ChooseMadnessCast`/
/// `ChooseCastMode`/`ChooseSpellMode`/`OrderTriggers`): off-trace, Java's
/// real `PreprobeController.onDecision` intercepts and answers ALL of
/// these too via the same shared semantic policy (see module doc point 3),
/// so this driver must build real (if in some cases best-effort) text for
/// them rather than reusing the heuristic auto-resolvers.
///
/// Per-decision-kind text confirmation status:
/// - `CastSpellOrPass`: see `cast_spell_or_pass_native`'s doc.
/// - `ChooseTargets`/`ChooseCostTargets`: CONFIRMED (`target_name`, plain
///   card/player names -- Java's `SELECT_TARGETS` `candidateTexts` are
///   always names too, per `trace.rs`'s own module doc).
/// - `Discard`: CONFIRMED as `SELECT_TARGETS` (plain card names), NOT
///   `SELECT_CARD` -- corrected this round (cross-engine campaign round 1,
///   Pattern A): every real discard pick, cost or effect, even a lone
///   1-card pick, surfaces as a `SELECT_TARGETS` window against the live
///   oracle; a multi-card discard is that many sequential `SELECT_TARGETS`
///   windows, never one batched `SELECT_CARD` pick (see `HarnessSurfaceV2`'s
///   `DiscardReshape`, which now decomposes the engine's still-batched
///   `Decision::Discard` into exactly this shape before it ever reaches
///   here -- `count` is always `1` by the time this function sees it).
/// - `DeclareAttackers`/`DeclareBlockersForAttacker`: CONFIRMED (`"DONE"`
///   sentinel text verified verbatim in `ComputerPlayerRL.java` at the
///   `CombatCandidate.toString()`/`isDone()` sites read this session,
///   e.g. line 12469).
/// - `ChooseMadnessCast`: CONFIRMED (`"Yes"`/`"No"`, hardcoded in
///   `trace.rs`'s own `CHOOSE_USE` synthetic-record construction, sourced
///   from the real `CHOOSE_USE: msg=... decision=YES|NO` log line shape).
/// - `ChooseOptionalCost`: CONFIRMED as a two-stage `CHOOSE_USE` sequence,
///   not one 3-way pick -- corrected this round (Pattern B), against
///   `DoIfCostPaid.apply`'s own `chooseUse` gate and `OrCost.pay`'s
///   `usable.size() == 2` gate (both read in full this session). See
///   `HarnessSurfaceV2`'s `OptionalCostReshape`: the `(discard_payable,
///   sacrifice_payable)` sentinel this function reads below (`(false,
///   false)` = the `Use` gate, `(true, true)` = `Which`) is that reshape's
///   own presentation contract, not a real engine state combination.
/// - `ChooseCastMode`, `ChooseSpellMode`, `OrderTriggers`: UNVERIFIED
///   best-effort placeholder text (see each arm below) -- flagged
///   explicitly for the Java-side agent; these are the decisions
///   `branch_diff.rs`'s own shape-sniffing heuristics already treat as
///   structurally uncertain (see `apply_silent_window`'s doc, ported
///   unchanged above), so this is a pre-existing, not newly-introduced,
///   gap.
fn decision_texts(
    state: &GameState,
    decision: &SurfaceDecision,
    p0_name: &str,
    p1_name: &str,
) -> Option<(&'static str, Vec<String>)> {
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass {
            land_drops,
            castable_spells,
            mana_abilities,
            activatable_abilities,
            plot_actions,
            ..
        }) => Some((
            "ACTIVATE_ABILITY_OR_SPELL",
            cast_spell_or_pass_native(
                state,
                land_drops,
                castable_spells,
                mana_abilities,
                activatable_abilities,
                plot_actions,
            )
            .into_iter()
            .map(|(t, _)| t)
            .collect(),
        )),
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => Some((
            "SELECT_TARGETS",
            legal_targets
                .iter()
                .map(|t| target_name(state, t, p0_name, p1_name))
                .collect(),
        )),
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => Some((
            "SELECT_TARGETS",
            candidates
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect(),
        )),
        SurfaceDecision::Decision(Decision::Discard { choices, .. }) => Some((
            "SELECT_TARGETS",
            choices
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect(),
        )),
        SurfaceDecision::Decision(Decision::DeclareAttackers { eligible, .. }) => {
            let mut v: Vec<String> = eligible
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            v.push("DONE".to_string());
            Some(("DECLARE_ATTACKS", v))
        }
        SurfaceDecision::DeclareBlockersForAttacker { legal_blockers, .. } => {
            let mut v: Vec<String> = legal_blockers
                .iter()
                .map(|&id| state.objects.get(id).name.clone())
                .collect();
            v.push("DONE".to_string());
            Some(("DECLARE_BLOCKS", v))
        }
        SurfaceDecision::Decision(Decision::ChooseMadnessCast { .. }) => {
            Some(("CHOOSE_USE", vec!["Yes".to_string(), "No".to_string()]))
        }
        SurfaceDecision::Decision(Decision::ChooseOptionalCost {
            discard_payable,
            sacrifice_payable,
            ..
        }) => {
            // See this function's doc: `(false, false)` is `HarnessSurfaceV2`'s
            // `OptionalCostReshape` `Use`-stage sentinel (Yes/No: pay this
            // cost at all?); any other combination only ever reaches here
            // as `(true, true)`, the `Which`-stage sentinel (pick between
            // the two payable sub-costs) -- see `OptionalCostReshape`'s doc
            // for why the engine itself never emits a real `Decision::
            // ChooseOptionalCost` with `(false, false)`.
            if !*discard_payable && !*sacrifice_payable {
                Some(("CHOOSE_USE", vec!["Yes".to_string(), "No".to_string()]))
            } else {
                Some((
                    "CHOOSE_USE",
                    vec!["Discard a card".to_string(), "Sacrifice a land".to_string()],
                ))
            }
        }
        SurfaceDecision::Decision(Decision::ChooseCastMode { options, .. }) => Some((
            "CHOOSE_MODE",
            options
                .iter()
                .map(|m| match m {
                    CastMode::Normal => "Pay mana cost".to_string(),
                    CastMode::Alternative => "Pay alternative cost".to_string(),
                })
                .collect(),
        )),
        SurfaceDecision::Decision(Decision::ChooseSpellMode { mode_count, .. }) => Some((
            "CHOOSE_MODE",
            (0..*mode_count).map(|i| format!("mode#{i}")).collect(),
        )),
        SurfaceDecision::Decision(Decision::ChooseEffectOption { option_count, .. }) => Some((
            "CHOOSE_MODE",
            (0..*option_count)
                .map(|i| format!("effect-option#{i}"))
                .collect(),
        )),
        SurfaceDecision::Decision(Decision::OrderTriggers { pending, .. }) => Some((
            "ORDER_TRIGGERS",
            vec![format!("identity_order({})", pending.len())],
        )),
        // Not in the Burn corpus this walker replays (Goblin Bushwhacker/
        // Kicker is Rally-only) -- same binary "CHOOSE_USE" shape as Madness.
        SurfaceDecision::Decision(Decision::ChooseKicker { .. }) => {
            Some(("CHOOSE_USE", vec!["Yes".to_string(), "No".to_string()]))
        }
        SurfaceDecision::Decision(Decision::ChooseSpellCopyPayment { .. })
        | SurfaceDecision::Decision(Decision::ChooseSpellCopyRetarget { .. }) => {
            Some(("CHOOSE_USE", vec!["Yes".to_string(), "No".to_string()]))
        }
        // Not in the Burn corpus this walker replays (Chain Lightning is
        // Rally-only).
        SurfaceDecision::Decision(Decision::Halted { .. }) => None,
        SurfaceDecision::Decision(Decision::GameOver { .. }) => None,
        SurfaceDecision::Decision(Decision::DeclareBlockers { .. }) => None,
    }
}

/// Applies the shared-semantic-policy's chosen index (native order, same as
/// `decision_texts`) to `decision`. Every arm answers a single pick
/// (`indices[0]`/`i0`) -- including `Discard`, whose `count` is always `1`
/// by the time a decision reaches here (see `DiscardReshape`'s doc); the
/// `Discard` arm below still generically applies however many indices it's
/// handed (`picks.len() != *count`), so it stays correct even though `count`
/// is now fixed at 1, without singling that case out from every other arm.
fn apply_by_indices(
    surface: &mut HarnessSurfaceV2,
    state: &mut GameState,
    decision: &SurfaceDecision,
    indices: &[usize],
) -> Result<(), String> {
    let i0 = *indices
        .first()
        .ok_or("apply_by_indices:no-chosen-indices")?;
    match decision {
        SurfaceDecision::Decision(Decision::CastSpellOrPass {
            land_drops,
            castable_spells,
            mana_abilities,
            activatable_abilities,
            plot_actions,
            ..
        }) => {
            let native = cast_spell_or_pass_native(
                state,
                land_drops,
                castable_spells,
                mana_abilities,
                activatable_abilities,
                plot_actions,
            );
            let (_, choice) = native
                .into_iter()
                .nth(i0)
                .ok_or("apply_by_indices:index-out-of-range:CastSpellOrPass")?;
            surface
                .apply(
                    state,
                    SurfaceAction::Action(kernel_choice_to_action(&choice)),
                )
                .map_err(|e| format!("engine-step-error:walk:CastSpellOrPass:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseTargets { legal_targets, .. }) => {
            let t = *legal_targets
                .get(i0)
                .ok_or("apply_by_indices:index-out-of-range:ChooseTargets")?;
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseTarget(t)))
                .map_err(|e| format!("engine-step-error:walk:ChooseTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseCostTargets { candidates, .. }) => {
            let id = *candidates
                .get(i0)
                .ok_or("apply_by_indices:index-out-of-range:ChooseCostTargets")?;
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseCostTarget(id)))
                .map_err(|e| format!("engine-step-error:walk:ChooseCostTargets:{e}"))
        }
        SurfaceDecision::Decision(Decision::Discard { choices, count, .. }) => {
            let picks: Vec<ObjectId> = indices
                .iter()
                .filter_map(|&i| choices.get(i).copied())
                .collect();
            if picks.len() != *count as usize {
                return Err("apply_by_indices:policy-index-count-mismatch:Discard".to_string());
            }
            surface
                .apply(state, SurfaceAction::Action(Action::Discard(picks)))
                .map_err(|e| format!("engine-step-error:walk:Discard:{e}"))
        }
        SurfaceDecision::Decision(Decision::DeclareAttackers { eligible, .. }) => {
            let attackers: Vec<ObjectId> = if i0 < eligible.len() {
                vec![eligible[i0]]
            } else {
                vec![]
            };
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::DeclareAttackers(attackers)),
                )
                .map_err(|e| format!("engine-step-error:walk:DeclareAttackers:{e}"))
        }
        SurfaceDecision::DeclareBlockersForAttacker { legal_blockers, .. } => {
            let blockers: Vec<ObjectId> = if i0 < legal_blockers.len() {
                vec![legal_blockers[i0]]
            } else {
                vec![]
            };
            surface
                .apply(state, SurfaceAction::DeclareBlockersForAttacker(blockers))
                .map_err(|e| format!("engine-step-error:walk:DeclareBlockers:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseMadnessCast { .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::ChooseMadnessCast(i0 == 0)),
            )
            .map_err(|e| format!("engine-step-error:walk:ChooseMadnessCast:{e}")),
        SurfaceDecision::Decision(Decision::ChooseOptionalCost { .. }) => {
            // Answers whichever stage `decision_texts` just rendered
            // (`["Yes","No"]` at `Use`, `["Discard a card","Sacrifice a
            // land"]` at `Which`) -- index 0 is always "yes"/"the first
            // option" in both shapes, so `i0 == 0` is the right generic
            // answer regardless of stage. See `OptionalCostReshape`'s doc.
            surface
                .apply(
                    state,
                    SurfaceAction::Action(Action::ChooseOptionalCostStage(i0 == 0)),
                )
                .map_err(|e| format!("engine-step-error:walk:ChooseOptionalCost:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseCastMode { options, .. }) => {
            let m = *options
                .get(i0)
                .ok_or("apply_by_indices:index-out-of-range:ChooseCastMode")?;
            surface
                .apply(state, SurfaceAction::Action(Action::ChooseCastMode(m)))
                .map_err(|e| format!("engine-step-error:walk:ChooseCastMode:{e}"))
        }
        SurfaceDecision::Decision(Decision::ChooseSpellMode { .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::ChooseSpellMode(i0 as u8)),
            )
            .map_err(|e| format!("engine-step-error:walk:ChooseSpellMode:{e}")),
        SurfaceDecision::Decision(Decision::ChooseEffectOption { .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::ChooseEffectOption(i0 as u16)),
            )
            .map_err(|e| format!("engine-step-error:walk:ChooseEffectOption:{e}")),
        SurfaceDecision::Decision(Decision::OrderTriggers { pending, .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::OrderTriggers((0..pending.len()).collect())),
            )
            .map_err(|e| format!("engine-step-error:walk:OrderTriggers:{e}")),
        SurfaceDecision::Decision(Decision::ChooseKicker { .. }) => surface
            .apply(state, SurfaceAction::Action(Action::ChooseKicker(i0 == 0)))
            .map_err(|e| format!("engine-step-error:walk:ChooseKicker:{e}")),
        SurfaceDecision::Decision(Decision::ChooseSpellCopyPayment { .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::ChooseSpellCopyPayment(i0 == 0)),
            )
            .map_err(|e| format!("engine-step-error:walk:ChooseSpellCopyPayment:{e}")),
        SurfaceDecision::Decision(Decision::ChooseSpellCopyRetarget { .. }) => surface
            .apply(
                state,
                SurfaceAction::Action(Action::ChooseSpellCopyRetarget(i0 == 0)),
            )
            .map_err(|e| format!("engine-step-error:walk:ChooseSpellCopyRetarget:{e}")),
        SurfaceDecision::Decision(Decision::GameOver { .. })
        | SurfaceDecision::Decision(Decision::DeclareBlockers { .. })
        | SurfaceDecision::Decision(Decision::Halted { .. }) => {
            Err("apply_by_indices:unreachable-decision-kind".to_string())
        }
    }
}

// ==================== the walk-and-diff driver ====================

fn walk_and_diff(t: &GoldenTrace, spec: &WalkSpec) -> WalkDiffResult {
    let (p0_name, p1_name) = match seat_names(t) {
        Ok(v) => v,
        Err(e) => return mk_err(spec, "setup_error", e, String::new()),
    };

    let (Some(opening0), Some(opening1)) =
        (t.opening_hand_for(&p0_name), t.opening_hand_for(&p1_name))
    else {
        return mk_err(
            spec,
            "setup_error",
            "no-opening-hand-record".to_string(),
            String::new(),
        );
    };
    let lib0 = match card_ids_for(opening0.hand.iter().chain(opening0.library.iter())) {
        Ok(v) => v,
        Err(e) => return mk_err(spec, "setup_error", e, String::new()),
    };
    let lib1 = match card_ids_for(opening1.hand.iter().chain(opening1.library.iter())) {
        Ok(v) => v,
        Err(e) => return mk_err(spec, "setup_error", e, String::new()),
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

    // Address the root by (game, record_id): find which seat's queue holds
    // it, and its position within that queue (== target_forced_call_index,
    // reusing branch_diff.rs's exact cursor-matching mechanism below).
    let target_lookup = [PlayerId::P0, PlayerId::P1].into_iter().find_map(|seat| {
        ctx.queues[seat.index()]
            .iter()
            .position(|r| r.record_id == spec.record_id)
            .map(|idx| (seat, idx))
    });
    let Some((target_seat, target_forced_call_index)) = target_lookup else {
        return mk_err(
            spec,
            "setup_error",
            format!(
                "record_id={} not found among non-mulligan decisions in this trace",
                spec.record_id
            ),
            String::new(),
        );
    };
    let target_player_name = if target_seat == PlayerId::P0 {
        p0_name.clone()
    } else {
        p1_name.clone()
    };

    let mut surface = HarnessSurfaceV2::new();
    let mut steps: Vec<WalkStep> = Vec::new();

    // ---- phase 1: follow the trace exactly up to the root decision ----
    loop {
        let decision = surface.next_decision(&mut state);
        let player = match decision_player_for(&decision, &state) {
            Some(p) => p,
            None => match decision {
                SurfaceDecision::Decision(Decision::GameOver { .. }) => {
                    return mk_err(
                        spec,
                        "trace_exhausted_before_root",
                        "GameOver reached before the target decision".to_string(),
                        target_player_name,
                    )
                }
                _ => {
                    if let Err(e) =
                        apply_silent_window(&mut surface, &mut state, &decision, &mut ctx)
                    {
                        return mk_err(spec, "prefix_replay_error", e, target_player_name);
                    }
                    continue;
                }
            },
        };
        skip_stale_forced_discards(&state, &mut ctx, player);

        let is_target_call =
            player == target_seat && ctx.cursors[player.index()] == target_forced_call_index;
        if !is_target_call {
            if let Err(e) = apply_from_trace(&mut surface, &mut state, &mut ctx, player, &decision)
            {
                return mk_err(spec, "prefix_replay_error", e, target_player_name);
            }
            continue;
        }

        // No more "skip SELECT_TARGETS previews before the real target"
        // step needed here (removed this round): `HarnessSurfaceV2`'s
        // `DiscardReshape` (Pattern A) makes every `Decision::Discard`
        // single-pick, so it now maps 1:1 onto exactly the one trace record
        // at `target_forced_call_index` -- there is no longer a batch of
        // several trace records sharing one kernel decision to skip ahead
        // through. Keeping that skip would wrongly consume the real target
        // record itself (also `SELECT_TARGETS`-shaped) before this cursor
        // check below ever saw it.
        let Some(&rec) = ctx.next(player) else {
            return mk_err(
                spec,
                "trace_exhausted_before_root",
                "no trace record at target cursor".to_string(),
                target_player_name,
            );
        };
        if rec.record_id != spec.record_id {
            return mk_err(
                spec,
                "alignment_error",
                format!(
                    "record_id_mismatch_at_cursor:expected={}:actual={}",
                    spec.record_id, rec.record_id
                ),
                target_player_name,
            );
        }
        if !spec.target_action_type.is_empty() {
            let expected_kind = match &decision {
                SurfaceDecision::Decision(Decision::CastSpellOrPass { .. }) => {
                    "ACTIVATE_ABILITY_OR_SPELL"
                }
                SurfaceDecision::Decision(Decision::ChooseTargets { .. })
                | SurfaceDecision::Decision(Decision::ChooseCostTargets { .. })
                | SurfaceDecision::Decision(Decision::Discard { .. }) => "SELECT_TARGETS",
                SurfaceDecision::Decision(Decision::DeclareAttackers { .. }) => "DECLARE_ATTACKS",
                SurfaceDecision::DeclareBlockersForAttacker { .. } => "DECLARE_BLOCKS",
                _ => "OTHER",
            };
            if expected_kind != spec.target_action_type
                || rec.action_type != spec.target_action_type
            {
                return mk_err(
                    spec,
                    "alignment_error",
                    format!(
                        "action_type_mismatch:expected={}:kernel={expected_kind}:trace={}",
                        spec.target_action_type, rec.action_type
                    ),
                    target_player_name,
                );
            }
        }
        if spec.target_candidate_count >= 0
            && rec.candidate_count as i64 != spec.target_candidate_count
        {
            return mk_err(
                spec,
                "alignment_error",
                format!(
                    "candidate_count_mismatch:expected={}:actual={}",
                    spec.target_candidate_count, rec.candidate_count
                ),
                target_player_name,
            );
        }

        if let Err(e) = check_state(&state, player, rec) {
            return mk_err(
                spec,
                "alignment_error",
                format!("pre_root_state_mismatch:{e}"),
                target_player_name,
            );
        }
        let root_owns_candidates = matches!(
            decision,
            SurfaceDecision::Decision(
                Decision::CastSpellOrPass { .. }
                    | Decision::ChooseCostTargets { .. }
                    | Decision::Discard { .. }
            )
        );
        learn_token_ids(
            &mut ctx,
            &state,
            rec,
            root_owns_candidates.then_some(player),
        );

        // ---- phase 2: canonical texts + force the alternate ----
        //
        // The ROOT step's `legal_multiset`/`chosen_text` are read DIRECTLY
        // from `rec.candidate_texts` (the trace's own recorded text --
        // literally Java's `candidateTexts` for this exact decision, both
        // engines replaying the identical prefix up to here) rather than
        // from this driver's own `decision_texts` reconstruction. Two
        // reasons, found via a live cross-check this session (see
        // `cast_spell_or_pass_native`'s doc for the bug this avoided):
        // 1. `rec.candidate_texts` IS the ground truth -- no
        //    reconstruction risk at all for this one step, unlike every
        //    off-trace `STEP_N` where no such ground truth exists.
        // 2. `spec.alt_index` is, by this driver's own contract (matching
        //    `branch_diff.rs`'s established `BranchSpec::alt_index`
        //    convention), an index into the TRACE's own candidate order --
        //    the same order `force_alternate` (below) resolves it against
        //    via `cast_spell_or_pass_candidates`'s `trace_keys`. This
        //    driver's OWN native-order reconstruction
        //    (`decision_texts`/`cast_spell_or_pass_native`) is a
        //    DIFFERENT order (grouped by candidate kind, not the trace's
        //    interleaved order) -- indexing into it with the SAME
        //    `alt_index` would silently record the wrong "chosen" text for
        //    whatever was actually applied. Reading `rec.candidate_texts`
        //    directly sidesteps the whole index-space question.
        let root_state = canonical_state_json(&state, &p0_name, &p1_name);
        if spec.alt_index >= rec.candidate_texts.len() {
            return mk_err(
                spec,
                "alignment_error",
                format!(
                    "alt_index_out_of_range:alt_index={}:candidate_count={}",
                    spec.alt_index,
                    rec.candidate_texts.len()
                ),
                target_player_name,
            );
        }
        let mut sorted_multiset = rec.candidate_texts.clone();
        sorted_multiset.sort();
        let chosen_text = rec.candidate_texts[spec.alt_index].clone();

        if let Err(e) = force_alternate(
            &mut surface,
            &mut state,
            rec,
            &decision,
            spec.alt_index,
            &ctx.id_map,
            &ctx.seat_uuid,
        ) {
            return mk_err(spec, "force_error", e, target_player_name);
        }
        steps.push(WalkStep {
            step_index: 0,
            marker: "ROOT".to_string(),
            action_type: rec.action_type.clone(),
            legal_multiset: sorted_multiset,
            chosen_text,
            chosen_indices: vec![spec.alt_index],
            forced: true,
            state_hash: state_hash_of(&root_state),
            state: root_state,
        });
        break;
    }

    // ---- phase 3: off-trace shared-semantic-policy walk ----
    while steps.len() < spec.max_steps {
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
            steps.push(WalkStep {
                step_index: steps.len(),
                marker: format!("STEP_{}", steps.len()),
                action_type: "GAME_OVER".to_string(),
                legal_multiset: vec![],
                chosen_text: winner_name.unwrap_or_else(|| "draw_or_unknown".to_string()),
                chosen_indices: vec![],
                forced: false,
                state_hash: state_hash_of(&over_state),
                state: over_state,
            });
            return finish(
                spec,
                "ok",
                "reached_game_over_during_walk".to_string(),
                target_player_name,
                steps,
            );
        }

        let step_state = canonical_state_json(&state, &p0_name, &p1_name);
        let Some((action_type, texts)) = decision_texts(&state, &decision, &p0_name, &p1_name)
        else {
            return finish(
                spec,
                "walk_blocked",
                "unreachable-decision-kind-mid-walk".to_string(),
                target_player_name,
                steps,
            );
        };
        if texts.is_empty() {
            return finish(
                spec,
                "walk_blocked",
                format!("no-candidates:{action_type}"),
                target_player_name,
                steps,
            );
        }
        let mut sorted_multiset = texts.clone();
        sorted_multiset.sort();

        // `Decision::Discard` no longer needs `shared_semantic_policy_top_n`
        // (the multi-index generalization this module's top doc, point 2,
        // flagged as UNVERIFIED): `HarnessSurfaceV2`'s `DiscardReshape`
        // (cross-engine campaign round 1, Pattern A) now decomposes every
        // multi-card discard into that many sequential single-pick
        // decisions, so `count` is always `1` here, same as every other
        // decision kind -- the plain, Java-confirmed single-index policy
        // applies uniformly.
        let chosen_indices: Vec<usize> = match shared_semantic_policy_index(&texts) {
            Some(i) => vec![i],
            None => vec![],
        };
        if chosen_indices.is_empty() {
            return finish(
                spec,
                "walk_blocked",
                "shared_semantic_policy_produced_no_choice".to_string(),
                target_player_name,
                steps,
            );
        }
        let chosen_text = chosen_indices
            .iter()
            .map(|&i| texts[i].clone())
            .collect::<Vec<_>>()
            .join(", ");
        steps.push(WalkStep {
            step_index: steps.len(),
            marker: format!("STEP_{}", steps.len()),
            action_type: action_type.to_string(),
            legal_multiset: sorted_multiset,
            chosen_text,
            chosen_indices: chosen_indices.clone(),
            forced: false,
            state_hash: state_hash_of(&step_state),
            state: step_state,
        });
        if let Err(e) = apply_by_indices(&mut surface, &mut state, &decision, &chosen_indices) {
            return finish(spec, "walk_engine_error", e, target_player_name, steps);
        }
    }

    finish(spec, "ok", String::new(), target_player_name, steps)
}

fn finish(
    spec: &WalkSpec,
    status: &str,
    detail: String,
    target_player: String,
    steps: Vec<WalkStep>,
) -> WalkDiffResult {
    WalkDiffResult {
        branch_id: spec.branch_id.clone(),
        status: status.to_string(),
        detail,
        kernel_version: mtg_kernel::KERNEL_VERSION,
        record_id: spec.record_id,
        target_player,
        alt_index: spec.alt_index,
        max_steps: spec.max_steps,
        steps,
    }
}
