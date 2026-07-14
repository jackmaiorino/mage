//! Golden-trace ingestion: parses the reference engine's replay-logged games.
//!
//! A golden trace is a game log produced by the Java reference engine under
//! `--deterministic-eval --eval-game-logging --replay-metadata`. The lines we
//! consume:
//!
//! - `REPLAY:` / `REPLAY_RANDOM:` headers — seeds, decks, scenario.
//! - `REPLAY_DECISION_JSON: {...}` — one record per agent decision, carrying the
//!   full candidate list, the chosen action, and a complete snapshot of the
//!   actor's hidden state (hand + library with stable object ids). The first
//!   record's library ordering IS the post-shuffle deck order, which together
//!   with the decision sequence fully specifies the game.
//! - `RESULT:` / winner lines — terminal outcome.
//!
//! The kernel's correctness contract: replaying (initial libraries, decision
//! sequence) must reproduce every intermediate legal-action set and the final
//! outcome. See `comparator` for the equivalence checks.

use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DecisionRecord {
    pub ordinal: u32,
    #[serde(default)]
    pub decision_number: u32,
    /// v5 corpus schema (`local-training/kernel_oracle/
    /// v5_capture_schema_addendum.md`, Sol #95/#96/#98): a per-game,
    /// cross-player, genuinely-incrementing join key -- unlike
    /// `decision_number` (which only *peeks* at a counter and can collide
    /// across action types/players, confirmed in the addendum's own
    /// worked example), `record_id` is unique 1..N across every
    /// `logReplayDecision` call for both seats sharing one `GameLogger`.
    /// This is the field the branch-differential campaign addresses a
    /// walk's root decision by (`(game, record_id)`, not `decision_number`
    /// or a positional "Nth call" index). `#[serde(default)]` so v3/v4-era
    /// fixtures and any pre-record_id corpus still parse (defaulting to 0,
    /// which is never a real record_id -- Java's counter starts at 1).
    #[serde(default)]
    pub record_id: u32,
    pub player: String,
    #[serde(default)]
    pub actor: String,
    #[serde(default)]
    pub active_player: String,
    pub action_type: String,
    pub candidate_count: u32,
    pub candidate_texts: Vec<String>,
    #[serde(default)]
    pub candidate_object_ids: Vec<String>,
    pub chosen_indices: Vec<u32>,
    #[serde(default)]
    pub chosen_texts: Vec<String>,
    /// Object UUIDs of the chosen candidates, in `chosen_indices` order.
    /// Populated for `SELECT_CARD` (`logReplayCardSelection`) so
    /// `skip_stale_forced_discards` (`examples/replay_burn.rs`) can check
    /// whether a discard the trace describes has already been applied.
    #[serde(default)]
    pub chosen_object_ids: Vec<String>,
    #[serde(default)]
    pub selected_index: i64,
    #[serde(default)]
    pub selected_text: String,
    #[serde(default)]
    pub hand: Vec<String>,
    #[serde(default)]
    pub hand_object_ids: Vec<String>,
    #[serde(default)]
    pub library: Vec<String>,
    #[serde(default)]
    pub library_object_ids: Vec<String>,
    #[serde(default)]
    pub graveyard: Vec<String>,
    #[serde(default)]
    pub turn: u32,
    #[serde(default)]
    pub phase: String,
    /// The acting (`player`) seat's own life total at the moment of this
    /// decision. Added for `HarnessSurfaceV2`'s strengthened state gate
    /// (`examples/replay_burn_v2.rs`'s `check_state`) -- H1's `check_state`
    /// (`examples/replay_burn.rs`) never read this field. `#[serde(default)]`
    /// so older/H1-era fixtures without it still parse.
    #[serde(default)]
    pub life: i32,
    /// The non-acting player's life total at the moment of this decision --
    /// see `life`'s doc. Together `(life, opp_life)` are always a reciprocal
    /// pair against the *other* seat's own record at the same instant
    /// (confirmed empirically against the v4 corpus).
    #[serde(default)]
    pub opp_life: i32,
    /// Display name of the spell/ability this decision is being asked on
    /// behalf of (e.g. `"Faithless Looting"`), when the reference logs one --
    /// empty for decision kinds that don't have a single source (`MULLIGAN`,
    /// bare `ACTIVATE_ABILITY_OR_SPELL` priority windows). Used by the H2
    /// corpus invariant validator (`examples/replay_burn_v2.rs`) to group a
    /// multi-slot target sequence (e.g. Faithless Looting's 2-card discard
    /// cost, logged as 2 `SELECT_TARGETS` records) into one semantic
    /// target-choice episode.
    #[serde(default)]
    pub source_name: String,
    #[serde(default)]
    pub random_util_count: u64,
    /// The training episode this decision was actually applied under.
    /// `ComputerPlayerRL` clones itself (sharing the same `GameLogger` by
    /// reference) to run internal legality/lookahead probes before
    /// offering some options; those clones still call `logReplayDecision`
    /// against the shared log, producing phantom records that were never
    /// applied to the real game. Real records always carry the actual
    /// non-negative episode number; phantoms are always `-1`. See
    /// `GoldenTrace::parse_file`, which filters these out before they
    /// reach `decisions` -- callers should never need to check this field
    /// themselves.
    #[serde(default)]
    pub episode: i64,
}

#[derive(Debug, Clone, Default)]
pub struct TraceHeader {
    pub scenario: u64,
    pub seed: u64,
    pub agent_deck: String,
    pub opp_deck: String,
    pub random_util_seed: Option<i128>,
}

#[derive(Debug, Default)]
pub struct GoldenTrace {
    pub header: TraceHeader,
    pub decisions: Vec<DecisionRecord>,
    pub winner: Option<String>,
    pub source_path: String,
    /// Count of `REPLAY_DECISION_JSON` records dropped by `parse_file`
    /// because they were phantom clone-probe records (`episode < 0`) --
    /// see `DecisionRecord::episode`. Exposed so diagnostics can report
    /// how much of the raw log was discarded, without callers needing to
    /// re-derive it.
    pub phantom_decisions_skipped: usize,
}

/// A player's fully-resolved opening hand and remaining library, read off
/// their terminal `MULLIGAN` decision record (see `GoldenTrace::
/// opening_hand_for`). By that point every mulligan/London-bottoming this
/// player took is already reflected in `hand`/`library`, in real order,
/// strictly before any turn-based draw.
#[derive(Debug, Clone, Default)]
pub struct OpeningHand {
    pub hand: Vec<String>,
    pub hand_object_ids: Vec<String>,
    pub library: Vec<String>,
    pub library_object_ids: Vec<String>,
}

impl GoldenTrace {
    pub fn parse_file(path: &Path) -> Result<GoldenTrace, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        parse_text(&text, path.display().to_string())
    }

    /// The agent's post-shuffle opening library order: first record that carries
    /// a full library snapshot (the mulligan decision, before any draw).
    pub fn opening_library(&self) -> Option<(&[String], &[String])> {
        self.decisions
            .iter()
            .find(|d| !d.library.is_empty())
            .map(|d| (d.library.as_slice(), d.library_object_ids.as_slice()))
    }

    /// `player`'s true, final opening hand and remaining library: `hand`/
    /// `library` off their *last* mulligan-phase decision record (the
    /// mulligan loop is `MULLIGAN -> [LONDON_MULLIGAN -> MULLIGAN ->]*`,
    /// usually terminating on a `MULLIGAN` record where "KEEP" was chosen,
    /// so the last one is usually that terminal, fully-resolved-hand
    /// snapshot -- taken strictly before any turn-based draw has
    /// happened).
    ///
    /// This is *not* simply "the first non-mulligan decision record", even
    /// though that record's `hand`/`library` do reflect the post-mulligan
    /// state too: this player's first *logged* real decision can be
    /// several turns later than their mulligan-keep, because the
    /// reference engine only logs an `ACTIVATE_ABILITY_OR_SPELL` decision
    /// when there's a real alternative to passing -- if this player has
    /// nothing playable for a few turns, those turns' natural draws are
    /// already baked into that later record's `hand`/`library`, silently
    /// inflating the "opening" hand by however many turns elapsed
    /// (verified empirically against the real corpus: 6 of the first 10
    /// player-openings checked have a smaller last-MULLIGAN hand than
    /// first-non-mulligan hand, by exactly the elapsed-turn draw count).
    ///
    /// ## The "ends on LONDON_MULLIGAN, no trailing MULLIGAN(KEEP)" case
    ///
    /// `LondonMulligan.mulligan()` (`Mage/src/main/java/mage/game/mulligan/
    /// LondonMulligan.java`) bottoms exactly *one* card per
    /// `chooseLondonMulliganCards` call (routed through `genericChoose`,
    /// which is where the `LONDON_MULLIGAN` record gets logged -- see the
    /// replay comparator's visibility-predicate doc), looping
    /// `while (hand.size() > newHandSize)` where `newHandSize =
    /// startingHandSize - mulligansTaken`. Each call's candidate pool is
    /// the *current* remaining hand, so `candidate_count` strictly
    /// decreases by 1 every call in the same mulligan attempt. The one call
    /// that skips logging entirely is the terminal one whose candidate pool
    /// has already shrunk to exactly 1 card (`genericChoose`'s bare
    /// single-legal-candidate shortcut returns before scoring/logging --
    /// same shortcut documented for `ACTIVATE_ABILITY_OR_SPELL`/
    /// `SELECT_TARGETS`). That can only happen when `mulligansTaken` equals
    /// the starting hand size (candidate_count sequence
    /// startingHandSize, startingHandSize-1, ..., 1): the player has
    /// mulliganed all the way down to an empty hand. At that point
    /// `Mulligan.canTakeMulligan` (`Mage/src/main/java/mage/game/mulligan/
    /// Mulligan.java`) returns `false` (`!player.getHand().isEmpty()` fails,
    /// and `LondonMulligan.canTakeMulligan` also requires
    /// `openingHandSizes > 0`), so `Mulligan.executeMulliganPhase`'s
    /// `while(true)` loop `break`s *without ever calling
    /// `player.chooseMulligan(game)`* -- the player is silently forced to
    /// keep a 0-card hand, and no `MULLIGAN` record (terminal or otherwise)
    /// is ever logged for that keep.
    ///
    /// So: whenever the last mulligan-phase record for a player is a
    /// `LONDON_MULLIGAN` (not a terminal `MULLIGAN`), the true final hand is
    /// empty and every card that record's own `hand` snapshot still shows
    /// (the 1-2 cards left before that final silent single-candidate pick)
    /// ends up on the bottom of the library along with everything already
    /// in `library`. Confirmed against the real v3 corpus: exactly one
    /// player-chain (`game_20260712_194635_0036.txt`, `PlayerRL1`,
    /// `mulligans_taken=7` reaching a 7-card starting hand) has this shape;
    /// its last logged record has `candidate_count=2`,
    /// `hand=["Mountain","Fiery Temper"]`, `library_size=58`
    /// (`58 + 2 == 60`, the full deck) -- i.e. `hand_size + library_size`
    /// is already deck-complete, so appending the 2 leftover `hand` cards
    /// onto `library` (in either order -- both are headed to the bottom
    /// regardless) reconstructs the correct post-keep state.
    pub fn opening_hand_for(&self, player: &str) -> Option<OpeningHand> {
        let rec = self
            .decisions
            .iter()
            .rev()
            .find(|d| d.player == player && (d.action_type == "MULLIGAN" || d.action_type == "LONDON_MULLIGAN"))?;

        if rec.action_type == "MULLIGAN" {
            return Some(OpeningHand {
                hand: rec.hand.clone(),
                hand_object_ids: rec.hand_object_ids.clone(),
                library: rec.library.clone(),
                library_object_ids: rec.library_object_ids.clone(),
            });
        }

        // `rec.action_type == "LONDON_MULLIGAN"`: the silent-forced-empty-
        // keep shape documented above. Every remaining `hand` card is bound
        // for the bottom of the library; see the doc comment for why this
        // is always a full keep-to-empty, never a partial reconstruction.
        let mut library = rec.library.clone();
        library.extend(rec.hand.iter().cloned());
        let mut library_object_ids = rec.library_object_ids.clone();
        library_object_ids.extend(rec.hand_object_ids.iter().cloned());
        Some(OpeningHand { hand: Vec::new(), hand_object_ids: Vec::new(), library, library_object_ids })
    }
}

/// Parses a human-readable decision header line of the shape `DECISION #322
/// - Turn 16 (PlayerRL1 turn), Postcombat Main (CHOOSE_USE) - PlayerRL1`,
/// returning `(decision_number, turn, phase, player)`. Manual token
/// parsing (no regex dependency anywhere in this crate) rather than a
/// regex, matching this module's existing style for `REPLAY:`/
/// `REPLAY_RANDOM:`'s own hand-rolled tokenizing.
fn parse_choose_use_header(line: &str) -> Option<(u32, u32, String, String)> {
    let rest = line.strip_prefix("DECISION #")?;
    let (num_str, rest) = rest.split_once(" - Turn ")?;
    let decision_number: u32 = num_str.parse().ok()?;
    let (turn_str, rest) = rest.split_once(' ')?;
    let turn: u32 = turn_str.parse().ok()?;
    let rest = rest.split_once("), ")?.1;
    let (phase, player) = rest.split_once(" (CHOOSE_USE) - ")?;
    Some((decision_number, turn, phase.to_string(), player.trim().to_string()))
}

/// Parses the data line that always follows a `(CHOOSE_USE)` header a few
/// lines later (confirmed empirically: every corpus file has exactly as
/// many `CHOOSE_USE: msg=...` lines as `(CHOOSE_USE)` headers, no
/// duplicates or phantoms by `decision_number`) -- `CHOOSE_USE: msg="..."
/// outcome=... decision=YES|NO scores=[...]`. Returns `(msg, decision_is_yes)`.
fn parse_choose_use_line(line: &str) -> Option<(String, bool)> {
    let rest = line.strip_prefix("CHOOSE_USE: msg=\"")?;
    let (msg, rest) = rest.split_once("\" outcome=")?;
    let rest = rest.split_once(" decision=")?.1;
    match rest.split_whitespace().next()? {
        "YES" => Some((msg.to_string(), true)),
        "NO" => Some((msg.to_string(), false)),
        _ => None,
    }
}

/// The one substring of `CHOOSE_USE`'s `msg` this parser recognizes as a
/// real, ground-truth-bearing decision to surface into `decisions` --
/// Fiery Temper's Madness offer (`MadnessTriggeredAbility.resolve()`'s
/// `chooseUse` prompt, the only card with Madness in this pool -- see
/// `card_def.rs`'s `fiery_temper_has_madness_r` test). Root-caused (this
/// increment) against `game_20260713_002156_0015.txt` decision 322:
/// `examples/replay_burn_v2.rs`'s prior "always attempt the Madness cast"
/// default was silently wrong here (the reference actually said `NO`) --
/// this text is the *only* place that real yes/no answer is ever logged
/// (Madness offers get no `REPLAY_DECISION_JSON` record at all). The other
/// two `CHOOSE_USE` message shapes in this corpus (Highway Robbery's
/// resolution-time discard-or-sacrifice offer, `engine::Decision::
/// ChooseOptionalCost`) are deliberately *not* matched here and never reach
/// `decisions`: that decision already has its own working (shape-sniffing)
/// inference in `examples/replay_burn_v2.rs::apply_choose_optional_cost`,
/// and surfacing its `CHOOSE_USE` lines too would insert new records into
/// the per-player queue that function doesn't expect, at a real risk of
/// desyncing every trace that uses Highway Robbery -- out of scope for the
/// Madness fix this increment.
const MADNESS_CHOOSE_USE_MARKER: &str = "instead of putting it into your graveyard";

/// Parses one trace's full text (already read off disk, or an in-memory
/// fixture in tests). `source_path` is only used to build error messages.
fn parse_text(text: &str, source_path: String) -> Result<GoldenTrace, String> {
    let mut trace = GoldenTrace {
        source_path: source_path.clone(),
        ..Default::default()
    };
    let mut pending_choose_use: Option<(u32, u32, String, String)> = None;
    for line in text.lines() {
        let line = line.trim_start();
        if let Some(header) = parse_choose_use_header(line) {
            pending_choose_use = Some(header);
        } else if let Some((msg, is_yes)) = parse_choose_use_line(line) {
            if let Some((decision_number, turn, phase, player)) = pending_choose_use.take() {
                if msg.contains(MADNESS_CHOOSE_USE_MARKER) {
                    trace.decisions.push(DecisionRecord {
                        decision_number,
                        player,
                        action_type: "CHOOSE_USE".to_string(),
                        candidate_count: 2,
                        candidate_texts: vec!["Yes".to_string(), "No".to_string()],
                        chosen_indices: vec![if is_yes { 0 } else { 1 }],
                        chosen_texts: vec![(if is_yes { "Yes" } else { "No" }).to_string()],
                        turn,
                        phase,
                        source_name: msg,
                        ..Default::default()
                    });
                }
            }
        } else if let Some(rest) = line.strip_prefix("REPLAY_DECISION_JSON: ") {
            match serde_json::from_str::<DecisionRecord>(rest) {
                // `episode < 0` marks a phantom record from one of
                // ComputerPlayerRL's internal lookahead clones -- it
                // was never applied to the real game (see
                // `DecisionRecord::episode`'s doc). Drop it here so
                // `decisions` really does "fully specify the game", as
                // the module doc claims.
                Ok(rec) if rec.episode < 0 => trace.phantom_decisions_skipped += 1,
                Ok(rec) => trace.decisions.push(rec),
                Err(e) => return Err(format!("{source_path}: bad decision json: {e}")),
            }
        } else if let Some(rest) = line.strip_prefix("REPLAY: ") {
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("scenario=") {
                    trace.header.scenario = v.parse().unwrap_or(0);
                } else if let Some(v) = tok.strip_prefix("seed=") {
                    trace.header.seed = v.parse().unwrap_or(0);
                }
            }
            if let Some(idx) = rest.find("agent_deck=") {
                let tail = &rest[idx + "agent_deck=".len()..];
                trace.header.agent_deck =
                    tail.split(" opp_deck=").next().unwrap_or("").to_string();
                if let Some(o) = tail.find("opp_deck=") {
                    let otail = &tail[o + "opp_deck=".len()..];
                    trace.header.opp_deck = otail
                        .split(" action_counterfactual")
                        .next()
                        .unwrap_or(otail)
                        .to_string();
                }
            }
        } else if let Some(rest) = line.strip_prefix("REPLAY_RANDOM: ") {
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("random_util_seed=") {
                    trace.header.random_util_seed = v.parse().ok();
                }
            }
        } else if let Some(rest) = line.strip_prefix("RESULT: ") {
            trace.winner = Some(rest.trim().to_string());
        } else if line.starts_with("Game finished. Winner: ") {
            trace.winner = line
                .strip_prefix("Game finished. Winner: ")
                .map(|s| s.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("Winner: ") {
            // `GameLogger.logOutcome`'s real self-play format: a bare
            // "Winner: <name>" line inside a GAME OUTCOME block, with
            // neither the "RESULT: " nor "Game finished. " prefix
            // above. Anchored to the (already trimmed) start of the
            // line, not a substring search, so free-text lines that
            // merely mention "winner" elsewhere don't match.
            trace.winner = Some(rest.trim().to_string());
        }
    }
    if trace.decisions.is_empty() {
        return Err(format!("{source_path}: no decision records"));
    }
    Ok(trace)
}

/// Load every game trace under a sweep's game_logs directory.
pub fn load_corpus(root: &Path) -> (Vec<GoldenTrace>, Vec<String>) {
    let mut traces = Vec::new();
    let mut errors = Vec::new();
    fn walk(dir: &Path, traces: &mut Vec<GoldenTrace>, errors: &mut Vec<String>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, traces, errors);
            } else if p.extension().is_some_and(|e| e == "txt")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("game_"))
            {
                match GoldenTrace::parse_file(&p) {
                    Ok(t) => traces.push(t),
                    Err(e) => errors.push(e),
                }
            }
        }
    }
    walk(root, &mut traces, &mut errors);
    (traces, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same shape as `decision_line`, but also carries a `library` snapshot
    /// -- needed for the silent-forced-keep mulligan test, which reads
    /// `opening_hand_for`'s reconstructed `library`, not just `hand`.
    fn decision_line_with_library(player: &str, action_type: &str, hand: &[&str], library: &[&str]) -> String {
        let hand_json = hand.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(",");
        let library_json = library.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(",");
        format!(
            "REPLAY_DECISION_JSON: {{\"ordinal\":0,\"player\":\"{player}\",\"action_type\":\"{action_type}\",\
             \"candidate_count\":1,\"candidate_texts\":[\"Pass\"],\"chosen_indices\":[0],\
             \"hand\":[{hand_json}],\"library\":[{library_json}],\"episode\":0}}"
        )
    }

    /// Minimal `REPLAY_DECISION_JSON` line: only the fields that matter for
    /// the assertion at hand are varied, everything else defaults exactly
    /// like a real log's less-populated records (mulligan JSON, for
    /// example, omits `hand`/`library` entirely on some builds).
    fn decision_line(player: &str, action_type: &str, episode: i64, hand: &[&str]) -> String {
        let hand_json = hand
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "REPLAY_DECISION_JSON: {{\"ordinal\":0,\"player\":\"{player}\",\"action_type\":\"{action_type}\",\
             \"candidate_count\":1,\"candidate_texts\":[\"Pass\"],\"chosen_indices\":[0],\
             \"hand\":[{hand_json}],\"episode\":{episode}}}"
        )
    }

    #[test]
    fn phantom_decisions_with_negative_episode_are_dropped() {
        let text = [
            decision_line("PlayerRL1", "MULLIGAN", 2, &[]),
            decision_line("PlayerRL1", "ACTIVATE_ABILITY_OR_SPELL", 2, &["Mountain"]),
            // A clone-probe SELECT_TARGETS fired ahead of its real cast,
            // sharing the same GameLogger -- see DecisionRecord::episode's
            // doc. This must never reach `decisions`.
            decision_line("PlayerRL1", "SELECT_TARGETS", -1, &["Lava Dart"]),
            decision_line("PlayerRL1", "SELECT_TARGETS", 2, &["Lava Dart"]),
        ]
        .join("\n");

        let trace = parse_text(&text, "fixture".to_string()).unwrap();

        assert_eq!(trace.decisions.len(), 3, "the episode=-1 phantom record must be dropped");
        assert_eq!(trace.phantom_decisions_skipped, 1);
        assert!(trace.decisions.iter().all(|d| d.episode >= 0));
    }

    #[test]
    fn bare_winner_line_is_recognized_as_a_whole_line_prefix() {
        let text = [
            decision_line("PlayerRL1", "ACTIVATE_ABILITY_OR_SPELL", 2, &["Mountain"]),
            "MULLIGAN_DECISION: player=PlayerRL1 decision=KEEP note=this Winner talk is not the outcome line".to_string(),
            "Winner: SelfPlay".to_string(),
        ]
        .join("\n");

        let trace = parse_text(&text, "fixture".to_string()).unwrap();
        assert_eq!(trace.winner.as_deref(), Some("SelfPlay"));
    }

    #[test]
    fn result_and_game_finished_winner_formats_still_parse() {
        let with_result = [decision_line("P", "ACTIVATE_ABILITY_OR_SPELL", 0, &[]), "RESULT: PlayerRL1".to_string()].join("\n");
        assert_eq!(parse_text(&with_result, "fixture".to_string()).unwrap().winner.as_deref(), Some("PlayerRL1"));

        let with_game_finished =
            [decision_line("P", "ACTIVATE_ABILITY_OR_SPELL", 0, &[]), "Game finished. Winner: PlayerRL1".to_string()].join("\n");
        assert_eq!(parse_text(&with_game_finished, "fixture".to_string()).unwrap().winner.as_deref(), Some("PlayerRL1"));
    }

    #[test]
    fn opening_hand_for_returns_the_terminal_mulligan_keep_hand_not_a_later_inflated_one() {
        let text = [
            // Mulligans once: MULLIGAN(mulligan) -> LONDON_MULLIGAN(fresh 7,
            // bottom 1) -> MULLIGAN(keep, terminal -- last MULLIGAN record).
            decision_line("PlayerRL1", "MULLIGAN", 0, &["Mountain", "Mountain"]),
            decision_line(
                "PlayerRL1",
                "LONDON_MULLIGAN",
                0,
                &["Mountain", "Lava Dart", "Lightning Bolt", "Guttersnipe", "Sneaky Snacker", "Masked Meower", "Fireblast"],
            ),
            decision_line(
                "PlayerRL1",
                "MULLIGAN",
                0,
                &["Mountain", "Lava Dart", "Lightning Bolt", "Guttersnipe", "Sneaky Snacker", "Masked Meower"],
            ),
            // This player's first *logged* real decision doesn't arrive
            // until a couple of turns later (nothing playable earlier, so
            // the reference engine never logged those windows) -- by which
            // point 1 extra card has been drawn naturally. Using this
            // record's hand instead of the terminal-mulligan one would
            // silently seed an inflated opening hand.
            decision_line(
                "PlayerRL1",
                "ACTIVATE_ABILITY_OR_SPELL",
                0,
                &["Mountain", "Lava Dart", "Lightning Bolt", "Guttersnipe", "Sneaky Snacker", "Masked Meower", "Fiery Temper"],
            ),
            // SelfPlay keeps on the first 7, no mulligan loop at all.
            decision_line("SelfPlay", "MULLIGAN", 0, &["Mountain", "Mountain", "Mountain", "Mountain", "Mountain", "Grab the Prize", "Lava Dart"]),
            decision_line(
                "SelfPlay",
                "ACTIVATE_ABILITY_OR_SPELL",
                0,
                &["Mountain", "Mountain", "Mountain", "Mountain", "Mountain", "Grab the Prize", "Lava Dart", "Lightning Bolt"],
            ),
        ]
        .join("\n");

        let trace = parse_text(&text, "fixture".to_string()).unwrap();

        let opening = trace.opening_hand_for("PlayerRL1").expect("PlayerRL1 has a MULLIGAN decision");
        assert_eq!(opening.hand, vec!["Mountain", "Lava Dart", "Lightning Bolt", "Guttersnipe", "Sneaky Snacker", "Masked Meower"]);

        let other = trace.opening_hand_for("SelfPlay").expect("SelfPlay has a MULLIGAN decision");
        assert_eq!(other.hand, vec!["Mountain", "Mountain", "Mountain", "Mountain", "Mountain", "Grab the Prize", "Lava Dart"]);

        assert!(trace.opening_hand_for("Nobody").is_none());
    }

    #[test]
    fn opening_hand_for_reconstructs_an_empty_hand_when_the_chain_ends_on_a_silent_forced_keep() {
        // Real-corpus shape (game_20260712_194635_0036.txt, PlayerRL1,
        // mulligans_taken=7 against a 7-card starting hand): the terminal
        // single-card bottom pick is silently auto-resolved (genericChoose's
        // bare single-candidate shortcut skips logging), so there's no
        // trailing MULLIGAN(KEEP) record at all -- the last logged record
        // for this player is this LONDON_MULLIGAN one, candidate_count=2,
        // hand=["Mountain","Fiery Temper"], library_size=58 (58+2==60, the
        // full deck already, confirming the silent final pick just moves
        // both remaining hand cards to the bottom).
        let text = decision_line_with_library(
            "PlayerRL1",
            "LONDON_MULLIGAN",
            &["Mountain", "Fiery Temper"],
            &["Grab the Prize", "Highway Robbery", "Mountain"], // stand-in for the real 58-card library
        );

        let trace = parse_text(&text, "fixture".to_string()).unwrap();
        let opening = trace.opening_hand_for("PlayerRL1").expect("PlayerRL1 has a mulligan-phase record");

        assert_eq!(opening.hand, Vec::<String>::new(), "mulliganing all the way to 0 cards forces a silent keep with an empty hand");
        // Both leftover `hand` cards from the last logged pick are headed to
        // the bottom of the library either way, appended after the record's
        // own `library` snapshot.
        assert_eq!(opening.library, vec!["Grab the Prize", "Highway Robbery", "Mountain", "Mountain", "Fiery Temper"]);
    }
}
