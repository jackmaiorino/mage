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

#[derive(Debug, Clone, Deserialize)]
pub struct DecisionRecord {
    pub ordinal: u32,
    #[serde(default)]
    pub decision_number: u32,
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
    #[serde(default)]
    pub random_util_count: u64,
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
}

impl GoldenTrace {
    pub fn parse_file(path: &Path) -> Result<GoldenTrace, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut trace = GoldenTrace {
            source_path: path.display().to_string(),
            ..Default::default()
        };
        for line in text.lines() {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix("REPLAY_DECISION_JSON: ") {
                match serde_json::from_str::<DecisionRecord>(rest) {
                    Ok(rec) => trace.decisions.push(rec),
                    Err(e) => return Err(format!("{}: bad decision json: {e}", path.display())),
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
            }
        }
        if trace.decisions.is_empty() {
            return Err(format!("{}: no decision records", path.display()));
        }
        Ok(trace)
    }

    /// The agent's post-shuffle opening library order: first record that carries
    /// a full library snapshot (the mulligan decision, before any draw).
    pub fn opening_library(&self) -> Option<(&[String], &[String])> {
        self.decisions
            .iter()
            .find(|d| !d.library.is_empty())
            .map(|d| (d.library.as_slice(), d.library_object_ids.as_slice()))
    }
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
