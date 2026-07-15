//! Interactive RL session protocol for the kernel Burn-mirror environment.
//!
//! This module owns the reset/step state machine used by both the JSONL
//! process wrapper and the batch rollout recorder, so action validation and
//! terminal classification cannot drift between interactive and offline use.

use crate::card_def::KERNEL_CARDDB_HASH;
use crate::engine::Decision;
use crate::ids::PlayerId;
use crate::rl::{
    acting_player_for_surface_decision, build_burn_mirror_state, legal_action_candidates_v1,
    observe_v1, validate_selected_action, EpisodeTerminalSummaryV1, LegalActionCandidateV1,
    LegalActionV1, ObservationV1, PlayerSeatV1, RlContractError, TerminalOutcomeV1,
};
use crate::surface_v2::{HarnessSurfaceV2, SurfaceDecision, H2_PREDICATE_VERSION};
use crate::KERNEL_VERSION;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

pub const RL_SESSION_SCHEMA_VERSION: u32 = 1;
pub const RL_SESSION_PROTOCOL_VERSION: u32 = 1;
pub const RL_SESSION_PROTOCOL_NAME: &str = "kernel_rl_jsonl";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlSessionProvenanceV1 {
    pub protocol: String,
    pub protocol_version: u32,
    pub schema_version: u32,
    pub kernel_version: String,
    pub surface_version: u32,
    pub card_db_hash: u64,
}

impl RlSessionProvenanceV1 {
    pub fn current() -> Self {
        RlSessionProvenanceV1 {
            protocol: RL_SESSION_PROTOCOL_NAME.to_string(),
            protocol_version: RL_SESSION_PROTOCOL_VERSION,
            schema_version: RL_SESSION_SCHEMA_VERSION,
            kernel_version: KERNEL_VERSION.to_string(),
            surface_version: H2_PREDICATE_VERSION,
            card_db_hash: KERNEL_CARDDB_HASH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlSessionDecisionV1 {
    pub schema_version: u32,
    pub episode_id: u64,
    pub step: u64,
    pub acting_player: PlayerSeatV1,
    pub observation: Box<ObservationV1>,
    pub legal_actions: Vec<LegalActionV1>,
    pub reward: [i32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlSessionTerminalV1 {
    pub schema_version: u32,
    pub episode_id: u64,
    pub terminal_outcome: TerminalOutcomeV1,
    pub winner: Option<PlayerSeatV1>,
    pub terminal_reward: [i32; 2],
    pub terminal_reason: String,
    pub decision_count: u64,
}

impl From<RlSessionTerminalV1> for EpisodeTerminalSummaryV1 {
    fn from(value: RlSessionTerminalV1) -> Self {
        EpisodeTerminalSummaryV1 {
            episode_id: value.episode_id,
            outcome: value.terminal_outcome,
            winner: value.winner,
            terminal_reward: value.terminal_reward,
            terminal_reason: value.terminal_reason,
            decision_count: value.decision_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlSessionResponseV1 {
    Decision(RlSessionDecisionV1),
    Terminal(RlSessionTerminalV1),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlSessionErrorCode {
    EpisodeAlreadyTerminal,
    EpisodeIdMismatch,
    ExpectedStepMismatch,
    SelectedIndexOutOfRange,
    SelectedActionIdMismatch,
    SelectedActionIdUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlSessionError {
    pub code: RlSessionErrorCode,
    pub message: String,
}

impl fmt::Display for RlSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for RlSessionError {}

impl From<RlSessionError> for RlContractError {
    fn from(value: RlSessionError) -> Self {
        RlContractError(value.to_string())
    }
}

#[derive(Debug, Clone)]
struct CurrentDecisionV1 {
    actor: PlayerId,
    observation: ObservationV1,
    candidates: Vec<LegalActionCandidateV1>,
}

pub struct RlEpisodeSessionV1 {
    episode_id: u64,
    max_decisions: u64,
    state: crate::state::GameState,
    surface: HarnessSurfaceV2,
    decision_count: u64,
    current: Option<CurrentDecisionV1>,
    terminal: Option<RlSessionTerminalV1>,
}

impl RlEpisodeSessionV1 {
    pub fn reset(episode_id: u64, env_seed: u64, max_decisions: u64) -> Self {
        let mut session = RlEpisodeSessionV1 {
            episode_id,
            max_decisions,
            state: build_burn_mirror_state(env_seed),
            surface: HarnessSurfaceV2::new(),
            decision_count: 0,
            current: None,
            terminal: None,
        };
        session.advance_to_decision_or_terminal();
        session
    }

    pub fn current_response(&self) -> RlSessionResponseV1 {
        if let Some(terminal) = &self.terminal {
            return RlSessionResponseV1::Terminal(terminal.clone());
        }
        let current = self
            .current
            .as_ref()
            .expect("session has either a current decision or terminal");
        RlSessionResponseV1::Decision(RlSessionDecisionV1 {
            schema_version: RL_SESSION_SCHEMA_VERSION,
            episode_id: self.episode_id,
            step: self.decision_count,
            acting_player: current.actor.into(),
            observation: Box::new(current.observation.clone()),
            legal_actions: current
                .candidates
                .iter()
                .map(|c| c.record.clone())
                .collect(),
            reward: [0, 0],
        })
    }

    pub fn decision_count(&self) -> u64 {
        self.decision_count
    }

    pub fn diagnostic_state_hash(&self) -> u64 {
        self.state.state_hash()
    }

    pub fn step(
        &mut self,
        episode_id: u64,
        expected_step: u64,
        selected_index: u32,
        selected_action_id: &str,
    ) -> Result<RlSessionResponseV1, RlSessionError> {
        if episode_id != self.episode_id {
            return Err(session_error(
                RlSessionErrorCode::EpisodeIdMismatch,
                "step request episode_id does not match the active episode",
            ));
        }
        if expected_step != self.decision_count {
            return Err(session_error(
                RlSessionErrorCode::ExpectedStepMismatch,
                "step request expected_step does not match the active decision step",
            ));
        }
        if self.terminal.is_some() {
            return Err(session_error(
                RlSessionErrorCode::EpisodeAlreadyTerminal,
                "episode is already terminal; reset before stepping again",
            ));
        }
        let current = self
            .current
            .as_ref()
            .expect("nonterminal session has current decision");
        let selected_index_usize = selected_index as usize;
        let Some(selected) = current.candidates.get(selected_index_usize) else {
            return Err(session_error(
                RlSessionErrorCode::SelectedIndexOutOfRange,
                "selected_index is outside the current legal action list",
            ));
        };
        if selected.record.stable_id != selected_action_id {
            let code = if current
                .candidates
                .iter()
                .any(|candidate| candidate.record.stable_id == selected_action_id)
            {
                RlSessionErrorCode::SelectedActionIdMismatch
            } else {
                RlSessionErrorCode::SelectedActionIdUnknown
            };
            return Err(session_error(
                code,
                "selected_index and selected_action_id do not identify the same current action",
            ));
        }
        validate_selected_action(
            &current.candidates,
            selected_index_usize,
            selected_action_id,
        )
        .map_err(|_| {
            session_error(
                RlSessionErrorCode::SelectedActionIdMismatch,
                "selected action failed current-action validation",
            )
        })?;
        let surface_action = selected.surface_action.clone();
        self.current = None;
        if let Err(err) = self.surface.apply(&mut self.state, surface_action) {
            self.decision_count += 1;
            self.terminal = Some(halted_terminal(
                self.episode_id,
                format!("fail_closed:surface_apply:{err}"),
                self.decision_count,
            ));
            return Ok(self.current_response());
        }
        self.decision_count += 1;
        self.advance_to_decision_or_terminal();
        Ok(self.current_response())
    }

    fn advance_to_decision_or_terminal(&mut self) {
        self.current = None;
        let surfaced = self.surface.next_decision(&mut self.state);
        match &surfaced {
            SurfaceDecision::Decision(Decision::GameOver { winner }) => {
                self.terminal = Some(terminal_from_winner(
                    self.episode_id,
                    *winner,
                    "game_over".to_string(),
                    self.decision_count,
                ));
                return;
            }
            SurfaceDecision::Decision(Decision::Halted { mechanic, source }) => {
                self.terminal = Some(halted_terminal(
                    self.episode_id,
                    format!("engine_halted:{mechanic:?}:source:{}", source.0),
                    self.decision_count,
                ));
                return;
            }
            _ => {}
        }
        if self.decision_count >= self.max_decisions {
            self.terminal = Some(halted_terminal(
                self.episode_id,
                format!("decision_cap_reached:{}", self.max_decisions),
                self.decision_count,
            ));
            return;
        }
        let Some(actor) = acting_player_for_surface_decision(&surfaced, &self.state) else {
            self.terminal = Some(halted_terminal(
                self.episode_id,
                "fail_closed:nonterminal decision without acting player".to_string(),
                self.decision_count,
            ));
            return;
        };
        let observation = match observe_v1(&self.state, actor, self.decision_count) {
            Ok(observation) => observation,
            Err(err) => {
                self.terminal = Some(halted_terminal(
                    self.episode_id,
                    format!("fail_closed:observation:{err}"),
                    self.decision_count,
                ));
                return;
            }
        };
        let candidates = match legal_action_candidates_v1(&surfaced, &self.state) {
            Ok(candidates) => candidates,
            Err(err) => {
                self.terminal = Some(halted_terminal(
                    self.episode_id,
                    format!("fail_closed:{err}"),
                    self.decision_count,
                ));
                return;
            }
        };
        if candidates.is_empty() {
            self.terminal = Some(halted_terminal(
                self.episode_id,
                "fail_closed:nonterminal decision produced zero legal actions".to_string(),
                self.decision_count,
            ));
            return;
        }
        self.current = Some(CurrentDecisionV1 {
            actor,
            observation,
            candidates,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "request_type", rename_all = "snake_case")]
pub enum KernelRlRequestV1 {
    Reset {
        schema_version: u32,
        request_id: String,
        episode_id: u64,
        env_seed: u64,
        max_decisions: u64,
    },
    Step {
        schema_version: u32,
        request_id: String,
        episode_id: u64,
        expected_step: u64,
        selected_index: u32,
        selected_action_id: String,
    },
}

impl KernelRlRequestV1 {
    fn request_id(&self) -> &str {
        match self {
            KernelRlRequestV1::Reset { request_id, .. }
            | KernelRlRequestV1::Step { request_id, .. } => request_id,
        }
    }

    fn schema_version(&self) -> u32 {
        match self {
            KernelRlRequestV1::Reset { schema_version, .. }
            | KernelRlRequestV1::Step { schema_version, .. } => *schema_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelRlErrorV1 {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "response_type", rename_all = "snake_case")]
pub enum KernelRlResponseV1 {
    Decision {
        schema_version: u32,
        request_id: String,
        provenance: RlSessionProvenanceV1,
        episode_id: u64,
        step: u64,
        acting_player: PlayerSeatV1,
        observation: Box<ObservationV1>,
        legal_actions: Vec<LegalActionV1>,
        reward: [i32; 2],
    },
    Terminal {
        schema_version: u32,
        request_id: String,
        provenance: RlSessionProvenanceV1,
        episode_id: u64,
        terminal_outcome: TerminalOutcomeV1,
        winner: Option<PlayerSeatV1>,
        terminal_reward: [i32; 2],
        terminal_reason: String,
        decision_count: u64,
    },
    Error {
        schema_version: u32,
        request_id: Option<String>,
        error: KernelRlErrorV1,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedProtocolExchangeV1 {
    // Request ids are process-unique except for one immediate identical retry.
    // The cache is deliberately one entry so stale-step safety comes from the
    // episode_id/expected_step preconditions, not an unbounded replay table.
    request_id: String,
    request: KernelRlRequestV1,
    response_line: String,
}

#[derive(Default)]
pub struct KernelRlJsonlServerV1 {
    session: Option<RlEpisodeSessionV1>,
    last_exchange: Option<CachedProtocolExchangeV1>,
}

impl KernelRlJsonlServerV1 {
    pub fn new() -> Self {
        KernelRlJsonlServerV1::default()
    }

    pub fn handle_line(&mut self, line: &str) -> String {
        let value = match serde_json::from_str::<Value>(line) {
            Ok(value) => value,
            Err(_) => {
                return serialize_response(error_response(
                    None,
                    "malformed_json",
                    "request line is not valid JSON",
                ));
            }
        };
        let request_id = request_id_from_value(&value);
        let request = match serde_json::from_value::<KernelRlRequestV1>(value) {
            Ok(request) => request,
            Err(_) => {
                return serialize_response(error_response(
                    request_id,
                    "malformed_request",
                    "request does not match the v1 protocol schema",
                ));
            }
        };
        if let Some(cached) = &self.last_exchange {
            if cached.request_id == request.request_id() {
                if cached.request == request {
                    return cached.response_line.clone();
                }
                return serialize_response(error_response(
                    Some(request.request_id().to_string()),
                    "request_id_reuse_mismatch",
                    "request_id was reused for a different immediate request payload",
                ));
            }
        }
        let should_cache = request.schema_version() == RL_SESSION_SCHEMA_VERSION;
        let request_for_cache = request.clone();
        let response = self.handle_request(request);
        let response_line = serialize_response(response);
        if should_cache {
            self.last_exchange = Some(CachedProtocolExchangeV1 {
                request_id: request_for_cache.request_id().to_string(),
                request: request_for_cache,
                response_line: response_line.clone(),
            });
        }
        response_line
    }

    fn handle_request(&mut self, request: KernelRlRequestV1) -> KernelRlResponseV1 {
        match request {
            KernelRlRequestV1::Reset {
                schema_version,
                request_id,
                episode_id,
                env_seed,
                max_decisions,
            } => {
                if schema_version != RL_SESSION_SCHEMA_VERSION {
                    return error_response(
                        Some(request_id),
                        "schema_version_mismatch",
                        "unsupported request schema_version",
                    );
                }
                let session = RlEpisodeSessionV1::reset(episode_id, env_seed, max_decisions);
                let response = session_response_to_protocol(request_id, session.current_response());
                self.session = Some(session);
                response
            }
            KernelRlRequestV1::Step {
                schema_version,
                request_id,
                episode_id,
                expected_step,
                selected_index,
                selected_action_id,
            } => {
                if schema_version != RL_SESSION_SCHEMA_VERSION {
                    return error_response(
                        Some(request_id),
                        "schema_version_mismatch",
                        "unsupported request schema_version",
                    );
                }
                let Some(session) = self.session.as_mut() else {
                    return error_response(
                        Some(request_id),
                        "step_before_reset",
                        "step request received before reset",
                    );
                };
                match session.step(
                    episode_id,
                    expected_step,
                    selected_index,
                    &selected_action_id,
                ) {
                    Ok(response) => session_response_to_protocol(request_id, response),
                    Err(err) => error_response(
                        Some(request_id),
                        session_error_code(&err.code),
                        &err.message,
                    ),
                }
            }
        }
    }
}

fn session_response_to_protocol(
    request_id: String,
    response: RlSessionResponseV1,
) -> KernelRlResponseV1 {
    match response {
        RlSessionResponseV1::Decision(decision) => KernelRlResponseV1::Decision {
            schema_version: RL_SESSION_SCHEMA_VERSION,
            request_id,
            provenance: RlSessionProvenanceV1::current(),
            episode_id: decision.episode_id,
            step: decision.step,
            acting_player: decision.acting_player,
            observation: decision.observation,
            legal_actions: decision.legal_actions,
            reward: decision.reward,
        },
        RlSessionResponseV1::Terminal(terminal) => KernelRlResponseV1::Terminal {
            schema_version: RL_SESSION_SCHEMA_VERSION,
            request_id,
            provenance: RlSessionProvenanceV1::current(),
            episode_id: terminal.episode_id,
            terminal_outcome: terminal.terminal_outcome,
            winner: terminal.winner,
            terminal_reward: terminal.terminal_reward,
            terminal_reason: terminal.terminal_reason,
            decision_count: terminal.decision_count,
        },
    }
}

fn serialize_response(response: KernelRlResponseV1) -> String {
    serde_json::to_string(&response).expect("protocol response serializes")
}

fn request_id_from_value(value: &Value) -> Option<String> {
    value
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn error_response(request_id: Option<String>, code: &str, message: &str) -> KernelRlResponseV1 {
    KernelRlResponseV1::Error {
        schema_version: RL_SESSION_SCHEMA_VERSION,
        request_id,
        error: KernelRlErrorV1 {
            code: code.to_string(),
            message: message.to_string(),
        },
    }
}

fn session_error(code: RlSessionErrorCode, message: &str) -> RlSessionError {
    RlSessionError {
        code,
        message: message.to_string(),
    }
}

fn session_error_code(code: &RlSessionErrorCode) -> &'static str {
    match code {
        RlSessionErrorCode::EpisodeAlreadyTerminal => "episode_already_terminal",
        RlSessionErrorCode::EpisodeIdMismatch => "episode_id_mismatch",
        RlSessionErrorCode::ExpectedStepMismatch => "expected_step_mismatch",
        RlSessionErrorCode::SelectedIndexOutOfRange => "selected_index_out_of_range",
        RlSessionErrorCode::SelectedActionIdMismatch => "selected_action_id_mismatch",
        RlSessionErrorCode::SelectedActionIdUnknown => "selected_action_id_unknown",
    }
}

fn terminal_from_winner(
    episode_id: u64,
    winner: Option<PlayerId>,
    terminal_reason: String,
    decision_count: u64,
) -> RlSessionTerminalV1 {
    let (terminal_outcome, terminal_reward) = match winner {
        Some(PlayerId::P0) => (TerminalOutcomeV1::P0Win, [1, -1]),
        Some(PlayerId::P1) => (TerminalOutcomeV1::P1Win, [-1, 1]),
        None => (TerminalOutcomeV1::Draw, [0, 0]),
        Some(other) => panic!("unsupported winner player id {}", other.0),
    };
    RlSessionTerminalV1 {
        schema_version: RL_SESSION_SCHEMA_VERSION,
        episode_id,
        terminal_outcome,
        winner: winner.map(Into::into),
        terminal_reward,
        terminal_reason,
        decision_count,
    }
}

fn halted_terminal(
    episode_id: u64,
    terminal_reason: String,
    decision_count: u64,
) -> RlSessionTerminalV1 {
    RlSessionTerminalV1 {
        schema_version: RL_SESSION_SCHEMA_VERSION,
        episode_id,
        terminal_outcome: TerminalOutcomeV1::Halted,
        winner: None,
        terminal_reward: [0, 0],
        terminal_reason,
        decision_count,
    }
}
