use mtg_kernel::rl::{derive_env_seed, derive_policy_seed, record_burn_mirror_episode};
use mtg_kernel::rl_session::{
    KernelRlJsonlServerV1, KernelRlResponseV1, RlEpisodeSessionV1, RlSessionErrorCode,
    RlSessionResponseV1, RL_SESSION_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn reset_line(request_id: &str, max_decisions: u64) -> String {
    reset_line_for_episode(request_id, 0, max_decisions)
}

fn reset_line_for_episode(request_id: &str, episode_id: u64, max_decisions: u64) -> String {
    json!({
        "request_type": "reset",
        "schema_version": RL_SESSION_SCHEMA_VERSION,
        "request_id": request_id,
        "episode_id": episode_id,
        "env_seed": derive_env_seed(5151, episode_id),
        "max_decisions": max_decisions,
    })
    .to_string()
}

fn step_line(
    request_id: &str,
    episode_id: u64,
    expected_step: u64,
    selected_index: u32,
    selected_action_id: &str,
) -> String {
    json!({
        "request_type": "step",
        "schema_version": RL_SESSION_SCHEMA_VERSION,
        "request_id": request_id,
        "episode_id": episode_id,
        "expected_step": expected_step,
        "selected_index": selected_index,
        "selected_action_id": selected_action_id,
    })
    .to_string()
}

fn step_line_from_decision(request_id: &str, decision: &Value, selected_index: usize) -> String {
    let action = &decision["legal_actions"].as_array().unwrap()[selected_index];
    step_line(
        request_id,
        decision["episode_id"].as_u64().unwrap(),
        decision["step"].as_u64().unwrap(),
        action["selected_index"].as_u64().unwrap() as u32,
        action["stable_id"].as_str().unwrap(),
    )
}

fn parse_response(line: &str) -> Value {
    serde_json::from_str(line).expect("response is valid JSON")
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(map) => {
            map.contains_key(key) || map.values().any(|child| contains_key(child, key))
        }
        Value::Array(items) => items.iter().any(|child| contains_key(child, key)),
        _ => false,
    }
}

fn action_ids(response: &RlSessionResponseV1) -> Vec<String> {
    match response {
        RlSessionResponseV1::Decision(decision) => decision
            .legal_actions
            .iter()
            .map(|action| action.stable_id.clone())
            .collect(),
        RlSessionResponseV1::Terminal(_) => vec![],
    }
}

fn decision_step(response: &RlSessionResponseV1) -> u64 {
    match response {
        RlSessionResponseV1::Decision(decision) => decision.step,
        RlSessionResponseV1::Terminal(terminal) => terminal.decision_count,
    }
}

fn current_snapshot(session: &RlEpisodeSessionV1) -> (String, u64, u64) {
    (
        serde_json::to_string(&session.current_response()).unwrap(),
        session.diagnostic_state_hash(),
        session.decision_count(),
    )
}

#[test]
fn rl_session_decision_response_has_no_full_state_diagnostic_hash() {
    let mut server = KernelRlJsonlServerV1::new();
    let output = server.handle_line(&reset_line("reset-no-hash", 32));
    let value = parse_response(&output);

    assert_eq!(value["response_type"], "decision");
    assert!(!contains_key(&value, "diagnostic_state_hash"));
    assert!(!contains_key(
        &value,
        "diagnostic_state_hash_includes_hidden_state"
    ));
    assert!(!contains_key(&value, "state_hash"));
    assert!(!value["legal_actions"].as_array().unwrap().is_empty());
}

#[test]
fn rl_session_terminal_response_has_provenance_and_no_diagnostic_hash() {
    let mut server = KernelRlJsonlServerV1::new();
    let output = server.handle_line(&reset_line("terminal-provenance", 0));
    let value = parse_response(&output);

    assert_eq!(value["response_type"], "terminal");
    assert_eq!(value["request_id"], "terminal-provenance");
    assert_eq!(value["provenance"]["protocol"], "kernel_rl_jsonl");
    assert_eq!(value["provenance"]["protocol_version"], 1);
    assert_eq!(
        value["provenance"]["schema_version"],
        RL_SESSION_SCHEMA_VERSION
    );
    assert!(value["provenance"]["kernel_version"].is_string());
    assert!(value["provenance"]["surface_version"].is_number());
    assert!(value["provenance"]["card_db_hash"].is_number());
    assert!(!contains_key(&value, "diagnostic_state_hash"));
    assert!(!contains_key(
        &value,
        "diagnostic_state_hash_includes_hidden_state"
    ));
    assert!(!contains_key(&value, "state_hash"));
}

#[test]
fn rl_session_reset_response_is_deterministic_for_identical_inputs() {
    let mut server = KernelRlJsonlServerV1::new();
    let line = reset_line("reset-deterministic", 32);
    let first = server.handle_line(&line);
    let second = server.handle_line(&line);
    assert_eq!(first, second);
}

#[test]
fn rl_session_deterministic_action_sequence_reaches_same_terminal() {
    fn run_sequence() -> Vec<String> {
        let mut server = KernelRlJsonlServerV1::new();
        let mut outputs = Vec::new();
        let mut current = server.handle_line(&reset_line("r0", 8));
        outputs.push(current.clone());
        for step in 0..16 {
            let value = parse_response(&current);
            match value["response_type"].as_str().unwrap() {
                "decision" => {
                    current = server.handle_line(&step_line_from_decision(
                        &format!("s{step}"),
                        &value,
                        0,
                    ));
                    outputs.push(current.clone());
                }
                "terminal" => break,
                other => panic!("unexpected response type {other}"),
            }
        }
        outputs
    }

    let a = run_sequence();
    let b = run_sequence();
    assert_eq!(a, b);
    let terminal = parse_response(a.last().unwrap());
    assert_eq!(terminal["response_type"], "terminal");
    assert_eq!(terminal["terminal_outcome"], "halted");
    assert_eq!(terminal["terminal_reward"], json!([0, 0]));
    assert_eq!(terminal["decision_count"], 8);
}

#[test]
fn rl_session_immediate_step_retry_is_byte_identical_and_does_not_advance() {
    let mut server = KernelRlJsonlServerV1::new();
    let reset = server.handle_line(&reset_line("retry-reset", 32));
    let reset_value = parse_response(&reset);
    let step_request = step_line_from_decision("retry-step", &reset_value, 0);

    let first = server.handle_line(&step_request);
    let retry = server.handle_line(&step_request);
    assert_eq!(retry, first);

    let current_after_retry = parse_response(&retry);
    assert_eq!(current_after_retry["response_type"], "decision");
    let next = server.handle_line(&step_line_from_decision(
        "after-retry",
        &current_after_retry,
        0,
    ));
    let next_value = parse_response(&next);
    assert_ne!(next_value["response_type"], "error");
}

#[test]
fn rl_session_immediate_request_id_reuse_with_different_payload_fails_closed() {
    let mut server = KernelRlJsonlServerV1::new();
    let reset = server.handle_line(&reset_line("reuse-reset", 32));
    let reset_value = parse_response(&reset);
    assert!(
        reset_value["legal_actions"].as_array().unwrap().len() >= 2,
        "initial Burn decision should have at least two legal actions"
    );
    let step_request = step_line_from_decision("reuse-step", &reset_value, 0);
    let first = server.handle_line(&step_request);

    let different_same_id = step_line_from_decision("reuse-step", &reset_value, 1);
    let rejected = server.handle_line(&different_same_id);
    let rejected_value = parse_response(&rejected);
    assert_eq!(rejected_value["response_type"], "error");
    assert_eq!(rejected_value["error"]["code"], "request_id_reuse_mismatch");
    assert_eq!(rejected_value["request_id"], "reuse-step");

    let current = parse_response(&first);
    assert_eq!(current["response_type"], "decision");
    let next = server.handle_line(&step_line_from_decision(
        "after-reuse-mismatch",
        &current,
        0,
    ));
    let next_value = parse_response(&next);
    assert_ne!(next_value["response_type"], "error");
}

#[test]
fn rl_session_protocol_precondition_errors_are_typed_and_non_mutating() {
    let mut server = KernelRlJsonlServerV1::new();
    let reset = server.handle_line(&reset_line("precondition-reset", 32));
    let reset_value = parse_response(&reset);
    let action = &reset_value["legal_actions"].as_array().unwrap()[0];

    let wrong_episode = step_line(
        "wrong-episode",
        reset_value["episode_id"].as_u64().unwrap() + 1,
        reset_value["step"].as_u64().unwrap(),
        action["selected_index"].as_u64().unwrap() as u32,
        action["stable_id"].as_str().unwrap(),
    );
    let wrong_episode_response = parse_response(&server.handle_line(&wrong_episode));
    assert_eq!(wrong_episode_response["response_type"], "error");
    assert_eq!(
        wrong_episode_response["error"]["code"],
        "episode_id_mismatch"
    );

    let wrong_step = step_line(
        "wrong-step",
        reset_value["episode_id"].as_u64().unwrap(),
        reset_value["step"].as_u64().unwrap() + 1,
        action["selected_index"].as_u64().unwrap() as u32,
        action["stable_id"].as_str().unwrap(),
    );
    let wrong_step_response = parse_response(&server.handle_line(&wrong_step));
    assert_eq!(wrong_step_response["response_type"], "error");
    assert_eq!(
        wrong_step_response["error"]["code"],
        "expected_step_mismatch"
    );

    let valid = server.handle_line(&step_line_from_decision(
        "after-precondition-errors",
        &reset_value,
        0,
    ));
    let valid_value = parse_response(&valid);
    assert_ne!(valid_value["response_type"], "error");
}

#[test]
fn rl_session_invalid_index_and_id_errors_do_not_mutate() {
    let mut session = RlEpisodeSessionV1::reset(0, derive_env_seed(5151, 0), 32);
    let ids = action_ids(&session.current_response());
    assert!(
        ids.len() >= 2,
        "initial Burn decision should have multiple actions"
    );

    let before = current_snapshot(&session);
    let err = session.step(0, 0, 9999, &ids[0]).unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedIndexOutOfRange);
    assert_eq!(current_snapshot(&session), before);

    let before = current_snapshot(&session);
    let err = session.step(0, 0, 0, &ids[1]).unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedActionIdMismatch);
    assert_eq!(current_snapshot(&session), before);

    let before = current_snapshot(&session);
    let err = session
        .step(0, 0, 0, "legal-action-v1:ffffffffffffffff")
        .unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedActionIdUnknown);
    assert_eq!(current_snapshot(&session), before);
}

#[test]
fn rl_session_expected_step_mismatch_precedes_repeated_action_identity() {
    let mut session = RlEpisodeSessionV1::reset(0, derive_env_seed(5151, 0), 128);

    for _ in 0..64 {
        let prior_response = session.current_response();
        let prior_step = decision_step(&prior_response);
        let prior_ids = action_ids(&prior_response);
        let Some(selected) = prior_ids.first().cloned() else {
            break;
        };
        session.step(0, prior_step, 0, &selected).unwrap();

        if matches!(session.current_response(), RlSessionResponseV1::Terminal(_)) {
            break;
        }
        let current_ids = action_ids(&session.current_response());
        if let Some((current_index, repeated_id)) = current_ids
            .iter()
            .enumerate()
            .find(|(_, current_id)| prior_ids.contains(current_id))
        {
            let before = current_snapshot(&session);
            let err = session
                .step(0, prior_step, current_index as u32, repeated_id)
                .unwrap_err();
            assert_eq!(err.code, RlSessionErrorCode::ExpectedStepMismatch);
            assert_eq!(current_snapshot(&session), before);
            return;
        }
    }

    panic!("did not find a repeated legal action id across adjacent decisions");
}

#[test]
fn rl_session_prior_episode_step_is_rejected_before_legal_action_lookup() {
    let prior = RlEpisodeSessionV1::reset(0, derive_env_seed(5151, 0), 32);
    let prior_ids = action_ids(&prior.current_response());
    let mut current = RlEpisodeSessionV1::reset(1, derive_env_seed(5151, 0), 32);
    let current_ids = action_ids(&current.current_response());
    let (current_index, repeated_id) = current_ids
        .iter()
        .enumerate()
        .find(|(_, current_id)| prior_ids.contains(current_id))
        .expect("same env seed should expose a legal action id shared across episodes");

    let before = current_snapshot(&current);
    let err = current
        .step(0, 0, current_index as u32, repeated_id)
        .unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::EpisodeIdMismatch);
    assert_eq!(current_snapshot(&current), before);
}

#[test]
fn rl_session_stale_id_error_does_not_mutate_current_decision() {
    let mut session = RlEpisodeSessionV1::reset(0, derive_env_seed(5151, 0), 64);
    let mut prior_ids: BTreeSet<String> = BTreeSet::new();

    for _ in 0..16 {
        let current_ids = action_ids(&session.current_response());
        let current_set: BTreeSet<_> = current_ids.iter().cloned().collect();
        if let Some(stale_id) = prior_ids
            .iter()
            .find(|prior_id| !current_set.contains(*prior_id))
            .cloned()
        {
            let before = current_snapshot(&session);
            let err = session
                .step(0, decision_step(&session.current_response()), 0, &stale_id)
                .unwrap_err();
            assert_eq!(err.code, RlSessionErrorCode::SelectedActionIdUnknown);
            assert_eq!(current_snapshot(&session), before);
            return;
        }
        for id in &current_ids {
            prior_ids.insert(id.clone());
        }
        let selected = current_ids
            .first()
            .expect("nonterminal response has legal actions")
            .clone();
        session
            .step(0, decision_step(&session.current_response()), 0, &selected)
            .unwrap();
        if matches!(session.current_response(), RlSessionResponseV1::Terminal(_)) {
            break;
        }
    }
    panic!("did not find a stale action id within the bounded smoke sequence");
}

#[test]
fn rl_session_step_before_reset_and_malformed_input_recover_cleanly() {
    let mut server = KernelRlJsonlServerV1::new();

    let step_before_reset =
        server.handle_line(&step_line("early-step", 0, 0, 0, "legal-action-v1:0"));
    let step_value: KernelRlResponseV1 = serde_json::from_str(&step_before_reset).unwrap();
    match step_value {
        KernelRlResponseV1::Error {
            request_id, error, ..
        } => {
            assert_eq!(request_id.as_deref(), Some("early-step"));
            assert_eq!(error.code, "step_before_reset");
        }
        other => panic!("expected step-before-reset error, got {other:?}"),
    }

    let malformed = server.handle_line("{not json");
    let malformed_value = parse_response(&malformed);
    assert_eq!(malformed_value["response_type"], "error");
    assert_eq!(malformed_value["error"]["code"], "malformed_json");

    let reset = server.handle_line(&reset_line("after-malformed", 16));
    let reset_value = parse_response(&reset);
    assert_eq!(reset_value["response_type"], "decision");
    assert_eq!(reset_value["request_id"], "after-malformed");
    assert!(!reset_value["legal_actions"].as_array().unwrap().is_empty());

    let active_malformed = server.handle_line("{not json");
    let active_malformed_value = parse_response(&active_malformed);
    assert_eq!(active_malformed_value["response_type"], "error");
    assert_eq!(active_malformed_value["error"]["code"], "malformed_json");

    let first_action = &reset_value["legal_actions"].as_array().unwrap()[0];
    let bad_schema = json!({
        "request_type": "step",
        "schema_version": RL_SESSION_SCHEMA_VERSION + 1,
        "request_id": "bad-schema",
        "episode_id": reset_value["episode_id"].as_u64().unwrap(),
        "expected_step": reset_value["step"].as_u64().unwrap(),
        "selected_index": first_action["selected_index"].as_u64().unwrap() as u32,
        "selected_action_id": first_action["stable_id"].as_str().unwrap(),
    })
    .to_string();
    let bad_schema_value = parse_response(&server.handle_line(&bad_schema));
    assert_eq!(bad_schema_value["response_type"], "error");
    assert_eq!(bad_schema_value["error"]["code"], "schema_version_mismatch");

    let valid = server.handle_line(&step_line_from_decision(
        "after-active-recovery-errors",
        &reset_value,
        0,
    ));
    let valid_value = parse_response(&valid);
    assert_ne!(valid_value["response_type"], "error");
}

#[test]
fn rl_session_batch_rollout_uses_session_path_and_stays_deterministic() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let a = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    let b = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();

    assert_eq!(a, b);
    let jsonl = a
        .records
        .iter()
        .map(|record| serde_json::to_string(record).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!jsonl.contains("diagnostic_state_hash_includes_hidden_state"));
    assert!(jsonl.contains("diagnostic_state_hash"));
}
