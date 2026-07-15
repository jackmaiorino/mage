use mtg_kernel::rl::{derive_env_seed, derive_policy_seed, record_burn_mirror_episode};
use mtg_kernel::rl_session::{
    KernelRlJsonlServerV1, KernelRlResponseV1, RlEpisodeSessionV1, RlSessionErrorCode,
    RlSessionResponseV1, RL_SESSION_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn reset_line(request_id: &str, max_decisions: u64) -> String {
    json!({
        "request_type": "reset",
        "schema_version": RL_SESSION_SCHEMA_VERSION,
        "request_id": request_id,
        "episode_id": 0,
        "env_seed": derive_env_seed(5151, 0),
        "max_decisions": max_decisions,
    })
    .to_string()
}

fn step_line(request_id: &str, selected_index: u32, selected_action_id: &str) -> String {
    json!({
        "request_type": "step",
        "schema_version": RL_SESSION_SCHEMA_VERSION,
        "request_id": request_id,
        "selected_index": selected_index,
        "selected_action_id": selected_action_id,
    })
    .to_string()
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
                    let first = &value["legal_actions"].as_array().unwrap()[0];
                    let selected_id = first["stable_id"].as_str().unwrap();
                    current = server.handle_line(&step_line(
                        &format!("s{step}"),
                        first["selected_index"].as_u64().unwrap() as u32,
                        selected_id,
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
fn rl_session_invalid_index_and_id_errors_do_not_mutate() {
    let mut session = RlEpisodeSessionV1::reset(0, derive_env_seed(5151, 0), 32);
    let ids = action_ids(&session.current_response());
    assert!(
        ids.len() >= 2,
        "initial Burn decision should have multiple actions"
    );

    let before = current_snapshot(&session);
    let err = session.step(9999, &ids[0]).unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedIndexOutOfRange);
    assert_eq!(current_snapshot(&session), before);

    let before = current_snapshot(&session);
    let err = session.step(0, &ids[1]).unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedActionIdMismatch);
    assert_eq!(current_snapshot(&session), before);

    let before = current_snapshot(&session);
    let err = session
        .step(0, "legal-action-v1:ffffffffffffffff")
        .unwrap_err();
    assert_eq!(err.code, RlSessionErrorCode::SelectedActionIdUnknown);
    assert_eq!(current_snapshot(&session), before);
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
            let err = session.step(0, &stale_id).unwrap_err();
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
        session.step(0, &selected).unwrap();
        if matches!(session.current_response(), RlSessionResponseV1::Terminal(_)) {
            break;
        }
    }
    panic!("did not find a stale action id within the bounded smoke sequence");
}

#[test]
fn rl_session_step_before_reset_and_malformed_input_recover_cleanly() {
    let mut server = KernelRlJsonlServerV1::new();

    let step_before_reset = server.handle_line(&step_line("early-step", 0, "legal-action-v1:0"));
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
