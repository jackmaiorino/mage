use mtg_kernel::card_def::{card_id_by_name, TargetSpec, CARD_DEFS};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const FIXTURE_JSON: &str = include_str!("../../data/xmage_counter_reference_windows_v1.json");
const OPPONENT_DECISION_PREFIX: &str = "REPLAY_OPPONENT_DECISION_JSON: ";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE_JSON).expect("counter reference fixture is valid JSON")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_uuid(value: &str) -> bool {
    let parts: Vec<&str> = value.split('-').collect();
    parts.len() == 5
        && parts.iter().zip([8, 4, 4, 4, 12]).all(|(part, len)| {
            part.len() == len
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn count_entries(value: &Value) -> BTreeMap<String, u64> {
    let entries = value.as_array().expect("graveyard additions are an array");
    let mut out = BTreeMap::new();
    let mut source_order = Vec::new();
    for entry in entries {
        let name = entry["name"]
            .as_str()
            .expect("graveyard addition name")
            .to_string();
        let count = entry["count"].as_u64().expect("graveyard addition count");
        assert!(count > 0, "graveyard addition counts must be positive");
        assert!(
            out.insert(name.clone(), count).is_none(),
            "duplicate {name}"
        );
        source_order.push(name);
    }
    let sorted: Vec<String> = out.keys().cloned().collect();
    assert_eq!(source_order, sorted, "graveyard additions stay canonical");
    out
}

#[test]
fn tracked_counter_windows_are_structured_and_match_kernel_card_types() {
    let root = fixture();
    assert_eq!(root["schema"], "kernel_xmage_counter_reference_windows/v1");
    assert_eq!(
        root["evidence_scope"],
        "source-hash-backed XMage micro-reference windows; not fixed-provenance whole-deck parity"
    );
    assert!(root["oracle"]["java_oracle_commit"].is_null());
    assert_eq!(
        root["oracle"]["provenance_status"],
        "trace_and_manifest_digests_pinned_commit_unrecorded"
    );
    assert!(is_lower_sha256(
        root["run_manifest"]["sha256"].as_str().unwrap()
    ));

    let traces = root["traces"].as_array().expect("traces array");
    assert_eq!(traces.len(), 3);
    let mut trace_ids = BTreeSet::new();
    let mut trace_hashes = BTreeSet::new();
    let mut total_records = 0;

    for trace in traces {
        assert!(trace_ids.insert(trace["id"].as_str().unwrap()));
        let trace_hash = trace["sha256"].as_str().unwrap();
        assert!(is_lower_sha256(trace_hash));
        assert!(trace_hashes.insert(trace_hash));
        assert!(trace["bytes"].as_u64().unwrap() > 0);
        assert!(trace["scenario"].as_u64().unwrap() > 0);
        assert!(trace["seed"].as_u64().unwrap() > 0);
        trace["random_seed"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .expect("random seed is exact decimal u64");

        let records = trace["records"].as_array().expect("records array");
        total_records += records.len();
        let mut prior_index = None;
        for record in records {
            let decision_index = record["decision_index"].as_u64().unwrap();
            if let Some(prior) = prior_index {
                assert!(decision_index > prior, "record order is canonical");
            }
            prior_index = Some(decision_index);
            assert!(record["line"].as_u64().unwrap() > 0);
            assert!(is_lower_sha256(record["record_sha256"].as_str().unwrap()));

            let source = record["source_card"].as_str().unwrap();
            let source_id = record["source_card_object_id"].as_str().unwrap();
            let target = record["target_card"].as_str().unwrap();
            let target_id = record["target_card_object_id"].as_str().unwrap();
            let target_stack_id = record["target_stack_object_id"].as_str().unwrap();
            for id in [source_id, target_id, target_stack_id] {
                assert!(is_lower_uuid(id), "invalid UUID {id}");
            }

            let action = record["action"].as_str().unwrap();
            assert!(
                action.starts_with(&format!(
                    "{source} [{}]: Cast {source} -> ",
                    &source_id[..3]
                )),
                "source tag must link to the full card UUID: {action}"
            );
            assert!(
                action.ends_with(&format!("{target} [{}]", &target_id[..3])),
                "target tag must link to the full card UUID: {action}"
            );

            let source_def = &CARD_DEFS[card_id_by_name(source)
                .unwrap_or_else(|| panic!("missing source {source}"))
                as usize];
            assert!(source_def.has_full_support(), "{source}");
            assert_eq!(
                source_def.target_spec,
                if source == "Dispel" {
                    TargetSpec::InstantSpellOnStack
                } else {
                    TargetSpec::AnySpellOnStack
                }
            );

            let target_def = &CARD_DEFS[card_id_by_name(target)
                .unwrap_or_else(|| panic!("missing target {target}"))
                as usize];
            let actual_types: Vec<String> = target_def
                .types
                .iter()
                .map(|card_type| format!("{card_type:?}"))
                .collect();
            let expected_types: Vec<String> = record["target_card_types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_string())
                .collect();
            assert_eq!(actual_types, expected_types, "{target}");
            if source == "Dispel" {
                assert!(actual_types.iter().any(|kind| kind == "Instant"));
            }
        }

        let window = &trace["state_window"];
        let start = window["start_decision_index"].as_u64().unwrap();
        let end = window["end_decision_index"].as_u64().unwrap();
        assert!(start < end);
        assert!(records.iter().all(|record| {
            let index = record["decision_index"].as_u64().unwrap();
            (start..=end).contains(&index)
        }));
        assert!(is_lower_sha256(
            window["start_visible_state_sha256"].as_str().unwrap()
        ));
        assert!(is_lower_sha256(
            window["end_visible_state_sha256"].as_str().unwrap()
        ));
        count_entries(&window["self_graveyard_added"]);
        count_entries(&window["opponent_graveyard_added"]);
    }
    assert_eq!(total_records, 6);
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_and_verify(root: &Path, spec: &Value) -> Vec<u8> {
    let path = root.join(spec["path"].as_str().unwrap());
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len() as u64, spec["bytes"].as_u64().unwrap());
    assert_eq!(sha256(&bytes), spec["sha256"].as_str().unwrap());
    bytes
}

fn opponent_record(text: &str, decision_index: u64) -> (usize, String, Value) {
    for (offset, line) in text.lines().enumerate() {
        let Some(payload) = line.strip_prefix(OPPONENT_DECISION_PREFIX) else {
            continue;
        };
        let value: Value = serde_json::from_str(payload).expect("opponent decision JSON");
        if value["decision_index"].as_u64() == Some(decision_index) {
            return (offset + 1, payload.to_string(), value);
        }
    }
    panic!("missing opponent decision {decision_index}");
}

fn visible_zone(state: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key}=");
    let raw = state
        .split(';')
        .find_map(|field| field.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing visible-state field {key}"));
    if raw.is_empty() {
        Vec::new()
    } else {
        raw.split('|').map(str::to_string).collect()
    }
}

fn positive_multiset_delta(before: Vec<String>, after: Vec<String>) -> BTreeMap<String, u64> {
    fn counts(values: Vec<String>) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        for value in values {
            *out.entry(value).or_insert(0) += 1;
        }
        out
    }

    let before = counts(before);
    let after = counts(after);
    let mut delta = BTreeMap::new();
    for (name, after_count) in after {
        let before_count = before.get(&name).copied().unwrap_or(0);
        if after_count > before_count {
            delta.insert(name, after_count - before_count);
        }
    }
    delta
}

#[test]
#[ignore = "requires the ignored local XMage golden trace material"]
fn source_traces_match_fixture() {
    let root = repository_root();
    let fixture = fixture();
    read_and_verify(&root, &fixture["run_manifest"]);

    for trace in fixture["traces"].as_array().unwrap() {
        let bytes = read_and_verify(&root, trace);
        let text = std::str::from_utf8(&bytes).expect("trace is UTF-8");
        let scenario = trace["scenario"].as_u64().unwrap();
        let seed = trace["seed"].as_u64().unwrap();
        let random_seed = trace["random_seed"].as_str().unwrap();
        assert!(text.contains(&format!(
            "REPLAY_RANDOM: scenario={scenario} seed={seed} random_util_seed={random_seed} scope=league_bench"
        )));

        for record in trace["records"].as_array().unwrap() {
            let index = record["decision_index"].as_u64().unwrap();
            let (line, payload, value) = opponent_record(text, index);
            assert_eq!(line as u64, record["line"].as_u64().unwrap());
            assert_eq!(sha256(payload.as_bytes()), record["record_sha256"]);
            assert_eq!(value["scenario"].as_u64(), Some(scenario));
            assert_eq!(value["seed"].as_u64(), Some(seed));
            assert_eq!(value["actor"], "EvalBot-Skill7");
            assert_eq!(value["chosen_action_text"], record["action"]);
            assert_eq!(value["source_object_id"], record["source_card_object_id"]);
            assert_eq!(
                value["target_object_ids"][0],
                record["target_stack_object_id"]
            );
        }

        let window = &trace["state_window"];
        let (_, _, start) = opponent_record(text, window["start_decision_index"].as_u64().unwrap());
        let (_, _, end) = opponent_record(text, window["end_decision_index"].as_u64().unwrap());
        let start_state = start["visible_state"].as_str().unwrap();
        let end_state = end["visible_state"].as_str().unwrap();
        assert_eq!(
            sha256(start_state.as_bytes()),
            window["start_visible_state_sha256"]
        );
        assert_eq!(
            sha256(end_state.as_bytes()),
            window["end_visible_state_sha256"]
        );
        assert_eq!(
            positive_multiset_delta(
                visible_zone(start_state, "selfGraveyard"),
                visible_zone(end_state, "selfGraveyard"),
            ),
            count_entries(&window["self_graveyard_added"])
        );
        assert_eq!(
            positive_multiset_delta(
                visible_zone(start_state, "oppGraveyard"),
                visible_zone(end_state, "oppGraveyard"),
            ),
            count_entries(&window["opponent_graveyard_added"])
        );
    }
}
