use mtg_kernel::card_def::card_id_by_name;
use mtg_kernel::ids::PlayerId;
use mtg_kernel::rl::{
    burn_deck_hash, card_name, derive_env_seed, derive_policy_seed, legal_action_candidates_v1,
    make_legal_action_v1, observe_v1, record_burn_mirror_episode, ActionSemanticV1, LegalActionV1,
    PlayerSeatV1, LEGAL_ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION,
};
use mtg_kernel::state::GameState;
use mtg_kernel::surface_v2::HarnessSurfaceV2;
use serde_json::Value;
use std::collections::BTreeSet;

fn ids(names: &[&str]) -> Vec<u16> {
    names
        .iter()
        .map(|name| {
            card_id_by_name(name).unwrap_or_else(|| panic!("{name} missing from CARD_DEFS"))
        })
        .collect()
}

fn hidden_information_state() -> GameState {
    let p0 = ids(&["Lightning Bolt", "Mountain", "Fireblast"]);
    let p1 = ids(&["Fiery Temper", "Lava Dart", "Highway Robbery"]);
    let mut state = GameState::new_from_libraries(&p0, &p1, card_name, 123);
    state.draw_card(PlayerId::P0);
    state.draw_card(PlayerId::P1);
    state
}

fn collect_arena_ids(value: &Value, out: &mut BTreeSet<u64>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::Number(n)) = map.get("arena_id") {
                if let Some(id) = n.as_u64() {
                    out.insert(id);
                }
            }
            for child in map.values() {
                collect_arena_ids(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_arena_ids(child, out);
            }
        }
        _ => {}
    }
}

fn records_to_jsonl(records: &[mtg_kernel::rl::EpisodeRecordV1]) -> String {
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record).unwrap());
        out.push('\n');
    }
    out
}

#[test]
fn rl_contract_serde_roundtrip_and_schema_versions() {
    let state = hidden_information_state();
    let obs = observe_v1(&state, PlayerId::P0, 7).unwrap();
    assert_eq!(obs.schema_version, OBSERVATION_SCHEMA_VERSION);
    assert_eq!(
        obs.surface_version,
        mtg_kernel::surface_v2::H2_PREDICATE_VERSION
    );
    assert_eq!(obs.card_db_hash, mtg_kernel::card_def::KERNEL_CARDDB_HASH);
    assert!(obs.diagnostic_state_hash_includes_hidden_state);
    assert_ne!(obs.visible_projection_hash, 0);

    let json = serde_json::to_string(&obs).unwrap();
    let roundtrip: mtg_kernel::rl::ObservationV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip, obs);

    let mut state = mtg_kernel::rl::build_burn_mirror_state(derive_env_seed(5151, 0));
    let mut surface = HarnessSurfaceV2::new();
    let decision = surface.next_decision(&mut state);
    let action = legal_action_candidates_v1(&decision, &state)
        .unwrap()
        .remove(0)
        .record;
    assert_eq!(action.schema_version, LEGAL_ACTION_SCHEMA_VERSION);
    let action_roundtrip: LegalActionV1 =
        serde_json::from_str(&serde_json::to_string(&action).unwrap()).unwrap();
    assert_eq!(action_roundtrip, action);
    assert_ne!(burn_deck_hash(), 0);
}

#[test]
fn rl_contract_observation_perspective_safety_excludes_hidden_identities() {
    let state = hidden_information_state();
    let obs = observe_v1(&state, PlayerId::P0, 0).unwrap();
    let json = serde_json::to_string(&obs).unwrap();
    assert!(
        json.contains("Lightning Bolt"),
        "acting player's own hand identity should be visible"
    );
    assert!(
        !json.contains("Fiery Temper"),
        "opponent hand identity leaked"
    );
    assert!(
        !json.contains("Mountain"),
        "acting player's library identity leaked"
    );
    assert!(
        !json.contains("Fireblast"),
        "acting player's library identity leaked"
    );
    assert!(
        !json.contains("Lava Dart"),
        "opponent library identity leaked"
    );
    assert!(
        !json.contains("Highway Robbery"),
        "opponent library identity leaked"
    );

    let value = serde_json::to_value(&obs).unwrap();
    let mut arena_ids = BTreeSet::new();
    collect_arena_ids(&value, &mut arena_ids);
    assert!(
        arena_ids.contains(&0),
        "own hand object id should be visible"
    );
    for hidden in [1, 2, 3, 4, 5] {
        assert!(
            !arena_ids.contains(&hidden),
            "hidden arena id {hidden} leaked"
        );
    }
}

#[test]
fn rl_contract_legal_action_ids_are_structured_unique_and_display_independent() {
    let mut state = mtg_kernel::rl::build_burn_mirror_state(derive_env_seed(5151, 0));
    let mut surface = HarnessSurfaceV2::new();
    let decision = surface.next_decision(&mut state);
    let actions = legal_action_candidates_v1(&decision, &state).unwrap();
    assert!(!actions.is_empty());

    let ids: BTreeSet<_> = actions.iter().map(|a| a.record.stable_id.clone()).collect();
    assert_eq!(
        ids.len(),
        actions.len(),
        "stable action ids must be unique within a decision"
    );

    let first = &actions[0].record;
    let roundtrip: LegalActionV1 =
        serde_json::from_str(&serde_json::to_string(first).unwrap()).unwrap();
    assert_eq!(roundtrip.stable_id, first.stable_id);
    assert_eq!(roundtrip.semantic, first.semantic);

    let a = make_legal_action_v1(
        0,
        first.semantic.clone(),
        Some("old diagnostic text".to_string()),
    )
    .unwrap();
    let b = make_legal_action_v1(
        0,
        first.semantic.clone(),
        Some("new diagnostic text".to_string()),
    )
    .unwrap();
    assert_eq!(
        a.stable_id, b.stable_id,
        "display text must not participate in the stable id"
    );
    assert_ne!(a.display_text, b.display_text);
}

#[test]
fn rl_contract_identical_seeds_produce_identical_episode_records() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let a = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    let b = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    assert_eq!(records_to_jsonl(&a.records), records_to_jsonl(&b.records));
    assert_eq!(a.terminal, b.terminal);
}

#[test]
fn rl_contract_different_perspectives_only_expose_that_players_hand() {
    let state = hidden_information_state();
    let p0 = observe_v1(&state, PlayerId::P0, 0).unwrap();
    let p1 = observe_v1(&state, PlayerId::P1, 0).unwrap();
    let p0_json = serde_json::to_string(&p0).unwrap();
    let p1_json = serde_json::to_string(&p1).unwrap();

    assert!(p0_json.contains("Lightning Bolt"));
    assert!(!p0_json.contains("Fiery Temper"));
    assert!(p1_json.contains("Fiery Temper"));
    assert!(!p1_json.contains("Lightning Bolt"));
    assert_eq!(p0.acting_player, PlayerSeatV1::P0);
    assert_eq!(p1.acting_player, PlayerSeatV1::P1);
}

#[test]
fn rl_contract_invalid_ambiguous_action_representation_fails_closed() {
    let err = make_legal_action_v1(
        0,
        ActionSemanticV1::Ambiguous {
            reason: "display-text-only candidate".to_string(),
        },
        Some("Cast the thing".to_string()),
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("ambiguous legal action representation refused"));
}
