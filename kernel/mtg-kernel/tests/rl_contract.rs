use mtg_kernel::card_def::{card_id_by_name, CardType, TargetSpec, CARD_DEFS};
use mtg_kernel::engine::{
    self, Action, CastMode, DiscardResume, EffectDuration, Layers, PendingCast, PendingDiscard,
    PlayOrCast, PlayPermission, PlayPermissionExpiry, UntilEndOfTurnEffect,
};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::rl::{
    build_run_manifest, burn_deck_hash, card_name, derive_env_seed, derive_policy_seed,
    legal_action_candidates_v1, make_legal_action_v1, observe_v2, record_burn_mirror_episode,
    ActionSemanticV1, EpisodeTerminalSummaryV1, GitDirtyFlagV1, GitMetadataV1, LegalActionV1,
    ObservationV2, PlayerSeatV1, TerminalClassificationV1, TerminalOutcomeV1,
    AUDIT_EPISODE_JSONL_FILENAME, LEGAL_ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION,
    POLICY_EPISODE_JSONL_FILENAME,
};
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Target, Zone};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SurfaceAction};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

fn ids(names: &[&str]) -> Vec<u16> {
    names
        .iter()
        .map(|name| {
            card_id_by_name(name).unwrap_or_else(|| panic!("{name} missing from CARD_DEFS"))
        })
        .collect()
}

fn hidden_information_state() -> GameState {
    hidden_information_state_with_p1(&["Fiery Temper", "Lava Dart", "Highway Robbery"])
}

fn hidden_information_state_with_p1(p1_names: &[&str]) -> GameState {
    let p0 = ids(&["Lightning Bolt", "Mountain", "Fireblast"]);
    let p1 = ids(p1_names);
    let mut state = GameState::new_from_libraries(&p0, &p1, card_name, 123);
    state.draw_card(PlayerId::P0);
    state.draw_card(PlayerId::P1);
    state
}

fn empty_state() -> GameState {
    GameState::new_from_libraries(&[], &[], card_name, 99)
}

fn make_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id_by_name(name).unwrap_or_else(|| panic!("{name} missing from CARD_DEFS"));
    let id = state.objects.push(GameObject {
        card_def,
        name: name.to_string(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Counters::default(),
        attachments: Vec::new(),
        plotted_turn: None,
        zone_change_count: 0,
    });
    match zone {
        Zone::Hand => state.players[player.index()].hand.push(id),
        Zone::Battlefield => state.players[player.index()].battlefield.push(id),
        Zone::Graveyard => state.players[player.index()].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Library => state.players[player.index()].library.push(id),
        Zone::Stack => {}
        Zone::Command => state.command.push(id),
    }
    id
}

fn observe_for_test(state: &GameState, acting_player: PlayerId, step: u64) -> ObservationV2 {
    let surface = HarnessSurfaceV2::new();
    observe_v2(state, &surface, acting_player, step).unwrap()
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

fn records_to_jsonl<T: Serialize>(records: &[T]) -> String {
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record).unwrap());
        out.push('\n');
    }
    out
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

fn first_battlefield_card<'a>(
    obs: &'a ObservationV2,
    player: PlayerSeatV1,
    name: &str,
) -> &'a mtg_kernel::rl::CardPublicV2 {
    let idx = match player {
        PlayerSeatV1::P0 => 0,
        PlayerSeatV1::P1 => 1,
    };
    obs.projection.battlefield[idx]
        .iter()
        .find(|card| card.card_name == name)
        .unwrap_or_else(|| panic!("{name} missing from observed battlefield"))
}

#[test]
fn rl_contract_serde_roundtrip_and_schema_versions() {
    let state = hidden_information_state();
    let obs = observe_for_test(&state, PlayerId::P0, 7);
    assert_eq!(obs.schema_version, OBSERVATION_SCHEMA_VERSION);
    assert_eq!(
        obs.surface_version,
        mtg_kernel::surface_v2::H2_PREDICATE_VERSION
    );
    assert_eq!(obs.card_db_hash, mtg_kernel::card_def::KERNEL_CARDDB_HASH);
    assert_ne!(obs.visible_projection_hash, 0);

    let json = serde_json::to_string(&obs).unwrap();
    let roundtrip: ObservationV2 = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip, obs);

    let mut state = mtg_kernel::rl::build_burn_mirror_state(derive_env_seed(5151, 0));
    let mut surface = HarnessSurfaceV2::new();
    let decision = surface.next_decision(&mut state);
    let action = legal_action_candidates_v1(&decision, &state)
        .unwrap()
        .remove(0)
        .record;
    assert_eq!(action.schema_version, LEGAL_ACTION_SCHEMA_VERSION);
    assert!(action.stable_id.starts_with("legal-action-v1:"));
    let action_roundtrip: LegalActionV1 =
        serde_json::from_str(&serde_json::to_string(&action).unwrap()).unwrap();
    assert_eq!(action_roundtrip, action);
    assert_ne!(burn_deck_hash(), 0);
}

#[test]
fn rl_contract_observation_is_byte_invariant_to_opponent_hidden_identities() {
    let a = hidden_information_state_with_p1(&["Fiery Temper", "Lava Dart", "Highway Robbery"]);
    let b = hidden_information_state_with_p1(&["Grab the Prize", "Fireblast", "Mountain"]);

    assert_eq!(a.turn, b.turn);
    assert_eq!(a.active_player, b.active_player);
    assert_eq!(a.priority_player, b.priority_player);
    assert_eq!(a.step, b.step);
    assert_eq!(a.players[0].hand.len(), b.players[0].hand.len());
    assert_eq!(a.players[1].hand.len(), b.players[1].hand.len());
    assert_eq!(a.players[0].library.len(), b.players[0].library.len());
    assert_eq!(a.players[1].library.len(), b.players[1].library.len());
    assert_ne!(
        a.state_hash(),
        b.state_hash(),
        "internal full-state hash must still detect hidden identity/order changes"
    );

    let obs_a = observe_for_test(&a, PlayerId::P0, 11);
    let obs_b = observe_for_test(&b, PlayerId::P0, 11);
    assert_eq!(obs_a.visible_projection_hash, obs_b.visible_projection_hash);
    assert_eq!(
        serde_json::to_vec(&obs_a).unwrap(),
        serde_json::to_vec(&obs_b).unwrap(),
        "serialized policy observation must be byte-identical"
    );
}

#[test]
fn rl_contract_observation_perspective_safety_excludes_hidden_identities() {
    let state = hidden_information_state();
    let obs = observe_for_test(&state, PlayerId::P0, 0);
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
fn rl_contract_public_combat_assignment_order_changes_observation_hash() {
    let mut a = empty_state();
    let attacker = make_object(
        &mut a,
        PlayerId::P0,
        "Goblin Tomb Raider",
        Zone::Battlefield,
    );
    let blocker_0 = make_object(
        &mut a,
        PlayerId::P1,
        "Human Soldier Token",
        Zone::Battlefield,
    );
    let blocker_1 = make_object(&mut a, PlayerId::P1, "Voldaren Epicure", Zone::Battlefield);
    a.engine.combat.attackers_declared = true;
    a.engine.combat.blockers_declared = true;
    a.engine.combat.attackers = vec![attacker];
    a.engine.combat.blocked_by = vec![(attacker, vec![blocker_0, blocker_1])];

    let mut b = a.clone();
    b.engine.combat.blocked_by = vec![(attacker, vec![blocker_1, blocker_0])];

    let obs_a = observe_for_test(&a, PlayerId::P0, 4);
    let obs_b = observe_for_test(&b, PlayerId::P0, 4);
    assert_ne!(obs_a.visible_projection_hash, obs_b.visible_projection_hash);
    assert_ne!(
        serde_json::to_vec(&obs_a.projection.combat).unwrap(),
        serde_json::to_vec(&obs_b.projection.combat).unwrap()
    );
}

#[test]
fn rl_contract_effective_characteristics_match_engine_queries() {
    let mut state = empty_state();
    let raider = make_object(
        &mut state,
        PlayerId::P0,
        "Goblin Tomb Raider",
        Zone::Battlefield,
    );
    let obs_without_artifact = observe_for_test(&state, PlayerId::P0, 0);
    let raider_without = first_battlefield_card(
        &obs_without_artifact,
        PlayerSeatV1::P0,
        "Goblin Tomb Raider",
    );
    assert_eq!(
        raider_without.characteristics.effective_power,
        Some(engine::effective_power(&state, raider))
    );
    assert_eq!(
        raider_without.characteristics.effective_keywords.haste,
        engine::has_effective_keyword(&state, raider, mtg_kernel::card_def::Keywords::HASTE)
    );

    make_object(&mut state, PlayerId::P0, "Blood Token", Zone::Battlefield);
    let obs_with_artifact = observe_for_test(&state, PlayerId::P0, 1);
    let raider_with =
        first_battlefield_card(&obs_with_artifact, PlayerSeatV1::P0, "Goblin Tomb Raider");
    assert_ne!(
        raider_without.characteristics.effective_power,
        raider_with.characteristics.effective_power
    );
    assert_eq!(
        raider_with.characteristics.effective_power,
        Some(engine::effective_power(&state, raider))
    );
    assert_eq!(
        raider_with.characteristics.effective_keywords.haste,
        engine::has_effective_keyword(&state, raider, mtg_kernel::card_def::Keywords::HASTE)
    );

    state
        .engine
        .until_end_of_turn
        .push(UntilEndOfTurnEffect::ResolvedSetEffect {
            object_ids: vec![raider],
            layer: Layers::ABILITY_ADDING | Layers::POWER_TOUGHNESS,
            timestamp: 7,
            duration: EffectDuration::EndOfTurn,
            power: 1,
            toughness: 0,
            grant_haste: true,
        });
    let obs_pumped = observe_for_test(&state, PlayerId::P0, 2);
    let raider_pumped = first_battlefield_card(&obs_pumped, PlayerSeatV1::P0, "Goblin Tomb Raider");
    assert_eq!(
        raider_pumped.characteristics.effective_power,
        Some(engine::effective_power(&state, raider))
    );
    assert!(raider_pumped.characteristics.effective_keywords.haste);
    assert_eq!(obs_pumped.projection.continuous_effects.len(), 1);
}

#[test]
fn rl_contract_active_exile_permission_holder_and_expiry_are_public() {
    let mut base = empty_state();
    let bolt = make_object(&mut base, PlayerId::P0, "Lightning Bolt", Zone::Exile);
    base.objects.get_mut(bolt).zone_change_count = 3;

    let mut holder_p0 = base.clone();
    holder_p0
        .engine
        .exile_play_permissions
        .push(PlayPermission {
            object: bolt,
            holder: PlayerId::P0,
            zone_change_generation: 3,
            play_or_cast: PlayOrCast::Cast,
            expiry: PlayPermissionExpiry::EndOfTurn,
        });
    let mut holder_p1 = holder_p0.clone();
    holder_p1.engine.exile_play_permissions[0].holder = PlayerId::P1;
    let mut next_turn = holder_p0.clone();
    next_turn.engine.exile_play_permissions[0].expiry =
        PlayPermissionExpiry::UntilHoldersNextTurn {
            holder_turn_started: false,
        };
    let mut stale = holder_p0.clone();
    stale.engine.exile_play_permissions[0].zone_change_generation = 2;

    let obs_p0 = observe_for_test(&holder_p0, PlayerId::P0, 0);
    let obs_p1 = observe_for_test(&holder_p1, PlayerId::P0, 0);
    let obs_next = observe_for_test(&next_turn, PlayerId::P0, 0);
    let obs_stale = observe_for_test(&stale, PlayerId::P0, 0);

    assert_ne!(
        obs_p0.visible_projection_hash,
        obs_p1.visible_projection_hash
    );
    assert_ne!(
        obs_p0.visible_projection_hash,
        obs_next.visible_projection_hash
    );
    assert_eq!(obs_p0.projection.exile_play_permissions.len(), 1);
    assert!(
        obs_stale.projection.exile_play_permissions.is_empty(),
        "stale/void permissions must be excluded"
    );
}

#[test]
fn rl_contract_h2_blocker_and_discard_reshape_context_changes_hash() {
    let mut state_a = empty_state();
    state_a.step = Step::DeclareBlockers;
    state_a.active_player = PlayerId::P0;
    state_a.priority_player = PlayerId::P1;
    let attacker_0 = make_object(
        &mut state_a,
        PlayerId::P0,
        "Goblin Tomb Raider",
        Zone::Battlefield,
    );
    let attacker_1 = make_object(&mut state_a, PlayerId::P0, "Guttersnipe", Zone::Battlefield);
    let blocker_0 = make_object(
        &mut state_a,
        PlayerId::P1,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let blocker_1 = make_object(
        &mut state_a,
        PlayerId::P1,
        "Human Soldier Token",
        Zone::Battlefield,
    );
    state_a.engine.combat.attackers_declared = true;
    state_a.engine.combat.attackers = vec![attacker_0, attacker_1];

    let mut state_b = state_a.clone();
    let mut surface_a = HarnessSurfaceV2::new();
    let mut surface_b = HarnessSurfaceV2::new();
    surface_a.next_decision(&mut state_a);
    surface_b.next_decision(&mut state_b);
    surface_a
        .apply(
            &mut state_a,
            SurfaceAction::DeclareBlockersForAttacker(vec![blocker_0]),
        )
        .unwrap();
    surface_b
        .apply(
            &mut state_b,
            SurfaceAction::DeclareBlockersForAttacker(vec![blocker_1]),
        )
        .unwrap();
    let blocker_obs_a = observe_v2(&state_a, &surface_a, PlayerId::P1, 10).unwrap();
    let blocker_obs_b = observe_v2(&state_b, &surface_b, PlayerId::P1, 10).unwrap();
    assert_ne!(
        blocker_obs_a.visible_projection_hash,
        blocker_obs_b.visible_projection_hash
    );

    let mut discard_a = empty_state();
    let card_0 = make_object(&mut discard_a, PlayerId::P0, "Lightning Bolt", Zone::Hand);
    let card_1 = make_object(&mut discard_a, PlayerId::P0, "Mountain", Zone::Hand);
    let card_2 = make_object(&mut discard_a, PlayerId::P0, "Fireblast", Zone::Hand);
    discard_a.engine.pending_discard = Some(PendingDiscard {
        player: PlayerId::P0,
        count: 2,
        resume: DiscardResume::None,
    });
    let mut discard_b = discard_a.clone();
    let mut discard_surface_a = HarnessSurfaceV2::new();
    let mut discard_surface_b = HarnessSurfaceV2::new();
    discard_surface_a.next_decision(&mut discard_a);
    discard_surface_b.next_decision(&mut discard_b);
    discard_surface_a
        .apply(
            &mut discard_a,
            SurfaceAction::Action(Action::Discard(vec![card_0])),
        )
        .unwrap();
    discard_surface_b
        .apply(
            &mut discard_b,
            SurfaceAction::Action(Action::Discard(vec![card_1])),
        )
        .unwrap();
    let discard_obs_a = observe_v2(&discard_a, &discard_surface_a, PlayerId::P0, 11).unwrap();
    let discard_obs_b = observe_v2(&discard_b, &discard_surface_b, PlayerId::P0, 11).unwrap();
    assert_ne!(
        discard_obs_a.visible_projection_hash,
        discard_obs_b.visible_projection_hash
    );
    assert!(discard_obs_a
        .projection
        .surface_context
        .discard
        .as_ref()
        .unwrap()
        .chosen
        .contains(&card_0));
    assert!(discard_obs_b
        .projection
        .surface_context
        .discard
        .as_ref()
        .unwrap()
        .chosen
        .contains(&card_1));
    assert!(discard_obs_a
        .projection
        .surface_context
        .discard
        .as_ref()
        .unwrap()
        .remaining_choices
        .contains(&card_2));
}

#[test]
fn rl_contract_engine_pending_cast_context_changes_hash() {
    let mut a = empty_state();
    let spell = make_object(&mut a, PlayerId::P0, "Lightning Bolt", Zone::Stack);
    a.engine.pending_cast = Some(PendingCast {
        spell,
        controller: PlayerId::P0,
        target_spec: TargetSpec::AnyTarget,
        targets_chosen: vec![Target::Player(PlayerId::P1)],
        is_flashback: false,
        cast_mode: Some(CastMode::Normal),
        additional_cost_discarded: Some(Vec::new()),
        cost_override: None,
        mode_chosen: Some(0),
        origin_zone: Zone::Hand,
        sacrifice_chosen: Vec::new(),
        kicked: Some(false),
    });
    let mut b = a.clone();
    b.engine.pending_cast.as_mut().unwrap().targets_chosen = vec![Target::Player(PlayerId::P0)];

    let obs_a = observe_for_test(&a, PlayerId::P0, 20);
    let obs_b = observe_for_test(&b, PlayerId::P0, 20);
    assert_ne!(obs_a.visible_projection_hash, obs_b.visible_projection_hash);
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
fn rl_contract_identical_seeds_produce_identical_policy_and_audit_records() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let a = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    let b = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    assert_eq!(
        records_to_jsonl(&a.audit_records),
        records_to_jsonl(&b.audit_records)
    );
    assert_eq!(
        records_to_jsonl(&a.policy_records),
        records_to_jsonl(&b.policy_records)
    );
    assert_eq!(a.terminal, b.terminal);
}

#[test]
fn rl_contract_policy_stream_is_safe_and_audit_stream_is_privileged() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let run = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    let policy_jsonl = records_to_jsonl(&run.policy_records);
    let audit_jsonl = records_to_jsonl(&run.audit_records);

    for line in policy_jsonl.lines() {
        let value: Value = serde_json::from_str(line).unwrap();
        for forbidden in [
            "diagnostic_state_hash",
            "state_hash",
            "env_seed",
            "policy_seed",
            "library_setup",
            "rng_state",
            "hidden_state_marker",
            "event_history",
        ] {
            assert!(
                !contains_key(&value, forbidden),
                "policy-safe record contains forbidden key {forbidden}: {line}"
            );
        }
    }
    assert!(!policy_jsonl.contains("privileged"));
    assert!(audit_jsonl.contains("diagnostic_state_hash"));
    assert!(audit_jsonl.contains("env_seed"));
    assert!(audit_jsonl.contains("policy_seed"));
}

#[test]
fn rl_contract_different_perspectives_only_expose_that_players_hand() {
    let state = hidden_information_state();
    let p0 = observe_for_test(&state, PlayerId::P0, 0);
    let p1 = observe_for_test(&state, PlayerId::P1, 0);
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

#[test]
fn rl_contract_terminal_outcome_accounting_is_explicit() {
    let summaries = vec![
        EpisodeTerminalSummaryV1 {
            episode_id: 0,
            outcome: TerminalOutcomeV1::P0Win,
            classification: TerminalClassificationV1::Natural,
            winner: Some(PlayerSeatV1::P0),
            terminal_reward: [1, -1],
            terminal_reason: "game_over".to_string(),
            decision_count: 10,
        },
        EpisodeTerminalSummaryV1 {
            episode_id: 1,
            outcome: TerminalOutcomeV1::P1Win,
            classification: TerminalClassificationV1::Natural,
            winner: Some(PlayerSeatV1::P1),
            terminal_reward: [-1, 1],
            terminal_reason: "game_over".to_string(),
            decision_count: 11,
        },
        EpisodeTerminalSummaryV1 {
            episode_id: 2,
            outcome: TerminalOutcomeV1::Draw,
            classification: TerminalClassificationV1::Natural,
            winner: None,
            terminal_reward: [0, 0],
            terminal_reason: "game_over".to_string(),
            decision_count: 12,
        },
        EpisodeTerminalSummaryV1 {
            episode_id: 3,
            outcome: TerminalOutcomeV1::Truncated,
            classification: TerminalClassificationV1::Truncated,
            winner: None,
            terminal_reward: [0, 0],
            terminal_reason: "decision_cap_reached:12".to_string(),
            decision_count: 12,
        },
        EpisodeTerminalSummaryV1 {
            episode_id: 4,
            outcome: TerminalOutcomeV1::Halted,
            classification: TerminalClassificationV1::Halted,
            winner: None,
            terminal_reward: [0, 0],
            terminal_reason: "fail_closed:test".to_string(),
            decision_count: 12,
        },
    ];
    let manifest = build_run_manifest(
        5,
        5151,
        12,
        vec![],
        Path::new("local-training/kernel_rl/test"),
        &summaries,
        GitMetadataV1 {
            commit: "test".to_string(),
            dirty: GitDirtyFlagV1::Clean,
        },
    );
    assert_eq!(manifest.aggregate.p0_wins, 1);
    assert_eq!(manifest.aggregate.p1_wins, 1);
    assert_eq!(manifest.aggregate.draws, 1);
    assert_eq!(manifest.aggregate.truncated, 1);
    assert_eq!(manifest.aggregate.halted, 1);
    assert_eq!(manifest.aggregate.total_decisions, 57);

    let value = serde_json::to_value(&manifest).unwrap();
    assert!(value["aggregate"].get("wins").is_none());
    assert!(value["aggregate"].get("losses").is_none());
    assert_eq!(value["aggregate"]["p0_wins"], 1);
    assert_eq!(value["aggregate"]["p1_wins"], 1);
    assert_eq!(
        value["output_files"]["policy_episode_jsonl"],
        POLICY_EPISODE_JSONL_FILENAME
    );
    assert_eq!(
        value["output_files"]["audit_episode_jsonl"],
        AUDIT_EPISODE_JSONL_FILENAME
    );
    assert!(manifest.streams[0].policy_safe);
    assert!(manifest.streams[1].contains_hidden_state_diagnostics);
}

#[test]
fn rl_contract_card_type_flags_are_structured() {
    let mut state = empty_state();
    make_object(&mut state, PlayerId::P0, "Blood Token", Zone::Battlefield);
    let obs = observe_for_test(&state, PlayerId::P0, 0);
    let blood = first_battlefield_card(&obs, PlayerSeatV1::P0, "Blood Token");
    let def = &CARD_DEFS[blood.stable.card_db_id as usize];
    assert!(def.has_type(CardType::Artifact));
    assert!(blood.characteristics.type_flags.artifact);
    assert!(!blood.characteristics.type_flags.creature);
}
