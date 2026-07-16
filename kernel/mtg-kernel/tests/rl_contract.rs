use mtg_kernel::card_def::{card_id_by_name, CardType, TargetSpec, CARD_DEFS};
use mtg_kernel::engine::{
    self, Action, CastMode, DiscardResume, EffectDuration, Layers, PendingCast, PendingDiscard,
    PendingSpellCopy, PlayOrCast, PlayPermission, PlayPermissionExpiry, SpellCopyStage,
    UntilEndOfTurnEffect,
};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::rl::{
    build_run_manifest, burn_deck_hash, card_name, derive_env_seed, derive_policy_seed,
    legal_action_candidates_v1, make_legal_action_v1, observe_v2, parse_audit_episode_jsonl,
    parse_policy_episode_jsonl, parse_run_manifest_json, record_burn_mirror_episode,
    validate_policy_episode_records, validate_rollout_artifact_bundle, write_rollout_artifacts,
    ActionSemanticV1, EngineDecisionStageV2, EpisodeTerminalSummaryV1, GitDirtyFlagV1,
    GitMetadataV1, LegalActionV1, ObservationV2, PlayerSeatV1, PolicyEpisodeRecordV2,
    SpellCopyStageV2, TerminalClassificationV1, TerminalOutcomeV1, AUDIT_EPISODE_JSONL_FILENAME,
    AUDIT_EPISODE_SCHEMA_VERSION, LEGAL_ACTION_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION, POLICY_EPISODE_JSONL_FILENAME, POLICY_EPISODE_SCHEMA_VERSION,
};
use mtg_kernel::state::{
    Counters, GameObject, GameState, StackItem, StackItemKind, Step, Target, Zone,
    DIAGNOSTIC_STATE_HASH_ALGORITHM,
};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SurfaceAction, SurfaceDecision};
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
        v4: mtg_kernel::state::ObjectStateV4::from_card_def(card_def),
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
    assert!(action.stable_id.starts_with("legal-action-v4:"));
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
fn rl_contract_library_knowledge_is_visible_only_to_the_informed_observer() {
    let p0 = ids(&["Lightning Bolt", "Mountain", "Fireblast"]);
    let p1 = ids(&["Fiery Temper", "Lava Dart", "Highway Robbery"]);
    let mut informed = GameState::new_from_libraries(&p0, &p1, card_name, 123);
    let uninformed = informed.clone();
    informed.reveal_library_top(PlayerId::P0, PlayerId::P1, 2);

    let p0_obs = observe_for_test(&informed, PlayerId::P0, 3);
    assert_eq!(p0_obs.known_library_cards[1].len(), 2);
    assert_eq!(p0_obs.known_library_cards[1][0].position, 0);
    assert_eq!(
        p0_obs.known_library_cards[1][0].card.card_name,
        "Fiery Temper"
    );
    assert_eq!(p0_obs.known_library_cards[1][1].card.card_name, "Lava Dart");
    assert_ne!(
        p0_obs.visible_projection_hash,
        observe_for_test(&uninformed, PlayerId::P0, 3).visible_projection_hash
    );

    let p1_informed_state_obs = observe_for_test(&informed, PlayerId::P1, 3);
    let p1_uninformed_state_obs = observe_for_test(&uninformed, PlayerId::P1, 3);
    assert!(p1_informed_state_obs.known_library_cards[0].is_empty());
    assert!(p1_informed_state_obs.known_library_cards[1].is_empty());
    assert_eq!(
        serde_json::to_vec(&p1_informed_state_obs).unwrap(),
        serde_json::to_vec(&p1_uninformed_state_obs).unwrap(),
        "another observer's private library knowledge leaked into P1's bytes"
    );
}

#[test]
fn rl_contract_revealed_opponent_hand_knowledge_is_scoped_hashed_and_invalidated() {
    let mut informed = empty_state();
    let card = make_object(&mut informed, PlayerId::P1, "Fiery Temper", Zone::Hand);
    let uninformed = informed.clone();
    let uninformed_hash = informed.state_hash();

    informed
        .reveal_hand_card(PlayerId::P0, PlayerId::P1, card)
        .unwrap();
    assert_ne!(informed.state_hash(), uninformed_hash);
    let knowledge_snapshot = informed.snapshot();
    let knowledge_hash = informed.state_hash();

    let p0_view = observe_for_test(&informed, PlayerId::P0, 9);
    assert!(p0_view.known_hand_cards[0].is_empty());
    assert_eq!(p0_view.known_hand_cards[1].len(), 1);
    assert_eq!(p0_view.known_hand_cards[1][0].card_name, "Fiery Temper");
    assert_ne!(
        p0_view.visible_projection_hash,
        observe_for_test(&uninformed, PlayerId::P0, 9).visible_projection_hash
    );

    let p1_informed = observe_for_test(&informed, PlayerId::P1, 9);
    let p1_uninformed = observe_for_test(&uninformed, PlayerId::P1, 9);
    assert!(p1_informed.known_hand_cards[1].is_empty());
    assert_eq!(
        serde_json::to_vec(&p1_informed).unwrap(),
        serde_json::to_vec(&p1_uninformed).unwrap(),
        "another observer's private hand knowledge leaked into the owner's bytes"
    );

    let generation_before = informed.objects.get(card).zone_change_count;
    assert!(informed.move_hand_to_battlefield(PlayerId::P1, card));
    assert_eq!(
        informed.objects.get(card).zone_change_count,
        generation_before + 1
    );
    assert!(
        observe_for_test(&informed, PlayerId::P0, 10).known_hand_cards[1].is_empty(),
        "a revealed hand identity must disappear when that incarnation leaves hand"
    );
    assert!(
        observe_for_test(&informed, PlayerId::P1, 10).known_hand_cards[1].is_empty(),
        "own hand knowledge is represented only by own_hand, never duplicated here"
    );

    informed.restore(&knowledge_snapshot);
    assert_eq!(informed.state_hash(), knowledge_hash);
    assert_eq!(
        observe_for_test(&informed, PlayerId::P0, 9).known_hand_cards[1][0].card_name,
        "Fiery Temper"
    );
}

#[test]
fn rl_contract_paid_cost_refs_keep_known_hand_cards_without_leaking_unknown_ones() {
    let mut state = empty_state();
    let source = make_object(
        &mut state,
        PlayerId::P1,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let known = make_object(&mut state, PlayerId::P1, "Fiery Temper", Zone::Hand);
    let unknown = make_object(&mut state, PlayerId::P1, "Lava Dart", Zone::Hand);
    state
        .reveal_hand_card(PlayerId::P0, PlayerId::P1, known)
        .unwrap();
    let known_ref = mtg_kernel::state::PaidCostRefV4::capture(&state, known);
    let unknown_ref = mtg_kernel::state::PaidCostRefV4::capture(&state, unknown);
    state.stack.push(StackItem {
        kind: StackItemKind::TriggeredAbility,
        source,
        controller: PlayerId::P1,
        targets: Vec::new(),
        is_copy: false,
        inline_effect: None,
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false,
        v4: mtg_kernel::state::StackStateV4 {
            paid_cost_refs: vec![known_ref, unknown_ref],
            ..mtg_kernel::state::StackStateV4::default()
        },
    });

    let informed = observe_for_test(&state, PlayerId::P0, 4);
    assert_eq!(informed.projection.stack[0].paid_cost_refs.len(), 1);
    assert_eq!(
        informed.projection.stack[0].paid_cost_refs[0].arena_id,
        known.0
    );
    let mut visible_ids = BTreeSet::new();
    collect_arena_ids(&serde_json::to_value(&informed).unwrap(), &mut visible_ids);
    assert!(visible_ids.contains(&(known.0 as u64)));
    assert!(
        !visible_ids.contains(&(unknown.0 as u64)),
        "an unknown opponent-hand cost identity leaked"
    );

    let owner = observe_for_test(&state, PlayerId::P1, 4);
    assert_eq!(
        owner.projection.stack[0]
            .paid_cost_refs
            .iter()
            .map(|card| card.arena_id)
            .collect::<Vec<_>>(),
        vec![known.0, unknown.0],
        "the owner must retain both identities through ordinary own-hand visibility"
    );

    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(known, Zone::Library),
    );
    assert!(state
        .known_hand_cards(PlayerId::P0, PlayerId::P1)
        .is_empty());
    let historical = observe_for_test(&state, PlayerId::P0, 5);
    assert_eq!(historical.projection.stack[0].paid_cost_refs.len(), 1);
    assert_eq!(
        historical.projection.stack[0].paid_cost_refs[0].zone,
        Zone::Hand
    );
    assert_eq!(
        historical.projection.stack[0].paid_cost_refs[0].zone_change_count,
        known_ref.zone_change_count
    );
    assert_eq!(state.objects.get(known).zone, Zone::Library);
    assert_ne!(
        state.objects.get(known).zone_change_count,
        known_ref.zone_change_count
    );
}

#[test]
fn rl_contract_public_paid_cost_provenance_survives_later_hidden_zone_changes() {
    let mut state = empty_state();
    let source = make_object(
        &mut state,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let paid = make_object(&mut state, PlayerId::P1, "Mountain", Zone::Graveyard);
    let payment_ref = mtg_kernel::state::PaidCostRefV4::capture(&state, paid);
    state.stack.push(StackItem {
        kind: StackItemKind::TriggeredAbility,
        source,
        controller: PlayerId::P0,
        targets: Vec::new(),
        is_copy: false,
        inline_effect: None,
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false,
        v4: mtg_kernel::state::StackStateV4 {
            paid_cost_refs: vec![payment_ref],
            ..mtg_kernel::state::StackStateV4::default()
        },
    });

    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(paid, Zone::Library),
    );
    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(paid, Zone::Hand),
    );
    let observed = observe_for_test(&state, PlayerId::P0, 8);
    assert_eq!(observed.projection.stack[0].paid_cost_refs.len(), 1);
    let historical = &observed.projection.stack[0].paid_cost_refs[0];
    assert_eq!(historical.arena_id, paid.0);
    assert_eq!(historical.zone, Zone::Graveyard);
    assert_eq!(historical.zone_change_count, payment_ref.zone_change_count);
    assert_eq!(state.objects.get(paid).zone, Zone::Hand);
    assert_ne!(
        state.objects.get(paid).zone_change_count,
        historical.zone_change_count
    );
}

#[test]
fn rl_contract_known_library_draw_transfers_only_informed_identity_to_known_hand() {
    let p0 = ids(&["Mountain"]);
    let p1 = ids(&["Fiery Temper"]);
    let mut state = GameState::new_from_libraries(&p0, &p1, card_name, 44);
    state.reveal_library_top(PlayerId::P0, PlayerId::P1, 1);
    state.draw_card(PlayerId::P1).unwrap();

    let p0 = observe_for_test(&state, PlayerId::P0, 1);
    assert!(p0.known_library_cards[1].is_empty());
    assert_eq!(p0.known_hand_cards[1][0].card_name, "Fiery Temper");
    let p1 = observe_for_test(&state, PlayerId::P1, 1);
    assert!(p1.known_hand_cards[1].is_empty());
}

#[test]
fn rl_contract_public_zone_returns_to_hand_remain_known_but_library_moves_do_not() {
    let mut state = empty_state();
    state.step = Step::Main1;
    let public = make_object(
        &mut state,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let generation = state.objects.get(public).zone_change_count;
    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(public, Zone::Hand),
    );
    assert_eq!(state.objects.get(public).zone_change_count, generation + 1);
    let opponent = observe_for_test(&state, PlayerId::P1, 1);
    assert_eq!(
        opponent.known_hand_cards[0][0].card_name,
        "Voldaren Epicure"
    );
    let owner = observe_for_test(&state, PlayerId::P0, 1);
    assert!(owner.known_hand_cards[0].is_empty());
    assert_eq!(owner.own_hand[0].card_name, "Voldaren Epicure");

    let hidden = make_object(&mut state, PlayerId::P0, "Lightning Bolt", Zone::Library);
    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(hidden, Zone::Hand),
    );
    let opponent = observe_for_test(&state, PlayerId::P1, 2);
    assert_eq!(opponent.known_hand_cards[0].len(), 1);
    assert_eq!(
        opponent.known_hand_cards[0][0].card_name,
        "Voldaren Epicure"
    );
}

#[test]
fn rl_contract_registry_metadata_is_materialized_in_public_v4_objects() {
    let mut state = empty_state();
    let emissary = make_object(
        &mut state,
        PlayerId::P0,
        "Burning-Tree Emissary",
        Zone::Battlefield,
    );
    let blood = make_object(&mut state, PlayerId::P0, "Blood Token", Zone::Battlefield);
    state.objects.get_mut(emissary).counters = Counters {
        plus1_plus1: 300,
        minus1_minus1: 2,
        minus0_minus1: 1,
        stun: 4,
        lore: 5,
    };

    let def = &CARD_DEFS[state.objects.get(emissary).card_def as usize];
    let mut expected_subtypes: Vec<_> = def
        .subtypes
        .iter()
        .map(|subtype| subtype.stable_id())
        .collect();
    expected_subtypes.sort_unstable();
    expected_subtypes.dedup();
    assert_eq!(
        state.objects.get(emissary).v4.effective_subtype_ids,
        expected_subtypes
    );
    assert_eq!(
        state.objects.get(emissary).v4.effective_color_mask,
        mtg_kernel::card_def::mana_color_mask(mtg_kernel::mana::ManaColor::R)
            | mtg_kernel::card_def::mana_color_mask(mtg_kernel::mana::ManaColor::G)
    );
    assert!(!state.objects.get(emissary).v4.is_token);
    assert!(state.objects.get(blood).v4.is_token);

    let observation = observe_for_test(&state, PlayerId::P0, 0);
    let public = first_battlefield_card(&observation, PlayerSeatV1::P0, "Burning-Tree Emissary");
    assert_eq!(public.counters.plus1_plus1, 300);
    assert_eq!(public.counters.minus1_minus1, 2);
    assert_eq!(public.counters.minus0_minus1, 1);
    assert_eq!(public.counters.stun, 4);
    assert_eq!(public.counters.lore, 5);
    assert_eq!(
        public.characteristics.effective_subtype_ids,
        expected_subtypes
    );
    assert!(first_battlefield_card(&observation, PlayerSeatV1::P0, "Blood Token").is_token);
}

#[test]
fn rl_contract_object_relations_are_semantic_and_stale_links_fail_closed() {
    let mut state = empty_state();
    let attachment = make_object(&mut state, PlayerId::P0, "Blood Token", Zone::Battlefield);
    let host = make_object(
        &mut state,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    state.objects.get_mut(attachment).v4.attached_to = Some(mtg_kernel::state::ObjectLinkV4 {
        object: host,
        zone_change_count: state.objects.get(host).zone_change_count,
    });
    let observed = observe_for_test(&state, PlayerId::P0, 0);
    let relations = serde_json::to_value(&observed.projection.object_relations).unwrap();
    assert_eq!(relations[0]["relation_kind"], "attached_to");
    assert_eq!(relations[0]["object"]["arena_id"], attachment.0);
    assert_eq!(relations[0]["attached_to"]["arena_id"], host.0);

    state.objects.get_mut(host).zone_change_count += 1;
    let error = observe_v2(&state, &HarnessSurfaceV2::new(), PlayerId::P0, 0).unwrap_err();
    assert!(error.to_string().contains("stale object incarnation"));
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
fn rl_contract_continuous_effects_filter_hidden_affected_objects_by_actor() {
    let mut opponent_hand = empty_state();
    let hidden = make_object(
        &mut opponent_hand,
        PlayerId::P1,
        "Lightning Bolt",
        Zone::Hand,
    );
    opponent_hand
        .engine
        .until_end_of_turn
        .push(UntilEndOfTurnEffect::ResolvedSetEffect {
            object_ids: vec![hidden],
            layer: Layers::POWER_TOUGHNESS,
            timestamp: 42,
            duration: EffectDuration::EndOfTurn,
            power: 1,
            toughness: 0,
            grant_haste: false,
        });

    let p0_view = observe_for_test(&opponent_hand, PlayerId::P0, 0);
    assert!(
        p0_view.projection.continuous_effects.is_empty(),
        "opponent hand affected object must be filtered"
    );
    let p0_json = serde_json::to_string(&p0_view).unwrap();
    assert!(!p0_json.contains("Lightning Bolt"));
    assert!(!p0_json.contains(&format!("\"arena_id\":{}", hidden.0)));

    let p1_view = observe_for_test(&opponent_hand, PlayerId::P1, 0);
    assert_eq!(p1_view.projection.continuous_effects.len(), 1);
    assert_eq!(
        p1_view.projection.continuous_effects[0].affected_objects[0].arena_id,
        hidden.0
    );

    let mut library = empty_state();
    let library_hidden = make_object(&mut library, PlayerId::P1, "Lightning Bolt", Zone::Library);
    library
        .engine
        .until_end_of_turn
        .push(UntilEndOfTurnEffect::ResolvedSetEffect {
            object_ids: vec![library_hidden],
            layer: Layers::POWER_TOUGHNESS,
            timestamp: 43,
            duration: EffectDuration::EndOfTurn,
            power: 1,
            toughness: 0,
            grant_haste: false,
        });
    let p1_library_view = observe_for_test(&library, PlayerId::P1, 0);
    assert!(
        p1_library_view.projection.continuous_effects.is_empty(),
        "library affected object identity must be filtered even for its owner"
    );
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
fn rl_contract_stack_items_expose_explicit_public_kind() {
    let mut state = empty_state();
    let spell = make_object(&mut state, PlayerId::P0, "Lightning Bolt", Zone::Stack);
    let activated = make_object(&mut state, PlayerId::P0, "Blood Token", Zone::Battlefield);
    let triggered = make_object(
        &mut state,
        PlayerId::P1,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let madness = make_object(&mut state, PlayerId::P0, "Fiery Temper", Zone::Exile);
    for (kind, source, controller, madness_offer) in [
        (StackItemKind::Spell, spell, PlayerId::P0, false),
        (
            StackItemKind::ActivatedAbility,
            activated,
            PlayerId::P0,
            false,
        ),
        (
            StackItemKind::TriggeredAbility,
            triggered,
            PlayerId::P1,
            false,
        ),
        (StackItemKind::MadnessOffer, madness, PlayerId::P0, true),
    ] {
        state.stack.push(StackItem {
            kind,
            source,
            controller,
            targets: Vec::new(),
            is_copy: false,
            inline_effect: None,
            discarded: Vec::new(),
            is_flashback: false,
            mode_chosen: 0,
            madness_offer,
            kicked: false,
            v4: if kind == StackItemKind::Spell {
                mtg_kernel::state::StackStateV4::spell(mtg_kernel::state::CastMethodV4::Normal)
            } else {
                mtg_kernel::state::StackStateV4::default()
            },
        });
    }

    let obs = observe_for_test(&state, PlayerId::P0, 0);
    let stack = serde_json::to_value(&obs.projection.stack).unwrap();
    assert_eq!(stack[0]["stack_item_kind"], "spell");
    assert_eq!(stack[1]["stack_item_kind"], "activated_ability");
    assert_eq!(stack[2]["stack_item_kind"], "triggered_ability");
    assert_eq!(stack[3]["stack_item_kind"], "madness_offer");
    assert!(!contains_key(&stack, "is_trigger_or_ability"));
}

#[test]
fn rl_contract_spell_copy_state_and_binary_actions_are_explicit() {
    let mut state = empty_state();
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state.step = Step::Main1;
    make_object(&mut state, PlayerId::P1, "Great Furnace", Zone::Battlefield);
    make_object(&mut state, PlayerId::P1, "Great Furnace", Zone::Battlefield);
    let parent = make_object(&mut state, PlayerId::P0, "Chain Lightning", Zone::Stack);
    state.stack.push(StackItem {
        kind: StackItemKind::Spell,
        source: parent,
        controller: PlayerId::P0,
        targets: vec![Target::Player(PlayerId::P1)],
        is_copy: false,
        inline_effect: None,
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false,
        v4: mtg_kernel::state::StackStateV4::spell(mtg_kernel::state::CastMethodV4::Normal),
    });
    state.engine.pending_spell_copy = Some(PendingSpellCopy {
        resolving_source: parent,
        player: PlayerId::P1,
        inherited_target: Target::Player(PlayerId::P1),
        stage: SpellCopyStage::Payment,
        copy_source: None,
    });

    let payment_obs = observe_for_test(&state, PlayerId::P1, 4);
    assert_eq!(
        payment_obs.projection.engine_context.current_stage,
        EngineDecisionStageV2::PendingSpellCopy
    );
    let pending = payment_obs
        .projection
        .engine_context
        .pending_spell_copy
        .as_ref()
        .expect("payment continuation is public");
    assert_eq!(pending.stage, SpellCopyStageV2::Payment);
    assert_eq!(pending.parent.as_ref().unwrap().arena_id, parent.0);
    assert!(pending.copy.is_none());
    assert!(!payment_obs.projection.stack[0].is_copy);

    let payment_decision = engine::advance_until_decision(&mut state);
    let payment_actions =
        legal_action_candidates_v1(&SurfaceDecision::Decision(payment_decision), &state).unwrap();
    assert_eq!(payment_actions.len(), 2);
    assert!(matches!(
        payment_actions[0].record.semantic,
        ActionSemanticV1::ChooseSpellCopyPayment {
            actor: PlayerSeatV1::P1,
            pay: true,
            ..
        }
    ));
    assert!(matches!(
        payment_actions[1].record.semantic,
        ActionSemanticV1::ChooseSpellCopyPayment {
            actor: PlayerSeatV1::P1,
            pay: false,
            ..
        }
    ));

    engine::step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
    let copy = state
        .engine
        .pending_spell_copy
        .as_ref()
        .unwrap()
        .copy_source
        .unwrap();
    let retarget_obs = observe_for_test(&state, PlayerId::P1, 5);
    let pending = retarget_obs
        .projection
        .engine_context
        .pending_spell_copy
        .as_ref()
        .unwrap();
    assert_eq!(pending.stage, SpellCopyStageV2::Retarget);
    assert_eq!(pending.copy.as_ref().unwrap().arena_id, copy.0);
    assert_eq!(retarget_obs.projection.stack.len(), 2);
    assert!(!retarget_obs.projection.stack[0].is_copy);
    assert!(retarget_obs.projection.stack[1].is_copy);

    let retarget_decision = engine::advance_until_decision(&mut state);
    let retarget_actions =
        legal_action_candidates_v1(&SurfaceDecision::Decision(retarget_decision), &state).unwrap();
    assert_eq!(retarget_actions.len(), 2);
    assert!(matches!(
        retarget_actions[0].record.semantic,
        ActionSemanticV1::ChooseSpellCopyRetarget {
            actor: PlayerSeatV1::P1,
            change_target: true,
            ..
        }
    ));
    assert!(matches!(
        retarget_actions[1].record.semantic,
        ActionSemanticV1::ChooseSpellCopyRetarget {
            actor: PlayerSeatV1::P1,
            change_target: false,
            ..
        }
    ));

    engine::step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
    let target_obs = observe_for_test(&state, PlayerId::P1, 6);
    assert_eq!(
        target_obs
            .projection
            .engine_context
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .stage,
        SpellCopyStageV2::Target
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
    assert_ne!(
        blocker_obs_a.projection.surface_context.private_blockers,
        blocker_obs_b.projection.surface_context.private_blockers,
        "blocker owner must receive enough private reshape state to distinguish progress"
    );
    let opponent_blocker_obs = observe_v2(&state_a, &surface_a, PlayerId::P0, 10).unwrap();
    assert!(
        opponent_blocker_obs
            .projection
            .surface_context
            .private_blockers
            .is_none(),
        "attacking player must not see defender's partial blocker reshape"
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
        .private_discard
        .as_ref()
        .unwrap()
        .chosen
        .iter()
        .any(|card| card.arena_id == card_0.0));
    assert!(discard_obs_b
        .projection
        .surface_context
        .private_discard
        .as_ref()
        .unwrap()
        .chosen
        .iter()
        .any(|card| card.arena_id == card_1.0));
    assert!(discard_obs_a
        .projection
        .surface_context
        .private_discard
        .as_ref()
        .unwrap()
        .remaining_choices
        .iter()
        .any(|card| card.arena_id == card_2.0));
    let opponent_discard_obs =
        observe_v2(&discard_a, &discard_surface_a, PlayerId::P1, 11).unwrap();
    assert!(
        opponent_discard_obs
            .projection
            .surface_context
            .private_discard
            .is_none(),
        "opponent must not see partial discard choices"
    );
    let opponent_json = serde_json::to_string(&opponent_discard_obs).unwrap();
    for hidden in [card_0, card_1, card_2] {
        assert!(
            !opponent_json.contains(&format!("\"arena_id\":{}", hidden.0)),
            "opponent recovered hidden discard object id {}",
            hidden.0
        );
    }
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
fn rl_contract_semantic_flags_change_hash_but_raw_priority_counter_offset_does_not() {
    let base = empty_state();
    let base_obs = observe_for_test(&base, PlayerId::P0, 20);

    let mut pass_changed = base.clone();
    pass_changed.engine.priority_passes = [true, false];
    let pass_obs = observe_for_test(&pass_changed, PlayerId::P0, 20);
    assert_ne!(
        base_obs.visible_projection_hash,
        pass_obs.visible_projection_hash
    );

    let mut raw_counter_offset = base.clone();
    raw_counter_offset.engine.priority_round += 99;
    raw_counter_offset.engine.next_effect_timestamp += 99;
    let raw_counter_obs = observe_for_test(&raw_counter_offset, PlayerId::P0, 20);
    assert_eq!(
        serde_json::to_vec(&base_obs).unwrap(),
        serde_json::to_vec(&raw_counter_obs).unwrap(),
        "irrelevant raw monotonic counters must not affect serialized ObservationV2"
    );
}

#[test]
fn rl_contract_mana_activity_context_is_priority_boundary_scoped() {
    let mut state = empty_state();
    let mountain = make_object(&mut state, PlayerId::P0, "Mountain", Zone::Battlefield);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;

    let initial = observe_for_test(&state, PlayerId::P0, 1);
    assert!(
        !initial
            .projection
            .engine_context
            .mana_activity_since_priority_boundary
    );
    assert_eq!(
        initial
            .projection
            .engine_context
            .last_mana_ability_activator_since_priority_boundary,
        None
    );

    engine::step(&mut state, Action::ActivateManaAbility(mountain)).unwrap();
    let after_mana = observe_for_test(&state, PlayerId::P0, 2);
    assert!(
        after_mana
            .projection
            .engine_context
            .mana_activity_since_priority_boundary
    );
    assert_eq!(
        after_mana
            .projection
            .engine_context
            .last_mana_ability_activator_since_priority_boundary,
        Some(PlayerSeatV1::P0)
    );

    engine::step(&mut state, Action::Pass).unwrap();
    engine::step(&mut state, Action::Pass).unwrap();
    let _ = engine::advance_until_decision(&mut state);
    assert_eq!(state.engine.mana_ability_activations, 1);
    assert_eq!(state.engine.mana_ability_count_at_round_open, 1);
    assert_eq!(state.engine.last_mana_ability_activator, Some(PlayerId::P0));

    let after_boundary = observe_for_test(&state, PlayerId::P0, 3);
    assert!(
        !after_boundary
            .projection
            .engine_context
            .mana_activity_since_priority_boundary
    );
    assert_eq!(
        after_boundary
            .projection
            .engine_context
            .last_mana_ability_activator_since_priority_boundary,
        None
    );
}

#[test]
fn rl_contract_mana_boundary_projection_ignores_absolute_counter_offsets() {
    let mut base = empty_state();
    base.engine.mana_ability_activations = 17;
    base.engine.mana_ability_count_at_round_open = 17;
    base.engine.last_mana_ability_activator = Some(PlayerId::P1);

    let mut offset = base.clone();
    offset.engine.mana_ability_activations += 1_000;
    offset.engine.mana_ability_count_at_round_open += 1_000;

    let base_obs = observe_for_test(&base, PlayerId::P0, 20);
    let offset_obs = observe_for_test(&offset, PlayerId::P0, 20);
    assert_eq!(
        serde_json::to_vec(&base_obs).unwrap(),
        serde_json::to_vec(&offset_obs).unwrap(),
        "absolute mana count and boundary baseline offsets must not affect serialized ObservationV2"
    );
    assert!(
        !base_obs
            .projection
            .engine_context
            .mana_activity_since_priority_boundary
    );
    assert_eq!(
        base_obs
            .projection
            .engine_context
            .last_mana_ability_activator_since_priority_boundary,
        None
    );
}

#[test]
fn rl_contract_mana_boundary_delta_changes_visible_projection_hash() {
    let mut base = empty_state();
    base.engine.mana_ability_activations = 17;
    base.engine.mana_ability_count_at_round_open = 17;
    base.engine.last_mana_ability_activator = Some(PlayerId::P1);

    let mut delta = base.clone();
    delta.engine.mana_ability_activations += 1;

    let base_obs = observe_for_test(&base, PlayerId::P0, 20);
    let delta_obs = observe_for_test(&delta, PlayerId::P0, 20);
    assert_ne!(
        base_obs.visible_projection_hash,
        delta_obs.visible_projection_hash
    );
    assert!(
        delta_obs
            .projection
            .engine_context
            .mana_activity_since_priority_boundary
    );
    assert_eq!(
        delta_obs
            .projection
            .engine_context
            .last_mana_ability_activator_since_priority_boundary,
        Some(PlayerSeatV1::P1)
    );
}

#[test]
fn rl_contract_observation_v2_uses_exact_boundary_semantic_field_names() {
    let obs = observe_for_test(&empty_state(), PlayerId::P0, 20);
    let value = serde_json::to_value(&obs).unwrap();
    let engine_context = value["projection"]["engine_context"]
        .as_object()
        .expect("engine_context must be an object");
    assert!(engine_context.contains_key("mana_activity_since_priority_boundary"));
    assert!(engine_context.contains_key("last_mana_ability_activator_since_priority_boundary"));
    for forbidden in [
        "last_mana_ability_activator",
        "mana_ability_activations",
        "mana_ability_count_at_round_open",
        "priority_round",
        "stack_len_at_round_open",
    ] {
        assert!(
            !contains_key(&value, forbidden),
            "ObservationV2 must not serialize raw or misleading engine field {forbidden}"
        );
    }

    let surface_context = value["projection"]["surface_context"]
        .as_object()
        .expect("surface_context must be an object");
    assert!(surface_context.contains_key("stack_grew_since_round_open"));
    assert!(!surface_context.contains_key("stack_activity_since_round_open"));
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
fn rl_contract_reserved_typed_choices_and_actions_have_exact_v4_wire_shapes() {
    use mtg_kernel::mana::ManaColor;
    use mtg_kernel::rl::{
        BooleanChoicePurposeV4, PendingEffectChoiceSemanticV4, TargetRefV1,
        TargetSelectionPurposeV4,
    };

    let mut state = empty_state();
    make_object(
        &mut state,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let source = first_battlefield_card(
        &observe_for_test(&state, PlayerId::P0, 0),
        PlayerSeatV1::P0,
        "Voldaren Epicure",
    )
    .stable
    .clone();
    let target = TargetRefV1::Player {
        player: PlayerSeatV1::P1,
    };

    let targets = PendingEffectChoiceSemanticV4::Targets {
        player: PlayerSeatV1::P0,
        structural_path: vec![2, 1],
        selected_targets: vec![target.clone()],
        legal_targets: vec![target.clone()],
        min_targets: 0,
        max_targets: 2,
        can_finish: true,
        ordered: true,
        purpose: TargetSelectionPurposeV4::CardSelection,
    };
    let targets_json = serde_json::to_value(targets).unwrap();
    assert_eq!(targets_json["choice_kind"], "targets");
    assert_eq!(targets_json["purpose"], "card_selection");
    assert_eq!(targets_json["ordered"], true);

    let boolean = PendingEffectChoiceSemanticV4::Boolean {
        player: PlayerSeatV1::P1,
        structural_path: vec![0],
        default: Some(false),
        purpose: BooleanChoicePurposeV4::Shuffle,
    };
    let boolean_json = serde_json::to_value(boolean).unwrap();
    assert_eq!(boolean_json["choice_kind"], "boolean");
    assert_eq!(boolean_json["purpose"], "shuffle");

    let semantics = vec![
        ActionSemanticV1::ChooseEffectTarget {
            actor: PlayerSeatV1::P0,
            source: source.clone(),
            target: target.clone(),
            selected_count: 1,
            min_targets: 0,
            max_targets: 2,
        },
        ActionSemanticV1::FinishEffectSelection {
            actor: PlayerSeatV1::P0,
            source: source.clone(),
            selected_count: 1,
        },
        ActionSemanticV1::ChooseEffectColor {
            actor: PlayerSeatV1::P0,
            source: source.clone(),
            color: ManaColor::R,
        },
        ActionSemanticV1::ChooseEffectNumber {
            actor: PlayerSeatV1::P0,
            source: source.clone(),
            number: 3,
            minimum: 0,
            maximum: 5,
        },
        ActionSemanticV1::ChooseEffectBoolean {
            actor: PlayerSeatV1::P0,
            source: source.clone(),
            value: true,
        },
        ActionSemanticV1::FinishTargetSelection {
            actor: PlayerSeatV1::P0,
            source,
            selected_count: 2,
        },
    ];
    let actions: Vec<_> = semantics
        .into_iter()
        .enumerate()
        .map(|(index, semantic)| make_legal_action_v1(index as u32, semantic, None).unwrap())
        .collect();
    let stable_ids: Vec<_> = actions
        .iter()
        .map(|action| action.stable_id.as_str())
        .collect();
    assert_eq!(
        stable_ids,
        vec![
            "legal-action-v4:16d67be112c3e80b",
            "legal-action-v4:23c7819511bc912c",
            "legal-action-v4:248a1391b07dccb0",
            "legal-action-v4:9955e40522887330",
            "legal-action-v4:41dbdb0c5ce04513",
            "legal-action-v4:76516736379e762f",
        ],
        "reserved semantic wire changes must deliberately update these exact ids"
    );
    assert_eq!(
        actions
            .iter()
            .map(
                |action| serde_json::to_value(&action.semantic).unwrap()["action_kind"]
                    .as_str()
                    .unwrap()
                    .to_string()
            )
            .collect::<Vec<_>>(),
        vec![
            "choose_effect_target",
            "finish_effect_selection",
            "choose_effect_color",
            "choose_effect_number",
            "choose_effect_boolean",
            "finish_target_selection",
        ]
    );
}

#[test]
fn rl_contract_mana_choice_generic_costs_and_effect_durations_are_reserved() {
    use mtg_kernel::engine::{CostKind, Decision};
    use mtg_kernel::mana::ManaColor;

    let mut state = empty_state();
    state.step = Step::Main1;
    let island = make_object(&mut state, PlayerId::P0, "Island", Zone::Battlefield);
    let decision = SurfaceDecision::Decision(Decision::CastSpellOrPass {
        player: PlayerId::P0,
        castable_spells: Vec::new(),
        mana_abilities: vec![island],
        land_drops: Vec::new(),
        activatable_abilities: Vec::new(),
        plot_actions: Vec::new(),
    });
    let actions = legal_action_candidates_v1(&decision, &state).unwrap();
    assert!(matches!(
        actions[0].record.semantic,
        ActionSemanticV1::ActivateManaAbility {
            mana_choice: Some(ManaColor::U),
            ..
        }
    ));
    assert_eq!(
        actions[0].record.stable_id, "legal-action-v4:477dcbe7e756ee24",
        "adding mana-choice semantics must remain an intentional stable-id change"
    );
    let SurfaceAction::Action(activation) = actions[0].surface_action.clone() else {
        panic!("mana semantic must map to an engine action");
    };
    engine::step(&mut state, activation).unwrap();
    let observed = observe_for_test(&state, PlayerId::P0, 1);
    let island = first_battlefield_card(&observed, PlayerSeatV1::P0, "Island");
    assert_eq!(island.ability_uses_this_turn.len(), 1);
    assert_eq!(
        serde_json::to_value(island.ability_uses_this_turn[0]).unwrap()["ability_kind"],
        "mana"
    );
    assert_eq!(island.ability_uses_this_turn[0].ability_index, 0);
    assert_eq!(island.ability_uses_this_turn[0].uses, 1);

    let costs = [
        CostKind::SacrificeLands,
        CostKind::SacrificePermanents,
        CostKind::SacrificeCreatures,
        CostKind::SacrificeArtifacts,
        CostKind::DiscardCards,
        CostKind::ExileFromGraveyard,
        CostKind::TapPermanents,
        CostKind::ReturnPermanentsToHand,
        CostKind::PayLife,
        CostKind::RemoveCounters,
        CostKind::PutCounters,
    ];
    assert_eq!(
        costs
            .iter()
            .map(|cost| serde_json::to_value(cost)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string())
            .collect::<Vec<_>>(),
        vec![
            "SacrificeLands",
            "SacrificePermanents",
            "SacrificeCreatures",
            "SacrificeArtifacts",
            "DiscardCards",
            "ExileFromGraveyard",
            "TapPermanents",
            "ReturnPermanentsToHand",
            "PayLife",
            "RemoveCounters",
            "PutCounters",
        ]
    );

    let durations = [
        mtg_kernel::rl::EffectDurationV2::EndOfTurn,
        mtg_kernel::rl::EffectDurationV2::UntilControllersNextTurn,
        mtg_kernel::rl::EffectDurationV2::WhileAttached,
        mtg_kernel::rl::EffectDurationV2::WhileSourcePresent,
    ];
    assert_eq!(
        durations
            .iter()
            .map(|duration| serde_json::to_value(duration)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string())
            .collect::<Vec<_>>(),
        vec![
            "end_of_turn",
            "until_controllers_next_turn",
            "while_attached",
            "while_source_present",
        ]
    );
}

#[test]
fn rl_contract_pending_effect_reuses_the_public_resolving_stack_source() {
    let mut state = empty_state();
    let source = make_object(&mut state, PlayerId::P0, "Voldaren Epicure", Zone::Hand);
    let resolving_item = StackItem {
        kind: StackItemKind::TriggeredAbility,
        source,
        controller: PlayerId::P0,
        targets: Vec::new(),
        is_copy: false,
        inline_effect: None,
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false,
        v4: mtg_kernel::state::StackStateV4::default(),
    };
    let effect = mtg_kernel::effect::EffectOp::Choice {
        controller: mtg_kernel::effect::PlayerRef::Controller,
        options: vec![
            mtg_kernel::effect::EffectOp::GainLife {
                player: mtg_kernel::effect::PlayerRef::Controller,
                amount: 1,
            },
            mtg_kernel::effect::EffectOp::GainLife {
                player: mtg_kernel::effect::PlayerRef::Controller,
                amount: 2,
            },
        ],
    };
    assert_eq!(
        mtg_kernel::effect::begin_resumable_resolution(
            &effect,
            &mtg_kernel::effect::ExecCtx::no_targets(source, PlayerId::P0),
            resolving_item,
            &mut state,
        )
        .unwrap(),
        mtg_kernel::effect::ResumableProgress::Suspended
    );

    let opponent_view = observe_for_test(&state, PlayerId::P1, 7);
    assert!(opponent_view.known_hand_cards[0].is_empty());
    let projected_source = opponent_view
        .projection
        .engine_context
        .pending_effect
        .as_ref()
        .unwrap()
        .source
        .as_ref()
        .expect("a resolving stack source stays public after leaving a public zone");
    assert_eq!(projected_source.arena_id, source.0);
    assert_eq!(projected_source.zone, Zone::Hand);
}

#[test]
fn rl_contract_live_generic_option_choice_projects_and_keeps_exact_action_ids() {
    let mut state = empty_state();
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    let source = make_object(
        &mut state,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    state.stack.push(StackItem {
        kind: StackItemKind::TriggeredAbility,
        source,
        controller: PlayerId::P0,
        targets: Vec::new(),
        is_copy: false,
        inline_effect: Some(mtg_kernel::effect::EffectOp::Choice {
            controller: mtg_kernel::effect::PlayerRef::Controller,
            options: vec![
                mtg_kernel::effect::EffectOp::GainLife {
                    player: mtg_kernel::effect::PlayerRef::Controller,
                    amount: 2,
                },
                mtg_kernel::effect::EffectOp::LoseLife {
                    player: mtg_kernel::effect::PlayerRef::Controller,
                    amount: 3,
                },
            ],
        }),
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false,
        v4: mtg_kernel::state::StackStateV4::default(),
    });
    state.engine.priority_passes = [true, true];

    let decision = engine::advance_until_decision(&mut state);
    let observation = observe_for_test(&state, PlayerId::P0, 22);
    let choice = serde_json::to_value(
        observation
            .projection
            .engine_context
            .pending_effect
            .as_ref()
            .unwrap()
            .choice
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(choice["choice_kind"], "options");
    assert_eq!(choice["option_count"], 2);

    let candidates =
        legal_action_candidates_v1(&SurfaceDecision::Decision(decision), &state).unwrap();
    assert_eq!(candidates.len(), 2);
    let ids: Vec<_> = candidates
        .iter()
        .map(|candidate| candidate.record.stable_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            "legal-action-v4:6851ecf7178d6c8a",
            "legal-action-v4:c00d3dd4f6e51099",
        ]
    );
    assert!(matches!(
        candidates[0].record.semantic,
        ActionSemanticV1::ChooseEffectOption {
            option_index: 0,
            option_count: 2,
            ..
        }
    ));

    let SurfaceAction::Action(action) = candidates[0].surface_action.clone() else {
        panic!("generic option must map to an engine action");
    };
    engine::step(&mut state, action).unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(state.players[0].life, 22);
    assert!(state.engine.pending_effect.is_none());
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
fn rl_contract_episode_records_use_independent_schema_versions_and_pin_hash_algorithm() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let run = record_burn_mirror_episode(0, env_seed, policy_seed, 64).unwrap();

    assert_eq!(AUDIT_EPISODE_SCHEMA_VERSION, 5);
    assert_eq!(MANIFEST_SCHEMA_VERSION, 5);
    assert_eq!(POLICY_EPISODE_SCHEMA_VERSION, 4);
    let audit_header = serde_json::to_value(&run.audit_records[0]).unwrap();
    assert_eq!(audit_header["schema_version"], AUDIT_EPISODE_SCHEMA_VERSION);
    assert_eq!(
        audit_header["diagnostic_state_hash_algorithm"],
        DIAGNOSTIC_STATE_HASH_ALGORITHM
    );
    assert!(audit_header.get("game_id").is_some());
    assert!(audit_header.get("env_seed").is_some());
    assert!(audit_header.get("policy_seed").is_some());

    let policy_header = serde_json::to_value(&run.policy_records[0]).unwrap();
    assert_eq!(
        policy_header["schema_version"],
        POLICY_EPISODE_SCHEMA_VERSION
    );
    assert!(policy_header.get("episode_key").is_some());
    assert!(policy_header.get("game_id").is_none());
    assert!(policy_header.get("env_seed").is_none());
    assert!(policy_header.get("policy_seed").is_none());
    assert!(policy_header
        .get("diagnostic_state_hash_algorithm")
        .is_none());

    let policy_decision = run
        .policy_records
        .iter()
        .find(|record| {
            matches!(
                record,
                mtg_kernel::rl::PolicyEpisodeRecordV2::Decision { .. }
            )
        })
        .expect("policy stream should contain a decision");
    let policy_decision = serde_json::to_value(policy_decision).unwrap();
    assert_eq!(
        policy_decision["schema_version"],
        POLICY_EPISODE_SCHEMA_VERSION
    );
    assert_eq!(
        policy_decision["observation"]["schema_version"],
        OBSERVATION_SCHEMA_VERSION
    );
    assert_eq!(
        policy_decision["legal_actions"][0]["schema_version"],
        LEGAL_ACTION_SCHEMA_VERSION
    );

    let policy_terminal = run
        .policy_records
        .iter()
        .find(|record| {
            matches!(
                record,
                mtg_kernel::rl::PolicyEpisodeRecordV2::Terminal { .. }
            )
        })
        .expect("policy stream should contain a terminal");
    let policy_terminal = serde_json::to_value(policy_terminal).unwrap();
    assert_eq!(
        policy_terminal["schema_version"],
        POLICY_EPISODE_SCHEMA_VERSION
    );
    assert!(policy_terminal.get("terminal_code").is_some());
    assert!(policy_terminal.get("terminal_reason").is_none());
}

#[test]
fn rl_contract_audit_reader_fails_closed_on_legacy_missing_or_unknown_hash_contracts() {
    let run =
        record_burn_mirror_episode(0, derive_env_seed(9999, 0), derive_policy_seed(9999, 0), 16)
            .unwrap();
    let valid_jsonl = records_to_jsonl(&run.audit_records);
    assert_eq!(
        parse_audit_episode_jsonl(&valid_jsonl).unwrap(),
        run.audit_records
    );

    let mut values = run
        .audit_records
        .iter()
        .map(|record| serde_json::to_value(record).unwrap())
        .collect::<Vec<_>>();
    values[0]["diagnostic_state_hash_algorithm"] = Value::String("unknown-v99".to_string());
    let unknown = records_to_jsonl(&values);
    assert!(parse_audit_episode_jsonl(&unknown)
        .unwrap_err()
        .to_string()
        .contains("unsupported diagnostic_state_hash_algorithm"));

    values[0]["diagnostic_state_hash_algorithm"] =
        Value::String(DIAGNOSTIC_STATE_HASH_ALGORITHM.to_string());
    values[0]["schema_version"] = Value::from(AUDIT_EPISODE_SCHEMA_VERSION - 1);
    let legacy = records_to_jsonl(&values);
    assert!(parse_audit_episode_jsonl(&legacy)
        .unwrap_err()
        .to_string()
        .contains("unsupported audit schema_version"));

    values[0]["schema_version"] = Value::from(AUDIT_EPISODE_SCHEMA_VERSION);
    values[0]
        .as_object_mut()
        .unwrap()
        .remove("diagnostic_state_hash_algorithm");
    let missing = records_to_jsonl(&values);
    assert!(parse_audit_episode_jsonl(&missing)
        .unwrap_err()
        .to_string()
        .contains("missing field `diagnostic_state_hash_algorithm`"));

    let mut mixed = run
        .audit_records
        .iter()
        .map(|record| serde_json::to_value(record).unwrap())
        .collect::<Vec<_>>();
    mixed[1]["schema_version"] = Value::from(AUDIT_EPISODE_SCHEMA_VERSION - 1);
    assert!(parse_audit_episode_jsonl(&records_to_jsonl(&mixed))
        .unwrap_err()
        .to_string()
        .contains("unsupported audit schema_version"));

    mixed[1]["schema_version"] = Value::from(AUDIT_EPISODE_SCHEMA_VERSION);
    assert!(parse_audit_episode_jsonl(&records_to_jsonl(&mixed[1..]))
        .unwrap_err()
        .to_string()
        .contains("out-of-order episode_id/step"));
}

#[test]
fn rl_contract_policy_reader_rejects_empty_legacy_mixed_and_headerless_streams() {
    let run =
        record_burn_mirror_episode(0, derive_env_seed(9999, 0), derive_policy_seed(9999, 0), 16)
            .unwrap();
    let valid_jsonl = records_to_jsonl(&run.policy_records);
    assert_eq!(
        parse_policy_episode_jsonl(&valid_jsonl).unwrap(),
        run.policy_records
    );
    assert!(parse_policy_episode_jsonl(&valid_jsonl[0..0])
        .unwrap_err()
        .to_string()
        .contains("policy stream is empty"));

    let mut values = run
        .policy_records
        .iter()
        .map(|record| serde_json::to_value(record).unwrap())
        .collect::<Vec<_>>();
    values[0]["schema_version"] = Value::from(POLICY_EPISODE_SCHEMA_VERSION - 1);
    assert!(parse_policy_episode_jsonl(&records_to_jsonl(&values))
        .unwrap_err()
        .to_string()
        .contains("unsupported policy schema_version"));

    values[0]["schema_version"] = Value::from(POLICY_EPISODE_SCHEMA_VERSION);
    values[1]["schema_version"] = Value::from(POLICY_EPISODE_SCHEMA_VERSION - 1);
    assert!(parse_policy_episode_jsonl(&records_to_jsonl(&values))
        .unwrap_err()
        .to_string()
        .contains("unsupported policy schema_version"));

    values[1]["schema_version"] = Value::from(POLICY_EPISODE_SCHEMA_VERSION);
    assert!(parse_policy_episode_jsonl(&records_to_jsonl(&values[1..]))
        .unwrap_err()
        .to_string()
        .contains("out-of-order episode_id/step"));
}

#[test]
fn rl_contract_policy_reader_rejects_unknown_fields_and_duplicate_keys_recursively() {
    let run =
        record_burn_mirror_episode(0, derive_env_seed(9999, 0), derive_policy_seed(9999, 0), 16)
            .unwrap();
    let values = run
        .policy_records
        .iter()
        .map(|record| serde_json::to_value(record).unwrap())
        .collect::<Vec<_>>();
    let decision_index = values
        .iter()
        .position(|value| value["record_type"] == "decision")
        .expect("test episode must contain a policy decision");
    assert!(!values[decision_index]["legal_actions"]
        .as_array()
        .unwrap()
        .is_empty());

    for location in ["record", "observation", "action"] {
        for key in ["diagnostic_state_hash", "env_seed", "unknown_field"] {
            let mut corrupted = values.clone();
            let target = match location {
                "record" => &mut corrupted[decision_index],
                "observation" => &mut corrupted[decision_index]["observation"],
                "action" => &mut corrupted[decision_index]["legal_actions"][0],
                _ => unreachable!(),
            };
            target[key] = Value::from(0x5afe_u64);
            let error = parse_policy_episode_jsonl(&records_to_jsonl(&corrupted))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("does not exactly match the policy schema"),
                "{location} injection of {key} was not rejected strictly: {error}"
            );
        }
    }

    let valid_lines = values
        .iter()
        .map(|value| serde_json::to_string(value).unwrap())
        .collect::<Vec<_>>();
    let decision_line = &valid_lines[decision_index];
    let duplicate_lines = [
        decision_line.replacen('{', "{\"record_type\":\"decision\",", 1),
        decision_line.replacen(
            "\"observation\":{",
            &format!("\"observation\":{{\"schema_version\":{OBSERVATION_SCHEMA_VERSION},"),
            1,
        ),
        decision_line.replacen(
            "\"legal_actions\":[{",
            &format!("\"legal_actions\":[{{\"schema_version\":{LEGAL_ACTION_SCHEMA_VERSION},"),
            1,
        ),
    ];
    for (location, duplicate_line) in ["record", "observation", "action"]
        .into_iter()
        .zip(duplicate_lines)
    {
        assert_ne!(duplicate_line, *decision_line);
        let mut corrupted_lines = valid_lines.clone();
        corrupted_lines[decision_index] = duplicate_line;
        let error = parse_policy_episode_jsonl(&(corrupted_lines.join("\n") + "\n"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("duplicate JSON object key"),
            "{location} duplicate key was not rejected before typed bundle acceptance: {error}"
        );
    }
}

#[test]
fn rl_contract_bundle_validation_is_one_to_one_and_precedes_writes() {
    let base_seed = 9999;
    let max_decisions = 16;
    let run = record_burn_mirror_episode(
        0,
        derive_env_seed(base_seed, 0),
        derive_policy_seed(base_seed, 0),
        max_decisions,
    )
    .unwrap();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "mtg-kernel-invalid-bundle-{}-{unique}",
        std::process::id()
    ));
    assert!(!out_dir.exists());
    let manifest = build_run_manifest(
        1,
        base_seed,
        max_decisions,
        Vec::new(),
        &out_dir,
        std::slice::from_ref(&run.terminal),
        GitMetadataV1 {
            commit: "test".to_string(),
            dirty: GitDirtyFlagV1::Clean,
        },
    )
    .unwrap();
    validate_rollout_artifact_bundle(&run.audit_records, &run.policy_records, &manifest).unwrap();

    let mut selected_action_corruption = run.policy_records.clone();
    let changed = selected_action_corruption.iter_mut().find_map(|record| {
        let PolicyEpisodeRecordV2::Decision {
            legal_actions,
            selected_index,
            selected_action_id,
            ..
        } = record
        else {
            return None;
        };
        if legal_actions.len() < 2 {
            return None;
        }
        let replacement = if *selected_index == 0 { 1 } else { 0 };
        *selected_index = replacement;
        *selected_action_id = legal_actions[replacement as usize].stable_id.clone();
        Some(())
    });
    assert!(
        changed.is_some(),
        "test episode needs one multi-action decision"
    );
    validate_policy_episode_records(&selected_action_corruption).unwrap();
    assert!(validate_rollout_artifact_bundle(
        &run.audit_records,
        &selected_action_corruption,
        &manifest
    )
    .unwrap_err()
    .to_string()
    .contains("audit/policy decision mismatch"));

    let mut header_corruption = run.policy_records.clone();
    let PolicyEpisodeRecordV2::Header { kernel_version, .. } = &mut header_corruption[0] else {
        panic!("first policy record must be a header");
    };
    kernel_version.push_str("-corrupt");
    validate_policy_episode_records(&header_corruption).unwrap();
    assert!(
        validate_rollout_artifact_bundle(&run.audit_records, &header_corruption, &manifest)
            .unwrap_err()
            .to_string()
            .contains("shared header mismatch")
    );

    let mut terminal_corruption = run.policy_records.clone();
    let terminal = terminal_corruption.last_mut().unwrap();
    let PolicyEpisodeRecordV2::Terminal {
        terminal_outcome,
        terminal_classification,
        terminal_code,
        winner,
        terminal_reward,
        ..
    } = terminal
    else {
        panic!("last policy record must be terminal");
    };
    *terminal_outcome = TerminalOutcomeV1::Halted;
    *terminal_classification = TerminalClassificationV1::Halted;
    *terminal_code = mtg_kernel::rl::TerminalSafeCodeV2::FailClosed;
    *winner = None;
    *terminal_reward = [0, 0];
    validate_policy_episode_records(&terminal_corruption).unwrap();
    assert!(
        validate_rollout_artifact_bundle(&run.audit_records, &terminal_corruption, &manifest)
            .unwrap_err()
            .to_string()
            .contains("audit/policy terminal mismatch")
    );

    let mut manifest_corruption = manifest.clone();
    manifest_corruption.game_count += 1;
    assert!(validate_rollout_artifact_bundle(
        &run.audit_records,
        &run.policy_records,
        &manifest_corruption
    )
    .unwrap_err()
    .to_string()
    .contains("manifest counts"));

    assert!(write_rollout_artifacts(
        &out_dir,
        &run.audit_records,
        &selected_action_corruption,
        &manifest,
    )
    .is_err());
    assert!(
        !out_dir.exists(),
        "invalid bundle validation must finish before creating the output directory"
    );
}

#[test]
fn rl_contract_policy_stream_is_safe_and_audit_stream_is_privileged() {
    let env_seed = derive_env_seed(9999, 0);
    let policy_seed = derive_policy_seed(9999, 0);
    let run = record_burn_mirror_episode(0, env_seed, policy_seed, 200_000).unwrap();
    let policy_jsonl = records_to_jsonl(&run.policy_records);
    let audit_jsonl = records_to_jsonl(&run.audit_records);
    let audit_game_id = format!(
        "burn_mirror_env_{env_seed:016x}_policy_{policy_seed:016x}_game_{:06}",
        0
    );

    for line in policy_jsonl.lines() {
        let value: Value = serde_json::from_str(line).unwrap();
        for forbidden in [
            "diagnostic_state_hash",
            "diagnostic_state_hash_algorithm",
            "state_hash",
            "env_seed",
            "policy_seed",
            "library_setup",
            "rng_state",
            "hidden_state_marker",
            "event_history",
            "library_setup",
            "shuffle_algorithm",
        ] {
            assert!(
                !contains_key(&value, forbidden),
                "policy-safe record contains forbidden key {forbidden}: {line}"
            );
        }
    }
    assert!(!policy_jsonl.contains("privileged"));
    for forbidden_value in [
        env_seed.to_string(),
        policy_seed.to_string(),
        format!("{env_seed:016x}"),
        format!("{policy_seed:016x}"),
        format!("{env_seed:016X}"),
        format!("{policy_seed:016X}"),
        audit_game_id.clone(),
        "engine_halted".to_string(),
        "fail_closed:".to_string(),
        "decision_cap_reached:".to_string(),
        "ObjectId".to_string(),
    ] {
        assert!(
            !policy_jsonl.contains(&forbidden_value),
            "policy stream leaked forbidden value {forbidden_value}"
        );
    }
    assert!(policy_jsonl.contains("terminal_code"));
    assert!(!policy_jsonl.contains("terminal_reason"));
    assert!(audit_jsonl.contains("diagnostic_state_hash"));
    assert!(audit_jsonl.contains(DIAGNOSTIC_STATE_HASH_ALGORITHM));
    assert!(audit_jsonl.contains("env_seed"));
    assert!(audit_jsonl.contains("policy_seed"));
    assert!(audit_jsonl.contains(&audit_game_id));
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
    )
    .unwrap();
    assert_eq!(manifest.aggregate.p0_wins, 1);
    assert_eq!(manifest.aggregate.p1_wins, 1);
    assert_eq!(manifest.aggregate.draws, 1);
    assert_eq!(manifest.aggregate.truncated, 1);
    assert_eq!(manifest.aggregate.halted, 1);
    assert_eq!(manifest.aggregate.total_decisions, 57);

    let value = serde_json::to_value(&manifest).unwrap();
    assert_eq!(value["schema_version"], MANIFEST_SCHEMA_VERSION);
    assert_eq!(
        value["diagnostic_state_hash_algorithm"],
        DIAGNOSTIC_STATE_HASH_ALGORITHM
    );
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

    let valid_json = serde_json::to_string(&manifest).unwrap();
    assert_eq!(parse_run_manifest_json(&valid_json).unwrap(), manifest);

    let mut unknown_algorithm = value.clone();
    unknown_algorithm["diagnostic_state_hash_algorithm"] = Value::String("unknown-v99".to_string());
    assert!(parse_run_manifest_json(&unknown_algorithm.to_string())
        .unwrap_err()
        .to_string()
        .contains("unsupported diagnostic_state_hash_algorithm"));

    let mut legacy = value.clone();
    legacy["schema_version"] = Value::from(MANIFEST_SCHEMA_VERSION - 1);
    assert!(parse_run_manifest_json(&legacy.to_string())
        .unwrap_err()
        .to_string()
        .contains("unsupported manifest schema_version"));

    let mut missing_algorithm = value;
    missing_algorithm
        .as_object_mut()
        .unwrap()
        .remove("diagnostic_state_hash_algorithm");
    assert!(parse_run_manifest_json(&missing_algorithm.to_string())
        .unwrap_err()
        .to_string()
        .contains("missing field `diagnostic_state_hash_algorithm`"));
}

#[test]
fn rl_contract_manifest_rejects_inconsistent_terminal_tuples() {
    let valid = EpisodeTerminalSummaryV1 {
        episode_id: 0,
        outcome: TerminalOutcomeV1::P0Win,
        classification: TerminalClassificationV1::Natural,
        winner: Some(PlayerSeatV1::P0),
        terminal_reward: [1, -1],
        terminal_reason: "game_over".to_string(),
        decision_count: 10,
    };
    let err = build_run_manifest(
        2,
        5151,
        12,
        vec![],
        Path::new("local-training/kernel_rl/test"),
        std::slice::from_ref(&valid),
        GitMetadataV1 {
            commit: "test".to_string(),
            dirty: GitDirtyFlagV1::Clean,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("game_count"));

    let mut invalid_winner = valid.clone();
    invalid_winner.winner = Some(PlayerSeatV1::P1);
    let err = build_run_manifest(
        1,
        5151,
        12,
        vec![],
        Path::new("local-training/kernel_rl/test"),
        &[invalid_winner],
        GitMetadataV1 {
            commit: "test".to_string(),
            dirty: GitDirtyFlagV1::Clean,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("invalid terminal tuple"));

    let mut invalid_truncated = valid;
    invalid_truncated.outcome = TerminalOutcomeV1::Truncated;
    invalid_truncated.classification = TerminalClassificationV1::Truncated;
    invalid_truncated.winner = Some(PlayerSeatV1::P0);
    invalid_truncated.terminal_reward = [1, -1];
    let err = build_run_manifest(
        1,
        5151,
        12,
        vec![],
        Path::new("local-training/kernel_rl/test"),
        &[invalid_truncated],
        GitMetadataV1 {
            commit: "test".to_string(),
            dirty: GitDirtyFlagV1::Clean,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("invalid terminal tuple"));
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
