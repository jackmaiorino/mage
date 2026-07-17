use mtg_kernel::card_def::card_id_by_name;
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::rl::{
    legal_action_candidates_v1, observe_v2, ActionSemanticV1, PendingEffectChoiceSemanticV4,
    TargetRefV1, TargetSelectionPurposeV4,
};
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Target, Zone};
use mtg_kernel::surface_v2::{HarnessSurfaceV2, SurfaceDecision};

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"))
}

fn card_name(card_def: u16) -> String {
    mtg_kernel::card_def::CARD_DEFS[card_def as usize]
        .name
        .to_string()
}

fn put_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id(name);
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
        Zone::Library => state.players[player.index()].library.push(id),
        Zone::Graveyard => state.players[player.index()].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Command => state.command.push(id),
        Zone::Stack => panic!("test helper does not create stack objects"),
    }
    id
}

fn ready_winding_way_with_library(library_names: &[&str]) -> (GameState, ObjectId, Vec<ObjectId>) {
    let library_defs = library_names
        .iter()
        .map(|name| card_id(name))
        .collect::<Vec<_>>();
    let mut state = GameState::new_from_libraries(&library_defs, &[], card_name, 77);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    put_object(&mut state, PlayerId::P0, "Forest", Zone::Battlefield);
    put_object(&mut state, PlayerId::P0, "Forest", Zone::Battlefield);
    let winding = put_object(&mut state, PlayerId::P0, "Winding Way", Zone::Hand);
    let library = state.players[0].library.clone();
    (state, winding, library)
}

fn ready_winding_way() -> (GameState, ObjectId, Vec<ObjectId>) {
    ready_winding_way_with_library(&[
        "Elvish Mystic",
        "Snow-Covered Forest",
        "Lightning Bolt",
        "Quirion Ranger",
        "Island",
    ])
}

fn advance_to_choice(state: &mut GameState, winding: ObjectId) -> Decision {
    for _ in 0..8 {
        let decision = engine::advance_until_decision(state);
        match decision {
            Decision::ChooseEffectOption {
                player: PlayerId::P0,
                source,
                option_count: 2,
            } if source == winding => return decision,
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision before Winding Way choice: {other:?}"),
        }
    }
    panic!("Winding Way did not reach its resolution choice")
}

fn choose_and_finish(state: &mut GameState, option: u16, graveyard_order: &[ObjectId]) {
    engine::step(state, Action::ChooseEffectOption(option)).unwrap();
    for &object in graveyard_order {
        let decision = engine::advance_until_decision(state);
        let Decision::ChooseEffectTargets { legal_targets, .. } = decision else {
            panic!("expected graveyard-order choice, got {decision:?}");
        };
        assert!(legal_targets.contains(&Target::Object(object)));
        engine::step(state, Action::ChooseEffectTarget(Target::Object(object))).unwrap();
    }
    let finished = engine::advance_until_decision(state);
    assert!(matches!(finished, Decision::CastSpellOrPass { .. }));
    assert!(state.engine.pending_effect.is_none());
    assert!(state.stack.is_empty());
}

fn target_actions(state: &GameState, decision: &Decision) -> Vec<(ObjectId, String)> {
    legal_action_candidates_v1(&SurfaceDecision::Decision(decision.clone()), state)
        .unwrap()
        .into_iter()
        .map(|candidate| match candidate.record.semantic {
            ActionSemanticV1::ChooseEffectTarget {
                target: TargetRefV1::Object { object },
                ..
            } => (ObjectId(object.arena_id), candidate.record.stable_id),
            other => panic!("unexpected Winding Way order action: {other:?}"),
        })
        .collect()
}

fn choice_actions(state: &GameState, decision: &Decision) -> Vec<(u16, String)> {
    legal_action_candidates_v1(&SurfaceDecision::Decision(decision.clone()), state)
        .unwrap()
        .into_iter()
        .map(|candidate| match candidate.record.semantic {
            ActionSemanticV1::ChooseEffectOption {
                option_index,
                option_count: 2,
                ..
            } => (option_index, candidate.record.stable_id),
            other => panic!("unexpected Winding Way action: {other:?}"),
        })
        .collect()
}

#[test]
fn winding_way_choice_is_mid_resolution_private_then_public_and_deterministic() {
    let (mut state, winding, library) = ready_winding_way();
    engine::step(&mut state, Action::CastSpell(winding)).unwrap();
    let decision = advance_to_choice(&mut state, winding);

    assert_eq!(
        state.stack.len(),
        1,
        "resolving spell stays publicly on stack"
    );
    assert_eq!(state.stack[0].source, winding);
    assert_eq!(state.objects.get(winding).zone, Zone::Stack);
    let actions = choice_actions(&state, &decision);
    assert_eq!(
        actions,
        vec![
            (0, "legal-action-v4:421bbc1e7151bc7c".to_string()),
            (1, "legal-action-v4:c72a26e3682d12db".to_string()),
        ]
    );

    for observer in [PlayerId::P0, PlayerId::P1] {
        let pending_observation =
            observe_v2(&state, &HarnessSurfaceV2::new(), observer, 1).unwrap();
        assert!(
            pending_observation
                .known_library_cards
                .iter()
                .all(Vec::is_empty),
            "neither perspective sees the library before the type choice"
        );
        assert!(matches!(
            pending_observation
                .projection
                .engine_context
                .pending_effect
                .unwrap()
                .choice,
            Some(PendingEffectChoiceSemanticV4::Options {
                player: mtg_kernel::rl::PlayerSeatV1::P0,
                option_count: 2,
                ..
            })
        ));
    }

    let invalid_before = state.clone();
    assert!(engine::step(&mut state, Action::ChooseEffectOption(2)).is_err());
    assert_eq!(state, invalid_before);

    let option_snapshot = state.snapshot();
    let option_hash = state.state_hash();
    state.restore(&option_snapshot);
    assert_eq!(state.state_hash(), option_hash);
    let restored_option_decision = engine::advance_until_decision(&mut state);
    assert_eq!(choice_actions(&state, &restored_option_decision), actions);

    engine::step(&mut state, Action::ChooseEffectOption(0)).unwrap();
    let order_decision = engine::advance_until_decision(&mut state);
    assert!(matches!(
        order_decision,
        Decision::ChooseEffectTargets {
            player: PlayerId::P0,
            source,
            selected_count: 0,
            min_targets: 2,
            max_targets: 2,
            can_finish: false,
            ..
        } if source == winding
    ));
    assert_eq!(state.players[0].hand, vec![library[0], library[3]]);
    assert_eq!(
        state.players[0].library,
        vec![library[1], library[2], library[4]]
    );
    assert!(state.players[0].graveyard.is_empty());
    assert_eq!(state.stack[0].source, winding);

    let p0_order = observe_v2(&state, &HarnessSurfaceV2::new(), PlayerId::P0, 2).unwrap();
    let p1_order = observe_v2(&state, &HarnessSurfaceV2::new(), PlayerId::P1, 2).unwrap();
    let p0_choice = p0_order
        .projection
        .engine_context
        .pending_effect
        .as_ref()
        .unwrap()
        .choice
        .as_ref()
        .unwrap();
    let p1_choice = p1_order
        .projection
        .engine_context
        .pending_effect
        .as_ref()
        .unwrap()
        .choice
        .as_ref()
        .unwrap();
    assert_eq!(p0_choice, p1_choice, "graveyard ordering is fully public");
    assert!(matches!(
        p0_choice,
        PendingEffectChoiceSemanticV4::Targets {
            player: mtg_kernel::rl::PlayerSeatV1::P0,
            structural_path,
            selected_targets,
            legal_targets,
            min_targets: 2,
            max_targets: 2,
            can_finish: false,
            ordered: true,
            purpose: TargetSelectionPurposeV4::CardSelection,
        } if structural_path == &vec![0, 1]
            && selected_targets.is_empty()
            && legal_targets.len() == 2
    ));
    assert_eq!(
        p1_order.known_library_cards[0]
            .iter()
            .map(|known| known.card.card_name.as_str())
            .collect::<Vec<_>>(),
        vec!["Snow-Covered Forest", "Lightning Bolt"]
    );

    let order_actions = target_actions(&state, &order_decision);
    assert_eq!(
        order_actions,
        vec![
            (library[1], "legal-action-v4:655d446d79d710ae".to_string(),),
            (library[2], "legal-action-v4:1140aaf95a8c2f64".to_string(),),
        ]
    );
    let invalid_order_before = state.clone();
    for action in [
        Action::FinishEffectSelection,
        Action::ChooseEffectOption(1),
        Action::ChooseEffectTarget(Target::Player(PlayerId::P1)),
        Action::ChooseEffectTarget(Target::Object(library[4])),
    ] {
        assert!(engine::step(&mut state, action).is_err());
        assert_eq!(state, invalid_order_before);
    }

    let order_snapshot = state.snapshot();
    let order_hash = state.state_hash();
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[2])),
    )
    .unwrap();
    assert!(state
        .engine
        .pending_effect
        .as_ref()
        .unwrap()
        .choice
        .is_none());
    assert_eq!(state.stack[0].source, winding);
    assert!(state.players[0].graveyard.is_empty());
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.players[0].hand, vec![library[0], library[3]]);
    assert_eq!(
        state.players[0].graveyard,
        vec![library[2], library[1], winding]
    );
    assert_eq!(state.players[0].library, vec![library[4]]);
    assert_eq!(state.players[0].draws_this_turn, 0);
    assert!(!state.players[0].drew_from_empty);

    let opponent = observe_v2(&state, &HarnessSurfaceV2::new(), PlayerId::P1, 2).unwrap();
    assert_eq!(
        opponent.known_hand_cards[0]
            .iter()
            .map(|known| known.card_name.as_str())
            .collect::<Vec<_>>(),
        vec!["Elvish Mystic", "Quirion Ranger"]
    );
    assert!(opponent.known_library_cards[0].is_empty());

    let creature_result = state.clone();
    let creature_hash = state.state_hash();
    state.restore(&order_snapshot);
    assert_eq!(state.state_hash(), order_hash);
    let restored_decision = engine::advance_until_decision(&mut state);
    assert_eq!(target_actions(&state, &restored_decision), order_actions);
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[2])),
    )
    .unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(state.state_hash(), creature_hash);
    assert_eq!(state, creature_result);

    let zone_changes = state
        .engine
        .event_history
        .iter()
        .filter_map(|event| match event {
            mtg_kernel::event::CommittedEvent::ZoneChange { object, .. } => Some(*object),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        zone_changes,
        vec![library[0], library[3], library[2], library[1], winding]
    );
}

#[test]
fn winding_way_land_branch_keeps_land_and_mills_everything_else_in_order() {
    let (mut state, winding, library) = ready_winding_way();
    engine::step(&mut state, Action::CastSpell(winding)).unwrap();
    advance_to_choice(&mut state, winding);
    engine::step(&mut state, Action::ChooseEffectOption(1)).unwrap();
    let first_order = engine::advance_until_decision(&mut state);
    let first_actions = target_actions(&state, &first_order);
    assert_eq!(
        first_actions,
        vec![
            (library[0], "legal-action-v4:ba8f42fdb31689ce".to_string(),),
            (library[2], "legal-action-v4:2cdd63b48fff9f14".to_string(),),
            (library[3], "legal-action-v4:58cec77aa2dae4c1".to_string(),),
        ]
    );
    let initial_order_snapshot = state.snapshot();

    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[0])),
    )
    .unwrap();
    let second_order = engine::advance_until_decision(&mut state);
    assert!(matches!(
        second_order,
        Decision::ChooseEffectTargets {
            selected_count: 1,
            min_targets: 3,
            max_targets: 3,
            can_finish: false,
            ..
        }
    ));
    let second_actions = target_actions(&state, &second_order);
    assert_eq!(
        second_actions,
        vec![
            (library[2], "legal-action-v4:6543709b715b0e6d".to_string(),),
            (library[3], "legal-action-v4:df00ca365e42dea8".to_string(),),
        ]
    );

    let invalid_before = state.clone();
    for action in [
        Action::ChooseEffectTarget(Target::Object(library[0])),
        Action::ChooseEffectTarget(Target::Object(library[4])),
        Action::ChooseEffectTarget(Target::Player(PlayerId::P0)),
        Action::FinishEffectSelection,
        Action::ChooseEffectOption(0),
    ] {
        assert!(engine::step(&mut state, action).is_err());
        assert_eq!(state, invalid_before);
    }

    let second_snapshot = state.snapshot();
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[3])),
    )
    .unwrap();
    assert!(state
        .engine
        .pending_effect
        .as_ref()
        .unwrap()
        .choice
        .is_none());
    engine::advance_until_decision(&mut state);

    assert_eq!(state.players[0].hand, vec![library[1]]);
    assert_eq!(
        state.players[0].graveyard,
        vec![library[0], library[3], library[2], winding]
    );
    assert_eq!(state.players[0].library, vec![library[4]]);
    let opponent = observe_v2(&state, &HarnessSurfaceV2::new(), PlayerId::P1, 3).unwrap();
    assert_eq!(opponent.known_hand_cards[0].len(), 1);
    assert_eq!(
        opponent.known_hand_cards[0][0].card_name,
        "Snow-Covered Forest"
    );

    let first_result = state.clone();
    state.restore(&second_snapshot);
    assert_eq!(target_actions(&state, &second_order), second_actions);
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[3])),
    )
    .unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(state, first_result);

    state.restore(&initial_order_snapshot);
    assert_eq!(target_actions(&state, &first_order), first_actions);
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[3])),
    )
    .unwrap();
    engine::advance_until_decision(&mut state);
    engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(library[2])),
    )
    .unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(
        state.players[0].graveyard,
        vec![library[3], library[2], library[0], winding]
    );
    assert_ne!(state.state_hash(), first_result.state_hash());
}

#[test]
fn winding_way_three_card_graveyard_order_covers_all_six_permutations() {
    let permutations = [
        [0_usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    for permutation in permutations {
        let (mut state, winding, library) = ready_winding_way();
        let remainder = [library[0], library[2], library[3]];
        engine::step(&mut state, Action::CastSpell(winding)).unwrap();
        advance_to_choice(&mut state, winding);
        engine::step(&mut state, Action::ChooseEffectOption(1)).unwrap();

        for (pick_number, &index) in permutation[..2].iter().enumerate() {
            let decision = engine::advance_until_decision(&mut state);
            let Decision::ChooseEffectTargets {
                selected_count,
                can_finish: false,
                ref legal_targets,
                ..
            } = decision
            else {
                panic!("expected ordered graveyard selection, got {decision:?}");
            };
            assert_eq!(selected_count, pick_number as u16);
            let object = remainder[index];
            assert!(legal_targets.contains(&Target::Object(object)));
            assert!(
                legal_action_candidates_v1(&SurfaceDecision::Decision(decision), &state)
                    .unwrap()
                    .iter()
                    .all(|candidate| !matches!(
                        candidate.record.semantic,
                        ActionSemanticV1::FinishEffectSelection { .. }
                    ))
            );
            engine::step(
                &mut state,
                Action::ChooseEffectTarget(Target::Object(object)),
            )
            .unwrap();
        }

        assert!(state
            .engine
            .pending_effect
            .as_ref()
            .unwrap()
            .choice
            .is_none());
        assert!(matches!(
            engine::advance_until_decision(&mut state),
            Decision::CastSpellOrPass { .. }
        ));
        let expected = permutation.map(|index| remainder[index]);
        assert_eq!(&state.players[0].graveyard[..3], &expected);
        assert_eq!(
            state.players[0].graveyard[2], expected[2],
            "the unchosen forced final card is topmost within the moved batch"
        );
        assert_eq!(state.players[0].graveyard[3], winding);
    }
}

#[test]
fn winding_way_stale_library_incarnation_fails_closed() {
    let (mut state, winding, library) = ready_winding_way();
    engine::step(&mut state, Action::CastSpell(winding)).unwrap();
    advance_to_choice(&mut state, winding);
    engine::step(&mut state, Action::ChooseEffectOption(0)).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::ChooseEffectTargets { .. }
    ));
    let stale = library[1];
    let pending_snapshot = state.engine.pending_effect.clone();

    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(stale, Zone::Graveyard),
    );
    mtg_kernel::event::propose_and_commit(
        &mut state,
        mtg_kernel::event::ProposedEvent::zone_change(stale, Zone::Library),
    );
    state.engine.pending_effect = pending_snapshot;
    assert_eq!(state.objects.get(stale).zone, Zone::Library);
    assert_eq!(state.objects.get(stale).zone_change_count, 2);

    let before_action = state.clone();
    assert!(engine::step(
        &mut state,
        Action::ChooseEffectTarget(Target::Object(stale))
    )
    .is_err());
    assert_eq!(state, before_action);

    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::Halted {
            mechanic: mtg_kernel::engine::UnsupportedMechanic::InvalidEffectContinuation,
            source,
        } if source == winding
    ));
    assert_eq!(state.objects.get(stale).zone, Zone::Library);
    assert_eq!(state.objects.get(stale).zone_change_count, 2);
    assert!(state.players[0].graveyard.is_empty());
    assert_eq!(state.stack[0].source, winding);
}

#[test]
fn winding_way_matches_checked_in_xmage_top_four_golden_with_registered_cards() {
    // Bounded source-level oracle: XMage's checked-in
    // ImpulseDrawAndMillZoneTest#testWindingWayRevealSplitsCorrectlyBetweenHandAndGraveyard
    // at 3a86580e051f257d1f939ae67f7c06f2dbcef96e (whose parent rules baseline is
    // 0723fc0c2be922af47b0ef0539f28114cc23b998). This is intentionally a focused
    // top-four split golden, not a whole-game trace or fixed-provenance v2 replay.
    // Grizzly Bears is outside the frozen Pauper registry, so the registered
    // Quirion Ranger occupies that oracle's second Creature slot.
    let (mut state, winding, library) = ready_winding_way_with_library(&[
        "Elvish Mystic",
        "Quirion Ranger",
        "Counterspell",
        "Lightning Bolt",
    ]);
    engine::step(&mut state, Action::CastSpell(winding)).unwrap();
    let decision = advance_to_choice(&mut state, winding);
    assert_eq!(
        choice_actions(&state, &decision)
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        vec![0, 1],
        "the Java oracle chooses Creature first; Land remains option one"
    );
    choose_and_finish(&mut state, 0, &[library[2]]);

    assert_eq!(state.players[0].hand, vec![library[0], library[1]]);
    assert_eq!(
        state.players[0].graveyard,
        vec![library[2], library[3], winding]
    );
    assert!(state.players[0].library.is_empty());
    assert_eq!(
        state.players[0]
            .hand
            .iter()
            .map(|id| state.objects.get(*id).name.as_str())
            .collect::<Vec<_>>(),
        vec!["Elvish Mystic", "Quirion Ranger"]
    );
    assert_eq!(
        state.players[0]
            .graveyard
            .iter()
            .map(|id| state.objects.get(*id).name.as_str())
            .collect::<Vec<_>>(),
        vec!["Counterspell", "Lightning Bolt", "Winding Way"]
    );
}

#[test]
fn winding_way_skips_graveyard_order_when_zero_or_one_card_moves_there() {
    for (library_names, expected_graveyard_name) in [
        (
            vec![
                "Elvish Mystic",
                "Quirion Ranger",
                "Llanowar Elves",
                "Priest of Titania",
            ],
            None,
        ),
        (
            vec![
                "Elvish Mystic",
                "Quirion Ranger",
                "Llanowar Elves",
                "Lightning Bolt",
            ],
            Some("Lightning Bolt"),
        ),
    ] {
        let (mut state, winding, _) = ready_winding_way_with_library(&library_names);
        engine::step(&mut state, Action::CastSpell(winding)).unwrap();
        advance_to_choice(&mut state, winding);
        choose_and_finish(&mut state, 0, &[]);
        assert_eq!(
            state.players[0]
                .graveyard
                .iter()
                .filter(|&&object| object != winding)
                .map(|&object| state.objects.get(object).name.as_str())
                .collect::<Vec<_>>(),
            expected_graveyard_name.into_iter().collect::<Vec<_>>()
        );
    }
}
