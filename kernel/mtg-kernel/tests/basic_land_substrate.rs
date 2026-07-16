use mtg_kernel::card_def::{card_id_by_name, preflight_fully_supported_deck};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{self, Cost, ManaColor, Pip};
use mtg_kernel::rl::{legal_action_candidates_v1, ActionSemanticV1};
use mtg_kernel::rl_session::RL_SESSION_SCHEMA_VERSION;
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Zone};
use mtg_kernel::surface_v2::SurfaceDecision;

fn ready_main1() -> GameState {
    let mut state = GameState::new_from_libraries(&[], &[], |_| String::new(), 7);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state
}

fn put_on_battlefield(state: &mut GameState, name: &str) -> ObjectId {
    let card_def = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"));
    let id = state.objects.push(GameObject {
        card_def,
        name: name.to_string(),
        owner: PlayerId::P0,
        controller: PlayerId::P0,
        zone: Zone::Battlefield,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Counters::default(),
        attachments: Vec::new(),
        v4: mtg_kernel::state::ObjectStateV4::from_card_def(card_def),
        plotted_turn: None,
        zone_change_count: 0,
    });
    state.players[0].battlefield.push(id);
    id
}

fn mana_action_id(state: &GameState, island: ObjectId) -> String {
    let decision = engine::advance_until_decision(&mut state.clone());
    assert!(matches!(decision, Decision::CastSpellOrPass { .. }));
    legal_action_candidates_v1(&SurfaceDecision::Decision(decision), state)
        .expect("legal action projection")
        .into_iter()
        .find_map(|candidate| match candidate.record.semantic {
            ActionSemanticV1::ActivateManaAbility { source, .. } if source.arena_id == island.0 => {
                Some(candidate.record.stable_id)
            }
            _ => None,
        })
        .expect("Island activation is a legal action")
}

#[test]
fn island_activation_is_u_taps_once_and_reuses_after_untap() {
    let mut state = ready_main1();
    let island = put_on_battlefield(&mut state, "Island");

    engine::step(&mut state, Action::ActivateManaAbility(island)).unwrap();
    assert!(state.objects.get(island).tapped);
    assert_eq!(state.players[0].mana_pool[ManaColor::U.pool_index()], 1);
    assert!(engine::step(&mut state, Action::ActivateManaAbility(island)).is_err());

    // The turn engine's untap action clears this same durable bit. Set it
    // directly here to isolate the reusable mana-source contract from the
    // phase driver, then prove the ability is available again.
    state.objects.get_mut(island).tapped = false;
    engine::step(&mut state, Action::ActivateManaAbility(island)).unwrap();
    assert_eq!(state.players[0].mana_pool[ManaColor::U.pool_index()], 2);
}

#[test]
fn two_islands_pay_uu_exactly() {
    let mut state = ready_main1();
    let first = put_on_battlefield(&mut state, "Island");
    let second = put_on_battlefield(&mut state, "Island");
    let uu = Cost {
        pips: &[Pip::Colored(ManaColor::U), Pip::Colored(ManaColor::U)],
        generic: 0,
        x_count: 0,
    };

    let source_plan = mana::can_pay(&uu, 0, PlayerId::P0, &state).expect("two Islands pay UU");
    assert_eq!(
        source_plan.taps,
        vec![(first, ManaColor::U), (second, ManaColor::U)]
    );

    engine::step(&mut state, Action::ActivateManaAbility(first)).unwrap();
    engine::step(&mut state, Action::ActivateManaAbility(second)).unwrap();
    let pool_plan = mana::can_pay(&uu, 0, PlayerId::P0, &state).expect("floating UU pays UU");
    assert!(pool_plan.taps.is_empty());
    assert_eq!(pool_plan.pool_used[ManaColor::U.pool_index()], 2);
}

#[test]
fn island_snapshot_restore_preserves_schema_v4_and_stable_action_identity() {
    assert_eq!(RL_SESSION_SCHEMA_VERSION, 4);
    let mut state = ready_main1();
    let island = put_on_battlefield(&mut state, "Island");
    preflight_fully_supported_deck(&[state.objects.get(island).card_def]).unwrap();

    let action_id = mana_action_id(&state, island);
    let hash = state.state_hash();
    let snapshot = state.snapshot();
    engine::step(&mut state, Action::ActivateManaAbility(island)).unwrap();
    assert_ne!(state.state_hash(), hash);

    state.restore(&snapshot);
    assert_eq!(state.state_hash(), hash);
    assert_eq!(mana_action_id(&state, island), action_id);
    engine::step(&mut state, Action::ActivateManaAbility(island)).unwrap();
    assert_eq!(state.players[0].mana_pool[ManaColor::U.pool_index()], 1);
}
