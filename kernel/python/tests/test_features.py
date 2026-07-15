from __future__ import annotations

import copy
import hashlib
import unittest

import torch

from mtg_kernel_rl.features import (
    FeatureSchemaError,
    assert_action_classified,
    assert_observation_classified,
    classification_registry,
    encode_decision,
    encoding_contract_fingerprint,
    every_action_variant_fixture,
    feature_contract_fingerprint,
    iter_classified_leaves,
    validate_legal_actions_contract,
)
from mtg_kernel_rl.model import KernelPolicyValueNet

from fixtures import complete_legal_actions, complete_observation, deep_copy, legal_actions, observation, public_card, stable_ref


def assert_encoded_equal(test: unittest.TestCase, a, b) -> None:
    test.assertTrue(torch.equal(a.state, b.state))
    test.assertTrue(torch.equal(a.object_features, b.object_features))
    test.assertTrue(torch.equal(a.object_card_ids, b.object_card_ids))
    test.assertTrue(torch.equal(a.object_groups, b.object_groups))
    test.assertTrue(torch.equal(a.object_node_ids, b.object_node_ids))
    test.assertTrue(torch.equal(a.edge_features, b.edge_features))
    test.assertTrue(torch.equal(a.edge_source_indices, b.edge_source_indices))
    test.assertTrue(torch.equal(a.edge_target_indices, b.edge_target_indices))
    test.assertTrue(torch.equal(a.action_features, b.action_features))
    test.assertTrue(torch.equal(a.action_ref_features, b.action_ref_features))
    test.assertTrue(torch.equal(a.action_ref_card_ids, b.action_ref_card_ids))
    test.assertTrue(torch.equal(a.action_ref_action_indices, b.action_ref_action_indices))
    test.assertTrue(torch.equal(a.action_ref_node_indices, b.action_ref_node_indices))


def encoded_digest(encoded) -> str:
    h = hashlib.sha256()
    for tensor in (
        encoded.state,
        encoded.object_features,
        encoded.object_card_ids,
        encoded.object_groups,
        encoded.object_node_ids,
        encoded.edge_features,
        encoded.edge_source_indices,
        encoded.edge_target_indices,
        encoded.action_features,
        encoded.action_ref_features,
        encoded.action_ref_card_ids,
        encoded.action_ref_action_indices,
        encoded.action_ref_node_indices,
    ):
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def action_representation(encoded, index: int) -> bytes:
    h = hashlib.sha256()
    h.update(encoded.action_features[index].detach().cpu().contiguous().numpy().tobytes())
    mask = encoded.action_ref_action_indices == index
    h.update(encoded.action_ref_features[mask].detach().cpu().contiguous().numpy().tobytes())
    h.update(encoded.action_ref_card_ids[mask].detach().cpu().contiguous().numpy().tobytes())
    h.update(encoded.action_ref_node_indices[mask].detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def renumber_arena_ids(value, delta: int = 1000):
    if isinstance(value, dict):
        return {k: (v + delta if k == "arena_id" and type(v) is int else renumber_arena_ids(v, delta)) for k, v in value.items()}
    if isinstance(value, list):
        return [renumber_arena_ids(v, delta) for v in value]
    return value


def renumber_all_arena_references(value, delta: int = 1000):
    out = renumber_arena_ids(value, delta)

    def walk(v):
        if isinstance(v, dict):
            for key, child in v.items():
                if key == "attachments":
                    v[key] = [item + delta if type(item) is int else item for item in child]
                else:
                    walk(child)
        elif isinstance(v, list):
            for child in v:
                walk(child)

    walk(out)
    return out


def remap_all_arena_references(value, mapping: dict[int, int]):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key == "arena_id" and type(child) is int:
                out[key] = mapping.get(child, child)
            elif key == "attachments":
                out[key] = [mapping.get(item, item) if type(item) is int else item for item in child]
            else:
                out[key] = remap_all_arena_references(child, mapping)
        return out
    if isinstance(value, list):
        return [remap_all_arena_references(v, mapping) for v in value]
    return value


def shift_zone_generations(value, delta: int):
    if isinstance(value, dict):
        return {
            key: (child + delta if key in {"zone_change_count", "zone_change_generation"} and type(child) is int else shift_zone_generations(child, delta))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [shift_zone_generations(v, delta) for v in value]
    return value


def shift_card_db_for_arena(value, arena_id: int, delta: int):
    if isinstance(value, dict):
        out = {}
        is_target_ref = value.get("arena_id") == arena_id and "card_db_id" in value
        for key, child in value.items():
            if key == "card_db_id" and is_target_ref and type(child) is int:
                out[key] = child + delta
            else:
                out[key] = shift_card_db_for_arena(child, arena_id, delta)
        return out
    if isinstance(value, list):
        return [shift_card_db_for_arena(v, arena_id, delta) for v in value]
    return value


def swap_seats_value(value):
    if isinstance(value, str):
        return {"p0": "p1", "p1": "p0"}.get(value, value)
    if isinstance(value, list):
        return [swap_seats_value(v) for v in value]
    if isinstance(value, dict):
        return {k: swap_seats_value(v) for k, v in value.items()}
    return value


def seat_swapped(obs, actions):
    swapped_obs = swap_seats_value(deep_copy(obs))
    p = swapped_obs["projection"]
    for key in ("life_totals", "mana_pools", "hand_counts", "library_counts", "player_status", "battlefield", "graveyards"):
        p[key] = [p[key][1], p[key][0]]
    swapped_actions = swap_seats_value(deep_copy(actions))
    return swapped_obs, swapped_actions


class FeatureEncodingTest(unittest.TestCase):
    def test_observation_and_all_action_variants_are_classified(self) -> None:
        obs = complete_observation()
        assert_observation_classified(obs)
        actions = every_action_variant_fixture(obs["own_hand"][0]["stable"], obs["projection"]["battlefield"][1][0]["stable"], obs["own_hand"][1]["stable"])
        for action in actions:
            assert_action_classified(action)
        validate_legal_actions_contract(actions, "p0")
        encoded = encode_decision(obs, actions)
        self.assertEqual(encoded.action_features.shape[0], len(actions))
        self.assertGreater(encoded.action_ref_features.shape[0], len(actions))
        self.assertEqual(len(feature_contract_fingerprint()), 64)
        self.assertEqual(len(encoding_contract_fingerprint()), 64)
        self.assertIn("observation.projection.stack.[].controller", classification_registry())

    def test_unknown_leaf_and_ambiguous_action_fail_closed(self) -> None:
        obs = observation()
        obs["projection"]["unknown_leaf"] = 1
        with self.assertRaises(FeatureSchemaError):
            encode_decision(obs, legal_actions())
        ambiguous = {"schema_version": 2, "selected_index": 0, "stable_id": "x", "display_text": None, "semantic": {"action_kind": "ambiguous", "reason": "text"}}
        with self.assertRaises(FeatureSchemaError):
            encode_decision(observation(), [ambiguous])

    def test_path_aware_schema_rejects_wrong_context_and_bad_shapes(self) -> None:
        cases = []
        wrong_context = complete_observation()
        wrong_context["projection"]["card_name"] = "known field wrong object"
        cases.append(wrong_context)
        extra_known = complete_observation()
        extra_known["projection"]["winner"] = "p0"
        cases.append(extra_known)
        missing_nested = complete_observation()
        del missing_nested["projection"]["battlefield"][0][0]["characteristics"]["type_flags"]["land"]
        cases.append(missing_nested)
        wrong_shape = complete_observation()
        wrong_shape["projection"]["life_totals"] = {"p0": 20, "p1": 20}
        cases.append(wrong_shape)
        bool_as_int = complete_observation()
        bool_as_int["projection"]["turn"] = True
        cases.append(bool_as_int)
        invalid_enum = complete_observation()
        invalid_enum["projection"]["phase"] = "combat"
        cases.append(invalid_enum)
        negative_id = complete_observation()
        negative_id["projection"]["battlefield"][0][0]["stable"]["card_db_id"] = -1
        cases.append(negative_id)
        bad_tuple = complete_observation()
        bad_tuple["projection"]["active_player"] = "p1"
        bad_tuple["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {
                    "current_attacker": bad_tuple["projection"]["battlefield"][1][0]["stable"],
                    "accumulated": [],
                    "remaining": [[bad_tuple["projection"]["battlefield"][1][0]["stable"], [bad_tuple["projection"]["battlefield"][0][0]["stable"]]]],
                },
            }
        )
        bad_tuple["projection"]["surface_context"]["private_blockers"]["remaining"][0] = [stable_ref(1, 1)]
        cases.append(bad_tuple)
        for case in cases:
            with self.assertRaises(FeatureSchemaError):
                assert_observation_classified(case)

    def test_complete_fixture_leaf_paths_are_classified(self) -> None:
        obs_leaves = iter_classified_leaves(complete_observation(), "observation")
        action_leaves = []
        for action in complete_legal_actions():
            action_leaves.extend(iter_classified_leaves(action, "legal_action"))
        self.assertGreater(len(obs_leaves), 100)
        self.assertGreater(len(action_leaves), 30)
        for _path, classification in obs_leaves + action_leaves:
            self.assertIn(classification, {"model_input", "operational_only", "forbidden"})

    def test_valid_low_frequency_engine_and_surface_branches_pass(self) -> None:
        base = complete_observation()
        p0_creature = base["projection"]["battlefield"][0][0]["stable"]
        p0_land = base["projection"]["battlefield"][0][1]["stable"]
        p1_creature = base["projection"]["battlefield"][1][0]["stable"]
        hand0 = base["own_hand"][0]["stable"]
        hand1 = base["own_hand"][1]["stable"]
        engine_cases = [
            (
                "pending_cast",
                {
                    "source": hand0,
                    "controller": "p0",
                    "chosen_targets": [{"target_kind": "object", "object": p1_creature}],
                    "is_flashback": False,
                    "cast_mode": "Alternative",
                    "additional_cost_discarded": [hand1],
                    "mode_chosen": 1,
                    "origin_zone": "Hand",
                    "sacrifice_chosen": [p0_land],
                    "kicked": True,
                },
            ),
            (
                "pending_activation",
                {"source": p0_land, "controller": "p0", "ability_index": 2, "chosen_targets": [{"target_kind": "player", "player": "p1"}], "cost_discard_paid": [hand1]},
            ),
            ("pending_discard", {"player": "p0", "count": 1, "resume_stage": "finish_spell_resolution", "resume_source": p0_creature}),
            (
                "pending_optional_cost",
                {
                    "player": "p0",
                    "source": hand0,
                    "discard_cards": 1,
                    "sacrifice_lands": 1,
                    "discard_payable": True,
                    "sacrifice_payable": True,
                    "spell_resume_source": hand0,
                    "spell_resume_zone": "Hand",
                },
            ),
            (
                "pending_optional_cost_sacrifice",
                {"player": "p0", "source": hand0, "remaining": 1, "chosen": [p0_land], "spell_resume_source": hand0, "spell_resume_zone": "Hand"},
            ),
            (
                "pending_triggers",
                [
                    {"source": p0_creature, "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
                    {"source": p1_creature, "controller": "p1", "trigger_kind": "madness_offer", "kicked": True},
                ],
            ),
        ]
        for stage, payload in engine_cases:
            obs = complete_observation()
            engine = obs["projection"]["engine_context"]
            engine["current_stage"] = stage
            engine[stage] = payload
            assert_observation_classified(obs)
            encode_decision(obs, complete_legal_actions())

        paused_activation = complete_observation()
        engine = paused_activation["projection"]["engine_context"]
        engine["current_stage"] = "pending_activation"
        engine["pending_activation"] = {"source": p0_land, "controller": "p0", "ability_index": 2, "chosen_targets": [], "cost_discard_paid": None}
        engine["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "finish_activation", "resume_source": None}
        paused_activation["projection"]["surface_context"].update(
            {"current_stage": "discard_pick", "private_discard": {"chosen": [], "remaining_choices": [hand0, hand1], "remaining_needed": 1}}
        )
        assert_observation_classified(paused_activation)
        encode_decision(paused_activation, complete_legal_actions())

        blockers = complete_observation()
        blockers["projection"]["active_player"] = "p1"
        blockers["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {"current_attacker": p1_creature, "accumulated": [], "remaining": [[p1_creature, [p0_creature]]]},
            }
        )
        assert_observation_classified(blockers)
        discard = complete_observation()
        discard["projection"]["surface_context"].update(
            {"current_stage": "discard_pick", "private_discard": {"chosen": [hand0], "remaining_choices": [hand1], "remaining_needed": 1}}
        )
        assert_observation_classified(discard)
        for stage in ("optional_cost_use", "optional_cost_which"):
            optional = complete_observation()
            optional["projection"]["surface_context"].update(
                {"current_stage": stage, "private_optional_cost": {"discard_payable": True, "sacrifice_payable": False, "stage": stage}}
            )
            assert_observation_classified(optional)

    def test_invalid_engine_surface_context_combinations_fail(self) -> None:
        bad = complete_observation()
        bad["projection"]["engine_context"]["current_stage"] = "pending_cast"
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)
        bad = complete_observation()
        bad["projection"]["engine_context"]["current_stage"] = "pending_cast"
        bad["projection"]["engine_context"]["pending_cast"] = {
            "source": bad["own_hand"][0]["stable"],
            "controller": "p0",
            "chosen_targets": [],
            "is_flashback": False,
            "cast_mode": None,
            "additional_cost_discarded": None,
            "mode_chosen": None,
            "origin_zone": "Hand",
            "sacrifice_chosen": [],
            "kicked": None,
        }
        bad["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "none", "resume_source": None}
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)
        bad = complete_observation()
        bad["projection"]["surface_context"]["private_discard"] = {"chosen": [], "remaining_choices": [bad["own_hand"][0]["stable"]], "remaining_needed": 1}
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)
        bad = complete_observation()
        bad["projection"]["active_player"] = "p0"
        bad["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {"current_attacker": bad["projection"]["battlefield"][0][0]["stable"], "accumulated": [], "remaining": []},
            }
        )
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)

    def test_names_display_stable_ids_hashes_do_not_change_features_or_logits(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        mutated_obs = deep_copy(obs)
        mutated_actions = deep_copy(actions)
        mutated_obs["visible_projection_hash"] = 999
        mutated_obs["own_hand"][0]["card_name"] = "Different"
        mutated_obs["projection"]["battlefield"][0][0]["card_name"] = "Renamed"
        mutated_obs["projection"]["continuous_effects"][0]["timestamp"] = 777
        mutated_actions[0]["stable_id"] = "changed"
        mutated_actions[1]["display_text"] = "Changed text"
        a = encode_decision(obs, actions)
        b = encode_decision(mutated_obs, mutated_actions)
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, value_a = model(a)
        logits_b, value_b = model(b)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))

    def test_card_db_id_changes_tokens_and_can_change_logits(self) -> None:
        obs = observation()
        actions = legal_actions()
        changed = shift_card_db_for_arena(obs, 1, 1)
        changed_actions = shift_card_db_for_arena(actions, 1, 1)
        a = encode_decision(obs, actions)
        b = encode_decision(changed, changed_actions)
        self.assertFalse(torch.equal(a.object_card_ids, b.object_card_ids))
        model = KernelPolicyValueNet.from_encoded(a)
        self.assertFalse(torch.equal(model(a)[0], model(b)[0]))

    def test_representative_model_leaf_mutations_affect_encoding(self) -> None:
        base_obs = complete_observation()
        base_actions = complete_legal_actions()
        base_digest = encoded_digest(encode_decision(base_obs, base_actions))
        mutations = []
        changed = deep_copy(base_obs)
        changed["projection"]["stack"][0]["controller"] = "p0"
        mutations.append((changed, base_actions, "stack controller"))
        changed = deep_copy(base_obs)
        changed["projection"]["stack"][0]["targets"][1]["object"] = changed["projection"]["battlefield"][0][0]["stable"]
        mutations.append((changed, base_actions, "stack object target identity"))
        changed = deep_copy(base_obs)
        changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1] = list(reversed(changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1]))
        mutations.append((changed, base_actions, "combat blocker ordering"))
        changed = deep_copy(base_obs)
        changed["projection"]["surface_context"]["stack_length_changed_since_observed"] = False
        mutations.append((changed, base_actions, "surface stack length observed flag"))
        changed = deep_copy(base_obs)
        changed["projection"]["surface_context"]["mana_activity_since_last_stack_change"] = True
        mutations.append((changed, base_actions, "surface mana since stack change flag"))
        changed = deep_copy(base_obs)
        changed["projection"]["exile_play_permissions"][0]["expiry"]["holder_turn_started"] = True
        mutations.append((changed, base_actions, "permission expiry"))
        changed = deep_copy(base_obs)
        changed["projection"]["continuous_effects"][0]["duration"] = "end_of_turn"
        changed["projection"]["continuous_effects"][0]["layers"] = 6
        mutations.append((changed, base_actions, "continuous effect layer"))
        changed = deep_copy(base_obs)
        changed["projection"]["battlefield"][0][0]["attachments"] = [changed["projection"]["battlefield"][0][1]["stable"]["arena_id"]]
        mutations.append((changed, base_actions, "attachment relationship"))
        changed_actions = deep_copy(base_actions)
        changed_actions[4]["semantic"]["order"] = [0, 1]
        mutations.append((base_obs, changed_actions, "trigger order values"))
        for obs, actions, label in mutations:
            self.assertNotEqual(base_digest, encoded_digest(encode_decision(obs, actions)), label)

    def test_arena_id_renumbering_does_not_change_features_or_logits(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        mapping = {1: 101, 2: 17, 3: 444, 4: 29, 5: 301, 6: 8, 7: 700, 8: 55, 9: 3, 10: 999, 12: 42}
        a = encode_decision(obs, actions)
        b = encode_decision(remap_all_arena_references(obs, mapping), remap_all_arena_references(actions, mapping))
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        self.assertTrue(torch.equal(model(a)[0], model(b)[0]))

    def test_generation_offset_invariant_and_mismatches_fail(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        a = encode_decision(obs, actions)
        b = encode_decision(shift_zone_generations(obs, 17), shift_zone_generations(actions, 17))
        assert_encoded_equal(self, a, b)
        mismatched_permission = deep_copy(obs)
        mismatched_permission["projection"]["exile_play_permissions"][0]["zone_change_generation"] += 1
        with self.assertRaises(FeatureSchemaError):
            encode_decision(mismatched_permission, actions)
        mismatched_action = deep_copy(actions)
        mismatched_action[1]["semantic"]["source"]["zone_change_count"] += 1
        with self.assertRaises(FeatureSchemaError):
            encode_decision(obs, mismatched_action)

    def test_same_card_target_actions_join_distinct_nodes_and_permute_logits(self) -> None:
        obs = observation()
        first = public_card(20, 21, "p1")
        second = public_card(21, 21, "p1")
        first["tapped"] = True
        first["damage"] = 2
        first["counters"]["plus1_plus1"] = 1
        obs["projection"]["battlefield"][1] = [first, second]
        obs["projection"]["combat"]["attacker_to_ordered_blockers"] = [[obs["projection"]["battlefield"][0][0]["stable"], [first["stable"]]]]
        src = obs["own_hand"][0]["stable"]
        actions = [
            {"schema_version": 2, "selected_index": 0, "stable_id": "t0", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "object", "object": first["stable"]}}, "display_text": None},
            {"schema_version": 2, "selected_index": 1, "stable_id": "t1", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "object", "object": second["stable"]}}, "display_text": None},
        ]
        encoded = encode_decision(obs, actions)
        self.assertNotEqual(action_representation(encoded, 0), action_representation(encoded, 1))
        model = KernelPolicyValueNet.from_encoded(encoded)
        logits, _ = model(encoded)
        self.assertNotEqual(float(logits[0].detach()), float(logits[1].detach()))
        swapped = [deep_copy(actions[1]), deep_copy(actions[0])]
        for i, action in enumerate(swapped):
            action["selected_index"] = i
        swapped_encoded = encode_decision(obs, swapped)
        swapped_logits, _ = model(swapped_encoded)
        self.assertTrue(torch.equal(logits[[1, 0]], swapped_logits))

    def test_attachment_topology_changes_and_arena_bijection_does_not(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        moved = deep_copy(obs)
        attachment_id = moved["projection"]["battlefield"][0][2]["stable"]["arena_id"]
        moved["projection"]["battlefield"][0][0]["attachments"] = []
        moved["projection"]["battlefield"][0][1]["attachments"] = [attachment_id]
        self.assertNotEqual(encoded_digest(encode_decision(obs, actions)), encoded_digest(encode_decision(moved, actions)))
        mapping = {1: 1001, 2: 22, 3: 333, 4: 4444, 5: 77, 6: 18, 7: 909, 8: 64, 9: 120, 10: 2, 12: 88}
        assert_encoded_equal(self, encode_decision(obs, actions), encode_decision(remap_all_arena_references(obs, mapping), remap_all_arena_references(actions, mapping)))

    def test_order_boundaries_match_contract(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        base_digest = encoded_digest(encode_decision(obs, actions))
        changed = deep_copy(obs)
        changed["projection"]["stack"].append(
            {
                "stack_index": 1,
                "source": stable_ref(14, 30, "p0", "Stack"),
                "controller": "p0",
                "targets": [{"target_kind": "player", "player": "p1"}],
                "stack_item_kind": "spell",
                "is_flashback": False,
                "mode_chosen": 0,
                "madness_offer": False,
                "kicked": False,
            }
        )
        reversed_stack = deep_copy(changed)
        reversed_stack["projection"]["stack"] = list(reversed(reversed_stack["projection"]["stack"]))
        for i, item in enumerate(reversed_stack["projection"]["stack"]):
            item["stack_index"] = i
        self.assertNotEqual(encoded_digest(encode_decision(changed, actions)), encoded_digest(encode_decision(reversed_stack, actions)))
        changed = deep_copy(obs)
        changed["projection"]["stack"][0]["targets"] = list(reversed(changed["projection"]["stack"][0]["targets"]))
        self.assertNotEqual(base_digest, encoded_digest(encode_decision(changed, actions)))
        changed = deep_copy(obs)
        changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1] = list(reversed(changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1]))
        self.assertNotEqual(base_digest, encoded_digest(encode_decision(changed, actions)))
        trigger_obs = deep_copy(obs)
        trigger_obs["projection"]["engine_context"]["current_stage"] = "pending_triggers"
        trigger_obs["projection"]["engine_context"]["pending_triggers"] = [
            {"source": trigger_obs["projection"]["battlefield"][0][0]["stable"], "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
            {"source": trigger_obs["projection"]["battlefield"][1][0]["stable"], "controller": "p1", "trigger_kind": "madness_offer", "kicked": True},
        ]
        reversed_triggers = deep_copy(trigger_obs)
        reversed_triggers["projection"]["engine_context"]["pending_triggers"] = list(reversed(reversed_triggers["projection"]["engine_context"]["pending_triggers"]))
        self.assertNotEqual(encoded_digest(encode_decision(trigger_obs, actions)), encoded_digest(encode_decision(reversed_triggers, actions)))

    def test_plotted_turn_is_relative_to_current_turn(self) -> None:
        obs = observation()
        actions = legal_actions()
        current = deep_copy(obs)
        current["projection"]["exile"][0]["plotted_turn"] = current["projection"]["turn"]
        shifted = deep_copy(current)
        shifted["projection"]["turn"] += 9
        shifted["projection"]["exile"][0]["plotted_turn"] += 9
        assert_encoded_equal(self, encode_decision(current, actions), encode_decision(shifted, actions))
        prior = deep_copy(current)
        prior["projection"]["exile"][0]["plotted_turn"] = prior["projection"]["turn"] - 1
        none = deep_copy(current)
        none["projection"]["exile"][0]["plotted_turn"] = None
        self.assertNotEqual(encoded_digest(encode_decision(current, actions)), encoded_digest(encode_decision(prior, actions)))
        self.assertNotEqual(encoded_digest(encode_decision(current, actions)), encoded_digest(encode_decision(none, actions)))

    def test_multiplicity_unresolved_and_inconsistent_refs_fail_closed(self) -> None:
        obs = observation()
        actions = legal_actions()
        one = encode_decision(obs, actions)
        two_obs = deep_copy(obs)
        duplicate = public_card(30, two_obs["projection"]["battlefield"][1][0]["stable"]["card_db_id"], "p1")
        two_obs["projection"]["battlefield"][1].append(duplicate)
        self.assertNotEqual(encoded_digest(one), encoded_digest(encode_decision(two_obs, actions)))
        unresolved = deep_copy(actions)
        unresolved[1]["semantic"]["source"] = stable_ref(999, 30, "p0", "Hand")
        with self.assertRaises(FeatureSchemaError):
            encode_decision(obs, unresolved)
        inconsistent = deep_copy(actions)
        inconsistent[1]["semantic"]["source"]["card_db_id"] += 1
        with self.assertRaises(FeatureSchemaError):
            encode_decision(obs, inconsistent)

    def test_legal_action_permutation_only_permutes_action_rows_and_logits(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        perm = [2, 0, 4, 1, 3]
        permuted = [copy.deepcopy(actions[i]) for i in perm]
        for idx, action in enumerate(permuted):
            action["selected_index"] = idx
        a = encode_decision(obs, actions)
        b = encode_decision(obs, permuted)
        self.assertTrue(torch.equal(a.state, b.state))
        self.assertTrue(torch.equal(a.object_features, b.object_features))
        self.assertTrue(torch.equal(a.action_features[perm], b.action_features))
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, _ = model(a)
        logits_b, _ = model(b)
        self.assertTrue(torch.equal(logits_a[perm], logits_b))

    def test_actor_relative_seat_swap_encodes_identically(self) -> None:
        obs = observation()
        actions = legal_actions()
        swapped_obs, swapped_actions = seat_swapped(obs, actions)
        a = encode_decision(obs, actions)
        b = encode_decision(swapped_obs, swapped_actions)
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, value_a = model(a)
        logits_b, value_b = model(b)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))

    def test_recorded_zone_order_changes_representation(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        a = encode_decision(obs, actions)
        changed = deep_copy(obs)
        changed["own_hand"] = list(reversed(changed["own_hand"]))
        self.assertNotEqual(encoded_digest(a), encoded_digest(encode_decision(changed, actions)))
        changed = deep_copy(obs)
        changed["projection"]["battlefield"][0] = list(reversed(changed["projection"]["battlefield"][0]))
        self.assertNotEqual(encoded_digest(a), encoded_digest(encode_decision(changed, actions)))
        changed = deep_copy(obs)
        changed["projection"]["graveyards"][0].append(deep_copy(changed["projection"]["graveyards"][0][0]))
        changed["projection"]["graveyards"][0][-1]["stable"] = stable_ref(13, 32, "p0", "Graveyard")
        reversed_changed = deep_copy(changed)
        reversed_changed["projection"]["graveyards"][0] = list(reversed(reversed_changed["projection"]["graveyards"][0]))
        self.assertNotEqual(encoded_digest(encode_decision(changed, actions)), encoded_digest(encode_decision(reversed_changed, actions)))

    def test_set_like_collections_are_canonicalized(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        permuted = deep_copy(obs)
        permuted["projection"]["continuous_effects"][0]["affected_objects"] = list(reversed(permuted["projection"]["continuous_effects"][0]["affected_objects"]))
        permuted_actions = deep_copy(actions)
        permuted_actions[3]["semantic"]["cards"] = list(reversed(permuted_actions[3]["semantic"]["cards"]))
        a = encode_decision(obs, actions)
        b = encode_decision(permuted, permuted_actions)
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, value_a = model(a)
        logits_b, value_b = model(b)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))

    def test_distinct_action_semantics_do_not_collide(self) -> None:
        obs = complete_observation()
        actions = every_action_variant_fixture(obs["own_hand"][0]["stable"], obs["projection"]["battlefield"][1][0]["stable"], obs["own_hand"][1]["stable"])
        encoded = encode_decision(obs, actions)
        reps = [action_representation(encoded, i) for i in range(len(actions))]
        self.assertEqual(len(reps), len(set(reps)))

    def test_adversarial_action_role_and_order_collisions_are_distinguishable(self) -> None:
        obs = complete_observation()
        src = stable_ref(1, 30, "p0", "Hand")
        other = stable_ref(12, 31, "p0", "Hand")
        base = {"schema_version": 2, "selected_index": 0, "stable_id": "x", "display_text": None}
        pairs = [
            (
                {**base, "semantic": {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [src, other], "order": [0, 1]}},
                {**base, "semantic": {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [src, other], "order": [1, 0]}},
            ),
            (
                {**base, "semantic": {"action_kind": "choose_cost_target", "actor": "p0", "source": src, "cost_kind": "SacrificeLands", "remaining": 1, "candidate": other}},
                {**base, "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "object", "object": other}}},
            ),
            (
                {**base, "semantic": {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": src, "blockers": [other]}},
                {**base, "semantic": {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": other, "blockers": [src]}},
            ),
        ]
        for left, right in pairs:
            left["selected_index"] = 0
            right["selected_index"] = 0
            a = encode_decision(obs, [left])
            b = encode_decision(obs, [right])
            self.assertNotEqual(action_representation(a, 0), action_representation(b, 0))


if __name__ == "__main__":
    unittest.main()
