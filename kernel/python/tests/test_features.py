from __future__ import annotations

import copy
import hashlib
import unittest

import torch

from mtg_kernel_rl.features import (
    ACTION_KINDS,
    COST_KINDS,
    EDGE_ROLES,
    MANA_COLORS,
    OBJECT_GROUPS,
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


def reverse_object_keys(value):
    if isinstance(value, dict):
        return {key: reverse_object_keys(value[key]) for key in reversed(list(value.keys()))}
    if isinstance(value, list):
        return [reverse_object_keys(v) for v in value]
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
    for key in ("known_library_cards", "known_hand_cards"):
        swapped_obs[key] = [swapped_obs[key][1], swapped_obs[key][0]]
    for effect in p["continuous_effects"]:
        effect["affected_players"] = sorted(effect["affected_players"], key=("p0", "p1").index)
    for zone in p["battlefield"] + p["graveyards"] + [p["exile"]]:
        for card in zone:
            card["goaded_by"] = sorted(card["goaded_by"], key=lambda item: ("p0", "p1").index(item["player"]))
    swapped_actions = swap_seats_value(deep_copy(actions))
    return swapped_obs, swapped_actions


def pending_effect_target_decision(can_finish: bool = True) -> tuple[dict, list[dict]]:
    obs = complete_observation()
    projection = obs["projection"]
    source = deep_copy(projection["stack"][-1]["source"])
    first = {
        "target_kind": "object",
        "object": deep_copy(obs["known_library_cards"][0][0]["card"]["stable"]),
    }
    second = {
        "target_kind": "object",
        "object": deep_copy(obs["known_library_cards"][1][0]["card"]["stable"]),
    }
    selected_targets = [first] if can_finish else []
    legal_targets = [second] if can_finish else [first, second]
    choice = {
        "choice_kind": "targets",
        "player": "p0",
        "structural_path": [1, 0],
        "selected_targets": selected_targets,
        "legal_targets": legal_targets,
        "min_targets": 1,
        "max_targets": 2,
        "can_finish": can_finish,
        "ordered": True,
        "purpose": "library_order",
    }
    projection["engine_context"].update(
        {
            "current_stage": "pending_effect",
            "pending_effect": {
                "source": source,
                "controller": source["controller"],
                "choice": choice,
            },
        }
    )

    selected_count = len(selected_targets)
    actions = [
        {
            "schema_version": 4,
            "selected_index": i,
            "stable_id": f"legal-action-v4:pending-target-{i}",
            "semantic": {
                "action_kind": "choose_effect_target",
                "actor": "p0",
                "source": deep_copy(source),
                "target": deep_copy(target),
                "selected_count": selected_count,
                "min_targets": 1,
                "max_targets": 2,
            },
            "display_text": None,
        }
        for i, target in enumerate(legal_targets)
    ]
    if can_finish:
        actions.append(
            {
                "schema_version": 4,
                "selected_index": len(actions),
                "stable_id": "legal-action-v4:pending-finish",
                "semantic": {
                    "action_kind": "finish_effect_selection",
                    "actor": "p0",
                    "source": deep_copy(source),
                    "selected_count": selected_count,
                },
                "display_text": None,
            }
        )
    return obs, actions


def pending_effect_option_decision(option_count: int = 2) -> tuple[dict, list[dict]]:
    obs = complete_observation()
    projection = obs["projection"]
    source = deep_copy(projection["stack"][-1]["source"])
    projection["engine_context"].update(
        {
            "current_stage": "pending_effect",
            "pending_effect": {
                "source": source,
                "controller": source["controller"],
                "choice": {
                    "choice_kind": "options",
                    "player": "p0",
                    "structural_path": [0],
                    "option_count": option_count,
                },
            },
        }
    )
    actions = [
        {
            "schema_version": 4,
            "selected_index": option_index,
            "stable_id": f"legal-action-v4:pending-option-{option_index}",
            "semantic": {
                "action_kind": "choose_effect_option",
                "actor": "p0",
                "source": deep_copy(source),
                "option_index": option_index,
                "option_count": option_count,
            },
            "display_text": None,
        }
        for option_index in range(option_count)
    ]
    return obs, actions


class FeatureEncodingTest(unittest.TestCase):
    def test_observation_and_all_action_variants_are_classified(self) -> None:
        obs = complete_observation()
        assert_observation_classified(obs)
        actions = every_action_variant_fixture(obs["own_hand"][0]["stable"], obs["projection"]["battlefield"][1][0]["stable"], obs["own_hand"][1]["stable"])
        for action in actions:
            assert_action_classified(action)
        validate_legal_actions_contract(actions, "p0")
        bad_prefix = deep_copy(actions)
        bad_prefix[0]["stable_id"] = "fixture-0"
        with self.assertRaises(FeatureSchemaError):
            validate_legal_actions_contract(bad_prefix, "p0")
        legacy_v3 = deep_copy(actions)
        legacy_v3[0]["schema_version"] = 3
        legacy_v3[0]["stable_id"] = "legal-action-v3:legacy"
        with self.assertRaises(FeatureSchemaError):
            validate_legal_actions_contract(legacy_v3, "p0")
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
        ambiguous = {"schema_version": 4, "selected_index": 0, "stable_id": "legal-action-v4:ambiguous", "display_text": None, "semantic": {"action_kind": "ambiguous", "reason": "text"}}
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

    def test_comprehensive_v4_contract_identity_and_dimensions_are_reserved(self) -> None:
        encoded = encode_decision(complete_observation(), complete_legal_actions())
        self.assertEqual(encoded.schema.version, "actor-relative-v4-python-3")
        self.assertEqual(encoded.schema.registry_version, "rust-observation-v4-action-v4-registry-3")
        self.assertEqual(encoded.schema.state_dim, 211)
        self.assertEqual(encoded.schema.object_feature_dim, 98)
        self.assertEqual(encoded.schema.edge_feature_dim, 41)
        self.assertEqual(encoded.schema.action_feature_dim, 195)
        self.assertEqual(encoded.schema.object_group_count, 20)
        self.assertEqual(encoded.schema.action_ref_feature_dim, 24)
        self.assertEqual(encoded.schema.contract_digest, "697174215cd8c8b04ede32e7a8bf8ef3f8fcd47ad2b4fc024cd0b6646bcbca01")
        self.assertEqual(encoded.schema.encoding_digest, "fa4e3db3845f2386cae233bf5538e43cf2b21c3cbdc3ca1f0d93b426ae22dab1")

    def test_detached_historical_paid_cost_refs_get_dedicated_nodes(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        current_land = obs["projection"]["battlefield"][0][1]["stable"]
        historical = obs["projection"]["stack"][0]["paid_cost_refs"][0]
        self.assertEqual(current_land["arena_id"], historical["arena_id"])
        self.assertNotEqual(current_land["zone_change_count"], historical["zone_change_count"])

        encoded = encode_decision(obs, actions)
        paid_group = OBJECT_GROUPS.index("paid_cost")
        paid_nodes = torch.nonzero(encoded.object_groups == paid_group, as_tuple=False).flatten()
        self.assertEqual(paid_nodes.numel(), 1)
        paid_node = int(paid_nodes.item())
        self.assertEqual(int(encoded.object_card_ids[paid_node]), historical["card_db_id"] + 1)
        paid_role = EDGE_ROLES.index("paid_cost")
        paid_edges = torch.nonzero(encoded.edge_features[:, paid_role] == 1.0, as_tuple=False).flatten()
        self.assertEqual(paid_edges.numel(), 1)
        self.assertEqual(int(encoded.edge_target_indices[int(paid_edges.item())]), paid_node)

        without_paid = deep_copy(obs)
        without_paid["projection"]["stack"][0]["paid_cost_refs"] = []
        without_encoded = encode_decision(without_paid, actions)
        self.assertEqual(encoded.object_features.shape[0], without_encoded.object_features.shape[0] + 1)
        self.assertNotEqual(encoded_digest(encoded), encoded_digest(without_encoded))

        mapping = {historical["arena_id"]: 303}
        assert_encoded_equal(
            self,
            encoded,
            encode_decision(remap_all_arena_references(obs, mapping), remap_all_arena_references(actions, mapping)),
        )
        changed_identity = deep_copy(obs)
        changed_identity["projection"]["stack"][0]["paid_cost_refs"][0]["card_db_id"] += 1
        self.assertNotEqual(encoded_digest(encoded), encoded_digest(encode_decision(changed_identity, actions)))
        changed_zone = deep_copy(obs)
        changed_zone["projection"]["stack"][0]["paid_cost_refs"][0]["zone"] = "Graveyard"
        self.assertNotEqual(encoded_digest(encoded), encoded_digest(encode_decision(changed_zone, actions)))

        duplicate = deep_copy(obs)
        duplicate["projection"]["stack"][0]["paid_cost_refs"].append(deep_copy(historical))
        conflicting = deep_copy(obs)
        conflict_ref = deep_copy(current_land)
        conflict_ref["card_db_id"] += 1
        conflicting["projection"]["stack"][0]["paid_cost_refs"] = [conflict_ref]
        for invalid in (duplicate, conflicting):
            with self.assertRaises(FeatureSchemaError):
                assert_observation_classified(invalid)

    def test_reserved_v4_observation_families_all_affect_encoding(self) -> None:
        base = complete_observation()
        actions = complete_legal_actions()
        base_digest = encoded_digest(encode_decision(base, actions))
        card_path = lambda obs: obs["projection"]["battlefield"][0][0]
        effect_path = lambda obs: obs["projection"]["continuous_effects"][0]
        mutations = []

        def add(label, mutate):
            changed = deep_copy(base)
            mutate(changed)
            mutations.append((label, changed))

        for counter in ("minus1_minus1", "minus0_minus1", "stun", "lore"):
            add(f"counter {counter}", lambda obs, key=counter: card_path(obs)["counters"].__setitem__(key, card_path(obs)["counters"][key] + 1))
        add("token", lambda obs: card_path(obs).__setitem__("is_token", False))
        add("face", lambda obs: card_path(obs).__setitem__("face_index", 2))
        add("chosen color", lambda obs: card_path(obs).__setitem__("chosen_color", "G"))
        add("entered turn", lambda obs: card_path(obs).__setitem__("entered_battlefield_turn", 1))
        add("ability kind", lambda obs: card_path(obs)["ability_uses_this_turn"][0].__setitem__("ability_kind", "activated"))
        add("ability index", lambda obs: card_path(obs)["ability_uses_this_turn"][0].__setitem__("ability_index", 2))
        add("ability uses", lambda obs: card_path(obs)["ability_uses_this_turn"][0].__setitem__("uses", 3))
        add("skip untap", lambda obs: card_path(obs).__setitem__("skip_next_untap", False))
        add("goad player", lambda obs: card_path(obs)["goaded_by"][0].__setitem__("player", "p0"))
        add("goad expiry", lambda obs: card_path(obs)["goaded_by"][0].__setitem__("expires_at_turn", 3))
        add("effective colors", lambda obs: card_path(obs)["characteristics"].__setitem__("effective_color_mask", 8))
        add("effective subtypes", lambda obs: card_path(obs)["characteristics"].__setitem__("effective_subtype_ids", [1, 18]))
        for keyword in ("lifelink", "hexproof", "indestructible", "protection_from_monocolored"):
            add(f"keyword {keyword}", lambda obs, key=keyword: card_path(obs)["characteristics"]["effective_keywords"].__setitem__(key, False))
        add("ward", lambda obs: card_path(obs)["characteristics"]["effective_keywords"].__setitem__("ward_generic", 3))
        add("minimum blockers", lambda obs: card_path(obs)["characteristics"]["effective_keywords"].__setitem__("minimum_blockers", 3))
        add("landwalk", lambda obs: card_path(obs)["characteristics"]["effective_keywords"].__setitem__("landwalk_mask", 4))
        add("stack cast method", lambda obs: obs["projection"]["stack"][0].__setitem__("cast_method", "normal"))
        add("stack face", lambda obs: obs["projection"]["stack"][0].__setitem__("face_index", 2))
        add("stack x", lambda obs: obs["projection"]["stack"][0].__setitem__("x_value", 4))
        add("stack paid refs", lambda obs: obs["projection"]["stack"][0].__setitem__("paid_cost_refs", [card_path(obs)["stable"]]))
        add("spells cast", lambda obs: obs["projection"]["player_status"][0].__setitem__("spells_cast_this_turn", 3))
        add("dungeon id", lambda obs: obs["projection"]["player_status"][0]["dungeon"].__setitem__("dungeon_id", 5))
        add("dungeon room", lambda obs: obs["projection"]["player_status"][0]["dungeon"].__setitem__("room_id", 5))
        add("completed dungeons", lambda obs: obs["projection"]["player_status"][0]["dungeon"].__setitem__("completed_dungeons", [1, 5]))
        add("initiative", lambda obs: obs["projection"].__setitem__("initiative", "p0"))
        effect_fields = {
            "source": lambda obs: card_path(obs)["stable"],
            "controller": lambda _obs: "p0",
            "affected_objects": lambda obs: [card_path(obs)["stable"]],
            "affected_players": lambda _obs: ["p0"],
            "global": lambda _obs: False,
            "duration": lambda _obs: "while_source_present",
            "set_power": lambda _obs: 5,
            "set_toughness": lambda _obs: 6,
            "add_color_mask": lambda _obs: 4,
            "remove_color_mask": lambda _obs: 1,
            "add_subtype_ids": lambda _obs: [2, 10],
            "remove_subtype_ids": lambda _obs: [3],
            "add_keyword_mask": lambda _obs: 18,
            "remove_keyword_mask": lambda _obs: 3,
            "ward_generic_delta": lambda _obs: 3,
            "minimum_blockers": lambda _obs: 4,
            "add_landwalk_mask": lambda _obs: 4,
            "remove_landwalk_mask": lambda _obs: 2,
            "prevent_damage_from_color_mask": lambda _obs: 8,
            "damage_cannot_be_prevented": lambda _obs: False,
        }
        for field, value in effect_fields.items():
            add(f"effect {field}", lambda obs, key=field, make=value: effect_path(obs).__setitem__(key, make(obs)))
        add("attached relation", lambda obs: obs["projection"]["object_relations"][0].__setitem__("attached_to", obs["projection"]["battlefield"][0][1]["stable"]))
        add("exile relation", lambda obs: obs["projection"]["object_relations"][1].__setitem__("exiled_by", obs["projection"]["battlefield"][0][1]["stable"]))
        add("known library", lambda obs: obs["known_library_cards"][0][0].__setitem__("position", 2))
        add("known hand", lambda obs: obs["known_hand_cards"][1][0]["stable"].__setitem__("card_db_id", 33))

        for label, changed in mutations:
            with self.subTest(label=label):
                self.assertNotEqual(base_digest, encoded_digest(encode_decision(changed, actions)))

    def test_typed_pending_effect_choices_and_reserved_actions_are_distinct(self) -> None:
        base = complete_observation()
        source = base["projection"]["stack"][0]["source"]
        target = base["projection"]["battlefield"][1][0]["stable"]
        choices = [
            {"choice_kind": "options", "player": "p0", "structural_path": [0], "option_count": 2},
            {"choice_kind": "targets", "player": "p0", "structural_path": [1], "selected_targets": [{"target_kind": "player", "player": "p1"}], "legal_targets": [{"target_kind": "object", "object": target}], "min_targets": 1, "max_targets": 2, "can_finish": True, "ordered": True, "purpose": "damage_division"},
            {"choice_kind": "color", "player": "p0", "structural_path": [2], "legal_colors": ["W", "R"]},
            {"choice_kind": "number", "player": "p0", "structural_path": [3], "minimum": -2, "maximum": 4},
            {"choice_kind": "boolean", "player": "p0", "structural_path": [4], "default": True, "purpose": "pay_cost"},
        ]
        for i, purpose in enumerate(("effect_targets", "card_selection", "permanent_selection", "player_selection", "cost_payment", "library_order", "search_result"), start=10):
            choices.append({"choice_kind": "targets", "player": "p0", "structural_path": [i], "selected_targets": [], "legal_targets": [{"target_kind": "object", "object": target}], "min_targets": 0, "max_targets": 1, "can_finish": True, "ordered": False, "purpose": purpose})
        for i, purpose in enumerate(("optional_effect", "shuffle"), start=30):
            choices.append({"choice_kind": "boolean", "player": "p0", "structural_path": [i], "default": None, "purpose": purpose})
        digests = []
        for choice in choices:
            obs = deep_copy(base)
            obs["projection"]["engine_context"].update(
                {
                    "current_stage": "pending_effect",
                    "pending_effect": {"source": source, "controller": "p0", "choice": choice},
                }
            )
            assert_observation_classified(obs)
            choice_actions = complete_legal_actions()
            if choice["choice_kind"] == "options":
                choice_actions = [
                    {
                        "schema_version": 4,
                        "selected_index": option_index,
                        "stable_id": f"legal-action-v4:typed-option-{option_index}",
                        "semantic": {
                            "action_kind": "choose_effect_option",
                            "actor": "p0",
                            "source": source,
                            "option_index": option_index,
                            "option_count": choice["option_count"],
                        },
                        "display_text": None,
                    }
                    for option_index in range(choice["option_count"])
                ]
            elif choice["choice_kind"] == "targets":
                selected_count = len(choice["selected_targets"])
                choice_actions = [
                    {
                        "schema_version": 4,
                        "selected_index": index,
                        "stable_id": f"legal-action-v4:typed-target-{index}",
                        "semantic": {
                            "action_kind": "choose_effect_target",
                            "actor": "p0",
                            "source": source,
                            "target": legal_target,
                            "selected_count": selected_count,
                            "min_targets": choice["min_targets"],
                            "max_targets": choice["max_targets"],
                        },
                        "display_text": None,
                    }
                    for index, legal_target in enumerate(choice["legal_targets"])
                ]
                if choice["can_finish"]:
                    choice_actions.append(
                        {
                            "schema_version": 4,
                            "selected_index": len(choice_actions),
                            "stable_id": "legal-action-v4:typed-finish",
                            "semantic": {
                                "action_kind": "finish_effect_selection",
                                "actor": "p0",
                                "source": source,
                                "selected_count": selected_count,
                            },
                            "display_text": None,
                        }
                    )
            digests.append(encoded_digest(encode_decision(obs, choice_actions)))
        self.assertEqual(len(digests), len(set(digests)))

        actions = every_action_variant_fixture(base["own_hand"][0]["stable"], target, base["own_hand"][1]["stable"])
        self.assertEqual(set(ACTION_KINDS), {action["semantic"]["action_kind"] for action in actions})
        encoded = encode_decision(base, actions)
        representations = [action_representation(encoded, i) for i in range(len(actions))]
        self.assertEqual(len(representations), len(set(representations)))

        land = base["projection"]["battlefield"][0][1]["stable"]
        reserved = []
        semantics = [
            {"action_kind": "activate_mana_ability", "actor": "p0", "source": land, "mana_choice": color}
            for color in [None, *MANA_COLORS]
        ] + [
            {"action_kind": "choose_cost_target", "actor": "p0", "source": base["own_hand"][0]["stable"], "cost_kind": cost, "remaining": 1, "candidate": land}
            for cost in COST_KINDS
        ]
        for i, semantic in enumerate(semantics):
            reserved.append({"schema_version": 4, "selected_index": i, "stable_id": f"legal-action-v4:reserved-{i}", "semantic": semantic, "display_text": None})
        encoded_reserved = encode_decision(base, reserved)
        reserved_representations = [action_representation(encoded_reserved, i) for i in range(len(reserved))]
        self.assertEqual(len(reserved_representations), len(set(reserved_representations)))

    def test_pending_effect_target_choice_and_actions_cross_validate(self) -> None:
        obs, actions = pending_effect_target_decision(can_finish=True)
        encoded = encode_decision(obs, actions)
        self.assertEqual(encoded.action_features.shape[0], 2)

        required_obs, required_actions = pending_effect_target_decision(can_finish=False)
        required_encoded = encode_decision(required_obs, required_actions)
        self.assertEqual(required_encoded.action_features.shape[0], 2)

    def test_pending_effect_target_observation_invariants_fail_closed(self) -> None:
        base, _ = pending_effect_target_decision(can_finish=True)
        choice_path = ("projection", "engine_context", "pending_effect", "choice")

        def choice(obs):
            value = obs
            for key in choice_path:
                value = value[key]
            return value

        wrong_actor = deep_copy(base)
        choice(wrong_actor)["player"] = "p1"
        wrong_source = deep_copy(base)
        wrong_source["projection"]["engine_context"]["pending_effect"]["source"] = deep_copy(
            wrong_source["own_hand"][0]["stable"]
        )
        wrong_source["projection"]["engine_context"]["pending_effect"]["source"]["zone"] = "Stack"
        duplicate_selected = deep_copy(base)
        choice(duplicate_selected)["selected_targets"].append(
            deep_copy(choice(duplicate_selected)["selected_targets"][0])
        )
        overlap = deep_copy(base)
        choice(overlap)["legal_targets"] = [deep_copy(choice(overlap)["selected_targets"][0])]
        unreachable_minimum = deep_copy(base)
        choice(unreachable_minimum).update(
            {"selected_targets": [], "legal_targets": [], "can_finish": False}
        )
        legal_at_maximum = deep_copy(base)
        choice(legal_at_maximum).update({"min_targets": 1, "max_targets": 1})
        wrong_can_finish = deep_copy(base)
        choice(wrong_can_finish)["can_finish"] = False

        cases = {
            "wrong chooser": wrong_actor,
            "wrong resolving source": wrong_source,
            "duplicate selected target": duplicate_selected,
            "selected/legal overlap": overlap,
            "unreachable minimum": unreachable_minimum,
            "legal target at maximum": legal_at_maximum,
            "wrong can_finish": wrong_can_finish,
        }
        for label, malformed in cases.items():
            with self.subTest(label=label), self.assertRaises(FeatureSchemaError):
                assert_observation_classified(malformed)

    def test_pending_effect_target_action_correspondence_fails_closed(self) -> None:
        obs, actions = pending_effect_target_decision(can_finish=True)
        pending_choice = obs["projection"]["engine_context"]["pending_effect"]["choice"]

        wrong_source = deep_copy(actions)
        wrong_source[0]["semantic"]["source"] = deep_copy(obs["own_hand"][0]["stable"])
        wrong_target = deep_copy(actions)
        wrong_target[0]["semantic"]["target"] = deep_copy(pending_choice["selected_targets"][0])
        wrong_count = deep_copy(actions)
        wrong_count[0]["semantic"]["selected_count"] = 0
        wrong_bounds = deep_copy(actions)
        wrong_bounds[0]["semantic"]["max_targets"] = 3
        missing_target = [deep_copy(actions[1])]
        missing_target[0]["selected_index"] = 0
        duplicate_target = deep_copy(actions)
        duplicate = deep_copy(duplicate_target[0])
        duplicate["stable_id"] = "legal-action-v4:pending-target-duplicate"
        duplicate_target.insert(1, duplicate)
        for index, action in enumerate(duplicate_target):
            action["selected_index"] = index
        missing_finish = [deep_copy(actions[0])]
        finish_first = [deep_copy(actions[1]), deep_copy(actions[0])]
        for index, action in enumerate(finish_first):
            action["selected_index"] = index

        malformed_sets = {
            "wrong source": wrong_source,
            "wrong target": wrong_target,
            "wrong selected count": wrong_count,
            "wrong bounds": wrong_bounds,
            "missing target": missing_target,
            "duplicate target": duplicate_target,
            "missing finish": missing_finish,
            "finish before targets": finish_first,
        }
        for label, malformed in malformed_sets.items():
            with self.subTest(label=label), self.assertRaises(FeatureSchemaError):
                encode_decision(obs, malformed)

        required_obs, required_actions = pending_effect_target_decision(can_finish=False)
        unexpected_finish = deep_copy(actions[1])
        unexpected_finish["selected_index"] = len(required_actions)
        unexpected_finish["semantic"]["selected_count"] = 0
        required_actions.append(unexpected_finish)
        with self.assertRaises(FeatureSchemaError):
            encode_decision(required_obs, required_actions)

    def test_pending_effect_option_action_correspondence_fails_closed(self) -> None:
        obs, actions = pending_effect_option_decision(option_count=2)

        wrong_source = deep_copy(actions)
        wrong_source[0]["semantic"]["source"] = deep_copy(obs["own_hand"][0]["stable"])
        wrong_count = deep_copy(actions)
        wrong_count[0]["semantic"]["option_count"] = 3
        wrong_order = list(reversed(deep_copy(actions)))
        for selected_index, action in enumerate(wrong_order):
            action["selected_index"] = selected_index
        missing_option = [deep_copy(actions[0])]
        unrelated_action = [deep_copy(action) for action in complete_legal_actions()[:2]]
        for selected_index, action in enumerate(unrelated_action):
            action["selected_index"] = selected_index

        malformed_sets = {
            "wrong source": wrong_source,
            "wrong option count": wrong_count,
            "wrong option order": wrong_order,
            "missing option": missing_option,
            "unrelated actions": unrelated_action,
        }
        for label, malformed in malformed_sets.items():
            with self.subTest(label=label), self.assertRaises(FeatureSchemaError):
                encode_decision(obs, malformed)

    def test_known_hand_perspective_safety_and_reserved_shape_errors_fail_closed(self) -> None:
        obs = complete_observation()
        assert_observation_classified(obs)
        self.assertEqual(obs["known_hand_cards"][0], [])
        swapped_obs, swapped_actions = seat_swapped(obs, complete_legal_actions())
        assert_encoded_equal(self, encode_decision(obs, complete_legal_actions()), encode_decision(swapped_obs, swapped_actions))

        own_leak = deep_copy(obs)
        own_leak["known_hand_cards"][0] = [deep_copy(own_leak["own_hand"][0])]
        wrong_owner = deep_copy(obs)
        wrong_owner["known_hand_cards"][1][0]["stable"]["owner"] = "p0"
        wrong_zone = deep_copy(obs)
        wrong_zone["known_hand_cards"][1][0]["stable"]["zone"] = "Library"
        bad_ability_order = deep_copy(obs)
        bad_ability_order["projection"]["battlefield"][0][0]["ability_uses_this_turn"] = list(reversed(bad_ability_order["projection"]["battlefield"][0][0]["ability_uses_this_turn"]))
        bad_subtypes = deep_copy(obs)
        bad_subtypes["projection"]["battlefield"][0][0]["characteristics"]["effective_subtype_ids"] = [17, 1]
        bad_mask = deep_copy(obs)
        bad_mask["projection"]["continuous_effects"][0]["prevent_damage_from_color_mask"] = 64
        bad_relation = deep_copy(obs)
        bad_relation["projection"]["object_relations"][0]["attached_to"] = stable_ref(999, 10, "p0", "Battlefield")
        for case in (own_leak, wrong_owner, wrong_zone, bad_ability_order, bad_subtypes, bad_mask, bad_relation):
            with self.assertRaises(FeatureSchemaError):
                assert_observation_classified(case)

        invalid_actions = every_action_variant_fixture(obs["own_hand"][0]["stable"], obs["projection"]["battlefield"][1][0]["stable"], obs["own_hand"][1]["stable"])
        by_kind = {action["semantic"]["action_kind"]: action for action in invalid_actions}
        bad_target = deep_copy(by_kind["choose_effect_target"])
        bad_target["semantic"]["selected_count"] = bad_target["semantic"]["max_targets"]
        bad_number = deep_copy(by_kind["choose_effect_number"])
        bad_number["semantic"]["number"] = bad_number["semantic"]["maximum"] + 1
        bad_mana = deep_copy(by_kind["activate_mana_ability"])
        bad_mana["semantic"]["mana_choice"] = "X"
        for action in (bad_target, bad_number, bad_mana):
            with self.assertRaises(FeatureSchemaError):
                assert_action_classified(action)

    def test_accumulated_blockers_follow_rust_blocker_attacker_tuple_order(self) -> None:
        blockers = complete_observation()
        blockers["projection"]["active_player"] = "p1"
        blocker = blockers["projection"]["battlefield"][0][0]["stable"]
        attacker = blockers["projection"]["battlefield"][1][0]["stable"]
        blockers["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {
                    "current_attacker": attacker,
                    "accumulated": [[blocker, attacker]],
                    "remaining": [],
                },
            }
        )

        assert_observation_classified(blockers)

        swapped = copy.deepcopy(blockers)
        swapped["projection"]["surface_context"]["private_blockers"]["accumulated"] = [[attacker, blocker]]
        with self.assertRaisesRegex(FeatureSchemaError, "impossible attacker/blocker ownership"):
            assert_observation_classified(swapped)

    def test_engine_surface_tuple_ladder_matches_rust_projection(self) -> None:
        base = complete_observation()
        p0_creature = base["projection"]["battlefield"][0][0]["stable"]
        p0_land = base["projection"]["battlefield"][0][1]["stable"]
        p1_creature = base["projection"]["battlefield"][1][0]["stable"]
        stack_source = base["projection"]["stack"][0]["source"]
        hand0 = base["own_hand"][0]["stable"]
        hand1 = base["own_hand"][1]["stable"]
        trigger_pair = [
            {"source": p0_creature, "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
            {"source": p0_land, "controller": "p0", "trigger_kind": "madness_offer", "kicked": True},
            {"source": p1_creature, "controller": "p1", "trigger_kind": "triggered_ability", "kicked": False},
        ]

        def pending_cast_payload():
            return {
                "source": stack_source,
                "controller": "p0",
                "chosen_targets": [{"target_kind": "object", "object": p1_creature}],
                "is_flashback": False,
                "cast_mode": "Alternative",
                "additional_cost_discarded": [hand1],
                "mode_chosen": 1,
                "origin_zone": "Hand",
                "sacrifice_chosen": [p0_land],
                "kicked": True,
            }

        def pending_activation_payload():
            return {"source": p0_land, "controller": "p0", "ability_index": 2, "chosen_targets": [{"target_kind": "player", "player": "p1"}], "cost_discard_paid": [hand1]}

        def pending_optional_payload(discard_payable=True, sacrifice_payable=True):
            return {
                "player": "p0",
                "source": stack_source,
                "discard_cards": 1,
                "sacrifice_lands": 1,
                "discard_payable": discard_payable,
                "sacrifice_payable": sacrifice_payable,
                "spell_resume_source": stack_source,
                "spell_resume_zone": "Graveyard",
            }

        def set_discard_surface(obs, chosen=None, remaining=None):
            obs["projection"]["surface_context"].update(
                {
                    "current_stage": "discard_pick",
                    "private_discard": {"chosen": [] if chosen is None else chosen, "remaining_choices": [hand0, hand1] if remaining is None else remaining, "remaining_needed": 1},
                }
            )

        valid = []
        valid.append(("normal priority", complete_observation()))
        blockers = complete_observation()
        blockers["projection"]["active_player"] = "p1"
        blockers["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {"current_attacker": p1_creature, "accumulated": [], "remaining": [[p1_creature, [p0_creature]]]},
            }
        )
        valid.append(("blockers priority", blockers))
        for stage, payload in (("pending_cast", pending_cast_payload()), ("pending_activation", pending_activation_payload())):
            obs = complete_observation()
            obs["projection"]["engine_context"]["current_stage"] = stage
            obs["projection"]["engine_context"][stage] = payload
            valid.append((f"{stage} priority", obs))
        paused_cast = complete_observation()
        paused_cast["projection"]["engine_context"]["current_stage"] = "pending_cast"
        paused_cast["projection"]["engine_context"]["pending_cast"] = pending_cast_payload()
        paused_cast["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "finish_cast", "resume_source": None}
        set_discard_surface(paused_cast)
        valid.append(("paused cast discard", paused_cast))
        paused_activation = complete_observation()
        paused_activation["projection"]["engine_context"]["current_stage"] = "pending_activation"
        paused_activation["projection"]["engine_context"]["pending_activation"] = pending_activation_payload()
        paused_activation["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "finish_activation", "resume_source": None}
        set_discard_surface(paused_activation)
        valid.append(("paused activation discard", paused_activation))
        detached_stack_source = stable_ref(68, 34, "p0", "Stack")
        for label, resume_stage, resume_source in (
            ("direct discard", "none", None),
            ("spell-resolution discard", "finish_spell_resolution", detached_stack_source),
            ("optional-cost discard", "finish_optional_cost", detached_stack_source),
        ):
            obs = complete_observation()
            obs["projection"]["engine_context"]["current_stage"] = "pending_discard"
            obs["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": resume_stage, "resume_source": resume_source}
            set_discard_surface(obs)
            valid.append((label, obs))
        discard_triggers = complete_observation()
        discard_triggers["projection"]["engine_context"]["current_stage"] = "pending_discard"
        discard_triggers["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "none", "resume_source": None}
        discard_triggers["projection"]["engine_context"]["pending_triggers"] = trigger_pair
        set_discard_surface(discard_triggers)
        valid.append(("discard plus queued triggers", discard_triggers))
        optional_use = complete_observation()
        optional_use["projection"]["engine_context"]["current_stage"] = "pending_optional_cost"
        optional_use["projection"]["engine_context"]["pending_optional_cost"] = pending_optional_payload(discard_payable=True, sacrifice_payable=False)
        optional_use["projection"]["surface_context"].update(
            {"current_stage": "optional_cost_use", "private_optional_cost": {"discard_payable": True, "sacrifice_payable": False, "stage": "optional_cost_use"}}
        )
        valid.append(("optional cost use", optional_use))
        optional_which = complete_observation()
        optional_which["projection"]["engine_context"]["current_stage"] = "pending_optional_cost"
        optional_which["projection"]["engine_context"]["pending_optional_cost"] = pending_optional_payload(discard_payable=True, sacrifice_payable=True)
        optional_which["projection"]["surface_context"].update(
            {"current_stage": "optional_cost_which", "private_optional_cost": {"discard_payable": True, "sacrifice_payable": True, "stage": "optional_cost_which"}}
        )
        valid.append(("optional cost which", optional_which))
        sacrifice = complete_observation()
        sacrifice["projection"]["engine_context"]["current_stage"] = "pending_optional_cost_sacrifice"
        sacrifice["projection"]["engine_context"]["pending_optional_cost_sacrifice"] = {
            "player": "p0",
            "source": stack_source,
            "remaining": 2,
            "chosen": [p0_land],
            "spell_resume_source": stack_source,
            "spell_resume_zone": "Graveyard",
        }
        valid.append(("sacrifice with priority", sacrifice))
        triggers = complete_observation()
        triggers["projection"]["engine_context"]["current_stage"] = "pending_triggers"
        triggers["projection"]["engine_context"]["pending_triggers"] = trigger_pair
        valid.append(("trigger order", triggers))
        halted = complete_observation()
        halted["projection"]["engine_context"]["current_stage"] = "halted"
        valid.append(("halted", halted))

        for label, obs in valid:
            with self.subTest(label=label):
                assert_observation_classified(obs)
                encode_decision(obs, complete_legal_actions())

    def test_spell_copy_state_machine_and_actions_are_encoded_fail_closed(self) -> None:
        def spell_copy_observation(stage: str) -> tuple[dict, dict, dict | None]:
            obs = complete_observation()
            obs["acting_player"] = "p1"
            obs["projection"]["priority_player"] = "p1"
            obs["known_hand_cards"] = [[], []]
            engine = obs["projection"]["engine_context"]
            engine["current_stage"] = "pending_spell_copy"
            parent = obs["projection"]["stack"][0]["source"]
            obs["projection"]["stack"][0]["targets"] = [
                {"target_kind": "player", "player": "p1"}
            ]
            copy_ref = None
            if stage != "payment":
                copy_ref = stable_ref(68, parent["card_db_id"], "p1", "Stack", controller="p1")
                obs["projection"]["stack"].append(
                    {
                        "stack_index": 1,
                        "source": copy_ref,
                        "controller": "p1",
                        "targets": [{"target_kind": "player", "player": "p1"}],
                        "stack_item_kind": "spell",
                        "is_copy": True,
                        "is_flashback": False,
                        "mode_chosen": 0,
                        "madness_offer": False,
                        "kicked": False,
                        "cast_method": "normal",
                        "face_index": 0,
                        "x_value": 0,
                        "paid_cost_refs": [],
                    }
                )
            engine["pending_spell_copy"] = {
                "parent": parent,
                "player": "p1",
                "inherited_target": {"target_kind": "player", "player": "p1"},
                "stage": stage,
                "copy": copy_ref,
            }
            return obs, parent, copy_ref

        payment, parent, _ = spell_copy_observation("payment")
        payment_actions = [
            {
                "schema_version": 4,
                "selected_index": i,
                "stable_id": f"legal-action-v4:copy-pay-{i}",
                "semantic": {"action_kind": "choose_spell_copy_payment", "actor": "p1", "source": parent, "pay": pay},
                "display_text": None,
            }
            for i, pay in enumerate((True, False))
        ]
        assert_observation_classified(payment)
        payment_encoded = encode_decision(payment, payment_actions)
        self.assertNotEqual(action_representation(payment_encoded, 0), action_representation(payment_encoded, 1))

        retarget, _, copy_ref = spell_copy_observation("retarget")
        assert copy_ref is not None
        retarget_actions = [
            {
                "schema_version": 4,
                "selected_index": i,
                "stable_id": f"legal-action-v4:copy-retarget-{i}",
                "semantic": {
                    "action_kind": "choose_spell_copy_retarget",
                    "actor": "p1",
                    "source": copy_ref,
                    "change_target": change_target,
                },
                "display_text": None,
            }
            for i, change_target in enumerate((True, False))
        ]
        assert_observation_classified(retarget)
        retarget_encoded = encode_decision(retarget, retarget_actions)
        self.assertNotEqual(action_representation(retarget_encoded, 0), action_representation(retarget_encoded, 1))

        with self.assertRaises(FeatureSchemaError):
            encode_decision(payment, retarget_actions)
        wrong_payment_source = deep_copy(payment_actions)
        wrong_payment_source[0]["semantic"]["source"] = copy_ref
        with self.assertRaises(FeatureSchemaError):
            encode_decision(payment, wrong_payment_source)
        wrong_payment_order = deep_copy(payment_actions)
        wrong_payment_order[0]["semantic"]["pay"] = False
        wrong_payment_order[1]["semantic"]["pay"] = True
        with self.assertRaises(FeatureSchemaError):
            encode_decision(payment, wrong_payment_order)

        target, _, target_copy = spell_copy_observation("target")
        assert target_copy is not None
        target_actions = [
            {
                "schema_version": 4,
                "selected_index": i,
                "stable_id": f"legal-action-v4:copy-target-{i}",
                "semantic": {
                    "action_kind": "choose_target",
                    "actor": "p1",
                    "source": target_copy,
                    "remaining": 1,
                    "target": {"target_kind": "player", "player": player},
                },
                "display_text": None,
            }
            for i, player in enumerate(("p0", "p1"))
        ]
        assert_observation_classified(target)
        encode_decision(target, target_actions)

        invalid_payment = deep_copy(payment)
        invalid_payment["projection"]["engine_context"]["pending_spell_copy"]["copy"] = parent
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_payment)

        invalid_missing_copy = deep_copy(retarget)
        invalid_missing_copy["projection"]["engine_context"]["pending_spell_copy"]["copy"] = None
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_missing_copy)

        invalid_copy_marker = deep_copy(retarget)
        invalid_copy_marker["projection"]["stack"][1]["is_copy"] = False
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_copy_marker)

        invalid_parent_target = deep_copy(payment)
        invalid_parent_target["projection"]["engine_context"]["pending_spell_copy"]["inherited_target"] = {
            "target_kind": "player",
            "player": "p0",
        }
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_parent_target)

        invalid_copy_target = deep_copy(retarget)
        invalid_copy_target["projection"]["stack"][1]["targets"] = [
            {"target_kind": "player", "player": "p0"}
        ]
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_copy_target)

        invalid_controller = deep_copy(retarget)
        invalid_controller["projection"]["stack"][1]["controller"] = "p0"
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_controller)

        invalid_actor = deep_copy(retarget)
        invalid_actor["acting_player"] = "p0"
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(invalid_actor)

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
        bad["projection"]["engine_context"]["current_stage"] = "pending_optional_cost"
        bad["projection"]["engine_context"]["pending_optional_cost"] = {
            "player": "p0",
            "source": bad["own_hand"][0]["stable"],
            "discard_cards": 1,
            "sacrifice_lands": 1,
            "discard_payable": True,
            "sacrifice_payable": True,
            "spell_resume_source": None,
            "spell_resume_zone": None,
        }
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
        bad = complete_observation()
        bad["projection"]["engine_context"]["current_stage"] = "pending_discard"
        bad["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "finish_cast", "resume_source": None}
        bad["projection"]["surface_context"].update(
            {"current_stage": "discard_pick", "private_discard": {"chosen": [], "remaining_choices": [bad["own_hand"][0]["stable"]], "remaining_needed": 1}}
        )
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)
        bad = complete_observation()
        bad["projection"]["engine_context"]["current_stage"] = "pending_optional_cost_sacrifice"
        bad["projection"]["engine_context"]["pending_optional_cost_sacrifice"] = {
            "player": "p0",
            "source": bad["own_hand"][0]["stable"],
            "remaining": 1,
            "chosen": [],
            "spell_resume_source": None,
            "spell_resume_zone": None,
        }
        bad["projection"]["surface_context"].update(
            {"current_stage": "optional_cost_use", "private_optional_cost": {"discard_payable": True, "sacrifice_payable": False, "stage": "optional_cost_use"}}
        )
        with self.assertRaises(FeatureSchemaError):
            assert_observation_classified(bad)

    def test_pending_context_decision_owner_rules_fail_closed(self) -> None:
        base = complete_observation()
        stack_source = base["projection"]["stack"][0]["source"]
        p0_land = base["projection"]["battlefield"][0][1]["stable"]
        p1_creature = base["projection"]["battlefield"][1][0]["stable"]
        hand0 = base["own_hand"][0]["stable"]

        def discard_surface(obs: dict) -> None:
            obs["projection"]["surface_context"].update(
                {"current_stage": "discard_pick", "private_discard": {"chosen": [], "remaining_choices": [hand0], "remaining_needed": 1}}
            )

        def optional_surface(obs: dict) -> None:
            obs["projection"]["surface_context"].update(
                {"current_stage": "optional_cost_use", "private_optional_cost": {"discard_payable": True, "sacrifice_payable": False, "stage": "optional_cost_use"}}
            )

        cases = []
        pending_cast = complete_observation()
        pending_cast["acting_player"] = "p1"
        pending_cast["projection"]["engine_context"]["current_stage"] = "pending_cast"
        pending_cast["projection"]["engine_context"]["pending_cast"] = {
            "source": stack_source,
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
        cases.append(("pending cast owner", pending_cast))

        pending_activation = complete_observation()
        pending_activation["acting_player"] = "p1"
        pending_activation["projection"]["engine_context"]["current_stage"] = "pending_activation"
        pending_activation["projection"]["engine_context"]["pending_activation"] = {
            "source": p0_land,
            "controller": "p0",
            "ability_index": 0,
            "chosen_targets": [],
            "cost_discard_paid": [],
        }
        cases.append(("pending activation owner", pending_activation))

        pending_discard = complete_observation()
        pending_discard["acting_player"] = "p1"
        pending_discard["projection"]["engine_context"]["current_stage"] = "pending_discard"
        pending_discard["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "none", "resume_source": None}
        discard_surface(pending_discard)
        cases.append(("pending discard owner", pending_discard))

        optional = complete_observation()
        optional["acting_player"] = "p1"
        optional["projection"]["engine_context"]["current_stage"] = "pending_optional_cost"
        optional["projection"]["engine_context"]["pending_optional_cost"] = {
            "player": "p0",
            "source": stack_source,
            "discard_cards": 1,
            "sacrifice_lands": 0,
            "discard_payable": True,
            "sacrifice_payable": False,
            "spell_resume_source": stack_source,
            "spell_resume_zone": "Graveyard",
        }
        optional_surface(optional)
        cases.append(("optional cost owner", optional))

        sacrifice = complete_observation()
        sacrifice["acting_player"] = "p1"
        sacrifice["projection"]["engine_context"]["current_stage"] = "pending_optional_cost_sacrifice"
        sacrifice["projection"]["engine_context"]["pending_optional_cost_sacrifice"] = {
            "player": "p0",
            "source": stack_source,
            "remaining": 1,
            "chosen": [],
            "spell_resume_source": stack_source,
            "spell_resume_zone": "Graveyard",
        }
        cases.append(("optional sacrifice owner", sacrifice))

        trigger_order = complete_observation()
        trigger_order["acting_player"] = "p1"
        trigger_order["projection"]["engine_context"]["current_stage"] = "pending_triggers"
        trigger_order["projection"]["engine_context"]["pending_triggers"] = [
            {"source": stack_source, "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
            {"source": p0_land, "controller": "p0", "trigger_kind": "madness_offer", "kicked": False},
        ]
        cases.append(("trigger order owner", trigger_order))

        trigger_singleton_group = complete_observation()
        trigger_singleton_group["projection"]["engine_context"]["current_stage"] = "pending_triggers"
        trigger_singleton_group["projection"]["engine_context"]["pending_triggers"] = [
            {"source": stack_source, "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
            {"source": p1_creature, "controller": "p1", "trigger_kind": "triggered_ability", "kicked": False},
        ]
        cases.append(("trigger singleton group", trigger_singleton_group))

        blockers = complete_observation()
        blockers["acting_player"] = "p1"
        blockers["projection"]["active_player"] = "p1"
        blockers["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {"current_attacker": p1_creature, "accumulated": [], "remaining": [[p1_creature, [base["projection"]["battlefield"][0][0]["stable"]]]]},
            }
        )
        cases.append(("blocker defender owner", blockers))

        for label, obs in cases:
            with self.subTest(label=label):
                with self.assertRaises(FeatureSchemaError):
                    assert_observation_classified(obs)

        assert_observation_classified(complete_observation())
        halted = complete_observation()
        halted["projection"]["engine_context"]["current_stage"] = "halted"
        assert_observation_classified(halted)

    def test_active_pending_source_shapes_fail_closed(self) -> None:
        base = complete_observation()
        stack_source = base["projection"]["stack"][0]["source"]
        battlefield_source = base["projection"]["battlefield"][0][1]["stable"]

        def cast_obs(source):
            obs = complete_observation()
            obs["projection"]["engine_context"]["current_stage"] = "pending_cast"
            obs["projection"]["engine_context"]["pending_cast"] = {
                "source": source,
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
            return obs

        def activation_obs(source):
            obs = complete_observation()
            obs["projection"]["engine_context"]["current_stage"] = "pending_activation"
            obs["projection"]["engine_context"]["pending_activation"] = {
                "source": source,
                "controller": "p0",
                "ability_index": 0,
                "chosen_targets": [],
                "cost_discard_paid": [],
            }
            return obs

        assert_observation_classified(cast_obs(stack_source))
        assert_observation_classified(activation_obs(battlefield_source))
        bad_cast_sources = [
            None,
            {**stack_source, "zone": "Hand"},
            {**stack_source, "controller": "p1"},
        ]
        bad_activation_sources = [
            None,
            {**battlefield_source, "zone": "Stack"},
            {**battlefield_source, "controller": "p1"},
        ]
        for source in bad_cast_sources:
            with self.subTest(kind="cast", source=source):
                with self.assertRaises(FeatureSchemaError):
                    assert_observation_classified(cast_obs(source))
        for source in bad_activation_sources:
            with self.subTest(kind="activation", source=source):
                with self.assertRaises(FeatureSchemaError):
                    assert_observation_classified(activation_obs(source))

    def test_detached_context_ref_allowlist_is_narrow(self) -> None:
        base = complete_observation()
        hand0 = base["own_hand"][0]["stable"]
        observed_stack = base["projection"]["stack"][0]["source"]
        detached_same_card_stack = stable_ref(68, observed_stack["card_db_id"], "p0", "Stack")

        admitted = complete_observation()
        admitted["projection"]["engine_context"]["current_stage"] = "pending_discard"
        admitted["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "finish_spell_resolution", "resume_source": detached_same_card_stack}
        admitted["projection"]["surface_context"].update(
            {"current_stage": "discard_pick", "private_discard": {"chosen": [], "remaining_choices": [hand0], "remaining_needed": 1}}
        )
        admitted_encoded = encode_decision(admitted, complete_legal_actions())

        observed_resume = deep_copy(admitted)
        observed_resume["projection"]["engine_context"]["pending_discard"]["resume_source"] = observed_stack
        observed_encoded = encode_decision(observed_resume, complete_legal_actions())
        self.assertNotEqual(encoded_digest(admitted_encoded), encoded_digest(observed_encoded))

        cases = []
        wrong_zone_resume = deep_copy(admitted)
        wrong_zone_resume["projection"]["engine_context"]["pending_discard"]["resume_source"] = stable_ref(69, observed_stack["card_db_id"], "p0", "Graveyard")
        cases.append(("wrong-zone resume source", wrong_zone_resume, complete_legal_actions()))

        optional_detached = complete_observation()
        optional_detached["projection"]["engine_context"]["current_stage"] = "pending_optional_cost"
        optional_detached["projection"]["engine_context"]["pending_optional_cost"] = {
            "player": "p0",
            "source": detached_same_card_stack,
            "discard_cards": 1,
            "sacrifice_lands": 0,
            "discard_payable": True,
            "sacrifice_payable": False,
            "spell_resume_source": detached_same_card_stack,
            "spell_resume_zone": "Graveyard",
        }
        optional_detached["projection"]["surface_context"].update(
            {"current_stage": "optional_cost_use", "private_optional_cost": {"discard_payable": True, "sacrifice_payable": False, "stage": "optional_cost_use"}}
        )
        encode_decision(optional_detached, complete_legal_actions())

        optional_sacrifice_unresolved_chosen = complete_observation()
        optional_sacrifice_unresolved_chosen["projection"]["engine_context"]["current_stage"] = "pending_optional_cost_sacrifice"
        optional_sacrifice_unresolved_chosen["projection"]["engine_context"]["pending_optional_cost_sacrifice"] = {
            "player": "p0",
            "source": detached_same_card_stack,
            "remaining": 2,
            "chosen": [stable_ref(98, hand0["card_db_id"], "p0", "Battlefield")],
            "spell_resume_source": detached_same_card_stack,
            "spell_resume_zone": "Graveyard",
        }
        cases.append(("unresolved optional sacrifice choice", optional_sacrifice_unresolved_chosen, complete_legal_actions()))

        private_discard = complete_observation()
        private_discard["projection"]["engine_context"]["current_stage"] = "pending_discard"
        private_discard["projection"]["engine_context"]["pending_discard"] = {"player": "p0", "count": 1, "resume_stage": "none", "resume_source": None}
        private_discard["projection"]["surface_context"].update(
            {
                "current_stage": "discard_pick",
                "private_discard": {"chosen": [], "remaining_choices": [stable_ref(99, hand0["card_db_id"], "p0", "Hand")], "remaining_needed": 1},
            }
        )
        cases.append(("unresolved private discard choice", private_discard, complete_legal_actions()))

        private_blocker = complete_observation()
        private_blocker["projection"]["active_player"] = "p1"
        p1_attacker = private_blocker["projection"]["battlefield"][1][0]["stable"]
        private_blocker["projection"]["surface_context"].update(
            {
                "current_stage": "declare_blockers_for_attacker",
                "private_blockers": {"current_attacker": p1_attacker, "accumulated": [], "remaining": [[p1_attacker, [stable_ref(100, hand0["card_db_id"], "p0", "Battlefield")]]]},
            }
        )
        cases.append(("unresolved private blocker", private_blocker, complete_legal_actions()))

        pending_target = complete_observation()
        pending_target["projection"]["engine_context"]["current_stage"] = "pending_cast"
        pending_target["projection"]["engine_context"]["pending_cast"] = {
            "source": observed_stack,
            "controller": "p0",
            "chosen_targets": [{"target_kind": "object", "object": stable_ref(101, hand0["card_db_id"], "p1", "Battlefield")}],
            "is_flashback": False,
            "cast_mode": None,
            "additional_cost_discarded": None,
            "mode_chosen": None,
            "origin_zone": "Hand",
            "sacrifice_chosen": [],
            "kicked": None,
        }
        cases.append(("unresolved pending target", pending_target, complete_legal_actions()))

        unresolved_action_target = complete_observation()
        bad_actions = complete_legal_actions()
        bad_actions[1]["semantic"]["target"]["object"] = stable_ref(102, hand0["card_db_id"], "p1", "Battlefield")
        cases.append(("unresolved action target", unresolved_action_target, bad_actions))

        for label, obs, actions in cases:
            with self.subTest(label=label):
                with self.assertRaises(FeatureSchemaError):
                    encode_decision(obs, actions)

    def test_names_display_stable_ids_hashes_do_not_change_features_or_logits(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        mutated_obs = deep_copy(obs)
        mutated_actions = deep_copy(actions)
        mutated_obs["visible_projection_hash"] = 999
        mutated_obs["own_hand"][0]["card_name"] = "Different"
        mutated_obs["projection"]["battlefield"][0][0]["card_name"] = "Renamed"
        mutated_obs["projection"]["continuous_effects"][0]["timestamp"] = 777
        mutated_actions[0]["stable_id"] = "legal-action-v4:changed"
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
            {"schema_version": 4, "selected_index": 0, "stable_id": "legal-action-v4:t0", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "object", "object": first["stable"]}}, "display_text": None},
            {"schema_version": 4, "selected_index": 1, "stable_id": "legal-action-v4:t1", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "object", "object": second["stable"]}}, "display_text": None},
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
                "is_copy": False,
                "is_flashback": False,
                "mode_chosen": 0,
                "madness_offer": False,
                "kicked": False,
                "cast_method": "normal",
                "face_index": 0,
                "x_value": 0,
                "paid_cost_refs": [],
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
            {"source": trigger_obs["projection"]["battlefield"][0][1]["stable"], "controller": "p0", "trigger_kind": "madness_offer", "kicked": True},
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

    def test_actor_swap_with_attachment_edges_on_both_seats_is_exact(self) -> None:
        obs = complete_observation()
        p1_attachment = public_card(13, 41, "p1")
        obs["projection"]["battlefield"][1].append(p1_attachment)
        obs["projection"]["battlefield"][1][0]["attachments"] = [p1_attachment["stable"]["arena_id"]]
        actions = complete_legal_actions()
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

    def test_json_object_key_permutation_invariant_and_named_context_roles_sensitive(self) -> None:
        obs = complete_observation()
        first = obs["projection"]["stack"][0]["source"]
        second_stack = stable_ref(68, first["card_db_id"], "p0", "Stack")
        obs["projection"]["stack"].append(
            {
                "stack_index": 1,
                "source": second_stack,
                "controller": "p0",
                "targets": [],
                "stack_item_kind": "spell",
                "is_copy": False,
                "is_flashback": False,
                "mode_chosen": 0,
                "madness_offer": False,
                "kicked": False,
                "cast_method": "normal",
                "face_index": 0,
                "x_value": 0,
                "paid_cost_refs": [],
            }
        )
        second = obs["own_hand"][1]["stable"]
        second["card_db_id"] = first["card_db_id"]
        obs["projection"]["engine_context"]["current_stage"] = "pending_cast"
        obs["projection"]["engine_context"]["pending_cast"] = {
            "source": first,
            "controller": "p0",
            "chosen_targets": [],
            "is_flashback": False,
            "cast_mode": None,
            "additional_cost_discarded": [second],
            "mode_chosen": None,
            "origin_zone": "Hand",
            "sacrifice_chosen": [],
            "kicked": None,
        }
        actions = shift_card_db_for_arena(complete_legal_actions(), second["arena_id"], first["card_db_id"] - 31)
        permuted = reverse_object_keys(obs)
        permuted_actions = reverse_object_keys(actions)
        a = encode_decision(obs, actions)
        b = encode_decision(permuted, permuted_actions)
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        self.assertTrue(torch.equal(model(a)[0], model(b)[0]))

        same_card_other_stack_copy = deep_copy(obs)
        same_card_other_stack_copy["projection"]["engine_context"]["pending_cast"]["source"] = second_stack
        self.assertNotEqual(encoded_digest(a), encoded_digest(encode_decision(same_card_other_stack_copy, actions)))

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
        base = {"schema_version": 4, "selected_index": 0, "stable_id": "legal-action-v4:x", "display_text": None}
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
