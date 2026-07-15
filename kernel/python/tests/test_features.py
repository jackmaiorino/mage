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

from fixtures import complete_legal_actions, complete_observation, deep_copy, legal_actions, observation, stable_ref


def assert_encoded_equal(test: unittest.TestCase, a, b) -> None:
    test.assertTrue(torch.equal(a.state, b.state))
    test.assertTrue(torch.equal(a.object_features, b.object_features))
    test.assertTrue(torch.equal(a.object_card_ids, b.object_card_ids))
    test.assertTrue(torch.equal(a.object_groups, b.object_groups))
    test.assertTrue(torch.equal(a.action_features, b.action_features))
    test.assertTrue(torch.equal(a.action_ref_features, b.action_ref_features))
    test.assertTrue(torch.equal(a.action_ref_card_ids, b.action_ref_card_ids))
    test.assertTrue(torch.equal(a.action_ref_action_indices, b.action_ref_action_indices))


def encoded_digest(encoded) -> str:
    h = hashlib.sha256()
    for tensor in (
        encoded.state,
        encoded.object_features,
        encoded.object_card_ids,
        encoded.object_groups,
        encoded.action_features,
        encoded.action_ref_features,
        encoded.action_ref_card_ids,
        encoded.action_ref_action_indices,
    ):
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def action_representation(encoded, index: int) -> bytes:
    h = hashlib.sha256()
    h.update(encoded.action_features[index].detach().cpu().contiguous().numpy().tobytes())
    mask = encoded.action_ref_action_indices == index
    h.update(encoded.action_ref_features[mask].detach().cpu().contiguous().numpy().tobytes())
    h.update(encoded.action_ref_card_ids[mask].detach().cpu().contiguous().numpy().tobytes())
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
        actions = every_action_variant_fixture(stable_ref(99, 33, "p0", "Hand"))
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
        changed = deep_copy(obs)
        changed["own_hand"][0]["stable"]["card_db_id"] += 1
        a = encode_decision(obs, actions)
        b = encode_decision(changed, actions)
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
        changed["projection"]["stack"][0]["targets"][1]["object"]["card_db_id"] += 5
        mutations.append((changed, base_actions, "stack object target identity"))
        changed = deep_copy(base_obs)
        changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1] = list(reversed(changed["projection"]["combat"]["attacker_to_ordered_blockers"][0][1]))
        mutations.append((changed, base_actions, "combat blocker ordering"))
        changed = deep_copy(base_obs)
        changed["projection"]["engine_context"]["pending_cast"]["source"]["card_db_id"] += 7
        mutations.append((changed, base_actions, "pending cast source"))
        changed = deep_copy(base_obs)
        changed["projection"]["surface_context"]["private_discard"]["remaining_needed"] = 2
        mutations.append((changed, base_actions, "private discard count"))
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
        a = encode_decision(obs, actions)
        b = encode_decision(renumber_all_arena_references(obs), renumber_all_arena_references(actions))
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        self.assertTrue(torch.equal(model(a)[0], model(b)[0]))

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

    def test_unordered_object_permutation_keeps_pooled_model_output_identical(self) -> None:
        obs = complete_observation()
        actions = complete_legal_actions()
        permuted = deep_copy(obs)
        permuted["projection"]["battlefield"][0] = list(reversed(permuted["projection"]["battlefield"][0]))
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
        actions = every_action_variant_fixture(stable_ref(20, 30, "p0", "Hand"))
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
