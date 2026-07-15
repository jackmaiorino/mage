from __future__ import annotations

import copy
import unittest

import torch

from mtg_kernel_rl.features import (
    FeatureSchemaError,
    assert_action_classified,
    assert_observation_classified,
    encode_decision,
    every_action_variant_fixture,
)
from mtg_kernel_rl.model import KernelPolicyValueNet

from fixtures import deep_copy, legal_actions, observation, stable_ref


def assert_encoded_equal(test: unittest.TestCase, a, b) -> None:
    test.assertTrue(torch.equal(a.state, b.state))
    test.assertTrue(torch.equal(a.object_features, b.object_features))
    test.assertTrue(torch.equal(a.object_card_ids, b.object_card_ids))
    test.assertTrue(torch.equal(a.object_groups, b.object_groups))
    test.assertTrue(torch.equal(a.action_features, b.action_features))
    test.assertTrue(torch.equal(a.action_card_ids, b.action_card_ids))


def renumber_arena_ids(value, delta: int = 1000):
    if isinstance(value, dict):
        return {k: (v + delta if k == "arena_id" and type(v) is int else renumber_arena_ids(v, delta)) for k, v in value.items()}
    if isinstance(value, list):
        return [renumber_arena_ids(v, delta) for v in value]
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
        obs = observation()
        assert_observation_classified(obs)
        actions = every_action_variant_fixture(stable_ref(99, 33, "p0", "Hand"))
        for action in actions:
            assert_action_classified(action)
        encoded = encode_decision(obs, actions)
        self.assertEqual(encoded.action_features.shape[0], len(actions))

    def test_unknown_leaf_and_ambiguous_action_fail_closed(self) -> None:
        obs = observation()
        obs["projection"]["unknown_leaf"] = 1
        with self.assertRaises(FeatureSchemaError):
            encode_decision(obs, legal_actions())
        ambiguous = {"schema_version": 2, "selected_index": 0, "stable_id": "x", "display_text": None, "semantic": {"action_kind": "ambiguous", "reason": "text"}}
        with self.assertRaises(FeatureSchemaError):
            encode_decision(observation(), [ambiguous])

    def test_names_display_stable_ids_hashes_do_not_change_features_or_logits(self) -> None:
        obs = observation()
        actions = legal_actions()
        mutated_obs = deep_copy(obs)
        mutated_actions = deep_copy(actions)
        mutated_obs["visible_projection_hash"] = 999
        mutated_obs["own_hand"][0]["card_name"] = "Different"
        mutated_obs["projection"]["battlefield"][0][0]["card_name"] = "Renamed"
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

    def test_arena_id_renumbering_does_not_change_features_or_logits(self) -> None:
        obs = observation()
        actions = legal_actions()
        a = encode_decision(obs, actions)
        b = encode_decision(renumber_arena_ids(obs), renumber_arena_ids(actions))
        assert_encoded_equal(self, a, b)
        model = KernelPolicyValueNet.from_encoded(a)
        self.assertTrue(torch.equal(model(a)[0], model(b)[0]))

    def test_legal_action_permutation_only_permutes_action_rows_and_logits(self) -> None:
        obs = observation()
        actions = legal_actions()
        permuted = [copy.deepcopy(actions[i]) for i in [2, 0, 1]]
        for idx, action in enumerate(permuted):
            action["selected_index"] = idx
        a = encode_decision(obs, actions)
        b = encode_decision(obs, permuted)
        self.assertTrue(torch.equal(a.state, b.state))
        self.assertTrue(torch.equal(a.object_features, b.object_features))
        self.assertTrue(torch.equal(a.action_features[[2, 0, 1]], b.action_features))
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, _ = model(a)
        logits_b, _ = model(b)
        self.assertTrue(torch.equal(logits_a[[2, 0, 1]], logits_b))

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
        obs = observation()
        actions = legal_actions()
        permuted = deep_copy(obs)
        permuted["projection"]["battlefield"][0] = list(reversed(permuted["projection"]["battlefield"][0]))
        a = encode_decision(obs, actions)
        b = encode_decision(permuted, actions)
        model = KernelPolicyValueNet.from_encoded(a)
        logits_a, value_a = model(a)
        logits_b, value_b = model(b)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))


if __name__ == "__main__":
    unittest.main()
