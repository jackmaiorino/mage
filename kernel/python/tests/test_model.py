from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from mtg_kernel_rl.determinism import configure_torch_determinism
from mtg_kernel_rl.features import (
    ACTION_FEATURE_DIM,
    ACTION_REF_FEATURE_DIM,
    CARD_TOKEN_VOCAB_SIZE,
    EDGE_FEATURE_DIM,
    FeatureSchemaError,
    OBJECT_FEATURE_DIM,
    OBJECT_GROUPS,
    STATE_FEATURE_DIM,
    encode_decision,
)
from mtg_kernel_rl.model import KernelPolicyValueNet, ModelConfig, greedy_action

from fixtures import complete_legal_actions, complete_observation, deep_copy, legal_actions, observation


def pass_action() -> list[dict[str, object]]:
    return [{"schema_version": 3, "selected_index": 0, "stable_id": "legal-action-v3:pass", "semantic": {"action_kind": "pass", "actor": "p0"}, "display_text": "Pass"}]


def zero_object_observation() -> dict[str, object]:
    obs = observation()
    obs["own_hand"] = []
    p = obs["projection"]
    p["battlefield"] = [[], []]
    p["graveyards"] = [[], []]
    p["exile"] = []
    p["stack"] = []
    p["combat"] = {"attackers_declared": False, "blockers_declared": False, "ordered_attackers": [], "attacker_to_ordered_blockers": []}
    p["continuous_effects"] = []
    p["exile_play_permissions"] = []
    p["engine_context"].update({"stack_nonempty": False, "stack_activity_since_priority_boundary": False})
    p["surface_context"].update({"stack_grew_since_round_open": False})
    return obs


class ModelTest(unittest.TestCase):
    def test_forward_is_deterministic_and_shapes_match_legal_actions(self) -> None:
        configure_torch_determinism()
        encoded = encode_decision(observation(), legal_actions())
        model_a = KernelPolicyValueNet.from_encoded(encoded)
        model_b = KernelPolicyValueNet.from_encoded(encoded)
        logits_a, value_a = model_a(encoded)
        logits_b, value_b = model_b(encoded)
        self.assertEqual(tuple(logits_a.shape), (len(legal_actions()),))
        self.assertEqual(tuple(value_a.shape), ())
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))
        self.assertTrue(torch.isfinite(logits_a).all())
        self.assertTrue(torch.isfinite(value_a).all())

    def test_model_config_round_trips_and_rejects_bad_dicts(self) -> None:
        encoded = encode_decision(observation(), legal_actions())
        model = KernelPolicyValueNet.from_encoded(encoded)
        data = model.config.to_dict()
        reconstructed = ModelConfig.from_dict(data)
        self.assertEqual(reconstructed, model.config)
        self.assertEqual(len(reconstructed.contract_fingerprint()), 64)
        missing = dict(data)
        del missing["state_dim"]
        with self.assertRaises(ValueError):
            ModelConfig.from_dict(missing)
        extra = dict(data)
        extra["unexpected"] = 1
        with self.assertRaises(ValueError):
            ModelConfig.from_dict(extra)
        wrong_type = dict(data)
        wrong_type["card_vocab_size"] = True
        with self.assertRaises(TypeError):
            ModelConfig.from_dict(wrong_type)
        expected = {
            "card_vocab_size": CARD_TOKEN_VOCAB_SIZE,
            "card_embedding_dim": 16,
            "hidden_dim": 64,
            "state_dim": STATE_FEATURE_DIM,
            "object_feature_dim": OBJECT_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "object_group_count": len(OBJECT_GROUPS),
            "action_ref_feature_dim": ACTION_REF_FEATURE_DIM,
        }
        for key, value in expected.items():
            self.assertEqual(data[key], value)
        for key in expected:
            tampered = dict(data)
            tampered[key] = 10**9 if data[key] != 10**9 else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                ModelConfig.from_dict(tampered)
        for key in ("schema_version", "model_architecture_version", "feature_schema_version", "feature_registry_version", "feature_contract_digest", "feature_encoding_digest"):
            tampered = dict(data)
            tampered[key] = "tampered" if isinstance(data[key], str) else 999
            with self.subTest(key=key), self.assertRaises(ValueError):
                ModelConfig.from_dict(tampered)

    def test_one_model_forwards_constant_schema_edge_and_object_width_variants(self) -> None:
        variants = []
        zero = encode_decision(zero_object_observation(), pass_action())
        variants.append(zero)
        no_edges_obs = observation()
        no_edges_obs["projection"]["stack"] = []
        no_edges_obs["projection"]["combat"] = {"attackers_declared": False, "blockers_declared": False, "ordered_attackers": [], "attacker_to_ordered_blockers": []}
        no_edges_obs["projection"]["continuous_effects"] = []
        no_edges_obs["projection"]["exile_play_permissions"] = []
        no_edges_obs["projection"]["engine_context"].update({"stack_nonempty": False, "stack_activity_since_priority_boundary": False})
        no_edges_obs["projection"]["surface_context"].update({"stack_grew_since_round_open": False})
        variants.append(encode_decision(no_edges_obs, legal_actions()))
        base_edges = complete_observation()
        base_edges["projection"]["continuous_effects"] = []
        base_edges["projection"]["exile_play_permissions"] = []
        variants.append(encode_decision(base_edges, complete_legal_actions()))
        effects = complete_observation()
        effects["projection"]["exile_play_permissions"] = []
        variants.append(encode_decision(effects, complete_legal_actions()))
        permissions = complete_observation()
        permissions["projection"]["continuous_effects"] = []
        variants.append(encode_decision(permissions, complete_legal_actions()))

        schema_dims = {(v.schema.state_dim, v.schema.object_feature_dim, v.schema.edge_feature_dim, v.schema.action_feature_dim, v.schema.action_ref_feature_dim) for v in variants}
        self.assertEqual(schema_dims, {(STATE_FEATURE_DIM, OBJECT_FEATURE_DIM, EDGE_FEATURE_DIM, ACTION_FEATURE_DIM, ACTION_REF_FEATURE_DIM)})
        self.assertEqual(tuple(zero.object_features.shape), (1, OBJECT_FEATURE_DIM))
        self.assertEqual(tuple(zero.edge_features.shape), (0, EDGE_FEATURE_DIM))
        model = KernelPolicyValueNet.from_encoded(zero)
        for encoded in variants:
            with self.subTest(edges=int(encoded.edge_features.shape[0]), objects=int(encoded.object_features.shape[0])):
                logits, value = model(encoded)
                self.assertEqual(tuple(logits.shape), (encoded.action_features.shape[0],))
                self.assertEqual(tuple(value.shape), ())

    def test_model_constructed_before_high_card_id_forwards_full_u16_domain(self) -> None:
        initial = encode_decision(zero_object_observation(), pass_action())
        model = KernelPolicyValueNet.from_encoded(initial)
        later_obs = observation()
        later_actions = legal_actions()
        later_obs["own_hand"][0]["stable"]["card_db_id"] = 65_535
        later_actions[1]["semantic"]["source"]["card_db_id"] = 65_535
        later_actions[2]["semantic"]["source"]["card_db_id"] = 65_535
        later = encode_decision(later_obs, later_actions)
        self.assertEqual(int(later.object_card_ids.max().item()), 65_536)
        logits, value = model(later)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())
        bad_obs = deep_copy(later_obs)
        bad_obs["own_hand"][0]["stable"]["card_db_id"] = 65_536
        bad_actions = deep_copy(later_actions)
        bad_actions[1]["semantic"]["source"]["card_db_id"] = 65_536
        bad_actions[2]["semantic"]["source"]["card_db_id"] = 65_536
        with self.assertRaises(FeatureSchemaError):
            encode_decision(bad_obs, bad_actions)

    def test_forward_rejects_out_of_range_tokens_instead_of_clamping(self) -> None:
        encoded = encode_decision(observation(), legal_actions())
        model = KernelPolicyValueNet.from_encoded(encoded)
        bad_object_ids = encoded.object_card_ids.clone()
        bad_object_ids[0] = model.config.card_vocab_size
        with self.assertRaises(ValueError):
            model(replace(encoded, object_card_ids=bad_object_ids))
        bad_ref_ids = encoded.action_ref_card_ids.clone()
        bad_ref_ids[0] = model.config.card_vocab_size
        with self.assertRaises(ValueError):
            model(replace(encoded, action_ref_card_ids=bad_ref_ids))

    def test_forward_rejects_bad_encoded_shape_dtype_and_nonfinite(self) -> None:
        encoded = encode_decision(observation(), legal_actions())
        model = KernelPolicyValueNet.from_encoded(encoded)
        with self.assertRaises(ValueError):
            model(replace(encoded, state=encoded.state.double()))
        with self.assertRaises(ValueError):
            model(replace(encoded, action_features=encoded.action_features[:, :-1]))
        bad = encoded.state.clone()
        bad[0] = float("nan")
        with self.assertRaises(ValueError):
            model(replace(encoded, state=bad))
        bad_group = encoded.object_groups.clone()
        bad_group[0] = model.config.object_group_count
        with self.assertRaises(ValueError):
            model(replace(encoded, object_groups=bad_group))

    def test_greedy_lowest_index_tie_break(self) -> None:
        logits = torch.tensor([1.0, 2.0, 2.0])
        self.assertEqual(greedy_action(logits), 1)
        with self.assertRaises(ValueError):
            greedy_action(torch.tensor([float("nan")], dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
