from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from mtg_kernel_rl.determinism import configure_torch_determinism
from mtg_kernel_rl.features import encode_decision
from mtg_kernel_rl.model import KernelPolicyValueNet, ModelConfig, greedy_action

from fixtures import legal_actions, observation


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
