from __future__ import annotations

import unittest

import torch

from mtg_kernel_rl.determinism import configure_torch_determinism
from mtg_kernel_rl.features import encode_decision
from mtg_kernel_rl.model import KernelPolicyValueNet, greedy_action

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

    def test_greedy_lowest_index_tie_break(self) -> None:
        logits = torch.tensor([1.0, 2.0, 2.0])
        self.assertEqual(greedy_action(logits), 1)


if __name__ == "__main__":
    unittest.main()
