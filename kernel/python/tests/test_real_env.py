from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.client import KernelRlClient
from mtg_kernel_rl.determinism import derive_env_seed
from mtg_kernel_rl.features import assert_action_classified, assert_observation_classified, encode_decision
from mtg_kernel_rl.model import KernelPolicyValueNet, greedy_action


class RealEnvTest(unittest.TestCase):
    def test_release_kernel_env_smoke_and_model_greedy_path(self) -> None:
        env_bin = os.environ.get("MTG_KERNEL_RL_ENV_BIN")
        if not env_bin:
            self.skipTest("MTG_KERNEL_RL_ENV_BIN not set")
        self.assertTrue(Path(env_bin).is_file())
        with KernelRlClient(env_bin, timeout_s=5.0) as client:
            decision = client.reset(episode_id=0, env_seed=derive_env_seed(71501, 0), max_decisions=64)
            self.assertEqual(decision.provenance["protocol"], "kernel_rl_jsonl")
            self.assertEqual(decision.provenance["protocol_version"], 2)
            self.assertEqual(decision.provenance["schema_version"], 2)
            self.assertEqual(decision.observation["schema_version"], 2)
            assert_observation_classified(decision.observation)
            for action in decision.legal_actions:
                assert_action_classified(action)
            encoded = encode_decision(decision.observation, decision.legal_actions)
            model = KernelPolicyValueNet.from_encoded(encoded)
            logits, value = model(encoded)
            self.assertEqual(tuple(logits.shape), (len(decision.legal_actions),))
            self.assertEqual(tuple(value.shape), ())
            selected = greedy_action(logits)
            self.assertGreaterEqual(selected, 0)
            self.assertLess(selected, len(decision.legal_actions))
            action = decision.legal_actions[selected]
            next_response = client.step(action["selected_index"], action["stable_id"])
            self.assertIn(type(next_response).__name__, {"Decision", "Terminal"})
            reset2 = client.reset(episode_id=1, env_seed=derive_env_seed(71501, 1), max_decisions=64)
            self.assertEqual(reset2.episode_id, 1)
            self.assertEqual(reset2.step, 0)


if __name__ == "__main__":
    unittest.main()
